# src/qectostim/gadgets/teleportation.py
"""
Teleportation-based logical gate gadgets.

Teleportation-based gates implement logical Clifford gates using:
1. Preparation of ancilla qubits in specific states
2. Transversal Bell measurements between data and ancilla
3. Classical correction based on measurement outcomes

This approach enables:
- Logical Cliffords on any CSS code
- Gate teleportation for non-transversal gates
- Magic state injection for T gates (requires magic state distillation)

The key insight is that the logical Clifford is determined by the
ancilla preparation, not the code structure.

Two teleportation approaches are implemented:

1. PRODUCT-STATE TELEPORTATION (2 blocks):
   - Ancilla prepared in product state (|+⟩, |0⟩, etc.)
   - Works for gates where the ancilla state is a +1 eigenstate of the output basis
   - Examples: Hadamard, Identity
   
2. BELL-STATE TELEPORTATION (3 blocks):
   - Two ancilla blocks prepared in Bell pair (|00⟩ + |11⟩)/√2
   - Gate applied to one ancilla before Bell measurement
   - Works for ALL Clifford gates, including S and S†
   - Pauli frame correction is deterministic (tracked in observable)

NOTE ON T GATE (Non-Clifford):
   The T gate (π/8 phase gate) is a non-Clifford gate and CANNOT be directly
   simulated in Stim. Stim only supports Clifford operations. To implement
   fault-tolerant T gates, one must use:
   
   - Magic state distillation: Prepare noisy |T⟩ states and distill them
     into high-fidelity magic states through multi-round protocols
   - State injection: Inject the magic state via teleportation
   
   This is outside the scope of Stim's stabilizer simulation capabilities.
   For T gate simulation, consider using other tools like Qiskit or Cirq.

Example usage:
    >>> from qectostim.gadgets.teleportation import TeleportedHadamard, BellTeleportedS
    >>> from qectostim.codes.surface import SurfaceCode
    >>> from qectostim.noise.models import UniformDepolarizing
    >>>
    >>> code = SurfaceCode(distance=3)
    >>> noise = UniformDepolarizing(p=0.001)
    >>> 
    >>> # Teleported Hadamard (product-state approach)
    >>> h_gadget = TeleportedHadamard()
    >>> circuit = h_gadget.to_stim([code], noise)
    >>>
    >>> # Teleported S gate (Bell-state approach)
    >>> s_gadget = BellTeleportedS()
    >>> circuit = s_gadget.to_stim([code], noise)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Tuple,
    Set,
    Union,
)
from enum import Enum

import numpy as np
import stim

from qectostim.codes.abstract_code import Code
from qectostim.noise.models import NoiseModel
from qectostim.gadgets.base import (
    Gadget,
    GadgetMetadata,
    StabilizerTransform,
    ObservableTransform,
    TeleportationGadgetMixin,
    PhaseResult,
    PhaseType,
)
from qectostim.gadgets.coordinates import (
    CoordND,
    get_code_dimension,
    get_bounding_box,
    translate_coords,
    pad_coord_to_dim,
    emit_qubit_coords_nd,
    get_code_coords,
    compute_non_overlapping_offset,
)
from qectostim.gadgets.layout import (
    GadgetLayout,
    BlockInfo,
    BridgeAncilla,
    QubitIndexMap,
    QubitAllocation,
)
from qectostim.gadgets.preparation import (
    InjectionGate,
    find_injection_qubit,
    get_injection_gates,
    CSSStatePreparation,
    ProjectionResult,
    LogicalBasis,
)

# TYPE_CHECKING imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds import DetectorContext


class AncillaState(Enum):
    """States for teleportation ancilla preparation."""
    PLUS = "|+⟩"           # |+⟩ = (|0⟩ + |1⟩)/√2
    ZERO = "|0⟩"           # |0⟩
    Y_PLUS = "|+i⟩"        # |+i⟩ = (|0⟩ + i|1⟩)/√2  
    Y_MINUS = "|-i⟩"       # |-i⟩ = (|0⟩ - i|1⟩)/√2 = S†|+⟩
    BELL = "|Φ+⟩"          # Bell state (|00⟩ + |11⟩)/√2
    MAGIC_T = "|T⟩"        # Magic state for T gate


@dataclass
class TeleportationProtocol:
    """
    Specification of a teleportation protocol for a logical gate.
    
    Attributes:
        gate_name: Name of logical gate being implemented
        ancilla_state: State to prepare ancilla in
        measurement_basis: Basis for measurement ("X", "Z", or "XZ" for Bell)
        correction_map: Dict mapping measurement outcomes to Pauli corrections
        requires_magic_state: Whether this needs a prepared magic state
    """
    gate_name: str
    ancilla_state: AncillaState
    measurement_basis: str
    correction_map: Dict[Tuple[int, ...], str] = field(default_factory=dict)
    requires_magic_state: bool = False


# Predefined protocols for standard Clifford gates
#
# GATE TELEPORTATION THEORY:
# ==========================
# To teleport a state |ψ⟩ through gate G, we use:
#   1. Prepare ancilla in |0⟩
#   2. Apply preparation gates to create entanglement resource
#   3. CNOT from data to ancilla
#   4. H on data, then measure
#   5. Output on ancilla is G|ψ⟩ (with Pauli corrections)
#
# For IDENTITY: Ancilla starts in |0⟩. After CNOT+H+M, output is |ψ⟩.
# For HADAMARD: Ancilla starts in |+⟩ = H|0⟩. Works because H maps computational to Hadamard basis.
# For S GATE: We use a DIFFERENT approach - Bell-state based teleportation (see BellStateTeleportedS)
#
# The simple product-state approach works for H because:
#   H|0⟩ = |+⟩, H|1⟩ = |-⟩, and X|+⟩ = |-⟩
#   So CNOT correctly maps |0⟩→|+⟩, |1⟩→|-⟩ = H-transformed basis
#
# The simple approach FAILS for S because:
#   S|0⟩ = |0⟩, S|1⟩ = i|1⟩ (NOT equal to X|+i⟩ = |-i⟩)
#
TELEPORTATION_PROTOCOLS = {
    "H": TeleportationProtocol(
        gate_name="H",
        ancilla_state=AncillaState.PLUS,
        measurement_basis="XZ",
        correction_map={
            (0, 0): "I",
            (1, 0): "Z",
            (0, 1): "X",
            (1, 1): "Y",
        },
    ),
    "S": TeleportationProtocol(
        gate_name="S",
        ancilla_state=AncillaState.Y_PLUS,
        measurement_basis="XZ",
        correction_map={
            (0, 0): "I",
            (1, 0): "Z",
            (0, 1): "X",
            (1, 1): "Y",
        },
    ),
    "S_DAG": TeleportationProtocol(
        gate_name="S_DAG",
        ancilla_state=AncillaState.Y_PLUS,
        measurement_basis="XZ",
        correction_map={
            (0, 0): "I",
            (1, 0): "Z",
            (0, 1): "X",
            (1, 1): "Y",
        },
    ),
    "T": TeleportationProtocol(
        gate_name="T",
        ancilla_state=AncillaState.MAGIC_T,
        measurement_basis="XZ",
        correction_map={
            (0, 0): "I",
            (1, 0): "S",  # S correction for T gate
            (0, 1): "X",
            (1, 1): "XS",
        },
        requires_magic_state=True,
    ),
    "IDENTITY": TeleportationProtocol(
        gate_name="I",
        ancilla_state=AncillaState.ZERO,  # Identity: ancilla stays in |0⟩
        measurement_basis="XZ",
        correction_map={
            (0, 0): "I",
            (1, 0): "Z",
            (0, 1): "X",
            (1, 1): "Y",
        },
    ),
}


class TeleportedGate(Gadget, TeleportationGadgetMixin):
    """
    Base class for teleportation-based logical gates.
    
    Implements the teleportation protocol:
    1. Prepare ancilla block in appropriate state
    2. Create Bell pairs between data and ancilla
    3. Measure in X and Z bases
    4. Apply Pauli frame correction based on outcomes
    
    The logical gate performed depends on the ancilla preparation state.
    
    Attributes:
        protocol: TeleportationProtocol specifying the gate
        include_stabilizer_rounds: Whether to include syndrome measurements
        num_rounds_before: Stabilizer rounds before teleportation
        num_rounds_after: Stabilizer rounds after teleportation
        projection_rounds: Number of projection rounds for FT preparation (0 = skip)
    """
    
    def __init__(
        self,
        protocol: TeleportationProtocol,
        include_stabilizer_rounds: bool = True,
        num_rounds_before: int = 1,
        num_rounds_after: int = 1,
        skip_projection: bool = True,  # Legacy parameter, use projection_rounds instead
        projection_rounds: int = 0,  # FT preparation: d rounds for distance-d code
        use_hybrid_decoding: bool = False,  # Use DEM for EC + classical frame tracking
        input_state: str = "0",
    ):
        """
        Initialize teleported gate gadget.
        
        Args:
            protocol: TeleportationProtocol for this gate
            include_stabilizer_rounds: Include stabilizer measurements
            num_rounds_before: Stabilizer rounds before teleportation
            num_rounds_after: Stabilizer rounds after teleportation
            skip_projection: Legacy parameter. If False and projection_rounds=0,
                will do 1 round of projection (non-FT).
            projection_rounds: Number of repeated stabilizer measurements for
                fault-tolerant preparation. For distance-d code, use d rounds
                to correct up to floor((d-1)/2) measurement errors. 
                Set to 0 to skip projection (state already in codespace).
            use_hybrid_decoding: If True, uses hybrid DEM + classical Pauli frame
                tracking. The DEM handles error correction via detectors, while
                the logical observable is computed classically by combining:
                1. Decoded data block Z_L (OBSERVABLE_0)
                2. Decoded ancilla block X_L (OBSERVABLE_1) 
                3. Projection frame (XOR of relevant projection measurements)
            input_state: Logical input state ("0" or "+")
        """
        super().__init__(input_state=input_state)
        
        self.protocol = protocol
        self.include_stabilizer_rounds = include_stabilizer_rounds
        self.num_rounds_before = num_rounds_before
        self.num_rounds_after = num_rounds_after
        self._use_hybrid_decoding = use_hybrid_decoding
        
        # Handle legacy skip_projection parameter
        if projection_rounds > 0:
            self.projection_rounds = projection_rounds
            self.skip_projection = False
        elif skip_projection:
            self.projection_rounds = 0
            self.skip_projection = True
        else:
            # Legacy: skip_projection=False means do 1 round
            self.projection_rounds = 1
            self.skip_projection = False
        
        # State tracking
        self._ancilla_block_name = "ancilla_block"
        self._data_block_name = "data_block"
        
        # Injection qubit tracking (set during emit_next_phase)
        self._injection_qubit_local = None
        self._injection_qubit_global = None
        
        # Projection measurement tracking (for Pauli frame in observable)
        self._projection_x_meas = []
        self._projection_z_meas = []
        
        # Measurement index tracking for hybrid decoding
        self._prep_meas_start: Optional[int] = None  # Global index of first prep meas
        self._teleport_meas_start: Optional[int] = None  # Global index of data Z measurements
    
    def validate_codes(self, codes: List[Code]) -> None:
        """Validate codes for teleportation."""
        super().validate_codes(codes)
        
        if len(codes) != 1:
            raise ValueError(
                f"Teleported gates operate on single code blocks, got {len(codes)}"
            )
        
        # Check if code is CSS (required for standard teleportation)
        code = codes[0]
        if not (hasattr(code, 'is_css') and code.is_css):
            # Warning but don't fail
            pass
    
    # =========================================================================
    # PRIMARY INTERFACE - Required by FaultTolerantGadgetExperiment
    # =========================================================================
    
    @property
    def num_phases(self) -> int:
        """
        Teleportation has 3 phases:
        Phase 0: CSS state preparation (|+⟩_L via R-H-measure stabilizers)
        Phase 1: Transversal CNOT (data → ancilla)
        Phase 2: Measure data (Z-basis measurement)
        """
        return 3
    
    def emit_next_phase(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """
        Emit the next phase of the teleportation protocol.
        
        CSS STATE PREPARATION PROTOCOL:
        ===============================
        For CSS codes, we use the simple projection protocol:
        
        Phase 0: CSS STATE PREPARATION
            - Initialize all ancilla qubits to |+⟩ (R then H)
            - Measure all stabilizers (outcomes define Pauli frame)
            - This projects into |+⟩_L up to the tracked Pauli frame
            - Do NOT force outcomes to +1; record them for classical tracking
        
        Phase 1: TRANSVERSAL CNOT
            - Apply transversal CNOT from data block to ancilla block
            - This entangles the input |ψ⟩_L with the resource state
        
        Phase 2: MEASUREMENT
            - Measure data block in Z basis (logical measurement)
            - Combined with Pauli frame from Phase 0, determines corrections
            - Output on ancilla block is U|ψ⟩_L (up to tracked Pauli frame)
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit gates into.
        alloc : QubitAllocation
            Qubit allocation with block-to-global mappings.
        ctx : DetectorContext
            Detector context (for measurement tracking).
            
        Returns
        -------
        PhaseResult
            Result describing what was emitted.
        """
        data_block = alloc.get_block("block_0")
        ancilla_block = alloc.get_block("block_1")
        
        if data_block is None:
            data_block = alloc.get_block(self._data_block_name)
            ancilla_block = alloc.get_block(self._ancilla_block_name)
        
        if data_block is None or ancilla_block is None:
            raise ValueError("Teleportation requires data and ancilla blocks")
        
        n = min(data_block.data_count, ancilla_block.data_count)
        data_qubits = list(data_block.data_range)[:n]
        ancilla_data_qubits = list(ancilla_block.data_range)[:n]
        ancilla_x_anc = list(ancilla_block.x_anc_range)
        ancilla_z_anc = list(ancilla_block.z_anc_range)
        
        # Get the code from the block for logical operator info
        code = ancilla_block.code
        
        phase = self._current_phase
        self._current_phase += 1
        
        if phase == 0:
            # =================================================================
            # PHASE 0: CSS STATE PREPARATION BY PROJECTION
            # 
            # For teleportation, we need to prepare the ancilla in |+⟩_L.
            # Using the simple CSS protocol:
            #   |+⟩_L: Initialize |+⟩^⊗n (R all, H all), then measure stabilizers
            #
            # The stabilizer measurements define the Pauli frame - we don't
            # force them to +1, we just record the outcomes.
            #
            # This replaces the complex encoding circuit approach with the
            # fundamental CSS preparation protocol.
            # =================================================================
            
            # For hybrid decoding, SKIP projection rounds
            # The Z stabilizer measurement in projection uses CX(data→anc) which
            # spreads X from data to ancilla. Since OBSERVABLE_1 is X_L on the
            # ancilla data qubits, backward tracking from the final measurement
            # through projection would hit R(anc) and anti-commute.
            #
            # Instead, do simplified prep with just X stabilizer measurement:
            # 1. H on all qubits (they start in |0⟩ from global R)
            # 2. Measure X stabilizers to establish reference frame
            # 3. The X_L observable compares final to this reference
            if self._use_hybrid_decoding:
                # Apply H to convert |0⟩^⊗n to |+⟩^⊗n
                circuit.append("H", ancilla_data_qubits)
                circuit.append("TICK")
                
                # Measure X stabilizers to establish reference
                # This absorbs backward X sensitivity from the final measurement
                if ancilla_z_anc:
                    meas_anc = ancilla_z_anc[0]
                elif ancilla_x_anc:
                    meas_anc = ancilla_x_anc[0]
                else:
                    meas_anc = ancilla_data_qubits[0]  # Fallback
                
                x_stab_meas = []
                if code.hx is not None:
                    for stab_idx in range(code.hx.shape[0]):
                        circuit.append("R", [meas_anc])
                        circuit.append("H", [meas_anc])
                        support = list(np.where(code.hx[stab_idx])[0])
                        for q in support:
                            circuit.append("CX", [meas_anc, ancilla_data_qubits[q]])
                        circuit.append("H", [meas_anc])
                        circuit.append("M", [meas_anc])
                        x_stab_meas.append(stab_idx)  # Track which stabilizer
                    circuit.append("TICK")
                
                # Store for frame tracking
                self._projection_x_meas = list(range(len(x_stab_meas)))  # Local indices
                self._projection_z_meas = []
                self._prep_result = None
                
                return PhaseResult(
                    phase_type=PhaseType.PREPARATION,
                    is_final=False,
                    needs_stabilizer_rounds=0,
                    measurement_count=len(x_stab_meas),
                )
            
            # Use CSSStatePreparation for proper |+⟩_L preparation
            css_prep = CSSStatePreparation(code)
            
            # Get ancilla for stabilizer measurements
            if ancilla_z_anc:
                meas_anc = ancilla_z_anc[0]
            elif ancilla_x_anc:
                meas_anc = ancilla_x_anc[0]
            else:
                meas_anc = getattr(ancilla_block, 'proj_anc', None)
                if meas_anc is None:
                    raise ValueError("No ancilla qubit available for stabilizer projection")
            
            # Prepare |+⟩_L: R all, H all, measure stabilizers
            # (This combines what was Phase 0 and Phase 1)
            prep_result = css_prep.prepare_plus(
                circuit,
                ancilla_data_qubits,
                meas_anc,
                num_rounds=max(1, self.projection_rounds),
                emit_detectors=(self.projection_rounds > 1),
            )
            
            # Store projection measurements for Pauli frame tracking
            self._projection_x_meas = prep_result.final_x_meas
            self._projection_z_meas = prep_result.final_z_meas
            self._prep_result = prep_result
            
            # Track for compatibility
            self._injection_qubit_local = 0
            self._injection_qubit_global = ancilla_data_qubits[0]
            
            total_meas = len(prep_result.all_measurements)
            
            return PhaseResult(
                phase_type=PhaseType.PREPARATION,
                is_final=False,
                needs_stabilizer_rounds=0,
                measurement_count=total_meas,
                pauli_frame_update={
                    'block_name': self._ancilla_block_name,
                    'projection': True,
                    'x_stab_count': css_prep.n_x,
                    'z_stab_count': css_prep.n_z,
                    'x_stab_meas_indices': self._projection_x_meas,
                    'z_stab_meas_indices': self._projection_z_meas,
                    'num_projection_rounds': prep_result.num_rounds,
                },
            )
        
        elif phase == 1:
            # =================================================================
            # PHASE 1: TRANSVERSAL CNOT
            # CNOT from data block to ancilla block (qubit-wise)
            # =================================================================
            for d, a in zip(data_qubits, ancilla_data_qubits):
                circuit.append("CNOT", [d, a])
            circuit.append("TICK")
            
            return PhaseResult(
                phase_type=PhaseType.GATE,
                is_final=False,
                needs_stabilizer_rounds=0,  # NO rounds while entangled!
            )
        
        elif phase == 2:
            # =================================================================
            # PHASE 2: MEASUREMENT
            # Measure data block in Z basis (computational basis)
            # This completes the teleportation
            # =================================================================
            circuit.append("M", data_qubits)
            circuit.append("TICK")
            
            # Get the gate-specific stabilizer transform
            gate_transform = self.get_stabilizer_transform()
            
            n_data = len(data_qubits)
            
            # Get the Z_L support for determining which measurements contribute
            # to the X frame correction. Only measurements on Z_L support matter.
            code = alloc.get_block("block_0").code if alloc.get_block("block_0") else None
            if code is None:
                code = alloc.get_block(self._data_block_name)
                if code is not None:
                    code = code.code
            
            z_l_support_meas = []
            if code is not None:
                lz = getattr(code, 'Lz', None)
                if lz is None:
                    lz = getattr(code, 'lz', None)
                if lz is not None:
                    lz = np.atleast_2d(lz)
                    z_l_support = list(np.where(lz[0])[0])
                    z_l_support_meas = z_l_support  # Local indices within data block
            
            return PhaseResult(
                phase_type=PhaseType.MEASUREMENT,
                is_final=True,
                measurement_count=n_data,
                measurement_qubits=list(data_qubits),
                measured_blocks=["data_block"],
                stabilizer_transform=gate_transform,
                pauli_frame_update={
                    'block_name': self._ancilla_block_name,
                    # Only Z_L support measurements contribute to X frame correction
                    'x_meas': z_l_support_meas,
                    'z_meas': [],
                    'teleport': True,
                },
            )
        
        else:
            return PhaseResult.complete()
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """
        Return how this teleportation transforms stabilizer types.
        
        Teleportation transfers state to ancilla block, so this depends
        on the gate being teleported. All teleportation transforms include
        skip_first_round=True because the ancilla's stabilizers are entangled
        with the Bell measurements and cannot have independent first-round detectors.
        
        Gates that transform X to Y (like S) require clearing stabilizer
        history because CSS codes only measure X and Z, not Y.
        """
        gate = self.protocol.gate_name.upper()
        
        if gate == "H":
            # Hadamard swaps X↔Z, use teleportation helper
            return StabilizerTransform.teleportation(swap_xz=True)
        elif gate in ("S", "S_DAG"):
            # S: X→Y, Z→Z. We must clear history because X→Y can't be tracked
            # in CSS stabilizer measurements (we only measure X and Z, not Y)
            return StabilizerTransform(
                x_becomes="Y", z_becomes="Z", 
                clear_history=True, skip_first_round=True
            )
        elif gate in ("T", "T_DAG"):
            # T gate doesn't change stabilizers, but teleportation still needs
            # skip_first_round due to entanglement with Bell measurements
            return StabilizerTransform.teleportation(swap_xz=False)
        elif gate in ("I", "IDENTITY"):
            # Identity teleportation still needs skip_first_round
            return StabilizerTransform.teleportation(swap_xz=False)
        else:
            return StabilizerTransform.teleportation(swap_xz=False)
    
    def get_observable_transform(self) -> ObservableTransform:
        """
        Return how this teleportation transforms the logical observable.
        """
        return ObservableTransform.from_gate_name(self.protocol.gate_name)
    
    def get_input_block_names(self) -> List[str]:
        """
        Return names of blocks that contain input logical states.
        
        For teleportation, only the data block is an input. The ancilla block
        is prepared by the gadget and should not be initialized by the experiment.
        
        Returns
        -------
        List[str]
            Block names containing input logical states.
        """
        return [self._data_block_name]
    
    def get_ancilla_block_names(self) -> List[str]:
        """
        Return names of blocks used as ancilla (prepared by gadget, not experiment).
        
        For teleportation, the ancilla block is prepared in a gate-specific state
        by the gadget itself. The experiment should not prepare or run pre-gadget
        rounds on these blocks.
        
        Returns
        -------
        List[str]
            Block names that are ancilla/output blocks.
        """
        return [self._ancilla_block_name]
    
    def get_output_block_name(self) -> str:
        """
        Return the name of the block containing the output logical state.
        
        For teleportation, the logical state is teleported from the data block
        to the ancilla block. After teleportation, the logical observable
        should be measured on the ancilla block, not the original data block.
        
        Returns
        -------
        str
            The block name where the output logical state resides.
        """
        return self._ancilla_block_name
    
    def is_teleportation_gadget(self) -> bool:
        """Return True to indicate this is a teleportation-based gadget."""
        return True
    
    def requires_raw_sampling(self) -> bool:
        """
        Return True to indicate this gadget requires raw measurement sampling.
        
        Teleportation-based gadgets have RANDOM measurement outcomes that encode
        both the teleportation byproduct and syndrome information. These cannot
        be expressed as deterministic observables in Stim's DEM framework.
        
        If use_hybrid_decoding=True, we return False because the circuit will
        have deterministic observables (just final data/ancilla measurements)
        and the Pauli frame corrections are applied classically after decoding.
        
        If use_hybrid_decoding=False, returns True and the experiment should use:
        1. Raw measurement sampling (compile_sampler)
        2. Python-based Pauli frame tracking
        3. Classical post-processing to determine logical outcome
        """
        return not self._use_hybrid_decoding
    
    def get_projection_measurements_for_observable(self, basis: str, code=None) -> List[int]:
        """
        Get projection measurements that must be included in OBSERVABLE_INCLUDE.
        
        For gate teleportation via injection + projection:
        - The state after injection is U|0⟩ on one qubit, |0⟩^⊗(n-1) on rest
        - Projection measurements are random but define the Pauli frame
        - The observable must XOR in the relevant projection measurements
        
        For X observable (H gate → |+⟩_L preparation):
        - X stabilizers with ODD overlap with X_L contribute to the frame
        
        For Z observable (Identity or other):
        - Z stabilizers with ODD overlap with Z_L contribute to the frame
        
        Parameters
        ----------
        basis : str
            "X" or "Z" - the measurement basis
        code : optional
            The code (for computing stabilizer overlaps)
            
        Returns
        -------
        List[int]
            Measurement indices to include in OBSERVABLE_INCLUDE
        """
        if code is None:
            return []
        
        if basis == "X":
            # For X_L, X stabilizers with odd overlap contribute
            lx = getattr(code, 'Lx', None)
            if lx is None:
                lx = getattr(code, 'lx', None)
            if lx is None:
                return []
            lx = np.atleast_2d(lx)
            hx = code.hx
            if hx is None:
                return []
            
            x_l_support = set(np.where(lx[0])[0])
            frame_meas = []
            
            for stab_idx, meas_idx in enumerate(self._projection_x_meas):
                if stab_idx < hx.shape[0]:
                    stab_support = set(np.where(hx[stab_idx])[0])
                    overlap = len(stab_support & x_l_support)
                    if overlap % 2 == 1:
                        frame_meas.append(meas_idx)
            
            return frame_meas
        
        elif basis == "Z":
            # For Z_L, Z stabilizers with odd overlap contribute
            lz = getattr(code, 'Lz', None)
            if lz is None:
                lz = getattr(code, 'lz', None)
            if lz is None:
                return []
            lz = np.atleast_2d(lz)
            hz = code.hz
            if hz is None:
                return []
            
            z_l_support = set(np.where(lz[0])[0])
            frame_meas = []
            
            for stab_idx, meas_idx in enumerate(self._projection_z_meas):
                if stab_idx < hz.shape[0]:
                    stab_support = set(np.where(hz[stab_idx])[0])
                    overlap = len(stab_support & z_l_support)
                    if overlap % 2 == 1:
                        frame_meas.append(meas_idx)
            
            return frame_meas
        
        return []
    
    def get_frame_tracking_info(self, code=None) -> Dict[str, Any]:
        """
        Get information needed for hybrid Pauli frame tracking.
        
        For hybrid decoding, we need to know:
        1. Which projection X measurements have odd overlap with X_L (for X frame)
        2. The teleportation measurement indices (for Z frame)
        
        After decoding:
        - projection_frame = XOR of projection measurements with odd X_L overlap
        - teleport_frame = decoded observable 0 (data Z_L)
        - final_result = ancilla_X_L ^ projection_frame ^ teleport_frame
        
        Returns
        -------
        Dict with:
            'projection_frame_meas': List of measurement indices for projection frame
            'teleport_meas_start': Starting measurement index for teleportation
            'teleport_meas_count': Number of data qubits measured
            'ancilla_code': The code for the ancilla block
        """
        info = {
            'projection_frame_meas': [],
            'teleport_meas_start': self._teleport_meas_start,
            'teleport_meas_count': 0,
            'prep_meas_start': self._prep_meas_start,
            'prep_result': getattr(self, '_prep_result', None),
            'projection_x_meas': self._projection_x_meas,
            'projection_z_meas': self._projection_z_meas,
        }
        
        if code is not None:
            # Get X stabilizers with odd overlap with X_L
            info['projection_frame_meas'] = self.get_projection_measurements_for_observable("X", code)
            
        return info

    def compute_layout(self, codes: List[Code]) -> GadgetLayout:
        """
        Compute layout for teleportation.
        
        Places data block and ancilla block with spacing.
        """
        self._dimension = self.infer_dimension(codes)
        code = codes[0]
        
        layout = GadgetLayout(target_dim=self._dimension)
        
        # Add data block at origin
        layout.add_block(
            name=self._data_block_name,
            code=code,
            offset=(0.0,) * self._dimension,
        )
        
        # Add ancilla block (copy of code structure) offset in first dimension
        # Compute offset based on bounding box
        if hasattr(code, 'qubit_coords'):
            coords = code.qubit_coords()
            if coords:
                # qubit_coords() returns List[Tuple], not Dict
                _, max_corner = get_bounding_box(coords)
                offset_dist = max_corner[0] + 3.0  # Spacing
            else:
                offset_dist = code.n + 3.0
        else:
            offset_dist = code.n + 3.0
        
        ancilla_offset = (offset_dist,) + (0.0,) * (self._dimension - 1)
        layout.add_block(
            name=self._ancilla_block_name,
            code=code,
            offset=ancilla_offset,
        )
        
        self._cached_layout = layout
        return layout
    
    
    def _add_detectors(
        self,
        circuit: stim.Circuit,
        layout: GadgetLayout,
        codes: List[Code],
    ) -> None:
        """Add detector instructions."""
        # Placeholder - full implementation tracks stabilizers
        pass
    
    def _add_logical_observable(
        self,
        circuit: stim.Circuit,
        layout: GadgetLayout,
        codes: List[Code],
    ) -> None:
        """Add logical observable tracking."""
        # Placeholder - tracks logical operator through teleportation
        pass
    
    def _build_metadata(
        self,
        layout: GadgetLayout,
        codes: List[Code],
    ) -> GadgetMetadata:
        """Build metadata for decoder."""
        qubit_map = layout.get_qubit_index_map()
        
        return GadgetMetadata(
            gadget_type="teleportation",
            logical_operation=self.protocol.gate_name,
            input_codes=[type(c).__name__ for c in codes],
            output_codes=[type(c).__name__ for c in codes],
            qubit_index_map=qubit_map,
            detector_coords=self.get_detector_coords(layout),
            timing_info={
                "rounds_before": self.num_rounds_before,
                "rounds_after": self.num_rounds_after,
                "total_layers": len(self._cached_scheduler.layers) if self._cached_scheduler else 0,
            },
            extra={
                "protocol": self.protocol.gate_name,
                "ancilla_state": self.protocol.ancilla_state.value,
                "requires_magic_state": self.protocol.requires_magic_state,
            },
        )
    
    def get_metadata(self) -> GadgetMetadata:
        """Get cached metadata."""
        if self._cached_metadata is None:
            raise RuntimeError("Metadata not available. Call to_stim() first.")
        return self._cached_metadata


# =============================================================================
# Bell-State Based Gate Teleportation
# =============================================================================
#
# For gates that cannot be teleported using simple product-state ancilla
# (like S gate), we use Bell-state based teleportation:
#
# Circuit:
#   |ψ⟩ ────●────H────M────  (data, measured)
#           │
#   |0⟩ ─H──X──●─────────M────  (ancilla1, measured)  
#              │
#   |0⟩ ───────X──G────────────  (ancilla2, output = G|ψ⟩)
#
# Where G is the gate to teleport (applied to ancilla2 after Bell creation).
#
# This creates the resource state (|00⟩ + |11⟩)/√2 on (a1, a2), then applies G
# to a2, giving (|0⟩G|0⟩ + |1⟩G|1⟩)/√2. Bell measurement between data and a1
# teleports G|ψ⟩ to a2.
#

class BellStateTeleportedGate(Gadget, TeleportationGadgetMixin):
    """
    Bell-state based gate teleportation for Clifford gates.
    
    This implements the correct teleportation protocol for gates like S
    that cannot be teleported using simple product-state ancilla preparation.
    
    The protocol uses THREE logical blocks:
    - data: The input state |ψ⟩ (destroyed by Bell measurement)
    - ancilla1: First half of Bell pair (destroyed by Bell measurement)  
    - ancilla2: Second half of Bell pair with gate applied (OUTPUT = G|ψ⟩)
    
    Circuit structure:
    1. Create Bell pair on (ancilla1, ancilla2)
    2. Apply gate G to ancilla2
    3. Bell measurement between data and ancilla1
    4. Output on ancilla2 is G|ψ⟩ (with Pauli corrections)
    """
    
    def __init__(
        self,
        gate_name: str,
        include_stabilizer_rounds: bool = True,
        num_rounds_before: int = 1,
        num_rounds_after: int = 1,
        input_state: str = "0",
    ):
        super().__init__(input_state=input_state)
        self.gate_name = gate_name.upper()
        self.include_stabilizer_rounds = include_stabilizer_rounds
        self.num_rounds_before = num_rounds_before
        self.num_rounds_after = num_rounds_after
        
        self._data_block_name = "data_block"
        self._ancilla1_block_name = "ancilla1_block"
        self._ancilla2_block_name = "ancilla2_block"
    
    def validate_codes(self, codes: List[Code]) -> None:
        """Validate codes for Bell-state teleportation."""
        super().validate_codes(codes)
        if len(codes) != 1:
            raise ValueError(
                f"BellStateTeleportedGate operates on single code blocks, got {len(codes)}"
            )
    
    @property
    def num_phases(self) -> int:
        """
        Bell-state teleportation has 4 phases:
        Phase 0: Create Bell pair on ancilla blocks
        Phase 1: Apply gate G to ancilla2
        Phase 2: Bell measurement setup (CNOT data→ancilla1, H on data)
        Phase 3: Measurements
        """
        return 4
    
    def emit_next_phase(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """Emit the next phase of Bell-state teleportation."""
        # Get all three blocks
        data_block = alloc.get_block("block_0")
        ancilla1_block = alloc.get_block("block_1")
        ancilla2_block = alloc.get_block("block_2")
        
        if data_block is None:
            data_block = alloc.get_block(self._data_block_name)
            ancilla1_block = alloc.get_block(self._ancilla1_block_name)
            ancilla2_block = alloc.get_block(self._ancilla2_block_name)
        
        if data_block is None or ancilla1_block is None or ancilla2_block is None:
            raise ValueError("BellStateTeleportedGate requires 3 blocks: data, ancilla1, ancilla2")
        
        n = min(data_block.data_count, ancilla1_block.data_count, ancilla2_block.data_count)
        data_qubits = list(data_block.data_range)[:n]
        ancilla1_qubits = list(ancilla1_block.data_range)[:n]
        ancilla2_qubits = list(ancilla2_block.data_range)[:n]
        
        phase = self._current_phase
        self._current_phase += 1
        
        if phase == 0:
            # =================================================================
            # PHASE 0: CSS STATE PREPARATION FOR BELL PAIR
            #
            # For logical Bell pair creation with CSS codes:
            # 1. Prepare ancilla1 in |+⟩_L (R all, H all, measure stabilizers)
            # 2. Prepare ancilla2 in |0⟩_L (R all, measure stabilizers)
            # 3. Transversal CNOT from ancilla1 to ancilla2 creates logical Bell state
            #
            # This replaces the naive physical-qubit approach with the proper
            # CSS projection protocol that correctly tracks the Pauli frame.
            # =================================================================
            
            code = data_block.code  # All blocks use same code
            css_prep = CSSStatePreparation(code)
            
            # Get ancillas for stabilizer measurements
            ancilla1_x_anc = list(ancilla1_block.x_anc_range)
            ancilla1_z_anc = list(ancilla1_block.z_anc_range)
            ancilla2_x_anc = list(ancilla2_block.x_anc_range)
            ancilla2_z_anc = list(ancilla2_block.z_anc_range)
            
            # Pick measurement ancillas
            if ancilla1_z_anc:
                meas_anc1 = ancilla1_z_anc[0]
            elif ancilla1_x_anc:
                meas_anc1 = ancilla1_x_anc[0]
            else:
                meas_anc1 = getattr(ancilla1_block, 'proj_anc', ancilla1_qubits[-1] + 1)
            
            if ancilla2_z_anc:
                meas_anc2 = ancilla2_z_anc[0]
            elif ancilla2_x_anc:
                meas_anc2 = ancilla2_x_anc[0]
            else:
                meas_anc2 = getattr(ancilla2_block, 'proj_anc', ancilla2_qubits[-1] + 1)
            
            # Prepare ancilla1 in |+⟩_L
            prep_result1 = css_prep.prepare_plus(
                circuit,
                ancilla1_qubits,
                meas_anc1,
                num_rounds=1,  # Single round for now
                emit_detectors=False,
            )
            
            # Prepare ancilla2 in |0⟩_L
            prep_result2 = css_prep.prepare_zero(
                circuit,
                ancilla2_qubits,
                meas_anc2,
                num_rounds=1,
                emit_detectors=False,
            )
            
            # Transversal CNOT: ancilla1 → ancilla2
            # This creates the logical Bell state (|00⟩_L + |11⟩_L)/√2
            for a1, a2 in zip(ancilla1_qubits, ancilla2_qubits):
                circuit.append("CNOT", [a1, a2])
            circuit.append("TICK")
            
            # Store projection measurements for Pauli frame tracking
            self._projection1_x_meas = prep_result1.final_x_meas
            self._projection1_z_meas = prep_result1.final_z_meas
            self._projection2_x_meas = prep_result2.final_x_meas
            self._projection2_z_meas = prep_result2.final_z_meas
            
            total_meas = len(prep_result1.all_measurements) + len(prep_result2.all_measurements)
            
            return PhaseResult(
                phase_type=PhaseType.PREPARATION,
                is_final=False,
                needs_stabilizer_rounds=0,
                measurement_count=total_meas,
                pauli_frame_update={
                    'ancilla1_block': self._ancilla1_block_name,
                    'ancilla2_block': self._ancilla2_block_name,
                    'projection': True,
                    'ancilla1_x_meas': self._projection1_x_meas,
                    'ancilla1_z_meas': self._projection1_z_meas,
                    'ancilla2_x_meas': self._projection2_x_meas,
                    'ancilla2_z_meas': self._projection2_z_meas,
                },
            )
        
        elif phase == 1:
            # Phase 1: Apply gate G to ancilla2
            # This transforms the Bell state to encode the gate
            if self.gate_name == "S":
                circuit.append("S", ancilla2_qubits)
            elif self.gate_name == "S_DAG":
                circuit.append("S_DAG", ancilla2_qubits)
            elif self.gate_name == "H":
                circuit.append("H", ancilla2_qubits)
            elif self.gate_name in ("I", "IDENTITY"):
                pass  # Identity - no gate needed
            else:
                raise ValueError(f"Unsupported gate for Bell-state teleportation: {self.gate_name}")
            circuit.append("TICK")
            
            return PhaseResult(
                phase_type=PhaseType.GATE,
                is_final=False,
                needs_stabilizer_rounds=0,
            )
        
        elif phase == 2:
            # Phase 2: Bell measurement setup
            # CNOT from data to ancilla1, H on data
            for d, a1 in zip(data_qubits, ancilla1_qubits):
                circuit.append("CNOT", [d, a1])
            circuit.append("TICK")
            
            circuit.append("H", data_qubits)
            circuit.append("TICK")
            
            return PhaseResult(
                phase_type=PhaseType.GATE,
                is_final=False,
                needs_stabilizer_rounds=0,
            )
        
        elif phase == 3:
            # Phase 3: Measure data and ancilla1 (Bell measurement)
            # Output is on ancilla2
            circuit.append("M", data_qubits + ancilla1_qubits)
            circuit.append("TICK")
            
            return PhaseResult(
                phase_type=PhaseType.MEASUREMENT,
                is_final=True,
                measurement_count=len(data_qubits) + len(ancilla1_qubits),
                measurement_qubits=list(data_qubits) + list(ancilla1_qubits),
                measured_blocks=["block_0", "block_1"],  # block_0=data, block_1=ancilla1
                stabilizer_transform=self.get_stabilizer_transform(),
            )
        
        else:
            return PhaseResult.complete()
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """Get stabilizer transform for the gate."""
        if self.gate_name == "H":
            return StabilizerTransform.teleportation(swap_xz=True)
        elif self.gate_name in ("S", "S_DAG"):
            return StabilizerTransform(
                x_becomes="Y", z_becomes="Z",
                clear_history=True, skip_first_round=True
            )
        else:
            return StabilizerTransform.teleportation(swap_xz=False)
    
    def get_observable_transform(self) -> ObservableTransform:
        """Get observable transform for the gate."""
        return ObservableTransform.from_gate_name(self.gate_name)
    
    def get_output_block_name(self) -> str:
        """Output is on block_2 (ancilla2)."""
        return "block_2"
    
    def is_teleportation_gadget(self) -> bool:
        """Return True to indicate this is a teleportation-based gadget."""
        return True
    
    def requires_three_blocks(self) -> bool:
        """Indicate this gadget needs 3 code blocks."""
        return True
    
    def get_metadata(self) -> GadgetMetadata:
        """Get cached metadata."""
        if self._cached_metadata is None:
            raise RuntimeError("Metadata not available. Call to_stim() first.")
        return self._cached_metadata
    
    def compute_layout(self, codes: List[Code]) -> GadgetLayout:
        """Compute layout for 3-block Bell-state teleportation."""
        code = codes[0]
        
        # Get data coordinates for positioning
        data_coords, _, _ = get_code_coords(code)
        dim = get_code_dimension(code)  # Pass code, not coords
        
        # Use defaults if no coords available
        if not data_coords:
            data_coords = [(float(i), 0.0) for i in range(code.n)]
            dim = 2
        
        # Compute offsets for 3 blocks (data, ancilla1, ancilla2)
        offset1 = compute_non_overlapping_offset(data_coords, data_coords, margin=1.0)
        offset2 = compute_non_overlapping_offset(
            data_coords, 
            translate_coords(data_coords, offset1),
            margin=1.0
        )
        # Add the first offset to the second for correct positioning
        offset2 = tuple(o1 + o2 for o1, o2 in zip(offset1, offset2))
        
        # Create layout and add blocks properly
        layout = GadgetLayout(target_dim=dim)
        
        # Block 0: data (input state)
        layout.add_block(
            name="block_0",
            code=code,
            offset=tuple(0.0 for _ in range(dim)),
        )
        
        # Block 1: ancilla1 (first Bell pair half, will be measured)
        layout.add_block(
            name="block_1",
            code=code,
            offset=offset1,
        )
        
        # Block 2: ancilla2 (second Bell pair half with gate applied, output)
        layout.add_block(
            name="block_2",
            code=code,
            offset=offset2,
        )
        
        return layout


# Convenience classes for specific gates

class TeleportedHadamard(TeleportedGate):
    """
    Teleported Hadamard gate.
    
    Args:
        projection_rounds: Number of stabilizer rounds for FT state preparation.
            For distance-d code, use d rounds to correct floor((d-1)/2) measurement
            errors. Set to 0 (default) to skip projection (state already encoded).
        **kwargs: Additional arguments passed to TeleportedGate.
    """
    
    def __init__(self, projection_rounds: int = 0, **kwargs):
        super().__init__(
            protocol=TELEPORTATION_PROTOCOLS["H"],
            projection_rounds=projection_rounds,
            **kwargs
        )


class BellTeleportedS(BellStateTeleportedGate):
    """
    Teleported S (phase) gate using Bell-state teleportation.
    
    This is the CORRECT implementation using Bell-state based gate teleportation.
    It requires 3 code blocks (data, ancilla1, ancilla2) but correctly implements
    the S gate teleportation.
    
    Circuit:
      |ψ⟩ ────●────H────M────  (data, measured)
              │
      |0⟩ ─H──X──●─────────M────  (ancilla1, measured)
                 │
      |0⟩ ───────X──S────────────  (ancilla2, output = S|ψ⟩)
    """
    
    def __init__(self, **kwargs):
        super().__init__(gate_name="S", **kwargs)


class BellTeleportedSDag(BellStateTeleportedGate):
    """
    Teleported S† gate using Bell-state teleportation.
    
    This is the CORRECT implementation using Bell-state based gate teleportation.
    """
    
    def __init__(self, **kwargs):
        super().__init__(gate_name="S_DAG", **kwargs)


# Legacy classes with warnings (kept for backwards compatibility)
class TeleportedS(TeleportedGate):
    """
    Teleported S (phase) gate - DEPRECATED, use BellTeleportedS instead.
    
    WARNING: This legacy protocol produces incorrect results!
    Use BellTeleportedS for correct S gate teleportation.
    """
    
    def __init__(self, **kwargs):
        import warnings
        warnings.warn(
            "TeleportedS uses a broken protocol. Use BellTeleportedS instead.",
            DeprecationWarning
        )
        super().__init__(protocol=TELEPORTATION_PROTOCOLS["S"], **kwargs)


class TeleportedSDag(TeleportedGate):
    """
    Teleported S† gate - DEPRECATED, use BellTeleportedSDag instead.
    
    WARNING: This legacy protocol produces incorrect results!
    Use BellTeleportedSDag for correct S† gate teleportation.
    """
    
    def __init__(self, **kwargs):
        import warnings
        warnings.warn(
            "TeleportedSDag uses a broken protocol. Use BellTeleportedSDag instead.",
            DeprecationWarning
        )
        super().__init__(protocol=TELEPORTATION_PROTOCOLS["S_DAG"], **kwargs)


class TeleportedT(TeleportedGate):
    """
    Teleported T gate - NOT SUPPORTED in Stim.
    
    WARNING: Stim cannot simulate non-Clifford T gates!
    The T gate requires magic state distillation which involves non-Clifford
    operations. This class is provided only as a placeholder/documentation.
    
    For T gate simulation, you would need:
    1. A separate magic state factory that produces noisy T states
    2. The error model from magic state distillation
    3. Inject these states into the teleportation circuit
    
    This is beyond Stim's Clifford-only simulation capabilities.
    """
    
    def __init__(self, **kwargs):
        import warnings
        warnings.warn(
            "TeleportedT cannot be simulated in Stim (T gate is non-Clifford). "
            "This gadget produces incorrect results and should not be used.",
            UserWarning
        )
        super().__init__(protocol=TELEPORTATION_PROTOCOLS["T"], **kwargs)


class TeleportedIdentity(TeleportedGate):
    """
    Teleported identity (for testing or as building block).
    
    Implements logical identity via teleportation, useful for:
    - Testing teleportation infrastructure
    - Logical state transfer between code blocks
    """
    
    def __init__(self, **kwargs):
        super().__init__(protocol=TELEPORTATION_PROTOCOLS["IDENTITY"], **kwargs)


def get_teleported_gadget(gate_name: str, use_bell_state: bool = True, **kwargs):
    """
    Factory function to get a teleported gadget by gate name.
    
    Args:
        gate_name: Name of gate (H, S, S_DAG, T, I/IDENTITY)
        use_bell_state: If True (default), use Bell-state teleportation for S/S_DAG
                       which is correct but requires 3 code blocks.
                       If False, use legacy 2-block protocol (broken for S/S_DAG).
        **kwargs: Additional arguments for gadget
        
    Returns:
        TeleportedGate or BellStateTeleportedGate instance
    """
    gate_name_upper = gate_name.upper()
    
    if use_bell_state and gate_name_upper in ("S", "S_DAG"):
        gate_map = {
            "S": BellTeleportedS,
            "S_DAG": BellTeleportedSDag,
        }
    else:
        gate_map = {
            "H": TeleportedHadamard,
            "S": TeleportedS,
            "S_DAG": TeleportedSDag,
            "T": TeleportedT,
            "I": TeleportedIdentity,
            "IDENTITY": TeleportedIdentity,
        }
    
    if gate_name_upper not in gate_map:
        raise ValueError(
            f"Unknown teleported gate '{gate_name}'. "
            f"Supported: H, S, S_DAG, T, I/IDENTITY"
        )
    
    return gate_map[gate_name_upper](**kwargs)
