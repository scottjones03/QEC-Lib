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
from qectostim.gadgets.scheduling import GadgetScheduler, CircuitLayer

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
    """
    
    def __init__(
        self,
        protocol: TeleportationProtocol,
        include_stabilizer_rounds: bool = True,
        num_rounds_before: int = 1,
        num_rounds_after: int = 1,
    ):
        """
        Initialize teleported gate gadget.
        
        Args:
            protocol: TeleportationProtocol for this gate
            include_stabilizer_rounds: Include stabilizer measurements
            num_rounds_before: Stabilizer rounds before teleportation
            num_rounds_after: Stabilizer rounds after teleportation
        """
        super().__init__()
        
        self.protocol = protocol
        self.include_stabilizer_rounds = include_stabilizer_rounds
        self.num_rounds_before = num_rounds_before
        self.num_rounds_after = num_rounds_after
        
        # State tracking
        self._ancilla_block_name = "ancilla_block"
        self._data_block_name = "data_block"
    
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
        Phase 0: Ancilla preparation
        Phase 1: CNOT + H for Bell measurement setup
        Phase 2: Measurements (is_final=True with MEASUREMENT type)
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
        
        Phase 0: Ancilla preparation (H, S, etc.)
        Phase 1: CNOT + H for Bell measurement setup
        Phase 2: Measurements
        
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
        ancilla_qubits = list(ancilla_block.data_range)[:n]
        
        phase = self._current_phase
        self._current_phase += 1
        
        if phase == 0:
            # Phase 0: Ancilla preparation
            if self.protocol.ancilla_state == AncillaState.PLUS:
                circuit.append("H", ancilla_qubits)
            elif self.protocol.ancilla_state == AncillaState.ZERO:
                pass  # Already in |0⟩
            elif self.protocol.ancilla_state == AncillaState.Y_PLUS:
                circuit.append("H", ancilla_qubits)
                circuit.append("S", ancilla_qubits)
            elif self.protocol.ancilla_state == AncillaState.BELL:
                circuit.append("H", ancilla_qubits)
            elif self.protocol.ancilla_state == AncillaState.MAGIC_T:
                # IDEALIZED magic state preparation
                # In reality, magic states are prepared via distillation
                # Stim cannot simulate the T gate (non-Clifford), so we 
                # prepare |+⟩ as a placeholder. The error model for magic
                # state distillation should be applied separately.
                circuit.append("H", ancilla_qubits)
                # NOTE: T gate omitted - Stim only supports Clifford gates
            
            circuit.append("TICK")
            
            return PhaseResult(
                phase_type=PhaseType.PREPARATION,
                is_final=False,
                needs_stabilizer_rounds=0,  # No rounds needed after prep
            )
        
        elif phase == 1:
            # Phase 1: CNOT + H for Bell measurement setup
            # After this phase, data and ancilla qubits are entangled in Bell pairs.
            # We CANNOT emit stabilizer rounds here because the stabilizers are
            # non-local (spanning both blocks) and measuring them individually
            # would give random results.
            for d, a in zip(data_qubits, ancilla_qubits):
                circuit.append("CNOT", [d, a])
            circuit.append("TICK")
            
            circuit.append("H", data_qubits)
            circuit.append("TICK")
            
            return PhaseResult(
                phase_type=PhaseType.GATE,
                is_final=False,
                needs_stabilizer_rounds=0,  # NO stabilizer rounds while entangled!
            )
        
        elif phase == 2:
            # Phase 2: Measurements (Bell measurement on data qubits only)
            # In teleportation, we measure the DATA qubits in Bell basis
            # The ANCILLA qubits are the OUTPUT and should NOT be measured here
            # (Pauli corrections would be applied based on measurement outcomes)
            # 
            # After this measurement, the ancilla block has the teleported state.
            # We must clear the stabilizer history because:
            # 1. The data block is destroyed (measured)
            # 2. The ancilla block's previous measurements were while entangled
            # 3. The ancilla now holds independent stabilizers again
            #
            # For gates like Hadamard that swap X↔Z, we also need to set swap_xz=True
            # so the builder knows to swap its measurement_basis accordingly.
            circuit.append("M", data_qubits)
            circuit.append("TICK")
            
            # Get the gate-specific stabilizer transform
            # This handles gates like H (swap X↔Z) and S (X→Y, Z→Z)
            gate_transform = self.get_stabilizer_transform()
            
            # Return with the full stabilizer transform including swap_xz if needed
            return PhaseResult(
                phase_type=PhaseType.MEASUREMENT,
                is_final=True,
                measurement_count=len(data_qubits),
                measurement_qubits=list(data_qubits),
                measured_blocks=["data_block"],  # Only data block is measured/destroyed
                stabilizer_transform=gate_transform,  # Use gate-specific transform
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
    
    def build_schedule(
        self,
        layout: GadgetLayout,
        codes: List[Code],
        noise_model: Optional[NoiseModel] = None,
    ) -> GadgetScheduler:
        """
        Build schedule for teleportation protocol.
        
        Layers:
        1. Pre-stabilizer rounds on data block
        2. Ancilla preparation
        3. CNOT between data and ancilla (Bell measurement prep)
        4. X measurement on data, Z measurement on ancilla
        5. Post-stabilizer rounds (with Pauli frame update)
        """
        scheduler = GadgetScheduler(layout)
        code = codes[0]
        
        data_block = layout.blocks[self._data_block_name]
        ancilla_block = layout.blocks[self._ancilla_block_name]
        
        n = code.n
        time = 0.0
        
        # Pre-stabilizer rounds
        if self.include_stabilizer_rounds:
            for _ in range(self.num_rounds_before):
                scheduler.schedule_stabilizer_round(
                    code=code,
                    data_offset=data_block.data_qubit_range[0],
                    ancilla_offset=data_block.ancilla_qubit_range[0],
                    time=time,
                )
                time += 1.0
        
        # Ancilla preparation layer
        layer = CircuitLayer(time=time)
        ancilla_start = ancilla_block.data_qubit_range[0]
        
        if self.protocol.ancilla_state == AncillaState.PLUS:
            # Prepare |+⟩: Reset then H
            layer.resets.extend(range(ancilla_start, ancilla_start + n))
            scheduler.layers.append(layer)
            time += 1.0
            
            layer = CircuitLayer(time=time)
            layer.single_qubit_gates.extend([
                ("H", ancilla_start + i) for i in range(n)
            ])
            scheduler.layers.append(layer)
            
        elif self.protocol.ancilla_state == AncillaState.ZERO:
            # Prepare |0⟩: Just reset
            layer.resets.extend(range(ancilla_start, ancilla_start + n))
            scheduler.layers.append(layer)
            
        elif self.protocol.ancilla_state == AncillaState.Y_PLUS:
            # Prepare |+i⟩: Reset, H, S
            layer.resets.extend(range(ancilla_start, ancilla_start + n))
            scheduler.layers.append(layer)
            time += 1.0
            
            layer = CircuitLayer(time=time)
            layer.single_qubit_gates.extend([
                ("H", ancilla_start + i) for i in range(n)
            ])
            scheduler.layers.append(layer)
            time += 1.0
            
            layer = CircuitLayer(time=time)
            layer.single_qubit_gates.extend([
                ("S", ancilla_start + i) for i in range(n)
            ])
            scheduler.layers.append(layer)
            
        elif self.protocol.ancilla_state == AncillaState.BELL:
            # Prepare Bell pair: Reset both, H on ancilla, CNOT
            layer.resets.extend(range(ancilla_start, ancilla_start + n))
            scheduler.layers.append(layer)
            time += 1.0
            
            layer = CircuitLayer(time=time)
            layer.single_qubit_gates.extend([
                ("H", ancilla_start + i) for i in range(n)
            ])
            scheduler.layers.append(layer)
            
        elif self.protocol.ancilla_state == AncillaState.MAGIC_T:
            # Magic state injection - placeholder
            # In practice, this requires distillation
            layer.resets.extend(range(ancilla_start, ancilla_start + n))
            scheduler.layers.append(layer)
            time += 1.0
            
            # Approximate |T⟩ state prep
            layer = CircuitLayer(time=time)
            layer.single_qubit_gates.extend([
                ("H", ancilla_start + i) for i in range(n)
            ])
            scheduler.layers.append(layer)
            time += 1.0
            
            layer = CircuitLayer(time=time)
            layer.single_qubit_gates.extend([
                ("T", ancilla_start + i) for i in range(n)
            ])
            scheduler.layers.append(layer)
        
        time += 1.0
        
        # CNOT between data and ancilla (Bell measurement preparation)
        layer = CircuitLayer(time=time)
        data_start = data_block.data_qubit_range[0]
        layer.two_qubit_gates.extend([
            ("CNOT", data_start + i, ancilla_start + i)
            for i in range(n)
        ])
        scheduler.layers.append(layer)
        time += 1.0
        
        # Hadamard on data qubits (for X measurement)
        layer = CircuitLayer(time=time)
        layer.single_qubit_gates.extend([
            ("H", data_start + i) for i in range(n)
        ])
        scheduler.layers.append(layer)
        time += 1.0
        
        # Measurements
        layer = CircuitLayer(time=time)
        # Measure data (X basis, after H)
        layer.measurements.extend([(data_start + i, "Z") for i in range(n)])
        # Measure ancilla (Z basis)
        layer.measurements.extend([(ancilla_start + i, "Z") for i in range(n)])
        scheduler.layers.append(layer)
        time += 1.0
        
        # Post-stabilizer rounds on remaining qubits
        # (In practice, the output is now on the ancilla side)
        if self.include_stabilizer_rounds:
            for _ in range(self.num_rounds_after):
                scheduler.schedule_stabilizer_round(
                    code=code,
                    data_offset=ancilla_block.data_qubit_range[0],
                    ancilla_offset=ancilla_block.ancilla_qubit_range[0],
                    time=time,
                )
                time += 1.0
        
        self._cached_scheduler = scheduler
        return scheduler
    
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
    ):
        super().__init__()
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
            # Phase 0: Create Bell pair on (ancilla1, ancilla2)
            # H on ancilla1, then CNOT(ancilla1 → ancilla2)
            circuit.append("H", ancilla1_qubits)
            circuit.append("TICK")
            for a1, a2 in zip(ancilla1_qubits, ancilla2_qubits):
                circuit.append("CNOT", [a1, a2])
            circuit.append("TICK")
            
            return PhaseResult(
                phase_type=PhaseType.PREPARATION,
                is_final=False,
                needs_stabilizer_rounds=0,
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
    """Teleported Hadamard gate."""
    
    def __init__(self, **kwargs):
        super().__init__(protocol=TELEPORTATION_PROTOCOLS["H"], **kwargs)


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
