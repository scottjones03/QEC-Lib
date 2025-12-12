# src/qectostim/experiments/ft_gadget_experiment.py
"""
Fault-Tolerant Gadget Experiment.

Implements the TQEC-style pattern for measuring logical gate error rates:

    Memory → Gadget → Memory → Measure

This ensures proper fault-tolerance by:
1. Running stabilizer rounds before the gadget (establish baseline)
2. Applying the logical gate via the gadget
3. Running stabilizer rounds after the gadget (verify stabilizer continuity)
4. Measuring all data qubits (extract logical result)

The detector network spans all phases, enabling decoding of the full
experiment including the gadget operation.

Example usage:
    >>> from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
    >>> from qectostim.codes.surface import RotatedSurfaceCode
    >>> from qectostim.gadgets import TransversalHadamard
    >>> from qectostim.noise.models import UniformDepolarizing
    >>>
    >>> code = RotatedSurfaceCode(distance=3)
    >>> gadget = TransversalHadamard()
    >>> noise = UniformDepolarizing(p=0.001)
    >>> 
    >>> exp = FaultTolerantGadgetExperiment(
    ...     codes=[code],
    ...     gadget=gadget,
    ...     noise_model=noise,
    ...     num_rounds_before=3,
    ...     num_rounds_after=3,
    ... )
    >>> circuit = exp.to_stim()
"""
from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple, Union, Set
from dataclasses import dataclass, field

import numpy as np
import stim

from qectostim.codes.abstract_code import Code, StabilizerCode
from qectostim.codes.abstract_css import CSSCode
from qectostim.experiments.experiment import Experiment
from qectostim.noise.models import NoiseModel
from qectostim.gadgets.base import Gadget, GadgetMetadata, StabilizerTransform, PhaseResult, PhaseType
from qectostim.gadgets.layout import QubitAllocation
from qectostim.experiments.stabilizer_rounds import (
    DetectorContext,
    StabilizerRoundBuilder,
    StabilizerBasis,
    get_logical_support,
)


def apply_noise_to_circuit(circuit: stim.Circuit, noise_model: Optional[NoiseModel]) -> stim.Circuit:
    """Apply noise model to circuit if provided."""
    if noise_model is None:
        return circuit
    return noise_model.apply(circuit)


@dataclass
class FTGadgetExperimentResult:
    """
    Result of a fault-tolerant gadget experiment.
    
    Attributes:
        logical_error_rate: Measured logical error rate.
        num_shots: Number of shots sampled.
        num_errors: Number of logical errors observed.
        gadget_metadata: Metadata from the gadget.
        decoder_used: Name of decoder used.
    """
    logical_error_rate: float
    num_shots: int
    num_errors: int
    gadget_metadata: Optional[GadgetMetadata] = None
    decoder_used: str = "unknown"
    extra: Dict[str, Any] = field(default_factory=dict)


class FaultTolerantGadgetExperiment(Experiment):
    """
    Fault-tolerant experiment for measuring logical gate error rates.
    
    This experiment implements the TQEC pattern:
    
    1. **Initialization**: Reset all qubits
    2. **Pre-gadget memory**: num_rounds_before stabilizer rounds
       - Establishes baseline syndrome measurements
       - Creates detectors comparing consecutive rounds
    3. **Gadget execution**: Logical gate operation
       - Uses gadget.to_stim() which should emit internal detectors
    4. **Post-gadget memory**: num_rounds_after stabilizer rounds  
       - Verifies stabilizer continuity after gate
       - Creates detectors linking back to pre-gadget rounds
    5. **Final measurement**: Measure all data qubits
       - Creates space-like detectors
       - Emits OBSERVABLE_INCLUDE for logical operator
    
    The detector graph spans all phases, enabling unified decoding.
    
    Parameters
    ----------
    codes : List[Code]
        Code(s) to apply gadget to.
    gadget : Gadget
        The logical gate gadget to test.
    noise_model : NoiseModel
        Noise model for the experiment.
    num_rounds_before : int
        Number of stabilizer rounds before gadget.
    num_rounds_after : int
        Number of stabilizer rounds after gadget.
    measurement_basis : str
        Basis for final measurement ("Z" or "X").
    metadata : Optional[Dict]
        Additional experiment metadata.
    """
    
    def __init__(
        self,
        codes: List[Code],
        gadget: Gadget,
        noise_model: NoiseModel,
        num_rounds_before: int = 3,
        num_rounds_after: int = 3,
        measurement_basis: str = "Z",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Use first code as primary for base class
        super().__init__(codes[0], noise_model, metadata)
        
        self.codes = codes
        self.gadget = gadget
        self.num_rounds_before = num_rounds_before
        self.num_rounds_after = num_rounds_after
        self.measurement_basis = measurement_basis.upper()
        
        # Cached state
        self._ctx: Optional[DetectorContext] = None
        self._builders: List[StabilizerRoundBuilder] = []
        self._qubit_allocation: Optional[Dict[str, Any]] = None
        
    def _compute_qubit_allocation(self) -> Dict[str, Any]:
        """
        Compute global qubit indices for all codes.
        
        Uses the gadget's compute_layout() if available to get the actual blocks.
        This is important for gadgets like teleportation that create more blocks
        than codes passed in.
        
        Returns a dictionary with:
        - Per-code data qubit ranges
        - Per-code ancilla qubit ranges  
        - Total qubit count
        """
        alloc = {}
        idx = 0
        
        # Try to use gadget's layout if available
        layout = None
        if hasattr(self.gadget, 'compute_layout'):
            try:
                layout = self.gadget.compute_layout(self.codes)
            except Exception:
                layout = None
        
        if layout is not None and hasattr(layout, 'blocks') and layout.blocks:
            # Use blocks from gadget's layout
            for block_name, block_info in layout.blocks.items():
                code = block_info.code
                n = code.n
                
                # Safely get hx/hz
                hx_raw = getattr(code, 'hx', None)
                hz_raw = getattr(code, 'hz', None)
                hx = hx_raw if hx_raw is not None and hasattr(hx_raw, 'shape') else np.zeros((0, code.n), dtype=np.uint8)
                hz = hz_raw if hz_raw is not None and hasattr(hz_raw, 'shape') else np.zeros((0, code.n), dtype=np.uint8)
                nx = hx.shape[0] if hx.size else 0
                nz = hz.shape[0] if hz.size else 0
                
                data_start = idx
                idx += n
                x_anc_start = idx
                idx += nx
                z_anc_start = idx
                idx += nz
                
                alloc[block_name] = {
                    "code": code,
                    "data": (data_start, n),
                    "x_anc": (x_anc_start, nx),
                    "z_anc": (z_anc_start, nz),
                }
        else:
            # Fallback: create blocks from self.codes
            for i, code in enumerate(self.codes):
                n = code.n
                # Safely get hx/hz - CSS codes have @property returning arrays
                # Non-CSS codes may not have these or may have them as methods
                hx_raw = getattr(code, 'hx', None)
                hz_raw = getattr(code, 'hz', None)
                # Only use if it's actually a numpy array (has .shape attribute)
                hx = hx_raw if hx_raw is not None and hasattr(hx_raw, 'shape') else np.zeros((0, code.n), dtype=np.uint8)
                hz = hz_raw if hz_raw is not None and hasattr(hz_raw, 'shape') else np.zeros((0, code.n), dtype=np.uint8)
                nx = hx.shape[0] if hx.size else 0
                nz = hz.shape[0] if hz.size else 0
                
                block_name = f"block_{i}"
                
                data_start = idx
                idx += n
                x_anc_start = idx
                idx += nx
                z_anc_start = idx
                idx += nz
                
                alloc[block_name] = {
                    "code": code,
                    "data": (data_start, n),
                    "x_anc": (x_anc_start, nx),
                    "z_anc": (z_anc_start, nz),
                }
        
        alloc["total"] = idx
        return alloc
    
    def _to_unified_allocation(self, alloc: Dict[str, Any]) -> QubitAllocation:
        """
        Convert dict-based allocation to QubitAllocation for gadget interface.
        
        This bridges the legacy dict format used internally with the new
        QubitAllocation dataclass used by gadgets.
        """
        from qectostim.gadgets.layout import BlockAllocation
        
        unified = QubitAllocation()
        
        for block_name, block_info in alloc.items():
            if block_name == "total":
                continue
            
            code = block_info["code"]
            data_start, n = block_info["data"]
            x_anc_start, nx = block_info["x_anc"]
            z_anc_start, nz = block_info["z_anc"]
            
            block = BlockAllocation(
                block_name=block_name,
                code=code,
                data_start=data_start,
                data_count=n,
                x_anc_start=x_anc_start,
                x_anc_count=nx,
                z_anc_start=z_anc_start,
                z_anc_count=nz,
                offset=(0.0, 0.0),  # Offset computed elsewhere
            )
            unified.blocks[block_name] = block
        
        unified._total_qubits = alloc.get("total", 0)
        return unified
    
    def _emit_qubit_coords(
        self,
        circuit: stim.Circuit,
        alloc: Dict[str, Any],
    ) -> None:
        """
        Emit QUBIT_COORDS for all qubits using gadget's compute_layout().
        
        Uses the gadget's compute_layout() method to get proper block offsets
        based on code bounding boxes instead of hardcoded offsets.
        """
        from qectostim.gadgets.coordinates import get_code_coords, get_bounding_box
        
        # Get layout from gadget
        layout = self.gadget.compute_layout(self.codes)
        
        # Build offset map from layout
        block_offsets = {}
        for block_name, block_info in layout.blocks.items():
            block_offsets[block_name] = block_info.offset
        
        for block_name, block_info in alloc.items():
            if block_name == "total":
                continue
            
            code = block_info["code"]
            
            # Get offset from layout, default to (0, 0) if not found
            offset = block_offsets.get(block_name, (0.0, 0.0))
            x_offset = offset[0] if len(offset) > 0 else 0.0
            y_offset = offset[1] if len(offset) > 1 else 0.0
            
            # Get code's coordinate metadata
            data_coords, x_stab_coords, z_stab_coords = get_code_coords(code)
            
            data_start, n = block_info["data"]
            x_anc_start, nx = block_info["x_anc"]
            z_anc_start, nz = block_info["z_anc"]
            
            # Data qubits
            for i in range(n):
                if data_coords and i < len(data_coords):
                    coord = data_coords[i]
                    circuit.append("QUBIT_COORDS", [data_start + i],
                                 [float(coord[0]) + x_offset, float(coord[1]) + y_offset])
                else:
                    circuit.append("QUBIT_COORDS", [data_start + i],
                                 [float(i) + x_offset, y_offset])
            
            # X ancillas
            for i in range(nx):
                if x_stab_coords and i < len(x_stab_coords):
                    coord = x_stab_coords[i]
                    circuit.append("QUBIT_COORDS", [x_anc_start + i],
                                 [float(coord[0]) + x_offset, float(coord[1]) + y_offset])
            
            # Z ancillas
            for i in range(nz):
                if z_stab_coords and i < len(z_stab_coords):
                    coord = z_stab_coords[i]
                    circuit.append("QUBIT_COORDS", [z_anc_start + i],
                                 [float(coord[0]) + x_offset, float(coord[1]) + y_offset])
    
    def _create_builders(
        self,
        alloc: Dict[str, Any],
        ctx: DetectorContext,
    ) -> List[StabilizerRoundBuilder]:
        """Create StabilizerRoundBuilder for each code block."""
        builders = []
        
        for block_name, block_info in alloc.items():
            if block_name == "total":
                continue
            
            code = block_info["code"]
            data_start, _ = block_info["data"]
            x_anc_start, _ = block_info["x_anc"]
            
            builder = StabilizerRoundBuilder(
                code=code,
                ctx=ctx,
                block_name=block_name,
                data_offset=data_start,
                ancilla_offset=x_anc_start,
            )
            builders.append(builder)
        
        return builders
    
    def _emit_pre_gadget_memory(
        self,
        circuit: stim.Circuit,
        builders: List[StabilizerRoundBuilder],
    ) -> None:
        """Emit pre-gadget stabilizer rounds."""
        for _ in range(self.num_rounds_before):
            for builder in builders:
                builder.emit_round(circuit, StabilizerBasis.BOTH, emit_detectors=True)
    
    def _emit_post_gadget_memory(
        self,
        circuit: stim.Circuit,
        builders: List[StabilizerRoundBuilder],
        destroyed_blocks: Optional[Set[str]] = None,
    ) -> None:
        """Emit post-gadget stabilizer rounds, skipping destroyed blocks."""
        if destroyed_blocks is None:
            destroyed_blocks = set()
        
        for _ in range(self.num_rounds_after):
            for builder in builders:
                # Skip builders for destroyed blocks (e.g., data block after teleportation)
                if builder.block_name in destroyed_blocks:
                    continue
                builder.emit_round(circuit, StabilizerBasis.BOTH, emit_detectors=True)
    
    def _emit_final_measurement(
        self,
        circuit: stim.Circuit,
        builders: List[StabilizerRoundBuilder],
        alloc: Dict[str, Any],
        ctx: DetectorContext,
        destroyed_blocks: Optional[Set[str]] = None,
        teleport_correction_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit final data qubit measurements and observables, skipping destroyed blocks."""
        if destroyed_blocks is None:
            destroyed_blocks = set()
        
        # Collect data qubits only from surviving blocks
        all_data_qubits = []
        surviving_alloc = {}
        for block_name, block_info in alloc.items():
            if block_name == "total":
                continue
            if block_name in destroyed_blocks:
                continue  # Skip destroyed blocks
            data_start, n = block_info["data"]
            all_data_qubits.extend(range(data_start, data_start + n))
            surviving_alloc[block_name] = block_info
        
        if not all_data_qubits:
            # All blocks destroyed - nothing to measure
            return
        
        # Get the effective measurement basis after transformations
        # If a Hadamard was applied and we started with Z basis, 
        # we need to measure in X basis to get deterministic results
        # (or vice versa)
        effective_basis = ctx.get_transformed_basis(0, self.measurement_basis)
        
        # Apply basis rotation if needed
        if effective_basis == "X":
            circuit.append("H", all_data_qubits)
            circuit.append("TICK")
        
        # Measure surviving data qubits (always in Z after possible H rotation)
        meas_start = ctx.add_measurement(len(all_data_qubits))
        circuit.append("M", all_data_qubits)
        
        # Emit space-like detectors (compare final data with last stabilizer round)
        # Only for surviving blocks, and skip for teleportation gadgets
        # Teleportation gadgets transfer state between blocks, so the ancilla's
        # stabilizers after teleportation don't match its own previous rounds
        is_teleportation = hasattr(self.gadget, 'is_teleportation_gadget') and self.gadget.is_teleportation_gadget()
        
        if not is_teleportation:
            surviving_builders = [b for b in builders if b.block_name not in destroyed_blocks]
            for builder in surviving_builders:
                builder.emit_space_like_detectors(circuit, effective_basis)
        
        # Emit observable include using surviving allocation
        self._emit_observable(circuit, surviving_alloc, ctx, meas_start, teleport_correction_info)
    
    def _emit_observable(
        self,
        circuit: stim.Circuit,
        alloc: Dict[str, Any],
        ctx: DetectorContext,
        meas_start: int,
        teleport_correction_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Emit OBSERVABLE_INCLUDE for logical observable.
        
        This method handles three cases:
        1. Single-qubit gates: Observable on the single code block
        2. Two-qubit gates (CNOT, CZ, SWAP): Observable may span both blocks
        3. Teleportation gadgets: Observable is on the ancilla block (output)
           - For teleportation, we also include Bell measurement corrections
        
        The key insight is that after a gadget, the logical observable may be:
        - Transformed (e.g., H: Z→X)
        - Spread across blocks (e.g., CNOT: Z_ctrl → Z_ctrl ⊗ Z_tgt)
        - Relocated to a different block (teleportation: data→ancilla)
        - Require classical corrections (teleportation: XOR with Bell measurement)
        """
        # Build a mapping from global qubit index to measurement index
        # Data qubits are measured in order they appear in all_data_qubits
        qubit_to_meas = {}
        meas_idx = meas_start
        for block_name, block_info in alloc.items():
            if block_name == "total":
                continue
            data_start, n = block_info["data"]
            for i in range(n):
                qubit_to_meas[data_start + i] = meas_idx
                meas_idx += 1
        
        # Get the effective observable basis after transformations
        # If a Hadamard was applied, Z becomes X and vice versa
        effective_basis = ctx.get_transformed_basis(0, self.measurement_basis)
        
        # Determine which blocks contribute to the observable
        observable_meas = []
        
        # Check for teleportation gadget - output is on ancilla block
        is_teleportation = hasattr(self.gadget, 'is_teleportation_gadget') and self.gadget.is_teleportation_gadget()
        
        # Check for two-qubit gate with cross-block observable propagation
        has_two_qubit_transform = hasattr(self.gadget, 'get_two_qubit_observable_transform')
        
        if is_teleportation:
            # For teleportation, the output logical state is on the ancilla block
            # But we also need to include the Bell measurement corrections
            #
            # In teleportation:
            # - Data qubits are measured in X basis (via H then M)
            # - The X measurement outcomes determine Z corrections needed on output
            # - For Z observable: include output Z support AND input X measurement support
            #
            # Pauli correction rule for teleportation with Z observable:
            # Observable = Z_output XOR X_correction_from_data
            # where X_correction_from_data = XOR of data qubit measurements that support logical X
            
            output_block_name = self.gadget.get_output_block_name()
            
            # Find the output block and add its measurements
            for block_name, block_info in alloc.items():
                if block_name == "total":
                    continue
                
                # Match by name or by index (block_1 is typically the ancilla)
                is_output_block = (
                    block_name == output_block_name or 
                    (output_block_name == "ancilla_block" and block_name == "block_1")
                )
                
                if is_output_block:
                    code = block_info["code"]
                    data_start, _ = block_info["data"]
                    
                    # Get logical operator support in the effective basis
                    support = get_logical_support(code, effective_basis, 0)
                    
                    for local_idx in support:
                        global_qubit_idx = data_start + local_idx
                        if global_qubit_idx in qubit_to_meas:
                            observable_meas.append(qubit_to_meas[global_qubit_idx])
                    break
            
            # Now add the Bell measurement corrections from the data block
            # For Z-basis observable, we need to XOR with the X-basis measurements
            # on qubits that support the logical X operator (dual to the Z we're measuring)
            if teleport_correction_info is not None and effective_basis == "Z":
                bell_meas_start = teleport_correction_info['meas_start']
                bell_qubits = teleport_correction_info['measurement_qubits']
                
                # Get the input code (first/only code in self.codes)
                input_code = self.codes[0]
                
                # For Z observable, we need X correction - get logical X support
                # The X measurement on data qubits that support logical X gives Z correction
                x_support = get_logical_support(input_code, "X", 0)
                
                # The data block is block_0, starting at qubit index 0
                # Bell measurement measures data qubits in order [0, 1, 2, 3, ...]
                # bell_qubits gives us the global indices that were measured
                
                # Map bell_qubits to their measurement indices
                for i, qubit in enumerate(bell_qubits):
                    # Check if this qubit is in the logical X support
                    # The qubit's local index within the data block
                    local_idx = qubit  # For block_0, data starts at 0
                    if local_idx in x_support:
                        # This Bell measurement should be XORed into observable
                        observable_meas.append(bell_meas_start + i)
            
            # For X-basis observable, we would need Z correction from the Bell measurement
            # (the CNOT direction determines which Pauli correction is needed)
            elif teleport_correction_info is not None and effective_basis == "X":
                bell_meas_start = teleport_correction_info['meas_start']
                bell_qubits = teleport_correction_info['measurement_qubits']
                
                input_code = self.codes[0]
                
                # For X observable, we need Z correction - get logical Z support
                z_support = get_logical_support(input_code, "Z", 0)
                
                for i, qubit in enumerate(bell_qubits):
                    local_idx = qubit
                    if local_idx in z_support:
                        observable_meas.append(bell_meas_start + i)
        
        elif has_two_qubit_transform and self.measurement_basis == "Z":
            # For two-qubit gates, check if Z observable spreads across blocks
            # CNOT: Z_ctrl → Z_ctrl ⊗ Z_tgt (need Z on both blocks)
            # CZ: Z unchanged on both blocks
            # SWAP: Z_ctrl ↔ Z_tgt (swapped)
            two_q_transform = self.gadget.get_two_qubit_observable_transform()
            
            # Determine which blocks contribute to the Z observable
            # We're measuring observable 0 (first logical qubit) which starts on block_0
            ctrl_z_to = two_q_transform.control_z_to  # (block_0_component, block_1_component)
            
            blocks_needed = []
            if ctrl_z_to[0] is not None:
                blocks_needed.append("block_0")
            if ctrl_z_to[1] is not None:
                blocks_needed.append("block_1")
            
            for block_name, block_info in alloc.items():
                if block_name == "total":
                    continue
                if block_name not in blocks_needed:
                    continue
                
                code = block_info["code"]
                data_start, _ = block_info["data"]
                
                # Get the appropriate logical operator for this block
                # After CNOT, if Z_ctrl spreads to Z_tgt, both blocks have Z
                support = get_logical_support(code, "Z", 0)
                
                for local_idx in support:
                    global_qubit_idx = data_start + local_idx
                    if global_qubit_idx in qubit_to_meas:
                        observable_meas.append(qubit_to_meas[global_qubit_idx])
        
        else:
            # Standard case: single-qubit gate or simple two-qubit gate
            # Observable is on all blocks with the effective basis
            for i, (block_name, block_info) in enumerate(alloc.items()):
                if block_name == "total":
                    continue
                
                code = block_info["code"]
                data_start, _ = block_info["data"]
                
                # Get logical operator support using the EFFECTIVE basis
                # After Hadamard, if we measure in Z, we need the logical that
                # WAS X before the Hadamard (since H swaps X<->Z)
                support = get_logical_support(code, effective_basis, 0)
                
                # Map local qubit indices to global measurement indices
                for local_idx in support:
                    global_qubit_idx = data_start + local_idx
                    if global_qubit_idx in qubit_to_meas:
                        observable_meas.append(qubit_to_meas[global_qubit_idx])
                
                # For single-block gadgets, only use the first block
                if not has_two_qubit_transform:
                    break
        
        # Emit observable
        if observable_meas:
            lookbacks = [idx - ctx.measurement_index for idx in observable_meas]
            targets = [stim.target_rec(lb) for lb in lookbacks]
            circuit.append("OBSERVABLE_INCLUDE", targets, 0)
    
    def to_stim(self) -> stim.Circuit:
        """
        Generate complete fault-tolerant Stim circuit.
        
        The circuit includes:
        - QUBIT_COORDS for visualization
        - Pre-gadget memory rounds with DETECTOR
        - Gadget circuit (from gadget.to_stim, but we use our own detector tracking)
        - Post-gadget memory rounds with DETECTOR
        - Final measurement with OBSERVABLE_INCLUDE
        
        Returns
        -------
        stim.Circuit
            Complete Stim circuit for the experiment.
        """
        circuit = stim.Circuit()
        
        # Initialize context
        ctx = DetectorContext()
        self._ctx = ctx
        
        # Compute qubit allocation
        alloc = self._compute_qubit_allocation()
        self._qubit_allocation = alloc
        
        # =====================================================================
        # Phase 1: QUBIT_COORDS and Reset
        # =====================================================================
        self._emit_qubit_coords(circuit, alloc)
        
        total = alloc["total"]
        if total > 0:
            circuit.append("R", list(range(total)))
            circuit.append("TICK")
        
        # =====================================================================
        # Phase 2: Create builders for each code block
        # =====================================================================
        builders = self._create_builders(alloc, ctx)
        self._builders = builders
        
        # =====================================================================
        # Phase 3: Pre-gadget memory rounds
        # =====================================================================
        self._emit_pre_gadget_memory(circuit, builders)
        
        # =====================================================================
        # Phase 4: Gadget execution via emit_next_phase() interface
        # =====================================================================
        # Convert dict-based allocation to QubitAllocation for gadget interface
        unified_alloc = self._to_unified_allocation(alloc)
        
        # Reset gadget phase counter
        self.gadget.reset_phases()
        
        # Track which blocks are destroyed during gadget execution
        destroyed_blocks = set()
        
        # Track teleportation Bell measurement info for observable correction
        teleport_correction_info = None
        is_teleportation = hasattr(self.gadget, 'is_teleportation_gadget') and self.gadget.is_teleportation_gadget()
        
        # Execute all phases of the gadget
        for phase_idx in range(self.gadget.num_phases):
            # Emit the next phase
            result = self.gadget.emit_next_phase(circuit, unified_alloc, ctx)
            
            # Track measured/destroyed blocks
            if result.measured_blocks:
                destroyed_blocks.update(result.measured_blocks)
            
            # Track gadget measurements in the detector context
            # This is crucial for correct detector references after gadget
            if result.measurement_count > 0:
                meas_start = ctx.add_measurement(result.measurement_count)
                
                # For teleportation, track the Bell measurement info
                # This is needed for observable corrections
                if is_teleportation and result.phase_type == PhaseType.MEASUREMENT:
                    teleport_correction_info = {
                        'meas_start': meas_start,
                        'measurement_qubits': result.measurement_qubits,
                        'measurement_count': result.measurement_count,
                    }
            
            # Handle stabilizer rounds between phases if requested
            if result.needs_stabilizer_rounds > 0 and not result.is_final:
                for _ in range(result.needs_stabilizer_rounds):
                    for builder in builders:
                        # Skip stabilizer rounds on destroyed blocks
                        if builder.block_name not in destroyed_blocks:
                            builder.emit_round(circuit, StabilizerBasis.BOTH, emit_detectors=True)
            
            # Apply stabilizer transform if provided
            if result.stabilizer_transform is not None:
                transform = result.stabilizer_transform
                if transform.clear_history or transform.swap_xz:
                    for block_name in alloc:
                        if block_name != "total":
                            ctx.clear_stabilizer_history(
                                block_name=block_name,
                                swap_xz=transform.swap_xz
                            )
            
            # Check if gadget is done
            if result.is_final:
                break
        
        # Apply overall observable transform from the gadget
        if hasattr(self.gadget, 'gate_name'):
            ctx.update_for_gate(self.gadget.gate_name)
        elif hasattr(self.gadget, 'protocol') and hasattr(self.gadget.protocol, 'gate_name'):
            # Teleportation gadgets
            ctx.update_for_gate(self.gadget.protocol.gate_name)
        
        # Get overall stabilizer transform and apply
        overall_transform = self.gadget.get_stabilizer_transform()
        if overall_transform.clear_history or overall_transform.swap_xz:
            for block_name in alloc:
                if block_name != "total":
                    ctx.clear_stabilizer_history(
                        block_name=block_name,
                        swap_xz=overall_transform.swap_xz
                    )
        
        # =====================================================================
        # Phase 5: Post-gadget memory rounds (skip destroyed blocks)
        # =====================================================================
        self._emit_post_gadget_memory(circuit, builders, destroyed_blocks)
        
        # =====================================================================
        # Phase 6: Final measurement (skip destroyed blocks)
        # =====================================================================
        self._emit_final_measurement(circuit, builders, alloc, ctx, destroyed_blocks, teleport_correction_info)
        
        # Apply noise model
        circuit = apply_noise_to_circuit(circuit, self.noise_model)
        
        return circuit
    
    def run_decode(
        self,
        decoder_name: str = "pymatching",
        num_shots: int = 10000,
        **kwargs,
    ) -> FTGadgetExperimentResult:
        """
        Run the experiment and decode results.
        
        Parameters
        ----------
        decoder_name : str
            Decoder to use ("pymatching", "beliefmatching", etc.)
        num_shots : int
            Number of shots to sample.
        **kwargs
            Additional decoder arguments.
            
        Returns
        -------
        FTGadgetExperimentResult
            Experiment results including logical error rate.
        """
        from qectostim.decoders.registry import get_decoder
        
        # Generate circuit
        circuit = self.to_stim()
        
        # Get decoder
        decoder = get_decoder(decoder_name, circuit)
        
        # Sample and decode
        sampler = circuit.compile_detector_sampler()
        samples = sampler.sample(num_shots, append_observables=True)
        
        # Extract detectors and observables
        num_detectors = circuit.num_detectors
        num_observables = circuit.num_observables
        
        detector_shots = samples[:, :num_detectors]
        observable_shots = samples[:, num_detectors:num_detectors + num_observables]
        
        # Decode
        predictions = decoder.decode_batch(detector_shots)
        
        # Count errors
        num_errors = int(np.sum(predictions != observable_shots))
        logical_error_rate = num_errors / num_shots
        
        # Get gadget metadata if available
        gadget_metadata = None
        if hasattr(self.gadget, 'get_metadata'):
            try:
                gadget_metadata = self.gadget.get_metadata()
            except RuntimeError:
                pass
        
        return FTGadgetExperimentResult(
            logical_error_rate=logical_error_rate,
            num_shots=num_shots,
            num_errors=num_errors,
            gadget_metadata=gadget_metadata,
            decoder_used=decoder_name,
            extra={
                "num_detectors": num_detectors,
                "num_observables": num_observables,
                "rounds_before": self.num_rounds_before,
                "rounds_after": self.num_rounds_after,
            },
        )
    
    def verify_zero_noise_ler(
        self,
        num_shots: int = 1000,
        tolerance: float = 0.0,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Verify that the circuit has zero logical error rate with no noise.
        
        This is a critical sanity check: if the gadget correctly implements
        the logical gate and the observable is properly tracked, then with
        no noise the logical error rate MUST be zero. Any non-zero LER
        indicates a bug in:
        - Observable tracking (wrong qubits included in OBSERVABLE_INCLUDE)
        - Stabilizer transform application (detectors reference wrong measurements)
        - Gate implementation (wrong physical gates emitted)
        
        Parameters
        ----------
        num_shots : int
            Number of shots to sample (default 1000).
        tolerance : float
            Acceptable error rate (default 0.0 for strict verification).
            
        Returns
        -------
        Tuple[bool, float, Optional[str]]
            (passed, ler, error_message)
            - passed: True if LER <= tolerance
            - ler: Measured logical error rate
            - error_message: Description of failure if any
        """
        # Create a copy of the experiment with no noise
        no_noise_exp = FaultTolerantGadgetExperiment(
            codes=self.codes,
            gadget=self.gadget,
            noise_model=None,  # No noise!
            num_rounds_before=self.num_rounds_before,
            num_rounds_after=self.num_rounds_after,
            measurement_basis=self.measurement_basis,
        )
        
        try:
            circuit = no_noise_exp.to_stim()
        except Exception as e:
            return False, 1.0, f"Circuit generation failed: {e}"
        
        # Check for required components
        circuit_str = str(circuit)
        if "DETECTOR" not in circuit_str:
            return False, 1.0, "No DETECTOR instructions in circuit"
        if "OBSERVABLE_INCLUDE" not in circuit_str:
            return False, 1.0, "No OBSERVABLE_INCLUDE in circuit"
        
        try:
            # Sample from circuit (no noise, so should be deterministic)
            sampler = circuit.compile_detector_sampler()
            samples = sampler.sample(num_shots, append_observables=True)
            
            num_detectors = circuit.num_detectors
            num_observables = circuit.num_observables
            
            if num_observables == 0:
                return False, 1.0, "Circuit has 0 observables"
            
            # Check detector values (should all be 0 with no noise)
            detector_shots = samples[:, :num_detectors]
            observable_shots = samples[:, num_detectors:num_detectors + num_observables]
            
            # Detector check: any non-zero detector indicates a problem
            total_detector_flips = np.sum(detector_shots)
            if total_detector_flips > 0:
                flip_rate = total_detector_flips / (num_shots * num_detectors)
                return False, 1.0, f"Detectors firing with no noise (flip_rate={flip_rate:.4f})"
            
            # Observable check: should all be 0 (identity result)
            ler = float(np.mean(observable_shots[:, 0]))
            
            if ler > tolerance:
                return False, ler, f"Zero-noise LER = {ler:.4f} > {tolerance} (observable tracking bug)"
            
            return True, ler, None
            
        except Exception as e:
            return False, 1.0, f"Verification failed: {e}"


def run_ft_gadget_experiment(
    codes: List[Code],
    gadget: Gadget,
    noise_model: NoiseModel,
    num_rounds_before: int = 3,
    num_rounds_after: int = 3,
    decoder_name: str = "pymatching",
    num_shots: int = 10000,
) -> FTGadgetExperimentResult:
    """
    Convenience function to run a fault-tolerant gadget experiment.
    
    Parameters
    ----------
    codes : List[Code]
        Code(s) to apply gadget to.
    gadget : Gadget
        The logical gate gadget.
    noise_model : NoiseModel
        Noise model.
    num_rounds_before : int
        Stabilizer rounds before gadget.
    num_rounds_after : int
        Stabilizer rounds after gadget.
    decoder_name : str
        Decoder to use.
    num_shots : int
        Number of shots.
        
    Returns
    -------
    FTGadgetExperimentResult
        Experiment results.
    """
    exp = FaultTolerantGadgetExperiment(
        codes=codes,
        gadget=gadget,
        noise_model=noise_model,
        num_rounds_before=num_rounds_before,
        num_rounds_after=num_rounds_after,
    )
    return exp.run_decode(decoder_name=decoder_name, num_shots=num_shots)
