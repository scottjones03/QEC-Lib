# src/qectostim/experiments/gadgets/combinators.py
"""
Meta-Gadgets / Gadget Combinators for composing EC operations.

This module provides a clean separation of concerns for fault-tolerant EC:

GATE IMPLEMENTATION STRATEGIES (How gates are physically realized):
-------------------------------------------------------------------
- TransversalStrategy: Bitwise parallel gates (default for CSS codes)
- TeleportationStrategy: Gate teleportation (future)
- LatticeSurgeryStrategy: Lattice surgery operations (future)

COMPOSITION PATTERNS (Meta-Gadgets):
------------------------------------
- ParallelGadget: Apply a gadget to multiple blocks simultaneously
- RecursiveGadget: Apply gadgets hierarchically (children-first)
- ChainedGadget: Apply multiple gadgets in sequence
- ConditionalGadget: Apply gadget based on syndrome conditions

EC PROTOCOL GADGETS (Base - what error correction to do):
---------------------------------------------------------
- SteaneECGadget: Cat-state ancilla syndrome extraction
- KnillECGadget: Teleportation-based EC with Bell pairs

ARCHITECTURE:
-------------
The architecture cleanly separates three concerns:

1. EC PROTOCOL (What): SteaneEC, KnillEC define the error correction protocol
2. GATE STRATEGY (How): Transversal, Teleportation, LatticeSurgery define
   how logical gates are physically implemented
3. COMPOSITION (Structure): Parallel, Recursive, Chained define how gadgets
   are combined across blocks

USAGE EXAMPLES:
---------------
# Apply Steane EC to all 7 inner blocks with transversal gates (default)
inner_steane = SteaneECGadget(inner_code)  # Uses TransversalStrategy by default
parallel_inner = ParallelGadget(inner_steane, n_blocks=7)

# Explicit strategy selection (for future use)
from qectostim.experiments.gadgets import TransversalStrategy
steane_transversal = SteaneECGadget(code, gate_strategy=TransversalStrategy())

# Recursive EC: Knill at inner level, Steane at outer
recursive_ec = RecursiveGadget(
    code=concatenated_code,
    level_gadgets={
        0: SteaneECGadget(outer_code),   # Outer uses Steane
        1: KnillECGadget(inner_code),     # Inner uses Knill  
    }
)

# Chain: parallel inner EC, then outer EC
full_ec = ChainedGadget([parallel_inner, outer_steane])
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple, Callable, Union

import stim
import numpy as np

from .base import Gadget, MeasurementMap, SyndromeSchedule, LogicalMeasurementMap

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode
    from qectostim.codes.composite.multilevel_concatenated import MultiLevelConcatenatedCode
    from qectostim.noise.models import NoiseModel


# =============================================================================
# GATE IMPLEMENTATION STRATEGIES
# =============================================================================

class GateStrategyType(Enum):
    """Enumeration of gate implementation strategies."""
    TRANSVERSAL = auto()      # Bitwise parallel gates
    TELEPORTATION = auto()    # Gate teleportation
    LATTICE_SURGERY = auto()  # Lattice surgery operations
    

class GateImplementationStrategy(ABC):
    """
    Abstract base class for gate implementation strategies.
    
    A strategy defines HOW logical gates are physically implemented on
    encoded qubits. This is separate from WHAT the EC protocol does.
    
    For example:
    - TransversalStrategy: CNOT_L = CNOT^⊗n (bitwise parallel)
    - TeleportationStrategy: CNOT_L via gate teleportation with magic states
    - LatticeSurgeryStrategy: CNOT_L via merge/split operations
    
    This abstraction allows EC protocol gadgets (Steane, Knill) to work
    with different physical implementations without modification.
    """
    
    @property
    @abstractmethod
    def strategy_type(self) -> GateStrategyType:
        """Return the type of this strategy."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the strategy."""
        pass
    
    @abstractmethod
    def emit_logical_cnot(
        self,
        circuit: stim.Circuit,
        control_qubits: List[int],
        target_qubits: List[int],
        noise_model: Optional["NoiseModel"] = None,
    ) -> int:
        """
        Emit a logical CNOT between two encoded blocks.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to append gates to.
        control_qubits : List[int]
            Physical qubits of the control block.
        target_qubits : List[int]
            Physical qubits of the target block.
        noise_model : NoiseModel, optional
            Noise to apply.
            
        Returns
        -------
        int
            Number of measurements added (for non-unitary strategies).
        """
        pass
    
    @abstractmethod
    def emit_logical_h(
        self,
        circuit: stim.Circuit,
        qubits: List[int],
        noise_model: Optional["NoiseModel"] = None,
    ) -> int:
        """Emit a logical Hadamard on an encoded block."""
        pass
    
    @abstractmethod
    def emit_measurement(
        self,
        circuit: stim.Circuit,
        qubits: List[int],
        basis: str = 'Z',
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> List[int]:
        """
        Emit logical measurement.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to append to.
        qubits : List[int]
            Physical qubits to measure.
        basis : str
            'Z' or 'X' basis.
        noise_model : NoiseModel, optional
            Measurement noise.
            
        Returns
        -------
        List[int]
            Indices of the measurements added (absolute, offset included).
        """
        pass


class TransversalStrategy(GateImplementationStrategy):
    """
    Transversal (bitwise parallel) gate implementation.
    
    For CSS codes, logical gates are implemented as:
    - CNOT_L = CNOT^⊗n (CNOT on each qubit pair)
    - H_L = H^⊗n (Hadamard on each qubit)
    - M_L = M^⊗n (measure each qubit)
    
    This is the default strategy for most concatenated code constructions.
    
    Properties:
    - Fault-tolerant: single-qubit errors don't spread
    - Simple: direct physical implementation
    - Limited: only works for gates in code's transversal gate set
    """
    
    @property
    def strategy_type(self) -> GateStrategyType:
        return GateStrategyType.TRANSVERSAL
    
    @property
    def name(self) -> str:
        return "Transversal"
    
    def emit_logical_cnot(
        self,
        circuit: stim.Circuit,
        control_qubits: List[int],
        target_qubits: List[int],
        noise_model: Optional["NoiseModel"] = None,
    ) -> int:
        """
        Emit transversal CNOT: CNOT[i] for each qubit pair.
        
        This maps:
        - X errors: control → target (X_c → X_c X_t)
        - Z errors: target → control (Z_t → Z_c Z_t)
        """
        if len(control_qubits) != len(target_qubits):
            raise ValueError(
                f"Block sizes must match for transversal CNOT: "
                f"{len(control_qubits)} vs {len(target_qubits)}"
            )
        
        # Apply bitwise CNOTs
        for ctrl, tgt in zip(control_qubits, target_qubits):
            circuit.append("CNOT", [ctrl, tgt])
        
        # Apply noise if provided
        if noise_model is not None:
            for ctrl, tgt in zip(control_qubits, target_qubits):
                noise_model.apply_two_qubit_noise(circuit, ctrl, tgt)
        
        return 0  # No measurements in transversal CNOT
    
    def emit_logical_h(
        self,
        circuit: stim.Circuit,
        qubits: List[int],
        noise_model: Optional["NoiseModel"] = None,
    ) -> int:
        """Emit transversal Hadamard: H on each qubit."""
        circuit.append("H", qubits)
        
        if noise_model is not None:
            for q in qubits:
                noise_model.apply_single_qubit_noise(circuit, q)
        
        return 0
    
    def emit_measurement(
        self,
        circuit: stim.Circuit,
        qubits: List[int],
        basis: str = 'Z',
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> List[int]:
        """Emit transversal measurement in specified basis."""
        if basis.upper() == 'X':
            circuit.append("H", qubits)
        
        circuit.append("M", qubits)
        
        # Return absolute measurement indices for this emission
        return [measurement_offset + i for i in range(len(qubits))]


class TeleportationStrategy(GateImplementationStrategy):
    """
    Gate teleportation strategy (placeholder for future implementation).
    
    Uses magic states and teleportation to implement logical gates.
    This is useful for:
    - Non-transversal gates (T, Toffoli)
    - Some topological codes
    
    TODO: Implement full teleportation protocol.
    """
    
    @property
    def strategy_type(self) -> GateStrategyType:
        return GateStrategyType.TELEPORTATION
    
    @property
    def name(self) -> str:
        return "Teleportation"
    
    def emit_logical_cnot(
        self,
        circuit: stim.Circuit,
        control_qubits: List[int],
        target_qubits: List[int],
        noise_model: Optional["NoiseModel"] = None,
    ) -> int:
        # For now, fall back to transversal
        # TODO: Implement proper teleportation-based CNOT
        return TransversalStrategy().emit_logical_cnot(
            circuit, control_qubits, target_qubits, noise_model
        )
    
    def emit_logical_h(
        self,
        circuit: stim.Circuit,
        qubits: List[int],
        noise_model: Optional["NoiseModel"] = None,
    ) -> int:
        return TransversalStrategy().emit_logical_h(circuit, qubits, noise_model)
    
    def emit_measurement(
        self,
        circuit: stim.Circuit,
        qubits: List[int],
        basis: str = 'Z',
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> List[int]:
        return TransversalStrategy().emit_measurement(
            circuit, qubits, basis, noise_model, measurement_offset
        )


class LatticeSurgeryStrategy(GateImplementationStrategy):
    """
    Lattice surgery strategy (placeholder for future implementation).
    
    Uses merge/split operations on surface code patches.
    This is the primary method for surface code quantum computation.
    
    TODO: Implement lattice surgery operations.
    """
    
    @property
    def strategy_type(self) -> GateStrategyType:
        return GateStrategyType.LATTICE_SURGERY
    
    @property
    def name(self) -> str:
        return "LatticeSurgery"
    
    def emit_logical_cnot(
        self,
        circuit: stim.Circuit,
        control_qubits: List[int],
        target_qubits: List[int],
        noise_model: Optional["NoiseModel"] = None,
    ) -> int:
        # Placeholder - would implement merge/split protocol
        raise NotImplementedError(
            "Lattice surgery CNOT not yet implemented. "
            "Use TransversalStrategy for now."
        )
    
    def emit_logical_h(
        self,
        circuit: stim.Circuit,
        qubits: List[int],
        noise_model: Optional["NoiseModel"] = None,
    ) -> int:
        raise NotImplementedError("Lattice surgery Hadamard not yet implemented.")
    
    def emit_measurement(
        self,
        circuit: stim.Circuit,
        qubits: List[int],
        basis: str = 'Z',
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> List[int]:
        # Measurement can still be transversal for many codes
        return TransversalStrategy().emit_measurement(
            circuit, qubits, basis, noise_model, measurement_offset
        )


# Default strategy instance
_DEFAULT_STRATEGY = TransversalStrategy()


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _ensure_mmap_fields(mmap: MeasurementMap) -> None:
    """Ensure a MeasurementMap has all expected dict fields."""
    if not hasattr(mmap, 'stabilizer_measurements') or mmap.stabilizer_measurements is None:
        mmap.stabilizer_measurements = {'X': {}, 'Z': {}}
    if not hasattr(mmap, 'pauli_frame') or mmap.pauli_frame is None:
        mmap.pauli_frame = {'X': {}, 'Z': {}}
    if not hasattr(mmap, 'flag_measurements') or mmap.flag_measurements is None:
        mmap.flag_measurements = {'X': {}, 'Z': {}}
    if not hasattr(mmap, 'verification_measurements') or mmap.verification_measurements is None:
        mmap.verification_measurements = {}
    if not hasattr(mmap, 'output_qubits') or mmap.output_qubits is None:
        mmap.output_qubits = {}

def _count_measurements(circuit: stim.Circuit) -> int:
    """Count measurement instructions in a Stim circuit."""
    count = 0
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            # Not inspecting repeat bodies here; invariants use deltas within emits
            continue
        name = inst.name.upper()
        if name in {"M", "MR", "MX", "MY", "MZ", "MRX", "MRY", "MRZ"}:
            count += len([t for t in inst.targets_copy() if t.is_qubit_target])
    return count


# =============================================================================
# HIERARCHICAL SYNDROME LAYOUT (for decoder integration)
# =============================================================================

@dataclass
class HierarchicalSyndromeLayout:
    """
    Tracks syndrome measurement layout for hierarchical decoding.
    
    This structure is passed to the decoder so it knows where to find
    syndrome measurements for each level and block in a multi-level
    concatenated code.
    
    Attributes
    ----------
    level_layouts : Dict[int, Dict[int, Dict[str, Any]]]
        Nested dict: level_idx -> block_idx -> measurement info
        Each block has: z_start, z_count, x_start, x_count, address,
        and optional flag_z_start/flag_z_count/flag_x_start/flag_x_count
    """
    level_layouts: Dict[int, Dict[int, Dict[str, Any]]] = field(default_factory=dict)
    
    def add_block(
        self,
        level: int,
        block_idx: int,
        z_start: int,
        z_count: int,
        x_start: int,
        x_count: int,
        address: tuple = (),
        flag_z_start: Optional[int] = None,
        flag_z_count: int = 0,
        flag_x_start: Optional[int] = None,
        flag_x_count: int = 0,
    ) -> None:
        """Record measurement layout for a block."""
        if level not in self.level_layouts:
            self.level_layouts[level] = {}
        self.level_layouts[level][block_idx] = {
            'z_start': z_start,
            'z_count': z_count,
            'x_start': x_start,
            'x_count': x_count,
            'address': address,
            'flag_z_start': flag_z_start,
            'flag_z_count': flag_z_count,
            'flag_x_start': flag_x_start,
            'flag_x_count': flag_x_count,
        }
    
    def to_dict(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """Convert to plain dict for metadata."""
        return dict(self.level_layouts)


# =============================================================================
# PARALLEL GADGET COMBINATOR
# =============================================================================

class ParallelGadget(Gadget):
    """
    Meta-gadget that applies a base gadget to multiple blocks in parallel.
    
    This combinator takes a single gadget and applies it independently to
    multiple code blocks, with operations batched for maximum parallelism.
    
    Parameters
    ----------
    base_gadget : Gadget
        The EC gadget to apply (e.g., SteaneECGadget, KnillECGadget).
    n_blocks : int, optional
        Number of blocks to operate on. If None, inferred from data_qubits.
    parallel_cnots : bool
        If True, schedule CNOTs across blocks for parallelism.
        
    Example
    -------
    >>> steane = SteaneECGadget(steane_code)
    >>> parallel = ParallelGadget(steane, n_blocks=7)
    >>> mmap = parallel.emit(circuit, data_qubits)  # All 7 blocks in parallel
    """
    
    def __init__(
        self,
        base_gadget: Gadget,
        n_blocks: Optional[int] = None,
        parallel_cnots: bool = True,
    ):
        self.base_gadget = base_gadget
        self._n_blocks = n_blocks
        self.parallel_cnots = parallel_cnots
    
    @property
    def name(self) -> str:
        return f"Parallel({self.base_gadget.name})"
    
    @property
    def requires_ancillas(self) -> bool:
        return self.base_gadget.requires_ancillas
    
    @property
    def ancillas_per_block(self) -> int:
        return self.base_gadget.ancillas_per_block
    
    @property
    def changes_data_identity(self) -> bool:
        """Delegate to base gadget."""
        return getattr(self.base_gadget, 'changes_data_identity', False)
    
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        ancilla_qubits: Optional[Dict[int, List[int]]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> MeasurementMap:
        """
        Emit base gadget operations on all blocks in parallel.
        
        Currently delegates to base gadget for each block sequentially,
        but groups operations for parallelism when possible.
        """
        n_blocks = self._n_blocks or len(data_qubits)
        
        combined_mmap = MeasurementMap(offset=measurement_offset)
        combined_mmap.stabilizer_measurements = {'X': {}, 'Z': {}}
        combined_mmap.pauli_frame = {'X': {}, 'Z': {}}  # Initialize pauli_frame
        combined_mmap.output_qubits = {}  # Track output qubits for teleportation
        combined_mmap.verification_measurements = {}  # For FT ancilla verification
        combined_mmap.shor_measurement_info = {}  # For Shor-style EC
        combined_mmap.measurement_type = "raw_ancilla"  # Default, may be updated
        current_meas = measurement_offset
        
        if self.parallel_cnots:
            # Emit all blocks together for parallelism
            # The base gadget handles batching internally
            mmap = self.base_gadget.emit(
                circuit=circuit,
                data_qubits=data_qubits,
                ancilla_qubits=ancilla_qubits,
                noise_model=noise_model,
                measurement_offset=current_meas,
            )
            _ensure_mmap_fields(mmap)
            combined_mmap = mmap
            # Ensure shor_measurement_info is propagated
            if hasattr(mmap, 'shor_measurement_info') and mmap.shor_measurement_info:
                combined_mmap.shor_measurement_info = mmap.shor_measurement_info
            if hasattr(mmap, 'measurement_type'):
                combined_mmap.measurement_type = mmap.measurement_type
            current_meas += mmap.total_measurements  # Update current_meas for correct total
        else:
            # Sequential emission (for debugging or when parallelism isn't wanted)
            for block_idx in sorted(data_qubits.keys()):
                single_block = {block_idx: data_qubits[block_idx]}
                single_ancilla = None
                if ancilla_qubits and block_idx in ancilla_qubits:
                    single_ancilla = {block_idx: ancilla_qubits[block_idx]}
                
                mmap = self.base_gadget.emit(
                    circuit=circuit,
                    data_qubits=single_block,
                    ancilla_qubits=single_ancilla,
                    noise_model=noise_model,
                    measurement_offset=current_meas,
                )
                _ensure_mmap_fields(mmap)
                
                # Merge stabilizer measurements
                for stype in ['X', 'Z']:
                    if stype in mmap.stabilizer_measurements:
                        combined_mmap.stabilizer_measurements[stype].update(
                            mmap.stabilizer_measurements[stype]
                        )
                
                # Merge pauli_frame for teleportation EC
                if hasattr(mmap, 'pauli_frame') and mmap.pauli_frame:
                    for ftype in ['X', 'Z']:
                        if ftype in mmap.pauli_frame:
                            combined_mmap.pauli_frame[ftype].update(mmap.pauli_frame[ftype])
                
                # Merge output_qubits for teleportation EC
                if hasattr(mmap, 'output_qubits') and mmap.output_qubits:
                    combined_mmap.output_qubits.update(mmap.output_qubits)
                
                # Merge verification_measurements for FT ancilla prep
                if hasattr(mmap, 'verification_measurements') and mmap.verification_measurements:
                    combined_mmap.verification_measurements.update(mmap.verification_measurements)
                
                # Merge shor_measurement_info for Shor-style EC
                if hasattr(mmap, 'shor_measurement_info') and mmap.shor_measurement_info:
                    combined_mmap.shor_measurement_info.update(mmap.shor_measurement_info)
                # Inherit measurement_type from base gadget
                if hasattr(mmap, 'measurement_type'):
                    combined_mmap.measurement_type = mmap.measurement_type
                
                current_meas += mmap.total_measurements
        
        combined_mmap.total_measurements = current_meas - measurement_offset
        return combined_mmap
    
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        return self.base_gadget.get_syndrome_schedule()
    
    def get_logical_map(self, inner_code: Any, outer_code: Any) -> LogicalMeasurementMap:
        return self.base_gadget.get_logical_map(inner_code, outer_code)


# =============================================================================
# CHAINED GADGET COMBINATOR
# =============================================================================

class ChainedGadget(Gadget):
    """
    Meta-gadget that applies multiple gadgets in sequence.
    
    Operations are emitted in order, with TICK separating stages.
    Useful for multi-stage EC or combining different EC methods.
    
    Parameters
    ----------
    gadgets : List[Gadget]
        Gadgets to apply in sequence.
    tick_between : bool
        If True, insert TICK between gadgets.
        
    Example
    -------
    >>> # First do inner EC, then outer EC
    >>> chain = ChainedGadget([inner_ec, outer_ec])
    >>> mmap = chain.emit(circuit, data_qubits)
    """
    
    def __init__(
        self,
        gadgets: List[Gadget],
        tick_between: bool = True,
    ):
        self.gadgets = gadgets
        self.tick_between = tick_between
    
    @property
    def name(self) -> str:
        names = [g.name for g in self.gadgets]
        return f"Chain({' -> '.join(names)})"
    
    @property
    def requires_ancillas(self) -> bool:
        return any(g.requires_ancillas for g in self.gadgets)
    
    @property
    def ancillas_per_block(self) -> int:
        return max((g.ancillas_per_block for g in self.gadgets), default=0)
    
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        ancilla_qubits: Optional[Dict[int, List[int]]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> MeasurementMap:
        """Emit all gadgets in sequence."""
        combined_mmap = MeasurementMap(offset=measurement_offset)
        combined_mmap.stabilizer_measurements = {'X': {}, 'Z': {}}
        combined_mmap.pauli_frame = {'X': {}, 'Z': {}}
        combined_mmap.output_qubits = {}
        combined_mmap.verification_measurements = {}
        combined_mmap.flag_measurements = {'X': {}, 'Z': {}}
        current_meas = measurement_offset
        
        for i, gadget in enumerate(self.gadgets):
            mmap = gadget.emit(
                circuit=circuit,
                data_qubits=data_qubits,
                ancilla_qubits=ancilla_qubits,
                noise_model=noise_model,
                measurement_offset=current_meas,
            )
            _ensure_mmap_fields(mmap)
            
            # Merge with stage prefix
            for stype in ['X', 'Z']:
                if stype in mmap.stabilizer_measurements:
                    for block_id, indices in mmap.stabilizer_measurements[stype].items():
                        key = (f"stage{i}", block_id)
                        combined_mmap.stabilizer_measurements[stype][key] = indices

            # Merge pauli frame
            if hasattr(mmap, 'pauli_frame') and mmap.pauli_frame:
                for ftype in ['X', 'Z']:
                    if ftype in mmap.pauli_frame:
                        for block_id, indices in mmap.pauli_frame[ftype].items():
                            key = (f"stage{i}", block_id)
                            combined_mmap.pauli_frame[ftype][key] = indices

            # Merge output qubits
            if hasattr(mmap, 'output_qubits') and mmap.output_qubits:
                for block_id, qubits in mmap.output_qubits.items():
                    key = (f"stage{i}", block_id)
                    combined_mmap.output_qubits[key] = qubits

            # Merge verification measurements
            if hasattr(mmap, 'verification_measurements') and mmap.verification_measurements:
                for block_id, indices in mmap.verification_measurements.items():
                    key = (f"stage{i}", block_id)
                    combined_mmap.verification_measurements[key] = indices

            # Merge flag measurements
            if hasattr(mmap, 'flag_measurements') and mmap.flag_measurements:
                for ftype in ['X', 'Z']:
                    if ftype in mmap.flag_measurements:
                        for block_id, indices in mmap.flag_measurements[ftype].items():
                            key = (f"stage{i}", block_id)
                            combined_mmap.flag_measurements[ftype][key] = indices
            
            current_meas += mmap.total_measurements
            
            if self.tick_between and i < len(self.gadgets) - 1:
                circuit.append("TICK")
        
        combined_mmap.total_measurements = current_meas - measurement_offset
        return combined_mmap
    
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        # Combine schedules from all gadgets
        combined = SyndromeSchedule()
        for i, gadget in enumerate(self.gadgets):
            sched = gadget.get_syndrome_schedule()
            for stype in sched.stabilizer_types:
                prefixed = f"S{i}_{stype}"
                combined.stabilizer_types.append(prefixed)
                combined.rounds_per_type[prefixed] = sched.rounds_per_type.get(stype, 1)
        return combined
    
    def get_logical_map(self, inner_code: Any, outer_code: Any) -> LogicalMeasurementMap:
        # Use first gadget's logical map as default
        if self.gadgets:
            return self.gadgets[0].get_logical_map(inner_code, outer_code)
        return LogicalMeasurementMap()


# =============================================================================
# RECURSIVE GADGET COMBINATOR
# =============================================================================

class RecursiveGadget(Gadget):
    """
    Meta-gadget for hierarchical EC on concatenated codes.
    
    Applies gadgets level-by-level from innermost to outermost,
    enabling different EC methods at different concatenation levels.
    
    The recursive structure is:
    1. Apply gadget at innermost (leaf) level to all leaf blocks
    2. Move up one level, apply that level's gadget
    3. Continue to root
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The concatenated code structure.
    level_gadgets : Dict[int, Gadget]
        Mapping from level index to gadget for that level.
        Level 0 is outermost, level depth-1 is innermost.
    default_gadget_factory : Callable, optional
        Factory to create gadgets for levels not in level_gadgets.
        
    Example
    -------
    >>> # Different EC at each level
    >>> recursive = RecursiveGadget(
    ...     code=concatenated_code,
    ...     level_gadgets={
    ...         0: SteaneECGadget(outer_code),   # Outer: Steane
    ...         1: KnillECGadget(middle_code),    # Middle: Knill
    ...         2: SteaneECGadget(inner_code),    # Inner: Steane
    ...     }
    ... )
    """
    
    def __init__(
        self,
        code: 'MultiLevelConcatenatedCode',
        level_gadgets: Dict[int, Gadget],
        default_gadget_factory: Optional[Callable[['CSSCode'], Gadget]] = None,
        parallelize_levels: Optional[set] = None,
    ):
        self.code = code
        self.depth = code.depth
        self.level_codes = code.level_codes
        # Opt-in parallelization: when None, treat as empty set (no forced parallel)
        self.parallelize_levels = parallelize_levels or set()
        
        # Build gadgets for each level
        self.level_gadgets = {}
        for level_idx in range(self.depth):
            if level_idx in level_gadgets:
                self.level_gadgets[level_idx] = level_gadgets[level_idx]
            elif default_gadget_factory:
                self.level_gadgets[level_idx] = default_gadget_factory(
                    self.level_codes[level_idx]
                )
            # else: level has no gadget (skip it)
        
        # Precompute block structure
        self._compute_block_structure()
        
        # Track syndrome layout for decoder integration
        self._last_syndrome_layout: Optional[HierarchicalSyndromeLayout] = None
    
    def _compute_block_structure(self) -> None:
        """Compute hierarchical block counts."""
        self.blocks_per_level = [1]
        cumulative = 1
        for level_idx in range(self.depth - 1):
            cumulative *= self.level_codes[level_idx].n
            self.blocks_per_level.append(cumulative)
        self.n_leaf_blocks = self.blocks_per_level[-1]
    
    @property
    def name(self) -> str:
        gadget_names = [
            f"L{i}:{g.name}" for i, g in sorted(self.level_gadgets.items())
        ]
        return f"Recursive({', '.join(gadget_names)})"
    
    @property
    def requires_ancillas(self) -> bool:
        return any(g.requires_ancillas for g in self.level_gadgets.values())
    
    @property
    def ancillas_per_block(self) -> int:
        total = 0
        for level_idx, gadget in self.level_gadgets.items():
            n_blocks = self.blocks_per_level[level_idx]
            total += gadget.ancillas_per_block * n_blocks
        return total
    
    @property
    def syndrome_layout(self) -> Optional[HierarchicalSyndromeLayout]:
        """Return the syndrome layout from the last emit() call."""
        return self._last_syndrome_layout

    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        ancilla_qubits: Optional[Dict[int, List[int]]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> MeasurementMap:
        """
        Emit recursive EC from innermost to outermost level.
        
        Also updates self._last_syndrome_layout for decoder integration.
        
        IMPORTANT: For gadgets that change data identity (like TeleportationEC),
        the data_qubits are updated after each level to track where the data
        actually lives. This is essential for correct circuit construction.
        """
        combined_mmap = MeasurementMap(offset=measurement_offset)
        combined_mmap.stabilizer_measurements = {'X': {}, 'Z': {}}
        combined_mmap.output_qubits = {}  # Track final data qubit locations
        combined_mmap.pauli_frame = {'X': {}, 'Z': {}}  # Initialize pauli_frame
        combined_mmap.verification_measurements = {}  # For FT ancilla verification
        combined_mmap.flag_measurements = {'X': {}, 'Z': {}}
        current_meas = measurement_offset
        
        # Initialize syndrome layout tracking
        self._last_syndrome_layout = HierarchicalSyndromeLayout()
        
        # Track current data qubit locations (may change after teleportation)
        current_data_qubits = dict(data_qubits)
        
        # Process from innermost to outermost
        for level_idx in reversed(range(self.depth)):
            gadget = self.level_gadgets.get(level_idx)
            if gadget is None:
                continue
            
            # Get blocks for this level using CURRENT data locations
            level_blocks = self._get_level_blocks(level_idx, current_data_qubits)
            if not level_blocks:
                continue
            
            # Allocate ancillas based on current data locations (respect caller-provided mapping)
            level_ancillas = self._allocate_ancillas(
                level_idx,
                current_data_qubits,
                ancilla_qubits,
            )
            
            # Emit gadget (wrapped in Parallel for this level's blocks)
            parallel = level_idx in self.parallelize_levels
            parallel_gadget = ParallelGadget(
                gadget,
                n_blocks=len(level_blocks),
                parallel_cnots=parallel,
            )
            mmap = parallel_gadget.emit(
                circuit=circuit,
                data_qubits=level_blocks,
                ancilla_qubits=level_ancillas,
                noise_model=noise_model,
                measurement_offset=current_meas,
            )
            _ensure_mmap_fields(mmap)
            
            # Validate output mapping when gadget changes data identity
            if hasattr(gadget, 'changes_data_identity') and gadget.changes_data_identity:
                if hasattr(mmap, 'output_qubits') and mmap.output_qubits:
                    missing = set(level_blocks.keys()) - set(mmap.output_qubits.keys())
                    if missing:
                        raise ValueError(f"Output qubits missing for blocks {sorted(missing)} at level {level_idx}")
                else:
                    raise ValueError(f"Gadget at level {level_idx} changes data identity but provided no output_qubits")
            
            # CRITICAL: Update data qubit locations if gadget changes data identity
            # This happens for TeleportationEC where data moves to ancilla qubits
            if hasattr(gadget, 'changes_data_identity') and gadget.changes_data_identity:
                if hasattr(mmap, 'output_qubits') and mmap.output_qubits:
                    # Update the leaf-level data qubits based on output_qubits
                    current_data_qubits = self._update_data_qubits_from_output(
                        level_idx, current_data_qubits, mmap.output_qubits
                    )
            
            # Track syndrome layout for decoder
            self._record_syndrome_layout(level_idx, level_blocks, mmap, current_meas)
            
            # Merge with level prefix
            for stype in ['X', 'Z']:
                if stype in mmap.stabilizer_measurements:
                    for block_id, indices in mmap.stabilizer_measurements[stype].items():
                        key = (level_idx, block_id)
                        combined_mmap.stabilizer_measurements[stype][key] = indices
            
            # Aggregate pauli_frame for teleportation EC
            if hasattr(mmap, 'pauli_frame') and mmap.pauli_frame:
                for ftype in ['X', 'Z']:
                    if ftype in mmap.pauli_frame:
                        for block_id, indices in mmap.pauli_frame[ftype].items():
                            key = (level_idx, block_id)
                            combined_mmap.pauli_frame[ftype][key] = indices
            
            # Aggregate verification_measurements for FT ancilla prep
            if hasattr(mmap, 'verification_measurements') and mmap.verification_measurements:
                for block_id, indices in mmap.verification_measurements.items():
                    key = (level_idx, block_id)
                    combined_mmap.verification_measurements[key] = indices

            # Aggregate flag measurements
            if hasattr(mmap, 'flag_measurements') and mmap.flag_measurements:
                for ftype in ['X', 'Z']:
                    if ftype in mmap.flag_measurements:
                        for block_id, indices in mmap.flag_measurements[ftype].items():
                            key = (level_idx, block_id)
                            combined_mmap.flag_measurements[ftype][key] = indices
            
            # Aggregate shor_measurement_info for Shor-style EC
            # Keys are remapped from (stype, block_id) to (stype, (level_idx, block_id))
            if hasattr(mmap, 'shor_measurement_info') and mmap.shor_measurement_info:
                if not hasattr(combined_mmap, 'shor_measurement_info'):
                    combined_mmap.shor_measurement_info = {}
                for orig_key, info in mmap.shor_measurement_info.items():
                    # orig_key is (stype, block_id), transform to (stype, (level_idx, block_id))
                    stype, block_id = orig_key
                    new_key = (stype, (level_idx, block_id))
                    combined_mmap.shor_measurement_info[new_key] = info
            
            # Inherit measurement_type from gadget (e.g., "shor_redundant")
            if hasattr(mmap, 'measurement_type') and mmap.measurement_type:
                if not hasattr(combined_mmap, 'measurement_type') or combined_mmap.measurement_type == "raw_ancilla":
                    combined_mmap.measurement_type = mmap.measurement_type
            
            current_meas += mmap.total_measurements
            circuit.append("TICK")
        
        # Set final output_qubits to current data locations after all teleportations
        # This tells the caller where the data actually is now
        combined_mmap.output_qubits = dict(current_data_qubits)
        
        combined_mmap.total_measurements = current_meas - measurement_offset
        return combined_mmap
    
    def _record_syndrome_layout(
        self,
        level_idx: int,
        level_blocks: Dict[int, List[int]],
        mmap: MeasurementMap,
        meas_start: int,
    ) -> None:
        """Record syndrome measurement positions for decoder."""
        for block_idx in sorted(level_blocks.keys()):
            # Get block address
            address = self._block_idx_to_address(level_idx, block_idx)
            
            # Extract Z and X measurement info from mmap
            z_indices = []
            x_indices = []
            flag_z_indices = []
            flag_x_indices = []
            if mmap.stabilizer_measurements:
                if 'Z' in mmap.stabilizer_measurements and block_idx in mmap.stabilizer_measurements['Z']:
                    z_indices = mmap.stabilizer_measurements['Z'][block_idx]
                if 'X' in mmap.stabilizer_measurements and block_idx in mmap.stabilizer_measurements['X']:
                    x_indices = mmap.stabilizer_measurements['X'][block_idx]
            if hasattr(mmap, 'flag_measurements') and mmap.flag_measurements:
                if 'Z' in mmap.flag_measurements and block_idx in mmap.flag_measurements['Z']:
                    for vals in mmap.flag_measurements['Z'][block_idx].values():
                        flag_z_indices.extend(vals)
                if 'X' in mmap.flag_measurements and block_idx in mmap.flag_measurements['X']:
                    for vals in mmap.flag_measurements['X'][block_idx].values():
                        flag_x_indices.extend(vals)
            
            self._last_syndrome_layout.add_block(
                level=level_idx,
                block_idx=block_idx,
                z_start=min(z_indices) if z_indices else meas_start,
                z_count=len(z_indices),
                x_start=min(x_indices) if x_indices else meas_start,
                x_count=len(x_indices),
                address=address,
                flag_z_start=min(flag_z_indices) if flag_z_indices else None,
                flag_z_count=len(flag_z_indices),
                flag_x_start=min(flag_x_indices) if flag_x_indices else None,
                flag_x_count=len(flag_x_indices),
            )
    
    def _block_idx_to_address(self, level_idx: int, block_idx: int) -> Tuple[int, ...]:
        """Convert flat block index to hierarchical address tuple."""
        if level_idx == 0:
            return ()  # Root has empty address
        
        address = []
        remaining = block_idx
        for l in range(level_idx, 0, -1):
            parent_n = self.level_codes[l - 1].n
            address.insert(0, remaining % parent_n)
            remaining //= parent_n
        return tuple(address)
    
    def _get_level_blocks(
        self,
        level_idx: int,
        data_qubits: Dict[int, List[int]],
    ) -> Dict[int, List[int]]:
        """Get data qubit blocks for a specific level."""
        if level_idx == self.depth - 1:
            return data_qubits
        
        # Aggregate leaf blocks into parent blocks
        level_blocks = {}
        n_blocks = self.blocks_per_level[level_idx]
        
        leaves_per_block = 1
        for l in range(level_idx + 1, self.depth):
            leaves_per_block *= self.level_codes[l].n
        
        for parent_idx in range(n_blocks):
            start_leaf = parent_idx * leaves_per_block
            end_leaf = start_leaf + leaves_per_block
            
            combined = []
            for leaf_idx in range(start_leaf, end_leaf):
                if leaf_idx in data_qubits:
                    combined.extend(data_qubits[leaf_idx])
            
            if combined:
                level_blocks[parent_idx] = combined
        
        return level_blocks
    
    def _update_data_qubits_from_output(
        self,
        level_idx: int,
        current_data_qubits: Dict[int, List[int]],
        output_qubits: Dict[int, List[int]],
    ) -> Dict[int, List[int]]:
        """
        Update data qubit locations after a gadget that changes data identity.
        
        For TeleportationEC at the inner level, the data moves from the original
        qubits to the ancilla2 qubits. This method updates the tracking dict.
        
        Parameters
        ----------
        level_idx : int
            Level at which the gadget was applied.
        current_data_qubits : Dict[int, List[int]]
            Current mapping of leaf block IDs to qubit lists.
        output_qubits : Dict[int, List[int]]
            New qubit locations from the gadget's MeasurementMap.
            
        Returns
        -------
        Dict[int, List[int]]
            Updated mapping with new qubit locations.
        """
        if level_idx == self.depth - 1:
            # At innermost level, output_qubits directly replaces data_qubits
            new_data_qubits = dict(current_data_qubits)
            for block_id, new_qubits in output_qubits.items():
                new_data_qubits[block_id] = new_qubits
            return new_data_qubits
        else:
            # At higher levels, need to map output_qubits to leaf blocks
            # For now, this handles the case where all data at a level is teleported
            # together (like outer teleportation on all 49 qubits)
            
            # Compute how many leaf blocks are covered by each block at this level
            leaves_per_block = 1
            for l in range(level_idx + 1, self.depth):
                leaves_per_block *= self.level_codes[l].n
            
            new_data_qubits = dict(current_data_qubits)
            n_inner = self.level_codes[self.depth - 1].n  # Qubits per leaf block
            
            for parent_idx, new_qubits in output_qubits.items():
                # Distribute the output qubits among the leaf blocks under this parent
                start_leaf = parent_idx * leaves_per_block
                
                for local_leaf in range(leaves_per_block):
                    leaf_idx = start_leaf + local_leaf
                    qubit_start = local_leaf * n_inner
                    qubit_end = qubit_start + n_inner
                    if qubit_end <= len(new_qubits):
                        new_data_qubits[leaf_idx] = new_qubits[qubit_start:qubit_end]
            
            return new_data_qubits
    
    def _allocate_ancillas(
        self,
        level_idx: int,
        data_qubits: Dict[int, List[int]],
        provided_ancillas: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[int, List[int]]:
        """Allocate ancillas for a level, preferring caller-provided mapping."""
        gadget = self.level_gadgets.get(level_idx)
        if gadget is None or not gadget.requires_ancillas:
            return {}

        level_blocks = self._get_level_blocks(level_idx, data_qubits)

        # Use caller-supplied ancillas when they match block ids
        if provided_ancillas:
            available_blocks = set(level_blocks.keys()) & set(provided_ancillas.keys())
            if available_blocks:
                return {bid: provided_ancillas[bid] for bid in sorted(available_blocks)}

        # Otherwise allocate after the highest existing qubit id
        max_data = max((max(qs) for qs in data_qubits.values() if qs), default=-1)
        ancilla_start = max_data + 1

        ancillas = {}
        for block_id in sorted(level_blocks.keys()):
            n_anc = gadget.ancillas_per_block
            ancillas[block_id] = list(range(ancilla_start, ancilla_start + n_anc))
            ancilla_start += n_anc

        return ancillas
    
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        combined = SyndromeSchedule()
        for level_idx in reversed(range(self.depth)):
            gadget = self.level_gadgets.get(level_idx)
            if gadget is None:
                continue
            sched = gadget.get_syndrome_schedule()
            for stype in sched.stabilizer_types:
                prefixed = f"L{level_idx}_{stype}"
                combined.stabilizer_types.append(prefixed)
                combined.rounds_per_type[prefixed] = sched.rounds_per_type.get(stype, 1)
        return combined
    
    def get_logical_map(self, inner_code: Any, outer_code: Any) -> LogicalMeasurementMap:
        if 0 in self.level_gadgets:
            return self.level_gadgets[0].get_logical_map(inner_code, outer_code)
        return LogicalMeasurementMap()


# =============================================================================
# CONDITIONAL GADGET COMBINATOR
# =============================================================================

class ConditionalGadget(Gadget):
    """
    Meta-gadget that conditionally applies a gadget based on a predicate.
    
    Useful for adaptive EC strategies where you only do EC when
    syndrome indicates errors, or for implementing flag-based protocols.
    
    Note: In Stim, true classical conditioning isn't directly supported,
    so this primarily serves for organizing circuit structure and
    documentation of the intended conditional logic.
    
    Parameters
    ----------
    gadget : Gadget
        The gadget to conditionally apply.
    condition_name : str
        Name describing the condition (for documentation).
    always_emit : bool
        If True, always emit the gadget (condition is semantic only).
    """
    
    def __init__(
        self,
        gadget: Gadget,
        condition_name: str = "syndrome_triggered",
        always_emit: bool = True,
    ):
        self.gadget = gadget
        self.condition_name = condition_name
        self.always_emit = always_emit
    
    @property
    def name(self) -> str:
        return f"Conditional({self.gadget.name}, if={self.condition_name})"
    
    @property
    def requires_ancillas(self) -> bool:
        return self.gadget.requires_ancillas
    
    @property
    def ancillas_per_block(self) -> int:
        return self.gadget.ancillas_per_block
    
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        ancilla_qubits: Optional[Dict[int, List[int]]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> MeasurementMap:
        """Emit gadget (unconditionally in Stim, but marked as conditional)."""
        if self.always_emit:
            return self.gadget.emit(
                circuit, data_qubits, ancilla_qubits, noise_model, measurement_offset
            )
        else:
            # Return empty measurement map (gadget not emitted)
            return MeasurementMap(offset=measurement_offset)
    
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        return self.gadget.get_syndrome_schedule()
    
    def get_logical_map(self, inner_code: Any, outer_code: Any) -> LogicalMeasurementMap:
        return self.gadget.get_logical_map(inner_code, outer_code)


# =============================================================================
# FLAGGED SYNDROME GADGET COMBINATOR
# =============================================================================

@dataclass
class FlagQubitConfig:
    """
    Configuration for flag qubit placement in syndrome extraction.
    
    Flag qubits detect "hook errors" - single faults that would propagate
    to weight-2+ errors on data qubits. For weight-w stabilizers:
    - CNOTs in sequence can propagate X errors from ancilla to data
    - Flag qubit placed mid-sequence detects if this propagation occurred
    
    Attributes
    ----------
    stabilizer_type : str
        'X' or 'Z' - which stabilizer this flag is for
    stabilizer_idx : int
        Index of the stabilizer generator
    flag_position : int
        Position in the CNOT sequence where flag is inserted (0-indexed)
        For weight-4 stabilizer with CNOTs at positions 0,1,2,3:
        flag_position=2 means flag inserted between CNOTs 1 and 2
    data_qubits_before : List[int]
        Data qubit indices that have been coupled BEFORE the flag
    data_qubits_after : List[int]
        Data qubit indices that will be coupled AFTER the flag
    """
    stabilizer_type: str
    stabilizer_idx: int
    flag_position: int
    data_qubits_before: List[int]
    data_qubits_after: List[int]


class FlaggedSyndromeGadget(Gadget):
    """
    Meta-gadget that adds flag qubits to syndrome extraction for hook error detection.
    
    Flag fault-tolerance (Chamberland et al., Quantum 2018) uses auxiliary "flag"
    qubits to detect when a single fault would cause a weight-2+ error. This allows
    the decoder to distinguish:
    
    - Flag=0, syndrome!=0: Weight-1 error occurred (standard correction)
    - Flag=1, syndrome!=0: Hook error detected (need flag-conditioned correction)
    - Flag=1, syndrome=0: Ancilla error caught by flag (ignore/repeat)
    
    For CSS codes, weight-w stabilizers with w≥4 need flags to prevent hook errors
    from CNOT gates propagating X errors: ancilla X → multiple data X errors.
    
    This gadget wraps a base syndrome gadget and inserts flag qubits at appropriate
    positions in the CNOT sequences. It is GENERAL and works for any CSS code.
    
    Parameters
    ----------
    base_gadget : Gadget
        The underlying syndrome extraction gadget to wrap
    code : CSSCode
        The CSS code (provides Hz, Hx for stabilizer weights)
    flag_all_weight4_plus : bool
        If True, automatically add flags to all weight-4+ stabilizers
    custom_flag_configs : List[FlagQubitConfig], optional
        Custom flag configurations (overrides automatic placement)
    measure_flags_after_syndrome : bool
        If True, measure flags after syndrome measurements (allows reuse)
        If False, measure flags inline (more deterministic timing)
    
    Example
    -------
    >>> from qectostim.experiments.gadgets import FlaggedSyndromeGadget, TransversalSyndromeGadget
    >>> from qectostim.codes.small import SteaneCode
    >>> 
    >>> code = SteaneCode()
    >>> base_gadget = TransversalSyndromeGadget(code)
    >>> flagged = FlaggedSyndromeGadget(base_gadget, code, flag_all_weight4_plus=True)
    >>> 
    >>> # Now flagged.emit() will include flag qubits and track flag measurements
    >>> mmap = flagged.emit(circuit, data_qubits, ancilla_qubits, noise_model, 0)
    >>> # mmap.flag_measurements['Z'][block_id][stab_idx] = [flag_meas_indices]
    
    References
    ----------
    - Chamberland, Beverland, "Flag fault-tolerant error correction with 
      arbitrary distance codes", Quantum 2 (2018)
    - Chamberland, Cross, "Fault-tolerant magic state preparation with flag qubits"
    - Reichardt, "Fault-tolerant quantum error correction for Steane's seven-qubit code"
    """
    
    def __init__(
        self,
        base_gadget: Gadget,
        code: Optional[Any] = None,
        flag_all_weight4_plus: bool = True,
        custom_flag_configs: Optional[List[FlagQubitConfig]] = None,
        measure_flags_after_syndrome: bool = False,
        ancillas_per_block_override: Optional[int] = None,
    ):
        self.base_gadget = base_gadget
        self.code = code
        self.flag_all_weight4_plus = flag_all_weight4_plus
        self.custom_flag_configs = custom_flag_configs or []
        self.measure_flags_after_syndrome = measure_flags_after_syndrome
        self._ancillas_per_block_override = ancillas_per_block_override
        
        # Build flag configurations
        self._flag_configs: List[FlagQubitConfig] = []
        if custom_flag_configs:
            self._flag_configs = list(custom_flag_configs)
        elif flag_all_weight4_plus and code is not None:
            self._flag_configs = self._generate_flag_configs_for_code(code)
        
        # Cache code matrices
        self._hz = None
        self._hx = None
        if code is not None:
            if hasattr(code, 'hz'):
                hz = code.hz
                self._hz = np.atleast_2d(np.array(hz() if callable(hz) else hz, dtype=int))
            if hasattr(code, 'hx'):
                hx = code.hx
                self._hx = np.atleast_2d(np.array(hx() if callable(hx) else hx, dtype=int))
    
    def _generate_flag_configs_for_code(self, code: Any) -> List[FlagQubitConfig]:
        """
        Generate flag configurations for all weight-4+ stabilizers.
        
        For each stabilizer with weight w ≥ 4, we place a flag qubit that:
        - Is prepared in |+⟩
        - Has CNOT(flag, ancilla) before the first half of data CNOTs
        - Has CNOT(ancilla, flag) after the first half of data CNOTs  
        - Is measured in X basis (via H then Z measurement)
        
        If a hook error occurs (X on ancilla propagates through CNOTs),
        the flag will flip. This lets the decoder know to look for
        weight-2 error patterns instead of weight-1.
        """
        configs = []
        
        # Get parity check matrices
        hz = hx = None
        if hasattr(code, 'hz'):
            hz_raw = code.hz
            hz = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        if hasattr(code, 'hx'):
            hx_raw = code.hx
            hx = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
        
        # Process Z stabilizers (detect X errors)
        if hz is not None:
            for stab_idx in range(hz.shape[0]):
                support = list(np.where(hz[stab_idx] == 1)[0])
                weight = len(support)
                
                if weight >= 4:
                    # Place flag in middle of CNOT sequence
                    # For weight-4: flag after position 1 (between qubits 1 and 2)
                    # For weight-6: flag after position 2 (between qubits 2 and 3)
                    flag_pos = weight // 2
                    configs.append(FlagQubitConfig(
                        stabilizer_type='Z',
                        stabilizer_idx=stab_idx,
                        flag_position=flag_pos,
                        data_qubits_before=support[:flag_pos],
                        data_qubits_after=support[flag_pos:],
                    ))
        
        # Process X stabilizers (detect Z errors)
        if hx is not None:
            for stab_idx in range(hx.shape[0]):
                support = list(np.where(hx[stab_idx] == 1)[0])
                weight = len(support)
                
                if weight >= 4:
                    flag_pos = weight // 2
                    configs.append(FlagQubitConfig(
                        stabilizer_type='X',
                        stabilizer_idx=stab_idx,
                        flag_position=flag_pos,
                        data_qubits_before=support[:flag_pos],
                        data_qubits_after=support[flag_pos:],
                    ))
        
        return configs
    
    @property
    def name(self) -> str:
        n_flags = len(self._flag_configs)
        return f"FlaggedSyndrome({self.base_gadget.name}, {n_flags} flags)"
    
    @property
    def requires_ancillas(self) -> bool:
        return True  # Need ancillas for syndrome + flags
    
    @property
    def ancillas_per_block(self) -> int:
        """Ancillas for all stabilizers plus flags (fall back to base if larger)."""
        base_ancillas = self.base_gadget.ancillas_per_block if hasattr(self.base_gadget, 'ancillas_per_block') else 0
        n_flags = len(self._flag_configs)
        n_z = self._hz.shape[0] if self._hz is not None else 0
        n_x = self._hx.shape[0] if self._hx is not None else 0
        computed = max(base_ancillas, n_z + n_x + n_flags)
        if self._ancillas_per_block_override is not None:
            return self._ancillas_per_block_override
        return computed
    
    @property
    def n_flag_qubits(self) -> int:
        """Number of flag qubits per block."""
        return len(self._flag_configs)
    
    @property
    def changes_data_identity(self) -> bool:
        return getattr(self.base_gadget, 'changes_data_identity', False)
    
    def get_flag_configs(self) -> List[FlagQubitConfig]:
        """Get the flag qubit configurations."""
        return list(self._flag_configs)
    
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        ancilla_qubits: Optional[Dict[int, List[int]]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> MeasurementMap:
        """
        Emit flagged syndrome extraction circuit.
        
        For each block, emits:
        1. Flag qubit preparation (|+⟩ state)
        2. First CNOT(flag → syndrome_ancilla) for each flagged stabilizer
        3. First half of data-ancilla CNOTs (before flag position)
        4. Second CNOT(syndrome_ancilla → flag) for each flagged stabilizer
        5. Second half of data-ancilla CNOTs (after flag position)
        6. Syndrome measurements
        7. Flag measurements (H then M)
        
        The flag qubit detects if an X error occurred on the syndrome ancilla
        AFTER the first data CNOTs but BEFORE the second data CNOTs.
        """
        pre_meas = _count_measurements(circuit)
        mmap = MeasurementMap(offset=measurement_offset)
        mmap.stabilizer_measurements = {'X': {}, 'Z': {}}
        mmap.flag_measurements = {'X': {}, 'Z': {}}
        mmap.verification_measurements = {}
        current_meas_idx = measurement_offset
        
        # If no flag configs, just delegate to base gadget
        if not self._flag_configs:
            base_mmap = self.base_gadget.emit(
                circuit, data_qubits, ancilla_qubits, noise_model, measurement_offset
            )
            _ensure_mmap_fields(base_mmap)
            # Copy base measurements
            mmap.stabilizer_measurements = base_mmap.stabilizer_measurements
            mmap.flag_measurements = getattr(base_mmap, 'flag_measurements', {'X': {}, 'Z': {}})
            mmap.total_measurements = base_mmap.total_measurements
            return mmap
        
        # Get code dimensions
        n_z_checks = self._hz.shape[0] if self._hz is not None else 0
        n_x_checks = self._hx.shape[0] if self._hx is not None else 0
        n_flags = len(self._flag_configs)
        base_ancillas = self.base_gadget.ancillas_per_block if hasattr(self.base_gadget, 'ancillas_per_block') else 0
        total_required = max(base_ancillas, n_z_checks + n_x_checks + n_flags)

        def _apply_single_noise(qs: List[int]) -> None:
            if not noise_model:
                return
            if hasattr(noise_model, 'apply_single_qubit_noise'):
                for q in qs:
                    noise_model.apply_single_qubit_noise(circuit, q)

        def _apply_two_noise(a: int, b: int) -> None:
            if not noise_model:
                return
            if hasattr(noise_model, 'apply_two_qubit_noise'):
                noise_model.apply_two_qubit_noise(circuit, a, b)

        def _apply_meas_noise(qs: List[int]) -> None:
            """Apply measurement noise if the model provides a hook."""
            if not noise_model or not qs:
                return
            if hasattr(noise_model, 'apply_measurement_noise'):
                res = noise_model.apply_measurement_noise(qs)
                # Some models return instructions instead of mutating
                if res:
                    for inst in res:
                        circuit.append(inst)
            elif hasattr(noise_model, 'apply_single_qubit_noise'):
                # Fallback: reuse single-qubit noise as a crude readout model
                for q in qs:
                    noise_model.apply_single_qubit_noise(circuit, q)
        
        for block_id in sorted(data_qubits.keys()):
            data_qs = data_qubits[block_id]
            provided_block_anc = None
            if ancilla_qubits and block_id in ancilla_qubits:
                provided_block_anc = ancilla_qubits[block_id]

            if provided_block_anc and len(provided_block_anc) >= total_required:
                z_ancilla_qs = provided_block_anc[:n_z_checks]
                x_ancilla_qs = provided_block_anc[n_z_checks:n_z_checks + n_x_checks]
                flag_qs = provided_block_anc[n_z_checks + n_x_checks:n_z_checks + n_x_checks + n_flags]
            else:
                # Allocate a contiguous slice large enough for base + flags
                max_existing = max(data_qs) if data_qs else -1
                if provided_block_anc:
                    max_existing = max(max_existing, max(provided_block_anc))
                ancilla_cursor = max_existing + 1
                allocated = list(range(ancilla_cursor, ancilla_cursor + total_required))
                z_ancilla_qs = allocated[:n_z_checks]
                x_ancilla_qs = allocated[n_z_checks:n_z_checks + n_x_checks]
                flag_qs = allocated[n_z_checks + n_x_checks:n_z_checks + n_x_checks + n_flags]
            
            # Initialize flag measurement tracking for this block
            mmap.flag_measurements['X'][block_id] = {}
            mmap.flag_measurements['Z'][block_id] = {}
            
            # Group flag configs by stabilizer type
            z_flag_configs = [c for c in self._flag_configs if c.stabilizer_type == 'Z']
            x_flag_configs = [c for c in self._flag_configs if c.stabilizer_type == 'X']
            
            # === Z Syndrome Extraction with Flags (detects X errors) ===
            # Prepare syndrome ancilla in |0⟩ (Z checks)
            if z_ancilla_qs:
                circuit.append("R", z_ancilla_qs)
                _apply_single_noise(z_ancilla_qs)
            
            # Prepare flag qubits in |+⟩
            z_flag_qs = flag_qs[:len(z_flag_configs)]
            if z_flag_qs:
                circuit.append("R", z_flag_qs)
                circuit.append("H", z_flag_qs)
                _apply_single_noise(z_flag_qs)
            
            circuit.append("TICK")
            
            # For each Z stabilizer, emit flagged CNOT sequence
            for cfg_idx, cfg in enumerate(z_flag_configs):
                flag_q = z_flag_qs[cfg_idx]
                main_ancilla = z_ancilla_qs[cfg.stabilizer_idx]
                
                # CNOT(flag → main_ancilla) to set up flag detection
                circuit.append("CNOT", [flag_q, main_ancilla])
                _apply_two_noise(flag_q, main_ancilla)
                
                # First half of data CNOTs
                for data_idx in cfg.data_qubits_before:
                    ctrl = data_qs[data_idx]
                    tgt = main_ancilla
                    circuit.append("CNOT", [ctrl, tgt])
                    _apply_two_noise(ctrl, tgt)
                
                circuit.append("TICK")
                
                # CNOT(main_ancilla → flag) to complete flag detection
                circuit.append("CNOT", [main_ancilla, flag_q])
                _apply_two_noise(main_ancilla, flag_q)
                
                # Second half of data CNOTs
                for data_idx in cfg.data_qubits_after:
                    ctrl = data_qs[data_idx]
                    tgt = main_ancilla
                    circuit.append("CNOT", [ctrl, tgt])
                    _apply_two_noise(ctrl, tgt)
                
                circuit.append("TICK")
            
            # Remaining Z syndrome CNOTs for non-flagged stabilizers
            # (stabilizers with weight < 4 don't need flags)
            flagged_z_stabs = {c.stabilizer_idx for c in z_flag_configs}
            if self._hz is not None:
                for stab_idx in range(n_z_checks):
                    if stab_idx not in flagged_z_stabs:
                        support = list(np.where(self._hz[stab_idx] == 1)[0])
                        main_ancilla = z_ancilla_qs[stab_idx]
                        for data_idx in support:
                            if data_idx < len(data_qs):
                                ctrl = data_qs[data_idx]
                                circuit.append("CNOT", [ctrl, main_ancilla])
                                _apply_two_noise(ctrl, main_ancilla)
            
            circuit.append("TICK")
            
            # Measure syndrome ancillas
            if z_ancilla_qs:
                circuit.append("M", z_ancilla_qs)
                _apply_meas_noise(z_ancilla_qs)
            z_meas_indices = list(range(current_meas_idx, current_meas_idx + len(z_ancilla_qs)))
            mmap.stabilizer_measurements['Z'][block_id] = z_meas_indices
            current_meas_idx += len(z_ancilla_qs)
            
            # Measure Z flags (H then M for X basis)
            if z_flag_qs:
                circuit.append("H", z_flag_qs)
                circuit.append("M", z_flag_qs)
                _apply_meas_noise(z_flag_qs)
                for cfg_idx, cfg in enumerate(z_flag_configs):
                    mmap.flag_measurements['Z'][block_id][cfg.stabilizer_idx] = [current_meas_idx + cfg_idx]
                current_meas_idx += len(z_flag_qs)
            
            circuit.append("TICK")
            
            # === X Syndrome Extraction with Flags (detects Z errors) ===
            # Prepare syndrome ancilla in |+⟩
            if x_ancilla_qs:
                circuit.append("R", x_ancilla_qs)
                circuit.append("H", x_ancilla_qs)
                _apply_single_noise(x_ancilla_qs)
            
            # Prepare flag qubits in |+⟩  
            x_flag_qs = flag_qs[len(z_flag_configs):len(z_flag_configs) + len(x_flag_configs)]
            if x_flag_qs:
                circuit.append("R", x_flag_qs)
                circuit.append("H", x_flag_qs)
                _apply_single_noise(x_flag_qs)
            
            circuit.append("TICK")
            
            # For each X stabilizer, emit flagged CNOT sequence
            # For X syndrome: CNOT(ancilla → data) - note direction is reversed
            for cfg_idx, cfg in enumerate(x_flag_configs):
                flag_q = x_flag_qs[cfg_idx]
                main_ancilla = x_ancilla_qs[cfg.stabilizer_idx]
                
                # CNOT(flag → main_ancilla)
                circuit.append("CNOT", [flag_q, main_ancilla])
                _apply_two_noise(flag_q, main_ancilla)
                
                # First half of data CNOTs (ancilla → data for X syndrome)
                for data_idx in cfg.data_qubits_before:
                    ctrl = main_ancilla
                    tgt = data_qs[data_idx]
                    circuit.append("CNOT", [ctrl, tgt])
                    _apply_two_noise(ctrl, tgt)
                
                circuit.append("TICK")
                
                # CNOT(main_ancilla → flag)
                circuit.append("CNOT", [main_ancilla, flag_q])
                _apply_two_noise(main_ancilla, flag_q)
                
                # Second half of data CNOTs
                for data_idx in cfg.data_qubits_after:
                    ctrl = main_ancilla
                    tgt = data_qs[data_idx]
                    circuit.append("CNOT", [ctrl, tgt])
                    _apply_two_noise(ctrl, tgt)
                
                circuit.append("TICK")
            
            # Remaining X syndrome CNOTs for non-flagged stabilizers
            flagged_x_stabs = {c.stabilizer_idx for c in x_flag_configs}
            if self._hx is not None:
                for stab_idx in range(n_x_checks):
                    if stab_idx not in flagged_x_stabs:
                        support = list(np.where(self._hx[stab_idx] == 1)[0])
                        main_ancilla = x_ancilla_qs[stab_idx]
                        for data_idx in support:
                            if data_idx < len(data_qs):
                                ctrl = main_ancilla
                                tgt = data_qs[data_idx]
                                circuit.append("CNOT", [ctrl, tgt])
                                _apply_two_noise(ctrl, tgt)
            
            circuit.append("TICK")
            
            # Measure syndrome ancillas in X basis (H then M)
            if x_ancilla_qs:
                circuit.append("H", x_ancilla_qs)
                circuit.append("M", x_ancilla_qs)
                _apply_meas_noise(x_ancilla_qs)
            x_meas_indices = list(range(current_meas_idx, current_meas_idx + len(x_ancilla_qs)))
            mmap.stabilizer_measurements['X'][block_id] = x_meas_indices
            current_meas_idx += len(x_ancilla_qs)
            
            # Measure X flags
            if x_flag_qs:
                circuit.append("H", x_flag_qs)
                circuit.append("M", x_flag_qs)
                _apply_meas_noise(x_flag_qs)
                for cfg_idx, cfg in enumerate(x_flag_configs):
                    mmap.flag_measurements['X'][block_id][cfg.stabilizer_idx] = [current_meas_idx + cfg_idx]
                current_meas_idx += len(x_flag_qs)
            
            circuit.append("TICK")
        
        mmap.total_measurements = current_meas_idx - measurement_offset
        # Invariant: measurement count delta equals mmap.total_measurements
        post_meas = _count_measurements(circuit)
        expected_delta = mmap.total_measurements
        actual_delta = post_meas - pre_meas
        if actual_delta != expected_delta:
            raise AssertionError(
                f"Measurement count mismatch: expected {expected_delta}, got {actual_delta}"
            )
        return mmap
    
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        """Get syndrome schedule including flag measurements."""
        base_schedule = self.base_gadget.get_syndrome_schedule()
        # Add flag information to schedule (append flag rounds at end)
        stabilizer_types = list(base_schedule.stabilizer_types)
        rounds_per_type = dict(base_schedule.rounds_per_type)
        sched = list(base_schedule.schedule)

        if self._flag_configs:
            stabilizer_types += ['FLAG_Z', 'FLAG_X']
            rounds_per_type['FLAG_Z'] = 1
            rounds_per_type['FLAG_X'] = 1
            sched += [('FLAG_Z', 0), ('FLAG_X', 0)]

        return SyndromeSchedule(
            stabilizer_types=stabilizer_types,
            rounds_per_type=rounds_per_type,
            schedule=sched,
        )
    
    def get_logical_map(self, inner_code: Any, outer_code: Any) -> LogicalMeasurementMap:
        """Delegate to base gadget."""
        return self.base_gadget.get_logical_map(inner_code, outer_code)


# =============================================================================
# REPEATED GADGET COMBINATOR  
# =============================================================================

class RepeatedGadget(Gadget):
    """
    Meta-gadget that repeats a base gadget multiple times.
    
    Useful for temporal redundancy in syndrome extraction
    or implementing multiple rounds of EC.
    
    Parameters
    ----------
    gadget : Gadget
        The gadget to repeat.
    n_rounds : int
        Number of repetitions.
    tick_between : bool
        If True, insert TICK between rounds.
        
    Example
    -------
    >>> # 3 rounds of Steane EC for temporal averaging
    >>> repeated = RepeatedGadget(SteaneECGadget(code), n_rounds=3)
    """
    
    def __init__(
        self,
        gadget: Gadget,
        n_rounds: int = 1,
        tick_between: bool = True,
    ):
        self.gadget = gadget
        self.n_rounds = n_rounds
        self.tick_between = tick_between
    
    @property
    def name(self) -> str:
        return f"Repeat({self.gadget.name}, {self.n_rounds}x)"
    
    @property
    def requires_ancillas(self) -> bool:
        return self.gadget.requires_ancillas
    
    @property
    def ancillas_per_block(self) -> int:
        return self.gadget.ancillas_per_block
    
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        ancilla_qubits: Optional[Dict[int, List[int]]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> MeasurementMap:
        """Emit gadget n_rounds times."""
        combined_mmap = MeasurementMap(offset=measurement_offset)
        combined_mmap.stabilizer_measurements = {'X': {}, 'Z': {}}
        combined_mmap.pauli_frame = {'X': {}, 'Z': {}}
        combined_mmap.output_qubits = {}
        combined_mmap.verification_measurements = {}
        combined_mmap.flag_measurements = {'X': {}, 'Z': {}}
        current_meas = measurement_offset
        
        for round_idx in range(self.n_rounds):
            mmap = self.gadget.emit(
                circuit, data_qubits, ancilla_qubits, noise_model, current_meas
            )
            _ensure_mmap_fields(mmap)
            
            # Merge with round prefix
            for stype in ['X', 'Z']:
                if stype in mmap.stabilizer_measurements:
                    for block_id, indices in mmap.stabilizer_measurements[stype].items():
                        key = (f"R{round_idx}", block_id)
                        combined_mmap.stabilizer_measurements[stype][key] = indices

            if hasattr(mmap, 'pauli_frame') and mmap.pauli_frame:
                for ftype in ['X', 'Z']:
                    if ftype in mmap.pauli_frame:
                        for block_id, indices in mmap.pauli_frame[ftype].items():
                            key = (f"R{round_idx}", block_id)
                            combined_mmap.pauli_frame[ftype][key] = indices

            if hasattr(mmap, 'output_qubits') and mmap.output_qubits:
                for block_id, qubits in mmap.output_qubits.items():
                    key = (f"R{round_idx}", block_id)
                    combined_mmap.output_qubits[key] = qubits

            if hasattr(mmap, 'verification_measurements') and mmap.verification_measurements:
                for block_id, indices in mmap.verification_measurements.items():
                    key = (f"R{round_idx}", block_id)
                    combined_mmap.verification_measurements[key] = indices

            if hasattr(mmap, 'flag_measurements') and mmap.flag_measurements:
                for ftype in ['X', 'Z']:
                    if ftype in mmap.flag_measurements:
                        for block_id, indices in mmap.flag_measurements[ftype].items():
                            key = (f"R{round_idx}", block_id)
                            combined_mmap.flag_measurements[ftype][key] = indices
            
            current_meas += mmap.total_measurements
            
            if self.tick_between and round_idx < self.n_rounds - 1:
                circuit.append("TICK")
        
        combined_mmap.total_measurements = current_meas - measurement_offset
        return combined_mmap
    
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        base = self.gadget.get_syndrome_schedule()
        combined = SyndromeSchedule()
        
        for r in range(self.n_rounds):
            for stype in base.stabilizer_types:
                prefixed = f"R{r}_{stype}"
                combined.stabilizer_types.append(prefixed)
                combined.rounds_per_type[prefixed] = base.rounds_per_type.get(stype, 1)
        
        return combined
    
    def get_logical_map(self, inner_code: Any, outer_code: Any) -> LogicalMeasurementMap:
        return self.gadget.get_logical_map(inner_code, outer_code)


# =============================================================================
# GADGET BUILDER (Fluent API)
# =============================================================================

class GadgetBuilder:
    """
    Fluent builder for constructing complex gadget compositions.
    
    Example
    -------
    >>> ec = (GadgetBuilder(concatenated_code)
    ...       .at_level(0, SteaneECGadget)      # Outer: Steane
    ...       .at_level(1, KnillECGadget)       # Inner: Knill
    ...       .parallelize_level(1)             # Inner blocks in parallel
    ...       .repeat(3)                        # 3 rounds
    ...       .build())
    """
    
    def __init__(self, code: Optional['MultiLevelConcatenatedCode'] = None):
        self.code = code
        self._level_gadgets: Dict[int, Gadget] = {}
        self._level_gadget_factories: Dict[int, type] = {}
        self._parallelize_levels: set = set()
        self._n_rounds: int = 1
        self._chain_after: List[Gadget] = []
    
    def at_level(
        self,
        level: int,
        gadget_or_factory: Union[Gadget, type],
    ) -> 'GadgetBuilder':
        """Set gadget or gadget class for a level."""
        if isinstance(gadget_or_factory, type):
            self._level_gadget_factories[level] = gadget_or_factory
        else:
            self._level_gadgets[level] = gadget_or_factory
        return self
    
    def parallelize_level(self, level: int) -> 'GadgetBuilder':
        """Mark a level for parallel execution."""
        self._parallelize_levels.add(level)
        return self
    
    def repeat(self, n_rounds: int) -> 'GadgetBuilder':
        """Repeat the entire EC structure n times."""
        self._n_rounds = n_rounds
        return self
    
    def then(self, gadget: Gadget) -> 'GadgetBuilder':
        """Chain another gadget after the main structure."""
        self._chain_after.append(gadget)
        return self
    
    def build(self) -> Gadget:
        """Build the final composed gadget."""
        if self.code is None:
            raise ValueError("Code must be set for GadgetBuilder")
        
        # Build level gadgets from factories
        level_gadgets = dict(self._level_gadgets)
        for level, factory in self._level_gadget_factories.items():
            if level < len(self.code.level_codes):
                level_gadgets[level] = factory(self.code.level_codes[level])
        
        # Validate that all requested parallelize levels have gadgets or factories
        missing_levels = [
            lvl for lvl in sorted(self._parallelize_levels)
            if lvl not in level_gadgets and lvl not in self._level_gadget_factories
        ]
        if missing_levels:
            raise ValueError(f"parallelize_level requested for levels without gadgets: {missing_levels}")
        
        # Create recursive structure
        gadget = RecursiveGadget(
            code=self.code,
            level_gadgets=level_gadgets,
            parallelize_levels=self._parallelize_levels if self._parallelize_levels else None,
        )
        
        # Wrap in repeat if needed
        if self._n_rounds > 1:
            gadget = RepeatedGadget(gadget, n_rounds=self._n_rounds)
        
        # Chain additional gadgets
        if self._chain_after:
            gadget = ChainedGadget([gadget] + self._chain_after)
        
        return gadget


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def make_recursive_ec(
    code: 'MultiLevelConcatenatedCode',
    inner_gadget: Gadget,
    outer_gadget: Optional[Gadget] = None,
    middle_gadgets: Optional[Dict[int, Gadget]] = None,
) -> RecursiveGadget:
    """
    Convenience function to create a recursive EC gadget.
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The concatenated code.
    inner_gadget : Gadget
        Gadget for innermost level.
    outer_gadget : Gadget, optional
        Gadget for outermost level. If None, uses inner_gadget's type.
    middle_gadgets : Dict[int, Gadget], optional
        Gadgets for middle levels.
        
    Returns
    -------
    RecursiveGadget
        Configured recursive EC gadget.
    """
    depth = code.depth
    level_gadgets = {depth - 1: inner_gadget}
    
    if outer_gadget is not None:
        level_gadgets[0] = outer_gadget
    
    if middle_gadgets:
        level_gadgets.update(middle_gadgets)
    
    return RecursiveGadget(code=code, level_gadgets=level_gadgets)


def make_parallel_steane_ec(
    code: 'CSSCode',
    n_blocks: int,
    extract_x: bool = True,
    extract_z: bool = True,
) -> ParallelGadget:
    """
    Create parallel Steane EC for multiple code blocks.
    
    Parameters
    ----------
    code : CSSCode
        The code for each block.
    n_blocks : int
        Number of blocks.
    extract_x : bool
        Extract X syndromes.
    extract_z : bool
        Extract Z syndromes.
        
    Returns
    -------
    ParallelGadget
        Parallel Steane EC gadget.
    """
    from .steane_ec_gadget import SteaneECGadget
    
    base = SteaneECGadget(
        code=code,
        extract_x_syndrome=extract_x,
        extract_z_syndrome=extract_z,
    )
    return ParallelGadget(base, n_blocks=n_blocks)
