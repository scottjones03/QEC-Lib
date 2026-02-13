# src/qectostim/experiments/hardware_simulation/core/pipeline.py
"""
Compilation pipeline data structures.

Defines the intermediate representations used during compilation
from logical circuits to hardware-executable sequences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    Sequence,
    Iterator,
    TYPE_CHECKING,
)

import stim

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.operations import PhysicalOperation
    from qectostim.experiments.hardware_simulation.core.gates import GateSpec, ParameterizedGate


@dataclass
class QubitMapping:
    """Mapping between logical and physical qubits.
    
    Attributes
    ----------
    logical_to_physical : Dict[int, int]
        Logical qubit index to physical qubit index.
    physical_to_logical : Dict[int, int]
        Physical qubit index to logical qubit index.
    zone_assignments : Dict[int, str]
        Physical qubit to zone ID mapping.
    """
    logical_to_physical: Dict[int, int] = field(default_factory=dict)
    physical_to_logical: Dict[int, int] = field(default_factory=dict)
    zone_assignments: Dict[int, str] = field(default_factory=dict)
    
    def add_mapping(self, logical: int, physical: int, zone: Optional[str] = None) -> None:
        """Add a logical-to-physical qubit mapping."""
        self.logical_to_physical[logical] = physical
        self.physical_to_logical[physical] = logical
        if zone is not None:
            self.zone_assignments[physical] = zone
    
    # Alias so compilers can call mapping.assign(logical, physical)
    assign = add_mapping
    
    def get_physical(self, logical: int) -> Optional[int]:
        """Get physical qubit for a logical qubit."""
        return self.logical_to_physical.get(logical)
    
    def get_logical(self, physical: int) -> Optional[int]:
        """Get logical qubit for a physical qubit."""
        return self.physical_to_logical.get(physical)
    
    def swap_physical(self, phys1: int, phys2: int) -> None:
        """Update mapping after a physical SWAP operation."""
        log1 = self.physical_to_logical.get(phys1)
        log2 = self.physical_to_logical.get(phys2)
        
        if log1 is not None:
            self.logical_to_physical[log1] = phys2
        if log2 is not None:
            self.logical_to_physical[log2] = phys1
        
        self.physical_to_logical[phys1] = log2
        self.physical_to_logical[phys2] = log1
    
    def num_qubits(self) -> int:
        """Get total number of mapped qubits."""
        return len(self.logical_to_physical)
    
    def copy(self) -> "QubitMapping":
        """Create a copy of this mapping."""
        return QubitMapping(
            logical_to_physical=dict(self.logical_to_physical),
            physical_to_logical=dict(self.physical_to_logical),
            zone_assignments=dict(self.zone_assignments),
        )


@dataclass
class ScheduledOperation:
    """An operation with scheduling information.
    
    Attributes
    ----------
    operation : PhysicalOperation
        The operation to execute.
    start_time : float
        Start time in microseconds.
    end_time : float
        End time in microseconds (start_time + duration).
    parallel_group : int
        ID of parallel execution group (operations with same ID run together).
    dependencies : List[int]
        Indices of operations that must complete before this one.
    """
    operation: "PhysicalOperation"
    start_time: float = 0.0
    end_time: float = 0.0
    parallel_group: int = 0
    dependencies: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if self.end_time == 0.0:
            self.end_time = self.start_time + self.operation.duration
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class CircuitLayer:
    """A layer of parallel operations in a circuit.
    
    Operations in the same layer can execute simultaneously.
    
    Attributes
    ----------
    operations : List[PhysicalOperation]
        Operations in this layer.
    start_time : float
        Start time of this layer.
    duration : float
        Duration of this layer (max of operation durations).
    """
    operations: List["PhysicalOperation"] = field(default_factory=list)
    start_time: float = 0.0
    
    @property
    def duration(self) -> float:
        if not self.operations:
            return 0.0
        return max(op.duration for op in self.operations)
    
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration
    
    def add_operation(self, op: "PhysicalOperation") -> None:
        """Add an operation to this layer."""
        self.operations.append(op)
    
    def qubits_used(self) -> set:
        """Get all qubits used in this layer."""
        qubits = set()
        for op in self.operations:
            qubits.update(op.qubits)
        return qubits


@dataclass
class NativeCircuit:
    """Circuit decomposed to native gates.
    
    First stage of compilation: gates are decomposed to the
    hardware's native gate set, but no qubit mapping yet.
    
    Attributes
    ----------
    operations : List[Any]
        Sequence of native gate operations.  Entries are either
        ``(GateSpec, Tuple[int, ...])`` pairs or ``DecomposedGate``
        objects — both forms are accepted.
    num_qubits : int
        Number of logical qubits.
    metadata : Dict[str, Any]
        Additional circuit metadata.
    stim_instruction_map : Dict[int, List[int]]
        Maps stim instruction index (in the flattened ideal circuit)
        to the list of native operation indices that were produced by
        decomposing that instruction.  This is the critical bridge
        between the original stim circuit and native ops.
    stim_source : Optional[stim.Circuit]
        Reference to the original stim circuit that was decomposed.
        Stored so that downstream stages can access annotations
        (DETECTOR, OBSERVABLE_INCLUDE, etc.) that were deliberately
        *not* decomposed into native ops.
    """
    operations: List[Any] = field(default_factory=list)
    num_qubits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    stim_instruction_map: Dict[int, List[int]] = field(default_factory=dict)
    stim_source: Optional[stim.Circuit] = None
    
    def add_gate(self, gate: "GateSpec", qubits: Tuple[int, ...]) -> None:
        """Add a native gate to the circuit."""
        self.operations.append((gate, qubits))
        self.num_qubits = max(self.num_qubits, max(qubits) + 1)
    
    def gate_count(self) -> int:
        """Total number of gates."""
        return len(self.operations)
    
    def two_qubit_count(self) -> int:
        """Number of two-qubit gates."""
        count = 0
        for op in self.operations:
            if hasattr(op, 'qubits'):
                # DecomposedGate
                if len(op.qubits) == 2:
                    count += 1
            elif isinstance(op, tuple) and len(op) >= 2:
                # (GateSpec, qubits) tuple
                if len(op[1]) == 2:
                    count += 1
        return count
    
    def __len__(self) -> int:
        return len(self.operations)
    
    def __iter__(self):
        return iter(self.operations)


@dataclass
class MappedCircuit:
    """Circuit with logical-to-physical qubit mapping.
    
    Second stage: logical qubits are assigned to physical qubits,
    but routing (SWAP insertion) has not been done yet.
    
    Attributes
    ----------
    native_circuit : NativeCircuit
        The underlying native circuit.
    mapping : QubitMapping
        Current qubit mapping.
    metadata : Dict[str, Any]
        Additional mapping metadata.
    """
    native_circuit: NativeCircuit
    mapping: QubitMapping
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def physical_operations(self) -> Iterator[Tuple["GateSpec", Tuple[int, ...]]]:
        """Iterate over operations with physical qubit indices."""
        for gate, logical_qubits in self.native_circuit:
            physical_qubits = tuple(
                self.mapping.get_physical(q) for q in logical_qubits
            )
            yield gate, physical_qubits


@dataclass
class RoutedCircuit:
    """Circuit with routing operations inserted.
    
    Third stage: SWAP/transport operations have been inserted
    to satisfy connectivity constraints.
    
    Attributes
    ----------
    operations : List[PhysicalOperation]
        Sequence of physical operations including routing.
    final_mapping : Optional[QubitMapping]
        Qubit mapping after all operations.
    routing_overhead : int
        Number of routing operations added.
    mapped_circuit : Optional[MappedCircuit]
        Reference to the pre-routing mapped circuit.
    routing_operations : List[Any]
        Explicit list of routing-only operations inserted.
    metadata : Dict[str, Any]
        Additional routing metadata.
    """
    operations: List["PhysicalOperation"] = field(default_factory=list)
    final_mapping: Optional[QubitMapping] = None
    routing_overhead: int = 0
    mapped_circuit: Optional[MappedCircuit] = None
    routing_operations: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_operation(self, op: "PhysicalOperation") -> None:
        """Add an operation to the routed circuit."""
        self.operations.append(op)
    
    def total_operations(self) -> int:
        """Total number of operations."""
        return len(self.operations)
    
    def __len__(self) -> int:
        return len(self.operations)
    
    def __iter__(self):
        return iter(self.operations)


@dataclass  
class ScheduledCircuit:
    """Circuit with timing and parallelization information.
    
    Fourth stage: operations are scheduled with start times
    and parallel execution groups.
    
    Attributes
    ----------
    layers : List[CircuitLayer]
        Sequence of parallel execution layers.
    scheduled_ops : List[ScheduledOperation]
        All operations with timing info.
    total_duration : float
        Total circuit execution time in microseconds.
    routed_circuit : Optional[RoutedCircuit]
        Reference to the pre-scheduling routed circuit.
    batches : List[Any]
        Parallel execution batches from the scheduler.
    metadata : Dict[str, Any]
        Additional scheduling metadata.
    """
    layers: List[CircuitLayer] = field(default_factory=list)
    scheduled_ops: List[ScheduledOperation] = field(default_factory=list)
    total_duration: float = 0.0
    routed_circuit: Optional[RoutedCircuit] = None
    batches: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_layer(self, layer: CircuitLayer) -> None:
        """Add a parallel execution layer."""
        if self.layers:
            layer.start_time = self.layers[-1].end_time
        self.layers.append(layer)
        self.total_duration = layer.end_time
    
    def depth(self) -> int:
        """Circuit depth (number of layers)."""
        return len(self.layers)
    
    @property
    def layer_count(self) -> int:
        """Number of layers (alias for depth())."""
        return self.depth()
    
    def parallelism(self) -> float:
        """Average operations per layer."""
        if not self.layers:
            return 0.0
        total_ops = sum(len(layer.operations) for layer in self.layers)
        return total_ops / len(self.layers)


@dataclass
class CompiledCircuit:
    """Fully compiled circuit ready for simulation.
    
    Final stage: contains all information needed to generate
    Stim circuits and run hardware simulation.
    
    Attributes
    ----------
    scheduled : ScheduledCircuit
        The scheduled circuit.
    mapping : QubitMapping
        Final qubit mapping.
    original_circuit : Optional[stim.Circuit]
        The original ideal stim circuit that was compiled.  This is
        the source of truth for noise injection — noise instructions
        are interleaved around the original gates, preserving all
        DETECTOR/OBSERVABLE_INCLUDE/QUBIT_COORDS annotations.
    stim_circuit : Optional[stim.Circuit]
        Generated Stim circuit (if already built).
    metrics : Dict[str, Any]
        Compilation metrics (depth, gate count, etc.).
    """
    scheduled: ScheduledCircuit
    mapping: QubitMapping
    original_circuit: Optional[stim.Circuit] = None
    stim_circuit: Optional[stim.Circuit] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_duration(self) -> float:
        """Total execution time in microseconds."""
        return self.scheduled.total_duration
    
    @property
    def depth(self) -> int:
        """Circuit depth (number of layers)."""
        return self.scheduled.depth()
    
    def to_stim(self) -> stim.Circuit:
        """Generate Stim circuit from compiled circuit.
        
        If the original ideal circuit is available, it is used as the
        source of truth — DETECTOR, OBSERVABLE_INCLUDE, QUBIT_COORDS,
        and SHIFT_COORDS annotations are preserved from it, with qubit
        indices remapped through ``self.mapping``.
        
        If no original circuit is stored, falls back to building a
        gate-only circuit from scheduled layers (no annotations).
        
        Returns cached circuit if already built.
        """
        if self.stim_circuit is not None:
            return self.stim_circuit
        
        if self.original_circuit is not None:
            self.stim_circuit = self._rebuild_from_original()
        else:
            self.stim_circuit = self._build_from_layers()
        
        return self.stim_circuit
    
    def _remap_qubit(self, logical: int) -> int:
        """Map a logical qubit index through the mapping, if available."""
        phys = self.mapping.get_physical(logical)
        return phys if phys is not None else logical
    
    def _rebuild_from_original(self) -> stim.Circuit:
        """Rebuild stim circuit from the original ideal circuit.
        
        Walks the original circuit and:
        - Remaps qubit indices on gate instructions through self.mapping
        - Copies annotations (DETECTOR, OBSERVABLE_INCLUDE, QUBIT_COORDS,
          SHIFT_COORDS) verbatim — these use relative rec[] targets, not
          qubit indices, so they don't need remapping.
        - Preserves REPEAT block structure.
        """
        circuit = stim.Circuit()
        
        # Emit QUBIT_COORDS preamble from original if present
        for inst in self.original_circuit:
            if isinstance(inst, stim.CircuitRepeatBlock):
                circuit.append(stim.CircuitRepeatBlock(
                    inst.repeat_count,
                    self._rebuild_body_from_original(inst.body_copy()),
                ))
            else:
                circuit.append(inst)
        
        return circuit
    
    def _rebuild_body_from_original(self, body: stim.Circuit) -> stim.Circuit:
        """Recursively rebuild a circuit body, preserving structure."""
        result = stim.Circuit()
        for inst in body:
            if isinstance(inst, stim.CircuitRepeatBlock):
                result.append(stim.CircuitRepeatBlock(
                    inst.repeat_count,
                    self._rebuild_body_from_original(inst.body_copy()),
                ))
            else:
                result.append(inst)
        return result
    
    def _build_from_layers(self) -> stim.Circuit:
        """Build gate-only circuit from scheduled layers (no annotations).
        
        This is the fallback when no original_circuit is available.
        Useful for debugging/metrics but NOT suitable for decoding.
        """
        circuit = stim.Circuit()
        
        for layer in self.scheduled.layers:
            for op in layer.operations:
                for instruction in op.to_stim_instructions():
                    if instruction.strip():
                        circuit.append_from_stim_program_text(instruction)
            circuit.append("TICK")
        
        return circuit
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute and cache compilation metrics."""
        if self.metrics:
            return self.metrics
        
        total_ops = sum(len(layer.operations) for layer in self.scheduled.layers)
        two_qubit_ops = sum(
            1 for layer in self.scheduled.layers
            for op in layer.operations
            if len(op.qubits) == 2
        )
        
        self.metrics = {
            "total_operations": total_ops,
            "two_qubit_operations": two_qubit_ops,
            "depth": self.depth,
            "duration_us": self.total_duration,
            "parallelism": self.scheduled.parallelism(),
            "num_qubits": self.mapping.num_qubits(),
        }

        # --- Propagate routing/mapping metadata for downstream use ---
        # Each platform puts its own keys into routing_result.metadata;
        # core forwards them generically without knowing key names.
        routed = self.scheduled.routed_circuit
        if routed is not None and routed.metadata:
            self.metrics.update(routed.metadata)

        if routed is not None and routed.mapped_circuit is not None:
            mmeta = getattr(routed.mapped_circuit, "metadata", None) or {}
            if mmeta:
                self.metrics.update(mmeta)

        return self.metrics
