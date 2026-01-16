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
    operations : List[Tuple[GateSpec, Tuple[int, ...]]]
        Sequence of (gate, logical_qubits) pairs.
    num_qubits : int
        Number of logical qubits.
    metadata : Dict[str, Any]
        Additional circuit metadata.
    """
    operations: List[Tuple["GateSpec", Tuple[int, ...]]] = field(default_factory=list)
    num_qubits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_gate(self, gate: "GateSpec", qubits: Tuple[int, ...]) -> None:
        """Add a native gate to the circuit."""
        self.operations.append((gate, qubits))
        self.num_qubits = max(self.num_qubits, max(qubits) + 1)
    
    def gate_count(self) -> int:
        """Total number of gates."""
        return len(self.operations)
    
    def two_qubit_count(self) -> int:
        """Number of two-qubit gates."""
        return sum(1 for gate, qubits in self.operations if len(qubits) == 2)
    
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
    """
    native_circuit: NativeCircuit
    mapping: QubitMapping
    
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
    final_mapping : QubitMapping
        Qubit mapping after all operations.
    routing_overhead : int
        Number of routing operations added.
    """
    operations: List["PhysicalOperation"] = field(default_factory=list)
    final_mapping: Optional[QubitMapping] = None
    routing_overhead: int = 0
    
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
    """
    layers: List[CircuitLayer] = field(default_factory=list)
    scheduled_ops: List[ScheduledOperation] = field(default_factory=list)
    total_duration: float = 0.0
    
    def add_layer(self, layer: CircuitLayer) -> None:
        """Add a parallel execution layer."""
        if self.layers:
            layer.start_time = self.layers[-1].end_time
        self.layers.append(layer)
        self.total_duration = layer.end_time
    
    def depth(self) -> int:
        """Circuit depth (number of layers)."""
        return len(self.layers)
    
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
    stim_circuit : Optional[stim.Circuit]
        Generated Stim circuit (if already built).
    metrics : Dict[str, Any]
        Compilation metrics (depth, gate count, etc.).
    """
    scheduled: ScheduledCircuit
    mapping: QubitMapping
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
        
        Returns cached circuit if already built.
        """
        if self.stim_circuit is not None:
            return self.stim_circuit
        
        # Build Stim circuit from scheduled operations
        circuit = stim.Circuit()
        
        for layer in self.scheduled.layers:
            for op in layer.operations:
                for instruction in op.to_stim_instructions():
                    if instruction.strip():
                        circuit.append_from_stim_program_text(instruction)
            # Add TICK between layers
            circuit.append("TICK")
        
        self.stim_circuit = circuit
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
        return self.metrics
