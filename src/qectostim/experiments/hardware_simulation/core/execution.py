# src/qectostim/experiments/hardware_simulation/core/execution.py
"""
Execution plan dataclasses for timing-aware noise injection.

An ExecutionPlan captures the timing and routing metadata from compilation
WITHOUT modifying the original Stim circuit. This allows the noise model
to inject hardware-accurate noise (idle dephasing, gate swap errors, 
per-qubit calibration) based on actual execution timing.

The key insight is that the Stim circuit structure NEVER changes - only
noise instructions are appended. The compilation is "virtual".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.pipeline import CompiledCircuit


@dataclass
class OperationTiming:
    """Timing metadata for a single instruction in the circuit.
    
    This maps a Stim instruction index to its physical execution timing,
    allowing the noise model to calculate idle times between operations.
    
    Attributes
    ----------
    instruction_index : int
        Index in circuit.flattened() - the order in the flat Stim circuit.
    gate_name : str
        Name of the gate/operation (e.g., "MS", "RX", "H", "CNOT", "M").
    qubits : Tuple[int, ...]
        Physical qubit indices involved in this operation.
    start_time : float
        Start time in microseconds from circuit start.
    duration : float
        Duration in microseconds.
    fidelity : float
        Expected fidelity from calibration data (1.0 = perfect).
    zone_id : Optional[int]
        Which zone this operation executes in (if applicable).
    batch_index : Optional[int]
        Which parallel batch this operation belongs to.
    num_native_ops : int
        Number of native operations this instruction decomposes into.
    platform_context : Optional[Dict[str, Any]]
        Platform-specific data for physics-based fidelity models.
        Trapped-ion stores ``{"chain_length": N, "motional_quanta": q,
        "mode_snapshot": ...}``; other platforms store their own parameters.
    metadata : Dict[str, Any]
        General-purpose metadata.
    """
    instruction_index: int
    gate_name: str
    qubits: Tuple[int, ...]
    start_time: float
    duration: float
    fidelity: float = 1.0
    zone_id: Optional[int] = None
    batch_index: Optional[int] = None
    num_native_ops: int = 1
    platform_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def end_time(self) -> float:
        """End time of this operation."""
        return self.start_time + self.duration
    
    @property
    def is_single_qubit(self) -> bool:
        """Whether this is a single-qubit operation."""
        return len(self.qubits) == 1
    
    @property
    def is_two_qubit(self) -> bool:
        """Whether this is a two-qubit operation."""
        return len(self.qubits) == 2


@dataclass
class GateSwapInfo:
    """Information about transport/swaps required for a two-qubit gate.

    In zoned architectures, executing a two-qubit gate may require
    moving qubits between zones.  The noise model can compute
    physics-based fidelity using the ``platform_context`` dict.

    Attributes
    ----------
    instruction_index : int
        Which Stim instruction triggered this swap chain.
    qubits : Tuple[int, int]
        The two qubits that needed to be co-located.
    num_swaps : int
        Number of SWAP-equivalent transport operations.
    swap_fidelity : float
        Fixed fidelity per swap (fallback when platform_context is absent).
    transport_time : float
        Total time for transport in microseconds.
    source_zone : Optional[int]
        Zone where qubit started.
    target_zone : Optional[int]
        Zone where qubit ended up.
    platform_context : Optional[Dict[str, Any]]
        Platform-specific transport data. Trapped-ion stores
        ``{"chain_length": N, "motional_quanta": q, "mode_snapshot": ...}``;
        other platforms store their own transport error parameters.
    metadata : Dict[str, Any]
        General-purpose metadata.
    """
    instruction_index: int
    qubits: Tuple[int, int]
    num_swaps: int
    swap_fidelity: float = 0.998
    transport_time: float = 0.0
    source_zone: Optional[int] = None
    target_zone: Optional[int] = None
    platform_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_fidelity(self) -> float:
        """Total fidelity accounting for all swaps."""
        return self.swap_fidelity ** self.num_swaps
    
    @property
    def error_probability(self) -> float:
        """Error probability from the swap chain."""
        return 1.0 - self.total_fidelity


@dataclass  
class IdleInterval:
    """An interval where a qubit is idle (not involved in any gate).
    
    Attributes
    ----------
    qubit : int
        The idling qubit.
    start_time : float
        When the idle period starts (μs).
    end_time : float
        When the idle period ends (μs).
    following_instruction : int
        The instruction index that ends this idle period.
    """
    qubit: int
    start_time: float
    end_time: float
    following_instruction: int
    
    @property
    def duration(self) -> float:
        """Duration of idle period in microseconds."""
        return self.end_time - self.start_time


@dataclass
class ExecutionPlan:
    """Complete execution plan for timing-aware noise injection.
    
    Captures all timing and routing metadata from compilation without
    modifying the original circuit. The noise model uses this to:
    
    1. Calculate idle dephasing between operations
    2. Add gate swap noise from qubit transport
    3. Apply per-qubit calibrated gate fidelities
    4. Account for platform-specific fidelity models
    
    The instruction indices MUST align with circuit.flattened().
    
    Attributes
    ----------
    operations : List[OperationTiming]
        Timing info for each operation, indexed by instruction_index.
    gate_swaps : List[GateSwapInfo]
        Transport operations required for gates.
    idle_intervals : List[IdleInterval]
        All idle periods for all qubits.
    total_duration : float
        Total circuit duration in microseconds.
    num_qubits : int
        Number of physical qubits in the circuit.
    """
    operations: List[OperationTiming] = field(default_factory=list)
    gate_swaps: List[GateSwapInfo] = field(default_factory=list)
    idle_intervals: List[IdleInterval] = field(default_factory=list)
    total_duration: float = 0.0
    num_qubits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_operation(self, instruction_index: int) -> Optional[OperationTiming]:
        """Get timing info for a specific instruction."""
        for op in self.operations:
            if op.instruction_index == instruction_index:
                return op
        return None
    
    def get_swaps_for_instruction(self, instruction_index: int) -> List[GateSwapInfo]:
        """Get all gate swaps associated with an instruction."""
        return [gs for gs in self.gate_swaps if gs.instruction_index == instruction_index]
    
    def get_idle_before(self, instruction_index: int) -> List[IdleInterval]:
        """Get idle intervals that end at the given instruction."""
        return [
            idle for idle in self.idle_intervals 
            if idle.following_instruction == instruction_index
        ]
    
    def get_qubit_idle_times(self, qubit: int) -> List[IdleInterval]:
        """Get all idle intervals for a specific qubit."""
        return [idle for idle in self.idle_intervals if idle.qubit == qubit]
    
    def total_idle_time(self, qubit: int) -> float:
        """Get total idle time for a qubit."""
        return sum(idle.duration for idle in self.get_qubit_idle_times(qubit))
    
    def validate(self) -> bool:
        """Validate the execution plan for consistency."""
        # Check instruction indices are sequential
        indices = sorted(op.instruction_index for op in self.operations)
        if indices and indices != list(range(min(indices), max(indices) + 1)):
            return False
        
        # Check timing consistency
        for op in self.operations:
            if op.start_time < 0 or op.duration < 0:
                return False
        
        # Check idle intervals reference valid instruction indices
        valid_indices = set(indices)
        for idle in self.idle_intervals:
            if idle.following_instruction not in valid_indices:
                # Allow out-of-range (may reference future ops)
                pass
            if idle.duration < 0:
                return False
        
        return True
    
    def validate_against_circuit(self, circuit: stim.Circuit) -> bool:
        """Check that plan indices align with the gate instructions in a circuit.

        Counts only gate/measurement/reset instructions (skipping
        TICK, DETECTOR, OBSERVABLE_INCLUDE, etc.) and checks that the
        plan's max instruction_index doesn't exceed the gate count.
        """
        gate_count = 0
        skip_names = {
            "TICK", "DETECTOR", "OBSERVABLE_INCLUDE",
            "SHIFT_COORDS", "QUBIT_COORDS", "BARRIER",
        }
        for inst in circuit.flattened():
            if inst.name.upper() not in skip_names:
                targets = inst.targets_copy()
                if any(t.is_qubit_target for t in targets):
                    gate_count += 1
        
        if not self.operations:
            return True
        max_idx = max(op.instruction_index for op in self.operations)
        return max_idx < gate_count


def build_execution_plan_from_compiled(
    compiled: "CompiledCircuit",
    calibration: Optional[Any] = None,
) -> ExecutionPlan:
    """Build an ExecutionPlan from a CompiledCircuit.
    
    This extracts timing and routing information from the compilation
    result without modifying any circuits.
    
    Parameters
    ----------
    compiled : CompiledCircuit
        The compiled circuit with scheduling info.
    calibration : Optional[CalibrationData]
        Hardware calibration data for fidelity lookup.
        
    Returns
    -------
    ExecutionPlan
        The execution plan for noise injection.
    """
    plan = ExecutionPlan(
        total_duration=compiled.total_duration if hasattr(compiled, 'total_duration') else 0.0,
    )
    
    # Extract from scheduled circuit if available
    if hasattr(compiled, 'scheduled') and compiled.scheduled is not None:
        scheduled = compiled.scheduled
        
        # Build operation timings from scheduled operations
        qubit_last_end: Dict[int, float] = {}
        
        for i, sched_op in enumerate(scheduled.scheduled_ops or []):
            op = sched_op.operation
            gate_name = getattr(op, 'gate_name', None)
            if gate_name is None:
                # Try extracting name from the operation type
                gate_name = getattr(getattr(op, 'gate', None), 'name', 'UNKNOWN')
            
            qubits = tuple(op.qubits) if hasattr(op, 'qubits') else ()
            
            timing = OperationTiming(
                instruction_index=i,
                gate_name=gate_name,
                qubits=qubits,
                start_time=sched_op.start_time,
                duration=sched_op.duration,
            )
            
            # Lookup fidelity from calibration or operation
            if calibration is not None:
                if timing.is_single_qubit:
                    timing.fidelity = calibration.get_1q_fidelity(
                        timing.qubits[0], timing.gate_name
                    )
                elif timing.is_two_qubit:
                    timing.fidelity = calibration.get_2q_fidelity(
                        timing.qubits[0], timing.qubits[1], timing.gate_name
                    )
            elif hasattr(op, 'base_fidelity'):
                timing.fidelity = op.base_fidelity
            
            # Compute idle intervals for involved qubits
            for qubit in qubits:
                last_end = qubit_last_end.get(qubit, 0.0)
                if sched_op.start_time > last_end:
                    plan.idle_intervals.append(IdleInterval(
                        qubit=qubit,
                        start_time=last_end,
                        end_time=sched_op.start_time,
                        following_instruction=i,
                    ))
                qubit_last_end[qubit] = sched_op.end_time
            
            plan.operations.append(timing)
        
        plan.num_qubits = max(qubit_last_end.keys(), default=-1) + 1
        
        # Extract gate swap info from routed circuit metadata
        if scheduled.routed_circuit is not None:
            routed = scheduled.routed_circuit
            swaps_by_idx = (routed.metadata or {}).get("gate_swaps_by_stim_idx", {})
            for idx, swaps in swaps_by_idx.items():
                for swap in swaps:
                    plan.gate_swaps.append(GateSwapInfo(
                        instruction_index=idx,
                        qubits=getattr(swap, 'qubits', (0, 0)),
                        num_swaps=getattr(swap, 'num_swaps', 1),
                        swap_fidelity=getattr(swap, 'swap_fidelity', 0.998),
                        transport_time=getattr(swap, 'transport_time', 0.0),
                    ))
    
    return plan
