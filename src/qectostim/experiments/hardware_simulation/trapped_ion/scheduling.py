# src/qectostim/experiments/hardware_simulation/trapped_ion/scheduling.py
"""
WISE-specific batch scheduling for trapped ion QCCD architectures.

This module provides schedulers optimized for WISE (Wiring-based Ion Shuttling
for Entanglement) architectures, which use:
- Type-based operation batching (all transports, then all gates, etc.)
- Global barriers between operation types
- SAT-optimized reconfiguration phases

The core BatchScheduler ABC is in core/operations.py; this module provides
trapped-ion specific implementations.

Ported and refactored from qccd_parallelisation.py.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np

from qectostim.experiments.hardware_simulation.core.operations import (
    BatchScheduler,
    TypeOrderedBatchScheduler,
    BarrierAwareScheduler as _CoreBarrierAwareScheduler,
    DependencyEdge,
    OperationBatch,
    OperationType,
    PhysicalOperation,
)

if TYPE_CHECKING:
    from .architecture import Ion, QCCDNode as QCCDComponent
    from .operations import QCCDOperationBase


_logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Dephasing time constant (T2) in seconds
# From https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
T2_DEPHASING_TIME: float = 2.2


# =============================================================================
# Dephasing Calculations
# =============================================================================

def calculate_dephasing_fidelity(time_us: float) -> float:
    """Calculate dephasing fidelity for a given idle time.
    
    Uses exponential decay model based on T2 dephasing time.
    
    Parameters
    ----------
    time_us : float
        Idle time in microseconds.
        
    Returns
    -------
    float
        Fidelity in [0, 1] where 1 is perfect.
        
    Notes
    -----
    Formula: F = 1 - (1 - exp(-t/T2))/2
    Reference: https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    """
    time_s = time_us * 1e-6  # Convert to seconds
    return 1.0 - (1.0 - np.exp(-time_s / T2_DEPHASING_TIME)) / 2.0


# =============================================================================
# WISE Batch Scheduler
# =============================================================================

# WISE-specific type ordering for trapped ion QCCD
WISE_TYPE_ORDER = [
    OperationType.TRANSPORT,
    OperationType.GATE_2Q,
    OperationType.GATE_1Q,
    OperationType.MEASUREMENT,
    OperationType.RESET,
    OperationType.RECOVERY,
]


class WISEBatchScheduler(TypeOrderedBatchScheduler):
    """Scheduler for WISE-style trapped ion architectures.
    
    Extends TypeOrderedBatchScheduler with WISE-specific type ordering:
    1. All transports (horizontal/vertical ion movements)
    2. Barrier
    3. All two-qubit gates
    4. All single-qubit gates
    5. Measurements
    
    This matches the WISE SAT-based reconfiguration model where
    all ions move in coordinated phases to avoid collisions and
    minimize total reconfiguration time.
    
    Inherits from core.operations.TypeOrderedBatchScheduler to reuse
    the generic type-ordered scheduling logic.
    
    Example
    -------
    >>> scheduler = WISEBatchScheduler()
    >>> batches = scheduler.schedule(operations)
    >>> for batch in batches:
    ...     print(f"{batch.batch_type}: {len(batch)} ops, {batch.duration}μs")
    """
    
    def __init__(
        self,
        name: str = "wise_scheduler",
        use_critical_path: bool = True,
    ) -> None:
        """Initialize the WISE batch scheduler.
        
        Parameters
        ----------
        name : str
            Scheduler name for identification.
        use_critical_path : bool
            If True, prioritize operations on the critical path.
        """
        super().__init__(name=name, type_order=WISE_TYPE_ORDER)
        self.use_critical_path = use_critical_path
    
    # build_dependency_dag is inherited from TypeOrderedBatchScheduler
    # schedule is inherited from TypeOrderedBatchScheduler
    # _split_conflicting_ops is inherited from TypeOrderedBatchScheduler


# =============================================================================
# Critical Path Scheduler (WISE variant)
# =============================================================================

class WISECriticalPathScheduler(WISEBatchScheduler):
    """WISE scheduler with critical path optimization.
    
    Extends WISEBatchScheduler to prioritize operations on the
    critical path (longest dependency chain) to minimize total
    circuit execution time.
    
    Uses the "longest path to any leaf" heuristic to determine
    operation priority within each type group.
    """
    
    def __init__(self, name: str = "wise_critical_path"):
        super().__init__(name, use_critical_path=True)
    
    def schedule(
        self,
        operations: List[PhysicalOperation],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[OperationBatch]:
        """Schedule with critical path prioritization."""
        if not operations:
            return []
        
        # Build dependency DAG
        dependencies = self.build_dependency_dag(operations)
        
        # Compute critical weights (longest path to leaf)
        critical_weights = self._compute_critical_weights(operations, dependencies)
        
        # Sort operations within each type by critical weight (descending)
        sorted_ops = sorted(
            operations,
            key=lambda op: (
                op.operation_type.value,  # Group by type
                -critical_weights.get(id(op), 0.0),  # Then by critical weight
            ),
        )
        
        # Use parent scheduler with sorted operations
        return super().schedule(sorted_ops, constraints)
    
    def _compute_critical_weights(
        self,
        operations: List[PhysicalOperation],
        dependencies: List[DependencyEdge],
    ) -> Dict[int, float]:
        """Compute critical weight for each operation.
        
        Critical weight = operation duration + max(successor critical weights)
        """
        n = len(operations)
        if n == 0:
            return {}
        
        # Build adjacency list
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for edge in dependencies:
            adj[edge.source].append(edge.target)
        
        # Map operation id to index
        idx_to_op = {i: operations[i] for i in range(n)}
        op_id_to_idx = {id(operations[i]): i for i in range(n)}
        
        # Compute critical weights in reverse topological order
        critical_weight: Dict[int, float] = {}
        
        # Simple DFS-based computation
        def compute(idx: int) -> float:
            if idx in critical_weight:
                return critical_weight[idx]
            
            op = idx_to_op[idx]
            successors = adj[idx]
            
            if not successors:
                critical_weight[idx] = op.duration
            else:
                max_succ = max(compute(s) for s in successors)
                critical_weight[idx] = op.duration + max_succ
            
            return critical_weight[idx]
        
        for i in range(n):
            compute(i)
        
        # Map back to operation ids
        return {id(operations[i]): critical_weight[i] for i in range(n)}


# =============================================================================
# Parallel Operation Scheduling (from qccd_parallelisation.py)
# =============================================================================

@dataclass
class ParallelOperationGroup:
    """A group of operations executing in parallel.
    
    Attributes
    ----------
    operations : List[QCCDOperationBase]
        Operations in this parallel group.
    active_operations : List[QCCDOperationBase]
        Operations that are still active from previous groups.
    start_time : float
        Start time of this group in microseconds.
    """
    operations: List["QCCDOperationBase"] = field(default_factory=list)
    active_operations: List["QCCDOperationBase"] = field(default_factory=list)
    start_time: float = 0.0
    
    @property
    def duration(self) -> float:
        """Duration of the longest operation in this group."""
        if not self.operations:
            return 0.0
        return max(op.calculate_time() for op in self.operations)
    
    @property
    def end_time(self) -> float:
        """End time of this group."""
        return self.start_time + self.duration


def build_happens_before_dag(
    operations: Sequence["QCCDOperationBase"],
    all_components: List["QCCDComponent"],
) -> Tuple[Dict["QCCDOperationBase", List["QCCDOperationBase"]], List["QCCDOperationBase"]]:
    """Build happens-before DAG for operations.
    
    Operations on the same component must be ordered.
    
    Parameters
    ----------
    operations : Sequence[QCCDOperationBase]
        Operations to analyze.
    all_components : List[QCCDComponent]
        All components involved.
        
    Returns
    -------
    Tuple[Dict, List]
        (happens_before adjacency list, topologically sorted operations)
    """
    happens_before: Dict["QCCDOperationBase", List["QCCDOperationBase"]] = {
        op: [] for op in operations
    }
    indegree: Dict["QCCDOperationBase", int] = {op: 0 for op in operations}
    operations_by_component: Dict["QCCDComponent", List["QCCDOperationBase"]] = {
        c: [] for c in all_components
    }
    
    # Build happens-before based on shared components
    for op in operations:
        for component in set(op.involved_components):
            for prev_op in operations_by_component.get(component, []):
                if op not in happens_before[prev_op]:
                    happens_before[prev_op].append(op)
                    indegree[op] += 1
            if component in operations_by_component:
                operations_by_component[component].append(op)
    
    # Topological sort using Kahn's algorithm
    zero_indegree = deque([op for op in operations if indegree[op] == 0])
    topo_sorted: List["QCCDOperationBase"] = []
    
    while zero_indegree:
        op = zero_indegree.popleft()
        topo_sorted.append(op)
        for neighbor in happens_before[op]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                zero_indegree.append(neighbor)
    
    return happens_before, topo_sorted


def schedule_operations_wise(
    operations: Sequence["QCCDOperationBase"],
    is_wise_arch: bool = True,
) -> Dict[float, ParallelOperationGroup]:
    """Schedule operations for WISE architecture.
    
    Uses critical-path scheduling with WISE-specific constraints:
    - Global barriers between operation batches
    - Type-based grouping
    - Greedy packing of non-conflicting operations
    
    Parameters
    ----------
    operations : Sequence[QCCDOperationBase]
        Operations to schedule.
    is_wise_arch : bool
        If True, enforce WISE batch barriers.
        
    Returns
    -------
    Dict[float, ParallelOperationGroup]
        Mapping from start time to parallel operation group.
        
    Notes
    -----
    Ported from qccd_parallelisation.py:paralleliseOperations()
    """
    if not operations:
        return {}
    
    operations = list(operations)
    
    # Collect all components
    all_components: List["QCCDComponent"] = []
    for op in operations:
        for c in op.involved_components:
            if c not in all_components:
                all_components.append(c)
    
    # Build DAG and topological order
    happens_before, topo_order = build_happens_before_dag(operations, all_components)
    
    # Predecessor map
    preds: Dict["QCCDOperationBase", List["QCCDOperationBase"]] = {
        op: [] for op in operations
    }
    for op, succs in happens_before.items():
        for nxt in succs:
            if nxt in preds:
                preds[nxt].append(op)
    
    # Topological position for tie-breaking
    topo_pos: Dict["QCCDOperationBase", int] = {op: i for i, op in enumerate(topo_order)}
    
    # Compute critical weight (longest path to leaf)
    critical_weight: Dict["QCCDOperationBase", float] = {}
    for op in reversed(topo_order):
        succs = happens_before.get(op, [])
        if succs:
            critical_weight[op] = op.calculate_time() + max(
                critical_weight[s] for s in succs
            )
        else:
            critical_weight[op] = op.calculate_time()
    
    # Scheduling state
    time_schedule: Dict[float, ParallelOperationGroup] = {}
    operation_end_times: Dict["QCCDOperationBase", float] = {}
    component_busy_until: Dict["QCCDComponent", float] = {c: 0.0 for c in all_components}
    earliest_start: Dict["QCCDOperationBase", float] = {op: 0.0 for op in operations}
    arch_busy_until: float = 0.0
    
    scheduled: Set["QCCDOperationBase"] = set()
    active_ops: List["QCCDOperationBase"] = []
    current_time = 0.0
    
    def all_preds_scheduled(op: "QCCDOperationBase") -> bool:
        return all(p in scheduled for p in preds[op])
    
    # Main scheduling loop
    while len(scheduled) < len(operations):
        remaining = [op for op in topo_order if op not in scheduled]
        frontier = [op for op in remaining if all_preds_scheduled(op)]
        
        if not frontier:
            break
        
        # Compute earliest start for each frontier op
        ready_data = []
        min_start = float("inf")
        
        for op in frontier:
            comp_ready = max(
                component_busy_until[comp] for comp in op.involved_components
            )
            dep_ready = earliest_start[op]
            if is_wise_arch:
                start_t = max(comp_ready, dep_ready, arch_busy_until)
            else:
                start_t = max(comp_ready, dep_ready)
            ready_data.append((op, start_t))
            min_start = min(min_start, start_t)
        
        current_time = min_start
        candidates = [op for op, t in ready_data if t == current_time]
        
        if not candidates:
            next_t = min(t for _, t in ready_data if t > current_time)
            current_time = next_t
            continue
        
        # For WISE: choose batch type based on critical weight
        if is_wise_arch:
            def type_choice_key(o: "QCCDOperationBase"):
                return (critical_weight.get(o, 0.0), -topo_pos.get(o, 0))
            
            first_op = max(candidates, key=type_choice_key)
            chosen_type = type(first_op)
        else:
            chosen_type = None
        
        # Build batch
        batch: List["QCCDOperationBase"] = []
        used_components: Set["QCCDComponent"] = set()
        
        candidates_sorted = sorted(
            candidates,
            key=lambda o: (-critical_weight.get(o, 0.0), topo_pos.get(o, 0)),
        )
        
        for op in candidates_sorted:
            if is_wise_arch and not isinstance(op, chosen_type):
                continue
            if any(comp in used_components for comp in op.involved_components):
                continue
            batch.append(op)
            for comp in op.involved_components:
                used_components.add(comp)
        
        if not batch:
            future_times = [t for _, t in ready_data if t > current_time]
            if not future_times:
                break
            current_time = min(future_times)
            continue
        
        # Schedule this batch
        for op in batch:
            end_t = current_time + op.calculate_time()
            operation_end_times[op] = end_t
            scheduled.add(op)
            
            for comp in op.involved_components:
                component_busy_until[comp] = end_t
            
            for succ in happens_before.get(op, []):
                if succ in earliest_start:
                    earliest_start[succ] = max(earliest_start[succ], end_t)
        
        batch_end = max(operation_end_times[o] for o in batch)
        
        if is_wise_arch:
            arch_busy_until = batch_end
        
        # Update active ops
        active_ops = [o for o in active_ops if operation_end_times.get(o, 0.0) > current_time]
        
        # Record in schedule
        time_schedule[current_time] = ParallelOperationGroup(
            operations=batch,
            active_operations=list(active_ops),
            start_time=current_time,
        )
        active_ops.extend(batch)
        
        # Advance time
        if is_wise_arch:
            current_time = arch_busy_until
        else:
            future_t = [t for t in component_busy_until.values() if t > current_time]
            if future_t:
                current_time = min(future_t)
            else:
                break
    
    return time_schedule


def schedule_operations_with_barriers(
    operations: Sequence["QCCDOperationBase"],
    barriers: List[int],
    is_wise_arch: bool = False,
) -> Dict[float, ParallelOperationGroup]:
    """Schedule operations with explicit barrier positions.
    
    Parameters
    ----------
    operations : Sequence[QCCDOperationBase]
        Operations to schedule.
    barriers : List[int]
        Indices where barriers should be inserted.
    is_wise_arch : bool
        Whether to use WISE scheduling within each segment.
        
    Returns
    -------
    Dict[float, ParallelOperationGroup]
        Scheduled operation groups.
    """
    time_schedule: Dict[float, ParallelOperationGroup] = {}
    barrier_positions = [0] + list(barriers) + [len(operations)]
    t = 0.0
    
    for start, end in zip(barrier_positions[:-1], barrier_positions[1:]):
        segment = list(operations)[start:end]
        segment_schedule = schedule_operations_wise(segment, is_wise_arch=is_wise_arch)
        
        for s, group in segment_schedule.items():
            group.start_time = s + t
            time_schedule[s + t] = group
        
        if time_schedule:
            t = max(
                group.start_time + group.duration
                for group in time_schedule.values()
            )
    
    return time_schedule


# =============================================================================
# Dephasing Analysis
# =============================================================================

def calculate_dephasing_from_idling(
    operations: Sequence["QCCDOperationBase"],
    is_wise_arch: bool = False,
) -> Dict["Ion", List[Tuple["QCCDOperationBase", float]]]:
    """Compute dephasing per ion based on idle intervals.
    
    Uses the same scheduling logic as schedule_operations_wise to
    compute when ions are idle between quantum operations.
    
    Parameters
    ----------
    operations : Sequence[QCCDOperationBase]
        Operations to analyze.
    is_wise_arch : bool
        Whether to use WISE scheduling.
        
    Returns
    -------
    Dict[Ion, List[Tuple[QCCDOperationBase, float]]]
        For each ion, list of (operation, dephasing_fidelity) pairs
        where the operation ends an idle interval.
    """
    from .architecture import Ion
    from .operations import QubitOperation as QuantumOperationBase
    
    # Get the schedule
    schedule = schedule_operations_wise(operations, is_wise_arch=is_wise_arch)
    if not schedule:
        return {}
    
    # Collect all components and ions
    all_components: List["QCCDComponent"] = []
    for op in operations:
        for c in op.involved_components:
            if c not in all_components:
                all_components.append(c)
    
    all_ions: List["Ion"] = [c for c in all_components if isinstance(c, Ion)]
    
    # Build timeline events
    events: List[Tuple[float, "QCCDOperationBase", str]] = []
    
    for t_start, group in schedule.items():
        for op in group.operations:
            t_end = t_start + op.calculate_time()
            events.append((t_start, op, "start"))
            events.append((t_end, op, "end"))
    
    # Sort by time, process "end" before "start" at same time
    events.sort(key=lambda e: (e[0], 0 if e[2] == "end" else 1))
    
    # Track idling
    ion_idling_times: Dict["Ion", List[Tuple[float, float]]] = {
        ion: [(0.0, 0.0)] for ion in all_ions
    }
    ion_idling_operations: Dict["Ion", List["QCCDOperationBase"]] = {
        ion: [] for ion in all_ions
    }
    idling_ions: Set["Ion"] = set(all_ions)
    
    # Process events
    for t, op, kind in events:
        # Only quantum operations affect idle status
        if not isinstance(op, QuantumOperationBase):
            continue
        
        ions = getattr(op, "_ions", [])
        
        if kind == "start":
            for ion in ions:
                if ion in idling_ions:
                    idling_ions.remove(ion)
                    if ion in ion_idling_times and ion_idling_times[ion]:
                        idle_start, _ = ion_idling_times[ion][-1]
                        idle_duration = t - idle_start
                        if idle_duration > 0.0:
                            ion_idling_times[ion][-1] = (idle_start, idle_duration)
                            ion_idling_operations[ion].append(op)
                        else:
                            ion_idling_times[ion].pop()
        
        elif kind == "end":
            for ion in ions:
                if ion not in idling_ions:
                    idling_ions.add(ion)
                    ion_idling_times[ion].append((t, 0.0))
    
    # Convert to dephasing fidelities
    ion_dephasing: Dict["Ion", List[Tuple["QCCDOperationBase", float]]] = {
        ion: [] for ion in all_ions
    }
    
    for ion, idling_times in ion_idling_times.items():
        num_completed = min(
            len(idling_times) - 1,
            len(ion_idling_operations.get(ion, [])),
        )
        for k in range(num_completed):
            idle_start, idle_duration = idling_times[k]
            op_at_end = ion_idling_operations[ion][k]
            if idle_duration > 0.0:
                deph = calculate_dephasing_fidelity(idle_duration)
                ion_dephasing[ion].append((op_at_end, deph))
    
    return ion_dephasing


# =============================================================================
# Barrier-Aware Scheduler
# =============================================================================

class BarrierAwareScheduler(_CoreBarrierAwareScheduler):
    """WISE-aware barrier scheduler.

    Thin subclass of :class:`core.operations.BarrierAwareScheduler`
    that defaults the inner delegate to :class:`WISEBatchScheduler`.
    """

    def __init__(
        self,
        delegate: Optional[BatchScheduler] = None,
        name: str = "barrier_aware_scheduler",
    ) -> None:
        super().__init__(
            delegate=delegate or WISEBatchScheduler(),
            name=name,
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Schedulers
    "WISEBatchScheduler",
    "WISECriticalPathScheduler",
    "BarrierAwareScheduler",
    # Scheduling functions
    "schedule_operations_wise",
    "schedule_operations_with_barriers",
    "build_happens_before_dag",
    # Dephasing
    "calculate_dephasing_fidelity",
    "calculate_dephasing_from_idling",
    # Data structures
    "ParallelOperationGroup",
    # Constants
    "T2_DEPHASING_TIME",
]
