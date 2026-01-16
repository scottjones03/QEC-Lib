
import numpy as np
from typing import (
    Sequence,
    List,
    Tuple,
    Mapping,
    Set,
    Dict,
)
from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *
from collections import defaultdict, deque


def paralleliseOperationsSimple(
    operationSequence: Sequence[Operation],
) -> Sequence[ParallelOperation]:
    operationSequence = list(operationSequence)
    parallelOperationsSequence: List[ParallelOperation] = []
    if not operationSequence:
        return parallelOperationsSequence
    while operationSequence:
        parallelOperations = [operationSequence.pop(0)]
        involvedComponents: Set[QCCDComponent] = set(
            parallelOperations[0].involvedComponents
        )
        for op in operationSequence:
            components = op.involvedComponents
            if involvedComponents.isdisjoint(components):
                parallelOperations.append(op)
            involvedComponents = involvedComponents.union(components)
        for op in parallelOperations[1:]:
            operationSequence.remove(op)
        parallelOperation = ParallelOperation.physicalOperation(parallelOperations, [])
        parallelOperationsSequence.append(parallelOperation)
    return parallelOperationsSequence

def calculateDephasingFidelity(time: float) -> None:

    # # log(error) = m*log(delay)+c
    # m = (np.log(0.008)-np.log(0.00001))/((np.log(1)-np.log(0.01)))
    # c = np.log(0.00001)-m*np.log(0.01)
    # dephasingInFidelity = np.exp(m*np.log(time)+c)
    # # return 0.99999
    # return 1-dephasingInFidelity
    T2 = 2.2 # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    return 1 - (1-np.exp(-time/T2))/2  # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330



def happensBeforeForOperations(
    operationSequence: Sequence[Operation], all_components: List[QCCDComponent]
) -> Tuple[Dict[Operation, List[Operation]], Sequence[Operation]]:
     # Step 1: Create a happens-before relation graph using adjacency list (DAG)
    happens_before: Dict[Operation, List[Operation]] = {op: [] for op in operationSequence} # Adjacency list for DAG
    indegree: Dict[Operation, int] = {op: 0 for op in operationSequence}  # Track number of dependencies for each operation
    operations_by_component: Dict[QCCDComponent, List[Operation]] = {c: [] for c in all_components}  # Track operations by QCCDComponent
    
    # Build the happens-before relation based on the components involved
    for op in operationSequence:
        for component in set(op.involvedComponents):
            for prev_op in operations_by_component[component]:
                # There is a happens-before relation (prev_op happens before op)
                if op not in happens_before[prev_op]:
                    happens_before[prev_op].append(op)
                    indegree[op] += 1
            operations_by_component[component].append(op)
    
    # Topologically sort the operations using Kahn's algorithm (BFS)
    zero_indegree_queue = deque([op for op in operationSequence if indegree[op] == 0])
    topologically_sorted_ops: List[Operation] = []
    while zero_indegree_queue:
        op = zero_indegree_queue.popleft()
        topologically_sorted_ops.append(op)
        for neighbor in happens_before[op]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                zero_indegree_queue.append(neighbor)
    return happens_before, topologically_sorted_ops

def paralleliseOperations(
    operationSequence: Sequence[Operation],
    isWISEArch: bool = False,
) -> Mapping[float, ParallelOperation]:
    # Collect all components in this slice
    all_components: List[QCCDComponent] = []
    for op in operationSequence:
        for c in op.involvedComponents:
            if c not in all_components:
                all_components.append(c)

    # Build happens-before DAG and a topological order
    happens_before, topo_order = happensBeforeForOperations(
        operationSequence, all_components
    )
    topo_order = list(topo_order)

    # Predecessor map: preds[op] = list of operations that must happen before op
    preds: Dict[Operation, List[Operation]] = {op: [] for op in operationSequence}
    for op, succs in happens_before.items():
        for nxt in succs:
            if nxt in preds:          # ignore edges leaving this slice
                preds[nxt].append(op)

    # Stable topological index for tie-breaking
    topo_pos: Dict[Operation, int] = {op: i for i, op in enumerate(topo_order)}

    # 3) Compute "critical weight" = longest total duration from op to any leaf
    #    in the happens-before DAG.
    critical_weight: Dict[Operation, float] = {}

    # Traverse topological order in reverse so successors are processed first.
    for op in reversed(topo_order):
        succs = happens_before.get(op, ())
        if succs:
            # op time + max of successor critical weights
            critical_weight[op] = op.operationTime() + max(
                critical_weight[s] for s in succs
            )
        else:
            # leaf: just its own duration
            critical_weight[op] = op.operationTime()

    # --- Scheduling state ---
    time_schedule: Dict[float, List[Operation]] = defaultdict(list)
    operation_end_times: Dict[Operation, float] = {}
    component_busy_until: Dict[QCCDComponent, float] = {
        c: 0.0 for c in all_components
    }

    # For dependency timing: earliest time an op may start because of its predecessors
    earliest_start: Dict[Operation, float] = {op: 0.0 for op in operationSequence}

    # For WISE: global batch barrier (no new batch until this time)
    arch_busy_until: float = 0.0 if isWISEArch else 0.0

    current_time = 0.0
    scheduled: Set[Operation] = set()
    active_ops: List[Operation] = []

    def all_preds_scheduled(op: Operation) -> bool:
        return all(p in scheduled for p in preds[op])

    # Main scheduling loop
    while len(scheduled) < len(operationSequence):
        # Frontier: unscheduled ops whose predecessors are all scheduled
        remaining_ops = [op for op in topo_order if op not in scheduled]
        frontier_ops = [op for op in remaining_ops if all_preds_scheduled(op)]

        if not frontier_ops:
            # Nothing schedulable: break to avoid infinite loop
            # (shouldn't normally happen unless there are external deps)
            break

        # Compute earliest feasible start time for each frontier op
        ready_data = []  # list of (op, earliest_possible_start)
        min_start = float("inf")

        for op in frontier_ops:
            comp_ready = max(
                component_busy_until[comp] for comp in op.involvedComponents
            )
            dep_ready = earliest_start[op]
            if isWISEArch:
                start_t = max(comp_ready, dep_ready, arch_busy_until)
            else:
                start_t = max(comp_ready, dep_ready)
            ready_data.append((op, start_t))
            if start_t < min_start:
                min_start = start_t

        # Set current_time to the earliest we can start anything
        current_time = min_start

        # Candidates that are actually ready at current_time
        candidates = [op for op, t in ready_data if t == current_time]
        if not candidates:
            # No-one actually starts exactly at min_start, so jump to the next
            # earliest possible start and retry.
            next_t = min(t for _, t in ready_data if t > current_time)
            current_time = next_t
            continue

        # --- Choose batch type (for WISE) and maximal non-conflicting subset ---
        if isWISEArch:
            # First choose a "leading" op: high fanout, then short duration, then topo order
            def type_choice_key(o: Operation):
                return (critical_weight[o], -topo_pos[o])

            firstOp = max(candidates, key=type_choice_key)
            chosen_type = type(firstOp)
        else:
            chosen_type = None  # unrestricted

        batch: List[Operation] = []
        used_components: Set[QCCDComponent] = set()

        # Sort candidates by fanout then topo order for greedy packing
        candidates_sorted = sorted(
            candidates,
            key=lambda o: (-critical_weight[o], topo_pos[o])
        )

        for op in candidates_sorted:
            if isWISEArch and not isinstance(op, chosen_type):
                continue
            # Check component conflicts
            if any(comp in used_components for comp in op.involvedComponents):
                continue
            batch.append(op)
            for comp in op.involvedComponents:
                used_components.add(comp)

        if not batch:
            # Fallback: advance to the next possible time (should be rare)
            future_times = [t for _, t in ready_data if t > current_time]
            if not future_times:
                break
            current_time = min(future_times)
            continue

        # --- Schedule this batch at current_time ---
        for op in batch:
            end_t = current_time + op.operationTime()
            operation_end_times[op] = end_t
            scheduled.add(op)

            # Update component busy times
            for comp in op.involvedComponents:
                component_busy_until[comp] = end_t

            # Update successors' earliest start times
            for succ in happens_before.get(op, ()):
                if succ in earliest_start:
                    earliest_start[succ] = max(earliest_start[succ], end_t)

        batch_end = max(operation_end_times[o] for o in batch)

        if isWISEArch:
            # Enforce global barrier until all ops in this batch are finished
            arch_busy_until = batch_end

        # Remove ops that just finished from the "currently active" list
        active_ops = [
            o for o in active_ops if operation_end_times[o] > current_time
        ]
        
        # Record in time_schedule as a ParallelOperation
        time_schedule[current_time] = ParallelOperation.physicalOperation(
            batch, active_ops
        )
        active_ops.extend(batch)

        

        # Advance time
        if isWISEArch:
            # Next batch cannot start until previous batch fully completed
            current_time = arch_busy_until
        else:
            # Non-WISE: can start as soon as any component becomes free
            future_t = [t for t in component_busy_until.values() if t > current_time]
            if future_t:
                current_time = min(future_t)
            else:
                # All done
                break

    # Convert list-of-ops schedule to expected mapping
    return dict(time_schedule)

def paralleliseOperationsWithBarriers(
    operationSequence: Sequence[Operation],
    barriers: List[int],
    isWiseArch: bool = False
) -> Mapping[float, ParallelOperation]:
    time_schedule = {}
    barriers.insert(0, 0)
    barriers.append(len(operationSequence))
    t=0.0
    for start, barrier in zip(barriers[:-1], barriers[1:]):
        for s, os in paralleliseOperations(operationSequence[start: barrier], isWISEArch=isWiseArch).items():
            time_schedule[s+t] = os 
        t = max(x+max(y.operationTime() for y in ys.operations) for x, ys in time_schedule.items())
    return time_schedule
from typing import Sequence, Mapping, Dict, List, Tuple, Set

def calculateDephasingFromIdling(
    operationSequence: Sequence[Operation],
    isWISEArch: bool = False,
) -> Mapping[Ion, Sequence[Tuple[Operation, float]]]:
    """
    Compute dephasing per ion, based on idling intervals between QubitOperations.

    Uses the *same scheduling logic* as `paralleliseOperations`, so the
    timing is consistent with the actual parallel execution (including WISE
    batch behaviour).
    """
    # 1) Get the scheduled parallel operations with start times
    schedule = paralleliseOperations(operationSequence, isWISEArch=isWISEArch)
    if not schedule:
        return {}

    # 2) Collect all ions from the components
    all_components: List[QCCDComponent] = []
    for op in operationSequence:
        for c in op.involvedComponents:
            if c not in all_components:
                all_components.append(c)

    all_ions: List[Ion] = [c for c in all_components if isinstance(c, Ion)]

    # 3) Build a list of (time, op, kind) events from the schedule
    #    kind = "start" or "end"
    events: List[Tuple[float, Operation, str]] = []

    for t_start, parOp in schedule.items():
        # paralleliseOperations currently stores ParallelOperation objects,
        # but we fall back to using it directly if it is a list.
        ops = getattr(parOp, "operations", parOp)
        for op in ops:
            t_end = t_start + op.operationTime()
            events.append((t_start, op, "start"))
            events.append((t_end,   op, "end"))

    # Sort by time; for identical times, process "end" before "start"
    events.sort(key=lambda e: (e[0], 0 if e[2] == "end" else 1))

    # 4) Prepare bookkeeping for idling
    #    ion_idling_times[ion] = list of (idle_start_time, idle_duration)
    ion_idling_times: Dict[Ion, List[Tuple[float, float]]] = {
        ion: [(0.0, 0.0)] for ion in all_ions
    }
    #    ion_idling_operations[ion] = QubitOperation that ends each idle interval
    ion_idling_operations: Dict[Ion, List[QubitOperation]] = {ion: [] for ion in all_ions}

    # Set of ions currently idle
    idling_ions: Set[Ion] = set(all_ions)

    # 5) Sweep through the timeline, updating idling intervals on QubitOperation start/end
    for t, op, kind in events:
        # Only QubitOperations determine "idle" vs "active" for dephasing
        if not isinstance(op, QubitOperation):
            continue

        if kind == "start":
            # Ions involved in this gate stop idling at time t
            for ion in op.ions:
                if ion in idling_ions:
                    idling_ions.remove(ion)
                    idle_start, _ = ion_idling_times[ion][-1]
                    idle_duration = t - idle_start
                    if idle_duration > 0.0:
                        # Close out this idle interval
                        ion_idling_times[ion][-1] = (idle_start, idle_duration)
                        ion_idling_operations[ion].append(op)
                    else:
                        # Zero-length idle interval: discard
                        ion_idling_times[ion].pop()

        elif kind == "end":
            # Ions involved in this gate become idle again from time t
            for ion in op.ions:
                if ion not in idling_ions:
                    idling_ions.add(ion)
                    ion_idling_times[ion].append((t, 0.0))

    # 6) Convert idle durations into dephasing fidelities
    #    (ignore the final open idle interval for each ion, like your original code)
    ion_dephasing: Dict[Ion, List[Tuple[Operation, float]]] = {ion: [] for ion in all_ions}

    for ion, idling_times in ion_idling_times.items():
        # Number of completed idle intervals = number of recorded end QubitOperations
        num_completed = min(len(idling_times) - 1, len(ion_idling_operations[ion]))
        for k in range(num_completed):
            idle_start, idle_duration = idling_times[k]
            op_at_end_of_idle = ion_idling_operations[ion][k]
            if idle_duration > 0.0:
                deph = calculateDephasingFidelity(idle_duration)
                ion_dephasing[ion].append((op_at_end_of_idle, deph))

    return ion_dephasing