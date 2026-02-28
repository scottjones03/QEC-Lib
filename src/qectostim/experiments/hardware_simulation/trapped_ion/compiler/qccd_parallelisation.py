
import numpy as np
from typing import (
    Sequence,
    List,
    Tuple,
    Mapping,
    Optional,
    Set,
    Dict,
)
from ..utils.qccd_nodes import *
from ..utils.qccd_operations import *
from ..utils.qccd_operations_on_qubits import *
from ..utils.physics import DEFAULT_CALIBRATION, DEFAULT_FIDELITY_MODEL
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

def calculateDephasingFidelity(time: float) -> float:
    """Compute dephasing fidelity for a given duration.

    Delegates to :data:`DEFAULT_FIDELITY_MODEL` so the formula lives in
    one place (``physics.py``).
    """
    return DEFAULT_FIDELITY_MODEL.dephasing_fidelity(time)


# ---------------------------------------------------------------------------
# Rotation reordering for batching (RX/RY between MS rounds)
# ---------------------------------------------------------------------------

# Type priority when grouping rotations for batching.
# Lower value = scheduled first.  We group all ops of one rotation type
# across different ions before moving to the next type, which maximises
# parallelism on WISE (where only one QubitOperation type can execute at
# a time) and also helps the augmented-grid scheduler.
_ROTATION_TYPE_PRIORITY: Dict[type, int] = {
    XRotation: 0,
    YRotation: 1,
}


def reorder_rotations_for_batching(
    operations: List[Operation],
) -> List[Operation]:
    """Reorder single-qubit rotations between MS-round boundaries to group
    by type, maximising same-type parallel batches.

    **Boundary rules**:

    * **MS gates** (``TwoQubitMSGate``) and transport operations are
      *hard barriers* — no rotation may cross them.  All rotations
      before an MS round must be emitted before the MS round starts,
      and all rotations after one MS round but before the next stay in
      their own window.
    * **Measurement / QubitReset** are *per-ion barriers* — a rotation
      on ion *X* cannot be moved past a measurement or reset on ion *X*,
      but rotations on *other* ions may freely reorder across that
      measurement/reset.  This allows batching optimisations while
      preserving correct per-qubit sequencing.

    Within each *rotation window* (delimited by the above barrier rules),
    rotations on **different ions** are rearranged so that all ops of
    one type (e.g. all ``XRotation``) come before all ops of the next
    type (e.g. all ``YRotation``).

    **Per-ion ordering is preserved** — RX and RY on the same ion are
    never swapped relative to each other, because they do not commute.

    Parameters
    ----------
    operations : list[Operation]
        The operation sequence (may contain any mix of operation types).

    Returns
    -------
    list[Operation]
        A new list with the same operations, potentially reordered within
        rotation windows.
    """
    if not operations:
        return list(operations)

    def _is_rotation(op: Operation) -> bool:
        return type(op) in _ROTATION_TYPE_PRIORITY

    def _is_hard_barrier(op: Operation) -> bool:
        """MS gates and transport ops are hard barriers for ALL rotations."""
        if isinstance(op, TwoQubitMSGate):
            return True
        # Non-QubitOperation = transport op = hard barrier
        if not isinstance(op, QubitOperation):
            return True
        return False

    def _is_per_ion_barrier(op: Operation) -> bool:
        """Measurement/Reset are barriers only for the involved ion."""
        return isinstance(op, (Measurement, QubitReset))

    result: List[Operation] = []
    # Window: rotations that can be freely reordered, plus per-ion
    # barriers that sit between them.  We track which ions are
    # barrier-constrained.
    window_rotations: List[Operation] = []
    # Per-ion barriers accumulated: (position_in_stream, op)
    # When we encounter a per-ion barrier, we flush rotations for
    # that specific ion, emit the barrier, then continue.
    pending_per_ion_barriers: List[Operation] = []

    def _flush_all() -> None:
        """Drain all pending rotations + per-ion barriers into result."""
        if not window_rotations and not pending_per_ion_barriers:
            return
        _emit_reordered(window_rotations, pending_per_ion_barriers)
        window_rotations.clear()
        pending_per_ion_barriers.clear()

    def _emit_reordered(
        rotations: List[Operation],
        per_ion_barriers: List[Operation],
    ) -> None:
        """Emit rotations type-grouped, interleaving per-ion barriers
        at the correct points."""
        if not rotations and not per_ion_barriers:
            return
        if not rotations:
            result.extend(per_ion_barriers)
            return
        if not per_ion_barriers:
            # Pure rotation window — group by type
            _emit_type_grouped(rotations)
            return

        # Mixed: rotations + per-ion barriers.
        # Strategy: build per-ion FIFO queues that include both
        # rotations and barriers.  Group rotations by type across
        # ions, but when a barrier is at the front of an ion's queue,
        # it must be emitted before any more rotations on that ion.
        #
        # Per-ion barriers whose ion has NO rotations in this window
        # are "unrelated" — they are emitted at the boundary between
        # rotation type groups (preserving their relative order among
        # the window's ops but NOT blocking rotations on other ions).
        rotation_ion_ids: Set[int] = {id(op.ions[0]) for op in rotations}

        # Separate barriers into related (ion has rotations) vs unrelated
        related_barriers: List[Operation] = []
        unrelated_barriers: List[Operation] = []
        for barrier in per_ion_barriers:
            barrier_ion_ids = {id(ion) for ion in barrier.ions}
            if barrier_ion_ids & rotation_ion_ids:
                related_barriers.append(barrier)
            else:
                unrelated_barriers.append(barrier)

        if not related_barriers:
            # No barriers constrain any rotation ion — group rotations
            # by type and emit unrelated barriers at the end
            _emit_type_grouped(rotations)
            result.extend(unrelated_barriers)
            return

        # Build per-ion queues with rotations + related barriers
        all_ops = rotations + related_barriers
        # Reconstruct original order to build per-ion queues
        _orig_pos: Dict[int, int] = {id(op): i for i, op in enumerate(operations)}
        all_ops.sort(key=lambda op: _orig_pos.get(id(op), 0))

        ion_queues: Dict[int, deque] = defaultdict(deque)
        for op in all_ops:
            if _is_rotation(op):
                ion_key = id(op.ions[0])
                ion_queues[ion_key].append(op)
            elif _is_per_ion_barrier(op):
                for ion in op.ions:
                    ion_key = id(ion)
                    if ion_key in rotation_ion_ids:
                        ion_queues[ion_key].append(op)

        # Track which barriers have been emitted
        emitted_barriers: Set[int] = set()
        # Track whether unrelated barriers have been emitted
        unrelated_emitted = False

        while any(q for q in ion_queues.values()):
            # First, emit per-ion barriers at the front of any queue
            emitted_any_barrier = True
            while emitted_any_barrier:
                emitted_any_barrier = False
                for ion_key in list(ion_queues.keys()):
                    q = ion_queues[ion_key]
                    while q and _is_per_ion_barrier(q[0]):
                        barrier_op = q.popleft()
                        if id(barrier_op) not in emitted_barriers:
                            result.append(barrier_op)
                            emitted_barriers.add(id(barrier_op))
                        emitted_any_barrier = True

            # Group rotations by type
            best_tp: Optional[int] = None
            for q in ion_queues.values():
                if q and _is_rotation(q[0]):
                    tp = _ROTATION_TYPE_PRIORITY.get(type(q[0]), 99)
                    if best_tp is None or tp < best_tp:
                        best_tp = tp
            if best_tp is None:
                break

            # Emit unrelated barriers between type groups (once)
            if not unrelated_emitted and best_tp > 0:
                result.extend(unrelated_barriers)
                unrelated_emitted = True

            for ion_key in list(ion_queues.keys()):
                q = ion_queues[ion_key]
                while (
                    q
                    and _is_rotation(q[0])
                    and _ROTATION_TYPE_PRIORITY.get(type(q[0]), 99) == best_tp
                ):
                    result.append(q.popleft())

        # Emit unrelated barriers if not yet emitted
        if not unrelated_emitted:
            result.extend(unrelated_barriers)

        # Safety: emit any remaining ops
        for q in ion_queues.values():
            for op in q:
                if id(op) not in emitted_barriers:
                    result.append(op)
                    emitted_barriers.add(id(op))

    def _emit_type_grouped(rotations: List[Operation]) -> None:
        """Emit rotations grouped by type, preserving per-ion order."""
        if len(rotations) <= 1:
            result.extend(rotations)
            return
        ion_queues: Dict[int, deque] = defaultdict(deque)
        for op in rotations:
            ion_key = id(op.ions[0])
            ion_queues[ion_key].append(op)

        while any(q for q in ion_queues.values()):
            best_tp: Optional[int] = None
            for q in ion_queues.values():
                if q:
                    tp = _ROTATION_TYPE_PRIORITY.get(type(q[0]), 99)
                    if best_tp is None or tp < best_tp:
                        best_tp = tp
            if best_tp is None:
                break
            for ion_key in list(ion_queues.keys()):
                q = ion_queues[ion_key]
                while q and _ROTATION_TYPE_PRIORITY.get(type(q[0]), 99) == best_tp:
                    result.append(q.popleft())

        for q in ion_queues.values():
            result.extend(q)

    # --- Walk operations ---
    for op in operations:
        if _is_rotation(op):
            window_rotations.append(op)
        elif _is_hard_barrier(op):
            _flush_all()
            result.append(op)
        elif _is_per_ion_barrier(op):
            # Measurement/Reset: only barrier for the specific ion(s)
            pending_per_ion_barriers.append(op)
        else:
            # Other QubitOperation types (shouldn't normally occur)
            _flush_all()
            result.append(op)
    _flush_all()

    return result


def happensBeforeForOperations(
    operationSequence: Sequence[Operation], all_components: List[QCCDComponent],
    epoch_mode: str = "edge",
) -> Tuple[Dict[Operation, List[Operation]], Sequence[Operation]]:
     # Step 1: Create a happens-before relation graph using adjacency list (DAG)
    happens_before: Dict[Operation, List[Operation]] = {op: [] for op in operationSequence} # Adjacency list for DAG
    indegree: Dict[Operation, int] = {op: 0 for op in operationSequence}  # Track number of dependencies for each operation
    operations_by_component: Dict[QCCDComponent, List[Operation]] = {c: [] for c in all_components}  # Track operations by QCCDComponent

    def _add_edge(src: Operation, dst: Operation) -> None:
        """Add a happens-before edge src→dst (idempotent)."""
        if dst not in happens_before[src]:
            happens_before[src].append(dst)
            indegree[dst] += 1

    # Build the happens-before relation based on the components involved.
    for op in operationSequence:
        for component in set(op.involvedComponents):
            for prev_op in operations_by_component[component]:
                # There is a happens-before relation (prev_op happens before op)
                _add_edge(prev_op, op)
            operations_by_component[component].append(op)

    # Step 1b: Tick-epoch edges (cycle-safe).
    # For same-ion QubitOperations in different tick epochs, add edges
    # ensuring earlier-epoch ops happen before later-epoch ones.
    # Only add an edge when it does NOT create a cycle (checked via BFS
    # reachability from dst→src in the current DAG).
    ion_to_qubit_ops: Dict[int, List[Operation]] = defaultdict(list)
    for op in operationSequence:
        if isinstance(op, QubitOperation) and hasattr(op, '_tick_epoch'):
            for ion in op.ions:
                ion_to_qubit_ops[id(ion)].append(op)

    def _has_path(src: Operation, dst: Operation) -> bool:
        """BFS: is there a directed path from *src* to *dst*?"""
        visited: Set[Operation] = set()
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node is dst:
                return True
            if node in visited:
                continue
            visited.add(node)
            for neighbour in happens_before.get(node, []):
                if neighbour not in visited:
                    queue.append(neighbour)
        return False

    for _ion_id, _ops in ion_to_qubit_ops.items():
        _sorted = sorted(
            _ops,
            key=lambda o: (o._tick_epoch, getattr(o, '_stim_origin', 0)),
        )
        for _i in range(len(_sorted) - 1):
            _a, _b = _sorted[_i], _sorted[_i + 1]
            if _a._tick_epoch < _b._tick_epoch:
                # Only add if dst→src path does not exist (avoids cycle).
                if not _has_path(_b, _a):
                    _add_edge(_a, _b)

    # Step 1c: Epoch-barrier / hybrid modes.
    if epoch_mode in ("barrier", "hybrid"):
        epoch_to_ops: Dict[int, List[Operation]] = defaultdict(list)
        for op in operationSequence:
            if isinstance(op, QubitOperation) and hasattr(op, '_tick_epoch'):
                epoch_to_ops[op._tick_epoch].append(op)

        sorted_epochs = sorted(epoch_to_ops.keys())
        for _ei in range(len(sorted_epochs) - 1):
            curr_ops = epoch_to_ops[sorted_epochs[_ei]]
            next_ops = epoch_to_ops[sorted_epochs[_ei + 1]]

            if epoch_mode == "hybrid":
                # Only insert barrier at measurement / reset epoch boundaries
                has_meas_reset = any(
                    isinstance(op, (Measurement, QubitReset))
                    for op in next_ops
                )
                if not has_meas_reset:
                    continue

            for a in curr_ops:
                for b in next_ops:
                    if not _has_path(b, a):
                        _add_edge(a, b)

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
    epoch_mode: str = "edge",
) -> Mapping[float, ParallelOperation]:
    # Collect all components in this slice
    all_components: List[QCCDComponent] = []
    for op in operationSequence:
        for c in op.involvedComponents:
            if c not in all_components:
                all_components.append(c)

    # Build happens-before DAG and a topological order
    happens_before, topo_order = happensBeforeForOperations(
        operationSequence, all_components, epoch_mode=epoch_mode
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
    arch_busy_until: float = 0.0

    # For WISE type commitment: track the most recent chosen type so
    # the scheduler can bias toward type continuity (fewer type switches).
    prev_chosen_type: Optional[type] = None

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
        frontier_set = set(candidates)

        if isWISEArch:
            # WISE: control pulses are multiplexed — only same-type
            # QubitOperations can execute simultaneously.  Transport ops
            # (Split, Move, Merge, etc.) are type-agnostic and always
            # allowed.
            #
            # Hybrid type selection: prefer types with high
            #   (count_in_frontier + near_ready_count) * max_critical_weight
            # so that large same-type batches are favoured over a single
            # high-critical-weight op of a different type.
            qubit_candidates = [
                op for op in candidates if isinstance(op, QubitOperation)
            ]
            if qubit_candidates:
                type_groups: Dict[type, List[Operation]] = defaultdict(list)
                for op in qubit_candidates:
                    type_groups[type(op)].append(op)

                # Phase 2b: lookahead — count near-ready ops of each type.
                # An op is "near-ready" if it's not yet in the frontier
                # but ALL its predecessors are either scheduled or in the
                # current frontier.
                def _near_ready_of_type(op_type: type) -> int:
                    count = 0
                    for op in remaining_ops:
                        if op in frontier_set or op in scheduled:
                            continue
                        if not isinstance(op, QubitOperation):
                            continue
                        if not isinstance(op, op_type):
                            continue
                        if all(p in scheduled or p in frontier_set for p in preds[op]):
                            count += 1
                    return count

                def _type_score(ops: List[Operation]) -> tuple:
                    count = len(ops)
                    max_cw = max(critical_weight[o] for o in ops)
                    lookahead = _near_ready_of_type(type(ops[0]))
                    # Type continuity bonus: bias toward staying with the
                    # previous type to reduce type-switch overhead.
                    # A 2x multiplier makes the scheduler prefer continuity
                    # unless a different type has significantly more ops.
                    continuity = 2.0 if (
                        prev_chosen_type is not None
                        and type(ops[0]) is prev_chosen_type
                    ) else 1.0
                    return (
                        (count + lookahead) * max_cw * continuity,
                        count + lookahead,
                    )

                chosen_type = max(
                    type_groups, key=lambda t: _type_score(type_groups[t])
                )
                prev_chosen_type = chosen_type
            else:
                # Only transport ops in frontier — no type constraint
                chosen_type = None

            # Phase 2a: aggressive deferred scheduling — hold back ALL
            # non-chosen-type QubitOperations.  With type-aware sub-barriers
            # from the router, each segment is typically pure-type, so
            # deferral rarely activates.  The stall guard prevents deadlock
            # in mixed-type segments.
            if chosen_type is not None:
                deferred = [
                    op for op in candidates
                    if isinstance(op, QubitOperation)
                    and not isinstance(op, chosen_type)
                ]
                candidates = [op for op in candidates if op not in deferred]
                # Stall guard: if deferral emptied candidates, undo
                if not candidates:
                    candidates = deferred
                    deferred = []
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
            if isWISEArch and chosen_type is not None:
                # WISE type constraint: QubitOperations must match the
                # chosen type.  Transport ops are always allowed.
                if isinstance(op, QubitOperation) and not isinstance(op, chosen_type):
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
    isWiseArch: bool = False,
    epoch_mode: str = "edge",
) -> Mapping[float, ParallelOperation]:
    time_schedule: Dict[float, ParallelOperation] = {}
    barriers = [0] + list(barriers) + [len(operationSequence)]
    t: float = 0.0
    for start, barrier in zip(barriers[:-1], barriers[1:]):
        seg_ops = operationSequence[start:barrier]
        if not seg_ops:
            continue
        seg_schedule = paralleliseOperations(
            seg_ops, isWISEArch=isWiseArch, epoch_mode=epoch_mode,
        )
        seg_end = t  # track this segment's max end-time incrementally
        for s, par_op in seg_schedule.items():
            key = s + t
            # Guard against floating-point key collisions: when an
            # operation has near-zero duration (e.g. 1e-20 reconfig),
            # the max-end-time offset *t* may not advance in float64,
            # causing the next segment's first entry to overwrite the
            # previous one.  Bump the key by a tiny epsilon until unique.
            while key in time_schedule:
                key += max(abs(key) * 1e-9, 1e-15)
            time_schedule[key] = par_op
            # Update segment end from this entry only (O(1) per entry)
            entry_end = key + max(
                op.operationTime() for op in par_op.operations
            )
            if entry_end > seg_end:
                seg_end = entry_end
            # Keep key as a lower bound too
            if key >= seg_end:
                seg_end = key + max(abs(key) * 1e-9, 1e-15)
        if seg_end > t:
            t = seg_end
    return time_schedule


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