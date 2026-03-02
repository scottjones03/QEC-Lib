
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


def _wise_parallel_pack(
    ops: List[Operation],
) -> List["ParallelOperation"]:
    """Pack operations into minimal parallel batches for WISE.

    Within a contiguous same-type section all operations are independent
    — the only constraint is that no two operations in the same batch
    may share a hardware component (trap, junction, ion).

    Unlike :func:`paralleliseOperationsSimple`, **input ordering is NOT
    a constraint**.  Operations are freely reordered to minimise the
    number of batches.

    Algorithm: **most-constrained-first greedy bin-packing**.

    1. Pre-compute component sets for every operation.
    2. Sort operations by *decreasing* number of involved components
       (most-constrained first) — ops that use more hardware are harder
       to place, so placing them first yields fewer total batches.
    3. Greedily assign each operation to the first batch whose used
       component set does not conflict, or start a new batch.

    This is the standard "first-fit decreasing" heuristic for bin-packing,
    adapted for set-intersection conflicts instead of size-based bins.
    """
    if not ops:
        return []

    # Pre-compute component sets and sort most-constrained first
    ops_with_comps = [(op, set(op.involvedComponents)) for op in ops]
    ops_with_comps.sort(key=lambda x: -len(x[1]))

    # First-fit decreasing into batches
    batches_data: List[Tuple[List[Operation], Set]] = []  # (ops, used_components)

    for op, comps in ops_with_comps:
        placed = False
        for batch_ops, used_components in batches_data:
            if comps.isdisjoint(used_components):
                batch_ops.append(op)
                used_components.update(comps)
                placed = True
                break
        if not placed:
            batches_data.append(([op], set(comps)))

    return [
        ParallelOperation.physicalOperation(batch_ops, [])
        for batch_ops, _ in batches_data
    ]


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
# Type-priority ordering for operation grouping
# ---------------------------------------------------------------------------

# Type priority when grouping operations for batching.
# Lower value = scheduled first.  We group all ops of one type across
# different ions before moving to the next type, which maximises
# parallelism on WISE (where only one QubitOperation type can execute at
# a time).
#
# ROTX before ROTY: within each non-MS window, all RX operations are
# emitted first, then all RY.  Barriers between type groups enforce
# the happens-before relation — no per-ion ordering is needed within
# a window.
_OP_TYPE_PRIORITY: Dict[type, int] = {
    QubitReset: 0,
    XRotation: 1,
    YRotation: 2,
    Measurement: 3,
}

# Backward compatibility alias
_ROTATION_TYPE_PRIORITY = _OP_TYPE_PRIORITY


# ---------------------------------------------------------------------------
# Phase-based barrier reordering (pre-routing, called from decompose_to_native)
# ---------------------------------------------------------------------------

def reorder_with_type_barriers(
    operations: List[Operation],
) -> Tuple[List[Operation], List[int]]:
    """Reorder native ops into barrier-separated type-homogeneous blocks.

    Implements the 5-step barrier algorithm:

    1. **Group MS gates by stim index** into contiguous blocks, barrier
       around each.
    2. **Group RESETs and MEASUREMENTs** by stim index between barriers.
    3. **Merge adjacent RESET / MEAS blocks** when only separated by
       rotations, pushing rotations to the correct side to respect the
       happens-before relation on each qubit.
    4. **Barrier around RESET and MEAS blocks**.
    5. **Group ROTX and ROTY** between barriers, barrier around each.

    Final pattern::

        RESET BLOCK | ROTX BLOCK | ROTY BLOCK |
        { MS BLOCK | ROTX BLOCK | ROTY BLOCK } × n |
        MEAS BLOCK | RESET BLOCK | ...

    Parameters
    ----------
    operations : list[Operation]
        Flat list of native ``QubitOperation``s from ``decompose_to_native``
        (no transport ops at this stage).

    Returns
    -------
    reordered : list[Operation]
        Operations reordered with type grouping.
    barriers : list[int]
        Barrier positions (indices into *reordered*) at block boundaries.
    """
    if not operations:
        return [], []

    # ── 1. Group MS gates by stim index into contiguous blocks ───────
    #
    # Build per-ion operation sequences to determine phase assignment.
    # Each unique _stim_origin with MS gates defines one MS layer.
    # Non-MS ops are assigned to the gap between the MS layers they
    # fall between *on each ion* (per-ion walk).

    ms_layer_keys: List[Tuple[int, int]] = []  # ordered unique keys
    ms_layer_set: Set[Tuple[int, int]] = set()
    for op in operations:
        if isinstance(op, TwoQubitMSGate):
            key = (
                getattr(op, '_tick_epoch', 0),
                getattr(op, '_stim_origin', 0),
            )
            if key not in ms_layer_set:
                ms_layer_set.add(key)
                ms_layer_keys.append(key)
    ms_layer_keys.sort()
    ms_layer_map: Dict[Tuple[int, int], int] = {
        k: i for i, k in enumerate(ms_layer_keys)
    }
    num_ms_layers = len(ms_layer_keys)

    if num_ms_layers == 0:
        # No MS gates — apply steps 2-5 directly
        result, barriers = _barrier_reorder_non_ms(operations)
        _tag_barrier_groups(result, barriers)
        return result, barriers

    # Per-ion operation sequences
    ion_ops: Dict[int, List[Tuple[int, Operation]]] = defaultdict(list)
    for i, op in enumerate(operations):
        if isinstance(op, QubitOperation):
            for ion in op.ions:
                ion_ops[id(ion)].append((i, op))

    # Assign each op to a phase:
    #   0          = before first MS
    #   2k+1       = MS layer k (odd)
    #   2(k+1)     = gap between MS layer k and k+1 (even)
    #   2N         = after last MS
    op_phase: Dict[int, int] = {}

    for _ion_id, ops_list in ion_ops.items():
        prev_layer = -1
        for _orig_idx, op in ops_list:
            oid = id(op)
            if isinstance(op, TwoQubitMSGate):
                key = (
                    getattr(op, '_tick_epoch', 0),
                    getattr(op, '_stim_origin', 0),
                )
                layer = ms_layer_map[key]
                phase = 2 * layer + 1
                op_phase[oid] = max(op_phase.get(oid, phase), phase)
                prev_layer = layer
            else:
                if prev_layer == -1:
                    phase = 0
                else:
                    phase = 2 * (prev_layer + 1)
                op_phase[oid] = max(op_phase.get(oid, phase), phase)

    # ── Phase fixup for ops on ions not participating in any MS gate ──
    # The per-ion walk above leaves prev_layer = -1 for such ions,
    # assigning ALL their ops to phase 0 (before first MS).  But if
    # those ops come from a stim instruction that is LATER than an MS
    # layer (e.g. MRX after CX in the stim circuit), they should be
    # in the gap AFTER that MS layer, not before.
    #
    # We skip ops whose (tick_epoch, stim_origin) matches an MS layer
    # key — those share a decomposition with MS ops and the per-ion
    # walk already handles them correctly.
    for op in operations:
        oid = id(op)
        if isinstance(op, TwoQubitMSGate):
            continue
        current_phase = op_phase.get(oid, 0)
        op_key = (
            getattr(op, '_tick_epoch', 0),
            getattr(op, '_stim_origin', 0),
        )
        if op_key in ms_layer_set:
            continue  # same decomposition as MS — per-ion walk is correct
        # Find the latest MS layer that precedes this op in stim order
        best_layer = -1
        for mk in ms_layer_keys:
            if mk < op_key:
                best_layer = ms_layer_map[mk]
        if best_layer >= 0:
            target_phase = 2 * (best_layer + 1)
            if target_phase > current_phase:
                op_phase[oid] = target_phase

    # Sort by (phase, original_index) and group by phase
    ops_with_phase = [
        (op_phase.get(id(op), 0), i, op)
        for i, op in enumerate(operations)
    ]
    ops_with_phase.sort(key=lambda x: (x[0], x[1]))

    phase_ops: Dict[int, List[Operation]] = defaultdict(list)
    for phase, _orig_idx, op in ops_with_phase:
        phase_ops[phase].append(op)

    # ── Emit phases: MS phases as-is, non-MS phases through steps 2-5 ──
    total_phases = 2 * num_ms_layers
    result: List[Operation] = []
    barriers: List[int] = []

    for phase in range(total_phases + 1):
        ops = phase_ops.get(phase)
        if not ops:
            continue

        if phase % 2 == 1:
            # Odd phase: MS layer — emit as contiguous block with barriers
            if result:
                barriers.append(len(result))
            result.extend(ops)
            barriers.append(len(result))
        else:
            # Even phase: non-MS window — apply steps 2-5
            if result:
                barriers.append(len(result))
            if phase == 0:
                pos = "first"
            elif phase >= total_phases:
                pos = "last"
            else:
                pos = "middle"
            grouped, sub_barriers = _barrier_reorder_non_ms(ops, position=pos)
            offset = len(result)
            result.extend(grouped)
            for b in sub_barriers:
                barriers.append(offset + b)

    barriers = sorted(set(b for b in barriers if 0 < b < len(result)))

    # ── Tag each op with its barrier group ───────────────────────────
    # The barrier group index is used post-routing to reconstruct
    # barriers after transport ops have been inserted between QubitOps.
    # The scheduler can then place barriers whenever _barrier_group
    # changes between consecutive QubitOps.
    _tag_barrier_groups(result, barriers)

    return result, barriers


def _tag_barrier_groups(
    ops: List[Operation],
    barriers: List[int],
) -> None:
    """Tag each op with ``_barrier_group`` based on barrier positions.

    Group 0 = ops before the first barrier, group 1 = between first and
    second barrier, etc.
    """
    barrier_set = set(barriers)
    group = 0
    for i, op in enumerate(ops):
        if i in barrier_set:
            group += 1
        op._barrier_group = group  # type: ignore[attr-defined]


# ── Steps 2-5: Type-block reordering for non-MS windows ──────────

def _barrier_reorder_non_ms(
    ops: List[Operation],
    position: str = "middle",
) -> Tuple[List[Operation], List[int]]:
    """Apply steps 2-5 of the barrier algorithm to a non-MS window.

    Uses **priority-driven layered emission** to group same-type
    operations across different ions while preserving per-ion ordering.
    In each pass, only the *highest-priority* available type is emitted;
    lower-priority types are deferred.  This lets ops from different ions
    accumulate at queue heads and merge into contiguous same-type blocks
    (e.g. two separate MEAS groups merge into one when intervening
    rotations are emitted first).

    Critical for MRX decompositions where one ion has
    ``XRot → YRot → Meas → Reset → YRot → XRot``.

    Algorithm:
      1. Build per-ion operation queues (preserving original order per ion).
      2. Iteratively: peek at each ion's next unplaced op, group available
         ops by type, emit **only the highest-priority type**, repeat.
      3. Insert barriers at every type transition.

    The priority ordering depends on ``position`` in the cycle:

    * ``"first"`` (before first MS):
      ``QubitReset → XRotation → YRotation → Measurement``
    * ``"middle"`` / ``"last"`` (between or after MS layers):
      ``XRotation → YRotation → Measurement → QubitReset``

    Returns (reordered_ops, barrier_positions).
    """
    if not ops:
        return [], []
    if len(ops) == 1:
        return list(ops), []

    # ── Priority map (position-aware) ────────────────────────────────
    if position == "first":
        _priority = {QubitReset: 0, XRotation: 1, YRotation: 2, Measurement: 3}
    else:
        _priority = {XRotation: 0, YRotation: 1, Measurement: 2, QubitReset: 3}

    # ── Build per-ion operation queues ────────────────────────────────
    # Each ion gets an ordered deque of its ops (preserving original
    # order within the phase for that ion).
    ion_queues: Dict[int, deque] = {}   # ion id(obj) → deque[op]
    no_ion_ops: List[Operation] = []    # ops without ion info (fallback)
    seen: Set[int] = set()

    for op in ops:
        oid = id(op)
        if oid in seen:
            continue
        seen.add(oid)
        ions = getattr(op, 'ions', None) or getattr(op, '_ions', [])
        if ions:
            ion_id = id(ions[0])
            if ion_id not in ion_queues:
                ion_queues[ion_id] = deque()
            ion_queues[ion_id].append(op)
        else:
            no_ion_ops.append(op)

    # ── Layered emission ─────────────────────────────────────────────
    # Each pass examines the HEAD of every ion's queue, groups the
    # available ops by type, and emits ONLY the highest-priority type.
    #
    # By deferring lower-priority types we give ops from different
    # ions time to align at the queue heads, enabling natural merging
    # of same-type blocks (e.g. MEAS from ion A + MEAS from ion B
    # merge into one MEAS block when we defer MEAS until all higher-
    # priority rotations have been emitted first).
    result: List[Operation] = []
    placed: Set[int] = set()

    while any(q for q in ion_queues.values()):
        # Clean up queue heads that were already placed (handles
        # multi-ion ops that appear in more than one queue).
        for _ion_id, q in ion_queues.items():
            while q and id(q[0]) in placed:
                q.popleft()

        # Collect the next available op from each ion
        available: Dict[type, List[Operation]] = defaultdict(list)
        available_ids: Set[int] = set()
        for _ion_id, q in ion_queues.items():
            if not q:
                continue
            op = q[0]
            oid = id(op)
            if oid not in available_ids:
                available_ids.add(oid)
                available[type(op)].append(op)

        if not available:
            break  # shouldn't happen, but safety valve

        # Emit ONLY the highest-priority available type.  This
        # maximises same-type block merging: lower-priority ops are
        # deferred to later passes where they may accumulate with
        # newly-available same-type ops from other ions.
        sorted_types = sorted(
            available.keys(),
            key=lambda t: _priority.get(t, 99),
        )
        tp = sorted_types[0]
        for op in available[tp]:
            oid = id(op)
            if oid in placed:
                continue
            placed.add(oid)
            result.append(op)
            # Pop from all ion queues that have this op at head
            for _ion_id2, q2 in ion_queues.items():
                if q2 and id(q2[0]) == oid:
                    q2.popleft()

    # Append any ops without ion info at the end
    for op in no_ion_ops:
        if id(op) not in placed:
            result.append(op)

    # ── Insert barriers at type transitions ──────────────────────────
    barriers: List[int] = []
    for i in range(1, len(result)):
        if type(result[i]) is not type(result[i - 1]):
            barriers.append(i)

    return result, barriers


# ---------------------------------------------------------------------------
# Legacy reordering (kept for backward compatibility, superseded by
# reorder_with_type_barriers for pre-routing use)
# ---------------------------------------------------------------------------

def reorder_rotations_for_batching(
    operations: List[Operation],
) -> List[Operation]:
    """Reorder single-qubit operations between MS-round boundaries.

    .. deprecated::
        Use :func:`reorder_with_type_barriers` in ``decompose_to_native``
        instead.  This function is kept for backward compatibility with
        callers that operate on post-routing operation lists.
    """
    if not operations:
        return list(operations)

    result: List[Operation] = []
    window_ops: List[Operation] = []

    def _flush() -> None:
        if not window_ops:
            return
        grouped, _ = _barrier_reorder_non_ms(window_ops)
        result.extend(grouped)
        window_ops.clear()

    for op in operations:
        if type(op) in _OP_TYPE_PRIORITY:
            window_ops.append(op)
        else:
            _flush()
            result.append(op)
    _flush()

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
    """Schedule operations into parallel batches."""
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

    # ── Type commitment state (WISE) ────────────────────────────────
    # Legacy: track the most recent chosen type for the heuristic
    # type-commitment logic.
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
            qubit_candidates = [
                op for op in candidates if isinstance(op, QubitOperation)
            ]
            if qubit_candidates:
                # ── Heuristic type commitment ─────────────────────
                type_groups: Dict[type, List[Operation]] = defaultdict(list)
                for op in qubit_candidates:
                    type_groups[type(op)].append(op)

                # Type exhaustion: keep the previous type if it still
                # has frontier ops.
                if (prev_chosen_type is not None
                        and prev_chosen_type in type_groups
                        and type_groups[prev_chosen_type]):
                    chosen_type = prev_chosen_type
                else:
                    # Pick the type with the most frontier ops,
                    # breaking ties by _OP_TYPE_PRIORITY.
                    chosen_type = max(
                        type_groups,
                        key=lambda t: (
                            len(type_groups[t]),
                            -_OP_TYPE_PRIORITY.get(t, 99),
                        ),
                    )

                prev_chosen_type = chosen_type
            else:
                # Only transport ops in frontier — no type constraint
                chosen_type = None

            # Defer non-chosen-type QubitOps.
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

        if isWiseArch:
            # ── WISE: separate transport from QubitOps ────────────
            # Transport ops (shuttling / reconfiguration) must be in
            # their own batches so they don't fragment QubitOp
            # packing.  Schedule transport first, then QubitOps.
            qubit_ops = [op for op in seg_ops
                         if isinstance(op, QubitOperation)]
            transport_ops = [op for op in seg_ops
                             if not isinstance(op, QubitOperation)]
            for sub_ops in (transport_ops, qubit_ops):
                if not sub_ops:
                    continue
                batches = _wise_parallel_pack(sub_ops)
                for batch in batches:
                    while t in time_schedule:
                        t += max(abs(t) * 1e-9, 1e-15)
                    time_schedule[t] = batch
                    batch_dur = max(
                        op.operationTime() for op in batch.operations
                    )
                    t += batch_dur
        else:
            seg_schedule = paralleliseOperations(
                seg_ops, isWISEArch=False, epoch_mode=epoch_mode,
            )
            seg_end = t  # track this segment's max end-time incrementally
            for s, par_op in seg_schedule.items():
                key = s + t
                # Guard against floating-point key collisions
                while key in time_schedule:
                    key += max(abs(key) * 1e-9, 1e-15)
                time_schedule[key] = par_op
                entry_end = key + max(
                    op.operationTime() for op in par_op.operations
                )
                if entry_end > seg_end:
                    seg_end = entry_end
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