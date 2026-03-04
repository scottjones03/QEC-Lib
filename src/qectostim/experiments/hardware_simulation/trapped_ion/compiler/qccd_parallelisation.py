
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
# Pre-routing barrier reorder (v4: full merge/push/reorder algorithm)
# ---------------------------------------------------------------------------

def reorder_with_type_barriers(
    operations: List[Operation],
) -> Tuple[List[Operation], List[int]]:
    """Reorder native ops into barrier-separated type-homogeneous blocks.

    Implements a 6-phase algorithm per user spec:

    1. **Per-origin canonicalization**: Convert each stim instruction block to::
           RX | RY | MS | RY | RX | MEAS | RESET

    2. **Linearize**: Concatenate all origins in stim-line order, producing
       a linear sequence of typed operations.

    3. **Merge at boundaries**: Join contiguous same-type blocks across
       origin boundaries (e.g., trailing RX from block1 + leading RX from
       block2 become one merged RX block).

    4. **Merge MEAS/RESET spans**: Push rotations out of MEAS/RESET spans.
       All MEAS ops merge together, all RESET ops merge together. Rotations
       that were between MEAS and RESET get pushed: rotations before MEAS
       stay before, rotations after RESET stay after.

    5. **Reorder rotations at MS-MS, MS-MEAS, RESET-MS boundaries**: Between
       two MS blocks, merge trailing rotations of first MS with leading
       rotations of second MS. At MS→MEAS boundary, rotations before MEAS.
       At RESET→MS boundary, rotations after RESET.

    6. **Insert barriers**: Between each final contiguous type block.

    Example transformation::

        Input stim blocks:
        |block1: RX,RY,MS,RY,RX| |block2: RX,RY,MS,RY,RX| |block3: RX,MEAS,RESET,RX| |block4: RX,MEAS,RESET| |block5: RX,RY,MS,RY,RX|

        After merge at boundaries (showing boundary merges with |):
        RX,RY,MS,RY,|RX|,RY,MS,RY,|RX|,MEAS,RESET,|RX|,MEAS,RESET,|RX|,RY,MS,RY,RX

        After MEAS/RESET merge + push rotations:
        RX RY MS RY RX RY MS RY RX MEAS RESET RX RY MS RY RX

        After reorder rotations between MS-MS:
        RX RY MS RY RX MS RY RX MEAS RESET RX RY MS RY RX

    Parameters
    ----------
    operations : list[Operation]
        Native ``QubitOperation``s from ``decompose_to_native`` (no transport).

    Returns
    -------
    reordered : list[Operation]
        Operations reordered with type grouping.
    barriers : list[int]
        Barrier positions (indices into *reordered*) at block boundaries.
    """
    if not operations:
        return [], []

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Group by stim origin and canonicalize each to:
    #          RX | RY | MS | RY | RX | MEAS | RESET
    # ══════════════════════════════════════════════════════════════════════
    ops_by_origin: Dict[Tuple[int, int], List[Operation]] = defaultdict(list)
    for op in operations:
        key = (
            getattr(op, '_tick_epoch', 0),
            getattr(op, '_stim_origin', 0),
        )
        ops_by_origin[key].append(op)

    sorted_origins = sorted(ops_by_origin.keys())

    # Canonicalize each origin into typed blocks
    # Format: list of (type_tag, ops) where type_tag is 'RX'|'RY'|'MS'|'MEAS'|'RESET'
    origin_blocks: List[List[Tuple[str, List[Operation]]]] = []
    for origin in sorted_origins:
        origin_ops = ops_by_origin[origin]
        blocks = _canonicalize_origin_to_blocks(origin_ops)
        if blocks:
            origin_blocks.append(blocks)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2 & 3: Linearize and merge at boundaries
    # Flatten all blocks, merging same-type adjacent blocks
    # ══════════════════════════════════════════════════════════════════════
    merged: List[Tuple[str, List[Operation]]] = []
    for blocks in origin_blocks:
        for type_tag, ops in blocks:
            if not ops:
                continue
            if merged and merged[-1][0] == type_tag:
                # Same type as previous - merge (boundary merge)
                merged[-1][1].extend(ops)
            else:
                merged.append((type_tag, list(ops)))

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4: Merge MEAS/RESET spans, pushing rotations out
    # Find spans of [rotations...MEAS...RESET...rotations] and consolidate:
    # - MEAS ops all merge together
    # - RESET ops all merge together
    # - Rotations before MEAS stay before, rotations after RESET stay after
    # ══════════════════════════════════════════════════════════════════════
    merged = _merge_meas_reset_spans(merged)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4b: Merge RESET+rotation spans (initialisation blocks)
    # Consolidate interleaved RESET|RX|RESET|RX → RESET|RX
    # ══════════════════════════════════════════════════════════════════════
    merged = _merge_reset_rotation_spans(merged)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4c: Consolidate rotation spans across stim origins
    # After Phases 4/4b, different stim origins each contribute RX/RY pairs
    # within the same anchor span.  This produces alternating RX, RY, RX, RY
    # blocks.  Consolidate each such span into ONE RX + ONE RY block.
    # ══════════════════════════════════════════════════════════════════════
    merged = _consolidate_rotation_spans(merged)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 5: Reorder rotations at MS-MS, MS-MEAS, RESET-MS boundaries
    # Between MS→MS: merge trailing rotations with leading rotations
    # Between MS→MEAS: rotations come before MEAS
    # Between RESET→MS: rotations come after RESET
    # ══════════════════════════════════════════════════════════════════════
    merged = _reorder_rotations_between_anchors(merged)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 5b: Final boundary merge — catch any new same-type adjacencies
    # created by Phases 4/4b/5.
    # ══════════════════════════════════════════════════════════════════════
    final_merged: List[Tuple[str, List[Operation]]] = []
    for type_tag, ops in merged:
        if not ops:
            continue
        if final_merged and final_merged[-1][0] == type_tag:
            final_merged[-1][1].extend(ops)
        else:
            final_merged.append((type_tag, list(ops)))
    merged = final_merged

    # ══════════════════════════════════════════════════════════════════════
    # Phase 5c: Deduplicate RESET and MEAS ops that target the same ion
    # within a merged block.  MR (measure-reset) already emits a RESET,
    # so a subsequent R on the same qubit produces a redundant QubitReset.
    # Keep only the first occurrence per ion within each RESET/MEAS block.
    # ══════════════════════════════════════════════════════════════════════
    merged = _dedup_same_ion_ops(merged)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 6: Build final result with barriers between each type block
    # ══════════════════════════════════════════════════════════════════════
    result: List[Operation] = []
    barriers: List[int] = []

    for type_tag, ops in merged:
        if not ops:
            continue
        # Barrier before this block (if there's prior content)
        if result:
            barriers.append(len(result))
        result.extend(ops)

    # Clean up barriers
    barriers = sorted(set(b for b in barriers if 0 < b < len(result)))

    # Tag barrier groups
    _tag_barrier_groups(result, barriers)

    return result, barriers


def _canonicalize_origin_to_blocks(
    ops: List[Operation],
) -> List[Tuple[str, List[Operation]]]:
    """Convert one stim origin's ops to ordered type blocks.

    Output order for MS-containing origins:
        [pre-RX][pre-RY][MS][post-RY][post-RX][MEAS][RESET]

    Output order for non-MS origins:
        [RESET][RX][RY] for initialization
        [RX][RY][MEAS][RESET] for measurement rounds
    """
    has_ms = any(isinstance(op, TwoQubitMSGate) for op in ops)
    has_meas = any(isinstance(op, Measurement) for op in ops)

    if has_ms:
        return _canonicalize_ms_origin_to_blocks(ops)
    elif has_meas:
        # Measurement round: RX | RY | MEAS | RESET
        return _canonicalize_meas_origin_to_blocks(ops)
    else:
        # Initialization or pure rotations: RESET | RX | RY
        return _canonicalize_init_origin_to_blocks(ops)


def _canonicalize_ms_origin_to_blocks(
    ops: List[Operation],
) -> List[Tuple[str, List[Operation]]]:
    """Canonicalize MS-containing origin to blocks using per-pair segmentation.

    Pattern: [pre-RX][pre-RY][MS][post-RY][post-RX][MEAS][RESET]

    For multi-pair CX (e.g. ``CX 4 1 3 6 …``), each pair decomposes as
    ``[RY, RX, RX, MS, RY]``.  We segment ops around each MS gate so
    that *every* pair's pre-MS rotations are collected (not just pair 0's).

    Result for 6-pair CX: ``RX(12), RY(6), MS(6), RY(6)``
    (not the old buggy ``RX(2), RY(1), MS(6), RY(11), RX(10)``).
    """
    ms_indices = [i for i, op in enumerate(ops) if isinstance(op, TwoQubitMSGate)]
    if not ms_indices:
        return []

    all_pre_rx: List[Operation] = []
    all_pre_ry: List[Operation] = []
    all_pre_reset: List[Operation] = []
    all_ms: List[Operation] = []
    all_post_ry: List[Operation] = []
    all_post_rx: List[Operation] = []
    all_post_meas: List[Operation] = []
    all_post_reset: List[Operation] = []

    # ── Ops before the first MS gate: always pre-MS ──────────
    for i in range(0, ms_indices[0]):
        op = ops[i]
        if isinstance(op, XRotation):
            all_pre_rx.append(op)
        elif isinstance(op, YRotation):
            all_pre_ry.append(op)
        elif isinstance(op, QubitReset):
            all_pre_reset.append(op)

    # ── Collect all MS gates ─────────────────────────────────
    all_ms = [ops[i] for i in ms_indices]

    # ── Ops between consecutive MS gates ─────────────────────
    # Per-pair decomposition: MS(k) is followed by one post-RY,
    # then the next pair's pre-MS rotations, then MS(k+1).
    # The first YRotation after MS(k) is the post-MS correction;
    # everything else before MS(k+1) is pre-MS for the next pair.
    for k in range(len(ms_indices) - 1):
        start = ms_indices[k] + 1
        end = ms_indices[k + 1]
        found_post_ry = False
        for i in range(start, end):
            op = ops[i]
            if isinstance(op, YRotation) and not found_post_ry:
                all_post_ry.append(op)
                found_post_ry = True
            elif isinstance(op, XRotation):
                all_pre_rx.append(op)
            elif isinstance(op, YRotation):
                all_pre_ry.append(op)
            elif isinstance(op, QubitReset):
                all_pre_reset.append(op)
            elif isinstance(op, Measurement):
                all_post_meas.append(op)

    # ── Ops after the last MS gate: always post-MS ───────────
    for i in range(ms_indices[-1] + 1, len(ops)):
        op = ops[i]
        if isinstance(op, YRotation):
            all_post_ry.append(op)
        elif isinstance(op, XRotation):
            all_post_rx.append(op)
        elif isinstance(op, Measurement):
            all_post_meas.append(op)
        elif isinstance(op, QubitReset):
            all_post_reset.append(op)

    # Build blocks: [RESET] [RX] [RY] [MS] [RY] [RX] [MEAS] [RESET]
    blocks: List[Tuple[str, List[Operation]]] = []
    if all_pre_reset:
        blocks.append(('RESET', all_pre_reset))
    if all_pre_rx:
        blocks.append(('RX', all_pre_rx))
    if all_pre_ry:
        blocks.append(('RY', all_pre_ry))
    if all_ms:
        blocks.append(('MS', all_ms))
    if all_post_ry:
        blocks.append(('RY', all_post_ry))
    if all_post_rx:
        blocks.append(('RX', all_post_rx))
    if all_post_meas:
        blocks.append(('MEAS', all_post_meas))
    if all_post_reset:
        blocks.append(('RESET', all_post_reset))

    return blocks


def _canonicalize_meas_origin_to_blocks(
    ops: List[Operation],
) -> List[Tuple[str, List[Operation]]]:
    """Canonicalize measurement origin (no MS) to blocks.

    Pattern: [pre-RX][pre-RY][MEAS][RESET][post-RX][post-RY]

    For MRX instructions, the decomposition per qubit is::

        XRot, YRot, MEAS, RESET, YRot, XRot

    Rotations *before* the qubit's MEAS are pre-MEAS (basis change).
    Rotations *after* the qubit's RESET are post-RESET (re-initialisation).
    Using per-qubit anchor tracking ensures multi-qubit MRX is classified
    correctly even when ops are interleaved by qubit.
    """
    meas_ops = [op for op in ops if isinstance(op, Measurement)]
    reset_ops = [op for op in ops if isinstance(op, QubitReset)]

    if not meas_ops:
        return _canonicalize_init_origin_to_blocks(ops)

    # Build per-qubit MEAS / RESET index maps
    qubit_meas_idx: Dict = {}   # ion → first MEAS index
    qubit_reset_idx: Dict = {}  # ion → last RESET index

    for i, op in enumerate(ops):
        if isinstance(op, Measurement):
            ion = op.ions[0] if op.ions else None
            if ion is not None and ion not in qubit_meas_idx:
                qubit_meas_idx[ion] = i
        elif isinstance(op, QubitReset):
            ion = op.ions[0] if op.ions else None
            if ion is not None:
                qubit_reset_idx[ion] = i  # last RESET for this qubit

    pre_rx: List[Operation] = []
    pre_ry: List[Operation] = []
    post_rx: List[Operation] = []
    post_ry: List[Operation] = []

    for i, op in enumerate(ops):
        if not isinstance(op, (XRotation, YRotation)):
            continue

        ion = op.ions[0] if op.ions else None
        is_pre = True  # default

        if ion is not None:
            m_idx = qubit_meas_idx.get(ion)
            r_idx = qubit_reset_idx.get(ion)

            if m_idx is not None and i < m_idx:
                is_pre = True   # before this qubit's MEAS
            elif r_idx is not None and i > r_idx:
                is_pre = False  # after this qubit's RESET
            # else: between MEAS and RESET – unusual, keep as pre

        if is_pre:
            if isinstance(op, XRotation):
                pre_rx.append(op)
            else:
                pre_ry.append(op)
        else:
            if isinstance(op, XRotation):
                post_rx.append(op)
            else:
                post_ry.append(op)

    blocks: List[Tuple[str, List[Operation]]] = []
    if pre_rx:
        blocks.append(('RX', pre_rx))
    if pre_ry:
        blocks.append(('RY', pre_ry))
    if meas_ops:
        blocks.append(('MEAS', meas_ops))
    if reset_ops:
        blocks.append(('RESET', reset_ops))
    if post_rx:
        blocks.append(('RX', post_rx))
    if post_ry:
        blocks.append(('RY', post_ry))
    return blocks


def _canonicalize_init_origin_to_blocks(
    ops: List[Operation],
) -> List[Tuple[str, List[Operation]]]:
    """Canonicalize initialization origin (no MS, no MEAS) to blocks.

    Pattern: [RESET][RX][RY]
    """
    rx = [op for op in ops if isinstance(op, XRotation)]
    ry = [op for op in ops if isinstance(op, YRotation)]
    reset = [op for op in ops if isinstance(op, QubitReset)]

    blocks: List[Tuple[str, List[Operation]]] = []
    if reset:
        blocks.append(('RESET', reset))
    if rx:
        blocks.append(('RX', rx))
    if ry:
        blocks.append(('RY', ry))
    return blocks


def _merge_meas_reset_spans(
    blocks: List[Tuple[str, List[Operation]]],
) -> List[Tuple[str, List[Operation]]]:
    """Merge MEAS/RESET spans, pushing rotations out.

    When we have a sequence like [...RX...RY...MEAS...RESET...RX...RY...],
    consolidate all MEAS together and all RESET together. Rotations that
    appear between MEAS and RESET get pushed:
    - Rotations immediately before MEAS stay before MEAS
    - Rotations immediately after RESET stay after RESET
    - Rotations BETWEEN MEAS and RESET: push to after RESET

    This ensures MEAS and RESET form contiguous blocks.
    """
    if not blocks:
        return []

    # Look for MEAS/RESET spans and merge them
    result: List[Tuple[str, List[Operation]]] = []
    i = 0

    while i < len(blocks):
        tag, ops = blocks[i]

        if tag == 'MEAS':
            # Found MEAS - collect following MEAS/RESET/rotations into a span
            all_meas = list(ops)
            all_reset: List[Operation] = []
            rotations_after_reset: List[Tuple[str, List[Operation]]] = []
            j = i + 1

            while j < len(blocks):
                next_tag, next_ops = blocks[j]
                if next_tag == 'MEAS':
                    all_meas.extend(next_ops)
                    j += 1
                elif next_tag == 'RESET':
                    all_reset.extend(next_ops)
                    j += 1
                elif next_tag in ('RX', 'RY'):
                    # Rotations - check if more MEAS/RESET follow
                    # If so, these rotations are BETWEEN MEAS/RESET spans
                    # Push them to after RESET
                    k = j + 1
                    has_more_meas_reset = False
                    while k < len(blocks):
                        if blocks[k][0] in ('MEAS', 'RESET'):
                            has_more_meas_reset = True
                            break
                        elif blocks[k][0] in ('RX', 'RY'):
                            k += 1
                        else:
                            break
                    if has_more_meas_reset:
                        # Push these rotations to after RESET
                        rotations_after_reset.append((next_tag, list(next_ops)))
                        j += 1
                    else:
                        # No more MEAS/RESET - these rotations end the span
                        break
                else:
                    # MS or other - end of span
                    break

            # Emit merged MEAS block
            if all_meas:
                result.append(('MEAS', all_meas))
            # Emit merged RESET block
            if all_reset:
                result.append(('RESET', all_reset))
            # Emit pushed rotations (after RESET)
            for rot_tag, rot_ops in rotations_after_reset:
                if rot_ops:
                    # Merge with previous if same type
                    if result and result[-1][0] == rot_tag:
                        result[-1][1].extend(rot_ops)
                    else:
                        result.append((rot_tag, rot_ops))

            i = j
        else:
            # Not MEAS - just pass through
            result.append((tag, list(ops)))
            i += 1

    return result


def _merge_reset_rotation_spans(
    blocks: List[Tuple[str, List[Operation]]],
) -> List[Tuple[str, List[Operation]]]:
    """Merge RESET+rotation spans without MS/MEAS (initialisation blocks).

    When consecutive blocks consist only of RESET and rotations (RX, RY) with
    no MS or MEAS anchors between them, collect all RESET ops into one block
    and all rotations into one block each:

        RESET | RX | RESET | RX | RY | RESET → RESET | RX | RY

    This handles initialisation sequences where multiple stim origins
    (R, RX, R, RX, ...) produce interleaved RESET/rotation blocks.
    """
    if not blocks:
        return []

    result: List[Tuple[str, List[Operation]]] = []
    i = 0

    while i < len(blocks):
        tag, ops = blocks[i]

        # Only start collecting when we see RESET
        if tag == 'RESET':
            # Look ahead — is there a span of RESET+rotation blocks
            # before the next MS/MEAS anchor?
            j = i + 1
            has_more_resets = False
            while j < len(blocks) and blocks[j][0] in ('RESET', 'RX', 'RY'):
                if blocks[j][0] == 'RESET':
                    has_more_resets = True
                j += 1

            if has_more_resets:
                # Collect the entire span [i..j)
                all_reset: List[Operation] = []
                all_rx: List[Operation] = []
                all_ry: List[Operation] = []
                for tag_s, ops_s in blocks[i:j]:
                    if tag_s == 'RESET':
                        all_reset.extend(ops_s)
                    elif tag_s == 'RX':
                        all_rx.extend(ops_s)
                    elif tag_s == 'RY':
                        all_ry.extend(ops_s)
                # Emit in canonical order: RESET | RX | RY
                if all_reset:
                    result.append(('RESET', all_reset))
                if all_rx:
                    result.append(('RX', all_rx))
                if all_ry:
                    result.append(('RY', all_ry))
                i = j
            else:
                result.append((tag, list(ops)))
                i += 1
        else:
            result.append((tag, list(ops)))
            i += 1

    return result


def _dedup_same_ion_ops(
    blocks: List[Tuple[str, List[Operation]]],
) -> List[Tuple[str, List[Operation]]]:
    """Remove duplicate RESET (or MEAS) ops targeting the same ion in a block.

    After merging, a single RESET block may contain two ``QubitReset`` ops
    for the same ion — one from an ``MR`` (measure-reset) stim instruction
    and another from a standalone ``R`` at the start of the next round.
    The second reset is physically redundant.

    This function keeps only the **first** occurrence per ion within each
    RESET or MEAS block.  Other block types are passed through unchanged.
    """
    if not blocks:
        return []

    result: List[Tuple[str, List[Operation]]] = []
    for tag, ops in blocks:
        if tag in ('RESET', 'MEAS') and ops:
            seen_ions: Set = set()
            deduped: List[Operation] = []
            for op in ops:
                # Single-qubit ops have exactly one ion in _ions
                ions = getattr(op, '_ions', None) or getattr(op, 'ions', None)
                if ions:
                    ion_key = id(ions[0])
                    if ion_key in seen_ions:
                        continue  # skip duplicate
                    seen_ions.add(ion_key)
                deduped.append(op)
            if deduped:
                result.append((tag, deduped))
        else:
            result.append((tag, list(ops)))
    return result


def _consolidate_rotation_spans(
    blocks: List[Tuple[str, List[Operation]]],
) -> List[Tuple[str, List[Operation]]]:
    """Consolidate consecutive rotation blocks into ONE RX + ONE RY.

    After Phases 4/4b, different stim origins may each contribute their own
    RX/RY pairs within a single anchor span (e.g. RESET→MS).  This produces
    alternating blocks like ``RX, RY, RX, RY, RX, RY`` that cannot be merged
    by adjacent-same-type logic alone.

    This function finds every maximal span of consecutive rotation blocks
    and replaces it with at most two blocks: ``RX(all), RY(all)``.

    Example::

        RESET, RX, RY, RX, RY, RX, RY, MS
        →  RESET, RX(all), RY(all), MS
    """
    if not blocks:
        return []

    result: List[Tuple[str, List[Operation]]] = []
    i = 0

    while i < len(blocks):
        tag, ops = blocks[i]

        if tag in ('RX', 'RY'):
            # Collect all consecutive rotation blocks
            all_rx: List[Operation] = []
            all_ry: List[Operation] = []
            while i < len(blocks) and blocks[i][0] in ('RX', 'RY'):
                if blocks[i][0] == 'RX':
                    all_rx.extend(blocks[i][1])
                else:
                    all_ry.extend(blocks[i][1])
                i += 1
            # Emit in canonical order: RX then RY
            if all_rx:
                result.append(('RX', all_rx))
            if all_ry:
                result.append(('RY', all_ry))
        else:
            result.append((tag, list(ops)))
            i += 1

    return result


def _reorder_rotations_between_anchors(
    blocks: List[Tuple[str, List[Operation]]],
) -> List[Tuple[str, List[Operation]]]:
    """Reorder rotations at MS-MS, MS-MEAS, RESET-MS boundaries.

    Between MS→MS: consolidate trailing rotations with leading rotations.
    If we have MS | RY | RX | RY | MS, the RY-RX-RY span gets simplified
    by merging the two RY blocks: MS | RY | RX | MS

    Between MS→MEAS: rotations stay before MEAS (already correct from Phase 4)

    Between RESET→MS: rotations stay after RESET (already correct from Phase 4)

    The key optimization is MS→MS boundaries where we can merge rotations.
    """
    if not blocks:
        return []

    result: List[Tuple[str, List[Operation]]] = []
    i = 0

    while i < len(blocks):
        tag, ops = blocks[i]

        if tag == 'MS':
            # Emit MS block
            result.append(('MS', list(ops)))
            i += 1

            # Collect rotations until next anchor (MS, MEAS, or end)
            rotations: List[Tuple[str, List[Operation]]] = []
            while i < len(blocks) and blocks[i][0] in ('RX', 'RY'):
                rotations.append((blocks[i][0], list(blocks[i][1])))
                i += 1

            if not rotations:
                continue

            # Check what's next
            if i < len(blocks):
                next_tag = blocks[i][0]

                if next_tag == 'MS':
                    # MS→rotations→MS: merge same-type rotations
                    # Emit consolidated rotations
                    merged_rots = _merge_rotation_blocks(rotations)
                    for rot_tag, rot_ops in merged_rots:
                        if rot_ops:
                            result.append((rot_tag, rot_ops))

                elif next_tag in ('MEAS', 'RESET'):
                    # MS→rotations→MEAS/RESET: merge rotations, keep before anchor
                    merged_rots = _merge_rotation_blocks(rotations)
                    for rot_tag, rot_ops in merged_rots:
                        if rot_ops:
                            result.append((rot_tag, rot_ops))
                else:
                    # Some other block - emit rotations
                    for rot_tag, rot_ops in rotations:
                        if rot_ops:
                            if result and result[-1][0] == rot_tag:
                                result[-1][1].extend(rot_ops)
                            else:
                                result.append((rot_tag, rot_ops))
            else:
                # End of sequence - emit rotations
                for rot_tag, rot_ops in rotations:
                    if rot_ops:
                        if result and result[-1][0] == rot_tag:
                            result[-1][1].extend(rot_ops)
                        else:
                            result.append((rot_tag, rot_ops))
        else:
            # Non-MS block — pass through, but handle RESET→rotations→MS
            if tag == 'RESET':
                result.append(('RESET', list(ops)))
                i += 1
                # Collect subsequent rotations
                rot_after_reset: List[Tuple[str, List[Operation]]] = []
                while i < len(blocks) and blocks[i][0] in ('RX', 'RY'):
                    rot_after_reset.append((blocks[i][0], list(blocks[i][1])))
                    i += 1
                if rot_after_reset:
                    if i < len(blocks) and blocks[i][0] == 'MS':
                        # RESET→rotations→MS: merge with RX,RY order
                        merged = _merge_rotation_blocks_pre_ms(rot_after_reset)
                        for rt, ro in merged:
                            if ro:
                                result.append((rt, ro))
                    else:
                        # No MS follows — emit rotations as-is
                        for rt, ro in rot_after_reset:
                            if ro:
                                if result and result[-1][0] == rt:
                                    result[-1][1].extend(ro)
                                else:
                                    result.append((rt, ro))
            elif result and result[-1][0] == tag:
                result[-1][1].extend(ops)
                i += 1
            else:
                result.append((tag, list(ops)))
                i += 1

    return result


def _merge_rotation_blocks(
    rotations: List[Tuple[str, List[Operation]]],
) -> List[Tuple[str, List[Operation]]]:
    """Merge all rotation blocks into one RY + one RX block.

    Collects all XRotation and YRotation ops from the input blocks
    and outputs them in **RY | RX** order, matching the post-MS canonical
    pattern (correction RY from the preceding MS comes first, then the
    pre-MS RX for the next MS)::

        { MS BLOCK | ROTY BLOCK | ROTX BLOCK } × n

    Used at inter-MS boundaries (MS→MS, MS→MEAS).
    """
    if not rotations:
        return []

    all_rx: List[Operation] = []
    all_ry: List[Operation] = []

    for tag, ops in rotations:
        if tag == 'RX':
            all_rx.extend(ops)
        elif tag == 'RY':
            all_ry.extend(ops)

    # Output in RY | RX order (post-MS canonical order)
    result: List[Tuple[str, List[Operation]]] = []
    if all_ry:
        result.append(('RY', all_ry))
    if all_rx:
        result.append(('RX', all_rx))
    return result


def _merge_rotation_blocks_pre_ms(
    rotations: List[Tuple[str, List[Operation]]],
) -> List[Tuple[str, List[Operation]]]:
    """Merge all rotation blocks into one RX + one RY block.

    Collects all XRotation and YRotation ops from the input blocks
    and outputs them in **RX | RY** order, matching the pre-MS canonical
    pattern (RESET → RX → RY → MS)::

        RESET | ROTX BLOCK | ROTY BLOCK | MS

    Used at RESET→MS boundaries.
    """
    if not rotations:
        return []

    all_rx: List[Operation] = []
    all_ry: List[Operation] = []

    for tag, ops in rotations:
        if tag == 'RX':
            all_rx.extend(ops)
        elif tag == 'RY':
            all_ry.extend(ops)

    # Output in RX | RY order (pre-MS canonical order)
    result: List[Tuple[str, List[Operation]]] = []
    if all_rx:
        result.append(('RX', all_rx))
    if all_ry:
        result.append(('RY', all_ry))
    return result


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


def _sort_by_type_with_context(
    ops: List[Operation],
    is_first_segment: bool,
) -> List[Operation]:
    """Sort ops by type priority (for legacy reorder_rotations_for_batching)."""
    if not ops:
        return []

    if is_first_segment:
        priority = {QubitReset: 0, XRotation: 1, YRotation: 2, Measurement: 3}
    else:
        priority = {XRotation: 0, YRotation: 1, Measurement: 2, QubitReset: 3}

    return sorted(ops, key=lambda o: priority.get(type(o), 99))


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
        # Sort by type: RX before RY before MEAS before RESET (for post-MS context)
        grouped = _sort_by_type_with_context(window_ops, is_first_segment=False)
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
            # ── WISE: type-homogeneous run-based packing ─────────
            # WISE requires type-homogeneous batches (global control
            # pulses).  Split the segment into consecutive runs by
            # (is_qubit, op_type).  Each run is packed independently,
            # and runs are emitted in their original order.
            runs: List[List[Operation]] = []
            current_run: List[Operation] = []
            current_key: Optional[Tuple[bool, type]] = None

            for op in seg_ops:
                is_qubit = isinstance(op, QubitOperation)
                op_type = type(op) if is_qubit else None
                key = (is_qubit, op_type)
                if key != current_key:
                    if current_run:
                        runs.append(current_run)
                    current_run = [op]
                    current_key = key
                else:
                    current_run.append(op)
            if current_run:
                runs.append(current_run)

            for run_ops in runs:
                if not run_ops:
                    continue
                # WISE: pack type-homogeneous runs into parallel
                # batches respecting component disjointness.
                packed = _wise_parallel_pack(run_ops)
                for batch in packed:
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