from typing import (
    Callable,
    Sequence,
    List,
    Tuple,
    Set,
    Dict,
    Mapping,
    Optional,
    Any,
)
from collections import defaultdict
from dataclasses import dataclass, field
import copy
import os
import time
import logging
import threading
from copy import deepcopy

import numpy as np

from ..utils.qccd_nodes import *
from ..utils.qccd_operations import *
from ..utils.qccd_operations_on_qubits import *
from ..utils.qccd_arch import *
from .routing_config import (
    WISERoutingConfig,
    RoutingProgress,
    STAGE_ROUTING,
    STAGE_APPLYING,
    STAGE_PATCHING,
    STAGE_PATCH_SOLVE,
    STAGE_RECONFIG,
    STAGE_COMPLETE,
)
from .qccd_qubits_to_ions import *


PATCH_LOG_PREFIX = "[PatchRoute]"
PATCH_VERBOSE_MOVES = os.environ.get("WISE_PATCH_VERBOSE", "0") not in ("0", "")

LOGGER_NAME = "wise.qccd.route"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())
logger.propagate = True

def _compute_patch_gating_capacity(
    n: int,
    m: int,
    col_offset: int,
    capacity: int,
) -> int:
    """
    Compute the maximum number of disjoint row/block 'gating zones'
    available in this patch.

    n         : patch height (rows)
    m         : patch width (columns, local to patch)
    col_offset: global column index of patch's first column
    capacity  : global block width (CAPACITY)
    """
    if capacity <= 0 or m <= 0 or n <= 0:
        return 0

    first_block_idx = col_offset // capacity
    last_block_idx  = (col_offset + m - 1) // capacity
    num_blocks      = last_block_idx - first_block_idx + 1

    total_per_row = 0
    for b_local in range(num_blocks):
        b_global    = first_block_idx + b_local
        global_start = b_global * capacity
        global_end   = (b_global + 1) * capacity

        local_start = max(0, global_start - col_offset)
        local_end   = min(m, global_end - col_offset)
        width       = max(0, local_end - local_start)

        # each block contributes floor(width/2) gating positions per row
        total_per_row += (width // 2)

    return n * total_per_row

def _generate_patch_regions(
    n_rows: int,
    n_cols: int,
    patch_h: int,
    patch_w: int,
    offset_row: int,
    offset_col: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate non-overlapping patch regions (r0, c0, r1, c1) that tile an n×m grid.
    The tiling is anchored at (offset_row, offset_col). Regions are clipped to the
    physical grid bounds.
    """
    if patch_h <= 0 or patch_w <= 0:
        return [(0, 0, n_rows, n_cols)]

    regions: List[Tuple[int, int, int, int]] = []

    start_row = min(max(offset_row, 0), n_rows - 1) if n_rows > 0 else 0
    start_col = min(max(offset_col, 0), n_cols - 1) if n_cols > 0 else 0

    row = start_row
    while row < n_rows:
        row_end = min(row + patch_h, n_rows)
        col = start_col
        while col < n_cols:
            col_end = min(col + patch_w, n_cols)
            regions.append((row, col, row_end, col_end))
            col += patch_w
        row += patch_h

    return regions


def _split_pairs_for_patch(
    pairs_per_round: List[List[Tuple[int, int]]],
    ion_positions: Mapping[int, Tuple[int, int]],
    region: Tuple[int, int, int, int],
) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
    """
    Split the pending pairs into (inside patch, outside patch) buckets for each round.
    """

    def _in_region(ion_idx: int) -> bool:
        pos = ion_positions.get(ion_idx)
        if pos is None:
            return False
        r, c = pos
        r0, c0, r1, c1 = region
        return r0 <= r < r1 and c0 <= c < c1

    patch_pairs: List[List[Tuple[int, int]]] = [[] for _ in range(len(pairs_per_round))]
    leftovers: List[List[Tuple[int, int]]] = [[] for _ in range(len(pairs_per_round))]

    for ridx, round_pairs in enumerate(pairs_per_round):
        for ion_a, ion_b in round_pairs:
            if _in_region(ion_a) and _in_region(ion_b):
                patch_pairs[ridx].append((ion_a, ion_b))
            else:
                leftovers[ridx].append((ion_a, ion_b))

    return patch_pairs, leftovers


def _compute_cross_boundary_prefs(
    region: Tuple[int, int, int, int],
    ion_positions: Mapping[int, Tuple[int, int]],
    pairs_per_round: List[List[Tuple[int, int]]],
) -> List[Dict[int, Set[str]]]:
    """
    For each round, record which boundary/boundaries an ion should approach based on
    gates with partners outside the region. Directions are one or more of
    {"top", "bottom", "left", "right"}.
    """
    r0, c0, r1, c1 = region
    prefs: List[Dict[int, Set[str]]] = [dict() for _ in range(len(pairs_per_round))]

    def _region_contains(pos: Tuple[int, int]) -> bool:
        rr, cc = pos
        return (r0 <= rr < r1) and (c0 <= cc < c1)

    for ridx, round_pairs in enumerate(pairs_per_round):
        pref_round = prefs[ridx]
        for ion_a, ion_b in round_pairs:
            pos_a = ion_positions.get(ion_a)
            pos_b = ion_positions.get(ion_b)
            if pos_a is None or pos_b is None:
                continue

            in_a = _region_contains(pos_a)
            in_b = _region_contains(pos_b)
            if in_a == in_b:
                continue

            inside_ion, outside_pos = (ion_a, pos_b) if in_a else (ion_b, pos_a)
            dirs: Set[str] = set()
            rr_o, cc_o = outside_pos
            if rr_o < r0:
                dirs.add("top")
            if rr_o >= r1:
                dirs.add("bottom")
            if cc_o < c0:
                dirs.add("left")
            if cc_o >= c1:
                dirs.add("right")

            if not dirs:
                continue

            pref_round.setdefault(inside_ion, set()).update(dirs)

    return prefs


def _simulate_schedule_replay(
    initial_layout: np.ndarray,
    schedule: List[Dict[str, Any]],
) -> np.ndarray:
    """Simulate replaying a merged schedule on a layout, returning the result.

    Used to verify that a schedule actually produces the expected layout
    without executing physicalOperation.  This catches mismatches between
    SAT-model layouts (decoded from assignment variables) and merged
    schedule outputs (decoded from swap variables) that can arise when
    multiple patch schedules are combined.

    Parameters
    ----------
    initial_layout : np.ndarray
        The starting ion arrangement (will not be mutated).
    schedule : list of dict
        Pass-info dicts with ``"phase"`` (``"H"``/``"V"``), ``"h_swaps"``,
        and ``"v_swaps"`` keys.  Swap coordinates are global (row, col).

    Returns
    -------
    np.ndarray
        The layout that results from applying every swap in *schedule*
        sequentially.
    """
    A = initial_layout.copy()
    n_rows, n_cols = A.shape
    for pass_info in schedule:
        phase = pass_info.get("phase", "H")
        if phase == "H":
            for (r, c) in pass_info.get("h_swaps", []):
                if 0 <= r < n_rows and 0 <= c < n_cols - 1:
                    A[r, c], A[r, c + 1] = A[r, c + 1], A[r, c]
        elif phase == "V":
            for (r, c) in pass_info.get("v_swaps", []):
                if 0 <= r < n_rows - 1 and 0 <= c < n_cols:
                    A[r, c], A[r + 1, c] = A[r + 1, c], A[r, c]
    return A


def _pass_has_swaps(pass_info: Dict[str, Any]) -> bool:
    if pass_info["phase"] == "H":
        return bool(pass_info.get("h_swaps"))
    return bool(pass_info.get("v_swaps"))


def _infer_pass_parity(pass_info: Dict[str, Any]) -> Optional[int]:
    """
    Determine whether this pass belongs to the "even" or "odd" family.
    For H-phase we inspect the column index of the first horizontal swap;
    for V-phase we inspect the row index of the first vertical swap.
    Returns 0/1 if swaps exist, otherwise None.
    """
    phase = pass_info["phase"]
    if phase == "H":
        swaps = pass_info.get("h_swaps", [])
        if not swaps:
            return None
        _, jcol = swaps[0]
        return jcol % 2
    swaps = pass_info.get("v_swaps", [])
    if not swaps:
        return None
    krow, _ = swaps[0]
    return krow % 2


def _split_patch_round_into_HV(
    patch_round_schedules: List[List[Dict[str, Any]]]
) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    per_patch_h: List[List[Dict[str, Any]]] = []
    per_patch_v: List[List[Dict[str, Any]]] = []

    for sched in patch_round_schedules:
        h_seg: List[Dict[str, Any]] = []
        v_seg: List[Dict[str, Any]] = []

        for pass_info in sched:
            if not _pass_has_swaps(pass_info):
                continue
            phase = pass_info["phase"]
            if phase == "H":
                h_seg.append(pass_info)
            else:
                v_seg.append(pass_info)

        per_patch_h.append(h_seg)
        per_patch_v.append(v_seg)

    return per_patch_h, per_patch_v


def _merge_phase_passes(
    per_patch_passes: List[List[Dict[str, Any]]],
    phase_label: str,
) -> List[Dict[str, Any]]:
    num_patches = len(per_patch_passes)
    indices = [0] * num_patches
    merged: List[Dict[str, Any]] = []

    while True:
        candidates = [
            p for p in range(num_patches)
            if indices[p] < len(per_patch_passes[p])
        ]
        if not candidates:
            break

        driver = candidates[0]
        base_pass = per_patch_passes[driver][indices[driver]]
        base_parity = _infer_pass_parity(base_pass)

        new_pass: Dict[str, Any] = {
            "phase": phase_label,
            "h_swaps": [],
            "v_swaps": [],
        }

        def _compatible(pinfo: Dict[str, Any]) -> bool:
            if pinfo["phase"] != phase_label:
                return False
            p_parity = _infer_pass_parity(pinfo)
            if base_parity is None or p_parity is None:
                return False
            return p_parity == base_parity

        # Always consume driver pass
        if phase_label == "H":
            new_pass["h_swaps"].extend(base_pass.get("h_swaps", []))
        else:
            new_pass["v_swaps"].extend(base_pass.get("v_swaps", []))
        indices[driver] += 1

        for patch_idx in candidates[1:]:
            local_idx = indices[patch_idx]
            if local_idx >= len(per_patch_passes[patch_idx]):
                continue
            patch_pass = per_patch_passes[patch_idx][local_idx]
            if not _compatible(patch_pass):
                continue
            if phase_label == "H":
                new_pass["h_swaps"].extend(patch_pass.get("h_swaps", []))
            else:
                new_pass["v_swaps"].extend(patch_pass.get("v_swaps", []))
            indices[patch_idx] += 1

        merged.append(new_pass)

    return merged


def _concatenate_same_grid_schedules(
    sched_a: List[Dict[str, Any]],
    sched_b: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge schedules from patches on the *same* (overlapping) grid.

    When patches share grid cells, their swaps cannot execute in
    parallel and must be serialized.  The output concatenates all H
    passes from A, then all H passes from B, then all V passes from A,
    then all V passes from B.  This guarantees the ``H…H V…V`` phase
    ordering and preserves each patch's internal odd-even structure.
    """
    h_a: List[Dict[str, Any]] = []
    v_a: List[Dict[str, Any]] = []
    h_b: List[Dict[str, Any]] = []
    v_b: List[Dict[str, Any]] = []

    for p in sched_a:
        if not _pass_has_swaps(p):
            continue
        if p["phase"] == "H":
            h_a.append(p)
        else:
            v_a.append(p)

    for p in sched_b:
        if not _pass_has_swaps(p):
            continue
        if p["phase"] == "H":
            h_b.append(p)
        else:
            v_b.append(p)

    return h_a + h_b + v_a + v_b


def _merge_patch_round_schedules(
    patch_round_schedules: List[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Merge per-patch schedules for a single round into a global schedule.

    Output structure: all H passes first (odd-even alternating), then all
    V passes (odd-even alternating).  H-V-H interleaving is FORBIDDEN.

    For patches on *disjoint* grid regions (the normal case within a
    tiling), passes at the same parity-index are merged in parallel.
    """
    if not patch_round_schedules:
        return []

    per_patch_h, per_patch_v = _split_patch_round_into_HV(patch_round_schedules)
    merged_h = _merge_phase_passes(per_patch_h, phase_label="H")
    merged_v = _merge_phase_passes(per_patch_v, phase_label="V")
    merged = merged_h + merged_v

    for pass_info in merged:
        if pass_info["h_swaps"] and pass_info["v_swaps"]:
            raise AssertionError(
                "Merged pass contains both horizontal and vertical swaps"
            )

    return merged


def _merge_patch_schedules(
    patch_schedules: List[List[List[Dict[str, Any]]]],
    R: int,
) -> List[List[Dict[str, Any]]]:
    """
    Given a list of patch schedules (one schedule per patch, each of which is a
    [round][pass] structure), merge them into a single schedule that respects
    odd-even and H/V constraints.
    """
    merged: List[List[Dict[str, Any]]] = [[] for _ in range(R)]
    if not patch_schedules:
        return merged

    for r in range(R):
        round_scheds = [sched[r] for sched in patch_schedules]
        merged[r] = _merge_patch_round_schedules(round_scheds)

    return merged


def _patch_and_route(
    oldArrangementArr: np.ndarray,
    wiseArch: QCCDWiseArch,
    P_arr: List[List[Tuple[int, int]]],
    subgridsize: Tuple[int, int, int],
    active_ions: List[int] = None,
    ignore_initial_reconfig: bool = False,
    base_pmax_in: int = None,
    BTs: Optional[List[Dict[Tuple[int, int], Tuple[Dict[int, Tuple[int, int]], List[Tuple[int, int]]]]]] = None,
    stop_event: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
    max_inner_workers: int | None = None,
) -> List[Tuple[int, int, List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]]]]:
    """
    Patch-based Level-1 slicer. Partition the device into checkerboard patches and
    solve each patch locally via _optimal_QMR_for_WISE. Multiple tiling phases are
    applied (base, vertical offset, horizontal offset) and repeated if needed until
    all gates are realised or progress stalls.

    Returns:
        A list of tiling steps, each consisting of:
            - a round-0 layout snapshot,
            - a merged schedule for round 0,
            - and the per-round set of pairs solved in that tiling.
    """
    R = len(P_arr)
    if R == 0:
        return [], []

    n_rows = wiseArch.n
    n_cols = wiseArch.m * wiseArch.k

    n_c = max(subgridsize[0], 1)
    n_r = max(subgridsize[1], 1)
    inc = subgridsize[2]



    layouts_after: List[np.ndarray] = [np.array(oldArrangementArr, copy=True) for _ in range(R)]
    remaining_pairs: List[List[Tuple[int, int]]] = [list(rp) for rp in P_arr]
    base_pmax = base_pmax_in or 1   # default to 1 so low-P_max (optimal) configs are always explored
    prev_pmax = None

    tiling_steps: List[Tuple[int, int, List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]]]] = []

    max_cycles = 10
    no_progress_cycles = 0
    cycle_idx = 0
    total_remaining = sum(len(rp) for rp in remaining_pairs)
    while total_remaining>0:
        total_remaining = sum(len(rp) for rp in remaining_pairs)
        if total_remaining == 0:
            break

        if n_c < wiseArch.k and (wiseArch.k % n_c != 0):
            n_c += int(np.mod(wiseArch.k, n_c))
        elif n_c > wiseArch.k and (n_c % wiseArch.k != 0):
            n_c = (n_c // wiseArch.k) * wiseArch.k

        patch_w = min(n_c, n_cols)
        patch_h = min(n_r, n_rows)

        # Warn when effective patch covers the entire grid — this
        # defeats Level-2 tiling and forces one large SAT call.
        if cycle_idx == 0 and patch_w >= n_cols and patch_h >= n_rows:
            import logging as _log_mod
            _pr_log = _log_mod.getLogger("wise.qccd.route")
            _pr_log.debug(
                "[_patch_and_route] subgridsize=(%d,%d) covers the "
                "full %dx%d grid as a single patch — SAT will be "
                "slow. Consider subgridsize=(4,3,0) for 4×3 patches.",
                n_c, n_r, n_rows, n_cols,
            )

        tilings: List[Tuple[int, int]] = [(0, 0)]

        # Helper: check if a width is "k-compatible" for WISE blocks
        def _k_compatible(width: int, k: int) -> bool:
            if width == 1:
                return False
            if width <= 0:
                return False
            if width == k:
                return True
            if width > k:
                return (width % k) == 0
            # width < k
            return (k % width) == 0

        # can always add vertical offset
        half_h = patch_h // 2
        if half_h > 0 and (half_h, 0) not in tilings:
            tilings.append((half_h, 0))

        # Optional horizontal offset (half patch width), only if:
        #  - patch_w even
        #  - half_w is k-compatible
        if patch_w % 2 == 0:
            half_w = patch_w // 2
            if (
                half_w > 0
                and _k_compatible(half_w, wiseArch.k)
                and (0, half_w) not in tilings
            ):
                tilings.append((0, half_w))

        logger.info(
            "%s starting patch routing: rounds=%d, total_gates=%d, patch_size=%dx%d, tilings=%d, BTs=%s",
            PATCH_LOG_PREFIX,
            R,
            total_remaining,
            patch_h,
            patch_w,
            len(tilings),
            BTs
        )

        logger.info(
            "%s beginning tiling cycle %d; remaining_gates=%d",
            PATCH_LOG_PREFIX,
            cycle_idx + 1,
            total_remaining,
        )
        cycle_start_remaining = total_remaining

        # Compute total patches across all tilings in this cycle for progress
        _total_patches_in_cycle = 0
        for _off_r, _off_c in tilings:
            _total_patches_in_cycle += len(
                _generate_patch_regions(n_rows, n_cols, patch_h, patch_w, _off_r, _off_c)
            )
        _patches_done_in_cycle = 0

        # Emit fine-grained progress at start of each cycle
        if progress_callback is not None:
            progress_callback(RoutingProgress(
                stage=STAGE_PATCHING,
                current=0,
                total=_total_patches_in_cycle,
                gates_remaining=total_remaining,
                message=f"Cycle {cycle_idx + 1}: 0/{_total_patches_in_cycle} patches",
            ))

        for tiling_idx, (off_r, off_c) in enumerate(tilings):
            if all(len(rp) == 0 for rp in remaining_pairs):
                break

            # Capture pre-tiling layouts for replay verification
            layouts_before_tiling = [la.copy() for la in layouts_after]

            ion_positions: Dict[int, Tuple[int, int]] = {
                int(layouts_after[0][r][c]): (r, c)
                for r in range(n_rows)
                for c in range(n_cols)
            }

            gates_before_tiling = sum(len(rp) for rp in remaining_pairs)
            # Save a deep copy of remaining_pairs before this tiling
            pairs_before_tiling: List[List[Tuple[int, int]]] = [list(rp) for rp in remaining_pairs]

            patch_regions = _generate_patch_regions(n_rows, n_cols, patch_h, patch_w, off_r, off_c)
            logger.info(
                "%s cycle %d tiling %d/%d offset=(%d,%d) patches=%d remaining_before=%d",
                PATCH_LOG_PREFIX,
                cycle_idx + 1,
                tiling_idx + 1,
                len(tilings),
                off_r,
                off_c,
                len(patch_regions),
                gates_before_tiling,
            )

            # Determine BT pins for this tiling, if provided.
            if BTs is not None and BTs:
                key = (cycle_idx, tiling_idx)
                BT_for_tiling: List[Dict[int, Tuple[int, int]]] = []
                pairs_for_tiling: List[Set[Tuple[int, int]]] = []
                for r in range(R):
                    entry = None
                    if r < len(BTs):
                        entry = BTs[r].get(key)
                    if isinstance(entry, tuple):
                        bt_map_full, solved_pairs_full = entry
                    elif isinstance(entry, dict):
                        bt_map_full = entry
                        solved_pairs_full = []
                    elif entry is None:
                        bt_map_full = {}
                        solved_pairs_full = []
                    else:
                        bt_map_full = dict(entry)
                        solved_pairs_full = []
                    BT_for_tiling.append(bt_map_full)
                    pairs_for_tiling.append(set(solved_pairs_full))
            else:
                BT_for_tiling = [dict() for _ in range(R)]
                pairs_for_tiling = [set() for _ in range(R)]

            patch_schedules_this_tiling: List[List[List[Dict[str, Any]]]] = []

            for region in patch_regions:
                patch_pairs, new_remaining = _split_pairs_for_patch(remaining_pairs, ion_positions, region)
                patch_gate_count = sum(len(round_pairs) for round_pairs in patch_pairs)

                r0, c0, r1, c1 = region
                patch_grid = np.array(layouts_after[0][r0:r1, c0:c1], dtype=int, copy=True)
                per_round_counts = [len(round_pairs) for round_pairs in patch_pairs]

                max_gating_zones = _compute_patch_gating_capacity(
                    n=r1-r0,
                    m=c1-c0,
                    col_offset=c0,
                    capacity=wiseArch.k,   
                )

                if max_gating_zones > 0:
                    # We cap *per-round* usage to this patch’s total gating capacity.
                    for ridx in range(R):
                        round_pairs = patch_pairs[ridx]
                        if len(round_pairs) > max_gating_zones:
                            # spill the excess pairs back to new_remaining
                            spill = round_pairs[max_gating_zones:]
                            patch_pairs[ridx] = round_pairs[:max_gating_zones]
                            new_remaining[ridx].extend(spill)

                    # Recompute count after capping
                    patch_gate_count = sum(len(round_pairs) for round_pairs in patch_pairs)
                    per_round_counts = [len(round_pairs) for round_pairs in patch_pairs]

                boundary_adjacent = {
                    "top": r0 > 0,
                    "bottom": r1 < n_rows,
                    "left": c0 > 0,
                    "right": c1 < n_cols,
                }
                cross_boundary_prefs = _compute_cross_boundary_prefs(
                    region=region,
                    ion_positions=ion_positions,
                    pairs_per_round=new_remaining,
                )

                has_boundary_prefs = bool(
                    cross_boundary_prefs
                    and any(pref_round for pref_round in cross_boundary_prefs)
                )

                if patch_gate_count == 0 and not has_boundary_prefs:
                    remaining_pairs = new_remaining
                    # Still count this patch so the patch progress bar advances
                    if progress_callback is not None:
                        _patches_done_in_cycle += 1
                        gates_left = sum(len(rp) for rp in remaining_pairs)
                        progress_callback(RoutingProgress(
                            stage=STAGE_PATCH_SOLVE,
                            current=_patches_done_in_cycle,
                            total=_total_patches_in_cycle,
                            gates_remaining=gates_left,
                            message=f"Patch {_patches_done_in_cycle}/{_total_patches_in_cycle} (skip, 0 gates)",
                        ))
                    continue

                ions_in_patch_pairs = [[] for _ in range(R)]
                for r, pairs in enumerate(patch_pairs):
                    for p in pairs:
                        ions_in_patch_pairs[r].append(p[0])
                        ions_in_patch_pairs[r].append(p[1])

                ions_in_tiling_pairs = [[] for _ in range(R)]
                for r, pairs in enumerate(pairs_for_tiling):
                    for p in pairs:
                        ions_in_tiling_pairs[r].append(p[0])
                        ions_in_tiling_pairs[r].append(p[1])

                BT_patch: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(R)]
                # Track (start_row_local, target_col_local) → ion we kept, per round
                bt_keys_used: List[Dict[Tuple[int, int], int]] = [dict() for _ in range(R)]
                use_bt_softs = False
                for ridx in range(R):
                    bt_full = BT_for_tiling[ridx] if ridx < len(BT_for_tiling) else {}
                    ions_patch = ions_in_patch_pairs[ridx]
                    ions_tiling = ions_in_tiling_pairs[ridx]

                    for ion, (global_r, global_c) in bt_full.items():
                        # # Only pin ions that are consistently in/out of this tiling’s pairs
                        if (ion in ions_patch) != (ion in ions_tiling):
                            continue
                        # Only pin ions whose *target* is inside this patch region
                        if not (r0 <= global_r < r1 and c0 <= global_c < c1):
                            continue

                        # # Local target coords in this patch
                        local_r = global_r - r0
                        local_c = global_c - c0

                        # Compute the ion's *start row* in this patch, to mirror the
                        # pre-sanity check in _optimal_QMR_for_WISE.
                        if ridx==0:
                            init_pos = ion_positions.get(ion)
                            if init_pos is None:
                                continue
                            init_row_global, _ = init_pos
                            start_row_local = init_row_global - r0
                            if not (0 <= start_row_local < (r1 - r0)):
                                # Shouldn't normally happen, but be defensive
                                continue

                            key = (start_row_local, local_c)
                            existing_ion = bt_keys_used[ridx].get(key)

                            if existing_ion is not None and existing_ion != ion:
                                # Conflict: another ion from the same start row already pinned
                                # to this target column in this patch. To keep BT consistent
                                # with _optimal_QMR_for_WISE's pre-check, we skip this one.
                                logger.warning(
                                    "%s BT pin conflict in patch r[%d:%d] c[%d:%d], round=%d: "
                                    "start_row_local=%d, target_col_local=%d; keeping ion %d, "
                                    "dropping ion %d",
                                    PATCH_LOG_PREFIX,
                                    r0,
                                    r1,
                                    c0,
                                    c1,
                                    ridx,
                                    start_row_local,
                                    local_c,
                                    existing_ion,
                                    ion,
                                )
                                continue

                        # No conflict: record and keep the pin.
                        bt_keys_used[ridx][key] = ion
                        BT_patch[ridx][ion] = (local_r, local_c)
                        use_bt_softs=True

                logger.info(
                    "%s solving patch r[%d:%d] c[%d:%d] gates=%d per_round=%s boundary_prefs=%s, BTs=%s",
                    PATCH_LOG_PREFIX,
                    r0,
                    r1,
                    c0,
                    c1,
                    patch_gate_count,
                    per_round_counts,
                    cross_boundary_prefs,
                    BT_patch
                )

                patch_start = time.time()
                try:
                    patch_layouts, patch_schedule, prev_pmax = GlobalReconfigurations._optimal_QMR_for_WISE(
                        patch_grid,
                        patch_pairs,
                        k=wiseArch.k,
                        BT=BT_patch,
                        active_ions=active_ions,
                        wB_col=0,
                        wB_row=0,
                        full_P_arr=remaining_pairs,
                        ignore_initial_reconfig=(ignore_initial_reconfig and tiling_idx==0 and cycle_idx==0),
                        base_pmax_in=base_pmax,
                        prev_pmax=prev_pmax,
                        grid_origin=(r0, c0),
                        boundary_adjacent=boundary_adjacent,
                        cross_boundary_prefs=cross_boundary_prefs,
                        bt_soft=use_bt_softs,
                        parent_stop_event=stop_event,
                        progress_callback=progress_callback,
                        max_inner_workers=max_inner_workers,
                    )
                except NoFeasibleLayoutError as exc:
                    logger.warning(
                        "%s ERROR solving patch r[%d:%d] c[%d:%d] max_gate_zones=%d gates=%d: %s",
                        PATCH_LOG_PREFIX,
                        r0,
                        r1,
                        c0,
                        c1,
                        max_gating_zones,
                        patch_gate_count,
                        exc,
                    )
                    # add patch pairs back to our remaining pairs
                    for ridx in range(R):
                        round_pairs = patch_pairs[ridx]
                        new_remaining[ridx].extend(round_pairs)
                    # Still count this patch so the patch progress bar advances
                    if progress_callback is not None:
                        _patches_done_in_cycle += 1
                        gates_left = sum(len(rp) for rp in new_remaining)
                        progress_callback(RoutingProgress(
                            stage=STAGE_PATCH_SOLVE,
                            current=_patches_done_in_cycle,
                            total=_total_patches_in_cycle,
                            gates_remaining=gates_left,
                            message=f"Patch {_patches_done_in_cycle}/{_total_patches_in_cycle} (infeasible)",
                        ))
                    remaining_pairs = new_remaining
                    continue
                    
                patch_elapsed = time.time() - patch_start

                initial_positions_patch: Dict[int, Tuple[int, int]] = {}
                final_positions_patch: Dict[int, Tuple[int, int]] = {}
                for dr in range(r1 - r0):
                    for dc in range(c1 - c0):
                        ion_initial = int(patch_grid[dr][dc])
                        ion_final = int(patch_layouts[0][dr][dc]) if patch_layouts else ion_initial
                        initial_positions_patch[ion_initial] = (r0 + dr, c0 + dc)
                        final_positions_patch[ion_final] = (r0 + dr, c0 + dc)

                moved_ions: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = []
                for ion, origin in initial_positions_patch.items():
                    final_pos = final_positions_patch.get(ion)
                    if final_pos is not None and final_pos != origin:
                        moved_ions.append((ion, origin, final_pos))

                move_detail = ", ".join(
                    [f"ion {ion}: {src}->{dst}" for ion, src, dst in moved_ions[:6]]
                )
                logger.info(
                    "%s patch r[%d:%d] c[%d:%d] solved in %.2fs; moved_ions=%d; details=%s",
                    PATCH_LOG_PREFIX,
                    r0,
                    r1,
                    c0,
                    c1,
                    patch_elapsed,
                    len(moved_ions),
                    move_detail if move_detail else "none",
                )
                if PATCH_VERBOSE_MOVES and len(moved_ions) > 6:
                    extra_detail = ", ".join(
                        [f"ion {ion}: {src}->{dst}" for ion, src, dst in moved_ions[6:]]
                    )
                    logger.debug("%s additional moves: %s", PATCH_LOG_PREFIX, extra_detail)

                for ridx in range(R):
                    layouts_after[ridx][r0:r1, c0:c1] = patch_layouts[ridx]

                # ── Per-patch replay verification ──────────────────
                # Replay THIS patch's schedule on the pre-tiling
                # layout and check that only this patch's cells
                # change, and that they match the SAT-decoded layout.
                for ridx in range(R):
                    if patch_schedule and ridx < len(patch_schedule) and patch_schedule[ridx]:
                        pp_replay = _simulate_schedule_replay(
                            layouts_before_tiling[ridx],
                            patch_schedule[ridx],
                        )
                        # Check patch region only
                        pp_region = pp_replay[r0:r1, c0:c1]
                        if not np.array_equal(pp_region, patch_layouts[ridx]):
                            pp_diff = int(np.count_nonzero(
                                pp_region != patch_layouts[ridx]
                            ))
                            logger.error(
                                "%s PER-PATCH REPLAY MISMATCH: patch "
                                "r[%d:%d] c[%d:%d] round %d: %d/%d "
                                "cells differ between schedule-replay "
                                "and SAT-decoded layout. This means "
                                "the SAT schedule is NOT self-consistent "
                                "with the SAT layout for this patch!",
                                PATCH_LOG_PREFIX, r0, r1, c0, c1,
                                ridx, pp_diff, patch_layouts[ridx].size,
                            )
                            # Log detailed cell diffs (first 10)
                            _pp_diffs_logged = 0
                            for _dr in range(r1 - r0):
                                for _dc in range(c1 - c0):
                                    if pp_region[_dr, _dc] != patch_layouts[ridx][_dr, _dc]:
                                        logger.error(
                                            "%s   cell (%d,%d) local=(%d,%d): "
                                            "replay=%d, SAT=%d, before=%d",
                                            PATCH_LOG_PREFIX,
                                            r0 + _dr, c0 + _dc, _dr, _dc,
                                            int(pp_region[_dr, _dc]),
                                            int(patch_layouts[ridx][_dr, _dc]),
                                            int(layouts_before_tiling[ridx][r0 + _dr, c0 + _dc]),
                                        )
                                        _pp_diffs_logged += 1
                                        if _pp_diffs_logged >= 10:
                                            break
                                if _pp_diffs_logged >= 10:
                                    break
                        # Also check that cells OUTSIDE this patch are
                        # unchanged (schedule shouldn't affect other regions)
                        _outside_mask = np.ones_like(pp_replay, dtype=bool)
                        _outside_mask[r0:r1, c0:c1] = False
                        _outside_changed = np.count_nonzero(
                            pp_replay[_outside_mask]
                            != layouts_before_tiling[ridx][_outside_mask]
                        )
                        if _outside_changed > 0:
                            logger.error(
                                "%s PER-PATCH BLEED: patch r[%d:%d] "
                                "c[%d:%d] round %d: %d cells OUTSIDE "
                                "patch changed by schedule replay! "
                                "This means swap coords leak across "
                                "patch boundaries.",
                                PATCH_LOG_PREFIX, r0, r1, c0, c1,
                                ridx, _outside_changed,
                            )

                patch_schedules_this_tiling.append(patch_schedule)

                remaining_pairs = new_remaining

                # Emit progress after each patch solve
                if progress_callback is not None:
                    _patches_done_in_cycle += 1
                    gates_left = sum(len(rp) for rp in remaining_pairs)
                    progress_callback(RoutingProgress(
                        stage=STAGE_PATCH_SOLVE,
                        current=_patches_done_in_cycle,
                        total=_total_patches_in_cycle,
                        gates_remaining=gates_left,
                        elapsed_seconds=patch_elapsed,
                        message=f"Patch {_patches_done_in_cycle}/{_total_patches_in_cycle} ({gates_left} gates left)",
                        extra={"patch_region": (r0, c0, r1, c1), "moved_ions": len(moved_ions)},
                    ))

            merged_tiling_schedule = _merge_patch_schedules(patch_schedules_this_tiling, R)

            # ── Replay verification (per round) ────────────────────
            # Verify that the merged schedule, when replayed on the
            # pre-tiling layout, actually produces the SAT-model
            # layout.  If they diverge, use the replay result so
            # that the snapshot is self-consistent.
            for ridx in range(R):
                if merged_tiling_schedule[ridx]:
                    replay_r = _simulate_schedule_replay(
                        layouts_before_tiling[ridx],
                        merged_tiling_schedule[ridx],
                    )
                    if not np.array_equal(replay_r, layouts_after[ridx]):
                        diff_count = int(
                            np.count_nonzero(replay_r != layouts_after[ridx])
                        )
                        logger.warning(
                            "%s patch-and-route round %d: merged schedule "
                            "replay diverges from SAT model layout "
                            "(%d/%d cells differ). Using replay result.",
                            PATCH_LOG_PREFIX, ridx, diff_count,
                            layouts_after[ridx].size,
                        )
                        layouts_after[ridx] = replay_r

            # Compute which pairs were solved in this tiling, per round
            solved_pairs_per_round: List[List[Tuple[int, int]]] = []
            for ridx in range(R):
                before_set = set(pairs_before_tiling[ridx])
                after_set = set(remaining_pairs[ridx])
                solved_here = list(before_set - after_set)
                solved_pairs_per_round.append(solved_here)

            tiling_snapshot = [
                (la.copy(), copy.deepcopy(sched), list(sp))
                for la, sched, sp in zip(layouts_after, merged_tiling_schedule, solved_pairs_per_round)
            ]
            tiling_steps.append((cycle_idx, tiling_idx, tiling_snapshot))

            gates_after_tiling = sum(len(rp) for rp in remaining_pairs)
            solved_this_tiling = gates_before_tiling - gates_after_tiling
            logger.info(
                "%s completed tiling %d: solved_gates=%d, remaining_gates=%d",
                PATCH_LOG_PREFIX,
                tiling_idx + 1,
                solved_this_tiling,
                gates_after_tiling,
            )

        gates_after_cycle = sum(len(rp) for rp in remaining_pairs)
        solved_cycle = cycle_start_remaining - gates_after_cycle
        logger.info(
            "%s completed cycle %d: solved=%d, remaining=%d",
            PATCH_LOG_PREFIX,
            cycle_idx + 1,
            solved_cycle,
            gates_after_cycle,
        )
        fully_global_patch = (patch_h >= n_rows) and (patch_w >= n_cols)
        if gates_after_cycle == 0:
            break
        if solved_cycle == 0:
            # Optional TODO: if complexity is terrible , Expand the subgrid when there has been no progress
            # n_c += max(inc , min(n_c, wiseArch.k))
            # n_r += inc
            no_progress_cycles += 1
            # When inc==0, retrying with identical params is futile — break after
            # one no-progress cycle instead of wasting max_cycles * SAT-solver time.
            if inc == 0 and no_progress_cycles >= 1:
                logger.info(
                    "%s no progress in cycle %d with inc=0; stopping (retrying identical config is futile)",
                    PATCH_LOG_PREFIX,
                    cycle_idx + 1,
                )
                break
            if no_progress_cycles >= max_cycles or fully_global_patch:
                logger.info(
                    "%s no progress in cycle %d; stopping patch routing loop",
                    PATCH_LOG_PREFIX,
                    cycle_idx + 1,
                )
                break
        else:
            no_progress_cycles = 0
        cycle_idx += 1
        n_c += max(inc , min(n_c, wiseArch.k))
        n_r += inc

    unresolved = sum(len(rp) for rp in remaining_pairs)
    if unresolved > 0:
        sample = []
        for ridx, rp in enumerate(remaining_pairs):
            for pair in rp:
                sample.append((ridx, pair))
                if len(sample) >= 5:
                    break
            if len(sample) >= 5:
                break
        logger.info(
            "%s %d gate(s) remain unresolved after patch routing. Example (round, (ion_a, ion_b)): %s",
            PATCH_LOG_PREFIX,
            unresolved,
            sample,
        )
    else:
        logger.info("%s finished patch routing successfully; all gates covered.", PATCH_LOG_PREFIX)

    return tiling_steps


def _escalate_sat_for_ms_reconfig(
    old_layout: np.ndarray,
    wiseArch: "QCCDWiseArch",
    target_layout: np.ndarray,
    subgridsize: Tuple[int, int, int],
    base_pmax_in: Optional[int] = None,
    stop_event: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
    max_inner_workers: Optional[int] = None,
    max_attempts: int = 3,
) -> Optional[List[Tuple[np.ndarray, Optional[List[Dict[str, Any]]], list]]]:
    """Try full-grid SAT with escalating *pmax* for MS-gate reconfig.

    This is the last-resort escalation for P1 (never heuristic for MS
    gates).  We call ``_rebuild_schedule_for_layout`` with increasing
    *pmax* and ``allow_heuristic_fallback=False``.  If any attempt
    produces a valid schedule for the final cycle, we return the full
    list of snapshots.  Otherwise we return ``None`` and the caller
    must raise an error.
    """
    _base = base_pmax_in or 1
    for attempt in range(max_attempts):
        pmax = _base + (attempt + 1) * 2          # +2, +4, +6 …
        logger.info(
            "%s SAT escalation attempt %d/%d with pmax=%d",
            PATCH_LOG_PREFIX, attempt + 1, max_attempts, pmax,
        )
        try:
            snaps = _rebuild_schedule_for_layout(
                np.array(old_layout, copy=True),
                wiseArch,
                np.array(target_layout, copy=True),
                subgridsize=subgridsize,
                base_pmax_in=pmax,
                allow_heuristic_fallback=False,
                stop_event=stop_event,
                progress_callback=progress_callback,
                max_inner_workers=max_inner_workers,
            )
        except Exception as exc:
            logger.warning(
                "%s SAT escalation attempt %d failed: %s",
                PATCH_LOG_PREFIX, attempt + 1, exc,
            )
            continue
        if snaps:
            # Check the last snapshot actually has a schedule
            _last_sched = snaps[-1][1]
            if _last_sched is not None:
                return snaps
            logger.warning(
                "%s SAT escalation attempt %d: last snapshot has "
                "schedule=None; trying higher pmax",
                PATCH_LOG_PREFIX, attempt + 1,
            )
    return None


def _rebuild_schedule_for_layout(
    oldArrangementArr: np.ndarray,
    wiseArch: QCCDWiseArch,
    target_layout: np.ndarray,
    subgridsize: Tuple[int, int, int],
    base_pmax_in: int = None,
    stop_event: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
    max_inner_workers: int | None = None,
    allow_heuristic_fallback: bool = True,
) -> List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]]:
    """
    Given a current global layout (oldArrangementArr) and a desired global layout
    (target_layout), rebuild a sequence of SAT-based reconfiguration schedules
    that progressively maps the former to the latter.

    We reuse the patching machinery from _patch_and_route but run **multiple
    cycles**. In each cycle we:
      - compute, for each patch, a set of BT pins such that no two reserved ions
        share the same (start_row_local, target_col_local) key (the condition
        enforced by _optimal_QMR_for_WISE);
      - call _optimal_QMR_for_WISE with P_arr=[[]] and these BT pins, producing
        a layout-only reconfiguration for that cycle;
      - update the current layout with the result.

    After each cycle we get closer to the target layout and can pin more ions
    without violating the start-row/target-column uniqueness constraint. Once
    all ions are effectively pinned (or no further progress is possible), we
    stop and return a list of tiling snapshots:

        [
          (layout_after_cycle0, merged_schedule_cycle0, []),
          (layout_after_cycle1, merged_schedule_cycle1, []),
          ...
        ]

    The caller should apply these schedules in order to reach target_layout.
    """
    R = 1  # single logical "round" per cycle (no MS gates, only BT pins)
    n_rows = wiseArch.n
    n_cols = wiseArch.m * wiseArch.k
    inc = 1

    if target_layout.shape != oldArrangementArr.shape:
        raise ValueError(
            f"_rebuild_schedule_for_layout: shape mismatch: old={oldArrangementArr.shape}, target={target_layout.shape}"
        )

    # Derive patch size from subgridsize (same convention as _patch_and_route)
    n_c = max(subgridsize[0], 1)
    n_r = max(subgridsize[1], 1)
    patch_w = min(n_c, n_cols)
    patch_h = min(n_r, n_rows)

    # Current layout that we iteratively push towards target_layout
    current_layout = np.array(oldArrangementArr, copy=True)

    # Pre-build ion -> global target position map (target_layout is constant)
    ion_target_pos: Dict[int, Tuple[int, int]] = {
        int(target_layout[r, c]): (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
    }

    snapshots: List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]] = []
    max_cycles = max(n_rows, n_cols)

    # Track how many grid positions differ from the target layout so we can
    # detect lack of progress and avoid burning many SAT calls with no gain.
    prev_mismatch = int(np.count_nonzero(current_layout != target_layout))
    logger.info(
        "%s schedule-only rebuild: initial mismatch cells=%d",
        PATCH_LOG_PREFIX,
        prev_mismatch,
    )

    for cycle_idx in range(max_cycles):
        if np.array_equal(current_layout, target_layout):
            logger.info(
                "%s schedule-only rebuild converged to target layout in %d cycle(s)",
                PATCH_LOG_PREFIX,
                cycle_idx,
            )
            break
        
        # ---- Checkerboard offset tilings (matching _patch_and_route) ----
        def _k_compatible(width: int, k_val: int) -> bool:
            if width <= 0 or width == 1:
                return False
            if width == k_val:
                return True
            if width > k_val:
                return (width % k_val) == 0
            return (k_val % width) == 0

        tilings: List[Tuple[int, int]] = [(0, 0)]
        half_h = patch_h // 2
        if half_h > 0 and (half_h, 0) not in tilings:
            tilings.append((half_h, 0))
        if patch_w % 2 == 0:
            half_w = patch_w // 2
            if (
                half_w > 0
                and _k_compatible(half_w, wiseArch.k)
                and (0, half_w) not in tilings
            ):
                tilings.append((0, half_w))

        mismatch_before = int(np.count_nonzero(current_layout != target_layout))

        cycle_total_pins = 0

        # FM2 fix: track per-ion nudge direction across tilings within
        # this cycle to prevent oscillation.
        nudge_history: Dict[int, Tuple[int, int]] = {}

        for tiling_idx, (off_r, off_c) in enumerate(tilings):
            if np.array_equal(current_layout, target_layout):
                break

            patch_regions = _generate_patch_regions(n_rows, n_cols, patch_h, patch_w, off_r, off_c)

            # Build ion -> position map from the current layout
            ion_positions: Dict[int, Tuple[int, int]] = {
                int(current_layout[r, c]): (r, c)
                for r in range(n_rows)
                for c in range(n_cols)
            }

            layouts_after_local: List[np.ndarray] = [np.array(current_layout, copy=True)]
            patch_schedules_this_tiling: List[List[List[Dict[str, Any]]]] = []
            tiling_pins = 0

            logger.info(
                "%s schedule-only rebuild: cycle %d tiling %d/%d offset=(%d,%d)",
                PATCH_LOG_PREFIX,
                cycle_idx,
                tiling_idx + 1,
                len(tilings),
                off_r,
                off_c,
            )

            for region in patch_regions:
                r0, c0, r1, c1 = region
                patch_grid = np.array(
                    layouts_after_local[0][r0:r1, c0:c1], dtype=int, copy=True
                )

                # ══ FM1+FM2 fix: Distance-sorted BT pin assignment ══
                # Build ALL candidate pins (exact-target + directional nudge)
                # into a single list, sort by Manhattan distance (shortest
                # first — cheapest to pin), then greedily assign respecting
                # both uniqueness constraints.  This maximises pin count
                # and prevents boundary-clamping oscillation.
                BT_patch: List[Dict[int, Tuple[int, int]]] = [dict()]
                bt_map = BT_patch[0]
                used_keys: Dict[Tuple[int, int], int] = {}
                used_targets: Dict[Tuple[int, int], int] = {}

                # Build row_of from patch_grid (same as SAT validator does)
                patch_row_of: Dict[int, int] = {}
                patch_col_of: Dict[int, int] = {}
                for lr in range(r1 - r0):
                    for lc in range(c1 - c0):
                        ion_id = int(patch_grid[lr, lc])
                        if ion_id != 0:
                            patch_row_of[ion_id] = lr
                            patch_col_of[ion_id] = lc

                patch_h_local = r1 - r0
                patch_w_local = c1 - c0

                # Collect ALL candidate pins: (manhattan_dist, ion, start_row, target_row, target_col, is_nudge)
                _candidates: list = []

                # Pass 1 candidates: ions whose global target is INSIDE this patch
                for dr in range(patch_h_local):
                    for dc in range(patch_w_local):
                        ionidx_target = int(target_layout[r0 + dr, c0 + dc])
                        if ionidx_target == 0:
                            continue
                        start_row = patch_row_of.get(ionidx_target)
                        if start_row is None:
                            continue  # Ion not in this patch
                        start_col = patch_col_of.get(ionidx_target, 0)
                        dist = abs(start_row - dr) + abs(start_col - dc)
                        _candidates.append((dist, ionidx_target, start_row, dr, dc, False))

                # Pass 2 candidates: ions in this patch whose global target
                # is OUTSIDE — compute directional nudge toward target
                for lr in range(patch_h_local):
                    for lc in range(patch_w_local):
                        ion_here = int(patch_grid[lr, lc])
                        if ion_here == 0:
                            continue
                        tgt_global = ion_target_pos.get(ion_here)
                        if tgt_global is None:
                            continue
                        tgt_r, tgt_c = tgt_global
                        # Skip if target is inside this patch (handled by Pass 1)
                        if r0 <= tgt_r < r1 and c0 <= tgt_c < c1:
                            continue
                        # Skip if already at global target
                        if tgt_r == r0 + lr and tgt_c == c0 + lc:
                            continue

                        # FM2 fix: directional nudge — move TOWARD
                        # target by max distance within patch
                        dir_r = 1 if tgt_r > r0 + lr else (-1 if tgt_r < r0 + lr else 0)
                        dir_c = 1 if tgt_c > c0 + lc else (-1 if tgt_c < c0 + lc else 0)

                        # Anti-oscillation: skip if this would reverse
                        # the previous nudge direction on BOTH axes
                        prev_dir = nudge_history.get(ion_here)
                        if prev_dir is not None:
                            prev_dr, prev_dc = prev_dir
                            if (prev_dr != 0 and dir_r != 0 and prev_dr == -dir_r and
                                    prev_dc != 0 and dir_c != 0 and prev_dc == -dir_c):
                                continue

                        # Compute nudge target: move toward global target
                        # by as much as possible within patch bounds
                        nudge_r = lr + dir_r * min(
                            abs(tgt_r - (r0 + lr)), patch_h_local - 1)
                        nudge_c = lc + dir_c * min(
                            abs(tgt_c - (c0 + lc)), patch_w_local - 1)
                        nudge_r = max(0, min(nudge_r, patch_h_local - 1))
                        nudge_c = max(0, min(nudge_c, patch_w_local - 1))

                        if nudge_r == lr and nudge_c == lc:
                            continue

                        dist = abs(tgt_r - (r0 + lr)) + abs(tgt_c - (c0 + lc))
                        _candidates.append((dist, ion_here, lr, nudge_r, nudge_c, True))

                # Sort by Manhattan distance (shortest first = easiest to route,
                # leaves more flexibility for harder ions)
                _candidates.sort(key=lambda x: x[0])

                # Greedy assignment respecting both uniqueness constraints
                for _dist, _ion, _sr, _tr, _tc, _is_nudge in _candidates:
                    if _ion in bt_map:
                        continue  # already pinned
                    key = (_sr, _tc)  # SAT precondition: (start_row, target_col)
                    if key in used_keys and used_keys[key] != _ion:
                        continue
                    tgt_key = (_tr, _tc)  # target position uniqueness
                    if tgt_key in used_targets and used_targets[tgt_key] != _ion:
                        continue
                    used_keys[key] = _ion
                    used_targets[tgt_key] = _ion
                    bt_map[_ion] = (_tr, _tc)
                    # Record nudge direction for anti-oscillation
                    if _is_nudge:
                        dir_r = 1 if _tr > _sr else (-1 if _tr < _sr else 0)
                        dir_c = 1 if _tc > patch_col_of.get(_ion, 0) else (
                            -1 if _tc < patch_col_of.get(_ion, 0) else 0)
                        nudge_history[_ion] = (dir_r, dir_c)

                if not bt_map:
                    continue

                tiling_pins += len(bt_map)

                boundary_adjacent = {
                    "top": r0 > 0,
                    "bottom": r1 < n_rows,
                    "left": c0 > 0,
                    "right": c1 < n_cols,
                }
                cross_boundary_prefs: List[Dict[int, Set[str]]] = [dict()]

                logger.info(
                    "%s rebuilding schedule-only patch (cycle %d, tiling %d) r[%d:%d] c[%d:%d] (no gates, BT pins=%d)",
                    PATCH_LOG_PREFIX,
                    cycle_idx,
                    tiling_idx,
                    r0,
                    r1,
                    c0,
                    c1,
                    len(bt_map),
                )

                try:
                    patch_layouts, patch_schedule, _ = GlobalReconfigurations._optimal_QMR_for_WISE(
                        patch_grid,
                        P_arr=[[]],  # no MS gates; layout-only reconfig
                        k=wiseArch.k,
                        BT=BT_patch,
                        active_ions=None,
                        wB_col=0,
                        wB_row=0,
                        full_P_arr=[[]],
                        ignore_initial_reconfig=False,
                        base_pmax_in=base_pmax_in,
                        prev_pmax=None,
                        grid_origin=(r0, c0),
                        boundary_adjacent=boundary_adjacent,
                        cross_boundary_prefs=cross_boundary_prefs,
                        bt_soft=False,
                        parent_stop_event=stop_event,
                        progress_callback=progress_callback,
                        max_inner_workers=max_inner_workers,
                    )
                except NoFeasibleLayoutError:
                    # Retry with soft BT constraints (relaxable pins)
                    logger.info(
                        "%s schedule-only patch (cycle %d, tiling %d) r[%d:%d] c[%d:%d]: "
                        "hard BT infeasible, retrying with bt_soft=True",
                        PATCH_LOG_PREFIX, cycle_idx, tiling_idx,
                        r0, r1, c0, c1,
                    )
                    try:
                        patch_layouts, patch_schedule, _ = GlobalReconfigurations._optimal_QMR_for_WISE(
                            patch_grid,
                            P_arr=[[]],
                            k=wiseArch.k,
                            BT=BT_patch,
                            active_ions=None,
                            wB_col=0,
                            wB_row=0,
                            full_P_arr=[[]],
                            ignore_initial_reconfig=False,
                            base_pmax_in=base_pmax_in,
                            prev_pmax=None,
                            grid_origin=(r0, c0),
                            boundary_adjacent=boundary_adjacent,
                            cross_boundary_prefs=cross_boundary_prefs,
                            bt_soft=True,
                            parent_stop_event=stop_event,
                            progress_callback=progress_callback,
                            max_inner_workers=max_inner_workers,
                        )
                    except NoFeasibleLayoutError:
                        # Even soft BT failed – try with fewer pins
                        # Reduce to only pins where the ion is already in the
                        # correct row (most likely to succeed).
                        reduced_bt: Dict[int, Tuple[int, int]] = {}
                        for ion_id, (tr, tc) in bt_map.items():
                            pos = ion_positions.get(ion_id)
                            if pos is not None:
                                sr_local = pos[0] - r0
                                if sr_local == tr:
                                    reduced_bt[ion_id] = (tr, tc)
                        if reduced_bt and len(reduced_bt) < len(bt_map):
                            logger.info(
                                "%s retrying with reduced pins (%d -> %d)",
                                PATCH_LOG_PREFIX, len(bt_map), len(reduced_bt),
                            )
                            try:
                                patch_layouts, patch_schedule, _ = GlobalReconfigurations._optimal_QMR_for_WISE(
                                    patch_grid,
                                    P_arr=[[]],
                                    k=wiseArch.k,
                                    BT=[reduced_bt],
                                    active_ions=None,
                                    wB_col=0,
                                    wB_row=0,
                                    full_P_arr=[[]],
                                    ignore_initial_reconfig=False,
                                    base_pmax_in=base_pmax_in,
                                    prev_pmax=None,
                                    grid_origin=(r0, c0),
                                    boundary_adjacent=boundary_adjacent,
                                    cross_boundary_prefs=cross_boundary_prefs,
                                    bt_soft=True,
                                    parent_stop_event=stop_event,
                                    progress_callback=progress_callback,
                                    max_inner_workers=max_inner_workers,
                                )
                            except NoFeasibleLayoutError:
                                logger.info(
                                    "%s schedule-only patch (cycle %d, tiling %d) r[%d:%d] c[%d:%d]: "
                                    "all BT strategies failed, skipping patch",
                                    PATCH_LOG_PREFIX, cycle_idx, tiling_idx,
                                    r0, r1, c0, c1,
                                )
                                continue
                        else:
                            logger.info(
                                "%s schedule-only patch (cycle %d, tiling %d) r[%d:%d] c[%d:%d]: "
                                "soft BT also infeasible, skipping patch",
                                PATCH_LOG_PREFIX, cycle_idx, tiling_idx,
                                r0, r1, c0, c1,
                            )
                            continue

                # Update the global layout snapshot with this patch's result.
                layouts_after_local[0][r0:r1, c0:c1] = patch_layouts[0]
                patch_schedules_this_tiling.append(patch_schedule)

            cycle_total_pins += tiling_pins

            if tiling_pins == 0 or not patch_schedules_this_tiling:
                logger.info(
                    "%s schedule-only rebuild: tiling %d made no progress (pins=%d); trying next tiling.",
                    PATCH_LOG_PREFIX,
                    tiling_idx,
                    tiling_pins,
                )
                continue

            # Merge patch schedules into a single global schedule for this tiling.
            merged_schedule_all_rounds = _merge_patch_schedules(patch_schedules_this_tiling, R)
            merged_schedule_round0 = merged_schedule_all_rounds[0] if merged_schedule_all_rounds else []

            # ── Replay verification ──────────────────────────────
            # Verify the merged schedule actually produces the expected
            # layout by simulating the replay on the pre-tiling layout.
            # If they diverge, use the replay result (which is guaranteed
            # self-consistent with the schedule) so that downstream
            # execution never encounters a layout/schedule mismatch.
            replay_result = _simulate_schedule_replay(current_layout, merged_schedule_round0)
            if not np.array_equal(replay_result, layouts_after_local[0]):
                diff_count = int(np.count_nonzero(replay_result != layouts_after_local[0]))
                logger.warning(
                    "%s schedule-only rebuild: merged schedule replay diverges "
                    "from SAT model layout (%d/%d cells differ). Using replay "
                    "result as snapshot layout for self-consistency.",
                    PATCH_LOG_PREFIX, diff_count, current_layout.size,
                )
                layouts_after_local[0] = replay_result

            current_layout = layouts_after_local[0]
            snapshots.append(
                (current_layout.copy(), merged_schedule_round0, [])
            )

        mismatch_after = int(np.count_nonzero(current_layout != target_layout))
        logger.info(
            "%s schedule-only rebuild: finished cycle %d; layout_matches_target=%s, mismatch_before=%d, mismatch_after=%d",
            PATCH_LOG_PREFIX,
            cycle_idx,
            np.array_equal(current_layout, target_layout),
            mismatch_before,
            mismatch_after,
        )

        if cycle_total_pins == 0:
            logger.info(
                "%s schedule-only rebuild made no progress in cycle %d "
                "(total_pins=%d); stopping early.",
                PATCH_LOG_PREFIX,
                cycle_idx,
                cycle_total_pins,
            )
            break

        # If mismatch is not strictly decreasing, grow patches and retry.
        # Only stop if the patch already covers the full grid (nowhere to expand).
        fully_global = (patch_h >= n_rows) and (patch_w >= n_cols)
        if mismatch_after >= mismatch_before:
            if fully_global:
                logger.info(
                    "%s schedule-only rebuild: no improvement with full-grid patch "
                    "in cycle %d (mismatch %d -> %d); stopping.",
                    PATCH_LOG_PREFIX,
                    cycle_idx,
                    mismatch_before,
                    mismatch_after,
                )
                break
            else:
                logger.info(
                    "%s schedule-only rebuild: plateau in cycle %d "
                    "(mismatch %d -> %d); growing patches.",
                    PATCH_LOG_PREFIX,
                    cycle_idx,
                    mismatch_before,
                    mismatch_after,
                )

        prev_mismatch = mismatch_after
        # Grow patches towards full grid; k-align column width
        patch_h = min(patch_h + 1, n_rows)
        new_w = patch_w + max(1, wiseArch.k)
        # Round up to k boundary for WISE compatibility
        if new_w % wiseArch.k != 0:
            new_w = ((new_w // wiseArch.k) + 1) * wiseArch.k
        patch_w = min(new_w, n_cols)

    if not np.array_equal(current_layout, target_layout):
        mismatch_remaining = int(np.count_nonzero(current_layout != target_layout))

        if not allow_heuristic_fallback:
            # ── MS-gate context: heuristic fallback is NOT allowed ──
            # Retry with a full-grid SAT solve covering the entire
            # grid at escalating pmax values.  This is expensive but
            # guarantees optimal SAT-based routing.
            logger.warning(
                "%s schedule-only rebuild could not converge "
                "(%d/%d cells mismatched after %d cycle(s)). "
                "allow_heuristic_fallback=False — retrying with "
                "full-grid SAT at escalating pmax.",
                PATCH_LOG_PREFIX,
                mismatch_remaining,
                current_layout.size,
                len(snapshots),
            )
            _escalation_pmax = max(base_pmax_in or 1, 2)
            for _esc_attempt in range(3):
                _esc_pmax = _escalation_pmax + _esc_attempt * 2
                try:
                    _esc_snapshots = _patch_and_route(
                        np.array(current_layout, copy=True),
                        wiseArch,
                        [[]],  # no MS pairs — BT-only
                        target_layout,
                        subgridsize=(n_rows, n_cols, 1),  # full grid as one patch
                        base_pmax_in=_esc_pmax,
                        stop_event=stop_event,
                        enable_bt=True,
                        progress_callback=progress_callback,
                        max_inner_workers=max_inner_workers,
                    )
                except Exception:
                    _esc_snapshots = []
                if _esc_snapshots:
                    _esc_layout = _esc_snapshots[-1][0]
                    if np.array_equal(_esc_layout, target_layout):
                        snapshots.extend(_esc_snapshots)
                        logger.info(
                            "%s full-grid SAT retry succeeded at "
                            "pmax=%d (attempt %d).",
                            PATCH_LOG_PREFIX, _esc_pmax,
                            _esc_attempt + 1,
                        )
                        return snapshots
                    else:
                        snapshots.extend(_esc_snapshots)
                        current_layout = np.array(_esc_layout, copy=True)
                        if np.array_equal(current_layout, target_layout):
                            return snapshots

            # Heuristic fallback is FORBIDDEN — SAT escalation failed.
            # This applies to MS-gate, return-reconfig, and transition
            # contexts. SAT routing MUST succeed; heuristic fallback is
            # only allowed for cache-replay contexts.
            raise ValueError(
                f"{PATCH_LOG_PREFIX} SAT routing FAILED "
                f"after full-grid SAT escalation "
                f"(3 attempts). {mismatch_remaining}/{current_layout.size} "
                f"cells still mismatched after {len(snapshots)} cycle(s). "
                f"Heuristic fallback is FORBIDDEN for this context. "
                f"This indicates a fundamental problem in _patch_and_route "
                f"that must be fixed."
            )

        # allow_heuristic_fallback=True path: cache replay / route-back
        # context where heuristic is acceptable.
        logger.warning(
            "%s schedule-only rebuild could not fully converge to target layout "
            "(%d/%d cells still mismatched after %d cycle(s)). "
            "Using heuristic odd-even transposition sort on the FULL "
            "grid (schedule=None → physicalOperation Phase B/C/D). "
            "This is acceptable for cache replay / route-back context.",
            PATCH_LOG_PREFIX,
            mismatch_remaining,
            current_layout.size,
            len(snapshots),
        )
        # Append a final snapshot whose schedule is None.  When
        # physicalOperation receives schedule=None it skips the SAT path
        # and falls through to the deterministic Phase B/C/D heuristic
        # (odd-even transposition sort) which is O(m+n) passes and
        # *always* succeeds for any permutation.
        snapshots.append(
            (target_layout.copy(), None, [])
        )

    return snapshots

def _apply_layout_as_reconfiguration(
    arch: QCCDArch,
    wiseArch: QCCDWiseArch,
    oldArrangementArr: np.ndarray,
    newArrangementArr: np.ndarray,
    layout_after: np.ndarray,
    allOps: List[Operation],
    schedule: Optional[List[Dict[str, Any]]]=None,
    initial_placement: bool = False,
    sorted_traps: Optional[List] = None,
    is_ms_reconfig: bool = False,
    subgridsize: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Internal helper: take the first layout in layouts_after (round 0 layout of
    the slice solver) and:

      - interpret it as a new global arrangement of ions over the full
        wiseArch.n × (wiseArch.m*wiseArch.k) grid,
      - group ions per manipulation trap,
      - build a GlobalReconfigurations.physicalOperation that moves ions from
        oldArrangementArr to newArrangementArr,
      - run that operation and append it to allOps,
      - refresh the architecture graph,
      - return the updated oldArrangementArr to be used as the new baseline.

    This encapsulates the “apply one big subgrid-based reconfiguration step”
    that is duplicated in the original implementation.

    Parameters
    ----------
    sorted_traps : list of ManipulationTrap, optional
        Traps in row-major grid order (row 0 blocks 0..m-1, row 1, etc.).
        When ``None``, the list is built internally by sorting
        ``arch._manipulationTraps`` by ``(pos[1], pos[0])``.
    """
    if sorted_traps is None:
        sorted_traps = sorted(
            arch._manipulationTraps,
            key=lambda t: (t.pos[1], t.pos[0]),
        )
    newArrangement: Dict[ManipulationTrap, List[Ion]] = {
        trap: [] for trap in arch._manipulationTraps
    }

    for d in range(wiseArch.n):
        for c in range(wiseArch.m * wiseArch.k):
            ionidx = int(layout_after[d][c])
            newArrangementArr[d][c] = ionidx

    # ── Defensive ion-set reconciliation ─────────────────────────
    # Merged routing steps (from block-level SAT + merge) can
    # occasionally produce a layout_after whose ion set diverges
    # from oldArrangementArr due to SAT solver non-determinism,
    # block-boundary edge cases, or return-reconfig convergence
    # failures.  physicalOperation's Phase B (phaseB_greedy_layout)
    # requires A and T to contain exactly the same ion IDs.
    #
    # Fix: detect mismatches before calling physicalOperation and
    # reconcile by placing missing ions into empty cells and
    # removing unexpected ions.
    old_ions = set(oldArrangementArr.flatten()) - {0}
    new_ions = set(newArrangementArr.flatten()) - {0}

    # Also detect DUPLICATE ions in newArrangementArr: set() hides
    # them, but physicalOperation requires exactly one copy of each.
    _flat_new = newArrangementArr.flatten()
    from collections import Counter as _Counter
    _ion_counts = _Counter(int(v) for v in _flat_new if int(v) != 0)
    _duplicates = {ion for ion, cnt in _ion_counts.items() if cnt > 1}

    _reconciliation_needed = False
    if old_ions != new_ions or _duplicates:
        _reconciliation_needed = True
        missing = old_ions - new_ions   # in old, not in new target
        extra   = new_ions - old_ions   # in new target, not in old

        logger.warning(
            "%s _apply_layout_as_reconfiguration: ion set mismatch — "
            "missing=%s, extra=%s, duplicates=%s. Reconciling arrays.",
            PATCH_LOG_PREFIX, sorted(missing), sorted(extra),
            sorted(_duplicates),
        )

        # 1a) Remove extra ions from newArrangementArr (set to 0)
        for d in range(wiseArch.n):
            for c in range(wiseArch.m * wiseArch.k):
                if int(newArrangementArr[d, c]) in extra:
                    newArrangementArr[d, c] = 0

        # 1b) De-duplicate: keep ONE copy of each duplicated ion
        #     (prefer the copy closest to the ion's old position)
        #     and zero out the rest.
        for dup_ion in sorted(_duplicates):
            dup_positions = []
            for d in range(wiseArch.n):
                for c in range(wiseArch.m * wiseArch.k):
                    if int(newArrangementArr[d, c]) == dup_ion:
                        dup_positions.append((d, c))
            if len(dup_positions) <= 1:
                continue
            # Find ideal keep position: closest to old position
            _old_pos = np.argwhere(oldArrangementArr == dup_ion)
            if len(_old_pos) > 0:
                od, oc = int(_old_pos[0][0]), int(_old_pos[0][1])
                dup_positions.sort(
                    key=lambda rc: abs(rc[0] - od) + abs(rc[1] - oc)
                )
            # Keep first, zero out the rest
            for dd, dc in dup_positions[1:]:
                newArrangementArr[dd, dc] = 0

        # 2) Place missing ions: prefer their old position if the
        #    cell is now empty; otherwise find the nearest empty cell.
        _empty_cells = []
        for d in range(wiseArch.n):
            for c in range(wiseArch.m * wiseArch.k):
                if int(newArrangementArr[d, c]) == 0:
                    _empty_cells.append((d, c))

        for ion in sorted(missing):
            # Find where the ion was in the old arrangement
            _old_pos = np.argwhere(oldArrangementArr == ion)
            placed = False
            if len(_old_pos) > 0:
                od, oc = int(_old_pos[0][0]), int(_old_pos[0][1])
                if int(newArrangementArr[od, oc]) == 0:
                    newArrangementArr[od, oc] = ion
                    _empty_cells = [
                        (d, c) for d, c in _empty_cells
                        if not (d == od and c == oc)
                    ]
                    placed = True
            if not placed and _empty_cells:
                ed, ec = _empty_cells.pop(0)
                newArrangementArr[ed, ec] = ion
                placed = True
            if not placed:
                logger.error(
                    "%s could not place missing ion %d — no empty cells",
                    PATCH_LOG_PREFIX, ion,
                )

    # ── If reconciliation changed the target, drop the SAT schedule ──
    # The existing schedule was computed to transform 
    # oldArrangementArr → original layout_after.  Reconciliation
    # changed the target to the modified newArrangementArr, so the
    # schedule can no longer reach it.  Dropping it (schedule=None)
    # forces Phase B/C/D heuristic for just this step, which
    # prevents a cascading layout mismatch for all subsequent steps.
    if _reconciliation_needed and schedule is not None:
        if is_ms_reconfig:
            # MS-gate context: heuristic fallback is NEVER allowed.
            # If ion reconciliation changed the target, it means the
            # SAT schedule can no longer reach the correct layout.
            # This is a fundamental bug in patch-and-route that must
            # be fixed — we refuse to silently fall back to the
            # heuristic (odd-even transposition sort) for MS gates.
            _n_changed = int(np.sum(oldArrangementArr != newArrangementArr))
            raise ValueError(
                f"[WISE Routing] Ion reconciliation changed the target layout "
                f"during MS-gate reconfiguration.  The SAT-solved schedule "
                f"can no longer reach the target.  Heuristic fallback is NOT "
                f"allowed for MS-gate reconfigs — this indicates a fundamental "
                f"problem in patch-and-route that needs fixing.  "
                f"Grid shape: {oldArrangementArr.shape}, "
                f"reconciliation modified {_n_changed} cell(s)."
            )
        else:
            logger.info(
                "%s ion reconciliation changed target layout; dropping SAT "
                "schedule to force heuristic (Phase B/C/D) for this step.",
                PATCH_LOG_PREFIX,
            )
            schedule = None

    # ── Build trap → ion mapping for physicalOperation ───────────
    for d in range(wiseArch.n):
        for c in range(wiseArch.m * wiseArch.k):
            ionidx = int(newArrangementArr[d][c])
            old_ionidx = int(oldArrangementArr[d][c])

            # Determine which physical trap owns grid cell (d, c).
            # Primary: look up the ion that was in this cell before.
            # Fallback: derive from grid geometry when the cell was empty.
            if old_ionidx != 0:
                trap = arch.ions[old_ionidx].parent
            else:
                # Empty cell (index 0 is not a real ion) -- map grid
                # position to the physical trap via row-major ordering.
                trap_col = c // wiseArch.k
                trap_linear = d * wiseArch.m + trap_col
                trap = sorted_traps[trap_linear]

            # Only record real ions in the new arrangement dict.
            if ionidx != 0:
                newArrangement[trap].append(arch.ions[ionidx])

    reconfig = GlobalReconfigurations.physicalOperation(
        newArrangement, wiseArch, oldArrangementArr, newArrangementArr,
        schedule=schedule, initial_placement=initial_placement,
        context="ms_gate" if is_ms_reconfig else "route_back",
    )
    allOps.append(reconfig)
    reconfig.run()
    arch.refreshGraph()

    return newArrangementArr.copy()


# =====================================================================
# Reconfig Merge Optimisation (Work Stream B)
# =====================================================================

def _find_ion_position(
    layout: np.ndarray,
    ion_idx: int,
) -> Optional[Tuple[int, int]]:
    """Find row, col of *ion_idx* in a layout array.

    Returns ``None`` if the ion is not present.
    """
    positions = np.argwhere(layout == ion_idx)
    if len(positions) == 0:
        return None
    return (int(positions[0][0]), int(positions[0][1]))


def _ions_unmoved(
    ion_set: Set[int],
    current_layout: np.ndarray,
    next_layout: np.ndarray,
) -> bool:
    """Check whether every ion in *ion_set* is in the same position
    in *current_layout* and *next_layout*.

    An ion that appears in one layout but not the other counts as moved.
    """
    for ion_idx in ion_set:
        cur = _find_ion_position(current_layout, ion_idx)
        nxt = _find_ion_position(next_layout, ion_idx)
        if cur is None or nxt is None or cur != nxt:
            return False
    return True


# =====================================================================
# Routing Engine Extraction — RoutingStep + _route_round_sequence
# (Spec §2.3: pure-array routing with all 10 tricks, no arch coupling)
# =====================================================================

@dataclass
class RoutingStep:
    """One reconfiguration + MS-gate step produced by the routing engine.

    This is the unit of work produced by ``_route_round_sequence`` and
    consumed by the execution layer in ``ionRoutingWISEArch`` or
    ``ionRoutingGadgetArch``.
    """

    layout_after: np.ndarray
    """Target ion arrangement after reconfiguration."""

    schedule: Optional[List[Dict[str, Any]]]
    """Swap passes to achieve *layout_after*.  ``None`` when replayed
    from cache (first step only) — the execution bridge will use
    heuristic reconfig."""

    solved_pairs: List[Tuple[int, int]]
    """MS pairs enabled by this layout (for this MS round)."""

    ms_round_index: int
    """Which ``parallelPairs`` round these pairs solve."""

    from_cache: bool
    """True if this step was replayed from the block cache."""

    tiling_meta: Tuple[int, int]
    """``(cycle_idx, tiling_idx)`` for debugging."""

    can_merge_with_next: bool
    """Hint for reconfig merge optimisation (Trick 5)."""

    is_initial_placement: bool = False
    """True for the very first reconfiguration (idx==0)."""

    is_layout_transition: bool = False
    """True for SAT-based layout transition steps (BT-driven, no MS pairs).
    These must always be applied by the execution loop regardless of
    the solved_pairs guard."""

    layout_before: Optional[np.ndarray] = None
    """Starting layout this step's schedule was built for.
    Used by the execution loop to detect planning-vs-execution layout
    divergence and rebuild the schedule when needed."""

    reconfig_context: str = "ms_gate"
    """Classification of what this reconfiguration achieves.

    * ``"ms_gate"`` — reconfiguration to place ions for an MS gate.
      **Must** always use SAT-based routing; heuristic fallback is
      forbidden.  If SAT fails after escalation, a ``ValueError`` is
      raised rather than silently degrading.
    * ``"cache_replay"`` — transition back to a cached starting layout
      so that pre-computed steps can be replayed.  SAT is attempted
      first; heuristic (Phase B/C/D) is acceptable as a fallback.
    * ``"return_round"`` — route-back to the round-start layout within
      an EC block.  Heuristic is acceptable."""


def _route_round_sequence(
    oldArrangementArr: np.ndarray,
    wiseArch: "QCCDWiseArch",
    parallelPairs: List[List[Tuple[int, int]]],
    *,
    lookahead: int,
    subgridsize: Tuple[int, int, int],
    base_pmax_in: int = 1,
    active_ions: List[int],
    toMoveOps: Optional[List] = None,
    stop_event: Optional[Any] = None,
    progress_callback: Optional[Callable] = None,
    max_inner_workers: Optional[int] = None,
    initial_BTs: Optional[List] = None,
    replay_level: int = 1,
) -> Tuple[List[RoutingStep], np.ndarray]:
    """Pure-array routing engine extracted from ``ionRoutingWISEArch``.

    Contains ALL 10 tricks from the working code:

    1. Gating-capacity overflow spilling
    2. Cross-boundary preference soft clauses
    3. BT pin conflict detection
    4. Multi-round lookahead BT pins
    5. Reconfig merge optimisation
    6. Block caching with per-offset replay
    7. ``_rebuild_schedule_for_layout``
    8. ``prev_pmax`` warm-starting
    9. ``NoFeasibleLayoutError`` graceful recovery
    10. Single-qubit scheduling metadata (epoch markers)

    No ``arch`` or ``Operation`` references — only numpy arrays, SAT calls,
    and scheduling logic.  Variable names preserved from the original main
    loop for readability and diff-friendliness.

    Parameters
    ----------
    oldArrangementArr : np.ndarray
        Initial ion arrangement on the WISE grid.
    wiseArch : QCCDWiseArch
        Grid geometry (``n`` rows, ``m`` blocks, ``k`` capacity).
    parallelPairs : list of list of (int, int)
        Per-round MS ion-index pairs.
    lookahead : int
        Number of future rounds visible to the SAT solver.
    subgridsize : (int, int, int)
        ``(cols, rows, increment)`` for the patch slicer.
    base_pmax_in : int
        Base ``pmax`` seed for the SAT solver.
    active_ions : list of int
        Indices of non-spectator ions.
    toMoveOps : list, optional
        Used for block-cache key building (original variable name).
    stop_event : threading.Event, optional
        Cooperative cancellation.
    progress_callback : callable, optional
        Progress reporting.
    max_inner_workers : int, optional
        SAT parallelism cap.

    Returns
    -------
    steps : List[RoutingStep]
        Ordered routing steps to be applied by the execution layer.
    final_layout : np.ndarray
        The arrangement array after all routing steps.
    """
    steps: List[RoutingStep] = []
    total_ms_rounds = len(parallelPairs)

    if total_ms_rounds == 0:
        return steps, oldArrangementArr.copy()

    # --- Variable names preserved from ionRoutingWISEArch main loop ---
    idx = 0
    tiling_steps_meta: List[Tuple[int, int, List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]]]] = []
    tiling_steps: List[List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]]] = []
    tiling_step: List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]] = []
    hadMultipleTilingSteps: bool = False
    BTs: Optional[List[Dict[Tuple[int, int], Tuple[Dict[int, Tuple[int, int]], List[Tuple[int, int]]]]]] = initial_BTs

    # --- Fix 3: Preserve return-round BT across routing windows ---
    # The last entry in initial_BTs carries ion-return pins that must
    # survive BT rebuilds for subsequent windows.  Extract it once and
    # re-inject it after every rebuild when the window covers the last
    # (return) round.
    _sticky_return_bt: Optional[Dict] = None
    if initial_BTs is not None and len(initial_BTs) > 0:
        _last_bt = initial_BTs[-1]
        if _last_bt:
            _sticky_return_bt = deepcopy(_last_bt)

    # --- Trick 6: Block cache infrastructure ---
    block_cache: Dict[
        Tuple[Tuple[Tuple[int, int], ...], ...],
        Dict[int, List[Tuple[int, int, List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]]]]]
    ] = {}
    tiling_steps_from_cache: bool = False
    pending_first_cached_reconfig: bool = False
    blk_idx: int = 0
    blk_end: int = 0
    recheck_cache: bool = True
    # When replay_level == 0 we disable block caching entirely so
    # every round is solved fresh with SAT (no replayed schedules).
    BLOCK_LEN = 4 if replay_level > 0 else float('inf')
    cache_for_block: Dict = {}

    # Track current layout (updated after each routing step)
    currentArr = oldArrangementArr.copy()

    while idx < total_ms_rounds:
        # Emit STAGE_ROUTING so the progress widget can track MS-round
        # advancement while the SAT solver is still working.
        if progress_callback is not None:
            progress_callback(RoutingProgress(
                stage=STAGE_ROUTING,
                current=idx,
                total=total_ms_rounds,
                message=f"Processing MS round {idx}/{total_ms_rounds}",
            ))

        logger.info(
            "%s _route_round_sequence: idx=%d/%d",
            PATCH_LOG_PREFIX, idx, total_ms_rounds,
        )

        if (not hadMultipleTilingSteps) and tiling_steps and tiling_steps[0]:
            # Reuse pre-computed tiling plan from previous iteration
            pass
        else:
            # 4a) Compute routing window
            window_start = idx
            window_end = min(len(parallelPairs), lookahead + idx)
            P_arr = parallelPairs[window_start:window_end].copy()

            if recheck_cache:
                # --- Trick 6: Block cache lookup ---
                new_blk_end = min(len(parallelPairs), window_start + BLOCK_LEN) - window_start
                new_block_key_rounds = parallelPairs[window_start:new_blk_end + window_start]
                new_block_key: Tuple[Tuple[Tuple[int, int], ...], ...] = tuple(
                    tuple(sorted(round_pairs)) for round_pairs in new_block_key_rounds
                )

                new_cache_for_block = block_cache.setdefault(new_block_key, {})
                logger.info(
                    "%s rechecked cache at idx=%d; blk_idx=%d, blk_end=%d, new_blk_end=%d, "
                    "cache_for_block=%d, new_cache=%d",
                    PATCH_LOG_PREFIX, idx, blk_idx, blk_end, new_blk_end,
                    len(cache_for_block), len(new_cache_for_block),
                )
                if len(new_cache_for_block) > 0:
                    cache_for_block = new_cache_for_block
                    blk_idx = 0
                    blk_end = new_blk_end
                    tiling_steps_from_cache = True
                elif len(cache_for_block) == 0:
                    cache_for_block = new_cache_for_block
                    blk_idx = 0
                    blk_end = new_blk_end
                    tiling_steps_from_cache = False
                else:
                    tiling_steps_from_cache = False

            if tiling_steps_from_cache:
                # --- Trick 6: Cache hit → replay ---
                tiling_steps_meta = deepcopy(cache_for_block[blk_idx])
                pending_first_cached_reconfig = (blk_idx == 0)
                logger.info(
                    "%s reusing cached block at idx=%d; blk_idx=%d, tilings=%d",
                    PATCH_LOG_PREFIX, idx, blk_idx, len(tiling_steps_meta),
                )
            else:
                # --- Tricks 1,2,3,8,9 inside _patch_and_route ---
                tiling_steps_meta = _patch_and_route(
                    currentArr,
                    wiseArch,
                    P_arr,
                    subgridsize,
                    active_ions=active_ions,
                    ignore_initial_reconfig=True,
                    base_pmax_in=base_pmax_in,
                    BTs=BTs,
                    stop_event=stop_event,
                    progress_callback=progress_callback,
                    max_inner_workers=max_inner_workers,
                )
                if len(cache_for_block) < BLOCK_LEN:
                    cache_for_block[blk_idx] = deepcopy(tiling_steps_meta)
                logger.info(
                    "%s patch routing idx=%d produced %d tiling steps; blk_idx=%d, cache_size=%d",
                    PATCH_LOG_PREFIX, idx, len(tiling_steps_meta),
                    blk_idx, len(cache_for_block),
                )

            # Strip metadata for execution ordering
            tiling_steps = [snapshot for (_cy, _ti, snapshot) in tiling_steps_meta]
            hadMultipleTilingSteps = len(tiling_steps) > 1

            # --- Trick 4: Build BTs from tiling_steps_meta for future windows ---
            if tiling_steps_meta:
                active_set = set(active_ions)
                R_local = len(P_arr)
                BTs = [dict() for _ in range(R_local - 1)]
                for cycle_idx_meta, tiling_idx_meta, tiling in tiling_steps_meta:
                    for r, (layout_after_r, _sched_r, _solved_pairs_r) in enumerate(tiling[1:]):
                        bt_map: Dict[int, Tuple[int, int]] = {}
                        layout_arr = layout_after_r
                        for rr in range(wiseArch.n):
                            for cc in range(wiseArch.m * wiseArch.k):
                                ionidx = int(layout_arr[rr][cc])
                                if ionidx in active_set:
                                    bt_map[ionidx] = (rr, cc)
                        key = (cycle_idx_meta, tiling_idx_meta)
                        BTs[r][key] = (bt_map, list(_solved_pairs_r))

                # --- Fix 3 (cont.): Re-inject sticky return-round BT ---
                # If the current window includes the return round (the
                # very last round in parallelPairs), merge the saved
                # return-round BT pins into the last BTs entry so that
                # ions are correctly pinned to their return positions.
                total_rounds = len(parallelPairs)
                if _sticky_return_bt is not None and BTs:
                    last_round_global = total_rounds - 1
                    last_round_in_window = window_end - 1
                    if last_round_in_window >= last_round_global:
                        for sbt_key, sbt_val in _sticky_return_bt.items():
                            if sbt_key not in BTs[-1]:
                                BTs[-1][sbt_key] = sbt_val
            else:
                BTs = None

        if not tiling_step:
            if not tiling_steps:
                # --- Fix B: Skip _patch_and_route for empty-pair windows ---
                # The trailing return round (appended by EC routing) has
                # no MS pairs.  Routing it through the SAT solver
                # yields 0 tiling steps, which triggers a false abort.
                # Instead, produce a no-op step that preserves the
                # current layout unchanged.
                if all(len(rp) == 0 for rp in P_arr):
                    logger.info(
                        "%s empty-pair window at idx=%d; emitting no-op step",
                        PATCH_LOG_PREFIX, idx,
                    )
                    tiling_steps = [[(currentArr.copy(), None, [])]]
                    tiling_step = [t.pop(0) for t in tiling_steps]
                    layout_after, schedule, solved_pairs = tiling_step.pop(0)
                    steps.append(RoutingStep(
                        layout_after=layout_after,
                        schedule=None,
                        solved_pairs=solved_pairs,
                        ms_round_index=idx,
                        from_cache=False,
                        tiling_meta=(0, 0),
                        can_merge_with_next=False,
                        is_initial_placement=False,
                        layout_before=currentArr.copy(),
                        reconfig_context="cache_replay",  # Fix A: no-op step
                    ))
                    currentArr = layout_after.copy()
                    idx += 1
                    blk_idx += 1
                    recheck_cache = (blk_idx not in cache_for_block)
                    if blk_idx == blk_end:
                        cache_for_block = {}
                        blk_idx = blk_end = 0
                    continue  # next iteration of while loop
                # _patch_and_route returned no results (all SAT configs
                # exhausted / UNSAT).  Retry with a *capped* window
                # instead of dumping ALL remaining rounds at once —
                # jumping from e.g. 1 → 31 rounds creates an
                # intractable SAT problem (784K+ clauses).
                remaining = parallelPairs[idx:]
                # Cap retry at 2×lookahead to prevent window explosion.
                # Previously used a fixed cap of 6 which could far exceed
                # the user-specified lookahead, creating intractable SAT
                # problems.  Now the retry window respects the lookahead
                # parameter while still allowing a modest expansion.
                _MAX_RETRY_WINDOW = max(2 * lookahead, 4)
                if len(remaining) > len(P_arr):
                    retry_window = min(
                        len(remaining),
                        _MAX_RETRY_WINDOW,
                    )
                    logger.warning(
                        "%s retrying with capped window (%d of %d remaining) "
                        "after empty result for window of %d",
                        PATCH_LOG_PREFIX, retry_window, len(remaining),
                        len(P_arr),
                    )
                    tiling_steps_meta = _patch_and_route(
                        currentArr,
                        wiseArch,
                        parallelPairs[idx:idx + retry_window],
                        subgridsize,
                        active_ions=active_ions,
                        ignore_initial_reconfig=True,
                        base_pmax_in=base_pmax_in,
                        BTs=BTs,
                        stop_event=stop_event,
                        progress_callback=progress_callback,
                        max_inner_workers=max_inner_workers,
                    )
                    tiling_steps = [snapshot for (_cy, _ti, snapshot) in tiling_steps_meta]
                if not tiling_steps:
                    logger.error(
                        "%s _patch_and_route returned empty results at idx=%d; "
                        "aborting routing loop",
                        PATCH_LOG_PREFIX, idx,
                    )
                    break
            tiling_step = [t.pop(0) for t in tiling_steps]

        layout_after, schedule, solved_pairs = tiling_step.pop(0)

        # Determine schedule argument for execution
        if tiling_steps_from_cache and pending_first_cached_reconfig:
            pending_first_cached_reconfig = False
            # Only drop schedule when layouts already match;
            # otherwise keep the cached schedule to avoid a
            # costly SAT rebuild in the execution loop.
            if np.array_equal(layout_after, currentArr):
                sched_arg = None
            else:
                sched_arg = schedule
        elif np.array_equal(layout_after, currentArr):
            # Identity reconfiguration — no schedule needed.  This also
            # guards against garbage schedules decoded from non-optimised
            # SAT rounds (ignore_initial_reconfig with R=1).
            sched_arg = None
        else:
            sched_arg = schedule

        # Build the RoutingStep
        _step_layout_before = currentArr.copy()
        steps.append(RoutingStep(
            layout_after=layout_after,
            schedule=sched_arg,
            solved_pairs=solved_pairs,
            ms_round_index=idx,
            from_cache=tiling_steps_from_cache,
            tiling_meta=(0, 0),  # simplified — full meta in tiling_steps_meta
            can_merge_with_next=False,  # will be set below in merge analysis
            is_initial_placement=False,
            layout_before=_step_layout_before,
            reconfig_context=(
                "cache_replay" if tiling_steps_from_cache else "ms_gate"
            ),
        ))

        # Update current layout
        currentArr = layout_after.copy()

        # --- Trick 5: Reconfig merge analysis ---
        # Check remaining tiling_step entries for merge opportunities
        # (This mirrors the inner while loop in section 4c of the original)
        # We'll check for merges in the MS-gate section below.

        # Now handle multiple tiling steps within the same MS round
        # (mirrors the inner while loop in section 4c)
        # NOTE: Reconfig merge (Trick 5) is DISABLED — naive schedule
        # concatenation is incorrect because the second schedule's swap
        # coordinates assume an intermediate layout that doesn't exist
        # when both are applied from the original state.  Always emit
        # separate steps.
        while tiling_step and any(t[2] for t in tiling_step):
            next_layout, next_schedule, next_solved = tiling_step[0]
            tiling_step.pop(0)
            steps.append(RoutingStep(
                layout_after=next_layout,
                schedule=next_schedule,
                solved_pairs=next_solved,
                ms_round_index=idx,
                from_cache=tiling_steps_from_cache,
                tiling_meta=(0, 0),
                can_merge_with_next=False,
                is_initial_placement=False,
                layout_before=currentArr.copy(),
                reconfig_context=(
                    "cache_replay" if tiling_steps_from_cache else "ms_gate"
                ),
            ))
            currentArr = next_layout.copy()

        # Advance cursors
        idx += 1
        blk_idx += 1
        recheck_cache = (blk_idx not in cache_for_block)
        if blk_idx == blk_end:
            cache_for_block = {}
            blk_idx = blk_end = 0

    return steps, currentArr


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  H5 + H6: Shared helpers for ionRoutingWISEArch / GadgetArch    ║
# ╚═══════════════════════════════════════════════════════════════════╝

def _build_parallel_pairs(
    operations: Sequence,
    toMoveOps: Optional[List[List]],
    parallelism: int,
    *,
    build_toMoves: bool = False,
) -> Tuple[List[List[Tuple[int, int]]], List[list]]:
    """Greedily partition 2-qubit MS gates into disjoint parallel rounds.

    Parameters
    ----------
    operations : sequence of ``QubitOperation``
    toMoveOps : pre-built MS gate buckets (from compiler); ``None`` triggers
        greedy construction from *operations*.
    parallelism : max pairs per round (typically ``wiseArch.m * wiseArch.n``).
    build_toMoves : when *True* and *toMoveOps is None*, also build
        ``toMoves`` (needed by ``ionRoutingWISEArch``).

    Returns
    -------
    parallelPairs : list of rounds, each round is a list of
        ``(ancilla_idx, data_idx)`` tuples.
    toMoves : list of rounds of ``TwoQubitMSGate`` ops (empty lists when
        *build_toMoves* is False and *toMoveOps* is None).
    """
    parallelPairs: List[List[Tuple[int, int]]] = []
    toMoves: List[list] = []

    if toMoveOps is None:
        _ops = list(operations)
        idx = 0
        while _ops:
            # Strip disjoint 1-qubit ops (they don't need routing)
            while True:
                toRemove: list = []
                ionsInvolved: Set[int] = set()
                for op in _ops:
                    if ionsInvolved.isdisjoint(op.ions) and len(op.ions) == 1:
                        toRemove.append(op)
                    ionsInvolved = ionsInvolved.union(op.ions)
                for g in toRemove:
                    _ops.remove(g)
                if not toRemove:
                    break

            # Form one round of disjoint 2-qubit MS gates
            toRemove = []
            ionsAdded: Set[int] = set()
            for op in _ops:
                if len(op.ions) == 2 and len(toRemove) < parallelism:
                    ion1, ion2 = op.ions
                    ancilla, data = sorted(
                        (ion1, ion2), key=lambda ion: ion.label[0] == "D"
                    )
                    if ancilla.idx in ionsAdded or data.idx in ionsAdded:
                        continue
                    toRemove.append(op)
                    if idx == len(parallelPairs):
                        parallelPairs.append([])
                    parallelPairs[idx].append((ancilla.idx, data.idx))
                    if build_toMoves:
                        if idx == len(toMoves):
                            toMoves.append([])
                        toMoves[idx].append(op)
                    ionsAdded.add(ancilla.idx)
                    ionsAdded.add(data.idx)
                else:
                    ionsAdded.add(op.ions[0].idx)
            for g in toRemove:
                _ops.remove(g)
            idx += int(len(toRemove) > 0)
    else:
        for toMove in toMoveOps:
            if build_toMoves:
                toMoves.append(list(toMove))
            parallelPairs.append([])
            for op in toMove:
                ion1, ion2 = op.ions
                ancilla, data = sorted(
                    (ion1, ion2), key=lambda ion: ion.label[0] == "D"
                )
                parallelPairs[-1].append((ancilla.idx, data.idx))

    return parallelPairs, toMoves


def _build_grid_state(
    arch,
    wiseArch,
) -> Tuple[np.ndarray, list, list]:
    """Build initial grid layout and sorted ion list.

    Returns
    -------
    oldArrangementArr : 2-D int array (n × m*k) of ion indices.
    ionsSorted : flat list of all ions in row-major trap order.
    active_ions : list of non-spectator ion indices.
    """
    oldArrangementArr = np.zeros(
        (wiseArch.n, wiseArch.m * wiseArch.k), dtype=int,
    )
    traps_sorted = sorted(
        arch._manipulationTraps,
        key=lambda t: (t.pos[1], t.pos[0]),
    )
    ionsSorted: list = []
    for trap_idx, trap in enumerate(traps_sorted):
        row = trap_idx // wiseArch.m
        block = trap_idx % wiseArch.m
        for ion_offset, ion in enumerate(trap.ions):
            col = block * wiseArch.k + ion_offset
            oldArrangementArr[row][col] = ion.idx
            ionsSorted.append(ion)
    active_ions = [
        ion.idx for ion in ionsSorted
        if not isinstance(ion, SpectatorIon)
    ]
    return oldArrangementArr, ionsSorted, active_ions


def ionRoutingWISEArch(
    arch: QCCDArch,
    wiseArch: QCCDWiseArch,
    operations: Sequence[QubitOperation],
    lookahead: int = 2,
    subgridsize: Tuple[int, int, int] = (6, 4, 1), 
    base_pmax_in: int = None,
    toMoveOps: Sequence[Sequence[TwoQubitMSGate]]= None,
    routing_config: Optional[WISERoutingConfig] = None,
    stop_event: Optional[Any] = None,
    max_inner_workers: int | None = None,
    _precomputed_routing_steps: Optional[List["RoutingStep"]] = None,
    skip_coalesce: bool = False,
) -> Tuple[Sequence[Operation], Sequence[int], float]:
    """
    Route a WISE-style QCCD architecture in **three optimisation levels**:

      Level 1 – Patch-based domain decomposition
      ------------------------------------------
      - Given a global ion layout and a sequence of two-qubit rounds, we:
          1. Encode the grid as an n×(m·k) ion index array oldArrangementArr.
          2. Partition the two-qubit MS gates into maximally parallel rounds
             (parallelPairs, toMoves).
          3. For each window of rounds P_arr (of length ≤ lookahead):
                * Tile the device into disjoint patches of size
                  subgridsize[1] rows × subgridsize[0] columns.
                * Run two checkerboard tilings (aligned and offset) so any
                  local pair lies fully inside at least one patch.
                * For each patch, extract the subset of ion pairs whose ions
                  both reside inside the region and call the per-patch
                  optimiser (Levels 2 & 3) on that sub-grid only.
                * Merge the patch layouts/schedules back into the global
                  structures and drop the realised pairs. Remaining pairs from
                  the first tiling are solved in the offset tiling.
                * Once all tilings are processed (or no pairs remain), apply
                  the resulting layouts sequentially as global reconfigurations.

      Level 2 – Per-slice D-minimising SAT (inside _optimal_QMR_for_WISE)
      --------------------------------------------------------------------
      - For a given slice (currentGrid) and a small number of rounds P_arr[r],
        _optimal_QMR_for_WISE first:
          * builds a purely hard CNF encoding:
              - exact-one-per-cell layouts a[r],
              - per-ion row/column targets t[r], x[r],
              - block membership w[r],
              - pair constraints (same row & block),
              - BT pins for reserved ions,
              - layout consistency a[r+1] <-> (x[r], t[r]),
              - row presence p and y[r,k,c,i] with reserved-aware semantics,
              - a movement bound: max horizontal/vertical displacement ≤ D.
          * performs a **binary search on D**, solving SAT instances until it
            finds the smallest D* for which the CNF is satisfiable.

        This yields a per-slice routing that respects BT and pairs while
        minimising the maximum displacement D* within that slice.

      Level 3 – Per-slice boundary-aware MaxSAT (also inside _optimal_QMR_for_WISE)
      ----------------------------------------------------------------------------
      - With D fixed to D*, the same structural CNF is rebuilt as a WCNF,
        and small soft clauses are added to discourage interacting ions from
        landing on the outermost row / column of the slice:
            ¬x[r, i, last_column],  ¬t[r, i, last_row]
      - A MaxSAT solver (RC2) is then used to minimise the weighted number
        of violated soft clauses, without breaking any of the hard constraints
        or increasing D beyond D*.

    The overall behaviour of ionRoutingWISEArch is thus:

      1. Partition the circuit’s two-qubit gates into parallel rounds (by ions).
      2. For each “chunk” of up to `lookahead` rounds:
           - Run the Level-1 slicer, which repeatedly calls the Level-2/3
             subgrid optimiser until the entire device is covered.
           - Promote the final subgrid layout to a global physical
             reconfiguration operation, append it to allOps, and update the
             global ion positions.
      3. Between these reconfigurations, execute all single-qubit operations
         that fit without further routing, and then the scheduled two-qubit
         MS gates for that chunk, inserting barriers to preserve the time
         structure for later analysis.

    Returns:
        allOps    : the full, time-ordered list of physical operations,
                    including reconfigurations, single-qubit gates, and MS gates.
        barriers  : indices in allOps that act as “barriers” between logical
                    layers / routing phases (useful for parallelisation analysis).
    """

    allOps: List[Operation] = []
    barriers: List[int] = []
    operationsLeft = list(operations)

    # ------------------------------------------------------------------
    # 1) Build parallel rounds of two-qubit MS gates (H5: shared helper)
    # ------------------------------------------------------------------
    parallelismAllowed = wiseArch.m * wiseArch.n
    parallelPairs, toMoves = _build_parallel_pairs(
        operations, toMoveOps, parallelismAllowed, build_toMoves=True,
    )

    # ------------------------------------------------------------------
    # 2) Encode initial ion positions into oldArrangementArr (H6: shared)
    # ------------------------------------------------------------------
    oldArrangementArr, ionsSorted, active_ions = _build_grid_state(
        arch, wiseArch,
    )
    newArrangementArr = np.zeros_like(oldArrangementArr)

    logger.info(
        "%s ionRoutingWISEArch: ops=%d, MS_rounds=%d, lookahead=%d, subgrid=(%d,%d,%d), active_ions=%d",
        PATCH_LOG_PREFIX,
        len(operations),
        len(parallelPairs),
        lookahead,
        subgridsize[0],
        subgridsize[1],
        subgridsize[2],
        len(active_ions),
    )

    # # ------------------------------------------------------------------
    # # 3) Initial global reconfiguration via Level-1/2/3 on the first chunk
    # # ------------------------------------------------------------------
    # P_arr = parallelPairs[: min(len(parallelPairs), lookahead)].copy()
    # layouts_after, schedule = _patch_and_route(
    #     oldArrangementArr, wiseArch, P_arr, subgridsize, active_ions=active_ions
    # )
    # oldArrangementArr = _apply_layout_as_reconfiguration(
    #     arch, wiseArch, oldArrangementArr, newArrangementArr, layouts_after, allOps, schedule
    # )

    # ------------------------------------------------------------------
    # 3) Route all MS rounds via the extracted routing engine
    # ------------------------------------------------------------------
    if _precomputed_routing_steps is not None:
        # Phase-aware routing was done externally (e.g., by
        # ionRoutingGadgetArch).  Use the precomputed steps directly.
        routing_steps = _precomputed_routing_steps

        # ----------------------------------------------------------
        # Fix 13: Force Path B execution for precomputed routing steps.
        #
        # When ionRoutingGadgetArch provides precomputed routing steps,
        # the ms_round_index values are in the Fix10-expanded space
        # (shared-ion splitting may expand N rounds → M > N rounds).
        # But _build_parallel_pairs above rebuilds toMoves in the
        # ORIGINAL (unexpanded) space, causing:
        #   - Index overflow: ms_round_index >= len(toMoves)
        #   - Content mismatch: toMoves[idx] contains wrong operations
        #     for rounds that were split by Fix10
        #
        # Setting toMoves=None forces _execute_ms_gates to always use
        # Path B (frozenset matching against operationsLeft), which
        # correctly matches operations by ion pair identity rather than
        # positional index.  This guarantees every MS gate for every
        # solved_pair is found and executed, regardless of how the
        # routing phase split or reordered rounds.
        # ----------------------------------------------------------
        toMoves = None
        logger.info(
            "%s Fix13: precomputed routing steps — toMoves nullified to "
            "force Path B execution (frozenset matching)",
            PATCH_LOG_PREFIX,
        )
    else:
        routing_steps, _final_layout = _route_round_sequence(
            oldArrangementArr,
            wiseArch,
            parallelPairs,
            lookahead=lookahead,
            subgridsize=subgridsize,
            base_pmax_in=base_pmax_in or 1,
            active_ions=active_ions,
            toMoveOps=toMoveOps,
            stop_event=stop_event,
            progress_callback=routing_config.progress_callback if routing_config else None,
            max_inner_workers=max_inner_workers,
        )

    reconfigTime = 0.0

    # ------------------------------------------------------------------
    # Progress tracking setup
    # ------------------------------------------------------------------
    if _precomputed_routing_steps is not None:
        # Use the max ms_round_index from routing steps for accurate
        # progress tracking (reflects Fix10 expansion).
        total_ms_rounds = max(
            (s.ms_round_index for s in routing_steps), default=0
        ) + 1
    else:
        total_ms_rounds = len(toMoves) if toMoves else len(parallelPairs)

    # When we are APPLYING precomputed routing steps, use STAGE_APPLYING
    # so the progress bar resets its HWM and shows execution progress
    # instead of staying stuck at the planning-phase "44/44".
    _progress_callback = (
        routing_config.progress_callback
        if routing_config is not None
        else None
    )
    _exec_stage = (
        STAGE_APPLYING
        if _precomputed_routing_steps is not None
        else STAGE_ROUTING
    )

    def _report_progress(current_round: int, stage: str | None = None) -> None:
        """Report routing progress if callback is configured."""
        if _progress_callback is not None:
            _st = stage if stage is not None else _exec_stage
            progress = RoutingProgress(
                stage=_st,
                current=current_round,
                total=total_ms_rounds,
                message=f"Applying step {current_round}/{total_ms_rounds}"
                if _st == STAGE_APPLYING
                else f"Processing MS round {current_round}/{total_ms_rounds}",
            )
            _progress_callback(progress)

    # ------------------------------------------------------------------
    # Nested helper: epoch-aware single-qubit drain (section 4b)
    # Extracted into a callable so it can run once per MS round AND
    # once after all MS rounds to drain remaining 1-qubit ops.
    # ------------------------------------------------------------------
    # INV-D1: Type-monotone drain order.  RESET first (prepare
    # qubits), then rotations grouped by type, MEAS last.  This
    # ensures the natural QEC phase ordering RESET→RX→RY→MEAS
    # within each drain window.
    _TYPE_ORDER = {
        QubitReset: 0,
        XRotation: 1,
        YRotation: 2,
        Measurement: 3,
    }

    def _drain_single_qubit_ops(
        round_idx: int,
        *,
        max_drainable_round: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Execute eligible single-qubit ops using epoch-aware drain.

        Correctness constraints (ported from committed ionRoutingWISEArch §4b):

        C1  Cross-epoch: ops before a TICK must finish before ops after
            that TICK.  Enforced by an *epoch ceiling*: we never pull
            single-qubit ops from epochs later than the earliest
            remaining multi-qubit (MS) gate.
        C2  Within-epoch decomposition order: if CX decomposes into
            RX → MS → RY on the same ion, RX must happen before MS
            and RY must wait until after MS.  Enforced by the
            *blocked-ions* scan: once we see a multi-qubit gate on an
            ion in operationsLeft, all later single-qubit ops on that
            ion are ineligible this round.

        Round-bounded drain (Fix 1):
            When *max_drainable_round* is set, only 1q ops whose
            ``_routing_round`` ≤ *max_drainable_round* are eligible.
            This prevents pre-MS rotations from future routing rounds
            (on non-conflicting ions) from draining early, which would
            scatter one CX instruction's ops across many segments.

        Type-aware sub-barriers:
            Within each epoch's eligible ops, we group by type using a
            shortest-gate-first drain.  Fast gates (rotations ~5µs) run
            first; slow gates (measurement ~400µs, reset ~50µs) are
            deferred, increasing the probability that measurements
            accumulate and get batched together.

        Returns (single_qubit_executed, group_count).
        """
        from collections import deque as _deque

        # -- C1: Global epoch ceiling --
        # Never pull single-qubit ops from epochs later than the earliest
        # remaining multi-qubit (MS) gate.  This prevents post-MS
        # rotations from executing before their MS gate.
        min_ms_epoch: int = 2**62  # sentinel: no MS gates remaining
        for op in operationsLeft:
            if len(op.ions) >= 2:
                ep = getattr(op, '_tick_epoch', 0)
                if ep < min_ms_epoch:
                    min_ms_epoch = ep

        # -- C2 + eligible list: interleaved scan --
        # Walk operationsLeft in decomposition order.  For each ion,
        # only mark it blocked AFTER we encounter its first 2q gate.
        # 1q ops that appear *before* the ion's first 2q gate in
        # operationsLeft order are eligible — these are pre-MS
        # rotations that must execute before the MS gate anyway.
        seen_2q_ions: Set[Ion] = set()
        eligible: List[Operation] = []
        for op in operationsLeft:
            if len(op.ions) >= 2:
                # Mark these ions as blocked for subsequent 1q ops
                seen_2q_ions.update(op.ions)
            elif len(op.ions) == 1:
                ep = getattr(op, '_tick_epoch', 0)
                ion = op.ions[0]
                # C1: epoch ceiling — skip ops beyond earliest MS epoch
                if ep > min_ms_epoch:
                    continue
                # C2: skip ops on ions already seen in a 2q gate
                if ion in seen_2q_ions:
                    continue
                # Fix 1: Round-bounded drain — skip ops whose CX
                # instruction belongs to a routing round beyond the
                # current round.  This keeps each CX instruction's
                # pre-MS ops in their own round's drain window.
                if max_drainable_round is not None:
                    op_round = getattr(op, '_routing_round', None)
                    if op_round is not None and op_round > max_drainable_round:
                        continue
                # Fix C: Relaxed getTrapForIons for single-qubit ops.
                # After reconfiguration the ion IS in a trap, but the
                # helper may return None due to stale parent refs.
                # Fall back to the ion's current .parent (trap) directly.
                trap = op.getTrapForIons()
                if not trap:
                    ion_parent = getattr(ion, 'parent', None)
                    if ion_parent is not None:
                        trap = ion_parent
                        logger.debug(
                            "%s drain: getTrapForIons() failed for 1q op %s on ion %d, "
                            "using ion.parent (trap %s) as fallback",
                            PATCH_LOG_PREFIX,
                            type(op).__name__,
                            ion.idx,
                            getattr(ion_parent, 'idx', '?'),
                        )
                if trap:
                    eligible.append(op)
                else:
                    logger.debug(
                        "%s drain: skipping 1q op %s on ion %d — no trap found",
                        PATCH_LOG_PREFIX,
                        type(op).__name__,
                        ion.idx,
                    )

        # Fix E: diagnostic logging for drain filtering
        if logger.isEnabledFor(logging.DEBUG):
            _total_1q = sum(1 for op in operationsLeft if len(op.ions) == 1)
            _total_2q = sum(1 for op in operationsLeft if len(op.ions) >= 2)
            logger.debug(
                "%s drain round=%d: operationsLeft has %d 1q ops, %d 2q ops; "
                "%d eligible after C1+C2+trap filters (min_ms_epoch=%d)",
                PATCH_LOG_PREFIX,
                round_idx,
                _total_1q,
                _total_2q,
                len(eligible),
                min_ms_epoch if min_ms_epoch < 2**62 else -1,
            )

        # -- Cross-epoch, shortest-gate-first drain --
        # Build unified per-ion queues across ALL eligible epochs,
        # maintaining epoch ordering within each ion's queue.  This
        # allows same-type ops on *disjoint* ions from different epochs
        # to be merged into a single parallel batch (e.g. resets from
        # epoch 0 on ions {A,B} and resets from epoch 1 on ions {C,D}
        # execute together).  Within each ion's queue, epoch ordering is
        # preserved so decomposition-order constraints (C2) still hold.
        epoch_buckets: Dict[int, List[Operation]] = defaultdict(list)
        for op in eligible:
            epoch_buckets[getattr(op, '_tick_epoch', 0)].append(op)

        single_qubit_executed = 0
        group_count = 0

        # Build per-ion queues across all epochs (epoch order preserved)
        ion_queues: Dict[Ion, _deque] = defaultdict(_deque)
        for epoch in sorted(epoch_buckets.keys()):
            for op in epoch_buckets[epoch]:
                ion_queues[op.ions[0]].append(op)

        # INV-D2: Single-type drain — each pass processes exactly
        # ONE type.  RX and RY are NOT coalesced.  This produces
        # type-monotone output that the downstream reorderer and
        # paralleliser can group into contiguous same-type batches.
        while any(q for q in ion_queues.values()):
            best_tk = None
            for q in ion_queues.values():
                if q:
                    tk = _TYPE_ORDER.get(type(q[0]), 99)
                    if best_tk is None or tk < best_tk:
                        best_tk = tk

            if best_tk is None:
                break

            # INV-D2: drain exactly one type at a time (no coalescing).
            drain_tks = {best_tk}

            # Drain all ions whose front matches the single drainable type
            grp_ops: List[Operation] = []
            for ion in list(ion_queues.keys()):
                q = ion_queues[ion]
                while q and _TYPE_ORDER.get(type(q[0]), 99) in drain_tks:
                    grp_ops.append(q.popleft())

            if not grp_ops:
                break

            # Greedy disjoint-ion execution within this type group
            grp_remaining = list(grp_ops)

            while grp_remaining:
                toRemove: List[Operation] = []
                ionsInvolved: Set[Ion] = set()
                for op in grp_remaining:
                    if ionsInvolved.isdisjoint(op.ions):
                        trap = op.getTrapForIons()
                        # Fix C: fallback to ion.parent for 1q ops
                        if not trap and len(op.ions) == 1:
                            ion_parent = getattr(op.ions[0], 'parent', None)
                            if ion_parent is not None:
                                trap = ion_parent
                        if trap:
                            op.setTrap(trap)
                            toRemove.append(op)
                    ionsInvolved = ionsInvolved.union(op.ions)

                if not toRemove:
                    break

                for op in toRemove:
                    op.run()
                    allOps.append(op)
                    operationsLeft.remove(op)
                    grp_remaining.remove(op)
                    single_qubit_executed += 1

                group_count += 1

            # Fix A revised: no per-type-group barrier.  The
            # downstream paralleliser enforces WISE single-type
            # batching via its type-selection heuristic, and the
            # happens-before DAG orders same-ion ops correctly.
            # Removing these barriers lets the paralleliser see
            # the full rotation+reset+measurement window and make
            # globally optimal type-batch decisions.
            pass

        if single_qubit_executed:
            logger.info(
                "%s round=%d: executed %d single-qubit ops in %d type-groups; "
                "remaining_operations=%d, total_ops=%d",
                PATCH_LOG_PREFIX,
                round_idx,
                single_qubit_executed,
                group_count,
                len(operationsLeft),
                len(allOps),
            )

        # No per-drain barrier needed — the caller inserts barriers
        # at transport boundaries (before/after reconfigs).  The
        # paralleliser's DAG handles ordering within segments.

        return single_qubit_executed, group_count

    # ------------------------------------------------------------------
    # Nested helper: execute MS gates for a round using solved_pairs
    # ------------------------------------------------------------------
    def _execute_ms_gates(
        round_idx: int,
        solved_pairs: List[Tuple[int, int]],
    ) -> Tuple[int, int]:
        """Execute two-qubit MS gates whose ions match *solved_pairs*.

        Uses a **two-path** execution strategy identical to the working
        baseline (commit 243a188):

        **Path A — toMoves available** (primary, backward-compatible):
          Iterates ``toMoves[round_idx]`` to get the specific Operation
          objects for this MS round.  Uses the old ``any(all(ion.idx in p
          for ion in op.ions) for p in solved_pairs)`` contains-check so
          that pair ordering and set-vs-tuple mismatches are tolerable.
          Includes ``op.ions[0].parent`` fallback when ``getTrapForIons()``
          returns ``None``.

        **Path B — no toMoves entry** (fallback for precomputed gadget
          phases where ``ms_round_index`` may not map to a ``toMoves``
          index):
          Scans all remaining two-qubit ops in ``operationsLeft``, sorted
          by epoch, matching via frozenset equality.

        Returns (ms_executed, unmatched_count).
        """
        ms_executed = 0
        if not solved_pairs:
            return ms_executed, 0

        # Convert solved_pairs to a set of frozensets for fast lookup
        # (used by both paths).
        solved_sets = [frozenset(p) for p in solved_pairs]

        # ── Path A: toMoves-based execution (old working approach) ──
        if toMoves and round_idx < len(toMoves) and toMoves[round_idx]:
            round_ops = toMoves[round_idx]
            for op in round_ops:
                if op not in operationsLeft:
                    continue
                # Old-style contains check: each ion.idx must appear
                # in at least one solved pair.
                if any(
                    all(ion.idx in p for ion in op.ions)
                    for p in solved_pairs
                ):
                    trap = op.getTrapForIons()
                    if trap is not None:
                        op.setTrap(trap)
                        op.run()
                        allOps.append(op)
                        operationsLeft.remove(op)
                        ms_executed += 1
                    else:
                        # Fallback: try op.ions[0].parent (old code L1870)
                        trap = op.ions[0].parent if op.ions else None
                        if trap is not None:
                            op.setTrap(trap)
                            try:
                                op.run()
                                allOps.append(op)
                                operationsLeft.remove(op)
                                ms_executed += 1
                            except (ValueError, RuntimeError) as _e:
                                logger.warning(
                                    "%s round=%d: fallback setTrap "
                                    "failed for op_ions=%s: %s",
                                    PATCH_LOG_PREFIX, round_idx,
                                    [ion.idx for ion in op.ions], _e,
                                )
                        else:
                            logger.warning(
                                "%s round=%d: MS gate ions NOT "
                                "co-located — no parent trap — "
                                "SKIPPING: op_ions=%s",
                                PATCH_LOG_PREFIX, round_idx,
                                [ion.idx for ion in op.ions],
                            )
            return ms_executed, 0

        # ── Path B: scan operationsLeft (for precomputed gadget steps
        #    where ms_round_index may not map to toMoves index) ──
        remaining = list(solved_sets)

        two_q_ops = [
            op for op in operationsLeft if len(op.ions) == 2
        ]
        two_q_ops.sort(key=lambda op: getattr(op, '_tick_epoch', 0))

        to_execute: list = []
        for op in two_q_ops:
            if not remaining:
                break
            ion_set = frozenset(ion.idx for ion in op.ions)
            for i, rp in enumerate(remaining):
                if ion_set == rp:
                    to_execute.append(op)
                    remaining.pop(i)
                    break

        if remaining:
            avail_ion_sets = [
                frozenset(ion.idx for ion in op.ions)
                for op in two_q_ops if op not in to_execute
            ]
            logger.error(
                "%s round=%d: %d/%d solved_pairs UNMATCHED in Path B. "
                "solved_pairs=%s, unmatched=%s, available=%s, "
                "ops_left=%d, two_q=%d",
                PATCH_LOG_PREFIX, round_idx,
                len(remaining), len(solved_pairs),
                list(solved_pairs)[:5],
                list(remaining)[:5],
                list(avail_ion_sets)[:5],
                len(operationsLeft), len(two_q_ops),
            )

        # Sort by stim-circuit origin for animation ordering.
        to_execute.sort(
            key=lambda op: getattr(op, '_stim_origin', float('inf'))
        )

        for op in to_execute:
            trap = op.getTrapForIons()
            if trap is not None:
                op.setTrap(trap)
                op.run()
                allOps.append(op)
                operationsLeft.remove(op)
                ms_executed += 1
            else:
                # Fallback: try op.ions[0].parent (restored from old code)
                trap = op.ions[0].parent if op.ions else None
                if trap is not None:
                    op.setTrap(trap)
                    try:
                        op.run()
                        allOps.append(op)
                        operationsLeft.remove(op)
                        ms_executed += 1
                    except (ValueError, RuntimeError) as _e:
                        logger.warning(
                            "%s round=%d: fallback setTrap failed "
                            "for op_ions=%s: %s",
                            PATCH_LOG_PREFIX, round_idx,
                            [ion.idx for ion in op.ions], _e,
                        )
                else:
                    ion_parents = [
                        (ion.idx, ion.parent.idx if ion.parent else None)
                        for ion in op.ions
                    ]
                    logger.warning(
                        "%s round=%d: MS gate ions NOT co-located — "
                        "SKIPPING: op_ions=%s, parents=%s",
                        PATCH_LOG_PREFIX, round_idx,
                        [ion.idx for ion in op.ions], ion_parents,
                    )

        unmatched_count = len(remaining) if remaining else 0
        return ms_executed, unmatched_count

    # ------------------------------------------------------------------
    # 4) Execute operations, applying routing steps from the engine
    # ------------------------------------------------------------------
    # Group steps by ms_round_index.  Typically one step per round,
    # but multi-tiling (aligned + offset checkerboard) can produce
    # multiple RoutingSteps for the same MS round.
    from itertools import groupby as _groupby

    # Fix 6: Defensive sort — groupby only groups *consecutive* elements
    # with the same key. Ensure routing_steps are ordered by ms_round_index
    # so that non-consecutive duplicates (e.g. from transition reconfigs)
    # are correctly grouped. Stable sort preserves order within same index.
    routing_steps = sorted(routing_steps, key=lambda s: s.ms_round_index)

    _total_unmatched_pairs = 0  # Cumulative count of unmatched MS pairs
    _last_round_had_ms = True    # INV-E2: track if MS ran since last reconfig

    # Extract heuristic_fallback_for_noncache from routing_config for
    # the execution loop's schedule-rebuild decisions.
    _heuristic_fallback_for_noncache = (
        routing_config.heuristic_fallback_for_noncache
        if routing_config else False
    )

    # ------------------------------------------------------------------
    # Fix 1: Tag every 1q op with _routing_round so the drain can
    # restrict itself to ops from rounds ≤ current.  This prevents
    # pre-MS rotations on non-conflicting ions from draining early
    # (the primary cause of cross-segment fragmentation).
    # ------------------------------------------------------------------
    # Step 1: ion-pair → earliest routing round from solved_pairs
    _pair_to_round: Dict[frozenset, int] = {}
    for _step in routing_steps:
        for _a_idx, _d_idx in _step.solved_pairs:
            _pk = frozenset((_a_idx, _d_idx))
            if _pk not in _pair_to_round:
                _pair_to_round[_pk] = _step.ms_round_index
            else:
                _pair_to_round[_pk] = min(
                    _pair_to_round[_pk], _step.ms_round_index
                )

    # Step 2: MS gates → routing round; collect _stim_origin → round
    _origin_to_round: Dict[int, int] = {}
    for _op in operationsLeft:
        if len(_op.ions) >= 2:
            _pk = frozenset(ion.idx for ion in _op.ions)
            _rd = _pair_to_round.get(_pk)
            if _rd is not None:
                _op._routing_round = _rd  # type: ignore[attr-defined]
                _ori = getattr(_op, '_stim_origin', -1)
                if _ori >= 0:
                    _prev = _origin_to_round.get(_ori)
                    _origin_to_round[_ori] = (
                        min(_prev, _rd) if _prev is not None else _rd
                    )

    # Step 3: Tag 1q ops with the routing round of their CX instruction
    for _op in operationsLeft:
        if len(_op.ions) == 1:
            _ori = getattr(_op, '_stim_origin', -1)
            _rd = _origin_to_round.get(_ori)
            if _rd is not None:
                _op._routing_round = _rd  # type: ignore[attr-defined]

    logger.info(
        "%s Fix1: tagged %d origins → routing rounds; %d operationsLeft ops tagged",
        PATCH_LOG_PREFIX,
        len(_origin_to_round),
        sum(1 for _op in operationsLeft if hasattr(_op, '_routing_round')),
    )

    for ms_round_idx, round_steps_iter in _groupby(
        routing_steps, key=lambda s: s.ms_round_index
    ):
        round_steps = list(round_steps_iter)
        _report_progress(ms_round_idx)

        logger.info(
            "%s execution: ms_round=%d/%d, routing_steps=%d, remaining_ops=%d",
            PATCH_LOG_PREFIX,
            ms_round_idx,
            total_ms_rounds,
            len(round_steps),
            len(operationsLeft),
        )

        # 4a) Apply reconfiguration from the first routing step
        first_step = round_steps[0]

        # ── Pre-flight schedule verification (Fix 14) ─────────────
        # Replay the schedule on the ACTUAL starting layout.  If the
        # result doesn't match the planned layout_after, rebuild the
        # schedule via SAT from the actual starting layout.  This
        # handles all sources of planning-vs-execution layout
        # divergence (per-block merge issues, ion reconciliation
        # cascade, cache replay on drifted layout).
        _step_schedule = first_step.schedule
        _step_target = first_step.layout_after
        _is_ms_context = False  # default; set inside rebuild block
        if not np.array_equal(oldArrangementArr, _step_target):
            # Decide whether we need to rebuild the schedule.
            # Rebuild is needed when schedule is None (dropped by
            # _merge_block_routing_steps Fix 15) or when replay
            # doesn't produce the target layout.
            _needs_rebuild = _step_schedule is None
            if not _needs_rebuild:
                _replay = _simulate_schedule_replay(
                    oldArrangementArr, _step_schedule,
                )
                _needs_rebuild = not np.array_equal(
                    _replay, _step_target,
                )
            if _needs_rebuild:
                if _step_schedule is not None:
                    _pf_diff = int(np.sum(_replay != _step_target))
                else:
                    _pf_diff = int(
                        np.sum(oldArrangementArr != _step_target)
                    )
                logger.warning(
                    "%s ms_round=%d: PRE-FLIGHT FAIL — schedule %s "
                    "(%d/%d cells to target).  Rebuilding "
                    "schedule from actual starting layout.",
                    PATCH_LOG_PREFIX, ms_round_idx,
                    "replay mismatch" if _step_schedule is not None
                    else "is None (dropped by merge verification)",
                    _pf_diff,
                    oldArrangementArr.size,
                )
                # Also log layout_before from planning for diagnostics
                if first_step.layout_before is not None:
                    _lb_diff = int(np.sum(
                        oldArrangementArr != first_step.layout_before
                    ))
                    if _lb_diff:
                        logger.warning(
                            "%s ms_round=%d: planning layout_before "
                            "differs from execution oldArrangementArr "
                            "by %d cells",
                            PATCH_LOG_PREFIX, ms_round_idx, _lb_diff,
                        )
                # Rebuild via SAT from actual starting layout.
                # For MS-gate reconfigs, heuristic is FORBIDDEN (P1).
                # For cache_replay / return_round, heuristic is OK (P2).
                _is_ms_context = (
                    first_step.reconfig_context == "ms_gate"
                )
                _allow_heuristic = (
                    not _is_ms_context
                    or _heuristic_fallback_for_noncache
                )
                try:
                    _rb_snaps = _rebuild_schedule_for_layout(
                        oldArrangementArr, wiseArch, _step_target,
                        subgridsize=subgridsize,
                        base_pmax_in=base_pmax_in or 1,
                        stop_event=stop_event,
                        progress_callback=_progress_callback,
                        max_inner_workers=max_inner_workers,
                        allow_heuristic_fallback=_allow_heuristic,
                    )
                except Exception as _rb_exc:
                    logger.error(
                        "%s ms_round=%d: schedule rebuild failed: %s",
                        PATCH_LOG_PREFIX, ms_round_idx, _rb_exc,
                    )
                    _rb_snaps = []
                if _rb_snaps:
                    # Apply intermediate rebuild cycles as separate
                    # reconfigurations; the last cycle replaces the
                    # current step's schedule.
                    for _rb_layout, _rb_sched, _ in _rb_snaps[:-1]:
                        oldArrangementArr = (
                            _apply_layout_as_reconfiguration(
                                arch, wiseArch, oldArrangementArr,
                                newArrangementArr, _rb_layout, allOps,
                                _rb_sched,
                                initial_placement=False,
                                is_ms_reconfig=False,
                                subgridsize=subgridsize,
                            )
                        )
                        barriers.append(len(allOps))
                        reconfigTime += getattr(
                            allOps[-1], "_reconfigTime", 0.0,
                        )
                    _last_layout, _last_sched, _ = _rb_snaps[-1]
                    _step_schedule = _last_sched
                    _step_target = _last_layout
                    # Re-verify: after applying intermediate rebuild
                    # cycles, the actual oldArrangementArr may have
                    # diverged.  Confirm the final schedule still
                    # transforms the actual starting layout to
                    # _step_target.
                    if _step_schedule is not None:
                        _rb_replay = _simulate_schedule_replay(
                            oldArrangementArr, _step_schedule,
                        )
                        if not np.array_equal(
                            _rb_replay, _step_target,
                        ):
                            _rb_diff2 = int(
                                np.sum(_rb_replay != _step_target)
                            )
                            logger.warning(
                                "%s ms_round=%d: POST-REBUILD "
                                "VERIFY FAIL — rebuilt schedule "
                                "replay differs from target by "
                                "%d/%d cells; dropping schedule",
                                PATCH_LOG_PREFIX, ms_round_idx,
                                _rb_diff2,
                                oldArrangementArr.size,
                            )
                            _step_schedule = None
                    logger.info(
                        "%s ms_round=%d: schedule rebuilt via %d SAT "
                        "cycle(s)",
                        PATCH_LOG_PREFIX, ms_round_idx,
                        len(_rb_snaps),
                    )
                else:
                    logger.error(
                        "%s ms_round=%d: schedule rebuild returned 0 "
                        "snapshots; dropping broken schedule so "
                        "escalation/heuristic can take over",
                        PATCH_LOG_PREFIX, ms_round_idx,
                    )
                    # The original schedule was already proven bad by
                    # the pre-flight replay check.  Keeping it would
                    # bypass the SAT escalation guard (which checks
                    # _step_schedule is None) and pass a broken
                    # schedule to _runOddEvenReconfig, triggering a
                    # ValueError for ms_gate context.
                    _step_schedule = None

            # ── SAT-only guard for MS-gate reconfig (P1) ──────────
            # If after rebuild attempts the schedule is still None AND
            # this is an MS-gate reconfig (not cache_replay or
            # return_round), escalate with increasing pmax.  MS-gate
            # reconfigs must NEVER fall back to heuristic.
            if (
                _is_ms_context
                and _step_schedule is None
                and not np.array_equal(oldArrangementArr, _step_target)
            ):
                _escalated = _escalate_sat_for_ms_reconfig(
                    oldArrangementArr, wiseArch, _step_target,
                    subgridsize=subgridsize,
                    base_pmax_in=base_pmax_in,
                    stop_event=stop_event,
                    progress_callback=_progress_callback,
                    max_inner_workers=max_inner_workers,
                )
                if _escalated:
                    # Apply intermediate cycles, keep last as schedule
                    for _esc_layout, _esc_sched, _ in _escalated[:-1]:
                        oldArrangementArr = (
                            _apply_layout_as_reconfiguration(
                                arch, wiseArch, oldArrangementArr,
                                newArrangementArr, _esc_layout, allOps,
                                _esc_sched,
                                initial_placement=False,
                                is_ms_reconfig=False,
                                subgridsize=subgridsize,
                            )
                        )
                        barriers.append(len(allOps))
                        reconfigTime += getattr(
                            allOps[-1], "_reconfigTime", 0.0,
                        )
                    _elast, _esched, _ = _escalated[-1]
                    _step_schedule = _esched
                    _step_target = _elast
                    # Re-verify after escalation intermediates
                    if _step_schedule is not None:
                        _esc_replay = _simulate_schedule_replay(
                            oldArrangementArr, _step_schedule,
                        )
                        if not np.array_equal(
                            _esc_replay, _step_target,
                        ):
                            _esc_diff2 = int(
                                np.sum(_esc_replay != _step_target)
                            )
                            logger.warning(
                                "%s ms_round=%d: POST-ESCALATION "
                                "VERIFY FAIL — schedule replay "
                                "differs from target by %d/%d "
                                "cells; dropping schedule",
                                PATCH_LOG_PREFIX, ms_round_idx,
                                _esc_diff2,
                                oldArrangementArr.size,
                            )
                            _step_schedule = None
                    logger.info(
                        "%s ms_round=%d: SAT escalation succeeded "
                        "via %d cycle(s)",
                        PATCH_LOG_PREFIX, ms_round_idx,
                        len(_escalated),
                    )
                else:
                    raise ValueError(
                        f"[WISE Routing] Cannot produce SAT schedule "
                        f"for MS-gate reconfig at ms_round="
                        f"{ms_round_idx} after full-grid escalation.  "
                        f"This is a routing failure, not recoverable "
                        f"via heuristic.  diff="
                        f"{int(np.sum(oldArrangementArr != _step_target))}"
                        f"/{oldArrangementArr.size} cells."
                    )

        # INV-E1: No-op reconfig elision — skip reconfig when the
        # target layout is identical to the current layout.
        if np.array_equal(oldArrangementArr, _step_target):
            logger.info(
                "%s ms_round=%d: skipping no-op reconfig (layout unchanged)",
                PATCH_LOG_PREFIX,
                ms_round_idx,
            )
        else:
            oldArrangementArr = _apply_layout_as_reconfiguration(
                arch,
                wiseArch,
                oldArrangementArr,
                newArrangementArr,
                _step_target,
                allOps,
                _step_schedule,
                initial_placement=first_step.is_initial_placement,
                is_ms_reconfig=(
                    first_step.reconfig_context == "ms_gate"
                ),
                subgridsize=subgridsize,
            )
            barriers.append(len(allOps))
            last_reconfig_time = getattr(allOps[-1], "_reconfigTime", 0.0)
            reconfigTime += last_reconfig_time
            logger.info(
                "%s ms_round=%d: reconfig applied; dt=%.6f, total_reconfig=%.6f, total_ops=%d",
                PATCH_LOG_PREFIX,
                ms_round_idx,
                last_reconfig_time,
                reconfigTime,
                len(allOps),
            )

        # 4b) Run as many single-qubit operations as possible without routing
        # Fix 1: round-bounded drain — only drain ops whose CX
        # instruction belongs to routing round ≤ current.
        _drain_single_qubit_ops(
            ms_round_idx, max_drainable_round=ms_round_idx,
        )

        # No explicit drain↔MS barrier needed: the happens-before DAG
        # from happensBeforeForOperations() orders same-ion rotations
        # before MS gates (shared-component edges), and the WISE
        # type-selection heuristic ensures rotation batches and MS
        # batches are in separate time-steps.

        if not operationsLeft:
            break

        # 4c) Execute MS gates for this round
        ms_gates_executed, _unmatched = _execute_ms_gates(ms_round_idx, first_step.solved_pairs)
        _total_unmatched_pairs += _unmatched

        # 4d) Fix 2: Post-MS drain — after MS execution removes 2q
        # gates from operationsLeft, their post-MS rotations become
        # unblocked by D5.  Draining them NOW keeps post-MS ops in
        # the SAME segment as their MS gate, eliminating the
        # cross-segment gap that caused non-contiguous batch ranges.
        if operationsLeft:
            _drain_single_qubit_ops(
                ms_round_idx, max_drainable_round=ms_round_idx,
            )

        # Handle subsequent routing steps for the same MS round
        # (e.g. aligned -> offset tiling transitions).
        # NOTE: Do NOT break on empty solved_pairs — a step with 0 pairs may
        # still carry a layout change needed by later steps that DO have pairs.
        remaining_subsequent = round_steps[1:]
        for sub_idx, subsequent_step in enumerate(remaining_subsequent):
            # Layout transition steps (is_layout_transition) ALWAYS carry
            # layout changes that must be applied regardless of solved_pairs.
            # For normal MS rounds, skip when *all* remaining subsequent
            # steps have empty solved_pairs AND are not layout transitions.
            if not any(
                s.solved_pairs or s.is_layout_transition
                for s in remaining_subsequent[sub_idx:]
            ):
                break
            # ── Pre-flight for subsequent steps (Fix 14b) ─────────
            _sub_schedule = subsequent_step.schedule
            _sub_target = subsequent_step.layout_after
            _sub_is_ms = (
                subsequent_step.reconfig_context == "ms_gate"
            )
            if not np.array_equal(oldArrangementArr, _sub_target):
                _sub_needs_rebuild = _sub_schedule is None
                if not _sub_needs_rebuild:
                    _sub_replay = _simulate_schedule_replay(
                        oldArrangementArr, _sub_schedule,
                    )
                    _sub_needs_rebuild = not np.array_equal(
                        _sub_replay, _sub_target,
                    )
                if _sub_needs_rebuild:
                    if _sub_schedule is not None:
                        _sub_pf_diff = int(
                            np.sum(_sub_replay != _sub_target)
                        )
                    else:
                        _sub_pf_diff = int(
                            np.sum(oldArrangementArr != _sub_target)
                        )
                    logger.warning(
                        "%s ms_round=%d sub=%d: PRE-FLIGHT FAIL — "
                        "schedule %s (%d/%d cells).  Rebuilding.",
                        PATCH_LOG_PREFIX, ms_round_idx, sub_idx,
                        "replay mismatch" if _sub_schedule is not None
                        else "is None",
                        _sub_pf_diff, oldArrangementArr.size,
                    )
                    # P1: MS-gate reconfig → heuristic FORBIDDEN.
                    # P2: cache_replay / return_round → heuristic OK.
                    _sub_allow_heuristic = (
                        not _sub_is_ms
                        or _heuristic_fallback_for_noncache
                    )
                    try:
                        _sub_rb = _rebuild_schedule_for_layout(
                            oldArrangementArr, wiseArch, _sub_target,
                            subgridsize=subgridsize,
                            base_pmax_in=base_pmax_in or 1,
                            stop_event=stop_event,
                            max_inner_workers=max_inner_workers,
                            allow_heuristic_fallback=_sub_allow_heuristic,
                        )
                    except Exception as _sub_exc:
                        logger.error(
                            "%s ms_round=%d sub=%d: rebuild failed: %s",
                            PATCH_LOG_PREFIX, ms_round_idx, sub_idx,
                            _sub_exc,
                        )
                        _sub_rb = []
                    if _sub_rb:
                        for _srb_layout, _srb_sched, _ in _sub_rb[:-1]:
                            oldArrangementArr = (
                                _apply_layout_as_reconfiguration(
                                    arch, wiseArch, oldArrangementArr,
                                    newArrangementArr, _srb_layout,
                                    allOps, _srb_sched,
                                    initial_placement=False,
                                    is_ms_reconfig=False,
                                    subgridsize=subgridsize,
                                )
                            )
                            barriers.append(len(allOps))
                            reconfigTime += getattr(
                                allOps[-1], "_reconfigTime", 0.0,
                            )
                        _slast_layout, _slast_sched, _ = _sub_rb[-1]
                        _sub_schedule = _slast_sched
                        _sub_target = _slast_layout
                    else:
                        # Rebuild failed — drop the broken schedule.
                        # Without this, a stale schedule reaches
                        # _runOddEvenReconfig which raises ValueError
                        # for ms_gate context.  Setting None lets the
                        # heuristic (route_back) or escalation (ms_gate)
                        # handle it.
                        _sub_schedule = None

            # ── SAT-only guard for subsequent MS-gate reconfig ────
            # Mirror the first-step escalation guard (P1) for
            # subsequent steps that carry MS-gate context.
            if (
                _sub_is_ms
                and _sub_schedule is None
                and not np.array_equal(oldArrangementArr, _sub_target)
            ):
                _sub_escalated = _escalate_sat_for_ms_reconfig(
                    oldArrangementArr, wiseArch, _sub_target,
                    subgridsize=subgridsize,
                    base_pmax_in=base_pmax_in,
                    stop_event=stop_event,
                    progress_callback=_progress_callback,
                    max_inner_workers=max_inner_workers,
                )
                if _sub_escalated:
                    for _se_layout, _se_sched, _ in _sub_escalated[:-1]:
                        oldArrangementArr = (
                            _apply_layout_as_reconfiguration(
                                arch, wiseArch, oldArrangementArr,
                                newArrangementArr, _se_layout,
                                allOps, _se_sched,
                                initial_placement=False,
                                is_ms_reconfig=False,
                                subgridsize=subgridsize,
                            )
                        )
                        barriers.append(len(allOps))
                        reconfigTime += getattr(
                            allOps[-1], "_reconfigTime", 0.0,
                        )
                    _se_last, _se_sched_last, _ = _sub_escalated[-1]
                    _sub_schedule = _se_sched_last
                    _sub_target = _se_last
                    logger.info(
                        "%s ms_round=%d sub=%d: SAT escalation "
                        "succeeded via %d cycle(s)",
                        PATCH_LOG_PREFIX, ms_round_idx, sub_idx,
                        len(_sub_escalated),
                    )
                else:
                    raise ValueError(
                        f"[WISE Routing] Cannot produce SAT schedule "
                        f"for MS-gate reconfig at ms_round="
                        f"{ms_round_idx} sub={sub_idx} after "
                        f"full-grid escalation.  diff="
                        f"{int(np.sum(oldArrangementArr != _sub_target))}"
                        f"/{oldArrangementArr.size} cells."
                    )

            # INV-E1: No-op reconfig elision for subsequent steps too
            if np.array_equal(oldArrangementArr, _sub_target):
                logger.info(
                    "%s ms_round=%d sub=%d: skipping no-op subsequent reconfig",
                    PATCH_LOG_PREFIX,
                    ms_round_idx,
                    sub_idx,
                )
            else:
                oldArrangementArr = _apply_layout_as_reconfiguration(
                    arch,
                    wiseArch,
                    oldArrangementArr,
                    newArrangementArr,
                    _sub_target,
                    allOps,
                    _sub_schedule,
                    initial_placement=False,
                    is_ms_reconfig=_sub_is_ms,
                    subgridsize=subgridsize,
                )
                barriers.append(len(allOps))
                last_reconfig_time = getattr(allOps[-1], "_reconfigTime", 0.0)
                reconfigTime += last_reconfig_time

            # Fix A/D: drain single-qubit ops before subsequent-step
            # MS gates.  After a reconfig, new 1q ops may be eligible
            # (their ions are now in traps).
            _drain_single_qubit_ops(
                ms_round_idx, max_drainable_round=ms_round_idx,
            )
            # No drain↔MS barrier: DAG handles ordering.

            _sub_exec, _sub_unmatched = _execute_ms_gates(
                ms_round_idx, subsequent_step.solved_pairs
            )
            ms_gates_executed += _sub_exec
            _total_unmatched_pairs += _sub_unmatched

            # Fix 2: Post-MS drain for subsequent steps too
            if operationsLeft:
                _drain_single_qubit_ops(
                    ms_round_idx, max_drainable_round=ms_round_idx,
                )

        if ms_gates_executed:
            _last_round_had_ms = True
            # Count solved pairs across all steps in this round as the
            # expected gate count.  We do NOT index toMoves[ms_round_idx]
            # because the routing phase order may differ from the stim
            # circuit CX order (e.g. CSS surgery interleaves phases).
            candidate_gates = sum(
                len(s.solved_pairs) for s in round_steps
            )
            logger.info(
                "%s ms_round=%d: executed %d/%d MS gates; remaining_ops=%d, total_ops=%d",
                PATCH_LOG_PREFIX,
                ms_round_idx,
                ms_gates_executed,
                candidate_gates,
                len(operationsLeft),
                len(allOps),
            )

        # No end-of-round barrier: the next round's post-reconfig
        # barrier (L2854 pattern) already separates rounds.  The DAG
        # orders current-round MS gates before next-round reconfig via
        # shared-trap component edges.

    # Final drain of any remaining single-qubit ops after all MS rounds
    if operationsLeft:
        _drain_single_qubit_ops(total_ms_rounds)

    # ── Unmatched MS pairs assertion ──
    if _total_unmatched_pairs > 0:
        logger.error(
            "%s UNMATCHED MS PAIRS: %d total pairs across all rounds could not be "
            "matched to operations in operationsLeft. This indicates a systematic "
            "ion index mismatch between the routing path and execution path.",
            PATCH_LOG_PREFIX,
            _total_unmatched_pairs,
        )

    # ── Bug 5 fix: verification — all operations should be consumed ──
    if operationsLeft:
        from collections import Counter as _Counter
        _remaining_types = _Counter(type(op).__name__ for op in operationsLeft)
        logger.warning(
            "%s VERIFICATION FAILED: %d operations remain after "
            "routing execution. By type: %s",
            PATCH_LOG_PREFIX,
            len(operationsLeft),
            dict(_remaining_types),
        )
    else:
        logger.info(
            "%s VERIFICATION PASSED: all operations consumed",
            PATCH_LOG_PREFIX,
        )

    # ------------------------------------------------------------------
    # Fix 8 (revised) + INV-E2: Coalesce reconfigs that have no MS
    # gate between them.  This handles both:
    # (a) directly adjacent reconfigs (original Fix 8)
    # (b) reconfigs separated only by single-qubit ops (INV-E2)
    # The second reconfig's target layout supersedes the first's.
    # Single-qubit ops between them are preserved but the first
    # reconfig is removed — its transport was unnecessary since the
    # second reconfig will produce the final layout anyway.
    #
    # skip_coalesce=True preserves all intermediate reconfigs so that
    # animation consumers retain per-step semantics.
    # ------------------------------------------------------------------
    if not skip_coalesce:
        _coalesced_allOps: List[Operation] = []
        i = 0
        while i < len(allOps):
            op = allOps[i]
            if isinstance(op, GlobalReconfigurations):
                # Look ahead: find the next GlobalReconfigurations.
                # If there is no MS gate (TwoQubitMSGate) between
                # this reconfig and the next, coalesce them: drop this
                # reconfig and keep only the later one.
                j = i + 1
                next_reconfig_idx = None
                has_ms_between = False
                while j < len(allOps):
                    if isinstance(allOps[j], TwoQubitMSGate):
                        has_ms_between = True
                        break
                    if isinstance(allOps[j], GlobalReconfigurations):
                        next_reconfig_idx = j
                        break
                    j += 1

                if next_reconfig_idx is not None and not has_ms_between:
                    # INV-E2: Drop the current reconfig.  Any 1q ops
                    # between are kept — they execute before the next
                    # reconfig.  The next reconfig establishes the
                    # correct final layout.
                    logger.debug(
                        "%s INV-E2: dropping reconfig at op %d (no MS before next reconfig at op %d)",
                        PATCH_LOG_PREFIX, i, next_reconfig_idx,
                    )
                    # Skip this reconfig, continue from i+1
                    i += 1
                else:
                    _coalesced_allOps.append(op)
                    i += 1
            else:
                _coalesced_allOps.append(op)
                i += 1

        # Rebuild barrier indices from scratch based on type transitions.
        # A barrier is placed wherever the operation type changes or at
        # each GlobalReconfigurations boundary (transport must be isolated).
        _coalesced_barriers: List[int] = [0]
        for idx in range(1, len(_coalesced_allOps)):
            prev_type = type(_coalesced_allOps[idx - 1])
            curr_type = type(_coalesced_allOps[idx])
            if prev_type != curr_type or isinstance(_coalesced_allOps[idx], GlobalReconfigurations):
                _coalesced_barriers.append(idx)
        _coalesced_barriers.append(len(_coalesced_allOps))

        if len(_coalesced_allOps) < len(allOps):
            logger.info(
                "%s coalesced reconfigs: %d ops → %d ops",
                PATCH_LOG_PREFIX, len(allOps), len(_coalesced_allOps),
            )
        allOps = _coalesced_allOps
        barriers = _coalesced_barriers

    # ── End-of-routing assertion: verify no MS gates left unexecuted ──
    # This catches silent failures where solved_pairs didn't match operationsLeft
    remaining_2q = [op for op in operationsLeft if len(op.ions) == 2]
    if remaining_2q:
        remaining_ion_pairs = [
            frozenset(ion.idx for ion in op.ions) for op in remaining_2q
        ]
        logger.error(
            "%s END-OF-ROUTING ASSERTION FAILED: %d MS gates left unexecuted!\n"
            "  These ion pairs were never matched by any solved_pairs:\n"
            "  %s\n"
            "  This indicates a systematic mismatch between routing and execution.",
            PATCH_LOG_PREFIX,
            len(remaining_2q),
            list(remaining_ion_pairs)[:20],  # First 20 for brevity
        )

    # Report completion
    _report_progress(total_ms_rounds, stage=STAGE_COMPLETE)

    return allOps, barriers, reconfigTime


# =====================================================================
# Phase-Aware Gadget Routing
# =====================================================================


def _remap_schedule_to_global(
    schedule: List[Dict[str, Any]],
    row_offset: int,
    col_offset_ion: int,
) -> List[Dict[str, Any]]:
    """Offset swap coordinates from block-local to global ion-column space.

    Parameters
    ----------
    schedule : list of dict
        Pass entries with ``"phase"``, ``"h_swaps"``, ``"v_swaps"`` in
        block-local coordinates.
    row_offset : int
        Row offset (``r0`` from sub-grid).
    col_offset_ion : int
        Ion-column offset (``c0 * k``; note: ``c0`` from grid_region is
        in *trap* columns).

    Returns
    -------
    list of dict
        Schedule with globally remapped coordinates.
    """
    if row_offset == 0 and col_offset_ion == 0:
        return schedule
    remapped: List[Dict[str, Any]] = []
    for pass_info in schedule:
        new_pass: Dict[str, Any] = {
            "phase": pass_info["phase"],
            "h_swaps": [
                (r + row_offset, c + col_offset_ion)
                for (r, c) in pass_info.get("h_swaps", [])
            ],
            "v_swaps": [
                (r + row_offset, c + col_offset_ion)
                for (r, c) in pass_info.get("v_swaps", [])
            ],
        }
        remapped.append(new_pass)
    return remapped


def _merge_disjoint_block_schedules(
    sched_a: List[Dict[str, Any]],
    sched_b: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge SAT schedules from two *disjoint* grid regions.

    Because blocks occupy disjoint grid regions, swaps from different
    blocks never conflict and can execute in parallel within the same
    pass (provided they share phase and comparator parity).

    Delegates to ``_merge_patch_round_schedules`` which:

    1. Splits each schedule into its H-segment and V-segment,
    2. Merges all H passes (parity-aligned) across blocks,
    3. Merges all V passes (parity-aligned) across blocks,
    4. Concatenates: merged-H then merged-V.

    This guarantees the global schedule obeys the required
    ``H_odd, H_even, … , V_odd, V_even, …`` ordering and that
    **no swaps are silently dropped** when two blocks have different
    H/V pass ratios (the previous index-aligned implementation used
    a single phase label per index and ``_runOddEvenReconfig`` only
    applies swaps matching the label, silently discarding cross-phase
    swaps).
    """
    return _merge_patch_round_schedules([sched_a, sched_b])


def _merge_block_routing_steps(
    block_steps: Dict[str, List[RoutingStep]],
    block_sub_grids: Dict[str, "BlockSubGrid"],
    base_layout: np.ndarray,
    k: int,
) -> Tuple[List[RoutingStep], np.ndarray]:
    """Merge per-block ``RoutingStep`` lists into global-grid steps.

    Each block was routed independently on its disjoint sub-grid.
    This function overlays layouts, remaps swap coordinates, and
    merges schedules so the execution layer sees a single stream
    of global ``RoutingStep`` objects.

    Parameters
    ----------
    block_steps : Dict[str, List[RoutingStep]]
        Per-block routing results.
    block_sub_grids : Dict[str, BlockSubGrid]
        Sub-grid allocations with ``grid_region = (r0, c0, r1, c1)``
        (trap columns).
    base_layout : np.ndarray
        Global layout before this set of steps.
    k : int
        Ions per trap.

    Returns
    -------
    Tuple[List[RoutingStep], np.ndarray]
        ``(merged_steps, final_layout)``.
    """
    if not block_steps:
        return [], base_layout.copy()

    max_steps = max(len(steps) for steps in block_steps.values())
    if max_steps == 0:
        return [], base_layout.copy()

    # Track each block's latest layout for padding shorter sequences
    block_latest_layouts: Dict[str, np.ndarray] = {}
    for block_name, sg in block_sub_grids.items():
        r0, c0, r1, c1 = sg.grid_region
        c0i, c1i = c0 * k, c1 * k
        block_latest_layouts[block_name] = base_layout[r0:r1, c0i:c1i].copy()

    merged_steps: List[RoutingStep] = []
    current_global = base_layout.copy()

    for step_idx in range(max_steps):
        merged_layout = current_global.copy()
        merged_schedule: Optional[List[Dict[str, Any]]] = None
        merged_pairs: List[Tuple[int, int]] = []
        ms_round_index = 0
        from_cache = True
        tiling_meta = (0, step_idx)
        # Propagate reconfig_context: if ANY block contributed a
        # cache_replay step at this index, the merged step inherits
        # cache_replay (most permissive).  return_round takes next
        # priority, then ms_gate (default / strictest).
        _merged_reconfig_context = "ms_gate"
        _block_owned_cells: Dict[Tuple[int, int], str] = {}

        for block_name, steps in block_steps.items():
            sg = block_sub_grids[block_name]
            r0, c0, r1, c1 = sg.grid_region
            c0i, c1i = c0 * k, c1 * k

            if step_idx < len(steps):
                step = steps[step_idx]
                block_layout = step.layout_after

                ms_round_index = step.ms_round_index
                from_cache = from_cache and step.from_cache
                tiling_meta = step.tiling_meta

                # Propagate most permissive reconfig_context
                if step.reconfig_context == "cache_replay":
                    _merged_reconfig_context = "cache_replay"
                elif (
                    step.reconfig_context == "return_round"
                    and _merged_reconfig_context == "ms_gate"
                ):
                    _merged_reconfig_context = "return_round"

                # Overlay block layout onto global
                merged_layout[r0:r1, c0i:c1i] = block_layout
                block_latest_layouts[block_name] = block_layout.copy()

                # Track which cells belong to each block for
                # ion-conservation repair below.
                # (block_owned_cells is declared before the block loop.)
                for _br in range(r0, r1):
                    for _bc in range(c0i, c1i):
                        _block_owned_cells[(_br, _bc)] = block_name

                # Remap + merge schedule (index-aligned for disjoint blocks)
                if step.schedule is not None:
                    remapped = _remap_schedule_to_global(
                        step.schedule, r0, c0i,
                    )
                    if merged_schedule is None:
                        merged_schedule = remapped
                    else:
                        merged_schedule = _merge_disjoint_block_schedules(
                            merged_schedule, remapped,
                        )

                merged_pairs.extend(step.solved_pairs)
            else:
                # Block already finished — keep its latest layout
                merged_layout[r0:r1, c0i:c1i] = block_latest_layouts[block_name]

        # ── Fix C: Post-merge ion conservation repair ─────────────
        # Each block's SAT solver works on its sub-grid independently.
        # "Spectator" ions (physical slots not mapped to that block's
        # qubits) can be rearranged by the solver.  When we overlay
        # block layouts onto the global grid, the same ion can appear
        # in multiple block regions (duplicate) or disappear (missing)
        # because two overlapping blocks both claimed or discarded it.
        #
        # Fix: after overlaying all blocks, enforce global ion
        # conservation by detecting duplicates and missing ions,
        # keeping the copy that is closest to the ion's position in
        # current_global and restoring missing ions to their original
        # positions.
        _n_rows, _n_cols = merged_layout.shape
        _old_ions = set(int(v) for v in current_global.flatten() if int(v) != 0)
        _flat_merged = merged_layout.flatten()
        from collections import Counter as _Counter
        _ion_counts = _Counter(int(v) for v in _flat_merged if int(v) != 0)
        _duplicates = {ion for ion, cnt in _ion_counts.items() if cnt > 1}
        _new_ions = set(_ion_counts.keys())
        _missing = _old_ions - _new_ions
        _extra = _new_ions - _old_ions

        _ion_conservation_repaired = False
        if _duplicates or _missing or _extra:
            _ion_conservation_repaired = True
            logger.warning(
                "%s _merge_block_routing_steps step=%d: ion conservation "
                "violation — duplicates=%s, missing=%s, extra=%s. "
                "Repairing merged layout.",
                PATCH_LOG_PREFIX, step_idx,
                sorted(_duplicates), sorted(_missing), sorted(_extra),
            )

            # 1) Remove extra ions (in merged but not in current_global)
            for _er in range(_n_rows):
                for _ec in range(_n_cols):
                    if int(merged_layout[_er, _ec]) in _extra:
                        merged_layout[_er, _ec] = 0

            # 2) De-duplicate: keep the copy closest to old position
            for _dup_ion in sorted(_duplicates):
                _dup_positions = []
                for _dr in range(_n_rows):
                    for _dc in range(_n_cols):
                        if int(merged_layout[_dr, _dc]) == _dup_ion:
                            _dup_positions.append((_dr, _dc))
                if len(_dup_positions) <= 1:
                    continue
                # Prefer the copy closest to old position
                _old_pos = np.argwhere(current_global == _dup_ion)
                if len(_old_pos) > 0:
                    _od, _oc = int(_old_pos[0][0]), int(_old_pos[0][1])
                    _dup_positions.sort(
                        key=lambda rc: abs(rc[0] - _od) + abs(rc[1] - _oc)
                    )
                for _dd, _dc in _dup_positions[1:]:
                    merged_layout[_dd, _dc] = 0

            # 3) Place missing ions back at their old positions
            _empty_cells = []
            for _mr in range(_n_rows):
                for _mc in range(_n_cols):
                    if int(merged_layout[_mr, _mc]) == 0:
                        _empty_cells.append((_mr, _mc))

            for _mi in sorted(_missing):
                _old_pos = np.argwhere(current_global == _mi)
                _placed = False
                if len(_old_pos) > 0:
                    _od, _oc = int(_old_pos[0][0]), int(_old_pos[0][1])
                    if int(merged_layout[_od, _oc]) == 0:
                        merged_layout[_od, _oc] = _mi
                        _empty_cells = [
                            (r, c) for r, c in _empty_cells
                            if not (r == _od and c == _oc)
                        ]
                        _placed = True
                if not _placed and _empty_cells:
                    _ed, _ec_cell = _empty_cells.pop(0)
                    merged_layout[_ed, _ec_cell] = _mi
                    _placed = True
                if not _placed:
                    logger.error(
                        "%s _merge_block_routing_steps: could not place "
                        "missing ion %d — no empty cells",
                        PATCH_LOG_PREFIX, _mi,
                    )

        # ── Fix B: Post-merge schedule verification ──────────────
        # Replay the merged schedule on current_global and compare
        # with merged_layout.  If they don't match, drop the
        # schedule (let execution-time Fix 14 handle it) and
        # downgrade context so heuristic fallback is allowed.
        if merged_schedule is not None and not np.array_equal(
            current_global, merged_layout,
        ):
            _replay_result = _simulate_schedule_replay(
                current_global, merged_schedule,
            )
            if not np.array_equal(_replay_result, merged_layout):
                _replay_diff = int(np.sum(_replay_result != merged_layout))
                logger.warning(
                    "%s _merge_block_routing_steps step=%d: merged "
                    "schedule replay mismatch (%d/%d cells).  "
                    "Dropping schedule.",
                    PATCH_LOG_PREFIX, step_idx,
                    _replay_diff, merged_layout.size,
                )
                merged_schedule = None
                _merged_reconfig_context = "cache_replay"

        # If ion conservation was repaired, the merged schedule
        # (computed for the pre-repair layout) is invalid.
        if _ion_conservation_repaired and merged_schedule is not None:
            logger.info(
                "%s _merge_block_routing_steps step=%d: ion conservation "
                "repair invalidated merged schedule; dropping.",
                PATCH_LOG_PREFIX, step_idx,
            )
            merged_schedule = None
            _merged_reconfig_context = "cache_replay"

        merged_steps.append(RoutingStep(
            layout_after=merged_layout,
            schedule=merged_schedule,
            solved_pairs=merged_pairs,
            ms_round_index=ms_round_index,
            from_cache=from_cache,
            tiling_meta=tiling_meta,
            can_merge_with_next=False,
            is_initial_placement=False,
            layout_before=current_global.copy(),
            reconfig_context=_merged_reconfig_context,
        ))
        current_global = merged_layout

    return merged_steps, current_global


def ionRoutingGadgetArch(
    arch: QCCDArch,
    wiseArch: QCCDWiseArch,
    operations: Sequence[QubitOperation],
    lookahead: int = 4,
    subgridsize: Optional[Tuple[int, int, int]] = None,
    base_pmax_in: int = None,
    toMoveOps: Sequence[Sequence[TwoQubitMSGate]] = None,
    routing_config: Optional[WISERoutingConfig] = None,
    stop_event: Optional[Any] = None,
    max_inner_workers: int | None = None,
    *,
    qec_metadata: Any = None,
    gadget: Any = None,
    qubit_allocation: Any = None,
    block_sub_grids: Optional[Dict[str, Any]] = None,
    toMove_phase_tags: Optional[List[int]] = None,
    _compiler_q2i: Optional[Dict[int, int]] = None,
    skip_coalesce: bool = False,
    allow_flat_fallback: bool = True,
    replay_level: int = 1,
    block_level_slicing: bool = True,
) -> Tuple[Sequence[Operation], Sequence[int], float]:
    """Phase-aware routing for fault-tolerant gadget experiments.

    Same interface and return type as ``ionRoutingWISEArch``.  Uses
    ``QECMetadata.phases`` to decompose the circuit into temporal phases
    (EC, gadget, transition) and routes each phase independently with
    phase-specific optimisations:

    - **EC phases**: ion-return BT constraints ensure ions return to
      their starting positions after each round.  Identical phases are
      detected via ``round_signature`` and the routing result is cached
      and replayed.
    - **Gadget phases**: routed without BT constraints to allow
      cross-block ion movement.

    Falls back to flat ``ionRoutingWISEArch`` routing when phase metadata
    is unavailable or the MS round count doesn't match.

    Parameters
    ----------
    arch, wiseArch, operations, lookahead, subgridsize, base_pmax_in,
    toMoveOps, routing_config, stop_event, max_inner_workers
        Same as ``ionRoutingWISEArch``.
    qec_metadata : QECMetadata, optional
        Rich metadata with ``phases`` list.
    gadget : Gadget, optional
        The gadget object.  Used to:
        - Query ``get_phase_active_blocks(phase_idx)`` for Level 1
          spatial slicing (excluding idle blocks in gadget phases).
        - Call ``derive_gadget_ms_pairs()`` to obtain/validate the
          cross-block MS pairs for each gadget phase.
        - Track multi-phase gadget state via ``num_phases``.
    qubit_allocation : QubitAllocation, optional
        Unified qubit allocation from the experiment layout.  Used to:
        - Auto-build ``block_sub_grids`` (per-block disjoint sub-grid
          regions) when they are not pre-supplied, enabling per-block
          EC routing and Level 1 gadget slicing.
        - Extract ``data_qubit_idxs`` for accurate data/ancilla
          classification (the QUBIT_COORDS parity heuristic breaks
          with multi-block x-offsets).
        - Feed ``derive_gadget_ms_pairs()`` with block data ranges
          for transversal pair derivation.

    Returns
    -------
    Tuple[Sequence[Operation], Sequence[int], float]
        ``(allOps, barriers, reconfigTime)`` — identical to
        ``ionRoutingWISEArch``.
    """
    # ------------------------------------------------------------------
    # 0a) Auto-build block_sub_grids from qubit_allocation if not provided
    # ------------------------------------------------------------------
    if (
        block_sub_grids is None
        and qubit_allocation is not None
        and qec_metadata is not None
        and hasattr(qec_metadata, 'block_allocations')
        and qec_metadata.block_allocations
    ):
        from ..utils.gadget_routing import partition_grid_for_blocks
        try:
            block_sub_grids = partition_grid_for_blocks(
                qec_metadata, qubit_allocation, wiseArch.k,
            )
            logger.info(
                "%s ionRoutingGadgetArch: auto-built block_sub_grids "
                "from qubit_allocation (%d blocks)",
                PATCH_LOG_PREFIX, len(block_sub_grids),
            )
        except (ValueError, KeyError, IndexError) as exc:
            logger.warning(
                "%s ionRoutingGadgetArch: failed to auto-build "
                "block_sub_grids: %s",
                PATCH_LOG_PREFIX, exc, exc_info=True,
            )

    # ------------------------------------------------------------------
    # 0a.5) Normalize block_sub_grids ion index space
    #
    # ``partition_grid_for_blocks`` may provide ``ion_indices`` in the
    # logical-qubit index space, while routing runs in physical ion-index
    # space.  When a compiler qubit->ion map is available, remap block
    # indices (and qubit_to_ion entries) so per-block EC routing and
    # ion-to-block assignment are consistent.
    # ------------------------------------------------------------------
    if block_sub_grids and _compiler_q2i:
        try:
            _q2i_keys = set(_compiler_q2i.keys())
            _q2i_vals = set(_compiler_q2i.values())
            _all_block_idxs: List[int] = []
            for _sg in block_sub_grids.values():
                _all_block_idxs.extend(getattr(_sg, 'ion_indices', []) or [])

            _key_hits = sum(1 for _idx in _all_block_idxs if _idx in _q2i_keys)
            _val_hits = sum(1 for _idx in _all_block_idxs if _idx in _q2i_vals)

            if _key_hits > _val_hits:
                for _sg in block_sub_grids.values():
                    _raw_idxs = getattr(_sg, 'ion_indices', []) or []
                    _mapped_idxs = [
                        int(_compiler_q2i.get(_idx, _idx))
                        for _idx in _raw_idxs
                    ]
                    _sg.ion_indices = sorted(set(_mapped_idxs))

                    _q2i = getattr(_sg, 'qubit_to_ion', None)
                    if isinstance(_q2i, dict) and _q2i:
                        for _q in list(_q2i.keys()):
                            _q2i[_q] = int(_compiler_q2i.get(_q, _q2i[_q]))

                logger.info(
                    "%s ionRoutingGadgetArch: normalized block_sub_grids "
                    "ion index space using compiler q2i map",
                    PATCH_LOG_PREFIX,
                )
        except (KeyError, AttributeError, TypeError) as _norm_exc:
            logger.warning(
                "%s ionRoutingGadgetArch: block_sub_grids normalization "
                "skipped: %s",
                PATCH_LOG_PREFIX, _norm_exc, exc_info=True,
            )

    # ------------------------------------------------------------------
    # 0b) Build parallelPairs and initial layout (H5+H6: shared helpers)
    # ------------------------------------------------------------------
    parallelismAllowed = wiseArch.m * wiseArch.n
    parallelPairs, _ = _build_parallel_pairs(
        operations, toMoveOps, parallelismAllowed,
    )

    oldArrangementArr, ionsSorted, active_ions = _build_grid_state(
        arch, wiseArch,
    )
    active_ions_set = set(active_ions)

    # ------------------------------------------------------------------
    # 1) Determine phase boundaries from qec_metadata.phases
    # ------------------------------------------------------------------
    phases = []
    if qec_metadata is not None and hasattr(qec_metadata, 'phases'):
        phases = [
            p for p in qec_metadata.phases
            if getattr(p, 'phase_type', '') not in ('init', 'measure', '')
        ]

    # ------------------------------------------------------------------
    # 1b) Read per-phase MS pair counts directly from QECMetadata.
    #
    # PhaseInfo carries ``ms_pair_count`` (populated post-build by
    # ``to_stim()`` via ``_count_cx_instructions``).  These are exact
    # circuit-counted values — one per CX/CZ stim instruction object =
    # one toMoveOps entry = one parallelPairs entry.  No heuristic or
    # formula-based estimation needed.
    # ------------------------------------------------------------------
    phase_pair_counts: Optional[List[int]] = None
    _cx_per_ec_round: Optional[int] = None

    if phases and len(parallelPairs) > 0:
        # Read cx_per_ec_round from metadata
        if qec_metadata is not None and hasattr(qec_metadata, 'cx_per_ec_round'):
            _cx_per_ec_round = qec_metadata.cx_per_ec_round

        # Build phase_pair_counts from circuit-counted ms_pair_count.
        # These values are patched by to_stim() after circuit construction,
        # so they reflect actual CX instruction counts — no heuristic needed.
        phase_pair_counts = [
            getattr(p, 'ms_pair_count', 0) for p in phases
        ]

        if phase_pair_counts is not None:
            # Sanity check
            if sum(phase_pair_counts) != len(parallelPairs):
                logger.warning(
                    "%s metadata phase pair counts sum to %d, "
                    "expected %d — falling back to flat routing",
                    PATCH_LOG_PREFIX,
                    sum(phase_pair_counts),
                    len(parallelPairs),
                )
                phase_pair_counts = None

        # ----------------------------------------------------------
        # Fix 10: Split shared-ion rounds into conflict-free sub-rounds
        #
        # Bridge CX instructions (e.g. CX 0 21 1 21 2 21 3 21) have
        # multiple pairs sharing one bridge ancilla ion.  The SAT solver
        # can only co-locate one pair at a time (the shared ion can only
        # be in one trap), so we must split these into separate routing
        # rounds.
        # ----------------------------------------------------------
        if phase_pair_counts is not None:
            _orig_len = len(parallelPairs)
            _new_pp: List[List[Tuple[int, int]]] = []
            # Track how many new entries each original entry expands to,
            # keyed by original index.
            _split_counts: List[int] = []

            for _round_pairs in parallelPairs:
                if len(_round_pairs) <= 1:
                    # No conflict possible with 0 or 1 pair.
                    _new_pp.append(_round_pairs)
                    _split_counts.append(1)
                    continue

                # Greedy bin-packing: assign each pair to the first
                # sub-round where neither ion is already present.
                _sub_rounds: List[List[Tuple[int, int]]] = []
                _sub_ions: List[set] = []  # ions used in each sub-round
                for _pair in _round_pairs:
                    _placed = False
                    for _si, _sr in enumerate(_sub_rounds):
                        if _pair[0] not in _sub_ions[_si] and _pair[1] not in _sub_ions[_si]:
                            _sr.append(_pair)
                            _sub_ions[_si].add(_pair[0])
                            _sub_ions[_si].add(_pair[1])
                            _placed = True
                            break
                    if not _placed:
                        _sub_rounds.append([_pair])
                        _sub_ions.append({_pair[0], _pair[1]})

                for _sr in _sub_rounds:
                    _new_pp.append(_sr)
                _split_counts.append(len(_sub_rounds))

            if len(_new_pp) != _orig_len:
                logger.info(
                    "%s Fix10: split shared-ion rounds: %d → %d entries",
                    PATCH_LOG_PREFIX, _orig_len, len(_new_pp),
                )
                parallelPairs = _new_pp

                # Update phase_pair_counts to reflect the expanded rounds.
                _new_ppc: List[int] = []
                _cursor = 0
                for _count in phase_pair_counts:
                    _new_count = 0
                    for _i in range(_cursor, _cursor + _count):
                        _new_count += _split_counts[_i]
                    _cursor += _count
                    _new_ppc.append(_new_count)
                phase_pair_counts = _new_ppc

                logger.info(
                    "%s Fix10: updated phase_pair_counts: %s (sum=%d)",
                    PATCH_LOG_PREFIX, phase_pair_counts,
                    sum(phase_pair_counts),
                )

        # ----------------------------------------------------------
        # Fix 2b: Validate toMove_phase_tags consistency
        # ----------------------------------------------------------
        if (
            phase_pair_counts is not None
            and toMove_phase_tags is not None
            and len(toMove_phase_tags) == len(parallelPairs)
        ):
            # Verify that per-phase grouping implied by tags matches
            # the expected phase_pair_counts.
            _tag_counts = [0] * len(phases)
            for _tag in toMove_phase_tags:
                if 0 <= _tag < len(_tag_counts):
                    _tag_counts[_tag] += 1
            if _tag_counts != phase_pair_counts:
                logger.warning(
                    "%s toMove_phase_tags grouping %s doesn't match "
                    "phase_pair_counts %s — tags will be ignored",
                    PATCH_LOG_PREFIX, _tag_counts, phase_pair_counts,
                )
            else:
                logger.debug(
                    "%s toMove_phase_tags validated: %d entries "
                    "across %d phases",
                    PATCH_LOG_PREFIX,
                    len(toMove_phase_tags),
                    len(phases),
                )

        # ----------------------------------------------------------
        # Fix 3: Per-phase structural validation (beyond count-only)
        # ----------------------------------------------------------
        if phase_pair_counts is not None and block_sub_grids:
            _pair_cursor = 0
            for _ph_i, (_phase, _n_pairs) in enumerate(
                zip(phases, phase_pair_counts)
            ):
                if _n_pairs <= 0:
                    continue
                _ph_pairs = parallelPairs[_pair_cursor:_pair_cursor + _n_pairs]
                _pair_cursor += _n_pairs

                _ph_type = getattr(_phase, 'phase_type', '') or ''
                _is_ec_phase = (
                    _ph_type in ('ec', 'stabilizer_round', 'final_round')
                    or _ph_type.startswith('stabilizer_round')
                )

                if _is_ec_phase and block_sub_grids:
                    # EC phases: all ion pairs should be within a
                    # single block's ion range (no cross-block pairs).
                    _block_ion_sets = {}
                    for _bname, _bsg in block_sub_grids.items():
                        _b_ions = set()
                        _bq2i = getattr(_bsg, 'qubit_to_ion', {})
                        _b_ions.update(_bq2i.values())
                        if not _b_ions and hasattr(_bsg, 'ion_indices'):
                            _b_ions = set(_bsg.ion_indices)  # Fix A: use physical IDs directly
                        _block_ion_sets[_bname] = _b_ions

                    # Fix D: Track already-warned cross-block pairs to
                    # avoid duplicate log lines.
                    _warned_cross_pairs: set = set()
                    for _round_pairs in _ph_pairs:
                        for _a, _d in _round_pairs:
                            _a_block = None
                            _d_block = None
                            for _bn, _bions in _block_ion_sets.items():
                                if _a in _bions:
                                    _a_block = _bn
                                if _d in _bions:
                                    _d_block = _bn
                            if (
                                _a_block is not None
                                and _d_block is not None
                                and _a_block != _d_block
                            ):
                                _pair_key = (min(_a, _d), max(_a, _d), _ph_i)
                                if _pair_key not in _warned_cross_pairs:
                                    _warned_cross_pairs.add(_pair_key)
                                    logger.warning(
                                        "%s Fix3: EC phase %d has "
                                        "cross-block pair (%d, %d): "
                                        "blocks %s vs %s",
                                        PATCH_LOG_PREFIX, _ph_i,
                                        _a, _d, _a_block, _d_block,
                                    )

    if not phases or phase_pair_counts is None:
        # Metadata unavailable or epoch analysis failed.
        if not allow_flat_fallback:
            raise RuntimeError(
                f"{PATCH_LOG_PREFIX} ionRoutingGadgetArch: phase metadata "
                f"unavailable (phases={len(phases)}, "
                f"phase_pair_counts={phase_pair_counts}) and "
                f"allow_flat_fallback=False"
            )
        # Fall back to flat ionRoutingWISEArch routing.
        logger.warning(
            "%s ionRoutingGadgetArch: falling back to flat routing "
            "(phases=%d, phase_pair_counts=%s, circuit_rounds=%d)",
            PATCH_LOG_PREFIX,
            len(phases),
            phase_pair_counts,
            len(parallelPairs),
        )
        return ionRoutingWISEArch(
            arch, wiseArch, operations,
            lookahead=lookahead,
            subgridsize=subgridsize,
            base_pmax_in=base_pmax_in,
            toMoveOps=toMoveOps,
            routing_config=routing_config,
            stop_event=stop_event,
            max_inner_workers=max_inner_workers,
            skip_coalesce=skip_coalesce,
        )

    # ------------------------------------------------------------------
    # 2) Phase-aware routing via unified core
    # ------------------------------------------------------------------
    from ..utils.gadget_routing import (
        route_full_experiment_as_steps,
        _build_plans_from_compiler_pairs,
        decompose_into_phases,
        partition_grid_for_blocks,
    )

    plans = _build_plans_from_compiler_pairs(
        phases, parallelPairs, phase_pair_counts, block_sub_grids or {},
    )

    # ------------------------------------------------------------------
    # Fix 1: Cross-validate compiler pairs against analytics derivation
    # ------------------------------------------------------------------
    if (
        qec_metadata is not None
        and gadget is not None
        and qubit_allocation is not None
        and block_sub_grids
    ):
        try:
            # Use the actual compiler qubit→ion mapping if available
            # (Fix 4).  Falls back to the q+1 convention otherwise.
            if _compiler_q2i:
                _xv_q2i = dict(_compiler_q2i)
                # Also include bridge ancilla qubits
                if hasattr(qubit_allocation, 'bridge_ancillas'):
                    for _bi in (qubit_allocation.bridge_ancillas or []):
                        _gi = _bi[0]
                        if _gi not in _xv_q2i:
                            _xv_q2i[_gi] = _compiler_q2i.get(_gi, _gi + 1)
            else:
                _xv_q2i = {
                    q: q + 1
                    for ba in qec_metadata.block_allocations
                    for q in (
                        list(ba.data_qubits)
                        + list(ba.x_ancilla_qubits)
                        + list(ba.z_ancilla_qubits)
                    )
                }
                # Include bridge ancilla qubits
                if hasattr(qubit_allocation, 'bridge_ancillas'):
                    for _bi in (qubit_allocation.bridge_ancillas or []):
                        _gi = _bi[0]
                        if _gi not in _xv_q2i:
                            _xv_q2i[_gi] = _gi + 1

            _xv_plans = decompose_into_phases(
                qec_metadata, gadget, qubit_allocation,
                block_sub_grids, _xv_q2i, wiseArch.k,
            )

            # Compare pair sets per matched phase
            _compiler_cursor = 0
            _xv_idx = 0
            for _c_ph_i, (_phase, _n_pairs) in enumerate(
                zip(phases, phase_pair_counts)
            ):
                _c_pairs_flat = set()
                if _n_pairs > 0:
                    for _rp in parallelPairs[
                        _compiler_cursor:_compiler_cursor + _n_pairs
                    ]:
                        for _p in _rp:
                            _c_pairs_flat.add(tuple(sorted(_p)))
                _compiler_cursor += _n_pairs

                # Find matching analytics plan by phase_index
                _a_pairs_flat = set()
                for _xp in _xv_plans:
                    if getattr(_xp, 'phase_index', -1) == getattr(
                        _phase, '_original_index', _c_ph_i
                    ):
                        for _a_rp in (_xp.ms_pairs_per_round or []):
                            for _ap in _a_rp:
                                _a_pairs_flat.add(tuple(sorted(_ap)))
                        break

                if _c_pairs_flat and _a_pairs_flat:
                    _diff_ca = _c_pairs_flat - _a_pairs_flat
                    _diff_ac = _a_pairs_flat - _c_pairs_flat
                    if _diff_ca or _diff_ac:
                        logger.debug(
                            "%s Fix1 cross-validation: phase %d "
                            "(%s) pair mismatch — "
                            "compiler-only: %s, analytics-only: %s",
                            PATCH_LOG_PREFIX, _c_ph_i,
                            getattr(_phase, 'phase_type', '?'),
                            _diff_ca, _diff_ac,
                        )
        except Exception as _xv_exc:
            logger.warning(
                "%s Fix1 cross-validation skipped: %s",
                PATCH_LOG_PREFIX, _xv_exc, exc_info=True,
            )

    all_routing_steps, _ = route_full_experiment_as_steps(
        initial_layout=oldArrangementArr,
        n=wiseArch.n,
        m=wiseArch.m,
        k=wiseArch.k,
        active_ions=active_ions,
        plans=plans,
        block_sub_grids=block_sub_grids or {},
        subgridsize=subgridsize,
        base_pmax_in=base_pmax_in or 1,
        lookahead=lookahead,
        max_inner_workers=max_inner_workers,
        stop_event=stop_event,
        cx_per_ec_round=_cx_per_ec_round,
        cache_ec_rounds=(
            routing_config.cache_ec_rounds if routing_config else True
        ),
        replay_level=(
            routing_config.replay_level if routing_config else replay_level
        ),
        block_level_slicing=(
            routing_config.block_level_slicing if routing_config else block_level_slicing
        ),
        heuristic_cache_replay=(
            routing_config.heuristic_cache_replay if routing_config else False
        ),
        heuristic_route_back=(
            routing_config.heuristic_route_back if routing_config else False
        ),
        heuristic_fallback_for_noncache=(
            routing_config.heuristic_fallback_for_noncache if routing_config else False
        ),
        progress_callback=routing_config.progress_callback if routing_config else None,
    )

    # ------------------------------------------------------------------
    # Fix 12: Fallback to flat routing if phase-aware produced zero MS pairs
    # ------------------------------------------------------------------
    _total_solved_pairs = sum(len(s.solved_pairs) for s in all_routing_steps)
    if _total_solved_pairs == 0 and len(parallelPairs) > 0:
        if not allow_flat_fallback:
            raise RuntimeError(
                f"{PATCH_LOG_PREFIX} ionRoutingGadgetArch: phase-aware "
                f"routing returned 0 solved pairs but expected "
                f"{len(parallelPairs)} MS rounds, and "
                f"allow_flat_fallback=False"
            )
        logger.warning(
            "%s ionRoutingGadgetArch: phase-aware routing returned 0 "
            "solved pairs but expected %d MS rounds — falling back to flat",
            PATCH_LOG_PREFIX, len(parallelPairs),
        )
        return ionRoutingWISEArch(
            arch, wiseArch, operations,
            lookahead=lookahead,
            subgridsize=subgridsize,
            base_pmax_in=base_pmax_in,
            toMoveOps=toMoveOps,
            routing_config=routing_config,
            stop_event=stop_event,
            max_inner_workers=max_inner_workers,
            skip_coalesce=skip_coalesce,
        )

    logger.info(
        "%s ionRoutingGadgetArch: total routing_steps=%d across %d phases",
        PATCH_LOG_PREFIX,
        len(all_routing_steps),
        len(plans),
    )

    # ── Bug 5: Pre-execution verification of solved pairs ──
    _total_expected_pairs = sum(len(pp) for pp in parallelPairs)
    # Also count unique pairs to detect duplicates
    _all_solved = []
    for s in all_routing_steps:
        _all_solved.extend(frozenset(p) for p in s.solved_pairs)
    _unique_solved = len(set(_all_solved))
    logger.debug(
        "%s PAIR DIAG: expected=%d, total_solved=%d, unique_solved=%d, "
        "routing_steps=%d, parallelPairs_len=%d",
        PATCH_LOG_PREFIX,
        _total_expected_pairs,
        _total_solved_pairs,
        _unique_solved,
        len(all_routing_steps),
        len(parallelPairs),
    )
    if _total_solved_pairs < _total_expected_pairs:
        logger.warning(
            "%s PAIR COVERAGE: solved %d / %d expected pairs (%.1f%%). "
            "%d pairs may remain unexecuted.",
            PATCH_LOG_PREFIX,
            _total_solved_pairs,
            _total_expected_pairs,
            100.0 * _total_solved_pairs / max(_total_expected_pairs, 1),
            _total_expected_pairs - _total_solved_pairs,
        )

    # ------------------------------------------------------------------
    # 3) Delegate execution to ionRoutingWISEArch
    # ------------------------------------------------------------------
    return ionRoutingWISEArch(
        arch, wiseArch, operations,
        lookahead=lookahead,
        subgridsize=subgridsize,
        base_pmax_in=base_pmax_in,
        toMoveOps=toMoveOps,
        routing_config=routing_config,
        stop_event=stop_event,
        max_inner_workers=max_inner_workers,
        _precomputed_routing_steps=all_routing_steps,
        skip_coalesce=skip_coalesce,
    )
