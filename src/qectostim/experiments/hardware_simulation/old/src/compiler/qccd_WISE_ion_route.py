from typing import (
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
import copy
import os
import time
import logging
from copy import deepcopy

import numpy as np

from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *
from src.utils.qccd_arch import *
from src.compiler._qccd_WISE_ion_routing import *
from src.compiler.qccd_qubits_to_ions import *


PATCH_LOG_PREFIX = "[PatchRoute]"
PATCH_VERBOSE_MOVES = os.environ.get("WISE_PATCH_VERBOSE", "0") not in ("0", "")

LOGGER_NAME = "wise.qccd.route"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())
logger.propagate = False

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
        seen_v = False

        for pass_info in sched:
            if not _pass_has_swaps(pass_info):
                continue
            phase = pass_info["phase"]
            if phase == "H" and not seen_v:
                h_seg.append(pass_info)
            else:
                if phase == "V" and not seen_v:
                    seen_v = True
                else:
                    seen_v = True
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


def _merge_patch_round_schedules(
    patch_round_schedules: List[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Merge the per-patch schedule for a single round into a global schedule where
    all H passes occur before any V pass, ensuring passes only combine when they
    share phase and comparator parity.
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
    base_pmax = base_pmax_in or R
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

        for tiling_idx, (off_r, off_c) in enumerate(tilings):
            if all(len(rp) == 0 for rp in remaining_pairs):
                break

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
                    # No gates to solve in this patch; skip SAT even if boundary prefs exist.
                    remaining_pairs = new_remaining
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
                        bt_soft=use_bt_softs
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
                patch_schedules_this_tiling.append(patch_schedule)

                remaining_pairs = new_remaining

            merged_tiling_schedule = _merge_patch_schedules(patch_schedules_this_tiling, R)

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
            if no_progress_cycles >= max_cycles or fully_global_patch:
                logger.warning(
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
        logger.warning(
            "%s %d gate(s) remain unresolved after patch routing. Example (round, (ion_a, ion_b)): %s",
            PATCH_LOG_PREFIX,
            unresolved,
            sample,
        )
    else:
        logger.info("%s finished patch routing successfully; all gates covered.", PATCH_LOG_PREFIX)

    return tiling_steps

def _rebuild_schedule_for_layout(
    oldArrangementArr: np.ndarray,
    wiseArch: QCCDWiseArch,
    target_layout: np.ndarray,
    subgridsize: Tuple[int, int, int],
    base_pmax_in: int = None,
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
        
        # Start tiling from (0, 0) only; no offset tilings needed for this layout-only step.
        patch_regions = _generate_patch_regions(n_rows, n_cols, patch_h, patch_w, 0, 0)

        mismatch_before = int(np.count_nonzero(current_layout != target_layout))

        # Build ion -> position map from the current layout
        ion_positions: Dict[int, Tuple[int, int]] = {
            int(current_layout[r, c]): (r, c)
            for r in range(n_rows)
            for c in range(n_cols)
        }

        layouts_after_local: List[np.ndarray] = [np.array(current_layout, copy=True)]
        patch_schedules_this_tiling: List[List[List[Dict[str, Any]]]] = []
        total_pins = 0

        logger.info(
            "%s schedule-only rebuild: starting cycle %d",
            PATCH_LOG_PREFIX,
            cycle_idx,
        )

        for region in patch_regions:
            r0, c0, r1, c1 = region
            patch_grid = np.array(
                layouts_after_local[0][r0:r1, c0:c1], dtype=int, copy=True
            )

            # Build BT pins that force as many ions as possible in this patch
            # towards their target positions, subject to the uniqueness
            # constraint on (start_row_local, target_col_local).
            BT_patch: List[Dict[int, Tuple[int, int]]] = [dict()]
            bt_map = BT_patch[0]
            used_keys: Dict[Tuple[int, int], int] = {}

            for dr in range(r1 - r0):
                for dc in range(c1 - c0):
                    ionidx_target = int(target_layout[r0 + dr, c0 + dc])
                    pos = ion_positions.get(ionidx_target)
                    if pos is None:
                        # Should not happen, but be defensive.
                        continue
                    start_row_global, _ = pos
                    start_row_local = start_row_global - r0
                    if not (0 <= start_row_local < (r1 - r0)):
                        # Ion currently outside this patch row range; skip in this cycle.
                        continue


                    _, start_col_global = pos
                    start_col_local = start_col_global - c0
                    if not (0 <= start_col_local < (c1 - c0)):
                        # Ion currently outside this patch col range; skip in this cycle.
                        continue

                    key = (start_row_local, start_col_local)
                    existing_ion = used_keys.get(key)
                    if existing_ion is not None and existing_ion != ionidx_target:
                        # Another ion from the same start row already wants this column
                        # in this patch and has been pinned in this cycle; skip to
                        # avoid violating the SAT precondition.
                        continue

                    used_keys[key] = ionidx_target
                    bt_map[ionidx_target] = (dr, dc)

            if not bt_map:
                # No pins for this patch in this cycle; skip SAT for this patch.
                continue

            total_pins += len(bt_map)

            boundary_adjacent = {
                "top": r0 > 0,
                "bottom": r1 < n_rows,
                "left": c0 > 0,
                "right": c1 < n_cols,
            }
            cross_boundary_prefs: List[Dict[int, Set[str]]] = [dict()]

            logger.info(
                "%s rebuilding schedule-only patch (cycle %d) r[%d:%d] c[%d:%d] (no gates, BT pins=%d)",
                PATCH_LOG_PREFIX,
                cycle_idx,
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
                )
            except NoFeasibleLayoutError as exc:
                logger.warning(
                    "%s ERROR rebuilding schedule-only patch (cycle %d) r[%d:%d] c[%d:%d]: %s",
                    PATCH_LOG_PREFIX,
                    cycle_idx,
                    r0,
                    r1,
                    c0,
                    c1,
                    repr(exc),
                )
                raise

            # Update the global layout snapshot with this patch's result.
            layouts_after_local[0][r0:r1, c0:c1] = patch_layouts[0]
            patch_schedules_this_tiling.append(patch_schedule)

        if total_pins == 0 or not patch_schedules_this_tiling:
            logger.warning(
                "%s schedule-only rebuild made no progress in cycle %d "
                "(total_pins=%d); stopping early.",
                PATCH_LOG_PREFIX,
                cycle_idx,
                total_pins,
            )
            break

        # Merge patch schedules into a single global schedule for this cycle.
        merged_schedule_all_rounds = _merge_patch_schedules(patch_schedules_this_tiling, R)
        merged_schedule_round0 = merged_schedule_all_rounds[0] if merged_schedule_all_rounds else []

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

        # If we are no longer strictly reducing the mismatch, further cycles are
        # unlikely to help and will just spend more SAT time; stop early.
        if mismatch_after >= mismatch_before:
            logger.warning(
                "%s schedule-only rebuild: no further improvement in cycle %d (mismatch %d -> %d); stopping early.",
                PATCH_LOG_PREFIX,
                cycle_idx,
                mismatch_before,
                mismatch_after,
            )
            break

        prev_mismatch = mismatch_after
        patch_w = min(patch_w+1, n_cols)
        patch_h = min(patch_h+1, n_rows)

    if not np.array_equal(current_layout, target_layout):
        logger.warning(
            "%s schedule-only rebuild terminated without exactly reaching target layout.",
            PATCH_LOG_PREFIX,
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
    initial_placement: bool = False
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
    """
    newArrangement: Dict[ManipulationTrap, List[Ion]] = {
        trap: [] for trap in arch._manipulationTraps
    }

    for d in range(wiseArch.n):
        for c in range(wiseArch.m * wiseArch.k):
            ionidx = int(layout_after[d][c])
            newArrangementArr[d][c] = ionidx
            # Note: trap membership is based on the *old* arrangement
            trap = arch.ions[int(oldArrangementArr[d][c])].parent
            newArrangement[trap].append(arch.ions[ionidx])

    reconfig = GlobalReconfigurations.physicalOperation(
        newArrangement, wiseArch, oldArrangementArr, newArrangementArr, schedule=schedule, initial_placement=initial_placement
    )
    allOps.append(reconfig)
    reconfig.run()
    arch.refreshGraph()

    return newArrangementArr.copy()


def ionRoutingWISEArch(
    arch: QCCDArch,
    wiseArch: QCCDWiseArch,
    operations: Sequence[QubitOperation],
    lookahead: int = 2,
    subgridsize: Tuple[int, int, int] = (6, 4, 1), 
    base_pmax_in: int = None,
    toMoveOps: Sequence[Sequence[TwoQubitMSGate]]= None
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
    # 1) Build parallel rounds of two-qubit MS gates (parallelPairs/toMoves)
    # ------------------------------------------------------------------
    parallelismAllowed = wiseArch.m * wiseArch.n
    parallelPairs: List[List[Tuple[int, int]]] = []
    toMoves: List[List[TwoQubitMSGate]] = []


    if toMoveOps is None:
        idx = 0
        _opstogothrough = list(operations).copy()
        while _opstogothrough:
            # First, greedily take as many disjoint 1-qubit ops as possible
            while True:
                toRemove: List[Operation] = []
                ionsInvolved: Set[Ion] = set()

                for op in _opstogothrough:
                    trap = op.getTrapForIons()
                    if ionsInvolved.isdisjoint(op.ions) and len(op.ions) == 1:
                        toRemove.append(op)
                    ionsInvolved = ionsInvolved.union(op.ions)

                for g in toRemove:
                    _opstogothrough.remove(g)

                if len(toRemove) == 0:
                    break

            # Then, form one round of disjoint 2-qubit MS gates
            toRemove = []
            ionsAdded: Set[int] = set()
            for op in _opstogothrough:
                if len(op.ions) == 2 and len(toRemove) < parallelismAllowed:
                    ion1, ion2 = op.ions
                    ancilla, data = sorted(
                        (ion1, ion2), key=lambda ion: ion.label[0] == "D"
                    )
                    if (ancilla.idx in ionsAdded) or (data.idx in ionsAdded):
                        continue
                    toRemove.append(op)
                    if idx == len(parallelPairs):
                        parallelPairs.append([])
                    parallelPairs[idx].append((ancilla.idx, data.idx))
                    if idx == len(toMoves):
                        toMoves.append([])
                    toMoves[idx].append(op)

                    ionsAdded.add(ancilla.idx)
                    ionsAdded.add(data.idx)
                else:
                    ionsAdded.add(op.ions[0].idx)

            for g in toRemove:
                _opstogothrough.remove(g)

            idx += int(len(toRemove) > 0)

    else:
        for toMove in toMoveOps:
            toMoves.append(list(toMove))
            parallelPairs.append([])
            for op in toMove:
                ion1, ion2 = op.ions
                ancilla, data = sorted(
                    (ion1, ion2), key=lambda ion: ion.label[0] == "D"
                )
                parallelPairs[-1].append((ancilla.idx, data.idx))


    # ------------------------------------------------------------------
    # 2) Encode initial ion positions into oldArrangementArr
    # ------------------------------------------------------------------
    oldArrangementArr = np.array(
        [[0 for _ in range(wiseArch.m * wiseArch.k)] for _ in range(wiseArch.n)],
        dtype=int,
    )
    newArrangementArr = np.array(
        [[0 for _ in range(wiseArch.m * wiseArch.k)] for _ in range(wiseArch.n)],
        dtype=int,
    )

    ionsSorted = sorted(
        list(arch.ions.values()), key=lambda ion: ion.pos[0] + 100 * ion.pos[1]
    )
    for i, ion in enumerate(ionsSorted):
        r = i // (wiseArch.m * wiseArch.k)
        c = i % (wiseArch.m * wiseArch.k)
        oldArrangementArr[r][c] = ion.idx

    active_ions = [ion.idx for ion in ionsSorted if not isinstance(ion, SpectatorIon)]

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

    idx = 0
    tiling_steps_meta: List[Tuple[int, int, List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]]]] = []
    tiling_steps: List[List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]]] = []
    tiling_step: List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]] = []
    reconfigTime = 0.0
    hadMultipleTilingSteps: bool = False
    BTs: Optional[List[Dict[Tuple[int, int], Tuple[Dict[int, Tuple[int, int]], List[Tuple[int, int]]]]]] = None
    # Cache for repeated routing blocks: map a canonical key for a window of
    # parallel MS rounds (P_arr) to a mapping from offset within the block to its tiling metadata.
    block_cache: Dict[
        Tuple[Tuple[Tuple[int, int], ...], ...],
        Dict[int, List[Tuple[int, int, List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]]]]]
    ] = {}
    # Whether the currently active tiling plan (tiling_steps / tiling_step)
    # was obtained from the cache. If True, we will replay layouts with
    # schedule=None so GlobalReconfigurations can treat them as repeats.
    tiling_steps_from_cache: bool = False
    # When reusing a cached block, we only want to call
    # _apply_layout_as_reconfiguration with schedule=None for the *first*
    # reconfiguration in that block, because the initial layout may differ.
    # Subsequent steps within the same block occurrence should reuse the
    # saved schedules.
    pending_first_cached_reconfig: bool = False
    blk_idx: int = 0
    blk_end: int = 0
    recheck_cache: bool =True
    BLOCK_LEN = 4
    cache_for_block = {}

    # ------------------------------------------------------------------
    # 4) Execute operations, routing between parallel MS rounds as needed
    # ------------------------------------------------------------------
    while operationsLeft:
        logger.info(
            "%s main loop: idx=%d, remaining_operations=%d, remaining_MS_rounds=%d",
            PATCH_LOG_PREFIX,
            idx,
            len(operationsLeft),
            max(0, len(toMoves) - idx),
        )

        if len(toMoves) > idx:
            if (not hadMultipleTilingSteps) and tiling_steps and tiling_steps[0]:
                # Reuse the pre-computed tiling plan from the previous iteration.
                # The flag `tiling_steps_from_cache` remains whatever it was when
                # this tiling plan was first constructed (fresh or cached).
                pass
            else:
                # 4a) Between MS rounds: re-route using next lookahead window of pairs.
                # The routing horizon is controlled by `lookahead`, but the caching
                # of routing results (block_cache) is based on a fixed-size block
                # defined purely over `parallelPairs`, independent of lookahead.
                window_start = idx
                window_end = min(len(parallelPairs), lookahead + idx)
                logger.info(
                    "%s computing routing window: idx=%d, rounds=[%d,%d), lookahead=%d, window_len=%d",
                    PATCH_LOG_PREFIX,
                    idx,
                    window_start,
                    window_end,
                    lookahead,
                    window_end - window_start,
                )
                P_arr = parallelPairs[window_start:window_end].copy()

                if recheck_cache:
                    # Build a canonical, hashable key for a block of MS rounds using
                    # only `parallelPairs`, independent of the lookahead window size.
                    # We use a fixed block length (e.g. 4 rounds) so that repeated
                    # 4-round patterns like parallelPairs[0:4], [9:13], ... share
                    # the same cache entry even if lookahead != 4.
                    new_blk_end = min(len(parallelPairs), window_start + BLOCK_LEN)-window_start
                    new_block_key_rounds = parallelPairs[window_start:new_blk_end+window_start]
                    new_block_key: Tuple[Tuple[Tuple[int, int], ...], ...] = tuple(
                        tuple(sorted(round_pairs)) for round_pairs in new_block_key_rounds
                    )

                    # For a given block (block_key), we may call _patch_and_route multiple
                    # times with different starting rounds inside that block, especially
                    # when lookahead is small (e.g. lookahead == 1). We therefore cache
                    # tiling metadata *per offset* inside the block, so that for a
                    # repeated block pattern we can reuse the appropriate tiling plan
                    # for each round position within that block.
                    new_cache_for_block = block_cache.setdefault(new_block_key, {})
                    logger.info(
                        "%s rechecked cache at idx=%d (rounds=[%d,%d)); old_blk_idx=%d, old_blk_end=%d, new_blk_end=%d ,cache_for_block length=%d, new_cache_for_block length=%d",
                        PATCH_LOG_PREFIX,
                        idx,
                        window_start,
                        window_end,
                        blk_idx,
                        blk_end,
                        new_blk_end,
                        len(cache_for_block),
                        len(new_cache_for_block)
                    )
                    if len(new_cache_for_block)>0:
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
                        # new_cache_for_block is not full
                        # current cache_for_block is filling up
                        tiling_steps_from_cache = False
               

                if tiling_steps_from_cache:
                    # We have solved this block *at this offset* before; reuse it.
                    tiling_steps_meta = deepcopy(cache_for_block[blk_idx])
                    pending_first_cached_reconfig = (blk_idx==0)
                    logger.info(
                        "%s reusing cached routing block at idx=%d (rounds=[%d,%d)); blk_idx=%d, cached_tilings=%d",
                        PATCH_LOG_PREFIX,
                        idx,
                        window_start,
                        window_end,
                        blk_idx,
                        len(tiling_steps_meta),
                    )
                else:
                    # New (block, offset) combination: call the expensive patch-based
                    # router and cache its result under this offset so that repeated
                    # instances of the same block pattern reuse it.
                    tiling_steps_meta = _patch_and_route(
                        oldArrangementArr,
                        wiseArch,
                        P_arr,
                        subgridsize,
                        active_ions=active_ions,
                        ignore_initial_reconfig=False,
                        base_pmax_in=base_pmax_in,
                        BTs=BTs,
                    )
                    if len(cache_for_block)<BLOCK_LEN:
                        cache_for_block[blk_idx] = deepcopy(tiling_steps_meta)
                    else: 
                        logger.warn(
                            "%s cache_for_block is already full! idx=%d cache_for_block size=%d",
                            PATCH_LOG_PREFIX,
                            idx,
                            len(cache_for_block)
                        )
                    logger.info(
                        "%s patch routing window idx=%d produced %d tiling steps (meta entries); block_offset=%s, cache_for_block size=%d",
                        PATCH_LOG_PREFIX,
                        idx,
                        len(tiling_steps_meta),
                        str(blk_idx),
                        len(cache_for_block)
                    )

                # Strip metadata for execution ordering; keep only per-tiling round lists.
                tiling_steps = [snapshot for (_cy, _ti, snapshot) in tiling_steps_meta]
                hadMultipleTilingSteps = len(tiling_steps) > 1

                # Build new BTs (pins) from this returned long-horizon plan for future windows.
                if tiling_steps_meta:
                    active_set = set(active_ions)
                    R_local = len(P_arr)
                    # For rounds r >= 1, map (cycle_idx, tiling_idx) -> (bt_map, solved_pairs_r)
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

                    num_bt_rounds = len(BTs)
                    num_bt_windows = sum(len(round_dict) for round_dict in BTs)
                    total_bt_pins = sum(
                        len(bt_map)
                        for round_dict in BTs
                        for (bt_map, _pairs_r) in round_dict.values()
                    )
                    total_bt_pairs = sum(
                        len(_pairs_r)
                        for round_dict in BTs
                        for (_bt_map, _pairs_r) in round_dict.values()
                    )
                    logger.info(
                        "%s built BTs for window idx=%d: rounds=%d, windows=%d, pins=%d, pinned_pairs=%d",
                        PATCH_LOG_PREFIX,
                        idx,
                        num_bt_rounds,
                        num_bt_windows,
                        total_bt_pins,
                        total_bt_pairs,
                    )
                else:
                    BTs = None
            if not tiling_step:
                tiling_step = [t.pop(0) for t in tiling_steps]
                logger.info(
                    "%s idx=%d: initialised tiling_step list with %d entries (from_cache=%s)",
                    PATCH_LOG_PREFIX,
                    idx,
                    len(tiling_step),
                    tiling_steps_from_cache,
                )
            layout_after, schedule, solved_pairs = tiling_step.pop(0)
            if tiling_steps_from_cache and pending_first_cached_reconfig:
                sched_arg = None
                pending_first_cached_reconfig = False
                logger.info(
                    "%s idx=%d: applying cached reconfiguration step; schedule_passes=%d, solved_pairs=%d, remaining_tiling_steps=%d, using_cached_schedule=%s",
                    PATCH_LOG_PREFIX,
                    idx,
                    len(sched_arg) if (sched_arg is not None) else 0,
                    len(solved_pairs),
                    len(tiling_step),
                    tiling_steps_from_cache
                )
                oldArrangementArr = _apply_layout_as_reconfiguration(
                    arch,
                    wiseArch,
                    oldArrangementArr,
                    newArrangementArr,
                    layout_after,
                    allOps,
                    sched_arg,
                    initial_placement=(idx == 0),
                )
                barriers.append(len(allOps))
                last_reconfig_time = getattr(allOps[-1], "_reconfigTime", 0.0)
                reconfigTime += last_reconfig_time
                logger.info(
                    "%s idx=%d: reconfiguration applied; delta_reconfigTime=%.6f, total_reconfigTime=%.6f, total_ops=%d",
                    PATCH_LOG_PREFIX,
                    idx,
                    last_reconfig_time,
                    reconfigTime,
                    len(allOps),
                )
            else:
                sched_arg = schedule
                logger.info(
                    "%s idx=%d: applying reconfiguration step; schedule_passes=%d, solved_pairs=%d, remaining_tiling_steps=%d, using_cached_schedule=%s",
                    PATCH_LOG_PREFIX,
                    idx,
                    len(sched_arg) if (sched_arg is not None) else 0,
                    len(solved_pairs),
                    len(tiling_step),
                    tiling_steps_from_cache
                )
                oldArrangementArr = _apply_layout_as_reconfiguration(
                    arch,
                    wiseArch,
                    oldArrangementArr,
                    newArrangementArr,
                    layout_after,
                    allOps,
                    sched_arg,
                    initial_placement=(idx == 0),
                )
                barriers.append(len(allOps))
                last_reconfig_time = getattr(allOps[-1], "_reconfigTime", 0.0)
                reconfigTime += last_reconfig_time
                logger.info(
                    "%s idx=%d: reconfiguration applied; delta_reconfigTime=%.6f, total_reconfigTime=%.6f, total_ops=%d",
                    PATCH_LOG_PREFIX,
                    idx,
                    last_reconfig_time,
                    reconfigTime,
                    len(allOps),
                )

        # 4b) Run as many single-qubit operations as possible without routing
        single_qubit_executed = 0
        while True:
            toRemove: List[Operation] = []
            ionsInvolved: Set[Ion] = set()
            for op in operationsLeft:
                trap = op.getTrapForIons()
                if ionsInvolved.isdisjoint(op.ions) and trap and len(op.ions) == 1:
                    op.setTrap(trap)
                    toRemove.append(op)
                ionsInvolved = ionsInvolved.union(op.ions)

            for op in toRemove:
                op.run()
                allOps.append(op)
                operationsLeft.remove(op)
                single_qubit_executed += 1

            if len(toRemove) == 0:
                break

        if single_qubit_executed:
            logger.info(
                "%s idx=%d: executed %d single-qubit ops; remaining_operations=%d, total_ops=%d",
                PATCH_LOG_PREFIX,
                idx,
                single_qubit_executed,
                len(operationsLeft),
                len(allOps),
            )

        barriers.append(len(allOps))

        if not operationsLeft:
            break

        ms_gates_executed = 0
        starting_idx = idx
        while idx < len(toMoves):
            # 4c) Execute one parallel round of two-qubit MS gates
            for op in toMoves[idx]:
                if any(all((ion.idx in p) for ion in op.ions) for p in solved_pairs): 
                    trap = op.getTrapForIons()
                    op.setTrap(trap)
                    op.run()
                    allOps.append(op)
                    operationsLeft.remove(op)
                    ms_gates_executed += 1
            if not tiling_step:
                break
            if not any(t[2] for t in tiling_step):
                tiling_step = []
                break
            layout_after, schedule, solved_pairs = tiling_step.pop(0)
            oldArrangementArr = _apply_layout_as_reconfiguration(
                arch,
                wiseArch,
                oldArrangementArr,
                newArrangementArr,
                layout_after,
                allOps,
                schedule,
                initial_placement=False,
            )
            barriers.append(len(allOps))
            last_reconfig_time = getattr(allOps[-1], "_reconfigTime", 0.0)
            reconfigTime += last_reconfig_time
            logger.info(
                "%s idx=%d: reconfiguration applied after MS round; delta_reconfigTime=%.6f, total_reconfigTime=%.6f, total_ops=%d",
                PATCH_LOG_PREFIX,
                idx,
                last_reconfig_time,
                reconfigTime,
                len(allOps),
            )

        if ms_gates_executed:
            candidate_gates = len(toMoves[starting_idx]) if starting_idx < len(toMoves) else 0
            logger.info(
                "%s idx=%d: executed %d/%d two-qubit MS gates; remaining_operations=%d, total_ops=%d",
                PATCH_LOG_PREFIX,
                starting_idx,
                ms_gates_executed,
                candidate_gates,
                len(operationsLeft),
                len(allOps),
            )

        barriers.append(len(allOps))
        idx+=1
        blk_idx+=1
        recheck_cache = (blk_idx not in cache_for_block)
        if blk_idx == blk_end:
            cache_for_block = {}
            blk_idx = blk_end = 0

    return allOps, barriers, reconfigTime
