# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/layout_utils.py
"""
Layout utility functions for WISE routing.

This module provides helper functions for:
- Computing patch gating capacity
- Cross-boundary preference calculation
- Target layout computation from gate pairs
- Pre-SAT sanity checks for feasibility
- Heuristic reconfiguration fallbacks (Phases B/C/D)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from .config import wise_logger

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .sat_encoder import WISESATContext


__all__ = [
    "NoFeasibleLayoutError",
    "compute_patch_gating_capacity",
    "compute_cross_boundary_prefs",
    "compute_target_layout_from_pairs",
    "pre_sat_sanity_checks",
    "heuristic_phaseB_greedy_layout",
    "heuristic_odd_even_reconfig",
    # Physical reconfig (ported from old _runOddEvenReconfig)
    "run_reconfig_from_schedule",
    "rebuild_schedule_for_layout",
    # Schedule merging helpers (ported from old qccd_WISE_ion_route.py)
    "pass_has_swaps",
    "infer_pass_parity",
    "split_patch_round_into_HV",
    "merge_phase_passes",
    "merge_patch_round_schedules",
    "merge_patch_schedules",
]


class NoFeasibleLayoutError(RuntimeError):
    """Raised when no feasible layout can be found by the SAT solver."""

    pass


# =============================================================================
# Patch Gating Capacity & Cross-Boundary Preference Helpers
# =============================================================================


def compute_patch_gating_capacity(
    n: int,
    m: int,
    col_offset: int,
    capacity: int,
) -> int:
    """Compute the maximum number of disjoint gating zones in a patch.

    Parameters
    ----------
    n : int
        Patch height (rows).
    m : int
        Patch width (columns, local to patch).
    col_offset : int
        Global column index of patch's first column.
    capacity : int
        Global block width (ions per segment).

    Returns
    -------
    int
        Maximum number of disjoint gating positions in this patch.
    """
    if capacity <= 0 or m <= 0 or n <= 0:
        return 0

    first_block_idx = col_offset // capacity
    last_block_idx = (col_offset + m - 1) // capacity
    num_blocks = last_block_idx - first_block_idx + 1

    total_per_row = 0
    for b_local in range(num_blocks):
        b_global = first_block_idx + b_local
        global_start = b_global * capacity
        global_end = (b_global + 1) * capacity

        local_start = max(0, global_start - col_offset)
        local_end = min(m, global_end - col_offset)
        width = max(0, local_end - local_start)

        # each block contributes floor(width/2) gating positions per row
        total_per_row += width // 2

    return n * total_per_row


def compute_cross_boundary_prefs(
    region: Tuple[int, int, int, int],
    ion_positions: Mapping[int, Tuple[int, int]],
    pairs_per_round: List[List[Tuple[int, int]]],
) -> List[Dict[int, Set[str]]]:
    """Compute cross-boundary directional preferences for patch routing.

    For each round, record which boundary/boundaries an ion should approach
    based on gates with partners outside the region. Directions are one or
    more of {"top", "bottom", "left", "right"}.

    Parameters
    ----------
    region : Tuple[int, int, int, int]
        Patch region as (r0, c0, r1, c1).
    ion_positions : Mapping[int, Tuple[int, int]]
        Current positions of all ions.
    pairs_per_round : List[List[Tuple[int, int]]]
        Gate pairs per round.

    Returns
    -------
    List[Dict[int, Set[str]]]
        Per-round dict mapping ion ID to set of boundary directions.
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
                continue  # both inside or both outside

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


def compute_target_layout_from_pairs(
    A: np.ndarray,
    gate_pairs: List[Tuple[int, int]],
    capacity: int,
) -> np.ndarray:
    """Compute a target layout that positions paired ions in the same block.

    When SAT fails and we need to fall back to heuristic routing, we need
    a target layout T. This function computes a layout where ions that need
    to interact are placed in the same gating block.

    Parameters
    ----------
    A : np.ndarray
        Current layout (n × m).
    gate_pairs : List[Tuple[int, int]]
        Ion pairs that need to interact.
    capacity : int
        Block width (ions per segment).

    Returns
    -------
    np.ndarray
        Target layout where paired ions are adjacent in the same block.
    """
    n, m = A.shape
    T = np.array(A, dtype=int)  # Start from current layout

    if not gate_pairs:
        return T

    # Build position lookup from current layout
    ion_pos: Dict[int, Tuple[int, int]] = {}
    for r in range(n):
        for c in range(m):
            ion_pos[int(A[r, c])] = (r, c)

    # For each pair, try to position them in the same block
    for ion_a, ion_b in gate_pairs:
        if ion_a not in ion_pos or ion_b not in ion_pos:
            continue

        ra, ca = ion_pos[ion_a]
        rb, cb = ion_pos[ion_b]

        # Already in same row and block?
        if ra == rb and (ca // capacity) == (cb // capacity):
            continue

        # Try to move ion_b to same row as ion_a in an adjacent column
        block_a = ca // capacity
        block_start = block_a * capacity
        block_end = min(block_start + capacity, m)

        target_col = ca + 1 if ca + 1 < block_end else ca - 1
        if 0 <= target_col < m:
            # Find current position of ion_b in T
            tb_r, tb_c = -1, -1
            for r in range(n):
                for c in range(m):
                    if int(T[r, c]) == ion_b:
                        tb_r, tb_c = r, c
                        break
                if tb_r >= 0:
                    break

            # Swap ion_b to (ra, target_col)
            if tb_r >= 0 and (tb_r != ra or tb_c != target_col):
                other_ion = int(T[ra, target_col])
                T[ra, target_col] = ion_b
                T[tb_r, tb_c] = other_ion
                ion_pos[ion_b] = (ra, target_col)
                ion_pos[other_ion] = (tb_r, tb_c)

    return T


# =============================================================================
# Pre-SAT Sanity Checks
# =============================================================================


def pre_sat_sanity_checks(
    context: "WISESATContext",
    bt_soft_enabled: bool = False,
    capacity: int = 2,
) -> None:
    """Perform pre-SAT sanity checks to detect obviously infeasible instances.

    Parameters
    ----------
    context : WISESATContext
        Routing context with layout and target info.
    bt_soft_enabled : bool
        If True, log warnings instead of raising for soft-BT violations.
    capacity : int
        Block capacity (k).

    Raises
    ------
    NoFeasibleLayoutError
        If an obviously infeasible condition is detected and bt_soft_enabled
        is False.
    """
    A_in = context.initial_layout
    n, m = context.n_rows, context.n_cols
    R = context.num_rounds
    BT = context.target_positions
    P_arr = context.gate_pairs

    # Build ion position lookup
    ions_all = set(int(x) for x in A_in.flatten())
    row_of = {int(A_in[r, c]): r for r in range(n) for c in range(m)}

    # (a) No two ions pinned to the same (d, c) in a round
    for r, bt in enumerate(BT):
        seen: Dict[Tuple[int, int], int] = {}
        for i, (d, c) in bt.items():
            if i not in ions_all:
                continue
            key = (d, c)
            if key in seen:
                msg = (
                    f"BT[{r}] pins ions {seen[key]} and {i} "
                    f"to the same cell (d={d}, c={c})."
                )
                if bt_soft_enabled:
                    wise_logger.warning("UNSAT-soft: %s (will leave to Max-SAT)", msg)
                    continue
                raise NoFeasibleLayoutError(f"UNSAT: {msg}")
            seen[key] = i

    # (b) Pair vs BT conflicts
    for r in range(R):
        pairs = P_arr[r] if r < len(P_arr) else []
        bt = BT[r] if r < len(BT) else {}
        for i1, i2 in pairs:
            if i1 not in ions_all or i2 not in ions_all:
                continue
            if i1 in bt and i2 in bt:
                d1, c1 = bt[i1]
                d2, c2 = bt[i2]
                if d1 != d2:
                    msg = f"round {r} pair {(i1, i2)} BT rows differ: {d1} vs {d2}."
                    if bt_soft_enabled:
                        wise_logger.warning("UNSAT-soft: %s", msg)
                        continue
                    raise NoFeasibleLayoutError(f"UNSAT: {msg}")
                if (c1 // capacity) != (c2 // capacity):
                    msg = (
                        f"round {r} pair {(i1, i2)} BT blocks differ: "
                        f"{c1}//{capacity} vs {c2}//{capacity}."
                    )
                    if bt_soft_enabled:
                        wise_logger.warning("UNSAT-soft: %s", msg)
                        continue
                    raise NoFeasibleLayoutError(f"UNSAT: {msg}")

    # (c) Round-0: same source row & same target column
    if len(BT) >= 1:
        buckets: Dict[Tuple[int, int], List[int]] = {}
        for i, (d0, c0) in BT[0].items():
            if i not in ions_all:
                continue
            sr = row_of.get(i, -1)
            if sr < 0:
                continue
            buckets.setdefault((sr, c0), []).append(i)
        bad = {key: vs for key, vs in buckets.items() if len(vs) > 1}
        if bad:
            msg = f"round 0 has reserved ions from same row targeting same column: {bad}"
            if bt_soft_enabled:
                wise_logger.warning("UNSAT-soft: %s", msg)
            else:
                raise NoFeasibleLayoutError(f"UNSAT: {msg}")

    # (d) Column oversubscription
    for r, bt in enumerate(BT):
        col_counts: Dict[int, int] = {}
        for i, (_, c) in bt.items():
            if i not in ions_all:
                continue
            col_counts[c] = col_counts.get(c, 0) + 1
        bad_cols = {c: cnt for c, cnt in col_counts.items() if cnt > n}
        if bad_cols:
            msg = f"BT[{r}] pins {bad_cols} ions to one column, exceeds n={n}."
            if bt_soft_enabled:
                wise_logger.warning("UNSAT-soft: %s", msg)
            else:
                raise NoFeasibleLayoutError(f"UNSAT: {msg}")


# =============================================================================
# Heuristic Reconfiguration Fallback (Phases B/C/D)
# =============================================================================


def heuristic_phaseB_greedy_layout(
    A_in: np.ndarray,
    T_in: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """Phase-B layout helper: greedy row permutation via DFS matching.

    Construct a row-wise permutation B of A_in such that each column
    of B contains ions with all distinct destination rows (according to T_in).

    Returns (max_horizontal_displacement, B).
    """
    A = np.asarray(A_in, dtype=int)
    T = np.asarray(T_in, dtype=int)
    n, m = A.shape
    if T.shape != (n, m):
        raise ValueError("phaseB_greedy_layout: A and T must have same shape")

    # Destination row for every ion (from T)
    ion_to_dest_row: Dict[int, int] = {}
    for r in range(n):
        for c in range(m):
            ion_to_dest_row[int(T[r, c])] = r

    if set(int(x) for x in A.flatten()) != set(int(x) for x in T.flatten()):
        raise ValueError("phaseB_greedy_layout: A and T must contain the same ion IDs")

    # counts[r, d] = number of ions in row r of A whose dest is row d
    counts = np.zeros((n, n), dtype=int)
    ions_per_row_dest: List[List[List[int]]] = [[[] for _ in range(n)] for _ in range(n)]

    invpos: List[Dict[int, int]] = []
    for r in range(n):
        row_map: Dict[int, int] = {}
        for c in range(m):
            row_map[int(A[r, c])] = c
        invpos.append(row_map)

    for r in range(n):
        for c in range(m):
            ion = int(A[r, c])
            d = ion_to_dest_row[ion]
            counts[r, d] += 1
            ions_per_row_dest[r][d].append(ion)

    # DFS-based perfect matching from count matrix
    def _perfect_matching_from_counts(counts_mat: np.ndarray) -> List[int]:
        nn = counts_mat.shape[0]
        neighbors: List[List[int]] = [
            [dd for dd in range(nn) if counts_mat[rr, dd] > 0] for rr in range(nn)
        ]
        match_to_row = [-1] * nn

        def dfs(r: int, seen: List[bool]) -> bool:
            for d in neighbors[r]:
                if seen[d]:
                    continue
                seen[d] = True
                if match_to_row[d] == -1 or dfs(match_to_row[d], seen):
                    match_to_row[d] = r
                    return True
            return False

        for r in range(nn):
            seen = [False] * nn
            if not dfs(r, seen):
                raise RuntimeError("no perfect matching for current counts")

        match_row_to_dest = [-1] * nn
        for d, r in enumerate(match_to_row):
            if r != -1:
                match_row_to_dest[r] = d
        return match_row_to_dest

    desired = np.zeros_like(A)
    max_disp = 0

    for c in range(m):
        match = _perfect_matching_from_counts(counts)
        for r in range(n):
            d = int(match[r])
            bucket = ions_per_row_dest[r][d]
            if not bucket:
                raise RuntimeError(
                    f"phaseB_greedy_layout: empty bucket for (row={r}, dest={d})"
                )
            ion = bucket.pop()
            counts[r, d] -= 1
            desired[r, c] = ion
            j0 = invpos[r][ion]
            disp = abs(j0 - c)
            if disp > max_disp:
                max_disp = disp

    return max_disp, desired


def heuristic_odd_even_reconfig(
    A_in: np.ndarray,
    T_in: np.ndarray,
    k: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[int, float], float]:
    """Heuristic odd-even reconfiguration: Phases B/C/D.

    Fallback path when SAT schedule is unavailable.

    Parameters
    ----------
    A_in : np.ndarray
        Current layout (n×m).
    T_in : np.ndarray
        Target layout (n×m).
    k : int
        Column stride for junction batching (ions_per_segment).

    Returns
    -------
    schedule : List[Dict[str, Any]]
        List of pass dicts with "phase" and swap info.
    heating_rates : Dict[int, float]
        Per-ion motional quanta deposited.
    time_elapsed : float
        Total reconfiguration time in seconds.
    """
    # Import transport primitives for heating computation
    from qectostim.experiments.hardware_simulation.trapped_ion.transport import (
        Move as _Move,
        JunctionCrossing as _JC,
    )

    A = np.array(A_in, dtype=int)
    T = np.array(T_in, dtype=int)
    n, m = A.shape

    heating_rates: Dict[int, float] = {}
    for ion_id in set(int(x) for x in A.flatten()):
        heating_rates[ion_id] = 0.0

    row_swap_time = _Move.MOVING_TIME + 80e-6 + 42e-6 + 80e-6 + _Move.MOVING_TIME
    row_swap_heating = (
        _Move.MOVING_TIME * _Move.HEATING_RATE
        + 80e-6 * 6
        + 42e-6 * 6
        + 80e-6 * 6
        + _Move.MOVING_TIME * _Move.HEATING_RATE
    )
    col_swap_time = 2 * _JC.CROSSING_TIME + (4 * _JC.CROSSING_TIME + _Move.MOVING_TIME) * 2
    col_swap_heating = 6 * _JC.CROSSING_TIME * _JC.CROSSING_HEATING + _Move.MOVING_TIME * _Move.HEATING_RATE

    schedule: List[Dict[str, Any]] = []
    time_elapsed = 0.0

    # Phase A: parallel split
    time_elapsed += 80e-6
    for idx in heating_rates:
        heating_rates[idx] += 6 * 80e-6

    # Ion destination maps
    ion_to_dest_row: Dict[int, int] = {}
    ion_to_dest_col: Dict[int, int] = {}
    for r in range(n):
        for c in range(m):
            ion = int(T[r, c])
            ion_to_dest_row[ion] = r
            ion_to_dest_col[ion] = c

    # Helper: row pass by rank
    def row_pass_by_rank(
        even_phase: bool, row_rank: List[Dict[int, int]]
    ) -> Tuple[bool, Dict[str, Any]]:
        start = 0 if even_phase else 1
        swaps: List[Tuple[int, int]] = []
        for r in range(n):
            rank = row_rank[r]
            for c in range(start, m - 1, 2):
                a = int(A[r, c])
                b = int(A[r, c + 1])
                if rank[a] > rank[b]:
                    A[r, c], A[r, c + 1] = b, a
                    heating_rates[a] = heating_rates.get(a, 0.0) + row_swap_heating
                    heating_rates[b] = heating_rates.get(b, 0.0) + row_swap_heating
                    swaps.append((r, c))
        did = len(swaps) > 0
        info = {"phase": "H", "h_swaps": swaps, "v_swaps": []}
        return did, info

    # Helper: col bucket pass
    def col_bucket_pass(even_phase: bool, bucket_mod: int) -> Tuple[bool, Dict[str, Any]]:
        start = 0 if even_phase else 1
        swaps: List[Tuple[int, int]] = []
        for c in range(bucket_mod, m, k):
            for r in range(start, n - 1, 2):
                a = int(A[r, c])
                b = int(A[r + 1, c])
                if ion_to_dest_row.get(a, 0) > ion_to_dest_row.get(b, 0):
                    A[r, c], A[r + 1, c] = b, a
                    heating_rates[a] = heating_rates.get(a, 0.0) + col_swap_heating
                    heating_rates[b] = heating_rates.get(b, 0.0) + col_swap_heating
                    swaps.append((r, c))
        did = len(swaps) > 0
        info = {"phase": "V", "h_swaps": [], "v_swaps": swaps}
        return did, info

    # Phase B: greedy target layout + row sort
    _, B = heuristic_phaseB_greedy_layout(A, T)

    row_rank_phaseB: List[Dict[int, int]] = []
    for r in range(n):
        row_rank_phaseB.append({int(ion): idx for idx, ion in enumerate(B[r])})

    for _ in range(m):
        did_even, info_even = row_pass_by_rank(True, row_rank_phaseB)
        did_odd, info_odd = row_pass_by_rank(False, row_rank_phaseB)
        if did_even:
            time_elapsed += row_swap_time
            schedule.append(info_even)
        if did_odd:
            time_elapsed += row_swap_time
            schedule.append(info_odd)

    # Phase C: vertical odd-even with k-way parallel buckets
    for t in range(k):
        for _ in range(n):
            did_even, info_even = col_bucket_pass(True, t)
            did_odd, info_odd = col_bucket_pass(False, t)
            if did_even:
                time_elapsed += col_swap_time
                schedule.append(info_even)
            if did_odd:
                time_elapsed += col_swap_time
                schedule.append(info_odd)
        if t < k - 1:
            time_elapsed += k * row_swap_time
            for idx in heating_rates:
                heating_rates[idx] += row_swap_heating

    # Phase D: final row-wise odd-even to exact target order
    row_rank_final: List[Dict[int, int]] = []
    for r in range(n):
        row_rank_final.append({int(ion): idx for idx, ion in enumerate(T[r, :])})

    for _ in range(m):
        did_even, info_even = row_pass_by_rank(True, row_rank_final)
        did_odd, info_odd = row_pass_by_rank(False, row_rank_final)
        if did_even:
            time_elapsed += row_swap_time
            schedule.append(info_even)
        if did_odd:
            time_elapsed += row_swap_time
            schedule.append(info_odd)

    return schedule, heating_rates, time_elapsed


# =============================================================================
# Schedule Merging Helpers (ported from old qccd_WISE_ion_route.py)
# =============================================================================


def pass_has_swaps(pass_info: Dict[str, Any]) -> bool:
    """Check if a pass contains any swaps.
    
    Parameters
    ----------
    pass_info : Dict[str, Any]
        Pass dictionary with "phase" and swap info.
    
    Returns
    -------
    bool
        True if the pass has horizontal or vertical swaps.
    """
    if pass_info.get("phase") == "H":
        return bool(pass_info.get("h_swaps"))
    return bool(pass_info.get("v_swaps"))


def infer_pass_parity(pass_info: Dict[str, Any]) -> Optional[int]:
    """Determine whether this pass belongs to the "even" or "odd" family.
    
    For H-phase we inspect the column index of the first horizontal swap;
    for V-phase we inspect the row index of the first vertical swap.
    
    Parameters
    ----------
    pass_info : Dict[str, Any]
        Pass dictionary with "phase" and swap info.
    
    Returns
    -------
    Optional[int]
        0 for even, 1 for odd, None if no swaps exist.
    """
    phase = pass_info.get("phase", "")
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


def split_patch_round_into_HV(
    patch_round_schedules: List[List[Dict[str, Any]]]
) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    """Split per-patch schedules into H and V phase segments.
    
    For each patch's schedule, separate the H-phase passes from V-phase
    passes, preserving the relative order within each phase.
    
    Parameters
    ----------
    patch_round_schedules : List[List[Dict[str, Any]]]
        List of schedules, one per patch. Each schedule is a list of passes.
    
    Returns
    -------
    per_patch_h : List[List[Dict[str, Any]]]
        H-phase passes for each patch.
    per_patch_v : List[List[Dict[str, Any]]]
        V-phase passes for each patch.
    """
    per_patch_h: List[List[Dict[str, Any]]] = []
    per_patch_v: List[List[Dict[str, Any]]] = []

    for sched in patch_round_schedules:
        h_seg: List[Dict[str, Any]] = []
        v_seg: List[Dict[str, Any]] = []
        seen_v = False

        for pass_info in sched:
            if not pass_has_swaps(pass_info):
                continue
            phase = pass_info.get("phase", "")
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


def merge_phase_passes(
    per_patch_passes: List[List[Dict[str, Any]]],
    phase_label: str,
) -> List[Dict[str, Any]]:
    """Merge passes from multiple patches into a global schedule.
    
    Combines passes with compatible parity from different patches into
    single passes for parallel execution. Passes only combine when they
    share phase and comparator parity.
    
    Parameters
    ----------
    per_patch_passes : List[List[Dict[str, Any]]]
        H or V passes for each patch.
    phase_label : str
        "H" or "V" to indicate the phase type.
    
    Returns
    -------
    List[Dict[str, Any]]
        Merged list of passes.
    """
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
        base_parity = infer_pass_parity(base_pass)

        new_pass: Dict[str, Any] = {
            "phase": phase_label,
            "h_swaps": [],
            "v_swaps": [],
        }

        def _compatible(pinfo: Dict[str, Any]) -> bool:
            if pinfo.get("phase") != phase_label:
                return False
            p_parity = infer_pass_parity(pinfo)
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


def merge_patch_round_schedules(
    patch_round_schedules: List[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Merge per-patch schedules for a single round into a global schedule.
    
    Ensures all H passes occur before any V pass, and passes only combine
    when they share phase and comparator parity.
    
    Parameters
    ----------
    patch_round_schedules : List[List[Dict[str, Any]]]
        List of schedules, one per patch.
    
    Returns
    -------
    List[Dict[str, Any]]
        Merged global schedule for this round.
    """
    if not patch_round_schedules:
        return []

    per_patch_h, per_patch_v = split_patch_round_into_HV(patch_round_schedules)
    merged_h = merge_phase_passes(per_patch_h, phase_label="H")
    merged_v = merge_phase_passes(per_patch_v, phase_label="V")
    merged = merged_h + merged_v

    for pass_info in merged:
        if pass_info.get("h_swaps") and pass_info.get("v_swaps"):
            raise AssertionError(
                "Merged pass contains both horizontal and vertical swaps"
            )

    return merged


def merge_patch_schedules(
    patch_schedules: List[List[List[Dict[str, Any]]]],
    num_rounds: int,
) -> List[List[Dict[str, Any]]]:
    """Merge per-patch schedules across all rounds.
    
    Given a list of patch schedules (one schedule per patch, each of which
    is a [round][pass] structure), merge them into a single schedule that
    respects odd-even and H/V constraints.
    
    Parameters
    ----------
    patch_schedules : List[List[List[Dict[str, Any]]]]
        List of schedules, one per patch. Each is [round][pass].
    num_rounds : int
        Total number of rounds.
    
    Returns
    -------
    List[List[Dict[str, Any]]]
        Merged schedule indexed by round.
    """
    merged: List[List[Dict[str, Any]]] = [[] for _ in range(num_rounds)]
    if not patch_schedules:
        return merged

    for r in range(num_rounds):
        round_scheds = [sched[r] for sched in patch_schedules if r < len(sched)]
        merged[r] = merge_patch_round_schedules(round_scheds)

    return merged

# =============================================================================
# SAT-schedule-driven odd-even reconfiguration (port of _runOddEvenReconfig)
# =============================================================================


def run_reconfig_from_schedule(
    old_layout: np.ndarray,
    new_layout: np.ndarray,
    sat_schedule: Optional[List[Dict[str, Any]]],
    k: int = 2,
    spectator_ions: Optional[Set[int]] = None,
    ignore_spectators: bool = False,
) -> Tuple[Dict[int, float], float]:
    """Execute a reconfiguration from old_layout to new_layout.

    When *sat_schedule* is not ``None``, the schedule is applied directly
    (faithful port of old ``_runOddEvenReconfig`` SAT-schedule path).
    When it is ``None``, the heuristic odd-even (Phases B/C/D) fallback
    is used.

    Parameters
    ----------
    old_layout : np.ndarray
        Current layout (n×m, int).
    new_layout : np.ndarray
        Target layout (n×m, int).
    sat_schedule : list or None
        List of pass dicts ``{"phase": "H"|"V", "h_swaps": [...], "v_swaps": [...]}``.
        Pass ``None`` to use heuristic fallback.
    k : int
        Column stride for junction batching (ions_per_segment).
    spectator_ions : set or None
        Ion indices that are spectators (optionally skip their swaps).
    ignore_spectators : bool
        If True AND both ions in a swap are spectators, skip that swap.

    Returns
    -------
    heating_rates : Dict[int, float]
        Per-ion motional quanta deposited.
    time_elapsed : float
        Total reconfiguration time in seconds.
    """
    from qectostim.experiments.hardware_simulation.trapped_ion.transport import (
        Move as _Move,
        Merge as _Merge,
        CrystalRotation as _CRot,
        Split as _Split,
        JunctionCrossing as _JC,
    )

    A = np.array(old_layout, dtype=int)
    T = np.array(new_layout, dtype=int)
    n, m = A.shape
    specs = spectator_ions or set()

    # Initialise heating rates for all ions in A
    heating_rates: Dict[int, float] = {}
    for ion_id in set(int(x) for x in A.flatten()):
        heating_rates[ion_id] = 0.0

    row_swap_time = (
        _Move.MOVING_TIME + _Merge.MERGING_TIME
        + _CRot.ROTATION_TIME + _Split.SPLITTING_TIME + _Move.MOVING_TIME
    )
    row_swap_heating = (
        _Move.MOVING_TIME * _Move.HEATING_RATE
        + _Merge.MERGING_TIME * _Merge.HEATING_RATE
        + _CRot.ROTATION_TIME * _CRot.ROTATION_HEATING
        + _Split.SPLITTING_TIME * _Split.HEATING_RATE
        + _Move.MOVING_TIME * _Move.HEATING_RATE
    )
    col_swap_time = (
        2 * _JC.CROSSING_TIME
        + (4 * _JC.CROSSING_TIME + _Move.MOVING_TIME) * 2
    )
    col_swap_heating = (
        6 * _JC.CROSSING_TIME * _JC.CROSSING_HEATING
        + _Move.MOVING_TIME * _Move.HEATING_RATE
    )

    time_elapsed = 0.0

    # Phase A: parallel split
    time_elapsed += _Split.SPLITTING_TIME
    for idx in heating_rates:
        heating_rates[idx] += _Split.HEATING_RATE * _Split.SPLITTING_TIME

    if sat_schedule is not None:
        # ---- SAT-schedule-driven path (faithful port) ----
        for info in sat_schedule:
            phase = info.get("phase", "H")
            h_swaps = info.get("h_swaps", [])
            v_swaps = info.get("v_swaps", [])
            did_any = False

            if phase == "H":
                for (r, c) in h_swaps:
                    a = int(A[r, c])
                    b = int(A[r, c + 1])
                    if ignore_spectators and (a in specs and b in specs):
                        continue
                    A[r, c], A[r, c + 1] = b, a
                    heating_rates[a] = heating_rates.get(a, 0.0) + row_swap_heating
                    heating_rates[b] = heating_rates.get(b, 0.0) + row_swap_heating
                    did_any = True
                if did_any:
                    time_elapsed += row_swap_time

            elif phase == "V":
                for (r, c) in v_swaps:
                    a = int(A[r, c])
                    b = int(A[r + 1, c])
                    if ignore_spectators and (a in specs and b in specs):
                        continue
                    A[r, c], A[r + 1, c] = b, a
                    heating_rates[a] = heating_rates.get(a, 0.0) + col_swap_heating
                    heating_rates[b] = heating_rates.get(b, 0.0) + col_swap_heating
                    did_any = True
                if did_any:
                    time_elapsed += col_swap_time

        if not np.array_equal(A, T):
            raise RuntimeError("SAT schedule did not realise target layout")

        return heating_rates, time_elapsed

    # ---- Heuristic fallback (Phases B/C/D) ----
    _, heur_heating, heur_time = heuristic_odd_even_reconfig(
        old_layout, new_layout, k=k,
    )
    # Merge heating
    for ion_id, h in heur_heating.items():
        heating_rates[ion_id] = heating_rates.get(ion_id, 0.0) + h
    time_elapsed += heur_time

    return heating_rates, time_elapsed


# =============================================================================
# Schedule-only rebuild for cache replay transitions
# (port of OLD _rebuild_schedule_for_layout)
# =============================================================================


def rebuild_schedule_for_layout(
    current_layout: np.ndarray,
    target_layout: np.ndarray,
    n_rows: int,
    n_cols: int,
    capacity: int,
    subgrid_size: Tuple[int, int, int] = (6, 4, 1),
    base_pmax_in: Optional[int] = None,
    router: Optional[Any] = None,
) -> List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]]:
    """Rebuild a SAT-based schedule to transition from current_layout to target_layout.

    Faithfully ported from OLD ``_rebuild_schedule_for_layout``.

    Iteratively solves SAT instances with BT pins (no gate pairs) to
    progressively move ions towards their target positions. Each cycle
    pins as many ions as possible (subject to (start_row, target_col)
    uniqueness) and solves a layout-only SAT.

    Parameters
    ----------
    current_layout : np.ndarray
        Starting layout (n_rows × n_cols).
    target_layout : np.ndarray
        Desired layout (n_rows × n_cols).
    n_rows, n_cols : int
        Grid dimensions.
    capacity : int
        Ions per segment (k).
    subgrid_size : tuple
        Patch sizing (cols, rows, increment).
    base_pmax_in : int or None
        Base P_max hint.
    router : WiseSatRouter or None
        Router instance. If None, creates a default one.

    Returns
    -------
    snapshots : list of (layout_after, schedule, [])
        One entry per cycle. Solved pairs is always empty (no gates).
    """
    if target_layout.shape != current_layout.shape:
        raise ValueError(
            f"rebuild_schedule_for_layout: shape mismatch: "
            f"old={current_layout.shape}, target={target_layout.shape}"
        )

    if router is None:
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.routers import (
            WiseSatRouter,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.config import (
            WISERoutingConfig,
        )
        router = WiseSatRouter(config=WISERoutingConfig())

    from qectostim.experiments.hardware_simulation.core.pipeline import QubitMapping

    n_c = max(subgrid_size[0], 1)
    n_r = max(subgrid_size[1], 1)
    patch_w = min(n_c, n_cols)
    patch_h = min(n_r, n_rows)

    cur = np.array(current_layout, copy=True)
    snapshots: List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]] = []
    max_cycles = max(n_rows, n_cols)

    _logger.info(
        "rebuild_schedule_for_layout: mismatch=%d",
        int(np.count_nonzero(cur != target_layout)),
    )

    for cycle_idx in range(max_cycles):
        if np.array_equal(cur, target_layout):
            _logger.info(
                "rebuild_schedule_for_layout: converged in %d cycle(s)", cycle_idx,
            )
            break

        mismatch_before = int(np.count_nonzero(cur != target_layout))

        # Generate patches (no offset)
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.orchestrator import (
            _generate_patch_regions,
        )
        patch_regions = _generate_patch_regions(
            n_rows, n_cols, patch_h, patch_w, 0, 0,
        )

        ion_positions: Dict[int, Tuple[int, int]] = {
            int(cur[r, c]): (r, c) for r in range(n_rows) for c in range(n_cols)
        }
        layout_snapshot = np.array(cur, copy=True)
        cycle_schedules: List[List[Dict[str, Any]]] = []

        for region in patch_regions:
            r0, c0, r1, c1 = region
            patch_grid = np.array(cur[r0:r1, c0:c1], dtype=int, copy=True)

            # Build BT pins: force ions toward target positions
            bt_map: Dict[int, Tuple[int, int]] = {}
            used_keys: Dict[Tuple[int, int], int] = {}

            for dr in range(r1 - r0):
                for dc in range(c1 - c0):
                    target_ion = int(target_layout[r0 + dr, c0 + dc])
                    pos = ion_positions.get(target_ion)
                    if pos is None:
                        continue
                    sr_global, sc_global = pos
                    sr_local = sr_global - r0
                    sc_local = sc_global - c0
                    if not (0 <= sr_local < (r1 - r0)):
                        continue
                    if not (0 <= sc_local < (c1 - c0)):
                        continue
                    key = (sr_local, sc_local)
                    existing = used_keys.get(key)
                    if existing is not None and existing != target_ion:
                        continue
                    used_keys[key] = target_ion
                    bt_map[target_ion] = (dr, dc)

            if not bt_map:
                continue

            boundary_adjacent = {
                "top": r0 > 0,
                "bottom": r1 < n_rows,
                "left": c0 > 0,
                "right": c1 < n_cols,
            }

            from qectostim.experiments.hardware_simulation.trapped_ion.routing.data_structures import (
                GridLayout,
            )
            grid_layout = GridLayout(
                grid=patch_grid,
                item_positions={
                    int(patch_grid[dr, dc]): (dr, dc)
                    for dr in range(r1 - r0) for dc in range(c1 - c0)
                },
            )

            try:
                # Solve with empty gate pairs — layout-only reconfig
                # Override boundary weights to 0 matching OLD code
                _saved_wB_col = router.config.boundary_soft_weight_col
                _saved_wB_row = router.config.boundary_soft_weight_row
                router.config.boundary_soft_weight_col = 0
                router.config.boundary_soft_weight_row = 0
                try:
                    result = router.route_batch(
                        physical_pairs=[],
                        current_mapping=QubitMapping(),
                        architecture=None,
                        initial_layout=grid_layout,
                        bt_positions=[bt_map],
                        full_gate_pairs=[[]],
                        col_offset=c0,
                        grid_origin=(r0, c0),
                        boundary_adjacent=boundary_adjacent,
                        cross_boundary_prefs=[{}],
                        ignore_initial_reconfig=False,
                    )
                finally:
                    router.config.boundary_soft_weight_col = _saved_wB_col
                    router.config.boundary_soft_weight_row = _saved_wB_row
            except NoFeasibleLayoutError:
                _logger.warning(
                    "rebuild_schedule_for_layout: patch r[%d:%d] c[%d:%d] infeasible",
                    r0, r1, c0, c1,
                )
                continue

            if result.success:
                fl = result.metrics.get("_final_layout", None)
                if fl is not None:
                    cur[r0:r1, c0:c1] = fl
                sched = result.metrics.get("schedule", result.operations)
                if isinstance(sched, list):
                    cycle_schedules.append(sched)

        # Merge this cycle's schedules
        merged: List[Dict[str, Any]] = []
        for s in cycle_schedules:
            if isinstance(s, list):
                for entry in s:
                    if isinstance(entry, dict):
                        merged.append(entry)
                    elif isinstance(entry, list):
                        merged.extend(e for e in entry if isinstance(e, dict))

        snapshots.append((cur.copy(), merged, []))

        mismatch_after = int(np.count_nonzero(cur != target_layout))
        _logger.info(
            "rebuild_schedule_for_layout: cycle %d mismatch %d→%d",
            cycle_idx, mismatch_before, mismatch_after,
        )

        if mismatch_after >= mismatch_before:
            _logger.warning(
                "rebuild_schedule_for_layout: no improvement, stopping",
            )
            break

        patch_w = min(patch_w + 1, n_cols)
        patch_h = min(patch_h + 1, n_rows)

    if not np.array_equal(cur, target_layout):
        _logger.warning(
            "rebuild_schedule_for_layout: did not fully converge",
        )

    return snapshots


# Aliases for backward compatibility
_compute_patch_gating_capacity = compute_patch_gating_capacity
_compute_cross_boundary_prefs = compute_cross_boundary_prefs
_pre_sat_sanity_checks = pre_sat_sanity_checks
_heuristic_phaseB_greedy_layout = heuristic_phaseB_greedy_layout
_heuristic_odd_even_reconfig = heuristic_odd_even_reconfig
