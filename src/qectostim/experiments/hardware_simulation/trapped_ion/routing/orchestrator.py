# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/orchestrator.py
"""
WISE Routing Orchestrator for high-level routing control.

This module provides the WISERoutingOrchestrator class that wraps
the SAT router with block caching, multi-round lookahead, BT propagation,
adaptive patch growth, cross-boundary preference forwarding, and per-patch
BT conflict detection.

Faithfully ported from old ionRoutingWISEArch() + _patch_and_route()
in qccd_WISE_ion_route.py.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

from qectostim.experiments.hardware_simulation.core.pipeline import QubitMapping
from qectostim.experiments.hardware_simulation.core.architecture import (
    LayoutTracker,
)

from qectostim.experiments.hardware_simulation.trapped_ion.routing.config import (
    wise_logger,
    WISERoutingConfig,
)
from qectostim.experiments.hardware_simulation.trapped_ion.routing.routers import (
    WiseSatRouter,
)
from qectostim.experiments.hardware_simulation.trapped_ion.routing.data_structures import (
    GridLayout,
)
from qectostim.experiments.hardware_simulation.trapped_ion.routing.layout_utils import (
    NoFeasibleLayoutError,
    compute_patch_gating_capacity,
    compute_cross_boundary_prefs,
    merge_patch_schedules,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WISERoutingBlock:
    """Cached routing block metadata."""
    gate_pattern: Tuple[Tuple[Tuple[int, int], ...], ...]
    tiling_steps_meta: List[Tuple[int, int, List[Tuple[Any, Any, Any]]]]
    reconfig_time: float


@dataclass
class PatchResult:
    """Result from solving a single patch."""
    schedule: List[Dict[str, Any]]
    final_layout: np.ndarray
    bt_pins: Dict[int, Tuple[int, int]]
    pairs_solved: List[Tuple[int, int]]
    time_elapsed: float


@dataclass
class TilingStep:
    """One entry per round of a tiling: layout + schedule + solved pairs."""
    layout_after: np.ndarray
    schedule: List[Dict[str, Any]]
    solved_pairs: List[Tuple[int, int]]


# ---------------------------------------------------------------------------
# Helper functions (ported from OLD _patch_and_route helpers)
# ---------------------------------------------------------------------------

def _generate_patch_regions(
    n_rows: int,
    n_cols: int,
    patch_h: int,
    patch_w: int,
    offset_row: int,
    offset_col: int,
) -> List[Tuple[int, int, int, int]]:
    """Generate non-overlapping patch regions that tile the grid."""
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
    ion_positions: Dict[int, Tuple[int, int]],
    region: Tuple[int, int, int, int],
) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
    """Split pairs into (inside patch, outside patch) buckets per round."""
    r0, c0, r1, c1 = region

    def _in_region(ion_idx: int) -> bool:
        pos = ion_positions.get(ion_idx)
        if pos is None:
            return False
        rr, cc = pos
        return (r0 <= rr < r1) and (c0 <= cc < c1)

    R = len(pairs_per_round)
    patch_pairs: List[List[Tuple[int, int]]] = [[] for _ in range(R)]
    leftovers: List[List[Tuple[int, int]]] = [[] for _ in range(R)]

    for ridx, round_pairs in enumerate(pairs_per_round):
        for ion_a, ion_b in round_pairs:
            if _in_region(ion_a) and _in_region(ion_b):
                patch_pairs[ridx].append((ion_a, ion_b))
            else:
                leftovers[ridx].append((ion_a, ion_b))

    return patch_pairs, leftovers


def _k_compatible(width: int, k: int) -> bool:
    """Check if a width is k-compatible for WISE block alignment."""
    if width <= 0 or width == 1:
        return False
    if width == k:
        return True
    if width > k:
        return (width % k) == 0
    return (k % width) == 0


def _filter_bt_for_patch(
    bt_full: Dict[int, Tuple[int, int]],
    region: Tuple[int, int, int, int],
    ion_positions: Dict[int, Tuple[int, int]],
    ions_in_patch_pairs: List[int],
    ions_in_tiling_pairs: List[int],
    ridx: int,
) -> Tuple[Dict[int, Tuple[int, int]], bool]:
    """Filter BT pins for a specific patch with conflict detection.

    Faithfully ported from OLD _patch_and_route BT filtering logic.

    Only keeps pins whose target falls inside the patch region,
    checks ion consistency between patch and tiling pairs,
    and detects (start_row_local, target_col_local) conflicts.
    """
    r0, c0, r1, c1 = region
    bt_patch: Dict[int, Tuple[int, int]] = {}
    bt_keys_used: Dict[Tuple[int, int], int] = {}
    patch_ion_set = set(ions_in_patch_pairs)
    tiling_ion_set = set(ions_in_tiling_pairs)

    for ion, (global_r, global_c) in bt_full.items():
        # Only pin ions that are consistently in/out of this tiling's pairs
        if (ion in patch_ion_set) != (ion in tiling_ion_set):
            continue
        # Only pin ions whose *target* is inside this patch region
        if not (r0 <= global_r < r1 and c0 <= global_c < c1):
            continue

        local_r = global_r - r0
        local_c = global_c - c0

        # For round 0: check (start_row_local, target_col_local) uniqueness
        if ridx == 0:
            init_pos = ion_positions.get(ion)
            if init_pos is None:
                continue
            init_row_global, _ = init_pos
            start_row_local = init_row_global - r0
            if not (0 <= start_row_local < (r1 - r0)):
                continue

            key = (start_row_local, local_c)
            existing_ion = bt_keys_used.get(key)
            if existing_ion is not None and existing_ion != ion:
                _logger.warning(
                    "BT pin conflict in patch r[%d:%d] c[%d:%d], round=%d: "
                    "start_row_local=%d, target_col_local=%d; keeping ion %d, "
                    "dropping ion %d",
                    r0, r1, c0, c1, ridx,
                    start_row_local, local_c, existing_ion, ion,
                )
                continue

            bt_keys_used[key] = ion

        bt_patch[ion] = (local_r, local_c)

    return bt_patch, len(bt_patch) > 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class WISERoutingOrchestrator:
    """High-level orchestrator for WISE routing.

    Wraps WiseSatRouter with:
    - Block caching for repeated gate patterns (OLD block_cache)
    - Multi-round lookahead with BT propagation (OLD BTs)
    - Patch-based decomposition with gating capacity capping
    - Per-patch BT conflict detection (start_row/target_col uniqueness)
    - Cross-boundary preference forwarding to SAT encoder
    - Adaptive patch growth when progress stalls
    - Schedule merging across patches

    Faithfully ported from ``ionRoutingWISEArch`` + ``_patch_and_route``
    in the old ``qccd_WISE_ion_route.py``.
    """

    MAX_CYCLES = 10

    def __init__(
        self,
        router: Optional[WiseSatRouter] = None,
        config: Optional[WISERoutingConfig] = None,
        lookahead: int = 2,
        block_size: int = 4,
        subgrid_size: Tuple[int, int, int] = (6, 4, 1),
    ):
        self.router = router or WiseSatRouter(config=config)
        self.config = config or WISERoutingConfig()
        self.lookahead = lookahead
        self.block_size = block_size
        self.subgrid_size = subgrid_size

        # Block cache: maps canonical gate pattern key ->
        #   { offset_within_block -> tiling_steps_meta }
        self._block_cache: Dict[
            Tuple[Tuple[Tuple[int, int], ...], ...],
            Dict[int, List[Tuple[int, int, List[Tuple[Any, Any, Any]]]]]
        ] = {}

    # ------------------------------------------------------------------
    # Block-cache helpers
    # ------------------------------------------------------------------

    def _make_block_key(
        self,
        parallel_pairs: List[List[Tuple[int, int]]],
        start_idx: int,
    ) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        """Create a hashable key for a block of gate rounds."""
        end_idx = min(len(parallel_pairs), start_idx + self.block_size)
        block_rounds = parallel_pairs[start_idx:end_idx]
        return tuple(
            tuple(sorted(round_pairs)) for round_pairs in block_rounds
        )

    # ------------------------------------------------------------------
    # Core: patch_and_route (faithful port of OLD _patch_and_route)
    # ------------------------------------------------------------------

    def patch_and_route(
        self,
        current_layout: np.ndarray,
        P_arr: List[List[Tuple[int, int]]],
        n_rows: int,
        n_cols: int,
        capacity: int,
        active_ions: Optional[Set[int]] = None,
        ignore_initial_reconfig: bool = False,
        base_pmax_in: Optional[int] = None,
        BTs: Optional[List[Dict]] = None,
    ) -> List[Tuple[int, int, List[Tuple[Any, Any, Any]]]]:
        """Patch-based Level-1 slicer. Partition the device into
        checkerboard patches and solve each patch locally via SAT.

        Faithfully ported from OLD ``_patch_and_route``.

        Returns a list of tiling steps, each (cycle_idx, tiling_idx, snapshots)
        where snapshots is [(layout_after_r, schedule_r, solved_pairs_r)]
        for each round.
        """
        R = len(P_arr)
        if R == 0:
            return []

        n_c = max(self.subgrid_size[0], 1)
        n_r = max(self.subgrid_size[1], 1)
        inc = self.subgrid_size[2]

        layouts_after: List[np.ndarray] = [
            np.array(current_layout, copy=True) for _ in range(R)
        ]
        remaining_pairs: List[List[Tuple[int, int]]] = [
            list(rp) for rp in P_arr
        ]
        base_pmax = base_pmax_in or R
        # GAP 10 FIX: track prev_pmax across patches so subsequent
        # patches can start from the best P_max found so far.
        prev_pmax: Optional[int] = None

        tiling_steps: List[Tuple[int, int, List[Tuple[Any, Any, Any]]]] = []

        no_progress_cycles = 0
        cycle_idx = 0
        total_remaining = sum(len(rp) for rp in remaining_pairs)

        while total_remaining > 0:
            total_remaining = sum(len(rp) for rp in remaining_pairs)
            if total_remaining == 0:
                break

            # ---- k-align patch width (faithful port from OLD) ----
            if n_c < capacity and (capacity % n_c != 0):
                n_c += int(np.mod(capacity, n_c))
            elif n_c > capacity and (n_c % capacity != 0):
                n_c = (n_c // capacity) * capacity

            patch_w = min(n_c, n_cols)
            patch_h = min(n_r, n_rows)

            # ---- Build tiling offsets ----
            tilings: List[Tuple[int, int]] = [(0, 0)]
            half_h = patch_h // 2
            if half_h > 0 and (half_h, 0) not in tilings:
                tilings.append((half_h, 0))
            if patch_w % 2 == 0:
                half_w = patch_w // 2
                if (
                    half_w > 0
                    and _k_compatible(half_w, capacity)
                    and (0, half_w) not in tilings
                ):
                    tilings.append((0, half_w))

            wise_logger.info(
                "patch_and_route cycle %d: remaining=%d, patch=%dx%d, tilings=%d",
                cycle_idx + 1, total_remaining, patch_h, patch_w, len(tilings),
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
                pairs_before_tiling: List[List[Tuple[int, int]]] = [
                    list(rp) for rp in remaining_pairs
                ]

                patch_regions = _generate_patch_regions(
                    n_rows, n_cols, patch_h, patch_w, off_r, off_c,
                )

                # ---- Determine BT pins for this tiling ----
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
                    BT_for_tiling = [{} for _ in range(R)]
                    pairs_for_tiling = [set() for _ in range(R)]

                patch_schedules_this_tiling: List[List[List[Dict[str, Any]]]] = []

                for region in patch_regions:
                    patch_pairs, new_remaining = _split_pairs_for_patch(
                        remaining_pairs, ion_positions, region,
                    )
                    r0, c0, r1, c1 = region
                    patch_grid = np.array(
                        layouts_after[0][r0:r1, c0:c1], dtype=int, copy=True,
                    )

                    # ---- Gating capacity capping ----
                    max_gating_zones = compute_patch_gating_capacity(
                        n=r1 - r0, m=c1 - c0, col_offset=c0, capacity=capacity,
                    )
                    if max_gating_zones > 0:
                        for ridx in range(R):
                            rp = patch_pairs[ridx]
                            if len(rp) > max_gating_zones:
                                spill = rp[max_gating_zones:]
                                patch_pairs[ridx] = rp[:max_gating_zones]
                                new_remaining[ridx].extend(spill)

                    patch_gate_count = sum(len(rp) for rp in patch_pairs)

                    # ---- Boundary adjacency + cross-boundary prefs ----
                    boundary_adjacent = {
                        "top": r0 > 0,
                        "bottom": r1 < n_rows,
                        "left": c0 > 0,
                        "right": c1 < n_cols,
                    }
                    cross_boundary_prefs = compute_cross_boundary_prefs(
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
                        continue

                    # ---- BT conflict detection per-patch ----
                    ions_in_pp: List[List[int]] = [[] for _ in range(R)]
                    for ridx, ppairs in enumerate(patch_pairs):
                        for p in ppairs:
                            ions_in_pp[ridx].append(p[0])
                            ions_in_pp[ridx].append(p[1])

                    ions_in_tp: List[List[int]] = [[] for _ in range(R)]
                    for ridx, tpairs in enumerate(pairs_for_tiling):
                        for p in tpairs:
                            ions_in_tp[ridx].append(p[0])
                            ions_in_tp[ridx].append(p[1])

                    BT_patch: List[Dict[int, Tuple[int, int]]] = [{} for _ in range(R)]
                    use_bt_softs = False
                    for ridx in range(R):
                        bt_full = BT_for_tiling[ridx] if ridx < len(BT_for_tiling) else {}
                        if not bt_full:
                            continue
                        filtered, has_pins = _filter_bt_for_patch(
                            bt_full, region, ion_positions,
                            ions_in_pp[ridx], ions_in_tp[ridx], ridx,
                        )
                        BT_patch[ridx] = filtered
                        if has_pins:
                            use_bt_softs = True

                    wise_logger.info(
                        "solving patch r[%d:%d] c[%d:%d] gates=%d BT=%s",
                        r0, r1, c0, c1, patch_gate_count,
                        [len(bt) for bt in BT_patch],
                    )

                    # ---- Solve this patch via WiseSatRouter ----
                    # GAP 2 FIX: OLD _patch_and_route passes wB_col=0, wB_row=0
                    # to _optimal_QMR_for_WISE.  Override config temporarily.
                    _saved_wB_col = self.router.config.boundary_soft_weight_col
                    _saved_wB_row = self.router.config.boundary_soft_weight_row
                    self.router.config.boundary_soft_weight_col = 0
                    self.router.config.boundary_soft_weight_row = 0
                    # GAP 10: propagate prev_pmax so subsequent patches
                    # start from a proven feasible P_max
                    _saved_base_pmax = self.router.config.base_pmax_in
                    if prev_pmax is not None:
                        self.router.config.base_pmax_in = prev_pmax
                    patch_start = time.time()
                    try:
                        grid_layout = GridLayout(
                            grid=patch_grid,
                            item_positions={
                                int(patch_grid[dr, dc]): (dr, dc)
                                for dr in range(r1 - r0)
                                for dc in range(c1 - c0)
                            },
                        )

                        current_pairs = patch_pairs[0] if patch_pairs else []
                        lookahead_pairs_l = patch_pairs[1:] if len(patch_pairs) > 1 else None

                        result = self.router.route_batch(
                            physical_pairs=current_pairs,
                            current_mapping=QubitMapping(),
                            architecture=None,
                            initial_layout=grid_layout,
                            lookahead_pairs=lookahead_pairs_l,
                            bt_positions=BT_patch,
                            full_gate_pairs=remaining_pairs,
                            col_offset=c0,
                            grid_origin=(r0, c0),
                            cross_boundary_prefs=cross_boundary_prefs,
                            boundary_adjacent=boundary_adjacent,
                            ignore_initial_reconfig=(
                                ignore_initial_reconfig
                                and tiling_idx == 0
                                and cycle_idx == 0
                            ),
                        )
                    except NoFeasibleLayoutError as exc:
                        wise_logger.warning(
                            "Error solving patch r[%d:%d] c[%d:%d]: %s",
                            r0, r1, c0, c1, exc,
                        )
                        for ridx in range(R):
                            new_remaining[ridx].extend(patch_pairs[ridx])
                        remaining_pairs = new_remaining
                        continue
                    finally:
                        # GAP 2 FIX: restore original boundary weights
                        self.router.config.boundary_soft_weight_col = _saved_wB_col
                        self.router.config.boundary_soft_weight_row = _saved_wB_row
                        # GAP 10: restore base_pmax_in
                        self.router.config.base_pmax_in = _saved_base_pmax

                    patch_elapsed = time.time() - patch_start

                    if result.success:
                        # GAP 10: update prev_pmax from successful solve
                        _p = result.metrics.get("passes")
                        if _p is not None and isinstance(_p, int) and _p > 0:
                            if prev_pmax is None or _p < prev_pmax:
                                prev_pmax = _p

                        final_layouts = result.metrics.get("_final_layouts", None)
                        final_layout_single = result.metrics.get("_final_layout", None)

                        if final_layouts is not None:
                            for ridx in range(min(R, len(final_layouts))):
                                layouts_after[ridx][r0:r1, c0:c1] = final_layouts[ridx]
                        elif final_layout_single is not None:
                            layouts_after[0][r0:r1, c0:c1] = final_layout_single

                        schedule_data = result.metrics.get("schedule", None)
                        if schedule_data is None:
                            schedule_data = [result.operations]

                        patch_schedules_this_tiling.append(schedule_data)
                    else:
                        for ridx in range(R):
                            new_remaining[ridx].extend(patch_pairs[ridx])

                    remaining_pairs = new_remaining

                # ---- Merge patch schedules for this tiling ----
                merged_tiling_schedule = merge_patch_schedules(
                    patch_schedules_this_tiling, R,
                )

                solved_pairs_per_round: List[List[Tuple[int, int]]] = []
                for ridx in range(R):
                    before_set = set(pairs_before_tiling[ridx])
                    after_set = set(remaining_pairs[ridx])
                    solved_here = list(before_set - after_set)
                    solved_pairs_per_round.append(solved_here)

                tiling_snapshot = [
                    (la.copy(), copy.deepcopy(sched), list(sp))
                    for la, sched, sp in zip(
                        layouts_after,
                        merged_tiling_schedule,
                        solved_pairs_per_round,
                    )
                ]
                tiling_steps.append((cycle_idx, tiling_idx, tiling_snapshot))

                gates_after_tiling = sum(len(rp) for rp in remaining_pairs)
                solved_this_tiling = gates_before_tiling - gates_after_tiling
                wise_logger.info(
                    "tiling %d: solved=%d, remaining=%d",
                    tiling_idx + 1, solved_this_tiling, gates_after_tiling,
                )

            # ---- End of cycle: adaptive growth ----
            gates_after_cycle = sum(len(rp) for rp in remaining_pairs)
            solved_cycle = cycle_start_remaining - gates_after_cycle
            fully_global = (patch_h >= n_rows) and (patch_w >= n_cols)

            if gates_after_cycle == 0:
                break
            if solved_cycle == 0:
                no_progress_cycles += 1
                if no_progress_cycles >= self.MAX_CYCLES or fully_global:
                    wise_logger.warning(
                        "No progress in cycle %d; stopping patch routing",
                        cycle_idx + 1,
                    )
                    break
            else:
                no_progress_cycles = 0

            cycle_idx += 1
            n_c += max(inc, min(n_c, capacity))
            n_r += inc
            total_remaining = gates_after_cycle

        unresolved = sum(len(rp) for rp in remaining_pairs)
        if unresolved > 0:
            wise_logger.warning(
                "%d gate(s) remain unresolved after patch routing", unresolved,
            )

        return tiling_steps

    # ------------------------------------------------------------------
    # Core: route_all_rounds -- main entry (port of ionRoutingWISEArch)
    # ------------------------------------------------------------------

    def route_all_rounds(
        self,
        initial_layout: np.ndarray,
        parallel_pairs: List[List[Tuple[int, int]]],
        n_rows: int,
        n_cols: int,
        capacity: int,
        active_ions: Optional[Set[int]] = None,
        ignore_initial_reconfig: bool = False,
        base_pmax_in: Optional[int] = None,
    ) -> Tuple[
        List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]],
        float,
    ]:
        """Route all rounds of gate pairs with BT propagation and block caching.

        Faithfully ported from OLD ``ionRoutingWISEArch`` main loop.

        Returns
        -------
        all_reconfigs : List of (layout_after, schedule, solved_pairs)
            One entry per reconfiguration step.
        total_time : float
            Estimated total reconfiguration time.
        """
        if not parallel_pairs:
            return [], 0.0

        current_layout = np.array(initial_layout, dtype=int)
        all_reconfigs: List[Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]] = []
        total_time = 0.0

        if active_ions is None:
            active_ions = set(int(x) for x in initial_layout.flatten())
        active_set = active_ions

        blk_idx = 0
        blk_end = 0
        cache_for_block: Dict[int, Any] = {}
        recheck_cache = True
        tiling_steps_from_cache = False
        pending_first_cached_reconfig = False

        BTs: Optional[List[Dict]] = None

        tiling_steps_meta: List[Tuple[int, int, List[Tuple[Any, Any, Any]]]] = []
        tiling_steps: List[List[Tuple[Any, Any, Any]]] = []
        tiling_step: List[Tuple[Any, Any, Any]] = []
        had_multiple_tiling_steps = False

        idx = 0
        while idx < len(parallel_pairs):
            if (not had_multiple_tiling_steps) and tiling_steps and tiling_steps[0]:
                pass
            else:
                window_start = idx
                window_end = min(len(parallel_pairs), self.lookahead + idx)
                P_arr = parallel_pairs[window_start:window_end]

                if recheck_cache:
                    new_blk_end = min(len(parallel_pairs), window_start + self.block_size) - window_start
                    new_block_key = self._make_block_key(parallel_pairs, window_start)
                    new_cache = self._block_cache.setdefault(new_block_key, {})

                    if len(new_cache) > 0:
                        cache_for_block = new_cache
                        blk_idx = 0
                        blk_end = new_blk_end
                        tiling_steps_from_cache = True
                    elif len(cache_for_block) == 0:
                        cache_for_block = new_cache
                        blk_idx = 0
                        blk_end = new_blk_end
                        tiling_steps_from_cache = False
                    else:
                        tiling_steps_from_cache = False

                if tiling_steps_from_cache and blk_idx in cache_for_block:
                    tiling_steps_meta = copy.deepcopy(cache_for_block[blk_idx])
                    pending_first_cached_reconfig = (blk_idx == 0)
                    wise_logger.info(
                        "Reusing cached block at idx=%d, blk_idx=%d", idx, blk_idx,
                    )
                else:
                    tiling_steps_meta = self.patch_and_route(
                        current_layout, P_arr,
                        n_rows, n_cols, capacity,
                        active_ions=active_ions,
                        ignore_initial_reconfig=(ignore_initial_reconfig and idx == 0),
                        base_pmax_in=base_pmax_in,
                        BTs=BTs,
                    )
                    if len(cache_for_block) < self.block_size:
                        cache_for_block[blk_idx] = copy.deepcopy(tiling_steps_meta)
                    wise_logger.info(
                        "patch_and_route idx=%d: %d steps, cache=%d",
                        idx, len(tiling_steps_meta), len(cache_for_block),
                    )

                tiling_steps = [snap for (_cy, _ti, snap) in tiling_steps_meta]
                had_multiple_tiling_steps = len(tiling_steps) > 1

                # ---- Build BTs for next window ----
                if tiling_steps_meta:
                    R_local = len(P_arr)
                    BTs = [{} for _ in range(max(0, R_local - 1))]
                    for cy_m, ti_m, tiling in tiling_steps_meta:
                        for r, (layout_r, _sched_r, _solved_r) in enumerate(tiling[1:]):
                            bt_map: Dict[int, Tuple[int, int]] = {}
                            for rr in range(n_rows):
                                for cc in range(n_cols):
                                    ionidx = int(layout_r[rr][cc])
                                    if ionidx in active_set:
                                        bt_map[ionidx] = (rr, cc)
                            bt_key = (cy_m, ti_m)
                            if r < len(BTs):
                                BTs[r][bt_key] = (bt_map, list(_solved_r))
                else:
                    BTs = None

            if not tiling_step:
                tiling_step = [t.pop(0) for t in tiling_steps if t]

            if not tiling_step:
                idx += 1
                blk_idx += 1
                recheck_cache = (blk_idx not in cache_for_block)
                if blk_idx == blk_end:
                    cache_for_block = {}
                    blk_idx = blk_end = 0
                continue

            layout_after, schedule, solved_pairs = tiling_step.pop(0)

            if tiling_steps_from_cache and pending_first_cached_reconfig:
                all_reconfigs.append((layout_after, [], solved_pairs))
                pending_first_cached_reconfig = False
            else:
                all_reconfigs.append((layout_after, schedule, solved_pairs))

            current_layout = np.array(layout_after, dtype=int)

            while tiling_step:
                la, sch, sp = tiling_step.pop(0)
                all_reconfigs.append((la, sch, sp))
                current_layout = np.array(la, dtype=int)

            idx += 1
            blk_idx += 1
            recheck_cache = (blk_idx not in cache_for_block)
            if blk_idx == blk_end:
                cache_for_block = {}
                blk_idx = blk_end = 0

        return all_reconfigs, total_time

    def clear_cache(self) -> None:
        """Clear the block cache."""
        self._block_cache.clear()
        wise_logger.info("Block cache cleared")


__all__ = [
    "WISERoutingBlock",
    "WISERoutingOrchestrator",
    "TilingStep",
]
