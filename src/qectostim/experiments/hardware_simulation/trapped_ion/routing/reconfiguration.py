# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/reconfiguration.py
"""
Reconfiguration planner for the WISE grid architecture.

This module provides ``ReconfigurationPlanner`` for computing and executing
global ion reconfigurations on the WISE trapped-ion architecture:

* ``physicalOperation()`` — main entry point, returns a planner instance.
* ``_optimal_QMR_for_WISE()`` — SAT-pool orchestrator for optimal schedules.
* ``_runOddEvenReconfig()`` — schedule execution (SAT path + heuristic B/C/D).
* ``phaseB_greedy_layout()`` — Hungarian-matching layout helper.
"""
from __future__ import annotations

import logging
import math
import os
import pickle
import signal
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from scipy.optimize import linear_sum_assignment

# Architecture types
from ..architecture import (
    Ion,
    QubitIon,
    CoolingIon,
    SpectatorIon,
    QCCDNode,
    Trap,
    ManipulationTrap,
    StorageTrap,
    Junction,
    Crossing,
    Operations,
    QCCDWiseArch,
    QCCDComponent,
)

# Physics constants — single source of truth
from ..physics import DEFAULT_CALIBRATION as _CAL

# Transport data classes — timing / heating constants live here
from ..operations import (
    Split as _TSplit,
    Merge as _TMerge,
    Move as _TMove,
    JunctionCrossing as _TJunctionCrossing,
    CrystalRotation as _TCrystalRotation,
)

# SAT solver infrastructure (same routing sub-package)
from .sat_solver import (
    NoFeasibleLayoutError,
    _WiseSatBuilderContext,
    _wise_build_structural_cnf,
    _wise_decode_schedule_from_model,
    _wise_extract_round_pass_usage,
    _wise_sat_config_worker,
    _wise_safe_sat_pool_workers,
    run_sat_with_timeout_file,
    run_rc2_with_timeout_file,
    wise_logger,
    CoreGroups,
    WISE_LOGGER_NAME,
)

import multiprocessing as mp
from pysat.examples.rc2 import RC2


# ============================================================================
# ReconfigurationPlanner
# ============================================================================

class ReconfigurationPlanner:
    """
    Planner for global ion reconfigurations on the WISE grid architecture.

    This class computes optimal (SAT-based) or heuristic reconfiguration
    schedules and provides a ``run()`` method to execute the ion movements.
    """

    KEY = Operations.GLOBAL_RECONFIG

    def __init__(
        self,
        run_fn: Callable[[], None],
        involvedComponents: Sequence[QCCDComponent],
        wiseArch: QCCDWiseArch,
        reconfigTime: float,
    ) -> None:
        self._run_fn = run_fn
        self._involvedComponents: List[QCCDComponent] = list(involvedComponents)
        self._wiseArch: QCCDWiseArch = wiseArch
        self._reconfigTime: float = reconfigTime
        self._fidelity: float = 1.0
        self._dephasingFidelity: float = 1.0
        # Per-ion movement metadata for animation: [(ion_idx, src_zone, tgt_zone), ...]
        self._ion_movements: List[Tuple[int, str, str]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def involvedComponents(self) -> Sequence[QCCDComponent]:
        """Components (traps) involved in this reconfiguration."""
        return self._involvedComponents

    def operationTime(self) -> float:
        """Total time for this reconfiguration in microseconds.

        ``_reconfigTime`` is computed internally in seconds (summing
        calibration constants which are in seconds).  Convert to µs
        here so that all ``operationTime()`` calls across the codebase
        (``MSGate``, ``SingleQubitGate``, ``Measurement``, …) return
        the same unit, which ``paralleliseOperationsWithBarriers``
        relies on for correct scheduling.
        """
        return self._reconfigTime * 1e6

    def fidelity(self) -> float:
        """Fidelity of this operation (noise in heating model)."""
        return self._fidelity

    def dephasingFidelity(self) -> float:
        """Dephasing fidelity based on T2 time."""
        t2 = _CAL.t2_time
        return 1 - (1 - np.exp(-self._reconfigTime / t2)) / 2

    def run(self) -> None:
        """Execute the reconfiguration, moving ions to their new positions."""
        self._run_fn()

    @classmethod
    def physicalOperation(
        cls,
        arrangement: Mapping[Trap, Sequence[Ion]],
        wiseArch: QCCDWiseArch,
        oldAssignment: Sequence[Sequence[int]],
        newAssignment: Sequence[Sequence[int]],
        schedule: Optional[List[Dict[str, Any]]] = None,
        initial_placement: bool = False,
    ):
        heatingRates, reconfigTime = cls._runOddEvenReconfig(
            wiseArch,
            arrangement,
            oldAssignment,
            newAssignment,
            sat_schedule=schedule,
            initial_placement=initial_placement,
        )
        reconfigTime = 1e-20 if initial_placement else reconfigTime

        def run_fn():
            for trap in arrangement.keys():
                while trap.ions:
                    trap.removeIon(trap.ions[0])
            for trap, ions in arrangement.items():
                for i, ion in enumerate(ions):
                    trap.addIon(ion, offset=i)
                    if initial_placement:
                        continue
                    ion.addMotionalEnergy(heatingRates[ion.idx])

        return cls(
            run_fn=run_fn,
            involvedComponents=list(arrangement.keys()),
            wiseArch=wiseArch,
            reconfigTime=reconfigTime,
        )

    # ------------------------------------------------------------------
    # SAT-pool orchestrator
    # ------------------------------------------------------------------

    @staticmethod
    def _optimal_QMR_for_WISE(
        A_in: np.ndarray,
        P_arr: List[List[Tuple[int, int]]],
        *,
        k: int,
        BT: List[Dict[int, Tuple[int, int]]] = None,
        wH: List[int] = None,    # unused in D-min version
        wV: List[int] = None,    # unused in D-min version
        wB_col: int = 1,
        wB_row: int = 1,
        max_rc2_time: float = 4800.0,
        max_sat_time=4800.0,
        active_ions: Set[int] = None,
        full_P_arr: List[List[Tuple[int, int]]] = [],
        ignore_initial_reconfig: bool = False,
        base_pmax_in: int = None,
        prev_pmax: int = None,
        grid_origin: Tuple[int, int] = (0, 0),
        boundary_adjacent: Optional[Dict[str, bool]] = None,
        cross_boundary_prefs: Optional[List[Dict[int, Set[str]]]] = None,
        bt_soft: bool = False,
    ) -> Tuple[List[np.ndarray], List[List[Dict[str, Any]]], int]:
        DEBUG_DIAG = True
        DEBUG_DIAG_DETAILED = False

        A_in = np.asarray(A_in, dtype=int)
        n, m = A_in.shape
        R = len(P_arr)
        optimize_round_start = 1 if (ignore_initial_reconfig and R > 0) else 0

        def _apply_time_env(default_value: Optional[float], env_var: str) -> Optional[float]:
            env_val = os.environ.get(env_var)
            if not env_val:
                return default_value
            try:
                parsed = float(env_val)
            except ValueError:
                return default_value
            if parsed <= 0:
                return default_value
            if default_value is None or default_value <= 0:
                return parsed
            return min(default_value, parsed)

        max_sat_time = _apply_time_env(max_sat_time, "WISE_MAX_SAT_TIME")
        max_rc2_time = _apply_time_env(max_rc2_time, "WISE_MAX_RC2_TIME")

        if BT is None:
            BT = [{} for _ in range(R)]
        if len(full_P_arr) == 0:
            full_P_arr = P_arr

        if boundary_adjacent is None:
            boundary_adjacent = {"top": False, "bottom": False, "left": False, "right": False}
        else:
            boundary_adjacent = {
                "top": bool(boundary_adjacent.get("top", False)),
                "bottom": bool(boundary_adjacent.get("bottom", False)),
                "left": bool(boundary_adjacent.get("left", False)),
                "right": bool(boundary_adjacent.get("right", False)),
            }

        if cross_boundary_prefs is None or len(cross_boundary_prefs) != R:
            cross_boundary_prefs = [dict() for _ in range(R)]

        bt_soft_enabled = bool(
            bt_soft and BT and any(bt_round for bt_round in BT)
        )
        bt_soft_weight_value = 0
        if bt_soft_enabled:
            base_pref_weight = max(wB_col, wB_row, 1)
            bt_soft_weight_value = max(100, base_pref_weight * 10)
            if DEBUG_DIAG:
                wise_logger.info(
                    "[WISE] soft BT enabled: weight=%d", bt_soft_weight_value
                )

        if base_pmax_in is None:
            base_pmax_in = R

        if prev_pmax is None:
            prev_pmax = 0

        row_offset = 0
        col_offset = 0
        if grid_origin is not None:
            row_offset, col_offset = grid_origin

        CAPACITY = k

        # -------------------------------
        # Basic sets of ions
        # -------------------------------
        row_of = {int(A_in[r, c]): r for r in range(n) for c in range(m)}
        col_of = {int(A_in[r, c]): c for r in range(n) for c in range(m)}
        ions_all = set(int(x) for x in A_in.flatten())

        if active_ions is None:
            # Identify "active" vs "spectator" ions
            active_ions = set()
            # Any ion in a pair is active
            for r, pairs in enumerate(P_arr):
                for i1, i2 in pairs:
                    if i1 in ions_all:
                        active_ions.add(i1)
                    if i2 in ions_all:
                        active_ions.add(i2)
            # Any ion pinned in BT is also active
            for r, bt in enumerate(BT):
                for i in bt.keys():
                    if i in ions_all:
                        active_ions.add(i)

        spectator_ions = ions_all - set(active_ions)

        if DEBUG_DIAG:
            wise_logger.debug(
                "[WISE] ions_all=%d, active_ions=%d, spectator_ions=%d",
                len(ions_all),
                len(active_ions),
                len(spectator_ions),
            )

        # -------------------------------
        # Pre-checks (same semantics as original)
        # -------------------------------

        # (a) No two ions pinned to the same (d,c) in a round
        for r, bt in enumerate(BT):
            seen = {}
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
                        wise_logger.warning(
                            "UNSAT-soft: %s (will leave to Max-SAT)", msg
                        )
                        continue
                    raise ValueError(f"UNSAT: {msg}")
                seen[key] = i

        # (b) Pair vs BT conflicts: same round, incompatible BT rows/blocks
        for r, pairs in enumerate(P_arr):
            for i1, i2 in pairs:
                if i1 not in ions_all or i2 not in ions_all:
                    continue
                if i1 in BT[r] and i2 in BT[r]:
                    d1, c1 = BT[r][i1]
                    d2, c2 = BT[r][i2]
                    if d1 != d2:
                        msg = (
                            f"round {r} pair {(i1,i2)} BT rows differ: {d1} vs {d2}."
                        )
                        if bt_soft_enabled:
                            wise_logger.warning(
                                "UNSAT-soft: %s (will leave to Max-SAT)", msg
                            )
                            continue
                        raise ValueError(f"UNSAT: {msg}")
                    if (c1 // k) != (c2 // k):
                        msg = (
                            f"round {r} pair {(i1,i2)} BT blocks differ: {c1}//{k} vs {c2}//{k}."
                        )
                        if bt_soft_enabled:
                            wise_logger.warning(
                                "UNSAT-soft: %s (will leave to Max-SAT)", msg
                            )
                            continue
                        raise ValueError(f"UNSAT: {msg}")

        # (c) Round-0: same source row & same target column among reserved ions
        if len(BT) >= 1:
            buckets = {}
            for i, (d0, c0) in BT[0].items():
                if i not in ions_all:
                    continue
                sr = row_of[i]
                buckets.setdefault((sr, c0), []).append(i)
            bad = {key: vs for key, vs in buckets.items() if len(vs) > 1}
            if bad:
                msg = (
                    "round 0 has reserved ions from the same start row "
                    f"targeting the same column: {bad}"
                )
                if bt_soft_enabled:
                    wise_logger.warning(
                        "UNSAT-soft: %s (will leave to Max-SAT)", msg
                    )
                else:
                    raise ValueError(f"UNSAT: {msg}")

        # (d) Column oversubscription from BT
        for r, bt in enumerate(BT):
            col_counts = {}
            for i, (_, c) in bt.items():
                if i not in ions_all:
                    continue
                col_counts[c] = col_counts.get(c, 0) + 1
            bad_cols = {c: cnt for c, cnt in col_counts.items() if cnt > n}
            if bad_cols:
                msg = f"BT[{r}] pins {bad_cols} ions to one column, exceeds n={n}."
                if bt_soft_enabled:
                    wise_logger.warning(
                        "UNSAT-soft: %s (will leave to Max-SAT)", msg
                    )
                else:
                    raise ValueError(f"UNSAT: {msg}")

        ions = sorted(ions_all)
        first_block_idx = col_offset // CAPACITY
        last_block_idx = (col_offset + m - 1) // CAPACITY
        num_blocks = last_block_idx - first_block_idx + 1

        block_cells: List[List[Tuple[int, int]]] = []
        block_fully_inside: List[bool] = []
        block_widths: List[int] = []
        global_patch_start = col_offset
        global_patch_end = col_offset + m

        for b_local in range(num_blocks):
            b_global = first_block_idx + b_local
            global_start = b_global * CAPACITY
            global_end = (b_global + 1) * CAPACITY
            local_start = max(0, global_start - col_offset)
            local_end = min(m, global_end - col_offset)
            cells: List[Tuple[int, int]] = []
            for d in range(n):
                for j_local in range(local_start, local_end):
                    cells.append((d, j_local))
            block_cells.append(cells)
            block_widths.append(local_end - local_start)
            block_fully_inside.append(
                (global_start >= global_patch_start)
                and (global_end <= global_patch_end)
            )
        ions_set = set(ions)

        builder_ctx = _WiseSatBuilderContext(
            A_in=A_in,
            BT=BT,
            P_arr=P_arr,
            full_P_arr=full_P_arr,
            ions=ions,
            n=n,
            m=m,
            R=R,
            block_cells=block_cells,
            block_fully_inside=block_fully_inside,
            block_widths=block_widths,
            num_blocks=num_blocks,
            wB_col=wB_col,
            wB_row=wB_row,
            debug_diag=DEBUG_DIAG,
        )

        def compute_outer_pairs() -> Optional[List[List[Tuple[int, int]]]]:
            if not full_P_arr:
                return None
            outer: List[List[Tuple[int, int]]] = []
            for r in range(R):
                inner_set = set(P_arr[r]) if r < len(P_arr) else set()
                round_outer: List[Tuple[int, int]] = []
                full_round = full_P_arr[r] if r < len(full_P_arr) else []
                for pair in full_round:
                    if pair in inner_set:
                        continue
                    i1, i2 = pair
                    if i1 in ions_set or i2 in ions_set:
                        round_outer.append(pair)
                outer.append(round_outer)
            return outer

        outer_pairs = compute_outer_pairs()

        # -------------------------------
        # Structural CNF / WCNF builder for given P_bound
        # -------------------------------
        def wise_debug_boundary_stats(
            label: str,
            model: Iterable[int],
            vpool,
            var_a,
            ions: Iterable[int],
            n_sub: int,
            m_sub: int,
            R: int,
            P_bound: Union[int, Sequence[int]],
            inner_pairs: Optional[List[List[Tuple[int, int]]]] = None,
            outer_pairs: Optional[List[List[Tuple[int, int]]]] = None,
            boundary_adjacent: Optional[Dict[str, bool]] = None,
        ) -> Dict[str, Any]:
            """
            Patch-aware boundary stats. Counts how often ions involved in gates land
            on boundaries that actually have adjacent patches.
            """
            model_set = {l for l in model if l > 0}

            def lit_true(v: int) -> bool:
                return v in model_set

            if isinstance(P_bound, int):
                P_bounds = [P_bound] * R
            else:
                P_bounds = list(P_bound)
                if len(P_bounds) != R:
                    raise ValueError(
                        f"wise_debug_boundary_stats: len(P_bounds)={len(P_bounds)} != R={R}"
                    )

            last_row = n_sub - 1
            last_col = m_sub - 1

            if boundary_adjacent is None:
                boundary_adjacent = {
                    "top": True,
                    "bottom": True,
                    "left": True,
                    "right": True,
                }
            else:
                boundary_adjacent = {
                    "top": bool(boundary_adjacent.get("top", False)),
                    "bottom": bool(boundary_adjacent.get("bottom", False)),
                    "left": bool(boundary_adjacent.get("left", False)),
                    "right": bool(boundary_adjacent.get("right", False)),
                }

            total_positions = 0
            boundary_hits_row = 0
            boundary_hits_col = 0
            boundary_hits_corner = 0

            per_ion_counts: Dict[int, Dict[str, Any]] = {
                ion: {
                    "total": 0,
                    "row": 0,
                    "col": 0,
                    "corner": 0,
                    "inner_hits": 0,
                    "outer_hits": 0,
                }
                for ion in ions
            }

            ions_set_local = set(ions)

            for r in range(R):
                ions_r: Set[int] = set()
                if inner_pairs is not None and r < len(inner_pairs):
                    for (i1, i2) in inner_pairs[r]:
                        ions_r.add(i1)
                        ions_r.add(i2)
                if outer_pairs is not None and r < len(outer_pairs):
                    for (i1, i2) in outer_pairs[r]:
                        ions_r.add(i1)
                        ions_r.add(i2)
                if not ions_r:
                    ions_r = ions_set_local
                else:
                    ions_r &= ions_set_local

                if not ions_r:
                    continue

                inner_ions_r: Set[int] = set()
                outer_ions_r: Set[int] = set()
                if inner_pairs is not None and r < len(inner_pairs):
                    for (i1, i2) in inner_pairs[r]:
                        inner_ions_r.add(i1)
                        inner_ions_r.add(i2)
                if outer_pairs is not None and r < len(outer_pairs):
                    for (i1, i2) in outer_pairs[r]:
                        outer_ions_r.add(i1)
                        outer_ions_r.add(i2)

                p_final = P_bounds[r]

                for ion in ions_r:
                    if ion not in per_ion_counts:
                        per_ion_counts[ion] = {
                            "total": 0,
                            "row": 0,
                            "col": 0,
                            "corner": 0,
                            "inner_hits": 0,
                            "outer_hits": 0,
                        }
                    for d in range(n_sub):
                        for c in range(m_sub):
                            v = var_a(r, p_final, d, c, ion)
                            if not lit_true(v):
                                continue

                            total_positions += 1
                            per_ion_counts[ion]["total"] += 1

                            on_top = (d == 0 and boundary_adjacent["top"])
                            on_bottom = (d == last_row and boundary_adjacent["bottom"])
                            on_left = (c == 0 and boundary_adjacent["left"])
                            on_right = (c == last_col and boundary_adjacent["right"])

                            on_row_boundary = on_top or on_bottom
                            on_col_boundary = on_left or on_right

                            if on_row_boundary:
                                boundary_hits_row += 1
                                per_ion_counts[ion]["row"] += 1
                            if on_col_boundary:
                                boundary_hits_col += 1
                                per_ion_counts[ion]["col"] += 1
                            if on_row_boundary and on_col_boundary:
                                boundary_hits_corner += 1
                                per_ion_counts[ion]["corner"] += 1

                            if ion in inner_ions_r:
                                per_ion_counts[ion]["inner_hits"] += 1
                            if ion in outer_ions_r:
                                per_ion_counts[ion]["outer_hits"] += 1

            total_positions = max(total_positions, 1)

            summary = {
                "label": label,
                "n_sub": n_sub,
                "m_sub": m_sub,
                "R": R,
                "P_bounds": P_bounds,
                "total_positions": total_positions,
                "boundary_hits_row": boundary_hits_row,
                "boundary_hits_col": boundary_hits_col,
                "boundary_hits_corner": boundary_hits_corner,
                "frac_row": boundary_hits_row / total_positions,
                "frac_col": boundary_hits_col / total_positions,
                "frac_corner": boundary_hits_corner / total_positions,
                "boundary_adjacent": boundary_adjacent,
                "per_ion_counts": per_ion_counts,
            }

            wise_logger.debug(
                "[WISE-DEBUG] boundary-stats %s: subgrid=%dx%d, pos=%d, row_hits=%d (%.3f), "
                "col_hits=%d (%.3f), corner_hits=%d (%.3f), adjacent=%s",
                label,
                n_sub,
                m_sub,
                total_positions,
                boundary_hits_row,
                summary["frac_row"],
                boundary_hits_col,
                summary["frac_col"],
                boundary_hits_corner,
                summary["frac_corner"],
                boundary_adjacent,
            )

            worst = sorted(
                per_ion_counts.items(),
                key=lambda kv: (kv[1]["row"] + kv[1]["col"]),
                reverse=True,
            )[:5]
            for ion, stats in worst:
                if stats["total"] == 0:
                    continue
                fr = stats["row"] / stats["total"]
                fc = stats["col"] / stats["total"]
                wise_logger.debug(
                    "[WISE-DEBUG]   ion %d: total=%d, row=%d (%.3f), col=%d (%.3f), corner=%d, inner_hits=%d, outer_hits=%d",
                    ion,
                    stats["total"],
                    stats["row"],
                    fr,
                    stats["col"],
                    fc,
                    stats["corner"],
                    stats["inner_hits"],
                    stats["outer_hits"],
                )

            return summary

        def _enumerate_pmax_configs(
            P_min: int,
            P_max_limit: int,
            step: int = 1,
            *,
            capacity_steps: int = 6,
            capacity_min: float = 0.0,
        ) -> Iterable[Tuple[int, float]]:
            """
            Enumerate (P_max, boundary_capacity_factor) pairs. The capacity factor
            scales the number of ions that are forced into boundary bands for
            CROSS_BOUNDARY constraints. The final factor is capacity_min (typically 0).
            Each worker now performs its own Σ_r P_r search, so we no longer enumerate
            sum_bound_B here.
            """
            if capacity_steps <= 1:
                factors = [1.0]
            else:
                factors = [
                    max(
                        capacity_min,
                        1.0 - i * (1.0 - capacity_min) / (capacity_steps - 1),
                    )
                    for i in range(capacity_steps)
                ]

            for P_max in range(P_min, P_max_limit + 1, max(1, step)):
                for factor in factors:
                    yield (P_max, factor)

        base_pmax = max(base_pmax_in, 1)
        limit_pmax = base_pmax + n + m
        configs = list(
            _enumerate_pmax_configs(
                base_pmax,
                limit_pmax,
                step=max(int(np.floor((limit_pmax - base_pmax) / 4)), 1),
                capacity_steps=6,
                capacity_min=0.0,
            )
        )
        if not configs:
            raise NoFeasibleLayoutError("No feasible layout: empty SAT configuration set.")

        if DEBUG_DIAG:
            wise_logger.info(
                "[WISE] launching SAT pool over %d configs (P_max∈[%d, %d])",
                len(configs),
                base_pmax,
                limit_pmax,
            )

        try:
            pool_context = mp.get_context("fork")
        except ValueError:
            pool_context = mp.get_context()

        try:
            available_cpus = pool_context.cpu_count()
        except (AttributeError, NotImplementedError):
            try:
                available_cpus = mp.cpu_count()
            except (AttributeError, NotImplementedError):
                available_cpus = 1
        if available_cpus is None or available_cpus <= 0:
            available_cpus = 1

        # Use a conservative helper to decide how many worker processes
        # the SAT pool is allowed to spawn. This prevents runaway process
        # creation when _optimal_QMR_for_WISE is called from within an
        # outer ProcessPoolExecutor.
        max_workers = _wise_safe_sat_pool_workers(len(configs))

        results: List[Dict[str, Any]] = []
        solver_timeout = max_rc2_time if bt_soft_enabled else max_sat_time
        global_budget_s = (
            solver_timeout * 2.0 if (solver_timeout is not None and solver_timeout > 0) else None
        )

        manager = pool_context.Manager()
        sat_children = manager.dict()
        rc2_children = manager.dict()
        stop_event = manager.Event()

        start_pool = time.time()
        progress_dir = tempfile.mkdtemp(prefix="wise_sat_pool_")
        progress_paths: Dict[int, str] = {}
        executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=pool_context)
        futures: List[Tuple[int, Tuple[int, float], Any]] = []
        timed_out = False
        try:
            for idx, cfg in enumerate(configs):
                progress_path = os.path.join(progress_dir, f"cfg_{idx}.pkl")
                progress_paths[idx] = progress_path
                fut = executor.submit(
                    _wise_sat_config_worker,
                    cfg,
                    context=builder_ctx,
                    optimize_round_start=optimize_round_start,
                    max_sat_time=max_sat_time,
                    max_rc2_time=max_rc2_time,
                    boundary_adjacent=boundary_adjacent,
                    cross_boundary_prefs=cross_boundary_prefs,
                    ignore_initial_reconfig=ignore_initial_reconfig,
                    progress_path=progress_path,
                    bt_soft_weight=bt_soft_weight_value,
                    sat_children=sat_children,
                    rc2_children=rc2_children,
                    stop_event=stop_event,
                )
                futures.append((idx, cfg, fut))

            while True:
                unfinished = [1 for _, _, fut in futures if not fut.done()]
                if not unfinished:
                    break
                if (
                    global_budget_s is not None
                    and (time.time() - start_pool) >= global_budget_s
                ):
                    if DEBUG_DIAG:
                        wise_logger.info(
                            "[WISE] global SAT pool budget exhausted; collecting best-so-far results and cancelling remaining workers."
                        )
                    timed_out = True
                    # Cooperatively ask inner workers to stop, then cancel futures.
                    stop_event.set()
                    for _, _, fut in futures:
                        fut.cancel()
                    break
                time.sleep(0.05)

            for idx, cfg, fut in futures:
                if fut.done():
                    try:
                        res = fut.result()
                        results.append(res)
                        if DEBUG_DIAG:
                            usage = res.get("per_round_usage") or [1e9]
                            sum_usage = int(sum(usage[(optimize_round_start * (R > 1)):]))
                            wise_logger.info(
                                "[WISE] pool result: P_max=%s, sum_usage=%d cap_factor=%.2f, status=%s, sat=%s",
                                res.get("P_max"),
                                sum_usage,
                                res.get("boundary_capacity_factor", float("nan")),
                                res.get("status"),
                                res.get("sat"),
                            )
                    except Exception as e:
                        if DEBUG_DIAG:
                            wise_logger.warning("[WISE] config %s crashed: %s", cfg, e)
                else:
                    progress_path = progress_paths.get(idx)
                    if progress_path and os.path.exists(progress_path):
                        try:
                            with open(progress_path, "rb") as f:
                                res = pickle.load(f)
                            results.append(res)
                            if DEBUG_DIAG:
                                usage = res.get("per_round_usage") or [1e9]
                                sum_usage = int(sum(usage[(optimize_round_start * (R > 1)):]))
                                wise_logger.info(
                                    "[WISE] partial result (timeout): P_max=%s, sum_usage=%d cap_factor=%.2f, status=%s, sat=%s",
                                    res.get("P_max"),
                                    sum_usage,
                                    res.get("boundary_capacity_factor", float("nan")),
                                    res.get("status"),
                                    res.get("sat")
                                )
                        except Exception:
                            pass
        finally:
            try:
                if "executor" in locals() and executor is not None:
                    procs = getattr(executor, "_processes", None)

                    if procs:
                        # First, terminate any live workers
                        for proc in list(procs.values()):
                            if proc.is_alive():
                                proc.terminate()

                        # Wait briefly for clean exit
                        for proc in list(procs.values()):
                            try:
                                proc.join(timeout=1.0)
                            except Exception:
                                pass

                        # Escalate to kill if still alive
                        for proc in list(procs.values()):
                            if proc.is_alive():
                                try:
                                    proc.kill()
                                except Exception:
                                    pass

                    # Release executor resources and cancel any not-yet-started futures.
                    executor.shutdown(wait=False, cancel_futures=True)

                # If we hit the global timeout, aggressively kill any known SAT/RC2 helpers.
                if timed_out:
                    try:
                        child_pids = list(sat_children.values()) + list(rc2_children.values())
                    except Exception:
                        child_pids = []

                    for pid in child_pids:
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        except Exception:
                            pass

                    # short grace
                    time.sleep(0.5)

                    for pid in child_pids:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        except Exception:
                            pass

            except Exception as e:
                wise_logger.error(
                    "[WISE] error during forced pool cleanup: %s", e
                )

        sat_results = [
            r for r in results if r.get("sat") and r.get("status") == "ok"
        ]

        if not sat_results:
            raise NoFeasibleLayoutError(
                f"No feasible layout for any Σ_r P_r bound over {len(configs)} configs."
            )

        def _score_config(res: Dict[str, Any]) -> Tuple[int, float, int]:
            usage = res.get("per_round_usage") or [1e9]
            sum_usage = int(sum(usage[(optimize_round_start * (R > 1)):]))
            return (
                -res["boundary_capacity_factor"],
                sum_usage,
                res["P_max"]
            )

        best_res = min(sat_results, key=_score_config)
        best_usage = best_res.get("per_round_usage") or []
        best_sum_bound = int(sum(best_usage))
        chosen_boundary_capacity_factor = best_res["boundary_capacity_factor"]
        P_max = best_res["P_max"]
        sat_model_star = best_res.get("model")

        if sat_model_star is None:
            raise NoFeasibleLayoutError(
                "SAT pool returned no model for the chosen configuration."
            )

        rounds_under_sum = max(1, R - optimize_round_start)
        sum_bound_B = rounds_under_sum * P_max

        (
            _,
            vpool_sat,
            ions_sat,
            _,
            _,
            _,
        ) = _wise_build_structural_cnf(
            builder_ctx,
            P_max,
            sum_bound_B=sum_bound_B,
            use_wcnf=False,
            add_boundary_soft=False,
            phase_label=f"ΣP<={sum_bound_B}/REBUILD",
            optimize_round_start=optimize_round_start,
            debug_core=False,
            core_granularity="coarse",
            debug_skip_cardinality=False,
            boundary_adjacent=boundary_adjacent,
            cross_boundary_prefs=cross_boundary_prefs,
            boundary_capacity_factor=chosen_boundary_capacity_factor,
            bt_soft_weight=bt_soft_weight_value,
        )

        if DEBUG_DIAG:
            wise_logger.info(
                "[WISE] chosen config: P_max=%d, cap_factor=%.2f, ΣP=%d",
                P_max,
                chosen_boundary_capacity_factor,
                best_sum_bound,
            )

        def var_a_sat(r, p, d, c, ion):
            return vpool_sat.id(("a", r, p, d, c, ion))

        if DEBUG_DIAG:
            wise_logger.info("[WISE] minimal ΣP found: %d (P_max=%d)", best_sum_bound, P_max)
        pass_horizon = P_max
        P_bounds = ([pass_horizon + n + m] * optimize_round_start + [pass_horizon] * (R - optimize_round_start))

        layouts_before: List[np.ndarray] = []
        cur = A_in.copy()

        for r in range(R):
            nxt = np.empty_like(cur)
            for d in range(n):
                for c in range(m):
                    found = None
                    for ion in ions_sat:
                        if var_a_sat(r, pass_horizon, d, c, ion) in sat_model_star:
                            found = ion
                            break
                    if found is None:
                        raise RuntimeError(
                            f"Could not reconstruct cell (round={r}, d={d}, c={c})"
                        )
                    nxt[d, c] = found
            layouts_before.append(nxt)
            cur = nxt

        _ = wise_debug_boundary_stats(
            label="BEFORE MAXSAT",
            model=sat_model_star,
            vpool=vpool_sat,
            var_a=var_a_sat,
            ions=ions,
            n_sub=n,
            m_sub=m,
            R=R,
            P_bound=P_bounds,
            inner_pairs=P_arr,
            outer_pairs=outer_pairs,
            boundary_adjacent=boundary_adjacent,
        )
        # -------------------------------
        # Level 3: MaxSAT refinement under ΣP*
        # -------------------------------
        ENABLE_MAXSAT = False
        if ENABLE_MAXSAT:
            if DEBUG_DIAG:
                wise_logger.info(
                    "[WISE] building WCNF at ΣP*=%d, P_max=%d for MaxSAT...",
                    best_sum_bound,
                    pass_horizon,
                )
            t_build_start = time.time()
            wcnf, vpool_w, ions_w, var_a_w, _, _ = _wise_build_structural_cnf(
                builder_ctx,
                pass_horizon,
                sum_bound_B=best_sum_bound,
                use_wcnf=True,
                add_boundary_soft=True,
                phase_label=f"ΣP*={best_sum_bound}/WCNF",
                optimize_round_start=optimize_round_start,
                debug_skip_cardinality=False,
                boundary_adjacent=boundary_adjacent,
                cross_boundary_prefs=cross_boundary_prefs,
                boundary_capacity_factor=chosen_boundary_capacity_factor,
                bt_soft_weight=bt_soft_weight_value,
            )
            t_build_end = time.time()

            if DEBUG_DIAG:
                wise_logger.info(
                    "[WISE] WCNF built: vars=%d, hard=%d, soft=%d, time=%.3fs",
                    wcnf.nv,
                    len(wcnf.hard),
                    len(wcnf.soft),
                    t_build_end - t_build_start,
                )

            rc2 = RC2(wcnf)
            model_rc2 = rc2.compute()
            cost_rc2 = rc2.cost if model_rc2 is not None else None
            status_rc2 = "ok" if model_rc2 is not None else "error"

            if DEBUG_DIAG:
                wise_logger.info("[WISE] RC2 status=%s, opt_cost=%s", status_rc2, cost_rc2)

            if status_rc2 == "ok" and model_rc2 is not None:
                model_used = model_rc2
                vpool_used = vpool_w
                var_a_used = var_a_w
                ions_used = ions_w
                if DEBUG_DIAG:
                    wise_logger.info("[WISE] using RC2 MaxSAT model at P*")
            else:
                model_used = sat_model_star
                vpool_used = vpool_sat
                var_a_used = var_a_sat
                ions_used = ions_sat
                if DEBUG_DIAG:
                    wise_logger.info(
                        "[WISE] MaxSAT unavailable (timeout/error); falling back to SAT model at P*."
                    )
        else:
            model_used = sat_model_star
            vpool_used = vpool_sat
            var_a_used = var_a_sat
            ions_used = ions_sat
            if DEBUG_DIAG:
                wise_logger.info("[WISE] MaxSAT disabled; using SAT model at P*")

        # Decide which ions are "core" for this slice.
        core_ions_this_slice = ions   # or a filtered subset if you prefer

        _ = wise_debug_boundary_stats(
            label="AFTER MAXSAT",
            model=model_used,
            vpool=vpool_used,
            var_a=var_a_used,
            ions=ions_used,
            n_sub=n,
            m_sub=m,
            R=R,
            P_bound=P_bounds,
            inner_pairs=P_arr,
            outer_pairs=outer_pairs,
            boundary_adjacent=boundary_adjacent,
        )

        # -------------------------------
        # Decode layouts a[r,P_max] from model_used
        # -------------------------------
        model_set = {lit for lit in model_used if lit > 0}

        def lit_true(v: int) -> bool:
            return v in model_set

        layouts: List[np.ndarray] = []
        cur = A_in.copy()

        for r in range(R):
            nxt = np.empty_like(cur)
            for d in range(n):
                for c in range(m):
                    found = None
                    for ion in ions_used:
                        if lit_true(var_a_used(r, P_bounds[r], d, c, ion)):
                            found = ion
                            break
                    if found is None:
                        raise RuntimeError(
                            f"Could not reconstruct cell (round={r}, d={d}, c={c})"
                        )
                    nxt[d, c] = found
            layouts.append(nxt)
            cur = nxt

        # After RC2:
        schedule = _wise_decode_schedule_from_model(
            model=model_used,
            vpool=vpool_used,
            n=n,
            m=m,
            R=R,
            P_bound=pass_horizon,
            ignore_initial_reconfig=ignore_initial_reconfig,
        )

        row_offset, col_offset = grid_origin
        if row_offset != 0 or col_offset != 0:
            for round_schedule in schedule:
                for pass_info in round_schedule:
                    if "h_swaps" in pass_info:
                        pass_info["h_swaps"] = [
                            (r + row_offset, c + col_offset) for (r, c) in pass_info["h_swaps"]
                        ]
                    if "v_swaps" in pass_info:
                        pass_info["v_swaps"] = [
                            (r + row_offset, c + col_offset) for (r, c) in pass_info["v_swaps"]
                        ]

        per_round_z = _wise_extract_round_pass_usage(
            model_used,
            vpool_used,
            R,
            pass_horizon,
            ignore_initial_reconfig,
            n,
            m,
        )
        sum_all = sum(per_round_z)
        sum_tail = sum(per_round_z[optimize_round_start:])

        wise_logger.info(
            "[WISE] ΣP per round: %s, Σ_all=%d, Σ_tail=%d, best_sum_bound=%d",
            per_round_z,
            sum_all,
            sum_tail,
            best_sum_bound,
        )

        def _wise_assert_bt_consistency(layouts, BT, R, n, m, logger):
            """
            Check BT pin consistency against layouts.

            Logs errors instead of raising and returns True/False.
            """
            if BT is None or len(BT) == 0:
                return True

            ok = True

            if len(layouts) < R:
                logger.error(
                    "[WISE] BT consistency check: expected at least %d layouts, got %d",
                    R, len(layouts),
                )
                return False

            for r in range(R):
                layout_r = np.asarray(layouts[r], dtype=int)
                if layout_r.shape != (n, m):
                    logger.error(
                        "[WISE] BT consistency check: layout[%d] has shape %s, expected (%d, %d)",
                        r, layout_r.shape, n, m,
                    )
                    ok = False
                    continue

                bt_round = BT[r] if r < len(BT) else {}
                for ion, (d, c) in bt_round.items():
                    if not (0 <= d < n and 0 <= c < m):
                        logger.error(
                            "[WISE] BT consistency check: BT[%d] pins ion %d to out-of-bounds cell (d=%d, c=%d)",
                            r, ion, d, c,
                        )
                        ok = False
                        continue

                    found = int(layout_r[d, c])
                    if found != ion:
                        logger.error(
                            "[WISE] BT consistency mismatch at round %d: expected ion %d at (d=%d, c=%d), but found %d",
                            r, ion, d, c, found,
                        )
                        ok = False

            return ok

        bt_ok = _wise_assert_bt_consistency(layouts, BT, R, n, m, wise_logger)
        if not bt_ok:
            wise_logger.warning(
                "[WISE] BT consistency check failed; proceeding with returned layouts anyway"
            )

        return layouts, schedule, pass_horizon

    # ------------------------------------------------------------------
    # Odd-even reconfiguration execution
    # ------------------------------------------------------------------

    @classmethod
    def _runOddEvenReconfig(
        cls,
        wiseArch: "QCCDWiseArch",
        arrangement: Mapping["Trap", Sequence["Ion"]],
        oldAssignment: Sequence[Sequence[int]],
        newAssignment: Sequence[Sequence[int]],
        ignoreSpectators: bool = False,
        sat_schedule: Optional[List[Dict[str, Any]]] = None,
        initial_placement: bool = False,
    ) -> Tuple[Mapping[int, float], float]:
        # Schedule-aware reconfiguration fallback when SAT results are available.
        heatingRates: Dict[int, float] = {}
        for _, ions in arrangement.items():
            for ion in ions:
                heatingRates[ion.idx] = 0.0

        heatingRates: Mapping[int, float] = {}

        # Initialise from arrangement (all ions we know about physically)
        for _, ions in arrangement.items():
            for ion in ions:
                heatingRates[ion.idx] = 0.0

        # ALSO initialise any ions present in the layout matrices A/T
        try:
            A_all = np.array(oldAssignment, dtype=int)
            for ion_id in set(int(x) for x in A_all.flatten()):
                if ion_id not in heatingRates:
                    # These might be spectators / padding but we still give them an entry
                    heatingRates[ion_id] = 0.0
        except Exception as e:
            print("[DEBUG _runOddEvenReconfig] error while initialising heatingRates from assignment:", repr(e))
        timeElapsed = 0.0

        # Timing / heating constants from transport.py data classes
        row_swap_time = (
            _TMove.MOVING_TIME
            + _TMerge.MERGING_TIME
            + _TCrystalRotation.ROTATION_TIME
            + _TSplit.SPLITTING_TIME
            + _TMove.MOVING_TIME
        )
        row_swap_heating = (
            _TMove.MOVING_TIME * _TMove.HEATING_RATE
            + _TMerge.MERGING_TIME * _TMerge.HEATING_RATE
            + _TCrystalRotation.ROTATION_TIME * _TCrystalRotation.ROTATION_HEATING
            + _TSplit.SPLITTING_TIME * _TSplit.HEATING_RATE
            + _TMove.MOVING_TIME * _TMove.HEATING_RATE
        )
        col_swap_time = (2 * _TJunctionCrossing.CROSSING_TIME) + (
            4 * _TJunctionCrossing.CROSSING_TIME + _TMove.MOVING_TIME
        ) * 2
        col_swap_heating_rate = (
            6 * _TJunctionCrossing.CROSSING_TIME * _TJunctionCrossing.CROSSING_HEATING
            + _TMove.MOVING_TIME * _TMove.HEATING_RATE
        )

        n = wiseArch.n  # rows
        m = wiseArch.m * wiseArch.k  # full columns (matches CNF m if built that way)
        k = wiseArch.k  # column stride for junction batching

        A = np.array(oldAssignment, dtype=int)  # current layout
        T = np.array(newAssignment, dtype=int)  # target layout

        spectatorIons: List[int] = []
        for ions in arrangement.values():
            spectatorIons.extend([ion.idx for ion in ions if isinstance(ion, SpectatorIon)])

        # ==========================================================
        # Phase A: parallel split (this is arch-specific overhead)
        # ==========================================================
        timeElapsed += _TSplit.SPLITTING_TIME
        for idx in heatingRates.keys():
            heatingRates[idx] += _TSplit.HEATING_RATE * _TSplit.SPLITTING_TIME

        # If we have a SAT schedule, use it directly and SKIP Phases B/C/D.
        if sat_schedule is not None:
            acc_passes = 0

            for pass_idx, info in enumerate(sat_schedule):
                phase = info.get("phase", "H")
                h_swaps = info.get("h_swaps", [])
                v_swaps = info.get("v_swaps", [])

                # NOTE: to stay consistent with the SAT model, we SHOULD NOT
                #       skip swaps just because both ions are spectators.
                #       If you want spectators to be immobile, that must be
                #       encoded in the SAT itself.
                did_any_swap = False

                if phase == "H":
                    # All horizontal swaps in this pass are parallel
                    for (r, c) in h_swaps:
                        a = int(A[r, c])
                        b = int(A[r, c + 1])
                        if ignoreSpectators and (a in spectatorIons and b in spectatorIons):
                            continue
                        # Perform swap
                        A[r, c], A[r, c + 1] = b, a
                        heatingRates[a] += row_swap_heating
                        heatingRates[b] += row_swap_heating
                        did_any_swap = True

                    if did_any_swap:
                        timeElapsed += row_swap_time
                        acc_passes += 1

                elif phase == "V":
                    # All vertical swaps in this pass are parallel
                    for (r, c) in v_swaps:
                        a = int(A[r, c])
                        b = int(A[r + 1, c])
                        if ignoreSpectators and (a in spectatorIons and b in spectatorIons):
                            continue
                        A[r, c], A[r + 1, c] = b, a
                        heatingRates[a] += col_swap_heating_rate
                        heatingRates[b] += col_swap_heating_rate
                        did_any_swap = True

                    if did_any_swap:
                        timeElapsed += col_swap_time
                        acc_passes += 1

                else:
                    # Should never happen; phase is either H or V
                    pass

            # After executing the SAT schedule, check we reached the target.
            if not np.array_equal(A, T):
                raise RuntimeError("SAT schedule did not realise target layout")

            return heatingRates, timeElapsed

        # ---------- helper: odd-even passes ----------
        def row_pass_by_rank(even_phase: bool, row_rank: List[Dict[int, int]]) -> bool:
            maxSwapsInRow = 0
            start = 0 if even_phase else 1
            phase_label = "even" if even_phase else "odd"
            swaps_this_phase = 0

            for r in range(n):
                swapsInRow = 0
                rank = row_rank[r]
                for c in range(start, m - 1, 2):
                    a = int(A[r, c])
                    b = int(A[r, c + 1])
                    if rank[a] > rank[b]:
                        A[r, c], A[r, c + 1] = b, a
                        heatingRates[a] += row_swap_heating
                        heatingRates[b] += row_swap_heating
                        swapsInRow += 1
                if swapsInRow > maxSwapsInRow:
                    maxSwapsInRow = swapsInRow
                swaps_this_phase += swapsInRow

            return maxSwapsInRow > 0

        def col_bucket_pass(
            even_phase: bool, bucket_mod: int, ion_to_dest_row: Dict[int, int]
        ) -> bool:
            maxSwapsInCol = 0
            start = 0 if even_phase else 1
            phase_label = "even" if even_phase else "odd"
            swaps_this_phase = 0

            for c in range(bucket_mod, m, k):
                swapsInCol = 0
                for r in range(start, n - 1, 2):
                    a = int(A[r, c])
                    b = int(A[r + 1, c])
                    if ion_to_dest_row[a] > ion_to_dest_row[b]:
                        A[r, c], A[r + 1, c] = b, a
                        heatingRates[a] += col_swap_heating_rate
                        heatingRates[b] += col_swap_heating_rate
                        swapsInCol += 1
                if swapsInCol > maxSwapsInCol:
                    maxSwapsInCol = swapsInCol
                swaps_this_phase += swapsInCol

            return maxSwapsInCol > 0

        # ---------- Phase B greedy target layout ----------
        B = GlobalReconfigurations.phaseB_greedy_layout(A, T)[1]

        # ---------- destination row/col maps (ion -> dest row/col) ----------
        ion_to_dest_row: Dict[int, int] = {}
        ion_to_dest_col: Dict[int, int] = {}
        for r in range(n):
            for c in range(m):
                ion = int(T[r, c])
                ion_to_dest_row[ion] = r
                ion_to_dest_col[ion] = c

        # Sanity 1: each row of B must be a permutation of the row of A
        try:
            for r in range(n):
                if set(B[r]) != set(A[r]):
                    raise AssertionError("row permutation mismatch")
        except AssertionError:
            print(
                "[DEBUG _runOddEvenReconfig][Phase B] assertion failed: "
                "B row not a permutation of A row",
                "row_idx=", r,
                "A_row=", list(A[r]),
                "B_row=", list(B[r]),
            )
            raise

        # Sanity 2: for each column of B, destination rows must be all distinct
        try:
            for c in range(m):
                dest_rows_col = [ion_to_dest_row[int(B[r, c])] for r in range(n)]
                if len(set(dest_rows_col)) != n:
                    raise AssertionError("dest_rows column clash")
        except AssertionError:
            print(
                "[DEBUG _runOddEvenReconfig][Phase B] assertion failed: "
                "duplicate dest_rows in a column",
                "m=", m,
                "n=", n,
                "dest_rows_col0=",
                [ion_to_dest_row[int(B[r, 0])] for r in range(n)] if m > 0 else [],
            )
            raise

        # Phase B target row order is exactly B
        desired_row_order = B.copy()

        # Row ranks for Phase B permutation
        row_rank_phaseB: List[Dict[int, int]] = []
        for r in range(n):
            row_rank_phaseB.append(
                {ion: idx for idx, ion in enumerate(desired_row_order[r])}
            )

        acc_cost = 0

        # Execute ≤ m odd–even steps to realise the permutation per row
        for _ in range(m):
            oddpass = row_pass_by_rank(True, row_rank_phaseB)
            evenpass = row_pass_by_rank(False, row_rank_phaseB)
            timeElapsed += oddpass * row_swap_time
            timeElapsed += evenpass * row_swap_time
            acc_cost += int(oddpass) + int(evenpass)

        # ==========================================================
        # Phase C: vertical odd–even with k-way parallel buckets.
        # ==========================================================
        for t in range(k):
            for _ in range(n):
                oddpass = col_bucket_pass(True, t, ion_to_dest_row)
                evenpass = col_bucket_pass(False, t, ion_to_dest_row)
                timeElapsed += oddpass * col_swap_time
                timeElapsed += evenpass * col_swap_time
                acc_cost += int(oddpass) + int(evenpass)
            if t < k - 1:
                # "Parallel row reconfig"
                timeElapsed += k * row_swap_time
                for idx in heatingRates.keys():
                    heatingRates[idx] += row_swap_heating

        # ==========================================================
        # Phase D: final row-wise odd–even to exact target order.
        # ==========================================================
        row_rank_final: List[Dict[int, int]] = []
        for r in range(n):
            row_rank_final.append({ion: idx for idx, ion in enumerate(T[r, :])})

        for _ in range(m):
            oddpass = row_pass_by_rank(True, row_rank_final)
            evenpass = row_pass_by_rank(False, row_rank_final)
            timeElapsed += oddpass * row_swap_time
            timeElapsed += evenpass * row_swap_time
            acc_cost += int(oddpass) + int(evenpass)

        # Final sanity: how close are we to the target layout?
        try:
            final_diff = int(np.sum(A != T))
        except Exception:
            final_diff = "NA"

        if not initial_placement:
            print(
                f"RECONFIGURATION: {acc_cost} passes were needed for the "
                f"current reconfiguration round, taking {timeElapsed} time and "
                f"{heatingRates} heating"
            )

        return heatingRates, timeElapsed

    # ------------------------------------------------------------------
    # Phase-B layout helper
    # ------------------------------------------------------------------

    @staticmethod
    def phaseB_greedy_layout(
        A_in: np.ndarray,
        T_in: np.ndarray,
    ) -> Tuple[int, np.ndarray]:
        """
        Phase-B layout helper.

        Given current layout A_in (n×m) and target layout T_in (n×m),
        construct a row-wise permutation B of A_in such that:

          • Each column of B contains ions with *all distinct* destination rows
            (according to T_in).
          • Returns (max_horizontal_displacement, B), where the displacement
            is |orig_col - new_col| measured per ion within its row.

        This assumes A_in and T_in contain exactly the same ion IDs.
        """
        A = np.asarray(A_in, dtype=int)
        T = np.asarray(T_in, dtype=int)
        n, m = A.shape
        if T.shape != (n, m):
            raise ValueError("phaseB_greedy_layout: A and T must have same shape")

        # ---- dest row for every ion id (from T) ----
        ion_to_dest_row: Dict[int, int] = {}
        for r in range(n):
            for c in range(m):
                ion_to_dest_row[int(T[r, c])] = r

        # sanity: same ions
        if set(A.flatten()) != set(T.flatten()):
            raise ValueError(
                "phaseB_greedy_layout: A and T must contain the same ion IDs"
            )

        # ---- counts[r, d] & ions_per_row_dest[r][d] ----
        counts = np.zeros((n, n), dtype=int)
        ions_per_row_dest: List[List[List[int]]] = [[[] for _ in range(n)] for _ in range(n)]

        # inverse position map per row (ion -> original column)
        invpos: List[Dict[int, int]] = []
        for r in range(n):
            mp_r: Dict[int, int] = {}
            for c in range(m):
                ion = int(A[r, c])
                mp_r[ion] = c
            invpos.append(mp_r)

        for r in range(n):
            for c in range(m):
                ion = int(A[r, c])
                d = ion_to_dest_row[ion]
                counts[r, d] += 1
                ions_per_row_dest[r][d].append(ion)

        # consistency check between counts and buckets
        for r in range(n):
            for d in range(n):
                if counts[r, d] != len(ions_per_row_dest[r][d]):
                    raise RuntimeError(
                        "phaseB_greedy_layout: counts/buckets mismatch at "
                        f"(row={r}, dest={d}): counts={counts[r,d]}, "
                        f"bucket_len={len(ions_per_row_dest[r][d])}"
                    )

        # row/col sums must be m
        for r in range(n):
            if counts[r, :].sum() != m:
                raise RuntimeError(
                    f"phaseB_greedy_layout: row {r} total count {counts[r,:].sum()} != m={m}"
                )
        for d in range(n):
            if counts[:, d].sum() != m:
                raise RuntimeError(
                    f"phaseB_greedy_layout: dest-row {d} total count {counts[:,d].sum()} != m={m}"
                )

        desired = np.zeros_like(A)
        max_disp = 0

        # ---------- internal helper: perfect matching from counts ----------
        def _perfect_matching_from_counts(counts_mat: np.ndarray) -> List[int]:
            """
            Given current counts[r, d] (non-negative ints) with
            sum_d counts[r,d] == m' for all r and sum_r counts[r,d] == m' for all d,
            find a perfect matching on the graph of edges where counts[r,d] > 0.

            Returns match_row_to_dest[r] = d for all rows r.
            Raises RuntimeError if no perfect matching exists.
            """
            nn = counts_mat.shape[0]
            # adjacency: neighbors[r] = [d | counts[r,d]>0]
            neighbors: List[List[int]] = []
            for rr in range(nn):
                nbrs = [dd for dd in range(nn) if counts_mat[rr, dd] > 0]
                neighbors.append(nbrs)

            match_to_row = [-1] * nn  # dest d -> row r

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
                    raise RuntimeError(
                        "phaseB_greedy_layout: no perfect matching for current counts"
                    )

            # convert dest->row into row->dest
            match_row_to_dest = [-1] * nn
            for d, r in enumerate(match_to_row):
                if r == -1:
                    continue
                match_row_to_dest[r] = d

            if any(d == -1 for d in match_row_to_dest):
                raise RuntimeError(
                    "phaseB_greedy_layout internal: incomplete matching"
                )
            return match_row_to_dest

        # ---- build columns one by one via perfect matchings ----
        for c in range(m):
            # find any perfect matching consistent with current counts
            match = _perfect_matching_from_counts(counts)

            # reconstruct column c
            for r in range(n):
                d = int(match[r])

                # sanity: we must have capacity here
                if counts[r, d] <= 0:
                    raise RuntimeError(
                        "phaseB_greedy_layout: internal bug — picked (row={r}, dest={d}) "
                        "with zero capacity despite matching only on counts>0; "
                        f"(r={r}, d={d}, counts[r,d]={counts[r,d]})"
                    )

                bucket = ions_per_row_dest[r][d]
                if not bucket:
                    raise RuntimeError(
                        "phaseB_greedy_layout: empty bucket for "
                        f"(row={r}, dest={d}) while reconstructing column {c}; "
                        f"counts[r,d]={counts[r,d]}"
                    )

                # take any ion with dest row d from this row
                ion = bucket.pop()
                counts[r, d] -= 1
                desired[r, c] = ion

                # update displacement
                j0 = invpos[r][ion]
                disp = abs(j0 - c)
                if disp > max_disp:
                    max_disp = disp

        # final sanity checks
        for r in range(n):
            if set(desired[r, :]) != set(A[r, :]):
                raise RuntimeError(
                    f"phaseB_greedy_layout: desired row {r} not a permutation of A row {r}"
                )

        for c in range(m):
            dests_col = [ion_to_dest_row[int(desired[r, c])] for r in range(n)]
            if len(set(dests_col)) != n:
                raise RuntimeError(
                    f"phaseB_greedy_layout: column {c} does not have unique "
                    f"destination rows; dest_rows_col={dests_col}"
                )

        return max_disp, desired


# ---------------------------------------------------------------------------
# Backwards compatibility alias (deprecated)
# ---------------------------------------------------------------------------
GlobalReconfigurations = ReconfigurationPlanner
