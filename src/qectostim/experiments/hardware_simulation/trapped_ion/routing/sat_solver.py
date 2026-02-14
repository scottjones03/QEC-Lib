# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/sat_solver.py
"""
SAT / MaxSAT solver infrastructure for WISE grid reconfiguration.

Extracted from ``qccd_operations.py``.  This module contains:

* ``_WiseSatBuilderContext`` — dataclass holding grid geometry for the CNF builder.
* ``_wise_build_structural_cnf`` — builds the full odd-even sorting-network CNF/WCNF.
* ``_wise_decode_schedule_from_model`` — extracts H/V swap schedule from a SAT model.
* ``_wise_extract_round_pass_usage`` — counts active passes per round.
* ``_wise_sat_config_worker`` — per-configuration binary-search worker.
* ``run_sat_with_timeout_file`` / ``run_rc2_with_timeout_file`` — process-isolated solvers.
* ``NoFeasibleLayoutError`` / ``CoreGroups`` — error and UNSAT-core helpers.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import signal
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import Process, Pipe
from typing import (
    Any,
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
from pysat.card import CardEnc, EncType
from pysat.examples.rc2 import RC2
from pysat.formula import CNF, IDPool, WCNF
from pysat.solvers import Minisat22, Solver

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

WISE_LOGGER_NAME = "wise.qccd.sat"
wise_logger = logging.getLogger(WISE_LOGGER_NAME)
if not wise_logger.handlers:
    wise_logger.addHandler(logging.NullHandler())
wise_logger.propagate = False

# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class NoFeasibleLayoutError(RuntimeError):
    ...

# ---------------------------------------------------------------------------
# UNSAT core helpers
# ---------------------------------------------------------------------------

class CoreGroups:
    """
    Manage assumption selectors for clause families to enable UNSAT core debug.
    Clauses guarded under selector s_g are added as (clause ∨ s_g).
    Solving with assumption ¬s_g activates the group; cores then map back to names.
    """

    def __init__(self, vpool: IDPool, enabled: bool, granularity: str = "coarse"):
        self.vpool = vpool
        self.enabled = enabled
        self.granularity = granularity
        self.sel: Dict[str, int] = {}
        self.meta: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def lit(self, name: str) -> int:
        if name not in self.sel:
            self.sel[name] = self.vpool.id(("grp", name))
        return self.sel[name]

    def add(
        self,
        formula: CNF,
        cl: List[int],
        group: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            formula.append(cl)
            return
        formula.append(cl + [self.lit(group)])
        if meta is not None:
            self.meta[group].append(meta)


# ---------------------------------------------------------------------------
# Worker-count limiter
# ---------------------------------------------------------------------------

def _wise_safe_sat_pool_workers(num_configs: int) -> int:
    """
    Compute a conservative upper bound on the number of worker processes
    used by the SAT pool, to avoid oversubscribing CPUs when this function
    is called from within another ProcessPoolExecutor.

    Priority:
      1. Hard cap (HARD_CAP).
      2. Environment variable WISE_SAT_WORKERS (if present and > 0).
      3. Available CPUs.
      4. Number of configs.
    """
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()

    try:
        available_cpus = ctx.cpu_count()
    except (AttributeError, NotImplementedError):
        try:
            available_cpus = mp.cpu_count()
        except (AttributeError, NotImplementedError):
            available_cpus = 1

    if not isinstance(available_cpus, int) or available_cpus <= 0:
        available_cpus = 1

    # Hard upper bound to avoid explosion under nested parallelism.
    HARD_CAP = 4

    # Base clamp by configs, cpus, and hard cap.
    workers = max(1, min(num_configs, available_cpus, HARD_CAP))

    # Optional environment override to further limit workers.
    env_val = os.environ.get("WISE_SAT_WORKERS")
    if env_val is not None:
        try:
            cap = int(env_val)
            if cap > 0:
                workers = max(1, min(workers, cap))
        except ValueError:
            # Ignore invalid env setting.
            pass

    # If we are on a very small machine (<=2 CPUs), default to single-worker
    # to avoid competing too hard with any outer pools.
    if available_cpus <= 2:
        workers = 1

    return workers


# ---------------------------------------------------------------------------
# SAT builder context
# ---------------------------------------------------------------------------

@dataclass
class _WiseSatBuilderContext:
    A_in: np.ndarray
    BT: List[Dict[int, Tuple[int, int]]]
    P_arr: List[List[Tuple[int, int]]]
    full_P_arr: List[List[Tuple[int, int]]]
    ions: List[int]
    n: int
    m: int
    R: int
    block_cells: List[List[Tuple[int, int]]]
    block_fully_inside: List[bool]
    block_widths: List[int]
    num_blocks: int
    wB_col: int
    wB_row: int
    debug_diag: bool


# ---------------------------------------------------------------------------
# Structural CNF / WCNF builder
# ---------------------------------------------------------------------------

def _wise_build_structural_cnf(
    ctx: _WiseSatBuilderContext,
    P_bound: int,
    sum_bound_B: Optional[int] = None,
    use_wcnf: bool = False,
    add_boundary_soft: bool = False,
    phase_label: str = "",
    debug_skip_pair_constraints: bool = False,
    debug_allow_phase_flips: bool = False,
    optimize_round_start: int = 0,
    debug_core: bool = False,
    core_granularity: str = "coarse",
    debug_skip_cardinality: bool = False,
    debug_disable_pairs_rounds: Optional[Set[int]] = None,
    boundary_adjacent: Optional[Dict[str, bool]] = None,
    cross_boundary_prefs: Optional[List[Dict[int, Set[str]]]] = None,
    boundary_capacity_factor: float = 1.0,
    *,
    bt_soft_weight: int = 0,
):
    A_in = ctx.A_in
    BT = ctx.BT
    DEBUG_DIAG = ctx.debug_diag
    P_arr = ctx.P_arr
    R = ctx.R
    block_cells = ctx.block_cells
    block_fully_inside = ctx.block_fully_inside
    block_widths = ctx.block_widths
    full_P_arr = ctx.full_P_arr
    ions = ctx.ions
    m = ctx.m
    n = ctx.n
    num_blocks = ctx.num_blocks
    wB_col = ctx.wB_col
    wB_row = ctx.wB_row

    vpool = IDPool()

    # ------------- variable helpers -------------

    def var_a(r, p, krow, jcol, ion):
        return vpool.id(("a", r, p, krow, jcol, ion))

    def var_s_h(r, p, krow, jcol):
        return vpool.id(("s_h", r, p, krow, jcol))

    def var_s_v(r, p, krow, jcol):
        return vpool.id(("s_v", r, p, krow, jcol))

    def var_phase(r, p):
        return vpool.id(("phase", r, p))

    def var_row_end(r, ion, d):
        return vpool.id(("row_end", r, ion, d))

    def var_w_end(r, ion, b):
        return vpool.id(("w_end", r, ion, b))

    def var_u(r, p):
        return vpool.id(("u", r, p))

    def is_reserved(r, ion):
        return ion in BT[r]

    def ion_in_full_P_arr(r, ion):
        return any((ion in g) for g in full_P_arr[r])

    def ion_in_minor_P_arr(r, ion):
        return any((ion in g) for g in P_arr[r]) or is_reserved(r, ion)

    # ------------- choose CNF or WCNF -------------

    if use_wcnf:
        formula = WCNF()
    else:
        formula = CNF()

    grp = CoreGroups(
        vpool=vpool,
        enabled=(debug_core and not use_wcnf),
        granularity=core_granularity,
    )

    def add_hard(
        cl: List[int],
        group_name: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if use_wcnf:
            formula.append(cl)
        else:
            grp.add(formula, cl, group_name, meta=meta)

    def add_soft(cl: List[int], weight: int = 1) -> None:
        if use_wcnf:
            formula.append(cl, weight=weight)
        else:
            raise RuntimeError("Soft clauses not allowed in pure CNF mode")

    if debug_disable_pairs_rounds is None:
        disable_pairs_rounds: Set[int] = set()
    else:
        disable_pairs_rounds = set(debug_disable_pairs_rounds)

    P_bounds = ([P_bound + n + m] * optimize_round_start + [P_bound] * (R - optimize_round_start))

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

    if cross_boundary_prefs is None:
        cross_boundary_prefs_norm: List[Dict[int, Set[str]]] = [dict() for _ in range(R)]
    else:
        cross_boundary_prefs_norm = []
        for r in range(R):
            prefs_r = cross_boundary_prefs[r] if r < len(cross_boundary_prefs) else {}
            normalized: Dict[int, Set[str]] = {}
            for ion, dirs in prefs_r.items():
                normalized[ion] = set(dirs)
            cross_boundary_prefs_norm.append(normalized)

    half_h = max(1, n // 2)
    half_w = max(1, m // 2)

    def cells_for_direction(direction: str) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        if direction == "left":
            if not boundary_adjacent["left"]:
                return cells
            for d in range(n):
                for c in range(half_w):
                    cells.append((d, c))
        elif direction == "right":
            if not boundary_adjacent["right"]:
                return cells
            start = max(0, m - half_w)
            for d in range(n):
                for c in range(start, m):
                    cells.append((d, c))
        elif direction == "top":
            if not boundary_adjacent["top"]:
                return cells
            for d in range(half_h):
                for c in range(m):
                    cells.append((d, c))
        elif direction == "bottom":
            if not boundary_adjacent["bottom"]:
                return cells
            start = max(0, n - half_h)
            for d in range(start, n):
                for c in range(m):
                    cells.append((d, c))
        return cells

    # ------------------------------------------------------------------
    # (0) Global permutation: exactly one ion per cell AND each ion in exactly one cell
    # ------------------------------------------------------------------
    if not debug_skip_cardinality:
        # (0a) Exactly one ion per cell (r,p,k,j)
        for r in range(R):
            g_cell = f"CARD_CELL:r{r}"
            for p in range(P_bounds[r] + 1):
                for krow in range(n):
                    for jcol in range(m):
                        lits = [var_a(r, p, krow, jcol, ion) for ion in ions]
                        enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                        for cl in enc.clauses:
                            add_hard(cl, g_cell)

        g_cell_final = "CARD_CELL:FINAL"
        for krow in range(n):
            for jcol in range(m):
                lits = [var_a(R, 0, krow, jcol, ion) for ion in ions]
                enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                for cl in enc.clauses:
                    add_hard(cl, g_cell_final)

        # (0b) Each ion occupies exactly one cell (k,j) at every (r,p)
        for r in range(R):
            g_ion_r = f"CARD_ION:r{r}"
            for p in range(P_bounds[r] + 1):
                for ion in ions:
                    lits = [var_a(r, p, krow, jcol, ion) for krow in range(n) for jcol in range(m)]
                    enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        add_hard(cl, g_ion_r)

        g_ion_final = "CARD_ION:FINAL"
        for ion in ions:
            lits = [var_a(R, 0, krow, jcol, ion) for krow in range(n) for jcol in range(m)]
            enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
            for cl in enc.clauses:
                add_hard(cl, g_ion_final)

    # ------------------------------------------------------------------
    # (1) Initial layout and inter-round chaining
    # ------------------------------------------------------------------

    # (1a) Initial layout: a[0,0] equals A_in
    for krow in range(n):
        for jcol in range(m):
            ion0 = int(A_in[krow, jcol])
            # Ion ion0 must be there
            add_hard([var_a(0, 0, krow, jcol, ion0)], "INIT")
            # No other ion may be there at (0,0)
            for ion in ions:
                if ion != ion0:
                    add_hard([-var_a(0, 0, krow, jcol, ion)], "INIT")

    # (1b) Chaining: a[r+1,0] <-> a[r,P_bounds[r]] for r=0..R-1
    for r in range(R):
        g_chain_r = f"CHAIN:r{r}"
        for krow in range(n):
            for jcol in range(m):
                for ion in ions:
                    a_next = var_a(r + 1, 0, krow, jcol, ion)
                    a_end  = var_a(r, P_bounds[r], krow, jcol, ion)
                    add_hard([-a_next, a_end], g_chain_r)
                    add_hard([-a_end, a_next], g_chain_r)

    # ------------------------------------------------------------------
    # (2) BT pinning at end-of-round (FINAL STATE ONLY)
    # ------------------------------------------------------------------

    use_soft_bt = use_wcnf and bt_soft_weight > 0

    for r in range(R):
        P_final = P_bounds[r]
        for ion in ions:
            if not is_reserved(r, ion):
                continue
            d_fix, c_fix = BT[r][ion]
            pin_lit = var_a(r, P_final, d_fix, c_fix, ion)

            if use_soft_bt:
                # Encourage pinned location but allow solver to deviate if needed.
                add_soft([pin_lit], weight=bt_soft_weight)
                continue

            # HARD BT (existing behaviour)
            add_hard([pin_lit], "BT")
            for krow in range(n):
                for jcol in range(m):
                    if (krow, jcol) != (d_fix, c_fix):
                        add_hard([-var_a(r, P_final, krow, jcol, ion)], "BT")

    # ------------------------------------------------------------------
    # (3) Phase structure: horizontal -> vertical
    # ------------------------------------------------------------------

    # (3a) Monotonicity: phase[r,p] -> phase[r,p+1] for each round r
    if not debug_allow_phase_flips:
        for r in range(R):
            if P_bounds[r] > 1:
                for p in range(P_bounds[r] - 1):
                    add_hard([-var_phase(r, p), var_phase(r, p + 1)], "PHASE_MONO")

    # ------------------------------------------------------------------
    # (4) Horizontal and (5) Vertical comparators + semantics + copy constraints
    # ------------------------------------------------------------------

    for r in range(R):
        for p in range(P_bounds[r]):
            phase_p = var_phase(r,p)

            # ---------------- Horizontal comparators (row-wise) ----------------
            for krow in range(n):
                for jcol in range(m - 1):
                    sh = var_s_h(r, p, krow, jcol)

                    # Gating: if phase[p] = 1 (vertical), horizontal comparators must be off
                    # phase[p] = 1 ⇒ ¬s_h  ==  (¬phase[p] ∨ ¬s_h)
                    add_hard([-phase_p, -sh], "H_GATE")

                    # Parity (odd-even) for horizontal comparisons:
                    # here we just use p%2 as horizontal parity:
                    #   if jcol % 2 != p % 2 then s_h must be 0
                    if jcol % 2 != (p % 2):
                        add_hard([-sh], "H_GATE")

                    # Forward-only SWAP semantics between (krow,jcol) and (krow,jcol+1)
                    # NOTE: we ONLY constrain the s=1 (swap) case here.
                    #       The s=0 (identity) case is handled by the horizontal copy
                    #       constraints below, which are guarded on ¬phase and ¬s_left/¬s_right.
                    for ion in ions:
                        a_cur_j = var_a(r, p, krow, jcol, ion)
                        a_cur_j1 = var_a(r, p, krow, jcol + 1, ion)
                        a_next_j = var_a(r, p + 1, krow, jcol, ion)
                        a_next_j1 = var_a(r, p + 1, krow, jcol + 1, ion)

                        # (H3) If s=1 and ion is in right cell at p, it must be in left at p+1:
                        #      (s ∧ a_cur_j1) -> a_next_j
                        add_hard([-sh, -a_cur_j1, a_next_j], "H_SEM")

                        # (H4) If s=1 and ion is in left cell at p, it must be in right at p+1:
                        #      (s ∧ a_cur_j) -> a_next_j1
                        add_hard([-sh, -a_cur_j, a_next_j1], "H_SEM")

            # ---------------- Vertical comparators (column-wise) ----------------
            for krow in range(n - 1):
                for jcol in range(m):
                    sv = var_s_v(r, p, krow, jcol)

                    # Gating: if phase[p] = 0 (horizontal), vertical comparators must be off
                    # phase[p] = 0 ⇒ ¬s_v  ==  (phase[p] ∨ ¬s_v)
                    add_hard([phase_p, -sv], "V_GATE")

                    # Parity (odd-even) for vertical comparisons, using p%2 as v-index:
                    #   if krow % 2 != p % 2 then s_v must be 0
                    if krow % 2 != (p % 2):
                        add_hard([-sv], "V_GATE")

                    # Forward-only SWAP semantics between (krow,jcol) and (krow+1,jcol)
                    # Again, ONLY the s=1 case is constrained here; s=0 is enforced by
                    # the vertical copy constraints (phase ∧ ¬s_up ∧ ¬s_down).
                    for ion in ions:
                        a_cur_top = var_a(r, p, krow, jcol, ion)
                        a_cur_bot = var_a(r, p, krow + 1, jcol, ion)
                        a_next_top = var_a(r, p + 1, krow, jcol, ion)
                        a_next_bot = var_a(r, p + 1, krow + 1, jcol, ion)

                        # (V3) If s=1 and ion is in bottom cell at p, it must be in top at p+1:
                        #      (s ∧ a_cur_bot) -> a_next_top
                        add_hard([-sv, -a_cur_bot, a_next_top], "V_SEM")

                        # (V4) If s=1 and ion is in top cell at p, it must be in bottom at p+1:
                        #      (s ∧ a_cur_top) -> a_next_bot
                        add_hard([-sv, -a_cur_top, a_next_bot], "V_SEM")

    # -------- Copy constraints for non-participating cells (horizontal phase) --------
    for r in range(R):
        for p in range(P_bounds[r]):
            phase_p = var_phase(r,p)
            for krow in range(n):
                for jcol in range(m):
                    # horizontal comparators that touch (krow,jcol)
                    s_left = var_s_h(r, p, krow, jcol - 1) if jcol > 0 else None
                    s_right = var_s_h(r, p, krow, jcol) if jcol < m - 1 else None

                    # antecedent: (¬phase[p] ∧ ¬s_left ∧ ¬s_right)
                    lits_ante_neg = [phase_p]  # this is ¬(¬phase), used on the clause side

                    if s_left is not None:
                        lits_ante_neg.append(s_left)
                    if s_right is not None:
                        lits_ante_neg.append(s_right)

                    for ion in ions:
                        a_cur = var_a(r, p, krow, jcol, ion)
                        a_next = var_a(r, p + 1, krow, jcol, ion)

                        # (¬phase ∧ ¬s_left ∧ ¬s_right ∧ a_cur) -> a_next
                        # CNF: (phase ∨ s_left ∨ s_right ∨ ¬a_cur ∨ a_next)
                        add_hard(lits_ante_neg + [-a_cur, a_next], "H_COPY")

                        # (¬phase ∧ ¬s_left ∧ ¬s_right ∧ a_next) -> a_cur
                        # CNF: (phase ∨ s_left ∨ s_right ∨ ¬a_next ∨ a_cur)
                        add_hard(lits_ante_neg + [-a_next, a_cur], "H_COPY")

    # -------- Copy constraints for non-participating cells (vertical phase) --------
    for r in range(R):
        for p in range(P_bounds[r]):
            phase_p = var_phase(r,p)
            for krow in range(n):
                for jcol in range(m):
                    # vertical comparators that touch (krow,jcol)
                    s_up = var_s_v(r, p, krow - 1, jcol) if krow > 0 else None
                    s_down = var_s_v(r, p, krow, jcol) if krow < n - 1 else None

                    # antecedent: (phase[p] ∧ ¬s_up ∧ ¬s_down)
                    lits_ante_neg = [-phase_p]  # this is ¬phase on the clause side

                    if s_up is not None:
                        lits_ante_neg.append(s_up)
                    if s_down is not None:
                        lits_ante_neg.append(s_down)

                    for ion in ions:
                        a_cur = var_a(r, p, krow, jcol, ion)
                        a_next = var_a(r, p + 1, krow, jcol, ion)

                        # (phase ∧ ¬s_up ∧ ¬s_down ∧ a_cur) -> a_next
                        # CNF: (¬phase ∨ s_up ∨ s_down ∨ ¬a_cur ∨ a_next)
                        add_hard(lits_ante_neg + [-a_cur, a_next], "V_COPY")

                        # (phase ∧ ¬s_up ∧ ¬s_down ∧ a_next) -> a_cur
                        # CNF: (¬phase ∨ s_up ∨ s_down ∨ ¬a_next ∨ a_cur)
                        add_hard(lits_ante_neg + [-a_next, a_cur], "V_COPY")

    # ------------------------------------------------------------------
    # (5) End-of-round row/block abstraction and pair constraints
    # ------------------------------------------------------------------

    for r in range(R):
        g_rb_r = f"ROWBLOCK_LINK:r{r}"
        g_pair_r = f"PAIR_REQ:r{r}"

        # row_end / w_end linkage from final layout a[r,P_bounds[r]]
        for ion in ions:
            # row_end
            for d in range(n):
                re = var_row_end(r, ion, d)
                cell_lits = [var_a(r, P_bounds[r], d, j, ion) for j in range(m)]
                add_hard([-re] + cell_lits, g_rb_r)
                for aj in cell_lits:
                    add_hard([-aj, re], g_rb_r)

            # w_end (global block alignment)
            for b_local in range(num_blocks):
                we = var_w_end(r, ion, b_local)
                cell_list = block_cells[b_local]
                cells = [var_a(r, P_bounds[r], d, j_local, ion) for (d, j_local) in cell_list]
                if block_fully_inside[b_local] or (block_widths[b_local]>1):
                    add_hard([-we] + cells, g_rb_r)
                    for aj in cells:
                        add_hard([-aj, we], g_rb_r)
                else:
                    add_hard([-we], g_rb_r)

        # Pairs must share same row and block (unless skipping for debug)
        if not debug_skip_pair_constraints:
            if r in disable_pairs_rounds:
                pass
            else:
                for (i1, i2) in P_arr[r]:
                    if i1 not in ions or i2 not in ions:
                        continue

                    # Same row
                    for d in range(n):
                        re1 = var_row_end(r, i1, d)
                        re2 = var_row_end(r, i2, d)
                        add_hard([-re1, re2], g_pair_r)
                        add_hard([-re2, re1], g_pair_r)
                    # Same block (global aligned); only enforce for fully covered blocks
                    for b_local in range(num_blocks):
                        we1 = var_w_end(r, i1, b_local)
                        we2 = var_w_end(r, i2, b_local)
                        add_hard([-we1, we2], g_pair_r)
                        add_hard([-we2, we1], g_pair_r)

    # ------------------------------------------------------------------
    # (6) Optional Level-3 soft clauses (boundary avoidance, swap-cost)
    # ------------------------------------------------------------------
    if use_wcnf and add_boundary_soft and (wB_row > 0 or wB_col > 0):
        inner_ions_per_round: List[Set[int]] = []
        for r in range(R):
            inner_ions = set()
            for (i1, i2) in P_arr[r]:
                inner_ions.add(i1)
                inner_ions.add(i2)
            inner_ions.update(BT[r].keys())
            inner_ions_per_round.append(inner_ions)

        for r in range(R):
            inner_ions = inner_ions_per_round[r]
            cross_prefs_r = cross_boundary_prefs[r] if r < len(cross_boundary_prefs) else {}
            for ion in ions:
                dirs = cross_prefs_r.get(ion)
                if dirs:
                    for direction in dirs:
                        if direction in ("left", "right") and wB_col > 0:
                            if not boundary_adjacent.get(direction, False):
                                continue
                            target_col = 0 if direction == "left" else m - 1
                            lits = [var_a(r, P_bounds[r], d, target_col, ion) for d in range(n)]
                            if lits:
                                add_soft(lits, weight=wB_col)
                        if direction in ("top", "bottom") and wB_row > 0:
                            if not boundary_adjacent.get(direction, False):
                                continue
                            target_row = 0 if direction == "top" else n - 1
                            lits = [var_a(r, P_bounds[r], target_row, jcol, ion) for jcol in range(m)]
                            if lits:
                                add_soft(lits, weight=wB_row)

                if ion in inner_ions:
                    if boundary_adjacent.get("left", False) and wB_col > 0:
                        for d in range(n):
                            add_soft([-var_a(r, P_bounds[r], d, 0, ion)], weight=wB_col)
                    if boundary_adjacent.get("right", False) and wB_col > 0:
                        for d in range(n):
                            add_soft([-var_a(r, P_bounds[r], d, m - 1, ion)], weight=wB_col)
                    if boundary_adjacent.get("top", False) and wB_row > 0:
                        for jcol in range(m):
                            add_soft([-var_a(r, P_bounds[r], 0, jcol, ion)], weight=wB_row)
                    if boundary_adjacent.get("bottom", False) and wB_row > 0:
                        for jcol in range(m):
                            add_soft([-var_a(r, P_bounds[r], n - 1, jcol, ion)], weight=wB_row)

    if cross_boundary_prefs_norm and any(boundary_adjacent.values()):
        factor = max(0.0, min(1.0, boundary_capacity_factor))
        dir_capacity: Dict[str, int] = {}
        if boundary_adjacent.get("top", False):
            dir_capacity["top"] = int(round(half_h * m * factor))
        if boundary_adjacent.get("bottom", False):
            dir_capacity["bottom"] = int(round(half_h * m * factor))
        if boundary_adjacent.get("left", False):
            dir_capacity["left"] = int(round(half_w * n * factor))
        if boundary_adjacent.get("right", False):
            dir_capacity["right"] = int(round(half_w * n * factor))

        ions_per_round_dir: Dict[Tuple[int, str], List[int]] = defaultdict(list)
        for r, prefs_r in enumerate(cross_boundary_prefs_norm):
            for ion, dirs in prefs_r.items():
                for direction in dirs:
                    if direction in dir_capacity:
                        ions_per_round_dir[(r, direction)].append(ion)

        for key in ions_per_round_dir:
            ions_per_round_dir[key].sort()

        enforced_dirs_per_ion: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        for (r, direction), ion_list in ions_per_round_dir.items():
            cap = dir_capacity.get(direction, 0)
            if cap <= 0:
                continue
            for ion in ion_list[:cap]:
                enforced_dirs_per_ion[(r, ion)].add(direction)

        def _band_cells_for_dirs(directions: Set[str]) -> List[Tuple[int, int]]:
            row_min, row_max = 0, n - 1
            col_min, col_max = 0, m - 1
            if "top" in directions:
                row_max = min(row_max, half_h - 1)
            if "bottom" in directions:
                row_min = max(row_min, n - half_h)
            if "left" in directions:
                col_max = min(col_max, half_w - 1)
            if "right" in directions:
                col_min = max(col_min, m - half_w)
            if row_min > row_max or col_min > col_max:
                return []
            return [
                (rr, cc)
                for rr in range(row_min, row_max + 1)
                for cc in range(col_min, col_max + 1)
            ]

        for r in range(R):
            prefs_r = cross_boundary_prefs_norm[r]
            if not prefs_r:
                continue
            P_final = P_bounds[r]
            for ion in prefs_r.keys():
                enforced_dirs = enforced_dirs_per_ion.get((r, ion))
                if not enforced_dirs:
                    continue
                cells = _band_cells_for_dirs(enforced_dirs)
                if not cells:
                    union_cells: Set[Tuple[int, int]] = set()
                    for direction in enforced_dirs:
                        union_cells.update(_band_cells_for_dirs({direction}))
                    cells = list(union_cells)
                if not cells:
                    if DEBUG_DIAG:
                        wise_logger.debug(
                            "[CROSS_BOUNDARY] no valid cells for ion %d round %d dirs=%s; skipping",
                            ion,
                            r,
                            sorted(enforced_dirs),
                        )
                    continue
                clause = [var_a(r, P_final, d, c, ion) for (d, c) in cells]
                add_hard(clause, "CROSS_BOUNDARY")

    # -------------------------------
    # Per-round pass usage helpers (u) and global Σ_r P_r bound
    # -------------------------------
    # Keep u[r,p] as-is: u <-> (OR of comparators)
    for r in range(R):
        for p in range(P_bounds[r]):
            u_rp = var_u(r, p)

            comp_lits: List[int] = []
            for krow in range(n):
                for jcol in range(m - 1):
                    comp_lits.append(var_s_h(r, p, krow, jcol))
            for krow in range(n - 1):
                for jcol in range(m):
                    comp_lits.append(var_s_v(r, p, krow, jcol))

            if not comp_lits:
                # No comparators exist at all in this pass: u must be false.
                add_hard([-u_rp], "UTIL_U")
                continue

            # u[r,p] ↔ OR(comp_lits)
            add_hard([-u_rp] + comp_lits, "UTIL_U")
            for s_lit in comp_lits:
                add_hard([-s_lit, u_rp], "UTIL_U")

    if sum_bound_B is not None and optimize_round_start < R:
        sum_u_lits: List[int] = []
        for r in range(optimize_round_start, R):
            for p in range(P_bounds[r]):
                sum_u_lits.append(var_u(r, p))

        total_slots = len(sum_u_lits)
        bound = min(sum_bound_B, total_slots)
        if bound < total_slots:
            card_enc = CardEnc.atmost(
                lits=sum_u_lits,
                bound=bound,
                encoding=EncType.totalizer,
                vpool=vpool,
            )
            for clause in card_enc.clauses:
                add_hard(clause, "SUM_BOUND")

    selectors = grp.sel if grp.enabled else {}
    group_meta = grp.meta if grp.enabled else {}
    return formula, vpool, ions, var_a, selectors, group_meta


# ---------------------------------------------------------------------------
# Schedule / usage decoders
# ---------------------------------------------------------------------------

def _wise_decode_schedule_from_model(
    model: List[int],
    vpool,
    n: int,
    m: int,
    R: int,
    P_bound: int,
    ignore_initial_reconfig: bool,
) -> List[List[Dict[str, Any]]]:
    model_set = {lit for lit in model if lit > 0}
    P_bounds = ([P_bound+n+m]*int(ignore_initial_reconfig) + [P_bound]*(R-int(ignore_initial_reconfig)))

    def lit_true(v: int) -> bool:
        return v in model_set

    def var_s_h(r, p, krow, jcol):
        return vpool.id(("s_h", r, p, krow, jcol))

    def var_s_v(r, p, krow, jcol):
        return vpool.id(("s_v", r, p, krow, jcol))

    def var_phase(r, p):
        return vpool.id(("phase", r, p))

    schedule: List[List[Dict[str, Any]]] = [[] for _ in range(R)]
    for r in range(R):
        for p in range(P_bounds[r]):
            phase_lit = var_phase(r, p)
            is_vertical = (phase_lit <= vpool.top and lit_true(phase_lit))
            phase = "V" if is_vertical else "H"

            pass_info: Dict[str, Any] = {
                "phase": phase,
                "h_swaps": [],
                "v_swaps": [],
            }

            if phase == "H":
                # Horizontal comparators at this pass
                for krow in range(n):
                    for jcol in range(m - 1):
                        v = var_s_h(r, p, krow, jcol)
                        if v <= vpool.top and lit_true(v):
                            pass_info["h_swaps"].append((krow, jcol))
            else:
                # Vertical comparators at this pass
                for krow in range(n - 1):
                    for jcol in range(m):
                        v = var_s_v(r, p, krow, jcol)
                        if v <= vpool.top and lit_true(v):
                            pass_info["v_swaps"].append((krow, jcol))

            schedule[r].append(pass_info)

    return schedule


def _wise_extract_round_pass_usage(model, vpool, R, P_bound, ignore_initial_reconfig: bool, n: int, m: int):
    model_set = {lit for lit in model if lit > 0}
    P_bounds = ([P_bound+n+m]*int(ignore_initial_reconfig) + [P_bound]*(R-int(ignore_initial_reconfig)))

    def lit_true(v: int) -> bool:
        return v in model_set

    def var_u(r, p):
        return vpool.id(("u", r, p))

    per_round: List[int] = []
    for r in range(R):
        count = 0
        for p in range(P_bounds[r]):
            u_lit = var_u(r, p)
            if u_lit <= vpool.top and lit_true(u_lit):
                count += 1
        per_round.append(count)

    return per_round


# ---------------------------------------------------------------------------
# Per-configuration worker (binary search over ΣP)
# ---------------------------------------------------------------------------

def _wise_sat_config_worker(
    cfg: Tuple[int, float],
    *,
    context: _WiseSatBuilderContext,
    optimize_round_start: int,
    max_sat_time: float,
    max_rc2_time: float,
    boundary_adjacent: Dict[str, bool],
    cross_boundary_prefs: List[Dict[int, Set[str]]],
    ignore_initial_reconfig: bool,
    progress_path: Optional[str] = None,
    bt_soft_weight: int = 0,
    sat_children: Optional[Dict[Any, int]] = None,
    rc2_children: Optional[Dict[Any, int]] = None,
    stop_event: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    For a given (P_max, boundary_capacity_factor) configuration, perform a binary search
    over Σ_r P_r bounds to find the tightest satisfiable schedule.
    """
    P_max, boundary_capacity_factor = cfg
    rounds_under_sum = max(1, context.R - optimize_round_start)
    max_bound_B = rounds_under_sum * P_max
    low = 1
    high = max_bound_B
    best_result: Optional[Dict[str, Any]] = None
    last_result: Optional[Dict[str, Any]] = None
    use_soft_bt = bt_soft_weight > 0

    wise_logger.debug(
        "[WISE] config start: P_max=%d, cap_factor=%.2f, ΣP bound in [1,%d]",
        P_max,
        boundary_capacity_factor,
        max_bound_B,
    )

    def publish(res: Dict[str, Any]) -> None:
        if not progress_path:
            return
        try:
            tmp_path = f"{progress_path}.tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(res, f)
            os.replace(tmp_path, progress_path)
        except Exception:
            pass

    def _solve_with_bound(sum_bound_B: int) -> Dict[str, Any]:
        if stop_event is not None and stop_event.is_set():
            return {
                "P_max": P_max,
                "boundary_capacity_factor": boundary_capacity_factor,
                "status": "timeout",
                "sat": False,
                "schedule": None,
                "per_round_usage": None,
                "model": None,
            }
        try:
            (
                formula_mid,
                vpool_mid,
                _,
                _,
                grp_sel_mid,
                _,
            ) = _wise_build_structural_cnf(
                context,
                P_max,
                sum_bound_B=sum_bound_B,
                use_wcnf=use_soft_bt,
                add_boundary_soft=use_soft_bt,
                phase_label=f"ΣP<={sum_bound_B}",
                optimize_round_start=optimize_round_start,
                debug_core=False,
                core_granularity="coarse",
                debug_skip_cardinality=False,
                boundary_adjacent=boundary_adjacent,
                cross_boundary_prefs=cross_boundary_prefs,
                boundary_capacity_factor=boundary_capacity_factor,
                bt_soft_weight=bt_soft_weight,
            )

            cost_mid: Optional[int] = None
            worker_pid = os.getpid()
            job_base = (worker_pid, P_max, boundary_capacity_factor, sum_bound_B)
            if use_soft_bt:
                job_key = ("RC2",) + job_base
                model_mid, cost_mid, status_solver = run_rc2_with_timeout_file(
                    formula_mid,
                    timeout_s=max_rc2_time,
                    debug_prefix="[WISE-RC2]",
                    stop_event=stop_event,
                    job_key=job_key,
                    children_dict=rc2_children,
                )
                sat_ok = status_solver == "ok" and model_mid is not None
            else:
                assumptions_mid = (
                    [-lit for lit in grp_sel_mid.values()] if grp_sel_mid else None
                )
                job_key = ("SAT",) + job_base

                sat_ok, model_mid, status_solver = run_sat_with_timeout_file(
                    formula_mid,
                    timeout_s=max_sat_time,
                    debug_prefix=None,
                    assumptions=assumptions_mid,
                    stop_event=stop_event,
                    job_key=job_key,
                    children_dict=sat_children,
                )
        except Exception as e:
            return {
                "P_max": P_max,
                "boundary_capacity_factor": boundary_capacity_factor,
                "status": f"error:{repr(e)}",
                "sat": False,
                "schedule": None,
                "per_round_usage": None,
                "model": None,
            }

        if status_solver != "ok" or not sat_ok or model_mid is None:
            return {
                "P_max": P_max,
                "boundary_capacity_factor": boundary_capacity_factor,
                "status": status_solver if status_solver != "ok" else "unsat",
                "sat": bool(sat_ok),
                "schedule": None,
                "per_round_usage": None,
                "model": None,
                "cost": cost_mid,
            }

        schedule = _wise_decode_schedule_from_model(
            model_mid,
            vpool_mid,
            context.n,
            context.m,
            context.R,
            P_max,
            ignore_initial_reconfig,
        )
        per_round = _wise_extract_round_pass_usage(
            model_mid,
            vpool_mid,
            context.R,
            P_max,
            ignore_initial_reconfig,
            context.n,
            context.m,
        )

        return {
            "P_max": P_max,
            "boundary_capacity_factor": boundary_capacity_factor,
            "status": "ok",
            "sat": True,
            "schedule": schedule,
            "per_round_usage": per_round,
            "model": model_mid,
            "cost": cost_mid,
        }

    while low <= high:
        if stop_event is not None and stop_event.is_set():
            wise_logger.debug(
                "[WISE] config P_max=%d cap=%.2f: stop_event set; aborting ΣP search",
                P_max,
                boundary_capacity_factor,
            )
            break
        mid = (low + high) // 2

        wise_logger.debug(
            "[WISE] config P_max=%d cap=%.2f: trying ΣP<=%d",
            P_max,
            boundary_capacity_factor,
            mid,
        )
        result = _solve_with_bound(mid)
        last_result = result

        if result.get("status") == "ok" and result.get("sat") and result.get("model") is not None:
            sum_usage_mid = int(sum((result.get("per_round_usage") or [])))
            wise_logger.debug(
                "[WISE] config P_max=%d cap=%.2f: SAT at ΣP=%d (usage=%d)",
                P_max,
                boundary_capacity_factor,
                mid,
                sum_usage_mid,
            )
            result["sum_bound_B"] = mid
            best_result = result
            publish(result)
            high = mid - 1
        else:
            wise_logger.debug(
                "[WISE] config P_max=%d cap=%.2f: status=%s at ΣP=%d",
                P_max,
                boundary_capacity_factor,
                result.get("status"),
                mid,
            )
            low = mid + 1

    if best_result is not None:
        wise_logger.info(
            "[WISE] config done: P_max=%d cap=%.2f best ΣP=%d",
            P_max,
            boundary_capacity_factor,
            best_result.get("sum_bound_B"),
        )
        return best_result

    if last_result is not None:
        wise_logger.info(
            "[WISE] config partial: P_max=%d cap=%.2f returning last status=%s",
            P_max,
            boundary_capacity_factor,
            last_result.get("status"),
        )
        publish(last_result)
        return last_result

    res_default = {
        "P_max": P_max,
        "boundary_capacity_factor": boundary_capacity_factor,
        "status": "unsat",
        "sat": False,
        "schedule": None,
        "per_round_usage": None,
        "model": None,
    }
    publish(res_default)
    wise_logger.info(
        "[WISE] config fail: P_max=%d cap=%.2f no SAT solution found",
        P_max,
        boundary_capacity_factor,
    )
    return res_default


# ---------------------------------------------------------------------------
# Process-isolated SAT solver
# ---------------------------------------------------------------------------

def _sat_worker_from_file(
    cnf_path: str,
    result_path: str,
    assumptions: Optional[List[int]] = None,
):
    """
    Worker process entry for plain SAT:
      - loads CNF from cnf_path,
      - runs Minisat22,
      - dumps {'sat': bool, 'model': list[int] or None, 'time': elapsed}
        or {'error': repr(e)} to result_path via pickle.
    """
    try:
        t0 = time.time()
        with open(cnf_path, "rb") as f:
            cnf = pickle.load(f)   # CNF object or just clauses

        with Minisat22(bootstrap_with=cnf.clauses) as sat:
            if assumptions is None:
                sat_ok = sat.solve()
            else:
                sat_ok = sat.solve(assumptions=assumptions)
            model = sat.get_model() if sat_ok else None

        t1 = time.time()
        data = {
            "sat": bool(sat_ok),
            "model": model,
            "time": t1 - t0,
        }

    except KeyboardInterrupt as e:
        data = {
            "error": f"KeyboardInterrupt in SAT worker: {repr(e)}",
        }

    except Exception as e:
        data = {
            "error": repr(e),
        }

    try:
        with open(result_path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass


def run_sat_with_timeout_file(
    cnf: CNF,
    timeout_s: float,
    debug_prefix: str = "[WISE]",
    assumptions: Optional[List[int]] = None,
    stop_event: Optional[Any] = None,
    job_key: Optional[Any] = None,
    children_dict: Optional[Mapping[Any, int]] = None,
):
    """
    Run Minisat22 on 'cnf' in a separate process with a wall-clock timeout.

    Returns: (sat_ok, model, status)
      - status == "ok"        : sat_ok is True/False, model is list[int] or None.
      - status == "timeout"   : sat_ok, model are None (worker killed by timeout).
      - status == "error"     : sat_ok, model are None (worker crashed).
      - status == "user_abort": sat_ok, model are None (parent got KeyboardInterrupt
                                while waiting; worker terminated & cleaned up).

    If `assumptions` is provided, the SAT call uses those literals (usually the
    negated selector vars from CoreGroups) so guarded clauses remain active.
    """

    if timeout_s is not None and timeout_s <= 0:
        if debug_prefix:
            wise_logger.debug(
                "%s SAT disabled (timeout <= 0); treating as timeout.",
                debug_prefix,
            )
        return None, None, "timeout"
    
    def _unregister_child() -> None:
        if children_dict is not None and job_key is not None:
            try:
                children_dict.pop(job_key, None)
            except Exception:
                pass

    with tempfile.TemporaryDirectory() as tmpdir:
        cnf_path = os.path.join(tmpdir, "instance.cnf.pkl")
        result_path = os.path.join(tmpdir, "result_sat.pkl")

        with open(cnf_path, "wb") as f:
            pickle.dump(cnf, f)

        p = mp.Process(
            target=_sat_worker_from_file,
            args=(cnf_path, result_path, assumptions),
        )
        p.daemon = True

        if debug_prefix:
            wise_logger.debug(
                "%s SAT worker starting (timeout=%.1fs) for CNF: clauses=%d",
                debug_prefix,
                timeout_s,
                len(cnf.clauses),
            )

        try:
            p.start()
        except KeyboardInterrupt:
            if debug_prefix:
                wise_logger.debug(
                    "%s SAT start interrupted by user; terminating worker and returning user_abort.",
                    debug_prefix,
                )
            if p.is_alive():
                p.terminate()
                try:
                    p.join(5.0)
                except Exception:
                    pass
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        pass
            _unregister_child()
            return None, None, "user_abort"

        # Register child PID for global tracking
        if children_dict is not None and job_key is not None:
            try:
                children_dict[job_key] = p.pid
            except Exception:
                pass

        start = time.time()
        status = None

        try:
            while True:
                p.join(0.5)

                if stop_event is not None and stop_event.is_set():
                    status = "timeout"
                    break

                if not p.is_alive():
                    status = "finished"
                    break

                elapsed = time.time() - start
                if elapsed >= timeout_s:
                    status = "timeout"
                    break

        except KeyboardInterrupt:
            if debug_prefix:
                elapsed = time.time() - start
                wise_logger.debug(
                    "%s SAT join interrupted by user after %.3fs; terminating worker (user_abort).",
                    debug_prefix,
                    elapsed,
                )
            if p.is_alive():
                p.terminate()
                try:
                    p.join(5.0)
                except Exception:
                    pass
            _unregister_child()
            return None, None, "user_abort"

        if status == "timeout":
            if debug_prefix:
                elapsed = time.time() - start
                wise_logger.debug(
                    "%s SAT worker exceeded %.1fs (elapsed=%.3fs); terminating (timeout).",
                    debug_prefix,
                    timeout_s,
                    elapsed,
                )
            if p.is_alive():
                p.terminate()
                try:
                    p.join(5.0)
                except Exception:
                    pass
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        pass
            _unregister_child()
            return None, None, "timeout"

        if not os.path.exists(result_path):
            if debug_prefix:
                wise_logger.debug(
                    "%s SAT worker finished but produced no result file; treating as error.",
                    debug_prefix,
                )
            _unregister_child()
            return None, None, "error"

        try:
            with open(result_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            if debug_prefix:
                wise_logger.debug(
                    "%s SAT worker result read error: %r; treating as error.",
                    debug_prefix,
                    e,
                )
            _unregister_child()
            return None, None, "error"

        if "error" in data:
            if debug_prefix:
                wise_logger.debug(
                    "%s SAT worker raised: %s; treating as error.",
                    debug_prefix,
                    data["error"],
                )
            _unregister_child()
            return None, None, "error"

        if debug_prefix:
            wise_logger.debug(
                "%s SAT worker finished in %.3fs, SAT=%s",
                debug_prefix,
                data.get("time", 0.0),
                data.get("sat"),
            )
        _unregister_child()
        return bool(data.get("sat")), data.get("model"), "ok"


# ---------------------------------------------------------------------------
# Process-isolated MaxSAT (RC2) solver
# ---------------------------------------------------------------------------

def _rc2_worker_from_file(wcnf_path: str, result_path: str):
    """
    Worker process entry:
      - loads WCNF from wcnf_path,
      - runs RC2,
      - dumps {'model': model, 'cost': cost, 'time': elapsed}
        or {'error': repr(e)} to result_path via pickle.
    """
    try:
        t0 = time.time()
        with open(wcnf_path, "rb") as f:
            wcnf = pickle.load(f)

        rc2 = RC2(wcnf)
        model = rc2.compute()
        cost = rc2.cost if model is not None else None
        t1 = time.time()

        data = {
            "model": model,
            "cost": cost,
            "time": t1 - t0,
        }

    except KeyboardInterrupt as e:
        # Treat KeyboardInterrupt inside worker as an error; the parent
        # will see "error" status and can fall back to SAT.
        data = {
            "error": f"KeyboardInterrupt in worker: {repr(e)}",
        }

    except Exception as e:
        data = {
            "error": repr(e),
        }

    try:
        with open(result_path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        # If we cannot even write the result, there's nothing more to do.
        pass


def run_rc2_with_timeout_file(
    wcnf: WCNF,
    timeout_s: float,
    debug_prefix: str = "[WISE]",
    stop_event: Optional[Any] = None,
    job_key: Optional[Any] = None,
    children_dict: Optional[Mapping[Any, int]] = None,
):
    """
    Run RC2 on 'wcnf' in a separate process with a wall-clock timeout.

    Returns: (model, cost, status)
      - status == "ok"        : model, cost are valid.
      - status == "timeout"   : model, cost are None (worker killed by timeout).
      - status == "error"     : model, cost are None (worker crashed).
      - status == "user_abort": model, cost are None (parent got KeyboardInterrupt
                                while waiting; worker terminated & cleaned up).

    KeyboardInterrupt is *never* propagated outside this function.
    """

    # If timeout <= 0, treat as "no RC2".
    if timeout_s is not None and timeout_s <= 0:
        if debug_prefix:
            wise_logger.debug(
                "%s RC2 disabled (timeout <= 0); treating as timeout.",
                debug_prefix,
            )
        return None, None, "timeout"
    
    def _unregister_child() -> None:
        if children_dict is not None and job_key is not None:
            try:
                children_dict.pop(job_key, None)
            except Exception:
                pass

    with tempfile.TemporaryDirectory() as tmpdir:
        wcnf_path = os.path.join(tmpdir, "instance.wcnf.pkl")
        result_path = os.path.join(tmpdir, "result_rc2.pkl")

        # Dump WCNF to file for worker.
        with open(wcnf_path, "wb") as f:
            pickle.dump(wcnf, f)

        p = mp.Process(target=_rc2_worker_from_file,
                       args=(wcnf_path, result_path))
        # Optional: make sure child dies if parent dies badly.
        p.daemon = True

        if debug_prefix:
            wise_logger.debug(
                "%s RC2 worker starting (timeout=%.1fs) for WCNF: vars=%d, hard=%d, soft=%d",
                debug_prefix,
                timeout_s,
                wcnf.nv,
                len(wcnf.hard),
                len(wcnf.soft),
            )

        try:
            p.start()
        except KeyboardInterrupt:
            if debug_prefix:
                wise_logger.debug(
                    "%s RC2 start interrupted by user; terminating worker and returning user_abort.",
                    debug_prefix,
                )
            if p.is_alive():
                p.terminate()
                try:
                    p.join(5.0)
                except Exception:
                    pass
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        pass
            _unregister_child()
            return None, None, "user_abort"

        if children_dict is not None and job_key is not None:
            try:
                children_dict[job_key] = p.pid
            except Exception:
                pass

        # Our own timeout loop instead of a single long join(timeout_s)
        start = time.time()
        status = None

        try:
            while True:
                p.join(0.5)

                if stop_event is not None and stop_event.is_set():
                    status = "timeout"
                    break

                if not p.is_alive():
                    status = "finished"
                    break

                elapsed = time.time() - start
                if elapsed >= timeout_s:
                    status = "timeout"
                    break

        except KeyboardInterrupt:
            # Parent got Ctrl-C while waiting.
            if debug_prefix:
                elapsed = time.time() - start
                wise_logger.debug(
                    "%s RC2 join interrupted by user after %.3fs; terminating worker (user_abort).",
                    debug_prefix,
                    elapsed,
                )
            if p.is_alive():
                p.terminate()
                try:
                    p.join(5.0)
                except Exception:
                    pass
            _unregister_child()
            return None, None, "user_abort"

        if status == "timeout":
            if debug_prefix:
                elapsed = time.time() - start
                wise_logger.debug(
                    "%s RC2 worker exceeded %.1fs (elapsed=%.3fs); terminating (timeout).",
                    debug_prefix,
                    timeout_s,
                    elapsed,
                )
            if p.is_alive():
                p.terminate()
                try:
                    p.join(5.0)
                except Exception:
                    pass
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        pass
            _unregister_child()
            return None, None, "timeout"

        # status == "finished": worker exited within timeout
        if not os.path.exists(result_path):
            if debug_prefix:
                wise_logger.debug(
                    "%s RC2 worker finished but wrote no result file; treating as error.",
                    debug_prefix,
                )
            _unregister_child() 
            return None, None, "error"

        try:
            with open(result_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            if debug_prefix:
                wise_logger.debug(
                    "%s RC2 worker result read error: %r; treating as error.",
                    debug_prefix,
                    e,
                )
            _unregister_child()
            return None, None, "error"

        if "error" in data:
            if debug_prefix:
                wise_logger.debug(
                    "%s RC2 worker raised: %s; treating as error.",
                    debug_prefix,
                    data["error"],
                )
            _unregister_child()
            return None, None, "error"

        if debug_prefix:
            wise_logger.debug(
                "%s RC2 worker finished in %.3fs, opt_cost=%s",
                debug_prefix,
                data.get("time", 0.0),
                data.get("cost"),
            )
        _unregister_child()
        return data.get("model"), data.get("cost"), "ok"
