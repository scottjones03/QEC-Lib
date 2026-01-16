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
import numpy as np
from typing import (
    Sequence,
    List,
    Optional,
    Callable,
    Any,
    Mapping,
    Set,
    Dict,
    Iterable,
    Union
)
import abc
from src.utils.qccd_nodes import *
from typing import List, Sequence, Dict, Tuple
from collections import defaultdict, deque
from collections import defaultdict
from dataclasses import dataclass, field
import heapq, math, bisect
from typing import Dict, List, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from multiprocessing import Process, Pipe
from collections import Counter
import signal
from pysat.formula import IDPool, WCNF, CNF
from pysat.card import CardEnc, EncType
from pysat.solvers import Minisat22  
from pysat.examples.rc2 import RC2
import time
import pickle 
import os
import tempfile
import shutil
import multiprocessing as mp
from scipy import stats
from pysat.solvers import Solver
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from scipy.optimize import linear_sum_assignment


WISE_LOGGER_NAME = "wise.qccd.sat"
wise_logger = logging.getLogger(WISE_LOGGER_NAME)
if not wise_logger.handlers:
    wise_logger.addHandler(logging.NullHandler())
wise_logger.propagate = False



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




# ---------- UNSAT core helpers ----------

class NoFeasibleLayoutError(RuntimeError):
    ...
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


class Operation:
    KEY: Operations

    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        self._run = run
        self._kwargs = dict(kwargs)
        self._involvedIonsForLabel: List[Ion] = []
        self._involvedComponents: List[QCCDComponent] = involvedComponents
        self._addOns = ""
        self._fidelity: float = 1.0
        self._dephasingFidelity: float = 1.0
        self._operationTime: float = 0.0

    def addComponent(self, component: QCCDComponent) -> None:
        self._involvedComponents.append(component)

    @property
    def involvedComponents(self) -> Sequence[QCCDComponent]:
        return self._involvedComponents

    @property
    def color(self) -> str:
        return "lightgreen"

    @property
    def involvedIonsForLabel(self) -> Sequence[Ion]:
        return self._involvedIonsForLabel

    @property
    def label(self) -> str:
        return self.KEY.name + self._addOns

    @property
    @abc.abstractmethod
    def isApplicable(self) -> bool:
        return all(self.KEY in component.allowedOperations for component in self.involvedComponents)
    
    @abc.abstractmethod
    def _checkApplicability(self) -> None:
        for component in self.involvedComponents:
            if self.KEY not in component.allowedOperations:
                raise ValueError(f"Component {component} with index {component.idx} cannot complete {self.KEY.name}")

    @classmethod
    @abc.abstractmethod
    def physicalOperation(cls) -> "Operation": ...

    @abc.abstractmethod
    def calculateFidelity(self) -> None: ...

    @abc.abstractmethod
    def calculateDephasingFidelity(self) -> None: ...

    @abc.abstractmethod
    def calculateOperationTime(self) -> None: ...

    @abc.abstractmethod
    def _generateLabelAddOns(self) -> None: ...

    def run(self) -> None:
        self._checkApplicability()
        self.calculateOperationTime()
        self.calculateFidelity()
        self.calculateDephasingFidelity()
        self._run(())
        self._generateLabelAddOns()

    def dephasingFidelity(self) -> float:
        # Deprecated!
        return self._dephasingFidelity

    def fidelity(self) -> float:
        return self._fidelity
    
    def operationTime(self) -> float:
        return self._operationTime


class CrystalOperation(Operation):
    T2 = 2.2 # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents, **kwargs)
        self._trap: Trap = kwargs["trap"]


    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        self.calculateOperationTime()
        self._dephasingFidelity = 1 - (1-np.exp(-self.operationTime()/self.T2))/2  # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330

    @property
    def ionsInfluenced(self) -> Sequence[Ion]:
        return self._trap.ions
    
    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = list(self._trap.ions)
        self._addOns = ""
        for ion in self._involvedIonsForLabel:
            self._addOns += f" {ion.label}"


class GlobalReconfigurations(Operation):
    KEY = Operations.GLOBAL_RECONFIG

    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._wiseArch: QCCDWiseArch = kwargs['wiseArch']
        self._reconfigTime: float = kwargs['reconfigTime']

    def calculateOperationTime(self) -> None:
        self._operationTime =  self._reconfigTime

    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        # FIXME might be inaccurate
        self.calculateOperationTime()
        self._dephasingFidelity = 1 - (1-np.exp(-self.operationTime()/2.2))/2  # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330


    def _generateLabelAddOns(self) -> None:
        self._addOns = f""

    @property
    def isApplicable(self) -> bool:
        return True
    
    def _checkApplicability(self) -> None:
        return True

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
        # DEBUG: trace usage of schedule/sat_schedule for reconfiguration
        # print("[DEBUG GlobalReconfigurations.physicalOperation] initial_placement =", initial_placement)
        # if schedule is None:
        #     print("[DEBUG GlobalReconfigurations.physicalOperation] schedule is None (likely cached block first step)")
        # else:
        #     try:
        #         print("[DEBUG GlobalReconfigurations.physicalOperation] schedule len =", len(schedule))
        #         if schedule and isinstance(schedule[0], list):
        #             print("[DEBUG GlobalReconfigurations.physicalOperation] schedule[0] passes len =", len(schedule[0]))
        #     except Exception as e:
        #         print("[DEBUG GlobalReconfigurations.physicalOperation] error inspecting schedule:", repr(e))

        heatingRates, reconfigTime = cls._runOddEvenReconfig(
            wiseArch,
            arrangement,
            oldAssignment,
            newAssignment,
            sat_schedule=schedule,
            initial_placement=initial_placement,
        )
        reconfigTime = 1e-20 if initial_placement else reconfigTime
        def run():
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
            run=lambda _: run(),
            involvedComponents=list(arrangement.keys()),
            wiseArch=wiseArch,
            reconfigTime=reconfigTime

        )


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
        max_sat_time = 4800.0,  
        active_ions: Set[int] = None,
        full_P_arr: List[List[Tuple[int, int]]]=[],
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
        if len(full_P_arr)==0:
            full_P_arr=P_arr

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
            block_widths.append(local_end-local_start)
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

            ions_set = set(ions)

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
                    ions_r = ions_set
                else:
                    ions_r &= ions_set

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

        manager = mp.Manager()
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
            sum_usage = int(sum(usage[(optimize_round_start*(R>1)):]))
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
        # A reasonable default: all active ions that lie entirely in the current subgrid.
        # If you already have a list `core_ions_this_slice`, reuse that here.
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
        # print(schedule)
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
    



    
    @classmethod
    def _runOddEvenReconfig(
        cls,
        wiseArch: "QCCDWiseArch",
        arrangement: Mapping["Trap", Sequence["Ion"]],
        oldAssignment: Sequence[Sequence[int]],
        newAssignment: Sequence[Sequence[int]],
        ignoreSpectators: bool = False,
        sat_schedule: Optional[List[Dict[str, Any]]] = None,   # NEW: decoded schedule from RC2
        initial_placement: bool = False
    ) -> Tuple[Mapping[int, float], float]:
        # DEBUG: entry into _runOddEvenReconfig
        # print("[DEBUG _runOddEvenReconfig] called")
        # try:
        #     print("  initial_placement =", initial_placement)
        # except Exception:
        #     pass
        # try:
        #     if hasattr(oldAssignment, "shape"):
        #         print("  oldAssignment shape =", oldAssignment.shape)
        #     else:
        #         print("  oldAssignment len =", len(oldAssignment))
        # except Exception:
        #     pass
        # try:
        #     if hasattr(newAssignment, "shape"):
        #         print("  newAssignment shape =", newAssignment.shape)
        #     else:
        #         print("  newAssignment len =", len(newAssignment))
        # except Exception:
        #     pass
        # try:
        #     if sat_schedule is None:
        #         print("  sat_schedule is None (layout-only reconfig; cached block first step)")
        #     else:
        #         print("  sat_schedule type =", type(sat_schedule))
        #         print("  sat_schedule len =", len(sat_schedule))
        #         if sat_schedule and isinstance(sat_schedule[0], list):
        #             print("  sat_schedule[0] passes len =", len(sat_schedule[0]))
        # except Exception as e:
        #     print("[DEBUG _runOddEvenReconfig] error inspecting sat_schedule:", repr(e))

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

        row_swap_time = (
            Move.MOVING_TIME
            + Merge.MERGING_TIME
            + CrystalRotation.ROTATION_TIME
            + Split.SPLITTING_TIME
            + Move.MOVING_TIME
        )
        row_swap_heating = (
            Move.MOVING_TIME * Move.HEATING_RATE
            + Merge.MERGING_TIME * Merge.HEATING_RATE
            + CrystalRotation.ROTATION_TIME * CrystalRotation.HEATING_RATE
            + Split.SPLITTING_TIME * Split.HEATING_RATE
            + Move.MOVING_TIME * Move.HEATING_RATE
        )
        col_swap_time = (2 * JunctionCrossing.CROSSING_TIME) + (
            4 * JunctionCrossing.CROSSING_TIME + Move.MOVING_TIME
        ) * 2
        col_swap_heating_rate = (
            6 * JunctionCrossing.CROSSING_TIME * JunctionCrossing.HEATING_RATE
            + Move.MOVING_TIME * Move.HEATING_RATE
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
        timeElapsed += Split.SPLITTING_TIME
        for idx in heatingRates.keys():
            heatingRates[idx] += Split.HEATING_RATE * Split.SPLITTING_TIME

        # If we have a SAT schedule, use it directly and SKIP Phases B/C/D.
        if sat_schedule is not None:
            acc_passes = 0
            # print("[DEBUG _runOddEvenReconfig] entering schedule loop, len =", len(sat_schedule))

            for pass_idx, info in enumerate(sat_schedule):
                # try:
                #     print(
                #         f"[DEBUG _runOddEvenReconfig] round {pass_idx}: passes type={type(info)}, "
                #         f"h_swaps={len(info.get('h_swaps', [])) if hasattr(info, 'get') else 'NA'}, "
                #         f"v_swaps={len(info.get('v_swaps', [])) if hasattr(info, 'get') else 'NA'}"
                #     )
                # except Exception as e:
                #     print("[DEBUG _runOddEvenReconfig] error inspecting schedule entry:", repr(e))

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
                        # Optionally respect ignoreSpectators here, but that
                        # can deviate from the SAT layout. Safer to ignore it
                        # when using a SAT schedule:
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
                # print("[WARN] SAT-driven reconfig: final layout does NOT match newAssignment!")
                # print("  A (final):")
                # print(A)
                # print("  T (target):")
                # print(T)
                # You can raise if you want:
                raise RuntimeError("SAT schedule did not realise target layout")
            # if not initial_placement:
            #     print(
            #         f"RECONFIGURATION (SAT schedule): {acc_passes} passes were needed for the current reconfiguration round, "
            #         f"taking {timeElapsed} time and {heatingRates} heating"
            #     )
            return heatingRates, timeElapsed
        # else:
            # sat_schedule is None: fall back to heuristic odd-even reconfiguration.
            # try:
            #     diff = int(np.sum(A != T))
            # except Exception:
            #     diff = "NA"
            # print(
            #     "[DEBUG _runOddEvenReconfig] sat_schedule is None; using heuristic odd-even reconfig "
            #     f"(Phase B/C/D). layout_diffs={diff}"
            # )

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

            # print(
            #     f"[DEBUG _runOddEvenReconfig][row_pass_by_rank] "
            #     f"phase={phase_label}, swaps_total={swaps_this_phase}, "
            #     f"max_swaps_row={maxSwapsInRow}"
            # )
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

            # print(
            #     f"[DEBUG _runOddEvenReconfig][col_bucket_pass] "
            #     f"bucket_mod={bucket_mod}, phase={phase_label}, "
            #     f"swaps_total={swaps_this_phase}, max_swaps_col={maxSwapsInCol}"
            # )
            return maxSwapsInCol > 0

        # ---------- Phase B greedy target layout ----------
        B = GlobalReconfigurations.phaseB_greedy_layout(A, T)[1]
        # print(f"A={A}, B={B}, T={T}")

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
        # print(
        #     f"[DEBUG _runOddEvenReconfig] Phase B start: m={m}, "
        #     f"current_vs_target_diffs={int(np.sum(A != T))}"
        # )

        # Execute ≤ m odd–even steps to realise the permutation per row
        for _ in range(m):
            oddpass = row_pass_by_rank(True, row_rank_phaseB)
            evenpass = row_pass_by_rank(False, row_rank_phaseB)
            timeElapsed += oddpass * row_swap_time
            timeElapsed += evenpass * row_swap_time
            acc_cost += int(oddpass) + int(evenpass)

        # print(f"A={A}, B={B}, T={T}")

        # ==========================================================
        # Phase C: vertical odd–even with k-way parallel buckets.
        # ==========================================================
        # print(
        #     f"[DEBUG _runOddEvenReconfig] Phase C start: k={k}, "
        #     f"current_vs_target_diffs={int(np.sum(A != T))}"
        # )
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

        # print(f"A={A}, B={B}, T={T}")

        # ==========================================================
        # Phase D: final row-wise odd–even to exact target order.
        # ==========================================================
        row_rank_final: List[Dict[int, int]] = []
        for r in range(n):
            row_rank_final.append({ion: idx for idx, ion in enumerate(T[r, :])})

        # print(
        #     f"[DEBUG _runOddEvenReconfig] Phase D start: rows={n}, "
        #     f"current_vs_target_diffs={int(np.sum(A != T))}"
        # )
        for _ in range(m):
            oddpass = row_pass_by_rank(True, row_rank_final)
            evenpass = row_pass_by_rank(False, row_rank_final)
            timeElapsed += oddpass * row_swap_time
            timeElapsed += evenpass * row_swap_time
            acc_cost += int(oddpass) + int(evenpass)

        # print(f"A={A}, B={B}, T={T}")

        # Final sanity: how close are we to the target layout?
        try:
            final_diff = int(np.sum(A != T))
        except Exception:
            final_diff = "NA"

        # print(
        #     f"[DEBUG _runOddEvenReconfig] Phase D end: acc_cost={acc_cost}, "
        #     f"final_layout_diffs={final_diff}"
        # )

        if not initial_placement:
            print(
                f"RECONFIGURATION: {acc_cost} passes were needed for the "
                f"current reconfiguration round, taking {timeElapsed} time and "
                f"{heatingRates} heating"
            )

        return heatingRates, timeElapsed


 

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
            mp: Dict[int, int] = {}
            for c in range(m):
                ion = int(A[r, c])
                mp[ion] = c
            invpos.append(mp)

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
    



class Split(CrystalOperation):
    KEY = Operations.SPLIT
    SPLITTING_TIME = 80e-6  # TABLE I https://arxiv.org/pdf/2004.04706
    HEATING_RATE = (
        6  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.SPLITTING_TIME

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._crossing.ion]
        self._addOns = f" {self._crossing.ion.label}"

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasTrap(self._trap):
            return False
        if self._crossing.ion is not None:
            return False
        if len(self._trap.ions) == 0:
            return False
        if self._crossing.getEdgeIon(self._trap) != self._ion:
            return False
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        if not self._crossing.hasTrap(self._trap):
            raise ValueError(f"Split: crossing does not include trap {self._trap.idx}")
        if self._crossing.ion is not None:
            raise ValueError(
                f"Split: crossing is already occupied by ion {self._crossing.ion.idx}"
            )
        if len(self._trap.ions) == 0:
            raise ValueError(f"Split: trap {self._trap.idx} has no ions")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, trap: Trap, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            ion = crossing.getEdgeIon(trap)
            trap.removeIon(ion)
            crossing.setIon(ion, trap)
            trap.addMotionalEnergy(cls.HEATING_RATE * cls.SPLITTING_TIME)
            ion.addMotionalEnergy(cls.HEATING_RATE * cls.SPLITTING_TIME)

        return cls(
            run=lambda _: run(),
            ion=ion,
            trap=trap,
            crossing=crossing,
            involvedComponents=[trap, crossing, *crossing.connection],
        )


class Merge(CrystalOperation):
    KEY = Operations.MERGE
    MERGING_TIME = 80e-6  # TABLE I https://arxiv.org/pdf/2004.04706
    HEATING_RATE = (
        6  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.MERGING_TIME

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._crossing.getEdgeIon(self._trap)]
        self._addOns = f" {self._crossing.getEdgeIon(self._trap).label}"

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasTrap(self._trap):
            return False
        if self._crossing.ion is None:
            return False
        if self._crossing.ion != self._ion:
            return False
        return super().isApplicable

    def _checkApplicability(self) -> None:
        if not self._crossing.hasTrap(self._trap):
            raise ValueError(f"Merge: crossing does not include trap {self._trap.idx}")
        if self._crossing.ion is None:
            raise ValueError(f"Merge: crossing is empty")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, trap: Trap, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            ion = crossing.ion
            crossing.clearIon()
            edge_ion = crossing.getEdgeIon(trap) if trap.ions else None
            idx = trap.ions.index(edge_ion) if trap.ions else 0
            if len(trap.ions)==1:
                offset = 1 if ion.pos[0]-edge_ion.pos[0]+ion.pos[1]-edge_ion.pos[1]>0 else 0
                adjacentIon=None
            else:
                offset=idx>0
                adjacentIon=edge_ion
            trap.addIon(ion, adjacentIon=adjacentIon, offset=offset)
            trap.addMotionalEnergy(cls.HEATING_RATE * cls.MERGING_TIME)

        return cls(
            run=lambda _: run(),
            ion=ion,
            crossing=crossing,
            trap=trap,
            involvedComponents=[trap, crossing, *crossing.connection],
        )


class CrystalRotation(CrystalOperation):
    KEY = Operations.CRYSTAL_ROTATION
    ROTATION_TIME = (
        42e-6  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )
    HEATING_RATE = (
        0.3  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], bool],
        trap: Trap,
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._trap: Trap = trap

    def calculateOperationTime(self) -> None:
        self._operationTime = self.ROTATION_TIME

    @property
    def isApplicable(self) -> bool:
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(cls, trap: Trap):
        def run():
            ions = list(trap.ions).copy()[::-1]
            for ion in ions:
                trap.removeIon(ion)
            for i, ion in enumerate(ions):
                trap.addIon(ion, offset=i)
            trap.addMotionalEnergy(cls.HEATING_RATE * cls.ROTATION_TIME)

        return cls(
            run=lambda _: run(),
            trap=trap,
            involvedComponents=[trap],
        )



class CoolingOperation(CrystalOperation):
    KEY = Operations.RECOOLING
    COOLING_TIME = 400-6  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    HEATING_RATE = (
        0.1  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
    
    def calculateOperationTime(self) -> None:
        self._operationTime =  self.COOLING_TIME

    @property
    def isApplicable(self) -> bool:
        if not self._trap.hasCoolingIon:
            return False
        return super().isApplicable

    def _checkApplicability(self) -> None:
        if not self._trap.hasCoolingIon:
            raise ValueError(f"CoolingOperation: trap {self._trap.idx} does not include a cooling ion")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, trap: Trap
    ):
        def run():
            trap.coolTrap()
            trap.addMotionalEnergy(cls.HEATING_RATE * cls.COOLING_TIME)

        return cls(
            run=lambda _: run(),
            trap=trap,
            involvedComponents=[trap],
        )




class Move(Operation):
    KEY = Operations.MOVE
    MOVING_TIME = 5e-6  # TABLE I https://arxiv.org/pdf/2004.04706
    HEATING_RATE = (
        0.1  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.MOVING_TIME

    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        self._dephasingFidelity= 1 # little to no idling due to shuttling being fast

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._crossing.ion]
        self._addOns = f" {self._crossing.ion.label}"

    @property
    def isApplicable(self) -> bool:
        return bool(self._crossing.ion) and self._ion == self._crossing.ion and super().isApplicable

    def _checkApplicability(self) -> None:
        if not self._crossing.ion:
            raise ValueError(f"Move: crossing does not contain ion")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(cls, crossing: Crossing, ion: Optional[Ion] = None):
        def run():
            crossing.ion.addMotionalEnergy(cls.HEATING_RATE * cls.MOVING_TIME)
            crossing.moveIon()

        return cls(
            run=lambda _: run(),
            ion=ion,
            crossing=crossing,
            involvedComponents=[crossing],
        )

# TODO: junction crossing should really go over the junction to the next crossing
class JunctionCrossing(Operation):
    KEY = Operations.JUNCTION_CROSSING
    CROSSING_TIME = 50e-6  # TABLE I https://arxiv.org/pdf/2004.04706
    HEATING_RATE = (
        3  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._junction: Junction = kwargs["junction"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.CROSSING_TIME

    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        # FIXME might be inaccurate
        self.calculateOperationTime()
        self._dephasingFidelity = 1 - (1-np.exp(-self.operationTime()/2.2))/2  # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._ion] if self._ion else []
        self._addOns = f" {self._ion.label}" if self._ion else ""

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasJunction(self._junction):
            return False
        if not self._crossing.ion and len(self._junction.ions) == 0:
            return False
        if self._crossing.ion and self._crossing.ion != self._ion:
            return False
        if self._junction.ions and self._junction.ions[0] != self._ion:
            return False
        if self._crossing.ion and len(self._junction.ions) == self._junction.DEFAULT_CAPACITY:
            return False
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        if not self._crossing.hasJunction(self._junction):
            raise ValueError(
                f"JunctionCrossing: crossing does not contain junction {self._junction.idx}"
            )
        if not self._crossing.ion and len(self._junction.ions) == 0:
            raise ValueError(
                f"JunctionCrossing: neither junction nor crossing has an ion"
            )
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, junction: Junction, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            if not crossing.ion and len(junction.ions) > 0:
                ion = junction.ions[0]
                crossing.setIon(ion, junction)
                junction.removeIon(ion)
            else:
                ion = crossing.ion
                crossing.clearIon()
                junction.addIon(ion)
            ion.addMotionalEnergy(cls.HEATING_RATE * cls.CROSSING_TIME)

        return cls(
            run=lambda _: run(),
            ion=ion,
            junction=junction,
            crossing=crossing,
            involvedComponents=[junction, crossing],
        )






class PhysicalCrossingSwap(Operation):
    KEY = Operations.JUNCTION_CROSSING
    CROSSING_TIME = 100e-6  # TABLE I https://arxiv.org/pdf/2004.04706
    HEATING_RATE = (
        3  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._junction: Junction = kwargs["junction"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.CROSSING_TIME

    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        # FIXME might be inaccurate
        self.calculateOperationTime()
        self._dephasingFidelity = 1 - (1-np.exp(-self.operationTime()/2.2))/2  # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._ion] if self._ion else []
        self._addOns = f" {self._ion.label}" if self._ion else ""

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasJunction(self._junction):
            return False
        if not self._crossing.ion and len(self._junction.ions) == 0:
            return False
        if self._crossing.ion and self._crossing.ion != self._ion:
            return False
        if self._junction.ions and self._junction.ions[0] != self._ion:
            return False
        if self._crossing.ion and len(self._junction.ions) == self._junction.DEFAULT_CAPACITY:
            return False
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        if not self._crossing.hasJunction(self._junction):
            raise ValueError(
                f"JunctionCrossing: crossing does not contain junction {self._junction.idx}"
            )
        if not self._crossing.ion and len(self._junction.ions) == 0:
            raise ValueError(
                f"JunctionCrossing: neither junction nor crossing has an ion"
            )
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, junction: Junction, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            if not crossing.ion and len(junction.ions) > 0:
                ion = junction.ions[0]
                crossing.setIon(ion, junction)
                junction.removeIon(ion)
            else:
                ion = crossing.ion
                crossing.clearIon()
                junction.addIon(ion)
            ion.addMotionalEnergy(cls.HEATING_RATE * cls.CROSSING_TIME)

        return cls(
            run=lambda _: run(),
            ion=ion,
            junction=junction,
            crossing=crossing,
            involvedComponents=[junction, crossing],
        )


class ParallelOperation(Operation):
    KEY = Operations.PARALLEL

    def __init__(
        self, run: Callable[[Any], bool], operations: Sequence[Operation], **kwargs
    ) -> None:
        super().__init__(run, **kwargs, operations=operations)
        self._operations = operations

    def calculateOperationTime(self) -> None:
        for op in self._operations:
            op.calculateOperationTime()
        self._operationTime = max(op.operationTime() for op in self._operations)

    def calculateDephasingFidelity(self) -> None:
        for op in self._operations:
            op.calculateDephasingFidelity()
        self._dephasingFidelity = float(max([op.dephasingFidelity() for op in self._operations]))


    def calculateFidelity(self) -> None:
        for op in self._operations:
            op.calculateFidelity()
        # assuming independence between parallel operations
        self._fidelity = float(np.prod([op.fidelity() for op in self._operations]))

    def _generateLabelAddOns(self) -> None:
        self._addOns = ""
        for op in self._operations:
            self._addOns += f" {op.KEY.name}"

    @property
    def isApplicable(self) -> bool:
        return all(op.isApplicable for op in self.operations)
    
    def _checkApplicability(self) -> None:
        return True

    @property
    def operations(self) -> Sequence[Operation]:
        return self._operations

    @classmethod
    def physicalOperation(cls, operationsToStart: Sequence[Operation], operationsStarted: Sequence[Operation]):
        def run():
            for op in np.random.permutation(operationsToStart):
                op.run()

        involvedComponents = []
        operations = list(operationsStarted)+list(operationsToStart)
        for op in operations:
            involvedComponents += list(op.involvedComponents)
        return cls(
            run=lambda _: run(),
            operations=operations,
            involvedComponents=set(involvedComponents),
        )
    
