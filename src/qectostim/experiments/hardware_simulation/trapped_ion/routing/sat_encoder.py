# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/sat_encoder.py
"""
WISE SAT encoder for trapped ion grid routing.

This module provides the core SAT encoding for WISE grid architectures:
- WISESATContext: Encapsulates encoding parameters and layout info
- WISESATEncoder: Implements GridSATEncoder for WISE grids

The encoder translates routing problems into CNF formulas using:
- Odd-even transposition sort networks (H-V phases)
- Permutation constraints
- Phase monotonicity
- Swap semantics and copy constraints
- Gate pair adjacency requirements

SAT Solver Requirements:
    - pysat: For CNF/WCNF formula construction and solving
    - Optional: RC2 for MaxSAT optimization
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np

from qectostim.experiments.hardware_simulation.core.sat_interface import (
    GridSATEncoder,
    SATSolution,
    PlacementRequirement,
)

# Import from sibling modules
from qectostim.experiments.hardware_simulation.trapped_ion.routing.config import (
    WISE_LOGGER_NAME,
    wise_logger,
    WISERoutingConfig,
    _PYSAT_AVAILABLE,
)
from qectostim.experiments.hardware_simulation.trapped_ion.routing.layout_utils import (
    NoFeasibleLayoutError,
    pre_sat_sanity_checks as _pre_sat_sanity_checks,
)
from qectostim.experiments.hardware_simulation.trapped_ion.routing.solvers import (
    run_sat_with_timeout as _run_sat_with_timeout,
    run_rc2_with_timeout as _run_rc2_with_timeout,
    _run_solver_in_subprocess,
    _sat_subprocess_worker,
    _rc2_subprocess_worker,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.architecture import (
        HardwareArchitecture,
    )

# pysat imports (conditional)
if _PYSAT_AVAILABLE:
    from pysat.formula import IDPool, CNF, WCNF
    from pysat.card import CardEnc, EncType
    from pysat.solvers import Minisat22, Solver
else:
    IDPool = None  # type: ignore[misc,assignment]
    CNF = None  # type: ignore[misc,assignment]
    WCNF = None  # type: ignore[misc,assignment]
    CardEnc = None  # type: ignore[misc,assignment]
    EncType = None  # type: ignore[misc,assignment]
    Minisat22 = None  # type: ignore[misc,assignment]
    Solver = None  # type: ignore[misc,assignment]


# =============================================================================
# WISESATContext - Encoding Context
# =============================================================================

@dataclass
class WISESATContext:
    """Context for WISE SAT encoding.
    
    Encapsulates all parameters needed for SAT constraint generation.
    This is the refactored version of _WiseSatBuilderContext from the old code.
    
    Attributes
    ----------
    initial_layout : np.ndarray
        Initial ion arrangement (n x m array of ion IDs).
    target_positions : List[Dict[int, Tuple[int, int]]]
        Per-round target positions for BT (block target) ions.
    gate_pairs : List[List[Tuple[int, int]]]
        Per-round list of ion pairs that need to interact.
    full_gate_pairs : List[List[Tuple[int, int]]]
        Full gate pairs including all rounds (for constraint generation).
    ions : List[int]
        List of all ion IDs.
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    num_rounds : int
        Number of routing rounds.
    block_cells : List[List[Tuple[int, int]]]
        Cells in each gating block.
    block_fully_inside : List[bool]
        Whether each block is fully inside the grid.
    block_widths : List[int]
        Width of each gating block.
    num_blocks : int
        Total number of blocks.
    debug_diag : bool
        Enable diagnostic logging.
    grid_origin : Tuple[int, int]
        Grid origin for patch-based routing (row_offset, col_offset).
    cross_boundary_prefs : Optional[List[Dict[int, Set[str]]]]
        Cross-boundary directional preferences per round.
    boundary_adjacent : Optional[Dict[str, bool]]
        Boundary adjacency flags.
    ignore_initial_reconfig : bool
        Whether the first round starts from an arbitrary layout.
    """
    initial_layout: np.ndarray
    target_positions: List[Dict[int, Tuple[int, int]]]
    gate_pairs: List[List[Tuple[int, int]]]
    full_gate_pairs: List[List[Tuple[int, int]]]
    ions: List[int]
    n_rows: int
    n_cols: int
    num_rounds: int
    block_cells: List[List[Tuple[int, int]]]
    block_fully_inside: List[bool]
    block_widths: List[int]
    num_blocks: int
    debug_diag: bool = False
    grid_origin: Tuple[int, int] = (0, 0)
    cross_boundary_prefs: Optional[List[Dict[int, Set[str]]]] = None
    boundary_adjacent: Optional[Dict[str, bool]] = None
    ignore_initial_reconfig: bool = False


# =============================================================================
# WISESATEncoder - Core SAT Encoder
# =============================================================================

class WISESATEncoder(GridSATEncoder):
    """SAT encoder for WISE grid-based ion routing.
    
    Implements the GridSATEncoder interface with WISE-specific constraints:
    - Odd-even transposition sort network (H-V phases)
    - Permutation constraints (each ion at exactly one position)
    - Phase monotonicity (H phases before V phases within each round)
    - Swap semantics (correct ion exchange under swap operations)
    - Copy constraints (ions stay put when not swapped)
    - Gate pair adjacency (pairs must be in same block at round end)
    
    This is the refactored port of _wise_build_structural_cnf from
    old/utils/qccd_operations.py.
    
    Example
    -------
    >>> encoder = WISESATEncoder(rows=4, cols=4)
    >>> encoder.initialize(context)  # WISESATContext with layout info
    >>> encoder.add_permutation_constraints()
    >>> encoder.add_phase_constraints("H", 0)
    >>> encoder.add_swap_constraints()
    >>> encoder.add_gate_pair_constraints()
    >>> solution = encoder.solve(timeout=30.0)
    
    Attributes
    ----------
    vpool : IDPool
        pysat variable pool for unique variable IDs.
    formula : CNF | WCNF
        The SAT formula being built.
    config : WISERoutingConfig
        Configuration parameters.
    context : Optional[WISESATContext]
        Current encoding context.
    """
    
    def __init__(
        self,
        rows: int,
        cols: int,
        config: Optional[WISERoutingConfig] = None,
        use_maxsat: bool = False,
    ):
        """Initialize WISE SAT encoder.
        
        Parameters
        ----------
        rows : int
            Number of rows in the grid.
        cols : int
            Number of columns in the grid.
        config : Optional[WISERoutingConfig]
            Configuration parameters. Uses defaults if None.
        use_maxsat : bool
            If True, use WCNF for MaxSAT optimization.
        """
        super().__init__(rows, cols, layers=1)
        
        if not _PYSAT_AVAILABLE:
            raise ImportError(
                "pysat is required for WISESATEncoder. "
                "Install with: pip install python-sat"
            )
        
        self.config = config or WISERoutingConfig()
        self.use_maxsat = use_maxsat
        
        # pysat structures
        self.vpool: IDPool = IDPool()
        self.formula = WCNF() if self.use_maxsat else CNF()
        
        # Context (set by initialize())
        self.context: Optional[WISESATContext] = None
        
        # Pass bounds (can vary per round)
        self._pass_bounds: List[int] = []
        
        # Core group tracking for UNSAT debugging
        self._groups_enabled = False
        self._group_selectors: Dict[str, int] = {}
        self._group_meta: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    # -------------------------------------------------------------------------
    # Variable Helpers
    # -------------------------------------------------------------------------
    
    def var_a(self, r: int, p: int, krow: int, jcol: int, ion: int) -> int:
        """Variable: ion is at position (krow, jcol) at round r, pass p."""
        return self.vpool.id(("a", r, p, krow, jcol, ion))
    
    def var_s_h(self, r: int, p: int, krow: int, jcol: int) -> int:
        """Variable: horizontal swap at (krow, jcol)-(krow, jcol+1) in round r, pass p."""
        return self.vpool.id(("s_h", r, p, krow, jcol))
    
    def var_s_v(self, r: int, p: int, krow: int, jcol: int) -> int:
        """Variable: vertical swap at (krow, jcol)-(krow+1, jcol) in round r, pass p."""
        return self.vpool.id(("s_v", r, p, krow, jcol))
    
    def var_phase(self, r: int, p: int) -> int:
        """Variable: phase indicator (False=H, True=V) for round r, pass p."""
        return self.vpool.id(("phase", r, p))
    
    def var_row_end(self, r: int, ion: int, row: int) -> int:
        """Variable: ion is in row `row` at end of round r."""
        return self.vpool.id(("row_end", r, ion, row))
    
    def var_block_end(self, r: int, ion: int, block: int) -> int:
        """Variable: ion is in block `block` at end of round r."""
        return self.vpool.id(("w_end", r, ion, block))
    
    def var_u(self, r: int, p: int) -> int:
        """Variable: any comparator active at round r, pass p."""
        return self.vpool.id(("u", r, p))
    
    # -------------------------------------------------------------------------
    # Clause Addition Helpers
    # -------------------------------------------------------------------------
    
    def _add_hard(
        self,
        clause: List[int],
        group: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a hard clause to the formula."""
        if self.use_maxsat:
            self.formula.append(clause)
        elif self._groups_enabled and group:
            if group not in self._group_selectors:
                self._group_selectors[group] = self.vpool.id(("grp", group))
            sel = self._group_selectors[group]
            self.formula.append(clause + [sel])
            if meta:
                self._group_meta[group].append(meta)
        else:
            self.formula.append(clause)
    
    def _add_soft(self, clause: List[int], weight: int = 1) -> None:
        """Add a soft clause (only valid for WCNF)."""
        if not self.use_maxsat:
            raise RuntimeError("Soft clauses require MaxSAT mode (use_maxsat=True)")
        self.formula.append(clause, weight=weight)
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    
    def initialize(
        self,
        context: WISESATContext,
        pass_bound: int,
        sum_bound_B: Optional[int] = None,
    ) -> None:
        """Initialize the encoder with routing context.
        
        Parameters
        ----------
        context : WISESATContext
            Context containing layout and requirement info.
        pass_bound : int
            Maximum number of passes per round.
        sum_bound_B : Optional[int]
            If set, constrain the total number of *active* passes
            across all optimized rounds to at most this value.
        """
        self.context = context
        self.rows = context.n_rows
        self.cols = context.n_cols
        self._items = list(context.ions)
        self._sum_bound_B = sum_bound_B
        
        # =================================================================
        # CRITICAL: Normalize boundary_adjacent and cross_boundary_prefs
        # to match OLD code behavior (qccd_operations.py _wise_build_structural_cnf)
        # =================================================================
        R = context.num_rounds
        
        # Normalize boundary_adjacent — OLD code defaults to all True
        if context.boundary_adjacent is None:
            context.boundary_adjacent = {
                "top": True,
                "bottom": True,
                "left": True,
                "right": True,
            }
        else:
            context.boundary_adjacent = {
                "top": bool(context.boundary_adjacent.get("top", False)),
                "bottom": bool(context.boundary_adjacent.get("bottom", False)),
                "left": bool(context.boundary_adjacent.get("left", False)),
                "right": bool(context.boundary_adjacent.get("right", False)),
            }
        
        # Normalize cross_boundary_prefs — OLD code creates empty dicts for
        # missing rounds and converts direction lists to sets
        if context.cross_boundary_prefs is None:
            context.cross_boundary_prefs = [dict() for _ in range(R)]
        else:
            normalized_prefs: List[Dict[int, Set[str]]] = []
            for r in range(R):
                prefs_r = (
                    context.cross_boundary_prefs[r]
                    if r < len(context.cross_boundary_prefs)
                    else {}
                )
                normalized: Dict[int, Set[str]] = {}
                for ion, dirs in prefs_r.items():
                    normalized[ion] = set(dirs)
                normalized_prefs.append(normalized)
            context.cross_boundary_prefs = normalized_prefs
        
        # Pre-SAT sanity checks
        bt_soft_enabled = (
            self.use_maxsat
            and self.config.bt_soft_weight > 0
            and context.target_positions
            and any(bt for bt in context.target_positions)
        )
        try:
            _pre_sat_sanity_checks(
                context,
                bt_soft_enabled=bt_soft_enabled,
                capacity=self.config.capacity if hasattr(self.config, 'capacity') else 2,
            )
        except NoFeasibleLayoutError:
            raise
        
        # Enable UNSAT core debugging groups when debug_mode is on.
        if self.config.debug_mode and not self.use_maxsat:
            self._groups_enabled = True
            self._group_selectors.clear()
            self._group_meta.clear()
        
        # Determine which rounds get extra passes
        n, m = context.n_rows, context.n_cols
        R = context.num_rounds
        optimize_round_start = (
            1 if (context.ignore_initial_reconfig and R > 0) else 0
        )
        self._optimize_round_start = optimize_round_start
        
        # Variable P_bounds
        self._pass_bounds = (
            [pass_bound + n + m] * optimize_round_start
            + [pass_bound] * (R - optimize_round_start)
        )
    
    # -------------------------------------------------------------------------
    # GridSATEncoder Interface Implementation
    # -------------------------------------------------------------------------
    
    def add_placement_variables(self) -> None:
        """Create placement variables (lazy via vpool)."""
        pass
    
    def add_permutation_constraints(self) -> None:
        """Add permutation constraints: each ion at exactly one position."""
        if self.context is None:
            raise RuntimeError("Must call initialize() before adding constraints")
        
        ctx = self.context
        R = ctx.num_rounds
        n, m = ctx.n_rows, ctx.n_cols
        ions = ctx.ions
        
        # (0a) Exactly one ion per cell at each (r, p)
        for r in range(R):
            P_bound = self._pass_bounds[r]
            for p in range(P_bound + 1):
                for krow in range(n):
                    for jcol in range(m):
                        lits = [self.var_a(r, p, krow, jcol, ion) for ion in ions]
                        enc = CardEnc.equals(
                            lits=lits,
                            encoding=EncType.ladder,
                            vpool=self.vpool,
                        )
                        for cl in enc.clauses:
                            self._add_hard(cl, f"CARD_CELL:r{r}")
        
        # Final state at (R, 0)
        for krow in range(n):
            for jcol in range(m):
                lits = [self.var_a(R, 0, krow, jcol, ion) for ion in ions]
                enc = CardEnc.equals(
                    lits=lits,
                    encoding=EncType.ladder,
                    vpool=self.vpool,
                )
                for cl in enc.clauses:
                    self._add_hard(cl, "CARD_CELL:FINAL")
        
        # (0b) Each ion in exactly one cell at each (r, p)
        for r in range(R):
            P_bound = self._pass_bounds[r]
            for p in range(P_bound + 1):
                for ion in ions:
                    lits = [
                        self.var_a(r, p, krow, jcol, ion)
                        for krow in range(n)
                        for jcol in range(m)
                    ]
                    enc = CardEnc.equals(
                        lits=lits,
                        encoding=EncType.ladder,
                        vpool=self.vpool,
                    )
                    for cl in enc.clauses:
                        self._add_hard(cl, f"CARD_ION:r{r}")
        
        # Final state
        for ion in ions:
            lits = [
                self.var_a(R, 0, krow, jcol, ion)
                for krow in range(n)
                for jcol in range(m)
            ]
            enc = CardEnc.equals(
                lits=lits,
                encoding=EncType.ladder,
                vpool=self.vpool,
            )
            for cl in enc.clauses:
                self._add_hard(cl, "CARD_ION:FINAL")
    
    def add_initial_layout_constraints(self) -> None:
        """Add constraints fixing the initial layout."""
        if self.context is None:
            raise RuntimeError("Must call initialize() before adding constraints")
        
        ctx = self.context
        A_in = ctx.initial_layout
        n, m = ctx.n_rows, ctx.n_cols
        ions = ctx.ions
        
        for krow in range(n):
            for jcol in range(m):
                ion0 = int(A_in[krow, jcol])
                self._add_hard([self.var_a(0, 0, krow, jcol, ion0)], "INIT")
                for ion in ions:
                    if ion != ion0:
                        self._add_hard([-self.var_a(0, 0, krow, jcol, ion)], "INIT")
    
    def add_round_chaining_constraints(self) -> None:
        """Add constraints linking end of round r to start of round r+1."""
        if self.context is None:
            raise RuntimeError("Must call initialize() before adding constraints")
        
        ctx = self.context
        R = ctx.num_rounds
        n, m = ctx.n_rows, ctx.n_cols
        ions = ctx.ions
        
        for r in range(R):
            P_end = self._pass_bounds[r]
            for krow in range(n):
                for jcol in range(m):
                    for ion in ions:
                        a_next = self.var_a(r + 1, 0, krow, jcol, ion)
                        a_end = self.var_a(r, P_end, krow, jcol, ion)
                        self._add_hard([-a_next, a_end], f"CHAIN:r{r}")
                        self._add_hard([-a_end, a_next], f"CHAIN:r{r}")
    
    def add_row_sorting_constraints(self, row: int, phase: int) -> None:
        """Add constraints for horizontal sorting within a row."""
        pass  # Implemented in add_swap_constraints()
    
    def add_column_sorting_constraints(self, col: int, phase: int) -> None:
        """Add constraints for vertical sorting within a column."""
        pass  # Implemented in add_swap_constraints()
    
    def add_phase_constraints(self, phase_type: str, phase_number: int) -> None:
        """Add phase constraints (monotonicity)."""
        if self.context is None:
            raise RuntimeError("Must call initialize() before adding constraints")
        
        ctx = self.context
        R = ctx.num_rounds
        
        for r in range(R):
            P_bound = self._pass_bounds[r]
            if P_bound > 1:
                for p in range(P_bound - 1):
                    self._add_hard(
                        [-self.var_phase(r, p), self.var_phase(r, p + 1)],
                        "PHASE_MONO"
                    )
    
    def add_swap_constraints(self) -> None:
        """Add all swap and copy constraints for the odd-even transposition network."""
        if self.context is None:
            raise RuntimeError("Must call initialize() before adding constraints")
        
        ctx = self.context
        R = ctx.num_rounds
        n, m = ctx.n_rows, ctx.n_cols
        ions = ctx.ions
        
        for r in range(R):
            P_bound = self._pass_bounds[r]
            for p in range(P_bound):
                phase_p = self.var_phase(r, p)
                
                # --- Horizontal comparators ---
                for krow in range(n):
                    for jcol in range(m - 1):
                        sh = self.var_s_h(r, p, krow, jcol)
                        
                        self._add_hard([-phase_p, -sh], "H_GATE")
                        if jcol % 2 != p % 2:
                            self._add_hard([-sh], "H_GATE")
                        
                        for ion in ions:
                            a_cur_j = self.var_a(r, p, krow, jcol, ion)
                            a_cur_j1 = self.var_a(r, p, krow, jcol + 1, ion)
                            a_next_j = self.var_a(r, p + 1, krow, jcol, ion)
                            a_next_j1 = self.var_a(r, p + 1, krow, jcol + 1, ion)
                            
                            self._add_hard([-sh, -a_cur_j1, a_next_j], "H_SEM")
                            self._add_hard([-sh, -a_cur_j, a_next_j1], "H_SEM")
                
                # --- Vertical comparators ---
                for krow in range(n - 1):
                    for jcol in range(m):
                        sv = self.var_s_v(r, p, krow, jcol)
                        
                        self._add_hard([phase_p, -sv], "V_GATE")
                        if krow % 2 != p % 2:
                            self._add_hard([-sv], "V_GATE")
                        
                        for ion in ions:
                            a_cur_top = self.var_a(r, p, krow, jcol, ion)
                            a_cur_bot = self.var_a(r, p, krow + 1, jcol, ion)
                            a_next_top = self.var_a(r, p + 1, krow, jcol, ion)
                            a_next_bot = self.var_a(r, p + 1, krow + 1, jcol, ion)
                            
                            self._add_hard([-sv, -a_cur_bot, a_next_top], "V_SEM")
                            self._add_hard([-sv, -a_cur_top, a_next_bot], "V_SEM")
        
        self._add_copy_constraints()
    
    def _add_copy_constraints(self) -> None:
        """Add copy constraints for non-swapping cells."""
        if self.context is None:
            return
        
        ctx = self.context
        R = ctx.num_rounds
        n, m = ctx.n_rows, ctx.n_cols
        ions = ctx.ions
        
        # Horizontal phase copy constraints
        for r in range(R):
            P_bound = self._pass_bounds[r]
            for p in range(P_bound):
                phase_p = self.var_phase(r, p)
                
                for krow in range(n):
                    for jcol in range(m):
                        s_left = self.var_s_h(r, p, krow, jcol - 1) if jcol > 0 else None
                        s_right = self.var_s_h(r, p, krow, jcol) if jcol < m - 1 else None
                        
                        lits_ante_neg = [phase_p]
                        if s_left is not None:
                            lits_ante_neg.append(s_left)
                        if s_right is not None:
                            lits_ante_neg.append(s_right)
                        
                        for ion in ions:
                            a_cur = self.var_a(r, p, krow, jcol, ion)
                            a_next = self.var_a(r, p + 1, krow, jcol, ion)
                            
                            self._add_hard(lits_ante_neg + [-a_cur, a_next], "H_COPY")
                            self._add_hard(lits_ante_neg + [-a_next, a_cur], "H_COPY")
        
        # Vertical phase copy constraints
        for r in range(R):
            P_bound = self._pass_bounds[r]
            for p in range(P_bound):
                phase_p = self.var_phase(r, p)
                
                for krow in range(n):
                    for jcol in range(m):
                        s_up = self.var_s_v(r, p, krow - 1, jcol) if krow > 0 else None
                        s_down = self.var_s_v(r, p, krow, jcol) if krow < n - 1 else None
                        
                        lits_ante_neg = [-phase_p]
                        if s_up is not None:
                            lits_ante_neg.append(s_up)
                        if s_down is not None:
                            lits_ante_neg.append(s_down)
                        
                        for ion in ions:
                            a_cur = self.var_a(r, p, krow, jcol, ion)
                            a_next = self.var_a(r, p + 1, krow, jcol, ion)
                            
                            self._add_hard(lits_ante_neg + [-a_cur, a_next], "V_COPY")
                            self._add_hard(lits_ante_neg + [-a_next, a_cur], "V_COPY")
    
    def add_target_position_constraints(self, use_soft: bool = False) -> None:
        """Add constraints for target (BT) positions."""
        if self.context is None:
            return
        
        ctx = self.context
        R = ctx.num_rounds
        n, m = ctx.n_rows, ctx.n_cols
        BT = ctx.target_positions
        
        use_soft_bt = use_soft and self.use_maxsat and self.config.bt_soft_weight > 0
        
        for r in range(R):
            P_final = self._pass_bounds[r]
            for ion, (d_fix, c_fix) in BT[r].items():
                if ion not in ctx.ions:
                    continue
                
                pin_lit = self.var_a(r, P_final, d_fix, c_fix, ion)
                
                if use_soft_bt:
                    self._add_soft([pin_lit], weight=self.config.bt_soft_weight)
                else:
                    self._add_hard([pin_lit], "BT")
                    for krow in range(n):
                        for jcol in range(m):
                            if (krow, jcol) != (d_fix, c_fix):
                                self._add_hard(
                                    [-self.var_a(r, P_final, krow, jcol, ion)],
                                    "BT"
                                )
    
    def add_gate_pair_constraints(self) -> None:
        """Add constraints requiring gate pairs to be adjacent at round end."""
        if self.context is None:
            return
        
        ctx = self.context
        R = ctx.num_rounds
        n, m = ctx.n_rows, ctx.n_cols
        P_arr = ctx.gate_pairs
        
        for r in range(R):
            for (i1, i2) in P_arr[r]:
                if i1 not in ctx.ions or i2 not in ctx.ions:
                    continue
                
                # Same row at end of round
                for d in range(n):
                    re1 = self.var_row_end(r, i1, d)
                    re2 = self.var_row_end(r, i2, d)
                    self._add_hard([-re1, re2], f"PAIR_REQ:r{r}")
                    self._add_hard([-re2, re1], f"PAIR_REQ:r{r}")
                
                # Same block at end of round
                for b in range(ctx.num_blocks):
                    we1 = self.var_block_end(r, i1, b)
                    we2 = self.var_block_end(r, i2, b)
                    self._add_hard([-we1, we2], f"PAIR_REQ:r{r}")
                    self._add_hard([-we2, we1], f"PAIR_REQ:r{r}")

    def add_displacement_soft_constraints(self) -> None:
        """Add soft clauses penalising large ion displacement."""
        if self.context is None or not self.use_maxsat:
            return

        ctx = self.context
        R = ctx.num_rounds
        n, m = ctx.n_rows, ctx.n_cols

        w_row = self.config.boundary_soft_weight_row
        w_col = self.config.boundary_soft_weight_col

        if w_row <= 0 and w_col <= 0:
            return

        for r in range(R):
            P_final = self._pass_bounds[r]

            for ion in ctx.ions:
                if w_row > 0:
                    init_layout = ctx.initial_layout
                    home_row = None
                    for dr in range(n):
                        for dc in range(m):
                            if init_layout[dr][dc] == ion:
                                home_row = dr
                                break
                        if home_row is not None:
                            break
                    if home_row is not None:
                        self._add_soft(
                            [self.var_row_end(r, ion, home_row)],
                            weight=w_row,
                        )

                if w_col > 0:
                    init_layout = ctx.initial_layout
                    home_block = None
                    for dr in range(n):
                        for dc in range(m):
                            if init_layout[dr][dc] == ion:
                                if ctx.block_widths:
                                    cumw = 0
                                    for bidx, bw in enumerate(ctx.block_widths):
                                        cumw += bw
                                        if dc < cumw:
                                            home_block = bidx
                                            break
                                break
                        if home_block is not None:
                            break
                    if home_block is not None:
                        self._add_soft(
                            [self.var_block_end(r, ion, home_block)],
                            weight=w_col,
                        )
    
    def add_row_block_linkage_constraints(self) -> None:
        """Add constraints linking row_end/block_end variables to positions."""
        if self.context is None:
            return
        
        ctx = self.context
        R = ctx.num_rounds
        n, m = ctx.n_rows, ctx.n_cols
        ions = ctx.ions
        block_cells = ctx.block_cells
        block_fully_inside = ctx.block_fully_inside
        block_widths = ctx.block_widths
        num_blocks = ctx.num_blocks
        
        for r in range(R):
            P_final = self._pass_bounds[r]
            
            for ion in ions:
                # row_end linkage
                for d in range(n):
                    re = self.var_row_end(r, ion, d)
                    cell_lits = [self.var_a(r, P_final, d, j, ion) for j in range(m)]
                    self._add_hard([-re] + cell_lits, f"ROWBLOCK_LINK:r{r}")
                    for aj in cell_lits:
                        self._add_hard([-aj, re], f"ROWBLOCK_LINK:r{r}")
                
                # block_end linkage
                for b in range(num_blocks):
                    we = self.var_block_end(r, ion, b)
                    cells = [
                        self.var_a(r, P_final, d, j, ion)
                        for (d, j) in block_cells[b]
                    ]
                    
                    if block_fully_inside[b] or block_widths[b] > 1:
                        self._add_hard([-we] + cells, f"ROWBLOCK_LINK:r{r}")
                        for aj in cells:
                            self._add_hard([-aj, we], f"ROWBLOCK_LINK:r{r}")
                    else:
                        self._add_hard([-we], f"ROWBLOCK_LINK:r{r}")
    
    def add_adjacency_requirements(
        self,
        requirements: List[PlacementRequirement],
    ) -> None:
        """Add adjacency requirements (interface compliance)."""
        pass
    
    def add_pass_usage_constraints(self) -> None:
        """Add pass-usage (u-variable) constraints and global sum bound."""
        if self.context is None:
            return

        ctx = self.context
        R = ctx.num_rounds
        n, m = ctx.n_rows, ctx.n_cols
        optimize_round_start = getattr(self, '_optimize_round_start', 0)

        for r in range(R):
            P_bound = self._pass_bounds[r]
            for p in range(P_bound):
                u_rp = self.var_u(r, p)

                comp_lits: List[int] = []
                for krow in range(n):
                    for jcol in range(m - 1):
                        comp_lits.append(self.var_s_h(r, p, krow, jcol))
                for krow in range(n - 1):
                    for jcol in range(m):
                        comp_lits.append(self.var_s_v(r, p, krow, jcol))

                if not comp_lits:
                    self._add_hard([-u_rp], "UTIL_U")
                    continue

                self._add_hard([-u_rp] + comp_lits, "UTIL_U")
                for s_lit in comp_lits:
                    self._add_hard([-s_lit, u_rp], "UTIL_U")

        sum_bound = getattr(self, '_sum_bound_B', None)
        if sum_bound is not None and optimize_round_start < R:
            sum_u_lits: List[int] = []
            for r in range(optimize_round_start, R):
                for p in range(self._pass_bounds[r]):
                    sum_u_lits.append(self.var_u(r, p))

            total_slots = len(sum_u_lits)
            bound = min(sum_bound, total_slots)
            if bound < total_slots:
                card_enc = CardEnc.atmost(
                    lits=sum_u_lits,
                    bound=bound,
                    encoding=EncType.totalizer,
                    vpool=self.vpool,
                )
                for clause in card_enc.clauses:
                    self._add_hard(clause, "SUM_BOUND")

    def add_cross_boundary_constraints(self) -> None:
        """Add cross-boundary hard and soft constraints."""
        if self.context is None:
            return

        ctx = self.context
        R = ctx.num_rounds
        n, m = ctx.n_rows, ctx.n_cols
        prefs_list = ctx.cross_boundary_prefs
        boundary_adj = ctx.boundary_adjacent

        if not prefs_list or not boundary_adj:
            return
        if not any(boundary_adj.values()):
            return

        half_h = max(1, n // 2)
        half_w = max(1, m // 2)

        def _band_cells(directions: Set[str]) -> List[Tuple[int, int]]:
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

        factor = max(0.0, min(1.0, self.config.boundary_capacity_factor))
        dir_capacity: Dict[str, int] = {}
        if boundary_adj.get("top", False):
            dir_capacity["top"] = int(round(half_h * m * factor))
        if boundary_adj.get("bottom", False):
            dir_capacity["bottom"] = int(round(half_h * m * factor))
        if boundary_adj.get("left", False):
            dir_capacity["left"] = int(round(half_w * n * factor))
        if boundary_adj.get("right", False):
            dir_capacity["right"] = int(round(half_w * n * factor))

        ions_per_round_dir: Dict[Tuple[int, str], List[int]] = defaultdict(list)
        for r in range(R):
            prefs_r = prefs_list[r] if r < len(prefs_list) else {}
            for ion, dirs in prefs_r.items():
                if ion not in ctx.ions:
                    continue
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

        for r in range(R):
            prefs_r = prefs_list[r] if r < len(prefs_list) else {}
            if not prefs_r:
                continue
            P_final = self._pass_bounds[r]
            for ion in prefs_r.keys():
                if ion not in ctx.ions:
                    continue
                enforced_dirs = enforced_dirs_per_ion.get((r, ion))
                if not enforced_dirs:
                    continue
                cells = _band_cells(enforced_dirs)
                if not cells:
                    union: Set[Tuple[int, int]] = set()
                    for d in enforced_dirs:
                        union.update(_band_cells({d}))
                    cells = list(union)
                if not cells:
                    continue
                clause = [
                    self.var_a(r, P_final, d, c, ion)
                    for (d, c) in cells
                ]
                self._add_hard(clause, "CROSS_BOUNDARY")

        # Soft clauses
        if self.use_maxsat:
            w_row = self.config.boundary_soft_weight_row
            w_col = self.config.boundary_soft_weight_col
            if w_row <= 0 and w_col <= 0:
                return

            P_arr = ctx.gate_pairs
            BT = ctx.target_positions
            for r in range(R):
                inner_ions: Set[int] = set()
                for (i1, i2) in P_arr[r]:
                    inner_ions.add(i1)
                    inner_ions.add(i2)
                inner_ions.update(BT[r].keys())

                prefs_r = prefs_list[r] if r < len(prefs_list) else {}
                P_final = self._pass_bounds[r]

                for ion in ctx.ions:
                    dirs = prefs_r.get(ion)
                    if dirs:
                        for direction in dirs:
                            if direction in ("left", "right") and w_col > 0:
                                if not boundary_adj.get(direction, False):
                                    continue
                                target_col = 0 if direction == "left" else m - 1
                                lits = [
                                    self.var_a(r, P_final, d, target_col, ion)
                                    for d in range(n)
                                ]
                                if lits:
                                    self._add_soft(lits, weight=w_col)
                            if direction in ("top", "bottom") and w_row > 0:
                                if not boundary_adj.get(direction, False):
                                    continue
                                target_row = 0 if direction == "top" else n - 1
                                lits = [
                                    self.var_a(r, P_final, target_row, jcol, ion)
                                    for jcol in range(m)
                                ]
                                if lits:
                                    self._add_soft(lits, weight=w_row)

                    if ion in inner_ions:
                        if boundary_adj.get("left", False) and w_col > 0:
                            for d in range(n):
                                self._add_soft(
                                    [-self.var_a(r, P_final, d, 0, ion)],
                                    weight=w_col,
                                )
                        if boundary_adj.get("right", False) and w_col > 0:
                            for d in range(n):
                                self._add_soft(
                                    [-self.var_a(r, P_final, d, m - 1, ion)],
                                    weight=w_col,
                                )
                        if boundary_adj.get("top", False) and w_row > 0:
                            for jcol in range(m):
                                self._add_soft(
                                    [-self.var_a(r, P_final, 0, jcol, ion)],
                                    weight=w_row,
                                )
                        if boundary_adj.get("bottom", False) and w_row > 0:
                            for jcol in range(m):
                                self._add_soft(
                                    [-self.var_a(r, P_final, n - 1, jcol, ion)],
                                    weight=w_row,
                                )

    def add_all_constraints(
        self,
        skip_cardinality: bool = False,
        skip_pairs: bool = False,
        enable_displacement_soft: bool = False,
    ) -> None:
        """Add all standard WISE constraints.
        
        Parameters
        ----------
        skip_cardinality : bool
            Skip permutation/cardinality constraints.
        skip_pairs : bool
            Skip gate pair adjacency constraints.
        enable_displacement_soft : bool
            Enable displacement soft constraints (not in old code).
            Default False for faithful port of old behaviour.
        """
        if not skip_cardinality:
            self.add_permutation_constraints()
        
        self.add_initial_layout_constraints()
        self.add_round_chaining_constraints()
        self.add_phase_constraints("mono", 0)
        self.add_swap_constraints()
        self.add_row_block_linkage_constraints()
        self.add_pass_usage_constraints()
        self.add_cross_boundary_constraints()
        # NOTE: add_displacement_soft_constraints is NEW logic not in
        # the old ground truth. Only enable when explicitly requested.
        if enable_displacement_soft:
            self.add_displacement_soft_constraints()
        
        if any(bt for bt in self.context.target_positions if bt):
            use_soft = (
                self.use_maxsat
                and self.config.bt_soft_weight > 0
            )
            self.add_target_position_constraints(use_soft=use_soft)
        
        if not skip_pairs:
            self.add_gate_pair_constraints()
    
    # -------------------------------------------------------------------------
    # Solving
    # -------------------------------------------------------------------------
    
    def solve(
        self,
        timeout: Optional[float] = None,
        assumptions: Optional[List[int]] = None,
        in_process: bool = False,
    ) -> SATSolution:
        """Solve the SAT problem.
        
        Parameters
        ----------
        in_process : bool
            When True, run the solver in the current process (fast, avoids
            ~0.8 s fork+pickle overhead per call).  Use this for the inner
            config sweep loop.  MaxSAT always uses subprocess isolation.
        """
        if self.use_maxsat:
            return self._solve_maxsat(timeout)
        else:
            return self._solve_sat(timeout, assumptions, in_process=in_process)
    
    def _solve_sat(
        self,
        timeout: Optional[float] = None,
        assumptions: Optional[List[int]] = None,
        in_process: bool = False,
    ) -> SATSolution:
        """Solve using plain SAT.
        
        Parameters
        ----------
        timeout : float, optional
            Wall-clock timeout in seconds.
        assumptions : list of int, optional
            Extra assumptions to pass to the solver.
        in_process : bool
            If True, run Minisat22 in the current process (fast, no fork
            overhead).  Use this for the inner-loop config sweep where many
            calls are needed quickly.  Falls back to subprocess if False
            (the default).
        """
        t0 = time.time()

        if timeout is not None and timeout <= 0:
            return SATSolution(
                satisfiable=False,
                statistics={"status": "timeout_zero"},
            )

        # ==============================================================
        # CRITICAL FIX: When _groups_enabled is True, each clause was
        # added as [clause ∨ selector]. We MUST pass -selector for each
        # group to activate those clauses. Otherwise the solver sets all
        # selectors TRUE making clauses trivially satisfiable.
        # This matches OLD code behavior in qccd_operations.py.
        # ==============================================================
        final_assumptions = list(assumptions) if assumptions else []
        if self._groups_enabled and self._group_selectors:
            grp_assumptions = [-sel for sel in self._group_selectors.values()]
            final_assumptions.extend(grp_assumptions)
        
        try:
            if hasattr(self.formula, 'clauses'):
                clauses = self.formula.clauses
            elif hasattr(self.formula, 'hard'):
                clauses = self.formula.hard
            else:
                clauses = []

            eff_timeout = timeout if timeout is not None else 4800.0

            if in_process and _PYSAT_AVAILABLE:
                # ---- Fast in-process path ----
                # Avoids fork+pickle overhead (~0.8s per call).
                # Uses Minisat22's conflict budget as a heuristic timeout:
                # ~50k conflicts/sec is typical; scale by seconds.
                budget = int(eff_timeout * 200_000)
                try:
                    with Minisat22(bootstrap_with=clauses) as sat:
                        sat.conf_budget(budget)
                        # NOTE: solve_limited does NOT accept None for assumptions
                        solve_kwargs: Dict[str, Any] = {}
                        if final_assumptions:
                            solve_kwargs["assumptions"] = final_assumptions
                        sat_ok = (
                            sat.solve_limited(**solve_kwargs)
                            if hasattr(sat, 'solve_limited')
                            else sat.solve(**solve_kwargs)
                        )
                        model = sat.get_model() if sat_ok else None
                except Exception as exc:
                    return SATSolution(
                        satisfiable=False,
                        solve_time=time.time() - t0,
                        statistics={"status": "error", "error": str(exc)},
                    )

                t1 = time.time()
                if sat_ok and model:
                    self._last_model = model
                    item_positions = self._extract_positions(model)
                    return SATSolution(
                        satisfiable=True,
                        item_positions=item_positions,
                        cost=0.0,
                        solve_time=t1 - t0,
                        statistics={"status": "ok", "model_size": len(model)},
                    )
                elif sat_ok is None:
                    # Budget exhausted — treat as timeout / unknown
                    return SATSolution(
                        satisfiable=False,
                        solve_time=t1 - t0,
                        statistics={"status": "budget_exhausted"},
                    )
                else:
                    return SATSolution(
                        satisfiable=False,
                        solve_time=t1 - t0,
                        statistics={"status": "unsat"},
                    )

            # ---- Subprocess path (default) ----
            cnf_for_pickle = CNF()
            for cl in clauses:
                cnf_for_pickle.append(cl)

            result = _run_solver_in_subprocess(
                target_func=_sat_subprocess_worker,
                args=(cnf_for_pickle, final_assumptions if final_assumptions else None),
                timeout_s=eff_timeout,
                label="SAT",
            )

            t1 = time.time()
            status = result.get("status", "error")

            if status == "timeout":
                return SATSolution(
                    satisfiable=False,
                    solve_time=t1 - t0,
                    statistics={"status": "timeout"},
                )

            if status != "ok":
                return SATSolution(
                    satisfiable=False,
                    solve_time=t1 - t0,
                    statistics={"status": status,
                                "error": result.get("error", "")},
                )

            sat_ok = result.get("sat", False)
            model = result.get("model")

            if sat_ok and model:
                self._last_model = model
                item_positions = self._extract_positions(model)
                return SATSolution(
                    satisfiable=True,
                    item_positions=item_positions,
                    cost=0.0,
                    solve_time=t1 - t0,
                    statistics={"status": "ok", "model_size": len(model)},
                )
            else:
                return SATSolution(
                    satisfiable=False,
                    solve_time=t1 - t0,
                    statistics={"status": "unsat"},
                )

        except Exception as e:
            return SATSolution(
                satisfiable=False,
                solve_time=time.time() - t0,
                statistics={"status": "error", "error": str(e)},
            )

    def _solve_maxsat(self, timeout: Optional[float] = None) -> SATSolution:
        """Solve using MaxSAT via process isolation."""
        try:
            from pysat.examples.rc2 import RC2  # noqa: F811
        except ImportError:
            wise_logger.warning("RC2 not available, falling back to SAT")
            return self._solve_sat(timeout)

        t0 = time.time()
        eff_timeout = timeout if timeout is not None else 4800.0

        try:
            result = _run_solver_in_subprocess(
                target_func=_rc2_subprocess_worker,
                args=(self.formula,),
                timeout_s=eff_timeout,
                label="RC2",
            )

            t1 = time.time()
            status = result.get("status", "error")

            if status == "timeout":
                wise_logger.warning(
                    "MaxSAT RC2 timed out after %.1f s", eff_timeout
                )
                return SATSolution(
                    satisfiable=False,
                    solve_time=t1 - t0,
                    statistics={"status": "timeout"},
                )

            if status != "ok":
                return SATSolution(
                    satisfiable=False,
                    solve_time=t1 - t0,
                    statistics={"status": status,
                                "error": result.get("error", "")},
                )

            model = result.get("model")
            cost = result.get("cost", 0)

            if model:
                self._last_model = model
                item_positions = self._extract_positions(model)
                return SATSolution(
                    satisfiable=True,
                    item_positions=item_positions,
                    cost=float(cost) if cost is not None else 0.0,
                    solve_time=t1 - t0,
                    statistics={
                        "status": "ok",
                        "model_size": len(model),
                        "maxsat_cost": cost,
                    },
                )
            else:
                return SATSolution(
                    satisfiable=False,
                    solve_time=t1 - t0,
                    statistics={"status": "unsat"},
                )

        except Exception as e:
            return SATSolution(
                satisfiable=False,
                solve_time=time.time() - t0,
                statistics={"status": "error", "error": str(e)},
            )
    
    def _extract_positions(self, model: List[int]) -> Dict[int, Tuple[int, ...]]:
        """Extract item positions from SAT model."""
        if self.context is None:
            return {}
        
        model_set = {lit for lit in model if lit > 0}
        R = self.context.num_rounds
        n, m = self.context.n_rows, self.context.n_cols
        
        positions: Dict[int, Tuple[int, ...]] = {}
        
        for ion in self.context.ions:
            for krow in range(n):
                for jcol in range(m):
                    var = self.var_a(R, 0, krow, jcol, ion)
                    if var <= self.vpool.top and var in model_set:
                        positions[ion] = (krow, jcol)
                        break
                else:
                    continue
                break
        
        return positions
    
    def diagnose_unsat_core(
        self, timeout: Optional[float] = None
    ) -> List[str]:
        """Run UNSAT core extraction and return conflicting group names."""
        if not self._groups_enabled or not self._group_selectors:
            return []

        assumptions = [-sel for sel in self._group_selectors.values()]
        sel_to_name = {sel: name for name, sel in self._group_selectors.items()}

        sol = self._solve_sat(timeout=timeout, assumptions=assumptions)
        if sol.satisfiable:
            wise_logger.debug("UNSAT core diagnosis: formula is SAT — no core")
            return []

        core_lits = sol.statistics.get("core", [])

        if not core_lits:
            wise_logger.debug("UNSAT core: solver did not return core literals")
            return list(self._group_selectors.keys())

        conflicting: List[str] = []
        core_set = set(abs(l) for l in core_lits)
        for name, sel in self._group_selectors.items():
            if abs(sel) in core_set:
                conflicting.append(name)

        wise_logger.info(
            "UNSAT core groups (%d / %d): %s",
            len(conflicting), len(self._group_selectors), conflicting,
        )
        return conflicting
    
    def decode_schedule(self, model: List[int]) -> List[List[Dict[str, Any]]]:
        """Decode the routing schedule from a SAT model."""
        if self.context is None:
            return []
        
        model_set = {lit for lit in model if lit > 0}
        R = self.context.num_rounds
        n, m = self.context.n_rows, self.context.n_cols
        
        def lit_true(v: int) -> bool:
            return v in model_set
        
        schedule: List[List[Dict[str, Any]]] = [[] for _ in range(R)]
        
        for r in range(R):
            P_bound = self._pass_bounds[r]
            for p in range(P_bound):
                phase_lit = self.var_phase(r, p)
                is_vertical = phase_lit <= self.vpool.top and lit_true(phase_lit)
                phase = "V" if is_vertical else "H"
                
                pass_info: Dict[str, Any] = {
                    "phase": phase,
                    "h_swaps": [],
                    "v_swaps": [],
                }
                
                if phase == "H":
                    for krow in range(n):
                        for jcol in range(m - 1):
                            v = self.var_s_h(r, p, krow, jcol)
                            if v <= self.vpool.top and lit_true(v):
                                pass_info["h_swaps"].append((krow, jcol))
                else:
                    for krow in range(n - 1):
                        for jcol in range(m):
                            v = self.var_s_v(r, p, krow, jcol)
                            if v <= self.vpool.top and lit_true(v):
                                pass_info["v_swaps"].append((krow, jcol))
                
                schedule[r].append(pass_info)
        
        return schedule
    
    def reset(self) -> None:
        """Reset the encoder for a new problem."""
        self.vpool = IDPool()
        self.formula = WCNF() if self.use_maxsat else CNF()
        self.context = None
        self._pass_bounds = []
        self._group_selectors.clear()
        self._group_meta.clear()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "WISESATContext",
    "WISESATEncoder",
    "NoFeasibleLayoutError",
]
