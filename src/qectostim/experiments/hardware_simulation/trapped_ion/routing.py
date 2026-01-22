# src/qectostim/experiments/hardware_simulation/trapped_ion/routing.py
"""
WISE SAT-based routing for trapped ion QCCD architectures.

This module provides routing algorithms optimized for WISE (Wiring-based Ion
Shuttling for Entanglement) architectures:

- WISESATEncoder: SAT encoder implementing GridSATEncoder for WISE grids
- WiseSatRouter: SAT-based optimal routing using odd-even transposition sorts
- WisePatchRouter: Patch-based routing for large grids
- GreedyIonRouter: Fast heuristic routing for small instances

The core Router ABC is in core/compiler.py; this module provides
trapped-ion specific implementations.

SAT Solver Requirements:
    - pysat: For CNF/WCNF formula construction and solving
    - Optional: RC2 for MaxSAT optimization

Ported and refactored from:
    - old/utils/qccd_operations.py (SAT solver core)
    - old/compiler/qccd_WISE_ion_route.py (patch routing)
    - old/compiler/_qccd_WISE_ion_routing.py (routing utilities)
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np

from qectostim.experiments.hardware_simulation.core.compiler import (
    Router,
    RoutingResult,
    RoutingStrategy,
)
from qectostim.experiments.hardware_simulation.core.architecture import (
    LayoutTracker,
)
from qectostim.experiments.hardware_simulation.core.pipeline import QubitMapping
from qectostim.experiments.hardware_simulation.core.sat_interface import (
    GridSATEncoder,
    SATSolution,
    PlacementRequirement,
    ConstraintType,
    SATRoutingConfig,
    PatchRoutingConfig,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.architecture import (
        HardwareArchitecture,
    )
    from .components import Ion, QCCDComponent, Trap
    from .operations import QCCDOperationBase


# =============================================================================
# Constants and Configuration
# =============================================================================

WISE_LOGGER_NAME = "wise.qccd.routing"
wise_logger = logging.getLogger(WISE_LOGGER_NAME)
if not wise_logger.handlers:
    wise_logger.addHandler(logging.NullHandler())
wise_logger.propagate = False


# =============================================================================
# pysat Availability Check
# =============================================================================

_PYSAT_AVAILABLE = False
try:
    from pysat.formula import IDPool, CNF, WCNF
    from pysat.card import CardEnc, EncType
    from pysat.solvers import Minisat22, Solver
    _PYSAT_AVAILABLE = True
except ImportError:
    IDPool = None
    CNF = None
    WCNF = None
    CardEnc = None
    EncType = None
    Minisat22 = None
    Solver = None


@dataclass
class WISERoutingConfig(SATRoutingConfig):
    """Configuration for WISE SAT-based routing.
    
    Extends the base SATRoutingConfig with WISE-specific parameters
    for trapped-ion QCCD architectures.
    
    Attributes
    ----------
    (inherited from SATRoutingConfig)
    timeout_seconds : float
        SAT solver timeout per configuration.
    max_passes : int
        Maximum H-V passes per round.
    use_maxsat : bool
        If True, use MaxSAT for optimization; else pure SAT.
    debug_mode : bool
        Enable verbose logging.
    num_workers : int
        Number of parallel SAT workers.
        
    (WISE-specific)
    patch_enabled : bool
        If True, use patch-based routing for large grids.
    patch_height : int
        Patch height for patch routing.
    patch_width : int
        Patch width for patch routing.
    bt_soft_weight : int
        Weight for BT (block target) soft constraints in MaxSAT.
    boundary_soft_weight_row : int
        Weight for row boundary avoidance soft constraints.
    boundary_soft_weight_col : int
        Weight for column boundary avoidance soft constraints.
    """
    # WISE-specific parameters (base params inherited from SATRoutingConfig)
    patch_enabled: bool = False
    patch_height: int = 4
    patch_width: int = 4
    bt_soft_weight: int = 0
    boundary_soft_weight_row: int = 0
    boundary_soft_weight_col: int = 0
    
    @classmethod
    def from_env(cls) -> "WISERoutingConfig":
        """Create config from environment variables."""
        return cls(
            timeout_seconds=float(os.environ.get("WISE_SAT_TIMEOUT", "60")),
            max_passes=int(os.environ.get("WISE_MAX_PASSES", "10")),
            use_maxsat=os.environ.get("WISE_USE_MAXSAT", "1") != "0",
            patch_enabled=os.environ.get("WISE_PATCH_ENABLED", "0") != "0",
            patch_height=int(os.environ.get("WISE_PATCH_HEIGHT", "4")),
            patch_width=int(os.environ.get("WISE_PATCH_WIDTH", "4")),
            debug_mode=os.environ.get("WISE_DEBUG", "0") != "0",
            num_workers=int(os.environ.get("WISE_SAT_WORKERS", "1")),
            bt_soft_weight=int(os.environ.get("WISE_BT_SOFT_WEIGHT", "0")),
            boundary_soft_weight_row=int(os.environ.get("WISE_BOUNDARY_WEIGHT_ROW", "0")),
            boundary_soft_weight_col=int(os.environ.get("WISE_BOUNDARY_WEIGHT_COL", "0")),
        )


# =============================================================================
# WISESATEncoder - Core SAT Encoder for WISE Grids
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
        self.use_maxsat = use_maxsat or self.config.use_maxsat
        
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
            # Add with selector for UNSAT core debugging
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
    
    def initialize(self, context: WISESATContext, pass_bound: int) -> None:
        """Initialize the encoder with routing context.
        
        Parameters
        ----------
        context : WISESATContext
            Context containing layout and requirement info.
        pass_bound : int
            Maximum number of passes per round.
        """
        self.context = context
        self.rows = context.n_rows
        self.cols = context.n_cols
        self._items = list(context.ions)
        
        # Set pass bounds (uniform for now, can be customized later)
        self._pass_bounds = [pass_bound] * context.num_rounds
    
    # -------------------------------------------------------------------------
    # GridSATEncoder Interface Implementation
    # -------------------------------------------------------------------------
    
    def add_placement_variables(self) -> None:
        """Create placement variables for all positions and items.
        
        Variables are created lazily via vpool, so this is a no-op.
        The variables are created when first accessed via var_a().
        """
        pass  # Variables created lazily by IDPool
    
    def add_permutation_constraints(self) -> None:
        """Add permutation constraints: each ion at exactly one position.
        
        Constraints:
        (0a) Exactly one ion per cell at each (r, p, krow, jcol)
        (0b) Each ion in exactly one cell at each (r, p)
        
        Uses ladder encoding for cardinality constraints.
        """
        if self.context is None:
            raise RuntimeError("Must call initialize() before adding constraints")
        
        ctx = self.context
        R = ctx.num_rounds
        n, m = ctx.n_rows, ctx.n_cols
        ions = ctx.ions
        
        # (0a) Exactly one ion per cell at each (r, p)
        for r in range(R):
            P_bound = self._pass_bounds[r]
            for p in range(P_bound + 1):  # +1 for final state
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
        """Add constraints fixing the initial layout.
        
        At (r=0, p=0), each cell must contain its initial ion.
        """
        if self.context is None:
            raise RuntimeError("Must call initialize() before adding constraints")
        
        ctx = self.context
        A_in = ctx.initial_layout
        n, m = ctx.n_rows, ctx.n_cols
        ions = ctx.ions
        
        for krow in range(n):
            for jcol in range(m):
                ion0 = int(A_in[krow, jcol])
                # Ion ion0 must be at this cell
                self._add_hard([self.var_a(0, 0, krow, jcol, ion0)], "INIT")
                # No other ion may be there
                for ion in ions:
                    if ion != ion0:
                        self._add_hard([-self.var_a(0, 0, krow, jcol, ion)], "INIT")
    
    def add_round_chaining_constraints(self) -> None:
        """Add constraints linking end of round r to start of round r+1.
        
        a[r+1, 0] ↔ a[r, P_bound[r]] for all cells and ions.
        """
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
                        # a_next ↔ a_end (bidirectional implication)
                        self._add_hard([-a_next, a_end], f"CHAIN:r{r}")
                        self._add_hard([-a_end, a_next], f"CHAIN:r{r}")
    
    def add_row_sorting_constraints(self, row: int, phase: int) -> None:
        """Add constraints for horizontal sorting within a row.
        
        Part of odd-even transposition sort network.
        """
        # Implemented in add_swap_constraints() which handles all phases
        pass
    
    def add_column_sorting_constraints(self, col: int, phase: int) -> None:
        """Add constraints for vertical sorting within a column.
        
        Part of odd-even transposition sort network.
        """
        # Implemented in add_swap_constraints() which handles all phases
        pass
    
    def add_phase_constraints(self, phase_type: str, phase_number: int) -> None:
        """Add phase constraints.
        
        Phase monotonicity: once we switch from H to V, we stay in V.
        phase[r,p] → phase[r,p+1]
        """
        if self.context is None:
            raise RuntimeError("Must call initialize() before adding constraints")
        
        ctx = self.context
        R = ctx.num_rounds
        
        for r in range(R):
            P_bound = self._pass_bounds[r]
            if P_bound > 1:
                for p in range(P_bound - 1):
                    # phase[r,p] → phase[r,p+1]
                    self._add_hard(
                        [-self.var_phase(r, p), self.var_phase(r, p + 1)],
                        "PHASE_MONO"
                    )
    
    def add_swap_constraints(self) -> None:
        """Add all swap and copy constraints for the odd-even transposition network.
        
        This includes:
        - Horizontal comparator gating (active only in H phase)
        - Vertical comparator gating (active only in V phase)  
        - Parity constraints (odd-even structure)
        - Swap semantics (ions exchange positions when swap active)
        - Copy constraints (ions stay in place when no swap)
        """
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
                
                # --- Horizontal comparators (row-wise) ---
                for krow in range(n):
                    for jcol in range(m - 1):
                        sh = self.var_s_h(r, p, krow, jcol)
                        
                        # Gating: if phase[p]=1 (vertical), H comparators must be off
                        # phase[p] → ¬s_h  ≡  (¬phase[p] ∨ ¬s_h)
                        self._add_hard([-phase_p, -sh], "H_GATE")
                        
                        # Parity: if jcol % 2 != p % 2, s_h must be 0
                        if jcol % 2 != p % 2:
                            self._add_hard([-sh], "H_GATE")
                        
                        # Swap semantics
                        for ion in ions:
                            a_cur_j = self.var_a(r, p, krow, jcol, ion)
                            a_cur_j1 = self.var_a(r, p, krow, jcol + 1, ion)
                            a_next_j = self.var_a(r, p + 1, krow, jcol, ion)
                            a_next_j1 = self.var_a(r, p + 1, krow, jcol + 1, ion)
                            
                            # (s ∧ a_cur_j1) → a_next_j (ion moves left)
                            self._add_hard([-sh, -a_cur_j1, a_next_j], "H_SEM")
                            
                            # (s ∧ a_cur_j) → a_next_j1 (ion moves right)
                            self._add_hard([-sh, -a_cur_j, a_next_j1], "H_SEM")
                
                # --- Vertical comparators (column-wise) ---
                for krow in range(n - 1):
                    for jcol in range(m):
                        sv = self.var_s_v(r, p, krow, jcol)
                        
                        # Gating: if phase[p]=0 (horizontal), V comparators must be off
                        # ¬phase[p] → ¬s_v  ≡  (phase[p] ∨ ¬s_v)
                        self._add_hard([phase_p, -sv], "V_GATE")
                        
                        # Parity: if krow % 2 != p % 2, s_v must be 0
                        if krow % 2 != p % 2:
                            self._add_hard([-sv], "V_GATE")
                        
                        # Swap semantics
                        for ion in ions:
                            a_cur_top = self.var_a(r, p, krow, jcol, ion)
                            a_cur_bot = self.var_a(r, p, krow + 1, jcol, ion)
                            a_next_top = self.var_a(r, p + 1, krow, jcol, ion)
                            a_next_bot = self.var_a(r, p + 1, krow + 1, jcol, ion)
                            
                            # (s ∧ a_cur_bot) → a_next_top (ion moves up)
                            self._add_hard([-sv, -a_cur_bot, a_next_top], "V_SEM")
                            
                            # (s ∧ a_cur_top) → a_next_bot (ion moves down)
                            self._add_hard([-sv, -a_cur_top, a_next_bot], "V_SEM")
        
        # Add copy constraints
        self._add_copy_constraints()
    
    def _add_copy_constraints(self) -> None:
        """Add copy constraints for non-swapping cells.
        
        If a cell is not involved in a swap, its ion stays in place.
        """
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
                        # H comparators touching this cell
                        s_left = self.var_s_h(r, p, krow, jcol - 1) if jcol > 0 else None
                        s_right = self.var_s_h(r, p, krow, jcol) if jcol < m - 1 else None
                        
                        # Antecedent: (¬phase ∧ ¬s_left ∧ ¬s_right)
                        lits_ante_neg = [phase_p]  # ¬(¬phase) = phase
                        if s_left is not None:
                            lits_ante_neg.append(s_left)
                        if s_right is not None:
                            lits_ante_neg.append(s_right)
                        
                        for ion in ions:
                            a_cur = self.var_a(r, p, krow, jcol, ion)
                            a_next = self.var_a(r, p + 1, krow, jcol, ion)
                            
                            # (¬phase ∧ ¬s_left ∧ ¬s_right ∧ a_cur) → a_next
                            # CNF: (phase ∨ s_left ∨ s_right ∨ ¬a_cur ∨ a_next)
                            self._add_hard(lits_ante_neg + [-a_cur, a_next], "H_COPY")
                            
                            # (¬phase ∧ ¬s_left ∧ ¬s_right ∧ a_next) → a_cur  
                            # CNF: (phase ∨ s_left ∨ s_right ∨ ¬a_next ∨ a_cur)
                            self._add_hard(lits_ante_neg + [-a_next, a_cur], "H_COPY")
        
        # Vertical phase copy constraints
        for r in range(R):
            P_bound = self._pass_bounds[r]
            for p in range(P_bound):
                phase_p = self.var_phase(r, p)
                
                for krow in range(n):
                    for jcol in range(m):
                        # V comparators touching this cell
                        s_up = self.var_s_v(r, p, krow - 1, jcol) if krow > 0 else None
                        s_down = self.var_s_v(r, p, krow, jcol) if krow < n - 1 else None
                        
                        # Antecedent: (phase ∧ ¬s_up ∧ ¬s_down)
                        lits_ante_neg = [-phase_p]  # ¬phase
                        if s_up is not None:
                            lits_ante_neg.append(s_up)
                        if s_down is not None:
                            lits_ante_neg.append(s_down)
                        
                        for ion in ions:
                            a_cur = self.var_a(r, p, krow, jcol, ion)
                            a_next = self.var_a(r, p + 1, krow, jcol, ion)
                            
                            # (phase ∧ ¬s_up ∧ ¬s_down ∧ a_cur) → a_next
                            self._add_hard(lits_ante_neg + [-a_cur, a_next], "V_COPY")
                            
                            # (phase ∧ ¬s_up ∧ ¬s_down ∧ a_next) → a_cur
                            self._add_hard(lits_ante_neg + [-a_next, a_cur], "V_COPY")
    
    def add_target_position_constraints(self, use_soft: bool = False) -> None:
        """Add constraints for target (BT) positions.
        
        At the end of each round, certain ions must be at their target positions.
        
        Parameters
        ----------
        use_soft : bool
            If True and use_maxsat, add as soft constraints.
        """
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
                    # Hard: ion must be at target
                    self._add_hard([pin_lit], "BT")
                    # Hard: ion cannot be elsewhere
                    for krow in range(n):
                        for jcol in range(m):
                            if (krow, jcol) != (d_fix, c_fix):
                                self._add_hard(
                                    [-self.var_a(r, P_final, krow, jcol, ion)],
                                    "BT"
                                )
    
    def add_gate_pair_constraints(self) -> None:
        """Add constraints requiring gate pairs to be adjacent at round end.
        
        For each pair (i1, i2) in round r:
        - Both must be in the same row at round end
        - Both must be in the same gating block at round end
        """
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
    
    def add_row_block_linkage_constraints(self) -> None:
        """Add constraints linking row_end/block_end variables to positions.
        
        row_end[r,ion,d] ↔ (ion in row d at end of round r)
        block_end[r,ion,b] ↔ (ion in block b at end of round r)
        """
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
                    # re → OR(cell_lits)
                    self._add_hard([-re] + cell_lits, f"ROWBLOCK_LINK:r{r}")
                    # cell_lit → re
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
                        # we → OR(cells)
                        self._add_hard([-we] + cells, f"ROWBLOCK_LINK:r{r}")
                        # cell → we
                        for aj in cells:
                            self._add_hard([-aj, we], f"ROWBLOCK_LINK:r{r}")
                    else:
                        # Block not valid for gating
                        self._add_hard([-we], f"ROWBLOCK_LINK:r{r}")
    
    def add_adjacency_requirements(
        self,
        requirements: List[PlacementRequirement],
    ) -> None:
        """Add adjacency requirements (interface compliance).
        
        Converts PlacementRequirement to gate pair constraints.
        """
        # Convert to gate pairs format for the current round
        # This is a simplified version; full implementation would handle
        # multi-round requirements
        pass
    
    def add_all_constraints(
        self,
        skip_cardinality: bool = False,
        skip_pairs: bool = False,
    ) -> None:
        """Add all standard WISE constraints.
        
        Convenience method to add all constraints in order.
        
        Parameters
        ----------
        skip_cardinality : bool
            Skip permutation constraints (for debugging).
        skip_pairs : bool
            Skip gate pair constraints (for debugging).
        """
        if not skip_cardinality:
            self.add_permutation_constraints()
        
        self.add_initial_layout_constraints()
        self.add_round_chaining_constraints()
        self.add_phase_constraints("mono", 0)  # Phase monotonicity
        self.add_swap_constraints()  # Includes copy constraints
        self.add_row_block_linkage_constraints()
        self.add_target_position_constraints()
        
        if not skip_pairs:
            self.add_gate_pair_constraints()
    
    # -------------------------------------------------------------------------
    # Solving
    # -------------------------------------------------------------------------
    
    def solve(
        self,
        timeout: Optional[float] = None,
        assumptions: Optional[List[int]] = None,
    ) -> SATSolution:
        """Solve the SAT problem.
        
        Parameters
        ----------
        timeout : Optional[float]
            Maximum solve time in seconds.
        assumptions : Optional[List[int]]
            Literal assumptions for incremental solving.
            
        Returns
        -------
        SATSolution
            Solution with item positions and statistics.
        """
        if self.use_maxsat:
            return self._solve_maxsat(timeout)
        else:
            return self._solve_sat(timeout, assumptions)
    
    def _solve_sat(
        self,
        timeout: Optional[float] = None,
        assumptions: Optional[List[int]] = None,
    ) -> SATSolution:
        """Solve using plain SAT (Minisat22)."""
        t0 = time.time()
        
        if timeout is not None and timeout <= 0:
            return SATSolution(
                satisfiable=False,
                statistics={"status": "timeout_zero"},
            )
        
        try:
            with Minisat22(bootstrap_with=self.formula.clauses) as solver:
                if assumptions:
                    sat_ok = solver.solve(assumptions=assumptions)
                else:
                    sat_ok = solver.solve()
                
                model = solver.get_model() if sat_ok else None
                
                t1 = time.time()
                
                if sat_ok and model:
                    item_positions = self._extract_positions(model)
                    return SATSolution(
                        satisfiable=True,
                        item_positions=item_positions,
                        cost=0.0,
                        solve_time=t1 - t0,
                        statistics={
                            "status": "ok",
                            "model_size": len(model),
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
    
    def _solve_maxsat(self, timeout: Optional[float] = None) -> SATSolution:
        """Solve using MaxSAT (RC2)."""
        try:
            from pysat.examples.rc2 import RC2
        except ImportError:
            wise_logger.warning("RC2 not available, falling back to SAT")
            return self._solve_sat(timeout)
        
        t0 = time.time()
        
        try:
            with RC2(self.formula) as rc2:
                model = rc2.compute()
                cost = rc2.cost if hasattr(rc2, 'cost') else 0
                
                t1 = time.time()
                
                if model:
                    item_positions = self._extract_positions(model)
                    return SATSolution(
                        satisfiable=True,
                        item_positions=item_positions,
                        cost=float(cost),
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
        """Extract item positions from SAT model.
        
        Returns positions at the final state (R, 0).
        """
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
    
    def decode_schedule(self, model: List[int]) -> List[List[Dict[str, Any]]]:
        """Decode the routing schedule from a SAT model.
        
        Returns
        -------
        List[List[Dict[str, Any]]]
            Per-round list of pass info dicts with keys:
            - "phase": "H" or "V"
            - "h_swaps": list of (row, col) for H swaps
            - "v_swaps": list of (row, col) for V swaps
        """
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
# Routing Data Structures
# =============================================================================

@dataclass
class GridLayout:
    """Ion layout on a 2D grid.
    
    Attributes
    ----------
    grid : np.ndarray
        2D array of ion indices (n_rows x n_cols).
    ion_positions : Dict[int, Tuple[int, int]]
        Mapping from ion index to (row, col) position.
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    """
    grid: np.ndarray
    ion_positions: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.n_rows, self.n_cols = self.grid.shape
        self._rebuild_positions()
    
    def _rebuild_positions(self) -> None:
        """Rebuild ion_positions from grid."""
        self.ion_positions.clear()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                ion_idx = int(self.grid[r, c])
                self.ion_positions[ion_idx] = (r, c)
    
    def get_ion_at(self, row: int, col: int) -> int:
        """Get ion index at position."""
        return int(self.grid[row, col])
    
    def get_position(self, ion_idx: int) -> Optional[Tuple[int, int]]:
        """Get position of ion."""
        return self.ion_positions.get(ion_idx)
    
    def swap_horizontal(self, row: int, col: int) -> None:
        """Swap ions at (row, col) and (row, col+1)."""
        if col + 1 >= self.n_cols:
            return
        ion_a = self.grid[row, col]
        ion_b = self.grid[row, col + 1]
        self.grid[row, col] = ion_b
        self.grid[row, col + 1] = ion_a
        self.ion_positions[int(ion_a)] = (row, col + 1)
        self.ion_positions[int(ion_b)] = (row, col)
    
    def swap_vertical(self, row: int, col: int) -> None:
        """Swap ions at (row, col) and (row+1, col)."""
        if row + 1 >= self.n_rows:
            return
        ion_a = self.grid[row, col]
        ion_b = self.grid[row + 1, col]
        self.grid[row, col] = ion_b
        self.grid[row + 1, col] = ion_a
        self.ion_positions[int(ion_a)] = (row + 1, col)
        self.ion_positions[int(ion_b)] = (row, col)
    
    def copy(self) -> "GridLayout":
        """Create a deep copy."""
        return GridLayout(
            grid=self.grid.copy(),
            ion_positions=dict(self.ion_positions),
        )


@dataclass
class RoutingPass:
    """A single routing pass (H or V phase).
    
    Attributes
    ----------
    phase : str
        "H" for horizontal, "V" for vertical.
    h_swaps : List[Tuple[int, int]]
        Horizontal swap positions (row, col).
    v_swaps : List[Tuple[int, int]]
        Vertical swap positions (row, col).
    """
    phase: str = "H"
    h_swaps: List[Tuple[int, int]] = field(default_factory=list)
    v_swaps: List[Tuple[int, int]] = field(default_factory=list)
    
    @property
    def has_swaps(self) -> bool:
        return bool(self.h_swaps if self.phase == "H" else self.v_swaps)
    
    @property
    def swap_count(self) -> int:
        return len(self.h_swaps) + len(self.v_swaps)


@dataclass
class RoutingSchedule:
    """Schedule of routing passes for multiple rounds.
    
    Attributes
    ----------
    passes_per_round : List[List[RoutingPass]]
        For each round, list of routing passes.
    layouts : List[GridLayout]
        Layout after each round.
    total_passes : int
        Total number of passes across all rounds.
    total_swaps : int
        Total number of swaps across all rounds.
    """
    passes_per_round: List[List[RoutingPass]] = field(default_factory=list)
    layouts: List[GridLayout] = field(default_factory=list)
    
    @property
    def total_passes(self) -> int:
        return sum(len(passes) for passes in self.passes_per_round)
    
    @property
    def total_swaps(self) -> int:
        return sum(
            p.swap_count
            for passes in self.passes_per_round
            for p in passes
        )


# =============================================================================
# Gate Pair Analysis
# =============================================================================

@dataclass
class GatePairRequirement:
    """Requirement for a two-qubit gate.
    
    Attributes
    ----------
    ion_a : int
        First ion index.
    ion_b : int
        Second ion index.
    round_idx : int
        Which round this gate is in.
    gate_type : str
        Type of gate (e.g., "MS", "CZ").
    """
    ion_a: int
    ion_b: int
    round_idx: int = 0
    gate_type: str = "MS"
    
    @property
    def pair(self) -> Tuple[int, int]:
        return (min(self.ion_a, self.ion_b), max(self.ion_a, self.ion_b))


def compute_target_positions(
    pairs: List[Tuple[int, int]],
    n_rows: int,
    n_cols: int,
    capacity: int = 2,
) -> Dict[int, Tuple[int, int]]:
    """Compute target positions for ions to satisfy gate pairs.
    
    Uses bipartite matching to assign ions to gating positions.
    
    Parameters
    ----------
    pairs : List[Tuple[int, int]]
        Ion pairs that need to interact.
    n_rows : int
        Grid rows.
    n_cols : int
        Grid columns.
    capacity : int
        Block width for gating zones.
        
    Returns
    -------
    Dict[int, Tuple[int, int]]
        Target position for each ion involved in gates.
    """
    # Simple heuristic: place pairs adjacent horizontally
    # More sophisticated: use Hungarian algorithm
    targets: Dict[int, Tuple[int, int]] = {}
    
    # Collect all ions involved
    all_ions: Set[int] = set()
    for a, b in pairs:
        all_ions.add(a)
        all_ions.add(b)
    
    # Available gating positions (pairs of adjacent cells)
    gating_positions: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for r in range(n_rows):
        for c in range(0, n_cols - 1, capacity):
            gating_positions.append(((r, c), (r, c + 1)))
    
    # Simple greedy assignment
    used_positions: Set[Tuple[int, int]] = set()
    pair_idx = 0
    
    for ion_a, ion_b in pairs:
        if ion_a in targets and ion_b in targets:
            continue
        
        # Find an unused gating position
        for pos_a, pos_b in gating_positions:
            if pos_a not in used_positions and pos_b not in used_positions:
                if ion_a not in targets:
                    targets[ion_a] = pos_a
                    used_positions.add(pos_a)
                if ion_b not in targets:
                    targets[ion_b] = pos_b
                    used_positions.add(pos_b)
                break
    
    return targets


# =============================================================================
# WISE SAT Router
# =============================================================================

class WiseSatRouter(Router):
    """SAT-based optimal routing for WISE grid architectures.
    
    Uses SAT/MaxSAT solving to find optimal ion permutations that:
    1. Minimize total routing passes
    2. Bring gate pairs to adjacent positions
    3. Respect capacity constraints
    
    The solver encodes:
    - Ion positions as Boolean variables
    - Odd-even transposition sort structure (H-V phases)
    - Gate pair adjacency requirements
    - Optional soft constraints for optimization
    
    Example
    -------
    >>> router = WiseSatRouter(config=WISERoutingConfig(timeout_seconds=30))
    >>> result = router.route_batch(pairs, mapping, architecture)
    >>> if result.success:
    ...     for op in result.operations:
    ...         print(op)
    
    Notes
    -----
    This is a refactored port of the SAT solver from old/utils/qccd_operations.py.
    The core SAT encoding is in _build_sat_formula().
    
    See Also
    --------
    WisePatchRouter : For large grids, uses patch decomposition.
    """
    
    def __init__(
        self,
        config: Optional[WISERoutingConfig] = None,
        name: str = "wise_sat_router",
    ):
        super().__init__(RoutingStrategy.GLOBAL_OPTIMIZATION, name)
        self.config = config or WISERoutingConfig()
        self._sat_available = self._check_sat_available()
    
    def _check_sat_available(self) -> bool:
        """Check if pysat is available."""
        try:
            from pysat.formula import CNF
            from pysat.solvers import Minisat22
            return True
        except ImportError:
            wise_logger.warning(
                "pysat not available. WiseSatRouter will use fallback heuristics."
            )
            return False
    
    def route_gate(
        self,
        gate_qubits: Tuple[int, ...],
        current_mapping: QubitMapping,
        architecture: "HardwareArchitecture",
    ) -> RoutingResult:
        """Route for a single gate (wraps route_batch)."""
        if len(gate_qubits) != 2:
            return RoutingResult(
                success=False,
                metrics={"error": "WiseSatRouter only handles two-qubit gates"},
            )
        
        return self.route_batch(
            [gate_qubits],
            current_mapping,
            architecture,
        )
    
    def route_batch(
        self,
        gate_pairs: List[Tuple[int, int]],
        current_mapping: QubitMapping,
        architecture: "HardwareArchitecture",
    ) -> RoutingResult:
        """Route a batch of two-qubit gates using SAT optimization.
        
        Parameters
        ----------
        gate_pairs : List[Tuple[int, int]]
            Pairs of logical qubits needing interaction.
        current_mapping : QubitMapping
            Current qubit positions.
        architecture : HardwareArchitecture
            Target WISE grid architecture.
            
        Returns
        -------
        RoutingResult
            Routing operations and updated mapping.
        """
        if not gate_pairs:
            return RoutingResult(success=True, operations=[], cost=0.0)
        
        # Convert to physical qubits
        physical_pairs = []
        for q1, q2 in gate_pairs:
            p1 = current_mapping.get_physical(q1)
            p2 = current_mapping.get_physical(q2)
            if p1 is None or p2 is None:
                return RoutingResult(
                    success=False,
                    metrics={"error": f"Unmapped qubits: {q1}, {q2}"},
                )
            physical_pairs.append((p1, p2))
        
        # Get grid dimensions from architecture
        n_rows, n_cols = self._get_grid_dimensions(architecture)
        capacity = self._get_capacity(architecture)
        
        # Build initial layout from current mapping
        initial_layout = self._build_initial_layout(
            current_mapping, n_rows, n_cols
        )
        
        if not self._sat_available:
            # Fallback to heuristic routing
            return self._route_heuristic(
                physical_pairs, initial_layout, n_rows, n_cols, capacity
            )
        
        # Use SAT solver
        try:
            schedule = self._solve_sat(
                initial_layout,
                physical_pairs,
                n_rows,
                n_cols,
                capacity,
            )
            
            if schedule is None:
                return RoutingResult(
                    success=False,
                    metrics={"error": "SAT solver found no solution"},
                )
            
            # Convert schedule to operations
            operations = self._schedule_to_operations(schedule)
            
            # Update mapping
            final_layout = schedule.layouts[-1] if schedule.layouts else initial_layout
            final_mapping = self._layout_to_mapping(final_layout, current_mapping)
            
            return RoutingResult(
                success=True,
                operations=operations,
                cost=float(schedule.total_swaps),
                final_mapping=final_mapping,
                metrics={
                    "total_passes": schedule.total_passes,
                    "total_swaps": schedule.total_swaps,
                },
            )
            
        except Exception as e:
            wise_logger.error(f"SAT solver error: {e}")
            return RoutingResult(
                success=False,
                metrics={"error": str(e)},
            )
    
    def supports_batch_routing(self) -> bool:
        return True
    
    def _get_grid_dimensions(
        self, architecture: "HardwareArchitecture"
    ) -> Tuple[int, int]:
        """Extract grid dimensions from architecture."""
        # Try to get from architecture metadata
        if hasattr(architecture, "n_rows") and hasattr(architecture, "n_cols"):
            return architecture.n_rows, architecture.n_cols
        if hasattr(architecture, "grid_shape"):
            return architecture.grid_shape
        # Default fallback
        n_qubits = architecture.num_qubits
        side = int(np.ceil(np.sqrt(n_qubits)))
        return side, side
    
    def _get_capacity(self, architecture: "HardwareArchitecture") -> int:
        """Extract block capacity from architecture."""
        if hasattr(architecture, "capacity"):
            return architecture.capacity
        return 2  # Default WISE capacity
    
    def _build_initial_layout(
        self,
        mapping: QubitMapping,
        n_rows: int,
        n_cols: int,
    ) -> GridLayout:
        """Build GridLayout from QubitMapping."""
        grid = np.zeros((n_rows, n_cols), dtype=int)
        
        # Place mapped qubits
        for logical, physical in mapping.logical_to_physical.items():
            zone = mapping.zone_assignments.get(physical)
            if zone:
                # Parse zone to get row, col
                # Assuming zone format like "trap_r_c" or similar
                try:
                    parts = zone.split("_")
                    row, col = int(parts[-2]), int(parts[-1])
                    if 0 <= row < n_rows and 0 <= col < n_cols:
                        grid[row, col] = physical
                except (ValueError, IndexError):
                    pass
        
        # Fill remaining cells with unique indices
        used = set(grid.flatten())
        next_idx = max(used) + 1 if used else 0
        for r in range(n_rows):
            for c in range(n_cols):
                if grid[r, c] == 0 and 0 not in mapping.logical_to_physical.values():
                    grid[r, c] = next_idx
                    next_idx += 1
        
        return GridLayout(grid=grid)
    
    def _layout_to_mapping(
        self,
        layout: GridLayout,
        original_mapping: QubitMapping,
    ) -> QubitMapping:
        """Convert GridLayout back to QubitMapping."""
        new_mapping = original_mapping.copy()
        
        for ion_idx, (row, col) in layout.ion_positions.items():
            # Update zone assignment
            zone_id = f"trap_{row}_{col}"
            if ion_idx in new_mapping.physical_to_logical:
                new_mapping.zone_assignments[ion_idx] = zone_id
        
        return new_mapping
    
    def _solve_sat(
        self,
        initial_layout: GridLayout,
        pairs: List[Tuple[int, int]],
        n_rows: int,
        n_cols: int,
        capacity: int,
    ) -> Optional[RoutingSchedule]:
        """Solve routing using SAT/MaxSAT with WISESATEncoder.
        
        This creates a SAT encoding of the routing problem and solves it
        to find an optimal (or near-optimal) ion permutation schedule.
        
        Parameters
        ----------
        initial_layout : GridLayout
            Starting ion arrangement.
        pairs : List[Tuple[int, int]]
            Ion pairs that need to be adjacent.
        n_rows : int
            Grid rows.
        n_cols : int
            Grid columns.
        capacity : int
            Gating block width.
            
        Returns
        -------
        Optional[RoutingSchedule]
            Routing schedule if SAT, None if UNSAT or error.
        """
        if not _PYSAT_AVAILABLE:
            wise_logger.warning("pysat not available, cannot use SAT solver")
            return None
        
        try:
            # Build context
            ions = list(range(n_rows * n_cols))
            
            # Compute gating blocks
            block_cells, block_fully_inside, block_widths = self._compute_blocks(
                n_rows, n_cols, capacity
            )
            num_blocks = len(block_cells)
            
            # Create context (single round for now)
            context = WISESATContext(
                initial_layout=initial_layout.grid,
                target_positions=[{}],  # No fixed targets for this mode
                gate_pairs=[pairs],
                full_gate_pairs=[pairs],
                ions=ions,
                n_rows=n_rows,
                n_cols=n_cols,
                num_rounds=1,
                block_cells=block_cells,
                block_fully_inside=block_fully_inside,
                block_widths=block_widths,
                num_blocks=num_blocks,
                debug_diag=self.config.debug_mode,
            )
            
            # Create encoder
            encoder = WISESATEncoder(
                rows=n_rows,
                cols=n_cols,
                config=self.config,
                use_maxsat=self.config.use_maxsat,
            )
            
            # Initialize and add constraints
            pass_bound = self.config.max_passes
            encoder.initialize(context, pass_bound)
            encoder.add_all_constraints()
            
            # Solve
            solution = encoder.solve(timeout=self.config.timeout_seconds)
            
            if not solution.satisfiable:
                wise_logger.info(
                    "SAT solver returned UNSAT (status: %s)",
                    solution.statistics.get("status", "unknown")
                )
                return None
            
            # Get model and decode schedule
            # We need to re-solve to get the model for decoding
            # (The SATSolution currently only stores positions)
            if hasattr(encoder, 'formula') and encoder.formula:
                with Minisat22(bootstrap_with=encoder.formula.clauses) as solver:
                    if solver.solve():
                        model = solver.get_model()
                        schedule_data = encoder.decode_schedule(model)
                        
                        # Convert to RoutingSchedule
                        all_passes: List[RoutingPass] = []
                        for round_passes in schedule_data:
                            for pass_info in round_passes:
                                rp = RoutingPass(
                                    phase=pass_info["phase"],
                                    h_swaps=pass_info.get("h_swaps", []),
                                    v_swaps=pass_info.get("v_swaps", []),
                                )
                                if rp.has_swaps:
                                    all_passes.append(rp)
                        
                        # Compute final layout by applying swaps
                        final_layout = initial_layout.copy()
                        for rp in all_passes:
                            if rp.phase == "H":
                                for r, c in rp.h_swaps:
                                    final_layout.swap_horizontal(r, c)
                            else:
                                for r, c in rp.v_swaps:
                                    final_layout.swap_vertical(r, c)
                        
                        return RoutingSchedule(
                            passes_per_round=[all_passes],
                            layouts=[final_layout],
                        )
            
            wise_logger.warning("SAT solution found but could not decode schedule")
            return None
            
        except Exception as e:
            wise_logger.error(f"SAT solver error: {e}")
            if self.config.debug_mode:
                import traceback
                wise_logger.debug(traceback.format_exc())
            return None
    
    def _compute_blocks(
        self,
        n_rows: int,
        n_cols: int,
        capacity: int,
    ) -> Tuple[List[List[Tuple[int, int]]], List[bool], List[int]]:
        """Compute gating block information.
        
        Returns
        -------
        block_cells : List[List[Tuple[int, int]]]
            Cells in each block.
        block_fully_inside : List[bool]
            Whether each block is fully inside grid.
        block_widths : List[int]
            Width of each block.
        """
        block_cells: List[List[Tuple[int, int]]] = []
        block_fully_inside: List[bool] = []
        block_widths: List[int] = []
        
        for c in range(0, n_cols, capacity):
            cells = []
            width = min(capacity, n_cols - c)
            for r in range(n_rows):
                for dc in range(width):
                    cells.append((r, c + dc))
            
            block_cells.append(cells)
            block_fully_inside.append(width == capacity)
            block_widths.append(width)
        
        return block_cells, block_fully_inside, block_widths
    
    def _route_heuristic(
        self,
        pairs: List[Tuple[int, int]],
        initial_layout: GridLayout,
        n_rows: int,
        n_cols: int,
        capacity: int,
    ) -> RoutingResult:
        """Fallback heuristic routing when SAT is unavailable.
        
        Uses odd-even transposition sort phases.
        """
        layout = initial_layout.copy()
        all_passes: List[RoutingPass] = []
        
        # Compute target positions
        targets = compute_target_positions(pairs, n_rows, n_cols, capacity)
        
        # Simple odd-even transposition sort
        max_iterations = 2 * (n_rows + n_cols)
        
        for iteration in range(max_iterations):
            # H-phase (horizontal swaps)
            h_pass = RoutingPass(phase="H")
            parity = iteration % 2
            
            for r in range(n_rows):
                for c in range(parity, n_cols - 1, 2):
                    # Check if swap improves positions
                    ion_a = layout.get_ion_at(r, c)
                    ion_b = layout.get_ion_at(r, c + 1)
                    
                    if self._should_swap_h(ion_a, ion_b, r, c, targets):
                        layout.swap_horizontal(r, c)
                        h_pass.h_swaps.append((r, c))
            
            if h_pass.has_swaps:
                all_passes.append(h_pass)
            
            # V-phase (vertical swaps)
            v_pass = RoutingPass(phase="V")
            
            for c in range(n_cols):
                for r in range(parity, n_rows - 1, 2):
                    ion_a = layout.get_ion_at(r, c)
                    ion_b = layout.get_ion_at(r + 1, c)
                    
                    if self._should_swap_v(ion_a, ion_b, r, c, targets):
                        layout.swap_vertical(r, c)
                        v_pass.v_swaps.append((r, c))
            
            if v_pass.has_swaps:
                all_passes.append(v_pass)
            
            # Check if all pairs are satisfied
            if self._check_pairs_satisfied(layout, pairs, capacity):
                break
        
        schedule = RoutingSchedule(
            passes_per_round=[all_passes],
            layouts=[layout],
        )
        
        operations = self._schedule_to_operations(schedule)
        
        return RoutingResult(
            success=True,
            operations=operations,
            cost=float(schedule.total_swaps),
            metrics={
                "total_passes": schedule.total_passes,
                "total_swaps": schedule.total_swaps,
                "method": "heuristic",
            },
        )
    
    def _should_swap_h(
        self,
        ion_a: int,
        ion_b: int,
        row: int,
        col: int,
        targets: Dict[int, Tuple[int, int]],
    ) -> bool:
        """Check if horizontal swap improves target alignment."""
        target_a = targets.get(ion_a)
        target_b = targets.get(ion_b)
        
        if target_a is None and target_b is None:
            return False
        
        # Current distances
        dist_a_curr = abs(target_a[1] - col) if target_a else 0
        dist_b_curr = abs(target_b[1] - (col + 1)) if target_b else 0
        
        # Distances after swap
        dist_a_swap = abs(target_a[1] - (col + 1)) if target_a else 0
        dist_b_swap = abs(target_b[1] - col) if target_b else 0
        
        return (dist_a_swap + dist_b_swap) < (dist_a_curr + dist_b_curr)
    
    def _should_swap_v(
        self,
        ion_a: int,
        ion_b: int,
        row: int,
        col: int,
        targets: Dict[int, Tuple[int, int]],
    ) -> bool:
        """Check if vertical swap improves target alignment."""
        target_a = targets.get(ion_a)
        target_b = targets.get(ion_b)
        
        if target_a is None and target_b is None:
            return False
        
        # Current distances
        dist_a_curr = abs(target_a[0] - row) if target_a else 0
        dist_b_curr = abs(target_b[0] - (row + 1)) if target_b else 0
        
        # Distances after swap
        dist_a_swap = abs(target_a[0] - (row + 1)) if target_a else 0
        dist_b_swap = abs(target_b[0] - row) if target_b else 0
        
        return (dist_a_swap + dist_b_swap) < (dist_a_curr + dist_b_curr)
    
    def _check_pairs_satisfied(
        self,
        layout: GridLayout,
        pairs: List[Tuple[int, int]],
        capacity: int,
    ) -> bool:
        """Check if all gate pairs are at adjacent gating positions."""
        for ion_a, ion_b in pairs:
            pos_a = layout.get_position(ion_a)
            pos_b = layout.get_position(ion_b)
            
            if pos_a is None or pos_b is None:
                return False
            
            # Check if horizontally adjacent in same gating zone
            if pos_a[0] == pos_b[0]:  # Same row
                col_diff = abs(pos_a[1] - pos_b[1])
                if col_diff == 1:
                    # Check they're in a gating zone (even column starts)
                    min_col = min(pos_a[1], pos_b[1])
                    if min_col % capacity == 0:
                        continue
            
            return False
        
        return True
    
    def _schedule_to_operations(
        self, schedule: RoutingSchedule
    ) -> List["QCCDOperationBase"]:
        """Convert RoutingSchedule to QCCD operations.
        
        TODO: Create actual Split/Merge/Move operations.
        For now, returns swap coordinates as placeholder.
        """
        operations = []
        
        for round_passes in schedule.passes_per_round:
            for pass_info in round_passes:
                if pass_info.phase == "H":
                    for r, c in pass_info.h_swaps:
                        # Placeholder: record swap info
                        operations.append({
                            "type": "H_SWAP",
                            "row": r,
                            "col": c,
                        })
                else:
                    for r, c in pass_info.v_swaps:
                        operations.append({
                            "type": "V_SWAP",
                            "row": r,
                            "col": c,
                        })
        
        return operations


# =============================================================================
# Patch Router (for large grids)
# =============================================================================

class WisePatchRouter(WiseSatRouter):
    """Patch-based WISE routing for large grids.
    
    Decomposes the grid into overlapping patches and solves each
    patch independently, then merges the solutions. This scales
    better than full SAT for large grids.
    
    Algorithm:
    1. Divide grid into patches (with overlap for boundary handling)
    2. Assign gate pairs to patches based on qubit locations
    3. For each patch:
       a. Extract sub-layout and sub-pairs
       b. Solve using WiseSatRouter
       c. Collect routing operations
    4. Merge solutions and handle boundary interactions
    5. Iteratively refine until all pairs satisfied
    
    Uses checkerboard decomposition to avoid boundary conflicts:
    - Phase 1: Route all "white" patches (no shared boundaries)
    - Phase 2: Route all "black" patches
    
    Attributes
    ----------
    overlap : int
        Number of cells to overlap between adjacent patches.
    max_iterations : int
        Maximum number of refinement iterations.
    """
    
    def __init__(
        self,
        config: Optional[WISERoutingConfig] = None,
        name: str = "wise_patch_router",
        overlap: int = 1,
        max_iterations: int = 5,
    ):
        super().__init__(config, name)
        if self.config:
            self.config.patch_enabled = True
        self.overlap = overlap
        self.max_iterations = max_iterations
    
    def route_batch(
        self,
        gate_pairs: List[Tuple[int, int]],
        current_mapping: QubitMapping,
        architecture: "HardwareArchitecture",
    ) -> RoutingResult:
        """Route using patch decomposition.
        
        Parameters
        ----------
        gate_pairs : List[Tuple[int, int]]
            Pairs of physical qubits that need to interact.
        current_mapping : QubitMapping
            Current qubit positions.
        architecture : HardwareArchitecture
            Target architecture with grid dimensions.
            
        Returns
        -------
        RoutingResult
            Combined routing result from all patches.
        """
        if not gate_pairs:
            return RoutingResult(success=True, operations=[], cost=0.0)
        
        n_rows, n_cols = self._get_grid_dimensions(architecture)
        patch_h = self.config.patch_height if self.config else 4
        patch_w = self.config.patch_width if self.config else 4
        
        # Check if patch routing is beneficial
        if n_rows <= patch_h and n_cols <= patch_w:
            # Grid is small enough for direct solving
            return super().route_batch(gate_pairs, current_mapping, architecture)
        
        wise_logger.info(
            f"Patch routing: {n_rows}x{n_cols} grid with {len(gate_pairs)} pairs"
        )
        
        # Build initial layout from mapping
        initial_layout = self._build_initial_layout(current_mapping, n_rows, n_cols)
        
        # Generate patches with overlap
        patches = self._generate_overlapping_patches(
            n_rows, n_cols, patch_h, patch_w, self.overlap
        )
        wise_logger.debug(f"Generated {len(patches)} patches")
        
        # Assign pairs to patches
        pair_assignments = self._assign_pairs_to_patches(
            gate_pairs, initial_layout, patches
        )
        
        # Route using checkerboard pattern
        all_operations: List[Any] = []
        total_cost = 0.0
        layout = initial_layout.copy()
        
        for iteration in range(self.max_iterations):
            # Phase 1: "White" patches (even row + even col)
            white_patches = [
                (idx, p) for idx, p in enumerate(patches)
                if (p[0] // patch_h + p[1] // patch_w) % 2 == 0
            ]
            
            for patch_idx, patch in white_patches:
                patch_pairs = pair_assignments.get(patch_idx, [])
                if not patch_pairs:
                    continue
                
                result = self._route_patch(
                    patch, patch_pairs, layout, architecture
                )
                if result.success:
                    all_operations.extend(result.operations)
                    total_cost += result.cost
                    # Update layout with patch result
                    self._apply_patch_result(layout, result, patch)
            
            # Phase 2: "Black" patches (odd row + odd col)
            black_patches = [
                (idx, p) for idx, p in enumerate(patches)
                if (p[0] // patch_h + p[1] // patch_w) % 2 == 1
            ]
            
            for patch_idx, patch in black_patches:
                patch_pairs = pair_assignments.get(patch_idx, [])
                if not patch_pairs:
                    continue
                
                result = self._route_patch(
                    patch, patch_pairs, layout, architecture
                )
                if result.success:
                    all_operations.extend(result.operations)
                    total_cost += result.cost
                    self._apply_patch_result(layout, result, patch)
            
            # Check if all pairs are now satisfied
            if self._check_all_pairs_satisfied(gate_pairs, layout):
                wise_logger.info(
                    f"Patch routing converged in {iteration + 1} iterations"
                )
                break
        
        # Build final mapping
        final_mapping = self._layout_to_mapping(layout, current_mapping)
        
        return RoutingResult(
            success=True,
            operations=all_operations,
            cost=total_cost,
            final_mapping=final_mapping,
            metrics={
                "num_patches": len(patches),
                "total_operations": len(all_operations),
                "iterations": iteration + 1,
            },
        )
    
    def _generate_overlapping_patches(
        self,
        n_rows: int,
        n_cols: int,
        patch_h: int,
        patch_w: int,
        overlap: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Generate patches with overlap for boundary handling.
        
        Returns
        -------
        List[Tuple[int, int, int, int]]
            List of (r0, c0, r1, c1) patch boundaries.
        """
        patches = []
        stride_h = patch_h - overlap
        stride_w = patch_w - overlap
        
        r0 = 0
        while r0 < n_rows:
            r1 = min(r0 + patch_h, n_rows)
            c0 = 0
            while c0 < n_cols:
                c1 = min(c0 + patch_w, n_cols)
                patches.append((r0, c0, r1, c1))
                c0 += stride_w
                if c0 >= n_cols and c1 < n_cols:
                    c0 = n_cols - patch_w
            r0 += stride_h
            if r0 >= n_rows and r1 < n_rows:
                r0 = n_rows - patch_h
        
        return patches
    
    def _assign_pairs_to_patches(
        self,
        gate_pairs: List[Tuple[int, int]],
        layout: GridLayout,
        patches: List[Tuple[int, int, int, int]],
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Assign gate pairs to patches based on qubit positions.
        
        A pair is assigned to a patch if both qubits are within the patch
        (including overlap region).
        """
        assignments: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        
        for pair in gate_pairs:
            ion_a, ion_b = pair
            pos_a = layout.get_position(ion_a)
            pos_b = layout.get_position(ion_b)
            
            if pos_a is None or pos_b is None:
                continue
            
            # Find best patch for this pair
            best_patch_idx = None
            best_distance = float('inf')
            
            for idx, (r0, c0, r1, c1) in enumerate(patches):
                # Check if both qubits are in patch
                if (r0 <= pos_a[0] < r1 and c0 <= pos_a[1] < c1 and
                    r0 <= pos_b[0] < r1 and c0 <= pos_b[1] < c1):
                    # Compute distance to patch center
                    center_r = (r0 + r1) / 2
                    center_c = (c0 + c1) / 2
                    dist = (abs(pos_a[0] - center_r) + abs(pos_a[1] - center_c) +
                            abs(pos_b[0] - center_r) + abs(pos_b[1] - center_c))
                    if dist < best_distance:
                        best_distance = dist
                        best_patch_idx = idx
            
            if best_patch_idx is not None:
                assignments[best_patch_idx].append(pair)
            else:
                # Pair spans multiple patches - assign to first containing either
                for idx, (r0, c0, r1, c1) in enumerate(patches):
                    if (r0 <= pos_a[0] < r1 and c0 <= pos_a[1] < c1):
                        assignments[idx].append(pair)
                        break
        
        return assignments
    
    def _route_patch(
        self,
        patch: Tuple[int, int, int, int],
        pairs: List[Tuple[int, int]],
        layout: GridLayout,
        architecture: "HardwareArchitecture",
    ) -> RoutingResult:
        """Route within a single patch using base SAT router."""
        r0, c0, r1, c1 = patch
        patch_h = r1 - r0
        patch_w = c1 - c0
        
        # Extract sub-layout
        sub_grid = layout.grid[r0:r1, c0:c1].copy()
        sub_layout = GridLayout(grid=sub_grid)
        
        # Translate pairs to local indices
        local_pairs = []
        for ion_a, ion_b in pairs:
            pos_a = layout.get_position(ion_a)
            pos_b = layout.get_position(ion_b)
            if pos_a is None or pos_b is None:
                continue
            # Map global position to local
            if (r0 <= pos_a[0] < r1 and c0 <= pos_a[1] < c1 and
                r0 <= pos_b[0] < r1 and c0 <= pos_b[1] < c1):
                local_pairs.append((ion_a, ion_b))
        
        if not local_pairs:
            return RoutingResult(success=True, operations=[], cost=0.0)
        
        # Create mock architecture for sub-grid
        class SubArchitecture:
            def __init__(self, rows, cols, capacity):
                self.n_rows = rows
                self.n_cols = cols
                self.num_qubits = rows * cols
                self.capacity = capacity
                self.grid_shape = (rows, cols)
        
        sub_arch = SubArchitecture(
            patch_h, patch_w,
            getattr(architecture, 'capacity', 2)
        )
        
        # Build sub-mapping
        sub_mapping = QubitMapping()
        for ion_idx, (row, col) in sub_layout.ion_positions.items():
            sub_mapping.add_mapping(ion_idx, ion_idx, f"trap_{row}_{col}")
        
        # Route using base SAT solver
        result = super().route_batch(local_pairs, sub_mapping, sub_arch)
        
        # Translate operations back to global coordinates
        if result.success and result.operations:
            translated_ops = []
            for op in result.operations:
                if isinstance(op, dict):
                    new_op = dict(op)
                    if 'row' in new_op:
                        new_op['row'] += r0
                    if 'col' in new_op:
                        new_op['col'] += c0
                    translated_ops.append(new_op)
                else:
                    translated_ops.append(op)
            result.operations = translated_ops
        
        return result
    
    def _apply_patch_result(
        self,
        layout: GridLayout,
        result: RoutingResult,
        patch: Tuple[int, int, int, int],
    ) -> None:
        """Apply routing result to update the layout."""
        r0, c0, r1, c1 = patch
        
        for op in result.operations:
            if isinstance(op, dict):
                op_type = op.get('type', '')
                row = op.get('row', 0)
                col = op.get('col', 0)
                
                if op_type == 'H_SWAP':
                    if 0 <= row < layout.n_rows and 0 <= col < layout.n_cols - 1:
                        layout.swap_horizontal(row, col)
                elif op_type == 'V_SWAP':
                    if 0 <= row < layout.n_rows - 1 and 0 <= col < layout.n_cols:
                        layout.swap_vertical(row, col)
    
    def _check_all_pairs_satisfied(
        self,
        pairs: List[Tuple[int, int]],
        layout: GridLayout,
    ) -> bool:
        """Check if all gate pairs are now adjacent."""
        capacity = getattr(self.config, 'patch_width', 2)
        return self._check_pairs_satisfied(layout, pairs, capacity)


# =============================================================================
# Greedy Ion Router
# =============================================================================

class GreedyIonRouter(Router):
    """Fast greedy routing for small instances.
    
    Uses simple heuristics instead of SAT solving for cases where
    optimal routing is not critical.
    """
    
    def __init__(self, name: str = "greedy_ion_router"):
        super().__init__(RoutingStrategy.GREEDY, name)
    
    def route_gate(
        self,
        gate_qubits: Tuple[int, ...],
        current_mapping: QubitMapping,
        architecture: "HardwareArchitecture",
    ) -> RoutingResult:
        """Route greedily for a single gate."""
        if len(gate_qubits) != 2:
            return RoutingResult(
                success=False,
                metrics={"error": "Only two-qubit gates supported"},
            )
        
        q1, q2 = gate_qubits
        p1 = current_mapping.get_physical(q1)
        p2 = current_mapping.get_physical(q2)
        
        if p1 is None or p2 is None:
            return RoutingResult(success=False)
        
        # Simple: move p2 toward p1
        # This is a placeholder - real implementation would use architecture graph
        return RoutingResult(
            success=True,
            operations=[],  # No-op for now
            cost=0.0,
        )


# =============================================================================
# WISE Routing Pass (Compilation Pipeline Integration)
# =============================================================================

class WISERoutingPass:
    """Compilation pass that routes using WiseSatRouter.
    
    Integrates the WISE SAT-based router into the compilation pipeline.
    
    This pass:
    1. Analyzes the MappedCircuit to identify gate pairs per round
    2. Groups gates into batches that can be routed together
    3. Uses WiseSatRouter to compute optimal ion permutations
    4. Generates transport operations for each routing schedule
    5. Outputs a RoutedCircuit with all operations
    
    Example
    -------
    >>> config = WISERoutingConfig(timeout_seconds=30, max_passes=6)
    >>> routing_pass = WISERoutingPass(config)
    >>> routed = routing_pass.route(mapped_circuit, architecture)
    
    See Also
    --------
    WiseSatRouter : Underlying SAT-based router.
    WISEBatchScheduler : Schedules the resulting operations.
    """
    
    def __init__(
        self,
        config: Optional[WISERoutingConfig] = None,
        use_patch_routing: bool = False,
    ):
        """Initialize the routing pass.
        
        Parameters
        ----------
        config : Optional[WISERoutingConfig]
            Configuration for the SAT solver.
        use_patch_routing : bool
            If True, use patch-based routing for large grids.
        """
        self.config = config or WISERoutingConfig()
        self.use_patch_routing = use_patch_routing or self.config.patch_enabled
        
        # Create the appropriate router
        if self.use_patch_routing:
            self.router = WisePatchRouter(config=self.config)
        else:
            self.router = WiseSatRouter(config=self.config)
    
    def route(
        self,
        mapped_circuit: "MappedCircuit",
        architecture: "HardwareArchitecture",
    ) -> "RoutedCircuit":
        """Route a mapped circuit.
        
        Parameters
        ----------
        mapped_circuit : MappedCircuit
            Circuit with logical-to-physical mapping.
        architecture : HardwareArchitecture
            Target WISE grid architecture.
            
        Returns
        -------
        RoutedCircuit
            Circuit with routing operations inserted.
        """
        from qectostim.experiments.hardware_simulation.core.pipeline import (
            MappedCircuit,
            RoutedCircuit,
        )
        
        # Extract gate pairs from circuit
        gate_batches = self._extract_gate_batches(mapped_circuit)
        
        if not gate_batches:
            # No two-qubit gates, return empty routed circuit
            return RoutedCircuit(
                operations=[],
                final_mapping=mapped_circuit.mapping.copy(),
                routing_overhead=0,
            )
        
        # Route each batch and collect operations
        all_operations: List[Any] = []
        current_mapping = mapped_circuit.mapping.copy()
        total_routing_ops = 0
        
        for batch_idx, gate_pairs in enumerate(gate_batches):
            wise_logger.debug(f"Routing batch {batch_idx} with {len(gate_pairs)} pairs")
            
            # Route this batch
            result = self.router.route_batch(
                gate_pairs,
                current_mapping,
                architecture,
            )
            
            if not result.success:
                wise_logger.warning(
                    f"Routing failed for batch {batch_idx}: {result.metrics}"
                )
                # Use heuristic fallback or raise error
                continue
            
            # Add routing operations
            if result.operations:
                all_operations.extend(
                    self._convert_to_physical_ops(result.operations, architecture)
                )
                total_routing_ops += len(result.operations)
            
            # Add the original gates (now that qubits are in position)
            for logical_q1, logical_q2 in gate_pairs:
                # Get the gate from the mapped circuit
                # For now, create placeholder gate operations
                all_operations.append({
                    "type": "GATE",
                    "qubits": (logical_q1, logical_q2),
                    "batch": batch_idx,
                })
            
            # Update mapping for next batch
            if result.final_mapping:
                current_mapping = result.final_mapping
        
        return RoutedCircuit(
            operations=all_operations,
            final_mapping=current_mapping,
            routing_overhead=total_routing_ops,
        )
    
    def _extract_gate_batches(
        self,
        mapped_circuit: "MappedCircuit",
    ) -> List[List[Tuple[int, int]]]:
        """Extract two-qubit gate pairs grouped into batches.
        
        Gates that can execute in parallel form a batch.
        Currently uses simple greedy grouping.
        """
        batches: List[List[Tuple[int, int]]] = []
        current_batch: List[Tuple[int, int]] = []
        used_qubits: Set[int] = set()
        
        for gate, logical_qubits in mapped_circuit.native_circuit:
            if len(logical_qubits) != 2:
                continue
            
            q1, q2 = logical_qubits
            
            # Check if we can add to current batch (no qubit conflicts)
            if q1 in used_qubits or q2 in used_qubits:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [(q1, q2)]
                used_qubits = {q1, q2}
            else:
                current_batch.append((q1, q2))
                used_qubits.add(q1)
                used_qubits.add(q2)
        
        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _convert_to_physical_ops(
        self,
        routing_ops: List[Any],
        architecture: "HardwareArchitecture",
    ) -> List[Any]:
        """Convert routing schedule operations to physical operations.
        
        TODO: Generate actual Split/Merge/Move operations.
        For now, returns the dict-based operations.
        """
        physical_ops = []
        
        for op in routing_ops:
            if isinstance(op, dict):
                # Already in dict format from WiseSatRouter
                physical_ops.append(op)
            else:
                # Handle other operation types
                physical_ops.append({"type": "ROUTING", "data": op})
        
        return physical_ops


class WISECostModel:
    """Cost model for WISE trapped-ion routing.
    
    Estimates costs based on:
    - Transport time (proportional to distance)
    - Gate execution time
    - Reconfiguration overhead
    
    This implements the CostModel interface from core/compiler.py.
    """
    
    def __init__(
        self,
        transport_time_per_unit: float = 10.0,  # μs per grid unit
        gate_time_2q: float = 100.0,            # μs for two-qubit gate
        gate_time_1q: float = 10.0,             # μs for single-qubit gate
        reconfiguration_overhead: float = 50.0,  # μs per reconfiguration
    ):
        self.transport_time_per_unit = transport_time_per_unit
        self.gate_time_2q = gate_time_2q
        self.gate_time_1q = gate_time_1q
        self.reconfiguration_overhead = reconfiguration_overhead
    
    def operation_cost(
        self,
        operation_type: str,
        qubits: Tuple[int, ...],
        params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Estimate cost of a single operation."""
        params = params or {}
        
        if operation_type in ("H_SWAP", "V_SWAP", "transport", "swap"):
            distance = params.get("distance", 1.0)
            return distance * self.transport_time_per_unit
        elif operation_type in ("MS", "XX", "ZZ", "2Q", "gate_2q"):
            return self.gate_time_2q
        elif operation_type in ("R", "RZ", "RX", "RY", "1Q", "gate_1q"):
            return self.gate_time_1q
        elif operation_type == "reconfiguration":
            return self.reconfiguration_overhead
        else:
            return 1.0  # Unknown operation
    
    def sequence_cost(
        self,
        operations: List[Tuple[str, Tuple[int, ...], Optional[Dict[str, Any]]]],
    ) -> float:
        """Estimate total cost of an operation sequence.
        
        Accounts for parallelism within phases (H or V).
        """
        if not operations:
            return 0.0
        
        # Group by phase for parallelism
        phase_costs: Dict[str, float] = defaultdict(float)
        
        for op_type, qubits, params in operations:
            cost = self.operation_cost(op_type, qubits, params)
            
            if op_type == "H_SWAP":
                phase_costs["H"] = max(phase_costs["H"], cost)
            elif op_type == "V_SWAP":
                phase_costs["V"] = max(phase_costs["V"], cost)
            else:
                phase_costs["other"] += cost
        
        return sum(phase_costs.values())
    
    def compare(self, cost_a: float, cost_b: float) -> int:
        """Compare two costs (-1 if a better, 0 if equal, 1 if b better)."""
        if cost_a < cost_b:
            return -1
        elif cost_a > cost_b:
            return 1
        return 0
    
    def is_acceptable(self, cost: float, threshold: float = 10000.0) -> bool:
        """Check if a cost is acceptable."""
        return cost < threshold


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # SAT Encoder
    "WISESATEncoder",
    "WISESATContext",
    # Routers
    "WiseSatRouter",
    "WisePatchRouter",
    "GreedyIonRouter",
    # Compilation Pass
    "WISERoutingPass",
    # Cost Model
    "WISECostModel",
    # Configuration
    "WISERoutingConfig",
    # Data structures
    "GridLayout",
    "RoutingPass",
    "RoutingSchedule",
    "GatePairRequirement",
    # Utilities
    "compute_target_positions",
    # Logger
    "WISE_LOGGER_NAME",
    "wise_logger",
    # pysat availability
    "_PYSAT_AVAILABLE",
]
