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

# Soft import of tqdm for progress bars (falls back to no-op)
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    # Provide a no-op fallback when tqdm is not installed
    class _tqdm:  # type: ignore[no-redef]
        """Minimal no-op tqdm stand-in."""
        def __init__(self, iterable=None, *a, **kw):
            self._it = iterable
        def __iter__(self):
            return iter(self._it) if self._it is not None else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            pass
        def set_postfix_str(self, s, refresh=True):
            pass
        def set_description(self, desc, refresh=True):
            pass
        def close(self):
            pass

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
# Reconfiguration heating constants
# =============================================================================

# Import canonical transport constants from the transport module.
# These are derived from individual transport operation physics and kept
# in sync via the transport module's computed aggregates.
from qectostim.experiments.hardware_simulation.trapped_ion.transport import (
    ROW_SWAP_HEATING as _RSH,
    COL_SWAP_HEATING as _CSH,
    ROW_SWAP_TIME_S as _RST,
    COL_SWAP_TIME_S as _CST,
    Split as _Split,
    JunctionCrossing as _JC,
    Move as _Move,
)

# Motional quanta deposited per ion per swap.
ROW_SWAP_HEATING: float = _RSH   # ≈ 9.734e-4 quanta per ion per row swap
COL_SWAP_HEATING: float = _CSH   # ≈ 9.005e-4 quanta per ion per column swap
ROW_SWAP_TIME_US: float = _RST * 1e6   # μs
COL_SWAP_TIME_US: float = _CST * 1e6   # μs

# Per-pass reconfiguration timing (old code's _runOddEvenReconfig model).
# In the old code, one parallel H-pass takes one ``row_swap_time`` and one
# parallel V-pass takes one ``col_swap_time``, regardless of how many
# individual swaps occur in that pass (they execute in parallel).
# The initial split overhead is added once per reconfiguration step.
INITIAL_SPLIT_TIME_US: float = _Split.SPLITTING_TIME * 1e6  # 80 µs
H_PASS_TIME_US: float = ROW_SWAP_TIME_US  # 212 µs per H-pass
# Old code V-pass formula:
#   col_swap_time = 2*JC + (4*JC + Move)*2
# This models the full junction-crossing sequence for a vertical swap pass.
V_PASS_TIME_US: float = (
    2.0 * _JC.CROSSING_TIME + (4.0 * _JC.CROSSING_TIME + _Move.MOVING_TIME) * 2
) * 1e6  # 510 µs per V-pass


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
    subgridsize : Tuple[int, int, int]
        Patch decomposition dimensions (cols, rows, increment).
        Old code default: (6, 4, 1).
    base_pmax_in : Optional[int]
        Base pass-bound for SAT solver. If None, derived from
        number of rounds.
    lookahead_rounds : int
        Number of future gate batches to aggregate when routing.
        The solver considers current batch + lookahead_rounds
        batches simultaneously, then chains layouts forward.
    max_cycles : int
        Maximum tiling cycles for patch routing before giving up.
    """
    # WISE-specific parameters (base params inherited from SATRoutingConfig)
    patch_enabled: bool = False
    patch_height: int = 4
    patch_width: int = 4
    bt_soft_weight: int = 5
    boundary_soft_weight_row: int = 0
    boundary_soft_weight_col: int = 0
    subgridsize: Tuple[int, int, int] = (6, 4, 1)
    base_pmax_in: Optional[int] = None
    lookahead_rounds: int = 2
    max_cycles: int = 10
    boundary_capacity_factor: float = 1.0
    # --- fields ported from old best_effort_compilation_WISE ---
    barrier_threshold: float = float('inf')
    go_back_threshold: float = 0.0
    
    @classmethod
    def from_env(cls) -> "WISERoutingConfig":
        """Create config from environment variables."""
        # Parse subgridsize from env (format: "cols,rows,inc" e.g. "6,4,1")
        subgridsize_str = os.environ.get("WISE_SUBGRIDSIZE", "6,4,1")
        try:
            parts = [int(x.strip()) for x in subgridsize_str.split(",")]
            subgridsize = (parts[0], parts[1], parts[2]) if len(parts) >= 3 else (6, 4, 1)
        except (ValueError, IndexError):
            subgridsize = (6, 4, 1)
        
        base_pmax_str = os.environ.get("WISE_BASE_PMAX", "")
        base_pmax_in = int(base_pmax_str) if base_pmax_str.isdigit() else None
        
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
            subgridsize=subgridsize,
            base_pmax_in=base_pmax_in,
            lookahead_rounds=int(os.environ.get("WISE_LOOKAHEAD", "2")),
            max_cycles=int(os.environ.get("WISE_MAX_CYCLES", "10")),
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
    # Grid origin for patch-based routing (row_offset, col_offset)
    grid_origin: Tuple[int, int] = (0, 0)
    # Cross-boundary directional preferences per round
    cross_boundary_prefs: Optional[List[Dict[int, Set[str]]]] = None
    # Boundary adjacency flags
    boundary_adjacent: Optional[Dict[str, bool]] = None
    # Whether the first round starts from an arbitrary layout
    ignore_initial_reconfig: bool = False


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
        # Respect the explicit use_maxsat arg: only fall back to config
        # when the caller did NOT pass use_maxsat=True explicitly.
        self.use_maxsat = use_maxsat  # honour the caller's explicit choice
        
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
            Mirrors the old code's Σ_r P_r bound.
        """
        self.context = context
        self.rows = context.n_rows
        self.cols = context.n_cols
        self._items = list(context.ions)
        self._sum_bound_B = sum_bound_B
        
        # Determine which rounds get extra passes (ignore_initial_reconfig)
        n, m = context.n_rows, context.n_cols
        R = context.num_rounds
        optimize_round_start = (
            1 if (context.ignore_initial_reconfig and R > 0) else 0
        )
        self._optimize_round_start = optimize_round_start
        
        # Variable P_bounds: early rounds get extra passes for full sort
        self._pass_bounds = (
            [pass_bound + n + m] * optimize_round_start
            + [pass_bound] * (R - optimize_round_start)
        )
    
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

    def add_displacement_soft_constraints(self) -> None:
        """Add soft clauses penalising large ion displacement.

        For each round *r* and each ion, a soft clause encourages the
        ion to stay in the same row (weighted by
        ``config.boundary_soft_weight_row``) and the same column block
        (weighted by ``config.boundary_soft_weight_col``) it occupied at
        the start of the round.  This drives the MaxSAT solver to
        minimise total ion movement, which reduces transport heating.

        These clauses are only added when ``use_maxsat`` is True and the
        corresponding weight is > 0.
        """
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
            P_start = 0  # beginning of round

            for ion in ctx.ions:
                # --- row displacement penalty ---
                if w_row > 0:
                    for d in range(n):
                        # row_end(r, ion, d) at start_pass → row_end(r, ion, d) at final_pass
                        # Encourage: if ion was in row d at start, it stays in row d
                        re_start = self.var_row_end(r, ion, d) if P_start == 0 else None
                        re_end = self.var_row_end(r, ion, d)

                        # Simpler approach: penalise being in a different
                        # row at the end than we'd like.  Use the initial
                        # layout to determine the "home" row.
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
                            # Encourage ion to be in home_row at round end
                            self._add_soft(
                                [self.var_row_end(r, ion, home_row)],
                                weight=w_row,
                            )

                # --- column displacement penalty ---
                if w_col > 0:
                    init_layout = ctx.initial_layout
                    home_block = None
                    for dr in range(n):
                        for dc in range(m):
                            if init_layout[dr][dc] == ion:
                                # Block = dc // block_width
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
    
    def add_pass_usage_constraints(self) -> None:
        """Add pass-usage (u-variable) constraints and global Σ_r P_r bound.

        For every (r, p) define a Boolean ``u[r,p]`` that is true iff
        any comparator (horizontal or vertical) is active in that pass.
        Then, if ``_sum_bound_B`` is set, add an at-most cardinality
        constraint on the total number of active passes.

        This is the port of the old code's u-variable + SUM_BOUND logic.
        """
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

                # u ↔ OR(comp_lits)
                self._add_hard([-u_rp] + comp_lits, "UTIL_U")
                for s_lit in comp_lits:
                    self._add_hard([-s_lit, u_rp], "UTIL_U")

        # Global Σ_r P_r ≤ sum_bound_B cardinality constraint
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
        """Add cross-boundary hard and soft constraints.

        Ports the old code's ``CROSS_BOUNDARY`` hard clauses and
        boundary-preference soft clauses.  For each ion with
        cross-boundary directional preferences, the ion is forced
        into the band of cells nearest the target boundary.
        """
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

        # ---- capacity-factor limiting (matches old code) ----
        # Only the first ``dir_capacity`` ions per direction per round
        # get the hard CROSS_BOUNDARY clause.  The old code searched
        # over boundary_capacity_factor ∈ [0..1] in parallel; here we
        # use the single value from the config.
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

        # Collect ions per (round, direction) and sort for determinism.
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

        # Determine which directions each ion is *enforced* for.
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
                    # Fallback: union of individual direction bands
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

        # Soft clauses: encourage inner-pair ions *away* from boundaries,
        # and cross-pair ions *toward* their target boundary.
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
                    # Cross-boundary ions: encourage toward boundary
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

                    # Inner-pair ions: discourage boundary positions
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
        # NOTE: add_target_position_constraints() is NOT called here.
        # BT constraints are added explicitly in Phase 2 (MaxSAT) as
        # soft constraints — adding them as hard in Phase 1 (pure SAT)
        # can make the problem UNSAT or inflate D*.
        self.add_pass_usage_constraints()  # u-variables + sum_bound_B
        self.add_cross_boundary_constraints()  # cross-boundary hard + soft
        self.add_displacement_soft_constraints()
        
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
        """Solve using plain SAT (Minisat22) with timeout enforcement."""
        t0 = time.time()
        
        if timeout is not None and timeout <= 0:
            return SATSolution(
                satisfiable=False,
                statistics={"status": "timeout_zero"},
            )
        
        try:
            # WCNF has .hard, CNF has .clauses
            if hasattr(self.formula, 'clauses'):
                clauses = self.formula.clauses
            elif hasattr(self.formula, 'hard'):
                clauses = self.formula.hard
            else:
                clauses = []
            with Minisat22(bootstrap_with=clauses, use_timer=True) as solver:
                # Enforce timeout via conflict budget.  Empirically
                # ~50 000 conflicts ≈ 1 s on modern hardware for
                # formulas of the size we generate.  Scale linearly.
                eff_timeout = timeout if timeout is not None else 60.0
                conflict_budget = max(5000, int(50_000 * eff_timeout))
                solver.conf_budget(conflict_budget)

                if assumptions:
                    sat_ok = solver.solve_limited(assumptions=assumptions)
                else:
                    sat_ok = solver.solve_limited()

                # solve_limited returns True/False/None (None = interrupted)
                if sat_ok is None:
                    # Budget exhausted → treat as timeout
                    return SATSolution(
                        satisfiable=False,
                        solve_time=time.time() - t0,
                        statistics={"status": "timeout",
                                    "conflict_budget": conflict_budget},
                    )
                
                model = solver.get_model() if sat_ok else None
                
                t1 = time.time()
                
                if sat_ok and model:
                    # Cache the model so Phase 3 decode can use it directly
                    self._last_model = model
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
        """Solve using MaxSAT (RC2) with timeout enforcement."""
        try:
            from pysat.examples.rc2 import RC2
        except ImportError:
            wise_logger.warning("RC2 not available, falling back to SAT")
            return self._solve_sat(timeout)
        
        t0 = time.time()
        eff_timeout = timeout if timeout is not None else 60.0
        
        try:
            # Run RC2 in a thread with a timeout, since RC2.compute()
            # has no built-in timeout mechanism.
            import threading
            result_holder: List[Any] = [None]  # [model]
            cost_holder: List[Any] = [0]
            exc_holder: List[Optional[Exception]] = [None]

            def _rc2_worker():
                try:
                    with RC2(self.formula) as rc2:
                        model = rc2.compute()
                        result_holder[0] = list(model) if model else None
                        cost_holder[0] = rc2.cost if hasattr(rc2, 'cost') else 0
                except Exception as e:
                    exc_holder[0] = e

            worker = threading.Thread(target=_rc2_worker, daemon=True)
            worker.start()
            worker.join(timeout=eff_timeout)

            if worker.is_alive():
                # Timeout — thread is still running; we cannot kill it
                # but we return immediately with a timeout result.
                wise_logger.warning(
                    "MaxSAT RC2 timed out after %.1f s", eff_timeout
                )
                return SATSolution(
                    satisfiable=False,
                    solve_time=time.time() - t0,
                    statistics={"status": "timeout"},
                )

            if exc_holder[0] is not None:
                raise exc_holder[0]

            model = result_holder[0]
            cost = cost_holder[0]
            t1 = time.time()
                
            if model:
                # Cache the model so Phase 3 decode can use it directly
                self._last_model = model
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
        lookahead_pairs: Optional[List[List[Tuple[int, int]]]] = None,
        bt_positions: Optional[List[Dict[int, Tuple[int, int]]]] = None,
    ) -> RoutingResult:
        """Route a batch of two-qubit gates using SAT optimization.
        
        Parameters
        ----------
        gate_pairs : List[Tuple[int, int]]
            Pairs of logical qubits needing interaction (current batch).
        current_mapping : QubitMapping
            Current qubit positions.
        architecture : HardwareArchitecture
            Target WISE grid architecture.
        lookahead_pairs : Optional[List[List[Tuple[int, int]]]]
            Additional future batches to consider when optimizing.
            The solver will find a layout that is good for all batches,
            but only the routing for ``gate_pairs`` is applied.
        bt_positions : Optional[List[Dict[int, Tuple[int, int]]]]
            Boundary target (BT) positions from a previous multi-round
            solve.  Each list entry pins ions to their planned positions
            for that round, reducing the SAT search space.
            
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
        
        # Build full gate pair list for multi-round SAT solving
        all_round_pairs: List[List[Tuple[int, int]]] = [physical_pairs]
        if lookahead_pairs:
            for batch in lookahead_pairs:
                round_physical = []
                for q1, q2 in batch:
                    p1 = current_mapping.get_physical(q1)
                    p2 = current_mapping.get_physical(q2)
                    if p1 is not None and p2 is not None:
                        round_physical.append((p1, p2))
                if round_physical:
                    all_round_pairs.append(round_physical)
        
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
        
        # Use SAT solver with multi-round pairs if lookahead active
        try:
            schedule = self._solve_sat(
                initial_layout,
                physical_pairs,
                n_rows,
                n_cols,
                capacity,
                all_round_pairs=all_round_pairs if len(all_round_pairs) > 1 else None,
                bt_positions=bt_positions,
            )
            
            if schedule is None:
                return RoutingResult(
                    success=False,
                    metrics={"error": "SAT solver found no solution"},
                )
            
            # ----------------------------------------------------------
            # Only use ROUND-0 operations for execution.  The schedule
            # may contain multiple rounds (when lookahead > 0), but the
            # old code only applies the current round's reconfiguration.
            # Future round layouts are extracted as BTs for subsequent
            # iterations.
            # ----------------------------------------------------------
            round0_passes = (
                schedule.passes_per_round[0]
                if schedule.passes_per_round
                else []
            )
            round0_ops: List[Dict[str, Any]] = []
            for p_idx, p_info in enumerate(round0_passes):
                if p_info.phase == "H":
                    # Split H-swaps by column parity (port of old
                    # code's _infer_pass_parity for H-phase).
                    even_h = [(r, c) for r, c in p_info.h_swaps if c % 2 == 0]
                    odd_h  = [(r, c) for r, c in p_info.h_swaps if c % 2 == 1]
                    for parity_group in (even_h, odd_h):
                        if not parity_group:
                            continue
                        for r, c in parity_group:
                            round0_ops.append(
                                {"type": "H_SWAP", "row": r, "col": c}
                            )
                        round0_ops.append({"type": "PASS_BOUNDARY"})
                else:
                    # Split V-swaps by row parity (port of old
                    # code's _infer_pass_parity for V-phase).
                    even_v = [(r, c) for r, c in p_info.v_swaps if r % 2 == 0]
                    odd_v  = [(r, c) for r, c in p_info.v_swaps if r % 2 == 1]
                    for parity_group in (even_v, odd_v):
                        if not parity_group:
                            continue
                        for r, c in parity_group:
                            round0_ops.append(
                                {"type": "V_SWAP", "row": r, "col": c}
                            )
                        round0_ops.append({"type": "PASS_BOUNDARY"})
            # Strip trailing boundary
            if round0_ops and round0_ops[-1].get("type") == "PASS_BOUNDARY":
                round0_ops.pop()            
            # Update mapping from round-0 layout only
            round0_layout = (
                schedule.layouts[0]
                if schedule.layouts
                else initial_layout
            )
            final_mapping = self._layout_to_mapping(
                round0_layout, current_mapping
            )
            
            # Count H vs V swaps from round 0 only
            h_swap_count = sum(
                len(p.h_swaps) for p in round0_passes if p.phase == "H"
            )
            v_swap_count = sum(
                len(p.v_swaps) for p in round0_passes if p.phase == "V"
            )
            h_pass_count = sum(
                1 for p in round0_passes
                if p.phase == "H" and p.has_swaps
            )
            v_pass_count = sum(
                1 for p in round0_passes
                if p.phase == "V" and p.has_swaps
            )

            # Extract future-round layouts as BTs for the next
            # iteration.  layout[r] = ion positions after round r's
            # swaps.  For the next call (batch_idx+1), round 0 = the
            # current round 1, so BTs shift by one.
            future_bt_positions: List[Dict[int, Tuple[int, int]]] = []
            for future_idx in range(1, len(schedule.layouts)):
                layout = schedule.layouts[future_idx]
                future_bt_positions.append(
                    dict(layout.ion_positions)
                )

            return RoutingResult(
                success=True,
                operations=round0_ops,
                cost=float(h_swap_count + v_swap_count),
                final_mapping=final_mapping,
                metrics={
                    "total_passes": len(round0_passes),
                    "total_swaps": h_swap_count + v_swap_count,
                    "h_swaps": h_swap_count,
                    "v_swaps": v_swap_count,
                    "h_passes": h_pass_count,
                    "v_passes": v_pass_count,
                    "num_rounds_decoded": len(
                        schedule.passes_per_round
                    ),
                    "_future_bt_positions": future_bt_positions,
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
        """Build GridLayout from QubitMapping.

        Placement sources (tried in order):
        1. ``mapping.zone_assignments`` — zone IDs of the form
           ``"trap_{row}_{col}"`` are parsed to place each physical
           qubit on the grid.
        2. **Positional fallback** — when zone_assignments are empty
           (or unparseable) we derive ``(row, col)`` from the physical
           qubit index: ``row = phys // n_cols, col = phys % n_cols``.
           This matches the convention used by
           ``WISECompiler.map_qubits()`` and guarantees a valid,
           fully-populated grid for the SAT solver.

        Remaining cells (if any) are filled with unique indices so that
        every grid slot contains a distinct ion identifier, which is
        required by the permutation constraints in the SAT encoder.
        """
        grid = np.zeros((n_rows, n_cols), dtype=int)
        placed: set = set()

        # --- Strategy 1: zone_assignments ---------------------------------
        for logical, physical in mapping.logical_to_physical.items():
            zone = mapping.zone_assignments.get(physical)
            if zone:
                try:
                    parts = zone.split("_")
                    row, col = int(parts[-2]), int(parts[-1])
                    if 0 <= row < n_rows and 0 <= col < n_cols and (row, col) not in placed:
                        grid[row, col] = physical
                        placed.add((row, col))
                except (ValueError, IndexError):
                    pass

        # --- Strategy 2: positional fallback for any unplaced qubits ------
        if len(placed) < len(mapping.logical_to_physical):
            for logical, physical in mapping.logical_to_physical.items():
                row = physical // n_cols if n_cols > 0 else 0
                col = physical % n_cols if n_cols > 0 else physical
                if 0 <= row < n_rows and 0 <= col < n_cols and (row, col) not in placed:
                    grid[row, col] = physical
                    placed.add((row, col))

        # --- Fill remaining cells with unique indices ---------------------
        used_ids = set(grid.flatten()) - {0}
        # Also include 0 if it was legitimately placed
        if (0, 0) in placed or 0 in mapping.logical_to_physical.values():
            used_ids.add(0)
        next_idx = (max(used_ids) + 1) if used_ids else 0
        for r in range(n_rows):
            for c in range(n_cols):
                if (r, c) not in placed:
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
        all_round_pairs: Optional[List[List[Tuple[int, int]]]] = None,
        bt_positions: Optional[List[Dict[int, Tuple[int, int]]]] = None,
    ) -> Optional[RoutingSchedule]:
        """Solve routing using SAT/MaxSAT with WISESATEncoder.
        
        Parameters
        ----------
        initial_layout : GridLayout
            Starting ion arrangement.
        pairs : List[Tuple[int, int]]
            Ion pairs that need to be adjacent (primary batch).
        n_rows : int
            Grid rows.
        n_cols : int
            Grid columns.
        capacity : int
            Gating block width.
        all_round_pairs : Optional[List[List[Tuple[int, int]]]]
            Multi-round gate pairs for lookahead optimization.
        bt_positions : Optional[List[Dict[int, Tuple[int, int]]]]
            Boundary target positions from a previous solve.
            Each entry maps ion_idx -> (row, col) target for that round.
            
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
            # Use actual ion indices from the grid, NOT range(n*m).
            # In patch routing, a sub-grid can contain non-contiguous
            # global ion indices (e.g. [0,1,4,5]) and the old code
            # derives ions_all from A_in.flatten().
            ions = sorted(set(int(x) for x in initial_layout.grid.flatten()))
            
            # Compute gating blocks, aligned to the global grid when
            # this is a sub-patch (set by _route_patch via
            # _current_grid_origin).
            _origin = getattr(self, '_current_grid_origin', (0, 0))
            col_offset = _origin[1] if _origin else 0
            block_cells, block_fully_inside, block_widths = self._compute_blocks(
                n_rows, n_cols, capacity, col_offset=col_offset
            )
            num_blocks = len(block_cells)
            
            # Build encoding context.
            #
            # KEY INSIGHT: When BTs (boundary targets) from a previous
            # multi-round solve are available, we switch to SINGLE-ROUND
            # encoding.  BTs already capture where ions should end up
            # (from the previous window's future-round solutions), so
            # multi-round SAT is redundant and only inflates D* — making
            # more passes necessary.  Single-round + BT soft constraints
            # in Phase 2 achieves the same layout quality with minimal D*.
            #
            # Multi-round encoding is only used when NO BTs are available
            # (first window), to generate initial BTs for subsequent
            # windows.  This matches the old code's pattern where
            # ionRoutingWISEArch uses BTs from previous _patch_and_route
            # calls to guide single-round solves.
            #
            # NOTE: all_round_pairs already includes the current batch
            # (physical_pairs) as its first element, so we use it
            # directly — do NOT prepend ``pairs`` again.
            _have_bt = bt_positions and any(bt for bt in bt_positions if bt)
            if _have_bt:
                # BTs available → single-round (BTs guide Phase 2)
                num_rounds = 1
                gate_pairs_list = [pairs]
            elif all_round_pairs:
                # No BTs, lookahead available → multi-round for BT generation
                num_rounds = len(all_round_pairs)
                gate_pairs_list = list(all_round_pairs)
            else:
                num_rounds = 1
                gate_pairs_list = [pairs]
            
            # Create context (multi-round for lookahead)
            # Pick up cross-boundary prefs if set by the patch router.
            _cb_prefs = getattr(self, '_patch_cross_boundary_prefs', None)
            _b_adj = getattr(self, '_patch_boundary_adjacent', None)

            # Populate target_positions from BTs (boundary targets).
            # When a previous multi-round solve provided future-round
            # target layouts, using them as BTs pins ions to their
            # planned positions, dramatically reducing the SAT search
            # space and improving solution quality — matching the old
            # code's BT mechanism in ionRoutingWISEArch.
            if bt_positions:
                target_positions: List[Dict[int, Tuple[int, int]]] = []
                for r in range(num_rounds):
                    if r < len(bt_positions) and bt_positions[r]:
                        target_positions.append(dict(bt_positions[r]))
                    else:
                        target_positions.append({})
            else:
                target_positions = [{} for _ in range(num_rounds)]

            context = WISESATContext(
                initial_layout=initial_layout.grid,
                target_positions=target_positions,
                gate_pairs=gate_pairs_list,
                full_gate_pairs=gate_pairs_list,
                ions=ions,
                n_rows=n_rows,
                n_cols=n_cols,
                num_rounds=num_rounds,
                block_cells=block_cells,
                block_fully_inside=block_fully_inside,
                block_widths=block_widths,
                num_blocks=num_blocks,
                grid_origin=_origin,
                cross_boundary_prefs=_cb_prefs,
                boundary_adjacent=_b_adj,
                debug_diag=self.config.debug_mode,
            )
            
            # ---------------------------------------------------------------
            # Phase 1: D-minimization — binary search for minimum pass bound
            # ---------------------------------------------------------------
            # The old code searches for the smallest pass bound (displacement
            # proxy) at which the SAT problem is satisfiable.  This
            # minimises ion displacement and therefore transport heating.
            #
            # Binary search: try pass_bound in [1, max_passes].
            # Lower pass bounds → fewer swap passes → less displacement.
            max_p = self.config.max_passes
            if self.config.base_pmax_in is not None:
                max_p = self.config.base_pmax_in
            
            best_model = None
            best_pass_bound = max_p
            best_encoder = None
            
            lo, hi = 1, max_p
            # Wall-clock limit for the entire binary search (Phase 1).
            # Each iteration gets config.timeout_seconds; total budget
            # is 3× that to allow for ~log2(max_p) iterations.
            _phase1_deadline = time.time() + self.config.timeout_seconds * 3
            while lo <= hi:
                if time.time() > _phase1_deadline:
                    wise_logger.info(
                        "Phase 1 wall-clock deadline reached after binary "
                        "search narrowed to [%d, %d]", lo, hi,
                    )
                    break
                mid = (lo + hi) // 2

                # Inner binary search over sum_bound_B (total active
                # passes across all optimised rounds).  This mirrors
                # the old code's Σ_r P_r minimisation.
                optimize_start = (
                    1 if context.ignore_initial_reconfig else 0
                )
                rounds_under_sum = max(
                    1, num_rounds - optimize_start
                )
                sb_lo, sb_hi = 1, rounds_under_sum * mid
                inner_best_enc = None
                inner_best_sb = sb_hi

                while sb_lo <= sb_hi:
                    if time.time() > _phase1_deadline:
                        break
                    sb_mid = (sb_lo + sb_hi) // 2
                    enc = WISESATEncoder(
                        rows=n_rows,
                        cols=n_cols,
                        config=self.config,
                        use_maxsat=False,  # Phase 1 uses pure SAT
                    )
                    enc.initialize(context, mid, sum_bound_B=sb_mid)
                    enc.add_all_constraints()

                    sol = enc.solve(
                        timeout=self.config.timeout_seconds
                    )
                    if sol.satisfiable:
                        inner_best_enc = enc
                        inner_best_sb = sb_mid
                        sb_hi = sb_mid - 1
                    else:
                        sb_lo = sb_mid + 1

                if inner_best_enc is not None:
                    # Feasible — try smaller pass bound
                    best_pass_bound = mid
                    best_encoder = inner_best_enc
                    hi = mid - 1
                    wise_logger.debug(
                        "D-minimization: pass_bound=%d SAT "
                        "(best sum_bound=%d)", mid, inner_best_sb,
                    )
                else:
                    # Infeasible even at max sum — need more passes
                    lo = mid + 1
                    wise_logger.debug(
                        "D-minimization: pass_bound=%d UNSAT", mid
                    )
            
            if best_encoder is None:
                wise_logger.info(
                    "SAT UNSAT at all pass bounds up to %d", max_p
                )
                return None
            
            # ---------------------------------------------------------------
            # Phase 2: MaxSAT boundary optimisation (if configured)
            # ---------------------------------------------------------------
            # With the minimum pass bound D* found, re-encode with MaxSAT
            # soft clauses that penalise ions landing on patch boundaries.
            # This reduces cross-boundary routing overhead in later rounds.
            use_maxsat = (
                self.config.use_maxsat
                and (self.config.boundary_soft_weight_row > 0
                     or self.config.boundary_soft_weight_col > 0)
            )
            
            if use_maxsat:
                wise_logger.debug(
                    "Phase 2: MaxSAT with D*=%d, boundary weights "
                    "row=%d col=%d",
                    best_pass_bound,
                    self.config.boundary_soft_weight_row,
                    self.config.boundary_soft_weight_col,
                )
                maxsat_enc = WISESATEncoder(
                    rows=n_rows,
                    cols=n_cols,
                    config=self.config,
                    use_maxsat=True,
                )
                maxsat_enc.initialize(context, best_pass_bound)
                maxsat_enc.add_all_constraints()
                # Add BT soft constraints when target positions exist.
                # These guide the MaxSAT solver toward planned layouts
                # from the previous window without inflating D*.
                if any(bt for bt in context.target_positions if bt):
                    maxsat_enc.add_target_position_constraints(use_soft=True)
                
                maxsat_sol = maxsat_enc.solve(
                    timeout=self.config.timeout_seconds
                )
                if maxsat_sol.satisfiable:
                    best_encoder = maxsat_enc
                    wise_logger.debug(
                        "MaxSAT improved: cost=%.1f", maxsat_sol.cost
                    )
                else:
                    wise_logger.debug(
                        "MaxSAT failed, using plain SAT solution"
                    )
            
            # ---------------------------------------------------------------
            # Phase 3: Decode solution into RoutingSchedule
            # ---------------------------------------------------------------
            # Use the model cached by solve() — avoids re-solving and
            # handles both CNF and WCNF formulas correctly.
            model = getattr(best_encoder, '_last_model', None)
            if model is not None:
                schedule_data = best_encoder.decode_schedule(model)
                
                # -----------------------------------------------------------
                # Decode ALL rounds from the multi-round SAT solution.
                # When lookahead is active, schedule_data contains passes
                # for ALL rounds [0..R-1].  By decoding every round we
                # allow the routing loop to advance batch_idx by the
                # full window size, so one SAT solve covers multiple
                # batches — matching the old code's behaviour where the
                # tiling_steps list carried per-round reconfigurations.
                # -----------------------------------------------------------
                all_rounds_passes: List[List[RoutingPass]] = []
                all_layouts: List[GridLayout] = []
                current_layout = initial_layout.copy()

                for round_idx in range(len(schedule_data)):
                    round_passes: List[RoutingPass] = []
                    for pass_info in schedule_data[round_idx]:
                        rp = RoutingPass(
                            phase=pass_info["phase"],
                            h_swaps=pass_info.get("h_swaps", []),
                            v_swaps=pass_info.get("v_swaps", []),
                        )
                        if rp.has_swaps:
                            round_passes.append(rp)
                    all_rounds_passes.append(round_passes)

                    # Apply this round's swaps to track layout evolution
                    for rp in round_passes:
                        if rp.phase == "H":
                            for r, c in rp.h_swaps:
                                current_layout.swap_horizontal(r, c)
                        else:
                            for r, c in rp.v_swaps:
                                current_layout.swap_vertical(r, c)
                    all_layouts.append(current_layout.copy())

                return RoutingSchedule(
                    passes_per_round=all_rounds_passes,
                    layouts=all_layouts,
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
        col_offset: int = 0,
    ) -> Tuple[List[List[Tuple[int, int]]], List[bool], List[int]]:
        """Compute gating block information.

        Blocks are aligned to the *global* gating grid.  When this
        sub-grid starts at ``col_offset`` (its first local column maps
        to global column ``col_offset``), the block boundaries are
        computed in global space and then mapped back to local
        coordinates.  This matches the old code's ``first_block_idx /
        last_block_idx`` logic.

        Parameters
        ----------
        n_rows : int
            Rows in the (sub-)grid.
        n_cols : int
            Columns in the (sub-)grid.
        capacity : int
            Block width (= ions_per_segment *k*).
        col_offset : int
            Global column index of this sub-grid's first column.

        Returns
        -------
        block_cells : List[List[Tuple[int, int]]]
            Cells in each block (local coordinates).
        block_fully_inside : List[bool]
            Whether each block is fully inside the grid.
        block_widths : List[int]
            Width of each block.
        """
        if capacity <= 0 or n_cols <= 0:
            return [], [], []

        block_cells: List[List[Tuple[int, int]]] = []
        block_fully_inside: List[bool] = []
        block_widths: List[int] = []

        global_patch_start = col_offset
        global_patch_end = col_offset + n_cols
        first_block_idx = col_offset // capacity
        last_block_idx = (col_offset + n_cols - 1) // capacity

        for b_global in range(first_block_idx, last_block_idx + 1):
            global_start = b_global * capacity
            global_end = (b_global + 1) * capacity
            local_start = max(0, global_start - col_offset)
            local_end = min(n_cols, global_end - col_offset)
            width = max(0, local_end - local_start)

            cells: List[Tuple[int, int]] = []
            for r in range(n_rows):
                for j_local in range(local_start, local_end):
                    cells.append((r, j_local))

            block_cells.append(cells)
            block_widths.append(width)
            block_fully_inside.append(
                (global_start >= global_patch_start)
                and (global_end <= global_patch_end)
            )

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
        
        Emits ``PASS_BOUNDARY`` sentinels between each parallel pass
        so downstream consumers (animation, execution planner) can
        group swaps atomically — all swaps within a single pass
        execute in parallel and must be applied simultaneously.
        """
        operations = []
        
        for round_passes in schedule.passes_per_round:
            for pass_info in round_passes:
                if pass_info.phase == "H":
                    # Split by column parity (matching old code's
                    # _infer_pass_parity for H-phase).
                    even_h = [(r, c) for r, c in pass_info.h_swaps
                              if c % 2 == 0]
                    odd_h  = [(r, c) for r, c in pass_info.h_swaps
                              if c % 2 == 1]
                    for parity_group in (even_h, odd_h):
                        if not parity_group:
                            continue
                        for r, c in parity_group:
                            operations.append({
                                "type": "H_SWAP",
                                "row": r,
                                "col": c,
                            })
                        operations.append({"type": "PASS_BOUNDARY"})
                else:
                    # Split by row parity (matching old code's
                    # _infer_pass_parity for V-phase).
                    even_v = [(r, c) for r, c in pass_info.v_swaps
                              if r % 2 == 0]
                    odd_v  = [(r, c) for r, c in pass_info.v_swaps
                              if r % 2 == 1]
                    for parity_group in (even_v, odd_v):
                        if not parity_group:
                            continue
                        for r, c in parity_group:
                            operations.append({
                                "type": "V_SWAP",
                                "row": r,
                                "col": c,
                            })
                        operations.append({"type": "PASS_BOUNDARY"})

        # Remove trailing boundary (not needed)
        if operations and operations[-1].get("type") == "PASS_BOUNDARY":
            operations.pop()
        
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
        lookahead_pairs: Optional[List[List[Tuple[int, int]]]] = None,
        bt_positions: Optional[List[Dict[int, Tuple[int, int]]]] = None,
    ) -> RoutingResult:
        """Route using patch decomposition with cycling.
        
        Implements a cycling algorithm similar to the old code's
        ``_patch_and_route()``:
        
        1. Partition grid into patches using ``subgridsize``
        2. Use checkerboard tiling to avoid patch conflicts
        3. Cycle through tilings until all pairs resolved or
           progress stalls (matching old code's stall detection)
        
        Parameters
        ----------
        gate_pairs : List[Tuple[int, int]]
            Pairs of physical qubits that need to interact.
        current_mapping : QubitMapping
            Current qubit positions.
        architecture : HardwareArchitecture
            Target architecture with grid dimensions.
        lookahead_pairs : Optional[List[List[Tuple[int, int]]]]
            Additional future batches for lookahead optimization.
        bt_positions : Optional[List[Dict[int, Tuple[int, int]]]]
            Boundary target positions from a previous solve.
            
        Returns
        -------
        RoutingResult
            Combined routing result from all patches.
        """
        if not gate_pairs:
            return RoutingResult(success=True, operations=[], cost=0.0)
        
        n_rows, n_cols = self._get_grid_dimensions(architecture)
        capacity = self._get_capacity(architecture)
        
        # Use subgridsize from config if available
        if self.config and self.config.subgridsize:
            patch_w, patch_h, patch_inc = self.config.subgridsize
        else:
            patch_h = self.config.patch_height if self.config else 4
            patch_w = self.config.patch_width if self.config else 4
            patch_inc = 1
        
        # Use max_cycles from config
        max_cycles = self.config.max_cycles if self.config else 10
        
        # --- k-alignment (Fix R) ---
        # Patch width must be k-compatible so block boundaries align
        # with the global gating grid.  Port of old code logic.
        if patch_w < capacity and (capacity % patch_w != 0):
            patch_w += int(capacity % patch_w)
        elif patch_w > capacity and (patch_w % capacity != 0):
            patch_w = (patch_w // capacity) * capacity
        patch_w = min(patch_w, n_cols)
        patch_h = min(patch_h, n_rows)
        
        # Check if patch routing is beneficial
        if n_rows <= patch_h and n_cols <= patch_w:
            # Grid is small enough for direct solving
            return super().route_batch(gate_pairs, current_mapping, architecture, lookahead_pairs, bt_positions=bt_positions)
        
        wise_logger.info(
            f"Patch routing: {n_rows}x{n_cols} grid with {len(gate_pairs)} pairs, "
            f"patch_size={patch_h}x{patch_w}, max_cycles={max_cycles}"
        )
        
        # Build initial layout from mapping
        initial_layout = self._build_initial_layout(current_mapping, n_rows, n_cols)
        
        # Track remaining pairs to solve
        remaining_pairs = set(gate_pairs)
        solved_pairs: Set[Tuple[int, int]] = set()
        
        # Flatten lookahead pairs for cross-boundary context.
        # These future pairs are NOT routed in this call, but they
        # influence cross-boundary preferences so the solver places
        # ions in positions that are good for future batches.
        #
        # IMPORTANT: Only use the FIRST lookahead batch (1 round ahead).
        # Including all lookahead batches is too aggressive — it
        # amortizes multiple rounds' work into a single solve, producing
        # unrealistically low exec_time.  Limiting to 1 batch matches
        # the old code's approach where multi-round SAT optimises for
        # current + next round only.
        lookahead_flat: List[Tuple[int, int]] = []
        if lookahead_pairs and len(lookahead_pairs) > 0:
            lookahead_flat = list(lookahead_pairs[0])
        
        # Route using cycling pattern (matching old _patch_and_route)
        all_operations: List[Any] = []
        total_cost = 0.0
        total_h_passes = 0
        total_v_passes = 0
        layout = initial_layout.copy()
        
        cycle_idx = 0
        no_progress_cycles = 0
        _patch_pbar = _tqdm(
            total=max_cycles,
            desc="Patch routing",
            unit="cycle",
            leave=False,
        )
        
        while remaining_pairs and cycle_idx < max_cycles:
            cycle_start_remaining = len(remaining_pairs)
            _patch_pbar.set_postfix_str(
                f"remaining={len(remaining_pairs)}/{len(gate_pairs)}, "
                f"solved={len(solved_pairs)}"
            )
            
            # Generate tilings (base, vertical offset, horizontal offset)
            tilings = self._generate_tilings(
                patch_h, patch_w, n_rows, n_cols, capacity,
            )
            
            wise_logger.debug(
                f"Cycle {cycle_idx + 1}: {len(remaining_pairs)} pairs remaining, "
                f"{len(tilings)} tilings, patch={patch_h}x{patch_w}"
            )
            
            for tiling_idx, (off_r, off_c) in enumerate(tilings):
                if not remaining_pairs:
                    break
                
                # Generate NON-overlapping patches (Fix U) — matching old
                # _generate_patch_regions which tiles without overlap.
                offset_patches = self._generate_non_overlapping_patches(
                    n_rows, n_cols, patch_h, patch_w,
                    offset_r=off_r, offset_c=off_c,
                )
                
                # Assign remaining pairs to patches (both-in-patch only)
                pair_assignments = self._assign_pairs_to_patches(
                    list(remaining_pairs), layout, offset_patches
                )
                
                # Compute the set of unassigned pairs for cross-boundary
                # preference computation.
                assigned_pairs_set: Set[Tuple[int, int]] = set()
                for pairs_list in pair_assignments.values():
                    assigned_pairs_set.update(pairs_list)
                unassigned_pairs = list(remaining_pairs - assigned_pairs_set)
                
                # Route all patches (non-overlapping, so no conflict).
                for patch_idx, patch in enumerate(offset_patches):
                    patch_pairs = pair_assignments.get(patch_idx, [])
                    if not patch_pairs:
                        continue
                    
                    # Build remaining pairs for this patch = unassigned +
                    # pairs assigned to OTHER patches + lookahead pairs.
                    # Including lookahead_flat gives the sub-solver
                    # cross-boundary awareness of future-batch pairs so
                    # it positions ions favourably for upcoming rounds.
                    other_assigned = [
                        p for pidx, plist in pair_assignments.items()
                        if pidx != patch_idx for p in plist
                    ]
                    remaining_for_patch = (
                        unassigned_pairs + other_assigned + lookahead_flat
                    )
                    
                    result = self._route_patch(
                        patch, patch_pairs, layout, architecture,
                        remaining_pairs=remaining_for_patch,
                        bt_positions=bt_positions,
                        lookahead_pairs=lookahead_pairs,
                    )
                    if result.success:
                        all_operations.extend(result.operations)
                        total_cost += result.cost
                        self._apply_patch_result(layout, result, patch)
                        # Accumulate per-pass counts from sub-patch
                        if result.metrics:
                            total_h_passes += result.metrics.get("h_passes", 0)
                            total_v_passes += result.metrics.get("v_passes", 0)
                        
                        # Mark pairs as solved
                        for pair in patch_pairs:
                            if self._is_pair_satisfied(pair, layout, capacity):
                                remaining_pairs.discard(pair)
                                solved_pairs.add(pair)
            
            # Check progress for stall detection
            cycle_end_remaining = len(remaining_pairs)
            if cycle_end_remaining >= cycle_start_remaining:
                no_progress_cycles += 1
                wise_logger.debug(
                    f"Cycle {cycle_idx + 1}: no progress, "
                    f"stall count={no_progress_cycles}"
                )
                fully_global = (patch_h >= n_rows) and (patch_w >= n_cols)
                if no_progress_cycles >= 2 or fully_global:
                    wise_logger.warning(
                        f"Patch routing stalled after {cycle_idx + 1} cycles "
                        f"with {cycle_end_remaining} pairs remaining"
                    )
                    break
            else:
                no_progress_cycles = 0
            
            cycle_idx += 1
            _patch_pbar.update(1)
            
            # --- Patch size growth (Fix T) ---
            # After each cycle, expand patch dimensions so that
            # cross-boundary pairs can fit in a single patch.
            patch_w += max(patch_inc, min(patch_w, capacity))
            patch_h += patch_inc
            # Re-apply k-alignment after growth
            if patch_w < capacity and (capacity % patch_w != 0):
                patch_w += int(capacity % patch_w)
            elif patch_w > capacity and (patch_w % capacity != 0):
                patch_w = (patch_w // capacity) * capacity
            patch_w = min(patch_w, n_cols)
            patch_h = min(patch_h, n_rows)
        
        _patch_pbar.close()

        # --- Fallback: direct SAT solve for remaining pairs (Fix U2) ---
        # When the patch loop stalls at full-grid size with unsolved
        # pairs, try a single direct SAT solve using the current
        # (post-patch) layout.  This recovers from scenarios where
        # intermediate cycles scrambled the layout and the stall
        # detector gave up prematurely.
        if remaining_pairs:
            wise_logger.info(
                f"Patch routing completed {cycle_idx} cycles, "
                f"{len(remaining_pairs)}/{len(gate_pairs)} pairs unsolved — "
                f"attempting direct SAT fallback"
            )
            # Convert current layout to a mapping for direct SAT
            fallback_mapping = self._layout_to_mapping(layout, current_mapping)
            try:
                fallback_result = WiseSatRouter.route_batch(
                    self,
                    list(remaining_pairs),
                    fallback_mapping,
                    architecture,
                    lookahead_pairs=lookahead_pairs,
                )
                if fallback_result.success and fallback_result.operations:
                    all_operations.extend(fallback_result.operations)
                    total_cost += fallback_result.cost
                    if fallback_result.metrics:
                        total_h_passes += fallback_result.metrics.get("h_passes", 0)
                        total_v_passes += fallback_result.metrics.get("v_passes", 0)
                    # Update layout from fallback result
                    if fallback_result.final_mapping:
                        layout = self._build_initial_layout(
                            fallback_result.final_mapping, n_rows, n_cols
                        )
                    # Mark all remaining pairs as solved
                    solved_pairs.update(remaining_pairs)
                    remaining_pairs.clear()
                    wise_logger.info("Direct SAT fallback solved all remaining pairs")
                elif fallback_result.success:
                    # Success with no ops (pairs already satisfied)
                    solved_pairs.update(remaining_pairs)
                    remaining_pairs.clear()
                    wise_logger.info("Direct SAT fallback: pairs already satisfied")
                else:
                    wise_logger.warning(
                        f"Direct SAT fallback failed for "
                        f"{len(remaining_pairs)} remaining pairs"
                    )
            except Exception as exc:
                wise_logger.warning(f"Direct SAT fallback error: {exc}")
        else:
            wise_logger.info(
                f"Patch routing converged in {cycle_idx} cycles"
            )
        
        # Build final mapping
        final_mapping = self._layout_to_mapping(layout, current_mapping)
        
        # --- Generate BTs for the next round ---
        # When lookahead is active, downstream WISERoutingPass populates
        # _bt_cache from _future_bt_positions.  If the BT bootstrap
        # probe succeeded, propagate those BTs (they come from a
        # multi-round full-grid solve and contain actual future-round
        # layouts).  Otherwise, use the final layout as a fallback BT
        # (weaker signal, but non-empty so subsequent batches use
        # single-round SAT with soft BT constraints).
        future_bt_positions: List[Dict[int, Tuple[int, int]]] = []
        if bt_positions and any(bt for bt in bt_positions if bt):
            # Propagate BTs from bootstrap probe (shift by 1 round)
            future_bt_positions = list(bt_positions)
        elif lookahead_pairs:
            # Fallback: use final layout as BT target
            future_bt_positions.append(dict(layout.ion_positions))
        
        return RoutingResult(
            success=len(remaining_pairs) == 0,
            operations=all_operations,
            cost=total_cost,
            final_mapping=final_mapping,
            metrics={
                "num_cycles": cycle_idx,
                "pairs_solved": len(solved_pairs),
                "pairs_remaining": len(remaining_pairs),
                "total_operations": len(all_operations),
                "h_passes": total_h_passes,
                "v_passes": total_v_passes,
                "_future_bt_positions": future_bt_positions,
            },
        )
    
    def _generate_tilings(
        self,
        patch_h: int,
        patch_w: int,
        n_rows: int,
        n_cols: int,
        capacity: int = 2,
    ) -> List[Tuple[int, int]]:
        """Generate tiling offsets for checkerboard routing.
        
        Returns list of (row_offset, col_offset) tuples representing
        different tiling phases. Matches old code's tiling strategy:
        - (0, 0): Base tiling
        - (half_h, 0): Vertical offset
        - (0, half_w): Horizontal offset (if k-compatible)
        
        Parameters
        ----------
        capacity : int
            Block width (*k*) for k-compatibility checking.
        
        Returns
        -------
        List[Tuple[int, int]]
            List of tiling offsets.
        """
        tilings: List[Tuple[int, int]] = [(0, 0)]
        
        # Vertical offset (half patch height)
        half_h = patch_h // 2
        if half_h > 0 and half_h < n_rows:
            tilings.append((half_h, 0))
        
        # Horizontal offset (half patch width) if even AND k-compatible
        # (Fix S: port old code's _k_compatible check).
        if patch_w % 2 == 0:
            half_w = patch_w // 2
            if (
                half_w > 0
                and half_w < n_cols
                and self._k_compatible(half_w, capacity)
            ):
                tilings.append((0, half_w))
        
        return tilings
    
    @staticmethod
    def _k_compatible(width: int, k: int) -> bool:
        """Check if a width is k-compatible for WISE block alignment.
        
        Ported from old code's _k_compatible helper.
        """
        if width <= 0 or width == 1:
            return False
        if width == k:
            return True
        if width > k:
            return (width % k) == 0
        # width < k
        return (k % width) == 0
    
    def _generate_non_overlapping_patches(
        self,
        n_rows: int,
        n_cols: int,
        patch_h: int,
        patch_w: int,
        offset_r: int = 0,
        offset_c: int = 0,
    ) -> List[Tuple[int, int, int, int]]:
        """Generate non-overlapping patch regions (matching old code).
        
        Tiles the grid starting at ``(offset_r, offset_c)`` with patches
        of size ``patch_h × patch_w``, clipped to grid bounds.
        
        Parameters
        ----------
        n_rows, n_cols : int
            Grid dimensions.
        patch_h, patch_w : int
            Patch dimensions.
        offset_r, offset_c : int
            Starting offset for the tiling.
        
        Returns
        -------
        List[Tuple[int, int, int, int]]
            ``(r0, c0, r1, c1)`` bounding boxes.
        """
        if patch_h <= 0 or patch_w <= 0:
            return [(0, 0, n_rows, n_cols)]

        regions: List[Tuple[int, int, int, int]] = []
        start_row = min(max(offset_r, 0), n_rows - 1) if n_rows > 0 else 0
        start_col = min(max(offset_c, 0), n_cols - 1) if n_cols > 0 else 0

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
    
    @staticmethod
    def _compute_patch_gating_capacity(
        n: int,
        m: int,
        col_offset: int,
        capacity: int,
    ) -> int:
        """Compute the max number of disjoint gating zones in a patch.
        
        Ported from old code's ``_compute_patch_gating_capacity``.
        
        Parameters
        ----------
        n : int
            Patch height (rows).
        m : int
            Patch width (columns, local).
        col_offset : int
            Global column of patch's first column.
        capacity : int
            Global block width.
        
        Returns
        -------
        int
            Number of gating zones available.
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
            total_per_row += (width // 2)
        return n * total_per_row
    
    def _is_pair_satisfied(
        self,
        pair: Tuple[int, int],
        layout: GridLayout,
        capacity: int = 2,
    ) -> bool:
        """Check if a gate pair is satisfied (both in same gating block).
        
        The SAT encoder's gate-pair constraint requires both ions to be
        in the **same row** and **same gating block** (a contiguous
        group of *capacity* columns).  This must match: the old
        ``col_diff == 1`` check was correct only for k=2 where a block
        has exactly 2 columns; for k≥4 the block is wider and the pair
        is satisfied as long as both ions share the same block.
        
        Parameters
        ----------
        pair : Tuple[int, int]
            Ion pair to check.
        layout : GridLayout
            Current layout.
        capacity : int
            Gating block width (ions_per_segment *k*).
            
        Returns
        -------
        bool
            True if both ions are in the same row and same gating block.
        """
        ion_a, ion_b = pair
        pos_a = layout.get_position(ion_a)
        pos_b = layout.get_position(ion_b)
        
        if pos_a is None or pos_b is None:
            return False
        
        # Must be in same row
        if pos_a[0] != pos_b[0]:
            return False
        
        # Must be in same gating block (k-aligned)
        block_a = pos_a[1] // capacity
        block_b = pos_b[1] // capacity
        return block_a == block_b
    
    def _generate_overlapping_patches(
        self,
        n_rows: int,
        n_cols: int,
        patch_h: int,
        patch_w: int,
        overlap: int,
        offset_r: int = 0,
        offset_c: int = 0,
    ) -> List[Tuple[int, int, int, int]]:
        """Generate patches with overlap for boundary handling.
        
        Parameters
        ----------
        n_rows, n_cols : int
            Grid dimensions.
        patch_h, patch_w : int
            Patch dimensions.
        overlap : int
            Overlap between adjacent patches.
        offset_r, offset_c : int
            Tiling offset (for checkerboard cycling).
        
        Returns
        -------
        List[Tuple[int, int, int, int]]
            List of (r0, c0, r1, c1) patch boundaries.
        """
        patches = []
        stride_h = max(patch_h - overlap, 1)
        stride_w = max(patch_w - overlap, 1)
        
        # Start at offset, wrapping around
        r0 = offset_r
        while r0 < n_rows:
            r1 = min(r0 + patch_h, n_rows)
            c0 = offset_c
            while c0 < n_cols:
                c1 = min(c0 + patch_w, n_cols)
                patches.append((r0, c0, r1, c1))
                c0 += stride_w
                if c0 >= n_cols and c1 < n_cols:
                    c0 = n_cols - patch_w
            r0 += stride_h
            if r0 >= n_rows and r1 < n_rows:
                r0 = n_rows - patch_h
        
        # Also add patches starting from 0 to cover offset gap
        if offset_r > 0:
            for c0 in range(0, n_cols, stride_w):
                c1 = min(c0 + patch_w, n_cols)
                patches.append((0, c0, min(offset_r + patch_h, n_rows), c1))
        
        if offset_c > 0:
            for r0 in range(0, n_rows, stride_h):
                r1 = min(r0 + patch_h, n_rows)
                patches.append((r0, 0, r1, min(offset_c + patch_w, n_cols)))
        
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
            # Pairs spanning multiple patches are left unassigned —
            # they will be retried in a later tiling cycle with a
            # different offset or larger patch size (matching old code's
            # _split_pairs_for_patch which only assigns when BOTH ions
            # are in the patch).
        
        return assignments
    
    def _route_patch(
        self,
        patch: Tuple[int, int, int, int],
        pairs: List[Tuple[int, int]],
        layout: GridLayout,
        architecture: "HardwareArchitecture",
        remaining_pairs: Optional[List[Tuple[int, int]]] = None,
        bt_positions: Optional[List[Dict[int, Tuple[int, int]]]] = None,
        lookahead_pairs: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> RoutingResult:
        """Route within a single patch using base SAT router.

        Parameters
        ----------
        patch : Tuple[int, int, int, int]
            ``(r0, c0, r1, c1)`` bounding box of the patch.
        pairs : List[Tuple[int, int]]
            Ion pairs assigned to this patch (both ions inside).
        layout : GridLayout
            Current global layout.
        architecture : HardwareArchitecture
            Full grid architecture (for capacity info).
        remaining_pairs : Optional[List[Tuple[int, int]]]
            All pairs NOT assigned to this patch.  Used to compute
            cross-boundary preferences so the solver nudges ions
            toward boundaries where their cross-patch partners live.
        bt_positions : Optional[List[Dict[int, Tuple[int, int]]]]
            Boundary target positions (filtered to this patch).
        lookahead_pairs : Optional[List[List[Tuple[int, int]]]]
            Future batches' gate pairs.  Pairs where both ions are in
            this patch are passed to the sub-solver for multi-round
            optimisation so the resulting layout is future-aware.
        """
        r0, c0, r1, c1 = patch
        patch_h = r1 - r0
        patch_w = c1 - c0
        n_rows_global = layout.n_rows
        n_cols_global = layout.n_cols

        # Extract sub-layout
        sub_grid = layout.grid[r0:r1, c0:c1].copy()
        sub_layout = GridLayout(grid=sub_grid)

        # Translate pairs to local indices (only pairs with BOTH ions inside)
        local_pairs = []
        for ion_a, ion_b in pairs:
            pos_a = layout.get_position(ion_a)
            pos_b = layout.get_position(ion_b)
            if pos_a is None or pos_b is None:
                continue
            if (r0 <= pos_a[0] < r1 and c0 <= pos_a[1] < c1 and
                r0 <= pos_b[0] < r1 and c0 <= pos_b[1] < c1):
                local_pairs.append((ion_a, ion_b))

        if not local_pairs:
            return RoutingResult(success=True, operations=[], cost=0.0)

        # --- Compute boundary_adjacent (N3 fix) ---
        boundary_adjacent = {
            "top": r0 > 0,
            "bottom": r1 < n_rows_global,
            "left": c0 > 0,
            "right": c1 < n_cols_global,
        }

        # --- Compute cross-boundary prefs (N2 fix) ---
        # For each ion inside this patch that has a partner OUTSIDE,
        # record which boundary direction(s) the outside partner is in.
        cross_boundary_prefs: List[Dict[int, Set[str]]] = [{}]
        if remaining_pairs:
            pref_round: Dict[int, Set[str]] = {}
            for ion_a, ion_b in remaining_pairs:
                pos_a = layout.get_position(ion_a)
                pos_b = layout.get_position(ion_b)
                if pos_a is None or pos_b is None:
                    continue
                in_a = (r0 <= pos_a[0] < r1 and c0 <= pos_a[1] < c1)
                in_b = (r0 <= pos_b[0] < r1 and c0 <= pos_b[1] < c1)
                if in_a == in_b:
                    continue
                inside_ion = ion_a if in_a else ion_b
                outside_pos = pos_b if in_a else pos_a
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
                if dirs:
                    pref_round.setdefault(inside_ion, set()).update(dirs)
            if pref_round:
                cross_boundary_prefs = [pref_round]

        has_boundary_prefs = bool(
            cross_boundary_prefs
            and any(pr for pr in cross_boundary_prefs)
        )

        # --- Gating capacity check (port from old code) ---
        capacity = getattr(architecture, 'capacity', 2)
        max_gating_zones = self._compute_patch_gating_capacity(
            patch_h, patch_w, c0, capacity,
        )
        if max_gating_zones > 0 and len(local_pairs) > max_gating_zones:
            # Spill excess pairs — they'll be retried next cycle.
            local_pairs = local_pairs[:max_gating_zones]

        if not local_pairs and not has_boundary_prefs:
            return RoutingResult(success=True, operations=[], cost=0.0)

        # Create mock architecture for sub-grid
        class SubArchitecture:
            def __init__(self, rows, cols, cap):
                self.n_rows = rows
                self.n_cols = cols
                self.num_qubits = rows * cols
                self.capacity = cap
                self.grid_shape = (rows, cols)

        sub_arch = SubArchitecture(patch_h, patch_w, capacity)

        # Build sub-mapping
        sub_mapping = QubitMapping()
        for ion_idx, (row, col) in sub_layout.ion_positions.items():
            sub_mapping.add_mapping(ion_idx, ion_idx, f"trap_{row}_{col}")

        # Route using base SAT solver.
        # Propagate patch origin so _compute_blocks aligns to global grid.
        self._current_grid_origin = (r0, c0)
        # Pass cross-boundary context for this patch.
        self._patch_cross_boundary_prefs = cross_boundary_prefs
        self._patch_boundary_adjacent = boundary_adjacent
        # Zero boundary soft weights for patch routing (AA fix) —
        # the old code explicitly passes wB_col=0, wB_row=0.
        saved_w_row = self.config.boundary_soft_weight_row
        saved_w_col = self.config.boundary_soft_weight_col
        self.config.boundary_soft_weight_row = 0
        self.config.boundary_soft_weight_col = 0

        # --- Filter BTs to this patch ---
        # NOTE: We intentionally do NOT pass global BTs to the patch
        # sub-solver. Global BTs (from WISERoutingPass._bt_cache) are
        # designed for full-grid routing. When filtered to a small patch,
        # most entries are lost and the few survivors make _solve_sat
        # switch to single-round mode, which SUPPRESSES multi-round
        # lookahead encoding. The old code always routes all rounds'
        # pairs per patch (multi-round) regardless of BT state.
        # By omitting BTs here, we ensure the patch solver uses
        # multi-round SAT when lookahead_pairs are available.

        # --- Filter lookahead_pairs to this patch (both ions inside) ---
        # This gives the sub-solver multi-round context so it produces
        # a layout that is future-aware, matching the old code's
        # approach where all rounds' pairs are passed to the solver.
        local_lookahead: Optional[List[List[Tuple[int, int]]]] = None
        if lookahead_pairs:
            filtered: List[List[Tuple[int, int]]] = []
            for batch in lookahead_pairs:
                local_batch: List[Tuple[int, int]] = []
                for ion_a, ion_b in batch:
                    pos_a = layout.get_position(ion_a)
                    pos_b = layout.get_position(ion_b)
                    if pos_a is None or pos_b is None:
                        continue
                    if (r0 <= pos_a[0] < r1 and c0 <= pos_a[1] < c1 and
                            r0 <= pos_b[0] < r1 and c0 <= pos_b[1] < c1):
                        local_batch.append((ion_a, ion_b))
                if local_batch:
                    filtered.append(local_batch)
            if filtered:
                local_lookahead = filtered

        try:
            result = super().route_batch(
                local_pairs, sub_mapping, sub_arch,
                lookahead_pairs=local_lookahead,
            )
        finally:
            self._current_grid_origin = (0, 0)
            self._patch_cross_boundary_prefs = None
            self._patch_boundary_adjacent = None
            self.config.boundary_soft_weight_row = saved_w_row
            self.config.boundary_soft_weight_col = saved_w_col
        
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
        architecture: Optional["HardwareArchitecture"] = None,
        lookahead: int = 0,
    ):
        """Initialize the routing pass.
        
        Parameters
        ----------
        config : Optional[WISERoutingConfig]
            Configuration for the SAT solver.
        use_patch_routing : bool
            If True, use patch-based routing for large grids.
        architecture : Optional[HardwareArchitecture]
            Default architecture to use if not passed to route().
        lookahead : int
            Number of future gate batches to include in each routing
            window.  When > 0 the router solves for the current batch
            plus the next ``lookahead`` batches simultaneously, then
            chains the output layout to the next window.  This mirrors
            the old code's multi-round optimisation.
        """
        self.config = config or WISERoutingConfig()
        self.use_patch_routing = use_patch_routing or self.config.patch_enabled
        self.architecture = architecture
        self.lookahead = lookahead
        
        # Create the appropriate router
        if self.use_patch_routing:
            self.router = WisePatchRouter(config=self.config)
        else:
            self.router = WiseSatRouter(config=self.config)
    
    def route(
        self,
        mapped_circuit: "MappedCircuit",
        architecture: Optional["HardwareArchitecture"] = None,
    ) -> "RoutedCircuit":
        """Route a mapped circuit.
        
        Parameters
        ----------
        mapped_circuit : MappedCircuit
            Circuit with logical-to-physical mapping.
        architecture : Optional[HardwareArchitecture]
            Target WISE grid architecture.  Falls back to the
            architecture passed at construction time.
            
        Returns
        -------
        RoutedCircuit
            Circuit with routing operations inserted.
        """
        from qectostim.experiments.hardware_simulation.core.pipeline import (
            MappedCircuit,
            RoutedCircuit,
        )
        
        architecture = architecture or self.architecture
        if architecture is None:
            raise ValueError(
                "WISERoutingPass.route() requires an architecture. "
                "Pass it to __init__ or route()."
            )
        
        # Extract gate pairs from circuit
        gate_batches = self._extract_gate_batches(mapped_circuit)
        
        if not gate_batches:
            # No two-qubit gates, return empty routed circuit
            return RoutedCircuit(
                operations=[],
                final_mapping=mapped_circuit.mapping.copy(),
                routing_overhead=0,
                mapped_circuit=mapped_circuit,
            )
        
        # Route each batch and collect operations
        all_operations: List[Any] = []
        current_mapping = mapped_circuit.mapping.copy()
        total_routing_ops = 0
        total_reconfig_time = 0.0

        # --- Motional quanta accumulator ---
        # Tracks per-ion heating from reconfiguration swaps.
        # We take a SNAPSHOT after each batch so downstream code can
        # look up the n̄ state *at the time* each gate executes.

        motional_quanta: Dict[int, float] = {}
        for q in range(architecture.num_qubits):
            motional_quanta[q] = 0.0

        # Per-batch snapshots: motional_quanta_per_batch[b] is the
        # state of motional_quanta AFTER batch b's routing swaps
        # but BEFORE batch b's gates execute.  This is the n̄ that
        # each gate in batch b should use for its fidelity formula.
        motional_quanta_per_batch: List[Dict[int, float]] = []

        # --- Mode snapshot accumulator (3N normal-mode tracking) ---
        # Parallel to the scalar motional_quanta, this tracks the full
        # normal-mode structure (frequencies, eigenvectors, per-mode
        # occupancies) for each qubit's trap context.  The collaborator's
        # noise model will consume these to compute mode-resolved gate
        # infidelities using Lamb-Dicke parameters.
        #
        # Each entry maps qubit_idx → ModeSnapshot (the state of the
        # trap that qubit resides in at snapshot time).  We import
        # ModeSnapshot lazily to avoid circular imports.
        from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
            ModeSnapshot,
        )
        mode_snapshots_per_batch: List[Dict[int, Any]] = []

        # Maps (q1, q2) gate pairs to the batch index they belong to,
        # so the execution planner can look up which batch snapshot
        # applies to each gate instruction.
        gate_batch_map: Dict[Tuple[int, int], int] = {}

        # Determine effective lookahead from config or __init__ param
        lookahead = self.lookahead
        if lookahead == 0 and self.config and self.config.lookahead_rounds > 0:
            lookahead = self.config.lookahead_rounds

        # BT (boundary target) cache: carries forward future-round
        # target layouts from one SAT solve to constrain the next.
        # Matches old code's BTs mechanism in ionRoutingWISEArch.
        _bt_cache: List[Dict[int, Tuple[int, int]]] = []
        
        # Wall-clock deadline: abort routing if total time exceeds limit.
        # Uses timeout_seconds * num_batches with a generous floor of 120 s.
        _per_batch_timeout = (
            self.config.timeout_seconds if self.config else 60.0
        )
        _wall_limit = max(
            120.0,
            _per_batch_timeout * len(gate_batches) * 2,
        )
        _wall_start = time.monotonic()
        
        batch_idx = 0
        _pbar = _tqdm(
            total=len(gate_batches),
            desc="WISE routing",
            unit="batch",
            leave=True,
        )
        while batch_idx < len(gate_batches):
            # Check wall-clock deadline
            if time.monotonic() - _wall_start > _wall_limit:
                wise_logger.error(
                    "WISE routing wall-clock timeout (%.0f s) after %d/%d batches",
                    _wall_limit, batch_idx, len(gate_batches),
                )
                break
            
            _pbar.set_postfix_str(
                f"pairs={len(gate_batches[batch_idx])}, "
                f"ops={total_routing_ops}"
            )
            # ---------------------------------------------------------------
            # Lookahead aggregation: combine current batch + lookahead batches
            # ---------------------------------------------------------------
            # When lookahead > 0, solve for multiple batches simultaneously.
            # This matches old code's ionRoutingWISEArch() which passes
            # P_arr (a list of per-round gate pairs) to the solver.
            window_end = min(batch_idx + 1 + lookahead, len(gate_batches))
            aggregated_pairs: List[List[Tuple[int, int]]] = []
            for i in range(batch_idx, window_end):
                aggregated_pairs.append(gate_batches[i])
            
            # Flatten for single-round routing, or pass as multi-round
            # For now, flatten — multi-round SAT encoding is in WiseSatRouter
            gate_pairs = gate_batches[batch_idx]
            all_window_pairs = [p for batch in aggregated_pairs for p in batch]
            
            wise_logger.debug(
                f"Routing batch {batch_idx} with {len(gate_pairs)} pairs "
                f"(lookahead window: {len(aggregated_pairs)} batches, "
                f"{len(all_window_pairs)} total pairs)"
            )
            
            # Route using aggregated window context
            # The router can use all_window_pairs for global optimization
            # but we only apply swaps needed for gate_pairs
            lookahead_batch_pairs = aggregated_pairs[1:] if len(aggregated_pairs) > 1 else None
            result = self.router.route_batch(
                gate_pairs,
                current_mapping,
                architecture,
                lookahead_pairs=lookahead_batch_pairs,
                bt_positions=_bt_cache if _bt_cache else None,
            )
            
            if not result.success:
                wise_logger.warning(
                    f"Routing failed for batch {batch_idx}: {result.metrics}"
                )
                # Skip this batch and advance — retrying the same batch
                # infinitely was a bug (continue without incrementing).
                batch_idx += 1
                _pbar.update(1)
                continue
            
            # Accumulate reconfiguration heating from routing swaps.
            # Count actual H vs V swaps from the operations returned by
            # the SAT router — each op dict has type="H_SWAP" or "V_SWAP".
            if result.operations:
                row_swaps = sum(
                    1 for op in result.operations
                    if isinstance(op, dict) and op.get("type") == "H_SWAP"
                )
                col_swaps = sum(
                    1 for op in result.operations
                    if isinstance(op, dict) and op.get("type") == "V_SWAP"
                )
            else:
                total_swaps = result.metrics.get("total_swaps", 0) if result.metrics else 0
                row_swaps = total_swaps // 2
                col_swaps = total_swaps - row_swaps

            if row_swaps + col_swaps > 0:
                heating_per_ion = (
                    row_swaps * ROW_SWAP_HEATING + col_swaps * COL_SWAP_HEATING
                )

                # Per-pass reconfiguration time (matching old code's
                # _runOddEvenReconfig model): one parallel H-pass adds
                # one H_PASS_TIME, one V-pass adds one V_PASS_TIME,
                # plus initial split overhead.  Pass counts come from
                # the SAT schedule when available; fall back to the
                # coarser (row_swaps > 0) heuristic otherwise.
                h_passes = (result.metrics or {}).get("h_passes", None)
                v_passes = (result.metrics or {}).get("v_passes", None)
                if h_passes is None or v_passes is None:
                    # Fallback: at least 1 pass per phase that has swaps
                    h_passes = 1 if row_swaps > 0 else 0
                    v_passes = 1 if col_swaps > 0 else 0
                reconfig_time = (
                    INITIAL_SPLIT_TIME_US
                    + h_passes * H_PASS_TIME_US
                    + v_passes * V_PASS_TIME_US
                )
                total_reconfig_time += reconfig_time

                # Distribute heating to ALL ions (every ion in the trap
                # is affected by the crystal reconfiguration, matching
                # the old code's per-trap motionalMode accumulation).
                for ion_idx in motional_quanta:
                    motional_quanta[ion_idx] += heating_per_ion

            # ------ Snapshot n̄ AFTER routing swaps, BEFORE gates ------
            # This captures the system state at the time gates execute.
            motional_quanta_per_batch.append(dict(motional_quanta))

            # ------ Mode snapshot: capture 3N normal-mode state ------
            #
            # Snapshot the full vibrational state of every trap AFTER
            # transport/routing swaps but BEFORE gates execute.
            # Delegates to collect_mode_snapshots() in physics.py.
            from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
                collect_mode_snapshots as _collect_snaps,
            )
            batch_mode_snapshot = _collect_snaps(architecture, motional_quanta)
            mode_snapshots_per_batch.append(batch_mode_snapshot)

            # Add routing operations
            if result.operations:
                # Separate this batch's transports from the previous
                # batch's gate ops with a PASS_BOUNDARY sentinel so
                # the animation groups each parallel-swap set atomically.
                if all_operations:
                    from qectostim.experiments.hardware_simulation.core.operations import (
                        TransportOperation as _TO,
                    )
                    all_operations.append(_TO(
                        qubit=-1,
                        source_zone="__PASS_BOUNDARY__",
                        target_zone="__PASS_BOUNDARY__",
                        duration=0.0,
                    ))
                all_operations.extend(
                    self._convert_to_physical_ops(
                        result.operations, architecture, current_mapping
                    )
                )
                total_routing_ops += len(result.operations)
            
            # ---------------------------------------------------------------
            # Extract BTs (boundary targets) from multi-round solution.
            # Future-round layouts inform the NEXT iteration's SAT solve,
            # reducing the search space and improving solution quality.
            # This matches the old code's BT mechanism where future-round
            # target positions are pinned via hard/soft SAT constraints.
            # ---------------------------------------------------------------
            _future_bts = (result.metrics or {}).get(
                "_future_bt_positions", []
            )
            if _future_bts:
                _bt_cache = _future_bts
            else:
                _bt_cache = []

            # Update mapping for next batch (round-0 layout only)
            # IMPORTANT: this MUST happen BEFORE the co-location check
            # so that we verify positions from the POST-routing layout.
            if result.final_mapping:
                current_mapping = result.final_mapping

            # ------ Emit gates for co-located pairs (post-routing) ------
            # Uses the UPDATED mapping (after routing) so positions
            # reflect the actual post-routing layout.
            # Gate qubits are emitted as PHYSICAL IDs so the animation
            # (which keys positions by physical ID) renders beams on
            # the correct ions.
            from qectostim.experiments.hardware_simulation.core.operations import (
                GateOperation,
            )
            from qectostim.experiments.hardware_simulation.core.gates import (
                GateSpec,
                GateType,
            )
            ms_spec = GateSpec(
                name="MS",
                gate_type=GateType.TWO_QUBIT,
                num_qubits=2,
                is_native=True,
            )

            _post_map = current_mapping   # now reflects post-routing
            _n_k = getattr(architecture, "ions_per_segment", 1)
            _n_m = getattr(architecture, "col_groups", 1)
            _n_tc = _n_m * _n_k
            _post_pos: Dict[int, Tuple[int, int]] = {}
            for _lq, _pq in _post_map.logical_to_physical.items():
                _zn = _post_map.zone_assignments.get(_pq)
                if _zn:
                    try:
                        _pts = _zn.split("_")
                        _pr, _pc = int(_pts[-2]), int(_pts[-1])
                        _post_pos[_pq] = (_pr, _pc)
                    except (ValueError, IndexError):
                        pass
                if _pq not in _post_pos:
                    _post_pos[_pq] = (_pq // _n_tc if _n_tc else 0,
                                      _pq % _n_tc if _n_tc else _pq)

            for logical_q1, logical_q2 in gate_pairs:
                p1 = _post_map.get_physical(logical_q1)
                p2 = _post_map.get_physical(logical_q2)
                pos1 = _post_pos.get(p1) if p1 is not None else None
                pos2 = _post_pos.get(p2) if p2 is not None else None
                if pos1 is not None and pos2 is not None:
                    same_row = (pos1[0] == pos2[0])
                    # Both ions must be in the same gating block:
                    # block = col // k.  Within a block cols are adjacent.
                    same_block = (pos1[1] // _n_k == pos2[1] // _n_k)
                    if not (same_row and same_block):
                        wise_logger.debug(
                            "Gate pair (%d,%d) NOT co-located after "
                            "routing: pos=(%s,%s) — skipping gate",
                            logical_q1, logical_q2, pos1, pos2,
                        )
                        continue

                # Emit with PHYSICAL qubit IDs so the animation can
                # look up the correct (x,y) positions.
                gate_op = GateOperation(
                    gate=ms_spec,
                    qubits=(p1, p2),
                    duration=100.0,
                    base_fidelity=0.99,
                    metadata={
                        "batch": batch_idx,
                        "logical_qubits": (logical_q1, logical_q2),
                    },
                )
                all_operations.append(gate_op)

                gate_batch_map[(logical_q1, logical_q2)] = batch_idx
            
            # Advance to next batch (one round per iteration,
            # matching old code's idx += 1 with BT carry-forward)
            batch_idx += 1
            _pbar.update(1)
        
        _pbar.close()

        # Build metadata with heating and reconfiguration info
        routing_metadata: Dict[str, Any] = {
            "motional_quanta": motional_quanta,
            "motional_quanta_per_batch": motional_quanta_per_batch,
            "mode_snapshots_per_batch": mode_snapshots_per_batch,
            "gate_batch_map": gate_batch_map,
            "num_batches": len(gate_batches),
            "reconfiguration_time_us": total_reconfig_time,
            "total_routing_swaps": total_routing_ops,
        }

        return RoutedCircuit(
            operations=all_operations,
            final_mapping=current_mapping,
            routing_overhead=total_routing_ops,
            mapped_circuit=mapped_circuit,
            metadata=routing_metadata,
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
        
        for dg in mapped_circuit.native_circuit.operations:
            logical_qubits = dg.qubits
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
        mapping: Optional["QubitMapping"] = None,
    ) -> List[Any]:
        """Convert routing schedule operations to physical operations.

        The SAT solver produces dicts ``{"type": "H_SWAP"/"V_SWAP",
        "row": r, "col": c}`` where *col* is an absolute ion-column
        index (0 … total_columns-1).  An H_SWAP swaps the ion at
        (row, col) with the ion at (row, col+1).  A V_SWAP swaps
        (row, col) with (row+1, col).

        We emit **two** TransportOperations per swap (one for each
        ion), with ``source_zone`` / ``target_zone`` set to
        ``"trap_<row>_<col>"`` so downstream animation can locate them
        on the WISE grid.

        Parameters
        ----------
        mapping : QubitMapping, optional
            The current qubit-to-zone mapping *before* this batch's
            swaps.  Used to reconstruct the real grid permutation so
            that emitted qubit IDs are correct.  When ``None`` the
            grid defaults to the identity permutation (correct only
            for batch 0).
        """
        from qectostim.experiments.hardware_simulation.core.operations import (
            TransportOperation,
        )

        # Resolve grid dimensions from architecture
        _rows = getattr(architecture, "rows", 1)
        _k = getattr(architecture, "ions_per_segment", 1)
        _m = getattr(architecture, "col_groups", 1)
        _total_cols = _m * _k

        # --- Permutation-aware grid ---
        # Reconstruct which ion sits at each (row, col) from the
        # current QubitMapping's zone_assignments.  This mirrors the
        # logic in ``_build_initial_layout``.
        _grid = np.arange(_rows * _total_cols, dtype=int).reshape(
            _rows, _total_cols
        )
        if mapping is not None:
            placed: set = set()
            _n_cols = _total_cols
            # Strategy 1: parse zone_assignments "trap_<row>_<col>"
            for _log, _phys in mapping.logical_to_physical.items():
                zone = mapping.zone_assignments.get(_phys)
                if zone:
                    try:
                        parts = zone.split("_")
                        zr, zc = int(parts[-2]), int(parts[-1])
                        if 0 <= zr < _rows and 0 <= zc < _n_cols:
                            _grid[zr, zc] = _phys
                            placed.add((zr, zc))
                    except (ValueError, IndexError):
                        pass
            # Strategy 2: positional fallback for unplaced qubits
            if len(placed) < len(mapping.logical_to_physical):
                for _log, _phys in mapping.logical_to_physical.items():
                    pr = _phys // _n_cols if _n_cols > 0 else 0
                    pc = _phys % _n_cols if _n_cols > 0 else _phys
                    if (0 <= pr < _rows and 0 <= pc < _n_cols
                            and (pr, pc) not in placed):
                        _grid[pr, pc] = _phys
                        placed.add((pr, pc))
            # Fill remaining cells with unique indices
            used_ids = set(_grid.flatten())
            next_idx = int(max(used_ids) + 1) if used_ids else 0
            for _fr in range(_rows):
                for _fc in range(_n_cols):
                    if (_fr, _fc) not in placed:
                        _grid[_fr, _fc] = next_idx
                        next_idx += 1

        physical_ops: List[Any] = []

        for op in routing_ops:
            if isinstance(op, dict):
                op_type = op.get("type", "")

                # Pass-boundary sentinel — emit a marker so the
                # animation can group parallel swaps atomically.
                if op_type == "PASS_BOUNDARY":
                    physical_ops.append(TransportOperation(
                        qubit=-1,
                        source_zone="__PASS_BOUNDARY__",
                        target_zone="__PASS_BOUNDARY__",
                        duration=0.0,
                    ))
                    continue

                if op_type in ("H_SWAP", "V_SWAP", "transport", "swap", "ROUTING"):
                    r = op.get("row", 0)
                    c = op.get("col", 0)

                    # If the dict carries explicit source/target keys
                    # (legacy format) bypass the grid-based logic.
                    _has_legacy_keys = "source" in op or "target" in op

                    if op_type == "H_SWAP" and not _has_legacy_keys:
                        # Horizontal swap: ion at (r, c) ↔ (r, c+1)
                        c2 = c + 1
                        if c2 >= _grid.shape[1]:
                            wise_logger.debug(
                                f"H_SWAP col {c2} out of grid; skipping")
                            continue
                        q_a = int(_grid[r, c])
                        q_b = int(_grid[r, c2])
                        src_a = f"trap_{r}_{c}"
                        tgt_a = f"trap_{r}_{c2}"
                        src_b = f"trap_{r}_{c2}"
                        tgt_b = f"trap_{r}_{c}"
                        # Advance permutation state
                        _grid[r, c], _grid[r, c2] = _grid[r, c2], _grid[r, c]
                    elif op_type == "V_SWAP" and not _has_legacy_keys:
                        # Vertical swap: ion at (r, c) ↔ (r+1, c)
                        r2 = r + 1
                        if r2 >= _grid.shape[0]:
                            wise_logger.debug(
                                f"V_SWAP row {r2} out of grid; skipping")
                            continue
                        q_a = int(_grid[r, c])
                        q_b = int(_grid[r2, c])
                        src_a = f"trap_{r}_{c}"
                        tgt_a = f"trap_{r2}_{c}"
                        src_b = f"trap_{r2}_{c}"
                        tgt_b = f"trap_{r}_{c}"
                        # Advance permutation state
                        _grid[r, c], _grid[r2, c] = _grid[r2, c], _grid[r, c]
                    else:
                        # Legacy / generic routing dict — use source/target
                        # keys if present, otherwise fall back to row/col.
                        qubits = op.get("qubits", ())
                        qubit = qubits[0] if qubits else int(_grid[r, c])
                        src_zone = op.get("source", f"trap_{r}_{c}")
                        tgt_zone = op.get("target", f"trap_{r}_{c}")
                        physical_ops.append(TransportOperation(
                            qubit=qubit,
                            source_zone=str(src_zone),
                            target_zone=str(tgt_zone),
                            duration=op.get("distance", 1.0) * 10.0,
                        ))
                        continue

                    # Emit two transports for the swap pair.
                    # Attach swap metadata so the visualization can
                    # determine junction waypoints accurately.
                    _swap_meta = {
                        "swap_type": op_type,
                        "swap_row": r,
                        "swap_col": c,
                    }
                    physical_ops.append(TransportOperation(
                        qubit=q_a,
                        source_zone=src_a,
                        target_zone=tgt_a,
                        duration=10.0,
                        metadata=_swap_meta,
                    ))
                    physical_ops.append(TransportOperation(
                        qubit=q_b,
                        source_zone=src_b,
                        target_zone=tgt_b,
                        duration=10.0,
                        metadata=_swap_meta,
                    ))
                else:
                    wise_logger.debug(f"Skipping unknown routing op dict: {op}")
            else:
                # Non-dict (e.g., already a PhysicalOperation)
                physical_ops.append(op)

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
