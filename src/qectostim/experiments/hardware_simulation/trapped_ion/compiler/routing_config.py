"""
Configuration dataclass for SAT-based WISE routing.

Holds tuning parameters that control the SAT solver, patch slicer,
and parallel config search. Compatible with the existing routing
infrastructure in qccd_operations.py.

Progress Bars
-------------
Two progress bar helpers are provided:

- ``make_tqdm_progress_callback``: Single progress bar — simple but
  causes conflicting updates when both the outer WISE routing loop
  and the inner SAT solver emit progress.

- ``make_nested_tqdm_progress_callback``: **Recommended.** Creates two
  non-conflicting nested bars:
  
  - **Outer bar** (position=0): Tracks routing/patching cycles in
    ``qccd_WISE_ion_route``. Updated by stages: ``STAGE_ROUTING``,
    ``STAGE_PATCHING``, ``STAGE_TILING_CYCLE``, etc.
    
  - **Inner bar** (position=1): Tracks SAT solving within each patch
    in ``qccd_SAT_WISE_odd_even_sorter``. Updated by stages:
    ``STAGE_SAT_POOL_START``, ``STAGE_SAT_CONFIG_DONE``, etc.

Example
-------
>>> from src.utils.routing_config import WISERoutingConfig, make_nested_tqdm_progress_callback
>>> callback, close = make_tqdm_progress_callback("Routing")
>>> config = WISERoutingConfig(
...     timeout_seconds=600,
...     progress_callback=callback,
... )
>>> # ... run routing ...
>>> close()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, List


@dataclass
class RoutingProgress:
    """Snapshot of SAT-routing progress, emitted via a callback.
    
    Attributes
    ----------
    stage : str
        Human-readable name of the current phase.
        Use the ``STAGE_*`` module constants for consistent naming:
        ``STAGE_ROUTING``, ``STAGE_PATCHING``, ``STAGE_SAT_POOL``,
        ``STAGE_RECONFIG``, ``STAGE_COMPLETE``.
    current : int
        Current round index (0-based).
    total : int
        Total number of MS-gate rounds to route.
    gates_remaining : int
        Number of gate operations still unrouted.
    elapsed_seconds : float
        Wall-clock time since routing began.
    message : str
        Free-form status string.
    extra : dict
        Arbitrary extra data (reconfig time, cache hits, etc.).
    """
    stage: str = "routing"
    current: int = 0
    total: int = 0
    gates_remaining: int = 0
    elapsed_seconds: float = 0.0
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def percent(self) -> float:
        """Completion percentage (0-100)."""
        return (self.current / self.total * 100.0) if self.total > 0 else 0.0


# Stage name constants for RoutingProgress.stage
STAGE_ROUTING = "routing"
STAGE_PATCHING = "patching"
STAGE_SAT_POOL = "sat_pool"
STAGE_SAT_POOL_START = "sat_pool_start"
STAGE_SAT_CONFIG_START = "sat_config_start"
STAGE_SAT_CONFIG_DONE = "sat_config_done"
STAGE_SAT_BINARY_SEARCH = "sat_binary_search"
STAGE_SAT_CALL = "sat_call"
STAGE_PATCH_START = "patch_start"
STAGE_PATCH_SOLVE = "patch_solve"
STAGE_PATCH_DONE = "patch_done"
STAGE_TILING_CYCLE = "tiling_cycle"
STAGE_RECONFIG = "reconfig"
STAGE_RECONFIG_PROGRESS = "reconfig_progress"
STAGE_COMPLETE = "complete"

# Fine-grained stages for detailed progress tracking within SAT solver
STAGE_BINARY_SEARCH_ITER = "binary_search_iter"  # Emitted each binary search iteration
STAGE_SAT_SOLVING = "sat_solving"  # Emitted when SAT subprocess is running
STAGE_SAT_SOLVED = "sat_solved"  # Emitted when SAT subprocess returns


@dataclass
class WISESolverParams:
    """Tuning parameters for the per-patch SAT / MaxSAT solver.

    These control timeouts, soft-clause weights, and the starting search
    point inside ``optimal_QMR_for_WISE``.  Use :meth:`from_grid` for
    sensible defaults based on grid dimensions.

    Attributes
    ----------
    max_sat_time : float or None
        Wall-clock timeout (seconds) for each Minisat22 sub-process.
        ``None`` means auto-compute from grid dimensions at solve time.
    max_rc2_time : float or None
        Wall-clock timeout (seconds) for each RC2 (MaxSAT) sub-process.
        ``None`` means auto-compute from grid dimensions at solve time.
    wB_col : int
        Soft-clause weight penalising boundary column placement.
    wB_row : int
        Soft-clause weight penalising boundary row placement.
    base_pmax_in : int or None
        Starting P_max for the binary search.  ``None`` means auto
        (defaults to R inside the solver).
    sat_workers : int
        Maximum number of parallel SAT worker processes inside the
        ``optimal_QMR_for_WISE`` pool.
    """

    max_sat_time: Optional[float] = None
    max_rc2_time: Optional[float] = None
    wB_col: int = 1
    wB_row: int = 1
    base_pmax_in: Optional[int] = None
    sat_workers: int = 4

    @classmethod
    def from_grid(
        cls,
        n: int,
        m: int,
        k: int = 2,
        R: int = 1,
    ) -> "WISESolverParams":
        """Compute solver parameters from grid dimensions.

        Uses simple heuristics that scale with the grid size to avoid
        wasting time on tiny grids or timing out on large ones.

        Parameters
        ----------
        n : int
            Number of rows in the ion grid.
        m : int
            Number of trap columns.
        k : int
            Ions per trap (block width).
        R : int
            Number of MS-gate rounds in this window.

        Returns
        -------
        WISESolverParams
            A params object with reasonable defaults for the given grid.
        """
        total_cells = n * m * k
        # Use the unified difficulty-scaled timeout estimator (lazy import
        # to avoid circular dependency with the SAT module).
        from .qccd_SAT_WISE_odd_even_sorter import _estimate_sat_timeout

        base_pmax = max(R, 1)
        sat_time = _estimate_sat_timeout(n, m, total_cells, R, base_pmax)
        rc2_time = sat_time  # same budget for MaxSAT
        # Workers: cap at 4 for small grids, 2 for tiny ones
        if total_cells <= 8:
            workers = 1
        elif total_cells <= 24:
            workers = 2
        else:
            workers = 4
        return cls(
            max_sat_time=sat_time,
            max_rc2_time=rc2_time,
            wB_col=1,
            wB_row=1,
            base_pmax_in=base_pmax,
            sat_workers=workers,
        )


# Type alias for progress callback
ProgressCallback = Callable[[RoutingProgress], None]


@dataclass
class WISERoutingConfig:
    """Configuration for WISE SAT-based routing.
    
    Parameters
    ----------
    timeout_seconds : float
        Per-config SAT solver timeout (seconds).
    subgridsize : tuple of int
        ``(rows, cols, depth)`` for the patch slicer that tiles the
        grid into sub-problems before calling the SAT solver.
    lookahead : int
        Number of future gate rounds the SAT solver considers
        when building the CNF formula.
    max_passes : int
        Upper bound on the number of SAT passes explored by
        the binary-search solver.
    use_maxsat : bool
        If ``True``, use RC2 (MaxSAT) optimization; otherwise pure
        SAT with iterative deepening.
    parallel_sat_search : bool
        Run multiple configs in parallel via ProcessPoolExecutor.
    sat_workers : int
        Number of parallel SAT worker processes when
        ``parallel_sat_search`` is on.
    patch_enabled : bool
        Enable patch-based decomposition of the grid.
    debug_mode : bool
        Emit verbose SAT-level logging.
    block_aware_patching : bool
        Use block-aware patch boundaries.
    base_pmax_in : Optional[int]
        Base pmax value for SAT solver (None for auto).
    cache_ec_rounds : bool
        When ``True`` (default), identical EC stabilizer rounds are
        detected via ``round_signature`` hashing and the SAT-solved
        routing schedule is replayed instead of re-solved.  Set to
        ``False`` to force every EC phase to be SAT-routed from
        scratch (useful for benchmarking or debugging).
    progress_callback : Optional[ProgressCallback]
        Callback function for progress updates.
    solver_params : Optional[WISESolverParams]
        Fine-grained SAT/MaxSAT solver parameters.  When ``None``,
        the solver uses its built-in defaults.  Use
        ``WISESolverParams.from_grid(...)`` for auto-tuned values.
    """
    timeout_seconds: Optional[float] = None
    subgridsize: Tuple[int, int, int] = (6, 4, 1)
    lookahead: int = 2
    max_passes: int = 10
    use_maxsat: bool = True
    parallel_sat_search: bool = False
    sat_workers: int = 4
    patch_enabled: bool = False
    debug_mode: bool = False
    block_aware_patching: bool = True
    base_pmax_in: Optional[int] = None
    cache_ec_rounds: bool = True
    replay_level: int = 1
    progress_callback: Optional[ProgressCallback] = None
    solver_params: Optional[WISESolverParams] = None

    @classmethod
    def default(
        cls,
        *,
        lookahead: int = 2,
        subgridsize: Tuple[int, int, int] = (6, 4, 1),
        base_pmax_in: int = 1,
        sat_workers: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        show_progress: bool = True,
        cache_ec_rounds: bool = True,
        replay_level: int = 1,
    ) -> "WISERoutingConfig":
        """Create a routing config with production defaults.

        Mirrors the defaults used by ``search_configs_best_exec_time``
        in the best-effort compilation module: all CPU cores for SAT
        parallelism, ``base_pmax_in=1``, and a tqdm progress bar.

        Parameters
        ----------
        lookahead : int
            Future gate rounds in the SAT formula.
        subgridsize : tuple of int
            ``(rows, cols, increment)`` for the patch slicer.
        base_pmax_in : int
            Starting P_max for binary search.
        sat_workers : int or None
            Parallel SAT worker processes.  ``None`` (default) uses
            all available CPU cores (``mp.cpu_count()``).
        timeout_seconds : float or None
            Per-config SAT timeout.  ``None`` for auto.
        show_progress : bool
            Attach a tqdm progress bar callback.
        cache_ec_rounds : bool
            Cache and replay identical EC stabilizer rounds.
            ``True`` by default for performance; set ``False`` to
            force fresh SAT routing on every EC phase.
        replay_level : int
            Controls cache-replay granularity and route-back behaviour.

            - ``0``: No replay.  Every routing round is solved fresh
              with SAT.  No caching, no route-back reconfigurations.
              Guarantees optimal SAT-based routing everywhere at the
              cost of longer compilation.
            - ``1`` (default): Maximal replay.  Cache every single
              MS round and replay whenever the same ``round_signature``
              is seen.  Route-back after every sub-round.
            - *d* (code distance): Minimal replay.  Cache once per
              full EC block (all *d* stabiliser rounds).  Route-back
              only at EC-block and gadget–EC boundaries.

            In general, ``replay_level`` in ``[0, d]``: cache every
            ``replay_level`` stabiliser rounds as a unit and
            route-back after each cached unit.

        Returns
        -------
        WISERoutingConfig
        """
        import multiprocessing as mp

        if sat_workers is None:
            sat_workers = max(1, mp.cpu_count())

        solver_params = WISESolverParams(
            max_sat_time=timeout_seconds,
            max_rc2_time=timeout_seconds,
            base_pmax_in=base_pmax_in,
            sat_workers=sat_workers,
        )

        progress_cb = None
        progress_close = None
        if show_progress:
            progress_cb, progress_close = make_triple_tqdm_progress_callback(
                round_desc="MS Rounds",
                patch_desc="Patches",
                sat_desc="SAT Configs",
            )

        obj = cls(
            timeout_seconds=timeout_seconds,
            subgridsize=subgridsize,
            lookahead=lookahead,
            sat_workers=sat_workers,
            base_pmax_in=base_pmax_in,
            cache_ec_rounds=cache_ec_rounds,
            replay_level=replay_level,
            solver_params=solver_params,
            progress_callback=progress_cb,
        )
        # Stash close function so callers (e.g. the compiler) can
        # clean up the tqdm bar after routing completes.
        obj._progress_close = progress_close  # type: ignore[attr-defined]
        return obj

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert config to kwargs dict for backward compatibility.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of routing parameters.
        """
        return {
            "timeout": self.timeout_seconds,
            "subgridsize": self.subgridsize,
            "lookahead": self.lookahead,
            "maxPasses": self.max_passes,
            "use_maxsat": self.use_maxsat,
            "patchEnabled": self.patch_enabled,
            "debug_mode": self.debug_mode,
        }


def make_tqdm_progress_callback(
    desc: str = "SAT Routing",
) -> Tuple[ProgressCallback, Callable[[], None]]:
    """Create a tqdm-based progress callback for SAT routing.
    
    Returns ``(callback, close)`` — call ``close()`` when routing is
    done to finalize the progress bar.
    
    Uses ``tqdm.auto`` so the bar renders as a Jupyter widget when
    running inside a notebook, or as a plain-text bar in a terminal.
    
    Example
    -------
    >>> callback, close = make_tqdm_progress_callback("Routing")
    >>> config = WISERoutingConfig(progress_callback=callback)
    >>> # ... run routing ...
    >>> close()
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        # Fallback: no-op callback if tqdm not available
        def _noop(p: RoutingProgress) -> None:
            pass
        return _noop, lambda: None
    
    bar: Optional[tqdm] = None
    
    def _callback(p: RoutingProgress) -> None:
        nonlocal bar
        if bar is None:
            bar = tqdm(total=p.total, desc=desc, unit="round")
        bar.total = p.total
        bar.n = p.current
        bar.set_postfix_str(p.message[:30] if p.message else "")
        bar.refresh()
    
    def _close() -> None:
        nonlocal bar
        if bar is not None:
            bar.close()
    
    return _callback, _close


def make_simple_progress_callback() -> Tuple[ProgressCallback, Callable[[], List[RoutingProgress]]]:
    """Create a simple progress callback that stores progress history.
    
    Returns ``(callback, get_history)`` — call ``get_history()`` to
    retrieve all progress snapshots.
    
    Example
    -------
    >>> callback, get_history = make_simple_progress_callback()
    >>> config = WISERoutingConfig(progress_callback=callback)
    >>> # ... run routing ...
    >>> history = get_history()
    >>> print(f"Routed {history[-1].current}/{history[-1].total} rounds")
    """
    history: List[RoutingProgress] = []
    
    def _callback(p: RoutingProgress) -> None:
        history.append(p)
    
    def _get_history() -> List[RoutingProgress]:
        return history
    
    return _callback, _get_history


def make_logging_progress_callback(
    logger: Optional[Any] = None,
    level: str = "INFO",
) -> ProgressCallback:
    """Create a logging-based progress callback.
    
    Parameters
    ----------
    logger : Optional[logging.Logger]
        Logger to use. If None, uses print().
    level : str
        Log level (INFO, DEBUG, etc.).
        
    Returns
    -------
    ProgressCallback
        Callback that logs progress updates.
    """
    def _callback(p: RoutingProgress) -> None:
        msg = f"[{p.stage}] {p.current}/{p.total} ({p.percent:.1f}%) - {p.message}"
        if logger is not None:
            log_fn = getattr(logger, level.lower(), logger.info)
            log_fn(msg)
        else:
            print(msg)
    
    return _callback


# Stages that belong to the outer (WISE ion route) progress bar
_OUTER_STAGES = frozenset({
    STAGE_ROUTING,
    STAGE_PATCHING,
    STAGE_TILING_CYCLE,
    STAGE_PATCH_START,
    STAGE_PATCH_SOLVE,  # Patch solve reports per-patch progress from routing loop
    STAGE_PATCH_DONE,
    STAGE_RECONFIG,
    STAGE_COMPLETE,
})

# Stages that drive the patch (within-cycle) progress bar
_PATCH_STAGES = frozenset({
    STAGE_PATCHING,
    STAGE_TILING_CYCLE,
    STAGE_PATCH_START,
    STAGE_PATCH_SOLVE,
    STAGE_PATCH_DONE,
    STAGE_RECONFIG,
})

# Stages that belong to the inner (SAT solver) progress bar
_INNER_STAGES = frozenset({
    STAGE_SAT_POOL,
    STAGE_SAT_POOL_START,
    STAGE_SAT_CONFIG_START,
    STAGE_SAT_CONFIG_DONE,
    STAGE_SAT_BINARY_SEARCH,
    STAGE_SAT_CALL,
    STAGE_BINARY_SEARCH_ITER,
    STAGE_SAT_SOLVING,
    STAGE_SAT_SOLVED,
})


def make_sat_only_tqdm_progress_callback(
    sat_desc: str = "SAT Configs",
    print_routing_status: bool = True,
) -> Tuple[ProgressCallback, Callable[[], None]]:
    """Progress callback with tqdm bar for SAT only, text for routing.
    
    This avoids multi-bar conflicts in notebooks by:
    - Showing a single tqdm bar for SAT config progress (inner loop)
    - Printing text messages for routing cycle progress (outer loop)
    
    Returns ``(callback, close)`` — call ``close()`` when routing is done.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def _noop(p: RoutingProgress) -> None:
            pass
        return _noop, lambda: None
    
    sat_bar: Optional[tqdm] = None
    _last_sat_state: Tuple[int, int, str] = (-1, -1, "")
    _last_outer_msg: str = ""
    
    def _callback(p: RoutingProgress) -> None:
        nonlocal sat_bar, _last_sat_state, _last_outer_msg
        
        stage = p.stage
        
        if stage in _INNER_STAGES:
            # Update SAT progress bar
            new_state = (p.current, p.total, p.message[:40] if p.message else "")
            
            if sat_bar is None:
                sat_bar = tqdm(
                    total=max(p.total, 1),
                    desc=sat_desc,
                    unit="cfg",
                    leave=True,
                    dynamic_ncols=True,
                )
            
            # Update bar in-place without reset (avoids flashing to 0%)
            if new_state != _last_sat_state:
                sat_bar.total = max(p.total, 1)
                sat_bar.n = min(p.current, sat_bar.total)
                sat_bar.set_postfix_str(new_state[2])
                sat_bar.refresh()
                _last_sat_state = new_state
                
        elif stage in _OUTER_STAGES and print_routing_status:
            # Print routing status as text (only on significant changes)
            msg = p.message or ""
            if msg and msg != _last_outer_msg:
                # Use tqdm.write to print above the progress bar
                from tqdm.auto import tqdm as _tqdm_cls
                _tqdm_cls.write(f"[Routing] {msg}")
                _last_outer_msg = msg
    
    def _close() -> None:
        nonlocal sat_bar
        if sat_bar is not None:
            sat_bar.close()
            sat_bar = None
    
    return _callback, _close


def make_triple_tqdm_progress_callback(
    round_desc: str = "MS Rounds",
    patch_desc: str = "Patches",
    sat_desc: str = "SAT Configs",
) -> Tuple[ProgressCallback, Callable[[], None]]:
    """Progress callback with three tqdm bars: MS rounds, patches, SAT.

    Mirrors the 3-bar display from the ``ProgressTableWidget`` used by
    ``search_configs_best_exec_time`` but as plain tqdm bars that work in
    any notebook or terminal environment.

    Bars:
      - **MS Rounds** (position=0): Tracks which MS round is being routed.
        Updated by ``STAGE_ROUTING``.
      - **Patches** (position=1): Tracks patch progress within a tiling
        cycle. Updated by ``STAGE_PATCHING`` and ``STAGE_PATCH_SOLVE``.
      - **SAT Configs** (position=2): Tracks SAT config solving within
        a patch. Updated by ``STAGE_SAT_POOL_START`` and
        ``STAGE_SAT_CONFIG_DONE``.

    Returns ``(callback, close)`` — call ``close()`` when routing is done.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def _noop(p: RoutingProgress) -> None:
            pass
        return _noop, lambda: None

    round_bar: Optional[tqdm] = None
    patch_bar: Optional[tqdm] = None
    sat_bar: Optional[tqdm] = None
    _last_round: Tuple[int, int, str] = (-1, -1, "")
    _last_patch: Tuple[int, int, str] = (-1, -1, "")
    _last_sat: Tuple[int, int, str] = (-1, -1, "")

    def _callback(p: RoutingProgress) -> None:
        nonlocal round_bar, patch_bar, sat_bar
        nonlocal _last_round, _last_patch, _last_sat

        stage = p.stage

        # ── MS round bar ──────────────────────────────────────────
        if stage == STAGE_ROUTING:
            state = (p.current, p.total, p.message[:50] if p.message else "")
            if round_bar is None:
                round_bar = tqdm(
                    total=max(p.total, 1), desc=round_desc,
                    unit="rnd", position=0, leave=True, dynamic_ncols=True,
                )
            if state != _last_round:
                round_bar.total = max(p.total, 1)
                round_bar.n = min(p.current, round_bar.total)
                round_bar.set_postfix_str(state[2])
                round_bar.refresh()
                _last_round = state
            # Reset patch bar when a new round starts
            if patch_bar is not None:
                patch_bar.close()
                patch_bar = None
                _last_patch = (-1, -1, "")

        # ── Patch bar ─────────────────────────────────────────────
        elif stage in (STAGE_PATCHING, STAGE_PATCH_SOLVE, STAGE_PATCH_DONE):
            state = (p.current, p.total, p.message[:50] if p.message else "")
            if patch_bar is None:
                patch_bar = tqdm(
                    total=max(p.total, 1), desc=patch_desc,
                    unit="pat", position=1, leave=False, dynamic_ncols=True,
                )
            if state != _last_patch:
                patch_bar.total = max(p.total, 1)
                patch_bar.n = min(p.current, patch_bar.total)
                patch_bar.set_postfix_str(state[2])
                patch_bar.refresh()
                _last_patch = state

        # ── SAT bar ───────────────────────────────────────────────
        elif stage in _INNER_STAGES:
            state = (p.current, p.total, p.message[:50] if p.message else "")
            if stage == STAGE_SAT_POOL_START:
                if sat_bar is not None:
                    sat_bar.close()
                sat_bar = tqdm(
                    total=max(p.total, 1), desc=sat_desc,
                    unit="cfg", position=2, leave=False, dynamic_ncols=True,
                )
                _last_sat = (-1, -1, "")
            elif sat_bar is None:
                sat_bar = tqdm(
                    total=max(p.total, 1), desc=sat_desc,
                    unit="cfg", position=2, leave=False, dynamic_ncols=True,
                )
            if sat_bar is not None and state != _last_sat:
                sat_bar.total = max(p.total, 1)
                sat_bar.n = min(p.current, sat_bar.total)
                sat_bar.set_postfix_str(state[2])
                sat_bar.refresh()
                _last_sat = state

        # ── Reconfig bar ──────────────────────────────────────────
        # Transition/return reconfigs happen AFTER MS rounds complete.
        # Show the reconfig message on the round bar so users see
        # ongoing progress instead of a stalled 40/40 display.
        elif stage == STAGE_RECONFIG:
            if round_bar is not None:
                round_bar.set_postfix_str(
                    p.message[:60] if p.message else "reconfig"
                )
                round_bar.refresh()

        # ── Complete ──────────────────────────────────────────────
        elif stage == STAGE_COMPLETE:
            if round_bar is not None:
                round_bar.total = max(p.total, 1)
                round_bar.n = round_bar.total
                round_bar.set_postfix_str(
                    p.message[:60] if p.message else "done"
                )
                round_bar.refresh()

    def _close() -> None:
        nonlocal round_bar, patch_bar, sat_bar
        for bar in (sat_bar, patch_bar, round_bar):
            if bar is not None:
                bar.close()
        sat_bar = patch_bar = round_bar = None

    return _callback, _close


def make_nested_tqdm_progress_callback(
    outer_desc: str = "WISE Routing",
    inner_desc: str = "SAT Solving",
    leave_inner: bool = False,
) -> Tuple[ProgressCallback, Callable[[], None]]:
    """Create nested tqdm progress bars for WISE routing.
    
    Returns ``(callback, close)`` — call ``close()`` when routing is done.
    
    This creates two progress bars:
      - **Outer bar** (position=0): Tracks routing/patching cycles in
        ``qccd_WISE_ion_route``. Updated by stages: ``STAGE_ROUTING``,
        ``STAGE_PATCHING``, ``STAGE_PATCH_SOLVE``, etc.
      - **Inner bar** (position=1): Tracks SAT config solving within the
        SAT pool in ``qccd_SAT_WISE_odd_even_sorter``. Updated by stages:
        ``STAGE_SAT_POOL_START``, ``STAGE_SAT_CONFIG_DONE``, etc.
    
    The bars do not conflict because each only responds to its own stages.
    
    Note: In notebooks without ipywidgets, tqdm falls back to plain-text
    mode which doesn't handle multiple positioned bars well. In that case,
    the bars will print sequentially rather than updating in place.
    
    Parameters
    ----------
    outer_desc : str
        Description for the outer (routing) progress bar.
    inner_desc : str
        Description for the inner (SAT solving) progress bar.
    leave_inner : bool
        If True, leave the inner bar visible after completion.
        If False (default), the inner bar is cleared when a new SAT
        pool starts.
    
    Example
    -------
    >>> callback, close = make_nested_tqdm_progress_callback()
    >>> config = WISERoutingConfig(progress_callback=callback)
    >>> # ... run routing ...
    >>> close()
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def _noop(p: RoutingProgress) -> None:
            pass
        return _noop, lambda: None
    
    outer_bar: Optional[tqdm] = None
    inner_bar: Optional[tqdm] = None
    # Track last update to avoid duplicate refreshes (-1 forces initial update)
    _last_outer_state: Tuple[int, int, str] = (-1, -1, "")
    _last_inner_state: Tuple[int, int, str] = (-1, -1, "")
    
    def _callback(p: RoutingProgress) -> None:
        nonlocal outer_bar, inner_bar, _last_outer_state, _last_inner_state
        
        stage = p.stage
        
        if stage in _OUTER_STAGES:
            # Update outer bar
            new_state = (p.current, p.total, p.message[:40] if p.message else "")
            
            if outer_bar is None:
                outer_bar = tqdm(
                    total=max(p.total, 1),
                    desc=outer_desc,
                    unit="cycle",
                    position=0,
                    leave=True,
                    dynamic_ncols=True,
                )
            
            # Only update if state changed (avoid duplicate lines)
            if new_state != _last_outer_state:
                outer_bar.total = max(p.total, 1)
                outer_bar.n = min(p.current, outer_bar.total)
                outer_bar.set_postfix_str(new_state[2])
                outer_bar.refresh()
                _last_outer_state = new_state
            
            # If we're starting a new phase, reset inner bar
            if stage in (STAGE_ROUTING, STAGE_PATCHING, STAGE_TILING_CYCLE):
                if inner_bar is not None and not leave_inner:
                    inner_bar.close()
                    inner_bar = None
                    _last_inner_state = (-1, -1, "")  # Sentinel to force update
                    
        elif stage in _INNER_STAGES:
            new_state = (p.current, p.total, p.message[:40] if p.message else "")
            
            # Update inner bar
            if stage == STAGE_SAT_POOL_START:
                # Starting a new SAT pool — reset inner bar
                if inner_bar is not None:
                    inner_bar.close()
                inner_bar = tqdm(
                    total=max(p.total, 1),
                    desc=inner_desc,
                    unit="cfg",
                    position=1,
                    leave=leave_inner,
                    dynamic_ncols=True,
                )
                # Reset state to sentinel to force first update on new bar
                _last_inner_state = (-1, -1, "")
            elif inner_bar is None:
                # Create inner bar if it doesn't exist
                inner_bar = tqdm(
                    total=max(p.total, 1),
                    desc=inner_desc,
                    unit="cfg",
                    position=1,
                    leave=leave_inner,
                    dynamic_ncols=True,
                )
                # _last_inner_state already starts as (-1, -1, "") so first update is forced
            
            # Only update if state changed (avoid duplicate lines)
            if inner_bar is not None and new_state != _last_inner_state:
                inner_bar.total = max(p.total, 1)
                inner_bar.n = min(p.current, inner_bar.total)
                inner_bar.set_postfix_str(new_state[2])
                inner_bar.refresh()
                _last_inner_state = new_state

        elif stage == STAGE_RECONFIG:
            # Transition/return reconfigs: show message on outer bar
            if outer_bar is not None:
                outer_bar.set_postfix_str(
                    p.message[:60] if p.message else "reconfig"
                )
                outer_bar.refresh()

        elif stage == STAGE_COMPLETE:
            if outer_bar is not None:
                outer_bar.total = max(p.total, 1)
                outer_bar.n = outer_bar.total
                outer_bar.set_postfix_str(
                    p.message[:60] if p.message else "done"
                )
                outer_bar.refresh()
    
    def _close() -> None:
        nonlocal outer_bar, inner_bar
        if inner_bar is not None:
            inner_bar.close()
            inner_bar = None
        if outer_bar is not None:
            outer_bar.close()
            outer_bar = None
    
    return _callback, _close


def make_single_tqdm_progress_callback(
    desc: str = "WISE SAT Routing",
) -> Tuple[ProgressCallback, Callable[[], None]]:
    """Create a single tqdm progress bar that shows the most relevant progress.
    
    This is simpler than ``make_nested_tqdm_progress_callback`` and works
    better in notebooks without ipywidgets support. It shows:
    
    - SAT config progress when inside the SAT solver pool
    - Routing cycle progress otherwise
    
    The bar automatically switches between tracking modes based on the
    current stage, avoiding the multi-bar positioning issues.
    
    Returns ``(callback, close)`` — call ``close()`` when routing is done.
    
    Example
    -------
    >>> callback, close = make_single_tqdm_progress_callback()
    >>> config = WISERoutingConfig(progress_callback=callback)
    >>> # ... run routing ...
    >>> close()
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def _noop(p: RoutingProgress) -> None:
            pass
        return _noop, lambda: None
    
    bar: Optional[tqdm] = None
    current_mode: str = ""  # "outer" or "inner"
    _last_state: Tuple[int, int, str] = (-1, -1, "")  # Use -1 to force initial update
    
    def _callback(p: RoutingProgress) -> None:
        nonlocal bar, current_mode, _last_state
        
        stage = p.stage
        
        # Determine which mode we should be in
        if stage in _INNER_STAGES:
            target_mode = "inner"
            target_desc = "SAT Configs"
            target_unit = "cfg"
        else:
            target_mode = "outer"
            target_desc = "Routing"
            target_unit = "cycle"
        
        new_state = (p.current, p.total, p.message[:40] if p.message else "")
        
        if bar is None:
            # First call - create the bar
            bar = tqdm(
                total=max(p.total, 1),
                desc=target_desc,
                unit=target_unit,
                leave=True,
                dynamic_ncols=True,
            )
            current_mode = target_mode
        elif target_mode != current_mode:
            # Mode changed - update bar properties in-place (no reset/close)
            bar.set_description(target_desc)
            bar.unit = target_unit
            bar.total = max(p.total, 1)
            bar.n = 0  # Start fresh for new mode
            current_mode = target_mode
            # Reset last state to force an update
            _last_state = (-1, -1, "")
        
        # Only update if state changed
        if new_state != _last_state:
            bar.total = max(p.total, 1)
            bar.n = min(p.current, bar.total)
            bar.set_postfix_str(new_state[2])
            bar.refresh()
            _last_state = new_state
    
    def _close() -> None:
        nonlocal bar
        if bar is not None:
            bar.close()
            bar = None
    
    return _callback, _close


def make_queue_progress_callback(
    progress_queue,
    config_id: int = 0,
) -> Tuple[ProgressCallback, Callable[[], None]]:
    """Progress callback that sends updates to a multiprocessing Queue.
    
    Workers use this callback to send SAT progress updates back to the main
    process. The main process polls the queue and updates a single tqdm bar.
    This allows parallel workers with proper progress visualization.
    
    Parameters
    ----------
    progress_queue : multiprocessing.Queue
        Queue for sending progress updates. Each update is a tuple of
        (config_id, stage, current, total, message).
    config_id : int
        Unique identifier for this config (used by main process to aggregate).
    
    Returns
    -------
    callback : ProgressCallback
        Callback function that sends updates to the queue.
    close : Callable[[], None]
        No-op for cleanup compatibility.
    """
    def _callback(p: RoutingProgress) -> None:
        # Only send SAT stages (inner progress) to avoid queue overhead
        if p.stage in _INNER_STAGES:
            try:
                progress_queue.put_nowait((config_id, p.stage, p.current, p.total, p.message))
            except Exception:
                pass  # Queue full or closed - skip silently
    
    def _close() -> None:
        pass  # Nothing to clean up
    
    return _callback, _close
