# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/config.py
"""
Configuration dataclass for SAT-based WISE routing.

Holds tuning parameters that control the SAT solver, patch slicer,
and parallel config search.  Accepted by ``WISECompiler.__init__``
and passed through to ``ionRoutingWISEArch()`` /
``search_configs_best_exec_time()`` when present.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple  # noqa: F811


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
        Upper bound on the number of SAT passes (Σ Pr) explored by
        the binary-search solver.
    use_maxsat : bool
        If ``True``, use RC2 (MaxSAT) optimisation; otherwise pure
        SAT with iterative deepening.
    parallel_sat_search : bool
        Run multiple configs in parallel via ``ProcessPoolExecutor``
        (see ``search_configs_best_exec_time``).
    sat_workers : int
        Number of parallel SAT worker processes when
        ``parallel_sat_search`` is on.
    patch_enabled : bool
        Enable patch-based decomposition of the grid.
    debug_mode : bool
        Emit verbose SAT-level logging.
    """

    timeout_seconds: float = 4800.0
    subgridsize: Tuple[int, int, int] = (6, 4, 1)
    lookahead: int = 2
    max_passes: int = 10
    use_maxsat: bool = True
    parallel_sat_search: bool = False
    sat_workers: int = 4
    patch_enabled: bool = False
    debug_mode: bool = False
    base_pmax_in: Optional[int] = None
    progress_callback: Optional["ProgressCallback"] = None


@dataclass
class RoutingProgress:
    """Snapshot of SAT-routing progress, emitted via a callback.

    Attributes
    ----------
    stage : str
        Human-readable name of the current phase
        (e.g. ``"routing"``, ``"done"``).
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


ProgressCallback = Callable[[RoutingProgress], None]
"""Type alias for a routing progress callback."""


def make_tqdm_progress_callback(
    desc: str = "SAT Routing",
) -> Tuple["ProgressCallback", Callable[[], None]]:
    """Create a tqdm-based progress callback for SAT routing.

    Returns ``(callback, close)`` — call ``close()`` when routing is
    done to finalise the progress bar.

    Uses ``tqdm.auto`` so the bar renders as a Jupyter widget when
    running inside a notebook, or as a plain-text bar in a terminal.

    Example
    -------
    >>> callback, close = make_tqdm_progress_callback("Routing")
    >>> config = WISERoutingConfig(progress_callback=callback)
    >>> # ... run routing ...
    >>> close()
    """
    from tqdm.auto import tqdm  # type: ignore

    bar: Optional[tqdm] = None

    def _callback(p: RoutingProgress) -> None:
        nonlocal bar
        if bar is None:
            bar = tqdm(total=p.total, desc=desc, unit="round")
        bar.total = p.total
        bar.n = p.current
        bar.set_postfix_str(
            f"{p.gates_remaining} ops left | "
            f"reconfig {p.extra.get('reconfig_time', 0):.2f}s"
        )
        bar.refresh()

    def _close() -> None:
        nonlocal bar
        if bar is not None:
            bar.n = bar.total
            bar.refresh()
            bar.close()
            bar = None

    return _callback, _close
