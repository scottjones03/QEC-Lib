"""
WISE SAT-Based Ion Routing Solver
=================================

This module implements a SAT/MaxSAT-based solver for ion routing in QCCD
(Quantum Charge-Coupled Device) trapped-ion quantum computers, following
the WISE (Window-based Ion Swap Execution) framework.

Overview
--------
The solver takes:
  1. An initial ion layout (n × m grid)
  2. Per-round lists of ion pairs that must be brought together for MS gates
  3. Optional block-transfer (BT) pinning constraints

And produces:
  1. A sequence of swap schedules that route ions to satisfy the pairing constraints
  2. Minimised total passes (P_max) via binary search over the SAT solution space

Architecture
------------
The solver uses a multi-level parallel search:

  - **outer loop**: Enumerates (P_max, boundary_capacity_factor) configurations
  - **ProcessPoolExecutor**: Dispatches configs to worker processes
  - **per-worker**: Binary search over ΣP (total passes) constraint
  - **per-iteration**: Builds CNF/WCNF formula, calls Minisat22 or RC2 (MaxSAT)

Key Classes
-----------
- ``WiseSATSolver``: OOP wrapper; recommended entry point
- ``SATSolverConfig``: Configuration dataclass with ``from_heuristics()`` factory
- ``SATProcessManager``: Context manager for reliable subprocess cleanup
- ``SATPoolResult``: Structured result from ``solve_managed()``

Exception Hierarchy
-------------------
- ``WiseSATError``: Base for SAT-specific exceptions
- ``SATTimeoutError``: Solver exceeded time budget
- ``BTConflictError``: Block-transfer constraints are infeasible
- ``CapacityExceededError``: Too many ions pinned to one column/row
- ``NoFeasibleLayoutError``: No satisfiable schedule exists

Progress Reporting
------------------
Fine-grained progress is reported via ``RoutingProgress`` callbacks:
  - ``STAGE_SAT_POOL_START``: Pool of workers launched
  - ``STAGE_SAT_CONFIG_START``: Individual config worker started
  - ``STAGE_SAT_CONFIG_DONE``: Worker completed
  - ``STAGE_BINARY_SEARCH_ITER``: Each binary search iteration
  - ``STAGE_SAT_SOLVING``: SAT subprocess active
  - ``STAGE_SAT_SOLVED``: SAT subprocess returned

Example Usage
-------------
Basic (manual configuration)::

    from qccd_SAT_WISE_odd_even_sorter import WiseSATSolver

    solver = WiseSATSolver(
        A_in=initial_grid,
        P_arr=pairs_per_round,
        k=2,
        max_sat_time=600.0,
    )
    layouts, schedule, pass_horizon = solver.solve()

Recommended (heuristics + managed cleanup)::

    solver = WiseSATSolver.from_heuristics(
        A_in=grid,
        P_arr=pairs,
        k=2,
    )
    result = solver.solve_managed()
    print(f"P_max = {result.pass_horizon}, elapsed = {result.elapsed_s:.1f}s")

With explicit configuration::

    from qccd_SAT_WISE_odd_even_sorter import SATSolverConfig, WiseSATSolver

    config = SATSolverConfig.from_heuristics(n=6, m=4, k=2, R=3)
    config.max_sat_time = 300.0  # Override heuristic

    solver = WiseSATSolver.from_config(grid, pairs, k=2, config=config)
    result = solver.solve_managed()

Implementation Notes
--------------------
- The core SAT construction logic (``_wise_build_structural_cnf``,
  ``_wise_decode_schedule_from_model``, etc.) is intentionally preserved
  as module-level functions to maintain compatibility and enable copy-paste
  reuse in other contexts.
- The OOP layer (``WiseSATSolver``) is a thin wrapper that delegates to
  ``optimal_QMR_for_WISE``.
- Subprocess cleanup is critical: SAT solvers spawn child processes that
  can become zombies if not properly terminated. Use ``solve_managed()``
  or ``SATProcessManager`` in production.

Environment Variables
---------------------
- ``WISE_SAT_WORKERS``: Override max worker count (default: CPU-dependent)
- ``WISE_MAX_SAT_TIME``: Global cap on SAT timeout (seconds)
- ``WISE_MAX_RC2_TIME``: Global cap on RC2/MaxSAT timeout (seconds)

Author: QECToStim team
"""

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
from ..utils.qccd_nodes import *
from ..utils.physics import DEFAULT_CALIBRATION, DEFAULT_FIDELITY_MODEL, CalibrationConstants
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
import sys
from pysat.formula import IDPool, WCNF, CNF
from pysat.card import CardEnc, EncType
from pysat.solvers import Minisat22  
from pysat.examples.rc2 import RC2
import atexit
import time
import pickle 
import os
import tempfile
import shutil
import multiprocessing as mp
from scipy import stats
from pysat.solvers import Solver
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import threading as _threading
from scipy.optimize import linear_sum_assignment
from .routing_config import (
    RoutingProgress,
    STAGE_SAT_POOL_START,
    STAGE_SAT_CONFIG_START,
    STAGE_SAT_CONFIG_DONE,
    STAGE_SAT_BINARY_SEARCH,
    STAGE_SAT_CALL,
    STAGE_BINARY_SEARCH_ITER,
    STAGE_SAT_SOLVING,
    STAGE_SAT_SOLVED,
)


def _get_safe_mp_context():
    """Return a safe multiprocessing context for the current platform.
    
    On macOS, 'spawn' is preferred because 'fork' can cause hangs and
    zombie processes when nested inside spawned processes. On Linux,
    'fork' is safe and faster.
    """
    import sys
    if sys.platform == "darwin":
        return mp.get_context("spawn")
    else:
        try:
            return mp.get_context("fork")
        except ValueError:
            return mp.get_context()


def _in_notebook_env() -> bool:
    """Return *True* when running inside a Jupyter / IPython notebook kernel.

    Used to avoid ``multiprocessing.Manager()`` which spawns a server
    subprocess that deadlocks the Jupyter kernel on macOS (spawn context).
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


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
    ctx = _get_safe_mp_context()

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
    HARD_CAP = 8

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


def _estimate_sat_difficulty(
    n: int,
    m: int,
    num_ions: int,
    R: int,
    P_max: int,
) -> float:
    """Estimate SAT problem difficulty as a unitless scaling factor.

    Returns ~1.0 for the simplest baseline problem (d=2, k=2: n=3, m=2,
    6 ions, R=1, P_max=1) and scales up sub-linearly for harder instances.

    The factor is used as the **single source of truth** for all timeout
    caps in the module: per-subprocess timeouts in ``from_heuristics()``,
    pool budgets, and per-config caps in the sequential fallback loop.

    Scaling rationale
    -----------------
    The SAT problem size is dominated by the number of placement variables,
    which is roughly ``num_ions × n × m × P_max × R``.  Empirically, SAT
    solvers handle well-structured ion-placement instances in time that
    scales sub-linearly (≈√N) with problem size, so we use √(ratio) to
    convert the variable estimate into a difficulty multiplier.

    Parameters
    ----------
    n : int
        Grid rows.
    m : int
        Grid columns (traps).
    num_ions : int
        Total number of ions being placed.
    R : int
        Number of MS-gate rounds.
    P_max : int
        Representative pass bound (typically ``base_pmax``).

    Returns
    -------
    float
        Difficulty factor ≥ 1.0.
    """
    # Old formula multiplied num_ions × (n×m) × P_max × R, but for
    # full grids num_ions ≈ n×m, so grid area was effectively squared.
    # Use grid_area × P_max × R instead (num_ions removed).
    var_estimate = (
        max(1, n * m)
        * max(1, P_max)
        * max(1, R)
    )
    # Reference: d=2, k=2 baseline → 3×2 grid, P_max=1, R=1 → 6
    _REFERENCE = 6.0
    ratio = var_estimate / _REFERENCE
    return max(1.0, math.sqrt(ratio))


# ---------------------------------------------------------------------------
# Timeout estimation from problem dimensions
# ---------------------------------------------------------------------------

# Base timeout (seconds) per unit of difficulty.  For the simplest baseline
# problem (d=2, k=2: difficulty ≈ 1.0) the per-subprocess timeout is this
# value.  Larger problems scale linearly with the difficulty factor produced
# by ``_estimate_sat_difficulty``, with **no hard ceiling**.
_TIMEOUT_BASE_S: float = 300.0

# Minimum pool budget in seconds.  Even the simplest problem gets at least
# this much total wall-clock time for the multi-config SAT pool.
_MIN_POOL_BUDGET_S_FLOOR: float = 600.0

# Pool budget multiplier: the pool gets ``solver_timeout * _POOL_BUDGET_MULT``
# seconds of total wall-clock time.  The multiplier accounts for the fact that
# the binary search inside each config issues multiple SAT calls, so the pool
# needs more time than a single subprocess.
_POOL_BUDGET_MULT: float = 10.0
BASE_CONFIG_BUDGET: float = 300.0

# Maximum clause count for in-process SAT solving (bypass subprocess).
# For small CNFs the subprocess spawn overhead (~1.5-2 s on macOS) far
# exceeds the solve time (<0.1 s).  Solving in-process is safe because
# these instances terminate quickly and cannot hang.
# Empirically, d≤3 / 3×4 grids produce CNFs with ~5 000-15 000 clauses.
#
# On macOS, multiprocessing.Process(fork) can deadlock after importing
# large C-extension libraries (numpy, pysat, etc.).  Set the environment
# variable ``WISE_INPROCESS_LIMIT`` to a large value (e.g. 999999999) to
# force all SAT/RC2 solves in-process, avoiding the subprocess entirely.
_SAT_INPROCESS_CLAUSE_LIMIT: int = int(
    os.environ.get("WISE_INPROCESS_LIMIT", "50000")
)

def _estimate_sat_timeout(
    n: int,
    m: int,
    num_ions: int,
    R: int,
    P_max: int,
) -> float:
    """Estimate a conservative per-subprocess SAT/RC2 timeout in seconds.

    Uses the unified difficulty estimator (``_estimate_sat_difficulty``)
    and the module-level ``_TIMEOUT_BASE_S`` constant to produce a
    timeout that:

    * Is conservative (~300 s) for trivial problems (d=2, k=2).
    * Scales sub-linearly with problem size (via √-difficulty).
    * Has no hard upper ceiling — very large problems (d≥5) get
      proportionally more time without being capped at an arbitrary
      constant.

    The formula is simply::

        timeout = _TIMEOUT_BASE_S × difficulty

    Representative values
    ---------------------
    ======  ===  ====  ====  =====  ============  ============
      d      k    n     m    ions   difficulty    timeout (s)
    ======  ===  ====  ====  =====  ============  ============
      2      2    3     2      6       1.0          300
      2      8    3     1     24       1.4          420
      2     16    3     1     48       2.0          600
      3      2    5     3     30       3.5        1,050
      3     16    5     1     80       7.9        2,370
      4      2    6     3     36      12.0        3,600
      5      2    7     4     63      29.7        8,910
    ======  ===  ====  ====  =====  ============  ============

    Parameters
    ----------
    n : int
        Grid rows (of the sub-grid / patch being solved).
    m : int
        Grid columns.
    num_ions : int
        Number of ions in the grid.
    R : int
        Number of MS-gate rounds.
    P_max : int
        Representative pass bound.

    Returns
    -------
    float
        Timeout in seconds (≥ ``_TIMEOUT_BASE_S``).
    """
    difficulty = _estimate_sat_difficulty(n, m, num_ions, R, P_max)
    return _TIMEOUT_BASE_S * difficulty


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class WiseSATError(RuntimeError):
    """Base exception for WISE SAT solver errors.
    
    All SAT-specific exceptions inherit from this class, allowing callers
    to catch all SAT-related errors with a single except clause.
    """
    pass


class SATTimeoutError(WiseSATError):
    """Raised when SAT/MaxSAT solver exceeds its time budget.
    
    Attributes
    ----------
    elapsed_seconds : float
        Time spent before timeout.
    timeout_seconds : float
        The configured timeout limit.
    config : dict
        Configuration that was being solved (P_max, capacity_factor, etc.).
    """
    def __init__(self, message: str, elapsed_seconds: float = 0.0,
                 timeout_seconds: float = 0.0, config: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.elapsed_seconds = elapsed_seconds
        self.timeout_seconds = timeout_seconds
        self.config = config or {}


class BTConflictError(WiseSATError):
    """Raised when block-transfer (BT) pins have irreconcilable conflicts.
    
    Attributes
    ----------
    round_idx : int
        The round where the conflict was detected.
    ion_a, ion_b : int
        The conflicting ion indices.
    conflict_type : str
        Type of conflict ("same_cell", "different_row", "different_block").
    """
    def __init__(self, message: str, round_idx: int = -1,
                 ion_a: int = -1, ion_b: int = -1, conflict_type: str = ""):
        super().__init__(message)
        self.round_idx = round_idx
        self.ion_a = ion_a
        self.ion_b = ion_b
        self.conflict_type = conflict_type


class CapacityExceededError(WiseSATError):
    """Raised when BT pins exceed column or row capacity.
    
    Attributes
    ----------
    round_idx : int
        The round where capacity was exceeded.
    column : int
        The oversubscribed column index.
    count : int
        Number of ions pinned to that column.
    limit : int
        Maximum allowed (n, the number of rows).
    """
    def __init__(self, message: str, round_idx: int = -1,
                 column: int = -1, count: int = 0, limit: int = 0):
        super().__init__(message)
        self.round_idx = round_idx
        self.column = column
        self.count = count
        self.limit = limit


# ---------------------------------------------------------------------------
# Process Lifecycle Manager
# ---------------------------------------------------------------------------

# Global registry of active managers for signal handler access
_active_managers: List["SATProcessManager"] = []

# Global registry of executor worker PIDs for atexit cleanup
_tracked_executor_pids: set = set()

# H10: Use shared utility instead of inline copy
from ..utils.process_utils import get_child_pids as _get_child_pids


def _kill_process_tree(pid: int, sig: int = signal.SIGTERM) -> None:
    """Kill a process and all its descendants."""
    # First, collect all descendants
    to_kill = {pid}
    visited = set()
    queue = [pid]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        children = _get_child_pids(current)
        to_kill.update(children)
        queue.extend(children)
    
    # Kill all collected PIDs
    for p in to_kill:
        try:
            os.kill(p, sig)
        except (OSError, ProcessLookupError):
            pass


def _kill_executor_workers(executor: ProcessPoolExecutor) -> None:
    """Forcefully terminate ProcessPoolExecutor worker processes.

    ``executor.shutdown(wait=False)`` does NOT kill running workers — it
    merely stops accepting new submissions.  Workers are NOT daemon
    processes, so they survive parent exit and become zombies.
    This helper sends SIGTERM then SIGKILL to every worker and their children.

    Must be called BEFORE ``executor.shutdown()`` because shutdown
    sets ``_processes`` to ``None``.
    """
    pids_to_kill: set = set()
    # _processes is private but has been stable since Python 3.2
    if hasattr(executor, '_processes') and executor._processes is not None:
        for pid in list(executor._processes):
            pids_to_kill.add(pid)
            # Also collect grandchildren (nested executor workers, SAT solvers)
            pids_to_kill.update(_get_child_pids(pid))
    if not pids_to_kill:
        return
    wise_logger.debug("[WISE] killing %d executor worker processes (including children)", len(pids_to_kill))
    
    # First pass: SIGTERM to allow graceful shutdown
    for pid in pids_to_kill:
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    time.sleep(0.5)
    
    # Second pass: SIGKILL for any survivors
    for pid in pids_to_kill:
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    # Remove from global tracker
    _tracked_executor_pids.difference_update(pids_to_kill)


def _atexit_cleanup() -> None:
    """Kill any tracked executor workers on interpreter exit."""
    # Kill tracked PIDs and their children
    all_pids = set(_tracked_executor_pids)
    for pid in list(_tracked_executor_pids):
        all_pids.update(_get_child_pids(pid))
    
    for pid in all_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    _tracked_executor_pids.clear()
    
    # Also clean up any active SATProcessManagers
    for mgr in list(_active_managers):
        try:
            mgr._kill_all_children()
        except Exception:
            pass


atexit.register(_atexit_cleanup)


class SATProcessManager:
    """Context manager for reliable SAT subprocess lifecycle management.
    
    Ensures all child processes (SAT solvers, RC2 MaxSAT solvers) are
    properly terminated even when the parent receives KeyboardInterrupt,
    SIGTERM, or other exceptions. This prevents zombie processes that
    can accumulate when running `run_single_config` repeatedly.
    
    The manager:
    1. Creates a multiprocessing Manager for shared state (stop_event, PID dicts)
    2. Installs signal handlers that cooperatively set stop_event before re-raising
    3. Tracks all child PIDs via sat_children/rc2_children dicts
    4. On exit, terminates all children with SIGTERM, waits briefly, then SIGKILL
    
    Example
    -------
    >>> with SATProcessManager(parent_stop_event) as pm:
    ...     executor = ProcessPoolExecutor(max_workers=4)
    ...     # Submit SAT jobs, passing pm.sat_children, pm.rc2_children, pm.stop_event
    ...     # If interrupted, __exit__ ensures cleanup
    
    Attributes
    ----------
    stop_event : multiprocessing.Event
        Shared event; when set, workers should abort early.
    sat_children : dict
        Manager-backed dict mapping job_key -> PID for SAT processes.
    rc2_children : dict
        Manager-backed dict mapping job_key -> PID for RC2 processes.
    """
    
    def __init__(self, parent_stop_event: Optional[Any] = None):
        """
        Parameters
        ----------
        parent_stop_event : Event or None
            If provided, this event is checked on entry; if already set,
            the internal stop_event is pre-signaled.
        """
        self._parent_stop_event = parent_stop_event
        self._manager: Optional[mp.managers.SyncManager] = None
        self._pool_context = None
        self._original_sigterm = None
        self._original_sigint = None
        self._entered = False
        
        # Public attributes (populated in __enter__)
        self.stop_event: Optional[Any] = None
        self.sat_children: Optional[Dict[Any, int]] = None
        self.rc2_children: Optional[Dict[Any, int]] = None
    
    def __enter__(self) -> "SATProcessManager":
        global _active_managers
        
        # Get multiprocessing context - use spawn on macOS for safety
        self._pool_context = _get_safe_mp_context()
        
        # Create Manager for shared state
        # On macOS, spawn-context Manager() deadlocks — skip it and
        # fall back to plain dicts + threading.Event.
        if sys.platform == "darwin":
            self._manager = None
            self.sat_children = {}
            self.rc2_children = {}
            self.stop_event = _threading.Event()
        else:
            try:
                self._manager = self._pool_context.Manager()
                self.sat_children = self._manager.dict()
                self.rc2_children = self._manager.dict()
                self.stop_event = self._manager.Event()
            except RuntimeError:
                # Fall back to non-shared state (single-process mode)
                self._manager = None
                self.sat_children = {}
                self.rc2_children = {}
                self.stop_event = None
        
        # Propagate parent stop_event if already set
        if self._parent_stop_event is not None and self.stop_event is not None:
            try:
                if self._parent_stop_event.is_set():
                    self.stop_event.set()
            except Exception:
                pass
        
        # Install signal handlers for graceful shutdown
        self._install_signal_handlers()
        
        # Register self in global list
        _active_managers.append(self)
        self._entered = True
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        global _active_managers
        
        if not self._entered:
            return False
        
        # Remove from global registry
        try:
            _active_managers.remove(self)
        except ValueError:
            pass
        
        # Restore original signal handlers
        self._restore_signal_handlers()
        
        # Set stop_event to signal workers to abort
        if self.stop_event is not None:
            try:
                self.stop_event.set()
            except Exception:
                pass
        
        # Kill all tracked child processes
        self._kill_all_children()
        
        # Shut down the Manager
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception as e:
                wise_logger.debug("[SATProcessManager] Manager shutdown error: %r", e)
        
        self._entered = False
        return False  # Don't suppress exceptions
    
    def _install_signal_handlers(self) -> None:
        """Install signal handlers that set stop_event before re-raising."""
        def _handler(signum, frame):
            # Signal all active managers to stop
            for mgr in _active_managers:
                if mgr.stop_event is not None:
                    try:
                        mgr.stop_event.set()
                    except Exception:
                        pass
            # Re-raise as KeyboardInterrupt for SIGINT, SystemExit for SIGTERM
            if signum == signal.SIGTERM:
                raise SystemExit(128 + signum)
            elif signum == signal.SIGINT:
                raise KeyboardInterrupt()
        
        try:
            self._original_sigterm = signal.signal(signal.SIGTERM, _handler)
            self._original_sigint = signal.signal(signal.SIGINT, _handler)
        except (ValueError, OSError):
            # Signal handling not available (e.g., not main thread)
            pass
    
    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        try:
            if self._original_sigterm is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            if self._original_sigint is not None:
                signal.signal(signal.SIGINT, self._original_sigint)
        except (ValueError, OSError):
            pass
    
    def _kill_all_children(self) -> None:
        """Terminate all tracked child processes and their descendants."""
        all_pids: Set[int] = set()
        
        # Collect PIDs from tracking dicts
        for children_dict in [self.sat_children, self.rc2_children]:
            if children_dict:
                try:
                    all_pids.update(children_dict.values())
                except Exception:
                    pass
        
        # Also collect grandchildren (nested processes)
        for pid in list(all_pids):
            try:
                all_pids.update(_get_child_pids(pid))
            except Exception:
                pass
        
        if not all_pids:
            return
        
        wise_logger.debug("[SATProcessManager] Killing %d child processes (including descendants)", len(all_pids))
        
        # First pass: SIGTERM
        for pid in all_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        
        # Brief grace period
        time.sleep(0.3)
        
        # Second pass: SIGKILL for any survivors
        for pid in all_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass


# ---------------------------------------------------------------------------
# SAT Solver Configuration with Heuristics
# ---------------------------------------------------------------------------

@dataclass
class SATSolverConfig:
    """Fine-grained configuration for the SAT/MaxSAT solver.
    
    Consolidates all tuning parameters that control the per-patch SAT
    solving behavior. Use :meth:`from_heuristics` to compute sensible
    defaults based on grid dimensions.
    
    Attributes
    ----------
    max_sat_time : float
        Wall-clock timeout (seconds) for each Minisat22 sub-process.
    max_rc2_time : float
        Wall-clock timeout (seconds) for each RC2 (MaxSAT) sub-process.
    wB_col : int
        Soft-clause weight penalising boundary column placement.
    wB_row : int
        Soft-clause weight penalising boundary row placement.
    base_pmax_in : int or None
        Starting P_max for the binary search. None = auto (defaults to R).
    bt_soft : bool
        Use soft BT pinning via MaxSAT instead of hard constraints.
    capacity_steps : int
        Number of boundary capacity factor steps to enumerate.
    capacity_min : float
        Minimum boundary capacity factor (0.0 to 1.0).
    sat_workers : int
        Maximum parallel SAT worker processes.
    enable_maxsat_refinement : bool
        Run RC2 MaxSAT refinement after finding SAT solution.
    
    Example
    -------
    >>> config = SATSolverConfig.from_heuristics(n=6, m=4, k=2, R=3)
    >>> print(config.max_sat_time)  # Auto-scaled based on grid size
    5091.1...
    """
    max_sat_time: Optional[float] = None
    max_rc2_time: Optional[float] = None
    wB_col: int = 1
    wB_row: int = 1
    base_pmax_in: Optional[int] = None
    bt_soft: bool = False
    capacity_steps: int = 6
    capacity_min: float = 0.0
    sat_workers: int = 4
    enable_maxsat_refinement: bool = False
    
    @classmethod
    def from_heuristics(
        cls,
        n: int,
        m: int,
        k: int = 2,
        R: int = 1,
        num_pairs: int = 0,
        has_bt_pins: bool = False,
    ) -> "SATSolverConfig":
        """Compute solver configuration from grid dimensions using heuristics.
        
        Uses scaling rules derived from empirical testing to avoid wasting
        time on tiny grids or timing out on large ones.
        
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
        num_pairs : int
            Total number of ion pairs to route (across all rounds).
        has_bt_pins : bool
            Whether block-transfer pins are provided.
        
        Returns
        -------
        SATSolverConfig
            Configuration with heuristically-tuned parameters.
        """
        total_cells = n * m * k
        
        # base_pmax: at least R, scaled up for grids with many gates
        if num_pairs > 0:
            # More pairs may need more passes to route
            base_pmax = max(R, int(math.ceil(math.sqrt(num_pairs))))
        else:
            base_pmax = max(R, 1)
        
        # Derive per-subprocess timeout from the unified difficulty
        # estimator — scales with problem size, no hard ceiling.
        sat_time = _estimate_sat_timeout(n, m, total_cells, R, base_pmax)
        rc2_time = sat_time  # Same budget for MaxSAT
        
        # Capacity steps: fewer for small grids to reduce search space
        if total_cells <= 8:
            capacity_steps = 1
        elif total_cells <= 24:
            capacity_steps = 3
        else:
            capacity_steps = 6
        
        # Workers: scale with grid size, capped at 4
        if total_cells <= 8:
            workers = 1
        elif total_cells <= 24:
            workers = 2
        else:
            workers = 4
        
        # Auto-enable soft BT when pins exist
        bt_soft = has_bt_pins
        
        return cls(
            max_sat_time=sat_time,
            max_rc2_time=rc2_time,
            wB_col=1,
            wB_row=1,
            base_pmax_in=base_pmax,
            bt_soft=bt_soft,
            capacity_steps=capacity_steps,
            capacity_min=0.0,
            sat_workers=workers,
            enable_maxsat_refinement=False,
        )
    
    def to_kwargs(self) -> Dict[str, Any]:
        """Convert to keyword arguments for backward compatibility.
        
        Returns
        -------
        dict
            Parameters in the format expected by optimal_QMR_for_WISE.
        """
        return {
            "max_sat_time": self.max_sat_time,
            "max_rc2_time": self.max_rc2_time,
            "wB_col": self.wB_col,
            "wB_row": self.wB_row,
            "base_pmax_in": self.base_pmax_in,
            "bt_soft": self.bt_soft,
        }


WISE_LOGGER_NAME = "wise.qccd.sat"
wise_logger = logging.getLogger(WISE_LOGGER_NAME)
if not wise_logger.handlers:
    wise_logger.addHandler(logging.NullHandler())
wise_logger.propagate = True



@dataclass
class _WiseSatBuilderContext:
    """Immutable structural data shared by all SAT/MaxSAT workers for one patch.

    Created once in ``optimal_QMR_for_WISE`` and passed to every call of
    ``_wise_build_structural_cnf`` and ``_wise_sat_config_worker`` so that
    the grid geometry, pair lists, and block structure are never re-derived.

    Attributes
    ----------
    A_in : np.ndarray
        ``n × m`` ion-index grid (initial placement for this patch).
    BT : list of dict
        Per-round block-transfer pins ``{ion: (dest_row, dest_col)}``.
    P_arr : list of list of (int, int)
        Per-round MS-gate pairs local to this patch.
    full_P_arr : list of list of (int, int)
        Full (un-patched) pair list for diagnostics.
    ions : list of int
        Sorted set of distinct ion indices in *A_in*.
    n, m : int
        Grid rows and (block-width-normalised) columns.
    R : int
        Number of gate rounds.
    block_cells : list of list of (int, int)
        Per-block list of ``(row, col)`` cells.
    block_fully_inside : list of bool
        Whether each block lies entirely within the patch boundary.
    block_widths : list of int
        Width of each block (normally ``k``).
    num_blocks : int
        Total number of blocks.
    wB_col, wB_row : int
        Soft-clause weights for boundary column/row placement.
    debug_diag : bool
        Emit detailed diagnostic logging.
    """
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
    """Build a SAT / weighted-MaxSAT (WCNF) formula for the WISE ion schedule.

    Encodes a sorting-network-style model in which ions move through an
    ``n × m`` grid of traps over ``R`` rounds.  Each round is divided into
    up to ``P_bound`` passes where horizontal / vertical swap gates can fire,
    subject to odd-even constraints, pair-assignment constraints, and
    cardinality bounds on the total number of passes.

    Parameters
    ----------
    ctx : _WiseSatBuilderContext
        Shared structural data (grid, pairs, blocks, weights).
    P_bound : int
        Maximum number of passes per round (upper bound for binary search).
    sum_bound_B : int or None
        If given, a cardinality constraint on Σ_r P_r (total passes across
        all rounds starting from ``optimize_round_start``).
    use_wcnf : bool
        When ``True`` the formula is built as a WCNF (with soft clauses)
        rather than a plain CNF.  Required when ``add_boundary_soft`` or
        ``bt_soft_weight > 0``.
    add_boundary_soft : bool
        Add soft clauses penalising ions that end on patch boundaries.
    phase_label : str
        Human-readable label for log messages.
    optimize_round_start : int
        Skip pass-count optimisation for rounds before this index
        (used to ignore the initial reconfiguration round).
    boundary_adjacent : dict or None
        Which grid edges are on the physical boundary of the full chip
        (keys: ``"top"``, ``"bottom"``, ``"left"``, ``"right"``).
    cross_boundary_prefs : list or None
        Per-round dictionaries mapping ion ids to preferred boundary
        directions for cross-patch movement.
    boundary_capacity_factor : float
        Scaling factor for the capacity limit of boundary traps.
    bt_soft_weight : int
        Weight for soft clauses that prefer block-transfer moves.

    Returns
    -------
    tuple
        ``(formula, vpool, extra_info, ion_info, grp_sel, cardvars)``
        where *formula* is a ``CNF`` or ``WCNF`` instance and *vpool* is
        the ``IDPool`` mapping symbolic variable names to DIMACS ids.
    """
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

    def var_a(r: int, p: int, krow: int, jcol: int, ion: int) -> int:
        return vpool.id(("a", r, p, krow, jcol, ion))

    def var_s_h(r: int, p: int, krow: int, jcol: int) -> int:
        return vpool.id(("s_h", r, p, krow, jcol))

    def var_s_v(r: int, p: int, krow: int, jcol: int) -> int:
        return vpool.id(("s_v", r, p, krow, jcol))

    def var_phase(r: int, p: int) -> int:
        return vpool.id(("phase", r, p))

    def var_row_end(r: int, ion: int, d: int) -> int:
        return vpool.id(("row_end", r, ion, d))

    def var_w_end(r: int, ion: int, b: int) -> int:
        return vpool.id(("w_end", r, ion, b))

    def var_u(r: int, p: int) -> int:
        return vpool.id(("u", r, p))

    def is_reserved(r: int, ion: int) -> bool:
        return ion in BT[r]

    def ion_in_full_P_arr(r: int, ion: int) -> bool:
        return any((ion in g) for g in full_P_arr[r])

    def ion_in_minor_P_arr(r: int, ion: int) -> bool:
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

    P_bounds = [P_bound] * R

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
    """Decode a SAT model into a per-round list of pass dictionaries.

    Each pass dict has keys ``"phase"`` (``"H"`` or ``"V"``),
    ``"h_swaps"`` (list of ``(row, col)``), and ``"v_swaps"``.

    Parameters
    ----------
    model : list of int
        DIMACS-style model from the SAT solver (positive literals are true).
    vpool : IDPool
        Variable pool used when building the formula.
    n, m : int
        Grid dimensions.
    R : int
        Number of gate rounds.
    P_bound : int
        Maximum passes per round.
    ignore_initial_reconfig : bool
        Whether round 0 has an inflated pass budget.

    Returns
    -------
    list of list of dict
        ``schedule[r][p]`` is a pass-info dict for round *r*, pass *p*.
    """
    model_set = {lit for lit in model if lit > 0}
    P_bounds = [P_bound] * R

    def lit_true(v: int) -> bool:
        return v in model_set

    def var_s_h(r: int, p: int, krow: int, jcol: int) -> int:
        return vpool.id(("s_h", r, p, krow, jcol))

    def var_s_v(r: int, p: int, krow: int, jcol: int) -> int:
        return vpool.id(("s_v", r, p, krow, jcol))

    def var_phase(r: int, p: int) -> int:
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
    """Count how many passes are actually *used* per round in a solved model.

    A pass is "used" when its utilisation variable ``u(r, p)`` is true.

    Returns
    -------
    list of int
        ``per_round[r]`` = number of active passes in round *r*.
    """
    model_set = {lit for lit in model if lit > 0}
    P_bounds = [P_bound] * R

    def lit_true(v: int) -> bool:
        return v in model_set

    def var_u(r: int, p: int) -> int:
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
    inprocess_limit: Optional[int] = None,
    notebook_sat_timeout: Optional[float] = None,
    notebook_rc2_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a binary search for the tightest satisfiable schedule at a given P_max.

    Called once per ``(P_max, boundary_capacity_factor)`` configuration,
    typically dispatched by a ``ProcessPoolExecutor`` inside
    ``optimal_QMR_for_WISE``.

    The search bisects over the global Σ_r P_r bound to minimise
    total passes while remaining SAT.  At each iteration it calls
    ``_wise_build_structural_cnf`` to build the CNF/WCNF formula and
    then ``run_sat_with_timeout_file`` or ``run_rc2_with_timeout_file``
    to solve it in a child process.

    Parameters
    ----------
    cfg : tuple of (int, float)
        ``(P_max, boundary_capacity_factor)`` for this config point.
    context : _WiseSatBuilderContext
        Shared structural data for the current grid patch.
    optimize_round_start : int
        Skip pass optimisation for rounds before this index.
    max_sat_time, max_rc2_time : float
        Per-subprocess wall-clock timeout (seconds).
    boundary_adjacent : dict
        Which edges border the physical chip boundary.
    cross_boundary_prefs : list
        Per-round cross-boundary movement preferences.
    ignore_initial_reconfig : bool
        Whether the first round is a free reconfiguration.
    progress_path : str or None
        Path to write intermediate progress pickles.
    bt_soft_weight : int
        Weight for soft block-transfer clauses (0 = disabled).
    sat_children, rc2_children : dict or None
        Manager-backed dicts for PID registration of child processes.
    stop_event : multiprocessing.Event or None
        When set, the worker should abort early.

    Returns
    -------
    dict
        Result dictionary with keys ``"P_max"``, ``"status"``
        (``"sat"``/``"timeout"``/``"unsat"``), ``"schedule"``,
        ``"per_round_usage"``, ``"model"``, etc.
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
        except Exception as _pub_err:
            wise_logger.debug("Progress publish failed: %r", _pub_err)

    def _solve_with_bound(sum_bound_B: int, *, step_timeout: Optional[float] = None) -> Dict[str, Any]:
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

            # ── Adaptive per-step timeout ────────────────────────
            # When step_timeout is provided (bisection steps), cap the
            # solver timeout to avoid spending the full budget on a
            # single exploratory probe.  The feasibility check (loosest
            # bound) always uses the full timeout.
            _eff_sat_time = max_sat_time
            _eff_rc2_time = max_rc2_time
            _eff_nb_sat = notebook_sat_timeout
            _eff_nb_rc2 = notebook_rc2_timeout
            if step_timeout is not None:
                if _eff_sat_time is not None:
                    _eff_sat_time = min(_eff_sat_time, step_timeout)
                else:
                    _eff_sat_time = step_timeout
                if _eff_rc2_time is not None:
                    _eff_rc2_time = min(_eff_rc2_time, step_timeout)
                else:
                    _eff_rc2_time = step_timeout
                if _eff_nb_sat is not None:
                    _eff_nb_sat = min(_eff_nb_sat, step_timeout)
                if _eff_nb_rc2 is not None:
                    _eff_nb_rc2 = min(_eff_nb_rc2, step_timeout)

            cost_mid: Optional[int] = None
            worker_pid = os.getpid()
            job_base = (worker_pid, P_max, boundary_capacity_factor, sum_bound_B)
            if use_soft_bt:
                job_key = ("RC2",) + job_base
                model_mid, cost_mid, status_solver = run_rc2_with_timeout_file(
                    formula_mid,
                    timeout_s=_eff_rc2_time,
                    debug_prefix="[WISE-RC2]",
                    stop_event=stop_event,
                    job_key=job_key,
                    children_dict=rc2_children,
                    inprocess_limit=inprocess_limit,
                    notebook_rc2_timeout=_eff_nb_rc2,
                )
                sat_ok = status_solver == "ok" and model_mid is not None
            else:
                assumptions_mid = (
                    [-lit for lit in grp_sel_mid.values()] if grp_sel_mid else None
                )
                job_key = ("SAT",) + job_base

                sat_ok, model_mid, status_solver = run_sat_with_timeout_file(
                    formula_mid,
                    timeout_s=_eff_sat_time,
                    debug_prefix=None,
                    assumptions=assumptions_mid,
                    stop_event=stop_event,
                    job_key=job_key,
                    children_dict=sat_children,
                    inprocess_limit=inprocess_limit,
                    notebook_sat_timeout=_eff_nb_sat,
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

    # ------------------------------------------------------------------
    # Feasibility pre-check: test the loosest bound (max_bound_B) first.
    # If UNSAT even with maximum headroom, skip the entire binary search
    # — this P_max can never be feasible.  Saves ~log₂(max_bound_B)
    # iterations for every UNSAT config.
    # ------------------------------------------------------------------
    _feas_t0 = time.time()
    _feas_elapsed = 0.0
    if max_bound_B > 1:
        feas_result = _solve_with_bound(max_bound_B)
        _feas_elapsed = time.time() - _feas_t0
        last_result = feas_result
        if not (feas_result.get("status") == "ok"
                and feas_result.get("sat")
                and feas_result.get("model") is not None):
            wise_logger.debug(
                "[WISE] config P_max=%d cap=%.2f: UNSAT at loosest ΣP=%d — skipping bisection",
                P_max, boundary_capacity_factor, max_bound_B,
            )
            feas_result["sum_bound_B"] = max_bound_B
            publish(feas_result)
            if last_result is not None:
                return last_result
            return {
                "P_max": P_max,
                "boundary_capacity_factor": boundary_capacity_factor,
                "status": "unsat",
                "sat": False,
                "schedule": None,
                "per_round_usage": None,
                "model": None,
            }
        # Feasible at loosest bound — record as initial best and bisect
        # to find the tightest feasible bound.
        feas_result["sum_bound_B"] = max_bound_B
        best_result = feas_result
        publish(feas_result)
        high = max_bound_B - 1  # narrow search to [low, max_bound_B-1]
        wise_logger.debug(
            "[WISE] config P_max=%d cap=%.2f: SAT at loosest ΣP=%d — bisecting [%d,%d]",
            P_max, boundary_capacity_factor, max_bound_B, low, high,
        )

    # ------------------------------------------------------------------
    # Adaptive bisection budget
    # ------------------------------------------------------------------
    # The feasibility check (loosest bound) uses the full solver timeout.
    # For the bisection loop, we use adaptive timeouts:
    #   - Per-step cap: max(15s, 5× feasibility time) so easy problems
    #     get quick probes while hard problems get proportional headroom.
    #   - Total bisection budget: min(solver_timeout, 90s) to prevent
    #     spending hundreds of seconds on optimization when a feasible
    #     result is already in hand.
    # When a bisection step times out, it is treated as UNSAT (low rises),
    # which narrows toward the known-feasible upper bound.
    # ------------------------------------------------------------------
    _solver_timeout = max_rc2_time if use_soft_bt else max_sat_time
    _bisect_step_cap = max(15.0, min(5.0 * _feas_elapsed, _solver_timeout or 300.0))
    _BISECT_BUDGET_S = min(_solver_timeout or 300.0, 90.0)
    _bisect_t0 = time.time()

    while low <= high:
        if stop_event is not None and stop_event.is_set():
            wise_logger.debug(
                "[WISE] config P_max=%d cap=%.2f: stop_event set; aborting ΣP search",
                P_max,
                boundary_capacity_factor,
            )
            break
        # Check total bisection budget
        _bisect_elapsed = time.time() - _bisect_t0
        if _bisect_elapsed >= _BISECT_BUDGET_S:
            wise_logger.debug(
                "[WISE] config P_max=%d cap=%.2f: bisection budget exhausted "
                "(%.1fs >= %.1fs); returning best ΣP so far",
                P_max, boundary_capacity_factor,
                _bisect_elapsed, _BISECT_BUDGET_S,
            )
            break
        mid = (low + high) // 2

        wise_logger.debug(
            "[WISE] config P_max=%d cap=%.2f: trying ΣP<=%d (step_cap=%.0fs, budget_left=%.0fs)",
            P_max,
            boundary_capacity_factor,
            mid,
            _bisect_step_cap,
            max(0.0, _BISECT_BUDGET_S - _bisect_elapsed),
        )
        result = _solve_with_bound(mid, step_timeout=_bisect_step_cap)
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
    except Exception as _wr_err:
        wise_logger.debug("SAT worker result write failed: %r", _wr_err)

def run_sat_with_timeout_file(
    cnf: CNF,
    timeout_s: float,
    debug_prefix: str = "[WISE]",
    assumptions: Optional[List[int]] = None,
    stop_event: Optional[Any] = None,
    job_key: Optional[Any] = None,
    children_dict: Optional[Mapping[Any, int]] = None,
    *,
    inprocess_limit: Optional[int] = None,
    notebook_sat_timeout: Optional[float] = None,
):
    """
    Run Minisat22 on *cnf* in a separate process with a wall-clock timeout.

    For small CNFs (≤ ``_SAT_INPROCESS_CLAUSE_LIMIT`` clauses), the solver
    runs in-process to avoid macOS subprocess spawn overhead (~1.5-2 s per
    spawn).  Larger instances still use the subprocess path for timeout
    isolation.

    Returns
    -------
    (sat_ok, model, status)
        *status* is ``"ok"`` | ``"timeout"`` | ``"error"`` | ``"user_abort"``.
        On non-ok status, *sat_ok* and *model* are ``None``.
    """
    # --- In-process fast path ---
    # Use when: (1) in notebook (spawn context deadlocks), (2) inside
    # daemon process (can't spawn children), or (3) small CNF.
    # PySAT's C solver releases the GIL, so ThreadPoolExecutor
    # workers still get real parallelism.
    # A threading-based timeout prevents infinite hangs on hard instances.
    _is_pool_worker = mp.current_process().daemon
    _eff_inprocess_limit = inprocess_limit if inprocess_limit is not None else _SAT_INPROCESS_CLAUSE_LIMIT
    if _in_notebook_env() or _is_pool_worker or sys.platform == "darwin" or len(cnf.clauses) <= _eff_inprocess_limit:
        _ip_timeout = timeout_s if (timeout_s is not None and timeout_s > 0) else 60.0
        # Cap timeout in notebook to avoid multi-hour hangs on hard SAT instances.
        # Configured value wins; env var is fallback.
        if notebook_sat_timeout is not None:
            _NB_SAT_CAP = notebook_sat_timeout
        else:
            _NB_SAT_CAP = float(os.environ.get("WISE_NOTEBOOK_SAT_TIMEOUT", "120"))
        if (notebook_sat_timeout is not None or _in_notebook_env()) and _ip_timeout > _NB_SAT_CAP:
            _ip_timeout = _NB_SAT_CAP
        wise_logger.debug(
            "%s SAT in-process: clauses=%d, vars=%d, timeout=%.1fs",
            debug_prefix, len(cnf.clauses), cnf.nv, _ip_timeout,
        )
        try:
            sat = Minisat22(bootstrap_with=cnf.clauses)
            _ip_result = [None, None, "timeout"]
            _ip_exc_slot = [None]

            def _ip_sat_solve():
                try:
                    if assumptions is not None:
                        _ok = sat.solve(assumptions=assumptions)
                    else:
                        _ok = sat.solve()
                    _ip_result[0] = bool(_ok)
                    _ip_result[1] = sat.get_model() if _ok else None
                    _ip_result[2] = "ok"
                except Exception as _e:
                    _ip_exc_slot[0] = _e
                    _ip_result[2] = "error"

            _ip_t = _threading.Thread(target=_ip_sat_solve, daemon=True)
            _ip_t.start()
            _ip_t.join(timeout=_ip_timeout)
            if _ip_t.is_alive():
                wise_logger.debug(
                    "%s SAT in-process TIMEOUT after %.1fs (clauses=%d)",
                    debug_prefix, _ip_timeout, len(cnf.clauses),
                )
                try:
                    sat.interrupt()
                except Exception:
                    pass
                try:
                    sat.delete()
                except Exception:
                    pass
                return None, None, "timeout"
            sat.delete()
            if _ip_result[2] == "error" and _ip_exc_slot[0] is not None:
                raise _ip_exc_slot[0]
            return _ip_result[0], _ip_result[1], _ip_result[2]
        except Exception as _ip_exc:
            if sys.platform == "darwin" or _in_notebook_env():
                wise_logger.error(
                    "In-process SAT solve failed on macOS/notebook (%s); "
                    "cannot fall through to subprocess (spawn deadlock)",
                    _ip_exc,
                )
                raise
            wise_logger.warning(
                "In-process SAT solve failed (%s), falling through to subprocess",
                _ip_exc,
            )

    extra_args = (assumptions,) if assumptions is not None else ()

    data, status = _run_solver_in_subprocess(
        input_obj=cnf,
        input_filename="instance.cnf.pkl",
        result_filename="result_sat.pkl",
        worker_fn=_sat_worker_from_file,
        worker_extra_args=extra_args,
        timeout_s=timeout_s,
        solver_label="SAT",
        debug_prefix=debug_prefix,
        stop_event=stop_event,
        job_key=job_key,
        children_dict=children_dict,
        start_log_msg=(
            f"SAT worker starting (timeout={timeout_s:.1f}s) for CNF: clauses={len(cnf.clauses)}"
            if debug_prefix else None
        ),
        finish_log_fn=lambda d: (
            f"SAT worker finished in {d.get('time', 0.0):.3f}s, SAT={d.get('sat')}"
        ),
    )

    if status != "ok" or data is None:
        return None, None, status
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
    except Exception as _wr_err:
        wise_logger.debug("RC2 worker result write failed: %r", _wr_err)


# ---------------------------------------------------------------------------
# Common subprocess runner for SAT / RC2 solvers
# ---------------------------------------------------------------------------

def _run_solver_in_subprocess(
    *,
    input_obj: Any,
    input_filename: str,
    result_filename: str,
    worker_fn,
    worker_extra_args: tuple = (),
    timeout_s: float,
    solver_label: str,
    debug_prefix: str = "[WISE]",
    stop_event: Optional[Any] = None,
    job_key: Optional[Any] = None,
    children_dict: Optional[Mapping[Any, int]] = None,
    start_log_msg: Optional[str] = None,
    finish_log_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Shared subprocess runner for SAT and RC2 solvers.

    Serialises *input_obj* to a temp file, spawns an ``mp.Process`` running
    *worker_fn*, polls with timeout / stop_event / KeyboardInterrupt
    handling, reads back the pickled result, and cleans up the worker.

    Parameters
    ----------
    input_obj
        The object (CNF or WCNF) to serialise for the worker.
    input_filename, result_filename
        Filenames inside the temporary directory.
    worker_fn
        Callable used as ``target`` for ``mp.Process``.
        Called as ``worker_fn(input_path, result_path, *worker_extra_args)``.
    worker_extra_args
        Extra positional args forwarded after ``(input_path, result_path)``.
    timeout_s
        Wall-clock timeout in seconds (``<= 0`` → immediate "timeout").
    solver_label
        Short human-readable tag (``"SAT"`` / ``"RC2"``) used in log messages.
    debug_prefix
        Prefix string for logger output; ``None`` suppresses logs.
    stop_event
        ``multiprocessing.Event`` checked every 0.5 s; if set, treated as
        timeout.
    job_key, children_dict
        If both are not ``None``, the child PID is registered in
        ``children_dict[job_key]`` after ``p.start()`` and removed on
        exit.
    start_log_msg
        Optional one-line log message emitted right before ``p.start()``.
    finish_log_fn
        Optional callable ``(result_dict) -> str`` whose return value is
        logged on successful completion.

    Returns
    -------
    (result_data, status)
        *result_data* is the dict read from the worker's result file, or
        ``None`` on timeout / error / abort.
        *status* is one of ``"ok"``, ``"timeout"``, ``"error"``,
        ``"user_abort"``.
    """
    # ---- early exit if timeout disabled ----
    if timeout_s is not None and timeout_s <= 0:
        if debug_prefix:
            wise_logger.debug(
                "%s %s disabled (timeout <= 0); treating as timeout.",
                debug_prefix,
                solver_label,
            )
        return None, "timeout"

    def _unregister_child() -> None:
        if children_dict is not None and job_key is not None:
            try:
                children_dict.pop(job_key, None)
            except Exception as _unreg_err:
                wise_logger.debug(
                    "%s child unregister failed: %r", solver_label, _unreg_err
                )

    def _terminate_and_kill(p: mp.Process) -> None:
        """Best-effort terminate → join → kill escalation."""
        if not p.is_alive():
            return
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
            try:
                p.join(2.0)
            except Exception:
                pass

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, input_filename)
        result_path = os.path.join(tmpdir, result_filename)

        with open(input_path, "wb") as f:
            pickle.dump(input_obj, f)

        p = mp.Process(
            target=worker_fn,
            args=(input_path, result_path) + tuple(worker_extra_args),
        )
        p.daemon = True

        if debug_prefix and start_log_msg:
            wise_logger.debug("%s %s", debug_prefix, start_log_msg)

        # ---- start the child process ----
        try:
            p.start()
        except KeyboardInterrupt:
            if debug_prefix:
                wise_logger.debug(
                    "%s %s start interrupted by user; terminating worker and returning user_abort.",
                    debug_prefix,
                    solver_label,
                )
            _terminate_and_kill(p)
            _unregister_child()
            return None, "user_abort"

        # Register child PID for global tracking
        if children_dict is not None and job_key is not None:
            try:
                children_dict[job_key] = p.pid
            except Exception as _e:
                wise_logger.debug(
                    "%s child PID registration failed: %r", solver_label, _e
                )

        # ---- poll loop ----
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
                    "%s %s join interrupted by user after %.3fs; terminating worker (user_abort).",
                    debug_prefix,
                    solver_label,
                    elapsed,
                )
            _terminate_and_kill(p)
            _unregister_child()
            return None, "user_abort"

        # ---- handle timeout ----
        if status == "timeout":
            if debug_prefix:
                elapsed = time.time() - start
                wise_logger.debug(
                    "%s %s worker exceeded %.1fs (elapsed=%.3fs); terminating (timeout).",
                    debug_prefix,
                    solver_label,
                    timeout_s,
                    elapsed,
                )
            _terminate_and_kill(p)
            _unregister_child()
            return None, "timeout"

        # ---- read result file ----
        if not os.path.exists(result_path):
            if debug_prefix:
                wise_logger.debug(
                    "%s %s worker finished but produced no result file; treating as error.",
                    debug_prefix,
                    solver_label,
                )
            _unregister_child()
            return None, "error"

        try:
            with open(result_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            if debug_prefix:
                wise_logger.debug(
                    "%s %s worker result read error: %r; treating as error.",
                    debug_prefix,
                    solver_label,
                    e,
                )
            _unregister_child()
            return None, "error"

        if "error" in data:
            if debug_prefix:
                wise_logger.debug(
                    "%s %s worker raised: %s; treating as error.",
                    debug_prefix,
                    solver_label,
                    data["error"],
                )
            _unregister_child()
            return None, "error"

        # ---- success ----
        if debug_prefix and finish_log_fn is not None:
            wise_logger.debug("%s %s", debug_prefix, finish_log_fn(data))

        _unregister_child()
        return data, "ok"


def run_rc2_with_timeout_file(
    wcnf: WCNF,
    timeout_s: float,
    debug_prefix: str = "[WISE]",
    stop_event: Optional[Any] = None,
    job_key: Optional[Any] = None,
    children_dict: Optional[Mapping[Any, int]] = None,
    *,
    inprocess_limit: Optional[int] = None,
    notebook_rc2_timeout: Optional[float] = None,
):
    """
    Run RC2 on *wcnf* in a separate process with a wall-clock timeout.

    For small WCNFs (≤ ``_SAT_INPROCESS_CLAUSE_LIMIT`` hard clauses),
    the solver runs in-process to avoid subprocess spawn overhead.

    Returns
    -------
    (model, cost, status)
        *status* is ``"ok"`` | ``"timeout"`` | ``"error"`` | ``"user_abort"``.
        On non-ok status, *model* and *cost* are ``None``.
    """
    # --- In-process fast path ---
    # Use when: (1) in notebook (spawn context deadlocks), (2) inside
    # daemon process (can't spawn children), or (3) small WCNF.
    # PySAT's C solver releases the GIL, so ThreadPoolExecutor
    # workers still get real parallelism.
    # A threading-based timeout prevents infinite hangs on hard instances.
    _is_pool_worker = mp.current_process().daemon
    _eff_inprocess_limit = inprocess_limit if inprocess_limit is not None else _SAT_INPROCESS_CLAUSE_LIMIT
    if _in_notebook_env() or _is_pool_worker or sys.platform == "darwin" or len(wcnf.hard) <= _eff_inprocess_limit:
        _ip_timeout = timeout_s if (timeout_s is not None and timeout_s > 0) else 60.0
        # Cap timeout in notebook to avoid very long hangs on hard RC2 instances.
        # Configured value wins; env var is fallback.
        if notebook_rc2_timeout is not None:
            _NB_RC2_CAP = notebook_rc2_timeout
        else:
            _NB_RC2_CAP = float(os.environ.get("WISE_NOTEBOOK_RC2_TIMEOUT", "180"))
        if (notebook_rc2_timeout is not None or _in_notebook_env()) and _ip_timeout > _NB_RC2_CAP:
            _ip_timeout = _NB_RC2_CAP
        wise_logger.debug(
            "%s RC2 in-process: hard=%d, soft=%d, vars=%d, timeout=%.1fs",
            debug_prefix, len(wcnf.hard), len(wcnf.soft), wcnf.nv, _ip_timeout,
        )
        try:
            rc2 = RC2(wcnf)
            _ip_result = [None, None, "timeout"]
            _ip_exc_slot = [None]

            def _ip_rc2_solve():
                try:
                    _model = rc2.compute()
                    _ip_result[0] = _model
                    _ip_result[1] = rc2.cost if _model is not None else None
                    _ip_result[2] = "ok"
                except Exception as _e:
                    _ip_exc_slot[0] = _e
                    _ip_result[2] = "error"

            _ip_t = _threading.Thread(target=_ip_rc2_solve, daemon=True)
            _ip_t.start()
            _ip_t.join(timeout=_ip_timeout)
            if _ip_t.is_alive():
                wise_logger.debug(
                    "%s RC2 in-process TIMEOUT after %.1fs (hard=%d)",
                    debug_prefix, _ip_timeout, len(wcnf.hard),
                )
                try:
                    rc2.delete()
                except Exception:
                    pass
                return None, None, "timeout"
            rc2.delete()
            if _ip_result[2] == "error" and _ip_exc_slot[0] is not None:
                raise _ip_exc_slot[0]
            return _ip_result[0], _ip_result[1], _ip_result[2]
        except Exception as _ip_exc:
            if sys.platform == "darwin" or _in_notebook_env():
                wise_logger.error(
                    "In-process RC2 solve failed on macOS/notebook (%s); "
                    "cannot fall through to subprocess (spawn deadlock)",
                    _ip_exc,
                )
                raise
            wise_logger.warning(
                "In-process RC2 solve failed (%s), falling through to subprocess",
                _ip_exc,
            )

    data, status = _run_solver_in_subprocess(
        input_obj=wcnf,
        input_filename="instance.wcnf.pkl",
        result_filename="result_rc2.pkl",
        worker_fn=_rc2_worker_from_file,
        timeout_s=timeout_s,
        solver_label="RC2",
        debug_prefix=debug_prefix,
        stop_event=stop_event,
        job_key=job_key,
        children_dict=children_dict,
        start_log_msg=(
            f"RC2 worker starting (timeout={timeout_s:.1f}s) for WCNF: "
            f"vars={wcnf.nv}, hard={len(wcnf.hard)}, soft={len(wcnf.soft)}"
            if debug_prefix else None
        ),
        finish_log_fn=lambda d: (
            f"RC2 worker finished in {d.get('time', 0.0):.3f}s, opt_cost={d.get('cost')}"
        ),
    )

    if status != "ok" or data is None:
        return None, None, status
    return data.get("model"), data.get("cost"), "ok"


# ---------------------------------------------------------------------------
# Helper functions extracted from optimal_QMR_for_WISE
# ---------------------------------------------------------------------------


def _wise_apply_time_env(default_value: Optional[float], env_var: str) -> Optional[float]:
    """Apply environment variable as fallback when no explicit value is configured.

    Priority order:
      1. Explicit configured value (``default_value``) — always wins when set.
      2. Environment variable — used only when ``default_value`` is ``None``
         (i.e. the user did not configure a timeout).

    This ensures that ``WISERoutingConfig(timeout_seconds=X)`` or
    ``WISESolverParams(max_sat_time=X)`` always means exactly *X* seconds,
    without being silently clamped by an env var.
    """
    # If the caller explicitly configured a value, respect it.
    if default_value is not None and default_value > 0:
        return default_value

    # No explicit config — check env var as fallback.
    env_val = os.environ.get(env_var)
    if not env_val:
        return default_value          # stays None (auto-compute later)
    try:
        parsed = float(env_val)
    except ValueError:
        return default_value
    if parsed <= 0:
        return default_value
    return parsed


def _wise_normalize_inputs(
    BT: Optional[List[Dict[int, Tuple[int, int]]]],
    boundary_adjacent: Optional[Dict[str, bool]],
    cross_boundary_prefs: Optional[List[Dict[int, Set[str]]]],
    R: int,
    bt_soft: bool,
    wB_col: int,
    wB_row: int,
    base_pmax_in: Optional[int],
    prev_pmax: Optional[int],
    grid_origin: Optional[Tuple[int, int]],
    max_sat_time: Optional[float],
    max_rc2_time: Optional[float],
    debug_diag: bool,
) -> Tuple[
    List[Dict[int, Tuple[int, int]]],  # BT
    Dict[str, bool],                    # boundary_adjacent
    List[Dict[int, Set[str]]],          # cross_boundary_prefs
    bool,                               # bt_soft_enabled
    int,                                # bt_soft_weight_value
    int,                                # base_pmax
    int,                                # prev_pmax
    int,                                # row_offset
    int,                                # col_offset
    Optional[float],                    # max_sat_time
    Optional[float],                    # max_rc2_time
]:
    """Normalize and apply defaults to optimal_QMR_for_WISE inputs.

    Returns
    -------
    tuple
        Normalized versions of all input parameters plus derived values.
    """
    max_sat_time = _wise_apply_time_env(max_sat_time, "WISE_MAX_SAT_TIME")
    max_rc2_time = _wise_apply_time_env(max_rc2_time, "WISE_MAX_RC2_TIME")

    if BT is None:
        BT = [{} for _ in range(R)]

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
        # BT soft weight: high enough to strongly prefer BT pin
        # satisfaction, but not so high that it distorts the solver's
        # cost landscape.  Infeasible pins are handled by the
        # iterative pin-drop retry in the caller.
        bt_soft_weight_value = max(10000, base_pref_weight * 100)
        if debug_diag:
            wise_logger.info(
                "[WISE] soft BT enabled: weight=%d", bt_soft_weight_value
            )

    if base_pmax_in is None:
        base_pmax_in = 1          # always start from P_max=1 to explore tight (optimal) configs
    base_pmax = max(base_pmax_in, 1)

    if prev_pmax is None:
        prev_pmax = 0

    row_offset = 0
    col_offset = 0
    if grid_origin is not None:
        row_offset, col_offset = grid_origin

    return (
        BT,
        boundary_adjacent,
        cross_boundary_prefs,
        bt_soft_enabled,
        bt_soft_weight_value,
        base_pmax,
        prev_pmax,
        row_offset,
        col_offset,
        max_sat_time,
        max_rc2_time,
    )


def _wise_compute_ion_positions(
    A_in: np.ndarray,
    P_arr: List[List[Tuple[int, int]]],
    BT: List[Dict[int, Tuple[int, int]]],
    active_ions: Optional[Set[int]],
    debug_diag: bool,
) -> Tuple[
    Dict[int, int],  # row_of
    Dict[int, int],  # col_of
    Set[int],        # ions_all
    Set[int],        # active_ions
    Set[int],        # spectator_ions
]:
    """Compute ion position mappings and classify active vs spectator ions.

    Returns
    -------
    tuple
        (row_of, col_of, ions_all, active_ions, spectator_ions)
    """
    n, m = A_in.shape
    R = len(P_arr)

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

    if debug_diag:
        wise_logger.debug(
            "[WISE] ions_all=%d, active_ions=%d, spectator_ions=%d",
            len(ions_all),
            len(active_ions),
            len(spectator_ions),
        )

    return row_of, col_of, ions_all, active_ions, spectator_ions


def _wise_validate_bt_preconditions(
    BT: List[Dict[int, Tuple[int, int]]],
    P_arr: List[List[Tuple[int, int]]],
    ions_all: Set[int],
    row_of: Dict[int, int],
    n: int,
    k: int,
    bt_soft_enabled: bool,
    skip_check_c: bool = False,
) -> None:
    """Validate BT preconditions, raise ValueError on hard conflicts.

    Checks:
    (a) No two ions pinned to the same (d,c) in a round
    (b) Pair vs BT conflicts: same round, incompatible BT rows/blocks
    (c) Round-0: same source row & same target column among reserved ions
        (skipped when *skip_check_c* is True — the caller guarantees
        enough P_max to handle routing around collisions)
    (d) Column oversubscription from BT
    """
    R = len(P_arr)

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
    #     Skipped when skip_check_c is True — caller guarantees enough P_max
    #     to handle routing around (source_row, target_col) collisions.
    if len(BT) >= 1 and not skip_check_c:
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


def _wise_compute_block_geometry(
    n: int,
    m: int,
    col_offset: int,
    CAPACITY: int,
) -> Tuple[
    List[List[Tuple[int, int]]],  # block_cells
    List[bool],                   # block_fully_inside
    List[int],                    # block_widths
    int,                          # num_blocks
    int,                          # first_block_idx
]:
    """Compute block geometry for the grid patch.

    Returns
    -------
    tuple
        (block_cells, block_fully_inside, block_widths, num_blocks, first_block_idx)
    """
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

    return block_cells, block_fully_inside, block_widths, num_blocks, first_block_idx


def _wise_compute_outer_pairs(
    P_arr: List[List[Tuple[int, int]]],
    full_P_arr: List[List[Tuple[int, int]]],
    ions_set: Set[int],
    R: int,
) -> Optional[List[List[Tuple[int, int]]]]:
    """Compute pairs that are in full_P_arr but not in P_arr (outer pairs).

    Returns
    -------
    list or None
        Per-round list of outer pairs, or None if full_P_arr is empty.
    """
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


def _wise_debug_boundary_stats(
    label: str,
    model: Iterable[int],
    vpool,
    var_a: Callable[[int, int, int, int, int], int],
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


def _wise_enumerate_pmax_configs(
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

    # Interleave P_max from both ends: high (easy/fast SAT) and low
    # (hard but optimal if SAT).  With parallel workers both extremes
    # are explored simultaneously.  With a limited sequential budget
    # the early high-P_max configs guarantee *some* feasible solution
    # while low-P_max configs can still discover the true optimum.
    pmax_values = list(range(P_min, P_max_limit + 1, max(1, step)))
    lo, hi = 0, len(pmax_values) - 1
    interleaved: list[int] = []
    while lo <= hi:
        interleaved.append(pmax_values[hi])   # easy first
        if lo != hi:
            interleaved.append(pmax_values[lo])  # then hard
        lo += 1
        hi -= 1
    for P_max in interleaved:
        for factor in factors:
            yield (P_max, factor)


def _wise_estimate_reconfig_cost(schedule: Optional[List[List[Dict[str, Any]]]]) -> float:
    """Estimate total reconfiguration time from a SAT schedule.

    Counts the number of active H (horizontal / row-swap) and V (vertical /
    column-swap) passes and weights them by their physical cost.

    Physical time per pass type (from ``DEFAULT_CALIBRATION``):

    * **H pass** (row swap): Move + Merge + Rotation + Split + Move
    * **V pass** (col swap): 2×Junction + (4×Junction + Move) × 2

    V passes are ~2.4× more expensive than H passes, so raw pass counts
    are a poor proxy for reconfiguration time.

    Parameters
    ----------
    schedule : list of list of dict, or None
        ``schedule[r][p]`` with keys ``"phase"``, ``"h_swaps"``, ``"v_swaps"``.

    Returns
    -------
    float
        Estimated reconfiguration time in seconds.  Returns ``1e9`` when
        the schedule is ``None``.
    """
    if schedule is None:
        return 1e9

    cal = DEFAULT_CALIBRATION
    # H pass: Move + Merge + CrystalRotation + Split + Move
    row_swap_time = (
        cal.shuttle_time + cal.merge_time + cal.rotation_time
        + cal.split_time + cal.shuttle_time
    )
    # V pass: 2×Junction + (4×Junction + Move) × 2
    col_swap_time = (2 * cal.junction_time) + (
        4 * cal.junction_time + cal.shuttle_time
    ) * 2

    estimated = 0.0
    for round_passes in schedule:
        has_any_pass = False
        for pass_info in round_passes:
            h_active = bool(pass_info.get("h_swaps"))
            v_active = bool(pass_info.get("v_swaps"))
            if h_active:
                estimated += row_swap_time
                has_any_pass = True
            if v_active:
                estimated += col_swap_time
                has_any_pass = True
        # Each reconfiguration step adds one Split phase (Phase A)
        if has_any_pass:
            estimated += cal.split_time

    return estimated


def _wise_score_config(
    res: Dict[str, Any],
    optimize_round_start: int,
    R: int,
) -> Tuple[float, float, int]:
    """Score a SAT config result for selection (lower is better).

    Uses estimated reconfiguration time (weighted H/V pass cost) as the
    primary metric instead of raw pass count.  This correctly prioritises
    schedules with fewer expensive V (column/junction) passes.

    The tiebreaker prefers *lower* P_max so that the schedule uses the
    tightest feasible pass budget, matching the old module's behaviour.

    Returns
    -------
    tuple
        ``(-boundary_capacity_factor, estimated_reconfig_time, P_max)``
        for sorting.
    """
    schedule = res.get("schedule")
    if schedule is not None and optimize_round_start > 0 and R > 1:
        schedule = schedule[optimize_round_start:]
    estimated_reconfig = _wise_estimate_reconfig_cost(schedule)
    return (
        -res["boundary_capacity_factor"],
        estimated_reconfig,
        res["P_max"]                       # tiebreaker: prefer lower (tighter) P_max
    )


def _wise_assert_bt_consistency(
    layouts: List[np.ndarray],
    BT: List[Dict[int, Tuple[int, int]]],
    R: int,
    n: int,
    m: int,
    logger,
) -> Tuple[bool, Set[int]]:
    """
    Check BT pin consistency against layouts.

    Returns (ok, failed_ions) where failed_ions is the set of ions
    whose BT pins were not satisfied.
    """
    if BT is None or len(BT) == 0:
        return True, set()

    ok = True
    failed_ions: Set[int] = set()

    if len(layouts) < R:
        logger.error(
            "[WISE] BT consistency check: expected at least %d layouts, got %d",
            R, len(layouts),
        )
        return False, set()

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
                logger.debug(
                    "[WISE] BT consistency mismatch at round %d: expected ion %d at (d=%d, c=%d), but found %d",
                    r, ion, d, c, found,
                )
                ok = False
                failed_ions.add(ion)

    return ok, failed_ions


def optimal_QMR_for_WISE(
    A_in: np.ndarray,
    P_arr: List[List[Tuple[int, int]]],
    *,
    k: int,
    BT: List[Dict[int, Tuple[int, int]]] = None,
    wB_col: int = 1,
    wB_row: int = 1,
    max_rc2_time: Optional[float] = None,
    max_sat_time: Optional[float] = None,
    active_ions: Set[int] = None,
    full_P_arr: List[List[Tuple[int, int]]]=[],
    ignore_initial_reconfig: bool = False,
    base_pmax_in: int = None,
    prev_pmax: int = None,
    grid_origin: Tuple[int, int] = (0, 0),
    boundary_adjacent: Optional[Dict[str, bool]] = None,
    cross_boundary_prefs: Optional[List[Dict[int, Set[str]]]] = None,
    bt_soft: bool = False,
    parent_stop_event: Optional[Any] = None,
    progress_callback: Optional[Callable] = None,
    max_inner_workers: int | None = None,
    skip_bt_check_c: bool = False,
    solver_params: Optional[Any] = None,
    debug_diag: Optional[bool] = None,
) -> Tuple[List[np.ndarray], List[List[Dict[str, Any]]], int]:
    """Find the minimum-pass ion schedule for a (possibly patched) grid.

    Orchestrates a parallel search over ``(P_max, capacity_factor)``
    configurations using a ``ProcessPoolExecutor``.  Each worker
    (``_wise_sat_config_worker``) bisects over the total-pass bound.
    The best feasible schedule (lowest execution time) is selected from
    the completed workers.

    Parameters
    ----------
    A_in : np.ndarray
        ``n × m`` integer grid giving the initial ion placement.
    P_arr : list of list of (int, int)
        Per-round MS-gate pairs ``[(ion_a, ion_b), ...]``.
    k : int
        Block width (ions per trap).
    BT : list of dict, optional
        Per-round block-transfer map ``{ion: (dest_row, dest_col)}``.
    wB_col, wB_row : int
        Soft-clause weights for boundary placement penalties.
    max_sat_time, max_rc2_time : float
        Per-subprocess wall-clock timeout (seconds).
    active_ions : set of int, optional
        Ions participating in this routing window.
    full_P_arr : list, optional
        Full (un-patched) pair list used for diagnostics.
    ignore_initial_reconfig : bool
        Treat round 0 as a free reconfiguration (no pass cost).
    base_pmax_in : int or None
        Starting P_max; ``None`` defaults to ``R``.
    prev_pmax : int or None
        Best P_max from a prior solve (for warm-starting).
    grid_origin : tuple of (int, int)
        ``(row, col)`` origin of this patch in the full grid.
    boundary_adjacent : dict or None
        Chip-edge adjacency flags for this patch.
    cross_boundary_prefs : list or None
        Per-round cross-boundary movement preferences.
    bt_soft : bool
        Enable soft block-transfer clauses in the WCNF.
    parent_stop_event : multiprocessing.Event or None
        Cross-level cancellation signal; set by the outer search.
    progress_callback : callable or None
        ``(RoutingProgress) -> None`` for live progress updates.

    Returns
    -------
    tuple of (layouts, schedule, P_horizon)
        *layouts*: list of ``n × m`` ndarrays (one per round + final).
        *schedule*: per-round list of pass dicts with swap directions.
        *P_horizon*: the optimised P_max value.

    Raises
    ------
    NoFeasibleLayoutError
        If no satisfiable schedule is found within the time budget.
    """
    DEBUG_DIAG = debug_diag if debug_diag is not None else (os.environ.get("WISE_DEBUG_DIAG", "") == "1")
    DEBUG_DIAG_DETAILED = False

    A_in = np.asarray(A_in, dtype=int)

    # --- Sparse-grid padding: replace 0 (empty sentinel) cells with
    # unique dummy ion IDs.  The WISE permutation model requires every
    # cell to hold a distinct element; multiple zeros create an immediate
    # UNSAT because the CNF forces "ion 0" into all empty cells while
    # also requiring each ion to occupy exactly one cell.
    # Dummy IDs are negative to avoid collision with real 1-based ions.
    _dummy_ids: set = set()
    _next_dummy = -1
    n_tmp, m_tmp = A_in.shape
    for _r in range(n_tmp):
        for _c in range(m_tmp):
            if A_in[_r, _c] == 0:
                A_in[_r, _c] = _next_dummy
                _dummy_ids.add(_next_dummy)
                _next_dummy -= 1

    n, m = A_in.shape
    R = len(P_arr)

    # When R == 1 and ignore_initial_reconfig is set, the optimization
    # loop ``range(1, 1)`` is empty — no rounds are optimised at all and
    # the P-variable assignments for round 0 are unconstrained garbage.
    # Force ignore_initial_reconfig=False so round 0 is fully optimised.
    if R <= 1 and ignore_initial_reconfig:
        ignore_initial_reconfig = False

    optimize_round_start = 1 if (ignore_initial_reconfig and R > 0) else 0

    if len(full_P_arr) == 0:
        full_P_arr = P_arr

    # --- Normalize inputs via helper ---
    (
        BT,
        boundary_adjacent,
        cross_boundary_prefs,
        bt_soft_enabled,
        bt_soft_weight_value,
        base_pmax,
        prev_pmax,
        row_offset,
        col_offset,
        max_sat_time,
        max_rc2_time,
    ) = _wise_normalize_inputs(
        BT=BT,
        boundary_adjacent=boundary_adjacent,
        cross_boundary_prefs=cross_boundary_prefs,
        R=R,
        bt_soft=bt_soft,
        wB_col=wB_col,
        wB_row=wB_row,
        base_pmax_in=base_pmax_in,
        prev_pmax=prev_pmax,
        grid_origin=grid_origin,
        max_sat_time=max_sat_time,
        max_rc2_time=max_rc2_time,
        debug_diag=DEBUG_DIAG,
    )

    CAPACITY = k

    # --- Compute ion positions via helper ---
    row_of, col_of, ions_all, active_ions, spectator_ions = _wise_compute_ion_positions(
        A_in=A_in,
        P_arr=P_arr,
        BT=BT,
        active_ions=active_ions,
        debug_diag=DEBUG_DIAG,
    )

    # --- Validate BT preconditions via helper ---
    _wise_validate_bt_preconditions(
        BT=BT,
        P_arr=P_arr,
        ions_all=ions_all,
        row_of=row_of,
        n=n,
        k=k,
        bt_soft_enabled=bt_soft_enabled,
        skip_check_c=skip_bt_check_c,
    )

    ions = sorted(ions_all)
    ions_set = set(ions)

    # --- Auto-compute per-subprocess timeouts from problem dimensions ---
    if max_sat_time is None:
        max_sat_time = _estimate_sat_timeout(n, m, len(ions_set), R, base_pmax)
    if max_rc2_time is None:
        max_rc2_time = max_sat_time

    # --- Compute block geometry via helper ---
    block_cells, block_fully_inside, block_widths, num_blocks, first_block_idx = (
        _wise_compute_block_geometry(
            n=n,
            m=m,
            col_offset=col_offset,
            CAPACITY=CAPACITY,
        )
    )

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

    # --- Compute outer pairs via helper ---
    outer_pairs = _wise_compute_outer_pairs(
        P_arr=P_arr,
        full_P_arr=full_P_arr,
        ions_set=ions_set,
        R=R,
    )

    # --- Generate configs for SAT pool ---

    # Raise the P_max floor when low values are guaranteed UNSAT.
    # With many gates per round, P_max=1 (or similarly tiny values)
    # can never produce a feasible schedule — the solver wastes time
    # building and rejecting 784K-clause CNFs.  Use the max per-round
    # gate count as a sensible floor.
    _max_gates_per_round = max((len(rp) for rp in P_arr), default=0)
    if _max_gates_per_round > 1:
        base_pmax = max(base_pmax, _max_gates_per_round)

    # ── Pure-routing detection ────────────────────────────────────
    # When ALL rounds have zero MS-gate pairs, the SAT problem is
    # pure permutation routing: no pair-colocation constraints, no
    # V-penalty benefit.  We exploit this to:
    #   (a) raise base_pmax to max(n, m) — the minimum passes needed
    #       for odd-even transposition sorting of an n×m grid;
    #   (b) collapse capacity variants (boundary capacity has no
    #       interaction with pairs);
    #   (c) cap pool budget tightly (routing-only problems converge
    #       fast once pmax is adequate);
    #   (d) skip V-penalty MaxSAT (no pairs → H/V ratio irrelevant).
    _pure_routing = all(len(rp) == 0 for rp in P_arr)
    if _pure_routing and R >= 2:
        base_pmax = max(base_pmax, max(n, m))
        if DEBUG_DIAG:
            wise_logger.info(
                "[WISE] pure-routing mode: raised base_pmax to %d (n=%d, m=%d)",
                base_pmax, n, m,
            )

    # --- Fix E: Warm-start from prev_pmax to skip known-UNSAT configs ---
    # If a previous solve in this routing sequence found P_max=N, then
    # P_max < N-1 is very likely UNSAT for similar sub-problems.
    # Use prev_pmax - 1 as a floor to avoid wasted SAT attempts.
    if prev_pmax is not None and prev_pmax > 1:
        base_pmax = max(base_pmax, prev_pmax - 1)

    # Upper limit: base_pmax + max(n, m).  Odd-even transposition sort
    # needs at most max(n, m) extra passes beyond the base floor to
    # route any permutation.  The old formula (base + n + m) was overly
    # generous and produced many unnecessary SAT configs.
    # IMPORTANT: must be computed AFTER base_pmax adjustments above so
    # that limit_pmax >= base_pmax always holds.
    limit_pmax = max(base_pmax + max(n, m), R + max(n, m))

    # For pure R>=2 routing, the solver only needs enough passes to sort
    # the permutation.  An odd-even transposition network on a grid
    # of max(n,m) cells needs at most max(n,m) passes per phase.
    # When skip_bt_check_c is set, (source_row, target_col) collisions
    # require extra passes to route around — use a wider limit.
    # Gate on R >= 2: R=1 patches with empty pairs still need full
    # pmax range for boundary-adjacent capacity constraints.
    if _pure_routing and R >= 2:
        if skip_bt_check_c:
            # BT check (c) bypassed — ions can collide on (start_row,
            # target_col) which sometimes needs extra passes.  But with
            # BT[0]={} the first round is unconstrained, so the solver
            # can stage ions freely.  Cap to base_pmax + 2.
            limit_pmax = min(limit_pmax, base_pmax + 2)
        else:
            limit_pmax = min(limit_pmax, base_pmax + 1)

    # When no boundary is adjacent (full-grid patch), the capacity factor
    # has no effect on the CNF — all values produce identical formulas.
    # Collapse to a single factor to avoid redundant SAT calls.
    has_boundary = any(boundary_adjacent.get(d, False) for d in ("top", "bottom", "left", "right"))
    eff_capacity_steps = 6 if has_boundary else 1

    # For pure R>=2 routing, boundary capacity has no interaction
    # with pair constraints — collapse to a single variant.
    # Gate on R >= 2: R=1 patches need capacity variants for boundary.
    if _pure_routing and R >= 2:
        eff_capacity_steps = 1

    # For large round counts the SAT problems are already very heavy.
    # Reduce capacity variants to avoid spawning 18+ parallel instances
    # that each build 700K+ clause CNFs and thrash CPU / memory.
    if R > 8 and eff_capacity_steps > 2:
        eff_capacity_steps = 2

    configs = list(
        _wise_enumerate_pmax_configs(
            base_pmax,
            limit_pmax,
            step=max(int(np.floor((limit_pmax - base_pmax) / 8)), 1),
            capacity_steps=eff_capacity_steps,
            capacity_min=0.0,
        )
    )
    if not configs:
        raise NoFeasibleLayoutError("No feasible layout: empty SAT configuration set.")

    if DEBUG_DIAG:
        wise_logger.info(
            "[WISE] launching SAT pool over %d configs (P_max∈[%d, %d])  has_boundary=%s  base_pmax=%d",
            len(configs),
            base_pmax,
            limit_pmax,
            has_boundary,
            base_pmax,
        )
        import sys as _sys
        print(f"[DIAG] configs={len(configs)}, P_max range=[{base_pmax},{limit_pmax}], "
              f"has_boundary={has_boundary}, eff_cap_steps={eff_capacity_steps}, "
              f"step={max(int(np.floor((limit_pmax - base_pmax) / 4)), 1)}, "
              f"n={n}, m={m}, R={R}", file=_sys.stderr)

    # Use spawn on macOS for proper cleanup, fork on Linux for speed.
    # Mixing spawn (outer) and fork (inner) causes zombie processes on macOS.
    pool_context = _get_safe_mp_context()

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
    # the SAT pool is allowed to spawn.  This prevents runaway process
    # creation when _optimal_QMR_for_WISE is called from within an
    # outer ProcessPoolExecutor.
    max_workers = _wise_safe_sat_pool_workers(len(configs))
    # When the outer pool passes a coordinated budget, honour it so
    # total process count stays bounded across all nesting levels.
    if max_inner_workers is not None and max_inner_workers > 0:
        max_workers = min(max_workers, max_inner_workers)
    
    # On macOS with ThreadPoolExecutor, limit workers to reduce GIL contention.
    # PySAT's add_clause/append holds the GIL during CNF building, but the
    # actual solve() call RELEASES the GIL.  This means:
    # - CNF building is serialized (only 1 thread can build at a time)
    # - SAT solving is truly parallel (multiple threads can solve simultaneously)
    # With 4 threads, we get good pipelining: while threads A,B solve their
    # CNFs (GIL released), threads C,D build their CNFs (1 at a time).
    # Fewer threads = less GIL contention during build, but less solve parallelism.
    # More threads = more solve parallelism, but more GIL contention during build.
    # 4 is a good balance for typical WISE problems where solve >> build time.
    if sys.platform == "darwin":
        # Configured value wins; env var is fallback.
        _sp_macos_cap = getattr(solver_params, 'macos_thread_cap', None) if solver_params else None
        if _sp_macos_cap is not None:
            _MACOS_THREAD_CAP = _sp_macos_cap
        else:
            _MACOS_THREAD_CAP = int(os.environ.get("WISE_MACOS_THREAD_CAP", "4"))
        if max_workers > _MACOS_THREAD_CAP:
            wise_logger.debug(
                "[WISE] macOS: capping ThreadPool workers from %d to %d "
                "(PySAT GIL contention mitigation)",
                max_workers, _MACOS_THREAD_CAP,
            )
            max_workers = _MACOS_THREAD_CAP

    # ── Resolve solver_params overrides for inner functions ─────
    _sp_inprocess_limit = getattr(solver_params, 'inprocess_limit', None) if solver_params else None
    _sp_nb_sat_timeout = getattr(solver_params, 'notebook_sat_timeout', None) if solver_params else None
    _sp_nb_rc2_timeout = getattr(solver_params, 'notebook_rc2_timeout', None) if solver_params else None
    _sp_pool_budget_floor = getattr(solver_params, 'pool_budget_floor', None) if solver_params else None
    _sp_pool_budget_mult = getattr(solver_params, 'pool_budget_mult', None) if solver_params else None

    # ── Cap outer timeouts when notebook/per-call timeout is configured ──
    # Without this cap, the pool budget can balloon to 2*max_sat_time
    # (e.g. 2592s) even when the user wants 30s per SAT call.
    if _sp_nb_sat_timeout is not None and max_sat_time is not None:
        max_sat_time = min(max_sat_time, _sp_nb_sat_timeout)
    if _sp_nb_rc2_timeout is not None and max_rc2_time is not None:
        max_rc2_time = min(max_rc2_time, _sp_nb_rc2_timeout)

    results: List[Dict[str, Any]] = []
    solver_timeout = max_rc2_time if bt_soft_enabled else max_sat_time

    # Compute difficulty once for this invocation — used for pool budget
    # floor and adaptive sequential per-config caps.
    _difficulty = _estimate_sat_difficulty(n, m, len(ions), R, base_pmax)

    global_budget_s = (
        solver_timeout * 2.0 if (solver_timeout is not None and solver_timeout > 0) else None
    )
    # Safety-net: difficulty-aware floor so small demos don't stall but
    # hard problems get enough headroom.
    # No ceiling: solver_timeout already scales with difficulty.
    _eff_pool_floor = _sp_pool_budget_floor if _sp_pool_budget_floor is not None else _MIN_POOL_BUDGET_S_FLOOR
    _eff_pool_mult = _sp_pool_budget_mult if _sp_pool_budget_mult is not None else _POOL_BUDGET_MULT
    _MIN_POOL_BUDGET_S = max(
        _eff_pool_floor,
        (solver_timeout or 0.0) * _eff_pool_mult,
    )
    if global_budget_s is None:
        global_budget_s = _MIN_POOL_BUDGET_S
    else:
        global_budget_s = max(_MIN_POOL_BUDGET_S, global_budget_s)

    # ── Pure-routing budget cap ──────────────────────────────────
    # Routing-only problems (no MS pairs) converge fast once pmax is
    # adequate.  Cap the total pool budget so we don't waste hours on
    # a problem that should solve in seconds.  The cap is generous
    # enough for the solver to try all configs at reasonable pmax.
    _PURE_ROUTING_MAX_BUDGET_S: float = 60.0
    if _pure_routing and R >= 2:
        global_budget_s = min(global_budget_s, _PURE_ROUTING_MAX_BUDGET_S)
        if DEBUG_DIAG:
            wise_logger.info(
                "[WISE] pure-routing: capped pool budget to %.0f s",
                global_budget_s,
            )

    # Create a Manager for shared state (stop_event, children dicts).
    # Skip Manager on macOS (spawn-context Manager() deadlocks).
    # Never force serial mode — always use ThreadPoolExecutor on macOS
    # so that SAT configs run in parallel (PySAT releases the GIL)
    # and progress events are emitted via the poll loop.
    _notebook_env = _in_notebook_env()
    # On macOS, spawn-context Manager() deadlocks even from the terminal
    # (not just notebooks).  Skip Manager for ALL macOS environments.
    _skip_manager = _notebook_env or sys.platform == "darwin"
    if _skip_manager:
        # Skip Manager() — its server subprocess deadlocks on
        # macOS with spawn context (both notebook and terminal).
        # ThreadPoolExecutor: workers are threads sharing memory,
        # so threading.Event works for cooperative stop signaling.
        # Always create a FRESH stop_event — never alias
        # parent_stop_event.  Aliasing caused the parent event to
        # be set when global_budget_s expired, poisoning ALL
        # subsequent patch calls that share the same parent event.
        # The poll loop checks parent_stop_event separately.
        manager = None
        sat_children = {}
        rc2_children = {}
        stop_event = _threading.Event()
    else:
        try:
            manager = pool_context.Manager()
        except RuntimeError:
            try:
                manager = mp.Manager()
            except Exception:
                manager = None

        if manager is not None:
            sat_children = manager.dict()
            rc2_children = manager.dict()
            stop_event = manager.Event()
        else:
            sat_children = {}
            rc2_children = {}
            stop_event = None

    start_pool = time.time()
    progress_dir = tempfile.mkdtemp(prefix="wise_sat_pool_")
    progress_paths: Dict[int, str] = {}

    # Parallelism decision: always use a pool executor.  On macOS
    # (_skip_manager) use ThreadPoolExecutor; otherwise
    # ProcessPoolExecutor.  The worker count is already capped by
    # max_inner_workers at line ~3630, so total CPU usage stays bounded
    # even for nested calls.
    use_multiprocessing = True if _skip_manager else manager is not None
    
    if use_multiprocessing:
        try:
            if _skip_manager:
                # ThreadPoolExecutor avoids the spawn-context deadlock on
                # macOS (both notebook and terminal).  Workers are threads;
                # each uses the fast in-process path for SAT/RC2 solving.
                # PySAT's C solver releases the GIL, giving real parallelism.
                executor = ThreadPoolExecutor(max_workers=max_workers)
            else:
                # ProcessPoolExecutor gives true multiprocessing — each worker
                # gets its own GIL, so both CNF construction and SAT solving
                # run in parallel.
                executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=pool_context)
        except Exception as exc:
            # Catch ALL failures (RuntimeError, OSError, AssertionError
            # for daemon processes, etc.) and fall back to sequential.
            use_multiprocessing = False
            executor = None
    else:
        executor = None
    
    futures: List[Tuple[int, Tuple[int, float], Any]] = []
    timed_out = False
    
    if use_multiprocessing and executor is not None:
        # Emit progress: SAT pool is starting
        if progress_callback is not None:
            try:
                _pool_type = "threads" if _skip_manager else "processes"
                progress_callback(RoutingProgress(
                    stage=STAGE_SAT_POOL_START,
                    current=0,
                    total=len(configs),
                    message=f"Starting SAT pool: {max_workers} {_pool_type}, {len(configs)} configs",
                    extra={"num_configs": len(configs), "max_workers": max_workers,
                           "pool_type": _pool_type},
                ))
            except Exception:
                pass
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
                    inprocess_limit=_sp_inprocess_limit,
                    notebook_sat_timeout=_sp_nb_sat_timeout,
                    notebook_rc2_timeout=_sp_nb_rc2_timeout,
                )
                futures.append((idx, cfg, fut))

            # Register executor worker PIDs for atexit safety net.
            # Workers are started lazily so we collect after first submit.
            if hasattr(executor, '_processes'):
                _tracked_executor_pids.update(executor._processes.keys())

            # Track completed count for incremental progress during poll
            _poll_completed_count = 0
            # Early-exit threshold for the parallel path — once we
            # collect enough SAT results we break and let the finally
            # block handle cleanup.  IMPORTANT: never call fut.cancel()
            # on individual futures — on Python 3.11 this causes the
            # ProcessPoolExecutor management thread to crash with
            # InvalidStateError when it later tries set_exception on
            # already-cancelled futures, which hangs ALL subsequent
            # executor invocations.
            _grid_cells = n * m
            _POOL_EARLY_EXIT_SAT = 3 if _grid_cells <= 24 else 5
            _pool_sat_count = 0
            _last_progress_time = time.time()  # stall detection
            # Stall timeout: use solver_timeout (already capped by
            # notebook_sat_timeout / notebook_rc2_timeout) plus a
            # small margin for CNF construction overhead.  This way
            # the stall watchdog directly respects what the user
            # configured in WISERoutingConfig.
            _STALL_TIMEOUT_S = max(120.0, (solver_timeout or 300.0) + 60.0)
            
            while True:
                # Emit incremental progress as configs complete
                current_completed = sum(1 for _, _, fut in futures if fut.done())
                if current_completed > _poll_completed_count:
                    _last_progress_time = time.time()  # reset stall timer
                    # Count SAT results among newly completed futures
                    _pool_sat_count = 0
                    for _, _, fut in futures:
                        if fut.done():
                            try:
                                r = fut.result()
                                if r.get('sat') and r.get('status') == 'ok':
                                    _pool_sat_count += 1
                            except Exception:
                                pass
                    _poll_completed_count = current_completed
                    if progress_callback is not None:
                        try:
                            progress_callback(RoutingProgress(
                                stage=STAGE_SAT_CONFIG_DONE,
                                current=_poll_completed_count,
                                total=len(configs),
                                message=f"{_poll_completed_count}/{len(configs)} configs done",
                            ))
                        except Exception:
                            pass
                
                unfinished = [1 for _, _, fut in futures if not fut.done()]
                if not unfinished:
                    break
                # Early exit: enough SAT results found — break and let
                # the finally block shut down the executor cleanly.
                if _pool_sat_count >= _POOL_EARLY_EXIT_SAT:
                    if DEBUG_DIAG:
                        wise_logger.info(
                            "[WISE] pool early exit: %d SAT results from "
                            "%d/%d completed configs",
                            _pool_sat_count, _poll_completed_count,
                            len(configs),
                        )
                    if stop_event is not None:
                        stop_event.set()
                    # Do NOT cancel individual futures here — it crashes
                    # the executor's management thread on Python 3.11.
                    # The finally block calls executor.shutdown(cancel_futures=True).
                    break
                # Check parent stop_event (from outer best-effort search)
                if parent_stop_event is not None:
                    try:
                        if parent_stop_event.is_set():
                            if DEBUG_DIAG:
                                wise_logger.info(
                                    "[WISE] parent_stop_event set; propagating to inner SAT pool."
                                )
                            timed_out = True
                            if stop_event is not None:
                                stop_event.set()
                            break
                    except Exception:
                        pass
                if (
                    global_budget_s is not None
                    and (time.time() - start_pool) >= global_budget_s
                ):
                    if DEBUG_DIAG:
                        wise_logger.info(
                            "[WISE] global SAT pool budget exhausted; collecting best-so-far results."
                        )
                    timed_out = True
                    if stop_event is not None:
                        stop_event.set()
                    break
                # Stall detection: if no futures completed for
                # _STALL_TIMEOUT_S, the executor is likely broken
                # (e.g. management thread crashed).  Break out to
                # avoid hanging indefinitely.
                if (time.time() - _last_progress_time) >= _STALL_TIMEOUT_S:
                    wise_logger.info(
                        "[WISE] SAT pool stalled for %.0fs with "
                        "%d/%d completed; breaking out.",
                        _STALL_TIMEOUT_S, _poll_completed_count,
                        len(configs),
                    )
                    timed_out = True
                    break
                time.sleep(0.05)

            for idx, cfg, fut in futures:
                if fut.done():
                    try:
                        res = fut.result()
                        results.append(res)
                        # Emit progress: config completed
                        if progress_callback is not None:
                            try:
                                progress_callback(RoutingProgress(
                                    stage=STAGE_SAT_CONFIG_DONE,
                                    current=len(results),
                                    total=len(configs),
                                    message=f"Config {idx} done: P_max={res.get('P_max')}, sat={res.get('sat')}",
                                    extra={
                                        "config_idx": idx,
                                        "P_max": res.get("P_max"),
                                        "sat": res.get("sat"),
                                        "status": res.get("status"),
                                    },
                                ))
                            except Exception:
                                pass
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
            # First, terminate any nested SAT/RC2 solver processes tracked in the
            # manager dicts. This is more reliable than using executor._processes
            # (a private API) and ensures we clean up the *actual* solvers.
            try:
                for label, children_dict in [("SAT", sat_children), ("RC2", rc2_children)]:
                    if children_dict:
                        for job_key, pid in list(children_dict.items()):
                            try:
                                os.kill(pid, signal.SIGTERM)
                            except (OSError, ProcessLookupError):
                                pass
                        # Brief grace period for termination
                        time.sleep(0.2)
                        for job_key, pid in list(children_dict.items()):
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except (OSError, ProcessLookupError):
                                pass
            except Exception as e:
                wise_logger.debug("[WISE] error killing nested SAT/RC2 children: %s", e)

            # Now shut down the executor pool itself.
            try:
                if executor is not None:
                    # Kill executor workers BEFORE shutdown (shutdown clears
                    # _processes to None, so we must read PIDs first).
                    _kill_executor_workers(executor)
                    # Release executor resources and cancel any not-yet-started futures.
                    executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                wise_logger.error("[WISE] error during executor shutdown: %s", e)

            # FIX D: Reap zombie children after killing processes
            try:
                while True:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                    if pid == 0:
                        break
            except (ChildProcessError, OSError):
                pass

        # Signal SAT bar completion after multiprocessing path finishes.
        if progress_callback is not None:
            try:
                n_sat = sum(1 for r in results if r.get("sat"))
                n_unsat = sum(1 for r in results if not r.get("sat"))
                _pool_type = "threads" if _skip_manager else "processes"
                progress_callback(RoutingProgress(
                    stage=STAGE_SAT_CONFIG_DONE,
                    current=len(configs),
                    total=len(configs),
                    message=(f"SAT pool done ({_pool_type}): "
                             f"{n_sat} SAT + {n_unsat} UNSAT "
                             f"= {len(results)}/{len(configs)} configs"),
                ))
            except Exception:
                pass

        # If ALL workers crashed, fall through to sequential fallback.
        if not results and configs:
            wise_logger.warning(
                "[WISE] all %d MP workers produced no results; "
                "falling back to sequential.",
                len(configs),
            )
            use_multiprocessing = False  # flag for downstream logging

    # -- Sequential fallback (when all MP workers failed) --
    if not results and configs:
        # Emit SAT pool start so the progress bar initialises
        if progress_callback is not None:
            try:
                progress_callback(RoutingProgress(
                    stage=STAGE_SAT_POOL_START,
                    current=0,
                    total=len(configs),
                    message=f"SAT sequential fallback: 0/{len(configs)} configs",
                ))
            except Exception:
                pass

        for idx, cfg in enumerate(configs):
            if parent_stop_event is not None:
                try:
                    if parent_stop_event.is_set():
                        break
                except Exception:
                    pass
            progress_path = os.path.join(progress_dir, f"cfg_{idx}.pkl")
            try:
                res = _wise_sat_config_worker(
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
                    inprocess_limit=_sp_inprocess_limit,
                    notebook_sat_timeout=_sp_nb_sat_timeout,
                    notebook_rc2_timeout=_sp_nb_rc2_timeout,
                )
                results.append(res)

                # Emit per-config progress so the SAT bar updates
                if progress_callback is not None:
                    try:
                        progress_callback(RoutingProgress(
                            stage=STAGE_SAT_CONFIG_DONE,
                            current=idx + 1,
                            total=len(configs),
                            message=(f"SAT sequential: config {idx + 1}/{len(configs)} "
                                     f"({'SAT' if res.get('sat') else 'UNSAT'})"),
                        ))
                    except Exception:
                        pass

                if res.get("sat") and res.get("status") == "ok":
                    break  # one SAT result is enough for serial mode
            except Exception:
                pass

    # ---- Cleanup: shut down Manager and remove progress temp dir ----
    if manager is not None:
        try:
            manager.shutdown()
        except Exception as _mgr_err:
            wise_logger.debug("[WISE] Manager shutdown failed: %r", _mgr_err)
    try:
        shutil.rmtree(progress_dir, ignore_errors=True)
    except Exception as _rmdir_err:
        wise_logger.debug("[WISE] progress_dir cleanup failed: %r", _rmdir_err)

    sat_results = [
        r for r in results if r.get("sat") and r.get("status") == "ok"
    ]

    # Diagnostic: print all SAT results and their scores
    if DEBUG_DIAG:
        import sys as _sys
        print(f"[DIAG] Total results={len(results)}, SAT results={len(sat_results)}", file=_sys.stderr)
        for i, r in enumerate(results):
            pmax = r.get('P_max', '?')
            cap = r.get('boundary_capacity_factor', '?')
            usage = r.get('per_round_usage', [])
            sat = r.get('sat', False)
            status = r.get('status', '?')
            score = _wise_score_config(r, optimize_round_start, R) if sat and status == 'ok' else 'N/A'
            print(f"[DIAG]   result[{i}]: P_max={pmax}, cap_factor={cap}, usage={usage}, "
                  f"sat={sat}, status={status}, score={score}", file=_sys.stderr)

    if not sat_results:
        raise NoFeasibleLayoutError(
            f"No feasible layout for any Σ_r P_r bound over {len(configs)} configs."
        )

    best_res = min(
        sat_results,
        key=lambda res: _wise_score_config(res, optimize_round_start, R)
    )
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
    # Use the TIGHT sum_bound from the winning binary search, not the
    # loose maximum (rounds_under_sum * P_max).  This guarantees the
    # rebuild vpool creates variables in exactly the same order as the
    # worker that produced sat_model_star.
    worker_sum_bound = best_res.get("sum_bound_B")
    sum_bound_B = worker_sum_bound if worker_sum_bound is not None else rounds_under_sum * P_max

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

    def var_a_sat(r: int, p: int, d: int, c: int, ion: int) -> int:
        return vpool_sat.id(("a", r, p, d, c, ion))

    if DEBUG_DIAG:
        wise_logger.info("[WISE] minimal ΣP found: %d (P_max=%d)", best_sum_bound, P_max)
    pass_horizon = P_max
    P_bounds = [pass_horizon] * R

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

    _ = _wise_debug_boundary_stats(
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
    # Level 3: MaxSAT V-penalty refinement under ΣP*
    # -------------------------------
    # Build a WCNF that adds soft clauses penalising V (column/junction)
    # passes.  V passes cost ~2.4× more physical time than H (row) passes,
    # but the plain SAT solver does not know this — it can return any
    # satisfying assignment, including ones heavy on V passes.  MaxSAT
    # with V-penalty soft clauses finds the H/V mix that minimises
    # estimated reconfiguration time among all feasible routings at the
    # same total-pass budget.
    #
    # Boundary-crossing soft clauses are included when the subgrid is
    # boundary-adjacent, so both objectives are optimised jointly.
    has_boundary_adj = any(boundary_adjacent.values())

    # ── Pure-routing shortcut: skip V-penalty MaxSAT ─────────────
    # When there are no MS-gate pairs, the H/V pass ratio has no
    # effect on gate fidelity or pair co-location.  The SAT worker
    # model is already optimal for total passes — skip the expensive
    # (and potentially unbounded) RC2 MaxSAT refinement entirely.
    # Set defaults; the try block below overrides them when V-penalty
    # MaxSAT actually runs (i.e., when _pure_routing is False).
    model_used = sat_model_star
    vpool_used = vpool_sat
    var_a_used = var_a_sat
    ions_used = ions_sat

    if _pure_routing and R >= 2:
        if DEBUG_DIAG:
            wise_logger.info(
                "[WISE] pure-routing R>=%d: skipping V-penalty MaxSAT "
                "(no MS pairs → H/V ratio irrelevant)", R
            )

    try:
        if _pure_routing and R >= 2:
            pass  # Defaults (model_used etc.) already set above.
        else:
            if DEBUG_DIAG:
                print(f"[DIAG] V-penalty MaxSAT: P_max={pass_horizon}, sum_bound_B={sum_bound_B}, "
                      f"best_sum_bound={best_sum_bound}, worker_sum_bound={worker_sum_bound}, "
                      f"opt_round_start={optimize_round_start}", file=_sys.stderr)
            t_build_start = time.time()
            wcnf, vpool_w, ions_w, var_a_w, _, _ = _wise_build_structural_cnf(
                builder_ctx,
                pass_horizon,
                sum_bound_B=sum_bound_B,
                use_wcnf=True,
                add_boundary_soft=has_boundary_adj,
                phase_label=f"ΣP<={sum_bound_B}/V_OPT",
                optimize_round_start=optimize_round_start,
                debug_skip_cardinality=False,
                boundary_adjacent=boundary_adjacent,
                cross_boundary_prefs=cross_boundary_prefs,
                boundary_capacity_factor=chosen_boundary_capacity_factor,
                bt_soft_weight=bt_soft_weight_value,
            )

            # Add V-penalty soft clauses: prefer H passes (phase=False).
            # The phase variable is True ⇔ V pass.  Adding [-phase_var]
            # as a soft clause penalises each V-phase pass by *v_penalty*.
            # Weight is set so that eliminating one V pass is always worth
            # more than any boundary soft clause (those use weight ≤ wB_col).
            cal = DEFAULT_CALIBRATION
            # H pass cost: Move + Merge + Rotation + Split + Move
            _row_swap = (
                cal.shuttle_time + cal.merge_time + cal.rotation_time
                + cal.split_time + cal.shuttle_time
            )
            # V pass cost: 2×Junction + (4×Junction + Move) × 2
            _col_swap = (2 * cal.junction_time) + (
                4 * cal.junction_time + cal.shuttle_time
            ) * 2
            # Weight in µs (integer): extra cost of V over H
            v_penalty = int(round((_col_swap - _row_swap) * 1e6))  # ≈ 298 µs
            if v_penalty < 1:
                v_penalty = 1

            P_bounds_opt = [pass_horizon] * R
            v_soft_count = 0
            start_round = optimize_round_start if R > 1 else 0
            for r in range(start_round, R):
                for p in range(P_bounds_opt[r]):
                    phase_var = vpool_w.id(("phase", r, p))
                    if phase_var <= vpool_w.top:
                        wcnf.append([-phase_var], weight=v_penalty)
                        v_soft_count += 1

            t_build_end = time.time()

            if DEBUG_DIAG:
                print(f"[DIAG] WCNF built: vars={wcnf.nv}, hard={len(wcnf.hard)}, "
                      f"soft={len(wcnf.soft)} (V-penalty={v_soft_count}, weight={v_penalty}), "
                      f"time={t_build_end - t_build_start:.3f}s", file=_sys.stderr)

            rc2 = RC2(wcnf)
            model_rc2 = rc2.compute()
            cost_rc2 = rc2.cost if model_rc2 is not None else None
            status_rc2 = "ok" if model_rc2 is not None else "error"

            if DEBUG_DIAG:
                print(f"[DIAG] RC2 status={status_rc2}, opt_cost={cost_rc2}", file=_sys.stderr)

            if status_rc2 == "ok" and model_rc2 is not None:
                # Compare estimated reconfig cost: RC2 vs worker SAT model
                schedule_worker = _wise_decode_schedule_from_model(
                    sat_model_star, vpool_sat, n, m, R, P_max,
                    ignore_initial_reconfig,
                )
                schedule_rc2 = _wise_decode_schedule_from_model(
                    model_rc2, vpool_w, n, m, R, P_max,
                    ignore_initial_reconfig,
                )
                opt_start = optimize_round_start if R > 1 else 0
                cost_worker = _wise_estimate_reconfig_cost(
                    schedule_worker[opt_start:]
                )
                cost_rc2_est = _wise_estimate_reconfig_cost(
                    schedule_rc2[opt_start:]
                )

                if DEBUG_DIAG:
                    print(f"[DIAG] reconfig cost: worker={cost_worker:.6f}, RC2={cost_rc2_est:.6f}", file=_sys.stderr)

                if cost_rc2_est <= cost_worker:
                    model_used = model_rc2
                    vpool_used = vpool_w
                    var_a_used = var_a_w
                    ions_used = ions_w
                    if DEBUG_DIAG:
                        print(f"[DIAG] using RC2 V-optimised model (saved {(cost_worker - cost_rc2_est)*1e6:.1f} µs)", file=_sys.stderr)
                else:
                    model_used = sat_model_star
                    vpool_used = vpool_sat
                    var_a_used = var_a_sat
                    ions_used = ions_sat
                    if DEBUG_DIAG:
                        print("[DIAG] RC2 model worse; keeping worker SAT model", file=_sys.stderr)
            else:
                model_used = sat_model_star
                vpool_used = vpool_sat
                var_a_used = var_a_sat
                ions_used = ions_sat
                if DEBUG_DIAG:
                    print(f"[DIAG] MaxSAT unavailable (status={status_rc2}); falling back to SAT model.", file=_sys.stderr)
    except Exception as maxsat_exc:
        # MaxSAT refinement is best-effort; never block the pipeline.
        model_used = sat_model_star
        vpool_used = vpool_sat
        var_a_used = var_a_sat
        ions_used = ions_sat
        if DEBUG_DIAG:
            print(f"[DIAG] V-penalty MaxSAT failed ({maxsat_exc}); using SAT model.", file=_sys.stderr)
            import traceback; traceback.print_exc(file=_sys.stderr)

    # Decide which ions are "core" for this slice.
    # A reasonable default: all active ions that lie entirely in the current subgrid.
    # If you already have a list `core_ions_this_slice`, reuse that here.
    core_ions_this_slice = ions   # or a filtered subset if you prefer

    _ = _wise_debug_boundary_stats(
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

    # ── Self-consistency check: replay decoded schedule on A_in and
    #    verify it produces the decoded layouts.  A mismatch here means
    #    the variable-pool decode produced inconsistent (layout, schedule)
    #    pairs — catch it at the source rather than downstream.
    for _sc_r in range(R):
        _sc_base = A_in.copy() if _sc_r == 0 else layouts[_sc_r - 1].copy()
        if schedule[_sc_r]:
            for _sc_pass in schedule[_sc_r]:
                for (_sr, _sc) in _sc_pass.get("h_swaps", []):
                    if 0 <= _sr < _sc_base.shape[0] and 0 <= _sc < _sc_base.shape[1] - 1:
                        _sc_base[_sr, _sc], _sc_base[_sr, _sc + 1] = (
                            _sc_base[_sr, _sc + 1],
                            _sc_base[_sr, _sc],
                        )
                for (_sr, _sc) in _sc_pass.get("v_swaps", []):
                    if 0 <= _sr < _sc_base.shape[0] - 1 and 0 <= _sc < _sc_base.shape[1]:
                        _sc_base[_sr, _sc], _sc_base[_sr + 1, _sc] = (
                            _sc_base[_sr + 1, _sc],
                            _sc_base[_sr, _sc],
                        )
        _sc_diff_count = int(np.count_nonzero(_sc_base != layouts[_sc_r]))
        # Log at DEBUG when self-check passes (0 diffs); only WARNING+ for real issues
        if _sc_diff_count == 0:
            wise_logger.debug(
                "[WISE-SELFCHECK] Round %d: replay vs decoded: 0/%d cells differ "
                "(grid_origin=%s, grid=%dx%d) — OK",
                _sc_r, layouts[_sc_r].size,
                grid_origin, _sc_base.shape[0], _sc_base.shape[1],
            )
        else:
            wise_logger.warning(
                "[WISE-SELFCHECK] Round %d: replay vs decoded: %d/%d cells differ "
                "(grid_origin=%s, grid=%dx%d)",
                _sc_r, _sc_diff_count, layouts[_sc_r].size,
                grid_origin, _sc_base.shape[0], _sc_base.shape[1],
            )
        if not np.array_equal(_sc_base, layouts[_sc_r]):
            wise_logger.error(
                "[WISE] SAT DECODE SELF-CONSISTENCY FAIL round %d: "
                "schedule replay produces %d/%d cells different from "
                "decoded layout.  Using replay-derived layout.",
                _sc_r,
                _sc_diff_count,
                layouts[_sc_r].size,
            )
            layouts[_sc_r] = _sc_base

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

    bt_ok, _bt_failed_ions = _wise_assert_bt_consistency(layouts, BT, R, n, m, wise_logger)
    if not bt_ok and bt_soft and _bt_failed_ions:
        # --- Iterative pin-drop retry ---
        # Some BT pins are infeasible given the hard constraints (MS pairs).
        # Instead of retrying with hard BT (which is almost always UNSAT),
        # remove the failing ions from BT and re-solve with reduced soft BT.
        # Repeat up to _BT_PINDROP_MAX_ITER times until consistent.
        _BT_PINDROP_MAX_ITER = 3
        _dropped_all: Set[int] = set()
        for _pd_iter in range(_BT_PINDROP_MAX_ITER):
            _dropped_all.update(_bt_failed_ions)
            # Build reduced BT with failing ions removed
            BT_reduced: List[Dict[int, Tuple[int, int]]] = []
            for _bt_r_dict in (BT or []):
                BT_reduced.append(
                    {ion: pos for ion, pos in _bt_r_dict.items()
                     if ion not in _dropped_all}
                )
            _remaining_pins = sum(len(d) for d in BT_reduced)
            wise_logger.info(
                "[WISE] BT pin-drop iter %d: dropped %d ions (%s), "
                "%d pins remaining",
                _pd_iter + 1,
                len(_dropped_all),
                sorted(_dropped_all),
                _remaining_pins,
            )
            if _remaining_pins == 0:
                # No BT pins left — nothing to enforce, accept as-is
                wise_logger.info(
                    "[WISE] All BT pins dropped; accepting current result"
                )
                break
            try:
                layouts_pd, schedule_pd, ph_pd = optimal_QMR_for_WISE(
                    A_in=A_in,
                    P_arr=P_arr,
                    k=k,
                    BT=BT_reduced,
                    wB_col=wB_col,
                    wB_row=wB_row,
                    max_rc2_time=max_rc2_time,
                    max_sat_time=max_sat_time,
                    active_ions=active_ions,
                    full_P_arr=full_P_arr,
                    ignore_initial_reconfig=ignore_initial_reconfig,
                    base_pmax_in=base_pmax_in,
                    prev_pmax=prev_pmax,
                    grid_origin=grid_origin,
                    boundary_adjacent=boundary_adjacent,
                    cross_boundary_prefs=cross_boundary_prefs,
                    bt_soft=True,
                    parent_stop_event=parent_stop_event,
                    progress_callback=progress_callback,
                    max_inner_workers=max_inner_workers,
                    solver_params=solver_params,
                )
                bt_ok, _bt_failed_ions = _wise_assert_bt_consistency(
                    layouts_pd, BT_reduced, R, n, m, wise_logger
                )
                if bt_ok:
                    wise_logger.info(
                        "[WISE] Pin-drop retry succeeded after dropping %d ions",
                        len(_dropped_all),
                    )
                    layouts = layouts_pd
                    schedule = schedule_pd
                    pass_horizon = ph_pd
                    break
                # else: loop continues, dropping newly failed ions too
            except (NoFeasibleLayoutError, Exception) as _pd_exc:
                wise_logger.debug(
                    "[WISE] Pin-drop retry iter %d raised %s; "
                    "keeping previous result",
                    _pd_iter + 1,
                    _pd_exc,
                )
                break
        else:
            # Exhausted iterations — accept what we have
            wise_logger.info(
                "[WISE] BT pin-drop exhausted %d iterations; "
                "accepting result with %d unsatisfied pins",
                _BT_PINDROP_MAX_ITER,
                len(_bt_failed_ions),
            )
    elif not bt_ok:
        wise_logger.debug(
            "[WISE] BT consistency check failed (non-soft mode); "
            "proceeding with returned layouts"
        )

    # --- Reverse sparse-grid padding: replace dummy IDs back with 0 ---
    if _dummy_ids:
        for lay in layouts:
            for _r in range(lay.shape[0]):
                for _c in range(lay.shape[1]):
                    if int(lay[_r, _c]) in _dummy_ids:
                        lay[_r, _c] = 0

    return layouts, schedule, pass_horizon


# ---------------------------------------------------------------------------
# OOP wrapper — WiseSATSolver
# ---------------------------------------------------------------------------


@dataclass
class SATPoolResult:
    """Result from a SAT pool solve operation.
    
    Encapsulates all outputs from `optimal_QMR_for_WISE` in a structured
    dataclass for easier inspection and testing.
    
    Attributes
    ----------
    layouts : list of np.ndarray
        Per-round end-of-round ion layouts. Each is an ``n × m`` int array.
    schedule : list of list of dict
        Per-round sequence of swap passes. Each pass dict has keys
        ``"phase"`` (``"horizontal"`` or ``"vertical"``), ``"dir"``
        (``"up"``/``"down"``/``"left"``/``"right"``), and ``"swaps"``
        (list of ion pairs).
    pass_horizon : int
        The optimised P_max (maximum passes per round) found by the solver.
    status : str
        ``"ok"`` if a solution was found, otherwise an error code.
    elapsed_s : float
        Wall-clock time spent solving.
    num_rounds : int
        Number of MS-gate rounds in the input.
    solver_config : SATSolverConfig or None
        Configuration used for this solve (if available).
    """
    layouts: List[np.ndarray]
    schedule: List[List[Dict[str, Any]]]
    pass_horizon: int
    status: str = "ok"
    elapsed_s: float = 0.0
    num_rounds: int = 0
    solver_config: Optional["SATSolverConfig"] = None


class WiseSATSolver:
    """
    Object-oriented interface to the WISE SAT-based ion routing solver.

    Groups the many positional / keyword parameters of
    ``optimal_QMR_for_WISE`` into a single object with a clear ``solve()``
    method.  Callers can construct the solver once and adjust individual
    attributes before calling ``solve()`` again.

    The internal SAT construction logic (``_wise_build_structural_cnf``,
    ``_wise_decode_schedule_from_model``, etc.) is kept as module-level
    functions; this class is a thin, user-facing API that delegates to them.

    Parameters
    ----------
    A_in : np.ndarray
        Initial ``n × m`` ion layout (2-D integer array).
    P_arr : list of list of (int, int)
        Per-round lists of ion pairs that must end up in the same row
        and block.
    k : int
        Block / trap capacity.
    BT : list of dict, optional
        Per-round BT (boundary target) pinning maps ``{ion: (row, col)}``.
    active_ions : set of int, optional
        Ions considered active (if ``None``, auto-detected from *P_arr*
        and *BT*).
    full_P_arr : list of list of (int, int)
        Full (un-sliced) pair array; used for boundary preference
        computation.
    max_sat_time, max_rc2_time : float
        Per-instance timeout for SAT / RC2 solver sub-processes.
    wB_col, wB_row : int
        Soft-clause weights for column / row boundary avoidance in
        MaxSAT mode.
    ignore_initial_reconfig : bool
        If ``True`` the first reconfiguration round gets extra slack
        (``P_bound + n + m``).
    base_pmax_in, prev_pmax : int or None
        Starting search point and previous optimal P_max (for warm
        starts).
    grid_origin : (int, int)
        Global ``(row, col)`` offset of this sub-grid (used to shift
        swap coordinates back to global space).
    boundary_adjacent : dict of str → bool
        Which edges of this patch have neighbouring patches.
    cross_boundary_prefs : list of dict
        Per-round ``{ion: set_of_directions}`` preferences for boundary
        placement.
    bt_soft : bool
        Use soft BT pinning via MaxSAT instead of hard constraints.
    parent_stop_event : Event or None
        If set, checked periodically; triggers early abort of the SAT
        pool.
    progress_callback : callable or None
        ``(RoutingProgress) -> None`` emitted at key milestones.

    Example
    -------
    >>> solver = WiseSATSolver(
    ...     A_in=grid,
    ...     P_arr=pairs,
    ...     k=2,
    ...     max_sat_time=600,
    ... )
    >>> layouts, schedule, pass_horizon = solver.solve()
    """

    def __init__(
        self,
        A_in: np.ndarray,
        P_arr: List[List[Tuple[int, int]]],
        *,
        k: int,
        BT: Optional[List[Dict[int, Tuple[int, int]]]] = None,
        wB_col: int = 1,
        wB_row: int = 1,
        max_rc2_time: Optional[float] = None,
        max_sat_time: Optional[float] = None,
        active_ions: Optional[Set[int]] = None,
        full_P_arr: Optional[List[List[Tuple[int, int]]]] = None,
        ignore_initial_reconfig: bool = False,
        base_pmax_in: Optional[int] = None,
        prev_pmax: Optional[int] = None,
        grid_origin: Tuple[int, int] = (0, 0),
        boundary_adjacent: Optional[Dict[str, bool]] = None,
        cross_boundary_prefs: Optional[List[Dict[int, Set[str]]]] = None,
        bt_soft: bool = False,
        parent_stop_event: Optional[Any] = None,
        progress_callback: Optional[Callable] = None,
        config: Optional["SATSolverConfig"] = None,
        max_inner_workers: Optional[int] = None,
    ):
        self.A_in = A_in
        self.P_arr = P_arr
        self.k = k
        self.BT = BT
        self.active_ions = active_ions
        self.full_P_arr = full_P_arr if full_P_arr is not None else []
        self.ignore_initial_reconfig = ignore_initial_reconfig
        self.prev_pmax = prev_pmax
        self.grid_origin = grid_origin
        self.boundary_adjacent = boundary_adjacent
        self.cross_boundary_prefs = cross_boundary_prefs
        self.parent_stop_event = parent_stop_event
        self.progress_callback = progress_callback
        self.max_inner_workers = max_inner_workers
        
        # If a SATSolverConfig is provided, use its values; otherwise use explicit args
        if config is not None:
            self._config = config
            self.wB_col = config.wB_col
            self.wB_row = config.wB_row
            self.max_rc2_time = config.max_rc2_time
            self.max_sat_time = config.max_sat_time
            self.base_pmax_in = config.base_pmax_in
            self.bt_soft = config.bt_soft
        else:
            self._config = None
            self.wB_col = wB_col
            self.wB_row = wB_row
            self.max_rc2_time = max_rc2_time
            self.max_sat_time = max_sat_time
            self.base_pmax_in = base_pmax_in
            self.bt_soft = bt_soft
    
    @classmethod
    def from_config(
        cls,
        A_in: np.ndarray,
        P_arr: List[List[Tuple[int, int]]],
        k: int,
        config: "SATSolverConfig",
        *,
        BT: Optional[List[Dict[int, Tuple[int, int]]]] = None,
        active_ions: Optional[Set[int]] = None,
        full_P_arr: Optional[List[List[Tuple[int, int]]]] = None,
        ignore_initial_reconfig: bool = False,
        prev_pmax: Optional[int] = None,
        grid_origin: Tuple[int, int] = (0, 0),
        boundary_adjacent: Optional[Dict[str, bool]] = None,
        cross_boundary_prefs: Optional[List[Dict[int, Set[str]]]] = None,
        parent_stop_event: Optional[Any] = None,
        progress_callback: Optional[Callable] = None,
        max_inner_workers: Optional[int] = None,
    ) -> "WiseSATSolver":
        """Create a solver from a SATSolverConfig.
        
        Provides a cleaner interface when configuration is managed via
        ``SATSolverConfig.from_heuristics()`` or similar.
        
        Parameters
        ----------
        A_in : np.ndarray
            Initial ``n × m`` ion layout.
        P_arr : list of list of (int, int)
            Per-round MS-gate pairs.
        k : int
            Block capacity (ions per trap).
        config : SATSolverConfig
            Solver configuration with timeouts, weights, etc.
        BT, active_ions, etc.
            Same as ``__init__``.
        
        Returns
        -------
        WiseSATSolver
            A configured solver instance.
        
        Example
        -------
        >>> config = SATSolverConfig.from_heuristics(n=6, m=4, k=2, R=3)
        >>> solver = WiseSATSolver.from_config(grid, pairs, k=2, config=config)
        >>> result = solver.solve_managed()
        """
        return cls(
            A_in=A_in,
            P_arr=P_arr,
            k=k,
            BT=BT,
            active_ions=active_ions,
            full_P_arr=full_P_arr,
            ignore_initial_reconfig=ignore_initial_reconfig,
            prev_pmax=prev_pmax,
            grid_origin=grid_origin,
            boundary_adjacent=boundary_adjacent,
            cross_boundary_prefs=cross_boundary_prefs,
            parent_stop_event=parent_stop_event,
            progress_callback=progress_callback,
            config=config,
            max_inner_workers=max_inner_workers,
        )
    
    @classmethod
    def from_heuristics(
        cls,
        A_in: np.ndarray,
        P_arr: List[List[Tuple[int, int]]],
        k: int,
        *,
        BT: Optional[List[Dict[int, Tuple[int, int]]]] = None,
        active_ions: Optional[Set[int]] = None,
        full_P_arr: Optional[List[List[Tuple[int, int]]]] = None,
        ignore_initial_reconfig: bool = False,
        prev_pmax: Optional[int] = None,
        grid_origin: Tuple[int, int] = (0, 0),
        boundary_adjacent: Optional[Dict[str, bool]] = None,
        cross_boundary_prefs: Optional[List[Dict[int, Set[str]]]] = None,
        parent_stop_event: Optional[Any] = None,
        progress_callback: Optional[Callable] = None,
        max_inner_workers: Optional[int] = None,
    ) -> "WiseSATSolver":
        """Create a solver with heuristically-determined configuration.
        
        Computes reasonable defaults for timeouts, weights, etc. based
        on the grid dimensions and number of gate rounds. This is the
        recommended way to create a solver when you don't need fine
        control over SAT parameters.
        
        Parameters
        ----------
        A_in : np.ndarray
            Initial ``n × m`` ion layout.
        P_arr : list of list of (int, int)
            Per-round MS-gate pairs.
        k : int
            Block capacity.
        BT, active_ions, etc.
            Same as ``__init__``.
        
        Returns
        -------
        WiseSATSolver
            Solver with auto-tuned configuration.
        
        Example
        -------
        >>> solver = WiseSATSolver.from_heuristics(grid, pairs, k=2)
        >>> result = solver.solve_managed()
        """
        n, m = A_in.shape
        R = len(P_arr)
        num_pairs = sum(len(round_pairs) for round_pairs in P_arr)
        has_bt_pins = BT is not None and any(bt_round for bt_round in BT)
        
        config = SATSolverConfig.from_heuristics(
            n=n, m=m, k=k, R=R, num_pairs=num_pairs, has_bt_pins=has_bt_pins
        )
        
        return cls.from_config(
            A_in=A_in,
            P_arr=P_arr,
            k=k,
            config=config,
            BT=BT,
            active_ions=active_ions,
            full_P_arr=full_P_arr,
            ignore_initial_reconfig=ignore_initial_reconfig,
            prev_pmax=prev_pmax,
            grid_origin=grid_origin,
            boundary_adjacent=boundary_adjacent,
            cross_boundary_prefs=cross_boundary_prefs,
            parent_stop_event=parent_stop_event,
            progress_callback=progress_callback,
            max_inner_workers=max_inner_workers,
        )

    def solve(self) -> Tuple[List[np.ndarray], List[List[Dict[str, Any]]], int]:
        """
        Run the full SAT-based solve.

        Equivalent to calling ``optimal_QMR_for_WISE`` with the stored
        parameters.

        Returns
        -------
        layouts : list of np.ndarray
            Per-round end-of-round ion layouts.
        schedule : list of list of dict
            Per-round sequence of swap passes (phase + swap lists).
        pass_horizon : int
            The chosen P_max value.

        Raises
        ------
        NoFeasibleLayoutError
            If no satisfiable configuration is found.
        """
        return optimal_QMR_for_WISE(
            self.A_in,
            self.P_arr,
            k=self.k,
            BT=self.BT,
            wB_col=self.wB_col,
            wB_row=self.wB_row,
            max_rc2_time=self.max_rc2_time,
            max_sat_time=self.max_sat_time,
            active_ions=self.active_ions,
            full_P_arr=self.full_P_arr,
            ignore_initial_reconfig=self.ignore_initial_reconfig,
            base_pmax_in=self.base_pmax_in,
            prev_pmax=self.prev_pmax,
            grid_origin=self.grid_origin,
            boundary_adjacent=self.boundary_adjacent,
            cross_boundary_prefs=self.cross_boundary_prefs,
            bt_soft=self.bt_soft,
            parent_stop_event=self.parent_stop_event,
            progress_callback=self.progress_callback,
            max_inner_workers=self.max_inner_workers,
        )
    
    def solve_managed(self) -> SATPoolResult:
        """
        Run the SAT-based solve with guaranteed process cleanup.
        
        Wraps the solve in a ``SATProcessManager`` context manager,
        ensuring all SAT/RC2 child processes are terminated even if
        the caller receives a KeyboardInterrupt, SIGTERM, or raises
        an exception.
        
        This is the **recommended** entry point when running SAT solves
        in long-running applications, notebooks, or pipeline scripts
        where zombie processes can accumulate.
        
        Returns
        -------
        SATPoolResult
            Structured result containing layouts, schedule, and metadata.
        
        Raises
        ------
        NoFeasibleLayoutError
            If no satisfiable configuration is found.
        WiseSATError
            If SAT solving fails due to timeout or other SAT-specific error.
        
        Example
        -------
        >>> solver = WiseSATSolver.from_heuristics(grid, pairs, k=2)
        >>> result = solver.solve_managed()
        >>> print(f"Found schedule with P_max={result.pass_horizon}")
        >>> for r, layout in enumerate(result.layouts):
        ...     print(f"Round {r}: {layout}")
        """
        start_time = time.time()
        
        with SATProcessManager(self.parent_stop_event) as pm:
            # If the manager provides a stop_event, use it
            effective_stop_event = pm.stop_event if pm.stop_event is not None else self.parent_stop_event
            
            try:
                layouts, schedule, pass_horizon = optimal_QMR_for_WISE(
                    self.A_in,
                    self.P_arr,
                    k=self.k,
                    BT=self.BT,
                    wB_col=self.wB_col,
                    wB_row=self.wB_row,
                    max_rc2_time=self.max_rc2_time,
                    max_sat_time=self.max_sat_time,
                    active_ions=self.active_ions,
                    full_P_arr=self.full_P_arr,
                    ignore_initial_reconfig=self.ignore_initial_reconfig,
                    base_pmax_in=self.base_pmax_in,
                    prev_pmax=self.prev_pmax,
                    grid_origin=self.grid_origin,
                    boundary_adjacent=self.boundary_adjacent,
                    cross_boundary_prefs=self.cross_boundary_prefs,
                    bt_soft=self.bt_soft,
                    parent_stop_event=effective_stop_event,
                    progress_callback=self.progress_callback,
                    max_inner_workers=self.max_inner_workers,
                )
                
                elapsed_s = time.time() - start_time
                
                return SATPoolResult(
                    layouts=layouts,
                    schedule=schedule,
                    pass_horizon=pass_horizon,
                    status="ok",
                    elapsed_s=elapsed_s,
                    num_rounds=len(self.P_arr),
                    solver_config=self._config,
                )
            
            except NoFeasibleLayoutError:
                # Re-raise domain-specific errors unchanged
                raise
            except (KeyboardInterrupt, SystemExit):
                # Stop event should already be set by SATProcessManager
                elapsed_s = time.time() - start_time
                raise WiseSATError(
                    f"SAT solve interrupted after {elapsed_s:.1f}s"
                )
            except Exception as e:
                elapsed_s = time.time() - start_time
                raise WiseSATError(
                    f"SAT solve failed after {elapsed_s:.1f}s: {e}"
                ) from e
