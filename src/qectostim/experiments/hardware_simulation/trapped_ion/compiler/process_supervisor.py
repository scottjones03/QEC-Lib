"""
Process supervisor for WISE SAT routing.

Centralized tracking of child processes spawned by SAT/MaxSAT solvers
and pool executors. Ensures cleanup on exit - including during debugging
interrupts and notebook kernel restarts.

Backported from trapped_ion/routing/process_supervisor.py

Usage
-----
>>> from .process_supervisor import supervisor
>>> supervisor.register(pid)          # after p.start()
>>> supervisor.unregister(pid)        # after p.join() / p.kill()
>>> supervisor.kill_all()             # emergency teardown
>>> supervisor.active_count()         # for global caps
"""
from __future__ import annotations

import atexit
import logging
import os
import signal
import sys
import threading
import time
from typing import Set

__all__ = [
    "ProcessSupervisor",
    "supervisor",
    "safe_spawn_context",
]

_logger = logging.getLogger("wise.qccd.supervisor")


class ProcessSupervisor:
    """
    Thread-safe singleton that tracks all child PIDs spawned by the WISE
    SAT routing pipeline and kills them on interpreter exit.

    All public methods are safe to call from any thread or process by
    guarding state with a ``threading.Lock``. The ``atexit`` handler
    runs in the main thread when Python shuts down (including notebook
    kernel restarts).
    
    Attributes
    ----------
    MAX_CONCURRENT_PROCESSES : int
        Default maximum concurrent child processes (CPU count - 1).
    """
    
    MAX_CONCURRENT_PROCESSES: int = max(1, (os.cpu_count() or 4) - 1)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pids: Set[int] = set()
        self._atexit_registered = False

    # ------------------------------------------------------------------ public

    def register(self, pid: int) -> None:
        """Register a child PID for tracking. Installs atexit on first use."""
        with self._lock:
            self._pids.add(pid)
            if not self._atexit_registered:
                atexit.register(self._atexit_cleanup)
                self._atexit_registered = True

    def unregister(self, pid: int) -> None:
        """Remove a PID from tracking (child exited normally)."""
        with self._lock:
            self._pids.discard(pid)

    def active_count(self) -> int:
        """Return the number of currently-tracked child PIDs."""
        with self._lock:
            return len(self._pids)

    def active_pids(self) -> Set[int]:
        """Return a snapshot of currently-tracked PIDs."""
        with self._lock:
            return set(self._pids)
    
    def can_spawn(self, max_processes: int = None) -> bool:
        """Check if we can spawn another process without oversubscribing.
        
        Parameters
        ----------
        max_processes : int, optional
            Maximum allowed concurrent processes. 
            Defaults to MAX_CONCURRENT_PROCESSES.
            
        Returns
        -------
        bool
            True if spawning another process is safe.
        """
        limit = max_processes or self.MAX_CONCURRENT_PROCESSES
        return self.active_count() < limit

    def kill_all(self, grace_period: float = 1.0) -> int:
        """
        SIGTERM -> wait -> SIGKILL all tracked child PIDs.

        Returns the number of processes that were signalled.

        Safe to call multiple times; already-dead processes are silently
        ignored (catches ``ProcessLookupError``).
        
        Parameters
        ----------
        grace_period : float
            Time to wait between SIGTERM and SIGKILL.
            
        Returns
        -------
        int
            Number of processes signalled.
        """
        with self._lock:
            pids = set(self._pids)

        if not pids:
            return 0

        killed = 0

        # Phase 1: SIGTERM
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
            except (ProcessLookupError, PermissionError, OSError):
                pass

        if grace_period > 0:
            time.sleep(grace_period)

        # Phase 2: SIGKILL survivors
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass

        # Phase 3: reap zombies (best-effort, non-blocking)
        for pid in pids:
            try:
                os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                pass

        with self._lock:
            self._pids -= pids

        if killed > 0:
            _logger.info(
                "ProcessSupervisor.kill_all: signalled %d child processes",
                killed,
            )

        return killed

    # ----------------------------------------------------------------- private

    def _atexit_cleanup(self) -> None:
        """Called automatically by ``atexit`` when the interpreter exits."""
        n = self.kill_all(grace_period=0.5)
        if n > 0:
            # Use stderr directly - logging may already be torn down
            try:
                sys.stderr.write(
                    f"[ProcessSupervisor] atexit: cleaned up {n} "
                    f"orphaned child processes\n"
                )
                sys.stderr.flush()
            except Exception:
                pass


# ----------------------------------------------------------------- singleton
#: Module-level singleton - import this from any module.
supervisor: ProcessSupervisor = ProcessSupervisor()


# ----------------------------------------------------------------- helpers

def safe_spawn_context():
    """
    Return a ``multiprocessing.context`` that is safe for the current platform.

    * macOS: always ``"spawn"`` (fork is deprecated since Python 3.12 and
      causes deadlocks after prior ``spawn`` usage).
    * Linux: ``"fork"`` (faster, safe when not nested inside a spawned child).
    * Nested inside a spawned worker: always ``"spawn"``.
    
    Returns
    -------
    multiprocessing.context.BaseContext
        Safe spawn context for the platform.
    """
    import multiprocessing as mp

    # Detect if we're inside a spawned child process
    _in_spawned_child = (
        mp.current_process().name != "MainProcess"
        and mp.get_start_method(allow_none=True) == "spawn"
    )

    if _in_spawned_child or sys.platform == "darwin":
        method = "spawn"
    else:
        method = "fork"

    try:
        return mp.get_context(method)
    except ValueError:
        return mp.get_context()


def get_safe_worker_count(
    requested: int = None,
    min_workers: int = 1,
    max_workers: int = None,
) -> int:
    """
    Calculate safe number of workers for parallel SAT solving.
    
    Prevents CPU oversubscription by considering:
    - Currently active child processes
    - System CPU count
    - User-requested worker count
    
    Parameters
    ----------
    requested : int, optional
        Number of workers requested by user.
    min_workers : int
        Minimum workers to return.
    max_workers : int, optional
        Maximum workers to allow. Defaults to CPU count - 1.
        
    Returns
    -------
    int
        Safe number of workers to spawn.
    """
    cpu_count = os.cpu_count() or 4
    
    if max_workers is None:
        max_workers = max(1, cpu_count - 1)
    
    # Account for already-active processes
    active = supervisor.active_count()
    available = max(min_workers, max_workers - active)
    
    if requested is not None:
        return max(min_workers, min(requested, available))
    
    return available
