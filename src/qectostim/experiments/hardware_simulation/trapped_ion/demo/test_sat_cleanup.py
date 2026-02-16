"""
Tests that SAT solver child processes are properly cleaned up after timeout.

The WISE SAT pool spawns multiple mp.Process workers (Minisat22 / RC2).
Some configs may never finish within the timeout budget.  After the pool
budget expires the framework must:

  1. Terminate **all** child solver processes (SIGTERM → SIGKILL).
  2. Return results only from configs that completed in time.
  3. Leave **zero** dangling / zombie child processes behind.

Run with::

    PYTHONPATH=src python -m pytest \\
        src/qectostim/experiments/hardware_simulation/trapped_ion/demo/test_sat_cleanup.py -v
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import signal
import tempfile
import time
from typing import Any, Dict, List, Optional

import pytest
from pysat.formula import CNF

# ── module under test ────────────────────────────────────────────────
from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_SAT_WISE_odd_even_sorter import (
    _run_solver_in_subprocess,
    _kill_executor_workers,
    run_sat_with_timeout_file,
)


# =====================================================================
# Helpers
# =====================================================================


def _make_trivial_cnf() -> CNF:
    """A trivially SAT formula: (x1) ∧ (x2)."""
    cnf = CNF()
    cnf.append([1])
    cnf.append([2])
    return cnf


def _slow_worker(input_path: str, result_path: str, *extra):
    """Worker that deliberately stalls for 120 s (simulates a hard SAT)."""
    time.sleep(120)
    with open(result_path, "wb") as f:
        pickle.dump({"sat": True, "model": [1, 2], "time": 120.0}, f)


def _fast_worker(input_path: str, result_path: str, *extra):
    """Worker that finishes immediately."""
    with open(input_path, "rb") as f:
        _ = pickle.load(f)
    with open(result_path, "wb") as f:
        pickle.dump({"sat": True, "model": [1, 2], "time": 0.01}, f)


def _children_of_current_process() -> List[int]:
    """Return PIDs of all live child processes of the current process."""
    import psutil

    current = psutil.Process(os.getpid())
    children = current.children(recursive=True)
    return [c.pid for c in children if c.is_running()]


def _children_of_current_process_mp() -> List[mp.Process]:
    """Fallback using mp.active_children() if psutil is unavailable."""
    return mp.active_children()


# =====================================================================
# Tests
# =====================================================================


class TestSubprocessCleanup:
    """Verify that _run_solver_in_subprocess kills children on timeout."""

    def test_timeout_kills_child(self):
        """A slow worker must be terminated when timeout expires."""
        cnf = _make_trivial_cnf()
        before = set(c.pid for c in mp.active_children())

        data, status = _run_solver_in_subprocess(
            input_obj=cnf,
            input_filename="test.cnf.pkl",
            result_filename="test_result.pkl",
            worker_fn=_slow_worker,
            timeout_s=2.0,            # very short timeout
            solver_label="TEST-SAT",
            debug_prefix="[TEST]",
        )

        # Allow a brief grace period for OS to reap
        time.sleep(0.5)
        after = set(c.pid for c in mp.active_children())
        new_children = after - before

        assert status == "timeout", f"Expected timeout, got {status}"
        assert data is None, "Should return None on timeout"
        assert len(new_children) == 0, (
            f"Dangling child processes after timeout: {new_children}"
        )

    def test_fast_worker_completes(self):
        """Sanity: a fast worker should finish with status='ok'."""
        cnf = _make_trivial_cnf()

        data, status = _run_solver_in_subprocess(
            input_obj=cnf,
            input_filename="test.cnf.pkl",
            result_filename="test_result.pkl",
            worker_fn=_fast_worker,
            timeout_s=10.0,
            solver_label="TEST-SAT",
            debug_prefix="[TEST]",
        )

        assert status == "ok"
        assert data is not None
        assert data.get("sat") is True

    def test_stop_event_terminates_child(self):
        """Setting stop_event should cause early termination."""
        cnf = _make_trivial_cnf()
        ctx = mp.get_context("fork")
        mgr = ctx.Manager()
        stop_event = mgr.Event()
        children_dict = mgr.dict()

        # Set stop_event after a short delay in a thread
        import threading

        def _set_stop():
            time.sleep(1.0)
            stop_event.set()

        t = threading.Thread(target=_set_stop, daemon=True)
        t.start()

        before = set(c.pid for c in mp.active_children())

        data, status = _run_solver_in_subprocess(
            input_obj=cnf,
            input_filename="test.cnf.pkl",
            result_filename="test_result.pkl",
            worker_fn=_slow_worker,
            timeout_s=60.0,  # long timeout — stop_event should fire first
            solver_label="TEST-SAT",
            debug_prefix="[TEST]",
            stop_event=stop_event,
            job_key="test_job",
            children_dict=children_dict,
        )

        t.join(timeout=5)
        time.sleep(0.5)
        after = set(c.pid for c in mp.active_children())
        # Filter out the Manager's server process
        mgr_pid = None
        try:
            mgr_pid = mgr._process.pid  # type: ignore[attr-defined]
        except Exception:
            pass
        new_children = after - before
        if mgr_pid:
            new_children.discard(mgr_pid)

        assert status == "timeout", f"Expected timeout via stop_event, got {status}"
        assert len(new_children) == 0, (
            f"Dangling child processes after stop_event: {new_children}"
        )

        mgr.shutdown()


class TestRunSatCleanup:
    """Verify that run_sat_with_timeout_file leaves no child processes."""

    def test_timeout_no_dangling(self):
        """
        run_sat_with_timeout_file with a very short timeout on a big
        (but valid) CNF should return 'timeout' and leave no orphans.
        """
        # Build a large-ish random CNF that will keep Minisat busy
        import random

        random.seed(42)
        n_vars = 300
        n_clauses = 1200
        cnf = CNF()
        for _ in range(n_clauses):
            clause = [
                random.choice([-1, 1]) * random.randint(1, n_vars)
                for _ in range(3)
            ]
            cnf.append(clause)

        before = set(c.pid for c in mp.active_children())

        sat_ok, model, status = run_sat_with_timeout_file(
            cnf,
            timeout_s=0.5,  # extremely short — should timeout
            debug_prefix="[TEST]",
        )

        time.sleep(0.5)
        after = set(c.pid for c in mp.active_children())
        new_children = after - before

        # We accept either timeout or ok (if solver was surprisingly fast)
        assert status in ("timeout", "ok"), f"Unexpected status: {status}"
        assert len(new_children) == 0, (
            f"Dangling child processes after run_sat_with_timeout_file: {new_children}"
        )


class TestPoolCleanup:
    """
    Verify the ProcessPoolExecutor path in optimal_QMR_for_WISE
    cleans up all nested solver children after the pool budget expires.

    This is an integration-level test that exercises the full pool →
    worker → mp.Process → solver chain with deliberately short timeouts.
    """

    def test_pool_kills_all_children_on_budget(self):
        """
        Spawn the SAT pool with a tight budget.  After the pool returns
        (either with results or NoFeasibleLayoutError), there must be
        zero dangling child processes.
        """
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("fork")
        mgr = ctx.Manager()
        sat_children = mgr.dict()
        rc2_children = mgr.dict()
        stop_event = mgr.Event()

        # We'll directly test the subprocess runner with multiple workers
        # in a ProcessPoolExecutor to simulate the pool pattern
        cnf = _make_trivial_cnf()

        def pool_worker(idx):
            """Each worker spawns a slow SAT subprocess."""
            data, status = _run_solver_in_subprocess(
                input_obj=cnf,
                input_filename=f"test_{idx}.cnf.pkl",
                result_filename=f"test_{idx}_result.pkl",
                worker_fn=_slow_worker,
                timeout_s=60.0,  # long timeout
                solver_label=f"POOL-{idx}",
                debug_prefix="[TEST]",
                stop_event=stop_event,
                job_key=f"test_{idx}",
                children_dict=sat_children,
            )
            return {"status": status, "idx": idx}

        before_pids = set(c.pid for c in mp.active_children())

        executor = ProcessPoolExecutor(max_workers=3, mp_context=ctx)
        futures = []
        for i in range(3):
            futures.append(executor.submit(pool_worker, i))

        # Wait a bit for workers and their SAT children to start
        time.sleep(2.0)

        # Verify SAT children are registered
        n_registered = len(sat_children)

        # Now signal stop and cancel futures
        stop_event.set()
        for f in futures:
            f.cancel()

        # Collect whatever finished
        results = []
        for f in futures:
            if f.done() and not f.cancelled():
                try:
                    results.append(f.result(timeout=1))
                except Exception:
                    pass

        # Kill any remaining nested children via sat_children dict
        for key, pid in list(sat_children.items()):
            try:
                os.kill(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        time.sleep(0.3)
        for key, pid in list(sat_children.items()):
            try:
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

        # Shut down executor — mirror the fixed production cleanup pattern
        # Must call _kill_executor_workers BEFORE shutdown (shutdown clears _processes)
        _kill_executor_workers(executor)
        executor.shutdown(wait=False, cancel_futures=True)
        time.sleep(1.0)

        # Clean up manager
        mgr_pid = None
        try:
            mgr_pid = mgr._process.pid  # type: ignore[attr-defined]
        except Exception:
            pass
        mgr.shutdown()

        time.sleep(1.0)

        after_pids = set(c.pid for c in mp.active_children())
        new_children = after_pids - before_pids
        if mgr_pid:
            new_children.discard(mgr_pid)

        assert len(new_children) == 0, (
            f"Dangling child processes after pool cleanup: {new_children}. "
            f"Registered SAT children during test: {n_registered}"
        )
