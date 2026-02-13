# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/solvers.py
"""
Process-isolated SAT and MaxSAT solvers for WISE routing.

This module provides subprocess-based wrappers around pysat solvers to handle:
- Wall-clock timeouts (Minisat22 has no built-in timeout)
- Process isolation to prevent resource leaks from RC2.compute()
- Pickle-based IPC for solver results

The subprocess pattern is essential because:
1. Minisat22's conflict budget is a heuristic proxy, not a guarantee
2. RC2.compute() blocks indefinitely and cannot be interrupted from a thread
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import signal
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from .config import wise_logger, _PYSAT_AVAILABLE

if TYPE_CHECKING:
    from .sat_encoder import WISESATContext

# Conditional pysat imports
if _PYSAT_AVAILABLE:
    from pysat.solvers import Minisat22
else:
    Minisat22 = None  # type: ignore[misc,assignment]


__all__ = [
    "run_sat_with_timeout",
    "run_rc2_with_timeout",
    # Parallel config search (ported from old qccd_operations.py)
    "wise_safe_sat_pool_workers",
    "enumerate_pmax_configs",
    "sat_config_worker",
    "parallel_config_search",
]


# =============================================================================
# Subprocess Workers
# =============================================================================


def _sat_subprocess_worker(
    cnf_path: str,
    result_path: str,
    assumptions: Optional[List[int]] = None,
) -> None:
    """Subprocess entry: load CNF, run Minisat22, dump result via pickle."""
    try:
        t0 = time.time()
        with open(cnf_path, "rb") as f:
            cnf = pickle.load(f)
        with Minisat22(bootstrap_with=cnf.clauses) as sat:
            sat_ok = sat.solve(assumptions=assumptions) if assumptions else sat.solve()
            model = sat.get_model() if sat_ok else None
        data = {"sat": bool(sat_ok), "model": model, "time": time.time() - t0}
    except KeyboardInterrupt:
        data = {"error": "KeyboardInterrupt"}
    except Exception as e:
        data = {"error": repr(e)}
    try:
        with open(result_path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass


def _rc2_subprocess_worker(wcnf_path: str, result_path: str) -> None:
    """Subprocess entry: load WCNF, run RC2, dump result via pickle."""
    try:
        from pysat.examples.rc2 import RC2 as _RC2

        t0 = time.time()
        with open(wcnf_path, "rb") as f:
            wcnf = pickle.load(f)
        rc2 = _RC2(wcnf)
        model = rc2.compute()
        cost = rc2.cost if model is not None else None
        data = {"model": model, "cost": cost, "time": time.time() - t0}
    except KeyboardInterrupt:
        data = {"error": "KeyboardInterrupt"}
    except Exception as e:
        data = {"error": repr(e)}
    try:
        with open(result_path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass


# =============================================================================
# Subprocess Runner
# =============================================================================


def _run_solver_in_subprocess(
    target_func: Callable,
    args: Tuple,
    timeout_s: float,
    label: str = "solver",
) -> Dict[str, Any]:
    """Run a solver function in a subprocess with wall-clock timeout.

    Pattern from old code's run_sat_with_timeout_file / run_rc2_with_timeout_file:
      - Pickle IPC via temp files
      - 0.5s poll loop
      - Hard kill on timeout: terminate → join(5s) → kill

    Returns dict with solver result or {"error": ..., "status": "timeout"}.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"input_{label}.pkl")
        result_path = os.path.join(tmpdir, f"result_{label}.pkl")

        # Serialize input to file
        formula = args[0]
        with open(input_path, "wb") as f:
            pickle.dump(formula, f)

        extra_args = args[1:] if len(args) > 1 else ()
        proc_args = (input_path, result_path) + extra_args

        p = mp.Process(target=target_func, args=proc_args)
        p.daemon = True

        try:
            p.start()
        except Exception as e:
            return {"error": repr(e), "status": "error"}

        start = time.time()
        status = None

        try:
            while True:
                p.join(0.5)
                if not p.is_alive():
                    status = "finished"
                    break
                if time.time() - start >= timeout_s:
                    status = "timeout"
                    break
        except KeyboardInterrupt:
            if p.is_alive():
                p.terminate()
                try:
                    p.join(5.0)
                except Exception:
                    pass
            return {"error": "user_abort", "status": "user_abort"}

        if status == "timeout":
            wise_logger.debug(
                "%s subprocess exceeded %.1f s; killing", label, timeout_s
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
            return {"error": "timeout", "status": "timeout"}

        if not os.path.exists(result_path):
            return {"error": "no result file", "status": "error"}

        try:
            with open(result_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            return {"error": repr(e), "status": "error"}

        if "error" in data:
            data["status"] = "error"
        else:
            data["status"] = "ok"
        return data


# =============================================================================
# Public API
# =============================================================================


def run_sat_with_timeout(
    cnf,
    timeout_s: float,
    assumptions: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run Minisat22 SAT solver with wall-clock timeout.

    Parameters
    ----------
    cnf : pysat.formula.CNF
        The CNF formula to solve.
    timeout_s : float
        Maximum wall-clock time in seconds.
    assumptions : Optional[List[int]]
        Assumption literals for incremental solving.

    Returns
    -------
    dict
        Result with keys: "sat" (bool), "model" (list or None),
        "time" (float), "status" ("ok"|"timeout"|"error"),
        optionally "error" (str).
    """
    if not _PYSAT_AVAILABLE:
        return {"error": "pysat not available", "status": "error", "sat": False}

    args = (cnf, assumptions) if assumptions else (cnf,)
    return _run_solver_in_subprocess(
        _sat_subprocess_worker, args, timeout_s, label="sat"
    )


def run_rc2_with_timeout(wcnf, timeout_s: float) -> Dict[str, Any]:
    """Run RC2 MaxSAT solver with wall-clock timeout.

    Parameters
    ----------
    wcnf : pysat.formula.WCNF
        The weighted CNF formula to solve.
    timeout_s : float
        Maximum wall-clock time in seconds.

    Returns
    -------
    dict
        Result with keys: "model" (list or None), "cost" (int or None),
        "time" (float), "status" ("ok"|"timeout"|"error"),
        optionally "error" (str).
    """
    if not _PYSAT_AVAILABLE:
        return {"error": "pysat not available", "status": "error", "model": None}

    return _run_solver_in_subprocess(
        _rc2_subprocess_worker, (wcnf,), timeout_s, label="rc2"
    )

# =============================================================================
# Parallel Config Search (ported from old qccd_operations.py)
# =============================================================================


def wise_safe_sat_pool_workers(num_configs: int) -> int:
    """Determine safe number of SAT pool workers.
    
    Conservative helper to decide how many worker processes the SAT pool
    is allowed to spawn. This prevents runaway process creation when called
    from within an outer ProcessPoolExecutor.
    
    Parameters
    ----------
    num_configs : int
        Number of configurations to solve.
    
    Returns
    -------
    int
        Recommended number of workers.
    """
    # Check environment variable for explicit override
    env_workers = os.environ.get("WISE_SAT_WORKERS", "")
    if env_workers:
        try:
            return max(1, int(env_workers))
        except ValueError:
            pass
    
    try:
        available = mp.cpu_count() or 1
    except Exception:
        available = 1
    
    # Hard cap at available CPUs — old ground truth caps at 4
    # But also don't spawn more workers than configs
    max_workers = min(available, num_configs, 4)  # cap at 4 (matches old code)
    
    # On small machines (≤ 2 cores), use 1 worker (matches old code)
    if available <= 2:
        max_workers = 1
    
    return max(1, max_workers)


def enumerate_pmax_configs(
    P_min: int,
    P_max_limit: int,
    step: int = 1,
    *,
    capacity_steps: int = 6,
    capacity_min: float = 0.0,
) -> Iterable[Tuple[int, float]]:
    """Enumerate (P_max, boundary_capacity_factor) pairs for SAT search.
    
    The capacity factor scales the number of ions that are forced into
    boundary bands for CROSS_BOUNDARY constraints.
    
    Parameters
    ----------
    P_min : int
        Minimum P_max value.
    P_max_limit : int
        Maximum P_max value.
    step : int
        Step size for P_max enumeration.
    capacity_steps : int
        Number of capacity factor values to try.
    capacity_min : float
        Minimum capacity factor (typically 0).
    
    Yields
    ------
    Tuple[int, float]
        (P_max, boundary_capacity_factor) pairs.
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


def sat_config_worker(
    cfg: Tuple[int, float],
    *,
    build_formula_fn: Callable,
    solve_fn: Callable,
    decode_fn: Callable,
    context_data: Dict[str, Any],
    optimize_round_start: int,
    max_sat_time: float,
    max_rc2_time: float,
    use_soft_bt: bool = False,
    progress_path: Optional[str] = None,
    stop_event: Optional[Any] = None,
) -> Dict[str, Any]:
    """Worker for a single (P_max, capacity_factor) configuration.
    
    Performs binary search over Σ_r P_r bounds to find the tightest
    satisfiable schedule.
    
    Parameters
    ----------
    cfg : Tuple[int, float]
        (P_max, boundary_capacity_factor) configuration.
    build_formula_fn : Callable
        Function to build SAT/WCNF formula given (P_max, sum_bound_B, ...).
    solve_fn : Callable
        Function to solve the formula with timeout.
    decode_fn : Callable
        Function to decode the model into a schedule.
    context_data : Dict[str, Any]
        Serializable context data for formula building.
    optimize_round_start : int
        Round index where optimization starts.
    max_sat_time : float
        Timeout for SAT solving.
    max_rc2_time : float
        Timeout for RC2 MaxSAT solving.
    use_soft_bt : bool
        Whether to use soft BT constraints (MaxSAT).
    progress_path : Optional[str]
        Path to save partial progress.
    stop_event : Optional[Any]
        Event to signal early termination.
    
    Returns
    -------
    Dict[str, Any]
        Result with keys: "P_max", "boundary_capacity_factor", "status",
        "sat", "schedule", "per_round_usage", "model", optionally "cost".
    """
    P_max, boundary_capacity_factor = cfg
    num_rounds = context_data.get("num_rounds", 1)
    rounds_under_sum = max(1, num_rounds - optimize_round_start)
    max_bound_B = rounds_under_sum * P_max
    low = 1
    high = max_bound_B
    best_result: Optional[Dict[str, Any]] = None
    last_result: Optional[Dict[str, Any]] = None

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
            # Build formula
            formula_data = build_formula_fn(
                P_max=P_max,
                sum_bound_B=sum_bound_B,
                use_wcnf=use_soft_bt,
                boundary_capacity_factor=boundary_capacity_factor,
                context_data=context_data,
            )
            
            formula = formula_data.get("formula")
            vpool = formula_data.get("vpool")
            
            timeout = max_rc2_time if use_soft_bt else max_sat_time
            
            # Solve
            if use_soft_bt:
                result = run_rc2_with_timeout(formula, timeout)
                sat_ok = result.get("status") == "ok" and result.get("model") is not None
                model = result.get("model")
                cost = result.get("cost")
            else:
                result = run_sat_with_timeout(formula, timeout)
                sat_ok = result.get("sat", False)
                model = result.get("model")
                cost = None
            
            status = result.get("status", "error")
            
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

        if status != "ok" or not sat_ok or model is None:
            return {
                "P_max": P_max,
                "boundary_capacity_factor": boundary_capacity_factor,
                "status": status if status != "ok" else "unsat",
                "sat": bool(sat_ok),
                "schedule": None,
                "per_round_usage": None,
                "model": None,
                "cost": cost,
            }

        # Decode schedule and compute usage
        try:
            schedule, per_round_usage = decode_fn(
                model=model,
                vpool=vpool,
                P_max=P_max,
                context_data=context_data,
            )
        except Exception as e:
            wise_logger.warning("Failed to decode schedule: %s", e)
            schedule = None
            per_round_usage = None

        return {
            "P_max": P_max,
            "boundary_capacity_factor": boundary_capacity_factor,
            "status": "ok",
            "sat": True,
            "schedule": schedule,
            "per_round_usage": per_round_usage,
            "model": model,
            "cost": cost,
        }

    # Binary search over sum_bound_B
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

        if (
            result.get("status") == "ok"
            and result.get("sat")
            and result.get("model") is not None
        ):
            per_round = result.get("per_round_usage") or []
            sum_usage = int(sum(per_round)) if per_round else mid
            wise_logger.debug(
                "[WISE] config P_max=%d cap=%.2f: SAT at ΣP=%d (usage=%d)",
                P_max,
                boundary_capacity_factor,
                mid,
                sum_usage,
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


def parallel_config_search(
    configs: Sequence[Tuple[int, float]],
    *,
    build_formula_fn: Callable,
    solve_fn: Callable,
    decode_fn: Callable,
    context_data: Dict[str, Any],
    optimize_round_start: int = 0,
    max_sat_time: float = 300.0,
    max_rc2_time: float = 300.0,
    use_soft_bt: bool = False,
    global_timeout: Optional[float] = None,
    score_fn: Optional[Callable[[Dict[str, Any]], Tuple[Any, ...]]] = None,
) -> Dict[str, Any]:
    """Run parallel SAT config search over multiple (P_max, cap_factor) pairs.
    
    Each config worker performs binary search over Σ_r P_r bounds.
    Returns the best satisfiable result.
    
    Parameters
    ----------
    configs : Sequence[Tuple[int, float]]
        List of (P_max, boundary_capacity_factor) configurations.
    build_formula_fn : Callable
        Function to build SAT formula.
    solve_fn : Callable
        Function to solve the formula.
    decode_fn : Callable
        Function to decode the model.
    context_data : Dict[str, Any]
        Serializable context for formula building.
    optimize_round_start : int
        Round index where optimization starts.
    max_sat_time : float
        Timeout for SAT solving per config.
    max_rc2_time : float
        Timeout for RC2 MaxSAT per config.
    use_soft_bt : bool
        Whether to use soft BT constraints.
    global_timeout : Optional[float]
        Overall timeout for the entire search.
    score_fn : Optional[Callable]
        Function to score results for comparison.
    
    Returns
    -------
    Dict[str, Any]
        Best result found, or error dict if no satisfiable result.
    
    Raises
    ------
    RuntimeError
        If no feasible layout is found.
    """
    if not configs:
        return {"status": "error", "error": "empty config set", "sat": False}
    
    max_workers = wise_safe_sat_pool_workers(len(configs))
    
    wise_logger.info(
        "[WISE] launching parallel SAT pool over %d configs with %d workers",
        len(configs),
        max_workers,
    )
    
    # Set up progress tracking
    progress_dir = tempfile.mkdtemp(prefix="wise_sat_pool_")
    progress_paths: Dict[int, str] = {}
    
    # Manager for cross-process communication
    try:
        manager = mp.Manager()
        stop_event = manager.Event()
    except Exception:
        manager = None
        stop_event = None
    
    # Determine context for multiprocessing
    try:
        pool_context = mp.get_context("fork")
    except ValueError:
        pool_context = mp.get_context()
    
    results: List[Dict[str, Any]] = []
    start_time = time.time()
    timed_out = False
    
    solver_timeout = max_rc2_time if use_soft_bt else max_sat_time
    effective_global = global_timeout or (solver_timeout * 2.0)
    
    try:
        executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=pool_context)
        futures = []
        
        for idx, cfg in enumerate(configs):
            progress_path = os.path.join(progress_dir, f"cfg_{idx}.pkl")
            progress_paths[idx] = progress_path
            
            fut = executor.submit(
                sat_config_worker,
                cfg,
                build_formula_fn=build_formula_fn,
                solve_fn=solve_fn,
                decode_fn=decode_fn,
                context_data=context_data,
                optimize_round_start=optimize_round_start,
                max_sat_time=max_sat_time,
                max_rc2_time=max_rc2_time,
                use_soft_bt=use_soft_bt,
                progress_path=progress_path,
                stop_event=stop_event,
            )
            futures.append((idx, cfg, fut))
        
        # Poll for completion
        while True:
            unfinished = [1 for _, _, fut in futures if not fut.done()]
            if not unfinished:
                break
            if time.time() - start_time >= effective_global:
                wise_logger.info(
                    "[WISE] global SAT pool budget exhausted; collecting results"
                )
                timed_out = True
                if stop_event is not None:
                    stop_event.set()
                for _, _, fut in futures:
                    fut.cancel()
                break
            time.sleep(0.05)
        
        # Collect results
        for idx, cfg, fut in futures:
            if fut.done():
                try:
                    res = fut.result()
                    results.append(res)
                    wise_logger.debug(
                        "[WISE] pool result: P_max=%s, cap_factor=%.2f, status=%s",
                        res.get("P_max"),
                        res.get("boundary_capacity_factor", 0),
                        res.get("status"),
                    )
                except Exception as e:
                    wise_logger.warning("[WISE] config %s crashed: %s", cfg, e)
            else:
                # Try to recover partial result
                progress_path = progress_paths.get(idx)
                if progress_path and os.path.exists(progress_path):
                    try:
                        with open(progress_path, "rb") as f:
                            res = pickle.load(f)
                        results.append(res)
                    except Exception:
                        pass
    
    finally:
        # Cleanup — faithful port of old code's aggressive child PID cleanup.
        # On timeout the old code did SIGTERM → sleep(0.5) → SIGKILL on every
        # known SAT/RC2 subprocess PID.  We replicate this by grabbing the
        # executor's internal worker PIDs and escalating.
        try:
            if "executor" in locals():
                worker_pids = list(getattr(executor, '_processes', {}).keys())
                executor.shutdown(wait=False, cancel_futures=True)
                if timed_out and worker_pids:
                    for pid in worker_pids:
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except (ProcessLookupError, PermissionError, OSError):
                            pass
                    time.sleep(0.5)
                    for pid in worker_pids:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError, OSError):
                            pass
        except Exception:
            pass
        
        # Clean up temp dir
        try:
            import shutil
            shutil.rmtree(progress_dir, ignore_errors=True)
        except Exception:
            pass
    
    # Filter for satisfiable results
    sat_results = [r for r in results if r.get("sat") and r.get("status") == "ok"]
    
    if not sat_results:
        return {
            "status": "unsat",
            "error": f"No feasible layout for any config over {len(configs)} configs",
            "sat": False,
        }
    
    # Select best result
    def default_score(res: Dict[str, Any]) -> Tuple[float, int, int]:
        usage = res.get("per_round_usage") or []
        sum_usage = sum(usage) if usage else float("inf")
        return (
            -res.get("boundary_capacity_factor", 0),
            sum_usage,
            res.get("P_max", float("inf")),
        )
    
    score_fn = score_fn or default_score
    best_res = min(sat_results, key=score_fn)
    
    wise_logger.info(
        "[WISE] best config: P_max=%d, cap_factor=%.2f",
        best_res.get("P_max"),
        best_res.get("boundary_capacity_factor", 0),
    )
    
    return best_res