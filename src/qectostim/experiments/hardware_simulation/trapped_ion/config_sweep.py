"""
Parallel configuration search for WISE routing.

Faithful port of old ``best_effort_compilation_WISE.py``:
    ``search_configs_best_exec_time`` → ``WISEConfigSearch.search()``

The module uses ``ProcessPoolExecutor`` to evaluate multiple
(lookahead, subgrid_width, subgrid_height, subgrid_increment) tuples
in parallel, each running the full WISE compilation pipeline.  A
global time budget and per-config SAT budgets prevent runaway workers.

Usage
-----
>>> from qectostim.experiments.hardware_simulation.trapped_ion.config_sweep import (
...     WISEConfigSearch,
... )
>>> searcher = WISEConfigSearch(d=3, m_traps=6, n_traps=6, trap_capacity=2)
>>> best, all_results = searcher.search(
...     configs=[(2, 6, 4, 1), (3, 8, 6, 2)],
...     time_budget_s=120.0,
... )
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

wise_logger = logging.getLogger("wise.qccd.config_sweep")


# ====================================================================
# Helper: per-process logging setup (matches old configure_wise_logging)
# ====================================================================

def _configure_wise_logging(d: int, k: int, log_dir: str = "logs") -> None:
    """Set up file-based logging for WISE route / SAT inside a worker."""
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [pid=%(process)d] %(message)s"
    )

    for logger_name, tag in [
        ("wise.qccd.route", "patch_route"),
        ("wise.qccd.sat", "sat_pool"),
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        path = os.path.join(log_dir, f"{tag}_memory_{d}_{k}.log")
        path_abs = os.path.abspath(path)
        has_file = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == path_abs
            for h in logger.handlers
        )
        if not has_file:
            fh = logging.FileHandler(path_abs)
            fh.setFormatter(fmt)
            logger.addHandler(fh)


# ====================================================================
# Single-config worker (executed in a subprocess)
# ====================================================================

def _run_single_config(
    lookahead: int,
    subgrid_width: int,
    subgrid_height: int,
    subgrid_increment: int,
    *,
    d: int,
    m_traps: int,
    n_traps: int,
    trap_capacity: int,
    barrier_threshold: float,
    base_pmax_in: Optional[int],
    sat_workers_per_config: int,
    time_budget_s: Optional[float],
) -> Dict[str, Any]:
    """Worker: compile one config and return a result dict.

    This is the new-architecture equivalent of the old
    ``_run_single_config_entry``.  It imports the new pipeline
    components and runs the full compile → route → measure cycle.
    """
    _configure_wise_logging(d=d, k=trap_capacity)

    # Limit SAT parallelism inside this worker
    if sat_workers_per_config > 0:
        os.environ["WISE_SAT_WORKERS"] = str(sat_workers_per_config)

    # Per-config SAT time cap
    if time_budget_s is not None and time_budget_s > 0:
        cap = max(time_budget_s / 2.0, 1.0)
        for var in ("WISE_MAX_SAT_TIME", "WISE_MAX_RC2_TIME"):
            current = os.environ.get(var)
            value = cap
            if current is not None:
                try:
                    value = min(cap, float(current))
                except ValueError:
                    pass
            os.environ[var] = f"{value}"

    t0 = time.perf_counter()
    try:
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            WISEArchitecture,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler import (
            WISECompiler,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WISERoutingConfig,
        )

        arch = WISEArchitecture(m=m_traps, n=n_traps, k=trap_capacity)

        config = WISERoutingConfig(
            subgridsize=(subgrid_width, subgrid_height, subgrid_increment),
            lookahead_rounds=lookahead,
        )
        compiler = WISECompiler(architecture=arch, routing_config=config)

        # Import stim for circuit generation
        import stim

        stim_circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=1,
            distance=d,
        )

        result = compiler.compile(stim_circuit)

        t1 = time.perf_counter()
        comp_time = t1 - t0

        # Extract metrics
        meta = getattr(result, "metadata", {}) or {}
        exec_time = meta.get("reconfiguration_time_us", 0.0)
        total_ops = meta.get("total_routing_swaps", 0)
        num_batches = meta.get("num_batches", 0)

        return {
            "lookahead": lookahead,
            "subgrid_width": subgrid_width,
            "subgrid_height": subgrid_height,
            "subgrid_increment": subgrid_increment,
            "d": d,
            "m_traps": m_traps,
            "n_traps": n_traps,
            "trap_capacity": trap_capacity,
            "exec_time": exec_time,
            "comp_time": comp_time,
            "Operations": total_ops,
            "num_batches": num_batches,
        }
    except Exception as e:
        t1 = time.perf_counter()
        wise_logger.warning(
            "Config failed: lookahead=%d, subgrid=(%d,%d,%d), d=%d: %s",
            lookahead,
            subgrid_width,
            subgrid_height,
            subgrid_increment,
            d,
            repr(e),
        )
        return {
            "lookahead": lookahead,
            "subgrid_width": subgrid_width,
            "subgrid_height": subgrid_height,
            "subgrid_increment": subgrid_increment,
            "d": d,
            "m_traps": m_traps,
            "n_traps": n_traps,
            "trap_capacity": trap_capacity,
            "exec_time": float("nan"),
            "comp_time": t1 - t0,
            "error": repr(e),
        }


# ====================================================================
# Public API: WISEConfigSearch
# ====================================================================

@dataclass
class WISEConfigSearch:
    """Parallel configuration search for WISE routing parameters.

    Faithful port of old ``search_configs_best_exec_time``.

    Parameters
    ----------
    d : int
        Surface code distance.
    m_traps : int
        Number of trap columns.
    n_traps : int
        Number of trap rows.
    trap_capacity : int
        Ions per trap (k).
    barrier_threshold : float
        Barrier threshold for routing.
    base_pmax_in : int | None
        Base pmax for SAT solving.
    """

    d: int = 3
    m_traps: int = 6
    n_traps: int = 6
    trap_capacity: int = 2
    barrier_threshold: float = float("inf")
    base_pmax_in: Optional[int] = None

    def search(
        self,
        configs: List[Tuple[int, int, int, int]],
        *,
        time_budget_s: Optional[float] = None,
        sat_workers_per_config: int = 4,
        max_total_workers: Optional[int] = None,
        verbose: bool = False,
        leaderboard_top_k: int = 5,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Search configs in parallel, return best + all results.

        Each config is ``(lookahead, subgrid_width, subgrid_height,
        subgrid_increment)``.

        Returns
        -------
        (best_result, all_results)
            ``best_result`` is the dict with smallest ``exec_time``,
            or None if all configs failed.  ``all_results`` is the
            list of all result dicts (including failures with NaN).
        """
        if not configs:
            return None, []

        total_cpus = mp.cpu_count() or 1
        if max_total_workers is None:
            if sat_workers_per_config > 0:
                max_total_workers = max(1, total_cpus // sat_workers_per_config)
            else:
                max_total_workers = total_cpus
        max_total_workers = max(1, min(max_total_workers, len(configs)))

        start = time.time()
        all_results: List[Dict[str, Any]] = []
        best_result: Optional[Dict[str, Any]] = None

        def _is_better(a: Dict, b: Optional[Dict]) -> bool:
            if b is None:
                return True
            ea, eb = a["exec_time"], b["exec_time"]
            if np.isnan(ea):
                return False
            if np.isnan(eb):
                return True
            if ea < eb:
                return True
            if ea > eb:
                return False
            return a.get("comp_time", 0) < b.get("comp_time", 0)

        def _print_leaderboard() -> None:
            if not verbose or not all_results:
                return
            valid = [r for r in all_results if not np.isnan(r["exec_time"])]
            if not valid:
                return
            valid_sorted = sorted(
                valid, key=lambda r: (r["exec_time"], r.get("comp_time", 0))
            )
            top = valid_sorted[: max(1, leaderboard_top_k)]
            print("[WISE-SEARCH] Current top configs (by exec_time):")
            for r in top:
                la = r["lookahead"]
                w = r["subgrid_width"]
                h = r["subgrid_height"]
                inc = r["subgrid_increment"]
                et = r["exec_time"]
                ct = r.get("comp_time", 0)
                print(
                    f"  la={la}, w={w}, h={h}, inc={inc}: "
                    f"exec={et:.3f}, comp={ct:.1f}s"
                )
            print(flush=True)

        extra_kw: Dict[str, Any] = dict(
            d=self.d,
            m_traps=self.m_traps,
            n_traps=self.n_traps,
            trap_capacity=self.trap_capacity,
            barrier_threshold=self.barrier_threshold,
            base_pmax_in=self.base_pmax_in,
            sat_workers_per_config=sat_workers_per_config,
            time_budget_s=time_budget_s,
        )

        ctx = mp.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=max_total_workers, mp_context=ctx
        )
        future_to_cfg: Dict[Any, Tuple[int, int, int, int]] = {}

        try:
            for cfg in configs:
                fut = executor.submit(
                    _run_single_config,
                    lookahead=cfg[0],
                    subgrid_width=cfg[1],
                    subgrid_height=cfg[2],
                    subgrid_increment=cfg[3],
                    **extra_kw,
                )
                future_to_cfg[fut] = cfg

            pending = set(future_to_cfg.keys())
            timed_out = False

            while pending:
                if time_budget_s is not None and (time.time() - start) >= time_budget_s:
                    timed_out = True
                    if verbose:
                        print(
                            "[WISE-SEARCH] Global time budget reached; "
                            "stopping collection.",
                            flush=True,
                        )
                    break

                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=1.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                if not done:
                    continue

                for fut in done:
                    cfg = future_to_cfg[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        if verbose:
                            print(
                                f"[WISE-SEARCH] crashed for cfg={cfg}: {e}",
                                flush=True,
                            )
                        res = {
                            "lookahead": cfg[0],
                            "subgrid_width": cfg[1],
                            "subgrid_height": cfg[2],
                            "subgrid_increment": cfg[3],
                            "exec_time": float("nan"),
                            "comp_time": 0.0,
                            "error": repr(e),
                        }
                    all_results.append(res)
                    if _is_better(res, best_result):
                        best_result = res
                        if verbose:
                            print(
                                "[WISE-SEARCH] New best config found.",
                                flush=True,
                            )
                            _print_leaderboard()

        except KeyboardInterrupt:
            if verbose:
                print(
                    "[WISE-SEARCH] KeyboardInterrupt; cancelling.",
                    flush=True,
                )
        finally:
            # Cancel remaining futures
            for f in future_to_cfg:
                if not f.done():
                    f.cancel()

            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

            # Force-terminate lingering workers
            try:
                procs = getattr(executor, "_processes", None)
                if procs:
                    for pid, proc in list(procs.items()):
                        if proc.is_alive():
                            try:
                                proc.terminate()
                            except Exception:
                                pass
                    for pid, proc in list(procs.items()):
                        if proc.is_alive():
                            try:
                                proc.join(timeout=1.0)
                            except Exception:
                                pass
                    for pid, proc in list(procs.items()):
                        if proc.is_alive():
                            try:
                                proc.kill()
                            except Exception:
                                pass
            except Exception:
                pass

        if verbose:
            elapsed = time.time() - start
            print(
                f"[WISE-SEARCH] Finished in {elapsed:.1f}s; "
                f"{len(all_results)} configs completed.",
                flush=True,
            )
            _print_leaderboard()

        return best_result, all_results
