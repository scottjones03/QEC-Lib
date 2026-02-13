import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from typing import Sequence, Dict, Any
from src.compiler.qccd_WISE_ion_route import ionRoutingWISEArch
from src.simulator.qccd_circuit import QCCDCircuit, NDE_JZ, NDE_LZ, NSE_Z
from src.utils.qccd_arch import QCCDWiseArch
from src.compiler.qccd_parallelisation import paralleliseOperationsWithBarriers
import time
import os
import time
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np  # needed for np.nan checks
import stim
import concurrent

import logging
import os

def configure_wise_logging(d: int, k: int, log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [pid=%(process)d] %(message)s")

    # ---------------- wise.qccd.route ----------------
    route_logger = logging.getLogger("wise.qccd.route")
    route_logger.setLevel(logging.INFO)

    route_path = os.path.join(log_dir, f"patch_route_memory_{d}_{k}.log")
    route_path_abs = os.path.abspath(route_path)

    # Avoid adding duplicate handlers if this is called multiple times
    has_file = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == route_path_abs
        for h in route_logger.handlers
    )
    if not has_file:
        fh = logging.FileHandler(route_path_abs)
        fh.setFormatter(fmt)
        route_logger.addHandler(fh)

    # ---------------- wise.qccd.sat ----------------
    sat_logger = logging.getLogger("wise.qccd.sat")
    sat_logger.setLevel(logging.INFO)

    sat_path = os.path.join(log_dir, f"sat_pool_memory_{d}_{k}.log")
    sat_path_abs = os.path.abspath(sat_path)
    has_file_sat = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == sat_path_abs
        for h in sat_logger.handlers
    )
    if not has_file_sat:
        fh_sat = logging.FileHandler(sat_path_abs)
        fh_sat.setFormatter(fmt)
        sat_logger.addHandler(fh_sat)


def run_single_config_cnot(
    lookahead: int,
    subgrid_width: int,
    subgrid_height: int,
    subgrid_increment: int,
    *,
    d: int = 3,                 # code distance
    trap_capacity: int = 2,
    barrier_threshold: float = np.inf,
    go_back_threshold: float = 0.0,
    base_pmax_in: int = None,
    gate_improvements: Sequence[float] = [1.0],
    num_shots: int = 100_000
) -> tuple[float, float, Dict[str, Any], float]:
    """
    Run the WISE routing for a single (lookahead, subgridsize, increment) config.

    Parameters
    ----------
    lookahead : int
        Number of two-qubit rounds to include in the local optimisation window.
    subgrid_width : int
        Initial subgrid width (in columns of the m*k ion grid).
    subgrid_height : int
        Initial subgrid height (in rows).
    subgrid_increment : int
        How many rows/cols to grow the subgrid by at each expansion.
    d : int, optional
        Surface code distance (used in QCCDCircuit.generated).
    trap_capacity : int, optional
        Ions per trap (QCCDWiseArch.k).
    barrier_threshold, go_back_threshold : float, optional
        As in your existing script.

    Returns
    -------
    (exec_time, comp_time) : (float, float)
        exec_time : circuit execution time (max parallel timestep).
        comp_time : wall-clock time for compilation for this config.
    """
    t0 = time.perf_counter()
    # Generate circuit with the chosen code distance
    circuit = QCCDCircuit(str(stim.Circuit.from_file(f"logical_cnot_d={d//2}.stim")))

    nqubitsNeeded = 4*(np.ceil(circuit.num_qubits / 3))

    n_traps = int(np.ceil(np.sqrt(nqubitsNeeded)))
    m_traps = int(np.ceil(n_traps/trap_capacity))

    try:
       
        # Architecture with parameterised rows/cols
        wiseArch = QCCDWiseArch(
            m=m_traps,
            n=n_traps,
            k=trap_capacity,
        )

        arch, (instructions, opBarriers, toMoveOps) = circuit.processCircuitWiseArch(
            wiseArch=wiseArch
        )
        arch.refreshGraph()

        # Barrier logic as in your script
        opBarriers = opBarriers if trap_capacity <= barrier_threshold else []

        subgridsize = (subgrid_width, subgrid_height, subgrid_increment)

        # Core routing
        allOps, barriers, reconfigTime = ionRoutingWISEArch(
            arch,
            wiseArch,
            instructions,
            lookahead=lookahead,
            subgridsize=subgridsize,
            base_pmax_in=base_pmax_in,
            toMoveOps=toMoveOps
        )

        # Parallel schedule
        parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
        if len(parallelOpsMap) == 0:
            exec_time = 0.0
        else:
            exec_time = float(max(parallelOpsMap.keys()))

        t1 = time.perf_counter()
        comp_time = t1 - t0

        logicalErrors = []
        physicalZErrors = []
        physicalXErrors = []
        
        for gate_improvement in gate_improvements:
            logicalError, physicalXError, physicalZError = circuit.simulate(allOps, num_shots=num_shots, error_scaling=gate_improvement)
            logicalErrors.append(logicalError)
            physicalZErrors.append(physicalZError)
            physicalXErrors.append(physicalXError)
        
        for op in parallelOpsMap.values():
            op.calculateOperationTime()
            op.calculateFidelity()

        circuit.resetArch()
        arch.refreshGraph()

        results = {}

        results["ElapsedTime"] = max(parallelOpsMap.keys())
        results["Operations"] = len(allOps)
        results["MeanConcurrency"] = np.mean([len(op.operations) for op in parallelOpsMap.values()])
        results["QubitOperations"] = len(instructions)
        results["LogicalErrorRates"] = logicalErrors
        results["PhysicalZErrorRates"] = physicalZErrors
        results["PhysicalXErrorRates"] = physicalXErrors

        Njz = np.ceil(nqubitsNeeded / trap_capacity)
        Nlz = nqubitsNeeded - Njz # note the difference because we do not have vertical traps

        Nde = NDE_LZ*Nlz+NDE_JZ*Njz
        Nse = NSE_Z*(Njz+Nlz)

        Num_electrodes =int( Nde+Nse)
        Num_DACs = int(min(100, Nde)+np.ceil(Nse/100))
        results["DACs"] = Num_DACs
        results["Electrodes"] = Num_electrodes

        return exec_time, comp_time, results, reconfigTime

    except Exception as e:
        t1 = time.perf_counter()
        comp_time = t1 - t0
        print(
            f"[WARN] Failed for lookahead={lookahead}, "
            f"subgrid=({subgrid_width},{subgrid_height},{subgrid_increment}), "
            f"d={d}, m_traps={m_traps}, n_traps={n_traps}: {repr(e)}"
        )
        results = {}
        results["ElapsedTime"] = np.nan
        results["Operations"] = np.nan
        results["MeanConcurrency"] = np.nan
        results["QubitOperations"] = np.nan
        results["LogicalErrorRates"] = [np.nan for _ in gate_improvements]
        results["PhysicalZErrorRates"] = [np.nan for _ in gate_improvements]
        results["PhysicalXErrorRates"] = [np.nan for _ in gate_improvements]
        # exec_time NaN, but keep compilation time to see failures too
        return np.nan, comp_time, results, np.nan

def run_single_config(
    lookahead: int,
    subgrid_width: int,
    subgrid_height: int,
    subgrid_increment: int,
    *,
    d: int = 4,                 # code distance
    m_traps: int = 6,           # number of trap columns in the WISE arch
    n_traps: int = 6,           # number of trap rows in the WISE arch
    trap_capacity: int = 2,
    barrier_threshold: float = np.inf,
    go_back_threshold: float = 0.0,
    base_pmax_in: int = None,
    gate_improvements: Sequence[float] = [1.0],
    num_shots: int = 100_000
) -> tuple[float, float, Dict[str, Any], float]:
    """
    Run the WISE routing for a single (lookahead, subgridsize, increment) config.

    Parameters
    ----------
    lookahead : int
        Number of two-qubit rounds to include in the local optimisation window.
    subgrid_width : int
        Initial subgrid width (in columns of the m*k ion grid).
    subgrid_height : int
        Initial subgrid height (in rows).
    subgrid_increment : int
        How many rows/cols to grow the subgrid by at each expansion.
    d : int, optional
        Surface code distance (used in QCCDCircuit.generated).
    m_traps : int, optional
        Number of trap columns in the WISE architecture (QCCDWiseArch.m).
    n_traps : int, optional
        Number of trap rows in the WISE architecture (QCCDWiseArch.n).
    trap_capacity : int, optional
        Ions per trap (QCCDWiseArch.k).
    barrier_threshold, go_back_threshold : float, optional
        As in your existing script.

    Returns
    -------
    (exec_time, comp_time) : (float, float)
        exec_time : circuit execution time (max parallel timestep).
        comp_time : wall-clock time for compilation for this config.
    """
    t0 = time.perf_counter()
    try:
        # Generate circuit with the chosen code distance
        circuit = QCCDCircuit.generated(
            "surface_code:rotated_memory_z",
            rounds=1,
            distance=d,
        )
        nqubitsNeeded = m_traps*n_traps*trap_capacity

        # Architecture with parameterised rows/cols
        wiseArch = QCCDWiseArch(
            m=m_traps,
            n=n_traps,
            k=trap_capacity,
        )

        arch, (instructions, opBarriers, _) = circuit.processCircuitWiseArch(
            wiseArch=wiseArch
        )
        arch.refreshGraph()

        # Barrier logic as in your script
        opBarriers = opBarriers if trap_capacity <= barrier_threshold else []

        subgridsize = (subgrid_width, subgrid_height, subgrid_increment)

        # Core routing
        allOps, barriers, reconfigTime = ionRoutingWISEArch(
            arch,
            wiseArch,
            instructions,
            lookahead=lookahead,
            subgridsize=subgridsize,
            base_pmax_in=base_pmax_in
        )

        # Parallel schedule
        parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
        if len(parallelOpsMap) == 0:
            exec_time = 0.0
        else:
            exec_time = float(max(parallelOpsMap.keys()))

        t1 = time.perf_counter()
        comp_time = t1 - t0

        logicalErrors = []
        physicalZErrors = []
        physicalXErrors = []
        
        for gate_improvement in gate_improvements:
            logicalError, physicalXError, physicalZError = circuit.simulate(allOps, num_shots=num_shots, error_scaling=gate_improvement)
            logicalErrors.append(logicalError)
            physicalZErrors.append(physicalZError)
            physicalXErrors.append(physicalXError)
        
        for op in parallelOpsMap.values():
            op.calculateOperationTime()
            op.calculateFidelity()

        circuit.resetArch()
        arch.refreshGraph()

        results = {}

        results["ElapsedTime"] = max(parallelOpsMap.keys())
        results["Operations"] = len(allOps)
        results["MeanConcurrency"] = np.mean([len(op.operations) for op in parallelOpsMap.values()])
        results["QubitOperations"] = len(instructions)
        results["LogicalErrorRates"] = logicalErrors
        results["PhysicalZErrorRates"] = physicalZErrors
        results["PhysicalXErrorRates"] = physicalXErrors

        Njz = np.ceil(nqubitsNeeded / trap_capacity)
        Nlz = nqubitsNeeded - Njz # note the difference because we do not have vertical traps

        Nde = NDE_LZ*Nlz+NDE_JZ*Njz
        Nse = NSE_Z*(Njz+Nlz)

        Num_electrodes =int( Nde+Nse)
        Num_DACs = int(min(100, Nde)+np.ceil(Nse/100))
        results["DACs"] = Num_DACs
        results["Electrodes"] = Num_electrodes

        return exec_time, comp_time, results, reconfigTime

    except Exception as e:
        t1 = time.perf_counter()
        comp_time = t1 - t0
        print(
            f"[WARN] Failed for lookahead={lookahead}, "
            f"subgrid=({subgrid_width},{subgrid_height},{subgrid_increment}), "
            f"d={d}, m_traps={m_traps}, n_traps={n_traps}: {repr(e)}"
        )
        results = {}
        results["ElapsedTime"] = np.nan
        results["Operations"] = np.nan
        results["MeanConcurrency"] = np.nan
        results["QubitOperations"] = np.nan
        results["LogicalErrorRates"] = [np.nan for _ in gate_improvements]
        results["PhysicalZErrorRates"] = [np.nan for _ in gate_improvements]
        results["PhysicalXErrorRates"] = [np.nan for _ in gate_improvements]
        # exec_time NaN, but keep compilation time to see failures too
        return np.nan, comp_time, results, np.nan

# ---------- helper for outer pool ----------

def _run_single_config_entry(
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
    go_back_threshold: float,
    base_pmax_in: int,
    sat_workers_per_config: int,
    time_budget_s: float | None,
    gate_improvements: Sequence[float] = [1.0],
    num_shots: int = 100_000
):
    """
    Worker wrapper for run_single_config:
    - sets WISE_SAT_WORKERS so qccd_operations' SAT pool doesn't grab all cores
    - calls run_single_config and returns a structured dict.
    """

    configure_wise_logging(d=d, k =trap_capacity)
    if sat_workers_per_config is not None and sat_workers_per_config > 0:
        os.environ["WISE_SAT_WORKERS"] = str(sat_workers_per_config)

    if time_budget_s is not None and time_budget_s > 0:
        cap = max(time_budget_s / 2.0, 1.0)

        def _set_cap(var: str, value: float) -> None:
            current = os.environ.get(var)
            if current is not None:
                try:
                    current_val = float(current)
                    if current_val > 0:
                        value = min(value, current_val)
                except ValueError:
                    pass
            os.environ[var] = f"{value}"

        _set_cap("WISE_MAX_SAT_TIME", cap)
        _set_cap("WISE_MAX_RC2_TIME", cap)

    exec_time, comp_time, results, reconfigTime = run_single_config(
        lookahead=lookahead,
        subgrid_width=subgrid_width,
        subgrid_height=subgrid_height,
        subgrid_increment=subgrid_increment,
        d=d,
        m_traps=m_traps,
        n_traps=n_traps,
        trap_capacity=trap_capacity,
        barrier_threshold=barrier_threshold,
        go_back_threshold=go_back_threshold,
        base_pmax_in=base_pmax_in,
        gate_improvements=gate_improvements,
        num_shots=num_shots
    )

    dfrow = {
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
        "reconfigTime": reconfigTime,
    }

    for k, v in results.items():
        dfrow[k]=v
    return dfrow


def search_configs_best_exec_time(
    configs: list[tuple[int, int, int, int]],
    *,
    d: int,
    m_traps: int,
    n_traps: int,
    trap_capacity: int,
    barrier_threshold: float = np.inf,
    go_back_threshold: float = 0.0,
    base_pmax_in: int = None,
    time_budget_s: float | None = None,
    sat_workers_per_config: int = 4,
    max_total_workers: int | None = None,
    verbose: bool = False,
    leaderboard_top_k: int = 5,
    gate_improvements: Sequence[float] = [1.0],
    num_shots: int = 100_000,
):
    if not configs:
        return None, []

    total_cpus = mp.cpu_count() or 1

    # Simple heuristic: don't oversubscribe too much
    if max_total_workers is None:
        if sat_workers_per_config and sat_workers_per_config > 0:
            max_total_workers = max(1, total_cpus // sat_workers_per_config)
        else:
            max_total_workers = total_cpus

    max_total_workers = max(1, min(max_total_workers, len(configs)))

    start = time.time()
    all_results: list[dict] = []
    best_result: dict | None = None

    def _is_better(a: dict, b: dict | None) -> bool:
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
        ca, cb = a["comp_time"], b["comp_time"]
        return ca < cb

    def _print_leaderboard():
        if not verbose:
            return
        if not all_results:
            return
        valid = [r for r in all_results if not np.isnan(r["exec_time"])]
        if not valid:
            return
        valid_sorted = sorted(valid, key=lambda r: (r["exec_time"], r["comp_time"]))
        top = valid_sorted[: max(1, leaderboard_top_k)]
        print("[WISE-SEARCH] Current top configs (by exec_time):")
        for r in top:
            la = r["lookahead"]
            w = r["subgrid_width"]
            h = r["subgrid_height"]
            inc = r["subgrid_increment"]
            et = r["exec_time"]
            ct = r["comp_time"]
            print(f"  la={la}, w={w}, h={h}, inc={inc}: exec={et:.3f}, comp={ct:.1f}s")
        print(flush=True)

    extra_kw = dict(
        d=d,
        m_traps=m_traps,
        n_traps=n_traps,
        trap_capacity=trap_capacity,
        barrier_threshold=barrier_threshold,
        go_back_threshold=go_back_threshold,
        base_pmax_in=base_pmax_in,
        sat_workers_per_config=sat_workers_per_config,
        time_budget_s=time_budget_s,
        num_shots=num_shots,
        gate_improvements=gate_improvements,
    )

    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=max_total_workers, mp_context=ctx)
    future_to_cfg: dict = {}

    try:
        # Submit all configs up-front; outer pool caps concurrency.
        for cfg in configs:
            fut = executor.submit(
                _run_single_config_entry,
                lookahead=cfg[0],
                subgrid_width=cfg[1],
                subgrid_height=cfg[2],
                subgrid_increment=cfg[3],
                **extra_kw,
            )
            future_to_cfg[fut] = cfg

        pending = set(future_to_cfg.keys())
        timed_out = False

        # Polling loop instead of blocking as_completed
        while pending:
            # Budget check *before* blocking
            if time_budget_s is not None and (time.time() - start) >= time_budget_s:
                timed_out = True
                if verbose:
                    print(
                        "[WISE-SEARCH] Global time budget reached; "
                        "stopping collection of further results.",
                        flush=True,
                    )
                break

            done, pending = concurrent.futures.wait(
                pending,
                timeout=1.0,  # poll every second
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            if not done:
                # Nothing finished this tick; go back around and re-check budget.
                continue

            for fut in done:
                cfg = future_to_cfg[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    if verbose:
                        print(
                            f"[WISE-SEARCH] run_single_config crashed for cfg={cfg}: {e}",
                            flush=True,
                        )
                    res = {
                        "lookahead": cfg[0],
                        "subgrid_width": cfg[1],
                        "subgrid_height": cfg[2],
                        "subgrid_increment": cfg[3],
                        "d": d,
                        "m_traps": m_traps,
                        "n_traps": n_traps,
                        "trap_capacity": trap_capacity,
                        "exec_time": float("nan"),
                        "comp_time": 0.0,
                        "reconfigTime": float("nan"),
                        "error": repr(e),
                    }
                    res["ElapsedTime"] = np.nan
                    res["Operations"] = np.nan
                    res["MeanConcurrency"] = np.nan
                    res["QubitOperations"] = np.nan
                    res["LogicalErrorRates"] = [np.nan for _ in gate_improvements]
                    res["PhysicalZErrorRates"] = [np.nan for _ in gate_improvements]
                    res["PhysicalXErrorRates"] = [np.nan for _ in gate_improvements]

                all_results.append(res)
                if _is_better(res, best_result):
                    best_result = res
                    if verbose:
                        print("[WISE-SEARCH] New best config found.", flush=True)
                        _print_leaderboard()

    except KeyboardInterrupt:
        if verbose:
            print("[WISE-SEARCH] KeyboardInterrupt received; cancelling remaining configs.", flush=True)
        timed_out = True
        # fall through to finally for cleanup
    finally:
        # Cancel any futures that have not finished yet.
        for f in future_to_cfg:
            if not f.done():
                f.cancel()

        # Shut down the executor (do not wait for tasks that are still running).
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            # Even if shutdown fails, still attempt process cleanup below.
            pass

        # If we've hit the global time budget or a KeyboardInterrupt, be
        # aggressive about tearing down worker processes so no search
        # workers linger and hold on to nested SAT pools.
        try:
            procs = getattr(executor, "_processes", None)
            if procs:
                for pid, proc in list(procs.items()):
                    if proc.is_alive():
                        if verbose:
                            print(f"[WISE-SEARCH] Force-terminating search worker PID {pid}", flush=True)
                        try:
                            proc.terminate()
                        except Exception:
                            pass

                # Give workers a short grace period to exit cleanly.
                for pid, proc in list(procs.items()):
                    if proc.is_alive():
                        try:
                            proc.join(timeout=1.0)
                        except Exception:
                            pass

                # Escalate to kill any stubborn workers that refused to exit.
                for pid, proc in list(procs.items()):
                    if proc.is_alive():
                        if verbose:
                            print(f"[WISE-SEARCH] Force-killing stubborn search worker PID {pid}", flush=True)
                        try:
                            proc.kill()
                        except Exception:
                            pass
        except Exception:
            # Best-effort cleanup; ignore errors during forced teardown.
            pass

    if verbose:
        elapsed = time.time() - start
        print(
            f"[WISE-SEARCH] Finished search in {elapsed:.1f}s; "
            f"{len(all_results)} configs completed.",
            flush=True,
        )
        _print_leaderboard()

    return best_result, all_results