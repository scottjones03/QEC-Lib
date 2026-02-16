import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from typing import Sequence, Dict, Any, List, Tuple
from .qccd_nodes import QCCDWiseArch
from .architectures import WISEArchitecture
from .trapped_ion_compiler import TrappedIonCompiler
from .experiment import TrappedIonExperiment, NDE_JZ, NDE_LZ, NSE_Z
from .noise import TrappedIonNoiseModel
from qectostim.decoders.pymatching_decoder import PyMatchingDecoder
import time
import os
import signal
import time
import math
import atexit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Global registry for tracking spawned process PIDs for cleanup on exit
_ACTIVE_WORKER_PIDS: set[int] = set()
_ACTIVE_MANAGERS: set = set()


def _cleanup_on_exit():
    """
    Atexit handler to clean up any spawned worker processes and managers.
    This is a safety net if the main finally block doesn't complete.
    """
    import subprocess
    
    def _get_child_pids(parent_pid: int) -> set:
        children = set()
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(parent_pid)],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            children.add(int(line.strip()))
                        except ValueError:
                            pass
        except Exception:
            pass
        return children
    
    # Collect all PIDs including children and grandchildren
    all_pids = set(_ACTIVE_WORKER_PIDS)
    for pid in list(_ACTIVE_WORKER_PIDS):
        children = _get_child_pids(pid)
        all_pids.update(children)
        for child_pid in children:
            all_pids.update(_get_child_pids(child_pid))
    
    # SIGTERM then SIGKILL
    for pid in all_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    
    if all_pids:
        time.sleep(0.3)
    
    for pid in all_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    
    # Shutdown managers
    for mgr in _ACTIVE_MANAGERS:
        try:
            mgr.shutdown()
        except Exception:
            pass
    
    _ACTIVE_WORKER_PIDS.clear()
    _ACTIVE_MANAGERS.clear()


atexit.register(_cleanup_on_exit)
import numpy as np  # needed for np.nan checks
import stim
import concurrent

import logging
import os
from ..compiler.routing_config import WISERoutingConfig, WISESolverParams, make_queue_progress_callback
from .progress_table import (
    SharedProgressState,
    SharedProgressSlot,
    ProgressTableWidget,
    ProgressPoller,
    make_shared_progress_callback,
    STATUS_IDLE,
    STATUS_RUNNING,
    STATUS_DONE,
    STATUS_ERROR,
    _in_notebook,
)


# ---------------------------------------------------------------------------
# Progress queue helper for parallel execution
# ---------------------------------------------------------------------------

def _drain_progress_queue(queue, sat_bar) -> None:
    """Drain progress updates from queue and update the SAT bar.
    
    Workers send (config_id, stage, current, total, message) tuples.
    We aggregate across all workers and show the combined progress.
    """
    try:
        from queue import Empty
    except ImportError:
        return
    
    # Track latest state per config for aggregation
    if not hasattr(_drain_progress_queue, "_config_states"):
        _drain_progress_queue._config_states = {}
    
    states = _drain_progress_queue._config_states
    
    # Drain all pending messages (non-blocking)
    while True:
        try:
            msg = queue.get_nowait()
            if msg is None:
                continue
            config_id, stage, current, total, message = msg
            states[config_id] = (current, total, message)
        except Empty:
            break
        except Exception:
            break
    
    # Aggregate: sum current across configs, total is max seen
    if states:
        total_current = sum(s[0] for s in states.values())
        max_total = max(s[1] for s in states.values()) if states else 1
        # Show count of active configs in postfix
        active_count = len(states)
        latest_msg = ""
        for s in states.values():
            if s[2]:
                latest_msg = s[2][:30]
                break
        
        sat_bar.total = max(max_total * active_count, 1)
        sat_bar.n = min(total_current, sat_bar.total)
        sat_bar.set_postfix_str(f"{active_count} active" + (f" | {latest_msg}" if latest_msg else ""))
        sat_bar.refresh()


# ---------------------------------------------------------------------------
# FIX D: Zombie reaping utility
# ---------------------------------------------------------------------------

def _reap_zombies() -> int:
    """Reap zombie child processes left by killed workers.

    On macOS with spawn context, exited children remain as zombies until
    the parent calls waitpid().  The notebook kernel is long-lived, so
    zombies accumulate across cell runs.  This function calls waitpid()
    in a non-blocking loop to clean them up.

    Returns the number of zombies reaped.
    """
    reaped = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
            reaped += 1
        except ChildProcessError:
            # No child processes exist
            break
        except OSError:
            break
    return reaped

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


def _route_and_simulate(
    ideal: stim.Circuit,
    m_traps: int,
    n_traps: int,
    trap_capacity: int,
    nqubitsNeeded: float,
    lookahead: int,
    subgridsize: Tuple[int, int, int],
    base_pmax_in: int,
    barrier_threshold: float,
    gate_improvements: Sequence[float],
    num_shots: int,
    routing_config,
    stop_event,
    toMoveOps=None,
    show_progress: bool = True,
    max_inner_workers: int | None = None,
    progress_queue=None,
    config_id: int = 0,
    progress_slot: SharedProgressSlot | None = None,
) -> Tuple[float, Dict[str, Any], float, float]:
    """Shared routing and simulation logic for run_single_config and run_single_config_cnot.

    Uses the TrappedIonExperiment / TrappedIonCompiler framework:
      compiler.compile(ideal) → experiment.apply_hardware_noise() → decode.

    Parameters
    ----------
    ideal : stim.Circuit
        The ideal circuit to route.
    m_traps, n_traps : int
        Grid dimensions.
    trap_capacity : int
        Ions per trap.
    nqubitsNeeded : float
        Total qubits for electrode calculations.
    lookahead : int
        SAT solver lookahead.
    subgridsize : tuple
        (width, height, increment) for patch decomposition.
    base_pmax_in : int
        Starting P_max for binary search.
    barrier_threshold : float
        Barrier logic threshold (unused — kept for signature compatibility).
    gate_improvements : Sequence[float]
        Error scaling factors for simulation.
    num_shots : int
        Monte Carlo shots for error estimation.
    routing_config : WISERoutingConfig or None
        Routing configuration object.
    stop_event : Event or None
        Cancellation signal.
    toMoveOps : Sequence or None
        Optional toMove operations (for CNOT variant).
    show_progress : bool
        Display a tqdm progress bar during SAT routing (default: True).
    progress_queue : multiprocessing.Queue or None
        If provided, send SAT progress updates to this queue instead
        of creating a local tqdm bar. Used for parallel execution.
    config_id : int
        Identifier for this config (used with progress_queue).

    Returns
    -------
    tuple of (exec_time, results, reconfigTime, comp_time)
    """
    t0 = time.perf_counter()

    # 1. Architecture + compiler setup
    # When using progress_slot or progress_queue, we inject our own callback
    # into routing_config, so disable the compiler's default tqdm bar.
    use_shared_progress = progress_slot is not None
    use_local_progress = show_progress and not use_shared_progress and progress_queue is None
    
    wiseArch = QCCDWiseArch(m=m_traps, n=n_traps, k=trap_capacity)
    arch = WISEArchitecture(wise_config=wiseArch)
    compiler = TrappedIonCompiler(arch, is_wise=True, wise_config=wiseArch,
                                  show_progress=use_local_progress)

    # 2. If using shared memory progress, inject our callback into routing_config
    _progress_close = None
    if use_shared_progress and routing_config is not None:
        progress_slot.status = STATUS_RUNNING
        progress_slot.config_id = config_id
        _progress_cb, _progress_close = make_shared_progress_callback(progress_slot)
        routing_config.progress_callback = _progress_cb

    # 3. Set routing params on compiler (forwarded by route() to ionRoutingWISEArch)
    compiler.routing_kwargs = dict(
        lookahead=lookahead,
        subgridsize=subgridsize,
        base_pmax_in=base_pmax_in,
        routing_config=routing_config,
        stop_event=stop_event,
        max_inner_workers=max_inner_workers,
    )
    if toMoveOps is not None:
        compiler.routing_kwargs["toMoveOps"] = toMoveOps

    # 4. Compile: decompose → map → route → schedule (single call)
    try:
        compiled = compiler.compile(ideal)
    finally:
        # Mark slot as done
        if _progress_close is not None:
            _progress_close()


    t1 = time.perf_counter()
    comp_time = t1 - t0

    # 4. Extract metrics from CompiledCircuit
    scheduled = compiled.scheduled
    exec_time = scheduled.total_duration
    reconfig_time = scheduled.routed_circuit.metadata.get("reconfig_time", 0.0)
    all_operations = scheduled.metadata.get("all_operations", [])
    parallel_ops_map = scheduled.metadata.get("parallel_ops_map", {})
    native_ops = scheduled.routed_circuit.mapped_circuit.native_circuit.operations

    # 5. Noise + decode loop for each gate_improvement
    logicalErrors = []
    physicalZErrors = []
    physicalXErrors = []

    for gate_improvement in gate_improvements:
        noise = TrappedIonNoiseModel(error_scaling=gate_improvement)
        experiment = TrappedIonExperiment(
            code=None,
            architecture=arch,
            compiler=compiler,
            hardware_noise=noise,
        )
        experiment._compiled = compiled

        noisy_circuit = experiment.apply_hardware_noise()

        # Build decoder from the noisy circuit's detector error model
        dem = noisy_circuit.detector_error_model()
        decoder = PyMatchingDecoder(dem=dem)

        # Sample + decode
        sampler = noisy_circuit.compile_detector_sampler()
        detection_events, obs_flips = sampler.sample(
            num_shots, separate_observables=True
        )
        predictions = decoder.decode_batch(detection_events)
        num_errors = int(np.sum(np.any(predictions != obs_flips, axis=1)))

        logicalErrors.append(num_errors / num_shots)
        physicalXErrors.append(getattr(experiment, '_last_mean_phys_x', 0.0))
        physicalZErrors.append(getattr(experiment, '_last_mean_phys_z', 0.0))

    # 6. Build results dict
    results = {}
    results["ElapsedTime"] = exec_time
    results["Operations"] = len(all_operations)
    results["MeanConcurrency"] = (
        np.mean([len(op.operations) for op in parallel_ops_map.values()])
        if parallel_ops_map else 0.0
    )
    results["QubitOperations"] = len(native_ops)
    results["LogicalErrorRates"] = logicalErrors
    results["PhysicalZErrorRates"] = physicalZErrors
    results["PhysicalXErrorRates"] = physicalXErrors

    # 7. Electrode / DAC estimates
    Njz = np.ceil(nqubitsNeeded / trap_capacity)
    Nlz = nqubitsNeeded - Njz
    Nde = NDE_LZ * Nlz + NDE_JZ * Njz
    Nse = NSE_Z * (Njz + Nlz)
    results["DACs"] = int(min(100, Nde) + np.ceil(Nse / 100))
    results["Electrodes"] = int(Nde + Nse)

    return exec_time, results, reconfig_time, comp_time


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
    num_shots: int = 100_000,
    routing_config: "WISERoutingConfig | None" = None,
    stop_event=None,
    show_progress: bool = True,
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
    routing_config : WISERoutingConfig or None, optional
        Pre-built routing configuration.  When provided, SAT solver
        parameters are taken from this instead of environment variables.
    stop_event : multiprocessing.Event or None, optional
        Shared event; when set, workers abandon the current search.
    show_progress : bool, optional
        Display a tqdm progress bar during SAT routing (default: True).

    Returns
    -------
    (exec_time, comp_time) : (float, float)
        exec_time : circuit execution time (max parallel timestep).
        comp_time : wall-clock time for compilation for this config.
    """
    t0 = time.perf_counter()
    # Generate circuit with the chosen code distance
    ideal = stim.Circuit.from_file(f"logical_cnot_d={d//2}.stim")
    num_qubits = ideal.num_qubits

    nqubitsNeeded = 4 * (np.ceil(num_qubits / 3))
    n_traps = int(np.ceil(np.sqrt(nqubitsNeeded)))
    m_traps = int(np.ceil(n_traps / trap_capacity))

    # Extract toMoveOps from circuit metadata (CNOT-specific)
    # First we need to decompose to get metadata
    wiseArch_temp = QCCDWiseArch(m=m_traps, n=n_traps, k=trap_capacity)
    arch_temp = WISEArchitecture(wise_config=wiseArch_temp)
    compiler_temp = TrappedIonCompiler(arch_temp, is_wise=True, wise_config=wiseArch_temp,
                                       show_progress=False)
    native_temp = compiler_temp.decompose_to_native(ideal)
    toMoveOps = native_temp.metadata.get("toMoveOps", [])

    try:
        exec_time, results, reconfigTime, comp_time = _route_and_simulate(
            ideal=ideal,
            m_traps=m_traps,
            n_traps=n_traps,
            trap_capacity=trap_capacity,
            nqubitsNeeded=nqubitsNeeded,
            lookahead=lookahead,
            subgridsize=(subgrid_width, subgrid_height, subgrid_increment),
            base_pmax_in=base_pmax_in,
            barrier_threshold=barrier_threshold,
            gate_improvements=gate_improvements,
            num_shots=num_shots,
            routing_config=routing_config,
            stop_event=stop_event,
            toMoveOps=toMoveOps,
            show_progress=show_progress,
        )
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
    num_shots: int = 100_000,
    routing_config: "WISERoutingConfig | None" = None,
    stop_event=None,
    show_progress: bool = True,
    max_inner_workers: int | None = None,
    progress_queue=None,
    config_id: int = 0,
    progress_slot: SharedProgressSlot | None = None,
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
        Surface code distance (used to generate a surface code circuit).
    m_traps : int, optional
        Number of trap columns in the WISE architecture (QCCDWiseArch.m).
    n_traps : int, optional
        Number of trap rows in the WISE architecture (QCCDWiseArch.n).
    routing_config : WISERoutingConfig or None, optional
        Pre-built routing configuration.  When provided, SAT solver
        parameters are taken from this instead of environment variables.
    stop_event : multiprocessing.Event or None, optional
        Shared event; when set, workers abandon the current search.
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
        ideal = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=1,
            distance=d,
        )
        nqubitsNeeded = m_traps * n_traps * trap_capacity

        exec_time, results, reconfigTime, comp_time = _route_and_simulate(
            ideal=ideal,
            m_traps=m_traps,
            n_traps=n_traps,
            trap_capacity=trap_capacity,
            nqubitsNeeded=nqubitsNeeded,
            lookahead=lookahead,
            subgridsize=(subgrid_width, subgrid_height, subgrid_increment),
            base_pmax_in=base_pmax_in,
            barrier_threshold=barrier_threshold,
            gate_improvements=gate_improvements,
            num_shots=num_shots,
            routing_config=routing_config,
            stop_event=stop_event,
            toMoveOps=None,
            show_progress=show_progress,
            max_inner_workers=max_inner_workers,
            progress_queue=progress_queue,
            config_id=config_id,
            progress_slot=progress_slot,
        )
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
    stop_event=None,
    gate_improvements: Sequence[float] = [1.0],
    num_shots: int = 100_000,
    show_progress: bool = True,
    max_inner_workers: int | None = None,
    progress_queue=None,
    config_id: int = 0,
    progress_slot: SharedProgressSlot | None = None,
):
    """
    Worker wrapper for run_single_config:
    - builds WISESolverParams / WISERoutingConfig instead of env-var hacks
    - calls run_single_config and returns a structured dict.
    """

    configure_wise_logging(d=d, k=trap_capacity)

    # Build solver params via the dataclass instead of env vars.
    # When a time budget is provided, compute per-subprocess timeout
    # from problem dimensions and cap it; otherwise let the downstream
    # solver auto-compute from actual patch dimensions.
    if time_budget_s is not None and time_budget_s > 0:
        from ..compiler.qccd_SAT_WISE_odd_even_sorter import _estimate_sat_timeout

        num_ions_est = n_traps * m_traps * trap_capacity
        R_est = max(base_pmax_in or 1, 1)
        sat_time = _estimate_sat_timeout(
            n_traps, m_traps, num_ions_est, R_est, R_est,
        )
        rc2_time = sat_time
        cap = max(time_budget_s / 2.0, 1.0)
        sat_time = min(sat_time, cap)
        rc2_time = min(rc2_time, cap)
    else:
        sat_time = None
        rc2_time = None
    workers = (
        sat_workers_per_config
        if sat_workers_per_config and sat_workers_per_config > 0
        else 4
    )

    solver_params = WISESolverParams(
        max_sat_time=sat_time,
        max_rc2_time=rc2_time,
        base_pmax_in=base_pmax_in,
        sat_workers=workers,
    )

    routing_config = WISERoutingConfig(
        timeout_seconds=sat_time,
        subgridsize=(subgrid_width, subgrid_height, subgrid_increment),
        lookahead=lookahead,
        sat_workers=workers,
        base_pmax_in=base_pmax_in,
        solver_params=solver_params,
    )

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
        num_shots=num_shots,
        routing_config=routing_config,
        stop_event=stop_event,
        show_progress=show_progress,
        max_inner_workers=max_inner_workers,
        progress_queue=progress_queue,
        config_id=config_id,
        progress_slot=progress_slot,
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
        dfrow[k] = v
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
    show_progress: bool = True,
):
    if not configs:
        return None, []

    # FIX D: Reap any zombie children left from previous cell runs
    # before starting new processes.
    _reaped = _reap_zombies()
    if _reaped > 0 and verbose:
        print(f"[WISE-SEARCH] Reaped {_reaped} zombie processes from previous runs.", flush=True)

    total_cpus = mp.cpu_count() or 1

    # FIX C: Global process budget — prevent oversubscription.
    # Each outer worker may spawn an inner SAT pool (Fix A relaxed)
    # with up to inner_worker_budget processes.  Peak concurrent
    # processes ≈ outer_workers × (1 + inner_workers).
    # Cap outer_workers so peak stays ≤ total_cpus.
    if max_total_workers is None:
        # Each outer worker needs ~2 processes (itself + SAT child)
        max_total_workers = max(1, total_cpus // 2)

    max_total_workers = max(1, min(max_total_workers, len(configs)))

    # Hard cap: never exceed half the CPU count for outer workers,
    # so SAT subprocesses still have room.
    max_total_workers = min(max_total_workers, max(1, total_cpus // 2))

    if verbose:
        peak_procs = max_total_workers * 2 + 1  # workers + SAT children + kernel
        print(
            f"[WISE-SEARCH] Process budget: {max_total_workers} outer workers, "
            f"peak ~{peak_procs} processes (CPUs={total_cpus})",
            flush=True,
        )

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

    # CPU budget available for each outer worker's inner SAT pool.
    # With Fix A (relaxed), each worker's inner pool caps at this many
    # processes, so total ≈ outer_workers × inner_workers ≤ total_cpus.
    inner_worker_budget = max(1, total_cpus // max(1, max_total_workers) - 1)

    if show_progress or verbose:
        print(
            f"[WISE-SEARCH] Parallelism: {max_total_workers} outer workers × "
            f"{inner_worker_budget} inner SAT workers "
            f"(CPUs={total_cpus}, configs={len(configs)})",
            flush=True,
        )

    # When in a notebook, use the ipywidgets progress table to show per-worker
    # progress (even with just 1 outer worker). In terminal mode, fall back
    # to tqdm bars. The widget tracks inner SAT progress via shared memory.
    use_progress_widget = show_progress and _in_notebook()
    worker_show_progress = show_progress and not use_progress_widget

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
        show_progress=worker_show_progress,
        max_inner_workers=inner_worker_budget,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Create Manager FIRST - needed for both stop_event and progress state
    # ─────────────────────────────────────────────────────────────────────────
    ctx = mp.get_context("spawn")
    # Use a Manager to create proxy objects that can be pickled across
    # spawn-context workers.  A bare ctx.Event() uses POSIX semaphores
    # that are NOT picklable on macOS spawn, causing RuntimeError.
    # One Manager process is lightweight; the earlier problem was 4
    # nested Managers (now eliminated by Fix A).
    manager = ctx.Manager()
    _ACTIVE_MANAGERS.add(manager)
    stop_event = manager.Event()
    extra_kw["stop_event"] = stop_event

    # ─────────────────────────────────────────────────────────────────────────
    # Progress display: use ipywidgets table in notebooks, tqdm in terminal
    # ─────────────────────────────────────────────────────────────────────────
    _outer_bar = None
    _outer_bar_close = lambda: None
    _progress_state = None
    _progress_widget = None
    _progress_poller = None
    
    # Build config labels for the widget
    config_labels = [f"la={c[0]}, ({c[1]},{c[2]},{c[3]})" for c in configs]
    
    if use_progress_widget:
        # Create shared memory state for inter-process progress using the Manager
        _progress_state = SharedProgressState.create(max_total_workers, manager)
        
        # Create and display the widget
        _progress_widget = ProgressTableWidget(
            shared_state=_progress_state,
            config_labels=config_labels,
            outer_desc=f"WISE configs (d={d}, k={trap_capacity})",
        )
        
        # Display the widget
        try:
            from IPython.display import display
            display(_progress_widget.container)
        except Exception:
            pass
        
        # Start background poller to refresh widget
        _progress_poller = ProgressPoller(_progress_widget, poll_interval=0.15)
        _progress_poller.start()
        
        def _outer_bar_close():
            if _progress_poller is not None:
                _progress_poller.stop()
            if _progress_widget is not None:
                _progress_widget.close()
    
    elif show_progress and len(configs) > 1:
        # Fallback to tqdm for terminal mode
        try:
            from tqdm.auto import tqdm as _tqdm
            _outer_bar = _tqdm(
                total=len(configs),
                desc=f"WISE configs (d={d}, k={trap_capacity})",
                unit="cfg",
                position=0,
            )
            def _outer_bar_close():
                if _outer_bar is not None:
                    _outer_bar.close()
        except ImportError:
            pass

    executor = ProcessPoolExecutor(max_workers=max_total_workers, mp_context=ctx)
    future_to_cfg: dict = {}
    future_to_slot: dict = {}  # Track which slot each future uses
    available_slots: list = list(range(max_total_workers)) if _progress_state else []
    completed_count = 0

    try:
        # Submit all configs up-front; outer pool caps concurrency.
        # Track slot assignments for progress tracking.
        for idx, cfg in enumerate(configs):
            cfg_extra = dict(extra_kw)
            cfg_extra["config_id"] = idx
            
            # Assign a progress slot if we're using the widget
            assigned_slot = None
            if _progress_state and available_slots:
                slot_idx = available_slots.pop(0)
                assigned_slot = _progress_state.slots[slot_idx]
                assigned_slot.reset(config_id=idx)
                cfg_extra["progress_slot"] = assigned_slot
            
            fut = executor.submit(
                _run_single_config_entry,
                lookahead=cfg[0],
                subgrid_width=cfg[1],
                subgrid_height=cfg[2],
                subgrid_increment=cfg[3],
                **cfg_extra,
            )
            future_to_cfg[fut] = cfg
            if assigned_slot is not None:
                future_to_slot[fut] = (slot_idx, assigned_slot)

        pending = set(future_to_cfg.keys())
        timed_out = False

        # Polling loop instead of blocking as_completed
        while pending:
            # Register any newly spawned worker PIDs for atexit cleanup
            try:
                procs = getattr(executor, "_processes", None)
                if procs:
                    _ACTIVE_WORKER_PIDS.update(procs.keys())
            except Exception:
                pass

            # Budget check *before* blocking
            if time_budget_s is not None and (time.time() - start) >= time_budget_s:
                timed_out = True
                stop_event.set()
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
                
                # Release the slot back to the pool if we're using progress widget
                if fut in future_to_slot:
                    slot_idx, slot = future_to_slot.pop(fut)
                    available_slots.append(slot_idx)
                
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
                completed_count += 1
                
                if _is_better(res, best_result):
                    best_result = res
                    if verbose:
                        print("[WISE-SEARCH] New best config found.", flush=True)
                        _print_leaderboard()

                # Update progress display
                if _progress_widget is not None:
                    _progress_widget.update_completed(completed_count)
                elif _outer_bar is not None:
                    et = res.get("exec_time", float("nan"))
                    label = f"exec={et:.4f}" if not np.isnan(et) else "failed"
                    _outer_bar.set_postfix_str(label)
                    _outer_bar.update(1)

    except KeyboardInterrupt:
        if verbose:
            print("[WISE-SEARCH] KeyboardInterrupt received; cancelling remaining configs.", flush=True)
        timed_out = True
        stop_event.set()
        # fall through to finally for cleanup
    finally:
        # Cancel any futures that have not finished yet.
        for f in future_to_cfg:
            if not f.done():
                f.cancel()

        # Helper to get child PIDs of a process
        def _get_child_pids(parent_pid: int) -> set:
            children = set()
            try:
                import subprocess
                result = subprocess.run(
                    ["pgrep", "-P", str(parent_pid)],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            try:
                                children.add(int(line.strip()))
                            except ValueError:
                                pass
            except Exception:
                pass
            return children

        # Collect all PIDs to kill (workers + their children + grandchildren)
        all_pids_to_kill = set()
        try:
            procs = getattr(executor, "_processes", None)
            if procs:
                for pid in list(procs.keys()):
                    all_pids_to_kill.add(pid)
                    # Get children (nested SAT pool workers) 
                    children = _get_child_pids(pid)
                    all_pids_to_kill.update(children)
                    # Get grandchildren (actual SAT solver processes)
                    for child_pid in children:
                        all_pids_to_kill.update(_get_child_pids(child_pid))
        except Exception:
            pass

        # First pass: SIGTERM all processes
        for pid in all_pids_to_kill:
            try:
                os.kill(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass

        # Brief grace period
        time.sleep(0.5)

        # Second pass: SIGKILL any survivors
        for pid in all_pids_to_kill:
            try:
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

        if verbose and all_pids_to_kill:
            print(f"[WISE-SEARCH] Killed {len(all_pids_to_kill)} processes (workers + nested children)", flush=True)

        # Shut down the executor (do not wait for tasks that are still running).
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            # Even if shutdown fails, still attempt process cleanup below.
            pass

        # Shut down the Manager process (stop_event proxy server)
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass
            _ACTIVE_MANAGERS.discard(manager)

        # FIX D (outer): Reap zombie children left behind by killed
        # processes.  On macOS with spawn context, exited children
        # remain as zombies until waitpid() is called.
        _reap_zombies()

        # Clear the global registry now that cleanup is done
        _ACTIVE_WORKER_PIDS.clear()

        # Close progress bar
        _outer_bar_close()

    if verbose:
        elapsed = time.time() - start
        print(
            f"[WISE-SEARCH] Finished search in {elapsed:.1f}s; "
            f"{len(all_results)} configs completed.",
            flush=True,
        )
        _print_leaderboard()

    return best_result, all_results