#!/usr/bin/env python3
"""Instrumented test: trace where routing gets stuck."""
import sys, os, signal, time, logging, faulthandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Enable faulthandler for SIGABRT/SIGSEGV stack dumps
faulthandler.enable()
# Dump all threads on SIGUSR1
faulthandler.register(signal.SIGUSR1)

# Hard 5-minute timeout
signal.signal(signal.SIGALRM, lambda *_: (
    print("ALARM: 5-minute timeout reached!", flush=True),
    faulthandler.dump_traceback(),
    os._exit(1),
))
signal.alarm(300)

logging.basicConfig(level=logging.WARNING)

import numpy as np

print("[1] Importing modules...", flush=True)
from qectostim.codes.surface import RotatedSurfaceCode
from qectostim.gadgets.transversal_cnot import TransversalCNOTGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
print("[1] Imports done.", flush=True)

code = RotatedSurfaceCode(distance=2)
gadget = TransversalCNOTGadget()
d = code.distance
trap_capacity = 2

print("[2] Building FaultTolerantGadgetExperiment...", flush=True)
ft_experiment = FaultTolerantGadgetExperiment(
    codes=[code],
    gadget=gadget,
    noise_model=None,
    num_rounds_before=2,
    num_rounds_after=2,
)
print("[2] Experiment built.", flush=True)

print("[3] Generating ideal circuit via to_stim()...", flush=True)
ideal = ft_experiment.to_stim()
print(f"[3] Ideal circuit: {ideal.num_qubits} qubits, "
      f"{ideal.num_ticks} ticks.", flush=True)

print("[4] Extracting QEC metadata...", flush=True)
qec_meta = ft_experiment.qec_metadata
qubit_allocation = ft_experiment._unified_allocation
print(f"[4] Metadata: {len(qec_meta.block_allocations)} blocks, "
      f"{len(qec_meta.phases)} phases, "
      f"{qec_meta.total_qubits} total qubits.", flush=True)

print("[5] Computing grid dimensions...", flush=True)
num_qubits = ideal.num_qubits
nqubitsNeeded = 4 * (np.ceil(num_qubits / 3))
n_traps = int(np.ceil(np.sqrt(nqubitsNeeded)))
m_traps = int(np.ceil(n_traps / trap_capacity))
print(f"[5] Grid: n_traps={n_traps}, m_traps={m_traps}", flush=True)

print("[6] Calling route_full_experiment...", flush=True)
t0 = time.perf_counter()
from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
    route_full_experiment,
)

try:
    full_result = route_full_experiment(
        qec_meta=qec_meta,
        gadget=gadget,
        qubit_allocation=qubit_allocation,
        k=trap_capacity,
        subgridsize=(4, 3, 0),
        base_pmax_in=1,
        lookahead=2,
        max_inner_workers=1,
        stop_event=None,
        progress_callback=None,
    )
    wall = time.perf_counter() - t0
    print(f"[6] route_full_experiment completed in {wall:.1f}s", flush=True)
    print(f"    phases: {full_result.total_phases}, "
          f"cached: {full_result.cached_phases}", flush=True)
    print("TEST PASSED", flush=True)
except Exception as e:
    wall = time.perf_counter() - t0
    import traceback
    print(f"[6] FAILED after {wall:.1f}s: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
