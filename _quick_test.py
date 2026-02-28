#!/usr/bin/env python3
"""Quick test: verify routing completes with spectator BT fix."""
import sys, os, signal, time, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.stdout.reconfigure(line_buffering=True)

# Hard 5-minute timeout
signal.alarm(300)

logging.basicConfig(level=logging.WARNING)

from qectostim.codes.surface import RotatedSurfaceCode
from qectostim.gadgets.transversal_cnot import TransversalCNOTGadget
from qectostim.experiments.hardware_simulation.trapped_ion.utils.best_effort_compilation_WISE import (
    run_single_gadget_config,
)

code = RotatedSurfaceCode(distance=2)
gadget = TransversalCNOTGadget()

print("Starting d=2 TransversalCNOT routing...", flush=True)
t0 = time.perf_counter()

try:
    exec_time, comp_time, results, reconfig_time = run_single_gadget_config(
        gadget=gadget,
        code=code,
        lookahead=2,
        subgrid_width=4,
        subgrid_height=3,
        subgrid_increment=0,
        trap_capacity=2,
        base_pmax_in=1,
        gate_improvements=[1.0],
        num_shots=10,
        rounds=2,
        show_progress=False,
        max_inner_workers=1,
    )
    wall = time.perf_counter() - t0
    print(f"Routing completed in {wall:.1f}s")
    print(f"exec_time={exec_time}")
    print(f"phase_aware={results.get('phase_aware', False)}")
    print(f"cached_phases={results.get('cached_phases', 'N/A')}")
    print(f"total_phases={results.get('total_phases', 'N/A')}")
    print("TEST PASSED")
except Exception as e:
    wall = time.perf_counter() - t0
    print(f"FAILED after {wall:.1f}s: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
