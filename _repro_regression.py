"""Reproduce the IndexError regression from commit 243a188."""
import traceback
from qectostim.experiments.hardware_simulation.trapped_ion.utils.best_effort_compilation_WISE import run_single_config

# Config that FAILS: d=2, k=2, lookahead=1, subgrid=(4,3,0)
print("=== Test 1: d=2, k=2, la=1, subgrid=(4,3,0) — EXPECTED FAIL ===")
try:
    exec_t, comp_t, res, reconf_t = run_single_config(
        lookahead=1, subgrid_width=4, subgrid_height=3,
        subgrid_increment=0, d=2, trap_capacity=2,
        base_pmax_in=1, gate_improvements=[1.0], num_shots=10,
        show_progress=False, max_inner_workers=1,
    )
    import numpy as np
    if np.isnan(exec_t):
        print(f"FAILED (NaN) — comp_time={comp_t:.1f}s")
    else:
        print(f"SUCCESS: exec_t={exec_t}")
except Exception as e:
    traceback.print_exc()

# Config that FAILS: d=2, k=4, lookahead=1, subgrid=(2,2,2)
print("\n=== Test 2: d=2, k=4, la=1, subgrid=(2,2,2) — EXPECTED FAIL ===")
try:
    exec_t2, comp_t2, res2, reconf_t2 = run_single_config(
        lookahead=1, subgrid_width=2, subgrid_height=2,
        subgrid_increment=2, d=2, trap_capacity=4,
        base_pmax_in=1, gate_improvements=[1.0], num_shots=10,
        show_progress=False, max_inner_workers=1,
    )
    import numpy as np
    if np.isnan(exec_t2):
        print(f"FAILED (NaN) — comp_time={comp_t2:.1f}s")
    else:
        print(f"SUCCESS: exec_t={exec_t2}")
except Exception as e:
    traceback.print_exc()

# Config that WORKS: d=2, k=2, lookahead=4, subgrid=(4,3,0)
print("\n=== Test 3: d=2, k=2, la=4, subgrid=(4,3,0) — EXPECTED PASS ===")
try:
    exec_t3, comp_t3, res3, reconf_t3 = run_single_config(
        lookahead=4, subgrid_width=4, subgrid_height=3,
        subgrid_increment=0, d=2, trap_capacity=2,
        base_pmax_in=1, gate_improvements=[1.0], num_shots=10,
        show_progress=False, max_inner_workers=1,
    )
    import numpy as np
    if np.isnan(exec_t3):
        print(f"FAILED (NaN) — comp_time={comp_t3:.1f}s")
    else:
        print(f"SUCCESS: exec_t={exec_t3}")
except Exception as e:
    traceback.print_exc()
