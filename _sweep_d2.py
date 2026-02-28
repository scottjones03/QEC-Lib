"""Run d=2 sweep to validate all configs pass with the bug fixes."""
import sys
import numpy as np
import stim
import logging

logging.basicConfig(level=logging.WARNING)

from qectostim.experiments.hardware_simulation.trapped_ion.utils.best_effort_compilation_WISE import _route_and_simulate

d = 2
configs = []
for k_val in [2, 4]:
    for la in [1, 2, 4]:
        m, n = (2, 3) if k_val == 2 else (1, 3)
        for sg in [(4, 3, 0), (3, 3, 0), (2, 2, 2), (2, 2, 0)]:
            label = f"d{d}_k{k_val}_la{la}_sg{sg[0]}{sg[1]}{sg[2]}"
            configs.append((label, la, sg, m, n, k_val))

ideal = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=1, distance=d)

passed = 0
failed = 0
errors = []

for label, la, subgrid, m, n, k_val in configs:
    nq = m * n * k_val
    sys.stdout.write(f"{label:30s} ... ")
    sys.stdout.flush()
    try:
        exec_time, results, rt, ct = _route_and_simulate(
            ideal=ideal,
            m_traps=m,
            n_traps=n,
            trap_capacity=k_val,
            nqubitsNeeded=nq,
            lookahead=la,
            subgridsize=subgrid,
            base_pmax_in=1,
            barrier_threshold=np.inf,
            gate_improvements=[1.0],
            num_shots=10,
            routing_config=None,
            stop_event=None,
            toMoveOps=None,
            show_progress=False,
            max_inner_workers=1,
        )
        print(f"PASS (exec={exec_time:.6f})")
        passed += 1
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        failed += 1
        errors.append(label)

print(f"\n===== RESULTS: {passed}/{passed+failed} passed, {failed} failed =====")
if errors:
    print("Failed configs:", errors)
