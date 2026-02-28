"""Test if sg222 configs work. Avoids importing ionRoutingGadgetArch."""
import sys, traceback, numpy as np, stim

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    
    # Import only what exists in e7a1550
    from qectostim.experiments.hardware_simulation.trapped_ion.utils.best_effort_compilation_WISE import _route_and_simulate

    configs = [
        ("d2_k2_la1_sg430",   1, (4, 3, 0), 2, 3, 2),
        ("d2_k2_la1_sg222",   1, (2, 2, 2), 2, 3, 2),
        ("d2_k4_la1_sg222",   1, (2, 2, 2), 1, 3, 4),
    ]
    
    for label, la, subgrid, m, n, k in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        
        ideal = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=1, distance=2)
        nqubitsNeeded = m * n * k
        
        try:
            exec_time, results, reconfig_time, comp_time = _route_and_simulate(
                ideal=ideal,
                m_traps=m,
                n_traps=n,
                trap_capacity=k,
                nqubitsNeeded=nqubitsNeeded,
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
            print(f"  SUCCESS: exec_time={exec_time:.6f}")
        except Exception:
            print(f"  FAILED:")
            traceback.print_exc()
