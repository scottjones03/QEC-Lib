"""Quick CSS surgery compilation test — no logging."""
import sys, os
os.environ["WISE_INPROCESS_LIMIT"] = "999999999"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)

    from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
    from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
    from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
    from qectostim.experiments.hardware_simulation.trapped_ion.demo.run import compile_gadget_for_animation
    from collections import Counter
    import time

    gadget = CSSSurgeryCNOTGadget()
    code = RotatedSurfaceCode(distance=2)
    ft = FaultTolerantGadgetExperiment(codes=[code], gadget=gadget, noise_model=None,
                                        num_rounds_before=2, num_rounds_after=2)
    ideal = ft.to_stim()
    meta = ft.qec_metadata
    alloc = ft._unified_allocation

    print(f"Phases: {len(meta.phases)}")
    for i, ph in enumerate(meta.phases):
        print(f"  [{i}] {ph.phase_type}  blocks={ph.active_blocks}")

    t0 = time.perf_counter()
    result = compile_gadget_for_animation(
        ideal, qec_metadata=meta, gadget=gadget, qubit_allocation=alloc,
        trap_capacity=2, lookahead=1, subgridsize=(12, 12, 0),
        base_pmax_in=1, show_progress=False,
    )
    dt = time.perf_counter() - t0

    arch_g, compiler_g, compiled_g, batches_g = result[0], result[1], result[2], result[3]
    all_ops = compiled_g.scheduled.metadata.get('all_operations', [])
    op_types = Counter(type(op).__name__ for op in all_ops)

    print(f"\nCompile time: {dt:.1f}s")
    print(f"Ops: {len(all_ops)}")
    print(f"Batches: {len(batches_g)}")
    print(f"Op types: {dict(op_types)}")
