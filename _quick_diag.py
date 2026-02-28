#!/usr/bin/env python3
"""Quick diagnostic for phase-aware routing."""
import os
import sys
os.environ['WISE_INPROCESS_LIMIT'] = '999999999'

# Reduce logging noise
import logging
logging.basicConfig(level=logging.WARNING)

from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.experiments.hardware_simulation.trapped_ion.demo.run import (
    compile_gadget_for_animation,
)

print("Building experiment...", file=sys.stderr)
code = RotatedSurfaceCode(distance=2)
gadget = CSSSurgeryCNOTGadget()
ft = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=None,
    num_rounds_before=1, num_rounds_after=1,  # Minimal EC rounds
)
circuit = ft.to_stim()
meta = ft.qec_metadata
alloc = ft._unified_allocation

print("Compiling...", file=sys.stderr)
try:
    arch, compiler, compiled, batches, ion_roles, p2l, remap = (
        compile_gadget_for_animation(
            circuit,
            qec_metadata=meta,
            gadget=gadget,
            qubit_allocation=alloc,
            trap_capacity=2,
            lookahead=1,
            subgridsize=(8, 6, 0),  # Smaller subgrid for faster SAT
            base_pmax_in=1,
            show_progress=False,
        )
    )
    
    # Count MS gates
    ms_count = 0
    for batch in batches:
        ops = getattr(batch, 'operations', [batch])
        for op in ops:
            if "TwoQubitMS" in type(op).__name__:
                ms_count += 1
    
    print(f"\n=== RESULT: {ms_count} MS gates in {len(batches)} batches ===", file=sys.stderr)
    
except Exception as e:
    print(f"\n*** FAILED: {e} ***", file=sys.stderr)
    import traceback
    traceback.print_exc()
