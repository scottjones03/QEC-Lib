#!/usr/bin/env python3
"""Check detector counts with different round configurations."""
import stim
from qectostim.codes import SteaneCode713
from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment

inner = SteaneCode713()
code = ConcatenatedCSSCode(inner, inner)
gadget = CNOTHTeleportGadget(input_state="0")

for rb, ra, di in [(3, 3, None), (2, 2, 3), (2, 2, None), (3, 3, 3)]:
    kw = dict(codes=[code], gadget=gadget, noise_model=None,
              num_rounds_before=rb, num_rounds_after=ra)
    if di is not None:
        kw["d_inner"] = di
    exp = FaultTolerantGadgetExperiment(**kw)
    circ = exp.to_stim()
    n_det_instr = sum(1 for inst in circ.flattened() if inst.name == "DETECTOR")
    n_det_prop = circ.num_detectors
    print(f"rounds_b={rb} rounds_a={ra} d_inner={di}: "
          f"num_detectors={n_det_prop}, DETECTOR instructions={n_det_instr}")
