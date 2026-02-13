#!/usr/bin/env python3
"""Compare noise application methods for CNOT-HTEL."""
import stim
from qectostim.codes import SteaneCode713
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()
noise = StimStyleDepolarizingNoise(p=1e-8)
gadget = CNOTHTeleportGadget(input_state="0")

# Method 1: noise_model=noise in constructor
exp1 = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=noise,
    num_rounds_before=3, num_rounds_after=3,
)
circ1 = exp1.to_stim()

# Method 2: noise_model=None + noise.apply()  
exp2 = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=None,
    num_rounds_before=3, num_rounds_after=3,
)
base2 = exp2.to_stim()
circ2 = noise.apply(base2)

dem1 = circ1.detector_error_model(decompose_errors=False)
dem2 = circ2.detector_error_model(decompose_errors=False)

print(f"Method 1 (noise_model=noise): {circ1.num_detectors} det, {dem1.num_errors} err")
print(f"Method 2 (noise.apply):       {circ2.num_detectors} det, {dem2.num_errors} err")

# Check shortest graphlike for both
try:
    sp1 = dem1.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    print(f"Method 1 shortest graphlike: {len(sp1)}")
except Exception as e:
    print(f"Method 1 shortest graphlike: ERROR: {e}")

try:
    sp2 = dem2.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    print(f"Method 2 shortest graphlike: {len(sp2)}")
except Exception as e:
    print(f"Method 2 shortest graphlike: ERROR: {e}")

# Check weight-1 errors with L flip
for name, dem in [("Method 1", dem1), ("Method 2", dem2)]:
    w1_count = 0
    for error in dem:
        if error.type != "error":
            continue
        targets = error.targets_copy()
        dets = [t for t in targets if t.is_relative_detector_id()]
        obs = [t for t in targets if t.is_logical_observable_id()]
        if len(dets) == 1 and len(obs) > 0:
            w1_count += 1
    print(f"{name}: {w1_count} weight-1 errors with L flip")
