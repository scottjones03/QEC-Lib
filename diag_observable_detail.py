#!/usr/bin/env python3
"""Check observable definition and whether the shortest error path makes sense."""
import stim
import numpy as np

from qectostim.codes import SteaneCode713
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()
gadget = CNOTHTeleportGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=None,
    num_rounds_before=3, num_rounds_after=3,
)
base = exp.to_stim()

# Print the OBSERVABLE_INCLUDE instructions
print("OBSERVABLE_INCLUDE instructions:")
for i, inst in enumerate(base.flattened()):
    if inst.name == "OBSERVABLE_INCLUDE":
        print(f"  instr {i}: {inst}")

# Print first few and last few DETECTOR instructions
print("\nFirst 5 DETECTOR instructions:")
det_idx = 0
for inst in base.flattened():
    if inst.name == "DETECTOR":
        if det_idx < 5:
            print(f"  D{det_idx}: {inst}")
        det_idx += 1
print(f"\nTotal detectors: {det_idx}")

# Now create noisy circuit and examine the specific error mechanisms
# that form the weight-2 logical error path
noise = StimStyleDepolarizingNoise(p=1e-4)
noisy = noise.apply(base)
dem = noisy.detector_error_model(decompose_errors=True)

# Find the weight-2 logical error path
path = dem.shortest_graphlike_error(ignore_ungraphlike_errors=True)
print(f"\nShortest graphlike error path (weight {len(path)}):")
for e in path:
    targets = e.targets_copy()
    prob = e.args_copy()[0]
    det_ids = [t.val for t in targets if t.is_relative_detector_id()]
    obs_ids = [t.val for t in targets if t.is_logical_observable_id()]
    print(f"  prob={prob:.2e}  dets={det_ids}  obs={obs_ids}")

# Check if these are from different time slices or the same
# Also check the circuit explanation
print("\n\nCircuit-level explanation of logical error:")
try:
    explain = noisy.explain_detector_error_model_errors(
        dem_filter=stim.DetectorErrorModel(str(path[0])),
        reduce_to_one_representative_error=True,
    )
    print(f"Error 1: {explain}")
except Exception as e:
    print(f"Error 1 explain failed: {e}")

try:
    explain2 = noisy.explain_detector_error_model_errors(
        dem_filter=stim.DetectorErrorModel(str(path[1])),
        reduce_to_one_representative_error=True,
    )
    print(f"Error 2: {explain2}")
except Exception as e:
    print(f"Error 2 explain failed: {e}")

# Also check: what are D3 and D27?
print("\n\nDetector coordinates:")
for inst in dem.flattened():
    if inst.type == "detector" and inst.targets_copy()[0].val in (3, 27):
        print(f"  D{inst.targets_copy()[0].val}: coords={inst.args_copy()}")

# Count errors involving both D3 and D27
print("\n\nErrors involving D3 AND D27:")
count = 0
for inst in dem.flattened():
    if inst.type != "error":
        continue
    targets = inst.targets_copy()
    det_ids = {t.val for t in targets if t.is_relative_detector_id()}
    if 3 in det_ids and 27 in det_ids:
        prob = inst.args_copy()[0]
        obs = [t.val for t in targets if t.is_logical_observable_id()]
        print(f"  prob={prob:.2e}  dets={sorted(det_ids)}  obs={obs}")
        count += 1
print(f"Total: {count}")

print("\nDone.")
