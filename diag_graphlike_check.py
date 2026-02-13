#!/usr/bin/env python3
"""Check graphlike property of weight-1 errors."""
import stim
from qectostim.codes import SteaneCode713
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()
noise = StimStyleDepolarizingNoise(p=1e-8)
gadget = CNOTHTeleportGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=noise,
    num_rounds_before=3, num_rounds_after=3,
)
circ = exp.to_stim()
dem = circ.detector_error_model(decompose_errors=False)

# Show ALL weight-1 errors with L flip
print("Weight-1 errors with L flip (non-decomposed):")
for error in dem:
    if error.type != "error":
        continue
    targets = error.targets_copy()
    dets = [t for t in targets if t.is_relative_detector_id()]
    obs = [t for t in targets if t.is_logical_observable_id()]
    if len(dets) == 1 and len(obs) > 0:
        print(f"  prob={error.args_copy()[0]:.3e}, det=[D{dets[0].val}], obs=[L{obs[0].val}]")
        # A "graphlike" error can have at most 2 detector targets
        # 1 detector + 1 observable = 1 detector in the graph
        # This IS graphlike - it's a single-edge path from D to boundary

# Try without ignore_ungraphlike_errors
print("\nShortest graphlike error (ignore_ungraphlike=True):")
try:
    sp = dem.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    print(f"  Length: {len(sp)}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nShortest graphlike error (ignore_ungraphlike=False):")
try:
    sp = dem.shortest_graphlike_error(ignore_ungraphlike_errors=False)
    print(f"  Length: {len(sp)}")
except Exception as e:
    print(f"  ERROR: {e}")

# Now try with decompose_errors=True
dem_decomp = circ.detector_error_model(decompose_errors=True)
print(f"\nDecomposed DEM: {dem_decomp.num_errors} errors")

w1_decomp = 0
for error in dem_decomp:
    if error.type != "error":
        continue
    targets = error.targets_copy()
    dets = [t for t in targets if t.is_relative_detector_id()]
    obs = [t for t in targets if t.is_logical_observable_id()]
    if len(dets) == 1 and len(obs) > 0:
        w1_decomp += 1
        if w1_decomp <= 5:
            print(f"  prob={error.args_copy()[0]:.3e}, det=[D{dets[0].val}], obs=[L{obs[0].val}]")

print(f"Weight-1 decomposed errors with L flip: {w1_decomp}")

sp_decomp = dem_decomp.shortest_graphlike_error(ignore_ungraphlike_errors=True)
print(f"Shortest graphlike (decomposed): {len(sp_decomp)}")
