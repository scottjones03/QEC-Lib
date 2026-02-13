#!/usr/bin/env python3
"""Quick diagnostic: how many detectors survive into the DEM?"""
import stim
from qectostim.codes import SteaneCode713
from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

inner = SteaneCode713()
code = ConcatenatedCSSCode(inner, inner)

gadget = CNOTHTeleportGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=None,
    num_rounds_before=3, num_rounds_after=3,
)
base = exp.to_stim()

# Count detectors in circuit
det_count = sum(1 for inst in base.flattened() if inst.name == "DETECTOR")
obs_count = sum(1 for inst in base.flattened() if inst.name == "OBSERVABLE_INCLUDE")
print(f"Circuit: {det_count} DETECTOR instructions, {obs_count} OBSERVABLE_INCLUDE")

# Build DEM without noise — how many survive?
try:
    dem_noiseless = base.detector_error_model(decompose_errors=True)
    print(f"Noiseless DEM (decompose=True): {dem_noiseless.num_detectors} det, {dem_noiseless.num_errors} err")
except ValueError as e:
    print(f"Noiseless DEM (decompose=True) failed: {e}")

try:
    dem_noiseless_nd = base.detector_error_model(decompose_errors=False)
    print(f"Noiseless DEM (decompose=False): {dem_noiseless_nd.num_detectors} det, {dem_noiseless_nd.num_errors} err")
except ValueError as e:
    print(f"Noiseless DEM (decompose=False) failed: {e}")

# Now with noise
for p in [0.0005, 0.001]:
    noise = StimStyleDepolarizingNoise(p=p)
    noisy = noise.apply(base)
    
    det_count_noisy = sum(1 for inst in noisy.flattened() if inst.name == "DETECTOR")
    print(f"\np={p}: circuit has {det_count_noisy} DETECTOR instructions")
    
    try:
        dem = noisy.detector_error_model(decompose_errors=True)
        print(f"  DEM (decompose=True): {dem.num_detectors} det, {dem.num_errors} err")
    except ValueError as e:
        print(f"  DEM (decompose=True) failed: {str(e)[:80]}")
    
    try:
        dem_ig = noisy.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
        print(f"  DEM (ignore_failures): {dem_ig.num_detectors} det, {dem_ig.num_errors} err")
    except ValueError as e:
        print(f"  DEM (ignore_failures) failed: {str(e)[:80]}")
    
    try:
        dem_nd = noisy.detector_error_model(decompose_errors=False)
        print(f"  DEM (decompose=False): {dem_nd.num_detectors} det, {dem_nd.num_errors} err")
    except ValueError as e:
        print(f"  DEM (decompose=False) failed: {str(e)[:80]}")
