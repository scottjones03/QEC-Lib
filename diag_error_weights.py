#!/usr/bin/env python3
"""Analyze error weight distribution in the DEM."""
import numpy as np
import stim
from collections import Counter

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

noise = StimStyleDepolarizingNoise(p=1e-4)
noisy = noise.apply(base)
dem = noisy.detector_error_model(decompose_errors=True)

print(f"DEM: {dem.num_detectors} det, {dem.num_errors} err, {dem.num_observables} obs")

# Weight distribution
weight_counter = Counter()
weight_obs_counter = Counter()
total_prob_by_weight = Counter()

for instr in dem.flattened():
    if instr.type != "error":
        continue
    targets = instr.targets_copy()
    prob = instr.args_copy()[0]
    n_det = sum(1 for t in targets if t.is_relative_detector_id())
    has_obs = any(t.is_logical_observable_id() for t in targets)
    
    weight_counter[n_det] += 1
    total_prob_by_weight[n_det] += prob
    if has_obs:
        weight_obs_counter[n_det] += 1

print("\nError weight distribution (detector weight):")
print(f"{'Weight':<10} {'Count':<10} {'With L0':<10} {'Total prob':<15}")
print("-" * 45)
for w in sorted(weight_counter.keys()):
    print(f"{w:<10} {weight_counter[w]:<10} {weight_obs_counter.get(w,0):<10} {total_prob_by_weight[w]:<15.6e}")

# Also compare with Stim reference surface code
print("\n\nStim surface code d=3 r=3 for comparison:")
circ_ref = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=3, rounds=3,
    after_clifford_depolarization=1e-4,
)
dem_ref = circ_ref.detector_error_model(decompose_errors=True)
print(f"DEM: {dem_ref.num_detectors} det, {dem_ref.num_errors} err, {dem_ref.num_observables} obs")

wc_ref = Counter()
woc_ref = Counter()
for instr in dem_ref.flattened():
    if instr.type != "error":
        continue
    targets = instr.targets_copy()
    n_det = sum(1 for t in targets if t.is_relative_detector_id())
    has_obs = any(t.is_logical_observable_id() for t in targets)
    wc_ref[n_det] += 1
    if has_obs:
        woc_ref[n_det] += 1

print(f"\n{'Weight':<10} {'Count':<10} {'With L0':<10}")
print("-" * 30)
for w in sorted(wc_ref.keys()):
    print(f"{w:<10} {wc_ref[w]:<10} {woc_ref.get(w,0):<10}")

# Try shortest error
print(f"\nShortest graphlike error (Steane CNOT-HTEL):")
try:
    path = dem.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    print(f"  weight={len(path)}")
    for e in path:
        print(f"  {e}")
except Exception as ex:
    print(f"  failed: {ex}")

print(f"\nShortest graphlike error (Surface d=3):")
try:
    path = dem_ref.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    print(f"  weight={len(path)}")
    for e in path:
        print(f"  {e}")
except Exception as ex:
    print(f"  failed: {ex}")

print("\nDone.")
