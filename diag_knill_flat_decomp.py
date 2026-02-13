#!/usr/bin/env python3
"""Debug the flat KnillEC 0 decomposition failure."""
import stim
from qectostim.codes import SteaneCode713
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()
noise = StimStyleDepolarizingNoise(p=0.001)
gadget = KnillECGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=noise,
    num_rounds_before=3, num_rounds_after=3,
)
circ = exp.to_stim()

print(f"Detectors: {circ.num_detectors}")

# Try decompose_errors=False first
dem_nodecomp = circ.detector_error_model(decompose_errors=False)
print(f"DEM (no_decompose): {dem_nodecomp.num_errors} errors")

# Find errors with max symptoms
max_symp = 0
for error in dem_nodecomp:
    if error.type != "error":
        continue
    targets = error.targets_copy()
    dets = [t for t in targets if t.is_relative_detector_id()]
    if len(dets) > max_symp:
        max_symp = len(dets)
        print(f"  New max: {len(dets)} detectors, prob={error.args_copy()[0]:.3e}")
        det_ids = [t.val for t in dets]
        obs = [t.val for t in targets if t.is_logical_observable_id()]
        if len(dets) <= 20:
            print(f"    dets={det_ids}, obs={obs}")

print(f"\nMax detector symptoms per error: {max_symp}")

# Count errors by symptom weight
from collections import Counter
weight_counts = Counter()
for error in dem_nodecomp:
    if error.type != "error":
        continue
    targets = error.targets_copy()
    dets = [t for t in targets if t.is_relative_detector_id()]
    weight_counts[len(dets)] += 1

print(f"\nError weight distribution:")
for w in sorted(weight_counts.keys()):
    print(f"  weight {w}: {weight_counts[w]} errors")

# Try decompose_errors=True to see the actual error
try:
    dem_decomp = circ.detector_error_model(decompose_errors=True)
    print(f"\nDEM (decompose): {dem_decomp.num_errors} errors")
except Exception as e:
    print(f"\nDEM (decompose) FAILED: {str(e)[:200]}")

# Check which detectors are new (comparing to expected 48)
# First 48 should be the original, 49-62 are new?
# Actually, the auto-detector found 60 before, now 63 with 3 new boundary detectors
# The expected was 48 from the manual detector count
print(f"\n=== Detector count: {circ.num_detectors} (was expected 48 flat, 60 before our fix) ===")

# Check: old regression expected 48 detectors for flat KnillEC.
# Our changes added 3 bell_a Z boundary detectors (from the ancilla threshold fix).
# But the flat code previously had 48 detectors. Let's see what the auto-discovery gives.

# Also check: does the issue occur with decompose_errors=False? (for the regression)
print("\nWith decompose_errors=False:")
dem = circ.detector_error_model(decompose_errors=False)
undetectable = sum(1 for e in dem if e.type == "error" 
                   and not any(t.is_relative_detector_id() for t in e.targets_copy())
                   and any(t.is_logical_observable_id() for t in e.targets_copy()))
print(f"  Errors: {dem.num_errors}, Undetectable: {undetectable}")
