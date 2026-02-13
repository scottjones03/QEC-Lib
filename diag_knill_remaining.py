#!/usr/bin/env python3
"""Check what changed after the fix: which error is still undetectable?"""
import stim
from qectostim.codes import SteaneCode713
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()
gadget = KnillECGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=None,
    num_rounds_before=3, num_rounds_after=3,
)
base = exp.to_stim()
noise = StimStyleDepolarizingNoise(p=1e-8)
noisy = noise.apply(base)
dem = noisy.detector_error_model(decompose_errors=False)

print(f"Detectors: {dem.num_detectors}, Errors: {dem.num_errors}")

# Find the undetectable error
for instr in dem.flattened():
    if instr.type != "error":
        continue
    targets = instr.targets_copy()
    prob = instr.args_copy()[0]
    has_det = any(t.is_relative_detector_id() for t in targets)
    has_obs = any(t.is_logical_observable_id() for t in targets)
    if has_obs and not has_det:
        print(f"\nUndetectable error: prob={prob:.3e}")
        
        # Build filter DEM
        filter_dem = stim.DetectorErrorModel()
        filter_dem.append("error", [prob], targets)
        
        try:
            explanations = noisy.explain_detector_error_model_errors(dem_filter=filter_dem)
            print("Circuit sources:")
            for exp_item in explanations:
                # Just show first few
                lines = str(exp_item).split('\n')
                for line in lines[:15]:
                    print(f"  {line}")
                if len(lines) > 15:
                    print(f"  ... ({len(lines)-15} more lines)")
                break  # Just first explanation
        except Exception as e:
            print(f"  explain failed: {e}")

# Also check: what are the new detectors?
print(f"\n=== Checking new boundary detectors ===")
# Build measurement table
flat = list(base.flattened())
meas_table = []
for inst in flat:
    if inst.name in ("M", "MX", "MR", "MRX", "MY"):
        for t in inst.targets_copy():
            meas_table.append((len(meas_table), inst.name, t.value))

bell_a_data = {61,62,63,64,65,66,67}
data_mx = {54,55,56,57,58,59,60}

running_meas = 0
for inst in flat:
    if inst.name in ("M", "MX", "MR", "MRX", "MY"):
        running_meas += len(inst.targets_copy())
    elif inst.name == "DETECTOR":
        tgts = inst.targets_copy()
        abs_refs = set()
        for t in tgts:
            abs_refs.add(running_meas + t.value)
        
        refs_bella = abs_refs & bell_a_data
        refs_data = abs_refs & data_mx
        if refs_bella or refs_data:
            mapped = []
            for idx in sorted(abs_refs):
                if idx < len(meas_table):
                    _, name, qubit = meas_table[idx]
                    block = "data" if qubit <= 12 else ("bell_a" if qubit <= 25 else "bell_b")
                    mapped.append(f"{name}(Q{qubit})[{block}]")
                else:
                    mapped.append(f"meas[{idx}]?")
            print(f"  Boundary det: {mapped}")

print("\nDone.")
