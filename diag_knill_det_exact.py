#!/usr/bin/env python3
"""Print exact detector formulas referencing Q36 measurements, with meas indices."""
import stim
from qectostim.codes import SteaneCode713
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()
noise = StimStyleDepolarizingNoise(p=1e-8)
gadget = KnillECGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=noise,
    num_rounds_before=3, num_rounds_after=3,
)
circ = exp.to_stim()

# Measurement table
meas_table = []
for inst in circ.flattened():
    if inst.name in ("M", "MX", "MR", "MRX", "MY"):
        for t in inst.targets_copy():
            meas_table.append((len(meas_table), inst.name, t.value))
n_meas = len(meas_table)

q36_meas = {i for i, (_, n, q) in enumerate(meas_table) if q == 36}
q29_meas = {i for i, (_, n, q) in enumerate(meas_table) if q == 29}

print(f"Q36 measurements: {sorted(q36_meas)} (types: {[meas_table[i][1] for i in sorted(q36_meas)]})")
print(f"Q29 measurements: {sorted(q29_meas)} (types: {[meas_table[i][1] for i in sorted(q29_meas)]})")

# Now show ALL detectors, not just ones referencing Q36/Q29
det_idx = 0
print(f"\n=== ALL {circ.num_detectors} detectors ===")
for inst in circ.flattened():
    if inst.name == "DETECTOR":
        rec_targets = [t for t in inst.targets_copy() if t.is_measurement_record_target]
        meas_indices = sorted([n_meas + t.value for t in rec_targets])
        
        # Only print if references Q36 or Q29
        if any(m in q36_meas for m in meas_indices) or any(m in q29_meas for m in meas_indices):
            mapped = []
            for m in meas_indices:
                mapped.append(f"m[{m}]={meas_table[m][1]}(Q{meas_table[m][2]})")
            marker = ""
            if any(m in q36_meas for m in meas_indices):
                marker += " <-- Q36"
            if any(m in q29_meas for m in meas_indices):
                marker += " <-- Q29"
            print(f"  D{det_idx}: {' ⊕ '.join(mapped)}{marker}")
        det_idx += 1

# Now verify: is D39 = meas[48] ⊕ meas[68]?
# What's at meas[30]?
print(f"\nmeas[30] = {meas_table[30][1]}(Q{meas_table[30][2]})")
print(f"meas[48] = {meas_table[48][1]}(Q{meas_table[48][2]})")
print(f"meas[68] = {meas_table[68][1]}(Q{meas_table[68][2]})")

# Check: D15 is the anchor for Q36.
# D39 should be temporal: meas[30] or meas[48] vs meas[48] or meas[68]

# Now the KEY: observable formula
print(f"\n=== Observable ===")
for inst in circ.flattened():
    if inst.name == "OBSERVABLE_INCLUDE":
        rec_targets = [t for t in inst.targets_copy() if t.is_measurement_record_target]
        meas_indices = sorted([n_meas + t.value for t in rec_targets])
        mapped = [f"m[{m}]={meas_table[m][1]}(Q{meas_table[m][2]})" for m in meas_indices]
        print(f"  L{int(inst.gate_args_copy()[0])}: {' ⊕ '.join(mapped)}")

# Manual check: if X on Q29 flips m[89] and X on Q36 flips m[48]:
# Which detectors reference m[48]?
print(f"\n=== Detectors referencing m[48] ===")
det_idx = 0
for inst in circ.flattened():
    if inst.name == "DETECTOR":
        rec_targets = [t for t in inst.targets_copy() if t.is_measurement_record_target]
        meas_indices = sorted([n_meas + t.value for t in rec_targets])
        if 48 in meas_indices:
            mapped = [f"m[{m}]={meas_table[m][1]}(Q{meas_table[m][2]})" for m in meas_indices]
            print(f"  D{det_idx}: {' ⊕ '.join(mapped)}")
        det_idx += 1

print(f"\n=== Detectors referencing m[89] ===")
det_idx = 0
for inst in circ.flattened():
    if inst.name == "DETECTOR":
        rec_targets = [t for t in inst.targets_copy() if t.is_measurement_record_target]
        meas_indices = sorted([n_meas + t.value for t in rec_targets])
        if 89 in meas_indices:
            mapped = [f"m[{m}]={meas_table[m][1]}(Q{meas_table[m][2]})" for m in meas_indices]
            print(f"  D{det_idx}: {' ⊕ '.join(mapped)}")
        det_idx += 1

print("\nDone.")
