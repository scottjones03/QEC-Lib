#!/usr/bin/env python3
"""Precisely trace the tick-by-tick timeline of Q29 and Q36 to understand
which measurement is affected by the XX error at tick 117."""
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

# Track all operations on Q29 and Q36
print("=== Timeline for Q29 (bell_b data) and Q36 (bell_b Z ancilla) ===")
tick = 0
meas_idx = 0
for inst in circ.flattened():
    if inst.name == "TICK":
        tick += 1
        continue
    
    targets = inst.targets_copy()
    qs = set()
    for t in targets:
        if hasattr(t, 'value') and not t.is_measurement_record_target:
            qs.add(t.value)
    
    if inst.name in ("M", "MX", "MR", "MRX"):
        for t in targets:
            if hasattr(t, 'value') and not t.is_measurement_record_target:
                if t.value in {29, 36}:
                    print(f"  tick={tick}: {inst.name} Q{t.value} -> meas[{meas_idx}]")
                meas_idx += 1
    elif qs & {29, 36}:
        relevant = qs & {29, 36}
        print(f"  tick={tick}: {inst.name} qubits={sorted(qs)} (relevant: {sorted(relevant)})")

# Now check: what detector compares which pair of Q36 measurements
print("\n=== Detectors involving Q36 measurements ===")
n_meas = len([inst for inst in circ.flattened() 
              if inst.name in ("M", "MX", "MR", "MRX") for t in inst.targets_copy()])

# Re-count properly
meas_table = []
for inst in circ.flattened():
    if inst.name in ("M", "MX", "MR", "MRX", "MY"):
        for t in inst.targets_copy():
            meas_table.append((len(meas_table), inst.name, t.value))
n_meas = len(meas_table)

q36_meas = [i for i, (_, n, q) in enumerate(meas_table) if q == 36]
q29_meas = [i for i, (_, n, q) in enumerate(meas_table) if q == 29]
print(f"Q36 measurements: {q36_meas}")
print(f"Q29 measurements: {q29_meas}")

det_idx = 0
for inst in circ.flattened():
    if inst.name == "DETECTOR":
        rec_targets = [t for t in inst.targets_copy() if t.is_measurement_record_target]
        meas_indices = [n_meas + t.value for t in rec_targets]
        if any(m in q36_meas for m in meas_indices) or any(m in q29_meas for m in meas_indices):
            mapped = [(meas_table[m][1], f"Q{meas_table[m][2]}", m) for m in meas_indices]
            print(f"  D{det_idx}: {mapped}")
        det_idx += 1

# Key question: if the XX error at tick 117 flips Q36's measurement,
# WHICH measurement index is flipped?
print("\n=== What measurement is Q36 reset/measured after tick 117? ===")
tick = 0
for inst in circ.flattened():
    if inst.name == "TICK":
        tick += 1
        continue
    if tick >= 117 and inst.name in ("MR", "MRX", "M", "MX"):
        for t in inst.targets_copy():
            if hasattr(t, 'value') and t.value == 36:
                print(f"  tick={tick}: {inst.name} Q36")

# Also: what about the X error on Q29? Q29 has only 1 M at index 89.
# X on Q29: does it flip M(Q29)? M measures Z basis. X anti-commutes with Z.
# So X on Q29 flips M(Q29) → flips L0 (since Q29/meas[89] is in the observable)
# But does any detector reference meas[89]?
print(f"\n=== Does any detector reference M(Q29) at index {q29_meas}? ===")
det_idx = 0
for inst in circ.flattened():
    if inst.name == "DETECTOR":
        rec_targets = [t for t in inst.targets_copy() if t.is_measurement_record_target]
        meas_indices = [n_meas + t.value for t in rec_targets]
        if any(m in q29_meas for m in meas_indices):
            mapped = [(meas_table[m][1], f"Q{meas_table[m][2]}", m) for m in meas_indices]
            print(f"  D{det_idx}: {mapped}")
        det_idx += 1

print("\nDone.")
