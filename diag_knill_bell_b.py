#!/usr/bin/env python3
"""Understand the bell_b syndrome extraction and why XX error on Q29,Q36 is undetectable.

Key question: what bell_b detectors exist, and why don't they catch this X error?
"""
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

# Build measurement table
meas_table = []
for inst in circ.flattened():
    if inst.name in ("M", "MX", "MR", "MRX", "MY"):
        for t in inst.targets_copy():
            meas_table.append((len(meas_table), inst.name, t.value))

n_meas = len(meas_table)

# Map qubits to blocks
block_map = {}
for q in range(7): block_map[q] = "data"
for q in range(7,13): block_map[q] = "data_anc"
for q in range(13,20): block_map[q] = "bell_a"
for q in range(20,26): block_map[q] = "bell_a_anc"
for q in range(26,33): block_map[q] = "bell_b"
for q in range(33,39): block_map[q] = "bell_b_anc"

# Show all bell_b qubit measurements
print("=== Bell_b qubit measurements ===")
for q in range(26, 39):
    indices = [i for i, (_, n, q2) in enumerate(meas_table) if q2 == q]
    role = "data" if q < 33 else ("X_anc" if q < 36 else "Z_anc")
    types = [meas_table[i][1] for i in indices]
    print(f"  Q{q} ({block_map[q]} {role}): {len(indices)} meas at {indices}, types={types}")

# Show detectors referencing bell_b measurements
bell_b_meas_range = set()
for q in range(26, 39):
    for i, (_, n, q2) in enumerate(meas_table):
        if q2 == q:
            bell_b_meas_range.add(i)

print(f"\nBell_b measurement indices: {sorted(bell_b_meas_range)}")

# Parse detectors  
det_idx = 0
det_info = {}
for inst in circ.flattened():
    if inst.name == "DETECTOR":
        rec_targets = [t for t in inst.targets_copy() if t.is_measurement_record_target]
        meas_indices = [n_meas + t.value for t in rec_targets]
        det_info[det_idx] = meas_indices
        det_idx += 1

print(f"\nTotal detectors: {len(det_info)}")

# Which detectors reference bell_b?
print("\n=== Detectors referencing bell_b measurements ===")
for d, mindices in det_info.items():
    if any(m in bell_b_meas_range for m in mindices):
        mapped = [(meas_table[m][1], f"Q{meas_table[m][2]}({block_map.get(meas_table[m][2], '?')})") 
                  for m in mindices]
        print(f"  D{d}: {mapped}")

# What specific gates happen at tick 117 on Q29, Q36?
print("\n=== Circuit around tick 117 ===")
tick = 0
for inst in circ.flattened():
    if inst.name == "TICK":
        tick += 1
        continue
    if 115 <= tick <= 120:
        targets = inst.targets_copy()
        qs = []
        for t in targets:
            if hasattr(t, 'value'):
                qs.append(t.value)
        if any(q in {29, 36} for q in qs) or inst.name in ("TICK",):
            print(f"  tick={tick}: {inst.name} targets={[t for t in targets]}")

# Observable: which measurements?
print("\n=== Observable formula ===")
for inst in circ.flattened():
    if inst.name == "OBSERVABLE_INCLUDE":
        rec_targets = [t for t in inst.targets_copy() if t.is_measurement_record_target]
        meas_indices = [n_meas + t.value for t in rec_targets]
        mapped = [(meas_table[m][1], f"Q{meas_table[m][2]}({block_map.get(meas_table[m][2], '?')})") 
                  for m in meas_indices]
        print(f"  args={inst.gate_args_copy()}: {mapped}")

# Now: the key question. Q29 is bell_b data qubit in logical Z support.
# Q36 is bell_b X ancilla.
# XX error means: X on Q29 (data) + X on Q36 (X ancilla)
# X error on an X ancilla during X syndrome extraction = measurement flip
# X error on data qubit Q29 = should be caught by Z stabilizer

# Check: what Z stabilizers involve Q29?
print(f"\n=== Steane Z stabilizers involving Q29 ===")
print(f"Q29 in bell_b maps to local qubit {29-26}=3")
z_stabs = code.z_stabilizers  
print(f"Z stabilizers: {z_stabs}")
for i, stab in enumerate(z_stabs):
    if stab[3] == 1:  # qubit 3
        print(f"  Z_stab[{i}]: {stab}")

# Check: is Q29 in the logical Z operator support?
print(f"\nLogical Z operator: {code.z_logical}")
print(f"Q29 (local 3) in Z logical: {code.z_logical[3]}")

print("\nDone.")
