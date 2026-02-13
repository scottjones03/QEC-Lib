#!/usr/bin/env python3
"""Check what auto_detector_emission discovers for KnillEC.

Key questions:
1. Does it find boundary detectors for data_block (MX)?
2. Does it find boundary detectors for bell_a (M)?
3. What's the total detector count?
"""
import stim
from qectostim.codes import SteaneCode713
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.experiments.auto_detector_emission import discover_detectors

code = SteaneCode713()
gadget = KnillECGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=None,
    num_rounds_before=3, num_rounds_after=3,
)
base = exp.to_stim()

# Strip annotations for analysis
bare = stim.Circuit()
for inst in base.flattened():
    if inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
        bare.append(inst)

result = discover_detectors(bare, use_cache=False)
print(f"Discovered {len(result.detectors)} detectors (raw: {result.raw_detector_count})")

# Build measurement table
meas_table = []
for inst in bare.flattened():
    if inst.name in ("M", "MX", "MR", "MRX", "MY"):
        for t in inst.targets_copy():
            meas_table.append((len(meas_table), inst.name, t.value))

# Check which detectors reference destroyed block measurements
bell_a_data_indices = set()
data_block_mx_indices = set()
for idx, name, qubit in meas_table:
    if qubit in {13,14,15,16,17,18,19} and name == "M":
        bell_a_data_indices.add(idx)
    if qubit in {0,1,2,3,4,5,6} and name == "MX":
        data_block_mx_indices.add(idx)

print(f"\nbell_a data measurement indices: {sorted(bell_a_data_indices)}")
print(f"data_block MX measurement indices: {sorted(data_block_mx_indices)}")

n_refs_bella = 0
n_refs_data = 0
for det in result.detectors:
    refs_bella = det.measurement_indices & bell_a_data_indices
    refs_data = det.measurement_indices & data_block_mx_indices
    if refs_bella:
        n_refs_bella += 1
        print(f"\n  Detector refs bell_a: indices={sorted(det.measurement_indices)}")
        for idx in sorted(det.measurement_indices):
            if idx < len(meas_table):
                _, name, qubit = meas_table[idx]
                block = "data" if qubit <= 12 else ("bell_a" if qubit <= 25 else "bell_b")
                print(f"    meas[{idx}] = {name}(Q{qubit}) [{block}]")
    if refs_data:
        n_refs_data += 1
        print(f"\n  Detector refs data_block: indices={sorted(det.measurement_indices)}")
        for idx in sorted(det.measurement_indices):
            if idx < len(meas_table):
                _, name, qubit = meas_table[idx]
                block = "data" if qubit <= 12 else ("bell_a" if qubit <= 25 else "bell_b")
                print(f"    meas[{idx}] = {name}(Q{qubit}) [{block}]")

print(f"\nDetectors referencing bell_a data: {n_refs_bella}")
print(f"Detectors referencing data_block MX: {n_refs_data}")

# Also manually test if specific boundary flows work
print("\n=== Manual flow tests ===")

# Test: X boundary for data_block
# Last X ancilla round for data_block uses ancillas Q7, Q8, Q9
# Their 2nd-to-last measurement would be the syndrome round before Bell CNOT
data_x_anc = {7, 8, 9}  # X ancillas for data block (Steane code)
data_z_anc = {10, 11, 12}  # Z ancillas for data block

# Find measurements for these ancillas
for q in sorted(data_x_anc | data_z_anc):
    indices = [i for i, (_, name, qubit) in enumerate(meas_table) if qubit == q]
    print(f"  Q{q}: {len(indices)} measurements, indices={indices}")

# Test: the X stabilizer boundary: last MR(Q7) xor some MX(data)
# For Steane code, X stabilizers: X0X1X3, X0X2X5, X1X2X6 (or similar)
# Actually let me just try has_flow directly
n_meas = len(meas_table)

# Find last measurements for data block X ancillas and MX data
q7_meas = [i for i, (_, n, q) in enumerate(meas_table) if q == 7]
q8_meas = [i for i, (_, n, q) in enumerate(meas_table) if q == 8]
q9_meas = [i for i, (_, n, q) in enumerate(meas_table) if q == 9]

# data_block MX indices
data_mx = sorted(data_block_mx_indices)

# Test: does MR(Q7)_last xor MX(Q0) xor MX(Q3) xor MX(Q6) form a flow?
# (This would be an X stabilizer boundary detector)
for q_anc, anc_meas_list in [(7, q7_meas), (8, q8_meas), (9, q9_meas)]:
    if len(anc_meas_list) < 2:
        continue
    second_last = anc_meas_list[-2]  # 2nd-to-last (the last full syndrome round)
    
    # Try all combinations of 2-4 data qubits
    import itertools
    for w in range(2, 5):
        for combo in itertools.combinations(data_mx, w):
            candidate = [second_last] + list(combo)
            rec_parts = []
            for idx in sorted(candidate):
                rec_parts.append(f"rec[-{n_meas - idx}]")
            flow_str = "1 -> " + " xor ".join(rec_parts)
            try:
                if bare.has_flow(stim.Flow(flow_str), unsigned=True):
                    mapped = [(meas_table[i][1], meas_table[i][2]) for i in candidate]
                    print(f"  FOUND data_block X boundary: Q{q_anc} 2nd-last + data combo")
                    print(f"    Measurements: {mapped}")
            except:
                pass

# Test bell_a Z boundary 
bell_a_z_anc = {23, 24, 25}  # Z ancillas for bell_a
for q in sorted(bell_a_z_anc):
    indices = [i for i, (_, n, q2) in enumerate(meas_table) if q2 == q]
    print(f"  bell_a Z anc Q{q}: {len(indices)} measurements, indices={indices}")

q23_meas = [i for i, (_, n, q) in enumerate(meas_table) if q == 23]
q24_meas = [i for i, (_, n, q) in enumerate(meas_table) if q == 24]
q25_meas = [i for i, (_, n, q) in enumerate(meas_table) if q == 25]

bell_a_m = sorted(bell_a_data_indices)

for q_anc, anc_meas_list in [(23, q23_meas), (24, q24_meas), (25, q25_meas)]:
    if len(anc_meas_list) < 2:
        continue
    second_last = anc_meas_list[-2]
    
    for w in range(2, 5):
        for combo in itertools.combinations(bell_a_m, w):
            candidate = [second_last] + list(combo)
            rec_parts = []
            for idx in sorted(candidate):
                rec_parts.append(f"rec[-{n_meas - idx}]")
            flow_str = "1 -> " + " xor ".join(rec_parts)
            try:
                if bare.has_flow(stim.Flow(flow_str), unsigned=True):
                    mapped = [(meas_table[i][1], meas_table[i][2]) for i in candidate]
                    print(f"  FOUND bell_a Z boundary: Q{q_anc} 2nd-last + bell_a combo")
                    print(f"    Measurements: {mapped}")
            except:
                pass

# Also try cross-block: bell_a anc + bell_a data + data_block data
print("\n=== Cross-block boundary tests (bell_a anc + bell_a data + data data) ===")
for q_anc, anc_meas_list in [(23, q23_meas), (24, q24_meas), (25, q25_meas)]:
    if len(anc_meas_list) < 2:
        continue
    second_last = anc_meas_list[-2]
    
    # Try: anc + 1 bell_a data + 1 data_block MX
    for ba in bell_a_m:
        for dm in data_mx:
            candidate = [second_last, ba, dm]
            rec_parts = []
            for idx in sorted(candidate):
                rec_parts.append(f"rec[-{n_meas - idx}]")
            flow_str = "1 -> " + " xor ".join(rec_parts)
            try:
                if bare.has_flow(stim.Flow(flow_str), unsigned=True):
                    mapped = [(meas_table[i][1], meas_table[i][2]) for i in candidate]
                    print(f"  FOUND cross-block: Q{q_anc} 2nd-last + 1 bell_a + 1 data")
                    print(f"    Measurements: {mapped}")
            except:
                pass

    # Try: anc + 2 bell_a data + 2 data_block MX  
    for ba_combo in itertools.combinations(bell_a_m, 2):
        for dm_combo in itertools.combinations(data_mx, 2):
            candidate = [second_last] + list(ba_combo) + list(dm_combo)
            rec_parts = []
            for idx in sorted(candidate):
                rec_parts.append(f"rec[-{n_meas - idx}]")
            flow_str = "1 -> " + " xor ".join(rec_parts)
            try:
                if bare.has_flow(stim.Flow(flow_str), unsigned=True):
                    mapped = [(meas_table[i][1], meas_table[i][2]) for i in candidate]
                    print(f"  FOUND cross-block: Q{q_anc} 2nd-last + 2 bell_a + 2 data")
                    print(f"    Measurements: {mapped}")
            except:
                pass

print("\nDone.")
