#!/usr/bin/env python3
"""Check if using LAST ancilla measurement for boundary works for data_block."""
import stim
import itertools
from qectostim.codes import SteaneCode713
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment

code = SteaneCode713()
gadget = KnillECGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget, noise_model=None,
    num_rounds_before=3, num_rounds_after=3,
)
base = exp.to_stim()

# Strip
bare = stim.Circuit()
for inst in base.flattened():
    if inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
        bare.append(inst)

# Build measurement table
meas_table = []
for inst in bare.flattened():
    if inst.name in ("M", "MX", "MR", "MRX", "MY"):
        for t in inst.targets_copy():
            meas_table.append((len(meas_table), inst.name, t.value))

n_meas = len(meas_table)
print(f"Total measurements: {n_meas}")

# data_block X ancillas: Q7, Q8, Q9
# Their measurements:
for q in [7, 8, 9]:
    indices = [i for i, (_, n, q2) in enumerate(meas_table) if q2 == q]
    print(f"Q{q} (data X anc): measurements at {indices}")

data_mx = sorted(i for i, (_, n, q) in enumerate(meas_table) if q in {0,1,2,3,4,5,6} and n == "MX")
print(f"data_block MX: {data_mx}")

# Test LAST vs SECOND-TO-LAST for each data X ancilla
for q in [7, 8, 9]:
    indices = [i for i, (_, n, q2) in enumerate(meas_table) if q2 == q]
    last = indices[-1]
    second_last = indices[-2]
    
    print(f"\n=== Q{q}: last={last}, 2nd-last={second_last} ===")
    
    for label, anc_idx in [("LAST", last), ("2ND-LAST", second_last)]:
        for w in range(2, 6):
            for combo in itertools.combinations(data_mx, w):
                candidate = [anc_idx] + list(combo)
                rec_parts = [f"rec[-{n_meas - idx}]" for idx in sorted(candidate)]
                flow_str = "1 -> " + " xor ".join(rec_parts)
                try:
                    if bare.has_flow(stim.Flow(flow_str), unsigned=True):
                        mapped = [(meas_table[i][1], meas_table[i][2]) for i in candidate]
                        print(f"  {label} w={w}: FLOW! {mapped}")
                except:
                    pass

# Also test with data Z ancillas (Q10, Q11, Q12)
print("\n=== Z ancilla tests ===")
for q in [10, 11, 12]:
    indices = [i for i, (_, n, q2) in enumerate(meas_table) if q2 == q]
    last = indices[-1]
    
    for w in range(2, 6):
        for combo in itertools.combinations(data_mx, w):
            candidate = [last] + list(combo)
            rec_parts = [f"rec[-{n_meas - idx}]" for idx in sorted(candidate)]
            flow_str = "1 -> " + " xor ".join(rec_parts)
            try:
                if bare.has_flow(stim.Flow(flow_str), unsigned=True):
                    mapped = [(meas_table[i][1], meas_table[i][2]) for i in candidate]
                    print(f"  Q{q} LAST w={w}: FLOW! {mapped}")
            except:
                pass

# What about combining data ancilla LAST with data MX AND bell_a M?
bell_a_m = sorted(i for i, (_, n, q) in enumerate(meas_table) if q in {13,14,15,16,17,18,19} and n == "M")
print(f"\nbell_a M: {bell_a_m}")

print("\n=== Cross-block: data X anc + data MX + bell_a M ===")
for q in [7, 8, 9]:
    indices = [i for i, (_, n, q2) in enumerate(meas_table) if q2 == q]
    last = indices[-1]
    
    for dm_w in range(1, 4):
        for ba_w in range(1, 4):
            for dm_combo in itertools.combinations(data_mx, dm_w):
                for ba_combo in itertools.combinations(bell_a_m, ba_w):
                    candidate = [last] + list(dm_combo) + list(ba_combo)
                    if len(candidate) > 6:
                        continue
                    rec_parts = [f"rec[-{n_meas - idx}]" for idx in sorted(candidate)]
                    flow_str = "1 -> " + " xor ".join(rec_parts)
                    try:
                        if bare.has_flow(stim.Flow(flow_str), unsigned=True):
                            mapped = [(meas_table[i][1], meas_table[i][2]) for i in candidate]
                            print(f"  Q{q} LAST + {dm_w}data + {ba_w}bell_a: FLOW!")
                            print(f"    {mapped}")
                    except:
                        pass

print("\nDone.")
