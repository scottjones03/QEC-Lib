#!/usr/bin/env python3
"""Map every detector to its measurement references to find the gap.

For KnillEC, we need to know: does any detector reference the M(bell_a) 
measurements (group 19, meas indices 61-67)?

If not, that's the bug: errors that only affect M(bell_a) outcomes 
are invisible to all detectors but visible to the observable.
"""
import stim
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
flat = list(base.flattened())

# Build measurement index table
meas_table = []  # (abs_idx, instr_name, qubit)
meas_idx = 0
for instr in flat:
    if instr.name in ("M", "MX", "MR", "MRX", "MY"):
        tgts = instr.targets_copy()
        for t in tgts:
            meas_table.append((meas_idx, instr.name, t.value))
            meas_idx += 1

total_meas = len(meas_table)

print(f"=== All {total_meas} measurements ===")
for idx, name, qubit in meas_table:
    block = "data" if qubit <= 12 else ("bell_a" if qubit <= 25 else "bell_b")
    is_anc = qubit in {7,8,9,10,11,12, 20,21,22,23,24,25, 33,34,35,36,37,38}
    role = "ancilla" if is_anc else "data"
    print(f"  meas[{idx:2d}] = {name}(Q{qubit:2d}) [{block} {role}]")

# Observable measurements
print(f"\n=== Observable references ===")
running_meas = 0
obs_abs = []
for instr in flat:
    if instr.name in ("M", "MX", "MR", "MRX", "MY"):
        running_meas += len(instr.targets_copy())
    elif instr.name == "OBSERVABLE_INCLUDE":
        tgts = instr.targets_copy()
        for t in tgts:
            abs_idx = running_meas + t.value  # t.value is negative
            obs_abs.append(abs_idx)

for abs_idx in obs_abs:
    if 0 <= abs_idx < total_meas:
        _, name, qubit = meas_table[abs_idx]
        block = "data" if qubit <= 12 else ("bell_a" if qubit <= 25 else "bell_b")
        print(f"  OBS includes meas[{abs_idx}] = {name}(Q{qubit}) [{block}]")

# Check: which detectors reference bell_a data qubit measurements?
bell_a_data_meas = set()
for abs_idx in range(total_meas):
    _, name, qubit = meas_table[abs_idx]
    if qubit in {13, 14, 15, 16, 17, 18, 19} and name in ("M", "MX"):
        bell_a_data_meas.add(abs_idx)

print(f"\n=== bell_a data qubit measurements (destructive) ===")
for abs_idx in sorted(bell_a_data_meas):
    _, name, qubit = meas_table[abs_idx]
    print(f"  meas[{abs_idx}] = {name}(Q{qubit})")

print(f"\n=== Detectors referencing bell_a data measurements ===")
running_meas = 0
det_idx = 0
det_refs_bella = []
for instr in flat:
    if instr.name in ("M", "MX", "MR", "MRX", "MY"):
        running_meas += len(instr.targets_copy())
    elif instr.name == "DETECTOR":
        tgts = instr.targets_copy()
        abs_refs = []
        for t in tgts:
            abs_idx = running_meas + t.value
            abs_refs.append(abs_idx)
        
        refs_in_bella = [a for a in abs_refs if a in bell_a_data_meas]
        if refs_in_bella:
            det_refs_bella.append((det_idx, abs_refs, refs_in_bella))
        det_idx += 1

if det_refs_bella:
    for didx, all_refs, bella_refs in det_refs_bella:
        print(f"  D{didx}: refs={all_refs}, bell_a_data_refs={bella_refs}")
        for ref in all_refs:
            if 0 <= ref < total_meas:
                _, name, qubit = meas_table[ref]
                block = "data" if qubit <= 12 else ("bell_a" if qubit <= 25 else "bell_b")
                print(f"    meas[{ref}] = {name}(Q{qubit}) [{block}]")
else:
    print("  *** NO detector references bell_a data measurements! ***")
    print("  This is the ROOT CAUSE of the undetectable error!")

# Similarly check: MX(data_block) measurements
data_block_mx = set()
for abs_idx in range(total_meas):
    _, name, qubit = meas_table[abs_idx]
    if qubit in {0,1,2,3,4,5,6} and name == "MX":
        data_block_mx.add(abs_idx)

print(f"\n=== data_block MX measurements (destructive) ===")
for abs_idx in sorted(data_block_mx):
    _, name, qubit = meas_table[abs_idx]
    print(f"  meas[{abs_idx}] = {name}(Q{qubit})")

print(f"\n=== Detectors referencing data_block MX measurements ===")
running_meas = 0
det_idx = 0
det_refs_data = []
for instr in flat:
    if instr.name in ("M", "MX", "MR", "MRX", "MY"):
        running_meas += len(instr.targets_copy())
    elif instr.name == "DETECTOR":
        tgts = instr.targets_copy()
        abs_refs = []
        for t in tgts:
            abs_idx = running_meas + t.value
            abs_refs.append(abs_idx)
        
        refs_in_data = [a for a in abs_refs if a in data_block_mx]
        if refs_in_data:
            det_refs_data.append((det_idx, abs_refs, refs_in_data))
        det_idx += 1

if det_refs_data:
    for didx, all_refs, data_refs in det_refs_data:
        print(f"  D{didx}: refs={all_refs}, data_MX_refs={data_refs}")
else:
    print("  *** NO detector references data_block MX measurements! ***")

print("\nDone.")
