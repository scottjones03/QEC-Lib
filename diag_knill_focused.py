#!/usr/bin/env python3
"""Focused investigation: what circuit operations produce the undetectable error.

Key question: Which single-fault location(s) produce L0 with no detectors?
We need to understand the qubit roles to identify the bug.
"""
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

# Print qubit coordinates
print("=== Qubit Coordinates ===")
flat = list(base.flattened())
for instr in flat:
    if instr.name == "QUBIT_COORDS":
        tgts = instr.targets_copy()
        gate_args = instr.gate_args_copy()
        print(f"  Q{tgts[0].value}: coords=({', '.join(f'{a}' for a in gate_args)})")

# Print OBSERVABLE_INCLUDE and DETECTOR instructions  
print("\n=== OBSERVABLE_INCLUDE instructions ===")
total_meas = 0
for i, instr in enumerate(flat):
    if instr.name in ("M", "MX", "MR", "MRX", "MY"):
        tgts = instr.targets_copy()
        total_meas += len(tgts)
    elif instr.name == "OBSERVABLE_INCLUDE":
        tgts = instr.targets_copy()
        gate_args = instr.gate_args_copy()
        recs = [str(t) for t in tgts]
        print(f"  [{i}] OBS({int(gate_args[0])}) recs={recs} (total_meas_so_far={total_meas})")

print(f"\nTotal measurements in circuit: {total_meas}")

# Print the last section of the circuit around instruction 446
print("\n=== Circuit around instruction 446 (tick 130) ===")
tick_count = 0
for i, instr in enumerate(flat):
    if instr.name == "TICK":
        tick_count += 1
    if 125 <= tick_count <= 135:
        tgts = instr.targets_copy()
        gate_args = instr.gate_args_copy()
        tgt_str = ', '.join(str(t) for t in tgts)
        args_str = f"({', '.join(f'{a}' for a in gate_args)})" if gate_args else ""
        print(f"  [{i}] tick={tick_count} {instr.name}{args_str} {tgt_str}")

# Identify the 3 blocks by coordinate ranges
print("\n=== Block identification ===")
print("Steane code has 7 data qubits + ancillas per block")
print("data_block: qubits at low x coords")
print("bell_a: qubits at medium x coords (~8.5-10.5)")  
print("bell_b: qubits at high x coords (~17-19)")

# Now the KEY question: check what happens with a noiseless sim
# Inject a single X error on qubit 13 right before its measurement
# and see if any detector fires
print("\n=== Single-fault injection test ===")
# Build circuit with a single injected error
# First find the Bell measurement CNOT layer
tick_count = 0
bell_meas_tick = None
for i, instr in enumerate(flat):
    if instr.name == "TICK":
        tick_count += 1
    if tick_count == 130 and instr.name == "CNOT":
        tgts = instr.targets_copy()
        print(f"  tick=130 CNOT targets: {[str(t) for t in tgts]}")

# Count how many of each measurement type
meas_groups = []
for instr in flat:
    if instr.name in ("M", "MX", "MR", "MRX"):
        tgts = instr.targets_copy()
        meas_groups.append((instr.name, [t.value for t in tgts]))

print(f"\n=== Measurement groups ({len(meas_groups)} total) ===")
for i, (name, qubits) in enumerate(meas_groups):
    print(f"  group {i}: {name} on qubits {qubits}")

# The observable formula — which measurements does it reference?
print("\n=== Observable formula analysis ===")
total_meas = 0
obs_meas_indices = []
for instr in flat:
    if instr.name in ("M", "MX", "MR", "MRX", "MY"):
        tgts = instr.targets_copy()
        total_meas += len(tgts)
    elif instr.name == "OBSERVABLE_INCLUDE":
        tgts = instr.targets_copy()
        gate_args = instr.gate_args_copy()
        for t in tgts:
            # rec targets are negative offsets from current meas count
            rec_offset = t.value  # This is negative
            abs_idx = total_meas + rec_offset
            obs_meas_indices.append(abs_idx)
            print(f"  OBS includes measurement index {abs_idx} (rec[{rec_offset}], total_meas={total_meas})")

# Map measurement indices back to qubits
print("\n=== Mapping observable measurements to qubits ===")
meas_to_qubit = {}
meas_idx = 0
for instr in flat:
    if instr.name in ("M", "MX", "MR", "MRX", "MY"):
        tgts = instr.targets_copy()
        for t in tgts:
            meas_to_qubit[meas_idx] = (instr.name, t.value)
            meas_idx += 1

for obs_idx in obs_meas_indices:
    if obs_idx in meas_to_qubit:
        name, qubit = meas_to_qubit[obs_idx]
        print(f"  Observable references meas[{obs_idx}] = {name}(Q{qubit})")
    else:
        print(f"  Observable references meas[{obs_idx}] = NOT FOUND (out of range)")

# Also check the CNOT-HTEL for comparison
print("\n\n=== CNOT-HTEL Comparison ===")
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
gadget2 = CNOTHTeleportGadget(input_state="0")
exp2 = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gadget2, noise_model=None,
    num_rounds_before=3, num_rounds_after=3,
)
base2 = exp2.to_stim()
noise = StimStyleDepolarizingNoise(p=1e-8)
noisy2 = noise.apply(base2)
dem2 = noisy2.detector_error_model(decompose_errors=False)

undet2 = []
for instr in dem2.flattened():
    if instr.type != "error":
        continue
    targets = instr.targets_copy()
    prob = instr.args_copy()[0]
    has_det = any(t.is_relative_detector_id() for t in targets)
    has_obs = any(t.is_logical_observable_id() for t in targets)
    if has_obs and not has_det:
        undet2.append(prob)

print(f"CNOT-HTEL: {dem2.num_detectors} det, {dem2.num_errors} err, {len(undet2)} undetectable")

# Check CNOT-HTEL observable formula
flat2 = list(base2.flattened())
total_meas2 = 0
obs_refs2 = []
for instr in flat2:
    if instr.name in ("M", "MX", "MR", "MRX", "MY"):
        total_meas2 += len(instr.targets_copy())
    elif instr.name == "OBSERVABLE_INCLUDE":
        tgts = instr.targets_copy()
        gate_args = instr.gate_args_copy()
        for t in tgts:
            abs_idx = total_meas2 + t.value
            obs_refs2.append(abs_idx)

# Map meas indices to qubits for CNOT-HTEL
meas_to_qubit2 = {}
meas_idx2 = 0
for instr in flat2:
    if instr.name in ("M", "MX", "MR", "MRX", "MY"):
        tgts = instr.targets_copy()
        for t in tgts:
            meas_to_qubit2[meas_idx2] = (instr.name, t.value)
            meas_idx2 += 1

print(f"Observable references {len(obs_refs2)} measurements:")
for obs_idx in obs_refs2:
    if obs_idx in meas_to_qubit2:
        name, qubit = meas_to_qubit2[obs_idx]
        print(f"  meas[{obs_idx}] = {name}(Q{qubit})")

print("\nDone.")
