#!/usr/bin/env python3
"""Check if adding the missing temporal detector m[30]⊕m[48] would fix the issue.
Also verify the gap in the temporal detector chain for Q36."""
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

# Show ALL temporal detectors for Q36 and Q37
print("=== Temporal detector chain for Q36 (Z ancilla) ===")
q36_meas = [i for i, (_, n, q) in enumerate(meas_table) if q == 36]
q37_meas = [i for i, (_, n, q) in enumerate(meas_table) if q == 37]
print(f"Q36 measurements: {q36_meas}")
print(f"Q37 measurements: {q37_meas}")

det_idx = 0
for inst in circ.flattened():
    if inst.name == "DETECTOR":
        rec_targets = [t for t in inst.targets_copy() if t.is_measurement_record_target]
        meas_indices = sorted([n_meas + t.value for t in rec_targets])
        
        q36_in = [m for m in meas_indices if m in set(q36_meas)]
        q37_in = [m for m in meas_indices if m in set(q37_meas)]
        
        if q36_in or q37_in:
            mapped = [f"m[{m}]=Q{meas_table[m][2]}" for m in meas_indices]
            print(f"  D{det_idx}: {' ⊕ '.join(mapped)}")
        det_idx += 1

# Check if m[30]⊕m[48] is a valid flow
print("\n=== Testing missing temporal detector: m[30] ⊕ m[48] (Q36 round1 vs round2) ===")

# Strip noise and detectors
bare = stim.Circuit()
for inst in circ.flattened():
    if inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE") and "DEPOLARIZE" not in inst.name and "NOISE" not in inst.name:
        bare.append(inst)

# Test if m[30] ⊕ m[48] is a valid Stim flow
flow_str = f"1 -> rec[-{n_meas - 30}] xor rec[-{n_meas - 48}]"
print(f"Flow: {flow_str}")
try:
    result = bare.has_flow(stim.Flow(flow_str), unsigned=True)
    print(f"  Valid flow (unsigned): {result}")
except Exception as e:
    print(f"  Error: {e}")

# Also test m[31]⊕m[49] for Q37
flow_str2 = f"1 -> rec[-{n_meas - 31}] xor rec[-{n_meas - 49}]"
print(f"Flow: {flow_str2}")
try:
    result2 = bare.has_flow(stim.Flow(flow_str2), unsigned=True)
    print(f"  Valid flow (unsigned): {result2}")
except Exception as e:
    print(f"  Error: {e}")

# Also test m[32]⊕m[50] for Q38
flow_str3 = f"1 -> rec[-{n_meas - 32}] xor rec[-{n_meas - 50}]"
print(f"Flow: {flow_str3}")
try:
    result3 = bare.has_flow(stim.Flow(flow_str3), unsigned=True)
    print(f"  Valid flow (unsigned): {result3}")
except Exception as e:
    print(f"  Error: {e}")

# Now: try adding these detectors and see if it eliminates the undetectable error
print("\n=== Testing circuit with added temporal detectors ===")
# Build a new circuit with the missing temporal detectors added
augmented = stim.Circuit()
for inst in circ.flattened():
    augmented.append(inst)

# Add detectors: m[30]⊕m[48], m[31]⊕m[49], m[32]⊕m[50]
for m1, m2 in [(30, 48), (31, 49), (32, 50)]:
    augmented.append(stim.CircuitInstruction(
        "DETECTOR", 
        [stim.target_rec(m1 - n_meas), stim.target_rec(m2 - n_meas)],
        []
    ))

print(f"Original detectors: {circ.num_detectors}")
print(f"Augmented detectors: {augmented.num_detectors}")

# Check DEM
try:
    dem_aug = augmented.detector_error_model(decompose_errors=False)
    
    # Count undetectable
    undet = 0
    for error in dem_aug:
        if error.type != "error":
            continue
        targets = error.targets_copy()
        dets = [t for t in targets if t.is_relative_detector_id()]
        obs = [t for t in targets if t.is_logical_observable_id()]
        if len(dets) == 0 and len(obs) > 0:
            undet += 1
            obs_ids = [t.val for t in obs]
            print(f"  Still undetectable: prob={error.args_copy()[0]:.3e}, L={obs_ids}")
    
    print(f"\nUndetectable errors: {undet}")
    if undet == 0:
        print("SUCCESS: All errors now detectable!")
        
        # Check shortest graphlike error path
        shortest = min(
            (sum(1 for t in e.targets_copy() if t.is_relative_detector_id()) 
             for e in dem_aug if e.type == "error" and any(t.is_logical_observable_id() for t in e.targets_copy())),
            default=999
        )
        print(f"Shortest error path to logical: {shortest} detectors")
except Exception as e:
    print(f"DEM generation failed: {e}")

print("\nDone.")
