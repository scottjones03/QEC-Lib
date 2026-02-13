#!/usr/bin/env python3
"""Track FULL error propagation of X on Q29 and X on Q36 through all subsequent gates."""
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

# Track X error on a set of qubits through the circuit from tick 117 onward
# Starting state: X on Q29, X on Q36
x_errors = {29, 36}  # qubits with X error

tick = 0
meas_idx = 0
meas_flipped = []
past_start = False

for inst in circ.flattened():
    if inst.name == "TICK":
        tick += 1
        if tick == 118:  # error happens after tick 117 DEPOLARIZE2
            past_start = True
            print(f"Starting propagation at tick {tick}, X errors on: {sorted(x_errors)}")
        continue
    
    if not past_start:
        # Still need to count measurements
        if inst.name in ("M", "MX", "MR", "MRX"):
            for t in inst.targets_copy():
                meas_idx += 1
        continue
    
    if inst.name == "CX":
        targets = inst.targets_copy()
        qs = [t.value for t in targets]
        pairs = [(qs[i], qs[i+1]) for i in range(0, len(qs), 2)]
        
        changed = False
        for ctrl, tgt in pairs:
            # X on control → X on control AND X on target
            if ctrl in x_errors:
                if tgt not in x_errors:
                    x_errors.add(tgt)
                    changed = True
                    print(f"  tick={tick}: CX Q{ctrl}→Q{tgt}: X spreads to Q{tgt}")
                else:
                    # X on both: X_ctrl ⊗ X_tgt after CX = X_ctrl propagation cancels
                    x_errors.discard(tgt)
                    changed = True
                    print(f"  tick={tick}: CX Q{ctrl}→Q{tgt}: X cancels on Q{tgt}")
            # X on target only: X stays on target (no propagation)
            # Note: X on target of CX stays as X on target
        
        if changed:
            print(f"    X errors now: {sorted(x_errors)}")
    
    elif inst.name == "H":
        # H converts X↔Z, but for tracking X propagation:
        # H X H = Z (so X error becomes Z error after H)
        targets = inst.targets_copy()
        for t in targets:
            if t.value in x_errors:
                print(f"  tick={tick}: H on Q{t.value}: X→Z (X error becomes Z error)")
                # For simplicity, mark as Z error. But we're tracking X only.
                # Actually, we need to track both X and Z errors.
                # Let me use a dict instead.
                pass
    
    elif inst.name in ("M", "MX", "MR", "MRX"):
        targets = inst.targets_copy()
        for t in targets:
            q = t.value
            if q in x_errors:
                if inst.name in ("M", "MR"):
                    # Z basis measurement. X anti-commutes with Z → flips
                    meas_flipped.append(meas_idx)
                    print(f"  tick={tick}: {inst.name} Q{q}: X flips Z measurement → meas[{meas_idx}] FLIPPED")
                elif inst.name in ("MX", "MRX"):
                    # X basis measurement. X commutes with X → no flip
                    print(f"  tick={tick}: {inst.name} Q{q}: X commutes with X measurement → no flip")
                
                if inst.name in ("MR", "MRX"):
                    # Reset clears the error
                    x_errors.discard(q)
                    print(f"    Q{q} reset, error cleared")
            meas_idx += 1

print(f"\n=== Summary ===")
print(f"Final X errors on: {sorted(x_errors)}")
print(f"Flipped measurements: {meas_flipped}")

# Check which detectors fire with these flipped measurements
meas_table = []
for inst in circ.flattened():
    if inst.name in ("M", "MX", "MR", "MRX", "MY"):
        for t in inst.targets_copy():
            meas_table.append((len(meas_table), inst.name, t.value))
n_meas = len(meas_table)

flipped_set = set(meas_flipped)

det_idx = 0
fired_dets = []
for inst in circ.flattened():
    if inst.name == "DETECTOR":
        rec_targets = [t for t in inst.targets_copy() if t.is_measurement_record_target]
        meas_indices = sorted([n_meas + t.value for t in rec_targets])
        
        # Count how many referenced measurements are flipped
        n_flipped = sum(1 for m in meas_indices if m in flipped_set)
        if n_flipped % 2 == 1:  # odd number of flips → detector fires
            fired_dets.append(det_idx)
            mapped = [f"m[{m}]={meas_table[m][1]}(Q{meas_table[m][2]})" for m in meas_indices]
            print(f"  D{det_idx} FIRES ({n_flipped} flipped): {' ⊕ '.join(mapped)}")
        det_idx += 1

# Check observable
for inst in circ.flattened():
    if inst.name == "OBSERVABLE_INCLUDE":
        rec_targets = [t for t in inst.targets_copy() if t.is_measurement_record_target]
        meas_indices = sorted([n_meas + t.value for t in rec_targets])
        n_flipped = sum(1 for m in meas_indices if m in flipped_set)
        obs_val = n_flipped % 2
        print(f"  L0 flipped: {bool(obs_val)} ({n_flipped} meas flipped)")

print(f"\nFired detectors: {fired_dets}")
print(f"Expected: 0 detectors, L0 flipped (to match DEM)")
