#!/usr/bin/env python3
"""Check which detectors are new in flat KnillEC 0 after our changes.
Are detectors 57-62 the new bell_a boundary detectors?"""
import stim
from qectostim.codes import SteaneCode713
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()
noise = StimStyleDepolarizingNoise(p=0.001)
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

# Map blocks
block_map = {}
for q in range(7): block_map[q] = "data"
for q in range(7,13): block_map[q] = "data_anc"
for q in range(13,20): block_map[q] = "bell_a"
for q in range(20,26): block_map[q] = "bell_a_anc"
for q in range(26,33): block_map[q] = "bell_b"
for q in range(33,39): block_map[q] = "bell_b_anc"

# Print all detectors  
det_idx = 0
for inst in circ.flattened():
    if inst.name == "DETECTOR":
        rec_targets = [t for t in inst.targets_copy() if t.is_measurement_record_target]
        meas_indices = sorted([n_meas + t.value for t in rec_targets])
        mapped = []
        for m in meas_indices:
            q = meas_table[m][2]
            mapped.append(f"m[{m}]={meas_table[m][1]}(Q{q}/{block_map.get(q,'?')})")
        print(f"D{det_idx}: {' ⊕ '.join(mapped)}")
        det_idx += 1

# Check: what are the last 3 detectors?
print(f"\nTotal: {det_idx} detectors")

# Also: what's the detector count WITHOUT the threshold change?
# The threshold change affects _discover_boundary_detectors
# Bell_a ancilla (Q23,Q24,Q25) has indices with len >= 2 but < 4
# Let's check their measurement counts
print("\n=== Bell_a ancilla measurement counts ===")
for q in [20, 21, 22, 23, 24, 25]:
    indices = [i for i, (_, n, q2) in enumerate(meas_table) if q2 == q]
    role = "X_anc" if q < 23 else "Z_anc"
    print(f"  Q{q} ({role}): {len(indices)} measurements at {indices}")
