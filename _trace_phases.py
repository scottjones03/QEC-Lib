#!/usr/bin/env python
"""Trace phase routing for stabilizer_round_post to identify missing pairs."""
import os
import sys
os.environ.setdefault("WISE_INPROCESS_LIMIT", "999999999")
sys.path.insert(0, 'src')

from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
    decompose_into_phases,
    partition_grid_for_blocks,
)

print("=== Phase Pairs Trace ===")

# Setup
d2_code = RotatedSurfaceCode(distance=2)
gadget = CSSSurgeryCNOTGadget()
exp = FaultTolerantGadgetExperiment(
    codes=[d2_code],
    gadget=gadget,
    noise_model=None,
    num_rounds_before=2,
    num_rounds_after=2,
)
circuit = exp.to_stim()
meta = exp.qec_metadata
alloc = exp._unified_allocation

print("=== QEC Metadata Phases ===")
for i, phase in enumerate(meta.phases):
    active = phase.active_blocks if phase.active_blocks is not None else "ALL"
    print(f"Phase {i}: type={phase.phase_type}, active_blocks={active}, num_rounds={phase.num_rounds}")

# Build qubit_to_ion mapping
q2i = {}
for ba in meta.block_allocations:
    for q in (list(ba.data_qubits) + list(ba.x_ancilla_qubits) + list(ba.z_ancilla_qubits)):
        q2i[q] = q + 1  # Simple 1:1 mapping offset
if hasattr(alloc, "bridge_ancillas") and alloc.bridge_ancillas:
    for gi, _coord, _purpose in alloc.bridge_ancillas:
        if gi not in q2i:
            q2i[gi] = gi + 1

# Get sub_grids
sub_grids = partition_grid_for_blocks(meta, alloc, k=2)
print(f"\nsub_grids blocks: {list(sub_grids.keys())}")
for bname, sg in sub_grids.items():
    print(f"  {bname}: ion_indices={sg.ion_indices}, grid_region={sg.grid_region}")

# Decompose into phases
plans = decompose_into_phases(meta, gadget, alloc, sub_grids, q2i, k=2)
print(f"\nTotal plans: {len(plans)}")

# Analyze each phase
total_pairs = 0
for i, plan in enumerate(plans):
    n_rounds = len(plan.ms_pairs_per_round) if plan.ms_pairs_per_round else 0
    pairs_in_phase = 0
    if plan.ms_pairs_per_round:
        for round_pairs in plan.ms_pairs_per_round:
            pairs_in_phase += len(round_pairs)
    
    print(f"\nPlan {i}: type={plan.phase_type}, phase_index={plan.phase_index}")
    print(f"  interacting_blocks: {plan.interacting_blocks}")
    print(f"  ms_pairs_per_round length: {n_rounds}")
    print(f"  Total pairs in plan: {pairs_in_phase}")
    
    if plan.ms_pairs_per_round and n_rounds > 0:
        # Show first few rounds of pairs
        for ri, rp in enumerate(plan.ms_pairs_per_round[:3]):
            print(f"    Round {ri}: {len(rp)} pairs - sample: {rp[:3]}")
    
    total_pairs += pairs_in_phase

print(f"\n=== SUMMARY ===")
print(f"Total pairs across all plans: {total_pairs}")

# Count CX pairs in stim
def count_cx_pairs_detailed(circ):
    result = []
    for i, inst in enumerate(circ.flattened()):
        if inst.name in ("CX", "CZ", "XCZ", "ZCX", "ZCZ"):
            targets = inst.targets_copy()
            pairs = []
            for j in range(0, len(targets), 2):
                pairs.append((targets[j].value, targets[j+1].value))
            result.append((i, inst.name, pairs))
    return result

cx_detail = count_cx_pairs_detailed(circuit)
cx_pairs_flat = []
for idx, name, pairs in cx_detail:
    cx_pairs_flat.extend(pairs)
    
print(f"\n=== STIM CX INSTRUCTIONS ===")
print(f"Total CX instructions: {len(cx_detail)}")
print(f"Total CX pairs: {len(cx_pairs_flat)}")

# Group pairs by which block the qubits belong to
# block_0: qubits 0-6, 21
# block_1: qubits 7-13, 22, 23
# block_2: qubits 14-20

def get_block(q):
    if q in [0,1,2,3,4,5,6]:
        return 'block_0'
    elif q == 21:
        return 'bridge_Z'  # Z-bridge ancilla
    elif q in [7,8,9,10,11,12,13]:
        return 'block_1'
    elif q in [22, 23]:
        return 'bridge_X'  # X-bridge ancillas
    elif q in [14,15,16,17,18,19,20]:
        return 'block_2'
    return 'unknown'

# Collect all pairs from stim WITH COUNT
from collections import Counter
stim_pair_counts = Counter()
for idx, name, pairs in cx_detail:
    for p in pairs:
        stim_pair_counts[tuple(sorted(p))] += 1

# Collect all pairs from plans WITH COUNT
plan_pair_counts = Counter()
for plan in plans:
    if plan.ms_pairs_per_round:
        for round_pairs in plan.ms_pairs_per_round:
            for p in round_pairs:
                q0, q1 = p[0] - 1, p[1] - 1
                plan_pair_counts[tuple(sorted((q0, q1)))] += 1

# Find pairs with mismatched counts
print(f"\n=== PAIR COUNT ANALYSIS ===")
print(f"Stim unique pairs: {len(stim_pair_counts)}")
print(f"Plan unique pairs: {len(plan_pair_counts)}")
print(f"Stim total pair occurrences: {sum(stim_pair_counts.values())}")
print(f"Plan total pair occurrences: {sum(plan_pair_counts.values())}")

# Check for mismatches
mismatched = []
for pair, stim_count in stim_pair_counts.items():
    plan_count = plan_pair_counts.get(pair, 0)
    if stim_count != plan_count:
        mismatched.append((pair, stim_count, plan_count))

print(f"\nPairs with count mismatch: {len(mismatched)}")
if mismatched:
    by_block_pair = {}
    for pair, sc, pc in mismatched:
        b0, b1 = get_block(pair[0]), get_block(pair[1])
        key = tuple(sorted([b0, b1]))
        if key not in by_block_pair:
            by_block_pair[key] = []
        by_block_pair[key].append((pair, sc, pc))
    
    for key, items in by_block_pair.items():
        print(f"\n  {key}:")
        for pair, sc, pc in items[:5]:
            print(f"    {pair}: stim={sc}, plan={pc}, diff={sc-pc}")

expected = len(cx_pairs_flat)
print(f"\nExpected CX pairs from stim: {expected}")
if total_pairs == expected:
    print(f"✅ Plans pairs match expected!")
else:
    print(f"❌ MISMATCH: plans have {total_pairs}, expected {expected}")
    print(f"   Missing at decompose_into_phases level: {expected - total_pairs}")
