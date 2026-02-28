#!/usr/bin/env python
"""Trace actual routing to see where pairs are lost."""
import os
import sys
os.environ.setdefault("WISE_INPROCESS_LIMIT", "999999999")
sys.path.insert(0, 'src')

import numpy as np
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
    decompose_into_phases,
    partition_grid_for_blocks,
    route_full_experiment_as_steps,
)
from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
    QCCDWiseArch,
)

print("=== Routing Pairs Trace ===")

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

# Build qubit_to_ion mapping
q2i = {}
for ba in meta.block_allocations:
    for q in (list(ba.data_qubits) + list(ba.x_ancilla_qubits) + list(ba.z_ancilla_qubits)):
        q2i[q] = q + 1
if hasattr(alloc, "bridge_ancillas") and alloc.bridge_ancillas:
    for gi, _coord, _purpose in alloc.bridge_ancillas:
        if gi not in q2i:
            q2i[gi] = gi + 1

# Get sub_grids
sub_grids = partition_grid_for_blocks(meta, alloc, k=2)

# Decompose into phases
plans = decompose_into_phases(meta, gadget, alloc, sub_grids, q2i, k=2)

# Report plans
print("\n=== Plans from decompose_into_phases ===")
total_plan_pairs = 0
for i, plan in enumerate(plans):
    n_rounds = len(plan.ms_pairs_per_round) if plan.ms_pairs_per_round else 0
    n_pairs = 0
    if plan.ms_pairs_per_round:
        for rp in plan.ms_pairs_per_round:
            n_pairs += len(rp)
    print(f"Plan {i}: type={plan.phase_type}, phase_index={plan.phase_index}, rounds={n_rounds}, pairs={n_pairs}")
    total_plan_pairs += n_pairs

print(f"\nTotal pairs from plans: {total_plan_pairs}")

# Now route
print("\n=== Running route_full_experiment_as_steps ===")
print("This may take a while...")

# Architecture parameters
layout_shape = (6, 8 * 2)  # m=8, n=6, k=2 for CSS surgery d=2
wiseArch = QCCDWiseArch(m=8, n=6, k=2)
initial_layout = np.full((6, 16), -1, dtype=np.int32)

# Initialize layout with ions from block allocations
for ba in meta.block_allocations:
    all_qubits = list(ba.data_qubits) + list(ba.x_ancilla_qubits) + list(ba.z_ancilla_qubits)
    for q in all_qubits:
        ion_idx = q2i.get(q, q + 1)
        # Simple placement for now 
        row = (ion_idx - 1) // 8
        col = ((ion_idx - 1) % 8) * 2
        if row < 6 and col < 16:
            initial_layout[row, col] = ion_idx

# Also place bridge ancillas
if hasattr(alloc, "bridge_ancillas") and alloc.bridge_ancillas:
    for gi, _coord, _purpose in alloc.bridge_ancillas:
        ion_idx = q2i.get(gi, gi + 1)
        row = (ion_idx - 1) // 8
        col = ((ion_idx - 1) % 8) * 2
        if row < 6 and col < 16:
            initial_layout[row, col] = ion_idx

# Derive active_ions
active_ions = list(set(q2i.values()))

# Build ion_to_block mapping
ion_to_block = {}
for bname, sg in sub_grids.items():
    for ion_idx in sg.ion_indices:
        ion_to_block[ion_idx] = bname

print(f"initial_layout shape: {initial_layout.shape}")
print(f"active_ions: {len(active_ions)}")
print(f"sub_grids: {list(sub_grids.keys())}")

# Call route_full_experiment_as_steps
try:
    all_routing_steps, final_layout = route_full_experiment_as_steps(
        plans=plans,
        initial_layout=initial_layout,
        wiseArch=wiseArch,
        block_sub_grids=sub_grids,
        subgridsize=2,
        base_pmax_in=1,
        lookahead=2,
        max_inner_workers=1,
        stop_event=None,
        cx_per_ec_round=None,
        cache_ec_rounds=True,
        progress_callback=None,
    )
    
    print(f"\nRouting complete: {len(all_routing_steps)} steps")
    
    # Count solved pairs
    total_solved = 0
    for step in all_routing_steps:
        total_solved += len(step.solved_pairs)
    
    print(f"Total solved pairs: {total_solved}")
    print(f"Expected: {total_plan_pairs}")
    
    if total_solved != total_plan_pairs:
        print(f"❌ MISMATCH: {total_solved} vs {total_plan_pairs}")
        print(f"   Missing: {total_plan_pairs - total_solved}")
        
        # Group solved pairs by ms_round_index
        round_counts = {}
        for step in all_routing_steps:
            ri = step.ms_round_index
            if ri not in round_counts:
                round_counts[ri] = 0
            round_counts[ri] += len(step.solved_pairs)
        
        print(f"\nPairs per ms_round_index (first 20):")
        for ri in sorted(round_counts.keys())[:20]:
            print(f"  ms_round_index={ri}: {round_counts[ri]} pairs")
    else:
        print("✅ SUCCESS: All pairs routed")

except Exception as e:
    import traceback
    print(f"\nError during routing: {e}")
    traceback.print_exc()
