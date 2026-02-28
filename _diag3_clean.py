#!/usr/bin/env python3
"""
Clean diagnostic: capture CSS surgery compilation internals to a file.
Avoids SAT routing entirely — only inspects compiler + analytics.
"""
import sys, os
os.environ["WISE_INPROCESS_LIMIT"] = "999999999"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

OUT = os.path.join(os.path.dirname(__file__), "_diag3_output.txt")

def main():
    lines = []
    def p(s=""):
        lines.append(str(s))
        print(s)

    # ── imports ──
    from qectostim.experiments.hardware_simulation.trapped_ion.utils.trapped_ion_compiler import TrappedIonCompiler
    from qectostim.experiments.hardware_simulation.trapped_ion.architecture.wise_arch import WiseArch
    from qectostim.experiments.hardware_simulation.trapped_ion.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
    from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
        partition_grid_for_blocks, decompose_into_phases, map_qubits_per_block,
        PhaseRoutingPlan, BlockSubGrid,
    )
    import stim
    import numpy as np

    # ── build gadget ──
    gadget = CSSSurgeryCNOTGadget(code_distance=2, noise_model=None)
    ideal = gadget.ideal_gadget_circuit(include_noise=False)
    p(f"=== CSS Surgery d=2 Diagnostic ===")
    p(f"Stim circuit: {ideal.num_qubits} qubits, {ideal.num_ticks} ticks")

    # ── compiler setup ──  
    arch = WiseArch(code_distance=2)
    compiler = TrappedIonCompiler(arch)
    
    # Get allocation + metadata
    qec_meta = gadget.get_qec_metadata()
    qubit_alloc = gadget.get_qubit_allocation()
    
    p(f"\n--- Bridge ancillas ---")
    for gi, coord, purpose in qubit_alloc.bridge_ancillas:
        p(f"  qubit {gi}: coord={coord}, purpose={purpose}")
    
    p(f"\n--- Block offsets ---")
    for bn, off in qubit_alloc.block_offsets.items():
        p(f"  {bn}: offset={off}")
    
    p(f"\n--- QEC metadata phases ---")
    for i, phase in enumerate(qec_meta.phases):
        p(f"  Phase {i}: blocks={[b.block_name for b in phase.active_blocks]}, "
          f"ms_pair_count={phase.ms_pair_count}, "
          f"is_ec={phase.is_ec_round}")
    
    # ── partition grid ──
    sub_grids = partition_grid_for_blocks(arch, qubit_alloc)
    p(f"\n--- Sub-grids ---")
    for name, sg in sub_grids.items():
        p(f"  {name}: region={sg.grid_region}, ions={sorted(sg.ion_indices)}, "
          f"q2i={sg.qubit_to_ion}")
    
    # ── map qubits (hillClimb) ──
    map_qubits_per_block(arch, sub_grids, qubit_alloc)
    p(f"\n--- After hillClimb ---")
    for name, sg in sub_grids.items():
        p(f"  {name}: region={sg.grid_region}, ions={sorted(sg.ion_indices)}")
        if sg.initial_layout is not None:
            p(f"    layout shape={sg.initial_layout.shape}")
            p(f"    layout=\n{sg.initial_layout}")
    
    # ── decompose phases ──
    plans = decompose_into_phases(ideal, qec_meta, qubit_alloc, sub_grids)
    p(f"\n--- Phase routing plans ---")
    for i, plan in enumerate(plans):
        total_pairs = sum(len(r) for r in plan.ms_pairs_per_round)
        p(f"  Plan {i}: {len(plan.ms_pairs_per_round)} rounds, {total_pairs} pairs, "
          f"blocks={plan.active_blocks}")
        for j, rnd in enumerate(plan.ms_pairs_per_round):
            if rnd:
                p(f"    Round {j}: {rnd}")
    
    # ── compile to native (get Operations + Ions) ──
    p(f"\n--- Compiler decomposition ---")
    compiled = compiler.decompose_to_native(ideal, batch=True)
    p(f"  Native circuit: {compiled.num_qubits} qubits")
    
    # Count CX in stim
    cx_pairs_stim = []
    for inst in ideal.flattened():
        if inst.name == "CX":
            targets = inst.targets_copy()
            for k in range(0, len(targets), 2):
                q1, q2 = targets[k].value, targets[k+1].value
                cx_pairs_stim.append((q1, q2))
    p(f"  Stim CX pairs: {len(cx_pairs_stim)}")
    
    # Find bridge CX pairs
    bridge_qubits = {gi for gi, _, _ in qubit_alloc.bridge_ancillas}
    bridge_cx = [(a, b) for a, b in cx_pairs_stim if a in bridge_qubits or b in bridge_qubits]
    p(f"  Bridge CX pairs: {len(bridge_cx)}")
    for pair in bridge_cx[:20]:
        p(f"    {pair}")
    if len(bridge_cx) > 20:
        p(f"    ... and {len(bridge_cx) - 20} more")
    
    # ── Check Operations for bridge ions ──
    p(f"\n--- Compiler Operations (bridge ions) ---")
    ops = compiler._last_operations if hasattr(compiler, '_last_operations') else None
    if ops:
        bridge_ion_idxs = {gi + 1 for gi, _, _ in qubit_alloc.bridge_ancillas}
        p(f"  Bridge ion indices (q+1): {bridge_ion_idxs}")
        bridge_ops = []
        for op in ops:
            if hasattr(op, 'ions') and len(op.ions) == 2:
                ion_set = frozenset(ion.idx for ion in op.ions)
                if ion_set & bridge_ion_idxs:
                    bridge_ops.append(op)
        p(f"  Operations involving bridge ions: {len(bridge_ops)}")
        for op in bridge_ops[:10]:
            ion_idxs = tuple(ion.idx for ion in op.ions)
            ion_labels = tuple(ion.label for ion in op.ions)
            p(f"    {type(op).__name__}: ions={ion_idxs}, labels={ion_labels}")
    else:
        p("  (no _last_operations cached)")
    
    # ── Check ion.idx vs q+1 mapping ──
    p(f"\n--- Ion index verification ---")
    ions_seen = {}
    if ops:
        for op in ops:
            if hasattr(op, 'ions'):
                for ion in op.ions:
                    if ion.idx not in ions_seen:
                        ions_seen[ion.idx] = ion
        p(f"  Unique ions: {len(ions_seen)}")
        for idx in sorted(ions_seen.keys()):
            ion = ions_seen[idx]
            label = getattr(ion, 'label', '?')
            qubit_idx = getattr(ion, 'qubit_idx', '?')
            x = getattr(ion, 'x', '?')
            y = getattr(ion, 'y', '?')
            p(f"    ion.idx={idx}, label={label}, qubit_idx={qubit_idx}, x={x}, y={y}")
    
    # ── Check parallelPairs construction ──
    p(f"\n--- parallelPairs reconstruction ---")
    if ops:
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import ionRoutingGadgetArch
        # Just build parallelPairs without routing
        toMoveOps = [op for op in ops if hasattr(op, 'ions') and len(op.ions) == 2]
        p(f"  toMoveOps count: {len(toMoveOps)}")
        
        # Group into rounds like the compiler does
        phase_pair_counts = [phase.ms_pair_count for phase in qec_meta.phases]
        p(f"  phase_pair_counts: {phase_pair_counts}")
        p(f"  sum(phase_pair_counts): {sum(phase_pair_counts)}")
        p(f"  len(toMoveOps): {len(toMoveOps)}")
        
        # Build parallelPairs
        pPairs = {}
        for i, op in enumerate(toMoveOps):
            pPairs.setdefault(i, []).append(
                tuple(sorted([ion.idx for ion in op.ions]))
            )
        
        # Show first and last few
        p(f"  parallelPairs rounds: {len(pPairs)}")
        for idx in list(pPairs.keys())[:5]:
            p(f"    Round {idx}: {pPairs[idx]}")
        if len(pPairs) > 10:
            p(f"    ...")
            for idx in list(pPairs.keys())[-3:]:
                p(f"    Round {idx}: {pPairs[idx]}")
    
    # ── Check plan pairs vs compiler pairs ──
    p(f"\n--- Plan pairs vs Compiler pairs ---")
    all_plan_pairs = []
    for plan in plans:
        for rnd in plan.ms_pairs_per_round:
            all_plan_pairs.extend(rnd)
    p(f"  Total plan pairs: {len(all_plan_pairs)}")
    
    all_compiler_pairs = []
    if ops:
        for op in toMoveOps:
            pair = tuple(sorted([ion.idx for ion in op.ions]))
            all_compiler_pairs.append(pair)
    p(f"  Total compiler pairs: {len(all_compiler_pairs)}")
    
    # Check overlap
    plan_set = set(all_plan_pairs)
    compiler_set = set(all_compiler_pairs)
    p(f"  Unique plan pairs: {len(plan_set)}")
    p(f"  Unique compiler pairs: {len(compiler_set)}")
    p(f"  In plan but not compiler: {plan_set - compiler_set}")
    p(f"  In compiler but not plan: {compiler_set - plan_set}")
    
    # ── Write output ──
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    p(f"\nOutput written to {OUT}")

if __name__ == "__main__":
    main()
