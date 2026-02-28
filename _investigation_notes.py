#!/usr/bin/env python3
"""
EC ROUND MEASUREMENT INVESTIGATION NOTES
=========================================

Key Files:
- src/qectostim/gadgets/css_surgery_cnot.py - CSS surgery CNOT gadget (5 phases)
- src/qectostim/experiments/stabilizer_rounds/css.py - CSS builder, emit_round, emit_ancilla_measure_and_detectors
- src/qectostim/experiments/hardware_simulation/trapped_ion/utils/trapped_ion_compiler.py - decompose_to_native (line 157)
- src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py - _drain_single_qubit_ops (line 2000), execution loop (line 2258)
- src/qectostim/experiments/phase_orchestrator.py - PhaseOrchestrator.execute_phases
- src/qectostim/experiments/hardware_simulation/core/pipeline.py - QECMetadata, PhaseInfo
- src/qectostim/experiments/ft_gadget_experiment.py - ms_pair_count patching (line 1754)

FINDING 1: Stim Circuit Structure IS Correct
---------------------------------------------
The CSS surgery CNOT gadget emits EC rounds inline during merge phases (ZZ merge, XX merge).
Each merge round (d rounds total) has:
  1. R(bridge) → TICK
  2. CX(ctrl→bridge) → TICK  
  3. CX(anc→bridge) → TICK
  4. M(bridge) → TICK
  5. EC round via _emit_parallel_ec_round:
     a. R/RX(all block ancillas) → TICK
     b. CX phases (interleaved X+Z for all blocks) → TICK between phases
     c. MR/MRX(all block ancillas) + DETECTORS → TICK

So measurements ARE emitted between EC rounds in the stim circuit.

FINDING 2: decompose_to_native Handles MR/MRX Correctly
---------------------------------------------------------
Step 3 of decompose_to_native calls stim.Circuit(gate).decomposed() per-gate:
  MR  → M + R     (verified)
  MRX → H + M + R + H  (verified)
  RX  → R + H     (verified)
  MX  → H + M + H (verified)

The decomposition happens BEFORE the first-pass instr[0] check, so the
truncation issue (instr[0] in ("R","H","M") stripping multi-char names)
does NOT affect compound gates — they're already decomposed to M, R, H.

FINDING 3: _drain_single_qubit_ops Epoch Mechanism
----------------------------------------------------
Located at qccd_WISE_ion_route.py:2000.
Constraints:
  C1: Epoch ceiling = min epoch of all remaining 2-qubit (MS) gates
  C2: Blocked-ions scan — cumulative: once an MS gate is seen, its ions are blocked

For CSS surgery, between bridge CX MS and EC CX MS:
- min_ms_epoch = EC CX epoch (e.g., 5)
- Bridge M at epoch 3 ≤ 5 → eligible
- EC R/RX at epoch 4 ≤ 5 → eligible 
- EC ancilla M at epoch 7 > min_ms_epoch → NOT eligible (deferred to after EC CX)

Between EC CX MS and next bridge CX MS:
- min_ms_epoch = next bridge epoch (e.g., 9)
- EC M at epoch 7 ≤ 9 → eligible ✓
- Bridge R at epoch 8 ≤ 9 → eligible ✓

So measurements SHOULD be drained correctly by the epoch mechanism.

FINDING 4: CSS Builder's emit_ancilla_measure_and_detectors Emits MR/MRX
--------------------------------------------------------------------------
css.py:723 - emit_ancilla_measure_and_detectors:
  Normal mode:  X ancillas → MRX, Z ancillas → MR
  Swapped mode: X ancillas → MR,  Z ancillas → MRX

This correctly measures and resets ancillas in one instruction.

FINDING 5: Phase Pair Count Patching
--------------------------------------
ft_gadget_experiment.py:1754 - After circuit is built, QECMetadata.phases
get patched with actual CX counts from _count_cx_instructions. CSS surgery
merge phases include BOTH bridge CX AND EC CX in their ms_pair_count.

REMAINING INVESTIGATION:
- Check if _emit_parallel_ec_round fallback path (sequential) produces different structure
- Verify _emit_interleaved_round in css.py produces proper M/MR within its TICK structure
- Check if hierarchical_concatenated emit_inner_only_round skips measurements
- Check blocked_ions edge case: do EC ancilla ions appear in LATER MS gates before their M?
"""
