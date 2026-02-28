#!/usr/bin/env python3
"""
EC ROUND MEASUREMENT INVESTIGATION - FULL REPORT
=================================================

QUESTION: Do EC (stabilizer_round) phases include measurements (M, MR, MX, MRX)
between CX rounds in the CSS surgery CNOT gadget pipeline?

ANSWER: YES — the stim circuit is correct. But there may be an issue in the
trapped-ion execution pipeline's _drain_single_qubit_ops mechanism.

================================================================
1. STIM CIRCUIT STRUCTURE FOR EC ROUNDS
================================================================

The CSS surgery CNOT gadget (src/qectostim/gadgets/css_surgery_cnot.py) has 5 phases:
  Phase 0 (ZZ merge): d rounds of bridge CX + interleaved EC on all 3 blocks
  Phase 1 (ZZ split): Destructive M on bridge
  Phase 2 (XX merge): d rounds of bridge CX + interleaved EC on all 3 blocks  
  Phase 3 (XX split): Destructive MX on bridge
  Phase 4 (Anc MX): MX on ancilla data → destroy block_1

EC rounds are emitted INLINE during merge phases via _emit_parallel_ec_round()
(css_surgery_cnot.py:330-400), NOT as separate inter-phase rounds from the
orchestrator. All PhaseResults have needs_stabilizer_rounds=0.

Each merge round emits:
  1. R(bridge_qubits) → TICK
  2. CX(data→bridge) [ctrl block] → TICK  
  3. CX(data→bridge) [anc/tgt block] → TICK
  4. M(bridge_qubits) → TICK
  5. EC round (_emit_parallel_ec_round):
     a. R/RX for all block ancillas → TICK
     b. CX phases (interleaved X+Z across all blocks) → TICK × n_phases
     c. MR/MRX for all block ancillas + DETECTORS → TICK

Stim circuit structure CONFIRMED: measurements ARE emitted between each EC round.

Key code locations:
  - _emit_zz_merge: css_surgery_cnot.py:458-590 (ZZ merge loop)
  - _emit_xx_merge: css_surgery_cnot.py:632-748 (XX merge loop)
  - _emit_parallel_ec_round: css_surgery_cnot.py:330-400

================================================================
2. decompose_to_native() HANDLING OF MR/MRX
================================================================

File: src/qectostim/experiments/hardware_simulation/trapped_ion/utils/trapped_ion_compiler.py:157

Step 3 (line ~268): Per-gate decomposition using stim.Circuit(gate).decomposed()
  Verified: MR 5  → "M 5\nR 5"       (M then R, same epoch)
  Verified: MRX 5 → "H 5\nM 5\nR 5\nH 5"  (basis change, M, R, basis change)
  Verified: RX 5  → "R 5\nH 5"       (reset then Hadamard for |+⟩)
  
These decompositions happen BEFORE the first-pass instr[0] check (line ~381),
which only sees simple M, R, H instructions. So compound gates are correctly
preserved through decomposition.

Second pass (line ~400+) has explicit handlers for MR (line 453), MRX (line 494),
MX (line 477), RX (line 467) — but these are NOT needed because stim.decomposed()
already splits them. The handlers serve as a safety net for direct MR/MRX in
stim circuits that bypass step 3.

RESULT: ✓ decompose_to_native correctly produces Measurement + QubitReset for MR,
and basis-change + Measurement + QubitReset + basis-change for MRX.

================================================================
3. _drain_single_qubit_ops() ANALYSIS
================================================================

File: src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py:2000

Epoch-based drain mechanism:
  C1 (epoch ceiling): Cannot execute ops with epoch > min remaining MS epoch
  C2 (blocked ions): Cumulative scan — once an MS gate's ions are seen, all later
     single-qubit ops on those ions are blocked until that MS executes

For CSS surgery merge round with epochs:
  epoch 0: R(bridge)
  epoch 1: CX(ctrl→bridge) → MS gates  
  epoch 2: CX(anc→bridge) → MS gates
  epoch 3: M(bridge)
  epoch 4: R/RX(EC ancillas)
  epoch 5: CX(EC phase0) → MS gates
  epoch 6: CX(EC phase1) → MS gates
  epoch 7: MR/MRX(EC ancillas) [decomposed to H,M,R,H]
  epoch 8: R(next bridge)
  epoch 9: CX(next ctrl→bridge) → MS gates

Drain between ms_round for epoch-1 MS and epoch-2 MS:
  min_ms_epoch = 2
  Bridge M at epoch 3 > 2 → NOT eligible (correct: bridge CX not done yet)
  
Drain between ms_round for epoch-2 MS and epoch-5 MS:
  min_ms_epoch = 5
  Bridge M at epoch 3 ≤ 5 → eligible ✓ (bridge CX done)
  EC R/RX at epoch 4 ≤ 5 → eligible ✓

Drain between ms_round for epoch-6 MS and epoch-9 MS:
  min_ms_epoch = 9
  EC M at epoch 7 ≤ 9 → eligible ✓
  Bridge R at epoch 8 ≤ 9 → eligible ✓

POTENTIAL ISSUE: blocked_ions check
  The blocked_ions scan iterates ALL remaining ops. If an EC ancilla ion
  participates in a LATER MS gate that appears BEFORE the M(ec_anc) in the
  operationsLeft list, the measurement would be blocked.
  
  But EC ancilla M (epoch 7) comes BEFORE any future MS gate using that ancilla
  in the operationsLeft list (since ops are ordered by stim circuit order).
  Future EC CX MS gates (epoch ~13) come much later in the list.
  
  Bridge data ions appear in both bridge CX and EC CX, but M(ec_anc) operations
  use ANCILLA ions (not data ions), so they should not be blocked by bridge CX MS gates.

RESULT: ✓ _drain_single_qubit_ops should correctly drain EC measurements between rounds.
However, this depends on the precise operationsLeft ordering matching stim circuit order.

================================================================
4. EXECUTION LOOP ANALYSIS
================================================================

File: qccd_WISE_ion_route.py:2258-2360

The execution loop:
  for ms_round_idx, round_steps in groupby(routing_steps, key=ms_round_index):
      1. Apply reconfiguration (layout_after, schedule)
      2. _drain_single_qubit_ops(ms_round_idx)  ← drains M between EC rounds
      3. _execute_ms_gates(ms_round_idx, solved_pairs)
      4. Handle subsequent tiling steps

After all MS rounds: _drain_single_qubit_ops(total_ms_rounds) ← final drain

Each MS round causes a drain, so measurements appear between CX rounds.

Final drain at line 2357 catches any remaining operations after all MS gates.

================================================================
5. IDENTIFIED ISSUES / RISKS
================================================================

A) FIRST-PASS GATE NAME TRUNCATION (trapped_ion_compiler.py:381)
   The check `if instr[0] in ("R", "H", "M")` truncates "MR" → "M", "RX" → "R".
   NOT a bug in practice because stim.decomposed() in step 3 breaks these down first.
   BUT: if any stim circuit bypasses step 3 decomposition (e.g., exception path),
   MR, MRX, RX, MX would lose their compound semantics.
   
   Risk Level: LOW (step 3 decomposition covers this)
   
   NOTE: The second pass (line 400+) has dead-code handlers for MR (line 453),
   MRX (line 494), MX (line 477), RX (line 467) that can never be reached because
   the first pass already reduced all instructions to M, R, H. These handlers
   should either be removed or the first pass should preserve full gate names.

B) HIERARCHICAL CODE emit_inner_only_round
   Located at src/qectostim/experiments/stabilizer_rounds/hierarchical_concatenated.py:868
   When _emit_ec_round uses emit_inner_only_round (for hierarchical builders),
   it may skip outer-level measurements. Need to verify this path includes
   inner measurements.
   
   Risk Level: MEDIUM (affects concatenated code CSS surgery)

C) EC CX ROUNDS NOT IN get_phase_pairs (css_surgery_cnot.py:182)
   get_phase_pairs only returns BRIDGE CX pairs, not EC CX pairs from inline
   EC rounds. The ms_pair_count is patched post-build from actual circuit CX counts
   (ft_gadget_experiment.py:1754-1768), so the routing engine gets the correct total.
   
   But the analytics path (decompose_into_phases in gadget_routing.py:1166) uses
   get_phase_pairs for pair derivation, which won't include EC CX. This only affects
   the cross-validation (Fix 1) and timing estimation, not execution correctness.
   
   Risk Level: LOW (cross-validation mismatch only)

D) EPOCH CEILING WITH BLOCKED IONS INTERACTION
   If two CX instructions share the same data ion across consecutive TICK epochs
   (e.g., bridge CX on data[0] at epoch 1, then EC CX on data[0] at epoch 5),
   the decomposed MS gates for both would have the same data ion. The blocked_ions
   scan would block the data ion as soon as it scans the epoch-5 MS gate.
   
   But this doesn't affect M(ec_anc) because EC ancilla ions ≠ data ions.
   Only affects data ion single-qubit ops (post-MS rotations), which are handled
   by the per-ion queue drain.
   
   Risk Level: NONE for measurements

================================================================
6. CONCLUSION
================================================================

The stim circuit correctly includes ancilla measurements (MR/MRX) between each
EC round within CSS surgery merge phases. The decompose_to_native function
correctly decomposes these into physical operations with proper epoch tags.
The _drain_single_qubit_ops mechanism should correctly execute these measurements
between MS rounds via the epoch ceiling mechanism.

If EC rounds "don't seem right", possible causes to investigate further:
1. Verify with an actual d=3 CSS surgery run: print the operations list and
   check measurement positions relative to MS gates
2. Check if the hierarchical code path (emit_inner_only_round) properly includes
   inner measurements
3. Check if Fix 10 (shared-ion split) produces too many sub-rounds that confuse
   the epoch ordering
4. Check if the stim circuit for a specific code (e.g., Steane [[7,1,3]]) actually
   produces the expected MR/MRX instructions in emit_ancilla_measure_and_detectors
"""
