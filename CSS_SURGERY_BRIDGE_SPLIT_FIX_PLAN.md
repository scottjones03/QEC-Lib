# CSS Surgery Bridge CX Split Fix Plan

## Root Cause

Bridge CX instructions in the stim circuit (e.g. `CX 0 21 1 21 2 21 3 21`) have
**multiple qubit pairs that all share one bridge ancilla qubit**.  In the
trapped-ion pipeline:

1. `_count_cx_instructions()` counts this as **1** instruction (correct for stim).
2. `decompose_to_native()` creates **1** `toMoves`/`toMoveOps` entry with 4 pairs.
3. `ionRoutingGadgetArch()` blindly groups all 4 pairs into **1** `parallelPairs`
   round: `[(22,1), (22,2), (22,3), (22,4)]`.
4. The SAT solver receives this as one round but **ion 22 can only be in one trap
   at a time**, so only ~1 of 4 pairs gets co-located.  The other 3 are
   **silently dropped**.

This causes:
- **Bridge CX rounds not executed**: reconfiguration happens but most MS gates
  for bridge pairs never fire.
- **Bridge ancillas in wrong positions**: ion never shuttled to all partners.
- **Blue dots without CX links in animation**: bridge operations incomplete.
- **First two stim CX rounds missing**: those are the gadget-phase bridge CX
  rounds that get dropped.

## Fix Strategy

### Fix 10: Split Shared-Ion Rounds (CRITICAL)

**Location**: `ionRoutingGadgetArch()` in `qccd_WISE_ion_route.py`, after the
sanity check at ~line 2830.

**Algorithm**: For each `parallelPairs` entry, greedily split pairs into
conflict-free sub-rounds where no two pairs share an ion.  Update
`phase_pair_counts` to reflect the new round count.

**Example**: `[(22,1), (22,2), (22,3), (22,4)]` → 4 separate rounds:
- `[(22,1)]`, `[(22,2)]`, `[(22,3)]`, `[(22,4)]`

**Why here, not earlier?**
- `_count_cx_instructions` and `decompose_to_native` are correct about stim
  structure (one CX instruction = one toMoveOps entry).
- The `toMoveOps is None` branch already has ion-conflict detection
  (`ionsAdded` set), but the `toMoveOps is not None` branch lacks it.
- Splitting at the routing level is self-contained and preserves metadata
  consistency — the sanity check passes with original counts, then we
  split and update both `parallelPairs` and `phase_pair_counts` together.

**Downstream impact**:
- `_build_plans_from_compiler_pairs()`: receives already-split pairs + updated
  counts → builds correct plans.
- `_route_round_sequence()`: receives 1-pair rounds → SAT solver easily
  co-locates → `solved_pairs` contains all pairs.
- `_execute_ms_gates()`: matches `solved_pairs` against `operationsLeft` by
  ion-pair set → works correctly with smaller rounds.
- `toMove_phase_tags` validation: `len(toMove_phase_tags) != len(parallelPairs)`
  after split, so validation block is skipped (it's debug-only logging).

### Fix 11: EC Measurement Validation (MEDIUM)

Investigate whether EC rounds are correctly inserting ancilla measurements
between consecutive stabilizer rounds.  Likely separate from the bridge CX
issue but visible in animation because bridge rounds fail → EC rounds appear
out of sequence.

### Verification Test: Stim-Instruction-Level Execution Check

Create a test that:
1. Builds a CSS surgery CNOT gadget (d=2)
2. Runs `decompose_to_native()` → `map_qubits()` → `ionRoutingGadgetArch()`
3. Counts how many MS gates were **actually executed** in the output
4. Compares against the **expected** 160 MS pairs from the stim circuit
5. Verifies each stim CX instruction has corresponding executed MS gates

## Implementation Order

1. **Fix 10**: Split shared-ion rounds in `ionRoutingGadgetArch()`
2. **Verification test**: Confirm 160/160 MS gates execute
3. **Fix 11**: EC measurement audit (if still needed after Fix 10)
