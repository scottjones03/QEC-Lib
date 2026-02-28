# Fix Plan: Single-Qubit Ops Must Be Scheduled Before MS Gates Per Round

## Problem Statement

Rotation gates, measurements, and resets are NOT being scheduled before the MS
gates in each round — even for the **first** round.  The correct physical
ordering within each QEC/MS round is:

```
Reconfiguration → [Rotations/Measurements/Resets] → MS Gates
```

But the current output sometimes shows MS gates appearing before their pre-MS
single-qubit operations.

---

## Root Cause Analysis

### Key Code Locations

| File | Lines | Function / Section |
|------|-------|--------------------|
| `qccd_WISE_ion_route.py` | 1804-1893 | `ionRoutingWISEArch` — main router |
| `qccd_WISE_ion_route.py` | 1986-2155 | `_drain_single_qubit_ops` — epoch-aware drain |
| `qccd_WISE_ion_route.py` | 2156-2260 | `_execute_ms_gates` — MS gate execution |
| `qccd_WISE_ion_route.py` | 2263-2380 | Execution loop (section 4) |
| `qccd_WISE_ion_route.py` | 2696-3277 | `ionRoutingGadgetArch` — gadget routing |
| `qccd_parallelisation.py` | 258-527 | `paralleliseOperations` — DAG-based scheduler |
| `qccd_parallelisation.py` | 528-570 | `paralleliseOperationsWithBarriers` |
| `trapped_ion_compiler.py` | 160-600 | `decompose_to_native` — CX → [RY, RX, XRot, MS, RY] |
| `trapped_ion_compiler.py` | 1047-1110 | `schedule()` — invokes paralleliser |

### Current Execution Flow (per MS round)

```
4a) _apply_layout_as_reconfiguration(first_step.layout_after)
    barriers.append(len(allOps))             ← barrier after reconfig

4b) _drain_single_qubit_ops(ms_round_idx)    ← drain 1q ops (rotations, M, R)

4c) _execute_ms_gates(ms_round_idx, solved_pairs)  ← execute 2q MS gates

    For subsequent routing steps in SAME ms_round:
        _apply_layout_as_reconfiguration(subsequent_step.layout_after)
        barriers.append(len(allOps))
        _execute_ms_gates(ms_round_idx, subsequent_step.solved_pairs)
        ^^^ NO _drain_single_qubit_ops call here! ^^^

    barriers.append(len(allOps))             ← barrier after round
```

### Identified Issues

#### Issue 1: `getTrapForIons()` gate on drain eligibility (HIGH IMPACT)

In `_drain_single_qubit_ops` (line ~2044):
```python
trap = op.getTrapForIons()
if trap:
    eligible.append(op)
```

If `getTrapForIons()` returns `None` for single-qubit ops (e.g., ions not
associated with a manipulation trap after initial placement), those ops are
silently **dropped from the eligible set**.  They remain in `operationsLeft`
and are NOT drained until a later round or the final drain AFTER all MS rounds.

**Result**: Rotations stay in `operationsLeft`.  MS gates execute first
(via `_execute_ms_gates`, which has its OWN `getTrapForIons` check that
may succeed because the ion pair is properly co-located after reconfig).
Then the rotations eventually drain in a later round — AFTER the MS gates
they were supposed to precede.

#### Issue 2: Missing drain between subsequent steps (MEDIUM)

Lines 2338-2360: When multiple routing steps target the same `ms_round_idx`
(e.g., aligned + offset tiling), subsequent steps do:
```
reconfig → barriers → _execute_ms_gates
```
with NO `_drain_single_qubit_ops` call between the reconfig and MS gates.
Single-qubit ops that became eligible after the first step's MS gates
(e.g., post-MS rotations whose ions are now unblocked by C2) or that
became trap-resident after the subsequent reconfig are skipped.

#### Issue 3: allOps ordering ≠ batch ordering (LOW-MEDIUM)

The `paralleliseOperationsWithBarriers` function splits `allOps` at barrier
indices and schedules each segment independently via DAG-based scheduling.
Within a segment, operations are ordered by the happens-before DAG and
critical-path heuristics, not by their position in `allOps`.

If a barrier segment contains BOTH single-qubit ops AND MS gates (because
the drain failed to separate them), the DAG scheduler may interleave them
based on component dependencies rather than enforcing strict 1q-before-2q
ordering.

#### Issue 4: C2 epoch order sensitivity (CSS Surgery specific)

The C2 constraint walks `operationsLeft` sequentially:
```python
for op in operationsLeft:
    if len(op.ions) >= 2:
        seen_2q_ions.update(op.ions)
    elif len(op.ions) == 1:
        if ion in seen_2q_ions:
            continue  # BLOCKED
```

For CSS Surgery bridge CX patterns where a bridge ancilla participates in
multiple CX pairs within the same epoch, all pre-MS rotations on the bridge
ion (for pairs 2, 3, ...) are blocked by C2 because the first pair's MS
gate already added the bridge ion to `seen_2q_ions`.

This is actually CORRECT for safety (the bridge ion's state changes between
pairs), but it means subsequent pairs' pre-MS rotations don't drain until
after pair 1's MS executes — potentially pushing them past the barrier.

---

## Fix Plan

### Fix A: Add drain call before every MS gate execution (CRITICAL)

**Location**: `ionRoutingWISEArch` execution loop, around line 2316 and 2340.

Always call `_drain_single_qubit_ops` before `_execute_ms_gates`, including
for subsequent routing steps within the same MS round.

```python
# Current (line ~2323):
ms_gates_executed, _unmatched = _execute_ms_gates(ms_round_idx, first_step.solved_pairs)

# Current subsequent steps (line ~2355):
_sub_exec, _sub_unmatched = _execute_ms_gates(ms_round_idx, subsequent_step.solved_pairs)
```

**Change**: Add `_drain_single_qubit_ops(ms_round_idx)` before EACH
`_execute_ms_gates` call, INCLUDING subsequent steps.

```python
# After each reconfig within subsequent steps:
_drain_single_qubit_ops(ms_round_idx)    # ← ADD THIS
_sub_exec, _sub_unmatched = _execute_ms_gates(...)
```

### Fix B: Barrier after drain, before MS gates (CRITICAL)

Currently, the drain and MS gates share the same barrier segment.  Add an
explicit barrier AFTER drain and BEFORE MS gates so the scheduler treats them
as separate sequential segments:

```python
# After step 4b drain:
_drain_single_qubit_ops(ms_round_idx)
barriers.append(len(allOps))    # ← ADD: barrier between drain output and MS gates

# Then step 4c:
_execute_ms_gates(ms_round_idx, first_step.solved_pairs)
barriers.append(len(allOps))    # (already exists at round end)
```

This ensures `paralleliseOperationsWithBarriers` places ALL drained 1q ops
in a segment that finishes BEFORE the MS gate segment.

### Fix C: Relaxed `getTrapForIons()` for single-qubit ops (MEDIUM)

In `_drain_single_qubit_ops`, single-qubit ops that fail `getTrapForIons()`
are silently skipped.  These ops should still be emitted — the ion IS in a
trap (it was placed during reconfig), but the trap lookup may use stale
parent references.

**Option C1**: After reconfig, refresh ion-trap associations before drain.
**Option C2**: Use the ion's current `parent` (trap) directly instead of
`getTrapForIons()`.
**Option C3**: Log a warning when `getTrapForIons()` fails for a 1q op
whose ion has a valid `.parent` trap, and use the parent directly.

### Fix D: Barrier-aware drain within subsequent steps (MEDIUM)

Restructure the subsequent-step inner loop to always drain before MS:

```python
for sub_idx, subsequent_step in enumerate(remaining_subsequent):
    if not any(s.solved_pairs or s.is_layout_transition
               for s in remaining_subsequent[sub_idx:]):
        break
    oldArrangementArr = _apply_layout_as_reconfiguration(...)
    barriers.append(len(allOps))
    _drain_single_qubit_ops(ms_round_idx)        # ← ADD
    barriers.append(len(allOps))                  # ← ADD
    _sub_exec, _sub_unmatched = _execute_ms_gates(...)
```

### Fix E: Add diagnostic logging (LOW — debugging aid)

Add logging at the START of `_drain_single_qubit_ops` to report:
- Total 1q ops in operationsLeft
- Number passing C1 (epoch ceiling)
- Number passing C2 (not ion-blocked)
- Number passing getTrapForIons
- Number actually drained

This will quickly identify which filter is dropping ops.

---

## Implementation Order

1. **Fix E** — Add diagnostic logging first to confirm root cause
2. **Fix B** — Add barrier between drain and MS gates (critical correctness)
3. **Fix A** — Add drain before subsequent-step MS gates
4. **Fix D** — Restructure subsequent-step loop
5. **Fix C** — Investigate and fix getTrapForIons failures if still needed

## Testing

- Run CSS Surgery d=2 compilation and check batch order in animation
- Verify: every batch containing MS gates is preceded by batch(es) with
  rotations/measurements/resets (when such ops exist for that round)
- Run existing TransversalCNOT d=2 gadget test to ensure no regression
- Check batch counts before/after — more barriers = more batches, but
  the total operation count should be unchanged

## Validation Criteria

For EACH MS round in the output batches:
1. All pre-MS rotations for pairs in that round appear in batches BEFORE
   the MS gate batch
2. Post-MS rotations appear in batches AFTER the MS gate batch
3. Measurements/resets from the previous round appear BEFORE current
   round's MS gates
4. No single-qubit ops from future epochs appear before current epoch's
   MS gates
