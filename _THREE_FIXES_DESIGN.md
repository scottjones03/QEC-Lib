# Three Fixes Design Document

## Problem Summary
- d=2 CSS Surgery: 1003 parallel steps for 74 MS rounds (~14 steps/round)
- 2200s compile time for tiny d=2 gadget 
- Won't scale to d=12

## Fix A: Drain Step Fusion (Reduce ~14 steps/round → ~5)

### Root Cause
In `qccd_WISE_ion_route.py`, the execution loop per MS round does:
1. `_apply_layout_as_reconfiguration` → 1 GlobalReconfiguration
2. `barriers.append()` → boundary
3. `_drain_single_qubit_ops()` → N barriers (1 per type-group × greedy passes)
4. `barriers.append()` → boundary between drain and MS
5. `_execute_ms_gates()` → 1 MS step
6. For offset tilings: repeat 1-5
7. `barriers.append()` → final boundary

### The Excess
`_drain_single_qubit_ops()` (line 2019) has TWO layers of barrier inflation:
1. **Type-segregated drain**: picks lowest-priority type first (rotations), drains ALL matching, THEN moves to next type (resets, measurements). Each type = separate barrier group.
2. **Greedy disjoint-ion passes within each type**: The inner `while grp_remaining` loop (line ~2185) adds `barriers.append()` for EACH greedy pass - if 6 XRotations can't all run in parallel due to trap conflicts, they get 2-3 passes = 2-3 barriers.

Between two consecutive MS rounds, ALL single-qubit ops commute on disjoint ions. The type segregation is unnecessary - XRotation on ion A and YRotation on ion B can run simultaneously. The per-pass barriers are also unnecessary since the paralleliser already handles ion-disjointness.

### Fix
In `_drain_single_qubit_ops()` (qccd_WISE_ion_route.py line 2019):
- Remove the drain loop's per-type-group barriers
- Instead: collect ALL eligible 1q ops, emit them (run + append to allOps + remove from operationsLeft), add ONE barrier at the end
- The downstream `paralleliseOperationsSimple()` (qccd_parallelisation.py line 19) already handles ion-disjoint parallelism correctly — it greedily packs ops that don't share components into the same parallel step
- Single-qubit ops between two MS rounds are ALL commutable (different ions, no ordering dependency within the same epoch window), so they can all go in one fence-bounded region

### Expected impact
- Steps/round: ~14 → ~5 (1 reconfig + 1 all-1q-ops + 1 MS + 2 barriers)
- Total parallel steps: 1003 → ~370 (74 rounds × 5)
- Compile time: no change (drain is fast, SAT is the bottleneck)

### Code location
File: `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`
Function: `_drain_single_qubit_ops` starting at line 2019
Key change: Replace the drain while loop (lines ~2152-2210) with a single pass that executes all eligible ops and appends ONE barrier.

Specifically change the drain loop section (starting after the per-ion queue build at ~line 2137):
```python
# OLD: type-segregated drain with per-pass barriers
while any(q for q in ion_queues.values()):
    best_tk = ...
    drain_tks = ...
    grp_ops = ...
    while grp_remaining:
        # greedy pass
        barriers.append(len(allOps))  # <-- EXCESS BARRIER
    
# NEW: single-pass drain, one barrier at end
all_eligible_ops = []
for q in ion_queues.values():
    while q:
        all_eligible_ops.append(q.popleft())

if all_eligible_ops:
    # Execute all eligible ops at once - they commute on disjoint ions
    for op in all_eligible_ops:
        trap = op.getTrapForIons()
        if not trap and len(op.ions) == 1:
            trap = getattr(op.ions[0], 'parent', None)
        if trap:
            op.setTrap(trap)
            op.run()
            allOps.append(op)
            operationsLeft.remove(op)
            single_qubit_executed += 1
    group_count = 1
    barriers.append(len(allOps))
```

The downstream `paralleliseOperationsSimple` will handle disjoint-ion parallelism correctly.

---

## Fix B: Bridge CX Round Caching (Reduce SAT calls ~50%)

### Root Cause
In `route_full_experiment_as_steps()` (gadget_routing.py line 2484):
- EC phases use `round_signature` for caching — identical EC phases are detected and replayed
- But GADGET phases (including CSS Surgery merge phases with bridge CX) set `round_signature=None` (line 1575)
- This means every bridge CX round and interleaved EC round within a merge phase gets its own fresh SAT solve

### The Excess
For a CSS Surgery merge phase at d=2:
- 2 merge rounds, each with bridge CX sub-rounds + EC layers
- The bridge CX pattern is IDENTICAL across merge rounds (same pairs, same ion positions)
- The interleaved EC layers are also identical to standalone EC rounds
- But because round_signature=None for gadget phases, all are freshly routed

### Fix
In `decompose_into_phases()` (gadget_routing.py ~line 1349):
1. **Compute a round_signature for gadget phases too** — when the CSS Surgery gadget produces identical bridge CX pairs across merge rounds, compute a signature from the sorted pairs
2. **Within `route_full_experiment_as_steps()`** (line 2484): extend the `ec_cache` to also cache gadget sub-round routing results using the same signature mechanism

Specifically:
1. In `decompose_into_phases()` around line 1575 where `round_signature=None` for gadget phases:
```python
# OLD:
round_signature=None,

# NEW: compute signature from the ms_pairs if they repeat across rounds
_gadget_sig = None
if ms_pairs_per_round:
    # Check if all rounds have identical pair sets
    _sigs = [tuple(sorted(tuple(sorted(p)) for p in rnd)) for rnd in ms_pairs_per_round]
    if len(set(_sigs)) == 1:
        _gadget_sig = _sigs[0]
round_signature=_gadget_sig,
```

2. In `route_full_experiment_as_steps()` at the ec_cache lookup (~line 2864):
```python
# OLD: only cache EC phases
if (plan.phase_type in _EC_PHASE_TYPES 
    and plan.round_signature is not None
    and plan.round_signature in ec_cache):

# NEW: cache any phase with a round_signature (including gadget phases with repeating patterns)
if (plan.round_signature is not None
    and plan.round_signature in ec_cache):
```

And similarly at the cache store (~line 2954):
```python
# OLD:
if cache_ec_rounds and plan.round_signature is not None:

# Already correct — just needs the signature to be non-None for gadget phases
```

### Expected impact
- For d=2 CSS Surgery: each merge phase has 2 identical rounds → 1 fresh solve + 1 replay per merge
- Saves ~50% of gadget-phase SAT calls
- For d=7: each merge phase has 7 identical rounds → 1 fresh + 6 replays = ~85% savings
- Compile time: 2200s → ~1100s for d=2

### Code locations
- `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py`
  - `decompose_into_phases()` line ~1575: set round_signature for gadget phases
  - `route_full_experiment_as_steps()` line ~2864: extend cache lookup to all phases with signatures
  - `route_full_experiment_as_steps()` line ~2954: cache store already works if signature is non-None

---

## Fix C: Combined Bridge+EC Routing (Reduce round multiplication ~5×→2×)

### Root Cause  
In `decompose_into_phases()` (gadget_routing.py ~line 1349), for CSS Surgery merge phases:
1. `get_phase_pairs()` returns bridge CX pairs per routing layer
2. `_split_shared_ion_rounds()` splits bridge pairs that share ions into sub-rounds
3. After each bridge sub-round, a FULL EC layer set (~4 CX phases) is appended as separate routing rounds

This creates ~5× round multiplication: 1 bridge sub-round + 4 EC layers = 5 routing rounds per bridge operation, times d merge rounds.

### The Excess
At the **Stim circuit level**, `_emit_joint_merge_round()` in css_surgery_cnot.py already combines bridge CX and EC CX in the same TICK window. But the **routing layer** doesn't know this — it sees bridge pairs and EC pairs as separate routing problems, each requiring its own GlobalReconfiguration + SAT solve.

In reality, bridge CX and EC CX can often be routed together in a single SAT call when they operate on disjoint ions (bridges use bridge ancillas + boundary data qubits; EC uses internal ancillas + data qubits). They share some data qubits but in different CX phases, so with careful scheduling they can share a routing round.

### Fix
In `decompose_into_phases()`, for gadget merge phases: instead of appending bridge sub-rounds and EC layers as separate ms_pairs_per_round entries, **combine** them into composite routing rounds:

```python
# OLD (simplified):
for bridge_round in bridge_cx_rounds:
    ms_pairs_per_round.append(bridge_round)  # bridge CX sub-round
    for ec_layer in ec_layers:
        ms_pairs_per_round.append(ec_layer)  # separate EC layer

# NEW: combine bridge CX and EC CX into joint routing rounds
for bridge_round in bridge_cx_rounds:
    for ec_layer_idx, ec_layer in enumerate(ec_layers):
        # First EC layer includes the bridge CX pairs
        if ec_layer_idx == 0:
            combined = bridge_round + ec_layer
        else:
            combined = ec_layer
        ms_pairs_per_round.append(combined)
```

This reduces routing rounds from `bridge_sub_rounds × (1 + num_ec_layers) × d` to `bridge_sub_rounds × num_ec_layers × d`.

For d=2: from ~20 per merge to ~8 per merge = 2.5× reduction.

### Alternative (simpler, lower risk):
Instead of merging pairs, add a `composite_round` flag to `PhaseRoutingPlan.ms_pairs_per_round` entries that tells the SAT solver to route bridge+EC pairs together. This is lower risk because the SAT solver already handles multi-pair routing.

### Expected impact
- Routing rounds per merge phase: ~20 → ~8 (d=2)
- Total MS rounds: 74 → ~30
- Combined with Fix A: total parallel steps: ~30 × 5 = ~150 (down from 1003)
- Combined with Fix B: compile time: ~30 rounds × half cached = ~15 fresh SAT solves instead of ~74

### Code location
- `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py`
  - `decompose_into_phases()` around line 1530-1580 where gadget ms_pairs_per_round is built

---

## Combined Impact Projection

| Metric | Current | Fix A only | A+B | A+B+C |
|--------|---------|-----------|-----|-------|
| Steps/round | ~14 | ~5 | ~5 | ~5 |
| Total MS rounds | 74 | 74 | 74 | ~30 |
| Parallel steps | 1003 | ~370 | ~370 | ~150 |
| SAT calls (fresh) | ~74 | ~74 | ~37 | ~15 |
| Compile time (est.) | 2200s | ~2200s | ~1100s | ~450s |

### d=12 projection with all 3 fixes:
- Qubits/block: ~265, Grid: ~20×24
- MS rounds: ~300 (down from ~2500 without Fix C)
- SAT calls: ~150 fresh (with Fix B halving)
- Per-SAT time: ~10s (larger grid but Level 2 patching keeps patches small)
- Estimated compile time: ~150 × 10s = ~25 minutes (feasible!)
- Without fixes: ~2500 × 30s = ~21 hours (infeasible)
