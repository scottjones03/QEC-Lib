# Drain System Analysis — CRITICAL CONTEXT

## Hardware Constraint
Only one instruction type can happen in parallel at any time across ALL traps.
This means XRotation and Measurement CANNOT be in the same parallel step.
XRotation and YRotation CAN be merged (Fix 9 already does this via _ROTATION_TKS).

## Current Drain Code Location
File: `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`
Function: `_drain_single_qubit_ops` at line 2019-2227

## Current Drain Flow
1. Build eligible list (C1: epoch ceiling, C2: blocked ions)
2. Build per-ion queues sorted by epoch
3. **Outer drain loop** (line 2168): picks fastest (lowest _TYPE_ORDER) type among all queue fronts
4. **Drain determination** (line 2182): XRot+YRot merged as _ROTATION_TKS
5. **Pop all matching** from all ion queues (line 2188)
6. **Inner greedy loop** (line 2197): disjoint-ion execution within type group
7. **Per-greedy-pass barrier** (line 2218): `barriers.append(len(allOps))` — ONE BARRIER PER GREEDY PASS

## Where The Bloat Is
Line 2218: `barriers.append(len(allOps))` inside the inner `while grp_remaining` loop.

This means if 6 XRotations need 2 greedy passes (some share traps), that's 2 barriers = 2 parallel steps just for rotations. Then another 2 for resets, another 2 for measurements.

## Key Insight: The greedy passes within one type group ARE SAME TYPE
Since all ops in grp_ops have the same type (or XRot+YRot), each greedy pass within that group is the same hardware operation type. The hardware CAN execute them all in sequence within a single "step" — the greedy passes are just about ion-disjointness.

BUT: the paralleliser (paralleliseOperationsWithBarriers) treats each barrier-bounded segment as a separate batch. So 2 greedy passes = 2 batches, even though the hardware could do them as 1 long step.

## REVISED Fix A Plan (Respecting Hardware Type Constraint)

### Approach: Collapse greedy passes within each type group into ONE barrier

**Before (current):**
```
[XRot pass 1] barrier [XRot pass 2] barrier [YRot pass 1] barrier [Reset pass 1] barrier [Meas pass 1] barrier
= 5+ barriers per drain call
```

**After (revised):**
```
[XRot+YRot all passes] barrier [Reset all passes] barrier [Meas all passes] barrier
= 3 barriers per drain call (or fewer if some types absent)
```

### Implementation Detail
Move `barriers.append(len(allOps))` from INSIDE the inner `while grp_remaining` loop (line 2218) to AFTER the `while grp_remaining` loop completes (line 2221). This collapses all greedy passes of the same type group into one barrier-bounded region.

The paralleliser will see one region per type group. The hardware constraint (one type at a time) is respected because each region contains only one operation type.

### Code Change
```python
# OLD (line 2197-2218):
while grp_remaining:
    toRemove: List[Operation] = []
    ionsInvolved: Set[Ion] = set()
    for op in grp_remaining:
        if ionsInvolved.isdisjoint(op.ions):
            trap = op.getTrapForIons()
            if not trap and len(op.ions) == 1:
                ion_parent = getattr(op.ions[0], 'parent', None)
                if ion_parent is not None:
                    trap = ion_parent
            if trap:
                op.setTrap(trap)
                toRemove.append(op)
        ionsInvolved = ionsInvolved.union(op.ions)

    if not toRemove:
        break

    for op in toRemove:
        op.run()
        allOps.append(op)
        operationsLeft.remove(op)
        grp_remaining.remove(op)
        single_qubit_executed += 1

    group_count += 1
    barriers.append(len(allOps))  # <-- MOVE THIS

# NEW:
while grp_remaining:
    toRemove: List[Operation] = []
    ionsInvolved: Set[Ion] = set()
    for op in grp_remaining:
        if ionsInvolved.isdisjoint(op.ions):
            trap = op.getTrapForIons()
            if not trap and len(op.ions) == 1:
                ion_parent = getattr(op.ions[0], 'parent', None)
                if ion_parent is not None:
                    trap = ion_parent
            if trap:
                op.setTrap(trap)
                toRemove.append(op)
        ionsInvolved = ionsInvolved.union(op.ions)

    if not toRemove:
        break

    for op in toRemove:
        op.run()
        allOps.append(op)
        operationsLeft.remove(op)
        grp_remaining.remove(op)
        single_qubit_executed += 1

    group_count += 1

# ONE barrier per type group, not per greedy pass
barriers.append(len(allOps))
```

### Expected Impact
- Per MS round: type groups = typically 3 (rotations, resets, measurements)
- Each type group now = 1 barrier instead of 1-3
- Total barriers per drain: ~3 instead of ~8
- Steps per MS round: ~7 instead of ~14 (reconfig + 3 drain types + MS + 2 boundaries)
- Total parallel steps: ~518 instead of ~1003

## Other Fixes (B and C)

### Fix B: Bridge CX Round Caching
Location: `gadget_routing.py` line ~1575 (round_signature=None for gadget phases)
Change: Compute round_signature for gadget phases when all rounds have identical MS pairs
Also: line ~2864 in route_full_experiment_as_steps — extend cache lookup to all phases with signatures
Expected: SAT calls halved

### Fix C: Combined Bridge+EC Routing  
Location: `gadget_routing.py` line ~1530-1580 in decompose_into_phases
Change: Merge bridge CX and EC CX into joint routing rounds instead of separate ms_pairs_per_round entries
Expected: MS rounds 74 → ~30

### Combined Projection
| Metric | Current | A only | A+B | A+B+C |
|--------|---------|--------|-----|-------|
| Steps/round | ~14 | ~7 | ~7 | ~7 |
| MS rounds | 74 | 74 | 74 | ~30 |
| Parallel steps | 1003 | ~518 | ~518 | ~210 |
| Fresh SAT | ~74 | ~74 | ~37 | ~15 |
| Compile time | 2200s | ~2200s | ~1100s | ~450s |
