# CRITICAL IMPLEMENTATION STATE - ROTATION SERIALIZATION FIX

## Problem
Rotations (RX, RY) on different ions, even in completely different traps, are NOT executed in parallel on WISE.

## Root Cause Found
In `happensBeforeForOperations()` (qccd_parallelisation.py line ~320), shared `involvedComponents` create edges:

```python
for op in operationSequence:
    for component in set(op.involvedComponents):
        for prev_op in operations_by_component[component]:
            _add_edge(prev_op, op)        # FALSE EDGE when trap shared
        operations_by_component[component].append(op)
```

After `setTrap(trap)` is called in `_drain_single_qubit_ops`, each rotation has `involvedComponents = [ion, trap]`.
Two rotations on different ions in the SAME trap share the trap component → false dependency edge → serialized.

For DIFFERENT-trap rotations: The barrier positions from `_drain_single_qubit_ops` may split them into separate segments.
The drain code has a greedy loop that processes ops in passes. Each pass selects disjoint-ion ops.
The `ionsInvolved.union()` at line 2198 UNCONDITIONALLY adds ALL ion IDs (even non-selected ops), which limits parallelism.
Also, rotations are added to allOps SEQUENTIALLY across greedy passes, and the barrier is placed AFTER all passes.
After reorder_rotations_for_batching() reorders them by type, the paralleliser processes within barrier segments.

BUT the key issue may be: `reorder_rotations_for_batching()` at line 1075 in trapped_ion_compiler.py reorders ops by type,
but the barriers remain at their ORIGINAL positions. If rotations move across barrier boundaries during reordering,
the barriers would cut into the middle of what should be a contiguous type group.

WAIT — barriers are indices into the ops array. After `reorder_rotations_for_batching()` changes the ops order,
the barrier indices point to different ops! The barriers were computed on the ORIGINAL order but applied to the REORDERED ops!

No wait — the barriers are between type groups from `_drain_single_qubit_ops`. Like:
- [RX+RY group1 barrier] [MS ops barrier] [RX+RY group2 barrier] [MS ops barrier]
The reorder function only reorders WITHIN MS boundaries (hard barriers), so barriers at MS gates remain valid.
Barriers within rotation groups are also reordered to match.

Actually, let me re-read: barriers are INTEGER positions in the allOps array. After reordering, the barrier at 
position N now points to whatever op is at position N in the new order. The reorder function doesn't update barriers!

THIS IS THE ROOT CAUSE: `reorder_rotations_for_batching()` reorders allOps but barriers (stored separately in 
circuit.metadata["barriers"]) are NOT updated. The barrier positions become stale after reordering.

## Files to Edit
1. `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_parallelisation.py`
   - `happensBeforeForOperations()` line 303: builds DAG from involvedComponents
   - `paralleliseOperations()` line 418: WISE scheduling
   - `paralleliseOperationsWithBarriers()` line 666: splits at barrier positions
   - `reorder_rotations_for_batching()` line 66: reorders rotations — DOES NOT UPDATE BARRIERS

2. `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/utils/trapped_ion_compiler.py`
   - Line 1075: calls reorder_rotations_for_batching(allOps) but doesn't update barriers
   - Line 1078: passes stale barriers to paralleliseOperationsWithBarriers

3. `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`
   - `_drain_single_qubit_ops()` line 2031: drains 1q ops, calls setTrap(), creates barriers
   - `_TYPE_ORDER` at line 2024: XRotation=0, YRotation=1
   - Fix 9: RX+RY treated as one commuting group (_ROTATION_TKS), ONE barrier per group
   - Greedy disjoint-ion loop at line 2190: unconditional ionsInvolved.union() at line 2198

## TWO-PART FIX

### Fix Part 1: Don't include trap in involvedComponents for single-qubit gates on WISE
In `happensBeforeForOperations()`, skip trap-based edges for OneQubitGate when isWISEArch=True:
- Pass isWISEArch parameter through
- When building DAG edges, for single-qubit ops, use only ion-based components (not trap)

### Fix Part 2: Rebuild barriers after reordering
In `trapped_ion_compiler.py` line 1075, after reordering:
- Track each op's identity before reorder
- Map barrier positions to new positions in reordered list
- OR: regenerate barriers based on type transitions in reordered list

Actually, BETTER approach: Since reorder_rotations_for_batching already groups by type perfectly,
and the barriers from _drain exist to separate type groups, we should REGENERATE barriers after reordering.
The new barriers should be placed at type-transition points in the reordered sequence.

### Fix Part 3 (simplest, best): Skip trap component in DAG for 1q WISE ops
This alone would fix all the issues because:
1. Same-trap rotations no longer create false edges
2. Different-trap rotations already have no shared components (except maybe through barrier segmentation)

## Key Code Snippets

### happensBeforeForOperations (line 303-395)
```python
def happensBeforeForOperations(
    operationSequence, all_components, epoch_mode="edge",
):
    # ...
    for op in operationSequence:
        for component in set(op.involvedComponents):
            for prev_op in operations_by_component[component]:
                _add_edge(prev_op, op)  # TRAP CREATES FALSE EDGE
            operations_by_component[component].append(op)
```

### paralleliseOperations (line 418-660)
calls happensBeforeForOperations to build DAG, then schedules with WISE arch_busy_until global barrier

### _drain_single_qubit_ops (line 2031-2220)
Greedy disjoint-ion drain, ONE barrier per type group (Fix A)
setTrap() appends trap to involvedComponents

### trapped_ion_compiler.py schedule() (line 1060-1090)
```python
if is_wise:
    allOps = reorder_rotations_for_batching(list(allOps))
if barriers:
    parallelOpsMap = paralleliseOperationsWithBarriers(
        allOps, barriers, isWiseArch=is_wise, epoch_mode=epoch_mode,
    )
```
