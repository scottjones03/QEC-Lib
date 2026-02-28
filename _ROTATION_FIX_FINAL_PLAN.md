# ROTATION SERIALIZATION FIX — IMPLEMENTATION PLAN (FINAL)

## EXACT CHANGES NEEDED

### Change 1: qccd_parallelisation.py — happensBeforeForOperations()
File: src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_parallelisation.py

Add `isWISEArch: bool = False` parameter to function signature (line 303).
In the DAG building loop (line ~322), skip Trap-based edges for single-qubit QubitOps when isWISEArch=True:

OLD (line 321-327):
```python
    # Build the happens-before relation based on the components involved.
    for op in operationSequence:
        for component in set(op.involvedComponents):
            for prev_op in operations_by_component[component]:
                # There is a happens-before relation (prev_op happens before op)
                _add_edge(prev_op, op)
            operations_by_component[component].append(op)
```

NEW:
```python
    # Build the happens-before relation based on the components involved.
    for op in operationSequence:
        for component in set(op.involvedComponents):
            # WISE optimisation: single-qubit gates are broadcast to all
            # ions simultaneously — the trap is NOT an exclusive resource.
            # Skip trap-based edges for 1q ops to allow parallel batching.
            if isWISEArch and isinstance(component, Trap) and isinstance(op, QubitOperation) and len(op.ions) == 1:
                # Still track the op for MS gates and multi-qubit ops on this trap
                # but don't create edges from previous 1q ops on same trap
                _skip_1q = True
                for prev_op in operations_by_component[component]:
                    # Only add edge if prev_op is NOT also a 1q op
                    if not (isinstance(prev_op, QubitOperation) and len(prev_op.ions) == 1):
                        _add_edge(prev_op, op)
                operations_by_component[component].append(op)
                continue
            for prev_op in operations_by_component[component]:
                _add_edge(prev_op, op)
            operations_by_component[component].append(op)
```

Also update function signature: 
`def happensBeforeForOperations(operationSequence, all_components, epoch_mode="edge", isWISEArch=False):`

### Change 2: qccd_parallelisation.py — paralleliseOperations()
Pass isWISEArch to happensBeforeForOperations (line ~430):

OLD:
```python
    happens_before, topo_order = happensBeforeForOperations(
        operationSequence, all_components, epoch_mode=epoch_mode
    )
```

NEW:
```python
    happens_before, topo_order = happensBeforeForOperations(
        operationSequence, all_components, epoch_mode=epoch_mode,
        isWISEArch=isWISEArch,
    )
```

### Change 3: qccd_parallelisation.py — paralleliseOperations() batch packing
In the greedy packing loop (line ~617), skip trap conflicts for 1q same-type ops on WISE:

OLD (line ~617):
```python
        for op in candidates_sorted:
            if isWISEArch and chosen_type is not None:
                if isinstance(op, QubitOperation) and not isinstance(op, chosen_type):
                    continue
            # Check component conflicts
            if any(comp in used_components for comp in op.involvedComponents):
                continue
            batch.append(op)
            for comp in op.involvedComponents:
                used_components.add(comp)
```

NEW:
```python
        for op in candidates_sorted:
            if isWISEArch and chosen_type is not None:
                if isinstance(op, QubitOperation) and not isinstance(op, chosen_type):
                    continue
            # Check component conflicts
            # WISE: skip trap conflicts for single-qubit ops (broadcast)
            if isWISEArch and isinstance(op, QubitOperation) and len(op.ions) == 1:
                # Only check ion conflicts, not trap conflicts
                op_ions = set(op.ions)
                if any(comp in used_components for comp in op.involvedComponents
                       if not isinstance(comp, Trap)):
                    continue
            else:
                if any(comp in used_components for comp in op.involvedComponents):
                    continue
            batch.append(op)
            for comp in op.involvedComponents:
                # WISE 1q: don't mark trap as used (other 1q ops can share it)
                if isWISEArch and isinstance(op, QubitOperation) and len(op.ions) == 1 and isinstance(comp, Trap):
                    continue
                used_components.add(comp)
```

### Change 4: trapped_ion_compiler.py — rebuild barriers after reorder
File: src/qectostim/experiments/hardware_simulation/trapped_ion/utils/trapped_ion_compiler.py

OLD (line 1075-1078):
```python
        if is_wise:
            allOps = reorder_rotations_for_batching(list(allOps))

        if barriers:
```

NEW:
```python
        if is_wise:
            allOps = reorder_rotations_for_batching(list(allOps))
            # Rebuild barriers at type-transition points after reordering.
            # The original barrier positions become stale when ops move.
            if barriers:
                new_barriers = []
                for i in range(1, len(allOps)):
                    prev_op = allOps[i - 1]
                    curr_op = allOps[i]
                    # Barrier at MS gate boundaries
                    if isinstance(curr_op, TwoQubitMSGate) or isinstance(prev_op, TwoQubitMSGate):
                        if i not in new_barriers:
                            new_barriers.append(i)
                    # Barrier at transport op boundaries
                    elif not isinstance(curr_op, QubitOperation) or not isinstance(prev_op, QubitOperation):
                        if i not in new_barriers:
                            new_barriers.append(i)
                barriers = sorted(set(new_barriers))

        if barriers:
```

## IMPORTS NEEDED
In qccd_parallelisation.py, Trap is already imported via `from ..utils.qccd_nodes import *`
In trapped_ion_compiler.py, TwoQubitMSGate is imported via `from ..compiler.qccd_parallelisation import *` or similar

## TESTING
Run: PYTHONPATH=src WISE_INPROCESS_LIMIT=999999999 my_venv/bin/python -u _diag_rotation_serial.py 2>/dev/null
Compare rotation batches before and after fix.
