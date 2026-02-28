# IMPLEMENTATION PROGRESS — DO NOT DELETE

## COMPLETED
1. qccd_parallelisation.py — happensBeforeForOperations(): Added isWISEArch param, skip trap-based DAG edges for 1q ops
2. qccd_parallelisation.py — paralleliseOperations(): Pass isWISEArch to happensBeforeForOperations
3. qccd_parallelisation.py — batch packing: Skip trap conflicts for 1q ops on WISE

## REMAINING
4. trapped_ion_compiler.py line 1075 — Rebuild barriers after reorder_rotations_for_batching()

### Change 4 Detail:
File: src/qectostim/experiments/hardware_simulation/trapped_ion/utils/trapped_ion_compiler.py
Line ~1075. Currently:
```python
        if is_wise:
            allOps = reorder_rotations_for_batching(list(allOps))

        if barriers:
            parallelOpsMap = paralleliseOperationsWithBarriers(
```

Change to:
```python
        if is_wise:
            allOps = reorder_rotations_for_batching(list(allOps))
            # Rebuild barriers at type-transition boundaries after
            # reordering.  The original barrier positions (computed
            # on the pre-reorder sequence) become stale when ops move.
            if barriers:
                new_barriers = []
                for i in range(1, len(allOps)):
                    prev_op = allOps[i - 1]
                    curr_op = allOps[i]
                    # Barrier at MS / multi-qubit gate boundaries
                    if isinstance(curr_op, TwoQubitMSGate) or isinstance(prev_op, TwoQubitMSGate):
                        if i not in new_barriers:
                            new_barriers.append(i)
                    # Barrier at transport (non-QubitOperation) boundaries
                    elif not isinstance(curr_op, QubitOperation) or not isinstance(prev_op, QubitOperation):
                        if i not in new_barriers:
                            new_barriers.append(i)
                barriers = sorted(set(new_barriers))

        if barriers:
```

Need to ensure TwoQubitMSGate and QubitOperation are imported in trapped_ion_compiler.py.

5. Run syntax check: python3 -c "import ast; ast.parse(open('...').read()); print('OK')"
6. Run diagnostic: PYTHONPATH=src WISE_INPROCESS_LIMIT=999999999 my_venv/bin/python -u _diag_rotation_serial.py 2>/dev/null
