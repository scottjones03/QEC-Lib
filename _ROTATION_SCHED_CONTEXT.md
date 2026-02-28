# Essential context for rotation scheduling fix

## Problem
User: "significant serialisation issues of rotation gates between MS rounds"
Same-type rotation gates on ions in DIFFERENT traps are not parallelized.

## User constraints
1. Trap IS exclusive resource - ops in same trap MUST serialize
2. Same-type ops in different traps SHOULD parallelize
3. Different-type ops cannot parallelize (WISE multiplexing)

## Key files
1. `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_parallelisation.py` (840 lines) - scheduling
2. `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/trapped_ion_compiler.py` (1126 lines) - compiler
3. `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py` (3443 lines) - routing

## WRONG WISE optimizations to REVERT in qccd_parallelisation.py
1. Lines ~330-340 `happensBeforeForOperations()`: Skips trap-based DAG edges between 1q-1q ops 
2. Lines ~630-647 `paralleliseOperations()`: Skips trap component conflicts for 1q ops, doesn't mark trap

## Analysis
- These optimizations INCREASE parallelism (incorrectly allowing same-trap parallel)
- They are NOT the root cause of different-trap serialization
- With correct behavior: different-trap ops share NO components → no DAG edge → should parallelize
- Need diagnostic to find actual root cause

## Diagnostic script fix needed
`_diag_rotation_sched.py` failed with `AttributeError: 'CompiledCircuit' object has no attribute 'routed'`
Fix: use `compiled.scheduled.routed_circuit` instead of `compiled.routed`
- `compiled.scheduled.routed_circuit.metadata["barriers"]` = barriers 
- `compiled.scheduled.routed_circuit.metadata["all_operations"]` = raw ops

## compile_gadget_for_animation returns
`arch, compiler, compiled, batches, ion_roles, p2l, remap`
- `compiled.scheduled.batches` = ParallelOperation batches
- `compiled.scheduled.metadata["all_operations"]` = flat ops after scheduling
- `compiled.scheduled.metadata["parallel_ops_map"]` = timing map

## schedule() in trapped_ion_compiler.py line 1050
- Gets barriers from `circuit.metadata.get("barriers", [])`
- Gets allOps from `circuit.operations`  
- If is_wise: reorders rotations, rebuilds barriers at MS/transport transitions ONLY
- Calls paralleliseOperationsWithBarriers or paralleliseOperations

## Key section in happensBeforeForOperations (lines ~310-370)
```python
def _is_1q(op): return isinstance(op, QubitOperation) and len(op.ions) == 1
for op in operationSequence:
    for component in set(op.involvedComponents):
        if isWISEArch and isinstance(component, Trap) and _is_1q(op):
            for prev_op in operations_by_component[component]:
                if not _is_1q(prev_op):
                    _add_edge(prev_op, op)
            operations_by_component[component].append(op)
            continue
        for prev_op in operations_by_component[component]:
            _add_edge(prev_op, op)
        operations_by_component[component].append(op)
```

## Key section in paralleliseOperations greedy packing (lines ~620-650)
```python
for op in candidates_sorted:
    if isWISEArch and isinstance(op, QubitOperation) and len(op.ions) == 1:
        if any(comp in used_components for comp in op.involvedComponents
               if not isinstance(comp, Trap)):
            continue
    else:
        if any(comp in used_components for comp in op.involvedComponents):
            continue
    batch.append(op)
    for comp in op.involvedComponents:
        if (isWISEArch and isinstance(op, QubitOperation)
                and len(op.ions) == 1 and isinstance(comp, Trap)):
            continue
        used_components.add(comp)
```

## _drain_single_qubit_ops (line 2031 qccd_WISE_ion_route.py)
- Drains eligible 1q ops with epoch-aware filtering
- Groups XRotation+YRotation into one _ROTATION_TKS group
- Greedy disjoint-ion execution within type group
- ONE barrier per type group (Fix A)

## ROOT CAUSE HYPOTHESIS (high confidence)
Barrier-based scheduling splits same-type rotations across segments when transport ops 
(Split/Move/Merge) are interleaved between rotations. Each barrier segment is scheduled 
INDEPENDENTLY and SEQUENTIALLY in paralleliseOperationsWithBarriers(), so rotations in 
different segments can NEVER be parallel.

The barrier rebuild in schedule() (trapped_ion_compiler.py L1080) creates barriers at 
QubitOperation <-> non-QubitOperation transitions. Transport ops are non-QubitOperation, 
so they create barriers between rotation groups.

## FIX PLAN
1. Fix experiment.py syntax error: Class docstring at line 66 has code fragments mixed in (line ~170 has `parallelOpsMap = paralleliseOperationsWithBarriers(`). Need to find where docstring should close and fix.
2. Revert incorrect WISE optimizations in qccd_parallelisation.py:
   - Lines ~330-340: Remove trap-edge-skipping for 1q ops in happensBeforeForOperations
   - Lines ~630-647: Remove trap-conflict-skipping for 1q ops in paralleliseOperations
3. Modify barrier rebuild in schedule() (trapped_ion_compiler.py) to only create barriers at MS gates for WISE. Remove transport-boundary barriers. DAG handles transport-rotation ordering via shared trap components.
4. Run diagnostic to verify fix

## BLOCKER
experiment.py has a docstring corruption starting at class TrappedIonExperiment line 66. 
The docstring contains box-drawing chars (valid) BUT ALSO contains code fragments mixed in 
at line ~170. This causes a SyntaxError when importing the module. Must fix first.

## Todo state
1. Read key code - COMPLETED
2. Fix experiment.py syntax error - IN PROGRESS
3. Run diagnostic to verify root cause  
4. Revert incorrect WISE optimizations
5. Modify barrier rebuild for WISE
6. Test

## Run command
PYTHONPATH=src WISE_INPROCESS_LIMIT=999999999 my_venv/bin/python -u script.py 2>&1
