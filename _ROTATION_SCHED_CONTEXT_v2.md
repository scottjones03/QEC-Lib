# ROTATION SCHEDULING FIX - FULL CONTEXT (v2)

## Problem
User: "significant serialisation issues of rotation gates between MS rounds"
Same-type rotation gates on ions in DIFFERENT traps are not parallelized.

## User constraints
1. Trap IS exclusive resource - ops in same trap MUST serialize
2. Same-type ops in different traps SHOULD parallelize
3. Different-type ops cannot parallelize (WISE multiplexing)

## WORKSPACE
/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim

## Key file FULL PATHS
1. `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_parallelisation.py` (840 lines) - scheduling
2. `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/trapped_ion_compiler.py` (1126 lines) - compiler, schedule() at L1050
3. `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py` (3443 lines) - routing
4. `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/experiment.py` (855 lines) - HAS SYNTAX ERROR

## BLOCKING ISSUE: experiment.py syntax error
Class docstring for TrappedIonExperiment starts at line 66 with box-drawing chars.
At line ~170, there's code fragment `parallelOpsMap = paralleliseOperationsWithBarriers(` 
INSIDE the docstring. The docstring never properly closes.
Already fixed apply_hardware_noise docstring (line 405) to use r""" with ASCII art.
NOW need to fix the class docstring at line 66-170+.
The closing """ needs to be found/added at the proper location.

## ROOT CAUSE: Barriers create excessive segmentation
paralleliseOperationsWithBarriers() in qccd_parallelisation.py L710 schedules each 
barrier segment INDEPENDENTLY and SEQUENTIALLY. Ops in different segments NEVER parallelize.

The barrier rebuild in schedule() (trapped_ion_compiler.py L1080-1090) creates barriers at:
1. MS gate boundaries (CORRECT)
2. QubitOperation <-> non-QubitOperation transitions (TOO AGGRESSIVE for WISE)

Transport ops (Split/Move/Merge) are non-QubitOperation, so they create barriers between 
rotation groups. This splits same-type rotations into separate segments.

## FIX 1: barrier rebuild in schedule() trapped_ion_compiler.py
Current code (~L1080):
```python
if isinstance(curr_op, TwoQubitMSGate) or isinstance(prev_op, TwoQubitMSGate):
    new_barriers.append(i)
elif not isinstance(curr_op, QubitOperation) or not isinstance(prev_op, QubitOperation):
    new_barriers.append(i)  # <-- THIS LINE causes problem for WISE
```
FIX: For WISE, only keep MS-gate barriers. Remove the transport-boundary barriers.
The DAG handles transport-rotation ordering via shared trap components (transport ops
have trap in involvedComponents, rotation ops have trap after setTrap).

## FIX 2: Revert WRONG WISE optimizations in qccd_parallelisation.py

### happensBeforeForOperations (around line 330-340):
Remove the WISE trap-edge-skipping for 1q ops. Current code:
```python
if isWISEArch and isinstance(component, Trap) and _is_1q(op):
    for prev_op in operations_by_component[component]:
        if not _is_1q(prev_op):
            _add_edge(prev_op, op)
    operations_by_component[component].append(op)
    continue
```
This should be REMOVED. All component edges should be enforced (trap IS exclusive).
Replace with the standard path (no WISE special case).

### paralleliseOperations greedy packing (around line 630-647):
Remove the WISE trap-conflict-skipping for 1q ops. Current code:
```python
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
Replace with standard component conflict checking (no WISE 1q special case).

## Key data structures
- 1q gate involvedComponents = [ion, trap] (after setTrap)  
- 2q MS gate involvedComponents = [ion1, ion2, trap] (after setTrap)
- Transport (Split/Move/Merge) involvedComponents = [trap, crossing, *crossing.connection]
- Transport NOT QubitOperation, NOT have ions
- Rotations are QubitOperation with len(op.ions)==1

## paralleliseOperationsWithBarriers (L710-750)
```python
barriers = [0] + list(barriers) + [len(operationSequence)]
t: float = 0.0
for start, barrier in zip(barriers[:-1], barriers[1:]):
    seg_ops = operationSequence[start:barrier]
    seg_schedule = paralleliseOperations(seg_ops, isWISEArch=isWiseArch, ...)
    # offset by previous segment end time
    for s, par_op in seg_schedule.items():
        key = s + t
        time_schedule[key] = par_op
    t = seg_end
```

## WISE scheduling in paralleliseOperations
- arch_busy_until: global barrier - no new batch until ALL ops in current batch finish
- Type selection: picks type with highest (count+lookahead)*max_cw*continuity score
- Deferred scheduling: all non-chosen-type QubitOps are deferred
- Component conflicts enforce exclusive trap usage

## Todo state
1. Read key code sections - COMPLETED
2. Fix experiment.py docstring syntax error - IN PROGRESS  
3. Run diagnostic to verify root cause of serialization
4. Revert incorrect WISE optimizations in qccd_parallelisation.py
5. Modify barrier rebuild in schedule() for WISE
6. Test the fix

## Run command
cd "/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim"
PYTHONPATH=src WISE_INPROCESS_LIMIT=999999999 my_venv/bin/python -u script.py 2>&1

## compile_gadget_for_animation returns  
arch, compiler, compiled, batches, ion_roles, p2l, remap
- compiled.scheduled.routed_circuit = RoutedCircuit
- compiled.scheduled.batches = ParallelOperation batches
- compiled.scheduled.metadata["all_operations"] = flat ops
- compiled.scheduled.metadata["parallel_ops_map"] = timing map
