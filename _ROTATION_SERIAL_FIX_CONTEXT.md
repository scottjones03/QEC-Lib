# CRITICAL CONTEXT: Rotation Serialization Fix

## Problem
Rotations (RX, RY) on different ions, even in completely different traps, are NOT executed in parallel on the WISE architecture.

## Root Cause Analysis

### File Locations
1. `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_parallelisation.py`
   - `happensBeforeForOperations()` line ~310: builds happens-before DAG from shared `involvedComponents`
   - `paralleliseOperations()` line ~420: WISE scheduling with `arch_busy_until` global barrier 
   - `paralleliseOperationsWithBarriers()` line ~600: splits ops at barrier positions
   - `reorder_rotations_for_batching()` line ~66: reorders rotations by type between MS rounds

2. `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`
   - `_drain_single_qubit_ops()` line 2031: drains 1q ops, calls setTrap() which APPENDS trap to involvedComponents
   - After setTrap, OneQubitGate has involvedComponents = [ion, trap]

3. `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/qccd_operations_on_qubits.py`
   - `setTrap()` line 40: appends trap to _involvedComponents
   - `OneQubitGate.qubitOperation()` line 121: creates with involvedComponents=[ion]

### DAG Edge Problem
In `happensBeforeForOperations()`:
```python
for op in operationSequence:
    for component in set(op.involvedComponents):
        for prev_op in operations_by_component[component]:
            _add_edge(prev_op, op)  # <- FALSE EDGE when trap shared
        operations_by_component[component].append(op)
```
- Two RX ops on same trap share trap component → false happens-before edge → serialized
- Even different-trap ops may be serialized due to barrier segmentation and tick_epoch edges

### The WISE Physics
On WISE architecture, single-qubit gates are BROADCAST - all ions getting the same rotation type execute simultaneously. The trap is NOT an exclusive resource for single-qubit gates. Only MS gates (which need specific chain configurations) need exclusive trap access.

## Fix Strategy

### Option A: Strip trap from involvedComponents for 1q ops in WISE mode
In `happensBeforeForOperations()` or `paralleliseOperations()`, when building the DAG for WISE, skip trap-based edges for single-qubit (OneQubitGate) operations. Keep ion-based edges.

### Option B: Don't add trap to involvedComponents for 1q ops
Modify setTrap or the drain to not add trap to involvedComponents for OneQubitGate on WISE. But trap is needed for fidelity calculations.

### Recommended: Option A
Modify `happensBeforeForOperations()` to accept `isWISEArch` param. When true, for OneQubitGate ops, only use ion-based edges (not trap-based):

```python
for op in operationSequence:
    for component in set(op.involvedComponents):
        # WISE: skip trap-based edges for single-qubit gates
        if isWISEArch and isinstance(op, OneQubitGate) and isinstance(component, Trap):
            continue
        for prev_op in operations_by_component[component]:
            _add_edge(prev_op, op)
        operations_by_component[component].append(op)
```

This preserves fidelity calculations (setTrap still called) while allowing parallel scheduling.

## Also Check
- Barrier positions from _drain_single_qubit_ops may wrongly split rotations
- _tick_epoch edges may create false cross-epoch deps between different-ion rotations
- The ionsInvolved.union() in drain's greedy loop unconditionally adds ALL ions (even non-selected ops)
