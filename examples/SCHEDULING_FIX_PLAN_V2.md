# Scheduling Fix Plan V2 — AG Epoch Ordering + WISE Batch Optimisation

## IMPLEMENTATION STATUS
- [ ] Phase 1a: TICK epoch barriers in ionRouting (partition operations by epoch, route each group, insert barriers)
- [ ] Phase 1b: AG routing priority epoch-aware
- [ ] Phase 2a: WISE deferred scheduling (on by default)
- [ ] Phase 2b: WISE lookahead-based type selection
- [ ] Phase 3: Configurable epoch barrier mode in scheduler
- [ ] Phase 4: Tests — remove xfail, add new tests
- [ ] Phase 5: Regression validation (all 49 tests)

---

## 1. Problem Statement

### Two xfailed Tests
1. **`test_ag_tick_ordering_preserved`**: Ion D13 has MS(M18,D13) at epoch=5 scheduled at t=0.0016 *before* MS(D13,M10) at epoch=4 at t=0.0024. The scheduler can't fix this because component edges from transport ops already establish the wrong order.

2. **`test_ag_ms_stim_order`**: Same root cause — MS gates sharing ion D13 execute in wrong stim-origin order.

### Root Cause Analysis

The AG router `ionRouting()` in `qccd_ion_routing.py` is a greedy physical router:

```
while operationsLeft:
    # 1. Co-located loop: emit ops where ions are already in the same trap
    # 2. Routing: move ions for the next MS gate
    # 3. Go back to step 1
```

The **emission order depends on which ions become co-located first**, not the stim timeline. When M18 gets routed to D13's trap before M10 does:

1. MS(M18, D13) at epoch=5 becomes co-located → **emitted first**
2. Transport ops for routing M10 are emitted next
3. MS(D13, M10) at epoch=4 becomes co-located → **emitted second**

This creates a chain in `allOps`: `MS_epoch5 → transport_ops → MS_epoch4`. The transport ops share components (crossings, traps) creating DAG edges: `MS_epoch5 → transport → MS_epoch4`. When the cycle-safe epoch edge code tries to add `MS_epoch4 → MS_epoch5` (correct ordering), it detects a cycle and **refuses** — the damage is already done.

**Key insight**: The fix must happen in the **router**, not the scheduler. By the time operations reach the scheduler, the component-based DAG already encodes the wrong order.

### Why WISE Doesn't Have This Problem

WISE routing processes operations via `toMoves[idx]` arrays that preserve the original stim order. Reconfigurations are pre-computed by SAT solver. MS gates execute in the order they appear in `toMoves`, which mirrors the stim timeline. WISE never has to "wait and see" which ions become co-located first.

---

## 2. User Requirements & Design Input

### From the User
- **Cross-TICK ordering is mandatory**: "for stim instructions separated by a TICK we must make sure all their decomposed native operations of the first instruction happen before the second instruction"
- **Intra-TICK flexibility**: "we have flexibility to reorder operations associated with different stim instructions that are not separated by a TICK"
- **WISE batch grouping**: "try batch gates of the same type together where possible (delay measurement scheduling so all measurements execute together)"
- **Hybrid approach**: "concerned about removing critical weight scheduling — carefully test the tradeoff between critical weight and batching. Hybrid approach may work well"
- **Deferred scheduling**: "If an operation of a different type could be delayed without violating happens-before, defer it to the next batch to give same-type operations a chance to become ready"
- **Lookahead**: "prefer the type with the most candidates in the frontier rather than the one with the highest critical weight. More aggressive: wait one virtual time step"
- **Configurable epoch barriers**: user wants both per-ion edges and full barriers as options

### Design Decisions (confirmed with user)
1. **AG fix scope**: TICK epoch barriers in ionRouting (partition by epoch, route each group separately, insert barriers between) + epoch-aware routing priority within each epoch group
2. **Deferred scheduling**: On by default (not opt-in)
3. **Epoch mode**: Configurable — per-ion edges (default) + full epoch barriers (toggle)

---

## 3. Phase 1: Fix AG Router Epoch Ordering

### Phase 1a: TICK Epoch Barriers in ionRouting

**File**: `src/qectostim/.../compiler/qccd_ion_routing.py`

**What**: Partition the input `operations` list by `_tick_epoch` and route each epoch group through the existing `ionRouting` loop **separately**, inserting barriers between groups. This guarantees all epoch-N operations (and their transport) complete before any epoch-N+1 operation starts.

**Why this instead of a per-op epoch guard**: The per-op guard approach is fragile — it requires reasoning about whether routing will eventually deliver lower-epoch ions, and it interacts subtly with the greedy routing loop's progress guarantees. TICK barriers are simpler, proven (already used for CX group boundaries), and directly enforce the user's requirement: "all decomposed native operations of the first TICK-separated instruction must happen before the second".

**Algorithm**:
```python
def ionRouting(
    qccdArch: QCCDArch,
    operations: Sequence[QubitOperation],
    trapCapacity: int
) -> Tuple[Sequence[Operation], Sequence[int]]:
    # --- Step 0: Group operations by tick epoch ---
    epoch_groups: Dict[int, List[QubitOperation]] = defaultdict(list)
    for op in operations:
        ep = getattr(op, '_tick_epoch', 0)
        epoch_groups[ep].append(op)
    
    sorted_epochs = sorted(epoch_groups.keys())
    
    # If only one epoch (or no epoch tags), fall through to original logic
    if len(sorted_epochs) <= 1:
        return _ionRouting_inner(qccdArch, operations, trapCapacity)
    
    # Route each epoch group separately, accumulating allOps + barriers
    allOps: List[Operation] = []
    barriers: List[int] = []
    
    for epoch in sorted_epochs:
        epoch_ops = epoch_groups[epoch]
        group_ops, group_barriers = _ionRouting_inner(
            qccdArch, epoch_ops, trapCapacity
        )
        # Offset group_barriers by current allOps length
        offset = len(allOps)
        for b in group_barriers:
            barriers.append(b + offset)
        allOps.extend(group_ops)
        # Insert barrier at epoch boundary
        barriers.append(len(allOps))
    
    return allOps, barriers
```

The existing routing logic moves into `_ionRouting_inner()` (renamed from the current `ionRouting` body). The outer `ionRouting` function becomes a thin wrapper that partitions by epoch.

**Trace through the failing case**:
1. Operations are partitioned: epoch-4 ops → group A, epoch-5 ops → group B
2. `_ionRouting_inner(group A)` runs: routes MS(D13,M10)@epoch=4, emits it + transport
3. Barrier inserted at len(allOps)
4. `_ionRouting_inner(group B)` runs: routes MS(M18,D13)@epoch=5, emits it + transport
5. Result: epoch-4 fully before epoch-5 ✓, barrier separates them in the scheduler

**Performance impact**: Epoch groups are routed independently, which may produce slightly longer routing sequences than routing everything together (because the router can't opportunistically combine cross-epoch movements). However:
- Within each epoch, the router still has full freedom to reorder operations
- The user explicitly wants cross-TICK ordering enforced ("correctness first")
- For typical QEC circuits, most operations within the same epoch share the same structural pattern → few extra routing steps

**Architecture state between epochs**: The architecture state (ion positions, trap occupancy) is preserved between epoch calls since `_ionRouting_inner` operates on the same `qccdArch` object. The `refreshGraph()` calls maintain consistency.

### Phase 1b: Epoch-Aware Routing Priority

**File**: `src/qectostim/.../compiler/qccd_ion_routing.py` → routing priority (~line 95)

**What**: When selecting which MS gates to route, prioritize lower-epoch gates. This increases the likelihood that lower-epoch ions get moved first, reducing the time higher-epoch ops are held back.

**Current code**:
```python
opPriorities: Dict[Operation, int] = {op: i for i, op in enumerate(operations)}
# ...
toMove = sorted(
    [(k,o) for k,o in toMoveCandidates.items()],
    key=lambda ko: opPriorities[ko[1]]
)
```

**Fixed code** — change the priority key:
```python
toMove = sorted(
    [(k,o) for k,o in toMoveCandidates.items()],
    key=lambda ko: (
        getattr(ko[1], '_tick_epoch', -1),
        opPriorities[ko[1]],
    )
)
```

**Why**: Within each epoch group (after Phase 1a partitioning), this ensures that lower-stim-origin gates get first pick of available routing paths. Combined with Phase 1a's epoch partitioning, the overall routing order fully respects the stim timeline.

---

## 4. Phase 2: WISE Batch Type Optimisation

### Current State
The scheduler already has hybrid type selection:
```python
score = count(ops_of_type) × max_critical_weight
```
This balances batch size (count) with critical path urgency (max_cw). But it only considers operations that are **currently in the frontier** — it doesn't look ahead.

### Phase 2a: Deferred Scheduling (on by default)

**File**: `src/qectostim/.../compiler/qccd_parallelisation.py` → `paralleliseOperations()` (~line 230-280)

**Goal**: After choosing `chosen_type`, allow operations of non-chosen types to be **deferred** to the next batch, giving same-type operations a chance to accumulate.

**Key rule**: An operation can only be deferred if **none of its DAG successors are currently in the frontier**. If a successor is in the frontier, deferring the operation would delay the successor, increasing total schedule length.

**Algorithm** (inserted after `chosen_type` is selected):
```python
# Deferred scheduling: ops of non-chosen type stay in frontier
# for the next batch IF deferring won't delay any successor.
if isWISEArch and chosen_type is not None:
    deferred = []
    for op in candidates:
        if isinstance(op, QubitOperation) and not isinstance(op, chosen_type):
            # Check if any successor is already in the frontier
            succs_in_frontier = [
                s for s in happens_before.get(op, [])
                if s in frontier_set and s not in scheduled
            ]
            if not succs_in_frontier:
                deferred.append(op)
    # Remove deferred ops from candidates — they'll be reconsidered
    # in the next batch iteration.
    candidates = [op for op in candidates if op not in deferred]
```

**Why "on by default"**: The user explicitly chose this. The happens-before guard prevents any increase in total schedule length — deferral only happens when it's safe.

**Risk**: If ALL operations in the frontier are deferred, we'd have an empty batch → infinite loop. Guard against this:
```python
if not candidates:
    # Nothing left after deferral — un-defer everything to avoid stall
    candidates = deferred
    deferred = []
```

### Phase 2b: Lookahead-Based Type Selection

**File**: Same as 2a

**Goal**: Augment the type scoring to include "near-ready" operations — ops whose only unscheduled predecessors are in the **current frontier**. These ops will become available in the next batch.

**Algorithm**:
```python
# For each type, count "near-ready" ops in addition to frontier ops
def near_ready_count(op_type):
    count = 0
    for op in remaining_ops:
        if op in frontier_ops or op in scheduled:
            continue
        if not isinstance(op, QubitOperation) or not isinstance(op, op_type):
            continue
        # Check: all predecessors either scheduled or in current frontier?
        if all(p in scheduled or p in frontier_set for p in preds[op]):
            count += 1
    return count

# Updated scoring:
def _type_score(ops: List[Operation]) -> tuple:
    count = len(ops)
    max_cw = max(critical_weight[o] for o in ops)
    lookahead = near_ready_count(type(ops[0]))
    return ((count + lookahead) * max_cw, count + lookahead)
```

**Trade-off**: This adds ~O(n) overhead per batch selection. For typical trapped-ion circuits (hundreds of ops), this is negligible.

**Interaction with deferred scheduling**: Lookahead tells us which type will have more ops available soon. Deferral gives those ops time to become ready. Together, they maximise batch sizes:
1. Lookahead says "XRotation has 3 ready + 2 near-ready = 5 total potential"
2. Chosen type = XRotation
3. Deferral holds back the 1 Measurement in the frontier
4. Next batch: the 2 near-ready XRotations have arrived, creating a batch of 5

### Hybrid Scoring Summary

The final scoring for WISE type selection:

| Factor | Current | Proposed |
|--------|---------|----------|
| Batch count | `len(frontier_ops_of_type)` | `len(frontier) + near_ready_count` |
| Critical weight | `max(cw)` | `max(cw)` (unchanged) |
| Score formula | `count * max_cw` | `(count + lookahead) * max_cw` |
| Deferral | None | Defer non-chosen if no successors in frontier |

Critical weight is preserved — it still influences which type gets chosen when counts are similar. But count is augmented with lookahead, so "more of the same type available soon" gets proper weight.

---

## 5. Phase 3: Configurable Epoch Barrier Mode

**File**: `src/qectostim/.../compiler/qccd_parallelisation.py`

### Current State
Same-ion epoch edges via cycle-safe BFS (Fix 2b from V1 plan). These are "soft" — they only constrain operations that share an ion across epochs. Cross-ion synchronization at TICK boundaries is NOT enforced.

### Proposal: `epoch_mode` Parameter

Add `epoch_mode: str = "edge"` parameter to `happensBeforeForOperations()` and `paralleliseOperations()`:

| Mode | Behaviour | Parallelism | Correctness |
|------|-----------|-------------|-------------|
| `"edge"` (default) | Same-ion epoch edges only | Best | Per-ion only |
| `"barrier"` | Full barriers at epoch boundaries | Worst | All cross-ion sync enforced |
| `"hybrid"` | Same-ion edges + barriers only at measurement epochs | Medium | Good compromise |

**Implementation for `"barrier"` mode**:
```python
if epoch_mode == "barrier":
    # Group ops by epoch, add edges: ALL ops in epoch N → ALL ops in epoch N+1
    epoch_to_ops: Dict[int, List[Operation]] = defaultdict(list)
    for op in operationSequence:
        if isinstance(op, QubitOperation) and hasattr(op, '_tick_epoch'):
            epoch_to_ops[op._tick_epoch].append(op)
    
    sorted_epochs = sorted(epoch_to_ops.keys())
    for i in range(len(sorted_epochs) - 1):
        curr_epoch_ops = epoch_to_ops[sorted_epochs[i]]
        next_epoch_ops = epoch_to_ops[sorted_epochs[i + 1]]
        for a in curr_epoch_ops:
            for b in next_epoch_ops:
                _add_edge(a, b)
```

**Implementation for `"hybrid"` mode**:
Same as `"barrier"` but only at epoch boundaries where the next epoch contains Measurement or QubitReset operations (these are the ones that really need cross-ion synchronization — measurement results must be consistent).

**Plumbing**: The `epoch_mode` parameter would be passed from:
- `TrappedIonCompiler.schedule()` → `paralleliseOperations()`/`paralleliseOperationsWithBarriers()`
- Could be set via a compiler kwarg or inferred from the circuit (e.g., default to `"edge"` for WISE, `"hybrid"` for AG)

---

## 6. Phase 4: Testing

### Tests to Modify
1. **Remove `xfail`** from `test_ag_tick_ordering_preserved` (should now pass with Phase 1a epoch barriers)
2. **Remove `xfail`** from `test_ag_ms_stim_order` (should now pass with Phase 1a+1b)

### New Tests to Add

| Test | Class | Description |
|------|-------|-------------|
| `test_ag_epoch_barriers_present` | `TestAGEpochBarriers` | AG routing output has barriers at epoch boundaries |
| `test_ag_epoch_partitioning_correct` | `TestAGEpochBarriers` | Operations between epoch barriers all share the same epoch |
| `test_wise_deferred_same_type_batch` | `TestWISEDeferredScheduling` | With deferral, a lone Measurement in the frontier is deferred when XRotations dominate |
| `test_wise_deferred_no_stall` | `TestWISEDeferredScheduling` | Deferral doesn't stall when all frontier ops are the same type |
| `test_wise_deferred_respects_happens_before` | `TestWISEDeferredScheduling` | Ops with successors in the frontier are NOT deferred |
| `test_wise_lookahead_scores` | `TestWISELookahead` | Near-ready ops are counted in type score |
| `test_epoch_mode_edge` | `TestEpochMode` | Default edge mode: only same-ion epoch edges |
| `test_epoch_mode_barrier` | `TestEpochMode` | Barrier mode: all epoch-N ops before epoch-N+1 |
| `test_epoch_mode_hybrid` | `TestEpochMode` | Hybrid mode: barriers only at measurement epochs |
| `test_batch_type_consistency_with_deferral` | `TestWISESameTypeBatch` | WISE same-type invariant holds with deferral enabled |

### Regression Tests
- All 20 e2e tests must still pass
- All existing 29 scheduling tests must still pass (minus the 2 xfail removals)

---

## 7. Implementation Order

```
Phase 1a: TICK epoch barriers in ionRouting ──────────────┐
Phase 1b: AG routing priority epoch-aware ────────────────┤
                                                          ├─→ Remove xfail, run tests
Phase 2a: WISE deferred scheduling ───────────────────────┤
Phase 2b: WISE lookahead type selection ──────────────────┤
                                                          ├─→ Run WISE tests
Phase 3:  Configurable epoch barrier mode in scheduler ───┤
                                                          ├─→ Run all tests
Phase 4:  New tests ──────────────────────────────────────┘
Phase 5:  Full regression (49+ tests)
```

**Why this order**:
1. Phase 1a partitions operations by epoch in ionRouting → epoch ordering guaranteed by barriers → fixes both xfailed tests
2. Phase 1b (epoch-aware routing priority) further improves intra-epoch ordering
3. Phase 2 is the WISE optimization — independent of Phase 1
4. Phase 3 adds configurable epoch mode in the scheduler
5. Phase 4+5 validate everything

---

## 8. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Epoch partitioning slightly increases total routing steps (can't combine cross-epoch movements) | Medium | Within each epoch, full routing freedom is preserved. For typical QEC patterns, most ops in an epoch share the same structural pattern → minimal overhead. Monitor total `len(allOps)` before/after. |
| Architecture state inconsistency between epoch groups | Low | `_ionRouting_inner` operates on the same `qccdArch` object; ions stay where the previous epoch left them. `refreshGraph()` ensures topology is up-to-date. |
| Deferred scheduling reduces total throughput | Medium | The happens-before guard prevents deferring ops with frontier successors. Worst case: deferral is a no-op (nothing gets deferred). Monitor schedule depth before/after. |
| Full epoch barriers (`"barrier"` mode) in scheduler add too many DAG edges | Low | O(n²) edges per epoch boundary. For d=2 circuits (~50 ops, ~6 epochs), this is ~200 edges — manageable. For larger circuits, `"edge"` mode is default. |
| Lookahead computation is too slow | Low | O(n) per batch selection. Could be optimised with a "pending successors count" precomputed during scheduling. |

---

## 9. Key Files

| File | Changes |
|------|---------|
| `src/qectostim/.../compiler/qccd_ion_routing.py` | Phase 1a (epoch partitioning wrapper + `_ionRouting_inner` rename), Phase 1b (epoch-aware routing priority in `toMove` sort) |
| `src/qectostim/.../compiler/qccd_parallelisation.py` | Phase 2a (deferred scheduling), Phase 2b (lookahead scoring), Phase 3 (epoch_mode param) |
| `src/qectostim/.../utils/trapped_ion_compiler.py` | Phase 3 (pass epoch_mode to scheduler) |
| `src/qectostim/.../demo/test_scheduling.py` | Phase 4 (new tests, remove xfail) |
| `src/qectostim/.../demo/test_e2e.py` | Phase 5 (regression validation only, no changes) |

---

## 10. Operation Type Reference

```
Operation
├── CrystalOperation  (transport — NOT control-pulse-limited)
│   ├── Split, Merge, CrystalRotation, CoolingOperation
├── Move, JunctionCrossing, PhysicalCrossingSwap, GlobalReconfigurations  (transport)
├── ParallelOperation  (container)
└── QubitOperation  (control-pulse-limited in WISE)
    ├── OneQubitGate
    │   ├── XRotation  ← X-axis control pulse (5 μs)
    │   └── YRotation  ← Y-axis control pulse (5 μs)
    ├── Measurement    ← measurement laser (400 μs)
    ├── QubitReset     ← reset/cooling (400 μs)
    ├── TwoQubitMSGate ← MS control pulse (200 μs)
    └── GateSwap       ← routing swap
```

### WISE Type Constraint
- `isinstance(op, chosen_type)`: XRotation ≠ YRotation (different control pulses)
- Transport ops bypass the type constraint entirely
- Deferral only applies to QubitOperations, not transport ops

### involvedComponents
- `OneQubitGate`: `[ion]` → after `setTrap`: `[ion, trap]`
- `TwoQubitMSGate`: `[ion1, ion2]` → after `setTrap`: `[ion1, ion2, trap]`
- `Split/Merge`: `[trap, crossing, *crossing.connection]`
- `Move`: `[crossing]`

---

## 11. Existing Architecture Reference

### Tick Epoch Tagging (already implemented)
- `decompose_to_native()` tracks `_tick_epoch` counter incremented at each TICK
- Each `QubitOperation` tagged via `_tag(op, origin, epoch)` with `_tick_epoch` and `_stim_origin`
- `NativeCircuit.metadata["num_tick_epochs"]` stores total count

### DAG Construction (already implemented)
- `happensBeforeForOperations()` builds DAG from:
  1. Shared `involvedComponents` in list order
  2. Cycle-safe tick-epoch edges (BFS reachability check before adding)
- Kahn's algorithm topological sort

### Scheduler (already implemented)
- Greedy list-scheduling with critical-path priority
- WISE: hybrid type selection (`score = count × max_cw`), global batch barrier
- AG: no type constraint, component-based parallelism only

### Test Suite (29 tests, 14 classes)
- Run: `PYTHONPATH=src my_venv/bin/python -m pytest src/qectostim/.../demo/test_scheduling.py -v`
- E2E: `PYTHONPATH=src my_venv/bin/python -m pytest src/qectostim/.../demo/test_e2e.py -v`

---

## 12. Detailed Pseudocode

### Phase 1a: TICK Epoch Barriers in ionRouting

```python
# Rename current ionRouting body to _ionRouting_inner
def _ionRouting_inner(
    qccdArch: QCCDArch,
    operations: Sequence[QubitOperation],
    trapCapacity: int
) -> Tuple[Sequence[Operation], Sequence[int]]:
    # ... existing ionRouting body (unchanged) ...
    pass

# New ionRouting: partition by epoch, route each group, insert barriers
def ionRouting(
    qccdArch: QCCDArch,
    operations: Sequence[QubitOperation],
    trapCapacity: int
) -> Tuple[Sequence[Operation], Sequence[int]]:
    # Group operations by tick epoch
    epoch_groups: Dict[int, List[QubitOperation]] = defaultdict(list)
    for op in operations:
        ep = getattr(op, '_tick_epoch', 0)
        epoch_groups[ep].append(op)
    
    sorted_epochs = sorted(epoch_groups.keys())
    
    # Single epoch or no tags → original logic
    if len(sorted_epochs) <= 1:
        return _ionRouting_inner(qccdArch, operations, trapCapacity)
    
    allOps: List[Operation] = []
    barriers: List[int] = []
    
    for epoch in sorted_epochs:
        epoch_ops = epoch_groups[epoch]
        group_ops, group_barriers = _ionRouting_inner(
            qccdArch, epoch_ops, trapCapacity
        )
        offset = len(allOps)
        for b in group_barriers:
            barriers.append(b + offset)
        allOps.extend(group_ops)
        barriers.append(len(allOps))  # epoch boundary barrier
    
    return allOps, barriers
```

### Phase 2a: Deferred Scheduling

```python
# In paralleliseOperations(), after choosing chosen_type, before building batch:

if isWISEArch and chosen_type is not None:
    frontier_set = set(frontier_ops)
    _deferred_ops: List[Operation] = []
    _non_deferred: List[Operation] = []
    
    for op in candidates:
        if isinstance(op, QubitOperation) and not isinstance(op, chosen_type):
            # Can we safely defer? Check if any DAG successor is in frontier
            successors_in_frontier = any(
                s in frontier_set for s in happens_before.get(op, [])
            )
            if not successors_in_frontier:
                _deferred_ops.append(op)
                continue
        _non_deferred.append(op)
    
    # Safety: if we deferred everything, un-defer to avoid stall
    if not _non_deferred and _deferred_ops:
        _non_deferred = _deferred_ops
        _deferred_ops = []
    
    candidates = _non_deferred
```

### Phase 2b: Lookahead Scoring

```python
# Replace _type_score in paralleliseOperations():

def _near_ready_of_type(op_type: type) -> int:
    """Count ops not yet in frontier that will become ready
    once current frontier ops are scheduled."""
    count = 0
    frontier_set = set(frontier_ops)
    for op in remaining_ops:
        if op in frontier_set or op in scheduled:
            continue
        if not isinstance(op, op_type):
            continue
        # All predecessors either scheduled or in frontier?
        if all(p in scheduled or p in frontier_set for p in preds[op]):
            count += 1
    return count

def _type_score(ops: List[Operation]) -> tuple:
    count = len(ops)
    max_cw = max(critical_weight[o] for o in ops)
    lookahead = _near_ready_of_type(type(ops[0]))
    return ((count + lookahead) * max_cw, count + lookahead)
```

---

## 13. Correctness Arguments

### Phase 1a: Epoch Barriers
- **Correctness**: Each epoch group is routed independently. All operations in epoch N (and their transport ops) are emitted before any epoch-N+1 operations. The barrier index at the boundary ensures `paralleliseOperationsWithBarriers` treats them as separate scheduling segments.
  
- **Termination**: Each epoch group is a valid input to `_ionRouting_inner` (same invariants as the original `ionRouting`). The outer loop processes epochs in order, each terminates independently.

- **Architecture state**: `_ionRouting_inner` leaves ions in their final positions after routing. The next epoch group starts with ions wherever the previous group left them. This is correct because epoch-N+1 operations expect the architecture state that results from executing epoch-N — which is exactly what we get.

- **No deadlock**: Each epoch group is a complete, self-contained routing problem. The existing `ionRouting` body handles all routing within the group, with no cross-epoch dependencies to cause stalls.

### Phase 2a: Deferred Scheduling
- **No stall**: The "un-defer everything if candidates empty" guard prevents infinite loops.
- **No happens-before violation**: Deferred ops are only ops with no successors in the current frontier. Deferring them doesn't delay any successor.
- **Weak optimality**: Deferring an op pushes it one batch later. If its successor isn't ready yet, this costs nothing. If its successor becomes ready in the next batch, it costs one batch of latency on that path — but gains batch-size throughput for the chosen type.

### Phase 3: Epoch Barrier Mode
- **`"edge"` mode**: Existing behaviour — proven safe via cycle-safe BFS.
- **`"barrier"` mode**: Full linearisation across epochs. Can't violate epoch ordering. May increase schedule length.
- **`"hybrid"` mode**: Only barriers at measurement epochs. Good compromise for QEC circuits where measurement synchronization is critical but rotation ordering is flexible.
