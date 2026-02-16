# Scheduling Fix Plan

## IMPLEMENTATION STATUS (updated during work)

### Completed:
1. **Fix 1: pass isWiseArch** ✅ — `trapped_ion_compiler.py` schedule() now reads `circuit.metadata["is_wise"]` and passes it to `paralleliseOperationsWithBarriers` and `paralleliseOperations`
2. **Fix 2a: tick epoch tagging** ✅ — `decompose_to_native()` now tracks `_tick_epoch` counter (incremented at each TICK in decomposed circuit). Each QubitOperation is tagged with `_tick_epoch` via extended `_tag(op, origin, epoch)`.
3. **Fix 2b: DAG tick-epoch edges** ✅ — `happensBeforeForOperations()` in `qccd_parallelisation.py` now adds edges for same-ion ops across different tick epochs (Step 1b).

### Current Issue (CRITICAL):
The tick-epoch edges create CYCLES when the AG router has scrambled operation order. Example:
- Router emits ops as [QubitOp_A(epoch=3), transport_ops, QubitOp_B(epoch=1)] 
- Component edge (list order): A→B
- Tick-epoch edge (epoch order): B→A
- Result: CYCLE! Topo sort drops nodes → KeyError in critical_weight

**Root cause**: The `happensBeforeForOperations` trusts list order for component-based edges, but the AG router (`qccd_ion_routing.py ionRouting()`) may emit QubitOperations in a different order than the stim program.

**Solution needed**: Either:
- (a) Post-routing sort of allOps to restore stim program order (Fix 2c)
- (b) Smarter DAG that avoids conflicts between component edges and epoch edges
- (c) Only add tick-epoch edges when component edge doesn't already exist between the same pair

I tried edge reversal (reverse component edges when epoch disagrees) but it created cycles with transport ops that don't have epochs.

**Next step**: Debug with `_debug_dag_cycle.py` to verify cycles exist, then implement Fix 2c (post-routing sort) to ensure allOps order matches stim program order. Transport ops stay in place; only QubitOps get re-sorted by (_tick_epoch, _stim_origin, original_list_position).

### Files Modified So Far:
- `src/qectostim/.../utils/trapped_ion_compiler.py`:
  - `schedule()`: passes `isWiseArch=is_wise` (Fix 1)
  - `decompose_to_native()`: tracks `_tick_epoch`, `new_instr_epoch` array, `_tag()` takes 3 args, stores `num_tick_epochs` in metadata
- `src/qectostim/.../compiler/qccd_parallelisation.py`:
  - `happensBeforeForOperations()`: has `_add_edge()` helper, Step 1b tick-epoch edges (sort by `_program_order_key`, connect across epochs)

### Key Architecture:
- `paralleliseOperationsWithBarriers()` splits at barrier indices, schedules each slice via `paralleliseOperations()`
- `paralleliseOperations()` builds DAG via `happensBeforeForOperations()`, then does greedy list-scheduling
- DAG edges from: (1) shared `involvedComponents` in list order, (2) tick-epoch edges
- WISE: `isWISEArch=True` → same-type-only batches + global batch barrier (`arch_busy_until`)
- Transport ops (Split, Merge, Move, etc.) have no `_tick_epoch` — they're routing infrastructure
- QubitOperation subclasses: XRotation, YRotation, Measurement, QubitReset, TwoQubitMSGate, GateSwap
- `calculateDephasingFromIdling()` also calls `paralleliseOperations()` — must work correctly too

### WISE Batch Grouping (Fix 3) Design:
- Current: picks batch type by max critical_weight of frontier candidates
- User asked: use HYBRID approach — critical weight + batch size. Don't fully remove critical weight.
- Proposed: score = count(ops_of_type) * sum_critical_weight → picks type with best throughput
- Also: consider delaying ops to group by type (all measurements together = one control pulse)
- Must respect happens-before. Only defer if no successor is in frontier.
- WISE type check: `isinstance(op, chosen_type)` — XRotation ≠ YRotation (correct for WISE)
- Transport ops (Split, Merge, Move, CrystalRotation, CoolingOperation) should NOT be restricted by WISE type constraint

### Test Suite (Fix 4) Design:
Tests go in: `src/qectostim/.../trapped_ion/demo/test_scheduling.py`
Run with: `PYTHONPATH=src my_venv/bin/python -m pytest src/qectostim/.../trapped_ion/demo/test_scheduling.py -v`
Tests:
1. test_wise_batches_same_type - every batch has only one QubitOp type
2. test_wise_no_mixed_rotation_axes - XRotation and YRotation never in same WISE batch
3. test_wise_global_barrier - no WISE batch starts before prev batch completes
4. test_wise_transport_unrestricted - transport ops can be in any batch
5. test_ag_allows_mixed_types - AG batches CAN have mixed qubit op types
6. test_ag_ms_order_matches_stim - MS gates execute in stim program order
7. test_tick_ordering_preserved - same-qubit ops across TICKs maintain order
8. test_happens_before_respected - scheduled start times respect stim order
9. test_component_disjointness - no two ops in same batch share component

### Debug Files Created:
- `_test_tick_epoch.py` — smoke test for tick epoch tagging (PASSES)
- `_debug_dag_cycle.py` — cycle detection debug script (NOT YET RUN)

## Context & User Insights

### User's Problem Statement
- WISE architecture cannot schedule two gates of a different type simultaneously (e.g. reset + rotation) — control pulses are multiplexed
- Simultaneous rotX pulses are fine, but a rotY pulse cannot happen at the same time
- Augmented grid is different: it allows multiple different control pulses at once as long as they are in different traps
- In augmented grid, CX gates (MS in native) happen in different order to the stim timeline (not an issue in WISE, unique to AG)
- Need more extensive and careful testing of scheduling algorithm — must respect the happens-before graph of the original stim circuit

### User's Design Input
- For stim instructions separated by a TICK, all decomposed native operations of the first instruction must happen before the second instruction
- For stim instructions NOT separated by a TICK, we should have flexibility to reorder them
- TICK barriers reduce parallelism, so we want best of both worlds (but correctness first)
- For WISE: try to batch gates of the same type together where possible (e.g. delay scheduling of a measurement gate so we can schedule all measurement gates together = faster)
- This batching must also respect happens-before rules
- Concerned about removing critical weight scheduling — should carefully test the tradeoff between critical weight and batching. Hybrid approach may work well.

---

## Bug Analysis

### Bug 1: WISE `isWiseArch` never passed to scheduler
**File:** `trapped_ion_compiler.py` → `schedule()` (line ~610)
**Problem:** `paralleliseOperationsWithBarriers(allOps, barriers)` called without `isWiseArch=self.is_wise`. Defaults to False.
**Effect:** WISE same-type-only batches + global batch barrier logic is DEAD CODE. WISE scheduling acts identically to augmented grid.

### Bug 2: TICK ordering not enforced
**File:** `trapped_ion_compiler.py` → `decompose_to_native()` (line ~268)
**Problem:** `if instr.startswith("TICK"): continue` — TICKs are silently discarded. No barriers or ordering constraints generated.
**Effect:** Cross-qubit ordering from stim circuit (synchronization points) is lost. Operations on different qubits that should be serialized by TICKs can be freely reordered.
**Mitigation:** For d=2, most TICK boundaries between CX groups already get barriers (one per CX instruction line), and single-qubit gates are ordered via ion-based DAG. But measurement→next-round boundaries and cross-qubit synchronization are not enforced.

### Bug 3: Augmented grid routing reorders operations
**File:** `qccd_ion_routing.py` → `ionRouting()` (line ~42-68)
**Problem:** Greedy loop emits ops to `allOps` in the order they become executable (ions co-located), not in stim program order. MS gates that need routing may complete in different order than stim timeline.
**Effect:** Operations may execute in wrong dependency order if shared-component detection doesn't catch the dependency.

---

## Operation Type Hierarchy (for WISE type checking)
```
Operation
├── CrystalOperation  (transport — not control-pulse-limited)
│   ├── Split
│   ├── Merge
│   ├── CrystalRotation
│   └── CoolingOperation
├── Move             (transport)
├── JunctionCrossing (transport)
├── PhysicalCrossingSwap (transport)
├── GlobalReconfigurations (transport)
├── ParallelOperation (container)
└── QubitOperation   (control-pulse-limited in WISE)
    ├── OneQubitGate
    │   ├── XRotation  ← X-axis control pulse
    │   └── YRotation  ← Y-axis control pulse
    ├── Measurement    ← measurement laser
    ├── QubitReset     ← reset/cooling  
    ├── TwoQubitMSGate ← MS control pulse
    └── GateSwap       ← (used in routing)
```

In WISE: `isinstance(op, chosen_type)` check means XRotation ≠ YRotation (correct — different control pulses).
Transport ops (Split, Merge, Move, etc.) should NOT be restricted by WISE type constraint — they don't use control pulses.

### involvedComponents per type
- OneQubitGate: `[ion]` → after setTrap: `[ion, trap]`
- TwoQubitMSGate: `[ion1, ion2]` → after setTrap: `[ion1, ion2, trap]`
- Measurement: `[ion]` → after setTrap: `[ion, trap]`
- QubitReset: `[ion]` → after setTrap: `[ion, trap]`
- Split/Merge: `[trap, crossing, *crossing.connection]`
- Move: `[crossing]`
- JunctionCrossing: `[junction, crossing]`
- CrystalRotation: `[trap]`
- CoolingOperation: `[trap]`

---

## Fix Plan

### Fix 1: Pass `isWiseArch` to scheduler (5 min)
In `schedule()`, read `circuit.metadata["is_wise"]` and pass it:
```python
is_wise = circuit.metadata.get("is_wise", False)
paralleliseOperationsWithBarriers(allOps, barriers, isWiseArch=is_wise)
```
Also pass it to the `paralleliseOperations` fallback.

### Fix 2: TICK-aware happens-before (layered approach)

#### 2a. Tick epoch computation in `decompose_to_native` (30 min)
- Track TICK positions in the flattened stim circuit
- Assign each `_stim_origin` value a `tick_epoch` number (0, 1, 2, ...)
- Tag each QubitOperation with `_tick_epoch`
- Store `tick_epoch` map in `NativeCircuit.metadata["tick_epochs"]`

#### 2b. DAG enhancement in `happensBeforeForOperations` (20 min)
- Accept optional `tick_epoch` data
- Add edges: if op A and op B share an ion AND `A._tick_epoch < B._tick_epoch`, then A→B
- This is tighter than TICK barriers: only constrains same-qubit operations across TICKs, preserving cross-qubit parallelism

#### 2c. Post-routing topological sort (15 min)
- After routing emits `allOps`, re-sort QubitOperations to restore `_stim_origin` order
- Transport ops (Split, Merge, Move) stay in place relative to their surrounding qubit ops
- This fixes the AG routing reordering issue

### Fix 3: WISE type-aware batch grouping (30 min)
**Goal:** Delay operations to batch by type, maximizing throughput on WISE.
**Hybrid approach (preserving critical weight):**

When `isWISEArch=True` and choosing `chosen_type`:
1. Group frontier candidates by type
2. For each type, compute: `score = count(ops_of_type) * critical_weight_sum(ops_of_type)`
3. Pick the type with the highest score (balances critical path with batch utilization)
4. This preserves critical weight's influence while favoring types with more ready ops

**Deferred scheduling (optional, test carefully):**
- If an op of a non-chosen type has no immediate successors in the frontier, it can be deferred to the next batch
- Only defer if doing so doesn't violate any happens-before constraint
- Must test tradeoff carefully: deferring may improve batching but could increase total schedule length

### Fix 4: Comprehensive test suite (40 min)
New file: `test_scheduling.py`

| Test | Description |
|------|-------------|
| `test_wise_batches_same_type` | Every batch in WISE schedule has only one QubitOperation type |
| `test_wise_no_mixed_rotation_axes` | XRotation and YRotation never in same WISE batch |
| `test_wise_global_barrier` | No WISE batch starts before previous batch completes |
| `test_wise_transport_unrestricted` | Transport ops (Split, Merge, Move) can be in any batch |
| `test_ag_allows_mixed_types` | AG batches CAN have mixed qubit op types (different traps) |
| `test_ag_ms_order_matches_stim` | MS gates in AG execute in stim program order |
| `test_tick_ordering_preserved` | Same-qubit ops across TICKs maintain order |
| `test_happens_before_respected` | Scheduled start times respect stim program order for same-qubit ops |
| `test_component_disjointness` | No two ops in same batch share involvedComponent |
| `test_wise_batch_type_grouping` | WISE batches tend to maximize same-type throughput |

### Implementation Order
1. Fix 1 (trivial)
2. Fix 2a (tick epoch tagging)
3. Fix 2b (DAG enhancement)
4. Fix 2c (post-routing sort)
5. Fix 3 (WISE batch grouping)
6. Fix 4 (tests)
7. Run all tests, validate

---

## Key Files
- `src/qectostim/.../trapped_ion/compiler/qccd_parallelisation.py` — scheduler
- `src/qectostim/.../trapped_ion/utils/trapped_ion_compiler.py` — compiler pipeline
- `src/qectostim/.../trapped_ion/compiler/qccd_ion_routing.py` — AG routing
- `src/qectostim/.../trapped_ion/compiler/qccd_WISE_ion_route.py` — WISE routing
- `src/qectostim/.../trapped_ion/utils/qccd_operations_on_qubits.py` — operation types
- `src/qectostim/.../trapped_ion/demo/test_e2e.py` — existing tests (20 tests)
