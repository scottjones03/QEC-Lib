# Implementation Plan: Phase-Aware Integration + Reconfig Merge + SAT Direction Experiment

**Date**: 2025-01-XX
**Status**: Work Streams A & B COMPLETE — e2e passing, 26 unit tests green

### Implementation Status

| Item | Status |
|------|--------|
| **Work Stream A** — Integration Layer | ✅ COMPLETE |
| A1. `decompose_into_phases()` | ✅ Implemented + tested |
| A2. `route_full_experiment()` orchestrator | ✅ Implemented + tested |
| A3. Wire into `run_single_gadget_config()` | ✅ Implemented (auto-detects multi-block gadgets) |
| A4. Timing & Noise Integration | ✅ `compute_schedule_timing()` implemented |
| A5. `FullExperimentResult` dataclass | ✅ Implemented + tested |
| **Work Stream B** — Reconfig Merge | ✅ COMPLETE |
| `build_merged_layout()`, `merge_reconfig_schedules()` | ✅ Implemented + tested |
| **Work Stream C** — SAT Direction Experiment | ❌ NOT STARTED |
| **E2E Verification** | ✅ `_e2e_test.py` passes (d=2 TransversalCNOT, phase_aware=True) |
| **Unit Tests** | ✅ 26/26 passing |

### Bugs Fixed During Integration

| # | Bug | Fix |
|---|-----|-----|
| 1 | Missing `initial_layout` on `BlockSubGrid` | Added `_create_ions_from_allocation()` |
| 2 | `build_merged_layout` shape mismatch | Use `n_cols * k` for layout widths |
| 3 | 0-based ion index collision (0 = empty sentinel) | Changed to 1-based (`global_idx + 1`) |
| 4 | `max_inner_workers` not threaded to SAT solver | Added parameter through full call chain |
| 5 | Active ions scan used `n_cols` instead of `layout.shape[1]` | Fixed in `route_blocks_parallel._route_single_block` |
| 6 | SAT `Manager()` hang on macOS (spawn context) | `_force_serial` bypass when `max_inner_workers <= 1` |
| 7 | Phase type mismatch (`stabilizer_round_pre` ≠ `stabilizer_round`) | `startswith("stabilizer_round")` matching |
| 8 | Cross-block pairs can't fit in EC-sized patches | `gadget_subgridsize` override for full merged grid |
| 9 | `qubit_to_ion` 0-based vs 1-based mismatch | Changed to `{q: q + 1}` in `partition_grid_for_blocks` |

---

## Table of Contents

1. [Audit Summary](#1-audit-summary)
2. [Work Stream A — Integration Layer](#2-work-stream-a--integration-layer)
3. [Work Stream B — Reconfig Merge Optimisation](#3-work-stream-b--reconfig-merge-optimisation)
4. [Work Stream C — SAT Direction Constraint Experiment](#4-work-stream-c--sat-direction-constraint-experiment)
5. [Dependency Graph & Sequencing](#5-dependency-graph--sequencing)
6. [File Change Inventory](#6-file-change-inventory)
7. [Testing Strategy](#7-testing-strategy)
8. [Risk Register](#8-risk-register)

---

## 1. Audit Summary

### What exists (Steps 2-7 building blocks)

All implemented as standalone functions in `gadget_routing.py` (1303 lines):

| Step | Function(s) | Status |
|------|-------------|--------|
| 2 | `derive_ms_pairs_from_metadata()`, `derive_gadget_ms_pairs()` | ✅ Tested |
| 3 | `allocate_block_regions()`, `partition_grid_for_blocks()`, `map_qubits_per_block()` | ✅ Tested |
| 4 | `build_ion_return_bt()`, `build_ion_return_bt_for_patch_and_route()` | ✅ Tested |
| 5 | `replay_cached_round()`, `CachedRoundResult` | ✅ Tested |
| 6 | `_remap_block_schedule_to_global()`, `_merge_block_schedules()`, `route_blocks_parallel()` | ✅ Tested |
| 7 | `build_merged_layout()`, `route_gadget_with_transitions()`, `GadgetPhaseResult` | ✅ Tested |

### What's missing (the "glue")

| Gap | Description | Severity |
|-----|-------------|----------|
| **G1** | `run_single_gadget_config()` calls `_route_and_simulate()` — the old monolithic pipeline. Phase-aware routing never runs. | **CRITICAL** |
| **G2** | No `PhaseAwareRouter` class or `route_full_experiment()` orchestrator. | CRITICAL |
| **G3** | No phase decomposition function: `QECMetadata.phases` → `List[PhaseRoutingPlan]`. | HIGH |
| **G4** | No `get_phase_pairs()` on any gadget class. `derive_gadget_ms_pairs` falls back to transversal pairing. | MEDIUM |
| **G5** | `routing_config.py` missing `phase_routing_enabled`, `ion_return_constraint` fields. | LOW |
| **G6** | `trapped_ion_compiler.py` missing `phase_aware_route()` method. | LOW (bypass) |
| **G7** | `qccd_WISE_ion_route.py` unchanged — no `metadata_cache_seed` or `ion_return_bt` params. | LOW (bypass) |
| **G8** | No gadget sweep cells in the notebook. | LOW |

### Key Insight

The building blocks call `_patch_and_route()` directly (right level of abstraction).
The integration layer does NOT need to touch `trapped_ion_compiler.py` or
`ionRoutingWISEArch()` at all — it can bypass the `TrappedIonCompiler.compile()`
pipeline entirely and assemble operations + timing from the `_patch_and_route`
results. This is simpler and less risky than modifying the deep compiler stack.

---

## 2. Work Stream A — Integration Layer

### A1. Phase Decomposition Function

**File**: `gadget_routing.py`
**New function**: `decompose_into_phases()`

```python
def decompose_into_phases(
    qec_meta: QECMetadata,
    gadget: Gadget,
    qubit_allocation: QubitAllocation,
    sub_grids: Dict[str, BlockSubGrid],
    qubit_to_ion: Dict[int, int],
    k: int,
) -> List[PhaseRoutingPlan]:
    """Convert QECMetadata.phases into a sequence of PhaseRoutingPlan objects.

    For each phase in qec_meta.phases:
      - Determine phase_type: "ec", "gadget", or "transition"
      - Identify interacting_blocks (from phase.active_blocks)
      - Identify idle_blocks (excluded from SAT grid)
      - Extract MS pairs:
          - EC phases: derive_ms_pairs_from_metadata() per active block
          - Gadget phases: derive_gadget_ms_pairs()
      - Compute round count and cacheability
      - Build grid_region from interacting blocks' sub-grids
      - Compute round_signature for cache dedup

    Returns one PhaseRoutingPlan per phase, in temporal order.
    """
```

**Algorithm**:
```
plans = []
for i, phase in enumerate(qec_meta.phases):
    phase_type = phase.phase_type   # "ec" | "gadget" | "transition"
    active = phase.active_blocks    # List[str]
    idle = [b for b in all_block_names if b not in active]

    if phase_type == "ec":
        # Each active block routes independently
        ms_pairs = {}
        for block_name in active:
            ms_pairs[block_name] = derive_ms_pairs_from_metadata(
                qec_meta, sub_grids[block_name].qubit_to_ion, block_name
            )
        # EC phases are cacheable: identical stabilizer rounds repeat d times
        # Canonical signature = sorted tuple of (block_name, tuple of sorted pairs per round)
        sig = _compute_round_signature(ms_pairs)
        # Check if any previous EC phase has the same signature
        identical_to = _find_matching_phase(plans, sig)

        plan = PhaseRoutingPlan(
            phase_type="ec",
            phase_index=i,
            interacting_blocks=active,
            idle_blocks=idle,
            ms_pairs_per_round=_flatten_block_pairs(ms_pairs),
            num_rounds=phase.num_rounds,      # typically d
            is_cached=(identical_to is not None),
            identical_to_phase=identical_to,
            round_signature=sig,
        )

    elif phase_type == "gadget":
        # 2-block interaction on merged grid
        ms_pairs_list = derive_gadget_ms_pairs(
            gadget, i, qubit_allocation, active, qubit_to_ion
        )
        plan = PhaseRoutingPlan(
            phase_type="gadget",
            phase_index=i,
            interacting_blocks=active,
            idle_blocks=idle,
            ms_pairs_per_round=ms_pairs_list,
            num_rounds=1,
            is_cached=False,
        )

    plans.append(plan)
return plans
```

**Dependencies**: Existing `derive_ms_pairs_from_metadata`, `derive_gadget_ms_pairs`, `PhaseRoutingPlan` dataclass.

---

### A2. Route Full Experiment Orchestrator

**File**: `gadget_routing.py`
**New function**: `route_full_experiment()`

This is the main integration point. It replaces the `_route_and_simulate()` call
in `run_single_gadget_config()`.

```python
def route_full_experiment(
    qec_meta: QECMetadata,
    gadget: Gadget,
    qubit_allocation: QubitAllocation,
    k: int,
    subgridsize: Tuple[int, int, int] = (6, 4, 1),
    base_pmax_in: int = 1,
    lookahead: int = 4,
    max_inner_workers: int | None = None,
    stop_event=None,
    progress_callback=None,
) -> FullExperimentResult:
    """End-to-end phase-aware routing for a fault-tolerant gadget experiment.

    Orchestrates Steps 2-7: for each temporal phase, routes on the
    appropriate (sub-)grid using the existing building blocks, stitching
    results together with BT-embedded transitions.

    Returns layouts, schedules, and timing for every phase.
    """
```

**Algorithm (pseudocode)**:

```python
def route_full_experiment(...) -> FullExperimentResult:
    # === SETUP ===
    # 1. Partition grid into per-block sub-grids (Step 3)
    sub_grids = partition_grid_for_blocks(qec_meta, qubit_allocation, k)

    # 2. Build per-block ion mapping (Step 3)
    #    Need measurement_ions_per_block, data_ions_per_block from
    #    the Architecture/Ion objects. For now, derive from qubit_allocation.
    layouts = map_qubits_per_block(sub_grids, meas_ions, data_ions, k)

    # 3. Build qubit_to_ion mapping (union of all block mappings)
    qubit_to_ion = {}
    for sg in sub_grids.values():
        qubit_to_ion.update(sg.qubit_to_ion)

    # 4. Decompose into phases (Step A1 above)
    plans = decompose_into_phases(qec_meta, gadget, qubit_allocation,
                                   sub_grids, qubit_to_ion, k)

    # === PHASE LOOP ===
    phase_results = []
    cached_ec_results: Dict[Tuple, CachedRoundResult] = {}
    current_layouts: Dict[str, np.ndarray] = {
        name: sg.initial_layout for name, sg in sub_grids.items()
    }

    for plan in plans:
        if plan.phase_type == "ec":
            result = _route_ec_phase(
                plan, sub_grids, current_layouts,
                cached_ec_results, k, subgridsize, base_pmax_in,
                max_inner_workers, stop_event, progress_callback,
            )
        elif plan.phase_type == "gadget":
            result = _route_gadget_phase(
                plan, sub_grids, current_layouts,
                k, subgridsize, base_pmax_in,
                max_inner_workers, stop_event, progress_callback,
            )
        else:
            # transition phases are handled by BT constraints
            # in the adjacent EC/gadget phases
            continue

        phase_results.append(result)
        # Update current_layouts from result
        _update_current_layouts(current_layouts, result, sub_grids)

    # === ASSEMBLY ===
    # Concatenate all phase schedules into a single operation timeline
    # Compute total execution time from schedule pass counts × swap times
    total_schedule, total_time = _assemble_full_schedule(phase_results)

    return FullExperimentResult(
        phase_results=phase_results,
        total_schedule=total_schedule,
        total_exec_time=total_time,
        sub_grids=sub_grids,
    )
```

**Helper: `_route_ec_phase`**:
```python
def _route_ec_phase(plan, sub_grids, current_layouts, cache, ...):
    # Check cache first
    if plan.is_cached and plan.round_signature in cache:
        cached = cache[plan.round_signature]
        replays = replay_cached_round(cached, plan.num_rounds - 1)
        return ECPhaseResult(rounds=[cached] + replays, from_cache=True)

    # Build ion-return BT (Step 4) — ions must return to start after each round
    bt = build_ion_return_bt_for_patch_and_route(
        current_layouts[plan.interacting_blocks[0]], plan.num_rounds
    )

    # Route first round with BT (Step 6: parallel per-block)
    block_results = route_blocks_parallel(
        sub_grids={name: sub_grids[name] for name in plan.interacting_blocks},
        ms_pairs_per_block=_split_pairs_by_block(plan, sub_grids),
        wiseArch_params={"k": k},
        subgridsize=subgridsize,
        base_pmax_in=base_pmax_in,
        BTs_per_block={name: bt for name in plan.interacting_blocks},
    )

    # Merge block schedules (Step 6)
    merged = _merge_block_schedules(
        {name: r.schedule for name, r in block_results.items()},
        sub_grids, R=1,
    )

    # Cache the result (Step 5)
    first_round = CachedRoundResult(
        layouts=[r.layouts[-1] for r in block_results.values()],
        schedule=merged,
        initial_layout=current_layouts[plan.interacting_blocks[0]],
    )
    cache[plan.round_signature] = first_round

    # Replay for remaining d-1 rounds
    replays = replay_cached_round(first_round, plan.num_rounds - 1)
    return ECPhaseResult(rounds=[first_round] + replays, from_cache=False)
```

**Helper: `_route_gadget_phase`**:
```python
def _route_gadget_phase(plan, sub_grids, current_layouts, ...):
    # Use Step 7: route_gadget_with_transitions
    interacting = [sub_grids[name] for name in plan.interacting_blocks]
    result = route_gadget_with_transitions(
        interacting_blocks=interacting,
        ms_pairs=plan.ms_pairs_per_round,
        k=k,
        subgridsize=subgridsize,
        base_pmax_in=base_pmax_in,
        ec_final_layouts={name: current_layouts[name] for name in plan.interacting_blocks},
        ec_initial_layouts={name: current_layouts[name] for name in plan.interacting_blocks},
    )
    return result
```

---

### A3. Wire into `run_single_gadget_config()`

**File**: `best_effort_compilation_WISE.py`
**Change**: Replace the `_route_and_simulate()` call (lines 710-714) with `route_full_experiment()`.

**Current code** (lines 697-730):
```python
# 4. Route and simulate using existing pipeline
exec_time, results, reconfigTime, comp_time = _route_and_simulate(
    ideal=ideal,
    m_traps=m_traps,
    n_traps=n_traps,
    ...
)
```

**New code**:
```python
# 4. Route using phase-aware pipeline
from .gadget_routing import route_full_experiment
full_result = route_full_experiment(
    qec_meta=qec_meta,
    gadget=gadget,
    qubit_allocation=qubit_allocation,
    k=trap_capacity,
    subgridsize=(subgrid_width, subgrid_height, subgrid_increment),
    base_pmax_in=base_pmax_in,
    lookahead=lookahead,
    max_inner_workers=max_inner_workers,
    stop_event=stop_event,
    progress_callback=routing_config.progress_callback if routing_config else None,
)

# 5. Compute execution time from schedule
exec_time = full_result.total_exec_time
reconfigTime = full_result.total_reconfig_time

# 6. Noise injection + decode
# NOTE: This is the tricky part. The old pipeline uses TrappedIonExperiment
# which expects a compiled RoutedCircuit from TrappedIonCompiler. The new
# pipeline produces schedules directly. Two options:
#   Option A: Build a synthetic RoutedCircuit from the schedule data
#   Option B: Compute timing analytically and inject noise directly
# See section A4 for details.
```

---

### A4. Timing & Noise Integration

The trickiest part of the integration is going from schedules → noise → logical error rates.

**Current pipeline path**:
```
ideal stim.Circuit
  → compiler.compile(ideal)     # decompose → map → route → schedule
  → compiled.scheduled           # has total_duration, all_operations
  → TrappedIonExperiment.apply_hardware_noise()  # uses operations' individual fidelities
  → noisy stim.Circuit
  → decode
```

**New pipeline path (proposed)**:
```
QECMetadata + gadget
  → route_full_experiment()      # produces schedules + timing per phase
  → _compute_schedule_timing()   # schedule passes → per-ion heating + total time
  → _build_noise_circuit()       # inject noise into ideal stim circuit
  → decode
```

**Option A (Recommended): Analytic timing + direct noise injection**

We already have all the timing constants in `qccd_operations.py`:
- `row_swap_time` = Move + Merge + CrystalRotation + Split + Move
- `col_swap_time` = 2×JunctionCrossing + (4×JunctionCrossing + Move)×2
- `Split.SPLITTING_TIME` per reconfig overhead

From the schedule, we can compute:
1. **Total execution time** = Σ(reconfig_time per phase + MS gate time per round)
2. **Per-ion heating** = Σ(swap_heating × number_of_swaps_involving_ion)
3. **Noise model** = heating → motional error → gate infidelity

This lets us build a noisy stim circuit without going through the full
`TrappedIonCompiler`/`TrappedIonExperiment` stack.

**New helper**:
```python
def compute_schedule_timing(
    schedules: List[List[Dict[str, Any]]],
    k: int,
) -> Tuple[float, float, Dict[int, float]]:
    """Compute total execution time and per-ion heating from schedules.

    Returns (total_exec_time, total_reconfig_time, per_ion_heating).
    """
    row_swap_time = (Move.MOVING_TIME + Merge.MERGING_TIME +
                     CrystalRotation.ROTATION_TIME + Split.SPLITTING_TIME +
                     Move.MOVING_TIME)
    col_swap_time = (2 * JunctionCrossing.CROSSING_TIME +
                     (4 * JunctionCrossing.CROSSING_TIME + Move.MOVING_TIME) * 2)

    total_time = 0.0
    # ... iterate schedule passes, sum timing per H/V pass ...
    return total_time, reconfig_time, per_ion_heating
```

**Option B (Fallback): Synthetic RoutedCircuit**

If the noise model proves too complex to replicate analytically, we can
construct a minimal `RoutedCircuit` with synthetic `GlobalReconfigurations`
operations whose timing matches the schedule. This plugs into the existing
`TrappedIonExperiment.apply_hardware_noise()` path unchanged.

**Decision**: Start with Option A. Fall back to Option B if noise accuracy
diverges from the old pipeline on d=2 benchmarks.

---

### A5. New Dataclass: `FullExperimentResult`

**File**: `gadget_routing.py`

```python
@dataclass
class FullExperimentResult:
    """Complete routing result for a fault-tolerant gadget experiment."""
    phase_results: List[Any]                    # PhaseResult per phase
    total_schedule: List[List[Dict[str, Any]]]  # Concatenated schedules
    total_exec_time: float                       # Total execution time (μs)
    total_reconfig_time: float                   # Total reconfig time (μs)
    sub_grids: Dict[str, BlockSubGrid]           # Per-block allocations
    per_ion_heating: Dict[int, float]            # Per-ion motional heating
    cached_phases: int                           # Number of phases served from cache
    total_phases: int                            # Total number of phases
```

---

## 3. Work Stream B — Reconfig Merge Optimisation

### Location

**Inside `ionRoutingWISEArch()`** in `qccd_WISE_ion_route.py`, as requested.

### The Optimisation

**Observation**: The routing loop produces the pattern:
```
GR_0 → [1Q ops] → [MS gates] → barrier →
GR_1 → [1Q ops] → [MS gates] → barrier → ...
```

When the ions involved in MS round `i` are NOT moved by `GR_{i+1}`, we can
reorder to execute `GR_{i+1}` before `MS_i`, then merge consecutive GRs.

### Algorithm

```python
def _try_merge_reconfigs(
    idx: int,
    tiling_steps: List,
    parallelPairs: List[List[Tuple[int, int]]],
    oldArrangementArr: np.ndarray,
) -> Optional[Tuple[List[Dict], int]]:
    """Check if current MS round's ions are unmoved by the next GR.

    If so, return (merged_schedule, num_merged_rounds) where the caller
    should apply the merged GR and then execute all MS rounds in sequence.

    Returns None if merging is not possible.
    """
    # 1. Identify ions in current MS round
    current_ms_ions = set()
    for a, b in parallelPairs[idx]:
        current_ms_ions.add(a)
        current_ms_ions.add(b)

    # 2. Peek at next round's layout_after
    #    (available from tiling_steps lookahead)
    if idx + 1 >= len(parallelPairs):
        return None

    next_layout = tiling_steps[0][0]  # layout_after for next round

    # 3. Check: are current_ms_ions in the same positions in next_layout
    #    as they are in the current layout?
    for ion in current_ms_ions:
        # Find ion's position in current layout
        curr_pos = _find_ion_position(oldArrangementArr, ion)
        next_pos = _find_ion_position(next_layout, ion)
        if curr_pos != next_pos:
            return None  # Ion moved → can't merge

    # 4. Merge schedules
    #    Current GR's schedule + next GR's schedule → single merged schedule
    current_schedule = ...  # from current tiling_step
    next_schedule = ...     # from next tiling_step
    merged = _merge_reconfig_schedules(current_schedule, next_schedule)

    return merged, 2  # merged 2 GRs
```

### Schedule Merging Logic

When merging two GR schedules:

```python
def _merge_reconfig_schedules(
    sched_a: List[Dict[str, Any]],
    sched_b: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge two consecutive GR schedules into one.

    Strategy:
    1. Separate H passes and V passes from both schedules
    2. Concatenate: all H passes from A, then all H passes from B
    3. Concatenate: all V passes from A, then all V passes from B
    4. Re-index passes to maintain odd-even parity

    Since the two GRs operated on different ion subsets (the current
    MS ions didn't move in the second GR), their swaps are independent
    and can be freely interleaved.
    """
    h_passes_a = [p for p in sched_a if p["phase"] == "H"]
    v_passes_a = [p for p in sched_a if p["phase"] == "V"]
    h_passes_b = [p for p in sched_b if p["phase"] == "H"]
    v_passes_b = [p for p in sched_b if p["phase"] == "V"]

    # Merge H passes: interleave by parity
    merged_h = _interleave_by_parity(h_passes_a, h_passes_b)
    # Merge V passes: interleave by parity
    merged_v = _interleave_by_parity(v_passes_a, v_passes_b)

    return merged_h + merged_v
```

### Parity-Aware Interleaving

```python
def _interleave_by_parity(
    passes_a: List[Dict[str, Any]],
    passes_b: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Interleave two sets of passes maintaining odd-even parity.

    Odd-even rule: a horizontal swap at column c can only fire in a
    pass where c % 2 == pass_index % 2.

    Since passes_a and passes_b come from independent GRs operating
    on disjoint ion sets, swaps from different GRs at the same column
    parity can share the same pass.

    Algorithm:
    1. Group all swaps by their parity (even/odd column for H, even/odd row for V)
    2. Merge swaps of the same parity from both sources into the same pass
    3. Output passes in even-odd-even-odd order
    """
    even_swaps = []  # (row, col) tuples from both sources with even parity
    odd_swaps = []

    for pass_info in passes_a + passes_b:
        h_swaps = pass_info.get("h_swaps", [])
        v_swaps = pass_info.get("v_swaps", [])
        swaps = h_swaps or v_swaps
        phase = pass_info["phase"]

        for (r, c) in swaps:
            parity_key = c if phase == "H" else r
            if parity_key % 2 == 0:
                even_swaps.append((r, c))
            else:
                odd_swaps.append((r, c))

    result = []
    if even_swaps:
        result.append({
            "phase": passes_a[0]["phase"] if passes_a else passes_b[0]["phase"],
            "h_swaps": even_swaps if passes_a[0]["phase"] == "H" else [],
            "v_swaps": even_swaps if passes_a[0]["phase"] == "V" else [],
        })
    if odd_swaps:
        result.append({
            "phase": passes_a[0]["phase"] if passes_a else passes_b[0]["phase"],
            "h_swaps": odd_swaps if passes_a[0]["phase"] == "H" else [],
            "v_swaps": odd_swaps if passes_a[0]["phase"] == "V" else [],
        })
    return result
```

### Integration into `ionRoutingWISEArch`

**Location**: After `_apply_layout_as_reconfiguration` but before the MS gate execution loop.

```python
# After line ~1680 (layout_after, schedule, solved_pairs = tiling_step.pop(0))
# and before the MS gate execution loop

# --- Reconfig merge optimisation ---
# Check if we can merge this GR with the next one
merge_count = 0
merged_ms_rounds = [idx]

while True:
    next_idx = idx + merge_count + 1
    if next_idx >= len(toMoves):
        break

    # Check: are current accumulated MS ions unmoved by next GR?
    all_ms_ions = set()
    for midx in merged_ms_rounds:
        for a, b in parallelPairs[midx]:
            all_ms_ions.add(a)
            all_ms_ions.add(b)

    # Peek at next round's target layout
    # (need the next tiling_step's layout_after)
    next_layout = _peek_next_layout(tiling_steps, tiling_step)
    if next_layout is None:
        break

    if _ions_unmoved(all_ms_ions, oldArrangementArr, next_layout):
        # Can merge! Apply the next GR now, defer MS gates
        next_layout_after, next_schedule, next_solved = tiling_step.pop(0)
        merged_schedule = _merge_reconfig_schedules(schedule, next_schedule)
        # Apply merged GR
        oldArrangementArr = _apply_layout_as_reconfiguration(
            arch, wiseArch, oldArrangementArr, newArrangementArr,
            next_layout_after, allOps, merged_schedule,
        )
        merged_ms_rounds.append(next_idx)
        merge_count += 1
    else:
        break

# Now execute ALL deferred MS rounds
for ms_idx in merged_ms_rounds:
    _execute_ms_round(ms_idx, toMoves, solved_pairs, allOps, operationsLeft)
```

### Expected Speedup

- Each GR has fixed overhead: `Split.SPLITTING_TIME` (~10μs typical)
- Each merge eliminates one GR overhead: saves ~10μs per merged pair
- For d=3 surface code with 8 MS rounds: up to 7 possible merges = ~70μs saving
- Relative to total exec time (~500-1000μs for d=3): 7-14% improvement
- Larger codes benefit more: d=5 has ~16 MS rounds → up to ~150μs saving

### Correctness Invariant

The merge is safe if and only if: **every ion involved in the deferred MS gates
occupies the same grid position before and after the merged GR**. The check
`_ions_unmoved()` verifies this by comparing positions in `oldArrangementArr`
vs `next_layout_after`.

### Edge Cases

1. **First round (idx=0)**: `initial_placement=True` GR has special timing. Don't merge with it.
2. **Cached blocks**: When `tiling_steps_from_cache=True`, the schedule may be `None`. Can still merge by using the heuristic reconfig.
3. **1Q gates between GRs**: 1Q gates on the deferred MS ions must also be deferred. Check that 1Q gate ions don't overlap with deferred MS ions, or defer the 1Q gates too.

---

## 4. Work Stream C — SAT Direction Constraint Experiment

### Goal

Empirically determine the feasibility and impact of adding a monotonic
direction constraint to the SAT solver. The constraint says: once an ion
moves left/right (or up/down), it cannot reverse direction.

### Experimental Design

**Script**: `_experiment_sat_direction.py` (new top-level diagnostic)

1. **Baseline**: Run the existing SAT solver on d=2, d=3 instances and record:
   - Solve time
   - Displacement D*
   - Number of direction reversals per ion (post-hoc measurement)

2. **With hard constraint**: Add monotonic direction clauses and measure:
   - UNSAT rate
   - Solve time (for SAT instances)
   - D* change
   - Impact on schedule length

3. **With soft constraint (MaxSAT)**: Add direction reversal as weighted soft clauses:
   - Reversal penalty weight
   - Number of reversals allowed
   - D* and solve time

### SAT Encoding

**New variables** (added to `_wise_build_structural_cnf()` in
`qccd_SAT_WISE_odd_even_sorter.py`):

```python
# For each ion i, pass p: did ion i move right/left/up/down?
moved_right = {}   # (ion, round, pass) -> var_id
moved_left = {}
moved_up = {}
moved_down = {}

# Cumulative: has ion i ever moved right/left up to pass p?
ever_right = {}    # (ion, round, pass) -> var_id
ever_left = {}
ever_up = {}
ever_down = {}
```

**Clauses**:
```python
# 1. Define moved_right[i, r, p]:
#    moved_right[i,r,p] <-> OR(s_h[r,p,k,j] AND a[r,p,k,j,i]) for all (k,j)
#    (ion i was at position (k,j) and horizontal swap (k,j)↔(k,j+1) fired)

# 2. Cumulative chain:
#    ever_right[i,r,0] <-> moved_right[i,r,0]
#    ever_right[i,r,p] <-> ever_right[i,r,p-1] OR moved_right[i,r,p]

# 3. Monotonicity:
#    ever_right[i,r,p] -> NOT moved_left[i,r,p'] for all p' > p
#    ever_left[i,r,p]  -> NOT moved_right[i,r,p'] for all p' > p
#    (and similarly for up/down)
```

**Complexity estimate** (d=3, 17 ions, R=4, P=6):
- New variables: 4 × 17 × 4 × 6 + 4 × 17 × 4 × 6 = 3,264
- New clauses: ~5,000 (linking + monotonicity)
- Existing clause count: ~50,000-100,000
- **Overhead: ~5-10% more clauses**

### Implementation Steps

1. **Step C1**: Write post-hoc analyser
   - Given a solved SAT model, count direction reversals per ion
   - Output: histogram of reversals, identify which ions reverse most
   - **No SAT changes needed** — just reads the model

2. **Step C2**: Add optional hard constraint
   - New parameter `direction_constraint: bool = False` on `_wise_build_structural_cnf()`
   - When enabled, add the auxiliary variables and monotonicity clauses
   - Gated behind a flag so existing behaviour is unchanged

3. **Step C3**: Add soft constraint variant
   - When `direction_constraint_soft: bool = True`, add reversals as weighted
     soft clauses in the WCNF (for MaxSAT)
   - Weight: configurable, default = half the weight of boundary soft clauses

4. **Step C4**: Run experiment script
   - Grid: d ∈ {2, 3}, k ∈ {2}, subgridsize options
   - For each: baseline, hard constraint, soft constraint (weight sweep)
   - Collect: UNSAT rate, solve time, D*, schedule length, reversal count
   - Output: CSV + summary table

### Expected Results

Based on analysis of the odd-even sorting network:
- **Hard constraint**: HIGH UNSAT rate (>50%) for d≥3 because the sorting network
  inherently requires bidirectional movement
- **Soft constraint**: Should always find a solution. Interesting question is
  whether penalizing reversals reduces schedule length or increases D*
- **d=2 might work**: With only ~9 ions on a small grid, there may be enough
  slack for monotonic movement

### Risk

- Adding direction constraint variables increases memory proportionally.
  For d=5 (~49 ions), memory could be significant.
- If UNSAT rate is too high, the whole approach is infeasible.
- The experiment will answer this definitively before any production code changes.

---

## 5. Dependency Graph & Sequencing

```
A1 (decompose_into_phases) ←── depends on existing gadget_routing.py functions
    │
    v
A2 (route_full_experiment) ←── uses A1 + existing Steps 2-7
    │
    v
A3 (wire into run_single_gadget_config) ←── uses A2
    │
    v
A4 (timing & noise) ←── uses A3, provides decode path
    │
    v
A5 (FullExperimentResult) ←── dataclass, needed by A2

B (reconfig merge) ←── independent of A, inside ionRoutingWISEArch
    │
    ├── B benefits BOTH old and new pipelines
    └── can be developed/tested independently

C (SAT direction experiment) ←── independent of A and B
    │
    ├── C1 (post-hoc analyser) ←── no SAT changes
    ├── C2 (hard constraint) ←── modifies _wise_build_structural_cnf
    ├── C3 (soft constraint) ←── extends C2
    └── C4 (experiment script) ←── uses C1-C3
```

### Recommended Order

1. **A5** — Define `FullExperimentResult` dataclass (5 min)
2. **A1** — `decompose_into_phases()` (1-2 hours)
3. **A2** — `route_full_experiment()` orchestrator (2-3 hours)
4. **A3** — Wire into `run_single_gadget_config()` (30 min)
5. **A4** — Timing/noise integration (1-2 hours, most uncertain)
6. **Test A** — Run d=2 TransversalCNOT end-to-end, compare timing with old pipeline

In parallel (Work Stream B):
7. **B** — Reconfig merge in `ionRoutingWISEArch` (2-3 hours)
8. **Test B** — Run d=2 baseline, measure timing improvement

After A+B verified:
9. **C1** — Post-hoc direction analyser (1 hour)
10. **C4** — Run baseline measurements (30 min)
11. **C2** — Hard direction constraint (2-3 hours)
12. **C3** — Soft direction constraint (1 hour, extends C2)
13. **C4** — Full experiment suite (1 hour including run time)

---

## 6. File Change Inventory

### Work Stream A

| File | Changes | Risk |
|------|---------|------|
| `gadget_routing.py` | Add `decompose_into_phases()`, `route_full_experiment()`, `FullExperimentResult`, helper functions | LOW — additive only |
| `best_effort_compilation_WISE.py` | Replace `_route_and_simulate()` call in `run_single_gadget_config()` with `route_full_experiment()` | MEDIUM — changes main path |
| `gadget_routing.py` | Add `compute_schedule_timing()` for analytic timing | LOW — additive |

### Work Stream B

| File | Changes | Risk |
|------|---------|------|
| `qccd_WISE_ion_route.py` | Add merge logic inside `ionRoutingWISEArch()` main loop (after GR, before MS) | MEDIUM — modifies hot path |
| `qccd_WISE_ion_route.py` | Add `_merge_reconfig_schedules()`, `_ions_unmoved()`, `_interleave_by_parity()` helpers | LOW — additive |

### Work Stream C

| File | Changes | Risk |
|------|---------|------|
| `qccd_SAT_WISE_odd_even_sorter.py` | Add direction constraint variables and clauses in `_wise_build_structural_cnf()` (gated behind flag) | LOW — flag-gated |
| New: `_experiment_sat_direction.py` | Diagnostic script | NONE — standalone |

### Files NOT changed

| File | Reason |
|------|--------|
| `trapped_ion_compiler.py` | Integration bypasses compiler stack entirely |
| `routing_config.py` | No new config fields needed for MVP |
| `qccd_operations.py` | Timing constants read-only; no structural changes |
| Gadget classes | `get_phase_pairs()` deferred — transversal fallback sufficient for now |

---

## 7. Testing Strategy

### Unit Tests

| Test | What it validates |
|------|-------------------|
| `test_decompose_phases_transversal_cnot` | Phase decomposition for TransversalCNOT d=2: correct phase count, types, active blocks |
| `test_decompose_phases_knill_ec` | KnillEC has 3 blocks, multiple phases with different active sets |
| `test_route_full_experiment_d2` | End-to-end routing for TransversalCNOT d=2: produces valid schedules, timing > 0 |
| `test_ec_phase_caching` | Identical EC phases produce same `round_signature`, cache hit on second call |
| `test_reconfig_merge_ions_unmoved` | `_ions_unmoved()` correctly detects unmoved ions |
| `test_reconfig_merge_ions_moved` | `_ions_unmoved()` correctly rejects when ions move |
| `test_merge_reconfig_schedules` | Merged schedule has correct H/V pass structure with parity |
| `test_direction_reversal_count` | Post-hoc analyser correctly counts reversals in a known model |

### Integration Tests

| Test | What it validates |
|------|-------------------|
| `test_full_pipeline_d2_timing` | `run_single_gadget_config()` with phase-aware routing produces exec_time within 2× of old pipeline |
| `test_full_pipeline_d3_timing` | Same for d=3 (larger, slower) |
| `test_reconfig_merge_speedup` | With merge enabled, d=2 exec_time is ≤ baseline |
| `test_old_pipeline_unchanged` | Old `_route_and_simulate()` path still works (regression) |

### Validation Criteria

- **Timing**: Phase-aware pipeline timing should be within 1.5× of old pipeline for d=2 (better for d≥3 due to caching)
- **Correctness**: Decoded logical error rate from phase-aware pipeline must match old pipeline within statistical uncertainty (±2σ for 100k shots)
- **No regression**: Old pipeline (`_route_and_simulate` path) must continue to produce identical results

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Noise model accuracy diverges between analytic timing and old pipeline | MEDIUM | HIGH | Use Option B fallback (synthetic RoutedCircuit) |
| `ProcessPoolExecutor` per-block parallel routing hits macOS spawn issues | HIGH | MEDIUM | Default to `parallel=False` on macOS; test with `max_workers=1` |
| EC round caching produces incorrect layouts when initial placement varies | LOW | HIGH | Assert layout equality before and after cached replay |
| Reconfig merge misidentifies "unmoved" ions due to floating point or spectator issues | LOW | MEDIUM | Use integer ion-index comparison only; ignore spectators |
| SAT direction constraint causes widespread UNSAT | HIGH | LOW | Experiment-only; no production code depends on it |
| Phase decomposition logic incorrect for KnillEC (3-block, complex phases) | MEDIUM | MEDIUM | Start with TransversalCNOT (2-block, simple); add KnillEC after validation |
| `_patch_and_route` called from both old and new pipelines causes state conflicts | LOW | HIGH | New pipeline creates its own `QCCDWiseArch` instances; no shared state |

---

## Appendix: Key Code References

| Concept | File | Line(s) |
|---------|------|---------|
| Old monolithic pipeline | `best_effort_compilation_WISE.py` | L228 (`_route_and_simulate`) |
| Current gadget entry point | `best_effort_compilation_WISE.py` | L622 (`run_single_gadget_config`) |
| _route_and_simulate call | `best_effort_compilation_WISE.py` | L710-714 |
| All building blocks | `gadget_routing.py` | L1-1303 |
| Routing main loop | `qccd_WISE_ion_route.py` | L1186 (`ionRoutingWISEArch`) |
| Apply reconfig as GR op | `qccd_WISE_ion_route.py` | L1140 (`_apply_layout_as_reconfiguration`) |
| GR timing calculation | `qccd_operations.py` | L242 (`_runOddEvenReconfig`) |
| SAT formula builder | `qccd_SAT_WISE_odd_even_sorter.py` | L978 (`_wise_build_structural_cnf`) |
| Patch-based Level 1 slicer | `qccd_WISE_ion_route.py` | L370 (`_patch_and_route`) |
| Patch schedule merge | `qccd_WISE_ion_route.py` | L340 (`_merge_patch_schedules`) |
| Compiler route method | `trapped_ion_compiler.py` | L549 |
| WISERoutingConfig | `routing_config.py` | L198 |
