# Critical Context for SAT Routing Fix

## File Paths
- WISE route: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py` (3355 lines)
- Gadget routing: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py` (~3264 lines)
- Baseline commit: `243a188` (working SAT routing, no gadget support)
- Current HEAD: `41fb858` (broken MS gate execution)

## What I've Verified

### OLD CODE (243a188) MS Gate Execution (lines 1858-1943):
```python
while idx < len(toMoves):
    # 4c) Execute one parallel round of two-qubit MS gates
    for op in toMoves[idx]:
        if any(all((ion.idx in p) for ion in op.ions) for p in solved_pairs):
            trap = op.getTrapForIons()
            if trap is not None:
                op.setTrap(trap); op.run(); allOps.append(op); operationsLeft.remove(op)
                ms_gates_executed += 1
            else:
                # FALLBACK: try op.ions[0].parent
                trap = op.ions[0].parent if op.ions else None
                if trap is not None:
                    op.setTrap(trap)
                    try: op.run(); allOps.append(op); operationsLeft.remove(op); ms_gates_executed += 1
                    except ValueError as e: raise
    if not tiling_step: break
    if not any(t[2] for t in tiling_step): tiling_step = []; break
    layout_after, schedule, solved_pairs = tiling_step.pop(0)
    # Apply reconfig for next tiling step within same round...
```

### NEW CODE MS Gate Execution (`_execute_ms_gates` at line 2220):
```python
def _execute_ms_gates(round_idx, solved_pairs):
    remaining = [frozenset(p) for p in solved_pairs]
    two_q_ops = [op for op in operationsLeft if len(op.ions) == 2]
    two_q_ops.sort(key=lambda op: getattr(op, '_tick_epoch', 0))
    to_execute = []
    for op in two_q_ops:
        if not remaining: break
        ion_set = frozenset(ion.idx for ion in op.ions)
        for i, rp in enumerate(remaining):
            if ion_set == rp:  # EXACT frozenset EQUALITY
                to_execute.append(op); remaining.pop(i); break
    # ... then execute to_execute
    for op in to_execute:
        trap = op.getTrapForIons()
        if trap is not None:
            op.setTrap(trap); op.run(); allOps.append(op); operationsLeft.remove(op)
        else:
            # SKIPS the gate entirely (no fallback!)
            logger.warning("...SKIPPING...")
```

### NEW CODE Execution Loop (line 2310+):
```python
routing_steps = sorted(routing_steps, key=lambda s: s.ms_round_index)
for ms_round_idx, round_steps_iter in _groupby(routing_steps, key=lambda s: s.ms_round_index):
    round_steps = list(round_steps_iter)
    # 4a) Apply reconfig from first step
    _apply_layout_as_reconfiguration(...)
    # 4b) Drain single-qubit ops
    _drain_single_qubit_ops(ms_round_idx) 
    # 4c) Execute MS gates
    ms_gates_executed, _unmatched = _execute_ms_gates(ms_round_idx, first_step.solved_pairs)
    # Handle subsequent steps for same round
    for subsequent_step in round_steps[1:]:
        _apply_layout_as_reconfiguration(...)
        _drain_single_qubit_ops(ms_round_idx); barriers.append(len(allOps))
        _execute_ms_gates(ms_round_idx, subsequent_step.solved_pairs)
```

## 3 ROOT CAUSES

### RC1: `_execute_ms_gates` frozenset exact match drops MS gates
The old code used `any(all((ion.idx in p) for ion in op.ions) for p in solved_pairs)` which checks if EACH ion is CONTAINED in ANY pair. When pairs are tuples like `(3, 7)`, this works because `3 in (3, 7)` is True.

The new code uses `frozenset(ion.idx for ion in op.ions) == rp` which requires EXACT set equality. This SHOULD work if solved_pairs contain the same ion indices as the operations.

**ACTUAL ROOT CAUSE**: The solved_pairs from `route_full_experiment_as_steps` → `_route_round_sequence` use ion indices from the **sub-grid layout array**, which may differ from the global `Ion.idx` values when block sub-grids are involved. The `_route_round_sequence` takes `parallelPairs` (which DO have correct global Ion.idx), but the SAT solver maps them through the layout array. The `solved_pairs` returned are the pairs found by `_patch_and_route` which should be subsets of the input `parallelPairs` — so they should have the same indices. BUT `_merge_phase_pairs` in `route_full_experiment_as_steps` (gadget_routing.py ~L2860) MODIFIES the pairs before sending to `_route_round_sequence`, breaking the 1:1 correspondence.

### RC2: `_merge_phase_pairs` in route_full_experiment_as_steps changes round count
This merges adjacent disjoint-ion rounds, reducing pair count. But `phase_pair_counts` in `ionRoutingGadgetArch` uses ORIGINAL lengths from `_build_plans_from_compiler_pairs`. So the number of RoutingSteps returned differs from what the execution loop expects.

### RC3: No getTrapForIons() fallback
Old code tried `op.ions[0].parent` when getTrapForIons() returned None. New code skips the gate entirely.

## route_full_experiment_as_steps location and structure
- Defined at gadget_routing.py ~L2484
- Calls `_route_round_sequence` per phase
- Uses `_merge_phase_pairs` at ~L2860 to merge rounds (BAD)
- Returns `List[RoutingStep]` with ms_round_index set using cumulative round_cursor

## ionRoutingGadgetArch flow:
1. Builds parallelPairs from operations (L2953+)
2. Reads phase_pair_counts from QECMetadata (L2973+)
3. Calls _build_plans_from_compiler_pairs to map phases to pairs
4. Calls route_full_experiment_as_steps (L3175) → gets all_routing_steps
5. Passes all_routing_steps as _precomputed_routing_steps to ionRoutingWISEArch (L3334)
6. ionRoutingWISEArch receives them, skips building its own routing_steps, uses provided ones
7. Execution loop groups by ms_round_index, calls _execute_ms_gates

## _route_round_sequence location
- Defined at qccd_WISE_ion_route.py ~L1300-1700
- Creates RoutingStep objects with solved_pairs from _patch_and_route output
- solved_pairs come from the SAT solver output, should match input parallelPairs entries

## Key insight about _build_parallel_pairs
- Located at qccd_WISE_ion_route.py ~L1730
- Called TWICE: once in ionRoutingGadgetArch (to get parallelPairs for phase splitting) and once in ionRoutingWISEArch (to get toMoves for execution)
- toMoves maps round_idx → list of Operation objects
- parallelPairs maps round_idx → list of (ion_a, ion_b) tuples
- The ion indices in parallelPairs MATCH Ion.idx in operations because they're derived from the same Operation objects

## COMPREHENSIVE FIX PLAN

### Fix 1: Restore toMoves-based MS gate execution
In `_execute_ms_gates`, when `toMoves` is available and `round_idx` is valid, use `toMoves[round_idx]` to get specific operations (like old code). Use `solved_pairs` only as confirmation filter.

### Fix 2: Remove _merge_phase_pairs from route_full_experiment_as_steps
At gadget_routing.py ~L2860, remove or skip the `_merge_phase_pairs()` call.

### Fix 3: Restore getTrapForIons() fallback
In `_execute_ms_gates`, when `getTrapForIons()` returns None, try `op.ions[0].parent`.

### Fix 4: Add ion index consistency diagnostic
Before passing _precomputed_routing_steps to ionRoutingWISEArch, verify solved_pairs indices match operation ion indices.

### Implementation order: Fix 1 → Fix 3 → Fix 2 → Fix 4
