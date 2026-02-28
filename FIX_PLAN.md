# Comprehensive Bug Fix Plan

All bugs and concerns from the GADGET_COMPILATION_SPEC audit.
Fixes listed in dependency order (leaf fixes first, then the fixes that depend on them).

---

## Fix 1: Bug E — `_apply_layout_as_reconfiguration` empty-cell guard

**Severity**: MEDIUM  
**File**: `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`  
**Function**: `_apply_layout_as_reconfiguration` (line ~1163)  
**Depends on**: Nothing — standalone leaf fix.

### Problem

The inner loop does:

```python
for d in range(wiseArch.n):
    for c in range(wiseArch.m * wiseArch.k):
        ionidx = int(layout_after[d][c])
        newArrangementArr[d][c] = ionidx
        old_ionidx = int(oldArrangementArr[d][c])
        trap = arch.ions[old_ionidx].parent          # ← KeyError if old_ionidx == 0
        newArrangement[trap].append(arch.ions[ionidx]) # ← KeyError if ionidx == 0
```

`arch.ions` is a dynamically-built dict keyed by `ion.idx` (starting at 1).
Index `0` means "empty cell" in the arrangement array. Accessing `arch.ions[0]`
raises `KeyError`.

In the non-gadget flow, the grid starts fully populated so every cell ≠ 0.
But after gadget↔EC transitions with merged sub-grids re-embedded into the
global layout, cells CAN go from empty→occupied or occupied→empty.

### Fix

Replace the inner loop body with:

```python
for d in range(wiseArch.n):
    for c in range(wiseArch.m * wiseArch.k):
        ionidx = int(layout_after[d][c])
        newArrangementArr[d][c] = ionidx

        old_ionidx = int(oldArrangementArr[d][c])

        # Determine which physical trap owns grid cell (d, c).
        # Primary: look up the ion that was in this cell before.
        # Fallback: derive from grid geometry (trap = row * m + col // k).
        if old_ionidx != 0:
            trap = arch.ions[old_ionidx].parent
        else:
            # Empty cell → map grid position to physical trap.
            trap_col = c // wiseArch.k
            trap_linear = d * wiseArch.m + trap_col
            trap = arch._manipulationTraps[trap_linear]  # ← needs sorted list

        # Only append real ions to the new arrangement.
        if ionidx != 0:
            newArrangement[trap].append(arch.ions[ionidx])
```

**Important**: `arch._manipulationTraps` must be in the same row-major order
as the grid.  The code already sorts traps by `(pos[1], pos[0])` in
`ionRoutingWISEArch` at line ~1808.  We need to either:
- Pass the sorted trap list into `_apply_layout_as_reconfiguration`, OR
- Sort inside the function (slightly wasteful but safe).

**Recommended**: Add a `sorted_traps` parameter with a default of `None`.
When `None`, sort internally.  The caller in the execution loop can pass
the pre-sorted list for efficiency.

### Validation

- Run d=2 TransversalCNOT gadget compilation.  Before the fix, this crashes
  with KeyError on the transition reconfig.  After the fix, the transition
  reconfig should execute without error.
- Add unit test: create a layout array with some 0 cells, call
  `_apply_layout_as_reconfiguration`, verify no crash and correct
  `newArrangement` mapping.

---

## Fix 2: Bug D — `derive_gadget_ms_pairs` phase index

**Severity**: MEDIUM (latent — only affects KnillEC/CSSSurgery, not TransversalCNOT)  
**File**: `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py`  
**Function**: `decompose_into_phases` (line ~921)  
**Depends on**: Nothing — standalone leaf fix.

### Problem

```python
for i, phase in enumerate(qec_meta.phases):
    # ...
    elif phase_type == "gadget":
        ms_pairs = derive_gadget_ms_pairs(
            gadget, i, qubit_allocation,    # ← i is index into qec_meta.phases (global)
            active_blocks, qubit_to_ion,
        )
```

`derive_gadget_ms_pairs` passes `i` (the global phase index) as the
`phase_index` parameter.  But `gadget.get_phase_pairs(phase_index)` (if it
existed) expects a **gadget-phase-local** counter (0 for the first gadget
phase, 1 for the second, etc.).

For TransversalCNOT (1 gadget phase), both counters are the same.
For KnillEC (2 gadget phases), phase 3 and phase 7 in the global list
should map to gadget-local indices 0 and 1, but currently they'd get 3 and 7.

Currently no gadget class implements `get_phase_pairs`, so the `hasattr`
check falls through to the transversal fallback.  But this is a latent
bug that will bite when `get_phase_pairs` is implemented.

### Fix

Track a gadget-phase-local counter:

```python
_gadget_phase_counter = 0  # ← add before the loop

for i, phase in enumerate(qec_meta.phases):
    # ...
    elif phase_type == "gadget":
        ms_pairs = derive_gadget_ms_pairs(
            gadget, _gadget_phase_counter, qubit_allocation,  # ← use local counter
            active_blocks, qubit_to_ion,
        )
        _gadget_phase_counter += 1
        # ... rest of plan construction
```

### Validation

- For TransversalCNOT: no behavioral change (1 gadget phase, counter is 0
  either way).
- For future multi-phase gadgets: add a test with a mock gadget that has
  `get_phase_pairs(0)` and `get_phase_pairs(1)` returning different pairs.
  Verify the correct pairs are used for each gadget phase.

---

## Fix 3: Bug C — BT propagation across routing windows

**Severity**: LOW (only matters for large d with small lookahead)  
**File**: `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`  
**Function**: `_route_round_sequence` (line ~1570-1600)  
**Depends on**: Nothing — standalone leaf fix.

### Problem

`_route_round_sequence` receives `initial_BTs` which may include a
return-round BT (the last entry in the BTs list, with ion-return pins).
On the first routing window, `BTs = initial_BTs` is used directly.
After the first `_patch_and_route` call, BTs are rebuilt entirely from
the tiling metadata (line ~1497-1568).  The return-round pins from
`initial_BTs` are lost.

For EC phases, the return-round BT ensures ions return to their starting
positions after all MS rounds.  If `lookahead < len(parallelPairs)`,
the return round is in a later window that no longer has the return pins.

### Fix

After rebuilding BTs from tiling metadata (line ~1568), check if
`initial_BTs` had a return-round entry (typically the last element with
a special structure), and merge those pins into the **last** BTs
entry for the current window:

```python
# After the BTs rebuild block (around line 1570):
if initial_BTs is not None and len(initial_BTs) > 0:
    # The return-round BT is the last entry in initial_BTs.
    # It carries ion-return pins that must persist across all windows.
    last_initial_bt = initial_BTs[-1]

    # If this is the LAST routing window (i.e., the return round
    # falls within this window), merge the return BT.
    # The return round is at index len(parallelPairs) - 1 relative
    # to window_start.
    return_round_idx_in_window = len(parallelPairs) - 1 - window_start
    if 0 <= return_round_idx_in_window < len(BTs):
        # Merge initial return-round pins into the rebuilt BTs
        for key, (bt_map, pairs) in last_initial_bt.items():
            if key not in BTs[return_round_idx_in_window]:
                BTs[return_round_idx_in_window][key] = (bt_map, pairs)
            else:
                existing_map, existing_pairs = BTs[return_round_idx_in_window][key]
                existing_map.update(bt_map)
                # Don't duplicate pairs
```

**Alternative simpler approach**: Always append the return-round pins to
the very last BTs entry in every window, with a "sticky" flag.  The
return-round BT has key `(0, 0)` and pins `{ion_idx: (r, c)}`.  We can
save this on first call and re-inject it into every subsequent window's
last BTs entry.

```python
# At the top of the function, extract the return-round BT if present:
_sticky_return_bt: Optional[Dict] = None
if initial_BTs is not None and len(initial_BTs) > 0:
    last_bt_entry = initial_BTs[-1]
    if last_bt_entry:
        _sticky_return_bt = last_bt_entry

# After each BTs rebuild, if this window's last round is the actual
# last round (return round), inject the sticky BT:
if _sticky_return_bt is not None:
    total_rounds = len(parallelPairs)  # including return round
    last_round_in_window = min(window_end, total_rounds) - 1
    if last_round_in_window == total_rounds - 1 and BTs:
        # This window contains the return round
        for key, val in _sticky_return_bt.items():
            BTs[-1][key] = val
```

### Validation

- Run d=5 TransversalCNOT with `lookahead=1` (forces multiple routing
  windows).  Verify that ions return to their starting positions after
  each EC phase.
- Compare layouts before and after EC phase — they should match.

---

## Fix 4: Bug A+B — Transition Reconfig via SAT-Based Routing (THE BIG FIX)

**Severity**: HIGH — root cause of the ion 28 bug  
**Files**:
- `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py`
- `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`

**Depends on**: Fix 1 (Bug E empty-cell guard) must be in place first.

### Problem Summary

When transitioning between phases (notably gadget→EC), ions may have moved
to completely different positions on the grid.  The current code either:
1. Uses `schedule=None` (heuristic odd-even reconfig) — insufficient for large moves.
2. Falls back to routing without BT when BT causes UNSAT — leaves ions in post-gadget positions.

Both paths can leave ions in the wrong traps, causing the ion 28 bug.

### Design: SAT-Based Transition Reconfig

Replace all `schedule=None` heuristic transitions with proper SAT-based
`_patch_and_route` calls.  The algorithm:

1. **Check block purity**: After a gadget phase, check each block's spatial
   slice in `current_layout` to see if all non-zero ions in that slice
   belong to that block.

2. **If all blocks are pure** (no cross-block ion mixing):
   - Use Level 1 per-block spatial slicing.
   - For each block, compute `_patch_and_route(current_block_layout, ...,
     BTs=return_bt)` where `return_bt` pins ions to their target positions
     in `cached_starting_layout`.
   - Merge the per-block reconfigs with `_merge_block_routing_steps`.

3. **If blocks are impure** (ions mixed across block boundaries — typical
   after gadget phases):
   - Use full-grid `_patch_and_route(current_layout, ..., BTs=return_bt)`
     where `return_bt` pins ALL ions to their target positions in
     `cached_starting_layout`.
   - BT pins will move ions toward their targets; multiple tiling steps
     handle long-range moves (ions move to subgrid boundary per tiling,
     then continue in next tiling).

4. **No silent fallback**: If `_patch_and_route` fails, **raise an error**.
   It should work with enough tilings — a failure indicates a real bug.

### Implementation: New Helper Function

Add a new helper in `gadget_routing.py`:

```python
def _compute_transition_reconfig_steps(
    current_layout: np.ndarray,
    target_layout: np.ndarray,
    wiseArch: "QCCDWiseArch",
    block_sub_grids: Dict[str, "BlockSubGrid"],
    ion_to_block: Dict[int, str],
    k: int,
    *,
    subgridsize: Tuple[int, int, int],
    base_pmax_in: int = 1,
    max_inner_workers: int | None = None,
    stop_event: Any = None,
) -> List["RoutingStep"]:
    """Compute SAT-based transition reconfig from current_layout to target_layout.

    Uses spatial slicing when possible (all ions in their correct block
    regions), otherwise falls back to full-grid _patch_and_route with BT pins.

    Returns a list of RoutingSteps that transition the layout.
    """
    from ..compiler.qccd_WISE_ion_route import (
        _patch_and_route,
        _route_round_sequence,
        RoutingStep,
        _merge_block_routing_steps,
    )

    if np.array_equal(current_layout, target_layout):
        return []

    # --- Check block purity ---
    blocks_pure = True
    for bname, sg in block_sub_grids.items():
        r0, c0, r1, c1 = sg.grid_region
        c0i, c1i = c0 * k, c1 * k
        block_slice = current_layout[r0:r1, c0i:c1i]
        for r in range(block_slice.shape[0]):
            for c in range(block_slice.shape[1]):
                ion_idx = int(block_slice[r, c])
                if ion_idx != 0:
                    owner = ion_to_block.get(ion_idx)
                    if owner != bname:
                        blocks_pure = False
                        break
            if not blocks_pure:
                break
        if not blocks_pure:
            break

    n_rows, n_cols = current_layout.shape

    if blocks_pure and len(block_sub_grids) > 1:
        # --- Per-block spatial slicing ---
        per_block_steps: Dict[str, List[RoutingStep]] = {}
        for bname, sg in block_sub_grids.items():
            r0, c0, r1, c1 = sg.grid_region
            c0i, c1i = c0 * k, c1 * k
            block_current = current_layout[r0:r1, c0i:c1i].copy()
            block_target = target_layout[r0:r1, c0i:c1i].copy()

            if np.array_equal(block_current, block_target):
                per_block_steps[bname] = []
                continue

            # Build BT pins: target_layout positions for all ions in block
            bt_map: Dict[int, Tuple[int, int]] = {}
            for r in range(block_target.shape[0]):
                for c in range(block_target.shape[1]):
                    ion_idx = int(block_target[r, c])
                    if ion_idx != 0:
                        bt_map[ion_idx] = (r, c)

            # Solve with BT: P_arr = [[]] (no MS gates, just reconfig)
            block_rows, block_cols = block_current.shape
            block_wise = QCCDWiseArch(n=block_rows, m=block_cols // k, k=k)
            bts = [{}, {(0, 0): bt_map}]  # empty MS round + return round
            p_arr = [[]]  # one empty round (reconfig only)

            block_steps_raw, _ = _route_round_sequence(
                block_current, block_wise, p_arr,
                lookahead=1,
                subgridsize=subgridsize,
                base_pmax_in=base_pmax_in,
                active_ions=[idx for idx in sg.ion_indices if idx != 0],
                initial_BTs=bts,
                stop_event=stop_event,
                max_inner_workers=max_inner_workers,
            )
            per_block_steps[bname] = list(block_steps_raw)

        # Merge per-block steps into global steps
        merged, _ = _merge_block_routing_steps(
            per_block_steps, current_layout, block_sub_grids, k,
        )
        return merged

    else:
        # --- Full-grid routing (blocks impure or single block) ---
        # Build BT pins: target_layout positions for all ions
        bt_map: Dict[int, Tuple[int, int]] = {}
        for r in range(target_layout.shape[0]):
            for c in range(target_layout.shape[1]):
                ion_idx = int(target_layout[r, c])
                if ion_idx != 0:
                    bt_map[ion_idx] = (r, c)

        bts = [{}, {(0, 0): bt_map}]
        p_arr = [[]]

        transition_steps, _ = _route_round_sequence(
            current_layout.copy(), wiseArch, p_arr,
            lookahead=1,
            subgridsize=subgridsize,
            base_pmax_in=base_pmax_in,
            active_ions=[
                int(current_layout[r, c])
                for r in range(n_rows)
                for c in range(n_cols)
                if int(current_layout[r, c]) != 0
            ],
            initial_BTs=bts,
            stop_event=stop_event,
            max_inner_workers=max_inner_workers,
        )
        return list(transition_steps)
```

### 4a: Replace EC Cache Replay Transition

**File**: `gadget_routing.py`, `route_full_experiment_as_steps`, line ~1505-1525

**Current code** (the problem):
```python
if not np.array_equal(current_layout, cached_starting_layout):
    phase_steps.append(RoutingStep(
        layout_after=np.array(cached_starting_layout, copy=True),
        schedule=None,      # ← heuristic, INSUFFICIENT
        solved_pairs=[],
        ms_round_index=0,
        ...
    ))
```

**Replace with**:
```python
if not np.array_equal(current_layout, cached_starting_layout):
    transition_steps = _compute_transition_reconfig_steps(
        current_layout=current_layout,
        target_layout=cached_starting_layout,
        wiseArch=wiseArch,
        block_sub_grids=block_sub_grids,
        ion_to_block=ion_to_block,
        k=k,
        subgridsize=subgridsize,
        base_pmax_in=base_pmax_in or 1,
        max_inner_workers=max_inner_workers,
        stop_event=stop_event,
    )
    # Transition steps carry ms_round_index=0 (reconfig only, no MS gates)
    for ts in transition_steps:
        ts.ms_round_index = 0
        ts.from_cache = False
        ts.can_merge_with_next = True
        ts.is_initial_placement = False
    phase_steps.extend(transition_steps)
    # Update current_layout to the end of transition
    if transition_steps:
        current_layout = transition_steps[-1].layout_after.copy()
    logger.info(
        "[PhaseSteps] phase=%d (EC cached): SAT-based transition "
        "reconfig: %d steps",
        phase_idx, len(transition_steps),
    )
```

### 4b: Fix Gadget Phase Exit BT Fallback

**File**: `gadget_routing.py`, `route_full_experiment_as_steps`, line ~1770-1800

**Current code** (the problem):
```python
except Exception:
    logger.warning(
        "[PhaseSteps] gadget phase %d: BT routing failed, "
        "retrying without exit BT", phase_idx,
    )
    phase_steps_raw, _mf = _route_round_sequence(
        ..., list(phase_pairs), ...  # no BT → ions don't return to EC positions
    )
    phase_steps = list(phase_steps_raw)
```

**Replace with**: Remove the silent BT fallback entirely.  If exit BT fails,
add an explicit post-gadget transition reconfig:

```python
try:
    phase_steps_raw, _mf = _route_round_sequence(
        ..., p_arr_for_solve, ..., initial_BTs=bts, ...
    )
    phase_steps = list(phase_steps_raw)
except Exception as exc:
    logger.warning(
        "[PhaseSteps] gadget phase %d: BT routing failed (%s), "
        "routing without BT + adding post-gadget transition reconfig",
        phase_idx, exc,
    )
    # Route gadget pairs without BT
    phase_steps_raw, _mf = _route_round_sequence(
        np.array(merged_layout, copy=True),
        merged_wise,
        list(phase_pairs),
        lookahead=min(lookahead, len(phase_pairs)),
        subgridsize=merged_sgs,
        base_pmax_in=base_pmax_in or 1,
        active_ions=merged_active,
        stop_event=stop_event,
        max_inner_workers=max_inner_workers,
        progress_callback=progress_callback,
    )
    phase_steps = list(phase_steps_raw)

    # Re-embed gadget results into global layout (same as success path)
    for step in phase_steps:
        global_after = np.array(current_layout, copy=True)
        for sg in interacting_sgs:
            region = block_regions[sg.block_name]
            r0_m, c0_m, r1_m, c1_m = region
            r0_g, c0_g, r1_g, c1_g = sg.grid_region
            c0_gi = c0_g * k
            rows = min(r1_m - r0_m, r1_g - r0_g)
            cols = min(c1_m - c0_m, (c1_g - c0_g) * k)
            global_after[
                r0_g:r0_g + rows, c0_gi:c0_gi + cols
            ] = step.layout_after[
                r0_m:r0_m + rows, c0_m:c0_m + cols
            ]
        step.layout_after = global_after

    # Add post-gadget transition reconfig to return ions to EC positions
    post_gadget_layout = phase_steps[-1].layout_after if phase_steps else current_layout
    ec_target = _reconstruct_ec_target(ec_initial_layouts, current_layout, block_sub_grids, k)

    transition_steps = _compute_transition_reconfig_steps(
        current_layout=post_gadget_layout,
        target_layout=ec_target,
        wiseArch=wiseArch,
        block_sub_grids=block_sub_grids,
        ion_to_block=ion_to_block,
        k=k,
        subgridsize=subgridsize,
        base_pmax_in=base_pmax_in or 1,
        max_inner_workers=max_inner_workers,
        stop_event=stop_event,
    )
    # Assign transition steps to last MS round of the gadget phase
    for ts in transition_steps:
        ts.ms_round_index = len(phase_pairs) - 1
    phase_steps.extend(transition_steps)
```

Need a small helper to reconstruct the EC target from `ec_initial_layouts`:

```python
def _reconstruct_ec_target(
    ec_initial_layouts: Dict[str, np.ndarray],
    global_layout: np.ndarray,
    block_sub_grids: Dict[str, "BlockSubGrid"],
    k: int,
) -> np.ndarray:
    """Build a full-grid target from per-block EC initial layouts."""
    target = global_layout.copy()
    for bname, sg in block_sub_grids.items():
        if bname in ec_initial_layouts:
            r0, c0, r1, c1 = sg.grid_region
            c0i, c1i = c0 * k, c1 * k
            ec_lay = ec_initial_layouts[bname]
            rows = min(r1 - r0, ec_lay.shape[0])
            cols = min(c1i - c0i, ec_lay.shape[1])
            target[r0:r0 + rows, c0i:c0i + cols] = ec_lay[:rows, :cols]
    return target
```

### 4c: Fix Level 1 Slicing Condition

**File**: `gadget_routing.py`, `route_full_experiment_as_steps`, line ~1705

**Current code**:
```python
_use_level1_slicing = (
    _use_per_block
    and interacting_names is not None
    and len(interacting_names) < len(all_block_names)
)
```

**Change to**:
```python
_use_level1_slicing = (
    _use_per_block
    and interacting_names is not None
    and len(interacting_names) >= 1
)
```

This enables Level 1 merged sub-grid routing for gadget phases even when
all blocks are interacting.  When all blocks interact, the merged sub-grid
equals the full grid, but the code path is consistent and the re-embedding
logic still works correctly.

### 4d: Also need `wiseArch` and `ion_to_block` available in `route_full_experiment_as_steps`

The function currently receives `n, m, k` but not the `QCCDWiseArch` object.
It creates temporary `QCCDWiseArch` for merged grids but doesn't have a
full-grid one.

**Options**:
1. Create a `wiseArch = QCCDWiseArch(n=n, m=m, k=k)` locally for the full-grid
   transition reconfig (cheap — it's just a config object).
2. Pass `wiseArch` as a parameter from `ionRoutingGadgetArch`.

**Recommendation**: Option 1 — create locally.  The `QCCDWiseArch` constructor
is lightweight.

Also need `ion_to_block` mapping.  This is currently built for EC phases
(line ~1596) but not stored at the function level.

**Fix**: Move the `ion_to_block` construction to the top of
`route_full_experiment_as_steps`:

```python
# Build ion → block mapping for all sub-grids
ion_to_block: Dict[int, str] = {}
for bname, sg in block_sub_grids.items():
    for ion_idx in sg.ion_indices:
        ion_to_block[ion_idx] = bname
```

---

## Fix 5: Concern from Component 6 — Phase Pair Count Consistency Validation

**Severity**: LOW (defensive)  
**File**: `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`  
**Function**: `ionRoutingGadgetArch` (line ~2607)  
**Depends on**: Nothing — standalone.

### Problem

The gadget pair count is computed as `gadget_pairs = len(parallelPairs) - ec_total`.
There's already a sanity check `if sum(phase_pair_counts) != len(parallelPairs)`,
but the warning just falls back to flat routing.  For correctness, we should
also validate per-phase pair counts against the actual operations structure.

### Current Status

The existing sanity check at line ~2607 already handles this:
```python
if sum(phase_pair_counts) != len(parallelPairs):
    logger.warning(...)
    phase_pair_counts = None  # → falls back to flat routing
```

This is already defensive.  **No additional code change needed**, but we should
add a more detailed warning message that logs `ec_total`, `gadget_pairs`,
`n_gadget_phases`, and `gadget_per_phase` so discrepancies are diagnosable.

### Fix

Enhance the warning:
```python
if sum(phase_pair_counts) != len(parallelPairs):
    logger.warning(
        "%s metadata phase pair counts sum to %d, "
        "expected %d (ec_total=%d, gadget_pairs=%d, "
        "n_gadget_phases=%d, gadget_per_phase=%d) "
        "— falling back to flat routing",
        PATCH_LOG_PREFIX,
        sum(phase_pair_counts),
        len(parallelPairs),
        ec_total,
        gadget_pairs,
        n_gadget_phases,
        gadget_per_phase,
    )
```

---

## Fix 6: Concern from Component 9 — `itertools.groupby` ordering assumption

**Severity**: LOW (defensive)  
**File**: `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`  
**Function**: `ionRoutingWISEArch` execution loop (line ~2073)  
**Depends on**: Nothing — standalone.

### Problem

`itertools.groupby(routing_steps, key=lambda s: s.ms_round_index)` only
groups **consecutive** elements with the same key.  If `routing_steps` are
not sorted by `ms_round_index`, steps would be fragmented.

In practice, `_route_round_sequence` and `route_full_experiment_as_steps`
both produce steps in `ms_round_index` order.  But the transition reconfig
steps (from Fix 4) are assigned `ms_round_index=0` and prepended before
the cached EC steps.  This could create `[0, 0, 0, 1, 2, 3, ...]` which
groupby handles correctly (consecutive 0s grouped together).

### Fix (Defensive)

Add an assertion/sort before the groupby:

```python
# Ensure routing_steps are sorted by ms_round_index for groupby correctness
routing_steps = sorted(routing_steps, key=lambda s: s.ms_round_index)
```

This is safe because the execution order within the same `ms_round_index`
is preserved by stable sort.

**Alternative**: Add a debug assertion instead of sorting:
```python
# Verify monotonic ordering (consecutive same-index is fine)
for i in range(1, len(routing_steps)):
    assert routing_steps[i].ms_round_index >= routing_steps[i-1].ms_round_index, (
        f"RoutingSteps not in order: step[{i}].ms_round_index="
        f"{routing_steps[i].ms_round_index} < step[{i-1}].ms_round_index="
        f"{routing_steps[i-1].ms_round_index}"
    )
```

**Recommendation**: Use the sort (defensive, handles edge cases).

---

## Implementation Order

| Step | Fix | Reason |
|------|-----|--------|
| 1 | **Fix 1** (Bug E: empty-cell guard) | Leaf dependency, needed by Fix 4 |
| 2 | **Fix 2** (Bug D: phase index) | Leaf, standalone |
| 3 | **Fix 3** (Bug C: BT propagation) | Leaf, standalone |
| 4 | **Fix 5** (warning enhancement) | Leaf, standalone |
| 5 | **Fix 6** (groupby ordering) | Leaf, standalone |
| 6 | **Fix 4** (Bug A+B: SAT transition reconfig) | Depends on Fix 1, Fix 6 |

### Test Plan

1. **Fix 1 unit test**: Empty-cell layout → `_apply_layout_as_reconfiguration` → no crash.
2. **Fix 2 unit test**: Mock gadget with `get_phase_pairs` → correct local index.
3. **Fix 4 integration test**: d=2 TransversalCNOT gadget compile → ion 28 in correct trap at all steps.
4. **Fix 4 regression test**: d=3 TransversalCNOT matches previous results.
5. **Full pipeline test**: Run `_e2e_test.py` and notebook cell 21 (gadget compile) — no crashes, animation looks correct.

---

## Appendix: Full List of Concerns (Cross-Reference)

| # | Concern | From Component | Fix # | Status |
|---|---------|---------------|-------|--------|
| 1 | `arch.ions[0]` KeyError for empty cells | Component 10 | Fix 1 | Planned |
| 2 | `derive_gadget_ms_pairs` global vs local phase index | Component 1 | Fix 2 | Planned |
| 3 | `initial_BTs` lost after first routing window | Components 4, 9 | Fix 3 | Planned |
| 4 | `schedule=None` heuristic reconfig insufficient | Components 5 (Bug B) | Fix 4a | Planned |
| 5 | Gadget BT fallback drops return-to-EC constraint | Component 5 (Bug A) | Fix 4b | Planned |
| 6 | Level 1 slicing disabled when all blocks interact | Component 5 | Fix 4c | Planned |
| 7 | Phase pair count consistency | Component 6 | Fix 5 | Planned |
| 8 | `itertools.groupby` ordering assumption | Component 10 | Fix 6 | Planned |
| 9 | BT structure for `_route_round_sequence` first window | Component 4 | Fix 3 | Planned |
| 10 | Exit BT coordinate mapping fragility | Component 5 (Bug D) | N/A | Assessed as correct |
| 11 | `_merge_block_routing_steps` correctness | Component 7 | N/A | Assessed as correct |
| 12 | `_merge_disjoint_block_schedules` correctness | Component 8 | N/A | Assessed as correct |
| 13 | Block sub-grid partitioning correctness | Component 2 | N/A | Assessed as correct |
| 14 | Per-block qubit mapping correctness | Component 3 | N/A | Assessed as correct |
