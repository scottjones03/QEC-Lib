# CSS Surgery SAT Failure — Root Cause Analysis & Fix Plan

## Status of Previous Work

The **GADGET_METADATA_FIX_PLAN.md** ("circuit-counted ms_pair_count") changes
across three files were correct and necessary, but they solve a **secondary**
problem.  Even with perfectly accurate `ms_pair_count`, CSS Surgery still hits:

```
RuntimeError: SAT schedule did not realise target layout (2/36 cells differ, schedule_len=5)
```

The metadata-patching ensures the sanity check
`sum(phase_pair_counts) == len(parallelPairs)` passes and phase-aware routing
is *entered* (rather than falling back to flat routing).  But once inside the
phase-aware path, two deeper bugs cause the SAT schedule to produce wrong
swaps on the physical grid.

---

## Root Cause 1: Missing Schedule Coordinate Remapping (CRITICAL)

### The Problem

The gadget Level 1 slicing path in `route_full_experiment_as_steps`
(gadget_routing.py ~L2120-2240) builds a **merged sub-grid** that is smaller
than the global grid.  For CSS Surgery ZZ merge, only block_0 + block_1 are
selected (2 of 3 blocks), producing a merged grid that starts at column 0 and
spans fewer columns than the full architecture.

The SAT solver runs on this merged sub-grid and produces `RoutingStep` objects
whose:
- `layout_after`: array in merged-sub-grid shape
- `schedule`: list of `{phase: "H"/"V", h_swaps: [(r, c), ...], v_swaps: [(r, c), ...]}`
  in **merged-sub-grid coordinates**

The re-embedding loop (gadget_routing.py ~L2220-2235) correctly remaps
`step.layout_after` from merged coordinates to global coordinates:

```python
for step in phase_steps:
    global_after = np.array(current_layout, copy=True)
    for sg in interacting_sgs:
        region = block_regions[sg.block_name]       # merged coords
        r0_g, c0_g, r1_g, c1_g = sg.grid_region    # global trap coords
        global_after[r0_g:..., c0_gi:...] = step.layout_after[r0_m:..., c0_m:...]
    step.layout_after = global_after
    # step.schedule  → NOT TOUCHED!  Still in merged_wise coordinates.
```

**`step.schedule` is never remapped to global coordinates.**

### How It Causes the Error

The schedule flows through:
1. `ionRoutingGadgetArch` → `ionRoutingWISEArch(_precomputed_routing_steps=...)`
2. `ionRoutingWISEArch` → `_apply_layout_as_reconfiguration(schedule=step.schedule)`
3. → `GlobalReconfigurations.physicalOperation(schedule=schedule)`
4. → `_runOddEvenReconfig(sat_schedule=schedule)`

In `_runOddEvenReconfig` (qccd_operations.py ~L310-395), swap coordinates from
`schedule` are applied directly to the **global** layout array `A`:
```python
for (r, c) in h_swaps:      # merged-sub-grid coordinates!
    A[r, c], A[r, c + 1] = A[r, c + 1], A[r, c]   # applied to GLOBAL A
```

When the merged sub-grid doesn't align with column 0 of the global grid (e.g.
XX merge uses block_1 + block_2, where block_1 may start at global column 2),
swaps hit the wrong positions → final layout != target → RuntimeError.

### The Fix

After the `layout_after` re-embedding loop, add a schedule remapping step.
The function `_remap_schedule_to_global` already exists (qccd_WISE_ion_route.py
L2310-2350) and is used correctly in the EC per-block path via
`_merge_block_routing_steps` (L2478).

For the gadget merged sub-grid, we need to compute the **net offset** between
merged coordinates and global coordinates.  Since blocks are packed at
col_offset=0 in the merged grid but have arbitrary global positions, the
mapping is:

```
For each block sg:
  merged_col_start   = block_regions[sg.block_name][1]   # c0_m
  global_col_start   = sg.grid_region[1] * k              # c0_g * k
  row_offset         = sg.grid_region[0]                   # r0_g (merged r0 is always 0)
```

Because the entire schedule operates on the merged grid as one unit (not per-
block), we can compute a single net offset if the merged grid maps to a
contiguous global region.  For CSS Surgery, blocks are allocated in contiguous
rows, so we need:

```python
# After re-embedding layout_after, remap schedule:
if interacting_sgs:
    # The merged sub-grid is packed at (0, 0).
    # The first interacting block's global position gives the offset.
    first_sg = interacting_sgs[0]
    r0_g = first_sg.grid_region[0]
    c0_gi = first_sg.grid_region[1] * k
    step.schedule = _remap_schedule_to_global(step.schedule, r0_g, c0_gi)
```

**File**: `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py`
**Location**: After line ~2235 (the `step.layout_after = global_after` line)

---

## Root Cause 2: 3rd Block EC Pairs in 2-Block Merged Sub-Grid

### The Problem

CSS Surgery merge phases emit interleaved EC on **ALL 3 blocks**, not just the
2 interacting blocks.  From css_surgery_cnot.py ~L443-513 (ZZ merge):

```python
# --- Interleaved EC on all blocks ---
for builder in self._builders:          # ALL 3 builders
    block_name = builder.block_name
    if block_name == "block_1":
        self._emit_ec_round(builder, circuit, ...)
    elif block_name == "block_0":
        self._emit_ec_round(builder, circuit, ...)
    else:
        # block_2: Not involved in ZZ merge, both X and Z are valid
        self._emit_ec_round(builder, circuit, emit_detectors=True)
```

However, `get_phase_active_blocks(0)` returns only `["block_0", "block_1"]`.

This creates two sub-problems:

**2a. CX count mismatch**: The phase's `ms_pair_count` (from circuit-counting)
includes ALL CX instructions — bridge CNOTs for block_0/block_1 AND EC CNOTs
for block_2.  But the routing only covers block_0 + block_1, so `parallelPairs`
entries for block_2 EC have ion indices outside the merged sub-grid.

**2b. Unrouted 3rd-block ions**: During the gadget phase, block_2's ions need
to be shuffled for EC, but the Level 1 slicing only routes block_0 + block_1.
Block_2's EC requires its own routing, which isn't happening.

### The Fix

Two options:

**Option A (Recommended): Route ALL blocks during merge phases.**

Change `get_phase_active_blocks()` to return all 3 blocks for merge phases:
```python
def get_phase_active_blocks(self, phase_index: int) -> List[str]:
    if phase_index in (0, 2):
        return ["block_0", "block_1", "block_2"]  # EC on all blocks
    elif phase_index == 1:
        return ["block_0", "block_1"]
    elif phase_index == 3:
        return ["block_1", "block_2"]
    elif phase_index == 4:
        return ["block_1"]
    return self.get_block_names()
```

This means the merged sub-grid for merge phases spans the full grid (all 3
blocks).  The merged_wise coordinates then equal global coordinates, making
Root Cause 1 a no-op (offset = 0).  But we should still fix RC1 for
correctness with future gadgets that may have different block counts.

**Option B: Route 3rd block separately, then merge schedules.**

Keep `get_phase_active_blocks` as-is but add a second routing pass for the
non-interacting block's EC within the same phase, then merge the two schedules
using `_merge_disjoint_block_schedules`.  This is more complex and fragile.

**Recommended**: Option A.

**File**: `src/qectostim/gadgets/css_surgery_cnot.py`
**Location**: `get_phase_active_blocks()` method (~L157-172)

---

## Implementation Plan

### Step 1: Fix `get_phase_active_blocks` (Root Cause 2)

In `css_surgery_cnot.py`, change merge phases (0 and 2) to return all 3 block
names.  This ensures the merged sub-grid spans the full architecture and all
EC CX pairs are routable.

### Step 2: Fix schedule remapping (Root Cause 1)

In `gadget_routing.py`, after the layout re-embedding loop, add
`_remap_schedule_to_global` for each step's schedule.  Import the function
from `qccd_WISE_ion_route.py`.

Even though Step 1 makes the offset zero for CSS Surgery (all blocks →
merged = global), this fix is essential for correctness with:
- Future gadgets with >3 blocks where only a subset interacts per phase
- Architectures where blocks are non-contiguous

### Step 3: Verify metadata patching still works

The `ms_pair_count` patching from GADGET_METADATA_FIX_PLAN.md is still needed.
With Step 1, `phase_cx_counts` for merge phases now correctly includes ALL
block EC counts, matching the full-grid routing that now covers all 3 blocks.
The sanity check `sum(phase_pair_counts) == len(parallelPairs)` should pass.

### Step 4: Run tests

1. Integration tests: `PYTHONPATH=src pytest .../test_integration_layer.py -v` (30/30)
2. E2E tests: `PYTHONPATH=src pytest .../test_e2e.py -v` (20/20)
3. CSS Surgery notebook cell (cell 24) — should now compile without error
4. TransversalCNOT regression (cell 22) — should still work (2 blocks, no offset)

---

## Summary

| # | Root Cause | File | Fix |
|---|-----------|------|-----|
| 1 | Schedule swap coords not remapped from merged-sub-grid to global | gadget_routing.py | Add `_remap_schedule_to_global` after layout re-embedding |
| 2 | Merge phases report 2 active blocks but emit EC on 3 | css_surgery_cnot.py | Return all 3 blocks from `get_phase_active_blocks` for merge phases |

The prior GADGET_METADATA_FIX_PLAN.md changes (circuit-counted `ms_pair_count`)
remain correct and necessary — they ensure phase-aware routing is entered rather
than falling back to flat routing.  These two new fixes address the bugs that
occur once inside the phase-aware path.
