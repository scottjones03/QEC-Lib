# SAT Hang Diagnosis Notes

## Problem Summary
Per-block EC routing in `ionRoutingGadgetArch` creates a 3×4 block-local grid with 12 ions, 5 rounds (4 MS + 1 return), k=2. The SAT solver appears to hang. 

## Root Causes Identified

### 1. BT pins spectator ions, but `active_ions` excludes them
**File**: `qccd_WISE_ion_route.py` lines 2632-2660 (per-block path)
- `build_ion_return_bt_for_patch_and_route(block_layout, ...)` creates BT pins for ALL ions in block_layout (including spectators)
- `block_active = [idx for idx in sg.ion_indices if idx in active_ions_set]` only includes non-spectator ions
- Inside `_optimal_QMR_for_WISE`, `active_ions` is used to determine which ions the solver manages
- BT pins reference spectator ions that the solver doesn't track → potential conflict/UNSAT
**FIX**: Filter BT to only pin active (non-spectator) ions, OR pass all ions as active

### 2. Excessive timeouts for small per-block grids  
**File**: `qccd_SAT_WISE_odd_even_sorter.py` line 296
- `_TIMEOUT_BASE_S = 300.0` seconds
- For 3×4 grid: difficulty ≈ 4.47 → max_sat_time ≈ 1341s (22 minutes per config!)
- global_budget_s ≈ 13410s (3.7 hours!)
- 11 configs total; configs at P_max=5,6 are likely UNSAT and will eat timeout budget
- This makes it SEEM like a hang when in reality the solver is just given too much time
**FIX**: Could force `max_inner_workers=1` (serial mode) and set tighter timeout for per-block

### 3. `full_grid_subgridsize` override still in code (WRONG FIX from previous session)
**File**: `qccd_WISE_ion_route.py`
- Line ~2543: `full_grid_subgridsize = (n_grid_cols, n_grid_rows, 0)`
- Line ~2740: `subgridsize=full_grid_subgridsize` in EC fallback path
- User explicitly said this is wrong; cross-boundary preferences handle cross-patch pairs
**FIX**: Revert both lines

## Files to Edit

### qccd_WISE_ion_route.py
Path: `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`

#### A. Revert full_grid_subgridsize (line ~2520-2543)
Delete the comment block + variable. Keep `subgridsize` from the caller.

#### B. Revert EC fallback subgridsize (line ~2740)
Change `subgridsize=full_grid_subgridsize` back to `subgridsize=subgridsize`

#### C. Fix BT to only pin active ions (line ~2651)
In the per-block path, filter BT:
```python
block_bts = build_ion_return_bt_for_patch_and_route(
    block_layout, num_rounds=len(bp),
)
# Filter BT to only pin active (non-spectator) ions
if block_bts:
    for bt_round in block_bts:
        for key, bt_map in list(bt_round.items()):
            if isinstance(bt_map, dict):
                filtered = {ion: pos for ion, pos in bt_map.items()
                           if ion in active_ions_set}
                bt_round[key] = filtered
```

#### D. Force serial SAT + tight timeout for per-block (line ~2665)
```python
blk_steps, _ = _route_round_sequence(
    block_layout,
    block_arch,
    block_p_arr,
    lookahead=len(block_p_arr),
    subgridsize=block_sgs,
    base_pmax_in=base_pmax_in or 1,
    active_ions=block_active,
    initial_BTs=block_bts,
    stop_event=stop_event,
    max_inner_workers=1,  # Force serial to avoid MP overhead
)
```

### gadget_routing.py  
Path: `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py`
- `build_ion_return_bt_for_patch_and_route` (line 673): Consider adding `active_ions` parameter to filter BT at source

## Config Enumeration Order
`_wise_enumerate_pmax_configs`: interleaved high → low → high → low
For P_max range [5, 15]: [15, 5, 14, 6, 13, 7, 12, 8, 11, 9, 10]
P_max=15 is first → should be easy → SAT, then early exit (since grid_cells ≤ 24 → _SEQ_EARLY_EXIT_SAT=1)

## Grid Geometry (from diagnostic)
```
block_0 layout (3×4):  
row 0: [1, 2, 4, 5]
row 1: [13, 14, 16, 17]
row 2: [25, 26, 28, 29]

block_1 layout (3×4):
row 0: [7, 8, 10, 11]  
row 1: [19, 20, 22, 23]
row 2: [31, 32, 34, 35]
```

## Important Call Chain
compile_gadget_for_animation (run.py)
  → compiler.route() (trapped_ion_compiler.py:549)
    → ionRoutingGadgetArch (qccd_WISE_ion_route.py:2240)
      → per-block EC path (line 2592)
        → _route_round_sequence (line 1318)
          → _patch_and_route (line 360)
            → _optimal_QMR_for_WISE (qccd_SAT_WISE_odd_even_sorter.py:3334)
              → ProcessPoolExecutor / sequential fallback
                → _wise_sat_config_worker → binary search over D

## Key Parameters
- stop_event: None (from caller)
- max_inner_workers: None (from caller) → NOT forced serial → tries multiprocessing
- base_pmax_in: 1
- subgridsize: (4, 3, 0) for per-block
- bt_soft: True (because BT pins exist in return round)
