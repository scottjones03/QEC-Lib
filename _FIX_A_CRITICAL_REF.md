# FIX A CRITICAL IMPLEMENTATION REFERENCE

## File: /Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py

## STEP 1: Add helper function AFTER line 942 (after `return [dict() for _ in range(num_rounds)] + [{(0, 0): dict(bt_map)}]`)

INSERT AFTER the `build_ion_return_bt_for_patch_and_route` function. The function ends at L942 with the return statement.

## STEP 2: Location 1 — EC per-block (around L2644-2667)
The exact block to replace starts with `block_bts = build_ion_return_bt_for_patch_and_route(`
and ends with `per_block_steps[bname] = list(blk_steps)`.

Key changes:
- Remove `block_bts = build_ion_return_bt_for_patch_and_route(...)` (3 lines)
- Change `block_p_arr = list(bp) + [[]]` to `block_p_arr = list(bp)`
- Change `lookahead=min(lookahead, len(block_p_arr))` (no change needed - just uses shorter array)
- Remove `initial_BTs=block_bts,` parameter
- After `blk_steps, _blk_final = _route_round_sequence(...)`:
  Add return reconfig: `return_steps = _compute_return_reconfig(_blk_final, block_layout, block_arch, subgridsize, base_pmax_in, stop_event, max_inner_workers)`
  Then: `per_block_steps[bname] = list(blk_steps) + return_steps`

## STEP 3: Location 2 — Cross-block EC (around L2734-2757)
The exact block starts with `_xb_bts = build_ion_return_bt_for_patch_and_route(`
Key changes — same pattern as Location 1:
- Remove _xb_bts
- Change `_xb_p_arr = list(_xb_nonempty) + [[]]` to `_xb_p_arr = list(_xb_nonempty)` 
- Remove `initial_BTs=_xb_bts,`
- After _route_round_sequence: add return reconfig
- Capture final layout from _route_round_sequence second return value

## STEP 4: Location 3 — Single-grid fallback (around L2792-2810)
Starts with `bts = build_ion_return_bt_for_patch_and_route(`
Key changes — same pattern:
- Remove bts
- Change `p_arr_with_return = list(sr_pairs) + [[]]` to just `list(sr_pairs)`
- Remove `initial_BTs=bts,`
- After: add return reconfig

## STEP 5: Location 4 — Gadget phase (around L3375-3410)
Starts with `p_arr_for_solve, bts = _build_gadget_exit_bt(`
Key changes:
- Remove entire try/except block that uses _build_gadget_exit_bt
- Replace with direct: `phase_steps_raw, _mf = _route_round_sequence(..., list(phase_pairs), ...)`
- Keep _apply_post_gadget_transition handling that comes after

## KEY FUNCTION SIGNATURES:
- `_route_round_sequence(oldArr, arch, parallelPairs, *, lookahead, subgridsize, base_pmax_in, active_ions, ..., initial_BTs=None, ...) -> (List[RoutingStep], np.ndarray)`
- `_rebuild_schedule_for_layout(current, arch, target, subgridsize=..., base_pmax_in=..., stop_event=..., max_inner_workers=...) -> List[(layout, schedule, pairs)]`
- `RoutingStep(layout_after, schedule, solved_pairs, ms_round_index, from_cache, tiling_meta, can_merge_with_next, is_initial_placement, is_layout_transition)`

## ALREADY COMPLETED:
- Fix B: qccd_WISE_ion_route.py — retry window capped to `max(2*lookahead, 4)` ✓
- Fix C: qccd_SAT_WISE_odd_even_sorter.py — timeout scaling fixed (removed num_ions, _REFERENCE=6.0) ✓

## REMAINING AFTER FIX A:
- Fix D: Robust post-gadget return (gadget_routing.py _apply_post_gadget_transition)
- Fix E: blocks_pure diagnostic logging
- Fix F: Cell 24 lookahead=1 in notebook
