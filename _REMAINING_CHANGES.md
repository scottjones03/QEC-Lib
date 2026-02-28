# REMAINING FIX A CHANGES — READ THIS FIRST

## File: /Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py

## COMPLETED SO FAR:
- [x] Helper function `_compute_return_reconfig` added after `build_ion_return_bt_for_patch_and_route` (~L942)
- [x] Location 1: EC per-block routing - removed BT bundling, added separate return reconfig (~L2744)
- [x] Fix B: qccd_WISE_ion_route.py - retry window capped to `max(2*lookahead, 4)` 
- [x] Fix C: qccd_SAT_WISE_odd_even_sorter.py - timeout scaling fixed

## REMAINING Location 2: Cross-block EC pairs
### Search for this exact text in gadget_routing.py:
```
                        _xb_bts = build_ion_return_bt_for_patch_and_route(
                            _xb_layout,
                            num_rounds=len(_xb_nonempty),
                            active_ions=_xb_active,
                        )
                        _xb_p_arr = list(_xb_nonempty) + [[]]
```
### Replace the block from `_xb_bts =` through the _route_round_sequence call, removing BT and `+ [[]]`:
- Remove `_xb_bts = build_ion_return_bt_for_patch_and_route(...)` (4 lines)
- Change `_xb_p_arr = list(_xb_nonempty) + [[]]` to `_xb_p_arr = list(_xb_nonempty)`
- In _route_round_sequence call: remove `initial_BTs=_xb_bts,` parameter
- Change `_xb_steps, _ = _route_round_sequence(` to `_xb_steps, _xb_final = _route_round_sequence(`
- After _route_round_sequence, add:
```python
                        _xb_return = _compute_return_reconfig(
                            _xb_final, _xb_layout,
                            _xb_wise, subgridsize,
                            base_pmax_in=base_pmax_in or 1,
                            stop_event=stop_event,
                            max_inner_workers=max_inner_workers,
                        )
                        _xb_steps = list(_xb_steps) + _xb_return
```

## REMAINING Location 3: Single-grid fallback
### Search for this exact text in gadget_routing.py:
```
                bts = build_ion_return_bt_for_patch_and_route(
                    current, num_rounds=len(sr_pairs),
                    active_ions=active_ions,
                )
                p_arr_with_return = list(sr_pairs) + [[]]
```
### Changes:
- Remove `bts = build_ion_return_bt_for_patch_and_route(...)` (4 lines)
- Change `p_arr_with_return = list(sr_pairs) + [[]]` to just use `list(sr_pairs)`
- In _route_round_sequence: remove `initial_BTs=bts,`, change var name
- Change `sr_merged, current =` to capture final layout
- After _route_round_sequence, add return reconfig using full arch (n, m, k)
- Then update `current` to target layout after return

## REMAINING Location 4: Gadget phase
### Search for this exact text in gadget_routing.py:
```
                    p_arr_for_solve, bts = _build_gadget_exit_bt(
                        _bt_offsets, phase_pairs, ec_initial_layouts,
                    )
                    try:
                        phase_steps_raw, _mf = _route_round_sequence(
```
### Changes:
- Remove `p_arr_for_solve, bts = _build_gadget_exit_bt(...)` call
- Remove try/except block
- Replace with direct: `phase_steps_raw, _mf = _route_round_sequence(..., list(phase_pairs), ...)`
  (no initial_BTs, no exit BT)
- Keep all the _apply_post_gadget_transition code that follows

## REMAINING FIXES:
### Fix D: In _apply_post_gadget_transition, after transition, verify blocks_pure and log ERROR if not
### Fix E: Add blocks_pure diagnostic after each phase 
### Fix F: Cell 24 in notebook - change `lookahead=2` to `lookahead=1`
- Notebook: /Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/notebooks/trapped_ion_demo.ipynb
- Cell 24 ID: #VSC-76f13df6
- Search for `lookahead=2` in the compile_gadget_for_animation call
