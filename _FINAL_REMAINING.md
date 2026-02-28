# FINAL REMAINING CHANGES — READ THIS FIRST

## FILE: /Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py

## COMPLETED:
- Fix B: qccd_WISE_ion_route.py — retry window capped ✓
- Fix C: qccd_SAT_WISE_odd_even_sorter.py — timeout scaling ✓
- Fix A helper function: _compute_return_reconfig added after L942 ✓
- Fix A Location 1: EC per-block routing (removed BT, added separate return) ✓
- Fix A Location 2: Cross-block EC pairs (removed BT, added separate return) ✓

## REMAINING LOCATION 3: Single-grid fallback
### Search gadget_routing.py for:
```
                bts = build_ion_return_bt_for_patch_and_route(
                    current, num_rounds=len(sr_pairs),
                    active_ions=active_ions,
                )
                p_arr_with_return = list(sr_pairs) + [[]]
                sr_merged, current = _route_round_sequence(
```
### Replace with (remove bts, remove +[[]], add return reconfig after):
```python
                sr_merged, _sr_final = _route_round_sequence(
                    current,
                    QCCDWiseArch(m=m, n=n, k=k),  # or whatever arch is used
                    list(sr_pairs),
                    lookahead=min(lookahead, len(sr_pairs)),
                    subgridsize=subgridsize,
                    base_pmax_in=base_pmax_in or 1,
                    active_ions=active_ions,
                    stop_event=stop_event,
                    max_inner_workers=max_inner_workers,
                    progress_callback=_phase_cb,
                )
                # Separate return-round reconfig
                _sr_return = _compute_return_reconfig(
                    _sr_final, current,
                    QCCDWiseArch(m=m, n=n, k=k),
                    subgridsize,
                    base_pmax_in=base_pmax_in or 1,
                    stop_event=stop_event,
                    max_inner_workers=max_inner_workers,
                )
                sr_merged = list(sr_merged) + _sr_return
                current = _sr_final  # or target layout...
```
NOTE: Need to read the actual code to get exact variable names and arch construction.

## REMAINING LOCATION 4: Gadget phase
### Search gadget_routing.py for:
```
                    p_arr_for_solve, bts = _build_gadget_exit_bt(
```
### Replace the entire try/except block:
- Remove `p_arr_for_solve, bts = _build_gadget_exit_bt(...)` 
- Remove try/except
- Use `list(phase_pairs)` directly 
- Keep `_apply_post_gadget_transition` handling after

## REMAINING Fix D: In _apply_post_gadget_transition, log ERROR if blocks impure after transition
## REMAINING Fix E: Add blocks_pure diagnostic logging
## REMAINING Fix F: Cell 24 notebook — change `lookahead=2` to `lookahead=1`
- File: /Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/notebooks/trapped_ion_demo.ipynb
- Cell 24 ID: #VSC-76f13df6
- In compile_gadget_for_animation call: `lookahead=2` → `lookahead=1`
