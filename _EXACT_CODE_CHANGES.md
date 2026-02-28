# EXACT CODE SNIPPETS TO MODIFY

## FIX B — Cap retry window
### File: qccd_WISE_ion_route.py, around L1670-1710
### EXACT oldString for replacement:
```
                remaining = parallelPairs[idx:]
                _MAX_RETRY_WINDOW = 6  # at most 6 rounds per retry
                if len(remaining) > len(P_arr):
                    retry_window = min(
                        len(remaining),
                        max(2 * len(P_arr), _MAX_RETRY_WINDOW),
                    )
```
### newString:
```
                remaining = parallelPairs[idx:]
                # Cap retry at 2×lookahead to prevent window explosion.
                # Old: _MAX_RETRY_WINDOW = 6 (could escalate far beyond lookahead)
                _MAX_RETRY_WINDOW = max(2 * lookahead, 4)
                if len(remaining) > len(P_arr):
                    retry_window = min(
                        len(remaining),
                        _MAX_RETRY_WINDOW,
                    )
```

## FIX C — Timeout scaling
### File: qccd_SAT_WISE_odd_even_sorter.py, around L296-303
### EXACT current code:
```
    var_estimate = (
        max(1, num_ions)
        * max(1, n * m)
        * max(1, P_max)
        * max(1, R)
    )
    # Reference: d=2, k=2 baseline → 6 ions, 3×2 grid, P_max=1, R=1 → 36
    _REFERENCE = 36.0
```
### New code:
```
    # Old formula: num_ions × (n×m) × P_max × R — double-counts grid area
    # because num_ions ≈ n×m for full grids.  Use grid_area × P_max × R.
    var_estimate = (
        max(1, n * m)
        * max(1, P_max)
        * max(1, R)
    )
    # Reference: d=2, k=2 baseline → 3×2 grid, P_max=1, R=1 → 6
    _REFERENCE = 6.0
```

## FIX A — Separate return round SAT
### Key locations in gadget_routing.py:

### Location 1: EC per-block (around L2644-2667)
OLD:
```python
                    block_bts = build_ion_return_bt_for_patch_and_route(
                        block_layout, num_rounds=len(bp),
                        active_ions=block_active,
                    )
                    block_p_arr = list(bp) + [[]]  # empty return round
                    ...
                    blk_steps, _ = _route_round_sequence(
                        block_layout,
                        block_arch,
                        block_p_arr,
                        lookahead=min(lookahead, len(block_p_arr)),
                        ...
                        initial_BTs=block_bts,
                        ...
                    )
```
NEW: Remove block_bts and `+ [[]]`. After _route_round_sequence, if final layout != block_layout, call _compute_return_round().

### Location 2: Cross-block EC (around L2734-2757)
OLD:
```
                        _xb_bts = build_ion_return_bt_for_patch_and_route(...)
                        _xb_p_arr = list(_xb_nonempty) + [[]]
                        _xb_steps, _ = _route_round_sequence(..., _xb_p_arr, ..., initial_BTs=_xb_bts, ...)
```
Same pattern.

### Location 3: Single-grid fallback (around L2792-2810)
OLD:
```
                bts = build_ion_return_bt_for_patch_and_route(...)
                p_arr_with_return = list(sr_pairs) + [[]]
                sr_merged, current = _route_round_sequence(..., p_arr_with_return, ..., initial_BTs=bts, ...)
```
Same pattern.

### Location 4: Gadget phase (around L3375-3400)
OLD:
```
                    p_arr_for_solve, bts = _build_gadget_exit_bt(...)
                    try:
                        phase_steps_raw, _mf = _route_round_sequence(..., p_arr_for_solve, ..., initial_BTs=bts, ...)
                    except:
                        phase_steps_raw, _mf = _route_round_sequence(..., list(phase_pairs), ...)
```
NEW: Always use `list(phase_pairs)` without BT. Rely on `_apply_post_gadget_transition` for return.

## HELPER FUNCTION for return round separation
Need to add a helper function near `build_ion_return_bt_for_patch_and_route` that computes the return round separately:

```python
def _compute_return_reconfig(
    final_layout: np.ndarray,
    target_layout: np.ndarray,
    arch,
    subgridsize,
    base_pmax_in=1,
    stop_event=None,
    max_inner_workers=None,
):
    """Compute return-round reconfiguration as separate SAT call."""
    from ..compiler.qccd_WISE_ion_route import _rebuild_schedule_for_layout
    from ..compiler.qccd_SAT_WISE_odd_even_sorter import NoFeasibleLayoutError
    
    if np.array_equal(final_layout, target_layout):
        return []
    
    try:
        snaps = _rebuild_schedule_for_layout(
            final_layout.copy(), arch, target_layout,
            subgridsize=subgridsize,
            base_pmax_in=base_pmax_in,
            stop_event=stop_event,
            max_inner_workers=max_inner_workers or 1,
        )
    except (NoFeasibleLayoutError, Exception) as exc:
        import logging
        logging.getLogger("wise.qccd.gadget_routing").warning(
            "[ReturnRound] separate return-round SAT failed (%s)", exc)
        snaps = []
    
    from ..compiler.qccd_WISE_ion_route import RoutingStep
    steps = []
    for i, (layout, schedule, _) in enumerate(snaps):
        steps.append(RoutingStep(
            layout_after=np.array(layout, copy=True),
            schedule=schedule,
            solved_pairs=[],
            ms_round_index=-1,
            from_cache=False,
            tiling_meta=(i, 0),
            can_merge_with_next=False,
            is_initial_placement=False,
            is_layout_transition=True,
        ))
    return steps
```

## FIX F — Cell 24 lookahead
Change `lookahead=2` to `lookahead=1` in the compile_gadget_for_animation call.
