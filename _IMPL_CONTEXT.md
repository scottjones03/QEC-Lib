# Implementation Context — Critical Reference

## Files to modify:
1. `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py` (3681 lines)
2. `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py` (3529 lines)
3. `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_SAT_WISE_odd_even_sorter.py` (~4959 lines)
4. `notebooks/trapped_ion_demo.ipynb` cell 24 (ID #VSC-76f13df6)

## Fix A — Separate return round SAT

### Location 1: EC per-block routing (gadget_routing.py L2644-2667)
CURRENT CODE:
```python
                    block_bts = build_ion_return_bt_for_patch_and_route(
                        block_layout, num_rounds=len(bp),
                        active_ions=block_active,
                    )
                    block_p_arr = list(bp) + [[]]  # empty return round
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
CHANGE TO: Remove BTs and `+ [[]]`. After _route_round_sequence, call `_rebuild_schedule_for_layout` for return.

### Location 2: Cross-block EC pairs (gadget_routing.py L2734-2757)
CURRENT:
```python
                        _xb_bts = build_ion_return_bt_for_patch_and_route(
                            _xb_layout,
                            num_rounds=len(_xb_nonempty),
                            active_ions=_xb_active,
                        )
                        _xb_p_arr = list(_xb_nonempty) + [[]]
                        _xb_steps, _ = _route_round_sequence(
                            _xb_layout, _xb_wise, _xb_p_arr,
                            lookahead=min(lookahead, len(_xb_p_arr)),
                            ...
                            initial_BTs=_xb_bts,
                            ...
                        )
```
Same change pattern.

### Location 3: Single-grid fallback (gadget_routing.py L2792-2810)
CURRENT:
```python
                bts = build_ion_return_bt_for_patch_and_route(
                    current, num_rounds=len(sr_pairs),
                    active_ions=active_ions,
                )
                p_arr_with_return = list(sr_pairs) + [[]]
                sr_merged, current = _route_round_sequence(
                    ..., p_arr_with_return, ..., initial_BTs=bts, ...
                )
```
Same change pattern.

### Location 4: Gadget phase (gadget_routing.py L3375-3410)
CURRENT:
```python
                    p_arr_for_solve, bts = _build_gadget_exit_bt(
                        _bt_offsets, phase_pairs, ec_initial_layouts,
                    )
                    try:
                        phase_steps_raw, _mf = _route_round_sequence(
                            ..., p_arr_for_solve, ..., initial_BTs=bts, ...
                        )
                    except Exception as exc:
                        # fallback: route without BT
                        phase_steps_raw, _mf = _route_round_sequence(
                            ..., list(phase_pairs), ...
                        )
```
Change: Use `list(phase_pairs)` directly (no exit BT). Rely on `_apply_post_gadget_transition` for return.

## Fix B — Cap retry window (qccd_WISE_ion_route.py)
Look for `_MAX_RETRY_WINDOW = 6` and the retry logic around L1675-1705.
Change: `retry_window = min(len(remaining), 2 * lookahead)` (lookahead is available as param)

## Fix C — Timeout scaling (qccd_SAT_WISE_odd_even_sorter.py)
Look for `var_estimate` around L280-303.
CURRENT: `var_estimate = num_ions * (n * m) * P_max * R`
CHANGE TO: `var_estimate = max(1, n * m) * max(1, P_max) * max(1, R)`
And adjust _REFERENCE_VAR_ESTIMATE from 36 to 6 (baseline: 3x2 grid, P_max=1, R=1)

## Fix D — Robust post-gadget return
In `_apply_post_gadget_transition` (gadget_routing.py around L2830-2870):
- If transition fails, log ERROR not WARNING
- After transition, verify blocks_pure

## Fix E — blocks_pure diagnostic logging
Add logging after each phase showing ion block purity.

## Fix F — Cell 24 lookahead=1

## COMPLETED FIXES:
- Fix B: qccd_WISE_ion_route.py L1676: Changed `_MAX_RETRY_WINDOW = 6` to `_MAX_RETRY_WINDOW = max(2 * lookahead, 4)` and `retry_window = min(len(remaining), _MAX_RETRY_WINDOW)` — DONE
- Fix C: qccd_SAT_WISE_odd_even_sorter.py L296: Removed num_ions from var_estimate, changed _REFERENCE from 36 to 6 — DONE

## REMAINING: Fix A needs 4 code changes + 1 helper function. Fix D, E, F still to do.

## FIX A IMPLEMENTATION DETAILS:

### Step 1: Add `_compute_return_reconfig` helper function near L900 in gadget_routing.py
After `build_ion_return_bt_for_patch_and_route`, add:
```python
def _compute_return_reconfig(
    final_layout: np.ndarray,
    target_layout: np.ndarray,
    arch,
    subgridsize,
    base_pmax_in=1,
    stop_event=None,
    max_inner_workers=None,
) -> list:
    """Compute return-round reconfiguration as a separate SAT call.
    
    Instead of bundling the return round as an extra empty round with BT
    pins in the main MS-gate SAT instance (which inflates R and can cause
    UNSAT), this function solves the pure reconfiguration problem: 
    move all ions from *final_layout* to *target_layout*.
    
    Uses _rebuild_schedule_for_layout which has 3-level fallback:
      1. Direct SAT solve
      2. Greedy row-by-row
      3. Identity (no-op)
    """
    from ..compiler.qccd_WISE_ion_route import (
        _rebuild_schedule_for_layout,
        RoutingStep,
    )
    from ..compiler.qccd_SAT_WISE_odd_even_sorter import NoFeasibleLayoutError
    
    if np.array_equal(final_layout, target_layout):
        return []
    
    _logger = logging.getLogger("wise.qccd.gadget_routing")
    _logger.debug("[ReturnRound] computing separate return reconfig "
                  "(%d non-matching cells)",
                  int(np.sum(final_layout != target_layout)))
    try:
        snaps = _rebuild_schedule_for_layout(
            final_layout.copy(), arch, target_layout,
            subgridsize=subgridsize,
            base_pmax_in=base_pmax_in,
            stop_event=stop_event,
            max_inner_workers=max_inner_workers or 1,
        )
    except (NoFeasibleLayoutError, Exception) as exc:
        _logger.warning("[ReturnRound] separate return-round SAT failed (%s); "
                       "ions may not have returned to starting positions", exc)
        snaps = []
    
    steps = []
    for i, (layout, schedule, _pairs) in enumerate(snaps):
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

### Step 2: Modify Location 1 (EC per-block, L2644-2667)
Remove block_bts, remove `+ [[]]`, add return reconfig after _route_round_sequence

### Step 3: Modify Location 2 (Cross-block EC, L2734-2757)
Same pattern

### Step 4: Modify Location 3 (Single-grid fallback, L2792-2810)
Same pattern

### Step 5: Modify Location 4 (Gadget phase, L3375-3410)
Use list(phase_pairs) directly, no _build_gadget_exit_bt

## KEY LINE NUMBERS (gadget_routing.py, 3681 total lines):
- build_ion_return_bt_for_patch_and_route: ~L889-944
- _compute_transition_reconfig_steps: ~L2100-2160 (has imports for _rebuild_schedule_for_layout, RoutingStep)
- _build_gadget_exit_bt: ~L2258-2296
- EC per-block routing: ~L2644-2667
- Cross-block EC pairs: ~L2734-2757
- Single-grid fallback: ~L2792-2810
- Gadget phase: ~L3375-3410
- _apply_post_gadget_transition: ~L2830-2870
- `import logging` already at top of file
- `import numpy as np` already at top of file
Change `lookahead=2` to `lookahead=1` in cell 24

## Key imports needed for return round separation:
```python
from ..compiler.qccd_WISE_ion_route import (
    _rebuild_schedule_for_layout,
    RoutingStep,
)
from ..compiler.qccd_SAT_WISE_odd_even_sorter import NoFeasibleLayoutError
```
These are already imported at L2103-2110 inside `_compute_transition_reconfig_steps`.

## Helper for return round:
The return round needs `_rebuild_schedule_for_layout(current_layout, arch, target_layout, subgridsize=..., base_pmax_in=...)`.
This returns snapshots: list of (layout, schedule, pairs) tuples.
Need to convert to RoutingStep objects — the `_snapshots_to_steps` function at L2125 does this.
And `_consolidate_transition_steps` at L2142 merges them.

## _rebuild_schedule_for_layout signature:
```python
_rebuild_schedule_for_layout(
    current_layout, wiseArch, target_layout,
    subgridsize=subgridsize,
    base_pmax_in=base_pmax_in,
    stop_event=stop_event,
    max_inner_workers=max_inner_workers,
)
```
Returns: list of (layout, schedule, solved_pairs) snapshots
