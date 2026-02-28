# Critical Implementation Context — SAT Fix

## Files and Key Line Numbers

### gadget_routing.py (3334 lines)
Path: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py`

- `_build_gadget_exit_bt()` at LINE 2389-2428:
  - Builds exit BT, returns (p_arr_for_solve, bts)
  - Appends empty round `[[]]` to phase_pairs
  - Creates BTs list of dicts, sets bts[-1] with all ion return positions
  
- Gadget phase routing in `route_full_experiment_as_steps()` at LINE 2970-3220:
  - line 3108: `merged_sgs = subgridsize` (Issue 2 fix done)
  - line 3122-3124: BT routing try block
  - line 3130: `_route_round_sequence(..., initial_BTs=bts, ...)`
  - line 3140-3158: except block for BT routing failure → routes without BT
  - line 3175-3180: `_apply_post_gadget_transition(phase_steps, current_layout, n_pairs, "L1")`

- `_apply_post_gadget_transition()` at LINE ~2480-2520 (need to verify)

### qccd_WISE_ion_route.py (3334 lines)
Path: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`

- `_route_round_sequence()` at LINE 1350-1690:
  - Accepts `initial_BTs` parameter
  - var `_sticky_return_bt` saves last BT entry
  - Main loop: while idx < total_ms_rounds
  - Calls `_patch_and_route()` at ~LINE 1540
  
- `_patch_and_route()` at LINE 360-870:
  - Main patch-and-route loop
  - `_TIMEOUT_BASE_S` used via `_estimate_sat_timeout()` 
  - Expansion loop: `n_c += max(inc, min(n_c, wiseArch.k))` / `n_r += inc`
  - `max_cycles = 10` no-progress cycles before stopping

### qccd_SAT_WISE_odd_even_sorter.py (4986 lines)
Path: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_SAT_WISE_odd_even_sorter.py`
- `_TIMEOUT_BASE_S = 300.0` at LINE 313
- `_estimate_sat_timeout()` at LINE 340
- `optimal_QMR_for_WISE()` at ~LINE 3356+

## The Fix: Separate MS Solve from Ion-Return

### Current behavior (breaks):
1. `_build_gadget_exit_bt()` appends `[[]]` empty round and creates dense BT
2. `_route_round_sequence()` tries to solve MS rounds + return round together
3. BT-constrained SAT on merged grid either hangs or takes 300s+ per config

### New behavior (fix):
1. Route gadget MS pairs WITHOUT exit BT (just the gate routing)
2. After MS gates solved, compute dedicated transition reconfig to return ions
3. If needed, do a fast BT-constrained solve with short timeout as optimization

### Code changes needed:

#### In gadget_routing.py, gadget phase routing (~line 3120-3160):
Replace:
```python
p_arr_for_solve, bts = _build_gadget_exit_bt(
    _bt_offsets, phase_pairs, ec_initial_layouts,
)
try:
    phase_steps_raw, _mf = _route_round_sequence(
        ..., p_arr_for_solve, ..., initial_BTs=bts, ...
    )
except Exception as exc:
    # fallback without BT
    phase_steps_raw, _mf = _route_round_sequence(
        ..., list(phase_pairs), ...
    )
```

With:
```python
# Step 1: Route gadget MS pairs (no exit BT — fast, reliable)
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
    progress_callback=_phase_cb,
)
phase_steps = list(phase_steps_raw)

# Step 2: Build dedicated transition reconfig to return ions to EC positions
# (spec §3.5: explicit transition round if BT-embedded solve not used)
if ec_initial_layouts:
    # Build target layout from EC initial positions
    target_layout = np.array(merged_layout, copy=True)  # start from merged
    for bname, r0_off, c0_off in _bt_offsets:
        if bname not in ec_initial_layouts:
            continue
        ec_lay = ec_initial_layouts[bname]
        target_layout[r0_off:r0_off + ec_lay.shape[0],
                      c0_off:c0_off + ec_lay.shape[1]] = ec_lay
    
    # Route the transition as a single reconfig-only round
    transition_steps_raw, _tf = _route_round_sequence(
        np.array(_mf, copy=True),  # start from where MS routing ended
        merged_wise,
        [[]],  # no MS gates, just reconfig
        lookahead=1,
        subgridsize=merged_sgs,
        base_pmax_in=base_pmax_in or 1,
        active_ions=merged_active,
        initial_BTs=[{(0,0): ({ion: pos for ion, pos in _build_return_bt(...)}, [])}],
        stop_event=stop_event,
        max_inner_workers=max_inner_workers,
    )
    phase_steps.extend(list(transition_steps_raw))
```

## The `_apply_post_gadget_transition` Function
Need to check what this already does — it may already handle transition routing.

## ALSO: The full-grid routing path (~line 3230-3280) has the SAME pattern
Need to apply the same fix there too.

## Testing
Run from notebook cell with CSS Surgery d=2:
```python
compile_gadget_for_animation(
    ideal_gadget, qec_metadata=qec_meta, gadget=gadget_cnot,
    qubit_allocation=qubit_alloc, trap_capacity=k,
    lookahead=1, subgridsize=(12, 12, 0), base_pmax_in=1, show_progress=True,
)
```
