# CRITICAL CODE CONTEXT FOR SAT FIX IMPLEMENTATION

## Files
- gadget_routing.py: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py` (3334 lines)
- qccd_WISE_ion_route.py: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py` (3334 lines)  
- qccd_SAT_WISE_odd_even_sorter.py: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_SAT_WISE_odd_even_sorter.py` (4986 lines)
- run.py: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/demo/run.py` (670 lines)

## ROOT CAUSE
SAT solver hangs during CSS Surgery gadget phase routing because:
1. `_build_gadget_exit_bt()` (line 2389-2428 in gadget_routing.py) appends empty return round `[[]]` and creates dense BT constraints pinning ALL ions to EC-return positions
2. The combined MS+return problem is harder than routing the MS gates alone
3. SAT timeout is 300s base × difficulty factor — too long for tight BT problems
4. The try/except fallback (line 3130-3158) only triggers on Exception, but SAT may just hang

## WHAT ALREADY EXISTS
- `_apply_post_gadget_transition()` (line 2799-2838 in gadget_routing.py) already handles post-gadget transition:
  - calls `_reconstruct_ec_target()` to build target EC layout  
  - calls `_compute_transition_reconfig_steps()` to build reconfig steps
  - Sets `ts.ms_round_index = n_pairs` for transition steps
  - This function ALREADY does what we need for post-gadget ion return!

## THE FIX
Remove BT from gadget MS solve. Route MS pairs freely, then let `_apply_post_gadget_transition()` handle ion return.

### Changes in gadget_routing.py:

#### Location 1: L1 slicing path (line ~3120-3160)
REPLACE the try/except BT block with just the non-BT routing:

Old code at ~line 3118-3158:
```python
                _bt_offsets = [
                    (sg.block_name, block_regions[sg.block_name][0],
                     block_regions[sg.block_name][1])
                    for sg in interacting_sgs
                ]
                p_arr_for_solve, bts = _build_gadget_exit_bt(
                    _bt_offsets, phase_pairs, ec_initial_layouts,
                )
                try:
                    phase_steps_raw, _mf = _route_round_sequence(
                        np.array(merged_layout, copy=True),
                        merged_wise,
                        p_arr_for_solve,
                        lookahead=min(lookahead, len(p_arr_for_solve)),
                        subgridsize=merged_sgs,
                        base_pmax_in=base_pmax_in or 1,
                        active_ions=merged_active,
                        initial_BTs=bts,
                        stop_event=stop_event,
                        max_inner_workers=max_inner_workers,
                        progress_callback=_phase_cb,
                    )
                    phase_steps = list(phase_steps_raw)
                except Exception as exc:
                    logger.warning(...)
                    phase_steps_raw, _mf = _route_round_sequence(
                        np.array(merged_layout, copy=True),
                        merged_wise,
                        list(phase_pairs),
                        lookahead=min(lookahead, len(phase_pairs)),
                        ...
                    )
                    phase_steps = list(phase_steps_raw)
```

New code:
```python
                # Route gadget MS pairs without exit BT (spec §3.5:
                # separate MS solve from ion return). The post-gadget
                # transition handled by _apply_post_gadget_transition()
                # will return ions to EC positions after this phase.
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
```

#### Location 2: Full-grid routing path (line ~3230-3280)
SAME change: remove BT, let _apply_post_gadget_transition handle return.

Old code at ~line 3225-3280:
```python
                _fg_sgs = [
                    (bname, sg.grid_region[0], sg.grid_region[1] * k)
                    for bname, sg in block_sub_grids.items()
                ]
                p_arr_for_solve, bts = _build_gadget_exit_bt(
                    _fg_sgs, phase_pairs, ec_initial_layouts,
                )
                try:
                    phase_steps_raw, current_layout = _route_round_sequence(
                        ... p_arr_for_solve, ... initial_BTs=bts, ...
                    )
                except Exception as exc:
                    phase_steps_raw, current_layout = _route_round_sequence(
                        ... list(phase_pairs), ...
                    )
```

New code:
```python
                # Route gadget MS pairs without exit BT (spec §3.5)
                phase_steps_raw, current_layout = _route_round_sequence(
                    np.array(current_layout, copy=True),
                    wiseArch,
                    list(phase_pairs),
                    lookahead=min(lookahead or 2, len(phase_pairs)),
                    subgridsize=subgridsize,
                    base_pmax_in=base_pmax_in or 1,
                    active_ions=active_ions,
                    stop_event=stop_event,
                    max_inner_workers=max_inner_workers,
                    progress_callback=_phase_cb,
                )
                phase_steps = list(phase_steps_raw)
```

## POST-GADGET TRANSITION ALREADY WORKS
After both Location 1 and Location 2, the existing `_apply_post_gadget_transition()` is called:
- Line 3175: `phase_steps, current_layout = _apply_post_gadget_transition(phase_steps, current_layout, n_pairs, "L1")`
- Line 3290: `phase_steps, current_layout = _apply_post_gadget_transition(phase_steps, current_layout, n_pairs, "full-grid")`

This function already:
1. Checks if current layout matches EC target
2. If not, calls `_compute_transition_reconfig_steps()` to build transition
3. Appends transition steps to phase_steps

## WHAT `_compute_transition_reconfig_steps` DOES
Need to verify this exists/works. Search for it.

## REMOVE "Group return round" code
After the BT removal, the "Group return round with last MS round" code (that caps step.ms_round_index >= n_pairs) can be simplified since there won't be a BT return round anymore.

## TESTING
Create diagnostic script _test_sat_fix.py:
```python
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.experiments.hardware_simulation.trapped_ion.demo.run import compile_gadget_for_animation

d, k = 2, 2
gadget = CSSSurgeryCNOTGadget()
code = RotatedSurfaceCode(distance=d)
ft = FaultTolerantGadgetExperiment(codes=[code], gadget=gadget, noise_model=None, num_rounds_before=d, num_rounds_after=d)
ideal = ft.to_stim()
qec_meta = ft.qec_metadata
alloc = ft._unified_allocation

t0 = time.perf_counter()
arch, compiler, compiled, batches, ion_roles, p2l, remap = compile_gadget_for_animation(
    ideal, qec_metadata=qec_meta, gadget=gadget, qubit_allocation=alloc,
    trap_capacity=k, lookahead=1, subgridsize=(12, 12, 0), base_pmax_in=1, show_progress=False,
)
wall = time.perf_counter() - t0
print(f"Compile: {wall:.1f}s, batches={len(batches)}, ops={len(compiled.scheduled.metadata.get('all_operations', []))}")
# Count MS gates
ms_count = sum(1 for op in compiled.scheduled.metadata.get('all_operations', []) if 'MS' in type(op).__name__ or 'TwoQubit' in type(op).__name__)
print(f"MS gates: {ms_count}")
```
