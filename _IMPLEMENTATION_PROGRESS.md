# Implementation Progress Tracker

## Workspace
`/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim`

## Key Files Being Modified
- `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py` (~3294 lines)
- `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py` (~3134 lines)
- `src/qectostim/experiments/hardware_simulation/core/pipeline.py` (needs Issue 1 changes)
- `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/best_effort_compilation_WISE.py` (needs Issue 5 changes)
- `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/trapped_ion_compiler.py`

## COMPLETED

### 1. Diagnostic ERROR in _execute_ms_gates (qccd_WISE_ion_route.py ~line 2260)
- Changed `logger.debug()` to `logger.error()` with full diagnostic info
- Logs solved_pairs, unmatched pairs, available 2q ion sets, counts

### 2. End-of-routing assertion (qccd_WISE_ion_route.py ~line 2500)
- Before `return allOps, barriers, reconfigTime`
- Detects remaining 2-qubit ops in operationsLeft, logs ERROR with ion pairs

### 3. Restored C1 epoch ceiling in _drain_single_qubit_ops (qccd_WISE_ion_route.py line 2037)
- Replaced per-ion-only blocking with proper C1 + C2:
  - C1: Compute `min_ms_epoch` (global minimum epoch of remaining MS gates)
  - C2: Build `blocked_ions` set (all ions involved in any remaining MS gate)
  - Filter: 1q op eligible only if `epoch <= min_ms_epoch` AND `ion not in blocked_ions`
- The old code removed C1 and only kept per-ion min epoch check, which was insufficient

### 4. Issue 2 (Level 2 slicing) - Already fixed
- gadget_routing.py line 2908: `merged_sgs = subgridsize` already passes caller's subgridsize
- Comment confirms: "Issue 2 fix: Pass subgridsize unchanged to enable Level 2"

## COMPLETED (continued)

### 5. Issue 3 - Cross-block parallel reconfig - FIXED
**Function**: `_merge_disjoint_block_schedules` at qccd_WISE_ion_route.py line 2576
**Change**: Replaced split-by-phase-label approach with simple index-alignment.
Old code split by "H"/"V" labels then re-interleaved → inflated pass count when blocks
had different H/V ratios and dropped entries with non-H/V labels.
New code: index-aligns the two schedules and unions h_swaps + v_swaps at each index.
EC per-block routing already passes caller's subgridsize (verified).

### 6. Issue 1 - QECMetadata single source - ALREADY DONE
**Finding**: §1b epoch analysis already replaced with direct metadata reads:
- `PhaseInfo` already has `cx_per_round` and `ms_pair_count` fields
- `QECMetadata` already has `cx_per_ec_round` field
- Factory methods `from_css_memory()` and `from_gadget_experiment()` populate them
- `ionRoutingGadgetArch` §1b reads `[getattr(p, 'ms_pair_count', 0) for p in phases]`

### 7. Issue 5 - Eliminate double routing - ALREADY DONE
**Finding**: `run_single_gadget_config` already uses single `experiment.compile()` path.
No `route_full_experiment()` call exists. `_route_and_simulate` already accepts and
forwards `qec_metadata`, `qubit_allocation`, `block_sub_grids`.

### 8. Diagnostic test - PASSED
All imports OK, merge test passed (3 entries vs old 4), edge cases passed,
PhaseInfo/QECMetadata fields verified. py_compile checks passed on all 3 files.

## Spec Location
`src/qectostim/experiments/hardware_simulation/trapped_ion/GADGET_COMPILATION_SPEC.md`

## Python venv
`my_venv/bin/python`, activated with `source my_venv/bin/activate`
PYTHONPATH=src for imports
