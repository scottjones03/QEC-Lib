# STATUS: Fixes Applied

## COMPLETED FIXES

### Fix 1+3: toMoves-based MS gate execution + getTrapForIons fallback
**File**: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`
**What**: Replaced `_execute_ms_gates` function (nested helper in `ionRoutingWISEArch`) with two-path execution:
- **Path A (toMoves available)**: Uses `toMoves[round_idx]` to get specific Operation objects. Uses old-style `any(all(ion.idx in p for ion in op.ions) for p in solved_pairs)` contains-check. Has `op.ions[0].parent` fallback when `getTrapForIons()` returns None.
- **Path B (no toMoves entry)**: Scans all operationsLeft with frozenset match (for precomputed gadget steps). Also has fallback.
**Status**: APPLIED ✓

### Fix 2: Remove _merge_phase_pairs
**File**: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py`  
**What**: Removed `phase_pairs = _merge_phase_pairs(phase_pairs)` call at line 2851 in `route_full_experiment_as_steps`. This was changing the round count, breaking correspondence with `phase_pair_counts` and `toMoves` entries.
**Status**: APPLIED ✓

## REMAINING TODO

### Verify with test run
Run a quick diagnostic to confirm MS gates are now being executed:
```bash
cd "/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim"
PYTHONPATH=src WISE_INPROCESS_LIMIT=999999999 my_venv/bin/python -c "
import sys, os
sys.path.insert(0, 'src')
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.experiments.memory import CSSMemoryExperiment
from qectostim.experiments.hardware_simulation.trapped_ion.utils import (
    WISEArchitecture, TrappedIonCompiler,
)
from qectostim.experiments.hardware_simulation.trapped_ion.utils.qccd_nodes import QCCDWiseArch
from qectostim.experiments.hardware_simulation.trapped_ion.compiler.routing_config import WISERoutingConfig

dis = 2
code = RotatedSurfaceCode(distance=dis)
mem = CSSMemoryExperiment(code=code, rounds=1, noise_model=None, basis='z')
ideal = mem.to_stim()

wise_cfg = QCCDWiseArch(m=2, n=3, k=2)
arch = WISEArchitecture(wise_config=wise_cfg, add_spectators=True, compact_clustering=True)
compiler = TrappedIonCompiler(arch, is_wise=True, wise_config=wise_cfg)
compiler.routing_kwargs = dict(routing_config=WISERoutingConfig.default(lookahead=1, subgridsize=(12,12,1)))
compiled = compiler.compile(ideal)
batches = compiled.scheduled.batches
all_ops = compiled.scheduled.metadata.get('all_operations', [])
ms_count = sum(1 for op in all_ops if 'MS' in type(op).__name__ or 'TwoQubit' in type(op).__name__)
print(f'Total ops: {len(all_ops)}')
print(f'MS gates: {ms_count}')
print(f'Batches: {len(batches)}')
assert ms_count > 0, 'FAIL: No MS gates executed!'
print('PASS: MS gates present')
" 2>&1 | tail -20
```

### Additional verification (gadget path):
Test CSS Surgery or Transversal CNOT gadget compilation to verify gadget routing also produces MS gates.

## KEY INFORMATION

### Root causes fixed:
1. `_execute_ms_gates` was using frozenset exact match scanning ALL operationsLeft instead of toMoves[round_idx]. Now uses toMoves[round_idx] with old-style contains check (Path A).
2. `_merge_phase_pairs` was changing round counts, breaking toMoves correspondence. Now removed.
3. `getTrapForIons()` returning None caused silent skip. Now has `op.ions[0].parent` fallback.

### Backward compatibility:
- Non-gadget path (ionRoutingWISEArch without _precomputed_routing_steps) builds toMoves at L1920 and uses Path A in _execute_ms_gates. This is identical to old behavior.
- Gadget path (ionRoutingGadgetArch → ionRoutingWISEArch with _precomputed_routing_steps) also builds toMoves at L1920. If ms_round_index maps to a valid toMoves entry, Path A is used. Otherwise Path B (scan all) is used as fallback.
