# FINAL STATUS & REMAINING WORK

## FIXES ALREADY APPLIED

### Fix 1+3: _execute_ms_gates replacement (DONE)
**File**: `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`
**What**: Replaced `_execute_ms_gates` nested helper in `ionRoutingWISEArch` with:
- Path A (toMoves available): iterates `toMoves[round_idx]`, uses old-style `any(all(ion.idx in p for ion in op.ions) for p in solved_pairs)` contains-check, has `op.ions[0].parent` fallback
- Path B (no toMoves entry): scans all operationsLeft with frozenset match (for precomputed gadget steps), also has fallback

### Fix 2: Remove _merge_phase_pairs (DONE)  
**File**: `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py`
**What**: Replaced `phase_pairs = _merge_phase_pairs(phase_pairs)` with a comment explaining why merging is removed, keeping `n_pairs = len(phase_pairs)` intact.

## VERIFICATION NEEDED
Run: `cd "/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim" && PYTHONPATH=src WISE_INPROCESS_LIMIT=999999999 my_venv/bin/python -u _verify_ms_fix.py 2>&1`

If non-gadget path works, also test gadget path with TransversalCNOT.

## ROOT CAUSES FIXED
1. **RC1**: `_execute_ms_gates` was using frozenset exact match scanning ALL operationsLeft (new code) instead of iterating `toMoves[idx]` with contains-check (old working code). Fixed by restoring old-style `toMoves[round_idx]` iteration as Path A.
2. **RC2**: `_merge_phase_pairs()` at gadget_routing.py was merging rounds, changing pair count. This broke the 1:1 correspondence between routing steps and toMoves entries. Fixed by removing the merge call.  
3. **RC3**: No `getTrapForIons()` fallback — ions that were close but not perfectly co-located had MS gates dropped silently. Fixed with `op.ions[0].parent` fallback like old code.

## KEY FILE PATHS
- `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py` (~3355 lines)
- `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py` (~3274 lines)
- Working baseline commit: `243a188`
- Current HEAD with fixes: modified from `41fb858`

## OTHER CONTEXT FILES (can be cleaned up later)
- `_FIX_STATUS.md` - earlier status
- `_IMPL_GUIDE.md` - implementation guide
- `_FIX_CONTEXT.md` - root cause analysis
- `_COMPREHENSIVE_FIX_PLAN.md` - full fix plan
- `_verify_ms_fix.py` - verification script
