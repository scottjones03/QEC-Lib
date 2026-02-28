# Critical Implementation Context

## What was done
1. Added `_resolve_grown_stabs()` static method to CSSSurgeryCNOTGadget (after get_phase_pairs)
2. Added `BlockInfo` to layout imports  
3. Removed all fallback paths in compute_layout:
   - Removed `has_seam_api` check (all codes have it from base class)
   - Removed `add_boundary_bridges()` fallbacks for empty seam
   - Removed the `else` non-topological path  
   - Now raises ValueError/NotImplementedError for unsupported codes
4. Removed recompute fallbacks in _emit_zz_merge and _emit_xx_merge
   - Now raises RuntimeError if merge info not available
5. Wired grown CX in _emit_zz_merge and _emit_xx_merge
   - Replaces `grown_cx_per_phase = [[] for _ in range(n_phases)]` with 
     actual population from merge_info.grown_stabs
6. Left `_emit_joint_merge_round` sequential fallback for hierarchical builders

## Test issue
- `layout.build_allocation()` doesn't exist — should just skip that call
- The test needs to not rely on build_allocation
- Can call get_phase_pairs with phase_index and alloc=None (since alloc is only used
  if merge_info is None, which we removed)

## File modified
- `/Users/scottjones_admin/.../src/qectostim/gadgets/css_surgery_cnot.py` (1401 lines now)
- All syntax checks pass

## Changed sections in css_surgery_cnot.py
1. Import: Added `BlockInfo` to layout import (line 66)
2. `_resolve_grown_stabs` method: ~lines 241-302 (new static method)
3. `_emit_zz_merge`: grown_cx build from merge_info.grown_stabs (~line 620)
4. `_emit_zz_merge`: RuntimeError instead of recompute fallback (~line 598)
5. `_emit_xx_merge`: RuntimeError instead of recompute fallback (~line 745)
6. `_emit_xx_merge`: grown_cx build from merge_info.grown_stabs (~line 754)
7. `compute_layout`: single clean path instead of 4 fallbacks (~lines 1275-1340)
