# CSS Surgery CNOT Pipeline — Refactoring Plan

## Date: 2026-02-26

## Goal
Clean up the git diff: remove useless fallbacks, unify duplicated code paths,
fix silent error-swallowing, eliminate dead code, and give **one clean flow**
through the compilation pipeline.

---

## FILES ANALYZED (24 changed, +5007 / -817)

| File | Lines | Key Role |
|------|-------|----------|
| `gadget_routing.py` | 3183 (new) | Phase decomposition + routing engine |
| `qccd_WISE_ion_route.py` | 3239 (+2474) | Execution loop + ionRoutingGadgetArch |
| `trapped_ion_compiler.py` | 1060 (+511) | Stim → native decomposition + route dispatch |
| `pipeline.py` | 660 (+218) | QECMetadata, PhaseInfo, round signatures |
| `demo/run.py` | 774 (new) | Animation compilation helper |
| `best_effort_compilation_WISE.py` | 1882 (+721) | Best-effort sweep runner |
| `qccd_parallelisation.py` | 663 (+146) | Operation parallelizer |
| `qccd_SAT_WISE_odd_even_sorter.py` | 5001 (+281) | SAT solver |

---

## CRITICAL FINDINGS (must fix)

### C1. `continue` at gadget_routing.py ~L2520 drops routing steps entirely
**File**: `gadget_routing.py` line ~2520
**Problem**: When EC cache transition doesn't converge, fresh routing steps are
computed, but then `continue` skips:
1. `all_routing_steps.extend(phase_steps)` — steps never added to return value
2. `round_cursor += _phase_span` — cursor not advanced → index collision
3. `_global_round_offset += _phase_span` — progress tracking breaks

The freshly-computed EC routing results are **silently lost**. This is the
worst bug in the codebase — it means any cache-miss transition failure causes
an entire phase's operations to vanish.

**Fix**: Remove the `continue` and let execution fall through to the global
renumbering + extend at the bottom of the loop. Remove the dead
`all_phase_steps.append(phase_steps)` line above it.

### C2. Fix 8 (revised) `_use_bbox = True` is immediately overwritten
**File**: `gadget_routing.py` ~L2755-2770  
**Problem**:
```python
if plan.phase_type == 'gadget' and _use_level1_slicing:
    _use_bbox = True          # ← SET (Fix 8 revised)
# ...
_use_bbox = False             # ← OVERWRITTEN UNCONDITIONALLY (Fix 5/6)
```
The bbox override for gadget phases is dead code. Gadget phases only get bbox
routing if blocks span multiple row bands (the Fix 5/6 check). This means
gadget phases with same-row blocks use horizontal concatenation, which doesn't
preserve true spatial topology.

**Fix**: Move the unconditional `_use_bbox = False` to an `else` clause so
the gadget override survives. Or: restructure to compute `_use_bbox` once with
all conditions merged.

### C3. MS gate fallback executes physically-incorrect gate
**File**: `qccd_WISE_ion_route.py` L2235-2260  
**Problem**: When `getTrapForIons()` returns `None` (ions not co-located), the
code tries `op.ions[0].parent` as a trap anyway. If this "succeeds" (no
ValueError), a physically meaningless MS gate is counted as executed. The gate
pairs two ions that are NOT in the same trap.

**Fix**: If `getTrapForIons()` returns None, log an error and skip the gate.
Don't attempt the fallback — a successful "execution" of a non-co-located MS
gate produces invalid physics.

### C4. `toMoveIdx` desync in trapped_ion_compiler.py on skipped CX pairs
**File**: `trapped_ion_compiler.py` L508-540  
**Problem**: When a CNOT pair is skipped by `(ValueError, KeyError)` continue,
`toMoveIdx` never advances. All subsequent MS gates accumulate into the wrong
`toMoveOps` bucket, corrupting the CX-grouping that the SAT router relies on.

**Fix**: Track expected pair count per toMoveOps bucket. Advance `toMoveIdx`
when the expected count is reached OR when a pair is skipped.

---

## HIGH-PRIORITY FINDINGS (should fix)

### H1. `route_full_experiment_as_steps` is ~980 lines
**File**: `gadget_routing.py` L2205-3183  
**Problem**: Single function containing: EC cached path, cache-miss fallback,
per-stabilizer-round splitting, gadget L1-slicing bbox path, gadget L1-slicing
concat path, gadget full-grid path, and unknown-phase fallback — all inline.

**Fix**: Extract into helpers:
- `_route_ec_phase_cached(...)` 
- `_route_ec_phase_fresh(...)` 
- `_route_gadget_phase_l1(...)` 
- `_route_gadget_phase_fullgrid(...)` 
- `_apply_post_gadget_transition(...)` 
- `_build_exit_bt(...)` 

### H2. EC fresh routing code duplicated (cache-miss fallback vs non-cached)
**File**: `gadget_routing.py` L2472-2510 vs L2548-2647  
**Problem**: ~60 lines of identical per-stabilizer-round splitting logic in two
places. Cache-miss fallback is a copy-paste of the non-cached path.

**Fix**: Extract to `_route_ec_fresh(current_layout, phase_pairs, ...)` called
from both the cache-miss path and the non-cached path.

### H3. Post-gadget transition reconfig duplicated (L1 vs full-grid)
**File**: `gadget_routing.py` L2998-3030 vs L3109-3141  
**Problem**: Nearly identical ~30-line blocks for computing post-gadget
transition steps.

**Fix**: Extract to `_apply_post_gadget_transition(phase_steps, ...)`.

### H4. Exit BT dict construction duplicated (L1 vs full-grid)
**File**: `gadget_routing.py` L2863-2883 vs L3041-3061  
**Problem**: Identical ion-position → BT constraint dict assembly.

**Fix**: Extract to `_build_exit_bt(ec_initial_layouts, block_sub_grids, ...)`.

### H5. parallelPairs construction duplicated (WISE vs GadgetArch)
**File**: `qccd_WISE_ion_route.py` L1843-1885 vs L2810-2858  
**Problem**: ~40 lines of greedy round-building logic copy-pasted.

**Fix**: Extract to `_build_parallel_pairs(toMoveOps, ...)`.

### H6. Grid construction duplicated (WISE vs GadgetArch)
**File**: `qccd_WISE_ion_route.py` L1892-1916 vs L2862-2884  
**Problem**: ~25 lines of `oldArrangementArr`, `traps_sorted`, `ionsSorted`,
`active_ions` construction duplicated. GadgetArch's copy is partially unused.

**Fix**: Extract to `_initialize_grid_state(arch, wiseArch, operations)`.

### H7. Bare `except Exception` in auto-build block_sub_grids
**File**: `qccd_WISE_ion_route.py` L2750-2764  
**Problem**: Any exception in `partition_grid_for_blocks` is caught and logged
as warning, silently proceeding without block_sub_grids. A bug in the topology
code would silently degrade to flat routing with no clear diagnostic.

**Fix**: Narrow to specific expected exceptions (e.g. `ValueError` from bad
metadata) and re-raise anything else.

### H8. Bare `except Exception` in decompose_to_native
**File**: `trapped_ion_compiler.py` L288  
**Problem**: `stim.Circuit(stripped).decomposed()` failure causes raw
un-decomposed line to be inserted, likely misparse downstream.

**Fix**: Log a warning and attempt graceful handling rather than silent
fallback.

### H9. 10× silent `(ValueError, KeyError): continue` in gate dispatch
**File**: `trapped_ion_compiler.py` L426-540  
**Problem**: Every gate branch silently drops operations on missing-ion errors.
No diagnostic output makes debugging "missing gates" near-impossible.

**Fix**: Add `logger.debug(...)` before each `continue`. Count skipped ops.

### H10. Triplicated `_get_child_pids` function
**File**: `best_effort_compilation_WISE.py` L34, L1260, L1790  
**Problem**: Same function copy-pasted three times.

**Fix**: Extract to a shared utility module
(e.g. `trapped_ion/utils/process_utils.py`).

---

## MEDIUM-PRIORITY FINDINGS (nice to fix)

### M1. `_is_ec` check duplicated 3 times
**File**: `gadget_routing.py` L1225, L1879, L2418  
**Fix**: `_is_ec_phase_type(phase_type: str) -> bool` helper.

### M2. `all_phase_steps` populated but never read
**File**: `gadget_routing.py` L2394, L2520, L2730  
**Fix**: Remove entirely (or if intended for future use, document).

### M3. `_gadget_phase_counter` written but never read in route_full_experiment_as_steps
**File**: `gadget_routing.py` L2393, L3147  
**Fix**: Remove.

### M4. `derive_gadget_ms_pairs` calls `get_phase_pairs` with wrong signature
**File**: `gadget_routing.py` L259 — missing `qubit_allocation` arg  
**Fix**: Add `alloc` parameter (per repo memory, this was already flagged).

### M5. Data-qubit classification logic inverted in decompose_to_native
**File**: `trapped_ion_compiler.py` L361-362  
**Fix**: Use explicit `if qi in data_qubit_idxs:` check upfront.

### M6. `dataQubits.clear()` in "R" branch is dead code
**File**: `trapped_ion_compiler.py` L449  
**Fix**: Remove.

### M7. O(n²) frontier recomputation in paralleliseOperations
**File**: `qccd_parallelisation.py` L335-336  
**Fix**: Maintain incremental ready-set.

### M8. Debug `print()` left in _route_round_sequence
**File**: `qccd_WISE_ion_route.py` L1510  
**Fix**: Replace with `logger.debug()`.

### M9. Redundant `import logging` inside function bodies
**File**: `gadget_routing.py` L1310, L2312  
**Fix**: Use module-level logger consistently.

### M10. `_toMove_phase_tags` length can mismatch `toMoveOps`
**File**: `trapped_ion_compiler.py` L931-941  
**Fix**: Add assertion guard: `assert len(_toMove_phase_tags) == len(toMoveOps)`.

### M11. `compile_gadget_for_animation` is 346 lines
**File**: `demo/run.py` L406-752  
**Fix**: Decompose into grid-sizing, ion-assignment, pipeline-invocation, and
progress-widget helpers.

---

## LOW-PRIORITY FINDINGS

### L1. `can_merge_with_next` field on RoutingStep always False, never read
### L2. Commented-out "section 3" in ionRoutingWISEArch L1935-1947
### L3. _TYPE_ORDER dict re-created on every call (should be module-level)
### L4. Duplicate `_get_child_pids` also in qccd_SAT (~L490)
### L5. `arch_busy_until` ternary always 0.0 in parallelisation.py L309
### L6. `list.pop(0)` in paralleliseOperationsSimple (O(n²))
### L7. Float dict key collision workaround in parallelisation.py L546-552
### L8. Unused `SpectatorIon` import in run.py
### L9. Lazy `partition_grid_for_blocks` import inside function body

---

## PROPOSED EXECUTION ORDER

### Phase 1: Fix critical bugs (do first)
1. **C1** — Fix the `continue` that drops routing steps
2. **C2** — Fix the `_use_bbox` override being silently erased
3. **C3** — Remove MS gate fallback that produces invalid physics
4. **C4** — Fix `toMoveIdx` desync on skipped CX pairs

### Phase 2: Unify code paths (structural refactor)
5. **H1** — Break `route_full_experiment_as_steps` into helpers
6. **H2** — Unify EC routing (cache-miss vs non-cached)
7. **H3** + **H4** — Extract post-gadget transition + exit BT helpers
8. **H5** + **H6** — Extract shared WISE/GadgetArch helpers

### Phase 3: Narrow error handling
9. **H7** + **H8** — Narrow `except Exception` to specific types
10. **H9** — Add logging to silent `continue` blocks
11. **H10** — Extract `_get_child_pids` to shared utility

### Phase 4: Medium cleanup
12. **M1-M11** — Helper extraction, dead code removal, assertions

### Phase 5: Low-priority polish
13. **L1-L9** — Minor cleanups
