# GIT DIFF ANALYSIS — FULL REPORT
# Date: 2026-02-26
# Scope: src/qectostim/experiments/ folder (24 files, +5271/-1055 lines)

## FILES ANALYZED

### Core routing (biggest changes):
- qccd_WISE_ion_route.py: +2084/-743 lines (1943→3284)  
- gadget_routing.py: NEW untracked, 3133 lines
- pipeline.py: +218 lines
- trapped_ion_compiler.py: +586 lines
- best_effort_compilation_WISE.py: +749 lines
- qccd_SAT_WISE_odd_even_sorter.py: +306 lines
- qccd_parallelisation.py: +148 lines
- routing_config.py: +13 lines
- run.py (demo): +318 lines

## ============================================================
## CLASSIFICATION: SAFE TO COMMIT AS-IS
## ============================================================

### qccd_WISE_ion_route.py:
1. RoutingStep dataclass (L1380-1400) — clean, well-typed
2. _build_parallel_pairs extraction (L1736-1830) — correct
3. _build_grid_state extraction (L1822-1855) — trivial, correct
4. _find_ion_position, _ions_unmoved utilities (L1272-1300)
5. _remap_schedule_to_global (L2534-2575) — correct offset math
6. _merge_disjoint_block_schedules (L2576-2618) — index-aligned merge (already fixed this session)
7. _merge_block_routing_steps (L2619-2720) — well-structured block merge
8. _apply_layout_as_reconfiguration empty-cell fix (L1215-1249)
9. _rebuild_schedule_for_layout checkerboard tilings (L955-1135)
10. Fix 6: defensive sort before groupby (L2316-2319) — groupby needs consecutive keys
11. Fix 9: rotation batching XRot+YRot (L2115-2130) — correct physics
12. Fix 10: shared-ion round splitting (L2933-2985) — correct for bridge CX
13. End-of-routing ERROR assertions (L2430-2445, L2505-2520)

### pipeline.py:
- Interleaved round signature — GOOD
- Per-block stabilizer info — GOOD
- is_gadget property — GOOD
- cx_per_ec_round computation — GOOD
- PhaseInfo.cx_per_round, ms_pair_count fields — GOOD

### trapped_ion_compiler.py:
- Per-gate decomposition (fixes stim merge bug) — GOOD
- Multi-target gate iteration — GOOD
- RX/MX/MY/MRX handlers — GOOD
- C4 _advance_toMoveIdx_if_full — GOOD (fixes real toMoveIdx desync)
- _map_qubits_per_block per-block topology — GOOD
- Gadget dispatch in route() — GOOD
- _compiler_q2i real qubit→ion mapping (Fix 4) — GOOD
- _toMove_phase_tags construction (Fix 2) — GOOD

### qccd_SAT_WISE_odd_even_sorter.py:
- _in_notebook_env() helper — GOOD (fixes critical NameError)
- macOS Manager() guard — GOOD (fixes macOS deadlock)
- Feasibility pre-check — GOOD (perf optimization)
- In-process SAT fast path — GOOD
- Sparse-grid padding — GOOD
- R<=1 fix — GOOD
- ThreadPoolExecutor fallback — GOOD

### qccd_parallelisation.py:
- reorder_rotations_for_batching — GOOD algorithm
- Input list mutation fix (barriers.insert) — GOOD
- Float key collision guard — GOOD
- Incremental seg_end — GOOD (O(n) vs O(n²))

### routing_config.py:
- cache_ec_rounds field — GOOD

### gadget_routing.py (NEW):
- No if True: guards, no silent failures — GOOD
- All errors logged at warning level — GOOD
- Functionally correct routing pipeline

## ============================================================
## BUGS FOUND (Must fix before commit)
## ============================================================

### BUG 1 — CRITICAL: _execute_ms_gates silent pair drops
Location: qccd_WISE_ion_route.py L2250-2271
Problem: When solved_pairs don't match operations in operationsLeft,
logs ERROR but continues silently. MS gates permanently lost.
The ERROR log was added this session (was logger.debug before).
Still needs: return unmatched count so caller can act OR raise after
all phases if cumulative unmatched > 0.

### BUG 2 — MEDIUM: _TYPE_ORDER reordering (Resets before Rotations)
Location: qccd_WISE_ion_route.py L2030-2036
Problem: Changed from XRot:0, YRot:1, Reset:2, Meas:3 to
QubitReset:0, XRot:1, YRot:2, Meas:3. Docstring says "fast gates
(rotations ~5µs) first" but code does resets first.
Impact: Resets executing before pre-MS rotations could corrupt
state preparation within same epoch.

### BUG 3 — MEDIUM: _drain_single_qubit_ops over-blocking
Location: qccd_WISE_ion_route.py L2072-2080
Problem: Pre-computes blocked_ions from ALL remaining 2q gates,
then filters. Old code used interleaved scan where 1q ops BEFORE
their 2q gate (in operationsLeft order) were eligible.
New code is strictly more conservative.
Impact: May strand some pre-MS rotations that should execute.

### BUG 4 — HIGH: search_gadget_configs macOS deadlock
Location: best_effort_compilation_WISE.py search_gadget_configs()
Problem: Uses ctx.Manager() unconditionally. On macOS this deadlocks
(same bug fixed in optimal_QMR_for_WISE and SATProcessManager).
Must apply sys.platform == "darwin" guard.

### BUG 5 — LOW: Dead code _merge_reconfig_schedules
Location: qccd_WISE_ion_route.py L1304-1370
Problem: Defined but never called. Trick 5 explicitly DISABLED in
_route_round_sequence (L1705). Should be removed or marked TODO.

### BUG 6 — LOW: GR coalescing may lose barrier semantics
Location: qccd_WISE_ion_route.py L2449-2488
Problem: Barrier rebuild creates entirely new barriers based on type
transitions. Original barrier positions (routing round boundaries)
are discarded. May break downstream consumers relying on barrier semantics.

### BUG 7 — LOW: Exception swallowing in multiple locations
Locations:
- pipeline.py: except Exception on gadget phase pair counting → logger.debug (should be warning)
- trapped_ion_compiler.py: _try_notebook_progress except Exception: pass
- qccd_WISE_ion_route.py L2856-2860: block_sub_grids normalization → logger.debug
- qccd_WISE_ion_route.py L3189-3192: Fix 1 cross-validation except Exception → logger.debug
- qccd_SAT_WISE_odd_even_sorter.py: in-process SAT except Exception: pass (two locations)
- run.py: except Exception: pass (two locations)

## ============================================================
## DEFENSIVE FALLBACK PATTERNS (mask real bugs)
## ============================================================

### Pattern 1: _execute_ms_gates logs ERROR but continues
-> Produces output with missing MS gates silently

### Pattern 2: Fix 12 — fallback to flat routing on 0 solved pairs
Location: qccd_WISE_ion_route.py L3217-3228
-> If phase-aware router broken, falls through to flat routing.
User sees working but suboptimal result with no indication.

### Pattern 3: ionRoutingGadgetArch fallback on metadata parse failure
Location: qccd_WISE_ion_route.py L3078-3092
-> Any metadata issue → flat ionRoutingWISEArch. Metadata bugs never surface.

### Pattern 4: block_sub_grids normalization catch at debug level
Location: L2856-2860
-> If q2i mapping wrong, normalization silently skips, downstream pair matching fails.

### Pattern 5: Cross-validation catch-all except Exception
Location: L3189-3192
-> Entire Fix 1 cross-validation can crash silently.

### Pattern 6: run_single_gadget_config outer except Exception
-> Returns np.nan results, production search never sees crashes.

## ============================================================
## CODE QUALITY CONCERNS
## ============================================================

### 1. route_full_experiment_as_steps: 927 lines
-> Should be broken into sub-functions (ec_fresh, gadget_merge, etc.)

### 2. compile_gadget_for_animation in run.py: ~320 lines
-> Heavy duplication with trapped_ion_compiler.py::_map_qubits_per_block
-> Bridge ancilla heuristic uses hardcoded block name patterns

### 3. Branching fallback paths in ionRoutingGadgetArch
At least 4 paths to "success":
a) Phase-aware + block routing → route_full_experiment_as_steps (nominal)
b) Flat fallback when metadata unavailable (L3078-3092)
c) Flat fallback when 0 solved pairs in phase-aware (Fix 12, L3217-3228)
d) Direct RoutingStep injection via _precomputed_routing_steps
Each produces different quality results but the same API response.

### 4. compile_gadget_for_animation bridge ancilla logic
Uses string matching on merge type names ("zz_merge", "xx_merge").
Fragile for future gadgets with different naming.

## ============================================================
## WHAT'S CAUSING UNSOLVED MS GATE PAIRS?
## ============================================================

Root cause chain (from analysis):

1. Path A (routing) derives ion pairs from qec_metadata → decompose_into_phases
   → ms_pairs_per_round → _route_round_sequence → solved_pairs
   
2. Path B (execution) has ion pairs from TrappedIonCompiler.decompose_to_native
   → toMoveOps → operationsLeft → TwoQubitMSGate.ions

3. When Path A pairs ≠ Path B pairs → _execute_ms_gates can't match → silent drop

Why they differ:
- Path A uses qubit_to_ion built from metadata (may use q+1 convention)
- Path B uses actual Ion.idx from compact clustering (ion.idx ≠ q+1)
- Fix 4 added _compiler_q2i to bridge this, but cross-validation is advisory only
- If normalization catch (L2856-2860) silently fails, indices stay mismatched

Additionally even if pairs DO match:
- The C3 fix correctly skips MS gates where ions aren't co-located after reconfig
- But the skip has no retry mechanism — pair is consumed from solved_pairs but
  operation stays in operationsLeft with no future routing attempt
- Old code force-executed even when not co-located (wrong physics, but no drops)

## ============================================================
## WHAT'S CAUSING MISSED SINGLE-QUBIT GATES?
## ============================================================

1. _TYPE_ORDER change: Resets drain before rotations (BUG 2)
   -> Pre-MS rotations may be delayed past their epoch window

2. Blocked-ions over-conservative scan (BUG 3)  
   -> All ions in ANY remaining 2q gate are blocked
   -> 1q ops that should execute before their 2q gate get stranded

3. C1 epoch ceiling was already restored this session
   -> This addresses the global ordering constraint

4. If an MS gate is silently dropped (BUG 1), its post-MS rotations
   never see their epoch become eligible, and may be stranded forever.

## ============================================================
## RECOMMENDED FIX PLAN (DO NOT IMPLEMENT YET)
## ============================================================

### Tier 1: Commit-safe changes (no risk)
- Commit the GOOD items listed above
- These include all helper extractions, dataclasses, SAT fixes,
  parallelisation improvements, pipeline metadata additions

### Tier 2: Quick bug fixes (low risk)
- BUG 2: Restore _TYPE_ORDER to XRot:0, YRot:1, Reset:2, Meas:3
- BUG 4: Add macOS guard in search_gadget_configs
- BUG 5: Remove dead _merge_reconfig_schedules or add TODO comment
- BUG 7: Upgrade all bug-masking logger.debug to logger.warning

### Tier 3: Medium fixes (moderate risk)
- BUG 1: Make _execute_ms_gates return unmatched_count; add
  end-of-compilation assert that total_unmatched == 0
- BUG 3: Either restore interleaved scan or document why conservative
  blocking is intentional
- BUG 6: Add flag to disable GR coalescing for animation consumers

### Tier 4: Architecture improvement (higher risk, bigger payoff)
- Consolidate compile_gadget_for_animation duplication with compiler
- Break route_full_experiment_as_steps into sub-functions
- Reduce ionRoutingGadgetArch fallback paths: either phase-aware works
  or fail loudly (no silent flat fallback for development)
- Make Fix 12 flat fallback opt-in via config flag
- Standardize exception handling: warning for recoverable, error for
  data-loss situations, never debug for correctness-affecting failures
