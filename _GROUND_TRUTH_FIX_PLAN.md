# Comprehensive SAT Pipeline Analysis & Fix Plan
## Ground-Truth Code-Traced Analysis

---

## Question 1: Why is lookahead 4 when we set it to 1 in the cell?

### Answer: It isn't. The value IS correctly passed. Here's the exact trace:

```
Cell 24: compile_gadget_for_animation(..., lookahead=2, ...)   [was 1, changed to 2 in prev session]
  → demo/run.py: WISERoutingConfig.default(lookahead=2)
    → trapped_ion_compiler.py L937-941: kwargs["lookahead"] = rc.lookahead  => 2
      → qccd_WISE_ion_route.py L2952: ionRoutingGadgetArch(..., lookahead=2)  [default=4 OVERRIDDEN]
        → gadget_routing.py L2485: route_full_experiment_as_steps(..., lookahead=2)
          → gadget_routing.py L2648: _route_round_sequence(..., lookahead=min(2, len(block_p_arr)))
            → qccd_WISE_ion_route.py L1569: window_end = min(len(parallelPairs), 2 + idx)
```

**The `lookahead=4` default on function signatures is never used.** If you saw "lookahead=4" in log output, the most likely explanation is that Cell 22 (the TransversalCNOT cell) was run before Cell 24 — Cell 22 calls `compile_gadget_for_animation(..., lookahead=1, subgridsize=(12, 12, 0))` but the underlying `ionRoutingGadgetArch` default is `4`. However, even there, the explicit kwarg overrides it.

**Current cell 24 value:** `lookahead=2` (was changed from 1). Should be changed back to 1 if that's what you want.

---

## Question 2: Timeout formula — no ceiling, but improve scaling

### Current formula (qccd_SAT_WISE_odd_even_sorter.py L280-306):
```python
var_estimate = num_ions × (n × m) × P_max × R
difficulty = max(1.0, sqrt(var_estimate / 36))
timeout = 300.0 × difficulty     # _TIMEOUT_BASE_S = 300.0
```

### Problem: var_estimate double-counts grid area
- `num_ions` ≈ number of non-empty cells ≈ n × m (for a full grid)
- So `var_estimate ≈ (n×m)² × P_max × R` — grid area is squared
- This makes timeout scale as `O(n²m²)` instead of `O(nm)`

### Better approach — clause-count-based scaling:
The actual SAT difficulty is determined by the **number of CNF clauses**, not the variable count approximation. The CNF has:
- `n × m × (n×m)` assignment clauses (each cell gets one ion)
- `R × P_max × n × m` reconfiguration clauses per round
- `|P_arr[r]|` gate-pairing clauses per round
- BT pin clauses

A more accurate estimator would compute clause_count directly from these parameters, or use:
```python
var_estimate = max(1, n * m) * max(1, P_max) * max(1, R)  # drop num_ions term
```

### Confirmed: No ceiling exists. The formula has no cap.

### Env var overrides (cell 24):
Cell 24 sets `WISE_MAX_SAT_TIME=30` and `WISE_MAX_RC2_TIME=30` via env vars. These are checked at L3600-3606:
```python
max_sat_time = float(os.environ.get("WISE_MAX_SAT_TIME", max_sat_time))
max_rc2_time = float(os.environ.get("WISE_MAX_RC2_TIME", max_rc2_time))
```
So the env vars **override** the formula. These are safety caps, not the formula itself.

---

## Question 3: Return round — should be its own SAT instance

### How it currently works:

**EC phases** (gadget_routing.py L2648, L2783):
```python
block_p_arr = list(bp) + [[]]      # append empty round
block_bts = build_ion_return_bt_for_patch_and_route(block_layout, num_rounds=len(bp))
# BTs = [empty_dict, empty_dict, ..., {(0,0): pin_map}]
# Then _route_round_sequence sees P_arr with an extra empty round at end,
# with BT pins constraining all ions back to starting positions.
```

**Gadget phases** (gadget_routing.py L2288-2296 in `_build_gadget_exit_bt`):
```python
p_arr = list(phase_pairs) + [[]]   # append empty exit/return round
bts = [empty_dict, ..., {(0,0): exit_bt}]  # BT pins on last round
```

**The problem:**
1. The return round is bundled as an extra round in the SAME SAT instance as the MS gates
2. R = len(phase_pairs) + 1 — the `+1` inflates R in the difficulty/timeout calculation
3. The SAT solver sees the return round as "just another round" with 0 gates but heavy BT constraints
4. If the combined problem is UNSAT, you can't distinguish: "MS gates are infeasible" vs "return placement is infeasible"
5. The SAT solver wastes capacity_factor configs and P_max exploration on the return round which has fundamentally different constraints (position-only, no gate pairing)

### Why the user's suggestion is correct:
The return round is a **pure ion reconfiguration problem** — move all ions from their current (post-MS-gates) positions to target positions. This is:
- A simpler SAT problem (no gate-pairing constraints, just permutation)
- Already implemented as a separate SAT call in `_rebuild_schedule_for_layout` (used by `_compute_transition_reconfig_steps`)
- Could use its own timeout/difficulty parameters tuned for reconfiguration

### Proposed fix:
After `_route_round_sequence` completes the MS rounds, call `_rebuild_schedule_for_layout(final_ms_layout, wiseArch, target_layout)` to compute the return reconfiguration separately. Remove the `+ [[]]` appended round and the BT pins from the main SAT instance.

### Exact code locations to change:
1. `gadget_routing.py` L2648: `block_p_arr = list(bp) + [[]]` → `block_p_arr = list(bp)`
2. `gadget_routing.py` L2644: `block_bts = build_ion_return_bt_for_patch_and_route(...)` → remove
3. `gadget_routing.py` L2653: `initial_BTs=block_bts` → remove
4. After L2667 (_route_round_sequence returns): add a separate `_rebuild_schedule_for_layout(final_layout, block_arch, block_target)` call
5. Same pattern for:
   - L2783: `p_arr_with_return = list(sr_pairs) + [[]]` (single-grid EC)
   - L3362-3367: `p_arr_for_solve, bts = _build_gadget_exit_bt(...)` (gadget phase)

### Risk:
The current BT-pinned return round has the advantage that the SAT solver can co-optimize the last MS round's layout to make the return feasible. With a separate SAT call, the MS rounds might produce a layout that's hard to return from. Mitigation: use `_rebuild_schedule_for_layout` which already has 3-level fallback.

---

## Question 4: SAT rounds should scale with lookahead not R

### Current behavior:
The SAT window IS bounded by lookahead:
```python
# qccd_WISE_ion_route.py L1569
window_end = min(len(parallelPairs), lookahead + idx)
P_arr = parallelPairs[window_start:window_end]  # len(P_arr) ≤ lookahead
```

So R passed to `_patch_and_route` IS `min(lookahead, remaining_rounds)`. The timeout formula correctly uses this R.

### BUT: Two places inflate R beyond lookahead:

**1. The return round adds +1:**
```python
block_p_arr = list(bp) + [[]]  # R becomes len(bp) + 1
```
So even with `lookahead=1`, R=2 is passed to the SAT solver. With `lookahead=2`, R=3.

**2. The retry logic at L1675-1705 can escalate the window:**
```python
_MAX_RETRY_WINDOW = 6
retry_window = min(len(remaining), max(2 * len(P_arr), _MAX_RETRY_WINDOW))
```
So on retry failure, the SAT window can grow to `2*lookahead` or up to `_MAX_RETRY_WINDOW=6`, whichever is larger. This defeats the original intent of limiting to `lookahead` rounds.

### The real issue:
The user is saying: if I set `lookahead=1`, the SAT solver should solve **exactly 1 round at a time**, not 2 (with return) or 6 (on retry). The window size is the PRIMARY lever for controlling SAT difficulty.

### Fix:
1. Separate the return round (Question 3 fix eliminates the +1)
2. Cap retry window: `retry_window = min(len(remaining), 2 * lookahead)` — never exceed `2 * lookahead` on retry
3. Or better: on retry with same window size, try different P_max/capacity configs rather than expanding the window

---

## Question 5: Cross-block pairs

### How cross-block pairs arise — two root causes:

**Root cause A: Gadget phases create cross-block MS gates by design**

CSS Surgery gadgets have MS gates between ions in different blocks (bridge operations). These are intentionally cross-block. The code at `_preprocess_gadget_pairs` (L2353) separates them:
```python
if _ba != _bb:
    _cross.append(_p)  # cross-block pair
else:
    _blk_pairs[_ba].append(_p)  # intra-block pair
```

These cross-block pairs are routed on the **merged bounding-box grid** (L3280-3370). This is correct behavior for gadget phases.

**Root cause B: Post-gadget ion displacement — ions in wrong block regions**

After a gadget phase:
1. Ions from block_A may end up in block_B's region (the merged grid allowed free movement)
2. The `_build_gadget_exit_bt` function tries to pin ions back, but if BT routing fails...
3. The fallback at L3385 routes WITHOUT BT pins → ions stay displaced
4. `_apply_post_gadget_transition` (L2820) tries to move them back via `_compute_transition_reconfig_steps`
5. If THAT also fails (L2190 catches exceptions), the transition returns empty → ions stay in wrong blocks
6. Next EC phase: when splitting pairs by `ion_to_block`, an ion physically in block_B's region but logically belonging to block_A creates a cross-block pair in what should be a pure intra-block EC round

### The check at L2173-2193 (`blocks_pure`):
```python
blocks_pure = True
for bname, sg in block_sub_grids.items():
    block_slice = current_layout[r0:r1, c0i:c1i]
    for r, c in itertools.product(range(rows), range(cols)):
        ion_idx = int(block_slice[r, c])
        if ion_idx != 0:
            owner = ion_to_block.get(ion_idx)
            if owner != bname:
                blocks_pure = False
```

When `blocks_pure = False`, the transition reconfig uses the **full-grid** approach (L2231), which is correct but slower and may still fail.

### The user's two hypotheses:
1. **"initial ion mapping is not mapping blocks to disjoint regions"** — The `partition_grid_for_blocks` function assigns grid regions, and `ion_to_block` maps ions to blocks. If two blocks share a grid region (overlap), yes. But looking at the code, `partition_grid_for_blocks` uses `allocate_block_regions` which computes disjoint rectangles. This should be correct.

2. **"route back algorithm after gadget phase does not correctly return blocks"** — **This is the more likely cause.** The `_build_gadget_exit_bt` creates BT pins → SAT fails → fallback routes without BT → `_apply_post_gadget_transition` tries but may fail → ions remain displaced → cross-block EC pairs appear.

### Fix:
1. Make `_build_gadget_exit_bt` more robust (its BT pins may be too restrictive for the SAT solver — try pinning only active ions, more relaxed constraints)
2. If BT routing fails, make `_apply_post_gadget_transition` mandatory and robust (not allowed to silently return empty)
3. Add a diagnostic assertion: after every gadget → EC transition, verify `blocks_pure = True`. If not, log exactly which ions are displaced and where they should be.
4. Consider: instead of BT pins in gadget SAT, always use a separate return-round SAT call (Question 3 fix)

---

## Concrete Fix Plan

### Fix A: Separate Return Round SAT (HIGH PRIORITY)
**Files:** `gadget_routing.py`
**What:** Remove `+ [[]]` and BT pins from main SAT instance. After MS routing completes, call `_rebuild_schedule_for_layout(final_layout, arch, target_layout)` to compute return reconfiguration as a separate, smaller SAT problem.
**Locations:**
- L2644-2653: EC per-block routing — remove `build_ion_return_bt_for_patch_and_route`, remove `+ [[]]`, add post-SAT return call
- L2780-2790: EC single-grid routing — same pattern
- L2288-2296 (`_build_gadget_exit_bt`): keep function for `_apply_post_gadget_transition` target computation, but don't bundle exit BT into main SAT
- L3362-3367: gadget phase routing — use `phase_pairs` directly, no `p_arr_for_solve`
**Impact:** R decreases by 1 in every SAT call. Timeout decreases. Return becomes independently solvable.
**Risk:** MS layout may not be return-friendly. Mitigation: `_rebuild_schedule_for_layout` has 3-level fallback.

### Fix B: Cap Retry Window to 2×lookahead (HIGH PRIORITY)
**File:** `qccd_WISE_ion_route.py` L1675-1705
**What:** Change `_MAX_RETRY_WINDOW = 6` cap to `max(2 * lookahead, 4)` where `lookahead` is available in scope.
**Current:**
```python
retry_window = min(len(remaining), max(2 * len(P_arr), _MAX_RETRY_WINDOW))
```
**New:**
```python
retry_window = min(len(remaining), max(2 * len(P_arr), 2 * lookahead, 4))
```
**But wait:** `lookahead` is not in scope inside `_route_round_sequence` — it IS in scope. The function receives `lookahead` as a parameter (L1447).
**Impact:** Prevents retry from escalating to 6 rounds when lookahead=1.

### Fix C: Improve Timeout Scaling (MEDIUM PRIORITY)
**File:** `qccd_SAT_WISE_odd_even_sorter.py` L280-306
**What:** Remove `num_ions` from var_estimate (it double-counts grid area for full grids):
```python
# Current:
var_estimate = num_ions × (n × m) × P_max × R
# New:
var_estimate = max(1, n * m) × max(1, P_max) × max(1, R)
_REFERENCE = 6.0  # baseline: 3×2 grid, P_max=1, R=1
```
**Impact:** More accurate timeout for real problems. D=2 baseline stays ~300s. Large grids get less inflated timeouts.

### Fix D: Robust Post-Gadget Ion Return (MEDIUM PRIORITY)
**File:** `gadget_routing.py`
**What:** After gadget phase routing, ALWAYS verify `blocks_pure`. If not pure, call `_compute_transition_reconfig_steps` with a mandatory success requirement (retry with larger grid/P_max if needed). Log warnings with exact displaced ions.
**Location:** L3390-3430 (after gadget phase re-embedding)
**Impact:** Prevents cross-block pairs in EC phases caused by post-gadget ion displacement.

### Fix E: Add blocks_pure Diagnostic (LOW PRIORITY)
**File:** `gadget_routing.py`
**What:** After every phase transition, log ion block purity status. When impure, log which ions are displaced.
**Impact:** Debugging aid — makes it easy to verify Fix D is working.

### Fix F: Restore Cell 24 lookahead=1 (TRIVIAL)
**File:** `notebooks/trapped_ion_demo.ipynb`, cell 24
**What:** Change `lookahead=2` back to `lookahead=1` in the `compile_gadget_for_animation` call.
**Impact:** Matches user's original intent.
