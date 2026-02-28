# SAT Routing Fix Plan — Meeting the Spec

## Root Cause Analysis

The SAT solver hangs during CSS Surgery gadget phase routing due to a combination of:

1. **Dense exit BT constraints** from `_build_gadget_exit_bt()` pinning ALL ions to EC-return positions
   - Creates an extra empty return round `[[]]` appended to the MS pairs
   - All ions are pinned as hard constraints — on a packed merged grid this can be UNSAT or extremely slow
   
2. **Conservative SAT timeouts** (`_TIMEOUT_BASE_S = 300s` × difficulty) — each config can run 300+ seconds

3. **The `_patch_and_route` retry loop** — when progress stalls, it expands patches (`n_c += inc, n_r += inc`) and retries up to `max_cycles=10` times

4. **The BT-constrained problem is harder than the actual gadget MS routing** — the return round (empty pairs with dense BT) is the bottleneck, not the gate routing itself

## Key Files to Modify

1. **`gadget_routing.py`** — `_build_gadget_exit_bt()` at line 2389 and gadget phase routing at ~line 3120
2. **`qccd_WISE_ion_route.py`** — `_route_round_sequence()` at line 1350, `_patch_and_route()` at line 360
3. **`qccd_SAT_WISE_odd_even_sorter.py`** — timeout at line 313

## Fix Strategy (Spec-Compliant, No Heuristic Fallback)

### Fix A: Reduce SAT timeout for gadget phases
The 300s base timeout is designed for massive grids. For d=2 CSS Surgery merged grids (6×8 physical), a 30s timeout per config is more than enough. Add a `solver_timeout` parameter to `_route_round_sequence` that can be passed through.

### Fix B: Make exit BT a fast-fail with aggressive timeout
Instead of letting the BT-constrained SAT run for 300s+ before falling back to BT-free routing:
1. Try BT-constrained routing with a SHORT timeout (e.g., 30s)
2. If it times out/fails, immediately fall back to:
   a. Route gadget MS gates without BT (free placement)
   b. After all MS rounds, add explicit transition reconfig to return ions to EC positions
This matches the spec §3.5: "If a single-round gadget solve with BT pins is infeasible (SAT returns UNSAT), fall back to: 1. Solve gadget without BT pins (free placement) 2. Insert a dedicated transition round"

### Fix C: Separate the return round from the MS solve
Instead of appending `[[]]` to the MS pairs and letting the SAT solver handle both MS routing AND ion return in one call:
1. First, route ALL gadget MS pairs normally (no BT, just the MS gates)
2. Then, do a SEPARATE optimized reconfig-only step to return ions to EC positions
This separates concerns and makes each SAT call simpler.

### Fix D: Use soft BT instead of hard BT for return positions
The spec §6, Risk 1 says: "Start with BT as hard clause (user preference). Monitor SAT return codes; if UNSAT, log a warning and retry with loose BT (soft clause with high weight)."
However, the current `_build_gadget_exit_bt` only supports hard BTs. We should try soft BT first with high weight, falling back to hard only when feasible.

## Recommended Implementation Order

1. **Fix C** (Separate return round from MS solve) — This is the cleanest architectural fix:
   - Route gadget MS pairs using `_route_round_sequence` WITHOUT any exit BT
   - After MS gates solved, compute transition layout and add explicit reconfig step
   - This ensures the SAT solver only focuses on MS gate routing (tractable)
   - The transition routing is a simple subgrid-to-subgrid reconfig (fast)
   
2. **Fix B** (Fast-fail BT) as enhancement:
   - Try BT-constrained routing with aggressive timeout
   - Fall back to Fix C approach if it fails

## Current Code State Notes

- `route_full_experiment_as_steps()` is in `gadget_routing.py`:2520-3334
- Gadget phase routing starts at ~line 2970  
- `_build_gadget_exit_bt()` is at line 2389-2428
- The try/except at line 3130-3160 catches BT routing failure but may not trigger if SAT just hangs
- `_route_round_sequence()` is at `qccd_WISE_ion_route.py`:1350-1690
- `_patch_and_route()` is at `qccd_WISE_ion_route.py`:360-870
- `_TIMEOUT_BASE_S = 300.0` at `qccd_SAT_WISE_odd_even_sorter.py`:313
- Default `subgridsize=(6, 4, 1)` from routing_config.py:244
- Notebook CSS Surgery uses `subgridsize=(12, 12, 0)` with `inc=0`

## Block Sub-Grid Layout (CSS Surgery d=2)

3 blocks in L-shaped arrangement:
- Block 0: top-left
- Block 1: top-right  
- Block 2: bottom-right (L-shape)
Grid: 4×6 traps = 8×12 physical

Gadget phase (e.g., ZZ merge blk0+blk1): 
- Bbox covers 2 blocks + padding
- Merged grid ~6 rows × 8 cols physical
- With subgridsize=(12,12,0): entire grid is one patch — feasible for SAT
- With subgridsize=(6,4,1): patches of 4×6, needs 2+ patches — also feasible

The problem is NOT the MS gate routing itself but the BT-constrained return round.
