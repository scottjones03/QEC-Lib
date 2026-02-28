# Deep Research Findings — SAT Solver Pipeline Analysis

## Key Files
- `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py` (3529 lines)
- `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_SAT_WISE_odd_even_sorter.py` (4959 lines)
- `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py` (3492 lines)
- `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/routing_config.py`
- `src/qectostim/experiments/hardware_simulation/trapped_ion/demo/run.py` (670 lines)
- Notebook: `notebooks/trapped_ion_demo.ipynb` — cell 24 ID `#VSC-76f13df6`

## QUESTION 1: Why is lookahead 4 when we set it to 1?

### ANSWER: The lookahead IS correctly passed through. The value 4 is just the default on function signatures, never used.

**Full trace:**
1. Cell 24: `compile_gadget_for_animation(..., lookahead=2, ...)` (was changed from 1 to 2 in previous session)
2. `demo/run.py` L540: `WISERoutingConfig.default(lookahead=lookahead, ...)`
3. `trapped_ion_compiler.py` L937-941: `kwargs["lookahead"] = rc.lookahead` — this extracts it from the config
4. `trapped_ion_compiler.py` L1008: `ionRoutingGadgetArch(arch, wise_config, instructions, **kwargs)` — passes lookahead as kwarg
5. `qccd_WISE_ion_route.py` L2952: `def ionRoutingGadgetArch(..., lookahead: int = 4, ...)` — default 4 is OVERRIDDEN by kwarg
6. L3441: `route_full_experiment_as_steps(..., lookahead=lookahead, ...)` — passes it through
7. `gadget_routing.py` L2626: `_route_round_sequence(..., lookahead=min(lookahead, len(block_p_arr)), ...)`
8. `qccd_WISE_ion_route.py` L1569: `window_end = min(len(parallelPairs), lookahead + idx)` — THIS is where SAT window is sized

**If "lookahead=4" appeared in log output:** It would come from `ionRoutingWISEArch` which also defaults to `lookahead=4` at L1906. But this function also receives the correct value through kwargs. The "4" in the log is likely from the `WISERoutingConfig.default()` class default at `routing_config.py` L245: `lookahead: int = 2` — wait, that's 2 not 4.

**Actual answer:** The default on `ionRoutingGadgetArch` and `route_full_experiment_as_steps` is 4, but it's always overridden by the explicit kwarg from the routing config. If the user saw "lookahead=4" in the SAT output, it may have been from cell 22 (which calls `compile_gadget_for_animation` with `lookahead=1, subgridsize=(12,12,0)`) or from default parameter docs.

## QUESTION 2: Timeout formula scaling

### Current formula:
```python
# qccd_SAT_WISE_odd_even_sorter.py L251-396
var_estimate = num_ions × (n × m) × P_max × R
difficulty = max(1.0, sqrt(var_estimate / 36))
timeout = 300.0 × difficulty
```

### Key insight: R is the number of rounds in the SAT WINDOW, not the total rounds.
- `_patch_and_route` L387: `R = len(P_arr)` where P_arr = current window (not all rounds)
- The timeout is called at L881: `sat_time = _estimate_sat_timeout(n, m, total_cells, R, base_pmax)`
- And L3616: `max_sat_time = _estimate_sat_timeout(n, m, len(ions_set), R, base_pmax)`

### No ceiling exists currently - confirmed. The formula scales as sqrt(problem_size).

### User's complaint: "should improve the scaling"
- The var_estimate multiplies n×m by num_ions, but these are correlated (ions fill the grid). For a full grid, num_ions ≈ n×m×k, so var_estimate ≈ (n×m)² × k × P_max × R. This double-counts grid area.
- Better: var_estimate should use CLAUSE COUNT (which is the actual SAT difficulty) not variable count approximation.

## QUESTION 3: Return round handling

### Current mechanism:
- In `gadget_routing.py` L2619-2625 (EC phases): `block_p_arr = list(bp) + [[]]` — appends empty round
- Then `build_ion_return_bt_for_patch_and_route(block_layout, num_rounds=len(bp), ...)` creates BT pins
- Similarly L2648-2652 (single-grid EC): `p_arr_with_return = list(sr_pairs) + [[]]`

- In gadget phases L3191-3194: `_build_gadget_exit_bt(_bt_offsets, phase_pairs, ec_initial_layouts)` creates exit BTs and augmented P_arr.

### The return round IS bundled into the same SAT instance:
- `p_arr_with_return = list(sr_pairs) + [[]]` — the `[[]]` is the empty return round 
- BT pins constrain ions to return to starting positions during this empty round
- The SAT solver sees this as an extra round where no MS gates need to happen but all ions must reach specific positions

### User's suggestion: "return round should be solved as a SAT instance on its own"
- This is a good idea because:
  1. The return round has different constraints (position-only, no gate pairing)
  2. Bundling it with MS rounds inflates R by 1 in the difficulty estimate
  3. If the combined problem is UNSAT, you can't tell if it's the gates or the return that's infeasible
  4. A separate return-round SAT could use a specialized formulation (just find a reconfiguration from current layout to target layout)

## QUESTION 4: SAT window size scales with R not lookahead

### Current code:
```python
# qccd_WISE_ion_route.py L1569
window_end = min(len(parallelPairs), lookahead + idx)
P_arr = parallelPairs[window_start:window_end]
```

So the SAT window size IS `min(lookahead, remaining_rounds)`. It correctly scales with lookahead.

### But the TIMEOUT formula uses R = len(P_arr):
```python
# _patch_and_route L387
R = len(P_arr)  # This IS the window size, which is bounded by lookahead
```

So R in the timeout IS bounded by lookahead. BUT the `from_heuristics` method at L881 passes `R` to the timeout:
```python
sat_time = _estimate_sat_timeout(n, m, total_cells, R, base_pmax)
```

### The issue might be in the RETRY logic:
```python
# L1675-1705 retry
retry_window = min(len(remaining), max(2 * len(P_arr), _MAX_RETRY_WINDOW))
```
On retry, the window can DOUBLE up to _MAX_RETRY_WINDOW=6. This means R can grow beyond lookahead.

### User's point: "number of rounds in sat solver should scale with lookahead not R"
- The SAT window rounds ARE bounded by lookahead — this is correct
- But the RETRY path escalates the window beyond lookahead
- The return round adds +1 to R in every call
- Fix: cap retry window at `min(remaining, 2*lookahead)` instead of `max(2*len(P_arr), 6)`

## QUESTION 5: Cross-block pairs

### Origins:
1. `gadget_routing.py` L2297-2353: `_preprocess_gadget_pairs` — this is the key function
2. EC phases in `_route_ec_fresh`: pairs are split by `ion_to_block` mapping
3. Gadget phases: all pairs (including cross-block) are routed on merged grid

### The ion_to_block mapping:
```python
# gadget_routing.py L2499-2502
ion_to_block: Dict[int, str] = {}
if _use_per_block:
    for bname, sg in block_sub_grids.items():
        for iidx in sg.ion_indices:
            ion_to_block[iidx] = bname
```

### Cross-block pairs in EC phases:
```python
# L2579-2587 in _route_ec_fresh
for pair in pairs_in_round:
    a_idx, d_idx = pair
    blk_a = ion_to_block.get(a_idx)
    blk_d = ion_to_block.get(d_idx)
    if blk_a == blk_d and blk_a is not None:
        block_pairs_map[blk_a][round_i].append(pair)
    elif blk_a is not None:
        block_pairs_map[blk_a][round_i].append(pair)
    # PROBLEM: if blk_a != blk_d, the pair is assigned to blk_a only!
```

### Root cause options:
1. **partition_grid_for_blocks** assigns overlapping ion indices to multiple blocks
2. **Compiler's qubit→ion mapping** doesn't align with block partitions (compact_clustering assigns ions differently than partition_grid_for_blocks expects)
3. **Post-gadget ion positions**: after a gadget phase routes ions on a merged grid, ions from block_A may end up in block_B's region. The post-gadget transition reconfig tries to fix this but may fail.

### The `_build_gadget_exit_bt` function:
- `gadget_routing.py` L2258-2292: builds BT pins to force ions back to EC positions after gadget phase
- If BT causes UNSAT, falls back to routing without BT + post-gadget transition reconfig
- If the transition reconfig also fails, ions stay in wrong block regions → cross-block pairs in subsequent EC phases

## REMAINING WORK

### Still need to trace:
1. How `optimal_QMR_for_WISE` (L3359+) uses R for config enumeration
2. The actual SAT encoding to understand clause count vs variable count
3. `_preprocess_gadget_pairs` full logic
4. `build_ion_return_bt_for_patch_and_route` exact implementation
5. The connection between `_build_gadget_exit_bt` and return round BT

### Key function locations:
- `_route_round_sequence`: L1440-1730 in qccd_WISE_ion_route.py
- `_patch_and_route`: L360-550 in qccd_WISE_ion_route.py  
- `optimal_QMR_for_WISE`: L3359+ in qccd_SAT_WISE_odd_even_sorter.py
- `_estimate_sat_timeout`: L340-396 in qccd_SAT_WISE_odd_even_sorter.py
- `build_ion_return_bt_for_patch_and_route`: L889-944 in gadget_routing.py
- `_build_gadget_exit_bt`: L2258-2292 in gadget_routing.py
- `_preprocess_gadget_pairs`: L2297-2353 in gadget_routing.py
- `route_full_experiment_as_steps`: L2353-3492 in gadget_routing.py
