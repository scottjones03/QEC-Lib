# CRITICAL ANALYSIS: SAT-Based Fix for MS Gate Scheduling (No Heuristic Fallback)

## User Requirement
- SAT routing MUST work — no heuristic fallback allowed
- SAT works fine for memory experiments (flat ionRoutingWISEArch path)
- Under gadget slicing, it should work identically — just with phase-aware decomposition
- Pattern: GlobalReconfig → all parallel MS gates → GlobalReconfig → all parallel MS gates → ...
- NO MS gates should be dropped

## Complete Code Flow Understanding

### Working Path (Memory Experiments)
1. `trapped_ion_compiler.py route()` L849 → calls `ionRoutingWISEArch()`
2. Inside `ionRoutingWISEArch()` (qccd_WISE_ion_route.py L1804):
   a. `_build_parallel_pairs()` L1914: builds parallelPairs from toMoveOps
   b. `_build_grid_state()` L1921: encodes ion positions into oldArrangementArr
   c. `_route_round_sequence()` L1935: SAT routes ALL rounds at once with lookahead windows
   d. Execution loop L2342: for each ms_round_index group:
      - Apply reconfig → drain 1q → execute MS gates
   e. `_execute_ms_gates()` L2200: matches solved_pairs to operationsLeft by frozenset(ion.idx)
   THIS WORKS CORRECTLY

### Broken Path (CSS Surgery Gadget)
1. `trapped_ion_compiler.py route()` L849 → detects qec_metadata → calls `ionRoutingGadgetArch()`
2. Inside `ionRoutingGadgetArch()` (qccd_WISE_ion_route.py L2753):
   a. Builds phase_pair_counts from qec_metadata phases
   b. Calls `_build_plans_from_compiler_pairs()` (gadget_routing.py L1829)
      - Slices parallelPairs by phase_pair_counts into PhaseRoutingPlans
   c. Calls `route_full_experiment_as_steps()` (gadget_routing.py L2509)
      - For each phase: routes using `_route_round_sequence()` on phase-specific sub-grid
      - For EC phases: uses per-block sub-grids (small → SAT works)
      - For GADGET phases: extracts bounding-box of interacting blocks → CAN BE TOO LARGE → SAT HANGS
   d. Passes precomputed routing_steps to `ionRoutingWISEArch()` via `_precomputed_routing_steps`
   e. `ionRoutingWISEArch()` uses precomputed steps directly (L1932-1933)
   f. Same execution loop as memory path
   THE HANG IS IN STEP c FOR GADGET PHASES

## Why SAT Hangs for Gadget Phases

In `route_full_experiment_as_steps()` (gadget_routing.py), gadget phases use:
```python
# For gadget phases with multi-block interaction:
# 1. Extract bounding-box of all interacting blocks
# 2. Build merged_layout from block sub-grids
# 3. Call _route_round_sequence() on the FULL merged layout
```

The bounding-box for CSS Surgery can span the ENTIRE grid (all blocks participate in merge/split).
With a full-size grid, `_route_round_sequence()` calls `_patch_and_route()` which tiles the grid
into patches of `subgridsize` and runs SAT on each patch. But:
- The subgridsize may be too large for the grid
- The number of ions is larger (48 ions for d=2 CSS Surgery)
- Binary search on D is slow for complex configurations

## REAL Root Causes (Detailed)

### Root Cause 1: Grid too large for gadget phases in route_full_experiment_as_steps
The bounding-box extraction creates a grid that encompasses ALL interacting blocks.
For CSS Surgery d=2: 3 blocks in L-shape → bounding box = entire 4×6 grid.
The SAT solver for this grid size with 48 ions is very slow but should be solvable
with proper subgridsize tiling.

### Root Cause 2: Incorrect subgridsize passed to _route_round_sequence
The subgridsize parameter from the user's compilation config may not be optimal for
the gadget phase's grid dimensions. For a 4×6 grid with subgridsize=(12,12,0), the
entire grid fits in one patch → no tiling benefit → SAT sees the full problem.

### Root Cause 3: parallelPairs indices mismatch between plans and execution
When `_build_plans_from_compiler_pairs` slices parallelPairs by phase_pair_counts,
the routing steps get ms_round_index values relative to each phase's slice.
But the execution loop in ionRoutingWISEArch indexes into the GLOBAL toMoves array.
If ms_round_index doesn't match toMoves indices, _execute_ms_gates can't find the right ops.

### Root Cause 4: Phase pair counts may not exactly match parallelPairs slicing
If phase_pair_counts doesn't align with parallelPairs boundaries correctly,
some pairs may be assigned to wrong phases or dropped entirely.

## CORRECT Fix Plan (SAT-based, No Heuristic)

### Fix 1: Ensure proper subgridsize for gadget phases
In `route_full_experiment_as_steps()`, when routing gadget phases:
- The bounding-box sub-grid should be tiled with appropriate subgridsize
- For CSS Surgery d=2 with 4×6 traps grid: use subgridsize=(4,3,0) to tile into 
  manageable patches (each 3×4 = 12 traps, far below SAT complexity limit)
- KEY: The subgridsize for gadget phases should be derived from the sub-grid dimensions
  NOT from the global routing config. Use min(config_subgrid, grid_dimension) for each axis.

### Fix 2: Fix ms_round_index to match global parallelPairs indices
In `route_full_experiment_as_steps()`, the RoutingStep.ms_round_index should be
set to the GLOBAL parallelPairs index, not the phase-local index.
The execution loop groups by ms_round_index and the solved_pairs must match
the operations at that global index.

### Fix 3: Verify phase_pair_counts alignment with parallelPairs
Ensure that `_build_plans_from_compiler_pairs()` correctly slices parallelPairs
using phase_pair_counts, and that cursor advancement is correct.
The phase_pair_counts should sum to len(parallelPairs).

### Fix 4: Fix the flat-routing guard in ionRoutingGadgetArch  
Previous memory says there was a diagnostic `if True:` at ~L3023 that forced flat routing.
Verify this is correctly restored to `if not phases or phase_pair_counts is None:`.

### Fix 5: Ensure bridge ancilla ions are included in grid state
For CSS Surgery, bridge ancilla qubits need to be in the grid for routing.
Verify that `_build_grid_state()` or the gadget routing's merged_layout includes
bridge ancilla ions correctly.

### Fix 6: Ensure SAT timeout doesn't hang forever
Add a reasonable timeout to the SAT solver calls (e.g., 60s per patch) that
raises an error if exceeded, rather than hanging silently. This is NOT a
heuristic fallback — it's a diagnostic safeguard.

## Key File Locations
- qccd_WISE_ion_route.py: ionRoutingWISEArch L1804, ionRoutingGadgetArch L2753,
  _execute_ms_gates L2200, _route_round_sequence L1350, _build_parallel_pairs ~L1684
- gadget_routing.py: route_full_experiment_as_steps L2509, _build_plans_from_compiler_pairs L1829,
  _preprocess_gadget_pairs ~L2462, _route_ec_fresh ~L2692
- trapped_ion_compiler.py: route() L849

## Implementation Priority
1. Fix 4 (verify flat-routing guard)
2. Fix 1 (proper subgridsize for gadget phases) 
3. Fix 2 (ms_round_index global alignment)
4. Fix 3 (phase_pair_counts validation)
5. Fix 5 (bridge ancilla inclusion)
6. Fix 6 (SAT timeout safeguard)
