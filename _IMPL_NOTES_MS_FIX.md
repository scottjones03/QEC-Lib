# CRITICAL IMPLEMENTATION NOTES - Read before implementing fixes

## File Locations and Line Numbers

### qccd_WISE_ion_route.py (src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/)
- `ionRoutingGadgetArch()`: line 2753 - Phase-aware routing entry
- `ionRoutingWISEArch()`: line 1804 - Main execution loop with _precomputed_routing_steps support
- `_execute_ms_gates()`: ~line 2200 (nested inside ionRoutingWISEArch) - matches solved_pairs to ops
- `_drain_single_qubit_ops()`: ~line 1984 (nested inside ionRoutingWISEArch) 
- `_build_parallel_pairs()`: search for "def _build_parallel_pairs" - builds parallelPairs from toMoveOps
- `_route_round_sequence()`: ~line 1350 - SAT routing engine
- Execution loop: ~line 2355 (inside ionRoutingWISEArch) - iterates routing_steps by ms_round_index
  Pattern: for ms_round_idx, round_steps_iter in _groupby(routing_steps, key=lambda s: s.ms_round_index):
    -> _apply_layout_as_reconfiguration (GlobalReconfig)
    -> _drain_single_qubit_ops
    -> _execute_ms_gates(ms_round_idx, first_step.solved_pairs)
    -> For subsequent steps in same ms_round: apply reconfig + drain + execute_ms
  
### gadget_routing.py (src/qectostim/experiments/hardware_simulation/trapped_ion/utils/)
- `route_full_experiment_as_steps()`: line 2509 - Phase iteration loop
  - _route_ec_fresh() nested helper at ~line 2692
  - Gadget phase routing at ~line 2880 (look for "elif plan.phase_type == 'gadget':")
  - Uses _preprocess_gadget_pairs() for Fix 7 split + Bug 3/4 decompose
  - Per-block EC: decomposes into per-block routing steps, merges with _merge_block_routing_steps
  - EC caching: ec_cache maps round_signature → (starting_layout, steps)
- `_build_plans_from_compiler_pairs()`: ~line 1829 - builds PhaseRoutingPlan from parallelPairs

### trapped_ion_compiler.py (src/qectostim/experiments/hardware_simulation/trapped_ion/utils/)
- `route()`: ~line 849 - calls ionRoutingGadgetArch or ionRoutingWISEArch
  Contains WARNING: toMove_phase_tags length 40 != toMoveOps total 192

## Key Data Structures
- `parallelPairs`: List[List[Tuple[int,int]]] - each entry is one routing round containing (ion_a, ion_b) pairs
- `toMoveOps`: List[List[TwoQubitMSGate]] - same structure but actual operation objects
- `RoutingStep`: dataclass with layout_after, schedule, solved_pairs, ms_round_index, etc.
- `PhaseRoutingPlan`: has phase_type, ms_pairs_per_round, interacting_blocks, round_signature
- `phase_pair_counts`: List[int] - number of parallelPairs entries per phase

## What Compilation Actually Produces for CSS Surgery d=2
- 192 total CX pairs in ideal circuit
- 40 parallelPairs entries (each entry has 2-6 ion pairs)
- Phase pair counts: [0, 8, 12, 0, 12, 0, 0, 8, 0] sums to 40 ✓
- Phases: init(0), stab_pre(8), gadget(12), gadget(0), gadget(12), gadget(0), gadget(0), stab_post(8), measure(0)
- Compilation HANGS during route_full_experiment_as_steps() — SAT solver too slow on gadget phases

## The Actual Problem Chain
1. SAT solver hangs on gadget phases (bounding-box too large for CSS Surgery)
2. Even if SAT completes, solved_pairs may not cover all pairs (SAT unsatisfiable for some pairs)
3. _execute_ms_gates uses frozenset matching — works correctly if solved_pairs are right
4. Execution loop pattern (Reconfig→drain→MS) is correct but only runs once per routing step
5. If SAT doesn't solve all pairs in a round, those MS gates are silently dropped
6. User wants: EVERY MS gate in a round must be executed before moving to next round

## What Needs to Change
1. Add SAT timeout so compilation doesn't hang
2. Ensure routing fully covers all pairs per round (retry, fallback, or split further)
3. Verify ms_round_index numbering is consecutive and matches parallelPairs indices
4. After _execute_ms_gates, verify all expected MS gates were scheduled
5. Fix toMove_phase_tags warning (cosmetic but confusing)
