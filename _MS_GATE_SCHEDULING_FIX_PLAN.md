# COMPREHENSIVE FIX PLAN: MS Gate Scheduling in CSS Surgery Compilation

## Problem Statement
Each round of CX/MS gates should schedule ALL parallel MS gates before moving to the next round:
```
GlobalReconfig → parallel MS gates → GlobalReconfig → parallel MS gates → ...
```
But currently MS gates are being dropped completely.

## Root Cause Analysis

### Finding 1: phase_pair_counts Mismatch
The warning `toMove_phase_tags length 40 != toMoveOps total 192` is the smoking gun.
- `toMoveOps` has 192 entries (one per CX pair = one MS operation)
- `toMove_phase_tags` has 40 entries (one per CX *instruction* = groups of 4-6 pairs per instruction)
- The `_build_parallel_pairs()` groups ops by toMoveOps entries → 40 parallelPairs
- But `phase_pair_counts` from QECMetadata phases counts INDIVIDUAL CX pairs, not instructions
- `phase_pair_counts = [0, 8, 12, 0, 12, 0, 0, 8, 0]` sums to 40 — matching parallelPairs, so THIS is correct

### Finding 2: Compilation Hangs
The compilation hangs during `route_full_experiment_as_steps()` — likely the SAT solver taking too long on the CSS Surgery bounding-box sub-grid for gadget phases.

### Finding 3: Ion Mapping is Correct
Post `map_qubits()`, ion indices ARE correct (e.g., (1,4), (28,13), (73,76), etc.)
parallelPairs entries are correct with proper ion indices.

### Finding 4: _execute_ms_gates matching uses frozenset
In `_execute_ms_gates()`, matching uses `frozenset(ion.idx for ion in op.ions)` against `frozenset(p)` from `solved_pairs`. This should work IF:
- solved_pairs from routing use the same ion indices as the operations
- The SAT solver actually solves all pairs

## Key Code Locations

### File: qccd_WISE_ion_route.py
- `ionRoutingGadgetArch()` at line 2753: Phase-aware routing entry
- `ionRoutingWISEArch()` at line 1804: Main execution loop
- `_execute_ms_gates()` at line 2200: MS gate matching/execution
- `_drain_single_qubit_ops()` at line 1984: 1Q op execution between rounds
- `_build_parallel_pairs()` at line ~1684: Builds parallelPairs from toMoveOps
- `_route_round_sequence()` at line ~1350: SAT-based routing engine
- Execution loop at line ~2355: GlobalReconfig → drain → MS sequence

### File: gadget_routing.py
- `route_full_experiment_as_steps()` at line 2509: Phase iteration + RoutingStep generation
- `_route_ec_fresh()` at line 2692: Per-block EC routing with ion-return BT
- `_preprocess_gadget_pairs()` at line 2462: Fix 7 split + Bug 3/4 decompose
- `_build_plans_from_compiler_pairs()` at line ~1829: Builds PhaseRoutingPlans from parallelPairs

### File: trapped_ion_compiler.py
- `route()` at line ~849: Calls ionRoutingGadgetArch
- `decompose_to_native()`: Creates MS ops with placeholder ions (idx=0)
- `map_qubits()`: Updates ion indices on all operations

## Proposed Fixes

### Fix A: SAT Solver Timeout for Gadget Phases (BLOCKING)
The compilation hangs because the SAT solver for gadget phases (CSS Surgery merge/split)
routes on a bounding-box sub-grid that's too large.
- **Action**: Add timeout to SAT solver calls in `_route_round_sequence()` for gadget phases
- **Action**: Use smaller subgridsize or reduce lookahead for gadget phases
- **Action**: Add per-phase SAT timeout configuration

### Fix B: Ensure MS Gates per Round are ALL Scheduled
Current flow: GlobalReconfig → drain_1q → execute_ms (once)
Per the user's requirement, each round should complete ALL MS gates.
- **Action**: In execution loop (line ~2355), after _execute_ms_gates, check if all 
  solved_pairs were matched. If not, log warning but continue.
- **Action**: Verify that solved_pairs from routing actually contains ALL pairs for that round

### Fix C: Bridge Ancilla Routing Issue
CSS Surgery has bridge ancilla ions that are shared across multiple CX pairs in same instruction.
Fix 10 (shared-ion split) handles this in ionRoutingGadgetArch, but the gadget routing path
in `_preprocess_gadget_pairs()` also does Fix 7 split. These may conflict or double-split.
- **Action**: Verify that splitting is consistent between both paths

### Fix D: toMove_phase_tags Warning
`toMove_phase_tags length 40 != toMoveOps total 192` warning in trapped_ion_compiler.py
indicates tags are per-parallelPairs-entry, not per-MS-op.
- **Action**: Fix the comparison in trapped_ion_compiler.py route() method
- This is a diagnostic warning only and doesn't affect correctness if phase_pair_counts is correct

### Fix E: Ensure routing steps have correct ms_round_index
The execution loop groups routing_steps by ms_round_index. If ms_round_index is wrong,
steps get grouped incorrectly and MS gates get executed in wrong order or dropped.
- **Action**: Verify ms_round_index numbering in route_full_experiment_as_steps() 
  matches the parallelPairs indices expected by the execution loop

### Fix F: Gadget Phase Routing Strategy
For CSS Surgery gadget phases, the current strategy extracts a bounding-box of interacting
blocks and routes on that sub-grid. This may be too large for the SAT solver.
- **Action**: For gadget phases with known MS pairs, skip SAT solver entirely and
  just construct direct placement solutions (place paired ions in same trap, others in nearby traps)
- **Action**: Alternatively, use a heuristic fallback with timeout

## Execution Order
1. Fix A (SAT timeout) — unblock compilation
2. Fix B (ensure all MS gates scheduled per round)
3. Fix E (ms_round_index correctness)
4. Fix D (diagnostic warning cleanup)
5. Fix C (bridge ancilla consistency)
6. Fix F (gadget phase optimization)
