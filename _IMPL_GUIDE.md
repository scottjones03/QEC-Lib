# IMPLEMENTATION GUIDE — SAT Routing Fix

## Key File Paths
- WISE route: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py` (3355 lines)
- Gadget routing: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py` (~3274 lines)

## EXACT CURRENT CODE to change

### 1. `_execute_ms_gates` (qccd_WISE_ion_route.py L2220-2324)

CURRENT BROKEN CODE (exact text for replacement):
```python
    def _execute_ms_gates(
        round_idx: int,
        solved_pairs: List[Tuple[int, int]],
    ) -> Tuple[int, int]:
```
The function:
- Uses frozenset exact match (ion_set == rp) 
- Scans ALL operationsLeft instead of toMoves[round_idx]
- No getTrapForIons() fallback (just warns and skips)
- Sorts by _tick_epoch

WHAT IT SHOULD DO (like old code at 243a188 L1858-1880):
```python
for op in toMoves[idx]:
    if any(all((ion.idx in p) for ion in op.ions) for p in solved_pairs):
        trap = op.getTrapForIons()
        if trap is not None:
            op.setTrap(trap); op.run(); allOps.append(op); operationsLeft.remove(op)
        else:
            trap = op.ions[0].parent  # FALLBACK!
            if trap: op.setTrap(trap); op.run(); allOps.append(op); operationsLeft.remove(op)
```

### 2. `_merge_phase_pairs` call (gadget_routing.py L2851)

EXACT CURRENT LINE:
```python
        phase_pairs = _merge_phase_pairs(phase_pairs)
```
MUST REMOVE/SKIP this call.

### 3. Execution loop (qccd_WISE_ion_route.py L2340-2460)

The loop groups by ms_round_index then:
- Applies reconfig
- Drains 1q ops
- Calls _execute_ms_gates(ms_round_idx, first_step.solved_pairs)

### 4. ionRoutingGadgetArch → ionRoutingWISEArch delegation (L3335+)

Passes `_precomputed_routing_steps=all_routing_steps` to ionRoutingWISEArch.
In ionRoutingWISEArch, when _precomputed is provided, it skips building routing_steps
and uses the precomputed ones directly. BUT it STILL builds toMoves and parallelPairs
from operations (L1920-1930).

## FIXES TO IMPLEMENT

### Fix 1: Restore toMoves-based execution in _execute_ms_gates

Change _execute_ms_gates signature to accept toMoves dict:
```python
def _execute_ms_gates(
    round_idx: int,
    solved_pairs: List[Tuple[int, int]],
    round_ops: Optional[List] = None,
) -> Tuple[int, int]:
```

When round_ops is provided (from toMoves[ms_round_idx]):
- Iterate those specific ops using old-style contains check
- Add getTrapForIons() fallback with op.ions[0].parent

When round_ops is None (shouldn't happen but safety):
- Fall back to current scan-all-operationsLeft behavior

At the call site (L2396), change:
```python
ms_gates_executed, _unmatched = _execute_ms_gates(ms_round_idx, first_step.solved_pairs)
```
to:
```python
round_ops = toMoves.get(ms_round_idx, []) if toMoves else []
ms_gates_executed, _unmatched = _execute_ms_gates(ms_round_idx, first_step.solved_pairs, round_ops)
```

### Fix 2: Remove _merge_phase_pairs from route_full_experiment_as_steps

At gadget_routing.py L2851, comment out or remove:
```python
        phase_pairs = _merge_phase_pairs(phase_pairs)
```

### Fix 3: Restore getTrapForIons() fallback

In _execute_ms_gates execution section (L2298-2324), after:
```python
        for op in to_execute:
            trap = op.getTrapForIons()
            if trap is not None:
                ...
            else:
                # Add fallback here
                trap = op.ions[0].parent if op.ions else None
                if trap is not None:
                    op.setTrap(trap)
                    op.run()
                    allOps.append(op)
                    operationsLeft.remove(op)
                    ms_executed += 1
                else:
                    logger.warning("...")
```

## IMPORTANT: BACKWARD COMPATIBILITY

The non-gadget path (ionRoutingWISEArch without _precomputed_routing_steps):
- Builds its own routing_steps via _route_round_sequence (L1933+)
- ALSO builds toMoves and parallelPairs (L1920-1930)
- Uses the SAME execution loop
- So Fix 1 (toMoves-based execution) works for BOTH paths

## IMPLEMENTATION ORDER
1. Fix 1: toMoves-based MS gate execution (most critical)
2. Fix 3: getTrapForIons fallback (prevents individual gate drops)
3. Fix 2: Remove _merge_phase_pairs (prevents round count mismatch)
4. Verify with test run

## KEY LINES TO EDIT

### qccd_WISE_ion_route.py
- L2220-2230: _execute_ms_gates signature — add round_ops param
- L2245-2280: Replace frozenset matching with toMoves-based iteration 
- L2298-2324: Add getTrapForIons fallback  
- L2396: Call site — pass toMoves[ms_round_idx]
- L2440-2442: Call site for subsequent steps — pass round_ops

### gadget_routing.py
- L2851: Remove `phase_pairs = _merge_phase_pairs(phase_pairs)`
