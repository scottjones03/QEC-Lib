# SAT Pool Fix Context

## File Being Edited
`src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_SAT_WISE_odd_even_sorter.py`
(~4965 lines)

## Issues Found

### Issue 1: stop_event Poisoning Between Patches (CRITICAL BUG)
**Location**: Line ~3698 in `optimal_QMR_for_WISE()`
```python
stop_event = parent_stop_event if parent_stop_event is not None else _threading.Event()
```
When `_avoid_manager=True` (macOS), `stop_event` IS the same object as `parent_stop_event`.
If the global_budget_s timeout triggers `stop_event.set()` (lines 3828-3833), it sets
`parent_stop_event` — which is shared across ALL patch calls from `_patch_and_route()`.
Subsequent patches see `stop_event` already set and abort immediately.

**Fix**: Always create a NEW Event for the pool. Only CHECK parent_stop_event in the poll loop; never alias them.

### Issue 2: ThreadPoolExecutor on macOS = 1 Python Process
**Location**: Lines 3718-3721
```python
if _avoid_manager:
    executor = ThreadPoolExecutor(max_workers=max_workers)
```
On macOS, `_avoid_manager=True` forces ThreadPoolExecutor. This means:
- Only 1 Python process visible (expected but confusing to user)
- GIL contention: `_wise_build_structural_cnf` is pure Python, serializes on GIL
- Only the Minisat22 C solver releases GIL (in-process path)

**Fix**: Split `_avoid_manager` into:
- `_skip_manager`: True on macOS (Manager() deadlocks)
- `_use_threads`: True ONLY in notebook env (can't reliably spawn processes from notebook kernel)
On macOS terminal: use ProcessPoolExecutor (without Manager)
On macOS notebook: keep ThreadPoolExecutor

### Issue 3: UNSAT Configs Complete Instantly (Expected but Confusing)
In `_wise_sat_config_worker()` (line ~2006), the feasibility pre-check tests the loosest
bound (max_bound_B). For configs with small P_max, this is often UNSAT, and the config
returns immediately. Only configs with large enough P_max actually do bisection.
The progress bar shows configs completing very fast, then slow bisection for 1-2 configs.

**Fix**: Better progress messaging showing SAT vs UNSAT counts.

## Current Decision Tree (lines 3657-3716)
```
_force_serial = (max_inner_workers is not None and max_inner_workers <= 1)
_notebook_env = _in_notebook_env()
_avoid_manager = _notebook_env or sys.platform == "darwin"

if _force_serial:
    manager = None; sat/rc2_children = {}; stop_event = None
else:
    if _avoid_manager:
        manager = None
    else:
        manager = pool_context.Manager()
    
    if manager is not None:
        sat/rc2_children = manager.dict(); stop_event = manager.Event()
    elif _avoid_manager:
        sat/rc2_children = {}
        stop_event = parent_stop_event if parent_stop_event is not None else _threading.Event()  # BUG: aliases parent
    else:
        sat/rc2_children = {}; stop_event = None

# Executor decision (line ~3714)
if _force_serial:
    use_multiprocessing = False
else:
    use_multiprocessing = True if _avoid_manager else manager is not None

# Executor creation (line ~3718)
if _avoid_manager:
    executor = ThreadPoolExecutor(max_workers=max_workers)
else:
    executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=pool_context)
```

## New Decision Tree (proposed)
```
_force_serial = (max_inner_workers is not None and max_inner_workers <= 1)
_notebook_env = _in_notebook_env()
_skip_manager = _notebook_env or sys.platform == "darwin"
_use_threads = _notebook_env  # Only notebook needs ThreadPoolExecutor

if _force_serial:
    manager = None; sat/rc2_children = {}; stop_event = None
else:
    if _skip_manager:
        manager = None
    else:
        manager = pool_context.Manager()
    
    if manager is not None:
        sat/rc2_children = manager.dict(); stop_event = manager.Event()
    else:
        sat/rc2_children = {}
        stop_event = _threading.Event()  # Always create NEW event (never alias parent)

# Executor decision
if _force_serial:
    use_multiprocessing = False
else:
    use_multiprocessing = True if _skip_manager else manager is not None

# Executor creation
if _use_threads:
    executor = ThreadPoolExecutor(max_workers=max_workers)
elif _skip_manager:
    executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=pool_context)
else:
    executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=pool_context)
```

## Key Constants
- `_TIMEOUT_BASE_S = 300.0` (line 313)
- `_MIN_POOL_BUDGET_S_FLOOR = 600.0` (line 317)
- `_POOL_BUDGET_MULT = 10.0` (line 323)
- `_SAT_INPROCESS_CLAUSE_LIMIT = 50000` (line 335, or env WISE_INPROCESS_LIMIT)
- HARD_CAP = 8 (line 231 in `_wise_safe_sat_pool_workers`)

## Key Functions
- `optimal_QMR_for_WISE()`: line 3356 — main orchestrator
- `_wise_sat_config_worker()`: line 1785 — per-config binary search
- `_wise_safe_sat_pool_workers()`: line 201 — HARD_CAP=8
- `_wise_enumerate_pmax_configs()`: line 3166 — interleaved P_max values × capacity factors
- `run_sat_with_timeout_file()`: line ~2197 — SAT solver dispatch
- `_in_notebook_env()`: determines if running in Jupyter

## Call Chain
_patch_and_route (qccd_WISE_ion_route.py:360)
  → optimal_QMR_for_WISE (qccd_SAT_WISE_odd_even_sorter.py:3356)
    → ThreadPoolExecutor/ProcessPoolExecutor submits _wise_sat_config_worker
      → _solve_with_bound (nested function)
        → _wise_build_structural_cnf (pure Python, GIL-bound)
        → run_sat_with_timeout_file → Minisat22 in-process (GIL-releasing C code)

## Progress Reporting
- `STAGE_SAT_POOL_START`: emitted when pool starts (line 3745)
- `STAGE_SAT_CONFIG_DONE`: emitted when configs complete (lines 3791, 3844, 3937)
- Final progress at line 3939 shows `len(results)` vs `len(configs)`
- tqdm bars managed in routing_config.py and progress_table.py

## Where _patch_and_route passes stop_event to optimal_QMR_for_WISE
qccd_WISE_ion_route.py line 713:
```python
parent_stop_event=stop_event,
```
This is the SAME stop_event for ALL patch calls in the loop.
