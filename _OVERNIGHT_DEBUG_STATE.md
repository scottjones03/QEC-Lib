# Overnight Debug State - CSS Surgery Gadget

## Task
Iterative debug loop: restart kernel → run cell 24 → run cell 25 → analyze → fix → repeat

## Key Files
- **Notebook:** `notebooks/trapped_ion_demo.ipynb`
- **Cell 24 ID:** `#VSC-76f13df6` (lines 715-908) — CSS Surgery compilation (self-contained, all imports)
- **Cell 25 ID:** `#VSC-748076a4` (lines 911-1346) — Diagnostic verification cell
- **Venv:** `my_venv/` — Python 3.11.5, ipykernel 6.29.5

## Source Fix Already Applied
File: `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`
- `_rebuild_schedule_for_layout` at ~line 1085 catches `NoFeasibleLayoutError`
- 3 fallback levels: hard BT → soft BT → reduced pins → skip patch
- `NoFeasibleLayoutError` imported via wildcard from `qccd_operations.py` → `qccd_SAT_WISE_odd_even_sorter.py`

## Previous Cell 24 Error (from last successful run)
```
NoFeasibleLayoutError: No feasible layout for any Σ_r P_r bound over 16 configs.
```
Traceback: `compile_gadget_for_animation` → `ionRoutingGadgetArch` → `route_full_experiment_as_steps`  
→ `_apply_post_gadget_transition` → `_compute_transition_reconfig_steps`  
→ `_rebuild_schedule_for_layout` → `_optimal_QMR_for_WISE` → SAT solver

## Current Blocker: Kernel Death
- `configure_python_notebook` keeps dying
- All Python imports work perfectly from terminal
- ipykernel 6.29.5 installed, all deps (traitlets 5.14.3, jupyter_client 8.8.0, pyzmq 27.1.0, tornado 6.5.4) OK
- iCloud sync corruption cleaned (removed `* 2.py` files, `~ygments`)
- Possible causes: iCloud path with spaces, VS Code kernel launcher issue, or leftover port allocation

## Key Source Files
- `src/qectostim/.../compiler/qccd_WISE_ion_route.py` — Main WISE routing, 3512 lines
- `src/qectostim/.../utils/qccd_operations.py` — Operations, `_optimal_QMR_for_WISE`
- `src/qectostim/.../compiler/qccd_SAT_WISE_odd_even_sorter.py` — SAT solver, `NoFeasibleLayoutError`
- `src/qectostim/.../utils/gadget_routing.py` — Gadget routing, `_apply_post_gadget_transition`
- `src/qectostim/.../demo/run.py` — `compile_gadget_for_animation`

## Cell 24 Does
1. Imports everything (self-contained)
2. Builds `CSSSurgeryCNOTGadget` experiment with d=2, k=2
3. Generates ideal stim circuit, extracts QECMetadata
4. Partitions grid, decomposes phases
5. Compiles via `compile_gadget_for_animation`
6. Prints metrics

## Cell 25 Does (Diagnostic)
1. Ideal circuit analysis (CX gate count by round)
2. Native decomposed circuit stats
3. CX → MS mapping verification
4. Scheduled batches analysis
5. Round contiguity check
6. WISE constraint verification
7. Stim instruction → batch mapping
8. Summary statistics with pass/fail checks
9. Detailed native ops dump

## Next Steps
1. Fix kernel death (try: `configure_notebook` instead, or manually specify kernel, or check VS Code Jupyter logs)
2. Run cell 24, verify NoFeasibleLayoutError fix works
3. Run cell 25, analyze diagnostic output
4. Fix any issues found, repeat
