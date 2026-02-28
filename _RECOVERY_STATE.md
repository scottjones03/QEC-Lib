# RECOVERY STATE — CSS Bridge Fix Implementation

## STATUS (February 27, 2026)

### COMPLETED CHANGES (verified in files):
1. **P1 — layout.py L875-900**: Bridge coord midpoint DONE. `coord = tuple((ca[j] + cb[j]) / 2.0 for j in range(self.dim))`
2. **P0 — css_surgery_cnot.py**: `_emit_joint_merge_round()` method ADDED (~L455-570). Joint bridge+EC in same TICK window.
3. The file css_surgery_cnot.py is now 1436 lines (was 1339).

### WHAT STILL NEEDS VERIFICATION:
- Need to check if `_emit_zz_merge()` and `_emit_xx_merge()` ACTUALLY CALL `_emit_joint_merge_round()` 
- Need to check P2 fix in gadget_routing.py (round() vs int())
- Need to syntax check all files
- Need to restart kernel and run cell 24

### KEY FILES:
- `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/gadgets/css_surgery_cnot.py` (1436 lines)
- `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/gadgets/layout.py` (1090 lines)
- `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py`

### USER COMPLAINTS TO CHECK:
1. Bridge coords = midpoint → DONE (P1)
2. Joint stabilizer measurement → DONE (P0) — but verify `_emit_zz_merge` and `_emit_xx_merge` use it
3. Bridge ancillas as separate qubits → Code already has bridge ancillas via `add_boundary_bridges()` in layout.py
4. d rounds of joint measurement → Need to verify merge loop counts
5. Split phase measures bridges → Need to verify `_emit_zz_split` / `_emit_xx_split`
6. Bridge positions in animation → P1 fix should handle this
7. CX links for bridges → Need to check animation code
8. EC round structure → P0 joint rounds should fix this

### CRITICAL: Read these line ranges to continue:
- css_surgery_cnot.py L500-750 — `_emit_zz_merge`, `_emit_zz_split`, `_emit_xx_merge`
- css_surgery_cnot.py L750-950 — `_emit_xx_split`, `_emit_ancilla_measurement` 
- gadget_routing.py — search for `round(` to verify P2

### NOTEBOOK:
- File: `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/notebooks/trapped_ion_demo.ipynb`
- Cell 24 is the CSS Surgery cell (large cell with CSSSurgeryCNOTGadget)
- Must restart kernel then run cell 24

### VENV:
- `/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/my_venv/bin/python`
- PYTHONPATH=src
