# FINAL PROGRESS TRACKER

## WORKSPACE
/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim

## COMPLETED FIXES
1. P0: src/qectostim/gadgets/css_surgery_cnot.py - Added _emit_joint_merge_round(), rewrote _emit_zz_merge and _emit_xx_merge loops to use joint rounds
2. P1: src/qectostim/gadgets/layout.py line ~885 - Changed bridge coord from cb to (ca+cb)/2 midpoint 
3. P2: src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py line ~1712 - Changed int() to round() for bridge x,y coords

## REMAINING
4. Lines 1635-1636, 1649-1650, 1663-1664 in gadget_routing.py also use int(float(coord + offset)) for data/ancilla qubit ions. These are NOT bridges so they DON'T need round(). Only line ~1712 needed fixing (bridge-specific code).

5. P3: Check if partition_grid_for_blocks handles bridge positions between blocks.
   The grid size computation may need to expand to encompass midpoint bridge coords.
   Need to check: src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py
   Functions: partition_grid_for_blocks(), _expand_grid_for_bridges(), compute_gadget_grid_size()

6. P4: Animation bridge CX links - likely NO-OP since MS gates already render as links.
   Compilation translates CNOT→MS+single-qubit, and MS gates already shown with arcs in animation.

7. Syntax check all 3 modified files:
   - src/qectostim/gadgets/css_surgery_cnot.py
   - src/qectostim/gadgets/layout.py
   - src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py
   Command: source my_venv/bin/activate && python -c "import ast; ast.parse(open('FILE').read()); print('OK')"

8. Restart kernel and run cell 24 (id=#VSC-76f13df6)

## CELL 24 CONTENT
The big CSS Surgery cell that does:
- Build CSS Surgery CNOT experiment (d=2)
- Partition grid, decompose phases
- Compile for animation via compile_gadget_for_animation()
- Render animation to MP4

## KEY NOTES
- The int(float()) on lines 1635-1664 are for DATA/ANCILLA ions (integer coords), not bridges. Leave them.
- Only bridge ions (line 1712) have half-integer midpoint coords after P1, so only that line needs round().
- P3 grid expansion: partition_grid_for_blocks in gadget_routing.py computes sub-grid regions per block.
  Bridge ions may sit between blocks after P1. The assigned_block logic (line 1688-1706) assigns bridge
  to first connected block. Position (round of midpoint) should be close enough to block boundary.
  If grid doesn't encompass it, we may need to expand, but worth checking first.
- P4 animation: TwoQubitMSGate operations already get arc/link visualization. Bridge CX compiled to MS
  gates during trapped-ion native decomposition should automatically get rendered correctly.
