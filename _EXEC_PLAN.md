# IMPLEMENTATION PLAN - EXECUTE NOW
# All line numbers verified from actual file reads

## WORKSPACE PATH
/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim

## FILE 1: src/qectostim/gadgets/css_surgery_cnot.py (1339 lines)

### P0 CHANGE 1: Add _emit_joint_merge_round() method
INSERT AFTER line 430 (after _emit_parallel_ec_round's final `circuit.append("TICK")`)
BEFORE the emit_next_phase section (line 435: `# ------------------------------------------------------------------`)

New method signature:
```python
def _emit_joint_merge_round(
    self, builders, circuit, per_builder_kw,
    bridge_ancillas, bridge_cx_A, bridge_cx_B,
    bridge_reset_op, bridge_measure_op,
    ctx, rnd, prev_bridge_meas,
):
```
Emits: Reset(EC+bridge)→TICK, CX phases 0-3→TICK each, bridge CX A→TICK, bridge CX B→TICK, Measure(EC+bridge)→TICK

### P0 CHANGE 2: Rewrite _emit_zz_merge loop body (lines 541-616)
Current loop (line 540): `for rnd in range(d):`
REPLACE entire body with:
1. Build bridge_cx_A list: [(ctrl_data[ctrl_boundary_sorted[i]], bridge_zz[i]) for i]
2. Build bridge_cx_B list: [(anc_data[anc_boundary_sorted[i]], bridge_zz[i]) for i]
3. Build per_builder_kw (same logic as current)
4. Call _emit_joint_merge_round(...)
5. Append returned meas_indices to self._zz_bridge_meas

### P0 CHANGE 3: Rewrite _emit_xx_merge loop body (lines 747-796)
Current loop: `for rnd in range(d):`
REPLACE entire body with:
1. Build bridge_cx_A: [(bridge_xx[i], anc_data[anc_boundary_sorted[i]]) for i]
2. Build bridge_cx_B: [(bridge_xx[i], tgt_data[tgt_boundary_sorted[i]]) for i]
3. per_builder_kw = {all: emit_detectors=True}
4. Call _emit_joint_merge_round(...)  with "RX"/"MX"
5. Append returned meas_indices to self._xx_bridge_meas

## FILE 2: src/qectostim/gadgets/layout.py line 885
### P1: Change bridge coordinate from block_b to midpoint
Current: `coord = tuple(cb[j] for j in range(self.dim))`
New:     `coord = tuple((ca[j] + cb[j]) / 2.0 for j in range(self.dim))`

## FILE 3: src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py
### P2: Bridge ions position update (lines 1676-1760)
After P1, bridge coords become midpoints. The _create_ions_from_allocation() already
handles bridge positions. With midpoint coords, bridge ions will be between blocks.
Main fix: ensure the bridge ion x,y uses the midpoint coord correctly (line ~1712):
```python
x = int(float(coord[0])) if len(coord) > 0 else 0
y = int(float(coord[1])) if len(coord) > 1 else 0
```
Need to use round() instead of int() for half-integer coords, or keep float.
Also: the routing grid must encompass bridge positions between blocks.

## FILE 4: qccd_WISE_ion_route.py - P3: already handles shared-ion splitting (line 3046+)
Bridge ions in routing already work if positions are correct.

## FILE 5: viz/animation.py - P4: Bridge CX already compiled to MS gates
MS gates already get link visualization. P4 may be a no-op if P0-P3 work.

## KEY CONTEXT
- ZZ merge: bridge reset "R", measure "M", CX direction ctrl/anc→bridge (CONTROL on data)
- XX merge: bridge reset "RX", measure "MX", CX direction bridge→anc/tgt (CONTROL on bridge)
- Bridge temporal detectors: rnd >= 1, compare curr vs prev bridge meas
- For ZZ rnd 0: block_1 gets emit_x_anchors=True, emit_z_anchors=False
- For XX: all blocks get emit_detectors=True
- ctx.add_measurement(n) returns meas_start index
- ctx.measurement_index is the current total measurement count
- stim.target_rec(offset) creates a measurement record target (offset is negative)
- _emit_parallel_ec_round checks all_flat_interleave before using phase-decomposed path
- emit_ancilla_reset(circuit, emit_z_anchors=..., emit_x_anchors=...)
- emit_cx_for_phase(circuit, phase_idx)
- emit_ancilla_measure_and_detectors(circuit, emit_detectors=...)
- n_interleave_phases() returns 4 for surface codes
- coord for ZZ detector: (b_idx + 0.5, -0.5, rnd)
- coord for XX detector: (b_idx + 0.5, -0.5, rnd + 100)

## NOTEBOOK CELL 24
Cell 24 = id=#VSC-76f13df6, the big CSS Surgery cell.
