# Implementation State — All Files & Line References

## WORKSPACE
`/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim`

## STATUS
- P0: IN PROGRESS (not yet started editing)
- P1-P4: NOT STARTED

## P0: Joint Stabilizer Measurement
### File: `src/qectostim/gadgets/css_surgery_cnot.py` (1339 lines)

### WHAT TO DO:
Replace separate bridge+EC rounds in `_emit_zz_merge()` and `_emit_xx_merge()` with joint rounds where bridge and EC ancillas are reset/measured TOGETHER.

### NEW METHOD: `_emit_joint_merge_round()`
Add BEFORE `_emit_zz_merge` (around line 470). This method:
1. Resets EC ancillas (all builders) + bridge ancillas → TICK
2. EC CX phases 0-3 (interleaved across all builders) → TICK each
3. Bridge CX phase A → TICK
4. Bridge CX phase B → TICK
5. Measures EC ancillas (all builders) + bridge ancillas → TICK
6. Bridge temporal detectors (rnd >= 1)

### `_emit_zz_merge()` CHANGES (L476-625):
Current loop body (L540-617 per round):
```python
for rnd in range(d):
    # Bridge ZZ separate:
    circuit.append("R", bridge_zz)                    # L541
    circuit.append("TICK")                            # L542
    cx_targets = [...]                                 # L544-548
    circuit.append("CNOT", cx_targets)                 # L549
    circuit.append("TICK")                            # L550
    cx_targets = [...]                                 # L552-556
    circuit.append("CNOT", cx_targets)                 # L557
    circuit.append("TICK")                            # L558
    meas_start = ctx.add_measurement(n_z)              # L560
    circuit.append("M", bridge_zz)                     # L561
    self._zz_bridge_meas.append(...)                   # L562
    circuit.append("TICK")                            # L563
    # temporal detectors...                            # L565-571
    # per_builder_kw setup...                          # L573-612
    self._emit_parallel_ec_round(...)                   # L614-616
```
REPLACE the entire loop body with call to `_emit_joint_merge_round()`.

### `_emit_xx_merge()` CHANGES (L690-810):
Same pattern. Current loop body:
```python
for rnd in range(d):
    circuit.append("RX", bridge_xx)                   # L747
    circuit.append("TICK")                            # L748
    cx_targets = [bridge→anc...]                       # L750-754
    circuit.append("CNOT", cx_targets)                 # L755
    circuit.append("TICK")                            # L756
    cx_targets = [bridge→tgt...]                       # L758-762
    circuit.append("CNOT", cx_targets)                 # L763  
    circuit.append("TICK")                            # L764
    meas_start = ctx.add_measurement(n_x)              # L766
    circuit.append("MX", bridge_xx)                    # L767
    self._xx_bridge_meas.append(...)                   # L768
    circuit.append("TICK")                            # L769
    # temporal detectors...                            # L771-782
    # per_builder_kw = {all: emit_detectors=True}      # L790-792
    self._emit_parallel_ec_round(...)                   # L794-796
```
REPLACE with call to `_emit_joint_merge_round()`.

### `_emit_parallel_ec_round()` (L356-430):
Keep as-is (used for non-merge EC rounds). The new `_emit_joint_merge_round()` 
replicates the same EC interleaving logic but adds bridge ops.

### Bridge CX pair format:
- ZZ merge: CX(ctrl_data[ctrl_boundary_sorted[i]], bridge_zz[i]) — ctrl is CONTROL
  Then: CX(anc_data[anc_boundary_sorted[i]], bridge_zz[i]) — anc is CONTROL
- XX merge: CX(bridge_xx[i], anc_data[anc_boundary_sorted[i]]) — bridge is CONTROL
  Then: CX(bridge_xx[i], tgt_data[tgt_boundary_sorted[i]]) — bridge is CONTROL

## P1: Bridge Coordinate Fix
### File: `src/qectostim/gadgets/layout.py` line 885
Current:
```python
coord = tuple(cb[j] for j in range(self.dim))
```
Fixed:
```python
coord = tuple((ca[j] + cb[j]) / 2.0 for j in range(self.dim))
```

## P2: Bridge Ions in Routing Grid
### File: `src/qectostim/experiments/hardware_simulation/trapped_ion/utils/gadget_routing.py`
### `_create_ions_from_allocation()` ~L1676-1760
Bridge ions already handled here (assigned to blocks, positioned at boundaries).
The main issue is that bridge ions with midpoint coords might not fall within
any block's bounding region. After P1, the coords will be midpoints.
Fix: After P1, the bridge ion position should use the midpoint coord directly.
The code at L1710-1725 already handles this — just need to make sure the 
coordinate clamping doesn't break midpoint positioning.

## P3: Bridge Physical Grid Position
### File: `src/qectostim/experiments/hardware_simulation/trapped_ion/compiler/qccd_WISE_ion_route.py`
### Lines ~3046-3080: shared-ion round splitting already handles bridges.
Bridge ions need valid physical grid positions. The `_create_ions_from_allocation()`
already assigns bridge positions. The main fix is ensuring routing grid encompasses
the bridge positions (which are between blocks after P1).

## P4: Animation Bridge CX Links
### File: `src/qectostim/experiments/hardware_simulation/trapped_ion/viz/animation.py`
### Line 594: TwoQubitMSGate → "MS" label mapping exists.
### Line 1015: MS gate kind detection.
The animation already renders MS gates with links. Bridge CX gates that get compiled
to MS gates should automatically get the link visualization. This may already work
if the compilation pipeline correctly generates TwoQubitMSGate operations for bridges.
Check if P4 is actually needed after P0-P3 are done.

## NOTEBOOK
Cell 24 = the CSS Surgery cell (id=#VSC-76f13df6), the big combined cell.

## KEY VARIABLES USED IN css_surgery_cnot.py:
- `self._code` — CSSCode instance (set in compute_layout())
- `self._builders` — List[CSSStabilizerRoundBuilder] (set in set_builders())
- `self._zz_bridge_meas` — List[List[int]] tracking ZZ bridge measurement indices
- `self._xx_bridge_meas` — List[List[int]] tracking XX bridge measurement indices
- `ctx` — DetectorContext with `add_measurement()` and `measurement_index`
- `stim.target_rec()` — creates measurement record targets

## IMPORTANT: `_emit_parallel_ec_round` check
Line 393: `all_flat_interleave = all(not hasattr(b, 'emit_inner_only_round') and ...)`
This check determines if we use interleaved CX or sequential. My joint method should
do the same check — but for simplicity, can assume all_flat_interleave=True for now
since CSS Surgery always uses flat builders.
