# Implementation Notes for Grown Stabilizer Resolution + Fallback Cleanup

## File Paths
- `src/qectostim/gadgets/css_surgery_cnot.py` (1349 lines)
- `src/qectostim/codes/abstract_css.py` (2322 lines)
- `src/qectostim/codes/surface/rotated_surface.py` (~800 lines)
- `src/qectostim/gadgets/layout.py` (1149 lines)
- `src/qectostim/experiments/stabilizer_rounds/css.py` (1823 lines)

## Key Data Structures

### BlockInfo (layout.py ~L330)
```python
@dataclass
class BlockInfo:
    name: str
    code: Any
    offset: CoordND
    data_qubit_range: range
    x_ancilla_range: range  # Global indices for X stab ancillas
    z_ancilla_range: range  # Global indices for Z stab ancillas
    local_dim: int
```

### GrownStabilizer (abstract_css.py ~L1468)
```python
@dataclass
class GrownStabilizer:
    lattice_position: Tuple[float, ...]
    stab_type: str  # "X" or "Z"
    existing_ancilla_global: int  # -1 = placeholder, caller fills
    original_weight: int
    new_weight: int
    new_cx_per_phase: List[List[Tuple[int, int]]]  # pairs have -1 for ancilla
    belongs_to_block: str  # "" = placeholder, caller fills
    new_support_globals: List[int] = field(default_factory=list)
```

## Code's Stab Coord Access
- `code.get_x_stabilizer_coords()` → List[Tuple[float, ...]] - X stab lattice coords
- `code.get_z_stabilizer_coords()` → List[Tuple[float, ...]] - Z stab lattice coords
- For RotatedSurfaceCode, stored in metadata as `x_stab_coords`, `z_stab_coords`
- Both return sorted lists; local index in list = ancilla index relative to base

## Ancilla Global Index Computation
- X stab local_idx → global = `blk.x_ancilla_range.start + local_idx`
- Z stab local_idx → global = `blk.z_ancilla_range.start + local_idx`

## Grown Stab Resolution Logic (in compute_layout())
For ZZ merge: grown stabs are X-type boundary stabs from block_0
- Match gs.lattice_position against ctrl_code.get_x_stabilizer_coords()
- global = blk0.x_ancilla_range.start + matched_local_idx
- belongs_to_block = "block_0"
- Fill -1 placeholders in new_cx_per_phase with the resolved global idx

For XX merge: grown stabs are Z-type boundary stabs from block_1
- Match gs.lattice_position against anc_code.get_z_stabilizer_coords()
- global = blk1.z_ancilla_range.start + matched_local_idx
- belongs_to_block = "block_1"

CX pair direction:
- Z-type stab: (data_global, anc_global) - CX data→anc
- X-type stab: (anc_global, data_global) - CX anc→data

## Changes To Make

### 1. Add _resolve_grown_stabs() helper to CSSSurgeryCNOTGadget
Location: after get_phase_pairs(), around line 240
- Takes (merge_info, code, block_info, block_name)
- For each gs in merge_info.grown_stabs:
  - Gets stab coords from code for gs.stab_type
  - Matches gs.lattice_position (with tolerance ~0.5)
  - Sets gs.existing_ancilla_global
  - Sets gs.belongs_to_block
  - Replaces -1 in new_cx_per_phase with actual ancilla global

### 2. Call _resolve_grown_stabs() in compute_layout() (around L1283, L1301)
After add_seam_stabilizers() for each merge:
- ZZ: self._resolve_grown_stabs(zz_info, ctrl_code, blk0, "block_0")
- XX: self._resolve_grown_stabs(xx_info, anc_code, blk1, "block_1")

### 3. Wire grown CX in _emit_zz_merge() and _emit_xx_merge()
Replace `grown_cx_per_phase = [[] for _ in range(n_phases)]` with:
```python
grown_cx_per_phase = [[] for _ in range(n_phases)]
for gs in merge_info.grown_stabs:
    if gs.existing_ancilla_global >= 0:
        for ph in range(n_phases):
            if ph < len(gs.new_cx_per_phase):
                grown_cx_per_phase[ph].extend(gs.new_cx_per_phase[ph])
```

### 4. Remove fallback paths
- compute_layout L1286-1288: remove `add_boundary_bridges` fallback for empty seam
- compute_layout L1289-1290: remove `add_boundary_bridges` fallback for no seam API
- compute_layout L1304-1306: same for XX merge
- compute_layout L1307-1308: same for XX merge no API
- Remove the `has_seam_api` check entirely — base class always has it
- _emit_zz_merge L584-596: remove recompute fallback
- _emit_xx_merge L738-749: remove recompute fallback  
- _emit_joint_merge_round L407-440: remove sequential fallback → raise error
- Keep the `else` non-topological path at L1310 but raise NotImplementedError

### 5. compute_layout structure after cleanup
```
if has_boundaries:
    # topological path with seam API
    ...
    zz_info = ctrl_code.get_merge_stabilizers(...)
    if not zz_info.seam_stabs:
        raise ValueError(...)  
    layout.add_seam_stabilizers(...)
    self._resolve_grown_stabs(...)
    self._zz_merge_info = zz_info
    # same for XX
else:
    raise NotImplementedError("Lattice surgery requires topological code with boundaries")
```

## Current Line References (css_surgery_cnot.py)
- __init__: L91-128
- get_phase_pairs: L190-237
- _emit_ec_round: L259
- _emit_joint_merge_round: L367-530
- _emit_zz_merge: L548-660
- _emit_xx_merge: L704-790
- _emit_xx_split: L800+
- compute_layout: L1207-1320
- has_boundaries check: L1220
