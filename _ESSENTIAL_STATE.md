# ESSENTIAL STATE — Read this before continuing

## COMPLETED STEPS
1. ✅ abstract_css.py modified:
   - Added `from dataclasses import dataclass, field` import
   - Added SeamStabilizer, GrownStabilizer, MergeStabilizerInfo dataclasses
   - Added get_merge_stabilizers() default method on CSSCode class
   - Placed between get_merge_compatibility() and CSSCodeWithComplex

## NEXT STEPS (in order)
2. rotated_surface.py: Add get_merge_stabilizers() override at END of class (after hz property at L549)
   - File: src/qectostim/codes/surface/rotated_surface.py (551 lines)
   - Import: from qectostim.codes.abstract_css import MergeStabilizerInfo, SeamStabilizer, GrownStabilizer
     (add at top, around L33 area near existing imports from abstract_css)
   - Existing imports at L33: from qectostim.codes.abstract_css import TopologicalCSSCode

3. layout.py: Update add_boundary_bridges() to use seam positions
   - File: src/qectostim/gadgets/layout.py (~1090 lines)
   - add_boundary_bridges() at ~L791

4. css_surgery_cnot.py: Rewrite merge phases
   - File: src/qectostim/gadgets/css_surgery_cnot.py (~1436 lines)

5. Syntax verification

## ROTATED SURFACE CODE FACTS
- Data: odd-odd coords (2i+1, 2j+1) for i,j=0..d-1
- Even-even pos type: ((x//2)%2) != ((y//2)%2) → True=X, False=Z
- CX schedules:
  z_schedule = [(+1,+1),(+1,-1),(-1,+1),(-1,-1)] # data→anc
  x_schedule = [(+1,+1),(-1,+1),(+1,-1),(-1,-1)] # anc→data
- Boundary: rough = top/bottom (y=0,2d), Z-stabs truncated
            smooth = left/right (x=0,2d), X-stabs truncated

## SEAM GEOMETRY
### ZZ Merge (rough↔rough, bottom of block_A ↔ top of block_B)
Seam Z-stabs at (x, y_seam): x ∈ {0,4,8,...}, count=floor(d/2)+1
  Each has up to 4 diagonal neighbors: data at (x±1, y_seam±1)
  y_seam+1 → data from block_A (bottom row), y_seam-1 → data from block_B (top row)
  CX direction: data→anc (Z-type)

Grown X-stabs at (x, y_seam): x ∈ {2,6,10,...}, count=floor(d/2)
  Already have weight-2 from block_A; gain up to 2 from block_B
  CX direction: anc→data (X-type)

### XX Merge (smooth↔smooth, right of block_A ↔ left of block_B)
Seam X-stabs at (x_seam, y): positions where type=X → ((x_seam//2)%2) != ((y//2)%2)
Grown Z-stabs at (x_seam, y): positions where type=Z → ((x_seam//2)%2) == ((y//2)%2)

## KEY DATACLASSES (already in abstract_css.py)
SeamStabilizer: lattice_position, stab_type, global_ancilla_idx, weight, cx_per_phase, support_globals
GrownStabilizer: lattice_position, stab_type, existing_ancilla_global, original_weight, new_weight, new_cx_per_phase, belongs_to_block, new_support_globals
MergeStabilizerInfo: seam_stabs, grown_stabs, seam_type, grown_type, num_cx_phases=4

## rotated_surface.py END OF FILE (L540-551):
```python
        return list(self.metadata["data_coords"])

    @property
    def hx(self) -> np.ndarray:
        """X stabilisers: shape (#X-checks, #data)."""
        return self._hx

    @property
    def hz(self) -> np.ndarray:
        """Z stabilisers: shape (#Z-checks, #data)."""
        return self._hz
```
Insert new method BEFORE the final empty line after hz property.

## rotated_surface.py IMPORTS (L33):
```python
from qectostim.codes.abstract_css import TopologicalCSSCode
```
Need to add: MergeStabilizerInfo, SeamStabilizer, GrownStabilizer to this import.

## get_merge_stabilizers() SIGNATURE (already defined in abstract_css.py):
```python
def get_merge_stabilizers(
    self,
    merge_type: str,        # "ZZ" or "XX"
    my_edge: str,           # "bottom", "top", "left", "right"
    other_code: "CSSCode",
    other_edge: str,
    my_data_global: Dict[int, int],    # {local_data_idx: global_qubit_idx}
    other_data_global: Dict[int, int], # {local_data_idx: global_qubit_idx}
    seam_qubit_offset: int,            # starting global idx for seam ancillas
) -> MergeStabilizerInfo:
```
