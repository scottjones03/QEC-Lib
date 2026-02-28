# IMPLEMENTATION STATE — MUST READ BEFORE CONTINUING

## COMPLETED
1. ✅ abstract_css.py: Added `from dataclasses import dataclass, field` to imports (L33).
   Added SeamStabilizer, GrownStabilizer, MergeStabilizerInfo dataclasses + 
   get_merge_stabilizers() default method on CSSCode class.
   These sit between get_merge_compatibility() and CSSCodeWithComplex.

## TODO (in order)
2. rotated_surface.py: Add get_merge_stabilizers() override at end of class (before L551)
3. layout.py: Update add_boundary_bridges() to use seam positions 
4. css_surgery_cnot.py: Major rewrite of merge emission

## ROTATED SURFACE CODE KEY FACTS
- File: src/qectostim/codes/surface/rotated_surface.py (551 lines)
- Class RotatedSurfaceCode(TopologicalCSSCode) at L127
- Data: odd-odd (2i+1, 2j+1) for i,j=0..d-1 → d² qubits
- Even-even position type check: is_parity = ((x//2)%2) != ((y//2)%2)
  True → X-type, False → Z-type
- CX schedules in metadata:
  x_schedule: [(+1,+1), (-1,+1), (+1,-1), (-1,-1)]  # anc→data
  z_schedule: [(+1,+1), (+1,-1), (-1,+1), (-1,-1)]  # data→anc
- data_coords stored in self.metadata["data_coords"]
- coord_to_index mapping available in _build_rotated() 
- d = self._d (self.metadata["distance"])
- self.metadata["data_qubits"] = list(range(n_data))
- self.metadata["x_stab_coords"] = x_stab_lattice_coords
- self.metadata["z_stab_coords"] = z_stab_lattice_coords

## CSS_SURGERY_CNOT.PY KEY FACTS  
- File: src/qectostim/gadgets/css_surgery_cnot.py (~1436 lines)
- Class CSSSurgeryCNOTGadget
- __init__ tracks: self._bridge_zz_ancillas, self._bridge_xx_ancillas
- _emit_joint_merge_round(): Reset EC+bridge → EC CX (4 phases) → bridge CX A → bridge CX B → Measure
  NEED: Reset EC+seam → CX (4 phases with EC+seam+grown) → Measure EC+seam
- _emit_zz_merge(): iterates d rounds calling _emit_joint_merge_round()
  Uses self._bridge_zz_ancillas for bridges. NEED: use MergeStabilizerInfo instead
- _emit_xx_merge(): same but for XX merge with _bridge_xx_ancillas
- _emit_zz_split(): measures bridge_zz in Z basis (M)
- _emit_xx_split(): measures bridge_xx in X basis (MX)
- compute_layout(): builds GadgetLayout, calls add_boundary_bridges()
  NEED: call code.get_merge_stabilizers() and add_seam_stabilizers()
- get_phase_pairs(): returns CX pairs per phase for hardware routing

## LAYOUT.PY KEY FACTS
- File: src/qectostim/gadgets/layout.py (~1090 lines)
- add_boundary_bridges() at L791-890: creates d bridges per merge at midpoint coords
- add_bridge_ancilla(): adds a single bridge qubit to layout

## ABSTRACT_CSS.PY DATACLASS DEFINITIONS (already added)
```python
@dataclass
class SeamStabilizer:
    lattice_position: Tuple[float, ...]
    stab_type: str  # "X" or "Z"
    global_ancilla_idx: int
    weight: int
    cx_per_phase: List[List[Tuple[int, int]]]  # per-phase (ctrl, tgt) pairs
    support_globals: List[int] = field(default_factory=list)

@dataclass
class GrownStabilizer:
    lattice_position: Tuple[float, ...]
    stab_type: str
    existing_ancilla_global: int
    original_weight: int
    new_weight: int
    new_cx_per_phase: List[List[Tuple[int, int]]]  # additional CX pairs
    belongs_to_block: str
    new_support_globals: List[int] = field(default_factory=list)

@dataclass
class MergeStabilizerInfo:
    seam_stabs: List[SeamStabilizer]
    grown_stabs: List[GrownStabilizer]
    seam_type: str  # "Z" for ZZ merge, "X" for XX merge
    grown_type: str  # opposite
    num_cx_phases: int = 4
```

## SEAM GEOMETRY (ROTATED SURFACE CODE)

### ZZ Merge (rough↔rough)
Seam Z-stabs: at (x, y_seam) where x even, (x//2)%2==0 → positions {0,4,8,...}
  count = floor(d/2) + 1
  Z-schedule offsets: [(+1,+1),(+1,-1),(-1,+1),(-1,-1)]
  CX direction: data→anc

Grown X-stabs: at (x, y_seam) where x even, (x//2)%2==1 → positions {2,6,10,...}
  count = floor(d/2)
  X-schedule offsets: [(+1,+1),(-1,+1),(+1,-1),(-1,-1)]
  existing weight-2 from my_block, gains 2 from other_block
  CX direction: anc→data

### XX Merge (smooth↔smooth)  
Seam X-stabs: at (x_seam, y) where y even and ((x_seam//2)%2) != ((y//2)%2)
Grown Z-stabs: at (x_seam, y) where y even and ((x_seam//2)%2) == ((y//2)%2)

### Edge coordinate mapping
Block coords: data at (2i+1, 2j+1), range [1, 2d-1], ancillas at even-even [0, 2d]
- "bottom" edge: y_min = 0 (block data starts at y=1). Seam at y_seam = -1 or y_seam = 0
- "top" edge: y_max = 2d (block data ends at y=2d-1). 
- "left" edge: x_min = 0
- "right" edge: x_max = 2d

For ZZ merge ctrl_bottom ↔ anc_top:
  Seam y-coord = between ctrl y=0 and anc y=2d
  In merged frame: seam at ctrl's y=0 = anc's y=2d
  Each seam stab has diagonal neighbors: (x±1, y_seam±1)
  y_seam+1 → data from ctrl block (their y=1 data row)
  y_seam-1 → data from anc block (their y=2d-1 data row)
  (After coordinate transform to put anc below ctrl)
