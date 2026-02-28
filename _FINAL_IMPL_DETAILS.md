# FINAL IMPLEMENTATION DETAILS — Read this first

## COMPLETED
1. ✅ abstract_css.py: Added dataclass imports, SeamStabilizer, GrownStabilizer, 
   MergeStabilizerInfo dataclasses, and get_merge_stabilizers() default method.

## NEXT STEPS
2. rotated_surface.py: 
   - L120: Change `from ..abstract_css import TopologicalCSSCode, Coord2D`
     to: `from ..abstract_css import TopologicalCSSCode, Coord2D, MergeStabilizerInfo, SeamStabilizer, GrownStabilizer`
   - After L549 (hz property): Add get_merge_stabilizers() override method
   - File: src/qectostim/codes/surface/rotated_surface.py (551 lines)
   
3. layout.py: Update add_boundary_bridges() to use seam positions
   - File: src/qectostim/gadgets/layout.py (~1090 lines)
   
4. css_surgery_cnot.py: Rewrite merge phases
   - File: src/qectostim/gadgets/css_surgery_cnot.py (~1436 lines)

5. Syntax verification

## rotated_surface.py CRITICAL FACTS
- Import at L120: `from ..abstract_css import TopologicalCSSCode, Coord2D`
- Class at L127: RotatedSurfaceCode(TopologicalCSSCode)
- Data coords: odd-odd (2i+1, 2j+1) for i,j=0..d-1
- Even-even type: ((x//2)%2) != ((y//2)%2) → True=X, False=Z
- z_schedule = [(+1,+1),(+1,-1),(-1,+1),(-1,-1)] # data→anc
- x_schedule = [(+1,+1),(-1,+1),(+1,-1),(-1,-1)] # anc→data
- End of file:
```
    @property
    def hz(self) -> np.ndarray:
        """Z stabilisers: shape (#Z-checks, #data)."""
        return self._hz
```

## SEAM GEOMETRY FOR get_merge_stabilizers() IMPLEMENTATION

### Algorithm:
1. Determine seam coordinate based on my_edge:
   - "bottom": y_seam = 0, perpendicular axis = x, parallel axis = y
   - "top": y_seam = 2d, perpendicular axis = x, parallel axis = y  
   - "left": x_seam = 0, perpendicular axis = y, parallel axis = x
   - "right": x_seam = 2d, perpendicular axis = y, parallel axis = x

2. For ZZ merge: seam_stab_type="Z", grown_stab_type="X"
   For XX merge: seam_stab_type="X", grown_stab_type="Z"

3. Enumerate even-even positions along the seam line:
   For bottom/top: (x, y_seam) for x=0,2,4,...,2d
   For left/right: (x_seam, y) for y=0,2,4,...,2d

4. Check type of each position:
   type_is_x = ((x//2)%2) != ((y//2)%2)
   If type matches seam_stab_type → SeamStabilizer
   If type matches grown_stab_type → GrownStabilizer

5. For each stab, compute CX neighbors using schedule offsets:
   Z-schedule: [(+1,+1),(+1,-1),(-1,+1),(-1,-1)]
   X-schedule: [(+1,+1),(-1,+1),(+1,-1),(-1,-1)]
   
6. Each neighbor at (sx+dx, sy+dy):
   - If on my_block side → lookup in my_data_global
   - If on other_block side → lookup in other_data_global
   - If out of range → skip (weight reduction)

### How to determine which side a neighbor is on:
For ZZ merge with my_edge="bottom":
  - y_seam = 0
  - Neighbor at y_seam+1 = 1 → belongs to MY block (my data starts at y=1)
  - Neighbor at y_seam-1 = -1 → belongs to OTHER block
    But other block data is at y=1..2d-1 in its own coords
    After coord transform: other_block's y=2d-1 maps to y=-1 in merged frame
    So need to convert: merged_y = -(2d - other_local_y)
    For other data at local (ox, oy): merged = (ox, -(2d - oy)) = (ox, oy - 2d)
    
  Actually simpler: the neighbor exists if the corresponding local data qubit exists
  For my_edge="bottom", seam at y=0:
    Neighbors with dy>0 (y>0) → from my block → local coord is (x+dx, dy) where dy=1
    Neighbors with dy<0 (y<0) → from other block
      Other block's edge is "top" → their data at y=2d-1 are the boundary data
      In merged frame they are at y=-1
      Other local data at (ox, oy) → merged (ox, oy - 2*d_other)
      So merged y=-1 corresponds to other local y = 2*d-1
      Neighbor at (sx+dx, -1): other local coord = (sx+dx, 2*d_other - 1)

  For my_edge = "top", seam at y=2d:
    Neighbors with dy>0 (y=2d+1) → from other block (their bottom)
    Neighbors with dy<0 (y=2d-1) → from my block

  For my_edge = "left", seam at x=0:
    Neighbors with dx>0 (x=1) → from my block
    Neighbors with dx<0 (x=-1) → from other block (their right)

  For my_edge = "right", seam at x=2d:
    Neighbors with dx>0 (x=2d+1) → from other block (their left)
    Neighbors with dx<0 (x=2d-1) → from my block

### Determining local data qubit index from coordinate:
Data at (2i+1, 2j+1) → local index = i * d + j (for d×d data grid)
OR use coord_to_index map: self._coord_to_idx = {(2i+1, 2j+1): i*d+j}

Actually need to build this from self.metadata["data_coords"]:
data_coords = self.metadata["data_coords"]  # list of (x,y) tuples
coord_to_local = {coord: idx for idx, coord in enumerate(data_coords)}

### For other block's data - similar:
other data coords from other_code.metadata["data_coords"]
But we need to map to the correct side of the seam.

### SIMPLIFIED APPROACH:
Rather than complex coordinate transforms, pass in:
- my_boundary_data: list of (local_idx, coord) for data qubits on MY boundary
- other_boundary_data: list of (local_idx, coord) for data qubits on OTHER boundary
Then for each seam stab, check which neighbors are in my_boundary_data vs other_boundary_data.

Actually even simpler: the get_merge_stabilizers() method knows the code geometry.
It can compute everything from:
- d (code distance)
- my_edge (which side faces the merge)
- my_data_global (local_idx → global_idx)
- other_data_global (local_idx → global_idx)
- seam_qubit_offset (where to start numbering seam ancillas)

The method KNOWS the lattice positions of all data qubits (they're at (2i+1, 2j+1)).
So for each seam stab at position (sx, sy), it computes diagonal neighbors and checks
if they're valid data positions, then maps to global indices.

### CX Direction:
Z-type seam stab: CX data→anc → pairs are (data_global, seam_anc_global)
X-type seam stab: CX anc→data → pairs are (seam_anc_global, data_global)

### Grown stabs:
For grown stabs, the EXISTING ancilla is already allocated in one of the blocks.
We need to find its global qubit index. This requires knowing the ancilla allocation
of the block that owns it.

PROBLEM: The abstract interface doesn't pass ancilla global indices.
SOLUTION: Add `my_ancilla_global: Dict[int, int]` parameter (local_anc_idx → global_idx)
OR: Include ancilla info in the method call.

Actually, for grown stabs, the ancilla is an existing EC ancilla in one of the blocks.
Its global index depends on the UnifiedQubitAllocation from FaultTolerantGadgetExperiment.

SIMPLIFICATION: Let the CSSSurgeryCNOTGadget pass the grown stab ancilla info separately,
since it has access to the full qubit allocation. The get_merge_stabilizers() method
returns the lattice positions and CX schedule, and the caller maps to global indices.

REVISED: Instead of global indices in get_merge_stabilizers(), use LOCAL indices
and let the caller do the global mapping. But SeamStabilizer needs global indices
for the seam ancillas (which are newly allocated).

FINAL DESIGN:
- SeamStabilizer.cx_per_phase uses LOCAL data indices and the global seam ancilla idx
- GrownStabilizer.new_cx_per_phase uses LOCAL data indices  
- The caller (css_surgery_cnot.py) maps local→global using my_data_global/other_data_global
- Local indices: identified by (block, local_data_idx) tuples

Actually simplest: return lists of (coord, block_label) per phase, and let caller convert.

## REVISED DESIGN FOR get_merge_stabilizers():

```python
@dataclass
class SeamStabInfo:
    """Seam stabilizer description in coordinate space."""
    lattice_coord: Tuple[int, int]
    stab_type: str  # "X" or "Z"
    # Per-CX-phase list of neighbor coordinates + which block they belong to
    # Each entry: (data_coord, is_my_block: bool) 
    cx_neighbors_per_phase: List[List[Tuple[Tuple[int,int], bool]]]

@dataclass
class GrownStabInfo:
    """Grown boundary stabilizer description."""
    lattice_coord: Tuple[int, int]
    stab_type: str
    is_my_stab: bool  # True if this ancilla belongs to my block
    # NEW CX connections added during merge (from the other block)
    new_cx_per_phase: List[List[Tuple[Tuple[int,int], bool]]]
```

This way the method is pure geometry, independent of global qubit allocation.
The caller (CSSSurgeryCNOTGadget) handles global index mapping.

Let me revise the dataclasses in abstract_css.py accordingly.
Wait — I already committed the dataclasses with global indices. Let me keep them
but make get_merge_stabilizers() return coordinate-based info and let the caller
construct the final SeamStabilizer/GrownStabilizer objects with global indices.

OR: I can make RotatedSurfaceCode.get_merge_stabilizers() work with global indices
since they're passed in as parameters. The method has my_data_global and other_data_global
which give local→global mappings. And seam_qubit_offset gives global indices for new seam ancillas.

For GROWN stabs, I need the existing ancilla's global index. I should pass this as
another parameter or have the caller look it up separately.

DECISION: Add an optional `my_ancilla_global` and `other_ancilla_global` parameter
to get_merge_stabilizers(). But the abstract_css.py already has the signature committed.
Let me just add it.

## ACTUALLY — The simpler approach:

Return MergeStabilizerInfo with COORDINATE-BASED seam info. The SeamStabilizer.cx_per_phase
will use global indices since seam_qubit_offset is provided. For data qubit refs, the method
looks up in my_data_global/other_data_global.

For GrownStabilizer: set existing_ancilla_global = -1 (placeholder), and let caller fill it in
based on coordinate lookup. OR add `ancilla_coord` field and let caller resolve.

Let me just make it work with the existing interface and add ancilla_coord to GrownStabilizer.
