# Implementation Notes — Lattice Surgery Fix

## Files to Modify
1. `src/qectostim/codes/abstract_css.py` — Add MergeStabilizerInfo + get_merge_stabilizers()
2. `src/qectostim/codes/surface/rotated_surface.py` — Implement get_merge_stabilizers() 
3. `src/qectostim/gadgets/layout.py` — Update bridge allocation to use seam stabs
4. `src/qectostim/gadgets/css_surgery_cnot.py` — Major rewrite of merge phases

## Current css_surgery_cnot.py Structure (1436 lines)
- L1-120: imports, class init, __init__ with bridge tracking
- L120-140: set_builders()
- L140-310: get_phase_pairs() — returns layers for hardware routing
- L325-440: _emit_ec_round(), _emit_parallel_ec_round() — EC emission helpers 
- L440-570: _emit_joint_merge_round() — THE CORE MERGE ROUND (bridge + interleaved EC)
  - Currently: Reset EC+bridge → EC CX ph0-3 → bridge CX A → bridge CX B → Measure EC+bridge
  - SHOULD BE: Reset EC+seam → CX ph0-3 (EC+seam+grown together) → Measure EC+seam
- L570-600: emit_next_phase() dispatcher
- L600-750: _emit_zz_merge() — Phase 1 (bridge_zz, ctrl bottom ↔ anc top)
- L750-800: _emit_zz_split() — Phase 2 (M bridge_zz)
- L800-900: _emit_xx_merge() — Phase 3 (bridge_xx, anc right ↔ tgt left)
- L900-935: _emit_xx_split() — Phase 4 (MX bridge_xx)
- L935-1010: _emit_ancilla_measurement() — Phase 5 (MX anc data)
- L1010-1100: Stabiliser/observable transforms, observable config 
- L1100-1200: _get_frame_corrections, _get_zz/xx_logical_bridge_meas, _get_anc_x_logical_meas
- L1200-1436: compute_layout(), get_metadata(), get_preparation_config, etc.

## Current layout.py Bridge Code
- add_boundary_bridges() at ~L791-950 in layout.py
- Places bridges at midpoints of boundary data pairs
- d bridges per merge (one per boundary qubit pair)
- coord computed as midpoint: ((x0+x1)/2, (y0+y1)/2)

## Rotated Surface Code Coordinate Convention
- Data qubits: odd-odd positions (2i+1, 2j+1) for i,j=0..d-1
- Stabiliser ancillas: even-even positions (2i, 2j)
- X vs Z: (x//2)%2 != (y//2)%2 → X-type, else Z-type
- hx, hz: parity check matrices (n_stab × n_data)
- Bottom boundary (rough): y=0 side, Z-type truncated
- Top boundary (rough): y=2d side, Z-type truncated
- Left boundary (smooth): x=0 side, X-type truncated
- Right boundary (smooth): x=2d side, X-type truncated

## CX Schedule for Rotated Surface Code
Z-type ancilla at (sx,sy): 4 phases
  Phase 0: data at (sx+1, sy+1). CX data→anc
  Phase 1: data at (sx+1, sy-1). CX data→anc
  Phase 2: data at (sx-1, sy+1). CX data→anc
  Phase 3: data at (sx-1, sy-1). CX data→anc

X-type ancilla at (sx,sy): 4 phases
  Phase 0: data at (sx+1, sy+1). CX anc→data
  Phase 1: data at (sx-1, sy+1). CX anc→data
  Phase 2: data at (sx+1, sy-1). CX anc→data
  Phase 3: data at (sx-1, sy-1). CX anc→data

## The Z-schedule offsets from css.py builder:
_z_schedule = [(+1,+1), (+1,-1), (-1,+1), (-1,-1)]
_x_schedule = [(+1,+1), (-1,+1), (+1,-1), (-1,-1)]

## Seam Stabilizer Computation for ZZ Merge (bottom of block_0 ↔ top of block_1)

Block_0 data y=1..2d-1 (odd), bottom at y=1.
Block_1 data y' (placed below, with top boundary facing block_0 bottom).
Seam row at y=0 of block_0 (= y_top of block_1 after coordinate transform).

SEAM Z-stab positions (truncated in individual patches but restored during merge):
For ZZ merge (rough-rough boundary), these are Z-type positions at y=0:
- Position (x, 0) where x is even, 0 ≤ x ≤ 2d, AND (x//2)%2 == 0 (Z-type check)
- x ∈ {0, 4, 8, ...} — every 4th x-coordinate

For d=2: positions (0,0) and (4,0)
  (0,0): neighbors (1,1) from block_0, (1,-1) = top boundary of block_1 → weight 2
  (4,0): neighbors (3,1) from block_0, (3,-1) from block_1 → weight 2 (d=2 corner case)
  But (4,0) also could have (5,1) but that's missing for d=2 (max data at x=3)
  Actually for d=2: data at (1,1),(3,1),(1,3),(3,3). So max x_data=3.
  (0,0) neighbors: (1,1)✓ block_0, (-1,1)✗, (-1,-1)✗, (1,-1) needs block_1 → weight 2
  (4,0) neighbors: (3,1)✓ block_0, (5,1)✗ out of range, (3,-1) from block_1, (5,-1)✗ → weight 2

For d=3: positions (0,0) and (4,0)
  (0,0): (1,1)✓ block_0, (1,-1) from block_1 → weight 2 (corner)
  (4,0): (3,1)✓ block_0, (5,1)✓ block_0, (3,-1)✓ block_1, (5,-1)✓ block_1 → weight 4 (bulk)

GROWN X-stab positions (existing weight-2 boundary stabs that grow to weight-4):
X-type positions at y=0:
- Position (x, 0) where x is even, 0 ≤ x ≤ 2d, AND (x//2)%2 == 1 (X-type check)
- x ∈ {2, 6, 10, ...}

For d=2: position (2,0)
  Current (in single patch): at y=0 as boundary X-stab connecting (1,1) and (3,1) → weight 2
  During merge: GAINS (1,-1) and (3,-1) from block_1 → becomes weight 4

For d=3: position (2,0) and (6,0)
  (2,0): was weight-2{(1,1),(3,1)}, gains {(1,-1),(3,-1)} → weight 4
  (6,0): (5,1)✓ block_0, (7,1)✗ out of range for d=3 (max=5), (5,-1)✓ block_1, (7,-1)✗ → weight 2
  Wait... for d=3, max data x = 2d-1 = 5. So (6,0): only (5,1) and (5,-1) → stays weight 2.
  Hmm, but (6,0) with (x//2)%2 = 3%2 = 1 → X-type. And it's on the boundary so might be truncated.
  Actually wait, 2d = 6 for d=3. Position (6,0) is at the corner of the lattice. 
  For d=3: it has neighbors (5,1)✓ and (5,-1)✓ only → weight 2 boundary stab (not grown, just present).
  
## XX Merge Seam (right of block_1 ↔ left of block_2)
Seam column at x=2d of block_1 = x=0 of block_2.

SEAM X-stab positions (truncated at smooth boundaries, restored during merge):
X-type positions at x=2d:
- Position (2d, y) where y is even, 0 ≤ y ≤ 2d, AND type(2d,y) == X
- Type check: (2d//2)%2 = d%2, (y//2)%2 = (y//2)%2
  X-type when d%2 != (y//2)%2

GROWN Z-stab positions:
Z-type positions at x=2d:
- Position (2d, y) where y is even AND type = Z  
- Z-type when d%2 == (y//2)%2

## Key design principle: get_merge_stabilizers() on abstract_css.py
Returns:
- seam_positions: list of (x,y) lattice positions for new seam stabs
- seam_type: "Z" for ZZ merge, "X" for XX merge
- seam_cx_per_phase: 4 lists, each containing (data_qubit, seam_anc) pairs for that CX phase
- grown_positions: existing boundary stab positions that grow during merge
- grown_cx_per_phase: 4 lists of ADDITIONAL (anc, data) pairs for grown stabs
- seam_weights: weight of each seam stabilizer (for detector tracking)

## Implementation Order
1. ✅ Read files — DONE
2. Add MergeStabilizerInfo dataclass to abstract_css.py 
3. Add get_merge_stabilizers() default impl to CSSCode in abstract_css.py
4. Implement get_merge_stabilizers() for RotatedSurfaceCode in rotated_surface.py
5. Update layout.py add_boundary_bridges → add_seam_stabilizers
6. Rewrite _emit_joint_merge_round() 
7. Rewrite _emit_zz_merge(), _emit_xx_merge()
8. Update get_phase_pairs()
9. Update split phases + observable corrections
10. Syntax + import check
