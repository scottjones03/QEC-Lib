# Lattice Surgery Merge — Comprehensive Fix Plan

## The Problem

The current CSS Surgery merge implementation is **fundamentally wrong**: it uses
weight-2 bridge parity checks (`Z_Di ⊗ Z_Di'` for each boundary data qubit pair)
instead of performing **joint stabiliser measurement of the merged patch** as
Horsman/Fowler lattice surgery requires.

### What It Should Do (Horsman/Fowler)

During a **ZZ merge** (rough-rough), the two code patches become ONE UNIFIED PATCH
for `d` rounds:

1. **Seam Z-stabilisers emerge** at the lattice positions that were truncated at
   both patches' rough boundaries. These are weight-4 (bulk) or weight-2 (corner)
   plaquette operators spanning data qubits from BOTH patches.

2. **Boundary X-stabilisers GROW** from weight-2 to weight-4. The X-ancillas at
   the rough boundary that previously had only 2 data neighbors now gain 2 more
   from the other patch. This is REQUIRED for commutativity: the weight-2
   X-boundary stab `X_D0 X_D1` anti-commutes with the seam Z-stab containing
   `Z_D0` or `Z_D1`. Growing to `X_D0 X_D1 X_D0' X_D1'` restores commutativity.

3. **ALL stabilisers of the joint patch** are measured each round — the regular
   EC stabilisers of both patches + the new seam Z-stabs + the grown boundary
   X-stabs — in a single unified CX schedule (4 phases, no extra TICKs).

4. **d rounds** of joint measurement produce d syndrome layers.

5. `Z_L ⊗ Z_L` is the product of all seam Z-stab outcomes in any single round.

### What It Currently Does (WRONG)

1. Creates `d` bridge ancillas at midpoint positions (not lattice positions)
2. Each bridge measures weight-2 parity: `CX data_ctrl→bridge, CX data_anc→bridge, M bridge`
3. Bridge CX gates are in **2 extra TICK phases** after EC phases (6→8 TICKs/round)
4. Boundary X-stabilisers are NOT grown (remain weight-2, anti-commute with bridges)
5. Observable correction uses bridge split measurements (wrong stabiliser group)

### Concrete Example: d=3 ZZ Merge

**Current (WRONG):** 3 weight-2 bridges measuring:
- Bridge 0: `Z_D0_ctrl ⊗ Z_D0_anc`
- Bridge 1: `Z_D1_ctrl ⊗ Z_D1_anc`
- Bridge 2: `Z_D2_ctrl ⊗ Z_D2_anc`

**Correct (Horsman):** 2 seam Z-stabs measuring:
- Seam Z at (0,0): `Z_D0_ctrl ⊗ Z_D0_anc` (weight-2, corner)
- Seam Z at (4,0): `Z_D1_ctrl ⊗ Z_D2_ctrl ⊗ Z_D1_anc ⊗ Z_D2_anc` (weight-4, bulk)

Plus 1 grown X-stab:
- X at (2,0): `X_D0_ctrl ⊗ X_D1_ctrl ⊗ X_D0_anc ⊗ X_D1_anc`
  (was weight-2 `X_D0 X_D1`, now weight-4 in merged patch)

And ALL regular EC stabs of both patches measured in the same 4-phase schedule.

---

## Seam Stabiliser Geometry (Rotated Surface Code)

### Coordinate Convention
- Data qubits: odd-odd coordinates `(2i+1, 2j+1)` for `i,j = 0..d-1`
- Stabiliser ancillas: even-even coordinates `(2i, 2j)` for `i,j = 0..d`
- X vs Z: `(x//2)%2 != (y//2)%2` → X-type; else Z-type
- Z-ancillas truncated at rough boundaries (top, bottom: y=0 and y=2d)
- X-ancillas truncated at smooth boundaries (left, right: x=0 and x=2d)

### CX Schedule (Z-type ancilla at `(sx,sy)`):
```
Phase 0: data at (sx+1, sy+1)  — SE    CX data→anc
Phase 1: data at (sx+1, sy-1)  — NE    CX data→anc
Phase 2: data at (sx-1, sy+1)  — SW    CX data→anc
Phase 3: data at (sx-1, sy-1)  — NW    CX data→anc
```

### CX Schedule (X-type ancilla at `(sx,sy)`):
```
Phase 0: data at (sx+1, sy+1)  — SE    CX anc→data
Phase 1: data at (sx-1, sy+1)  — SW    CX anc→data
Phase 2: data at (sx+1, sy-1)  — NE    CX anc→data
Phase 3: data at (sx-1, sy-1)  — NW    CX anc→data
```

### ZZ Merge Seam (bottom of block_0 ↔ top of block_1)

Block_0 data at y=1,3,...,2d-1. Bottom boundary at y=1 (y_min of data).
Block_1 data at y=-1,-3,...,-(2d-1) (placed below with block_1 top facing block_0 bottom).
Seam row at y=0.

**Seam Z-stab positions**: `(x, 0)` where `x` is even AND `(x//2)%2 == 0`:
- `x ∈ {0, 4, 8, ..., 4·floor(d/2)}`
- Count = `floor(d/2) + 1`

| Position | Block_0 data (SE,NE) | Block_1 data (SW,NW) | Weight |
|----------|---------------------|---------------------|--------|
| (0, 0)   | D at (1,1)          | D at (1,-1)         | 2      |
| (4k, 0)  | D at (4k±1, 1)      | D at (4k±1, -1)     | 4      |
| Exception: endpoints may be weight-2 if `4k±1 > 2d-1` (out of range) |

**Seam counts by distance:**

| d  | Seam Z-stabs | Weights | Grown X-stabs |
|----|-------------|---------|---------------|
| 2  | 2           | 2,2     | 1             |
| 3  | 2           | 2,4     | 1             |
| 5  | 3           | 2,4,4   | 2             |
| 7  | 4           | 2,4,4,4 | 3             |

**Grown X-stab positions**: At the same seam row, the X-type positions are
`(x, 0)` where `(x//2)%2 == 1`: `x ∈ {2, 6, 10, ...}`.
These were boundary X-stabs (weight-2) in each patch; during merge they grow
to weight-4 by connecting to data from the other patch. Count = `floor(d/2)`.

### XX Merge Seam (right of block_1 ↔ left of block_2)

Analogous but along smooth boundaries. Seam X-stabs emerge, boundary Z-stabs grow.
The seam column is at `x = 2d` (right edge of block_1) = `x = 0` (left edge of block_2).

**Seam X-stab positions**: `(x_seam, y)` where `y` is even AND the position is X-type.
**Grown Z-stab positions**: the remaining even-even positions at the seam column.

### Observable Correction

`s_zz_log = XOR of all seam Z-stab split measurements`

Verification: product of all seam Z-stabs = `Z_D0·Z_D1·...·Z_{d-1}` from block_0
times `Z_D0'·Z_D1'·...·Z_{d-1}'` from block_1 = `Z_L_ctrl ⊗ Z_L_anc`. ✓

Similarly `s_xx_log = XOR of seam X-stabs with odd overlap with X_L`.

---

## Implementation Plan

### Files Changed

1. **`src/qectostim/codes/abstract_css.py`** — Add `get_seam_stabilizers()` method
2. **`src/qectostim/codes/surface/rotated_surface.py`** — Implement for rotated surface code
3. **`src/qectostim/gadgets/layout.py`** — Update bridge allocation to use lattice positions
4. **`src/qectostim/gadgets/css_surgery_cnot.py`** — Major rewrite of merge phases

### Part 1: Code-Level Seam Computation

**File: `abstract_css.py`** — Add abstract method to `TopologicalCSSCode`:

```python
def get_merge_stabilizers(
    self,
    merge_type: Literal["ZZ", "XX"],
    my_edge: str,
    other_boundary_data_global: List[int],
    other_boundary_coords: List[Tuple[float, ...]],
    my_data_global: List[int],
    seam_qubit_offset: int,
) -> MergeStabilizerInfo:
    """Compute seam stabilizers and grown boundary stabs for a merge.

    Parameters
    ----------
    merge_type : "ZZ" or "XX"
        ZZ for rough-rough merge, XX for smooth-smooth merge.
    my_edge : str
        Which edge of THIS block faces the merge boundary.
    other_boundary_data_global : list of int
        Global qubit indices of the OTHER block's boundary data qubits,
        sorted by the perpendicular coordinate.
    other_boundary_coords : list of tuple
        Coordinates of the other block's boundary data in the merged lattice.
    my_data_global : list of int
        Global qubit indices of THIS block's data qubits.
    seam_qubit_offset : int
        Starting global qubit index for seam ancilla allocation.

    Returns
    -------
    MergeStabilizerInfo
        Contains seam_stabilizers, grown_boundary_stabs, and per-phase
        CX schedules for both.
    """
```

**File: `rotated_surface.py`** — Implement `get_merge_stabilizers()`:

Uses the even-even lattice geometry to compute:
1. Seam stabilizer positions from truncated ancilla sites at the merge boundary
2. CX connections for each seam stab: 4 neighbors at (±1,±1) from seam position
3. Grown boundary stabs: existing weight-2 boundary stabs that gain new neighbors
4. CX phase assignment based on the Z/X schedule direction offsets

### Part 2: Layout Bridge Allocation

**File: `layout.py`** — New method `add_seam_stabilizers()`:

Instead of placing bridges at boundary data midpoints, compute the correct
lattice positions for seam ancillas. The coordinates should be at the even-even
positions between the two blocks' data grids.

```python
def add_seam_stabilizers(
    self,
    block_a: str,
    block_b: str,
    edge_a: str,
    edge_b: str,
    merge_type: str,
) -> List[int]:
    """Add seam stabilizer ancillas at correct lattice positions."""
    code = self.blocks[block_a].code
    # Compute seam positions from the code's lattice geometry
    seam_info = code.get_merge_stabilizers(...)
    for pos in seam_info.seam_positions:
        idx = self.add_bridge_ancilla(
            purpose=f"{merge_type}_{pos}",
            coord=pos,
            connected_blocks=[block_a, block_b],
        )
    return indices
```

`compute_layout()` in CSSSurgeryCNOTGadget calls this instead of `add_boundary_bridges`.

### Part 3: Merge Phase Emission (core change)

**File: `css_surgery_cnot.py`** — Rewrite `_emit_joint_merge_round()`:

The new structure for each merge round:

```
1. Reset ALL ancillas (all block EC ancillas + seam ancillas) → TICK
2. CX Phase 0: all block EC CX + seam CX (phase 0) + grown boundary CX (phase 0) → TICK
3. CX Phase 1: all block EC CX + seam CX (phase 1) + grown boundary CX (phase 1) → TICK
4. CX Phase 2: all block EC CX + seam CX (phase 2) + grown boundary CX (phase 2) → TICK
5. CX Phase 3: all block EC CX + seam CX (phase 3) + grown boundary CX (phase 3) → TICK
6. Measure ALL ancillas (all block EC + seam ancillas) + detectors → TICK
```

Key implementation details:

a) **Seam CX per phase**: Pre-compute from the seam stabilizer info. For ZZ merge:
   - Each seam Z-stab has up to 4 CX gates (data→anc), one per phase
   - Only emit CX for neighbors that exist (corner stabs have <4 neighbors)
   
b) **Grown boundary CX per phase**: Pre-compute from grown stab info. For ZZ merge:
   - Each grown X-boundary stab has 2 NEW CX gates (anc→data to OTHER block's data)
   - Assign to the correct CX phases based on the X-schedule offset direction

c) **No extra TICK phases**: Everything integrates into the standard 4-phase schedule.
   The 2 extra TICKs for bridge CX A and B are eliminated.

d) **Seam ancilla reset**: Together with EC ancilla reset (step 1).
   - ZZ merge: seam ancillas → `R` (|0⟩ for Z-type check)
   - XX merge: seam ancillas → `RX` (|+⟩ for X-type check)

e) **Seam ancilla measurement**: Together with EC ancilla measurement (step 6).
   - ZZ merge: seam ancillas → `M` (Z-basis)
   - XX merge: seam ancillas → `MX` (X-basis)

f) **Temporal detectors**: For round ≥ 1, compare current seam ancilla measurement
   with previous round. Same as EC temporal detectors. Round 0 is non-deterministic.

### Part 4: Updated `_emit_zz_merge()`

```python
def _emit_zz_merge(self, circuit, alloc, ctx, ctrl, anc, tgt):
    code = self._code
    d = self.merge_rounds or code.d

    # Compute merge structure (replaces bridge pairing)
    merge_info = code.get_merge_stabilizers(
        merge_type="ZZ",
        my_edge="bottom",  # ctrl's bottom boundary
        other_boundary_data_global=anc.get_boundary_data("top"),
        other_boundary_coords=...,
        my_data_global=ctrl.get_data_qubits(),
        seam_qubit_offset=seam_qubit_indices,
    )

    # Pre-compute per-phase CX lists
    seam_cx_per_phase = merge_info.seam_cx_per_phase   # 4 lists of (ctrl/tgt, anc) pairs
    grown_cx_per_phase = merge_info.grown_cx_per_phase # 4 lists of (anc, data) pairs

    for rnd in range(d):
        self._emit_joint_merge_round_v2(
            builders=self._builders,
            circuit=circuit,
            seam_ancillas=merge_info.seam_ancilla_indices,
            seam_cx_per_phase=seam_cx_per_phase,
            grown_cx_per_phase=grown_cx_per_phase,
            seam_reset_op="R",
            seam_measure_op="M",
            ctx=ctx,
            rnd=rnd,
            ...
        )
```

### Part 5: Updated `get_phase_pairs()` (for hardware routing)

The phase pairs for hardware routing must reflect the new CX structure:
- Each merge round has **4 CX phases** (not 2 bridge layers)
- Each phase may include CX gates from EC, seam, and grown boundary stabs
- Need to return per-phase MS pair lists for the SAT router

The pair structure changes from:
```
[layer_A_ctrl_bridge, layer_B_anc_bridge] × d rounds
```
to:
```
[all_pairs_phase_0, all_pairs_phase_1, all_pairs_phase_2, all_pairs_phase_3] × d rounds
```

But actually, the EC pairs are already handled by the EC routing. The seam + grown
pairs need to be added to the appropriate routing phases.

**KEY INSIGHT**: In the hardware compilation, the EC CX gates are already routed per
block. The seam CX gates need to be added as ADDITIONAL cross-block pairs in the
same routing rounds. This may require changes to the gadget routing pipeline.

### Part 6: Updated Split and Observable Corrections

**ZZ Split**: Destructive `M` on seam Z-ancillas.
- `s_zz_log = XOR of all seam Z-stab split measurements`
- This gives `Z_L_ctrl ⊗ Z_L_anc` (verified: product of all seam Z-stabs = Z_L ⊗ Z_L).

**XX Split**: Destructive `MX` on seam X-ancillas.
- `s_xx_log = XOR of seam X-stabs with odd overlap with X_L`

**Observable correction** remains the same formula:
- `Z_tgt_out = Z_tgt ⊕ Z_ctrl ⊕ s_zz_log`
- `X_ctrl_out = X_ctrl ⊕ m_anc_XL`
- `X_tgt_out = X_tgt ⊕ m_anc_XL ⊕ s_xx_log`

### Part 7: Updated Detectors

a) **Seam temporal detectors** (round ≥ 1): Same pattern as EC temporal detectors.
   Compare current seam ancilla measurement with previous round.

b) **Grown boundary stab detectors**: The grown X-boundary stabs change weight
   from 2 to 4 during merge. The FIRST merge round's grown stab measurement
   differs from the pre-merge round's weight-2 measurement. Need a crossing
   detector that accounts for the weight change.

c) **Split boundary detectors**: Compare split measurement with last merge round's
   seam ancilla measurement.

---

## Generalisation to Other CSS Codes

The implementation generalises via the `get_merge_stabilizers()` method on the code class:

1. **Rotated surface code**: Implemented geometrically using even-even lattice positions
   and ±1 CX offsets (described above).

2. **Unrotated surface code**: Same principle, different coordinate system.
   Seam stabs at the lattice positions truncated at the boundary.

3. **Color codes**: Seam stabs arise at the boundary between two patches.
   The merge structure depends on the lattice geometry (triangular, hexagonal).
   Would need a color-code-specific implementation of `get_merge_stabilizers()`.

4. **General topological CSS codes**: Any code with `has_physical_boundaries() == True`
   and a well-defined coordinate system can implement `get_merge_stabilizers()`.
   The abstract base provides a default that computes seam stabs from boundary
   adjacency analysis.

5. **Algebraic CSS codes** (Steane, etc.): Lattice surgery doesn't apply in the
   traditional sense. These codes would return empty seam info, and the gadget
   would fall back to a teleportation-based CNOT instead of merge/split.

---

## Summary of Changes

| File | Change | Complexity |
|------|--------|-----------|
| `abstract_css.py` | Add `get_merge_stabilizers()` abstract/default method | Medium |
| `rotated_surface.py` | Implement geometric seam computation | Medium |
| `layout.py` | New `add_seam_stabilizers()`, lattice-positioned ancillas | Small |
| `css_surgery_cnot.py` | Rewrite merge phases, integrate CX schedule | Large |
| `css_surgery_cnot.py` | Update `get_phase_pairs()` for hardware routing | Medium |
| `css_surgery_cnot.py` | Update observable + detector corrections | Medium |
| `gadget_routing.py` | Update routing plan for new merge structure | Medium |

**Total estimated effort**: Large. The merge phase emission is a complete rewrite.
The rest are moderate changes that follow from the new merge structure.

---

## Validation Plan

1. **Stim circuit correctness**: For d=3, manually verify that:
   - The merged patch has the correct stabiliser group
   - All stabilisers commute with each other
   - Seam Z-stabs correctly measure `Z_L ⊗ Z_L`
   - Temporal detectors are deterministic for noise-free circuits

2. **`stim.Circuit.detector_error_model()`**: Must run without errors
   (implies all detectors are valid and all measurements are properly tracked).

3. **Logical error rate**: Compare against known surface code lattice surgery
   error rates from literature.

4. **`has_flow()` check**: The stim circuit should have valid detector error model
   flow, meaning all detectors are deterministic in the absence of errors.

5. **Distance verification**: Use `stim.Circuit.shortest_graphlike_error()` to
   verify that the effective distance is preserved (should be d).
