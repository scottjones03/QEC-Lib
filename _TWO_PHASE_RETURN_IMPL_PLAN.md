# Two-Phase SAT-Based Return-to-Home Implementation Plan

## Problem Statement

`_rebuild_schedule_for_layout()` in `qccd_WISE_ion_route.py` is responsible for
routing ions back to their home positions after gadget operations. The current
implementation has 3 key issues:

1. **BT Pin Key Bug**: Uses `(start_row_local, dc)` where `dc` is the target column.
   Should use `(start_row_local, start_col_local)` which is always unique per ion
   (like the old code).

2. **Attempting Too Much Per Patch**: Tries to both move ions TO a patch AND sort
   them WITHIN the patch simultaneously. When ions are scattered across 3+ blocks
   after a gadget merge phase, this creates impossible BT pin configurations.

3. **Checkerboard Tilings Add Overhead**: Multiple overlapping tilings were designed
   for routing multiple rounds of MS gates; for layout-only reconfigurations they
   add SAT solver overhead without benefit.

## Solution: Two-Phase Return Strategy

### Phase 1: "Scatter-to-Home-Patch" (Coarse Routing)

**Goal**: Get every ion into its correct **target patch** (not exact position).

**Approach**:
- For each patch, classify ions into:
  - **Residents**: Already in target patch, want exact position
  - **Transients**: Currently in patch, need to leave (target is different patch)
  - **Incoming**: Currently in another patch, target is this patch

- For **residents**: Use `bt_soft=True` to encourage exact position without hard failure
- For **transients**: Use `cross_boundary_prefs` to push toward exit edge
- No pinning for incoming ions - they'll be handled when they enter their target patch

**Key Insight**: `cross_boundary_prefs` is already implemented and uses soft clauses
with weight ~100 to push ions toward patch edges. This is perfect for coarse routing.

**Tilings**: Use overlapping tilings (checkerboard) so ions can cross patch boundaries
via the offset patches.

**Convergence**: O(W/pw) cycles where W is grid width, pw is patch width.

### Phase 2: "Sort-Within-Patch" (Fine Routing)

**Goal**: All ions are now in their home patches; pin them to exact target positions.

**Approach**:
- Single tiling (0,0 offset) - no overlapping needed since all movements are local
- Use `bt_soft=False` (hard BT) since all ions are in correct patches
- BT pin key: `(start_row_local, start_col_local)` - always unique per ion

**Convergence**: Should succeed in 1 cycle since all ions are local.

### Phase 3: Heuristic Fallback (Already Implemented)

If SAT still fails after Phase 1+2, emit `schedule=None` which triggers
`physicalOperation`'s Phase B/C/D heuristic (odd-even transposition sort).
This is O(m+n) passes and **always succeeds**.

---

## Implementation Changes

### Change 1: Add Helper `_classify_ions_for_patch()`

```python
def _classify_ions_for_patch(
    patch_grid: np.ndarray,
    target_layout: np.ndarray,
    r0: int, c0: int, r1: int, c1: int,
    ion_target_pos: Dict[int, Tuple[int, int]],
) -> Tuple[Dict[int, Tuple[int,int]], Set[int], Set[int]]:
    """
    Classify ions in a patch for two-phase routing.

    Returns:
        residents: {ion_id: (local_target_row, local_target_col)}
            Ions currently in this patch whose global target is also in this patch.
        transients: set of ion_ids
            Ions currently in this patch whose global target is in a different patch.
        incoming_ions: set of ion_ids
            Ions NOT currently in this patch whose global target IS in this patch.
    """
```

### Change 2: Add Helper `_build_boundary_prefs_for_transients()`

```python
def _build_boundary_prefs_for_transients(
    transients: Set[int],
    patch_grid: np.ndarray,
    ion_target_pos: Dict[int, Tuple[int, int]],
    r0: int, c0: int,
) -> Dict[int, Set[str]]:
    """
    Build cross_boundary_prefs for transient ions that need to leave this patch.

    For each transient ion, compute which edge(s) they should move toward
    based on their global target position.
    """
```

### Change 3: Fix BT Pin Key in Phase 2

Change from:
```python
key = (start_row_local, dc)  # BUG: dc is target_col, not unique per ion
```

To:
```python
key = (start_row_local, start_col_local)  # Always unique - each ion has unique position
```

### Change 4: Restructure `_rebuild_schedule_for_layout()` with Two Phases

```python
def _rebuild_schedule_for_layout(...) -> List[...]:
    # ... setup code ...

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Scatter-to-Home-Patch (coarse routing)
    # ═══════════════════════════════════════════════════════════════
    # Goal: Get all ions into their correct target patches.
    # Uses cross_boundary_prefs for transients, soft BT for residents.

    while not all_ions_in_home_patch(current_layout, target_layout):
        for tiling in tilings:  # checkerboard tilings for cross-patch movement
            for patch in patches:
                residents, transients, _ = _classify_ions_for_patch(...)

                # Soft BT pins for residents (exact pos, but relaxable)
                bt_map = {ion: pos for ion, pos in residents.items()}
                # Boundary prefs for transients (push toward exit)
                boundary_prefs = _build_boundary_prefs_for_transients(transients, ...)

                # Run SAT with bt_soft=True
                result = _optimal_QMR_for_WISE(..., bt_soft=True, cross_boundary_prefs=[boundary_prefs])

        if no_progress:
            break  # Move to Phase 2 even if not fully converged

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Sort-Within-Patch (fine routing)
    # ═══════════════════════════════════════════════════════════════
    # Goal: Pin all ions to exact positions within their home patches.
    # Single tiling (0,0), hard BT, uses (start_row_local, start_col_local) key.

    if not np.array_equal(current_layout, target_layout):
        for patch in patches_single_tiling:
            residents, _, _ = _classify_ions_for_patch(...)

            # Hard BT pins with correct key
            bt_map = {}
            used_keys = {}
            for ion, (tr, tc) in residents.items():
                # Key: (start_row_local, start_col_local) - always unique!
                pos = ion_positions[ion]
                sr_local = pos[0] - r0
                sc_local = pos[1] - c0
                key = (sr_local, sc_local)
                used_keys[key] = ion
                bt_map[ion] = (tr, tc)

            result = _optimal_QMR_for_WISE(..., bt_soft=False, ...)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Heuristic Fallback (already implemented)
    # ═══════════════════════════════════════════════════════════════
    if not converged:
        snapshots.append((target_layout.copy(), None, []))  # → Phase B/C/D
```

### Change 5: Simplify Tilings

- Phase 1: Use checkerboard tilings (for cross-patch ion movement)
- Phase 2: Use single tiling only `[(0, 0)]` (all movement is local)

---

## Testing Strategy

### Unit Test 1: `test_classify_ions_for_patch`
```python
def test_classify_ions_for_patch():
    # Create a 4x4 grid with ions at known positions
    # Create target_layout with some ions needing to stay, some needing to leave
    # Verify residents, transients, incoming_ions are correctly classified
```

### Unit Test 2: `test_boundary_prefs_for_transients`
```python
def test_boundary_prefs_for_transients():
    # Create transient ions with known target directions
    # Verify correct boundary prefs are generated (top/bottom/left/right)
```

### Unit Test 3: `test_two_phase_return_simple`
```python
def test_two_phase_return_simple():
    # Create a layout where ions are scrambled within their home patches
    # Run _rebuild_schedule_for_layout
    # Verify it converges to target_layout
```

### Unit Test 4: `test_two_phase_return_cross_patch`
```python
def test_two_phase_return_cross_patch():
    # Create a layout where ions are in WRONG patches (cross-patch scramble)
    # Run _rebuild_schedule_for_layout
    # Verify Phase 1 moves ions to correct patches
    # Verify Phase 2 pins them to exact positions
```

### Integration Test: CSS Surgery Cell 24
```python
# Run cell 24 in trapped_ion_demo.ipynb
# Verify compilation completes successfully
# Check batches_g is populated
# Check no NoFeasibleLayoutError
```

---

## File Changes Summary

| File | Changes |
|------|---------|
| `qccd_WISE_ion_route.py` | Add `_classify_ions_for_patch()`, `_build_boundary_prefs_for_transients()`, restructure `_rebuild_schedule_for_layout()` |

---

## Risk Mitigation

1. **Git commit before refactor** - restore point if needed
2. **Keep existing fallback** - Phase 3 (schedule=None) unchanged
3. **Add logging** - extensive logging at each phase transition
4. **Unit tests** - verify helper functions before integration

---

## Expected Outcome

After implementation:
- CSS Surgery compilation should complete in ~30-60s (down from hanging)
- `_rebuild_schedule_for_layout` makes steady progress via two-phase approach
- BT pin collisions eliminated by correct key
- Graceful fallback if SAT still struggles
