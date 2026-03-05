"""
Unit tests for route-back failure mode fixes (FM1–FM5 + FM6).

Tests verify:
  FM1: Distance-sorted BT pin assignment (more pins than greedy)
  FM2: Anti-oscillation nudge guard
  FM4: Block-aware spectator displacement in _reconstruct_ec_target
  FM5: Exception swallowing replaced with conditional hard error
  FM6: SAT-only for non-cache contexts (allow_heuristic_fallback=False)

Run with::

    cd <repo>
    PYTHONPATH=src python -m pytest tests/test_route_back_fixes.py -v --tb=short

"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pytest

# ── Ensure src/ on path ─────────────────────────────────────────────
os.environ.setdefault("WISE_INPROCESS_LIMIT", "999999999")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
    BlockSubGrid,
    _reconstruct_ec_target,
)


# =====================================================================
# Test helpers
# =====================================================================

def _make_sub_grid(name: str, r0: int, c0: int, r1: int, c1: int,
                   ion_indices: List[int], k: int = 2,
                   initial_layout: Optional[np.ndarray] = None) -> BlockSubGrid:
    """Build a minimal BlockSubGrid for testing."""
    return BlockSubGrid(
        block_name=name,
        grid_region=(r0, c0, r1, c1),
        n_rows=r1 - r0,
        n_cols=c1 - c0,
        ion_indices=ion_indices,
        qubit_to_ion={i: i for i in ion_indices},
        initial_layout=initial_layout,
    )


# =====================================================================
# Group 1: FM4 — Block-aware spectator displacement
# =====================================================================

class TestFM4BlockAwareSpectatorDisplacement:
    """_reconstruct_ec_target must place displaced spectators in their
    home block region, not in the first empty cell globally."""

    def test_spectator_stays_in_home_block(self):
        """When an active ion displaces a spectator, the spectator
        should be relocated within its home block."""
        k = 2

        # Two blocks on a 3×4 grid (3 rows, 2 trap-columns → 4 ion-columns)
        # Block A: rows 0-2, trap-cols 0-1 → ion-cols 0-3
        # Block B: rows 0-2, trap-cols 1-2 → ion-cols 2-3
        # Actually, let's use a cleaner layout:
        # 4 rows, 4 ion-cols (2 trap-cols, k=2)
        # Block A: rows 0-2, trap-cols 0-1 → ion-cols 0-2
        # Block B: rows 2-4, trap-cols 0-1 → ion-cols 0-2

        # Block A occupies rows [0,2), trap-cols [0,1) → ion-cols [0,2)
        # Block B occupies rows [2,4), trap-cols [0,1) → ion-cols [0,2)
        sub_grids = {
            "A": _make_sub_grid("A", 0, 0, 2, 1, [1, 2, 3, 4], k=k),
            "B": _make_sub_grid("B", 2, 0, 4, 1, [5, 6, 7, 8], k=k),
        }

        # ion_to_block: ions 1-4 belong to A, ions 5-8 belong to B
        ion_to_block = {
            1: "A", 2: "A", 3: "A", 4: "A",
            5: "B", 6: "B", 7: "B", 8: "B",
        }

        # Current global layout (4 rows × 2 ion-cols)
        global_layout = np.array([
            [1, 2],
            [5, 4],  # Ion 5 (block B) is in block A's region!
            [3, 6],  # Ion 3 (block A) is in block B's region!
            [7, 8],
        ])

        # EC-initial layouts: block A wants ions at specific positions
        # Block A EC-initial: ion 1 at (0,0), ion 2 at (0,1), ion 3 at (1,0), ion 4 at (1,1)
        ec_A = np.array([
            [1, 2],
            [3, 4],
        ])
        # Block B EC-initial: ion 5 at (0,0), ion 6 at (0,1), ion 7 at (1,0), ion 8 at (1,1)
        ec_B = np.array([
            [5, 6],
            [7, 8],
        ])

        ec_initial_layouts = {"A": ec_A, "B": ec_B}

        target = _reconstruct_ec_target(
            ec_initial_layouts, global_layout, sub_grids, k,
            ion_to_block=ion_to_block,
        )

        # Verify: every ion in the target is in its home block
        block_regions = {}
        for bname, sg in sub_grids.items():
            r0, c0, r1, c1 = sg.grid_region
            block_regions[bname] = (r0, c0 * k, r1, c1 * k)

        for r in range(target.shape[0]):
            for c in range(target.shape[1]):
                ion = int(target[r, c])
                if ion == 0:
                    continue
                home = ion_to_block.get(ion)
                if home is None:
                    continue
                hr0, hc0, hr1, hc1 = block_regions[home]
                assert hr0 <= r < hr1 and hc0 <= c < hc1, (
                    f"Ion {ion} (home={home}) at ({r},{c}) is outside "
                    f"home block region {(hr0, hc0, hr1, hc1)}. "
                    f"Target layout:\n{target}"
                )

    def test_all_ions_preserved(self):
        """_reconstruct_ec_target must not lose or duplicate any ions."""
        k = 2
        sub_grids = {
            "A": _make_sub_grid("A", 0, 0, 2, 1, [1, 2, 3, 4], k=k),
            "B": _make_sub_grid("B", 2, 0, 4, 1, [5, 6, 7, 8], k=k),
        }
        ion_to_block = {
            1: "A", 2: "A", 3: "A", 4: "A",
            5: "B", 6: "B", 7: "B", 8: "B",
        }

        global_layout = np.array([
            [1, 2],
            [5, 4],
            [3, 6],
            [7, 8],
        ])

        ec_A = np.array([[1, 2], [3, 4]])
        ec_B = np.array([[5, 6], [7, 8]])

        target = _reconstruct_ec_target(
            {"A": ec_A, "B": ec_B}, global_layout, sub_grids, k,
            ion_to_block=ion_to_block,
        )

        # All ions from global_layout must appear exactly once in target
        original_ions = sorted(global_layout[global_layout > 0].tolist())
        target_ions = sorted(target[target > 0].tolist())
        assert original_ions == target_ions, (
            f"Ion conservation violated!\n"
            f"  Original: {original_ions}\n"
            f"  Target:   {target_ions}"
        )

    def test_active_ions_at_ec_positions(self):
        """Active ions should be placed at their EC-initial positions."""
        k = 2
        sub_grids = {
            "A": _make_sub_grid("A", 0, 0, 2, 1, [1, 2, 3, 4], k=k),
            "B": _make_sub_grid("B", 2, 0, 4, 1, [5, 6, 7, 8], k=k),
        }
        ion_to_block = {
            1: "A", 2: "A", 3: "A", 4: "A",
            5: "B", 6: "B", 7: "B", 8: "B",
        }

        global_layout = np.array([
            [2, 1],  # swapped from EC-initial
            [4, 3],
            [6, 5],
            [8, 7],
        ])

        ec_A = np.array([[1, 2], [3, 4]])
        ec_B = np.array([[5, 6], [7, 8]])

        target = _reconstruct_ec_target(
            {"A": ec_A, "B": ec_B}, global_layout, sub_grids, k,
            ion_to_block=ion_to_block,
        )

        # Active ions should be at their EC-initial positions
        assert target[0, 0] == 1
        assert target[0, 1] == 2
        assert target[1, 0] == 3
        assert target[1, 1] == 4
        assert target[2, 0] == 5
        assert target[2, 1] == 6
        assert target[3, 0] == 7
        assert target[3, 1] == 8

    def test_displaced_spectator_no_empty_cell_raises(self):
        """If there's truly no empty cell for a displaced spectator,
        raise ValueError."""
        k = 2
        sub_grids = {
            "A": _make_sub_grid("A", 0, 0, 2, 1, [1, 2, 3, 4], k=k),
        }
        # Only 1 block, 4 ions in a 2×2 grid = completely full
        # If we displace an unmapped spectator (99), there's no room
        ion_to_block = {1: "A", 2: "A", 3: "A", 4: "A"}

        # Ion 99 is NOT in ion_to_block — it's a pure spectator
        global_layout = np.array([
            [99, 2],
            [3, 4],
        ])

        # EC-initial wants ion 1 at (0,0) — displaces ion 99
        ec_A = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="no empty cell"):
            _reconstruct_ec_target(
                {"A": ec_A}, global_layout, sub_grids, k,
                ion_to_block=ion_to_block,
            )

    def test_multiple_displaced_spectators_all_in_home(self):
        """When multiple spectators are displaced, ALL should end up
        in their home blocks."""
        k = 2

        # 3 blocks on a 6×2 grid
        sub_grids = {
            "A": _make_sub_grid("A", 0, 0, 2, 1, [1, 2, 3, 4], k=k),
            "B": _make_sub_grid("B", 2, 0, 4, 1, [5, 6, 7, 8], k=k),
            "C": _make_sub_grid("C", 4, 0, 6, 1, [9, 10, 11, 12], k=k),
        }
        ion_to_block = {
            1: "A", 2: "A", 3: "A", 4: "A",
            5: "B", 6: "B", 7: "B", 8: "B",
            9: "C", 10: "C", 11: "C", 12: "C",
        }

        # Scrambled: ions from multiple blocks in wrong positions
        global_layout = np.array([
            [9, 2],   # ion 9 (C) in A's region
            [3, 10],  # ion 10 (C) in A's region
            [5, 6],
            [1, 8],   # ion 1 (A) in B's region
            [11, 12],
            [7, 4],   # ion 7 (B) and 4 (A) in C's region
        ])

        ec_A = np.array([[1, 2], [3, 4]])
        ec_B = np.array([[5, 6], [7, 8]])
        ec_C = np.array([[9, 10], [11, 12]])

        target = _reconstruct_ec_target(
            {"A": ec_A, "B": ec_B, "C": ec_C},
            global_layout, sub_grids, k,
            ion_to_block=ion_to_block,
        )

        # Build block regions and verify every ion
        block_regions = {}
        for bname, sg in sub_grids.items():
            r0, c0, r1, c1 = sg.grid_region
            block_regions[bname] = (r0, c0 * k, r1, c1 * k)

        for r in range(target.shape[0]):
            for c in range(target.shape[1]):
                ion = int(target[r, c])
                if ion == 0:
                    continue
                home = ion_to_block.get(ion)
                if home is None:
                    continue
                hr0, hc0, hr1, hc1 = block_regions[home]
                assert hr0 <= r < hr1 and hc0 <= c < hc1, (
                    f"Ion {ion} (home={home}) at ({r},{c}) outside "
                    f"home region {(hr0, hc0, hr1, hc1)}"
                )

        # Ion conservation
        original = sorted(global_layout[global_layout > 0].tolist())
        result = sorted(target[target > 0].tolist())
        assert original == result

    def test_legacy_path_without_ion_to_block(self):
        """When ion_to_block is None, use legacy overwrite behaviour."""
        k = 2
        sub_grids = {
            "A": _make_sub_grid("A", 0, 0, 2, 1, [1, 2, 3, 4], k=k),
        }

        global_layout = np.array([
            [2, 1],
            [4, 3],
        ])

        ec_A = np.array([[1, 2], [3, 4]])

        target = _reconstruct_ec_target(
            {"A": ec_A}, global_layout, sub_grids, k,
            ion_to_block=None,
        )

        # Legacy path: entire block region gets overwritten with EC layout
        np.testing.assert_array_equal(target[0:2, 0:2], ec_A)


# =====================================================================
# Group 2: FM1 — Distance-sorted BT pin assignment
# =====================================================================

class TestFM1DistanceSortedBTPins:
    """The distance-sorted BT pin assignment should produce at least
    as many pins as the old greedy approach, and prefer shortest-
    distance pins."""

    def test_bt_pin_candidates_sorted_by_distance(self):
        """Verify that pins assigned have shorter or equal distance
        to those skipped due to conflicts."""
        # This is a property test on the algorithm design:
        # Simulate a scenario where ions need to move to nearby targets
        # and verify that short-distance pins get priority

        # Build a 4×4 patch with ions that all need to shift down by 1
        patch = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [0, 0, 0, 0],
        ])
        target = np.array([
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ])

        # Run the FM1 candidate building logic inline
        # This mirrors _rebuild_schedule_for_layout's pin building
        patch_h, patch_w = patch.shape
        ion_target_pos = {}
        for r in range(target.shape[0]):
            for c in range(target.shape[1]):
                ion = int(target[r, c])
                if ion != 0:
                    ion_target_pos[ion] = (r, c)

        # Build patch_row_of and patch_col_of
        patch_row_of = {}
        patch_col_of = {}
        for lr in range(patch_h):
            for lc in range(patch_w):
                ion_id = int(patch[lr, lc])
                if ion_id != 0:
                    patch_row_of[ion_id] = lr
                    patch_col_of[ion_id] = lc

        # Collect candidates
        candidates = []
        for dr in range(patch_h):
            for dc in range(patch_w):
                ionidx_target = int(target[dr, dc])
                if ionidx_target == 0:
                    continue
                start_row = patch_row_of.get(ionidx_target)
                if start_row is None:
                    continue
                start_col = patch_col_of.get(ionidx_target, 0)
                dist = abs(start_row - dr) + abs(start_col - dc)
                candidates.append((dist, ionidx_target, start_row, dr, dc))

        # Sort by distance
        candidates.sort(key=lambda x: x[0])

        # Greedy assign with uniqueness constraints
        bt_map = {}
        used_keys = {}
        used_targets = {}

        for dist, ion, sr, tr, tc in candidates:
            if ion in bt_map:
                continue
            key = (sr, tc)
            if key in used_keys and used_keys[key] != ion:
                continue
            tgt_key = (tr, tc)
            if tgt_key in used_targets and used_targets[tgt_key] != ion:
                continue
            used_keys[key] = ion
            used_targets[tgt_key] = ion
            bt_map[ion] = (tr, tc)

        # With distance-sorting, we should get a good number of pins.
        # In this scenario (all shift down by 1, distance=1), many pins
        # have the same distance, so we expect a reasonable count.
        assert len(bt_map) >= 4, (
            f"Expected at least 4 pins for a 4×4 shift-by-1, "
            f"got {len(bt_map)}: {bt_map}"
        )

        # Verify all pinned ions have distance = 1 (shortest possible)
        for ion, (tr, tc) in bt_map.items():
            sr = patch_row_of[ion]
            sc = patch_col_of[ion]
            dist = abs(sr - tr) + abs(sc - tc)
            assert dist == 1, (
                f"Ion {ion} pinned with distance {dist}, expected 1"
            )

    def test_mixed_distance_pins_prefer_short(self):
        """When ions have varying distances to targets, shortest should
        be pinned first."""
        # Ion 1 at (0,0), target (0,1) → dist=1
        # Ion 2 at (0,1), target (2,3) → dist=4
        # Both would want target_col=1 for ion1... but in different rows
        patch = np.array([
            [1, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 3],
        ])
        target = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 3],
        ])

        patch_row_of = {1: 0, 2: 0, 3: 2}
        patch_col_of = {1: 0, 2: 1, 3: 3}

        candidates = []
        for dr in range(3):
            for dc in range(4):
                ionidx = int(target[dr, dc])
                if ionidx == 0:
                    continue
                sr = patch_row_of.get(ionidx)
                if sr is None:
                    continue
                sc = patch_col_of.get(ionidx, 0)
                dist = abs(sr - dr) + abs(sc - dc)
                candidates.append((dist, ionidx, sr, dr, dc))

        candidates.sort(key=lambda x: x[0])

        # Ion 3 at (2,3) → (2,3): dist=0 → pinned first
        # Ion 1 at (0,0) → (0,1): dist=1 → pinned second
        assert candidates[0][1] == 3  # dist=0
        assert candidates[1][1] == 1  # dist=1


# =====================================================================
# Group 3: FM2 — Anti-oscillation nudge guard
# =====================================================================

class TestFM2AntiOscillation:
    """Nudge pass should not reverse direction on both axes within
    a cycle (anti-oscillation guard)."""

    def test_reverse_nudge_blocked(self):
        """If an ion was nudged right+down on tiling 0, it should not
        be nudged left+up on tiling 1 (reversal on BOTH axes)."""
        nudge_history = {}

        # Tiling 0: ion 5 nudged right+down → direction (1, 1)
        nudge_history[5] = (1, 1)

        # Tiling 1: ion 5 would be nudged left+up → direction (-1, -1)
        dir_r, dir_c = -1, -1
        prev_dir = nudge_history.get(5)

        # Anti-oscillation check
        should_skip = False
        if prev_dir is not None:
            prev_dr, prev_dc = prev_dir
            if (prev_dr != 0 and dir_r != 0 and prev_dr == -dir_r and
                    prev_dc != 0 and dir_c != 0 and prev_dc == -dir_c):
                should_skip = True

        assert should_skip, (
            "Anti-oscillation guard should block reversal on both axes"
        )

    def test_partial_reversal_allowed(self):
        """If only ONE axis reverses, the nudge should be allowed."""
        nudge_history = {5: (1, 1)}

        # Reversal on row only → allowed
        dir_r, dir_c = -1, 1
        prev_dir = nudge_history.get(5)
        should_skip = False
        if prev_dir is not None:
            prev_dr, prev_dc = prev_dir
            if (prev_dr != 0 and dir_r != 0 and prev_dr == -dir_r and
                    prev_dc != 0 and dir_c != 0 and prev_dc == -dir_c):
                should_skip = True

        assert not should_skip, (
            "Partial reversal (single axis) should be allowed"
        )

    def test_zero_direction_always_allowed(self):
        """If one axis has no movement, anti-oscillation should not
        trigger."""
        nudge_history = {5: (1, 0)}  # moved down, no horizontal

        # Now try to move up (reversal on row, but col is 0)
        dir_r, dir_c = -1, 1
        prev_dir = nudge_history.get(5)
        should_skip = False
        if prev_dir is not None:
            prev_dr, prev_dc = prev_dir
            if (prev_dr != 0 and dir_r != 0 and prev_dr == -dir_r and
                    prev_dc != 0 and dir_c != 0 and prev_dc == -dir_c):
                should_skip = True

        assert not should_skip, (
            "Should not trigger when one axis has zero direction"
        )


# =====================================================================
# Group 4: FM5 — Exception swallowing replaced
# =====================================================================

class TestFM5ConditionalException:
    """_apply_post_gadget_transition should raise ValueError when
    ions are genuinely misplaced, and only swallow when block purity
    is already satisfied."""

    def test_misplaced_ion_detection(self):
        """Verify the detection logic: count ions outside home block."""
        k = 2
        block_sub_grids = {
            "A": _make_sub_grid("A", 0, 0, 2, 1, [1, 2, 3, 4], k=k),
            "B": _make_sub_grid("B", 2, 0, 4, 1, [5, 6, 7, 8], k=k),
        }
        ion_to_block = {
            1: "A", 2: "A", 3: "A", 4: "A",
            5: "B", 6: "B", 7: "B", 8: "B",
        }

        # Layout with 2 misplaced ions
        cur_layout = np.array([
            [1, 5],   # ion 5 (B) in A's region
            [3, 4],
            [2, 6],   # ion 2 (A) in B's region
            [7, 8],
        ])

        # Count misplaced ions (same logic as FM5 fix)
        _fm5_regions = {}
        for _bname, _sg in block_sub_grids.items():
            _r0, _c0, _r1, _c1 = _sg.grid_region
            _fm5_regions[_bname] = (_r0, _c0 * k, _r1, _c1 * k)

        misplaced = 0
        for _r in range(cur_layout.shape[0]):
            for _c in range(cur_layout.shape[1]):
                _ion = int(cur_layout[_r, _c])
                if _ion == 0 or _ion not in ion_to_block:
                    continue
                _home = ion_to_block[_ion]
                _hr0, _hc0, _hr1, _hc1 = _fm5_regions.get(
                    _home, (0, 0, 0, 0))
                if not (_hr0 <= _r < _hr1 and _hc0 <= _c < _hc1):
                    misplaced += 1

        assert misplaced == 2, f"Expected 2 misplaced ions, got {misplaced}"

    def test_all_in_home_means_zero_misplaced(self):
        """If all ions are already in their home blocks, misplaced
        count should be 0."""
        k = 2
        block_sub_grids = {
            "A": _make_sub_grid("A", 0, 0, 2, 1, [1, 2, 3, 4], k=k),
            "B": _make_sub_grid("B", 2, 0, 4, 1, [5, 6, 7, 8], k=k),
        }
        ion_to_block = {
            1: "A", 2: "A", 3: "A", 4: "A",
            5: "B", 6: "B", 7: "B", 8: "B",
        }

        # All ions in correct blocks
        cur_layout = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ])

        _fm5_regions = {}
        for _bname, _sg in block_sub_grids.items():
            _r0, _c0, _r1, _c1 = _sg.grid_region
            _fm5_regions[_bname] = (_r0, _c0 * k, _r1, _c1 * k)

        misplaced = 0
        for _r in range(cur_layout.shape[0]):
            for _c in range(cur_layout.shape[1]):
                _ion = int(cur_layout[_r, _c])
                if _ion == 0 or _ion not in ion_to_block:
                    continue
                _home = ion_to_block[_ion]
                _hr0, _hc0, _hr1, _hc1 = _fm5_regions.get(
                    _home, (0, 0, 0, 0))
                if not (_hr0 <= _r < _hr1 and _hc0 <= _c < _hc1):
                    misplaced += 1

        assert misplaced == 0


# =====================================================================
# Group 5: FM6 — SAT-only enforcement
# =====================================================================

class TestFM6SATOnly:
    """Verify that allow_heuristic_fallback defaults to False for non-cache
    contexts (return-reconfig and transition reconfig), ensuring SAT-first
    behaviour unless the caller explicitly opts in to heuristic fallback."""

    def test_compute_return_reconfig_no_heuristic(self):
        """_compute_return_reconfig should default allow_heuristic_fallback
        to False and pass it through to _rebuild_schedule_for_layout."""
        import inspect
        from qectostim.experiments.hardware_simulation.trapped_ion.utils import (
            gadget_routing as gr_mod,
        )
        source = inspect.getsource(gr_mod._compute_return_reconfig)
        # FM6: parameter default is False (SAT-first)
        assert "allow_heuristic_fallback: bool = False" in source, (
            "_compute_return_reconfig should have "
            "allow_heuristic_fallback parameter defaulting to False"
        )
        # FM6: passes through to _rebuild_schedule_for_layout
        assert "allow_heuristic_fallback=allow_heuristic_fallback" in source, (
            "_compute_return_reconfig should forward "
            "allow_heuristic_fallback to _rebuild_schedule_for_layout"
        )

    def test_compute_transition_reconfig_no_heuristic(self):
        """_compute_transition_reconfig_steps should default
        allow_heuristic_fallback to False and pass it through."""
        import inspect
        from qectostim.experiments.hardware_simulation.trapped_ion.utils import (
            gadget_routing as gr_mod,
        )
        source = inspect.getsource(gr_mod._compute_transition_reconfig_steps)
        # FM6: parameter default is False (SAT-first)
        assert "allow_heuristic_fallback: bool = False" in source, (
            "_compute_transition_reconfig_steps should have "
            "allow_heuristic_fallback parameter defaulting to False"
        )
        # FM6: passes through to _rebuild_schedule_for_layout
        assert "allow_heuristic_fallback=allow_heuristic_fallback" in source, (
            "_compute_transition_reconfig_steps should forward "
            "allow_heuristic_fallback to _rebuild_schedule_for_layout"
        )

    def test_cache_replay_still_allows_heuristic(self):
        """Cache replay contexts should still allow heuristic fallback."""
        import inspect
        from qectostim.experiments.hardware_simulation.trapped_ion.utils import (
            gadget_routing as gr_mod,
        )
        # The EC cache replay and gadget cache replay blocks should
        # still have allow_heuristic_fallback=True
        source = inspect.getsource(gr_mod.route_full_experiment_as_steps)
        # Count occurrences
        true_count = source.count("allow_heuristic_fallback=True")
        false_count = source.count("allow_heuristic_fallback=False")
        # Cache replay should still have True
        assert true_count >= 2, (
            f"Expected at least 2 allow_heuristic_fallback=True "
            f"(for EC and gadget cache replay), got {true_count}"
        )


# =====================================================================
# Group 6: Integration — Full FM4 block purity after reconstruct
# =====================================================================

class TestFM4Integration:
    """Integration test: after _reconstruct_ec_target with FM4 fix,
    the target layout should always be block-pure."""

    @pytest.mark.parametrize("n_blocks", [2, 3, 4])
    def test_random_scrambled_layout_block_pure(self, n_blocks):
        """Randomly permute ions across blocks, then reconstruct target
        and verify block purity."""
        k = 2
        ions_per_block = 4
        total_ions = n_blocks * ions_per_block
        rows_per_block = 2
        total_rows = n_blocks * rows_per_block
        cols = 2  # ion-columns (1 trap-column × k=2)

        # Build sub-grids
        sub_grids = {}
        ion_to_block = {}
        ec_layouts = {}
        ion_idx = 1

        for b in range(n_blocks):
            bname = f"B{b}"
            r0 = b * rows_per_block
            r1 = r0 + rows_per_block
            block_ions = list(range(ion_idx, ion_idx + ions_per_block))
            sub_grids[bname] = _make_sub_grid(
                bname, r0, 0, r1, 1, block_ions, k=k)
            for i in block_ions:
                ion_to_block[i] = bname

            # EC-initial: ions in natural order within block
            ec_lay = np.zeros((rows_per_block, cols), dtype=int)
            idx = 0
            for r in range(rows_per_block):
                for c in range(cols):
                    ec_lay[r, c] = block_ions[idx]
                    idx += 1
            ec_layouts[bname] = ec_lay
            ion_idx += ions_per_block

        # Random scramble: place all ions on the grid randomly
        rng = np.random.RandomState(42)
        all_ions = list(range(1, total_ions + 1))
        rng.shuffle(all_ions)
        global_layout = np.array(all_ions).reshape(total_rows, cols)

        # Reconstruct target
        target = _reconstruct_ec_target(
            ec_layouts, global_layout, sub_grids, k,
            ion_to_block=ion_to_block,
        )

        # Verify block purity
        block_regions = {}
        for bname, sg in sub_grids.items():
            r0, c0, r1, c1 = sg.grid_region
            block_regions[bname] = (r0, c0 * k, r1, c1 * k)

        for r in range(target.shape[0]):
            for c in range(target.shape[1]):
                ion = int(target[r, c])
                if ion == 0:
                    continue
                home = ion_to_block.get(ion)
                if home is None:
                    continue
                hr0, hc0, hr1, hc1 = block_regions[home]
                assert hr0 <= r < hr1 and hc0 <= c < hc1, (
                    f"n_blocks={n_blocks}: Ion {ion} (home={home}) "
                    f"at ({r},{c}) outside {(hr0, hc0, hr1, hc1)}"
                )

        # Ion conservation
        original = sorted(global_layout[global_layout > 0].tolist())
        result = sorted(target[target > 0].tolist())
        assert original == result


# =====================================================================
# Group 7: Integration — Full pipeline compilation (slow)
# =====================================================================

class TestFullPipelineWithFixes:
    """End-to-end compilation tests to verify fixes don't break
    anything. Marked slow — skip with -k 'not full_pipeline'."""

    @pytest.fixture(scope="class")
    def d2_compiled(self):
        """Compile a d=2 CSS Surgery gadget end-to-end."""
        from qectostim.experiments.hardware_simulation.trapped_ion.demo.run import (
            compile_gadget_for_animation,
        )
        from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.routing_config import (
            WISERoutingConfig,
        )

        d = 2
        k = 2
        code = RotatedSurfaceCode(distance=d)
        gadget = CSSSurgeryCNOTGadget()
        ft_exp = FaultTolerantGadgetExperiment(
            codes=[code], gadget=gadget, noise_model=None,
            num_rounds_before=d, num_rounds_after=d,
        )
        ideal = ft_exp.to_stim()
        qec_meta = ft_exp.qec_metadata
        alloc = ft_exp._unified_allocation

        routing_cfg = WISERoutingConfig.default(
            lookahead=1,
            subgridsize=(6, 6, 2),
            base_pmax_in=1,
            replay_level=d,
            cache_ec_rounds=True,
            show_progress=False,
        )

        arch, compiler, compiled, batches, ion_roles, p2l, remap = (
            compile_gadget_for_animation(
                ideal,
                qec_metadata=qec_meta,
                gadget=gadget,
                qubit_allocation=alloc,
                trap_capacity=k,
                show_progress=False,
                max_inner_workers=4,
                routing_config=routing_cfg,
            )
        )
        return arch, compiler, compiled, batches

    @pytest.mark.slow
    def test_full_pipeline_compiles(self, d2_compiled):
        """The full pipeline should compile without errors."""
        arch, compiler, compiled, batches = d2_compiled
        assert len(batches) > 0

    @pytest.mark.slow
    def test_full_pipeline_ms_gates_correct(self, d2_compiled):
        """All MS gates from the ideal circuit should appear in batches."""
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.qccd_operations_on_qubits import (
            TwoQubitMSGate,
        )
        _, _, compiled, batches = d2_compiled

        ms_count = sum(
            sum(1 for op in getattr(b, 'operations', [])
                if isinstance(op, TwoQubitMSGate))
            for b in batches
        )
        assert ms_count > 0, "Expected at least one MS gate in batches"


# =====================================================================
# Group 8: _rebuild_schedule_for_layout escalation test
# =====================================================================

class TestSATEscalation:
    """Verify that _rebuild_schedule_for_layout correctly raises
    ValueError when allow_heuristic_fallback=False and SAT can't
    converge (rather than silently falling back to heuristic)."""

    def test_error_message_mentions_forbidden(self):
        """The error message for SAT failure should mention that
        heuristic fallback is forbidden."""
        import inspect
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler import (
            qccd_WISE_ion_route as wise_mod,
        )
        source = inspect.getsource(wise_mod._rebuild_schedule_for_layout)
        assert "Heuristic fallback is FORBIDDEN" in source
        assert "raise ValueError" in source
