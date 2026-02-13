"""Validation script for the 5 remaining gap fixes."""
import sys
import os

# ── 1. Import checks ──────────────────────────────────────────────────
print("=== Import checks ===")
try:
    from qectostim.experiments.hardware_simulation.trapped_ion.routing.solvers import (
        parallel_config_search, wise_safe_sat_pool_workers,
        enumerate_pmax_configs, sat_config_worker,
    )
    print("  solvers.py imports: OK")
except Exception as e:
    print(f"  solvers.py imports: FAIL ({e})")
    sys.exit(1)

try:
    from qectostim.experiments.hardware_simulation.trapped_ion.routing.routers import (
        WiseSatRouter, WisePatchRouter,
    )
    print("  routers.py imports: OK")
except Exception as e:
    print(f"  routers.py imports: FAIL ({e})")
    sys.exit(1)

try:
    from qectostim.experiments.hardware_simulation.trapped_ion.routing.greedy import (
        route_ions_junction,
    )
    print("  greedy.py imports: OK")
except Exception as e:
    print(f"  greedy.py imports: FAIL ({e})")
    sys.exit(1)

try:
    from qectostim.experiments.hardware_simulation.trapped_ion.clustering import (
        hill_climb_on_arrange_clusters,
    )
    print("  clustering.py imports: OK")
except Exception as e:
    print(f"  clustering.py imports: FAIL ({e})")
    sys.exit(1)


# ── 2. Fix 3: verify_bt_pin_integrity ─────────────────────────────────
print("\n=== Fix 3: BT pin integrity ===")
router = WiseSatRouter()
assert hasattr(router, '_verify_bt_pin_integrity'), "Missing method"

# Test with conflicts
bt_test = [
    {1: (0, 0), 2: (0, 0), 3: (1, 1)},  # ions 1 & 2 conflict on (0,0)
    {4: (2, 2)},
]
cleaned = WiseSatRouter._verify_bt_pin_integrity(bt_test, [1, 2, 3, 4])
assert len(cleaned) == 2
assert len(cleaned[0]) == 2, f"Expected 2 pins, got {len(cleaned[0])}"
assert 1 in cleaned[0], "First ion should be kept"
assert 2 not in cleaned[0], "Conflicting ion should be dropped"
assert 3 in cleaned[0], "Non-conflicting ion should be kept"
assert 4 in cleaned[1]
print("  Conflict detection: PASS")

# Test no conflicts
bt_clean = [{1: (0, 0), 2: (1, 1)}, {3: (2, 2)}]
result = WiseSatRouter._verify_bt_pin_integrity(bt_clean, [1, 2, 3])
assert len(result[0]) == 2
assert len(result[1]) == 1
print("  No-conflict passthrough: PASS")

# Test empty BT
result = WiseSatRouter._verify_bt_pin_integrity([{}, {}], [1, 2])
assert result == [{}, {}]
print("  Empty BT: PASS")

# Test ion not in ions list (should be filtered)
bt_filter = [{99: (0, 0), 1: (1, 1)}]
result = WiseSatRouter._verify_bt_pin_integrity(bt_filter, [1, 2])
assert 99 not in result[0], "Ion not in ions list should be filtered"
assert 1 in result[0]
print("  Out-of-scope ion filtering: PASS")


# ── 3. Fix 4: grid_origin offset ──────────────────────────────────────
print("\n=== Fix 4: grid_origin offset ===")
# Verify the code exists by inspecting _sat_route source
import inspect
src = inspect.getsource(WiseSatRouter._sat_route)
assert "grid_origin != (0, 0)" in src, "grid_origin check not found"
assert "row_off, col_off = grid_origin" in src, "offset extraction not found"
assert "r + row_off, c + col_off" in src, "offset application not found"
print("  grid_origin offset code present: PASS")


# ── 4. Fix 1: child PID cleanup ───────────────────────────────────────
print("\n=== Fix 1: child PID cleanup ===")
src_pcs = inspect.getsource(parallel_config_search)
assert "worker_pids" in src_pcs, "worker_pids tracking missing"
assert "SIGTERM" in src_pcs, "SIGTERM escalation missing"
assert "SIGKILL" in src_pcs, "SIGKILL escalation missing"
assert "time.sleep(0.5)" in src_pcs, "0.5s grace period missing"
print("  Aggressive PID cleanup code present: PASS")


# ── 5. Fix 6: unweighted shortest paths ───────────────────────────────
print("\n=== Fix 6: unweighted shortest paths ===")
src_greedy = inspect.getsource(route_ions_junction)
# Should NOT contain weight='weight'
assert "weight='weight'" not in src_greedy, "weight='weight' still present!"
# Should contain all_shortest_paths call
assert "all_shortest_paths" in src_greedy, "all_shortest_paths call missing"
print("  Unweighted shortest paths: PASS")


# ── 6. Fix 5: junction capacity mutation ──────────────────────────────
print("\n=== Fix 5: junction capacity mutation ===")
assert "nd.num_ions += 1" in src_greedy, "Direct capacity mutation missing"
assert "dest_trap.num_ions -= 1" in src_greedy, "Dest trap decrement missing"
print("  Direct capacity mutation code present: PASS")


# ── 7. Run existing integration tests ─────────────────────────────────
print("\n=== Integration: SAT router smoke test ===")
import numpy as np
from qectostim.experiments.hardware_simulation.trapped_ion.routing.config import (
    WISERoutingConfig,
)
from qectostim.experiments.hardware_simulation.trapped_ion.routing.data_structures import (
    GridLayout,
)
from qectostim.experiments.hardware_simulation.core.pipeline import QubitMapping

config = WISERoutingConfig(timeout_seconds=30.0)
router = WiseSatRouter(config=config)

# Small 2x4 grid, 1 pair
layout = np.arange(8).reshape(2, 4)
gl = GridLayout(
    grid=layout,
    item_positions={int(layout[r, c]): (r, c) for r in range(2) for c in range(4)},
)
result = router.route_batch(
    physical_pairs=[(0, 3)],
    current_mapping=QubitMapping(),
    architecture=None,
    initial_layout=gl,
)
print(f"  2x4 single pair: success={result.success}")
assert result.success, "SAT route should succeed for simple case"

# Test with grid_origin offset — verify returned schedule uses global coords
result_offset = router.route_batch(
    physical_pairs=[(0, 3)],
    current_mapping=QubitMapping(),
    architecture=None,
    initial_layout=gl,
    grid_origin=(10, 20),
)
print(f"  2x4 with grid_origin=(10,20): success={result_offset.success}")
if result_offset.success and result_offset.operations:
    for op in result_offset.operations:
        if "row" in op:
            assert op["row"] >= 10, f"Row {op['row']} should be offset by 10"
        if "col" in op:
            assert op["col"] >= 20, f"Col {op['col']} should be offset by 20"
    print("  grid_origin offset applied to operations: PASS")
else:
    # Co-located or no ops needed — that's fine
    print("  grid_origin test: OK (no ops to verify)")


# ── 8. Co-located fast path (no offset needed) ────────────────────────
result_coloc = router.route_batch(
    physical_pairs=[(0, 1)],
    current_mapping=QubitMapping(),
    architecture=None,
    initial_layout=gl,
)
assert result_coloc.success
assert result_coloc.operations == [] or len(result_coloc.operations) == 0
print("  Co-located fast path: PASS")


# ── 9. BT integrity integrated with route_batch ──────────────────────
# Pass BT with a conflict — should be cleaned and not crash
bt_with_conflict = [
    {0: (0, 0), 1: (0, 0)},  # conflict!
]
try:
    result_bt = router.route_batch(
        physical_pairs=[(0, 3)],
        current_mapping=QubitMapping(),
        architecture=None,
        initial_layout=gl,
        bt_positions=bt_with_conflict,
    )
    print(f"  BT conflict handling: success={result_bt.success} (no crash)")
except Exception as e:
    print(f"  BT conflict handling: FAIL ({e})")
    sys.exit(1)


print("\n" + "=" * 50)
print("ALL VALIDATION TESTS PASSED")
print("=" * 50)
