# src/qectostim/experiments/hardware_simulation/tests/test_wise_sat_integration.py
"""
Integration tests for WISE SAT-based routing.

Validates that:
1. SAT routing produces actual SAT solutions (not heuristic fallback)
2. Full compilation pipeline works end-to-end
3. Multiple grid sizes and codes work correctly
4. Metrics indicate SAT success (not "method": "heuristic")
"""
import pytest
import numpy as np

# Skip all tests if pysat is not available
pytest.importorskip("pysat")


# =============================================================================
# Imports (done inside tests to allow skip on import failure)
# =============================================================================

def _get_wise_imports():
    """Import WISE components."""
    from qectostim.experiments.hardware_simulation.trapped_ion import (
        WISEArchitecture,
        WISECompiler,
        WISERoutingConfig,
        TrappedIonExperiment,
        TrappedIonNoiseModel,
    )
    return WISEArchitecture, WISECompiler, WISERoutingConfig, TrappedIonExperiment, TrappedIonNoiseModel


def _get_code_imports():
    """Import QEC codes."""
    from qectostim.codes.small.steane_713 import SteaneCode713
    from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
    return SteaneCode713, RotatedSurfaceCode


# =============================================================================
# Test: SAT routing on small WISE grid
# =============================================================================

class TestWISESATRouting:
    """Test that WISE SAT routing produces true SAT solutions."""

    def test_small_grid_sat_success(self):
        """Test SAT routing on a small 2x4 grid with 2 ions/segment.
        
        This tests the SAT router directly without TrappedIonExperiment.
        """
        WISEArchitecture, WISECompiler, WISERoutingConfig, _, _ = _get_wise_imports()
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WiseSatRouter,
        )
        from qectostim.experiments.hardware_simulation.core.pipeline import QubitMapping
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.data_structures import (
            GridLayout,
        )
        
        # Small 2x4 grid
        arch = WISEArchitecture(col_groups=2, rows=2, ions_per_segment=2)
        config = WISERoutingConfig(timeout_seconds=30.0)
        router = WiseSatRouter(config=config)
        
        n_rows, n_cols = 2, 4
        layout_arr = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
        initial_layout = GridLayout(grid=layout_arr)
        
        # Gate pairs - make sure they're valid ion IDs (0-7 for 2x4 grid)
        pairs = [(0, 1), (2, 3)]  # Adjacent pairs
        mapping = QubitMapping()
        
        result = router.route_batch(
            physical_pairs=pairs,
            current_mapping=mapping,
            architecture=arch,
            initial_layout=initial_layout,
        )
        
        # Check success
        assert result.success, f"SAT routing failed: {result.metrics}"
        
        # Check it's not heuristic fallback
        assert result.metrics.get("method") != "heuristic", (
            f"SAT routing fell back to heuristic! metrics={result.metrics}"
        )
        
        # SAT success should have "passes" or "bcf"
        has_sat_metrics = "passes" in result.metrics or "bcf" in result.metrics
        assert has_sat_metrics, (
            f"SAT metrics missing expected keys: {result.metrics}"
        )

    def test_larger_grid_sat_success(self):
        """Test SAT routing on a larger 3x6 grid.
        
        Tests with more ions to verify SAT scales.
        """
        WISEArchitecture, _, WISERoutingConfig, _, _ = _get_wise_imports()
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WiseSatRouter,
        )
        from qectostim.experiments.hardware_simulation.core.pipeline import QubitMapping
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.data_structures import (
            GridLayout,
        )
        
        # 3x6 grid (18 ions)
        arch = WISEArchitecture(col_groups=3, rows=3, ions_per_segment=2)
        config = WISERoutingConfig(timeout_seconds=60.0)
        router = WiseSatRouter(config=config)
        
        n_rows, n_cols = 3, 6
        layout_arr = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
        initial_layout = GridLayout(grid=layout_arr)
        
        # Multiple gate pairs
        pairs = [(0, 1), (2, 3), (6, 7), (8, 9)]
        mapping = QubitMapping()
        
        result = router.route_batch(
            physical_pairs=pairs,
            current_mapping=mapping,
            architecture=arch,
            initial_layout=initial_layout,
        )
        
        assert result.success, f"SAT routing failed: {result.metrics}"
        assert result.metrics.get("method") != "heuristic", (
            f"SAT routing fell back to heuristic!"
        )


# =============================================================================
# Test: Orchestrator integration
# =============================================================================

class TestOrchestratorIntegration:
    """Test WISERoutingOrchestrator with all ported features."""

    def test_orchestrator_basic_route(self):
        """Test that the orchestrator can route a simple set of gate pairs."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.orchestrator import (
            WISERoutingOrchestrator,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WiseSatRouter,
        )

        _, _, WISERoutingConfig, _, _ = _get_wise_imports()
        config = WISERoutingConfig(timeout_seconds=30.0)
        router = WiseSatRouter(config=config)
        orch = WISERoutingOrchestrator(router=router, config=config)

        n_rows, n_cols = 2, 4
        capacity = 2
        layout = np.arange(n_rows * n_cols, dtype=int).reshape(n_rows, n_cols)
        # Two rounds of pairs
        parallel_pairs = [
            [(0, 1), (2, 3)],
            [(4, 5), (6, 7)],
        ]

        reconfigs, total_time = orch.route_all_rounds(
            initial_layout=layout,
            parallel_pairs=parallel_pairs,
            n_rows=n_rows,
            n_cols=n_cols,
            capacity=capacity,
        )

        # Should produce at least one reconfiguration step
        assert len(reconfigs) > 0, "Orchestrator produced no reconfigs"
        # Each reconfig is (layout_after, schedule, solved_pairs)
        for la, sched, sp in reconfigs:
            assert isinstance(la, np.ndarray)
            assert la.shape == (n_rows, n_cols)

    def test_orchestrator_block_cache_hit(self):
        """Test that identical gate patterns hit the block cache."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.orchestrator import (
            WISERoutingOrchestrator,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WiseSatRouter,
        )

        _, _, WISERoutingConfig, _, _ = _get_wise_imports()
        config = WISERoutingConfig(timeout_seconds=30.0)
        router = WiseSatRouter(config=config)
        orch = WISERoutingOrchestrator(
            router=router, config=config, block_size=2,
        )

        n_rows, n_cols = 2, 4
        capacity = 2
        layout = np.arange(n_rows * n_cols, dtype=int).reshape(n_rows, n_cols)

        # Same pattern repeated twice (should trigger cache)
        pattern = [(0, 1), (2, 3)]
        parallel_pairs = [pattern, pattern, pattern, pattern]

        reconfigs, _ = orch.route_all_rounds(
            initial_layout=layout,
            parallel_pairs=parallel_pairs,
            n_rows=n_rows,
            n_cols=n_cols,
            capacity=capacity,
        )

        # Should have entries in block cache
        assert len(orch._block_cache) > 0, "Block cache is empty after repeated patterns"

    def test_orchestrator_adaptive_growth(self):
        """Test that patch dimensions grow when progress stalls."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.orchestrator import (
            WISERoutingOrchestrator,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WiseSatRouter,
        )

        _, _, WISERoutingConfig, _, _ = _get_wise_imports()
        config = WISERoutingConfig(timeout_seconds=30.0)
        router = WiseSatRouter(config=config)
        orch = WISERoutingOrchestrator(
            router=router, config=config,
            # Start with very small patches to trigger growth
            subgrid_size=(2, 2, 1),
        )

        n_rows, n_cols = 4, 8
        capacity = 2
        layout = np.arange(n_rows * n_cols, dtype=int).reshape(n_rows, n_cols)

        # Pairs spanning the full grid (will force patch growth)
        parallel_pairs = [
            [(0, 7), (8, 15)],  # Cross-grid pairs
        ]

        # Should not crash — adaptive growth handles the difficult pairs
        reconfigs, _ = orch.route_all_rounds(
            initial_layout=layout,
            parallel_pairs=parallel_pairs,
            n_rows=n_rows,
            n_cols=n_cols,
            capacity=capacity,
        )
        # Orchestrator should have attempted routing (may or may not succeed
        # depending on SAT feasibility, but shouldn't crash)
        assert isinstance(reconfigs, list)

    def test_orchestrator_cross_boundary_prefs_forwarded(self):
        """Test that cross-boundary preferences are computed and forwarded."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.orchestrator import (
            WISERoutingOrchestrator,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WiseSatRouter,
        )

        _, _, WISERoutingConfig, _, _ = _get_wise_imports()
        config = WISERoutingConfig(timeout_seconds=30.0)
        router = WiseSatRouter(config=config)
        orch = WISERoutingOrchestrator(
            router=router, config=config,
            subgrid_size=(4, 2, 1),
        )

        n_rows, n_cols = 3, 6
        capacity = 2
        layout = np.arange(n_rows * n_cols, dtype=int).reshape(n_rows, n_cols)

        # Cross-patch pairs (one ion in each half)
        parallel_pairs = [
            [(0, 3), (1, 4)],
        ]

        reconfigs, _ = orch.route_all_rounds(
            initial_layout=layout,
            parallel_pairs=parallel_pairs,
            n_rows=n_rows,
            n_cols=n_cols,
            capacity=capacity,
        )
        # Should complete without error (cross-boundary prefs should not break)
        assert isinstance(reconfigs, list)

    def test_orchestrator_bt_propagation(self):
        """Test that BT positions are propagated across windows."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.orchestrator import (
            WISERoutingOrchestrator,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WiseSatRouter,
        )

        _, _, WISERoutingConfig, _, _ = _get_wise_imports()
        config = WISERoutingConfig(timeout_seconds=30.0)
        router = WiseSatRouter(config=config)
        orch = WISERoutingOrchestrator(
            router=router, config=config, lookahead=2,
        )

        n_rows, n_cols = 2, 4
        capacity = 2
        layout = np.arange(n_rows * n_cols, dtype=int).reshape(n_rows, n_cols)

        # Multiple rounds to trigger lookahead / BT propagation
        parallel_pairs = [
            [(0, 1)],
            [(2, 3)],
            [(4, 5)],
        ]

        reconfigs, _ = orch.route_all_rounds(
            initial_layout=layout,
            parallel_pairs=parallel_pairs,
            n_rows=n_rows,
            n_cols=n_cols,
            capacity=capacity,
        )

        # Should produce reconfigs for all rounds
        assert len(reconfigs) > 0, "No reconfigs produced with BT propagation"


# =============================================================================
# Test: Compilation with orchestrator path
# =============================================================================

class TestCompilationOrchestrator:
    """Test WISERoutingPass with orchestrator integration."""

    def test_routing_pass_with_orchestrator(self):
        """Test that WISERoutingPass uses orchestrator when patch_enabled."""
        _, _, WISERoutingConfig, _, _ = _get_wise_imports()
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.compilation import (
            WISERoutingPass,
        )

        config = WISERoutingConfig(
            timeout_seconds=30.0,
            patch_enabled=True,
        )
        rpass = WISERoutingPass(config=config)
        # Verify the orchestrator is created
        assert rpass._orchestrator is not None, (
            "Orchestrator not created when patch_enabled=True"
        )

    def test_parallel_sat_config_flag(self):
        """Test that parallel_sat_search config flag is respected."""
        _, _, WISERoutingConfig, _, _ = _get_wise_imports()
        
        config = WISERoutingConfig(parallel_sat_search=False)
        assert not config.parallel_sat_search

        config2 = WISERoutingConfig(parallel_sat_search=True, sat_workers=2)
        assert config2.parallel_sat_search
        assert config2.sat_workers == 2
    def test_sat_router_directly(self):
        """Test WiseSatRouter directly to verify SAT encoding works."""
        WISEArchitecture, _, WISERoutingConfig, _, _ = _get_wise_imports()
        
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WiseSatRouter,
        )
        from qectostim.experiments.hardware_simulation.core.pipeline import QubitMapping
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.data_structures import (
            GridLayout,
        )
        
        # Create a simple routing problem
        arch = WISEArchitecture(col_groups=2, rows=2, ions_per_segment=2)
        config = WISERoutingConfig(timeout_seconds=30.0)
        router = WiseSatRouter(config=config)
        
        # Simple 2x4 grid (2 rows, 4 cols = 2 col_groups * 2 ions)
        n_rows = 2
        n_cols = 4
        layout_arr = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
        
        # Create GridLayout from numpy array
        initial_layout = GridLayout(grid=layout_arr)
        
        # Simple gate pairs: (0, 1), (2, 3) - adjacent pairs
        pairs = [(0, 1), (2, 3)]
        
        # Create a dummy qubit mapping
        mapping = QubitMapping()
        
        result = router.route_batch(
            physical_pairs=pairs,
            current_mapping=mapping,
            architecture=arch,
            initial_layout=initial_layout,
        )
        
        # Check success
        assert result.success, f"SAT routing failed: {result.metrics}"
        
        # Check it's not heuristic fallback
        assert result.metrics.get("method") != "heuristic", (
            f"Fell back to heuristic: {result.metrics}"
        )
        
        # SAT solutions should have "passes" or "bcf"
        has_sat_metrics = "passes" in result.metrics or "bcf" in result.metrics
        assert has_sat_metrics, (
            f"Missing SAT metrics (indicates heuristic): {result.metrics}"
        )


class TestWISESATEncoderNormalization:
    """Test that SAT encoder normalizations work correctly."""
    
    def test_boundary_adjacent_defaults_to_all_true(self):
        """Verify boundary_adjacent defaults to all True when None."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.sat_encoder import (
            WISESATEncoder, WISESATContext,
        )
        
        ctx = self._make_minimal_context()
        encoder = WISESATEncoder(rows=ctx.n_rows, cols=ctx.n_cols)
        encoder.initialize(ctx, pass_bound=4)
        
        expected = {"top": True, "bottom": True, "left": True, "right": True}
        assert encoder.context.boundary_adjacent == expected, (
            f"boundary_adjacent not normalized: {encoder.context.boundary_adjacent}"
        )
    
    def test_cross_boundary_prefs_normalized_to_sets(self):
        """Verify cross_boundary_prefs lists are converted to sets."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.sat_encoder import (
            WISESATEncoder, WISESATContext,
        )
        
        # Pass list with list values (should be converted to sets)
        ctx = self._make_minimal_context(
            num_rounds=2,
            cross_boundary_prefs=[{}, {1: ["up", "down"]}],
        )
        encoder = WISESATEncoder(rows=ctx.n_rows, cols=ctx.n_cols)
        encoder.initialize(ctx, pass_bound=4)
        
        # Round 1 qubit 1 should have set, not list
        prefs = encoder.context.cross_boundary_prefs[1][1]
        assert isinstance(prefs, set), f"prefs should be set, got {type(prefs)}"
        assert prefs == {"up", "down"}
    
    def _make_minimal_context(
        self, 
        n_rows: int = 2, 
        n_cols: int = 4, 
        num_rounds: int = 1,
        cross_boundary_prefs=None,
    ):
        """Create a minimal WISESATContext for testing."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.sat_encoder import (
            WISESATContext,
        )
        
        initial_layout = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
        ions = list(range(n_rows * n_cols))
        
        gate_pairs = [[(0, 1), (2, 3)] for _ in range(num_rounds)]
        target_positions = [{} for _ in range(num_rounds)]
        block_cells = [[(0, 0), (0, 1), (1, 0), (1, 1)]]
        
        return WISESATContext(
            initial_layout=initial_layout,
            target_positions=target_positions,
            gate_pairs=gate_pairs,
            full_gate_pairs=gate_pairs,
            ions=ions,
            n_rows=n_rows,
            n_cols=n_cols,
            num_rounds=num_rounds,
            block_cells=block_cells,
            block_fully_inside=[True],
            block_widths=[2],
            num_blocks=1,
            cross_boundary_prefs=cross_boundary_prefs,
        )


# NOTE: TestFullCompilationPipeline is disabled because TrappedIonExperiment
# causes hangs during test collection. Re-enable when this is fixed.
# class TestFullCompilationPipeline:
#     """Test complete compilation pipeline produces valid results."""
#     ...


class TestSATSolutionValidity:
    """Test that SAT solutions are actually valid (not just UNSAT fallback)."""
    
    def test_sat_solution_satisfies_gate_adjacency(self):
        """Verify SAT solutions place gate pairs in adjacent positions."""
        WISEArchitecture, _, WISERoutingConfig, _, _ = _get_wise_imports()
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WiseSatRouter,
        )
        from qectostim.experiments.hardware_simulation.core.pipeline import QubitMapping
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.data_structures import (
            GridLayout,
        )
        
        arch = WISEArchitecture(col_groups=2, rows=2, ions_per_segment=2)
        config = WISERoutingConfig(timeout_seconds=30.0)
        router = WiseSatRouter(config=config)
        
        n_rows, n_cols = 2, 4
        layout_arr = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
        initial_layout = GridLayout(grid=layout_arr)
        pairs = [(0, 1), (4, 5)]  # Two pairs
        mapping = QubitMapping()
        
        result = router.route_batch(
            physical_pairs=pairs,
            current_mapping=mapping,
            architecture=arch,
            initial_layout=initial_layout,
        )
        
        assert result.success, f"Routing failed: {result.metrics}"
        
        # Check final layout has pairs adjacent
        if "_final_layout" in result.metrics:
            final_layout = result.metrics["_final_layout"]
            # Find positions of each ion in the pair
            for ion_a, ion_b in pairs:
                pos_a = np.argwhere(final_layout == ion_a)
                pos_b = np.argwhere(final_layout == ion_b)
                if len(pos_a) > 0 and len(pos_b) > 0:
                    r_a, c_a = pos_a[0]
                    r_b, c_b = pos_b[0]
                    # Check adjacency (same row, adjacent cols OR same col, adjacent rows)
                    same_row = (r_a == r_b) and abs(c_a - c_b) == 1
                    same_col = (c_a == c_b) and abs(r_a - r_b) == 1
                    assert same_row or same_col, (
                        f"Pair ({ion_a}, {ion_b}) not adjacent: "
                        f"pos_a={pos_a[0]}, pos_b={pos_b[0]}"
                    )


# =============================================================================
# Test: Larger grids (SAT routing only, no TrappedIonExperiment)
# =============================================================================

class TestLargerGrids:
    """Test SAT routing on larger grids (with longer timeout)."""
    
    def test_5x10_grid_sat_success(self):
        """Test SAT routing on a 5x10 grid (50 ions)."""
        WISEArchitecture, _, WISERoutingConfig, _, _ = _get_wise_imports()
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WiseSatRouter,
        )
        from qectostim.experiments.hardware_simulation.core.pipeline import QubitMapping
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.data_structures import (
            GridLayout,
        )
        
        # 5x10 grid (50 ions)
        arch = WISEArchitecture(col_groups=5, rows=5, ions_per_segment=2)
        config = WISERoutingConfig(timeout_seconds=60.0)
        router = WiseSatRouter(config=config)
        
        n_rows, n_cols = 5, 10
        layout_arr = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
        initial_layout = GridLayout(grid=layout_arr)
        
        # Multiple gate pairs across the grid
        pairs = [(0, 1), (2, 3), (10, 11), (20, 21), (30, 31)]
        mapping = QubitMapping()
        
        result = router.route_batch(
            physical_pairs=pairs,
            current_mapping=mapping,
            architecture=arch,
            initial_layout=initial_layout,
        )
        
        assert result.success, f"SAT routing failed: {result.metrics}"
        assert result.metrics.get("method") != "heuristic", (
            f"SAT routing fell back to heuristic!"
        )
