# src/qectostim/experiments/hardware_simulation/tests/test_augmented_networked.py
"""
Tests for the Augmented Grid and Networked Grid architectures,
clustering module, and their compilers.
"""
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestAugmentedGridArchitecture:
    """Tests for AugmentedGridArchitecture construction and topology."""

    def test_basic_construction(self):
        """Build a small augmented grid and check node counts."""
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            AugmentedGridArchitecture,
            ManipulationTrap,
            Junction,
        )
        arch = AugmentedGridArchitecture(rows=2, cols=3, ions_per_trap=2)
        graph = arch.qccd_graph

        traps = [n for n in graph.nodes.values() if isinstance(n, ManipulationTrap)]
        junctions = [n for n in graph.nodes.values() if isinstance(n, Junction)]

        # Main traps: 2×3=6, diagonal traps: (2-1)×(3-1)=2
        assert len(traps) == 6 + 2
        # Junctions: between vertical pairs in even columns (3 cols × 1 gap = 3)
        assert len(junctions) == 3

    def test_single_row(self):
        """Single-row augmented grid should have no junctions."""
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            AugmentedGridArchitecture,
            Junction,
        )
        arch = AugmentedGridArchitecture(rows=1, cols=4, ions_per_trap=2)
        junctions = [
            n for n in arch.qccd_graph.nodes.values() if isinstance(n, Junction)
        ]
        assert len(junctions) == 0

    def test_ions_populated(self):
        """All traps should have the correct number of ions."""
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            AugmentedGridArchitecture,
            ManipulationTrap,
        )
        arch = AugmentedGridArchitecture(rows=2, cols=2, ions_per_trap=3)
        for node in arch.qccd_graph.nodes.values():
            if isinstance(node, ManipulationTrap):
                assert len(node.ions) == 3

    def test_routing_table_exists(self):
        """Routing table should be computed on construction."""
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            AugmentedGridArchitecture,
        )
        arch = AugmentedGridArchitecture(rows=2, cols=3, ions_per_trap=2)
        rt = arch.qccd_graph._routing_table
        assert len(rt) > 0

    def test_repr(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            AugmentedGridArchitecture,
        )
        arch = AugmentedGridArchitecture(rows=2, cols=3, ions_per_trap=2)
        r = repr(arch)
        assert "AugmentedGrid" in r


class TestNetworkedGridArchitecture:
    """Tests for NetworkedGridArchitecture construction and topology."""

    def test_basic_construction(self):
        """Build a small networked grid and check counts."""
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            NetworkedGridArchitecture,
            ManipulationTrap,
            Junction,
        )
        arch = NetworkedGridArchitecture(num_traps=4, ions_per_trap=2)
        graph = arch.qccd_graph

        traps = [n for n in graph.nodes.values() if isinstance(n, ManipulationTrap)]
        junctions = [n for n in graph.nodes.values() if isinstance(n, Junction)]

        assert len(traps) == 4
        # One junction per trap
        assert len(junctions) == 4

    def test_all_to_all_connectivity(self):
        """Every pair of traps should have a routing path."""
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            NetworkedGridArchitecture,
            ManipulationTrap,
        )
        arch = NetworkedGridArchitecture(num_traps=3, ions_per_trap=2)
        trap_ids = [
            n.idx
            for n in arch.qccd_graph.nodes.values()
            if isinstance(n, ManipulationTrap)
        ]
        for src in trap_ids:
            for dst in trap_ids:
                if src != dst:
                    path = arch.qccd_graph.get_routing_path(src, dst)
                    assert len(path) > 0, f"No path from {src} to {dst}"

    def test_single_trap(self):
        """Degenerate case: single trap."""
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            NetworkedGridArchitecture,
        )
        arch = NetworkedGridArchitecture(num_traps=1, ions_per_trap=3)
        assert len(list(arch.qccd_graph.nodes.values())) == 2  # 1 trap + 1 junction

    def test_repr(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            NetworkedGridArchitecture,
        )
        arch = NetworkedGridArchitecture(num_traps=5, ions_per_trap=2)
        assert "Networked" in repr(arch)


# ---------------------------------------------------------------------------
# Clustering tests
# ---------------------------------------------------------------------------

class TestClustering:
    """Tests for the clustering module."""

    def _make_ions(self, n, offset=0):
        """Helper to create n ions with grid-like positions."""
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import Ion
        ions = []
        cols = max(1, int(np.sqrt(n)))
        for i in range(n):
            r, c = divmod(i, cols)
            ions.append(
                Ion(idx=offset + i, position=(float(c), float(r)), label="Q")
            )
        return ions

    def test_regular_partition_basic(self):
        """Partition 6 ions into clusters of capacity 3."""
        from qectostim.experiments.hardware_simulation.trapped_ion.clustering import (
            regular_partition,
        )
        m_ions = self._make_ions(3, offset=0)
        d_ions = self._make_ions(3, offset=3)
        clusters = regular_partition(m_ions, d_ions, trap_capacity=4)
        total = sum(len(c[0]) for c in clusters)
        assert total == 6
        for ions, _ in clusters:
            assert len(ions) <= 3  # eff_capacity = 4-1 = 3

    def test_regular_partition_with_max_clusters(self):
        """Test merging when max_clusters is specified."""
        from qectostim.experiments.hardware_simulation.trapped_ion.clustering import (
            regular_partition,
        )
        m_ions = self._make_ions(4, offset=0)
        d_ions = self._make_ions(6, offset=4)
        clusters = regular_partition(
            m_ions, d_ions, trap_capacity=6, max_clusters=3,
        )
        assert len(clusters) <= 3
        total = sum(len(c[0]) for c in clusters)
        assert total == 10

    def test_merge_clusters_to_limit(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.clustering import (
            merge_clusters_to_limit,
        )
        ions = self._make_ions(8)
        clusters = [
            (ions[0:2], np.array([0.0, 0.0])),
            (ions[2:4], np.array([1.0, 0.0])),
            (ions[4:6], np.array([0.0, 1.0])),
            (ions[6:8], np.array([1.0, 1.0])),
        ]
        merged = merge_clusters_to_limit(clusters, max_clusters=2, capacity=4)
        assert len(merged) == 2
        total = sum(len(c[0]) for c in merged)
        assert total == 8

    def test_merge_raises_on_impossible(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.clustering import (
            merge_clusters_to_limit,
        )
        ions = self._make_ions(6)
        clusters = [
            (ions[0:3], np.array([0.0, 0.0])),
            (ions[3:6], np.array([1.0, 0.0])),
        ]
        with pytest.raises(RuntimeError):
            merge_clusters_to_limit(clusters, max_clusters=1, capacity=4)

    def test_arrange_clusters(self):
        """Test cluster arrangement on a small grid."""
        from qectostim.experiments.hardware_simulation.trapped_ion.clustering import (
            arrange_clusters,
        )
        ions = self._make_ions(4)
        clusters = [
            (ions[0:2], np.array([0.0, 0.0])),
            (ions[2:4], np.array([1.0, 1.0])),
        ]
        grid_pos = [(0, 0), (2, 0), (0, 2), (2, 2)]
        result = arrange_clusters(clusters, grid_pos)
        assert len(result) == 2
        for pos in result:
            assert pos in grid_pos

    def test_hill_climb_on_arrange(self):
        """Hill-climbing should return valid positions."""
        from qectostim.experiments.hardware_simulation.trapped_ion.clustering import (
            hill_climb_on_arrange_clusters,
        )
        ions = self._make_ions(6)
        clusters = [
            (ions[0:2], np.array([0.0, 0.0])),
            (ions[2:4], np.array([1.0, 0.0])),
            (ions[4:6], np.array([0.5, 1.0])),
        ]
        grid_pos = [(0, 0), (2, 0), (0, 2), (2, 2), (1, 1)]
        result = hill_climb_on_arrange_clusters(clusters, grid_pos)
        assert len(result) == 3
        for pos in result:
            assert pos in grid_pos


# ---------------------------------------------------------------------------
# Compiler tests
# ---------------------------------------------------------------------------

class TestAugmentedGridCompiler:
    """Tests for the AugmentedGridCompiler."""

    def test_import(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compilers import (
            AugmentedGridCompiler,
        )
        assert AugmentedGridCompiler is not None

    def test_instantiation(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            AugmentedGridArchitecture,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.compilers import (
            AugmentedGridCompiler,
        )
        arch = AugmentedGridArchitecture(rows=2, cols=3, ions_per_trap=3)
        compiler = AugmentedGridCompiler(arch)
        assert compiler is not None


class TestNetworkedGridCompiler:
    """Tests for the NetworkedGridCompiler."""

    def test_import(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compilers import (
            NetworkedGridCompiler,
        )
        assert NetworkedGridCompiler is not None

    def test_instantiation(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            NetworkedGridArchitecture,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.compilers import (
            NetworkedGridCompiler,
        )
        arch = NetworkedGridArchitecture(num_traps=3, ions_per_trap=3)
        compiler = NetworkedGridCompiler(arch)
        assert compiler is not None


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

class TestNewImports:
    """Verify all new symbols are importable from the package."""

    def test_architecture_imports(self):
        from qectostim.experiments.hardware_simulation.trapped_ion import (
            AugmentedGridArchitecture,
            NetworkedGridArchitecture,
        )
        assert AugmentedGridArchitecture is not None
        assert NetworkedGridArchitecture is not None

    def test_compiler_imports(self):
        from qectostim.experiments.hardware_simulation.trapped_ion import (
            AugmentedGridCompiler,
            NetworkedGridCompiler,
        )
        assert AugmentedGridCompiler is not None
        assert NetworkedGridCompiler is not None

    def test_clustering_imports(self):
        from qectostim.experiments.hardware_simulation.trapped_ion import (
            regular_partition,
            arrange_clusters,
            hill_climb_on_arrange_clusters,
            merge_clusters_to_limit,
        )
        assert regular_partition is not None
        assert arrange_clusters is not None
        assert hill_climb_on_arrange_clusters is not None
        assert merge_clusters_to_limit is not None
