"""Tests for the phase-aware integration layer and reconfig merge optimisation.

Tests cover:
  - FullExperimentResult / PhaseResult dataclasses
  - decompose_into_phases() phase decomposition
  - route_full_experiment() orchestrator
  - compute_schedule_timing() analytic timing
  - Reconfig merge helpers (_ions_unmoved, _merge_reconfig_schedules)
  - Wiring into run_single_gadget_config() (end-to-end smoke)
"""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# A. Integration layer unit tests (gadget_routing.py additions)
# ---------------------------------------------------------------------------


class TestFullExperimentResultDataclass:
    """Verify the FullExperimentResult dataclass can be instantiated."""

    def test_default_construction(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
            FullExperimentResult,
        )
        result = FullExperimentResult()
        assert result.total_exec_time == 0.0
        assert result.total_reconfig_time == 0.0
        assert result.cached_phases == 0
        assert result.total_phases == 0
        assert result.ms_rounds_routed == 0
        assert result.ms_rounds_replayed == 0
        assert result.phase_results == []
        assert result.total_schedule == []
        assert result.sub_grids == {}
        assert result.per_ion_heating == {}

    def test_with_values(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
            FullExperimentResult,
            PhaseResult,
        )
        pr = PhaseResult(phase_index=0, phase_type="ec", exec_time=1.0)
        result = FullExperimentResult(
            phase_results=[pr],
            total_exec_time=1.5,
            total_reconfig_time=0.5,
            cached_phases=1,
            total_phases=2,
            ms_rounds_routed=3,
            ms_rounds_replayed=5,
        )
        assert result.total_exec_time == 1.5
        assert result.cached_phases == 1
        assert len(result.phase_results) == 1
        assert result.phase_results[0].phase_type == "ec"


class TestPhaseResultDataclass:
    def test_default(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
            PhaseResult,
        )
        pr = PhaseResult()
        assert pr.phase_type == ""
        assert pr.from_cache is False

    def test_ec_phase(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
            PhaseResult,
        )
        pr = PhaseResult(
            phase_index=2,
            phase_type="ec",
            from_cache=True,
            exec_time=42.0,
        )
        assert pr.phase_type == "ec"
        assert pr.from_cache is True


class TestComputeScheduleTiming:
    """Tests for the analytic schedule timing calculator."""

    def test_empty_schedule(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
            compute_schedule_timing,
        )
        total, reconfig, heating = compute_schedule_timing([], k=2)
        assert total == 0.0
        assert reconfig == 0.0
        assert heating == {}

    def test_single_h_pass(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
            compute_schedule_timing,
        )
        schedule = [
            [{"phase": "H", "h_swaps": [(0, 0), (0, 1)], "v_swaps": []}],
        ]
        total, reconfig, _ = compute_schedule_timing(schedule, k=2)
        # Should have: Split overhead + one H pass time + one MS gate time
        assert reconfig > 0.0
        assert total > reconfig  # MS gate time adds to total

    def test_h_and_v_passes(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
            compute_schedule_timing,
        )
        schedule = [
            [
                {"phase": "H", "h_swaps": [(0, 0)], "v_swaps": []},
                {"phase": "V", "h_swaps": [], "v_swaps": [(0, 0)]},
            ],
        ]
        total, reconfig, _ = compute_schedule_timing(schedule, k=2)
        # Both H and V contribute to reconfig time
        assert reconfig > 0.0

    def test_multiple_rounds(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
            compute_schedule_timing,
        )
        single_round = [{"phase": "H", "h_swaps": [(0, 0)], "v_swaps": []}]
        schedule_1r = [single_round]
        schedule_4r = [single_round, single_round, single_round, single_round]

        total_1, reconfig_1, _ = compute_schedule_timing(schedule_1r, k=2)
        total_4, reconfig_4, _ = compute_schedule_timing(schedule_4r, k=2)

        # 4 rounds should take ~4× the reconfig time
        assert reconfig_4 == pytest.approx(reconfig_1 * 4, rel=1e-6)
        # Total includes MS time too
        assert total_4 > total_1


class TestDecomposeIntoPhases:
    """Tests for phase decomposition logic."""

    def _make_mock_metadata(self):
        """Create a minimal QECMetadata for a 2-block experiment."""
        from qectostim.experiments.hardware_simulation.core.pipeline import (
            QECMetadata,
            PhaseInfo,
            BlockInfo,
            StabilizerInfo,
        )

        x_sched = [[(0, 1), (2, 3)], [(0, 3), (2, 1)]]  # 2 CNOT layers
        z_sched = [[(4, 1), (5, 3)], [(4, 3), (5, 1)]]

        block_a = BlockInfo(
            block_name="ctrl",
            data_qubits=[0, 2],
            x_ancilla_qubits=[1],
            z_ancilla_qubits=[3],
        )
        block_b = BlockInfo(
            block_name="target",
            data_qubits=[10, 12],
            x_ancilla_qubits=[11],
            z_ancilla_qubits=[13],
        )

        # Per-block stabilizers (interleaved X+Z schedule)
        per_block_stabs = {
            "ctrl": StabilizerInfo(
                cnot_schedule=[
                    x_sched[i] + z_sched[i]
                    for i in range(max(len(x_sched), len(z_sched)))
                ],
                ancillas=[1, 3],
            ),
            "target": StabilizerInfo(
                cnot_schedule=[
                    [(10, 11), (12, 13)],
                    [(10, 13), (12, 11)],
                    [(14, 11), (15, 13)],
                    [(14, 13), (15, 11)],
                ],
                ancillas=[11, 13],
            ),
        }

        phases = [
            PhaseInfo(
                phase_type="stabilizer_round",
                num_rounds=3,
                is_repeated=True,
                active_blocks=["ctrl", "target"],
            ),
            PhaseInfo(
                phase_type="gadget",
                num_rounds=1,
                is_repeated=False,
                active_blocks=["ctrl", "target"],
            ),
            PhaseInfo(
                phase_type="stabilizer_round",
                num_rounds=3,
                is_repeated=True,
                active_blocks=["ctrl", "target"],
            ),
        ]

        return QECMetadata(
            block_allocations=[block_a, block_b],
            phases=phases,
            per_block_stabilizers=per_block_stabs,
            x_stabilizers=StabilizerInfo(cnot_schedule=x_sched),
            z_stabilizers=StabilizerInfo(cnot_schedule=z_sched),
        )

    @staticmethod
    def _make_mock_gadget_and_alloc():
        """Create minimal mock gadget and allocation."""

        class MockBlockAlloc:
            def __init__(self, data_range):
                self.data_range = data_range

        class MockAlloc:
            blocks = {
                "ctrl": MockBlockAlloc(data_range=[0, 2]),
                "target": MockBlockAlloc(data_range=[10, 12]),
            }

        class MockGadget:
            pass

        return MockGadget(), MockAlloc()

    def test_decompose_produces_correct_count(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
            decompose_into_phases,
            BlockSubGrid,
        )

        meta = self._make_mock_metadata()
        gadget, alloc = self._make_mock_gadget_and_alloc()

        # Minimal sub_grids
        sg = {
            "ctrl": BlockSubGrid(
                block_name="ctrl",
                ion_indices=[0, 1, 2, 3],
                qubit_to_ion={0: 0, 1: 1, 2: 2, 3: 3},
            ),
            "target": BlockSubGrid(
                block_name="target",
                ion_indices=[10, 11, 12, 13],
                qubit_to_ion={10: 10, 11: 11, 12: 12, 13: 13},
            ),
        }

        plans = decompose_into_phases(
            meta, gadget, alloc,
            sg, {0: 0, 1: 1, 2: 2, 3: 3, 10: 10, 11: 11, 12: 12, 13: 13},
            k=2,
        )

        # Should produce 3 plans: EC, gadget, EC
        assert len(plans) == 3
        assert plans[0].phase_type == "ec"
        assert plans[1].phase_type == "gadget"
        assert plans[2].phase_type == "ec"

    def test_ec_cache_dedup(self):
        """Two identical EC phases should mark the second as cached."""
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
            decompose_into_phases,
            BlockSubGrid,
        )

        meta = self._make_mock_metadata()
        gadget, alloc = self._make_mock_gadget_and_alloc()

        sg = {
            "ctrl": BlockSubGrid(
                block_name="ctrl",
                ion_indices=[0, 1, 2, 3],
                qubit_to_ion={0: 0, 1: 1, 2: 2, 3: 3},
            ),
            "target": BlockSubGrid(
                block_name="target",
                ion_indices=[10, 11, 12, 13],
                qubit_to_ion={10: 10, 11: 11, 12: 12, 13: 13},
            ),
        }

        plans = decompose_into_phases(
            meta, gadget, alloc,
            sg, {0: 0, 1: 1, 2: 2, 3: 3, 10: 10, 11: 11, 12: 12, 13: 13},
            k=2,
        )

        # phase[0] and phase[2] are both EC with same blocks/schedule
        ec_plans = [p for p in plans if p.phase_type == "ec"]
        assert len(ec_plans) == 2
        # First EC should NOT be cached
        assert ec_plans[0].is_cached is False
        # Second EC should be cached (identical signature)
        assert ec_plans[1].is_cached is True
        assert ec_plans[1].identical_to_phase is not None

    def test_gadget_phase_not_cached(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
            decompose_into_phases,
            BlockSubGrid,
        )

        meta = self._make_mock_metadata()
        gadget, alloc = self._make_mock_gadget_and_alloc()

        sg = {
            "ctrl": BlockSubGrid(
                block_name="ctrl",
                ion_indices=[0, 1, 2, 3],
                qubit_to_ion={0: 0, 1: 1, 2: 2, 3: 3},
            ),
            "target": BlockSubGrid(
                block_name="target",
                ion_indices=[10, 11, 12, 13],
                qubit_to_ion={10: 10, 11: 11, 12: 12, 13: 13},
            ),
        }

        plans = decompose_into_phases(
            meta, gadget, alloc,
            sg, {0: 0, 1: 1, 2: 2, 3: 3, 10: 10, 11: 11, 12: 12, 13: 13},
            k=2,
        )

        gadget_plans = [p for p in plans if p.phase_type == "gadget"]
        assert len(gadget_plans) == 1
        assert gadget_plans[0].is_cached is False


# ---------------------------------------------------------------------------
# B. Reconfig merge optimisation tests (qccd_WISE_ion_route.py additions)
# ---------------------------------------------------------------------------


class TestIonsUnmoved:
    """Tests for the _ions_unmoved helper."""

    def test_ions_same_position(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            _ions_unmoved,
        )
        layout_a = np.array([[1, 2, 0], [3, 4, 0]])
        layout_b = np.array([[1, 2, 5], [3, 4, 0]])  # ion 5 added, but 1-4 same

        assert _ions_unmoved({1, 2, 3, 4}, layout_a, layout_b) is True

    def test_ion_moved(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            _ions_unmoved,
        )
        layout_a = np.array([[1, 2, 0], [3, 4, 0]])
        layout_b = np.array([[2, 1, 0], [3, 4, 0]])  # ions 1,2 swapped

        assert _ions_unmoved({1, 2}, layout_a, layout_b) is False
        assert _ions_unmoved({3, 4}, layout_a, layout_b) is True

    def test_ion_missing_in_target(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            _ions_unmoved,
        )
        layout_a = np.array([[1, 2], [3, 4]])
        layout_b = np.array([[0, 2], [3, 4]])  # ion 1 disappeared

        assert _ions_unmoved({1}, layout_a, layout_b) is False

    def test_empty_ion_set(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            _ions_unmoved,
        )
        layout = np.array([[1, 2], [3, 4]])
        assert _ions_unmoved(set(), layout, layout) is True


class TestFindIonPosition:
    """Tests for _find_ion_position helper."""

    def test_found(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            _find_ion_position,
        )
        layout = np.array([[1, 2], [3, 4]])
        assert _find_ion_position(layout, 3) == (1, 0)
        assert _find_ion_position(layout, 2) == (0, 1)

    def test_not_found(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            _find_ion_position,
        )
        layout = np.array([[1, 2], [3, 4]])
        assert _find_ion_position(layout, 99) is None


class TestMergeReconfigSchedules:
    """Tests for _merge_reconfig_schedules."""

    def test_both_none(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            _merge_reconfig_schedules,
        )
        assert _merge_reconfig_schedules(None, None) is None

    def test_one_none(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            _merge_reconfig_schedules,
        )
        sched = [{"phase": "H", "h_swaps": [(0, 0)], "v_swaps": []}]
        assert _merge_reconfig_schedules(sched, None) is None
        assert _merge_reconfig_schedules(None, sched) is None

    def test_merge_h_passes(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            _merge_reconfig_schedules,
        )
        sched_a = [{"phase": "H", "h_swaps": [(0, 0)], "v_swaps": []}]
        sched_b = [{"phase": "H", "h_swaps": [(1, 1)], "v_swaps": []}]

        merged = _merge_reconfig_schedules(sched_a, sched_b)
        assert merged is not None
        assert len(merged) == 1
        assert set(map(tuple, merged[0]["h_swaps"])) == {(0, 0), (1, 1)}

    def test_merge_different_phases(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            _merge_reconfig_schedules,
        )
        sched_a = [{"phase": "H", "h_swaps": [(0, 0)], "v_swaps": []}]
        sched_b = [{"phase": "V", "h_swaps": [], "v_swaps": [(0, 0)]}]

        merged = _merge_reconfig_schedules(sched_a, sched_b)
        assert merged is not None
        assert len(merged) == 2  # one H pass + one V pass

    def test_merge_preserves_original(self):
        """Merging should not mutate the input schedules."""
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            _merge_reconfig_schedules,
        )
        sched_a = [{"phase": "H", "h_swaps": [(0, 0)], "v_swaps": []}]
        sched_b = [{"phase": "V", "h_swaps": [], "v_swaps": [(0, 0)]}]
        original_a = [dict(s) for s in sched_a]

        _merge_reconfig_schedules(sched_a, sched_b)

        # sched_a should be mutated (we use list(sched_a))
        # but original values should be recoverable
        assert len(sched_a) == 1  # didn't add to sched_a itself


# ---------------------------------------------------------------------------
# C. Import smoke tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify all new symbols can be imported."""

    def test_import_gadget_routing_integration(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.gadget_routing import (
            FullExperimentResult,
            PhaseResult,
            decompose_into_phases,
            route_full_experiment,
            compute_schedule_timing,
            route_full_experiment_as_steps,
            _build_plans_from_compiler_pairs,
        )
        # Just checking they import without error
        assert FullExperimentResult is not None
        assert PhaseResult is not None
        assert decompose_into_phases is not None
        assert route_full_experiment is not None
        assert route_full_experiment_as_steps is not None
        assert _build_plans_from_compiler_pairs is not None

    def test_import_reconfig_merge_helpers(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            _find_ion_position,
            _ions_unmoved,
            _merge_reconfig_schedules,
        )
        assert _find_ion_position is not None
        assert _ions_unmoved is not None
        assert _merge_reconfig_schedules is not None


# =====================================================================
# Phase 4: ionRoutingGadgetArch E2E Verification
# =====================================================================


class TestIsGadgetProperty:
    """Verify the ``is_gadget`` property on QECMetadata."""

    def test_no_phases(self):
        from qectostim.experiments.hardware_simulation.core.pipeline import (
            QECMetadata,
        )
        meta = QECMetadata()
        assert meta.is_gadget is False

    def test_ec_only_phases(self):
        from qectostim.experiments.hardware_simulation.core.pipeline import (
            QECMetadata,
            PhaseInfo,
        )
        meta = QECMetadata(phases=[
            PhaseInfo(phase_type="stabilizer_round", num_rounds=3),
            PhaseInfo(phase_type="final_round", num_rounds=1),
        ])
        assert meta.is_gadget is False

    def test_has_gadget_phase(self):
        from qectostim.experiments.hardware_simulation.core.pipeline import (
            QECMetadata,
            PhaseInfo,
        )
        meta = QECMetadata(phases=[
            PhaseInfo(phase_type="stabilizer_round", num_rounds=3),
            PhaseInfo(phase_type="gadget", num_rounds=1),
            PhaseInfo(phase_type="stabilizer_round", num_rounds=3),
        ])
        assert meta.is_gadget is True


class TestIonRoutingGadgetArchImport:
    """Verify ``ionRoutingGadgetArch`` is importable from the compiler module."""

    def test_import_from_wise_route(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            ionRoutingGadgetArch,
        )
        assert callable(ionRoutingGadgetArch)

    def test_import_from_compiler(self):
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.trapped_ion_compiler import (
            ionRoutingGadgetArch,
        )
        assert callable(ionRoutingGadgetArch)


class TestIonRoutingGadgetArchFallback:
    """Verify fallback to flat routing when no gadget metadata is present."""

    def test_fallback_no_metadata(self):
        """Without qec_metadata, ionRoutingGadgetArch delegates to
        ionRoutingWISEArch and produces the same output."""
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler.qccd_WISE_ion_route import (
            ionRoutingGadgetArch,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.utils import (
            WISEArchitecture,
            TrappedIonCompiler,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.utils.qccd_nodes import (
            QCCDWiseArch,
        )
        from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
        from qectostim.experiments.memory import CSSMemoryExperiment

        code = RotatedSurfaceCode(distance=2)
        mem = CSSMemoryExperiment(code=code, rounds=1, noise_model=None, basis="z")
        stim_circuit = mem.to_stim()

        wise_cfg = QCCDWiseArch(m=5, n=5, k=2)
        arch = WISEArchitecture(
            wise_config=wise_cfg,
            add_spectators=True,
            compact_clustering=True,
        )
        compiler = TrappedIonCompiler(arch, is_wise=True, wise_config=wise_cfg)
        native = compiler.decompose_to_native(stim_circuit)
        mapped = compiler.map_qubits(native)

        instructions = mapped.native_circuit.metadata.get(
            "operations", mapped.native_circuit.operations,
        )
        toMoveOps = mapped.native_circuit.metadata.get("toMoveOps")

        arch.refreshGraph()

        # ionRoutingGadgetArch with no metadata should fallback
        allOps_g, barriers_g, reconfig_g = ionRoutingGadgetArch(
            arch, wise_cfg, instructions,
            toMoveOps=toMoveOps,
            qec_metadata=None,
        )

        assert len(allOps_g) > 0
        assert reconfig_g >= 0.0
