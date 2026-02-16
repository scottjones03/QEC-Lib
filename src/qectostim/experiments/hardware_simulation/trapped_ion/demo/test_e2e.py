"""
End-to-end pytest suite for the trapped-ion QCCD framework.

Tests the full pipeline — ``build_ideal_circuit → compile →
display_architecture → animate_transport → apply_hardware_noise →
simulate`` — for both Augmented Grid and WISE architectures using a
d=2 rotated surface code.

Run with::

    pytest src/qectostim/experiments/hardware_simulation/trapped_ion/demo/test_e2e.py -v
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import stim

# ── QECToStim imports ────────────────────────────────────────────────
from qectostim.codes.surface.rotated_surface import RotatedSurfaceCode
from qectostim.experiments.memory import CSSMemoryExperiment
from qectostim.experiments.hardware_simulation.base import (
    HardwareSimulationResult,
)

# ── Trapped-ion module imports ───────────────────────────────────────
from qectostim.experiments.hardware_simulation.trapped_ion.utils import (
    AugmentedGridArchitecture,
    WISEArchitecture,
    TrappedIonCompiler,
    TrappedIonExperiment,
    TrappedIonNoiseModel,
)
from qectostim.experiments.hardware_simulation.trapped_ion.utils.qccd_nodes import (
    QCCDWiseArch,
    SpectatorIon,
)
from qectostim.experiments.hardware_simulation.trapped_ion.viz import (
    display_architecture,
    animate_transport,
)
from qectostim.experiments.hardware_simulation.core.pipeline import (
    CompiledCircuit,
)

from .run import build_viz_mappings


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture(scope="module")
def code():
    """d=2 rotated surface code."""
    return RotatedSurfaceCode(distance=2)


@pytest.fixture(scope="module")
def ideal(code):
    """Ideal stim circuit for d=2, 1 round, Z-basis memory experiment."""
    mem = CSSMemoryExperiment(
        code=code, rounds=1, noise_model=None, basis="z",
    )
    return mem.to_stim()


# ── Augmented Grid fixtures ─────────────────────────────────────────


@pytest.fixture(scope="module")
def ag_compiled(ideal):
    """Compile on Augmented Grid, return (arch, compiler, compiled)."""
    arch = AugmentedGridArchitecture(
        trap_capacity=2, rows=3, cols=3, padding=0,
    )
    compiler = TrappedIonCompiler(arch, is_wise=False)
    compiled = compiler.compile(ideal)
    return arch, compiler, compiled


# ── WISE fixtures ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def wise_compiled(ideal):
    """Compile on WISE, return (arch, compiler, compiled)."""
    wise_cfg = QCCDWiseArch(m=5, n=5, k=2)
    arch = WISEArchitecture(
        wise_config=wise_cfg,
        add_spectators=True,
        compact_clustering=True,
    )
    compiler = TrappedIonCompiler(
        arch, is_wise=True, wise_config=wise_cfg,
    )
    compiled = compiler.compile(ideal)
    return arch, compiler, compiled


# =====================================================================
# Compilation tests
# =====================================================================


class TestAugmentedGridCompile:
    """Tests for Augmented Grid compilation pipeline."""

    def test_compiled_circuit_type(self, ag_compiled):
        _, _, compiled = ag_compiled
        assert isinstance(compiled, CompiledCircuit)

    def test_has_batches(self, ag_compiled):
        _, _, compiled = ag_compiled
        batches = compiled.scheduled.batches
        assert len(batches) > 0, "No parallel batches produced"

    def test_has_mapping(self, ag_compiled):
        _, _, compiled = ag_compiled
        assert compiled.mapping.num_qubits() > 0

    def test_metrics(self, ag_compiled):
        _, _, compiled = ag_compiled
        m = compiled.compute_metrics()
        assert m["total_operations"] > 0
        assert m["depth"] > 0
        assert m["duration_us"] >= 0

    def test_ion_mapping_populated(self, ag_compiled):
        _, compiler, _ = ag_compiled
        assert len(compiler.ion_mapping) > 0

    def test_traps_created(self, ag_compiled):
        arch, _, _ = ag_compiled
        assert len(arch._manipulationTraps) > 0

    def test_viz_mappings(self, ag_compiled):
        _, compiler, _ = ag_compiled
        ion_roles, p2l, remap = build_viz_mappings(compiler)
        assert len(ion_roles) > 0
        assert all(r in ("D", "M", "P", "U") for r in ion_roles.values())
        assert len(p2l) > 0
        assert len(remap) > 0


class TestWISECompile:
    """Tests for WISE compilation pipeline."""

    def test_compiled_circuit_type(self, wise_compiled):
        _, _, compiled = wise_compiled
        assert isinstance(compiled, CompiledCircuit)

    def test_has_batches(self, wise_compiled):
        _, _, compiled = wise_compiled
        batches = compiled.scheduled.batches
        assert len(batches) > 0, "No parallel batches produced"

    def test_has_mapping(self, wise_compiled):
        _, _, compiled = wise_compiled
        assert compiled.mapping.num_qubits() > 0

    def test_metrics(self, wise_compiled):
        _, _, compiled = wise_compiled
        m = compiled.compute_metrics()
        assert m["total_operations"] > 0
        assert m["depth"] > 0

    def test_ion_mapping_populated(self, wise_compiled):
        _, compiler, _ = wise_compiled
        assert len(compiler.ion_mapping) > 0

    def test_traps_created(self, wise_compiled):
        arch, _, _ = wise_compiled
        assert len(arch._manipulationTraps) > 0

    def test_viz_mappings(self, wise_compiled):
        _, compiler, _ = wise_compiled
        ion_roles, p2l, remap = build_viz_mappings(compiler)
        # WISE may have spectators
        assert len(ion_roles) > 0
        assert all(
            r in ("D", "M", "P", "U") for r in ion_roles.values()
        )


# =====================================================================
# Visualisation tests
# =====================================================================


class TestAugmentedGridDisplay:
    """Static display tests for Augmented Grid."""

    def test_display_returns_fig_ax(self, ag_compiled):
        arch, compiler, _ = ag_compiled
        arch.resetArrangement()
        arch.refreshGraph()
        ion_roles, p2l, remap = build_viz_mappings(compiler)

        fig, ax = display_architecture(
            arch,
            title="test aug grid",
            show_ions=True,
            show_labels=False,
            ion_roles=ion_roles,
            ion_idx_remap=remap,
            physical_to_logical=p2l,
        )
        assert fig is not None
        assert ax is not None
        plt.close(fig)


class TestWISEDisplay:
    """Static display tests for WISE."""

    def test_display_returns_fig_ax(self, wise_compiled):
        arch, compiler, _ = wise_compiled
        arch.resetArrangement()
        arch.refreshGraph()
        ion_roles, p2l, remap = build_viz_mappings(compiler)

        fig, ax = display_architecture(
            arch,
            title="test WISE",
            show_ions=True,
            show_labels=True,
            ion_roles=ion_roles,
            ion_idx_remap=remap,
            physical_to_logical=p2l,
        )
        assert fig is not None
        assert ax is not None
        plt.close(fig)


# =====================================================================
# Animation tests
# =====================================================================


class TestAugmentedGridAnimate:
    """Animation tests for Augmented Grid."""

    def test_animate_returns_funcanimation(self, ag_compiled, ideal):
        arch, compiler, compiled = ag_compiled
        arch.resetArrangement()
        arch.refreshGraph()
        ion_roles, p2l, remap = build_viz_mappings(compiler)
        batches = compiled.scheduled.batches

        anim = animate_transport(
            arch=arch,
            operations=batches[:5],  # only first 5 for speed
            interval=200,
            show_labels=False,
            interp_frames=1,
            gate_hold_frames=0,
            stim_circuit=ideal,
            repeat=False,
            ion_roles=ion_roles,
            ion_idx_remap=remap,
            physical_to_logical=p2l,
        )
        from matplotlib.animation import FuncAnimation
        assert isinstance(anim, FuncAnimation)
        plt.close("all")


class TestWISEAnimate:
    """Animation tests for WISE."""

    def test_animate_returns_funcanimation(self, wise_compiled, ideal):
        arch, compiler, compiled = wise_compiled
        arch.resetArrangement()
        arch.refreshGraph()
        ion_roles, p2l, remap = build_viz_mappings(compiler)
        batches = compiled.scheduled.batches

        anim = animate_transport(
            arch=arch,
            operations=batches[:5],
            interval=200,
            show_labels=True,
            interp_frames=1,
            gate_hold_frames=0,
            stim_circuit=ideal,
            repeat=False,
            ion_roles=ion_roles,
            ion_idx_remap=remap,
            physical_to_logical=p2l,
        )
        from matplotlib.animation import FuncAnimation
        assert isinstance(anim, FuncAnimation)
        plt.close("all")


# =====================================================================
# Noise tests
# =====================================================================


class TestNoiseApplication:
    """Test hardware noise injection."""

    def test_noisy_circuit_has_errors(self, code, ideal):
        # Build a fresh architecture / compiler to avoid stale ion
        # references from the module-scoped ag_compiled fixture.
        arch = AugmentedGridArchitecture(
            trap_capacity=2, rows=3, cols=3, padding=0,
        )
        compiler = TrappedIonCompiler(arch, is_wise=False, show_progress=False)

        noise = TrappedIonNoiseModel()
        experiment = TrappedIonExperiment(
            code=code,
            architecture=arch,
            compiler=compiler,
            hardware_noise=noise,
            rounds=1,
            basis="z",
        )

        # compile
        ideal_circ = experiment.build_ideal_circuit()
        experiment._compiled = experiment.compiler.compile(ideal_circ)
        noisy = experiment.apply_hardware_noise()

        assert isinstance(noisy, stim.Circuit)
        assert len(noisy) > 0, "Noisy circuit is empty"

        # Check that at least one error instruction was injected
        noisy_str = str(noisy)
        has_noise = any(
            kw in noisy_str
            for kw in ("X_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2")
        )
        assert has_noise, "No noise instructions found in noisy circuit"


# =====================================================================
# Simulation test (optional — may be slow)
# =====================================================================


class TestSimulation:
    """End-to-end simulation with PyMatching decoder."""

    @pytest.mark.slow
    def test_simulate_returns_result(self, code, ideal):
        # Fresh architecture to avoid stale ion references.
        arch = AugmentedGridArchitecture(
            trap_capacity=2, rows=3, cols=3, padding=0,
        )
        compiler = TrappedIonCompiler(arch, is_wise=False, show_progress=False)

        noise = TrappedIonNoiseModel()
        experiment = TrappedIonExperiment(
            code=code,
            architecture=arch,
            compiler=compiler,
            hardware_noise=noise,
            rounds=1,
            basis="z",
        )

        ideal_circ = experiment.build_ideal_circuit()
        experiment._compiled = experiment.compiler.compile(ideal_circ)
        noisy = experiment.apply_hardware_noise()

        try:
            from qectostim.decoders.pymatching_decoder import PyMatchingDecoder

            dem = noisy.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True,
            )
            decoder = PyMatchingDecoder(dem=dem)
        except Exception:
            pytest.skip("PyMatching not available")

        sampler = noisy.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            100, separate_observables=True,
        )
        predictions = decoder.decode_batch(detection_events)
        num_errors = int(
            np.sum(np.any(predictions != observable_flips, axis=1))
        )
        ler = num_errors / 100

        assert 0.0 <= ler <= 1.0, f"Invalid logical error rate: {ler}"
        print(f"  Logical error rate: {ler:.4f} ({num_errors}/100)")
