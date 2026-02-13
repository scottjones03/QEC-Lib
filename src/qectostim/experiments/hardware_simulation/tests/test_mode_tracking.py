# src/qectostim/experiments/hardware_simulation/tests/test_mode_tracking.py
"""
Tests for 3N normal-mode frequency tracking infrastructure.

Validates:
1. ModeStructure eigenvalue solver against known analytical results.
2. Per-mode heating distributes correctly.
3. Mode remapping on crystal split/merge conserves energy.
4. ModeSnapshot carries correct data through the pipeline.
5. Noise model callback hook works without disrupting scalar fidelity.

Physics references are given inline; see architecture.py ModeStructure
docstring for the full derivation.
"""
import numpy as np
import pytest

from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
    ModeStructure,
    ModeSnapshot,
)
from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
    ManipulationTrap,
    Ion,
)


# =====================================================================
# 1.  Eigenvalue solver — known analytical results
# =====================================================================


class TestModeStructureCompute:
    """Test ModeStructure.compute() against known results."""

    def test_single_ion_frequencies(self):
        """N=1: three modes at exactly (ω_z, ω_x, ω_y).

        A single ion in a harmonic trap has no Coulomb interaction,
        so the normal-mode frequencies are just the bare secular
        frequencies of the trap along each axis.
        """
        wz, wx, wy = 1.0e6, 5.0e6, 5.0e6
        ms = ModeStructure.compute(1, axial_freq=wz, radial_freqs=(wx, wy))

        assert ms.n_ions == 1
        assert len(ms.mode_frequencies) == 3  # 3×1 = 3

        # Axial COM = ω_z
        np.testing.assert_allclose(ms.mode_frequencies[0], wz, rtol=1e-10)
        # Radial-x = ω_x
        np.testing.assert_allclose(ms.mode_frequencies[1], wx, rtol=1e-10)
        # Radial-y = ω_y
        np.testing.assert_allclose(ms.mode_frequencies[2], wy, rtol=1e-10)

    def test_single_ion_eigenvectors(self):
        """N=1: trivial eigenvectors [1] for each axis."""
        ms = ModeStructure.compute(1)
        # Each block is (1, 1) with value 1.0
        np.testing.assert_allclose(ms.eigenvectors, np.ones((3, 1)), atol=1e-10)

    def test_single_ion_occupancies_zero(self):
        """N=1: all modes start at ground state (n̄ = 0)."""
        ms = ModeStructure.compute(1)
        np.testing.assert_array_equal(ms.occupancies, [0.0, 0.0, 0.0])

    def test_two_ion_axial_modes(self):
        """N=2: COM mode at ω_z, breathing mode at √3·ω_z.

        For two ions, the axial Hessian eigenvalues are:
        - μ₁² = 1  →  ω_COM = ω_z           (in-phase, centre-of-mass)
        - μ₂² = 3  →  ω_breath = √3 · ω_z   (out-of-phase, breathing)

        This is the classic textbook result.  See James, Appl. Phys. B
        66, 181 (1998), Table III (N=2).
        """
        wz = 1.0e6
        ms = ModeStructure.compute(2, axial_freq=wz, radial_freqs=(5e6, 5e6))

        assert ms.n_ions == 2
        assert len(ms.mode_frequencies) == 6  # 3×2 = 6

        # First two entries are axial modes (sorted ascending by eigh)
        axial = ms.axial_frequencies
        np.testing.assert_allclose(axial[0], wz, rtol=1e-6,
                                   err_msg="COM mode should be ω_z")
        np.testing.assert_allclose(axial[1], wz * np.sqrt(3), rtol=1e-6,
                                   err_msg="Breathing mode should be √3·ω_z")

    def test_two_ion_com_eigenvector(self):
        """N=2: COM mode eigenvector is (1/√2, 1/√2) — ions in phase."""
        ms = ModeStructure.compute(2, axial_freq=1e6, radial_freqs=(5e6, 5e6))
        com_evec = ms.eigenvectors[0, :]  # First axial mode
        # Both ions participate equally (possibly with a global sign)
        np.testing.assert_allclose(
            np.abs(com_evec),
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            atol=1e-6,
        )

    def test_two_ion_breathing_eigenvector(self):
        """N=2: breathing mode eigenvector is (1/√2, −1/√2) — ions out of phase."""
        ms = ModeStructure.compute(2, axial_freq=1e6, radial_freqs=(5e6, 5e6))
        breath_evec = ms.eigenvectors[1, :]  # Second axial mode
        np.testing.assert_allclose(
            np.abs(breath_evec),
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            atol=1e-6,
        )
        # The two components must have opposite signs
        assert breath_evec[0] * breath_evec[1] < 0, \
            "Breathing mode ions should be out of phase"

    def test_three_ion_com_mode(self):
        """N=3: COM mode still at ω_z (centre-of-mass always at bare freq)."""
        wz = 2.0e6
        ms = ModeStructure.compute(3, axial_freq=wz, radial_freqs=(10e6, 10e6))
        # COM mode is the lowest axial eigenvalue (always μ²=1)
        np.testing.assert_allclose(ms.axial_frequencies[0], wz, rtol=1e-5)

    def test_three_ion_total_modes(self):
        """N=3: should have 9 = 3×3 modes total."""
        ms = ModeStructure.compute(3)
        assert len(ms.mode_frequencies) == 9
        assert ms.eigenvectors.shape == (9, 3)
        assert len(ms.occupancies) == 9

    def test_radial_modes_above_axial(self):
        """Radial modes should be higher frequency than axial modes.

        When ω_x, ω_y >> ω_z (as required for a linear chain), all
        radial modes are at higher frequency than all axial modes.
        """
        ms = ModeStructure.compute(4, axial_freq=1e6, radial_freqs=(5e6, 5e6))
        axial_max = np.max(ms.axial_frequencies)
        radial_min = min(np.min(ms.radial_x_frequencies),
                         np.min(ms.radial_y_frequencies))
        assert radial_min > axial_max, \
            f"Radial modes ({radial_min:.0f}) should be above axial ({axial_max:.0f})"

    def test_mode_count_scales_with_n(self):
        """Verify 3N mode count for various N."""
        for n in [1, 2, 3, 5, 8]:
            ms = ModeStructure.compute(n)
            assert len(ms.mode_frequencies) == 3 * n
            assert ms.eigenvectors.shape == (3 * n, n)

    def test_invalid_n_raises(self):
        """N=0 or negative should raise ValueError."""
        with pytest.raises(ValueError):
            ModeStructure.compute(0)
        with pytest.raises(ValueError):
            ModeStructure.compute(-1)


# =====================================================================
# 2.  Equilibrium positions
# =====================================================================


class TestEquilibriumPositions:
    """Test the equilibrium position solver against known values."""

    def test_single_ion(self):
        """N=1: equilibrium at origin."""
        pos = ModeStructure._equilibrium_positions(1)
        np.testing.assert_allclose(pos, [0.0], atol=1e-12)

    def test_two_ions_symmetric(self):
        """N=2: positions at ±(1/4)^{1/3} ≈ ±0.6300.

        James (1998), Table I.
        """
        pos = ModeStructure._equilibrium_positions(2)
        expected = 0.25 ** (1.0 / 3.0)
        np.testing.assert_allclose(pos, [-expected, expected], atol=1e-10)

    def test_three_ions_symmetric(self):
        """N=3: centre ion at 0, outer ions symmetric.

        James (1998), Table I: u₁ = −1.0772, u₂ = 0, u₃ = +1.0772.
        """
        pos = ModeStructure._equilibrium_positions(3)
        assert len(pos) == 3
        # Centre ion at 0
        np.testing.assert_allclose(pos[1], 0.0, atol=1e-8)
        # Symmetry
        np.testing.assert_allclose(pos[0], -pos[2], atol=1e-8)
        # Known value
        np.testing.assert_allclose(abs(pos[2]), 1.0772, atol=1e-3)

    def test_positions_sorted(self):
        """Equilibrium positions should be sorted ascending."""
        for n in [2, 3, 5, 8]:
            pos = ModeStructure._equilibrium_positions(n)
            assert np.all(np.diff(pos) > 0), f"Positions not sorted for N={n}"


# =====================================================================
# 3.  Per-mode heating
# =====================================================================


class TestPerModeHeating:
    """Test that heat_modes distributes quanta correctly."""

    def test_single_ion_all_quanta_to_each_mode(self):
        """N=1: COM ratio is always 1, so each mode gets full heating.

        For a single ion, there's only one mode per axis, each at
        frequency (ω_z, ω_x, ω_y).  The COM frequency is ω_z.
        Axial mode gets full quanta.  Radial modes get reduced quanta
        because ω_radial > ω_COM.
        """
        ms = ModeStructure.compute(1, axial_freq=1e6, radial_freqs=(5e6, 5e6))
        ms.heat_modes(com_heating_quanta=1.0, noise_exponent=1.0)

        # Axial mode: ω_COM/ω_COM = 1, so gets full 1.0 quanta
        np.testing.assert_allclose(ms.occupancies[0], 1.0, atol=1e-10)
        # Radial modes: ω_COM/ω_radial = 1e6/5e6 = 0.2, so get 0.2 quanta
        np.testing.assert_allclose(ms.occupancies[1], 0.2, atol=1e-10)
        np.testing.assert_allclose(ms.occupancies[2], 0.2, atol=1e-10)

    def test_heating_accumulates(self):
        """Multiple heat_modes calls accumulate occupancies."""
        ms = ModeStructure.compute(1, axial_freq=1e6, radial_freqs=(5e6, 5e6))
        ms.heat_modes(1.0, noise_exponent=1.0)
        ms.heat_modes(1.0, noise_exponent=1.0)
        np.testing.assert_allclose(ms.occupancies[0], 2.0, atol=1e-10)

    def test_zero_heating(self):
        """Zero heating doesn't change occupancies."""
        ms = ModeStructure.compute(2)
        ms.heat_modes(0.0)
        np.testing.assert_array_equal(ms.occupancies, np.zeros(6))

    def test_noise_exponent_effect(self):
        """Higher noise exponent → steeper falloff for high-frequency modes.

        For α=2 (Johnson noise), Δn̄_m ∝ (ω_COM/ω_m)², so the
        breathing mode (√3·ω_z) gets 1/3 the heating of the COM.
        """
        ms = ModeStructure.compute(2, axial_freq=1e6, radial_freqs=(5e6, 5e6))
        ms.heat_modes(com_heating_quanta=3.0, noise_exponent=2.0)

        # COM mode (ω_z) gets full 3.0
        np.testing.assert_allclose(ms.occupancies[0], 3.0, atol=1e-6)
        # Breathing mode (√3·ω_z) gets 3.0 × (1/√3)² = 3.0 × 1/3 = 1.0
        np.testing.assert_allclose(ms.occupancies[1], 1.0, atol=1e-6)

    def test_scalar_nbar_matches_sum(self):
        """scalar_nbar property should be the sum of all occupancies."""
        ms = ModeStructure.compute(3, axial_freq=1e6, radial_freqs=(5e6, 5e6))
        ms.heat_modes(1.0, noise_exponent=1.0)
        expected_total = np.sum(ms.occupancies)
        np.testing.assert_allclose(ms.scalar_nbar, expected_total, atol=1e-12)


# =====================================================================
# 4.  Cooling
# =====================================================================


class TestCooling:
    """Test ground-state cooling resets mode occupancies."""

    def test_cool_to_ground(self):
        """cool_to_ground zeros all mode occupancies."""
        ms = ModeStructure.compute(3)
        ms.heat_modes(5.0, noise_exponent=1.0)
        assert ms.scalar_nbar > 0

        ms.cool_to_ground()
        np.testing.assert_array_equal(ms.occupancies, np.zeros(9))
        assert ms.scalar_nbar == 0.0


# =====================================================================
# 5.  Mode remapping (split/merge)
# =====================================================================


class TestModeRemapping:
    """Test remap_after_split for crystal-changing operations."""

    def test_remap_preserves_total_energy_approximately(self):
        """Total phonon energy should be approximately conserved.

        The overlap integral redistribution is unitary within each
        block (axial, radial-x, radial-y), so the sum of occupancies
        within each block should be exactly conserved.
        """
        ms_old = ModeStructure.compute(3, axial_freq=1e6, radial_freqs=(5e6, 5e6))
        ms_old.heat_modes(2.0, noise_exponent=1.0)
        old_total = ms_old.scalar_nbar

        # Remove the last ion (split: 3 → 2)
        ms_new = ModeStructure.remap_after_split(
            old_structure=ms_old,
            new_n_ions=2,
            kept_ion_indices=[0, 1],
        )

        new_total = ms_new.scalar_nbar
        # Energy is redistributed, not necessarily exactly conserved
        # because the overlap is not exactly unitary when N changes,
        # but should be of the same order of magnitude.
        assert new_total > 0, "Remapped modes should have some occupancy"
        # The new total should be at most the old total (energy can't
        # increase in an adiabatic process, but the overlap projection
        # can lose some)
        assert new_total <= old_total * 1.01, \
            f"Remapped total {new_total:.6f} should not exceed old {old_total:.6f}"

    def test_remap_ground_state_stays_ground(self):
        """If old crystal is in ground state, new one should be too."""
        ms_old = ModeStructure.compute(3, axial_freq=1e6, radial_freqs=(5e6, 5e6))
        # Don't heat — occupancies are all zero
        ms_new = ModeStructure.remap_after_split(
            old_structure=ms_old,
            new_n_ions=2,
            kept_ion_indices=[0, 1],
        )
        np.testing.assert_allclose(ms_new.occupancies, 0.0, atol=1e-14)

    def test_remap_merge_increases_modes(self):
        """Merging 2→3 ions creates 9 modes from 6."""
        ms_old = ModeStructure.compute(2, axial_freq=1e6, radial_freqs=(5e6, 5e6))
        ms_old.heat_modes(1.0, noise_exponent=1.0)

        ms_new = ModeStructure.remap_after_split(
            old_structure=ms_old,
            new_n_ions=3,
            kept_ion_indices=[0, 1],  # old 2 ions mapped to first 2 of new 3
        )
        assert ms_new.n_ions == 3
        assert len(ms_new.mode_frequencies) == 9

    def test_remap_new_structure_has_correct_frequencies(self):
        """Remapped structure should have correct frequencies for new N."""
        ms_old = ModeStructure.compute(3)
        ms_new = ModeStructure.remap_after_split(
            old_structure=ms_old,
            new_n_ions=2,
            kept_ion_indices=[0, 1],
        )
        # The new mode frequencies should match a fresh compute for N=2
        ms_fresh = ModeStructure.compute(2)
        np.testing.assert_allclose(
            ms_new.mode_frequencies, ms_fresh.mode_frequencies, rtol=1e-10,
        )


# =====================================================================
# 6.  ModeSnapshot
# =====================================================================


class TestModeSnapshot:
    """Test ModeSnapshot creation and immutability."""

    def test_snapshot_from_structure(self):
        """snapshot() should produce a valid ModeSnapshot."""
        ms = ModeStructure.compute(2, axial_freq=1e6, radial_freqs=(5e6, 5e6))
        ms.heat_modes(0.5, noise_exponent=1.0)
        snap = ms.snapshot()

        assert isinstance(snap, ModeSnapshot)
        assert snap.n_ions == 2
        assert len(snap.mode_frequencies) == 6
        assert len(snap.occupancies) == 6
        assert snap.eigenvectors.shape == (6, 2)
        np.testing.assert_allclose(snap.scalar_nbar, ms.scalar_nbar, atol=1e-12)

    def test_snapshot_is_independent_copy(self):
        """Mutating ModeStructure after snapshot should not affect snapshot."""
        ms = ModeStructure.compute(2)
        ms.heat_modes(1.0, noise_exponent=1.0)
        snap = ms.snapshot()

        old_nbar = snap.scalar_nbar
        # Mutate the original
        ms.heat_modes(100.0, noise_exponent=1.0)

        # Snapshot should be unchanged
        assert snap.scalar_nbar == old_nbar

    def test_snapshot_copy(self):
        """ModeSnapshot.copy() produces independent data."""
        ms = ModeStructure.compute(2)
        ms.heat_modes(1.0, noise_exponent=1.0)
        snap1 = ms.snapshot()
        snap2 = snap1.copy()

        np.testing.assert_array_equal(snap1.mode_frequencies, snap2.mode_frequencies)
        np.testing.assert_array_equal(snap1.occupancies, snap2.occupancies)
        assert snap1.scalar_nbar == snap2.scalar_nbar


# =====================================================================
# 7.  ManipulationTrap integration
# =====================================================================


class TestManipulationTrapModes:
    """Test ManipulationTrap auto-recomputes modes on ion changes."""

    def _make_ion(self, idx: int = 0) -> Ion:
        """Create a test Ion."""
        return Ion(idx=idx)

    def test_empty_trap_no_modes(self):
        """Empty trap should have mode_structure = None."""
        trap = ManipulationTrap(idx=0, position=(0, 0), capacity=10)
        assert trap.mode_structure is None

    def test_add_first_ion_creates_modes(self):
        """Adding first ion creates 3 modes."""
        trap = ManipulationTrap(idx=0, position=(0, 0), capacity=10)
        trap.add_ion(self._make_ion(0))
        assert trap.mode_structure is not None
        assert trap.mode_structure.n_ions == 1
        assert len(trap.mode_structure.mode_frequencies) == 3

    def test_add_second_ion_recomputes(self):
        """Adding second ion recomputes to 6 modes."""
        trap = ManipulationTrap(idx=0, position=(0, 0), capacity=10)
        trap.add_ion(self._make_ion(0))
        trap.add_ion(self._make_ion(1))
        assert trap.mode_structure is not None
        assert trap.mode_structure.n_ions == 2
        assert len(trap.mode_structure.mode_frequencies) == 6

    def test_remove_ion_recomputes(self):
        """Removing an ion recomputes modes for smaller crystal."""
        trap = ManipulationTrap(idx=0, position=(0, 0), capacity=10)
        ion0 = self._make_ion(0)
        ion1 = self._make_ion(1)
        trap.add_ion(ion0)
        trap.add_ion(ion1)
        assert trap.mode_structure.n_ions == 2

        trap.remove_ion(ion1)
        assert trap.mode_structure.n_ions == 1
        assert len(trap.mode_structure.mode_frequencies) == 3

    def test_remove_all_ions_clears_modes(self):
        """Removing all ions sets mode_structure to None."""
        trap = ManipulationTrap(idx=0, position=(0, 0), capacity=10)
        ion0 = self._make_ion(0)
        trap.add_ion(ion0)
        trap.remove_ion(ion0)
        assert trap.mode_structure is None

    def test_cool_trap_resets_modes(self):
        """cool_trap should reset mode occupancies to zero."""
        trap = ManipulationTrap(idx=0, position=(0, 0), capacity=10)
        # Need a cooling ion for cool_trap to work
        cooling_ion = Ion(idx=99, is_cooling=True)
        data_ion = self._make_ion(0)
        trap.add_ion(cooling_ion)
        trap.add_ion(data_ion)

        # Heat the modes
        trap.mode_structure.heat_modes(5.0, noise_exponent=1.0)
        assert trap.mode_structure.scalar_nbar > 0

        trap.cool_trap()
        np.testing.assert_allclose(trap.mode_structure.scalar_nbar, 0.0, atol=1e-14)

    def test_custom_secular_frequencies(self):
        """Custom secular frequencies propagate to mode structure."""
        trap = ManipulationTrap(
            idx=0, position=(0, 0), capacity=10,
            secular_frequencies=(2.0e6, 8.0e6, 8.0e6),
        )
        trap.add_ion(self._make_ion(0))
        # Single ion: modes at exactly (ω_z, ω_x, ω_y)
        np.testing.assert_allclose(
            trap.mode_structure.mode_frequencies,
            [2.0e6, 8.0e6, 8.0e6],
            rtol=1e-10,
        )


# =====================================================================
# 8.  Noise model callback hook
# =====================================================================


class TestNoiseModelCallback:
    """Test mode_noise_callback in TrappedIonNoiseModel."""

    def test_callback_not_invoked_when_none(self):
        """Without mode_snapshot, callback is never called."""
        from qectostim.experiments.hardware_simulation.trapped_ion.noise import (
            TrappedIonNoiseModel,
        )
        from qectostim.experiments.hardware_simulation.core.execution import (
            OperationTiming,
        )

        called = []

        def spy(gate, qubits, snap, channels):
            called.append(True)
            return channels

        model = TrappedIonNoiseModel(mode_noise_callback=spy)
        timing = OperationTiming(
            instruction_index=0,
            gate_name="MS",
            qubits=(0, 1),
            start_time=0.0,
            duration=40e-6,
            fidelity=0.99,
            batch_index=0,
            platform_context={
                "chain_length": 2,
                "motional_quanta": 0.1,
                "mode_snapshot": None,
            },
        )
        channels = model.apply_to_operation_timing(timing)
        assert len(called) == 0, "Callback should not be invoked without mode_snapshot"
        assert len(channels) > 0, "Standard channels should still be produced"

    def test_callback_invoked_with_snapshot(self):
        """With mode_snapshot, callback is invoked and can modify channels."""
        from qectostim.experiments.hardware_simulation.trapped_ion.noise import (
            TrappedIonNoiseModel,
        )
        from qectostim.experiments.hardware_simulation.core.execution import (
            OperationTiming,
        )

        received = {}

        def capture(gate, qubits, snap, channels):
            received["gate"] = gate
            received["qubits"] = qubits
            received["snap"] = snap
            received["n_channels"] = len(channels)
            return channels  # Pass through unchanged

        model = TrappedIonNoiseModel(mode_noise_callback=capture)

        ms = ModeStructure.compute(2, axial_freq=1e6, radial_freqs=(5e6, 5e6))
        ms.heat_modes(0.1, noise_exponent=1.0)
        snap = ms.snapshot()

        timing = OperationTiming(
            instruction_index=0,
            gate_name="MS",
            qubits=(0, 1),
            start_time=0.0,
            duration=40e-6,
            fidelity=0.99,
            batch_index=0,
            platform_context={
                "chain_length": 2,
                "motional_quanta": ms.scalar_nbar,
                "mode_snapshot": snap,
            },
        )
        channels = model.apply_to_operation_timing(timing)

        assert "gate" in received, "Callback should have been invoked"
        assert received["gate"] == "MS"
        assert received["qubits"] == (0, 1)
        assert isinstance(received["snap"], ModeSnapshot)
        assert received["snap"].n_ions == 2
        assert received["n_channels"] > 0

    def test_scalar_fidelity_unchanged_with_callback(self):
        """The scalar fidelity formula should produce identical results
        whether or not a mode_noise_callback is registered.

        This ensures the new mode-tracking infrastructure doesn't
        perturb the existing noise pipeline.
        """
        from qectostim.experiments.hardware_simulation.trapped_ion.noise import (
            TrappedIonNoiseModel,
        )
        from qectostim.experiments.hardware_simulation.core.execution import (
            OperationTiming,
        )

        model_plain = TrappedIonNoiseModel()
        model_with_cb = TrappedIonNoiseModel(
            mode_noise_callback=lambda g, q, s, ch: ch,  # passthrough
        )

        ms = ModeStructure.compute(3, axial_freq=1e6, radial_freqs=(5e6, 5e6))
        ms.heat_modes(0.5, noise_exponent=1.0)
        snap = ms.snapshot()

        timing = OperationTiming(
            instruction_index=0,
            gate_name="MS",
            qubits=(0, 1),
            start_time=0.0,
            duration=40e-6,
            fidelity=0.99,
            batch_index=0,
            platform_context={
                "chain_length": 3,
                "motional_quanta": ms.scalar_nbar,
                "mode_snapshot": snap,
            },
        )

        ch_plain = model_plain.apply_to_operation_timing(timing)
        ch_with = model_with_cb.apply_to_operation_timing(timing)

        assert len(ch_plain) == len(ch_with)
        for a, b in zip(ch_plain, ch_with):
            np.testing.assert_allclose(a.probability, b.probability, atol=1e-15)


# =====================================================================
# 9.  Hessian properties
# =====================================================================


class TestHessianProperties:
    """Test mathematical properties of the Hessian matrices."""

    def test_axial_hessian_symmetric(self):
        """Axial Hessian should be real symmetric."""
        pos = ModeStructure._equilibrium_positions(5)
        A = ModeStructure._axial_hessian(pos)
        np.testing.assert_allclose(A, A.T, atol=1e-12)

    def test_radial_hessian_symmetric(self):
        """Radial Hessian should be real symmetric."""
        pos = ModeStructure._equilibrium_positions(5)
        B = ModeStructure._radial_hessian(pos, omega_ratio=5.0)
        np.testing.assert_allclose(B, B.T, atol=1e-12)

    def test_axial_hessian_com_eigenvalue_is_one(self):
        """The smallest axial eigenvalue should be 1 (COM mode).

        This is a fundamental result: the centre-of-mass mode always
        oscillates at the bare trap frequency ω_z, regardless of N.
        """
        for n in [2, 3, 5, 8]:
            pos = ModeStructure._equilibrium_positions(n)
            A = ModeStructure._axial_hessian(pos)
            eigenvalues = np.sort(np.linalg.eigvalsh(A))
            np.testing.assert_allclose(
                eigenvalues[0], 1.0, atol=1e-8,
                err_msg=f"COM eigenvalue should be 1.0 for N={n}",
            )

    def test_radial_eigenvalues_positive(self):
        """All radial eigenvalues should be positive (chain is stable).

        This requires ω_radial >> ω_axial (tight radial confinement).
        """
        pos = ModeStructure._equilibrium_positions(5)
        B = ModeStructure._radial_hessian(pos, omega_ratio=5.0)
        eigenvalues = np.linalg.eigvalsh(B)
        assert np.all(eigenvalues > 0), \
            f"Radial eigenvalues should all be positive: {eigenvalues}"

    def test_axial_hessian_row_structure(self):
        """Verify diagonal = 1 + Σ_{j≠i} |A_{ij}|.

        The axial Hessian A has:
            A_{ii} = 1 + Σ_{j≠i} 2/|u_i-u_j|³   (trap + Coulomb)
            A_{ij} = -2/|u_i-u_j|³               (i ≠ j)

        So A_{ii} = 1 + Σ_{j≠i} |A_{ij}|.
        """
        pos = ModeStructure._equilibrium_positions(4)
        A = ModeStructure._axial_hessian(pos)
        for i in range(4):
            diag = A[i, i]
            off_diag_sum = sum(abs(A[i, j]) for j in range(4) if j != i)
            np.testing.assert_allclose(diag, 1.0 + off_diag_sum, atol=1e-10,
                                       err_msg=f"Row {i}: diagonal should equal 1 + sum of |off-diagonal|")
