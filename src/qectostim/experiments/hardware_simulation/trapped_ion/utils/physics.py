# src/qectostim/experiments/hardware_simulation/trapped_ion/physics.py
r"""
Trapped-ion physics engine — the single source of truth for all
normal-mode, fidelity, and calibration computations.

3N Normal-Mode Tracking: An Intuitive Explanation
=================================================

Start with the simplest picture: a ball on a spring
---------------------------------------------------

A single trapped ion is a charged particle sitting in an electromagnetic
"bowl" (the Paul trap).  If nudged, it oscillates at the *secular
frequency* of the trap — exactly like a marble rolling back and forth
in a bowl.

But the ion lives in 3D, so it can oscillate in three directions:

    * **Axially** (along the trap, like rolling left-right in a trough)
      — frequency ω_z
    * **Radially in x** (perpendicular to the trap) — frequency ω_x
    * **Radially in y** (other perpendicular direction) — frequency ω_y

So 1 ion → 3 modes of vibration, each at its own frequency::

    N = 1 ion:   3 modes
      ω_z = 1 MHz    (axial, gentle confinement)
      ω_x = 5 MHz    (radial, tight confinement)
      ω_y = 5 MHz    (radial, tight confinement)

Collective modes: why N ions ≠ N independent oscillators
--------------------------------------------------------

Put two ions in the same trap.  They repel each other (both positively
charged), so they settle at two equilibrium positions.  Now when you
nudge them they *interact* through the Coulomb repulsion, giving rise
to *collective* oscillation patterns::

    Mode 1: Centre-of-Mass (COM)         Mode 2: Breathing
      ← ● ● →                              ← ● → ← ● →
      Both ions move together               Ions move apart & together
      Frequency: ω_z (same as 1 ion!)      Frequency: √3 × ω_z ≈ 1.73 ω_z

The COM mode is the whole crystal sloshing as one unit — the trap
doesn't "know" there are two ions; it sees one blob, so the frequency
stays at ω_z.  The breathing mode has ions oscillating *against* each
other; Coulomb repulsion adds an extra restoring force, making it
stiffer → higher frequency.

This generalises: **N ions → N axial modes**, each a different collective
pattern.  Plus 2N radial modes.  Total = **3N modes**.

The eigenvalue solver: three physical steps
-------------------------------------------

**Step 1 — Where do the ions sit?** (``_equilibrium_positions``)

Each ion feels the trap pulling it toward the centre and every other
ion pushing it away (Coulomb repulsion).  Equilibrium is where these
balance for every ion simultaneously.  For 2 ions: ±0.63 (dimensionless).
For larger N the spacing becomes non-uniform — edge ions are farther
apart because fewer neighbours push them outward.

**Step 2 — How stiff are the connections?** (``_axial_hessian``)

Once we know where the ions sit, we ask: "if I slightly displace
ion *i*, how much force does that create on ion *j*?"  This yields
the N×N *Hessian* (stiffness matrix)::

    Think: N marbles connected by springs of different strengths.

      spring₁₂    spring₂₃
    ●──/\/\/──●──/\/\/──●
    ion1      ion2      ion3

    A[i,i] = 1 + Σ_{j≠i} 2/|distance|³   (trap + Coulomb stiffness)
    A[i,j] = −2/|distance|³               (coupling between ions)

**Step 3 — Diagonalise → collective patterns** (``np.linalg.eigh``)

Diagonalising the Hessian converts from "which ion moves" coordinates
to "which collective pattern moves" coordinates.  Each eigenvalue μ²
gives a mode frequency (ω_m = ω_z × √μ²) and each eigenvector tells
you the oscillation pattern::

    N = 4 ions, axial modes:
    Mode 1 (COM,  μ²=1):   [0.5,  0.5,  0.5,  0.5]   ← all equal
    Mode 2 (breathing):     [0.7,  0.2, -0.2, -0.7]   ← outer vs inner
    Mode 3 (quadrupole):    [0.5, -0.5, -0.5,  0.5]   ← alternating
    Mode 4 (octupole):      [0.2, -0.7,  0.7, -0.2]   ← highest freq

Radial modes follow the same procedure with one crucial sign difference:
Coulomb repulsion *reduces* radial stiffness (ions in a line are pushed
apart transversely), so the trap must confine them tightly radially
(ω_x, ω_y >> ω_z) to prevent the chain from buckling into a zigzag.

Why track modes?  Because heating is mode-dependent
---------------------------------------------------

Fluctuating electric fields at the trap electrodes inject energy
(anomalous heating), but these fields are nearly uniform across the
small crystal (~10 μm crystal, ~50 μm electrode distance).  A uniform
kick drives the COM mode strongly but barely affects the breathing
mode (where opposite ion motions cancel).  Quantitatively::

    ṅ_m = ṅ_COM × (ω_COM / ω_m)^α     (α ≈ 1–2)

    N = 4 ions, α = 1:
    COM       (ω = ω_z):       ṅ = ṅ_COM × 1.00    ← hottest
    Breathing (ω ≈ 1.7·ω_z):  ṅ = ṅ_COM × 0.58    ← cooler
    Quadrupole(ω ≈ 2.4·ω_z):  ṅ = ṅ_COM × 0.42    ← cooler still
    Octupole  (ω ≈ 2.9·ω_z):  ṅ = ṅ_COM × 0.35    ← coolest

This matters for gate fidelity because different gates couple to
different modes via the Lamb-Dicke parameter η_m^(i).  An MS gate
between ions 1 and 4 couples strongly to modes where those ions have
large eigenvector components — and the error depends on how hot
*those specific modes* are.

Split / merge: mode remapping
-----------------------------

When a 4-ion crystal splits into two 2-ion crystals, 12 modes become
6 + 6 modes.  Under the adiabatic approximation (split slow compared
to oscillation periods), each old mode smoothly deforms into a
combination of new modes.  Occupation transfers via the overlap::

    n̄_m^(new) = Σ_{m'} |⟨new_mode_m | old_mode_m'⟩|² × n̄_{m'}^(old)

The overlap is computed from eigenvectors restricted to shared ions.
See ``ModeStructure.remap_after_split()``.

Pipeline flow
-------------
::

    Circuit batch k:
      Transport: split/merge/move ions, each op heats modes via
        Δn̄_m = Δn̄_COM × (ω_COM / ω_m)^α
      ↓ SNAPSHOT: freeze 3N frequencies + occupancies
      ↓
    Gate executes (e.g. MS on ions 0,1):
      Old path (unchanged):  F = 1 − (ṅ·t + A·N/ln(N)·(2·scalar_nbar+1))
      New path (callback):   receives ModeSnapshot with per-mode detail
      ↓
    Next batch...

Key contract: **every gate sees the mode state at the instant it
executes**, not some end-of-circuit average.

Two-tier fidelity model: simple vs. mode-resolved
--------------------------------------------------

This module supports two levels of physics fidelity, selected
automatically based on what information the user provides:

**Tier 1 — Dimensionless / calibration-only (default)**

    Uses only ``CalibrationConstants`` (gate times, heating rate,
    scaling constant *A*, T2).  No ion species, trap geometry, or
    Lamb-Dicke parameters needed.  The fidelity formula is:

        F = 1 − (heating_rate × t_gate + A × N/ln(N) × (2n̄ + 1))

    This is the formula from [Pino20] p. 7 and is sufficient for
    circuit-level noise modelling where you only need a scalar n̄.

    **Classes/methods that use ONLY Tier 1 (no physical properties):**

    * ``CalibrationConstants``        — all fields & methods
    * ``ModeStructure.compute()``     — needs only N, ω_z, (ω_x, ω_y)
    * ``ModeStructure.heat_modes()``  — power-law heating, no mass/charge
    * ``ModeStructure.remap_after_split()`` — eigenvector overlap
    * ``ModeSnapshot``                — frozen DTO, no physics
    * ``IonChainFidelityModel.gate_fidelity()``         — scalar formula
    * ``IonChainFidelityModel.ms_gate_fidelity()``      — wrapper
    * ``IonChainFidelityModel.single_qubit_gate_fidelity()`` — wrapper
    * ``IonChainFidelityModel.dephasing_fidelity()``    — exp(−t/T2)

**Tier 2 — Physical properties (opt-in)**

    When you supply a ``TrapParameters`` (which bundles an
    ``IonSpecies`` with trap frequencies), the module can compute
    Lamb-Dicke parameters, mode-resolved gate infidelity, physical
    ion spacings, and heating rates from spectral density.  These
    require SI constants (ℏ, e, ε₀) from ``PhysicalConstants``.

    **Classes/methods that REQUIRE physical properties:**

    * ``PhysicalConstants``               — SI constants (ℏ, e, ε₀, k_B)
    * ``IonSpecies``                      — mass, laser wavelength
    * ``TrapParameters.length_scale_m()`` — needs ion mass + charge
    * ``TrapParameters.physical_positions_m()``  — needs length_scale_m
    * ``TrapParameters.lamb_dicke_parameter()``  — needs mass + wavevector
    * ``TrapParameters.lamb_dicke_matrix()``     — needs mass + wavevector
    * ``TrapParameters.heating_rate_from_spectral_density()`` — needs mass + charge
    * ``IonChainFidelityModel.ms_gate_fidelity_mode_resolved()``
      — needs ``TrapParameters`` for Lamb-Dicke; **falls back to
      scalar formula** if ``trap=None``
    * ``IonChainFidelityModel.transport_phase_error()``
      — simplified model, no species needed

    To opt in, either:
      1. Pass ``trap=TrapParameters(...)`` to ``CalibrationConstants``.
      2. Pass ``trap=...`` directly to ``ms_gate_fidelity_mode_resolved()``.

    If ``CalibrationConstants.trap`` is ``None`` (the default), all
    Tier-2 methods that need it will gracefully fall back to Tier-1
    scalar calculations.

Physics references
------------------
[James98]     D.F.V. James, Appl. Phys. B 66, 181 (1998)
              — Equilibrium positions & axial mode eigenvalues (Eqs. 3, 6, 17).
[Leibfried03] Leibfried et al., Rev. Mod. Phys. 75, 281 (2003)
              — Comprehensive trapped-ion review (§II.D normal modes,
                Eq. 20 Lamb-Dicke parameter).
[Brownnutt15] Brownnutt et al., Rev. Mod. Phys. 87, 1419 (2015)
              — Electric-field noise and anomalous heating (§III.A).
[Morigi06]    Morigi & Fishman, J. Phys. B 39, S403 (2006)
              — Radial modes with opposite-sign Hessian (§2).
[Home11]      Home et al., New J. Phys. 13, 073026 (2011)
              — Adiabatic split/merge and mode remapping (§3.2).
[Pino20]      Pino et al., arXiv:2004.04706
              — QCCD architecture, Table I times, p. 6-7 fidelity.
[Bermudez19]  Bermudez et al., Phys. Rev. A 99, 022330 (2019)
              — Table IV gate durations, heating rates, T2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np


# ============================================================================
# Physical constants — SI units, single source of truth
# ============================================================================

@dataclass(frozen=True)
class PhysicalConstants:
    r"""Fundamental physical constants in SI units.

    .. note:: **Tier 2 (physical-properties path only).**
       These constants are consumed by :class:`TrapParameters` and
       :class:`IonSpecies` for Lamb-Dicke parameters, physical ion
       spacings, and spectral-density heating rates.  The default
       Tier-1 fidelity path does **not** use them.

    Values are exact or CODATA-2018.  Having them in one frozen
    dataclass prevents magic numbers from scattering across modules.
    """
    hbar: float = 1.054571817e-34         # J·s   (reduced Planck constant)
    e_charge: float = 1.602176634e-19     # C     (elementary charge)
    epsilon_0: float = 8.8541878128e-12   # F/m   (vacuum permittivity)
    amu_to_kg: float = 1.66053906660e-27  # kg    (atomic mass unit)
    k_boltzmann: float = 1.380649e-23     # J/K   (Boltzmann constant)


CONSTANTS = PhysicalConstants()


# ============================================================================
# Ion species — all species-dependent physics in one place
# ============================================================================

@dataclass(frozen=True)
class IonSpecies:
    r"""Physical properties of a trapped-ion species.

    .. note:: **Tier 2 (physical-properties path only).**
       This class is only needed when you want to compute Lamb-Dicke
       parameters, physical ion spacings, or heating rates from
       electric-field spectral density.  The default Tier-1 scalar
       fidelity formula does **not** require an ``IonSpecies``.

    Every species-dependent quantity (mass, transition wavelengths,
    natural linewidths) lives here.  Gate physics, Lamb-Dicke
    parameters, and length scales all derive from these.

    Attributes
    ----------
    name : str
        Human-readable label, e.g. ``"171Yb+"``.
    mass_amu : float
        Atomic mass in unified atomic mass units.
    laser_wavelength_m : float
        Primary gate-laser wavelength in metres.  For Yb-171 this
        is the 355 nm Raman beam pair.
    transition_frequency_Hz : float
        Qubit transition frequency in Hz (hyperfine or optical).
    scattering_rate_per_gate : float
        Spontaneous photon scattering probability per gate operation.
        Dominant error source for Raman gates; zero for direct
        microwave/laser gates.
    """
    name: str = "171Yb+"
    mass_amu: float = 170.936326
    laser_wavelength_m: float = 355e-9           # Raman beam, [Olmschenk07]
    transition_frequency_Hz: float = 12.642812118e9  # hyperfine splitting
    scattering_rate_per_gate: float = 1e-4       # typical Raman scattering

    @property
    def mass_kg(self) -> float:
        """Ion mass in kilograms."""
        return self.mass_amu * CONSTANTS.amu_to_kg

    @property
    def laser_wavevector(self) -> float:
        """Laser wavevector magnitude k = 2π/λ (rad/m)."""
        return 2.0 * np.pi / self.laser_wavelength_m


# Common species presets
YB171 = IonSpecies()
BA137 = IonSpecies(
    name="137Ba+",
    mass_amu=136.905827,
    laser_wavelength_m=493e-9,
    transition_frequency_Hz=8.037e9,
    scattering_rate_per_gate=5e-5,
)
CA40 = IonSpecies(
    name="40Ca+",
    mass_amu=39.962591,
    laser_wavelength_m=729e-9,
    transition_frequency_Hz=411.0e12,  # optical qubit
    scattering_rate_per_gate=0.0,      # no Raman scattering for direct drive
)


# ============================================================================
# Trap geometry — physical parameters of the Paul trap
# ============================================================================

@dataclass
class TrapParameters:
    r"""Physical parameters of a linear Paul trap.

    .. note:: **Tier 2 (physical-properties path only).**
       Only needed for Lamb-Dicke parameters, physical ion spacings,
       and spectral-density heating rates.  The default Tier-1 scalar
       fidelity formula does **not** need a ``TrapParameters``.

    Collects the trap frequencies and derived length scale in one
    place.  The length scale
    :math:`\ell = (e^2 / 4\pi\varepsilon_0\, m\omega_z^2)^{1/3}`
    converts between the dimensionless equilibrium positions used by
    :class:`ModeStructure` and physical distances in metres.

    **Tier-1 overlap:** The trap *frequencies* (ω_z, ω_x, ω_y) are
    also used by ``ModeStructure.compute()`` for the dimensionless
    eigenvalue solver, but that path only uses the frequency ratios
    (ω_x/ω_z, ω_y/ω_z) — it does **not** need the ion species.

    Attributes
    ----------
    axial_freq_Hz : float
        Axial secular frequency ω_z in Hz.
    radial_freq_x_Hz : float
        Radial secular frequency ω_x in Hz.
    radial_freq_y_Hz : float
        Radial secular frequency ω_y in Hz.
    ion_species : IonSpecies
        The ion species in this trap.  Only used by Tier-2 methods
        (``length_scale_m``, ``lamb_dicke_*``,
        ``heating_rate_from_spectral_density``).
    """
    axial_freq_Hz: float = 1.0e6
    radial_freq_x_Hz: float = 5.0e6
    radial_freq_y_Hz: float = 5.0e6
    ion_species: IonSpecies = field(default_factory=lambda: YB171)

    @property
    def radial_freqs(self) -> Tuple[float, float]:
        """Radial frequencies as a tuple (ω_x, ω_y)."""
        return (self.radial_freq_x_Hz, self.radial_freq_y_Hz)

    @property
    def length_scale_m(self) -> float:
        r"""Characteristic length scale ℓ in metres.

        ℓ = (e² / (4πε₀ m ω_z²))^{1/3}

        Multiply dimensionless equilibrium positions by ℓ to get
        physical ion separations.  [James98] Eq. 2.
        """
        m = self.ion_species.mass_kg
        omega_z = 2.0 * np.pi * self.axial_freq_Hz
        numerator = CONSTANTS.e_charge ** 2 / (4.0 * np.pi * CONSTANTS.epsilon_0)
        denominator = m * omega_z ** 2
        return float((numerator / denominator) ** (1.0 / 3.0))

    def physical_positions_m(self, dimensionless_positions: np.ndarray) -> np.ndarray:
        """Convert dimensionless equilibrium positions to metres."""
        return dimensionless_positions * self.length_scale_m

    def lamb_dicke_parameter(
        self,
        mode_frequency_Hz: float,
        eigenvector_component: float,
    ) -> float:
        r"""Compute Lamb-Dicke parameter η for one ion in one mode.

        η_m^{(i)} = k · √(ℏ / (2 M ω_m)) · |b_{mi}|

        [Leibfried03] Eq. 20.

        Parameters
        ----------
        mode_frequency_Hz : float
            Normal-mode frequency ω_m in Hz.
        eigenvector_component : float
            Eigenvector element b_{mi} for ion i in mode m.

        Returns
        -------
        float
            Dimensionless Lamb-Dicke parameter.
        """
        if mode_frequency_Hz <= 0:
            return 0.0
        k = self.ion_species.laser_wavevector
        m = self.ion_species.mass_kg
        omega_m = 2.0 * np.pi * mode_frequency_Hz
        return float(k * np.sqrt(CONSTANTS.hbar / (2.0 * m * omega_m)) * abs(eigenvector_component))

    def lamb_dicke_matrix(
        self,
        mode_frequencies: np.ndarray,
        eigenvectors: np.ndarray,
    ) -> np.ndarray:
        r"""Compute full Lamb-Dicke matrix η[m, i] for all modes and ions.

        Parameters
        ----------
        mode_frequencies : np.ndarray
            Shape ``(n_modes,)``.  Normal-mode frequencies in Hz.
        eigenvectors : np.ndarray
            Shape ``(n_modes, n_ions)``.  Participation matrix.

        Returns
        -------
        np.ndarray
            Shape ``(n_modes, n_ions)``.  Lamb-Dicke parameters.
        """
        k = self.ion_species.laser_wavevector
        m = self.ion_species.mass_kg
        omega = 2.0 * np.pi * np.maximum(mode_frequencies, 1e-10)
        # shape (n_modes,)
        scale = k * np.sqrt(CONSTANTS.hbar / (2.0 * m * omega))
        # broadcast: (n_modes, 1) * (n_modes, n_ions)
        return scale[:, np.newaxis] * np.abs(eigenvectors)

    def heating_rate_from_spectral_density(
        self,
        S_E: float,
        mode_frequency_Hz: float,
    ) -> float:
        r"""Convert electric-field noise spectral density to heating rate.

        ṅ = e² S_E(ω) / (4 m ℏ ω)

        [Brownnutt15] Eq. 1.

        Parameters
        ----------
        S_E : float
            Single-sided electric-field noise spectral density in V²/m²/Hz.
        mode_frequency_Hz : float
            Mode frequency in Hz.

        Returns
        -------
        float
            Heating rate in quanta/second.
        """
        if mode_frequency_Hz <= 0:
            return 0.0
        omega = 2.0 * np.pi * mode_frequency_Hz
        m = self.ion_species.mass_kg
        return float(
            CONSTANTS.e_charge ** 2 * S_E
            / (4.0 * m * CONSTANTS.hbar * omega)
        )


DEFAULT_TRAP = TrapParameters()


# ============================================================================
# Calibration constants — single source of truth
# ============================================================================

@dataclass
class CalibrationConstants:
    """Unified calibration constants for trapped-ion QCCD hardware.

    .. note:: **Tier 1 by default.**
       All scalar fields (gate times, heating rate, *A*, T2, etc.)
       work without any physical properties.  Set ``trap`` to a
       :class:`TrapParameters` instance to unlock Tier-2
       mode-resolved fidelity.

    Merges what was previously split between module-level globals in
    transport.py (BACKGROUND_HEATING_RATE, FIDELITY_SCALING_A, T2) and
    the TrappedIonCalibration dataclass in noise.py.

    All values are from two references:

    * **[1]** Pino *et al.*, arXiv:2004.04706 — Table I (transport
      times), page 6–7 (gate-time models, fidelity formula).
    * **[2]** Bermudez *et al.*, Phys. Rev. A **99**, 022330 (2019) —
      Table IV (gate durations, heating rates, measurement/reset
      infidelities, T2 time).

    Attributes
    ----------
    heating_rate : float
        Background motional heating rate (quanta/s).  Used in the
        fidelity formula ``F = 1 − (heating_rate·t_gate + A·N/ln(N)·(2n̄+1))``.
        Default 3.9996 q/s from [2] Table IV.
    fidelity_scaling_A : float
        Dimensionless scaling constant *A* in the fidelity formula.
        Default 0.003680029 from [1] page 7.
    t2_time : float
        T2 dephasing time in seconds.  Default 2.2 s from [2].
    ms_gate_time : float
        MS (two-qubit) gate duration in seconds.  Default 40 μs
        from [1] Table I / [2] Table IV.
    single_qubit_gate_time : float
        Single-qubit gate duration in seconds.  Default 5 μs from [2].
    measurement_time : float
        Measurement duration in seconds.  Default 400 μs from [2].
    reset_time : float
        Qubit reset duration in seconds.  Default 50 μs from [2].
    measurement_infidelity : float
        Measurement bit-flip probability.  Default 1e-3 from [2].
    reset_infidelity : float
        Qubit-reset error probability.  Default 5e-3 from [2].
    recool_time : float
        Sympathetic re-cooling duration in seconds.  Default 400 μs.
    transport_heating : Dict[str, float]
        Motional quanta per transport operation type.  Computed as
        ``HEATING_RATE × OP_TIME`` from [1] Table I / [2] Table IV.
    """

    # --- Background heating (for gate fidelity formula) ---
    heating_rate: float = 3.9996319971   # quanta/s  [2] Table IV

    # --- Fidelity formula constant ---
    fidelity_scaling_A: float = 0.003680029   # dimensionless, [1] p.7

    # --- Coherence ---
    t2_time: float = 2.2   # seconds  [2]

    # --- Gate times (seconds) ---
    ms_gate_time: float = 40e-6            # seconds  [1] Table I
    single_qubit_gate_time: float = 5e-6   # seconds  [2] Table IV

    # --- Measurement / reset ---
    measurement_time: float = 400e-6       # seconds  [2] Table IV
    reset_time: float = 50e-6              # seconds  [2] Table IV
    measurement_infidelity: float = 1e-3   # [2] Table IV
    reset_infidelity: float = 5e-3         # [2] Table IV
    recool_time: float = 400e-6            # seconds

    # --- Transport operation times (seconds) ---
    split_time: float = 80e-6             # [1] Table I
    merge_time: float = 80e-6             # [1] Table I
    shuttle_time: float = 5e-6            # [1] Table I
    junction_time: float = 50e-6          # [1] Table I
    rotation_time: float = 42e-6          # [1] Table I
    crossing_swap_time: float = 100e-6    # [1] Table I

    # --- Transport heating rates (quanta/second) ---
    split_heating_rate: float = 6.0       # [1] Table I
    merge_heating_rate: float = 6.0       # [1] Table I
    shuttle_heating_rate: float = 0.1     # [1] Table I
    junction_heating_rate: float = 3.0    # [1] Table I
    rotation_heating_rate: float = 0.3    # [2] Table IV
    cooling_heating_rate: float = 0.1     # [2] Table IV (residual)
    crossing_swap_heating_rate: float = 3.0  # [1] Table I

    # --- Trap physics (frequencies, ion species) ---
    # None by default → Tier-1 scalar fidelity only.
    # Set to a TrapParameters instance to enable Tier-2
    # mode-resolved fidelity / Lamb-Dicke calculations.
    trap: Optional[TrapParameters] = None

    # --- Per-transport-operation heating (quanta, derived) ---
    transport_heating: Dict[str, float] = field(default=None)

    def __post_init__(self):
        if self.transport_heating is None:
            self.transport_heating = {
                "split":         self.split_heating_rate * self.split_time,
                "merge":         self.merge_heating_rate * self.merge_time,
                "shuttle":       self.shuttle_heating_rate * self.shuttle_time,
                "move":          self.shuttle_heating_rate * self.shuttle_time,
                "junction":      self.junction_heating_rate * self.junction_time,
                "rotation":      self.rotation_heating_rate * self.rotation_time,
                "cooling":       self.cooling_heating_rate * self.recool_time,
                "crossing_swap": self.crossing_swap_heating_rate * self.crossing_swap_time,
            }

    # ------------------------------------------------------------------
    # Transport times / heating look-ups
    # ------------------------------------------------------------------

    def transport_times_s(self) -> Dict[str, float]:
        """Transport operation durations in seconds."""
        return {
            "split":         self.split_time,
            "merge":         self.merge_time,
            "shuttle":       self.shuttle_time,
            "move":          self.shuttle_time,
            "junction":      self.junction_time,
            "rotation":      self.rotation_time,
            "cooling":       self.recool_time,
            "crossing_swap": self.crossing_swap_time,
        }

    def transport_heating_rates(self) -> Dict[str, float]:
        """Transport operation heating rates in quanta/second."""
        return {
            "split":         self.split_heating_rate,
            "merge":         self.merge_heating_rate,
            "shuttle":       self.shuttle_heating_rate,
            "move":          self.shuttle_heating_rate,
            "junction":      self.junction_heating_rate,
            "rotation":      self.rotation_heating_rate,
            "cooling":       self.cooling_heating_rate,
            "crossing_swap": self.crossing_swap_heating_rate,
        }

    # ------------------------------------------------------------------
    # Derived gate-time and gate-fidelity look-ups
    # ------------------------------------------------------------------

    def gate_times_us(self) -> Dict[str, float]:
        """Derive per-gate execution times in **microseconds**.

        Times for composite gates (CNOT, CZ, SWAP, …) are computed
        from the native-gate times using the decompositions in
        ``TrappedIonGateDecomposer``:

        * CNOT = 1 MS + 3 single-qubit rotations
        * CZ   = CNOT + 2 Hadamard-equivalent rotations
        * SWAP  = 3 × CNOT

        Virtual-Z rotations are effectively free (< 0.1 μs).
        """
        sq = self.single_qubit_gate_time * 1e6   # μs
        ms = self.ms_gate_time * 1e6             # μs
        meas = self.measurement_time * 1e6       # μs
        rst = self.reset_time * 1e6              # μs
        rz = 0.1                                 # virtual, ~free

        cnot = ms + 3 * sq          # 1 MS + 3 rotations
        cz = cnot + 2 * sq          # CNOT + 2 Hadamard-rotations
        swap = 3 * cnot             # 3 × CNOT

        return {
            "MS":   ms,
            "RX":   sq,
            "RY":   sq,
            "RZ":   rz,
            "H":    sq,              # RY(π/2)·RZ(π); RZ is virtual
            "CNOT": cnot,
            "CX":   cnot,
            "CZ":   cz,
            "M":    meas,
            "R":    rst,
            "MR":   meas + rst,
            "SWAP": swap,
        }

    def gate_fidelities(self, chain_length: int = 2) -> Dict[str, float]:
        """Derive per-gate fidelities from calibration constants.

        .. note:: **Tier 1** — uses only scalar calibration data;
           no physical properties (ion species, trap geometry) needed.

        Single/two-qubit gate fidelities use the chain-length-dependent
        formula from arXiv:2004.04706 page 7:

            F = 1 − A · N / ln(N) · (2n̄ + 1)

        where *N* = ``chain_length``, *A* = ``fidelity_scaling_A``, and
        n̄ = ``heating_rate × gate_time``.  Measurement and reset use
        the fixed infidelity fields.

        Parameters
        ----------
        chain_length : int
            Number of ions in the local crystal (default 2).
        """
        import math

        A = self.fidelity_scaling_A
        N = max(chain_length, 2)
        ln_N = math.log(N) if N > 1 else 1.0

        def _gate_fidelity(t_gate: float) -> float:
            nbar = self.heating_rate * t_gate
            return 1.0 - A * N / ln_N * (2 * nbar + 1)

        sq_f = _gate_fidelity(self.single_qubit_gate_time)
        ms_f = _gate_fidelity(self.ms_gate_time)

        # Composite fidelities — product of constituents
        cnot_f = ms_f * sq_f ** 3
        cz_f = cnot_f * sq_f ** 2
        swap_f = cnot_f ** 3

        return {
            "MS":   ms_f,
            "RX":   sq_f,
            "RY":   sq_f,
            "RZ":   0.99999,                        # virtual, ~perfect
            "H":    sq_f,
            "CNOT": cnot_f,
            "CX":   cnot_f,
            "CZ":   cz_f,
            "M":    1.0 - self.measurement_infidelity,
            "R":    1.0 - self.reset_infidelity,
            "MR":   (1.0 - self.measurement_infidelity)
                    * (1.0 - self.reset_infidelity),
            "SWAP": swap_f,
        }


# Singleton default instance (used by transport.py, noise.py, etc.)
DEFAULT_CALIBRATION = CalibrationConstants()


# ============================================================================
# ModeSnapshot — frozen DTO
# ============================================================================

@dataclass
class ModeSnapshot:
    """Frozen snapshot of the normal-mode state of an ion chain.

    .. note:: **Tier 1** — this data class is produced entirely by
       :class:`ModeStructure` (Tier 1).  It carries everything the
       Tier-1 scalar fidelity formula needs (``scalar_nbar``,
       ``n_ions``) as well as the per-mode data that Tier 2 consumes.

    This is the lightweight, immutable data object that gets passed
    through the pipeline (routing → execution planner → noise model).
    The collaborator's noise model will consume this to compute
    mode-resolved gate infidelities.

    Attributes
    ----------
    n_ions : int
        Number of ions in the crystal when this snapshot was taken.
    mode_frequencies : np.ndarray
        Shape ``(3*n_ions,)``.  Normal-mode frequencies in Hz.
        Convention: first ``n_ions`` entries are axial modes (sorted
        ascending), next ``2*n_ions`` are radial modes (x then y,
        each sorted ascending).
    eigenvectors : np.ndarray
        Shape ``(3*n_ions, n_ions)``.  Participation matrix *b_{mi}*
        where ``eigenvectors[m, i]`` is how much ion *i* participates
        in mode *m*.  Needed for Lamb-Dicke parameter calculation:
        ``η_m^(i) = k * sqrt(ℏ / 2Mω_m) * b_{mi}``
        (Leibfried et al., Rev. Mod. Phys. 75, 281 (2003), Eq. 20).
    occupancies : np.ndarray
        Shape ``(3*n_ions,)``.  Per-mode mean phonon number n̄_m.
        Accumulated from transport heating; reset by cooling.
    scalar_nbar : float
        Sum of all mode occupancies, ``Σ_m n̄_m``.  This is the
        backward-compatible scalar consumed by the existing gate
        fidelity formula (which uses total n̄ in
        ``F = 1 − (ṅ·t + A·N/ln(N)·(2n̄+1))``).
    """
    n_ions: int
    mode_frequencies: np.ndarray
    eigenvectors: np.ndarray
    occupancies: np.ndarray
    scalar_nbar: float

    def copy(self) -> "ModeSnapshot":
        """Return a deep copy of this snapshot."""
        return ModeSnapshot(
            n_ions=self.n_ions,
            mode_frequencies=self.mode_frequencies.copy(),
            eigenvectors=self.eigenvectors.copy(),
            occupancies=self.occupancies.copy(),
            scalar_nbar=self.scalar_nbar,
        )


# ============================================================================
# ModeStructure — eigenvalue solver for N-ion Coulomb crystal
# ============================================================================

@dataclass
class ModeStructure:
    """Normal-mode structure of an N-ion linear Coulomb crystal.

    .. note:: **Tier 1** — all methods in this class work with only
       dimensionless / frequency-ratio physics.  No ion species, mass,
       or laser wavelength is needed.  The mode spectrum depends only
       on N, ω_z, ω_x, ω_y.

    Computes and stores the 3N normal-mode frequencies, eigenvectors,
    and per-mode phonon occupancies for a chain of N ions in a linear
    Paul trap.  This is the "physics engine" that lives on each
    ManipulationTrap and gets recomputed whenever the ion count changes
    (split, merge, ion load/unload).

    The mode spectrum is fully determined by:
    1. **N** — the number of ions in the crystal.
    2. **ω_z** — the axial secular frequency of the trap.
    3. **(ω_x, ω_y)** — the two radial secular frequencies.

    The axial modes are found by solving the eigenvalue problem for the
    Hessian of the combined harmonic trap + Coulomb potential, evaluated
    at the equilibrium positions.  For N ions in a harmonic axial
    potential V = ½mω_z²z², the equilibrium positions u_i⁰ (in units
    of the length scale l = (e²/4πε₀mω_z²)^{1/3}) satisfy:

        u_i − Σ_{j≠i} sgn(u_i−u_j) / (u_i−u_j)² = 0    [James98 Eq. 3]

    The axial normal-mode eigenvalues μ_m² come from:

        Σ_{j≠i} 2/|u_i⁰−u_j⁰|³ · (b_{mj}−b_{mi}) + μ_m² · b_{mi} = 0
                                                          [James98 Eq. 17]

    which is an N×N real symmetric eigenvalue problem Ax = μ²x where:

        A_{ii} = Σ_{j≠i} 2/|u_i⁰−u_j⁰|³
        A_{ij} = −2/|u_i⁰−u_j⁰|³           (i ≠ j)

    The physical mode frequencies are then ω_m = ω_z · √(μ_m²) for
    axial modes.

    Radial modes use the same equilibrium positions with a different
    Hessian.  For the x-direction (analogous for y):

        B_{ii} = (ω_x/ω_z)² − Σ_{j≠i} 1/|u_i⁰−u_j⁰|³
        B_{ij} = +1/|u_i⁰−u_j⁰|³           (i ≠ j)

    The radial mode frequencies are ω_m = ω_z · √(μ_m²) where μ_m²
    are the eigenvalues of B.  Note the sign difference: radial modes
    have *repulsive* Coulomb correction (ions push each other apart
    radially), while axial modes have *attractive* correction (ions
    bunch together axially).  See [Morigi06] §2.

    Attributes
    ----------
    n_ions : int
        Number of ions in the crystal.
    axial_freq : float
        Axial secular frequency ω_z in Hz.
    radial_freqs : Tuple[float, float]
        Radial secular frequencies (ω_x, ω_y) in Hz.
    equilibrium_positions : np.ndarray
        Shape ``(n_ions,)``.  Dimensionless equilibrium positions.
    mode_frequencies : np.ndarray
        Shape ``(3*n_ions,)``.  All normal-mode frequencies in Hz.
    eigenvectors : np.ndarray
        Shape ``(3*n_ions, n_ions)``.  Participation matrix.
    occupancies : np.ndarray
        Shape ``(3*n_ions,)``.  Per-mode mean phonon numbers.
    """
    n_ions: int
    axial_freq: float
    radial_freqs: Tuple[float, float]
    equilibrium_positions: np.ndarray
    mode_frequencies: np.ndarray
    eigenvectors: np.ndarray
    occupancies: np.ndarray

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def compute(
        cls,
        n_ions: int,
        axial_freq: float = 1.0e6,
        radial_freqs: Tuple[float, float] = (5.0e6, 5.0e6),
    ) -> "ModeStructure":
        """Compute the full 3N normal-mode structure for *n_ions* ions.

        .. note:: **Tier 1** — uses only N, ω_z, (ω_x, ω_y).  No ion
           species or physical constants needed.  The equilibrium
           positions and eigenvalues are dimensionless [James98].

        This is the main entry point.  It:
        1. Finds equilibrium positions by minimising the axial potential.
        2. Builds the axial Hessian A and diagonalises it → N axial modes.
        3. Builds radial Hessians B_x, B_y and diagonalises → 2N radial modes.
        4. Combines into a single ``(3N,)`` frequency array and
           ``(3N, N)`` eigenvector matrix.

        Parameters
        ----------
        n_ions : int
            Number of ions (must be ≥ 1).
        axial_freq : float
            Axial secular frequency ω_z in Hz.  Default 1 MHz.
        radial_freqs : Tuple[float, float]
            Radial secular frequencies (ω_x, ω_y) in Hz.  Default (5, 5) MHz.

        Returns
        -------
        ModeStructure
            The computed mode structure with all occupancies at zero.
        """
        if n_ions < 1:
            raise ValueError(f"n_ions must be >= 1, got {n_ions}")

        eq_pos = cls._equilibrium_positions(n_ions)

        if n_ions == 1:
            axial_eigenvalues = np.array([1.0])
            axial_eigenvectors = np.array([[1.0]])
        else:
            A = cls._axial_hessian(eq_pos)
            axial_eigenvalues, axial_eigenvectors = np.linalg.eigh(A)

        axial_freqs = axial_freq * np.sqrt(np.maximum(axial_eigenvalues, 0.0))

        omega_x, omega_y = radial_freqs
        if n_ions == 1:
            radial_x_freqs = np.array([omega_x])
            radial_y_freqs = np.array([omega_y])
            radial_x_evecs = np.array([[1.0]])
            radial_y_evecs = np.array([[1.0]])
        else:
            Bx = cls._radial_hessian(eq_pos, omega_x / axial_freq)
            radial_x_evals, radial_x_evecs = np.linalg.eigh(Bx)
            radial_x_freqs = axial_freq * np.sqrt(
                np.maximum(radial_x_evals, 0.0)
            )
            By = cls._radial_hessian(eq_pos, omega_y / axial_freq)
            radial_y_evals, radial_y_evecs = np.linalg.eigh(By)
            radial_y_freqs = axial_freq * np.sqrt(
                np.maximum(radial_y_evals, 0.0)
            )

        all_freqs = np.concatenate([axial_freqs, radial_x_freqs, radial_y_freqs])
        all_evecs = np.vstack([
            axial_eigenvectors,
            radial_x_evecs,
            radial_y_evecs,
        ])

        return cls(
            n_ions=n_ions,
            axial_freq=axial_freq,
            radial_freqs=radial_freqs,
            equilibrium_positions=eq_pos,
            mode_frequencies=all_freqs,
            eigenvectors=all_evecs,
            occupancies=np.zeros(3 * n_ions),
        )

    # ------------------------------------------------------------------
    # Equilibrium positions  [James98 Eq. 3]
    # ------------------------------------------------------------------

    @staticmethod
    def _equilibrium_positions(n: int) -> np.ndarray:
        """Find dimensionless equilibrium positions for N ions.

        Solves for positions u_i that minimise the total potential:
            V(u) = Σ_i ½u_i² + Σ_{i<j} 1/|u_i − u_j|

        Parameters
        ----------
        n : int
            Number of ions.

        Returns
        -------
        np.ndarray
            Shape ``(n,)``, sorted equilibrium positions.
        """
        if n == 1:
            return np.array([0.0])
        if n == 2:
            d = 0.25 ** (1.0 / 3.0)
            return np.array([-d, d])

        u = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * (n ** (1.0 / 3.0)) * 0.5

        for _iteration in range(200):
            force = -u.copy()
            for i in range(n):
                for j in range(n):
                    if i != j:
                        diff = u[i] - u[j]
                        force[i] += np.sign(diff) / (diff * diff)

            jac = -np.eye(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        diff = u[i] - u[j]
                        deriv = -2.0 / (abs(diff) ** 3)
                        jac[i, i] += deriv
                        jac[i, j] -= deriv

            try:
                delta = np.linalg.solve(jac, -force)
            except np.linalg.LinAlgError:
                break
            u += delta

            if np.max(np.abs(force)) < 1e-12:
                break

        return np.sort(u)

    # ------------------------------------------------------------------
    # Axial Hessian  [James98 Eq. 17]
    # ------------------------------------------------------------------

    @staticmethod
    def _axial_hessian(eq_pos: np.ndarray) -> np.ndarray:
        """Build the axial-mode Hessian matrix.

        A_{ii} = Σ_{j≠i} 2/|u_i⁰ − u_j⁰|³  (trap + Coulomb)
        A_{ij} = −2/|u_i⁰ − u_j⁰|³          (i ≠ j)

        Returns
        -------
        np.ndarray
            Shape ``(N, N)``, real symmetric Hessian.
        """
        n = len(eq_pos)
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = 1.0
            for j in range(n):
                if i != j:
                    diff = abs(eq_pos[i] - eq_pos[j])
                    coupling = 2.0 / (diff ** 3)
                    A[i, i] += coupling
                    A[i, j] -= coupling
        return A

    # ------------------------------------------------------------------
    # Radial Hessian  [Morigi06 §2]
    # ------------------------------------------------------------------

    @staticmethod
    def _radial_hessian(eq_pos: np.ndarray, omega_ratio: float) -> np.ndarray:
        """Build the radial-mode Hessian matrix for one transverse axis.

        B_{ii} = (ω_r/ω_z)² − Σ_{j≠i} 1/|u_i⁰ − u_j⁰|³
        B_{ij} = +1/|u_i⁰ − u_j⁰|³     (i ≠ j)

        Note the sign is opposite to the axial case.

        Returns
        -------
        np.ndarray
            Shape ``(N, N)``, real symmetric Hessian.
        """
        n = len(eq_pos)
        B = np.zeros((n, n))
        for i in range(n):
            B[i, i] = omega_ratio ** 2
            for j in range(n):
                if i != j:
                    diff = abs(eq_pos[i] - eq_pos[j])
                    coupling = 1.0 / (diff ** 3)
                    B[i, i] -= coupling
                    B[i, j] += coupling
        return B

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def scalar_nbar(self) -> float:
        """Total mean phonon number: Σ_m n̄_m."""
        return float(np.sum(self.occupancies))

    @property
    def axial_frequencies(self) -> np.ndarray:
        """The N axial mode frequencies (first N entries)."""
        return self.mode_frequencies[:self.n_ions]

    @property
    def radial_x_frequencies(self) -> np.ndarray:
        """The N radial-x mode frequencies."""
        return self.mode_frequencies[self.n_ions:2 * self.n_ions]

    @property
    def radial_y_frequencies(self) -> np.ndarray:
        """The N radial-y mode frequencies."""
        return self.mode_frequencies[2 * self.n_ions:]

    @property
    def com_frequency(self) -> float:
        """Centre-of-mass (COM) mode frequency (always = ω_z)."""
        return float(self.mode_frequencies[0])

    def snapshot(self) -> ModeSnapshot:
        """Take a frozen snapshot of the current mode state."""
        return ModeSnapshot(
            n_ions=self.n_ions,
            mode_frequencies=self.mode_frequencies.copy(),
            eigenvectors=self.eigenvectors.copy(),
            occupancies=self.occupancies.copy(),
            scalar_nbar=self.scalar_nbar,
        )

    def heat_modes(
        self,
        com_heating_quanta: float,
        noise_exponent: float = 1.0,
    ) -> None:
        """Distribute heating across modes according to the noise spectrum.

        .. note:: **Tier 1** — uses a power-law approximation.
           No ion mass or charge needed.

            Δn̄_m = Δn̄_COM × (ω_COM / ω_m)^α

        Parameters
        ----------
        com_heating_quanta : float
            Motional quanta deposited into the COM mode.
        noise_exponent : float
            Power-law exponent α.  Default 1.0 (1/f noise).
        """
        if com_heating_quanta <= 0 or self.n_ions == 0:
            return

        omega_com = self.mode_frequencies[0]
        if omega_com <= 0:
            self.occupancies += com_heating_quanta / max(len(self.occupancies), 1)
            return

        for m in range(len(self.occupancies)):
            omega_m = self.mode_frequencies[m]
            if omega_m > 0:
                ratio = omega_com / omega_m
                self.occupancies[m] += com_heating_quanta * (ratio ** noise_exponent)
            else:
                self.occupancies[m] += com_heating_quanta

    def cool_to_ground(self) -> None:
        """Reset all mode occupancies to zero (ground-state cooling)."""
        self.occupancies[:] = 0.0

    @classmethod
    def remap_after_split(
        cls,
        old_structure: "ModeStructure",
        new_n_ions: int,
        kept_ion_indices: List[int],
        axial_freq: Optional[float] = None,
        radial_freqs: Optional[Tuple[float, float]] = None,
    ) -> "ModeStructure":
        """Compute new mode structure after a crystal-changing operation.

        .. note:: **Tier 1** — dimensionless eigenvector overlap;
           no ion species or physical constants needed.

        Redistributes phonon occupancies via eigenvector overlap:
            n̄_m^(new) = Σ_{m'} |⟨m_new | m'_old⟩|² × n̄_{m'}^(old)

        Parameters
        ----------
        old_structure : ModeStructure
            Mode structure before the crystal change.
        new_n_ions : int
            Number of ions in the new crystal.
        kept_ion_indices : List[int]
            Which ions from the old crystal remain.  Length = new_n_ions.
        axial_freq : float, optional
            Axial frequency for new trap (defaults to old value).
        radial_freqs : tuple, optional
            Radial frequencies for new trap (defaults to old value).

        Returns
        -------
        ModeStructure
            New structure with redistributed occupancies.
        """
        ax = axial_freq if axial_freq is not None else old_structure.axial_freq
        rf = radial_freqs if radial_freqs is not None else old_structure.radial_freqs

        new_struct = cls.compute(new_n_ions, ax, rf)

        if not kept_ion_indices or old_structure.n_ions == 0:
            return new_struct

        old_n = old_structure.n_ions
        new_n = new_n_ions
        n_shared = len(kept_ion_indices)

        for block_idx in range(3):  # axial, radial-x, radial-y
            old_start = block_idx * old_n
            old_end = old_start + old_n
            new_start = block_idx * new_n
            new_end = new_start + new_n

            old_evecs = old_structure.eigenvectors[old_start:old_end, :]
            new_evecs = new_struct.eigenvectors[new_start:new_end, :]

            old_sub = old_evecs[:, kept_ion_indices]
            new_sub = new_evecs[:, :n_shared]

            overlap = new_sub @ old_sub.T
            old_occ = old_structure.occupancies[old_start:old_end]
            new_occ = (overlap ** 2) @ old_occ
            new_struct.occupancies[new_start:new_end] = new_occ

        return new_struct


# ============================================================================
# IonChainFidelityModel — canonical fidelity formula (single copy)
# ============================================================================

class IonChainFidelityModel:
    r"""Canonical gate fidelity model for trapped-ion chains.

    Supports two tiers of physics, selected automatically:

    **Tier 1 — Scalar formula (default, no physical properties):**

    .. math::

        F = 1 - (\dot{n} \cdot t_{\text{gate}}
              + A \cdot \frac{N}{\ln N} \cdot (2\bar{n} + 1))

    This uses only ``CalibrationConstants`` fields and is the same
    formula from arXiv:2004.04706 p. 7.  Called via
    :meth:`gate_fidelity`, :meth:`ms_gate_fidelity`, and
    :meth:`single_qubit_gate_fidelity`.

    **Tier 2 — Mode-resolved formula (opt-in, needs TrapParameters):**

    .. math::

        1 - F_{\text{MS}} \propto \sum_m
            \eta_m^{(i)2}\, \eta_m^{(j)2}\, (2\bar{n}_m + 1)

    where :math:`\eta_m^{(i)}` is the Lamb-Dicke parameter for ion
    *i* in mode *m*, computed from ion mass and laser wavevector.
    Called via :meth:`ms_gate_fidelity_mode_resolved`.  If no
    ``TrapParameters`` is available (``trap=None``), this method
    **automatically falls back to the Tier-1 scalar formula**.

    Parameters
    ----------
    calibration : CalibrationConstants
        Calibration data.  Uses ``DEFAULT_CALIBRATION`` if not provided.
    """

    def __init__(self, calibration: Optional[CalibrationConstants] = None):
        self.cal = calibration or DEFAULT_CALIBRATION

    def gate_fidelity(
        self,
        chain_length: int,
        motional_quanta: float = 0.0,
        *,
        is_two_qubit: bool = True,
    ) -> float:
        """Compute gate fidelity from the scalar physics formula.

        .. note:: **Tier 1** — no physical properties required.

        F = 1 − (heating_rate × t_gate + A × N/ln(N) × (2·n̄ + 1))

        Parameters
        ----------
        chain_length : int
            Number of ions in the trap (N).
        motional_quanta : float
            Accumulated motional quanta (n̄).  This is a *scalar*
            total across all modes.  For per-mode detail, use
            :meth:`ms_gate_fidelity_mode_resolved` (Tier 2).
        is_two_qubit : bool
            True → MS gate time (40 μs); False → 1Q gate time (5 μs).

        Returns
        -------
        float
            Gate fidelity clamped to [0, 1].
        """
        t_gate = self.cal.ms_gate_time if is_two_qubit else self.cal.single_qubit_gate_time
        n_over_ln = (
            chain_length / np.log(chain_length) if chain_length >= 2 else 1.0
        )
        infidelity = (
            self.cal.heating_rate * t_gate
            + self.cal.fidelity_scaling_A * n_over_ln * (2.0 * motional_quanta + 1.0)
        )
        return max(0.0, min(1.0, 1.0 - infidelity))

    def ms_gate_fidelity(
        self,
        chain_length: int,
        motional_quanta: float = 0.0,
    ) -> float:
        """MS (two-qubit) gate fidelity.  Convenience wrapper.

        .. note:: **Tier 1** — scalar formula; no physical properties.
        """
        return self.gate_fidelity(chain_length, motional_quanta, is_two_qubit=True)

    def single_qubit_gate_fidelity(
        self,
        chain_length: int,
        motional_quanta: float = 0.0,
    ) -> float:
        """Single-qubit gate fidelity.  Convenience wrapper.

        .. note:: **Tier 1** — scalar formula; no physical properties.
        """
        return self.gate_fidelity(chain_length, motional_quanta, is_two_qubit=False)

    def dephasing_fidelity(self, duration: float) -> float:
        """Dephasing fidelity: F = 1 − (1 − exp(−t/T2))/2.

        .. note:: **Tier 1** — uses only ``CalibrationConstants.t2_time``.
        """
        if self.cal.t2_time <= 0 or duration <= 0:
            return 1.0
        return 1.0 - (1.0 - np.exp(-duration / self.cal.t2_time)) / 2.0

    def measurement_fidelity(self) -> float:
        """Measurement fidelity: F = 1 − measurement_infidelity.

        .. note:: **Tier 1** — uses only ``CalibrationConstants.measurement_infidelity``.
        """
        return 1.0 - self.cal.measurement_infidelity

    def reset_fidelity(self) -> float:
        """Reset fidelity: F = 1 − reset_infidelity.

        .. note:: **Tier 1** — uses only ``CalibrationConstants.reset_infidelity``.
        """
        return 1.0 - self.cal.reset_infidelity

    # ------------------------------------------------------------------
    # Mode-resolved fidelity (uses Lamb-Dicke parameters)
    # ------------------------------------------------------------------

    def ms_gate_fidelity_mode_resolved(
        self,
        ion_i: int,
        ion_j: int,
        mode_snapshot: "ModeSnapshot",
        trap: Optional[TrapParameters] = None,
    ) -> float:
        r"""Mode-resolved MS gate infidelity.

        .. note:: **Tier 2** — requires a ``TrapParameters`` (with
           ``IonSpecies``) for the Lamb-Dicke matrix.  If ``trap`` is
           ``None`` **and** ``self.cal.trap`` is ``None``, this method
           **falls back to the Tier-1 scalar formula** using
           ``mode_snapshot.scalar_nbar`` and ``mode_snapshot.n_ions``.

        When Tier 2 is available:

        .. math::

            1 - F \propto \sum_m
                \eta_m^{(i)2}\,\eta_m^{(j)2}\,(2\bar{n}_m + 1)

        This captures how gate error depends on *which* modes are hot,
        not just total n̄.

        Parameters
        ----------
        ion_i, ion_j : int
            Ion indices for the two-qubit gate.
        mode_snapshot : ModeSnapshot
            Current crystal mode state with per-mode occupancies.
        trap : TrapParameters, optional
            Trap parameters for Lamb-Dicke computation.  Uses
            ``self.cal.trap`` if not ``None``.  If both are ``None``,
            falls back to scalar formula.

        Returns
        -------
        float
            Gate fidelity in [0, 1].
        """
        trap = trap or self.cal.trap

        # ----- Tier-1 fallback: no physical properties available -----
        if trap is None:
            return self.ms_gate_fidelity(
                chain_length=mode_snapshot.n_ions,
                motional_quanta=mode_snapshot.scalar_nbar,
            )

        # ----- Tier-2: mode-resolved with Lamb-Dicke parameters ------
        infidelity = self.cal.heating_rate * self.cal.ms_gate_time  # baseline

        ld_matrix = trap.lamb_dicke_matrix(
            mode_snapshot.mode_frequencies, mode_snapshot.eigenvectors
        )
        for m in range(len(mode_snapshot.mode_frequencies)):
            eta_i = ld_matrix[m, ion_i] if ion_i < ld_matrix.shape[1] else 0.0
            eta_j = ld_matrix[m, ion_j] if ion_j < ld_matrix.shape[1] else 0.0
            n_bar_m = mode_snapshot.occupancies[m]
            infidelity += (
                self.cal.fidelity_scaling_A
                * eta_i ** 2 * eta_j ** 2
                * (2 * n_bar_m + 1)
            )

        return max(0.0, min(1.0, 1.0 - infidelity))

    def transport_phase_error(
        self,
        distance_um: float,
        velocity_um_per_us: float = 1.0,
        electric_field_noise_V_per_m: float = 1e-2,
    ) -> float:
        r"""Estimate Z-error probability from transport-induced phase.

        .. note:: **Tier 1** — simplified model.  A Tier-2 version
           would integrate the AC Stark shift along the actual
           transport path using the ion's charge-to-mass ratio.

        During shuttling, stray electric fields cause differential
        phase accumulation.  Returns dephasing probability.

        Parameters
        ----------
        distance_um : float
            Transport distance in micrometres.
        velocity_um_per_us : float
            Ion velocity during transport.
        electric_field_noise_V_per_m : float
            RMS electric field noise amplitude.

        Returns
        -------
        float
            Phase-error probability in [0, 0.5].
        """
        duration_us = distance_um / velocity_um_per_us if velocity_um_per_us > 0 else 0.0
        # Simplified: phase noise ~ E_noise * distance * charge / (hbar * velocity)
        # For a proper model, integrate the AC Stark shift along the path
        p_phase = min(0.5, electric_field_noise_V_per_m * distance_um * 1e-6 * duration_us * 1e-6)
        return p_phase


# Module-level convenience instance
DEFAULT_FIDELITY_MODEL = IonChainFidelityModel()


# ============================================================================
# Snapshot collection helper (extracted from routing.py)
# ============================================================================

# =============================================================================
# Legacy timing / heating-rate dataclasses — thin wrappers over
# CalibrationConstants for backward compatibility.
# =============================================================================

@dataclass(frozen=True)
class TrappedIonTimings:
    """Timing constants for trapped ion operations (microseconds).

    .. deprecated::
        Use :attr:`CalibrationConstants` fields directly.  This class
        exists only for backward compatibility with old code that
        imported ``DEFAULT_TIMINGS``.
    """
    splitting_time: float = DEFAULT_CALIBRATION.split_time * 1e6
    merging_time: float = DEFAULT_CALIBRATION.merge_time * 1e6
    junction_time: float = DEFAULT_CALIBRATION.junction_time * 1e6
    linear_move_time: float = DEFAULT_CALIBRATION.shuttle_time * 1e6
    one_qubit_gate: float = DEFAULT_CALIBRATION.single_qubit_gate_time * 1e6
    ms_gate: float = DEFAULT_CALIBRATION.ms_gate_time * 1e6
    measurement: float = DEFAULT_CALIBRATION.measurement_time * 1e6
    reset: float = DEFAULT_CALIBRATION.reset_time * 1e6
    recooling_time: float = DEFAULT_CALIBRATION.recool_time * 1e6


@dataclass(frozen=True)
class HeatingRates:
    """Heating rates for different operations (quanta/operation).

    .. deprecated::
        Use :attr:`CalibrationConstants` transport_heating dict directly.
    """
    split: float = DEFAULT_CALIBRATION.transport_heating["split"]
    merge: float = DEFAULT_CALIBRATION.transport_heating["merge"]
    junction_crossing: float = DEFAULT_CALIBRATION.transport_heating["junction"]
    linear_move: float = DEFAULT_CALIBRATION.transport_heating["move"]
    background_rate: float = DEFAULT_CALIBRATION.heating_rate


DEFAULT_TIMINGS = TrappedIonTimings()
DEFAULT_HEATING_RATES = HeatingRates()


def collect_mode_snapshots(
    architecture: Any,
    qubit_ids: Dict[int, float],
) -> Dict[int, ModeSnapshot]:
    """Collect ModeSnapshots for all qubits from their parent traps.

    Iterates over ManipulationTraps in the architecture, takes a
    snapshot of each trap's mode structure, and maps every qubit in
    that trap to the snapshot.

    Parameters
    ----------
    architecture
        The trapped-ion architecture (e.g. QCCDArch / QCCDWiseArch).
    qubit_ids : Dict[int, float]
        Map of qubit IDs to track (e.g. ``motional_quanta`` dict).
        Only qubits present in this map will appear in the result.

    Returns
    -------
    Dict[int, ModeSnapshot]
        Maps qubit_idx → ModeSnapshot for each qubit whose parent
        trap has mode tracking enabled.
    """
    from .qccd_nodes import ManipulationTrap as _MT

    result: Dict[int, ModeSnapshot] = {}
    try:
        # Walk trap nodes in the architecture
        traps = getattr(architecture, 'traps', [])
        for trap in traps:
            if isinstance(trap, _MT) and trap.mode_structure is not None:
                snap = trap.mode_structure.snapshot()
                for ion in trap.ions:
                    if hasattr(ion, 'idx') and ion.idx in qubit_ids:
                        result[ion.idx] = snap
    except Exception:
        pass  # Mode tracking not available; degrade gracefully
    return result
