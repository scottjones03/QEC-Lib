# src/qectostim/experiments/hardware_simulation/trapped_ion/transport.py
"""
Transport operations for QCCD trapped-ion architectures.

Implements the physical transport operations that move ions between traps
via crossings and junctions.  Each operation carries calibrated timing and
motional-heating constants derived from:

* TABLE I  — https://arxiv.org/pdf/2004.04706   (operation durations)
* TABLE IV — https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
  (heating rates, gate fidelities)

The classes here are *data-oriented*: they compute time / heating / fidelity
without mutating architecture state.  State mutation (ion movement) is
handled by the router that consumes these objects.

Hierarchy
---------
TransportOp (abstract base)
├── Split       — detach edge ion from trap into crossing
├── Merge       — absorb crossing ion into trap
├── Move        — shuttle ion along a crossing channel
├── JunctionCrossing — move ion through a junction node
├── CrystalRotation  — reverse ion order in a trap
├── CoolingOp   — sympathetic cooling in a trap
└── ParallelOp  — wrapper that time-aligns concurrent ops
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
        Ion,
        QCCDNode,
        ManipulationTrap,
        StorageTrap,
        Junction,
        Crossing,
        ModeStructure,
    )


# ============================================================================
# Physical constants — delegated to physics.py (single source of truth)
# ============================================================================

from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
    DEFAULT_CALIBRATION as _CAL,
    DEFAULT_FIDELITY_MODEL as _FIDELITY,
)

# Backward-compat aliases (read-only, for any external code referencing these)
BACKGROUND_HEATING_RATE: float = _CAL.heating_rate
FIDELITY_SCALING_A: float = _CAL.fidelity_scaling_A
T2: float = _CAL.t2_time


# ============================================================================
# Operation type enum (mirrors old Operations enum)
# ============================================================================

class TransportOpType(enum.Enum):
    """Enumeration of all transport / physical operations."""
    SPLIT = enum.auto()
    MOVE = enum.auto()
    MERGE = enum.auto()
    GATE_SWAP = enum.auto()
    CRYSTAL_ROTATION = enum.auto()
    ONE_QUBIT_GATE = enum.auto()
    TWO_QUBIT_MS_GATE = enum.auto()
    JUNCTION_CROSSING = enum.auto()
    MEASUREMENT = enum.auto()
    QUBIT_RESET = enum.auto()
    RECOOLING = enum.auto()
    PARALLEL = enum.auto()
    GLOBAL_RECONFIG = enum.auto()


# ============================================================================
# Abstract base
# ============================================================================

class TransportOp(abc.ABC):
    """Abstract base class for a single transport / physical operation.

    Subclasses provide calibrated ``time_s``, ``heating_rate`` and
    ``heating_quanta`` values.  The router / execution planner reads
    these to accumulate motional energy and elapsed time.

    The ``heat_mode_structure`` method distributes the scalar
    ``heating_quanta`` across the 3N normal modes of a mode structure,
    weighting by the electric-field noise spectrum S_E(ω) ∝ 1/ω^α.
    """

    op_type: TransportOpType

    @property
    @abc.abstractmethod
    def time_s(self) -> float:
        """Wall-clock duration of this operation in seconds."""
        ...

    @property
    @abc.abstractmethod
    def heating_rate(self) -> float:
        """Motional-heating rate in quanta/second for this operation."""
        ...

    @property
    def heating_quanta(self) -> float:
        """Total motional quanta deposited by this operation."""
        return self.heating_rate * self.time_s

    @property
    def dephasing_fidelity(self) -> float:
        """Dephasing fidelity loss: F_deph = 1 - (1 - exp(-t/T2)) / 2."""
        return _FIDELITY.dephasing_fidelity(self.time_s)

    def heat_mode_structure(
        self,
        mode_structure: "ModeStructure",
        noise_exponent: float = 1.0,
    ) -> None:
        """Distribute this operation's heating across the 3N normal modes.

        Physical picture: every transport operation (split, merge, move,
        junction crossing) takes some wall-clock time during which the
        ions are exposed to fluctuating electric fields from the trap
        electrodes.  These random kicks inject vibrational energy
        (motional quanta) into the ion crystal.

        The key insight is that heating is **mode-dependent**: the
        COM mode (all ions moving together) absorbs the most energy
        because electrode noise is nearly uniform across the small
        crystal.  Higher-frequency modes — where ions move in opposite
        directions — are barely excited by a uniform kick.

        Calls ``ModeStructure.heat_modes()`` with this operation's
        ``heating_quanta`` as the COM-mode heating amount.  The mode
        structure distributes this across all modes according to:

            Δn̄_m = Δn̄_COM × (ω_COM / ω_m)^α

        The scalar ``heating_quanta`` corresponds to the COM-mode
        contribution.  Higher-frequency modes receive less heating
        because the electric field noise spectral density S_E(ω)
        falls off with frequency.

        Parameters
        ----------
        mode_structure : ModeStructure
            The trap's current mode structure (modified in-place).
        noise_exponent : float
            Noise spectral exponent α.  Default 1.0 (1/f noise).

        References
        ----------
        Brownnutt et al., Rev. Mod. Phys. 87, 1419 (2015), §III.A.
        """
        if mode_structure is not None and self.heating_quanta > 0:
            mode_structure.heat_modes(self.heating_quanta, noise_exponent)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"t={self.time_s * 1e6:.1f}μs, "
            f"ΔQ={self.heating_quanta:.4e})"
        )


# ============================================================================
# Concrete transport operations
# ============================================================================

class Split(TransportOp):
    """Detach edge ion from a trap, placing it in a crossing.

    Physics: 80 μs, 6 quanta/s heating.
    Ref: TABLE I — https://arxiv.org/pdf/2004.04706
    """
    op_type = TransportOpType.SPLIT
    SPLITTING_TIME: float = _CAL.split_time
    HEATING_RATE: float = _CAL.split_heating_rate

    def __init__(
        self,
        trap_idx: int,
        crossing_idx: int,
        ion_idx: int,
    ) -> None:
        self.trap_idx = trap_idx
        self.crossing_idx = crossing_idx
        self.ion_idx = ion_idx

    @property
    def time_s(self) -> float:
        return self.SPLITTING_TIME

    @property
    def heating_rate(self) -> float:
        return self.HEATING_RATE


class Merge(TransportOp):
    """Absorb a crossing ion into a trap.

    Physics: 80 μs, 6 quanta/s heating.
    Ref: TABLE I — https://arxiv.org/pdf/2004.04706
    """
    op_type = TransportOpType.MERGE
    MERGING_TIME: float = _CAL.merge_time
    HEATING_RATE: float = _CAL.merge_heating_rate

    def __init__(
        self,
        trap_idx: int,
        crossing_idx: int,
        ion_idx: int,
    ) -> None:
        self.trap_idx = trap_idx
        self.crossing_idx = crossing_idx
        self.ion_idx = ion_idx

    @property
    def time_s(self) -> float:
        return self.MERGING_TIME

    @property
    def heating_rate(self) -> float:
        return self.HEATING_RATE


class Move(TransportOp):
    """Shuttle an ion along a crossing channel (between nodes).

    Physics: 5 μs, 0.1 quanta/s heating.
    Ref: TABLE I — https://arxiv.org/pdf/2004.04706
    """
    op_type = TransportOpType.MOVE
    MOVING_TIME: float = _CAL.shuttle_time
    HEATING_RATE: float = _CAL.shuttle_heating_rate

    def __init__(
        self,
        crossing_idx: int,
        ion_idx: int,
    ) -> None:
        self.crossing_idx = crossing_idx
        self.ion_idx = ion_idx

    @property
    def time_s(self) -> float:
        return self.MOVING_TIME

    @property
    def heating_rate(self) -> float:
        return self.HEATING_RATE


class JunctionCrossing(TransportOp):
    """Move an ion through a junction node.

    Physics: 50 μs, 3 quanta/s heating.
    Ref: TABLE I — https://arxiv.org/pdf/2004.04706
    """
    op_type = TransportOpType.JUNCTION_CROSSING
    CROSSING_TIME: float = _CAL.junction_time
    CROSSING_HEATING: float = _CAL.junction_heating_rate

    def __init__(
        self,
        junction_idx: int,
        crossing_idx: int,
        ion_idx: int,
    ) -> None:
        self.junction_idx = junction_idx
        self.crossing_idx = crossing_idx
        self.ion_idx = ion_idx

    @property
    def time_s(self) -> float:
        return self.CROSSING_TIME

    @property
    def heating_rate(self) -> float:
        return self.CROSSING_HEATING


class CrystalRotation(TransportOp):
    """Reverse the ion order inside a trap.

    Physics: 42 μs, 0.3 quanta/s heating.
    Ref: TABLE IV — https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    """
    op_type = TransportOpType.CRYSTAL_ROTATION
    ROTATION_TIME: float = _CAL.rotation_time
    ROTATION_HEATING: float = _CAL.rotation_heating_rate

    def __init__(self, trap_idx: int) -> None:
        self.trap_idx = trap_idx

    @property
    def time_s(self) -> float:
        return self.ROTATION_TIME

    @property
    def heating_rate(self) -> float:
        return self.ROTATION_HEATING


class CoolingOp(TransportOp):
    """Sympathetic re-cooling in a trap.

    Physics: 400 μs, 0.1 quanta/s heating (net cooling dominates).
    Ref: TABLE IV — https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330

    Note: The old code had a bug ``400-6`` (=394 s) instead of ``400e-6``.
    We use the correct 400 μs here.
    """
    op_type = TransportOpType.RECOOLING
    COOLING_TIME: float = _CAL.recool_time
    COOLING_HEATING: float = _CAL.cooling_heating_rate

    def __init__(self, trap_idx: int) -> None:
        self.trap_idx = trap_idx

    @property
    def time_s(self) -> float:
        return self.COOLING_TIME

    @property
    def heating_rate(self) -> float:
        return self.COOLING_HEATING


# ============================================================================
# Composite / derived operations
# ============================================================================

class PhysicalCrossingSwap(TransportOp):
    """Ion swap via a physical crossing (junction-mediated).

    Functionally a JunctionCrossing but with a longer time budget.
    Physics: 100 μs, 3 quanta/s heating.
    Ref: TABLE I — https://arxiv.org/pdf/2004.04706
    """
    op_type = TransportOpType.JUNCTION_CROSSING
    SWAP_TIME: float = _CAL.crossing_swap_time
    SWAP_HEATING: float = _CAL.crossing_swap_heating_rate

    def __init__(
        self,
        junction_idx: int,
        crossing_idx: int,
        ion_idx: int,
    ) -> None:
        self.junction_idx = junction_idx
        self.crossing_idx = crossing_idx
        self.ion_idx = ion_idx

    @property
    def time_s(self) -> float:
        return self.SWAP_TIME

    @property
    def heating_rate(self) -> float:
        return self.SWAP_HEATING


class ParallelOp(TransportOp):
    """Wrapper for operations executed in parallel.

    Time = max(sub-op times), heating = sum(sub-op heatings).
    The caller is responsible for ensuring the sub-ops do not
    conflict (no shared ions / crossings).
    """
    op_type = TransportOpType.PARALLEL

    def __init__(self, sub_ops: Sequence[TransportOp]) -> None:
        self.sub_ops: List[TransportOp] = list(sub_ops)

    @property
    def time_s(self) -> float:
        if not self.sub_ops:
            return 0.0
        return max(op.time_s for op in self.sub_ops)

    @property
    def heating_rate(self) -> float:
        """Effective rate: total heating / parallel time."""
        t = self.time_s
        if t == 0.0:
            return 0.0
        return self.heating_quanta / t

    @property
    def heating_quanta(self) -> float:
        """Sum of all sub-operation heating contributions."""
        return sum(op.heating_quanta for op in self.sub_ops)


# ============================================================================
# Qubit-level gate operations (physics data only, no state mutation)
# ============================================================================

@dataclass
class GatePhysics:
    """Physics parameters for a quantum gate at a specific chain state.

    This is a *snapshot* — the fidelity depends on chain length and motional
    quanta at the time the gate is executed.
    """
    gate_type: str                  # "1Q" or "2Q_MS"
    chain_length: int               # ions in the trap
    motional_quanta: float          # n̄ at time of execution
    ion_indices: Tuple[int, ...]    # logical qubit indices involved

    # -- 1Q gate constants --
    SINGLE_QUBIT_TIME: float = _CAL.single_qubit_gate_time
    # -- 2Q MS gate constants --
    MS_GATE_TIME: float = _CAL.ms_gate_time

    @property
    def time_s(self) -> float:
        return self.SINGLE_QUBIT_TIME if self.gate_type == "1Q" else self.MS_GATE_TIME

    @property
    def fidelity(self) -> float:
        """Compute gate fidelity from chain length and motional quanta.

        Delegates to ``IonChainFidelityModel`` in physics.py.
        F = 1 − (heating_rate × t_gate + A × N/ln(N) × (2·n̄ + 1))
        """
        return _FIDELITY.gate_fidelity(
            self.chain_length,
            self.motional_quanta,
            is_two_qubit=(self.gate_type != "1Q"),
        )

    @property
    def dephasing_fidelity(self) -> float:
        if self.gate_type == "1Q":
            return 1.0  # single-qubit dephasing negligible
        return _FIDELITY.dephasing_fidelity(self.time_s)


@dataclass
class MeasurementPhysics:
    """Physics for a measurement operation."""
    MEASUREMENT_TIME: float = _CAL.measurement_time
    INFIDELITY: float = _CAL.measurement_infidelity

    @property
    def time_s(self) -> float:
        return self.MEASUREMENT_TIME

    @property
    def fidelity(self) -> float:
        return 1.0 - self.INFIDELITY


@dataclass
class ResetPhysics:
    """Physics for a qubit reset operation."""
    RESET_TIME: float = _CAL.reset_time
    INFIDELITY: float = _CAL.reset_infidelity

    @property
    def time_s(self) -> float:
        return self.RESET_TIME

    @property
    def fidelity(self) -> float:
        return 1.0 - self.INFIDELITY


# ============================================================================
# GateSwap (3 MS gates to swap two ions)
# ============================================================================

@dataclass
class GateSwapPhysics:
    """Physics for a gate swap = 3 consecutive MS gates.

    Ref: Fig 5 — https://arxiv.org/pdf/2004.04706
    Effectively 3 × MS gate time, fidelity = product of 3 MS fidelities.
    """
    chain_length: int
    motional_quanta: float
    ion1_idx: int
    ion2_idx: int

    @property
    def time_s(self) -> float:
        return 3.0 * GatePhysics.MS_GATE_TIME

    @property
    def fidelity(self) -> float:
        """Product of 3 MS gate fidelities."""
        single_ms = GatePhysics(
            gate_type="2Q_MS",
            chain_length=self.chain_length,
            motional_quanta=self.motional_quanta,
            ion_indices=(self.ion1_idx, self.ion2_idx),
        )
        return single_ms.fidelity ** 3

    @property
    def dephasing_fidelity(self) -> float:
        single_ms = GatePhysics(
            gate_type="2Q_MS",
            chain_length=self.chain_length,
            motional_quanta=self.motional_quanta,
            ion_indices=(self.ion1_idx, self.ion2_idx),
        )
        return single_ms.dephasing_fidelity ** 3


# ============================================================================
# WISE aggregate reconfiguration constants
# ============================================================================

# These are pre-derived from the individual transport operations above
# and are used by the SAT router for efficiency.
# row swap = Split(80μs,6) + Move(5μs,0.1) + CrystalRot(42μs,0.3)
#          + Merge(80μs,6) + Move(5μs,0.1)
ROW_SWAP_TIME_S: float = (
    Split.SPLITTING_TIME + Move.MOVING_TIME
    + CrystalRotation.ROTATION_TIME
    + Merge.MERGING_TIME + Move.MOVING_TIME
)  # = 212 μs

ROW_SWAP_HEATING: float = (
    Split.SPLITTING_TIME * Split.HEATING_RATE
    + Move.MOVING_TIME * Move.HEATING_RATE
    + CrystalRotation.ROTATION_TIME * CrystalRotation.ROTATION_HEATING
    + Merge.MERGING_TIME * Merge.HEATING_RATE
    + Move.MOVING_TIME * Move.HEATING_RATE
)  # ≈ 9.734e-4 quanta per ion

# col swap = 6 × JunctionCrossing(50μs,3) + Move(5μs,0.1)
COL_SWAP_TIME_S: float = (
    6.0 * JunctionCrossing.CROSSING_TIME + Move.MOVING_TIME
)  # = 305 μs

COL_SWAP_HEATING: float = (
    6.0 * JunctionCrossing.CROSSING_TIME * JunctionCrossing.CROSSING_HEATING
    + Move.MOVING_TIME * Move.HEATING_RATE
)  # ≈ 9.005e-4 quanta per ion


# ============================================================================
# Transport sequence builder — generates Split→Move→Merge for a hop
# ============================================================================

def build_hop_operations(
    source_idx: int,
    target_idx: int,
    crossing_idx: int,
    ion_idx: int,
    *,
    source_is_junction: bool = False,
    target_is_junction: bool = False,
    needs_rotation: bool = False,
) -> List[TransportOp]:
    """Build the sequence of transport ops for one hop across a crossing.

    The canonical sequence for trap-to-trap is:
        [CrystalRotation?] → Split → Move → Merge

    When a junction is involved, JunctionCrossing replaces
    Split (at source) or Merge (at target).

    Parameters
    ----------
    source_idx : int
        Index of the source node (trap or junction).
    target_idx : int
        Index of the destination node.
    crossing_idx : int
        Index of the crossing connecting source and target.
    ion_idx : int
        Index of the ion being moved.
    source_is_junction : bool
        True if the source node is a Junction.
    target_is_junction : bool
        True if the target node is a Junction.
    needs_rotation : bool
        True if the source trap has exactly 1 ion and needs a
        crystal rotation before splitting.

    Returns
    -------
    list[TransportOp]
        Ordered sequence of transport operations for this hop.
    """
    ops: List[TransportOp] = []

    # Optional rotation when source trap has 1 ion
    if needs_rotation and not source_is_junction:
        ops.append(CrystalRotation(trap_idx=source_idx))

    # Detach ion from source
    if source_is_junction:
        ops.append(JunctionCrossing(
            junction_idx=source_idx,
            crossing_idx=crossing_idx,
            ion_idx=ion_idx,
        ))
    else:
        ops.append(Split(
            trap_idx=source_idx,
            crossing_idx=crossing_idx,
            ion_idx=ion_idx,
        ))

    # Move through crossing
    ops.append(Move(crossing_idx=crossing_idx, ion_idx=ion_idx))

    # Absorb ion into target
    if target_is_junction:
        ops.append(JunctionCrossing(
            junction_idx=target_idx,
            crossing_idx=crossing_idx,
            ion_idx=ion_idx,
        ))
    else:
        ops.append(Merge(
            trap_idx=target_idx,
            crossing_idx=crossing_idx,
            ion_idx=ion_idx,
        ))

    return ops


def total_transport_time(ops: Sequence[TransportOp]) -> float:
    """Total sequential time of a list of transport operations in seconds."""
    return sum(op.time_s for op in ops)


def total_transport_heating(ops: Sequence[TransportOp]) -> float:
    """Total motional quanta deposited by a sequence of transport operations."""
    return sum(op.heating_quanta for op in ops)
