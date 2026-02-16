"""
Type-safe dataclass wrappers for old/ node types.

Provides immutable configuration dataclasses with .build() factory
methods that create old/ mutable objects. This gives the benefits of
validated, frozen configs without changing the working old/ classes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .qccd_nodes import (
    Ion,
    QubitIon,
    Trap,
    ManipulationTrap,
    Junction,
    Crossing,
    QCCDNode,
    QCCDWiseArch,
)
from .qccd_arch import QCCDArch


@dataclass(frozen=True)
class IonConfig:
    """Immutable configuration for an ion.

    Parameters
    ----------
    idx : int
        Unique ion index.
    x : int
        X position.
    y : int
        Y position.
    color : str
        Display color.
    label : str
        Display label (e.g. "D0" for data qubit 0).
    is_qubit : bool
        If True, creates a QubitIon; otherwise a plain Ion.
    """

    idx: int
    x: int
    y: int
    color: str = "lightblue"
    label: str = "Q"
    is_qubit: bool = True

    def build(self, parent: Optional[QCCDNode] = None) -> Ion:
        """Create a mutable Ion/QubitIon from this config."""
        if self.is_qubit:
            ion = QubitIon(color=self.color, label=self.label)
        else:
            ion = Ion(color=self.color, label=self.label)
        ion.set(self.idx, self.x, self.y, parent)
        return ion


@dataclass(frozen=True)
class TrapConfig:
    """Immutable configuration for a manipulation trap.

    Parameters
    ----------
    idx : int
        Unique trap index.
    x : float
        X position.
    y : float
        Y position.
    capacity : int
        Maximum number of ions.
    """

    idx: int
    x: float
    y: float
    capacity: int = 4

    def build(self) -> ManipulationTrap:
        """Create a mutable ManipulationTrap from this config."""
        trap = ManipulationTrap()
        trap.set(self.idx, self.x, self.y)
        return trap


@dataclass(frozen=True)
class WISEGridConfig:
    """Immutable configuration for a WISE architecture grid.

    Parameters
    ----------
    n : int
        Number of rows.
    m : int
        Number of columns.
    k : int
        Ions per trap.
    metadata : dict
        Optional metadata.
    """

    n: int
    m: int
    k: int
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.m < 1:
            raise ValueError(f"m must be >= 1, got {self.m}")
        if self.k < 2:
            raise ValueError(f"k must be >= 2, got {self.k}")

    @property
    def num_traps(self) -> int:
        return self.n * self.m

    @property
    def num_qubits(self) -> int:
        return self.n * self.m * self.k

    def build_wise_arch(self) -> QCCDWiseArch:
        """Create a QCCDWiseArch from this config."""
        return QCCDWiseArch(m=self.m, n=self.n, k=self.k)

    def build_architecture(self) -> QCCDWiseArch:
        """Alias for build_wise_arch()."""
        return self.build_wise_arch()

    def summary(self) -> str:
        """Human-readable summary string."""
        return (
            f"WISE-{self.n}x{self.m}x{self.k} "
            f"({self.num_traps} traps, {self.num_qubits} qubits)"
        )


@dataclass(frozen=True)
class ArchitectureSnapshot:
    """Frozen snapshot of a QCCDArch state for logging/serialization.

    Captures the essential state without holding mutable references.
    """

    num_traps: int
    num_junctions: int
    num_ions: int
    num_crossings: int
    ion_positions: Tuple[Tuple[int, int, int], ...] = ()  # (idx, x, y)

    @classmethod
    def from_arch(cls, arch: QCCDArch) -> "ArchitectureSnapshot":
        """Create a snapshot from a live QCCDArch."""
        ion_positions = tuple(
            (ion.idx, ion.pos[0], ion.pos[1])
            for ion in arch.ions.values()
        )
        return cls(
            num_traps=len(arch._manipulationTraps),
            num_junctions=len(arch._junctions),
            num_ions=len(arch.ions),
            num_crossings=len(arch._crossings),
            ion_positions=ion_positions,
        )
