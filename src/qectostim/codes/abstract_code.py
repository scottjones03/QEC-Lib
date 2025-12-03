# src/qec_to_stim/codes/abstract_code.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .abstract_css import CSSCode

PauliString = Dict[int, str]  # e.g. {0: 'X', 3: 'Z'} means X on qubit0, Z on qubit3
Coord = Tuple[float, ...]  # D-dimensional coordinates


@dataclass
class CellEmbedding:
    """Embedding of cells of a given grade into D-dimensional space."""
    grade: int
    coords: List[Coord]   # length = dim(C_grade)


class Code(ABC):
    """
    Abstract base class for ANY quantum code (stabilizer, subsystem, homological, etc.).
    This is the interface Experiments talk to.
    """

    @property
    @abstractmethod
    def n(self) -> int:
        """Number of physical qubits."""

    @property
    @abstractmethod
    def k(self) -> int:
        """Number of logical qubits."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def metadata(self) -> Dict[str, Any]:
        """Arbitrary metadata associated with the code instance."""
        return getattr(self, "_metadata", {})

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        self._metadata = value

    # --- Logical operators ---

    @abstractmethod
    def logical_x_ops(self) -> List[PauliString]:
        """Logical X operators (one per logical qubit)."""

    @abstractmethod
    def logical_z_ops(self) -> List[PauliString]:
        """Logical Z operators (one per logical qubit)."""

    # --- Stabilizers / gauge ---

    @abstractmethod
    def stabilizers(self) -> List[PauliString]:
        """List of stabilizer generators as Pauli strings."""

    def gauge_ops(self) -> List[PauliString]:
        """Optional: gauge operators if this is a subsystem code."""
        return []

    # --- Geometry / metadata (optional but useful) ---

    def qubit_coords(self) -> Optional[List[Tuple[float, float]]]:
        """
        Optional 2D embedding for visualization / nearest-neighbour constraints.
        None if not defined.
        """
        return None

    # --- Homological / CSS-specific accessors (may be overridden) ---

    def as_css(self) -> Optional["CSSCode"]:
        """Return self as a CSSCode if applicable, else None."""
        return None

    def extra_metadata(self) -> Dict[str, Any]:
        """Arbitrary metadata for advanced use."""
        return {}
