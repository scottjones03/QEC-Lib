from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .abstract_code import Code, CellEmbedding, PauliString
from .complexes.chain_complex import ChainComplex

Coord2D = Tuple[float, float]




class HomologicalCode(Code, ABC):
    """
    Base class for codes defined via chain complexes.
    """

    def __init__(
        self,
        complex: ChainComplex,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(metadata=metadata)
        self._complex = complex
        self._metadata = metadata or {}

    @property
    def chain_complex(self) -> ChainComplex:
        return self._complex

    @property
    def qubit_grade(self) -> int:
        return self._complex.qubit_grade

    @property
    def max_grade(self) -> int:
        return self._complex.max_grade

    @abstractmethod
    def build_stabilizers(self) -> None:
        """
        Populate whatever CSS/stabilizer data this code exposes
        (e.g. Hx, Hz, metachecks, etc.) from the chain complex.

        Called by subclasses during __init__.
        """
        ...



class TopologicalCode(HomologicalCode):
    """
    A homological code with a geometric/topological embedding.

    This is what allows a homological tensor product of two TopologicalCodes
    to produce another TopologicalCode with a derived dimension and cell
    layout (e.g. dim = dimA + dimB).
    """

    def __init__(
        self,
        complex: ChainComplex,
        embeddings: Dict[int, CellEmbedding],
        dim: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(complex=complex, metadata=metadata)
        self._embeddings = embeddings
        self._dim = dim

    @property
    def dimension(self) -> int:
        """Ambient topological dimension (e.g. 2 for surface, 4 for 4D code)."""
        return self._dim

    def cell_coords(self, grade: int) -> List[Coord2D]:
        return self._embeddings[grade].coords

    def data_qubit_coords(self) -> List[Coord2D]:
        """Coordinates for the chain group that holds qubits."""
        return self.cell_coords(self.qubit_grade)
