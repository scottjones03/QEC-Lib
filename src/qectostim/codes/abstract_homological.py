# src/qectostim/codes/abstract_homological.py
"""
Homological code base classes - codes defined via chain complexes.

Class Hierarchy:
    StabilizerCode (from abstract_code.py)
    └── HomologicalCode - Codes defined by chain complexes
        └── TopologicalCode - With geometric embedding

Chain Complex Structure:
    - 2-chain: C1 → C0 (repetition code)
    - 3-chain: C2 → C1 → C0 (2D surface/toric codes)
    - 4-chain: C3 → C2 → C1 → C0 (3D toric codes)
    - 5-chain: C4 → C3 → C2 → C1 → C0 (4D tesseract, hypergraph products)

The chain_length property returns the number of boundary maps + 1.
E.g., a 3-chain has 2 boundary maps (∂2: C2→C1, ∂1: C1→C0).

Code Parameters:
    Determined by chain complex dimensions.  For a complex
    C_n → … → C_0 with qubits on grade g, the number of physical
    qubits is dim(C_g) and the number of logical qubits follows from
    the homology group H_g = ker(∂_g) / im(∂_{g+1}).

Stabiliser Structure:
    Boundary operators ∂_i define stabilisers.  X-stabilisers come from
    im(∂_{g+1}) and Z-stabilisers from im(∂_gᵀ); the condition ∂² = 0
    guarantees that Hx·Hzᵀ = 0.

Raises:
    TypeError
        If a subclass omits the required ``build_stabilizers`` method.
    ValueError
        If the supplied ``ChainComplex`` has inconsistent boundary-map
        dimensions (∂_{k} columns ≠ ∂_{k+1} rows).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .abstract_code import StabilizerCode, CellEmbedding, PauliString

if TYPE_CHECKING:
    from .complexes.chain_complex import ChainComplex

Coord2D = Tuple[float, float]
Coord = Tuple[float, ...]  # D-dimensional coordinates


class HomologicalCode(StabilizerCode, ABC):
    """
    Base class for codes defined via chain complexes.
    
    A homological code is constructed from a chain complex:
        C_n → C_{n-1} → ... → C_1 → C_0
    
    where qubits live on one of the chain groups (typically C_{qubit_grade}).
    
    Key properties:
    - chain_length: Number of chain groups (n+1 for an n-chain)
    - chain_complex: The underlying ChainComplex object
    - qubit_grade: Which chain group holds the physical qubits
    
    CSS codes come from the property that ∂² = 0, giving:
    - X-stabilizers from im(∂_{qubit_grade+1})
    - Z-stabilizers from im(∂_{qubit_grade}^T)
    
    The condition Hx @ Hz.T = 0 follows from ∂² = 0.
    """

    def __init__(
        self,
        complex: "ChainComplex",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a homological code from a chain complex.
        
        Parameters
        ----------
        complex : ChainComplex
            The chain complex defining the code structure.
        metadata : dict, optional
            Arbitrary metadata (distance, name, etc.).

        Raises
        ------
        TypeError
            If *complex* is not a ``ChainComplex`` instance.
        """
        self._complex = complex
        self._metadata = metadata or {}

    @property
    def chain_complex(self) -> "ChainComplex":
        """The underlying chain complex."""
        return self._complex

    @property
    def chain_length(self) -> int:
        """
        Number of chain groups in the complex.
        
        Returns
        -------
        int
            For a complex C_n → C_{n-1} → ... → C_0, returns n+1.
            - 2-chain: 2 (repetition code)
            - 3-chain: 3 (2D toric/surface)
            - 4-chain: 4 (3D toric)
            - 5-chain: 5 (4D tesseract)
        """
        return self._complex.max_grade + 1

    @property
    def qubit_grade(self) -> int:
        """Which chain group holds the physical qubits."""
        return self._complex.qubit_grade

    @property
    def max_grade(self) -> int:
        """Maximum grade in the chain complex."""
        return self._complex.max_grade

    @abstractmethod
    def build_stabilizers(self) -> None:
        """
        Populate stabilizer/CSS data from the chain complex.
        
        Called by subclasses during __init__ to extract:
        - For CSS codes: Hx, Hz matrices from boundary maps
        - For non-CSS codes: the full stabilizer_matrix
        """
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Arbitrary metadata associated with the code instance."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        self._metadata = value


class TopologicalCode(HomologicalCode):
    """
    A homological code with a geometric/topological embedding.
    
    This adds spatial coordinates to the algebraic chain complex structure,
    enabling visualization and physically-motivated noise models.
    
    The ambient dimension (self.dimension) is independent of chain_length:
    - A 2D surface code is embedded in dim=2 with chain_length=3
    - A 3D toric code is embedded in dim=3 with chain_length=4
    - A 4D hypercubic code is embedded in dim=4 with chain_length=5
    
    The homological product of two d1-dimensional and d2-dimensional
    topological codes gives a (d1 + d2)-dimensional code.
    """

    def __init__(
        self,
        complex: "ChainComplex",
        embeddings: Dict[int, CellEmbedding],
        dim: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a topological code with geometric embedding.
        
        Parameters
        ----------
        complex : ChainComplex
            The chain complex defining the code structure.
        embeddings : dict
            Maps grade -> CellEmbedding with coordinates for each cell.
        dim : int
            Ambient geometric dimension (2 for surface codes, 3 for 3D, etc.).
        metadata : dict, optional
            Arbitrary metadata.
        """
        super().__init__(complex=complex, metadata=metadata)
        self._embeddings = embeddings
        self._dim = dim

    @property
    def dimension(self) -> int:
        """
        Ambient topological/geometric dimension.
        
        E.g., 2 for surface codes, 3 for 3D toric, 4 for 4D tesseract.
        """
        return self._dim

    def cell_coords(self, grade: int) -> List[Coord]:
        """Get coordinates for all cells of a given grade."""
        if grade not in self._embeddings:
            return []
        return self._embeddings[grade].coords

    def data_qubit_coords(self) -> List[Coord]:
        """Coordinates for the physical qubits (at qubit_grade)."""
        return self.cell_coords(self.qubit_grade)

    def qubit_coords(self) -> Optional[List[Tuple[float, ...]]]:
        """Override Code.qubit_coords() with actual coordinates."""
        coords = self.data_qubit_coords()
        return coords if coords else None
