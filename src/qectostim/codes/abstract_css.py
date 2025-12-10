# src/qectostim/codes/abstract_css.py
"""
CSS and related code base classes.

Class Hierarchy:
    StabilizerCode (from abstract_code.py)
    ├── HomologicalCode (from abstract_homological.py) - inherits from StabilizerCode
    │   └── TopologicalCode
    └── SubsystemCode
    
    CSSCode(HomologicalCode) - Inherits from HomologicalCode (which inherits from StabilizerCode)
    └── CSSCodeWithComplex(CSSCode) - CSS codes with required chain complex
        ├── TopologicalCSSCode - For 3-chain (2D surface/toric)
        ├── TopologicalCSSCode3D - For 4-chain (3D toric)
        └── TopologicalCSSCode4D - For 5-chain (4D tesseract)
    
    SubsystemCSSCode(SubsystemCode, CSSCode) - Subsystem codes with CSS structure
    
    FloquetCode(StabilizerCode) - Time-dependent stabilizer codes (NOT CSS)

Chain Complex to Code Mapping:
    - 2-chain (C1 → C0): Simple repetition-like codes
    - 3-chain (C2 → C1 → C0): 2D CSS codes (surface, toric, color)
    - 4-chain (C3 → C2 → C1 → C0): 3D CSS codes (3D toric)
    - 5-chain (C4 → C3 → C2 → C1 → C0): 4D CSS codes (tesseract, HGP)

Note: CSSCode now inherits from HomologicalCode only (not StabilizerCode directly)
since HomologicalCode already inherits from StabilizerCode. This avoids MRO issues.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .abstract_code import PauliString, StabilizerCode, SubsystemCode, CellEmbedding
from .abstract_homological import HomologicalCode, TopologicalCode
from .utils import gf2_rank

if TYPE_CHECKING:
    from .complexes.chain_complex import ChainComplex
    from .complexes.css_complex import CSSChainComplex2, CSSChainComplex3, CSSChainComplex4, FiveCSSChainComplex

Coord2D = Tuple[float, float]
Coord = Tuple[float, ...]


class CSSCode(HomologicalCode):
    """
    CSS code based on a chain complex with separate X and Z stabilizers.
    
    A CSS code has the property that all stabilizers are either:
    - Pure X-type: tensor products of X and I only
    - Pure Z-type: tensor products of Z and I only
    
    This arises from the chain complex property ∂² = 0:
    - X stabilizers: rows of Hx (from ∂_{qubit_grade+1})
    - Z stabilizers: rows of Hz (from ∂_{qubit_grade}^T)
    - Commutativity: Hx @ Hz.T = 0 mod 2
    
    Inherits from HomologicalCode (which inherits from StabilizerCode),
    providing both chain complex structure and stabilizer formalism.
    
    The base CSSCode class works with any chain length. Use the specific
    TopologicalCSSCode, TopologicalCSSCode3D, TopologicalCSSCode4D for
    codes with known geometric structure.
    """

    def __init__(
        self,
        hx: np.ndarray,
        hz: np.ndarray,
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a CSS code from parity check matrices.
        
        Parameters
        ----------
        hx : np.ndarray
            X-stabilizer parity check matrix, shape (mx, n).
        hz : np.ndarray
            Z-stabilizer parity check matrix, shape (mz, n).
        logical_x : list of PauliString
            Logical X operators (one per logical qubit).
        logical_z : list of PauliString
            Logical Z operators (one per logical qubit).
        metadata : dict, optional
            Arbitrary metadata (distance, chain_complex, etc.).
        """
        self._hx = np.array(hx, dtype=np.uint8)
        self._hz = np.array(hz, dtype=np.uint8)
        self._logical_x = logical_x
        self._logical_z = logical_z
        self._metadata = metadata or {}
        self._validate_css()

    def _validate_css(self) -> None:
        """Validate CSS constraints: Hx and Hz must commute (Hx @ Hz.T = 0 mod 2)."""
        if self._hx.size == 0 or self._hz.size == 0:
            return
        assert self._hx.shape[1] == self._hz.shape[1], \
            "Hx, Hz must have same number of columns (qubits)"
        comm = (self._hx @ self._hz.T) % 2
        if np.any(comm):
            raise ValueError("Hx Hz^T != 0 mod 2; not a valid CSS code")

    # --- Code interface ---

    @property
    def n(self) -> int:
        """Number of physical qubits."""
        if self._hx.size > 0:
            return self._hx.shape[1]
        elif self._hz.size > 0:
            return self._hz.shape[1]
        return 0

    @property
    def k(self) -> int:
        """Number of logical qubits: k = n - rank(Hx) - rank(Hz) over GF(2)."""
        if self.n == 0:
            return 0
        rank_hx = gf2_rank(self._hx) if self._hx.size > 0 else 0
        rank_hz = gf2_rank(self._hz) if self._hz.size > 0 else 0
        return self.n - rank_hx - rank_hz

    @property
    def logical_x_ops(self) -> List[PauliString]:
        """Logical X operators."""
        return self._logical_x

    @property
    def logical_z_ops(self) -> List[PauliString]:
        """Logical Z operators."""
        return self._logical_z

    # --- StabilizerCode interface ---

    @property
    def stabilizer_matrix(self) -> np.ndarray:
        """
        Return stabilizer generators in symplectic form [X_part | Z_part].
        
        For CSS codes:
        - X stabilizers: [hx | 0]
        - Z stabilizers: [0 | hz]
        
        Returns shape (mx + mz, 2*n).
        """
        n = self.n
        if n == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        
        num_x_stabs = self._hx.shape[0] if self._hx.size > 0 else 0
        num_z_stabs = self._hz.shape[0] if self._hz.size > 0 else 0
        
        stab_mat = np.zeros((num_x_stabs + num_z_stabs, 2 * n), dtype=np.uint8)
        
        if num_x_stabs > 0:
            stab_mat[:num_x_stabs, :n] = self._hx
        if num_z_stabs > 0:
            stab_mat[num_x_stabs:, n:] = self._hz
        
        return stab_mat

    @property
    def is_css(self) -> bool:
        """CSS codes are always CSS by construction."""
        return True

    def as_css(self) -> "CSSCode":
        """Return self (already a CSSCode)."""
        return self

    def stabilizers(self) -> List[PauliString]:
        """Convert Hx/Hz to list of Pauli strings."""
        stabs: List[PauliString] = []
        # X stabilizers from Hx
        for row in self._hx:
            pauli = {i: "X" for i, bit in enumerate(row) if bit}
            stabs.append(pauli)
        # Z stabilizers from Hz
        for row in self._hz:
            pauli = {i: "Z" for i, bit in enumerate(row) if bit}
            stabs.append(pauli)
        return stabs

    # --- CSS-specific properties ---

    @property
    def hx(self) -> np.ndarray:
        """X-stabilizer parity check matrix."""
        return self._hx

    @property
    def hz(self) -> np.ndarray:
        """Z-stabilizer parity check matrix."""
        return self._hz

    @property
    def chain_complex(self) -> Optional["ChainComplex"]:
        """Return chain complex if stored in metadata, else None."""
        return self._metadata.get("chain_complex", None)

    @property
    def chain_length(self) -> int:
        """
        Chain length for CSS codes.
        
        Returns the value from metadata if set, otherwise defaults to 3
        (standard 3-chain for 2D CSS codes).
        """
        if "chain_length" in self._metadata:
            return self._metadata["chain_length"]
        cc = self.chain_complex
        if cc is not None:
            return cc.max_grade + 1
        return 3  # Default for traditional 2D CSS codes

    def build_stabilizers(self) -> None:
        """
        Satisfy the HomologicalCode API.
        
        For CSSCode we already store Hx/Hz in __init__, so this is a no-op.
        Subclasses that build from chain complexes can override.
        """
        pass

    def extra_metadata(self) -> Dict[str, Any]:
        """Return additional metadata."""
        return dict(self._metadata)


class CSSCodeWithComplex(CSSCode):
    """
    CSS code with a required chain complex.
    
    This is the base class for all CSS codes that have an associated chain complex.
    The chain complex determines the structure of the code:
    - Hx and Hz are derived from the boundary maps
    - The chain length determines the code dimension
    
    All TopologicalCSSCode variants inherit from this class.
    
    Chain Complex Structure:
    - 2-chain (C1 → C0): Repetition codes, classical codes
    - 3-chain (C2 → C1 → C0): 2D CSS codes (surface, toric, color)
    - 4-chain (C3 → C2 → C1 → C0): 3D CSS codes (3D toric)
    - 5-chain (C4 → C3 → C2 → C1 → C0): 4D CSS codes (tesseract, HGP of 2D codes)
    
    The qubit_grade property of the chain complex determines which cells host qubits:
    - grade 1: qubits on edges (most common for 2D/3D codes)
    - grade 2: qubits on faces (common for 4D codes)
    """
    
    def __init__(
        self,
        chain_complex: "ChainComplex",
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a CSS code from a chain complex.
        
        Parameters
        ----------
        chain_complex : ChainComplex
            The chain complex defining the code structure. Required.
        logical_x, logical_z : list of PauliString
            Logical operators.
        metadata : dict, optional
            Additional metadata.
        """
        if chain_complex is None:
            raise ValueError("CSSCodeWithComplex requires a non-None chain_complex")
        
        meta = dict(metadata or {})
        meta["chain_complex"] = chain_complex
        meta["chain_length"] = chain_complex.max_grade + 1
        meta["qubit_grade"] = chain_complex.qubit_grade
        
        # Derive Hx and Hz from chain complex
        hx = chain_complex.hx if hasattr(chain_complex, 'hx') else self._derive_hx(chain_complex)
        hz = chain_complex.hz if hasattr(chain_complex, 'hz') else self._derive_hz(chain_complex)
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
        
        self._chain_complex = chain_complex
    
    def _derive_hx(self, cc: "ChainComplex") -> np.ndarray:
        """Derive Hx from chain complex boundary maps."""
        qg = cc.qubit_grade
        # Hx comes from ∂_{qg+1}^T if it exists
        if qg + 1 in cc.boundary_maps:
            return (cc.boundary_maps[qg + 1].T.astype(np.uint8) % 2)
        return np.zeros((0, cc.boundary_maps[qg].shape[1] if qg in cc.boundary_maps else 0), dtype=np.uint8)
    
    def _derive_hz(self, cc: "ChainComplex") -> np.ndarray:
        """Derive Hz from chain complex boundary maps."""
        qg = cc.qubit_grade
        # Hz comes from ∂_{qg}
        if qg in cc.boundary_maps:
            return (cc.boundary_maps[qg].astype(np.uint8) % 2)
        return np.zeros((0, 0), dtype=np.uint8)
    
    @property
    def chain_complex(self) -> "ChainComplex":
        """The underlying chain complex (required, never None)."""
        return self._chain_complex
    
    @property
    def chain_length(self) -> int:
        """Chain length (max_grade + 1)."""
        return self._chain_complex.max_grade + 1
    
    @property
    def qubit_grade(self) -> int:
        """Grade of the chain group hosting qubits."""
        return self._chain_complex.qubit_grade


class TopologicalCSSCode(CSSCodeWithComplex, TopologicalCode):
    """
    CSS code defined by a 3-term chain complex (C2 → C1 → C0) with 2D geometry.
    
    This is the standard class for 2D topological codes like:
    - Surface codes
    - Toric codes
    - Color codes (2D)
    
    Chain structure:
    - C2: Plaquettes/faces
    - C1: Edges (qubits)
    - C0: Vertices
    
    Stabilizers:
    - X stabilizers from ∂2 (faces → edges)
    - Z stabilizers from ∂1^T (vertices → edges)
    """

    def __init__(
        self,
        chain_complex: "CSSChainComplex3",
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        *,
        embeddings: Optional[Dict[int, CellEmbedding]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a 2D topological CSS code.
        
        Parameters
        ----------
        chain_complex : CSSChainComplex3
            The 3-term chain complex.
        logical_x, logical_z : list of PauliString
            Logical operators.
        embeddings : dict, optional
            Maps grade → CellEmbedding for geometry.
        metadata : dict, optional
            Additional metadata.
        """
        meta = dict(metadata or {})
        
        # Initialize via CSSCodeWithComplex (handles chain_complex storage and Hx/Hz derivation)
        CSSCodeWithComplex.__init__(
            self,
            chain_complex=chain_complex,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
        
        # Store embeddings for TopologicalCode
        self._embeddings = embeddings or {}
        self._dim = 2

    @property
    def chain_complex(self) -> "CSSChainComplex3":
        """The underlying 3-term chain complex."""
        return self._chain_complex

    @property
    def dimension(self) -> int:
        """Ambient dimension (2 for surface codes)."""
        return self._dim

    def cell_coords(self, grade: int) -> List[Coord]:
        """Get coordinates for cells of given grade."""
        if grade not in self._embeddings:
            return []
        return self._embeddings[grade].coords

    def qubit_coords(self) -> Optional[List[Coord2D]]:
        """Return 2D coordinates for data qubits."""
        coords = self.cell_coords(1)  # Qubits on edges (grade 1)
        return coords if coords else None


class TopologicalCSSCode3D(CSSCodeWithComplex, TopologicalCode):
    """
    CSS code defined by a 4-term chain complex (C3 → C2 → C1 → C0) with 3D geometry.
    
    This is for 3D topological codes like:
    - 3D toric codes
    - 3D color codes
    
    Chain structure:
    - C3: 3-cells (cubes)
    - C2: 2-cells (faces)
    - C1: 1-cells (edges, typically qubits)
    - C0: 0-cells (vertices)
    
    For the standard 3D toric code, qubits are on edges (grade 1).
    """

    def __init__(
        self,
        chain_complex: "CSSChainComplex4",
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        *,
        embeddings: Optional[Dict[int, CellEmbedding]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a 3D topological CSS code.
        
        Parameters
        ----------
        chain_complex : CSSChainComplex4
            The 4-term chain complex. Required.
        logical_x, logical_z : list of PauliString
            Logical operators.
        embeddings : dict, optional
            Maps grade → CellEmbedding for geometry.
        metadata : dict, optional
            Additional metadata.
        """
        meta = dict(metadata or {})

        CSSCodeWithComplex.__init__(
            self,
            chain_complex=chain_complex,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
        
        self._embeddings = embeddings or {}
        self._dim = 3

    @property
    def chain_complex(self) -> "CSSChainComplex4":
        """The underlying 4-term chain complex."""
        return self._chain_complex

    @property
    def dimension(self) -> int:
        """Ambient dimension (3)."""
        return self._dim

    def cell_coords(self, grade: int) -> List[Coord]:
        """Get coordinates for cells of given grade."""
        if grade not in self._embeddings:
            return []
        return self._embeddings[grade].coords

    def qubit_coords(self) -> Optional[List[Tuple[float, float, float]]]:
        """Return 3D coordinates for data qubits."""
        coords = self.cell_coords(1)  # Typically qubits on edges
        return coords if coords else None


class TopologicalCSSCode4D(CSSCodeWithComplex, TopologicalCode):
    """
    CSS code defined by a 5-term chain complex with 4D geometry.
    
    This is for 4D topological codes like:
    - 4D toric code (tesseract code)
    - Homological products of two 2D toric codes
    - Hypergraph product codes
    
    Chain structure:
    - C4: 4-cells (hypercubes)
    - C3: 3-cells (cubes)
    - C2: 2-cells (faces, typically qubits)
    - C1: 1-cells (edges)
    - C0: 0-cells (vertices)
    
    For the standard 4D toric code, qubits are on faces (grade 2).
    """

    def __init__(
        self,
        chain_complex: "FiveCSSChainComplex",
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        *,
        embeddings: Optional[Dict[int, CellEmbedding]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a 4D topological CSS code.
        
        Parameters
        ----------
        chain_complex : FiveCSSChainComplex
            The 5-term chain complex. Required.
        logical_x, logical_z : list of PauliString
            Logical operators.
        embeddings : dict, optional
            Maps grade → CellEmbedding for geometry.
        metadata : dict, optional
            Additional metadata.
        """
        meta = dict(metadata or {})

        CSSCodeWithComplex.__init__(
            self,
            chain_complex=chain_complex,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
        
        self._embeddings = embeddings or {}
        self._dim = 4

    @property
    def chain_complex(self) -> "FiveCSSChainComplex":
        """The underlying 5-term chain complex."""
        return self._chain_complex

    @property
    def dimension(self) -> int:
        """Ambient dimension (4)."""
        return self._dim

    def cell_coords(self, grade: int) -> List[Coord]:
        """Get coordinates for cells of given grade."""
        if grade not in self._embeddings:
            return []
        return self._embeddings[grade].coords

    def qubit_coords(self) -> Optional[List[Tuple[float, ...]]]:
        """Return 4D coordinates for data qubits."""
        coords = self.cell_coords(2)  # Typically qubits on faces for 4D toric
        return coords if coords else None


class SubsystemCSSCode(SubsystemCode, CSSCode):
    """
    Subsystem code with CSS structure.
    
    A subsystem CSS code has:
    - Gauge operators that factor into X-type and Z-type
    - Stabilizers are products of gauge operators (center of gauge group)
    - CSS structure: Hx @ Hz.T = 0 mod 2 for stabilizers
    
    Examples:
    - Bacon-Shor code
    - Subsystem surface codes
    - Gauge color codes
    """

    def __init__(
        self,
        hx: np.ndarray,
        hz: np.ndarray,
        gauge_x: np.ndarray,
        gauge_z: np.ndarray,
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a subsystem CSS code.
        
        Parameters
        ----------
        hx, hz : np.ndarray
            Stabilizer parity check matrices (from center of gauge group).
        gauge_x, gauge_z : np.ndarray
            Gauge operator matrices (X-type and Z-type).
        logical_x, logical_z : list of PauliString
            Logical operators.
        metadata : dict, optional
            Additional metadata.
        """
        # Store CSS structure
        self._hx = np.array(hx, dtype=np.uint8)
        self._hz = np.array(hz, dtype=np.uint8)
        self._gauge_x = np.array(gauge_x, dtype=np.uint8)
        self._gauge_z = np.array(gauge_z, dtype=np.uint8)
        self._logical_x = logical_x
        self._logical_z = logical_z
        self._metadata = metadata or {}

    @property
    def n(self) -> int:
        """Number of physical qubits."""
        if self._hx.size > 0:
            return self._hx.shape[1]
        elif self._gauge_x.size > 0:
            return self._gauge_x.shape[1]
        return 0

    @property
    def k(self) -> int:
        """Number of logical qubits."""
        return len(self._logical_x)

    @property
    def logical_x_ops(self) -> List[PauliString]:
        return self._logical_x

    @property
    def logical_z_ops(self) -> List[PauliString]:
        return self._logical_z

    @property
    def stabilizer_matrix(self) -> np.ndarray:
        """Stabilizer generators in symplectic form."""
        n = self.n
        if n == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        
        num_x_stabs = self._hx.shape[0] if self._hx.size > 0 else 0
        num_z_stabs = self._hz.shape[0] if self._hz.size > 0 else 0
        
        stab_mat = np.zeros((num_x_stabs + num_z_stabs, 2 * n), dtype=np.uint8)
        
        if num_x_stabs > 0:
            stab_mat[:num_x_stabs, :n] = self._hx
        if num_z_stabs > 0:
            stab_mat[num_x_stabs:, n:] = self._hz
        
        return stab_mat

    @property
    def gauge_matrix(self) -> np.ndarray:
        """
        Gauge operators in symplectic form [X_part | Z_part].
        
        Combines gauge_x and gauge_z matrices.
        """
        n = self.n
        if n == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        
        num_gauge_x = self._gauge_x.shape[0] if self._gauge_x.size > 0 else 0
        num_gauge_z = self._gauge_z.shape[0] if self._gauge_z.size > 0 else 0
        
        gauge_mat = np.zeros((num_gauge_x + num_gauge_z, 2 * n), dtype=np.uint8)
        
        if num_gauge_x > 0:
            gauge_mat[:num_gauge_x, :n] = self._gauge_x
        if num_gauge_z > 0:
            gauge_mat[num_gauge_x:, n:] = self._gauge_z
        
        return gauge_mat

    @property
    def hx(self) -> np.ndarray:
        """X-stabilizer parity check matrix."""
        return self._hx

    @property
    def hz(self) -> np.ndarray:
        """Z-stabilizer parity check matrix."""
        return self._hz

    @property
    def gauge_x(self) -> np.ndarray:
        """X-type gauge operators."""
        return self._gauge_x

    @property
    def gauge_z(self) -> np.ndarray:
        """Z-type gauge operators."""
        return self._gauge_z

    def build_stabilizers(self) -> None:
        """No-op for SubsystemCSSCode."""
        pass


class FloquetCode(StabilizerCode, ABC):
    """
    Abstract base class for Floquet codes (time-dependent stabilizer codes).
    
    Floquet codes have measurement schedules that cycle through different
    sets of check operators. They are NOT CSS codes because:
    - Check operators may mix X and Z
    - The measured operators change over time
    - Logical operators are defined only over complete cycles
    
    Examples:
    - Honeycomb code (Hastings-Haah)
    - ISG Floquet codes
    - Dynamical surface codes
    
    Key properties:
    - period: Number of rounds in one measurement cycle
    - round_checks(t): Check operators measured at time t
    """

    @property
    @abstractmethod
    def period(self) -> int:
        """Number of measurement rounds in one complete cycle."""
        ...

    @abstractmethod
    def round_checks(self, t: int) -> List[PauliString]:
        """
        Get the check operators measured at time t.
        
        Parameters
        ----------
        t : int
            Time step (0 to period-1).
        
        Returns
        -------
        List[PauliString]
            Check operators to measure at this time step.
        """
        ...

    @property
    def is_css(self) -> bool:
        """Floquet codes are generally not CSS."""
        return False

    def stabilizers(self) -> List[PauliString]:
        """
        Return the instantaneous stabilizers (checks measured in one round).
        
        For Floquet codes, this returns the checks from round 0.
        The full stabilizer group is only well-defined over a complete cycle.
        """
        return self.round_checks(0)
