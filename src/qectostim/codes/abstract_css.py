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

    # =========================================================================
    # Logical Operator Support Methods (for gadgets and surgery)
    # =========================================================================
    
    def logical_x_support(self, logical_idx: int = 0) -> List[int]:
        """
        Get qubit indices in the support of a logical X operator.
        
        This is essential for CSS surgery, which needs to identify boundary
        qubits where logical operators can be coupled between codes.
        
        Parameters
        ----------
        logical_idx : int
            Index of the logical qubit (default 0 for single-logical codes).
            
        Returns
        -------
        List[int]
            Physical qubit indices where logical X has non-identity support.
            
        Raises
        ------
        IndexError
            If logical_idx >= k (number of logical qubits).
            
        Examples
        --------
        >>> code = RotatedSurfaceCode(distance=3)
        >>> code.logical_x_support()  # Returns qubits along one boundary
        [0, 1, 2]
        """
        if logical_idx >= self.k:
            raise IndexError(f"Logical index {logical_idx} >= k={self.k}")
        
        # Check metadata first (many codes pre-compute this)
        if 'logical_x_support' in self._metadata:
            support = self._metadata['logical_x_support']
            # Handle both single-logical (list of ints) and multi-logical (list of lists)
            if support and isinstance(support[0], (list, tuple)):
                return list(support[logical_idx]) if logical_idx < len(support) else []
            return list(support)  # Single logical case
        
        # Fall back to parsing logical_x_ops
        if logical_idx < len(self._logical_x):
            L = self._logical_x[logical_idx]
            return self._pauli_support(L, ('X', 'Y'))
        
        return list(range(self.n))  # Ultimate fallback: all qubits
    
    def logical_z_support(self, logical_idx: int = 0) -> List[int]:
        """
        Get qubit indices in the support of a logical Z operator.
        
        This is essential for CSS surgery, which needs to identify boundary
        qubits where logical operators can be coupled between codes.
        
        Parameters
        ----------
        logical_idx : int
            Index of the logical qubit (default 0 for single-logical codes).
            
        Returns
        -------
        List[int]
            Physical qubit indices where logical Z has non-identity support.
            
        Raises
        ------
        IndexError
            If logical_idx >= k (number of logical qubits).
        """
        if logical_idx >= self.k:
            raise IndexError(f"Logical index {logical_idx} >= k={self.k}")
        
        # Check metadata first
        if 'logical_z_support' in self._metadata:
            support = self._metadata['logical_z_support']
            if support and isinstance(support[0], (list, tuple)):
                return list(support[logical_idx]) if logical_idx < len(support) else []
            return list(support)
        
        # Fall back to parsing logical_z_ops
        if logical_idx < len(self._logical_z):
            L = self._logical_z[logical_idx]
            return self._pauli_support(L, ('Z', 'Y'))
        
        return list(range(self.n))
    
    def _pauli_support(self, pauli_op: PauliString, paulis: Tuple[str, ...]) -> List[int]:
        """
        Extract qubit indices where a Pauli operator has specified Pauli types.
        
        Parameters
        ----------
        pauli_op : PauliString
            The Pauli operator (str or dict format).
        paulis : Tuple[str, ...]
            Which Pauli types to look for (e.g., ('X', 'Y') or ('Z', 'Y')).
            
        Returns
        -------
        List[int]
            Qubit indices in the support.
        """
        support = []
        if isinstance(pauli_op, str):
            for q, p in enumerate(pauli_op):
                if p in paulis:
                    support.append(q)
        elif isinstance(pauli_op, dict):
            for q, p in pauli_op.items():
                if p in paulis:
                    support.append(q)
        elif isinstance(pauli_op, np.ndarray):
            # Symplectic form: first n bits are X, second n bits are Z
            n = len(pauli_op) // 2
            x_part = pauli_op[:n]
            z_part = pauli_op[n:]
            for q in range(n):
                has_x = bool(x_part[q]) if q < len(x_part) else False
                has_z = bool(z_part[q]) if q < len(z_part) else False
                if has_x and has_z:
                    p = 'Y'
                elif has_x:
                    p = 'X'
                elif has_z:
                    p = 'Z'
                else:
                    continue
                if p in paulis:
                    support.append(q)
        return support
    
    def logical_x_string(self, logical_idx: int = 0) -> str:
        """
        Get logical X operator as a Pauli string.
        
        Parameters
        ----------
        logical_idx : int
            Index of the logical qubit.
            
        Returns
        -------
        str
            Pauli string like "XXIIIXXII".
        """
        if logical_idx >= self.k:
            raise IndexError(f"Logical index {logical_idx} >= k={self.k}")
        
        if logical_idx < len(self._logical_x):
            L = self._logical_x[logical_idx]
            return self._pauli_to_string(L)
        return "I" * self.n
    
    def logical_z_string(self, logical_idx: int = 0) -> str:
        """
        Get logical Z operator as a Pauli string.
        
        Parameters
        ----------
        logical_idx : int
            Index of the logical qubit.
            
        Returns
        -------
        str
            Pauli string like "ZZIIIZZZII".
        """
        if logical_idx >= self.k:
            raise IndexError(f"Logical index {logical_idx} >= k={self.k}")
        
        if logical_idx < len(self._logical_z):
            L = self._logical_z[logical_idx]
            return self._pauli_to_string(L)
        return "I" * self.n
    
    def _pauli_to_string(self, pauli_op: PauliString) -> str:
        """Convert a PauliString to string format."""
        result = ['I'] * self.n
        if isinstance(pauli_op, str):
            for i, p in enumerate(pauli_op):
                if i < self.n:
                    result[i] = p
        elif isinstance(pauli_op, dict):
            for q, p in pauli_op.items():
                if 0 <= q < self.n:
                    result[q] = p
        elif isinstance(pauli_op, np.ndarray):
            n = self.n
            x_part = pauli_op[:n] if len(pauli_op) >= n else pauli_op
            z_part = pauli_op[n:2*n] if len(pauli_op) >= 2*n else np.zeros(n)
            for q in range(n):
                has_x = bool(x_part[q]) if q < len(x_part) else False
                has_z = bool(z_part[q]) if q < len(z_part) else False
                if has_x and has_z:
                    result[q] = 'Y'
                elif has_x:
                    result[q] = 'X'
                elif has_z:
                    result[q] = 'Z'
        return ''.join(result)
    
    def get_observable_transformation(
        self,
        gate_name: str,
        logical_idx: int = 0,
    ) -> Dict[str, str]:
        """
        Compute how logical operators transform under a transversal gate.
        
        This is essential for tracking logical observables through gadget circuits.
        
        Parameters
        ----------
        gate_name : str
            Gate name: 'H', 'S', 'X', 'Z', 'CNOT', etc.
        logical_idx : int
            Which logical qubit.
            
        Returns
        -------
        Dict[str, str]
            Mapping from observable type to transformed type.
            E.g., {'X': 'Z', 'Z': 'X'} for Hadamard.
            
        Examples
        --------
        >>> code.get_observable_transformation('H')
        {'X': 'Z', 'Z': 'X'}
        >>> code.get_observable_transformation('S')
        {'X': 'Y', 'Z': 'Z'}
        """
        # Standard Clifford transformations
        gate = gate_name.upper()
        
        if gate == 'H':
            return {'X': 'Z', 'Z': 'X', 'Y': '-Y'}
        elif gate == 'S':
            return {'X': 'Y', 'Z': 'Z', 'Y': '-X'}
        elif gate == 'S_DAG':
            return {'X': '-Y', 'Z': 'Z', 'Y': 'X'}
        elif gate == 'X':
            return {'X': 'X', 'Z': '-Z', 'Y': '-Y'}
        elif gate == 'Z':
            return {'X': '-X', 'Z': 'Z', 'Y': '-Y'}
        elif gate == 'Y':
            return {'X': '-X', 'Z': '-Z', 'Y': 'Y'}
        elif gate in ('CX', 'CNOT'):
            # Returns transformation for control qubit
            # Target would be: X: X, Z: Z_ctrl @ Z_tgt
            return {'X': 'X', 'Z': 'Z'}  # Control is unchanged
        elif gate == 'CZ':
            return {'X': 'X', 'Z': 'Z'}  # Control is unchanged for Z obs
        else:
            # Unknown gate - assume identity
            return {'X': 'X', 'Z': 'Z', 'Y': 'Y'}


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


class TopologicalCSSCode3D(CSSCode, TopologicalCode):
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
    
    This class supports two initialization modes:
    1. With chain_complex: Hx and Hz are derived from the complex
    2. With hx/hz directly: For codes that compute parity matrices directly
    """

    def __init__(
        self,
        hx: Optional[np.ndarray] = None,
        hz: Optional[np.ndarray] = None,
        logical_x: Optional[List[PauliString]] = None,
        logical_z: Optional[List[PauliString]] = None,
        *,
        chain_complex: Optional["CSSChainComplex4"] = None,
        embeddings: Optional[Dict[int, CellEmbedding]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a 3D topological CSS code.
        
        Parameters
        ----------
        hx : np.ndarray, optional
            X-stabilizer parity check matrix. Required if chain_complex is None.
        hz : np.ndarray, optional  
            Z-stabilizer parity check matrix. Required if chain_complex is None.
        logical_x, logical_z : list of PauliString
            Logical operators.
        chain_complex : CSSChainComplex4, optional
            The 4-term chain complex. If provided, hx and hz are derived from it.
        embeddings : dict, optional
            Maps grade → CellEmbedding for geometry.
        metadata : dict, optional
            Additional metadata.
        """
        meta = dict(metadata or {})
        
        # If chain_complex provided, derive hx/hz from it
        if chain_complex is not None:
            self._chain_complex = chain_complex
            meta["chain_complex"] = chain_complex
            if hx is None:
                hx = chain_complex.hx if hasattr(chain_complex, 'hx') else np.zeros((0, 0), dtype=np.uint8)
            if hz is None:
                hz = chain_complex.hz if hasattr(chain_complex, 'hz') else np.zeros((0, 0), dtype=np.uint8)
        else:
            self._chain_complex = None
        
        if hx is None or hz is None:
            raise ValueError("Either chain_complex or both hx and hz must be provided")
        
        if logical_x is None:
            logical_x = []
        if logical_z is None:
            logical_z = []

        CSSCode.__init__(
            self,
            hx=hx,
            hz=hz,
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
