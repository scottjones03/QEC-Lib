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
        skip_validation: bool = False,
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
        skip_validation : bool, optional
            If True, skip CSS commutativity validation. Useful for testing
            or codes that may not be strictly valid CSS.
        """
        self._hx = np.array(hx, dtype=np.uint8)
        self._hz = np.array(hz, dtype=np.uint8)
        self._logical_x = logical_x
        self._logical_z = logical_z
        self._metadata = metadata or {}
        self._skip_validation = skip_validation
        self._validate_css()

    def _validate_css(self) -> None:
        """Validate CSS constraints: Hx and Hz must commute (Hx @ Hz.T = 0 mod 2)."""
        if self._skip_validation:
            return
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
        """
        Number of logical qubits.
        
        Uses metadata['k'] if provided (needed for non-strict CSS codes),
        otherwise computes as k = n - rank(Hx) - rank(Hz) over GF(2).
        """
        # Use metadata value if provided (needed for non-strict CSS codes)
        if 'k' in self._metadata:
            return self._metadata['k']
        # Otherwise compute from parity check matrices
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

    # =========================================================================
    # Stabilizer Support Methods (Goal 2: Library integration)
    # =========================================================================
    
    def get_z_stabilizers(self) -> List[List[int]]:
        """
        Get Z stabilizer supports as list of qubit index lists.
        
        Returns
        -------
        List[List[int]]
            For each Z stabilizer, the list of qubit indices in its support.
            
        Example
        -------
        >>> code.get_z_stabilizers()
        [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]  # For Steane code
        """
        return [list(np.where(row)[0]) for row in self._hz]
    
    def get_x_stabilizers(self) -> List[List[int]]:
        """
        Get X stabilizer supports as list of qubit index lists.
        
        Returns
        -------
        List[List[int]]
            For each X stabilizer, the list of qubit indices in its support.
            
        Example
        -------
        >>> code.get_x_stabilizers()
        [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]  # For Steane code
        """
        return [list(np.where(row)[0]) for row in self._hx]
    
    def get_logical_z_support(self, logical_idx: int = 0) -> List[int]:
        """
        Get support of logical Z operator.
        
        Parameters
        ----------
        logical_idx : int
            Which logical qubit (default 0).
            
        Returns
        -------
        List[int]
            Qubit indices in the logical Z operator's support.
        """
        return list(np.where(self.logical_z_array(logical_idx))[0])
    
    def get_logical_x_support(self, logical_idx: int = 0) -> List[int]:
        """
        Get support of logical X operator.
        
        Parameters
        ----------
        logical_idx : int
            Which logical qubit (default 0).
            
        Returns
        -------
        List[int]
            Qubit indices in the logical X operator's support.
        """
        return list(np.where(self.logical_x_array(logical_idx))[0])

    # --- CSS-specific properties ---

    @property
    def hx(self) -> np.ndarray:
        """X-stabilizer parity check matrix."""
        return self._hx

    @property
    def hz(self) -> np.ndarray:
        """Z-stabilizer parity check matrix."""
        return self._hz

    # =========================================================================
    # Uppercase Aliases and Array-Format Accessors
    # =========================================================================
    
    @property
    def Hz(self) -> np.ndarray:
        """Z-stabilizer check matrix (uppercase alias for hz)."""
        return self._hz
    
    @property
    def Hx(self) -> np.ndarray:
        """X-stabilizer check matrix (uppercase alias for hx)."""
        return self._hx
    
    @property
    def d(self) -> int:
        """Code distance (alias for distance property, returns from metadata)."""
        return self._metadata.get('d', self._metadata.get('distance', 3))
    
    def logical_z_array(self, logical_idx: int = 0) -> np.ndarray:
        """
        Get logical Z operator as a binary numpy array.
        
        Returns the SUPPORT of the logical Z operator - all qubits where
        the operator has a non-identity Pauli (X, Y, or Z).
        
        NOTE: The logical Z operator may be X-type (like Shor code) or Z-type
        (like Steane code). This method returns the support regardless of the
        Pauli type. Use lz_pauli_type property to determine the actual type.
        
        Parameters
        ----------
        logical_idx : int
            Index of the logical qubit (default 0).
            
        Returns
        -------
        np.ndarray
            Binary array where 1 indicates the logical Z operator has support.
        """
        if logical_idx >= len(self._logical_z):
            return np.zeros(self.n, dtype=np.int64)
        return self._pauli_string_to_support_array(self._logical_z[logical_idx])
    
    def logical_x_array(self, logical_idx: int = 0) -> np.ndarray:
        """
        Get logical X operator as a binary numpy array.
        
        Returns the SUPPORT of the logical X operator - all qubits where
        the operator has a non-identity Pauli (X, Y, or Z).
        
        NOTE: The logical X operator may be Z-type (like Shor code) or X-type
        (like Steane code). This method returns the support regardless of the
        Pauli type. Use lx_pauli_type property to determine the actual type.
        
        Parameters
        ----------
        logical_idx : int
            Index of the logical qubit (default 0).
            
        Returns
        -------
        np.ndarray
            Binary array where 1 indicates the logical X operator has support.
        """
        if logical_idx >= len(self._logical_x):
            return np.zeros(self.n, dtype=np.int64)
        return self._pauli_string_to_support_array(self._logical_x[logical_idx])
    
    def _pauli_string_to_support_array(self, pauli: PauliString) -> np.ndarray:
        """
        Convert a PauliString to a binary numpy array indicating support.
        
        Parameters
        ----------
        pauli : PauliString
            Dict mapping qubit indices to Pauli operators, or string format.
            
        Returns
        -------
        np.ndarray
            Binary array where 1 indicates any non-identity Pauli (X, Y, or Z).
        """
        arr = np.zeros(self.n, dtype=np.int64)
        if isinstance(pauli, str):
            for i, op in enumerate(pauli):
                if i < self.n and op in ('X', 'Y', 'Z'):
                    arr[i] = 1
        elif isinstance(pauli, dict):
            for i, op in pauli.items():
                if op in ('X', 'Y', 'Z'):
                    arr[i] = 1
        return arr
    
    def _pauli_string_to_binary_array(self, pauli: PauliString, pauli_type: str) -> np.ndarray:
        """
        Convert a PauliString to a binary numpy array.
        
        Parameters
        ----------
        pauli : PauliString
            Dict mapping qubit indices to Pauli operators, or string format.
        pauli_type : str
            The Pauli type to extract ('X' or 'Z').
            
        Returns
        -------
        np.ndarray
            Binary array where 1 indicates the specified Pauli type.
        """
        arr = np.zeros(self.n, dtype=np.int64)
        if isinstance(pauli, str):
            for i, op in enumerate(pauli):
                if i < self.n and (op == pauli_type or op == 'Y'):
                    arr[i] = 1
        elif isinstance(pauli, dict):
            for i, op in pauli.items():
                if op == pauli_type or op == 'Y':
                    arr[i] = 1
        return arr
    
    @property
    def Lz(self) -> np.ndarray:
        """Logical Z operator for first logical qubit as binary array."""
        return self.logical_z_array(0)
    
    @property
    def Lx(self) -> np.ndarray:
        """Logical X operator for first logical qubit as binary array."""
        return self.logical_x_array(0)
    
    @property
    def Lz2(self) -> Optional[np.ndarray]:
        """Logical Z operator for second logical qubit (None if k < 2)."""
        if self.k < 2:
            return None
        return self.logical_z_array(1)
    
    @property
    def Lx2(self) -> Optional[np.ndarray]:
        """Logical X operator for second logical qubit (None if k < 2)."""
        if self.k < 2:
            return None
        return self.logical_x_array(1)
    
    @property
    def lz_pauli_type(self) -> str:
        """
        Get the Pauli type of the logical Z operator ('Z' or 'X').
        
        For standard CSS codes like Steane [[7,1,3]], Lz is Z-type (ZZZIIII).
        For codes like Shor [[9,1,3]], Lz is X-type (XXXIIIIII).
        
        The Pauli type determines which check matrix to use for decoding:
        - Z-type Lz: X errors anti-commute → use Hz (detects X errors)
        - X-type Lz: Z errors anti-commute → use Hx (detects Z errors)
        
        Returns 'Z' by default, or from metadata['lz_pauli_type'] if set.
        """
        if 'lz_pauli_type' in self._metadata:
            return self._metadata['lz_pauli_type']
        
        # Infer from logical_z string
        if self._logical_z:
            lz_str = self._logical_z[0]
            if isinstance(lz_str, str):
                has_x = 'X' in lz_str or 'Y' in lz_str
                has_z = 'Z' in lz_str or 'Y' in lz_str
                if has_x and not has_z:
                    return 'X'
                elif has_z and not has_x:
                    return 'Z'
        
        return 'Z'  # Default for standard CSS codes
    
    @property
    def lx_pauli_type(self) -> str:
        """
        Get the Pauli type of the logical X operator ('X' or 'Z').
        
        For standard CSS codes like Steane [[7,1,3]], Lx is X-type (XXXIIII).
        For codes like Shor [[9,1,3]], Lx is Z-type (ZIIZIIZII).
        
        The Pauli type determines which check matrix to use for decoding:
        - X-type Lx: Z errors anti-commute → use Hx (detects Z errors)
        - Z-type Lx: X errors anti-commute → use Hz (detects X errors)
        
        Returns 'X' by default, or from metadata['lx_pauli_type'] if set.
        """
        if 'lx_pauli_type' in self._metadata:
            return self._metadata['lx_pauli_type']
        
        # Infer from logical_x string
        if self._logical_x:
            lx_str = self._logical_x[0]
            if isinstance(lx_str, str):
                has_x = 'X' in lx_str or 'Y' in lx_str
                has_z = 'Z' in lx_str or 'Y' in lx_str
                if has_z and not has_x:
                    return 'Z'
                elif has_x and not has_z:
                    return 'X'
        
        return 'X'  # Default for standard CSS codes

    @property
    def num_x_stabilizers(self) -> int:
        """Number of X stabilizer generators."""
        return self._hx.shape[0] if self._hx.size > 0 else 0
    
    @property
    def num_z_stabilizers(self) -> int:
        """Number of Z stabilizer generators."""
        return self._hz.shape[0] if self._hz.size > 0 else 0
    
    def get_stabilizer_support(self, stab_type: str, index: int) -> List[int]:
        """
        Get qubit indices in the support of a stabilizer.
        
        Parameters
        ----------
        stab_type : str
            'x' for X-type stabilizer, 'z' for Z-type stabilizer.
        index : int
            Index of the stabilizer generator.
            
        Returns
        -------
        List[int]
            List of qubit indices where the stabilizer has support.
        """
        matrix = self._hx if stab_type == 'x' else self._hz
        return [i for i, v in enumerate(matrix[index]) if v == 1]

    # =========================================================================
    # Stabilizer Block Structure Analysis
    # =========================================================================
    
    def _find_block_structure(self, check_matrix: np.ndarray) -> List[List[int]]:
        """
        Find disjoint blocks from check matrix structure using union-find.
        
        Parameters
        ----------
        check_matrix : np.ndarray
            Parity check matrix (Hx or Hz).
            
        Returns
        -------
        List[List[int]]
            List of blocks (each block is a list of qubit indices), or empty
            if all qubits are in a single connected component.
        """
        if check_matrix.shape[0] == 0:
            return []
        
        n = check_matrix.shape[1]
        from collections import defaultdict
        
        parent = list(range(n))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for stab_idx in range(check_matrix.shape[0]):
            support = [i for i in range(n) if check_matrix[stab_idx, i] == 1]
            for i in range(len(support) - 1):
                union(support[i], support[i+1])
        
        blocks_dict = defaultdict(list)
        for q in range(n):
            blocks_dict[find(q)].append(q)
        
        blocks = sorted(blocks_dict.values(), key=lambda b: min(b))
        return blocks if len(blocks) > 1 else []
    
    @property
    def stabilizer_block_structure(self) -> Optional[Dict[str, List[List[int]]]]:
        """
        Detect block structure in stabilizers.
        
        Returns
        -------
        Dict or None
            Dictionary with 'z_blocks' and 'x_blocks' if structure exists,
            otherwise None.
        """
        z_blocks = self._find_block_structure(self._hz)
        x_blocks = self._find_block_structure(self._hx)
        if z_blocks or x_blocks:
            return {'z_blocks': z_blocks, 'x_blocks': x_blocks}
        return None
    
    @property
    def has_superposition_codewords(self) -> bool:
        """
        Check if code has superposition structure in |0⟩_L.
        
        Returns True if the code has block structure indicating superposition
        codewords (like Shor code).
        """
        blocks = self.stabilizer_block_structure
        if blocks and blocks.get('z_blocks') and len(blocks['z_blocks']) > 1:
            return True
        return False
    
    @property
    def measurement_strategy(self) -> str:
        """
        Determine measurement/detector strategy for this code.
        
        Returns
        -------
        str
            'parity' for k>=2 codes, 'relative' for superposition codes,
            'absolute' otherwise.
        """
        if self.k >= 2:
            return 'parity'
        if self.has_superposition_codewords:
            return 'relative'
        return 'absolute'
    
    def decode_z_basis_measurement(self, m: np.ndarray) -> int:
        """
        Decode a Z-basis measurement to logical value.
        
        Parameters
        ----------
        m : np.ndarray
            Measurement outcomes as binary array.
            
        Returns
        -------
        int
            Logical value (0 or 1).
        """
        m = np.array(m).flatten()
        
        if self.measurement_strategy == 'relative':
            blocks = self.stabilizer_block_structure
            if blocks and blocks.get('z_blocks'):
                block_values = []
                for block in blocks['z_blocks']:
                    block_bits = [m[q] for q in block]
                    block_values.append(int(sum(block_bits) > len(block_bits) / 2))
                return sum(block_values) % 2
        
        return int(np.dot(self.Lz, m) % 2)

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
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Return the metadata dictionary."""
        return self._metadata

    # =========================================================================
    # Self-Duality and Transversal H Properties
    # =========================================================================
    
    @property
    def is_self_dual(self) -> bool:
        """
        Check if this CSS code is self-dual (Hx = Hz up to row permutation).
        
        For self-dual codes, transversal H implements logical H.
        This is crucial for state preparation circuits.
        """
        if self._hx.shape != self._hz.shape:
            return False
        return np.array_equal(self._hx, self._hz)
    
    @property
    def has_transversal_logical_h(self) -> bool:
        """
        Check if transversal H implements logical H.
        
        For CSS codes, this is true iff the code is self-dual (Hx = Hz).
        Examples:
        - Steane [[7,1,3]]: self-dual, transversal H = logical H ✓
        - Shor [[9,1,3]]: NOT self-dual, transversal H ≠ logical H ✗
        """
        return self.is_self_dual
    
    @property
    def plus_h_qubits(self) -> Optional[List[int]]:
        """Qubits to apply H gates for direct |+⟩_L preparation (non-self-dual codes)."""
        return self._metadata.get('plus_h_qubits')
    
    @property
    def plus_encoding_cnots(self) -> Optional[List[Tuple[int, int]]]:
        """CNOT gates for direct |+⟩_L preparation (non-self-dual codes)."""
        return self._metadata.get('plus_encoding_cnots')
    
    @property
    def plus_state_h_qubits(self) -> Optional[List[int]]:
        """
        Qubits to apply H gates for TRUE |+⟩_L preparation (Lx = +1 eigenstate).
        
        This uses the UNIVERSAL FORMULA for any CSS code:
          H on complement(zero_h_qubits)
          Reversed CNOTs from zero_encoding_cnots
        
        This is different from plus_h_qubits, which creates a "standard encoding"
        state with 50/50 Lz=0/Lz=1 (useful for L2 inner blocks).
        """
        return self._metadata.get('plus_state_h_qubits')
    
    @property
    def plus_state_encoding_cnots(self) -> Optional[List[Tuple[int, int]]]:
        """
        CNOT gates for TRUE |+⟩_L preparation (Lx = +1 eigenstate).
        
        Uses reversed CNOTs from zero_encoding_cnots (control <-> target swapped).
        This creates |+⟩_L = (|0⟩_L + |1⟩_L)/√2 with deterministic Lx=+1.
        """
        return self._metadata.get('plus_state_encoding_cnots')
    
    @property
    def requires_direct_plus_prep(self) -> bool:
        """
        Check if this code requires direct |+⟩_L preparation.
        
        Returns True if the code is NOT self-dual AND has plus_h_qubits defined.
        For such codes, transversal H ≠ logical H, so we need explicit |+⟩_L encoding.
        """
        return not self.is_self_dual and self.plus_h_qubits is not None
    
    @property
    def name(self) -> str:
        """Return the code name from metadata."""
        return self._metadata.get('name', f'CSS_[[{self.n},{self.k}]]')

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

    # =========================================================================
    # FT Gadget Experiment Hooks (CSS-specific)
    # =========================================================================

    def stabilizer_coords(self) -> Optional[List[Tuple[float, ...]]]:
        """
        Combined stabilizer coordinates (X then Z) for detector assignment.
        
        Returns None if neither X nor Z stabilizer coords are available.
        """
        x_coords = self.get_x_stabilizer_coords()
        z_coords = self.get_z_stabilizer_coords()
        if x_coords is None and z_coords is None:
            return None
        result = []
        if x_coords is not None:
            result.extend(x_coords)
        if z_coords is not None:
            result.extend(z_coords)
        return result if result else None

    def get_x_stabilizer_coords(self) -> Optional[List[Tuple[float, ...]]]:
        """
        Get coordinates for X stabilizer ancillas.
        
        Override in subclasses to provide custom coordinate layouts.
        Default implementation checks metadata.
        
        Returns
        -------
        Optional[List[Tuple[float, ...]]]
            Coordinates for each X stabilizer, or None if not available.
        """
        meta = self._metadata or {}
        if 'x_stab_coords' in meta:
            return [tuple(c) for c in meta['x_stab_coords']]
        return None
    
    def get_z_stabilizer_coords(self) -> Optional[List[Tuple[float, ...]]]:
        """
        Get coordinates for Z stabilizer ancillas.
        
        Override in subclasses to provide custom coordinate layouts.
        Default implementation checks metadata.
        
        Returns
        -------
        Optional[List[Tuple[float, ...]]]
            Coordinates for each Z stabilizer, or None if not available.
        """
        meta = self._metadata or {}
        if 'z_stab_coords' in meta:
            return [tuple(c) for c in meta['z_stab_coords']]
        return None
    
    def get_stabilizer_schedule(
        self,
        basis: str,
    ) -> Optional[List[List[Tuple[int, int, int]]]]:
        """
        Get CNOT schedule for stabilizer measurements.
        
        Override in subclasses to provide custom schedules (e.g., XZZX codes
        have diagonal patterns that require specific CNOT ordering).
        
        Parameters
        ----------
        basis : str
            'x' or 'z' for X-type or Z-type stabilizers.
            
        Returns
        -------
        Optional[List[List[Tuple[int, int, int]]]]
            Schedule as list of timesteps, each containing (stab_idx, data_qubit, step)
            tuples. Returns None to use default scheduling.
            
        Notes
        -----
        When None is returned, the builder will use automatic scheduling based
        on the code's FTGadgetCodeConfig.schedule_mode setting.
        """
        return None  # Use default scheduling
    
    def get_measurement_order(self, basis: str) -> Optional[List[int]]:
        """
        Get the order to measure stabilizers of given type.
        
        Override in subclasses that need specific measurement ordering
        (e.g., to enable detector comparisons).
        
        Parameters
        ----------
        basis : str
            'x' or 'z' for X-type or Z-type stabilizers.
            
        Returns
        -------
        Optional[List[int]]
            Order of stabilizer indices, or None for default (0, 1, 2, ...).
        """
        return None  # Use default order


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

    @property
    def meta_x(self) -> Optional[np.ndarray]:
        """X-type metacheck matrix if available.
        
        For 4D codes (5-chain): checks on Z syndrome measurements
        For 3D codes (4-chain): checks on Z syndrome (if qubits on faces)
        For 2D codes (3-chain): None (no metachecks)
        
        Metachecks satisfy: meta_x @ hz = 0 (checks Z syndrome parity)
        Used for single-shot error correction.
        
        For a chain complex with qubits on grade q:
        - Hz = boundary_q (qubits → grade q-1)
        - meta_x = boundary_{q-1} (grade q-1 → grade q-2)
        - Chain condition: boundary_{q-1} @ boundary_q = 0
          ensures meta_x @ hz = 0
        """
        cc = self._chain_complex
        # First check if the chain complex has explicit meta_x
        if hasattr(cc, 'meta_x') and cc.meta_x is not None:
            return cc.meta_x
        
        # Otherwise, derive from boundary maps for 4-chain or higher
        # meta_x checks Z syndrome, which comes from boundary_q
        # So meta_x = boundary_{q-1} (one grade below Hz source)
        q = cc.qubit_grade
        q_minus_1 = q - 1
        
        # Need boundary_{q-1} to exist (chain length >= 4)
        if q_minus_1 >= 1 and q_minus_1 in cc.boundary_maps:
            return (cc.boundary_maps[q_minus_1].astype(np.uint8) % 2)
        
        return None

    @property
    def meta_z(self) -> Optional[np.ndarray]:
        """Z-type metacheck matrix if available.
        
        For 4D codes (5-chain): checks on X syndrome measurements
        For 3D codes (4-chain): checks on X syndrome (if qubits on edges)
        For 2D codes (3-chain): None (no metachecks)
        
        Metachecks satisfy: meta_z @ hx = 0 (checks X syndrome parity)
        Used for single-shot error correction.
        
        For a chain complex with qubits on grade q:
        - Hx = boundary_{q+1}^T (grade q+1 → qubits)
        - meta_z = boundary_{q+2}^T (grade q+2 → grade q+1)
        - Chain condition: boundary_{q+1} @ boundary_{q+2} = 0
          ensures (boundary_{q+2}^T) @ (boundary_{q+1}^T) = 0
          i.e., meta_z @ hx = 0
        """
        cc = self._chain_complex
        # First check if the chain complex has explicit meta_z
        if hasattr(cc, 'meta_z') and cc.meta_z is not None:
            return cc.meta_z
        
        # Otherwise, derive from boundary maps for 5-chain or higher
        # meta_z checks X syndrome, which comes from boundary_{q+1}^T
        # So meta_z = boundary_{q+2}^T (one grade above Hx source)
        q = cc.qubit_grade
        q_plus_2 = q + 2
        
        # Need boundary_{q+2} to exist (chain length >= 5 for qubits on grade 2)
        if q_plus_2 in cc.boundary_maps:
            return (cc.boundary_maps[q_plus_2].T.astype(np.uint8) % 2)
        
        return None

    @property
    def has_metachecks(self) -> bool:
        """Whether this code has metacheck structure for single-shot correction."""
        return self.meta_x is not None or self.meta_z is not None

    def validate_chain_complex(self) -> Dict[str, Any]:
        """Validate the chain complex satisfies ∂² = 0 and report diagnostics.

        Checks every consecutive pair of boundary maps
        (∂_{k-1} ∘ ∂_k = 0 mod 2) and returns a summary dict.

        Returns
        -------
        dict
            ``{"valid": bool, "grades_checked": list[tuple],
              "boundary_shapes": dict, "failures": list[str]}``

        Raises
        ------
        RuntimeError
            If ``valid`` is ``False`` (any ∂² ≠ 0).

        Examples
        --------
        >>> code = RotatedSurfaceCode(distance=3)
        >>> info = code.validate_chain_complex()
        >>> info["valid"]
        True
        """
        cc = self._chain_complex
        result: Dict[str, Any] = {
            "valid": True,
            "grades_checked": [],
            "boundary_shapes": {},
            "failures": [],
        }
        for k, sigma_k in sorted(cc.boundary_maps.items()):
            result["boundary_shapes"][k] = sigma_k.shape
            prev = k - 1
            if prev in cc.boundary_maps:
                comp = (cc.boundary_maps[prev] @ sigma_k) % 2
                pair = (prev, k)
                result["grades_checked"].append(pair)
                if np.any(comp):
                    result["valid"] = False
                    result["failures"].append(
                        f"∂_{prev} ∘ ∂_{k} ≠ 0  (non-zero entries: "
                        f"{int(np.sum(comp))})"
                    )
        if not result["valid"]:
            raise RuntimeError(
                f"Chain complex ∂²≠0 failures: {result['failures']}"
            )
        return result


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

    def validate_chain_complex(self) -> Dict[str, Any]:
        """Validate ∂²=0 for the stored chain complex.

        Delegates to the same logic as ``CSSCodeWithComplex.validate_chain_complex``.

        Returns
        -------
        dict
            ``{"valid": bool, "grades_checked": [...], "boundary_shapes": {...},
            "failures": [...]}``

        Raises
        ------
        RuntimeError
            If any consecutive boundary map pair has ∂²≠0.
        AttributeError
            If no chain complex is stored.
        """
        cc = self._chain_complex
        if cc is None:
            raise AttributeError("No chain complex stored on this code.")
        grades_checked: List[Tuple[int, int]] = []
        failures: List[str] = []
        shapes: Dict[str, Tuple[int, ...]] = {}
        sorted_grades = sorted(cc.boundary_maps.keys())
        for idx in range(len(sorted_grades) - 1):
            k_lo = sorted_grades[idx]
            k_hi = sorted_grades[idx + 1]
            sigma_lo = cc.boundary_maps[k_lo]
            sigma_hi = cc.boundary_maps[k_hi]
            shapes[f"sigma_{k_lo}"] = sigma_lo.shape
            shapes[f"sigma_{k_hi}"] = sigma_hi.shape
            comp = (sigma_lo @ sigma_hi) % 2
            grades_checked.append((k_lo, k_hi))
            if np.any(comp):
                failures.append(
                    f"sigma_{k_lo} @ sigma_{k_hi} != 0  "
                    f"(shapes {sigma_lo.shape} @ {sigma_hi.shape}, "
                    f"nonzero entries: {int(np.sum(comp))})"
                )
        result = {
            "valid": len(failures) == 0,
            "grades_checked": grades_checked,
            "boundary_shapes": shapes,
            "failures": failures,
        }
        if failures:
            raise RuntimeError(
                f"Chain complex validation FAILED:\n" + "\n".join(failures)
            )
        return result


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
