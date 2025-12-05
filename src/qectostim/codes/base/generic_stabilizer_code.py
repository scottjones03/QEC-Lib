"""
GenericStabilizerCode: Flexible stabilizer code construction from symplectic matrices.

This module provides a complete implementation for constructing general stabilizer
codes (both CSS and non-CSS) from a symplectic stabilizer matrix. Analogous to
GenericCSSCode but for the more general StabilizerCode case.

Features:
- Accepts stabilizer generators in symplectic form [X_part | Z_part]
- Automatic logical operator inference using GF(2) linear algebra
- Supports both CSS and non-CSS codes
- Automatic is_css detection
- Compatible with StabilizerMemoryExperiment and decoders

Examples:
    # Create [[5,1,3]] Perfect code from stabilizer matrix
    stab_mat = np.array([
        [1, 0, 0, 0, 1,  0, 1, 1, 0, 0],  # XZZXI
        [0, 1, 0, 0, 1,  0, 0, 1, 1, 0],  # IXZZX
        [1, 0, 1, 0, 0,  0, 0, 0, 1, 1],  # XIXZZ
        [0, 1, 0, 1, 0,  1, 0, 0, 0, 1],  # ZXIXZ
    ])
    logical_x = [{0: 'X', 1: 'X', 2: 'X', 3: 'X', 4: 'X'}]  # XXXXX
    logical_z = [{0: 'Z', 1: 'Z', 2: 'Z', 3: 'Z', 4: 'Z'}]  # ZZZZZ
    code = GenericStabilizerCode(stab_mat, logical_x, logical_z)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np

from qectostim.codes.abstract_code import StabilizerCode, PauliString


def _gf2_rref(matrix: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Compute the reduced row echelon form of a matrix over GF(2).
    
    Returns:
        rref_matrix: The matrix in reduced row echelon form
        pivot_cols: List of pivot column indices
    """
    mat = matrix.copy().astype(np.uint8)
    rows, cols = mat.shape
    pivot_cols = []
    pivot_row = 0
    
    for col in range(cols):
        # Find pivot
        found = False
        for row in range(pivot_row, rows):
            if mat[row, col] == 1:
                # Swap rows
                mat[[pivot_row, row]] = mat[[row, pivot_row]]
                found = True
                break
        
        if not found:
            continue
        
        pivot_cols.append(col)
        
        # Eliminate other 1s in this column
        for row in range(rows):
            if row != pivot_row and mat[row, col] == 1:
                mat[row] = (mat[row] + mat[pivot_row]) % 2
        
        pivot_row += 1
        if pivot_row >= rows:
            break
    
    return mat, pivot_cols


def _gf2_kernel(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the kernel (null space) of a matrix over GF(2).
    
    Returns vectors v such that matrix @ v = 0 (mod 2).
    """
    if matrix.size == 0:
        return np.eye(matrix.shape[1] if matrix.ndim == 2 else 0, dtype=np.uint8)
    
    rows, cols = matrix.shape
    if rows == 0:
        return np.eye(cols, dtype=np.uint8)
    
    # Augment matrix with identity to track column operations
    augmented = np.hstack([matrix.T, np.eye(cols, dtype=np.uint8)])
    rref, pivot_cols = _gf2_rref(augmented)
    
    # Extract kernel vectors from the identity part
    kernel_vectors = []
    for i in range(cols):
        row_sum = np.sum(rref[i, :rows])
        if row_sum == 0:
            kernel_vectors.append(rref[i, rows:])
    
    if not kernel_vectors:
        return np.zeros((0, cols), dtype=np.uint8)
    
    return np.array(kernel_vectors, dtype=np.uint8)


def _gf2_rowspace(matrix: np.ndarray) -> np.ndarray:
    """
    Compute a basis for the row space of a matrix over GF(2).
    """
    if matrix.size == 0:
        return np.zeros((0, matrix.shape[1] if matrix.ndim == 2 else 0), dtype=np.uint8)
    
    rref, pivot_cols = _gf2_rref(matrix)
    non_zero_rows = [row for row in rref if np.any(row)]
    
    if not non_zero_rows:
        return np.zeros((0, matrix.shape[1]), dtype=np.uint8)
    
    return np.array(non_zero_rows, dtype=np.uint8)


def _gf2_rank(matrix: np.ndarray) -> int:
    """Compute the rank of a matrix over GF(2)."""
    if matrix.size == 0:
        return 0
    _, pivot_cols = _gf2_rref(matrix)
    return len(pivot_cols)


def _in_rowspace(vector: np.ndarray, matrix: np.ndarray) -> bool:
    """Check if a vector is in the row space of a matrix over GF(2)."""
    if matrix.size == 0:
        return np.allclose(vector, 0)
    
    augmented = np.vstack([matrix, vector.reshape(1, -1)])
    return _gf2_rank(augmented) == _gf2_rank(matrix)


def _symplectic_inner_product(v1: np.ndarray, v2: np.ndarray) -> int:
    """
    Compute symplectic inner product of two vectors.
    
    For vectors of length 2n representing [X | Z]:
    <v1, v2> = v1_X @ v2_Z + v1_Z @ v2_X (mod 2)
    
    Returns 0 if they commute, 1 if they anticommute.
    """
    n = len(v1) // 2
    x1, z1 = v1[:n], v1[n:]
    x2, z2 = v2[:n], v2[n:]
    return (np.dot(x1, z2) + np.dot(z1, x2)) % 2


def _pauli_string_to_symplectic(pauli: PauliString, n: int) -> np.ndarray:
    """Convert a Pauli string to symplectic form [X | Z]."""
    result = np.zeros(2 * n, dtype=np.uint8)
    
    if isinstance(pauli, str):
        for i, p in enumerate(pauli):
            if p in ('X', 'Y'):
                result[i] = 1
            if p in ('Z', 'Y'):
                result[n + i] = 1
    else:  # Dict format
        for i, p in pauli.items():
            if p in ('X', 'Y'):
                result[i] = 1
            if p in ('Z', 'Y'):
                result[n + i] = 1
    
    return result


def _symplectic_to_pauli_string(vec: np.ndarray) -> PauliString:
    """Convert symplectic vector to PauliString dict."""
    n = len(vec) // 2
    x_part = vec[:n]
    z_part = vec[n:]
    
    pauli: PauliString = {}
    for i in range(n):
        if x_part[i] and z_part[i]:
            pauli[i] = 'Y'
        elif x_part[i]:
            pauli[i] = 'X'
        elif z_part[i]:
            pauli[i] = 'Z'
    
    return pauli


class GenericStabilizerCode(StabilizerCode):
    """
    User-constructed stabilizer code from symplectic stabilizer matrix.
    
    This class provides a complete implementation for constructing general
    stabilizer codes (both CSS and non-CSS) from a symplectic stabilizer
    matrix, with automatic logical operator inference when not provided.
    
    Parameters
    ----------
    stabilizer_matrix : np.ndarray
        Stabilizer generators in symplectic form, shape (m, 2*n) where
        m = number of stabilizers, n = number of qubits.
        Format: [X_part | Z_part] where entry [i,j]=1 means X on qubit j
        for j<n, or Z on qubit j-n for j>=n.
    logical_x : Optional[List[PauliString]]
        Logical X operators. If None, will be inferred.
    logical_z : Optional[List[PauliString]]
        Logical Z operators. If None, will be inferred.
    n_qubits : Optional[int]
        Number of physical qubits. If None, inferred from stabilizer_matrix.
    metadata : Optional[Dict[str, Any]]
        Additional metadata (name, distance, coordinates, etc.)
    
    Attributes
    ----------
    n : int
        Number of physical qubits
    k : int  
        Number of logical qubits
    is_css : bool
        Whether this is a CSS code (all stabilizers pure X or pure Z)
    
    Examples
    --------
    >>> # Create [[5,1,3]] Perfect code
    >>> stab_mat = np.array([
    ...     [1, 0, 0, 0, 1,  0, 1, 1, 0, 0],  # XZZXI
    ...     [0, 1, 0, 0, 1,  0, 0, 1, 1, 0],  # IXZZX
    ...     [1, 0, 1, 0, 0,  0, 0, 0, 1, 1],  # XIXZZ
    ...     [0, 1, 0, 1, 0,  1, 0, 0, 0, 1],  # ZXIXZ
    ... ], dtype=np.uint8)
    >>> code = GenericStabilizerCode(stab_mat)
    >>> print(code.n, code.k, code.is_css)  # 5 1 False
    """
    
    def __init__(
        self,
        stabilizer_matrix: np.ndarray,
        logical_x: Optional[List[PauliString]] = None,
        logical_z: Optional[List[PauliString]] = None,
        n_qubits: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Ensure proper array type
        self._stab_mat = np.asarray(stabilizer_matrix, dtype=np.uint8)
        
        # Validate shape
        if self._stab_mat.ndim != 2:
            raise ValueError("stabilizer_matrix must be 2D")
        
        if self._stab_mat.shape[1] % 2 != 0:
            raise ValueError("stabilizer_matrix must have even number of columns (2*n)")
        
        # Infer n
        if n_qubits is not None:
            self._n = n_qubits
            if self._stab_mat.shape[1] != 2 * n_qubits:
                raise ValueError(f"stabilizer_matrix width {self._stab_mat.shape[1]} != 2*n_qubits {2*n_qubits}")
        else:
            self._n = self._stab_mat.shape[1] // 2
        
        # Validate stabilizers commute
        self._validate_stabilizers_commute()
        
        # Infer k from stabilizer rank
        m = self._stab_mat.shape[0]
        rank = _gf2_rank(self._stab_mat)
        self._k = self._n - rank
        
        # Infer logical operators if not provided
        if logical_x is None or logical_z is None:
            inferred_x, inferred_z = self._infer_logicals()
            logical_x = logical_x if logical_x is not None else inferred_x
            logical_z = logical_z if logical_z is not None else inferred_z
        
        self._logical_x = logical_x
        self._logical_z = logical_z
        
        # Build metadata
        self._metadata = dict(metadata or {})
        self._metadata.setdefault("n", self._n)
        self._metadata.setdefault("k", self._k)
        self._metadata.setdefault("is_css", self.is_css)
        
        # Add default coordinates if not provided
        if "data_coords" not in self._metadata:
            self._metadata["data_coords"] = [(float(i), 0.0) for i in range(self._n)]
        
        # Validate logical operators
        self._validate_logicals()
    
    def _validate_stabilizers_commute(self) -> None:
        """Verify all stabilizer generators commute pairwise."""
        m = self._stab_mat.shape[0]
        for i in range(m):
            for j in range(i + 1, m):
                if _symplectic_inner_product(self._stab_mat[i], self._stab_mat[j]) != 0:
                    raise ValueError(f"Stabilizers {i} and {j} do not commute")
    
    def _infer_logicals(self) -> Tuple[List[PauliString], List[PauliString]]:
        """
        Infer logical operators using symplectic linear algebra.
        
        Algorithm:
        1. Find centralizer of stabilizers (operators that commute with all stabs)
        2. Remove stabilizers from centralizer to get logical space
        3. Pair logicals so (Lx_i, Lz_i) anticommute
        """
        n = self._n
        k = self._k
        
        if k <= 0:
            return [], []
        
        # Find centralizer: vectors v such that <v, s_i> = 0 for all stabilizers
        # This is the kernel of the symplectic form matrix
        # Build the symplectic form matrix: for each stab s, the row [z | x]
        symplectic_form = np.zeros((self._stab_mat.shape[0], 2 * n), dtype=np.uint8)
        for i, stab in enumerate(self._stab_mat):
            # Swap X and Z parts for symplectic form
            symplectic_form[i, :n] = stab[n:]  # Z part goes first
            symplectic_form[i, n:] = stab[:n]  # X part goes second
        
        # Find kernel (centralizer)
        centralizer = _gf2_kernel(symplectic_form)
        
        if len(centralizer) == 0:
            # Fallback: construct from scratch
            return self._infer_logicals_fallback()
        
        # Remove stabilizers from centralizer
        stab_rowspace = _gf2_rowspace(self._stab_mat)
        
        logical_candidates = []
        for vec in centralizer:
            if not _in_rowspace(vec, stab_rowspace):
                logical_candidates.append(vec)
        
        if len(logical_candidates) < 2 * k:
            return self._infer_logicals_fallback()
        
        # Pair logicals: find pairs that anticommute
        logical_x_list: List[PauliString] = []
        logical_z_list: List[PauliString] = []
        used = set()
        
        for i, v1 in enumerate(logical_candidates):
            if i in used or len(logical_x_list) >= k:
                break
            
            for j, v2 in enumerate(logical_candidates):
                if j in used or i == j:
                    continue
                
                # Check anticommutation
                if _symplectic_inner_product(v1, v2) == 1:
                    logical_x_list.append(_symplectic_to_pauli_string(v1))
                    logical_z_list.append(_symplectic_to_pauli_string(v2))
                    used.add(i)
                    used.add(j)
                    break
        
        return logical_x_list, logical_z_list
    
    def _infer_logicals_fallback(self) -> Tuple[List[PauliString], List[PauliString]]:
        """Simple fallback: all-X and all-Z as logicals if k=1."""
        if self._k <= 0:
            return [], []
        
        # Very simple: try all-X and all-Z
        all_x: PauliString = {i: 'X' for i in range(self._n)}
        all_z: PauliString = {i: 'Z' for i in range(self._n)}
        
        return [all_x] * self._k, [all_z] * self._k
    
    def _validate_logicals(self) -> None:
        """Validate logical operators."""
        n = self._n
        
        for i, lx in enumerate(self._logical_x):
            lx_sym = _pauli_string_to_symplectic(lx, n)
            
            # Check Lx commutes with all stabilizers
            for j, stab in enumerate(self._stab_mat):
                if _symplectic_inner_product(lx_sym, stab) != 0:
                    raise ValueError(f"Logical X[{i}] does not commute with stabilizer {j}")
        
        for i, lz in enumerate(self._logical_z):
            lz_sym = _pauli_string_to_symplectic(lz, n)
            
            # Check Lz commutes with all stabilizers
            for j, stab in enumerate(self._stab_mat):
                if _symplectic_inner_product(lz_sym, stab) != 0:
                    raise ValueError(f"Logical Z[{i}] does not commute with stabilizer {j}")
        
        # Check paired logicals anticommute
        for i, (lx, lz) in enumerate(zip(self._logical_x, self._logical_z)):
            lx_sym = _pauli_string_to_symplectic(lx, n)
            lz_sym = _pauli_string_to_symplectic(lz, n)
            
            if _symplectic_inner_product(lx_sym, lz_sym) != 1:
                raise ValueError(f"Logical X[{i}] and Z[{i}] do not anticommute")
    
    @property
    def n(self) -> int:
        """Number of physical qubits."""
        return self._n
    
    @property
    def k(self) -> int:
        """Number of logical qubits."""
        return self._k
    
    @property
    def stabilizer_matrix(self) -> np.ndarray:
        """Stabilizer generators in symplectic form."""
        return self._stab_mat
    
    def logical_x_ops(self) -> List[PauliString]:
        """Logical X operators."""
        return self._logical_x
    
    def logical_z_ops(self) -> List[PauliString]:
        """Logical Z operators."""
        return self._logical_z
    
    def qubit_coords(self) -> Optional[List[Tuple[float, float]]]:
        """Return 2D coordinates for data qubits."""
        return self._metadata.get("data_coords")
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata dict."""
        return self._metadata
    
    @classmethod
    def from_stabilizer_strings(
        cls,
        stabilizers: List[str],
        logical_x: Optional[List[str]] = None,
        logical_z: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "GenericStabilizerCode":
        """
        Create a GenericStabilizerCode from Pauli string representations.
        
        Parameters
        ----------
        stabilizers : List[str]
            Stabilizer generators as strings, e.g. ["XZZXI", "IXZZX", ...]
        logical_x : Optional[List[str]]
            Logical X operators as strings
        logical_z : Optional[List[str]]
            Logical Z operators as strings
        metadata : Optional[Dict[str, Any]]
            Additional metadata
        
        Returns
        -------
        GenericStabilizerCode
            The constructed code
        
        Examples
        --------
        >>> code = GenericStabilizerCode.from_stabilizer_strings(
        ...     ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"],
        ...     logical_x=["XXXXX"],
        ...     logical_z=["ZZZZZ"]
        ... )
        """
        if not stabilizers:
            raise ValueError("Must provide at least one stabilizer")
        
        n = len(stabilizers[0])
        for stab in stabilizers:
            if len(stab) != n:
                raise ValueError(f"All stabilizers must have same length, got {len(stab)} vs {n}")
        
        # Convert to symplectic matrix
        stab_mat = np.zeros((len(stabilizers), 2 * n), dtype=np.uint8)
        for i, stab in enumerate(stabilizers):
            for j, p in enumerate(stab):
                if p in ('X', 'Y'):
                    stab_mat[i, j] = 1
                if p in ('Z', 'Y'):
                    stab_mat[i, n + j] = 1
        
        # Convert logical operators
        lx_pauli = None
        lz_pauli = None
        
        if logical_x is not None:
            lx_pauli = []
            for lx in logical_x:
                pauli = {i: p for i, p in enumerate(lx) if p != 'I'}
                lx_pauli.append(pauli)
        
        if logical_z is not None:
            lz_pauli = []
            for lz in logical_z:
                pauli = {i: p for i, p in enumerate(lz) if p != 'I'}
                lz_pauli.append(pauli)
        
        return cls(stab_mat, lx_pauli, lz_pauli, n, metadata)
    
    @classmethod
    def from_code(cls, code: StabilizerCode, metadata: Optional[Dict[str, Any]] = None) -> "GenericStabilizerCode":
        """
        Create a GenericStabilizerCode from an existing StabilizerCode.
        
        Useful for wrapping/converting other code types.
        """
        meta = dict(code.metadata) if hasattr(code, 'metadata') else {}
        if metadata:
            meta.update(metadata)
        
        # Handle both method and property for logical ops
        lx = code.logical_x_ops() if callable(code.logical_x_ops) else code.logical_x_ops
        lz = code.logical_z_ops() if callable(code.logical_z_ops) else code.logical_z_ops
        
        return cls(
            stabilizer_matrix=code.stabilizer_matrix,
            logical_x=list(lx) if lx else None,
            logical_z=list(lz) if lz else None,
            n_qubits=code.n,
            metadata=meta,
        )
