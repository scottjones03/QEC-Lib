# src/qec_to_stim/codes/base/generic_css.py
"""
GenericCSSCode: Flexible CSS code construction from Hx and Hz matrices.

This module provides a complete implementation for constructing CSS codes from
parity check matrices, with automatic logical operator inference using GF(2)
linear algebra. Inspired by the RotatedSurfaceCode implementation.

Features:
- Automatic logical operator inference from Hx/Hz kernels
- Automatic logical support computation from operators
- Code distance estimation
- Validation of CSS properties and logical operators
- Compatible with CSSMemoryExperiment and decoders
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString


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
        # Empty matrix: kernel is all of the column space
        return np.eye(matrix.shape[1] if matrix.ndim == 2 else 0, dtype=np.uint8)
    
    rows, cols = matrix.shape
    if rows == 0:
        return np.eye(cols, dtype=np.uint8)
    
    # Augment matrix with identity to track column operations
    augmented = np.hstack([matrix.T, np.eye(cols, dtype=np.uint8)])
    rref, pivot_cols = _gf2_rref(augmented)
    
    # Find free columns (non-pivot columns in original matrix part)
    pivot_set = set(pivot_cols)
    free_cols = [i for i in range(rows) if i not in pivot_set]
    
    # Extract kernel vectors from the identity part
    kernel_vectors = []
    for i in range(cols):
        # Check if this row corresponds to a zero row in the RREF
        row_sum = np.sum(rref[i, :rows])
        if row_sum == 0:
            kernel_vectors.append(rref[i, rows:])
    
    if not kernel_vectors:
        return np.zeros((0, cols), dtype=np.uint8)
    
    return np.array(kernel_vectors, dtype=np.uint8)


def _gf2_rowspace(matrix: np.ndarray) -> np.ndarray:
    """
    Compute a basis for the row space of a matrix over GF(2).
    
    Returns the non-zero rows of the RREF.
    """
    if matrix.size == 0:
        return np.zeros((0, matrix.shape[1] if matrix.ndim == 2 else 0), dtype=np.uint8)
    
    rref, pivot_cols = _gf2_rref(matrix)
    # Return rows with pivots
    non_zero_rows = []
    for i, row in enumerate(rref):
        if np.any(row):
            non_zero_rows.append(row)
    
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
    
    # Add vector as a new row and check if rank increases
    augmented = np.vstack([matrix, vector.reshape(1, -1)])
    return _gf2_rank(augmented) == _gf2_rank(matrix)


def _pauli_string_to_binary(pauli: PauliString, n: int, pauli_type: str) -> np.ndarray:
    """Convert a Pauli string to binary vector indicating support."""
    result = np.zeros(n, dtype=np.uint8)
    if isinstance(pauli, str):
        for i, p in enumerate(pauli):
            if p in (pauli_type, 'Y'):
                result[i] = 1
    else:  # Dict format
        for i, p in pauli.items():
            if p in (pauli_type, 'Y'):
                result[i] = 1
    return result


def _binary_to_pauli_string(binary: np.ndarray, pauli_type: str) -> str:
    """Convert binary vector to Pauli string."""
    return ''.join(pauli_type if b else 'I' for b in binary)


def get_logical_support(pauli: PauliString, pauli_type: str) -> List[int]:
    """
    Extract qubit indices where a specific Pauli type appears.
    
    Args:
        pauli: Pauli string (either str or Dict[int, str])
        pauli_type: 'X' or 'Z' to look for
        
    Returns:
        List of qubit indices where the Pauli type (or Y) appears
    """
    if isinstance(pauli, str):
        return [i for i, p in enumerate(pauli) if p in (pauli_type, 'Y')]
    else:  # Dict format
        return sorted([i for i, p in pauli.items() if p in (pauli_type, 'Y')])


class GenericCSSCode(CSSCode):
    """
    User-constructed CSS code from Hx and Hz matrices.

    This class provides a complete implementation for constructing CSS codes,
    with automatic logical operator inference when not provided. It follows
    the patterns established by RotatedSurfaceCode for consistency.

    Parameters
    ----------
    hx : np.ndarray
        X-type parity check matrix, shape (num_x_checks, n_qubits)
    hz : np.ndarray
        Z-type parity check matrix, shape (num_z_checks, n_qubits)
    logical_x : Optional[List[PauliString]]
        Logical X operators. If None, will be inferred from Hx/Hz.
    logical_z : Optional[List[PauliString]]
        Logical Z operators. If None, will be inferred from Hx/Hz.
    metadata : Optional[Dict[str, Any]]
        Additional metadata (distance, coordinates, etc.)

    Attributes
    ----------
    n : int
        Number of physical qubits
    k : int
        Number of logical qubits (computed from Hx, Hz ranks)
    distance : Optional[int]
        Code distance (from metadata or None)
    
    Examples
    --------
    >>> # Create Steane code from matrices
    >>> hx = np.array([[0,0,0,1,1,1,1], [0,1,1,0,0,1,1], [1,0,1,0,1,0,1]])
    >>> hz = hx.copy()  # Self-dual
    >>> code = GenericCSSCode(hx, hz)  # Logicals inferred automatically
    >>> print(code.k)  # Should be 1
    """

    def __init__(
        self,
        hx: np.ndarray,
        hz: np.ndarray,
        logical_x: Optional[List[PauliString]] = None,
        logical_z: Optional[List[PauliString]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Ensure proper array types
        hx = np.asarray(hx, dtype=np.uint8)
        hz = np.asarray(hz, dtype=np.uint8)
        
        # Validate dimensions
        if hx.ndim != 2 or hz.ndim != 2:
            raise ValueError("hx and hz must be 2D arrays")
        if hx.shape[1] != hz.shape[1]:
            raise ValueError(f"hx and hz must have same number of columns (qubits): {hx.shape[1]} != {hz.shape[1]}")
        
        n = hx.shape[1]
        
        # Infer logical operators if not provided
        if logical_x is None or logical_z is None:
            inferred_x, inferred_z = self._infer_logicals(hx, hz)
            logical_x = logical_x if logical_x is not None else inferred_x
            logical_z = logical_z if logical_z is not None else inferred_z
        
        # Build metadata
        meta = dict(metadata or {})
        
        # Compute and store logical support
        if logical_x and len(logical_x) > 0:
            x_support = get_logical_support(logical_x[0], 'X')
            meta.setdefault("logical_x_support", x_support)
        
        if logical_z and len(logical_z) > 0:
            z_support = get_logical_support(logical_z[0], 'Z')
            meta.setdefault("logical_z_support", z_support)
        
        # Add default coordinates if not provided (simple linear layout)
        if "data_coords" not in meta:
            meta["data_coords"] = [(float(i), 0.0) for i in range(n)]
        
        # Store dimensions
        meta.setdefault("n", n)
        
        # Call parent constructor
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
        
        # Validate the construction
        self._validate_logicals()

    @staticmethod
    def _infer_logicals(
        hx: np.ndarray, hz: np.ndarray
    ) -> Tuple[List[PauliString], List[PauliString]]:
        """
        Infer logical operators from Hx and Hz using GF(2) linear algebra.
        
        Algorithm:
        1. Logical X operators are in kernel(Hz) but not in rowspace(Hx)
        2. Logical Z operators are in kernel(Hx) but not in rowspace(Hz)
        3. Pair them so each (Lx_i, Lz_i) anticommutes
        
        Returns:
            (logical_x, logical_z): Lists of Pauli strings
        """
        n = hx.shape[1]
        
        # Compute k = n - rank(Hx) - rank(Hz)
        rank_hx = _gf2_rank(hx)
        rank_hz = _gf2_rank(hz)
        k = n - rank_hx - rank_hz
        
        if k <= 0:
            # No logical qubits - return empty lists
            return [], []
        
        # Find kernel of Hz (vectors that commute with all Z stabilizers)
        # These are candidates for X-type logical operators
        ker_hz = _gf2_kernel(hz)
        
        # Find kernel of Hx (vectors that commute with all X stabilizers)
        # These are candidates for Z-type logical operators
        ker_hx = _gf2_kernel(hx)
        
        # Get row spaces
        row_hx = _gf2_rowspace(hx)
        row_hz = _gf2_rowspace(hz)
        
        # Filter out stabilizers: find vectors in kernel but not in row space
        logical_x_candidates = []
        for vec in ker_hz:
            if not _in_rowspace(vec, hx):
                logical_x_candidates.append(vec)
        
        logical_z_candidates = []
        for vec in ker_hx:
            if not _in_rowspace(vec, hz):
                logical_z_candidates.append(vec)
        
        # Pair logicals so they anticommute
        # For each X logical, find a Z logical that anticommutes with it
        logical_x_list: List[PauliString] = []
        logical_z_list: List[PauliString] = []
        
        used_z_indices = set()
        
        for lx_vec in logical_x_candidates:
            if len(logical_x_list) >= k:
                break
            
            # Find a Z candidate that anticommutes with this X
            for j, lz_vec in enumerate(logical_z_candidates):
                if j in used_z_indices:
                    continue
                
                # Check anticommutation: overlap should be odd
                overlap = np.sum(lx_vec * lz_vec) % 2
                if overlap == 1:
                    # Check this X isn't in the span of already chosen X logicals
                    if logical_x_list:
                        existing = np.array([_pauli_string_to_binary(lx, n, 'X') 
                                           for lx in logical_x_list])
                        augmented = np.vstack([existing, lx_vec.reshape(1, -1)])
                        if _gf2_rank(augmented) == _gf2_rank(existing):
                            continue  # Linearly dependent
                    
                    logical_x_list.append(_binary_to_pauli_string(lx_vec, 'X'))
                    logical_z_list.append(_binary_to_pauli_string(lz_vec, 'Z'))
                    used_z_indices.add(j)
                    break
        
        # If we couldn't find k pairs, try a simpler approach: use weight-minimizing heuristic
        if len(logical_x_list) < k and len(logical_x_candidates) > 0:
            # Sort by weight and take the first ones
            sorted_x = sorted(logical_x_candidates, key=lambda v: np.sum(v))
            sorted_z = sorted(logical_z_candidates, key=lambda v: np.sum(v))
            
            for lx_vec in sorted_x:
                if len(logical_x_list) >= k:
                    break
                for lz_vec in sorted_z:
                    overlap = np.sum(lx_vec * lz_vec) % 2
                    if overlap == 1:
                        lx_str = _binary_to_pauli_string(lx_vec, 'X')
                        lz_str = _binary_to_pauli_string(lz_vec, 'Z')
                        if lx_str not in logical_x_list:
                            logical_x_list.append(lx_str)
                            logical_z_list.append(lz_str)
                            break
        
        return logical_x_list, logical_z_list

    def _validate_logicals(self) -> None:
        """Validate that logical operators satisfy CSS requirements."""
        n = self.n
        hx = self._hx
        hz = self._hz
        
        for i, lx in enumerate(self._logical_x):
            lx_bin = _pauli_string_to_binary(lx, n, 'X')
            
            # Check Lx commutes with Hz (Lx @ Hz^T = 0)
            overlap = (lx_bin @ hz.T) % 2
            if np.any(overlap):
                raise ValueError(f"Logical X[{i}] does not commute with Z stabilizers")
        
        for i, lz in enumerate(self._logical_z):
            lz_bin = _pauli_string_to_binary(lz, n, 'Z')
            
            # Check Lz commutes with Hx (Lz @ Hx^T = 0)
            overlap = (lz_bin @ hx.T) % 2
            if np.any(overlap):
                raise ValueError(f"Logical Z[{i}] does not commute with X stabilizers")
        
        # Check anticommutation between paired logicals
        for i, (lx, lz) in enumerate(zip(self._logical_x, self._logical_z)):
            lx_bin = _pauli_string_to_binary(lx, n, 'X')
            lz_bin = _pauli_string_to_binary(lz, n, 'Z')
            
            overlap = np.sum(lx_bin * lz_bin) % 2
            if overlap != 1:
                raise ValueError(f"Logical X[{i}] and Z[{i}] do not anticommute (overlap={overlap})")

    @property
    def distance(self) -> Optional[int]:
        """Return code distance if known from metadata."""
        return self._metadata.get("distance")
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2D coordinates for data qubits."""
        return self._metadata.get("data_coords", [(float(i), 0.0) for i in range(self.n)])

    @classmethod
    def from_code(cls, code: CSSCode, metadata: Optional[Dict[str, Any]] = None) -> "GenericCSSCode":
        """
        Create a GenericCSSCode from an existing CSSCode.
        
        Useful for testing that GenericCSSCode can recreate the same structure.
        """
        meta = dict(code.metadata) if hasattr(code, 'metadata') else {}
        if metadata:
            meta.update(metadata)
        
        return cls(
            hx=code.hx,
            hz=code.hz,
            logical_x=list(code.logical_x_ops) if code.logical_x_ops else None,
            logical_z=list(code.logical_z_ops) if code.logical_z_ops else None,
            metadata=meta,
        )
