# src/qectostim/codes/utils.py
"""
GF(2) linear algebra utilities and helper functions for composite codes.

This module provides:
- Binary matrix operations: RREF, kernel, rank, nullspace
- Kronecker product for binary matrices
- Pauli string manipulation utilities
- Symplectic algebra tools

All matrix operations are performed over GF(2) (mod 2 arithmetic).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .abstract_code import PauliString


# ============================================================================
# GF(2) Matrix Operations
# ============================================================================

def gf2_rref(matrix: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Compute the reduced row echelon form (RREF) of a binary matrix over GF(2).
    
    Parameters
    ----------
    matrix : np.ndarray
        Binary matrix (entries 0 or 1).
        
    Returns
    -------
    rref : np.ndarray
        The reduced row echelon form over GF(2).
    pivot_cols : List[int]
        List of pivot column indices.
    
    Example
    -------
    >>> mat = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    >>> rref, pivots = gf2_rref(mat)
    >>> print(rref)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(pivots)
    [0, 1, 2]
    """
    mat = np.array(matrix, dtype=np.uint8) % 2
    nrows, ncols = mat.shape
    pivot_cols: List[int] = []
    
    row = 0
    for col in range(ncols):
        if row >= nrows:
            break
        # Find pivot in this column
        pivot_row = None
        for r in range(row, nrows):
            if mat[r, col]:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        
        # Swap rows
        mat[[row, pivot_row]] = mat[[pivot_row, row]]
        pivot_cols.append(col)
        
        # Eliminate all other entries in this column
        for r in range(nrows):
            if r != row and mat[r, col]:
                mat[r] = (mat[r] + mat[row]) % 2
        
        row += 1
    
    return mat, pivot_cols


def gf2_rank(matrix: np.ndarray) -> int:
    """
    Compute the rank of a binary matrix over GF(2).
    
    Parameters
    ----------
    matrix : np.ndarray
        Binary matrix.
        
    Returns
    -------
    int
        The GF(2) rank of the matrix.
    """
    if matrix.size == 0:
        return 0
    _, pivots = gf2_rref(matrix)
    return len(pivots)


def gf2_kernel(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the kernel (null space) of a binary matrix over GF(2).
    
    Finds all vectors v such that matrix @ v = 0 (mod 2).
    
    Parameters
    ----------
    matrix : np.ndarray
        Binary matrix of shape (m, n).
        
    Returns
    -------
    np.ndarray
        Matrix of shape (dim_kernel, n) whose rows span the kernel.
        Returns empty array of shape (0, n) if kernel is trivial.
    """
    mat = np.array(matrix, dtype=np.uint8) % 2
    nrows, ncols = mat.shape
    
    if nrows == 0:
        # Kernel is all of GF(2)^n
        return np.eye(ncols, dtype=np.uint8)
    
    rref, pivot_cols = gf2_rref(mat)
    
    # Free columns are those not in pivot_cols
    free_cols = [c for c in range(ncols) if c not in pivot_cols]
    
    if not free_cols:
        return np.zeros((0, ncols), dtype=np.uint8)
    
    # Build kernel basis vectors
    kernel_rows = []
    for fc in free_cols:
        vec = np.zeros(ncols, dtype=np.uint8)
        vec[fc] = 1
        # Back-substitute: for each pivot column, determine its value
        for row_idx, pc in enumerate(pivot_cols):
            if row_idx < rref.shape[0] and rref[row_idx, fc]:
                vec[pc] = 1
        kernel_rows.append(vec)
    
    return np.array(kernel_rows, dtype=np.uint8)


def gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    """
    Alias for gf2_kernel. Computes the null space over GF(2).
    """
    return gf2_kernel(matrix)


def gf2_rowspace(matrix: np.ndarray) -> np.ndarray:
    """
    Compute a basis for the row space of a binary matrix over GF(2).
    
    Parameters
    ----------
    matrix : np.ndarray
        Binary matrix.
        
    Returns
    -------
    np.ndarray
        Matrix whose rows form a basis for the row space.
    """
    if matrix.size == 0:
        return np.zeros((0, matrix.shape[1] if matrix.ndim > 1 else 0), dtype=np.uint8)
    
    rref, pivots = gf2_rref(matrix)
    return rref[:len(pivots)]


def gf2_colspace(matrix: np.ndarray) -> np.ndarray:
    """
    Compute a basis for the column space of a binary matrix over GF(2).
    
    Parameters
    ----------
    matrix : np.ndarray
        Binary matrix.
        
    Returns
    -------
    np.ndarray
        Matrix whose rows form a basis for the column space.
        (Equivalent to row space of the transpose.)
    """
    return gf2_rowspace(matrix.T)


def gf2_solve(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """
    Solve Ax = b over GF(2), returning one solution or None if no solution exists.
    
    Parameters
    ----------
    A : np.ndarray
        Binary coefficient matrix of shape (m, n).
    b : np.ndarray
        Binary target vector of shape (m,).
        
    Returns
    -------
    Optional[np.ndarray]
        A solution vector x of shape (n,), or None if no solution exists.
    """
    A = np.array(A, dtype=np.uint8) % 2
    b = np.array(b, dtype=np.uint8) % 2
    
    m, n = A.shape
    # Augment [A | b]
    aug = np.column_stack([A, b.reshape(-1, 1)])
    rref, pivots = gf2_rref(aug)
    
    # Check for inconsistency: pivot in augmented column
    for row_idx, pc in enumerate(pivots):
        if pc == n:  # Pivot in the 'b' column
            return None
    
    # Build solution
    x = np.zeros(n, dtype=np.uint8)
    for row_idx, pc in enumerate(pivots):
        if pc < n:
            x[pc] = rref[row_idx, n]
    
    return x


# ============================================================================
# Kronecker Product Utilities
# ============================================================================

def kron_gf2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute Kronecker (tensor) product of two binary matrices over GF(2).
    
    Parameters
    ----------
    A : np.ndarray
        First binary matrix of shape (m1, n1).
    B : np.ndarray
        Second binary matrix of shape (m2, n2).
        
    Returns
    -------
    np.ndarray
        Kronecker product of shape (m1*m2, n1*n2), reduced mod 2.
    """
    return (np.kron(A, B) % 2).astype(np.uint8)


def block_diag_gf2(blocks: List[np.ndarray]) -> np.ndarray:
    """
    Create a block diagonal matrix from a list of binary matrices.
    
    Parameters
    ----------
    blocks : List[np.ndarray]
        List of binary matrices to place on the diagonal.
        
    Returns
    -------
    np.ndarray
        Block diagonal matrix.
    """
    if not blocks:
        return np.zeros((0, 0), dtype=np.uint8)
    
    total_rows = sum(b.shape[0] for b in blocks)
    total_cols = sum(b.shape[1] for b in blocks)
    
    result = np.zeros((total_rows, total_cols), dtype=np.uint8)
    
    row_offset = 0
    col_offset = 0
    for block in blocks:
        r, c = block.shape
        result[row_offset:row_offset + r, col_offset:col_offset + c] = block % 2
        row_offset += r
        col_offset += c
    
    return result


# ============================================================================
# Symplectic Algebra
# ============================================================================

def symplectic_inner_product(v1: np.ndarray, v2: np.ndarray) -> int:
    """
    Compute the symplectic inner product of two vectors.
    
    For vectors in symplectic form [X_part | Z_part], computes:
        ⟨v1, v2⟩ = v1_x · v2_z + v1_z · v2_x (mod 2)
    
    Parameters
    ----------
    v1, v2 : np.ndarray
        Vectors of length 2n in symplectic form.
        
    Returns
    -------
    int
        0 if the operators commute, 1 if they anticommute.
    """
    n = len(v1) // 2
    x1, z1 = v1[:n], v1[n:]
    x2, z2 = v2[:n], v2[n:]
    return int((np.dot(x1, z2) + np.dot(z1, x2)) % 2)


def check_commutation(stab1: np.ndarray, stab2: np.ndarray) -> bool:
    """
    Check if two stabilizer generators commute.
    
    Parameters
    ----------
    stab1, stab2 : np.ndarray
        Stabilizer generators in symplectic form [X | Z].
        
    Returns
    -------
    bool
        True if they commute, False if they anticommute.
    """
    return symplectic_inner_product(stab1, stab2) == 0


# ============================================================================
# Pauli String Utilities
# ============================================================================

def pauli_to_symplectic(pauli: PauliString, n: int) -> np.ndarray:
    """
    Convert a PauliString to symplectic vector representation.
    
    Parameters
    ----------
    pauli : PauliString
        Dict mapping qubit indices to 'X', 'Y', or 'Z'.
    n : int
        Total number of qubits.
        
    Returns
    -------
    np.ndarray
        Symplectic vector of length 2n: [x_0, ..., x_{n-1}, z_0, ..., z_{n-1}].
    """
    vec = np.zeros(2 * n, dtype=np.uint8)
    for qubit, op in pauli.items():
        if op in ('X', 'Y'):
            vec[qubit] = 1
        if op in ('Z', 'Y'):
            vec[n + qubit] = 1
    return vec


def symplectic_to_pauli(vec: np.ndarray) -> PauliString:
    """
    Convert a symplectic vector to PauliString representation.
    
    Parameters
    ----------
    vec : np.ndarray
        Symplectic vector of length 2n.
        
    Returns
    -------
    PauliString
        Dict mapping qubit indices to Pauli operators.
    """
    n = len(vec) // 2
    x_part = vec[:n]
    z_part = vec[n:]
    
    pauli: PauliString = {}
    for i in range(n):
        has_x = bool(x_part[i])
        has_z = bool(z_part[i])
        if has_x and has_z:
            pauli[i] = 'Y'
        elif has_x:
            pauli[i] = 'X'
        elif has_z:
            pauli[i] = 'Z'
    return pauli


def pauli_weight(pauli: PauliString) -> int:
    """Return the weight (number of non-identity sites) of a Pauli string."""
    return len(pauli)


def pauli_support(pauli: PauliString) -> List[int]:
    """Return sorted list of qubit indices where Pauli is non-identity."""
    return sorted(pauli.keys())


def str_to_pauli(s: str) -> PauliString:
    """
    Convert a Pauli string like "XXIZZI" to a PauliString dict.
    
    Parameters
    ----------
    s : str
        String of Pauli operators (I, X, Y, Z).
        
    Returns
    -------
    PauliString
        Dict mapping qubit index to Pauli operator (non-identity only).
        
    Example
    -------
    >>> str_to_pauli("XXIZ")
    {0: 'X', 1: 'X', 3: 'Z'}
    >>> str_to_pauli("IIII")
    {}
    """
    return {i: op for i, op in enumerate(s) if op != 'I'}


def pauli_to_str(pauli: PauliString, n: int) -> str:
    """
    Convert a PauliString dict to a string of length n.
    
    Parameters
    ----------
    pauli : PauliString
        Dict mapping qubit index to Pauli operator.
    n : int
        Total number of qubits.
        
    Returns
    -------
    str
        String of Pauli operators of length n.
        
    Example
    -------
    >>> pauli_to_str({0: 'X', 1: 'X', 3: 'Z'}, 4)
    'XXIZ'
    """
    result = ['I'] * n
    for i, op in pauli.items():
        result[i] = op
    return ''.join(result)


def pauli_product(p1: PauliString, p2: PauliString, n: int) -> PauliString:
    """
    Compute the product of two Pauli strings (ignoring phase).
    
    Parameters
    ----------
    p1, p2 : PauliString
        Pauli strings to multiply.
    n : int
        Total number of qubits.
        
    Returns
    -------
    PauliString
        The product P1 * P2 (up to phase).
    """
    v1 = pauli_to_symplectic(p1, n)
    v2 = pauli_to_symplectic(p2, n)
    v_prod = (v1 + v2) % 2
    return symplectic_to_pauli(v_prod)


# ============================================================================
# Logical Operator Lifting Utilities
# ============================================================================

def lift_pauli_through_inner(
    outer_pauli: PauliString,
    inner_logical_x: List[PauliString],
    inner_logical_z: List[PauliString],
    n_inner: int,
) -> PauliString:
    """
    Lift a Pauli operator from outer code qubits to concatenated physical qubits.
    
    For each outer qubit q where outer_pauli acts:
    - X_q -> inner logical X on block q
    - Z_q -> inner logical Z on block q  
    - Y_q -> inner logical X * Z on block q (product)
    
    Parameters
    ----------
    outer_pauli : PauliString
        Pauli operator on the outer code's physical qubits.
    inner_logical_x : List[PauliString]
        Logical X operators of the inner code (one per logical qubit, typically just [0]).
    inner_logical_z : List[PauliString]
        Logical Z operators of the inner code.
    n_inner : int
        Number of physical qubits in the inner code.
        
    Returns
    -------
    PauliString
        Lifted Pauli operator on the concatenated physical qubits.
        
    Notes
    -----
    This assumes the inner code encodes 1 logical qubit (k_inner = 1).
    The first logical operator in inner_logical_x/z is used.
    """
    if not inner_logical_x or not inner_logical_z:
        raise ValueError("Inner code must have at least one logical operator")
    
    # Use the first logical pair
    log_x = inner_logical_x[0]
    log_z = inner_logical_z[0]
    
    result: PauliString = {}
    
    for outer_q, op in outer_pauli.items():
        offset = outer_q * n_inner
        
        if op == 'X':
            # Map to inner logical X
            for inner_q, inner_op in log_x.items():
                phys_q = offset + inner_q
                result[phys_q] = inner_op
        elif op == 'Z':
            # Map to inner logical Z
            for inner_q, inner_op in log_z.items():
                phys_q = offset + inner_q
                result[phys_q] = inner_op
        elif op == 'Y':
            # Map to inner logical X * Z
            # First apply log_x
            temp_x: Dict[int, str] = {}
            for inner_q, inner_op in log_x.items():
                temp_x[offset + inner_q] = inner_op
            # Then apply log_z
            temp_z: Dict[int, str] = {}
            for inner_q, inner_op in log_z.items():
                temp_z[offset + inner_q] = inner_op
            # Multiply them
            block_product = pauli_product(temp_x, temp_z, n_inner * (max(outer_pauli.keys()) + 1))
            for phys_q, phys_op in block_product.items():
                if phys_q in result:
                    # Multiply with existing
                    existing = {phys_q: result[phys_q]}
                    new = {phys_q: phys_op}
                    combined = pauli_product(existing, new, n_inner * (max(outer_pauli.keys()) + 1))
                    if phys_q in combined:
                        result[phys_q] = combined[phys_q]
                    else:
                        del result[phys_q]
                else:
                    result[phys_q] = phys_op
    
    return result


def binary_row_to_x_stabilizer(row: np.ndarray, offset: int = 0) -> PauliString:
    """
    Convert a binary row to an X-type Pauli stabilizer.
    
    Parameters
    ----------
    row : np.ndarray
        Binary row vector.
    offset : int
        Qubit index offset.
        
    Returns
    -------
    PauliString
        X-type Pauli operator.
    """
    return {offset + i: 'X' for i, bit in enumerate(row) if bit}


def binary_row_to_z_stabilizer(row: np.ndarray, offset: int = 0) -> PauliString:
    """
    Convert a binary row to a Z-type Pauli stabilizer.
    
    Parameters
    ----------
    row : np.ndarray
        Binary row vector.
    offset : int
        Qubit index offset.
        
    Returns
    -------
    PauliString
        Z-type Pauli operator.
    """
    return {offset + i: 'Z' for i, bit in enumerate(row) if bit}


# ============================================================================
# Utility Functions for Composite Codes
# ============================================================================

def css_intersection_check(hx: np.ndarray, hz: np.ndarray) -> bool:
    """
    Check that hx and hz satisfy the CSS constraint: hx @ hz.T = 0 (mod 2).
    
    Parameters
    ----------
    hx, hz : np.ndarray
        X and Z parity check matrices.
        
    Returns
    -------
    bool
        True if the CSS constraint is satisfied.
    """
    return np.all((hx @ hz.T) % 2 == 0)


def compute_css_logicals(
    hx: np.ndarray,
    hz: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Compute logical operators for a CSS code from Hx and Hz matrices.
    
    Uses the prescription:
    - Logical Z: vectors in ker(Hx) but not in rowspace(Hz)
    - Logical X: vectors in ker(Hz) but not in rowspace(Hx)
    
    Parameters
    ----------
    hx : np.ndarray
        X-type parity check matrix (detects Z errors).
    hz : np.ndarray
        Z-type parity check matrix (detects X errors).
        
    Returns
    -------
    logical_x : List[np.ndarray]
        Logical X operators as binary vectors.
    logical_z : List[np.ndarray]
        Logical Z operators as binary vectors.
    """
    n = hx.shape[1]
    
    # Logical Z lives in ker(Hx) / rowspace(Hz)
    # ker(Hx) = {v : Hx @ v = 0} - this is the RIGHT kernel
    # gf2_kernel computes right kernel, so use gf2_kernel(hx)
    ker_hx = gf2_kernel(hx)  # Vectors v with Hx @ v = 0
    
    # Similarly, logical X lives in ker(Hz) / rowspace(Hx)
    ker_hz = gf2_kernel(hz)  # Vectors v with Hz @ v = 0
    
    # Return kernel vectors that are not in the stabilizer row space
    logical_x = []
    logical_z = []
    
    # Check each kernel vector of Hz (potential logical X)
    row_hx = gf2_rowspace(hx)
    for v in ker_hz:
        # Check if v is in rowspace of Hx
        if row_hx.shape[0] > 0:
            # Try to express v as linear combo of row_hx
            aug = np.vstack([row_hx, v])
            if gf2_rank(aug) > gf2_rank(row_hx):
                logical_x.append(v)
        else:
            logical_x.append(v)
    
    # Check each kernel vector of Hx (potential logical Z)
    row_hz = gf2_rowspace(hz)
    for v in ker_hx:
        if row_hz.shape[0] > 0:
            aug = np.vstack([row_hz, v])
            if gf2_rank(aug) > gf2_rank(row_hz):
                logical_z.append(v)
        else:
            logical_z.append(v)
    
    return logical_x, logical_z


def vectors_to_paulis_x(vectors: List[np.ndarray]) -> List[PauliString]:
    """Convert list of binary vectors to X-type PauliStrings."""
    return [binary_row_to_x_stabilizer(v) for v in vectors]


def vectors_to_paulis_z(vectors: List[np.ndarray]) -> List[PauliString]:
    """Convert list of binary vectors to Z-type PauliStrings."""
    return [binary_row_to_z_stabilizer(v) for v in vectors]


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # GF(2) operations
    'gf2_rref',
    'gf2_rank',
    'gf2_kernel',
    'gf2_nullspace',
    'gf2_rowspace',
    'gf2_colspace',
    'gf2_solve',
    # Kronecker products
    'kron_gf2',
    'block_diag_gf2',
    # Symplectic algebra
    'symplectic_inner_product',
    'check_commutation',
    # Pauli utilities
    'pauli_to_symplectic',
    'symplectic_to_pauli',
    'pauli_weight',
    'pauli_support',
    'pauli_product',
    # Lifting
    'lift_pauli_through_inner',
    'binary_row_to_x_stabilizer',
    'binary_row_to_z_stabilizer',
    # CSS utilities
    'css_intersection_check',
    'compute_css_logicals',
    'vectors_to_paulis_x',
    'vectors_to_paulis_z',
]
