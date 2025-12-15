"""Bivariate Bicycle (BB) Codes

Bivariate Bicycle codes are a family of QLDPC codes constructed from 
circulant matrices over the group Z_l × Z_m. They were recently proposed
by IBM as practical candidates for near-term quantum error correction.

The code is defined by polynomials A(x,y) and B(x,y) over Z_l × Z_m:
  A(x,y) = Σ x^{a_i} y^{b_i}
  B(x,y) = Σ x^{c_j} y^{d_j}

The parity check matrices are:
  Hx = [A | B]
  Hz = [B^T | A^T]

where A and B are (l×m) × (l×m) circulant matrices.

Key properties:
- LDPC: Each row/column has constant weight
- Good rates: k/n can approach constant
- Efficient decoding: BP+OSD works well

Reference: Bravyi et al., "High-threshold and low-overhead fault-tolerant 
quantum memory" (2024)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np

from qectostim.codes.generic.qldpc_base import QLDPCCode
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.utils import compute_css_logicals, vectors_to_paulis_x, vectors_to_paulis_z


def _circulant_from_polynomial(l: int, m: int, terms: List[Tuple[int, int]]) -> np.ndarray:
    """Build circulant matrix from polynomial terms over Z_l × Z_m.
    
    Parameters
    ----------
    l : int
        Size of first cyclic group
    m : int  
        Size of second cyclic group
    terms : List[Tuple[int, int]]
        List of (a, b) pairs representing x^a y^b terms
        
    Returns
    -------
    np.ndarray
        (l*m) × (l*m) circulant matrix
    """
    n = l * m
    matrix = np.zeros((n, n), dtype=np.uint8)
    
    for row in range(n):
        # Row index corresponds to (i, j) where row = i*m + j
        i, j = row // m, row % m
        
        for a, b in terms:
            # The column for term x^a y^b at position (i,j) is ((i+a) mod l, (j+b) mod m)
            col_i = (i + a) % l
            col_j = (j + b) % m
            col = col_i * m + col_j
            matrix[row, col] = (matrix[row, col] + 1) % 2
    
    return matrix


class BivariateBicycleCode(QLDPCCode):
    """
    Bivariate Bicycle QLDPC code.
    
    IBM's proposed QLDPC codes with excellent practical properties.
    Features constant row/column weight in parity check matrices.
    
    Parameters
    ----------
    l : int
        Size of first cyclic group (Z_l)
    m : int
        Size of second cyclic group (Z_m)
    A_terms : List[Tuple[int, int]]
        Terms (a_i, b_i) for polynomial A(x,y) = Σ x^{a_i} y^{b_i}
    B_terms : List[Tuple[int, int]]
        Terms (c_j, d_j) for polynomial B(x,y) = Σ x^{c_j} y^{d_j}
    """

    def __init__(
        self, 
        l: int, 
        m: int, 
        A_terms: List[Tuple[int, int]], 
        B_terms: List[Tuple[int, int]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize bivariate bicycle code."""
        
        # Build circulant matrices
        A = _circulant_from_polynomial(l, m, A_terms)
        B = _circulant_from_polynomial(l, m, B_terms)
        
        # Total qubits: 2 * l * m (two blocks)
        block_size = l * m
        n_qubits = 2 * block_size
        
        # Hx = [A | B]
        hx = np.hstack([A, B]) % 2
        
        # Hz = [B^T | A^T]
        hz = np.hstack([B.T, A.T]) % 2
        
        # Verify CSS condition: Hx @ Hz.T = 0
        # For BB codes: A @ B^T + B @ A^T = 0 (mod 2) by construction
        comm = (hx @ hz.T) % 2
        if np.any(comm):
            raise ValueError("BB code construction failed: Hx Hz^T != 0")
        
        # Compute k
        rank_hx = np.linalg.matrix_rank(hx)
        rank_hz = np.linalg.matrix_rank(hz)
        k = n_qubits - rank_hx - rank_hz
        
        # Compute proper logical operators using CSS kernel/image prescription
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
        except Exception:
            # Fallback to single-qubit placeholder if computation fails
            logical_x = [{0: 'X'}]
            logical_z = [{0: 'Z'}]
        
        meta = dict(metadata or {})
        meta["name"] = f"BB_{l}x{m}"
        meta["n"] = n_qubits
        meta["k"] = k
        meta["l"] = l
        meta["m"] = m
        meta["A_terms"] = A_terms
        meta["B_terms"] = B_terms
        meta["is_qldpc"] = True
        meta["row_weight"] = len(A_terms) + len(B_terms)
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
        
        # Store l, m for qubit_coords
        self._l = l
        self._m = m
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2D qubit coordinates using torus grid layout.
        
        The BB code has two blocks of l×m qubits. We layout:
        - Block 0: positions (i, j) for i in [0,l), j in [0,m)
        - Block 1: positions (i + l + 1, j) for offset separation
        """
        coords = []
        block_size = self._l * self._m
        
        # Block 0: (i, j) grid
        for idx in range(block_size):
            i, j = idx // self._m, idx % self._m
            coords.append((float(i), float(j)))
        
        # Block 1: offset by (l + 1, 0) for visual separation
        for idx in range(block_size):
            i, j = idx // self._m, idx % self._m
            coords.append((float(i + self._l + 1), float(j)))
        
        return coords


# Pre-built BB codes from literature

def create_bb_gross_code() -> BivariateBicycleCode:
    """Create the [[144, 12, 12]] Gross code (a famous BB code).
    
    This is one of the best-known BB codes, with:
    - n = 144 qubits
    - k = 12 logical qubits  
    - d = 12 distance
    - Rate = 12/144 = 1/12 ≈ 8.3%
    
    Parameters: l=12, m=6
    A(x,y) = 1 + x^3 + x^6 + x^9 (4 terms)
    B(x,y) = y + y^2 + x^4*y + x^8*y^2 (4 terms)
    """
    l, m = 12, 6
    A_terms = [(0, 0), (3, 0), (6, 0), (9, 0)]
    B_terms = [(0, 1), (0, 2), (4, 1), (8, 2)]
    
    code = BivariateBicycleCode(l, m, A_terms, B_terms)
    code._metadata["name"] = "Gross_144_12_12"
    code._metadata["distance"] = 12
    return code


def create_bb_small_12() -> BivariateBicycleCode:
    """Create a small [[72, 8, ?]] BB code.
    
    Good for testing: moderate size with reasonable parameters.
    Uses pure x-powers in A and pure y-powers in B for guaranteed k>0.
    """
    l, m = 6, 6
    # A = x + x^2 + x^3 (pure x-powers)
    A_terms = [(1, 0), (2, 0), (3, 0)]
    # B = y + y^2 + y^3 (pure y-powers)
    B_terms = [(0, 1), (0, 2), (0, 3)]
    
    code = BivariateBicycleCode(l, m, A_terms, B_terms)
    code._metadata["name"] = "BB_72_8"
    return code


def create_bb_tiny() -> BivariateBicycleCode:
    """Create a tiny BB code for fast testing.
    
    Parameters: l=3, m=3 giving n=18 qubits.
    """
    l, m = 3, 3
    A_terms = [(0, 0), (1, 0)]
    B_terms = [(0, 1), (1, 1)]
    
    code = BivariateBicycleCode(l, m, A_terms, B_terms)
    code._metadata["name"] = "BB_tiny_18"
    return code


# Convenience instances  
BBGrossCode = create_bb_gross_code
BBCode72 = create_bb_small_12
BBCodeTiny = create_bb_tiny
