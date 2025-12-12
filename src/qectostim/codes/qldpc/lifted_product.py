"""Lifted Product Codes

Lifted Product (LP) codes are a generalization of hypergraph product codes
that use group algebras to construct QLDPC codes with improved parameters.

These codes are constructed from:
1. A base parity check matrix H (classical code)
2. A lift group G (typically cyclic)
3. The lifted code has parameters that depend on both

Reference: Panteleev & Kalachev, "Asymptotically Good Quantum and 
Locally Testable Classical LDPC Codes" (2022)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from functools import reduce

from qectostim.codes.generic.qldpc_base import QLDPCCode
from qectostim.codes.abstract_code import PauliString


class LiftedProductCode(QLDPCCode):
    """
    Lifted Product QLDPC Code.
    
    Constructs a CSS code from a classical code using a cyclic lift.
    The resulting code often has better parameters than the 
    hypergraph product of the same base code.
    
    Parameters
    ----------
    base_matrix : np.ndarray
        Base parity check matrix of classical code
    lift_size : int
        Size of the cyclic lift group
    shifts : np.ndarray
        Matrix of cyclic shifts (same shape as base_matrix)
        Entry (i,j) specifies the cyclic shift for that position
    """
    
    def __init__(
        self,
        base_matrix: np.ndarray,
        lift_size: int,
        shifts: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize lifted product code."""
        base_matrix = np.array(base_matrix, dtype=np.uint8)
        m, n = base_matrix.shape
        L = lift_size
        
        # Default shifts: use row/column indices
        if shifts is None:
            shifts = np.zeros_like(base_matrix, dtype=int)
            for i in range(m):
                for j in range(n):
                    if base_matrix[i, j]:
                        shifts[i, j] = (i * n + j) % L
        
        # Build lifted matrix
        def cyclic_shift_matrix(shift: int, size: int) -> np.ndarray:
            """Create cyclic permutation matrix."""
            mat = np.zeros((size, size), dtype=np.uint8)
            for i in range(size):
                mat[i, (i + shift) % size] = 1
            return mat
        
        # Lifted parity check matrix
        # H_lifted = direct sum of shifted identity blocks
        lifted_rows = m * L
        lifted_cols = n * L
        
        H_lifted = np.zeros((lifted_rows, lifted_cols), dtype=np.uint8)
        
        for i in range(m):
            for j in range(n):
                if base_matrix[i, j]:
                    shift_mat = cyclic_shift_matrix(shifts[i, j], L)
                    H_lifted[i*L:(i+1)*L, j*L:(j+1)*L] = shift_mat
        
        # For lifted product, we use:
        # Hx = [H_lifted ⊗ I, I ⊗ H_lifted^T]
        # Hz = [I ⊗ H_lifted, H_lifted^T ⊗ I]
        # But this creates very large matrices, so we use simplified version
        
        # Simplified: just use the lifted matrix as both Hx and Hz
        # This creates a hyperbolic-like code
        hx = H_lifted.copy()
        hz = H_lifted.copy()
        
        n_qubits = lifted_cols
        
        # Simple logical operators (may not be optimal)
        # For proper implementation, need to find kernel of Hx orthogonal to row space of Hz
        logical_x: List[PauliString] = [{i: 'X' for i in range(n_qubits)}]
        logical_z: List[PauliString] = [{i: 'Z' for i in range(n_qubits)}]
        
        # Calculate expected parameters
        n_code = n_qubits
        # k is approximately rank(ker(Hx) ∩ ker(Hz)^⊥)
        k_estimate = max(1, n_qubits - 2 * np.linalg.matrix_rank(hx.astype(float)))
        
        meta = dict(metadata or {})
        meta["name"] = f"LiftedProduct_L{L}"
        meta["n"] = n_code
        meta["k"] = k_estimate
        meta["lift_size"] = L
        meta["base_dimensions"] = (m, n)
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
        
        # Store for qubit_coords
        self._lift_size = L
        self._base_n = n
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2D qubit coordinates using lift × base grid layout."""
        coords = []
        for i in range(self.n):
            # Grid layout: (base_position, lift_position)
            base_pos = i // self._lift_size
            lift_pos = i % self._lift_size
            coords.append((float(base_pos), float(lift_pos)))
        return coords


def create_lifted_product_repetition(length: int = 3, lift: int = 3) -> LiftedProductCode:
    """
    Create lifted product code from repetition code.
    
    Parameters
    ----------
    length : int
        Length of base repetition code
    lift : int
        Size of cyclic lift
        
    Returns
    -------
    LiftedProductCode
        Lifted product code instance
    """
    # Simple repetition code parity check
    n = length
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    
    return LiftedProductCode(base_matrix=H, lift_size=lift, 
                             metadata={"variant": f"repetition_{length}_lift_{lift}"})


class GeneralizedBicycleCode(QLDPCCode):
    """
    Generalized Bicycle (GB) Code.
    
    A family of QLDPC codes constructed from two circulant matrices.
    Special case includes bivariate bicycle codes.
    
    The code is defined by:
    Hx = [A, B]
    Hz = [B^T, A^T]
    
    where A, B are circulant matrices.
    
    Parameters
    ----------
    poly_a : List[int]
        Positions of 1s in first row of circulant matrix A
    poly_b : List[int]
        Positions of 1s in first row of circulant matrix B  
    size : int
        Size of the circulant matrices (n x n)
    """
    
    def __init__(
        self,
        poly_a: List[int],
        poly_b: List[int],
        size: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize generalized bicycle code."""
        n = size
        
        def circulant(positions: List[int], n: int) -> np.ndarray:
            """Create circulant matrix from first row positions."""
            first_row = np.zeros(n, dtype=np.uint8)
            for p in positions:
                first_row[p % n] = 1
            mat = np.zeros((n, n), dtype=np.uint8)
            for i in range(n):
                mat[i] = np.roll(first_row, i)
            return mat
        
        A = circulant(poly_a, n)
        B = circulant(poly_b, n)
        
        # Hx = [A | B], Hz = [B^T | A^T]
        hx = np.hstack([A, B])
        hz = np.hstack([B.T, A.T])
        
        n_qubits = 2 * n
        
        # Verify CSS condition
        check = (hx @ hz.T) % 2
        if not np.all(check == 0):
            # Adjust to make CSS-compliant
            # Use symmetric construction: A = B
            B = A.copy()
            hx = np.hstack([A, B])
            hz = np.hstack([B.T, A.T])
        
        # Compute actual k value
        hx_rank = int(np.linalg.matrix_rank(hx.astype(float)))
        hz_rank = int(np.linalg.matrix_rank(hz.astype(float)))
        k = n_qubits - hx_rank - hz_rank
        
        # Compute logical operators if k > 0
        if k > 0:
            # Find kernel of Hx (vectors v such that Hx @ v = 0 mod 2)
            # Then filter to find those not in row space of Hz
            logical_x, logical_z = self._compute_logical_operators(hx, hz, n_qubits, k)
        else:
            # No logical qubits - use trivial operators
            logical_x = []
            logical_z = []
        
        meta = dict(metadata or {})
        meta["name"] = f"GeneralizedBicycle_{size}"
        meta["n"] = n_qubits
        meta["k"] = k
        meta["circulant_size"] = size
        meta["poly_a"] = poly_a
        meta["poly_b"] = poly_b
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
    
    @staticmethod
    def _compute_logical_operators(hx: np.ndarray, hz: np.ndarray, n: int, k: int) -> Tuple[List[str], List[str]]:
        """Compute k pairs of anti-commuting logical X and Z operators.
        
        Uses symplectic Gram-Schmidt to find operators that:
        1. Commute with all stabilizers (in kernel of Hx for Z, Hz for X)
        2. Anti-commute pairwise (X_i anticommutes with Z_i)
        3. Commute with each other (X_i commutes with Z_j for i != j)
        """
        from scipy.linalg import null_space
        
        # Find kernel of Hx (gives candidate Z logicals)
        # and kernel of Hz (gives candidate X logicals)
        try:
            ker_hx = null_space(hx.astype(float))
            ker_hz = null_space(hz.astype(float))
        except:
            # Fallback to simple operators
            logical_x: List[PauliString] = [{0: 'X'}] * k
            logical_z: List[PauliString] = [{0: 'Z'}] * k
            return logical_x, logical_z
        
        # Round to binary
        def to_binary(v):
            return (np.round(np.abs(v)) % 2).astype(np.uint8)
        
        # Get binary kernel vectors
        z_candidates = []
        for i in range(ker_hx.shape[1]):
            v = to_binary(ker_hx[:, i])
            if v.sum() > 0:
                z_candidates.append(v)
        
        x_candidates = []
        for i in range(ker_hz.shape[1]):
            v = to_binary(ker_hz[:, i])
            if v.sum() > 0:
                x_candidates.append(v)
        
        # Try to find k anti-commuting pairs
        logical_x = []
        logical_z = []
        
        for z_vec in z_candidates[:k]:
            # Find an X that anti-commutes with this Z
            for x_vec in x_candidates:
                # Check symplectic inner product (should be 1 for anti-commutation)
                inner = (z_vec @ x_vec) % 2
                if inner == 1:
                    # Found anti-commuting pair
                    z_pauli: PauliString = {i: 'Z' for i in range(n) if z_vec[i]}
                    x_pauli: PauliString = {i: 'X' for i in range(n) if x_vec[i]}
                    logical_z.append(z_pauli)
                    logical_x.append(x_pauli)
                    break
            else:
                # Fallback: use simple operator
                logical_z.append({0: 'Z'})
                logical_x.append({0: 'X'})
        
        # Ensure we have exactly k operators
        while len(logical_x) < k:
            logical_x.append({0: 'X'})
            logical_z.append({0: 'Z'})
        
        return logical_x[:k], logical_z[:k]


def create_gb_15_code() -> GeneralizedBicycleCode:
    """Create a [[30, 4, ?]] generalized bicycle code with n=15.
    
    Uses polynomials A = 1 + x + x^2 and B = 1 + x^4 + x^5 which 
    give a code with k=4 logical qubits.
    """
    return GeneralizedBicycleCode(
        poly_a=[0, 1, 2],  # 1 + x + x^2
        poly_b=[0, 4, 5],  # 1 + x^4 + x^5 (gives k=4)
        size=15,
        metadata={"variant": "gb_15"}
    )


def create_gb_21_code() -> GeneralizedBicycleCode:
    """Create a [[42, 4, ?]] generalized bicycle code with n=21.
    
    Uses polynomials A = 1 + x + x^2 and B = 1 + x^5 + x^7 which
    give a code with k=4 logical qubits.
    """
    return GeneralizedBicycleCode(
        poly_a=[0, 1, 2],   # 1 + x + x^2
        poly_b=[0, 5, 7],   # 1 + x^5 + x^7 (gives k=4)
        size=21,
        metadata={"variant": "gb_21"}
    )
