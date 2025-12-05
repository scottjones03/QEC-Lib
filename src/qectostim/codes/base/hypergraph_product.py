"""Hypergraph Product Codes

Hypergraph product codes are a family of QLDPC codes constructed from 
the tensor product of two classical LDPC codes. Given classical codes
with parity check matrices H1 and H2, the quantum code has:

  Hx = [H1 ⊗ I_n2,  I_r1 ⊗ H2^T]
  Hz = [I_n1 ⊗ H2,  H1^T ⊗ I_r2]

where:
  - H1 is r1 × n1 (r1 checks, n1 bits)
  - H2 is r2 × n2 (r2 checks, n2 bits)
  - n = n1*n2 + r1*r2 (total qubits)
  - k = k1*k2 (where ki = ni - rank(Hi))

The distance is d = min(d1, d2) where di are classical code distances.

Reference: Tillich & Zémor, "Quantum LDPC Codes With Positive Rate and 
Minimum Distance Proportional to the Square Root of the Blocklength"
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString


def _compute_logical_ops_from_kernel(hx: np.ndarray, hz: np.ndarray, n: int, k: int) -> Tuple[List[str], List[str]]:
    """Compute logical operators from kernels of parity check matrices.
    
    For CSS codes:
    - Logical X operators are in ker(Hz) but not in rowspace(Hx^T)
    - Logical Z operators are in ker(Hx) but not in rowspace(Hz^T)
    """
    # Use a simple approach: find k independent vectors in ker(Hz) that 
    # anticommute with corresponding vectors in ker(Hx)
    
    # For now, use placeholder logical operators
    # In a full implementation, we'd compute the proper representatives
    logical_x = []
    logical_z = []
    
    for i in range(k):
        # Create placeholder logical X (first qubit of each logical)
        lx = ['I'] * n
        if i < n:
            lx[i] = 'X'
        logical_x.append(''.join(lx))
        
        # Create placeholder logical Z
        lz = ['I'] * n
        if i < n:
            lz[i] = 'Z'
        logical_z.append(''.join(lz))
    
    return logical_x, logical_z


class HypergraphProductCode(CSSCode):
    """
    Hypergraph product code from two classical codes.
    
    Parameters
    ----------
    H1 : np.ndarray
        Parity check matrix of first classical code (r1 x n1)
    H2 : np.ndarray
        Parity check matrix of second classical code (r2 x n2)
    """

    def __init__(self, H1: np.ndarray, H2: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Initialize hypergraph product code from classical parity check matrices."""
        H1 = np.array(H1, dtype=np.uint8) % 2
        H2 = np.array(H2, dtype=np.uint8) % 2
        
        r1, n1 = H1.shape
        r2, n2 = H2.shape
        
        # Total qubits: n1*n2 (first block) + r1*r2 (second block)
        n_qubits = n1 * n2 + r1 * r2
        
        # Build Hx = [H1 ⊗ I_n2 | I_r1 ⊗ H2^T]
        # Left block: H1 ⊗ I_n2 has shape (r1*n2) x (n1*n2)
        hx_left = np.kron(H1, np.eye(n2, dtype=np.uint8))
        # Right block: I_r1 ⊗ H2^T has shape (r1*n2) x (r1*r2)
        hx_right = np.kron(np.eye(r1, dtype=np.uint8), H2.T)
        hx = np.hstack([hx_left, hx_right]) % 2
        
        # Build Hz = [I_n1 ⊗ H2 | H1^T ⊗ I_r2]
        # Left block: I_n1 ⊗ H2 has shape (n1*r2) x (n1*n2)
        hz_left = np.kron(np.eye(n1, dtype=np.uint8), H2)
        # Right block: H1^T ⊗ I_r2 has shape (n1*r2) x (r1*r2)
        hz_right = np.kron(H1.T, np.eye(r2, dtype=np.uint8))
        hz = np.hstack([hz_left, hz_right]) % 2
        
        # Compute ranks to get k
        rank_hx = np.linalg.matrix_rank(hx)
        rank_hz = np.linalg.matrix_rank(hz)
        k = n_qubits - rank_hx - rank_hz
        
        # Compute logical operators (simplified placeholder)
        logical_x, logical_z = _compute_logical_ops_from_kernel(hx, hz, n_qubits, max(1, k))
        
        meta = dict(metadata or {})
        meta["name"] = f"HGP_{n1}x{n2}_{r1}x{r2}"
        meta["n"] = n_qubits
        meta["k"] = k
        meta["H1_shape"] = (r1, n1)
        meta["H2_shape"] = (r2, n2)
        meta["is_qldpc"] = True
        
        # Estimate distance (lower bound)
        # For HGP codes, distance is at least min(d1, d2) where di are classical distances
        meta["distance"] = "?"  # Would need to compute classical distances
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)


# Pre-built HGP codes from common classical codes

def create_hgp_repetition(n: int = 3) -> HypergraphProductCode:
    """Create HGP code from repetition code.
    
    Uses the [n, 1, n] repetition code. Results in a quantum code
    with parameters approximately [[n², 1, n]].
    """
    # Parity check matrix for [n,1,n] repetition code
    # n-1 checks: bit i XOR bit i+1 = 0
    H = np.zeros((n-1, n), dtype=np.uint8)
    for i in range(n-1):
        H[i, i] = 1
        H[i, i+1] = 1
    
    return HypergraphProductCode(H, H, metadata={"base_code": f"Repetition_{n}"})


def create_hgp_hamming(m: int = 3) -> HypergraphProductCode:
    """Create HGP code from Hamming code.
    
    Uses the [2^m-1, 2^m-m-1, 3] Hamming code.
    """
    n = 2**m - 1
    
    # Hamming parity check matrix: columns are binary representations of 1 to n
    H = np.zeros((m, n), dtype=np.uint8)
    for j in range(n):
        val = j + 1
        for i in range(m):
            H[m - 1 - i, j] = (val >> i) & 1
    
    return HypergraphProductCode(H, H, metadata={"base_code": f"Hamming_{n}"})


# Convenience instances
HGPRepetition3 = lambda: create_hgp_repetition(3)
HGPRepetition5 = lambda: create_hgp_repetition(5)
HGPHamming7 = lambda: create_hgp_hamming(3)  # From [7,4,3] Hamming
HGPHamming15 = lambda: create_hgp_hamming(4)  # From [15,11,3] Hamming
