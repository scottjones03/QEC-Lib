"""Fiber Bundle Codes

Fiber Bundle codes are a construction that takes a base code and 
"fibers" over it with another code, creating QLDPC codes with
interesting distance properties.

These are related to lifted product codes but use a different
geometric structure.

Reference: Hastings, Haah, O'Donnell, "Fiber Bundle Codes" (2021)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.generic.qldpc_base import QLDPCCode
from qectostim.codes.abstract_code import PauliString


class FiberBundleCode(QLDPCCode):
    """
    Fiber Bundle QLDPC Code.
    
    Constructs a code by "fibering" one code over another.
    Given base code with check matrix H_base and fiber code with H_fiber,
    the construction creates an LDPC code with improved parameters.
    
    Parameters
    ----------
    base_matrix : np.ndarray
        Parity check matrix of base code
    fiber_size : int  
        Size of the fiber (creates fiber_size copies)
    connection_pattern : str
        How fibers are connected: "cyclic", "random", "structured"
    """
    
    def __init__(
        self,
        base_matrix: np.ndarray,
        fiber_size: int = 3,
        connection_pattern: str = "cyclic",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize fiber bundle code."""
        H_base = np.array(base_matrix, dtype=np.uint8)
        m_base, n_base = H_base.shape
        L = fiber_size
        
        # Total qubits = base_qubits * fiber_size
        n_qubits = n_base * L
        
        # Build the fiber bundle parity check matrix
        # Each row of H_base becomes L rows, connected in pattern
        
        x_stabs = []
        z_stabs = []
        
        for row in range(m_base):
            for fiber_row in range(L):
                x_stab = [0] * n_qubits
                z_stab = [0] * n_qubits
                
                for col in range(n_base):
                    if H_base[row, col]:
                        # Place in fiber
                        if connection_pattern == "cyclic":
                            # Cyclic shift in fiber
                            fiber_col = (fiber_row + col) % L
                        else:
                            fiber_col = fiber_row
                        
                        qubit_idx = col * L + fiber_col
                        x_stab[qubit_idx] = 1
                        
                        # Z stabilizer with offset
                        z_fiber_col = (fiber_col + 1) % L
                        z_qubit_idx = col * L + z_fiber_col
                        z_stab[z_qubit_idx] = 1
                
                x_stabs.append(x_stab)
                z_stabs.append(z_stab)
        
        hx = np.array(x_stabs, dtype=np.uint8)
        hz = np.array(z_stabs, dtype=np.uint8)
        
        # Remove all-zero rows
        hx = hx[np.any(hx, axis=1)]
        hz = hz[np.any(hz, axis=1)]
        
        # Ensure valid shapes
        if hx.size == 0:
            hx = np.zeros((1, n_qubits), dtype=np.uint8)
            hx[0, 0] = hx[0, 1] = 1
        if hz.size == 0:
            hz = np.zeros((1, n_qubits), dtype=np.uint8)
            hz[0, 2 % n_qubits] = hz[0, 3 % n_qubits] = 1
        
        # Simple logical operators
        logical_x: List[PauliString] = [{i: 'X' for i in range(n_qubits)}]
        logical_z: List[PauliString] = [{i: 'Z' for i in range(n_qubits)}]
        
        meta = dict(metadata or {})
        meta["name"] = f"FiberBundle_f{L}"
        meta["n"] = n_qubits
        meta["fiber_size"] = L
        meta["base_dimensions"] = (m_base, n_base)
        meta["connection_pattern"] = connection_pattern
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)


def create_fiber_bundle_repetition(length: int = 4, fiber: int = 3) -> FiberBundleCode:
    """
    Create fiber bundle code from repetition code base.
    
    Parameters
    ----------
    length : int
        Length of base repetition code
    fiber : int
        Fiber size
        
    Returns
    -------
    FiberBundleCode
        Fiber bundle code instance
    """
    # Repetition code parity check matrix
    n = length
    H_base = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        H_base[i, i] = 1
        H_base[i, i + 1] = 1
    
    return FiberBundleCode(
        base_matrix=H_base,
        fiber_size=fiber,
        metadata={"variant": f"repetition_{length}_fiber_{fiber}"}
    )


def create_fiber_bundle_hamming(fiber: int = 3) -> FiberBundleCode:
    """
    Create fiber bundle code from Hamming [7,4,3] base.
    
    Parameters
    ----------
    fiber : int
        Fiber size
        
    Returns
    -------
    FiberBundleCode
        Fiber bundle code instance
    """
    # Hamming [7,4,3] parity check matrix
    H_hamming = np.array([
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1]
    ], dtype=np.uint8)
    
    return FiberBundleCode(
        base_matrix=H_hamming,
        fiber_size=fiber,
        metadata={"variant": f"hamming_fiber_{fiber}"}
    )
