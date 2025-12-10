"""Hamming-based CSS Codes

CSS codes constructed from classical Hamming codes. The [2^m-1, 2^m-m-1, 3]
Hamming code is self-dual under certain conditions, making it easy to
construct CSS codes.

The Steane [[7,1,3]] code is the m=3 case (from [7,4,3] Hamming).

General [[2^m-1, 2^m-2m-1, 3]] codes for m >= 3.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import math

import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex3


def _hamming_parity_check(m: int) -> np.ndarray:
    """Build parity check matrix for [2^m-1, 2^m-m-1, 3] Hamming code.
    
    Columns are binary representations of 1 to 2^m-1.
    """
    n = 2**m - 1
    H = np.zeros((m, n), dtype=np.uint8)
    for j in range(n):
        val = j + 1
        for i in range(m):
            H[m - 1 - i, j] = (val >> i) & 1
    return H


class HammingCSSCode(TopologicalCSSCode):
    """
    CSS code from Hamming code.
    
    Inherits from TopologicalCSSCode with chain complex structure.
    
    Parameters
    ----------
    m : int
        Hamming parameter. Creates [[2^m-1, 2^m-2m-1, 3]] code.
        Must be >= 3.
    """

    def __init__(self, m: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialize Hamming CSS code with chain complex."""
        if m < 3:
            raise ValueError(f"m must be >= 3, got {m}")
        
        n = 2**m - 1
        k = n - 2 * m  # = 2^m - 2m - 1
        
        # Hamming parity check matrix
        H = _hamming_parity_check(m)
        
        # For CSS code, use H for both X and Z checks
        # This works because Hamming code is self-orthogonal for m >= 3
        hx = H.copy()
        hz = H.copy()
        
        # Build chain complex for CSS code structure:
        #   C2 (X stabilizers) --∂2--> C1 (qubits) --∂1--> C0 (Z stabilizers)
        #
        # boundary_2 = Hx.T: maps faces (X stabs) → edges (qubits), shape (n, #X_checks)
        # boundary_1 = Hz:   maps edges (qubits) → vertices (Z stabs), shape (#Z_checks, n)
        boundary_2 = hx.T.astype(np.uint8)  # shape (n, m)
        boundary_1 = hz.astype(np.uint8)    # shape (m, n)
        
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)
        
        # Logical operators - minimum weight codewords of Hamming code
        # Weight-3 logical operators exist for all Hamming codes
        logical_x = []
        logical_z = []
        
        for _ in range(max(1, k)):
            lx = ['I'] * n
            lz = ['I'] * n
            for i in range(min(3, n)):
                lx[i] = 'X'
                lz[i] = 'Z'
            logical_x.append(''.join(lx))
            logical_z.append(''.join(lz))
        
        # Generate coordinates (circular layout)
        coords = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            coords.append((math.cos(angle), math.sin(angle)))
        
        # Stabilizer coordinates (at center with offset)
        stab_coords = []
        for i in range(m):
            angle = 2 * math.pi * i / m
            r = 0.3
            stab_coords.append((r * math.cos(angle), r * math.sin(angle)))
        
        meta = dict(metadata or {})
        meta["name"] = f"Hamming_CSS_{n}"
        meta["n"] = n
        meta["k"] = k
        meta["distance"] = 3  # Hamming codes have distance 3
        meta["hamming_m"] = m
        meta["data_coords"] = coords
        meta["x_stab_coords"] = stab_coords
        meta["z_stab_coords"] = stab_coords  # Self-dual
        
        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)
        
        # Override parity check matrices
        self._hx = hx.astype(np.uint8)
        self._hz = hz.astype(np.uint8)
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates for visualization."""
        return list(self.metadata.get("data_coords", []))


# Pre-built instances
HammingCSS7 = lambda: HammingCSSCode(m=3)   # [[7,1,3]] Steane
HammingCSS15 = lambda: HammingCSSCode(m=4)  # [[15,7,3]]
HammingCSS31 = lambda: HammingCSSCode(m=5)  # [[31,21,3]]
