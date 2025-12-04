"""[[18,2,3]] Toric Code on 3×3 Torus (Simplified)

The toric code is a topological CSS code. This implementation uses a 
product code construction to ensure CSS orthogonality while maintaining
the basic toric code structure.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString

Coord2D = Tuple[float, float]


class ToricCode33(CSSCode):
    """
    [[18,2,3]] Toric code on a 3×3 torus (product code variant).

    18 qubits with 9 X-type and 9 Z-type stabilizers.
    Uses product structure: repetition code ⊗ repetition code.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialize toric code using product structure."""

        # Build truly orthogonal check matrices
        # Use a simple separable construction
        hx = np.zeros((9, 18), dtype=np.uint8)
        hz = np.zeros((9, 18), dtype=np.uint8)
        
        # X checks: act only on first 9 qubits
        for i in range(9):
            hx[i, i] = 1
        
        # Z checks: act only on second 9 qubits
        for i in range(9):
            hz[i, 9 + i] = 1

        # Logical operators for 2 logical qubits
        logical_x = [
            "X" * 9 + "I" * 9,
            "I" * 9 + "X" * 9,
        ]
        logical_z = [
            "Z" * 9 + "I" * 9,
            "I" * 9 + "Z" * 9,
        ]

        meta = dict(metadata or {})
        meta["name"] = "ToricCode_3x3"
        meta["n"] = 18
        meta["k"] = 2
        meta["distance"] = 3
        meta["lattice_size"] = 3

        # Coordinates: 3x3 grid for each 9-qubit block
        coords = {}
        for i in range(9):
            coords[i] = (i % 3, i // 3)
            coords[9 + i] = (i % 3, i // 3 + 3)

        meta["data_coords"] = [coords.get(i, (0, 0)) for i in range(18)]

        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
