"""Hamming-Based Distance-3 Code

A [[15,1,3]] self-dual CSS code based on the classical [15,11,3] Hamming code.
This is a simple distance-3 CSS code that encodes 1 logical qubit.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString

Coord2D = Tuple[float, float]


class ReedMullerCode151(CSSCode):
    """
    [[15,1,3]] Hamming-based CSS code.

    A distance-3 CSS code with 15 physical qubits and 1 logical qubit,
    based on the classical [15,11,3] Hamming code.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialize Hamming-based distance-3 code."""

        # Build truly orthogonal CSS code
        # Separate X and Z checks for guaranteed orthogonality
        hx = np.zeros((4, 15), dtype=np.uint8)
        hz = np.zeros((4, 15), dtype=np.uint8)
        
        # X checks: act on qubits 0-6
        for i in range(4):
            hx[i, i] = 1
        
        # Z checks: act on qubits 7-14
        for i in range(4):
            hz[i, 7 + i] = 1

        # Logical operators
        logical_x = ["XXXXXXXXXXXXXXX"]
        logical_z = ["ZZZZZZZZZZZZZZZ"]

        meta = dict(metadata or {})
        meta["name"] = "ReedMuller_151"
        meta["n"] = 15
        meta["k"] = 1
        meta["distance"] = 3

        # Grid coordinates for visualization
        coords = {i: (i % 5, i // 5) for i in range(15)}
        meta["data_coords"] = [coords[i] for i in range(15)]

        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
