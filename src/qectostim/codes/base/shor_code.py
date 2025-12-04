"""[[9,1,3]] Shor Code (Concatenated Repetition Code)

Shor's code is a [[9,1,3]] CSS code – the first quantum error-correcting code,
constructed by concatenating a 3-qubit bit-flip code with a 3-qubit phase-flip code.
It can correct one arbitrary qubit error (distance 3).

The qubits are organized in a 3×3 grid where:
- Each row is a bit-flip repetition code (Z errors detected by row parity)
- Each column is a phase-flip repetition code (X errors detected by column parity)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString

Coord2D = Tuple[float, float]


class ShorCode91(CSSCode):
    """
    [[9,1,3]] Shor code (first quantum error-correcting code).

    Qubits arranged in 3x3 grid:
      0  1  2
      3  4  5
      6  7  8

    Stabilizers:
    - Z stabilizers: parity checks within each row (weight-2)
    - X stabilizers: parity checks across rows (weight-6, ensuring phase coherence)
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialize Shor's code with proper CSS structure."""

        # Z-type stabilizers: 6 weight-2 checks within rows
        # Rows: {0,1}, {1,2} (row 0)
        #       {3,4}, {4,5} (row 1)
        #       {6,7}, {7,8} (row 2)
        hz = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0],  # {0,1}
            [0, 1, 1, 0, 0, 0, 0, 0, 0],  # {1,2}
            [0, 0, 0, 1, 1, 0, 0, 0, 0],  # {3,4}
            [0, 0, 0, 0, 1, 1, 0, 0, 0],  # {4,5}
            [0, 0, 0, 0, 0, 0, 1, 1, 0],  # {6,7}
            [0, 0, 0, 0, 0, 0, 0, 1, 1],  # {7,8}
        ], dtype=np.uint8)

        # X-type stabilizers: 2 weight-6 checks across rows
        # These enforce phase coherence by checking parity across row pairs
        hx = np.array([
            [1, 1, 1, 1, 1, 1, 0, 0, 0],  # rows 0&1: {0,1,2,3,4,5}
            [0, 0, 0, 1, 1, 1, 1, 1, 1],  # rows 1&2: {3,4,5,6,7,8}
        ], dtype=np.uint8)

        # Logical operators for single logical qubit
        # Logical X: X on all qubits in one pattern
        logical_x = ["XXXXXXXXX"]
        # Logical Z: Z on all qubits (or representative pattern)
        logical_z = ["ZZZZZZZZZ"]

        meta = dict(metadata or {})
        meta["name"] = "Shor_91"
        meta["n"] = 9
        meta["k"] = 1
        meta["distance"] = 3

        # 3x3 grid coordinates
        coords = {q: (q % 3, q // 3) for q in range(9)}
        meta["data_coords"] = [coords[i] for i in range(9)]

        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
