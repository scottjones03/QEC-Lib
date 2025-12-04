"""[[7,1,3]] Steane Code (Triangular Color Code)

The 7-qubit Steane code is a distance-3 CSS code that encodes 1 logical qubit
and can correct any single-qubit error. It is constructed from the classical
[7,4,3] Hamming code and is equivalent to the 2D color code on a triangle
(the smallest color code).

There are 3 independent X-type and 3 Z-type stabilizers, each of weight 4,
which correspond to the faces of a triangular tiling.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString

Coord2D = Tuple[float, float]


class SteanCode713(CSSCode):
    """
    [[7,1,3]] Steane code (triangular color code).

    The Steane code is a distance-3 CSS code with 7 physical qubits and 1 logical qubit.
    It can correct any single Pauli error and has transversal Clifford gates.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the Steane code with proper CSS structure.

        Stabilizers from Hamming [7,4,3] code:
        X stabilizers: {3,4,5,6}, {1,2,5,6}, {0,2,4,6}
        Z stabilizers: {3,4,5,6}, {1,2,5,6}, {0,2,4,6}
        (Self-dual CSS code)
        """
        # X-type check matrix (3 checks, each on 4 qubits)
        hx = np.array([
            [0, 0, 0, 1, 1, 1, 1],  # qubits {3,4,5,6}
            [0, 1, 1, 0, 0, 1, 1],  # qubits {1,2,5,6}
            [1, 0, 1, 0, 1, 0, 1],  # qubits {0,2,4,6}
        ], dtype=np.uint8)

        # Z-type check matrix (3 checks, same support as X due to self-duality)
        hz = np.array([
            [0, 0, 0, 1, 1, 1, 1],  # qubits {3,4,5,6}
            [0, 1, 1, 0, 0, 1, 1],  # qubits {1,2,5,6}
            [1, 0, 1, 0, 1, 0, 1],  # qubits {0,2,4,6}
        ], dtype=np.uint8)

        # Logical operators
        # The all-X and all-Z strings are not stabilizers (they anticommute with certain checks)
        logical_x = ["XXXXXXX"]  # Single logical X on all qubits
        logical_z = ["ZZZZZZZ"]  # Single logical Z on all qubits

        meta = dict(metadata or {})
        meta["name"] = "Steane_713"
        meta["n"] = 7
        meta["k"] = 1
        meta["distance"] = 3

        # Geometric layout: triangular arrangement
        # Vertices at corners, face centers, and body center
        coords = {
            0: (1.0, 2.0),    # vertex
            1: (0.0, 0.0),    # vertex
            2: (0.5, 1.0),    # edge center
            3: (2.0, 0.0),    # vertex
            4: (1.5, 1.0),    # edge center
            5: (1.0, 0.0),    # edge center
            6: (1.0, 1.0),    # face center
        }
        meta["data_coords"] = [coords[i] for i in range(7)]

        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
