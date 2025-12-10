"""[[7,1,3]] Steane Code (Triangular Color Code)

The 7-qubit Steane code is a distance-3 CSS code that encodes 1 logical qubit
and can correct any single-qubit error. It is constructed from the classical
[7,4,3] Hamming code and is equivalent to the 2D color code on a triangle
(the smallest color code).

There are 3 independent X-type and 3 Z-type stabilizers, each of weight 4,
which correspond to the faces of a triangular tiling.

Qubit layout (triangular):
        0
       / \\
      2---6
     / \\ / \\
    1---5---3
         \\
          4

Stabilizers (weight 4, corresponding to faces):
- S_X^0, S_Z^0: qubits {3,4,5,6} - bottom right face
- S_X^1, S_Z^1: qubits {1,2,5,6} - bottom left face  
- S_X^2, S_Z^2: qubits {0,2,4,6} - top face

Logical operators (minimum weight 3):
- L_X: qubits {0,1,3} - left edge chain
- L_Z: qubits {0,1,3} - left edge chain (self-dual code)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex3


class SteanCode713(TopologicalCSSCode):
    """
    [[7,1,3]] Steane code (triangular color code).

    The Steane code is a distance-3 CSS code with 7 physical qubits and 1 logical qubit.
    It can correct any single Pauli error and has transversal Clifford gates.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the Steane code with proper CSS structure and chain complex.

        Stabilizers from Hamming [7,4,3] code:
        X stabilizers: {3,4,5,6}, {1,2,5,6}, {0,2,4,6}
        Z stabilizers: {3,4,5,6}, {1,2,5,6}, {0,2,4,6}
        (Self-dual CSS code)
        """
        # X-type check matrix (3 checks, each on 4 qubits)
        # Matches the standard [7,4,3] Hamming code parity check matrix
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

        # Build chain complex
        n_qubits = 7
        boundary_2_x = hx.T.astype(np.uint8)  # shape (7, 3)
        boundary_2_z = hz.T.astype(np.uint8)  # shape (7, 3)
        boundary_2 = np.concatenate([boundary_2_x, boundary_2_z], axis=1)
        boundary_1 = np.zeros((0, n_qubits), dtype=np.uint8)
        
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        # Logical operators - use minimum weight representatives (weight 3)
        logical_x = ["XXXIIII"]  # Logical X on qubits {0,1,2}, weight 3
        logical_z = ["ZZZIIII"]  # Logical Z on qubits {0,1,2}, weight 3

        # Geometric layout: triangular arrangement
        coords = {
            0: (1.0, 2.0),    # top vertex
            1: (0.0, 0.0),    # bottom-left vertex
            2: (0.5, 1.0),    # left edge center
            3: (2.0, 0.0),    # bottom-right vertex
            4: (1.5, 1.0),    # right edge center
            5: (1.0, 0.0),    # bottom edge center
            6: (1.0, 1.0),    # face center
        }
        data_coords = [coords[i] for i in range(7)]
        
        # Face centers for stabilizer coordinates (color code faces)
        stab_coords = [
            (1.5, 0.5),  # Face {3,4,5,6}
            (0.5, 0.5),  # Face {1,2,5,6}
            (1.0, 1.33), # Face {0,2,4,6}
        ]

        meta = dict(metadata or {})
        meta["name"] = "Steane_713"
        meta["n"] = 7
        meta["k"] = 1
        meta["distance"] = 3
        meta["is_colour_code"] = True
        meta["tiling"] = "triangular"
        meta["data_coords"] = data_coords
        meta["x_stab_coords"] = stab_coords
        meta["z_stab_coords"] = stab_coords  # Self-dual
        meta["logical_x_support"] = [0, 1, 2]
        meta["logical_z_support"] = [0, 1, 2]
        
        # NOTE: We deliberately omit x_schedule/z_schedule here.
        # The geometric schedule approach requires stabilizer coords + offsets to exactly
        # match data qubit coords, which is complex for colour codes. Instead, we use
        # the fallback matrix-based circuit construction in CSSMemoryExperiment.

        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)
        
        # Override parity check matrices
        self._hx = hx.astype(np.uint8)
        self._hz = hz.astype(np.uint8)
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates for visualization."""
        return list(self.metadata.get("data_coords", []))
