"""[[5,1,3]] Perfect Code - The smallest non-CSS quantum code

The [[5,1,3]] perfect code is the smallest quantum error-correcting code that
can correct any single-qubit error. It is called "perfect" because it saturates
the quantum Hamming bound.

Unlike CSS codes, the stabilizers mix X and Z operators. The code has 4 stabilizers
of weight 4, encodes 1 logical qubit in 5 physical qubits, and has distance 3.

Stabilizers (cyclic):
  g1 = XZZXI
  g2 = IXZZX
  g3 = XIXZZ
  g4 = ZXIXZ

Logical operators:
  X_L = XXXXX
  Z_L = ZZZZZ
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import math

import numpy as np

from qectostim.codes.abstract_code import StabilizerCode, PauliString

Coord2D = Tuple[float, float]


class PerfectCode513(StabilizerCode):
    """
    [[5,1,3]] Perfect code - a non-CSS stabilizer code.

    The smallest quantum error-correcting code, saturating the quantum
    Hamming bound. Stabilizers are mixed X/Z weight-4 operators.
    
    The stabilizers in symplectic form [X_part | Z_part]:
      g1 = XZZXI -> X: 10010, Z: 01100 
      g2 = IXZZX -> X: 01001, Z: 00110
      g3 = XIXZZ -> X: 10100, Z: 00011
      g4 = ZXIXZ -> X: 01010, Z: 10001
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialize the [[5,1,3]] perfect code."""
        
        # Full stabilizer generators in symplectic form [X | Z]
        # Each row: [X_0, X_1, X_2, X_3, X_4, Z_0, Z_1, Z_2, Z_3, Z_4]
        self._stabilizer_matrix = np.array([
            # g1 = XZZXI -> X on (0,3), Z on (1,2)
            [1, 0, 0, 1, 0,  0, 1, 1, 0, 0],
            # g2 = IXZZX -> X on (1,4), Z on (2,3)
            [0, 1, 0, 0, 1,  0, 0, 1, 1, 0],
            # g3 = XIXZZ -> X on (0,2), Z on (3,4)
            [1, 0, 1, 0, 0,  0, 0, 0, 1, 1],
            # g4 = ZXIXZ -> X on (1,3), Z on (0,4)
            [0, 1, 0, 1, 0,  1, 0, 0, 0, 1],
        ], dtype=np.uint8)
        
        # Logical operators (weight 5, transversal)
        self._logical_x = [{i: 'X' for i in range(5)}]  # XXXXX
        self._logical_z = [{i: 'Z' for i in range(5)}]  # ZZZZZ
        
        # Metadata
        meta = dict(metadata or {})
        meta["name"] = "Perfect_513"
        meta["n"] = 5
        meta["k"] = 1
        meta["distance"] = 3
        meta["is_css"] = False
        meta["full_stabilizers"] = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
        # Decoder compatibility: Only 16% of L0 errors trigger 0 detectors
        # This code CAN be effectively decoded unlike other non-CSS codes
        meta["decoder_compatible"] = True
        meta["naked_l0_percentage"] = 16
        
        # Pentagon geometry for visualization
        coords = []
        for i in range(5):
            angle = math.pi / 2 + 2 * math.pi * i / 5
            coords.append((math.cos(angle), math.sin(angle)))
        meta["data_coords"] = coords
        
        self._metadata = meta

    @property
    def n(self) -> int:
        """Number of physical qubits."""
        return 5
    
    @property
    def k(self) -> int:
        """Number of logical qubits."""
        return 1
    
    @property
    def distance(self) -> int:
        """Code distance."""
        return 3
    
    @property
    def stabilizer_matrix(self) -> np.ndarray:
        """Stabilizer generators in symplectic form [X_part | Z_part]."""
        return self._stabilizer_matrix
    
    def logical_x_ops(self) -> List[PauliString]:
        """Logical X operators."""
        return self._logical_x
    
    def logical_z_ops(self) -> List[PauliString]:
        """Logical Z operators."""
        return self._logical_z
    
    def qubit_coords(self) -> Optional[List[Coord2D]]:
        """Return pentagon coordinates for visualization."""
        return self._metadata.get("data_coords")
