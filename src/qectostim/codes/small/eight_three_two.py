"""[[8,3,2]] Code

The [[8,3,2]] code is a CSS stabilizer code that encodes 
3 logical qubits in 8 physical qubits with distance 2.

It is one of the most efficient codes for its parameters.
Despite its small size, it can detect single-qubit errors.

Stabilizers (5 generators):
  X-type:
    g1 = XXXXXXXX (weight 8)
    g2 = XXXXI I I I (weight 4)
  Z-type:
    g3 = ZZZZIIII
    g4 = ZZIIZZII
    g5 = ZIZIZIZI
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString


class EightThreeTwoCode(CSSCode):
    """
    [[8,3,2]] CSS code.
    
    A compact CSS code encoding 3 logical qubits in 8 physical qubits.
    Distance 2 means it can detect but not correct single errors.
    
    This IS a valid CSS code since hx @ hz.T = 0 mod 2.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialize the [[8,3,2]] code."""
        n_qubits = 8
        
        # X-type stabilizer checks
        hx = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],  # XXXXXXXX (weight 8)
        ], dtype=np.uint8)
        
        # Z-type stabilizer checks  
        hz = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],  # ZZZZIIII
            [1, 1, 0, 0, 1, 1, 0, 0],  # ZZIIZZII
            [1, 0, 1, 0, 1, 0, 1, 0],  # ZIZIZIZI
            [0, 0, 0, 0, 1, 1, 1, 1],  # IIIIZZZZ
        ], dtype=np.uint8)
        
        # 3 logical qubits
        # These are representatives of the logical X and Z operators
        logical_x = [
            {0: 'X', 1: 'X'},  # L1_X
            {0: 'X', 2: 'X'},  # L2_X
            {0: 'X', 4: 'X'},  # L3_X
        ]
        
        logical_z = [
            {0: 'Z', 1: 'Z'},  # L1_Z
            {0: 'Z', 2: 'Z'},  # L2_Z
            {0: 'Z', 4: 'Z'},  # L3_Z
        ]
        
        meta = dict(metadata or {})
        meta["name"] = "Code_832"
        meta["n"] = 8
        meta["k"] = 3
        meta["distance"] = 2
        meta["data_coords"] = [(i, 0) for i in range(8)]
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)


# Convenience alias
Code832 = EightThreeTwoCode
