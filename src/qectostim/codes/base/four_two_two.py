# src/qectostim/codes/topological/four_qubit_422.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..complexes.css_complex import CSSChainComplex3
from ..abstract_css import TopologicalCSSCode
from ..abstract_homological import Coord2D
from ..abstract_code import PauliString


class FourQubit422Code(TopologicalCSSCode):
    """The [[4,2,2]] 'Little Shor' code as a tiny topological patch.

    We use the standard stabilisers:
        S_X = XXXX
        S_Z = ZZZZ
    and place 4 data qubits at the corners of a unit square.
    """

    def __init__(self, *, metadata: Optional[Dict[str, Any]] = None):
        # Data qubits at corners (0,0), (1,0), (1,1), (0,1).
        data_coords: List[Coord2D] = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]

        # boundary_2 has shape (#edges, #faces) = (4, 1). This defines one X stabilizer (XXXX).
        boundary_2 = np.array(
            [[1], [1], [1], [1]],  # All 4 data qubits are in the X stabilizer
            dtype=np.uint8,
        )

        # boundary_1 has shape (#vertices, #edges) = (1, 4). This defines one Z stabilizer (ZZZZ).
        boundary_1 = np.array(
            [[1, 1, 1, 1]],  # All 4 data qubits are in the Z stabilizer
            dtype=np.uint8,
        )

        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        logical_z = [
            "ZIZI",  # logical Z1: anticommutes with XXII (overlap on qubit 0)
            "IZIZ",  # logical Z2: anticommutes with IXXI (overlap on qubit 1)
        ]
        logical_x = [
            "XXII",  # logical X1
            "IXXI",  # logical X2
        ]
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update(
            {
                "distance": 2,
                "data_coords": data_coords,
                "x_stab_coords": [(0.5, 1.0)],  # X stabilizer (XXXX) at top
                "z_stab_coords": [(0.5, 0.0)],  # Z stabilizer (ZZZZ) at bottom
                "data_qubits": list(range(4)),
                "ancilla_qubits": [4, 5],  # One for X (4), one for Z (5)
                "logical_x_support": [0, 1],  # XXII qubits 0,1
                "logical_z_support": [0, 2],  # ZIZI qubits 0,2
                # Schedules for syndrome extraction (naive, single step)
            }
        )

        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)

    def qubit_coords(self) -> List[Coord2D]:
        # Use the metadata dict stored by the base class
        meta = getattr(self, "_metadata", {})
        return list(meta.get("data_coords", []))
