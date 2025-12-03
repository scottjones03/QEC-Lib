# src/qectostim/codes/topological/four_qubit_422.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..complexes.css_complex import CSSChainComplex3
from ..abstract_css import TopologicalCSSCode
from ..abstract_homological import Coord2D
from ..abstract_code import PauliString


class FourQubit422Code(TopologicalCSSCode):
    """The [[4,2,2]] Little Shor code.

    Stabilizers:
        S_X = XXXX
        S_Z = ZZZZ
    """

    def __init__(self, *, metadata: Optional[Dict[str, Any]] = None):
        data_coords: List[Coord2D] = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]

        # boundary_2 has shape (4, 1): 4 edges, 1 face per basis
        # This will give Hx = boundary_2.T = (1, 4)
        boundary_2 = np.array(
            [
                [1],
                [1],
                [1],
                [1],
            ],
            dtype=np.uint8,
        )

        # boundary_1 has shape (1, 4): 1 vertex, 4 edges
        # Hz = boundary_1 = (1, 4)
        boundary_1 = np.array(
            [
                [1, 1, 1, 1],
            ],
            dtype=np.uint8,
        )

        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        logical_z = [
            "ZZII",
            "IZZI",
        ]
        logical_x = [
            "XXII",
            "IXXI",
        ]
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update(
            {
                "distance": 2,
                "data_coords": data_coords,
                "x_stab_coords": [(0.4, 0.5)],
                "z_stab_coords": [(0.6, 0.5)],
                "data_qubits": [0, 1, 2, 3],
                "ancilla_qubits": [4, 5],
                "logical_x_support": [0, 1],
                "logical_z_support": [0, 1],
                "x_schedule": None,
                "z_schedule": None,
            }
        )

        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)

    def qubit_coords(self) -> List[Coord2D]:
        meta = getattr(self, "_metadata", {})
        return list(meta.get("data_coords", []))
