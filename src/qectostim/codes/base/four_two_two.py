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
        coord_to_index = {c: i for i, c in enumerate(data_coords)}

        # One X face and one Z face, both touching all 4 edges.
        # boundary_2 has shape (#edges, #faces) = (4, 2).
        # Each row is an edge (data qubit); each column is a face (stabilizer).
        # Row i, col j = 1 if edge i is part of face j.
        boundary_2 = np.array(
            [
                [1, 1],  # edge 0 (qubit 0): part of X and Z faces
                [1, 1],  # edge 1 (qubit 1): part of X and Z faces
                [1, 1],  # edge 2 (qubit 2): part of X and Z faces
                [1, 1],  # edge 3 (qubit 3): part of X and Z faces
            ],
            dtype=np.uint8,
        )

        # Provide a vertex structure: 1 vertex, all edges connect to it.
        # boundary_1 has shape (#vertices, #edges) = (1, 4).
        boundary_1 = np.array(
            [
                [1, 1, 1, 1],  # vertex 0: connected to all 4 edges
            ],
            dtype=np.uint8,
        )

        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        logical_z = [
            "ZZII",  # logical Z on qubit 0
            "IZZI",  # logical Z on qubit 1
        ]
        logical_x = [
            "XXII",  # logical X on qubit 0
            "IXXI",  # logical X on qubit 1
        ]
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update(
            {
                "distance": 2,
                "data_coords": data_coords,
                "x_stab_coords": [(0.5, 0.5)],  # Single X stabilizer (XXXX)
                "z_stab_coords": [(0.5, 0.5)],  # Single Z stabilizer (ZZZZ)
                "data_qubits": [0, 1, 2, 3],
                "ancilla_qubits": [4, 5],
                "logical_x_support": [0, 1],  # XXII - only qubits 0,1
                "logical_z_support": [0, 1],  # ZZII - only qubits 0,1
                # Schedules for syndrome extraction
                "x_schedule": [(0.0, 0.0)],  # No geometric decomposition; single phase
                "z_schedule": [(0.0, 0.0)],  # No geometric decomposition; single phase
            }
        )

        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)

    def qubit_coords(self) -> List[Coord2D]:
        # Use the metadata dict stored by the base class
        meta = getattr(self, "_metadata", {})
        return list(meta.get("data_coords", []))
