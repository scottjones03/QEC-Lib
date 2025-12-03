# src/qectostim/codes/topological/colour_488.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..abstract_homological import CSSChainComplex3, TopologicalCSSCode, Coord2D
from ..abstract_code import PauliString


class ColourCode488(TopologicalCSSCode):
    """2D 4.8.8 colour code on a small patch.

    This is a topological CSS code: qubits on vertices of a 4.8.8 tiling, with
    faces coloured in 3 colours; each face supports both an X and a Z check.
    """

    def __init__(self, distance: int = 3, *, metadata: Optional[Dict[str, Any]] = None):
        # distance parameter here roughly corresponds to the linear size of the patch
        self._d = distance

        # TODO: build an actual 4.8.8 tiling for given distance.
        # For now we use a small hand-crafted patch with:
        #   - data qubits at integer lattice points in a hex-like region,
        #   - faces defined by adjacency lists.
        data_coords, face_to_vertices = self._build_tiling(distance)

        n_edges = len(data_coords)
        n_faces = len(face_to_vertices)

        boundary_2 = np.zeros((n_edges, n_faces), dtype=np.uint8)
        coord_to_index = {c: i for i, c in enumerate(data_coords)}

        for f_idx, verts in enumerate(face_to_vertices):
            for v in verts:
                idx = coord_to_index[v]
                boundary_2[idx, f_idx] ^= 1

        # Again, use dummy boundary_1 for now; can be upgraded later with
        # explicit vertex set.
        boundary_1 = np.zeros((0, n_edges), dtype=np.uint8)

        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        # TODO: build actual logical operators using PauliString.
        logical_x: List[PauliString] = []
        logical_z: List[PauliString] = []

        # In colour codes, each face supports both X and Z checks.
        # For scheduling, you'll typically use 3 rounds corresponding to the 3
        # colours, or a 4-round pattern as in some implementations.
        x_stab_coords = self._face_centres(face_to_vertices)
        z_stab_coords = x_stab_coords  # same positions; different basis

        meta: Dict[str, Any] = dict(metadata or {})
        meta.update(
            {
                "distance": distance,
                "data_coords": data_coords,
                "x_stab_coords": x_stab_coords,
                "z_stab_coords": z_stab_coords,
                # rough default: 3 rounds, one per colour class
                "x_schedule": [(1.0, 0.0), (0.0, 1.0), (-1.0, -1.0)],
                "z_schedule": [(1.0, 0.0), (0.0, 1.0), (-1.0, -1.0)],
            }
        )

        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)

    # --- tiling construction -----------------------------------------------------

    @staticmethod
    def _build_tiling(distance: int) -> Tuple[List[Coord2D], List[List[Coord2D]]]:
        """Return (data_coords, faces) for a small 4.8.8-like patch.

        This is deliberately a placeholder; you can replace with a true 4.8.8
        construction from the literature.
        """
        # Simple placeholder: a 3x3 grid, 4 square faces.
        data_coords: List[Coord2D] = [
            (x, y) for x in range(3) for y in range(3)
        ]
        faces: List[List[Coord2D]] = [
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            [(1, 0), (2, 0), (2, 1), (1, 1)],
            [(0, 1), (1, 1), (1, 2), (0, 2)],
            [(1, 1), (2, 1), (2, 2), (1, 2)],
        ]
        return data_coords, faces

    @staticmethod
    def _face_centres(faces: List[List[Coord2D]]) -> List[Coord2D]:
        centres: List[Coord2D] = []
        for verts in faces:
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            centres.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        return centres

    @property
    def distance(self) -> int:
        return self._d

    def qubit_coords(self) -> List[Coord2D]:
        return list(self.metadata["data_coords"])