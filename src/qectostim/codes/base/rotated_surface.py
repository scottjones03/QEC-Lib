# src/qectostim/codes/topological/rotated_surface.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..abstract_css import TopologicalCSSCode, Coord2D
from ..abstract_code import PauliString
from ..complexes.css_complex import CSSChainComplex3


class RotatedSurfaceCode(TopologicalCSSCode):
    """Distance-d rotated surface code (planar patch, qubits on vertices).

    Geometry + schedule is chosen to match Stim's `gen_surface_code` rotated
    memory circuits as closely as possible.
    """

    def __init__(self, distance: int, *, metadata: Optional[Dict[str, Any]] = None):
        if distance < 2:
            raise ValueError("RotatedSurfaceCode distance must be >= 2")
        self._d = distance

        (
            data_coords,
            x_stab_coords,
            z_stab_coords,
            x_logical_coords,
            z_logical_coords,
        ) = self._build_lattice(distance)

        # index data coords
        data_coords_sorted = sorted(list(data_coords), key=lambda c: (c[1], c[0]))
        coord_to_index = {c: i for i, c in enumerate(data_coords_sorted)}

        # Build ∂2 as face→edge (stabiliser→data) incidence.
        # We treat edges as data-qubits; here we approximate by attaching each
        # stabiliser to up to 4 surrounding data qubits at ±(1,1).
        boundary_2_x = self._build_boundary2(x_stab_coords, coord_to_index)
        boundary_2_z = self._build_boundary2(z_stab_coords, coord_to_index)

        # Store explicit X/Z parity check matrices (rows = stabilisers, cols = data qubits).
        # These are used by CSSMemoryExperiment to align ancillas with stabiliser coords.
        hx = boundary_2_x.T.astype(np.uint8)
        hz = boundary_2_z.T.astype(np.uint8)

        # For a square surface code with qubits on vertices, C1 is “data” and
        # C2 is both X and Z faces. You can either build a single C2 with a
        # colour label per face, or keep two disconnected “layers”.
        # For simplicity we take:
        #   C2 = X_faces ⊕ Z_faces
        boundary_2 = np.concatenate([boundary_2_x, boundary_2_z], axis=1)

        # ∂1: edges→vertices. For now we use a simple “no-vertex” model where
        # H_Z is already represented by ∂1; to keep the chain-complex API happy
        # we can use a dummy boundary_1 with shape (#C0, #C1) = (0, n_data).
        # If you later want a strict homological model, you can fill in ∂1
        # from an explicit vertex set.
        boundary_1 = np.zeros((0, len(data_coords_sorted)), dtype=np.uint8)

        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        logical_x, logical_z, x_support, z_support = self._build_logicals(
            data_coords_sorted, coord_to_index, x_logical_coords, z_logical_coords
        )

        meta: Dict[str, Any] = dict(metadata or {})
        meta.update(
            {
                "distance": distance,
                "data_qubits": list(range(len(data_coords_sorted))),
                "data_coords": data_coords_sorted,
                "x_stab_coords": sorted(list(x_stab_coords)),
                "z_stab_coords": sorted(list(z_stab_coords)),
                "x_logical_coords": sorted(list(x_logical_coords)),
                "z_logical_coords": sorted(list(z_logical_coords)),
                "logical_x_support": x_support,
                "logical_z_support": z_support,
            }
        )

        # Stim-style 4-phase schedule for rotated surface code:
        # each stabiliser interacts with its neighbours in 4 rounds, going
        # roughly "clockwise" in the checkerboard.
        meta["x_schedule"] = [
            (1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
            (1.0, -1.0),
        ]
        meta["z_schedule"] = [
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (-1.0, 1.0),
        ]

        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)

        # Override CSSCode's parity-check matrices with the separated X/Z layers.
        # TopologicalCSSCode initialises Hx/Hz from the combined chain_complex; we want
        # experiments to see the physical X/Z stabiliser layers instead.
        self._hx = hx
        self._hz = hz

    # --- lattice -----------------------------------------------------------------

    @staticmethod
    def _build_lattice(
        d: int,
    ) -> Tuple[Set[Coord2D], Set[Coord2D], Set[Coord2D], Set[Coord2D], Set[Coord2D]]:
        data_coords: Set[Coord2D] = set()
        x_logical_coords: Set[Coord2D] = set()
        z_logical_coords: Set[Coord2D] = set()

        # Data qubits at odd-odd coordinates (x, y) both odd, within [1, 2d-1] range.
        # This matches Stim's rotated surface code: exactly d×d data qubits on vertices.
        for x in range(1, 2 * d, 2):
            for y in range(1, 2 * d, 2):
                q = (float(x), float(y))
                data_coords.add(q)
                # Track logical operator support
                if x == 1:
                    x_logical_coords.add(q)
                if y == 1:
                    z_logical_coords.add(q)

        x_stab_coords: Set[Coord2D] = set()
        z_stab_coords: Set[Coord2D] = set()

        # Stabilizers at even-even coordinates. For rotated surface code:
        # X-stabilizers (parity check for X-type errors) at specific even-even positions
        # Z-stabilizers (parity check for Z-type errors) at other even-even positions
        for x in range(0, 2 * d + 1, 2):
            for y in range(0, 2 * d + 1, 2):
                # Determine if this is an X or Z stabilizer based on parity of position
                parity = ((x // 2) % 2) != ((y // 2) % 2)
                
                # Skip boundary ancillas on appropriate edges
                on_left = x == 0
                on_right = x == 2 * d
                on_top = y == 0
                on_bottom = y == 2 * d
                
                # Boundary rules for rotated surface code
                if on_left and parity:
                    continue
                if on_right and parity:
                    continue
                if on_top and not parity:
                    continue
                if on_bottom and not parity:
                    continue
                
                q = (float(x), float(y))
                if parity:
                    x_stab_coords.add(q)
                else:
                    z_stab_coords.add(q)

        return (
            data_coords,
            x_stab_coords,
            z_stab_coords,
            x_logical_coords,
            z_logical_coords,
        )

    # --- incidence / chain complex ----------------------------------------------

    @staticmethod
    def _build_boundary2(
        stab_coords: Set[Coord2D],
        coord_to_index: Dict[Coord2D, int],
    ) -> np.ndarray:
        """∂2: faces->edges incidence using diagonal neighbours (±1,±1)."""
        n_edges = len(coord_to_index)
        deltas: List[Coord2D] = [
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        ]
        cols: List[List[int]] = []
        for s in sorted(stab_coords):
            col = [0] * n_edges
            sx, sy = s
            for dx, dy in deltas:
                nbr = (sx + dx, sy + dy)
                idx = coord_to_index.get(nbr)
                if idx is not None:
                    col[idx] ^= 1
            cols.append(col)
        if not cols:
            return np.zeros((n_edges, 0), dtype=np.uint8)
        return np.array(cols, dtype=np.uint8).T  # shape (#edges, #faces)

    # --- logicals ----------------------------------------------------------------

    def _build_logicals(
        self,
        data_coords: List[Coord2D],
        coord_to_index: Dict[Coord2D, int],
        x_logical_coords: Set[Coord2D],
        z_logical_coords: Set[Coord2D],
    ) -> Tuple[List[PauliString], List[PauliString], List[int], List[int]]:
        """Build logical X/Z operators and expose their supports."""

        n = len(data_coords)
        z_support = sorted(coord_to_index[c] for c in z_logical_coords if c in coord_to_index)
        x_support = sorted(coord_to_index[c] for c in x_logical_coords if c in coord_to_index)

        def pauli_string(support: List[int], pauli: str) -> str:
            ops = ["I"] * n
            for idx in support:
                ops[idx] = pauli
            return "".join(ops)

        logical_z = [pauli_string(z_support, "Z")] if z_support else []
        logical_x = [pauli_string(x_support, "X")] if x_support else []

        return logical_x, logical_z, x_support, z_support

    # --- convenience -------------------------------------------------------------

    @property
    def distance(self) -> int:
        return self._d

    def qubit_coords(self) -> List[Coord2D]:
        return list(self.metadata["data_coords"])

    @property
    def hx(self) -> np.ndarray:
        """X stabilisers: shape (#X-checks, #data)."""
        return self._hx

    @property
    def hz(self) -> np.ndarray:
        """Z stabilisers: shape (#Z-checks, #data)."""
        return self._hz
