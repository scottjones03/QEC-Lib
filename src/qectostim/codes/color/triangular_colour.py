"""Triangular Colour Code (6.6.6 Tiling)

Pure geometric construction of the triangular colour code on a
honeycomb (6.6.6) lattice.  Every property—qubit coordinates,
stabiliser faces, logical operators, and 3-colouring—is derived
algorithmically from the single distance parameter *d*.

Construction
------------
The code lives on a triangular patch of the 6.6.6 tiling.  Data
qubits sit on vertices; stabiliser generators correspond to
3-colourable faces (hexagons in the bulk, weight-4 triangles on
the boundary).  Because the lattice is 3-colourable, the code is
natively compatible with the Chromobius decoder.

Geometry
--------
- Row 0 contains *d* qubits; subsequent rows come in pairs
  ``(d-1, d-1), (d-2, d-2), …`` down to ``(2, 1, 1)``.
- Odd-numbered rows are offset by 0.5 in the *x* direction,
  producing the characteristic honeycomb stagger.
- Vertical spacing is ``√3 / 2`` (equilateral triangles).
- Bulk faces are weight-6 hexagons; boundary faces are weight-4
  triangles or quadrilaterals.

Code parameters
---------------
- ``n = (3 d² + 1) / 4``  data qubits (for odd *d* ≥ 3).
- ``k = 1``  logical qubit.
- ``d`` is the minimum weight of any undetectable error.
- Self-dual CSS code: ``Hx == Hz``.

Transversal gates
-----------------
The triangular colour code supports transversal implementations of
the full Clifford group:

* **H** – transversal Hadamard (self-duality).
* **S** – transversal phase gate.
* **CNOT** – transversal between two code blocks.

This is a key advantage over the surface code, which requires
lattice surgery or magic-state distillation for the full Clifford
group.

Stabiliser scheduling
---------------------
Because ``Hx == Hz``, all stabilisers can be measured in a single
round (fully parallel schedule).  The measurement order follows
the canonical hexagonal sweep: six CNOT directions at angles
``0°, 60°, 120°, 180°, 240°, 300°``.

Decoding
--------
* **Chromobius** – exploits the 3-colourability to decompose the
  decoding problem into independent matching sub-problems, one per
  colour pair.
* **MWPM / PyMatching** – standard minimum-weight matching on the
  detector-error-model hypergraph.
* **BP-OSD** – belief-propagation + ordered-statistics decoding;
  useful for higher distances.

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]]` where:

- :math:`n = (3d^2 + 1) / 4` data qubits (for odd :math:`d \ge 3`)
- :math:`k = 1` logical qubit
- :math:`d` = code distance (minimum-weight undetectable error)
- Rate :math:`k/n = 4 / (3d^2 + 1)`

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **X-type stabilisers**: weight-6 hexagons in the bulk, weight-4
  triangles / quadrilaterals on the boundary.  Self-dual: :math:`H_X = H_Z`.
- **Z-type stabilisers**: identical support to X-type (self-dual).
- Measurement schedule: single fully-parallel round (self-dual
  :math:`H_X = H_Z` allows all stabilisers to be measured simultaneously);
  six CNOT directions at 60° intervals.

Connections
-----------
* Equivalent to a surface code under local unitary transformations.
* 2-D version of the 3-D colour code (which additionally supports
  a transversal T gate).
* Related to the 4.8.8 colour code via lattice duality.

References
----------
.. [Bombin06]  Bombin & Martin-Delgado, *Phys. Rev. Lett.* **97**,
   180501 (2006).  arXiv:quant-ph/0605138.
.. [Bombin07]  Bombin & Martin-Delgado, *J. Math. Phys.* **48**,
   052105 (2007).  arXiv:quant-ph/0605138.
.. [Chromobius]  Gidney, "Chromobius: a fast implementation of the
   Mobius decoder for colour codes", 2023.

Error budget
------------
* Under depolarising noise the triangular colour code achieves a
  circuit-level threshold of ~0.2 % (lower than surface codes but
  offset by the richer transversal gate set).
* For d = 5 the break-even physical error rate is approximately 2 × 10⁻³.
* The uniform weight-6 stabilisers simplify scheduling relative to the
  mixed-weight 4.8.8 colour code.
* Boundary stabilisers (weight 4) have a slightly higher effective
  error rate, creating a "boundary-dominated" error regime at low p.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import math

from qectostim.codes.abstract_css import (
    TopologicalCSSCode,
    Coord2D,
    MergeStabilizerInfo,
    SeamStabilizer,
    GrownStabilizer,
)
from qectostim.codes.complexes.css_complex import CSSChainComplex3
from qectostim.codes.utils import validate_css_code


class TriangularColourCode(TopologicalCSSCode):
    r"""
    Triangular colour code on 6.6.6 tiling with pure geometric construction.
    
    All code properties are derived from the distance d without external dependencies.

    d = 3 layout ([[7, 1, 3]])::

        Row sizes: [3, 2, 1, 1]  →  7 qubits

           0───1───2          row 0 (d=3 qubits)
            ╲ F0 ╱ ╲
             3───4            row 1 (offset 0.5)
              ╲ ╱
               5              row 2
               |
               6              row 3

        F0 = face {1,2,3,4}, centroid of those qubits
        Faces are weight 4 (boundary) or weight 6 (bulk hexagons).
        Self-dual: Hx = Hz.  Stabiliser coords = centroids.
    
    Parameters
    ----------
    distance : int
        Code distance (must be odd, >= 3).

    Raises
    ------
    ValueError
        If ``distance < 3`` or ``distance`` is even.
    """

    def __init__(self, distance: int = 3, metadata: Optional[Dict[str, Any]] = None):
        if distance < 3:
            raise ValueError(f"Distance must be >= 3, got {distance}")
        if distance % 2 == 0:
            raise ValueError(f"Distance must be odd, got {distance}")
        
        d = distance
        self._distance = d
        
        # Build geometry from first principles
        row_sizes = self._compute_row_sizes(d)
        n_qubits = sum(row_sizes)
        coords, grid = self._build_coordinates(row_sizes)
        faces, stab_colors = self._build_stabilizer_faces(row_sizes, grid)
        
        # Verify expected qubit count: n = (3d² + 1) / 4
        n_expected = (3 * d * d + 1) // 4
        assert n_qubits == n_expected, f"Qubit count {n_qubits} != expected {n_expected}"
        
        # Build parity check matrices (self-dual)
        n_stabs = len(faces)
        hx = np.zeros((n_stabs, n_qubits), dtype=np.uint8)
        for i, face in enumerate(faces):
            for q in face:
                hx[i, q] = 1
        hz = hx.copy()
        
        # Logical operators along top boundary (row 0)
        logical_x, logical_z = self._build_logical_operators(d, n_qubits, coords)
        
        # Stabilizer coordinates (face centers)
        stab_coords = []
        for face in faces:
            cx = sum(coords[q][0] for q in face) / len(face)
            cy = sum(coords[q][1] for q in face) / len(face)
            stab_coords.append((cx, cy))
        
        # Build chain complex
        boundary_2 = np.concatenate([hx.T, hz.T], axis=1).astype(np.uint8)
        boundary_1 = np.zeros((0, n_qubits), dtype=np.uint8)
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)
        
        # Validate CSS orthogonality before proceeding
        validate_css_code(hx, hz, f"TriangularColour_d{d}", raise_on_error=True)
        
        # Metadata
        data_coords = [coords[i] for i in range(n_qubits)]
        meta = dict(metadata or {})
        meta.update({
            "name": f"TriangularColour_d{d}",
            "n": n_qubits,
            "k": 1,
            "distance": d,
            "is_colour_code": True,
            "code_family": "color",
            "code_type": "triangular_colour_6_6_6",
            "rate": 1.0 / n_qubits,
            "tiling": "6.6.6",
            "data_coords": data_coords,
            "row_sizes": row_sizes,
            "lx_pauli_type": "X",
            "lz_pauli_type": "Z",
            "logical_x_support": [i for i, c in enumerate(logical_x[0]) if c == 'X'],
            "logical_z_support": [i for i, c in enumerate(logical_z[0]) if c == 'Z'],
            "lx_support": [i for i, c in enumerate(logical_x[0]) if c == 'X'],
            "lz_support": [i for i, c in enumerate(logical_z[0]) if c == 'Z'],
            "x_stab_coords": stab_coords,
            "z_stab_coords": stab_coords,
            "stab_colors": stab_colors,
            "is_chromobius_compatible": True,
            "faces": faces,
            "stabiliser_schedule": {
                "x_rounds": {i: 0 for i in range(n_stabs)},
                "z_rounds": {i: 0 for i in range(n_stabs)},
                "n_rounds": 1,
                "description": "Fully parallel: self-dual, Hx = Hz.",
            },
            "x_schedule": [(1.0, 0.0), (0.5, 0.866), (-0.5, 0.866), (-1.0, 0.0), (-0.5, -0.866), (0.5, -0.866)],
            "z_schedule": [(1.0, 0.0), (0.5, 0.866), (-0.5, 0.866), (-1.0, 0.0), (-0.5, -0.866), (0.5, -0.866)],
            "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/color",
            "wikipedia_url": "https://en.wikipedia.org/wiki/Color_code",
            "canonical_references": [
                "Bombin & Martin-Delgado, Phys. Rev. Lett. 97, 180501 (2006). arXiv:quant-ph/0605138",
                "Bombin & Martin-Delgado, J. Math. Phys. 48, 052105 (2007). arXiv:quant-ph/0605138",
            ],
            "connections": [
                "Equivalent to a surface code under local unitary transformations",
                "Transversal Clifford gates (H, S, CNOT) unlike surface code",
                "2D version of the 3D color code (which has transversal T)",
                "Chromobius decoder exploits 3-colourability",
            ],
        })
        
        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)
        
        self._hx = hx
        self._hz = hz
        self._coords = coords
        self._faces = faces
        self._stab_colors = stab_colors
    
    @staticmethod
    def _compute_row_sizes(d: int) -> List[int]:
        """
        Compute row sizes for distance d.
        
        Pattern discovered by analyzing Stim's color code geometry:
        - Start with [d, d-1]
        - For val from d-2 down to 3:
          - If (d - val) is even: add pair [val, val]
          - If (d - val) is odd: add single [val]
        - End with [2, 1, 1]
        
        Examples:
        - d=3: [3, 2, 1, 1] → 7 qubits
        - d=5: [5, 4, 3, 3, 2, 1, 1] → 19 qubits
        - d=7: [7, 6, 5, 5, 4, 3, 3, 2, 1, 1] → 37 qubits
        - d=9: [9, 8, 7, 7, 6, 5, 5, 4, 3, 3, 2, 1, 1] → 61 qubits
        """
        row_sizes = [d, d - 1]
        
        val = d - 2  # Start at d-2
        while val >= 3:
            offset = d - val
            if offset % 2 == 0:  # Even offset from d: pair
                row_sizes.extend([val, val])
            else:  # Odd offset from d: single
                row_sizes.append(val)
            val -= 1
        
        # End with 2, 1, 1
        if 2 not in row_sizes:
            row_sizes.append(2)
        row_sizes.extend([1, 1])
        return row_sizes
    
    @staticmethod
    def _build_coordinates(row_sizes: List[int]) -> Tuple[Dict[int, Tuple[float, float]], Dict[Tuple[int, int], int]]:
        """
        Build qubit coordinates on honeycomb lattice.
        
        - Even rows: x starts at 0
        - Odd rows: x starts at 0.5 (honeycomb offset)
        - y spacing: sqrt(3)/2
        
        Returns:
            coords: idx -> (x, y)
            grid: (row, col) -> idx
        """
        coords = {}
        grid = {}  # (row, col) -> idx
        y_spacing = math.sqrt(3) / 2
        idx = 0
        
        for row_idx, n_in_row in enumerate(row_sizes):
            x_offset = 0.5 if row_idx % 2 == 1 else 0.0
            y = row_idx * y_spacing
            
            for col in range(n_in_row):
                coords[idx] = (col + x_offset, y)
                grid[(row_idx, col)] = idx
                idx += 1
        
        return coords, grid
    
    @staticmethod
    def _build_stabilizer_faces(row_sizes: List[int], grid: Dict[Tuple[int, int], int]) -> Tuple[List[List[int]], List[int]]:
        """
        Build stabilizer faces for triangular color code.
        
        The triangular color code has a specific face structure:
        - Top triangles (weight 4): along the top edge
        - Interior hexagons (weight 6): in the bulk
        - Boundary triangles (weight 4): along left, right, and bottom edges
        
        The face pattern depends on the relative row sizes:
        - s0 > s1 >= s2: first row is biggest
        - s0 == s1 > s2: first two rows equal
        
        Returns: (faces, colors) where each face is a list of qubit indices
        """
        num_rows = len(row_sizes)
        d = row_sizes[0]
        
        def get_q(r: int, c: int) -> Optional[int]:
            """Get qubit index, or None if out of bounds."""
            return grid.get((r, c))
        
        def make_face(coords: List[Tuple[int, int]]) -> Optional[List[int]]:
            """Create face from coordinates if all qubits exist."""
            qubits = [get_q(r, c) for r, c in coords]
            if all(q is not None for q in qubits):
                return sorted(qubits)
            return None
        
        faces = []
        
        # === TOP TRIANGLES (rows 0-1) ===
        # Pattern: (0, 2i+1), (0, 2i+2), (1, 2i), (1, 2i+1)
        for i in range((d - 1) // 2):
            face = make_face([
                (0, 2*i + 1), (0, 2*i + 2),
                (1, 2*i), (1, 2*i + 1)
            ])
            if face:
                faces.append(face)
        
        # === PROCESS EACH LAYER r to r+2 ===
        for r in range(num_rows - 2):
            s0 = row_sizes[r]
            s1 = row_sizes[r + 1]
            s2 = row_sizes[r + 2]
            
            if r == 0:
                # LAYER 0-2: Left edge + hexagons
                # Left edge triangle
                face = make_face([(0, 0), (0, 1), (1, 0), (2, 0)])
                if face:
                    faces.append(face)
                
                # Hexagons: r0(c,c+1), r1(c-1,c), r2(c-1,c)
                for c in range(2, s1, 2):
                    face = make_face([
                        (0, c), (0, c+1),
                        (1, c-1), (1, c),
                        (2, c-1), (2, c)
                    ])
                    if face:
                        faces.append(face)
                        
            elif s0 > s1 and s1 == s2:
                # Band like 1-3: s0 > s1 == s2
                # Hexagons are column-aligned across all rows
                for c in range(0, s2 - 1, 2):
                    face = make_face([
                        (r, c), (r, c+1),
                        (r+1, c), (r+1, c+1),
                        (r+2, c), (r+2, c+1)
                    ])
                    if face:
                        faces.append(face)
                
                # Right edge triangle: (r, s0-2), (r, s0-1), (r+1, s1-1), (r+2, s2-1)
                if s0 > s2:
                    face = make_face([
                        (r, s0-2), (r, s0-1),
                        (r+1, s1-1),
                        (r+2, s2-1)
                    ])
                    if face:
                        faces.append(face)
                        
            elif s0 == s1 and s1 > s2:
                # Band like 2-4: s0 == s1 > s2
                # Hexagons: r0(c,c+1), r1(c,c+1), r2(c-1,c)
                for c in range(1, s2 + 1, 2):
                    face = make_face([
                        (r, c), (r, c+1),
                        (r+1, c), (r+1, c+1),
                        (r+2, c-1), (r+2, c)
                    ])
                    if face:
                        faces.append(face)
                        
            elif s0 > s1 and s1 > s2:
                # Band like 3-5: s0 > s1 > s2
                # Left edge triangle
                face = make_face([
                    (r, 0), (r, 1),
                    (r+1, 0),
                    (r+2, 0)
                ])
                if face:
                    faces.append(face)
                
                # Hexagons: shifted pattern
                for c in range(1, s2, 2):
                    face = make_face([
                        (r, c+1), (r, c+2),
                        (r+1, c), (r+1, c+1),
                        (r+2, c), (r+2, c+1)
                    ])
                    if face:
                        faces.append(face)
        
        # Assign colors using proper graph coloring
        # Two faces are adjacent if they share at least 2 qubits (an edge)
        # We need a valid 3-coloring where no adjacent faces have the same color
        colors = TriangularColourCode._compute_face_colors(faces)
        
        return faces, colors
    
    @staticmethod
    def _compute_face_colors(faces: List[List[int]]) -> List[int]:
        """
        Compute valid 3-coloring for faces using graph coloring.
        
        Two faces are adjacent if they share at least 2 qubits.
        Returns a list of colors (0=red, 1=green, 2=blue) such that
        no two adjacent faces have the same color.
        """
        import networkx as nx
        
        n_faces = len(faces)
        if n_faces == 0:
            return []
        
        # Build face adjacency graph
        G = nx.Graph()
        for i in range(n_faces):
            G.add_node(i)
        for i in range(n_faces):
            for j in range(i + 1, n_faces):
                shared = set(faces[i]) & set(faces[j])
                if len(shared) >= 2:  # Adjacent faces share an edge
                    G.add_edge(i, j)
        
        # Use connected_sequential strategy which guarantees minimal colors
        # for graphs that are k-colorable (color codes are always 3-colorable)
        coloring = nx.coloring.greedy_color(G, strategy='connected_sequential')
        colors = [coloring[i] for i in range(n_faces)]
        
        # Verify we got a valid 3-coloring
        num_colors = len(set(colors))
        if num_colors > 3:
            raise ValueError(
                f"Face graph requires {num_colors} colors, but color codes "
                f"must be 3-colorable. This suggests an error in face construction."
            )
        
        return colors
    
    @staticmethod
    def _build_logical_operators(d: int, n_qubits: int, coords: Dict[int, Tuple[float, float]]) -> Tuple[List[str], List[str]]:
        """Build weight-d logical operators along top boundary (row 0)."""
        # Row 0 qubits (y ≈ 0)
        boundary = [idx for idx, (x, y) in coords.items() if abs(y) < 0.1]
        boundary.sort(key=lambda q: coords[q][0])
        
        support = boundary[:d] if len(boundary) >= d else list(range(min(d, n_qubits)))
        
        lx = ['I'] * n_qubits
        lz = ['I'] * n_qubits
        for i in support:
            lx[i] = 'X'
            lz[i] = 'Z'
        
        return [''.join(lx)], [''.join(lz)]
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates."""
        return list(self.metadata.get("data_coords", []))
    
    @property
    def distance(self) -> int:
        """Return code distance."""
        return self._distance
    
    @property
    def name(self) -> str:
        """Return human-readable code name."""
        return self._metadata.get("name", f"TriangularColour_d{self._distance}")
    
    def get_detector_coords_4d(self, stab_idx: int, round_idx: int, is_x_type: bool) -> Tuple[float, float, float, int]:
        """
        Get 4D detector coordinates for Chromobius.
        
        Returns (x, y, t, color) where color ∈ {0,1,2} for X-type, {3,4,5} for Z-type.
        """
        stab_coords = self.metadata.get("x_stab_coords", [])
        x, y = stab_coords[stab_idx] if stab_idx < len(stab_coords) else (0.0, 0.0)
        base_color = self._stab_colors[stab_idx] if stab_idx < len(self._stab_colors) else 0
        color = base_color if is_x_type else base_color + 3
        return (x, y, float(round_idx), color)
    
    def validate_chromobius_coloring(self) -> bool:
        """Validate 3-coloring: adjacent faces must have different colors."""
        for i, face_i in enumerate(self._faces):
            set_i = set(face_i)
            for j, face_j in enumerate(self._faces):
                if i >= j:
                    continue
                if len(set_i & set(face_j)) == 2:  # Adjacent
                    if self._stab_colors[i] == self._stab_colors[j]:
                        raise ValueError(f"Adjacent faces {i},{j} have same color")
        return True

    # ------------------------------------------------------------------
    # Lattice-surgery merge-stabilizer computation
    # ------------------------------------------------------------------

    def get_merge_stabilizers(
        self,
        merge_type: str,
        my_edge: str,
        other_code: "TriangularColourCode",
        other_edge: str,
        my_data_global: Dict[int, int],
        other_data_global: Dict[int, int],
        seam_qubit_offset: int,
    ) -> MergeStabilizerInfo:
        """Compute seam and grown-boundary stabilizers for colour-code merge.

        Colour-code lattice surgery (Kesselring et al. 2018, Thomsen et al.
        2022) is more complex than surface-code surgery due to the
        3-colourable face structure.  The merge connects two triangular
        patches along a shared boundary, creating new face operators
        (seam stabilisers) that span both patches.

        Because the code is **self-dual** (``Hx = Hz``), every seam
        stabiliser acts as both an X-type and Z-type check.  The
        *merge_type* parameter therefore selects the Pauli basis for
        the ancilla preparation/measurement (``R/M`` for ZZ, ``RX/MX``
        for XX) rather than the stabiliser type.

        The merge boundary is the **top** edge (row 0) for ``my_edge="top"``
        or the **bottom** edge for ``my_edge="bottom"``.

        CX schedule: 6-phase hexagonal schedule matching the bulk EC.

        Parameters
        ----------
        merge_type : ``\"ZZ\"`` or ``\"XX\"``
        my_edge : ``\"bottom\" | \"top\"``  (triangular patches have 3 edges;
            left/right map to the two diagonal edges)
        other_code : colour code on the other side of the merge
        other_edge : edge of *other_code* facing the boundary
        my_data_global : ``{local_data_idx: global_qubit_idx}`` for this block
        other_data_global : ``{local_data_idx: global_qubit_idx}`` for other block
        seam_qubit_offset : starting global qubit index for seam ancillas
        """
        d = self._distance

        seam_stab_type = "Z" if merge_type == "ZZ" else "X"
        grown_stab_type = "X" if merge_type == "ZZ" else "Z"

        my_coords = list(self.metadata["data_coords"])
        my_coord_to_local = {(round(c[0], 4), round(c[1], 4)): i
                             for i, c in enumerate(my_coords)}

        other_coords = list(other_code.metadata["data_coords"])
        other_coord_to_local = {(round(c[0], 4), round(c[1], 4)): i
                                for i, c in enumerate(other_coords)}

        # Identify boundary qubits from each patch.
        # "top" boundary = row 0 (y ≈ 0) for both patches.
        # "bottom" boundary = last row (max y).
        my_y_vals = sorted(set(round(c[1], 4) for c in my_coords))
        oth_y_vals = sorted(set(round(c[1], 4) for c in other_coords))

        if my_edge == "top":
            my_bnd_y = my_y_vals[0]  # smallest y
        elif my_edge == "bottom":
            my_bnd_y = my_y_vals[-1]  # largest y
        else:
            # For left/right diagonal edges, pick the nearest y extremum
            my_bnd_y = my_y_vals[0] if my_edge == "left" else my_y_vals[-1]

        if other_edge == "top":
            oth_bnd_y = oth_y_vals[0]
        elif other_edge == "bottom":
            oth_bnd_y = oth_y_vals[-1]
        else:
            oth_bnd_y = oth_y_vals[0] if other_edge == "left" else oth_y_vals[-1]

        # Collect boundary qubits (sorted by x)
        my_bnd_qubits = sorted(
            [i for i, c in enumerate(my_coords)
             if abs(round(c[1], 4) - my_bnd_y) < 0.01],
            key=lambda i: my_coords[i][0]
        )
        oth_bnd_qubits = sorted(
            [i for i, c in enumerate(other_coords)
             if abs(round(c[1], 4) - oth_bnd_y) < 0.01],
            key=lambda i: other_coords[i][0]
        )

        seam_stabs: List[SeamStabilizer] = []
        grown_stabs: List[GrownStabilizer] = []
        seam_idx = 0

        # Create seam stabilisers: pairs of adjacent boundary qubits
        # from each patch form weight-4 face operators spanning both patches.
        n_pairs = min(len(my_bnd_qubits) - 1, len(oth_bnd_qubits) - 1)
        for p in range(n_pairs):
            g_anc = seam_qubit_offset + seam_idx
            seam_idx += 1

            my_q1_local = my_bnd_qubits[p]
            my_q2_local = my_bnd_qubits[p + 1]
            oth_q1_local = oth_bnd_qubits[p]
            oth_q2_local = oth_bnd_qubits[p + 1]

            my_q1 = my_data_global[my_q1_local]
            my_q2 = my_data_global[my_q2_local]
            oth_q1 = other_data_global[oth_q1_local]
            oth_q2 = other_data_global[oth_q2_local]

            support = [my_q1, my_q2, oth_q1, oth_q2]

            # 6-phase CX schedule (colour-code hexagonal).
            # Distribute 4 CX across first 4 of 6 phases.
            if seam_stab_type == "Z":
                cx_phases: List[List[Tuple[int, int]]] = [
                    [(my_q1, g_anc)],
                    [(my_q2, g_anc)],
                    [(oth_q1, g_anc)],
                    [(oth_q2, g_anc)],
                    [],
                    [],
                ]
            else:
                cx_phases = [
                    [(g_anc, my_q1)],
                    [(g_anc, my_q2)],
                    [(g_anc, oth_q1)],
                    [(g_anc, oth_q2)],
                    [],
                    [],
                ]

            seam_stabs.append(SeamStabilizer(
                lattice_position=(
                    (my_coords[my_q1_local][0] + other_coords[oth_q1_local][0]) / 2.0,
                    (my_bnd_y + oth_bnd_y) / 2.0,
                ),
                stab_type=seam_stab_type,
                global_ancilla_idx=g_anc,
                weight=4,
                cx_per_phase=cx_phases,
                support_globals=support,
            ))

        # Grown stabs: boundary faces that touch the merge boundary gain
        # new data-qubit support from the other patch.
        my_bnd_set = set(my_bnd_qubits)
        for face_idx, face in enumerate(self._faces):
            face_set = set(face)
            bnd_overlap = face_set & my_bnd_set
            if len(bnd_overlap) < 2:
                continue  # Face doesn't touch boundary edge

            # This face touches the boundary — it gains data from the
            # other patch.  Match boundary qubits by x-position.
            new_cx: List[List[Tuple[int, int]]] = [[] for _ in range(6)]
            new_support: List[int] = []

            for my_local in sorted(bnd_overlap):
                my_x = round(my_coords[my_local][0], 4)
                # Find matching other-patch qubit by x-coord
                for oth_local in oth_bnd_qubits:
                    oth_x = round(other_coords[oth_local][0], 4)
                    if abs(my_x - oth_x) < 0.01:
                        g_data = other_data_global[oth_local]
                        new_support.append(g_data)
                        phase = len(new_support) % 6
                        if grown_stab_type == "Z":
                            new_cx[phase].append((g_data, -1))
                        else:
                            new_cx[phase].append((-1, g_data))
                        break

            if new_support:
                stab_coord = self.metadata.get("x_stab_coords", [(0.0, 0.0)] * len(self._faces))
                sc = stab_coord[face_idx] if face_idx < len(stab_coord) else (0.0, 0.0)

                grown_stabs.append(GrownStabilizer(
                    lattice_position=(float(sc[0]), float(sc[1])),
                    stab_type=grown_stab_type,
                    existing_ancilla_global=-1,
                    original_weight=len(face),
                    new_weight=len(face) + len(new_support),
                    new_cx_per_phase=new_cx,
                    belongs_to_block="",
                    new_support_globals=new_support,
                ))

        return MergeStabilizerInfo(
            seam_stabs=seam_stabs,
            grown_stabs=grown_stabs,
            seam_type=seam_stab_type,
            grown_type=grown_stab_type,
            num_cx_phases=6,
        )


# Pre-built instances
TriangularColour3 = lambda: TriangularColourCode(distance=3)
TriangularColour5 = lambda: TriangularColourCode(distance=5)
TriangularColour7 = lambda: TriangularColourCode(distance=7)
