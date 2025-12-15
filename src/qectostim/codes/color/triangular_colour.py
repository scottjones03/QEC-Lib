"""Triangular Colour Code (6.6.6 Tiling)

Pure geometric construction of the triangular colour code.
All properties are derived algorithmically from the distance parameter d.

The triangular colour code is a topological CSS code defined on a 
triangular lattice with 3-colourable faces (6.6.6 tiling).

Key properties:
- n = (3*d^2 + 1) / 4 data qubits (for odd d)
- k = 1 logical qubit
- Self-dual: X and Z stabilizers have identical support
- Transversal Clifford gates (H, S, CNOT)

Geometry:
- Row 0 has d qubits, then pairs (d-1, d-1), (d-2, d-2), ... down to (2, 2)
- Odd rows offset by 0.5 in x (honeycomb structure)
- Bulk faces are hexagons (weight 6), boundary faces are weight 4
- 3-colorable: colors assigned by (row + col) % 3

Reference: Bombin & Martin-Delgado, "Topological Quantum Distillation" (2006)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import math

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
from qectostim.codes.complexes.css_complex import CSSChainComplex3


class TriangularColourCode(TopologicalCSSCode):
    """
    Triangular colour code on 6.6.6 tiling with pure geometric construction.
    
    All code properties are derived from the distance d without external dependencies.
    
    Parameters
    ----------
    distance : int
        Code distance (must be odd, >= 3).
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
        
        # Metadata
        data_coords = [coords[i] for i in range(n_qubits)]
        meta = dict(metadata or {})
        meta.update({
            "name": f"TriangularColour_d{d}",
            "n": n_qubits,
            "k": 1,
            "distance": d,
            "is_colour_code": True,
            "tiling": "6.6.6",
            "data_coords": data_coords,
            "row_sizes": row_sizes,
            "logical_x_support": [i for i, c in enumerate(logical_x[0]) if c == 'X'],
            "logical_z_support": [i for i, c in enumerate(logical_z[0]) if c == 'Z'],
            "x_stab_coords": stab_coords,
            "z_stab_coords": stab_coords,
            "stab_colors": stab_colors,
            "is_chromobius_compatible": True,
            "faces": faces,
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


# Pre-built instances
TriangularColour3 = lambda: TriangularColourCode(distance=3)
TriangularColour5 = lambda: TriangularColourCode(distance=5)
TriangularColour7 = lambda: TriangularColourCode(distance=7)
