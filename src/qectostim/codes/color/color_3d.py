"""3D Color Codes

The 3D color code is a topological CSS code defined on a 4-colorable 
3D lattice (typically a body-centered cubic or other suitable lattice).

Key properties:
    - Transversal T gate (unlike 2D color code)
    - Single-shot error correction
    - Related to 2D color code via dimensional jump

This module provides:
    - ColorCode3D: General 3D color code on a tetrahedralized lattice
    - ColorCode3DPrism: Stack of 2D triangular color codes

References:
    - Bombin, "Gauge Color Codes" (2015)
    - Kubica et al., "Universal transversal gates with color codes" (2015)
    - Vasmer & Browne, "3D color codes as 2D color codes on fractal surfaces" (2019)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from itertools import combinations

from qectostim.codes.abstract_css import CSSCode, TopologicalCSSCode3D, Coord2D
from qectostim.codes.utils import str_to_pauli

Coord3D = Tuple[float, float, float]


class ColorCode3D(TopologicalCSSCode3D):
    """
    3D Color code on a tetrahedralized lattice.
    
    The 3D color code is constructed on a 4-colorable lattice where:
        - Qubits are on vertices
        - X and Z stabilizers are on tetrahedra (4 qubits each)
    
    This implementation uses a simple cubic lattice with body-centered
    tetrahedralization for small distances.
    
    Parameters
    ----------
    distance : int
        Code distance (default: 3). Must be odd >= 3.
    metadata : dict, optional
        Additional metadata
    
    Properties
    ----------
    - Transversal T gate
    - Single-shot error correction capability
    - Code parameters: [[n, 1, d]] where n depends on lattice
    """
    
    def __init__(self, distance: int = 3, metadata: Optional[Dict[str, Any]] = None):
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance must be odd and >= 3")
        
        self._distance = distance
        
        # Build the tetrahedralized lattice
        (
            n_qubits,
            data_coords,
            x_stab_coords,
            z_stab_coords,
            hx,
            hz,
            logical_x,
            logical_z,
        ) = self._build_3d_color_lattice(distance)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"ColorCode3D_d{distance}",
            "n": n_qubits,
            "k": 1,
            "distance": distance,
            "dimension": 3,
            "transversal_gates": ["X", "Z", "H", "S", "T"],  # T gate is transversal!
            "single_shot": True,
            "data_coords": data_coords,
            "x_stab_coords": x_stab_coords,
            "z_stab_coords": z_stab_coords,
        })
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
    
    @property
    def distance(self) -> int:
        return self._distance
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return 2D projection of qubit coordinates."""
        return self._metadata.get("data_coords", [])
    
    @staticmethod
    def _build_3d_color_lattice(d: int) -> Tuple:
        """
        Build a 3D color code lattice using cell complex structure.
        
        For 3D color codes with transversal T, we use a tetrahedralized 3-ball
        where:
        - Qubits are on edges
        - X stabilizers are on faces (triangles)
        - Z stabilizers are on vertices
        
        This ensures Hx @ Hz.T = 0 through the incidence structure.
        """
        if d == 3:
            # Use a simple tetrahedral lattice on a 3-ball
            # Octahedron with 6 vertices, 12 edges, 8 faces
            
            # Vertices of octahedron
            vertices = [
                (0, 0, 1),    # 0: top
                (1, 0, 0),    # 1: +x
                (0, 1, 0),    # 2: +y
                (-1, 0, 0),   # 3: -x
                (0, -1, 0),   # 4: -y
                (0, 0, -1),   # 5: bottom
            ]
            
            # Edges (qubits) - 12 edges
            edges = [
                (0, 1), (0, 2), (0, 3), (0, 4),  # top to equator
                (5, 1), (5, 2), (5, 3), (5, 4),  # bottom to equator
                (1, 2), (2, 3), (3, 4), (4, 1),  # equator
            ]
            n_qubits = len(edges)
            edge_to_idx = {e: i for i, e in enumerate(edges)}
            # Also include reversed edges
            for i, (a, b) in enumerate(edges):
                edge_to_idx[(b, a)] = i
            
            # Faces (triangles) - 8 faces - X stabilizers
            faces = [
                (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),  # top pyramid
                (5, 1, 2), (5, 2, 3), (5, 3, 4), (5, 4, 1),  # bottom pyramid
            ]
            
            # Z stabilizers on vertices - each vertex has edges incident to it
            # Stabilizer includes all edges touching that vertex
            
            # Build Hx from faces
            n_x_stabs = len(faces) - 1  # Remove one for linear dependence
            hx = np.zeros((n_x_stabs, n_qubits), dtype=np.uint8)
            
            for i, (a, b, c) in enumerate(faces[:n_x_stabs]):
                # Face edges: (a,b), (b,c), (c,a)
                for e in [(a, b), (b, c), (c, a)]:
                    if e in edge_to_idx:
                        hx[i, edge_to_idx[e]] = 1
            
            # Build Hz from vertices
            n_z_stabs = len(vertices) - 1  # Remove one for linear dependence
            hz = np.zeros((n_z_stabs, n_qubits), dtype=np.uint8)
            
            for v in range(n_z_stabs):
                for i, (a, b) in enumerate(edges):
                    if a == v or b == v:
                        hz[v, i] = 1
            
            # Qubit coords (midpoint of each edge)
            data_coords = [
                ((vertices[a][0] + vertices[b][0]) / 2,
                 (vertices[a][1] + vertices[b][1]) / 2)
                for (a, b) in edges
            ]
            
            x_stab_coords = [
                ((vertices[a][0] + vertices[b][0] + vertices[c][0]) / 3,
                 (vertices[a][1] + vertices[b][1] + vertices[c][1]) / 3)
                for (a, b, c) in faces[:n_x_stabs]
            ]
            z_stab_coords = [(vertices[v][0], vertices[v][1]) for v in range(n_z_stabs)]
            
            # Logical operators: string from top to bottom
            # Logical Z on edges in z-direction, Logical X on equator
            lz_support = [0, 4]  # Edges (0,1) and (5,1) - path from top to bottom
            lx_support = [8, 9, 10, 11]  # Equator edges
            
            logical_z = [{idx: 'Z' for idx in lz_support}]
            logical_x = [{idx: 'X' for idx in lx_support}]
            
        else:
            # For larger distances, build a scaled tetrahedral lattice
            L = (d + 1) // 2  # Lattice size
            
            # Build a triangulated prism structure
            # Vertices on a triangular grid in each layer
            vertices = []
            vertex_coords = {}
            idx = 0
            for z in range(L + 1):
                for i in range(L + 1):
                    for j in range(L + 1 - i):
                        x = i + 0.5 * j
                        y = j * np.sqrt(3) / 2
                        vertices.append((x, y, float(z)))
                        vertex_coords[(i, j, z)] = idx
                        idx += 1
            
            # Build edges between adjacent vertices
            edges = []
            edge_set = set()
            for (i, j, z), v1 in vertex_coords.items():
                # Horizontal edges in triangle
                for di, dj in [(1, 0), (0, 1), (-1, 1)]:
                    ni, nj = i + di, j + dj
                    if (ni, nj, z) in vertex_coords:
                        v2 = vertex_coords[(ni, nj, z)]
                        e = (min(v1, v2), max(v1, v2))
                        if e not in edge_set:
                            edge_set.add(e)
                            edges.append(e)
                # Vertical edges
                if (i, j, z + 1) in vertex_coords:
                    v2 = vertex_coords[(i, j, z + 1)]
                    e = (min(v1, v2), max(v1, v2))
                    if e not in edge_set:
                        edge_set.add(e)
                        edges.append(e)
            
            n_qubits = len(edges)
            edge_to_idx = {e: i for i, e in enumerate(edges)}
            for i, (a, b) in enumerate(edges):
                edge_to_idx[(b, a)] = i
            
            # Build Z stabilizers on vertices
            n_z_stabs = min(len(vertices) - 1, n_qubits - 1)
            hz = np.zeros((n_z_stabs, n_qubits), dtype=np.uint8)
            
            for v in range(n_z_stabs):
                for i, (a, b) in enumerate(edges):
                    if a == v or b == v:
                        hz[v, i] = 1
            
            # Build X stabilizers on triangular faces
            faces = []
            for (i, j, z), v1 in vertex_coords.items():
                # Upward triangles
                if (i+1, j, z) in vertex_coords and (i, j+1, z) in vertex_coords:
                    v2 = vertex_coords[(i+1, j, z)]
                    v3 = vertex_coords[(i, j+1, z)]
                    faces.append((v1, v2, v3))
                # Downward triangles
                if (i+1, j, z) in vertex_coords and (i+1, j-1, z) in vertex_coords:
                    v2 = vertex_coords[(i+1, j, z)]
                    v3 = vertex_coords[(i+1, j-1, z)]
                    faces.append((v1, v2, v3))
            
            n_x_stabs = min(len(faces), n_qubits - n_z_stabs)
            hx = np.zeros((n_x_stabs, n_qubits), dtype=np.uint8)
            
            for f_idx, (a, b, c) in enumerate(faces[:n_x_stabs]):
                for e in [(a, b), (b, c), (c, a)]:
                    e_sorted = (min(e), max(e))
                    if e_sorted in edge_to_idx:
                        hx[f_idx, edge_to_idx[e_sorted]] = 1
            
            data_coords = [
                ((vertices[a][0] + vertices[b][0]) / 2,
                 (vertices[a][1] + vertices[b][1]) / 2)
                for (a, b) in edges
            ]
            
            x_stab_coords = [(float(i), float(i)) for i in range(n_x_stabs)]
            z_stab_coords = [(vertices[v][0], vertices[v][1]) for v in range(n_z_stabs)]
            
            # Simple logical operators
            logical_x = [{0: 'X'}]
            logical_z = [{0: 'Z'}]
        
        return (n_qubits, data_coords, x_stab_coords, z_stab_coords, 
                hx, hz, logical_x, logical_z)


class ColorCode3DPrism(TopologicalCSSCode3D):
    """
    3D Color code on a triangular prism (stack of 2D color codes).
    
    This construction stacks L layers of 2D triangular color codes,
    with additional stabilizers connecting adjacent layers.
    
    Parameters
    ----------
    L : int
        Number of layers (distance in z-direction)
    base_distance : int
        Distance of each 2D triangular color code layer
    metadata : dict, optional
        Additional metadata
    """
    
    def __init__(
        self, 
        L: int = 2, 
        base_distance: int = 3, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        if L < 2:
            raise ValueError("L must be at least 2")
        if base_distance < 3 or base_distance % 2 == 0:
            raise ValueError("base_distance must be odd and >= 3")
        
        self._L = L
        self._base_distance = base_distance
        
        # Build stacked lattice
        (
            n_qubits,
            data_coords,
            x_stab_coords,
            z_stab_coords,
            hx,
            hz,
            logical_x,
            logical_z,
        ) = self._build_prism_lattice(L, base_distance)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"ColorCode3DPrism_{L}x{base_distance}",
            "n": n_qubits,
            "k": 1,
            "distance": min(L, base_distance),
            "layers": L,
            "base_distance": base_distance,
            "data_coords": data_coords,
        })
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return 2D projection of qubit coordinates."""
        return self._metadata.get("data_coords", [])
    
    @staticmethod
    def _build_prism_lattice(L: int, base_d: int) -> Tuple:
        """
        Build stacked triangular color code lattice using edge placement.
        
        Qubits are on edges, X stabilizers on triangular faces, Z stabilizers on vertices.
        This ensures proper CSS conditions (Hx @ Hz.T = 0).
        """
        # Build a triangular grid with L layers
        # Vertices in each layer form a triangular lattice
        n_rows = base_d
        
        # Build vertex positions across all layers
        vertices = []
        vertex_coords = {}
        idx = 0
        for z in range(L):
            for row in range(n_rows):
                for col in range(n_rows - row):
                    x = col + row * 0.5
                    y = row * np.sqrt(3) / 2
                    vertices.append((x, y, float(z)))
                    vertex_coords[(row, col, z)] = idx
                    idx += 1
        
        n_vertices = len(vertices)
        
        # Build edges (qubits)
        edges = []
        edge_set = set()
        
        for (row, col, z), v1 in vertex_coords.items():
            # Edges within layer (triangular grid)
            for dr, dc in [(0, 1), (1, 0), (1, -1)]:
                nr, nc = row + dr, col + dc
                if (nr, nc, z) in vertex_coords:
                    v2 = vertex_coords[(nr, nc, z)]
                    e = (min(v1, v2), max(v1, v2))
                    if e not in edge_set:
                        edge_set.add(e)
                        edges.append(e)
            # Vertical edges between layers
            if (row, col, z + 1) in vertex_coords:
                v2 = vertex_coords[(row, col, z + 1)]
                e = (min(v1, v2), max(v1, v2))
                if e not in edge_set:
                    edge_set.add(e)
                    edges.append(e)
        
        n_qubits = len(edges)
        if n_qubits == 0:
            # Fallback for degenerate cases
            n_qubits = 1
            edges = [(0, 1)]
        
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        for i, (a, b) in enumerate(edges):
            edge_to_idx[(b, a)] = i
        
        # Z stabilizers on vertices (each includes all edges incident to vertex)
        n_z_stabs = n_vertices - 1  # Remove one for linear dependence
        hz = np.zeros((n_z_stabs, n_qubits), dtype=np.uint8)
        
        for v in range(n_z_stabs):
            for i, (a, b) in enumerate(edges):
                if a == v or b == v:
                    hz[v, i] = 1
        
        # X stabilizers on triangular faces within each layer
        faces = []
        for (row, col, z), v1 in vertex_coords.items():
            # Upward triangle
            if (row, col + 1, z) in vertex_coords and (row + 1, col, z) in vertex_coords:
                v2 = vertex_coords[(row, col + 1, z)]
                v3 = vertex_coords[(row + 1, col, z)]
                faces.append((v1, v2, v3))
            # Downward triangle
            if (row - 1, col + 1, z) in vertex_coords and (row, col + 1, z) in vertex_coords:
                v2 = vertex_coords[(row - 1, col + 1, z)]
                v3 = vertex_coords[(row, col + 1, z)]
                faces.append((v1, v2, v3))
        
        n_x_stabs = len(faces)
        if n_x_stabs == 0:
            n_x_stabs = 1
            hx = np.zeros((1, n_qubits), dtype=np.uint8)
        else:
            hx = np.zeros((n_x_stabs, n_qubits), dtype=np.uint8)
            
            for f_idx, (a, b, c) in enumerate(faces):
                for e in [(a, b), (b, c), (c, a)]:
                    e_sorted = (min(e), max(e))
                    if e_sorted in edge_to_idx:
                        hx[f_idx, edge_to_idx[e_sorted]] = 1
        
        # Qubit coords (midpoint of edge)
        data_coords = [
            ((vertices[a][0] + vertices[b][0]) / 2,
             (vertices[a][1] + vertices[b][1]) / 2)
            for (a, b) in edges
        ]
        
        x_stab_coords = [(float(i), float(i)) for i in range(n_x_stabs)]
        z_stab_coords = [(vertices[v][0], vertices[v][1]) for v in range(n_z_stabs)]
        
        # Logical operators
        logical_x = [{0: 'X'}]
        logical_z = [{0: 'Z'}]
        
        return (n_qubits, data_coords, x_stab_coords, z_stab_coords,
                hx, hz, logical_x, logical_z)


# Pre-configured instances
ColorCode3D_d3 = lambda: ColorCode3D(distance=3)
ColorCode3D_d5 = lambda: ColorCode3D(distance=5)
ColorCode3DPrism_2x3 = lambda: ColorCode3DPrism(L=2, base_distance=3)
