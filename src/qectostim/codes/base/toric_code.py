"""[[18,2,3]] Toric Code on 3×3 Torus

The toric code is a topological CSS code defined on a torus (periodic boundary conditions).
For a 3x3 torus:
- 18 qubits on the edges (9 horizontal + 9 vertical)
- 9 X-type (plaquette/face) stabilizers
- 9 Z-type (vertex/star) stabilizers
- 2 logical qubits corresponding to the two non-contractible cycles

The stabilizers are:
- X plaquettes: 4-body operators on edges around each face
- Z stars: 4-body operators on edges meeting at each vertex
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString

Coord2D = Tuple[float, float]


class ToricCode33(CSSCode):
    """
    [[18,2,3]] Toric code on a 3×3 torus.

    18 qubits (edges), 9 X-type plaquette checks, 9 Z-type vertex checks.
    Encodes 2 logical qubits with distance 3.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialize the 3x3 toric code.
        
        Qubit layout on 3x3 torus:
        - Horizontal edges: qubits 0-8 (h[row,col] at edge between vertices)
        - Vertical edges: qubits 9-17 (v[row,col] at edge between vertices)
        
        Vertex (i,j) connects to:
        - horizontal edge h[i,j] (right)
        - horizontal edge h[i,(j-1)%3] (left)
        - vertical edge v[i,j] (down)
        - vertical edge v[(i-1)%3,j] (up)
        
        Plaquette (i,j) (face) is bounded by:
        - horizontal edge h[i,j] (top)
        - horizontal edge h[(i+1)%3,j] (bottom)
        - vertical edge v[i,j] (left)
        - vertical edge v[i,(j+1)%3] (right)
        """
        L = 3  # Lattice size
        n_qubits = 2 * L * L  # 18 qubits
        
        # Edge indexing:
        # Horizontal edge at (row, col): index = row * L + col
        # Vertical edge at (row, col): index = L*L + row * L + col
        def h_edge(row, col):
            return (row % L) * L + (col % L)
        
        def v_edge(row, col):
            return L * L + (row % L) * L + (col % L)
        
        # Z-type stabilizers (vertex/star operators)
        # Each vertex (i,j) has a Z stabilizer on its 4 incident edges
        # Note: Only L*L-1 are independent (product of all = identity)
        hz_full = np.zeros((L * L, n_qubits), dtype=np.uint8)
        for i in range(L):
            for j in range(L):
                vertex_idx = i * L + j
                # Four edges incident to vertex (i,j)
                hz_full[vertex_idx, h_edge(i, j)] = 1        # right horizontal
                hz_full[vertex_idx, h_edge(i, (j - 1) % L)] = 1  # left horizontal
                hz_full[vertex_idx, v_edge(i, j)] = 1        # down vertical
                hz_full[vertex_idx, v_edge((i - 1) % L, j)] = 1  # up vertical
        
        # Remove last row (dependent on others since sum of all = 0)
        hz = hz_full[:-1]
        
        # X-type stabilizers (plaquette/face operators)
        # Each plaquette (i,j) has an X stabilizer on its 4 boundary edges
        # Note: Only L*L-1 are independent (product of all = identity)
        hx_full = np.zeros((L * L, n_qubits), dtype=np.uint8)
        for i in range(L):
            for j in range(L):
                plaq_idx = i * L + j
                # Four edges bounding plaquette (i,j)
                hx_full[plaq_idx, h_edge(i, j)] = 1          # top horizontal
                hx_full[plaq_idx, h_edge((i + 1) % L, j)] = 1  # bottom horizontal
                hx_full[plaq_idx, v_edge(i, j)] = 1          # left vertical
                hx_full[plaq_idx, v_edge(i, (j + 1) % L)] = 1  # right vertical
        
        # Remove last row (dependent on others since sum of all = 0)
        hx = hx_full[:-1]
        
        # Verify CSS orthogonality
        css_check = np.dot(hx, hz.T) % 2
        # Note: For toric code, each vertex-plaquette pair shares 0 or 2 edges,
        # so the overlap is always even.
        assert np.allclose(css_check, 0), f"CSS orthogonality violated"
        
        # Logical operators (2 logical qubits)
        # Logical X1: horizontal string across one cycle (all horizontal edges in row 0)
        lx1 = ['I'] * n_qubits
        for j in range(L):
            lx1[h_edge(0, j)] = 'X'
        
        # Logical Z1: vertical string across dual cycle (all vertical edges in column 0)
        lz1 = ['I'] * n_qubits
        for i in range(L):
            lz1[v_edge(i, 0)] = 'Z'
        
        # Logical X2: vertical string (all vertical edges in row 0)
        lx2 = ['I'] * n_qubits
        for j in range(L):
            lx2[v_edge(0, j)] = 'X'
        
        # Logical Z2: horizontal string on dual (all horizontal edges in column 0)
        lz2 = ['I'] * n_qubits
        for i in range(L):
            lz2[h_edge(i, 0)] = 'Z'
        
        logical_x = [''.join(lx1), ''.join(lx2)]
        logical_z = [''.join(lz1), ''.join(lz2)]

        meta = dict(metadata or {})
        meta["name"] = "ToricCode_3x3"
        meta["n"] = n_qubits
        meta["k"] = 2
        meta["distance"] = L
        meta["lattice_size"] = L
        
        # Logical supports
        meta["logical_x_support"] = [h_edge(0, j) for j in range(L)]  # First logical
        meta["logical_z_support"] = [v_edge(i, 0) for i in range(L)]  # First logical

        # Coordinates: place horizontal edges at (col+0.5, row), vertical at (col, row+0.5)
        coords = {}
        for i in range(L):
            for j in range(L):
                coords[h_edge(i, j)] = (j + 0.5, i)
                coords[v_edge(i, j)] = (j, i + 0.5)
        meta["data_coords"] = [coords.get(i, (0, 0)) for i in range(n_qubits)]

        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
