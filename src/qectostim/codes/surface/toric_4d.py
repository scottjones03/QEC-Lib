"""4D Toric Code

The 4D toric code is a topological CSS code defined on a 4-torus (L×L×L×L with 
periodic boundary conditions in all four dimensions).

Chain complex structure (proper 5-chain for 4D code):
    C4 (4-cells) --∂4--> C3 (3-cells) --∂3--> C2 (2-cells) --∂2--> C1 (1-cells) --∂1--> C0 (0-cells)

For the standard 4D toric code with qubits on 2-cells (faces):
    - X stabilizers from ∂3^T (3-cells acting on boundary faces)
    - Z stabilizers from ∂2 (edges acting on incident faces)

The 4D toric code has remarkable properties:
    - Self-correcting at finite temperature (in both X and Z sectors)
    - Both X and Z excitations are loop-like (1D)
    - Related to the 4D surface code via boundary conditions

Note: This implementation uses FiveCSSChainComplex (5-chain) for proper 4D topology.

References:
    - Dennis et al., "Topological quantum memory" (2002)
    - Alicki et al., "On thermal stability of topological qubit in Kitaev's 4D model" (2010)
    - Bombin et al., "Experiments with the 4D surface code" (2024)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from itertools import product

from qectostim.codes.abstract_css import TopologicalCSSCode4D, Coord2D
from qectostim.codes.complexes.css_complex import FiveCSSChainComplex

Coord4D = Tuple[float, float, float, float]


class ToricCode4D(TopologicalCSSCode4D):
    """
    4D Toric code with qubits on 2-cells (faces).
    
    For an L×L×L×L 4-torus:
        - n = 6L⁴ qubits (faces in xy, xz, xw, yz, yw, zw orientations)
        - k = 6 logical qubits (six independent 2-cycles)
        - d = L distance
    
    X stabilizers are weight-6 3-cell (cube) operators.
    Z stabilizers are weight-4 edge operators.
    
    Both X and Z excitations are loop-like, leading to self-correction
    at finite temperature.
    
    Parameters
    ----------
    L : int
        Linear size of the hypercubic lattice (default: 2)
    metadata : dict, optional
        Additional metadata
    """
    
    def __init__(self, L: int = 2, metadata: Optional[Dict[str, Any]] = None):
        if L < 2:
            raise ValueError("L must be at least 2")
        
        self._L = L
        n_qubits = 6 * L**4  # faces in 6 orientations
        
        # Build lattice
        (
            data_coords,
            x_stab_coords,
            z_stab_coords,
            hx,
            hz,
            sigma4,
            sigma3,
            sigma2,
            sigma1,
        ) = self._build_4d_toric_lattice(L)
        
        # Create 5-chain complex for proper 4D topology
        chain_complex = FiveCSSChainComplex(
            sigma4=sigma4,
            sigma3=sigma3,
            sigma2=sigma2,
            sigma1=sigma1,
            qubit_grade=2,  # Qubits on 2-cells (faces)
        )
        
        logical_x, logical_z = self._build_logicals(L, n_qubits)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"ToricCode4D_{L}",
            "n": n_qubits,
            "k": 6,
            "distance": L,
            "lattice_size": L,
            "dimension": 4,
            "qubits_on": "2-cells",
        })
        
        # Call parent constructor with chain_complex
        super().__init__(
            chain_complex=chain_complex,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
    
    @property
    def L(self) -> int:
        return self._L
    
    @staticmethod
    def _build_4d_toric_lattice(L: int) -> Tuple:
        """Build the 4D toric code lattice with qubits on 2-cells."""
        # 6 face orientations: xy, xz, xw, yz, yw, zw
        n_faces_per_orient = L**4
        n_qubits = 6 * n_faces_per_orient
        
        # Indexing for faces
        # Face orientations: 0=xy, 1=xz, 2=xw, 3=yz, 4=yw, 5=zw
        face_offsets = {
            'xy': 0, 'xz': 1, 'xw': 2, 'yz': 3, 'yw': 4, 'zw': 5
        }
        
        def face_idx(orient: str, i: int, j: int, k: int, l: int) -> int:
            """Get index of face with given orientation at position (i,j,k,l)."""
            base = face_offsets[orient] * n_faces_per_orient
            return base + ((((i % L) * L + (j % L)) * L + (k % L)) * L + (l % L))
        
        def idx4d(i, j, k, l):
            """Linear index for 4D position."""
            return ((((i % L) * L + (j % L)) * L + (k % L)) * L + (l % L))
        
        # Build coordinates (projected to 2D for visualization)
        data_coords = []
        for orient in ['xy', 'xz', 'xw', 'yz', 'yw', 'zw']:
            for coords in product(range(L), repeat=4):
                # Simple projection: use first two indices plus offset
                x_proj = coords[0] + 0.5 * (orient in ['xy', 'xz', 'xw'])
                y_proj = coords[1] + 0.5 * (orient in ['xy', 'yz', 'yw'])
                data_coords.append((x_proj, y_proj))
        
        # X stabilizers: 3-cell (cube) operators
        # Each 3-cell has 6 boundary 2-cells
        # 4 types of 3-cells: xyz, xyw, xzw, yzw (4 orientations)
        hx_list = []
        x_stab_coords = []
        
        # xyz-cubes (constant w)
        for i, j, k, l in product(range(L), repeat=4):
            row = np.zeros(n_qubits, dtype=np.uint8)
            # 6 faces of xyz-cube at (i,j,k,l)
            row[face_idx('xy', i, j, k, l)] = 1
            row[face_idx('xy', i, j, (k+1) % L, l)] = 1
            row[face_idx('xz', i, j, k, l)] = 1
            row[face_idx('xz', i, (j+1) % L, k, l)] = 1
            row[face_idx('yz', i, j, k, l)] = 1
            row[face_idx('yz', (i+1) % L, j, k, l)] = 1
            hx_list.append(row)
            x_stab_coords.append((i + 0.5, j + 0.5))
        
        # xyw-cubes (constant z)
        for i, j, k, l in product(range(L), repeat=4):
            row = np.zeros(n_qubits, dtype=np.uint8)
            row[face_idx('xy', i, j, k, l)] = 1
            row[face_idx('xy', i, j, k, (l+1) % L)] = 1
            row[face_idx('xw', i, j, k, l)] = 1
            row[face_idx('xw', i, (j+1) % L, k, l)] = 1
            row[face_idx('yw', i, j, k, l)] = 1
            row[face_idx('yw', (i+1) % L, j, k, l)] = 1
            hx_list.append(row)
            x_stab_coords.append((i + 0.5, j + 0.5))
        
        # xzw-cubes (constant y)
        for i, j, k, l in product(range(L), repeat=4):
            row = np.zeros(n_qubits, dtype=np.uint8)
            row[face_idx('xz', i, j, k, l)] = 1
            row[face_idx('xz', i, j, k, (l+1) % L)] = 1
            row[face_idx('xw', i, j, k, l)] = 1
            row[face_idx('xw', i, j, (k+1) % L, l)] = 1
            row[face_idx('zw', i, j, k, l)] = 1
            row[face_idx('zw', (i+1) % L, j, k, l)] = 1
            hx_list.append(row)
            x_stab_coords.append((i + 0.5, k + 0.5))
        
        # yzw-cubes (constant x)
        for i, j, k, l in product(range(L), repeat=4):
            row = np.zeros(n_qubits, dtype=np.uint8)
            row[face_idx('yz', i, j, k, l)] = 1
            row[face_idx('yz', i, j, k, (l+1) % L)] = 1
            row[face_idx('yw', i, j, k, l)] = 1
            row[face_idx('yw', i, j, (k+1) % L, l)] = 1
            row[face_idx('zw', i, j, k, l)] = 1
            row[face_idx('zw', i, (j+1) % L, k, l)] = 1
            hx_list.append(row)
            x_stab_coords.append((j + 0.5, k + 0.5))
        
        hx_full = np.array(hx_list, dtype=np.uint8)
        
        # Z stabilizers: edge operators
        # Each edge has 4 incident faces
        # 4 edge directions: x, y, z, w
        hz_list = []
        z_stab_coords = []
        
        # x-edges: incident faces are xy, xz, xw (at two positions each)
        for i, j, k, l in product(range(L), repeat=4):
            row = np.zeros(n_qubits, dtype=np.uint8)
            row[face_idx('xy', i, j, k, l)] = 1
            row[face_idx('xy', i, (j-1) % L, k, l)] = 1
            row[face_idx('xz', i, j, k, l)] = 1
            row[face_idx('xz', i, j, (k-1) % L, l)] = 1
            row[face_idx('xw', i, j, k, l)] = 1
            row[face_idx('xw', i, j, k, (l-1) % L)] = 1
            hz_list.append(row)
            z_stab_coords.append((i + 0.5, float(j)))
        
        # y-edges
        for i, j, k, l in product(range(L), repeat=4):
            row = np.zeros(n_qubits, dtype=np.uint8)
            row[face_idx('xy', i, j, k, l)] = 1
            row[face_idx('xy', (i-1) % L, j, k, l)] = 1
            row[face_idx('yz', i, j, k, l)] = 1
            row[face_idx('yz', i, j, (k-1) % L, l)] = 1
            row[face_idx('yw', i, j, k, l)] = 1
            row[face_idx('yw', i, j, k, (l-1) % L)] = 1
            hz_list.append(row)
            z_stab_coords.append((float(i), j + 0.5))
        
        # z-edges
        for i, j, k, l in product(range(L), repeat=4):
            row = np.zeros(n_qubits, dtype=np.uint8)
            row[face_idx('xz', i, j, k, l)] = 1
            row[face_idx('xz', (i-1) % L, j, k, l)] = 1
            row[face_idx('yz', i, j, k, l)] = 1
            row[face_idx('yz', i, (j-1) % L, k, l)] = 1
            row[face_idx('zw', i, j, k, l)] = 1
            row[face_idx('zw', i, j, k, (l-1) % L)] = 1
            hz_list.append(row)
            z_stab_coords.append((float(i), k + 0.5))
        
        # w-edges
        for i, j, k, l in product(range(L), repeat=4):
            row = np.zeros(n_qubits, dtype=np.uint8)
            row[face_idx('xw', i, j, k, l)] = 1
            row[face_idx('xw', (i-1) % L, j, k, l)] = 1
            row[face_idx('yw', i, j, k, l)] = 1
            row[face_idx('yw', i, (j-1) % L, k, l)] = 1
            row[face_idx('zw', i, j, k, l)] = 1
            row[face_idx('zw', i, j, (k-1) % L, l)] = 1
            hz_list.append(row)
            z_stab_coords.append((float(i), l + 0.5))
        
        hz_full = np.array(hz_list, dtype=np.uint8)
        
        # Remove dependent rows
        # Number of independent X stabilizers: 4L^4 - 1
        # Number of independent Z stabilizers: 4L^4 - 4
        n_x_remove = 1
        n_z_remove = 4
        hx = hx_full[:-n_x_remove] if n_x_remove > 0 else hx_full
        hz = hz_full[:-n_z_remove] if n_z_remove > 0 else hz_full
        x_stab_coords = x_stab_coords[:-n_x_remove] if n_x_remove > 0 else x_stab_coords
        z_stab_coords = z_stab_coords[:-n_z_remove] if n_z_remove > 0 else z_stab_coords
        
        # Build full chain complex boundary maps
        # sigma_k: C_k -> C_{k-1}
        # For qubits on 2-cells: Hx = sigma3^T, Hz = sigma2
        
        # sigma1: edges -> vertices (n_vertices × n_edges)
        # sigma2: faces -> edges (n_edges × n_faces) 
        # sigma3: cubes -> faces (n_faces × n_cubes), and Hx = sigma3^T
        # sigma4: 4-cells -> cubes (n_cubes × n_4cells)
        
        n_faces = n_qubits
        n_edges = 4 * L**4  # 4 edge directions
        n_cubes = 4 * L**4  # 4 cube orientations (xyz, xyw, xzw, yzw)
        n_4cells = L**4     # Only 1 type of 4-cell orientation
        n_vertices = L**4
        
        # sigma3: from Hx (3-cells to 2-cells)
        sigma3 = hx_full.T  # shape (n_faces, n_cubes)
        
        # sigma2: from Hz (2-cells to 1-cells)  
        sigma2 = hz_full.T  # shape (n_edges, n_faces)
        
        # sigma1: edges -> vertices (build from edge structure)
        # Each edge connects two vertices
        sigma1 = np.zeros((n_vertices, n_edges), dtype=np.uint8)
        
        def idx4d(i, j, k, l):
            return ((((i % L) * L + (j % L)) * L + (k % L)) * L + (l % L))
        
        edge_offsets = {'x': 0, 'y': 1, 'z': 2, 'w': 3}
        
        def edge_idx(direction: str, i: int, j: int, k: int, l: int) -> int:
            base = edge_offsets[direction] * n_vertices
            return base + idx4d(i, j, k, l)
        
        for i, j, k, ll in product(range(L), repeat=4):
            v = idx4d(i, j, k, ll)
            # x-edge at (i,j,k,l) connects vertex (i,j,k,l) to vertex (i+1,j,k,l)
            sigma1[v, edge_idx('x', i, j, k, ll)] = 1
            sigma1[idx4d((i+1) % L, j, k, ll), edge_idx('x', i, j, k, ll)] = 1
            # y-edge
            sigma1[v, edge_idx('y', i, j, k, ll)] = 1
            sigma1[idx4d(i, (j+1) % L, k, ll), edge_idx('y', i, j, k, ll)] = 1
            # z-edge
            sigma1[v, edge_idx('z', i, j, k, ll)] = 1
            sigma1[idx4d(i, j, (k+1) % L, ll), edge_idx('z', i, j, k, ll)] = 1
            # w-edge
            sigma1[v, edge_idx('w', i, j, k, ll)] = 1
            sigma1[idx4d(i, j, k, (ll+1) % L), edge_idx('w', i, j, k, ll)] = 1
        
        # sigma4: 4-cells -> cubes
        # Each 4-cell has 8 boundary cubes (like a tesseract)
        sigma4 = np.zeros((n_cubes, n_4cells), dtype=np.uint8)
        
        cube_offsets = {'xyz': 0, 'xyw': 1, 'xzw': 2, 'yzw': 3}
        
        def cube_idx(orient: str, i: int, j: int, k: int, l: int) -> int:
            base = cube_offsets[orient] * n_vertices
            return base + idx4d(i, j, k, l)
        
        for i, j, k, ll in product(range(L), repeat=4):
            cell4 = idx4d(i, j, k, ll)
            # 8 cubes bounding the 4-cell
            # xyz at w=l and w=l+1
            sigma4[cube_idx('xyz', i, j, k, ll), cell4] = 1
            sigma4[cube_idx('xyz', i, j, k, (ll+1) % L), cell4] = 1
            # xyw at z=k and z=k+1
            sigma4[cube_idx('xyw', i, j, k, ll), cell4] = 1
            sigma4[cube_idx('xyw', i, j, (k+1) % L, ll), cell4] = 1
            # xzw at y=j and y=j+1
            sigma4[cube_idx('xzw', i, j, k, ll), cell4] = 1
            sigma4[cube_idx('xzw', i, (j+1) % L, k, ll), cell4] = 1
            # yzw at x=i and x=i+1
            sigma4[cube_idx('yzw', i, j, k, ll), cell4] = 1
            sigma4[cube_idx('yzw', (i+1) % L, j, k, ll), cell4] = 1
        
        return (data_coords, x_stab_coords, z_stab_coords, hx, hz, sigma4, sigma3, sigma2, sigma1)
    
    @staticmethod
    def _build_logicals(L: int, n_qubits: int) -> Tuple[List[str], List[str]]:
        """Build logical operators for 4D toric code."""
        n_faces_per_orient = L**4
        face_offsets = {'xy': 0, 'xz': 1, 'xw': 2, 'yz': 3, 'yw': 4, 'zw': 5}
        
        def face_idx(orient: str, i: int, j: int, k: int, l: int) -> int:
            base = face_offsets[orient] * n_faces_per_orient
            return base + ((((i % L) * L + (j % L)) * L + (k % L)) * L + (l % L))
        
        logical_x = []
        logical_z = []
        
        # 6 pairs of logical operators corresponding to 6 independent 2-cycles
        # Logical X: membrane operators (L² faces)
        # Logical Z: membrane operators in dual direction
        
        # X1 on xy-plane (at z=0, w=0), Z1 on zw-plane
        lx1 = ['I'] * n_qubits
        for i in range(L):
            for j in range(L):
                lx1[face_idx('xy', i, j, 0, 0)] = 'X'
        logical_x.append(''.join(lx1))
        
        lz1 = ['I'] * n_qubits
        for k in range(L):
            for l in range(L):
                lz1[face_idx('zw', 0, 0, k, l)] = 'Z'
        logical_z.append(''.join(lz1))
        
        # X2 on xz-plane, Z2 on yw-plane
        lx2 = ['I'] * n_qubits
        for i in range(L):
            for k in range(L):
                lx2[face_idx('xz', i, 0, k, 0)] = 'X'
        logical_x.append(''.join(lx2))
        
        lz2 = ['I'] * n_qubits
        for j in range(L):
            for l in range(L):
                lz2[face_idx('yw', 0, j, 0, l)] = 'Z'
        logical_z.append(''.join(lz2))
        
        # X3 on xw-plane, Z3 on yz-plane
        lx3 = ['I'] * n_qubits
        for i in range(L):
            for l in range(L):
                lx3[face_idx('xw', i, 0, 0, l)] = 'X'
        logical_x.append(''.join(lx3))
        
        lz3 = ['I'] * n_qubits
        for j in range(L):
            for k in range(L):
                lz3[face_idx('yz', 0, j, k, 0)] = 'Z'
        logical_z.append(''.join(lz3))
        
        # X4 on yz-plane, Z4 on xw-plane
        lx4 = ['I'] * n_qubits
        for j in range(L):
            for k in range(L):
                lx4[face_idx('yz', 0, j, k, 0)] = 'X'
        logical_x.append(''.join(lx4))
        
        lz4 = ['I'] * n_qubits
        for i in range(L):
            for l in range(L):
                lz4[face_idx('xw', i, 0, 0, l)] = 'Z'
        logical_z.append(''.join(lz4))
        
        # X5 on yw-plane, Z5 on xz-plane
        lx5 = ['I'] * n_qubits
        for j in range(L):
            for l in range(L):
                lx5[face_idx('yw', 0, j, 0, l)] = 'X'
        logical_x.append(''.join(lx5))
        
        lz5 = ['I'] * n_qubits
        for i in range(L):
            for k in range(L):
                lz5[face_idx('xz', i, 0, k, 0)] = 'Z'
        logical_z.append(''.join(lz5))
        
        # X6 on zw-plane, Z6 on xy-plane
        lx6 = ['I'] * n_qubits
        for k in range(L):
            for l in range(L):
                lx6[face_idx('zw', 0, 0, k, l)] = 'X'
        logical_x.append(''.join(lx6))
        
        lz6 = ['I'] * n_qubits
        for i in range(L):
            for j in range(L):
                lz6[face_idx('xy', i, j, 0, 0)] = 'Z'
        logical_z.append(''.join(lz6))
        
        return logical_x, logical_z


# Pre-configured instances
ToricCode4D_2 = lambda: ToricCode4D(L=2)
ToricCode4D_3 = lambda: ToricCode4D(L=3)
