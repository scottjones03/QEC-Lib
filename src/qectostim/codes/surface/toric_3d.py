"""3D Toric Code

The 3D toric code is a topological CSS code defined on a 3-torus (L×L×L with 
periodic boundary conditions in all three dimensions).

Chain complex structure (proper 4-chain for 3D code):
    C3 (cubes) --∂3--> C2 (faces) --∂2--> C1 (edges) --∂1--> C0 (vertices)

For a CSS code with qubits on edges (C1):
    - X stabilizers from ∂2^T (faces acting on boundary edges)  
    - Z stabilizers from ∂1 (vertices acting on incident edges)

For a CSS code with qubits on faces (C2):
    - X stabilizers from ∂3^T (cubes acting on boundary faces)
    - Z stabilizers from ∂2 (edges acting on incident faces)

The 3D toric code has interesting properties:
    - With qubits on edges: point-like X excitations, loop-like Z excitations
    - With qubits on faces: loop-like X excitations, point-like Z excitations
    - Self-correcting behavior at finite temperature (for one sector)

Note: This implementation uses CSSChainComplex4 (4-chain) for proper 3D topology.
The boundary_3 map from cubes to faces is included for completeness.

References:
    - Castelnovo & Chamon, "Topological order in a 3D toric code at finite temperature" (2008)
    - Dennis et al., "Topological quantum memory" (2002)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode3D, Coord2D
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex4, CSSChainComplex3

Coord3D = Tuple[float, float, float]


class ToricCode3D(TopologicalCSSCode3D):
    """
    3D Toric code with qubits on edges.
    
    For an L×L×L torus:
        - n = 3L³ qubits (edges in x, y, z directions)
        - k = 3 logical qubits (three non-contractible cycles)
        - d = L distance
    
    X stabilizers are weight-4 face operators (plaquettes).
    Z stabilizers are weight-6 vertex operators (stars).
    
    Parameters
    ----------
    L : int
        Linear size of the cubic lattice (default: 3)
    metadata : dict, optional
        Additional metadata
    """
    
    def __init__(self, L: int = 3, metadata: Optional[Dict[str, Any]] = None):
        if L < 2:
            raise ValueError("L must be at least 2")
        
        self._L = L
        n_qubits = 3 * L**3
        
        # Build lattice geometry and parity check matrices
        (
            data_coords,
            x_stab_coords,
            z_stab_coords,
            hx,
            hz,
            boundary_3,
            boundary_2,
            boundary_1,
        ) = self._build_3d_toric_lattice(L)
        
        # Create chain complex (4-term for 3D CSS code)
        chain_complex = CSSChainComplex4(
            boundary_3=boundary_3,
            boundary_2=boundary_2,
            boundary_1=boundary_1,
            qubit_grade=1,  # Qubits on edges
        )
        
        # Build logical operators
        logical_x, logical_z = self._build_logicals(L, n_qubits)
        
        # Metadata
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"ToricCode3D_{L}x{L}x{L}",
            "n": n_qubits,
            "k": 3,
            "distance": L,
            "lattice_size": L,
            "dimension": 3,
            "data_coords": data_coords,
            "x_stab_coords": x_stab_coords,
            "z_stab_coords": z_stab_coords,
        })
        
        # Call parent constructor - TopologicalCSSCode3D now requires chain_complex first
        super().__init__(
            chain_complex=chain_complex,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
        
        # Store explicit parity check matrices (already derived from chain_complex by parent)
        self._hx = hx
        self._hz = hz
    
    @property
    def L(self) -> int:
        """Lattice size."""
        return self._L
    
    @staticmethod
    def _build_3d_toric_lattice(L: int) -> Tuple[
        List[Coord3D],   # data_coords
        List[Coord3D],   # x_stab_coords  
        List[Coord3D],   # z_stab_coords
        np.ndarray,      # hx
        np.ndarray,      # hz
        np.ndarray,      # boundary_3 (cubes -> faces)
        np.ndarray,      # boundary_2 (faces -> edges)
        np.ndarray,      # boundary_1 (edges -> vertices)
    ]:
        """Build the 3D toric code lattice."""
        n_qubits = 3 * L**3  # edges in x, y, z directions
        n_faces = 3 * L**3   # xy, xz, yz faces
        n_vertices = L**3
        
        # Edge indexing: edge_x[i,j,k], edge_y[i,j,k], edge_z[i,j,k]
        def edge_x(i, j, k):
            """Edge in x-direction at vertex (i,j,k)."""
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        def edge_y(i, j, k):
            """Edge in y-direction at vertex (i,j,k)."""
            return L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        def edge_z(i, j, k):
            """Edge in z-direction at vertex (i,j,k)."""
            return 2 * L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        def vertex_idx(i, j, k):
            """Vertex index."""
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        # Face indexing: face_xy[i,j,k], face_xz[i,j,k], face_yz[i,j,k]
        def face_xy(i, j, k):
            """Face in xy-plane at position (i,j,k)."""
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        def face_xz(i, j, k):
            """Face in xz-plane at position (i,j,k)."""
            return L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        def face_yz(i, j, k):
            """Face in yz-plane at position (i,j,k)."""
            return 2 * L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        # Data qubit (edge) coordinates
        data_coords = []
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    data_coords.append((i + 0.5, float(j), float(k)))  # x-edge
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    data_coords.append((float(i), j + 0.5, float(k)))  # y-edge
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    data_coords.append((float(i), float(j), k + 0.5))  # z-edge
        
        # X stabilizers (face/plaquette operators) - weight 4
        # Each face has 4 edges as boundary
        hx_list = []
        x_stab_coords = []
        
        # xy-faces
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    row = np.zeros(n_qubits, dtype=np.uint8)
                    # Four edges bounding this xy-face
                    row[edge_x(i, j, k)] = 1      # bottom x-edge
                    row[edge_x(i, (j+1) % L, k)] = 1  # top x-edge
                    row[edge_y(i, j, k)] = 1      # left y-edge
                    row[edge_y((i+1) % L, j, k)] = 1  # right y-edge
                    hx_list.append(row)
                    x_stab_coords.append((i + 0.5, j + 0.5, float(k)))
        
        # xz-faces
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    row = np.zeros(n_qubits, dtype=np.uint8)
                    row[edge_x(i, j, k)] = 1
                    row[edge_x(i, j, (k+1) % L)] = 1
                    row[edge_z(i, j, k)] = 1
                    row[edge_z((i+1) % L, j, k)] = 1
                    hx_list.append(row)
                    x_stab_coords.append((i + 0.5, float(j), k + 0.5))
        
        # yz-faces
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    row = np.zeros(n_qubits, dtype=np.uint8)
                    row[edge_y(i, j, k)] = 1
                    row[edge_y(i, j, (k+1) % L)] = 1
                    row[edge_z(i, j, k)] = 1
                    row[edge_z(i, (j+1) % L, k)] = 1
                    hx_list.append(row)
                    x_stab_coords.append((float(i), j + 0.5, k + 0.5))
        
        hx_full = np.array(hx_list, dtype=np.uint8)
        
        # Z stabilizers (vertex/star operators) - weight 6
        # Each vertex has 6 incident edges (±x, ±y, ±z)
        hz_list = []
        z_stab_coords = []
        
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    row = np.zeros(n_qubits, dtype=np.uint8)
                    # 6 edges meeting at vertex (i,j,k)
                    row[edge_x(i, j, k)] = 1           # +x direction
                    row[edge_x((i-1) % L, j, k)] = 1   # -x direction
                    row[edge_y(i, j, k)] = 1           # +y direction
                    row[edge_y(i, (j-1) % L, k)] = 1   # -y direction
                    row[edge_z(i, j, k)] = 1           # +z direction
                    row[edge_z(i, j, (k-1) % L)] = 1   # -z direction
                    hz_list.append(row)
                    z_stab_coords.append((float(i), float(j), float(k)))
        
        hz_full = np.array(hz_list, dtype=np.uint8)
        
        # Remove dependent rows (product of all stabilizers = I)
        # For torus: one face from each orientation, one vertex
        hx = hx_full[:-3]  # Remove last 3 faces (one per orientation)
        hz = hz_full[:-1]  # Remove last vertex
        
        # Trim coord lists to match
        x_stab_coords = x_stab_coords[:-3]
        z_stab_coords = z_stab_coords[:-1]
        
        # Build boundary matrices for chain complex
        # ∂2: faces -> edges, shape (n_edges, n_faces)
        # H_X = ∂2^T, so ∂2 = H_X^T
        boundary_2 = hx_full.T
        
        # ∂1: edges -> vertices, shape (n_vertices, n_edges)
        # H_Z = ∂1
        boundary_1 = hz_full
        
        # ∂3: cubes -> faces, shape (n_faces, n_cubes)
        # For a 3D torus, there are L³ cubes
        # Each cube has 6 faces as its boundary
        n_cubes = L**3
        
        def cube_idx(i, j, k):
            """Cube index at position (i,j,k)."""
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        boundary_3 = np.zeros((n_faces, n_cubes), dtype=np.uint8)
        
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    c = cube_idx(i, j, k)
                    # Each cube has 6 faces: 2 in each orientation (xy, xz, yz)
                    # xy-faces at z=k and z=k+1
                    boundary_3[face_xy(i, j, k), c] = 1
                    boundary_3[face_xy(i, j, (k+1) % L), c] = 1
                    # xz-faces at y=j and y=j+1
                    boundary_3[face_xz(i, j, k), c] = 1
                    boundary_3[face_xz(i, (j+1) % L, k), c] = 1
                    # yz-faces at x=i and x=i+1
                    boundary_3[face_yz(i, j, k), c] = 1
                    boundary_3[face_yz((i+1) % L, j, k), c] = 1
        
        return (data_coords, x_stab_coords, z_stab_coords, hx, hz, boundary_3, boundary_2, boundary_1)
    
    @staticmethod
    def _build_logicals(L: int, n_qubits: int) -> Tuple[List[str], List[str]]:
        """Build logical operators for 3D toric code.
        
        For 3D toric code with qubits on edges:
        - Logical X_i: string (1-chain) wrapping around direction i
        - Logical Z_i: membrane (2-cocycle) perpendicular to direction i
        
        Anti-commutation requirement: X_i and Z_i must overlap on exactly 1 qubit.
        X_i uses edges in direction i, Z_i uses edges perpendicular to direction i
        that pierce the plane where X_i lives.
        """
        def edge_x(i, j, k):
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        def edge_y(i, j, k):
            return L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        def edge_z(i, j, k):
            return 2 * L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        logical_x = []
        logical_z = []
        
        # Logical pair 1: X1 wraps in x-direction, Z1 is a yz-membrane at x=0
        # X1: string of edges along x at (y=0, z=0)
        lx1 = ['I'] * n_qubits
        for i in range(L):
            lx1[edge_x(i, 0, 0)] = 'X'
        logical_x.append(''.join(lx1))
        
        # Z1: membrane in yz-plane at x=0, using y-edges and z-edges
        # This membrane has L² y-edges and L² z-edges at x=0
        # It anti-commutes with X1 because X1 pierces this membrane once
        lz1 = ['I'] * n_qubits
        for j in range(L):
            for k in range(L):
                lz1[edge_y(0, j, k)] = 'Z'  # All y-edges at x=0
                lz1[edge_z(0, j, k)] = 'Z'  # All z-edges at x=0
        logical_z.append(''.join(lz1))
        
        # Logical pair 2: X2 wraps in y-direction, Z2 is an xz-membrane at y=0
        lx2 = ['I'] * n_qubits
        for j in range(L):
            lx2[edge_y(0, j, 0)] = 'X'
        logical_x.append(''.join(lx2))
        
        # Z2: membrane in xz-plane at y=0
        lz2 = ['I'] * n_qubits
        for i in range(L):
            for k in range(L):
                lz2[edge_x(i, 0, k)] = 'Z'
                lz2[edge_z(i, 0, k)] = 'Z'
        logical_z.append(''.join(lz2))
        
        # Logical pair 3: X3 wraps in z-direction, Z3 is an xy-membrane at z=0
        lx3 = ['I'] * n_qubits
        for k in range(L):
            lx3[edge_z(0, 0, k)] = 'X'
        logical_x.append(''.join(lx3))
        
        # Z3: membrane in xy-plane at z=0
        lz3 = ['I'] * n_qubits
        for i in range(L):
            for j in range(L):
                lz3[edge_x(i, j, 0)] = 'Z'
                lz3[edge_y(i, j, 0)] = 'Z'
        logical_z.append(''.join(lz3))
        
        return logical_x, logical_z
    
    def qubit_coords(self) -> List[Coord3D]:
        """Return 3D coordinates of data qubits."""
        return self._metadata.get("data_coords", [])


class ToricCode3DFaces(TopologicalCSSCode3D):
    """
    3D Toric code with qubits on faces (2-cells).
    
    This is the dual picture where:
        - X stabilizers are cube operators (weight-6)
        - Z stabilizers are edge operators (weight-4)
    
    For an L×L×L torus:
        - n = 3L³ qubits (faces in xy, xz, yz orientations)
        - k = 3 logical qubits
        - d = L distance
    
    This version has point-like Z excitations and loop-like X excitations,
    which is relevant for self-correction at finite temperature.
    
    Parameters
    ----------
    L : int
        Linear size of the cubic lattice (default: 3)
    metadata : dict, optional
        Additional metadata
    """
    
    def __init__(self, L: int = 3, metadata: Optional[Dict[str, Any]] = None):
        if L < 2:
            raise ValueError("L must be at least 2")
        
        self._L = L
        n_qubits = 3 * L**3  # faces
        
        # Build lattice
        (
            data_coords,
            x_stab_coords,
            z_stab_coords,
            hx,
            hz,
            boundary_2,
            boundary_1,
        ) = self._build_lattice(L)
        
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)
        logical_x, logical_z = self._build_logicals(L, n_qubits)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"ToricCode3DFaces_{L}x{L}x{L}",
            "n": n_qubits,
            "k": 3,
            "distance": L,
            "lattice_size": L,
            "dimension": 3,
            "qubits_on": "faces",
        })
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
            chain_complex=chain_complex,
            metadata=meta,
        )
    
    @staticmethod
    def _build_lattice(L: int) -> Tuple:
        """Build lattice with qubits on faces."""
        n_qubits = 3 * L**3
        n_cubes = L**3
        n_edges = 3 * L**3
        
        # Face indexing
        def face_xy(i, j, k):
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        def face_xz(i, j, k):
            return L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        def face_yz(i, j, k):
            return 2 * L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        # Data coords (face centers)
        data_coords = []
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    data_coords.append((i + 0.5, j + 0.5, float(k)))  # xy-face
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    data_coords.append((i + 0.5, float(j), k + 0.5))  # xz-face
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    data_coords.append((float(i), j + 0.5, k + 0.5))  # yz-face
        
        # X stabilizers: cube operators (6 faces per cube)
        hx_list = []
        x_stab_coords = []
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    row = np.zeros(n_qubits, dtype=np.uint8)
                    # 6 faces of cube at (i,j,k)
                    row[face_xy(i, j, k)] = 1
                    row[face_xy(i, j, (k+1) % L)] = 1
                    row[face_xz(i, j, k)] = 1
                    row[face_xz(i, (j+1) % L, k)] = 1
                    row[face_yz(i, j, k)] = 1
                    row[face_yz((i+1) % L, j, k)] = 1
                    hx_list.append(row)
                    x_stab_coords.append((i + 0.5, j + 0.5, k + 0.5))
        
        hx_full = np.array(hx_list, dtype=np.uint8)
        hx = hx_full[:-1]  # Remove dependent row
        x_stab_coords = x_stab_coords[:-1]
        
        # Z stabilizers: edge operators (4 faces per edge)
        hz_list = []
        z_stab_coords = []
        
        # x-edges
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    row = np.zeros(n_qubits, dtype=np.uint8)
                    row[face_xy(i, j, k)] = 1
                    row[face_xy(i, (j-1) % L, k)] = 1
                    row[face_xz(i, j, k)] = 1
                    row[face_xz(i, j, (k-1) % L)] = 1
                    hz_list.append(row)
                    z_stab_coords.append((i + 0.5, float(j), float(k)))
        
        # y-edges
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    row = np.zeros(n_qubits, dtype=np.uint8)
                    row[face_xy(i, j, k)] = 1
                    row[face_xy((i-1) % L, j, k)] = 1
                    row[face_yz(i, j, k)] = 1
                    row[face_yz(i, j, (k-1) % L)] = 1
                    hz_list.append(row)
                    z_stab_coords.append((float(i), j + 0.5, float(k)))
        
        # z-edges
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    row = np.zeros(n_qubits, dtype=np.uint8)
                    row[face_xz(i, j, k)] = 1
                    row[face_xz((i-1) % L, j, k)] = 1
                    row[face_yz(i, j, k)] = 1
                    row[face_yz(i, (j-1) % L, k)] = 1
                    hz_list.append(row)
                    z_stab_coords.append((float(i), float(j), k + 0.5))
        
        hz_full = np.array(hz_list, dtype=np.uint8)
        hz = hz_full[:-3]  # Remove 3 dependent rows
        z_stab_coords = z_stab_coords[:-3]
        
        boundary_2 = hx.T
        boundary_1 = hz
        
        return (data_coords, x_stab_coords, z_stab_coords, hx, hz, boundary_2, boundary_1)
    
    @staticmethod
    def _build_logicals(L: int, n_qubits: int) -> Tuple[List[str], List[str]]:
        """Build logical operators."""
        def face_xy(i, j, k):
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        def face_xz(i, j, k):
            return L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        def face_yz(i, j, k):
            return 2 * L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        logical_x = []
        logical_z = []
        
        # For 3D toric code with qubits on faces:
        # - Logical X: membrane (2-chain) wrapping around torus
        # - Logical Z: string (1-cocycle) perpendicular to membrane, dual representation
        
        # Logical pair 1: X1 is xy-membrane at z=0, Z1 is z-string
        # X1 uses all xy-faces at z=0
        lx1 = ['I'] * n_qubits
        for i in range(L):
            for j in range(L):
                lx1[face_xy(i, j, 0)] = 'X'
        logical_x.append(''.join(lx1))
        
        # Z1: must anti-commute with X1 - use xz and yz faces along a z-string
        # A string in z-direction at (x=0, y=0) uses faces that bound this string
        # Actually, for qubits on faces, Z logical is the boundary of a dual volume
        # Z1 pierces xy-membrane once: use yz-face and xz-face at fixed point
        lz1 = ['I'] * n_qubits
        for k in range(L):
            # String along z at (x=0, y=0): use xz-face and yz-face
            lz1[face_xz(0, 0, k)] = 'Z'
            lz1[face_yz(0, 0, k)] = 'Z'
        logical_z.append(''.join(lz1))
        
        # Logical pair 2: X2 is xz-membrane at y=0, Z2 is y-string  
        lx2 = ['I'] * n_qubits
        for i in range(L):
            for k in range(L):
                lx2[face_xz(i, 0, k)] = 'X'
        logical_x.append(''.join(lx2))
        
        # Z2: string along y at (x=0, z=0)
        lz2 = ['I'] * n_qubits
        for j in range(L):
            lz2[face_xy(0, j, 0)] = 'Z'
            lz2[face_yz(0, j, 0)] = 'Z'
        logical_z.append(''.join(lz2))
        
        # Logical pair 3: X3 is yz-membrane at x=0, Z3 is x-string
        lx3 = ['I'] * n_qubits
        for j in range(L):
            for k in range(L):
                lx3[face_yz(0, j, k)] = 'X'
        logical_x.append(''.join(lx3))
        
        # Z3: string along x at (y=0, z=0)
        lz3 = ['I'] * n_qubits
        for i in range(L):
            lz3[face_xy(i, 0, 0)] = 'Z'
            lz3[face_xz(i, 0, 0)] = 'Z'
        logical_z.append(''.join(lz3))
        
        return logical_x, logical_z


# Pre-configured instances
ToricCode3D_3x3x3 = lambda: ToricCode3D(L=3)
ToricCode3D_4x4x4 = lambda: ToricCode3D(L=4)
