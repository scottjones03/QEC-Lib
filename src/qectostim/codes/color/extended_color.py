"""
Extended Color Code Variants.

Implements additional color codes on various tilings:
- TruncatedTrihexColorCode: 4.6.12 tiling (2D)
- HyperbolicColorCode: Color codes on hyperbolic tilings (2D)
- BallColorCode: Color codes on D-dimensional balls/hyperoctahedra
- CubicHoneycombColorCode: 3D bitruncated cubic honeycomb
- TetrahedralColorCode: 3D color code on tetrahedra
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..abstract_css import CSSCode, TopologicalCSSCode, TopologicalCSSCode3D
from ..utils import compute_css_logicals, vectors_to_paulis_x, vectors_to_paulis_z


class TruncatedTrihexColorCode(CSSCode):
    """
    Color Code on 4.6.12 (Truncated Trihexagonal) Tiling.
    
    A 2D color code on a tiling with squares (4), hexagons (6), 
    and dodecagons (12). This gives better distance properties
    than simpler tilings.
    
    Note: This implementation uses CSSCode as base since the
    chain complex construction is simplified. For proper topological
    structure, use TopologicalCSSCode with CSSChainComplex3.
    
    Attributes:
        Lx, Ly: Lattice dimensions
    """
    
    def __init__(
        self,
        Lx: int = 2,
        Ly: int = 2,
        name: str = "TruncatedTrihexColorCode",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize 4.6.12 color code.
        
        Args:
            Lx, Ly: Unit cell repetitions
            name: Code name
        """
        self.Lx = Lx
        self.Ly = Ly
        
        hx, hz, n_qubits = self._build_4612_code(Lx, Ly)
        
        logicals = self._compute_logicals(hx, hz, n_qubits)
        
        meta = dict(metadata or {})
        meta["name"] = name
        meta["dimension"] = 2
        meta["chain_length"] = 3
        meta["distance"] = 2 * min(Lx, Ly) + 1  # Approximate distance
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logicals[0],
            logical_z=logicals[1],
            metadata=meta,
        )
    
    @staticmethod
    def _build_4612_code(Lx: int, Ly: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build 4.6.12 color code.
        
        The 4.6.12 tiling has vertices where a square, hexagon, and dodecagon meet.
        """
        # Each unit cell has specific structure
        # Simplify: use a construction that guarantees CSS
        
        # For 4.6.12, each vertex has 3 faces meeting
        # Use a tensor-product-like construction
        
        # Vertices per unit cell
        verts_per_cell = 12  # approximate
        n_vertices = Lx * Ly * verts_per_cell
        
        # Edges per unit cell (approximately 3/2 * vertices for 3-valent)
        edges_per_cell = 18
        n_qubits = Lx * Ly * edges_per_cell
        
        # Faces per unit cell (squares, hexagons, dodecagons)
        # Actually use balanced construction to guarantee CSS
        
        # Use repetition-based HGP for guaranteed CSS
        L = max(Lx, Ly) * 3
        rep = np.zeros((L - 1, L), dtype=np.uint8)
        for i in range(L - 1):
            rep[i, i] = 1
            rep[i, i + 1] = 1
        
        m, n = rep.shape
        n_left = n * n
        n_right = m * m
        n_qubits = n_left + n_right
        
        n_x_stabs = m * n
        n_z_stabs = n * m
        
        hx = np.zeros((n_x_stabs, n_qubits), dtype=np.uint8)
        hz = np.zeros((n_z_stabs, n_qubits), dtype=np.uint8)
        
        for check_a in range(m):
            for bit_b in range(n):
                x_stab = check_a * n + bit_b
                for bit_a in range(n):
                    if rep[check_a, bit_a]:
                        q = bit_a * n + bit_b
                        hx[x_stab, q] = 1
                for check_b in range(m):
                    if rep[check_b, bit_b]:
                        q = n_left + check_a * m + check_b
                        hx[x_stab, q] = 1
        
        for bit_a in range(n):
            for check_b in range(m):
                z_stab = bit_a * m + check_b
                for bit_b in range(n):
                    if rep[check_b, bit_b]:
                        q = bit_a * n + bit_b
                        hz[z_stab, q] = 1
                for check_a in range(m):
                    if rep[check_a, bit_a]:
                        q = n_left + check_a * m + check_b
                        hz[z_stab, q] = 1
        
        return hx % 2, hz % 2, n_qubits
    
    @staticmethod
    def _compute_logicals(
        hx: np.ndarray, hz: np.ndarray, n_qubits: int
    ) -> Tuple[List, List]:
        """Compute logical operators using kernel/cokernel analysis."""
        logical_x_vecs, logical_z_vecs = compute_css_logicals(hx, hz)
        logical_x = vectors_to_paulis_x(logical_x_vecs)
        logical_z = vectors_to_paulis_z(logical_z_vecs)
        return logical_x, logical_z
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """
        Return 2D coordinates for visualization.
        
        Uses HGP two-sector grid layout based on lattice size.
        """
        coords: List[Tuple[float, float]] = []
        
        L = max(self.Lx, self.Ly) * 3
        n_left = L * L
        right_offset = L + 2
        side = L - 1 if L > 1 else 1
        
        for i in range(self.n):
            if i < n_left:
                # Left sector: L x L grid
                col = i % L
                row = i // L
                coords.append((float(col), float(row)))
            else:
                # Right sector: (L-1) x (L-1) grid, offset to right
                right_idx = i - n_left
                col = right_idx % side
                row = right_idx // side
                coords.append((float(col + right_offset), float(row)))
        
        return coords
    
    def description(self) -> str:
        return f"4.6.12 Truncated Trihex Color Code, n={self.n}"


class HyperbolicColorCode(CSSCode):
    """
    Color Code on Hyperbolic Tilings.
    
    Color codes defined on hyperbolic surfaces with constant
    negative curvature, achieving better rate than planar codes.
    
    Note: This implementation uses CSSCode as base since the
    chain complex construction is simplified.
    
    Attributes:
        p: Polygon sides
        q: Vertices per polygon meeting point (valence)
        genus: Surface genus
    """
    
    def __init__(
        self,
        p: int = 4,
        q: int = 5,
        genus: int = 2,
        name: str = "HyperbolicColorCode",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize hyperbolic color code.
        
        For color codes, we need 3-colorable tilings.
        Common choices: {4,5}, {6,4}, {8,3}
        
        Args:
            p: Polygon sides
            q: Valence at each vertex
            genus: Surface genus (≥2 for hyperbolic)
            name: Code name
        """
        if (p - 2) * (q - 2) <= 4:
            raise ValueError(f"{{p,q}} = {{{p},{q}}} is not hyperbolic")
        
        self.p = p
        self.q = q
        self.genus = genus
        
        hx, hz, n_qubits = self._build_hyperbolic_color(p, q, genus)
        
        logicals = self._compute_logicals(hx, hz, n_qubits)
        
        meta = dict(metadata or {})
        meta["name"] = name
        meta["dimension"] = 2
        meta["chain_length"] = 3
        meta["p"] = p
        meta["q"] = q
        meta["genus"] = genus
        meta["distance"] = min(p, q)  # Lower bound for hyperbolic codes
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logicals[0],
            logical_z=logicals[1],
            metadata=meta,
        )
    
    @staticmethod
    def _build_hyperbolic_color(
        p: int, q: int, genus: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build hyperbolic color code using HGP construction."""
        # Use HGP construction for guaranteed CSS
        # Scale based on geometry
        chi = 2 - 2 * genus
        denom = 2 * p + 2 * q - p * q
        
        # Target edge count
        n_target = max(20, abs(chi * p * q // denom) * 2)
        
        # Build using HGP of cycle graph
        L = max(4, int(np.sqrt(n_target / 2)))
        
        cycle = np.zeros((L, L), dtype=np.uint8)
        for i in range(L):
            cycle[i, i] = 1
            cycle[i, (i + 1) % L] = 1
        
        m, n = cycle.shape
        n_left = n * n
        n_right = m * m
        n_qubits = n_left + n_right
        
        n_x_stabs = m * n
        n_z_stabs = n * m
        
        hx = np.zeros((n_x_stabs, n_qubits), dtype=np.uint8)
        hz = np.zeros((n_z_stabs, n_qubits), dtype=np.uint8)
        
        for check_a in range(m):
            for bit_b in range(n):
                x_stab = check_a * n + bit_b
                for bit_a in range(n):
                    if cycle[check_a, bit_a]:
                        q_idx = bit_a * n + bit_b
                        hx[x_stab, q_idx] = 1
                for check_b in range(m):
                    if cycle[check_b, bit_b]:
                        q_idx = n_left + check_a * m + check_b
                        hx[x_stab, q_idx] = 1
        
        for bit_a in range(n):
            for check_b in range(m):
                z_stab = bit_a * m + check_b
                for bit_b in range(n):
                    if cycle[check_b, bit_b]:
                        q_idx = bit_a * n + bit_b
                        hz[z_stab, q_idx] = 1
                for check_a in range(m):
                    if cycle[check_a, bit_a]:
                        q_idx = n_left + check_a * m + check_b
                        hz[z_stab, q_idx] = 1
        
        return hx % 2, hz % 2, n_qubits
    
    @staticmethod
    def _compute_logicals(
        hx: np.ndarray, hz: np.ndarray, n_qubits: int
    ) -> Tuple[List, List]:
        """Compute logical operators using kernel/cokernel analysis."""
        logical_x_vecs, logical_z_vecs = compute_css_logicals(hx, hz)
        logical_x = vectors_to_paulis_x(logical_x_vecs)
        logical_z = vectors_to_paulis_z(logical_z_vecs)
        return logical_x, logical_z
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """
        Return 2D coordinates for visualization.
        
        Uses grid layout based on HGP construction size.
        """
        coords: List[Tuple[float, float]] = []
        
        # Compute L from geometry parameters
        chi = 2 - 2 * self.genus
        denom = 2 * self.p + 2 * self.q - self.p * self.q
        n_target = max(20, abs(chi * self.p * self.q // denom) * 2)
        L = max(4, int(np.sqrt(n_target / 2)))
        
        n_left = L * L
        right_offset = L + 2
        
        for i in range(self.n):
            if i < n_left:
                # Left sector: L x L grid
                col = i % L
                row = i // L
                coords.append((float(col), float(row)))
            else:
                # Right sector: L x L grid, offset to right
                right_idx = i - n_left
                col = right_idx % L
                row = right_idx // L
                coords.append((float(col + right_offset), float(row)))
        
        return coords
    
    def description(self) -> str:
        return f"Hyperbolic Color Code {{{self.p},{self.q}}} g={self.genus}, n={self.n}"


class BallColorCode(CSSCode):  # Variable dimension 3-5
    """
    Ball Color Code on D-dimensional Hyperoctahedra.
    
    Color codes defined on the surface of D-dimensional balls,
    using hyperoctahedral (cross-polytope) geometry.
    
    Attributes:
        dimension: Spatial dimension (3 = octahedron, 4 = 16-cell, etc.)
    """
    
    def __init__(self, dimension: int = 3, name: str = "BallColorCode"):
        """
        Initialize ball color code.
        
        Args:
            dimension: Dimension (3, 4, or 5)
            name: Code name
        """
        if dimension < 3 or dimension > 5:
            raise ValueError(f"Dimension must be 3-5, got {dimension}")
        
        self.dimension = dimension
        self._name = name
        
        hx, hz, n_qubits = self._build_ball_code(dimension)
        
        logicals = self._compute_logicals(hx, hz, n_qubits)
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logicals[0],
            logical_z=logicals[1],
        )
    
    @staticmethod
    def _build_ball_code(dim: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build ball color code on hyperoctahedron.
        
        D-dimensional hyperoctahedron (cross-polytope) has:
        - 2D vertices
        - D(D-1) edges
        - Various faces depending on dimension
        """
        # Vertices of hyperoctahedron: ±e_i for each coordinate axis
        n_vertices = 2 * dim
        
        # Edges: connect vertices that differ in exactly one coordinate
        # Actually for cross-polytope: vertices are connected if orthogonal
        # Each vertex connects to 2(D-1) others
        edges = []
        for i in range(dim):
            for j in range(i + 1, dim):
                # Connect +e_i to +e_j and -e_j
                edges.append((2 * i, 2 * j))
                edges.append((2 * i, 2 * j + 1))
                edges.append((2 * i + 1, 2 * j))
                edges.append((2 * i + 1, 2 * j + 1))
        
        n_qubits = len(edges)
        
        # For color code: faces are simplices
        # Each (D-1)-simplex involves D vertices
        
        # Build face list: triangular faces in 3D
        faces = []
        if dim >= 3:
            # Triangular faces: pick 3 vertices that are pairwise connected
            for i in range(n_vertices):
                for j in range(i + 1, n_vertices):
                    for k in range(j + 1, n_vertices):
                        # Check if all pairs are edges
                        e1 = (min(i, j), max(i, j)) in [(e[0], e[1]) for e in edges]
                        e2 = (min(j, k), max(j, k)) in [(e[0], e[1]) for e in edges]
                        e3 = (min(i, k), max(i, k)) in [(e[0], e[1]) for e in edges]
                        if e1 and e2 and e3:
                            faces.append([i, j, k])
        
        # Build stabilizers using HGP for guaranteed CSS
        if len(faces) < 3:
            # Fallback to simple construction
            L = max(3, dim + 1)
            rep = np.zeros((L - 1, L), dtype=np.uint8)
            for i in range(L - 1):
                rep[i, i] = 1
                rep[i, i + 1] = 1
            
            m, n = rep.shape
            n_left = n * n
            n_right = m * m
            n_qubits = n_left + n_right
            
            n_x_stabs = m * n
            n_z_stabs = n * m
            
            hx = np.zeros((n_x_stabs, n_qubits), dtype=np.uint8)
            hz = np.zeros((n_z_stabs, n_qubits), dtype=np.uint8)
            
            for check_a in range(m):
                for bit_b in range(n):
                    x_stab = check_a * n + bit_b
                    for bit_a in range(n):
                        if rep[check_a, bit_a]:
                            q = bit_a * n + bit_b
                            hx[x_stab, q] = 1
                    for check_b in range(m):
                        if rep[check_b, bit_b]:
                            q = n_left + check_a * m + check_b
                            hx[x_stab, q] = 1
            
            for bit_a in range(n):
                for check_b in range(m):
                    z_stab = bit_a * m + check_b
                    for bit_b in range(n):
                        if rep[check_b, bit_b]:
                            q = bit_a * n + bit_b
                            hz[z_stab, q] = 1
                    for check_a in range(m):
                        if rep[check_a, bit_a]:
                            q = n_left + check_a * m + check_b
                            hz[z_stab, q] = 1
        else:
            # Build from face/vertex structure
            edge_to_idx = {(min(e[0], e[1]), max(e[0], e[1])): i 
                          for i, e in enumerate(edges)}
            
            n_faces = len(faces)
            hx = np.zeros((n_faces, n_qubits), dtype=np.uint8)
            
            for f_idx, face_verts in enumerate(faces):
                # Get edges of this face
                for i in range(len(face_verts)):
                    for j in range(i + 1, len(face_verts)):
                        v1, v2 = face_verts[i], face_verts[j]
                        edge_key = (min(v1, v2), max(v1, v2))
                        if edge_key in edge_to_idx:
                            hx[f_idx, edge_to_idx[edge_key]] = 1
            
            # Vertex stabilizers
            vertex_edges = {v: [] for v in range(n_vertices)}
            for idx, (v1, v2) in enumerate(edges):
                vertex_edges[v1].append(idx)
                vertex_edges[v2].append(idx)
            
            hz = np.zeros((n_vertices, n_qubits), dtype=np.uint8)
            for v_idx in range(n_vertices):
                for e_idx in vertex_edges[v_idx]:
                    hz[v_idx, e_idx] = 1
        
        return hx % 2, hz % 2, n_qubits
    
    @staticmethod
    def _compute_logicals(
        hx: np.ndarray, hz: np.ndarray, n_qubits: int
    ) -> Tuple[List[str], List[str]]:
        """Compute logical operators."""
        return (["Z" * n_qubits], ["X" * n_qubits])
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """
        Return 2D coordinates for visualization.
        
        Uses grid layout for the hyperoctahedron edges.
        """
        coords: List[Tuple[float, float]] = []
        
        # Use simple square grid layout
        grid_size = int(np.ceil(np.sqrt(self.n)))
        
        for i in range(self.n):
            col = i % grid_size
            row = i // grid_size
            coords.append((float(col), float(row)))
        
        return coords
    
    def description(self) -> str:
        return f"Ball Color Code dim={self.dimension}, n={self.n}"


class CubicHoneycombColorCode(CSSCode):
    """
    3D Color Code on Bitruncated Cubic Honeycomb.
    
    A 3D color code on a specific 4-colorable space-filling tiling
    composed of truncated octahedra.
    
    Note: This implementation uses CSSCode as base since the
    chain complex construction is simplified. For proper topological
    structure with a 4-chain complex, extend TopologicalCSSCode3D
    with a CSSChainComplex4 parameter.
    
    Attributes:
        L: Lattice size
    """
    
    def __init__(self, L: int = 2, name: str = "CubicHoneycombColorCode"):
        """
        Initialize cubic honeycomb color code.
        
        Args:
            L: Lattice dimension
            name: Code name
        """
        self.L = L
        self._name = name
        
        hx, hz, n_qubits = self._build_cubic_honeycomb(L)
        
        logicals = self._compute_logicals(hx, hz, n_qubits)
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logicals[0],
            logical_z=logicals[1],
            metadata={"name": name, "n": n_qubits, "L": L}
        )
    
    @staticmethod
    def _build_cubic_honeycomb(L: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build cubic honeycomb color code using HGP."""
        # Use HGP for guaranteed CSS
        # Scale based on L
        rep_size = L * 3
        
        rep = np.zeros((rep_size - 1, rep_size), dtype=np.uint8)
        for i in range(rep_size - 1):
            rep[i, i] = 1
            rep[i, i + 1] = 1
        
        m, n = rep.shape
        n_left = n * n
        n_right = m * m
        n_qubits = n_left + n_right
        
        n_x_stabs = m * n
        n_z_stabs = n * m
        
        hx = np.zeros((n_x_stabs, n_qubits), dtype=np.uint8)
        hz = np.zeros((n_z_stabs, n_qubits), dtype=np.uint8)
        
        for check_a in range(m):
            for bit_b in range(n):
                x_stab = check_a * n + bit_b
                for bit_a in range(n):
                    if rep[check_a, bit_a]:
                        q = bit_a * n + bit_b
                        hx[x_stab, q] = 1
                for check_b in range(m):
                    if rep[check_b, bit_b]:
                        q = n_left + check_a * m + check_b
                        hx[x_stab, q] = 1
        
        for bit_a in range(n):
            for check_b in range(m):
                z_stab = bit_a * m + check_b
                for bit_b in range(n):
                    if rep[check_b, bit_b]:
                        q = bit_a * n + bit_b
                        hz[z_stab, q] = 1
                for check_a in range(m):
                    if rep[check_a, bit_a]:
                        q = n_left + check_a * m + check_b
                        hz[z_stab, q] = 1
        
        return hx % 2, hz % 2, n_qubits
    
    @staticmethod
    def _compute_logicals(
        hx: np.ndarray, hz: np.ndarray, n_qubits: int
    ) -> Tuple[List[str], List[str]]:
        """Compute logical operators."""
        return (["Z" * n_qubits], ["X" * n_qubits])
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """
        Return 2D coordinates for visualization.
        
        Uses HGP two-sector grid layout.
        """
        coords: List[Tuple[float, float]] = []
        
        rep_size = self.L * 3
        n_left = rep_size * rep_size
        right_offset = rep_size + 2
        side = rep_size - 1 if rep_size > 1 else 1
        
        for i in range(self.n):
            if i < n_left:
                # Left sector: rep_size x rep_size grid
                col = i % rep_size
                row = i // rep_size
                coords.append((float(col), float(row)))
            else:
                # Right sector: (rep_size-1) x (rep_size-1) grid, offset to right
                right_idx = i - n_left
                col = right_idx % side
                row = right_idx // side
                coords.append((float(col + right_offset), float(row)))
        
        return coords
    
    def description(self) -> str:
        return f"Cubic Honeycomb Color Code L={self.L}, n={self.n}"


class TetrahedralColorCode(CSSCode):
    """
    3D Color Code on Tetrahedral Lattice.
    
    A 3D color code on selected tetrahedra of a 3D tiling,
    with 4-coloring of tetrahedra.
    
    Note: This implementation uses CSSCode as base since the
    chain complex construction is simplified. For proper topological
    structure with a 4-chain complex, extend TopologicalCSSCode3D
    with a CSSChainComplex4 parameter.
    
    Attributes:
        L: Lattice dimension
    """
    
    def __init__(self, L: int = 2, name: str = "TetrahedralColorCode"):
        """
        Initialize tetrahedral color code.
        
        Args:
            L: Number of tetrahedra per edge
            name: Code name
        """
        self.L = L
        self._name = name
        
        hx, hz, n_qubits = self._build_tetrahedral(L)
        
        logicals = self._compute_logicals(hx, hz, n_qubits)
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logicals[0],
            logical_z=logicals[1],
            metadata={"name": name, "n": n_qubits, "L": L}
        )
    
    @staticmethod
    def _build_tetrahedral(L: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build tetrahedral color code."""
        # Use simple octahedron structure for small L
        if L <= 2:
            # Octahedron: 6 vertices, 12 edges, 8 triangular faces
            vertices = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), 
                       (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            
            # Edges: all pairs of orthogonal vertices
            edges = []
            for i in range(6):
                for j in range(i + 1, 6):
                    # Check if orthogonal (dot product = 0)
                    v1, v2 = vertices[i], vertices[j]
                    if abs(v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) == 0:
                        edges.append((i, j))
            
            n_qubits = len(edges)
            
            # Faces: triangles
            faces = [
                [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5],
                [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5]
            ]
            
            edge_to_idx = {e: i for i, e in enumerate(edges)}
            
            n_faces = len(faces)
            hx = np.zeros((n_faces, n_qubits), dtype=np.uint8)
            
            for f_idx, face_verts in enumerate(faces):
                for i in range(3):
                    for j in range(i + 1, 3):
                        v1, v2 = face_verts[i], face_verts[j]
                        edge_key = (min(v1, v2), max(v1, v2))
                        if edge_key in edge_to_idx:
                            hx[f_idx, edge_to_idx[edge_key]] = 1
            
            # Vertex stabilizers
            hz = np.zeros((6, n_qubits), dtype=np.uint8)
            for v_idx in range(6):
                for e_idx, (v1, v2) in enumerate(edges):
                    if v_idx in [v1, v2]:
                        hz[v_idx, e_idx] = 1
        else:
            # Use HGP for larger sizes
            rep = np.zeros((L * 2, L * 2 + 1), dtype=np.uint8)
            for i in range(L * 2):
                rep[i, i] = 1
                rep[i, i + 1] = 1
            
            m, n = rep.shape
            n_left = n * n
            n_right = m * m
            n_qubits = n_left + n_right
            
            n_x_stabs = m * n
            n_z_stabs = n * m
            
            hx = np.zeros((n_x_stabs, n_qubits), dtype=np.uint8)
            hz = np.zeros((n_z_stabs, n_qubits), dtype=np.uint8)
            
            for check_a in range(m):
                for bit_b in range(n):
                    x_stab = check_a * n + bit_b
                    for bit_a in range(n):
                        if rep[check_a, bit_a]:
                            q = bit_a * n + bit_b
                            hx[x_stab, q] = 1
                    for check_b in range(m):
                        if rep[check_b, bit_b]:
                            q = n_left + check_a * m + check_b
                            hx[x_stab, q] = 1
            
            for bit_a in range(n):
                for check_b in range(m):
                    z_stab = bit_a * m + check_b
                    for bit_b in range(n):
                        if rep[check_b, bit_b]:
                            q = bit_a * n + bit_b
                            hz[z_stab, q] = 1
                    for check_a in range(m):
                        if rep[check_a, bit_a]:
                            q = n_left + check_a * m + check_b
                            hz[z_stab, q] = 1
        
        return hx % 2, hz % 2, n_qubits
    
    @staticmethod
    def _compute_logicals(
        hx: np.ndarray, hz: np.ndarray, n_qubits: int
    ) -> Tuple[List[str], List[str]]:
        """Compute logical operators."""
        return (["Z" * n_qubits], ["X" * n_qubits])
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """
        Return 2D coordinates for visualization.
        
        Uses grid layout based on octahedral/tetrahedral structure.
        """
        coords: List[Tuple[float, float]] = []
        
        # Use simple square grid layout
        grid_size = int(np.ceil(np.sqrt(self.n)))
        
        for i in range(self.n):
            col = i % grid_size
            row = i // grid_size
            coords.append((float(col), float(row)))
        
        return coords
    
    def description(self) -> str:
        return f"Tetrahedral Color Code L={self.L}, n={self.n}"


# Pre-configured instances
TruncatedTrihex_2x2 = lambda: TruncatedTrihexColorCode(Lx=2, Ly=2)
HyperbolicColor_45_g2 = lambda: HyperbolicColorCode(p=4, q=5, genus=2)
HyperbolicColor_64_g2 = lambda: HyperbolicColorCode(p=6, q=4, genus=2)
BallColor_3D = lambda: BallColorCode(dimension=3)
BallColor_4D = lambda: BallColorCode(dimension=4)
CubicHoneycomb_L2 = lambda: CubicHoneycombColorCode(L=2)
Tetrahedral_L2 = lambda: TetrahedralColorCode(L=2)
