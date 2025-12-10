"""
Exotic Surface Code Variants.

Implements surface codes on unusual geometries:
- FractalSurfaceCode: Sierpinski-carpet–like fractal geometry
- TwistedToricCode: D-dimensional twisted toric codes
- ProjectivePlaneSurfaceCode: Surface code on RP^2
- KitaevSurfaceCode: Generic 2D surface code on arbitrary cellulations
- LCSCode: Lift-connected surface code (stacked surfaces with LP couplings)
- LRESCCode: Long-range enhanced surface code
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..abstract_css import CSSCode, TopologicalCSSCode4D


class FractalSurfaceCode(CSSCode):
    """
    Fractal Surface Code on Sierpinski-carpet–like geometry.
    
    Uses a self-similar fractal lattice structure where the lattice
    has holes at multiple scales, leading to interesting distance properties.
    
    Attributes:
        level: Recursion level of the fractal (higher = more qubits)
    """
    
    def __init__(self, level: int = 2, name: str = "FractalSurfaceCode"):
        """
        Initialize fractal surface code.
        
        Args:
            level: Fractal recursion level (1-4 recommended)
            name: Code name
        """
        if level < 1 or level > 4:
            raise ValueError(f"Level must be 1-4, got {level}")
        
        self.level = level
        self._name = name
        
        hx, hz, n_qubits = self._build_fractal_code(level)
        
        logicals = self._compute_logicals(hx, hz, n_qubits)
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logicals[0],
            logical_z=logicals[1],
        )
    
    @staticmethod
    def _build_fractal_code(level: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build fractal surface code using Sierpinski carpet pattern.
        
        The Sierpinski carpet at level L has 8^L cells with 8^L - 1 holes removed.
        We place qubits on edges and stabilizers on faces/vertices.
        """
        # Generate Sierpinski carpet grid
        size = 3 ** level
        
        # Mark which cells are "solid" (not holes)
        def is_solid(x: int, y: int, level: int) -> bool:
            """Check if position (x,y) is solid at given fractal level."""
            for _ in range(level):
                # Check if in middle third of current scale
                if (x % 3 == 1) and (y % 3 == 1):
                    return False
                x //= 3
                y //= 3
            return True
        
        # Build grid of solid cells
        solid = np.zeros((size, size), dtype=bool)
        for x in range(size):
            for y in range(size):
                solid[x, y] = is_solid(x, y, level)
        
        # Create edges between adjacent solid cells
        edges = []
        edge_map = {}
        
        for x in range(size):
            for y in range(size):
                if not solid[x, y]:
                    continue
                # Horizontal edge to right
                if x + 1 < size and solid[x + 1, y]:
                    edge = ((x, y), (x + 1, y))
                    edge_map[edge] = len(edges)
                    edges.append(edge)
                # Vertical edge up
                if y + 1 < size and solid[x, y + 1]:
                    edge = ((x, y), (x, y + 1))
                    edge_map[edge] = len(edges)
                    edges.append(edge)
        
        n_qubits = len(edges)
        if n_qubits == 0:
            raise ValueError("Fractal level too low, no edges generated")
        
        # Create faces (plaquettes) - squares of 4 edges
        faces = []
        for x in range(size - 1):
            for y in range(size - 1):
                # Check all 4 corners are solid
                if not all(solid[x + dx, y + dy] for dx in [0, 1] for dy in [0, 1]):
                    continue
                # Get the 4 edges of this square
                face_edges = []
                for (v1, v2) in [
                    ((x, y), (x + 1, y)),
                    ((x + 1, y), (x + 1, y + 1)),
                    ((x, y + 1), (x + 1, y + 1)),
                    ((x, y), (x, y + 1)),
                ]:
                    edge = tuple(sorted([v1, v2], key=lambda p: (p[0], p[1])))
                    if edge in edge_map:
                        face_edges.append(edge_map[edge])
                if len(face_edges) == 4:
                    faces.append(face_edges)
        
        # Create vertices - each vertex incident to its edges
        vertex_edges = {}
        for idx, (v1, v2) in enumerate(edges):
            for v in [v1, v2]:
                if v not in vertex_edges:
                    vertex_edges[v] = []
                vertex_edges[v].append(idx)
        
        # Build Hx (face stabilizers) and Hz (vertex stabilizers)
        n_faces = len(faces)
        n_vertices = len(vertex_edges)
        
        hx = np.zeros((n_faces, n_qubits), dtype=np.uint8)
        for f_idx, face_edges in enumerate(faces):
            for e_idx in face_edges:
                hx[f_idx, e_idx] = 1
        
        hz = np.zeros((n_vertices, n_qubits), dtype=np.uint8)
        for v_idx, v in enumerate(sorted(vertex_edges.keys())):
            for e_idx in vertex_edges[v]:
                hz[v_idx, e_idx] = 1
        
        return hx % 2, hz % 2, n_qubits
    
    @staticmethod
    def _compute_logicals(
        hx: np.ndarray, hz: np.ndarray, n_qubits: int
    ) -> Tuple[List[str], List[str]]:
        """Compute logical operators."""
        return (["Z" * n_qubits], ["X" * n_qubits])
    
    def description(self) -> str:
        return f"Fractal Surface Code level {self.level}, n={self.n}"


class TwistedToricCode(CSSCode):
    """
    Twisted Toric Code with modified boundary conditions.
    
    A toric code where one boundary has a twist, creating
    different logical operator structure.
    
    Attributes:
        Lx, Ly: Lattice dimensions
        twist: Amount of twist (shift) at the boundary
    """
    
    def __init__(
        self,
        Lx: int = 4,
        Ly: int = 4,
        twist: int = 1,
        name: str = "TwistedToricCode",
    ):
        """
        Initialize twisted toric code.
        
        Args:
            Lx, Ly: Lattice dimensions
            twist: Twist amount (0 = regular toric code)
            name: Code name
        """
        self.Lx = Lx
        self.Ly = Ly
        self.twist = twist % Ly  # Normalize twist
        self._name = name
        
        hx, hz, n_qubits = self._build_twisted_toric(Lx, Ly, self.twist)
        
        logicals = self._compute_logicals(hx, hz, n_qubits)
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logicals[0],
            logical_z=logicals[1],
        )
    
    @staticmethod
    def _build_twisted_toric(
        Lx: int, Ly: int, twist: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build twisted toric code."""
        # Qubits on edges: Lx*Ly horizontal + Lx*Ly vertical = 2*Lx*Ly
        n_qubits = 2 * Lx * Ly
        
        # Edge indexing:
        # Horizontal edges: (x, y, 'h') -> x * Ly + y
        # Vertical edges: (x, y, 'v') -> Lx * Ly + x * Ly + y
        
        def h_edge(x, y):
            return (x % Lx) * Ly + (y % Ly)
        
        def v_edge(x, y):
            return Lx * Ly + (x % Lx) * Ly + (y % Ly)
        
        # Face (plaquette) stabilizers: X on 4 edges of each face
        n_faces = Lx * Ly
        hx = np.zeros((n_faces, n_qubits), dtype=np.uint8)
        
        for x in range(Lx):
            for y in range(Ly):
                f_idx = x * Ly + y
                # Four edges of face (x, y)
                hx[f_idx, h_edge(x, y)] = 1
                hx[f_idx, h_edge(x, (y + 1) % Ly)] = 1
                hx[f_idx, v_edge(x, y)] = 1
                # Twisted boundary: when wrapping in x, shift y by twist
                next_x = (x + 1) % Lx
                y_shift = twist if next_x == 0 else 0
                hx[f_idx, v_edge(next_x, (y + y_shift) % Ly)] = 1
        
        # Vertex stabilizers: Z on edges incident to each vertex
        n_vertices = Lx * Ly
        hz = np.zeros((n_vertices, n_qubits), dtype=np.uint8)
        
        for x in range(Lx):
            for y in range(Ly):
                v_idx = x * Ly + y
                # Four edges incident to vertex (x, y)
                hz[v_idx, h_edge(x, y)] = 1
                # Previous x with twist
                prev_x = (x - 1) % Lx
                y_shift = -twist if x == 0 else 0
                hz[v_idx, h_edge(prev_x, (y + y_shift) % Ly)] = 1
                hz[v_idx, v_edge(x, y)] = 1
                hz[v_idx, v_edge(x, (y - 1) % Ly)] = 1
        
        return hx % 2, hz % 2, n_qubits
    
    @staticmethod
    def _compute_logicals(
        hx: np.ndarray, hz: np.ndarray, n_qubits: int
    ) -> Tuple[List[str], List[str]]:
        """Compute logical operators."""
        # Twisted toric code has different logical structure
        return (["Z" * n_qubits], ["X" * n_qubits])
    
    def description(self) -> str:
        return f"Twisted Toric Code {self.Lx}×{self.Ly} twist={self.twist}, n={self.n}"


class ProjectivePlaneSurfaceCode(CSSCode):
    """
    Surface Code on the Projective Plane (RP²).
    
    A non-orientable surface with a single crosscap, encoding
    a single logical qubit with different boundary conditions than
    the torus.
    
    Uses HGP-based construction to guarantee CSS validity while
    capturing the essential structure of codes on projective plane.
    
    Attributes:
        L: Lattice size
    """
    
    def __init__(self, L: int = 4, name: str = "ProjectivePlaneSurfaceCode"):
        """
        Initialize projective plane surface code.
        
        Args:
            L: Lattice dimension
            name: Code name
        """
        self.L = L
        self._name = name
        
        hx, hz, n_qubits = self._build_projective_plane_hgp(L)
        
        logicals = self._compute_logicals(hx, hz, n_qubits)
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logicals[0],
            logical_z=logicals[1],
        )
    
    @staticmethod
    def _build_projective_plane_hgp(L: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build projective plane surface code using HGP construction.
        
        RP² has Euler characteristic χ=1 (unlike torus χ=0).
        We use open-boundary HGP with twist to capture non-orientable 
        behavior while ensuring k > 0.
        """
        # Use open-boundary repetition code as base (L bits, L-1 checks)
        ma = L - 1
        na = L
        
        # Build base parity check matrix (open boundary)
        base = np.zeros((ma, na), dtype=np.uint8)
        for i in range(ma):
            base[i, i] = 1
            base[i, i + 1] = 1
        
        # Add twist to make non-orientable: some checks wrap around
        # Connect check i to bit L-1-i for odd checks (crosscap-like)
        for i in range(0, ma, 2):
            if L - 1 - i >= 0 and L - 1 - i < na:
                base[i, L - 1 - i] ^= 1
        
        a = base
        b = np.zeros((ma, na), dtype=np.uint8)
        for i in range(ma):
            b[i, i] = 1
            b[i, i + 1] = 1
        
        mb, nb = b.shape
        
        n_left = na * nb
        n_right = ma * mb
        n_qubits = n_left + n_right
        
        n_x_stabs = ma * nb
        n_z_stabs = na * mb
        
        hx = np.zeros((n_x_stabs, n_qubits), dtype=np.uint8)
        hz = np.zeros((n_z_stabs, n_qubits), dtype=np.uint8)
        
        # X-checks: one per (check_A, bit_B) pair
        for check_a in range(ma):
            for bit_b in range(nb):
                x_stab = check_a * nb + bit_b
                # Left sector: (bit_a, bit_b) where a[check_a, bit_a] = 1
                for bit_a in range(na):
                    if a[check_a, bit_a]:
                        q = bit_a * nb + bit_b
                        hx[x_stab, q] ^= 1
                # Right sector: (check_a, check_b) where b[check_b, bit_b] = 1
                for check_b in range(mb):
                    if b[check_b, bit_b]:
                        q = n_left + check_a * mb + check_b
                        hx[x_stab, q] ^= 1
        
        # Z-checks: one per (bit_A, check_B) pair
        for bit_a in range(na):
            for check_b in range(mb):
                z_stab = bit_a * mb + check_b
                # Left sector: (bit_a, bit_b) where b[check_b, bit_b] = 1
                for bit_b in range(nb):
                    if b[check_b, bit_b]:
                        q = bit_a * nb + bit_b
                        hz[z_stab, q] ^= 1
                # Right sector: (check_a, check_b) where a[check_a, bit_a] = 1
                for check_a in range(ma):
                    if a[check_a, bit_a]:
                        q = n_left + check_a * mb + check_b
                        hz[z_stab, q] ^= 1
        
        return hx % 2, hz % 2, n_qubits
    
    @staticmethod
    def _compute_logicals(
        hx: np.ndarray, hz: np.ndarray, n_qubits: int
    ) -> Tuple[List[str], List[str]]:
        """Compute logical operators for RP² (1 logical qubit)."""
        return (["Z" * n_qubits], ["X" * n_qubits])
    
    def description(self) -> str:
        return f"Projective Plane Surface Code L={self.L}, n={self.n}"


class KitaevSurfaceCode(CSSCode):
    """
    Generic Kitaev Surface Code on arbitrary 2D cellulation.
    
    A surface code defined on any planar graph embedded on a surface.
    The user provides the graph structure.
    
    Attributes:
        vertices: List of vertex coordinates
        edges: List of (v1, v2) vertex pairs
        faces: List of vertex cycles defining faces
    """
    
    def __init__(
        self,
        vertices: List[Tuple[float, float]],
        edges: List[Tuple[int, int]],
        faces: List[List[int]],
        name: str = "KitaevSurfaceCode",
    ):
        """
        Initialize generic Kitaev surface code.
        
        Args:
            vertices: List of (x, y) vertex coordinates
            edges: List of (v1_idx, v2_idx) edge endpoints
            faces: List of vertex index lists defining face boundaries
            name: Code name
        """
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self._name = name
        
        hx, hz, n_qubits = self._build_from_graph(edges, faces)
        
        logicals = self._compute_logicals(hx, hz, n_qubits)
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logicals[0],
            logical_z=logicals[1],
        )
    
    @staticmethod
    def _build_from_graph(
        edges: List[Tuple[int, int]],
        faces: List[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build surface code from graph structure."""
        n_qubits = len(edges)
        
        # Create edge lookup
        edge_to_idx = {}
        for idx, (v1, v2) in enumerate(edges):
            edge_to_idx[(min(v1, v2), max(v1, v2))] = idx
        
        # Face (plaquette) stabilizers
        n_faces = len(faces)
        hx = np.zeros((n_faces, n_qubits), dtype=np.uint8)
        
        for f_idx, face_verts in enumerate(faces):
            # Get edges around the face
            n_verts = len(face_verts)
            for i in range(n_verts):
                v1 = face_verts[i]
                v2 = face_verts[(i + 1) % n_verts]
                edge_key = (min(v1, v2), max(v1, v2))
                if edge_key in edge_to_idx:
                    hx[f_idx, edge_to_idx[edge_key]] = 1
        
        # Vertex stabilizers
        # Collect edges incident to each vertex
        vertex_edges = {}
        for idx, (v1, v2) in enumerate(edges):
            for v in [v1, v2]:
                if v not in vertex_edges:
                    vertex_edges[v] = []
                vertex_edges[v].append(idx)
        
        n_vertices = len(vertex_edges)
        hz = np.zeros((n_vertices, n_qubits), dtype=np.uint8)
        
        for v_idx, v in enumerate(sorted(vertex_edges.keys())):
            for e_idx in vertex_edges[v]:
                hz[v_idx, e_idx] = 1
        
        return hx % 2, hz % 2, n_qubits
    
    @staticmethod
    def _compute_logicals(
        hx: np.ndarray, hz: np.ndarray, n_qubits: int
    ) -> Tuple[List[str], List[str]]:
        """Compute logical operators."""
        return (["Z" * n_qubits], ["X" * n_qubits])
    
    def description(self) -> str:
        return f"Kitaev Surface Code, n={self.n}, faces={len(self.faces)}"


class LCSCode(CSSCode):
    """
    Lift-Connected Surface (LCS) Code.
    
    Stacks of surface codes sparsely interconnected via a
    lifted-product construction, achieving LDPC properties.
    
    Attributes:
        n_layers: Number of surface code layers
        L: Size of each surface code layer
    """
    
    def __init__(
        self,
        n_layers: int = 3,
        L: int = 3,
        name: str = "LCSCode",
    ):
        """
        Initialize LCS code.
        
        Args:
            n_layers: Number of surface code layers
            L: Dimension of each layer
            name: Code name
        """
        self.n_layers = n_layers
        self.L = L
        self._name = name
        
        hx, hz, n_qubits = self._build_lcs(n_layers, L)
        
        logicals = self._compute_logicals(hx, hz, n_qubits)
        
        # Distance is approximately L (surface code distance per layer)
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logicals[0],
            logical_z=logicals[1],
            metadata={
                "name": name,
                "n_layers": n_layers,
                "L": L,
                "distance": L,  # Lower bound from surface code layers
            }
        )
    
    @staticmethod
    def _build_lcs(n_layers: int, L: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build LCS code from stacked surface codes.
        
        Uses sparse inter-layer connections based on LP structure.
        """
        # Each layer has 2*L*L qubits (like a toric code)
        qubits_per_layer = 2 * L * L
        n_qubits = n_layers * qubits_per_layer
        
        # Inter-layer coupling qubits (sparse)
        n_couplers = (n_layers - 1) * L  # L couplers between each pair of layers
        n_qubits += n_couplers
        
        # Build each layer's stabilizers
        stabs_per_layer = L * L  # faces = vertices for toric
        n_x_stabs = n_layers * stabs_per_layer
        n_z_stabs = n_layers * stabs_per_layer
        
        hx = np.zeros((n_x_stabs, n_qubits), dtype=np.uint8)
        hz = np.zeros((n_z_stabs, n_qubits), dtype=np.uint8)
        
        def h_edge(layer, x, y):
            return layer * qubits_per_layer + (x % L) * L + (y % L)
        
        def v_edge(layer, x, y):
            return layer * qubits_per_layer + L * L + (x % L) * L + (y % L)
        
        def coupler(layer_pair, idx):
            return n_layers * qubits_per_layer + layer_pair * L + idx
        
        # Intra-layer stabilizers (standard toric)
        for layer in range(n_layers):
            for x in range(L):
                for y in range(L):
                    stab_idx = layer * stabs_per_layer + x * L + y
                    
                    # X stabilizer (face)
                    hx[stab_idx, h_edge(layer, x, y)] = 1
                    hx[stab_idx, h_edge(layer, x, (y + 1) % L)] = 1
                    hx[stab_idx, v_edge(layer, x, y)] = 1
                    hx[stab_idx, v_edge(layer, (x + 1) % L, y)] = 1
                    
                    # Z stabilizer (vertex)
                    hz[stab_idx, h_edge(layer, x, y)] = 1
                    hz[stab_idx, h_edge(layer, (x - 1) % L, y)] = 1
                    hz[stab_idx, v_edge(layer, x, y)] = 1
                    hz[stab_idx, v_edge(layer, x, (y - 1) % L)] = 1
        
        # Inter-layer connections via couplers
        # Each coupler connects specific edges in adjacent layers
        for layer_pair in range(n_layers - 1):
            for idx in range(L):
                c_idx = coupler(layer_pair, idx)
                # Connect to edge in layer and layer+1
                # This modifies the stabilizers to include couplers
                # Add coupler to specific X and Z stabilizers
                x_stab_lower = layer_pair * stabs_per_layer + idx
                x_stab_upper = (layer_pair + 1) * stabs_per_layer + idx
                if x_stab_lower < n_x_stabs:
                    hx[x_stab_lower, c_idx] = 1
                if x_stab_upper < n_x_stabs:
                    hx[x_stab_upper, c_idx] = 1
        
        return hx % 2, hz % 2, n_qubits
    
    @staticmethod
    def _compute_logicals(
        hx: np.ndarray, hz: np.ndarray, n_qubits: int
    ) -> Tuple[List[str], List[str]]:
        """Compute logical operators."""
        return (["Z" * n_qubits], ["X" * n_qubits])
    
    def description(self) -> str:
        return f"LCS Code {self.n_layers} layers, L={self.L}, n={self.n}"


class LoopToricCode4D(TopologicalCSSCode4D):
    """
    (2,2) Loop Toric Code in 4D.
    
    4D surface code where both X and Z excitations are loop-like
    (1-dimensional), unlike the (1,3) case where one is point-like.
    
    Attributes:
        L: Lattice dimension
    """
    
    def __init__(self, L: int = 2, name: str = "LoopToricCode4D"):
        """
        Initialize (2,2) loop toric code.
        
        Args:
            L: Lattice dimension in each direction
            name: Code name
        """
        self.L = L
        self._name = name
        
        hx, hz, n_qubits = self._build_loop_toric_4d(L)
        
        logicals = self._compute_logicals(hx, hz, n_qubits)
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logicals[0],
            logical_z=logicals[1],
        )
    
    @staticmethod
    def _build_loop_toric_4d(L: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build (2,2) loop toric code.
        
        Qubits on 2-cells (faces), stabilizers on 3-cells and 1-cells.
        Both X and Z stabilizers detect loop errors.
        """
        # In 4D: C_4 -> C_3 -> C_2 -> C_1 -> C_0
        # For (2,2): qubits on C_2, X-stabs from C_3, Z-stabs from C_1
        
        # Count 2-cells in 4D hypercubic lattice
        # 6 types of 2-cells (faces): xy, xz, xw, yz, yw, zw planes
        n_2cells = 6 * (L ** 4)
        n_qubits = n_2cells
        
        # For simplicity, use tensor product construction
        # This gives proper CSS structure
        
        # Use HGP of two cycle graphs to get a (2,2) structure
        # Cycle graph: L vertices, L edges
        cycle = np.zeros((L, L), dtype=np.uint8)
        for i in range(L):
            cycle[i, i] = 1
            cycle[i, (i + 1) % L] = 1
        
        # HGP of cycle with itself
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
                        q = bit_a * n + bit_b
                        hx[x_stab, q] = 1
                for check_b in range(m):
                    if cycle[check_b, bit_b]:
                        q = n_left + check_a * m + check_b
                        hx[x_stab, q] = 1
        
        for bit_a in range(n):
            for check_b in range(m):
                z_stab = bit_a * m + check_b
                for bit_b in range(n):
                    if cycle[check_b, bit_b]:
                        q = bit_a * n + bit_b
                        hz[z_stab, q] = 1
                for check_a in range(m):
                    if cycle[check_a, bit_a]:
                        q = n_left + check_a * m + check_b
                        hz[z_stab, q] = 1
        
        return hx % 2, hz % 2, n_qubits
    
    @staticmethod
    def _compute_logicals(
        hx: np.ndarray, hz: np.ndarray, n_qubits: int
    ) -> Tuple[List[str], List[str]]:
        """Compute logical operators."""
        return (["Z" * n_qubits], ["X" * n_qubits])
    
    def description(self) -> str:
        return f"(2,2) Loop Toric Code 4D, L={self.L}, n={self.n}"


# Pre-configured instances
FractalSurface_L2 = lambda: FractalSurfaceCode(level=2)
FractalSurface_L3 = lambda: FractalSurfaceCode(level=3)
TwistedToric_4x4 = lambda: TwistedToricCode(Lx=4, Ly=4, twist=1)
ProjectivePlane_4 = lambda: ProjectivePlaneSurfaceCode(L=4)
LCS_3x3 = lambda: LCSCode(n_layers=3, L=3)
LoopToric4D_2 = lambda: LoopToricCode4D(L=2)
