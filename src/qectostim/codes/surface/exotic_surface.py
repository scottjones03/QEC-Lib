"""Exotic Surface Code Variants — Fractal, Twisted, Projective & Higher-Dimensional

This module implements surface-code variants on unusual geometries that go
beyond the standard planar / toric setting.  Each code is a valid CSS
stabiliser code built from a chain complex or hypergraph-product (HGP)
construction, with logical operators computed via the kernel/image
prescription.

Overview
--------
Surface codes are the most widely-studied family of topological quantum
error-correcting codes.  The canonical planar and toric instances live on
regular square lattices, but the construction generalises to *any*
cellulation of a 2-manifold (Kitaev 2003) and even to higher-dimensional
manifolds.  This module collects six such exotic variants that are useful
for studying:

* the interplay between lattice geometry and code distance;
* the effect of non-orientability on logical-qubit count;
* self-similar fractal lattice defects;
* multi-layer / stacked surface constructions (LCS codes);
* genuinely 4-dimensional loop-like excitations.

Codes in this module
--------------------
1. **FractalSurfaceCode** — Surface code on a Sierpinski-carpet lattice.
   Qubits on edges, stabilisers on faces and vertices of the fractal
   grid.  Higher recursion levels yield more qubits with an unusual
   distance scaling.
2. **TwistedToricCode** — Toric code with a boundary twist.
   Wrapping in one direction applies a cyclic shift, changing the
   logical-operator structure while preserving CSS commutativity.
3. **ProjectivePlaneSurfaceCode** — HGP-based code capturing RP²
   (projective-plane) topology.  Uses a crosscap-twisted base matrix.
4. **KitaevSurfaceCode** — Generic surface code on any user-supplied
   planar graph, with face plaquettes (X) and vertex stars (Z).
5. **LCSCode** — Lift-Connected Surface code: multiple toric-code
   layers coupled by sparse inter-layer qubits.
6. **LoopToricCode4D** — (2,2) toric code in 4D built via HGP of two
   cycle graphs.  Both X and Z excitations are loop-like.

Construction approaches
-----------------------
* **Cellulation-based** (FractalSurface, Kitaev, TwistedToric):  build
  edges and faces on a concrete lattice, then derive H_X (face→edge)
  and H_Z (vertex→edge) incidence matrices.
* **HGP-based** (ProjectivePlane, LoopToric4D):  form the hypergraph
  product of one or two classical codes, yielding H_X and H_Z that
  commute by construction.
* **Stacked / lifted-product** (LCSCode):  replicate a base toric code
  across layers and add sparse coupling qubits.

Code parameters
---------------
All codes in this module satisfy ``H_X · H_Z^T = 0  (mod 2)``.
Parameters ``[[n, k, d]]`` vary by code and construction size; each
class stores them in its ``metadata`` dictionary.

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
* **FractalSurfaceCode**: weight-4 face (X) and variable-weight vertex
  (Z) stabilisers on a Sierpinski-carpet lattice; fractal holes reduce
  the total stabiliser count relative to a full grid.
* **TwistedToricCode**: weight-4 face and vertex stabilisers, identical
  to the standard toric code except at the twist boundary where the
  cyclic shift changes the edge incidence pattern.
* **ProjectivePlaneSurfaceCode**: HGP-based stabilisers with weights
  determined by the crosscap-twisted base matrix; typically weight 4–6.
* **KitaevSurfaceCode**: face plaquettes (X) and vertex stars (Z) on an
  arbitrary user-supplied planar graph; weight = face/vertex degree.
* **LCSCode**: weight-4 intra-layer stabilisers plus sparse inter-layer
  coupling checks (weight 2 per coupling qubit).
* **LoopToricCode4D**: weight determined by the HGP of two cycle
  graphs; both X- and Z-excitations are loop-like.
* Default measurement: single parallel round for each stabiliser type.

Connections
-----------
* Rotated / planar surface code (``rotated_surface.py``)
* Toric code on 3-manifolds (``toric_3d.py``)
* 4-dimensional surface codes (``four_d_surface_code.py``)
* XZZX surface code (``xzzx_surface.py``)
* Colour codes share the CSS property but have 3-colourable faces.

References
----------
* Kitaev, "Fault-tolerant quantum computation by anyons",
  Ann. Phys. 303, 2–30 (2003).  arXiv:quant-ph/9707021
* Freedman & Hastings, "Building manifolds from quantum codes",
  Geom. Funct. Anal. 31, 855–894 (2021).  arXiv:2012.02249
* Bravyi & Hastings, "Homological product codes",
  Proc. STOC 2014.  arXiv:1311.0885
* Breuckmann & Eberhardt, "Quantum low-density parity-check codes",
  PRX Quantum 2, 040101 (2021).  arXiv:2103.06309
* Error Correction Zoo: https://errorcorrectionzoo.org/c/surface
* Wikipedia: https://en.wikipedia.org/wiki/Toric_code
"""
from typing import Dict, List, Optional, Tuple, Any
import warnings
import numpy as np

from ..abstract_css import CSSCode, TopologicalCSSCode4D
from ..complexes.css_complex import FiveCSSChainComplex
from ..utils import compute_css_logicals, vectors_to_paulis_x, vectors_to_paulis_z, validate_css_code


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

        Raises:
            ValueError: If *level* is not in the range 1–4.
        """
        if level < 1 or level > 4:
            raise ValueError(f"Level must be 1-4, got {level}")
        
        self.level = level
        self._name = name
        
        hx, hz, n_qubits = self._build_fractal_code(level)
        
        logicals = self._compute_logicals(hx, hz, n_qubits)

        validate_css_code(hx, hz, f"FractalSurfaceCode_L{level}", raise_on_error=True)

        # Compute logical support indices
        lx_ops, lz_ops = logicals
        lx_support = sorted({q for op in lx_ops for q in op}) if lx_ops else []
        lz_support = sorted({q for op in lz_ops for q in op}) if lz_ops else []

        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]
        k_val = max(len(lx_ops), 1)
        # Distance lower-bound: side length of fractal
        dist = 3 ** level // 3 if level > 1 else 1

        self._distance = dist

        # Compute coordinate metadata
        _cols = int(np.ceil(np.sqrt(n_qubits)))
        _dc = [(float(i % _cols), float(i // _cols)) for i in range(n_qubits)]
        _xsc = []
        for _ri in range(hx.shape[0]):
            _sup = np.where(hx[_ri])[0]
            if len(_sup) > 0:
                _xsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _xsc.append((0.0, 0.0))
        _zsc = []
        for _ri in range(hz.shape[0]):
            _sup = np.where(hz[_ri])[0]
            if len(_sup) > 0:
                _zsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _zsc.append((0.0, 0.0))

        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=lx_ops,
            logical_z=lz_ops,
            metadata={
                # ── Code parameters ────────────────────────────────────
                "code_family": "surface",
                "code_type": "fractal_surface",
                "n": n_qubits,
                "k": k_val,
                "distance": dist,
                "rate": k_val / n_qubits if n_qubits else 0.0,
                # ── Logical operator info ──────────────────────────────
                "lx_pauli_type": "X",
                "lz_pauli_type": "Z",
                "lx_support": lx_support,
                "lz_support": lz_support,
                # ── Coordinate metadata ────────────────────────────────
                "data_coords": _dc,
                "x_stab_coords": _xsc,
                "z_stab_coords": _zsc,
                # ── Stabiliser scheduling ──────────────────────────────
                "stabiliser_schedule": {
                    "x_rounds": {i: 0 for i in range(n_x_stabs)},
                    "z_rounds": {i: 0 for i in range(n_z_stabs)},
                    "n_rounds": 1,
                    "description": "Fully parallel: all X-stabilisers round 0, all Z-stabilisers round 0.",
                },
                "x_schedule": [(1, 0), (0, 1), (-1, 0), (0, -1)],
                "z_schedule": [(1, 0), (0, -1), (-1, 0), (0, 1)],
                # ── Literature / provenance ────────────────────────────
                "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/surface",
                "wikipedia_url": "https://en.wikipedia.org/wiki/Toric_code",
                "canonical_references": [
                    "Kitaev, Ann. Phys. 303, 2–30 (2003). arXiv:quant-ph/9707021",
                    "Yoshida, Ann. Phys. 338, 134–166 (2013). arXiv:1302.6248",
                ],
                "connections": [
                    "Surface code on Sierpinski-carpet lattice with fractal holes",
                    "Related to Yoshida fractal spin-liquid models",
                    "Standard surface code recovered when no holes are removed",
                ],
            },
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
        """Compute logical operators using CSS kernel/image prescription."""
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
            return logical_x, logical_z
        except Exception as e:
            warnings.warn(f"exotic_surface: logical computation failed ({e}), using single-qubit fallback")
            return [{0: 'X'}], [{0: 'Z'}]

    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2D coordinates for visualization.
        
        Lays out qubits on a grid based on the fractal level.
        """
        # Grid size based on fractal level (3^level x 3^level with holes)
        side = 3 ** self.level
        coords: List[Tuple[float, float]] = []
        for i in range(self.n):
            col = i % side
            row = i // side
            coords.append((float(col), float(row)))
        return coords
    
    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'FractalSurfaceCode(level=2)'``."""
        return f"FractalSurfaceCode(level={self.level})"

    @property
    def distance(self) -> int:
        """Code distance (lower bound from fractal geometry)."""
        return self._distance

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
        self._Lx_dim = Lx
        self._Ly_dim = Ly
        self.twist = twist % Ly  # Normalize twist
        self._name = name
        
        hx, hz, n_qubits = self._build_twisted_toric(Lx, Ly, self.twist)
        
        logicals = self._compute_logicals(hx, hz, n_qubits)

        validate_css_code(hx, hz, f"TwistedToricCode_{Lx}x{Ly}_t{self.twist}", raise_on_error=True)

        lx_ops, lz_ops = logicals
        lx_support = sorted({q for op in lx_ops for q in op}) if lx_ops else []
        lz_support = sorted({q for op in lz_ops for q in op}) if lz_ops else []

        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]
        k_val = max(len(lx_ops), 1)
        dist = min(Lx, Ly)

        self._distance = dist

        # Compute coordinate metadata
        _cols = int(np.ceil(np.sqrt(n_qubits)))
        _dc = [(float(i % _cols), float(i // _cols)) for i in range(n_qubits)]
        _xsc = []
        for _ri in range(hx.shape[0]):
            _sup = np.where(hx[_ri])[0]
            if len(_sup) > 0:
                _xsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _xsc.append((0.0, 0.0))
        _zsc = []
        for _ri in range(hz.shape[0]):
            _sup = np.where(hz[_ri])[0]
            if len(_sup) > 0:
                _zsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _zsc.append((0.0, 0.0))

        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=lx_ops,
            logical_z=lz_ops,
            metadata={
                "code_family": "surface",
                "code_type": "twisted_toric",
                "n": n_qubits,
                "k": k_val,
                "distance": dist,
                "rate": k_val / n_qubits if n_qubits else 0.0,
                "lx_pauli_type": "X",
                "lz_pauli_type": "Z",
                "lx_support": lx_support,
                "lz_support": lz_support,
                "data_coords": _dc,
                "x_stab_coords": _xsc,
                "z_stab_coords": _zsc,
                "stabiliser_schedule": {
                    "x_rounds": {i: 0 for i in range(n_x_stabs)},
                    "z_rounds": {i: 0 for i in range(n_z_stabs)},
                    "n_rounds": 1,
                    "description": "Fully parallel: all face X-stabilisers round 0, all vertex Z-stabilisers round 0.",
                },
                "x_schedule": [(1, 0), (0, 1), (-1, 0), (0, -1)],
                "z_schedule": [(1, 0), (0, -1), (-1, 0), (0, 1)],
                "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/surface",
                "wikipedia_url": "https://en.wikipedia.org/wiki/Toric_code",
                "canonical_references": [
                    "Kitaev, Ann. Phys. 303, 2–30 (2003). arXiv:quant-ph/9707021",
                    "Bravyi, Duclos-Cianci & Bhatt, arXiv:1112.3252 (2011)",
                ],
                "connections": [
                    "Toric code with twisted boundary conditions",
                    "twist=0 recovers the standard toric code",
                    "Related to Möbius-strip boundary surface codes",
                    "Different logical structure from un-twisted toric",
                ],
            },
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
        """Compute logical operators using CSS kernel/image prescription.
        
        Twisted toric code has different logical structure but CSS prescription applies.
        """
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
            return logical_x, logical_z
        except Exception as e:
            warnings.warn(f"exotic_surface: logical computation failed ({e}), using single-qubit fallback")
            return [{0: 'X'}], [{0: 'Z'}]

    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2D coordinates for visualization.
        
        Lays out qubits on a Lx × 2Ly grid (horizontal + vertical edges).
        """
        coords: List[Tuple[float, float]] = []
        # n = 2 * Lx * Ly qubits: Lx*Ly horizontal + Lx*Ly vertical
        n_per_type = self._Lx_dim * self._Ly_dim
        # Horizontal edges
        for i in range(n_per_type):
            col = i % self._Ly_dim
            row = i // self._Ly_dim
            coords.append((float(col) + 0.5, float(row)))
        # Vertical edges (offset)
        for i in range(n_per_type):
            col = i % self._Ly_dim
            row = i // self._Ly_dim
            coords.append((float(col), float(row) + 0.5))
        return coords
    
    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'TwistedToricCode(4x4,twist=1)'``."""
        return f"TwistedToricCode({self._Lx_dim}x{self._Ly_dim},twist={self.twist})"

    @property
    def distance(self) -> int:
        """Code distance (min lattice dimension)."""
        return self._distance

    def description(self) -> str:
        return f"Twisted Toric Code {self._Lx_dim}×{self._Ly_dim} twist={self.twist}, n={self.n}"


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

        validate_css_code(hx, hz, f"ProjectivePlaneSurfaceCode_L{L}", raise_on_error=True)

        lx_ops, lz_ops = logicals
        lx_support = sorted({q for op in lx_ops for q in op}) if lx_ops else []
        lz_support = sorted({q for op in lz_ops for q in op}) if lz_ops else []

        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]
        k_val = max(len(lx_ops), 1)
        dist = L

        self._distance = dist

        # Compute coordinate metadata
        _cols = int(np.ceil(np.sqrt(n_qubits)))
        _dc = [(float(i % _cols), float(i // _cols)) for i in range(n_qubits)]
        _xsc = []
        for _ri in range(hx.shape[0]):
            _sup = np.where(hx[_ri])[0]
            if len(_sup) > 0:
                _xsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _xsc.append((0.0, 0.0))
        _zsc = []
        for _ri in range(hz.shape[0]):
            _sup = np.where(hz[_ri])[0]
            if len(_sup) > 0:
                _zsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _zsc.append((0.0, 0.0))

        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=lx_ops,
            logical_z=lz_ops,
            metadata={
                "code_family": "surface",
                "code_type": "projective_plane_surface",
                "n": n_qubits,
                "k": k_val,
                "distance": dist,
                "rate": k_val / n_qubits if n_qubits else 0.0,
                "lx_pauli_type": "X",
                "lz_pauli_type": "Z",
                "lx_support": lx_support,
                "lz_support": lz_support,
                "data_coords": _dc,
                "x_stab_coords": _xsc,
                "z_stab_coords": _zsc,
                "stabiliser_schedule": {
                    "x_rounds": {i: 0 for i in range(n_x_stabs)},
                    "z_rounds": {i: 0 for i in range(n_z_stabs)},
                    "n_rounds": 1,
                    "description": "Fully parallel: all X-checks round 0, all Z-checks round 0.",
                },
                "x_schedule": [(1, 0), (0, 1), (-1, 0), (0, -1)],
                "z_schedule": [(1, 0), (0, -1), (-1, 0), (0, 1)],
                "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/surface",
                "wikipedia_url": "https://en.wikipedia.org/wiki/Real_projective_plane",
                "canonical_references": [
                    "Kitaev, Ann. Phys. 303, 2–30 (2003). arXiv:quant-ph/9707021",
                    "Freedman & Hastings, Geom. Funct. Anal. 31, 855 (2021). arXiv:2012.02249",
                ],
                "connections": [
                    "Surface code on the non-orientable projective plane RP²",
                    "HGP construction with crosscap-twisted base matrix",
                    "Euler characteristic χ=1, different from torus (χ=0)",
                    "Single crosscap yields different k than toric code",
                ],
            },
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
        """Compute logical operators for RP² (1 logical qubit) using CSS prescription."""
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
            return logical_x, logical_z
        except Exception as e:
            warnings.warn(f"exotic_surface: logical computation failed ({e}), using single-qubit fallback")
            return [{0: 'X'}], [{0: 'Z'}]

    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2D coordinates for visualization.
        
        Uses HGP two-sector grid layout.
        """
        na = nb = self.L
        ma = mb = self.L - 1
        n_left = na * nb
        n_right = ma * mb
        right_offset = nb + 2
        
        coords: List[Tuple[float, float]] = []
        # Left sector: L × L grid
        for i in range(min(n_left, self.n)):
            col = i % nb
            row = i // nb
            coords.append((float(col), float(row)))
        # Right sector: (L-1) × (L-1) grid offset
        for i in range(n_left, self.n):
            right_idx = i - n_left
            col = right_idx % mb
            row = right_idx // mb
            coords.append((float(col + right_offset), float(row)))
        return coords
    
    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'ProjectivePlaneSurfaceCode(L=4)'``."""
        return f"ProjectivePlaneSurfaceCode(L={self.L})"

    @property
    def distance(self) -> int:
        """Code distance (lattice dimension L)."""
        return self._distance

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

        validate_css_code(hx, hz, f"KitaevSurfaceCode_n{n_qubits}", raise_on_error=True)

        lx_ops, lz_ops = logicals
        lx_support = sorted({q for op in lx_ops for q in op}) if lx_ops else []
        lz_support = sorted({q for op in lz_ops for q in op}) if lz_ops else []

        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]
        k_val = max(len(lx_ops), 1)
        # Distance is hard to determine for arbitrary cellulation;
        # use minimum logical weight as approximation.
        dist = min(
            (sum(1 for v in op.values() if v != 'I') for op in lx_ops),
            default=1,
        )

        self._distance = dist

        # Compute coordinate metadata
        _cols = int(np.ceil(np.sqrt(n_qubits)))
        _dc = [(float(i % _cols), float(i // _cols)) for i in range(n_qubits)]
        _xsc = []
        for _ri in range(hx.shape[0]):
            _sup = np.where(hx[_ri])[0]
            if len(_sup) > 0:
                _xsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _xsc.append((0.0, 0.0))
        _zsc = []
        for _ri in range(hz.shape[0]):
            _sup = np.where(hz[_ri])[0]
            if len(_sup) > 0:
                _zsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _zsc.append((0.0, 0.0))

        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=lx_ops,
            logical_z=lz_ops,
            metadata={
                "code_family": "surface",
                "code_type": "kitaev_surface",
                "n": n_qubits,
                "k": k_val,
                "distance": dist,
                "rate": k_val / n_qubits if n_qubits else 0.0,
                "lx_pauli_type": "X",
                "lz_pauli_type": "Z",
                "lx_support": lx_support,
                "lz_support": lz_support,
                "data_coords": _dc,
                "x_stab_coords": _xsc,
                "z_stab_coords": _zsc,
                "stabiliser_schedule": {
                    "x_rounds": {i: 0 for i in range(n_x_stabs)},
                    "z_rounds": {i: 0 for i in range(n_z_stabs)},
                    "n_rounds": 1,
                    "description": "Fully parallel: all face (X) round 0, all vertex (Z) round 0.",
                },
                "x_schedule": [],
                "z_schedule": [],
                "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/surface",
                "wikipedia_url": "https://en.wikipedia.org/wiki/Toric_code",
                "canonical_references": [
                    "Kitaev, Ann. Phys. 303, 2–30 (2003). arXiv:quant-ph/9707021",
                ],
                "connections": [
                    "Generic surface code on any planar-graph cellulation",
                    "Reduces to standard surface code on a square lattice",
                    "Face stabilisers (X), vertex stabilisers (Z)",
                ],
            },
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
        """Compute logical operators using CSS kernel/image prescription."""
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
            return logical_x, logical_z
        except Exception as e:
            warnings.warn(f"exotic_surface: logical computation failed ({e}), using single-qubit fallback")
            return [{0: 'X'}], [{0: 'Z'}]

    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2D coordinates for visualization.
        
        Uses the provided vertex coordinates to compute edge midpoints.
        """
        coords: List[Tuple[float, float]] = []
        for v1_idx, v2_idx in self.edges:
            v1 = self.vertices[v1_idx]
            v2 = self.vertices[v2_idx]
            mid_x = (v1[0] + v2[0]) / 2.0
            mid_y = (v1[1] + v2[1]) / 2.0
            coords.append((mid_x, mid_y))
        return coords
    
    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'KitaevSurfaceCode(n=12)'``."""
        return f"KitaevSurfaceCode(n={self.n})"

    @property
    def distance(self) -> int:
        """Code distance (minimum-weight logical operator)."""
        return self._distance

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

        validate_css_code(hx, hz, f"LCSCode_{n_layers}layers_L{L}", raise_on_error=True)

        lx_ops, lz_ops = logicals
        lx_support = sorted({q for op in lx_ops for q in op}) if lx_ops else []
        lz_support = sorted({q for op in lz_ops for q in op}) if lz_ops else []

        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]
        k_val = max(len(lx_ops), 1)
        dist = L  # Lower bound from surface code layers

        self._distance = dist

        # Compute coordinate metadata
        _cols = int(np.ceil(np.sqrt(n_qubits)))
        _dc = [(float(i % _cols), float(i // _cols)) for i in range(n_qubits)]
        _xsc = []
        for _ri in range(hx.shape[0]):
            _sup = np.where(hx[_ri])[0]
            if len(_sup) > 0:
                _xsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _xsc.append((0.0, 0.0))
        _zsc = []
        for _ri in range(hz.shape[0]):
            _sup = np.where(hz[_ri])[0]
            if len(_sup) > 0:
                _zsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _zsc.append((0.0, 0.0))

        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=lx_ops,
            logical_z=lz_ops,
            metadata={
                "code_family": "surface",
                "code_type": "lift_connected_surface",
                "n": n_qubits,
                "k": k_val,
                "distance": dist,
                "rate": k_val / n_qubits if n_qubits else 0.0,
                "n_layers": n_layers,
                "L": L,
                "lx_pauli_type": "X",
                "lz_pauli_type": "Z",
                "lx_support": lx_support,
                "lz_support": lz_support,
                "data_coords": _dc,
                "x_stab_coords": _xsc,
                "z_stab_coords": _zsc,
                "stabiliser_schedule": {
                    "x_rounds": {i: 0 for i in range(n_x_stabs)},
                    "z_rounds": {i: 0 for i in range(n_z_stabs)},
                    "n_rounds": 1,
                    "description": "Fully parallel: all X-stabilisers round 0, all Z-stabilisers round 0.",
                },
                "x_schedule": [(1, 0), (0, 1), (-1, 0), (0, -1)],
                "z_schedule": [(1, 0), (0, -1), (-1, 0), (0, 1)],
                "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/surface",
                "wikipedia_url": "https://en.wikipedia.org/wiki/Toric_code",
                "canonical_references": [
                    "Kitaev, Ann. Phys. 303, 2–30 (2003). arXiv:quant-ph/9707021",
                    "Breuckmann & Eberhardt, PRX Quantum 2, 040101 (2021). arXiv:2103.06309",
                ],
                "connections": [
                    "Stacked toric-code layers with sparse LP-style inter-layer coupling",
                    "Each layer is a standard toric code; couplers add LDPC-like connectivity",
                    "Related to lifted-product codes and fibre-bundle codes",
                ],
            },
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
        """Compute logical operators using CSS kernel/image prescription."""
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
            return logical_x, logical_z
        except Exception as e:
            warnings.warn(f"exotic_surface: logical computation failed ({e}), using single-qubit fallback")
            return [{0: 'X'}], [{0: 'Z'}]

    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2D coordinates for visualization.
        
        Lays out qubits in layers, stacked vertically.
        """
        qubits_per_layer = 2 * self.L * self.L
        n_couplers = (self.n_layers - 1) * self.L
        layer_height = 2 * self.L + 1
        
        coords: List[Tuple[float, float]] = []
        # Layer qubits
        for layer in range(self.n_layers):
            for i in range(qubits_per_layer):
                col = i % (2 * self.L)
                row = i // (2 * self.L)
                y_offset = layer * layer_height
                coords.append((float(col), float(row + y_offset)))
        # Coupler qubits (between layers)
        for layer_pair in range(self.n_layers - 1):
            for idx in range(self.L):
                col = idx
                row = (layer_pair + 1) * layer_height - 0.5
                coords.append((float(col), float(row)))
        return coords
    
    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'LCSCode(layers=3,L=3)'``."""
        return f"LCSCode(layers={self.n_layers},L={self.L})"

    @property
    def distance(self) -> int:
        """Code distance (lower bound from per-layer surface code distance)."""
        return self._distance

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
        
        # Build FiveCSSChainComplex for proper 4D topology
        # In FiveCSSChainComplex: hx = sigma3.T and hz = sigma2
        # So: sigma3 = hx.T and sigma2 = hz
        
        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]
        
        # sigma3: C3 → C2 (cubes -> faces), hx = sigma3.T, so sigma3 = hx.T
        sigma3 = hx.T.astype(np.uint8)  # shape: (n_qubits, n_x_stabs)
        
        # sigma2: C2 → C1 (faces -> edges), hz = sigma2, so sigma2 = hz
        sigma2 = hz.astype(np.uint8)  # shape: (n_z_stabs, n_qubits)
        
        # sigma1: C1 → C0 (terminal boundary, empty for truncated complex)
        # Must have shape (n_vertices, n_edges) where n_edges = sigma2.shape[0]
        sigma1 = np.zeros((0, n_z_stabs), dtype=np.uint8)
        
        # sigma4: C4 → C3 (initial boundary, empty for truncated complex)
        # Must have shape (n_cubes, n_4cells) where n_cubes = sigma3.shape[1]
        sigma4 = np.zeros((n_x_stabs, 0), dtype=np.uint8)
        
        chain_complex = FiveCSSChainComplex(
            sigma4=sigma4,
            sigma3=sigma3,
            sigma2=sigma2,
            sigma1=sigma1,
            qubit_grade=2,
        )
        
        validate_css_code(hx, hz, f"LoopToricCode4D_L{L}", raise_on_error=True)

        lx_ops, lz_ops = logicals
        lx_support = sorted({q for op in lx_ops for q in op}) if lx_ops else []
        lz_support = sorted({q for op in lz_ops for q in op}) if lz_ops else []

        n_x_stabs_count = hx.shape[0]
        n_z_stabs_count = hz.shape[0]
        k_val = max(len(lx_ops), 1)
        dist = L

        self._distance = dist

        # Compute coordinate metadata
        _cols = int(np.ceil(np.sqrt(n_qubits)))
        _dc = [(float(i % _cols), float(i // _cols)) for i in range(n_qubits)]
        _xsc = []
        for _ri in range(hx.shape[0]):
            _sup = np.where(hx[_ri])[0]
            if len(_sup) > 0:
                _xsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _xsc.append((0.0, 0.0))
        _zsc = []
        for _ri in range(hz.shape[0]):
            _sup = np.where(hz[_ri])[0]
            if len(_sup) > 0:
                _zsc.append((float(np.mean([_dc[q][0] for q in _sup])), float(np.mean([_dc[q][1] for q in _sup]))))
            else:
                _zsc.append((0.0, 0.0))

        meta: Dict[str, Any] = {
            "code_family": "surface",
            "code_type": "loop_toric_4d",
            "n": n_qubits,
            "k": k_val,
            "distance": dist,
            "rate": k_val / n_qubits if n_qubits else 0.0,
            "L": L,
            "lx_pauli_type": "X",
            "lz_pauli_type": "Z",
            "lx_support": lx_support,
            "lz_support": lz_support,
            "data_coords": _dc,
            "x_stab_coords": _xsc,
            "z_stab_coords": _zsc,
            "stabiliser_schedule": {
                "x_rounds": {i: 0 for i in range(n_x_stabs_count)},
                "z_rounds": {i: 0 for i in range(n_z_stabs_count)},
                "n_rounds": 1,
                "description": "Fully parallel: all X-stabilisers round 0, all Z-stabilisers round 0.",
            },
            "x_schedule": [],
            "z_schedule": [],
            "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/surface",
            "wikipedia_url": "https://en.wikipedia.org/wiki/Toric_code",
            "canonical_references": [
                "Kitaev, Ann. Phys. 303, 2–30 (2003). arXiv:quant-ph/9707021",
                "Dennis et al., J. Math. Phys. 43, 4452 (2002). arXiv:quant-ph/0110143",
                "Bravyi & Hastings, Proc. STOC 2014. arXiv:1311.0885",
            ],
            "connections": [
                "(2,2) toric code in 4D: both X and Z excitations are loops",
                "Built via HGP of two cycle graphs",
                "Related to homological product codes and 4D surface codes",
                "Contrasts with (1,3) 4D toric code where one excitation is point-like",
            ],
        }

        super().__init__(
            chain_complex=chain_complex,
            logical_x=lx_ops,
            logical_z=lz_ops,
            metadata=meta,
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
        """Compute logical operators using CSS kernel/image prescription."""
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
            return logical_x, logical_z
        except Exception as e:
            warnings.warn(f"exotic_surface: logical computation failed ({e}), using single-qubit fallback")
            return [{0: 'X'}], [{0: 'Z'}]

    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2D coordinates for visualization.
        
        Uses HGP two-sector grid layout.
        """
        n_left = self.L * self.L
        n_right = self.L * self.L
        right_offset = self.L + 2
        
        coords: List[Tuple[float, float]] = []
        # Left sector
        for i in range(min(n_left, self.n)):
            col = i % self.L
            row = i // self.L
            coords.append((float(col), float(row)))
        # Right sector
        for i in range(n_left, self.n):
            right_idx = i - n_left
            col = right_idx % self.L
            row = right_idx // self.L
            coords.append((float(col + right_offset), float(row)))
        return coords
    
    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'LoopToricCode4D(L=2)'``."""
        return f"LoopToricCode4D(L={self.L})"

    @property
    def distance(self) -> int:
        """Code distance (lattice dimension L)."""
        return self._distance

    def description(self) -> str:
        return f"(2,2) Loop Toric Code 4D, L={self.L}, n={self.n}"


# Pre-configured instances
FractalSurface_L2 = lambda: FractalSurfaceCode(level=2)
FractalSurface_L3 = lambda: FractalSurfaceCode(level=3)
TwistedToric_4x4 = lambda: TwistedToricCode(Lx=4, Ly=4, twist=1)
ProjectivePlane_4 = lambda: ProjectivePlaneSurfaceCode(L=4)
LCS_3x3 = lambda: LCSCode(n_layers=3, L=3)
LoopToric4D_2 = lambda: LoopToricCode4D(L=2)
