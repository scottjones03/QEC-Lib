"""3D Toric Code — [[3L³, 3, L]] Topological Code on a 3-Torus

The 3D toric code generalises Kitaev's 2D toric code to three spatial
dimensions.  Data qubits live on the **edges** (or **faces**) of a cubic
lattice with periodic boundary conditions in all three directions.

Overview
--------
In the **qubits-on-edges** picture:

* **X-stabilisers** are **weight-4 plaquette operators**: each face of
  the cubic lattice contributes an X-type check on its 4 boundary edges.
  There are 3L³ faces (xy, xz, yz), of which 3L³ − 3 are independent.
* **Z-stabilisers** are **weight-6 star operators**: each vertex
  contributes a Z-type check on its 6 incident edges.  There are L³
  vertices, of which L³ − 1 are independent.

Code parameters (qubits on edges, L×L×L torus)
------------------------------------------------
* **n** = 3L³ physical qubits
* **k** = 3 logical qubits (three non-contractible 1-cycles)
* **d** = L

Excitation structure (key difference from 2D)
---------------------------------------------
* **X excitations** (violated plaquettes) are **point-like** (0-dimensional).
* **Z excitations** (violated stars) are **loop-like** (1-dimensional).

This asymmetry means that X errors create point-like anyons that can be
paired by minimum-weight matching (just like the 2D toric code), but Z
errors create loop-like excitations that require more sophisticated decoding.

Self-correction
---------------
In the qubits-on-faces picture (the dual), Z excitations become point-like
and X excitations become loop-like.  The loop-like excitations cost energy
proportional to their length, providing a **finite-temperature self-correcting
memory** for the logical qubits encoded in sheet-like operators.

Single-shot error correction
----------------------------
Because the 3D toric code has a 4-chain complex (C₃ → C₂ → C₁ → C₀),
the stabiliser syndromes themselves have redundancy (meta-checks from ∂₃).
This enables **single-shot** error correction: a noisy syndrome measurement
can be decoded in one round without repetition, because measurement errors
are detectable via the meta-check structure.

Chain complex (4-term for 3D)
-----------------------------
    C₃ (cubes) —∂₃→ C₂ (faces) —∂₂→ C₁ (edges/qubits) —∂₁→ C₀ (vertices)

* H_X = ∂₂ᵀ (faces → boundary edges)
* H_Z = ∂₁   (vertices → incident edges)
* Meta-X-checks from ∂₃ (cubes constrain faces)

Logical operators
-----------------
For qubits on edges:
* **X̄ᵢ** : weight-L string of edges wrapping around direction i
* **Z̄ᵢ** : weight-L² sheet of edges perpendicular to direction i

The string–sheet duality reflects the point–loop duality of excitations.

Connections to other codes
--------------------------
* **2D toric code**: direct lower-dimensional analogue; both are
  CSS codes from cellulations of tori.
* **4D toric code**: further generalisation where both excitation types
  become loop-like, enabling full self-correction.
* **Homological product**: ToricCode ⊗ RepetitionCode yields 3D-like codes.
* **Haah’s cubic code**: another 3D topological code with fractal
  excitations (very different from the toric code's loop-like excitations).

References
----------
* Castelnovo & Chamon, "Topological order in a 3D toric code at finite
  temperature", Phys. Rev. B 78, 155120 (2008).  arXiv:0804.3591
* Dennis, Kitaev, Landahl & Preskill, "Topological quantum memory",
  J. Math. Phys. 43, 4452 (2002).  arXiv:quant-ph/0110143
* Kubica & Preskill, "Cellular-automaton decoders with provable
  thresholds for topological codes", Phys. Rev. Lett. 123, 020501 (2019).
* Error Correction Zoo: https://errorcorrectionzoo.org/c/3d_surface
* Wikipedia: https://en.wikipedia.org/wiki/Toric_code (3D section)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode3D, Coord2D
from qectostim.codes.abstract_code import PauliString, FTGadgetCodeConfig, ScheduleMode
from qectostim.codes.complexes.css_complex import CSSChainComplex4, CSSChainComplex3
from qectostim.codes.utils import validate_css_code

Coord3D = Tuple[float, float, float]


class ToricCode3D(TopologicalCSSCode3D):
    """3D Toric code with qubits on edges.

    For an L×L×L torus:

    * **n** = 3L³ qubits (edges in x, y, z directions)
    * **k** = 3 logical qubits (three non-contractible 1-cycles)
    * **d** = L distance
    * X-stabilisers: weight-4 face/plaquette operators (3L³ − 3 independent)
    * Z-stabilisers: weight-6 vertex/star operators (L³ − 1 independent)

    Parameters
    ----------
    L : int
        Linear size of the cubic lattice (must be ≥ 2, default 3).
    metadata : dict, optional
        Extra metadata merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (3L³).
    k : int
        Number of logical qubits (3).
    distance : int
        Code distance (L).
    hx : np.ndarray
        X-stabiliser parity-check matrix.
    hz : np.ndarray
        Z-stabiliser parity-check matrix.

    Examples
    --------
    >>> code = ToricCode3D(L=3)
    >>> code.n, code.k, code.distance
    (81, 3, 3)

    Notes
    -----
    The 3D toric code is important for its:

    1. **Single-shot decoding**: the 4-chain structure provides meta-checks
       that constrain measurement errors.
    2. **Finite-temperature self-correction** (in the dual/faces picture):
       loop-like excitations cost energy proportional to their length.
    3. **String–sheet duality**: logical X is a string (weight L),
       logical Z is a sheet (weight L²).

    See Also
    --------
    ToricCode33 : 2D toric code (simpler, no meta-checks).
    ToricCode3DFaces : Dual picture with qubits on faces.
    """
    
    def __init__(self, L: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialise the 3D toric code with qubits on edges.

        Builds the periodic cubic lattice, computes the 4-chain complex
        (cubes → faces → edges → vertices), and populates all standard
        metadata fields.

        Parameters
        ----------
        L : int, default 3
            Linear lattice size (must be ≥ 2).
        metadata : dict, optional
            Extra key/value pairs merged into auto-generated metadata.
        """
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
            # ── Code parameters ────────────────────────────────────
            "code_family": "surface",
            "code_type": "toric_3d",
            "name": f"ToricCode3D_{L}x{L}x{L}",
            "n": n_qubits,
            "k": 3,
            "distance": L,
            "rate": 3.0 / n_qubits,
            "lattice_size": L,
            "dimension": 3,
            # ── Geometry ───────────────────────────────────────────
            "data_coords": data_coords,
            "x_stab_coords": x_stab_coords,
            "z_stab_coords": z_stab_coords,
            # ── Logical operator Pauli types ───────────────────────
            "lx_pauli_type": "X",
            "lz_pauli_type": "Z",
            # ── Logical operator supports (k=3: list-of-lists) ────
            # Lx are weight-L strings, Lz are weight-L² sheets
            "lx_support": [
                [((i % L) * L) * L for i in range(L)],         # x-edges at (i,0,0)
                [L**3 + ((0 % L) * L + (j % L)) * L for j in range(L)],  # y-edges at (0,j,0)
                [2 * L**3 + ((0 % L) * L) * L + k for k in range(L)],    # z-edges at (0,0,k)
            ],
            "lz_support": [
                [((0 % L) * L + j) * L + k for j in range(L) for k in range(L)],                          # x-edges i=0
                [L**3 + ((i % L) * L + (0 % L)) * L + k for i in range(L) for k in range(L)],             # y-edges j=0
                [2 * L**3 + ((i % L) * L + j) * L + (0 % L) for i in range(L) for j in range(L)],        # z-edges k=0
            ],
            # ── Stabiliser scheduling ──────────────────────────────
            # 3D codes use graph-colouring-based scheduling rather
            # than offset vectors, because 3D coords don’t map to
            # 2D geometric offset patterns.
            "stabiliser_schedule": {
                "x_rounds": {i: 0 for i in range(3 * L**3 - 3)},
                "z_rounds": {i: 0 for i in range(L**3 - 1)},
                "n_rounds": 1,
                "description": (
                    "Nominally parallel (round 0 for all stabilisers). "
                    "Actual circuit scheduling uses GRAPH_COLORING mode "
                    "to handle 3D geometry."
                ),
            },
            # 3D codes use graph coloring scheduling for proper 3D coord handling
            "requires_3d_support": True,
            # NOTE: x_schedule/z_schedule set to None for 3D codes.
            # 3D geometry uses GRAPH_COLORING mode, not 2D offset vectors.
            "x_schedule": None,
            "z_schedule": None,
            # ── Literature / provenance ────────────────────────────
            "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/3d_surface",
            "wikipedia_url": "https://en.wikipedia.org/wiki/Toric_code",
            "canonical_references": [
                "Castelnovo & Chamon, Phys. Rev. B 78, 155120 (2008). arXiv:0804.3591",
                "Dennis et al., J. Math. Phys. 43, 4452 (2002). arXiv:quant-ph/0110143",
            ],
            "connections": [
                "Generalisation of 2D toric code to 3 spatial dimensions",
                "Single-shot decoding via 4-chain meta-checks",
                "Self-correcting memory (qubits-on-faces dual)",
                "Haah's cubic code: alternative 3D code with fractal excitations",
            ],
        })
        
        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, f"ToricCode3D_{L}x{L}x{L}", raise_on_error=True)

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

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'ToricCode3D(3x3x3)'``."""
        return f"ToricCode3D({self._L}x{self._L}x{self._L})"

    @property
    def distance(self) -> int:
        """Code distance (= L)."""
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
        - Logical Z_i: sheet of edges perpendicular to that direction
        
        The key insight is that L_Z should be a "sheet" of edges that:
        1. Commutes with all X stabilizers (lies in kernel of Hx)
        2. Anti-commutes with exactly its paired L_X (overlap = 1 mod 2)
        
        Correct construction:
        - L_X0: x-edges at (i, 0, 0) for all i → wraps in x-direction
        - L_Z0: x-edges at (0, j, k) for all j,k → sheet in yz-plane
        
        - L_X1: y-edges at (0, j, 0) for all j → wraps in y-direction  
        - L_Z1: y-edges at (i, 0, k) for all i,k → sheet in xz-plane
        
        - L_X2: z-edges at (0, 0, k) for all k → wraps in z-direction
        - L_Z2: z-edges at (i, j, 0) for all i,j → sheet in xy-plane
        """
        def edge_x(i, j, k):
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        def edge_y(i, j, k):
            return L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        def edge_z(i, j, k):
            return 2 * L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        logical_x = []
        logical_z = []
        
        # Logical pair 0: X wraps in x-direction, Z is sheet of x-edges in yz-plane
        # L_X0: string of x-edges at (i, 0, 0) for all i
        lx0 = ['I'] * n_qubits
        for i in range(L):
            lx0[edge_x(i, 0, 0)] = 'X'
        logical_x.append(''.join(lx0))
        
        # L_Z0: sheet of x-edges at i=0 (yz-plane)
        # This has L² x-edges and overlaps L_X0 at exactly edge_x(0,0,0)
        lz0 = ['I'] * n_qubits
        for j in range(L):
            for k in range(L):
                lz0[edge_x(0, j, k)] = 'Z'
        logical_z.append(''.join(lz0))
        
        # Logical pair 1: X wraps in y-direction, Z is sheet of y-edges in xz-plane
        # L_X1: string of y-edges at (0, j, 0) for all j
        lx1 = ['I'] * n_qubits
        for j in range(L):
            lx1[edge_y(0, j, 0)] = 'X'
        logical_x.append(''.join(lx1))
        
        # L_Z1: sheet of y-edges at j=0 (xz-plane)
        # This has L² y-edges and overlaps L_X1 at exactly edge_y(0,0,0)
        lz1 = ['I'] * n_qubits
        for i in range(L):
            for k in range(L):
                lz1[edge_y(i, 0, k)] = 'Z'
        logical_z.append(''.join(lz1))
        
        # Logical pair 2: X wraps in z-direction, Z is sheet of z-edges in xy-plane
        # L_X2: string of z-edges at (0, 0, k) for all k
        lx2 = ['I'] * n_qubits
        for k in range(L):
            lx2[edge_z(0, 0, k)] = 'X'
        logical_x.append(''.join(lx2))
        
        # L_Z2: sheet of z-edges at k=0 (xy-plane)
        # This has L² z-edges and overlaps L_X2 at exactly edge_z(0,0,0)
        lz2 = ['I'] * n_qubits
        for i in range(L):
            for j in range(L):
                lz2[edge_z(i, j, 0)] = 'Z'
        logical_z.append(''.join(lz2))
        
        return logical_x, logical_z
    
    def qubit_coords(self) -> List[Coord3D]:
        """Return 3D coordinates of data qubits."""
        return self._metadata.get("data_coords", [])

    def get_ft_gadget_config(self) -> FTGadgetCodeConfig:
        """
        Return FT gadget configuration for 3D toric codes.
        
        3D toric codes require:
        - GRAPH_COLORING scheduling (3D coords don't map to 2D geometric patterns)
        - Coordinate projection for Stim's 2D detector visualization
        - Potentially metachecks for single-shot error correction
        """
        return FTGadgetCodeConfig(
            schedule_mode=ScheduleMode.GRAPH_COLORING,  # 3D needs graph coloring
            first_round_x_detectors=True,
            first_round_z_detectors=True,
            enable_metachecks=True,  # 3D codes have metachecks (4-chain)
            project_coords_to_2d=True,  # Need to project 3D coords for Stim
        )

    def project_coords_to_2d(
        self,
        coords: List[Tuple[float, ...]],
    ) -> List[Tuple[float, float]]:
        """
        Project 3D coordinates to 2D for Stim visualization.
        
        Uses an isometric projection: x' = x - y, y' = z - (x+y)/2
        This preserves distinguishability of different 3D positions.
        """
        result = []
        for c in coords:
            if len(c) >= 3:
                x, y, z = c[0], c[1], c[2]
                # Isometric projection
                x2d = x - y
                y2d = z - (x + y) / 2
                result.append((float(x2d), float(y2d)))
            elif len(c) >= 2:
                result.append((float(c[0]), float(c[1])))
            elif len(c) == 1:
                result.append((float(c[0]), 0.0))
            else:
                result.append((0.0, 0.0))
        return result


class ToricCode3DFaces(TopologicalCSSCode3D):
    """3D Toric code with qubits on faces (2-cells).

    This is the **dual picture** of the standard 3D toric code:

    * X-stabilisers are **cube operators** (weight 6, one per 3-cell).
    * Z-stabilisers are **edge operators** (weight 4, one per 1-cell).

    For an L×L×L torus:

    * **n** = 3L³ (faces in xy, xz, yz orientations)
    * **k** = 3
    * **d** = L

    Excitation structure:

    * **Z excitations** (violated edges) are **point-like** (0-D).
    * **X excitations** (violated cubes) are **loop-like** (1-D).

    This makes the Z-sector decodable by MWPM and gives a
    **finite-temperature self-correcting memory** because the loop-like X
    excitations cost energy proportional to their length.

    The 4-chain complex ``C₃ → C₂ → C₁ → C₀`` (cubes → faces → edges →
    vertices) provides **meta-checks** (∂₁: vertices constrain
    edge-Z syndromes) enabling single-shot error correction.

    Parameters
    ----------
    L : int
        Linear size of the cubic lattice (must be ≥ 2, default 3).
    metadata : dict, optional
        Extra metadata merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (3L³).
    k : int
        Number of logical qubits (3).
    distance : int
        Code distance (L).
    hx : np.ndarray
        X-stabiliser parity-check matrix (cube operators).
    hz : np.ndarray
        Z-stabiliser parity-check matrix (edge operators).

    Examples
    --------
    >>> code = ToricCode3DFaces(L=2)
    >>> code.n, code.k, code.distance
    (24, 3, 2)
    >>> code = ToricCode3DFaces(L=3)
    >>> code.n, code.k, code.distance
    (81, 3, 3)

    Notes
    -----
    Uses ``CSSChainComplex4`` (4-term chain complex) for the full
    ``C₃ → C₂ → C₁ → C₀`` structure.  The meta-checks from ∂₁ enable
    **single-shot** error correction of Z-type syndromes.

    See Also
    --------
    ToricCode3D : Dual picture with qubits on edges.
    ToricCode33 : 2D toric code (no meta-checks).
    """

    def __init__(self, L: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialise the 3D toric code with qubits on faces.

        Builds the cubic lattice on a 3-torus, computes the 4-chain
        complex (cubes → faces → edges → vertices), and derives the
        X-stabiliser (cube) and Z-stabiliser (edge) parity-check matrices.

        Parameters
        ----------
        L : int, default 3
            Linear lattice size (must be ≥ 2).
        metadata : dict, optional
            Extra key/value pairs merged into auto-generated metadata.
        """
        if L < 2:
            raise ValueError("L must be at least 2")

        self._L = L
        n_qubits = 3 * L**3  # faces

        # Build lattice — returns the three boundary maps for the 4-chain
        (
            data_coords,
            x_stab_coords,
            z_stab_coords,
            hx,
            hz,
            boundary_3,   # cubes → faces  (n_faces × n_cubes)
            boundary_2,   # faces → edges  (n_edges × n_faces)
            boundary_1,   # edges → vertices (n_vertices × n_edges)
        ) = self._build_lattice(L)

        # 4-chain complex: C₃ → C₂ → C₁ → C₀, qubits on grade 2 (faces)
        chain_complex = CSSChainComplex4(
            boundary_3=boundary_3,
            boundary_2=boundary_2,
            boundary_1=boundary_1,
            qubit_grade=2,  # qubits live on faces
        )
        logical_x, logical_z = self._build_logicals(L, n_qubits)

        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            # ── Code parameters ────────────────────────────────────
            "code_family": "surface",
            "code_type": "toric_3d_faces",
            "name": f"ToricCode3DFaces_{L}x{L}x{L}",
            "n": n_qubits,
            "k": 3,
            "distance": L,
            "rate": 3.0 / n_qubits,
            "lattice_size": L,
            "dimension": 3,
            "qubits_on": "faces",
            # ── Geometry ───────────────────────────────────────────
            "data_coords": data_coords,
            "x_stab_coords": x_stab_coords,
            "z_stab_coords": z_stab_coords,
            # ── Logical operator Pauli types ───────────────────────
            "lx_pauli_type": "X",
            "lz_pauli_type": "Z",
            # ── Logical operator supports (k=3: list-of-lists) ────
            # Lx are weight-L² membranes, Lz are weight-L strings
            "lx_support": [
                [((i % L) * L + (j % L)) * L for i in range(L) for j in range(L)],              # xy-faces at z=0
                [L**3 + ((i % L) * L) * L + k for i in range(L) for k in range(L)],             # xz-faces at y=0
                [2 * L**3 + ((0 % L) * L + j) * L + k for j in range(L) for k in range(L)],    # yz-faces at x=0
            ],
            "lz_support": [
                [k for k in range(L)],                                  # xy-faces at (0,0,k)
                [L**3 + j * L**2 for j in range(L)],                   # xz-faces at (0,j,0)
                [2 * L**3 + i * L**2 for i in range(L)],               # yz-faces at (i,0,0)
            ],
            # ── Stabiliser scheduling ──────────────────────────────
            "stabiliser_schedule": {
                "x_rounds": {i: 0 for i in range(L**3 - 1)},
                "z_rounds": {i: 0 for i in range(3 * L**3 - 3)},
                "n_rounds": 1,
                "description": (
                    "Nominally parallel (round 0). 3D geometry uses "
                    "GRAPH_COLORING scheduling mode."
                ),
            },
            # NOTE: x_schedule/z_schedule set to None for 3D codes.
            "x_schedule": None,
            "z_schedule": None,
            # 3D codes use graph coloring scheduling for proper 3D coord handling
            "requires_3d_support": True,
            # ── Literature / provenance ────────────────────────────
            "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/3d_surface",
            "wikipedia_url": "https://en.wikipedia.org/wiki/Toric_code",
            "canonical_references": [
                "Castelnovo & Chamon, Phys. Rev. B 78, 155120 (2008). arXiv:0804.3591",
                "Dennis et al., J. Math. Phys. 43, 4452 (2002). arXiv:quant-ph/0110143",
            ],
            "connections": [
                "Dual of ToricCode3D (qubits on edges → qubits on faces)",
                "Self-correcting memory at finite temperature",
                "Point-like Z excitations, loop-like X excitations",
            ],
        })

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, f"ToricCode3DFaces_{L}x{L}x{L}", raise_on_error=True)

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
        """Build lattice with qubits on faces and full 4-chain boundaries.

        Returns the three boundary maps for the 4-chain complex
        ``C₃ (cubes) → C₂ (faces) → C₁ (edges) → C₀ (vertices)``.

        Parameters
        ----------
        L : int
            Linear lattice size.

        Returns
        -------
        data_coords : list of Coord3D
            Face-centre coordinates for data qubits.
        x_stab_coords : list of Coord3D
            Cube-centre coordinates (X-stabiliser ancillas).
        z_stab_coords : list of Coord3D
            Edge-midpoint coordinates (Z-stabiliser ancillas).
        hx, hz : np.ndarray
            Parity-check matrices (dependent rows removed).
        boundary_3 : np.ndarray
            ∂₃ map (n_faces × n_cubes): cube → face incidence.
        boundary_2 : np.ndarray
            ∂₂ map (n_edges × n_faces): face → edge incidence.
        boundary_1 : np.ndarray
            ∂₁ map (n_vertices × n_edges): edge → vertex incidence.
        """
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

        # ── 4-chain boundary maps (full, no dependent-row removal) ─────
        # ∂₃: cubes → faces.  hx_full has shape (n_cubes, n_faces), so
        #     ∂₃ = hx_full.T  →  shape (n_faces, n_cubes)
        boundary_3 = hx_full.T.astype(np.uint8)

        # ∂₂: faces → edges.  hz_full has shape (n_edges, n_faces), so
        #     ∂₂ = hz_full.T  →  shape (n_faces … wait, hz_full rows are
        #     edge-stabilisers acting on faces).  Each row is an edge's
        #     4 incident faces.  So hz_full.T[face, edge] = incidence,
        #     which is the transpose of what we want.
        #     Actually ∂₂ maps faces→edges: ∂₂[edge, face] = 1 iff face
        #     has that edge on its boundary.  That is exactly hz_full,
        #     transposed from the stabiliser convention.
        #     hz_full shape = (n_edges, n_faces), and ∂₂ should have shape
        #     (n_edges, n_faces) — mapping grade-2 (faces) to grade-1 (edges).
        boundary_2 = hz_full.astype(np.uint8)

        # ∂₁: edges → vertices.  Each edge connects two vertices.
        n_vertices = L**3

        def edge_x(i, j, k):
            return ((i % L) * L + (j % L)) * L + (k % L)

        def edge_y(i, j, k):
            return L**3 + ((i % L) * L + (j % L)) * L + (k % L)

        def edge_z(i, j, k):
            return 2 * L**3 + ((i % L) * L + (j % L)) * L + (k % L)

        def vertex_idx(i, j, k):
            return ((i % L) * L + (j % L)) * L + (k % L)

        boundary_1 = np.zeros((n_vertices, n_edges), dtype=np.uint8)
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    # x-edge at (i,j,k) connects vertices (i,j,k) and (i+1,j,k)
                    boundary_1[vertex_idx(i, j, k), edge_x(i, j, k)] = 1
                    boundary_1[vertex_idx((i + 1) % L, j, k), edge_x(i, j, k)] = 1
                    # y-edge at (i,j,k) connects vertices (i,j,k) and (i,j+1,k)
                    boundary_1[vertex_idx(i, j, k), edge_y(i, j, k)] = 1
                    boundary_1[vertex_idx(i, (j + 1) % L, k), edge_y(i, j, k)] = 1
                    # z-edge at (i,j,k) connects vertices (i,j,k) and (i,j,k+1)
                    boundary_1[vertex_idx(i, j, k), edge_z(i, j, k)] = 1
                    boundary_1[vertex_idx(i, j, (k + 1) % L), edge_z(i, j, k)] = 1

        return (data_coords, x_stab_coords, z_stab_coords, hx, hz,
                boundary_3, boundary_2, boundary_1)
    
    @staticmethod
    def _build_logicals(L: int, n_qubits: int) -> Tuple[List[str], List[str]]:
        """Build logical operators for 3D toric code with qubits on faces.
        
        For 3D toric code with qubits on faces (2-cells):
        - Logical X_i: membrane (2-chain) wrapping around direction i
        - Logical Z_i: string of faces perpendicular to the membrane
        
        The key insight is that L_Z should be a line of faces of a single type
        that commutes with all X stabilizers and anti-commutes with its paired L_X.
        
        Correct construction:
        - L_X0: xy-faces at z=0 (all i,j) → membrane in xy-plane
        - L_Z0: xy-faces at i=0,j=0 (all k) → string in z-direction
        
        - L_X1: xz-faces at y=0 (all i,k) → membrane in xz-plane
        - L_Z1: xz-faces at i=0,k=0 (all j) → string in y-direction
        
        - L_X2: yz-faces at x=0 (all j,k) → membrane in yz-plane
        - L_Z2: yz-faces at j=0,k=0 (all i) → string in x-direction
        """
        def face_xy(i, j, k):
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        def face_xz(i, j, k):
            return L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        def face_yz(i, j, k):
            return 2 * L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        logical_x = []
        logical_z = []
        
        # Logical pair 0: X is xy-membrane at z=0, Z is xy-string at (0,0,k)
        # L_X0: all xy-faces at z=0
        lx0 = ['I'] * n_qubits
        for i in range(L):
            for j in range(L):
                lx0[face_xy(i, j, 0)] = 'X'
        logical_x.append(''.join(lx0))
        
        # L_Z0: xy-faces at (0,0,k) for all k - string in z-direction
        # Overlaps L_X0 at face_xy(0,0,0)
        lz0 = ['I'] * n_qubits
        for k in range(L):
            lz0[face_xy(0, 0, k)] = 'Z'
        logical_z.append(''.join(lz0))
        
        # Logical pair 1: X is xz-membrane at y=0, Z is xz-string at (0,j,0)
        # L_X1: all xz-faces at y=0
        lx1 = ['I'] * n_qubits
        for i in range(L):
            for k in range(L):
                lx1[face_xz(i, 0, k)] = 'X'
        logical_x.append(''.join(lx1))
        
        # L_Z1: xz-faces at (0,j,0) for all j - string in y-direction
        # Overlaps L_X1 at face_xz(0,0,0)
        lz1 = ['I'] * n_qubits
        for j in range(L):
            lz1[face_xz(0, j, 0)] = 'Z'
        logical_z.append(''.join(lz1))
        
        # Logical pair 2: X is yz-membrane at x=0, Z is yz-string at (i,0,0)
        # L_X2: all yz-faces at x=0
        lx2 = ['I'] * n_qubits
        for j in range(L):
            for k in range(L):
                lx2[face_yz(0, j, k)] = 'X'
        logical_x.append(''.join(lx2))
        
        # L_Z2: yz-faces at (i,0,0) for all i - string in x-direction
        # Overlaps L_X2 at face_yz(0,0,0)
        lz2 = ['I'] * n_qubits
        for i in range(L):
            lz2[face_yz(i, 0, 0)] = 'Z'
        logical_z.append(''.join(lz2))
        
        return logical_x, logical_z

    def qubit_coords(self) -> List[Coord3D]:
        """Return 3D coordinates of data qubits (face centers)."""
        return self._metadata.get("data_coords", [])

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'ToricCode3DFaces(3x3x3)'``."""
        return f"ToricCode3DFaces({self._L}x{self._L}x{self._L})"

    @property
    def distance(self) -> int:
        """Code distance (= L)."""
        return self._L

    def get_ft_gadget_config(self) -> FTGadgetCodeConfig:
        """
        Return FT gadget configuration for 3D toric code with qubits on faces.
        """
        return FTGadgetCodeConfig(
            schedule_mode=ScheduleMode.GRAPH_COLORING,
            first_round_x_detectors=True,
            first_round_z_detectors=True,
            enable_metachecks=True,  # 3D codes have metachecks
            project_coords_to_2d=True,
        )

    def project_coords_to_2d(
        self,
        coords: List[Tuple[float, ...]],
    ) -> List[Tuple[float, float]]:
        """Project 3D coordinates to 2D using isometric projection."""
        result = []
        for c in coords:
            if len(c) >= 3:
                x, y, z = c[0], c[1], c[2]
                x2d = x - y
                y2d = z - (x + y) / 2
                result.append((float(x2d), float(y2d)))
            elif len(c) >= 2:
                result.append((float(c[0]), float(c[1])))
            elif len(c) == 1:
                result.append((float(c[0]), 0.0))
            else:
                result.append((0.0, 0.0))
        return result


# Pre-configured instances
ToricCode3D_3x3x3 = lambda: ToricCode3D(L=3)
ToricCode3D_4x4x4 = lambda: ToricCode3D(L=4)
