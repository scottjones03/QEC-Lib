"""4D Toric Code
=============

The 4D toric code is a topological CSS code defined on a 4-torus
(L × L × L × L with periodic boundary conditions in all four dimensions).
It is the canonical example of a self-correcting quantum memory in
four spatial dimensions.

Construction
------------
The code is built on a regular hypercubic lattice with periodic boundary
conditions along every axis.  Qubits reside on the **2-cells** (faces)
of the cellulation.  There are six face orientations—xy, xz, xw, yz,
yw, zw—each contributing L⁴ faces, for a total of n = 6 L⁴ qubits.

Chain complex
-------------
A proper 4D cellulation requires a **5-chain** (grades 0–4)::

    C₄ (4-cells) ─∂₄─▸ C₃ (3-cells) ─∂₃─▸ C₂ (2-cells)
                                              ─∂₂─▸ C₁ (1-cells) ─∂₁─▸ C₀ (0-cells)

With qubits on grade 2:

* **X stabilisers** = rows of ∂₃ᵀ   (one per 3-cell / cube)
* **Z stabilisers** = rows of ∂₂    (one per 1-cell / edge)
* **X metachecks**  = rows of ∂₄ᵀ   (one per 4-cell / tesseract)
* **Z metachecks**  = rows of ∂₁    (one per 0-cell / vertex)

Code parameters
---------------
For lattice size L:

* n = 6 L⁴  (qubits)
* k = 6     (logical qubits, corresponding to six independent 2-cycles)
* d = L     (code distance, minimum-weight non-trivial 2-cycle)

Self-correction
---------------
The 4D toric code is the textbook example of a self-correcting quantum
memory.  Because both X-type and Z-type excitations are **loop-like**
(1-dimensional), creating a logical error requires nucleating a large
loop whose energy cost scales with L.  The resulting energy barrier
protects quantum information at any temperature below a critical
threshold (Alicki et al., 2010).

Excitation structure
--------------------
* **X excitations** live on edges (violated Z stabilisers) and form
  closed loops in the dual 1-skeleton.
* **Z excitations** live on cubes (violated X stabilisers) and form
  closed loops in the 3-skeleton.
* In both sectors the excitations are 1-dimensional, unlike the 2D
  toric code where excitations are point-like (0-dimensional).

Logical operators
-----------------
Each of the six logical pairs is associated with one face orientation.
The logical-X operator is a planar sheet of X operators spanning two
lattice directions at fixed complementary coordinates; the logical-Z
operator is a complementary sheet of Z operators shifted so that the
two sheets share exactly one face, guaranteeing anti-commutation.

Single-shot error correction
----------------------------
Because the chain complex extends above and below the qubit grade,
both X and Z syndromes possess metachecks (from ∂₄ᵀ and ∂₁
respectively).  This enables **single-shot** error correction: a
single noisy syndrome measurement suffices for reliable decoding
without temporal repetition (Bombín, 2015).

Connections
-----------
* 4D generalisation of the 2D toric code.
* Self-correcting quantum memory at finite temperature.
* Both X and Z excitations are loop-like (unlike 2D where they are
  point-like).
* Related to the 4D surface code via boundary conditions.
* Single-shot error correction via redundant stabilisers (metachecks).

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]] = [[6L^4, 6, L]]` where:

- :math:`n = 6L^4` physical qubits (faces in xy, xz, xw, yz, yw, zw orientations)
- :math:`k = 6` logical qubits (six independent 2-cycles on the 4-torus)
- :math:`d = L` (minimum-weight non-trivial 2-cycle)
- Rate :math:`k/n = 1/L^4`

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **X-type stabilisers**: weight-6 cube (3-cell) operators; :math:`4L^4`
  generators corresponding to 3-cells in four orientations.
- **Z-type stabilisers**: weight-4 edge (1-cell) operators; :math:`4L^4`
  generators corresponding to edges in four directions.
- Measurement schedule: fully parallel — all X-stabilisers in one round,
  all Z-stabilisers in one round.  Meta-checks from :math:`\partial_4^T`
  and :math:`\partial_1` enable single-shot error correction.

References
----------
.. [Dennis2002] Dennis, Kitaev, Landahl & Preskill,
   "Topological quantum memory",
   J. Math. Phys. **43**, 4452 (2002).  arXiv:quant-ph/0110143
.. [Alicki2010] Alicki, Horodecki, Horodecki & Horodecki,
   "On thermal stability of topological qubit in Kitaev's 4D model",
   Open Syst. Inf. Dyn. **17**, 1 (2010).  arXiv:0811.0033
.. [Bombin2015] Bombín, "Single-shot fault-tolerant quantum error
   correction", Phys. Rev. X **5**, 031043 (2015).  arXiv:1404.5504
.. [Bombin2024] Bombín et al., "Experiments with the 4D surface code"
   (2024).
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from itertools import product

from qectostim.codes.abstract_css import TopologicalCSSCode4D, Coord2D
from qectostim.codes.abstract_code import FTGadgetCodeConfig, ScheduleMode
from qectostim.codes.complexes.css_complex import FiveCSSChainComplex
from qectostim.codes.utils import validate_css_code

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
        """Initialise the 4D toric code on an L×L×L×L hyper-torus.

        Parameters
        ----------
        L : int, default 2
            Linear lattice size (must be ≥ 2).
        metadata : dict, optional
            Additional metadata merged into the code's metadata dict.

        Raises
        ------
        ValueError
            If ``L < 2``.
        """
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
        
        # Validate CSS orthogonality before constructing
        validate_css_code(hx, hz, f"ToricCode4D_{L}", raise_on_error=True)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"ToricCode4D_{L}",
            "n": n_qubits,
            "k": 6,
            "distance": L,
            "lattice_size": L,
            "dimension": 4,
            "qubits_on": "2-cells",
            "data_coords": data_coords,
            "x_stab_coords": x_stab_coords,
            "z_stab_coords": z_stab_coords,
            "code_family": "toric",
            "code_type": "toric_4d",
            "rate": 6.0 / n_qubits,
            "lx_pauli_type": "X",
            "lx_support": [i for i, c in enumerate(logical_x[0]) if c == 'X'],
            "lz_pauli_type": "Z",
            "lz_support": [i for i, c in enumerate(logical_z[0]) if c == 'Z'],
            "stabiliser_schedule": {
                "x_rounds": {i: 0 for i in range(hx.shape[0])},
                "z_rounds": {i: 0 for i in range(hz.shape[0])},
                "n_rounds": 1,
                "description": "Fully parallel: all X-stabilisers round 0, all Z-stabilisers round 0.",
            },
            "x_schedule": None,
            "z_schedule": None,
            "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/4d_surface",
            "wikipedia_url": "https://en.wikipedia.org/wiki/Toric_code",
            "canonical_references": [
                "Dennis et al., J. Math. Phys. 43, 4452 (2002). arXiv:quant-ph/0110143",
                "Alicki et al., Open Syst. Inf. Dyn. 17, 1 (2010). arXiv:0811.0033",
            ],
            "connections": [
                "4D generalisation of the 2D toric code",
                "Self-correcting quantum memory at finite temperature",
                "Both X and Z excitations are loop-like (unlike 2D where they are point-like)",
                "Related to the 4D surface code via boundary conditions",
                "Single-shot error correction via redundant stabilisers (metachecks)",
            ],
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
    
    @property
    def name(self) -> str:
        return self._metadata.get("name", f"ToricCode4D_{self._L}")

    @property
    def distance(self) -> int:
        return self._metadata.get("distance", self._L)
    
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
        # sigma2: faces -> edges (n_edges × n_faces), and Hz = sigma2
        # sigma3: cubes -> faces (n_faces × n_cubes), and Hx = sigma3^T
        # sigma4: 4-cells -> cubes (n_cubes × n_4cells)
        
        n_faces = n_qubits
        n_edges = 4 * L**4  # 4 edge directions
        n_cubes = 4 * L**4  # 4 cube orientations (xyz, xyw, xzw, yzw)
        n_4cells = L**4     # Only 1 type of 4-cell orientation
        n_vertices = L**4
        
        # In FiveCSSChainComplex: hx = sigma3.T and hz = sigma2
        # So: sigma3 = hx.T and sigma2 = hz
        
        # sigma3: C3 → C2 (cubes -> faces), shape (n_faces, n_cubes)
        # hx_full has shape (n_cubes, n_faces), so sigma3 = hx_full.T
        sigma3 = hx_full.T  # shape (n_faces, n_cubes) = (96, 64) for L=2
        
        # sigma2: C2 → C1 (faces -> edges), shape (n_edges, n_faces)
        # hz_full has shape (n_edges, n_faces), so sigma2 = hz_full
        sigma2 = hz_full  # shape (n_edges, n_faces) = (64, 96) for L=2
        
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
        """Build logical operators for 4D toric code.
        
        For each face type (e.g., xy), both LX and LZ are defined on that face type:
        - LX spans the face's primary coordinates (i,j for xy) at fixed complementary coords
        - LZ spans the face's complementary coordinates (k,l for xy) at shifted primary coords
        
        The shift ensures exactly one overlap point between LX and LZ, giving
        the required anti-commutation. Each logical operator is a 2-torus of L² faces.
        """
        n_faces_per_orient = L**4
        face_offsets = {'xy': 0, 'xz': 1, 'xw': 2, 'yz': 3, 'yw': 4, 'zw': 5}
        
        def face_idx(orient: str, i: int, j: int, k: int, l: int) -> int:
            base = face_offsets[orient] * n_faces_per_orient
            return base + ((((i % L) * L + (j % L)) * L + (k % L)) * L + (l % L))
        
        logical_x = []
        logical_z = []
        
        # 6 pairs of logical operators, one per face orientation
        # Each pair uses the SAME face type for both X and Z operators
        
        # Pair 0: xy faces
        # LX: xy(i,j,0,0) spanning i,j at fixed k=0,l=0
        # LZ: xy(1,1,k,l) spanning k,l at fixed i=1,j=1
        lx0 = ['I'] * n_qubits
        for i in range(L):
            for j in range(L):
                lx0[face_idx('xy', i, j, 0, 0)] = 'X'
        logical_x.append(''.join(lx0))
        
        lz0 = ['I'] * n_qubits
        for k in range(L):
            for l in range(L):
                lz0[face_idx('xy', 1, 1, k, l)] = 'Z'
        logical_z.append(''.join(lz0))
        
        # Pair 1: xz faces
        # LX: xz(i,0,k,0) spanning i,k at fixed j=0,l=0
        # LZ: xz(1,j,1,l) spanning j,l at fixed i=1,k=1
        lx1 = ['I'] * n_qubits
        for i in range(L):
            for k in range(L):
                lx1[face_idx('xz', i, 0, k, 0)] = 'X'
        logical_x.append(''.join(lx1))
        
        lz1 = ['I'] * n_qubits
        for j in range(L):
            for l in range(L):
                lz1[face_idx('xz', 1, j, 1, l)] = 'Z'
        logical_z.append(''.join(lz1))
        
        # Pair 2: xw faces
        # LX: xw(i,0,0,l) spanning i,l at fixed j=0,k=0
        # LZ: xw(1,j,k,1) spanning j,k at fixed i=1,l=1
        lx2 = ['I'] * n_qubits
        for i in range(L):
            for l in range(L):
                lx2[face_idx('xw', i, 0, 0, l)] = 'X'
        logical_x.append(''.join(lx2))
        
        lz2 = ['I'] * n_qubits
        for j in range(L):
            for k in range(L):
                lz2[face_idx('xw', 1, j, k, 1)] = 'Z'
        logical_z.append(''.join(lz2))
        
        # Pair 3: yz faces
        # LX: yz(0,j,k,0) spanning j,k at fixed i=0,l=0
        # LZ: yz(i,1,1,l) spanning i,l at fixed j=1,k=1
        lx3 = ['I'] * n_qubits
        for j in range(L):
            for k in range(L):
                lx3[face_idx('yz', 0, j, k, 0)] = 'X'
        logical_x.append(''.join(lx3))
        
        lz3 = ['I'] * n_qubits
        for i in range(L):
            for l in range(L):
                lz3[face_idx('yz', i, 1, 1, l)] = 'Z'
        logical_z.append(''.join(lz3))
        
        # Pair 4: yw faces
        # LX: yw(0,j,0,l) spanning j,l at fixed i=0,k=0
        # LZ: yw(i,1,k,1) spanning i,k at fixed j=1,l=1
        lx4 = ['I'] * n_qubits
        for j in range(L):
            for l in range(L):
                lx4[face_idx('yw', 0, j, 0, l)] = 'X'
        logical_x.append(''.join(lx4))
        
        lz4 = ['I'] * n_qubits
        for i in range(L):
            for k in range(L):
                lz4[face_idx('yw', i, 1, k, 1)] = 'Z'
        logical_z.append(''.join(lz4))
        
        # Pair 5: zw faces
        # LX: zw(0,0,k,l) spanning k,l at fixed i=0,j=0
        # LZ: zw(i,j,1,1) spanning i,j at fixed k=1,l=1
        lx5 = ['I'] * n_qubits
        for k in range(L):
            for l in range(L):
                lx5[face_idx('zw', 0, 0, k, l)] = 'X'
        logical_x.append(''.join(lx5))
        
        lz5 = ['I'] * n_qubits
        for i in range(L):
            for j in range(L):
                lz5[face_idx('zw', i, j, 1, 1)] = 'Z'
        logical_z.append(''.join(lz5))
        
        return logical_x, logical_z
    
    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2D coordinates for visualization.
        
        Projects 4D lattice onto 2D, grouping by face orientation.
        Each orientation gets a separate block in a 2x3 grid layout.
        """
        L = self._L
        n_per_orient = L ** 4
        # 6 orientations arranged in 2 rows of 3
        orient_offset = [
            (0, 0),      # xy: top-left
            (L + 1, 0),  # xz: top-middle  
            (2*L + 2, 0), # xw: top-right
            (0, L + 1),  # yz: bottom-left
            (L + 1, L + 1), # yw: bottom-middle
            (2*L + 2, L + 1), # zw: bottom-right
        ]
        
        coords: List[Tuple[float, float]] = []
        for orient_idx in range(6):
            x_off, y_off = orient_offset[orient_idx]
            for i in range(n_per_orient):
                # Map linear index to 2D position within the L x L^3 block
                # Use first two dimensions for 2D projection
                col = i % L
                row = (i // L) % L
                coords.append((float(col + x_off), float(row + y_off)))
        return coords

    def get_ft_gadget_config(self) -> FTGadgetCodeConfig:
        """
        Return FT gadget configuration for 4D toric codes.
        
        4D codes require:
        - GRAPH_COLORING scheduling (4D geometry doesn't map to geometric patterns)
        - Coordinate projection (already handled by qubit_coords)
        - Metachecks for single-shot error correction (5-chain provides meta_x and meta_z)
        """
        return FTGadgetCodeConfig(
            schedule_mode=ScheduleMode.GRAPH_COLORING,
            first_round_x_detectors=True,
            first_round_z_detectors=True,
            enable_metachecks=True,  # 4D codes have full metachecks
            project_coords_to_2d=True,
        )


# Pre-configured instances
ToricCode4D_2 = lambda: ToricCode4D(L=2)
ToricCode4D_3 = lambda: ToricCode4D(L=3)
