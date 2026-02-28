"""Toric Code — [[2L², 2, L]] Topological Stabiliser Code on a Torus

The toric code is the foundational topological quantum error-correcting code,
introduced by Kitaev (1997).  It encodes **2 logical qubits** in **2L² physical
qubits** (edges of an L×L square lattice on a torus) with code distance **L**.

Overview
--------
The toric code places data qubits on the **edges** of a square lattice with
*periodic boundary conditions* (i.e. a torus).  Two types of stabilisers
act on these edges:

* **X-type plaquette operators** (faces): the product of X on the 4 edges
  bounding each face of the lattice.
* **Z-type star operators** (vertices): the product of Z on the 4 edges
  meeting at each vertex.

Every plaquette commutes with every star because any pair shares exactly
0 or 2 edges (even overlap → commutation).  The product of *all* plaquettes
is the identity, and similarly for all stars, so one stabiliser of each type
is dependent.  This gives:

    #independent stabilisers = 2(L² − 1) = 2L² − 2
    k = n − #stabilisers = 2L² − (2L² − 2) = 2

Code parameters (L×L torus)
---------------------------
* **n** = 2L² physical qubits (L² horizontal edges + L² vertical edges)
* **k** = 2 logical qubits (two non-contractible cycles on the torus)
* **d** = L (minimum-weight non-contractible loop)
* **Rate** R = 2/(2L²) = 1/L² → 0 as L → ∞

Logical operators
-----------------
The two logical qubits correspond to the two independent non-contractible
cycles on the torus (horizontal and vertical):

* **X̄₁** : X on all horizontal edges in row 0 (horizontal cycle)
* **Z̄₁** : Z on all vertical edges in column 0 (vertical cycle)
* **X̄₂** : X on all vertical edges in column 0 (vertical cycle)
* **Z̄₂** : Z on all horizontal edges in row 0 (horizontal cycle)

Each X̄ᵢ anti-commutes with its paired Z̄ᵢ (odd overlap) and commutes
with the other pair (zero overlap).

Chain complex
-------------
The toric code has a natural **3-term chain complex**:

    C₂ (faces/plaquettes) —∂₂→ C₁ (edges/qubits) —∂₁→ C₀ (vertices)

* H_X = ∂₂ᵀ  (plaquettes → boundary edges)
* H_Z = ∂₁    (vertices → incident edges)
* ∂₂ ∘ ∂₁ = 0 ↔ CSS commutativity
* Logical operators = non-trivial homology classes of H₁(T²; 𝔽₂)

Homological interpretation
--------------------------
The toric code is the prototypical example of a **homological code**.
Errors are 1-chains (sets of edges), syndromes are 0-chains (vertices)
and 2-chains (faces), and decoding is equivalent to finding a minimum-
weight 1-chain with the given boundary.

Connections to other codes
--------------------------
* **Rotated surface code**: obtained by cutting the torus open (removing
  periodic BCs) and rotating 45°, yielding [[d², 1, d]].
* **Hypergraph product**: the toric code is the **HGP of two repetition
  codes**: RepetitionCode ⊗ RepetitionCode → ToricCode.
* **3D/4D toric codes**: higher-dimensional generalisations with
  self-correction and single-shot properties.
* **Colour codes**: the toric code is *not* a colour code (faces are not
  3-colourable), but both are topological CSS codes.

This implementation
-------------------
Currently implements the **L = 3** toric code as a concrete [[18, 2, 3]]
instance.  The lattice size L is not yet parameterised.

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]] = [[2L^2, 2, L]]` where:

- :math:`n = 2L^2` physical qubits (edges of the square lattice on a torus)
- :math:`k = 2` logical qubits (two non-contractible cycles on the torus)
- :math:`d = L` (minimum-weight non-contractible loop)
- Rate :math:`k/n = 1/L^2`

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **X-type stabilisers**: weight-4 plaquette (face) operators; :math:`L^2 - 1`
  independent generators (one global dependency).
- **Z-type stabilisers**: weight-4 star (vertex) operators; :math:`L^2 - 1`
  independent generators (one global dependency).
- Measurement schedule: 4-phase parallel CNOT schedule; all plaquettes
  and all vertices can be measured simultaneously within each phase.

References
----------
* Kitaev, "Fault-tolerant quantum computation by anyons",
  Ann. Phys. 303, 2–30 (2003).  arXiv:quant-ph/9707021
* Dennis, Kitaev, Landahl & Preskill, "Topological quantum memory",
  J. Math. Phys. 43, 4452 (2002).  arXiv:quant-ph/0110143
* Error Correction Zoo: https://errorcorrectionzoo.org/c/surface
* Wikipedia: https://en.wikipedia.org/wiki/Toric_code
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
from qectostim.codes.abstract_code import PauliString, FTGadgetCodeConfig, ScheduleMode
from qectostim.codes.complexes.css_complex import CSSChainComplex3
from qectostim.codes.utils import validate_css_code


class ToricCode33(TopologicalCSSCode):
    r"""[[18, 2, 3]] Toric code on a 3x3 torus.

    18 data qubits (edges), 8 X-type plaquette stabilisers, 8 Z-type
    vertex stabilisers.  Encodes 2 logical qubits with distance 3.

    Lattice layout (L = 3, periodic boundaries)::

        ○ = Z-vertex (star)   □ = X-plaquette (face)
        — = horizontal edge   | = vertical edge

        ○——0——○——1——○——2——○        (row 0: h-edges 0,1,2)
        |      |      |      |
        9  □  10  □  11  □         (v-edges 9,10,11; plaquettes)
        |      |      |      |
        ○——3——○——4——○——5——○        (row 1: h-edges 3,4,5)
        |      |      |      |
       12  □  13  □  14  □         (v-edges 12,13,14; plaquettes)
        |      |      |      |
        ○——6——○——7——○——8——○        (row 2: h-edges 6,7,8)
        |      |      |      |
       15  □  16  □  17  □         (v-edges 15,16,17; plaquettes)
        |      |      |      |
        ○——0——○——1——○——2——○        (wraps to row 0)

    Qubit coordinates:
        h-edge(i,j) → (j+0.5, i)     v-edge(i,j) → (j, i+0.5)
    Stabiliser coords = centroids of support qubits.

    Parameters
    ----------
    metadata : dict, optional
        Extra metadata merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (18).
    k : int
        Number of logical qubits (2).
    distance : int
        Code distance (3).
    hx : np.ndarray
        X-stabiliser parity-check matrix, shape ``(8, 18)``.
    hz : np.ndarray
        Z-stabiliser parity-check matrix, shape ``(8, 18)``.

    Examples
    --------
    >>> code = ToricCode33()
    >>> code.n, code.k, code.distance
    (18, 2, 3)

    Notes
    -----
    The toric code is the first topological code ever proposed (Kitaev 1997)
    and remains the benchmark for topological QEC research.  Its periodic
    boundary conditions make it impractical for physical devices but
    invaluable for theoretical analysis.

    The code is equivalent to the hypergraph product of two [[3,1,3]]
    repetition codes.

    See Also
    --------
    RotatedSurfaceCode : Open-boundary variant for physical devices.
    ToricCode3D : Three-dimensional generalisation.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialize the 3x3 toric code with chain complex structure.
        
        Qubit layout on 3x3 torus:
        - Horizontal edges: qubits 0-8 (h[row,col] at edge between vertices)
        - Vertical edges: qubits 9-17 (v[row,col] at edge between vertices)

        Raises
        ------
        ValueError
            If the lattice size L is less than 2 (fixed at L = 3 in this
            class, so this cannot occur unless subclassed).
        """
        L = 3  # Lattice size
        n_qubits = 2 * L * L  # 18 qubits
        
        # Build lattice geometry and parity check matrices
        (
            data_coords,
            x_stab_coords,
            z_stab_coords,
            hx,
            hz,
            boundary_2,
            boundary_1,
        ) = self._build_toric_lattice(L)
        
        # Create chain complex
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)
        
        # Build logical operators
        logical_x, logical_z = self._build_logicals(L, n_qubits)
        
        # Metadata
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            # ── Code parameters ────────────────────────────────────
            "code_family": "surface",
            "code_type": "toric",
            "name": "ToricCode_3x3",
            "n": n_qubits,
            "k": 2,
            "distance": L,
            "rate": 2.0 / n_qubits,
            "lattice_size": L,
            # ── Geometry ───────────────────────────────────────────
            "data_coords": data_coords,
            "x_stab_coords": x_stab_coords,
            "z_stab_coords": z_stab_coords,
            # ── Logical operator Pauli types ───────────────────────
            "lx_pauli_type": "X",
            "lz_pauli_type": "Z",
            # ── Logical operator supports (k=2: list-of-lists) ────
            "lx_support": [
                list(range(L)),             # Lx₁: horizontal edges in row 0
                list(range(L * L, L * L + L)),  # Lx₂: vertical edges in row 0
            ],
            "lz_support": [
                list(range(L * L, L * L + L)),  # Lz₁: vertical edges in col 0
                list(range(0, L * L, L)),       # Lz₂: horizontal edges in col 0
            ],
            "data_qubits": list(range(n_qubits)),
            "x_logical_coords": [
                [data_coords[i] for i in range(L)],
                [data_coords[i] for i in range(L * L, L * L + L)],
            ],
            "z_logical_coords": [
                [data_coords[i] for i in range(L * L, L * L + L)],
                [data_coords[i] for i in range(0, L * L, L)],
            ],
            # ── Stabiliser scheduling ──────────────────────────────
            "stabiliser_schedule": {
                "x_rounds": {i: 0 for i in range(L * L - 1)},
                "z_rounds": {i: 0 for i in range(L * L - 1)},
                "n_rounds": 1,
                "description": (
                    "Fully parallel: all plaquette (X) stabilisers in "
                    "round 0, all star (Z) stabilisers in round 0."
                ),
            },
            # ── Literature / provenance ────────────────────────────
            "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/toric",
            "wikipedia_url": "https://en.wikipedia.org/wiki/Toric_code",
            "canonical_references": [
                "Kitaev, Ann. Phys. 303, 2–30 (2003). arXiv:quant-ph/9707021",
                "Dennis et al., J. Math. Phys. 43, 4452 (2002). arXiv:quant-ph/0110143",
            ],
            "connections": [
                "HGP of two repetition codes: Rep(L) ⊗ Rep(L) = Toric(L)",
                "Open boundaries → Rotated surface code [[d², 1, d]]",
                "3D generalisation: ToricCode3D with qubits on edges",
            ],
        })
        
        # Measurement schedules for toric code (4-phase schedule)
        meta["x_schedule"] = [
            (0.5, 0.0),    # right horizontal edge
            (-0.5, 0.0),   # left horizontal edge
            (0.0, 0.5),    # bottom vertical edge
            (0.0, -0.5),   # top vertical edge
        ]
        meta["z_schedule"] = [
            (0.5, 0.0),    # right horizontal edge
            (0.0, 0.5),    # down vertical edge
            (-0.5, 0.0),   # left horizontal edge
            (0.0, -0.5),   # up vertical edge
        ]
        
        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, "ToricCode33", raise_on_error=True)

        # Call TopologicalCSSCode constructor
        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)
        
        # Override with explicit X/Z parity check matrices
        self._hx = hx
        self._hz = hz

    @staticmethod
    def _build_toric_lattice(L: int) -> Tuple[
        List[Coord2D],   # data_coords
        List[Coord2D],   # x_stab_coords
        List[Coord2D],   # z_stab_coords
        np.ndarray,      # hx
        np.ndarray,      # hz
        np.ndarray,      # boundary_2
        np.ndarray,      # boundary_1
    ]:
        """Build the L×L toric code lattice, parity-check matrices, and chain complex.

        Constructs the periodic square lattice with:

        * 2L² edges (qubits): L² horizontal + L² vertical.
        * L² faces (plaquettes → X stabilisers, weight 4 each).
        * L² vertices (stars → Z stabilisers, weight 4 each).

        The last plaquette and the last vertex are linearly dependent
        (product of all rows = 0), so the returned ``hx``/``hz`` have
        L²−1 rows each.

        Parameters
        ----------
        L : int
            Linear lattice size.

        Returns
        -------
        data_coords : list of (float, float)
            2D coordinates for each data qubit.
        x_stab_coords, z_stab_coords : list of (float, float)
            Coordinates for X / Z stabiliser ancillas.
        hx, hz : np.ndarray
            Parity-check matrices, shapes ``(L²−1, 2L²)``.
        boundary_2, boundary_1 : np.ndarray
            Chain-complex boundary maps (``∂₂`` and ``∂₁``).
        """
        n_qubits = 2 * L * L
        
        # Edge indexing functions
        def h_edge(row, col):
            return (row % L) * L + (col % L)
        
        def v_edge(row, col):
            return L * L + (row % L) * L + (col % L)
        
        # Data qubit coordinates
        coords = {}
        for i in range(L):
            for j in range(L):
                coords[h_edge(i, j)] = (j + 0.5, float(i))
                coords[v_edge(i, j)] = (float(j), i + 0.5)
        data_coords = [coords.get(i, (0.0, 0.0)) for i in range(n_qubits)]
        
        # X-type stabilizers (plaquette/face operators)
        hx_full = np.zeros((L * L, n_qubits), dtype=np.uint8)
        for i in range(L):
            for j in range(L):
                plaq_idx = i * L + j
                hx_full[plaq_idx, h_edge(i, j)] = 1
                hx_full[plaq_idx, h_edge((i + 1) % L, j)] = 1
                hx_full[plaq_idx, v_edge(i, j)] = 1
                hx_full[plaq_idx, v_edge(i, (j + 1) % L)] = 1
        hx = hx_full[:-1]  # Remove last dependent row
        
        # Z-type stabilizers (vertex/star operators)
        hz_full = np.zeros((L * L, n_qubits), dtype=np.uint8)
        for i in range(L):
            for j in range(L):
                vertex_idx = i * L + j
                hz_full[vertex_idx, h_edge(i, j)] = 1
                hz_full[vertex_idx, h_edge(i, (j - 1) % L)] = 1
                hz_full[vertex_idx, v_edge(i, j)] = 1
                hz_full[vertex_idx, v_edge((i - 1) % L, j)] = 1
        hz = hz_full[:-1]  # Remove last dependent row
        
        # Build chain complex boundary matrices
        # ∂2: shape (n_edges, n_faces) - faces to edges incidence
        # For CSS: H_X = ∂2^T, so ∂2 = H_X^T
        boundary_2 = hx.T
        
        # ∂1: shape (n_vertices, n_edges) - edges to vertices incidence
        # For CSS: H_Z = ∂1, so ∂1 = H_Z
        boundary_1 = hz
        
        # Stabiliser coordinates: use natural lattice face/vertex positions.
        # Plaquette (i, j) sits at face centre (j + 0.5, i + 0.5).
        # Vertex   (i, j) sits at            (j,       i      ).
        #
        # The geometric CX schedule uses (±0.5, 0) / (0, ±0.5) offsets
        # from these positions.  Using centroids of support qubits
        # produces WRONG positions for wrapping stabilisers because the
        # centroid of data qubits at opposite edges of the fundamental
        # domain lands at the centre of the patch, not at the face/vertex.
        #
        # The H-matrix rows are ordered (i=0,j=0), (i=0,j=1), ...,
        # with the last row (i=L-1, j=L-1) removed.  We replicate that
        # ordering here.
        x_stab_all = []
        for i in range(L):
            for j in range(L):
                x_stab_all.append((j + 0.5, i + 0.5))
        x_stab_coords = x_stab_all[:-1]  # match hx (last row removed)

        z_stab_all = []
        for i in range(L):
            for j in range(L):
                z_stab_all.append((float(j), float(i)))
        z_stab_coords = z_stab_all[:-1]  # match hz (last row removed)
        
        return (data_coords, x_stab_coords, z_stab_coords, hx, hz, boundary_2, boundary_1)

    @staticmethod
    def _build_logicals(L: int, n_qubits: int) -> Tuple[List[str], List[str]]:
        """Build two pairs of logical operators for the L×L toric code.

        The torus supports two non-contractible cycles, giving k = 2:

        * **Pair 1**: X̄₁ on horizontal edges in row 0 (horizontal cycle);
          Z̄₁ on vertical edges in column 0 (vertical cycle).
        * **Pair 2**: X̄₂ on vertical edges in row 0;
          Z̄₂ on horizontal edges in column 0.

        Each pair anti-commutes internally (odd overlap = 1) and commutes
        with the other pair (zero overlap).

        Parameters
        ----------
        L : int
            Lattice size.
        n_qubits : int
            Total number of qubits (= 2L²).

        Returns
        -------
        logical_x : list of str
            Two Pauli-string logical X operators.
        logical_z : list of str
            Two Pauli-string logical Z operators.
        """
        def h_edge(row, col):
            return (row % L) * L + (col % L)
        
        def v_edge(row, col):
            return L * L + (row % L) * L + (col % L)
        
        # Logical X1: horizontal string (all horizontal edges in row 0)
        lx1 = ['I'] * n_qubits
        for j in range(L):
            lx1[h_edge(0, j)] = 'X'
        
        # Logical Z1: vertical string (all vertical edges in column 0)
        lz1 = ['I'] * n_qubits
        for i in range(L):
            lz1[v_edge(i, 0)] = 'Z'
        
        # Logical X2: vertical string (all vertical edges in row 0)
        lx2 = ['I'] * n_qubits
        for j in range(L):
            lx2[v_edge(0, j)] = 'X'
        
        # Logical Z2: horizontal string (all horizontal edges in column 0)
        lz2 = ['I'] * n_qubits
        for i in range(L):
            lz2[h_edge(i, 0)] = 'Z'
        
        return [''.join(lx1), ''.join(lx2)], [''.join(lz1), ''.join(lz2)]

    def has_physical_boundaries(self) -> bool:
        """Toric codes have periodic boundaries — no open edges for surgery."""
        return False

    def qubit_coords(self) -> List[Coord2D]:
        """Return 2D coordinates for data qubits."""
        return list(self.metadata["data_coords"])

    @property
    def hx(self) -> np.ndarray:
        """X stabilizers: shape (#X-checks, #data)."""
        return self._hx

    @property
    def hz(self) -> np.ndarray:
        """Z stabilizers: shape (#Z-checks, #data)."""
        return self._hz

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'ToricCode(3x3)'``."""
        return "ToricCode(3x3)"

    @property
    def distance(self) -> int:
        """Code distance."""
        return self.metadata.get("distance", 3)

    def get_ft_gadget_config(self) -> FTGadgetCodeConfig:
        """
        Return FT gadget configuration for 2D toric codes.
        
        Toric codes have periodic boundary conditions which can cause 
        coordinate lookup issues with geometric scheduling. While we've
        added periodic boundary wrapping support, graph coloring is more
        robust for codes with non-trivial topology.
        
        Additionally, toric codes have 2 logical qubits, so gadget
        experiments need to be aware of this.
        """
        return FTGadgetCodeConfig(
            schedule_mode=ScheduleMode.GRAPH_COLORING,  # Robust for periodic BCs
            first_round_x_detectors=True,
            first_round_z_detectors=True,
            enable_metachecks=False,
        )
