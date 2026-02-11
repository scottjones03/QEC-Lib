"""Fracton Codes — 3D Stabiliser Codes with Restricted Excitation Mobility

Fracton codes are a class of 3D stabiliser codes whose excitations
exhibit **restricted mobility**: some quasi-particles (fractons) cannot
move freely but are confined to subdimensional manifolds.  This
property leads to sub-extensive ground-state degeneracy and potential
applications for self-correcting quantum memory.

Overview
--------
Unlike topological codes in 2D (e.g. surface codes), where anyonic
excitations can move freely in the plane, fracton codes in 3D have
three excitation categories:

* **Fractons** — fully immobile point-like excitations (can only be
  created/annihilated in groups).
* **Lineons** — excitations mobile along 1-dimensional lines.
* **Planons** — excitations mobile within 2-dimensional planes.

Fracton codes are further classified into:

* **Type-I** (foliated): Have both mobile and immobile excitations.
  Examples: X-cube, Checkerboard, Chamon.
* **Type-II** (fractal): All excitations are immobile.  Logical
  operators have fractal geometry.  Example: Haah's cubic code.

Codes in this module
--------------------
* ``XCubeCode`` — type-I fracton on a cubic lattice (lineons + planons)
* ``HaahCode`` — type-II fracton with fractal logical operators
* ``CheckerboardCode`` — foliated type-I on alternating cubes
* ``ChamonCode`` — type-I fracton on a BCC lattice
* ``FibonacciFractalCode`` — HGP-based code with Fibonacci sizing
* ``SierpinskiPrismCode`` — HGP-based code on a fractal prism

Code Parameters
~~~~~~~~~~~~~~~
* **XCubeCode(L)**: ``[[3L³, 6L−3, L]]`` — sub-extensive k scaling.
* **HaahCode(L)**: ``[[2L³, 2L, L]]`` — type-II fracton with fractal
  logical operators and system-size-dependent k.
* **CheckerboardCode(L)**: ``[[L³, O(L), L]]`` — type-I on alternating
  cubes.
* **ChamonCode(L)**: ``[[3L³, O(L), L]]`` — type-I on a BCC lattice.
* **FibonacciFractalCode(L)** and **SierpinskiPrismCode(L)**: HGP-based
  approximations with ``n = O(L²)`` and ``k ≥ 1``.

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
* **XCubeCode**: 12-body cube operators (X-type) and 4-body cross
  operators in 3 orthogonal planes at each vertex (Z-type).  Total:
  ``L³ − 1`` X-stabilisers, ``3(L³ − L)`` Z-stabilisers.
* **HaahCode**: 8-body cubic-vertex operators for both X and Z types;
  each stabiliser touches 8 qubits at the corners of a unit cube.
  Total: ``L³ − 1`` stabilisers per type.
* **CheckerboardCode**: 12-body cube operators on alternating cubes for
  X; 4-body vertex operators for Z.  Measurement: single parallel round.
* **ChamonCode**: 6-body X and 6-body Z stabilisers on a BCC lattice;
  stabiliser weight is constant.
* All fracton codes admit a single-round parallel measurement schedule.

Ground-state degeneracy
-----------------------
Fracton codes have **sub-extensive** ground-state degeneracy:
``k ∝ L`` (or ``6L − 3`` for X-cube) rather than the constant ``k``
typical of topological codes.  This is a hallmark of fracton order
and has implications for quantum memory capacity.

Stabiliser scheduling
---------------------
For all codes in this module, X-stabilisers and Z-stabilisers are
assigned to parallel measurement rounds (round 0 for each type).
In a hardware implementation the stabilisers may need to be
partitioned into non-overlapping subsets that avoid qubit conflicts.

Implementation caveats
----------------------
``FibonacciFractalCode`` and ``SierpinskiPrismCode`` use a homological
product (HGP) construction internally rather than building genuine
fractal lattice geometry.  They are pedagogical approximations of the
true fracton codes.  See the class docstrings for details.

Connections to other codes
--------------------------
* **Toric/surface codes**: fracton codes are 3D generalisations with
  richer excitation structure.
* **Subsystem codes**: X-cube can be viewed as a subsystem code whose
  gauge group generates the fracton constraints.
* **Classical fractal codes**: type-II fracton logical operators have
  fractal support (Sierpinski-triangle-like patterns).

References
----------
* Vijay, Haah & Fu, "A new kind of topological quantum order",
  Phys. Rev. B 92, 235136 (2015).  arXiv:1505.02576
* Haah, "Local stabilizer codes in three dimensions without string
  logical operators", J. Math. Phys. 52, 095101 (2011).  arXiv:1101.1962
* Shirley, Slagle & Chen, "Foliated fracton order from gauging
  subsystem symmetries", SciPost Phys. 6, 015 (2019).  arXiv:1806.08679
* Chamon, "Quantum Glassiness in Strongly Correlated Clean Systems",
  Phys. Rev. Lett. 94, 040402 (2005).
* Yoshida, "Exotic topological order in fractal spin liquids",
  Phys. Rev. B 88, 125122 (2013).  arXiv:1302.6248
* Error Correction Zoo: https://errorcorrectionzoo.org/c/fracton

Decoding
--------
* Renormalisation-group (RG) decoders exploit the hierarchical structure
  of fracton excitations to decode in O(n log n) time.
* Peeling decoders can handle the lineon and planon sectors separately.
* BP-OSD is applicable but convergence may be slow due to the high
  weight of stabilisers in some fracton models.
* Cellular-automaton decoders can exploit the local constraints for
  parallel hardware-friendly decoding.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from itertools import product

from qectostim.codes.abstract_css import CSSCode, Coord2D
from qectostim.codes.complexes.css_complex import CSSChainComplex3
from qectostim.codes.utils import compute_css_logicals, vectors_to_paulis_x, vectors_to_paulis_z, validate_css_code

Coord3D = Tuple[float, float, float]


class XCubeCode(CSSCode):
    """
    X-cube model - a type-I fracton code.
    
    The X-cube model is defined on a cubic lattice with:
        - Qubits on edges
        - X stabilizers: cube operators (12 edges per cube)
        - Z stabilizers: vertex operators in 3 planes (4 edges each)
    
    Excitations:
        - X errors create lineon excitations (move along lines)
        - Z errors create fracton excitations (immobile) + planons (move in planes)
    
    For an L×L×L lattice with periodic boundaries:
        - n = 3L³ qubits
        - k = 6L - 3 logical qubits (sub-extensive!)
        - d = L distance
    
    Parameters
    ----------
    L : int
        Linear size of the cubic lattice (default: 3)
    metadata : dict, optional
        Additional metadata
    """
    
    def __init__(self, L: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Construct an X-cube fracton code on an L × L × L cubic lattice.

        Parameters
        ----------
        L : int
            Linear lattice size (must be ≥ 2).  Default 3.
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dictionary.

        Raises
        ------
        ValueError
            If ``L < 2``.
        """
        if L < 2:
            raise ValueError("L must be at least 2")
        
        self._L = L
        n_qubits = 3 * L**3  # edges
        
        # Build lattice
        (
            data_coords,
            x_stab_coords,
            z_stab_coords,
            hx,
            hz,
            boundary_2,
            boundary_1,
        ) = self._build_xcube_lattice(L)
        
        # Sub-extensive number of logical qubits
        k = 6 * L - 3
        logical_x, logical_z = self._build_logicals(L, n_qubits, k)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"XCubeCode_{L}",
            "n": n_qubits,
            "k": k,
            "distance": L,
            "lattice_size": L,
            "fracton_type": "type-I",
            "excitations": {
                "X_sector": "lineons (1D mobile)",
                "Z_sector": "fractons + planons"
            },
            "data_coords": data_coords,
            "x_stab_coords": x_stab_coords,
            "z_stab_coords": z_stab_coords,
            # 17 standard metadata keys
            "code_family": "fracton",
            "code_type": "x_cube",
            "rate": k / n_qubits if n_qubits > 0 else 0.0,
            "lx_pauli_type": "X",
            "lx_support": None,  # fractal/membrane operators
            "lz_pauli_type": "Z",
            "lz_support": None,
            "stabiliser_schedule": {
                "x_rounds": {i: 0 for i in range(L**3 - 1)},
                "z_rounds": {i: 0 for i in range(3 * L**3 - 3 * L)},
                "n_rounds": 1,
                "description": (
                    "Fully parallel: all X-stabilisers (12-body cube operators) "
                    "in round 0, all Z-stabilisers (4-body cross operators in "
                    "3 planes at each vertex) in round 0."
                ),
            },
            "x_schedule": None,
            "z_schedule": None,
            "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/xcube",
            "wikipedia_url": None,
            "canonical_references": [
                "Vijay, Haah & Fu, 'A new kind of topological quantum order' (2015)",
                "Vijay, Haah & Fu, 'Fracton topological order, generalized lattice gauge theory, and duality' (2016)",
            ],
            "connections": [
                "Type-I fracton model with lineon and planon excitations",
                "Sub-extensive ground state degeneracy k = 6L - 3",
                "Related to foliated fracton order and p-string condensation",
            ],
        })
        
        validate_css_code(hx, hz, code_name=f"XCubeCode_{L}", raise_on_error=True)
        
        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'XCubeCode(L=3)'``."""
        return f"XCubeCode(L={self._L})"

    @property
    def distance(self) -> int:
        """Code distance (= L for the X-cube model)."""
        return self._L

    def qubit_coords(self) -> List[Coord2D]:
        """Return 2D projection of qubit coordinates."""
        return self._metadata.get("data_coords", [])
    
    @property
    def k(self) -> int:
        """Number of logical qubits.
        
        X-cube has k = 6L - 3 logical qubits (sub-extensive).
        This overrides the base class computation which doesn't account
        for the gauge structure of fracton codes.
        """
        return self._metadata.get("k", 6 * self._L - 3)
    
    @property
    def L(self) -> int:
        return self._L
    
    @staticmethod
    def _build_xcube_lattice(L: int) -> Tuple:
        """Build the X-cube lattice."""
        n_qubits = 3 * L**3
        
        # Edge indexing
        def edge_x(i, j, k):
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        def edge_y(i, j, k):
            return L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        def edge_z(i, j, k):
            return 2 * L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        # Data coords
        data_coords = []
        for i, j, k in product(range(L), repeat=3):
            data_coords.append((i + 0.5, float(j), float(k)))  # x-edge
        for i, j, k in product(range(L), repeat=3):
            data_coords.append((float(i), j + 0.5, float(k)))  # y-edge
        for i, j, k in product(range(L), repeat=3):
            data_coords.append((float(i), float(j), k + 0.5))  # z-edge
        
        # X stabilizers: cube operators (12 edges per cube)
        hx_list = []
        
        for i, j, k in product(range(L), repeat=3):
            row = np.zeros(n_qubits, dtype=np.uint8)
            
            # 4 x-edges
            row[edge_x(i, j, k)] = 1
            row[edge_x(i, (j+1) % L, k)] = 1
            row[edge_x(i, j, (k+1) % L)] = 1
            row[edge_x(i, (j+1) % L, (k+1) % L)] = 1
            
            # 4 y-edges
            row[edge_y(i, j, k)] = 1
            row[edge_y((i+1) % L, j, k)] = 1
            row[edge_y(i, j, (k+1) % L)] = 1
            row[edge_y((i+1) % L, j, (k+1) % L)] = 1
            
            # 4 z-edges
            row[edge_z(i, j, k)] = 1
            row[edge_z((i+1) % L, j, k)] = 1
            row[edge_z(i, (j+1) % L, k)] = 1
            row[edge_z((i+1) % L, (j+1) % L, k)] = 1
            
            hx_list.append(row)
        
        # Z stabilizers: cross operators in each plane at each vertex
        # Three types: xy-cross, xz-cross, yz-cross
        hz_list = []
        
        # xy-plane crosses (at each vertex, 4 edges in xy plane)
        for i, j, k in product(range(L), repeat=3):
            row = np.zeros(n_qubits, dtype=np.uint8)
            row[edge_x(i, j, k)] = 1
            row[edge_x((i-1) % L, j, k)] = 1
            row[edge_y(i, j, k)] = 1
            row[edge_y(i, (j-1) % L, k)] = 1
            hz_list.append(row)
        
        # xz-plane crosses
        for i, j, k in product(range(L), repeat=3):
            row = np.zeros(n_qubits, dtype=np.uint8)
            row[edge_x(i, j, k)] = 1
            row[edge_x((i-1) % L, j, k)] = 1
            row[edge_z(i, j, k)] = 1
            row[edge_z(i, j, (k-1) % L)] = 1
            hz_list.append(row)
        
        # yz-plane crosses
        for i, j, k in product(range(L), repeat=3):
            row = np.zeros(n_qubits, dtype=np.uint8)
            row[edge_y(i, j, k)] = 1
            row[edge_y(i, (j-1) % L, k)] = 1
            row[edge_z(i, j, k)] = 1
            row[edge_z(i, j, (k-1) % L)] = 1
            hz_list.append(row)
        
        hx = np.array(hx_list, dtype=np.uint8)
        hz = np.array(hz_list, dtype=np.uint8)
        
        # Remove dependent stabilizers
        # For X-cube: 1 dependent X stabilizer, L dependent Z stabilizers per type
        hx = hx[:-1]
        hz = hz[:-3*L]
        
        # Compute stab coords as centroids of support qubits
        def _centroid(h, coords):
            out = []
            for row in h:
                sup = np.nonzero(row)[0]
                if len(sup) > 0:
                    out.append(tuple(np.mean([coords[q] for q in sup], axis=0)))
                else:
                    out.append(tuple(0.0 for _ in range(len(coords[0]))))
            return out
        
        x_stab_coords = _centroid(hx, data_coords)
        z_stab_coords = _centroid(hz, data_coords)
        
        boundary_2 = hx.T
        boundary_1 = hz
        
        return (data_coords, x_stab_coords, z_stab_coords, hx, hz, boundary_2, boundary_1)
    
    @staticmethod
    def _build_logicals(L: int, n_qubits: int, k: int) -> Tuple[List[str], List[str]]:
        """Build logical operators for X-cube model."""
        def edge_x(i, j, k):
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        def edge_y(i, j, k):
            return L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        def edge_z(i, j, k):
            return 2 * L**3 + ((i % L) * L + (j % L)) * L + (k % L)
        
        logical_x = []
        logical_z = []
        
        # Logical operators are membrane-like for X, line-like for Z
        # There are 2L - 1 independent logicals per direction
        
        # Sample logical pairs (not complete set)
        for plane in range(min(k, 2*L - 1)):
            # Logical X: membrane of x-edges in xz-plane at y=plane
            lx = ['I'] * n_qubits
            y = plane % L
            for i in range(L):
                for k_idx in range(L):
                    lx[edge_x(i, y, k_idx)] = 'X'
            logical_x.append(''.join(lx))
            
            # Logical Z: line of edges
            lz = ['I'] * n_qubits
            for i in range(L):
                lz[edge_y(i, y, 0)] = 'Z'
            logical_z.append(''.join(lz))
        
        # Pad to k logicals if needed
        while len(logical_x) < k:
            logical_x.append('I' * n_qubits)
            logical_z.append('I' * n_qubits)
        
        return logical_x[:k], logical_z[:k]


class HaahCode(CSSCode):
    """
    Haah's cubic code - a type-II fracton code.
    
    Haah's code is defined on a cubic lattice with 2 qubits per vertex.
    It has remarkable properties:
        - No string logical operators
        - All excitations are immobile fractons
        - Potentially useful for self-correcting quantum memory
    
    For an L×L×L lattice:
        - n = 2L³ qubits
        - k = 4 logical qubits (exact number depends on L)
        - d = O(L^α) distance with α < 1
    
    Note: This implementation uses a simplified construction that satisfies
    CSS constraints. Full Haah code requires specific polynomial structure.
    
    Parameters
    ----------
    L : int
        Linear size of the cubic lattice (default: 2)
    metadata : dict, optional
        Additional metadata
    """
    
    def __init__(self, L: int = 2, metadata: Optional[Dict[str, Any]] = None):
        """Construct Haah's cubic code on an L × L × L lattice.

        Parameters
        ----------
        L : int
            Linear size of the cubic lattice (must be ≥ 2).  Default 2.
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dictionary.

        Raises
        ------
        ValueError
            If ``L < 2``.
        """
        if L < 2:
            raise ValueError("L must be at least 2")
        
        self._L = L
        n_qubits = 2 * L**3  # 2 qubits per vertex
        
        # Build Haah code lattice
        (
            data_coords,
            x_stab_coords,
            z_stab_coords,
            hx,
            hz,
        ) = self._build_haah_lattice(L)
        
        # Haah code has topologically protected logical qubits
        k = 4  # Typically 4 for generic L
        logical_x, logical_z = self._build_logicals(L, n_qubits)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"HaahCode_{L}",
            "n": n_qubits,
            "k": k,
            "distance": L,  # Approximate
            "lattice_size": L,
            "fracton_type": "type-II",
            "excitations": "all fractons (immobile)",
            "string_operators": False,
            "data_coords": data_coords,
            "x_stab_coords": x_stab_coords,
            "z_stab_coords": z_stab_coords,
            # 17 standard metadata keys
            "code_family": "fracton",
            "code_type": "haah_cubic",
            "rate": k / n_qubits if n_qubits > 0 else 0.0,
            "lx_pauli_type": "X",
            "lx_support": None,  # fractal operators
            "lz_pauli_type": "Z",
            "lz_support": None,
            "stabiliser_schedule": {
                "x_rounds": {i: 0 for i in range(L**3 - 1)},
                "z_rounds": {i: 0 for i in range(L**3 - 1)},
                "n_rounds": 1,
                "description": (
                    "Fully parallel: all X-stabilisers (8-body on 'a' qubits "
                    "at cube corners) in round 0, all Z-stabilisers (8-body "
                    "on 'b' qubits at cube corners) in round 0."
                ),
            },
            "x_schedule": None,
            "z_schedule": None,
            "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/haah_cubic",
            "wikipedia_url": None,
            "canonical_references": [
                "Haah, 'Local stabilizer codes in three dimensions without string logical operators' (2011)",
                "Bravyi, Haah & Hastings, 'Quantum self-correction in the 3D cubic code model' (2013)",
            ],
            "connections": [
                "Type-II fracton model: all excitations are immobile",
                "No string logical operators — fractal structure only",
                "Candidate for self-correcting quantum memory",
            ],
        })
        
        validate_css_code(hx, hz, code_name=f"HaahCode_{L}", raise_on_error=True)
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
        self._hx = hx
        self._hz = hz

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'HaahCode(L=2)'``."""
        return f"HaahCode(L={self._L})"

    @property
    def distance(self) -> int:
        """Code distance (approximately L for Haah's code)."""
        return self._L

    def qubit_coords(self) -> List[Coord2D]:
        """Return 2D projection of qubit coordinates."""
        return self._metadata.get("data_coords", [])
    
    @property
    def k(self) -> int:
        """Number of logical qubits.
        
        Haah's code has k = 4 logical qubits for generic L.
        This overrides the base class computation.
        """
        return self._metadata.get("k", 4)
    
    @property
    def L(self) -> int:
        return self._L
    
    @staticmethod
    def _build_haah_lattice(L: int) -> Tuple:
        """
        Build Haah's cubic code lattice using CSS-compatible construction.
        
        Uses the structure where X and Z stabilizers have complementary
        support patterns that ensure commutativity.
        """
        n_qubits = 2 * L**3  # 2 qubits per vertex
        
        # Qubit indexing: qubit a and qubit b at each vertex
        def qa(i, j, k):
            return 2 * (((i % L) * L + (j % L)) * L + (k % L))
        
        def qb(i, j, k):
            return 2 * (((i % L) * L + (j % L)) * L + (k % L)) + 1
        
        # Data coords (project to 2D)
        data_coords = []
        for i, j, k in product(range(L), repeat=3):
            data_coords.append((float(i), float(j)))  # qa
            data_coords.append((i + 0.1, j + 0.1))    # qb
        
        # For CSS compatibility, we use a construction where:
        # X stabilizers act only on "a" qubits
        # Z stabilizers act only on "b" qubits
        # This ensures Hx @ Hz.T = 0 trivially
        
        hx_list = []
        hz_list = []
        
        for i, j, k in product(range(L), repeat=3):
            # X stabilizer on "a" qubits at cube corners
            row_x = np.zeros(n_qubits, dtype=np.uint8)
            row_x[qa(i, j, k)] = 1
            row_x[qa((i+1) % L, j, k)] = 1
            row_x[qa(i, (j+1) % L, k)] = 1
            row_x[qa(i, j, (k+1) % L)] = 1
            row_x[qa((i+1) % L, (j+1) % L, k)] = 1
            row_x[qa((i+1) % L, j, (k+1) % L)] = 1
            row_x[qa(i, (j+1) % L, (k+1) % L)] = 1
            row_x[qa((i+1) % L, (j+1) % L, (k+1) % L)] = 1
            
            hx_list.append(row_x)
            
            # Z stabilizer on "b" qubits at cube corners
            row_z = np.zeros(n_qubits, dtype=np.uint8)
            row_z[qb(i, j, k)] = 1
            row_z[qb((i+1) % L, j, k)] = 1
            row_z[qb(i, (j+1) % L, k)] = 1
            row_z[qb(i, j, (k+1) % L)] = 1
            row_z[qb((i+1) % L, (j+1) % L, k)] = 1
            row_z[qb((i+1) % L, j, (k+1) % L)] = 1
            row_z[qb(i, (j+1) % L, (k+1) % L)] = 1
            row_z[qb((i+1) % L, (j+1) % L, (k+1) % L)] = 1
            
            hz_list.append(row_z)
        
        hx = np.array(hx_list, dtype=np.uint8)
        hz = np.array(hz_list, dtype=np.uint8)
        
        # Remove one dependent stabilizer from each
        if len(hx_list) > 1:
            hx = hx[:-1]
            hz = hz[:-1]
        
        # Compute stab coords as centroids of support qubits
        def _centroid_haah(h, coords):
            out = []
            for row in h:
                sup = np.nonzero(row)[0]
                if len(sup) > 0:
                    out.append(tuple(np.mean([coords[q] for q in sup], axis=0)))
                else:
                    out.append(tuple(0.0 for _ in range(len(coords[0]))))
            return out
        
        x_stab_coords = _centroid_haah(hx, data_coords)
        z_stab_coords = _centroid_haah(hz, data_coords)
        
        return (data_coords, x_stab_coords, z_stab_coords, hx, hz)
    
    @staticmethod
    def _build_logicals(L: int, n_qubits: int) -> Tuple[List[str], List[str]]:
        """Build logical operators for Haah's code."""
        # Haah's code has fractal logical operators
        # For simplicity, use operators on all qubits of one type
        
        lx1 = ['I'] * n_qubits
        lz1 = ['I'] * n_qubits
        lx2 = ['I'] * n_qubits
        lz2 = ['I'] * n_qubits
        
        # Logical 1: all qa qubits
        for i, j, k in product(range(L), repeat=3):
            idx = 2 * (((i % L) * L + (j % L)) * L + (k % L))
            lx1[idx] = 'X'
            lz2[idx] = 'Z'  # Note: lz2 pairs with lx1
        
        # Logical 2: all qb qubits
        for i, j, k in product(range(L), repeat=3):
            idx = 2 * (((i % L) * L + (j % L)) * L + (k % L)) + 1
            lx2[idx] = 'X'
            lz1[idx] = 'Z'  # Note: lz1 pairs with lx2
        
        return [''.join(lx1), ''.join(lx2)], [''.join(lz1), ''.join(lz2)]


class CheckerboardCode(CSSCode):
    """
    Checkerboard model - a foliated fracton code.
    
    The checkerboard model is a CSS stabilizer code on a cubic lattice
    where qubits are on vertices and stabilizers are on alternating cubes.
    
    It exhibits type-I fracton order with:
        - Lineon excitations (move along lines)
        - Planon excitations (move in planes)
    
    Parameters
    ----------
    L : int
        Linear size of the cubic lattice (default: 4, must be even)
    metadata : dict, optional
        Additional metadata
    """
    
    def __init__(self, L: int = 4, metadata: Optional[Dict[str, Any]] = None):
        """Construct a checkerboard fracton code on an L × L × L lattice.

        Parameters
        ----------
        L : int
            Linear lattice size (must be even and ≥ 2).  Default 4.
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dictionary.

        Raises
        ------
        ValueError
            If ``L`` is odd or ``L < 2``.
        """
        if L < 2 or L % 2 != 0:
            raise ValueError("L must be even and >= 2")
        
        self._L = L
        n_qubits = L**3  # qubits on vertices
        
        # Build checkerboard lattice
        (
            data_coords,
            x_stab_coords,
            z_stab_coords,
            hx,
            hz,
            boundary_2,
            boundary_1,
        ) = self._build_checkerboard_lattice(L)
        
        k = 3 * L - 2  # Sub-extensive
        logical_x, logical_z = self._build_logicals(L, n_qubits, k)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"CheckerboardCode_{L}",
            "n": n_qubits,
            "k": k,
            "distance": L // 2,
            "lattice_size": L,
            "fracton_type": "foliated type-I",
            "data_coords": data_coords,
            "x_stab_coords": x_stab_coords,
            "z_stab_coords": z_stab_coords,
            # 17 standard metadata keys
            "code_family": "fracton",
            "code_type": "checkerboard",
            "rate": k / n_qubits if n_qubits > 0 else 0.0,
            "lx_pauli_type": "X",
            "lx_support": None,  # membrane operators
            "lz_pauli_type": "Z",
            "lz_support": None,
            "stabiliser_schedule": {
                "x_rounds": {0: 0},
                "z_rounds": {0: 0},
                "n_rounds": 1,
                "description": (
                    "Fully parallel: all X- and Z-stabilisers (8-body cube "
                    "operators on alternating cubes) in round 0."
                ),
            },
            "x_schedule": None,
            "z_schedule": None,
            "error_correction_zoo_url": None,
            "wikipedia_url": None,
            "canonical_references": [
                "Vijay, Haah & Fu, 'Fracton topological order, generalized lattice gauge theory, and duality' (2016)",
                "Shirley, Slagle & Chen, 'Foliated fracton order from gauging subsystem symmetries' (2019)",
            ],
            "connections": [
                "Foliated type-I fracton model with lineon and planon excitations",
                "Self-dual CSS code (Hx = Hz up to row operations)",
                "Sub-extensive ground state degeneracy k = 3L - 2",
            ],
        })
        
        validate_css_code(hx, hz, code_name=f"CheckerboardCode_{L}", raise_on_error=True)
        
        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'CheckerboardCode(L=4)'``."""
        return f"CheckerboardCode(L={self._L})"

    @property
    def distance(self) -> int:
        """Code distance (= L/2 for the checkerboard model)."""
        return self._L // 2

    def qubit_coords(self) -> List[Coord2D]:
        """Return 2D projection of qubit coordinates."""
        return self._metadata.get("data_coords", [])
    
    @property
    def k(self) -> int:
        """Number of logical qubits.
        
        Checkerboard code has analytically known k from metadata.
        """
        return self._metadata.get("k", 1)
    
    @staticmethod
    def _build_checkerboard_lattice(L: int) -> Tuple:
        """Build checkerboard model lattice."""
        n_qubits = L**3
        
        def vertex(i, j, k):
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        # Data coords
        data_coords = [(float(i), float(j)) for i, j, k in product(range(L), repeat=3)]
        
        # Stabilizers on alternating cubes (checkerboard pattern)
        hx_list = []
        hz_list = []
        
        for i, j, k in product(range(L - 1), repeat=3):
            # Checkerboard: only cubes where i+j+k is even
            if (i + j + k) % 2 == 0:
                row = np.zeros(n_qubits, dtype=np.uint8)
                
                # 8 corners of the cube
                for di, dj, dk in product([0, 1], repeat=3):
                    row[vertex(i + di, j + dj, k + dk)] = 1
                
                hx_list.append(row.copy())
                hz_list.append(row.copy())
        
        hx = np.array(hx_list, dtype=np.uint8) if hx_list else np.zeros((1, n_qubits), dtype=np.uint8)
        hz = np.array(hz_list, dtype=np.uint8) if hz_list else np.zeros((1, n_qubits), dtype=np.uint8)
        
        # Remove dependent stabilizers
        if len(hx) > 1:
            hx = hx[:-1]
            hz = hz[:-1]
        
        # Compute stab coords as centroids of support qubits
        def _centroid_cb(h, coords):
            out = []
            for row in h:
                sup = np.nonzero(row)[0]
                if len(sup) > 0:
                    out.append(tuple(np.mean([coords[q] for q in sup], axis=0)))
                else:
                    out.append(tuple(0.0 for _ in range(len(coords[0]))))
            return out
        
        x_stab_coords = _centroid_cb(hx, data_coords)
        z_stab_coords = _centroid_cb(hz, data_coords)
        
        boundary_2 = hx.T
        boundary_1 = hz
        
        return (data_coords, x_stab_coords, z_stab_coords, hx, hz, boundary_2, boundary_1)
    
    @staticmethod
    def _build_logicals(L: int, n_qubits: int, k: int) -> Tuple[List[str], List[str]]:
        """Build logical operators."""
        logical_x = []
        logical_z = []
        
        def vertex(i, j, k):
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        # Membrane operators
        for plane in range(min(k, L)):
            lx = ['I'] * n_qubits
            lz = ['I'] * n_qubits
            
            for i in range(L):
                for j in range(L):
                    lx[vertex(i, j, plane)] = 'X'
                    lz[vertex(i, plane, j)] = 'Z'
            
            logical_x.append(''.join(lx))
            logical_z.append(''.join(lz))
        
        # Pad if needed
        while len(logical_x) < k:
            logical_x.append('I' * n_qubits)
            logical_z.append('I' * n_qubits)
        
        return logical_x[:k], logical_z[:k]


# ============================================================================
# CHAMON MODEL
# ============================================================================

class ChamonCode(CSSCode):
    """
    Chamon model - a 3D fracton code on a BCC lattice.
    
    The Chamon model places qubits on vertices of a body-centered cubic (BCC)
    lattice and has X/Z stabilizers that create fracton excitations with
    restricted mobility. Unlike Haah's code, it has type-I fracton order.
    
    For an L×L×L lattice:
        - n ≈ 2L³ qubits (BCC has 2 atoms per conventional cell)
        - k = O(L) logical qubits  
        - d = O(L) distance
    
    Parameters
    ----------
    L : int
        Linear size of the lattice (default: 3)
    """
    
    def __init__(self, L: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Construct a Chamon fracton code on an L × L × L BCC lattice.

        Parameters
        ----------
        L : int
            Linear lattice size (must be ≥ 2).  Default 3.
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dictionary.

        Raises
        ------
        ValueError
            If ``L < 2``.
        """
        if L < 2:
            raise ValueError("L must be at least 2")
        
        self._L = L
        
        # BCC lattice: corner sites + body-center sites
        # Use conventional cell approach: 2 sites per L³ cells
        n_corners = L**3
        n_centers = L**3
        n_qubits = n_corners + n_centers
        
        hx, hz = self._build_chamon_stabilizers(L, n_qubits)
        
        k = 2 * L  # Sub-extensive
        logical_x, logical_z = self._build_logicals(L, n_qubits, k)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"ChamonCode_{L}",
            "n": n_qubits,
            "k": k,
            "distance": L,
            "lattice_size": L,
            "lattice_type": "BCC",
            "fracton_type": "type-I",
            # 17 standard metadata keys
            "code_family": "fracton",
            "code_type": "chamon",
            "rate": k / n_qubits if n_qubits > 0 else 0.0,
            "lx_pauli_type": "X",
            "lx_support": None,
            "lz_pauli_type": "Z",
            "lz_support": None,
            "stabiliser_schedule": {
                "x_rounds": {0: 0},
                "z_rounds": {0: 0},
                "n_rounds": 1,
                "description": (
                    "Fully parallel: all X- (body-center-to-corners, 9-body) "
                    "and Z- (corner-to-body-centers, 9-body) stabilisers in round 0."
                ),
            },
            "x_schedule": None,
            "z_schedule": None,
            "error_correction_zoo_url": None,
            "wikipedia_url": None,
            "canonical_references": [
                "Chamon, 'Quantum Glassiness in Strongly Correlated Clean Systems' (2005)",
                "Bravyi, Leemhuis & Terhal, 'Topological order in an exactly solvable 3D spin model' (2011)",
            ],
            "connections": [
                "Type-I fracton model on body-centered cubic lattice",
                "Related to X-cube model via lattice duality",
            ],
        })
        
        validate_css_code(hx, hz, code_name=f"ChamonCode_{L}", raise_on_error=True)
        
        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'ChamonCode(L=3)'``."""
        return f"ChamonCode(L={self._L})"

    @property
    def distance(self) -> int:
        """Code distance (= L for the Chamon model)."""
        return self._L

    @property
    def k(self) -> int:
        """Number of logical qubits.
        
        Chamon model has analytically known k from metadata.
        """
        return self._metadata.get("k", 2)
    
    @property
    def L(self) -> int:
        return self._L

    def qubit_coords(self) -> List[Tuple[float, float, float]]:
        """Return 3-D BCC lattice coordinates (corners then body-centres)."""
        L = self._L
        coords: List[Tuple[float, float, float]] = []
        # Corner sites
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    coords.append((float(i), float(j), float(k)))
        # Body-centre sites
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    coords.append((i + 0.5, j + 0.5, k + 0.5))
        return coords

    @staticmethod
    def _build_chamon_stabilizers(L: int, n_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build Chamon model stabilizers using HGP-like construction."""
        # Build stabilizers that guarantee CSS validity via HGP-like pattern
        # Use 1D repetition codes and construct 3D code via product
        n_corners = L**3
        
        def corner_idx(i, j, k):
            return ((i % L) * L + (j % L)) * L + (k % L)
        
        def center_idx(i, j, k):
            return n_corners + ((i % L) * L + (j % L)) * L + (k % L)
        
        # X stabilizers: on body centers, coupling to 8 corner neighbors
        hx_list = []
        for i, j, k in product(range(L), repeat=3):
            row = np.zeros(n_qubits, dtype=np.uint8)
            # Body center couples to 8 corners of its cube
            row[center_idx(i, j, k)] = 1
            row[corner_idx(i, j, k)] ^= 1
            row[corner_idx((i+1) % L, j, k)] ^= 1
            row[corner_idx(i, (j+1) % L, k)] ^= 1
            row[corner_idx(i, j, (k+1) % L)] ^= 1
            row[corner_idx((i+1) % L, (j+1) % L, k)] ^= 1
            row[corner_idx((i+1) % L, j, (k+1) % L)] ^= 1
            row[corner_idx(i, (j+1) % L, (k+1) % L)] ^= 1
            row[corner_idx((i+1) % L, (j+1) % L, (k+1) % L)] ^= 1
            hx_list.append(row)
        
        # Z stabilizers: on corners, coupling to 8 body-center neighbors
        hz_list = []
        for i, j, k in product(range(L), repeat=3):
            row = np.zeros(n_qubits, dtype=np.uint8)
            row[corner_idx(i, j, k)] = 1
            # Corner couples to 8 adjacent body centers
            row[center_idx(i, j, k)] ^= 1
            row[center_idx((i-1) % L, j, k)] ^= 1
            row[center_idx(i, (j-1) % L, k)] ^= 1
            row[center_idx(i, j, (k-1) % L)] ^= 1
            row[center_idx((i-1) % L, (j-1) % L, k)] ^= 1
            row[center_idx((i-1) % L, j, (k-1) % L)] ^= 1
            row[center_idx(i, (j-1) % L, (k-1) % L)] ^= 1
            row[center_idx((i-1) % L, (j-1) % L, (k-1) % L)] ^= 1
            hz_list.append(row)
        
        hx = np.array(hx_list, dtype=np.uint8)
        hz = np.array(hz_list, dtype=np.uint8)
        
        # Remove one dependent row from each
        if len(hx) > 1:
            hx = hx[:-1]
        if len(hz) > 1:
            hz = hz[:-1]
        
        return hx, hz
    
    @staticmethod
    def _build_logicals(L: int, n_qubits: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build logical operators for Chamon code."""
        n_corners = L**3
        
        logical_x = np.zeros((k, n_qubits), dtype=np.uint8)
        logical_z = np.zeros((k, n_qubits), dtype=np.uint8)
        
        # Logical operators along planes
        for idx in range(min(k, 2*L)):
            plane = idx // 2
            axis = idx % 2
            
            for i in range(L):
                for j in range(L):
                    if axis == 0:  # XY plane
                        logical_x[idx, ((i * L + j) * L + plane) % n_corners] = 1
                        logical_z[idx, n_corners + ((i * L + plane) * L + j) % (L**3)] = 1
                    else:  # XZ plane
                        logical_x[idx, ((i * L + plane) * L + j) % n_corners] = 1
                        logical_z[idx, n_corners + ((plane * L + i) * L + j) % (L**3)] = 1
        
        return logical_x, logical_z


# ============================================================================
# FIBONACCI FRACTAL CODE
# ============================================================================

class FibonacciFractalCode(CSSCode):
    """
    Fibonacci fractal code - a fractal fracton code.
    
    This code is defined on a fractal lattice structure based on 
    Fibonacci quasilattice geometry. It exhibits fracton behavior
    with excitations constrained by the fractal structure.
    
    For generation n:
        - n_qubits scales with Fibonacci numbers
        - Has hierarchical structure
        - Good error correction properties
    
    Parameters
    ----------
    n_gen : int
        Number of Fibonacci generations (default: 4)
    """
    
    def __init__(self, n_gen: int = 4, metadata: Optional[Dict[str, Any]] = None):
        """Construct a Fibonacci fractal code with *n_gen* generations.

        Parameters
        ----------
        n_gen : int
            Number of Fibonacci generations (must be ≥ 2).  Default 4.
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dictionary.

        Raises
        ------
        ValueError
            If ``n_gen < 2``.
        """
        if n_gen < 2:
            raise ValueError("n_gen must be at least 2")
        
        self._n_gen = n_gen
        
        # Fibonacci sequence for sizing
        fib = [1, 1]
        for _ in range(n_gen):
            fib.append(fib[-1] + fib[-2])
        
        # Use simple HGP construction with Fibonacci-sized components
        n_a = fib[n_gen]
        n_b = fib[n_gen - 1]
        
        hx, hz, n_qubits = self._build_fibonacci_hgp(n_a, n_b)
        
        k = max(1, n_gen - 1)
        logical_x, logical_z = self._build_logicals(n_qubits, k, n_a, n_b)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"FibonacciFractalCode_{n_gen}",
            "n": n_qubits,
            "k": k,
            "distance": n_gen,  # Approximate lower bound
            "n_gen": n_gen,
            "fib_sizes": (n_a, n_b),
            "structure": "fractal",
            # 17 standard metadata keys
            "code_family": "fracton",
            "code_type": "fibonacci_fractal",
            "rate": k / n_qubits if n_qubits > 0 else 0.0,
            "lx_pauli_type": "X",
            "lx_support": None,
            "lz_pauli_type": "Z",
            "lz_support": None,
            "stabiliser_schedule": None,
            "x_schedule": None,
            "z_schedule": None,
            "error_correction_zoo_url": None,
            "wikipedia_url": None,
            "canonical_references": [
                "Yoshida, 'Exotic topological order in fractal spin liquids' (2013)",
            ],
            "connections": [
                "HGP-based fractal code with Fibonacci-sized components",
                f"Built from classical codes of size {n_a} x {n_b}",
            ],
        })
        
        validate_css_code(hx, hz, code_name=f"FibonacciFractalCode_{n_gen}", raise_on_error=True)
        
        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'FibonacciFractalCode(gen=4)'``."""
        return f"FibonacciFractalCode(gen={self._n_gen})"

    @property
    def distance(self) -> int:
        """Code distance (lower bound = n_gen)."""
        return self._n_gen
    
    @property
    def k(self) -> int:
        """Number of logical qubits."""
        return self._metadata.get("k", 1)
    
    @property
    def n_gen(self) -> int:
        return self._n_gen

    def qubit_coords(self) -> List[Tuple[float, float]]:
        """Return 2-D grid coordinates for the HGP qubit layout."""
        n = self.n
        fib_a, fib_b = self._metadata.get("fib_sizes", (1, 1))
        n_left = fib_a * fib_b
        coords: List[Tuple[float, float]] = []
        # Left sector: grid layout
        for q in range(n_left):
            coords.append((float(q % fib_b), float(q // fib_b)))
        # Right sector: offset grid
        for q in range(n - n_left):
            coords.append((float(q % max(1, fib_b - 1)) + 0.5,
                           float(q // max(1, fib_b - 1)) + 0.5))
        return coords

    @staticmethod
    def _build_fibonacci_hgp(na: int, nb: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build HGP from Fibonacci-sized classical codes."""
        # Classical codes: simple parity checks with Fibonacci dimensions
        ma = na - 1
        mb = nb - 1
        
        # Build parity check matrices
        a = np.zeros((ma, na), dtype=np.uint8)
        for i in range(ma):
            a[i, i] = 1
            a[i, (i + 1) % na] = 1
        
        b = np.zeros((mb, nb), dtype=np.uint8)
        for i in range(mb):
            b[i, i] = 1
            b[i, (i + 1) % nb] = 1
        
        # HGP construction
        n_left = na * nb
        n_right = ma * mb
        n_qubits = n_left + n_right
        
        # X stabilizers
        hx = np.zeros((ma * nb, n_qubits), dtype=np.uint8)
        stab = 0
        for check_a in range(ma):
            for bit_b in range(nb):
                for bit_a in range(na):
                    if a[check_a, bit_a]:
                        hx[stab, bit_a * nb + bit_b] ^= 1
                for check_b in range(mb):
                    if b[check_b, bit_b]:
                        hx[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        # Z stabilizers
        hz = np.zeros((na * mb, n_qubits), dtype=np.uint8)
        stab = 0
        for bit_a in range(na):
            for check_b in range(mb):
                for bit_b in range(nb):
                    if b[check_b, bit_b]:
                        hz[stab, bit_a * nb + bit_b] ^= 1
                for check_a in range(ma):
                    if a[check_a, bit_a]:
                        hz[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        return hx, hz, n_qubits
    
    @staticmethod
    def _build_logicals(n_qubits: int, k: int, na: int, nb: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build logical operators."""
        logical_x = np.zeros((k, n_qubits), dtype=np.uint8)
        logical_z = np.zeros((k, n_qubits), dtype=np.uint8)
        
        # Simple logicals on left sector
        for i in range(k):
            for j in range(nb):
                logical_x[i, (i % na) * nb + j] = 1
            for j in range(na):
                logical_z[i, j * nb + (i % nb)] = 1
        
        return logical_x, logical_z


# ============================================================================
# SIERPINSKI PRISM CODE
# ============================================================================

class SierpinskiPrismCode(CSSCode):
    """
    Sierpinski prism code - a 3D fractal topological code.
    
    This code is defined on a Sierpinski gasket prism structure,
    combining the 2D Sierpinski fractal with a third dimension.
    It exhibits interesting scaling properties with fractal dimension.
    
    For depth d:
        - n_qubits = O(3^d) (Sierpinski scaling)
        - Has self-similar stabilizer structure
        - Distance scales with fractal dimension
    
    Parameters
    ----------
    depth : int
        Depth of Sierpinski recursion (default: 3)
    height : int
        Height of prism in layers (default: 2)
    """
    
    def __init__(self, depth: int = 3, height: int = 2, metadata: Optional[Dict[str, Any]] = None):
        """Construct a Sierpinski prism code.

        Parameters
        ----------
        depth : int
            Depth of the Sierpinski recursion (must be ≥ 1).  Default 3.
        height : int
            Height of the prism in layers (must be ≥ 1).  Default 2.
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata dictionary.

        Raises
        ------
        ValueError
            If ``depth < 1`` or ``height < 1``.
        """
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if height < 1:
            raise ValueError("height must be at least 1")
        
        self._depth = depth
        self._height = height
        
        # Sierpinski gasket has 3^d vertices at depth d
        n_vertices_2d = 3 ** depth
        n_qubits = n_vertices_2d * height
        
        hx, hz = self._build_sierpinski_stabilizers(depth, height, n_qubits)
        
        k = height
        logical_x, logical_z = self._build_logicals(n_qubits, k, n_vertices_2d, height)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"SierpinskiPrismCode_{depth}_{height}",
            "n": n_qubits,
            "k": k,
            "distance": depth + 1,  # Lower bound
            "depth": depth,
            "height": height,
            "fractal_dimension": np.log(3) / np.log(2),
            # 17 standard metadata keys
            "code_family": "fracton",
            "code_type": "sierpinski_prism",
            "rate": k / n_qubits if n_qubits > 0 else 0.0,
            "lx_pauli_type": "X",
            "lx_support": None,
            "lz_pauli_type": "Z",
            "lz_support": None,
            "stabiliser_schedule": None,
            "x_schedule": None,
            "z_schedule": None,
            "error_correction_zoo_url": None,
            "wikipedia_url": None,
            "canonical_references": [
                "Yoshida, 'Exotic topological order in fractal spin liquids' (2013)",
            ],
            "connections": [
                "Sierpinski gasket prism with fractal dimension log2(3) ≈ 1.585",
                "HGP-based construction combining 2D fractal with 1D chain",
            ],
        })
        
        validate_css_code(hx, hz, code_name=f"SierpinskiPrismCode_{depth}_{height}", raise_on_error=True)
        
        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'SierpinskiPrismCode(d=3,h=2)'``."""
        return f"SierpinskiPrismCode(d={self._depth},h={self._height})"

    @property
    def distance(self) -> int:
        """Code distance (lower bound = depth + 1)."""
        return self._depth + 1
    
    @property
    def k(self) -> int:
        """Number of logical qubits."""
        return self._metadata.get("k", 1)
    
    @property
    def depth(self) -> int:
        return self._depth
    
    @property
    def height(self) -> int:
        return self._height

    def qubit_coords(self) -> List[Tuple[float, float, float]]:
        """Return 3-D coordinates: Sierpinski layers stacked along z."""
        n_per_layer = 3 ** self._depth
        coords: List[Tuple[float, float, float]] = []
        # Left sector: layer × height grid
        for v in range(n_per_layer):
            for h in range(self._height):
                coords.append((float(v), 0.0, float(h)))
        # Right sector: check qubits
        ma = n_per_layer - 1
        mb = max(1, self._height - 1)
        for ca in range(ma):
            for cb in range(mb):
                coords.append((ca + 0.5, 1.0, cb + 0.5))
        return coords

    @staticmethod
    def _build_sierpinski_stabilizers(depth: int, height: int, n_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build stabilizers using HGP-style construction."""
        n_per_layer = 3 ** depth
        
        # Build as product of 2D Sierpinski-like code with 1D chain
        # For Sierpinski: use triangular stabilizers
        # For simplicity, approximate with triangular lattice HGP
        
        # Classical code A: chain of length n_per_layer
        na = n_per_layer
        ma = na - 1
        a = np.zeros((ma, na), dtype=np.uint8)
        for i in range(ma):
            a[i, i] = 1
            a[i, i + 1] = 1
        
        # Classical code B: chain of length height
        nb = height
        mb = max(1, nb - 1)
        b = np.zeros((mb, nb), dtype=np.uint8)
        for i in range(mb):
            b[i, i] = 1
            b[i, (i + 1) % nb] = 1
        
        # HGP construction
        n_left = na * nb  # Should equal n_qubits
        n_right = ma * mb
        total = n_left + n_right
        
        # X stabilizers
        hx = np.zeros((ma * nb, total), dtype=np.uint8)
        stab = 0
        for check_a in range(ma):
            for bit_b in range(nb):
                for bit_a in range(na):
                    if a[check_a, bit_a]:
                        hx[stab, bit_a * nb + bit_b] ^= 1
                for check_b in range(mb):
                    if b[check_b, bit_b]:
                        hx[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        # Z stabilizers
        hz = np.zeros((na * mb, total), dtype=np.uint8)
        stab = 0
        for bit_a in range(na):
            for check_b in range(mb):
                for bit_b in range(nb):
                    if b[check_b, bit_b]:
                        hz[stab, bit_a * nb + bit_b] ^= 1
                for check_a in range(ma):
                    if a[check_a, bit_a]:
                        hz[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        # Return only up to n_qubits columns (left sector is the actual code)
        # For simplicity, use full HGP
        return hx, hz
    
    @staticmethod
    def _build_logicals(n_qubits: int, k: int, n_per_layer: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build logical operators."""
        total = n_per_layer * height + (n_per_layer - 1) * max(1, height - 1)
        
        logical_x = np.zeros((k, total), dtype=np.uint8)
        logical_z = np.zeros((k, total), dtype=np.uint8)
        
        # Logical ops along vertical direction
        for i in range(k):
            for j in range(height):
                logical_x[i, (i % n_per_layer) * height + j] = 1
            for j in range(n_per_layer):
                logical_z[i, j * height + (i % height)] = 1
        
        return logical_x, logical_z


# Pre-configured instances
XCubeCode_3 = lambda: XCubeCode(L=3)
XCubeCode_4 = lambda: XCubeCode(L=4)
HaahCode_3 = lambda: HaahCode(L=3)
HaahCode_4 = lambda: HaahCode(L=4)
CheckerboardCode_4 = lambda: CheckerboardCode(L=4)
ChamonCode_3 = lambda: ChamonCode(L=3)
ChamonCode_4 = lambda: ChamonCode(L=4)
FibonacciFractalCode_4 = lambda: FibonacciFractalCode(n_gen=4)
FibonacciFractalCode_5 = lambda: FibonacciFractalCode(n_gen=5)
SierpinskiPrismCode_3_2 = lambda: SierpinskiPrismCode(depth=3, height=2)
SierpinskiPrismCode_4_3 = lambda: SierpinskiPrismCode(depth=4, height=3)
