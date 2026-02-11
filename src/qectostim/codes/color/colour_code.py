# src/qectostim/codes/color/colour_code.py
r"""
4.8.8 Colour Code — Toric Truncated-Square Tiling
==================================================

This module implements the **4.8.8 colour code** on the *truncated-square*
(Archimedean) tiling with periodic (toric) boundary conditions.

Background
----------
The **4.8.8 colour code** lives on a tiling of the plane by **squares**
(weight-4) and **octagons** (weight-8).  Every vertex is shared by
exactly one square and two octagons (the vertex figure is *4.8.8*).

Each face supports both an X-stabiliser and a Z-stabiliser — the code
is **self-dual** (:math:`H_X = H_Z`).  Self-orthogonality
(:math:`H_X H_X^T = 0 \pmod 2`) is guaranteed because every pair of
adjacent faces shares exactly **2** qubits (even overlap), and any two
non-adjacent faces share 0 qubits.

Periodic boundary conditions (an :math:`L \times L` torus, *L* even)
give the cleanest construction:

- :math:`n = 4L^2` data qubits
- :math:`k = 4` logical qubits
- :math:`d = 2L` code distance
- Valid 3-colouring (squares = colour 0, octagons = colours 1 / 2)

The **3-colourability** of the face graph is the defining property of a
colour code and enables:

* Transversal **Hadamard** (from :math:`H_X = H_Z`)
* Transversal **S** gate (from triorthogonality / colour structure)
* Transversal **CNOT** (standard CSS property)
* **Chromobius** decoding via the colour structure

Code parameters
~~~~~~~~~~~~~~~
+-----+-----+-----+-----+-----+
|  L  |  n  |  k  |  d  | faces |
+=====+=====+=====+=====+=====+
|  2  |  16 |  4  |  4  |  8  |
+-----+-----+-----+-----+-----+
|  4  |  64 |  4  |  8  | 32  |
+-----+-----+-----+-----+-----+
|  6  | 144 |  4  | 12  | 72  |
+-----+-----+-----+-----+-----+

Construction
~~~~~~~~~~~~
At each lattice point :math:`(i, j)` for :math:`0 \le i, j < L` there
are **4 data qubits** labelled N, E, S, W (offsets from the lattice
point towards its four compass neighbours).

- **Square face** at :math:`(i, j)`: the 4 qubits
  :math:`\{N, E, S, W\}` around that lattice point.
- **Octagon face** at :math:`(i, j)`: the 8 qubits formed by taking
  the E qubit of :math:`(i, j)`, the W qubit of :math:`(i{+}1, j)`,
  the N qubit of :math:`(i{+}1, j)`, the S qubit of :math:`(i{+}1, j{+}1)`,
  the W qubit of :math:`(i{+}1, j{+}1)`, the E qubit of :math:`(i, j{+}1)`,
  the S qubit of :math:`(i, j{+}1)`, and the N qubit of :math:`(i, j)`.
  (All indices mod *L*.)

All face overlaps are even (2 shared qubits between any adjacent square
and octagon, 4 between adjacent octagons), ensuring self-orthogonality.

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]]` where:

- :math:`n = 4L^2` physical qubits (for an :math:`L \times L` torus, *L* even)
- :math:`k = 4` logical qubits
- :math:`d = 2L` code distance
- Rate :math:`k/n = 1/L^2`

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **X-type stabilisers**: weight-4 (squares) and weight-8 (octagons); one per face,
  :math:`L^2` squares + :math:`L^2` octagons = :math:`2L^2` total.
- **Z-type stabilisers**: identical support (self-dual, :math:`H_X = H_Z`).
- Measurement schedule: 3-round schedule from the 3-colouring; each colour
  group can be measured in parallel.

References
----------
.. [1] Bombin & Martin-Delgado, "Topological quantum distillation",
   Phys. Rev. Lett. 97, 180501 (2006).  arXiv:quant-ph/0604161
.. [2] Kubica & Beverland, "Universal transversal gates with color
   codes", Phys. Rev. A 91, 032330 (2015).  arXiv:1410.0069
.. [3] Error Correction Zoo — 4.8.8 colour code,
   https://errorcorrectionzoo.org/c/488_color

See Also
--------
qectostim.codes.color.hexagonal_colour : 2D colour code on the 4.8.8
    tiling (alternative implementation).
qectostim.codes.small.steane_713 : Smallest colour code ([[7,1,3]]).

Fault tolerance
~~~~~~~~~~~~~~~
* Transversal H, S, and CNOT give the full Clifford group without
  magic-state distillation.
* For a universal gate set, magic states can be injected via a colour-
  code-native distillation protocol that exploits triorthogonality.

Decoding
~~~~~~~~
* **Chromobius** — the native colour-code decoder; decomposes the
  matching problem by colour pairs.
* **Restriction decoder** — projects onto surface-code sub-problems
  for each colour pair and solves them with MWPM.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from itertools import product as _product

import numpy as np

from ..abstract_css import TopologicalCSSCode, Coord2D
from ..abstract_code import PauliString
from ..utils import compute_css_logicals, vectors_to_paulis_x, vectors_to_paulis_z, validate_css_code
from ..complexes.css_complex import CSSChainComplex3


# Compass directions within each lattice site
_N, _E, _S, _W = 0, 1, 2, 3


def _gf2_rank(M: np.ndarray) -> int:
    """Compute matrix rank over GF(2) via row-echelon reduction."""
    M = M.copy().astype(np.uint8) % 2
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        pivot = None
        for row in range(r, rows):
            if M[row, c]:
                pivot = row
                break
        if pivot is None:
            continue
        M[[r, pivot]] = M[[pivot, r]]
        for row in range(rows):
            if row != r and M[row, c]:
                M[row] = (M[row] + M[r]) % 2
        r += 1
    return r


def _build_toric_488(L: int):
    r"""Build a 4.8.8 colour code on an L × L torus.

    Parameters
    ----------
    L : int
        Lattice side length.  Must be even and ≥ 2 for valid
        3-colouring.

    Returns
    -------
    hx : np.ndarray, shape (n_faces, n_qubits)
        Parity-check matrix (= Hz, self-dual).
    faces : list[list[int]]
        Qubit indices for each face.
    colors : list[int]
        Face colour assignment (0 = red/square, 1 = green/octagon,
        2 = blue/octagon).
    coords : list[tuple[float, float]]
        2D coordinates for each qubit.
    n : int
        Number of data qubits (= 4 L²).
    """
    offsets = {_N: (0.0, 0.4), _E: (0.4, 0.0), _S: (0.0, -0.4), _W: (-0.4, 0.0)}

    qubit_map: Dict[Tuple[int, int, int], int] = {}
    coords: List[Tuple[float, float]] = []
    idx = 0
    for i in range(L):
        for j in range(L):
            for d in range(4):
                qubit_map[(i, j, d)] = idx
                dx, dy = offsets[d]
                coords.append((2.0 * i + dx, 2.0 * j + dy))
                idx += 1
    n = idx  # = 4 * L * L

    faces: List[List[int]] = []
    colors: List[int] = []

    # Square faces — one per lattice point, weight 4
    for i in range(L):
        for j in range(L):
            faces.append(sorted(qubit_map[(i, j, d)] for d in range(4)))
            colors.append(0)

    # Octagon faces — one per lattice point, weight 8
    for i in range(L):
        for j in range(L):
            ni = (i + 1) % L
            nj = (j + 1) % L
            face = sorted([
                qubit_map[(i, j, _E)],
                qubit_map[(ni, j, _W)],
                qubit_map[(ni, j, _N)],
                qubit_map[(ni, nj, _S)],
                qubit_map[(ni, nj, _W)],
                qubit_map[(i, nj, _E)],
                qubit_map[(i, nj, _S)],
                qubit_map[(i, j, _N)],
            ])
            faces.append(face)
            # Alternating green / blue by parity of (i + j)
            colors.append(1 + (i + j) % 2)

    hx = np.zeros((len(faces), n), dtype=np.uint8)
    for fi, f in enumerate(faces):
        for q in f:
            hx[fi, q] = 1

    return hx, faces, colors, coords, n


def _compute_valid_3_coloring(faces: List[List[int]], n_faces: int) -> Optional[List[int]]:
    """Compute a valid 3-colouring by back-tracking.

    Returns a list of colours (0, 1, 2) or *None* if no valid
    colouring exists.  Used as a fallback verification.
    """
    # Build adjacency (shared qubits)
    adj: List[List[int]] = [[] for _ in range(n_faces)]
    for i in range(n_faces):
        si = set(faces[i])
        for j in range(i + 1, n_faces):
            if si & set(faces[j]):
                adj[i].append(j)
                adj[j].append(i)

    coloring = [-1] * n_faces

    def _bt(node: int) -> bool:
        if node == n_faces:
            return True
        for c in range(3):
            if all(coloring[nb] != c for nb in adj[node]):
                coloring[node] = c
                if _bt(node + 1):
                    return True
                coloring[node] = -1
        return False

    return coloring if _bt(0) else None


class ColourCode488(TopologicalCSSCode):
    r"""2D 4.8.8 colour code on a toric truncated-square lattice.

    Builds a **self-dual CSS code** (:math:`H_X = H_Z`) on the
    4.8.8 Archimedean tiling with periodic (toric) boundary
    conditions.  Each vertex of the lattice is incident to exactly
    one weight-4 square and two weight-8 octagons.

    Parameters
    ----------
    distance : int, optional
        Target code distance (default 4).  The lattice side length
        *L* is chosen as ``max(2, distance // 2)``, rounded up to the
        next even integer if necessary.  Actual code distance is
        ``d = 2 * L``.
    metadata : dict, optional
        Additional metadata merged into the code's metadata dict.

    Attributes
    ----------
    n : int
        Number of physical qubits (= 4 L²).
    k : int
        Number of logical qubits (= 4 on the torus).
    distance : int
        Code distance (= 2 L).
    hx : np.ndarray
        X-stabiliser parity-check matrix (= hz, self-dual).
    hz : np.ndarray
        Z-stabiliser parity-check matrix.

    Examples
    --------
    >>> code = ColourCode488(distance=4)
    >>> code.n, code.k, code.distance
    (16, 4, 4)
    >>> (code.hx == code.hz).all()   # self-dual
    True
    >>> code.metadata["is_chromobius_compatible"]
    True

    Notes
    -----
    The toric construction guarantees:

    * **Self-orthogonality** — every pair of adjacent faces shares an
      even number of qubits (2 between square–octagon, 4 between
      octagon–octagon).
    * **3-colourability** — squares are colour 0, octagons alternate
      between colours 1 and 2 by lattice-point parity.  Valid for
      even *L*.
    * **Vertex figure 4.8.8** — every qubit belongs to exactly 3
      faces (1 square + 2 octagons).

    References
    ----------
    .. [1] Bombin & Martin-Delgado, "Topological quantum distillation",
       Phys. Rev. Lett. 97, 180501 (2006).  arXiv:quant-ph/0604161
    .. [2] Kubica & Beverland, PRA 91, 032330 (2015).  arXiv:1410.0069
    .. [3] Error Correction Zoo — 4.8.8 colour code,
       https://errorcorrectionzoo.org/c/488_color

    See Also
    --------
    qectostim.codes.color.hexagonal_colour.HexagonalColourCode :
        Alternative 4.8.8 colour-code implementation.
    """

    def __init__(
        self,
        distance: int = 4,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialise the 4.8.8 colour code on a toric lattice.

        Parameters
        ----------
        distance : int, optional
            Target code distance (default 4).  Must be ≥ 2.
        metadata : dict, optional
            Additional metadata merged into the code's metadata dict.

        Raises
        ------
        ValueError
            If ``distance < 2``.
        """
        if distance < 2:
            raise ValueError(f"Distance must be >= 2, got {distance}")

        # Determine lattice size: d = 2L, so L = ceil(distance / 2),
        # rounded up to next even integer.
        L = max(2, (distance + 1) // 2)
        if L % 2 != 0:
            L += 1
        self._L = L
        actual_distance = 2 * L

        # ── Build tiling ───────────────────────────────────────────
        hx, faces, colors, coords, n = _build_toric_488(L)
        hz = hx.copy()  # self-dual

        # ── Verify self-orthogonality ──────────────────────────────
        so = (hx @ hx.T) % 2
        if np.any(so):
            raise ValueError(
                f"Self-orthogonality check failed for L={L}: "
                f"Hx @ Hx^T has {int(np.sum(so))} non-zero entries"
            )

        # ── Verify 3-colouring ─────────────────────────────────────
        n_faces = len(faces)
        for i in range(n_faces):
            si = set(faces[i])
            for j in range(i + 1, n_faces):
                if si & set(faces[j]):
                    if colors[i] == colors[j]:
                        raise ValueError(
                            f"3-colouring violated: faces {i} and {j} "
                            f"share qubits but both have colour {colors[i]}"
                        )

        # ── Logical operators ──────────────────────────────────────
        # Compute k from GF(2) rank: k = n - 2 * rank(Hx)
        rank_hx = _gf2_rank(hx)
        k = n - 2 * rank_hx

        lx_vecs, lz_vecs = compute_css_logicals(hx, hz)
        # compute_css_logicals may return dependent vectors; trim to k
        lx_vecs = lx_vecs[:k]
        lz_vecs = lz_vecs[:k]
        logical_x: List[PauliString] = vectors_to_paulis_x(lx_vecs)
        logical_z: List[PauliString] = vectors_to_paulis_z(lz_vecs)

        # ── Compute logical supports ──────────────────────────────
        def _support_from_vec(vec: np.ndarray) -> List[int]:
            return [int(i) for i in np.nonzero(vec)[0]]

        if k == 1:
            lx_support = _support_from_vec(lx_vecs[0])
            lz_support = _support_from_vec(lz_vecs[0])
        else:
            lx_support = [_support_from_vec(v) for v in lx_vecs]
            lz_support = [_support_from_vec(v) for v in lz_vecs]

        # ── Chain complex ──────────────────────────────────────────
        # For a self-dual colour code Hx = Hz = hx.
        # CSSChainComplex3 expects:
        #   boundary_2: shape (n_qubits, n_x_stabs) — transpose of Hx
        #   boundary_1: shape (n_z_stabs, n_qubits) — Hz
        # But for self-dual colour codes, Hx = Hz = hx, so we can
        # build a chain complex where ∂2 = hx^T and ∂1 = hx.
        # ∂1 ∘ ∂2 = hx @ hx^T = 0 (by self-orthogonality) ✓
        boundary_2 = hx.T.astype(np.uint8)   # (n, n_faces)
        boundary_1 = hx.astype(np.uint8)     # (n_faces, n)
        chain_complex = CSSChainComplex3(
            boundary_2=boundary_2,
            boundary_1=boundary_1,
        )

        # ── Stabiliser coordinates (centroids) ─────────────────────
        stab_coords: List[Coord2D] = []
        for f in faces:
            cx = sum(coords[q][0] for q in f) / len(f)
            cy = sum(coords[q][1] for q in f) / len(f)
            stab_coords.append((cx, cy))

        data_coords: List[Coord2D] = list(coords)

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, f"ColourCode488_L{L}", raise_on_error=True)

        # ── Metadata ───────────────────────────────────────────────
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update(
            {
                # 17 standard keys
                "code_family": "colour_code",
                "code_type": "4.8.8 colour code (toric)",
                "distance": actual_distance,
                "n": n,
                "k": k,
                "rate": k / n,
                "lx_pauli_type": "X",
                "lz_pauli_type": "Z",
                "lx_support": lx_support,
                "lz_support": lz_support,
                "data_qubits": list(range(n)),
                "stabiliser_schedule": {
                    "x_rounds": {i: 0 for i in range(n_faces)},
                    "z_rounds": {i: 0 for i in range(n_faces)},
                    "n_rounds": 1,
                    "description": (
                        "Fully parallel (round 0).  All stabilisers "
                        "measured simultaneously; colour-code geometry "
                        "uses matrix-based circuit construction."
                    ),
                },
                "x_schedule": None,
                "z_schedule": None,
                "error_correction_zoo_url": (
                    "https://errorcorrectionzoo.org/c/488_color"
                ),
                "wikipedia_url": (
                    "https://en.wikipedia.org/wiki/"
                    "Color_code_(quantum_computing)"
                ),
                "canonical_references": [
                    "Bombin & Martin-Delgado, PRL 97, 180501 (2006). "
                    "arXiv:quant-ph/0604161",
                    "Kubica & Beverland, PRA 91, 032330 (2015). "
                    "arXiv:1410.0069",
                ],
                "connections": [
                    "Self-dual CSS code (Hx = Hz)",
                    "Transversal full Clifford group (H, S, CNOT)",
                    "4.8.8 tiling: truncated-square lattice",
                    "Triangular (6.6.6) colour code: same family, "
                    "uniform weight-6 stabilisers",
                    "Chromobius-compatible via 3-colourability",
                ],
                # Additional colour-code metadata
                "is_self_dual": True,
                "is_colour_code": True,
                "tiling": "4.8.8",
                "boundary_conditions": "periodic (torus)",
                "lattice_size": L,
                "data_coords": data_coords,
                "x_stab_coords": stab_coords,
                "z_stab_coords": stab_coords,
                "stab_colors": colors,
                "is_chromobius_compatible": True,
                "dimension": 2,
                "chain_length": 3,
                "face_weights": {
                    "square": 4,
                    "octagon": 8,
                },
                "n_squares": L * L,
                "n_octagons": L * L,
            }
        )

        # ── Initialise TopologicalCSSCode ──────────────────────────
        super().__init__(
            chain_complex=chain_complex,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )

        # Override Hx / Hz to ensure exact self-dual matrices
        self._hx = hx.astype(np.uint8)
        self._hz = hz.astype(np.uint8)
        self._d = actual_distance

    # ── Properties ─────────────────────────────────────────────────

    @property
    def distance(self) -> int:
        """Code distance (= 2 × lattice side length *L*)."""
        return self._d

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'ColourCode488(d=4)'``."""
        return f"ColourCode488(d={self._d})"

    @property
    def lattice_size(self) -> int:
        """Lattice side length *L*."""
        return self._L

    def qubit_coords(self) -> List[Coord2D]:
        """Return 2D coordinates for each data qubit."""
        return list(self.metadata["data_coords"])