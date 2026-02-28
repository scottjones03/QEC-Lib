"""Rotated Surface Code — [[d², 1, d]] Topological Stabiliser Code

The rotated surface code is the workhorse topological code for near-term
quantum error correction.  It encodes **1 logical qubit** in **d²** physical
qubits with code distance **d**, achieving the best qubit-to-distance ratio
among the surface-code family.

Overview
--------
The standard (un-rotated) surface code places data qubits on edges of a
square lattice and stabilisers on faces (X-type plaquettes) and vertices
(Z-type stars).  Rotating the lattice 45° and discarding half the ancillas
yields the *rotated* variant, which uses only d² data qubits instead of
2d² − 2d + 1.  Despite this compression, the code distance remains d.

Lattice geometry (Stim convention)
----------------------------------
Data qubits live at **odd-odd** integer coordinates (x, y) ∈ [1, 2d−1]².
Stabiliser ancillas live at **even-even** coordinates.  A stabiliser at
(sx, sy) acts on up to four data qubits at (sx ± 1, sy ± 1):

    ○ — □ — ○        ○ = data qubit (odd, odd)
    |   |   |        □ = Z-ancilla  (even, even, checkerboard rule A)
    □ — ○ — ■        ■ = X-ancilla  (even, even, checkerboard rule B)
    |   |   |
    ○ — ■ — ○

Boundary conditions determine which ancillas are present:

* **Smooth boundaries** (left/right): truncate X-stabilisers → expose
  X-type logical string along a column.
* **Rough boundaries** (top/bottom): truncate Z-stabilisers → expose
  Z-type logical string along a row.

This means logical X̄ runs vertically (left column) and logical Z̄ runs
horizontally (top row), matching Stim's observable convention.

Code parameters
---------------
* **n** = d² physical data qubits
* **k** = 1 logical qubit
* **d** = code distance (minimum-weight logical operator)
* **Rate** R = 1/d² → 0 as d → ∞ (overhead for topological protection)
* **X-stabilisers**: (d² − 1)/2  weight-4 bulk + weight-2 boundary
* **Z-stabilisers**: (d² − 1)/2  weight-4 bulk + weight-2 boundary

Stabiliser measurement schedule
--------------------------------
Stim uses a **4-layer CNOT schedule** so that X-ancillas and Z-ancillas
touching the *same* data qubit never interfere:

* X-ancilla (ancilla controls data): SE → SW → NE → NW
* Z-ancilla (data controls ancilla): SE → NE → SW → NW

All stabilisers can be measured in parallel within each
layer — there is **no inter-stabiliser ordering dependency** within X
or within Z.

Connections to other codes
--------------------------
* **Toric code**: adding periodic boundary conditions yields the
  [[2L², 2, L]] toric code; the rotated surface code is a single
  *planar patch* with open boundaries.
* **Un-rotated (planar) surface code**: the rotated layout is obtained
  by a 45° rotation + boundary truncation, halving the qubit count.
* **XZZX surface code**: replacing the uniform X/Z stabilisers with
  alternating XZZX operators gives improved performance under biased noise.
* **Colour codes**: the d = 3 rotated surface code is *not* a colour code,
  but shares the same parameters [[9, 1, 3]] as the Shor code.

Homological interpretation
--------------------------
The code arises from a **2-chain complex on a cellulation of the disk**:

    C₂ (faces / stabilisers) —∂₂→ C₁ (edges / data qubits) —∂₁→ C₀ (vertices)

* H_X = ∂₂ᵀ  (X-stabiliser parity-check matrix)
* H_Z = ∂₁   (Z-stabiliser parity-check matrix)
* ∂₂ ∘ ∂₁ = 0  guarantees CSS commutativity (H_X · H_Zᵀ = 0)
* Logical operators correspond to non-trivial 1-cycles / 1-cocycles.

For this implementation, ∂₁ is left as a dummy zero matrix because the
boundary vertices are implicit in the planar patch.

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]] = [[d^2, 1, d]]` where:

- :math:`n = d^2` physical data qubits
- :math:`k = 1` logical qubit
- :math:`d` = code distance (minimum-weight logical operator)
- Rate :math:`k/n = 1/d^2`

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **X-type stabilisers**: weight-4 in the bulk, weight-2 on smooth boundaries;
  :math:`(d^2 - 1)/2` total generators.
- **Z-type stabilisers**: weight-4 in the bulk, weight-2 on rough boundaries;
  :math:`(d^2 - 1)/2` total generators.
- Measurement schedule: 4-phase parallel CNOT schedule (SE → SW → NE → NW
  for X-ancillas; SE → NE → SW → NW for Z-ancillas).  All stabilisers of
  the same type are measured simultaneously within each phase.

References
----------
* Kitaev, "Fault-tolerant quantum computation by anyons",
  Ann. Phys. 303, 2–30 (2003).  arXiv:quant-ph/9707021
* Horsman, Fowler, Devitt & Van Meter, "Surface code quantum computing
  by lattice surgery", New J. Phys. 14, 123011 (2012).  arXiv:1111.4022
* Tomita & Svore, "Low-distance surface codes under realistic quantum
  noise", Phys. Rev. A 90, 062320 (2014).  arXiv:1404.3747
* Error Correction Zoo: https://errorcorrectionzoo.org/c/surface
* Wikipedia: https://en.wikipedia.org/wiki/Toric_code  (surface-code section)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..abstract_css import (
    TopologicalCSSCode,
    Coord2D,
    MergeStabilizerInfo,
    SeamStabilizer,
    GrownStabilizer,
)
from ..abstract_code import PauliString
from ..complexes.css_complex import CSSChainComplex3
from ..utils import validate_css_code


class RotatedSurfaceCode(TopologicalCSSCode):
    """Distance-*d* rotated surface code on a planar patch.

    Encodes 1 logical qubit in d² physical qubits with code distance d.
    Geometry and measurement schedule match Stim's ``gen_surface_code``
    rotated-memory circuits.

    Parameters
    ----------
    distance : int
        Code distance (must be ≥ 2).  The number of physical qubits is d².
    metadata : dict, optional
        Extra metadata merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical data qubits (= d²).
    k : int
        Number of logical qubits (= 1).
    hx : np.ndarray
        X-stabiliser parity-check matrix, shape ``(n_x_checks, n)``.
    hz : np.ndarray
        Z-stabiliser parity-check matrix, shape ``(n_z_checks, n)``.

    Examples
    --------
    >>> code = RotatedSurfaceCode(distance=3)
    >>> code.n, code.k, code.distance
    (9, 1, 3)
    >>> code.hx.shape   # 4 X-stabilisers
    (4, 9)

    Notes
    -----
    The rotated surface code is the most widely studied topological code for
    superconducting-qubit architectures.  Google's 2023 *Nature* paper
    demonstrated below-threshold operation with the d = 3 and d = 5
    instances of this code.

    See Also
    --------
    ToricCode33 : Periodic-boundary variant encoding 2 logical qubits.
    XZZXSurfaceCode : Bias-tailored variant with XZZX stabilisers.
    """

    def __init__(self, distance: int, *, metadata: Optional[Dict[str, Any]] = None):
        """Construct a distance-*d* rotated surface code.

        Builds the planar lattice, derives the chain complex, parity-check
        matrices, logical operators, and Stim-compatible measurement
        schedules.

        Parameters
        ----------
        distance : int
            Code distance (≥ 2).  Physical qubit count = d².
        metadata : dict, optional
            User-supplied metadata merged into the code's metadata dict.

        Raises
        ------
        ValueError
            If ``distance < 2``.
        """
        if distance < 2:
            raise ValueError("RotatedSurfaceCode distance must be >= 2")
        self._d = distance

        (
            data_coords,
            x_stab_coords,
            z_stab_coords,
            x_logical_coords,
            z_logical_coords,
        ) = self._build_lattice(distance)

        # index data coords
        data_coords_sorted = sorted(list(data_coords), key=lambda c: (c[1], c[0]))
        coord_to_index = {c: i for i, c in enumerate(data_coords_sorted)}

        # Build ∂2 as face→edge (stabiliser→data) incidence.
        # We treat edges as data-qubits; here we approximate by attaching each
        # stabiliser to up to 4 surrounding data qubits at ±(1,1).
        boundary_2_x = self._build_boundary2(x_stab_coords, coord_to_index)
        boundary_2_z = self._build_boundary2(z_stab_coords, coord_to_index)

        # Store explicit X/Z parity check matrices (rows = stabilisers, cols = data qubits).
        # These are used by CSSMemoryExperiment to align ancillas with stabiliser coords.
        hx = boundary_2_x.T.astype(np.uint8)
        hz = boundary_2_z.T.astype(np.uint8)

        # Use the original even-even lattice positions for stabiliser
        # ancilla coordinates.  Boundary stabilisers (weight 2) sit at the
        # lattice edge, outside the convex hull of their data qubits.  This
        # is intentional: the geometric CX schedule uses (±1,±1) offsets
        # from the ancilla coordinate to find data qubits, and this only
        # works when boundary ancillae are at the even-even grid position
        # (matching Stim's convention), NOT at the centroid of their support.
        #
        # Previously centroids were used, which moved boundary ancillae
        # inward (e.g. (0,4) → (1,4) for a weight-2 Z-stab with data at
        # (1,3) and (1,5)).  That broke the (±1,±1) validation, causing
        # the scheduler to fall through to arbitrary graph-colouring and
        # producing hook errors that reduced effective distance.
        x_stab_lattice_coords: List[Coord2D] = sorted(x_stab_coords)
        z_stab_lattice_coords: List[Coord2D] = sorted(z_stab_coords)

        # For a square surface code with qubits on vertices, C1 is “data” and
        # C2 is both X and Z faces. You can either build a single C2 with a
        # colour label per face, or keep two disconnected “layers”.
        # For simplicity we take:
        #   C2 = X_faces ⊕ Z_faces
        boundary_2 = np.concatenate([boundary_2_x, boundary_2_z], axis=1)

        # ∂1: edges→vertices. For now we use a simple “no-vertex” model where
        # H_Z is already represented by ∂1; to keep the chain-complex API happy
        # we can use a dummy boundary_1 with shape (#C0, #C1) = (0, n_data).
        # If you later want a strict homological model, you can fill in ∂1
        # from an explicit vertex set.
        boundary_1 = np.zeros((0, len(data_coords_sorted)), dtype=np.uint8)

        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        logical_x, logical_z, x_support, z_support = self._build_logicals(
            data_coords_sorted, coord_to_index, x_logical_coords, z_logical_coords
        )

        meta: Dict[str, Any] = dict(metadata or {})

        n_x_stabs = hx.shape[0]
        n_z_stabs = hz.shape[0]

        meta.update(
            {
                # ── Code parameters ────────────────────────────────────
                "code_family": "surface",
                "code_type": "rotated_surface",
                "distance": distance,
                "n": len(data_coords_sorted),
                "k": 1,
                "rate": 1.0 / (distance * distance),
                # ── Geometry ───────────────────────────────────────────
                "data_qubits": list(range(len(data_coords_sorted))),
                "data_coords": data_coords_sorted,
                "x_stab_coords": x_stab_lattice_coords,
                "z_stab_coords": z_stab_lattice_coords,
                "x_logical_coords": sorted(list(x_logical_coords)),
                "z_logical_coords": sorted(list(z_logical_coords)),
                "logical_x_support": x_support,
                "logical_z_support": z_support,
                # ── Logical operator Pauli types ───────────────────────
                # Standard CSS convention: X̄ is X-type, Z̄ is Z-type.
                "lx_pauli_type": "X",
                "lx_support": x_support,
                "lz_pauli_type": "Z",
                "lz_support": z_support,
                # ── Stabiliser scheduling ──────────────────────────────
                # All X-stabilisers are measured in parallel (round 0),
                # and all Z-stabilisers in parallel (round 0).  There is
                # no inter-stabiliser ordering within a type.
                "stabiliser_schedule": {
                    "x_rounds": {i: 0 for i in range(n_x_stabs)},
                    "z_rounds": {i: 0 for i in range(n_z_stabs)},
                    "n_rounds": 1,
                    "description": (
                        "Fully parallel: every X-stabiliser in round 0, "
                        "every Z-stabiliser in round 0."
                    ),
                },
                # ── Literature / provenance ────────────────────────────
                "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/surface",
                "wikipedia_url": "https://en.wikipedia.org/wiki/Toric_code",
                "canonical_references": [
                    "Kitaev, Ann. Phys. 303, 2–30 (2003). arXiv:quant-ph/9707021",
                    "Horsman et al., New J. Phys. 14, 123011 (2012). arXiv:1111.4022",
                    "Tomita & Svore, Phys. Rev. A 90, 062320 (2014). arXiv:1404.3747",
                ],
                "connections": [
                    "Planar patch of the toric code with open boundaries",
                    "Stim's default code (gen_surface_code rotated_memory)",
                    "Basis for lattice surgery and logical CNOT",
                    "45° rotation of the un-rotated surface code, halving qubit count",
                    "Related: XZZX surface code (bias-tailored variant)",
                ],
            }
        )

        # Stim-style 4-phase schedule for rotated surface code.
        # The schedule must ensure that X and Z stabilizers touching the same
        # data qubit do so in the correct order to avoid introducing unwanted
        # correlations. Stim's convention is:
        #   - For shared data qubits, Z-stabilizer should read BEFORE X-stabilizer writes
        #   - This is achieved by the specific ordering below
        #
        # X-ancilla schedule (ancilla controls data):
        #   Layer 1: (+1, +1) = SE neighbor
        #   Layer 2: (-1, +1) = SW neighbor  
        #   Layer 3: (+1, -1) = NE neighbor
        #   Layer 4: (-1, -1) = NW neighbor
        #
        # Z-ancilla schedule (data controls ancilla):
        #   Layer 1: (+1, +1) = SE neighbor
        #   Layer 2: (+1, -1) = NE neighbor
        #   Layer 3: (-1, +1) = SW neighbor
        #   Layer 4: (-1, -1) = NW neighbor
        meta["x_schedule"] = [
            (1.0, 1.0),    # SE
            (-1.0, 1.0),   # SW
            (1.0, -1.0),   # NE
            (-1.0, -1.0),  # NW
        ]
        meta["z_schedule"] = [
            (1.0, 1.0),    # SE
            (1.0, -1.0),   # NE
            (-1.0, 1.0),   # SW
            (-1.0, -1.0),  # NW
        ]

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, f"RotatedSurface_d{distance}", raise_on_error=True)

        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)

        # Override CSSCode's parity-check matrices with the separated X/Z layers.
        # TopologicalCSSCode initialises Hx/Hz from the combined chain_complex; we want
        # experiments to see the physical X/Z stabiliser layers instead.
        self._hx = hx
        self._hz = hz

    # --- lattice -----------------------------------------------------------------

    @staticmethod
    def _build_lattice(
        d: int,
    ) -> Tuple[Set[Coord2D], Set[Coord2D], Set[Coord2D], Set[Coord2D], Set[Coord2D]]:
        """Build the rotated-surface-code lattice for distance *d*.

        Returns five coordinate sets:

        1. **data_coords** — d² data qubits at odd-odd positions.
        2. **x_stab_coords** — X-ancillas at even-even positions
           (checkerboard rule, smooth-boundary truncation).
        3. **z_stab_coords** — Z-ancillas at even-even positions
           (complementary checkerboard, rough-boundary truncation).
        4. **x_logical_coords** — data qubits in the support of X̄
           (left column, x = 1).
        5. **z_logical_coords** — data qubits in the support of Z̄
           (top row, y = 1).

        Parameters
        ----------
        d : int
            Code distance.

        Returns
        -------
        tuple of five ``Set[Coord2D]``
        """
        data_coords: Set[Coord2D] = set()
        x_logical_coords: Set[Coord2D] = set()
        z_logical_coords: Set[Coord2D] = set()

        # Data qubits at odd-odd coordinates (x, y) both odd, within [1, 2d-1] range.
        # This matches Stim's rotated surface code: exactly d×d data qubits on vertices.
        x_max = 2 * d - 1  # rightmost data-qubit column
        for x in range(1, 2 * d, 2):
            for y in range(1, 2 * d, 2):
                q = (float(x), float(y))
                data_coords.add(q)
                # Track logical operator support for rotated surface code:
                # - Logical Z runs horizontally along top row (y=1) - rough boundary
                # - Logical X runs vertically along the RIGHTMOST column (x=2d-1)
                #
                # We deliberately choose the rightmost column instead of the
                # traditional leftmost column (x=1).  Both are equally valid
                # minimum-weight representatives of the same homology class
                # (they differ by a product of X-stabilisers).  The rightmost
                # column is preferred because in lattice surgery the leftmost
                # column is adjacent to the ZZ merge boundary; placing the
                # observable there creates weight-2 graphlike errors that
                # reduce the effective distance to d − 1.  Using the rightmost
                # column keeps the observable far from merge bridges, restoring
                # full distance protection.
                if y == 1:
                    z_logical_coords.add(q)  # Top row for Z (rough boundary)
                if x == x_max:
                    x_logical_coords.add(q)  # Right column for X (smooth boundary)

        x_stab_coords: Set[Coord2D] = set()
        z_stab_coords: Set[Coord2D] = set()

        # Stabilizers at even-even coordinates. For rotated surface code:
        # X-stabilizers (parity check for X-type errors) at specific even-even positions
        # Z-stabilizers (parity check for Z-type errors) at other even-even positions
        for x in range(0, 2 * d + 1, 2):
            for y in range(0, 2 * d + 1, 2):
                # Determine if this is an X or Z stabilizer based on parity of position
                parity = ((x // 2) % 2) != ((y // 2) % 2)
                
                # Skip boundary ancillas on appropriate edges
                on_left = x == 0
                on_right = x == 2 * d
                on_top = y == 0
                on_bottom = y == 2 * d
                
                # Boundary rules for rotated surface code
                if on_left and parity:
                    continue
                if on_right and parity:
                    continue
                if on_top and not parity:
                    continue
                if on_bottom and not parity:
                    continue
                
                q = (float(x), float(y))
                if parity:
                    x_stab_coords.add(q)
                else:
                    z_stab_coords.add(q)

        return (
            data_coords,
            x_stab_coords,
            z_stab_coords,
            x_logical_coords,
            z_logical_coords,
        )

    # --- incidence / chain complex ----------------------------------------------

    @staticmethod
    def _build_boundary2(
        stab_coords: Set[Coord2D],
        coord_to_index: Dict[Coord2D, int],
    ) -> np.ndarray:
        """∂2: faces->edges incidence using diagonal neighbours (±1,±1)."""
        n_edges = len(coord_to_index)
        deltas: List[Coord2D] = [
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        ]
        cols: List[List[int]] = []
        for s in sorted(stab_coords):
            col = [0] * n_edges
            sx, sy = s
            for dx, dy in deltas:
                nbr = (sx + dx, sy + dy)
                idx = coord_to_index.get(nbr)
                if idx is not None:
                    col[idx] ^= 1
            cols.append(col)
        if not cols:
            return np.zeros((n_edges, 0), dtype=np.uint8)
        return np.array(cols, dtype=np.uint8).T  # shape (#edges, #faces)

    # --- logicals ----------------------------------------------------------------

    def _build_logicals(
        self,
        data_coords: List[Coord2D],
        coord_to_index: Dict[Coord2D, int],
        x_logical_coords: Set[Coord2D],
        z_logical_coords: Set[Coord2D],
    ) -> Tuple[List[PauliString], List[PauliString], List[int], List[int]]:
        """Derive minimum-weight logical X̄ and Z̄ operators.

        For the rotated surface code:

        * **Z̄** = Z on every data qubit in the top row (y = 1).
          This is a weight-*d* string connecting the two rough boundaries.
        * **X̄** = X on every data qubit in the right column (x = 2d−1).
          This is a weight-*d* string connecting the two smooth boundaries.
          The rightmost column is chosen (instead of the equivalent leftmost
          column) because it keeps the X-logical far from the ZZ merge boundary
          in lattice surgery, preventing weight-2 graphlike errors.

        Both are minimum-weight representatives of their homology classes.

        Returns
        -------
        logical_x : list[PauliString]
        logical_z : list[PauliString]
        x_support : list[int]
            Sorted data-qubit indices in the X̄ support.
        z_support : list[int]
            Sorted data-qubit indices in the Z̄ support.
        """

        z_support = sorted(coord_to_index[c] for c in z_logical_coords if c in coord_to_index)
        x_support = sorted(coord_to_index[c] for c in x_logical_coords if c in coord_to_index)

        logical_z: List[PauliString] = [{q: 'Z' for q in z_support}] if z_support else []
        logical_x: List[PauliString] = [{q: 'X' for q in x_support}] if x_support else []

        return logical_x, logical_z, x_support, z_support

    # --- convenience -------------------------------------------------------------

    @property
    def distance(self) -> int:
        """Code distance *d*; the minimum weight of any logical operator."""
        return self._d

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'RotatedSurfaceCode(d=3)'``."""
        return f"RotatedSurfaceCode(d={self._d})"

    def qubit_coords(self) -> List[Coord2D]:
        return list(self.metadata["data_coords"])

    @property
    def hx(self) -> np.ndarray:
        """X stabilisers: shape (#X-checks, #data)."""
        return self._hx

    @property
    def hz(self) -> np.ndarray:
        """Z stabilisers: shape (#Z-checks, #data)."""
        return self._hz

    # ------------------------------------------------------------------
    # Lattice-surgery merge-stabilizer computation
    # ------------------------------------------------------------------

    def get_merge_stabilizers(
        self,
        merge_type: str,
        my_edge: str,
        other_code: "RotatedSurfaceCode",
        other_edge: str,
        my_data_global: Dict[int, int],
        other_data_global: Dict[int, int],
        seam_qubit_offset: int,
    ) -> MergeStabilizerInfo:
        """Compute seam and grown-boundary stabilizers for a lattice-surgery merge.

        This implements the Horsman/Fowler lattice-surgery protocol for the
        rotated surface code.  During a merge the two patches are treated as
        one unified code block; *seam stabilizers* emerge at the lattice
        positions that were truncated at each patch's boundary, and existing
        boundary stabilizers *grow* to maintain commutativity.

        Parameters
        ----------
        merge_type : ``"ZZ"`` or ``"XX"``
        my_edge : ``"bottom" | "top" | "left" | "right"``
        other_code : code instance on the other side of the merge
        other_edge : edge of *other_code* facing the merge boundary
        my_data_global : ``{local_data_idx: global_qubit_idx}`` for this block
        other_data_global : ``{local_data_idx: global_qubit_idx}`` for other block
        seam_qubit_offset : starting global qubit index for seam ancillas
        """
        d = self._d
        d_other = other_code._d if hasattr(other_code, '_d') else d

        # -- schedules -----------------------------------------------
        z_sched = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
        x_sched = [(+1, +1), (-1, +1), (+1, -1), (-1, -1)]

        seam_stab_type = "Z" if merge_type == "ZZ" else "X"
        grown_stab_type = "X" if merge_type == "ZZ" else "Z"

        # -- build coord → local-index maps --------------------------
        my_coords = list(self.metadata["data_coords"])
        my_coord_to_local = {tuple(c): i for i, c in enumerate(my_coords)}

        other_coords = list(other_code.metadata["data_coords"])
        other_coord_to_local = {tuple(c): i for i, c in enumerate(other_coords)}

        # -- determine seam geometry from edge -----------------------
        # my_to_seam / other_to_seam transform a local data coordinate
        # into the "merged lattice" frame where the seam sits at a
        # fixed line.
        #
        # Convention: seam coordinate = 0 along the perpendicular axis.
        #  - data from THIS block has perp > 0
        #  - data from OTHER block has perp < 0
        #
        # For "bottom": my data y=1..2d-1 → keep as-is (perp = y, y>0).
        #               Other data y=1..2d_o-1 → map to -(2d_o - y).
        # For "top":    my data y=1..2d-1 → map to (2d - y) so perp>0.
        #               Other data y=1..2d_o-1 → map to -(y).
        # For "left":   my data x → keep (perp = x, x>0).
        #               Other data x → map to -(2d_o - x).
        # For "right":  my data x → map to (2d - x) so perp>0.
        #               Other data x → -(x).

        horizontal_seam = my_edge in ("bottom", "top")

        if horizontal_seam:
            # seam runs along x-axis; perpendicular is y
            if my_edge == "bottom":
                def my_perp(c):   return c[1]
                def oth_perp(c):  return -(2 * d_other - c[1])
                parallel_range = range(0, 2 * d + 1, 2)
            else:  # top
                def my_perp(c):   return 2 * d - c[1]
                def oth_perp(c):  return -(c[1])
                parallel_range = range(0, 2 * d + 1, 2)

            def par_coord(c):  return c[0]
            def make_neighbor(par, perp):
                if my_edge == "bottom":
                    return (par, perp)
                else:
                    return (par, 2 * d - perp)
            def make_other_neighbor(par, perp):
                """perp is negative in merged frame → convert to other-local."""
                if my_edge == "bottom":
                    return (par, 2 * d_other + perp)
                else:
                    return (par, -perp)
        else:
            # seam runs along y-axis; perpendicular is x
            if my_edge == "left":
                def my_perp(c):   return c[0]
                def oth_perp(c):  return -(2 * d_other - c[0])
                parallel_range = range(0, 2 * d + 1, 2)
            else:  # right
                def my_perp(c):   return 2 * d - c[0]
                def oth_perp(c):  return -(c[0])
                parallel_range = range(0, 2 * d + 1, 2)

            def par_coord(c):  return c[1]
            def make_neighbor(par, perp):
                if my_edge == "left":
                    return (perp, par)
                else:
                    return (2 * d - perp, par)
            def make_other_neighbor(par, perp):
                if my_edge == "left":
                    return (2 * d_other + perp, par)
                else:
                    return (-perp, par)

        # -- helper: classify even-even position type ----------------
        def _stab_type_at(px: int, py: int) -> str:
            """Return 'X' or 'Z' for the even-even lattice position."""
            return "X" if ((px // 2) % 2) != ((py // 2) % 2) else "Z"

        # -- enumerate seam and grown positions ----------------------
        seam_stabs: List[SeamStabilizer] = []
        grown_stabs: List[GrownStabilizer] = []
        seam_idx = 0  # counter for seam ancilla allocation

        for par in parallel_range:
            # Lattice position in the ORIGINAL my-block coordinate frame
            if horizontal_seam:
                if my_edge == "bottom":
                    lx, ly = par, 0
                else:
                    lx, ly = par, 2 * d
            else:
                if my_edge == "left":
                    lx, ly = 0, par
                else:
                    lx, ly = 2 * d, par

            pos_type = _stab_type_at(lx, ly)
            sched = z_sched if pos_type == "Z" else x_sched

            # Compute CX neighbors per phase
            cx_phases: List[List[Tuple[int, int]]] = [[] for _ in range(4)]
            support_globals: List[int] = []
            new_cx_phases: List[List[Tuple[int, int]]] = [[] for _ in range(4)]
            new_support_globals: List[int] = []
            weight = 0
            new_weight_delta = 0

            for phase_idx, (dx, dy) in enumerate(sched):
                # Neighbor in merged frame: perpendicular offset = dy if horizontal, dx if vertical
                if horizontal_seam:
                    nb_par = par + dx
                    nb_perp = dy   # +1 or -1 from seam
                else:
                    nb_par = par + dy
                    nb_perp = dx

                if nb_perp > 0:
                    # Data from MY block
                    nb_local = make_neighbor(nb_par, nb_perp)
                    if nb_local in my_coord_to_local:
                        local_idx = my_coord_to_local[nb_local]
                        g_data = my_data_global[local_idx]
                        weight += 1
                        support_globals.append(g_data)
                        # Will be filled with global ancilla idx below
                        cx_phases[phase_idx].append((g_data, -1))  # placeholder
                elif nb_perp < 0:
                    # Data from OTHER block
                    nb_other_local = make_other_neighbor(nb_par, nb_perp)
                    if nb_other_local in other_coord_to_local:
                        local_idx = other_coord_to_local[nb_other_local]
                        g_data = other_data_global[local_idx]
                        weight += 1
                        support_globals.append(g_data)
                        new_weight_delta += 1
                        new_support_globals.append(g_data)
                        cx_phases[phase_idx].append((g_data, -1))
                        new_cx_phases[phase_idx].append((g_data, -1))

            if weight == 0:
                continue  # degenerate position, skip

            if pos_type == seam_stab_type:
                # --- SEAM STABILIZER ---
                g_anc = seam_qubit_offset + seam_idx
                seam_idx += 1

                # Fill in the actual ancilla global index in CX pairs
                final_cx: List[List[Tuple[int, int]]] = [[] for _ in range(4)]
                for ph in range(4):
                    for (g_data, _) in cx_phases[ph]:
                        if pos_type == "Z":
                            final_cx[ph].append((g_data, g_anc))  # CX data→anc
                        else:
                            final_cx[ph].append((g_anc, g_data))  # CX anc→data

                seam_stabs.append(SeamStabilizer(
                    lattice_position=(float(lx), float(ly)),
                    stab_type=pos_type,
                    global_ancilla_idx=g_anc,
                    weight=weight,
                    cx_per_phase=final_cx,
                    support_globals=support_globals,
                ))

            elif pos_type == grown_stab_type and new_weight_delta > 0:
                # --- GROWN STABILIZER ---
                # Compute original weight (neighbors from my block only)
                orig_w = weight - new_weight_delta

                # If original weight < 2, the code doesn't have an ancilla at
                # this position (weight-1 corner stabs are not included).
                # During merge it gains enough support to become a real stab,
                # so reclassify it as a *seam* stabilizer (gets a new ancilla).
                if orig_w < 2:
                    g_anc = seam_qubit_offset + seam_idx
                    seam_idx += 1
                    final_cx_promoted: List[List[Tuple[int, int]]] = [[] for _ in range(4)]
                    for ph in range(4):
                        for (g_data, _) in cx_phases[ph]:
                            if pos_type == "Z":
                                final_cx_promoted[ph].append((g_data, g_anc))
                            else:
                                final_cx_promoted[ph].append((g_anc, g_data))
                    seam_stabs.append(SeamStabilizer(
                        lattice_position=(float(lx), float(ly)),
                        stab_type=pos_type,
                        global_ancilla_idx=g_anc,
                        weight=weight,
                        cx_per_phase=final_cx_promoted,
                        support_globals=support_globals,
                    ))
                    continue

                # The existing ancilla at this position belongs to my block
                # (it's a boundary stab that was already allocated).
                # We record the lattice coord; the caller resolves the global idx.
                final_new_cx: List[List[Tuple[int, int]]] = [[] for _ in range(4)]
                for ph in range(4):
                    for (g_data, _) in new_cx_phases[ph]:
                        # placeholder ancilla=-1; caller fills from block allocation
                        if pos_type == "Z":
                            final_new_cx[ph].append((g_data, -1))
                        else:
                            final_new_cx[ph].append((-1, g_data))

                grown_stabs.append(GrownStabilizer(
                    lattice_position=(float(lx), float(ly)),
                    stab_type=pos_type,
                    existing_ancilla_global=-1,  # caller fills
                    original_weight=orig_w,
                    new_weight=weight,
                    new_cx_per_phase=final_new_cx,
                    belongs_to_block="",  # caller fills
                    new_support_globals=new_support_globals,
                ))

        return MergeStabilizerInfo(
            seam_stabs=seam_stabs,
            grown_stabs=grown_stabs,
            seam_type=seam_stab_type,
            grown_type=grown_stab_type,
            num_cx_phases=4,
        )
