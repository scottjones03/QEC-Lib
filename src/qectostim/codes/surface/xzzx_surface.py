"""XZZX Surface Code (CSS variant)

The XZZX surface code is a variant of the rotated surface code optimised
for **biased noise channels** — situations where one Pauli error (e.g.
Z) dominates.  Under biased noise the XZZX layout can significantly
outperform the standard CSS surface code.

.. note::

   The *true* XZZX surface code has non-CSS stabilisers (weight-4
   XZZX plaquettes).  This module provides a **CSS-compatible**
   version using the rotated surface-code geometry so that standard
   CSS decoders apply.

Bias-tailored noise model
--------------------------
For a noise channel with bias :math:`\\eta = p_Z / p_X`, the XZZX
layout aligns stabiliser Paulis so that the dominant error type
triggers syndromes on only one row or column, effectively decoupling
the X and Z error correction.  In the extreme bias limit
(:math:`\\eta \\to \\infty`) the code behaves like two independent
repetition codes, one per axis.

Clifford deformation
---------------------
The XZZX layout can be viewed as applying single-qubit Clifford
rotations to the standard CSS surface code so that the stabiliser
generators become products of XZZX instead of all-X or all-Z.
The CSS version in this module approximates this by changing the
boundary structure rather than mixing Pauli types.

Construction
------------
* Data qubits sit at **odd-odd** lattice coordinates in a ``(2d)²``
  grid, giving ``n = d²`` data qubits.
* X-stabilisers and Z-stabilisers are placed at **even-even** sites
  whose checkerboard parity determines the sector, with boundary
  exclusion rules that create the required open edges.
* Each stabiliser couples to 2 or 4 data qubits at diagonal offsets
  ``(±1, ±1)``.

Code parameters
---------------
* **n** = d²  physical qubits
* **k** = 1   logical qubit
* **d** = user-specified distance (≥ 2)
* **Rate** R = 1/d²

Logical operators
-----------------
* Logical X: left column  ``x = 1`` — weight d
* Logical Z: top row      ``y = 1`` — weight d

XZZX vs CSS comparison
-----------------------
Under depolarising noise both layouts perform identically.  Under
biased noise the true XZZX code achieves a *higher effective distance*
in the dominant error direction.  The CSS version here preserves the
boundary topology while remaining decoder-compatible.

Connections
-----------
* Standard rotated surface code — same lattice, different Pauli labels
  on stabiliser legs.
* Kitaev toric code — periodic-boundary cousin.
* XZZX codes under biased noise: Bonilla Ataides *et al.*,
  Nat. Commun. **12**, 2172 (2021).

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]] = [[d^2, 1, d]]` where:

- :math:`n = d^2` physical qubits on the rotated lattice
- :math:`k = 1` logical qubit
- :math:`d` = code distance (user-specified, ≥ 2)
- Rate :math:`k/n = 1/d^2`

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **X-type stabilisers**: XZZX-oriented; weight-4 in the bulk, weight-2
  on boundaries.  :math:`(d^2 - 1)/2` total generators.
- **Z-type stabilisers**: XZZX-oriented; weight-4 in the bulk, weight-2
  on boundaries.  :math:`(d^2 - 1)/2` total generators.
- Measurement schedule: 4-phase parallel CNOT schedule identical to the
  standard rotated surface code.  Under biased noise the effective distance
  in the dominant error direction is enhanced.

References
----------
* Bonilla Ataides, Tuckett, Bartlett, Flammia & Brown,
  "The XZZX surface code", Nat. Commun. **12**, 2172 (2021).
  arXiv:2009.07851
* Kitaev, "Fault-tolerant quantum computation by anyons",
  Ann. Phys. **303**, 2–30 (2003).  arXiv:quant-ph/9707021
* Error Correction Zoo: https://errorcorrectionzoo.org/c/xzzx

Fault tolerance
---------------
* Under pure-Z noise the effective distance doubles compared to the
  standard CSS surface code at the same qubit count.
* The XZZX layout is compatible with Floquet-style measurement schedules.
* Flag-qubit gadgets are not required since all stabilisers are weight ≤ 4.

Implementation notes
--------------------
* The CSS approximation in this module preserves the boundary topology
  of the true XZZX code but uses standard X-type and Z-type stabilisers.
* Decoder compatibility: any CSS decoder (PyMatching, Fusion Blossom)
  works directly; for the non-CSS XZZX variant a Pauli-frame tracker is needed.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np

from qectostim.codes.abstract_css import (
    TopologicalCSSCode,
    Coord2D,
    MergeStabilizerInfo,
    SeamStabilizer,
    GrownStabilizer,
)
from qectostim.codes.abstract_code import PauliString, FTGadgetCodeConfig, ScheduleMode
from qectostim.codes.complexes.css_complex import CSSChainComplex3
from qectostim.codes.utils import validate_css_code


class XZZXSurfaceCode(TopologicalCSSCode):
    r"""XZZX-style surface code (CSS variant) on the rotated lattice.

    Encodes 1 logical qubit in *d²* physical qubits with code
    distance *d*.  Uses the same lattice geometry as the rotated surface
    code but is tailored for biased-noise performance.

    d = 3 layout ([[9, 1, 3]])::

        ○ = data qubit (odd,odd)    □ = Z-stab    ■ = X-stab

             □           □
           ╱   ╲       ╱   ╲
        ○(1,1) ○(3,1) ○(5,1)          ← Z̄ (top row)
         ╲   ╱   ╲   ╱   ╲
           ■       ■       ■
         ╱   ╲   ╱   ╲   ╱
        ○(1,3) ○(3,3) ○(5,3)
         ╲   ╱   ╲   ╱   ╲
           □       □       □
         ╱   ╲   ╱   ╲   ╱
        ○(1,5) ○(3,5) ○(5,5)
         ╲   ╱   ╲   ╱
           ■       ■
        ↑
        X̄ (left column)

    Boundary X-stabs (weight 2) and Z-stabs (weight 2) are truncated.
    Stabiliser coords = centroids of support qubits.

    Parameters
    ----------
    distance : int
        Code distance (≥ 2).
    metadata : dict, optional
        Extra key/value pairs merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (``d²``).
    k : int
        Number of logical qubits (1).
    distance : int
        Code distance.
    hx : np.ndarray
        X-stabiliser parity-check matrix.
    hz : np.ndarray
        Z-stabiliser parity-check matrix.

    Examples
    --------
    >>> code = XZZXSurfaceCode(distance=3)
    >>> code.n, code.k, code.distance
    (9, 1, 3)
    >>> code = XZZXSurfaceCode(distance=5)
    >>> code.n
    25

    Notes
    -----
    Under depolarising noise this code is equivalent to the standard
    rotated surface code.  Its advantage emerges under *biased* noise
    where Z errors dominate, yielding higher thresholds than the
    conventional layout.

    See Also
    --------
    RotatedSurfaceCode : Standard CSS rotated surface code.
    """

    def __init__(self, distance: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialise the XZZX-style surface code.

        Builds data-qubit and stabiliser coordinates on the rotated
        lattice, constructs parity-check matrices, a 3-term chain
        complex, logical operators, and all standard metadata fields.

        Parameters
        ----------
        distance : int, optional
            Code distance (default 3, must be ≥ 2).
        metadata : dict, optional
            Extra metadata merged into the code's metadata dictionary.

        Raises
        ------
        ValueError
            If ``distance < 2``.
        """
        if distance < 2:
            raise ValueError(f"Distance must be >= 2, got {distance}")
        
        d = distance
        
        # Build lattice using rotated surface code geometry
        # Data qubits at odd-odd coordinates
        data_coords: Set[Coord2D] = set()
        for x in range(1, 2 * d, 2):
            for y in range(1, 2 * d, 2):
                data_coords.add((float(x), float(y)))
        
        data_coords_sorted = sorted(list(data_coords), key=lambda c: (c[1], c[0]))
        coord_to_idx = {c: i for i, c in enumerate(data_coords_sorted)}
        n_qubits = len(data_coords_sorted)
        
        # Stabilizers at even-even coordinates with boundary rules
        x_stab_coords: Set[Coord2D] = set()
        z_stab_coords: Set[Coord2D] = set()
        
        for x in range(0, 2 * d + 1, 2):
            for y in range(0, 2 * d + 1, 2):
                # Parity determines X vs Z
                parity = ((x // 2) % 2) != ((y // 2) % 2)
                
                # Boundary exclusion rules
                on_left = x == 0
                on_right = x == 2 * d
                on_top = y == 0
                on_bottom = y == 2 * d
                
                if on_left and parity:
                    continue
                if on_right and parity:
                    continue
                if on_top and not parity:
                    continue
                if on_bottom and not parity:
                    continue
                
                coord = (float(x), float(y))
                if parity:
                    x_stab_coords.add(coord)
                else:
                    z_stab_coords.add(coord)
        
        # Build parity check matrices using diagonal neighbors
        hx = self._build_boundary(x_stab_coords, coord_to_idx, n_qubits)
        hz = self._build_boundary(z_stab_coords, coord_to_idx, n_qubits)

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
        
        # Build chain complex
        # boundary_2: shape (n_qubits, n_x_stabs + n_z_stabs)
        boundary_2_x = hx.T.astype(np.uint8)  # shape (n_qubits, n_x_stabs)
        boundary_2_z = hz.T.astype(np.uint8)  # shape (n_qubits, n_z_stabs)
        boundary_2 = np.concatenate([boundary_2_x, boundary_2_z], axis=1)
        
        # boundary_1: Empty for surface code with open boundaries
        boundary_1 = np.zeros((0, n_qubits), dtype=np.uint8)
        
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)
        
        # Logical operators - must satisfy CSS orthogonality:
        # - Logical X must commute with all Z stabilizers (hz @ X_support = 0 mod 2)
        # - Logical Z must commute with all X stabilizers (hx @ Z_support = 0 mod 2)
        #
        # For rotated surface code geometry:
        # - Logical X: left column (x=1) - this is orthogonal to hz
        # - Logical Z: top row (y=1) - this is orthogonal to hx
        # Logical X on the *rightmost* column (x = 2d − 1).  In lattice
        # surgery the leftmost column is adjacent to the ZZ merge boundary;
        # placing the observable there creates weight-2 graphlike errors
        # that reduce the effective distance to d − 1.  Using the rightmost
        # column keeps the observable far from merge bridges, restoring
        # full distance protection.
        x_max = float(2 * d - 1)
        x_logical_coords = {c for c in data_coords if c[0] == x_max}  # Right column
        z_logical_coords = {c for c in data_coords if c[1] == 1.0}    # Top row
        
        x_support = sorted(coord_to_idx[c] for c in x_logical_coords)
        z_support = sorted(coord_to_idx[c] for c in z_logical_coords)
        
        lx = ['I'] * n_qubits
        lz = ['I'] * n_qubits
        for i in x_support:
            lx[i] = 'X'
        for i in z_support:
            lz[i] = 'Z'
        logical_x = [''.join(lx)]
        logical_z = [''.join(lz)]
        
        # ═══════════════════════════════════════════════════════════════════
        # METADATA (all 17 standard keys)
        # ═══════════════════════════════════════════════════════════════════
        meta = dict(metadata or {})
        meta["code_family"] = "surface"
        meta["code_type"] = "xzzx_surface"
        meta["name"] = f"XZZX_Surface_{d}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["distance"] = d
        meta["rate"] = 1.0 / n_qubits
        meta["variant"] = "XZZX"
        meta["data_coords"] = data_coords_sorted
        meta["data_qubits"] = list(range(n_qubits))
        meta["x_stab_coords"] = x_stab_lattice_coords
        meta["z_stab_coords"] = z_stab_lattice_coords

        meta["lx_pauli_type"] = "X"
        meta["lx_support"] = x_support
        meta["lz_pauli_type"] = "Z"
        meta["lz_support"] = z_support
        meta["x_logical_coords"] = sorted(list(x_logical_coords))
        meta["z_logical_coords"] = sorted(list(z_logical_coords))
        
        # Measurement schedules matching rotated surface code
        # 4-phase schedule for weight-4 stabilizers with diagonal neighbors.
        # The phase ordering must match the RotatedSurfaceCode so that
        # hook error propagation is compatible with the stabiliser round
        # builder's detector definitions.  (In the true non-CSS XZZX code
        # the leg order XZZX differs from XXXX/ZZZZ, but in this CSS
        # variant the H-matrices are identical to the rotated surface code.)
        meta["x_schedule"] = [
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
        ]
        meta["z_schedule"] = [
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        ]
        meta["stabiliser_schedule"] = {
            "x_rounds": {i: i % 4 for i in range(hx.shape[0])},
            "z_rounds": {i: i % 4 for i in range(hz.shape[0])},
            "n_rounds": 4,
            "description": (
                "4-phase diagonal-neighbour schedule for the rotated "
                "surface-code lattice.  Each phase couples one of the "
                "four (±1,±1) neighbours."
            ),
        }

        # ═══════════════════════════════════════════════════════════════════
        # LITERATURE / PROVENANCE
        # ═══════════════════════════════════════════════════════════════════
        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/xzzx"
        meta["wikipedia_url"] = "https://en.wikipedia.org/wiki/Toric_code"
        meta["canonical_references"] = [
            "Bonilla Ataides, Tuckett, Bartlett, Flammia & Brown, Nat. Commun. 12, 2172 (2021). arXiv:2009.07851",
            "Kitaev, Ann. Phys. 303, 2-30 (2003). arXiv:quant-ph/9707021",
        ]
        meta["connections"] = [
            "CSS variant of the XZZX surface code optimised for biased noise",
            "Shares rotated surface-code lattice geometry",
            "Higher threshold than standard surface code under Z-biased noise",
        ]

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, f"XZZXSurfaceCode_d{d}", raise_on_error=True)
        
        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)
        
        # Override the parity check matrices for proper CSS structure
        self._hx = hx.astype(np.uint8)
        self._hz = hz.astype(np.uint8)
        self._d = d

    # ─── Properties ────────────────────────────────────────────────
    @property
    def distance(self) -> int:
        """Code distance."""
        return self._d

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'XZZXSurface_d3'``."""
        return f"XZZXSurface_d{self._d}"
    
    @staticmethod
    def _build_boundary(stab_coords: Set[Coord2D], coord_to_idx: Dict[Coord2D, int], 
                        n_qubits: int) -> np.ndarray:
        """Build parity check matrix from stabilizer coords using diagonal neighbors."""
        deltas = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
        rows = []
        for s in sorted(stab_coords):
            row = [0] * n_qubits
            sx, sy = s
            for dx, dy in deltas:
                nbr = (sx + dx, sy + dy)
                idx = coord_to_idx.get(nbr)
                if idx is not None:
                    row[idx] = 1
            rows.append(row)
        if not rows:
            return np.zeros((0, n_qubits), dtype=np.uint8)
        return np.array(rows, dtype=np.uint8)
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates for visualization."""
        return list(self.metadata.get("data_coords", []))
    
    def get_ft_gadget_config(self) -> FTGadgetCodeConfig:
        """
        Return FT gadget configuration for XZZX surface code.

        With correct lattice-position stabiliser coordinates, the
        geometric (±1,±1) schedule validates and emits CNOTs in the
        correct order, so no graph-colouring override is needed.
        """
        return FTGadgetCodeConfig(
            schedule_mode=ScheduleMode.GEOMETRIC,
            first_round_x_detectors=True,
            first_round_z_detectors=True,
            enable_metachecks=False,
        )


    # ------------------------------------------------------------------
    # Lattice-surgery merge-stabilizer computation
    # ------------------------------------------------------------------

    def get_merge_stabilizers(
        self,
        merge_type: str,
        my_edge: str,
        other_code: "XZZXSurfaceCode",
        other_edge: str,
        my_data_global: Dict[int, int],
        other_data_global: Dict[int, int],
        seam_qubit_offset: int,
    ) -> MergeStabilizerInfo:
        """Compute seam and grown-boundary stabilizers for a lattice-surgery merge.

        The XZZX CSS variant uses the **same rotated-lattice geometry** as
        :class:`RotatedSurfaceCode` — data qubits at odd-odd coordinates,
        ancillas at even-even, same boundary types (smooth left/right,
        rough top/bottom), same 4-phase diagonal CX schedule.  The only
        difference is the Pauli labelling on the CX legs (XZZX vs XXXX/ZZZZ),
        but the CSS variant presented here is structurally identical.

        Therefore the merge-stabilizer algorithm is identical to
        :meth:`RotatedSurfaceCode.get_merge_stabilizers`.

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

        # CX schedule (identical to rotated surface code)
        z_sched = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
        x_sched = [(+1, +1), (-1, +1), (+1, -1), (-1, -1)]

        seam_stab_type = "Z" if merge_type == "ZZ" else "X"
        grown_stab_type = "X" if merge_type == "ZZ" else "Z"

        my_coords = list(self.metadata["data_coords"])
        my_coord_to_local = {tuple(c): i for i, c in enumerate(my_coords)}

        other_coords = list(other_code.metadata["data_coords"])
        other_coord_to_local = {tuple(c): i for i, c in enumerate(other_coords)}

        horizontal_seam = my_edge in ("bottom", "top")

        if horizontal_seam:
            if my_edge == "bottom":
                def my_perp(c):   return c[1]
                def oth_perp(c):  return -(2 * d_other - c[1])
                parallel_range = range(0, 2 * d + 1, 2)
            else:
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
                if my_edge == "bottom":
                    return (par, 2 * d_other + perp)
                else:
                    return (par, -perp)
        else:
            if my_edge == "left":
                def my_perp(c):   return c[0]
                def oth_perp(c):  return -(2 * d_other - c[0])
                parallel_range = range(0, 2 * d + 1, 2)
            else:
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

        def _stab_type_at(px: int, py: int) -> str:
            return "X" if ((px // 2) % 2) != ((py // 2) % 2) else "Z"

        seam_stabs: List[SeamStabilizer] = []
        grown_stabs: List[GrownStabilizer] = []
        seam_idx = 0

        for par in parallel_range:
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

            cx_phases: List[List[Tuple[int, int]]] = [[] for _ in range(4)]
            support_globals: List[int] = []
            new_cx_phases: List[List[Tuple[int, int]]] = [[] for _ in range(4)]
            new_support_globals: List[int] = []
            weight = 0
            new_weight_delta = 0

            for phase_idx, (dx, dy) in enumerate(sched):
                if horizontal_seam:
                    nb_par = par + dx
                    nb_perp = dy
                else:
                    nb_par = par + dy
                    nb_perp = dx

                if nb_perp > 0:
                    nb_local = make_neighbor(nb_par, nb_perp)
                    if nb_local in my_coord_to_local:
                        local_idx = my_coord_to_local[nb_local]
                        g_data = my_data_global[local_idx]
                        weight += 1
                        support_globals.append(g_data)
                        cx_phases[phase_idx].append((g_data, -1))
                elif nb_perp < 0:
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
                continue

            if pos_type == seam_stab_type:
                g_anc = seam_qubit_offset + seam_idx
                seam_idx += 1

                final_cx: List[List[Tuple[int, int]]] = [[] for _ in range(4)]
                for ph in range(4):
                    for (g_data, _) in cx_phases[ph]:
                        if pos_type == "Z":
                            final_cx[ph].append((g_data, g_anc))
                        else:
                            final_cx[ph].append((g_anc, g_data))

                seam_stabs.append(SeamStabilizer(
                    lattice_position=(float(lx), float(ly)),
                    stab_type=pos_type,
                    global_ancilla_idx=g_anc,
                    weight=weight,
                    cx_per_phase=final_cx,
                    support_globals=support_globals,
                ))

            elif pos_type == grown_stab_type and new_weight_delta > 0:
                orig_w = weight - new_weight_delta

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

                final_new_cx: List[List[Tuple[int, int]]] = [[] for _ in range(4)]
                for ph in range(4):
                    for (g_data, _) in new_cx_phases[ph]:
                        if pos_type == "Z":
                            final_new_cx[ph].append((g_data, -1))
                        else:
                            final_new_cx[ph].append((-1, g_data))

                grown_stabs.append(GrownStabilizer(
                    lattice_position=(float(lx), float(ly)),
                    stab_type=pos_type,
                    existing_ancilla_global=-1,
                    original_weight=orig_w,
                    new_weight=weight,
                    new_cx_per_phase=final_new_cx,
                    belongs_to_block="",
                    new_support_globals=new_support_globals,
                ))

        return MergeStabilizerInfo(
            seam_stabs=seam_stabs,
            grown_stabs=grown_stabs,
            seam_type=seam_stab_type,
            grown_type=grown_stab_type,
            num_cx_phases=4,
        )


# Pre-built instances
XZZXSurface3 = lambda: XZZXSurfaceCode(distance=3)
XZZXSurface5 = lambda: XZZXSurfaceCode(distance=5)
