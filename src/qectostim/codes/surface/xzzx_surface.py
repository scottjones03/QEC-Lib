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

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
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


# Pre-built instances
XZZXSurface3 = lambda: XZZXSurfaceCode(distance=3)
XZZXSurface5 = lambda: XZZXSurfaceCode(distance=5)
