"""Bacon–Shor Subsystem Code

The Bacon–Shor code is a **subsystem code** on an *m × n* grid of
physical qubits encoding **1 logical qubit** with distance
``d = min(m, n)``.

Key advantage
-------------
All syndrome information can be extracted with **weight-2 gauge
operators** (nearest-neighbour XX or ZZ), unlike surface codes
which require weight-4 stabiliser measurements.

Construction
------------
* Physical qubits are arranged on an ``m × n`` grid
  (``n_phys = m · n``).
* **Gauge operators** (weight 2):

  - X-type: ``XX`` on each horizontal pair in every row
  - Z-type: ``ZZ`` on each vertical pair in every column

* **Stabilisers** are *products* of gauge operators:

  - X-stabiliser for row pair ``(r, r+1)``: X on all qubits in
    rows *r* and *r+1*  → weight 2n
  - Z-stabiliser for col pair ``(c, c+1)``: Z on all qubits in
    cols *c* and *c+1*  → weight 2m

Code parameters
---------------
* **n** = m · n  physical qubits
* **k** = 1      logical qubit
* **d** = min(m, n)
* **Rate** R = 1 / (m · n)
* **Gauge qubits**: ``m · n − 1 − (m − 1) − (n − 1)`` = ``(m−1)(n−1)``

Logical operators
-----------------
* X̄ = X on every qubit in any single row   (weight n)
* Z̄ = Z on every qubit in any single column (weight m)

Subsystem structure
-------------------
The gauge group ``G`` factorises as ``G = S × L_g`` where
``S = ⟨X-stabs, Z-stabs⟩`` is the stabiliser group and ``L_g``
contains the independent gauge operators.  The *dressed* distance
equals ``min(m, n)`` (weight of the minimum-weight dressed logical),
while the *bare* distance is always 1 (single-qubit errors can
flip a gauge logical).

Gauge fixing
------------
Fixing all gauge operators converts the Bacon–Shor code into a
standard CSS stabiliser code.  See
:mod:`qectostim.codes.composite.gauge_fixed` for details.

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]]` where:

- :math:`n = m \times n` physical qubits (grid dimensions)
- :math:`k = 1` logical qubit
- :math:`d = \min(m, n)` code distance
- Rate :math:`k/n = 1 / (m \times n)`
- Gauge qubits: :math:`(m-1)(n-1)`

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **X-type stabilisers**: :math:`m - 1` row-pair stabilisers, each of
  weight :math:`2n` (X on all qubits in two adjacent rows).
- **Z-type stabilisers**: :math:`n - 1` column-pair stabilisers, each
  of weight :math:`2m` (Z on all qubits in two adjacent columns).
- **Gauge operators**: weight-2 XX on horizontal pairs (per row),
  weight-2 ZZ on vertical pairs (per column).
- Measurement schedule: gauge operators are measured independently;
  stabiliser outcomes are inferred from products of gauge measurements.

Connections
-----------
* **Shor code** [[9,1,3]] is the 3 × 3 Bacon–Shor code.
* **Subsystem surface codes** generalise this idea to 2-D topologies.
* **Compass codes** interpolate between Bacon–Shor and surface codes.

References
----------
* Bacon, "Operator quantum error-correcting subsystems for
  self-correcting quantum memories", Phys. Rev. A **73**, 012340
  (2006).  arXiv:quant-ph/0506023
* Aliferis & Cross, "Subsystem fault tolerance with the Bacon–Shor
  code", Phys. Rev. Lett. **98**, 220502 (2007).  arXiv:quant-ph/0610063
* Error Correction Zoo: https://errorcorrectionzoo.org/c/bacon_shor_classical

Fault tolerance
---------------
* All gauge measurements are weight-2, so they require only a single
  CNOT gate and one ancilla qubit each — no hook errors arise.
* The code achieves fault-tolerant error correction with the simplest
  possible measurement circuits of any known code family.
* Transversal CNOT is available between two Bacon–Shor blocks of the
  same geometry.

Decoding
--------
* Row and column parity votes: for an m × n code, measure all m(n−1)
  horizontal XX pairs and n(m−1) vertical ZZ pairs, then majority-vote
  each row/column parity to infer X- and Z-stabiliser outcomes.
* The decoding problem separates into independent 1-D repetition-code
  decoders along rows (for Z errors) and columns (for X errors).
* Effective distance equals min(m, n) under this voting decoder.

Error budget
------------
* With depolarising noise at physical rate p, the logical error rate
  scales as O(p^{⌈d/2⌉}) where d = min(m, n).
* The 3 × 3 Bacon–Shor (Shor [[9,1,3]]) code breaks even at p ≈ 1 %.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import (
    CSSCode,
    MergeStabilizerInfo,
    SeamStabilizer,
    GrownStabilizer,
)
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex3
from qectostim.codes.utils import validate_css_code

Coord2D = Tuple[float, float]


class BaconShorCode(CSSCode):
    """Bacon–Shor subsystem code on an *m × n* grid.

    Encodes 1 logical qubit in ``m × n`` physical qubits with distance
    ``min(m, n)``.  Syndrome extraction uses only **weight-2** gauge
    operators, simplifying hardware requirements.

    Parameters
    ----------
    m : int
        Number of rows (≥ 2).
    n : int
        Number of columns (≥ 2).
    metadata : dict, optional
        Extra key/value pairs merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (``m × n``).
    k : int
        Number of logical qubits (1).
    distance : int
        Code distance (``min(m, n)``).
    hx : np.ndarray
        X-stabiliser parity-check matrix, shape ``(m-1, m*n)``.
    hz : np.ndarray
        Z-stabiliser parity-check matrix, shape ``(n-1, m*n)``.

    Examples
    --------
    >>> code = BaconShorCode(3, 3)
    >>> code.n, code.k, code.distance
    (9, 1, 3)
    >>> x_gauges, z_gauges = code.gauge_operators()
    >>> len(x_gauges), len(z_gauges)  # weight-2 gauge ops
    (6, 6)

    Notes
    -----
    The 3 × 3 Bacon–Shor code is equivalent to Shor's [[9,1,3]] code
    up to a local Clifford rotation.  The gauge structure allows single-
    shot error correction with only 2-body measurements.

    See Also
    --------
    ShorCode91 : The 3×3 Bacon–Shor code in CSS form.
    """

    def __init__(self, m: int = 3, n: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialise the Bacon–Shor code on an *m × n* grid.

        Builds row-pair X-stabilisers, column-pair Z-stabilisers,
        weight-2 gauge operators, logical operators, a chain complex,
        and all standard metadata fields.

        Parameters
        ----------
        m : int, optional
            Number of rows (default 3, must be ≥ 2).
        n : int, optional
            Number of columns (default 3, must be ≥ 2).
        metadata : dict, optional
            Extra metadata merged into the code's metadata dictionary.

        Raises
        ------
        ValueError
            If ``m < 2`` or ``n < 2``.
        """
        if m < 2 or n < 2:
            raise ValueError(f"Grid must be at least 2×2, got {m}×{n}")
        
        n_qubits = m * n
        
        def qubit_idx(row: int, col: int) -> int:
            return row * n + col
        
        # Gauge operators (weight-2):
        # X-type: XX on horizontal pairs (same row)
        # Z-type: ZZ on vertical pairs (same column)
        
        # Stabilizers are products of gauge operators:
        # X stabilizers: product of XX gauges in each row (gives all-X on row)
        # Z stabilizers: product of ZZ gauges in each column (gives all-Z on column)
        
        # For CSS framework, we use the stabilizers (not gauges)
        # X-stabilizer: All X on each row (m-1 independent)
        # Z-stabilizer: All Z on each column (n-1 independent)
        
        # Actually, for Bacon-Shor the stabilizers are:
        # - Product of all X in row i times all X in row j (i<j)
        # - Product of all Z in col i times all Z in col j (i<j)
        # This gives row-pair and column-pair stabilizers
        
        # Simpler: X-type on pairs of rows, Z-type on pairs of columns
        x_stabs = []
        z_stabs = []
        
        # X stabilizers: for each pair of adjacent rows
        for r in range(m - 1):
            stab = [0] * n_qubits
            for c in range(n):
                stab[qubit_idx(r, c)] = 1
                stab[qubit_idx(r + 1, c)] = 1
            x_stabs.append(stab)
        
        # Z stabilizers: for each pair of adjacent columns
        for c in range(n - 1):
            stab = [0] * n_qubits
            for r in range(m):
                stab[qubit_idx(r, c)] = 1
                stab[qubit_idx(r, c + 1)] = 1
            z_stabs.append(stab)
        
        hx = np.array(x_stabs, dtype=np.uint8) if x_stabs else np.zeros((0, n_qubits), dtype=np.uint8)
        hz = np.array(z_stabs, dtype=np.uint8) if z_stabs else np.zeros((0, n_qubits), dtype=np.uint8)
        
        # Gauge operators (store in metadata)
        x_gauges = []  # XX on horizontal pairs
        z_gauges = []  # ZZ on vertical pairs
        
        for r in range(m):
            for c in range(n - 1):
                x_gauges.append((qubit_idx(r, c), qubit_idx(r, c + 1)))
        
        for c in range(n):
            for r in range(m - 1):
                z_gauges.append((qubit_idx(r, c), qubit_idx(r + 1, c)))
        
        # Logical operators
        # Logical X: all X on any row
        lx_support = [qubit_idx(0, c) for c in range(n)]
        
        # Logical Z: all Z on any column
        lz_support = [qubit_idx(r, 0) for r in range(m)]
        
        logical_x: List[PauliString] = [{q: 'X' for q in lx_support}]
        logical_z: List[PauliString] = [{q: 'Z' for q in lz_support}]
        
        distance = min(m, n)

        # ═══════════════════════════════════════════════════════════════════
        # CHAIN COMPLEX
        # ═══════════════════════════════════════════════════════════════════
        boundary_2 = hx.T.astype(np.uint8)
        boundary_1 = hz.astype(np.uint8)
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        # ═══════════════════════════════════════════════════════════════════
        # GEOMETRY — m × n grid
        # ═══════════════════════════════════════════════════════════════════
        coords: Dict[int, Coord2D] = {}
        for r in range(m):
            for c in range(n):
                coords[qubit_idx(r, c)] = (float(c), float(m - 1 - r))
        data_coords = [coords[i] for i in range(n_qubits)]

        # ═══════════════════════════════════════════════════════════════════
        # METADATA (all 17 standard keys)
        # ═══════════════════════════════════════════════════════════════════
        meta = dict(metadata or {})
        meta["code_family"] = "subsystem"
        meta["code_type"] = "bacon_shor"
        meta["name"] = f"BaconShor_{m}x{n}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["distance"] = distance
        meta["rate"] = 1.0 / n_qubits
        meta["m"] = m
        meta["grid_n"] = n
        meta["is_subsystem"] = True
        meta["x_gauges"] = x_gauges
        meta["z_gauges"] = z_gauges

        meta["lx_pauli_type"] = "X"
        meta["lx_support"] = lx_support
        meta["lz_pauli_type"] = "Z"
        meta["lz_support"] = lz_support
        meta["data_coords"] = data_coords
        meta["data_qubits"] = list(range(n_qubits))
        meta["x_logical_coords"] = [data_coords[q] for q in lx_support]
        meta["z_logical_coords"] = [data_coords[q] for q in lz_support]

        _xsc = []
        for _ri in range(hx.shape[0]):
            _sup = np.where(hx[_ri])[0]
            if len(_sup) > 0:
                _xsc.append((float(np.mean([data_coords[q][0] for q in _sup])), float(np.mean([data_coords[q][1] for q in _sup]))))
            else:
                _xsc.append((0.0, 0.0))
        meta["x_stab_coords"] = _xsc
        _zsc = []
        for _ri in range(hz.shape[0]):
            _sup = np.where(hz[_ri])[0]
            if len(_sup) > 0:
                _zsc.append((float(np.mean([data_coords[q][0] for q in _sup])), float(np.mean([data_coords[q][1] for q in _sup]))))
            else:
                _zsc.append((0.0, 0.0))
        meta["z_stab_coords"] = _zsc

        # Schedules — gauge-based extraction uses sequential weight-2 ops
        meta["x_schedule"] = None   # gauge-based, not plaquette-based
        meta["z_schedule"] = None
        meta["stabiliser_schedule"] = {
            "x_rounds": {i: 0 for i in range(len(x_stabs))},
            "z_rounds": {i: 0 for i in range(len(z_stabs))},
            "n_rounds": 1,
            "description": (
                "Stabilisers are products of weight-2 gauge operators; "
                "all row-pair X-stabilisers in round 0, all column-pair "
                "Z-stabilisers in round 0."
            ),
        }

        # ═══════════════════════════════════════════════════════════════════
        # LITERATURE / PROVENANCE
        # ═══════════════════════════════════════════════════════════════════
        meta["error_correction_zoo_url"] = (
            "https://errorcorrectionzoo.org/c/bacon_shor_classical"
        )
        meta["wikipedia_url"] = (
            "https://en.wikipedia.org/wiki/Bacon%E2%80%93Shor_code"
        )
        meta["canonical_references"] = [
            "Bacon, Phys. Rev. A 73, 012340 (2006). arXiv:quant-ph/0506023",
            "Aliferis & Cross, Phys. Rev. Lett. 98, 220502 (2007). arXiv:quant-ph/0610063",
        ]
        meta["connections"] = [
            "Subsystem code: syndrome via weight-2 gauge operators only",
            "3×3 case is equivalent to Shor's [[9,1,3]] code",
            "Compass codes interpolate between Bacon-Shor and surface codes",
        ]

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, f"BaconShor_{m}x{n}", raise_on_error=True)

        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
        self._m_grid = m
        self._n_grid = n
        self._distance = distance

    # ─── Properties ────────────────────────────────────────────────
    @property
    def distance(self) -> int:
        """Code distance (``min(m, n)``)."""
        return self._distance

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'BaconShor_3x3'``."""
        return f"BaconShor_{self._m_grid}x{self._n_grid}"

    def qubit_coords(self) -> List:
        """Return 2-D grid coordinates for the *m × n* data qubits."""
        coords = []
        for r in range(self._m_grid):
            for c in range(self._n_grid):
                coords.append((float(c), float(r)))
        return coords
    
    def gauge_operators(self) -> Tuple[List[PauliString], List[PauliString]]:
        """Return the gauge operators (weight-2).
        
        Returns
        -------
        Tuple[List[PauliString], List[PauliString]]
            (X-type gauges, Z-type gauges)
        """
        x_gauges = self._metadata.get("x_gauges", [])
        z_gauges = self._metadata.get("z_gauges", [])
        
        x_ops = []
        for q1, q2 in x_gauges:
            op = {q1: 'X', q2: 'X'}
            x_ops.append(op)
        
        z_ops = []
        for q1, q2 in z_gauges:
            op = {q1: 'Z', q2: 'Z'}
            z_ops.append(op)
        
        return x_ops, z_ops


    # ------------------------------------------------------------------
    # Physical boundaries
    # ------------------------------------------------------------------

    def has_physical_boundaries(self) -> bool:
        """Bacon–Shor codes have physical boundaries (planar grid)."""
        return True

    # ------------------------------------------------------------------
    # Lattice-surgery merge-stabilizer computation
    # ------------------------------------------------------------------

    def get_merge_stabilizers(
        self,
        merge_type: str,
        my_edge: str,
        other_code: "BaconShorCode",
        other_edge: str,
        my_data_global: Dict[int, int],
        other_data_global: Dict[int, int],
        seam_qubit_offset: int,
    ) -> MergeStabilizerInfo:
        """Compute seam and grown-boundary stabilizers for Bacon–Shor merge.

        Bacon–Shor stabilisers are non-local:

        * **X-stabilisers** are row-pair operators of weight 2n
          (act on all qubits in two adjacent rows).
        * **Z-stabilisers** are column-pair operators of weight 2m
          (act on all qubits in two adjacent columns).

        For a **ZZ merge** along bottom/top edges:

        * Seam Z-stabs are column-pair operators spanning one column from
          each patch (weight 2).
        * Grown X-stabs are existing row-pair operators that gain an
          entire row from the other patch.

        CX schedule: because stabilisers can be high-weight, a serial
        schedule is used (one CX per phase, 2m or 2n phases total).

        Parameters
        ----------
        merge_type : ``\"ZZ\"`` or ``\"XX\"``
        my_edge : ``\"bottom\" | \"top\" | \"left\" | \"right\"``
        other_code : Bacon–Shor code on the other side
        other_edge : edge of *other_code* facing the boundary
        my_data_global : ``{local_data_idx: global_qubit_idx}`` for this block
        other_data_global : ``{local_data_idx: global_qubit_idx}`` for other block
        seam_qubit_offset : starting global qubit index for seam ancillas
        """
        m, n = self._m_grid, self._n_grid
        m_o = other_code._m_grid if hasattr(other_code, '_m_grid') else m
        n_o = other_code._n_grid if hasattr(other_code, '_n_grid') else n

        seam_stab_type = "Z" if merge_type == "ZZ" else "X"
        grown_stab_type = "X" if merge_type == "ZZ" else "Z"

        seam_stabs: List[SeamStabilizer] = []
        grown_stabs: List[GrownStabilizer] = []
        seam_idx = 0

        horizontal_seam = my_edge in ("bottom", "top")

        def my_idx(r: int, c: int) -> int:
            return r * n + c

        def other_idx(r: int, c: int) -> int:
            return r * n_o + c

        if horizontal_seam:
            # ZZ merge bottom/top:  seam Z-stabs are column-pair, grown X-stabs are row-pair
            if merge_type == "ZZ":
                # Seam: for each adjacent pair of columns, create a Z-stab
                # spanning one qubit from this patch's boundary row and one
                # from other patch's boundary row.
                my_boundary_row = m - 1 if my_edge == "bottom" else 0
                oth_boundary_row = 0 if other_edge == "top" else m_o - 1

                n_cols = min(n, n_o)
                for c in range(n_cols - 1):
                    g_anc = seam_qubit_offset + seam_idx
                    seam_idx += 1

                    support: List[int] = []
                    cx_phases: List[List[Tuple[int, int]]] = []

                    # Data from my patch: two qubits in boundary row, columns c and c+1
                    my_q1 = my_data_global[my_idx(my_boundary_row, c)]
                    my_q2 = my_data_global[my_idx(my_boundary_row, c + 1)]
                    # Data from other patch
                    oth_q1 = other_data_global[other_idx(oth_boundary_row, c)]
                    oth_q2 = other_data_global[other_idx(oth_boundary_row, c + 1)]

                    support = [my_q1, my_q2, oth_q1, oth_q2]

                    # 4-phase schedule: one CX per phase (Z-type: data→anc)
                    cx_phases = [
                        [(my_q1, g_anc)],
                        [(my_q2, g_anc)],
                        [(oth_q1, g_anc)],
                        [(oth_q2, g_anc)],
                    ]

                    seam_stabs.append(SeamStabilizer(
                        lattice_position=(float(c) + 0.5, float(my_boundary_row) + 0.5),
                        stab_type="Z",
                        global_ancilla_idx=g_anc,
                        weight=4,
                        cx_per_phase=cx_phases,
                        support_globals=support,
                    ))

                # Grown: row-pair X-stabs at the boundary row gain an extra
                # row from the other patch.  The existing row-pair X-stab for
                # rows (boundary_row - 1, boundary_row) gains new CX to the
                # other patch's boundary row.
                if my_edge == "bottom" and m >= 2:
                    grown_row_pair = m - 2  # stab index for rows (m-2, m-1)
                elif my_edge == "top" and m >= 2:
                    grown_row_pair = 0  # stab index for rows (0, 1)
                else:
                    grown_row_pair = -1

                if grown_row_pair >= 0:
                    new_cx: List[List[Tuple[int, int]]] = [[] for _ in range(4)]
                    new_support: List[int] = []
                    for c in range(min(n, n_o)):
                        g_data = other_data_global[other_idx(oth_boundary_row, c)]
                        new_support.append(g_data)
                        # X-type: anc→data.  Distributed across phases.
                        phase = c % 4
                        new_cx[phase].append((-1, g_data))

                    data_coords = self.metadata.get("data_coords", [])
                    if data_coords:
                        stab_x = float(np.mean([data_coords[my_idx(my_boundary_row, c)][0]
                                                for c in range(n)]))
                        stab_y = float(np.mean([data_coords[my_idx(my_boundary_row, c)][1]
                                                for c in range(n)]))
                    else:
                        stab_x, stab_y = 0.0, 0.0

                    grown_stabs.append(GrownStabilizer(
                        lattice_position=(stab_x, stab_y),
                        stab_type="X",
                        existing_ancilla_global=-1,
                        original_weight=2 * n,
                        new_weight=2 * n + min(n, n_o),
                        new_cx_per_phase=new_cx,
                        belongs_to_block="",
                        new_support_globals=new_support,
                    ))

            else:  # XX merge along bottom/top
                # Seam X-stabs: row-pair operators spanning boundary rows
                my_boundary_row = m - 1 if my_edge == "bottom" else 0
                oth_boundary_row = 0 if other_edge == "top" else m_o - 1

                n_cols = min(n, n_o)
                for c in range(n_cols):
                    g_anc = seam_qubit_offset + seam_idx
                    seam_idx += 1

                    my_q = my_data_global[my_idx(my_boundary_row, c)]
                    oth_q = other_data_global[other_idx(oth_boundary_row, c)]

                    support = [my_q, oth_q]
                    cx_phases = [
                        [(g_anc, my_q)],
                        [(g_anc, oth_q)],
                        [],
                        [],
                    ]

                    seam_stabs.append(SeamStabilizer(
                        lattice_position=(float(c), float(my_boundary_row) + 0.5),
                        stab_type="X",
                        global_ancilla_idx=g_anc,
                        weight=2,
                        cx_per_phase=cx_phases,
                        support_globals=support,
                    ))

        else:  # Vertical seam (left/right)
            if merge_type == "ZZ":
                # Seam Z-stabs: column-pair spanning boundary columns
                my_boundary_col = n - 1 if my_edge == "right" else 0
                oth_boundary_col = 0 if other_edge == "left" else n_o - 1

                n_rows = min(m, m_o)
                for r in range(n_rows - 1):
                    g_anc = seam_qubit_offset + seam_idx
                    seam_idx += 1

                    my_q1 = my_data_global[my_idx(r, my_boundary_col)]
                    my_q2 = my_data_global[my_idx(r + 1, my_boundary_col)]
                    oth_q1 = other_data_global[other_idx(r, oth_boundary_col)]
                    oth_q2 = other_data_global[other_idx(r + 1, oth_boundary_col)]

                    support = [my_q1, my_q2, oth_q1, oth_q2]
                    cx_phases = [
                        [(my_q1, g_anc)],
                        [(my_q2, g_anc)],
                        [(oth_q1, g_anc)],
                        [(oth_q2, g_anc)],
                    ]

                    seam_stabs.append(SeamStabilizer(
                        lattice_position=(float(my_boundary_col) + 0.5, float(r) + 0.5),
                        stab_type="Z",
                        global_ancilla_idx=g_anc,
                        weight=4,
                        cx_per_phase=cx_phases,
                        support_globals=support,
                    ))

            else:  # XX merge left/right
                my_boundary_col = n - 1 if my_edge == "right" else 0
                oth_boundary_col = 0 if other_edge == "left" else n_o - 1

                n_rows = min(m, m_o)
                # Seam X-stabs: row-pair operators spanning boundary columns
                for r in range(n_rows):
                    g_anc = seam_qubit_offset + seam_idx
                    seam_idx += 1

                    my_q = my_data_global[my_idx(r, my_boundary_col)]
                    oth_q = other_data_global[other_idx(r, oth_boundary_col)]

                    support = [my_q, oth_q]
                    cx_phases = [
                        [(g_anc, my_q)],
                        [(g_anc, oth_q)],
                        [],
                        [],
                    ]

                    seam_stabs.append(SeamStabilizer(
                        lattice_position=(float(my_boundary_col) + 0.5, float(r)),
                        stab_type="X",
                        global_ancilla_idx=g_anc,
                        weight=2,
                        cx_per_phase=cx_phases,
                        support_globals=support,
                    ))

        n_phases = max(4, max((len(s.cx_per_phase) for s in seam_stabs), default=4))

        return MergeStabilizerInfo(
            seam_stabs=seam_stabs,
            grown_stabs=grown_stabs,
            seam_type=seam_stab_type,
            grown_type=grown_stab_type,
            num_cx_phases=n_phases,
        )


# Pre-built instances
BaconShor3x3 = lambda: BaconShorCode(3, 3)
BaconShor4x4 = lambda: BaconShorCode(4, 4)
BaconShor5x5 = lambda: BaconShorCode(5, 5)
