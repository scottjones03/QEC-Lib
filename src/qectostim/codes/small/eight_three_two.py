"""[[8, 3, 2]] Code — Compact 3-Logical-Qubit Error-Detecting CSS Code

The [[8, 3, 2]] code encodes **3 logical qubits** in **8 physical qubits**
with distance 2.  It is one of the most parameter-efficient CSS codes for
its size, achieving a rate of 3/8 while detecting any single-qubit error.

Construction
------------
The stabiliser group has 5 independent generators (n − k = 8 − 3 = 5):

* **X-type** (1 generator, weight 8):
      X₀X₁X₂X₃X₄X₅X₆X₇ = XXXXXXXX

* **Z-type** (4 generators, weight 4 each):
      Z₀Z₁Z₂Z₃ = ZZZZIIII
      Z₀Z₁Z₄Z₅ = ZZIIZZII
      Z₀Z₂Z₄Z₆ = ZIZIZIZI
      Z₄Z₅Z₆Z₇ = IIIIZZZZ

The Z-type generators form the parity-check matrix of the
[8, 4, 4] extended Hamming code.

Code parameters
---------------
* **n** = 8  physical qubits
* **k** = 3  logical qubits
* **d** = 2  (detects single errors; cannot correct)
* **Rate** R = 3/8 = 0.375

Logical operators
-----------------
Three anticommuting pairs (Lx weight 4, Lz weight 2):

    X̄₁ = X₀X₁X₂X₃  (XXXXIIII)     Z̄₁ = Z₀Z₄  (ZIIIZIII)
    X̄₂ = X₀X₁X₄X₅  (XXIIXXII)     Z̄₂ = Z₀Z₂  (ZIZIIIII)
    X̄₃ = X₀X₂X₄X₆  (XIXIXIXI)     Z̄₃ = Z₀Z₁  (ZZIIIIII)

Qubit layout
------------
Data qubits on a 2×4 grid (viewed as two rows of 4).  The X-stabiliser
acts on all 8 qubits.  Z-stabilisers partition qubits in binary patterns.

::

    Row 1:    4 ─────── 5 ─────── 6 ─────── 7
              │         │         │         │
              │  [X: acts on all 8 qubits]  │
              │                             │
    Row 0:    0 ─────── 1 ─────── 2 ─────── 3

    Z-stabilisers (extended Hamming structure):
      Z₀ = {0,1,2,3}  "ZZZZIIII" — left half
      Z₁ = {0,1,4,5}  "ZZIIZZII" — even columns
      Z₂ = {0,2,4,6}  "ZIZIZIZI" — even positions
      Z₃ = {4,5,6,7}  "IIIIZZZZ" — right half

    Data qubit coordinates (2×4 grid):
      0: (0, 0)   1: (1, 0)   2: (2, 0)   3: (3, 0)   — row 0
      4: (0, 1)   5: (1, 1)   6: (2, 1)   7: (3, 1)   — row 1

    X-stabiliser centroid: (1.5, 0.5) — centre of grid
    Z-stabiliser centroids:
      Z₀: mean of {0,1,2,3} → (1.5, 0.0)
      Z₁: mean of {0,1,4,5} → (0.5, 0.5)
      Z₂: mean of {0,2,4,6} → (1.0, 0.5)
      Z₃: mean of {4,5,6,7} → (1.5, 1.0)

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]] = [[8, 3, 2]]` where:

- :math:`n = 8` physical qubits (2 × 4 grid)
- :math:`k = 3` logical qubits
- :math:`d = 2` (detects single errors; cannot correct)
- Rate :math:`k/n = 3/8 = 0.375`

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **X-type stabilisers**: 1 generator, weight 8 — ``XXXXXXXX``
  (overall parity check on all 8 qubits).
- **Z-type stabilisers**: 4 generators, each weight 4;
  ``ZZZZIIII``, ``ZZIIZZII``, ``ZIZIZIZI``, ``IIIIZZZZ``
  (extended Hamming parity-check structure).
- Measurement schedule: all 5 stabilisers in parallel;
  X-stabiliser uses a depth-8 CNOT circuit, Z-stabilisers depth-4.

Connections to other codes
--------------------------
* **Extended Hamming code**: Z-stabilisers are the parity-check matrix
  of the [8, 4, 4] extended Hamming code.
* **[[4, 2, 2]] code**: obtained by puncturing (removing 4 qubits).
* **Colour codes**: related to colour code constructions through the
  Reed–Muller hierarchy.

Fault tolerance
---------------
* The code detects any weight-1 error but cannot correct it.
* In a **post-selection** scheme, detected errors trigger a restart.
* Transversal CCZ gate can be implemented on the three logical qubits.
* The high encoding rate (3/8) makes it attractive for magic-state factories.

Implementation notes
--------------------
* Stabiliser measurements use one weight-8 X ancilla and four weight-4 Z ancillas.
* The Z-stabiliser circuit depth is 4 (one CNOT per support qubit).
* Hook errors on the X ancilla have weight ≤ 4, which is still detectable.

References
----------
* Gottesman, "Stabilizer codes and quantum error correction",
  Caltech PhD thesis (1997).  arXiv:quant-ph/9705052
* Grassl, code tables at http://www.codetables.de/
* Error Correction Zoo: https://errorcorrectionzoo.org/c/stab_8_3_2
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex3
from qectostim.codes.utils import validate_css_code


class EightThreeTwoCode(CSSCode):
    """[[8, 3, 2]] CSS code — compact 3-logical-qubit error detector.

    Encodes 3 logical qubits in 8 physical qubits with distance 2.
    A single weight-8 X-stabiliser and four weight-4 Z-stabilisers
    detect any single-qubit Pauli error.

    Parameters
    ----------
    metadata : dict, optional
        Extra metadata merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (8).
    k : int
        Number of logical qubits (3).
    distance : int
        Code distance (2).
    hx : np.ndarray
        X-stabiliser parity-check matrix, shape ``(1, 8)``.
    hz : np.ndarray
        Z-stabiliser parity-check matrix, shape ``(4, 8)``.

    Examples
    --------
    >>> code = EightThreeTwoCode()
    >>> code.n, code.k, code.distance
    (8, 3, 2)

    Notes
    -----
    The Z-stabilisers form the parity-check matrix of the [8, 4, 4]
    extended Hamming code.  The X-stabiliser is simply the all-ones
    row (overall parity).  Together they define a highly efficient
    error-detecting code.

    See Also
    --------
    FourQubit422Code : Smaller [[4, 2, 2]] code (obtained by puncturing).
    SteaneCode713 : Error-*correcting* code with similar qubit count.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialise the [[8, 3, 2]] code.

        Builds parity-check matrices, chain complex, logical operators,
        and all standard metadata fields.

        Parameters
        ----------
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata
            dictionary.  User-supplied entries override auto-generated
            ones with the same key.

        Raises
        ------
        No ``ValueError`` raised — all code parameters are fixed.

        Notes
        -----
        The logical X operators have weight 4 (codewords of the [8, 4, 4]
        extended Hamming code) while the logical Z operators have weight 2
        (minimum-weight representatives), giving code distance d = 2.
        """
        # ═══════════════════════════════════════════════════════════════════
        # STABILISER MATRICES
        # ═══════════════════════════════════════════════════════════════════
        # Hx: 1 X-stabiliser (weight 8) — overall parity
        hx = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],  # XXXXXXXX
        ], dtype=np.uint8)

        # Hz: 4 Z-stabilisers (weight 4) — extended Hamming check matrix
        hz = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],  # ZZZZIIII
            [1, 1, 0, 0, 1, 1, 0, 0],  # ZZIIZZII
            [1, 0, 1, 0, 1, 0, 1, 0],  # ZIZIZIZI
            [0, 0, 0, 0, 1, 1, 1, 1],  # IIIIZZZZ
        ], dtype=np.uint8)

        # ═══════════════════════════════════════════════════════════════════
        # CHAIN COMPLEX
        # ═══════════════════════════════════════════════════════════════════
        boundary_2 = hx.T.astype(np.uint8)  # shape (8, 1)
        boundary_1 = hz.astype(np.uint8)    # shape (4, 8)
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        # ═══════════════════════════════════════════════════════════════════
        # LOGICAL OPERATORS (3 anticommuting pairs)
        # ═══════════════════════════════════════════════════════════════════
        # Each Lx_i and Lz_i anticommute (odd overlap), while Lx_i and
        # Lz_j commute for i ≠ j (even overlap).
        #
        # Pair 1:  Lx1 = X₀X₁X₂X₃  —  Lz1 = Z₀Z₄
        #   overlap({0,1,2,3}, {0,4}) = {0} → 1 (odd) → anticommute ✓
        #   Lx1 in ker(Hz): checked ✓   Lz1 in ker(Hx): weight 2 (even) ✓
        #
        # Pair 2:  Lx2 = X₀X₁X₄X₅  —  Lz2 = Z₀Z₂
        #   overlap({0,1,4,5}, {0,2}) = {0} → 1 (odd) → anticommute ✓
        #
        # Pair 3:  Lx3 = X₀X₂X₄X₆  —  Lz3 = Z₀Z₁
        #   overlap({0,2,4,6}, {0,1}) = {0} → 1 (odd) → anticommute ✓
        #
        # Cross-checks (even overlap → commute):
        #   Lx1 ∩ Lz2 = {0,1,2,3} ∩ {0,2} = {0,2} → 2 (even) ✓
        #   Lx1 ∩ Lz3 = {0,1,2,3} ∩ {0,1} = {0,1} → 2 (even) ✓
        #   Lx2 ∩ Lz1 = {0,1,4,5} ∩ {0,4} = {0,4} → 2 (even) ✓
        #   Lx2 ∩ Lz3 = {0,1,4,5} ∩ {0,1} = {0,1} → 2 (even) ✓
        #   Lx3 ∩ Lz1 = {0,2,4,6} ∩ {0,4} = {0,4} → 2 (even) ✓
        #   Lx3 ∩ Lz2 = {0,2,4,6} ∩ {0,2} = {0,2} → 2 (even) ✓
        logical_x = [
            {0: 'X', 1: 'X', 2: 'X', 3: 'X'},  # L1_X = XXXXIIII
            {0: 'X', 1: 'X', 4: 'X', 5: 'X'},  # L2_X = XXIIXXII
            {0: 'X', 2: 'X', 4: 'X', 6: 'X'},  # L3_X = XIXIXIXI
        ]

        logical_z = [
            {0: 'Z', 4: 'Z'},  # L1_Z = ZIIIZIII
            {0: 'Z', 2: 'Z'},  # L2_Z = ZIZIIIII
            {0: 'Z', 1: 'Z'},  # L3_Z = ZZIIIIII
        ]

        # ═══════════════════════════════════════════════════════════════════
        # GEOMETRY — 2×4 grid layout (see module docstring for ASCII diagram)
        # ═══════════════════════════════════════════════════════════════════
        data_coords = [
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0),  # row 0
            (0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0),  # row 1
        ]

        # ═══════════════════════════════════════════════════════════════════
        # METADATA (all 17 standard keys)
        # ═══════════════════════════════════════════════════════════════════
        meta = dict(metadata or {})
        meta["code_family"] = "small_css"
        meta["code_type"] = "eight_three_two"
        meta["n"] = 8
        meta["k"] = 3
        meta["distance"] = 2
        meta["rate"] = 3.0 / 8.0
        meta["data_coords"] = data_coords
        meta["data_qubits"] = list(range(8))
        # X-stab spans all 8 qubits; centroid at grid center
        meta["x_stab_coords"] = [(1.5, 0.5)]
        # Z-stab centroids (calculated from support sets)
        # Z₀={0,1,2,3}: mean of (0,0),(1,0),(2,0),(3,0) = (1.5, 0.0)
        # Z₁={0,1,4,5}: mean of (0,0),(1,0),(0,1),(1,1) = (0.5, 0.5)
        # Z₂={0,2,4,6}: mean of (0,0),(2,0),(0,1),(2,1) = (1.0, 0.5)
        # Z₃={4,5,6,7}: mean of (0,1),(1,1),(2,1),(3,1) = (1.5, 1.0)
        meta["z_stab_coords"] = [
            (1.5, 0.0),   # Z₀ = {0,1,2,3}
            (0.5, 0.5),   # Z₁ = {0,1,4,5}
            (1.0, 0.5),   # Z₂ = {0,2,4,6}
            (1.5, 1.0),   # Z₃ = {4,5,6,7}
        ]
        meta["lx_pauli_type"] = "X"
        meta["lx_support"] = [0, 1, 2, 3]          # first logical X
        meta["lz_pauli_type"] = "Z"
        meta["lz_support"] = [0, 4]                 # first logical Z
        meta["x_logical_coords"] = [data_coords[i] for i in [0, 1, 2, 3]]
        meta["z_logical_coords"] = [data_coords[i] for i in [0, 4]]

        # Schedules
        meta["x_schedule"] = None  # single X-stab → matrix scheduling
        meta["z_schedule"] = None
        meta["stabiliser_schedule"] = {
            "x_rounds": {0: 0},
            "z_rounds": {0: 0, 1: 0, 2: 0, 3: 0},
            "n_rounds": 1,
            "description": (
                "Fully parallel: 1 X-stabiliser and 4 Z-stabilisers "
                "all measured in round 0."
            ),
        }

        # ═══════════════════════════════════════════════════════════════════
        # LITERATURE / PROVENANCE
        # ═══════════════════════════════════════════════════════════════════
        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/stab_8_3_2"
        meta["wikipedia_url"] = "https://en.wikipedia.org/wiki/Quantum_error_correction"
        meta["canonical_references"] = [
            "Gottesman, Caltech PhD thesis (1997). arXiv:quant-ph/9705052",
            "Grassl, code tables: http://www.codetables.de/",
        ]
        meta["connections"] = [
            "Z-stabilisers are parity-check matrix of [8,4,4] extended Hamming code",
            "Obtained by appending overall parity to [[7,1,3]] Hamming construction",
            "Puncturing yields the [[4,2,2]] code",
        ]

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, "EightThreeTwoCode", raise_on_error=True)

        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)

    @property
    def distance(self) -> int:
        """Code distance (2)."""
        return 2

    @property
    def name(self) -> str:
        """Human-readable name: ``'EightThreeTwoCode'``."""
        return "EightThreeTwoCode"

    def qubit_coords(self) -> List:
        """Return 2×4 grid coordinates for the 8 data qubits."""
        return self._metadata.get("data_coords", list(range(8)))


# Convenience alias
Code832 = EightThreeTwoCode
