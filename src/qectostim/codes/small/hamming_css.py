"""Hamming-based CSS Codes — ``[[2^m − 1, 2^m − 2m − 1, 3]]`` Family

CSS codes constructed from classical Hamming codes.  For every integer
*m* ≥ 3 the classical ``[2^m − 1, 2^m − m − 1, 3]`` Hamming code is
**self-orthogonal** (its dual is contained in itself), so the CSS
construction can be applied with ``H_X = H_Z = H`` and the commutativity
condition ``H_X · H_Zᵀ = 0`` is automatically satisfied.

The resulting quantum code has parameters

    ``[[n, k, d]] = [[2^m − 1, 2^m − 2m − 1, 3]]``

and corrects any single-qubit Pauli error.

Construction from Hamming codes
-------------------------------
The classical ``[n, n − m, 3]`` Hamming code has parity-check matrix
``H`` whose columns are the binary representations of 1 … n = 2^m − 1.
Because every codeword of the dual code ``H⊥`` is also a codeword of
``H`` (i.e. ``H⊥ ⊂ H``), the CSS construction yields a valid quantum
code with identical X- and Z-stabiliser supports.

Code parameters
---------------
* **n** = 2^m − 1 physical qubits
* **k** = 2^m − 2m − 1 logical qubits
* **d** = 3 (minimum weight of non-stabiliser codewords)
* **Rate** R = k/n → 1 as m → ∞

+-----+------+-----+--------+
|  m  |   n  |  k  | Rate   |
+=====+======+=====+========+
|  3  |   7  |  1  | 0.143  |
|  4  |  15  |  7  | 0.467  |
|  5  |  31  | 21  | 0.677  |
|  6  |  63  | 51  | 0.810  |
+-----+------+-----+--------+

Parity-check matrix
--------------------
Each column *j* of ``H`` (for j = 1 … n) is the *m*-bit binary expansion
of *j*.  Stabiliser *i* therefore acts on every qubit whose index has a
1 in binary position *i*.  Every stabiliser has weight ``2^(m−1)`` and any
two stabilisers share ``2^(m−2)`` qubits.

Self-duality of Hamming codes
-----------------------------
The Hamming code satisfies ``H · Hᵀ = 0`` over GF(2) for m ≥ 3, so the
dual codewords are a subset of the code.  This means the CSS code is
**self-dual**: X-stabilisers and Z-stabilisers have *identical* supports,
which implies a transversal Hadamard ``H^{⊗n}`` maps the code to itself.

Logical operators
-----------------
Logical X and Z representatives are obtained from the kernel of the
combined stabiliser matrix.  Minimum-weight representatives have weight 3
for all values of *m*.

Qubit layout
------------
Due to the exponential growth of Hamming codes, a fixed geometric layout
is not practical.  Instead, qubits are arranged on a circle for
visualisation.

::

    For m=3 (Steane code, n=7):        For m=4 (n=15):

            0                                0
          ╱   ╲                           ╱     ╲
        6       1                       14       1
        │       │                      │           │
        5       2                     13     ●     2
          ╲   ╱                        │   (stabs) │
            4                          12   ...   3
             ╲                           ╲       ╱
              3                            ...

    Data qubits placed at angles θ_i = 2πi/n on a unit circle.
    Stabiliser coords at the centroid of each stabiliser's support.

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]] = [[2^r - 1,\; 2^r - 1 - 2r,\; 3]]` where:

- :math:`n = 2^r - 1` physical qubits
- :math:`k = 2^r - 1 - 2r` logical qubits
- :math:`d = 3` (corrects any single Pauli error)
- Rate :math:`k/n \to 1` as :math:`r \to \infty`

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **X-type stabilisers**: :math:`r` generators, each weight :math:`2^{r-1}`;
  columns of the parity-check matrix are binary expansions of :math:`1 \ldots n`.
  Self-dual: :math:`H_X = H_Z`.
- **Z-type stabilisers**: :math:`r` generators, each weight :math:`2^{r-1}`;
  identical supports to X-type stabilisers.
- Measurement schedule: all :math:`2r` stabilisers in parallel;
  circuit depth equals stabiliser weight :math:`2^{r-1}`.

Connections to other codes
--------------------------
* **Steane code**: the m = 3 instance ``[[7, 1, 3]]`` is exactly the
  Steane code / smallest triangular colour code.
* **Reed–Muller codes**: Hamming CSS codes are related to first-order
  quantum Reed–Muller codes.
* **Concatenated codes**: commonly used as inner codes in concatenated
  fault-tolerant architectures.

References
----------
* Steane, "Error correcting codes in quantum theory",
  Proc. R. Soc. A 452, 2551 (1996).
* Calderbank & Shor, "Good quantum error-correcting codes exist",
  Phys. Rev. A 54, 1098 (1996).  arXiv:quant-ph/9512032
* Hamming, "Error detecting and error correcting codes",
  Bell Syst. Tech. J. 29, 147–160 (1950).
* Error Correction Zoo: https://errorcorrectionzoo.org/c/hamming_css
* Wikipedia: https://en.wikipedia.org/wiki/Hamming_code
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import math

import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex3
from qectostim.codes.utils import (
    compute_css_logicals,
    vectors_to_paulis_x,
    vectors_to_paulis_z,
    validate_css_code,
)


def _hamming_parity_check(m: int) -> np.ndarray:
    """Build parity check matrix for [2^m-1, 2^m-m-1, 3] Hamming code.
    
    Columns are binary representations of 1 to 2^m-1.
    """
    n = 2**m - 1
    H = np.zeros((m, n), dtype=np.uint8)
    for j in range(n):
        val = j + 1
        for i in range(m):
            H[m - 1 - i, j] = (val >> i) & 1
    return H


class HammingCSSCode(TopologicalCSSCode):
    """``[[2^m − 1, 2^m − 2m − 1, 3]]`` CSS code from the classical Hamming code.

    Encodes *k* = 2^m − 2m − 1 logical qubits in *n* = 2^m − 1 physical
    qubits with code distance 3.  Self-dual: ``H_X = H_Z``.

    Parameters
    ----------
    m : int
        Hamming parameter (must be ≥ 3).  Creates the
        ``[[2^m − 1, 2^m − 2m − 1, 3]]`` code.
    metadata : dict, optional
        Extra metadata merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (``2^m − 1``).
    k : int
        Number of logical qubits (``2^m − 2m − 1``).
    distance : int
        Code distance (always 3).
    hx : np.ndarray
        X-stabiliser parity-check matrix, shape ``(m, n)``.
    hz : np.ndarray
        Z-stabiliser parity-check matrix, shape ``(m, n)``.
        Identical to ``hx`` (self-dual).

    Examples
    --------
    >>> code = HammingCSSCode(m=3)
    >>> code.n, code.k, code.distance
    (7, 1, 3)
    >>> (code.hx == code.hz).all()   # self-dual
    True

    Notes
    -----
    The m = 3 case is the well-known Steane ``[[7, 1, 3]]`` code.

    See Also
    --------
    SteaneCode713 : Dedicated implementation for m = 3.
    """

    def __init__(self, m: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialise Hamming CSS code with chain complex.

        Builds the parity-check matrices from the Hamming code, derives
        logical operators via kernel computation, constructs the 3-term
        chain complex, and populates all standard metadata fields.

        Parameters
        ----------
        m : int
            Hamming parameter (≥ 3).
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata
            dictionary.  User-supplied entries override auto-generated
            ones with the same key.

        Raises
        ------
        ValueError
            If ``m < 3``.
        """
        if m < 3:
            raise ValueError(f"m must be >= 3, got {m}")

        n = 2**m - 1
        k = n - 2 * m  # = 2^m - 2m - 1

        # Hamming parity check matrix
        H = _hamming_parity_check(m)

        # For CSS code, use H for both X and Z checks
        # This works because Hamming code is self-orthogonal for m >= 3
        hx = H.copy()
        hz = H.copy()

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, f"HammingCSS_m{m}", raise_on_error=True)

        # Build chain complex for CSS code structure:
        #   C2 (X stabilizers) --∂2--> C1 (qubits) --∂1--> C0 (Z stabilizers)
        #
        # boundary_2 = Hx.T: maps faces (X stabs) → edges (qubits), shape (n, #X_checks)
        # boundary_1 = Hz:   maps edges (qubits) → vertices (Z stabs), shape (#Z_checks, n)
        boundary_2 = hx.T.astype(np.uint8)  # shape (n, m)
        boundary_1 = hz.astype(np.uint8)    # shape (m, n)

        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        # ═══════════════════════════════════════════════════════════════════
        # LOGICAL OPERATORS — kernel-based derivation (not placeholders)
        # ═══════════════════════════════════════════════════════════════════
        log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
        # compute_css_logicals may return dependent vectors; trim to k
        log_x_vecs = log_x_vecs[:k]
        log_z_vecs = log_z_vecs[:k]
        logical_x: List[PauliString] = vectors_to_paulis_x(log_x_vecs)
        logical_z: List[PauliString] = vectors_to_paulis_z(log_z_vecs)

        # ═══════════════════════════════════════════════════════════════════
        # GEOMETRIC LAYOUT — circular arrangement
        # ═══════════════════════════════════════════════════════════════════
        coords: List[Coord2D] = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            coords.append((math.cos(angle), math.sin(angle)))

        # Stabiliser coordinates (centroids of support qubits)
        # Each row of H defines which qubits are in that stabiliser's support
        x_stab_coords: List[Coord2D] = []
        for row in range(m):
            support = [j for j in range(n) if hx[row, j]]
            if support:
                cx = sum(coords[j][0] for j in support) / len(support)
                cy = sum(coords[j][1] for j in support) / len(support)
                x_stab_coords.append((cx, cy))
            else:
                x_stab_coords.append((0.0, 0.0))

        z_stab_coords = x_stab_coords  # Self-dual

        # ═══════════════════════════════════════════════════════════════════
        # LOGICAL OPERATOR SUPPORT (for metadata)
        # ═══════════════════════════════════════════════════════════════════
        def _support_indices(pauli: PauliString) -> List[int]:
            """Extract qubit indices from a PauliString."""
            if isinstance(pauli, dict):
                return sorted(pauli.keys())
            # String form: find non-I positions
            return [i for i, c in enumerate(str(pauli)) if c != 'I']

        lx_supports = [_support_indices(lx) for lx in logical_x]
        lz_supports = [_support_indices(lz) for lz in logical_z]

        # ═══════════════════════════════════════════════════════════════════
        # METADATA
        # ═══════════════════════════════════════════════════════════════════
        meta = dict(metadata or {})
        meta["name"] = f"HammingCSS_m{m}"
        meta["code_family"] = "small_css"
        meta["code_type"] = "hamming_css"
        meta["n"] = n
        meta["k"] = k
        meta["distance"] = 3  # Hamming codes always have distance 3
        meta["rate"] = k / n
        meta["hamming_m"] = m
        meta["data_coords"] = coords
        meta["x_stab_coords"] = x_stab_coords
        meta["z_stab_coords"] = z_stab_coords  # Self-dual

        # ── Logical-operator Pauli types (required for CSS decoding) ──
        meta["lx_pauli_type"] = "X"
        meta["lz_pauli_type"] = "Z"
        meta["lx_support"] = lx_supports
        meta["lz_support"] = lz_supports

        # ── Stabiliser schedule ────────────────────────────────────
        # All m X-stabilisers and all m Z-stabilisers can be measured in
        # a single parallel round (no geometric conflicts on a circular
        # register).
        meta["stabiliser_schedule"] = {
            "x_rounds": {i: 0 for i in range(m)},
            "z_rounds": {i: 0 for i in range(m)},
            "n_rounds": 1,
            "description": (
                f"Fully parallel: all {m} X-stabilisers in round 0, "
                f"all {m} Z-stabilisers in round 0.  Offset-based "
                "scheduling omitted (circular geometry)."
            ),
        }
        meta["x_schedule"] = None  # circular geometry → matrix-based scheduling
        meta["z_schedule"] = None  # circular geometry → matrix-based scheduling

        # ── Literature / provenance ────────────────────────────────
        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/hamming_css"
        meta["wikipedia_url"] = "https://en.wikipedia.org/wiki/Hamming_code"
        meta["canonical_references"] = [
            "Steane, Proc. R. Soc. A 452, 2551 (1996). arXiv:quant-ph/9601029",
            "Calderbank & Shor, Phys. Rev. A 54, 1098 (1996). arXiv:quant-ph/9512032",
            "Hamming, Bell Syst. Tech. J. 29, 147-160 (1950).",
        ]
        meta["connections"] = [
            "CSS lift of classical [2^m-1, 2^m-m-1, 3] Hamming code",
            "m=3 instance is the Steane [[7,1,3]] code / triangular colour code",
            "Related to first-order quantum Reed-Muller codes",
            "Common inner code for concatenated fault-tolerant schemes",
        ]

        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)

        # Override parity check matrices
        self._hx = hx.astype(np.uint8)
        self._hz = hz.astype(np.uint8)

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'HammingCSS_m3'``."""
        return self._metadata.get("name", "HammingCSS")

    @property
    def distance(self) -> int:
        """Code distance (always 3 for Hamming CSS codes)."""
        return self._metadata.get("distance", 3)

    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates for visualisation (circular layout)."""
        return list(self.metadata.get("data_coords", []))


# Pre-built instances
HammingCSS7 = lambda: HammingCSSCode(m=3)   # [[7,1,3]] Steane
HammingCSS15 = lambda: HammingCSSCode(m=4)  # [[15,7,3]]
HammingCSS31 = lambda: HammingCSSCode(m=5)  # [[31,21,3]]
