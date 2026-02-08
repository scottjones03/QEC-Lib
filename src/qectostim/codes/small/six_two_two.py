# src/qectostim/codes/small/six_two_two.py
"""[[6, 2, 2]] CSS Code — The "Iceberg" Code

The [[6, 2, 2]] code encodes **2 logical qubits** in **6 physical qubits**
with distance 2.  It is the **smallest CSS code with k > 1** and is
sometimes called the "iceberg" code.

Construction
------------
The stabiliser group has 4 independent generators:

* **X-type** (2 generators, weight 4 each):
      X₀X₁X₂X₃ = XXXXII
      X₀X₁X₄X₅ = XXIIXX

* **Z-type** (2 generators, weight 3 each):
      Z₀Z₂Z₄ = ZIZIZI
      Z₁Z₃Z₅ = IZIZIZ

CSS orthogonality: each X-row overlaps each Z-row in exactly 2 positions
(even), so Hx·Hz^T = 0 (mod 2).

Code parameters
---------------
* **n** = 6  physical qubits
* **k** = 2  logical qubits  (6 − 2 − 2 = 2)
* **d** = 2  (detects single errors; cannot correct)
* **Rate** R = 1/3

Logical operators
-----------------
Two anticommuting pairs (weight 2):

    X̄₁ = X₀X₂ (XIXIII)     Z̄₁ = Z₀Z₁ (ZZIIII)
    X̄₂ = X₃X₅ (IIIXIX)     Z̄₂ = Z₄Z₅ (IIIIZZ)

Key properties:
- Detects any single-qubit error (distance 2)
- Transversal Hadamard swaps the two logical qubits

Connections to other codes
--------------------------
* **[[4, 2, 2]] code**: obtained by puncturing two qubits.
* **Colour codes**: related to small colour code constructions.
* **Concatenated codes**: can serve as outer code in concatenated schemes.

References
----------
* Knill, "Benchmarking Quantum Computers and the Approach to Scalable
  Quantum Computing", arXiv:quant-ph/0404104 (2004).
* Chao & Reichardt, "Quantum error correction with only two extra qubits",
  Phys. Rev. Lett. 121, 050502 (2018).
* Error Correction Zoo: https://errorcorrectionzoo.org/c/stab_6_2_2
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex3
from qectostim.codes.utils import validate_css_code

Coord2D = Tuple[float, float]


class SixQubit622Code(CSSCode):
    """[[6, 2, 2]] CSS code (iceberg code).

    Encodes 2 logical qubits in 6 physical qubits with distance 2.
    This is the smallest CSS code with k > 1.

    Parameters
    ----------
    metadata : dict, optional
        Extra metadata merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (6).
    k : int
        Number of logical qubits (2).
    distance : int
        Code distance (2).
    hx : np.ndarray
        X-stabiliser parity-check matrix, shape ``(2, 6)``.
    hz : np.ndarray
        Z-stabiliser parity-check matrix, shape ``(2, 6)``.

    Examples
    --------
    >>> code = SixQubit622Code()
    >>> code.n, code.k, code.distance
    (6, 2, 2)

    Notes
    -----
    Transversal Hadamard on all 6 qubits swaps the two logical qubits
    (since Hx and Hz have different support, this is a non-trivial
    logical operation).

    See Also
    --------
    FourQubit422Code : Smaller k = 2 code (same distance).
    EightThreeTwoCode : Larger k = 3 code (same distance).
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialise the [[6, 2, 2]] code.

        Builds parity-check matrices, chain complex, logical operators,
        and all standard metadata fields.

        Parameters
        ----------
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata
            dictionary.  User-supplied entries override auto-generated
            ones with the same key.
        """
        # ═══════════════════════════════════════════════════════════════════
        # STABILISER MATRICES
        # ═══════════════════════════════════════════════════════════════════
        # X-type stabilisers (2 generators, weight 4)
        hx = np.array([
            [1, 1, 1, 1, 0, 0],  # XXXXII — qubits {0,1,2,3}
            [1, 1, 0, 0, 1, 1],  # XXIIXX — qubits {0,1,4,5}
        ], dtype=np.uint8)

        # Z-type stabilisers (2 generators, weight 3)
        # Designed so Hx @ Hz^T = 0 (mod 2):
        #   Row 0 ∩ Row 0: {0,1,2,3} ∩ {0,2,4} = {0,2} → 2 (even) ✓
        #   Row 0 ∩ Row 1: {0,1,2,3} ∩ {1,3,5} = {1,3} → 2 (even) ✓
        #   Row 1 ∩ Row 0: {0,1,4,5} ∩ {0,2,4} = {0,4} → 2 (even) ✓
        #   Row 1 ∩ Row 1: {0,1,4,5} ∩ {1,3,5} = {1,5} → 2 (even) ✓
        hz = np.array([
            [1, 0, 1, 0, 1, 0],  # ZIZIZI — qubits {0,2,4}
            [0, 1, 0, 1, 0, 1],  # IZIZIZ — qubits {1,3,5}
        ], dtype=np.uint8)

        # ═══════════════════════════════════════════════════════════════════
        # CHAIN COMPLEX
        # ═══════════════════════════════════════════════════════════════════
        boundary_2 = hx.T.astype(np.uint8)  # shape (6, 2)
        boundary_1 = hz.astype(np.uint8)    # shape (2, 6)
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        # ═══════════════════════════════════════════════════════════════════
        # LOGICAL OPERATORS (2 anticommuting pairs, weight 2)
        # ═══════════════════════════════════════════════════════════════════
        # Lx must be in ker(Hz): Hz·v = 0 mod 2
        #   Row 0 constraint: v₀+v₂+v₄ = 0   Row 1 constraint: v₁+v₃+v₅ = 0
        # Lz must be in ker(Hx): Hx·v = 0 mod 2
        #   Row 0 constraint: v₀+v₁+v₂+v₃ = 0   Row 1 constraint: v₀+v₁+v₄+v₅ = 0
        #
        # Pair 1:  Lx1 = X₀X₂ (XIXIII)  —  Lz1 = Z₀Z₁ (ZZIIII)
        #   Lx1 in ker(Hz): 1+1+0 = 0 ✓, 0+0+0 = 0 ✓
        #   Lz1 in ker(Hx): 1+1+0+0 = 0 ✓, 1+1+0+0 = 0 ✓
        #   overlap({0,2}, {0,1}) = {0} → 1 (odd) → anticommute ✓
        #
        # Pair 2:  Lx2 = X₃X₅ (IIIXIX)  —  Lz2 = Z₄Z₅ (IIIIZZ)
        #   Lx2 in ker(Hz): 0+0+0 = 0 ✓, 1+0+1 = 0 ✓
        #   Lz2 in ker(Hx): 0+0+0+0 = 0 ✓, 0+0+1+1 = 0 ✓
        #   overlap({3,5}, {4,5}) = {5} → 1 (odd) → anticommute ✓
        #
        # Cross:  Lx1∩Lz2 = {0,2}∩{4,5} = ∅ → 0 (even) → commute ✓
        #         Lx2∩Lz1 = {3,5}∩{0,1} = ∅ → 0 (even) → commute ✓
        logical_x: List[PauliString] = [
            "XIXIII",  # Lx1: X₀X₂
            "IIIXIX",  # Lx2: X₃X₅
        ]
        logical_z: List[PauliString] = [
            "ZZIIII",  # Lz1: Z₀Z₁
            "IIIIZZ",  # Lz2: Z₄Z₅
        ]

        # ═══════════════════════════════════════════════════════════════════
        # GEOMETRY — 2×3 grid
        # ═══════════════════════════════════════════════════════════════════
        data_coords = [
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
            (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),
        ]

        # ═══════════════════════════════════════════════════════════════════
        # METADATA (all 17 standard keys)
        # ═══════════════════════════════════════════════════════════════════
        meta = dict(metadata or {})
        meta["code_family"] = "small_css"
        meta["code_type"] = "six_two_two"
        meta["n"] = 6
        meta["k"] = 2
        meta["distance"] = 2
        meta["rate"] = 2.0 / 6.0
        meta["data_coords"] = data_coords
        meta["data_qubits"] = list(range(6))
        meta["x_stab_coords"] = [(0.5, -0.5), (1.5, 0.5)]
        meta["z_stab_coords"] = [(0.5, 0.5), (1.5, 1.5)]
        meta["lx_pauli_type"] = "X"
        meta["lx_support"] = [0, 2]           # first logical X: X₀X₂
        meta["lz_pauli_type"] = "Z"
        meta["lz_support"] = [0, 1]           # first logical Z: Z₀Z₁
        meta["x_logical_coords"] = [data_coords[0], data_coords[2]]
        meta["z_logical_coords"] = [data_coords[0], data_coords[1]]

        # Schedules
        meta["x_schedule"] = None  # small code → matrix scheduling
        meta["z_schedule"] = None
        meta["stabiliser_schedule"] = {
            "x_rounds": {0: 0, 1: 0},
            "z_rounds": {0: 0, 1: 0},
            "n_rounds": 1,
            "description": (
                "Fully parallel: 2 X-stabilisers and 2 Z-stabilisers "
                "all measured in round 0."
            ),
        }

        # ═══════════════════════════════════════════════════════════════════
        # LITERATURE / PROVENANCE
        # ═══════════════════════════════════════════════════════════════════
        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/stab_6_2_2"
        meta["wikipedia_url"] = "https://en.wikipedia.org/wiki/Quantum_error_correction"
        meta["canonical_references"] = [
            "Knill, arXiv:quant-ph/0404104 (2004)",
            "Chao & Reichardt, Phys. Rev. Lett. 121, 050502 (2018)",
        ]
        meta["connections"] = [
            "Smallest CSS code with k > 1",
            "Obtained by extending [[4,2,2]] with 2 extra qubits",
            "Transversal Hadamard swaps the two logical qubits",
        ]

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, "SixQubit622Code", raise_on_error=True)

        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )

    @property
    def name(self) -> str:
        """Human-readable name: ``'SixQubit622Code'``."""
        return "SixQubit622Code"

    @property
    def distance(self) -> int:
        """Code distance (2)."""
        return 2

    def qubit_coords(self) -> List[Coord2D]:
        """Return 2D coordinates for each data qubit (2 × 3 grid)."""
        return [
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
            (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),
        ]