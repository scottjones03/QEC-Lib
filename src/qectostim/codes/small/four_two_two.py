"""[[4, 2, 2]] Code — The Smallest Error-Detecting CSS Code

The [[4, 2, 2]] code, often called the **"Little Shor"** or **"smallest
interesting code"**, encodes 2 logical qubits in 4 physical qubits with
distance 2.  It is the smallest CSS code capable of detecting any
single-qubit error.

Construction
------------
The code has just one X-stabiliser and one Z-stabiliser, both weight 4:

    S_X = X₀X₁X₂X₃ = XXXX
    S_Z = Z₀Z₁Z₂Z₃ = ZZZZ

Because each stabiliser spans all 4 qubits, any single-qubit Pauli
error anticommutes with at least one stabiliser and is therefore
detected.

Code parameters
---------------
* **n** = 4  physical qubits
* **k** = 2  logical qubits  (n − rank(Hx) − rank(Hz) = 4 − 1 − 1 = 2)
* **d** = 2  (detects single errors; cannot correct)
* **Rate** R = 1/2

Logical operators
-----------------
Two anticommuting pairs (weight 2 each):

    X̄₁ = X₀X₁ (XXII)     Z̄₁ = Z₀Z₂ (ZIZI)
    X̄₂ = X₁X₂ (IXXI)     Z̄₂ = Z₁Z₃ (IZIZ)

Qubit layout
------------
Data qubits at corners of a unit square::

    3 --- 2
    |     |
    0 --- 1

Connections to other codes
--------------------------
* **Surface codes**: smallest patch of a rotated surface code (d = 2).
* **Bacon–Shor**: the [[4, 2, 2]] is a special case of a 2 × 2 Bacon–Shor
  code with all gauge operators fixed.
* **Quantum error detection**: foundational building block for flag-based
  fault-tolerant gadgets and post-selection experiments.

References
----------
* Vaidman, Goldenberg & Wiesner, "Error prevention scheme with four
  particles", Phys. Rev. A 54, R1745 (1996).
* Grassl, Beth & Pellizzari, "Codes for the quantum erasure channel",
  Phys. Rev. A 56, 33 (1997).  arXiv:quant-ph/9610042
* Error Correction Zoo: https://errorcorrectionzoo.org/c/stab_4_2_2
* Wikipedia: https://en.wikipedia.org/wiki/Quantum_error_correction
  (section on the [[4, 2, 2]] code)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..complexes.css_complex import CSSChainComplex3
from ..abstract_css import TopologicalCSSCode
from ..abstract_homological import Coord2D
from ..abstract_code import PauliString
from ..utils import validate_css_code


class FourQubit422Code(TopologicalCSSCode):
    """[[4, 2, 2]] code — the smallest error-detecting CSS code.

    Encodes 2 logical qubits in 4 physical qubits with distance 2.
    A single weight-4 X-stabiliser (XXXX) and a single weight-4
    Z-stabiliser (ZZZZ) detect any single-qubit Pauli error.

    Parameters
    ----------
    metadata : dict, optional
        Extra metadata merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (4).
    k : int
        Number of logical qubits (2).
    distance : int
        Code distance (2).
    hx : np.ndarray
        X-stabiliser parity-check matrix, shape ``(1, 4)``.
    hz : np.ndarray
        Z-stabiliser parity-check matrix, shape ``(1, 4)``.

    Examples
    --------
    >>> code = FourQubit422Code()
    >>> code.n, code.k, code.distance
    (4, 2, 2)

    Notes
    -----
    The code is self-dual (H_X and H_Z have the same support).  Its
    stabilisers span all 4 qubits, making it the densest CSS code
    for k = 2.

    See Also
    --------
    SixQubit622Code : Next-smallest k = 2 CSS code with the same distance.
    SteaneCode713 : Smallest single-error-*correcting* CSS code.
    """

    def __init__(self, *, metadata: Optional[Dict[str, Any]] = None):
        """Initialise the [[4, 2, 2]] code.

        Builds parity-check matrices, the 3-term chain complex, logical
        operators, and all standard metadata fields.

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
        # Hx: 1 X-stabiliser XXXX (weight 4)
        hx = np.array([[1, 1, 1, 1]], dtype=np.uint8)
        # Hz: 1 Z-stabiliser ZZZZ (weight 4)
        hz = np.array([[1, 1, 1, 1]], dtype=np.uint8)

        # ═══════════════════════════════════════════════════════════════════
        # CHAIN COMPLEX
        # ═══════════════════════════════════════════════════════════════════
        # C2 (1 X-stab) --∂2--> C1 (4 qubits) --∂1--> C0 (1 Z-stab)
        boundary_2 = hx.T.astype(np.uint8)  # shape (4, 1)
        boundary_1 = hz.astype(np.uint8)    # shape (1, 4)
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        # ═══════════════════════════════════════════════════════════════════
        # LOGICAL OPERATORS (2 anticommuting pairs, weight 2)
        # ═══════════════════════════════════════════════════════════════════
        # Pair 1:  Lx1 = X₀X₁ (XXII)  —  Lz1 = Z₀Z₂ (ZIZI)
        #   overlap({0,1}, {0,2}) = {0} → odd → anticommute ✓
        # Pair 2:  Lx2 = X₁X₂ (IXXI)  —  Lz2 = Z₁Z₃ (IZIZ)
        #   overlap({1,2}, {1,3}) = {1} → odd → anticommute ✓
        logical_x = [
            {0: 'X', 1: 'X'},  # Lx1 = XXII
            {1: 'X', 2: 'X'},  # Lx2 = IXXI
        ]
        logical_z = [
            {0: 'Z', 2: 'Z'},  # Lz1 = ZIZI
            {1: 'Z', 3: 'Z'},  # Lz2 = IZIZ
        ]

        # ═══════════════════════════════════════════════════════════════════
        # GEOMETRY — unit-square layout
        # ═══════════════════════════════════════════════════════════════════
        #   3 --- 2
        #   |     |
        #   0 --- 1
        data_coords: List[Coord2D] = [
            (0.0, 0.0),  # qubit 0
            (1.0, 0.0),  # qubit 1
            (1.0, 1.0),  # qubit 2
            (0.0, 1.0),  # qubit 3
        ]

        # ═══════════════════════════════════════════════════════════════════
        # METADATA (all 17 standard keys)
        # ═══════════════════════════════════════════════════════════════════
        meta: Dict[str, Any] = dict(metadata or {})
        meta["code_family"] = "small_css"
        meta["code_type"] = "four_two_two"
        meta["n"] = 4
        meta["k"] = 2
        meta["distance"] = 2
        meta["rate"] = 2.0 / 4.0
        meta["data_coords"] = data_coords
        meta["data_qubits"] = list(range(4))
        meta["x_stab_coords"] = [(0.5, 0.5)]   # face centre
        meta["z_stab_coords"] = [(0.5, -0.5)]   # below the square
        meta["lx_pauli_type"] = "X"
        meta["lx_support"] = [0, 1]           # first logical X: XXII
        meta["lz_pauli_type"] = "Z"
        meta["lz_support"] = [0, 2]           # first logical Z: ZIZI
        meta["x_logical_coords"] = [data_coords[0], data_coords[1]]
        meta["z_logical_coords"] = [data_coords[0], data_coords[2]]

        # Schedules
        meta["x_schedule"] = None  # single-stabiliser code → matrix scheduling
        meta["z_schedule"] = None
        meta["stabiliser_schedule"] = {
            "x_rounds": {0: 0},
            "z_rounds": {0: 0},
            "n_rounds": 1,
            "description": (
                "Trivial schedule: 1 X-stabiliser and 1 Z-stabiliser, "
                "both measured in round 0."
            ),
        }

        # ═══════════════════════════════════════════════════════════════════
        # LITERATURE / PROVENANCE
        # ═══════════════════════════════════════════════════════════════════
        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/stab_4_2_2"
        meta["wikipedia_url"] = "https://en.wikipedia.org/wiki/Quantum_error_correction"
        meta["canonical_references"] = [
            "Vaidman, Goldenberg & Wiesner, Phys. Rev. A 54, R1745 (1996)",
            "Grassl, Beth & Pellizzari, Phys. Rev. A 56, 33 (1997). arXiv:quant-ph/9610042",
        ]
        meta["connections"] = [
            "Smallest rotated surface code patch (d=2)",
            "2x2 Bacon-Shor code with gauge operators fixed",
            "Building block for flag-based fault-tolerant gadgets",
        ]

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, "FourQubit422Code", raise_on_error=True)

        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)

        # Override parity check matrices
        self._hx = hx.astype(np.uint8)
        self._hz = hz.astype(np.uint8)

    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates for visualisation (unit square)."""
        meta = getattr(self, "_metadata", {})
        return list(meta.get("data_coords", []))

    @property
    def distance(self) -> int:
        """Code distance (2)."""
        return 2

    @property
    def name(self) -> str:
        """Human-readable name: ``'FourQubit422Code'``."""
        return "FourQubit422Code"
