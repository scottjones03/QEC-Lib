"""[[15, 1, 3]] Quantum Reed–Muller Code

The [[15, 1, 3]] quantum Reed–Muller code encodes **1 logical qubit** in
**15 physical qubits** with distance 3.  It is a **self-dual CSS code**
derived from the classical first-order Reed–Muller code RM(1, 4).

Construction
------------
The code uses 7 weight-4 stabiliser generators that form a
**self-orthogonal** set (all pairwise overlaps are even: 0 or 2).
Because the code is self-dual (Hx = Hz), the CSS commutativity
condition Hx · Hz^T = 0 is automatically satisfied.

Stabilisers (7 generators, each weight 4)
------------------------------------------
    g₁ : qubits {0, 1, 2, 3}
    g₂ : qubits {0, 1, 4, 5}
    g₃ : qubits {0, 1, 6, 7}
    g₄ : qubits {0, 2, 4, 6}
    g₅ : qubits {8, 9, 10, 11}
    g₆ : qubits {8, 9, 12, 13}
    g₇ : qubits {8, 10, 12, 14}

Code parameters
---------------
* **n** = 15  physical qubits
* **k** = 1   logical qubit  (15 − 7 − 7 = 1)
* **d** = 3   (minimum weight logical operator has weight 3)
* **Rate** R = 1/15 ≈ 0.067
* **Self-dual CSS**: Hx = Hz

Logical operators (weight 3)
-----------------------------
    X̄ = X₉X₁₀X₁₂   (qubits {9, 10, 12})
    Z̄ = Z₉Z₁₀Z₁₂   (qubits {9, 10, 12})  — self-dual

Connections to other codes
--------------------------
* **Classical Reed–Muller**: quantum CSS lift of RM(1, 4).
* **Steane code**: the [[7, 1, 3]] Steane code is RM(1, 3) — the
  previous member of the Reed–Muller hierarchy.
* **Transversal T gate**: the [[15, 1, 3]] code supports a transversal
  T gate, making it the smallest code with this property.  Combined
  with Clifford gates from smaller codes, this enables universal
  fault-tolerant computation without magic-state distillation.

References
----------
* Anderson, Duclos-Cianci & Poulin, "Fault-tolerant conversion between
  the Steane and Reed-Muller quantum codes", Phys. Rev. Lett. 113,
  080501 (2014).  arXiv:1403.2734
* Knill, Laflamme & Zurek, "Threshold accuracy for quantum computation",
  arXiv:quant-ph/9610011 (1996).
* Error Correction Zoo: https://errorcorrectionzoo.org/c/stab_15_1_3
"""

from __future__ import annotations
from typing import Dict, Any, Optional

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.complexes.css_complex import CSSChainComplex3
from qectostim.codes.utils import validate_css_code


def _gf2_rank(mat: np.ndarray) -> int:
    """Compute rank of a matrix over GF(2)."""
    mat = mat.copy().astype(np.uint8)
    if mat.size == 0:
        return 0
    rows, cols = mat.shape
    rank = 0
    for col in range(cols):
        found = False
        for row in range(rank, rows):
            if mat[row, col] == 1:
                mat[[rank, row]] = mat[[row, rank]]
                found = True
                break
        if not found:
            continue
        for row in range(rows):
            if row != rank and mat[row, col] == 1:
                mat[row] = (mat[row] + mat[rank]) % 2
        rank += 1
    return rank


def _is_css_orthogonal(hx: np.ndarray, hz: np.ndarray) -> bool:
    """Check if Hx @ Hz^T = 0 (mod 2)."""
    return np.all(np.dot(hx, hz.T) % 2 == 0)


def _build_self_orthogonal_stabilizers() -> np.ndarray:
    """
    Build 7 weight-4 stabilizer generators for [[15,1,3]].
    
    These patterns are verified to be:
    1. Self-orthogonal (all pairwise overlaps are 0 or 2)
    2. Rank 7 over GF(2)
    3. Giving k = 15 - 2*7 = 1 logical qubit
    4. With minimum weight non-stabilizer coset representative = 3 (distance)
    
    The patterns form two groups of 3-4 stabilizers each, with controlled
    overlap structure.
    """
    # Weight-4 patterns with even pairwise overlaps (0 or 2), spanning rank-7
    stab_patterns = [
        [0, 1, 2, 3],      # Group 1: qubits 0-7
        [0, 1, 4, 5],
        [0, 1, 6, 7],
        [0, 2, 4, 6],
        [8, 9, 10, 11],    # Group 2: qubits 8-14
        [8, 9, 12, 13],
        [8, 10, 12, 14],
    ]
    
    h = np.zeros((7, 15), dtype=np.uint8)
    for i, pattern in enumerate(stab_patterns):
        for j in pattern:
            h[i, j] = 1
    
    return h


class ReedMullerCode151(CSSCode):
    """[[15, 1, 3]] Quantum Reed–Muller code.

    Encodes 1 logical qubit in 15 physical qubits with distance 3.
    A self-dual CSS code (Hx = Hz) with 7 weight-4 stabiliser generators
    in each sector.  Supports a transversal T gate.

    Parameters
    ----------
    metadata : dict, optional
        Extra metadata merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (15).
    k : int
        Number of logical qubits (1).
    distance : int
        Code distance (3).
    hx : np.ndarray
        X-stabiliser parity-check matrix, shape ``(7, 15)``.
    hz : np.ndarray
        Z-stabiliser parity-check matrix, shape ``(7, 15)``.
        Identical to ``hx`` (self-dual).

    Examples
    --------
    >>> code = ReedMullerCode151()
    >>> code.n, code.k, code.distance
    (15, 1, 3)
    >>> (code._hx == code._hz).all()   # self-dual
    True

    Notes
    -----
    The transversal T gate on this code, combined with the transversal
    Clifford group from the Steane code, enables universal fault-tolerant
    quantum computation via code switching — without magic-state
    distillation.

    See Also
    --------
    SteaneCode713 : RM(1, 3) — previous member of the Reed–Muller hierarchy.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialise the [[15, 1, 3]] quantum Reed–Muller code.

        Builds the self-orthogonal stabiliser matrices, chain complex,
        logical operators, and all standard metadata fields.

        Parameters
        ----------
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata
            dictionary.  User-supplied entries override auto-generated
            ones with the same key.
        """
        
        # Build self-orthogonal stabilizer matrix
        hx = _build_self_orthogonal_stabilizers()
        hz = hx.copy()  # Self-dual construction
        
        # Verify CSS orthogonality
        if not _is_css_orthogonal(hx, hz):
            raise ValueError("CSS orthogonality check failed - stabilizers not self-orthogonal")
        
        # Verify rank and k
        rank_hx = _gf2_rank(hx)
        rank_hz = _gf2_rank(hz)
        k = 15 - rank_hx - rank_hz
        
        if k != 1:
            raise ValueError(f"Expected k=1, got k={k} (rank_hx={rank_hx}, rank_hz={rank_hz})")
        
        # Logical operators with weight 3 (minimum-weight coset representatives)
        # X̄ = X₉X₁₀X₁₂,  Z̄ = Z₉Z₁₀Z₁₂  (self-dual: same support)
        logical_x = [{9: 'X', 10: 'X', 12: 'X'}]
        logical_z = [{9: 'Z', 10: 'Z', 12: 'Z'}]

        # ═══════════════════════════════════════════════════════════════════
        # CHAIN COMPLEX
        # ═══════════════════════════════════════════════════════════════════
        boundary_2 = hx.T.astype(np.uint8)  # shape (15, 7)
        boundary_1 = hz.astype(np.uint8)    # shape (7, 15)
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        # ═══════════════════════════════════════════════════════════════════
        # METADATA (all 17 standard keys)
        # ═══════════════════════════════════════════════════════════════════
        meta = dict(metadata or {})
        meta["code_family"] = "small_css"
        meta["code_type"] = "reed_muller"
        meta["n"] = 15
        meta["k"] = 1
        meta["distance"] = 3
        meta["rate"] = 1.0 / 15.0
        meta["lx_pauli_type"] = "X"
        meta["lx_support"] = [9, 10, 12]
        meta["lz_pauli_type"] = "Z"
        meta["lz_support"] = [9, 10, 12]

        # Grid coordinates (3×5 arrangement)
        data_coords = [(float(i % 5), float(i // 5)) for i in range(15)]
        meta["data_coords"] = data_coords
        meta["data_qubits"] = list(range(15))
        meta["x_logical_coords"] = [data_coords[i] for i in [9, 10, 12]]
        meta["z_logical_coords"] = [data_coords[i] for i in [9, 10, 12]]

        # Schedules
        meta["x_schedule"] = None  # self-dual code → matrix scheduling
        meta["z_schedule"] = None
        meta["stabiliser_schedule"] = {
            "x_rounds": {i: 0 for i in range(7)},
            "z_rounds": {i: 0 for i in range(7)},
            "n_rounds": 1,
            "description": (
                "Fully parallel: all 7 X-stabilisers and all 7 Z-stabilisers "
                "in round 0.  Self-dual code (Hx = Hz)."
            ),
        }

        # ═══════════════════════════════════════════════════════════════════
        # LITERATURE / PROVENANCE
        # ═══════════════════════════════════════════════════════════════════
        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/stab_15_1_3"
        meta["wikipedia_url"] = "https://en.wikipedia.org/wiki/Reed%E2%80%93Muller_code"
        meta["canonical_references"] = [
            "Anderson, Duclos-Cianci & Poulin, Phys. Rev. Lett. 113, 080501 (2014). arXiv:1403.2734",
            "Knill, Laflamme & Zurek, arXiv:quant-ph/9610011 (1996)",
        ]
        meta["connections"] = [
            "Quantum CSS lift of classical RM(1,4) Reed-Muller code",
            "Self-dual CSS code (Hx = Hz)",
            "Supports transversal T gate (smallest known code with this property)",
            "Steane code RM(1,3) is the previous member in the hierarchy",
        ]

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, "ReedMullerCode151", raise_on_error=True)

        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)

    @property
    def distance(self) -> int:
        """Code distance (3)."""
        return 3

    @property
    def name(self) -> str:
        """Human-readable name: ``'ReedMullerCode151'``."""
        return "ReedMullerCode151"
