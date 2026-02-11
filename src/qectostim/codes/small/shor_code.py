"""[[9, 1, 3]] Shor Code — Concatenated Repetition Code

Shor's code is the **first quantum error-correcting code**, published
in 1995.  It is a [[9, 1, 3]] CSS code constructed by concatenating
a 3-qubit bit-flip repetition code (inner) with a 3-qubit phase-flip
repetition code (outer).  It can correct any single-qubit Pauli error
(distance 3).

Construction
------------
The 9 physical qubits are arranged in a 3 × 3 grid.  Each **row** is
an independent 3-qubit repetition code protecting against bit-flip
(X) errors, while the **columns** are linked by weight-6 X-type
stabilisers that detect phase-flip (Z) errors.

Stabilisers
-----------
* **Z-type** (6 generators, weight 2 each):
      Z₀Z₁, Z₁Z₂, Z₃Z₄, Z₄Z₅, Z₆Z₇, Z₇Z₈

* **X-type** (2 generators, weight 6 each):
      X₀X₁X₂X₃X₄X₅, X₃X₄X₅X₆X₇X₈

Logical operators (minimum weight)
-----------------------------------
    X̄ = X₀X₃X₆   (one qubit per row-block)
    Z̄ = Z₀Z₁Z₂   (entire first block)

Code parameters
---------------
* **n** = 9  physical qubits
* **k** = 1  logical qubit
* **d** = 3
* **Rate** R = 1/9 ≈ 0.111

Qubit layout
------------
Data qubits on a 3×3 grid.  Z-stabilisers are weight-2 checks within
each row; X-stabilisers are weight-6 checks spanning adjacent row-pairs.

::

            col 0     col 1     col 2
              │         │         │
    row 2:    6 ─ [Z₄] ─ 7 ─ [Z₅] ─ 8
              │         │         │
              ├─────────┴─────────┤
              │       [X₁]        │   X₁ = X₃X₄X₅X₆X₇X₈
              ├─────────┬─────────┤
    row 1:    3 ─ [Z₂] ─ 4 ─ [Z₃] ─ 5
              │         │         │
              ├─────────┴─────────┤
              │       [X₀]        │   X₀ = X₀X₁X₂X₃X₄X₅
              ├─────────┬─────────┤
    row 0:    0 ─ [Z₀] ─ 1 ─ [Z₁] ─ 2

    Data qubit coordinates (col, row):
      0: (0, 0)   1: (1, 0)   2: (2, 0)   — row 0
      3: (0, 1)   4: (1, 1)   5: (2, 1)   — row 1
      6: (0, 2)   7: (1, 2)   8: (2, 2)   — row 2

    Z-stabiliser coordinates (between adjacent qubits in row):
      Z₀: (0.5, 0)   Z₁: (1.5, 0)   — row 0
      Z₂: (0.5, 1)   Z₃: (1.5, 1)   — row 1
      Z₄: (0.5, 2)   Z₅: (1.5, 2)   — row 2

    X-stabiliser coordinates (between row pairs):
      X₀: (1.0, 0.5)   — between rows 0 and 1
      X₁: (1.0, 1.5)   — between rows 1 and 2

Encoding circuits
-----------------
|0⟩_L: Apply H to qubits {0, 3, 6}, then CNOT within each block to
create (|000⟩ + |111⟩)^⊗3 / 2√2.

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]] = [[9, 1, 3]]` where:

- :math:`n = 9` physical qubits (3 × 3 grid)
- :math:`k = 1` logical qubit
- :math:`d = 3` (corrects any single Pauli error)
- Rate :math:`k/n = 1/9 \approx 0.111`

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **X-type stabilisers**: 2 generators, each weight 6;
  ``X₀X₁X₂X₃X₄X₅`` and ``X₃X₄X₅X₆X₇X₈`` (span adjacent row-block pairs).
- **Z-type stabilisers**: 6 generators, each weight 2;
  ``ZᵢZᵢ₊₁`` within each row-block (adjacent-pair parity checks).
- Measurement schedule: Z-stabilisers are depth-2 circuits within each
  block; X-stabilisers are depth-6 circuits across blocks.

Connections to other codes
--------------------------
* **Repetition codes**: direct concatenation of two 3-qubit repetition
  codes (bit-flip inner, phase-flip outer).
* **Bacon–Shor codes**: the [[9, 1, 3]] Shor code is the smallest
  Bacon–Shor code (3 × 3 lattice with gauge operators fixed).
* **Surface codes**: can be viewed as a defective rotated surface code
  on a 3 × 3 patch.

References
----------
* Shor, "Scheme for reducing decoherence in quantum computer memory",
  Phys. Rev. A 52, R2493 (1995).
* Calderbank & Shor, Phys. Rev. A 54, 1098 (1996).
  arXiv:quant-ph/9512032
* Error Correction Zoo: https://errorcorrectionzoo.org/c/shor_nine
* Wikipedia: https://en.wikipedia.org/wiki/Shor%27s_nine-qubit_code
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex3
from qectostim.codes.utils import validate_css_code


class ShorCode91(TopologicalCSSCode):
    """[[9, 1, 3]] Shor code — the first quantum error-correcting code.

    Encodes 1 logical qubit in 9 physical qubits with distance 3 via
    concatenation of 3-qubit bit-flip and phase-flip repetition codes.

    Parameters
    ----------
    metadata : dict, optional
        Extra metadata merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (9).
    k : int
        Number of logical qubits (1).
    distance : int
        Code distance (3).
    hx : np.ndarray
        X-stabiliser parity-check matrix, shape ``(2, 9)``.
    hz : np.ndarray
        Z-stabiliser parity-check matrix, shape ``(6, 9)``.

    Examples
    --------
    >>> code = ShorCode91()
    >>> code.n, code.k, code.distance
    (9, 1, 3)

    Notes
    -----
    Qubits are arranged in a 3 × 3 grid::

        0  1  2
        3  4  5
        6  7  8

    Z-stabilisers are weight-2 parity checks *within* each row-block.
    X-stabilisers are weight-6 parity checks *across* adjacent blocks.

    See Also
    --------
    RepetitionCode : 1D repetition code (inner building block).
    BaconShorCode : Subsystem generalisation of the Shor code.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialise the [[9, 1, 3]] Shor code.

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
        ValueError
            If the internally constructed parity-check matrices fail
            CSS validation (should never happen for a fixed code).
        """

        # ═══════════════════════════════════════════════════════════════════
        # SHOR CODE STABILIZER STRUCTURE (Standard CSS Convention)
        # ═══════════════════════════════════════════════════════════════════
        # Shor code = concatenation of:
        #   Inner: 3-qubit bit-flip repetition code (protects against X errors)
        #   Outer: 3-qubit phase-flip repetition code (protects against Z errors)
        #
        # This creates 9 qubits organized in 3 blocks of 3:
        #   Block 0: qubits 0,1,2
        #   Block 1: qubits 3,4,5
        #   Block 2: qubits 6,7,8
        #
        # Using STANDARD CSS CONVENTION:
        #   Hx defines X-type stabilizers (detect Z errors)
        #   Hz defines Z-type stabilizers (detect X errors)
        # ═══════════════════════════════════════════════════════════════════

        # X-type stabilizers (hx): 2 weight-6 checks ACROSS blocks
        # These are X-Pauli operators that detect Z errors
        hx = np.array([
            [1, 1, 1, 1, 1, 1, 0, 0, 0],  # X_0 X_1 X_2 X_3 X_4 X_5
            [0, 0, 0, 1, 1, 1, 1, 1, 1],  # X_3 X_4 X_5 X_6 X_7 X_8
        ], dtype=np.uint8)

        # Z-type stabilizers (hz): 6 weight-2 checks WITHIN each block
        # These are Z-Pauli operators that detect X errors (bit-flips)
        # Rows: {0,1}, {1,2} (block 0)
        #       {3,4}, {4,5} (block 1)
        #       {6,7}, {7,8} (block 2)
        hz = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0],  # Z_0 Z_1
            [0, 1, 1, 0, 0, 0, 0, 0, 0],  # Z_1 Z_2
            [0, 0, 0, 1, 1, 0, 0, 0, 0],  # Z_3 Z_4
            [0, 0, 0, 0, 1, 1, 0, 0, 0],  # Z_4 Z_5
            [0, 0, 0, 0, 0, 0, 1, 1, 0],  # Z_6 Z_7
            [0, 0, 0, 0, 0, 0, 0, 1, 1],  # Z_7 Z_8
        ], dtype=np.uint8)

        # Build chain complex for CSS code structure:
        #   C2 (X stabilizers) --∂2--> C1 (qubits) --∂1--> C0 (Z stabilizers)
        #
        # boundary_2 = Hx.T: maps faces (X stabs) → edges (qubits)
        # boundary_1 = Hz:   maps edges (qubits) → vertices (Z stabs)
        boundary_2 = hx.T.astype(np.uint8)  # shape (9, 2)
        boundary_1 = hz.astype(np.uint8)    # shape (6, 9)
        
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        # ═══════════════════════════════════════════════════════════════════
        # SHOR CODE LOGICAL OPERATORS (Standard CSS Convention)
        # ═══════════════════════════════════════════════════════════════════
        # Using standard CSS convention where:
        #   Logical X is X-type (X on qubits 0, 3, 6 - one per block)
        #   Logical Z is Z-type (Z on qubits 0, 1, 2 - first block)
        #
        # This matches the standard definition where Lz commutes with Hz
        # and Lx commutes with Hx.
        # ═══════════════════════════════════════════════════════════════════
        
        # Logical X: X on one qubit per block (qubits 0, 3, 6)
        # Qubits: 0  1  2  3  4  5  6  7  8
        #         X  I  I  X  I  I  X  I  I
        logical_x = ["XIIXIIXII"]
        
        # Logical Z: Z on first block (qubits 0, 1, 2)
        # Qubits: 0  1  2  3  4  5  6  7  8
        #         Z  Z  Z  I  I  I  I  I  I
        logical_z = ["ZZZIIIIII"]

        # 3x3 grid coordinates
        coords = {q: (float(q % 3), float(q // 3)) for q in range(9)}
        data_coords = [coords[i] for i in range(9)]
        
        # X stabilizer coordinates (between rows)
        x_stab_coords = [(1.0, 0.5), (1.0, 1.5)]  # between row 0-1, row 1-2
        # Z stabilizer coordinates (within each row, between adjacent qubits)
        z_stab_coords = [
            (0.5, 0.0), (1.5, 0.0),  # row 0
            (0.5, 1.0), (1.5, 1.0),  # row 1
            (0.5, 2.0), (1.5, 2.0),  # row 2
        ]

        meta = dict(metadata or {})
        meta["name"] = "ShorCode91"
        meta["code_family"] = "small_css"
        meta["code_type"] = "shor"
        meta["n"] = 9
        meta["k"] = 1
        meta["distance"] = 3
        meta["rate"] = 1.0 / 9.0
        meta["data_coords"] = data_coords
        meta["x_stab_coords"] = x_stab_coords
        meta["z_stab_coords"] = z_stab_coords
        meta["data_qubits"] = list(range(9))
        meta["x_logical_coords"] = [data_coords[i] for i in [0, 3, 6]]
        meta["z_logical_coords"] = [data_coords[i] for i in [0, 1, 2]]
        
        # ═══════════════════════════════════════════════════════════════════
        # LOGICAL OPERATOR PAULI TYPES (Standard CSS Convention)
        # ═══════════════════════════════════════════════════════════════════
        # Using standard CSS convention:
        #   Logical Z is Z-type (Z on qubits 0, 1, 2)
        #   Logical X is X-type (X on qubits 0, 3, 6)
        #
        # This means:
        # - To measure Lz, we need Z-basis measurements (M), standard path
        # - Lz commutes with Hz (Z-type stabilizers)
        # - Lx commutes with Hx (X-type stabilizers)
        # ═══════════════════════════════════════════════════════════════════
        meta["lz_pauli_type"] = "Z"  # Lz is Z-type (standard)
        meta["lz_support"] = [0, 1, 2]  # Z₀Z₁Z₂
        
        meta["lx_pauli_type"] = "X"  # Lx is X-type (standard)
        meta["lx_support"] = [0, 3, 6]  # X₀X₃X₆
        
        # Measurement schedules
        meta["x_schedule"] = [(0.0, 0.5), (1.0, 0.5), (2.0, 0.5)]  # 3 qubits per X stab
        meta["z_schedule"] = [(0.5, 0.0), (-0.5, 0.0)]  # 2 qubits per Z stab

        # ═══════════════════════════════════════════════════════════════════
        # STABILISER SCHEDULE
        # ═══════════════════════════════════════════════════════════════════
        # Z-stabilisers: 6 weight-2 checks (3 pairs per block, all parallel)
        # X-stabilisers: 2 weight-6 checks (between blocks, all parallel)
        meta["stabiliser_schedule"] = {
            "x_rounds": {0: 0, 1: 0},
            "z_rounds": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "n_rounds": 1,
            "description": (
                "Fully parallel: all 2 X-stabilisers in round 0, "
                "all 6 Z-stabilisers in round 0."
            ),
        }

        # ═══════════════════════════════════════════════════════════════════
        # LITERATURE / PROVENANCE
        # ═══════════════════════════════════════════════════════════════════
        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/shor_nine"
        meta["wikipedia_url"] = "https://en.wikipedia.org/wiki/Shor%27s_nine-qubit_code"
        meta["canonical_references"] = [
            "Shor, Phys. Rev. A 52, R2493 (1995)",
            "Calderbank & Shor, Phys. Rev. A 54, 1098 (1996). arXiv:quant-ph/9512032",
        ]
        meta["connections"] = [
            "Concatenation of 3-qubit bit-flip and phase-flip repetition codes",
            "Smallest Bacon-Shor code (3x3 lattice)",
            "Defective rotated surface code on 3x3 patch",
        ]
        
        # ═══════════════════════════════════════════════════════════════════
        # |0⟩_L AND |+⟩_L ENCODING FOR NON-SELF-DUAL CODES
        # ═══════════════════════════════════════════════════════════════════
        # Shor code is NOT self-dual (Hx ≠ Hz), so transversal H ≠ logical H.
        # We must define explicit encoding circuits.
        #
        # Shor codewords are SUPERPOSITION states (not computational basis):
        #   |0⟩_L = (|000⟩ + |111⟩)^⊗3 / 2√2  (GHZ-like per block)
        #   |1⟩_L = (|000⟩ - |111⟩)^⊗3 / 2√2
        #
        # Encoding circuit for |0⟩_L (same as |+⟩_L since both use GHZ per block):
        #   1. Reset all qubits to |0⟩
        #   2. Apply H to first qubit of each block (qubits 0, 3, 6)
        #   3. CNOT to spread within each block
        #
        # This creates: |0⟩ → H → (|0⟩+|1⟩) → CNOT → (|00⟩+|11⟩) → CNOT → (|000⟩+|111⟩)
        # Per block, giving |0⟩_L = (|000⟩+|111⟩)^⊗3 / 2√2
        #
        # Note: |+⟩_L requires correct phase coherence between blocks, which
        # is achieved by the same encoding followed by appropriate logical
        # operations. For FT prep, Shor verification handles the phases.
        # ═══════════════════════════════════════════════════════════════════
        
        # |0⟩_L encoding (creates GHZ-like superposition per block)
        meta["zero_h_qubits"] = [0, 3, 6]  # H on first qubit of each row/block
        meta["zero_encoding_cnots"] = [
            (0, 1), (0, 2),  # Block 0: qubit 0 controls 1, 2
            (3, 4), (3, 5),  # Block 1: qubit 3 controls 4, 5  
            (6, 7), (6, 8),  # Block 2: qubit 6 controls 7, 8
        ]
        
        # |+⟩_L encoding (same circuit as |0⟩_L for Shor)
        meta["plus_h_qubits"] = [0, 3, 6]  # H on first qubit of each row/block
        meta["plus_encoding_cnots"] = [
            (0, 1), (0, 2),  # Block 0: qubit 0 controls 1, 2
            (3, 4), (3, 5),  # Block 1: qubit 3 controls 4, 5  
            (6, 7), (6, 8),  # Block 2: qubit 6 controls 7, 8
        ]

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, "ShorCode91", raise_on_error=True)

        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)
        
        # Override parity check matrices
        self._hx = hx.astype(np.uint8)
        self._hz = hz.astype(np.uint8)
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates for visualisation (3 × 3 grid)."""
        return list(self.metadata.get("data_coords", []))

    @property
    def distance(self) -> int:
        """Code distance (3)."""
        return 3

    @property
    def name(self) -> str:
        """Human-readable name: ``'ShorCode91'``."""
        return "ShorCode91"
