"""[[5, 1, 3]] Perfect Code — The Smallest Quantum Error-Correcting Code

The [[5, 1, 3]] perfect code is the **smallest quantum error-correcting code**
that can correct any single-qubit error.  It is called "perfect" because it
saturates the **quantum Hamming bound**: every syndrome maps to exactly one
single-qubit error.

Unlike CSS codes, the stabilisers mix X and Z operators.  The code has
4 stabiliser generators of weight 4, encodes 1 logical qubit in 5 physical
qubits, and has distance 3.

Construction (cyclic)
---------------------
The stabiliser generators are cyclic shifts of XZZXI:

    g₁ = X Z Z X I
    g₂ = I X Z Z X
    g₃ = X I X Z Z
    g₄ = Z X I X Z

Code parameters
---------------
* **n** = 5  physical qubits
* **k** = 1  logical qubit
* **d** = 3  (corrects any single Pauli error)
* **Rate** R = 1/5 = 0.2
* **Non-CSS**: stabilisers mix X and Z operators

Logical operators (weight 5, transversal)
-----------------------------------------
    X̄ = XXXXX     Z̄ = ZZZZZ

Qubit layout
------------
Five qubits placed at the vertices of a regular pentagon, reflecting
the cyclic symmetry of the code.  No X/Z stabiliser separation exists
since this is a non-CSS code.

::

              0
            ╱   ╲
          ╱       ╲
        4           1
        │           │
        │           │
        3 ───────── 2

    Data qubit coordinates (unit pentagon, centred at origin):
      0: (0.0, 1.0)       — top vertex
      1: (0.951, 0.309)   — upper-right
      2: (0.588, −0.809)  — lower-right
      3: (−0.588, −0.809) — lower-left
      4: (−0.951, 0.309)  — upper-left

    (Coordinates are cos/sin of angles π/2 + 2πk/5 for k=0,1,2,3,4)

    Stabiliser positions are not well-defined geometrically for
    non-CSS codes (stabilisers mix X and Z on overlapping qubits).

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]] = [[5, 1, 3]]` where:

- :math:`n = 5` physical qubits (vertices of a regular pentagon)
- :math:`k = 1` logical qubit
- :math:`d = 3` (corrects any single Pauli error; saturates the quantum Hamming bound)
- Rate :math:`k/n = 1/5 = 0.2`

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **Stabiliser generators** (non-CSS — mixed X/Z): 4 generators, each
  weight 4, arranged cyclically:
  ``XZZXI``, ``IXZZX``, ``XIXZZ``, ``ZXIXZ``.
- No X-type / Z-type separation (non-CSS code).
- Measurement schedule: all 4 stabilisers measured with joint X/Z
  ancilla circuits; each requires a depth-4 entangling circuit.

Connections to other codes
--------------------------
* **Quantum Hamming bound**: saturates the bound with equality.
* **Stabiliser codes**: the smallest non-trivial stabiliser code.
* **Laflamme code**: sometimes called the Laflamme code after one of
  its discoverers.

Fault tolerance
---------------
* All 15 non-trivial single-qubit errors map to distinct syndromes,
  confirming the perfect-code property (4 syndrome bits → 15 outcomes).
* Transversal gates are limited; the code does **not** support a
  transversal Clifford gate set.
* Fault-tolerant error correction requires Knill or Steane-type
  ancilla-verification gadgets.

Decoding
--------
* Syndrome lookup is the simplest decoder: a 15-entry table maps each
  non-zero syndrome to the unique correctable single-qubit Pauli error.
* Because the code is non-CSS, X and Z syndromes cannot be decoded
  independently; the full 4-bit syndrome must be used jointly.
* For concatenated [[5,1,3]] codes, level-by-level lookup decoding
  achieves the threshold theorem's doubly-exponential suppression.

References
----------
* Laflamme, Miquel, Paz & Zurek, "Perfect quantum error correcting
  code", Phys. Rev. Lett. 77, 198 (1996).  arXiv:quant-ph/9602019
* Bennett, DiVincenzo, Smolin & Wootters, "Mixed-state entanglement
  and quantum error correction", Phys. Rev. A 54, 3824 (1996).
  arXiv:quant-ph/9604024
* Error Correction Zoo: https://errorcorrectionzoo.org/c/stab_5_1_3
* Wikipedia: https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import math

import numpy as np

from qectostim.codes.abstract_code import StabilizerCode, PauliString

Coord2D = Tuple[float, float]


class PerfectCode513(StabilizerCode):
    """[[5, 1, 3]] Perfect code — a non-CSS stabiliser code.

    The smallest quantum error-correcting code, saturating the quantum
    Hamming bound.  Stabilisers are mixed X/Z weight-4 operators arranged
    cyclically.

    Parameters
    ----------
    metadata : dict, optional
        Extra metadata merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (5).
    k : int
        Number of logical qubits (1).
    distance : int
        Code distance (3).

    Examples
    --------
    >>> code = PerfectCode513()
    >>> code.n, code.k, code.distance
    (5, 1, 3)

    Notes
    -----
    This is a **non-CSS** code — the stabilisers contain both X and Z
    on the same qubits.  CSS-specific metadata keys (``x_schedule``,
    ``z_schedule``, ``stabiliser_schedule``) are set to ``None``.

    The stabilisers in symplectic form [X_part | Z_part]::

        g₁ = XZZXI → X: 10010, Z: 01100
        g₂ = IXZZX → X: 01001, Z: 00110
        g₃ = XIXZZ → X: 10100, Z: 00011
        g₄ = ZXIXZ → X: 01010, Z: 10001

    See Also
    --------
    SteaneCode713 : Smallest CSS error-correcting code.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialise the [[5, 1, 3]] perfect code.

        Builds the stabiliser matrix in symplectic form, logical
        operators, and all standard metadata fields.

        Parameters
        ----------
        metadata : dict, optional
            Extra key/value pairs merged into the code's metadata
            dictionary.  User-supplied entries override auto-generated
            ones with the same key.

        Raises
        ------
        No ``ValueError`` raised — all code parameters are fixed.
        """
        
        # Full stabilizer generators in symplectic form [X | Z]
        # Each row: [X_0, X_1, X_2, X_3, X_4, Z_0, Z_1, Z_2, Z_3, Z_4]
        self._stabilizer_matrix = np.array([
            # g1 = XZZXI -> X on (0,3), Z on (1,2)
            [1, 0, 0, 1, 0,  0, 1, 1, 0, 0],
            # g2 = IXZZX -> X on (1,4), Z on (2,3)
            [0, 1, 0, 0, 1,  0, 0, 1, 1, 0],
            # g3 = XIXZZ -> X on (0,2), Z on (3,4)
            [1, 0, 1, 0, 0,  0, 0, 0, 1, 1],
            # g4 = ZXIXZ -> X on (1,3), Z on (0,4)
            [0, 1, 0, 1, 0,  1, 0, 0, 0, 1],
        ], dtype=np.uint8)
        
        # Logical operators (weight 5, transversal)
        self._logical_x = [{i: 'X' for i in range(5)}]  # XXXXX
        self._logical_z = [{i: 'Z' for i in range(5)}]  # ZZZZZ
        
        # Metadata
        meta = dict(metadata or {})
        meta["code_family"] = "small_stabiliser"
        meta["code_type"] = "perfect_513"
        meta["n"] = 5
        meta["k"] = 1
        meta["distance"] = 3
        meta["rate"] = 1.0 / 5.0
        meta["is_css"] = False
        meta["full_stabilizers"] = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
        meta["decoder_compatible"] = True
        meta["naked_l0_percentage"] = 16

        # Pentagon geometry for visualization
        coords = []
        for i in range(5):
            angle = math.pi / 2 + 2 * math.pi * i / 5
            coords.append((math.cos(angle), math.sin(angle)))
        meta["data_coords"] = coords
        meta["data_qubits"] = list(range(5))

        # Logical operator metadata
        meta["lx_pauli_type"] = "X"
        meta["lx_support"] = list(range(5))    # XXXXX
        meta["lz_pauli_type"] = "Z"
        meta["lz_support"] = list(range(5))    # ZZZZZ
        meta["x_logical_coords"] = coords       # all qubits
        meta["z_logical_coords"] = coords       # all qubits

        # CSS-specific keys set to None (non-CSS code)
        meta["x_schedule"] = None
        meta["z_schedule"] = None
        meta["stabiliser_schedule"] = None

        # ═══════════════════════════════════════════════════════════════════
        # LITERATURE / PROVENANCE
        # ═══════════════════════════════════════════════════════════════════
        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/stab_5_1_3"
        meta["wikipedia_url"] = "https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code"
        meta["canonical_references"] = [
            "Laflamme, Miquel, Paz & Zurek, Phys. Rev. Lett. 77, 198 (1996). arXiv:quant-ph/9602019",
            "Bennett, DiVincenzo, Smolin & Wootters, Phys. Rev. A 54, 3824 (1996). arXiv:quant-ph/9604024",
        ]
        meta["connections"] = [
            "Saturates the quantum Hamming bound (perfect code)",
            "Smallest non-trivial quantum error-correcting code",
            "Cyclic stabiliser structure (shift by 1 qubit)",
            "Non-CSS: stabilisers mix X and Z operators",
        ]
        
        self._metadata = meta

    @property
    def n(self) -> int:
        """Number of physical qubits."""
        return 5
    
    @property
    def k(self) -> int:
        """Number of logical qubits."""
        return 1
    
    @property
    def distance(self) -> int:
        """Code distance (3)."""
        return 3

    @property
    def name(self) -> str:
        """Human-readable name: ``'PerfectCode513'``."""
        return "PerfectCode513"
    
    @property
    def stabilizer_matrix(self) -> np.ndarray:
        """Stabilizer generators in symplectic form [X_part | Z_part]."""
        return self._stabilizer_matrix
    
    def logical_x_ops(self) -> List[PauliString]:
        """Logical X operators."""
        return self._logical_x
    
    def logical_z_ops(self) -> List[PauliString]:
        """Logical Z operators."""
        return self._logical_z
    
    def qubit_coords(self) -> Optional[List[Coord2D]]:
        """Return pentagon coordinates for visualization."""
        return self._metadata.get("data_coords")
