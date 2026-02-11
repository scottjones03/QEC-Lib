"""Steane Code — [[7, 1, 3]] Self-Dual CSS Code

The Steane code is the smallest CSS code that can correct an arbitrary
single-qubit error.  It encodes **1 logical qubit** in **7 physical qubits**
with code distance **3**.

Construction from the Hamming code
-----------------------------------
The Steane code is built by applying the CSS construction to the classical
[7, 4, 3] Hamming code *H* and its dual [7, 3, 4] code *H⊥*:

    H_X = H_Z = parity-check matrix of the [7, 4, 3] Hamming code

Because the Hamming code is **self-orthogonal** (every codeword of H⊥
is also a codeword of H, i.e. H⊥ ⊂ H), the CSS commutativity condition
H_X · H_Zᵀ = 0 is automatically satisfied.  This self-duality means
X-stabilisers and Z-stabilisers have *identical supports*.

Code parameters
---------------
* **n** = 7 physical qubits
* **k** = 1 logical qubit
* **d** = 3 (corrects any single Pauli error)
* **Rate** R = 1/7 ≈ 0.143

Stabilisers (systematic Hamming form)
--------------------------------------
Using H = [I₃ | P] where P encodes the parity relations:

    S_X⁰ = S_Z⁰ : X₀X₃X₅X₆  (qubits {0, 3, 5, 6})
    S_X¹ = S_Z¹ : X₁X₃X₄X₆  (qubits {1, 3, 4, 6})
    S_X² = S_Z² : X₂X₄X₅X₆  (qubits {2, 4, 5, 6})

Logical operators (minimum weight 3)
--------------------------------------
    X̄ = X₀X₁X₃    Z̄ = Z₀Z₁Z₃

Both commute with all six stabilisers (even overlap with each) and
anti-commute with each other (overlap on 3 qubits → odd).

Qubit layout (triangular colour-code embedding)
-------------------------------------------------
The Steane code is the smallest triangular colour code.  Data qubits sit
on the **vertices** of a triangulated region; each stabiliser corresponds
to one of the three quadrilateral **faces** (plaquettes).

::

            0                 y=2
          /   \\
        3 - 6 - 5             y=1
      /   \\ | /   \\
    1 ----- 4 ----- 2         y=0

    x:  0   1   2   3   4

    Data qubits at vertices:
      0: (2, 2)   top vertex
      1: (0, 0)   bottom-left
      2: (4, 0)   bottom-right
      3: (1, 1)   mid-left
      4: (2, 0)   bottom-center
      5: (3, 1)   mid-right
      6: (2, 1)   center

    Stabiliser plaquettes (quadrilateral faces):
      [0] → {0, 3, 5, 6}  upper face (vertices 0-3-6-5)
      [1] → {1, 3, 4, 6}  lower-left face (vertices 1-3-6-4)
      [2] → {2, 4, 5, 6}  lower-right face (vertices 2-5-6-4)

    Stabiliser coords (face centroids):
      [0]: centroid of (2,2),(1,1),(3,1),(2,1) → (2.0, 1.25)
      [1]: centroid of (0,0),(1,1),(2,0),(2,1) → (1.25, 0.5)
      [2]: centroid of (4,0),(2,0),(3,1),(2,1) → (2.75, 0.5)

The three weight-4 stabilisers correspond to the three faces of this
triangulation.  This makes the Steane code equivalent to the **smallest
instance of the 2D triangular colour code** (distance 3 on the Fano plane).

Transversal gates
-----------------
The Steane code supports transversal implementations of the full
**Clifford group** {H, S, CNOT}.  In particular:
z
* Transversal H: H⊗⁷ (because H_X = H_Z, self-dual)
* Transversal CNOT: bitwise CNOT between two code blocks
* Transversal S (phase): requires the triorthogonality property

This makes it a key building block for fault-tolerant quantum computation
when combined with magic-state distillation for the T gate.

Encoding circuits
-----------------
|0⟩_L preparation (puts qubits in the +1 eigenstate of all stabilisers
with logical Z̄ = +1):

    1. Apply H to qubits {3, 4, 5}  (message bits in systematic form)
    2. CNOT cascade to compute parity bits {0, 1, 2}

|+⟩_L preparation (logical X̄ = +1 eigenstate):

    1. Apply H to qubits {0, 1, 2, 6}  (complement of {3, 4, 5})
    2. Reversed CNOT cascade

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]] = [[7, 1, 3]]` where:

- :math:`n = 7` physical qubits (vertices of the Fano-plane triangulation)
- :math:`k = 1` logical qubit
- :math:`d = 3` (corrects any single Pauli error)
- Rate :math:`k/n = 1/7 \approx 0.143`

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **X-type stabilisers**: 3 generators, each weight 4; supports are
  ``{0,3,5,6}``, ``{1,3,4,6}``, ``{2,4,5,6}`` (faces of the triangulation).
- **Z-type stabilisers**: 3 generators, each weight 4; identical supports
  to the X-type stabilisers (self-dual: :math:`H_X = H_Z`).
- Measurement schedule: all 6 stabilisers can be measured in a single
  round (parallel); each requires a depth-4 CNOT circuit.

Connections to other codes
--------------------------
* **Hamming code**: direct CSS lift of the classical [7, 4, 3] Hamming code.
* **Triangular colour code**: the d = 3 triangular colour code on the
  Fano plane is exactly the Steane code.
* **Reed–Muller codes**: the Steane code is the smallest quantum
  Reed–Muller code, RM(1, 3).
* **Concatenated codes**: commonly used as the inner code in concatenated
  fault-tolerant schemes.

References
----------
* Steane, "Error correcting codes in quantum theory",
  Proc. R. Soc. A 452, 2551 (1996).
* Steane, "Multiple-particle interference and quantum error correction",
  Proc. R. Soc. A 452, 2551–2577 (1996).  arXiv:quant-ph/9601029
* Calderbank & Shor, "Good quantum error-correcting codes exist",
  Phys. Rev. A 54, 1098 (1996).  arXiv:quant-ph/9512032
* Error Correction Zoo: https://errorcorrectionzoo.org/c/steane
* Wikipedia: https://en.wikipedia.org/wiki/Steane_code
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex3
from qectostim.codes.utils import validate_css_code


class SteaneCode713(TopologicalCSSCode):
    """[[7, 1, 3]] Steane code (self-dual CSS / triangular colour code).

    Encodes 1 logical qubit in 7 physical qubits with distance 3.
    Corrects any single Pauli error.  Self-dual: H_X = H_Z.

    Parameters
    ----------
    metadata : dict, optional
        Extra metadata merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (7).
    k : int
        Number of logical qubits (1).
    distance : int
        Code distance (3).
    hx : np.ndarray
        X-stabiliser parity-check matrix, shape ``(3, 7)``.
    hz : np.ndarray
        Z-stabiliser parity-check matrix, shape ``(3, 7)``.
        Identical to ``hx`` (self-dual).

    Examples
    --------
    >>> code = SteaneCode713()
    >>> code.n, code.k, code.distance
    (7, 1, 3)
    >>> (code.hx == code.hz).all()   # self-dual
    True

    Notes
    -----
    The Steane code is the most widely used small CSS code for
    fault-tolerant demonstrations.  Its self-duality gives it a
    transversal Hadamard, and its colour-code structure gives it a
    transversal S gate — together yielding the full Clifford group
    transversally.

    See Also
    --------
    RepetitionCode : Simpler code that only corrects one error type.
    RotatedSurfaceCode : Scalable topological code for larger distances.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialise the [[7, 1, 3]] Steane code.

        Builds the parity-check matrices from the systematic form of the
        classical [7, 4, 3] Hamming code, constructs the 3-term chain
        complex, and populates all standard metadata fields.

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
        Stabilisers are defined in Hamming systematic form H = [I₃ | P]:

        * S₀ : qubits {0, 3, 5, 6}
        * S₁ : qubits {1, 3, 4, 6}
        * S₂ : qubits {2, 4, 5, 6}

        Because the code is self-dual (H_X = H_Z), all three X-stabilisers
        have identical support to the three Z-stabilisers.
        """
        # X-type check matrix (3 checks, each on 4 qubits)
        # Systematic form: H = [I_3 | P] where P defines parity relations
        # Row 0: X_0 X_3 X_5 X_6 (qubits {0,3,5,6})
        # Row 1: X_1 X_3 X_4 X_6 (qubits {1,3,4,6})  
        # Row 2: X_2 X_4 X_5 X_6 (qubits {2,4,5,6})
        hx = np.array([
            [1, 0, 0, 1, 0, 1, 1],  # qubits {0,3,5,6}
            [0, 1, 0, 1, 1, 0, 1],  # qubits {1,3,4,6}
            [0, 0, 1, 0, 1, 1, 1],  # qubits {2,4,5,6}
        ], dtype=np.uint8)

        # Z-type check matrix (same as Hx due to self-duality)
        hz = np.array([
            [1, 0, 0, 1, 0, 1, 1],  # qubits {0,3,5,6}
            [0, 1, 0, 1, 1, 0, 1],  # qubits {1,3,4,6}
            [0, 0, 1, 0, 1, 1, 1],  # qubits {2,4,5,6}
        ], dtype=np.uint8)

        # Build chain complex for CSS code structure:
        #   C2 (X stabilizers) --∂2--> C1 (qubits) --∂1--> C0 (Z stabilizers)
        #
        # boundary_2 = Hx.T: maps faces (X stabs) → edges (qubits), shape (n, #X_checks)
        # boundary_1 = Hz:   maps edges (qubits) → vertices (Z stabs), shape (#Z_checks, n)
        boundary_2 = hx.T.astype(np.uint8)  # shape (7, 3) - 7 qubits, 3 X stabilizers
        boundary_1 = hz.astype(np.uint8)    # shape (3, 7) - 3 Z stabilizers, 7 qubits
        
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        # Logical operators - use minimum weight representatives (weight 3)
        # IMPORTANT: These must COMMUTE with all stabilizers!
        # For stabilizers {0,3,5,6}, {1,3,4,6}, {2,4,5,6}:
        # - Z_L = Z₀Z₁Z₃ commutes with all X stabilizers (even overlap)
        # - X_L = X₀X₁X₃ commutes with all Z stabilizers (even overlap)
        # Note: Z₀Z₁Z₂ ANTICOMMUTES with all X stabs (odd overlap) - WRONG!
        logical_x=[{0: 'X', 1: 'X', 3: 'X'}]
        logical_z=[{0: 'Z', 1: 'Z', 3: 'Z'}]

        # ═══════════════════════════════════════════════════════════════════
        # GEOMETRIC LAYOUT: Triangular colour-code embedding
        # ═══════════════════════════════════════════════════════════════════
        # The Steane code lives on a triangulated region with data qubits
        # at vertices and stabilizers as face operators.
        #
        #         0                 y=2
        #       /   \
        #     3 - 6 - 5             y=1
        #   /   \ | /   \
        # 1 ----- 4 ----- 2         y=0
        #
        # Stabilizer [0] = {0,3,5,6} = upper face
        # Stabilizer [1] = {1,3,4,6} = lower-left face
        # Stabilizer [2] = {2,4,5,6} = lower-right face
        #
        # Each face is a quadrilateral with 4 vertices (qubits).
        # Check overlaps (must share exactly 2 qubits for colour code):
        #   {0,3,5,6} ∩ {1,3,4,6} = {3,6} ✓
        #   {0,3,5,6} ∩ {2,4,5,6} = {5,6} ✓
        #   {1,3,4,6} ∩ {2,4,5,6} = {4,6} ✓
        coords = {
            0: (2.0, 2.0),    # top vertex
            1: (0.0, 0.0),    # bottom-left vertex
            2: (4.0, 0.0),    # bottom-right vertex
            3: (1.0, 1.0),    # mid-left vertex
            4: (2.0, 0.0),    # bottom-center vertex
            5: (3.0, 1.0),    # mid-right vertex
            6: (2.0, 1.0),    # center vertex
        }
        data_coords = [coords[i] for i in range(7)]
        
        # Face centers for stabilizer coordinates (centroids of face qubits)
        # Stabilizer [0] = {0,3,5,6}: centroid of (2,2), (1,1), (3,1), (2,1)
        # Stabilizer [1] = {1,3,4,6}: centroid of (0,0), (1,1), (2,0), (2,1)
        # Stabilizer [2] = {2,4,5,6}: centroid of (4,0), (2,0), (3,1), (2,1)
        stab_coords = [
            (2.0, 1.25),   # Face [0]: {0,3,5,6} → (2+1+3+2)/4=2, (2+1+1+1)/4=1.25
            (1.25, 0.5),   # Face [1]: {1,3,4,6} → (0+1+2+2)/4=1.25, (0+1+0+1)/4=0.5
            (2.75, 0.5),   # Face [2]: {2,4,5,6} → (4+2+3+2)/4=2.75, (0+0+1+1)/4=0.5
        ]

        meta = dict(metadata or {})
        meta["distance"] = 3
        meta["code_family"] = "small_css"
        meta["code_type"] = "steane"
        meta["n"] = 7
        meta["k"] = 1
        meta["rate"] = 1.0 / 7.0
        meta["is_colour_code"] = True
        meta["tiling"] = "triangular"
        meta["data_coords"] = data_coords
        meta["x_stab_coords"] = stab_coords
        meta["z_stab_coords"] = stab_coords  # Self-dual
        meta["logical_x_support"] = [0, 1, 3]
        meta["logical_z_support"] = [0, 1, 3]
        meta["data_qubits"] = list(range(7))
        meta["x_logical_coords"] = [data_coords[i] for i in [0, 1, 3]]
        meta["z_logical_coords"] = [data_coords[i] for i in [0, 1, 3]]
        
        # # ═══════════════════════════════════════════════════════════════════
        # # LOGICAL OPERATOR PAULI TYPES (required for universal CSS decoding)
        # # ═══════════════════════════════════════════════════════════════════
        # # Steane code is standard CSS: Lz is Z-type, Lx is X-type
        # # This means:
        # # - To decode Z-basis measurement: use Hz (detects X errors that flip Z)
        # # - To decode X-basis measurement: use Hx (detects Z errors that flip X)
        meta["lz_pauli_type"] = "Z"  # Lz = Z₀Z₁Z₃ is Z-type (standard)
        meta["lz_support"] = [0, 1, 3]  # Z₀Z₁Z₃
        meta["lx_pauli_type"] = "X"  # Lx = X₀X₁X₃ is X-type (standard)
        meta["lx_support"] = [0, 1, 3]  # X₀X₁X₃
        
        # ═══════════════════════════════════════════════════════════════════
        # |0⟩_L ENCODING CIRCUIT
        # ═══════════════════════════════════════════════════════════════════
        # For the [[7,1,3]] Steane code (CSS from [7,4,3] Hamming):
        #
        # Systematic form interpretation:
        #   - Qubits 0,1,2 = parity bits (computed from message bits)
        #   - Qubits 3,4,5,6 = message bits (4 bits, but Lz constraint → 3 free)
        #
        # |0⟩_L = equal superposition of 8 codewords with Lz = 0
        # where Lz = Z₀Z₁Z₂ (parity on first 3 qubits must be even)
        #
        # Encoding:
        #   1. H on qubits 3,4,5 (3 message bits with H → 8 states)
        #   2. Qubit 6 stays |0⟩ (Lz constraint - qubit 6 affects parity)
        #   3. CNOTs compute parity bits 0,1,2 from message bits
        #
        # Parity relations from Hz:
        #   q₀ = m₃ ⊕ m₅ ⊕ m₆  → CNOT from 3,5,6 to 0 (but 6=0, so 3,5 to 0)
        #   q₁ = m₃ ⊕ m₄ ⊕ m₆  → CNOT from 3,4 to 1
        #   q₂ = m₄ ⊕ m₅ ⊕ m₆  → CNOT from 4,5 to 2
        # ═══════════════════════════════════════════════════════════════════
        meta["zero_h_qubits"] = [3, 4, 5]  # H on 3 message qubits
        
        # CNOTs to compute parity bits (systematic encoding)
        meta["zero_encoding_cnots"] = [
            (3, 0), (5, 0),  # q₀ = m₃ ⊕ m₅ (m₆=0)
            (3, 1), (4, 1),  # q₁ = m₃ ⊕ m₄
            (4, 2), (5, 2),  # q₂ = m₄ ⊕ m₅
        ]
        
        # ═══════════════════════════════════════════════════════════════════
        # STANDARD ENCODING CIRCUIT (for L2 inner blocks)
        # ═══════════════════════════════════════════════════════════════════
        # For L2 inner blocks that will have "outer H" applied, we need an
        # encoding that allows both Lz=0 and Lz=1 codewords (all 16).
        # 
        # The outer CNOT will correlate the Lz values between blocks.
        #
        # Encoding:
        #   1. H on qubits 3,4,5,6 (ALL message qubits → 16 states)
        #   2. CNOTs compute parity bits 0,1,2
        #
        # This produces all 16 codewords with equal probability.
        # ═══════════════════════════════════════════════════════════════════
        meta["plus_h_qubits"] = [3, 4, 5, 6]  # All message qubits
        meta["plus_encoding_cnots"] = [
            (3, 0), (5, 0), (6, 0),  # q₀ = m₃ ⊕ m₅ ⊕ m₆
            (3, 1), (4, 1), (6, 1),  # q₁ = m₃ ⊕ m₄ ⊕ m₆
            (4, 2), (5, 2), (6, 2),  # q₂ = m₄ ⊕ m₅ ⊕ m₆
        ]
        
        # ═══════════════════════════════════════════════════════════════════
        # TRUE |+⟩_L ENCODING (Lx = +1 eigenstate)
        # ═══════════════════════════════════════════════════════════════════
        # For Knill EC Bell pairs, we need TRUE |+⟩_L with deterministic Lx=+1.
        #
        # UNIVERSAL FORMULA (works for any CSS code):
        #   |+⟩_L = H^⊗n |0⟩_L
        #   Using identity: H^⊗n · CNOT(a,b) = CNOT(b,a) · H^⊗n
        #   
        # If |0⟩_L encoding = (CNOTs) · H_{S} · |0⟩^⊗n
        # Then |+⟩_L = H^⊗n · (CNOTs) · H_{S} · |0⟩^⊗n
        #            = (reversed CNOTs) · H_{S̄} · |0⟩^⊗n
        # where S̄ = complement of S
        #
        # For Steane: S = {3,4,5}, so S̄ = {0,1,2,6}
        # ═══════════════════════════════════════════════════════════════════
        meta["plus_state_h_qubits"] = [0, 1, 2, 6]  # Complement of {3,4,5}
        meta["plus_state_encoding_cnots"] = [
            # Reversed CNOTs (control <-> target swapped)
            (0, 3), (0, 5),  # was (3,0), (5,0)
            (1, 3), (1, 4),  # was (3,1), (4,1)
            (2, 4), (2, 5),  # was (4,2), (5,2)
        ]

        # ═══════════════════════════════════════════════════════════════════
        # STABILISER SCHEDULE
        # ═══════════════════════════════════════════════════════════════════
        # For the Steane code, all 3 X-stabilisers and all 3 Z-stabilisers
        # can be measured in parallel (no geometric conflicts on a 7-qubit
        # register).  We deliberately omit x_schedule/z_schedule offset
        # vectors because the colour-code geometry requires matrix-based
        # circuit construction rather than offset-based scheduling.
        meta["stabiliser_schedule"] = {
            "x_rounds": {0: 0, 1: 0, 2: 0},
            "z_rounds": {0: 0, 1: 0, 2: 0},
            "n_rounds": 1,
            "description": (
                "Fully parallel: all 3 X-stabilisers in round 0, "
                "all 3 Z-stabilisers in round 0.  Offset-based "
                "scheduling omitted (colour-code geometry)."
            ),
        }

        # ═══════════════════════════════════════════════════════════════════
        # LITERATURE / PROVENANCE
        # ═══════════════════════════════════════════════════════════════════
        meta["error_correction_zoo_url"] = "https://errorcorrectionzoo.org/c/steane"
        meta["wikipedia_url"] = "https://en.wikipedia.org/wiki/Steane_code"
        meta["canonical_references"] = [
            "Steane, Proc. R. Soc. A 452, 2551 (1996). arXiv:quant-ph/9601029",
            "Calderbank & Shor, Phys. Rev. A 54, 1098 (1996). arXiv:quant-ph/9512032",
        ]
        meta["connections"] = [
            "CSS lift of classical [7,4,3] Hamming code",
            "Smallest instance of 2D triangular colour code (Fano plane)",
            "Smallest quantum Reed-Muller code RM(1,3)",
            "Common inner code for concatenated fault-tolerant schemes",
        ]
        
        # NOTE: We deliberately set x_schedule/z_schedule to None here.
        # The geometric schedule approach requires stabilizer coords + offsets to exactly
        # match data qubit coords, which is complex for colour codes. Instead, we use
        # the fallback matrix-based circuit construction in CSSMemoryExperiment.
        meta["x_schedule"] = None  # colour-code geometry → matrix-based scheduling
        meta["z_schedule"] = None  # colour-code geometry → matrix-based scheduling

        # ── Validate CSS structure ─────────────────────────────────
        validate_css_code(hx, hz, "SteaneCode713", raise_on_error=True)

        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)
        
        # Override parity check matrices
        self._hx = hx.astype(np.uint8)
        self._hz = hz.astype(np.uint8)
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates for visualization (triangular layout)."""
        return list(self.metadata.get("data_coords", []))

    @property
    def distance(self) -> int:
        """Code distance (3)."""
        return 3

    @property
    def name(self) -> str:
        """Human-readable name: ``'SteaneCode713'``."""
        return "SteaneCode713"
