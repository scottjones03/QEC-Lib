"""[[9,1,3]] Shor Code (Concatenated Repetition Code)

Shor's code is a [[9,1,3]] CSS code – the first quantum error-correcting code,
constructed by concatenating a 3-qubit bit-flip code with a 3-qubit phase-flip code.
It can correct one arbitrary qubit error (distance 3).

The qubits are organized in a 3×3 grid where:
- Each row is a bit-flip repetition code (Z errors detected by row parity)
- Each column is a phase-flip repetition code (X errors detected by column parity)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex3


class ShorCode91(TopologicalCSSCode):
    """
    [[9,1,3]] Shor code (first quantum error-correcting code).

    Qubits arranged in 3x3 grid:
      0  1  2
      3  4  5
      6  7  8

    Stabilizers:
    - Z stabilizers: parity checks within each row (weight-2)
    - X stabilizers: parity checks across rows (weight-6, ensuring phase coherence)
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialize Shor's code with proper CSS structure and chain complex."""

        # ═══════════════════════════════════════════════════════════════════
        # SHOR CODE STABILIZER STRUCTURE
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
        # Codewords (the 8 basis states of |0⟩_L and |1⟩_L combined):
        #   |000000000⟩, |000000111⟩, |000111000⟩, |000111111⟩,
        #   |111000000⟩, |111000111⟩, |111111000⟩, |111111111⟩
        # (Each block is either all-0 or all-1, and the pattern repeats)
        # ═══════════════════════════════════════════════════════════════════

        # Z-type stabilizers (hx): 6 weight-2 checks WITHIN each block
        # These detect X errors (bit-flips within a GHZ block)
        # Rows: {0,1}, {1,2} (block 0)
        #       {3,4}, {4,5} (block 1)
        #       {6,7}, {7,8} (block 2)
        hx = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0],  # Z_0 Z_1
            [0, 1, 1, 0, 0, 0, 0, 0, 0],  # Z_1 Z_2
            [0, 0, 0, 1, 1, 0, 0, 0, 0],  # Z_3 Z_4
            [0, 0, 0, 0, 1, 1, 0, 0, 0],  # Z_4 Z_5
            [0, 0, 0, 0, 0, 0, 1, 1, 0],  # Z_6 Z_7
            [0, 0, 0, 0, 0, 0, 0, 1, 1],  # Z_7 Z_8
        ], dtype=np.uint8)

        # X-type stabilizers (hz): 2 weight-6 checks ACROSS blocks
        # These detect Z errors (phase-flips that affect block coherence)
        hz = np.array([
            [1, 1, 1, 1, 1, 1, 0, 0, 0],  # X_0 X_1 X_2 X_3 X_4 X_5
            [0, 0, 0, 1, 1, 1, 1, 1, 1],  # X_3 X_4 X_5 X_6 X_7 X_8
        ], dtype=np.uint8)

        # Build chain complex for CSS code structure:
        #   C2 (X stabilizers) --∂2--> C1 (qubits) --∂1--> C0 (Z stabilizers)
        #
        # boundary_2 = Hz.T: maps faces (X stabs) → edges (qubits)
        # boundary_1 = Hx:   maps edges (qubits) → vertices (Z stabs)
        boundary_2 = hz.T.astype(np.uint8)  # shape (9, 2)
        boundary_1 = hx.astype(np.uint8)    # shape (6, 9)
        
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        # ═══════════════════════════════════════════════════════════════════
        # SHOR CODE LOGICAL OPERATORS (SWAPPED vs. naive expectation!)
        # ═══════════════════════════════════════════════════════════════════
        # 
        # CRITICAL: Shor code has SWAPPED logical operators due to its
        # concatenated structure. The codewords are:
        #   |0⟩_L = (|000⟩ + |111⟩)⊗³ / 2√2
        #   |1⟩_L = (|000⟩ - |111⟩)⊗³ / 2√2
        #
        # The difference is a PHASE (minus sign), not a Z-basis amplitude!
        # This means:
        #   - Z-type operators CANNOT distinguish |0⟩_L from |1⟩_L
        #   - X-type operators CAN distinguish them (via phase detection)
        #
        # Mathematical proof:
        #   Z₀Z₃Z₆ |0⟩_L = |1⟩_L  (acts as logical X, NOT logical Z!)
        #   X₀X₁X₂ |0⟩_L = +|0⟩_L, X₀X₁X₂ |1⟩_L = -|1⟩_L  (eigenvalue ±1)
        #
        # Therefore:
        #   Logical Z = X₀X₁X₂ (X-type! measures ±1 for |0⟩_L vs |1⟩_L)
        #   Logical X = Z₀Z₃Z₆ (Z-type! flips |0⟩_L ↔ |1⟩_L)
        # ═══════════════════════════════════════════════════════════════════
        
        # Logical X: Z on one qubit per block (flips between |0⟩_L and |1⟩_L)
        # Z₀Z₃Z₆ maps (|000⟩+|111⟩)⊗³ → (|000⟩-|111⟩)⊗³
        logical_x = ["ZIIZIIZII"]  # Z_0 Z_3 Z_6 (one per block)
        
        # Logical Z: X on one complete block (distinguishes |0⟩_L from |1⟩_L)
        # X₀X₁X₂ gives +1 eigenvalue on |0⟩_L, -1 eigenvalue on |1⟩_L
        logical_z = ["XXXIIIIII"]  # X_0 X_1 X_2 (first block)

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
        meta["name"] = "Shor_91"
        meta["n"] = 9
        meta["k"] = 1
        meta["distance"] = 3
        meta["data_coords"] = data_coords
        meta["x_stab_coords"] = x_stab_coords
        meta["z_stab_coords"] = z_stab_coords
        
        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL: Logical Z measurement basis indicator
        # ═══════════════════════════════════════════════════════════════════
        # Shor code's logical Z is X-type (X₀X₁X₂), not Z-type.
        # This means:
        # - To measure Lz, we need X-basis measurements (MX), not Z-basis (M)
        # - For FT verification, we need X-basis comparison instead of Z-basis
        # - The observable parity must be computed from MX results
        #
        # lz_pauli_type = 'X' indicates Lz is an X-type operator
        # lz_support = [0, 1, 2] gives the qubit support of Lz
        # ═══════════════════════════════════════════════════════════════════
        meta["lz_pauli_type"] = "X"  # Lz is X-type (unusual!)
        meta["lz_support"] = [0, 1, 2]  # X₀X₁X₂
        
        # Also indicate that Lx is Z-type (swapped from normal CSS)
        meta["lx_pauli_type"] = "Z"
        meta["lx_support"] = [0, 3, 6]  # Z₀Z₃Z₆
        
        # Measurement schedules
        meta["x_schedule"] = [(0.0, 0.5), (1.0, 0.5), (2.0, 0.5)]  # 3 qubits per X stab
        meta["z_schedule"] = [(0.5, 0.0), (-0.5, 0.0)]  # 2 qubits per Z stab
        
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

        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)
        
        # Override parity check matrices
        self._hx = hx.astype(np.uint8)
        self._hz = hz.astype(np.uint8)
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates for visualization."""
        return list(self.metadata.get("data_coords", []))
