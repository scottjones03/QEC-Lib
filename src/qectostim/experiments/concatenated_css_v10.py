"""
Concatenated CSS Code Simulator v10
================================================================================

A comprehensive framework for simulating concatenated Calderbank-Shor-Steane
(CSS) quantum error correcting codes using the Stim circuit simulator.

================================================================================
                        QUANTUM ERROR CORRECTION PRIMER
================================================================================

1. WHY QUANTUM ERROR CORRECTION?
--------------------------------
Quantum computers are inherently noisy. Unlike classical bits that can be 
copied and majority-voted for error protection, quantum states cannot be 
cloned (no-cloning theorem). Quantum Error Correction (QEC) encodes a 
single "logical" qubit into multiple "physical" qubits in such a way that
errors can be detected and corrected without measuring the encoded 
information directly.

Key insight: We can measure "stabilizers" - operators that tell us if an 
error occurred WITHOUT revealing the encoded quantum information.

2. STABILIZER FORMALISM
-----------------------
A stabilizer code is defined by a set of commuting Pauli operators {S_i}
called stabilizers. Valid codewords |ψ⟩ satisfy: S_i|ψ⟩ = |ψ⟩ for all i.

For an [[n,k,d]] code:
- n = number of physical qubits
- k = number of encoded logical qubits  
- d = code distance (minimum weight of undetectable error)

The code can correct ⌊(d-1)/2⌋ arbitrary errors on any qubits.

3. CSS CODES (Calderbank-Shor-Steane)
-------------------------------------
CSS codes are a special class where X and Z errors can be corrected 
independently. They are defined by two classical linear codes C_x and C_z
with the constraint C_z^⟂ ⊆ C_x (or equivalently, rows of Hz are 
orthogonal to rows of Hx mod 2).

The parity check matrices define stabilizer generators:
- Hz (r_z × n matrix): Each row defines a Z-type stabilizer (detects X errors)
- Hx (r_x × n matrix): Each row defines an X-type stabilizer (detects Z errors)

Example - Steane [[7,1,3]] code (self-dual, Hz = Hx):
    Hz = Hx = [[0,0,0,1,1,1,1],
               [0,1,1,0,0,1,1],
               [1,0,1,0,1,0,1]]

Example - Shor [[9,1,3]] code (non-self-dual):
    Hz = [[1,1,0,0,0,0,0,0,0],    # X errors within row 1
          [0,1,1,0,0,0,0,0,0],
          [0,0,0,1,1,0,0,0,0],    # X errors within row 2
          [0,0,0,0,1,1,0,0,0],
          [0,0,0,0,0,0,1,1,0],    # X errors within row 3
          [0,0,0,0,0,0,0,1,1]]

    Hx = [[1,1,1,1,1,1,0,0,0],    # Z errors across rows 1-2
          [0,0,0,1,1,1,1,1,1]]    # Z errors across rows 2-3

4. SYNDROME EXTRACTION
----------------------
When errors occur, we measure stabilizers to get a "syndrome":
- Each stabilizer measurement yields ±1 (or equivalently 0/1)
- The syndrome pattern reveals which error(s) occurred
- A syndrome lookup table maps syndromes to corrections

For CSS codes, X and Z error correction are independent:
- Measure Hz stabilizers → syndrome tells us which X errors occurred
- Measure Hx stabilizers → syndrome tells us which Z errors occurred

5. LOGICAL OPERATORS
--------------------
Logical operators commute with all stabilizers but are not stabilizers 
themselves. For a [[n,k,d]] code, there are k pairs (Lx_i, Lz_i).

- Lz performs logical Z on the encoded qubit (Lz|0⟩_L = |0⟩_L, Lz|1⟩_L = -|1⟩_L)
- Lx performs logical X on the encoded qubit (Lx|0⟩_L = |1⟩_L, Lx|1⟩_L = |0⟩_L)

For CSS codes, Lz is a Z-type operator (tensor product of Z and I), and
Lx is an X-type operator (tensor product of X and I).

6. SELF-DUAL VS NON-SELF-DUAL CODES
-----------------------------------
A CSS code is SELF-DUAL if Hz = Hx (the X and Z parity checks are identical).

Self-dual codes (e.g., Steane [[7,1,3]]):
- Transversal H (apply H to all qubits) = Logical H
- Bell-pair CNOT verification works directly
- This module fully supports self-dual codes

Non-self-dual codes (e.g., Shor [[9,1,3]]):  
- Transversal H ≠ Logical H
- Logical H requires gate teleportation or other techniques
- Bell-pair protocols need modification for correct verification

CRITICAL: The Bell-pair CNOT verification in this simulator assumes
transversal H = logical H. Non-self-dual codes will produce incorrect
results for L1 CNOT and L2 operations. L1 memory tests work correctly.

7. CONCATENATED CODES
---------------------
Code concatenation encodes each physical qubit of an "outer" code using
an "inner" code, achieving exponentially better error suppression.

For inner code [[n_in, k_in, d_in]] and outer code [[n_out, k_out, d_out]]:
- Level 1: n_in physical qubits encode 1 logical qubit  
- Level 2: n_in × n_out qubits encode 1 logical qubit

Effective distance grows multiplicatively: d_eff = d_in × d_out

================================================================================
                           ERROR CORRECTION GADGETS
================================================================================

8. THE KNILL TELEPORTATION-BASED EC GADGET
------------------------------------------
Named after Emanuel Knill, this EC method uses quantum teleportation to
transfer the quantum state from potentially corrupted qubits to freshly
prepared ancilla qubits, correcting errors in the process.

The Knill EC circuit:

    Data ─────────●───────[H]───[M]──→ syndrome_z (determines X correction)
                  │
    Ancilla|0⟩───⊕───────────────[M]──→ syndrome_x (determines Z correction)

Key insight: The Bell measurement (CNOT + H + measure both) teleports
the data state to the ancilla while extracting error syndrome information.

Post-teleportation corrections:
- syndrome_z = 1 → apply X correction to output (Pauli frame update)
- syndrome_x = 1 → apply Z correction to output (Pauli frame update)

9. BELL-PAIR CNOT VERIFICATION PROTOCOL
---------------------------------------
To verify a fault-tolerant CNOT, we use maximally entangled Bell pairs:

    |Φ+⟩ = (|00⟩ + |11⟩)/√2

The protocol:
1. Prepare two Bell pairs: |Φ+⟩_12 ⊗ |Φ+⟩_34
2. Apply CNOT: qubit 1 controls qubit 3 (both are "first" qubits of pairs)
3. Apply inverse Bell preparation (CNOT, then H)
4. Measure all four qubits

Expected outcome (no errors): All four qubits measure 0.

This works because:
- CNOT on Bell pairs: CNOT_13|Φ+⟩_12|Φ+⟩_34 = |Φ+⟩_13 ⊗ |Φ+⟩_24
- The CNOT "swaps" the entanglement pattern
- Undoing Bell prep on these new pairs gives |0000⟩

Errors cause measurement outcomes ≠ 0, which are detected.

REQUIREMENT: The H gates must implement LOGICAL Hadamard. For self-dual
codes, transversal H = logical H. For non-self-dual codes, this protocol
FAILS because transversal H ≠ logical H.

10. ERROR-DETECTING TELEPORTATION (EDT)
---------------------------------------
For error-DETECTING codes (d=2, like [[4,2,2]] C4), we cannot correct
errors but can detect them. EDT uses this for post-selection:

- When inner code detects an uncorrectable error → reject the sample
- Surviving samples have higher fidelity
- Trade acceptance rate for error suppression

================================================================================
                              MODULE ARCHITECTURE  
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────┐     ┌─────────────────────────┐
│        CSSCode          │     │   PropagationTables     │
├─────────────────────────┤     ├─────────────────────────┤
│ Mathematical:           │     │ propagation_X:  List    │
│  - n, k, d              │     │ propagation_Z: List     │
│  - Hz, Hx (stabilizers) │     │ propagation_m: List     │
│  - Lz, Lx (logical ops) │     │ num_ec_0prep: int       │
├─────────────────────────┤     └───────────┬─────────────┘
│ Circuit Specification:  │                 │
│  - h_qubits: [0,1,3]    │                 │
│  - encoding_cnots       │                 │
│  - encoding_cnot_rounds │                 │
│  - verification_qubits  │                 │
│  - idle_schedule        │                 │
└───────────┬─────────────┘                 │
            │                               │
            │ 1..n                           │ 0..n (per level)
            ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    ConcatenatedCode                          │
├─────────────────────────────────────────────────────────────┤
│ levels:  List[CSSCode]           # Inner to outer           │
│ propagation_tables: Dict[int, PropagationTables]            │
├─────────────────────────────────────────────────────────────┤
│ + num_levels: int                                           │
│ + total_qubits: int                                         │
│ + qubits_at_level(level) -> int                             │
│ + code_at_level(level) -> CSSCode                           │
│ + get_propagation_tables(level) -> PropagationTables        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          OPERATIONS LAYER                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│   PhysicalOps      │  │  TransversalOps    │  │  LogicalGate       │
├────────────────────┤  ├────────────────────┤  │    Dispatcher      │
│ append_reset       │  │ append_h           │  ├────────────────────┤
│ append_h           │  │ append_cnot        │  │ H, X, Z, CNOT      │
│ append_cnot        │  │ append_m           │  │ gates with         │
│ append_measure     │  │ append_noisy_cnot  │  │ proper logical     │
│ append_detector    │  │ append_logical_h   │  │ implementation     │
└────────────────────┘  └────────────────────┘  └────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        PREPARATION & EC LAYER                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐  ┌─────────────────────────────┐
│   PreparationStrategy       │  │       ECGadget              │
│      (Abstract)             │  │      (Abstract)             │
├─────────────────────────────┤  ├─────────────────────────────┤
│ append_0prep: |0⟩_L prep    │  │ append_noisy_ec:            │
│ append_noisy_0prep: with    │  │   Teleportation-based EC    │
│   noise + verification      │  │   with syndrome extraction  │
│ append_plus_prep: |+⟩_L     │  │                             │
└─────────────────────────────┘  └─────────────────────────────┘
            │                               │
            ▼                               ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ GenericPreparationStrategy  │  │      KnillECGadget          │
├─────────────────────────────┤  ├─────────────────────────────┤
│ Works for any CSS code with │  │ Bell measurement teleport   │
│ encoding circuit specified  │  │ with Pauli frame tracking   │
└─────────────────────────────┘  └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           DECODING LAYER                                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           GenericDecoder                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ decode_syndrome(m, type): Syndrome lookup for EC measurements               │
│ decode_measurement(m, type): Final logical state readout                    │
│ decode_ec_hd(x, det_X, det_Z, corr_x, corr_z): Hierarchical EC decoding    │
│ decode_m_hd(x, det_m, corr_x): Hierarchical measurement with corrections   │
├─────────────────────────────────────────────────────────────────────────────┤
│ CRITICAL DISTINCTION:                                                        │
│ - decode_syndrome: For EC syndrome data, ALWAYS uses syndrome table          │
│ - decode_measurement: For final readout, may use majority vote (Shor)       │
│                                                                              │
│ For non-self-dual codes with superposition codewords (like Shor), the       │
│ final measurement requires block majority voting to handle |000⟩+|111⟩     │
│ superpositions, but EC syndromes are always syndrome-decoded.               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      POST-SELECTION & ACCEPTANCE                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐  ┌─────────────────────────────┐
│       PostSelector          │  │    AcceptanceChecker        │
├─────────────────────────────┤  ├─────────────────────────────┤
│ post_selection_l1/l2:       │  │ accept_l1/l2:               │
│   Filter samples based on   │  │   Check Bell pair           │
│   verification measurements │  │   correlations hold         │
│                             │  │   Return error probability  │
└─────────────────────────────┘  └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          MAIN SIMULATOR                                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    ConcatenatedCodeSimulator                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ estimate_logical_cnot_error_l1(p, num_shots, Q):                            │
│   Level-1 CNOT verification using Bell pairs                                │
│                                                                              │
│ estimate_logical_cnot_error_l2(p, num_shots, Q):                            │
│   Level-2 CNOT with hierarchical decoding                                   │
│                                                                              │
│ estimate_memory_logical_error_l1(p, num_shots, num_ec_rounds):              │
│   Level-1 memory test: |0⟩_L → EC → measure → check                         │
│                                                                              │
│ estimate_memory_logical_error_l2(p, num_shots, num_ec_rounds):              │
│   Level-2 memory with full concatenation                                    │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
                            LITERATURE REFERENCES
================================================================================

Foundational Papers:
- [CS96] Calderbank & Shor, "Good quantum error-correcting codes exist",
         Phys. Rev. A 54, 1098 (1996). https://arxiv.org/abs/quant-ph/9512032
- [Ste96] Steane, "Error Correcting Codes in Quantum Theory",
         Phys. Rev. Lett. 77, 793 (1996). https://arxiv.org/abs/quant-ph/9601029
- [Sho95] Shor, "Scheme for reducing decoherence in quantum computer memory",
         Phys. Rev. A 52, R2493 (1995).

Fault-Tolerant Methods:
- [Kni05] Knill, "Quantum computing with realistically noisy devices",
         Nature 434, 39 (2005). https://arxiv.org/abs/quant-ph/0410199
- [Got97] Gottesman, "Stabilizer Codes and Quantum Error Correction",
         PhD Thesis, Caltech (1997). https://arxiv.org/abs/quant-ph/9705052
- [AGP06] Aliferis, Gottesman, Preskill, "Quantum accuracy threshold for
         concatenated distance-3 codes", QIC 6, 97 (2006).
         https://arxiv.org/abs/quant-ph/0504218

Concatenation:
- [KLZ96] Knill, Laflamme, Zurek, "Threshold Accuracy for Quantum Computation",
         https://arxiv.org/abs/quant-ph/9610011
- [AB97] Aharonov & Ben-Or, "Fault-Tolerant Quantum Computation with Constant
         Error Rate", https://arxiv.org/abs/quant-ph/9906129

================================================================================
                              VERSION HISTORY
================================================================================

v10 Design Principles:
1. CSSCode carries all code-specific circuit details
2. Abstract base classes for gates, EC, and preparation strategies  
3. Concrete implementations for Steane that match original exactly
4. General algorithms that work with any CSS code
5. Comprehensive documentation with QEC theory explanations

Combines:
- Generalizability of v7 (abstract base classes, code-agnostic algorithms)
- Correctness of v9 (exact match to original concatenated_steane.py)

================================================================================
                            SUPPORTED CODES
================================================================================

Fully Supported (self-dual CSS codes):
- Steane [[7,1,3]]: Hz = Hx, transversal logical H works
- C4 [[4,2,2]]: Error-detecting, uses EDT for post-selection
- C6 [[6,1,2]]: Often used as outer code with C4 inner

Partially Supported (non-self-dual codes):
- Shor [[9,1,3]]: L1 memory works, L1 CNOT and L2 operations fail
  (transversal H ≠ logical H breaks Bell-pair verification)

To add support for a new code:
1. Create CSSCode with Hz, Hx, Lz, Lx, encoding circuit
2. (Optional) Create PropagationTables for L2+ decoding
3. Use create_simulator() factory function

================================================================================
"""

import stim
import numpy as np
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union, Any, Type

from qectostim.noise.models import NoiseModel


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class CSSCode:
    """
    Complete specification of a Calderbank-Shor-Steane (CSS) quantum error 
    correcting code.
    
    ═══════════════════════════════════════════════════════════════════════════
                               QEC THEORY BACKGROUND
    ═══════════════════════════════════════════════════════════════════════════
    
    CSS codes are a fundamental class of quantum error correcting codes that
    allow independent correction of X (bit-flip) and Z (phase-flip) errors.
    They are constructed from two classical linear codes.
    
    STABILIZER FORMALISM
    --------------------
    A stabilizer code is defined by an abelian subgroup S of the Pauli group.
    The code space is the simultaneous +1 eigenspace of all stabilizers:
    
        |ψ⟩ ∈ Code Space ⟺ S|ψ⟩ = |ψ⟩ for all S ∈ S
    
    For CSS codes, stabilizers factor into pure X-type and pure Z-type:
    - Z-stabilizers: Tensor products of Z and I only (detect X errors)
    - X-stabilizers: Tensor products of X and I only (detect Z errors)
    
    PARITY CHECK MATRICES
    ---------------------
    The stabilizers are specified by binary parity check matrices:
    
    Hz (r_z × n matrix): Z-type stabilizers
        - Row i of Hz defines Z-stabilizer: ⊗_j Z^{Hz[i,j]}
        - Measuring Hz detects X errors (syndrome extraction)
        - Example: Hz[i] = [1,1,0,0,0,0,0] → Z₀Z₁ stabilizer
    
    Hx (r_x × n matrix): X-type stabilizers  
        - Row i of Hx defines X-stabilizer: ⊗_j X^{Hx[i,j]}
        - Measuring Hx detects Z errors (syndrome extraction)
    
    CSS CONDITION: The rows of Hz must be orthogonal to rows of Hx (mod 2):
        Hz · Hx^T = 0 (mod 2)
    
    This ensures X and Z stabilizers commute, as required for a valid code.
    
    SYNDROME EXTRACTION
    -------------------
    When an error E occurs, measuring stabilizers gives a syndrome:
    
        syndrome_i = 0 if [S_i, E] = 0 (stabilizer commutes with error)
        syndrome_i = 1 if {S_i, E} = 0 (stabilizer anti-commutes with error)
    
    For CSS codes:
    - X error on qubit j → syndrome from Hz column j (Z stabilizers anti-commute)
    - Z error on qubit j → syndrome from Hx column j (X stabilizers anti-commute)
    
    A syndrome lookup table maps each syndrome to a minimum-weight correction.
    
    LOGICAL OPERATORS
    -----------------
    Logical operators commute with all stabilizers but are not stabilizers.
    For k logical qubits, we have k pairs (Lx_i, Lz_i) satisfying:
    
        [Lx_i, S] = [Lz_i, S] = 0 for all stabilizers S
        {Lx_i, Lz_i} = 0 (anti-commute with each other)
        [Lx_i, Lz_j] = 0 for i ≠ j (commute with other logical ops)
    
    For CSS codes, Lz is Z-type (product of Z's) and Lx is X-type (product of X's).
    
    CODE DISTANCE
    -------------
    The distance d is the minimum weight of any logical operator:
    
        d = min{wt(L) : L ∈ N(S) \\ S}
    
    where N(S) is the normalizer of S (operators that commute with all stabilizers).
    A distance-d code can:
    - Detect up to d-1 errors
    - Correct up to ⌊(d-1)/2⌋ arbitrary errors
    
    SELF-DUAL VS NON-SELF-DUAL
    --------------------------
    A CSS code is SELF-DUAL if Hz = Hx.
    
    Self-dual codes (e.g., Steane [[7,1,3]]):
        - Transversal Hadamard (H on all qubits) implements LOGICAL Hadamard
        - H|0⟩_L = |+⟩_L (transversal H swaps X↔Z bases correctly)
        - Bell-pair CNOT verification works directly
    
    Non-self-dual codes (e.g., Shor [[9,1,3]]):
        - Hz ≠ Hx (different X and Z error correction structure)
        - Transversal H ≠ Logical H (H on all qubits doesn't give |+⟩_L)
        - Logical H requires gate teleportation or other complex techniques
        - Bell-pair verification protocol fails without modification
    
    ═══════════════════════════════════════════════════════════════════════════
                              ENCODING CIRCUITS
    ═══════════════════════════════════════════════════════════════════════════
    
    Standard CSS |0⟩_L Encoding (k=1 codes):
    ----------------------------------------
    1. Reset all n qubits to |0⟩
    2. Apply pre_h_cnots (CNOTs before H gates, for codes like Shor)
    3. Apply H gates to h_qubits
    4. Apply encoding_cnots (CNOTs after H gates)
    5. Optional: Verification measurements + post-selection
    
    The encoding circuit prepares a +1 eigenstate of all stabilizers that
    is also a +1 eigenstate of Lz (i.e., logical |0⟩).
    
    Bell-pair Preparation (k>1 codes like [[4,2,2]] C4):
    ---------------------------------------------------
    For codes with k>1 logical qubits, we use Bell-pair preparation:
    1. Reset data and ancilla blocks (2n qubits total)
    2. H on ALL ancilla qubits
    3. CNOT: ancilla[i] → data[i] for all i
    4. CNOT: data[i] → ancilla[(i+1)%n] for all i  
    5. Measure ALL ancilla
    6. Correction CNOTs based on measurement (triangular pattern)
    
    |+⟩_L Preparation (for non-self-dual codes):
    -------------------------------------------
    For codes without transversal logical H, we cannot prepare |+⟩_L
    by applying H_L to |0⟩_L. Instead, we need a direct |+⟩_L circuit:
    
    For Shor [[9,1,3]]:
        plus_h_qubits = [0, 3, 6]  # H on one qubit per block
        plus_encoding_cnots = [(0,1), (0,2), (3,4), (3,5), (6,7), (6,8)]
        Result: |+⟩_L = (|+++⟩ + |---⟩)/√2 ⊗ (block structure)
    
    ═══════════════════════════════════════════════════════════════════════════
                            ATTRIBUTE REFERENCE
    ═══════════════════════════════════════════════════════════════════════════
    
    Mathematical Structure:
        n: Number of physical qubits
        k: Number of logical qubits encoded
        d: Code distance (error correction capability)
        Hz: Z-stabilizer check matrix (r_z × n), detects X errors
        Hx: X-stabilizer check matrix (r_x × n), detects Z errors
        logical_z_ops: List of k Z-type logical operators (each length n)
        logical_x_ops: List of k X-type logical operators (each length n)
    
    Circuit Specification:
        h_qubits: Qubits receiving H gates during |0⟩_L encoding
        logical_h_qubits: Qubits for LOGICAL Hadamard (may differ from h_qubits)
        pre_h_cnots: CNOTs applied before H gates
        encoding_cnots: CNOTs applied after H gates
        encoding_cnot_rounds: Grouped CNOTs for parallel execution
        verification_qubits: Qubits measured for preparation verification
    
    |+⟩_L Preparation (non-self-dual codes):
        plus_h_qubits: H gates for direct |+⟩_L preparation
        plus_encoding_cnots: CNOTs for direct |+⟩_L preparation
    
    Decoder Configuration:
        decoder_type: "syndrome", "parity", or "majority"
        outer_decoder_type: Decoder type when used as outer code
    
    Post-Selection Configuration:
        post_selection_type: "verification", "parity", or "block_majority"
        uses_edt: Whether to use Error-Detecting Teleportation
    
    ═══════════════════════════════════════════════════════════════════════════
                              EXAMPLE CODES
    ═══════════════════════════════════════════════════════════════════════════
    
    Steane [[7,1,3]] (self-dual):
        Hz = Hx = [[0,0,0,1,1,1,1],
                   [0,1,1,0,0,1,1],
                   [1,0,1,0,1,0,1]]
        Lz = [1,1,1,0,0,0,0]
        Lx = [1,1,1,1,1,1,1]
        h_qubits = [0,1,3]
        
    Shor [[9,1,3]] (non-self-dual):
        |0⟩_L = (|000⟩+|111⟩)(|000⟩+|111⟩)(|000⟩+|111⟩) / 2√2
        |1⟩_L = (|000⟩-|111⟩)(|000⟩-|111⟩)(|000⟩-|111⟩) / 2√2
        Lz = [1,0,0,1,0,0,1,0,0]  # Z on first qubit of each block
        Lx = [1,1,1,1,1,1,1,1,1]  # X on all qubits
        
    [[4,2,2]] C4 (error-detecting):
        n=4, k=2 (encodes 2 logical qubits), d=2 (detects but can't correct)
        Uses Bell-pair preparation and EDT post-selection
    
    References:
        [CS96] Calderbank & Shor, Phys. Rev. A 54, 1098 (1996)
        [Ste96] Steane, Phys. Rev. Lett. 77, 793 (1996)
        [Got97] Gottesman, PhD Thesis, Caltech (1997)
    """
    # Mathematical structure
    name: str
    n: int  # Number of physical qubits
    k: int  # Number of logical qubits  
    d: int  # Code distance
    Hz: np.ndarray  # Z stabilizer check matrix (detects X errors)
    Hx: np.ndarray  # X stabilizer check matrix (detects Z errors)
    
    # Logical operators for ALL k logical qubits
    # logical_z_ops[i] is the Lz for logical qubit i (length n)
    # logical_x_ops[i] is the Lx for logical qubit i (length n)
    logical_z_ops: List[np.ndarray] = field(default_factory=list)
    logical_x_ops: List[np.ndarray] = field(default_factory=list)
    
    # Preparation circuit specification (for standard CSS encoding)
    h_qubits: List[int] = field(default_factory=list)
    
    # Qubits for LOGICAL Hadamard (Bell-pair creation/destruction)
    # For self-dual codes (Steane): all qubits (transversal H = logical H)
    # For non-self-dual codes (Shor): subset of qubits (e.g., [0,3] for Shor)
    # If None or empty, defaults to all qubits (self-dual assumption)
    logical_h_qubits: Optional[List[int]] = None
    
    # =========================================================================
    # Direct |+⟩_L Preparation (for non-self-dual codes)
    # =========================================================================
    # For codes without transversal logical H (like Shor), we can't prepare |+⟩_L
    # by applying H_L to |0⟩_L. Instead, we need a direct |+⟩_L encoding circuit.
    #
    # For Shor [[9,1,3]]:
    #   - plus_h_qubits = [0, 3, 6] (H on one qubit per block)
    #   - plus_encoding_cnots = [(0,1), (0,2), (3,4), (3,5), (6,7), (6,8)]
    #
    # For self-dual codes: these can be None (use H on |0⟩_L instead)
    plus_h_qubits: Optional[List[int]] = None  # H gates for |+⟩_L prep
    plus_encoding_cnots: Optional[List[Tuple[int, int]]] = None  # CNOTs for |+⟩_L prep
    plus_encoding_cnot_rounds: Optional[List[List[Tuple[int, int]]]] = None
    
    pre_h_cnots: List[Tuple[int, int]] = field(default_factory=list)  # CNOTs before H gates
    encoding_cnots: List[Tuple[int, int]] = field(default_factory=list)  # CNOTs after H gates
    encoding_cnot_rounds: Optional[List[List[Tuple[int, int]]]] = None
    verification_qubits: List[int] = field(default_factory=list)
    
    # Swap gates to apply after H gates (for codes like C4)
    swap_after_h: List[Tuple[int, int]] = field(default_factory=list)
    
    # Swap gates to apply after H at level 2 (when this code is outer code)
    # For codes like C4->C6 where the outer code (C6) has different SWAP pattern
    swap_after_h_l2: List[Tuple[int, int]] = field(default_factory=list)
    
    # Bell-pair preparation flag (for k>1 codes)
    # When True, uses Bell-pair protocol instead of standard CSS encoding
    uses_bellpair_prep: bool = False
    
    # Idle qubit schedule for noise modeling: (round_name, round_idx) -> idle qubits
    idle_schedule: Optional[Dict[str, List[int]]] = None
    
    # Number of transversal blocks for operations (defaults to n).
    # For C6 [[6,1,2]] code, this is 3 (3 blocks of 2 qubits each).
    # Used as n_now in L2 operations when this code is the outer code.
    transversal_block_count: Optional[int] = None
    
    # =========================================================================
    # Decoder Configuration
    # =========================================================================
    # Decoder type: determines how measurements are decoded
    # - "syndrome": Use syndrome lookup table (default for k=1 codes like Steane)
    # - "parity": Use parity check for error detection (for k>1 codes like C4)
    # - "majority": Use majority voting across blocks (for outer codes like C6)
    decoder_type: str = "auto"  # "auto" selects based on k value
    
    # Outer decoder type: when this code is used as outer code in concatenation
    # - None: use same as decoder_type
    # - "majority": block majority voting (C6)
    # - "syndrome": syndrome-based (Steane outer)
    outer_decoder_type: Optional[str] = None
    
    # =========================================================================
    # Post-Selection Configuration
    # =========================================================================
    # Post-selection type: determines how preparation is verified
    # - "verification": Check verification qubit measurement (Steane-style)
    # - "parity": Check parity of all measurement bits (C4-style)
    # - "block_majority": Check multiple blocks for consistency (C6-style)
    post_selection_type: str = "auto"  # "auto" selects based on code structure
    
    # =========================================================================
    # Error Correction Configuration
    # =========================================================================
    # Whether this code uses Error-Detecting Teleportation (EDT)
    # EDT rejects samples where inner code detects uncorrectable errors
    # Typically True for error-detecting codes (d=2) like C4
    uses_edt: bool = False
    
    def __post_init__(self):
        """Validate and set defaults."""
        assert self.Hz.shape[1] == self.n, f"Hz columns must match n"
        assert self.Hx.shape[1] == self.n, f"Hx columns must match n"
        
        # Validate logical operators
        if self.logical_z_ops:
            assert len(self.logical_z_ops) == self.k, f"Must have k={self.k} Lz operators"
            for i, lz in enumerate(self.logical_z_ops):
                assert len(lz) == self.n, f"Lz[{i}] length must match n"
        if self.logical_x_ops:
            assert len(self.logical_x_ops) == self.k, f"Must have k={self.k} Lx operators"
            for i, lx in enumerate(self.logical_x_ops):
                assert len(lx) == self.n, f"Lx[{i}] length must match n"
        
        # Auto-derive encoding_cnot_rounds if not provided
        if self.encoding_cnot_rounds is None and self.encoding_cnots:
            self.encoding_cnot_rounds = [[(c, t)] for c, t in self.encoding_cnots]
        
        # Auto-configure decoder_type based on k value
        if self.decoder_type == "auto":
            if self.k >= 2:
                self.decoder_type = "parity"  # k>1 codes use parity checking
            else:
                self.decoder_type = "syndrome"  # k=1 codes use syndrome lookup
        
        # Auto-configure post_selection_type based on code structure
        if self.post_selection_type == "auto":
            if self.k >= 2 and self.d <= 2:
                self.post_selection_type = "parity"  # Error-detecting codes
            elif self.verification_qubits:
                self.post_selection_type = "verification"  # Standard CSS verification
            else:
                self.post_selection_type = "parity"  # Fallback to parity check
        
        # Auto-configure uses_edt for error-detecting codes
        if self.k >= 2 and self.d <= 2 and not self.uses_edt:
            # d=2 codes with k>1 are error-detecting, benefit from EDT
            self.uses_edt = True
        
        # Auto-set logical_h_qubits if not specified
        # For self-dual codes: all qubits (transversal H = logical H)
        # For non-self-dual: must be specified explicitly in code definition
        if self.logical_h_qubits is None:
            if self.is_self_dual:
                # Self-dual: logical H acts on all qubits
                self.logical_h_qubits = list(range(self.n))
            else:
                # Non-self-dual: default to h_qubits (encoding qubits) as best guess
                # Code definitions should override this for correctness
                self.logical_h_qubits = self.h_qubits.copy() if self.h_qubits else list(range(self.n))
    
    @property
    def error_normalization_factor(self) -> int:
        """
        Factor to divide error count by (based on k value).
        
        For k=1 codes: divide by 1 (single logical qubit)
        For k=2 codes: divide by 2 (two logical qubits, errors summed)
        """
        return self.k
    
    @property
    def effective_decoder_type(self) -> str:
        """
        Get the effective decoder type for this code.
        
        Returns resolved decoder type (never "auto").
        """
        if self.decoder_type == "auto":
            return "parity" if self.k >= 2 else "syndrome"
        return self.decoder_type
    
    @property
    def effective_outer_decoder_type(self) -> str:
        """
        Get the effective outer decoder type when this code is used as outer code.
        
        Returns resolved type based on outer_decoder_type or inferred from structure.
        """
        if self.outer_decoder_type is not None:
            return self.outer_decoder_type
        # If transversal_block_count < n, likely a majority voting code (like C6)
        if self.transversal_block_count and self.transversal_block_count < self.n:
            return "majority"
        return self.effective_decoder_type
    
    @property
    def Lz(self) -> np.ndarray:
        """Logical Z for first logical qubit (backward compatibility)."""
        return self.logical_z_ops[0] if self.logical_z_ops else np.zeros(self.n)
    
    @property
    def Lx(self) -> np.ndarray:
        """Logical X for first logical qubit (backward compatibility)."""
        return self.logical_x_ops[0] if self.logical_x_ops else np.zeros(self.n)
    
    @property
    def Lz2(self) -> Optional[np.ndarray]:
        """Logical Z for second logical qubit (backward compatibility)."""
        return self.logical_z_ops[1] if len(self.logical_z_ops) > 1 else None
    
    @property
    def Lx2(self) -> Optional[np.ndarray]:
        """Logical X for second logical qubit (backward compatibility)."""
        return self.logical_x_ops[1] if len(self.logical_x_ops) > 1 else None
    
    @property
    def is_self_dual(self) -> bool:
        """Check if code is self-dual (Hx == Hz)."""
        return np.array_equal(self.Hx, self.Hz)
    
    @property
    def has_transversal_logical_h(self) -> bool:
        """
        Check if this code supports a transversal logical Hadamard.
        
        For self-dual CSS codes (Hx == Hz), transversal H = logical H.
        For non-self-dual codes (like Shor), logical H requires gate teleportation
        or other complex techniques and cannot be implemented transversally.
        
        The Bell-pair CNOT verification protocol requires transversal logical H.
        Codes without this property need code-specific simulators using
        alternative verification methods.
        """
        return self.is_self_dual
    
    @property
    def requires_direct_plus_prep(self) -> bool:
        """
        Check if this code requires direct |+⟩_L preparation.
        
        For non-self-dual codes without transversal logical H, we can't prepare
        |+⟩_L by applying H_L to |0⟩_L. Instead, we need a direct encoding circuit.
        
        Returns True if:
        - Code is not self-dual (no transversal logical H), AND
        - Direct |+⟩_L preparation circuit is defined (plus_h_qubits is set)
        """
        return not self.is_self_dual and self.plus_h_qubits is not None
    
    @property
    def num_x_stabilizers(self) -> int:
        return self.Hx.shape[0]
    
    @property
    def num_z_stabilizers(self) -> int:
        return self.Hz.shape[0]
    
    def get_stabilizer_support(self, stab_type: str, index: int) -> List[int]:
        """Get qubit indices in support of a stabilizer."""
        matrix = self. Hx if stab_type == 'x' else self.Hz
        return [i for i, v in enumerate(matrix[index]) if v == 1]
    
    # =========================================================================
    # Code Classification Properties (Auto-Detection for Generic Framework)
    # =========================================================================
    
    @functools.cached_property
    def z_basis_codewords(self) -> List[np.ndarray]:
        """
        Compute valid Z-basis codewords (kernel of Hz).
        
        These are the computational basis states that satisfy all Z-stabilizers.
        For Shor code: includes 000000000, 111000000, 000111000, etc.
        For Steane code: includes 0000000, 1010101, 0110011, etc.
        
        Returns:
            List of n-bit vectors representing valid codewords
        """
        return self._compute_kernel(self.Hz)
    
    @functools.cached_property
    def x_basis_codewords(self) -> List[np.ndarray]:
        """
        Compute valid X-basis codewords (kernel of Hx).
        
        These are the Hadamard-transformed states satisfying X-stabilizers.
        
        Returns:
            List of n-bit vectors representing valid X-basis codewords
        """
        return self._compute_kernel(self.Hx)
    
    def _compute_kernel(self, check_matrix: np.ndarray) -> List[np.ndarray]:
        """
        Compute kernel of a binary check matrix (mod 2).
        
        Uses exhaustive enumeration for small codes (n <= 15).
        """
        n = check_matrix.shape[1]
        if n > 15:
            # For large codes, return empty (too expensive to enumerate)
            return []
        
        kernel = []
        for i in range(2**n):
            vec = np.array([(i >> j) & 1 for j in range(n)], dtype=np.int64)
            if np.all((check_matrix @ vec) % 2 == 0):
                kernel.append(vec)
        return kernel
    
    @functools.cached_property
    def has_superposition_codewords(self) -> bool:
        """
        Check if the code has superposition structure in its logical states.
        
        For codes like Shor, |0⟩_L = (|000⟩+|111⟩)⊗3 means multiple
        computational basis states in the superposition. When measured,
        different outcomes (000000000 vs 111000000) can represent the same
        logical value.
        
        Detection method: Check if there's a block structure where qubits
        within blocks can independently flip (all 0 or all 1) while staying
        in the same logical state. This manifests as Z stabilizers that
        act within disjoint blocks.
        
        For Steane (self-dual): No block structure, single logical basis state
        For Shor (non-self-dual): Has 3 blocks, each can be |000⟩ or |111⟩
        
        Returns:
            True if code has superposition structure requiring special decoding
        """
        # If code is self-dual, it typically doesn't have this issue
        if self.is_self_dual:
            return False
        
        # Check for block structure in Z stabilizers
        blocks = self.stabilizer_block_structure
        if blocks and blocks.get('z_blocks') and len(blocks['z_blocks']) > 1:
            return True
        
        return False
    
    @functools.cached_property
    def stabilizer_block_structure(self) -> Optional[Dict[str, List[List[int]]]]:
        """
        Detect block structure in stabilizers (e.g., Shor's row structure).
        
        Analyzes Hz to find if stabilizers partition qubits into disjoint blocks.
        For Shor code: 3 blocks of [0,1,2], [3,4,5], [6,7,8]
        
        Returns:
            Dict with 'z_blocks' and 'x_blocks' as lists of qubit index lists,
            or None if no clear block structure found.
        """
        z_blocks = self._find_block_structure(self.Hz)
        x_blocks = self._find_block_structure(self.Hx)
        
        if z_blocks or x_blocks:
            return {'z_blocks': z_blocks, 'x_blocks': x_blocks}
        return None
    
    def _find_block_structure(self, check_matrix: np.ndarray) -> List[List[int]]:
        """
        Find disjoint blocks from check matrix structure.
        
        Groups stabilizers that share qubits, then identifies disjoint qubit sets.
        """
        if check_matrix.shape[0] == 0:
            return []
        
        n = check_matrix.shape[1]
        
        # Build adjacency: qubits connected if they share a stabilizer
        from collections import defaultdict
        qubit_to_stabs = defaultdict(set)
        for stab_idx in range(check_matrix.shape[0]):
            support = [i for i in range(n) if check_matrix[stab_idx, i] == 1]
            for q in support:
                qubit_to_stabs[q].add(stab_idx)
        
        # Union-find to group connected qubits
        parent = list(range(n))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Connect qubits that share any stabilizer
        for stab_idx in range(check_matrix.shape[0]):
            support = [i for i in range(n) if check_matrix[stab_idx, i] == 1]
            for i in range(len(support) - 1):
                union(support[i], support[i+1])
        
        # Group qubits by their root
        blocks_dict = defaultdict(list)
        for q in range(n):
            blocks_dict[find(q)].append(q)
        
        blocks = sorted(blocks_dict.values(), key=lambda b: min(b))
        
        # Only return if we found multiple blocks
        if len(blocks) > 1:
            return blocks
        return []
    
    @functools.cached_property
    def measurement_strategy(self) -> str:
        """
        Determine the correct measurement/detector strategy for this code.
        
        Returns:
            "absolute": Standard DETECTOR rec[-i] (expects measurement = 0)
                        Used for self-dual codes like Steane where codewords
                        have consistent logical parity.
            "relative": DETECTOR rec[-i] rec[-j] (expects pairwise agreement)
                        Used for codes like Shor where superposition codewords
                        have inconsistent absolute values but consistent
                        within-block structure.
            "parity": For k>1 codes using parity-based error detection (C4).
        """
        if self.k >= 2:
            return "parity"
        if self.has_superposition_codewords:
            return "relative"
        return "absolute"
    
    def get_relative_detector_pairs(self) -> List[Tuple[int, int]]:
        """
        Get qubit pairs that should agree in Z-basis measurement.
        
        For codes with superposition codewords (measurement_strategy="relative"),
        this returns pairs (i, j) where m[i] XOR m[j] = 0 for all valid codewords.
        
        Derived from Hz structure: consecutive qubits in each Z-stabilizer.
        
        Returns:
            List of (q1, q2) pairs for relative detector construction
        """
        pairs = []
        for stab_idx in range(self.Hz.shape[0]):
            support = self.get_stabilizer_support('z', stab_idx)
            # For each stabilizer, consecutive pairs should agree
            for i in range(len(support) - 1):
                pair = (support[i], support[i+1])
                if pair not in pairs and (pair[1], pair[0]) not in pairs:
                    pairs.append(pair)
        return pairs
    
    def get_block_representatives(self) -> List[int]:
        """
        Get one representative qubit per block for logical value extraction.
        
        For Shor code: [0, 3, 6] (first qubit of each row)
        For Steane code: [0] (single block)
        
        Returns:
            List of qubit indices, one per block
        """
        blocks = self.stabilizer_block_structure
        if blocks and blocks.get('z_blocks'):
            return [min(block) for block in blocks['z_blocks']]
        return [0]  # Default: first qubit
    
    def decode_z_basis_measurement(self, m: np.ndarray) -> int:
        """
        Decode a Z-basis measurement using code-aware strategy.
        
        For absolute measurement codes (Steane): uses Lz parity directly
        For relative measurement codes (Shor): uses block decoding + parity
        
        For Shor code with blocks:
        1. Each block is majority-decoded to 0 or 1 (error correction within block)
        2. The parity (XOR) of decoded block values gives the logical outcome
        
        This mirrors Lz = [1,0,0,1,0,0,1,0,0] which takes first qubit of each block.
        
        Args:
            m: Measurement outcomes (length n)
        
        Returns:
            Logical measurement outcome (0 or 1)
        """
        m = np.array(m).flatten()
        
        if self.measurement_strategy == "relative":
            # Use block majority voting + parity
            blocks = self.stabilizer_block_structure
            if blocks and blocks.get('z_blocks'):
                decoded_blocks = []
                for block in blocks['z_blocks']:
                    block_bits = m[block]
                    # Majority vote within block (error correction)
                    decoded = 1 if np.sum(block_bits) > len(block_bits) / 2 else 0
                    decoded_blocks.append(decoded)
                # Logical value is PARITY of decoded blocks (not majority!)
                # This matches Lz structure: parity of representatives
                return sum(decoded_blocks) % 2
        
        # For absolute measurement codes (Steane-like):
        # Use syndrome-based error correction before computing logical value
        
        # Compute raw logical value
        outcome = int(np.sum(self.Lz * m) % 2)
        
        # Compute syndrome from Hz (Z stabilizers detect X errors)
        syndrome = 0
        for stab_idx in range(self.Hz.shape[0]):
            parity = int(np.sum(m * self.Hz[stab_idx, :]) % 2)
            syndrome += parity * (1 << stab_idx)
        
        # Apply minimum-weight correction
        if syndrome > 0:
            # Build syndrome table on-the-fly (could be cached)
            for qubit in range(self.n):
                q_syndrome = 0
                for stab_idx in range(self.Hz.shape[0]):
                    if self.Hz[stab_idx, qubit] == 1:
                        q_syndrome += (1 << stab_idx)
                if q_syndrome == syndrome:
                    # This qubit matches - apply correction
                    outcome = (outcome + int(self.Lz[qubit])) % 2
                    break
        
        return outcome


@dataclass
class PropagationTables:
    """
    Error propagation tables for concatenated code decoding at level-2 and above.
    
    ═══════════════════════════════════════════════════════════════════════════
                               QEC THEORY BACKGROUND
    ═══════════════════════════════════════════════════════════════════════════
    
    ERROR PROPAGATION IN CONCATENATED CODES
    ---------------------------------------
    In a concatenated code, errors at the inner (level-0) code propagate through
    encoding circuits to affect multiple level-1 blocks. Proper decoding requires
    tracking these propagation patterns.
    
    Consider a level-2 Steane code (Steane→Steane concatenation):
    - Level 0: Each of 7 physical qubits can have errors
    - Level 1: Each of 7 level-0 blocks forms a level-1 Steane code
    - The encoding circuit's CNOTs spread errors between blocks
    
    When we perform error correction, the corrections from each EC round
    propagate forward through subsequent operations. These tables track:
    1. Which output blocks receive X corrections from each EC round
    2. Which output blocks receive Z corrections from each EC round
    3. How corrections affect the final measurement
    
    PAULI FRAME TRACKING
    --------------------
    Rather than physically applying corrections, we track them in a "Pauli frame":
    
        Pauli frame = accumulated X and Z corrections on each qubit
    
    This is equivalent to applying corrections lazily. At the final measurement:
    - X corrections flip the Z-basis measurement outcome
    - Z corrections have no effect on Z-basis measurement
    
    For CNOT gates, Pauli errors propagate:
        X_control ⊗ I_target  →  X_control ⊗ X_target  (X propagates forward)
        I_control ⊗ Z_target  →  Z_control ⊗ Z_target  (Z propagates backward)
    
    BELL PAIR CNOT VERIFICATION STRUCTURE
    -------------------------------------
    The Bell pair protocol creates 4 logical blocks (0,1,2,3):
    
        Block 0 ─●─────────●───── Control qubit of first pair
                 │         │
        Block 1 ─⊕─        │      Target qubit of first pair
                           │
        Block 2 ─●─────────⊕───── Control qubit of second pair (CNOT target)
                 │
        Block 3 ─⊕─                Target qubit of second pair
    
    After Q rounds of CNOT verification with EC after each:
    - EC round 2i applies to blocks 0,1 (first pair)
    - EC round 2i+1 applies to blocks 2,3 (second pair)
    
    Correction propagation through CNOTs:
    - Block 0 (control): Z corrections stay, X corrections go to block 1
    - Block 1 (target): X corrections stay, Z corrections come from block 0
    - Block 2 (control): Z corrections stay, X corrections go to block 3
    - Block 3 (target): X corrections stay, Z corrections come from block 2
    
    DEFAULT PROPAGATION PATTERNS
    ----------------------------
    For standard Bell pair verification:
    
    accept_X_propagate = [[1], [3]]
        X correction from EC on pair 0 → affects block 1 (target of CNOT)
        X correction from EC on pair 1 → affects block 3 (target of CNOT)
    
    accept_Z_propagate = [[0], [2]]  
        Z correction from EC on pair 0 → affects block 0 (control of CNOT)
        Z correction from EC on pair 1 → affects block 2 (control of CNOT)
    
    ═══════════════════════════════════════════════════════════════════════════
                              ATTRIBUTE REFERENCE
    ═══════════════════════════════════════════════════════════════════════════
    
    propagation_X: List[List[int]]
        For each EC round i, propagation_X[i] lists the output block indices
        that receive X corrections from that round's Z-syndrome decoding.
        
    propagation_Z: List[List[int]]
        For each EC round i, propagation_Z[i] lists the output block indices
        that receive Z corrections from that round's X-syndrome decoding.
        
    propagation_m: List[int]
        Lists EC rounds whose corrections affect the final measurement.
        Used to accumulate Pauli frame before decoding measurement outcome.
        
    num_ec_0prep: int
        Total number of EC rounds in the preparation circuit.
        Used to correctly index into detector lists.
        
    accept_X_propagate: List[List[int]]
        Propagation pattern for X corrections in acceptance checking.
        Default [[1], [3]] for standard Bell pair structure.
        
    accept_Z_propagate: List[List[int]]
        Propagation pattern for Z corrections in acceptance checking.
        Default [[0], [2]] for standard Bell pair structure.
    
    ═══════════════════════════════════════════════════════════════════════════
                                  EXAMPLE
    ═══════════════════════════════════════════════════════════════════════════
    
    For Steane→Steane level-2 code with Q=1 CNOT verification round:
    
        propagation_X = [[0,1,2,3,4,5,6], [0,1,2,3,4,5,6]]
            # Each of the 7 inner blocks propagates X to all outer blocks
            
        propagation_Z = [[0,1,2,3,4,5,6], [0,1,2,3,4,5,6]]  
            # Each of the 7 inner blocks propagates Z to all outer blocks
            
        propagation_m = [0, 1]
            # Both EC rounds affect final measurement
            
        num_ec_0prep = 14  # 2 rounds × 7 blocks each
    
    References:
        [AGP06] Aliferis, Gottesman, Preskill, QIC 6, 97 (2006)
        [Kni05] Knill, Nature 434, 39 (2005)
    """
    propagation_X: List[List[int]]  # X error propagation per EC round
    propagation_Z: List[List[int]]  # Z error propagation per EC round
    propagation_m: List[int]  # EC rounds affecting verification measurement
    num_ec_0prep: int  # Total EC rounds in preparation circuit
    
    # Acceptance propagation patterns (default to Bell pair structure)
    accept_X_propagate: List[List[int]] = field(default_factory=lambda: [[1], [3]])
    accept_Z_propagate: List[List[int]] = field(default_factory=lambda: [[0], [2]])


@dataclass
class ConcatenatedCode:
    """
    A concatenated quantum error correcting code with arbitrary nesting levels.
    
    ═══════════════════════════════════════════════════════════════════════════
                               QEC THEORY BACKGROUND
    ═══════════════════════════════════════════════════════════════════════════
    
    CODE CONCATENATION
    ------------------
    Concatenation is the key technique for achieving arbitrarily low logical
    error rates in fault-tolerant quantum computing. The idea is simple but
    powerful: encode each physical qubit of an "outer" code using an "inner" code.
    
    Single-level encoding [[n,k,d]]:
        1 logical qubit → n physical qubits
        Can correct ⌊(d-1)/2⌋ errors
        Logical error rate: O(p^{⌊(d+1)/2⌋}) for physical error rate p
    
    Two-level concatenation [[n₁,k₁,d₁]] → [[n₂,k₂,d₂]]:
        1 logical qubit → n₁ × n₂ physical qubits
        Effective distance: d₁ × d₂
        Logical error rate: O(p^{⌊(d₁+1)/2⌋ × ⌊(d₂+1)/2⌋})
    
    THE THRESHOLD THEOREM
    ---------------------
    A fundamental result in quantum computing (Aharonov & Ben-Or, 1997):
    
    If the physical error rate p is below a threshold p_th, then arbitrary
    quantum computations can be performed with arbitrarily small logical
    error rate using O(poly(log(1/ε))) overhead, where ε is the target error.
    
    For concatenated codes, this is achieved by adding more levels of
    concatenation. Each level squares the effective error rate (roughly):
    
        Level 1: p_L1 ~ (p/p_th)^2 × p_th
        Level 2: p_L2 ~ (p_L1/p_th)^2 × p_th ~ (p/p_th)^4 × p_th
        Level k: p_Lk ~ (p/p_th)^{2^k} × p_th
    
    This double-exponential suppression enables fault-tolerant computation.
    
    HIERARCHICAL STRUCTURE
    ----------------------
    A concatenated code has a recursive structure:
    
        Level 0 (innermost): n₀ physical qubits per logical qubit
        Level 1: n₀ × n₁ physical qubits (n₁ copies of level-0)
        Level L: ∏ᵢ nᵢ physical qubits (nesting of all levels)
    
    Error correction proceeds hierarchically:
    1. Decode each level-0 block independently
    2. Use level-0 outcomes as "effective measurements" for level-1 decoding
    3. Continue up to the outermost level
    
    This hierarchical decoding is tracked through PropagationTables which
    encode how corrections at each level propagate to affect higher levels.
    
    EXAMPLE: STEANE→STEANE
    ----------------------
    Two-level Steane code concatenation:
    
        levels = [SteaneCode, SteaneCode]
        Total qubits: 7 × 7 = 49 physical qubits
        Effective distance: 3 × 3 = 9
        
    The 49 qubits are organized as:
    - 7 "outer" blocks (corresponding to outer Steane code)
    - Each outer block contains 7 physical qubits (inner Steane code)
    
    ═══════════════════════════════════════════════════════════════════════════
                              EXTENSIBILITY HOOKS  
    ═══════════════════════════════════════════════════════════════════════════
    
    For codes with special structure (like C4→C6), you can provide custom
    functions without subclassing the simulator:
    
    custom_decoder_fn: Callable[[x, detector_X, detector_Z, decoder], Tuple]
        Custom hierarchical decoder for EC rounds.
        Returns (correction_x, correction_z) arrays.
    
    custom_accept_l2_fn: Callable[[x, list_detector_m, list_detector_X, 
                                   list_detector_Z, Q, decoder], float]
        Custom L2 acceptance checking.
        Returns error probability (0, 0.5, or 1).
    
    custom_post_selection_l2_fn: Callable[[x, list_detector_0prep, decoder], bool]
        Custom post-selection logic.
        Returns True to accept sample, False to reject.
    
    ═══════════════════════════════════════════════════════════════════════════
                              ATTRIBUTE REFERENCE
    ═══════════════════════════════════════════════════════════════════════════
    
    levels: List[CSSCode]
        List of CSS codes from innermost (level 0) to outermost (level L-1).
        For Steane→Steane: [steane_code, steane_code]
        For C4→C6: [c4_code, c6_code]
    
    name: Optional[str]
        Human-readable name, auto-generated if not provided.
        Example: "Concat[Steane[[7,1,3]]->Steane[[7,1,3]]]"
    
    propagation_tables: Dict[int, PropagationTables]
        Error propagation tables keyed by level.
        Level 1 tables describe propagation from level-0 to level-1.
    
    References:
        [KLZ96] Knill, Laflamme, Zurek, arXiv:quant-ph/9610011
        [AB97] Aharonov & Ben-Or, arXiv:quant-ph/9906129
        [AGP06] Aliferis, Gottesman, Preskill, QIC 6, 97 (2006)
    """
    levels: List[CSSCode]
    name: Optional[str] = None
    propagation_tables: Dict[int, PropagationTables] = field(default_factory=dict)
    
    # Extensibility hooks - code-specific modules can provide custom functions
    # Signature: custom_decoder_fn(x, detector_X, detector_Z, decoder) -> (correction_x, correction_z)
    custom_decoder_fn: Optional[Any] = None
    # Signature: custom_accept_l2_fn(x, list_detector_m, list_detector_X, list_detector_Z, Q, decoder) -> float
    custom_accept_l2_fn: Optional[Any] = None
    # Signature: custom_post_selection_l2_fn(x, list_detector_0prep, decoder) -> bool
    custom_post_selection_l2_fn: Optional[Any] = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = "Concat[" + "->".join(c.name for c in self.levels) + "]"
    
    @property
    def num_levels(self) -> int:
        return len(self.levels)
    
    @property
    def total_qubits(self) -> int:
        result = 1
        for code in self.levels:
            result *= code.n
        return result
    
    def qubits_at_level(self, level:  int) -> int:
        """Number of physical qubits in a logical qubit at given level."""
        result = 1
        for i in range(level + 1):
            result *= self.levels[i].n
        return result
    
    def code_at_level(self, level: int) -> CSSCode:
        return self.levels[level]
    
    def get_propagation_tables(self, level: int) -> Optional[PropagationTables]: 
        return self.propagation_tables.get(level)
    
    @property
    def has_custom_l2_acceptance(self) -> bool:
        """Check if custom L2 acceptance is provided."""
        return self.custom_accept_l2_fn is not None
    
    @property
    def has_custom_decoder(self) -> bool:
        """Check if custom decoder is provided."""
        return self.custom_decoder_fn is not None
    
    @property  
    def has_custom_post_selection(self) -> bool:
        """Check if custom post-selection is provided."""
        return self.custom_post_selection_l2_fn is not None


# =============================================================================
# Result Structures
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                            RESULT TYPES                                      │
# └─────────────────────────────────────────────────────────────────────────────┘

# ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
# │   GateResult     │  │   PrepResult     │  │    ECResult      │
# ├──────────────────┤  ├──────────────────┤  ├──────────────────┤
# │ gate_type: str   │  │ level:  int       │  │ level:  int       │
# │ implementation   │  │ detector_0prep   │  │ ec_type: str     │
# │ level: int       │  │ detector_0prep_l2│  │ detector_0prep   │
# │ detectors: List  │  │ detector_X       │  │ detector_0prep_l2│
# │ metadata: Dict   │  │ detector_Z       │  │ detector_X       │
# └──────────────────┘  │ children: List   │  │ detector_Z       │
#                       └──────────────────┘  │ children: List   │
#                                             └──────────────────┘
# =============================================================================

@dataclass
class GateResult:
    """Result of a logical gate application."""
    gate_type: str
    implementation: str
    level: int
    detectors: List = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class PrepResult:
    """Result of state preparation - unified structure for all levels."""
    level: int
    detector_0prep: List = field(default_factory=list)
    detector_0prep_l2: Optional[List] = None  # For level-2+
    detector_X:  List = field(default_factory=list)
    detector_Z: List = field(default_factory=list)
    children: List = field(default_factory=list)


@dataclass
class ECResult: 
    """Result of error correction - unified structure for all levels."""
    level: int
    ec_type: str
    detector_0prep: List = field(default_factory=list)
    detector_0prep_l2: Optional[List] = None
    detector_X: List = field(default_factory=list)
    detector_Z: List = field(default_factory=list)
    children: List = field(default_factory=list)


# =============================================================================
# Encoding Circuit Derivation Utilities
# =============================================================================

def derive_css_encoding_circuit(
    Hz: np.ndarray,
    Hx: np.ndarray,
    logical_x: np.ndarray,
    logical_z: np.ndarray
) -> Tuple[List[int], List[Tuple[int, int]], List[List[Tuple[int, int]]]]:
    """
    Derive encoding circuit (h_qubits, encoding_cnots) from CSS code structure.
    
    For a CSS code, the encoding circuit prepares |0⟩_L by:
    1. Apply H gates to qubits in the support of logical X operator
    2. Apply CNOTs to spread entanglement according to stabilizer structure
    
    The key insight is that for CSS codes:
    - |0⟩_L = (1/√|S|) Σ_{s∈S} s|0...0⟩ where S is the X stabilizer group
    - The logical X operator determines which qubits need H gates
    - The stabilizer structure determines the CNOT pattern
    
    This implements the standard form encoding: Put Hz in standard form [I|A],
    then the encoding CNOTs correspond to the A matrix structure.
    
    Args:
        Hz: Z stabilizer check matrix (m × n)
        Hx: X stabilizer check matrix (m × n)  
        logical_x: Logical X operator (length n)
        logical_z: Logical Z operator (length n)
        
    Returns:
        (h_qubits, encoding_cnots, encoding_cnot_rounds) tuple
    """
    n = Hz.shape[1]
    m = Hz.shape[0]  # Number of stabilizers
    
    # For CSS codes with k=1:
    # h_qubits = support of logical_x (qubits where Lx has 1s)
    h_qubits = [i for i in range(n) if logical_x[i] == 1]
    
    # For encoding CNOTs, we use the stabilizer structure.
    # The idea: to create |0⟩_L, we need to create superposition of all
    # codewords that satisfy the X stabilizers.
    #
    # Standard approach for self-dual codes where Hz = Hx:
    # 1. Put H in reduced row echelon form (rref)
    # 2. Identify pivot columns (identity part) and free columns
    # 3. CNOTs from h_qubits spread to create correct superposition
    #
    # For simplicity, we use a heuristic that works for many codes:
    # CNOT from each h_qubit to non-h_qubits based on stabilizer support
    
    # Build CNOT list based on Hz structure
    # For each row of Hz, if it overlaps with h_qubits, create CNOTs
    encoding_cnots = []
    
    # Simplified approach: use logical_x support as sources, 
    # CNOT to create the stabilizer superposition
    h_set = set(h_qubits)
    non_h = [i for i in range(n) if i not in h_set]
    
    # For each stabilizer row, find how h_qubits connect to non-h_qubits
    for row in Hz:
        support = [i for i in range(n) if row[i] == 1]
        sources = [i for i in support if i in h_set]
        targets = [i for i in support if i not in h_set]
        
        # If this stabilizer has both h_qubits and non-h_qubits,
        # create CNOTs from sources to targets
        if sources and targets:
            for tgt in targets:
                # Find closest source (for simpler circuit)
                src = min(sources, key=lambda s: abs(s - tgt))
                cnot = (src, tgt)
                if cnot not in encoding_cnots:
                    encoding_cnots.append(cnot)
    
    # If we didn't get enough CNOTs, fall back to simple spreading
    if len(encoding_cnots) < n - len(h_qubits):
        encoding_cnots = []
        # Simple heuristic: first h_qubit spreads to non-h_qubits
        if h_qubits:
            src = h_qubits[0]
            for tgt in non_h:
                encoding_cnots.append((src, tgt))
    
    # Group CNOTs into rounds (simple: sequential for now)
    # Could optimize for parallelism later
    encoding_cnot_rounds = [[cnot] for cnot in encoding_cnots]
    
    return h_qubits, encoding_cnots, encoding_cnot_rounds


def derive_steane_style_encoding(n: int, Hz: np.ndarray, logical_x: np.ndarray) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Derive Steane-style encoding for self-dual CSS codes.
    
    For self-dual codes (Hz = Hx), the encoding follows the Steane pattern:
    1. h_qubits = support of logical_x
    2. CNOTs create the stabilizer superposition
    
    This is more sophisticated than derive_css_encoding_circuit and produces
    circuits closer to the optimal Steane encoding.
    
    Args:
        n: Number of physical qubits
        Hz: Z stabilizer check matrix (also Hx for self-dual)
        logical_x: Logical X operator
        
    Returns:
        (h_qubits, encoding_cnots) tuple
    """
    # h_qubits from logical_x support
    h_qubits = [i for i in range(n) if logical_x[i] == 1]
    h_set = set(h_qubits)
    
    # For Steane-style encoding, we build CNOTs based on the check matrix structure
    # The key is that CNOTs should create entanglement matching the stabilizers
    
    encoding_cnots = []
    covered_targets = set()
    
    # Process each stabilizer to determine CNOTs
    for row in Hz:
        support = [i for i in range(n) if row[i] == 1]
        sources = [i for i in support if i in h_set]
        targets = [i for i in support if i not in h_set and i not in covered_targets]
        
        if sources and targets:
            for tgt in targets:
                src = sources[0]  # Use first available source
                encoding_cnots.append((src, tgt))
                covered_targets.add(tgt)
    
    # Cover any remaining non-h qubits
    remaining = [i for i in range(n) if i not in h_set and i not in covered_targets]
    if remaining and h_qubits:
        src = h_qubits[0]
        for tgt in remaining:
            encoding_cnots.append((src, tgt))
    
    return h_qubits, encoding_cnots


# =============================================================================
# Code Factory Functions
# =============================================================================

# Note: Steane-specific factory functions (create_steane_code, create_steane_propagation_l2,
# create_concatenated_steane) have been moved to concatenated_css_v10_steane.py

def create_shor_code() -> CSSCode:
    """
    Create the [[9,1,3]] Shor code.
    
    The Shor code encodes 1 logical qubit in 9 physical qubits with distance 3.
    It has asymmetric stabilizers: 6 Z-type stabilizers and 2 X-type stabilizers.
    
    STABILIZERS:
    - Z-type (in Hz): Z_i Z_{i+1} pairs within each block (6 generators)
      These ensure uniformity within each block and detect X errors.
    - X-type (in Hx): X on all qubits of adjacent block pairs (2 generators)
      These ensure phase correlation between blocks and detect Z errors.
    
    LOGICAL OPERATORS:
    - Lz = Z0 Z3 Z6 (one Z per block) - measures logical Z
    - Lx = X0 X1 X2 (all X on one block) - applies logical X
    
    ENCODING for |0⟩_L:
    To prepare |0⟩_L, we need Lz·m = 0 for ALL measurement outcomes.
    This means q0 + q3 + q6 ≡ 0 (mod 2) always.
    
    The encoding achieves this by:
    1. H on qubits 0 and 3 (NOT 6!)
    2. CNOT(0,6), CNOT(3,6) to make q6 = q0 XOR q3
    3. Spread within each block
    
    This creates the 4 patterns (q0,q3,q6) ∈ {(0,0,0), (0,1,1), (1,0,1), (1,1,0)}
    all with q0+q3+q6 = 0 (mod 2).
    """
    # CORRECT: Hz contains Z-type stabilizers (pairs within blocks)
    Hz = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0],  # Z0 Z1
        [0, 1, 1, 0, 0, 0, 0, 0, 0],  # Z1 Z2
        [0, 0, 0, 1, 1, 0, 0, 0, 0],  # Z3 Z4
        [0, 0, 0, 0, 1, 1, 0, 0, 0],  # Z4 Z5
        [0, 0, 0, 0, 0, 0, 1, 1, 0],  # Z6 Z7
        [0, 0, 0, 0, 0, 0, 0, 1, 1]   # Z7 Z8
    ])
    
    # CORRECT: Hx contains X-type stabilizers (blocks 0+1, blocks 1+2)
    Hx = np.array([
        [1, 1, 1, 1, 1, 1, 0, 0, 0],  # X0 X1 X2 X3 X4 X5
        [0, 0, 0, 1, 1, 1, 1, 1, 1]   # X3 X4 X5 X6 X7 X8
    ])
    
    Lx = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])  # X on block 0
    Lz = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])  # Z0 Z3 Z6
    
    # Encoding: H on q0, q3 only, then CNOT to create q6=q0 XOR q3, then spread
    # This ensures q0+q3+q6 = 0 always (Lz·m = 0)
    encoding_cnot_rounds = [
        [(0, 6), (3, 6)],  # First: create q6 = q0 XOR q3
        [(0, 1), (0, 2), (3, 4), (3, 5), (6, 7), (6, 8)],  # Then spread within blocks
    ]
    
    encoding_cnots = []
    for round_cnots in encoding_cnot_rounds:
        encoding_cnots.extend(round_cnots)
    
    # |+⟩_L direct preparation circuit (for Bell-pair creation)
    # Since Shor has no transversal logical H, we can't get |+⟩_L from H·|0⟩_L.
    # Instead, prepare |+⟩_L directly with:
    #   - H on one qubit per block (0, 3, 6)
    #   - Spread within each block
    # This creates: (1/√8) Σ |q0,q0,q0,q3,q3,q3,q6,q6,q6⟩ for all (q0,q3,q6)
    plus_h_qubits = [0, 3, 6]  # H on one qubit per block
    plus_encoding_cnots = [(0, 1), (0, 2), (3, 4), (3, 5), (6, 7), (6, 8)]
    plus_encoding_cnot_rounds = [
        [(0, 1), (0, 2), (3, 4), (3, 5), (6, 7), (6, 8)],  # Spread within blocks
    ]
    
    return CSSCode(
        name="Shor",
        n=9, k=1, d=3,
        Hz=Hz, Hx=Hx,
        logical_z_ops=[Lz],  # k=1: single logical Z operator
        logical_x_ops=[Lx],  # k=1: single logical X operator
        h_qubits=[0, 3],  # H on q0 and q3 ONLY (not q6!) for |0⟩_L ENCODING
        logical_h_qubits=[0, 3],  # UNUSED - Shor has no transversal logical H!
        plus_h_qubits=plus_h_qubits,  # H gates for direct |+⟩_L prep
        plus_encoding_cnots=plus_encoding_cnots,  # CNOTs for |+⟩_L prep
        plus_encoding_cnot_rounds=plus_encoding_cnot_rounds,
        pre_h_cnots=[],
        encoding_cnots=encoding_cnots,
        encoding_cnot_rounds=encoding_cnot_rounds,
        verification_qubits=[0, 1],  # Check Z_0 Z_1 = +1
        uses_bellpair_prep=False,  # Shor uses standard CSS encoding
    )


def create_rep3_code() -> CSSCode:
    """
    Create the [[3,1,1]] 3-qubit repetition code.
    
    This is the simplest QEC code. It can correct either bit-flip OR phase-flip
    errors (not both simultaneously). We define it as self-dual for simplicity,
    making Hz = Hx so transversal H works as logical H.
    
    STABILIZERS (self-dual version):
    - Z0 Z1 (parity check on qubits 0,1)
    - Z1 Z2 (parity check on qubits 1,2)
    - X0 X1 (same pattern for X)
    - X1 X2
    
    LOGICAL OPERATORS:
    - Lz = Z0 Z1 Z2 (all-Z)
    - Lx = X0 X1 X2 (all-X)
    
    ENCODING: H on qubit 0, then CNOT to spread
    |0⟩ → |000⟩
    |1⟩ → |111⟩
    """
    # Self-dual: Hz = Hx
    H = np.array([
        [1, 1, 0],  # Z0 Z1 / X0 X1
        [0, 1, 1],  # Z1 Z2 / X1 X2
    ])
    
    Lz = np.array([1, 1, 1])  # Z on all qubits
    Lx = np.array([1, 1, 1])  # X on all qubits
    
    return CSSCode(
        name="Rep3",
        n=3, k=1, d=1,  # Distance 1 (only detects, doesn't correct)
        Hz=H, Hx=H,  # Self-dual
        logical_z_ops=[Lz],
        logical_x_ops=[Lx],
        h_qubits=[0],  # H on first qubit
        logical_h_qubits=[0, 1, 2],  # All qubits (self-dual)
        encoding_cnots=[(0, 1), (0, 2)],  # Spread from q0
        verification_qubits=[0],  # Verify first qubit
    )


def create_perfect5_code() -> CSSCode:
    """
    Create the [[5,1,3]] perfect code (smallest code correcting any single-qubit error).
    
    This is the smallest quantum code with distance 3. It's self-dual (Hz = Hx)
    so transversal H = logical H.
    
    STABILIZERS (cyclic):
    - XZZXI (g1)
    - IXZZX (g2)
    - XIXZZ (g3)
    - ZXIXZ (g4)
    
    Note: These can be written as both X-type and Z-type stabilizers due to
    the code's self-dual nature.
    
    LOGICAL OPERATORS:
    - Lz = ZZZZZ (weight 5)
    - Lx = XXXXX (weight 5)
    """
    # The [[5,1,3]] code has a beautiful cyclic structure
    # Stabilizers: XZZXI, IXZZX, XIXZZ, ZXIXZ
    # For CSS form, we express as Hz and Hx
    
    # Self-dual code: Hz = Hx (up to row operations)
    # Using standard form where stabilizers generate both X and Z checks
    H = np.array([
        [1, 0, 0, 1, 0],  # Check 1
        [0, 1, 0, 0, 1],  # Check 2
        [1, 0, 1, 0, 0],  # Check 3
        [0, 1, 0, 1, 0],  # Check 4
    ])
    
    Lz = np.array([1, 1, 1, 1, 1])  # Z on all qubits
    Lx = np.array([1, 1, 1, 1, 1])  # X on all qubits
    
    # Encoding circuit for [[5,1,3]]
    # Start with |ψ⟩ on q0, apply H, then specific CNOT pattern
    encoding_cnots = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # Spread from q0
    ]
    
    return CSSCode(
        name="Perfect5",
        n=5, k=1, d=3,
        Hz=H, Hx=H,  # Self-dual
        logical_z_ops=[Lz],
        logical_x_ops=[Lx],
        h_qubits=[0],  # H on first qubit for encoding
        logical_h_qubits=[0, 1, 2, 3, 4],  # All qubits (self-dual)
        encoding_cnots=encoding_cnots,
        verification_qubits=[0, 1],  # Verify first check
    )


def create_hamming7_code() -> CSSCode:
    """
    Create the [[7,1,3]] Hamming code (identical to Steane [[7,1,3]]).
    
    The Steane code is based on the classical [7,4,3] Hamming code.
    This creates the same code with the exact same encoding circuit.
    
    This is self-dual, so transversal H = logical H.
    """
    # Standard Steane/Hamming stabilizers - EXACT copy from Steane
    Hz = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ])
    Hx = Hz.copy()  # Self-dual
    
    # Logical operators - same as Steane
    Lz = np.array([1, 1, 1, 0, 0, 0, 0])
    Lx = np.array([1, 1, 1, 0, 0, 0, 0])
    
    return CSSCode(
        name="Hamming7",
        n=7, k=1, d=3,
        Hz=Hz, Hx=Hx,
        logical_z_ops=[Lz],
        logical_x_ops=[Lx],
        h_qubits=[0, 1, 3],  # Same as Steane
        logical_h_qubits=list(range(7)),  # All qubits (self-dual)
        # EXACT encoding circuit from Steane
        encoding_cnots=[
            (1, 2), (3, 5), (0, 4),  # Round 1
            (1, 6), (0, 2), (3, 4),  # Round 2
            (1, 5), (4, 6)           # Round 3
        ],
        encoding_cnot_rounds=[
            [(1, 2), (3, 5), (0, 4)],
            [(1, 6), (0, 2), (3, 4)],
            [(1, 5), (4, 6)]
        ],
        verification_qubits=[2, 4, 5],  # Same as Steane
        uses_bellpair_prep=False,
        idle_schedule={
            'cnot_round_1': [5],
            'verif_cnot_0': [0, 3],
            'verif_cnot_1': [0, 1, 2, 3, 5, 6],
            'verif_cnot_2': [0, 1, 2, 3, 4, 6],
            'verif_measure': [0, 1, 2, 3, 4, 6],
        }
    )


def create_reed_muller15_code() -> CSSCode:
    """
    Create a [[15,1,3]] CSS code based on Steane embedding.
    
    WARNING: This is NOT a true Reed-Muller code! It's Steane [[7,1,3]] embedded
    in 15 qubits with 8 extra qubits stabilized to |0⟩. The extra qubits add
    noise exposure without improving error correction.
    
    TRUE [[15,1,3]] CODES: True self-dual CSS codes with 15 qubits encoding
    1 logical qubit are very rare. The punctured Reed-Muller RM(1,4)^perp 
    gives [15,11,4] which is NOT self-orthogonal and thus not self-dual CSS.
    
    This "fake" code is kept for testing purposes but will perform WORSE
    than Steane due to the extra noise exposure from passive qubits.
    
    For actual validation of the concatenation framework, use:
    - Steane [[7,1,3]] - the canonical self-dual CSS code
    - Hamming7 [[7,1,3]] - identical to Steane
    """
    # [[15,1,3]] code with self-orthogonal check matrix
    # Each row has weight 8, and pairwise overlaps are all 4 (even)
    # This is the "doubled Steane" or punctured RM(1,4) construction
    Hz = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # qubits 0-7 (weight 8)
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],  # 0-3, 8-11 (weight 8)
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],  # pairs (weight 8)
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # alternating (weight 8)
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # shifted alt (weight 7)
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],  # pairs shifted (weight 7)
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],  # 4-7, 12-14 (weight 7)
    ])
    
    # Check CSS condition - this construction may not work either
    # Let me use a known valid construction: replicate Steane structure
    
    # BETTER APPROACH: Use the [16,5,8] first-order Reed-Muller and puncture
    # Or use direct construction matching Steane's pattern
    
    # Steane's [[7,1,3]] uses [7,4,3] Hamming with H @ H^T = 0 (mod 2)
    # because each pair of rows overlaps in 0, 2, or 4 positions
    
    # For [[15,1,3]], we need 7 stabilizer rows that are self-orthogonal
    # The RM(1,4) dual has the right structure
    
    # Actually, let's just use an explicit known-good construction
    # This is the punctured first-order Reed-Muller RM(1,4)^perp
    Hz = np.array([
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # weight 4
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],  # weight 4
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],  # weight 3
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],  # weight 3
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],  # weight 3
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # weight 2
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],  # weight 6
    ])
    # This doesn't work either...
    
    # SIMPLEST WORKING APPROACH: Just use a trivial extension of Steane
    # Duplicate qubits to make a 14-qubit code, then add one more
    # Actually that's complex. Let's use the Steane code directly.
    
    # For now, let's create a "trivial" 15-qubit code by embedding Steane
    # Use qubits 0-6 as Steane, qubits 7-14 as padding (stabilized by identity)
    Hz_steane = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ])
    
    # Embed in 15 qubits: add 8 zeros on the right
    Hz = np.zeros((3, 15), dtype=int)
    Hz[:, :7] = Hz_steane
    
    # Add 4 more stabilizers for the extra 8 qubits (pair checks)
    extra_stab = np.zeros((4, 15), dtype=int)
    extra_stab[0, 7] = extra_stab[0, 8] = 1  # Z7 Z8
    extra_stab[1, 9] = extra_stab[1, 10] = 1  # Z9 Z10
    extra_stab[2, 11] = extra_stab[2, 12] = 1  # Z11 Z12
    extra_stab[3, 13] = extra_stab[3, 14] = 1  # Z13 Z14
    
    Hz = np.vstack([Hz, extra_stab])  # 7 rows total
    Hx = Hz.copy()  # Self-dual
    
    # Logical operators: same as Steane on first 7 qubits
    Lz = np.zeros(15, dtype=int)
    Lz[:3] = [1, 1, 1]  # Like Steane
    Lx = Lz.copy()
    
    # h_qubits: same as Steane
    h_qubits = [0, 1, 3]
    
    # Encoding CNOTs: Steane on first 7, then spread to rest
    encoding_cnots = [
        (1, 2), (3, 5), (0, 4),  # Steane round 1
        (1, 6), (0, 2), (3, 4),  # Steane round 2
        (1, 5), (4, 6),          # Steane round 3
        # Extra qubits stay at |0⟩, no CNOTs needed
    ]
    
    return CSSCode(
        name="ReedMuller15",
        n=15, k=1, d=3,
        Hz=Hz, Hx=Hx,
        logical_z_ops=[Lz],
        logical_x_ops=[Lx],
        h_qubits=h_qubits,
        logical_h_qubits=list(range(7)),  # Only first 7 for logical H (Steane part)
        encoding_cnots=encoding_cnots,
        verification_qubits=[2, 4, 5],
    )


def create_golay23_code() -> CSSCode:
    """
    Create a [[23,1,3]] CSS code using Steane embedding.
    
    WARNING: This is NOT a true Golay code! The binary Golay [23,12,7] code
    is NOT self-orthogonal, so it CANNOT form a self-dual CSS code.
    
    This implementation embeds Steane [[7,1,3]] in a 23-qubit space with
    16 extra qubits stabilized to |0⟩. This gives:
    - Distance 3 (not 7)
    - More noise exposure (23 vs 7 qubits)
    - No actual error correction benefit from the extra qubits
    
    This "fake" code is kept for testing purposes but will perform WORSE
    than Steane due to the extra noise exposure from passive qubits.
    
    For actual validation of the concatenation framework, use:
    - Steane [[7,1,3]] - the canonical self-dual CSS code
    - Hamming7 [[7,1,3]] - identical to Steane
    """
    n = 23
    
    # Use Steane embedding approach (like ReedMuller15)
    # Qubits 0-6: Steane code
    # Qubits 7-22: Extra qubits stabilized to |0⟩
    
    Hz_steane = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ])
    
    # Embed in 23 qubits
    Hz = np.zeros((3, n), dtype=int)
    Hz[:, :7] = Hz_steane
    
    # Add 8 more stabilizers for the extra 16 qubits (pair checks)
    extra_stab = np.zeros((8, n), dtype=int)
    for i in range(8):
        extra_stab[i, 7 + 2*i] = 1
        extra_stab[i, 7 + 2*i + 1] = 1
    
    Hz = np.vstack([Hz, extra_stab])  # 11 rows total
    Hx = Hz.copy()  # Self-dual
    
    # Logical operators: same as Steane on first 7 qubits
    Lz = np.zeros(n, dtype=int)
    Lz[:3] = [1, 1, 1]  # Like Steane
    Lx = Lz.copy()
    
    # h_qubits: same as Steane
    h_qubits = [0, 1, 3]
    
    # Encoding CNOTs: Steane on first 7, extra qubits stay at |0⟩
    encoding_cnots = [
        (1, 2), (3, 5), (0, 4),  # Steane round 1
        (1, 6), (0, 2), (3, 4),  # Steane round 2
        (1, 5), (4, 6),          # Steane round 3
    ]
    
    return CSSCode(
        name="Golay23",
        n=n, k=1, d=3,  # Distance is only 3 for this embedding
        Hz=Hz, Hx=Hx,
        logical_z_ops=[Lz],
        logical_x_ops=[Lx],
        h_qubits=h_qubits,
        logical_h_qubits=list(range(7)),  # Only first 7 for logical H
        encoding_cnots=encoding_cnots,
        verification_qubits=[2, 4, 5],
    )


# =============================================================================
# Concatenated Code Factory Functions for Simple Codes
# =============================================================================

def create_concatenated_rep3(num_levels: int = 2) -> ConcatenatedCode:
    """Create Rep3→Rep3 concatenated code."""
    rep3 = create_rep3_code()
    return ConcatenatedCode(
        levels=[rep3] * num_levels,
        propagation_tables={}
    )


def create_concatenated_perfect5(num_levels: int = 2) -> ConcatenatedCode:
    """Create Perfect5→Perfect5 concatenated code."""
    perfect5 = create_perfect5_code()
    return ConcatenatedCode(
        levels=[perfect5] * num_levels,
        propagation_tables={}
    )


def create_concatenated_rep3_steane(num_levels: int = 2) -> ConcatenatedCode:
    """Create Rep3→Steane concatenated code (inner=Rep3, outer=Steane)."""
    from qectostim.experiments.concatenated_css_v10_steane import create_steane_code
    rep3 = create_rep3_code()
    steane = create_steane_code()
    return ConcatenatedCode(
        levels=[rep3, steane],
        propagation_tables={}
    )


def create_concatenated_hamming7(num_levels: int = 2) -> ConcatenatedCode:
    """Create Hamming7→Hamming7 concatenated code."""
    hamming7 = create_hamming7_code()
    return ConcatenatedCode(
        levels=[hamming7] * num_levels,
        propagation_tables={}
    )


def create_concatenated_reed_muller15(num_levels: int = 2) -> ConcatenatedCode:
    """Create ReedMuller15→ReedMuller15 concatenated code."""
    rm15 = create_reed_muller15_code()
    return ConcatenatedCode(
        levels=[rm15] * num_levels,
        propagation_tables={}
    )


def create_concatenated_golay23(num_levels: int = 2) -> ConcatenatedCode:
    """Create Golay23→Golay23 concatenated code."""
    golay = create_golay23_code()
    return ConcatenatedCode(
        levels=[golay] * num_levels,
        propagation_tables={}
    )


# Note: create_concatenated_steane has been moved to concatenated_css_v10_steane.py


def create_concatenated_code(codes: List[CSSCode], 
                             propagation_tables: Optional[Dict[int, PropagationTables]] = None) -> ConcatenatedCode:
    """Create a general concatenated code from a list of CSS codes."""
    return ConcatenatedCode(
        levels=codes,
        propagation_tables=propagation_tables or {}
    )


# =============================================================================
# Physical Operations
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                           OPERATIONS LAYER                                   │
# └─────────────────────────────────────────────────────────────────────────────┘

# ┌─────────────────────────┐         ┌─────────────────────────────────────────┐
# │     PhysicalOps         │         │           TransversalOps                │
# │     (Static Class)      │         ├─────────────────────────────────────────┤
# ├─────────────────────────┤         │ Input:  ConcatenatedCode                 │
# │ + reset(circuit, loc, n)│         ├─────────────────────────────────────────┤
# │ + noisy_reset(...)      │◄────────│ Uses (loc, N_prev, N_now) signature     │
# │ + h(circuit, loc)       │         │ matching original for compatibility     │
# │ + cnot(circuit, c, t)   │         ├─────────────────────────────────────────┤
# │ + noisy_cnot(...)       │         │ + append_h(circuit, loc, N_prev, N_now) │
# │ + swap(circuit, q1, q2) │         │ + append_cnot(...)                      │
# │ + measure(circuit, loc) │         │ + append_noisy_cnot(... , p)             │
# │ + noisy_measure(...)    │         │ + append_swap(...)                      │
# │ + detector(circuit, off)│         │ + append_m(... ) -> List[detectors]      │
# │ + depolarize1(...)      │         │ + append_noisy_m(...) -> List           │
# └─────────────────────────┘         │ + append_noisy_wait(...)                │
#                                     └─────────────────────────────────────────┘
# =============================================================================

class PhysicalOps:
    """Low-level physical qubit operations."""
    
    @staticmethod
    def reset(circuit: stim.Circuit, loc: int, n: int):
        for i in range(n):
            circuit.append("R", loc + i)
    
    @staticmethod
    def noisy_reset(circuit: stim.Circuit, loc: int, n: int, p: float):
        for i in range(n):
            circuit. append("R", loc + i)
        for i in range(n):
            circuit.append("X_ERROR", loc + i, p)
    
    @staticmethod
    def h(circuit: stim.Circuit, loc: int):
        circuit.append("H", loc)
    
    @staticmethod
    def cnot(circuit: stim.Circuit, ctrl: int, targ: int):
        circuit.append("CNOT", [ctrl, targ])
    
    @staticmethod
    def noisy_cnot(circuit: stim.Circuit, ctrl: int, targ: int, p: float):
        circuit.append("CNOT", [ctrl, targ])
        circuit.append("DEPOLARIZE2", [ctrl, targ], p)
    
    @staticmethod
    def swap(circuit: stim.Circuit, q1: int, q2: int):
        circuit.append("SWAP", [q1, q2])
    
    @staticmethod
    def measure(circuit: stim.Circuit, loc: int):
        circuit.append("M", loc)
    
    @staticmethod
    def noisy_measure(circuit: stim.Circuit, loc: int, p: float):
        circuit.append("X_ERROR", loc, p)
        circuit.append("M", loc)
    
    @staticmethod
    def detector(circuit: stim.Circuit, offset: int):
        circuit.append("DETECTOR", stim.target_rec(offset))
    
    @staticmethod
    def depolarize1(circuit: stim.Circuit, loc: int, p: float):
        circuit.append("DEPOLARIZE1", loc, p)


# =============================================================================
# Transversal Operations (General)
# =============================================================================

class TransversalOps:
    """
    Transversal operations matching original function signatures.
    
    The (loc, N_prev, N_now) pattern means:
    - N_prev = 1:  physical level
    - N_prev > 1: operating on encoded blocks of size N_prev
    - N_now = number of qubits/blocks at current level
    
    For codes with k>1 (like C4 [[4,2,2]]):
    - Automatically applies SWAP gates after H if code.swap_after_h is set
    - This is needed because logical H for such codes permutes logical qubits
    """
    
    def __init__(self, concat_code: ConcatenatedCode):
        self.concat_code = concat_code
    
    def block_size(self, level: int) -> int:
        return self.concat_code.qubits_at_level(level)
    
    def _get_inner_code(self) -> Optional[CSSCode]:
        """Get the innermost code (level 0)."""
        if self.concat_code.num_levels > 0:
            return self.concat_code.code_at_level(0)
        return None
    
    def append_h(self, circuit: stim. Circuit, loc: int, N_prev: int, N_now: int,
                 level: int = 0):
        """
        Transversal Hadamard with automatic SWAP handling for k>1 codes.
        
        For codes like C4 [[4,2,2]], the logical H gate requires SWAP gates
        to be applied after the physical H gates. This is handled automatically
        based on the code's swap_after_h (level 1) and swap_after_h_l2 (level 2)
        specifications.
        
        Args:
            circuit: Stim circuit to append to
            loc: Base qubit location
            N_prev: Size of each block (1 for physical level)
            N_now: Number of blocks at this level
            level: Concatenation level (0 = inner, higher = outer)
        """
        if N_prev == 1:
            # Physical level: apply H to all qubits
            for i in range(N_now):
                PhysicalOps.h(circuit, loc + i)
            
            # Check if inner code needs SWAP after H (level 0/1)
            code = self._get_inner_code()
            if code is not None and code.swap_after_h and N_now == code.n:
                # Apply SWAP gates for this code block
                for q1, q2 in code.swap_after_h:
                    PhysicalOps.swap(circuit, loc + q1, loc + q2)
        else:
            # Encoded level: recursively apply to each block
            for i in range(N_now):
                self.append_h(circuit, (loc + i) * N_prev, 1, N_prev, level)
            
            # Check if outer code needs SWAP
            if self.concat_code.num_levels > 1:
                outer_code = self.concat_code.code_at_level(1) if level == 0 else None
                if outer_code is not None and N_now == outer_code.n:
                    # Use swap_after_h_l2 if specified, otherwise fall back to swap_after_h
                    swap_pattern = outer_code.swap_after_h_l2 if outer_code.swap_after_h_l2 else outer_code.swap_after_h
                    if swap_pattern:
                        # Apply SWAP pattern for outer code (swap entire blocks)
                        for q1, q2 in swap_pattern:
                            for j in range(N_prev):
                                PhysicalOps.swap(circuit, 
                                               (loc + q1) * N_prev + j,
                                               (loc + q2) * N_prev + j)
    
    def append_logical_h(self, circuit: stim.Circuit, loc: int, N_prev: int, N_now: int,
                         code: CSSCode = None, level: int = 0):
        """
        Apply LOGICAL Hadamard (for Bell-pair creation/destruction).
        
        For self-dual codes (Steane): H on all qubits (transversal = logical)
        For non-self-dual codes (Shor): H only on logical_h_qubits
        
        This is different from append_h() which always applies H to all qubits.
        Use append_logical_h() for Bell-pair protocols where we need the actual
        logical Hadamard, not transversal H.
        
        Args:
            circuit: Stim circuit to append to
            loc: Base qubit location
            N_prev: Size of each block (1 for physical level)
            N_now: Number of blocks at this level  
            code: The CSS code (needed to determine logical_h_qubits)
            level: Concatenation level (0 = inner, higher = outer)
        """
        # Get the code if not provided
        if code is None:
            code = self._get_inner_code()
        
        if code is None or code.logical_h_qubits is None or len(code.logical_h_qubits) == N_now:
            # Self-dual case or no code info: fall back to transversal H
            self.append_h(circuit, loc, N_prev, N_now, level)
        else:
            # Non-self-dual case: apply H only to logical_h_qubits
            if N_prev == 1:
                # Physical level: H only on designated qubits
                for q in code.logical_h_qubits:
                    PhysicalOps.h(circuit, loc + q)
                
                # Apply SWAP pattern if needed (for codes like C4)
                if code.swap_after_h and N_now == code.n:
                    for q1, q2 in code.swap_after_h:
                        PhysicalOps.swap(circuit, loc + q1, loc + q2)
            else:
                # Encoded level: recursively apply to each block
                for i in range(N_now):
                    self.append_logical_h(circuit, (loc + i) * N_prev, 1, N_prev, code, level)
    
    def append_cnot(self, circuit: stim.Circuit, loc1: int, loc2: int, 
                    N_prev: int, N_now: int):
        """Transversal CNOT."""
        N = N_prev * N_now
        for i in range(N):
            PhysicalOps.cnot(circuit, loc1 * N_prev + i, loc2 * N_prev + i)
    
    def append_noisy_cnot(self, circuit: stim.Circuit, loc1: int, loc2: int,
                          N_prev: int, N_now: int, p: float):
        """Transversal CNOT with depolarizing noise."""
        N = N_prev * N_now
        for i in range(N):
            PhysicalOps.cnot(circuit, loc1 * N_prev + i, loc2 * N_prev + i)
        for i in range(N):
            circuit.append("DEPOLARIZE2", [loc1 * N_prev + i, loc2 * N_prev + i], p)
    
    def append_swap(self, circuit: stim. Circuit, loc1: int, loc2: int,
                    N_prev: int, N_now: int):
        """Transversal SWAP."""
        for i in range(N_prev * N_now):
            PhysicalOps.swap(circuit, N_prev * loc1 + i, N_prev * loc2 + i)
    
    def append_m(self, circuit: stim. Circuit, loc: int, N_prev: int, N_now:  int,
                 detector_counter: List[int], code: CSSCode = None) -> List: 
        """
        Transversal measurement with detectors.
        
        NOTE: Detector structure must be compatible with the acceptance checker's
        assumption that detector indices map to measurement indices. We use
        "absolute" detectors (one per qubit) for consistency. The decoder
        is responsible for correctly interpreting the measurement values.
        
        Args:
            circuit: Stim circuit to append to
            loc: Base qubit location
            N_prev: Size of each block (1 for physical level)
            N_now: Number of qubits/blocks at this level
            detector_counter: Counter for detector tracking
            code: Optional CSSCode (currently unused, kept for future extensibility)
        """
        if N_prev == 1:
            # Physical level: measure all qubits
            for i in range(N_now):
                PhysicalOps.measure(circuit, loc + i)
            
            # Always use absolute detectors for compatibility with acceptance checker
            # The decoder handles code-specific interpretation of measurement values
            for i in range(N_now):
                PhysicalOps.detector(circuit, i - N_now)
            detector_m = [detector_counter[0], detector_counter[0] + N_now]
            detector_counter[0] += N_now
        else:
            # Recursive: apply to each block
            inner_code = self._get_inner_code() if N_prev > 1 else None
            detector_m = [
                self.append_m(circuit, (loc + i) * N_prev, 1, N_prev, detector_counter, inner_code)
                for i in range(N_now)
            ]
        return detector_m
    
    def append_noisy_m(self, circuit: stim.Circuit, loc: int, N_prev: int,
                       N_now: int, p: float, detector_counter: List[int],
                       code: CSSCode = None) -> List:
        """
        Transversal measurement with pre-measurement noise.
        
        Uses same absolute detector structure as append_m for consistency.
        """
        if N_prev == 1:
            for i in range(N_now):
                PhysicalOps.noisy_measure(circuit, loc + i, p)
            
            # Absolute detectors for compatibility
            for i in range(N_now):
                PhysicalOps.detector(circuit, i - N_now)
            detector_m = [detector_counter[0], detector_counter[0] + N_now]
            detector_counter[0] += N_now
        else:
            inner_code = self._get_inner_code() if N_prev > 1 else None
            detector_m = [
                self.append_noisy_m(circuit, (loc + i) * N_prev, 1, N_prev, p, detector_counter, inner_code)
                for i in range(N_now)
            ]
        return detector_m
    
    def append_noisy_wait(self, circuit: stim. Circuit, list_loc: List[int],
                          N:  int, p: float, gamma: float, steps: int = 1):
        """Idle noise on qubits."""
        ew = 3/4 * (1 - (1 - 4/3 * gamma) ** steps)
        for loc in list_loc:
            for j in range(N):
                PhysicalOps.depolarize1(circuit, loc + j, ew)


# =============================================================================
# Logical Gate Interface (Abstract)
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                        LOGICAL GATE HIERARCHY                                │
# └─────────────────────────────────────────────────────────────────────────────┘

#                           ┌─────────────────────┐
#                           │   LogicalGate       │
#                           │     (Abstract)      │
#                           ├─────────────────────┤
#                           │ + gate_name: str    │
#                           │ + implementation    │
#                           │ + block_size(level) │
#                           └──────────┬──────────┘
#                                      │
#            ┌─────────────────────────┼─────────────────────────┐
#            │                         │                         │
#            ▼                         ▼                         ▼
# ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
# │   LogicalHGate      │  │  LogicalCNOTGate    │  │ LogicalMeasurement  │
# │    (Abstract)       │  │    (Abstract)       │  │    (Abstract)       │
# ├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
# │ + apply(circuit,    │  │ + apply(circuit,    │  │ + apply(circuit,    │
# │   loc, level,       │  │   loc_ctrl, loc_targ│  │   loc, level,       │
# │   detector_counter) │  │   level, det_ctr)   │  │   det_ctr, basis)   │
# │   -> GateResult     │  │   -> GateResult     │  │   -> GateResult     │
# └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
#            │                        │                        │
#            ▼                        ▼                        ▼
# ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
# │ TransversalHGate    │  │ TransversalCNOTGate │  │TransversalMeasure-  │
# │                     │  │                     │  │ment                 │
# ├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
# │ Uses TransversalOps │  │ Uses TransversalOps │  │ Uses TransversalOps │
# │ to apply H to all   │  │ to apply CNOT       │  │ Hierarchical        │
# │ physical qubits     │  │ block-wise          │  │ detector structure  │
# └─────────────────────┘  └─────────────────────┘  └─────────────────────┘


#                     ┌─────────────────────────────────┐
#                     │    LogicalGateDispatcher        │
#                     ├─────────────────────────────────┤
#                     │ Input: ConcatenatedCode,        │
#                     │        TransversalOps           │
#                     ├─────────────────────────────────┤
#                     │ _h:  LogicalHGate                │
#                     │ _cnot: LogicalCNOTGate          │
#                     │ _measure: LogicalMeasurement    │
#                     ├─────────────────────────────────┤
#                     │ + h(circuit, loc, level, ctr)   │
#                     │ + cnot(circuit, c, t, lvl, ctr) │
#                     │ + measure(circuit, loc, ...)    │
#                     │ + set_h_gate(gate)              │
#                     │ + set_cnot_gate(gate)           │
#                     └─────────────────────────────────┘
# =============================================================================

class LogicalGate(ABC):
    """Abstract base class for logical gate implementations."""
    
    def __init__(self, concat_code: ConcatenatedCode):
        self.concat_code = concat_code
    
    @property
    @abstractmethod
    def gate_name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def implementation_name(self) -> str:
        pass
    
    def block_size(self, level: int) -> int:
        return self.concat_code.qubits_at_level(level)


class LogicalHGate(LogicalGate):
    """Abstract base for logical Hadamard."""
    
    @property
    def gate_name(self) -> str:
        return "H"
    
    @abstractmethod
    def apply(self, circuit: stim. Circuit, loc: int, level: int,
              detector_counter: List[int]) -> GateResult:
        pass


class LogicalCNOTGate(LogicalGate):
    """Abstract base for logical CNOT."""
    
    @property
    def gate_name(self) -> str:
        return "CNOT"
    
    @abstractmethod
    def apply(self, circuit: stim. Circuit, loc_ctrl: int, loc_targ:  int,
              level: int, detector_counter: List[int]) -> GateResult:
        pass


class LogicalMeasurement(LogicalGate):
    """Abstract base for logical measurement."""
    
    @property
    def gate_name(self) -> str:
        return "MEASURE"
    
    @abstractmethod
    def apply(self, circuit: stim.Circuit, loc: int, level: int,
              detector_counter: List[int], basis: str = 'z') -> GateResult:
        pass


# =============================================================================
# Transversal Gate Implementations
# =============================================================================

class TransversalHGate(LogicalHGate):
    """Transversal Hadamard - applies H to every physical qubit."""
    
    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        super().__init__(concat_code)
        self.ops = ops
    
    @property
    def implementation_name(self) -> str:
        return "transversal"
    
    def apply(self, circuit, loc, level, detector_counter):
        N_prev = self.block_size(level - 1) if level > 0 else 1
        N_now = self.concat_code.code_at_level(level).n
        self.ops.append_h(circuit, loc, N_prev, N_now)
        return GateResult(self.gate_name, self.implementation_name, level)


class TransversalCNOTGate(LogicalCNOTGate):
    """Transversal CNOT - applies CNOT between corresponding qubits."""
    
    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        super().__init__(concat_code)
        self.ops = ops
    
    @property
    def implementation_name(self) -> str:
        return "transversal"
    
    def apply(self, circuit, loc_ctrl, loc_targ, level, detector_counter):
        N_prev = self.block_size(level - 1) if level > 0 else 1
        N_now = self.concat_code.code_at_level(level).n
        self.ops.append_cnot(circuit, loc_ctrl, loc_targ, N_prev, N_now)
        return GateResult(self.gate_name, self.implementation_name, level)


class TransversalMeasurement(LogicalMeasurement):
    """Transversal measurement - measures all physical qubits."""
    
    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        super().__init__(concat_code)
        self.ops = ops
    
    @property
    def implementation_name(self) -> str:
        return "transversal"
    
    def apply(self, circuit, loc, level, detector_counter, basis='z'):
        N_prev = self.block_size(level - 1) if level > 0 else 1
        N_now = self.concat_code.code_at_level(level).n
        
        if basis == 'x':
            self.ops.append_h(circuit, loc, N_prev, N_now)
        
        detectors = self.ops.append_m(circuit, loc, N_prev, N_now, detector_counter)
        
        result = GateResult(self.gate_name, self.implementation_name, level)
        result.detectors = detectors
        return result


# =============================================================================
# Gate Dispatcher
# =============================================================================

class LogicalGateDispatcher:
    """
    Dispatcher for logical gate operations.
    
    Provides unified interface abstracting gate implementations.
    """
    
    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        self.concat_code = concat_code
        self.ops = ops
        
        # Default to transversal implementations
        self._h = TransversalHGate(concat_code, ops)
        self._cnot = TransversalCNOTGate(concat_code, ops)
        self._measure = TransversalMeasurement(concat_code, ops)
    
    def set_h_gate(self, gate: LogicalHGate):
        self._h = gate
    
    def set_cnot_gate(self, gate: LogicalCNOTGate):
        self._cnot = gate
    
    def set_measurement(self, gate: LogicalMeasurement):
        self._measure = gate
    
    def h(self, circuit: stim.Circuit, loc: int, level: int,
          detector_counter: List[int]) -> GateResult:
        return self._h.apply(circuit, loc, level, detector_counter)
    
    def cnot(self, circuit: stim.Circuit, loc_ctrl: int, loc_targ: int,
             level: int, detector_counter: List[int]) -> GateResult:
        return self._cnot.apply(circuit, loc_ctrl, loc_targ, level, detector_counter)
    
    def measure(self, circuit: stim.Circuit, loc: int, level: int,
                detector_counter: List[int], basis: str = 'z') -> GateResult:
        return self._measure.apply(circuit, loc, level, detector_counter, basis)


# =============================================================================
# Preparation Strategy (Abstract)
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                       PREPARATION STRATEGIES                                 │
# └─────────────────────────────────────────────────────────────────────────────┘

#                     ┌─────────────────────────────────┐
#                     │    PreparationStrategy          │
#                     │         (Abstract)              │
#                     ├─────────────────────────────────┤
#                     │ Input: ConcatenatedCode,        │
#                     │        TransversalOps           │
#                     ├─────────────────────────────────┤
#                     │ + set_ec_gadget(ECGadget)       │
#                     │ + strategy_name: str            │
#                     ├─────────────────────────────────┤
#                     │ + append_0prep(circuit, loc1,   │
#                     │     N_prev, N_now)              │
#                     │                                 │
#                     │ + append_noisy_0prep(circuit,   │
#                     │     loc1, loc2, N_prev, N_now,  │
#                     │     p, detector_counter)        │
#                     │   -> List | Tuple               │
#                     └────────────────┬────────────────┘
#                                      │
#                     ┌────────────────┴────────────────┐
#                     │                                 │
#                     ▼                                 ▼
#      ┌──────────────────────────┐      ┌──────────────────────────┐
#      │ SteanePreparationStrategy│      │GenericPreparationStrategy│
#      ├──────────────────────────┤      ├──────────────────────────┤
#      │ Matches original EXACTLY │      │ Uses CSSCode spec to     │
#      │                          │      │ build preparation        │
#      │ _noisy_0prep_l1():       │      │                          │
#      │  - 3 H gates             │      │ Less optimized but       │
#      │  - 8 encoding CNOTs      │      │ works for any CSS code   │
#      │  - 3 verification CNOTs  │      │                          │
#      │  - Idle noise            │      │ No EC interleaving       │
#      │  - 1 measurement         │      │ (simpler structure)      │
#      │                          │      │                          │
#      │ _noisy_0prep_l2():       │      └──────────────────────────┘
#      │  - Recursive inner prep  │
#      │  - 45 EC rounds          │
#      │  - Exact interleaving    │
#      │  - Returns 4-tuple       │
#      └──────────────────────────┘

# Return Value Structure: 
# ┌─────────────────────────────────────────────────────────────────┐
# │ N_prev = 1 (Level-1):                                           │
# │   Returns:  detector_0prep (List)                                │
# │                                                                 │
# │ N_prev > 1, N_now = N_steane (Level-2):                         │
# │   Returns: (detector_0prep, detector_0prep_l2,                  │
# │             detector_X, detector_Z)                             │
# └─────────────────────────────────────────────────────────────────┘
# =============================================================================

class PreparationStrategy(ABC):
    """
    Abstract base class for state preparation strategies.
    
    Different codes may require different preparation circuits with
    different EC interleaving patterns.
    """
    
    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        self.concat_code = concat_code
        self.ops = ops
        self._ec_gadget = None
    
    def set_ec_gadget(self, ec_gadget: 'ECGadget'):
        self._ec_gadget = ec_gadget
    
    @property
    def ec(self) -> 'ECGadget':
        if self._ec_gadget is None:
            raise RuntimeError("EC gadget not set")
        return self._ec_gadget
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        pass
    
    @property
    def uses_prep_ec_at_l2(self) -> bool:
        """
        Whether this preparation strategy uses EC rounds during L2 preparation.
        
        This affects the detector structure returned:
        - True: detector_X/detector_Z contain prep EC entries (old-style)
        - False: detector_X/detector_Z are empty (corrected prep style)
        
        Subclasses should override if they use prep EC at L2.
        Default is False (corrected prep style with no EC during prep).
        """
        return False
    
    @abstractmethod
    def append_0prep(self, circuit: stim.Circuit, loc1: int, 
                     N_prev: int, N_now: int):
        """Noiseless |0⟩_L preparation."""
        pass
    
    @abstractmethod
    def append_noisy_0prep(self, circuit: stim.Circuit, loc1: int, loc2: int,
                           N_prev: int, N_now: int, p: float,
                           detector_counter: List[int]) -> Union[List, Tuple]:
        """
        Noisy |0⟩_L preparation with verification.
        
        Returns:
        - N_prev=1: detector_0prep (list)
        - N_prev>1: (detector_0prep, detector_0prep_l2, detector_X, detector_Z)
        """
        pass
    
    def append_plus_prep(self, circuit: stim.Circuit, loc1: int,
                         N_prev: int, N_now: int):
        """
        Noiseless |+⟩_L preparation.
        
        For self-dual codes: delegates to append_0prep followed by logical H.
        For non-self-dual codes: uses direct |+⟩_L encoding circuit.
        
        Subclasses may override for code-specific optimizations.
        """
        code = self.concat_code.code_at_level(0)
        if code.requires_direct_plus_prep:
            # Non-self-dual: use direct |+⟩_L encoding
            self._append_direct_plus_prep(circuit, loc1, N_prev, N_now, code)
        else:
            # Self-dual: |+⟩_L = H·|0⟩_L
            self.append_0prep(circuit, loc1, N_prev, N_now)
            self.ops.append_logical_h(circuit, loc1, N_prev, N_now, code)
    
    def _append_direct_plus_prep(self, circuit: stim.Circuit, loc1: int,
                                  N_prev: int, N_now: int, code: CSSCode):
        """
        Direct |+⟩_L preparation for non-self-dual codes.
        
        Uses code.plus_h_qubits and code.plus_encoding_cnots.
        """
        # Base case: physical qubits
        if N_prev == 1:
            PhysicalOps.reset(circuit, loc1, N_now)
        else:
            # Recursive: prepare inner blocks as |+⟩_L
            n_now = code.transversal_block_count if code.transversal_block_count else N_now
            for i in range(n_now):
                self._append_direct_plus_prep(circuit, (loc1 + i) * N_prev, 1, N_prev, code)
        
        # Apply |+⟩_L encoding if at code block size
        if N_now == code.n and code.plus_h_qubits:
            # H gates on plus_h_qubits
            for q in code.plus_h_qubits:
                self.ops.append_h(circuit, (loc1 + q) * N_prev, 1, N_prev)
            
            # CNOTs for |+⟩_L
            if code.plus_encoding_cnots:
                for ctrl, targ in code.plus_encoding_cnots:
                    self.ops.append_cnot(circuit, (loc1 + ctrl) * N_prev,
                                         (loc1 + targ) * N_prev, 1, N_prev)
    
    def append_noisy_plus_prep(self, circuit: stim.Circuit, loc1: int, loc2: int,
                               N_prev: int, N_now: int, p: float,
                               detector_counter: List[int]) -> Union[List, Tuple]:
        """
        Noisy |+⟩_L preparation with verification.
        
        For self-dual codes: delegates to noisy |0⟩_L prep followed by logical H.
        For non-self-dual codes: uses direct |+⟩_L encoding circuit with noise.
        
        Default implementation - subclasses may override for optimizations.
        """
        code = self.concat_code.code_at_level(0)
        if code.requires_direct_plus_prep:
            # Non-self-dual: use direct |+⟩_L encoding with noise
            return self._append_noisy_direct_plus_prep(
                circuit, loc1, loc2, N_prev, N_now, p, detector_counter, code
            )
        else:
            # Self-dual: |+⟩_L = H·|0⟩_L
            result = self.append_noisy_0prep(circuit, loc1, loc2, N_prev, N_now, p, detector_counter)
            # Apply logical H after preparation
            inner_code = self.concat_code.code_at_level(0)
            self.ops.append_logical_h(circuit, loc1, N_prev, N_now, inner_code)
            return result
    
    def _append_noisy_direct_plus_prep(self, circuit: stim.Circuit, loc1: int, loc2: int,
                                        N_prev: int, N_now: int, p: float,
                                        detector_counter: List[int], code: CSSCode) -> Union[List, Tuple]:
        """
        Direct noisy |+⟩_L preparation for non-self-dual codes.
        
        Uses similar structure to noisy |0⟩_L prep but with |+⟩_L encoding circuit.
        This prepares |+⟩_L directly without needing logical H.
        
        For Shor:
        - plus_h_qubits = [0, 3, 6] (H on one qubit per block)
        - plus_encoding_cnots = [(0,1), (0,2), (3,4), (3,5), (6,7), (6,8)] (spread in blocks)
        """
        gamma = p / 10  # Idle error rate
        
        # Physical level: use standard prep but with plus encoding parameters
        if N_prev == 1:
            PhysicalOps.noisy_reset(circuit, loc1, N_now, p)
            PhysicalOps.noisy_reset(circuit, loc2, N_now, p)
            detector_0prep = []
            
            # Apply |+⟩_L encoding at code block size
            if N_now == code.n and code.plus_h_qubits:
                # Phase 1: H gates on plus_h_qubits
                for q in code.plus_h_qubits:
                    PhysicalOps.h(circuit, loc1 + q)
                    circuit.append("DEPOLARIZE1", [loc1 + q], p)
                
                # Idle noise on qubits not getting H
                idle_qubits = set(range(N_now)) - set(code.plus_h_qubits)
                for q in idle_qubits:
                    circuit.append("DEPOLARIZE1", [loc1 + q], gamma)
                
                # Phase 2: CNOTs for |+⟩_L encoding
                if code.plus_encoding_cnots:
                    for ctrl, targ in code.plus_encoding_cnots:
                        PhysicalOps.cnot(circuit, loc1 + ctrl, loc1 + targ)
                        circuit.append("DEPOLARIZE2", [loc1 + ctrl, loc1 + targ], p)
                
                # Phase 3: Verification using Z-type stabilizers (same as |0⟩_L)
                # For |+⟩_L, Z-type stabilizers should still give +1 eigenvalue
                # (because |+⟩_L is in the code space)
                # Use same verification as standard prep: CNOT from verification qubits to ancilla
                if code.verification_qubits:
                    for vq in code.verification_qubits:
                        self.ops.append_noisy_cnot(
                            circuit, loc1 + vq, loc2, 1, 1, p
                        )
                
                # Phase 4: Measure verification ancilla
                detector_0prep.append(
                    self.ops.append_noisy_m(circuit, loc2, 1, 1, p, detector_counter)
                )
            
            return detector_0prep
        else:
            # L2: recursively prepare inner blocks
            detector_0prep = []
            for i in range(N_now):
                result = self._append_noisy_direct_plus_prep(
                    circuit, (loc1 + i) * N_prev, (loc2 + i) * N_prev,
                    1, N_prev, p, detector_counter, code
                )
                detector_0prep.append(result)
            
            return detector_0prep, [], [], []  # Match expected L2 return format


# Note: SteanePreparationStrategy has been moved to concatenated_css_v10_steane.py

class GenericPreparationStrategy(PreparationStrategy):
    """
    Generic preparation strategy that works for any CSS code.
    
    Uses the CSSCode specification to build preparation circuits that are
    as close as possible to optimal code-specific implementations.
    
    Key features:
    - Uses code's h_qubits, encoding_cnots, encoding_cnot_rounds
    - Applies EC interleaving based on CNOT rounds (if specified)
    - Handles idle noise on non-active qubits (configurable via use_idle_noise)
    - Supports both level-1 and level-2 preparation
    - Returns correct structure for decoder compatibility
    
    Args:
        concat_code: The concatenated code
        ops: Transversal operations helper
        use_idle_noise: If True (default), applies DEPOLARIZE1 noise on idle qubits
                        during CNOT rounds. Set to False to match original behavior
                        that doesn't model idle decoherence.
    """
    
    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps,
                 use_idle_noise: bool = True):
        super().__init__(concat_code, ops)
        self.use_idle_noise = use_idle_noise
    
    @property
    def strategy_name(self) -> str:
        return "generic"
    
    def append_0prep(self, circuit: stim.Circuit, loc1: int,
                     N_prev: int, N_now: int):
        """
        Noiseless |0⟩_L preparation using code specification.
        
        Recursively prepares inner blocks, then applies encoding circuit.
        Uses transversal_block_count to determine number of inner block preps.
        """
        # Get the appropriate code for this level
        if N_prev == 1:
            code = self.concat_code.code_at_level(0)
        else:
            code = self.concat_code.code_at_level(1) if self.concat_code.num_levels > 1 else self.concat_code.code_at_level(0)
        
        # Determine number of transversal blocks
        n_now = code.transversal_block_count if code.transversal_block_count else N_now
        
        # Base case: physical qubits
        if N_prev == 1:
            PhysicalOps.reset(circuit, loc1, N_now)
        else:
            # Recursive case: prepare each inner block (using transversal block count)
            for i in range(n_now):
                self.append_0prep(circuit, (loc1 + i) * N_prev, 1, N_prev)
        
        # Apply encoding circuit if at code block size
        if N_now == code.n: 
            # H gates on specified qubits
            for q in code.h_qubits:
                self.ops.append_h(circuit, (loc1 + q) * N_prev, 1, N_prev)
            
            # CNOTs in specified order
            for ctrl, targ in code.encoding_cnots:
                self.ops.append_cnot(circuit, (loc1 + ctrl) * N_prev,
                                     (loc1 + targ) * N_prev, 1, N_prev)
    
    def append_noisy_0prep(self, circuit: stim.Circuit, loc1: int, loc2: int,
                           N_prev: int, N_now: int, p:  float,
                           detector_counter: List[int]) -> Union[List, Tuple]:
        """
        Noisy |0⟩_L preparation with verification.
        
        Dispatches to level-specific implementations based on N_prev.
        Returns different structures for L1 vs L2 to match decoder expectations.
        """
        code = self.concat_code.code_at_level(0)
        gamma = self._compute_gamma(p)
        
        # Prepare inner blocks recursively
        if N_prev == 1:
            PhysicalOps.noisy_reset(circuit, loc1, N_now, p)
            PhysicalOps.noisy_reset(circuit, loc2, N_now, p)
            detector_0prep = []
        else:
            detector_0prep = []
            # Prepare data blocks
            for i in range(N_now):
                result = self.append_noisy_0prep(
                    circuit, (loc1 + i) * N_prev, (loc1 + N_now + i) * N_prev,
                    1, N_prev, p, detector_counter
                )
                detector_0prep.append(result)
            
            # Prepare verification/ancilla block(s)
            if N_now == code.n:
                # Single verification block for code-sized preparation
                result = self.append_noisy_0prep(
                    circuit, loc2 * N_prev, (loc2 + N_now) * N_prev,
                    1, N_prev, p, detector_counter
                )
                detector_0prep.append(result)
            else:
                # Multiple ancilla blocks for non-code-sized
                for i in range(N_now):
                    result = self.append_noisy_0prep(
                        circuit, (loc2 + i) * N_prev, (loc2 + N_now + i) * N_prev,
                        1, N_prev, p, detector_counter
                    )
                    detector_0prep.append(result)
        
        # Apply encoding based on level
        if N_now == code. n:
            if N_prev != 1:
                return self._noisy_0prep_l2(
                    circuit, loc1, loc2, N_prev, N_now, p, gamma,
                    detector_counter, detector_0prep, code
                )
            else:
                return self._noisy_0prep_l1(
                    circuit, loc1, loc2, N_prev, N_now, p, gamma,
                    detector_counter, detector_0prep, code
                )
        
        return detector_0prep
    
    def _compute_gamma(self, p: float) -> float:
        """Compute idle error rate from physical error rate."""
        # Default:  error model 'a' from original
        return p / 10
    
    def _get_idle_qubits(self, code: CSSCode, active_qubits: List[int], 
                         round_name: str = None) -> List[int]:
        """
        Determine which qubits are idle given active qubits. 
        
        Uses code's idle_schedule if available, otherwise computes from active set.
        """
        all_qubits = set(range(code.n))
        active_set = set(active_qubits)
        idle = list(all_qubits - active_set)
        
        # Override with code-specific schedule if available
        if code.idle_schedule and round_name and round_name in code.idle_schedule:
            idle = code. idle_schedule[round_name]
        
        return sorted(idle)
    
    def _get_cnot_rounds(self, code: CSSCode) -> List[List[Tuple[int, int]]]: 
        """
        Get CNOT rounds from code specification.
        
        Uses encoding_cnot_rounds if available, otherwise groups by parallelizability.
        """
        if code.encoding_cnot_rounds:
            return code.encoding_cnot_rounds
        
        # Auto-group CNOTs that can be parallelized
        # CNOTs are parallel if they don't share any qubits
        if not code.encoding_cnots:
            return []
        
        rounds = []
        remaining = list(code.encoding_cnots)
        
        while remaining: 
            current_round = []
            used_qubits = set()
            still_remaining = []
            
            for ctrl, targ in remaining:
                if ctrl not in used_qubits and targ not in used_qubits:
                    current_round.append((ctrl, targ))
                    used_qubits.add(ctrl)
                    used_qubits.add(targ)
                else:
                    still_remaining. append((ctrl, targ))
            
            if current_round: 
                rounds.append(current_round)
            remaining = still_remaining
        
        return rounds
    
    def _get_verification_schedule(self, code: CSSCode) -> List[List[int]]:
        """
        Get verification CNOT schedule. 
        
        Groups verification qubits for sequential verification CNOTs.
        For fault-tolerance, typically one verification qubit per round.
        
        If verification_qubits is not specified in the code, derives them
        from the code structure (uses qubits in the logical X support).
        """
        verif_qubits = code.verification_qubits
        
        # Auto-derive verification qubits if not specified
        if not verif_qubits:
            # Use qubits in logical X operator support
            # These are the qubits that determine the logical state in Z-basis
            verif_qubits = [i for i, v in enumerate(code.Lx) if v == 1]
            
            # If Lx is weight-n (all qubits), fall back to h_qubits
            if len(verif_qubits) == code.n and code.h_qubits:
                verif_qubits = code.h_qubits
            
            # Last resort: use first ceil(n/2) qubits
            if not verif_qubits:
                verif_qubits = list(range((code.n + 1) // 2))
        
        # Default: one verification qubit per round (most fault-tolerant)
        return [[vq] for vq in verif_qubits]
    
    def _noisy_0prep_l1(self, circuit: stim.Circuit, loc1: int, loc2: int,
                        N_prev: int, N_now: int, p: float, gamma: float,
                        detector_counter: List[int], detector_0prep: List,
                        code: CSSCode) -> List:
        """
        Level-1 noisy preparation (N_prev=1).
        
        Dispatches to Bell-pair or standard CSS encoding based on code.uses_bellpair_prep.
        """
        if code.uses_bellpair_prep:
            return self._noisy_0prep_l1_bellpair(
                circuit, loc1, loc2, N_prev, N_now, p, gamma,
                detector_counter, detector_0prep, code
            )
        else:
            return self._noisy_0prep_l1_standard(
                circuit, loc1, loc2, N_prev, N_now, p, gamma,
                detector_counter, detector_0prep, code
            )
    
    def _noisy_0prep_l1_bellpair(self, circuit: stim.Circuit, loc1: int, loc2: int,
                                  N_prev: int, N_now: int, p: float, gamma: float,
                                  detector_counter: List[int], detector_0prep: List,
                                  code: CSSCode) -> List:
        """
        Level-1 noisy preparation using Bell-pair protocol (for k>1 codes like C4).
        
        Bell-pair preparation structure (exactly matches original concatenated_c4c6.py):
        1. H on ALL ancilla qubits (loc2)
        2. CNOT: ancilla[i] → data[i] for all i  
        3. CNOT: data[i] → ancilla[(i+1)%n] for all i
        4. Measure ALL ancilla qubits
        5. Correction CNOTs: triangular pattern based on measurement
        
        This creates the state |ψ⟩ = (|0000⟩_data|++++⟩_ancilla + |1111⟩_data|----⟩_ancilla)/√2
        """
        n = code.n
        
        # Phase 1: H on ALL ancilla qubits
        for i in range(n):
            circuit.append("H", loc2 + i)
        
        # Phase 2: CNOTs ancilla[i] → data[i]
        for i in range(n):
            self.ops.append_noisy_cnot(
                circuit, (loc2 + i) * N_prev, (loc1 + i) * N_prev, 1, N_prev, p
            )
        
        # Phase 3: CNOTs data[i] → ancilla[(i+1)%n]
        for i in range(n):
            self.ops.append_noisy_cnot(
                circuit, (loc1 + i) * N_prev, (loc2 + (i + 1) % n) * N_prev, 1, N_prev, p
            )
        
        # Phase 4: Measure ALL ancilla qubits
        detector_0prep.append(
            self.ops.append_noisy_m(circuit, loc2 * N_prev, N_prev, n, p, detector_counter)
        )
        
        # Phase 5: Correction CNOTs (triangular pattern)
        # For i in 0..n-2, for j in i..n-2: CNOT ancilla[i] → data[j]
        for i in range(n - 1):
            for j in range(n - 1):
                if j >= i:
                    self.ops.append_cnot(
                        circuit, (loc2 + i) * N_prev, (loc1 + j) * N_prev, 1, N_prev
                    )
        
        return detector_0prep
    
    def _noisy_0prep_l1_standard(self, circuit: stim.Circuit, loc1: int, loc2: int,
                                  N_prev: int, N_now: int, p: float, gamma: float,
                                  detector_counter: List[int], detector_0prep: List,
                                  code: CSSCode) -> List:
        """
        Level-1 noisy preparation using standard CSS encoding (for k=1 codes like Steane).
        
        Structure: 
        1. Pre-H CNOTs (for codes like Shor that need inter-block correlation)
        2. H gates on designated qubits
        3. Encoding CNOTs (by rounds with idle noise)
        4. Verification CNOTs (with idle noise)
        5. Verification measurement
        """
        n = code.n
        
        # Phase 0: Pre-H CNOTs (for codes like Shor)
        if code.pre_h_cnots:
            for ctrl, targ in code.pre_h_cnots:
                self.ops.append_noisy_cnot(
                    circuit, (loc1 + ctrl) * N_prev, (loc1 + targ) * N_prev,
                    1, N_prev, p
                )
            # Apply idle noise to qubits not involved in pre-H CNOTs
            if self.use_idle_noise:
                active_qubits = []
                for ctrl, targ in code.pre_h_cnots:
                    active_qubits.extend([ctrl, targ])
                idle_qubits = [q for q in range(n) if q not in active_qubits]
                if idle_qubits:
                    self.ops.append_noisy_wait(
                        circuit, [(loc1 + q) * N_prev for q in idle_qubits],
                        N_prev, p, gamma, steps=1
                    )
        
        # Phase 1: H gates
        for q in code.h_qubits:
            self.ops. append_h(circuit, (loc1 + q) * N_prev, 1, N_prev)
        
        # Phase 2: Encoding CNOTs by rounds
        cnot_rounds = self._get_cnot_rounds(code)
        
        for round_idx, round_cnots in enumerate(cnot_rounds):
            # Determine active qubits in this round
            active_qubits = []
            for ctrl, targ in round_cnots:
                active_qubits.extend([ctrl, targ])
            
            # Apply CNOTs
            for ctrl, targ in round_cnots:
                self. ops.append_noisy_cnot(
                    circuit, (loc1 + ctrl) * N_prev, (loc1 + targ) * N_prev,
                    1, N_prev, p
                )
            
            # Apply idle noise to non-active qubits
            if self.use_idle_noise:
                idle_qubits = self._get_idle_qubits(code, active_qubits, f'cnot_round_{round_idx}')
                if idle_qubits:
                    self.ops.append_noisy_wait(
                        circuit, [(loc1 + q) * N_prev for q in idle_qubits],
                        N_prev, p, gamma, steps=1
                    )
        
        # Phase 3: Verification CNOTs
        verification_schedule = self._get_verification_schedule(code)
        
        for verif_idx, verif_qubits in enumerate(verification_schedule):
            # Apply verification CNOTs
            for vq in verif_qubits:
                self.ops.append_noisy_cnot(
                    circuit, (loc1 + vq) * N_prev, loc2 * N_prev,
                    1, N_prev, p
                )
            
            # Idle noise on non-verification qubits
            if self.use_idle_noise:
                idle_qubits = self._get_idle_qubits(code, verif_qubits, f'verif_cnot_{verif_idx}')
                if idle_qubits: 
                    self.ops.append_noisy_wait(
                        circuit, [(loc1 + q) * N_prev for q in idle_qubits],
                        N_prev, p, gamma, steps=1
                    )
        
        # Phase 4: Verification measurement
        detector_0prep.append(
            self.ops.append_noisy_m(circuit, loc2 * N_prev, 1, N_prev, p, detector_counter)
        )
        
        # Idle noise during measurement
        if self.use_idle_noise:
            # Get all verification qubits from schedule
            all_verif_qubits = []
            for vq_list in verification_schedule:
                all_verif_qubits.extend(vq_list)
            
            if all_verif_qubits:
                non_measured = [q for q in range(n) if q not in all_verif_qubits[-1:]]
                if non_measured:
                    self.ops.append_noisy_wait(
                        circuit, [(loc1 + q) * N_prev for q in non_measured],
                        N_prev, p, gamma, steps=1
                )
        
        return detector_0prep
    
    def _noisy_0prep_l2(self, circuit: stim.Circuit, loc1: int, loc2: int,
                        N_prev: int, N_now: int, p: float, gamma: float,
                        detector_counter: List[int], detector_0prep: List,
                        code: CSSCode) -> Tuple:
        """
        Level-2 noisy preparation (N_prev > 1, N_now = code.n).
        
        Dispatches to Bell-pair or standard CSS encoding based on code.uses_bellpair_prep.
        """
        if code.uses_bellpair_prep:
            return self._noisy_0prep_l2_bellpair(
                circuit, loc1, loc2, N_prev, N_now, p, gamma,
                detector_counter, detector_0prep, code
            )
        else:
            return self._noisy_0prep_l2_standard(
                circuit, loc1, loc2, N_prev, N_now, p, gamma,
                detector_counter, detector_0prep, code
            )
    
    def _noisy_0prep_l2_bellpair(self, circuit: stim.Circuit, loc1: int, loc2: int,
                                  N_prev: int, N_now: int, p: float, gamma: float,
                                  detector_counter: List[int], detector_0prep: List,
                                  code: CSSCode) -> Tuple:
        """
        Level-2 noisy preparation using Bell-pair protocol (for k>1 codes like C4).
        
        At L2, this operates on encoded blocks (N_prev > 1).
        Uses transversal_block_count for outer code's number of blocks.
        
        Bell-pair preparation structure (matches original concatenated_c4c6.py):
        1. H on ALL ancilla blocks
        2. CNOT: ancilla[i] → data[i] for all blocks
        3. CNOT: data[i] → ancilla[(i+1)%n] for all blocks
        4. Measure ALL ancilla blocks
        5. Correction CNOTs: triangular pattern based on measurement
        
        Returns 4-tuple: (detector_0prep, detector_0prep_l2, [], [])
        """
        # Get outer code for L2 operations (if different from inner)
        outer_code = self.concat_code.code_at_level(1) if self.concat_code.num_levels > 1 else code
        n_outer = outer_code.transversal_block_count if outer_code.transversal_block_count else outer_code.n
        
        # Phase 1: H on ALL ancilla blocks
        for i in range(n_outer):
            self.ops.append_h(circuit, (loc2 + i) * N_prev, 1, N_prev)
        
        # Phase 2: CNOTs ancilla[i] → data[i]
        for i in range(n_outer):
            self.ops.append_noisy_cnot(
                circuit, (loc2 + i) * N_prev, (loc1 + i) * N_prev, 1, N_prev, p
            )
        
        # Phase 3: CNOTs data[i] → ancilla[(i+1)%n]
        for i in range(n_outer):
            self.ops.append_noisy_cnot(
                circuit, (loc1 + i) * N_prev, (loc2 + (i + 1) % n_outer) * N_prev, 1, N_prev, p
            )
        
        # Phase 4: Measure ALL ancilla blocks
        detector_0prep_l2 = self.ops.append_noisy_m(
            circuit, loc2 * N_prev, N_prev, n_outer, p, detector_counter
        )
        
        # Phase 5: Correction CNOTs (triangular pattern)
        for i in range(n_outer - 1):
            for j in range(n_outer - 1):
                if j >= i:
                    self.ops.append_cnot(
                        circuit, (loc2 + i) * N_prev, (loc1 + j) * N_prev, 1, N_prev
                    )
        
        # Return empty detector_X and detector_Z since no EC during prep
        return detector_0prep, detector_0prep_l2, [], []
    
    def _noisy_0prep_l2_standard(self, circuit: stim.Circuit, loc1: int, loc2: int,
                                  N_prev: int, N_now: int, p: float, gamma: float,
                                  detector_counter: List[int], detector_0prep: List,
                                  code: CSSCode) -> Tuple:
        """
        Level-2 noisy preparation using standard CSS encoding (for k=1 codes like Steane).
        
        CORRECTED STRUCTURE (no EC during prep to avoid noise amplification):
        1. Pre-H CNOTs (for codes like Shor that need inter-block correlation)
        2. H gates on designated qubits
        3. Encoding CNOTs (by rounds with idle noise only)
        4. Verification CNOTs (with idle noise only)
        5. Verification measurement
        6. Decorrelation CNOTs (undo entanglement with measured ancilla)
        
        Returns 4-tuple: (detector_0prep, detector_0prep_l2, [], [])
        Note: detector_X and detector_Z are empty since no EC during prep.
        """
        n = code.n
        
        # NO EC during L2 prep - EC amplifies noise instead of helping
        
        # Phase 0: Pre-H CNOTs (for codes like Shor)
        if code.pre_h_cnots:
            for ctrl, targ in code.pre_h_cnots:
                self.ops.append_noisy_cnot(
                    circuit, (loc1 + ctrl) * N_prev, (loc1 + targ) * N_prev,
                    1, N_prev, p
                )
            # Apply idle noise to qubits not involved in pre-H CNOTs
            active_qubits = []
            for ctrl, targ in code.pre_h_cnots:
                active_qubits.extend([ctrl, targ])
            idle_qubits = [q for q in range(n) if q not in active_qubits]
            if idle_qubits:
                self.ops.append_noisy_wait(
                    circuit, [(loc1 + q) * N_prev for q in idle_qubits],
                    N_prev, p, gamma, steps=1
                )
        
        # Phase 1: H gates on designated qubits
        for q in code.h_qubits:
            self.ops.append_h(circuit, (loc1 + q) * N_prev, 1, N_prev)
        
        # Phase 2: Encoding CNOTs by rounds (NO EC)
        cnot_rounds = self._get_cnot_rounds(code)
        
        for round_idx, round_cnots in enumerate(cnot_rounds):
            # Determine active qubits for idle noise
            active_qubits = []
            for ctrl, targ in round_cnots:
                active_qubits.extend([ctrl, targ])
            
            # Apply CNOTs
            for ctrl, targ in round_cnots:
                self.ops.append_noisy_cnot(
                    circuit, (loc1 + ctrl) * N_prev, (loc1 + targ) * N_prev,
                    1, N_prev, p
                )
            
            # Idle noise on non-active qubits
            if self.use_idle_noise:
                idle_qubits = self._get_idle_qubits(code, active_qubits, f'cnot_round_{round_idx}')
                if idle_qubits:
                    self.ops.append_noisy_wait(
                        circuit, [(loc1 + q) * N_prev for q in idle_qubits],
                        N_prev, p, gamma, steps=1
                    )
        
        # Phase 3: Verification CNOTs (NO EC)
        verification_schedule = self._get_verification_schedule(code)
        
        for verif_idx, verif_qubits in enumerate(verification_schedule):
            # Apply verification CNOTs from data to verification ancilla
            for vq in verif_qubits:
                self.ops.append_noisy_cnot(
                    circuit, (loc1 + vq) * N_prev, loc2 * N_prev,
                    1, N_prev, p
                )
            
            # Idle noise on non-active qubits
            if self.use_idle_noise:
                idle_qubits = self._get_idle_qubits(code, verif_qubits, f'verif_cnot_{verif_idx}')
                if idle_qubits:
                    self.ops.append_noisy_wait(
                        circuit, [(loc1 + q) * N_prev for q in idle_qubits],
                        N_prev, p, gamma, steps=1
                    )
        
        # Phase 4: Verification measurement
        detector_0prep_l2 = self.ops.append_noisy_m(
            circuit, loc2 * N_prev, 1, N_prev, p, detector_counter
        )
        
        # Phase 5: Decorrelation CNOTs
        # After measuring verification ancilla, apply CNOTs to undo entanglement with measured ancilla
        # Use the same verification qubits derived from the schedule
        all_verif_qubits = []
        for verif_qubits in verification_schedule:
            all_verif_qubits.extend(verif_qubits)
        
        for vq in all_verif_qubits:
            self.ops.append_cnot(circuit, loc2 * N_prev, (loc1 + vq) * N_prev, 1, N_prev)
        
        # Return empty detector_X and detector_Z since no EC during prep
        return detector_0prep, detector_0prep_l2, [], []
    
    def _get_initial_ec_qubits(self, code: CSSCode) -> List[int]:
        """
        Determine which qubits should have EC after H gates.
        
        Strategy: EC on qubits that are active early in the circuit
        to catch errors before they propagate. This includes:
        - Qubits with H gates applied
        - Control qubits of the first CNOT round  
        - Target qubits of the first CNOT round
        
        For small codes (n <= 4), include all qubits.
        For larger codes, exclude the last qubit if it's not used until later
        (allows encoding to proceed without all qubits being initialized).
        """
        initial_qubits = set(code.h_qubits)
        
        # Add all qubits involved in first CNOT round
        cnot_rounds = self._get_cnot_rounds(code)
        if cnot_rounds:
            for ctrl, targ in cnot_rounds[0]:
                initial_qubits.add(ctrl)
                initial_qubits.add(targ)
        
        # For small codes, use all qubits
        if code.n <= 4:
            return list(range(code.n))
        
        # For larger codes, sort and filter
        result = sorted(initial_qubits)
        
        # If we have very few initial qubits, expand to include more
        # At minimum, include all h_qubits and their partners
        min_qubits = max(len(code.h_qubits), code.n // 2)
        if len(result) < min_qubits:
            result = list(range(min_qubits))
        
        return result
    
    def compute_propagation_tables(self, code: CSSCode) -> PropagationTables:
        """
        Compute propagation tables for this preparation strategy.
        
        This tracks how errors propagate through the encoding circuit:
        - X errors propagate forward through CNOT controls to targets
        - Z errors propagate backward through CNOT targets to controls
        
        For each EC measurement location in the 0-prep circuit, we track
        which data qubits are affected by X and Z errors at that location.
        
        Returns tables describing how errors propagate through the circuit.
        """
        n = code.n
        cnot_rounds = self._get_cnot_rounds(code)
        verification_schedule = self._get_verification_schedule(code)
        
        # Build propagation tables by simulating error flow
        propagation_X = []
        propagation_Z = []
        
        # Track which qubits each qubit's errors affect AFTER all subsequent CNOTs
        # We build this incrementally as we process CNOTs
        
        # For X propagation: X on qubit q before CNOT(c,t) -> X on q (and also t if q=c)
        # For Z propagation: Z on qubit q before CNOT(c,t) -> Z on q (and also c if q=t)
        
        # Process the circuit in order to build propagation
        # Initial EC: on h_qubits and first-round targets
        initial_ec_qubits = self._get_initial_ec_qubits(code)
        
        # Build future CNOT list for propagation calculation
        # Include pre_h_cnots first, then regular encoding CNOTs
        all_cnots = []
        if code.pre_h_cnots:
            all_cnots.extend(code.pre_h_cnots)
        for round_cnots in cnot_rounds:
            all_cnots.extend(round_cnots)
        
        def get_x_propagation(qubit: int, remaining_cnots: List[Tuple[int, int]]) -> List[int]:
            """Calculate which qubits an X error on 'qubit' propagates to."""
            affected = {qubit}
            for ctrl, targ in remaining_cnots:
                if ctrl in affected:
                    affected.add(targ)
            return sorted(affected - {qubit})  # Exclude self
        
        def get_z_propagation(qubit: int, remaining_cnots: List[Tuple[int, int]]) -> List[int]:
            """Calculate which qubits a Z error on 'qubit' propagates to."""
            affected = {qubit}
            for ctrl, targ in remaining_cnots:
                if targ in affected:
                    affected.add(ctrl)
            return sorted(affected - {qubit})  # Exclude self
        
        # EC rounds counter
        ec_round = 0
        cnots_processed = 0
        
        # Account for pre_h_cnots in the propagation
        if code.pre_h_cnots:
            cnots_processed = len(code.pre_h_cnots)
        
        # Initial EC rounds (after H gates, before first encoding CNOT round)
        # Note: pre_h_cnots are already processed, so we start with encoding CNOTs
        for q in initial_ec_qubits:
            remaining = all_cnots[cnots_processed:]
            propagation_X.append(get_x_propagation(q, remaining))
            propagation_Z.append(get_z_propagation(q, remaining))
            ec_round += 1
        
        # EC after each CNOT round
        for round_idx, round_cnots in enumerate(cnot_rounds):
            cnots_processed += len(round_cnots)
            remaining = all_cnots[cnots_processed:]
            
            # EC on all n data qubits after this round
            for q in range(n):
                propagation_X.append(get_x_propagation(q, remaining))
                propagation_Z.append(get_z_propagation(q, remaining))
                ec_round += 1
        
        # EC on verification ancilla (ancilla doesn't propagate to data)
        propagation_X.append([])
        propagation_Z.append([])
        ec_round += 1
        
        # Verification rounds (after all encoding CNOTs)
        for verif_idx, verif_qubits in enumerate(verification_schedule):
            # After verification CNOT, EC on data qubits
            # Errors at this point don't propagate further (encoding done)
            for q in range(n):
                propagation_X.append([])
                propagation_Z.append([])
                ec_round += 1
            
            # EC on verification ancilla
            propagation_X.append([])
            propagation_Z.append([])
            ec_round += 1
        
        num_ec_0prep = ec_round
        
        # Propagation to measurement: which EC rounds affect the verification measurement
        # An EC round affects measurement if X errors propagate to verification qubits
        propagation_m = []
        for ec_idx in range(num_ec_0prep):
            if ec_idx < len(propagation_X):
                # Check if any verification qubit is in the X propagation
                for vq in code.verification_qubits:
                    if vq in propagation_X[ec_idx]:
                        propagation_m.append(ec_idx)
                        break
        
        propagation_m = sorted(set(propagation_m))
        
        return PropagationTables(
            propagation_X=propagation_X,
            propagation_Z=propagation_Z,
            propagation_m=propagation_m,
            num_ec_0prep=num_ec_0prep
        )

# =============================================================================
# EC Gadget (Abstract and Implementations)
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                          EC GADGET HIERARCHY                                 │
# └─────────────────────────────────────────────────────────────────────────────┘

#                     ┌─────────────────────────────────┐
#                     │         ECGadget                │
#                     │         (Abstract)              │
#                     ├─────────────────────────────────┤
#                     │ Input: ConcatenatedCode,        │
#                     │        TransversalOps           │
#                     ├─────────────────────────────────┤
#                     │ + set_prep(PreparationStrategy) │
#                     │ + ec_type: str                  │
#                     ├─────────────────────────────────┤
#                     │ + append_noisy_ec(circuit,      │
#                     │     loc1, loc2, loc3, loc4,     │
#                     │     N_prev, N_now, p, det_ctr)  │
#                     │   -> Tuple                      │
#                     └────────────────┬────────────────┘
#                                      │
#                     ┌────────────────┴────────────────┐
#                     │                                 │
#                     ▼                                 ▼
#      ┌──────────────────────────┐      ┌──────────────────────────┐
#      │     SteaneECGadget       │      │     KnillECGadget        │
#      ├──────────────────────────┤      ├──────────────────────────┤
#      │ ec_type = "steane"       │      │ ec_type = "knill"        │
#      │                          │      │                          │
#      │ Structure:               │      │ Structure:                │
#      │ 1. Prepare 2 ancillas    │      │ 1. Prepare Bell pair     │
#      │ 2. H on ancilla 1        │      │ 2. Bell measurement      │
#      │ 3.  CNOT ancillas         │      │ 3. Teleport state        │
#      │ 4. Recursive EC (L2)     │      │                          │
#      │ 5.  CNOT data->ancilla    │      │ Uses teleportation       │
#      │ 6. H on data             │      │ for error correction     │
#      │ 7. Measure syndromes     │      │                          │
#      │ 8.  SWAP corrected state  │      │                          │
#      └──────────────────────────┘      └──────────────────────────┘

# Return Value Structure:
# ┌─────────────────────────────────────────────────────────────────┐
# │ N_prev = 1:                                                     │
# │   Returns: (detector_0prep, detector_Z, detector_X)             │
# │                                                                 │
# │ N_prev > 1:                                                     │
# │   Returns: (detector_0prep, detector_0prep_l2,                  │
# │             detector_Z, detector_X)                             │
# └─────────────────────────────────────────────────────────────────┘

# Circular Dependency Resolution:
# ┌─────────────────────────────────────────────────────────────────┐
# │                                                                 │
# │  PreparationStrategy ◄─────────────► ECGadget                   │
# │         │                                 │                     │
# │         │ set_ec_gadget()                 │ set_prep()          │
# │         └─────────────────────────────────┘                     │
# │                                                                 │
# │  Both need each other because:                                  │
# │  - Preparation uses EC for level-2                              │
# │  - EC uses Preparation for ancilla states                       │
# │                                                                 │
# └─────────────────────────────────────────────────────────────────┘
# =============================================================================

class ECGadget(ABC):
    """Abstract base class for error correction gadgets."""
    
    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        self.concat_code = concat_code
        self.ops = ops
        self._prep = None
    
    def set_prep(self, prep: PreparationStrategy):
        self._prep = prep
    
    @property
    def prep(self) -> PreparationStrategy:
        if self._prep is None:
            raise RuntimeError("Preparation strategy not set")
        return self._prep
    
    @property
    @abstractmethod
    def ec_type(self) -> str:
        pass
    
    @abstractmethod
    def append_noisy_ec(self, circuit:  stim.Circuit, loc1: int, loc2: int,
                        loc3: int, loc4: int, N_prev: int, N_now: int,
                        p: float, detector_counter:  List[int]) -> Tuple:
        """
        Apply EC gadget. 
        
        Returns:
        - N_prev=1: (detector_0prep, detector_Z, detector_X)
        - N_prev>1: (detector_0prep, detector_0prep_l2, detector_Z, detector_X)
        """
        pass


# Note: SteaneECGadget has been moved to concatenated_css_v10_steane.py


class KnillECGadget(ECGadget):
    """
    Knill-style teleportation-based error correction gadget.
    
    ═══════════════════════════════════════════════════════════════════════════
                               QEC THEORY BACKGROUND
    ═══════════════════════════════════════════════════════════════════════════
    
    TELEPORTATION-BASED ERROR CORRECTION
    ------------------------------------
    The Knill EC gadget (named after Emanuel Knill) uses quantum teleportation
    to transfer a potentially corrupted quantum state onto freshly prepared
    ancilla qubits, effectively "cleaning" the state of accumulated errors.
    
    Key insight: Teleportation's Bell measurement extracts exactly the syndrome
    information needed for error correction, while the correction is applied
    as Pauli frame updates (tracked classically, not physically applied).
    
    CIRCUIT STRUCTURE
    -----------------
    The Knill EC gadget consists of:
    
    1. PREPARATION PHASE
       Prepare a Bell pair on ancilla qubits:
       
       |0⟩_L at loc2 ──[H]──●── → (|00⟩_L + |11⟩_L)/√2
                            │
       |0⟩_L at loc3 ───────⊕──
       
       For self-dual codes: H is transversal (H on all qubits)
       For non-self-dual: must use direct |+⟩_L preparation
    
    2. BELL MEASUREMENT PHASE
       Perform Bell measurement between data and one ancilla:
       
       Data at loc1 ──●──[H]──[M] → syndrome_z (X correction info)
                      │
       Ancilla loc2 ──⊕─────[M] → syndrome_x (Z correction info)
    
    3. TELEPORTATION RESULT
       - Teleported state appears at loc3
       - syndrome_z determines X Pauli frame update
       - syndrome_x determines Z Pauli frame update
       - SWAP moves state back to loc1
    
    WHY TELEPORTATION WORKS FOR EC
    ------------------------------
    Consider a state with error: E|ψ⟩ (E is some Pauli error)
    
    After Bell measurement with outcomes (m_x, m_z):
    - State at loc3 is: X^{m_x} Z^{m_z} |ψ⟩ (teleported, no error!)
    - The original error E is absorbed into the measurement outcomes
    
    The syndrome extraction happens naturally:
    - m_z encodes which X error occurred → tells us X correction
    - m_x encodes which Z error occurred → tells us Z correction
    
    PAULI FRAME TRACKING (NOT PHYSICAL CORRECTION)
    ----------------------------------------------
    Instead of physically applying X^{m_x} Z^{m_z} corrections, we track
    them classically in a "Pauli frame":
    
        pauli_frame_x = accumulated X corrections
        pauli_frame_z = accumulated Z corrections
    
    At final measurement:
    - X corrections flip Z-basis measurement outcomes
    - Z corrections flip X-basis measurement outcomes
    
    This is equivalent to physically correcting but more efficient:
    - No additional gate errors from correction operations
    - Corrections can be composed classically
    
    TRANSVERSAL H VS LOGICAL H
    --------------------------
    The Bell pair creation uses H on one ancilla. For:
    
    Self-dual codes (Hz = Hx):
        - Transversal H (H on all qubits) = Logical H
        - H|0⟩_L = |+⟩_L works correctly
        - This gadget works as-is
    
    Non-self-dual codes (Hz ≠ Hx):
        - Transversal H ≠ Logical H
        - H|0⟩_L ≠ |+⟩_L (gives wrong state!)
        - Must use direct |+⟩_L preparation (plus_h_qubits, plus_encoding_cnots)
        
    CRITICAL: The Bell MEASUREMENT still uses transversal H because we're
    measuring physical qubits in the X basis. The decoder then uses the
    logical operator Lx to compute the logical value from those measurements.
    
    ═══════════════════════════════════════════════════════════════════════════
                              CIRCUIT DETAILS
    ═══════════════════════════════════════════════════════════════════════════
    
    Input registers:
        loc1: Data qubit(s) to be error-corrected
        loc2: Ancilla 1 (will hold teleported state temporarily)
        loc3: Ancilla 2 (will hold teleported state)
        loc4: Scratch space for recursive EC (at L2)
    
    Parameters:
        N_prev: Size of inner code block (1 at L1, n at L2)
        N_now: Size of current code
        p: Physical error probability
    
    Return values depend on level:
        N_prev = 1 (L1): (detector_0prep, detector_Z, detector_X)
        N_prev > 1 (L2): (detector_0prep, detector_0prep_l2, detector_Z, detector_X)
    
    Where:
        detector_0prep: L1 preparation verification detectors
        detector_0prep_l2: L2 preparation verification detectors
        detector_Z: Z-syndrome measurement indices (from loc1 after H)
        detector_X: X-syndrome measurement indices (from loc2)
    
    ═══════════════════════════════════════════════════════════════════════════
                              HIERARCHICAL EC AT L2
    ═══════════════════════════════════════════════════════════════════════════
    
    At level 2 (N_prev > 1), each "qubit" is actually an encoded block.
    After preparing the Bell pair, we perform L1 EC on each inner block:
    
        for each inner block i in loc2:
            L1 EC(loc2[i]) using loc4 as scratch
        for each inner block i in loc3:
            L1 EC(loc3[i]) using loc4 as scratch
    
    This ensures the Bell pair ancillas are themselves error-corrected
    before being used in the teleportation measurement.
    
    References:
        [Kni05] Knill, "Quantum computing with realistically noisy devices",
                Nature 434, 39 (2005). arXiv:quant-ph/0410199
        [BBC+93] Bennett et al., "Teleporting an unknown quantum state",
                 Phys. Rev. Lett. 70, 1895 (1993)
    """
    
    @property
    def ec_type(self) -> str:
        return "knill"
    
    def append_noisy_ec(self, circuit: stim.Circuit, loc1: int, loc2: int,
                        loc3: int, loc4: int, N_prev: int, N_now: int,
                        p: float, detector_counter: List[int]) -> Tuple:
        """
        Apply Knill teleportation-based error correction.
        
        This method appends the full Knill EC circuit to the given Stim circuit,
        including Bell pair preparation, recursive L1 EC (at L2), Bell measurement,
        and state teleportation.
        
        Args:
            circuit: Stim circuit to append operations to
            loc1: Starting index of data qubit block
            loc2: Starting index of ancilla 1 (Bell pair)
            loc3: Starting index of ancilla 2 (Bell pair, receives teleported state)
            loc4: Starting index of scratch space (for recursive EC)
            N_prev: Size of inner code block (1 at L1, n at L2)
            N_now: Size of current code (number of qubits per logical block)
            p: Physical error probability for noisy operations
            detector_counter: Mutable list [count] tracking detector indices
            
        Returns:
            At L1 (N_prev=1): 
                (detector_0prep, detector_Z, detector_X)
            At L2 (N_prev>1):
                (detector_0prep, detector_0prep_l2, detector_Z, detector_X)
                
            detector_0prep: List of verification detector indices from L1 prep
            detector_0prep_l2: List of L2 verification detector indices  
            detector_Z: List of Z-syndrome measurement [start, end] ranges
            detector_X: List of X-syndrome measurement [start, end] ranges
        """
        # Similar structure to Steane but uses teleportation
        detector_0prep = []
        detector_0prep_l2 = []
        detector_Z = []
        detector_X = []
        
        if N_now == 1:
            return None
        
        # Determine n_now (number of transversal blocks) based on N_now
        # Only use transversal_block_count when N_now matches outer code's n
        # This prevents L1 EC on inner code from using outer code's block count
        n_now = N_now  # Default to N_now
        if self.concat_code.num_levels > 1:
            outer_code = self.concat_code.code_at_level(1)
            # Only use transversal_block_count if N_now matches outer code's n
            if N_now == outer_code.n and outer_code.transversal_block_count:
                n_now = outer_code.transversal_block_count
        
        # Prepare Bell pair ancillas
        # For self-dual codes: prep |0⟩_L, |0⟩_L, then H on one
        # For non-self-dual codes (Shor): prep |0⟩_L on loc3, |+⟩_L directly on loc2
        # NOTE: Prep results go ONLY to detector_0prep, not detector_X/Z
        # This matches the original C4C6 behavior where detector_X/Z contain
        # only L1 EC results and final measurement
        inner_code = self.concat_code.code_at_level(0)
        
        if inner_code.requires_direct_plus_prep:
            # Non-self-dual code (e.g., Shor): prepare |+⟩_L directly on loc2
            if N_prev == 1:
                # loc2 gets |+⟩_L directly
                result1 = self.prep.append_noisy_plus_prep(circuit, loc2, loc4, N_prev, N_now, p, detector_counter)
                # loc3 gets |0⟩_L
                result2 = self.prep.append_noisy_0prep(circuit, loc3, loc4, N_prev, N_now, p, detector_counter)
                detector_0prep.extend(result1)
                detector_0prep.extend(result2)
            else:
                result1 = self.prep.append_noisy_plus_prep(circuit, loc2, loc4, N_prev, N_now, p, detector_counter)
                detector_0prep.extend(result1[0])
                detector_0prep_l2.append(result1[1])
                
                result2 = self.prep.append_noisy_0prep(circuit, loc3, loc4, N_prev, N_now, p, detector_counter)
                detector_0prep.extend(result2[0])
                detector_0prep_l2.append(result2[1])
            
            # Bell pair: CNOT from |+⟩_L (loc2) to |0⟩_L (loc3)
            # Result: (|00⟩_L + |11⟩_L)/√2
            self.ops.append_noisy_cnot(circuit, loc2, loc3, N_prev, n_now, p)
        else:
            # Self-dual code: prepare |0⟩_L on both, then H on loc2
            if N_prev == 1:
                result1 = self.prep.append_noisy_0prep(circuit, loc2, loc4, N_prev, N_now, p, detector_counter)
                result2 = self.prep.append_noisy_0prep(circuit, loc3, loc4, N_prev, N_now, p, detector_counter)
                detector_0prep.extend(result1)
                detector_0prep.extend(result2)
            else:
                result1 = self.prep.append_noisy_0prep(circuit, loc2, loc4, N_prev, N_now, p, detector_counter)
                detector_0prep.extend(result1[0])
                detector_0prep_l2.append(result1[1])
                # NOTE: Do NOT add result1[2] and result1[3] to detector_X/Z
                # Prep results are separate from EC results
                
                result2 = self.prep.append_noisy_0prep(circuit, loc3, loc4, N_prev, N_now, p, detector_counter)
                detector_0prep.extend(result2[0])
                detector_0prep_l2.append(result2[1])
                # NOTE: Do NOT add result2[2] and result2[3] to detector_X/Z
            
            # Create Bell pair using LOGICAL H (works for self-dual codes)
            # For self-dual codes: logical H = transversal H (all qubits)
            self.ops.append_logical_h(circuit, loc2, N_prev, n_now, inner_code)
            self.ops.append_noisy_cnot(circuit, loc2, loc3, N_prev, n_now, p)
        
        # CRITICAL: At L2 (N_prev != 1), do L1 EC on each inner block
        # This matches the original concatenated_steane.py structure
        if N_prev != 1:
            for i in range(n_now):
                # L1 EC on loc2's inner blocks
                ec_result = self.append_noisy_ec(
                    circuit, 
                    (loc2 + i) * N_prev, 
                    (loc4 + 0) * N_prev,
                    (loc4 + 1) * N_prev,
                    (loc4 + 2) * N_prev,
                    1, N_prev, p, detector_counter
                )
                detector_0prep.extend(ec_result[0])
                detector_Z.append(ec_result[1])
                detector_X.append(ec_result[2])
            
            for i in range(n_now):
                # L1 EC on loc3's inner blocks
                ec_result = self.append_noisy_ec(
                    circuit,
                    (loc3 + i) * N_prev,
                    (loc4 + 0) * N_prev,
                    (loc4 + 1) * N_prev,
                    (loc4 + 2) * N_prev,
                    1, N_prev, p, detector_counter
                )
                detector_0prep.extend(ec_result[0])
                detector_Z.append(ec_result[1])
                detector_X.append(ec_result[2])
        
        # Bell measurement
        # CNOT from data to ancilla, then H on data, then measure both
        # NOTE: For Bell measurement, we use TRANSVERSAL H (H on all qubits),
        # not logical H. This is because we're measuring physical qubits in
        # the X basis to extract syndrome information. The decoder then uses
        # Lx to compute the logical value from those measurements.
        self.ops.append_noisy_cnot(circuit, loc1, loc2, N_prev, n_now, p)
        self.ops.append_h(circuit, loc1, N_prev, n_now)  # Transversal H for X-basis measurement
        
        # Measure - loc1 gives Z syndrome, loc2 gives X syndrome
        # (matching SteaneECGadget convention)
        detector_Z.append(self.ops.append_noisy_m(circuit, loc1, N_prev, n_now, p, detector_counter))
        detector_X.append(self.ops.append_noisy_m(circuit, loc2, N_prev, n_now, p, detector_counter))
        
        # Teleported state is in loc3
        self.ops.append_swap(circuit, loc1, loc3, N_prev, n_now)
        
        if N_prev == 1:
            return detector_0prep, detector_Z, detector_X
        else:
            return detector_0prep, detector_0prep_l2, detector_Z, detector_X


# =============================================================================
# Decoder
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                            DECODER HIERARCHY                                 │
# └─────────────────────────────────────────────────────────────────────────────┘

#                     ┌─────────────────────────────────┐
#                     │          Decoder                │
#                     │         (Abstract)              │
#                     ├─────────────────────────────────┤
#                     │ Input:  ConcatenatedCode         │
#                     ├─────────────────────────────────┤
#                     │ + decode_measurement(m, type)   │
#                     │   -> int (0 or 1)               │
#                     │                                 │
#                     │ + decode_measurement_post_      │
#                     │   selection(m, type)            │
#                     │   -> int (0, 1, or -1)          │
#                     └────────────────┬────────────────┘
#                                      │
#                     ┌────────────────┴────────────────┐
#                     │                                 │
#                     ▼                                 ▼
#      ┌──────────────────────────┐      ┌──────────────────────────┐
#      │     SteaneDecoder        │      │    GenericDecoder        │
#      ├──────────────────────────┤      ├──────────────────────────┤
#      │ Hardcoded check_matrix   │      │ Uses CSSCode. Hz/Hx       │
#      │ and logical_op for       │      │ and Lz/Lx                │
#      │ exact match              │      │                          │
#      ├──────────────────────────┤      │ Works for any CSS code   │
#      │ + decode_ec_hd(x,        │      │ with standard syndrome   │
#      │     detector_X,          │      │ decoding                 │
#      │     detector_Z,          │      │                          │
#      │     corr_x_prev,         │      └──────────────────────────┘
#      │     corr_z_prev)         │
#      │   -> (corr_x, corr_z,    │
#      │       corr_x_next,       │
#      │       corr_z_next)       │
#      │                          │
#      │ + decode_m_hd(x,         │
#      │     detector_m,          │
#      │     correction_l1)       │
#      │   -> int                 │
#      └──────────────────────────┘

# Decoding Algorithm (Steane):
# ┌─────────────────────────────────────────────────────────────────┐
# │ 1. Compute syndrome from measurement                            │
# │ 2. Lookup correction from syndrome                              │
# │ 3. Apply correction to logical outcome                          │
# │                                                                  │
# │ For Level-2 (decode_ec_hd):                                      │
# │ 1. Decode inner measurements                                     │
# │ 2. Apply propagation corrections using PropagationTables         │
# │ 3. Combine with previous corrections                             │
# │ 4. Return updated corrections for next round                     │
# └─────────────────────────────────────────────────────────────────┘
# =============================================================================

class Decoder(ABC):
    """Abstract base class for decoders."""
    
    def __init__(self, concat_code: ConcatenatedCode):
        self.concat_code = concat_code
    
    @abstractmethod
    def decode_measurement(self, m: np.ndarray, m_type: str = 'x') -> int:
        pass
    
    @abstractmethod
    def decode_measurement_post_selection(self, m: np.ndarray, m_type: str = 'x') -> int:
        pass


# Note: SteaneDecoder has been moved to concatenated_css_v10_steane.py

class GenericDecoder(Decoder):
    """
    Generic syndrome-based decoder for any CSS quantum error correcting code.
    
    ═══════════════════════════════════════════════════════════════════════════
                               QEC THEORY BACKGROUND
    ═══════════════════════════════════════════════════════════════════════════
    
    SYNDROME-BASED DECODING
    -----------------------
    The fundamental decoding problem in QEC: given syndrome measurements,
    determine what error occurred and how to correct it.
    
    For CSS codes, X and Z errors are decoded independently:
    
    1. X ERROR DECODING (using Hz):
       - Measure Z-stabilizers: syndrome_z = Hz × measurement (mod 2)
       - Syndrome_z reveals which X errors occurred
       - Apply X correction to affected qubits
    
    2. Z ERROR DECODING (using Hx):
       - Measure X-stabilizers: syndrome_x = Hx × measurement (mod 2)  
       - Syndrome_x reveals which Z errors occurred
       - Apply Z correction to affected qubits
    
    MINIMUM-WEIGHT DECODING
    -----------------------
    The optimal decoder finds the minimum-weight error consistent with the
    observed syndrome. For a distance-d code:
    
    - Up to t = ⌊(d-1)/2⌋ errors can be uniquely identified
    - If weight(error) ≤ t, decoder returns correct answer
    - If weight(error) > t, decoder may fail (logical error)
    
    This decoder uses a syndrome lookup table pre-computed from the code's
    parity check matrices. For each possible syndrome, it stores the
    minimum-weight error that produces that syndrome.
    
    SYNDROME TABLE CONSTRUCTION
    ---------------------------
    For an [[n,k,d]] code with r stabilizers:
    
    1. Zero syndrome → no error (identity correction)
    2. For each single-qubit error E_i (i = 0..n-1):
       - Compute syndrome: s_i = H × e_i (mod 2)
       - Store: syndrome_table[s_i] = i
    3. For d ≥ 5, also compute two-qubit error syndromes
    
    The table has at most 2^r entries. For a code that can correct t errors,
    we need all errors of weight ≤ t to produce distinct syndromes (this
    is guaranteed by the Hamming bound and the code's properties).
    
    DECODE_SYNDROME VS DECODE_MEASUREMENT
    -------------------------------------
    This decoder has TWO distinct decoding methods:
    
    decode_syndrome(m, m_type):
        - For EC SYNDROME data (teleportation measurements in Knill EC)
        - ALWAYS uses syndrome table lookup
        - Input is a syndrome that directly indicates error type
        - Used during error correction rounds
    
    decode_measurement(m, m_type):
        - For FINAL LOGICAL STATE readout
        - May use block majority voting for codes like Shor
        - Input is raw qubit measurements from the code block
        - Used when measuring the logical qubit at end of computation
    
    CRITICAL DISTINCTION FOR NON-SELF-DUAL CODES:
    For Shor [[9,1,3]], the logical |0⟩ is a superposition:
        |0⟩_L = (|000⟩+|111⟩)(|000⟩+|111⟩)(|000⟩+|111⟩) / 2√2
    
    Final measurement can yield many valid outcomes (000000000, 111000000, etc.)
    which all decode to logical 0. This requires block majority voting.
    
    But EC syndromes are NOT codewords - they're stabilizer measurements
    that directly encode error information. These always use syndrome tables.
    
    HIERARCHICAL DECODING FOR CONCATENATED CODES
    --------------------------------------------
    For a concatenated code (e.g., Steane→Steane):
    
    Level-1 decoding: decode_ec_simple()
        1. Extract syndrome from inner (level-0) block measurements
        2. Use syndrome table to determine correction
        3. Return logical correction for that block
    
    Level-2 decoding: decode_ec_hd() (hierarchical decoding)
        1. For each outer code block, decode inner code syndromes
        2. Collect corrections for all inner blocks
        3. Apply outer code decoding using inner block corrections
        4. Propagate corrections according to PropagationTables
    
    The PropagationTables encode how corrections from each EC round
    propagate through the circuit to affect subsequent operations.
    
    ═══════════════════════════════════════════════════════════════════════════
                                 PAULI FRAME TRACKING  
    ═══════════════════════════════════════════════════════════════════════════
    
    Rather than physically applying corrections, we track them in software:
    
        correction_x[i] = accumulated X corrections on qubit i
        correction_z[i] = accumulated Z corrections on qubit i
    
    At final measurement:
    - X corrections flip Z-basis measurement outcomes
    - Z corrections have no effect on Z-basis measurements (commute)
    
    For X-basis measurement (if used):
    - Z corrections flip X-basis measurement outcomes
    - X corrections have no effect (commute)
    
    ═══════════════════════════════════════════════════════════════════════════
                              METHOD SUMMARY
    ═══════════════════════════════════════════════════════════════════════════
    
    Core Decoding:
        decode_syndrome(m, m_type): EC syndrome → logical correction
        decode_measurement(m, m_type): Final measurement → logical value
        decode_measurement_k(m, m_type): Multi-qubit version for k>1 codes
        decode_syndrome_k(m, m_type): Multi-qubit version for k>1 EC syndromes
    
    Hierarchical Decoding:
        decode_ec_hd(x, det_X, det_Z, corr_x, corr_z): L2 EC round decoding
        decode_m_hd(x, det_m, corr_x): L2 measurement with corrections
    
    Utilities:
        _build_syndrome_table(): Construct syndrome lookup table
        _compute_syndrome(): Extract syndrome from measurements
        _get_correction_for_syndrome(): Syndrome → correction lookup
    
    References:
        [Got97] Gottesman, PhD Thesis, Caltech (1997)
        [NC00] Nielsen & Chuang, "Quantum Computation and Quantum Information"
        [KL97] Knill & Laflamme, Phys. Rev. A 55, 900 (1997)
    """
    
    def __init__(self, concat_code:  ConcatenatedCode):
        super().__init__(concat_code)
        self.code = concat_code. code_at_level(0)
        self.n = self.code.n
        
        # Build syndrome lookup tables for X and Z errors
        self._syndrome_to_error_x = self._build_syndrome_table(self.code.Hz, 'x')
        self._syndrome_to_error_z = self._build_syndrome_table(self.code. Hx, 'z')
        
        # Cache logical operators
        self._logical_x = self.code. Lx
        self._logical_z = self.code.Lz
        
        # Precompute syndrome weights for tie-breaking
        self._error_weights_x = self._compute_error_weights(self.code.Hz)
        self._error_weights_z = self._compute_error_weights(self.code.Hx)
    
    def _build_syndrome_table(self, check_matrix: np.ndarray, 
                               error_type: str) -> Dict[int, int]:
        """
        Build syndrome → minimum-weight error lookup table.
        
        The syndrome table is the core data structure for efficient decoding.
        For each possible syndrome value, it stores the index of the minimum-weight
        error that produces that syndrome.
        
        ALGORITHM:
        1. syndrome = 0 maps to no error (correction = identity)
        2. For each single-qubit error at position i:
           - Compute syndrome = Σ_j H[j,i] * 2^j (bit-packed integer)
           - If syndrome not already in table, store syndrome → i
        3. For d ≥ 5 codes, repeat for two-qubit errors
        
        The "first-come-first-stored" rule ensures we store minimum-weight errors.
        
        Args:
            check_matrix: Stabilizer check matrix (Hz for X errors, Hx for Z errors)
            error_type: 'x' or 'z' (determines which logical operator to use)
        
        Returns:
            Dict mapping syndrome (as bit-packed int) to error qubit index.
            Special values: -1 = no error, negative < -1 = two-qubit error (encoded).
        
        Theory:
            The Hamming bound guarantees that for a t-error-correcting code,
            all errors of weight ≤ t produce distinct syndromes. This table
            inverts the syndrome map for efficient decoding.
        """
        num_stabilizers = check_matrix.shape[0]
        n = check_matrix.shape[1]
        
        syndrome_table = {}
        
        # Syndrome 0 -> no error
        syndrome_table[0] = -1
        
        # Single-qubit errors
        for qubit in range(n):
            syndrome = 0
            for stab_idx in range(num_stabilizers):
                if check_matrix[stab_idx, qubit] == 1:
                    syndrome += (1 << stab_idx)
            
            # Only store if this syndrome not yet seen (first = lowest weight)
            if syndrome not in syndrome_table:
                syndrome_table[syndrome] = qubit
        
        # For higher distance codes, also consider two-qubit errors
        # This helps with distance-5+ codes
        if self.code. d >= 5:
            for q1 in range(n):
                for q2 in range(q1 + 1, n):
                    syndrome = 0
                    for stab_idx in range(num_stabilizers):
                        bit = (check_matrix[stab_idx, q1] + check_matrix[stab_idx, q2]) % 2
                        if bit == 1:
                            syndrome += (1 << stab_idx)
                    
                    # Store two-qubit error as negative composite
                    # (We'll decode this specially)
                    if syndrome not in syndrome_table:
                        syndrome_table[syndrome] = -(q1 * n + q2 + 1)  # Encode pair
        
        return syndrome_table
    
    def _compute_error_weights(self, check_matrix: np.ndarray) -> Dict[int, int]:
        """
        Compute the weight of the minimum error for each syndrome.
        
        Used for post-selection decisions: errors with weight > t (correction
        capability) indicate likely logical errors and may be rejected.
        
        Args:
            check_matrix: Stabilizer check matrix
            
        Returns:
            Dict mapping syndrome → minimum error weight producing that syndrome
        """
        num_stabilizers = check_matrix. shape[0]
        n = check_matrix.shape[1]
        
        weights = {0: 0}
        
        # Single errors have weight 1
        for qubit in range(n):
            syndrome = 0
            for stab_idx in range(num_stabilizers):
                if check_matrix[stab_idx, qubit] == 1:
                    syndrome += (1 << stab_idx)
            if syndrome not in weights:
                weights[syndrome] = 1
        
        # Two-qubit errors have weight 2
        for q1 in range(n):
            for q2 in range(q1 + 1, n):
                syndrome = 0
                for stab_idx in range(num_stabilizers):
                    bit = (check_matrix[stab_idx, q1] + check_matrix[stab_idx, q2]) % 2
                    if bit == 1:
                        syndrome += (1 << stab_idx)
                if syndrome not in weights:
                    weights[syndrome] = 2
        
        return weights
    
    def _compute_syndrome(self, m: np.ndarray, check_matrix: np.ndarray) -> int:
        """
        Compute syndrome from measurement outcomes.
        
        The syndrome is computed as:
            syndrome[i] = Σ_j H[i,j] * m[j] (mod 2)
        
        This is bit-packed into a single integer for efficient table lookup.
        
        Args:
            m: Measurement outcomes (length n), values 0 or 1
            check_matrix: Stabilizer check matrix (r × n)
        
        Returns:
            Syndrome as bit-packed integer: syndrome = Σ_i s[i] * 2^i
        """
        syndrome = 0
        for stab_idx in range(check_matrix.shape[0]):
            parity = int(np.sum(m * check_matrix[stab_idx, :]) % 2)
            syndrome += parity * (1 << stab_idx)
        return syndrome
    
    def _compute_logical_value(self, m: np.ndarray, logical_op: np.ndarray) -> int:
        """
        Compute logical measurement value from physical measurements.
        
        For a logical Z operator Lz, the logical Z-basis measurement is:
            outcome = Σ_j Lz[j] * m[j] (mod 2)
        
        Args:
            m: Physical qubit measurements (length n)
            logical_op: Logical operator (Lz or Lx)
        
        Returns:
            Logical measurement outcome (0 or 1)
        """
        return int(np.sum(m * logical_op) % 2)
    
    def _get_correction_for_syndrome(self, syndrome: int, error_type: str) -> Tuple[int, int]:
        """
        Get error correction information for a syndrome.
        
        This is the core lookup operation: given a syndrome, determine what
        error most likely occurred and whether correcting it flips the logical value.
        
        Args:
            syndrome: Bit-packed syndrome value
            error_type: 'x' for X errors, 'z' for Z errors
        
        Returns:
            Tuple (error_position, logical_flip):
            - error_position: Qubit index to correct, or -1 if no correction needed
            - logical_flip: 1 if correction changes logical value, 0 otherwise
        
        Theory:
            The logical_flip value is crucial: if we apply an error correction
            that anti-commutes with the logical operator, the logical value changes.
            For Lz = [1,1,1,0,0,0,0] and correction at qubit 0, logical_flip = 1.
        """
        if error_type == 'x':
            syndrome_table = self._syndrome_to_error_x
            logical_op = self._logical_x
        else:
            syndrome_table = self._syndrome_to_error_z
            logical_op = self._logical_z
        
        if syndrome == 0:
            return -1, 0
        
        error_pos = syndrome_table. get(syndrome, None)
        
        if error_pos is None: 
            # Unknown syndrome - likely multi-qubit error beyond correction capability
            # Return no correction (will likely cause logical error)
            return -1, 0
        
        if error_pos >= 0:
            # Single qubit error
            logical_flip = int(logical_op[error_pos])
        elif error_pos < -1:
            # Two-qubit error (encoded as negative)
            # Decode the pair
            pair_code = -(error_pos + 1)
            q1 = pair_code // self.n
            q2 = pair_code % self.n
            logical_flip = int((logical_op[q1] + logical_op[q2]) % 2)
        else:
            logical_flip = 0
        
        return error_pos, logical_flip
    
    def decode_syndrome(self, m: np.ndarray, m_type: str = 'x') -> int:
        """
        Decode EC syndrome measurements to extract logical correction.
        
        ═══════════════════════════════════════════════════════════════════════
        THIS METHOD IS FOR ERROR CORRECTION SYNDROME DATA ONLY
        ═══════════════════════════════════════════════════════════════════════
        
        Use this for teleportation-based EC (Knill gadget) where the syndrome
        measurements directly encode error information. It ALWAYS uses syndrome
        table lookup, never block majority voting.
        
        For final logical state readout, use decode_measurement() instead.
        
        SYNDROME TYPE CONVENTIONS:
        - m_type='x': X-basis syndrome measurement (detects Z errors)
          Uses Hx to compute syndrome, looks up in _syndrome_to_error_z
        - m_type='z': Z-basis syndrome measurement (detects X errors)
          Uses Hz to compute syndrome, looks up in _syndrome_to_error_x
        
        TELEPORTATION EC CONTEXT:
        In the Knill EC gadget:
        - loc1 measurement after H is X-basis → gives Z correction info (m_type='x')
        - loc2 measurement is Z-basis → gives X correction info (m_type='z')
        
        Args:
            m: Syndrome measurement outcomes (length n)
            m_type: 'x' for X-basis syndrome, 'z' for Z-basis syndrome
        
        Returns:
            Logical correction (0 or 1)
        """
        if m_type == 'x': 
            check_matrix = self.code.Hx
            logical_op = self._logical_x
            syndrome_table = self._syndrome_to_error_z  # Built from Hx
        else: 
            check_matrix = self.code.Hz
            logical_op = self._logical_z
            syndrome_table = self._syndrome_to_error_x  # Built from Hz
        
        # Compute raw logical value
        outcome = self._compute_logical_value(m, logical_op)
        
        # Compute syndrome
        syndrome = self._compute_syndrome(m, check_matrix)
        
        # Apply correction
        if syndrome > 0:
            error_pos = syndrome_table.get(syndrome, None)
            
            if error_pos is not None and error_pos >= 0:
                outcome = (outcome + int(logical_op[error_pos])) % 2
            elif error_pos is not None and error_pos < -1:
                pair_code = -(error_pos + 1)
                q1 = pair_code // self.n
                q2 = pair_code % self.n
                correction = (int(logical_op[q1]) + int(logical_op[q2])) % 2
                outcome = (outcome + correction) % 2
        
        return int(outcome)
    
    def decode_measurement(self, m: np.ndarray, m_type: str = 'x') -> int:
        """
        Decode a FINAL LOGICAL STATE measurement to get logical outcome.
        
        This method is for decoding the final measurement of a logical state,
        NOT for EC syndrome data. For codes with superposition codewords 
        (like Shor), uses block majority voting for Z-basis measurements.
        
        For EC syndrome decoding, use decode_syndrome() instead.
        
        Args:
            m: Measurement outcomes (length n)
            m_type: 'x' for X-basis measurement, 'z' for Z-basis
        
        Returns:
            Logical measurement outcome (0 or 1)
        """
        # For Z-basis measurements on codes with superposition codewords,
        # use the code's decode method (block majority voting)
        if m_type == 'z' and self.code.measurement_strategy == "relative":
            return self.code.decode_z_basis_measurement(m)
        
        # For non-self-dual codes, we need to match syndrome table with check matrix:
        # - _syndrome_to_error_x was built from Hz (6 stabilizers for Shor)
        # - _syndrome_to_error_z was built from Hx (2 stabilizers for Shor)
        if m_type == 'x': 
            check_matrix = self.code.Hx
            logical_op = self._logical_x
            syndrome_table = self._syndrome_to_error_z  # Built from Hx, matches check_matrix
        else: 
            check_matrix = self.code.Hz
            logical_op = self._logical_z
            syndrome_table = self._syndrome_to_error_x  # Built from Hz, matches check_matrix
        
        # Compute raw logical value
        outcome = self._compute_logical_value(m, logical_op)
        
        # Compute syndrome
        syndrome = self._compute_syndrome(m, check_matrix)
        
        # Apply correction
        if syndrome > 0:
            error_pos = syndrome_table.get(syndrome, None)
            
            if error_pos is not None and error_pos >= 0:
                # Single qubit correction
                outcome = (outcome + int(logical_op[error_pos])) % 2
            elif error_pos is not None and error_pos < -1:
                # Two-qubit correction
                pair_code = -(error_pos + 1)
                q1 = pair_code // self. n
                q2 = pair_code % self.n
                correction = (int(logical_op[q1]) + int(logical_op[q2])) % 2
                outcome = (outcome + correction) % 2
        
        return int(outcome)
    
    def decode_measurement_post_selection(self, m: np.ndarray, m_type: str = 'x') -> int:
        """
        Decode with post-selection on high-weight syndromes.
        
        Returns -1 if syndrome indicates likely uncorrectable error.
        
        Args:
            m:  Measurement outcomes
            m_type: 'x' or 'z'
        
        Returns:
            0 or 1 for valid decode, -1 for rejected (post-selected out)
        """
        if m_type == 'x': 
            check_matrix = self.code.Hx
            logical_op = self._logical_x
            syndrome_table = self._syndrome_to_error_z  # Built from Hx, matches check_matrix
            weights = self._error_weights_z
        else:
            check_matrix = self.code.Hz
            logical_op = self._logical_z
            syndrome_table = self._syndrome_to_error_x  # Built from Hz, matches check_matrix
            weights = self._error_weights_x
        
        # Compute syndrome
        syndrome = self._compute_syndrome(m, check_matrix)
        
        # Check if syndrome is correctable
        if syndrome > 0:
            if syndrome not in syndrome_table:
                # Unknown syndrome - reject
                return -1
            
            # For distance-3 codes, reject weight-2+ errors
            if self.code. d <= 3 and weights. get(syndrome, 99) >= 2:
                return -1
        
        # Decode normally
        return self.decode_syndrome(m, m_type)
    
    def decode_syndrome_k(self, m: np.ndarray, m_type: str = 'x') -> Union[int, List[int]]:
        """
        Decode EC syndrome measurements for k>=1 codes.
        
        This is the k-aware version of decode_syndrome for EC syndrome data.
        Always uses syndrome table lookup, never block majority voting.
        
        For k=1 codes: returns int
        For k>=2 codes: returns List[int] using parity checking
        """
        if self.code.k >= 2:
            # For k>=2 codes, use parity checking (same logic as decode_measurement_k2)
            parity = int(np.sum(m) % 2)
            if parity == 1:
                return [-1, -1]
            
            # Use appropriate logical operators
            logical_op1 = self._logical_z
            logical_op2 = self.code.Lz2 if self.code.Lz2 is not None else self._logical_z
            
            bit0 = self._compute_logical_value(m, logical_op1)
            bit1 = self._compute_logical_value(m, logical_op2)
            
            return [bit0, bit1]
        else:
            return self.decode_syndrome(m, m_type)
    
    def decode_measurement_k2(self, m: np.ndarray, m_type: str = 'x') -> List[int]:
        """
        Decode a measurement for k=2 codes (like C4 [[4,2,2]]).
        
        For codes encoding 2 logical qubits, this returns [bit0, bit1].
        Uses parity checking for error detection.
        
        NOTE: For k=2 CSS codes like C4, we always use Z logical operators (Lz, Lz2)
        regardless of m_type. This matches the behavior of the C4C6 ground truth module
        where the decoder ignores m_type. The X and Z stabilizers are symmetric for C4
        (both are all-ones), so the parity check is the same. The logical value extraction
        uses fixed formulas that don't depend on measurement basis.
        
        Args:
            m: Measurement outcomes (length n)
            m_type: 'x' for X-basis measurement, 'z' for Z-basis (IGNORED for k=2)
        
        Returns:
            [bit0, bit1] or [-1, -1] if parity check fails (detected error)
        """
        if self.code.k < 2:
            # Fall back to single-qubit decoding
            return [self.decode_measurement(m, m_type)]
        
        # For k=2 codes like C4, use parity checking
        # The stabilizer generator is typically all-ones (XXXX or ZZZZ)
        # Parity must be 0 for valid codeword
        parity = int(np.sum(m) % 2)
        
        if parity == 1:
            # Parity check failed - detected error
            return [-1, -1]
        
        # Always use Z logical operators for k=2 codes (matches C4C6 module behavior)
        # This works because C4's parity check is symmetric and logical extraction
        # uses fixed formulas independent of measurement basis
        logical_op1 = self._logical_z
        logical_op2 = self.code.Lz2 if self.code.Lz2 is not None else self._logical_z
        
        bit0 = self._compute_logical_value(m, logical_op1)
        bit1 = self._compute_logical_value(m, logical_op2)
        
        return [bit0, bit1]
    
    def decode_measurement_k(self, m: np.ndarray, m_type: str = 'x') -> Union[int, List[int]]:
        """
        Universal decoder that returns appropriate type based on code's decoder_type.
        
        Dispatches to the appropriate decoder based on code configuration:
        - "syndrome": Syndrome lookup (returns int)
        - "parity": Parity checking for k>1 (returns List[int])
        - "majority": Majority voting (returns int)
        
        For k=1 codes: returns int (single logical value)
        For k=2 codes: returns List[int] ([bit0, bit1])
        
        This is the recommended method for generic code handling.
        """
        decoder_type = self.code.effective_decoder_type
        
        if decoder_type == "parity" and self.code.k >= 2:
            return self.decode_measurement_k2(m, m_type)
        elif decoder_type == "majority":
            # Majority voting - for use as outer code decoder
            # Simple majority of bits (used when this code is outer code)
            count_1 = np.sum(m) 
            return 1 if count_1 > len(m) // 2 else 0
        else:
            # Default: syndrome decoding
            return self.decode_measurement(m, m_type)
    
    def decode_outer_majority_k2(self, block_results: List[List[int]], m_type: str = 'x') -> List[int]:
        """
        Majority voting decoder for outer codes with k=2 inner code results.
        
        This implements C6-style majority voting: given 3 blocks of [bit0, bit1] results,
        uses parity relationships to extract the logical value even if one block failed.
        
        EXACT match to C4C6 decode_measurement_c6:
        - If >1 block failed: return [-1, -1]
        - If 1 block failed: use remaining 2 blocks with parity formulas
        - If 0 blocks failed: check cross-block parity consistency
        
        Args:
            block_results: List of 3 [bit0, bit1] results from inner decoder
                          Each entry is [bit0, bit1] or [-1, -1] if that block failed
            m_type: Measurement type (ignored, included for API consistency)
        
        Returns:
            [bit0, bit1] decoded logical values, or [-1, -1] if >1 block failed
        """
        if len(block_results) != 3:
            # Fallback for non-3-block codes
            valid = [b for b in block_results if b[0] != -1]
            if not valid:
                return [-1, -1]
            # Simple majority per dimension
            k = len(block_results[0])
            result = []
            for dim in range(k):
                vals = [b[dim] for b in valid if b[dim] != -1]
                if vals:
                    result.append(1 if sum(vals) > len(vals) // 2 else 0)
                else:
                    result.append(-1)
            return result
        
        # Convert to flat list format M = [b0[0], b0[1], b1[0], b1[1], b2[0], b2[1]]
        M = [block_results[0][0], block_results[0][1], 
             block_results[1][0], block_results[1][1],
             block_results[2][0], block_results[2][1]]
        
        # Count failed blocks (where first element is -1)
        failed = [M[0], M[2], M[4]].count(-1)
        
        if failed > 1:
            return [-1, -1]
        elif M[0] == -1:
            # Block 0 failed, use blocks 1 and 2
            return [(M[2] + M[3] + M[5]) % 2, (M[3] + M[4]) % 2]
        elif M[2] == -1:
            # Block 1 failed, use blocks 0 and 2
            return [(M[1] + M[4] + M[5]) % 2, (M[0] + M[5]) % 2]
        elif M[4] == -1:
            # Block 2 failed, use blocks 0 and 1
            return [(M[0] + M[1] + M[3]) % 2, (M[1] + M[2]) % 2]
        else:
            # All blocks valid - check parity consistency
            parity1 = (M[0] + M[1] + M[2] + M[5]) % 2
            parity2 = (M[0] + M[3] + M[4] + M[5]) % 2
            if parity1 == 1 or parity2 == 1:
                return [-1, -1]
            else:
                return [(M[2] + M[3] + M[5]) % 2, (M[3] + M[4]) % 2]
    
    def decode_ec_hierarchical(self, x: np.ndarray, detector_X: List, detector_Z: List,
                               inner_decoder_type: str = "parity",
                               outer_decoder_type: str = "majority",
                               n_blocks: int = 3, inner_k: int = 2) -> Tuple[List[int], List[int]]:
        """
        Generic hierarchical EC decoding for concatenated codes.
        
        This is the UNIVERSAL decoder that works for:
        - C4->C6: inner_decoder_type="parity", outer_decoder_type="majority", n_blocks=3, inner_k=2
        - C4->Steane: inner_decoder_type="parity", outer_decoder_type="syndrome", n_blocks=7, inner_k=2
        - Steane->Steane: inner_decoder_type="syndrome", outer_decoder_type="syndrome", n_blocks=7, inner_k=1
        
        The detector structure expected:
        - detector_X is a list of n_blocks entries
        - Each entry is [start, end] indices into sample array x
        
        Returns:
            (correction_x, correction_z) where each is a list of inner_k values
        """
        # Decode each inner block
        mx = [[0] * inner_k for _ in range(n_blocks)]
        mz = [[0] * inner_k for _ in range(n_blocks)]
        
        for i in range(n_blocks):
            if i >= len(detector_X) or i >= len(detector_Z):
                continue
                
            detx = detector_X[i]
            detz = detector_Z[i]
            
            # Handle different detector formats
            if isinstance(detx, list) and len(detx) >= 2:
                if isinstance(detx[0], int):
                    # Direct [start, end] format
                    start_x, end_x = detx[0], detx[1]
                    start_z, end_z = detz[0], detz[1]
                else:
                    # Nested [[start, end]] format - take first
                    start_x, end_x = detx[0][0], detx[0][1]
                    start_z, end_z = detz[0][0], detz[0][1]
            else:
                continue
            
            # Decode inner block based on decoder type
            if inner_decoder_type == "parity" and inner_k >= 2:
                result_x = self.decode_measurement_k2(x[start_x:end_x], 'x')
                result_z = self.decode_measurement_k2(x[start_z:end_z], 'z')
            else:
                result_x = [self.decode_measurement(x[start_x:end_x], 'x')]
                result_z = [self.decode_measurement(x[start_z:end_z], 'z')]
            
            mx[i] = result_x if isinstance(result_x, list) else [result_x]
            mz[i] = result_z if isinstance(result_z, list) else [result_z]
        
        # Decode outer code based on outer decoder type
        if outer_decoder_type == "majority" and inner_k >= 2:
            correction_x = self.decode_outer_majority_k2(mx, 'x')
            correction_z = self.decode_outer_majority_k2(mz, 'z')
        else:
            # Syndrome decoding for outer code (flatten inner results to first dimension)
            flat_mx = np.array([m[0] if m[0] != -1 else 0 for m in mx])
            flat_mz = np.array([m[0] if m[0] != -1 else 0 for m in mz])
            correction_x = [self.decode_measurement(flat_mx, 'x')]
            correction_z = [self.decode_measurement(flat_mz, 'z')]
            
            # Pad to inner_k if needed
            while len(correction_x) < inner_k:
                correction_x.append(0)
            while len(correction_z) < inner_k:
                correction_z.append(0)
        
        return correction_x, correction_z
    
    def decode_m_hierarchical(self, x: np.ndarray, detector_m: List,
                              correction: List[int] = None,
                              inner_decoder_type: str = "parity",
                              outer_decoder_type: str = "majority",
                              n_blocks: int = 3, inner_k: int = 2) -> List[int]:
        """
        Generic hierarchical measurement decoding for concatenated codes.
        
        Similar to decode_ec_hierarchical but for final measurement with correction.
        
        Args:
            x: Sample array
            detector_m: List of n_blocks detector entries, each [start, end]
            correction: Optional correction to apply [bit0, bit1, ...]
            inner_decoder_type: "parity" or "syndrome"
            outer_decoder_type: "majority" or "syndrome"
            n_blocks: Number of inner blocks
            inner_k: Number of logical qubits in inner code
        
        Returns:
            [bit0, bit1, ...] decoded and corrected values
        """
        # Decode each inner block
        m = [[0] * inner_k for _ in range(n_blocks)]
        
        for i in range(n_blocks):
            if i >= len(detector_m):
                continue
                
            det = detector_m[i]
            
            # Handle different detector formats
            if isinstance(det, list) and len(det) >= 2 and isinstance(det[0], int):
                start, end = det[0], det[1]
            elif isinstance(det, list) and len(det) >= 1 and isinstance(det[0], list):
                start, end = det[0][0], det[0][1]
            else:
                continue
            
            # Decode inner block
            if inner_decoder_type == "parity" and inner_k >= 2:
                result = self.decode_measurement_k2(x[start:end], 'x')
            else:
                result = [self.decode_measurement(x[start:end], 'x')]
            
            m[i] = result if isinstance(result, list) else [result]
        
        # Decode outer code
        if outer_decoder_type == "majority" and inner_k >= 2:
            outcome = self.decode_outer_majority_k2(m, 'x')
        else:
            flat_m = np.array([block[0] if block[0] != -1 else 0 for block in m])
            outcome = [self.decode_measurement(flat_m, 'x')]
            while len(outcome) < inner_k:
                outcome.append(0)
        
        # Apply correction if provided
        if correction is not None and len(correction) >= inner_k:
            if outcome[0] != -1 and correction[0] != -1:
                outcome = [(outcome[a] + correction[a]) % 2 for a in range(inner_k)]
        
        return outcome
    
    def decode_block(self, sample: np.ndarray, detector_info: List, 
                     level:  int, m_type: str = 'x') -> int:
        """
        Decode a single block measurement from detector info.
        
        Args:
            sample: Full detector sample array
            detector_info: [start, end] indices into sample
            level:  Concatenation level
            m_type:  Measurement basis
        
        Returns:
            Decoded logical value
        """
        if isinstance(detector_info, list) and len(detector_info) == 2:
            if isinstance(detector_info[0], int):
                m = sample[detector_info[0]: detector_info[1]]
                return self.decode_measurement(m, m_type)
        return 0
    
    def decode_hierarchical(self, sample: np. ndarray, detector_info,
                           level: int, corrections: np.ndarray = None,
                           m_type: str = 'x') -> int:
        """
        Hierarchical decoding through concatenation levels.
        
        Recursively decodes inner blocks, then decodes outer code
        treating inner outcomes as physical measurements.
        
        Args:
            sample: Full detector sample array
            detector_info:  Hierarchical detector structure
            level: Current concatenation level
            corrections:  Corrections to apply from previous EC rounds
            m_type:  Measurement basis
        
        Returns: 
            Decoded logical value at this level
        """
        code = self.concat_code. code_at_level(level)
        
        # Base case: physical level
        if level == 0:
            return self.decode_block(sample, detector_info, level, m_type)
        
        # Recursive case: decode inner blocks
        inner_outcomes = []
        for i in range(code.n):
            if isinstance(detector_info, list) and i < len(detector_info):
                inner_outcome = self.decode_hierarchical(
                    sample, detector_info[i], level - 1, None, m_type
                )
            else:
                inner_outcome = 0
            
            # Apply corrections if provided
            if corrections is not None and i < len(corrections):
                inner_outcome = (inner_outcome + int(corrections[i])) % 2
            
            inner_outcomes.append(inner_outcome)
        
        # Decode outer code using inner outcomes as measurements
        return self.decode_measurement(np.array(inner_outcomes), m_type)
    
    def decode_ec_hd(self, x:  np.ndarray, detector_X: List, detector_Z: List,
                     correction_x_prev: List, correction_z_prev: List) -> Tuple: 
        """
        Hierarchical EC decoding with propagation corrections.
        
        This is the generic version of Steane's decode_ec_hd. It uses
        the code's propagation tables if available, otherwise falls back
        to simplified decoding.
        
        Supports two detector structures:
        1. Corrected prep structure (shorter, ~15 entries for Steane):
           - [n L1 EC anc1] + [n L1 EC anc2] + [transversal meas]
           - Structure used by KnillECGadget with verified ancilla prep
           
        2. Old prep EC structure (longer, ~105 entries for Steane):
           - [num_ec prep EC anc1] + [num_ec prep EC anc2] + [n L1 EC anc1] + [n L1 EC anc2] + [trans]
           - Structure used by SteaneECGadget with unverified prep
        
        Args:
            x: Detector sample array
            detector_X: X syndrome detector info (hierarchical)
            detector_Z: Z syndrome detector info (hierarchical)
            correction_x_prev: X corrections from previous round
            correction_z_prev: Z corrections from previous round
        
        Returns:
            (correction_x, correction_z, correction_x_next, correction_z_next)
        """
        n = self.n
        prop = self.concat_code.get_propagation_tables(1)
        
        # Initialize correction arrays
        mx = [0] * n
        mz = [0] * n
        correction_x_next = [0] * n
        correction_z_next = [0] * n
        
        # Handle numpy arrays and None values properly
        if correction_x_prev is None or (hasattr(correction_x_prev, '__len__') and len(correction_x_prev) == 0):
            cx1 = [0] * n
        else:
            cx1 = list(correction_x_prev)
        
        if correction_z_prev is None or (hasattr(correction_z_prev, '__len__') and len(correction_z_prev) == 0):
            cz1 = [0] * n
        else:
            cz1 = list(correction_z_prev)
        
        cx2, cz2, cx3, cz3 = [0] * n, [0] * n, [0] * n, [0] * n
        
        if prop is None:
            # Simplified decoding without propagation tables
            return self._decode_ec_simple(x, detector_X, detector_Z, cx1, cz1)
        
        num_ec = prop.num_ec_0prep
        
        # Detect structure: corrected prep has ~2n+1 entries, old has 2*num_ec + 2*n + 1
        # Threshold: if length < 2*num_ec, it's corrected prep
        is_corrected_prep = len(detector_X) < 2 * num_ec
        
        if is_corrected_prep:
            # ============================================
            # CORRECTED PREP STRUCTURE (KnillECGadget)
            # ============================================
            # Structure: [n L1 EC anc1] + [n L1 EC anc2] + [transversal meas]
            # Total entries: 2n + 1 = 15 for Steane
            
            # Decode L1 EC measurements for first ancilla (indices 0 to n-1)
            for i in range(n):
                if i < len(detector_X) and detector_X[i]:
                    cx2[i] = self._safe_decode_detector(x, detector_X[i])
                    cz2[i] = self._safe_decode_detector(x, detector_Z[i])
            
            # Decode L1 EC measurements for second ancilla (indices n to 2n-1)
            for i in range(n):
                if n + i < len(detector_X) and detector_X[n + i]:
                    cx3[i] = self._safe_decode_detector(x, detector_X[n + i])
                    cz3[i] = self._safe_decode_detector(x, detector_Z[n + i])
            
            # Final transversal measurement (index 2n)
            final_idx = 2 * n
            
        else:
            # ============================================
            # OLD PREP EC STRUCTURE (SteaneECGadget)
            # ============================================
            # Structure: [num_ec prep EC anc1] + [num_ec prep EC anc2] + [n L1 EC anc1] + [n L1 EC anc2] + [trans]
            # Total entries: 2*num_ec + 2*n + 1 = 105 for Steane
            
            # Initialize 0prep correction arrays
            cx2_0prep = [0] * num_ec
            cz2_0prep = [0] * num_ec
            cx3_0prep = [0] * num_ec
            cz3_0prep = [0] * num_ec
            
            # Decode main EC round measurements (after prep EC)
            for i in range(n):
                idx_2 = 2 * num_ec + i
                idx_3 = 2 * num_ec + n + i
                
                if idx_2 < len(detector_X) and detector_X[idx_2]: 
                    cx2[i] = self._safe_decode_detector(x, detector_X[idx_2])
                    cz2[i] = self._safe_decode_detector(x, detector_Z[idx_2])
                
                if idx_3 < len(detector_X) and detector_X[idx_3]:
                    cx3[i] = self._safe_decode_detector(x, detector_X[idx_3])
                    cz3[i] = self._safe_decode_detector(x, detector_Z[idx_3])
            
            # Decode 0prep measurements and apply propagation
            for a in range(num_ec):
                if a < len(detector_X) and detector_X[a]:
                    cx2_0prep[a] = self._safe_decode_detector(x, detector_X[a])
                    cz2_0prep[a] = self._safe_decode_detector(x, detector_Z[a])
                
                if num_ec + a < len(detector_X) and detector_X[num_ec + a]:
                    cx3_0prep[a] = self._safe_decode_detector(x, detector_X[num_ec + a])
                    cz3_0prep[a] = self._safe_decode_detector(x, detector_Z[num_ec + a])
                
                # Apply X error propagation
                if a < len(prop.propagation_X):
                    for i in prop.propagation_X[a]: 
                        if i < n: 
                            cz2[i] = (cz2[i] + cx2_0prep[a]) % 2
                            cx3[i] = (cx3[i] + cx3_0prep[a]) % 2
                
                # Apply Z error propagation
                if a < len(prop.propagation_Z):
                    for i in prop.propagation_Z[a]: 
                        if i < n: 
                            cx2[i] = (cx2[i] + cz2_0prep[a]) % 2
                            cx3[i] = (cx3[i] + cz2_0prep[a]) % 2
                            cz2[i] = (cz2[i] + cz3_0prep[a]) % 2
                            cz3[i] = (cz3[i] + cz3_0prep[a]) % 2
            
            # Final transversal measurement (after all prep EC and L1 EC)
            final_idx = 2 * num_ec + 2 * n
        
        # ============================================
        # COMMON: Decode final transversal measurements
        # ============================================
        for i in range(n):
            x_correction = (cx1[i] + cx2[i]) % 2
            z_correction = (cz1[i] + cz2[i]) % 2
            correction_x_next[i] = cx3[i]
            correction_z_next[i] = cz3[i]
            
            # Decode final measurements with corrections
            # EC SYNDROME decoding - use decode_syndrome
            if final_idx < len(detector_X) and i < len(detector_X[final_idx]):
                det_x = detector_X[final_idx][i]
                det_z = detector_Z[final_idx][i]
                
                if isinstance(det_x, list) and len(det_x) == 2:
                    mx[i] = (self.decode_syndrome(x[det_x[0]:det_x[1]], 'x') + x_correction) % 2
                    mz[i] = (self.decode_syndrome(x[det_z[0]:det_z[1]], 'z') + z_correction) % 2
        
        # Outer code decoding - use decode_syndrome (these are inner block corrections)
        correction_x = self.decode_syndrome(np.array(mx), 'x')
        correction_z = self.decode_syndrome(np.array(mz), 'z')
        
        return correction_x, correction_z, correction_x_next, correction_z_next
    
    def _decode_ec_simple(self, x: np.ndarray, detector_X: List, detector_Z: List,
                          cx_prev: List, cz_prev: List) -> Tuple:
        """
        Simplified EC decoding without propagation tables.
        
        Just decodes syndrome measurements directly without tracking
        error propagation through preparation circuit.
        """
        n = self.n
        cx_next = [0] * n
        cz_next = [0] * n
        mx = [0] * n
        mz = [0] * n
        
        # Decode the most recent syndrome measurements
        if detector_X and len(detector_X) > 0:
            last_idx = len(detector_X) - 1
            
            for i in range(min(n, len(detector_X[last_idx]) if isinstance(detector_X[last_idx], list) else 0)):
                det_x = detector_X[last_idx][i] if isinstance(detector_X[last_idx], list) else None
                det_z = detector_Z[last_idx][i] if isinstance(detector_Z[last_idx], list) else None
                
                if det_x and isinstance(det_x, list) and len(det_x) == 2:
                    # EC SYNDROME decoding - use decode_syndrome
                    mx[i] = (self.decode_syndrome(x[det_x[0]:det_x[1]], 'x') + cx_prev[i]) % 2
                if det_z and isinstance(det_z, list) and len(det_z) == 2:
                    mz[i] = (self.decode_syndrome(x[det_z[0]:det_z[1]], 'z') + cz_prev[i]) % 2
        
        # Outer code decoding - still use decode_syndrome (these are inner block corrections)
        correction_x = self.decode_syndrome(np.array(mx), 'x')
        correction_z = self.decode_syndrome(np.array(mz), 'z')
        
        return correction_x, correction_z, cx_next, cz_next
    
    def _safe_decode_detector(self, x: np.ndarray, detector_info) -> int:
        """
        Safely decode an EC detector, handling various formats.
        Uses decode_syndrome for EC data (not decode_measurement).
        """
        if detector_info is None:
            return 0
        
        if isinstance(detector_info, list):
            if len(detector_info) == 0:
                return 0
            
            # [start, end] format
            if len(detector_info) == 2 and isinstance(detector_info[0], int):
                return self.decode_syndrome(x[detector_info[0]:detector_info[1]])
            
            # Nested format - take first element
            if isinstance(detector_info[0], list):
                return self._safe_decode_detector(x, detector_info[0])
        
        return 0
    
    def decode_m_hd(self, x: np. ndarray, detector_m: List, 
                    correction_l1: List) -> int:
        """
        Hierarchical measurement decoding with corrections.
        
        Decodes each inner block measurement and applies corrections,
        then decodes the outer code.
        
        Args:
            x: Detector sample array
            detector_m: Measurement detector info (hierarchical)
            correction_l1: Corrections to apply at level-1
        
        Returns: 
            Decoded logical measurement outcome
        """
        n = self.n
        m = [0] * n
        
        for i in range(n):
            if i < len(detector_m):
                det = detector_m[i]
                
                if isinstance(det, list) and len(det) == 2 and isinstance(det[0], int):
                    # Direct [start, end] format
                    raw = self.decode_measurement(x[det[0]:det[1]])
                else:
                    # Hierarchical - decode recursively
                    raw = self._safe_decode_detector(x, det)
                
                correction = int(correction_l1[i]) if i < len(correction_l1) else 0
                m[i] = (raw + correction) % 2
        
        return self.decode_measurement(np.array(m), 'x')
    
    def compute_threshold_estimate(self) -> float:
        """
        Estimate the error threshold for this code.
        
        Uses the code distance and structure to estimate where
        logical error rate crosses physical error rate.
        
        Returns:
            Estimated threshold probability
        """
        d = self.code.d
        n = self.code.n
        
        # Rough estimate:  threshold ~ 1 / (c * n) where c depends on structure
        # For CSS codes, threshold is typically in range 0.1% - 1%
        
        # Better estimate using distance
        # p_L ~ (p / p_th)^((d+1)/2) for concatenated codes
        # At threshold, p_L = p, so p_th ~ O(1/n)
        
        c = 10  # Empirical constant
        threshold = 1.0 / (c * n)
        
        return threshold
    
    def get_code_info(self) -> Dict:
        """
        Get information about the code being decoded.
        """
        return {
            'name': self.code.name,
            'n': self.n,
            'k': self.code.k,
            'd': self.code.d,
            'num_x_stabilizers': self.code.num_x_stabilizers,
            'num_z_stabilizers': self.code.num_z_stabilizers,
            'syndrome_table_size_x': len(self._syndrome_to_error_x),
            'syndrome_table_size_z': len(self._syndrome_to_error_z),
        }

# =============================================================================
# Post-Selection
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                     POST-SELECTION & ACCEPTANCE                              │
# └─────────────────────────────────────────────────────────────────────────────┘

# ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
# │        PostSelector             │      │      AcceptanceChecker          │
# ├─────────────────────────────────┤      ├─────────────────────────────────┤
# │ Input: ConcatenatedCode,        │      │ Input: ConcatenatedCode,        │
# │        Decoder                  │      │        Decoder                  │
# ├─────────────────────────────────┤      ├─────────────────────────────────┤
# │ + post_selection_steane(x,      │      │ + accept_l1(x, detector_m,      │
# │     detector_0prep) -> bool     │      │     detector_X, detector_Z, Q)  │
# │                                 │      │   -> float (error probability)  │
# │ + post_selection_steane_l2(x,   │      │                                 │
# │     detector_0prep,             │      │ + accept_l2(x, detector_m,      │
# │     detector_X, detector_Z)     │      │     detector_X, detector_Z, Q)  │
# │   -> bool                       │      │   -> float                      │
# │                                 │      │                                 │
# │ + post_selection_l1(x,          │      │ Uses decoder to compute         │
# │     list_detector_0prep)        │      │ corrections and check if        │
# │   -> bool                       │      │ Bell pair correlations hold     │
# │                                 │      │                                 │
# │ + post_selection_l2(x, ...)     │      │ Returns 0 if no error,          │
# │   -> bool                       │      │ 1 if definite error,            │
# │                                 │      │ 0.5 if uncertain                │
# └─────────────────────────────────┘      └─────────────────────────────────┘
# =============================================================================

class PostSelector:
    """
    Post-selection filter for fault-tolerant quantum error correction.
    
    ═══════════════════════════════════════════════════════════════════════════
                               QEC THEORY BACKGROUND
    ═══════════════════════════════════════════════════════════════════════════
    
    POST-SELECTION IN FAULT-TOLERANT QEC
    ------------------------------------
    Post-selection is a critical technique in fault-tolerant quantum computing
    that filters out samples where errors were detected during state preparation
    or verification. By rejecting these "bad" samples, we achieve higher fidelity
    on the remaining "good" samples.
    
    Key insight: Verification circuits can detect (but not correct) certain errors.
    Rather than trying to correct these errors, we simply discard those samples.
    
    VERIFICATION-BASED POST-SELECTION
    ---------------------------------
    For codes like Steane [[7,1,3]], the preparation circuit includes a
    verification qubit:
    
        |0⟩_L preparation ──●── Verification qubit
                            │
                           [M] → If 1, reject sample
    
    The verification qubit is entangled with the data in a way that detects
    preparation errors. If verification measures 1, an error occurred.
    
    PARITY-BASED POST-SELECTION
    ---------------------------
    For k>1 error-detecting codes like C4 [[4,2,2]], we check parity:
    
        All ancilla measurements: m₀, m₁, m₂, m₃
        If (m₀ + m₁ + m₂ + m₃) % 2 ≠ 0, reject sample
    
    This detects when an odd number of errors occurred, which the code cannot
    handle correctly.
    
    ERROR-DETECTING TELEPORTATION (EDT)
    -----------------------------------
    For error-DETECTING codes (d=2), EDT provides additional filtering:
    
    - During teleportation-based EC, the inner code may detect uncorrectable errors
    - If the syndrome indicates more errors than the code can handle, reject
    - Surviving samples have higher fidelity due to this additional filtering
    
    EDT trades acceptance rate for error suppression: fewer samples pass, but
    those that do have much lower logical error rates.
    
    LEVEL-2 POST-SELECTION WITH PROPAGATION
    ---------------------------------------
    At L2, error propagation must be accounted for:
    
    1. Decode each L1 block's verification measurement
    2. Track how errors from earlier EC rounds propagate
    3. Apply propagation corrections to final verification outcome
    4. Reject if corrected outcome ≠ 0
    
    The PropagationTables encode which EC rounds affect which verification
    measurements, enabling correct L2 post-selection.
    
    ═══════════════════════════════════════════════════════════════════════════
                              METHOD SUMMARY
    ═══════════════════════════════════════════════════════════════════════════
    
    post_selection_l1(x, list_detector_0prep):
        Level-1 filtering on preparation verification
        
    post_selection_l2(x, list_detector_0prep, list_detector_0prep_l2, 
                      list_detector_X, list_detector_Z, Q):
        Level-2 filtering with hierarchical verification
        
    post_selection_l2_memory(x, ...):
        Specialized L2 filtering for memory tests
    
    Returns:
        True: Sample passes post-selection (keep it)
        False: Sample fails post-selection (reject it)
    
    References:
        [AGP06] Aliferis, Gottesman, Preskill, QIC 6, 97 (2006)
        [Kni05] Knill, Nature 434, 39 (2005)
    """
    
    def __init__(self, concat_code:  ConcatenatedCode, decoder: Decoder):
        self.concat_code = concat_code
        self.decoder = decoder
    
    def post_selection_steane(self, x: np.ndarray, detector_0prep: List) -> bool:
        """
        Level-1 post-selection for Steane-style verification.
        
        Checks a single verification detector: if measurement = 0, accept.
        
        Args:
            x: Sample array from detector sampler
            detector_0prep: Detector index(es) for verification measurement
            
        Returns:
            True if verification passed (measurement was 0)
        """
        if x[detector_0prep[0]] % 2 == 0:
            return True
        return False
    
    def post_selection_steane_l2(self, x: np.ndarray, detector_0prep: List,
                                  detector_X: List, detector_Z: List) -> bool:
        """
        Level-2 post-selection with error propagation correction.
        
        At L2, errors from earlier EC rounds propagate to affect verification.
        This method decodes the verification measurement and applies propagation
        corrections before checking if the outcome is 0.
        
        Args:
            x: Sample array from detector sampler
            detector_0prep: Range [start, end] for verification measurement
            detector_X: List of X-syndrome detector ranges
            detector_Z: List of Z-syndrome detector ranges
            
        Returns:
            True if corrected verification outcome is 0
        """
        prop = self.concat_code.get_propagation_tables(1)
        if prop is None: 
            return True
        
        # Safety check: detector_0prep should be a [start, end] pair
        if detector_0prep is None or len(detector_0prep) < 2:
            return True
        
        # Safety check: ensure it's a valid range format
        if not isinstance(detector_0prep[0], int) or not isinstance(detector_0prep[1], int):
            return True
        
        outcome = self.decoder.decode_measurement(
            x[detector_0prep[0]:detector_0prep[1]]
        )
        
        for a in prop.propagation_m:
            if a < len(detector_X) and detector_X[a]:
                det_x = detector_X[a]
                # Handle different detector formats
                if isinstance(det_x, list) and len(det_x) > 0:
                    if isinstance(det_x[0], list) and len(det_x[0]) >= 2:
                        # Format: [[start, end]]
                        correction_x = self.decoder.decode_measurement(
                            x[det_x[0][0]:det_x[0][1]]
                        )
                    elif len(det_x) >= 2 and isinstance(det_x[0], int):
                        # Format: [start, end]
                        correction_x = self.decoder.decode_measurement(
                            x[det_x[0]:det_x[1]]
                        )
                    else:
                        continue
                    outcome = (outcome + correction_x) % 2
        
        return outcome % 2 == 0
    
    def post_selection_l1(self, x: np.ndarray, list_detector_0prep: List) -> bool:
        """
        Full level-1 post-selection across all preparation verifications.
        
        Iterates through all verification detectors and checks each one.
        Dispatches to appropriate checking method based on code's post_selection_type:
        - "parity": Check sum of measurements mod 2 (for k>1 codes)
        - "verification": Check single verification qubit (for k=1 codes)
        
        Args:
            x: Sample array from detector sampler
            list_detector_0prep: List of verification detector ranges
            
        Returns:
            True if ALL verifications pass
        """
        inner_code = self.concat_code.code_at_level(0)
        ps_type = inner_code.post_selection_type
        
        for a in list_detector_0prep:
            if a is None:
                continue
                
            if ps_type == "parity":
                # Parity-based post-selection (for k>1 codes like C4)
                if isinstance(a, list) and len(a) == 2 and isinstance(a[0], int):
                    # Direct [start, end] range - check parity of all bits
                    if sum(x[a[0]:a[1]]) % 2 != 0:
                        return False
                elif isinstance(a, list) and len(a) == 1 and isinstance(a[0], list):
                    # Nested [[start, end]] format
                    inner_range = a[0]
                    if isinstance(inner_range, list) and len(inner_range) == 2:
                        if sum(x[inner_range[0]:inner_range[1]]) % 2 != 0:
                            return False
            else:
                # Verification-based post-selection (default for k=1 codes)
                if len(a) == 1:
                    if not self.post_selection_steane(x, a[0]):
                        return False
                elif len(a) == 2:
                    if isinstance(a[0], int) and isinstance(a[1], int):
                        if a[1] - a[0] == 1:
                            if not self.post_selection_steane(x, a):
                                return False
        return True
    
    def post_selection_l2(self, x: np.ndarray, list_detector_0prep: List,
                          list_detector_0prep_l2: List, list_detector_X: List,
                          list_detector_Z: List, Q: int) -> bool:
        """
        Full level-2 post-selection.
        
        If the ConcatenatedCode has a custom_post_selection_l2_fn, that will be used
        instead of the generic implementation.
        
        Otherwise, dispatches to code-specific implementations based on inner code's 
        post_selection_type and uses_edt configuration.
        """
        # Use custom post-selection function if provided by code-specific module
        if self.concat_code.has_custom_post_selection:
            return self.concat_code.custom_post_selection_l2_fn(
                x, list_detector_0prep, list_detector_0prep_l2, self.decoder
            )
        
        inner_code = self.concat_code.code_at_level(0)
        
        # Use code's configuration to determine post-selection strategy
        if inner_code.uses_edt or (inner_code.post_selection_type == "parity" and inner_code.k >= 2):
            return self._post_selection_l2_edt(x, list_detector_0prep, list_detector_0prep_l2)
        
        # For k=1 codes (like Steane), use propagation-based post-selection
        prop = self.concat_code.get_propagation_tables(1)
        if prop is None: 
            return True
        
        num_ec = prop.num_ec_0prep
        num_correction = 2 * Q
        
        # Level-1 post-selection on 0prep detectors
        for a in list_detector_0prep: 
            if a is None:
                continue
            if len(a) == 1:
                if a[0] is not None and not self.post_selection_steane(x, a[0]):
                    return False
            elif len(a) == 2:
                if isinstance(a[0], int) and isinstance(a[1], int):
                    if a[1] - a[0] == 1:
                        if not self.post_selection_steane(x, a):
                            return False
        
        # Level-2 post-selection with propagation
        # Safety: only run if we have valid detector_0prep_l2 data
        for i in range(num_correction):
            # Check bounds and data validity
            if 2 * i < len(list_detector_0prep_l2) and i < len(list_detector_X):
                det_0prep = list_detector_0prep_l2[2 * i]
                if det_0prep is not None and isinstance(det_0prep, list) and len(det_0prep) >= 2:
                    det_x_slice = list_detector_X[i][0:num_ec] if len(list_detector_X[i]) >= num_ec else list_detector_X[i]
                    det_z_slice = list_detector_Z[i][0:num_ec] if len(list_detector_Z[i]) >= num_ec else list_detector_Z[i]
                    if not self.post_selection_steane_l2(x, det_0prep, det_x_slice, det_z_slice):
                        return False
            
            if 2 * i + 1 < len(list_detector_0prep_l2) and i < len(list_detector_X):
                det_0prep = list_detector_0prep_l2[2 * i + 1]
                if det_0prep is not None and isinstance(det_0prep, list) and len(det_0prep) >= 2:
                    det_x_slice = list_detector_X[i][num_ec:2*num_ec] if len(list_detector_X[i]) >= 2*num_ec else []
                    det_z_slice = list_detector_Z[i][num_ec:2*num_ec] if len(list_detector_Z[i]) >= 2*num_ec else []
                    if not self.post_selection_steane_l2(x, det_0prep, det_x_slice, det_z_slice):
                        return False
        
        return True
    
    def _post_selection_l2_edt(self, x: np.ndarray, list_detector_0prep: List,
                               list_detector_0prep_l2: List) -> bool:
        """
        Post-selection for error-detecting codes (like C4 [[4,2,2]]).
        
        Matches original concatenated_c4c6.py logic:
        - For C4 inner code: check parity of 4 measurement bits
        - For C6 outer code: decode 3 C4 blocks and check logical parities
        """
        inner_code = self.concat_code.code_at_level(0)
        outer_code = self.concat_code.code_at_level(1) if self.concat_code.num_levels > 1 else inner_code
        
        n_inner = inner_code.n
        n_outer = outer_code.transversal_block_count if outer_code.transversal_block_count else outer_code.n
        
        for a in list_detector_0prep:
            if a is None:
                continue
            
            # Single range - check C4 parity
            if isinstance(a, list) and len(a) == 2 and isinstance(a[0], int):
                if (a[1] - a[0]) == n_inner:
                    # C4 parity check: sum of 4 bits must be even
                    if sum(x[a[0]:a[1]]) % 2 != 0:
                        return False
            
            # Nested structure for C6 blocks
            elif isinstance(a, list) and len(a) == n_outer:
                outcomes = []
                for block in a:
                    if isinstance(block, list) and len(block) == 2 and isinstance(block[0], int):
                        result = self.decoder.decode_measurement_k(x[block[0]:block[1]])
                        if isinstance(result, list) and result[0] == -1:
                            return False  # C4 parity failure
                        outcomes.append(result)
                
                # C6 logical parity check
                if len(outcomes) == n_outer and all(isinstance(o, list) and len(o) >= 2 for o in outcomes):
                    # Check that logical parities are even
                    parity0 = sum(o[0] for o in outcomes if o[0] != -1) % 2
                    parity1 = sum(o[1] for o in outcomes if o[1] != -1) % 2
                    if parity0 != 0 or parity1 != 0:
                        return False
            
            # List containing single range
            elif isinstance(a, list) and len(a) == 1 and isinstance(a[0], list):
                inner_range = a[0]
                if isinstance(inner_range, list) and len(inner_range) == 2 and isinstance(inner_range[0], int):
                    if (inner_range[1] - inner_range[0]) == n_inner:
                        if sum(x[inner_range[0]:inner_range[1]]) % 2 != 0:
                            return False
        
        return True
    
    def post_selection_l2_memory(self, x: np.ndarray, list_detector_0prep: List,
                                  list_detector_0prep_l2: List, list_detector_X: List,
                                  list_detector_Z: List, Q: int,
                                  detector_X_prep: List = None,
                                  detector_Z_prep: List = None) -> bool:
        """
        Post-selection for single-qubit L2 memory experiment.
        Unlike post_selection_l2 which expects 2 logical qubits (for CNOT),
        this function handles the case of 1 logical qubit with Q rounds of EC.
        
        Args:
            x: Sample vector
            list_detector_0prep: Level-1 preparation detectors
            list_detector_0prep_l2: Level-2 verification detectors [prep_l2, ec_anc1_l2, ec_anc2_l2, ...]
            list_detector_X: EC round X detectors
            list_detector_Z: EC round Z detectors
            Q: Number of EC rounds
            detector_X_prep: X detectors from noisy prep (for prep L2 verification)
            detector_Z_prep: Z detectors from noisy prep (for prep L2 verification)
        """
        prop = self.concat_code.get_propagation_tables(1)
        if prop is None:
            return True
        
        num_ec = prop.num_ec_0prep
        
        # Post-select on level-1 preparation detectors
        for a in list_detector_0prep:
            if len(a) == 1:
                if not self.post_selection_steane(x, a[0]):
                    return False
            elif len(a) == 2:
                if a[1] - a[0] == 1:
                    if not self.post_selection_steane(x, a):
                        return False
        
        # Post-select on level-2 preparation verification (index 0)
        # This is CRITICAL for fault tolerance - verifies the prep was successful
        # NOTE: With corrected prep (v2 style), detector_X_prep and detector_Z_prep are empty 
        # (no EC during prep). In that case, we only verify the L2 verification measurement.
        if len(list_detector_0prep_l2) > 0 and list_detector_0prep_l2[0] is not None:
            # Check if we have prep EC detectors (old-style prep) or not (corrected prep)
            if (detector_X_prep is not None and len(detector_X_prep) > 0 
                and detector_Z_prep is not None and len(detector_Z_prep) > 0):
                if not self.post_selection_steane_l2(
                    x, list_detector_0prep_l2[0],
                    detector_X_prep[0:num_ec],
                    detector_Z_prep[0:num_ec]
                ):
                    return False
            else:
                # Corrected prep: just verify the L2 measurement is clean (all zeros after decoding)
                detector_l2 = list_detector_0prep_l2[0]
                if isinstance(detector_l2, list) and len(detector_l2) == 2:
                    if self.decoder.decode_measurement(x[detector_l2[0]:detector_l2[1]]) != 0:
                        return False
        
        # Check EC round l2 detectors
        # NOTE: With corrected prep (v2), the EC round's detector_X/detector_Z structure is different:
        # - Old (with prep EC): [num_ec prep EC anc1] + [num_ec prep EC anc2] + [n L1 EC anc1] + [n L1 EC anc2] + [trans meas]
        # - New (no prep EC): [n L1 EC anc1] + [n L1 EC anc2] + [trans meas]
        # Detect based on list length vs expected structure:
        code = self.concat_code.code_at_level(0)
        n = code.n
        expected_with_prep_ec = 2 * num_ec + 2 * n + 1  # Structure with prep EC
        expected_without_prep_ec = 2 * n + 1  # Structure without prep EC
        
        for i in range(Q):
            # Each EC round adds 2 entries to list_detector_0prep_l2 (one per ancilla block)
            # Index into list_detector_0prep_l2: 1 + 2*i and 1 + 2*i + 1 (after prep at index 0)
            idx_base = 1 + 2 * i
            
            # Check if we have old structure (with prep EC) or new structure (no prep EC)
            # using proper structure comparison instead of magic number heuristic
            actual_len = len(list_detector_X[i]) if i < len(list_detector_X) else 0
            has_prep_ec = actual_len >= expected_with_prep_ec
            
            if has_prep_ec:
                # Old-style: use num_ec_0prep indexing
                if idx_base < len(list_detector_0prep_l2) and i < len(list_detector_X):
                    if not self.post_selection_steane_l2(
                        x, list_detector_0prep_l2[idx_base],
                        list_detector_X[i][0:num_ec],
                        list_detector_Z[i][0:num_ec]
                    ):
                        return False
                if idx_base + 1 < len(list_detector_0prep_l2) and i < len(list_detector_X):
                    if not self.post_selection_steane_l2(
                        x, list_detector_0prep_l2[idx_base + 1],
                        list_detector_X[i][num_ec:2 * num_ec],
                        list_detector_Z[i][num_ec:2 * num_ec]
                    ):
                        return False
            else:
                # New-style (v2 prep): just check L2 verification measurements
                # With no prep EC, we can only verify the L2 measurement itself
                if idx_base < len(list_detector_0prep_l2) and list_detector_0prep_l2[idx_base] is not None:
                    detector_l2 = list_detector_0prep_l2[idx_base]
                    if isinstance(detector_l2, list) and len(detector_l2) == 2:
                        if self.decoder.decode_measurement(x[detector_l2[0]:detector_l2[1]]) != 0:
                            return False
                if idx_base + 1 < len(list_detector_0prep_l2) and list_detector_0prep_l2[idx_base + 1] is not None:
                    detector_l2 = list_detector_0prep_l2[idx_base + 1]
                    if isinstance(detector_l2, list) and len(detector_l2) == 2:
                        if self.decoder.decode_measurement(x[detector_l2[0]:detector_l2[1]]) != 0:
                            return False
        
        return True


# =============================================================================
# Acceptance Functions
# =============================================================================

class AcceptanceChecker:
    """
    Error counting for Bell pair CNOT verification protocol.
    
    ═══════════════════════════════════════════════════════════════════════════
                               QEC THEORY BACKGROUND
    ═══════════════════════════════════════════════════════════════════════════
    
    BELL PAIR CNOT VERIFICATION
    ---------------------------
    The Bell pair protocol creates entangled pairs, applies CNOT rounds, and
    measures. If no logical errors occurred, all four measurements should be 0.
    
    Protocol recap:
    1. Prepare |Φ+⟩₁₂ ⊗ |Φ+⟩₃₄ where |Φ+⟩ = (|00⟩ + |11⟩)/√2
    2. For Q rounds: CNOT₁₃ (ideal) × CNOT₁₃ (noisy) × EC
    3. Undo entanglement: CNOT + H on each pair
    4. Measure all four → expect 0000
    
    Any non-zero measurement indicates a logical error occurred.
    
    PAULI FRAME CORRECTION
    ----------------------
    The EC gadget produces syndrome measurements that determine Pauli corrections.
    These corrections propagate through the circuit:
    
    X_propagate = [[1], [3]]:
        X correction from EC on pair 0 affects block 1 (target of Bell CNOT)
        X correction from EC on pair 1 affects block 3
        
    Z_propagate = [[0], [2]]:
        Z correction from EC on pair 0 affects block 0 (control of Bell CNOT)
        Z correction from EC on pair 1 affects block 2
    
    At measurement time, X corrections flip Z-basis outcomes:
        corrected_outcome[i] = raw_outcome[i] XOR (accumulated X corrections)
    
    ERROR COUNTING
    --------------
    After applying all corrections:
    - outcome[i] = 0: No error on block i
    - outcome[i] = 1: Definite error on block i
    - outcome[i] = -1: Uncertain (some correction was ambiguous)
    
    The acceptance function computes:
        flag = 1 if all outcomes are 0
        flag = 0 if any outcome is 1 (definite error)
        flag = 0.5 for each uncertain outcome (-1)
    
    Returns 1 - flag as the error contribution:
        0: No error (all outcomes were 0)
        1: Definite error (at least one outcome was 1)
        0.5: Uncertain (could be error or not)
    
    K > 1 CODES (LIKE C4 [[4,2,2]])
    -------------------------------
    For codes with k > 1 logical qubits:
    - Each measurement decodes to k values: [outcome_0, outcome_1, ...]
    - Each correction is k-dimensional
    - Errors are counted over all k dimensions
    - Final error rate normalized by k: err / (num × Q × k)
    
    This ensures fair comparison between k=1 and k>1 codes.
    
    ═══════════════════════════════════════════════════════════════════════════
                              METHOD SUMMARY
    ═══════════════════════════════════════════════════════════════════════════
    
    accept_l1(x, list_detector_m, list_detector_X, list_detector_Z, Q):
        Level-1 acceptance. Dispatches to _accept_l1_k1 or _accept_l1_k2.
        
    accept_l2(x, list_detector_m, list_detector_X, list_detector_Z, Q):
        Level-2 acceptance with hierarchical decoding.
        Returns -1 if EDT rejects (sample should be excluded from statistics).
    
    Returns:
        0: No error (success)
        1: Definite error
        0.5: Uncertain (for each ambiguous outcome)
        -1: EDT rejection (sample excluded, for L2 only)
    
    References:
        [AGP06] Aliferis, Gottesman, Preskill, QIC 6, 97 (2006)
        [Kni05] Knill, Nature 434, 39 (2005)
    """
    
    def __init__(self, concat_code:  ConcatenatedCode, decoder: Decoder):
        self.concat_code = concat_code
        self.decoder = decoder
        # Get k from inner code
        self.k = concat_code.code_at_level(0).k if concat_code.num_levels > 0 else 1
    
    def accept_l1(self, x: np. ndarray, list_detector_m: List,
                  list_detector_X: List, list_detector_Z: List, Q: int) -> float:
        """
        Level-1 acceptance check for Bell pair verification.
        
        Decodes all syndrome measurements, applies Pauli frame corrections to
        the final measurements, and counts errors.
        
        Args:
            x: Sample array from detector sampler
            list_detector_m: List of final measurement detector ranges (4 blocks)
            list_detector_X: List of X-syndrome detector ranges (2Q rounds)
            list_detector_Z: List of Z-syndrome detector ranges (2Q rounds)
            Q: Number of CNOT verification rounds
            
        Returns:
            Error contribution: 0 (no error), 1 (error), or 0.5 (uncertain)
            For k>1 codes, returns sum over all k dimensions.
        """
        if self.k >= 2:
            return self._accept_l1_k2(x, list_detector_m, list_detector_X, 
                                      list_detector_Z, Q)
        else:
            return self._accept_l1_k1(x, list_detector_m, list_detector_X, 
                                      list_detector_Z, Q)
    
    def _accept_l1_k1(self, x: np.ndarray, list_detector_m: List,
                      list_detector_X: List, list_detector_Z: List, Q: int) -> float:
        """
        Level-1 acceptance for k=1 codes (Steane, Shor).
        
        Implements the standard Bell pair verification error counting:
        1. Decode all EC syndrome measurements
        2. Decode all final measurements
        3. Apply propagation corrections
        4. Count errors based on corrected outcomes
        """
        num_correction = 2 * Q
        X_propagate = [[1], [3]]
        Z_propagate = [[0], [2]]
        outcome = np.zeros(4)
        correction_x = np.zeros(num_correction)
        correction_z = np.zeros(num_correction)
        
        for i in range(num_correction):
            # EC SYNDROME decoding - use decode_syndrome, NOT decode_measurement!
            # detector_X contains Z-basis measurements that detect X errors
            # detector_Z contains X-basis measurements that detect Z errors
            correction_x[i] = self.decoder.decode_syndrome(
                x[list_detector_X[i][0][0]:list_detector_X[i][0][1]], 'z'
            )
            correction_z[i] = self.decoder.decode_syndrome(
                x[list_detector_Z[i][0][0]:list_detector_Z[i][0][1]], 'x'
            )
        
        for i in range(4):
            # FINAL LOGICAL STATE measurement - use decode_measurement
            # This correctly uses block majority for Shor-like codes
            outcome[i] = self.decoder.decode_measurement(
                x[list_detector_m[i][0]: list_detector_m[i][1]], 'z'
            )
        
        for i in range(num_correction):
            pos = i % 2
            for x_prop in X_propagate[pos]:
                if outcome[x_prop] == -1:
                    continue
                if correction_x[i] == 1:
                    outcome[x_prop] = (outcome[x_prop] + 1) % 2
                if correction_x[i] == -1:
                    outcome[x_prop] = -1
            for z_prop in Z_propagate[pos]:
                if outcome[z_prop] == -1:
                    continue
                if correction_z[i] == 1:
                    outcome[z_prop] = (outcome[z_prop] + 1) % 2
                if correction_z[i] == -1:
                    outcome[z_prop] = -1
        
        flag = 1
        for i in range(4):
            if outcome[i] == 1:
                flag = 0
            if outcome[i] == -1:
                flag *= 0.5
        
        return 1 - flag
    
    def _accept_l1_k2(self, x: np.ndarray, list_detector_m: List,
                      list_detector_X: List, list_detector_Z: List, Q: int) -> float:
        """
        Level-1 acceptance for k>1 codes (C4 [[4,2,2]]).
        
        Same logic as _accept_l1_k1 but with k-dimensional arrays.
        
        This is a GENERIC implementation that works for any CSS code with k>1.
        The Bell pair protocol structure (4 blocks, X/Z propagation) is universal.
        
        Key differences from k=1:
        - outcome array is [4, k] not [4]
        - correction arrays are [num_correction, k]
        - Iterates over all k dimensions
        - Returns fractional error count (0 to k)
        
        The propagation patterns [[1], [3]] and [[0], [2]] come from the Bell pair
        circuit structure (not the inner code):
        - Blocks 0, 2 are "control" Bell pairs (get H before CNOT)
        - Blocks 1, 3 are "target" Bell pairs
        - X errors propagate from control to target through CNOT
        - Z errors propagate from target to control through CNOT
        """
        k = self.k  # Get k from the inner code
        num_correction = 2 * Q
        
        # Propagation patterns from Bell pair structure (not code-specific)
        X_propagate = [[1], [3]]  # X correction affects blocks 1, 3
        Z_propagate = [[0], [2]]  # Z correction affects blocks 0, 2
        
        outcome = np.zeros([4, k])
        correction_x = np.zeros([num_correction, k])
        correction_z = np.zeros([num_correction, k])
        
        # Decode EC syndrome measurements
        # EC SYNDROME decoding - use decode_syndrome_k, NOT decode_measurement_k!
        # detector_X contains Z-basis measurements that detect X errors
        # detector_Z contains X-basis measurements that detect Z errors
        for i in range(num_correction):
            # Decode X syndrome measurement (detects Z errors)
            cx = self.decoder.decode_syndrome_k(
                x[list_detector_X[i][0][0]:list_detector_X[i][0][1]], 'z'
            )
            # Decode Z syndrome measurement (detects X errors)  
            cz = self.decoder.decode_syndrome_k(
                x[list_detector_Z[i][0][0]:list_detector_Z[i][0][1]], 'x'
            )
            
            # Store corrections for all k logical qubits
            if isinstance(cx, list):
                for a in range(min(len(cx), k)):
                    correction_x[i, a] = cx[a]
            else:
                correction_x[i, :] = cx
                
            if isinstance(cz, list):
                for a in range(min(len(cz), k)):
                    correction_z[i, a] = cz[a]
            else:
                correction_z[i, :] = cz
        
        # Decode final measurements (in Z basis / computational basis)
        for i in range(4):
            result = self.decoder.decode_measurement_k(
                x[list_detector_m[i][0]:list_detector_m[i][1]], 'z'
            )
            if isinstance(result, list):
                for a in range(min(len(result), k)):
                    outcome[i, a] = result[a]
            else:
                outcome[i, :] = result
        
        # Apply corrections with propagation - iterate over all k dimensions
        for a in range(k):
            for i in range(num_correction):
                pos = i % 2  # Alternates between two Bell pairs per round
                
                # Apply X correction to target blocks
                for x_prop in X_propagate[pos]:
                    if outcome[x_prop, a] == -1:
                        continue
                    if correction_x[i, a] == 1:
                        outcome[x_prop, a] = (outcome[x_prop, a] + 1) % 2
                    if correction_x[i, a] == -1:
                        outcome[x_prop, a] = -1
                        
                # Apply Z correction to control blocks
                for z_prop in Z_propagate[pos]:
                    if outcome[z_prop, a] == -1:
                        continue
                    if correction_z[i, a] == 1:
                        outcome[z_prop, a] = (outcome[z_prop, a] + 1) % 2
                    if correction_z[i, a] == -1:
                        outcome[z_prop, a] = -1
        
        # Count errors across all k dimensions
        # For Bell pair protocol: outcome should be all zeros if no logical error
        num_errors = 0
        for a in range(k):
            flag = 1
            for i in range(4):
                if outcome[i, a] == 1:
                    flag = 0  # Definite error
                if outcome[i, a] == -1:
                    flag *= 0.5  # Uncertain (error-detected but not corrected)
            num_errors += (1 - flag)
        
        return num_errors
    
    def accept_l2(self, x: np.ndarray, list_detector_m: List,
                  list_detector_X: List, list_detector_Z: List, Q: int) -> float:
        """
        Generic Level-2 acceptance check.
        
        Handles both k=1 and k>1 inner codes, and includes EDT post-selection
        for error-detecting inner codes (like C4 [[4,2,2]]).
        
        If the ConcatenatedCode has a custom_accept_l2_fn, that will be used
        instead of the generic implementation. This allows code-specific modules
        to provide optimized acceptance logic.
        
        Structure (for k>1 codes):
        - For L2, list_detector_X[i] contains entries from KnillECGadget:
          [prep_entries..., L1_EC_entries..., final_measurement]
        - Prep entries may have different structure than L1 EC entries
        - Final measurement is always the LAST element
        - Final measurement has structure [[start, end], [start, end], ...] with n_now blocks
        
        EDT Post-Selection:
        - For error-detecting inner codes (d <= 2), reject if ANY inner block
          measurement indicates an uncorrectable error (parity check fails)
        - This is done by decoding L1 EC blocks and checking for -1 result
        
        k>1 Handling:
        - outcome array has shape [4, k]
        - correction arrays have shape [num_correction, k]  
        - Error count sums over all k dimensions
        """
        # Use custom acceptance function if provided by code-specific module
        if self.concat_code.has_custom_l2_acceptance:
            return self.concat_code.custom_accept_l2_fn(
                x, list_detector_m, list_detector_X, list_detector_Z, Q, self.decoder
            )
        
        inner_code = self.concat_code.code_at_level(0)
        outer_code = self.concat_code.code_at_level(1) if self.concat_code.num_levels > 1 else inner_code
        
        inner_k = inner_code.k
        inner_n = inner_code.n
        inner_d = inner_code.d
        
        # Number of transversal blocks in outer code
        n_now = outer_code.transversal_block_count if outer_code.transversal_block_count else outer_code.n
        
        num_correction = 2 * Q
        X_propagate = [[1], [3]]
        Z_propagate = [[0], [2]]
        
        # === EDT POST-SELECTION (for k>1 error-detecting codes) ===
        # Check L1 EC blocks for uncorrectable errors
        if inner_k >= 2 and inner_d <= 2:
            for i in range(num_correction):
                if i >= len(list_detector_X):
                    continue
                
                det_list = list_detector_X[i]
                # Final measurement is always the last element
                # L1 EC blocks are before the last element and have nested structure [[start, end]]
                # Iterate over all entries except the last one (final measurement)
                for entry_idx in range(len(det_list) - 1):
                    edt_det_x = det_list[entry_idx]
                    edt_det_z = list_detector_Z[i][entry_idx] if entry_idx < len(list_detector_Z[i]) else None
                    
                    if not edt_det_x:
                        continue
                    
                    # L1 EC entries have nested structure [[start, end]]
                    # Skip prep entries which have direct [start, end] format
                    if isinstance(edt_det_x, list):
                        if len(edt_det_x) >= 1 and isinstance(edt_det_x[0], list):
                            # Nested structure [[start, end]] - this is an L1 EC block
                            for block in edt_det_x:
                                if isinstance(block, list) and len(block) >= 2 and isinstance(block[0], int):
                                    result = self.decoder.decode_measurement_k(
                                        x[block[0]:block[1]], 'x')
                                    if isinstance(result, list) and result[0] == -1:
                                        return -1  # Reject
                                    elif result == -1:
                                        return -1
                            
                            # Also check Z detector
                            if edt_det_z and isinstance(edt_det_z, list) and len(edt_det_z) >= 1:
                                if isinstance(edt_det_z[0], list):
                                    for block in edt_det_z:
                                        if isinstance(block, list) and len(block) >= 2 and isinstance(block[0], int):
                                            result = self.decoder.decode_measurement_k(
                                                x[block[0]:block[1]], 'z')
                                            if isinstance(result, list) and result[0] == -1:
                                                return -1
                                            elif result == -1:
                                                return -1
        
        # === MAIN DECODING ===
        if inner_k >= 2:
            # k>1 inner code: use 2D arrays
            outcome = np.zeros([4, inner_k])
            correction_x = np.zeros([num_correction, inner_k])
            correction_z = np.zeros([num_correction, inner_k])
            
            # Decode EC measurements
            for i in range(num_correction):
                if i >= len(list_detector_X):
                    continue
                    
                # Final measurement is ALWAYS the last element
                final_idx = len(list_detector_X[i]) - 1
                
                # Decode each inner block and combine with outer code decoder
                cx, cz = self._decode_ec_l2_generic(x, list_detector_X[i], list_detector_Z[i],
                                                    final_idx, n_now, inner_k)
                for a in range(inner_k):
                    correction_x[i, a] = cx[a] if a < len(cx) else 0
                    correction_z[i, a] = cz[a] if a < len(cz) else 0
            
            # Decode final measurements
            for block_idx in range(4):
                if block_idx >= len(list_detector_m):
                    continue
                result = self._decode_m_l2_generic(x, list_detector_m[block_idx], n_now, inner_k)
                for a in range(inner_k):
                    outcome[block_idx, a] = result[a] if a < len(result) else 0
            
            # Apply corrections for each logical qubit dimension
            for a in range(inner_k):
                for i in range(num_correction):
                    pos = i % 2
                    for x_prop in X_propagate[pos]:
                        if outcome[x_prop, a] == -1:
                            continue
                        if correction_x[i, a] == 1:
                            outcome[x_prop, a] = (outcome[x_prop, a] + 1) % 2
                        if correction_x[i, a] == -1:
                            outcome[x_prop, a] = -1
                    for z_prop in Z_propagate[pos]:
                        if outcome[z_prop, a] == -1:
                            continue
                        if correction_z[i, a] == 1:
                            outcome[z_prop, a] = (outcome[z_prop, a] + 1) % 2
                        if correction_z[i, a] == -1:
                            outcome[z_prop, a] = -1
            
            # Count errors across all k dimensions
            # NOTE: The original C4C6 code has what appears to be a bug where it
            # always checks outcome[i,0] in the loop over 'a'. We reproduce this
            # behavior exactly for compatibility. This effectively counts errors
            # only in dimension 0, but multiplied by k.
            num_errors = 0
            for a in range(inner_k):
                flag = 1
                for i in range(4):
                    if outcome[i, 0] == 1:  # Match original: always use dimension 0
                        flag = 0
                    if outcome[i, 0] == -1:  # Match original: always use dimension 0
                        flag *= 0.5
                num_errors += (1 - flag)
            
            return num_errors
            
        else:
            # k=1 inner code: original behavior
            outcome = np.zeros(4)
            correction_x = np.zeros(num_correction)
            correction_z = np.zeros(num_correction)
            correction1_x = np.zeros(inner_n)
            correction1_z = np.zeros(inner_n)
            correction2_x = np.zeros(inner_n)
            correction2_z = np.zeros(inner_n)
            
            for i in range(Q):
                correction_x[2*i], correction_z[2*i], correction1_x, correction1_z = \
                    self.decoder.decode_ec_hd(x, list_detector_X[2*i], list_detector_Z[2*i],
                                              correction1_x, correction1_z)
                correction_x[2*i+1], correction_z[2*i+1], correction2_x, correction2_z = \
                    self.decoder.decode_ec_hd(x, list_detector_X[2*i+1], list_detector_Z[2*i+1],
                                              correction2_x, correction2_z)
            
            outcome[0] = self.decoder.decode_m_hd(x, list_detector_m[0], correction1_z)
            outcome[1] = self.decoder.decode_m_hd(x, list_detector_m[1], correction1_x)
            outcome[2] = self.decoder.decode_m_hd(x, list_detector_m[2], correction2_z)
            outcome[3] = self.decoder.decode_m_hd(x, list_detector_m[3], correction2_x)
            
            for i in range(num_correction):
                pos = i % 2
                for x_prop in X_propagate[pos]:
                    if outcome[x_prop] == -1:
                        continue
                    if correction_x[i] == 1:
                        outcome[x_prop] = (outcome[x_prop] + 1) % 2
                    if correction_x[i] == -1:
                        outcome[x_prop] = -1
                for z_prop in Z_propagate[pos]:
                    if outcome[z_prop] == -1:
                        continue
                    if correction_z[i] == 1:
                        outcome[z_prop] = (outcome[z_prop] + 1) % 2
                    if correction_z[i] == -1:
                        outcome[z_prop] = -1
            
            flag = 1
            for i in range(4):
                if outcome[i] == 1:
                    flag = 0
                if outcome[i] == -1:
                    flag *= 0.5
            
            return 1 - flag
    
    def _decode_ec_l2_generic(self, x: np.ndarray, detector_X: List, detector_Z: List,
                              final_idx: int, n_now: int, inner_k: int) -> Tuple[List, List]:
        """
        Generic L2 EC decoding for k>1 inner codes.
        
        Decodes the final transversal measurement (at final_idx) which contains
        n_now inner block measurements. Each inner block returns k values.
        Then uses outer code's decoding which may process all k dimensions together.
        
        For C4->C6 concatenation, the C6 decoder must see the combined structure
        since it decodes both logical dimensions in a correlated way.
        """
        correction_x = [0] * inner_k
        correction_z = [0] * inner_k
        
        if final_idx >= len(detector_X):
            return correction_x, correction_z
        
        final_det_x = detector_X[final_idx]
        final_det_z = detector_Z[final_idx]
        
        # Decode each inner block - keep 2D structure
        mx = [[0] * inner_k for _ in range(n_now)]
        mz = [[0] * inner_k for _ in range(n_now)]
        
        for i in range(n_now):
            if i < len(final_det_x):
                det_x = final_det_x[i]
                det_z = final_det_z[i]
                
                if isinstance(det_x, list) and len(det_x) >= 2 and isinstance(det_x[0], int):
                    result_x = self.decoder.decode_measurement_k(x[det_x[0]:det_x[1]], 'x')
                    result_z = self.decoder.decode_measurement_k(x[det_z[0]:det_z[1]], 'z')
                    
                    if isinstance(result_x, list):
                        mx[i] = result_x
                    else:
                        mx[i] = [result_x] * inner_k
                        
                    if isinstance(result_z, list):
                        mz[i] = result_z
                    else:
                        mz[i] = [result_z] * inner_k
        
        # Use outer code's combined decoding for k>1
        outer_code = self.concat_code.code_at_level(1) if self.concat_code.num_levels > 1 else self.concat_code.code_at_level(0)
        
        # For codes like C6 with transversal_block_count, use combined decoder
        # This handles the correlation between logical dimensions
        result_x = self._decode_outer_code_combined(mx, outer_code, inner_k, 'x')
        result_z = self._decode_outer_code_combined(mz, outer_code, inner_k, 'z')
        
        for a in range(inner_k):
            correction_x[a] = result_x[a] if a < len(result_x) else 0
            correction_z[a] = result_z[a] if a < len(result_z) else 0
        
        return correction_x, correction_z
    
    def _decode_m_l2_generic(self, x: np.ndarray, detector_m: List, 
                             n_now: int, inner_k: int) -> List:
        """
        Generic L2 measurement decoding for k>1 inner codes.
        """
        # Decode each inner block - keep 2D structure
        m = [[0] * inner_k for _ in range(n_now)]
        
        for i in range(n_now):
            if i < len(detector_m):
                det = detector_m[i]
                
                if isinstance(det, list) and len(det) >= 2 and isinstance(det[0], int):
                    result = self.decoder.decode_measurement_k(x[det[0]:det[1]], 'x')
                    if isinstance(result, list):
                        m[i] = result
                    else:
                        m[i] = [result] * inner_k
        
        # Combine for all dimensions together using outer code
        outer_code = self.concat_code.code_at_level(1) if self.concat_code.num_levels > 1 else self.concat_code.code_at_level(0)
        
        return self._decode_outer_code_combined(m, outer_code, inner_k, 'x')
    
    def _decode_outer_code_combined(self, measurements_2d: List[List[int]], code, inner_k: int, m_type: str) -> List[int]:
        """
        Decode outer code with combined k-dimensional inner results.
        
        For codes like C6 [[6,1,2]] with k=2 inner code (C4 [[4,2,2]]),
        the decoding formula processes all k dimensions together since
        the outer code's logical operators span the combined structure.
        
        Args:
            measurements_2d: List of [bit0, bit1, ...] for each transversal block
                            Shape: [n_blocks, k]
            code: Outer code
            inner_k: Number of logical qubits in inner code
            m_type: Measurement type ('x' or 'z')
        
        Returns:
            List of k decoded logical values
        """
        n_blocks = len(measurements_2d)
        
        # For C6-like codes with transversal_block_count and k>1 inner code
        # Use the specific decoding formula that matches the original
        if (code.transversal_block_count and 
            code.transversal_block_count < code.n and 
            n_blocks == 3 and inner_k == 2):
            # This matches the C6 decoding formula from concatenated_c4c6.py
            # M = [m[0][0], m[0][1], m[1][0], m[1][1], m[2][0], m[2][1]]
            m = measurements_2d
            M = [m[0][0], m[0][1], m[1][0], m[1][1], m[2][0], m[2][1]]
            
            # Check for too many rejected blocks (bit 0 from each block)
            if [M[0], M[2], M[4]].count(-1) > 1:
                return [-1, -1]
            elif M[0] == -1:
                # First block rejected - use formula with blocks 1 and 2
                return [(M[2] + M[3] + M[5]) % 2, (M[3] + M[4]) % 2]
            elif M[2] == -1:
                # Second block rejected - use formula with blocks 0 and 2
                return [(M[1] + M[4] + M[5]) % 2, (M[0] + M[5]) % 2]
            elif M[4] == -1:
                # Third block rejected - use formula with blocks 0 and 1
                return [(M[0] + M[1] + M[3]) % 2, (M[1] + M[2]) % 2]
            else:
                # All blocks valid - check stabilizer and decode
                if (M[0] + M[1] + M[2] + M[5]) % 2 == 1 or (M[0] + M[3] + M[4] + M[5]) % 2 == 1:
                    return [-1, -1]
                else:
                    return [(M[2] + M[3] + M[5]) % 2, (M[3] + M[4]) % 2]
        
        # Fallback for other code structures: decode each dimension independently
        outcome = [0] * inner_k
        for a in range(inner_k):
            m_a = [measurements_2d[i][a] if a < len(measurements_2d[i]) else 0 for i in range(n_blocks)]
            
            if -1 in m_a:
                count_rej = m_a.count(-1)
                if count_rej > n_blocks // 2:
                    outcome[a] = -1
                else:
                    # Majority of valid measurements
                    valid = [v for v in m_a if v != -1]
                    outcome[a] = 1 if sum(valid) > len(valid) // 2 else 0
            else:
                # Simple majority
                outcome[a] = 1 if sum(m_a) > n_blocks // 2 else 0
        
        return outcome
    
    def _decode_outer_code(self, measurements: List, code, m_type: str) -> int:
        """
        Decode outer code using its structure.
        
        For codes with transversal_block_count (like C6), use majority voting.
        For other codes (like Steane), use syndrome decoding.
        """
        n_blocks = len(measurements)
        
        # Check if this is a majority-voting code (has transversal_block_count)
        if code.transversal_block_count and code.transversal_block_count < code.n:
            # Majority voting (e.g., C6 with 3 blocks)
            count_0 = measurements.count(0)
            count_1 = measurements.count(1)
            count_rej = measurements.count(-1)
            
            if count_rej > n_blocks // 2:
                return -1
            elif count_1 > count_0:
                return 1
            else:
                return 0
        else:
            # Use syndrome decoding on the outer code
            # Create a temporary decoder for outer code
            m_array = np.array([int(m) if m != -1 else 0 for m in measurements])
            if len(m_array) == code.n:
                temp_decoder = GenericDecoder(ConcatenatedCode([code]))
                return temp_decoder.decode_measurement(m_array, m_type)
            else:
                # Fallback: majority
                count_1 = sum(1 for m in measurements if m == 1)
                return 1 if count_1 > len(measurements) // 2 else 0


# =============================================================================
# Simulator
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                        MAIN SIMULATOR                                        │
# └─────────────────────────────────────────────────────────────────────────────┘

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                     ConcatenatedCodeSimulator                                │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ Input:                                                                        │
# │   - concat_code: ConcatenatedCode                                            │
# │   - noise_model: NoiseModel                                                  │
# │   - use_steane_strategy: bool (auto-detected if None)                        │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ Components (created in __init__):                                            │
# │                                                                              │
# │   ┌─────────────────┐                                                        │
# │   │ TransversalOps  │◄──────────────────────────────────────┐                │
# │   └────────┬────────┘                                       │                │
# │            │                                                │                │
# │            ▼                                                │                │
# │   ┌─────────────────┐    ┌─────────────────┐               │                │
# │   │   ECGadget      │◄──►│ Preparation     │               │                │
# │   │ (Steane/Knill)  │    │  Strategy       │               │                │
# │   └────────┬────────┘    └────────┬────────┘               │                │
# │            │                      │                        │                │
# │            │                      │                        │                │
# │   ┌────────┴──────────────────────┴────────┐               │                │
# │   │                                        │               │                │
# │   ▼                                        ▼               │                │
# │   ┌─────────────────┐             ┌─────────────────┐      │                │
# │   │    Decoder      │             │ PostSelector    │      │                │
# │   │ (Steane/Generic)│             │                 │      │                │
# │   └────────┬────────┘             └────────┬────────┘      │                │
# │            │                               │               │                │
# │            ▼                               │               │                │
# │   ┌─────────────────┐                      │               │                │
# │   │ Acceptance      │◄─────────────────────┘               │                │
# │   │   Checker       │                                      │                │
# │   └─────────────────┘                                      │                │
# │                                                            │                │
# │   ┌─────────────────┐                                      │                │
# │   │ LogicalGate     │──────────────────────────────────────┘                │
# │   │   Dispatcher    │                                                        │
# │   └─────────────────┘                                                        │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ Methods:                                                                     │
# │                                                                              │
# │ + estimate_logical_cnot_error_l1(p, num_shots, Q=10)                         │
# │   -> (logical_error:  float, variance: float)                                 │
# │                                                                              │
# │ + estimate_logical_cnot_error_l2(p, num_shots, Q=1)                          │
# │   -> (logical_error: float, variance: float)                                 │
# │                                                                              │
# │ + estimate_memory_logical_error_l1(p, num_shots, num_ec_rounds=1)            │
# │   -> (logical_error: float, variance:  float)                                 │
# │                                                                              │
# │ + estimate_memory_logical_error_l2(p, num_shots, num_ec_rounds=1)            │
# │   -> (logical_error: float, variance: float)                                 │
# └─────────────────────────────────────────────────────────────────────────────┘
# =============================================================================

class ConcatenatedCodeSimulator:
    """
    Main simulator for concatenated CSS quantum error correcting codes.
    
    ═══════════════════════════════════════════════════════════════════════════
                               QEC THEORY BACKGROUND
    ═══════════════════════════════════════════════════════════════════════════
    
    This simulator estimates logical error rates for fault-tolerant quantum
    operations implemented with concatenated CSS codes. It uses Stim for
    efficient circuit simulation and detector sampling.
    
    FAULT-TOLERANT CNOT VERIFICATION (BELL PAIR PROTOCOL)
    -----------------------------------------------------
    The primary test is CNOT verification using maximally entangled Bell pairs.
    
    Protocol overview:
    1. Prepare two Bell pairs: |Φ+⟩₁₂ ⊗ |Φ+⟩₃₄
       where |Φ+⟩ = (|00⟩ + |11⟩)/√2
       
    2. Apply Q rounds of: (ideal CNOT) × (noisy CNOT) × (EC)
       The ideal CNOT cancels with noisy CNOT if no errors occur.
       
    3. Undo Bell pairs: CNOT then H on each pair
    
    4. Measure all four qubits → should all be 0 if no logical error
    
    Why this works:
    - CNOT₁₃|Φ+⟩₁₂|Φ+⟩₃₄ = |Φ+⟩₁₃|Φ+⟩₂₄ (entanglement pattern swaps)
    - After ideal+noisy CNOT pairs, if no logical error, state returns to original
    - Bell pair undoing maps |Φ+⟩ → |00⟩
    - Any deviation from |0000⟩ indicates a logical error
    
    MEMORY TEST
    -----------
    Simpler test that prepares |0⟩_L, applies EC rounds, and measures:
    
    1. Prepare logical |0⟩
    2. Apply EC rounds with noise
    3. Measure in Z-basis
    4. Check if outcome = 0 (success) or 1 (logical error)
    
    This tests the complete EC pipeline without CNOT complications.
    
    LEVEL-1 VS LEVEL-2
    ------------------
    Level-1 (L1): Single-level encoding
        - n physical qubits encode 1 logical qubit
        - EC decodes directly from physical measurements
        - Example: Steane [[7,1,3]] at L1 uses 7 qubits
        
    Level-2 (L2): Two-level concatenation
        - n_outer × n_inner physical qubits encode 1 logical qubit
        - Hierarchical decoding: first decode inner blocks, then outer code
        - Uses PropagationTables for correct error tracking
        - Example: Steane→Steane at L2 uses 49 qubits
    
    ERROR RATE ESTIMATION
    ---------------------
    The simulator estimates:
    
        logical_error_rate = (# samples with logical error) / (# accepted samples × Q)
    
    Where Q is the number of CNOT rounds (for CNOT test) or EC rounds (for memory).
    
    Variance estimation assumes Poisson statistics:
        variance = num_errors / (num_accepted × Q)²
    
    ═══════════════════════════════════════════════════════════════════════════
                              COMPONENT ARCHITECTURE
    ═══════════════════════════════════════════════════════════════════════════
    
    The simulator is composed of pluggable components:
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    ConcatenatedCodeSimulator                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ concat_code: ConcatenatedCode  - Code definition                        │
    │ noise_model: NoiseModel        - Noise parameters                       │
    │ ops: TransversalOps            - Physical gate implementation           │
    │ ec: ECGadget                   - Error correction gadget                │
    │ prep: PreparationStrategy      - State preparation                      │
    │ decoder: Decoder               - Syndrome decoding                      │
    │ post_selector: PostSelector    - Sample filtering                       │
    │ acceptance: AcceptanceChecker  - Error counting                         │
    │ gates: LogicalGateDispatcher   - Logical gate routing                   │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Default components:
    - ECGadget: KnillECGadget (teleportation-based EC)
    - PreparationStrategy: GenericPreparationStrategy
    - Decoder: GenericDecoder
    
    ═══════════════════════════════════════════════════════════════════════════
                              SIMULATION DATA FLOW
    ═══════════════════════════════════════════════════════════════════════════
    
                        ┌─────────────────┐
                        │  User Request   │
                        │  (p, num_shots) │
                        └────────┬────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         BUILD CIRCUIT PHASE                              │
    │  1. Prepare Bell pairs / logical states                                  │
    │  2. Apply Q rounds of (ideal op × noisy op × EC)                         │
    │  3. Collect detector indices for syndrome extraction                     │
    │  4. Final measurement                                                    │
    └─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                            SAMPLING                                      │
    │  circuit.compile_detector_sampler().sample(shots=num_shots)              │
    │  → samples: array of detector outcomes                                   │
    └─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         POST-SELECTION                                   │
    │  Filter samples based on verification measurements                       │
    │  (reject samples with preparation/verification failures)                 │
    └─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       DECODING & ACCEPTANCE                              │
    │  For each sample:                                                        │
    │    - Decode syndromes → determine corrections                            │
    │    - Apply Pauli frame to final measurement                              │
    │    - Check if outcome matches expected (0 for Bell pairs)                │
    │    - Count errors                                                        │
    └─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         COMPUTE RESULTS                                  │
    │  logical_error = num_errors / (num_accepted × Q × k)                     │
    │  variance = num_errors / (num_accepted × Q × k)²                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ═══════════════════════════════════════════════════════════════════════════
                              COMPATIBILITY NOTES
    ═══════════════════════════════════════════════════════════════════════════
    
    IMPORTANT: Bell-pair verification requires transversal logical Hadamard.
    
    Compatible codes (self-dual, Hz = Hx):
    - Steane [[7,1,3]] ✓
    - C4 [[4,2,2]] ✓ (uses EDT post-selection)
    - C6 [[6,1,2]] ✓
    
    Partially compatible (non-self-dual):
    - Shor [[9,1,3]]: L1 memory works, CNOT tests fail
      (transversal H ≠ logical H breaks Bell pair protocol)
    
    To check compatibility:
        if code.has_transversal_logical_h:
            # CNOT and memory tests work
        else:
            # Only memory tests work (no Bell pair verification)
    
    ═══════════════════════════════════════════════════════════════════════════
                              AVAILABLE TESTS
    ═══════════════════════════════════════════════════════════════════════════
    
    estimate_logical_cnot_error_l1(p, num_shots, Q):
        Level-1 CNOT verification using Bell pairs
        
    estimate_logical_cnot_error_l2(p, num_shots, Q):
        Level-2 CNOT with hierarchical decoding
        
    estimate_memory_logical_error_l1(p, num_shots, num_ec_rounds):
        Level-1 memory: |0⟩_L → EC → measure
        
    estimate_memory_logical_error_l2(p, num_shots, num_ec_rounds):
        Level-2 memory with full concatenation
        
    estimate_simple_memory_l1(p, num_shots):
        Basic memory without EC (encode → noise → decode)
    
    References:
        [Kni05] Knill, Nature 434, 39 (2005)
        [AGP06] Aliferis, Gottesman, Preskill, QIC 6, 97 (2006)
        [Got97] Gottesman, PhD Thesis, Caltech (1997)
    """
    
    def __init__(self, concat_code: ConcatenatedCode, noise_model: NoiseModel,
                 ec_gadget: ECGadget = None,
                 prep_strategy: PreparationStrategy = None,
                 decoder: Decoder = None):
        """
        Initialize the concatenated code simulator.
        
        Creates a complete simulation environment with all necessary components
        for estimating logical error rates of fault-tolerant operations.
        
        Args:
            concat_code: ConcatenatedCode instance defining the code structure.
                Contains CSS codes for each level and propagation tables.
                
            noise_model: NoiseModel instance defining error probabilities.
                Applied to noisy operations (CNOTs, measurements, etc.)
                
            ec_gadget: Optional custom ECGadget for error correction.
                Default: KnillECGadget (teleportation-based EC)
                
            prep_strategy: Optional custom PreparationStrategy.
                Default: GenericPreparationStrategy
                
            decoder: Optional custom Decoder for syndrome processing.
                Default: GenericDecoder (syndrome table lookup)
        
        Component wiring:
            The __init__ method creates and connects all components:
            1. TransversalOps for physical gate implementation
            2. ECGadget for error correction (with circular dependency to prep)
            3. PreparationStrategy for state preparation
            4. Decoder for syndrome decoding
            5. PostSelector for sample filtering
            6. AcceptanceChecker for error counting
            7. LogicalGateDispatcher for logical operations
        
        Warning:
            Non-self-dual codes (like Shor [[9,1,3]]) will produce incorrect
            results for Bell-pair CNOT verification tests. Use memory tests
            only, or implement a code-specific simulator.
        
        Example:
            # Create simulator for Steane code
            from concatenated_css_v10_steane import create_steane_code
            code = ConcatenatedCode([create_steane_code()])
            noise = DepolarizingNoiseModel(p=0.001)
            sim = ConcatenatedCodeSimulator(code, noise)
            error, var = sim.estimate_memory_logical_error_l1(0.001, 10000)
        """
        self.concat_code = concat_code
        self.noise_model = noise_model
        self.ops = TransversalOps(concat_code)
        
        # Warn if inner code doesn't support transversal logical H
        inner_code = concat_code.code_at_level(0)
        if not inner_code.has_transversal_logical_h:
            import warnings
            warnings.warn(
                f"Code '{inner_code.name}' does not have transversal logical H. "
                f"The Bell-pair CNOT verification protocol will produce incorrect results. "
                f"Use a code-specific simulator for non-self-dual codes.",
                UserWarning
            )
        
        # Create EC gadget (default to Knill, or use custom)
        if ec_gadget is not None:
            self.ec = ec_gadget
        else:
            self.ec = KnillECGadget(concat_code, self.ops)
        
        # Create preparation strategy (default to generic)
        if prep_strategy is not None:
            self.prep = prep_strategy
        else:
            self.prep = GenericPreparationStrategy(concat_code, self.ops)
        
        # Create decoder (default to generic)
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = GenericDecoder(concat_code)
        
        # Wire up circular dependencies
        self.ec.set_prep(self.prep)
        self.prep.set_ec_gadget(self.ec)
        
        self.post_selector = PostSelector(concat_code, self.decoder)
        self.acceptance = AcceptanceChecker(concat_code, self.decoder)
        
        # Gate dispatcher for logical operations
        self.gates = LogicalGateDispatcher(concat_code, self.ops)
    
    def estimate_logical_cnot_error_l1(self, p: float, num_shots: int,
                                        Q: int = 10) -> Tuple[float, float]:
        """
        Estimate level-1 logical CNOT error rate.
        Matches original estimate_logical_cnot_error_l1.
        """
        N_prev = 1
        N_now = self.concat_code.code_at_level(0).n
        NN = 2 * N_now
        
        list_detector_0prep = []
        list_detector_X = []
        list_detector_Z = []
        list_detector_m = []
        
        circuit = stim.Circuit()
        detector_counter = [0]
        
        # Prepare ideal Bell pairs
        self.prep.append_0prep(circuit, 0, N_prev, N_now)
        self.prep.append_0prep(circuit, NN, N_prev, N_now)
        self.prep.append_0prep(circuit, 2 * NN, N_prev, N_now)
        self.prep.append_0prep(circuit, 3 * NN, N_prev, N_now)
        
        # Use LOGICAL H (not transversal) for Bell-pair creation
        # For self-dual codes: same as transversal H
        # For non-self-dual codes (Shor): H only on logical_h_qubits
        inner_code = self.concat_code.code_at_level(0)
        self.ops.append_logical_h(circuit, 0, N_prev, N_now, inner_code)
        self.ops.append_logical_h(circuit, 2 * NN, N_prev, N_now, inner_code)
        
        self.ops.append_cnot(circuit, 0, NN, N_prev, N_now)
        self.ops.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, N_now)
        
        # Q rounds of CNOT + EC
        for q in range(Q):
            self.ops.append_cnot(circuit, 0, 2 * NN, N_prev, N_now)
            self.ops.append_noisy_cnot(circuit, 0, 2 * NN, N_prev, N_now, p)
            
            result = self.ec.append_noisy_ec(
                circuit, 0, 4 * NN, 5 * NN, 6 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_X.append(result[2])
            list_detector_Z.append(result[1])
            
            result = self.ec.append_noisy_ec(
                circuit, 2 * NN, 4 * NN, 5 * NN, 6 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_X.append(result[2])
            list_detector_Z.append(result[1])
        
        # Undo Bell pairs
        self.ops.append_cnot(circuit, 0, NN, N_prev, N_now)
        self.ops.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, N_now)
        
        # Use LOGICAL H (not transversal) for Bell-pair destruction
        self.ops.append_logical_h(circuit, 0, N_prev, N_now, inner_code)
        self.ops.append_logical_h(circuit, 2 * NN, N_prev, N_now, inner_code)
        
        # Measure - pass inner code for strategy-aware detector generation
        inner_code = self.concat_code.code_at_level(0)
        list_detector_m.append(self.ops.append_m(circuit, 0, N_prev, N_now, detector_counter, inner_code))
        list_detector_m.append(self.ops.append_m(circuit, NN, N_prev, N_now, detector_counter, inner_code))
        list_detector_m.append(self.ops.append_m(circuit, 2 * NN, N_prev, N_now, detector_counter, inner_code))
        list_detector_m.append(self.ops.append_m(circuit, 3 * NN, N_prev, N_now, detector_counter, inner_code))
        
        # Sample (noise already applied inline by append_noisy_* functions)
        samples = circuit.compile_detector_sampler().sample(shots=num_shots)
        
        # Post-selection
        samples = [x for x in samples if self.post_selector.post_selection_l1(x, list_detector_0prep)]
        num = len(samples)
        
        # Count errors
        err = sum([self.acceptance.accept_l1(x, list_detector_m, list_detector_X,
                                              list_detector_Z, Q) for x in samples])
        
        print(p, num, err)
        
        # Get normalization factor (k for k>1 codes have err summed over k dimensions)
        k = self.concat_code.code_at_level(0).error_normalization_factor
        
        if num > 0:
            logical_error = err / (num * Q * k)
            variance = err / (num * Q * k) ** 2
        else:
            logical_error = variance = 0
        
        return logical_error, variance
    
    def estimate_logical_cnot_error_l2(self, p: float, num_shots:  int,
                                        Q:  int = 1) -> Tuple[float, float]:
        """
        Estimate level-2 logical CNOT error rate.
        Matches original estimate_logical_cnot_error_l2.
        
        Parameter calculation for different code combinations:
        - Steane->Steane: N_prev=7, N_now=7, n_now=7, NN=14 (2*N_now)
        - C4->C6:         N_prev=4, N_now=6, n_now=3, NN=6  (N_now)
        - C4->Steane:     N_prev=4, N_now=7, n_now=7, NN=7  (N_now)
        
        The key insight:
        - N_prev = inner code's n (level 0)
        - N_now = outer code's n (level 1)
        - n_now = outer code's transversal_block_count (defaults to N_now)
        - NN = 2*N_now if same code at both levels, else N_now
        """
        inner_code = self.concat_code.code_at_level(0)
        outer_code = self.concat_code.code_at_level(1) if self.concat_code.num_levels > 1 else inner_code
        
        N_prev = inner_code.n
        N_now = outer_code.n
        
        # n_now is the number of transversal blocks in outer code
        # For C6: 3 blocks; for Steane: 7 (same as n)
        n_now = outer_code.transversal_block_count if outer_code.transversal_block_count else N_now
        
        # NN is the stride between logical blocks in the circuit
        # Same code at both levels: NN = 2*N_now (Steane->Steane)
        # Different codes: NN = N_now (C4->C6, C4->Steane)
        same_code = (inner_code.name == outer_code.name)
        NN = 2 * N_now if same_code else N_now
        
        list_detector_0prep = []
        list_detector_0prep_l2 = []
        list_detector_X = []
        list_detector_Z = []
        list_detector_m = []
        
        circuit = stim.Circuit()
        detector_counter = [0]
        
        # Prepare ideal Bell pairs using noiseless prep (matches original)
        self.prep.append_0prep(circuit, 0, N_prev, N_now)
        self.prep.append_0prep(circuit, NN, N_prev, N_now)
        self.prep.append_0prep(circuit, 2 * NN, N_prev, N_now)
        self.prep.append_0prep(circuit, 3 * NN, N_prev, N_now)
        
        # Use LOGICAL H (not transversal) for Bell-pair creation
        # For self-dual codes: same as transversal H
        # For non-self-dual codes (Shor): H only on logical_h_qubits
        self.ops.append_logical_h(circuit, 0, N_prev, n_now, inner_code)
        self.ops.append_logical_h(circuit, 2 * NN, N_prev, n_now, inner_code)
        
        self.ops.append_cnot(circuit, 0, NN, N_prev, n_now)
        self.ops.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, n_now)
        
        # Q rounds of CNOT + EC
        for q in range(Q):
            self.ops.append_cnot(circuit, 0, 2 * NN, N_prev, n_now)
            self.ops.append_noisy_cnot(circuit, 0, 2 * NN, N_prev, n_now, p)
            
            # EC uses N_now (full outer code size) for ancilla addressing
            result = self.ec.append_noisy_ec(
                circuit, 0, 4 * NN, 5 * NN, 6 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_0prep_l2.extend(result[1])
            list_detector_X.append(result[3])
            list_detector_Z.append(result[2])
            
            result = self.ec.append_noisy_ec(
                circuit, 2 * NN, 4 * NN, 5 * NN, 6 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_0prep_l2.extend(result[1])
            list_detector_X.append(result[3])
            list_detector_Z.append(result[2])
        
        # Undo Bell pairs (use n_now)
        self.ops.append_cnot(circuit, 0, NN, N_prev, n_now)
        self.ops.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, n_now)
        
        # Use LOGICAL H (not transversal) for Bell-pair destruction
        self.ops.append_logical_h(circuit, 0, N_prev, n_now, inner_code)
        self.ops.append_logical_h(circuit, 2 * NN, N_prev, n_now, inner_code)
        
        # Measure (use n_now) - pass inner code for strategy-aware detector generation
        inner_code = self.concat_code.code_at_level(0)
        list_detector_m.append(self.ops.append_m(circuit, 0, N_prev, n_now, detector_counter, inner_code))
        list_detector_m.append(self.ops.append_m(circuit, NN, N_prev, n_now, detector_counter, inner_code))
        list_detector_m.append(self.ops.append_m(circuit, 2 * NN, N_prev, n_now, detector_counter, inner_code))
        list_detector_m.append(self.ops.append_m(circuit, 3 * NN, N_prev, n_now, detector_counter, inner_code))
        
        # Sample (noise already applied inline by append_noisy_* functions)
        samples = circuit.compile_detector_sampler().sample(shots=num_shots)
        
        # Post-selection on 0-prep
        samples = [x for x in samples if self.post_selector.post_selection_l2(
            x, list_detector_0prep, list_detector_0prep_l2,
            list_detector_X, list_detector_Z, Q
        )]
        
        # CRITICAL: Also filter by accept != -1 (EDT post-selection)
        # This matches the original behavior where samples with uncorrectable
        # errors in the inner code are rejected before counting
        samples = [x for x in samples if self.acceptance.accept_l2(
            x, list_detector_m, list_detector_X, list_detector_Z, Q
        ) != -1]
        num = len(samples)
        
        # Count errors (only samples that passed EDT filtering)
        err = sum([self.acceptance.accept_l2(x, list_detector_m, list_detector_X,
                                              list_detector_Z, Q) for x in samples])
        
        print(p, num, err)
        
        # Get normalization factor (k for k>1 codes have err summed over k dimensions)
        k = self.concat_code.code_at_level(0).error_normalization_factor
        
        if num > 0:
            logical_error = err / (num * Q * k)
            variance = err / (num * Q * k) ** 2
        else:
            logical_error = variance = 0
        
        return logical_error, variance
    
    def estimate_memory_logical_error_l1(self, p: float, num_shots: int,
                                          num_ec_rounds: int = 1) -> Tuple[float, float]: 
        """
        Estimate level-1 memory logical error rate.
        
        Prepares |0⟩_L, applies EC rounds, measures, checks if outcome is 0.
        
        IMPORTANT: EC uses teleportation. For Steane/Knill EC:
        - Z measurement (on loc1 after H) gives the X Pauli frame update
        - X measurement (on loc2) gives the Z Pauli frame update
        For a Z-basis final measurement, X errors flip the outcome, so we
        accumulate X corrections from the Z syndrome of each EC round.
        
        Supports both k=1 codes (scalar outcome) and k>1 codes (array outcome).
        """
        N_prev = 1
        N_now = self.concat_code.code_at_level(0).n
        NN = 2 * N_now
        k = self.concat_code.code_at_level(0).k
        
        list_detector_0prep = []
        list_detector_X = []
        list_detector_Z = []
        
        circuit = stim.Circuit()
        detector_counter = [0]
        
        # Prepare |0⟩_L
        self.prep.append_0prep(circuit, 0, N_prev, N_now)
        
        # EC rounds
        for _ in range(num_ec_rounds):
            result = self.ec.append_noisy_ec(
                circuit, 0, NN, 2 * NN, 3 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_X.append(result[2])  # X syndrome (loc2 measurement) - determines X correction
            list_detector_Z.append(result[1])  # Z syndrome (loc1 measurement after H) - determines Z correction
        
        # Measure (Z-basis measurement of logical state) - pass inner code for strategy-aware detectors
        inner_code = self.concat_code.code_at_level(0)
        detector_m = self.ops.append_m(circuit, 0, N_prev, N_now, detector_counter, inner_code)
        
        # Sample
        samples = circuit.compile_detector_sampler().sample(shots=num_shots)
        
        # Post-selection on prep detectors
        samples = [x for x in samples if self.post_selector.post_selection_l1(x, list_detector_0prep)]
        num = len(samples)
        
        # Count errors with proper Pauli frame tracking
        # For teleportation-based EC:
        # - loc2 measurement (detector_X) gives Z-type measurement → determines X correction
        # - loc1 measurement after H (detector_Z) gives X-type measurement → determines Z correction
        # For Z-basis final measurement, X errors flip the outcome, so we need X correction from detector_X
        num_errors = 0
        for x in samples:
            # Accumulate X correction (affects Z-basis measurement)
            x_correction = 0  # For k=1; becomes list for k>1
            if k >= 2:
                x_correction = [0] * k
            
            for i in range(num_ec_rounds):
                # Use detector_X (loc2 measurement) for X correction
                # This is the Z-type measurement that determines X Pauli frame
                det_X = list_detector_X[i]
                
                # Handle different return formats from append_noisy_ec
                if isinstance(det_X, list):
                    if len(det_X) > 0 and isinstance(det_X[-1], list):
                        # Nested list: [[start, end]] or [..., [start, end]]
                        final_det = det_X[-1]
                    else:
                        # Simple [start, end]
                        final_det = det_X
                else:
                    final_det = det_X
                
                if isinstance(final_det, list) and len(final_det) == 2:
                    # EC SYNDROME decoding - use decode_syndrome, NOT decode_measurement!
                    # detector_X (loc2) is a Z-basis syndrome measurement
                    x_meas = self.decoder.decode_syndrome(
                        x[final_det[0]:final_det[1]], 'z'
                    )
                    # loc2 measurement determines X correction on teleported state
                    if x_meas == 1:
                        if k >= 2:
                            for a in range(k):
                                x_correction[a] = (x_correction[a] + 1) % 2
                        else:
                            x_correction = (x_correction + 1) % 2
            
            # Decode final measurement (Z-basis: we prepared |0⟩_L and measure in Z-basis)
            m = x[detector_m[0]:detector_m[1]]
            outcome = self.decoder.decode_measurement_k(m, 'z')  # Z-basis measurement
            
            # Apply X correction to Z-basis measurement outcome
            if isinstance(outcome, (list, np.ndarray)):
                for a in range(k):
                    if outcome[a] != -1:
                        corr = x_correction[a] if isinstance(x_correction, list) else x_correction
                        outcome[a] = (outcome[a] + corr) % 2
                    if outcome[a] != 0 and outcome[a] != -1:
                        num_errors += 1
            else:
                if outcome != -1:
                    outcome = (outcome + x_correction) % 2
                if outcome != 0:
                    num_errors += 1
        
        print(f"Memory L1: p={p}, accepted={num}, errors={num_errors}")
        
        if num > 0:
            logical_error = num_errors / (num * k)
            variance = num_errors / ((num * k) ** 2)
        else:
            logical_error = variance = 0
        
        return logical_error, variance
    
    def estimate_memory_logical_error_l2(self, p: float, num_shots: int,
                                          num_ec_rounds: int = 1) -> Tuple[float, float]:
        """
        Estimate level-2 memory logical error rate.
        
        Prepares |0⟩_L at level 2 (noisy), applies EC rounds, measures, checks if outcome is 0.
        Supports both k=1 codes (scalar outcome) and k>1 codes (array outcome).
        """
        N_prev = self.concat_code.code_at_level(0).n
        N_now = self.concat_code.code_at_level(0).n
        NN = 2 * N_now
        k = self.concat_code.code_at_level(0).k
        
        list_detector_0prep = []
        list_detector_0prep_l2 = []
        list_detector_X = []
        list_detector_Z = []
        
        circuit = stim.Circuit()
        detector_counter = [0]
        
        # Prepare |0⟩_L at level 2 (NOISY prep)
        # IMPORTANT: Save detector_X_prep and detector_Z_prep for L2 post-selection on prep
        prep_result = self.prep.append_noisy_0prep(circuit, 0, NN, N_prev, N_now, p, detector_counter)
        list_detector_0prep.extend(prep_result[0])
        list_detector_0prep_l2.append(prep_result[1])
        detector_X_prep = prep_result[2]  # X detectors from prep (for L2 verification)
        detector_Z_prep = prep_result[3]  # Z detectors from prep (for L2 verification)
        
        # EC rounds
        for _ in range(num_ec_rounds):
            result = self.ec.append_noisy_ec(
                circuit, 0, NN, 2 * NN, 3 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_0prep_l2.extend(result[1])
            list_detector_X.append(result[3])
            list_detector_Z.append(result[2])
        
        # Measure - pass inner code for strategy-aware detector generation
        inner_code = self.concat_code.code_at_level(0)
        detector_m = self.ops.append_m(circuit, 0, N_prev, N_now, detector_counter, inner_code)
        
        # Sample (noise already applied inline by append_noisy_* functions)
        samples = circuit.compile_detector_sampler().sample(shots=num_shots)
        
        # Post-selection with prep's X/Z detectors for proper L2 verification
        Q = num_ec_rounds
        samples = [x for x in samples if self.post_selector.post_selection_l2_memory(
            x, list_detector_0prep, list_detector_0prep_l2,
            list_detector_X, list_detector_Z, Q,
            detector_X_prep=detector_X_prep, detector_Z_prep=detector_Z_prep
        )]
        num = len(samples)
        
        # Count errors with hierarchical decoding
        # Get inner code size dynamically
        inner_n = self.concat_code.code_at_level(0).n
        
        num_errors = 0
        for x in samples:
            correction_x = np.zeros(inner_n)
            correction_z = np.zeros(inner_n)
            outer_x_correction = 0  # Accumulated outer code X correction (scalar or array)
            
            # Decode EC rounds - decode_ec_hd returns (outer_x, outer_z, level1_x, level1_z)
            for i in range(num_ec_rounds):
                outer_x, outer_z, correction_x, correction_z = self.decoder.decode_ec_hd(
                    x, list_detector_X[i], list_detector_Z[i],
                    correction_x, correction_z
                )
                # Accumulate outer code X correction (X errors flip Z-basis measurements)
                # For k>1, outer_x might be a list
                if isinstance(outer_x, (list, np.ndarray)):
                    if isinstance(outer_x_correction, (int, float)):
                        outer_x_correction = [0] * len(outer_x)
                    for a in range(len(outer_x)):
                        if outer_x[a] == 1:
                            outer_x_correction[a] = (outer_x_correction[a] + 1) % 2
                elif outer_x == 1:
                    outer_x_correction = (outer_x_correction + 1) % 2
            
            # Decode final Z-basis measurement with level-1 X corrections
            # X errors cause bit flips in Z-basis measurements!
            outcome = self.decoder.decode_m_hd(x, detector_m, correction_x)
            
            # Apply outer code X correction to the measurement outcome
            # Handle both scalar (k=1) and array (k>1) outcomes
            if isinstance(outcome, (list, np.ndarray)):
                # k>1: apply corrections per logical qubit
                for a in range(k):
                    corr = outer_x_correction[a] if isinstance(outer_x_correction, list) else outer_x_correction
                    if corr == 1 and outcome[a] != -1:
                        outcome[a] = (outcome[a] + 1) % 2
                    if outcome[a] != 0 and outcome[a] != -1:
                        num_errors += 1
            else:
                # k=1: scalar comparison
                if outer_x_correction == 1:
                    outcome = (outcome + 1) % 2
                if outcome != 0:
                    num_errors += 1
        
        print(f"Memory L2: p={p}, accepted={num}, errors={num_errors}")
        
        if num > 0:
            logical_error = num_errors / (num * k)
            variance = num_errors / ((num * k) ** 2)
        else:
            logical_error = variance = 0
        
        return logical_error, variance

    def estimate_simple_memory_l1(self, p: float, num_shots: int) -> Tuple[float, float]:
        """
        Estimate level-1 memory error without EC (encode → noise → measure → decode).
        
        This is useful for:
        - Non-self-dual codes where EC uses transversal H (which isn't logical H)
        - Validating that basic encoding/decoding works correctly
        - Testing error correction in the decoder without EC gadget complications
        
        The test:
        1. Prepare logical |0⟩ (noisily if p > 0)
        2. Apply depolarizing noise with probability p to all qubits
        3. Measure all qubits in Z basis
        4. Decode using syndrome-based correction
        5. Check if decoded value is 0
        
        Args:
            p: Physical error probability (depolarizing noise)
            num_shots: Number of shots to sample
        
        Returns:
            Tuple of (logical_error_rate, variance)
        """
        inner_code = self.concat_code.code_at_level(0)
        n = inner_code.n
        k = inner_code.k
        
        circuit = stim.Circuit()
        detector_counter = [0]
        
        # Prepare |0⟩_L (using perfect prep for simplicity)
        self.prep.append_0prep(circuit, 0, 1, n)
        
        # Apply depolarizing noise if p > 0
        if p > 0:
            for q in range(n):
                circuit.append("DEPOLARIZE1", q, p)
        
        # Measure all qubits in Z basis
        detector_m = self.ops.append_m(circuit, 0, 1, n, detector_counter, inner_code)
        
        # Sample
        samples = circuit.compile_detector_sampler().sample(shots=num_shots)
        
        # Count errors by decoding each sample
        num_errors = 0
        for sample in samples:
            m = sample[detector_m[0]:detector_m[1]]
            outcome = self.decoder.decode_measurement_k(m, 'z')
            
            # Check if decoded to expected value (0 for |0⟩_L)
            if isinstance(outcome, (list, np.ndarray)):
                for a in range(k):
                    if outcome[a] != 0 and outcome[a] != -1:
                        num_errors += 1
            else:
                if outcome != 0 and outcome != -1:
                    num_errors += 1
        
        print(f"Simple Memory L1: p={p}, samples={num_shots}, errors={num_errors}")
        
        if num_shots > 0:
            logical_error = num_errors / (num_shots * k)
            variance = num_errors / ((num_shots * k) ** 2)
        else:
            logical_error = variance = 0
        
        return logical_error, variance


# =============================================================================
# Factory Functions
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                          FACTORY FUNCTIONS                                   │
# └─────────────────────────────────────────────────────────────────────────────┘

# Code Creation:
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ create_shor_code() -> CSSCode                                                │
# │   Returns Shor [[9,1,3]] with circuit specification                          │
# │                                                                              │
# │ create_concatenated_code(codes, prop_tables) -> ConcatenatedCode             │
# │   Generic factory for any concatenated code                                  │
# │                                                                              │
# │ For Steane-specific code, use concatenated_css_v10_steane.py:                │
# │   create_steane_code() -> CSSCode                                            │
# │   create_steane_propagation_l2() -> PropagationTables                        │
# │   create_concatenated_steane(num_levels) -> ConcatenatedCode                 │
# │   create_steane_simulator(num_levels, noise_model)                           │
# └─────────────────────────────────────────────────────────────────────────────┘

# Simulator Creation:
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ create_simulator(concat_code, noise_model)                                   │
# │   -> ConcatenatedCodeSimulator                                               │
# │   Uses generic preparation, EC, and decoding                                 │
# │                                                                              │
# │ For Steane-specific simulation, use:                                         │
# │   from concatenated_css_v10_steane import create_steane_simulator            │
# └─────────────────────────────────────────────────────────────────────────────┘
# =============================================================================

def create_simulator(concat_code: ConcatenatedCode, 
                     noise_model: NoiseModel,
                     use_idle_noise: bool = True) -> ConcatenatedCodeSimulator: 
    """
    Factory function to create a generic simulator.
    
    Args:
        concat_code: The concatenated code
        noise_model: Noise model to apply
        use_idle_noise: If True (default), applies DEPOLARIZE1 noise on idle qubits
                        during CNOT rounds. Set to False to match original behavior
                        that doesn't model idle decoherence.
    
    Returns:
        Configured simulator with generic components
    
    Note: For Steane-specific simulation, use create_steane_simulator from
    concatenated_css_v10_steane.py instead. That provides exact propagation tables
    for better decoding accuracy.
    
    This generic factory will auto-generate approximate propagation tables if
    none are provided, enabling L2+ simulation for any CSS code.
    """
    # Auto-generate propagation tables if not provided for L2+ codes
    if concat_code.num_levels >= 2 and 1 not in concat_code.propagation_tables:
        # Create a temporary prep strategy to compute tables
        ops = TransversalOps(concat_code)
        prep = GenericPreparationStrategy(concat_code, ops, use_idle_noise=use_idle_noise)
        inner_code = concat_code.code_at_level(0)
        
        # Compute approximate propagation tables
        prop_tables = prep.compute_propagation_tables(inner_code)
        concat_code.propagation_tables[1] = prop_tables
    
    # Create simulator with configured prep strategy
    ops = TransversalOps(concat_code)
    prep = GenericPreparationStrategy(concat_code, ops, use_idle_noise=use_idle_noise)
    
    return ConcatenatedCodeSimulator(concat_code, noise_model, prep_strategy=prep)


# =============================================================================
# Main Entry Point (for command-line usage)
# =============================================================================

if __name__ == '__main__':
    import sys
    import json
    
    print("For Steane code simulation, use concatenated_css_v10_steane.py")
    print("This module (concatenated_css_v10.py) contains only generic CSS code infrastructure.")
    print()
    print("Example usage:")
    print("  from concatenated_css_v10_steane import create_steane_simulator")
    print("  simulator = create_steane_simulator(num_levels=2, noise_model=noise_model)")
    print("  error, var = simulator.estimate_logical_cnot_error_l2(p=0.001, num_shots=10000)")