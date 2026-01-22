"""
Concatenated CSS Code Simulator v10 - C4Steane Code Implementation
─────────────────────────────────────────────────────────────────────────────

This module provides C4Steane-specific implementations for the concatenated CSS
code simulator. It imports generic base classes from concatenated_css_v10.py
and provides exact matching to concatenated_c4steane.py ground truth.

C4 [[4,2,2]] Code (inner code):
- Encodes 2 logical qubits in 4 physical (but we use 1)
- Decoder returns 2 bits: [bit0, bit1]
- Stabilizers: Hx = Hz = [[1,1,1,1]]

Steane [[7,1,3]] Code (outer code):
- 7 C4 blocks
- Syndrome-based decoding with check matrix
- Distance 3 allows single error correction

Key differences from C4C6:
1. Steane uses 7 blocks instead of 3
2. Steane uses syndrome decoding, not majority voting
3. post_selection_steane_l2 for Steane outer verification
4. N_steane = 7 (blocks), N_c6 = 12 (physical qubits)

Usage:
    from concatenated_css_v10_c4steane import create_c4steane_simulator
    
    simulator = create_c4steane_simulator(num_levels=2, noise_model=noise_model)
    error, variance = simulator.estimate_logical_cnot_error_l2(p=0.001, num_shots=10000)
"""

import stim
import numpy as np
from typing import List, Tuple, Optional, Dict, Union

# Import generic base classes from main module
from qectostim.experiments.concatenated_css_v10 import (
    CSSCode,
    PropagationTables,
    ConcatenatedCode,
    PhysicalOps,
    TransversalOps,
    PreparationStrategy,
    ECGadget,
    Decoder,
    PostSelector,
    AcceptanceChecker,
    ConcatenatedCodeSimulator,
    GateResult,
    PrepResult,
    ECResult,
)
from qectostim.noise.models import NoiseModel


# =============================================================================
# C4Steane Transversal Operations - Extends base with C4-specific H gate
# =============================================================================

class C4SteaneTransversalOps(TransversalOps):
    """
    Transversal operations extended for C4Steane codes.
    
    Key additions:
    - append_h: Overridden to include SWAP for C4 structure
    - reset: Reset qubits (delegated to PhysicalOps)
    - noisy_reset: Reset with noise (delegated to PhysicalOps)
    """
    
    def __init__(self, concat_code: ConcatenatedCode):
        super().__init__(concat_code)
        self.N_c4 = 4
        self.N_steane = 7
    
    def append_h(self, circuit: stim.Circuit, loc: int, N_prev: int, N_now: int):
        """
        Transversal Hadamard for C4Steane with SWAP gates.
        
        Overrides base class to add SWAP gates required by C4's logical H.
        - When N_now=4 and N_prev=1: add SWAP(1,2) after H
        - When N_now=7 and N_prev=4: apply H to each C4 block
        """
        if N_prev == 1:
            for i in range(N_now):
                circuit.append("H", loc + i)
            if N_now == 4:
                # C4 requires SWAP after H
                circuit.append("SWAP", [loc + 1, loc + 2])
        else:
            for i in range(N_now):
                self.append_h(circuit, (loc + i) * N_prev, 1, N_prev)
            # Steane outer doesn't need additional SWAPs at outer level
    
    def reset(self, circuit: stim.Circuit, loc: int, n: int):
        """Reset n qubits starting at loc."""
        PhysicalOps.reset(circuit, loc, n)
    
    def noisy_reset(self, circuit: stim.Circuit, loc: int, n: int, p: float):
        """Reset n qubits with X noise."""
        PhysicalOps.noisy_reset(circuit, loc, n, p)


# =============================================================================
# C4 and Steane Code Factory Functions
# =============================================================================

def create_c4_code() -> CSSCode:
    """
    Create the [[4,2,2]] C4 code.
    
    This is a [[4,2,2]] code encoding 2 logical qubits in 4 physical qubits.
    The decoder returns 2 bits for each measurement.
    
    Uses Bell-pair preparation protocol (NOT standard CSS encoding):
    1. H on ALL ancilla qubits
    2. CNOT: ancilla[i] → data[i] 
    3. CNOT: data[i] → ancilla[(i+1)%4]
    4. Measure ALL ancilla
    5. Correction CNOTs (triangular pattern)
    """
    Hz = np.array([[1, 1, 1, 1]])
    Hx = np.array([[1, 1, 1, 1]])
    
    # C4 has 2 logical operators since it's [[4,2,2]]
    Lz1 = np.array([1, 0, 1, 0])
    Lx1 = np.array([1, 1, 0, 0])
    Lz2 = np.array([0, 0, 1, 1])
    Lx2 = np.array([0, 1, 0, 1])
    
    return CSSCode(
        name="C4",
        n=4,
        k=2,  # k=2: encodes 2 logical qubits
        d=2,
        Hz=Hz,
        Hx=Hx,
        logical_z_ops=[Lz1, Lz2],  # k=2: two logical Z operators
        logical_x_ops=[Lx1, Lx2],  # k=2: two logical X operators
        h_qubits=[0],  # Not used for Bell-pair prep
        encoding_cnots=[(0, 1), (0, 2), (0, 3)],  # Not used for Bell-pair prep
        encoding_cnot_rounds=[
            [(0, 1)],
            [(0, 2)],
            [(0, 3)],
        ],
        verification_qubits=[],  # C4 uses parity check verification
        uses_bellpair_prep=True,  # CRITICAL: Use Bell-pair protocol
    )


def create_steane_outer_code() -> CSSCode:
    """
    Create the [[7,1,3]] Steane code as outer code for C4Steane.
    
    Steane code with 7 C4 blocks as inner codes.
    Standard check matrix construction.
    Uses Bell-pair prep at L2 level.
    """
    r = 3
    N = 7
    
    check_matrix = np.zeros([r, N], dtype=int)
    for i in range(r):
        for n in range(N):
            check_matrix[i, n] = ((n + 1) // (2 ** i)) % 2
    
    Hz = check_matrix
    Hx = check_matrix  # Steane is self-dual
    Lz = np.array([1, 1, 1, 0, 0, 0, 0])
    Lx = np.array([1, 1, 1, 0, 0, 0, 0])
    
    return CSSCode(
        name="Steane",
        n=7,
        k=1,
        d=3,
        Hz=Hz,
        Hx=Hx,
        logical_z_ops=[Lz],  # k=1: single logical Z operator
        logical_x_ops=[Lx],  # k=1: single logical X operator
        h_qubits=[0, 1, 3],  # Steane encoding H positions
        encoding_cnots=[
            (1, 2), (3, 5), (0, 4),  # Round 1 (parallel)
            (1, 6), (0, 2), (3, 4),  # Round 2
            (1, 5), (4, 6),          # Round 3
        ],
        encoding_cnot_rounds=[
            [(1, 2), (3, 5), (0, 4)],
            [(1, 6), (0, 2), (3, 4)],
            [(1, 5), (4, 6)],
        ],
        verification_qubits=[2, 4, 5],  # Standard Steane verification
        uses_bellpair_prep=True,  # L2 uses Bell-pair prep
    )


# =============================================================================
# C4Steane Decoder Implementation
# =============================================================================

class C4SteaneDecoder(Decoder):
    """
    Decoder for C4Steane concatenated codes.
    
    Uses:
    - decode_measurement_c4: Hard-decision decoder for C4 inner code
    - decode_measurement_steane: Syndrome-based decoder for Steane outer code
    """
    
    def __init__(self, concat_code: ConcatenatedCode):
        super().__init__(concat_code)
        self.N_c4 = 4
        self.N_steane = 7
        
        # Steane check matrix
        r = 3
        self.check_matrix_steane = np.zeros([r, 7])
        for i in range(r):
            for n in range(7):
                self.check_matrix_steane[i, n] = ((n + 1) // (2 ** i)) % 2
        
        self.logical_op_steane = [1, 1, 1, 0, 0, 0, 0]
    
    def decode_measurement_c4(self, m: np.ndarray, m_type: str = 'x') -> List[int]:
        """
        Hard-decision decoder for C4 code.
        
        Returns [bit0, bit1] or [-1, -1] if parity error detected.
        """
        if (int(m[0]) + int(m[1]) + int(m[2]) + int(m[3])) % 2 == 1:
            return [-1, -1]
        else:
            return [(int(m[0]) + int(m[2])) % 2, (int(m[2]) + int(m[3])) % 2]
    
    def decode_measurement(self, m: np.ndarray, m_type: str = 'x') -> int:
        """
        Generic single-bit decoder for compatibility.
        Returns first bit of C4 decode.
        """
        result = self.decode_measurement_c4(m, m_type)
        if result[0] == -1:
            return -1
        return result[0]
    
    def decode_measurement_post_selection(self, m: np.ndarray, m_type: str = 'x') -> int:
        """Post-selection version - returns -1 if parity check fails."""
        result = self.decode_measurement_c4(m, m_type)
        return result[0]
    
    def decode_measurement_steane(self, m: List[int], mtype: str = 'x') -> int:
        """
        Syndrome-based decoder for Steane code.
        
        Takes 7 bits from decoded C4 blocks, returns single logical bit or -1.
        """
        error_location = [i for i in range(len(m)) if m[i] == -1]
        num_error = len(error_location)
        
        outcome = sum([m[0], m[1], m[2]]) % 2
        
        r = 3
        N = 7
        
        if num_error == 2:
            detected = False
            for pq in range(4):
                p = pq // 2
                q = pq % 2
                flag = True
                for i in range(r):
                    e = sum(m[:] * self.check_matrix_steane[i, :]) % 2
                    e = (e + p * self.check_matrix_steane[i, error_location[0]] + 
                         q * self.check_matrix_steane[i, error_location[1]]) % 2
                    if e != 0:
                        flag = False
                if flag:
                    detected = True
                    break
            if not detected:
                return -1
            outcome = (outcome + p * self.logical_op_steane[error_location[0]] + 
                       q * self.logical_op_steane[error_location[1]]) % 2
        else:
            syndrome = 0
            for i in range(r):
                e = sum(m[:] * self.check_matrix_steane[i, :]) % 2
                syndrome += e * (2 ** i)
            if syndrome == 1 or syndrome == 2 or syndrome == 3:
                outcome = (outcome + 1) % 2
        
        return outcome
    
    def decode_ec_hd(self, x: np.ndarray, detector_X: List, detector_Z: List) -> Tuple[List[int], List[int]]:
        """
        Decode EC measurements using hard-decision decoder.
        
        Returns correction_x and correction_z for both C4 logical dimensions.
        """
        mx0 = [None] * 7
        mx1 = [None] * 7
        mz0 = [None] * 7
        mz1 = [None] * 7
        
        for i in range(7):
            detx = detector_X[i]
            detz = detector_Z[i]
            result_x = self.decode_measurement_c4(x[detx[0]:detx[1]], 'x')
            result_z = self.decode_measurement_c4(x[detz[0]:detz[1]], 'z')
            mx0[i], mx1[i] = result_x[0], result_x[1]
            mz0[i], mz1[i] = result_z[0], result_z[1]
        
        correction_x = [self.decode_measurement_steane(mx0), self.decode_measurement_steane(mx1)]
        correction_z = [self.decode_measurement_steane(mz0), self.decode_measurement_steane(mz1)]
        
        return correction_x, correction_z
    
    def decode_m_hd(self, x: np.ndarray, detector_m: List) -> List[int]:
        """
        Decode final measurements using hard-decision decoder.
        
        Returns outcome as [steane_bit0, steane_bit1].
        """
        m0 = [None] * 7
        m1 = [None] * 7
        
        for i in range(7):
            det = detector_m[i]
            result = self.decode_measurement_c4(x[det[0]:det[1]])
            m0[i], m1[i] = result[0], result[1]
        
        outcome = [self.decode_measurement_steane(m0), self.decode_measurement_steane(m1)]
        
        return outcome


# =============================================================================
# C4Steane Post-Selection Implementation
# =============================================================================

class C4SteanePostSelector(PostSelector):
    """
    Post-selection functions for C4Steane codes.
    
    Implements:
    - post_selection_c4: Parity check for C4 preparation
    - post_selection_steane_l2: Steane outer code verification
    """
    
    def __init__(self, concat_code: ConcatenatedCode, decoder: C4SteaneDecoder):
        super().__init__(concat_code, decoder)
        self.c4steane_decoder = decoder
    
    def post_selection_c4(self, x: np.ndarray, detector_0prep: List[int]) -> bool:
        """
        C4 post-selection based on parity check.
        
        Returns True if sum of measurements is even (valid).
        """
        if sum(x[detector_0prep[0]:detector_0prep[1]]) % 2 == 0:
            return True
        else:
            return False
    
    def post_selection_steane_l2(self, x: np.ndarray, detector_0prep: List[int], decoder: C4SteaneDecoder) -> bool:
        """
        Steane L2 post-selection.
        
        Uses C4 decoder output to check Steane verification.
        """
        outcome = decoder.decode_measurement_c4(x[detector_0prep[0]:detector_0prep[1]])
        if outcome[0] % 2 == 1:
            return False
        else:
            return True
    
    def apply_post_selection_l1(self, sample: List[np.ndarray], 
                                 list_detector_0prep: List) -> List[np.ndarray]:
        """Apply L1 (C4) post-selection to samples."""
        for det in list_detector_0prep:
            sample = [x for x in sample if self.post_selection_c4(x, det)]
        return sample
    
    def apply_post_selection_l2(self, sample: List[np.ndarray],
                                 list_detector_0prep: List,
                                 decoder: C4SteaneDecoder) -> List[np.ndarray]:
        """
        Apply L2 (C4Steane) post-selection to samples.
        
        Structure of list_detector_0prep for L2:
        - Each entry may have length 2 (C4 style) or 15 (Steane style)
        - Length 15: indices 0-13 are C4 parity checks, index 14 is Steane verification
        """
        for a in list_detector_0prep:
            if len(a) == 2:
                sample = [x for x in sample if self.post_selection_c4(x, a)]
            elif len(a) == 15:
                for b in range(14):
                    sample = [x for x in sample if self.post_selection_c4(x, a[b][0])]
                sample = [x for x in sample if self.post_selection_steane_l2(x, a[14], decoder)]
        return sample


# =============================================================================
# C4Steane Acceptance Checker Implementation
# =============================================================================

class C4SteaneAcceptanceChecker(AcceptanceChecker):
    """
    Acceptance checker for C4Steane codes.
    
    Handles the 2D outcome array structure from C4's k=2.
    """
    
    def __init__(self, concat_code: ConcatenatedCode, decoder: C4SteaneDecoder, Q: int):
        super().__init__(concat_code, decoder)
        self.Q = Q
        self.X_propagate = [[1], [3]]
        self.Z_propagate = [[0], [2]]
    
    def accept_c4(self, x: np.ndarray, 
                   list_detector_m: List,
                   list_detector_X: List,
                   list_detector_Z: List,
                   decoder: C4SteaneDecoder) -> float:
        """
        L1 (C4) acceptance check.
        
        Returns error probability estimate for C4 level.
        """
        num_correction = 2 * self.Q
        
        outcome = np.zeros([4, 2])
        correction_x = np.zeros([num_correction, 2])
        correction_z = np.zeros([num_correction, 2])
        
        for i in range(4):
            result = decoder.decode_measurement_c4(x[list_detector_m[i][0]:list_detector_m[i][1]], 'x')
            outcome[i, :] = result
        
        for i in range(num_correction):
            corr_x = decoder.decode_measurement_c4(x[list_detector_X[i][0][0]:list_detector_X[i][0][1]], 'x')
            corr_z = decoder.decode_measurement_c4(x[list_detector_Z[i][0][0]:list_detector_Z[i][0][1]], 'z')
            correction_x[i, :] = corr_x
            correction_z[i, :] = corr_z
        
        # Apply corrections for both C4 logical dimensions
        for a in range(2):
            for i in range(num_correction):
                pos = i % 2
                for x_prop in self.X_propagate[pos]:
                    if outcome[x_prop, a] == -1:
                        continue
                    if correction_x[i, a] == 1:
                        outcome[x_prop, a] = (outcome[x_prop, a] + 1) % 2
                    if correction_x[i, a] == -1:
                        outcome[x_prop, a] = -1
                for z_prop in self.Z_propagate[pos]:
                    if outcome[z_prop, a] == -1:
                        continue
                    if correction_z[i, a] == 1:
                        outcome[z_prop, a] = (outcome[z_prop, a] + 1) % 2
                    if correction_z[i, a] == -1:
                        outcome[z_prop, a] = -1
        
        flag = 1
        for i in range(4):
            if outcome[i, 0] == 1:
                flag = 0
            if outcome[i, 0] == -1:
                flag *= 0.5
        num_p = 1 - flag
        
        return num_p
    
    def accept(self, x: np.ndarray,
               list_detector_m: List,
               list_detector_X: List,
               list_detector_Z: List,
               decoder: C4SteaneDecoder) -> float:
        """
        L2 (C4Steane) acceptance check.
        
        Returns error probability estimate for C4Steane level, or -1 if rejected.
        """
        num_correction = 2 * self.Q
        
        outcome = np.zeros([4, 2])
        correction_x = np.zeros([num_correction, 2])
        correction_z = np.zeros([num_correction, 2])
        
        # EDT post-selection: check all inner EC measurements
        for i in range(num_correction):
            for a in range(14):  # 14 = 2*7 for Steane blocks
                if decoder.decode_measurement_c4(x[list_detector_X[i][a][0][0]:list_detector_X[i][a][0][1]])[0] == -1:
                    return -1
                if decoder.decode_measurement_c4(x[list_detector_Z[i][a][0][0]:list_detector_Z[i][a][0][1]])[0] == -1:
                    return -1
        
        # Decode EC corrections for outer level
        for i in range(num_correction):
            correction_x[i, :], correction_z[i, :] = decoder.decode_ec_hd(
                x, list_detector_X[i][14], list_detector_Z[i][14]
            )
        
        # Decode final measurements
        for i in range(4):
            outcome[i, :] = decoder.decode_m_hd(x, list_detector_m[i])
        
        # Apply corrections for both dimensions
        for a in range(2):
            for i in range(num_correction):
                pos = i % 2
                for x_prop in self.X_propagate[pos]:
                    if outcome[x_prop, a] == -1:
                        continue
                    if correction_x[i, a] == 1:
                        outcome[x_prop, a] = (outcome[x_prop, a] + 1) % 2
                    if correction_x[i, a] == -1:
                        outcome[x_prop, a] = -1
                for z_prop in self.Z_propagate[pos]:
                    if outcome[z_prop, a] == -1:
                        continue
                    if correction_z[i, a] == 1:
                        outcome[z_prop, a] = (outcome[z_prop, a] + 1) % 2
                    if correction_z[i, a] == -1:
                        outcome[z_prop, a] = -1
        
        # Calculate error probability
        num_p = 0
        for a in range(2):
            flag = 1
            for i in range(4):
                if outcome[i, 0] == 1:
                    flag = 0
                if outcome[i, 0] == -1:
                    flag *= 0.5
            num_p += (1 - flag)
        
        return num_p


# =============================================================================
# C4Steane Preparation Strategy
# =============================================================================

class C4SteanePreparationStrategy(PreparationStrategy):
    """
    State preparation for C4Steane codes.
    
    Handles verified ancilla preparation at both C4 and Steane levels.
    """
    
    def __init__(self, concat_code: ConcatenatedCode, ops: C4SteaneTransversalOps):
        super().__init__(concat_code, ops)
        self.N_c4 = 4
        self.N_steane = 7
    
    @property
    def strategy_name(self) -> str:
        return "C4Steane_verified_ancilla"
    
    def append_0prep(self, circuit: stim.Circuit, loc1: int, N_prev: int, N_now: int):
        """
        Ideal |0> state preparation for C4Steane.
        """
        if N_prev == 1:
            self.ops.reset(circuit, loc1, N_now)
        else:
            for i in range(N_now):
                self.append_0prep(circuit, (loc1 + i) * N_prev, 1, N_prev)
        
        if N_now == self.N_c4:
            self.ops.append_h(circuit, (loc1 + 0) * N_prev, 1, N_prev)
            self.ops.append_cnot(circuit, (loc1 + 0) * N_prev, (loc1 + 1) * N_prev, 1, N_prev)
            self.ops.append_cnot(circuit, (loc1 + 0) * N_prev, (loc1 + 2) * N_prev, 1, N_prev)
            self.ops.append_cnot(circuit, (loc1 + 0) * N_prev, (loc1 + 3) * N_prev, 1, N_prev)
    
    def append_noisy_0prep(self, circuit: stim.Circuit, loc1: int, loc2: int,
                            N_prev: int, N_now: int, p: float,
                            detector_counter) -> List:
        """
        Noisy verified |0> state preparation for C4Steane.
        
        Returns detector information for post-selection.
        """
        if N_now == self.N_steane:
            n_now = 7
        else:
            n_now = N_now
        
        if N_prev == 1:
            self.ops.noisy_reset(circuit, loc1, N_now, p)
            self.ops.noisy_reset(circuit, loc2, N_now, p)
            detector_0prep = []
        else:
            detector_0prep = []
            for i in range(n_now):
                detector_0prep.append(self.append_noisy_0prep(
                    circuit, (loc1 + i) * N_prev, (loc1 + n_now + i) * N_prev,
                    1, N_prev, p, detector_counter
                ))
            for i in range(n_now):
                detector_0prep.append(self.append_noisy_0prep(
                    circuit, (loc2 + i) * N_prev, (loc2 + n_now + i) * N_prev,
                    1, N_prev, p, detector_counter
                ))
        
        if N_now == self.N_c4:
            # C4 verification circuit
            for i in range(4):
                circuit.append("H", loc2 + i)
            
            for i in range(N_now):
                self.ops.append_noisy_cnot(circuit, (loc2 + i) * N_prev, (loc1 + i) * N_prev, 1, N_prev, p)
            
            for i in range(N_now):
                self.ops.append_noisy_cnot(circuit, (loc1 + i) * N_prev, (loc2 + (i + 1) % N_now) * N_prev, 1, N_prev, p)
            
            detector_0prep.append(self.ops.append_noisy_m(circuit, loc2, N_prev, N_now, p, detector_counter))
            
            for i in range(N_now - 1):
                for j in range(N_now - 1):
                    if j >= i:
                        self.ops.append_cnot(circuit, (loc2 + i) * N_prev, (loc1 + j) * N_prev, 1, N_prev)
            
            return detector_0prep
        
        elif N_now == self.N_steane:
            # Steane encoding circuit
            self.ops.append_h(circuit, (loc1 + 0) * N_prev, 1, N_prev)
            self.ops.append_h(circuit, (loc1 + 1) * N_prev, 1, N_prev)
            self.ops.append_h(circuit, (loc1 + 3) * N_prev, 1, N_prev)
            
            # Encoding CNOTs with idle wait noise
            self.ops.append_noisy_cnot(circuit, (loc1 + 1) * N_prev, (loc1 + 2) * N_prev, 1, N_prev, p)
            self.ops.append_noisy_cnot(circuit, (loc1 + 3) * N_prev, (loc1 + 5) * N_prev, 1, N_prev, p)
            self.ops.append_noisy_cnot(circuit, (loc1 + 0) * N_prev, (loc1 + 4) * N_prev, 1, N_prev, p)
            # idle: qubit 6
            
            self.ops.append_noisy_cnot(circuit, (loc1 + 1) * N_prev, (loc1 + 6) * N_prev, 1, N_prev, p)
            self.ops.append_noisy_cnot(circuit, (loc1 + 0) * N_prev, (loc1 + 2) * N_prev, 1, N_prev, p)
            self.ops.append_noisy_cnot(circuit, (loc1 + 3) * N_prev, (loc1 + 4) * N_prev, 1, N_prev, p)
            # idle: qubit 5
            
            self.ops.append_noisy_cnot(circuit, (loc1 + 1) * N_prev, (loc1 + 5) * N_prev, 1, N_prev, p)
            self.ops.append_noisy_cnot(circuit, (loc1 + 4) * N_prev, (loc1 + 6) * N_prev, 1, N_prev, p)
            self.ops.append_noisy_cnot(circuit, (loc1 + 2) * N_prev, (loc2) * N_prev, 1, N_prev, p)
            # idle: qubits 0, 3
            
            self.ops.append_noisy_cnot(circuit, (loc1 + 4) * N_prev, (loc2) * N_prev, 1, N_prev, p)
            # idle: rest
            
            self.ops.append_noisy_cnot(circuit, (loc1 + 5) * N_prev, (loc2) * N_prev, 1, N_prev, p)
            # idle: rest
            
            detector_0prep.append(self.ops.append_noisy_m(circuit, (loc2) * N_prev, 1, N_prev, p, detector_counter))
            # idle: rest
            
            return [detector_0prep]
        
        return detector_0prep


# =============================================================================
# C4Steane EC Gadget
# =============================================================================

class C4SteaneECGadget(ECGadget):
    """
    Error correction gadget for C4Steane codes.
    
    Implements Knill-style teleportation-based EC.
    """
    
    def __init__(self, concat_code: ConcatenatedCode, ops: C4SteaneTransversalOps,
                 prep_strategy: C4SteanePreparationStrategy):
        super().__init__(concat_code, ops)
        self.prep_strategy = prep_strategy  # Store prep strategy directly
        self.N_c4 = 4
        self.N_steane = 7
    
    @property
    def ec_type(self) -> str:
        return "C4Steane_knill_ec"
    
    def append_noisy_ec(self, circuit: stim.Circuit, loc1: int, loc2: int,
                         loc3: int, loc4: int, N_prev: int, N_now: int,
                         p: float, detector_counter,
                         no_verification: int = -1) -> Tuple[List, List, List]:
        """
        Noisy EC gadget for C4Steane.
        
        Returns (detector_0prep, detector_Z, detector_X).
        """
        detector_0prep = []
        detector_Z = []
        detector_X = []
        
        if N_now == 1:
            return None
        
        if N_now == self.N_steane:
            n_now = 7
        else:
            n_now = N_now
        
        # Prepare ancilla states
        detector_0prep.extend(self.prep_strategy.append_noisy_0prep(
            circuit, loc2, loc4, N_prev, N_now, p, detector_counter
        ))
        detector_0prep.extend(self.prep_strategy.append_noisy_0prep(
            circuit, loc3, loc4, N_prev, N_now, p, detector_counter
        ))
        
        self.ops.append_h(circuit, loc2, N_prev, n_now)
        self.ops.append_noisy_cnot(circuit, loc2, loc3, N_prev, n_now, p)
        
        if N_prev != 1:
            # Error-detecting teleportation for inner level
            for i in range(n_now):
                detector_0prep_c4, detector_Z_c4, detector_X_c4 = self.append_noisy_ec(
                    circuit, (loc2 + i) * N_prev, (loc4 + 0) * N_prev,
                    (loc4 + 1) * N_prev, (loc4 + 2) * N_prev, 1, N_prev, p, detector_counter
                )
                self.ops.append_h(circuit, (loc2 + i) * N_prev, 1, N_prev)
                self.ops.append_cnot(circuit, (loc4 + 1) * N_prev, (loc2 + i) * N_prev, 1, N_prev)
                self.ops.append_h(circuit, (loc2 + i) * N_prev, 1, N_prev)
                self.ops.append_cnot(circuit, (loc4 + 0) * N_prev, (loc2 + i) * N_prev, 1, N_prev)
                detector_0prep.extend(detector_0prep_c4)
                detector_Z.append(detector_Z_c4)
                detector_X.append(detector_X_c4)
            
            for i in range(n_now):
                detector_0prep_c4, detector_Z_c4, detector_X_c4 = self.append_noisy_ec(
                    circuit, (loc3 + i) * N_prev, (loc4 + 0) * N_prev,
                    (loc4 + 1) * N_prev, (loc4 + 2) * N_prev, 1, N_prev, p, detector_counter
                )
                self.ops.append_h(circuit, (loc3 + i) * N_prev, 1, N_prev)
                self.ops.append_cnot(circuit, (loc4 + 1) * N_prev, (loc3 + i) * N_prev, 1, N_prev)
                self.ops.append_h(circuit, (loc3 + i) * N_prev, 1, N_prev)
                self.ops.append_cnot(circuit, (loc4 + 0) * N_prev, (loc3 + i) * N_prev, 1, N_prev)
                detector_0prep.extend(detector_0prep_c4)
                detector_Z.append(detector_Z_c4)
                detector_X.append(detector_X_c4)
        
        self.ops.append_noisy_cnot(circuit, loc1, loc2, N_prev, n_now, p)
        self.ops.append_h(circuit, loc1, N_prev, n_now)
        
        detector_Z.append(self.ops.append_noisy_m(circuit, loc1, N_prev, n_now, p, detector_counter))
        detector_X.append(self.ops.append_noisy_m(circuit, loc2, N_prev, n_now, p, detector_counter))
        
        self.ops.append_swap(circuit, loc1, loc3, N_prev, n_now)
        
        return detector_0prep, detector_Z, detector_X


# =============================================================================
# C4Steane Simulator
# =============================================================================

class C4SteaneSimulator(ConcatenatedCodeSimulator):
    """
    Simulator for C4Steane concatenated codes.
    
    Overrides base simulator with C4Steane-specific components.
    Does NOT call super().__init__ - sets up its own components.
    """
    
    def __init__(self, concat_code: ConcatenatedCode, noise_model: NoiseModel,
                 num_levels: int = 2):
        # Don't call super().__init__ - we override everything
        self.concat_code = concat_code
        self.noise_model = noise_model
        self.num_levels = num_levels
        
        # Create C4Steane-specific components
        self.ops = C4SteaneTransversalOps(concat_code)
        self.decoder = C4SteaneDecoder(concat_code)
        self.post_selector = C4SteanePostSelector(concat_code, self.decoder)
        self.prep_strategy = C4SteanePreparationStrategy(concat_code, self.ops)
        self.ec_gadget = C4SteaneECGadget(concat_code, self.ops, self.prep_strategy)
        
        self.N_c4 = 4
        self.N_steane = 7
        self.gamma = None  # Set based on noise model
    
    def set_gamma(self, p: float, error_model: str = 'a'):
        """Set gamma based on error model."""
        if error_model == 'a':
            self.gamma = p / 10
        elif error_model == 'b':
            self.gamma = p / 2
        elif error_model == 'c':
            self.gamma = p
        else:
            self.gamma = p / 10
    
    def estimate_logical_cnot_error_l1(self, p: float, num_shots: int, Q: int = 10) -> Tuple[float, float]:
        """
        Estimate logical CNOT error rate at level 1 (C4 only).
        """
        self.set_gamma(p)
        
        N_prev = 1
        N_now = self.N_c4
        
        list_detector_0prep = []
        list_detector_X = []
        list_detector_Z = []
        list_detector_m = []
        
        detector_counter = [0]
        
        circuit = stim.Circuit()
        
        # Prepare ideal Bell pairs (4 data qubits per logical)
        self.prep_strategy.append_0prep(circuit, 0, N_prev, N_now)
        self.prep_strategy.append_0prep(circuit, 8, N_prev, N_now)
        self.prep_strategy.append_0prep(circuit, 16, N_prev, N_now)
        self.prep_strategy.append_0prep(circuit, 24, N_prev, N_now)
        
        self.ops.append_h(circuit, 0, N_prev, N_now)
        self.ops.append_h(circuit, 16, N_prev, N_now)
        
        self.ops.append_cnot(circuit, 0, 8, N_prev, N_now)
        self.ops.append_cnot(circuit, 16, 24, N_prev, N_now)
        
        # Q rounds of error-free CNOT + noisy CNOT + EC
        for q in range(Q):
            self.ops.append_cnot(circuit, 0, 16, N_prev, N_now)
            self.ops.append_noisy_cnot(circuit, 0, 16, N_prev, N_now, p)
            
            detector_0prep, detector_Z, detector_X = self.ec_gadget.append_noisy_ec(
                circuit, 0, 32, 40, 48, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(detector_0prep)
            list_detector_X.append(detector_X)
            list_detector_Z.append(detector_Z)
            
            detector_0prep, detector_Z, detector_X = self.ec_gadget.append_noisy_ec(
                circuit, 16, 32, 40, 48, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(detector_0prep)
            list_detector_X.append(detector_X)
            list_detector_Z.append(detector_Z)
        
        self.ops.append_cnot(circuit, 0, 8, N_prev, N_now)
        self.ops.append_cnot(circuit, 16, 24, N_prev, N_now)
        
        self.ops.append_h(circuit, 0, N_prev, N_now)
        self.ops.append_h(circuit, 16, N_prev, N_now)
        
        list_detector_m.append(self.ops.append_m(circuit, 0, N_prev, N_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, 8, N_prev, N_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, 16, N_prev, N_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, 24, N_prev, N_now, detector_counter))
        
        # Sample
        sample = circuit.compile_detector_sampler().sample(shots=num_shots)
        
        # Post-selection
        for a in list_detector_0prep:
            sample = [x for x in sample if self.post_selector.post_selection_c4(x, a)]
        
        num = len(sample)
        if num == 0:
            return 0, 0
        
        # Create acceptance checker
        checker = C4SteaneAcceptanceChecker(self.concat_code, self.decoder, Q)
        
        err = sum([checker.accept_c4(x, list_detector_m, list_detector_X, list_detector_Z, self.decoder) 
                   for x in sample])
        
        logical_error_l1 = err / (num * Q)
        variance_l1 = err / (num * Q) ** 2
        
        return logical_error_l1, variance_l1
    
    def estimate_logical_cnot_error_l2(self, p: float, num_shots: int, Q: int = 1) -> Tuple[float, float]:
        """
        Estimate logical CNOT error rate at level 2 (C4Steane).
        """
        self.set_gamma(p)
        
        N_prev = self.N_c4
        N_now = self.N_steane
        n_now = N_now
        NN = self.N_steane
        
        list_detector_0prep = []
        list_detector_X = []
        list_detector_Z = []
        list_detector_m = []
        
        detector_counter = [0]
        
        circuit = stim.Circuit()
        
        # Prepare ideal Bell pairs (4*7=28 qubits per logical)
        self.prep_strategy.append_noisy_0prep(circuit, 0, 4 * NN, N_prev, N_now, 0, detector_counter)
        self.prep_strategy.append_noisy_0prep(circuit, NN, 4 * NN, N_prev, N_now, 0, detector_counter)
        self.prep_strategy.append_noisy_0prep(circuit, 2 * NN, 4 * NN, N_prev, N_now, 0, detector_counter)
        self.prep_strategy.append_noisy_0prep(circuit, 3 * NN, 4 * NN, N_prev, N_now, 0, detector_counter)
        
        self.ops.append_h(circuit, 0, N_prev, n_now)
        self.ops.append_h(circuit, 2 * NN, N_prev, n_now)
        
        self.ops.append_cnot(circuit, 0, NN, N_prev, n_now)
        self.ops.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, n_now)
        
        # Q rounds of error-free CNOT + noisy CNOT + EC
        for q in range(Q):
            self.ops.append_cnot(circuit, 0, 2 * NN, N_prev, n_now)
            self.ops.append_noisy_cnot(circuit, 0, 2 * NN, N_prev, n_now, p)
            
            detector_0prep, detector_Z, detector_X = self.ec_gadget.append_noisy_ec(
                circuit, 0, 4 * NN, 5 * NN, NN * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(detector_0prep)
            list_detector_X.append(detector_X)
            list_detector_Z.append(detector_Z)
            
            detector_0prep, detector_Z, detector_X = self.ec_gadget.append_noisy_ec(
                circuit, 2 * NN, 4 * NN, 5 * NN, NN * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(detector_0prep)
            list_detector_X.append(detector_X)
            list_detector_Z.append(detector_Z)
        
        self.ops.append_cnot(circuit, 0, NN, N_prev, n_now)
        self.ops.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, n_now)
        
        self.ops.append_h(circuit, 0, N_prev, n_now)
        self.ops.append_h(circuit, 2 * NN, N_prev, n_now)
        
        list_detector_m.append(self.ops.append_m(circuit, 0, N_prev, n_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, NN, N_prev, n_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, 2 * NN, N_prev, n_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, 3 * NN, N_prev, n_now, detector_counter))
        
        # Sample
        sample = circuit.compile_detector_sampler().sample(shots=num_shots)
        sample = list(sample)  # Convert to list for filtering
        
        # Post-selection for L2
        sample = self.post_selector.apply_post_selection_l2(
            sample, list_detector_0prep, self.decoder
        )
        
        # Create acceptance checker
        checker = C4SteaneAcceptanceChecker(self.concat_code, self.decoder, Q)
        
        # Filter by acceptance
        sample = [x for x in sample if checker.accept(x, list_detector_m, list_detector_X, 
                                                        list_detector_Z, self.decoder) != -1]
        
        num = len(sample)
        if num == 0:
            return 0, 0
        
        err = sum([checker.accept(x, list_detector_m, list_detector_X, list_detector_Z, self.decoder) 
                   for x in sample])
        
        # Normalization: err/(num*Q*2) for 2 logical dimensions
        logical_error_l2 = err / (num * Q * 2)
        variance_l2 = err / (num * Q * 2) ** 2
        
        return logical_error_l2, variance_l2


# =============================================================================
# Propagation Tables Factory
# =============================================================================

def create_c4steane_propagation_l2() -> PropagationTables:
    """
    Create propagation tables for C4->Steane concatenation.
    
    Note: Similar to C4C6, propagation is handled differently.
    """
    return PropagationTables(
        propagation_X=[],  # Not used in same way as Steane-Steane
        propagation_Z=[],
        propagation_m=[],
        num_ec_0prep=0
    )


# =============================================================================
# C4Steane Custom Hook Functions for Generic Simulator
# =============================================================================

def _decode_measurement_c4_steane(m: np.ndarray, m_type: str = 'x') -> List[int]:
    """
    Hard-decision decoder for C4 - standalone function for use as hook.
    
    Returns [bit0, bit1] or [-1, -1] if parity check fails.
    """
    parity = (int(m[0]) + int(m[1]) + int(m[2]) + int(m[3])) % 2
    if parity == 1:
        return [-1, -1]
    else:
        bit0 = (int(m[0]) + int(m[2])) % 2
        bit1 = (int(m[2]) + int(m[3])) % 2
        return [bit0, bit1]


def _decode_measurement_steane_outer(m: List[int], mtype: str = 'x') -> int:
    """
    Syndrome-based decoder for Steane code as outer code.
    
    Takes 7 bits from decoded C4 blocks, returns single logical bit or -1.
    """
    # Steane check matrix
    r = 3
    N = 7
    check_matrix = np.zeros([r, N])
    for i in range(r):
        for n in range(N):
            check_matrix[i, n] = ((n + 1) // (2 ** i)) % 2
    
    logical_op = [1, 1, 1, 0, 0, 0, 0]
    
    error_location = [i for i in range(len(m)) if m[i] == -1]
    num_error = len(error_location)
    
    outcome = sum([m[0], m[1], m[2]]) % 2
    
    if num_error == 2:
        detected = False
        for pq in range(4):
            p = pq // 2
            q = pq % 2
            flag = True
            for i in range(r):
                e = sum(m[:] * check_matrix[i, :]) % 2
                e = (e + p * check_matrix[i, error_location[0]] + 
                     q * check_matrix[i, error_location[1]]) % 2
                if e != 0:
                    flag = False
            if flag:
                detected = True
                break
        if not detected:
            return -1
        outcome = (outcome + p * logical_op[error_location[0]] + 
                   q * logical_op[error_location[1]]) % 2
    else:
        syndrome = 0
        for i in range(r):
            e = sum(m[:] * check_matrix[i, :]) % 2
            syndrome += e * (2 ** i)
        if syndrome == 1 or syndrome == 2 or syndrome == 3:
            outcome = (outcome + 1) % 2
    
    return outcome


def c4steane_custom_accept_l2(x: np.ndarray, list_detector_m: List,
                               list_detector_X: List, list_detector_Z: List,
                               Q: int, decoder) -> float:
    """
    Custom L2 acceptance function for C4->Steane.
    
    This function can be passed to ConcatenatedCode.custom_accept_l2_fn.
    """
    num_correction = 2 * Q
    X_propagate = [[1], [3]]
    Z_propagate = [[0], [2]]
    
    outcome = np.zeros([4, 2])
    correction_x = np.zeros([num_correction, 2])
    correction_z = np.zeros([num_correction, 2])
    
    # Decode EC corrections using hierarchical decoder (C4 inner -> Steane outer)
    for i in range(num_correction):
        # For C4->Steane, detector structure is [7 blocks of C4]
        # Each block is [start, end] for 4 qubits
        det_x = list_detector_X[i]
        det_z = list_detector_Z[i]
        
        # Decode each of 7 C4 blocks
        mx0 = [None] * 7
        mx1 = [None] * 7
        mz0 = [None] * 7
        mz1 = [None] * 7
        
        for j in range(7):
            if j < len(det_x):
                result_x = _decode_measurement_c4_steane(x[det_x[j][0]:det_x[j][1]], 'x')
                mx0[j], mx1[j] = result_x[0], result_x[1]
            if j < len(det_z):
                result_z = _decode_measurement_c4_steane(x[det_z[j][0]:det_z[j][1]], 'z')
                mz0[j], mz1[j] = result_z[0], result_z[1]
        
        # Use Steane syndrome decoder for outer code
        cx0 = _decode_measurement_steane_outer(mx0)
        cx1 = _decode_measurement_steane_outer(mx1)
        cz0 = _decode_measurement_steane_outer(mz0)
        cz1 = _decode_measurement_steane_outer(mz1)
        
        correction_x[i, 0] = cx0
        correction_x[i, 1] = cx1
        correction_z[i, 0] = cz0
        correction_z[i, 1] = cz1
    
    # Decode final measurements
    for i in range(4):
        m0 = [None] * 7
        m1 = [None] * 7
        
        for j in range(7):
            if j < len(list_detector_m[i]):
                det = list_detector_m[i][j]
                result = _decode_measurement_c4_steane(x[det[0]:det[1]], 'x')
                m0[j], m1[j] = result[0], result[1]
        
        outcome[i, 0] = _decode_measurement_steane_outer(m0)
        outcome[i, 1] = _decode_measurement_steane_outer(m1)
    
    # Apply corrections
    for a in range(2):
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
    
    # Count errors
    num_errors = 0
    for a in range(2):
        flag = 1
        for i in range(4):
            if outcome[i, 0] == 1:
                flag = 0
            if outcome[i, 0] == -1:
                flag *= 0.5
        num_errors += (1 - flag)
    
    return num_errors


def c4steane_custom_post_selection_l2(x: np.ndarray, list_detector_0prep: List,
                                       list_detector_0prep_l2: List, decoder) -> bool:
    """
    Custom L2 post-selection function for C4->Steane.
    
    This function can be passed to ConcatenatedCode.custom_post_selection_l2_fn.
    """
    for a in list_detector_0prep:
        if len(a) == 1:
            # Single C4 prep detector
            if sum(x[a[0][0]:a[0][1]]) % 2 != 0:
                return False
        elif len(a) == 2:
            # Two-element detector range
            if isinstance(a[0], int):
                if sum(x[a[0]:a[1]]) % 2 != 0:
                    return False
        elif len(a) == 7:
            # Steane L2 prep detector (7 C4 blocks)
            for i in range(7):
                block_result = _decode_measurement_c4_steane(x[a[i][0]:a[i][1]])
                if block_result[0] == -1:
                    return False
    return True


def create_concatenated_c4steane(num_levels: int) -> ConcatenatedCode:
    """
    Create concatenated C4->Steane code.
    
    Args:
        num_levels: 1 for just C4, 2 for C4->Steane
    
    Returns:
        ConcatenatedCode for C4Steane with custom hooks for generic simulator
    """
    c4 = create_c4_code()
    
    if num_levels == 1:
        return ConcatenatedCode(
            levels=[c4],
            name="C4",
        )
    
    steane = create_steane_outer_code()
    return ConcatenatedCode(
        levels=[c4, steane],
        name="C4Steane",
        propagation_tables={2: create_c4steane_propagation_l2()},
        custom_accept_l2_fn=c4steane_custom_accept_l2,
        custom_post_selection_l2_fn=c4steane_custom_post_selection_l2,
    )


# =============================================================================
# Factory Function
# =============================================================================

def create_c4steane_simulator(num_levels: int = 2, 
                               noise_model: Optional[NoiseModel] = None) -> C4SteaneSimulator:
    """
    Create a C4Steane simulator with appropriate configuration.
    
    Args:
        num_levels: Number of concatenation levels (1=C4 only, 2=C4Steane)
        noise_model: Optional noise model (not used, noise is explicit in circuit)
    
    Returns:
        Configured C4SteaneSimulator instance
    """
    # Create concatenated code
    concat_code = create_concatenated_c4steane(num_levels)
    
    # Note: noise_model is not used - noise is added explicitly like original
    return C4SteaneSimulator(concat_code, noise_model, num_levels)


# =============================================================================
# Comparison Function
# =============================================================================

def compare_with_original(p: float, num_shots: int = 10000, level: int = 1) -> Dict:
    """
    Compare v10 C4Steane results with original implementation.
    
    Args:
        p: Physical error rate
        num_shots: Number of Monte Carlo shots
        level: 1 for C4 only, 2 for C4Steane
    
    Returns:
        Dictionary with comparison results
    """
    import sys
    import os
    
    # Create v10 simulator
    simulator = create_c4steane_simulator(num_levels=level)
    
    if level == 1:
        v10_error, v10_var = simulator.estimate_logical_cnot_error_l1(p, num_shots)
    else:
        v10_error, v10_var = simulator.estimate_logical_cnot_error_l2(p, num_shots)
    
    return {
        'p': p,
        'level': level,
        'v10_error': v10_error,
        'v10_variance': v10_var,
        'num_shots': num_shots,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 5:
        print("Usage: python concatenated_css_v10_c4steane.py <output_file> <num_shots> <case> <p> <error_model> <level>")
        sys.exit(1)
    
    output_file_name = str(sys.argv[1])
    num_shots = int(sys.argv[2])
    case_number = int(sys.argv[3])
    p = float(sys.argv[4])
    error_model = str(sys.argv[5]) if len(sys.argv) > 5 else 'a'
    level = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    
    simulator = create_c4steane_simulator(num_levels=level)
    simulator.set_gamma(p, error_model)
    
    if level == 1:
        logical_error, variance = simulator.estimate_logical_cnot_error_l1(p, num_shots)
    else:
        logical_error, variance = simulator.estimate_logical_cnot_error_l2(p, num_shots)
    
    with open(output_file_name, 'a') as file:
        output_data = {
            'physical_error': p,
            'logical_error': logical_error,
            'variance': variance
        }
        file.write('"case' + str(case_number) + '":' + json.dumps(output_data) + ',\n')
    
    print(f"Level {level} logical error: {logical_error:.6e} (variance: {variance:.6e})")
