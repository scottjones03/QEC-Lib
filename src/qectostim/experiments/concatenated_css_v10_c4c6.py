"""
Concatenated CSS Code Simulator v10 - C4C6 Code Implementation
─────────────────────────────────────────────────────────────────────────────

This module provides C4C6-specific implementations for the concatenated CSS
code simulator. It imports generic base classes from concatenated_css_v10.py
and provides exact matching to concatenated_c4c6.py ground truth.

C4 [[4,2,2]] Code:
- Actually encodes 2 logical qubits in 4 physical
- Treated as [[4,1,2]] by fixing one logical qubit
- Decoder returns 2 bits: [bit0, bit1]
- Stabilizers: Hx = Hz = [[1,1,1,1]]

C6 [[6,1,2]] Code:
- Implemented as 3 C4 blocks with majority voting
- Each block contributes 2 syndrome bits
- 3-block majority voting for error correction

Key differences from Steane:
1. k=2 for C4 means decoder returns 2-element arrays
2. C6 uses 3-block majority voting, not syndrome decoding
3. accept functions iterate over 2 dimensions
4. Different normalization: err/(num*Q*2)

Usage:
    from concatenated_css_v10_c4c6 import create_c4c6_simulator
    
    simulator = create_c4c6_simulator(num_levels=2, noise_model=noise_model)
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
# C4C6 Transversal Operations - Extends base with C4-specific H gate
# =============================================================================

class C4C6TransversalOps(TransversalOps):
    """
    Transversal operations extended for C4C6 codes.
    
    Key additions:
    - append_h: Overridden to include SWAP for C4 structure
    - append_h_c4: Explicit H gate with SWAP for C4 structure
    - reset: Reset qubits (delegated to PhysicalOps)
    - noisy_reset: Reset with noise (delegated to PhysicalOps)
    """
    
    def __init__(self, concat_code: ConcatenatedCode):
        super().__init__(concat_code)
        self.N_c4 = 4
    
    def append_h(self, circuit: stim.Circuit, loc: int, N_prev: int, N_now: int):
        """
        Transversal Hadamard for C4C6 with SWAP gates.
        
        Overrides base class to add SWAP gates required by C4's logical H.
        - When N_now=4 and N_prev=1: add SWAP(1,2) after H
        - When N_now=3 and N_prev=4: add SWAP pattern for each C4 block
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
            if N_now == 3:
                # C6 level: apply SWAP pattern to each C4 block
                for i in range(3):
                    circuit.append("SWAP", [(loc + i) * N_prev + 1, (loc + i) * N_prev + 3])
                    circuit.append("SWAP", [(loc + i) * N_prev + 1, (loc + i) * N_prev + 2])
    
    def append_h_c4(self, circuit: stim.Circuit, loc: int, N_prev: int, N_now: int):
        """
        Transversal Hadamard for C4 with SWAP gates.
        
        C4's logical H requires SWAP(1,2) after physical H gates when N_now=4.
        For C6 level (N_now=3), apply H to each C4 block with its SWAP pattern.
        """
        if N_prev == 1:
            for i in range(N_now):
                circuit.append("H", loc + i)
            if N_now == 4:
                # C4 requires SWAP after H
                circuit.append("SWAP", [loc + 1, loc + 2])
        else:
            for i in range(N_now):
                self.append_h_c4(circuit, (loc + i) * N_prev, 1, N_prev)
            if N_now == 3:
                # C6 level: apply SWAP pattern to each C4 block
                for i in range(3):
                    circuit.append("SWAP", [(loc + i) * N_prev + 1, (loc + i) * N_prev + 3])
                    circuit.append("SWAP", [(loc + i) * N_prev + 1, (loc + i) * N_prev + 2])
    
    def reset(self, circuit: stim.Circuit, loc: int, n: int):
        """Reset n qubits starting at loc."""
        PhysicalOps.reset(circuit, loc, n)
    
    def noisy_reset(self, circuit: stim.Circuit, loc: int, n: int, p: float):
        """Reset n qubits with X noise."""
        PhysicalOps.noisy_reset(circuit, loc, n, p)


# =============================================================================
# C4 and C6 Code Factory Functions
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
    # Lz1 = [1,0,1,0], Lx1 = [1,1,0,0]
    # Lz2 = [0,0,1,1], Lx2 = [0,1,0,1]
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
        h_qubits=[0],  # Not used for Bell-pair prep, but kept for compatibility
        encoding_cnots=[(0, 1), (0, 2), (0, 3)],  # Not used for Bell-pair prep
        encoding_cnot_rounds=[
            [(0, 1)],
            [(0, 2)],
            [(0, 3)],
        ],
        verification_qubits=[],  # C4 uses parity check verification
        uses_bellpair_prep=True,  # CRITICAL: Use Bell-pair protocol for k>1 codes
    )


def create_c6_code() -> CSSCode:
    """
    Create the [[6,1,2]] C6 code.
    
    C6 is implemented as 3 C4 blocks with special decoding.
    The outer code for C4->C6 concatenation operates on 3 blocks (n_now=3).
    
    Uses Bell-pair preparation at L2 (matching original C4C6 behavior).
    """
    # C6 has 3 stabilizers (one per block pair)
    Hz = np.array([
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
    ])
    Hx = np.array([
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
    ])
    Lz = np.array([1, 0, 1, 0, 1, 0])
    Lx = np.array([1, 1, 1, 1, 1, 1])
    
    return CSSCode(
        name="C6",
        n=6,  # 3 blocks * 2 (since each C4 block contributes 2 bits)
        k=1,
        d=2,
        Hz=Hz,
        Hx=Hx,
        logical_z_ops=[Lz],  # k=1: single logical Z operator
        logical_x_ops=[Lx],  # k=1: single logical X operator
        h_qubits=[0, 2, 4],
        encoding_cnots=[
            (0, 1), (2, 3), (4, 5),
        ],
        verification_qubits=[],
        transversal_block_count=3,  # C6 operates on 3 blocks of C4
        uses_bellpair_prep=True,  # C6 L2 prep also uses Bell-pair protocol
    )


def create_c4c6_propagation_l2() -> PropagationTables:
    """
    Create propagation tables for C4->C6 concatenation.
    
    Note: C4C6 doesn't use complex propagation like Steane.
    The no_ec flag in original controls whether EDT is used.
    """
    return PropagationTables(
        propagation_X=[],  # Not used in same way as Steane
        propagation_Z=[],
        propagation_m=[],
        num_ec_0prep=0  # C4 doesn't have EC during prep like Steane
    )


# =============================================================================
# C4C6 Custom Hook Functions for Generic Simulator
# =============================================================================

def _decode_measurement_c4(m: np.ndarray, m_type: str = 'x') -> List[int]:
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


def _decode_measurement_c6(m: List, m_type: str = 'x') -> List[int]:
    """
    Hard-decision decoder for C6 using 3-block majority voting - standalone function.
    
    Input m is either:
    - 3 pairs: [[b0,b1], [b2,b3], [b4,b5]]
    - Or flat list: [M0, M1, M2, M3, M4, M5]
    
    Returns [bit0, bit1] or [-1, -1] if >1 block failed.
    """
    # Convert to flat list if needed
    if len(m) == 3 and isinstance(m[0], list):
        M = [m[0][0], m[0][1], m[1][0], m[1][1], m[2][0], m[2][1]]
    else:
        M = list(m)
    
    # Count failed blocks (where first element is -1)
    failed = [M[0], M[2], M[4]].count(-1)
    
    if failed > 1:
        return [-1, -1]
    elif M[0] == -1:
        return [(M[2] + M[3] + M[5]) % 2, (M[3] + M[4]) % 2]
    elif M[2] == -1:
        return [(M[1] + M[4] + M[5]) % 2, (M[0] + M[5]) % 2]
    elif M[4] == -1:
        return [(M[0] + M[1] + M[3]) % 2, (M[1] + M[2]) % 2]
    else:
        parity1 = (M[0] + M[1] + M[2] + M[5]) % 2
        parity2 = (M[0] + M[3] + M[4] + M[5]) % 2
        if parity1 == 1 or parity2 == 1:
            return [-1, -1]
        else:
            return [(M[2] + M[3] + M[5]) % 2, (M[3] + M[4]) % 2]


def c4c6_custom_accept_l2(x: np.ndarray, list_detector_m: List,
                          list_detector_X: List, list_detector_Z: List,
                          Q: int, decoder, no_ec: bool = False) -> float:
    """
    Custom L2 acceptance function for C4->C6 - matches original accept_c6.
    
    This function can be passed to ConcatenatedCode.custom_accept_l2_fn.
    """
    num_correction = 2 * Q
    X_propagate = [[1], [3]]
    Z_propagate = [[0], [2]]
    
    outcome = np.zeros([4, 2])
    correction_x = np.zeros([num_correction, 2])
    correction_z = np.zeros([num_correction, 2])
    
    # Post-selection at error-detecting teleportation
    if not no_ec:
        for i in range(num_correction):
            for a in range(6):  # 6 EDT gadgets (3 per ancilla)
                result_x = _decode_measurement_c4(
                    x[list_detector_X[i][a][0][0]:list_detector_X[i][a][0][1]]
                )
                if result_x[0] == -1:
                    return -1
                result_z = _decode_measurement_c4(
                    x[list_detector_Z[i][a][0][0]:list_detector_Z[i][a][0][1]]
                )
                if result_z[0] == -1:
                    return -1
    
    # Decode EC corrections using hierarchical decoder
    for i in range(num_correction):
        if not no_ec:
            det_x = list_detector_X[i][6]
            det_z = list_detector_Z[i][6]
        else:
            det_x = list_detector_X[i][0]
            det_z = list_detector_Z[i][0]
        
        # Decode each of 3 C4 blocks
        mx = [_decode_measurement_c4(x[det_x[j][0]:det_x[j][1]], 'x') for j in range(3)]
        mz = [_decode_measurement_c4(x[det_z[j][0]:det_z[j][1]], 'z') for j in range(3)]
        
        cx = _decode_measurement_c6(mx, 'x')
        cz = _decode_measurement_c6(mz, 'z')
        
        correction_x[i, 0] = cx[0]
        correction_x[i, 1] = cx[1]
        correction_z[i, 0] = cz[0]
        correction_z[i, 1] = cz[1]
    
    # Decode final measurements
    for i in range(4):
        m = [_decode_measurement_c4(x[list_detector_m[i][j][0]:list_detector_m[i][j][1]], 'x') 
             for j in range(3)]
        result = _decode_measurement_c6(m, 'x')
        outcome[i, 0] = result[0]
        outcome[i, 1] = result[1]
    
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
    
    # Count errors (matching original "bug" where it always checks dimension 0)
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


def c4c6_custom_post_selection_l2(x: np.ndarray, list_detector_0prep: List,
                                   list_detector_0prep_l2: List, decoder) -> bool:
    """
    Custom L2 post-selection function for C4->C6.
    
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
        elif len(a) == 3:
            # C6 prep detector (3 blocks)
            outcome = []
            for i in range(3):
                block_result = _decode_measurement_c4(x[a[i][0]:a[i][1]])
                if block_result[0] == -1:
                    return False
                outcome.append(block_result)
            
            # Check parity consistency across blocks
            parity0 = (outcome[0][0] + outcome[1][0] + outcome[2][0]) % 2
            parity1 = (outcome[0][1] + outcome[1][1] + outcome[2][1]) % 2
            if parity0 != 0 or parity1 != 0:
                return False
    return True


def create_concatenated_c4c6(num_levels: int) -> ConcatenatedCode:
    """
    Create concatenated C4->C6 code.
    
    Args:
        num_levels: 1 for just C4, 2 for C4->C6
    
    Returns:
        ConcatenatedCode for C4C6 with custom hooks for generic simulator
    """
    c4 = create_c4_code()
    
    if num_levels == 1:
        return ConcatenatedCode(
            levels=[c4],
            name="C4",
            propagation_tables={}
        )
    else:
        c6 = create_c6_code()
        prop_tables = {1: create_c4c6_propagation_l2()}
        
        # Create wrapper functions that don't need the no_ec parameter
        # (generic simulator doesn't support no_ec mode)
        def accept_l2_wrapper(x, det_m, det_X, det_Z, Q, decoder):
            return c4c6_custom_accept_l2(x, det_m, det_X, det_Z, Q, decoder, no_ec=False)
        
        return ConcatenatedCode(
            levels=[c4, c6],
            name="C4->C6",
            propagation_tables=prop_tables,
            custom_accept_l2_fn=accept_l2_wrapper,
            custom_post_selection_l2_fn=c4c6_custom_post_selection_l2,
        )


# =============================================================================
# C4C6 Decoder - Exact match to original decode_measurement_c4/c6
# =============================================================================

class C4C6Decoder(Decoder):
    """
    Decoder for C4 and C6 codes matching original concatenated_c4c6.py exactly.
    
    Key difference from generic decoder:
    - decode_measurement_c4 returns [bit0, bit1] (2 values)
    - decode_measurement_c6 uses 3-block majority voting
    """
    
    def __init__(self, concat_code: ConcatenatedCode):
        super().__init__(concat_code)
        self.code = concat_code.code_at_level(0)
        self.n = self.code.n
    
    def decode_measurement_c4(self, m: np.ndarray, m_type: str = 'x') -> List[int]:
        """
        Hard-decision decoder for C4 - EXACT match to original.
        
        Returns [bit0, bit1] or [-1, -1] if parity check fails.
        """
        parity = (int(m[0]) + int(m[1]) + int(m[2]) + int(m[3])) % 2
        if parity == 1:
            return [-1, -1]
        else:
            bit0 = (int(m[0]) + int(m[2])) % 2
            bit1 = (int(m[2]) + int(m[3])) % 2
            return [bit0, bit1]
    
    def decode_measurement_c6(self, m: List, m_type: str = 'x') -> List[int]:
        """
        Hard-decision decoder for C6 using 3-block majority voting.
        
        Input m is either:
        - 3 pairs: [[b0,b1], [b2,b3], [b4,b5]]
        - Or flat list: [M0, M1, M2, M3, M4, M5]
        
        Returns [bit0, bit1] or [-1, -1] if >1 block failed.
        """
        # Convert to flat list if needed
        if len(m) == 3 and isinstance(m[0], list):
            M = [m[0][0], m[0][1], m[1][0], m[1][1], m[2][0], m[2][1]]
        else:
            M = list(m)
        
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
    
    def decode_ec_hd(self, x: np.ndarray, detector_X: List, detector_Z: List,
                     correction_x_prev: List = None, correction_z_prev: List = None) -> Tuple:
        """
        Hierarchical EC decoding for C4->C6.
        
        Decodes each of 3 C4 blocks, then applies C6 majority voting.
        
        Returns:
            (correction_x, correction_z, correction_x_next, correction_z_next)
            where correction_x/z are 2-element lists [bit0, bit1]
        """
        mx = [None] * 3
        mz = [None] * 3
        
        for i in range(3):
            detx = detector_X[i]
            detz = detector_Z[i]
            mx[i] = self.decode_measurement_c4(x[detx[0]:detx[1]], 'x')
            mz[i] = self.decode_measurement_c4(x[detz[0]:detz[1]], 'z')
        
        correction_x = self.decode_measurement_c6(mx, 'x')
        correction_z = self.decode_measurement_c6(mz, 'z')
        
        # C4C6 doesn't propagate corrections the same way as Steane
        # Return empty lists for next corrections
        return correction_x, correction_z, [0, 0], [0, 0]
    
    def decode_m_hd(self, x: np.ndarray, detector_m: List, 
                    correction: List = None) -> List[int]:
        """
        Decode final measurement using C4->C6 hierarchy.
        
        Returns 2-element list [bit0, bit1].
        """
        m = [None] * 3
        for i in range(3):
            det = detector_m[i]
            m[i] = self.decode_measurement_c4(x[det[0]:det[1]], 'x')
        
        outcome = self.decode_measurement_c6(m, 'x')
        
        # Apply correction if provided
        if correction is not None and correction[0] != -1 and outcome[0] != -1:
            outcome = [(outcome[0] + correction[0]) % 2, 
                       (outcome[1] + correction[1]) % 2]
        
        return outcome


# =============================================================================
# C4C6 Post Selector
# =============================================================================

class C4C6PostSelector(PostSelector):
    """
    Post-selection for C4C6 matching original exactly.
    """
    
    def __init__(self, concat_code: ConcatenatedCode, decoder: C4C6Decoder):
        super().__init__(concat_code, decoder)
        self.c4c6_decoder = decoder
    
    def post_selection_c4(self, x: np.ndarray, detector_0prep: List) -> bool:
        """Post-selection for C4 - checks parity is even."""
        if sum(x[detector_0prep[0]:detector_0prep[1]]) % 2 == 0:
            return True
        return False
    
    def post_selection_c6(self, x: np.ndarray, detector_0prep: List) -> bool:
        """Post-selection for C6 - checks all 3 C4 blocks pass."""
        outcome = [None] * 3
        for i in range(3):
            outcome[i] = self.c4c6_decoder.decode_measurement_c4(
                x[detector_0prep[i][0]:detector_0prep[i][1]]
            )
            if outcome[i][0] == -1:
                return False
        
        # Check parity consistency across blocks
        parity0 = (outcome[0][0] + outcome[1][0] + outcome[2][0]) % 2
        parity1 = (outcome[0][1] + outcome[1][1] + outcome[2][1]) % 2
        if parity0 == 0 and parity1 == 0:
            return True
        return False
    
    def post_selection_l1(self, x: np.ndarray, list_detector_0prep: List) -> bool:
        """Level-1 post-selection for C4."""
        for a in list_detector_0prep:
            if not self.post_selection_c4(x, a):
                return False
        return True
    
    def post_selection_l2(self, x: np.ndarray, list_detector_0prep: List,
                          list_detector_0prep_l2: List = None, 
                          list_detector_X: List = None,
                          list_detector_Z: List = None, Q: int = 1) -> bool:
        """
        Level-2 post-selection for C4->C6.
        
        From original concatenated_c4c6.py estimate_logical_cnot_error_l2:
        - Checks each 0prep detector with appropriate function
        - len(a)==1: single C4 prep detector
        - len(a)==2: two-element detector range
        - len(a)==3: C6 prep detector (3 blocks)
        """
        for a in list_detector_0prep:
            if len(a) == 1:
                if not self.post_selection_c4(x, a[0]):
                    return False
            elif len(a) == 2:
                if not self.post_selection_c4(x, a):
                    return False
            elif len(a) == 3:
                if not self.post_selection_c6(x, a):
                    return False
        return True


# =============================================================================
# C4C6 Acceptance Checker
# =============================================================================

class C4C6AcceptanceChecker(AcceptanceChecker):
    """
    Acceptance checking for C4C6 matching original exactly.
    
    Key differences from generic:
    - outcome array is [4, 2] not [4]
    - correction arrays are [num_correction, 2]
    - Iterates over both dimensions (a in range(2))
    """
    
    def __init__(self, concat_code: ConcatenatedCode, decoder: C4C6Decoder, no_ec: bool = False):
        super().__init__(concat_code, decoder)
        self.c4c6_decoder = decoder
        self.no_ec = no_ec  # Flag for error-detecting teleportation
    
    def accept_c4(self, x: np.ndarray, list_detector_m: List,
                  list_detector_X: List, list_detector_Z: List, Q: int) -> float:
        """
        Level-1 acceptance for C4 - EXACT match to original accept_c4.
        
        Returns fractional error count (0, 0.5, 1, 1.5, or 2).
        """
        num_correction = 2 * Q
        X_propagate = [[1], [3]]
        Z_propagate = [[0], [2]]
        
        outcome = np.zeros([4, 2])
        correction_x = np.zeros([num_correction, 2])
        correction_z = np.zeros([num_correction, 2])
        
        # Decode final measurements
        for i in range(4):
            result = self.c4c6_decoder.decode_measurement_c4(
                x[list_detector_m[i][0]:list_detector_m[i][1]], 'x'
            )
            outcome[i, 0] = result[0]
            outcome[i, 1] = result[1]
        
        # Decode EC corrections
        for i in range(num_correction):
            cx = self.c4c6_decoder.decode_measurement_c4(
                x[list_detector_X[i][0][0]:list_detector_X[i][0][1]], 'x'
            )
            cz = self.c4c6_decoder.decode_measurement_c4(
                x[list_detector_Z[i][0][0]:list_detector_Z[i][0][1]], 'z'
            )
            correction_x[i, 0] = cx[0]
            correction_x[i, 1] = cx[1]
            correction_z[i, 0] = cz[0]
            correction_z[i, 1] = cz[1]
        
        # Apply corrections with propagation
        for a in range(2):  # Both dimensions!
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
        
        # Count errors across both dimensions
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
    
    def accept_l1(self, x: np.ndarray, list_detector_m: List,
                  list_detector_X: List, list_detector_Z: List, Q: int) -> float:
        """Level-1 acceptance wrapper."""
        return self.accept_c4(x, list_detector_m, list_detector_X, list_detector_Z, Q)
    
    def accept_c6(self, x: np.ndarray, list_detector_m: List,
                  list_detector_X: List, list_detector_Z: List, Q: int) -> float:
        """
        Level-2 acceptance for C4->C6 - EXACT match to original accept.
        
        Returns -1 if should be rejected (EDT failure), else fractional error.
        """
        num_correction = 2 * Q
        X_propagate = [[1], [3]]
        Z_propagate = [[0], [2]]
        
        outcome = np.zeros([4, 2])
        correction_x = np.zeros([num_correction, 2])
        correction_z = np.zeros([num_correction, 2])
        
        # Post-selection at error-detecting teleportation
        if not self.no_ec:
            for i in range(num_correction):
                for a in range(6):  # 6 EDT gadgets (3 per ancilla)
                    result_x = self.c4c6_decoder.decode_measurement_c4(
                        x[list_detector_X[i][a][0][0]:list_detector_X[i][a][0][1]]
                    )
                    if result_x[0] == -1:
                        return -1
                    result_z = self.c4c6_decoder.decode_measurement_c4(
                        x[list_detector_Z[i][a][0][0]:list_detector_Z[i][a][0][1]]
                    )
                    if result_z[0] == -1:
                        return -1
        
        # Decode EC corrections using hierarchical decoder
        for i in range(num_correction):
            if not self.no_ec:
                cx, cz = self.c4c6_decoder.decode_ec_hd(
                    x, list_detector_X[i][6], list_detector_Z[i][6]
                )[:2]
            else:
                cx, cz = self.c4c6_decoder.decode_ec_hd(
                    x, list_detector_X[i][0], list_detector_Z[i][0]
                )[:2]
            correction_x[i, 0] = cx[0]
            correction_x[i, 1] = cx[1]
            correction_z[i, 0] = cz[0]
            correction_z[i, 1] = cz[1]
        
        # Decode final measurements
        for i in range(4):
            result = self.c4c6_decoder.decode_m_hd(x, list_detector_m[i])
            outcome[i, 0] = result[0]
            outcome[i, 1] = result[1]
        
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
    
    def accept_l2(self, x: np.ndarray, list_detector_m: List,
                  list_detector_X: List, list_detector_Z: List, Q: int) -> float:
        """Level-2 acceptance wrapper."""
        return self.accept_c6(x, list_detector_m, list_detector_X, list_detector_Z, Q)


# =============================================================================
# C4C6 Simulator - Extends generic with C4C6-specific behavior
# =============================================================================

class C4C6Simulator(ConcatenatedCodeSimulator):
    """
    C4C6-specific simulator matching concatenated_c4c6.py exactly.
    
    Key differences:
    - Uses C4C6Decoder for 2-bit decoding
    - Uses C4C6PostSelector with C4/C6 specific checks
    - Uses C4C6AcceptanceChecker with 2D outcomes
    - Different normalization: err/(num*Q*2)
    """
    
    def __init__(self, concat_code: ConcatenatedCode, noise_model: NoiseModel,
                 no_ec: bool = False):
        # Don't call super().__init__ - we override everything
        self.concat_code = concat_code
        self.noise_model = noise_model
        self.no_ec = no_ec
        
        # Create C4C6-specific components
        self.ops = C4C6TransversalOps(concat_code)
        self.prep = C4C6PreparationStrategy(concat_code, self.ops)
        self.decoder = C4C6Decoder(concat_code)
        self.ec = C4C6ECGadget(concat_code, self.ops, self.prep, no_ec=no_ec)
        self.post_selector = C4C6PostSelector(concat_code, self.decoder)
        self.acceptance = C4C6AcceptanceChecker(concat_code, self.decoder, no_ec=no_ec)
    
    def estimate_logical_cnot_error_l1(self, p: float, num_shots: int,
                                        Q: int = 10) -> Tuple[float, float]:
        """
        Estimate L1 CNOT error rate - exact match to original.
        
        Uses err/(num*Q*2) normalization for 2 logical dimensions.
        """
        N_prev = 1
        N_now = self.concat_code.code_at_level(0).n  # 4 for C4
        NN = 2 * N_now  # 8
        
        list_detector_0prep = []
        list_detector_X = []
        list_detector_Z = []
        list_detector_m = []
        
        circuit = stim.Circuit()
        detector_counter = [0]
        
        # Prepare ideal Bell pairs (4 blocks of 4 qubits each)
        self.prep.append_0prep(circuit, 0, N_prev, N_now)
        self.prep.append_0prep(circuit, NN, N_prev, N_now)
        self.prep.append_0prep(circuit, 2 * NN, N_prev, N_now)
        self.prep.append_0prep(circuit, 3 * NN, N_prev, N_now)
        
        self.ops.append_h(circuit, 0, N_prev, N_now)
        self.ops.append_h(circuit, 2 * NN, N_prev, N_now)
        
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
        
        self.ops.append_h(circuit, 0, N_prev, N_now)
        self.ops.append_h(circuit, 2 * NN, N_prev, N_now)
        
        # Measure
        list_detector_m.append(self.ops.append_m(circuit, 0, N_prev, N_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, NN, N_prev, N_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, 2 * NN, N_prev, N_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, 3 * NN, N_prev, N_now, detector_counter))
        
        # Sample
        samples = circuit.compile_detector_sampler().sample(shots=num_shots)
        
        # Post-selection
        samples = [x for x in samples if self.post_selector.post_selection_l1(x, list_detector_0prep)]
        num = len(samples)
        
        # Count errors
        err = sum([self.acceptance.accept_c4(x, list_detector_m, list_detector_X,
                                              list_detector_Z, Q) for x in samples])
        
        print(p, num, err)
        
        if num > 0:
            # KEY: divide by Q*2 for 2 logical dimensions
            logical_error = err / (num * Q * 2)
            variance = err / (num * Q * 2) ** 2
        else:
            logical_error = variance = 0
        
        return logical_error, variance
    
    def estimate_logical_cnot_error_l2(self, p: float, num_shots: int,
                                        Q: int = 10) -> Tuple[float, float]:
        """
        Estimate L2 CNOT error rate - exact match to original.
        """
        c4 = self.concat_code.code_at_level(0)
        N_prev = c4.n  # 4
        N_now = 6  # C6 has 6 "qubits" but operates on n_now=3 blocks
        n_now = 3  # Number of C4 blocks in C6
        
        list_detector_0prep = []
        list_detector_X = []
        list_detector_Z = []
        list_detector_m = []
        
        circuit = stim.Circuit()
        detector_counter = [0]
        
        # Prepare ideal Bell pairs at L2 (using noisy_0prep with p=0)
        self.prep.append_noisy_0prep(circuit, 0, 24, N_prev, N_now, 0, detector_counter)
        self.prep.append_noisy_0prep(circuit, 6, 24, N_prev, N_now, 0, detector_counter)
        self.prep.append_noisy_0prep(circuit, 12, 24, N_prev, N_now, 0, detector_counter)
        self.prep.append_noisy_0prep(circuit, 18, 24, N_prev, N_now, 0, detector_counter)
        
        # H on blocks 0 and 2
        self.ops.append_h(circuit, 0, N_prev, n_now)
        self.ops.append_h(circuit, 12, N_prev, n_now)
        
        # Create Bell pairs between blocks
        self.ops.append_cnot(circuit, 0, 6, N_prev, n_now)
        self.ops.append_cnot(circuit, 12, 18, N_prev, n_now)
        
        # Q rounds of CNOT + EC
        for q in range(Q):
            self.ops.append_cnot(circuit, 0, 12, N_prev, n_now)
            self.ops.append_noisy_cnot(circuit, 0, 12, N_prev, n_now, p)
            
            result = self.ec.append_noisy_ec(
                circuit, 0, 24, 30, 36, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_X.append(result[2])
            list_detector_Z.append(result[1])
            
            result = self.ec.append_noisy_ec(
                circuit, 12, 24, 30, 36, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_X.append(result[2])
            list_detector_Z.append(result[1])
        
        # Undo Bell pairs
        self.ops.append_cnot(circuit, 0, 6, N_prev, n_now)
        self.ops.append_cnot(circuit, 12, 18, N_prev, n_now)
        
        self.ops.append_h(circuit, 0, N_prev, n_now)
        self.ops.append_h(circuit, 12, N_prev, n_now)
        
        # Measure (returns list of 3 detector ranges for C6)
        list_detector_m.append(self.ops.append_m(circuit, 0, N_prev, n_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, 6, N_prev, n_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, 12, N_prev, n_now, detector_counter))
        list_detector_m.append(self.ops.append_m(circuit, 18, N_prev, n_now, detector_counter))
        
        # Sample
        samples = circuit.compile_detector_sampler().sample(shots=num_shots)
        
        # Post-selection
        samples = [x for x in samples if self.post_selector.post_selection_l2(
            x, list_detector_0prep, None, list_detector_X, list_detector_Z, Q
        )]
        
        # EDT post-selection
        samples = [x for x in samples if self.acceptance.accept_c6(
            x, list_detector_m, list_detector_X, list_detector_Z, Q
        ) != -1]
        
        num = len(samples)
        
        # Count errors
        err = sum([self.acceptance.accept_c6(x, list_detector_m, list_detector_X,
                                              list_detector_Z, Q) for x in samples])
        
        print(p, num, err)
        
        if num > 0:
            logical_error = err / (num * Q * 2)
            variance = err / (num * Q * 2) ** 2
        else:
            logical_error = variance = 0
        
        return logical_error, variance


# =============================================================================
# C4C6 Preparation Strategy
# =============================================================================

class C4C6PreparationStrategy(PreparationStrategy):
    """
    Preparation strategy for C4C6 matching original exactly.
    """
    
    def __init__(self, concat_code: ConcatenatedCode, ops: 'C4C6TransversalOps'):
        super().__init__(concat_code, ops)
        self.N_c4 = 4
        self.N_c6 = 6
    
    @property
    def strategy_name(self) -> str:
        return "C4C6"
    
    def append_0prep(self, circuit: stim.Circuit, loc: int, 
                     N_prev: int, N_now: int) -> None:
        """Ideal preparation without noise."""
        if N_prev == 1:
            self.ops.reset(circuit, loc, N_now)
        else:
            for i in range(N_now):
                self.append_0prep(circuit, (loc + i) * N_prev, 1, N_prev)
        
        if N_now == self.N_c4:
            self.ops.append_h_c4(circuit, (loc + 0) * N_prev, 1, N_prev)
            self.ops.append_cnot(circuit, (loc + 0) * N_prev, (loc + 1) * N_prev, 1, N_prev)
            self.ops.append_cnot(circuit, (loc + 0) * N_prev, (loc + 2) * N_prev, 1, N_prev)
            self.ops.append_cnot(circuit, (loc + 0) * N_prev, (loc + 3) * N_prev, 1, N_prev)
    
    def append_noisy_0prep(self, circuit: stim.Circuit, loc1: int, loc2: int,
                           N_prev: int, N_now: int, p: float,
                           detector_counter: List[int]) -> List:
        """Noisy preparation matching original."""
        if N_now == self.N_c6:
            n_now = 3
        else:
            n_now = N_now
        
        if N_prev == 1:
            self.ops.noisy_reset(circuit, loc1, N_now, p)
            self.ops.noisy_reset(circuit, loc2, N_now, p)
            detector_0prep = []
        else:
            detector_0prep = []
            for i in range(n_now):
                detector_0prep.append(
                    self.append_noisy_0prep(circuit, (loc1 + i) * N_prev, 
                                           (loc1 + n_now + i) * N_prev, 1, N_prev, p,
                                           detector_counter)
                )
            for i in range(n_now):
                detector_0prep.append(
                    self.append_noisy_0prep(circuit, (loc2 + i) * N_prev,
                                           (loc2 + n_now + i) * N_prev, 1, N_prev, p,
                                           detector_counter)
                )
        
        if N_now == self.N_c4:
            # H on ancilla
            for i in range(4):
                circuit.append("H", loc2 + i)
            
            # CNOTs
            for i in range(N_now):
                self.ops.append_noisy_cnot(circuit, (loc2 + i) * N_prev, 
                                           (loc1 + i) * N_prev, 1, N_prev, p)
            for i in range(N_now):
                self.ops.append_noisy_cnot(circuit, (loc1 + i) * N_prev,
                                           (loc2 + (i + 1) % N_now) * N_prev, 1, N_prev, p)
            
            # Measure
            detector_0prep.append(
                self.ops.append_noisy_m(circuit, loc2, N_prev, N_now, p, detector_counter)
            )
            
            # Correction CNOTs
            for i in range(N_now - 1):
                for j in range(N_now - 1):
                    if j >= i:
                        self.ops.append_cnot(circuit, (loc2 + i) * N_prev,
                                            (loc1 + j) * N_prev, 1, N_prev)
            
            return detector_0prep
        
        elif N_now == self.N_c6:
            # C6 preparation
            for i in range(3):
                self.ops.append_h_c4(circuit, (loc2 + i) * N_prev, 1, self.N_c4)
                self.ops.append_noisy_cnot(circuit, (loc2 + i) * N_prev,
                                           (loc1 + i) * N_prev, 1, self.N_c4, p)
            for i in range(3):
                self.ops.append_noisy_cnot(circuit, (loc1 + i) * N_prev,
                                           (loc2 + ((i + 1) % 3)) * N_prev, 1, self.N_c4, p)
            
            detector_0prep.append(
                self.ops.append_noisy_m(circuit, loc2, N_prev, 3, p, detector_counter)
            )
            
            # Correction CNOTs
            self.ops.append_cnot(circuit, (loc2 + 0) * N_prev, (loc1 + 0) * N_prev, 1, N_prev)
            self.ops.append_cnot(circuit, (loc2 + 0) * N_prev, (loc1 + 1) * N_prev, 1, N_prev)
            self.ops.append_cnot(circuit, (loc2 + 1) * N_prev, (loc1 + 1) * N_prev, 1, N_prev)
            
            # SWAPs for C4 structure
            circuit.append("SWAP", [(loc1 + 1) * self.N_c4 + 1, (loc1 + 1) * self.N_c4 + 2])
            circuit.append("SWAP", [(loc1 + 1) * self.N_c4 + 1, (loc1 + 1) * self.N_c4 + 3])
            circuit.append("SWAP", [(loc1 + 2) * self.N_c4 + 1, (loc1 + 2) * self.N_c4 + 3])
            circuit.append("SWAP", [(loc1 + 2) * self.N_c4 + 1, (loc1 + 2) * self.N_c4 + 2])
            
            return detector_0prep
        
        return detector_0prep


# =============================================================================
# C4C6 EC Gadget
# =============================================================================

class C4C6ECGadget(ECGadget):
    """
    EC gadget for C4C6 matching original exactly.
    """
    
    def __init__(self, concat_code: ConcatenatedCode, ops: 'C4C6TransversalOps',
                 prep: C4C6PreparationStrategy, no_ec: bool = False):
        super().__init__(concat_code, ops)
        self._prep = prep  # Store directly, don't use set_prep
        self.no_ec = no_ec
        self.N_c4 = 4
        self.N_c6 = 6
    
    @property
    def ec_type(self) -> str:
        return "c4c6_edt"  # Error-Detecting Teleportation for C4C6
    
    def append_noisy_ec(self, circuit: stim.Circuit, loc1: int, loc2: int,
                        loc3: int, loc4: int, N_prev: int, N_now: int, p: float,
                        detector_counter: List[int]) -> Tuple[List, List, List]:
        """
        Noisy EC gadget matching original append_noisy_ec.
        """
        detector_0prep = []
        detector_Z = []
        detector_X = []
        
        if N_now == 1:
            return None
        
        if N_now == self.N_c6:
            n_now = 3
        else:
            n_now = N_now
        
        # Prepare ancillas
        detector_0prep.extend(
            self.prep.append_noisy_0prep(circuit, loc2, loc4, N_prev, N_now, p, detector_counter)
        )
        detector_0prep.extend(
            self.prep.append_noisy_0prep(circuit, loc3, loc4, N_prev, N_now, p, detector_counter)
        )
        
        self.ops.append_h(circuit, loc2, N_prev, n_now)
        self.ops.append_noisy_cnot(circuit, loc2, loc3, N_prev, n_now, p)
        
        # Error-detecting teleportation at L2
        if N_now == self.N_c6 and not self.no_ec:
            for i in range(3):
                result = self.append_noisy_ec(
                    circuit, (loc2 + i) * N_prev, (loc4 + 0) * N_prev,
                    (loc4 + 1) * N_prev, (loc4 + 2) * N_prev, 1, N_prev, p, detector_counter
                )
                self.ops.append_h(circuit, (loc2 + i) * N_prev, 1, N_prev)
                self.ops.append_cnot(circuit, (loc4 + 1) * N_prev, (loc2 + i) * N_prev, 1, N_prev)
                self.ops.append_h(circuit, (loc2 + i) * N_prev, 1, N_prev)
                self.ops.append_cnot(circuit, (loc4 + 0) * N_prev, (loc2 + i) * N_prev, 1, N_prev)
                detector_0prep.extend(result[0])
                detector_Z.append(result[1])
                detector_X.append(result[2])
            
            for i in range(3):
                result = self.append_noisy_ec(
                    circuit, (loc3 + i) * N_prev, (loc4 + 0) * N_prev,
                    (loc4 + 1) * N_prev, (loc4 + 2) * N_prev, 1, N_prev, p, detector_counter
                )
                self.ops.append_h(circuit, (loc3 + i) * N_prev, 1, N_prev)
                self.ops.append_cnot(circuit, (loc4 + 1) * N_prev, (loc3 + i) * N_prev, 1, N_prev)
                self.ops.append_h(circuit, (loc3 + i) * N_prev, 1, N_prev)
                self.ops.append_cnot(circuit, (loc4 + 0) * N_prev, (loc3 + i) * N_prev, 1, N_prev)
                detector_0prep.extend(result[0])
                detector_Z.append(result[1])
                detector_X.append(result[2])
        
        self.ops.append_noisy_cnot(circuit, loc1, loc2, N_prev, n_now, p)
        self.ops.append_h(circuit, loc1, N_prev, n_now)
        
        detector_Z.append(self.ops.append_noisy_m(circuit, loc1, N_prev, n_now, p, detector_counter))
        detector_X.append(self.ops.append_noisy_m(circuit, loc2, N_prev, n_now, p, detector_counter))
        
        self.ops.append_swap(circuit, loc1, loc3, N_prev, n_now)
        
        return detector_0prep, detector_Z, detector_X


# =============================================================================
# Factory Function
# =============================================================================

def create_c4c6_simulator(num_levels: int, noise_model: NoiseModel,
                          no_ec: bool = False) -> C4C6Simulator:
    """
    Create a C4C6 simulator.
    
    Args:
        num_levels: 1 for C4 only, 2 for C4->C6
        noise_model: Noise model to use
        no_ec: If True, skip error-detecting teleportation
    
    Returns:
        C4C6Simulator configured for the specified levels
    """
    concat_code = create_concatenated_c4c6(num_levels)
    return C4C6Simulator(concat_code, noise_model, no_ec=no_ec)
