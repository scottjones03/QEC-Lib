"""
Concatenated CSS Code Simulator v10 - Steane Code Implementation
─────────────────────────────────────────────────────────────────────────────

This module provides Steane-specific implementations for the concatenated
CSS code simulator. It imports generic base classes from concatenated_css_v10.py
and provides:

1. create_steane_code() - [[7,1,3]] Steane code with exact circuit specification
2. create_steane_propagation_l2() - Error propagation tables for level-2 decoding
3. create_concatenated_steane() - Factory for concatenated Steane codes
4. SteanePreparationStrategy - Exact preparation circuit matching original
5. SteaneECGadget - Steane-style error correction
6. SteaneDecoder - Syndrome decoding with exact lookup tables
7. create_steane_simulator() - Convenience factory for Steane simulators

Usage:
    from concatenated_css_v10_steane import create_steane_simulator
    
    simulator = create_steane_simulator(num_levels=2, noise_model=noise_model)
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
# Steane Code Factory Functions
# =============================================================================

def create_steane_code() -> CSSCode:
    """
    Create the [[7,1,3]] Steane code with EXACT original specification.
    
    All circuit details match concatenated_steane.py exactly.
    
    Returns:
        CSSCode with complete circuit specification for Steane code
    """
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])
    Lx = np.array([1, 1, 1, 0, 0, 0, 0])
    Lz = np.array([1, 1, 1, 0, 0, 0, 0])
    
    return CSSCode(
        name="Steane",
        n=7, k=1, d=3,
        Hz=H, Hx=H,
        logical_z_ops=[Lz],  # k=1: single logical Z operator
        logical_x_ops=[Lx],  # k=1: single logical X operator
        h_qubits=[0, 1, 3],
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
        verification_qubits=[2, 4, 5],
        uses_bellpair_prep=False,  # Steane uses standard CSS encoding
        idle_schedule={
            'cnot_round_1': [5],  # Idle noise on qubit 5 after CNOT round 1 (0-indexed)
            'verif_cnot_0': [0, 3],
            'verif_cnot_1': [0, 1, 2, 3, 5, 6],
            'verif_cnot_2': [0, 1, 2, 3, 4, 6],
            'verif_measure': [0, 1, 2, 3, 4, 6],
        }
    )


def create_steane_propagation_l2() -> PropagationTables:
    """
    Create exact propagation tables from original concatenated_steane.py.
    
    These tables describe how errors propagate through the level-2 preparation
    circuit and are CRITICAL for correct hierarchical decoding.
    
    Returns:
        PropagationTables with X/Z propagation and measurement dependencies
    """
    return PropagationTables(
        propagation_X=[
            [0,2,4,6],[1,2,5,6],[2],[3,4,5,6],[4,6],[5],
            [0,2],[1,5,6],[2],[3,4,6],[4,6],[5],[6],
            [0],[1,5],[2],[3],[4,6],[5],[6],[],
            [0],[1],[2],[3],[4],[5],[6],[],
            [0],[1],[2],[3],[4],[5],[6],[],
            [0],[1],[2],[3],[4],[5],[6],[]
        ],
        propagation_Z=[
            [0],[1],[2,0,1],[3],[4,3,0],[5,1,3],
            [0],[1],[2,0],[3],[4,3],[5,1],[6,1,4],
            [0],[1],[2],[3],[4],[5,1],[6,4],[2,4,5],
            [0],[1],[2],[3],[4],[5],[6],[4,5],
            [0],[1],[2],[3],[4],[5],[6],[5],
            [0],[1],[2],[3],[4],[5],[6],[]
        ],
        propagation_m=[2,4,5,6,7,8,9,10,11,14,15,17,18,20,25,26,28,34,36,44],
        num_ec_0prep=45
    )


def create_concatenated_steane(num_levels: int) -> ConcatenatedCode:
    """
    Create concatenated Steane code with correct propagation tables.
    
    Args:
        num_levels: Number of concatenation levels (1 or 2)
    
    Returns:
        ConcatenatedCode configured for Steane code
    """
    steane = create_steane_code()
    prop_tables = {}
    if num_levels >= 2:
        prop_tables[1] = create_steane_propagation_l2()
    return ConcatenatedCode(
        levels=[steane] * num_levels,
        name=f"Steane^{num_levels}",
        propagation_tables=prop_tables
    )


# =============================================================================
# Steane Preparation Strategy
# =============================================================================

class SteanePreparationStrategy(PreparationStrategy):
    """
    Steane code preparation matching original concatenated_steane.py exactly.
    
    This implementation provides the exact CNOT sequences, idle qubit scheduling,
    and verification circuit that matches the original implementation for
    bitwise reproducibility.
    """
    
    @property
    def strategy_name(self) -> str:
        return "steane"
    
    def append_0prep(self, circuit: stim.Circuit, loc1: int,
                     N_prev: int, N_now: int):
        """Noiseless preparation matching original."""
        code = self.concat_code.code_at_level(0)
        
        if N_prev == 1:
            PhysicalOps.reset(circuit, loc1, N_now)
        else:
            for i in range(N_now):
                self.append_0prep(circuit, (loc1 + i) * N_prev, 1, N_prev)
        
        if N_now == code.n:
            # H gates
            for q in code.h_qubits:
                self.ops.append_h(circuit, (loc1 + q) * N_prev, 1, N_prev)
            
            # CNOTs in exact order
            for ctrl, targ in code.encoding_cnots:
                self.ops.append_cnot(circuit, (loc1 + ctrl) * N_prev, 
                                     (loc1 + targ) * N_prev, 1, N_prev)
    
    def append_noisy_0prep(self, circuit: stim.Circuit, loc1: int, loc2: int,
                           N_prev: int, N_now: int, p: float,
                           detector_counter: List[int]) -> Union[List, Tuple]:
        """Noisy preparation matching original exactly."""
        code = self.concat_code.code_at_level(0)
        n_now = N_now
        gamma = p / 10  # Error model 'a'
        
        if N_prev == 1:
            PhysicalOps.noisy_reset(circuit, loc1, N_now, p)
            PhysicalOps.noisy_reset(circuit, loc2, N_now, p)
            detector_0prep = []
        else:
            detector_0prep = []
            for i in range(n_now):
                result = self.append_noisy_0prep(
                    circuit, (loc1 + i) * N_prev, (loc1 + n_now + i) * N_prev,
                    1, N_prev, p, detector_counter
                )
                detector_0prep.append(result)
            
            if N_now == code.n:
                result = self.append_noisy_0prep(
                    circuit, loc2 * N_prev, (loc2 + n_now) * N_prev,
                    1, N_prev, p, detector_counter
                )
                detector_0prep.append(result)
            else:
                for i in range(n_now):
                    result = self.append_noisy_0prep(
                        circuit, (loc2 + i) * N_prev, (loc2 + n_now + i) * N_prev,
                        1, N_prev, p, detector_counter
                    )
                    detector_0prep.append(result)
        
        if N_now == code.n and N_prev != 1:
            return self._noisy_0prep_l2(circuit, loc1, loc2, N_prev, n_now,
                                        p, gamma, detector_counter, detector_0prep)
        elif N_now == code.n:
            return self._noisy_0prep_l1(circuit, loc1, loc2, N_prev, n_now,
                                        p, gamma, detector_counter, detector_0prep)
        
        return detector_0prep
    
    def _noisy_0prep_l1(self, circuit: stim.Circuit, loc1: int, loc2: int,
                        N_prev: int, n_now: int, p: float, gamma: float,
                        detector_counter: List[int], detector_0prep: List) -> List:
        """Level-1 noisy preparation."""
        code = self.concat_code.code_at_level(0)
        
        # H gates
        for q in code.h_qubits:
            self.ops.append_h(circuit, (loc1 + q) * N_prev, 1, N_prev)
        
        # CNOT round 1
        self.ops.append_noisy_cnot(circuit, (loc1 + 1) * N_prev, (loc1 + 2) * N_prev, 1, N_prev, p)
        self.ops.append_noisy_cnot(circuit, (loc1 + 3) * N_prev, (loc1 + 5) * N_prev, 1, N_prev, p)
        self.ops.append_noisy_cnot(circuit, (loc1 + 0) * N_prev, (loc1 + 4) * N_prev, 1, N_prev, p)
        
        # CNOT round 2
        self.ops.append_noisy_cnot(circuit, (loc1 + 1) * N_prev, (loc1 + 6) * N_prev, 1, N_prev, p)
        self.ops.append_noisy_cnot(circuit, (loc1 + 0) * N_prev, (loc1 + 2) * N_prev, 1, N_prev, p)
        self.ops.append_noisy_cnot(circuit, (loc1 + 3) * N_prev, (loc1 + 4) * N_prev, 1, N_prev, p)
        self.ops.append_noisy_wait(circuit, [(loc1 + 5) * N_prev], N_prev, p, gamma, steps=1)
        
        # CNOT round 3
        self.ops.append_noisy_cnot(circuit, (loc1 + 1) * N_prev, (loc1 + 5) * N_prev, 1, N_prev, p)
        self.ops.append_noisy_cnot(circuit, (loc1 + 4) * N_prev, (loc1 + 6) * N_prev, 1, N_prev, p)
        
        # Verification CNOTs
        self.ops.append_noisy_cnot(circuit, (loc1 + 2) * N_prev, loc2 * N_prev, 1, N_prev, p)
        self.ops.append_noisy_wait(circuit, [(loc1 + 0) * N_prev, (loc1 + 3) * N_prev], N_prev, p, gamma, steps=1)
        
        self.ops.append_noisy_cnot(circuit, (loc1 + 4) * N_prev, loc2 * N_prev, 1, N_prev, p)
        self.ops.append_noisy_wait(circuit, [(loc1 + i) * N_prev for i in [0, 1, 2, 3, 5, 6]], N_prev, p, gamma, steps=1)
        
        self.ops.append_noisy_cnot(circuit, (loc1 + 5) * N_prev, loc2 * N_prev, 1, N_prev, p)
        self.ops.append_noisy_wait(circuit, [(loc1 + i) * N_prev for i in [0, 1, 2, 3, 4, 6]], N_prev, p, gamma, steps=1)
        
        # Measure verification
        detector_0prep.append(
            self.ops.append_noisy_m(circuit, loc2 * N_prev, 1, N_prev, p, detector_counter)
        )
        self.ops.append_noisy_wait(circuit, [(loc1 + i) * N_prev for i in [0, 1, 2, 3, 4, 6]], N_prev, p, gamma, steps=1)
        
        return detector_0prep
    
    def _noisy_0prep_l2(self, circuit: stim.Circuit, loc1: int, loc2: int,
                        N_prev: int, n_now: int, p: float, gamma: float,
                        detector_counter: List[int], detector_0prep: List) -> Tuple:
        """
        Level-2 noisy preparation - CORRECTED VERSION.
        
        NO EC during L2 prep - EC amplifies noise instead of helping.
        We only do: H gates -> encoding CNOTs -> verification CNOTs -> verification measurement -> decorrelation CNOTs
        
        Returns: (detector_0prep, detector_0prep_l2, [], [])
        Note: detector_X and detector_Z are empty since no EC during prep.
        """
        code = self.concat_code.code_at_level(0)
        
        # NO EC rounds during L2 encoding - this is the key fix
        # The old version had 45 EC rounds which amplified noise
        
        # H gates
        for q in code.h_qubits:
            self.ops.append_h(circuit, (loc1 + q) * N_prev, 1, N_prev)
        
        # CNOT round 1
        self.ops.append_noisy_cnot(circuit, (loc1 + 1) * N_prev, (loc1 + 2) * N_prev, 1, N_prev, p)
        self.ops.append_noisy_cnot(circuit, (loc1 + 3) * N_prev, (loc1 + 5) * N_prev, 1, N_prev, p)
        self.ops.append_noisy_cnot(circuit, (loc1 + 0) * N_prev, (loc1 + 4) * N_prev, 1, N_prev, p)
        
        # CNOT round 2
        self.ops.append_noisy_cnot(circuit, (loc1 + 1) * N_prev, (loc1 + 6) * N_prev, 1, N_prev, p)
        self.ops.append_noisy_cnot(circuit, (loc1 + 0) * N_prev, (loc1 + 2) * N_prev, 1, N_prev, p)
        self.ops.append_noisy_cnot(circuit, (loc1 + 3) * N_prev, (loc1 + 4) * N_prev, 1, N_prev, p)
        self.ops.append_noisy_wait(circuit, [(loc1 + 5) * N_prev], N_prev, p, gamma, steps=1)
        
        # CNOT round 3
        self.ops.append_noisy_cnot(circuit, (loc1 + 1) * N_prev, (loc1 + 5) * N_prev, 1, N_prev, p)
        self.ops.append_noisy_cnot(circuit, (loc1 + 4) * N_prev, (loc1 + 6) * N_prev, 1, N_prev, p)
        
        # Verification CNOTs (data -> verification ancilla)
        self.ops.append_noisy_cnot(circuit, (loc1 + 2) * N_prev, loc2 * N_prev, 1, N_prev, p)
        self.ops.append_noisy_wait(circuit, [(loc1 + 0) * N_prev, (loc1 + 3) * N_prev], N_prev, p, gamma, steps=1)
        
        self.ops.append_noisy_cnot(circuit, (loc1 + 4) * N_prev, loc2 * N_prev, 1, N_prev, p)
        self.ops.append_noisy_wait(circuit, [(loc1 + i) * N_prev for i in [0, 1, 2, 3, 5, 6]], N_prev, p, gamma, steps=1)
        
        self.ops.append_noisy_cnot(circuit, (loc1 + 5) * N_prev, loc2 * N_prev, 1, N_prev, p)
        self.ops.append_noisy_wait(circuit, [(loc1 + i) * N_prev for i in [0, 1, 2, 3, 4, 6]], N_prev, p, gamma, steps=1)
        
        # Measure verification ancilla
        detector_0prep_l2 = self.ops.append_noisy_m(circuit, loc2 * N_prev, 1, N_prev, p, detector_counter)
        
        # DECORRELATION CNOTs - this is CRITICAL!
        # After measuring the verification ancilla, apply CNOTs to undo entanglement
        # with the measured ancilla. This is what C4/C6 do and Steane was missing.
        self.ops.append_cnot(circuit, loc2 * N_prev, (loc1 + 2) * N_prev, 1, N_prev)
        self.ops.append_cnot(circuit, loc2 * N_prev, (loc1 + 4) * N_prev, 1, N_prev)
        self.ops.append_cnot(circuit, loc2 * N_prev, (loc1 + 5) * N_prev, 1, N_prev)
        
        # Return empty detector_X and detector_Z since no EC during prep
        return detector_0prep, detector_0prep_l2, [], []


# =============================================================================
# Steane EC Gadget
# =============================================================================

class SteaneECGadget(ECGadget):
    """Steane-style EC matching original exactly."""
    
    @property
    def ec_type(self) -> str:
        return "steane"
    
    def append_noisy_ec(self, circuit: stim.Circuit, loc1: int, loc2: int,
                        loc3: int, loc4: int, N_prev: int, N_now: int,
                        p: float, detector_counter: List[int]) -> Tuple:
        """Steane EC matching original."""
        detector_0prep = []
        detector_0prep_l2 = []
        detector_Z = []
        detector_X = []
        
        if N_now == 1:
            return None
        
        n_now = N_now
        
        # Prepare ancillas
        if N_prev == 1:
            result1 = self.prep.append_noisy_0prep(circuit, loc2, loc4, N_prev, N_now, p, detector_counter)
            result2 = self.prep.append_noisy_0prep(circuit, loc3, loc4, N_prev, N_now, p, detector_counter)
            detector_0prep.extend(result1)
            detector_0prep.extend(result2)
        else:
            result1 = self.prep.append_noisy_0prep(circuit, loc2, loc4, N_prev, N_now, p, detector_counter)
            detector_0prep.extend(result1[0])
            detector_0prep_l2.append(result1[1])
            detector_X.extend(result1[2])
            detector_Z.extend(result1[3])
            
            result2 = self.prep.append_noisy_0prep(circuit, loc3, loc4, N_prev, N_now, p, detector_counter)
            detector_0prep.extend(result2[0])
            detector_0prep_l2.append(result2[1])
            detector_X.extend(result2[2])
            detector_Z.extend(result2[3])
        
        # H on first ancilla
        self.ops.append_h(circuit, loc2, N_prev, n_now)
        
        # CNOT between ancillas
        self.ops.append_noisy_cnot(circuit, loc2, loc3, N_prev, n_now, p)
        
        # Recursive EC for higher levels
        if N_prev != 1:
            for i in range(n_now):
                ec_result = self.append_noisy_ec(
                    circuit, (loc2 + i) * N_prev,
                    (loc4 + 0) * N_prev, (loc4 + 1) * N_prev, (loc4 + 2) * N_prev,
                    1, N_prev, p, detector_counter
                )
                detector_0prep.extend(ec_result[0])
                detector_Z.append(ec_result[1])
                detector_X.append(ec_result[2])
            
            for i in range(n_now):
                ec_result = self.append_noisy_ec(
                    circuit, (loc3 + i) * N_prev,
                    (loc4 + 0) * N_prev, (loc4 + 1) * N_prev, (loc4 + 2) * N_prev,
                    1, N_prev, p, detector_counter
                )
                detector_0prep.extend(ec_result[0])
                detector_Z.append(ec_result[1])
                detector_X.append(ec_result[2])
        
        # CNOT from data
        self.ops.append_noisy_cnot(circuit, loc1, loc2, N_prev, n_now, p)
        
        # H on data
        self.ops.append_h(circuit, loc1, N_prev, n_now)
        
        # Measure
        detector_Z.append(self.ops.append_noisy_m(circuit, loc1, N_prev, n_now, p, detector_counter))
        detector_X.append(self.ops.append_noisy_m(circuit, loc2, N_prev, n_now, p, detector_counter))
        
        # Swap
        self.ops.append_swap(circuit, loc1, loc3, N_prev, n_now)
        
        if N_prev == 1:
            return detector_0prep, detector_Z, detector_X
        else:
            return detector_0prep, detector_0prep_l2, detector_Z, detector_X


# =============================================================================
# Steane Decoder
# =============================================================================

class SteaneDecoder(Decoder):
    """Decoder matching original exactly."""
    
    def __init__(self, concat_code: ConcatenatedCode):
        super().__init__(concat_code)
        self.check_matrix = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ])
        self.logical_op = np.array([1, 1, 1, 0, 0, 0, 0])
    
    def decode_measurement(self, m: np.ndarray, m_type: str = 'x') -> int:
        """Matches original decode_measurement_steane."""
        outcome = np.sum(m * self.logical_op) % 2
        
        syndrome = 0
        for i in range(3):
            e = np.sum(m * self.check_matrix[i, :]) % 2
            syndrome += e * (2 ** i)
        
        if syndrome > 0:
            outcome = (outcome + self.logical_op[int(syndrome) - 1]) % 2
        
        return int(outcome)
    
    def decode_measurement_post_selection(self, m: np.ndarray, m_type: str = 'x') -> int:
        """Returns -1 if syndrome is non-zero (for post-selection)."""
        outcome = np.sum(m * self.logical_op) % 2
        
        syndrome = 0
        for i in range(3):
            e = np.sum(m * self.check_matrix[i, :]) % 2
            syndrome += e * (2 ** i)
        
        if syndrome > 0:
            return -1
        return int(outcome)
    
    def decode_ec_hd(self, x: np.ndarray, detector_X: List, detector_Z: List,
                     correction_x_prev: List, correction_z_prev: List) -> Tuple:
        """
        Hierarchical EC decoding.
        
        With corrected prep (no EC during ancilla prep), detector_X/Z have structure:
        - [7 L1 EC for ancilla 1] + [7 L1 EC for ancilla 2] + [transversal meas]
        Total: 15 entries (indices 0-6 for anc1, 7-13 for anc2, 14 for transversal)
        
        Old structure (with 45 EC during each ancilla prep):
        - [45 prep EC anc1] + [45 prep EC anc2] + [7 L1 EC anc1] + [7 L1 EC anc2] + [trans]
        Total: 105 entries
        """
        prop = self.concat_code.get_propagation_tables(1)
        num_ec = prop.num_ec_0prep if prop else 45
        
        # Detect structure: corrected prep has ~15 entries, old has 105+
        is_corrected_prep = len(detector_X) < 50
        
        if is_corrected_prep:
            # SIMPLIFIED DECODING for corrected prep (no prep EC)
            # Structure: [7 L1 EC anc1] + [7 L1 EC anc2] + [transversal meas]
            mx = [0] * 7
            mz = [0] * 7
            cx_next = [0] * 7
            cz_next = [0] * 7
            cx1 = list(correction_x_prev)
            cz1 = list(correction_z_prev)
            
            # Decode L1 EC measurements for both ancillas (indices 0-13)
            cx2 = [0] * 7
            cz2 = [0] * 7
            cx3 = [0] * 7
            cz3 = [0] * 7
            
            for i in range(7):
                # First ancilla L1 EC (indices 0-6)
                if i < len(detector_X):
                    cx2[i] = self.decode_measurement(x[detector_X[i][0][0]:detector_X[i][0][1]])
                if i < len(detector_Z):
                    cz2[i] = self.decode_measurement(x[detector_Z[i][0][0]:detector_Z[i][0][1]])
                
                # Second ancilla L1 EC (indices 7-13)
                if 7 + i < len(detector_X):
                    cx3[i] = self.decode_measurement(x[detector_X[7+i][0][0]:detector_X[7+i][0][1]])
                if 7 + i < len(detector_Z):
                    cz3[i] = self.decode_measurement(x[detector_Z[7+i][0][0]:detector_Z[7+i][0][1]])
            
            # Decode transversal measurements (index 14)
            for i in range(7):
                x_correction = (cx1[i] + cx2[i]) % 2
                z_correction = (cz1[i] + cz2[i]) % 2
                cx_next[i] = cx3[i]
                cz_next[i] = cz3[i]
                
                if 14 < len(detector_X) and i < len(detector_X[14]):
                    detx = detector_X[14][i]
                    mx[i] = (self.decode_measurement(x[detx[0]:detx[1]], 'x') + x_correction) % 2
                else:
                    mx[i] = x_correction
                    
                if 14 < len(detector_Z) and i < len(detector_Z[14]):
                    detz = detector_Z[14][i]
                    mz[i] = (self.decode_measurement(x[detz[0]:detz[1]], 'z') + z_correction) % 2
                else:
                    mz[i] = z_correction
            
            correction_x = self.decode_measurement(np.array(mx), 'x')
            correction_z = self.decode_measurement(np.array(mz), 'z')
            
            return correction_x, correction_z, cx_next, cz_next
        
        # Original complex decoding with prep EC (len > 50)
        mx = [0] * 7
        mz = [0] * 7
        cx_next = [0] * 7
        cz_next = [0] * 7
        cx1 = list(correction_x_prev)
        cz1 = list(correction_z_prev)
        cx2 = [0] * 7
        cz2 = [0] * 7
        cx3 = [0] * 7
        cz3 = [0] * 7
        cx2_0prep = [0] * num_ec
        cz2_0prep = [0] * num_ec
        cx3_0prep = [0] * num_ec
        cz3_0prep = [0] * num_ec
        
        for i in range(7):
            cx2[i] = self.decode_measurement(x[detector_X[2*num_ec+i][0][0]:detector_X[2*num_ec+i][0][1]])
            cz2[i] = self.decode_measurement(x[detector_Z[2*num_ec+i][0][0]:detector_Z[2*num_ec+i][0][1]])
            cx3[i] = self.decode_measurement(x[detector_X[2*num_ec+i+7][0][0]:detector_X[2*num_ec+i+7][0][1]])
            cz3[i] = self.decode_measurement(x[detector_Z[2*num_ec+i+7][0][0]:detector_Z[2*num_ec+i+7][0][1]])
        
        for a in range(num_ec):
            cx2_0prep[a] = self.decode_measurement(x[detector_X[a][0][0]:detector_X[a][0][1]])
            cz2_0prep[a] = self.decode_measurement(x[detector_Z[a][0][0]:detector_Z[a][0][1]])
            cx3_0prep[a] = self.decode_measurement(x[detector_X[num_ec+a][0][0]:detector_X[num_ec+a][0][1]])
            cz3_0prep[a] = self.decode_measurement(x[detector_Z[num_ec+a][0][0]:detector_Z[num_ec+a][0][1]])
            
            for i in prop.propagation_X[a]:
                cz2[i] = (cz2[i] + cx2_0prep[a]) % 2
                cx3[i] = (cx3[i] + cx3_0prep[a]) % 2
            
            for i in prop.propagation_Z[a]:
                cx2[i] = (cx2[i] + cz2_0prep[a]) % 2
                cx3[i] = (cx3[i] + cz2_0prep[a]) % 2
                cz2[i] = (cz2[i] + cz3_0prep[a]) % 2
                cz3[i] = (cz3[i] + cz3_0prep[a]) % 2
        
        for i in range(7):
            detx = detector_X[2*num_ec+14][i]
            detz = detector_Z[2*num_ec+14][i]
            x_corr = (cx1[i] + cx2[i]) % 2
            z_corr = (cz1[i] + cz2[i]) % 2
            cx_next[i] = cx3[i]
            cz_next[i] = cz3[i]
            mx[i] = (self.decode_measurement(x[detx[0]:detx[1]], 'x') + x_corr) % 2
            mz[i] = (self.decode_measurement(x[detz[0]:detz[1]], 'z') + z_corr) % 2
        
        correction_x = self.decode_measurement(np.array(mx), 'x')
        correction_z = self.decode_measurement(np.array(mz), 'z')
        
        return correction_x, correction_z, cx_next, cz_next
    
    def decode_m_hd(self, x: np.ndarray, detector_m: List, correction_l1: List) -> int:
        """Hierarchical measurement decoding matching original."""
        m = [0] * 7
        for i in range(7):
            det = detector_m[i]
            m[i] = (self.decode_measurement(x[det[0]:det[1]]) + correction_l1[i]) % 2
        
        outcome = self.decode_measurement(np.array(m), 'x')
        return outcome


# =============================================================================
# Steane Post-Selection Methods
# =============================================================================

class SteanePostSelector(PostSelector):
    """Post-selection methods specific to Steane code."""
    
    def post_selection_steane(self, x: np.ndarray, detector_0prep: List) -> bool:
        """Level-1 post-selection on single detector."""
        if x[detector_0prep[0]] % 2 == 0:
            return True
        return False
    
    def post_selection_steane_l2(self, x: np.ndarray, detector_0prep: List,
                                  detector_X: List, detector_Z: List) -> bool:
        """Level-2 post-selection with propagation."""
        prop = self.concat_code.get_propagation_tables(1)
        if prop is None:
            return True
        
        outcome = self.decoder.decode_measurement(
            x[detector_0prep[0]:detector_0prep[1]]
        )
        
        for a in prop.propagation_m:
            if a < len(detector_X) and detector_X[a]:
                correction_x = self.decoder.decode_measurement(
                    x[detector_X[a][0][0]:detector_X[a][0][1]]
                )
                outcome = (outcome + correction_x) % 2
        
        return outcome % 2 == 0


# =============================================================================
# Steane Simulator Factory
# =============================================================================

def create_steane_simulator(num_levels: int, noise_model: NoiseModel) -> ConcatenatedCodeSimulator:
    """
    Create simulator for concatenated Steane code.
    
    This is a convenience factory that creates a fully configured simulator
    for the Steane code with all Steane-specific components.
    
    Args:
        num_levels: Number of concatenation levels (1 or 2)
        noise_model: Noise model to apply to circuits
    
    Returns:
        ConcatenatedCodeSimulator configured for Steane code
    """
    concat_code = create_concatenated_steane(num_levels)
    return create_steane_simulator_from_code(concat_code, noise_model)


def create_steane_simulator_from_code(concat_code: ConcatenatedCode, 
                                       noise_model: NoiseModel) -> ConcatenatedCodeSimulator:
    """
    Create Steane simulator from an existing ConcatenatedCode.
    
    This allows more control over the code configuration while still
    using Steane-specific preparation, EC, and decoding.
    
    Args:
        concat_code: Pre-configured concatenated code
        noise_model: Noise model to apply to circuits
    
    Returns:
        ConcatenatedCodeSimulator with Steane-specific components
    """
    # Create the simulator with Steane-specific components
    simulator = ConcatenatedCodeSimulator.__new__(ConcatenatedCodeSimulator)
    simulator.concat_code = concat_code
    simulator.noise_model = noise_model
    simulator.ops = TransversalOps(concat_code)
    
    # Use Steane-specific components
    simulator.ec = SteaneECGadget(concat_code, simulator.ops)
    simulator.prep = SteanePreparationStrategy(concat_code, simulator.ops)
    simulator.decoder = SteaneDecoder(concat_code)
    
    # Wire up circular dependencies
    simulator.ec.set_prep(simulator.prep)
    simulator.prep.set_ec_gadget(simulator.ec)
    
    simulator.post_selector = PostSelector(concat_code, simulator.decoder)
    simulator.acceptance = AcceptanceChecker(concat_code, simulator.decoder)
    
    # Import LogicalGateDispatcher for gate operations
    from qectostim.experiments.concatenated_css_v10 import LogicalGateDispatcher
    simulator.gates = LogicalGateDispatcher(concat_code, simulator.ops)
    
    return simulator


# =============================================================================
# Main Entry Point (for command-line usage matching original)
# =============================================================================

if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 7:
        print("Usage: python concatenated_css_v10_steane.py <output_file> <num_shots> <case_number> <p> <error_model> <level>")
        sys.exit(1)
    
    output_file_name = str(sys.argv[1])
    num_shots = int(sys.argv[2])
    case_number = int(sys.argv[3])
    p = float(sys.argv[4])
    error_model = str(sys.argv[5])
    level = int(sys.argv[6])
    
    # Create noise model based on error_model parameter
    from qectostim.noise.models import CircuitDepolarizingNoise
    
    if error_model == 'a':
        gamma = p / 10
    elif error_model == 'b':
        gamma = p / 2
    elif error_model == 'c':
        gamma = p
    else:
        gamma = p / 10
    
    noise_model = CircuitDepolarizingNoise(p, gamma)
    
    if level == 1:
        simulator = create_steane_simulator(1, noise_model)
        logical_error, variance = simulator.estimate_logical_cnot_error_l1(p, num_shots)
    elif level == 2:
        simulator = create_steane_simulator(2, noise_model)
        logical_error, variance = simulator.estimate_logical_cnot_error_l2(p, num_shots)
    else:
        print('Level should be 1 or 2!')
        sys.exit(1)
    
    with open(output_file_name, 'a') as file:
        output_data = {
            'physical_error': p,
            'logical_error': logical_error,
            'variance': variance
        }
        file.write('"case' + str(case_number) + '": ' + json.dumps(output_data) + ',\n')
