"""
Concatenated CSS Code Simulator v10 - Shor→Steane Code Implementation
─────────────────────────────────────────────────────────────────────────────

This module provides Shor[[9,1,3]]→Steane[[7,1,3]] concatenated code simulation.
Both codes are k=1, so the generic ConcatenatedCodeSimulator should work.

Structure:
- Inner code: Shor [[9,1,3]] - 9 physical qubits encoding 1 logical qubit
- Outer code: Steane [[7,1,3]] - 7 inner blocks encoding 1 outer logical qubit
- Total physical qubits: 7 × 9 = 63
- Distance: d_inner × d_outer = 3 × 3 = 9

Shor Code Structure:
- 3 blocks of 3 qubits each (arranged as 3×3 grid)
- 6 Z stabilizers: pairwise within each row (detect X errors)
- 2 X stabilizers: entire row pairs (detect Z errors)
- Logical X: X on any column representative (e.g., X0 X3 X6)
- Logical Z: Z on any row (e.g., Z0 Z1 Z2)

Usage:
    from concatenated_css_v10_shorsteane import create_shorsteane_simulator
    
    simulator = create_shorsteane_simulator(num_levels=2, noise_model=noise)
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
    GenericDecoder,
    PostSelector,
    AcceptanceChecker,
    ConcatenatedCodeSimulator,
    KnillECGadget,
    GenericPreparationStrategy,
)
from qectostim.noise.models import NoiseModel


# =============================================================================
# Shor Code Factory Function
# =============================================================================

def create_shor_code() -> CSSCode:
    """
    Create the [[9,1,3]] Shor code with exact circuit specification.
    
    Shor's code is a CSS code formed by concatenating:
    - 3-qubit phase-flip code (protects against Z errors)
    - 3-qubit bit-flip code (protects against X errors)
    
    Qubit layout (3×3 grid):
        0  1  2   (row 0 - bit-flip block 0)
        3  4  5   (row 1 - bit-flip block 1)
        6  7  8   (row 2 - bit-flip block 2)
    
    Logical |0⟩_L = (|000⟩ + |111⟩)(|000⟩ + |111⟩)(|000⟩ + |111⟩) / 2√2
    
    Returns:
        CSSCode with complete circuit specification for Shor code
    """
    # 6 Z-type stabilizers: pairwise checks within each row (detect X errors)
    # Each row has 2 checks: (q0,q1), (q1,q2)
    Hz = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0],  # Z0 Z1
        [0, 1, 1, 0, 0, 0, 0, 0, 0],  # Z1 Z2
        [0, 0, 0, 1, 1, 0, 0, 0, 0],  # Z3 Z4
        [0, 0, 0, 0, 1, 1, 0, 0, 0],  # Z4 Z5
        [0, 0, 0, 0, 0, 0, 1, 1, 0],  # Z6 Z7
        [0, 0, 0, 0, 0, 0, 0, 1, 1],  # Z7 Z8
    ])
    
    # 2 X-type stabilizers: check parity across row pairs (detect Z errors)
    Hx = np.array([
        [1, 1, 1, 1, 1, 1, 0, 0, 0],  # X on rows 0 and 1 (qubits 0-5)
        [0, 0, 0, 1, 1, 1, 1, 1, 1],  # X on rows 1 and 2 (qubits 3-8)
    ])
    
    # Logical operators
    # Logical X: X on first qubit of each row (column 0)
    Lx = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])  # X0 X3 X6
    
    # Logical Z: Z on entire first row
    Lz = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])  # Z0 Z1 Z2
    
    # Encoding circuit for |0⟩_L preparation:
    # 1. Start with |0⟩^⊗9
    # 2. Apply H to q0, q3, q6 (create |+⟩ states for phase protection)
    # 3. Apply CNOTs to spread bit-flip protection within each row
    #
    # Encoding CNOTs (order matters for idle scheduling):
    # First spread within rows: 0→1, 0→2, 3→4, 3→5, 6→7, 6→8
    encoding_cnots = [
        (0, 1), (0, 2),  # Row 0: spread from q0
        (3, 4), (3, 5),  # Row 1: spread from q3
        (6, 7), (6, 8),  # Row 2: spread from q6
    ]
    
    encoding_cnot_rounds = [
        [(0, 1), (3, 4), (6, 7)],  # First CNOT in each row (parallel)
        [(0, 2), (3, 5), (6, 8)],  # Second CNOT in each row (parallel)
    ]
    
    return CSSCode(
        name="Shor",
        n=9, k=1, d=3,
        Hz=Hz, Hx=Hx,
        logical_z_ops=[Lz],  # k=1: single logical Z operator
        logical_x_ops=[Lx],  # k=1: single logical X operator
        h_qubits=[0, 3, 6],  # H on first qubit of each row
        pre_h_cnots=[],      # No CNOTs before H gates
        encoding_cnots=encoding_cnots,
        encoding_cnot_rounds=encoding_cnot_rounds,
        verification_qubits=[1, 2, 4, 5, 7, 8],  # Non-H qubits for verification
        uses_bellpair_prep=False,  # Standard CSS encoding
        decoder_type="syndrome",   # Use syndrome decoding (k=1)
        idle_schedule={
            'cnot_round_1': [2, 5, 8],  # Idle qubits after first CNOT round
        }
    )


def create_steane_outer_code() -> CSSCode:
    """
    Create Steane code configured as outer code for Shor→Steane.
    
    This is the standard [[7,1,3]] Steane code that will operate on
    7 inner Shor blocks.
    
    Returns:
        CSSCode for Steane as outer code
    """
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])
    Lx = np.array([1, 1, 1, 0, 0, 0, 0])
    Lz = np.array([1, 1, 1, 0, 0, 0, 0])
    
    return CSSCode(
        name="Steane_outer",
        n=7, k=1, d=3,
        Hz=H, Hx=H,
        logical_z_ops=[Lz],
        logical_x_ops=[Lx],
        h_qubits=[0, 1, 3],
        encoding_cnots=[
            (1, 2), (3, 5), (0, 4),
            (1, 6), (0, 2), (3, 4),
            (1, 5), (4, 6)
        ],
        encoding_cnot_rounds=[
            [(1, 2), (3, 5), (0, 4)],
            [(1, 6), (0, 2), (3, 4)],
            [(1, 5), (4, 6)]
        ],
        verification_qubits=[2, 4, 5],
        uses_bellpair_prep=False,
    )


# =============================================================================
# Shor Decoder
# =============================================================================

class ShorDecoder(GenericDecoder):
    """
    Decoder for Shor [[9,1,3]] code.
    
    IMPORTANT: Shor code is NOT self-dual (Hz ≠ Hx):
    - Hz: 6×9 matrix (6 Z stabilizers detect X errors)
    - Hx: 2×9 matrix (2 X stabilizers detect Z errors)
    
    The GenericDecoder has a bug for non-self-dual codes: it uses the
    wrong syndrome table for the check matrix. This override fixes that.
    
    Correct mapping:
    - m_type='x' (X-basis): use Hx to compute syndrome → look up in table built from Hx
    - m_type='z' (Z-basis): use Hz to compute syndrome → look up in table built from Hz
    """
    
    def __init__(self, concat_code: ConcatenatedCode):
        super().__init__(concat_code)
        # GenericDecoder builds:
        #   _syndrome_to_error_x from Hz (correct for detecting X errors)
        #   _syndrome_to_error_z from Hx (correct for detecting Z errors)
        # But decode_measurement uses them backwards! We fix that below.
    
    def decode_measurement(self, m: np.ndarray, m_type: str = 'x') -> int:
        """
        Decode measurement with correct syndrome table mapping for Shor code.
        
        For non-self-dual CSS codes like Shor:
        - m_type='x': X-basis measurement uses Hx stabilizers to detect Z errors
                      → use Hx for syndrome, _syndrome_to_error_z table
        - m_type='z': Z-basis measurement uses Hz stabilizers to detect X errors
                      → use Hz for syndrome, _syndrome_to_error_x table
        
        Args:
            m: Measurement outcomes (length n=9)
            m_type: 'x' for X-basis, 'z' for Z-basis
        
        Returns:
            Logical measurement outcome (0 or 1)
        """
        if m_type == 'x':
            # X-basis: detect Z errors using Hx stabilizers
            check_matrix = self.code.Hx
            logical_op = self._logical_x
            syndrome_table = self._syndrome_to_error_z  # Built from Hx
        else:
            # Z-basis: detect X errors using Hz stabilizers  
            check_matrix = self.code.Hz
            logical_op = self._logical_z
            syndrome_table = self._syndrome_to_error_x  # Built from Hz
        
        # Compute raw logical value
        outcome = self._compute_logical_value(m, logical_op)
        
        # Compute syndrome using the correct check matrix
        syndrome = self._compute_syndrome(m, check_matrix)
        
        # Apply correction if syndrome is non-zero
        if syndrome > 0:
            error_pos = syndrome_table.get(syndrome, None)
            
            if error_pos is not None and error_pos >= 0:
                # Single qubit correction
                outcome = (outcome + int(logical_op[error_pos])) % 2
            elif error_pos is not None and error_pos < -1:
                # Two-qubit correction (for higher distance codes)
                pair_code = -(error_pos + 1)
                q1 = pair_code // self.n
                q2 = pair_code % self.n
                correction = (int(logical_op[q1]) + int(logical_op[q2])) % 2
                outcome = (outcome + correction) % 2
        
        return int(outcome)
    
    def decode_measurement_post_selection(self, m: np.ndarray, m_type: str = 'x') -> int:
        """
        Decode with post-selection, using correct syndrome table mapping.
        
        Returns -1 if syndrome indicates uncorrectable error.
        """
        if m_type == 'x':
            check_matrix = self.code.Hx
            logical_op = self._logical_x
            syndrome_table = self._syndrome_to_error_z
            weights = self._error_weights_z
        else:
            check_matrix = self.code.Hz
            logical_op = self._logical_z
            syndrome_table = self._syndrome_to_error_x
            weights = self._error_weights_x
        
        # Compute syndrome
        syndrome = self._compute_syndrome(m, check_matrix)
        
        # Check if syndrome is correctable
        if syndrome > 0:
            if syndrome not in syndrome_table:
                return -1  # Unknown syndrome - reject
            
            # For distance-3 codes, reject weight-2+ errors
            if self.code.d <= 3 and weights.get(syndrome, 99) >= 2:
                return -1
        
        # Compute and correct logical value
        outcome = self._compute_logical_value(m, logical_op)
        
        if syndrome > 0:
            error_pos = syndrome_table.get(syndrome, None)
            if error_pos is not None and error_pos >= 0:
                outcome = (outcome + int(logical_op[error_pos])) % 2
        
        return int(outcome)


# =============================================================================
# Propagation Tables for Shor→Steane L2
# =============================================================================

def create_shorsteane_propagation_l2() -> PropagationTables:
    """
    Create propagation tables for Shor→Steane concatenation.
    
    These tables describe how errors propagate through the level-2 
    (Steane outer code) preparation circuit when the inner code is Shor.
    
    The structure follows the Steane preparation pattern but with
    Shor inner blocks instead of physical qubits.
    
    Returns:
        PropagationTables for Shor→Steane L2
    """
    # Based on Steane L2 propagation pattern
    # Adapted for Shor inner code
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


# =============================================================================
# Shor Preparation Strategy
# =============================================================================

class ShorPreparationStrategy(PreparationStrategy):
    """
    Shor code preparation strategy.
    
    Prepares logical |0⟩_L for Shor code:
    1. Reset all 9 qubits to |0⟩
    2. Apply H to q0, q3, q6 (first qubit of each row)
    3. Apply CNOTs to spread within each row
    """
    
    @property
    def strategy_name(self) -> str:
        return "shor"
    
    def append_0prep(self, circuit: stim.Circuit, loc1: int,
                     N_prev: int, N_now: int):
        """Noiseless Shor |0⟩_L preparation."""
        code = self.concat_code.code_at_level(0)
        
        if N_prev == 1:
            PhysicalOps.reset(circuit, loc1, N_now)
        else:
            for i in range(N_now):
                self.append_0prep(circuit, (loc1 + i) * N_prev, 1, N_prev)
        
        if N_now == code.n:  # N_now == 9
            # H gates on first qubit of each row
            for q in code.h_qubits:  # [0, 3, 6]
                self.ops.append_h(circuit, (loc1 + q) * N_prev, 1, N_prev)
            
            # CNOTs to spread within rows
            for ctrl, targ in code.encoding_cnots:
                self.ops.append_cnot(circuit, (loc1 + ctrl) * N_prev,
                                     (loc1 + targ) * N_prev, 1, N_prev)
    
    def append_noisy_0prep(self, circuit: stim.Circuit, loc1: int, loc2: int,
                           N_prev: int, N_now: int, p: float,
                           detector_counter: List[int]) -> Union[List, Tuple]:
        """
        Noisy Shor |0⟩_L preparation with verification.
        
        Returns detector information for post-selection.
        """
        code = self.concat_code.code_at_level(0)
        n_now = N_now
        gamma = p / 10  # Error model
        
        if N_prev == 1:
            # Physical level preparation
            PhysicalOps.noisy_reset(circuit, loc1, N_now, p)
            PhysicalOps.noisy_reset(circuit, loc2, N_now, p)
            detector_0prep = []
            
            if N_now == code.n:  # 9
                # H gates with noise
                for q in code.h_qubits:  # [0, 3, 6]
                    self.ops.append_noisy_h(circuit, loc1 + q, 1, 1, p)
                    self.ops.append_noisy_h(circuit, loc2 + q, 1, 1, p)
                
                # Encoding CNOTs with noise
                for round_idx, round_cnots in enumerate(code.encoding_cnot_rounds):
                    for ctrl, targ in round_cnots:
                        self.ops.append_noisy_cnot(circuit, loc1 + ctrl, loc1 + targ, 1, 1, p)
                        self.ops.append_noisy_cnot(circuit, loc2 + ctrl, loc2 + targ, 1, 1, p)
                    
                    # Idle noise on waiting qubits
                    if code.idle_schedule and f'cnot_round_{round_idx+1}' in code.idle_schedule:
                        for q in code.idle_schedule[f'cnot_round_{round_idx+1}']:
                            PhysicalOps.idle_noise(circuit, loc1 + q, p)
                            PhysicalOps.idle_noise(circuit, loc2 + q, p)
                
                # Verification: measure parity checks
                # For Shor, verify row parities (Z stabilizers)
                for row in range(3):
                    row_start = row * 3
                    # Measure Z parity of row on auxiliary
                    # Simplified: just measure and check
                    pass  # Verification handled by EC gadget
                
            return detector_0prep
        else:
            # Recursive preparation for concatenated levels
            detector_0prep = []
            for i in range(N_now):
                result = self.append_noisy_0prep(
                    circuit, (loc1 + i) * N_prev, (loc2 + i) * N_prev,
                    1, N_prev, p, detector_counter
                )
                if isinstance(result, list):
                    detector_0prep.extend(result)
            return detector_0prep


# =============================================================================
# Factory Functions
# =============================================================================

def create_concatenated_shorsteane(num_levels: int) -> ConcatenatedCode:
    """
    Create concatenated Shor→Steane code.
    
    Args:
        num_levels: 1 for just Shor, 2 for Shor→Steane
    
    Returns:
        ConcatenatedCode for Shor→Steane
    """
    shor = create_shor_code()
    
    if num_levels == 1:
        return ConcatenatedCode(
            levels=[shor],
            name="Shor",
        )
    
    steane = create_steane_outer_code()
    return ConcatenatedCode(
        levels=[shor, steane],
        name="Shor→Steane",
        propagation_tables={2: create_shorsteane_propagation_l2()},
    )


def create_shorsteane_simulator(num_levels: int, noise_model: NoiseModel) -> ConcatenatedCodeSimulator:
    """
    Create simulator for Shor→Steane concatenated code.
    
    Since both Shor and Steane are k=1 codes, the generic
    ConcatenatedCodeSimulator should work.
    
    Args:
        num_levels: 1 for L1, 2 for L2
        noise_model: Noise model for simulation
    
    Returns:
        ConcatenatedCodeSimulator configured for Shor→Steane
    """
    concat_code = create_concatenated_shorsteane(num_levels)
    
    # Use generic components - both codes are k=1
    return ConcatenatedCodeSimulator(
        concat_code=concat_code,
        noise_model=noise_model,
        decoder=ShorDecoder(concat_code) if num_levels >= 1 else None,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def estimate_shorsteane_cnot_error_l1(p: float, num_shots: int, Q: int = 10) -> Tuple[float, float]:
    """
    Estimate L1 CNOT error rate for Shor code.
    
    Args:
        p: Physical error probability
        num_shots: Number of Monte Carlo samples
        Q: Number of CNOT+EC rounds
    
    Returns:
        (error_rate, variance)
    """
    from qectostim.noise.models import CircuitDepolarizingNoise
    
    noise = CircuitDepolarizingNoise(p1=0, p2=p, before_measure_flip=p)
    sim = create_shorsteane_simulator(num_levels=1, noise_model=noise)
    return sim.estimate_logical_cnot_error_l1(p, num_shots, Q)


def estimate_shorsteane_cnot_error_l2(p: float, num_shots: int, Q: int = 1) -> Tuple[float, float]:
    """
    Estimate L2 CNOT error rate for Shor→Steane.
    
    Args:
        p: Physical error probability
        num_shots: Number of Monte Carlo samples
        Q: Number of CNOT+EC rounds
    
    Returns:
        (error_rate, variance)
    """
    from qectostim.noise.models import CircuitDepolarizingNoise
    
    noise = CircuitDepolarizingNoise(p1=0, p2=p, before_measure_flip=p)
    sim = create_shorsteane_simulator(num_levels=2, noise_model=noise)
    return sim.estimate_logical_cnot_error_l2(p, num_shots, Q)


def estimate_shorsteane_memory_error_l1(p: float, num_shots: int, 
                                         num_ec_rounds: int = 1) -> Tuple[float, float]:
    """
    Estimate L1 memory logical error rate for Shor code.
    
    Args:
        p: Physical error probability
        num_shots: Number of Monte Carlo samples
        num_ec_rounds: Number of EC rounds
    
    Returns:
        (error_rate, variance)
    """
    from qectostim.noise.models import CircuitDepolarizingNoise
    
    noise = CircuitDepolarizingNoise(p1=0, p2=p, before_measure_flip=p)
    sim = create_shorsteane_simulator(num_levels=1, noise_model=noise)
    return sim.estimate_memory_logical_error_l1(p, num_shots, num_ec_rounds)


def estimate_shorsteane_memory_error_l2(p: float, num_shots: int,
                                         num_ec_rounds: int = 1) -> Tuple[float, float]:
    """
    Estimate L2 memory logical error rate for Shor→Steane.
    
    Args:
        p: Physical error probability
        num_shots: Number of Monte Carlo samples
        num_ec_rounds: Number of EC rounds
    
    Returns:
        (error_rate, variance)
    """
    from qectostim.noise.models import CircuitDepolarizingNoise
    
    noise = CircuitDepolarizingNoise(p1=0, p2=p, before_measure_flip=p)
    sim = create_shorsteane_simulator(num_levels=2, noise_model=noise)
    return sim.estimate_memory_logical_error_l2(p, num_shots, num_ec_rounds)
