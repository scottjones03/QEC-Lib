# src/qectostim/decoders/joint_ml_decoder.py
"""
Joint Maximum Likelihood Decoder for Concatenated Codes.

This decoder achieves optimal error correction for concatenated codes by
treating the full [[n*k, 1, d*d']] code as a single entity rather than
decoding hierarchically.

Key Advantage:
- Hierarchical decoding fails on [2,2,0,...] patterns (weight 4) for [[49,1,9]]
- Joint decoding correctly handles ALL weight-4 errors (d=9 means correct up to 4)
- Achieves true p^5 scaling for [[49,1,9]] = [[7,1,3]] ⊗ [[7,1,3]]

Theoretical Results (from Step 29 analysis):
- All weight 1-4 errors: 100% corrected
- Weight 5 failure patterns: 854,931 (out of 1,906,884)
- Leading term: p_L ≈ 854,931 × p^5

Example:
    >>> from qectostim.decoders import JointMLDecoder
    >>> 
    >>> # For [[49,1,9]] concatenated Steane code
    >>> Hz = build_concatenated_Hz(inner_Hz, outer_Hz, inner_ZL)
    >>> ZL = build_full_ZL(inner_ZL, outer_ZL_blocks)
    >>> 
    >>> decoder = JointMLDecoder(Hz, ZL, max_weight=4)
    >>> logical = decoder.decode(final_data)  # 49 bits
"""
from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass


@dataclass
class JointMLConfig:
    """Configuration for Joint ML decoder."""
    max_weight: int = 4  # Maximum correctable weight
    verbose: bool = False


class JointMLDecoder:
    """
    Joint Maximum Likelihood decoder for concatenated codes.
    
    Uses precomputed syndrome lookup table for all error patterns up to
    the maximum correctable weight (floor((d-1)/2)).
    
    For [[49,1,9]] (d=9), this means storing all weight 0-4 patterns:
    - Weight 0: 1
    - Weight 1: 49
    - Weight 2: C(49,2) = 1,176
    - Weight 3: C(49,3) = 18,424
    - Weight 4: C(49,4) = 211,876
    Total: ~231,526 patterns → manageable
    
    Attributes:
        Hz: Z-stabilizer parity check matrix (n_checks × n_qubits)
        ZL: Set of qubit indices in logical Z operator support
        syndrome_table: Dict mapping syndrome tuple → correction array
    """
    
    def __init__(
        self,
        Hz: np.ndarray,
        ZL: List[int],
        config: Optional[JointMLConfig] = None,
    ):
        """
        Initialize Joint ML decoder.
        
        Args:
            Hz: Z-stabilizer parity check matrix (n_checks × n_qubits)
            ZL: Indices of qubits in Z logical operator
            config: Decoder configuration
        """
        self.Hz = np.asarray(Hz, dtype=np.uint8)
        self.ZL = set(ZL)
        self.n_qubits = Hz.shape[1]
        self.n_checks = Hz.shape[0]
        self.config = config or JointMLConfig()
        
        # Build syndrome lookup table
        self.syndrome_table = self._build_syndrome_table()
    
    def _build_syndrome_table(self) -> Dict[Tuple[int, ...], np.ndarray]:
        """
        Build syndrome → minimum weight error lookup table.
        
        Returns dictionary mapping syndrome tuples to correction arrays.
        Only stores the first (minimum weight) error pattern for each syndrome.
        """
        table: Dict[Tuple[int, ...], np.ndarray] = {}
        
        # Weight 0
        syn0 = tuple(np.zeros(self.n_checks, dtype=int))
        table[syn0] = np.zeros(self.n_qubits, dtype=np.uint8)
        
        # Weights 1 to max_weight
        for weight in range(1, self.config.max_weight + 1):
            if self.config.verbose:
                print(f"  Building weight-{weight} table...")
            
            for qubits in combinations(range(self.n_qubits), weight):
                e = np.zeros(self.n_qubits, dtype=np.uint8)
                for q in qubits:
                    e[q] = 1
                
                syn = tuple((self.Hz @ e) % 2)
                
                # Only store if this syndrome not seen (minimum weight)
                if syn not in table:
                    table[syn] = e.copy()
        
        if self.config.verbose:
            print(f"  Syndrome table has {len(table)} entries")
        
        return table
    
    def decode(self, final_data: np.ndarray) -> int:
        """
        Decode final data to logical value.
        
        Args:
            final_data: Array of n_qubits data bits
            
        Returns:
            Estimated logical value (0 or 1)
        """
        data = np.asarray(final_data, dtype=np.uint8).flatten()
        
        if len(data) != self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} data bits, got {len(data)}"
            )
        
        # Compute syndrome
        syndrome = tuple((self.Hz @ data) % 2)
        
        # Look up correction
        correction = self.syndrome_table.get(syndrome)
        
        if correction is None:
            # Syndrome not in table (high weight error)
            # Use no correction
            correction = np.zeros(self.n_qubits, dtype=np.uint8)
        
        # Apply correction
        corrected = (data + correction) % 2
        
        # Compute logical value
        logical = sum(corrected[q] for q in self.ZL) % 2
        
        return int(logical)
    
    def decode_batch(self, final_data_batch: np.ndarray) -> np.ndarray:
        """
        Decode batch of shots.
        
        Args:
            final_data_batch: Array of shape (n_shots, n_qubits)
            
        Returns:
            Array of n_shots logical values
        """
        n_shots = final_data_batch.shape[0]
        results = np.zeros(n_shots, dtype=np.uint8)
        
        for i in range(n_shots):
            results[i] = self.decode(final_data_batch[i])
        
        return results


def build_concatenated_Hz(
    inner_Hz: np.ndarray,
    outer_Hz: np.ndarray,
    inner_ZL: List[int],
) -> np.ndarray:
    """
    Build Hz matrix for concatenated code [[n×n', 1, d×d']].
    
    Structure:
    - n_inner × n_blocks inner stabilizers (block-diagonal)
    - n_outer outer stabilizers (acting on inner Z_L supports)
    
    Args:
        inner_Hz: Inner code Hz (n_inner_checks × n_inner)
        outer_Hz: Outer code Hz (n_outer_checks × n_blocks)
        inner_ZL: Qubit indices for inner Z logical
        
    Returns:
        Full Hz matrix (total_checks × total_qubits)
    """
    n_inner_checks, n_inner = inner_Hz.shape
    n_outer_checks, n_blocks = outer_Hz.shape
    
    n_physical = n_inner * n_blocks
    n_total_checks = n_inner_checks * n_blocks + n_outer_checks
    
    Hz = np.zeros((n_total_checks, n_physical), dtype=np.uint8)
    
    # Inner stabilizers (block diagonal)
    for block in range(n_blocks):
        for stab in range(n_inner_checks):
            row = block * n_inner_checks + stab
            for q in range(n_inner):
                if inner_Hz[stab, q] == 1:
                    Hz[row, block * n_inner + q] = 1
    
    # Outer stabilizers (act on inner Z_L positions)
    for stab in range(n_outer_checks):
        row = n_inner_checks * n_blocks + stab
        for block in range(n_blocks):
            if outer_Hz[stab, block] == 1:
                for q in inner_ZL:
                    Hz[row, block * n_inner + q] = 1
    
    return Hz


def build_full_ZL(
    inner_ZL: List[int],
    outer_ZL_blocks: List[int],
    n_inner: int,
) -> List[int]:
    """
    Build full Z logical support for concatenated code.
    
    Args:
        inner_ZL: Qubit indices for inner Z logical
        outer_ZL_blocks: Block indices for outer Z logical
        n_inner: Number of qubits per inner block
        
    Returns:
        List of physical qubit indices in full Z logical
    """
    full_ZL = []
    for block in outer_ZL_blocks:
        for q in inner_ZL:
            full_ZL.append(block * n_inner + q)
    return full_ZL


# =============================================================================
# Factory for common concatenated codes
# =============================================================================

def create_steane_steane_decoder(verbose: bool = False) -> JointMLDecoder:
    """
    Create Joint ML decoder for [[49,1,9]] = [[7,1,3]] ⊗ [[7,1,3]].
    
    This is the Steane code concatenated with itself.
    - Distance: 9 (corrects up to 4 errors)
    - Achieves p^5 scaling
    
    Returns:
        JointMLDecoder configured for [[49,1,9]]
    """
    # Steane code Hz (3×7)
    inner_Hz = np.array([
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1],
    ], dtype=np.uint8)
    
    outer_Hz = inner_Hz.copy()  # Same for Steane ⊗ Steane
    inner_ZL = [0, 1, 2]  # Z_L = I I I Z Z Z Z
    outer_ZL_blocks = [0, 1, 2]  # Outer Z_L on blocks 0, 1, 2
    
    # Build full Hz
    full_Hz = build_concatenated_Hz(inner_Hz, outer_Hz, inner_ZL)
    
    # Build full Z_L
    full_ZL = build_full_ZL(inner_ZL, outer_ZL_blocks, n_inner=7)
    
    if verbose:
        print(f"Creating [[49,1,9]] Joint ML decoder:")
        print(f"  Hz shape: {full_Hz.shape}")
        print(f"  Z_L support: {full_ZL} (weight {len(full_ZL)})")
    
    config = JointMLConfig(max_weight=4, verbose=verbose)
    return JointMLDecoder(full_Hz, full_ZL, config)
