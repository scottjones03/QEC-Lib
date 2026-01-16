# src/qectostim/decoders/hard_decision_hierarchical_decoder.py
"""
Hard-Decision Hierarchical Decoder for Concatenated Codes.

This implements the decoding strategy from "Concatenated codes, save qubits" paper:
- First decode each inner block to get logical measurement outcomes
- Then decode outer code using those logical outcomes

This is designed for MEMORY EXPERIMENTS (prepare → wait → measure),
NOT for fault-tolerant gadgets with mid-circuit corrections.

The Paper's Algorithm for Concatenated Steane Code:
==================================================
Given 7 inner block logical values (m₁, m₂, ..., m₇):

1. Compute syndrome bits:
   a₁ = m₁ ⊕ m₃ ⊕ m₅ ⊕ m₇
   a₂ = m₂ ⊕ m₃ ⊕ m₆ ⊕ m₇
   a₃ = m₄ ⊕ m₅ ⊕ m₆ ⊕ m₇

2. Identify error location:
   i = a₁ + 2·a₂ + 4·a₃

3. Decode with correction:
   if i ∈ {1, 2, 3}:  # Error in logical support
       m̄ = (m₁ + m₂ + m₃ + 1) mod 2
   else:
       m̄ = (m₁ + m₂ + m₃) mod 2

For Shor [[9,1,3]] Inner Code:
=============================
Each inner block has 9 qubits in 3 triplets.
Z_L = Z₀ ⊗ Z₁ ⊗ ... ⊗ Z₈ (all 9 qubits)

The logical value is the XOR of all 9 qubit measurements.
(Or equivalently, XOR of the 3 triplet majorities)

The Shor inner decoder can correct single X errors using the
6 weight-2 Z stabilizers that check adjacent pairs within triplets.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Any, Set
from dataclasses import dataclass

if TYPE_CHECKING:
    from qectostim.codes.abstract_code import Code

try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False

from .base import Decoder


@dataclass
class HardDecisionConfig:
    """Configuration for hard-decision hierarchical decoder."""
    verbose: bool = False
    use_inner_correction: bool = True  # Apply inner error correction before outer


class HardDecisionHierarchicalDecoder(Decoder):
    """
    Hard-decision hierarchical decoder matching the paper's approach.
    
    For a concatenated code like Steane⊗Shor [[63,1,9]]:
    1. Read final measurements for all 63 data qubits
    2. Decode each of 7 inner Shor blocks → 7 logical values
    3. Decode outer Steane code using those 7 values
    
    Inner Decoding Options:
    - use_inner_correction=True: Apply Shor decoder to correct X errors
    - use_inner_correction=False: Just compute raw Z_L (XOR of all 9 qubits)
    
    Outer Decoding:
    - Use Steane syndrome to identify which block had an error
    - Apply correction if error is in logical support {0,1,2}
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        config: Optional[HardDecisionConfig] = None,
        basis: str = "Z",
    ):
        self.metadata = metadata
        self.config = config or HardDecisionConfig()
        self.basis = basis.upper()
        
        # Extract code objects
        self.inner_code = metadata.get('inner_code')
        self.outer_code = metadata.get('outer_code')
        
        # Block structure
        self.n_data_blocks = metadata.get('n_data_blocks', 7)
        self.n_inner = self.inner_code.n if self.inner_code else 7
        
        # Get final data measurement indices
        self._final_data_measurements = metadata.get('final_data_measurements', {})
        
        # Setup code structures
        self._setup_inner_code()
        self._setup_outer_code()
        
        # Build inner matcher if using corrections
        self._inner_matcher = None
        if self.config.use_inner_correction and HAS_PYMATCHING:
            self._build_inner_matcher()
    
    def _setup_inner_code(self) -> None:
        """Setup inner code matrices and logical support."""
        if self.inner_code is None:
            # Default to Steane [[7,1,3]]
            self._inner_hz = np.array([
                [0, 0, 0, 1, 1, 1, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
            ], dtype=np.uint8)
            self._inner_z_support = [0, 1, 2]
            self._n_inner = 7
            return
        
        hz = getattr(self.inner_code, 'hz', None)
        self._inner_hz = np.asarray(hz, dtype=np.uint8) if hz is not None else np.zeros((0, self.n_inner), dtype=np.uint8)
        
        # Get Z logical support
        self._inner_z_support = self._get_logical_support(self.inner_code, 'Z')
        self._n_inner = self.inner_code.n
    
    def _setup_outer_code(self) -> None:
        """Setup outer code structures."""
        if self.outer_code is None:
            # Default to Steane [[7,1,3]]
            self._outer_hx = np.array([
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 0],
                [1, 0, 1, 0, 1, 0, 1],
            ], dtype=np.uint8)
            self._outer_z_support = {0, 1, 2}
            return
        
        hx = getattr(self.outer_code, 'hx', None)
        self._outer_hx = np.asarray(hx, dtype=np.uint8) if hx is not None else np.zeros((0, self.n_data_blocks), dtype=np.uint8)
        self._outer_z_support = set(self._get_logical_support(self.outer_code, 'Z'))
    
    def _get_logical_support(self, code: Any, basis: str) -> List[int]:
        """Get qubit indices in logical operator support."""
        try:
            from qectostim.experiments.stabilizer_rounds.utils import get_logical_support
            return list(get_logical_support(code, basis, 0))
        except Exception:
            pass
        
        # Fallback
        n = getattr(code, 'n', 7)
        return list(range(min(3, n)))
    
    def _build_inner_matcher(self) -> None:
        """Build PyMatching decoder for inner code."""
        if self._inner_hz.size == 0:
            return
        
        try:
            # Check column weights
            col_weights = self._inner_hz.sum(axis=0)
            if np.max(col_weights) > 2:
                if self.config.verbose:
                    print(f"Inner Hz has max column weight {np.max(col_weights)} > 2, "
                          f"cannot use PyMatching. Using lookup instead.")
                return
            
            self._inner_matcher = pymatching.Matching(self._inner_hz)
        except Exception as e:
            if self.config.verbose:
                print(f"Could not build inner matcher: {e}")
    
    # =========================================================================
    # Main Decoding Interface
    # =========================================================================
    
    def decode(self, measurements: np.ndarray) -> int:
        """
        Decode using hard-decision hierarchical approach.
        
        1. Extract final data measurements for each inner block
        2. Decode each inner block → logical value
        3. Decode outer code using the 7 logical values
        """
        # Step 1: Get final measurements organized by block
        block_measurements = self._extract_block_measurements(measurements)
        
        if self.config.verbose:
            print(f"Block measurements shapes: {[len(m) for m in block_measurements.values()]}")
        
        # Step 2: Decode each inner block
        inner_logicals = self._decode_inner_blocks(block_measurements)
        
        if self.config.verbose:
            print(f"Inner logicals: {inner_logicals}")
        
        # Step 3: Decode outer code
        final_logical = self._decode_outer(inner_logicals)
        
        if self.config.verbose:
            print(f"Final logical: {final_logical}")
        
        return final_logical
    
    def decode_batch(self, measurements_batch: np.ndarray) -> np.ndarray:
        """Decode a batch of measurement samples."""
        n_shots = measurements_batch.shape[0]
        results = np.zeros(n_shots, dtype=np.int32)
        for i in range(n_shots):
            results[i] = self.decode(measurements_batch[i])
        return results
    
    # =========================================================================
    # Step 1: Extract Block Measurements
    # =========================================================================
    
    def _extract_block_measurements(
        self,
        measurements: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """Extract final data measurements organized by block."""
        block_meas = {}
        
        # Try MCM structure first
        mcm = self.metadata.get('measurement_centric_metadata')
        if mcm and 'final_data_measurements' in mcm:
            for block_id_str, final_info in mcm['final_data_measurements'].items():
                block_id = int(block_id_str) if isinstance(block_id_str, str) else block_id_str
                indices = final_info.get('transversal_meas_indices', [])
                if indices:
                    block_meas[block_id] = np.array([
                        measurements[i] for i in indices if 0 <= i < len(measurements)
                    ], dtype=np.uint8)
            return block_meas
        
        # Try direct final_data_measurements
        if self._final_data_measurements:
            for block_id, indices in self._final_data_measurements.items():
                if indices:
                    block_meas[block_id] = np.array([
                        measurements[i] for i in indices if 0 <= i < len(measurements)
                    ], dtype=np.uint8)
            return block_meas
        
        # Fallback: assume final n_data qubits are at end
        total_meas = len(measurements)
        n_data_qubits = self.n_data_blocks * self._n_inner
        
        if total_meas >= n_data_qubits:
            final_start = total_meas - n_data_qubits
            for block_id in range(self.n_data_blocks):
                start = final_start + block_id * self._n_inner
                end = start + self._n_inner
                block_meas[block_id] = measurements[start:end].astype(np.uint8)
        
        return block_meas
    
    # =========================================================================
    # Step 2: Decode Inner Blocks
    # =========================================================================
    
    def _decode_inner_blocks(
        self,
        block_measurements: Dict[int, np.ndarray]
    ) -> Dict[int, int]:
        """Decode each inner block to get logical value."""
        logicals = {}
        
        for block_id in range(self.n_data_blocks):
            meas = block_measurements.get(block_id, np.zeros(self._n_inner, dtype=np.uint8))
            
            if len(meas) != self._n_inner:
                # Pad or truncate
                if len(meas) < self._n_inner:
                    meas = np.concatenate([meas, np.zeros(self._n_inner - len(meas), dtype=np.uint8)])
                else:
                    meas = meas[:self._n_inner]
            
            # Decode this inner block
            logicals[block_id] = self._decode_single_inner_block(meas)
        
        return logicals
    
    def _decode_single_inner_block(self, measurements: np.ndarray) -> int:
        """
        Decode a single inner block.
        
        For Shor [[9,1,3]]:
        - Compute syndrome from Hz (6 weight-2 checks on adjacent pairs)
        - Use PyMatching to find most likely X error
        - Apply correction to get logical Z value
        
        For codes without PyMatching support:
        - Just XOR the logical support qubits
        """
        # Raw Z_L value (XOR of logical support)
        raw_z_l = 0
        for idx in self._inner_z_support:
            if idx < len(measurements):
                raw_z_l ^= int(measurements[idx])
        
        if not self.config.use_inner_correction or self._inner_matcher is None:
            return raw_z_l
        
        # Compute syndrome
        syndrome = (self._inner_hz @ measurements) % 2
        
        # Decode
        try:
            correction = self._inner_matcher.decode(syndrome)
            
            # Apply correction: X error on qubit q flips Z measurement if q in Z_L support
            correction_parity = 0
            for idx in self._inner_z_support:
                if idx < len(correction) and correction[idx]:
                    correction_parity ^= 1
            
            return (raw_z_l + correction_parity) % 2
        except Exception:
            return raw_z_l
    
    # =========================================================================
    # Step 3: Decode Outer Code
    # =========================================================================
    
    def _decode_outer(self, inner_logicals: Dict[int, int]) -> int:
        """
        Decode outer Steane code using the paper's algorithm.
        
        Given 7 logical values (m₁, m₂, ..., m₇):
        1. Compute syndrome: a₁ = m₁⊕m₃⊕m₅⊕m₇, etc.
        2. Error location: i = a₁ + 2·a₂ + 4·a₃
        3. Decode with correction if i ∈ {1,2,3}
        """
        # Convert to array (1-indexed in paper, 0-indexed here)
        m = np.array([inner_logicals.get(i, 0) for i in range(7)], dtype=np.uint8)
        
        # Compute syndrome (paper's formula, converted to 0-indexed)
        # a₁ = m₁ ⊕ m₃ ⊕ m₅ ⊕ m₇ → m[0] ⊕ m[2] ⊕ m[4] ⊕ m[6]
        # a₂ = m₂ ⊕ m₃ ⊕ m₆ ⊕ m₇ → m[1] ⊕ m[2] ⊕ m[5] ⊕ m[6]
        # a₃ = m₄ ⊕ m₅ ⊕ m₆ ⊕ m₇ → m[3] ⊕ m[4] ⊕ m[5] ⊕ m[6]
        a1 = m[0] ^ m[2] ^ m[4] ^ m[6]
        a2 = m[1] ^ m[2] ^ m[5] ^ m[6]
        a3 = m[3] ^ m[4] ^ m[5] ^ m[6]
        
        # Error location (1-indexed in paper)
        i = a1 + 2 * a2 + 4 * a3
        
        if self.config.verbose:
            print(f"  Outer syndrome: a=({a1},{a2},{a3}), error location i={i}")
        
        # Raw logical (XOR of blocks in Z_L support = {0,1,2})
        raw_logical = m[0] ^ m[1] ^ m[2]
        
        # Apply correction if error is in logical support
        # Paper: i ∈ {1,2,3} means blocks 0,1,2 (0-indexed)
        if i in {1, 2, 3}:
            return (raw_logical + 1) % 2
        else:
            return raw_logical


class HardDecisionSteaneDecoder(HardDecisionHierarchicalDecoder):
    """
    Specialized decoder for Steane⊗Steane [[49,1,9]] or Steane⊗Shor [[63,1,9]].
    
    Uses the exact algorithm from the paper for outer Steane code.
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        verbose: bool = False,
        use_inner_correction: bool = True,
    ):
        config = HardDecisionConfig(
            verbose=verbose,
            use_inner_correction=use_inner_correction,
        )
        super().__init__(metadata, config, basis="Z")


class LookupTableInnerDecoder(HardDecisionHierarchicalDecoder):
    """
    Decoder using lookup tables for inner code (no PyMatching required).
    
    For small inner codes like Steane [[7,1,3]], we can build a lookup
    table mapping each syndrome to the most likely error correction.
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        verbose: bool = False,
    ):
        config = HardDecisionConfig(
            verbose=verbose,
            use_inner_correction=True,
        )
        super().__init__(metadata, config, basis="Z")
        
        # Build lookup tables
        self._syndrome_to_correction = self._build_syndrome_lookup()
    
    def _build_syndrome_lookup(self) -> Dict[Tuple[int, ...], int]:
        """Build syndrome → correction lookup table for inner code."""
        lookup = {}
        
        if self._inner_hz.size == 0:
            return lookup
        
        n = self._inner_hz.shape[1]
        n_stabs = self._inner_hz.shape[0]
        
        # For each single-qubit X error, compute syndrome and correction
        for q in range(n):
            error = np.zeros(n, dtype=np.uint8)
            error[q] = 1
            syndrome = tuple((self._inner_hz @ error) % 2)
            
            # Correction needed if qubit is in Z_L support
            correction = 1 if q in self._inner_z_support else 0
            lookup[syndrome] = correction
        
        # No error case
        lookup[tuple([0] * n_stabs)] = 0
        
        return lookup
    
    def _decode_single_inner_block(self, measurements: np.ndarray) -> int:
        """Decode using lookup table."""
        # Raw Z_L value
        raw_z_l = 0
        for idx in self._inner_z_support:
            if idx < len(measurements):
                raw_z_l ^= int(measurements[idx])
        
        if not self._syndrome_to_correction:
            return raw_z_l
        
        # Compute syndrome
        syndrome = tuple((self._inner_hz @ measurements) % 2)
        
        # Look up correction
        correction = self._syndrome_to_correction.get(syndrome, 0)
        
        return (raw_z_l + correction) % 2
