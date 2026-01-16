"""
Local Hard-Decision Decoders for Detector-Based Architecture.

This module provides code-specific decoders that operate on detector bit slices.
Each decoder uses lookup-table / parity-based rules (NOT MWPM or global matching).

Key Design Principles:
---------------------
1. **Local decoding**: Each gadget/block decoded independently
2. **Hard decision**: Return definite correction bits OR -1 for ambiguous
3. **No temporal matching**: Each round decoded separately, corrections accumulated
4. **Deterministic rules**: Syndrome → correction via lookup tables

Return Convention:
-----------------
- Success: (logical_bits, correction_x, correction_z) where each is List[int] of 0/1
- Ambiguous: -1 (indicates detected but uncorrectable error pattern)

Ambiguous Handling:
------------------
When a decoder returns -1:
- For verification: Shot is rejected (post-selection)
- For syndrome decoding: Either reject or count as 0.5 error contribution

Usage:
------
    >>> decoder = get_decoder_for_code("steane")
    >>> result = decoder.decode_syndrome(z_syndrome_bits, x_syndrome_bits)
    >>> if result == -1:
    ...     # ambiguous - reject or count as 0.5
    >>> else:
    ...     logical, corr_x, corr_z = result
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


# Type alias for decoder return
# Success: (logical_bits, correction_x, correction_z)
# Ambiguous: -1
DecoderResult = Union[Tuple[List[int], List[int], List[int]], int]

# Simpler result for single-block decoding
# Success: (correction_x, correction_z) 
# Ambiguous: -1
CorrectionResult = Union[Tuple[np.ndarray, np.ndarray], int]


@dataclass
class LocalDecoderResult:
    """
    Structured result from local decoding.
    
    Attributes:
        success: Whether decoding succeeded (False if ambiguous)
        logical_value: Decoded logical bit(s) (if success)
        correction_x: X correction to apply (if success)
        correction_z: Z correction to apply (if success)
        syndrome_x: Extracted X syndrome
        syndrome_z: Extracted Z syndrome
        confidence: 1.0 for definite, 0.5 for ambiguous (for fractional counting)
    """
    success: bool
    logical_value: Optional[List[int]] = None
    correction_x: Optional[np.ndarray] = None
    correction_z: Optional[np.ndarray] = None
    syndrome_x: Optional[Tuple[int, ...]] = None
    syndrome_z: Optional[Tuple[int, ...]] = None
    confidence: float = 1.0
    
    @classmethod
    def ambiguous(cls, syndrome_x=None, syndrome_z=None) -> 'LocalDecoderResult':
        """Create an ambiguous result."""
        return cls(
            success=False,
            confidence=0.5,
            syndrome_x=syndrome_x,
            syndrome_z=syndrome_z,
        )
    
    @classmethod
    def success_result(
        cls,
        logical_value: List[int],
        correction_x: np.ndarray,
        correction_z: np.ndarray,
        syndrome_x=None,
        syndrome_z=None,
    ) -> 'LocalDecoderResult':
        """Create a successful result."""
        return cls(
            success=True,
            logical_value=logical_value,
            correction_x=correction_x,
            correction_z=correction_z,
            confidence=1.0,
            syndrome_x=syndrome_x,
            syndrome_z=syndrome_z,
        )


class LocalDecoder(ABC):
    """
    Abstract base class for local hard-decision decoders.
    
    Each decoder knows how to decode detector patterns for a specific code.
    """
    
    @property
    @abstractmethod
    def code_name(self) -> str:
        """Name of the code this decoder handles."""
        pass
    
    @property
    @abstractmethod
    def n_data_qubits(self) -> int:
        """Number of data qubits in the code."""
        pass
    
    @property
    @abstractmethod
    def n_syndrome_bits_z(self) -> int:
        """Number of Z syndrome bits (detecting X errors)."""
        pass
    
    @property
    @abstractmethod
    def n_syndrome_bits_x(self) -> int:
        """Number of X syndrome bits (detecting Z errors)."""
        pass
    
    @abstractmethod
    def decode_syndrome(
        self,
        z_detectors: np.ndarray,
        x_detectors: np.ndarray,
    ) -> LocalDecoderResult:
        """
        Decode syndrome detector patterns to corrections.
        
        Args:
            z_detectors: Z syndrome detector bits (detects X errors)
            x_detectors: X syndrome detector bits (detects Z errors)
            
        Returns:
            LocalDecoderResult with corrections or ambiguous flag
        """
        pass
    
    def decode_verification(
        self,
        verification_detectors: np.ndarray,
    ) -> bool:
        """
        Check if verification detectors pass.
        
        Default: pass if parity is 0 (even).
        
        Returns:
            True if verification passed, False if should reject
        """
        return np.sum(verification_detectors) % 2 == 0


# =============================================================================
# STEANE [[7,1,3]] CODE DECODER
# =============================================================================

class SteaneDecoder(LocalDecoder):
    """
    Local decoder for Steane [[7,1,3]] code.
    
    Syndrome decoding via lookup table for weight-1 errors.
    Returns -1 (ambiguous) if syndrome doesn't match any weight-1 pattern.
    
    Steane code structure:
    - 7 data qubits (indices 0-6)
    - 3 Z stabilizers (detect X errors): 
        g1 = Z0 Z2 Z4 Z6
        g2 = Z1 Z2 Z5 Z6  
        g3 = Z3 Z4 Z5 Z6
    - 3 X stabilizers (detect Z errors): same support
    - Z_L = Z0 Z1 Z2 Z3 Z4 Z5 Z6 (all qubits)
    - X_L = X0 X1 X2 X3 X4 X5 X6 (all qubits)
    
    Syndrome interpretation:
    - 3-bit syndrome s = (s0, s1, s2)
    - s as binary number gives error location (1-7) or 0 for no error
    """
    
    # Parity check matrix H (same for X and Z in Steane code)
    # Rows are stabilizer generators, columns are qubit indices
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],  # g1: qubits 0,2,4,6
        [0, 1, 1, 0, 0, 1, 1],  # g2: qubits 1,2,5,6
        [0, 0, 0, 1, 1, 1, 1],  # g3: qubits 3,4,5,6
    ], dtype=np.uint8)
    
    # Syndrome -> error location lookup
    # syndrome (as int 0-7) -> qubit index to flip (-1 for no error)
    SYNDROME_TO_QUBIT = {
        0: -1,  # No error
        1: 0,   # syndrome 001 -> error on qubit 0
        2: 1,   # syndrome 010 -> error on qubit 1
        3: 2,   # syndrome 011 -> error on qubit 2
        4: 3,   # syndrome 100 -> error on qubit 3
        5: 4,   # syndrome 101 -> error on qubit 4
        6: 5,   # syndrome 110 -> error on qubit 5
        7: 6,   # syndrome 111 -> error on qubit 6
    }
    
    # Logical operator support (all qubits for Steane)
    Z_LOGICAL = [0, 1, 2, 3, 4, 5, 6]
    X_LOGICAL = [0, 1, 2, 3, 4, 5, 6]
    
    @property
    def code_name(self) -> str:
        return "steane"
    
    @property
    def n_data_qubits(self) -> int:
        return 7
    
    @property
    def n_syndrome_bits_z(self) -> int:
        return 3
    
    @property
    def n_syndrome_bits_x(self) -> int:
        return 3
    
    def _syndrome_to_int(self, syndrome: np.ndarray) -> int:
        """Convert 3-bit syndrome to integer."""
        return int(syndrome[0]) + 2 * int(syndrome[1]) + 4 * int(syndrome[2])
    
    def decode_syndrome(
        self,
        z_detectors: np.ndarray,
        x_detectors: np.ndarray,
    ) -> LocalDecoderResult:
        """
        Decode Steane code syndromes to corrections.
        
        Z syndrome detects X errors -> gives X correction
        X syndrome detects Z errors -> gives Z correction
        """
        z_syndrome = tuple(z_detectors.astype(int))
        x_syndrome = tuple(x_detectors.astype(int))
        
        # Convert syndromes to integers
        z_int = self._syndrome_to_int(z_detectors)
        x_int = self._syndrome_to_int(x_detectors)
        
        # Look up error locations
        x_error_qubit = self.SYNDROME_TO_QUBIT.get(z_int, None)
        z_error_qubit = self.SYNDROME_TO_QUBIT.get(x_int, None)
        
        # Check for valid syndromes (should always be valid for weight-1 errors)
        if x_error_qubit is None or z_error_qubit is None:
            return LocalDecoderResult.ambiguous(
                syndrome_x=x_syndrome,
                syndrome_z=z_syndrome,
            )
        
        # Build correction arrays
        correction_x = np.zeros(7, dtype=np.uint8)
        correction_z = np.zeros(7, dtype=np.uint8)
        
        if x_error_qubit >= 0:
            correction_x[x_error_qubit] = 1
        if z_error_qubit >= 0:
            correction_z[z_error_qubit] = 1
        
        # Compute logical value from correction
        # For |0_L⟩ state: logical Z = 0 if even number of X errors on Z_L support
        logical_z = int(np.sum(correction_x[self.Z_LOGICAL]) % 2)
        # For |+_L⟩ state: logical X = 0 if even number of Z errors on X_L support  
        logical_x = int(np.sum(correction_z[self.X_LOGICAL]) % 2)
        
        return LocalDecoderResult.success_result(
            logical_value=[logical_z, logical_x],
            correction_x=correction_x,
            correction_z=correction_z,
            syndrome_x=x_syndrome,
            syndrome_z=z_syndrome,
        )
    
    def decode_raw_ancilla(
        self,
        z_ancilla: np.ndarray,
        x_ancilla: np.ndarray,
    ) -> LocalDecoderResult:
        """
        Decode from raw ancilla measurements (7 bits each) instead of syndrome bits.
        
        For transversal syndrome extraction, we measure all n ancilla qubits.
        Syndrome bits are computed as H @ ancilla_bits.
        """
        # Compute syndromes: s = H @ a mod 2
        z_syndrome = (self.H @ z_ancilla) % 2
        x_syndrome = (self.H @ x_ancilla) % 2
        
        return self.decode_syndrome(z_syndrome, x_syndrome)


# =============================================================================
# SHOR [[9,1,3]] CODE DECODER  
# =============================================================================

class ShorDecoder(LocalDecoder):
    """
    Local decoder for Shor [[9,1,3]] code.
    
    Shor code structure:
    - 9 data qubits arranged as 3 blocks of 3:
        Block 0: qubits 0,1,2
        Block 1: qubits 3,4,5  
        Block 2: qubits 6,7,8
    
    - Z stabilizers (6 total, detect X errors):
        Within each block: Z_i Z_{i+1} for consecutive pairs
        ZZ on (0,1), (1,2), (3,4), (4,5), (6,7), (7,8)
    
    - X stabilizers (2 total, detect Z errors):
        Between blocks: X^⊗3 on each pair of blocks
        X0X1X2 X3X4X5 and X3X4X5 X6X7X8
    
    - Z_L = Z0 Z3 Z6 (one from each block)
    - X_L = X0 X1 X2 X3 X4 X5 X6 X7 X8 (all qubits)
    """
    
    # Z stabilizers: 6 checks
    # (qubit1, qubit2) pairs
    Z_CHECKS = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]
    
    # X stabilizers: 2 checks
    # Each is X on 6 qubits
    X_CHECKS = [
        [0, 1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7, 8],
    ]
    
    # Logical operators
    Z_LOGICAL = [0, 3, 6]  # One from each block
    X_LOGICAL = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # All qubits
    
    # Syndrome lookup for Z errors (from X syndrome)
    # X syndrome (s0, s1) indicates which block pair has Z error
    # Then use majority vote within block
    
    @property
    def code_name(self) -> str:
        return "shor"
    
    @property
    def n_data_qubits(self) -> int:
        return 9
    
    @property
    def n_syndrome_bits_z(self) -> int:
        return 6
    
    @property
    def n_syndrome_bits_x(self) -> int:
        return 2
    
    def decode_syndrome(
        self,
        z_detectors: np.ndarray,
        x_detectors: np.ndarray,
    ) -> LocalDecoderResult:
        """
        Decode Shor code syndromes.
        
        Z syndrome (6 bits): detect X errors within blocks
        X syndrome (2 bits): detect Z errors between blocks
        """
        z_syndrome = tuple(z_detectors.astype(int))
        x_syndrome = tuple(x_detectors.astype(int))
        
        correction_x = np.zeros(9, dtype=np.uint8)
        correction_z = np.zeros(9, dtype=np.uint8)
        
        # === Decode X errors from Z syndrome ===
        # Z syndrome structure: [s01, s12, s34, s45, s67, s78]
        # For each block of 3, use the two syndrome bits to locate X error
        
        for block in range(3):
            s0 = z_detectors[2 * block]      # First pair in block
            s1 = z_detectors[2 * block + 1]  # Second pair in block
            
            base = block * 3
            if s0 == 0 and s1 == 0:
                pass  # No X error in this block
            elif s0 == 1 and s1 == 0:
                correction_x[base] = 1  # X error on first qubit
            elif s0 == 1 and s1 == 1:
                correction_x[base + 1] = 1  # X error on middle qubit
            elif s0 == 0 and s1 == 1:
                correction_x[base + 2] = 1  # X error on last qubit
        
        # === Decode Z errors from X syndrome ===
        # X syndrome: [s_01, s_12] where s_01 = parity of blocks 0,1
        # Use to identify which block has Z error, then apply to representative
        
        s01, s12 = x_detectors[0], x_detectors[1]
        
        if s01 == 0 and s12 == 0:
            pass  # No Z error
        elif s01 == 1 and s12 == 0:
            # Error in block 0
            correction_z[0] = 1
        elif s01 == 1 and s12 == 1:
            # Error in block 1
            correction_z[3] = 1
        elif s01 == 0 and s12 == 1:
            # Error in block 2
            correction_z[6] = 1
        
        # Compute logical values
        logical_z = int(np.sum(correction_x[self.Z_LOGICAL]) % 2)
        logical_x = int(np.sum(correction_z[self.X_LOGICAL]) % 2)
        
        return LocalDecoderResult.success_result(
            logical_value=[logical_z, logical_x],
            correction_x=correction_x,
            correction_z=correction_z,
            syndrome_x=x_syndrome,
            syndrome_z=z_syndrome,
        )


# =============================================================================
# REPETITION CODE DECODER (for testing)
# =============================================================================

class RepetitionDecoder(LocalDecoder):
    """
    Decoder for repetition code [[n,1,n]].
    
    Simple majority vote decoding.
    """
    
    def __init__(self, n: int = 3):
        self.n = n
    
    @property
    def code_name(self) -> str:
        return f"repetition_{self.n}"
    
    @property
    def n_data_qubits(self) -> int:
        return self.n
    
    @property
    def n_syndrome_bits_z(self) -> int:
        return self.n - 1
    
    @property
    def n_syndrome_bits_x(self) -> int:
        return 0  # Repetition code only corrects one type
    
    def decode_syndrome(
        self,
        z_detectors: np.ndarray,
        x_detectors: np.ndarray,
    ) -> LocalDecoderResult:
        """
        Decode repetition code via syndrome.
        
        Syndrome bit i = data[i] XOR data[i+1]
        """
        z_syndrome = tuple(z_detectors.astype(int))
        
        # Reconstruct error pattern
        correction_x = np.zeros(self.n, dtype=np.uint8)
        
        # Count errors from syndrome
        error_count = 0
        for i, s in enumerate(z_detectors):
            if s == 1:
                # Flip between i and i+1
                # Simple heuristic: flip qubit i
                correction_x[i] ^= 1
                error_count += 1
        
        # Majority vote: if more than half flipped, we got it wrong
        if np.sum(correction_x) > self.n // 2:
            correction_x = 1 - correction_x
        
        logical_z = int(np.sum(correction_x) % 2)
        
        return LocalDecoderResult.success_result(
            logical_value=[logical_z, 0],
            correction_x=correction_x,
            correction_z=np.zeros(self.n, dtype=np.uint8),
            syndrome_z=z_syndrome,
        )


# =============================================================================
# CONCATENATED CODE DECODER
# =============================================================================

class HierarchicalDecoder:
    """
    Hierarchical decoder for concatenated codes.
    
    Decodes inner blocks first, then uses decoded logical bits
    as inputs to outer decoder.
    
    Architecture:
        1. Extract inner block syndromes from detector vector
        2. Decode each inner block with inner decoder
        3. If any inner returns -1: handle ambiguity (reject or 0.5)
        4. Form outer-level "data" from inner logical outcomes
        5. Decode outer syndrome with outer decoder
        6. Combine corrections
    
    Usage:
        >>> h_decoder = HierarchicalDecoder(
        ...     inner_decoder=SteaneDecoder(),
        ...     outer_decoder=SteaneDecoder(),
        ...     n_inner_blocks=7,
        ... )
        >>> result = h_decoder.decode(detector_vector, metadata)
    """
    
    def __init__(
        self,
        inner_decoder: LocalDecoder,
        outer_decoder: LocalDecoder,
        n_inner_blocks: int,
    ):
        self.inner_decoder = inner_decoder
        self.outer_decoder = outer_decoder
        self.n_inner_blocks = n_inner_blocks
        
    def decode(
        self,
        detector_vector: np.ndarray,
        inner_z_groups: List[Tuple[int, int]],  # (start, end) per block
        inner_x_groups: List[Tuple[int, int]],
        outer_z_group: Optional[Tuple[int, int]] = None,
        outer_x_group: Optional[Tuple[int, int]] = None,
    ) -> LocalDecoderResult:
        """
        Decode concatenated code hierarchically.
        
        Args:
            detector_vector: Full detector bit vector
            inner_z_groups: Z syndrome detector ranges per inner block
            inner_x_groups: X syndrome detector ranges per inner block
            outer_z_group: Outer Z syndrome range (if separate from inner)
            outer_x_group: Outer X syndrome range
            
        Returns:
            LocalDecoderResult with combined corrections
        """
        # Step 1: Decode each inner block
        inner_results = []
        inner_logical_z = []
        inner_logical_x = []
        
        total_confidence = 1.0
        any_ambiguous = False
        
        for block_id in range(self.n_inner_blocks):
            # Extract inner syndromes
            z_start, z_end = inner_z_groups[block_id]
            x_start, x_end = inner_x_groups[block_id]
            
            z_det = detector_vector[z_start:z_end]
            x_det = detector_vector[x_start:x_end]
            
            result = self.inner_decoder.decode_syndrome(z_det, x_det)
            inner_results.append(result)
            
            if not result.success:
                any_ambiguous = True
                total_confidence *= 0.5
                inner_logical_z.append(0)  # Default to 0 for ambiguous
                inner_logical_x.append(0)
            else:
                inner_logical_z.append(result.logical_value[0])
                inner_logical_x.append(result.logical_value[1])
        
        # Step 2: Form outer-level syndrome from inner logical outcomes
        # For Steane-in-Steane: outer syndrome comes from inner block parities
        # This depends on whether outer syndrome is measured separately
        
        if outer_z_group is not None and outer_x_group is not None:
            # Outer syndrome measured separately
            oz_start, oz_end = outer_z_group
            ox_start, ox_end = outer_x_group
            outer_z_det = detector_vector[oz_start:oz_end]
            outer_x_det = detector_vector[ox_start:ox_end]
        else:
            # Compute outer syndrome from inner logical outcomes
            # For CSS codes: outer syndrome = H_outer @ inner_logical
            outer_z_det = (self.outer_decoder.H @ np.array(inner_logical_x)) % 2
            outer_x_det = (self.outer_decoder.H @ np.array(inner_logical_z)) % 2
        
        # Step 3: Decode outer
        outer_result = self.outer_decoder.decode_syndrome(outer_z_det, outer_x_det)
        
        if not outer_result.success:
            any_ambiguous = True
            total_confidence *= 0.5
        
        # Step 4: Combine corrections
        # Final correction on physical qubits = inner corrections + outer logical correction
        n_total = self.inner_decoder.n_data_qubits * self.n_inner_blocks
        combined_correction_x = np.zeros(n_total, dtype=np.uint8)
        combined_correction_z = np.zeros(n_total, dtype=np.uint8)
        
        # Apply inner corrections
        for block_id, result in enumerate(inner_results):
            if result.success:
                base = block_id * self.inner_decoder.n_data_qubits
                n_inner = self.inner_decoder.n_data_qubits
                combined_correction_x[base:base + n_inner] = result.correction_x
                combined_correction_z[base:base + n_inner] = result.correction_z
        
        # Apply outer correction (maps to logical operators on inner blocks)
        if outer_result.success:
            for block_id in range(self.n_inner_blocks):
                if outer_result.correction_x[block_id]:
                    # X correction on inner block = X_L on that block
                    base = block_id * self.inner_decoder.n_data_qubits
                    for q in self.inner_decoder.X_LOGICAL:
                        combined_correction_x[base + q] ^= 1
                        
                if outer_result.correction_z[block_id]:
                    # Z correction on inner block = Z_L on that block
                    base = block_id * self.inner_decoder.n_data_qubits
                    for q in self.inner_decoder.Z_LOGICAL:
                        combined_correction_z[base + q] ^= 1
        
        # Compute final logical value
        # Z_L of concatenated = Z_L_outer composed with Z_L_inner on each block
        logical_z = 0
        logical_x = 0
        
        for block_id in self.outer_decoder.Z_LOGICAL:
            base = block_id * self.inner_decoder.n_data_qubits
            block_x_errors = np.sum(combined_correction_x[base:base + self.inner_decoder.n_data_qubits][self.inner_decoder.Z_LOGICAL])
            logical_z ^= int(block_x_errors % 2)
        
        for block_id in self.outer_decoder.X_LOGICAL:
            base = block_id * self.inner_decoder.n_data_qubits
            block_z_errors = np.sum(combined_correction_z[base:base + self.inner_decoder.n_data_qubits][self.inner_decoder.X_LOGICAL])
            logical_x ^= int(block_z_errors % 2)
        
        if any_ambiguous:
            result = LocalDecoderResult(
                success=False,
                logical_value=[logical_z, logical_x],
                correction_x=combined_correction_x,
                correction_z=combined_correction_z,
                confidence=total_confidence,
            )
        else:
            result = LocalDecoderResult.success_result(
                logical_value=[logical_z, logical_x],
                correction_x=combined_correction_x,
                correction_z=combined_correction_z,
            )
        
        return result


# =============================================================================
# POST-SELECTION FUNCTIONS
# =============================================================================

def post_select_verification(
    detector_vector: np.ndarray,
    verification_groups: List[Tuple[int, int]],
    mode: str = "parity",
) -> bool:
    """
    Check if verification detectors pass.
    
    Args:
        detector_vector: Full detector bit vector
        verification_groups: List of (start, end) ranges for verification
        mode: "parity" (reject if any group has odd parity) or 
              "any" (reject if any detector fires)
              
    Returns:
        True if verification passed (shot should be kept)
        False if verification failed (shot should be rejected)
    """
    for start, end in verification_groups:
        group = detector_vector[start:end]
        
        if mode == "parity":
            if np.sum(group) % 2 != 0:
                return False
        elif mode == "any":
            if np.any(group):
                return False
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    return True


def filter_shots_by_verification(
    detector_samples: np.ndarray,
    verification_groups: List[Tuple[int, int]],
    mode: str = "parity",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter detector samples by verification.
    
    Args:
        detector_samples: Shape (n_shots, n_detectors)
        verification_groups: List of (start, end) ranges
        mode: Verification mode
        
    Returns:
        (accepted_samples, accepted_indices)
    """
    accepted = []
    indices = []
    
    for i, sample in enumerate(detector_samples):
        if post_select_verification(sample, verification_groups, mode):
            accepted.append(sample)
            indices.append(i)
    
    return np.array(accepted), np.array(indices)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_DECODER_REGISTRY: Dict[str, type] = {
    "steane": SteaneDecoder,
    "shor": ShorDecoder,
}


def get_decoder_for_code(code_name: str, **kwargs) -> LocalDecoder:
    """
    Get appropriate decoder for a code.
    
    Args:
        code_name: Name of the code ("steane", "shor", etc.)
        **kwargs: Additional arguments for decoder constructor
        
    Returns:
        LocalDecoder instance
    """
    name_lower = code_name.lower()
    
    if name_lower in _DECODER_REGISTRY:
        return _DECODER_REGISTRY[name_lower](**kwargs)
    
    if "repetition" in name_lower:
        # Extract n from name like "repetition_3"
        parts = name_lower.split("_")
        n = int(parts[1]) if len(parts) > 1 else 3
        return RepetitionDecoder(n=n)
    
    raise ValueError(f"Unknown code: {code_name}. Available: {list(_DECODER_REGISTRY.keys())}")


def register_decoder(code_name: str, decoder_class: type) -> None:
    """Register a custom decoder."""
    _DECODER_REGISTRY[code_name.lower()] = decoder_class
