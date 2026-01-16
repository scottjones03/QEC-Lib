"""
Detector-Based Decoder for Fault-Tolerant Experiments.

This module provides the main decoder class that processes detector sampler output
using local hard-decision decoding. No temporal matching, no MWPM.

Architecture matches the reference code:
1. Detector sampler output (binary vector per shot)
2. Post-selection on verification detector groups
3. Per-round local decoding with code-specific decoders
4. Correction accumulation in software
5. Final logical outcome computation

Key Classes:
-----------
- DetectorBasedDecoder: Main decoder class
- DecodeSession: Tracks state during multi-shot decoding
- ShotResult: Result for a single shot

Usage:
------
    >>> from qectostim.decoders.detector_decoder import DetectorBasedDecoder
    >>> 
    >>> # Create decoder from metadata
    >>> decoder = DetectorBasedDecoder.from_metadata(circuit_metadata)
    >>> 
    >>> # Decode samples
    >>> results = decoder.decode_all(detector_samples)
    >>> 
    >>> # Get statistics
    >>> print(f"Logical error rate: {results.logical_error_rate}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

import numpy as np

from qectostim.experiments.gadgets.detector_primitives import (
    DetectorMap, DetectorGroup, DetectorType, CircuitMetadata,
)
from qectostim.decoders.local_decoders import (
    LocalDecoder, LocalDecoderResult, HierarchicalDecoder,
    SteaneDecoder, ShorDecoder, get_decoder_for_code,
    post_select_verification,
)


class RejectionReason(Enum):
    """Reason for shot rejection."""
    ACCEPTED = auto()
    VERIFICATION_FAILED = auto()
    EC_AMBIGUOUS = auto()      # Used if configured to reject on ambiguity
    FINAL_AMBIGUOUS = auto()


@dataclass
class ShotResult:
    """
    Result from decoding a single shot.
    
    Attributes:
        accepted: Whether shot was accepted (not rejected by post-selection)
        rejection_reason: Why shot was rejected (if applicable)
        logical_value: Decoded logical value (0 or 1)
        expected_value: Expected logical value from preparation
        is_error: Whether a logical error occurred
        is_ambiguous: Whether any decoder returned ambiguous
        confidence: Confidence in result (1.0 = certain, 0.5 = ambiguous)
        correction_x: Final X correction applied
        correction_z: Final Z correction applied
        round_results: Results from each EC round
    """
    accepted: bool
    rejection_reason: RejectionReason = RejectionReason.ACCEPTED
    logical_value: int = 0
    expected_value: int = 0
    is_error: bool = False
    is_ambiguous: bool = False
    confidence: float = 1.0
    correction_x: Optional[np.ndarray] = None
    correction_z: Optional[np.ndarray] = None
    round_results: List[LocalDecoderResult] = field(default_factory=list)
    
    @property
    def error_contribution(self) -> float:
        """
        Error contribution for this shot.
        
        For accepted shots:
        - Definite error: 1.0
        - Definite correct: 0.0
        - Ambiguous error: confidence (e.g., 0.5)
        - Ambiguous correct: 1 - confidence
        """
        if not self.accepted:
            return 0.0  # Rejected shots don't contribute
        
        if self.is_ambiguous:
            if self.is_error:
                return self.confidence
            else:
                return 1.0 - self.confidence
        else:
            return 1.0 if self.is_error else 0.0


@dataclass
class DecodingResults:
    """
    Aggregated results from decoding multiple shots.
    """
    n_shots: int
    n_accepted: int
    n_rejected: int
    n_logical_errors: int
    n_ambiguous: int
    
    # Error statistics
    logical_error_rate: float
    logical_error_rate_std: float
    fractional_error: float  # Accumulated fractional contributions
    
    # Rejection breakdown
    rejection_rate: float
    rejections_by_reason: Dict[RejectionReason, int] = field(default_factory=dict)
    
    # Per-shot results (optional, for debugging)
    shot_results: Optional[List[ShotResult]] = None
    
    @classmethod
    def from_shot_results(
        cls,
        results: List[ShotResult],
        include_shots: bool = False,
    ) -> 'DecodingResults':
        """Compute aggregate statistics from shot results."""
        n_shots = len(results)
        n_accepted = sum(1 for r in results if r.accepted)
        n_rejected = n_shots - n_accepted
        n_logical_errors = sum(1 for r in results if r.accepted and not r.is_ambiguous and r.is_error)
        n_ambiguous = sum(1 for r in results if r.accepted and r.is_ambiguous)
        
        # Fractional error from ambiguous shots
        fractional_error = sum(
            r.error_contribution 
            for r in results 
            if r.accepted and r.is_ambiguous
        )
        
        # Compute error rate
        if n_accepted > 0:
            total_error = n_logical_errors + fractional_error
            logical_error_rate = total_error / n_accepted
            logical_error_rate_std = np.sqrt(
                logical_error_rate * (1 - logical_error_rate) / n_accepted
            )
        else:
            logical_error_rate = 0.0
            logical_error_rate_std = 0.0
        
        # Rejection breakdown
        rejections_by_reason = {}
        for r in results:
            if not r.accepted:
                reason = r.rejection_reason
                rejections_by_reason[reason] = rejections_by_reason.get(reason, 0) + 1
        
        return cls(
            n_shots=n_shots,
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            n_logical_errors=n_logical_errors,
            n_ambiguous=n_ambiguous,
            logical_error_rate=logical_error_rate,
            logical_error_rate_std=logical_error_rate_std,
            fractional_error=fractional_error,
            rejection_rate=n_rejected / n_shots if n_shots > 0 else 0.0,
            rejections_by_reason=rejections_by_reason,
            shot_results=results if include_shots else None,
        )


class DetectorBasedDecoder:
    """
    Main decoder class for detector-based circuits.
    
    Implements local hard-decision decoding:
    1. Post-selection on verification detectors
    2. Per-round syndrome decoding with code-specific decoders
    3. Correction accumulation
    4. Final logical outcome computation
    
    No MWPM, no temporal matching - just local rule-based decoding.
    
    Parameters
    ----------
    local_decoder : LocalDecoder
        Decoder for single code blocks
    metadata : CircuitMetadata
        Circuit metadata with detector group info
    reject_on_ambiguity : bool
        If True, reject shots where decoder returns ambiguous
        If False, count ambiguous as 0.5 error contribution
    """
    
    def __init__(
        self,
        local_decoder: LocalDecoder,
        metadata: CircuitMetadata,
        reject_on_ambiguity: bool = False,
    ):
        self.local_decoder = local_decoder
        self.metadata = metadata
        self.reject_on_ambiguity = reject_on_ambiguity
        
        # Precompute verification ranges
        self._verification_ranges = metadata.get_all_verification_ranges()
    
    @classmethod
    def from_metadata(
        cls,
        metadata: CircuitMetadata,
        reject_on_ambiguity: bool = False,
    ) -> 'DetectorBasedDecoder':
        """
        Create decoder from circuit metadata.
        
        Automatically selects appropriate local decoder based on code info.
        """
        code_name = metadata.code_name.lower()
        
        if 'steane' in code_name or metadata.n_data_qubits == 7:
            local_decoder = SteaneDecoder()
        elif 'shor' in code_name or metadata.n_data_qubits == 9:
            local_decoder = ShorDecoder()
        else:
            # Default to Steane for unknown
            local_decoder = SteaneDecoder()
        
        return cls(
            local_decoder=local_decoder,
            metadata=metadata,
            reject_on_ambiguity=reject_on_ambiguity,
        )
    
    def decode_shot(self, detector_vector: np.ndarray) -> ShotResult:
        """
        Decode a single shot.
        
        Args:
            detector_vector: Binary detector vector for this shot
            
        Returns:
            ShotResult with decoded outcome and diagnostics
        """
        # === Step 1: Post-selection ===
        if self._verification_ranges:
            if not post_select_verification(detector_vector, self._verification_ranges):
                return ShotResult(
                    accepted=False,
                    rejection_reason=RejectionReason.VERIFICATION_FAILED,
                )
        
        # === Step 2: Decode EC rounds ===
        n_qubits = self.metadata.n_data_qubits
        accumulated_correction_x = np.zeros(n_qubits, dtype=np.uint8)
        accumulated_correction_z = np.zeros(n_qubits, dtype=np.uint8)
        
        round_results = []
        total_confidence = 1.0
        any_ambiguous = False
        
        for ec_round in self.metadata.ec_rounds:
            dmap = ec_round.detector_map
            
            # Get syndrome detector groups
            z_group = dmap.syndrome_z.get(0)
            x_group = dmap.syndrome_x.get(0)
            
            if z_group is not None and x_group is not None:
                z_det = detector_vector[z_group.start:z_group.end].astype(np.uint8)
                x_det = detector_vector[x_group.start:x_group.end].astype(np.uint8)
                
                # Handle raw ancilla mode: compute syndrome from H @ raw
                if len(z_det) > self.local_decoder.n_syndrome_bits_z:
                    # Raw ancilla measurements, need to compute syndrome
                    z_det = self._compute_syndrome(z_det, 'Z')
                    x_det = self._compute_syndrome(x_det, 'X')
                
                result = self.local_decoder.decode_syndrome(z_det, x_det)
                round_results.append(result)
                
                if not result.success:
                    any_ambiguous = True
                    total_confidence *= result.confidence
                    
                    if self.reject_on_ambiguity:
                        return ShotResult(
                            accepted=False,
                            rejection_reason=RejectionReason.EC_AMBIGUOUS,
                            round_results=round_results,
                        )
                else:
                    accumulated_correction_x ^= result.correction_x
                    accumulated_correction_z ^= result.correction_z
        
        # === Step 3: Decode final measurement ===
        if self.metadata.final_measurement is not None:
            final_group = self.metadata.final_measurement.final_data
            if final_group is not None:
                final_det = detector_vector[final_group.start:final_group.end].astype(np.uint8)
                final_data = final_det
            else:
                final_data = np.zeros(n_qubits, dtype=np.uint8)
        else:
            final_data = np.zeros(n_qubits, dtype=np.uint8)
        
        # === Step 4: Apply corrections ===
        if self.metadata.basis == 'Z':
            corrected_data = (final_data ^ accumulated_correction_x) % 2
            logical_support = self.metadata.z_logical_support
        else:
            corrected_data = (final_data ^ accumulated_correction_z) % 2
            logical_support = self.metadata.x_logical_support
        
        # === Step 5: Compute logical value ===
        if logical_support:
            logical_value = int(np.sum(corrected_data[logical_support]) % 2)
        else:
            logical_value = int(np.sum(corrected_data) % 2)
        
        # Expected value from preparation
        if self.metadata.basis == 'Z':
            expected = 0 if self.metadata.initial_state == '0' else 1
        else:
            expected = 0 if self.metadata.initial_state == '+' else 1
        
        is_error = (logical_value != expected)
        
        return ShotResult(
            accepted=True,
            logical_value=logical_value,
            expected_value=expected,
            is_error=is_error,
            is_ambiguous=any_ambiguous,
            confidence=total_confidence,
            correction_x=accumulated_correction_x,
            correction_z=accumulated_correction_z,
            round_results=round_results,
        )
    
    def _compute_syndrome(self, raw_ancilla: np.ndarray, basis: str) -> np.ndarray:
        """
        Compute syndrome from raw ancilla measurements.
        
        For Steane-style EC, syndrome = H @ raw_ancilla mod 2
        """
        if hasattr(self.local_decoder, 'H'):
            H = self.local_decoder.H
            return (H @ raw_ancilla) % 2
        else:
            # Assume raw is already syndrome
            return raw_ancilla
    
    def decode_all(
        self,
        detector_samples: np.ndarray,
        include_shots: bool = False,
    ) -> DecodingResults:
        """
        Decode all shots and compute aggregate statistics.
        
        Args:
            detector_samples: Shape (n_shots, n_detectors)
            include_shots: Whether to include per-shot results
            
        Returns:
            DecodingResults with statistics
        """
        shot_results = []
        
        for shot in detector_samples:
            result = self.decode_shot(shot)
            shot_results.append(result)
        
        return DecodingResults.from_shot_results(shot_results, include_shots)


class HierarchicalDetectorDecoder:
    """
    Decoder for concatenated codes using hierarchical decoding.
    
    Decodes inner blocks first, then outer level.
    No temporal matching - each round decoded independently.
    """
    
    def __init__(
        self,
        inner_decoder: LocalDecoder,
        outer_decoder: LocalDecoder,
        metadata: CircuitMetadata,
        reject_on_ambiguity: bool = False,
    ):
        self.inner_decoder = inner_decoder
        self.outer_decoder = outer_decoder
        self.metadata = metadata
        self.reject_on_ambiguity = reject_on_ambiguity
        
        self.n_inner_blocks = metadata.n_inner_blocks
        self._verification_ranges = metadata.get_all_verification_ranges()
        
        # Create hierarchical decoder
        self._hier_decoder = HierarchicalDecoder(
            inner_decoder=inner_decoder,
            outer_decoder=outer_decoder,
            n_inner_blocks=self.n_inner_blocks,
        )
    
    @classmethod
    def from_metadata(
        cls,
        metadata: CircuitMetadata,
        reject_on_ambiguity: bool = False,
    ) -> 'HierarchicalDetectorDecoder':
        """Create from metadata."""
        inner_name = metadata.inner_code_name.lower()
        outer_name = metadata.outer_code_name.lower()
        
        inner_decoder = SteaneDecoder()  # Default
        outer_decoder = SteaneDecoder()
        
        if 'shor' in inner_name:
            inner_decoder = ShorDecoder()
        if 'shor' in outer_name:
            outer_decoder = ShorDecoder()
        
        return cls(
            inner_decoder=inner_decoder,
            outer_decoder=outer_decoder,
            metadata=metadata,
            reject_on_ambiguity=reject_on_ambiguity,
        )
    
    def decode_shot(self, detector_vector: np.ndarray) -> ShotResult:
        """Decode a single shot using hierarchical decoding."""
        # Post-selection
        if self._verification_ranges:
            if not post_select_verification(detector_vector, self._verification_ranges):
                return ShotResult(
                    accepted=False,
                    rejection_reason=RejectionReason.VERIFICATION_FAILED,
                )
        
        n_qubits = self.metadata.n_data_qubits
        accumulated_correction_x = np.zeros(n_qubits, dtype=np.uint8)
        accumulated_correction_z = np.zeros(n_qubits, dtype=np.uint8)
        
        round_results = []
        total_confidence = 1.0
        any_ambiguous = False
        
        for ec_round in self.metadata.ec_rounds:
            dmap = ec_round.detector_map
            
            # Build inner syndrome groups from hierarchical structure
            inner_z_groups = []
            inner_x_groups = []
            
            for block_id in range(self.n_inner_blocks):
                if 1 in dmap.hierarchical and block_id in dmap.hierarchical[1]:
                    block_map = dmap.hierarchical[1][block_id]
                    
                    z_group = block_map.get('syndrome_z')
                    x_group = block_map.get('syndrome_x')
                    
                    if z_group:
                        inner_z_groups.append((z_group.start, z_group.end))
                    else:
                        inner_z_groups.append((0, 0))
                    
                    if x_group:
                        inner_x_groups.append((x_group.start, x_group.end))
                    else:
                        inner_x_groups.append((0, 0))
                else:
                    inner_z_groups.append((0, 0))
                    inner_x_groups.append((0, 0))
            
            # Hierarchical decode
            if inner_z_groups and inner_x_groups:
                result = self._hier_decoder.decode(
                    detector_vector,
                    inner_z_groups,
                    inner_x_groups,
                )
                round_results.append(result)
                
                if not result.success:
                    any_ambiguous = True
                    total_confidence *= result.confidence
                    
                    if self.reject_on_ambiguity:
                        return ShotResult(
                            accepted=False,
                            rejection_reason=RejectionReason.EC_AMBIGUOUS,
                        )
                else:
                    accumulated_correction_x ^= result.correction_x
                    accumulated_correction_z ^= result.correction_z
        
        # Final measurement
        final_data = np.zeros(n_qubits, dtype=np.uint8)
        
        if self.metadata.final_measurement is not None:
            for group in self.metadata.final_measurement.all_groups:
                if group.dtype == DetectorType.FINAL_DATA:
                    block_data = detector_vector[group.start:group.end]
                    base = group.block_id * (n_qubits // self.n_inner_blocks)
                    end = min(base + len(block_data), n_qubits)
                    final_data[base:end] = block_data[:end - base]
        
        # Apply corrections
        corrected = (final_data ^ accumulated_correction_x) % 2
        
        # Compute logical
        logical_support = self.metadata.z_logical_support or list(range(n_qubits))
        logical_value = int(np.sum(corrected[logical_support]) % 2)
        expected = 0 if self.metadata.initial_state == '0' else 1
        is_error = (logical_value != expected)
        
        return ShotResult(
            accepted=True,
            logical_value=logical_value,
            expected_value=expected,
            is_error=is_error,
            is_ambiguous=any_ambiguous,
            confidence=total_confidence,
            correction_x=accumulated_correction_x,
            correction_z=accumulated_correction_z,
        )
    
    def decode_all(
        self,
        detector_samples: np.ndarray,
        include_shots: bool = False,
    ) -> DecodingResults:
        """Decode all shots."""
        shot_results = [self.decode_shot(shot) for shot in detector_samples]
        return DecodingResults.from_shot_results(shot_results, include_shots)


# =============================================================================
# Convenience functions
# =============================================================================

def decode_detector_samples(
    detector_samples: np.ndarray,
    metadata: CircuitMetadata,
    reject_on_ambiguity: bool = False,
) -> DecodingResults:
    """
    Convenience function to decode detector samples.
    
    Automatically selects appropriate decoder based on metadata.
    """
    if metadata.n_levels > 1:
        decoder = HierarchicalDetectorDecoder.from_metadata(
            metadata, reject_on_ambiguity
        )
    else:
        decoder = DetectorBasedDecoder.from_metadata(
            metadata, reject_on_ambiguity
        )
    
    return decoder.decode_all(detector_samples)
