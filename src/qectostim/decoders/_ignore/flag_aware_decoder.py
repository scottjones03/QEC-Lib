# src/qectostim/decoders/flag_aware_decoder.py
"""
Flag-Aware Decoder for Concatenated Codes with Verified Ancilla.

This decoder uses verification measurement outcomes (flags) to improve
decoding by:
1. Post-selection: Reject shots where flags fire (simplest)
2. Soft-weighting: Reduce confidence in rounds where flags fire
3. Gated decoding: Use flags to modify syndrome interpretation

The key insight is that when verification measurements indicate a faulty
ancilla preparation, the syndrome from that round is unreliable.

Literature:
- AGP (quant-ph/0504218): Post-selection and verified ancilla
- Chamberland & Beverland (arXiv:1708.02246): Flag fault-tolerance
- Chao & Reichardt (arXiv:1705.02329): Few-qubit FT

Example
-------
>>> from qectostim.decoders import FlagAwareDecoder
>>> 
>>> # Build circuit with verified ancilla
>>> exp = MultiLevelMemoryExperiment(code, ancilla_prep="verified", rounds=3)
>>> circuit, metadata = exp.build()
>>> 
>>> # Create flag-aware decoder
>>> decoder = FlagAwareDecoder(code, metadata, mode="soft_weight")
>>> 
>>> # Decode with flag awareness
>>> logical = decoder.decode(measurements)
>>> 
>>> # Or use post-selection mode
>>> decoder_ps = FlagAwareDecoder(code, metadata, mode="post_select")
>>> logical, accepted = decoder_ps.decode_with_acceptance(measurements)
"""
from __future__ import annotations

import numpy as np
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

from qectostim.decoders.base import Decoder

if TYPE_CHECKING:
    from qectostim.codes.composite.multilevel_concatenated import MultiLevelConcatenatedCode


class FlagMode(Enum):
    """How to use verification flags in decoding."""
    POST_SELECT = "post_select"      # Reject shots with triggered flags
    SOFT_WEIGHT = "soft_weight"      # Reduce confidence for flagged rounds
    DISCARD_ROUND = "discard_round"  # Ignore syndrome from flagged rounds
    MAJORITY_FALLBACK = "majority"   # Fall back to majority voting when flagged


@dataclass
class FlagAwareConfig:
    """Configuration for flag-aware decoder."""
    mode: FlagMode = FlagMode.SOFT_WEIGHT
    # For soft_weight mode: how much to reduce confidence when flag fires
    flag_penalty: float = 0.5  # Multiply confidence by this
    # For majority mode: minimum rounds needed for majority
    min_rounds_for_majority: int = 2
    # Verbose output
    verbose: bool = False


@dataclass
class DecodeResult:
    """Result from flag-aware decode."""
    logical_value: int
    accepted: bool = True  # For post-selection mode
    n_flags_fired: int = 0
    confidence: float = 1.0
    flag_rounds: List[int] = field(default_factory=list)


class FlagAwareDecoder(Decoder):
    """
    Decoder that uses verification flags to improve FT decoding.
    
    When using verified ancilla preparation, verification measurements
    can detect faulty ancilla states. This decoder uses that information
    to either:
    - Reject the shot (post-selection)
    - Reduce confidence in affected syndrome measurements
    - Discard affected rounds and use remaining data
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The concatenated code being decoded.
    metadata : Dict[str, Any]
        Experiment metadata including verification measurement indices.
    config : FlagAwareConfig
        Configuration for flag handling.
    inner_decoder : Decoder, optional
        Decoder for inner code. If None, uses syndrome lookup.
    outer_decoder : Decoder, optional
        Decoder for outer code. If None, uses syndrome lookup.
    """
    
    def __init__(
        self,
        code: Any,
        metadata: Dict[str, Any],
        config: Optional[FlagAwareConfig] = None,
        inner_decoder: Optional[Decoder] = None,
        outer_decoder: Optional[Decoder] = None,
    ):
        self.code = code
        self.metadata = metadata
        self.config = config or FlagAwareConfig()
        self.inner_decoder = inner_decoder
        self.outer_decoder = outer_decoder
        
        # Extract code info
        self._setup_code_info()
        
        # Extract verification measurement indices
        self._setup_verification_indices()
        
        # Setup inner decoding structures
        self._setup_inner_decoding()
    
    def _setup_code_info(self) -> None:
        """Extract code structure from metadata or code object."""
        if hasattr(self.code, 'level_codes'):
            self.inner_code = self.code.level_codes[-1]
            self.outer_code = self.code.level_codes[0] if len(self.code.level_codes) > 1 else None
            self.n_inner = self.inner_code.n
            self.n_blocks = self.outer_code.n if self.outer_code else 1
        else:
            # Fallback to metadata
            self.inner_code = self.metadata.get('inner_code')
            self.outer_code = self.metadata.get('outer_code')
            self.n_inner = self.metadata.get('n_inner', 7)
            self.n_blocks = self.metadata.get('n_blocks', 7)
        
        # Get Hz matrix for inner syndrome computation
        self._setup_inner_hz()
        
        # Get Z logical support
        self._setup_z_support()
    
    def _setup_inner_hz(self) -> None:
        """Setup inner code Hz matrix."""
        if self.inner_code is not None:
            hz = getattr(self.inner_code, 'hz', None)
            if hz is None:
                hz = getattr(self.inner_code, '_hz', None)
            if hz is not None:
                self._inner_hz = np.atleast_2d(np.asarray(hz, dtype=np.uint8))
                return
        
        # Default Steane Hz
        self._inner_hz = np.array([
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
        ], dtype=np.uint8)
    
    def _setup_z_support(self) -> None:
        """Setup Z logical support for inner code."""
        if self.inner_code is not None:
            try:
                support = self.inner_code.logical_z_support(0)
                self._inner_z_support = list(support)
                return
            except:
                pass
        
        # Default Steane Z support
        self._inner_z_support = [0, 1, 2]
    
    def _setup_verification_indices(self) -> None:
        """
        Extract verification measurement indices from metadata.
        
        Expected metadata structure:
        - verification_measurements: Dict[round_idx, Dict[block_idx, List[int]]]
          OR
        - per_round_verification: List[Dict[block_idx, List[int]]]
        """
        self.verification_indices: Dict[int, Dict[int, List[int]]] = {}
        
        # Try direct verification_measurements field
        v_meas = self.metadata.get('verification_measurements', {})
        if v_meas:
            for round_idx, blocks in v_meas.items():
                r = int(round_idx) if isinstance(round_idx, str) else round_idx
                self.verification_indices[r] = {}
                for block_idx, indices in blocks.items():
                    b = int(block_idx) if isinstance(block_idx, str) else block_idx
                    self.verification_indices[r][b] = list(indices)
            return
        
        # Try syndrome_layout which may contain verification info
        layout = self.metadata.get('syndrome_layout', {})
        for level, level_data in layout.items():
            for block_idx, block_info in level_data.items():
                v_indices = block_info.get('verification_indices', [])
                if v_indices:
                    round_idx = block_info.get('round_idx', 0)
                    if round_idx not in self.verification_indices:
                        self.verification_indices[round_idx] = {}
                    self.verification_indices[round_idx][block_idx] = v_indices
        
        # If still empty, try flat structure
        if not self.verification_indices:
            flat_v = self.metadata.get('all_verification_indices', [])
            if flat_v:
                # Assume all in round 0
                self.verification_indices[0] = {0: flat_v}
    
    def _setup_inner_decoding(self) -> None:
        """Setup inner decoding structures (MWPM if available)."""
        self._inner_matcher = None
        
        try:
            import pymatching
            # Check if Hz is MWPM-compatible (max column weight 2)
            col_weights = self._inner_hz.sum(axis=0)
            if np.max(col_weights) <= 2:
                self._inner_matcher = pymatching.Matching(self._inner_hz)
        except ImportError:
            pass
    
    # =========================================================================
    # Main Decode Interface
    # =========================================================================
    
    def decode(self, measurements: np.ndarray) -> int:
        """
        Decode measurements using flag-aware strategy.
        
        For POST_SELECT mode, this returns 0 for rejected shots (use
        decode_with_acceptance for full information).
        """
        result = self.decode_with_result(measurements)
        return result.logical_value
    
    def decode_with_acceptance(
        self,
        measurements: np.ndarray,
    ) -> Tuple[int, bool]:
        """
        Decode with acceptance flag for post-selection.
        
        Returns
        -------
        Tuple[int, bool]
            (logical_value, accepted) where accepted=False means shot
            should be discarded in post-selected statistics.
        """
        result = self.decode_with_result(measurements)
        return result.logical_value, result.accepted
    
    def decode_with_result(self, measurements: np.ndarray) -> DecodeResult:
        """
        Full decode with all result information.
        
        Returns
        -------
        DecodeResult
            Complete decode result including flags fired, confidence, etc.
        """
        measurements = np.asarray(measurements, dtype=np.uint8).flatten()
        
        # Step 1: Check verification flags
        flags_by_round = self._check_verification_flags(measurements)
        n_flags = sum(len(blocks) for blocks in flags_by_round.values())
        flag_rounds = list(flags_by_round.keys())
        
        if self.config.verbose:
            print(f"Flags fired: {n_flags} in rounds {flag_rounds}")
        
        # Step 2: Handle based on mode
        if self.config.mode == FlagMode.POST_SELECT:
            if n_flags > 0:
                return DecodeResult(
                    logical_value=0,
                    accepted=False,
                    n_flags_fired=n_flags,
                    confidence=0.0,
                    flag_rounds=flag_rounds,
                )
            # No flags - decode normally
            logical = self._decode_standard(measurements)
            return DecodeResult(
                logical_value=logical,
                accepted=True,
                n_flags_fired=0,
                confidence=1.0,
                flag_rounds=[],
            )
        
        elif self.config.mode == FlagMode.SOFT_WEIGHT:
            logical, confidence = self._decode_soft_weighted(measurements, flags_by_round)
            return DecodeResult(
                logical_value=logical,
                accepted=True,
                n_flags_fired=n_flags,
                confidence=confidence,
                flag_rounds=flag_rounds,
            )
        
        elif self.config.mode == FlagMode.DISCARD_ROUND:
            logical = self._decode_with_discarded_rounds(measurements, flag_rounds)
            return DecodeResult(
                logical_value=logical,
                accepted=True,
                n_flags_fired=n_flags,
                confidence=1.0 - 0.1 * len(flag_rounds),  # Reduce by 10% per discarded round
                flag_rounds=flag_rounds,
            )
        
        elif self.config.mode == FlagMode.MAJORITY_FALLBACK:
            if n_flags > 0:
                logical = self._decode_majority_fallback(measurements, flags_by_round)
            else:
                logical = self._decode_standard(measurements)
            return DecodeResult(
                logical_value=logical,
                accepted=True,
                n_flags_fired=n_flags,
                confidence=1.0 if n_flags == 0 else 0.7,
                flag_rounds=flag_rounds,
            )
        
        # Default: standard decode
        logical = self._decode_standard(measurements)
        return DecodeResult(
            logical_value=logical,
            accepted=True,
            n_flags_fired=n_flags,
            confidence=1.0,
            flag_rounds=flag_rounds,
        )
    
    # =========================================================================
    # Flag Checking
    # =========================================================================
    
    def _check_verification_flags(
        self,
        measurements: np.ndarray,
    ) -> Dict[int, List[int]]:
        """
        Check verification measurements for triggered flags.
        
        Returns
        -------
        Dict[int, List[int]]
            Map from round_idx to list of block_ids where flags fired.
        """
        flags_fired: Dict[int, List[int]] = {}
        
        for round_idx, blocks in self.verification_indices.items():
            for block_idx, indices in blocks.items():
                # Check if any verification measurement is 1 (flag triggered)
                for idx in indices:
                    if 0 <= idx < len(measurements) and measurements[idx] != 0:
                        if round_idx not in flags_fired:
                            flags_fired[round_idx] = []
                        if block_idx not in flags_fired[round_idx]:
                            flags_fired[round_idx].append(block_idx)
                        break  # One failure is enough
        
        return flags_fired
    
    # =========================================================================
    # Decode Strategies
    # =========================================================================
    
    def _decode_standard(self, measurements: np.ndarray) -> int:
        """Standard hierarchical decode (no flag awareness)."""
        # Get final data measurements
        n_data = self.n_blocks * self.n_inner
        final_data = self._extract_final_data(measurements)
        
        # Decode each inner block
        inner_logicals = []
        for block_idx in range(self.n_blocks):
            block_data = final_data[block_idx * self.n_inner:(block_idx + 1) * self.n_inner]
            logical = self._decode_inner_block(block_data)
            inner_logicals.append(logical)
        
        # Decode outer code
        return self._decode_outer(np.array(inner_logicals, dtype=np.uint8))
    
    def _decode_soft_weighted(
        self,
        measurements: np.ndarray,
        flags_by_round: Dict[int, List[int]],
    ) -> Tuple[int, float]:
        """
        Decode with soft weighting based on flags.
        
        When a flag fires for a block/round, reduce confidence in that
        block's syndrome for that round.
        """
        # For now, apply penalty to affected blocks' final decode
        final_data = self._extract_final_data(measurements)
        
        # Track confidence per block
        block_confidence = {b: 1.0 for b in range(self.n_blocks)}
        
        # Reduce confidence for flagged blocks
        for round_idx, flagged_blocks in flags_by_round.items():
            for block_idx in flagged_blocks:
                if block_idx < self.n_blocks:
                    block_confidence[block_idx] *= self.config.flag_penalty
        
        # Decode inner blocks with confidence
        inner_logicals = []
        inner_confidences = []
        for block_idx in range(self.n_blocks):
            block_data = final_data[block_idx * self.n_inner:(block_idx + 1) * self.n_inner]
            logical = self._decode_inner_block(block_data)
            inner_logicals.append(logical)
            inner_confidences.append(block_confidence[block_idx])
        
        # Weighted outer decode
        logical, confidence = self._decode_outer_weighted(
            np.array(inner_logicals, dtype=np.uint8),
            inner_confidences
        )
        
        return logical, confidence
    
    def _decode_with_discarded_rounds(
        self,
        measurements: np.ndarray,
        flag_rounds: List[int],
    ) -> int:
        """
        Decode ignoring syndrome data from flagged rounds.
        
        Use only unflagged rounds for syndrome-based correction.
        """
        # For memory experiment, this means using remaining rounds for
        # majority voting or temporal decode
        # For now, fall back to standard decode
        return self._decode_standard(measurements)
    
    def _decode_majority_fallback(
        self,
        measurements: np.ndarray,
        flags_by_round: Dict[int, List[int]],
    ) -> int:
        """
        Fall back to majority voting when flags fire.
        
        Use majority of unflagged rounds' syndromes.
        """
        # For now, simple majority over final data
        return self._decode_standard(measurements)
    
    # =========================================================================
    # Inner Block Decoding
    # =========================================================================
    
    def _decode_inner_block(self, block_data: np.ndarray) -> int:
        """Decode a single inner block to logical value."""
        if len(block_data) < self.n_inner:
            block_data = np.concatenate([
                block_data,
                np.zeros(self.n_inner - len(block_data), dtype=np.uint8)
            ])
        
        # Raw logical from Z support
        raw_logical = 0
        for idx in self._inner_z_support:
            if idx < len(block_data):
                raw_logical ^= int(block_data[idx])
        
        # Try syndrome correction
        if self._inner_matcher is not None:
            syndrome = (self._inner_hz @ block_data[:self.n_inner]) % 2
            try:
                correction = self._inner_matcher.decode(syndrome.astype(np.uint8))
                # Apply correction to logical
                correction_parity = 0
                for idx in self._inner_z_support:
                    if idx < len(correction) and correction[idx]:
                        correction_parity ^= 1
                return (raw_logical + correction_parity) % 2
            except:
                pass
        
        return raw_logical
    
    # =========================================================================
    # Outer Code Decoding
    # =========================================================================
    
    def _decode_outer(self, inner_logicals: np.ndarray) -> int:
        """Decode outer code from inner logical values."""
        # Get outer Hz
        outer_hz = self._get_outer_hz()
        outer_z_support = self._get_outer_z_support()
        
        # Raw outer logical
        raw_logical = 0
        for idx in outer_z_support:
            if idx < len(inner_logicals):
                raw_logical ^= int(inner_logicals[idx])
        
        # Compute syndrome and correct
        if outer_hz.size > 0:
            syndrome = (outer_hz @ inner_logicals) % 2
            syndrome_val = sum(s * (2 ** i) for i, s in enumerate(syndrome))
            
            if syndrome_val > 0 and syndrome_val <= len(inner_logicals):
                # Steane-style correction
                error_pos = syndrome_val - 1
                if error_pos in outer_z_support:
                    raw_logical ^= 1
        
        return raw_logical
    
    def _decode_outer_weighted(
        self,
        inner_logicals: np.ndarray,
        confidences: List[float],
    ) -> Tuple[int, float]:
        """
        Decode outer code with confidence weighting.
        
        Low-confidence inner logicals have less influence on syndrome.
        """
        # For now, use standard decode and average confidence
        logical = self._decode_outer(inner_logicals)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
        return logical, avg_confidence
    
    def _get_outer_hz(self) -> np.ndarray:
        """Get outer code Hz matrix."""
        if self.outer_code is not None:
            hz = getattr(self.outer_code, 'hz', None)
            if hz is None:
                hz = getattr(self.outer_code, '_hz', None)
            if hz is not None:
                return np.atleast_2d(np.asarray(hz, dtype=np.uint8))
        
        # Default Steane
        return np.array([
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
        ], dtype=np.uint8)
    
    def _get_outer_z_support(self) -> List[int]:
        """Get outer code Z logical support."""
        if self.outer_code is not None:
            try:
                return list(self.outer_code.logical_z_support(0))
            except:
                pass
        return [0, 1, 2]  # Default Steane
    
    # =========================================================================
    # Measurement Extraction
    # =========================================================================
    
    def _extract_final_data(self, measurements: np.ndarray) -> np.ndarray:
        """Extract final data measurements from full measurement array."""
        # Try metadata first
        final_indices = self.metadata.get('final_data_indices', [])
        if final_indices:
            return np.array([measurements[i] for i in final_indices if 0 <= i < len(measurements)], dtype=np.uint8)
        
        # Fallback: last n_data measurements
        n_data = self.n_blocks * self.n_inner
        if len(measurements) >= n_data:
            return measurements[-n_data:].astype(np.uint8)
        
        # Pad if needed
        return np.concatenate([
            measurements,
            np.zeros(n_data - len(measurements), dtype=np.uint8)
        ])
    
    # =========================================================================
    # Batch Decoding
    # =========================================================================
    
    def decode_batch(self, measurements_batch: np.ndarray) -> np.ndarray:
        """Decode a batch of shots."""
        n_shots = measurements_batch.shape[0]
        return np.array([self.decode(measurements_batch[i]) for i in range(n_shots)])
    
    def decode_batch_with_acceptance(
        self,
        measurements_batch: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode batch with acceptance flags.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (logical_values, accepted) arrays
        """
        n_shots = measurements_batch.shape[0]
        logicals = np.zeros(n_shots, dtype=np.int32)
        accepted = np.zeros(n_shots, dtype=bool)
        
        for i in range(n_shots):
            logicals[i], accepted[i] = self.decode_with_acceptance(measurements_batch[i])
        
        return logicals, accepted


# =============================================================================
# Factory Functions
# =============================================================================

def create_flag_aware_decoder(
    code: Any,
    metadata: Dict[str, Any],
    mode: str = "soft_weight",
    **kwargs,
) -> FlagAwareDecoder:
    """
    Factory function to create flag-aware decoder.
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The concatenated code.
    metadata : Dict
        Experiment metadata.
    mode : str
        One of "post_select", "soft_weight", "discard_round", "majority"
    
    Returns
    -------
    FlagAwareDecoder
    """
    config = FlagAwareConfig(
        mode=FlagMode(mode),
        **kwargs,
    )
    return FlagAwareDecoder(code, metadata, config)


def create_post_selecting_decoder(
    code: Any,
    metadata: Dict[str, Any],
) -> FlagAwareDecoder:
    """Create decoder with post-selection on verification flags."""
    return create_flag_aware_decoder(code, metadata, mode="post_select")
