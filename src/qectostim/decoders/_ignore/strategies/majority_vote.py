# src/qectostim/decoders/strategies/majority_vote.py
"""
Majority Vote decoder strategy for temporal redundancy.

This strategy decodes by taking a majority vote across multiple rounds
of measurements, which helps correct measurement errors.

For concatenated codes with multiple EC rounds, this provides:
- Resilience to transient measurement errors
- Better thresholds when measurement errors dominate
- Simple implementation with good practical performance
"""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np

from qectostim.decoders.strategies.base import DecoderStrategy, StrategyOutput

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode


class MajorityVoteStrategy(DecoderStrategy):
    """
    Majority vote decoder for temporal redundancy.
    
    Given measurements from multiple rounds, computes the logical value
    for each round and returns the majority vote.
    
    This is particularly useful for:
    - Memory experiments with multiple EC rounds
    - Situations where measurement errors are significant
    - Simple, fast decoding with good practical performance
    
    Parameters
    ----------
    code : CSSCode
        The CSS code to decode.
    basis : str
        Decoding basis ('X' or 'Z').
    n_rounds : int
        Number of rounds to vote over.
    use_weighted_vote : bool
        If True, weight votes by syndrome agreement.
    
    Example
    -------
    >>> from qectostim.decoders.strategies import MajorityVoteStrategy
    >>> strategy = MajorityVoteStrategy(steane_code, basis='Z', n_rounds=3)
    >>> 
    >>> # Measurements from 3 rounds
    >>> round_measurements = [round1_meas, round2_meas, round3_meas]
    >>> result = strategy.decode_multi_round(round_measurements)
    """
    
    def __init__(
        self,
        code: 'CSSCode',
        basis: str = 'Z',
        n_rounds: int = 1,
        use_weighted_vote: bool = False,
    ):
        super().__init__(code, basis)
        self.n_rounds = n_rounds
        self.use_weighted_vote = use_weighted_vote
        
        # Build syndrome lookup for per-round decoding
        self._syndrome_lookup = self._build_syndrome_lookup()
    
    def _build_syndrome_lookup(self) -> Dict[tuple, int]:
        """Build syndrome → error qubit lookup for weight-1 errors."""
        lookup = {}
        H = self.hz if self.basis == 'Z' else self.hx
        
        if H.size == 0:
            return lookup
        
        n_checks, n_qubits = H.shape
        
        # Weight-1 errors
        for qubit in range(n_qubits):
            syndrome = tuple(H[:, qubit].tolist())
            if syndrome not in lookup:
                lookup[syndrome] = qubit
        
        # Zero syndrome = no error
        lookup[tuple([0] * n_checks)] = -1
        
        return lookup
    
    def decode(
        self,
        syndrome: np.ndarray,
        measurements: Optional[np.ndarray] = None,
        **kwargs
    ) -> StrategyOutput:
        """
        Decode using syndrome lookup (single round).
        
        For multi-round decoding, use decode_multi_round().
        """
        syndrome = np.asarray(syndrome, dtype=np.uint8).flatten()
        syn_tuple = tuple(syndrome.tolist())
        
        # Get error location
        error_qubit = self._syndrome_lookup.get(syn_tuple, -2)
        
        # Build correction
        correction = np.zeros(self.n, dtype=np.uint8)
        if error_qubit >= 0:
            correction[error_qubit] = 1
        
        # Compute logical value
        if measurements is not None:
            raw_logical = self.compute_logical(measurements)
            support = self._z_support if self.basis == 'Z' else self._x_support
            correction_parity = int(np.sum(correction[support.astype(bool)]) % 2)
            logical_value = (raw_logical + correction_parity) % 2
        else:
            support = self._z_support if self.basis == 'Z' else self._x_support
            logical_value = int(np.sum(correction[support.astype(bool)]) % 2)
        
        confidence = 1.0 if error_qubit != -2 else 0.5
        
        return StrategyOutput(
            logical_value=logical_value,
            confidence=confidence,
            correction=correction,
            syndrome_used=syndrome,
        )
    
    def decode_multi_round(
        self,
        round_measurements: List[np.ndarray],
        round_syndromes: Optional[List[np.ndarray]] = None,
    ) -> StrategyOutput:
        """
        Decode using majority vote across multiple rounds.
        
        Parameters
        ----------
        round_measurements : List[np.ndarray]
            Measurements from each round (each of length n).
        round_syndromes : List[np.ndarray], optional
            Pre-computed syndromes for each round.
            
        Returns
        -------
        StrategyOutput
            Decoded result with voted logical value.
        """
        if not round_measurements:
            return StrategyOutput(logical_value=0, confidence=0.0)
        
        n_rounds = len(round_measurements)
        round_logicals = []
        round_confidences = []
        
        H = self.hz if self.basis == 'Z' else self.hx
        support = self._z_support if self.basis == 'Z' else self._x_support
        
        for r, meas in enumerate(round_measurements):
            meas = np.asarray(meas, dtype=np.uint8).flatten()
            
            # Compute syndrome for this round
            if round_syndromes is not None and r < len(round_syndromes):
                syndrome = round_syndromes[r]
            elif H.size > 0:
                syndrome = (H @ meas) % 2
            else:
                syndrome = np.array([])
            
            # Decode this round
            result = self.decode(syndrome, meas)
            round_logicals.append(result.logical_value)
            round_confidences.append(result.confidence)
        
        # Take majority vote
        if self.use_weighted_vote:
            # Weight by confidence
            weighted_sum = sum(
                l * c for l, c in zip(round_logicals, round_confidences)
            )
            total_weight = sum(round_confidences)
            if total_weight > 0:
                voted_logical = 1 if weighted_sum / total_weight > 0.5 else 0
            else:
                voted_logical = 0
            # Confidence based on agreement
            agreement = sum(1 for l in round_logicals if l == voted_logical) / n_rounds
        else:
            # Simple majority
            vote_sum = sum(round_logicals)
            voted_logical = 1 if vote_sum > n_rounds / 2 else 0
            agreement = sum(1 for l in round_logicals if l == voted_logical) / n_rounds
        
        return StrategyOutput(
            logical_value=voted_logical,
            confidence=agreement,
            metadata={
                'round_logicals': round_logicals,
                'round_confidences': round_confidences,
                'n_rounds': n_rounds,
                'vote_type': 'weighted' if self.use_weighted_vote else 'simple',
            }
        )
    
    def compute_logical(self, measurements: np.ndarray) -> int:
        """Compute raw logical value from measurements."""
        measurements = np.asarray(measurements, dtype=np.uint8).flatten()
        support = self._z_support if self.basis == 'Z' else self._x_support
        return int(np.sum(measurements[support.astype(bool)]) % 2)
    
    @classmethod
    def is_compatible(cls, code: 'CSSCode') -> bool:
        """Majority vote works for any CSS code."""
        return True


class TemporalMajorityStrategy(MajorityVoteStrategy):
    """
    Temporal majority vote with syndrome-based weighting.
    
    This enhanced version uses syndrome information to weight votes:
    - Rounds with zero syndrome (no detected error) get higher weight
    - Rounds with high-weight syndrome get lower weight
    - Consistent syndrome patterns across rounds get bonus weight
    """
    
    def __init__(
        self,
        code: 'CSSCode',
        basis: str = 'Z',
        n_rounds: int = 3,
        syndrome_weight_factor: float = 0.5,
    ):
        super().__init__(code, basis, n_rounds, use_weighted_vote=True)
        self.syndrome_weight_factor = syndrome_weight_factor
    
    def decode_multi_round(
        self,
        round_measurements: List[np.ndarray],
        round_syndromes: Optional[List[np.ndarray]] = None,
    ) -> StrategyOutput:
        """
        Decode with syndrome-weighted voting.
        
        Rounds with lower syndrome weight get higher confidence.
        """
        if not round_measurements:
            return StrategyOutput(logical_value=0, confidence=0.0)
        
        n_rounds = len(round_measurements)
        round_logicals = []
        round_weights = []
        
        H = self.hz if self.basis == 'Z' else self.hx
        
        for r, meas in enumerate(round_measurements):
            meas = np.asarray(meas, dtype=np.uint8).flatten()
            
            # Compute syndrome
            if round_syndromes is not None and r < len(round_syndromes):
                syndrome = np.asarray(round_syndromes[r], dtype=np.uint8)
            elif H.size > 0:
                syndrome = (H @ meas) % 2
            else:
                syndrome = np.array([])
            
            # Decode this round
            result = self.decode(syndrome, meas)
            round_logicals.append(result.logical_value)
            
            # Weight based on syndrome
            if len(syndrome) > 0:
                syn_weight = np.sum(syndrome)
                # Lower syndrome weight = higher confidence
                weight = 1.0 / (1.0 + self.syndrome_weight_factor * syn_weight)
            else:
                weight = 1.0
            
            round_weights.append(weight)
        
        # Weighted vote
        weighted_ones = sum(l * w for l, w in zip(round_logicals, round_weights))
        total_weight = sum(round_weights)
        
        if total_weight > 0:
            voted_logical = 1 if weighted_ones / total_weight > 0.5 else 0
        else:
            voted_logical = 0
        
        # Confidence based on weighted agreement
        agreement_weight = sum(
            w for l, w in zip(round_logicals, round_weights)
            if l == voted_logical
        )
        confidence = agreement_weight / total_weight if total_weight > 0 else 0.0
        
        return StrategyOutput(
            logical_value=voted_logical,
            confidence=confidence,
            metadata={
                'round_logicals': round_logicals,
                'round_weights': round_weights,
                'n_rounds': n_rounds,
                'vote_type': 'temporal_weighted',
            }
        )
