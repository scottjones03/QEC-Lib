# src/qectostim/decoders/strategies/belief_propagation.py
"""
Belief Propagation decoder strategy.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import numpy as np

from qectostim.decoders.strategies.base import DecoderStrategy, StrategyOutput

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode

try:
    from ldpc import bp_decoder
    HAS_LDPC = True
except ImportError:
    HAS_LDPC = False


class BeliefPropagationStrategy(DecoderStrategy):
    """
    Belief Propagation decoder strategy.
    
    Uses the ldpc package for BP decoding. Good for LDPC codes.
    Falls back to syndrome lookup if ldpc not available.
    """
    
    def __init__(
        self,
        code: 'CSSCode',
        basis: str = 'Z',
        max_iters: int = 30,
        error_rate: float = 0.01,
    ):
        super().__init__(code, basis)
        self.max_iters = max_iters
        self.error_rate = error_rate
        
        h = self.hz if self.basis == 'Z' else self.hx
        self._decoder = None
        
        if HAS_LDPC and h.size > 0:
            try:
                # channel_probs: prior error probability per qubit
                channel_probs = np.full(self.n, error_rate)
                self._decoder = bp_decoder(
                    h,
                    channel_probs=channel_probs,
                    max_iter=max_iters,
                    bp_method="product_sum",
                )
            except Exception:
                self._decoder = None
        
        # Fallback: syndrome lookup
        if self._decoder is None:
            self._fallback_lookup = self._build_fallback_lookup(h)
    
    def _build_fallback_lookup(self, h: np.ndarray) -> dict:
        """Build simple lookup table as fallback."""
        lookup = {}
        if h.size == 0:
            return lookup
        n_checks, n_qubits = h.shape
        for q in range(n_qubits):
            syn = tuple(h[:, q].tolist())
            if syn not in lookup:
                lookup[syn] = q
        lookup[tuple([0] * n_checks)] = -1
        return lookup
    
    def decode(
        self,
        syndrome: np.ndarray,
        measurements: Optional[np.ndarray] = None,
        **kwargs
    ) -> StrategyOutput:
        """Decode using BP."""
        syndrome = np.asarray(syndrome, dtype=np.uint8).flatten()
        
        if self._decoder is not None:
            try:
                correction = self._decoder.decode(syndrome)
                correction = np.asarray(correction, dtype=np.uint8)
            except Exception:
                correction = self._fallback_decode(syndrome)
        else:
            correction = self._fallback_decode(syndrome)
        
        # Compute logical
        if measurements is not None:
            raw_logical = self.compute_logical(measurements)
            support = self._z_support if self.basis == 'Z' else self._x_support
            correction_parity = int(np.sum(correction[support.astype(bool)]) % 2)
            logical_value = (raw_logical + correction_parity) % 2
        else:
            support = self._z_support if self.basis == 'Z' else self._x_support
            logical_value = int(np.sum(correction[support.astype(bool)]) % 2)
        
        return StrategyOutput(
            logical_value=logical_value,
            confidence=1.0,
            correction=correction,
            syndrome_used=syndrome,
            metadata={'used_bp': self._decoder is not None}
        )
    
    def _fallback_decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Fallback to lookup when BP unavailable."""
        correction = np.zeros(self.n, dtype=np.uint8)
        syn_tuple = tuple(syndrome.tolist())
        error_qubit = self._fallback_lookup.get(syn_tuple, -1)
        if error_qubit >= 0:
            correction[error_qubit] = 1
        return correction
    
    @classmethod
    def is_compatible(cls, code: 'CSSCode') -> bool:
        """BP works for any CSS code."""
        return True
