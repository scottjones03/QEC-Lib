# src/qectostim/decoders/strategies/ml.py
"""
Maximum Likelihood decoder strategy for small codes.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np

from qectostim.decoders.strategies.base import DecoderStrategy, StrategyOutput

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode


class MLStrategy(DecoderStrategy):
    """
    Maximum Likelihood decoder for small CSS codes.
    
    Exhaustively precomputes all error patterns up to max_weight
    and finds the most likely correction for each syndrome.
    
    Only practical for small codes (n ≤ 15).
    """
    
    def __init__(
        self,
        code: 'CSSCode',
        basis: str = 'Z',
        max_weight: int = 2,
        error_rate: float = 0.01,
    ):
        super().__init__(code, basis)
        self.max_weight = min(max_weight, 3)  # Cap at 3 for performance
        self.error_rate = error_rate
        
        if self.n > 20:
            raise ValueError(f"MLStrategy not practical for n={self.n} > 20")
        
        self._lookup: Dict[Tuple[int, ...], Tuple[np.ndarray, float]] = {}
        self._build_lookup()
    
    def _build_lookup(self) -> None:
        """Build syndrome → (correction, probability) lookup."""
        h = self.hz if self.basis == 'Z' else self.hx
        if h.size == 0:
            return
        
        n_checks = h.shape[0]
        p = self.error_rate
        
        # No error
        zero_syn = tuple([0] * n_checks)
        self._lookup[zero_syn] = (np.zeros(self.n, dtype=np.uint8), 1.0)
        
        # Weight-1 errors
        for q in range(self.n):
            error = np.zeros(self.n, dtype=np.uint8)
            error[q] = 1
            syn = tuple((h @ error % 2).tolist())
            prob = p * ((1 - p) ** (self.n - 1))
            if syn not in self._lookup or prob > self._lookup[syn][1]:
                self._lookup[syn] = (error.copy(), prob)
        
        # Weight-2 errors
        if self.max_weight >= 2:
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    error = np.zeros(self.n, dtype=np.uint8)
                    error[i] = error[j] = 1
                    syn = tuple((h @ error % 2).tolist())
                    prob = (p ** 2) * ((1 - p) ** (self.n - 2))
                    if syn not in self._lookup or prob > self._lookup[syn][1]:
                        self._lookup[syn] = (error.copy(), prob)
        
        # Weight-3 errors
        if self.max_weight >= 3 and self.n <= 15:
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    for k in range(j + 1, self.n):
                        error = np.zeros(self.n, dtype=np.uint8)
                        error[i] = error[j] = error[k] = 1
                        syn = tuple((h @ error % 2).tolist())
                        prob = (p ** 3) * ((1 - p) ** (self.n - 3))
                        if syn not in self._lookup or prob > self._lookup[syn][1]:
                            self._lookup[syn] = (error.copy(), prob)
    
    def decode(
        self,
        syndrome: np.ndarray,
        measurements: Optional[np.ndarray] = None,
        **kwargs
    ) -> StrategyOutput:
        """Decode using ML lookup."""
        syndrome = np.asarray(syndrome, dtype=np.uint8).flatten()
        syn_tuple = tuple(syndrome.tolist())
        
        if syn_tuple in self._lookup:
            correction, prob = self._lookup[syn_tuple]
            confidence = 1.0
        else:
            correction = np.zeros(self.n, dtype=np.uint8)
            confidence = 0.5
        
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
            confidence=confidence,
            correction=correction,
            syndrome_used=syndrome,
        )
    
    @classmethod
    def is_compatible(cls, code: 'CSSCode') -> bool:
        """ML is compatible with small codes only."""
        return hasattr(code, 'n') and code.n <= 20
