# src/qectostim/decoders/strategies/mwpm.py
"""
MWPM decoder strategy using PyMatching.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import numpy as np

from qectostim.decoders.strategies.base import DecoderStrategy, StrategyOutput

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode

try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False


class MWPMStrategy(DecoderStrategy):
    """
    Minimum Weight Perfect Matching decoder using PyMatching.
    
    Only compatible with codes where the parity check matrix
    has maximum column weight ≤ 2 (graph-like codes).
    """
    
    def __init__(self, code: 'CSSCode', basis: str = 'Z'):
        if not HAS_PYMATCHING:
            raise ImportError("MWPMStrategy requires pymatching: pip install pymatching")
        
        super().__init__(code, basis)
        
        h = self.hz if self.basis == 'Z' else self.hx
        if h.size == 0:
            self._matcher = None
            return
        
        # Check column weight compatibility
        col_weights = h.sum(axis=0)
        max_weight = int(np.max(col_weights)) if col_weights.size > 0 else 0
        if max_weight > 2:
            raise ValueError(
                f"MWPMStrategy requires max column weight ≤ 2, got {max_weight}. "
                f"Use SyndromeLookupStrategy or BeliefPropagationStrategy instead."
            )
        
        self._matcher = pymatching.Matching(h)
    
    def decode(
        self,
        syndrome: np.ndarray,
        measurements: Optional[np.ndarray] = None,
        **kwargs
    ) -> StrategyOutput:
        """Decode using MWPM."""
        syndrome = np.asarray(syndrome, dtype=np.uint8).flatten()
        
        if self._matcher is None:
            # No parity checks, return raw logical
            logical_value = 0
            if measurements is not None:
                logical_value = self.compute_logical(measurements)
            return StrategyOutput(logical_value=logical_value, confidence=1.0)
        
        try:
            correction = self._matcher.decode(syndrome)
            correction = np.asarray(correction, dtype=np.uint8)
        except Exception:
            correction = np.zeros(self.n, dtype=np.uint8)
        
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
        )
    
    @classmethod
    def is_compatible(cls, code: 'CSSCode') -> bool:
        """Check if code has column weight ≤ 2."""
        if not HAS_PYMATCHING:
            return False
        
        for attr in ['hz', 'hx']:
            if hasattr(code, attr):
                h = np.asarray(getattr(code, attr))
                if h.size > 0:
                    col_weights = h.sum(axis=0)
                    if col_weights.size > 0 and np.max(col_weights) <= 2:
                        return True
        return False
