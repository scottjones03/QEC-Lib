# src/qectostim/decoders/strategies/base.py
"""
Base classes for decoder strategies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode


@dataclass
class StrategyOutput:
    """
    Output from a decoder strategy.
    
    Attributes
    ----------
    logical_value : int
        Decoded logical value (0 or 1).
    confidence : float
        Confidence in the prediction (0.0 to 1.0), or LLR.
        For hard decoders, use 1.0.
    correction : Optional[np.ndarray]
        Physical qubit correction pattern, if available.
    syndrome_used : Optional[np.ndarray]
        The syndrome that was decoded.
    metadata : Dict[str, Any]
        Additional decoder-specific info.
    """
    logical_value: int
    confidence: float = 1.0
    correction: Optional[np.ndarray] = None
    syndrome_used: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecoderStrategy(ABC):
    """
    Abstract base class for decoder strategies.
    
    A strategy decodes a syndrome to produce a logical value prediction.
    Strategies can be configured per-level in hierarchical decoding.
    
    Parameters
    ----------
    code : CSSCode
        The CSS code this strategy decodes.
    basis : str
        Decoding basis ('X' or 'Z').
    """
    
    def __init__(self, code: 'CSSCode', basis: str = 'Z'):
        self.code = code
        self.basis = basis.upper()
        self.n = code.n
        
        # Get parity check matrices
        self.hx = np.asarray(getattr(code, 'hx', np.zeros((0, self.n))), dtype=np.uint8)
        self.hz = np.asarray(getattr(code, 'hz', np.zeros((0, self.n))), dtype=np.uint8)
        
        # Get logical support
        self._z_support = self._get_logical_support('Z')
        self._x_support = self._get_logical_support('X')
    
    def _get_logical_support(self, basis: str) -> np.ndarray:
        """Get binary support vector for logical operator."""
        support = np.zeros(self.n, dtype=np.uint8)
        
        attr_names = ['lz', 'logical_z_ops'] if basis == 'Z' else ['lx', 'logical_x_ops']
        for attr in attr_names:
            if hasattr(self.code, attr):
                val = getattr(self.code, attr)
                if callable(val):
                    try:
                        val = val()
                    except:
                        continue
                if isinstance(val, np.ndarray):
                    arr = np.atleast_2d(val)
                    if arr.shape[-1] == self.n:
                        return (arr[0] != 0).astype(np.uint8)
                elif isinstance(val, list) and len(val) > 0:
                    first = val[0]
                    if isinstance(first, str):
                        target = 'Z' if basis == 'Z' else 'X'
                        for i, c in enumerate(first):
                            if i < self.n and c.upper() in (target, 'Y'):
                                support[i] = 1
                        if support.sum() > 0:
                            return support
        
        # Fallback: transversal
        return np.ones(self.n, dtype=np.uint8)
    
    @property
    def z_support(self) -> np.ndarray:
        """Binary vector of Z logical support."""
        return self._z_support
    
    @property
    def x_support(self) -> np.ndarray:
        """Binary vector of X logical support."""
        return self._x_support
    
    @abstractmethod
    def decode(
        self,
        syndrome: np.ndarray,
        measurements: Optional[np.ndarray] = None,
        **kwargs
    ) -> StrategyOutput:
        """
        Decode a syndrome.
        
        Parameters
        ----------
        syndrome : np.ndarray
            The syndrome to decode. For CSS codes:
            - Z-basis: Hz syndrome (detects X errors)
            - X-basis: Hx syndrome (detects Z errors)
        measurements : np.ndarray, optional
            Raw qubit measurements (for direct logical readout).
        **kwargs
            Additional decoder parameters.
            
        Returns
        -------
        StrategyOutput
            Decoding result with logical value and confidence.
        """
        ...
    
    def decode_batch(
        self,
        syndromes: np.ndarray,
        measurements: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[StrategyOutput]:
        """Decode multiple syndromes."""
        results = []
        for i in range(len(syndromes)):
            meas = measurements[i] if measurements is not None else None
            results.append(self.decode(syndromes[i], meas, **kwargs))
        return results
    
    def compute_syndrome(self, measurements: np.ndarray) -> np.ndarray:
        """Compute syndrome from measurements."""
        h = self.hz if self.basis == 'Z' else self.hx
        if h.size == 0:
            return np.array([], dtype=np.uint8)
        return (h @ measurements.astype(np.uint8)) % 2
    
    def compute_logical(self, measurements: np.ndarray) -> int:
        """Compute raw logical from measurements (XOR of support)."""
        support = self._z_support if self.basis == 'Z' else self._x_support
        return int(np.sum(measurements[support.astype(bool)]) % 2)
    
    @classmethod
    def is_compatible(cls, code: 'CSSCode') -> bool:
        """Check if this strategy can decode the given code."""
        return True
