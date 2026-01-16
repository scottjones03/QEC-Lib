# src/qectostim/decoders/strategies/syndrome_lookup.py
"""
Syndrome lookup decoder strategy for arbitrary CSS codes.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np

from qectostim.decoders.strategies.base import DecoderStrategy, StrategyOutput

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode


class SyndromeLookupStrategy(DecoderStrategy):
    """
    Syndrome lookup decoder for arbitrary CSS codes.
    
    Precomputes syndrome → single-qubit-error mapping for weight-1 errors.
    Falls back to random guess for unknown syndromes.
    
    Works for any CSS code (no column weight restrictions).
    """
    
    def __init__(self, code: 'CSSCode', basis: str = 'Z'):
        super().__init__(code, basis)
        self._lookup_table: Dict[Tuple[int, ...], int] = {}
        self._build_lookup_table()
    
    def _build_lookup_table(self) -> None:
        """Build syndrome → error qubit mapping for weight-1 errors."""
        # For Z-basis decoding, we correct X errors using Hz
        # For X-basis decoding, we correct Z errors using Hx
        h = self.hz if self.basis == 'Z' else self.hx
        
        if h.size == 0:
            return
        
        n_checks, n_qubits = h.shape
        
        # Weight-1 errors: each column of H gives the syndrome for error on that qubit
        for qubit in range(n_qubits):
            syndrome = tuple(h[:, qubit].tolist())
            if syndrome not in self._lookup_table:
                self._lookup_table[syndrome] = qubit
        
        # Zero syndrome maps to no error
        zero_syn = tuple([0] * n_checks)
        if zero_syn not in self._lookup_table:
            self._lookup_table[zero_syn] = -1  # No error
    
    def decode(
        self,
        syndrome: np.ndarray,
        measurements: Optional[np.ndarray] = None,
        **kwargs
    ) -> StrategyOutput:
        """
        Decode using syndrome lookup.
        
        Parameters
        ----------
        syndrome : np.ndarray
            Syndrome vector.
        measurements : np.ndarray, optional
            Raw measurements for logical computation.
            
        Returns
        -------
        StrategyOutput
            Decoded logical value.
        """
        syndrome = np.asarray(syndrome, dtype=np.uint8).flatten()
        syn_tuple = tuple(syndrome.tolist())
        
        # Get correction
        error_qubit = self._lookup_table.get(syn_tuple, -2)  # -2 = unknown
        
        # Build correction vector
        correction = np.zeros(self.n, dtype=np.uint8)
        if error_qubit >= 0:
            correction[error_qubit] = 1
        
        # Compute logical value
        if measurements is not None:
            raw_logical = self.compute_logical(measurements)
            # Apply correction to logical
            support = self._z_support if self.basis == 'Z' else self._x_support
            correction_parity = int(np.sum(correction[support.astype(bool)]) % 2)
            logical_value = (raw_logical + correction_parity) % 2
        else:
            # Without measurements, return correction parity
            support = self._z_support if self.basis == 'Z' else self._x_support
            logical_value = int(np.sum(correction[support.astype(bool)]) % 2)
        
        # Confidence based on whether syndrome was known
        confidence = 1.0 if error_qubit != -2 else 0.5
        
        return StrategyOutput(
            logical_value=logical_value,
            confidence=confidence,
            correction=correction,
            syndrome_used=syndrome,
            metadata={'error_qubit': error_qubit, 'known_syndrome': error_qubit != -2}
        )
    
    @classmethod
    def is_compatible(cls, code: 'CSSCode') -> bool:
        """Syndrome lookup works for any CSS code."""
        return True
