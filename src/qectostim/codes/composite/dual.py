# src/qectostim/codes/composite/dual.py
"""
Dual Code: Swap X and Z sectors of a CSS code.

For a CSS code C with:
- Hx detecting Z errors (X-type stabilizers)
- Hz detecting X errors (Z-type stabilizers)
- logical_x and logical_z operators

The dual code C^⊥ has:
- Hx' = Hz (detects X errors with Z-type stabilizers)
- Hz' = Hx (detects Z errors with X-type stabilizers)
- logical_x' = logical_z (swapped)
- logical_z' = logical_x (swapped)

This is equivalent to applying a transversal Hadamard to all qubits,
which exchanges X ↔ Z.

The dual code has the same [[n, k, d]] parameters as the original.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from qectostim.codes.abstract_code import PauliString
from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.utils import str_to_pauli


def _normalize_pauli(pauli: Any) -> PauliString:
    """Convert string or dict pauli to PauliString dict."""
    if isinstance(pauli, str):
        return str_to_pauli(pauli)
    return pauli


def swap_pauli_type(pauli: Any) -> PauliString:
    """
    Swap X ↔ Z in a Pauli string (Hadamard conjugation).
    
    X → Z, Z → X, Y → Y (up to phase)
    
    Parameters
    ----------
    pauli : PauliString or str
        Input Pauli operator (dict or string format).
        
    Returns
    -------
    PauliString
        Hadamard-conjugated Pauli operator.
    """
    # Normalize to dict format
    pauli_dict = _normalize_pauli(pauli)
    swap_map = {'X': 'Z', 'Z': 'X', 'Y': 'Y'}
    return {q: swap_map[op] for q, op in pauli_dict.items()}


class DualCode(CSSCode):
    """
    Dual of a CSS code: swap X and Z sectors.
    
    Given a CSS code C, the dual code C^⊥ exchanges:
    - Hx ↔ Hz
    - logical_x ↔ logical_z
    
    This is the code obtained by applying transversal Hadamard
    to all physical qubits.
    
    Parameters
    ----------
    base_code : CSSCode
        The CSS code to dualize.
    metadata : dict, optional
        Additional metadata (merged with base code metadata).
        
    Attributes
    ----------
    base_code : CSSCode
        The original code.
        
    Examples
    --------
    >>> from qectostim.codes.base import RepetitionCode
    >>> rep = RepetitionCode(3)  # Bit-flip code
    >>> dual = DualCode(rep)  # Phase-flip code
    >>> print(dual.hx.shape)  # Was hz
    >>> print(dual.hz.shape)  # Was hx
    
    Notes
    -----
    The dual of the dual is the original code: (C^⊥)^⊥ = C.
    
    For self-dual codes (like the [[7,1,3]] Steane code), the dual
    is isomorphic to the original.
    """
    
    def __init__(
        self,
        base_code: CSSCode,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_code = base_code
        
        # Swap Hx and Hz
        hx_dual = base_code.hz.copy()
        hz_dual = base_code.hx.copy()
        
        # Swap logical operators with X ↔ Z
        # Original logical X (X-type operators) become logical Z (Z-type operators)
        # Original logical Z (Z-type operators) become logical X (X-type operators)
        logical_x_dual = [swap_pauli_type(lz) for lz in base_code.logical_z_ops]
        logical_z_dual = [swap_pauli_type(lx) for lx in base_code.logical_x_ops]
        
        # Merge metadata
        base_meta = base_code.extra_metadata() if hasattr(base_code, 'extra_metadata') else {}
        meta: Dict[str, Any] = dict(base_meta)
        meta.update(metadata or {})
        meta["base_code_name"] = base_code.name
        meta["is_dual"] = True
        
        super().__init__(
            hx=hx_dual,
            hz=hz_dual,
            logical_x=logical_x_dual,
            logical_z=logical_z_dual,
            metadata=meta,
        )
    
    @property
    def name(self) -> str:
        return f"Dual({self.base_code.name})"
    
    def undual(self) -> CSSCode:
        """
        Return the original (base) code.
        
        Note: DualCode(DualCode(C)) creates a new DualCode wrapping the
        first DualCode, not the original C. Use this method or access
        base_code directly to get back to the original.
        """
        return self.base_code


class SelfDualCode(CSSCode):
    """
    Marker class for self-dual CSS codes (Hx = Hz).
    
    A code is self-dual if it equals its own dual, meaning Hx = Hz
    (up to row reordering) and logical_x can be mapped to logical_z.
    
    This class is mainly for type checking and discovery purposes.
    """
    
    @classmethod
    def is_self_dual(cls, code: CSSCode, tolerance: bool = True) -> bool:
        """
        Check if a CSS code is self-dual.
        
        Parameters
        ----------
        code : CSSCode
            The code to check.
        tolerance : bool
            If True, check if row spaces are equal (not exact matrix equality).
            
        Returns
        -------
        bool
            True if the code is self-dual.
        """
        from qectostim.codes.utils import gf2_rank, gf2_rowspace
        
        hx, hz = code.hx, code.hz
        
        if hx.shape != hz.shape:
            return False
        
        if tolerance:
            # Check if row spaces are the same
            combined = np.vstack([hx, hz])
            rank_hx = gf2_rank(hx)
            rank_hz = gf2_rank(hz)
            rank_combined = gf2_rank(combined)
            return rank_hx == rank_hz == rank_combined
        else:
            # Exact matrix equality (up to row sorting)
            return np.array_equal(np.sort(hx, axis=0), np.sort(hz, axis=0))


# Convenience function
def dual(code: CSSCode, metadata: Optional[Dict[str, Any]] = None) -> DualCode:
    """
    Create the dual of a CSS code.
    
    Convenience function for DualCode(code).
    
    Parameters
    ----------
    code : CSSCode
        The code to dualize.
    metadata : dict, optional
        Additional metadata.
        
    Returns
    -------
    DualCode
        The dual code.
    """
    return DualCode(code, metadata)
