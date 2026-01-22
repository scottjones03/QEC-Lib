# src/qectostim/experiments/stabilizer_rounds/utils.py
"""
Utility functions for stabilizer round builders.

Provides helper functions for parsing logical operators and getting
logical support from various code representations.
"""
from __future__ import annotations

from typing import Any, List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qectostim.codes.abstract_code import Code


def get_logical_support(code: "Code", basis: str, logical_idx: int = 0) -> List[int]:
    """
    Get logical operator support from a code.
    
    Handles both CSSCode (with logical_x_support/logical_z_support methods)
    and general codes.
    
    Parameters
    ----------
    code : Code
        The quantum error correcting code.
    basis : str
        "X" or "Z".
    logical_idx : int
        Which logical qubit.
        
    Returns
    -------
    List[int]
        Qubit indices in the logical operator support.
    """
    basis = basis.upper()
    
    # Try CSSCode methods first
    if basis == "X" and hasattr(code, 'logical_x_support'):
        return code.logical_x_support(logical_idx)
    if basis == "Z" and hasattr(code, 'logical_z_support'):
        return code.logical_z_support(logical_idx)
    
    # Helper to safely get logical ops - handles both properties and methods
    def _get_ops(attr_name):
        attr = getattr(code, attr_name, None)
        if attr is None:
            return None
        # If it's callable (a method), call it; otherwise assume it's a property/list
        if callable(attr):
            return attr()
        return attr
    
    # Fallback to parsing logical ops
    if hasattr(code, 'logical_x_ops') and basis == "X":
        ops = _get_ops('logical_x_ops')
        if ops and logical_idx < len(ops):
            support = _parse_pauli_support(ops[logical_idx], ('X', 'Y'), code.n)
            # Check for placeholder logical (all qubits = likely invalid)
            if support and len(support) < code.n:
                return support
            # Placeholder detected - return empty to signal unknown
            return []
    
    if hasattr(code, 'logical_z_ops') and basis == "Z":
        ops = _get_ops('logical_z_ops')
        if ops and logical_idx < len(ops):
            support = _parse_pauli_support(ops[logical_idx], ('Z', 'Y'), code.n)
            # Check for placeholder logical (all qubits = likely invalid)
            if support and len(support) < code.n:
                return support
            # Placeholder detected - return empty to signal unknown
            return []
    
    # Ultimate fallback - return empty to signal unknown logical operator
    # Using all qubits causes non-deterministic detectors
    return []


def _parse_pauli_support(
    pauli_op: Any,
    paulis: Tuple[str, ...],
    n: int,
) -> List[int]:
    """
    Parse Pauli operator support from various representations.
    
    Parameters
    ----------
    pauli_op : Any
        Pauli operator in string, dict, or numpy array format.
    paulis : Tuple[str, ...]
        Which Pauli types to include (e.g., ('X', 'Y') or ('Z', 'Y')).
    n : int
        Number of qubits.
        
    Returns
    -------
    List[int]
        Qubit indices where the operator has non-identity Paulis.
    """
    support = []
    if isinstance(pauli_op, str):
        for q, p in enumerate(pauli_op):
            if p in paulis:
                support.append(q)
    elif isinstance(pauli_op, dict):
        for q, p in pauli_op.items():
            if p in paulis:
                support.append(q)
    elif isinstance(pauli_op, np.ndarray):
        half = len(pauli_op) // 2
        for q in range(min(n, half)):
            has_x = bool(pauli_op[q])
            has_z = bool(pauli_op[half + q]) if half + q < len(pauli_op) else False
            if has_x and has_z:
                p = 'Y'
            elif has_x:
                p = 'X'
            elif has_z:
                p = 'Z'
            else:
                continue
            if p in paulis:
                support.append(q)
    return support
