# src/qectostim/codes/composite/gauge_fixed.py
"""
Gauge-Fixed Codes: Convert subsystem codes to stabilizer codes.

A subsystem code has:
- Stabilizer group S ⊂ G (commuting subgroup)
- Gauge group G = S × L where L contains logical gauge operators
- Bare logical operators that commute with G but aren't in G

Gauge fixing promotes some gauge generators to stabilizers, effectively
fixing the gauge degrees of freedom to specific values.

For CSS subsystem codes:
- X-type gauge generators: Can add to Hx to fix X gauge
- Z-type gauge generators: Can add to Hz to fix Z gauge

After gauge fixing, the code becomes a standard stabilizer code with
potentially modified distance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from qectostim.codes.abstract_code import PauliString
from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.utils import (
    gf2_rank,
    gf2_rowspace,
    css_intersection_check,
    pauli_to_symplectic,
    symplectic_to_pauli,
)


class GaugeFixedCode(CSSCode):
    """
    Convert a CSS subsystem code to a stabilizer code by gauge fixing.
    
    Given a subsystem code with gauge generators, this class promotes
    selected gauge generators to stabilizers, fixing the gauge degrees
    of freedom.
    
    Parameters
    ----------
    hx_stabilizer : np.ndarray
        X-type stabilizer generators (original stabilizer group).
    hz_stabilizer : np.ndarray
        Z-type stabilizer generators.
    hx_gauge : np.ndarray
        X-type gauge generators (not in stabilizer group).
    hz_gauge : np.ndarray
        Z-type gauge generators.
    logical_x : List[PauliString]
        Logical X operators of the subsystem code.
    logical_z : List[PauliString]
        Logical Z operators.
    fix_x_gauge : bool, optional
        If True, add X gauge generators to stabilizers. Default True.
    fix_z_gauge : bool, optional
        If True, add Z gauge generators to stabilizers. Default True.
    gauge_choice : Optional[np.ndarray], optional
        Specific gauge generators to include (binary vector selecting rows).
    metadata : dict, optional
        Additional metadata.
        
    Attributes
    ----------
    original_hx_stab : np.ndarray
        Original X stabilizers before gauge fixing.
    original_hz_stab : np.ndarray
        Original Z stabilizers before gauge fixing.
    hx_gauge : np.ndarray
        X gauge generators.
    hz_gauge : np.ndarray
        Z gauge generators.
        
    Examples
    --------
    >>> # Bacon-Shor code has gauge operators that can be fixed
    >>> # After fixing, get a standard CSS code
    >>> hx_stab = np.array([[1,1,1,1,0,0,0,0,0],  # X stabilizer
    ...                     [0,0,0,0,1,1,1,1,0]])
    >>> hz_stab = np.array([[1,0,0,1,0,0,1,0,0]])  # Z stabilizer
    >>> hx_gauge = np.array([[1,1,0,0,0,0,0,0,0]])  # X gauge
    >>> hz_gauge = np.array([[1,0,0,0,1,0,0,0,0]])  # Z gauge
    >>> fixed = GaugeFixedCode(hx_stab, hz_stab, hx_gauge, hz_gauge, [], [])
    
    Notes
    -----
    Gauge fixing does not change the logical information encoded, but it
    does change the code space dimension from 2^(k + gauge_qubits) to 2^k.
    
    The distance of the gauge-fixed code may differ from the subsystem
    code distance, depending on which gauge is fixed.
    """
    
    def __init__(
        self,
        hx_stabilizer: np.ndarray,
        hz_stabilizer: np.ndarray,
        hx_gauge: np.ndarray,
        hz_gauge: np.ndarray,
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        fix_x_gauge: bool = True,
        fix_z_gauge: bool = True,
        gauge_choice: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Store originals
        self.original_hx_stab = np.array(hx_stabilizer, dtype=np.uint8)
        self.original_hz_stab = np.array(hz_stabilizer, dtype=np.uint8)
        self.hx_gauge = np.array(hx_gauge, dtype=np.uint8)
        self.hz_gauge = np.array(hz_gauge, dtype=np.uint8)
        
        # Determine n from matrices
        n_candidates = []
        for m in [hx_stabilizer, hz_stabilizer, hx_gauge, hz_gauge]:
            if m.size > 0:
                n_candidates.append(m.shape[1])
        
        if not n_candidates:
            raise ValueError("At least one matrix must be non-empty")
        
        n = n_candidates[0]
        if not all(nc == n for nc in n_candidates):
            raise ValueError("All matrices must have the same number of columns (qubits)")
        
        # Validate gauge choice
        if gauge_choice is not None:
            # Select specific gauge generators
            if fix_x_gauge and hx_gauge.size > 0:
                x_select = gauge_choice[:hx_gauge.shape[0]] if len(gauge_choice) >= hx_gauge.shape[0] else gauge_choice
                hx_gauge_selected = hx_gauge[x_select.astype(bool)] if x_select.size > 0 else np.zeros((0, n), dtype=np.uint8)
            else:
                hx_gauge_selected = np.zeros((0, n), dtype=np.uint8)
            
            if fix_z_gauge and hz_gauge.size > 0:
                z_start = hx_gauge.shape[0] if hx_gauge.size > 0 else 0
                z_select = gauge_choice[z_start:z_start + hz_gauge.shape[0]] if len(gauge_choice) > z_start else np.array([])
                hz_gauge_selected = hz_gauge[z_select.astype(bool)] if z_select.size > 0 else np.zeros((0, n), dtype=np.uint8)
            else:
                hz_gauge_selected = np.zeros((0, n), dtype=np.uint8)
        else:
            # Fix all gauge operators
            hx_gauge_selected = hx_gauge if fix_x_gauge else np.zeros((0, n), dtype=np.uint8)
            hz_gauge_selected = hz_gauge if fix_z_gauge else np.zeros((0, n), dtype=np.uint8)
        
        # Build new parity check matrices by combining stabilizers and gauge
        if hx_stabilizer.size > 0 and hx_gauge_selected.size > 0:
            hx_new = np.vstack([hx_stabilizer, hx_gauge_selected])
        elif hx_stabilizer.size > 0:
            hx_new = hx_stabilizer.copy()
        elif hx_gauge_selected.size > 0:
            hx_new = hx_gauge_selected.copy()
        else:
            hx_new = np.zeros((0, n), dtype=np.uint8)
        
        if hz_stabilizer.size > 0 and hz_gauge_selected.size > 0:
            hz_new = np.vstack([hz_stabilizer, hz_gauge_selected])
        elif hz_stabilizer.size > 0:
            hz_new = hz_stabilizer.copy()
        elif hz_gauge_selected.size > 0:
            hz_new = hz_gauge_selected.copy()
        else:
            hz_new = np.zeros((0, n), dtype=np.uint8)
        
        # Verify CSS constraint
        if not css_intersection_check(hx_new, hz_new):
            raise ValueError(
                "Gauge-fixed code violates CSS constraint: Hx @ Hz^T != 0. "
                "This may happen if incompatible gauge operators are selected."
            )
        
        # Build metadata
        meta: Dict[str, Any] = dict(metadata or {})
        meta["is_gauge_fixed"] = True
        meta["fixed_x_gauge"] = fix_x_gauge
        meta["fixed_z_gauge"] = fix_z_gauge
        meta["num_x_gauge_fixed"] = hx_gauge_selected.shape[0]
        meta["num_z_gauge_fixed"] = hz_gauge_selected.shape[0]
        
        super().__init__(
            hx=hx_new,
            hz=hz_new,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
    
    @property
    def name(self) -> str:
        return f"GaugeFixed(n={self.n}, k={self.k})"
    
    @classmethod
    def from_bacon_shor(
        cls,
        rows: int,
        cols: int,
        fix_type: str = 'both',
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'GaugeFixedCode':
        """
        Create a gauge-fixed Bacon-Shor code.
        
        The Bacon-Shor code is a subsystem code on a rows × cols grid.
        Gauge fixing gives different stabilizer codes depending on 
        which gauge is fixed.
        
        Parameters
        ----------
        rows : int
            Number of rows in the grid.
        cols : int
            Number of columns in the grid.
        fix_type : str
            'x' to fix X gauge, 'z' to fix Z gauge, 'both' for full fixing.
            
        Returns
        -------
        GaugeFixedCode
            The gauge-fixed stabilizer code.
        """
        n = rows * cols
        
        # Bacon-Shor X gauge: XX on adjacent columns in each row
        # Bacon-Shor Z gauge: ZZ on adjacent rows in each column
        
        hx_gauge_list = []
        hz_gauge_list = []
        
        # X gauge: horizontal XX pairs
        for r in range(rows):
            for c in range(cols - 1):
                row = np.zeros(n, dtype=np.uint8)
                row[r * cols + c] = 1
                row[r * cols + c + 1] = 1
                hx_gauge_list.append(row)
        
        # Z gauge: vertical ZZ pairs
        for r in range(rows - 1):
            for c in range(cols):
                row = np.zeros(n, dtype=np.uint8)
                row[r * cols + c] = 1
                row[(r + 1) * cols + c] = 1
                hz_gauge_list.append(row)
        
        hx_gauge = np.array(hx_gauge_list, dtype=np.uint8) if hx_gauge_list else np.zeros((0, n), dtype=np.uint8)
        hz_gauge = np.array(hz_gauge_list, dtype=np.uint8) if hz_gauge_list else np.zeros((0, n), dtype=np.uint8)
        
        # Stabilizers are products of gauge operators (not needed for minimal definition)
        hx_stab = np.zeros((0, n), dtype=np.uint8)
        hz_stab = np.zeros((0, n), dtype=np.uint8)
        
        # Logical X: X on any row
        log_x: PauliString = {c: 'X' for c in range(cols)}
        # Logical Z: Z on any column
        log_z: PauliString = {r * cols: 'Z' for r in range(rows)}
        
        fix_x = fix_type in ('x', 'both')
        fix_z = fix_type in ('z', 'both')
        
        meta = dict(metadata or {})
        meta['bacon_shor'] = True
        meta['grid_size'] = (rows, cols)
        
        return cls(
            hx_stabilizer=hx_stab,
            hz_stabilizer=hz_stab,
            hx_gauge=hx_gauge,
            hz_gauge=hz_gauge,
            logical_x=[log_x],
            logical_z=[log_z],
            fix_x_gauge=fix_x,
            fix_z_gauge=fix_z,
            metadata=meta,
        )


def gauge_fix_css(
    hx_stabilizer: np.ndarray,
    hz_stabilizer: np.ndarray,
    hx_gauge: np.ndarray,
    hz_gauge: np.ndarray,
    logical_x: List[PauliString],
    logical_z: List[PauliString],
    **kwargs,
) -> GaugeFixedCode:
    """
    Convenience function to create a gauge-fixed CSS code.
    
    Parameters
    ----------
    hx_stabilizer, hz_stabilizer : np.ndarray
        Stabilizer generators.
    hx_gauge, hz_gauge : np.ndarray
        Gauge generators to promote.
    logical_x, logical_z : List[PauliString]
        Logical operators.
    **kwargs
        Additional arguments passed to GaugeFixedCode.
        
    Returns
    -------
    GaugeFixedCode
        The gauge-fixed stabilizer code.
    """
    return GaugeFixedCode(
        hx_stabilizer, hz_stabilizer,
        hx_gauge, hz_gauge,
        logical_x, logical_z,
        **kwargs
    )
