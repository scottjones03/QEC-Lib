"""Bacon-Shor Subsystem Code

The Bacon-Shor code is a subsystem code where some degrees of freedom
are designated as "gauge qubits" that don't encode information. This
allows for simplified syndrome measurement using only 2-body operators.

For an m×n Bacon-Shor code:
- n_physical = m × n qubits
- k = 1 logical qubit
- d = min(m, n) distance
- Gauge operators are weight-2 XX and ZZ on adjacent pairs
- Stabilizers are products of gauge operators

Key advantage: Only weight-2 measurements needed, unlike surface codes
which require weight-4.

Reference: Bacon, "Operator quantum error-correcting subsystems for 
self-correcting quantum memories" (2006)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString

Coord2D = Tuple[float, float]


class BaconShorCode(CSSCode):
    """
    Bacon-Shor subsystem code.
    
    Parameters
    ----------
    m : int
        Number of rows (>= 2)
    n : int
        Number of columns (>= 2)
    """

    def __init__(self, m: int = 3, n: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialize Bacon-Shor code on m×n grid."""
        if m < 2 or n < 2:
            raise ValueError(f"Grid must be at least 2×2, got {m}×{n}")
        
        n_qubits = m * n
        
        def qubit_idx(row: int, col: int) -> int:
            return row * n + col
        
        # Gauge operators (weight-2):
        # X-type: XX on horizontal pairs (same row)
        # Z-type: ZZ on vertical pairs (same column)
        
        # Stabilizers are products of gauge operators:
        # X stabilizers: product of XX gauges in each row (gives all-X on row)
        # Z stabilizers: product of ZZ gauges in each column (gives all-Z on column)
        
        # For CSS framework, we use the stabilizers (not gauges)
        # X-stabilizer: All X on each row (m-1 independent)
        # Z-stabilizer: All Z on each column (n-1 independent)
        
        # Actually, for Bacon-Shor the stabilizers are:
        # - Product of all X in row i times all X in row j (i<j)
        # - Product of all Z in col i times all Z in col j (i<j)
        # This gives row-pair and column-pair stabilizers
        
        # Simpler: X-type on pairs of rows, Z-type on pairs of columns
        x_stabs = []
        z_stabs = []
        
        # X stabilizers: for each pair of adjacent rows
        for r in range(m - 1):
            stab = [0] * n_qubits
            for c in range(n):
                stab[qubit_idx(r, c)] = 1
                stab[qubit_idx(r + 1, c)] = 1
            x_stabs.append(stab)
        
        # Z stabilizers: for each pair of adjacent columns
        for c in range(n - 1):
            stab = [0] * n_qubits
            for r in range(m):
                stab[qubit_idx(r, c)] = 1
                stab[qubit_idx(r, c + 1)] = 1
            z_stabs.append(stab)
        
        hx = np.array(x_stabs, dtype=np.uint8) if x_stabs else np.zeros((0, n_qubits), dtype=np.uint8)
        hz = np.array(z_stabs, dtype=np.uint8) if z_stabs else np.zeros((0, n_qubits), dtype=np.uint8)
        
        # Gauge operators (store in metadata)
        x_gauges = []  # XX on horizontal pairs
        z_gauges = []  # ZZ on vertical pairs
        
        for r in range(m):
            for c in range(n - 1):
                x_gauges.append((qubit_idx(r, c), qubit_idx(r, c + 1)))
        
        for c in range(n):
            for r in range(m - 1):
                z_gauges.append((qubit_idx(r, c), qubit_idx(r + 1, c)))
        
        # Logical operators
        # Logical X: all X on any row
        lx_support = [qubit_idx(0, c) for c in range(n)]
        
        # Logical Z: all Z on any column
        lz_support = [qubit_idx(r, 0) for r in range(m)]
        
        logical_x: List[PauliString] = [{q: 'X' for q in lx_support}]
        logical_z: List[PauliString] = [{q: 'Z' for q in lz_support}]
        
        distance = min(m, n)
        
        meta = dict(metadata or {})
        meta["name"] = f"BaconShor_{m}x{n}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["distance"] = distance
        meta["m"] = m
        meta["grid_n"] = n
        meta["is_subsystem"] = True
        meta["x_gauges"] = x_gauges
        meta["z_gauges"] = z_gauges
        
        # Grid coordinates
        coords = {}
        for r in range(m):
            for c in range(n):
                coords[qubit_idx(r, c)] = (c, m - 1 - r)
        meta["data_coords"] = [coords[i] for i in range(n_qubits)]
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)
    
    def gauge_operators(self) -> Tuple[List[PauliString], List[PauliString]]:
        """Return the gauge operators (weight-2).
        
        Returns
        -------
        Tuple[List[PauliString], List[PauliString]]
            (X-type gauges, Z-type gauges)
        """
        x_gauges = self._metadata.get("x_gauges", [])
        z_gauges = self._metadata.get("z_gauges", [])
        
        x_ops = []
        for q1, q2 in x_gauges:
            op = {q1: 'X', q2: 'X'}
            x_ops.append(op)
        
        z_ops = []
        for q1, q2 in z_gauges:
            op = {q1: 'Z', q2: 'Z'}
            z_ops.append(op)
        
        return x_ops, z_ops


# Pre-built instances
BaconShor3x3 = lambda: BaconShorCode(3, 3)
BaconShor4x4 = lambda: BaconShorCode(4, 4)
BaconShor5x5 = lambda: BaconShorCode(5, 5)
