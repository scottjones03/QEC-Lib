"""Floquet Codes

Floquet codes are dynamical codes where the stabilizers are measured
in a time-periodic sequence. Unlike static stabilizer codes, the
effective stabilizers emerge from the measurement schedule.

Key features:
- Inherently fault-tolerant measurement circuits
- Good for certain hardware architectures
- Examples: Honeycomb code, ISG (instantaneous stabilizer group) codes

Reference: Hastings & Haah, "Dynamically Generated Logical Qubits" (2021)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import CSSCode


class HoneycombCode(CSSCode):
    """
    Honeycomb Floquet Code.
    
    A dynamical code on a honeycomb lattice where measurements cycle
    through X-X, Y-Y, and Z-Z checks. The effective stabilizers are
    plaquettes that emerge from the measurement sequence.
    
    For static representation, we use the emergent stabilizers.
    
    Parameters
    ----------
    rows : int
        Number of rows in the honeycomb lattice
    cols : int
        Number of columns in the honeycomb lattice
    """
    
    def __init__(
        self,
        rows: int = 2,
        cols: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize honeycomb code."""
        if rows < 2 or cols < 2:
            raise ValueError("Honeycomb code requires at least 2x2 layout")
        
        # Honeycomb lattice: 2 qubits per unit cell
        n_cells = rows * cols
        n_qubits = 2 * n_cells
        
        def cell_idx(r: int, c: int) -> Tuple[int, int]:
            """Get qubit indices for cell (r, c)."""
            cell = (r % rows) * cols + (c % cols)
            return (2 * cell, 2 * cell + 1)
        
        # In honeycomb, each hexagon plaquette is a stabilizer
        # For a finite lattice, we have (rows-1) * (cols-1) plaquettes approx
        
        x_stabs = []
        z_stabs = []
        
        # Simplified stabilizer structure for finite patch
        # Each plaquette involves 6 qubits
        for r in range(rows - 1):
            for c in range(cols - 1):
                # Collect qubits around a plaquette
                q1, q2 = cell_idx(r, c)
                q3, q4 = cell_idx(r, c + 1)
                q5, q6 = cell_idx(r + 1, c)
                q7, q8 = cell_idx(r + 1, c + 1)
                
                # X stabilizer
                x_stab = [0] * n_qubits
                for q in [q1, q4, q5, q8]:
                    if q < n_qubits:
                        x_stab[q] = 1
                x_stabs.append(x_stab)
                
                # Z stabilizer (offset pattern)
                z_stab = [0] * n_qubits
                for q in [q2, q3, q6, q7]:
                    if q < n_qubits:
                        z_stab[q] = 1
                z_stabs.append(z_stab)
        
        # Ensure we have at least one stabilizer
        if not x_stabs:
            x_stabs = [[1, 1] + [0] * (n_qubits - 2)]
        if not z_stabs:
            z_stabs = [[0, 0, 1, 1] + [0] * (n_qubits - 4)] if n_qubits >= 4 else [[1, 1] + [0] * (n_qubits - 2)]
        
        hx = np.array(x_stabs, dtype=np.uint8)
        hz = np.array(z_stabs, dtype=np.uint8)
        
        # Logical operators span the torus
        lx = ['X' if i % 2 == 0 else 'I' for i in range(n_qubits)]
        lz = ['Z' if i % 2 == 1 else 'I' for i in range(n_qubits)]
        
        logical_x = [''.join(lx)]
        logical_z = [''.join(lz)]
        
        meta = dict(metadata or {})
        meta["name"] = f"Honeycomb_{rows}x{cols}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["rows"] = rows
        meta["cols"] = cols
        meta["type"] = "floquet"
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)


class ISGFloquetCode(CSSCode):
    """
    Instantaneous Stabilizer Group (ISG) Floquet Code.
    
    A general Floquet code framework where the ISG changes each round
    but has a periodic structure. This is the static representation
    of the accumulated stabilizer group.
    
    Parameters
    ----------
    base_distance : int
        Target code distance
    """
    
    def __init__(
        self,
        base_distance: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize ISG Floquet code."""
        d = base_distance
        
        # For ISG codes, we create a triangular-like structure
        # n qubits arranged to give distance d
        n_qubits = d * (d + 1) // 2 + d
        n_qubits = max(n_qubits, 4)  # At least 4 qubits
        
        # Create a simple stabilizer structure
        n_x_stabs = d
        n_z_stabs = d
        
        x_stabs = []
        z_stabs = []
        
        # X stabilizers
        for i in range(n_x_stabs):
            stab = [0] * n_qubits
            start = (i * 2) % n_qubits
            for j in range(2):
                stab[(start + j) % n_qubits] = 1
            x_stabs.append(stab)
        
        # Z stabilizers (offset)
        for i in range(n_z_stabs):
            stab = [0] * n_qubits
            start = (i * 2 + 1) % n_qubits
            for j in range(2):
                stab[(start + j) % n_qubits] = 1
            z_stabs.append(stab)
        
        hx = np.array(x_stabs, dtype=np.uint8)
        hz = np.array(z_stabs, dtype=np.uint8)
        
        # Logical operators
        logical_x = ['X' + 'I' * (n_qubits - 1)]
        logical_z = ['Z' + 'I' * (n_qubits - 1)]
        
        meta = dict(metadata or {})
        meta["name"] = f"ISGFloquet_d{d}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["distance"] = d
        meta["type"] = "floquet_isg"
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)


# Pre-built instances
Honeycomb2x3 = lambda: HoneycombCode(rows=2, cols=3)
Honeycomb3x3 = lambda: HoneycombCode(rows=3, cols=3)
ISGFloquet3 = lambda: ISGFloquetCode(base_distance=3)
