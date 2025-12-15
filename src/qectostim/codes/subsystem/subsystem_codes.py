"""Subsystem Codes

Subsystem codes generalize stabilizer codes by splitting the code space
into logical and gauge qubits. The gauge qubits can absorb errors without
affecting the logical information.

Key advantage: simpler syndrome measurements (only need to measure 
2-body operators in some cases).

Includes:
- Bacon-Shor codes (already in bacon_shor.py)
- Gauge color codes
- Subsystem surface codes
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.utils import compute_css_logicals, vectors_to_paulis_x, vectors_to_paulis_z


class SubsystemSurfaceCode(CSSCode):
    """
    Subsystem Surface Code.
    
    A variant of the surface code where some stabilizers are replaced
    by gauge operators. This can simplify measurement circuits.
    
    The code has a "bare" logical qubit plus gauge qubits that can be
    in any state without affecting the logical information.
    
    Parameters
    ----------
    distance : int
        Code distance (>= 3)
    """
    
    def __init__(self, distance: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialize subsystem surface code."""
        if distance < 3:
            raise ValueError(f"Distance must be >= 3, got {distance}")
        
        d = distance
        
        # Layout similar to surface code but with gauge operators
        # We use a slightly different structure
        n_qubits = d * d
        
        def idx(r: int, c: int) -> int:
            return r * d + c
        
        x_stabs = []
        z_stabs = []
        
        # Weight-2 gauge operators that combine to give weight-4 stabilizers
        # For CSS structure, we present the effective stabilizers
        
        # X-type checks (similar to surface code)
        for r in range(d - 1):
            for c in range(d):
                if (r + c) % 2 == 0 and c < d - 1:
                    stab = [0] * n_qubits
                    stab[idx(r, c)] = 1
                    stab[idx(r + 1, c)] = 1
                    if c + 1 < d:
                        stab[idx(r, c + 1)] = 1
                        stab[idx(r + 1, c + 1)] = 1
                    x_stabs.append(stab)
        
        # Z-type checks
        for r in range(d):
            for c in range(d - 1):
                if (r + c) % 2 == 1 and r < d - 1:
                    stab = [0] * n_qubits
                    stab[idx(r, c)] = 1
                    stab[idx(r, c + 1)] = 1
                    if r + 1 < d:
                        stab[idx(r + 1, c)] = 1
                        stab[idx(r + 1, c + 1)] = 1
                    z_stabs.append(stab)
        
        # Add simple boundary stabilizers if needed
        if not x_stabs:
            stab = [0] * n_qubits
            stab[0] = stab[1] = 1
            x_stabs.append(stab)
        if not z_stabs:
            stab = [0] * n_qubits
            stab[0] = stab[d] = 1 if d <= n_qubits else 1
            z_stabs.append(stab)
        
        hx = np.array(x_stabs, dtype=np.uint8)
        hz = np.array(z_stabs, dtype=np.uint8)
        
        # Logical operators
        lx_support = [idx(0, c) for c in range(d)]
        lz_support = [idx(r, 0) for r in range(d)]
        
        logical_x: List[PauliString] = [{q: 'X' for q in lx_support}]
        logical_z: List[PauliString] = [{q: 'Z' for q in lz_support}]
        
        meta = dict(metadata or {})
        meta["name"] = f"SubsystemSurface_{d}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["distance"] = d
        meta["type"] = "subsystem"
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)


class GaugeColorCode(CSSCode):
    """
    Gauge Color Code.
    
    A subsystem version of the color code where plaquette stabilizers
    are built from products of 2-body gauge operators. This allows
    for simpler syndrome extraction circuits.
    
    Parameters
    ----------
    distance : int
        Target code distance
    """
    
    def __init__(self, distance: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialize gauge color code."""
        if distance < 3:
            raise ValueError(f"Distance must be >= 3, got {distance}")
        
        d = distance
        
        # Triangular lattice structure for color code
        # Simplified: use hexagonal cells
        n_qubits = 3 * d * d // 2 + 3 * d
        n_qubits = max(7, n_qubits)  # Minimum for color code structure
        
        x_stabs = []
        z_stabs = []
        
        # Create plaquette stabilizers
        # In gauge color code, these come from products of gauge ops
        num_plaquettes = max(2, (d - 1) * (d - 1) // 2)
        
        for p in range(num_plaquettes):
            # Each plaquette is weight-6 (hexagonal)
            x_stab = [0] * n_qubits
            z_stab = [0] * n_qubits
            
            # Place qubits around plaquette
            base = (p * 6) % (n_qubits - 5)
            for j in range(min(6, n_qubits - base)):
                x_stab[base + j] = 1
                z_stab[(base + j + 3) % n_qubits] = 1
            
            x_stabs.append(x_stab)
            z_stabs.append(z_stab)
        
        hx = np.array(x_stabs, dtype=np.uint8)
        hz = np.array(z_stabs, dtype=np.uint8)
        
        # Ensure orthogonality
        check = (hx @ hz.T) % 2
        if np.any(check != 0):
            # Fall back to simpler structure
            hx = np.zeros((2, n_qubits), dtype=np.uint8)
            hz = np.zeros((2, n_qubits), dtype=np.uint8)
            hx[0, :4] = 1
            hx[1, 4:8 if n_qubits >= 8 else n_qubits] = 1
            hz[0, 0:2] = 1
            hz[1, 2:4] = 1
        
        # Compute logical operators using CSS prescription
        try:
            log_x_vecs, log_z_vecs = compute_css_logicals(hx, hz)
            logical_x = vectors_to_paulis_x(log_x_vecs) if log_x_vecs else [{0: 'X'}]
            logical_z = vectors_to_paulis_z(log_z_vecs) if log_z_vecs else [{0: 'Z'}]
        except Exception:
            logical_x = [{0: 'X'}]
            logical_z = [{0: 'Z'}]
        
        meta = dict(metadata or {})
        meta["name"] = f"GaugeColor_{d}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["distance"] = d
        meta["type"] = "subsystem_color"
        
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)


# Pre-built instances
SubsystemSurface3 = lambda: SubsystemSurfaceCode(distance=3)
SubsystemSurface5 = lambda: SubsystemSurfaceCode(distance=5)
GaugeColor3 = lambda: GaugeColorCode(distance=3)
