"""XY and XZ Surface Code Variants

Surface code variants optimized for biased noise channels.
When noise is biased (e.g., Z errors much more common than X errors),
these codes can outperform standard surface codes.

Note: The true XZZX surface code has non-CSS stabilizers (XZZX pattern).
This module provides a CSS-compatible version using the rotated surface code
geometry for compatibility with standard CSS decoders.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
from qectostim.codes.abstract_code import PauliString, FTGadgetCodeConfig, ScheduleMode
from qectostim.codes.complexes.css_complex import CSSChainComplex3


class XZZXSurfaceCode(TopologicalCSSCode):
    """
    XZZX-style Surface Code (CSS variant).
    
    This is a CSS-compatible version that uses the same geometry as the
    rotated surface code, with proper CSS orthogonality (each X stabilizer 
    shares 0 or 2 qubits with each Z stabilizer).
    
    Inherits from TopologicalCSSCode with full chain complex structure.
    
    Parameters
    ----------
    distance : int
        Code distance (>= 2)
    """

    def __init__(self, distance: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialize XZZX-style surface code using rotated geometry."""
        if distance < 2:
            raise ValueError(f"Distance must be >= 2, got {distance}")
        
        d = distance
        
        # Build lattice using rotated surface code geometry
        # Data qubits at odd-odd coordinates
        data_coords: Set[Coord2D] = set()
        for x in range(1, 2 * d, 2):
            for y in range(1, 2 * d, 2):
                data_coords.add((float(x), float(y)))
        
        data_coords_sorted = sorted(list(data_coords), key=lambda c: (c[1], c[0]))
        coord_to_idx = {c: i for i, c in enumerate(data_coords_sorted)}
        n_qubits = len(data_coords_sorted)
        
        # Stabilizers at even-even coordinates with boundary rules
        x_stab_coords: Set[Coord2D] = set()
        z_stab_coords: Set[Coord2D] = set()
        
        for x in range(0, 2 * d + 1, 2):
            for y in range(0, 2 * d + 1, 2):
                # Parity determines X vs Z
                parity = ((x // 2) % 2) != ((y // 2) % 2)
                
                # Boundary exclusion rules
                on_left = x == 0
                on_right = x == 2 * d
                on_top = y == 0
                on_bottom = y == 2 * d
                
                if on_left and parity:
                    continue
                if on_right and parity:
                    continue
                if on_top and not parity:
                    continue
                if on_bottom and not parity:
                    continue
                
                coord = (float(x), float(y))
                if parity:
                    x_stab_coords.add(coord)
                else:
                    z_stab_coords.add(coord)
        
        # Build parity check matrices using diagonal neighbors
        hx = self._build_boundary(x_stab_coords, coord_to_idx, n_qubits)
        hz = self._build_boundary(z_stab_coords, coord_to_idx, n_qubits)
        
        # Build chain complex
        # boundary_2: shape (n_qubits, n_x_stabs + n_z_stabs)
        boundary_2_x = hx.T.astype(np.uint8)  # shape (n_qubits, n_x_stabs)
        boundary_2_z = hz.T.astype(np.uint8)  # shape (n_qubits, n_z_stabs)
        boundary_2 = np.concatenate([boundary_2_x, boundary_2_z], axis=1)
        
        # boundary_1: Empty for surface code with open boundaries
        boundary_1 = np.zeros((0, n_qubits), dtype=np.uint8)
        
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)
        
        # Logical operators - must satisfy CSS orthogonality:
        # - Logical X must commute with all Z stabilizers (hz @ X_support = 0 mod 2)
        # - Logical Z must commute with all X stabilizers (hx @ Z_support = 0 mod 2)
        #
        # For rotated surface code geometry:
        # - Logical X: left column (x=1) - this is orthogonal to hz
        # - Logical Z: top row (y=1) - this is orthogonal to hx
        x_logical_coords = {c for c in data_coords if c[0] == 1.0}  # Left column
        z_logical_coords = {c for c in data_coords if c[1] == 1.0}  # Top row
        
        x_support = sorted(coord_to_idx[c] for c in x_logical_coords)
        z_support = sorted(coord_to_idx[c] for c in z_logical_coords)
        
        lx = ['I'] * n_qubits
        lz = ['I'] * n_qubits
        for i in x_support:
            lx[i] = 'X'
        for i in z_support:
            lz[i] = 'Z'
        logical_x = [''.join(lx)]
        logical_z = [''.join(lz)]
        
        meta = dict(metadata or {})
        meta["name"] = f"XZZX_Surface_{d}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["distance"] = d
        meta["variant"] = "XZZX"
        meta["data_coords"] = data_coords_sorted
        meta["x_stab_coords"] = sorted(list(x_stab_coords))
        meta["z_stab_coords"] = sorted(list(z_stab_coords))
        
        # Measurement schedules matching rotated surface code
        # 4-phase schedule for weight-4 stabilizers with diagonal neighbors
        meta["x_schedule"] = [
            (1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
            (1.0, -1.0),
        ]
        meta["z_schedule"] = [
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (-1.0, 1.0),
        ]
        
        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)
        
        # Override the parity check matrices for proper CSS structure
        self._hx = hx.astype(np.uint8)
        self._hz = hz.astype(np.uint8)
    
    @staticmethod
    def _build_boundary(stab_coords: Set[Coord2D], coord_to_idx: Dict[Coord2D, int], 
                        n_qubits: int) -> np.ndarray:
        """Build parity check matrix from stabilizer coords using diagonal neighbors."""
        deltas = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
        rows = []
        for s in sorted(stab_coords):
            row = [0] * n_qubits
            sx, sy = s
            for dx, dy in deltas:
                nbr = (sx + dx, sy + dy)
                idx = coord_to_idx.get(nbr)
                if idx is not None:
                    row[idx] = 1
            rows.append(row)
        if not rows:
            return np.zeros((0, n_qubits), dtype=np.uint8)
        return np.array(rows, dtype=np.uint8)
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates for visualization."""
        return list(self.metadata.get("data_coords", []))
    
    def get_ft_gadget_config(self) -> FTGadgetCodeConfig:
        """
        Return FT gadget configuration for XZZX surface code.
        
        XZZX surface codes use diagonal neighbor patterns (±1, ±1) which
        can have coordinate lookup issues with certain geometries.
        We use GRAPH_COLORING scheduling to ensure all CNOTs are emitted
        correctly, avoiding non-deterministic detector errors from
        missed coordinate lookups.
        """
        return FTGadgetCodeConfig(
            schedule_mode=ScheduleMode.GRAPH_COLORING,  # Force graph coloring for diagonal patterns
            first_round_x_detectors=True,
            first_round_z_detectors=True,
            enable_metachecks=False,
        )


# Pre-built instances
XZZXSurface3 = lambda: XZZXSurfaceCode(distance=3)
XZZXSurface5 = lambda: XZZXSurfaceCode(distance=5)
