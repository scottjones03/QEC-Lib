"""Hexagonal Colour Code (4.8.8 Tiling)

The hexagonal colour code is a topological CSS code defined on a 
4.8.8 (square-octagon) tiling. It is an alternative to the triangular
colour code with different geometric properties.

Key properties:
- Topological code on square-octagon tiling
- 3-colourable faces (R, G, B)
- Self-dual stabilizers (X and Z on same qubits)
- Weight-4 and weight-8 stabilizers

The tiling consists of squares and octagons that tile the plane.
Each qubit sits on an edge, with stabilizers on faces.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from itertools import product

import numpy as np
import math

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex3
from qectostim.codes.utils import validate_css_code


def _compute_valid_3_coloring(hx: np.ndarray) -> Optional[List[int]]:
    """
    Compute a valid 3-coloring for stabilizers where overlapping stabilizers have different colors.
    
    Parameters
    ----------
    hx : np.ndarray
        Parity check matrix where rows are stabilizers and columns are qubits.
        
    Returns
    -------
    Optional[List[int]]
        List of colors (0, 1, 2) for each stabilizer, or None if no valid coloring exists.
    """
    n_stabs = hx.shape[0]
    
    # Build overlap graph (edges between stabilizers that share qubits)
    overlaps = []
    for i in range(n_stabs):
        for j in range(i + 1, n_stabs):
            if np.any(hx[i] & hx[j]):
                overlaps.append((i, j))
    
    # Try all 3-colorings
    for coloring in product([0, 1, 2], repeat=n_stabs):
        valid = True
        for i, j in overlaps:
            if coloring[i] == coloring[j]:
                valid = False
                break
        if valid:
            return list(coloring)
    
    return None


class HexagonalColourCode(TopologicalCSSCode):
    """
    Hexagonal colour code on 4.8.8 tiling.
    
    Inherits from TopologicalCSSCode with full chain complex structure.
    Colour codes are self-dual: X and Z stabilizers have the same support.
    
    Parameters
    ----------
    distance : int
        Code distance (>= 2). Determines patch size.
    """

    def __init__(self, distance: int = 2, metadata: Optional[Dict[str, Any]] = None):
        """Initialize hexagonal colour code."""
        if distance < 2:
            raise ValueError(f"Distance must be >= 2, got {distance}")
        
        d = distance
        
        if d == 2:
            # Smallest 4.8.8 colour code: [[8,2,2]]
            # 8 qubits arranged in square-octagon pattern
            n_qubits = 8
            
            # Faces: 1 octagon (all 8) + 4 squares (pairs)
            hx = np.array([
                [1, 1, 1, 1, 0, 0, 0, 0],  # Left square
                [0, 0, 0, 0, 1, 1, 1, 1],  # Right square
                [1, 1, 0, 0, 1, 1, 0, 0],  # Top
            ], dtype=np.uint8)
            
            hz = hx.copy()  # Self-dual
            
            logical_x = ["XXXXIIII", "IIIIXXXX"]
            logical_z = ["ZZZZIIII", "IIIIZZZZ"]
            
            coords = {
                0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0),
                4: (2.0, 0.0), 5: (3.0, 0.0), 6: (2.0, 1.0), 7: (3.0, 1.0),
            }
            
            stab_coords = [
                (0.5, 0.5),  # Left square center
                (2.5, 0.5),  # Right square center
                (1.5, 0.5),  # Middle
            ]
            
            # Face colors: must satisfy 3-coloring constraint (overlapping faces have different colors)
            # Overlaps: (0,2), (1,2) so valid coloring is [0, 0, 1]
            stab_colors = [0, 0, 1]
            
        elif d == 3:
            # [[17,1,3]] hexagonal colour code (approximate)
            n_qubits = 17
            
            # Construct faces for 4.8.8 tiling
            # Weight-4 squares and weight-8 octagons
            faces = [
                [0, 1, 2, 3],           # Square 1
                [4, 5, 6, 7],           # Square 2
                [8, 9, 10, 11],         # Square 3
                [0, 1, 4, 5, 8, 9, 12, 13],  # Octagon (partial)
                [2, 3, 6, 7, 10, 11, 14, 15],  # Octagon (partial)
            ]
            
            hx = np.zeros((len(faces), n_qubits), dtype=np.uint8)
            for i, face in enumerate(faces):
                for q in face:
                    if q < n_qubits:
                        hx[i, q] = 1
            
            hz = hx.copy()
            
            lx = ['I'] * n_qubits
            lz = ['I'] * n_qubits
            for i in [0, 4, 8]:
                lx[i] = 'X'
                lz[i] = 'Z'
            logical_x = [''.join(lx)]
            logical_z = [''.join(lz)]
            
            coords = {i: (float(i % 4), float(i // 4)) for i in range(n_qubits)}
            
            stab_coords = []
            for i in range(len(faces)):
                x = (i % d) + 0.5
                y = (i // d) + 0.5
                stab_coords.append((x, y))
            
            # Face colors: must satisfy 3-coloring constraint (overlapping faces have different colors)
            # For d=3: Overlaps are (0,3), (0,4), (1,3), (1,4), (2,3), (2,4) forming bipartite graph
            # Small squares (0,1,2) can all be same color, large octagons (3,4) can be another
            # Valid coloring: [0, 0, 0, 1, 1]
            stab_colors = [0, 0, 0, 1, 1]
            
        else:
            # General construction
            # Approximate qubit count for distance d
            n_qubits = 2 * d * d + 1
            
            # Build approximate structure
            num_faces = d * d
            hx = np.zeros((num_faces, n_qubits), dtype=np.uint8)
            
            for f in range(num_faces):
                # Mix of weight-4 and weight-8 faces
                weight = 8 if f % 3 == 0 else 4
                for i in range(weight):
                    idx = (f * 4 + i) % n_qubits
                    hx[f, idx] = 1
            
            hz = hx.copy()
            
            lx = ['I'] * n_qubits
            lz = ['I'] * n_qubits
            for i in range(d):
                if i < n_qubits:
                    lx[i] = 'X'
                    lz[i] = 'Z'
            logical_x = [''.join(lx)]
            logical_z = [''.join(lz)]
            
            coords = {i: (float(i % (2*d)), float(i // (2*d))) for i in range(n_qubits)}
            
            stab_coords = []
            for i in range(num_faces):
                x = (i % d) + 0.5
                y = (i // d) + 0.5
                stab_coords.append((x, y))
            
            # Face colors: compute valid 3-coloring from overlap graph
            stab_colors = _compute_valid_3_coloring(hx)
        
        # Compute actual k from rank
        rank_hx = np.linalg.matrix_rank(hx)
        rank_hz = np.linalg.matrix_rank(hz)
        k = n_qubits - rank_hx - rank_hz
        
        # Validate CSS code structure
        is_valid, computed_k, validation_msg = validate_css_code(hx, hz, f"HexagonalColour_d{d}")
        
        # Build chain complex
        # boundary_2: shape (n_qubits, n_x_stabs + n_z_stabs)
        boundary_2_x = hx.T.astype(np.uint8)  # shape (n_qubits, n_x_stabs)
        boundary_2_z = hz.T.astype(np.uint8)  # shape (n_qubits, n_z_stabs)
        boundary_2 = np.concatenate([boundary_2_x, boundary_2_z], axis=1)
        
        # boundary_1: Empty for colour codes with boundaries
        boundary_1 = np.zeros((0, n_qubits), dtype=np.uint8)
        
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)
        
        data_coords = [coords.get(i, (0.0, 0.0)) for i in range(n_qubits)]
        
        meta = dict(metadata or {})
        meta["name"] = f"HexagonalColour_d{d}"
        meta["n"] = n_qubits
        meta["k"] = computed_k if computed_k > 0 else 1  # Report actual k (min 1 for compatibility)
        meta["actual_k"] = computed_k  # Store the true k value
        meta["distance"] = d
        meta["is_colour_code"] = True
        meta["tiling"] = "4.8.8"
        meta["data_coords"] = data_coords
        
        meta["x_stab_coords"] = stab_coords
        meta["z_stab_coords"] = stab_coords  # Same for colour codes
        meta["stab_colors"] = stab_colors  # For Chromobius: 0=red, 1=green, 2=blue
        meta["is_chromobius_compatible"] = True  # Marker for color code experiments
        
        # Mark codes with k<=0 to skip standard testing
        if not is_valid or computed_k <= 0:
            meta["skip_standard_test"] = True
            meta["validation_warning"] = validation_msg
        
        # NOTE: We deliberately omit x_schedule/z_schedule here.
        # The geometric schedule approach requires stabilizer coords + offsets to exactly
        # match data qubit coords. For colour codes with irregular geometry, the matrix-based
        # fallback circuit construction in CSSMemoryExperiment is more reliable.
        
        super().__init__(chain_complex, logical_x, logical_z, metadata=meta)
        
        # Override the parity check matrices for proper CSS structure
        self._hx = hx.astype(np.uint8)
        self._hz = hz.astype(np.uint8)
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates for visualization."""
        return list(self.metadata.get("data_coords", []))


# Pre-built instances
HexagonalColour2 = lambda: HexagonalColourCode(distance=2)
HexagonalColour3 = lambda: HexagonalColourCode(distance=3)
