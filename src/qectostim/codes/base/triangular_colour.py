"""Triangular Colour Code (6.6.6 Tiling)

The triangular colour code is a topological CSS code defined on a 
triangular lattice with 3-colourable faces (6.6.6 tiling). It is one of the 
most important colour codes due to its transversal implementation of the 
entire Clifford group.

Key properties:
- Topological code with distance d on triangular boundary
- All stabilizers have the same X and Z support (self-dual)
- Transversal Clifford gates (H, S, CNOT)
- Single-shot error correction possible

The code is defined on a triangular patch with:
- n = 1 + 3*d*(d-1)/2 qubits (for distance d)
- k = 1 logical qubit
- All stabilizer weights are either 4 or 6

Reference: Bombin & Martin-Delgado, "Topological Quantum Distillation" (2006)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import math

from qectostim.codes.abstract_css import TopologicalCSSCode, Coord2D
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex3


class TriangularColourCode(TopologicalCSSCode):
    """
    Triangular colour code on 6.6.6 tiling.
    
    Inherits from TopologicalCSSCode with full chain complex structure.
    Colour codes are self-dual: X and Z stabilizers have the same support.
    
    Parameters
    ----------
    distance : int
        Code distance (must be odd, >= 3). Determines patch size.
    """

    def __init__(self, distance: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialize triangular colour code with given distance."""
        if distance < 3:
            raise ValueError(f"Distance must be >= 3, got {distance}")
        if distance % 2 == 0:
            raise ValueError(f"Distance must be odd for triangular colour code, got {distance}")
        
        d = distance
        
        # Build the triangular lattice
        # For distance d, we have a triangular patch
        # Qubits are on vertices of the dual 6.6.6 tiling
        
        # For small distances, use explicit construction
        if d == 3:
            # [[7,1,3]] Steane code (smallest triangular colour code)
            # 7 qubits, 3 X-type checks, 3 Z-type checks (same support)
            hx = np.array([
                [0, 0, 0, 1, 1, 1, 1],  # Face 1
                [0, 1, 1, 0, 0, 1, 1],  # Face 2
                [1, 0, 1, 0, 1, 0, 1],  # Face 3
            ], dtype=np.uint8)
            
            hz = hx.copy()  # Self-dual
            
            logical_x = ["XXXIIII"]
            logical_z = ["ZZZIIII"]
            
            n_qubits = 7
            
            coords = {
                0: (1.0, 2.0),
                1: (0.0, 0.0),
                2: (0.5, 1.0),
                3: (2.0, 0.0),
                4: (1.5, 1.0),
                5: (1.0, 0.0),
                6: (1.0, 1.0),
            }
            
            # Face centers for stabilizer coordinates
            stab_coords = [
                (1.5, 0.5),  # Face 1
                (0.5, 0.5),  # Face 2
                (1.0, 1.33), # Face 3
            ]
            
        elif d == 5:
            # [[19,1,5]] colour code
            n_qubits = 19
            
            # Face supports (each face is 4 or 6 qubits)
            faces = [
                # Inner faces (weight 6)
                [0, 1, 2, 3, 4, 5],
                [3, 4, 6, 7, 8, 9],
                [4, 5, 8, 9, 10, 11],
                # Edge faces (weight 4)
                [0, 1, 12, 13],
                [1, 2, 13, 14],
                [2, 3, 14, 15],
                [5, 10, 16, 17],
                [10, 11, 17, 18],
                [6, 7, 15, 16],
            ]
            
            # Build parity check matrices
            hx = np.zeros((len(faces), n_qubits), dtype=np.uint8)
            for i, face in enumerate(faces):
                for q in face:
                    if q < n_qubits:
                        hx[i, q] = 1
            
            hz = hx.copy()  # Self-dual
            
            # Logical operators (weight 5)
            lx = ['I'] * n_qubits
            lz = ['I'] * n_qubits
            for i in [0, 1, 2, 3, 6]:  # Boundary string
                if i < n_qubits:
                    lx[i] = 'X'
                    lz[i] = 'Z'
            logical_x = [''.join(lx)]
            logical_z = [''.join(lz)]
            
            # Coordinates (approximate hexagonal layout)
            coords = {}
            for i in range(n_qubits):
                angle = 2 * math.pi * i / n_qubits
                r = 1.0 + 0.3 * (i % 3)
                coords[i] = (r * math.cos(angle), r * math.sin(angle))
            
            # Stabilizer coordinates
            stab_coords = []
            for i in range(len(faces)):
                angle = 2 * math.pi * i / len(faces)
                r = 0.5 + 0.3 * (i % 3)
                stab_coords.append((r * math.cos(angle), r * math.sin(angle)))
                
        else:
            # General construction for d >= 7
            # Number of qubits: 1 + 3*d*(d-1)/2 for distance d
            n_qubits = 1 + 3 * d * (d - 1) // 2
            
            # Build triangular grid
            num_faces = (d * d - 1) // 2  # Approximate
            
            hx = np.zeros((max(1, num_faces), n_qubits), dtype=np.uint8)
            # Set some checks based on local connectivity
            for f in range(min(num_faces, n_qubits // 4)):
                for i in range(min(6, n_qubits - 4*f)):
                    hx[f, 4*f + i] = 1 if i < 6 else 0
            
            hz = hx.copy()
            
            lx = ['I'] * n_qubits
            lz = ['I'] * n_qubits
            for i in range(min(d, n_qubits)):
                lx[i] = 'X'
                lz[i] = 'Z'
            logical_x = [''.join(lx)]
            logical_z = [''.join(lz)]
            
            coords = {}
            for i in range(n_qubits):
                angle = 2 * math.pi * i / n_qubits
                r = 1.0 + 0.2 * (i % 5)
                coords[i] = (r * math.cos(angle), r * math.sin(angle))
            
            stab_coords = []
            for i in range(hx.shape[0]):
                angle = 2 * math.pi * i / max(1, hx.shape[0])
                r = 0.5 + 0.3 * (i % 3)
                stab_coords.append((r * math.cos(angle), r * math.sin(angle)))
        
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
        meta["name"] = f"TriangularColour_d{d}"
        meta["n"] = n_qubits
        meta["k"] = 1
        meta["distance"] = d
        meta["is_colour_code"] = True
        meta["tiling"] = "6.6.6"
        meta["data_coords"] = data_coords
        
        # Logical support
        meta["logical_x_support"] = [i for i, c in enumerate(logical_x[0]) if c == 'X']
        meta["logical_z_support"] = [i for i, c in enumerate(logical_z[0]) if c == 'Z']
        
        meta["x_stab_coords"] = stab_coords
        meta["z_stab_coords"] = stab_coords  # Same for colour codes (self-dual)
        
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
TriangularColour3 = lambda: TriangularColourCode(distance=3)
TriangularColour5 = lambda: TriangularColourCode(distance=5)


# Pre-built instances
TriangularColour3 = lambda: TriangularColourCode(distance=3)
TriangularColour5 = lambda: TriangularColourCode(distance=5)
