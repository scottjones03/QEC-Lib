# src/qectostim/codes/color/colour_code.py
"""
4.8.8 Colour Code Implementation.

NOTE: The current implementation is a placeholder that creates a valid CSS code
on a 3x3 grid, but it's NOT a true 4.8.8 color code (which requires self-orthogonal
stabilizers). A proper 4.8.8 implementation would need octagons and squares with
carefully chosen face supports.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from itertools import product

import numpy as np

from ..abstract_css import CSSCode, Coord2D
from ..abstract_code import PauliString
from ..utils import compute_css_logicals, vectors_to_paulis_x, vectors_to_paulis_z, gf2_kernel


class ColourCode488(CSSCode):
    """2D 4.8.8-inspired colour code on a small patch.

    NOTE: This is a placeholder implementation that creates a valid CSS code
    on a 3x3 qubit grid. A true 4.8.8 color code requires self-orthogonal
    stabilizers (Hx = Hz with Hx @ Hx^T = 0), which the simple 3x3 grid
    doesn't satisfy.
    
    The current implementation uses:
    - X stabilizers from 4 square faces
    - Z stabilizers computed to satisfy CSS commutativity
    
    Parameters
    ----------
    distance : int
        Nominal code distance (placeholder uses fixed 3x3 grid)
    """

    def __init__(self, distance: int = 3, *, metadata: Optional[Dict[str, Any]] = None):
        self._d = distance

        # Build 3x3 grid with 4 square faces
        data_coords, face_to_vertices = self._build_tiling(distance)
        n_qubits = len(data_coords)
        n_faces = len(face_to_vertices)

        # Build X-stabilizer parity check matrix from face supports
        coord_to_index = {c: i for i, c in enumerate(data_coords)}
        hx = np.zeros((n_faces, n_qubits), dtype=np.uint8)
        
        for f_idx, verts in enumerate(face_to_vertices):
            for v in verts:
                idx = coord_to_index[v]
                hx[f_idx, idx] = 1

        # For CSS commutativity, Hz must satisfy: Hx @ Hz^T = 0 mod 2
        # Hz rows must be in the kernel of Hx
        # Use the kernel of Hx as the row space of Hz
        hz = self._compute_commuting_hz(hx, n_qubits)

        # Compute logical operators from Hx, Hz using kernel/cokernel
        logical_x_vecs, logical_z_vecs = compute_css_logicals(hx, hz)
        
        # Convert binary vectors to PauliString format
        logical_x: List[PauliString] = vectors_to_paulis_x(logical_x_vecs)
        logical_z: List[PauliString] = vectors_to_paulis_z(logical_z_vecs)

        # Stabilizer coordinates
        x_stab_coords = self._face_centres(face_to_vertices)
        z_stab_coords = x_stab_coords[:len(hz)] if len(hz) > 0 else []

        # Face colors: check if valid 3-coloring exists
        # For 4-square grid, all faces share the central qubit, forming K4 graph - NOT 3-colorable!
        # ColourCode488 is actually a 4.8.8 tiling approximation, not a true color code
        stab_colors = [i % 3 for i in range(n_faces)]  # Placeholder (not actually valid)
        
        # Check if this is actually Chromobius-compatible by checking if stabilizers
        # can be 3-colored (overlapping stabilizers need different colors)
        is_3_colorable = self._check_3_colorable(hx)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update(
            {
                "distance": distance,
                "data_coords": data_coords,
                "x_stab_coords": x_stab_coords,
                "z_stab_coords": z_stab_coords,
                "stab_colors": stab_colors,  # Placeholder colors (may not be valid 3-coloring)
                "is_chromobius_compatible": is_3_colorable,  # Only if actually 3-colorable
                "dimension": 2,
                "chain_length": 3,
                "note": "Placeholder CSS code, not true 4.8.8 color code" + 
                       (" - NOT 3-colorable" if not is_3_colorable else ""),
            }
        )

        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )

    @staticmethod
    def _compute_commuting_hz(hx: np.ndarray, n_qubits: int) -> np.ndarray:
        """Compute Hz that commutes with Hx (Hz rows in kernel of Hx).
        
        Uses a balanced approach: same number of X and Z stabilizers for
        a symmetric code with k = n - 2*rank(Hx) logical qubits.
        """
        # For CSS: need Hx @ Hz^T = 0 mod 2
        # Equivalently: Hz rows must be in nullspace of Hx
        
        kernel = gf2_kernel(hx)  # Rows of kernel span nullspace of Hx
        
        if kernel.shape[0] == 0:
            # No Z stabilizers possible
            return np.zeros((0, n_qubits), dtype=np.uint8)
        
        # Use same number of Z stabilizers as X stabilizers for balanced code
        # This gives k = n - rank(Hx) - rank(Hz) = n - 2*rank(Hx) logical qubits
        n_z_stabs = min(len(kernel), hx.shape[0])
        return kernel[:n_z_stabs].astype(np.uint8)

    @staticmethod
    def _check_3_colorable(hx: np.ndarray) -> bool:
        """Check if the stabilizer graph is 3-colorable.
        
        For Chromobius compatibility, overlapping stabilizers must have different colors.
        This checks if a valid 3-coloring exists where adjacent nodes (overlapping stabilizers)
        have different colors.
        
        Returns True if 3-colorable, False otherwise.
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
                return True
        
        return False

    # --- tiling construction -----------------------------------------------------

    @staticmethod
    def _build_tiling(distance: int) -> Tuple[List[Coord2D], List[List[Coord2D]]]:
        """Return (data_coords, faces) for a small 4.8.8-like patch.

        This is deliberately a placeholder; you can replace with a true 4.8.8
        construction from the literature.
        """
        # Simple placeholder: a 3x3 grid, 4 square faces.
        data_coords: List[Coord2D] = [
            (x, y) for x in range(3) for y in range(3)
        ]
        faces: List[List[Coord2D]] = [
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            [(1, 0), (2, 0), (2, 1), (1, 1)],
            [(0, 1), (1, 1), (1, 2), (0, 2)],
            [(1, 1), (2, 1), (2, 2), (1, 2)],
        ]
        return data_coords, faces

    @staticmethod
    def _face_centres(faces: List[List[Coord2D]]) -> List[Coord2D]:
        centres: List[Coord2D] = []
        for verts in faces:
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            centres.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        return centres

    @property
    def distance(self) -> int:
        return self._d

    def qubit_coords(self) -> List[Coord2D]:
        return list(self.metadata["data_coords"])