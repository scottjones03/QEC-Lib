# src/qec_to_stim/codes/base/rotated_surface.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from src.qectostim.codes.abstract_code import CSSCode, PauliString


class RotatedSurfaceCode(CSSCode):
    """
    Distance-d rotated surface code.

    Builds Hx, Hz from a 2D rotated lattice layout, plus logical X/Z paths.
    """

    def __init__(self, distance: int, metadata: Optional[Dict[str, Any]] = None):
        self._d = distance
        hx, hz = self._build_parity_checks(distance)
        logical_x, logical_z = self._build_logicals(distance)
        meta = dict(metadata or {})
        meta["distance"] = distance
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)

    @staticmethod
    def _build_parity_checks(d: int) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: construct proper Hx, Hz for rotated surface code of distance d.
        # For now, placeholders:
        hx = np.zeros((0, d * d), dtype=np.uint8)
        hz = np.zeros((0, d * d), dtype=np.uint8)
        return hx, hz

    @staticmethod
    def _build_logicals(d: int) -> Tuple[List[PauliString], List[PauliString]]:
        # TODO: build minimal-weight X/Z strings along boundaries.
        logical_x: List[PauliString] = []
        logical_z: List[PauliString] = []
        return logical_x, logical_z

    @property
    def distance(self) -> int:
        return self._d

    def qubit_coords(self) -> List[Tuple[float, float]]:
        # Optional: embed as rotated lattice for visualization / Stim QUBIT_COORDS
        coords: List[Tuple[float, float]] = []
        # TODO: assign meaningful coordinates
        for i in range(self.n):
            coords.append((float(i), 0.0))
        return coords