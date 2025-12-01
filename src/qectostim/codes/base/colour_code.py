# src/qec_to_stim/codes/base/color_code.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from src.qectostim.codes.abstract_code import CSSCode, PauliString


class ColorCode(CSSCode):
    """
    2D color code on a chosen lattice (e.g. 4.8.8).
    """

    def __init__(self, distance: int, lattice: str = "4.8.8",
                 metadata: Optional[Dict[str, Any]] = None):
        self._d = distance
        self._lattice = lattice
        hx, hz = self._build_parity_checks(distance, lattice)
        logical_x, logical_z = self._build_logicals(distance, lattice)
        meta = dict(metadata or {})
        meta.update({"distance": distance, "lattice": lattice})
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)

    @staticmethod
    def _build_parity_checks(d: int, lattice: str) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: implement color-code Hx/Hz construction for chosen lattice.
        hx = np.zeros((0, d * d), dtype=np.uint8)
        hz = np.zeros((0, d * d), dtype=np.uint8)
        return hx, hz

    @staticmethod
    def _build_logicals(d: int, lattice: str) -> Tuple[List[PauliString], List[PauliString]]:
        # TODO: build logical X/Z on boundaries or defects
        logical_x: List[PauliString] = []
        logical_z: List[PauliString] = []
        return logical_x, logical_z