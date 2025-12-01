# src/qec_to_stim/codes/base/four_qubit_422.py
from __future__ import annotations
from typing import List, Dict, Any, Optional

import numpy as np

from src.qectostim.codes.abstract_code import CSSCode, PauliString


class FourQubitCode(CSSCode):
    """
    [[4,2,2]] "Little Shor" code.

    Stabilizers: XXXX and ZZZZ.
    Hx = Hz = [1,1,1,1].
    2 logical qubits.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        hx = np.array([[1, 1, 1, 1]], dtype=np.uint8)
        hz = np.array([[1, 1, 1, 1]], dtype=np.uint8)
        logical_x, logical_z = self._default_logicals()
        meta = dict(metadata or {})
        meta.setdefault("distance", 2)
        super().__init__(hx=hx, hz=hz, logical_x=logical_x, logical_z=logical_z, metadata=meta)

    @staticmethod
    def _default_logicals() -> (List[PauliString], List[PauliString]):
        # One reasonable choice; you can plug in your favourite
        # e.g. from earlier discussions.
        lx: List[PauliString] = [
            {2: "X", 3: "X"},  # logical X1
            {0: "X", 1: "X"},  # logical X2
        ]
        lz: List[PauliString] = [
            {3: "Z"},          # logical Z1
            {1: "Z"},          # logical Z2
        ]
        return lx, lz