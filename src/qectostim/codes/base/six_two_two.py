# src/qectostim/codes/base/six_two_two.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from qectostim.codes.abstract_css import TopologicalCSSCode
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex3

Coord2D = Tuple[float, float]
class SixQubit622Code(TopologicalCSSCode):
    """
    [[6, 2, 2]] C6 code used in the C4/C6 concatenated scheme.

    This is a small CSS code formed by concatenating C4 blocks and wiring them
    with a specific stabilizer structure (see C4/C6 FT scheme).

    NOTE: Hx/Hz/logicals are currently placeholders; you should fill these
    from the construction in the C4/C6 paper / your derivation.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        # Define the X-type stabilizers
        # X stabilizers: XXXXII and IIXXXX
        hx = np.array([
            [1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1],
        ], dtype=np.uint8)

        # Define the Z-type stabilizers
        # Z stabilizers: ZZZZII and IIZZZZ
        hz = np.array([
            [1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1],
        ], dtype=np.uint8)

        # Construct the chain complex boundaries
        boundary_2 = hx.T
        boundary_1 = hz

        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        # Define logical operators for [6,2,2] code
        # Logical X operators (span kernel of Hz)
        logical_x: List[PauliString] = [
            "XXXIII",  # Logical X on first block (qubits 0-2)
            "IIIXXX",  # Logical X on second block (qubits 3-5)
        ]
        # Logical Z operators (span kernel of Hx)
        logical_z: List[PauliString] = [
            "ZZZIII",  # Logical Z on first block (qubits 0-2)
            "IIIZZI",  # Logical Z on second block (qubits 3,4)
        ]

        meta = dict(metadata or {})
        meta["name"] = "C6"
        meta["n"] = 6
        meta["k"] = 2  # by design
        meta["distance"] = 2
        # Geometric metadata for memory experiments
        data_coords_list = [
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
        ]
        meta["data_coords"] = data_coords_list
        # Two X-type checks and two Z-type checks
        meta["x_stab_coords"] = [(0.5, -0.5), (1.5, 0.5)]  # Positioned for X syndrome checks
        meta["z_stab_coords"] = [(0.5, 0.5), (1.5, 1.5)]   # Positioned for Z syndrome checks
        # Single-phase schedules
        meta["x_schedule"] = [(0.0, 0.0)]
        meta["z_schedule"] = [(0.0, 0.0)]

        super().__init__(
            chain_complex=chain_complex,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )

    @property
    def name(self) -> str:
        return "C6"

    def qubit_coords(self) -> List[Coord2D]:
        """
        Return 2D coordinates for each data qubit.

        This is a simple 2x3 grid layout for the 6 qubits.
        """
        return [
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
        ]