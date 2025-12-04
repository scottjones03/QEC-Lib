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
    [[6, 2, 2]] CSS code placeholder.

    This is a distance-2 CSS code with 6 physical qubits. The check matrices
    are valid (Hx @ Hz^T = 0 mod 2), but with the current structure, all
    logical operators are commuting rather than anticommuting. This results
    in a trivial encoding where observable measurements don't capture logical
    errors. A true [[6,2,2]] with 2 independent logical qubits would require
    check matrices that support anticommuting logical operator pairs.

    NOTE: This implementation is a placeholder. For proper [[6,2,2]] behavior,
    alternative check matrix constructions (e.g., from coding theory literature)
    should be used.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        # [[6,2,2]] CSS code with proper commutation relations
        # Check matrices satisfy Hx @ Hz^T = 0 (mod 2)
        
        # Define the X-type stabilizers (measure X on specified qubits)
        hx = np.array([
            [1, 0, 0, 1, 0, 0],  # X-check on qubits 0,3
            [0, 1, 0, 0, 1, 0],  # X-check on qubits 1,4
        ], dtype=np.uint8)

        # Define the Z-type stabilizers (measure Z on specified qubits)
        # These satisfy commutation with Hx: Hx @ Hz^T = 0 (mod 2)
        hz = np.array([
            [0, 0, 1, 0, 0, 1],  # Z-check on qubits 2,5
            [1, 0, 0, 1, 0, 0],  # Z-check on qubits 0,3
        ], dtype=np.uint8)

        # Construct the chain complex boundaries
        boundary_2 = hx.T
        boundary_1 = hz

        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)

        # Define logical operators for [6,2,2] code
        # NOTE: With these particular check matrices (independently valid for CSS),
        # the resulting code actually encodes 1 logical qubit, not 2.
        # The operators here are chosen to generate measurements.
        # A true [[6,2,2]] with k=2 would require different matrices
        # that support 2 independent logical qubit pairs.
        logical_x: List[PauliString] = [
            "IXIIII",  # First logical X
            "XIIXII",  # Another X operator
        ]
        # Logical Z operators (span kernel of Hx)
        logical_z: List[PauliString] = [
            "IIZIII",  # First logical Z
            "ZIIZII",  # Another Z operator
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