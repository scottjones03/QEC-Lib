from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional

import numpy as np

from .abstract_code import PauliString
from .abstract_homological import HomologicalCode, TopologicalCode, Coord2D
from .complexes.css_complex import CSSChainComplex3


class CSSCode(HomologicalCode):
    """
    CSS code based on a 3-chain complex: C2 --∂2--> C1 --∂1--> C0.

    Here we expose Hx and Hz as the usual parity-check matrices acting on qubits (C1).
    """

    def __init__(self, hx: np.ndarray, hz: np.ndarray, logical_x: List[PauliString],
                 logical_z: List[PauliString], metadata: Optional[Dict[str, Any]] = None):
        # validate types & commutativity
        self._hx = np.array(hx, dtype=np.uint8)
        self._hz = np.array(hz, dtype=np.uint8)
        self._logical_x = logical_x
        self._logical_z = logical_z
        self._metadata = metadata or {}
        self._validate_css()

    def _validate_css(self) -> None:
        # basic sanity checks: shapes and Hx Hz^T = 0 mod 2
        assert self._hx.shape[1] == self._hz.shape[1], "Hx, Hz must have same number of columns (qubits)"
        comm = (self._hx @ self._hz.T) % 2
        if np.any(comm):
            raise ValueError("Hx Hz^T != 0 mod 2; not a valid CSS code")

    # --- Code interface ---

    @property
    def n(self) -> int:
        return self._hx.shape[1]

    @property
    def k(self) -> int:
        # k = n - rank(Hx) - rank(Hz)
        rank_hx = np.linalg.matrix_rank(self._hx % 2)
        rank_hz = np.linalg.matrix_rank(self._hz % 2)
        return self.n - rank_hx - rank_hz

    @property
    def logical_x_ops(self) -> List[PauliString]:
        return self._logical_x

    @property
    def logical_z_ops(self) -> List[PauliString]:
        return self._logical_z

    def stabilizers(self) -> List[PauliString]:
        # convert Hx rows into X-type stabilizers and Hz rows into Z-type stabilizers
        stabs: List[PauliString] = []
        # X stabilizers
        for row in self._hx:
            pauli = {i: "X" for i, bit in enumerate(row) if bit}
            stabs.append(pauli)
        # Z stabilizers
        for row in self._hz:
            pauli = {i: "Z" for i, bit in enumerate(row) if bit}
            stabs.append(pauli)
        return stabs

    def as_css(self) -> "CSSCode":
        return self

    @property
    def hx(self) -> np.ndarray:
        return self._hx

    @property
    def hz(self) -> np.ndarray:
        return self._hz

    def extra_metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)
    
    def build_stabilizers(self) -> None:
        """
        Satisfy the HomologicalCode API.

        For CSSCode we already store Hx/Hz and logicals directly in __init__,
        so there's nothing to construct here. Subclasses that build Hx/Hz from
        a chain complex can override this, but they don't have to.
        """
        # No-op: stabilizer data is already in self._hx / self._hz / self._logical_*.
        return

class TopologicalCSSCode(CSSCode, TopologicalCode):
    """CSS code defined by a 3-term chain complex plus geometry."""

    def __init__(
        self,
        chain_complex: CSSChainComplex3,
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        meta = dict(metadata or {})
        meta["chain_complex_3"] = chain_complex

        super().__init__(
            hx=chain_complex.hx,
            hz=chain_complex.hz,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )

    @property
    def chain_complex(self) -> CSSChainComplex3:
        return self.metadata["chain_complex_3"]

    def qubit_coords(self) -> List[Coord2D]:
        """Return 2D coordinates for data qubits.

        Topological subclasses should override this method to provide
        the geometry of the data qubits in 2D space.
        """
        raise NotImplementedError("Subclasses should implement qubit_coords() to provide qubit geometry.")
