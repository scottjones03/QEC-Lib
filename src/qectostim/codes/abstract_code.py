# src/qec_to_stim/codes/abstract_code.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

PauliString = Dict[int, str]  # e.g. {0: 'X', 3: 'Z'} means X on qubit0, Z on qubit3


class Code(ABC):
    """
    Abstract base class for ANY quantum code (stabilizer, subsystem, homological, etc.).
    This is the interface Experiments talk to.
    """

    @property
    @abstractmethod
    def n(self) -> int:
        """Number of physical qubits."""

    @property
    @abstractmethod
    def k(self) -> int:
        """Number of logical qubits."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # --- Logical operators ---

    @abstractmethod
    def logical_x_ops(self) -> List[PauliString]:
        """Logical X operators (one per logical qubit)."""

    @abstractmethod
    def logical_z_ops(self) -> List[PauliString]:
        """Logical Z operators (one per logical qubit)."""

    # --- Stabilizers / gauge ---

    @abstractmethod
    def stabilizers(self) -> List[PauliString]:
        """List of stabilizer generators as Pauli strings."""

    def gauge_ops(self) -> List[PauliString]:
        """Optional: gauge operators if this is a subsystem code."""
        return []

    # --- Geometry / metadata (optional but useful) ---

    def qubit_coords(self) -> Optional[List[Tuple[float, float]]]:
        """
        Optional 2D embedding for visualization / nearest-neighbour constraints.
        None if not defined.
        """
        return None

    # --- Homological / CSS-specific accessors (may be overridden) ---

    def as_css(self) -> Optional["CSSCode"]:
        """Return self as a CSSCode if applicable, else None."""
        return None

    def extra_metadata(self) -> Dict[str, Any]:
        """Arbitrary metadata for advanced use."""
        return {}


class HomologicalCode(Code, ABC):
    """
    Abstract class for codes represented as chain complexes.

    This is intentionally abstract: you might have C_{q+1} -> C_q -> C_{q-1},
    with boundary maps, etc. CSSCode will be a concrete 3-chain (C2->C1->C0).
    """

    @abstractmethod
    def chain_complex(self) -> Any:
        """
        Return an object representing the chain complex (e.g. our own ChainComplex class).
        """
        ...


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

    def logical_x_ops(self) -> List[PauliString]:
        return self._logical_x

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