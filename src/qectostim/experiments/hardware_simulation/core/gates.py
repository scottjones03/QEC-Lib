# src/qectostim/experiments/hardware_simulation/core/gates.py
"""
Gate specifications and native gate sets.

Defines platform-agnostic gate representations that can be specialized
for each hardware platform's native operations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Callable,
    Any,
    Sequence,
    Union,
)

import numpy as np


class GateType(Enum):
    """Categories of quantum gates."""
    SINGLE_QUBIT = auto()
    TWO_QUBIT = auto()
    MULTI_QUBIT = auto()
    MEASUREMENT = auto()
    RESET = auto()
    BARRIER = auto()


@dataclass(frozen=True)
class GateSpec:
    """Specification of a quantum gate.
    
    Platform-agnostic representation of a gate that can be mapped
    to hardware-specific implementations.
    
    Attributes
    ----------
    name : str
        Gate name (e.g., "H", "CNOT", "MS", "CZ").
    gate_type : GateType
        Category of gate.
    num_qubits : int
        Number of qubits the gate acts on.
    parameters : Tuple[str, ...]
        Names of continuous parameters (e.g., ("theta", "phi")).
    is_clifford : bool
        Whether the gate is a Clifford gate.
    is_native : bool
        Whether this is a native hardware gate (vs. decomposed).
    stim_name : Optional[str]
        Corresponding Stim gate name, if different from name.
    matrix : Optional[np.ndarray]
        Unitary matrix representation (for small gates).
    metadata : Dict[str, Any]
        Platform-specific properties.
    """
    name: str
    gate_type: GateType
    num_qubits: int
    parameters: Tuple[str, ...] = ()
    is_clifford: bool = True
    is_native: bool = False
    stim_name: Optional[str] = None
    matrix: Optional[np.ndarray] = field(default=None, hash=False, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)
    
    def __hash__(self) -> int:
        return hash((self.name, self.gate_type, self.num_qubits, self.parameters))
    
    @property
    def has_parameters(self) -> bool:
        """Check if this gate has continuous parameters."""
        return len(self.parameters) > 0
    
    def with_parameters(self, **kwargs: float) -> "ParameterizedGate":
        """Create a parameterized instance of this gate."""
        return ParameterizedGate(spec=self, values=kwargs)
    
    def to_stim_name(self) -> str:
        """Get the Stim-compatible gate name."""
        return self.stim_name or self.name


@dataclass
class ParameterizedGate:
    """A gate with specific parameter values.
    
    Attributes
    ----------
    spec : GateSpec
        The gate specification.
    values : Dict[str, float]
        Parameter name to value mapping.
    """
    spec: GateSpec
    values: Dict[str, float] = field(default_factory=dict)
    
    @property
    def name(self) -> str:
        return self.spec.name
    
    @property
    def num_qubits(self) -> int:
        return self.spec.num_qubits
    
    def get_parameter(self, name: str) -> float:
        """Get a parameter value by name."""
        if name not in self.values:
            raise KeyError(f"Parameter {name!r} not set for gate {self.spec.name}")
        return self.values[name]


# Standard gate definitions (Stim-compatible)
STANDARD_GATES = {
    # Single-qubit Clifford gates
    "I": GateSpec("I", GateType.SINGLE_QUBIT, 1, is_clifford=True),
    "X": GateSpec("X", GateType.SINGLE_QUBIT, 1, is_clifford=True),
    "Y": GateSpec("Y", GateType.SINGLE_QUBIT, 1, is_clifford=True),
    "Z": GateSpec("Z", GateType.SINGLE_QUBIT, 1, is_clifford=True),
    "H": GateSpec("H", GateType.SINGLE_QUBIT, 1, is_clifford=True),
    "S": GateSpec("S", GateType.SINGLE_QUBIT, 1, is_clifford=True),
    "S_DAG": GateSpec("S_DAG", GateType.SINGLE_QUBIT, 1, is_clifford=True),
    "SQRT_X": GateSpec("SQRT_X", GateType.SINGLE_QUBIT, 1, is_clifford=True),
    "SQRT_X_DAG": GateSpec("SQRT_X_DAG", GateType.SINGLE_QUBIT, 1, is_clifford=True),
    "SQRT_Y": GateSpec("SQRT_Y", GateType.SINGLE_QUBIT, 1, is_clifford=True),
    "SQRT_Y_DAG": GateSpec("SQRT_Y_DAG", GateType.SINGLE_QUBIT, 1, is_clifford=True),
    
    # Two-qubit Clifford gates
    "CNOT": GateSpec("CNOT", GateType.TWO_QUBIT, 2, is_clifford=True, stim_name="CX"),
    "CX": GateSpec("CX", GateType.TWO_QUBIT, 2, is_clifford=True),
    "CZ": GateSpec("CZ", GateType.TWO_QUBIT, 2, is_clifford=True),
    "CY": GateSpec("CY", GateType.TWO_QUBIT, 2, is_clifford=True),
    "SWAP": GateSpec("SWAP", GateType.TWO_QUBIT, 2, is_clifford=True),
    "ISWAP": GateSpec("ISWAP", GateType.TWO_QUBIT, 2, is_clifford=True),
    "ISWAP_DAG": GateSpec("ISWAP_DAG", GateType.TWO_QUBIT, 2, is_clifford=True),
    "SQRT_XX": GateSpec("SQRT_XX", GateType.TWO_QUBIT, 2, is_clifford=True),
    "SQRT_YY": GateSpec("SQRT_YY", GateType.TWO_QUBIT, 2, is_clifford=True),
    "SQRT_ZZ": GateSpec("SQRT_ZZ", GateType.TWO_QUBIT, 2, is_clifford=True),
    
    # Non-Clifford gates (parameterized rotations)
    "RX": GateSpec("RX", GateType.SINGLE_QUBIT, 1, parameters=("theta",), is_clifford=False),
    "RY": GateSpec("RY", GateType.SINGLE_QUBIT, 1, parameters=("theta",), is_clifford=False),
    "RZ": GateSpec("RZ", GateType.SINGLE_QUBIT, 1, parameters=("theta",), is_clifford=False),
    "T": GateSpec("T", GateType.SINGLE_QUBIT, 1, is_clifford=False),
    "T_DAG": GateSpec("T_DAG", GateType.SINGLE_QUBIT, 1, is_clifford=False),
    
    # Two-qubit parameterized
    "XX": GateSpec("XX", GateType.TWO_QUBIT, 2, parameters=("theta",), is_clifford=False),
    "YY": GateSpec("YY", GateType.TWO_QUBIT, 2, parameters=("theta",), is_clifford=False),
    "ZZ": GateSpec("ZZ", GateType.TWO_QUBIT, 2, parameters=("theta",), is_clifford=False),
    
    # Measurement and reset
    "M": GateSpec("M", GateType.MEASUREMENT, 1),
    "MX": GateSpec("MX", GateType.MEASUREMENT, 1),
    "MY": GateSpec("MY", GateType.MEASUREMENT, 1),
    "MZ": GateSpec("MZ", GateType.MEASUREMENT, 1, stim_name="M"),
    "MR": GateSpec("MR", GateType.MEASUREMENT, 1),  # Measure and reset
    "R": GateSpec("R", GateType.RESET, 1),
    "RX": GateSpec("RX", GateType.RESET, 1),  # Reset to |+⟩
}




class NativeGateSet:
    """A set of native gates for a hardware platform.
    
    Defines which gates are natively available and provides
    methods for checking gate support and getting decompositions.
    
    Attributes
    ----------
    platform : str
        Platform name (e.g., "trapped_ion", "superconducting").
    gates : Dict[str, GateSpec]
        Native gate specifications by name.
    """
    
    def __init__(
        self,
        platform: str,
        gates: Optional[Dict[str, GateSpec]] = None,
    ):
        self.platform = platform
        self._gates: Dict[str, GateSpec] = {}
        
        # Add standard gates that most platforms support
        for name in ["I", "X", "Y", "Z", "M", "R"]:
            if name in STANDARD_GATES:
                self._gates[name] = STANDARD_GATES[name]
        
        # Add platform-specific gates
        if gates:
            self._gates.update(gates)
    
    def add_gate(self, spec: GateSpec) -> None:
        """Add a gate to the native set."""
        self._gates[spec.name] = spec
    
    def has_gate(self, name: str) -> bool:
        """Check if a gate is in the native set."""
        return name in self._gates
    
    def get_gate(self, name: str) -> Optional[GateSpec]:
        """Get a gate specification by name."""
        return self._gates.get(name)
    
    def single_qubit_gates(self) -> List[GateSpec]:
        """Get all native single-qubit gates."""
        return [g for g in self._gates.values() if g.gate_type == GateType.SINGLE_QUBIT]
    
    def two_qubit_gates(self) -> List[GateSpec]:
        """Get all native two-qubit gates."""
        return [g for g in self._gates.values() if g.gate_type == GateType.TWO_QUBIT]
    
    def clifford_gates(self) -> List[GateSpec]:
        """Get all native Clifford gates."""
        return [g for g in self._gates.values() if g.is_clifford]
    
    def __iter__(self):
        return iter(self._gates.values())
    
    def __len__(self) -> int:
        return len(self._gates)
    
    def __contains__(self, name: str) -> bool:
        return name in self._gates
    
    def __repr__(self) -> str:
        return f"NativeGateSet(platform={self.platform!r}, gates={list(self._gates.keys())})"


@dataclass
class GateDecomposition:
    """A decomposition of a gate into native gates.
    
    Attributes
    ----------
    original : GateSpec
        The gate being decomposed.
    sequence : List[Tuple[GateSpec, Tuple[int, ...]]]
        Sequence of (gate, qubits) pairs implementing the original gate.
    cost : float
        Cost metric (e.g., total gate time, number of 2-qubit gates).
    """
    original: GateSpec
    sequence: List[Tuple[Union[GateSpec, ParameterizedGate], Tuple[int, ...]]]
    cost: float = 0.0
    
    def __len__(self) -> int:
        return len(self.sequence)
    
    def two_qubit_count(self) -> int:
        """Count the number of two-qubit gates in the decomposition."""
        return sum(
            1 for gate, _ in self.sequence
            if (gate.num_qubits if isinstance(gate, GateSpec) else gate.spec.num_qubits) == 2
        )


class GateDecomposer(ABC):
    """Abstract base class for gate decomposition.
    
    Decomposes arbitrary gates into the native gate set of a platform.
    """
    
    def __init__(self, native_gates: NativeGateSet):
        self.native_gates = native_gates
    
    @abstractmethod
    def decompose(
        self,
        gate: Union[GateSpec, ParameterizedGate],
        qubits: Tuple[int, ...],
    ) -> GateDecomposition:
        """Decompose a gate into native gates.
        
        Parameters
        ----------
        gate : GateSpec or ParameterizedGate
            The gate to decompose.
        qubits : Tuple[int, ...]
            The qubits the gate acts on.
            
        Returns
        -------
        GateDecomposition
            The decomposed gate sequence.
        """
        ...
    
    def is_native(self, gate: Union[GateSpec, ParameterizedGate]) -> bool:
        """Check if a gate is native (no decomposition needed)."""
        name = gate.name if isinstance(gate, GateSpec) else gate.spec.name
        return self.native_gates.has_gate(name)
