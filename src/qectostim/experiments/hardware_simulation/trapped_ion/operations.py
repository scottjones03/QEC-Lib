"""
Trapped Ion QCCD Operations — simplified unified module.

This module provides a clean hierarchy for all QCCD operations:

**Base class**: QCCDOperation
  - Common interface: calculate_time(), calculate_fidelity(), run(), etc.
  - All physics constants from physics.py

**Operation categories**:
  - QubitGate: SingleQubitGate, MSGate, GateSwap, Measurement, Reset
  - Transport: Split, Merge, Move, JunctionCrossing  
  - Crystal: CrystalRotation, SympatheticCooling

**Gate definitions** (for decomposition):
  - MS_GATE, X_ROTATION, Y_ROTATION, Z_ROTATION, etc.
  - TrappedIonGateSet, TrappedIonGateDecomposer
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np

from qectostim.experiments.hardware_simulation.core.gates import (
    GateSpec,
    GateType,
    NativeGateSet,
    GateDecomposer,
    GateDecomposition,
    ParameterizedGate,
    STANDARD_GATES,
)
from .architecture import (
    Crossing,
    Ion,
    Junction,
    ManipulationTrap,
    QCCDNode,
    QCCDOperationType,
    QubitIon,
    CoolingIon,
    Trap,
    Operations,
)
from .physics import DEFAULT_CALIBRATION as _CAL, DEFAULT_FIDELITY_MODEL as _FIDELITY


# ============================================================================
# Result dataclass
# ============================================================================

@dataclass
class OperationResult:
    """Result of executing an operation."""
    success: bool = True
    time_us: float = 0.0
    fidelity: float = 1.0
    heating_added: float = 0.0
    error: Optional[str] = None


# ============================================================================
# Protocol for type hints
# ============================================================================

@runtime_checkable
class OperationLike(Protocol):
    """Duck-type interface for operations in routing/scheduling."""
    @property
    def involvedComponents(self) -> Sequence: ...
    def operationTime(self) -> float: ...
    def run(self) -> None: ...


# ============================================================================
# Base class — single inheritance, simple interface
# ============================================================================

class QCCDOperation(abc.ABC):
    """Base class for all trapped-ion QCCD operations.
    
    Subclasses must implement:
      - _execute(): perform the state mutation
      - label: human-readable description
      
    Optional overrides:
      - _time_us(): operation duration (default from TIME_US class var)
      - _heating_rate(): heating added (default from HEATING_RATE class var)
      - _fidelity(): operation fidelity (default 1.0)
      - _validate(): return error string if not applicable (default None)
    """
    
    # Subclasses set these class variables
    KEY: QCCDOperationType
    TIME_US: float = 0.0
    HEATING_RATE: float = 0.0
    
    def __init__(self, components: Sequence[QCCDNode] = ()) -> None:
        self._components: List[QCCDNode] = list(components)
        self._executed = False
    
    # --- Public interface (used by routing/scheduling) ---
    
    @property
    def involvedComponents(self) -> List[QCCDNode]:
        """Components involved in this operation (for scheduling)."""
        return self._components
    
    def operationTime(self) -> float:
        """Duration in microseconds."""
        return self._time_us()
    
    def fidelity(self) -> float:
        """Operation fidelity (0-1)."""
        return self._fidelity()
    
    def heatingRate(self) -> float:
        """Heating added in quanta."""
        return self._heating_rate()
    
    def run(self) -> OperationResult:
        """Execute the operation."""
        error = self._validate()
        if error:
            return OperationResult(success=False, error=error)
        
        time = self._time_us()
        fid = self._fidelity()
        heat = self._heating_rate()
        
        self._execute()
        self._executed = True
        
        return OperationResult(success=True, time_us=time, fidelity=fid, heating_added=heat)
    
    # --- Override in subclasses ---
    
    def _time_us(self) -> float:
        """Override for custom timing."""
        return self.TIME_US
    
    def _heating_rate(self) -> float:
        """Override for custom heating."""
        return self.HEATING_RATE
    
    def _fidelity(self) -> float:
        """Override for custom fidelity calculation."""
        return 1.0
    
    def _validate(self) -> Optional[str]:
        """Return error message if operation is invalid, None if OK."""
        return None
    
    @abc.abstractmethod
    def _execute(self) -> None:
        """Perform the state mutation. Must be implemented."""
        ...
    
    @property
    @abc.abstractmethod
    def label(self) -> str:
        """Human-readable label for this operation."""
        ...
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.label})"


# ============================================================================
# Qubit Gates
# ============================================================================

class QubitGate(QCCDOperation):
    """Base for qubit gate operations."""
    
    def __init__(self, ions: Sequence[QubitIon], trap: Optional[Trap] = None) -> None:
        super().__init__(components=list(ions) + ([trap] if trap else []))
        self._ions = list(ions)
        self._trap = trap
    
    @property
    def ions(self) -> List[QubitIon]:
        return self._ions
    
    @property
    def trap(self) -> Optional[Trap]:
        return self._trap
    
    def getTrapForIons(self) -> Optional[Trap]:
        """Return the trap if all ions are co-located in it, else None.
        
        Used by the routing layer to decide if an operation can execute
        without transport.
        """
        if not self._ions:
            return None
        parent = self._ions[0].parent
        if parent is None:
            return None
        for ion in self._ions[1:]:
            if ion.parent is not parent:
                return None
        return parent
    
    def setTrap(self, trap: Trap) -> None:
        """Set the execution trap (called by routing before run())."""
        self._trap = trap
        # Refresh component list so scheduling sees the trap
        self._components = list(self._ions) + [trap]
    
    def _validate(self) -> Optional[str]:
        if self._trap is None:
            return "Trap not set"
        for ion in self._ions:
            if ion.parent != self._trap:
                return f"Ion {ion.label} not in trap {self._trap.label}"
        return None
    
    def _execute(self) -> None:
        # Most gates don't mutate state
        pass
    
    def dephasingFidelity(self) -> float:
        """Dephasing fidelity (backward compat with old routing API)."""
        return 1.0
    
    def calculateDephasingFidelity(self) -> float:
        """Alias for dephasingFidelity (backward compat)."""
        return self.dephasingFidelity()


class SingleQubitGate(QubitGate):
    """Single-qubit rotation."""
    KEY = QCCDOperationType.ONE_QUBIT_GATE
    TIME_US = _CAL.single_qubit_gate_time * 1e6
    
    def __init__(self, ion: QubitIon, gate_type: str = "R",
                 theta: float = 0.0, phi: float = 0.0,
                 trap: Optional[Trap] = None) -> None:
        super().__init__(ions=[ion], trap=trap)
        self.gate_type = gate_type
        self.theta = theta
        self.phi = phi
    
    def _fidelity(self) -> float:
        if self._trap is None:
            return _FIDELITY.single_qubit_gate_fidelity(2, 0.0)
        return _FIDELITY.single_qubit_gate_fidelity(
            len(self._trap.ions), self._trap.motional_energy
        )
    
    @property
    def label(self) -> str:
        return f"{self.gate_type}({self._ions[0].label})"
    
    @classmethod
    def qubitOperation(cls, ion: QubitIon, gate_type: str = "R",
                       trap: Optional[Trap] = None) -> "SingleQubitGate":
        """Factory classmethod (backward compat with old routing API)."""
        return cls(ion, gate_type=gate_type, trap=trap)


class XRotation(SingleQubitGate):
    """X rotation — distinct type so the WISE scheduler batches it separately."""

    def __init__(self, ion: QubitIon, theta: float = 0.0,
                 trap: Optional[Trap] = None) -> None:
        super().__init__(ion, gate_type="RX", theta=theta, trap=trap)
    
    @classmethod
    def qubitOperation(cls, ion: QubitIon, trap: Optional[Trap] = None) -> "XRotation":
        """Factory classmethod (backward compat with old routing API)."""
        return cls(ion, trap=trap)


class YRotation(SingleQubitGate):
    """Y rotation — distinct type so the WISE scheduler batches it separately."""

    def __init__(self, ion: QubitIon, theta: float = 0.0,
                 trap: Optional[Trap] = None) -> None:
        super().__init__(ion, gate_type="RY", theta=theta, trap=trap)
    
    @classmethod
    def qubitOperation(cls, ion: QubitIon, trap: Optional[Trap] = None) -> "YRotation":
        """Factory classmethod (backward compat with old routing API)."""
        return cls(ion, trap=trap)


class MSGate(QubitGate):
    """Mølmer-Sørensen two-qubit entangling gate."""
    KEY = QCCDOperationType.TWO_QUBIT_MS_GATE
    TIME_US = _CAL.ms_gate_time * 1e6
    
    def __init__(self, ion1: QubitIon, ion2: QubitIon,
                 theta: float = np.pi / 4, phi: float = 0.0,
                 trap: Optional[Trap] = None) -> None:
        super().__init__(ions=[ion1, ion2], trap=trap)
        self.theta = theta
        self.phi = phi
    
    @classmethod
    def qubitOperation(cls, ion1: QubitIon, ion2: QubitIon,
                       trap: Optional[Trap] = None) -> "MSGate":
        """Factory classmethod (backward compat with old routing API)."""
        return cls(ion1, ion2, trap=trap)
    
    def _fidelity(self) -> float:
        if self._trap is None:
            return _FIDELITY.ms_gate_fidelity(2, 0.0)
        return _FIDELITY.ms_gate_fidelity(
            len(self._trap.ions), self._trap.motional_energy
        )
    
    @property
    def label(self) -> str:
        return f"MS({self._ions[0].label}, {self._ions[1].label})"


class GateSwap(QubitGate):
    """Swap two ions using MS gates."""
    KEY = QCCDOperationType.GATE_SWAP
    
    def __init__(self, ion1: QubitIon, ion2: QubitIon,
                 trap: Optional[Trap] = None) -> None:
        super().__init__(ions=[ion1, ion2], trap=trap)
    
    @classmethod
    def qubitOperation(cls, ion1: QubitIon, ion2: QubitIon,
                       trap: Optional[Trap] = None) -> "GateSwap":
        """Factory classmethod (backward compat with old routing API)."""
        return cls(ion1, ion2, trap=trap)
    
    def _time_us(self) -> float:
        return 3 * MSGate.TIME_US
    
    def _fidelity(self) -> float:
        ms_fid = MSGate(self._ions[0], self._ions[1], trap=self._trap)._fidelity()
        return ms_fid ** 3
    
    def _execute(self) -> None:
        if self._trap is None or self._ions[0] is self._ions[1]:
            return
        ion1, ion2 = self._ions
        idx1 = self._trap.ions.index(ion1)
        self._trap.remove_ion(ion1)
        self._trap.add_ion(ion1, position_idx=self._trap.ions.index(ion2) + 1)
        self._trap.remove_ion(ion2)
        self._trap.add_ion(ion2, position_idx=idx1)
    
    @property
    def label(self) -> str:
        return f"GateSwap({self._ions[0].label}, {self._ions[1].label})"


class Measurement(QubitGate):
    """Measure a qubit."""
    KEY = QCCDOperationType.MEASUREMENT
    TIME_US = _CAL.measurement_time * 1e6
    
    def __init__(self, ion: QubitIon, trap: Optional[Trap] = None) -> None:
        super().__init__(ions=[ion], trap=trap)
    
    def _fidelity(self) -> float:
        return 1.0 - _CAL.measurement_infidelity
    
    @classmethod
    def qubitOperation(cls, ion: QubitIon, trap: Optional[Trap] = None) -> "Measurement":
        """Factory classmethod (backward compat with old routing API)."""
        return cls(ion, trap=trap)
    
    @property
    def label(self) -> str:
        return f"Measure({self._ions[0].label})"


class QubitReset(QubitGate):
    """Reset qubit to |0⟩."""
    KEY = QCCDOperationType.QUBIT_RESET
    TIME_US = _CAL.reset_time * 1e6
    
    def __init__(self, ion: QubitIon, trap: Optional[Trap] = None) -> None:
        super().__init__(ions=[ion], trap=trap)
    
    def _fidelity(self) -> float:
        return 1.0 - _CAL.reset_infidelity
    
    @classmethod
    def qubitOperation(cls, ion: QubitIon, trap: Optional[Trap] = None) -> "QubitReset":
        """Factory classmethod (backward compat with old routing API)."""
        return cls(ion, trap=trap)
    
    @property
    def label(self) -> str:
        return f"Reset({self._ions[0].label})"


# ============================================================================
# Transport Operations
# ============================================================================

class Split(QCCDOperation):
    """Split an ion from trap into crossing. 80μs, 6 quanta/s."""
    KEY = QCCDOperationType.SPLIT
    TIME_US = _CAL.split_time * 1e6
    HEATING_RATE = _CAL.split_heating_rate
    # transport.py-compatible aliases (seconds)
    SPLITTING_TIME: float = _CAL.split_time
    
    def __init__(self, trap: Trap, crossing: Crossing, ion: Optional[Ion] = None) -> None:
        super().__init__(components=[trap, crossing])
        self._trap = trap
        self._crossing = crossing
        self._ion = ion
    
    def _validate(self) -> Optional[str]:
        if not self._crossing.connects(self._trap):
            return f"Crossing not connected to trap"
        if self._crossing.is_occupied:
            return f"Crossing already occupied"
        if self._trap.is_empty:
            return f"Trap empty"
        return None
    
    def _execute(self) -> None:
        ion = self._ion or self._trap.get_edge_ion(-1)
        self._trap.remove_ion(ion)
        self._crossing.set_ion(ion, self._trap)
        self._trap.distribute_heating(self.HEATING_RATE)
        ion.add_motional_energy(self.HEATING_RATE)
    
    @property
    def label(self) -> str:
        return f"Split({self._trap.label}→{self._crossing.label})"


class Merge(QCCDOperation):
    """Merge ion from crossing into trap. 80μs, 6 quanta/s."""
    KEY = QCCDOperationType.MERGE
    TIME_US = _CAL.merge_time * 1e6
    HEATING_RATE = _CAL.merge_heating_rate
    # transport.py-compatible aliases (seconds)
    MERGING_TIME: float = _CAL.merge_time
    
    def __init__(self, trap: Trap, crossing: Crossing) -> None:
        super().__init__(components=[trap, crossing])
        self._trap = trap
        self._crossing = crossing
    
    def _validate(self) -> Optional[str]:
        if not self._crossing.connects(self._trap):
            return f"Crossing not connected to trap"
        if not self._crossing.is_occupied:
            return f"Crossing has no ion"
        if self._trap.is_full:
            return f"Trap full"
        return None
    
    def _execute(self) -> None:
        ion = self._crossing.remove_ion()
        self._trap.add_ion(ion)
        self._trap.distribute_heating(self.HEATING_RATE)
    
    @property
    def label(self) -> str:
        return f"Merge({self._crossing.label}→{self._trap.label})"


class Move(QCCDOperation):
    """Move ion through crossing. 5μs, 0.1 quanta/s."""
    KEY = QCCDOperationType.MOVE
    TIME_US = _CAL.shuttle_time * 1e6
    HEATING_RATE = _CAL.shuttle_heating_rate
    # transport.py-compatible aliases (seconds)
    MOVING_TIME: float = _CAL.shuttle_time
    
    def __init__(self, crossing: Crossing) -> None:
        super().__init__(components=[crossing])
        self._crossing = crossing
    
    def _validate(self) -> Optional[str]:
        if not self._crossing.is_occupied:
            return f"Crossing has no ion"
        return None
    
    def _execute(self) -> None:
        self._crossing.move_ion()
        if self._crossing.ion:
            self._crossing.ion.add_motional_energy(self.HEATING_RATE)
    
    @property
    def label(self) -> str:
        return f"Move({self._crossing.label})"


class JunctionCrossing(QCCDOperation):
    """Cross junction. 50μs, 3 quanta/s."""
    KEY = QCCDOperationType.JUNCTION_CROSSING
    TIME_US = _CAL.junction_time * 1e6
    HEATING_RATE = _CAL.junction_heating_rate
    # transport.py-compatible aliases (seconds)
    CROSSING_TIME: float = _CAL.junction_time
    CROSSING_HEATING: float = _CAL.junction_heating_rate
    
    def __init__(self, junction: Junction, crossing: Crossing) -> None:
        super().__init__(components=[junction, crossing])
        self._junction = junction
        self._crossing = crossing
    
    def _validate(self) -> Optional[str]:
        if not self._crossing.connects(self._junction):
            return f"Crossing not connected to junction"
        has_crossing_ion = self._crossing.is_occupied
        has_junction_ion = not self._junction.is_empty
        if not (has_crossing_ion or has_junction_ion):
            return "No ion to move"
        if has_crossing_ion and self._junction.is_full:
            return f"Junction full"
        return None
    
    def _execute(self) -> None:
        if self._crossing.is_occupied:
            ion = self._crossing.remove_ion()
            self._junction.add_ion(ion)
        else:
            ion = self._junction.remove_ion()
            self._crossing.set_ion(ion, self._junction)
        ion.add_motional_energy(self.HEATING_RATE)
    
    @property
    def label(self) -> str:
        return f"JunctionCrossing({self._junction.label}↔{self._crossing.label})"


# ============================================================================
# Crystal Operations
# ============================================================================

class CrystalRotation(QCCDOperation):
    """Rotate ion order in trap. 42μs, 0.3 quanta/s."""
    KEY = QCCDOperationType.CRYSTAL_ROTATION
    TIME_US = _CAL.rotation_time * 1e6
    HEATING_RATE = _CAL.rotation_heating_rate
    # transport.py-compatible aliases (seconds)
    ROTATION_TIME: float = _CAL.rotation_time
    ROTATION_HEATING: float = _CAL.rotation_heating_rate
    
    def __init__(self, trap: Trap) -> None:
        super().__init__(components=[trap])
        self._trap = trap
    
    @property
    def ions(self) -> List[Ion]:
        return self._trap.ions
    
    def _validate(self) -> Optional[str]:
        if len(self._trap.ions) < 2:
            return "Need ≥2 ions"
        return None
    
    def _execute(self) -> None:
        ions = list(self._trap.ions)
        for ion in ions:
            self._trap.remove_ion(ion)
        for ion in reversed(ions):
            self._trap.add_ion(ion)
        self._trap.distribute_heating(self.HEATING_RATE)
    
    @property
    def label(self) -> str:
        return f"CrystalRotation({self._trap.label})"


class SympatheticCooling(QCCDOperation):
    """Cool ions via sympathetic cooling. 400μs."""
    KEY = QCCDOperationType.RECOOLING
    TIME_US = _CAL.recool_time * 1e6
    HEATING_RATE = _CAL.cooling_heating_rate
    
    def __init__(self, trap: Trap) -> None:
        super().__init__(components=[trap])
        self._trap = trap
    
    @property
    def ions(self) -> List[Ion]:
        return self._trap.ions
    
    def _validate(self) -> Optional[str]:
        if not self._trap.has_cooling_ion:
            return "No cooling ion"
        return None
    
    def _execute(self) -> None:
        self._trap.cool()
        self._trap.distribute_heating(self.HEATING_RATE)
    
    @property
    def label(self) -> str:
        return f"SympatheticCooling({self._trap.label})"


# ============================================================================
# Parallel Operation Wrapper
# ============================================================================

class ParallelOperation:
    """Wrap multiple operations for parallel execution."""
    
    KEY = QCCDOperationType.PARALLEL
    
    def __init__(self, operations: Sequence[QCCDOperation]) -> None:
        self._operations = list(operations)
        self._components: List[QCCDNode] = []
        for op in self._operations:
            self._components.extend(op.involvedComponents)
    
    @property
    def involvedComponents(self) -> List[QCCDNode]:
        return self._components
    
    @property
    def operations(self) -> List[QCCDOperation]:
        return self._operations
    
    def operationTime(self) -> float:
        if not self._operations:
            return 0.0
        return max(op.operationTime() for op in self._operations)
    
    def fidelity(self) -> float:
        return float(np.prod([op.fidelity() for op in self._operations]))
    
    def run(self) -> None:
        for op in np.random.permutation(self._operations):
            op.run()
    
    @classmethod
    def physicalOperation(cls, operationsToStart, operationsStarted=None):
        """Factory for backward compatibility with old routing code."""
        ops = list(operationsStarted or []) + list(operationsToStart)
        return cls(ops)
    
    @property
    def label(self) -> str:
        return f"Parallel({len(self._operations)} ops)"


# ============================================================================
# Batch Reconfiguration
# ============================================================================

@dataclass
class ReconfigurationStep:
    """Single step in batch reconfiguration."""
    operations: List[QCCDOperation] = field(default_factory=list)
    time_us: float = 0.0


@dataclass  
class GlobalReconfiguration:
    """Batch reconfiguration of ions."""
    steps: List[ReconfigurationStep] = field(default_factory=list)
    total_time_us: float = 0.0
    total_fidelity: float = 1.0


# ============================================================================
# Transport Aggregate Constants
# ============================================================================

ROW_SWAP_TIME_S = (
    Split.TIME_US + Move.TIME_US + CrystalRotation.TIME_US + Merge.TIME_US + Move.TIME_US
) / 1e6

ROW_SWAP_HEATING = (
    Split.TIME_US/1e6 * Split.HEATING_RATE
    + Move.TIME_US/1e6 * Move.HEATING_RATE
    + CrystalRotation.TIME_US/1e6 * CrystalRotation.HEATING_RATE
    + Merge.TIME_US/1e6 * Merge.HEATING_RATE
    + Move.TIME_US/1e6 * Move.HEATING_RATE
)

COL_SWAP_TIME_S = (6 * JunctionCrossing.TIME_US + Move.TIME_US) / 1e6

COL_SWAP_HEATING = (
    6 * JunctionCrossing.TIME_US/1e6 * JunctionCrossing.HEATING_RATE
    + Move.TIME_US/1e6 * Move.HEATING_RATE
)


# ============================================================================
# Gate Specifications (for decomposition)
# ============================================================================

MS_GATE = GateSpec(
    name="MS", gate_type=GateType.TWO_QUBIT, num_qubits=2,
    parameters=("theta", "phi"), is_native=True,
    metadata={"platform": "trapped_ion", "duration_us": _CAL.ms_gate_time * 1e6},
)

XX_GATE = GateSpec(
    name="XX", gate_type=GateType.TWO_QUBIT, num_qubits=2,
    parameters=("theta",), is_native=True,
    metadata={"platform": "trapped_ion"},
)

GLOBAL_ROTATION = GateSpec(
    name="GR", gate_type=GateType.MULTI_QUBIT, num_qubits=-1,
    parameters=("theta", "phi"), is_native=True,
    metadata={"platform": "trapped_ion"},
)

SINGLE_ION_ROTATION = GateSpec(
    name="R", gate_type=GateType.SINGLE_QUBIT, num_qubits=1,
    parameters=("theta", "phi"), is_native=True,
    metadata={"platform": "trapped_ion"},
)

X_ROTATION = GateSpec(
    name="RX", gate_type=GateType.SINGLE_QUBIT, num_qubits=1,
    parameters=("theta",), is_native=True,
    metadata={"platform": "trapped_ion"},
)

Y_ROTATION = GateSpec(
    name="RY", gate_type=GateType.SINGLE_QUBIT, num_qubits=1,
    parameters=("theta",), is_native=True,
    metadata={"platform": "trapped_ion"},
)

Z_ROTATION = GateSpec(
    name="RZ", gate_type=GateType.SINGLE_QUBIT, num_qubits=1,
    parameters=("theta",), is_native=True,
    metadata={"platform": "trapped_ion", "duration_us": 0.0},
)

ION_MEASUREMENT = GateSpec(
    name="M_ION", gate_type=GateType.MEASUREMENT, num_qubits=1,
    is_native=True,
    metadata={"platform": "trapped_ion"},
)

TRAPPED_ION_NATIVE_GATES: Dict[str, GateSpec] = {
    "MS": MS_GATE, "XX": XX_GATE, "GR": GLOBAL_ROTATION,
    "R": SINGLE_ION_ROTATION, "RX": X_ROTATION, "RY": Y_ROTATION,
    "RZ": Z_ROTATION, "M_ION": ION_MEASUREMENT,
}


# ============================================================================
# Gate Set and Decomposer
# ============================================================================

class TrappedIonGateSet(NativeGateSet):
    """Native gate set for trapped ion hardware."""
    
    def __init__(self):
        super().__init__(platform="trapped_ion")
        for spec in TRAPPED_ION_NATIVE_GATES.values():
            self.add_gate(spec)
        for name in ["H", "S", "S_DAG", "X", "Y", "Z"]:
            if name in STANDARD_GATES:
                self.add_gate(STANDARD_GATES[name])
    
    def entangling_gate(self) -> GateSpec:
        return self.get_gate("MS")
    
    def has_global_operations(self) -> bool:
        return True


class TrappedIonGateDecomposer(GateDecomposer):
    """Decompose standard gates to trapped ion natives."""
    
    def __init__(self):
        super().__init__(TrappedIonGateSet())
        self._table = self._build_table()
    
    def _build_table(self):
        pi = math.pi
        
        def h(q):
            return GateDecomposition(
                original=STANDARD_GATES["H"],
                sequence=[
                    (Y_ROTATION.with_parameters(theta=pi/2), (q[0],)),
                    (Z_ROTATION.with_parameters(theta=pi), (q[0],)),
                ],
                cost=_CAL.single_qubit_gate_time * 1e6,
            )
        
        def cnot(q):
            c, t = q
            ms = MS_GATE.with_parameters(theta=pi/4, phi=0.0)
            return GateDecomposition(
                original=STANDARD_GATES.get("CNOT", STANDARD_GATES.get("CX")),
                sequence=[
                    (Y_ROTATION.with_parameters(theta=-pi/2), (t,)),
                    (ms, (c, t)),
                    (X_ROTATION.with_parameters(theta=-pi/2), (c,)),
                    (X_ROTATION.with_parameters(theta=-pi/2), (t,)),
                    (Y_ROTATION.with_parameters(theta=pi/2), (t,)),
                ],
                cost=_CAL.ms_gate_time * 1e6 + 3 * _CAL.single_qubit_gate_time * 1e6,
            )
        
        return {"H": h, "CNOT": cnot, "CX": cnot}
    
    def decompose(self, gate, qubits: Tuple[int, ...]) -> GateDecomposition:
        name = gate.name if isinstance(gate, GateSpec) else gate.spec.name
        if self.is_native(gate):
            return GateDecomposition(
                original=gate if isinstance(gate, GateSpec) else gate.spec,
                sequence=[(gate, qubits)], cost=0.0,
            )
        if name in self._table:
            return self._table[name](qubits)
        raise NotImplementedError(f"No decomposition for {name}")


@dataclass
class TrappedIonGateTiming:
    """Gate timing parameters."""
    ms_gate_time: float = _CAL.ms_gate_time
    single_qubit_time: float = _CAL.single_qubit_gate_time
    measurement_time: float = _CAL.measurement_time
    reset_time: float = _CAL.reset_time


# ============================================================================
# Factory Functions
# ============================================================================

def create_ms_gate(ion1: QubitIon, ion2: QubitIon, trap: Trap, theta: float = np.pi/4) -> MSGate:
    return MSGate(ion1, ion2, theta=theta, trap=trap)

def create_single_qubit_gate(ion: QubitIon, trap: Trap, gate_type: str = "R",
                              theta: float = 0.0, phi: float = 0.0) -> SingleQubitGate:
    return SingleQubitGate(ion, gate_type, theta, phi, trap)

def create_measurement(ion: QubitIon, trap: Trap) -> Measurement:
    return Measurement(ion, trap)

def create_split(trap: Trap, crossing: Crossing, ion: Optional[Ion] = None) -> Split:
    return Split(trap, crossing, ion)

def create_merge(trap: Trap, crossing: Crossing) -> Merge:
    return Merge(trap, crossing)


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# Protocol alias
Operation = OperationLike

# Base class aliases (for external code)
QCCDOperationBase = QCCDOperation
QubitGateBase = QubitGate
TransportOperation = QCCDOperation
CrystalOperationBase = QCCDOperation
CrystalOperation = QCCDOperation

# Legacy routing class aliases
QubitOperation = QubitGate          # Old base class name
OneQubitGate = SingleQubitGate      # Old single-qubit gate name
TwoQubitMSGate = MSGate             # Old MS gate name
# XRotation / YRotation are now real subclasses defined above
MeasurementGate = Measurement       # Alternative measurement name
QubitResetGate = QubitReset         # Alternative reset name
GateSwapGate = GateSwap             # Alternative swap name

# Constants
BACKGROUND_HEATING_RATE = _CAL.heating_rate
FIDELITY_SCALING_A = _CAL.fidelity_scaling_A
T2 = _CAL.t2_time


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Result
    "OperationResult",
    # Protocol
    "Operation",
    "OperationLike",
    # Base classes
    "QCCDOperation",
    "QCCDOperationBase",  # alias
    "QubitGate",
    "QubitGateBase",  # alias
    # Qubit gates
    "SingleQubitGate",
    "XRotation",
    "YRotation",
    "MSGate",
    "GateSwap",
    "Measurement",
    "QubitReset",
    # Transport
    "Split",
    "Merge",
    "Move",
    "JunctionCrossing",
    # Crystal
    "CrystalRotation",
    "SympatheticCooling",
    # Parallel
    "ParallelOperation",
    # Batch
    "ReconfigurationStep",
    "GlobalReconfiguration",
    # Constants
    "ROW_SWAP_TIME_S",
    "ROW_SWAP_HEATING",
    "COL_SWAP_TIME_S",
    "COL_SWAP_HEATING",
    # Gate specs
    "MS_GATE",
    "XX_GATE",
    "GLOBAL_ROTATION",
    "SINGLE_ION_ROTATION",
    "X_ROTATION",
    "Y_ROTATION",
    "Z_ROTATION",
    "ION_MEASUREMENT",
    "TRAPPED_ION_NATIVE_GATES",
    # Gate classes
    "TrappedIonGateSet",
    "TrappedIonGateDecomposer",
    "TrappedIonGateTiming",
    # Factories
    "create_ms_gate",
    "create_single_qubit_gate",
    "create_measurement",
    "create_split",
    "create_merge",
    # Base class aliases (for external code)
    "TransportOperation",
    "CrystalOperationBase",
    "CrystalOperation",
    # Legacy routing class aliases
    "QubitOperation",
    "OneQubitGate",
    "TwoQubitMSGate",
    "XRotation",
    "YRotation",
    "MeasurementGate",
    "QubitResetGate",
    "GateSwapGate",
    # Constants
    "BACKGROUND_HEATING_RATE",
    "FIDELITY_SCALING_A",
    "T2",
]
