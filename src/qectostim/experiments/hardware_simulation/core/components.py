# src/qectostim/experiments/hardware_simulation/core/components.py
"""
Hardware component abstractions.

Defines physical elements that make up quantum hardware:
qubits, couplers, and other platform-specific components.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Any,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.architecture import Zone


class QubitState(Enum):
    """Possible states of a physical qubit."""
    IDLE = auto()           # Available for operations
    IN_GATE = auto()        # Currently executing a gate
    IN_TRANSPORT = auto()   # Being moved (for mobile architectures)
    IN_MEASUREMENT = auto() # Being measured
    MEASURED = auto()       # Has been measured (classical state)
    LOST = auto()           # Qubit lost (neutral atoms)
    ERROR = auto()          # In error state


class HardwareComponent(ABC):
    """Abstract base class for hardware components.
    
    Base class for all physical elements in the quantum hardware.
    Inspired by QCCDComponent from the trapped ion implementation.
    
    Attributes
    ----------
    id : str
        Unique identifier for the component.
    component_type : str
        Type of component (for serialization/display).
    metadata : Dict[str, Any]
        Platform-specific additional properties.
    """
    
    def __init__(
        self,
        id: str,
        component_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.component_type = component_type
        self.metadata = metadata or {}
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the component to initial state."""
        ...
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HardwareComponent):
            return NotImplemented
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class PhysicalQubit(HardwareComponent):
    """A physical qubit in the hardware.
    
    Represents the physical realization of a qubit with properties
    like position, state, and accumulated errors.
    
    Attributes
    ----------
    id : str
        Unique identifier.
    index : int
        Integer index for Stim circuit generation.
    position : Optional[Tuple[float, ...]]
        Physical coordinates.
    zone_id : Optional[str]
        ID of the zone containing this qubit.
    state : QubitState
        Current operational state.
    logical_index : Optional[int]
        Mapped logical qubit index (after compilation).
    idle_time : float
        Accumulated idle time in microseconds.
    error_accumulated : float
        Accumulated error probability from operations.
    t1_remaining : Optional[float]
        Remaining T1 coherence (for tracking).
    t2_remaining : Optional[float]
        Remaining T2 coherence (for tracking).
    metadata : Dict[str, Any]
        Platform-specific properties.
    """
    index: int = 0
    position: Optional[Tuple[float, ...]] = None
    zone_id: Optional[str] = None
    state: QubitState = QubitState.IDLE
    logical_index: Optional[int] = None
    idle_time: float = 0.0
    error_accumulated: float = 0.0
    t1_remaining: Optional[float] = None
    t2_remaining: Optional[float] = None
    
    def __init__(
        self,
        id: str,
        index: int,
        position: Optional[Tuple[float, ...]] = None,
        zone_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(id, "qubit", metadata)
        self.index = index
        self.position = position
        self.zone_id = zone_id
        self.state = QubitState.IDLE
        self.logical_index = None
        self.idle_time = 0.0
        self.error_accumulated = 0.0
        self.t1_remaining = None
        self.t2_remaining = None
    
    def reset(self) -> None:
        """Reset qubit to initial idle state."""
        self.state = QubitState.IDLE
        self.idle_time = 0.0
        self.error_accumulated = 0.0
    
    def add_idle_time(self, duration: float) -> None:
        """Add idle time and potentially update coherence tracking."""
        self.idle_time += duration
    
    def add_error(self, probability: float) -> None:
        """Accumulate error from an operation."""
        # Simple model: errors compound (not quite physical but useful)
        self.error_accumulated = 1 - (1 - self.error_accumulated) * (1 - probability)
    
    def move_to_zone(self, zone_id: str, position: Optional[Tuple[float, ...]] = None) -> None:
        """Move this qubit to a new zone."""
        self.zone_id = zone_id
        if position is not None:
            self.position = position
    
    @property
    def is_available(self) -> bool:
        """Check if qubit is available for operations."""
        return self.state == QubitState.IDLE
    
    def __hash__(self) -> int:
        return hash(self.id)


@dataclass  
class Coupler(HardwareComponent):
    """A coupler connecting two qubits.
    
    Represents the physical mechanism enabling two-qubit gates.
    Platform-specific (e.g., tunable coupler in superconducting,
    shared motional mode in trapped ions).
    
    Attributes
    ----------
    id : str
        Unique identifier.
    qubit1_id : str
        First connected qubit ID.
    qubit2_id : str
        Second connected qubit ID.
    coupling_strength : float
        Coupling strength (platform-specific units).
    is_active : bool
        Whether the coupler is currently active.
    gate_fidelity : float
        Typical two-qubit gate fidelity through this coupler.
    crosstalk_qubits : Set[str]
        IDs of qubits affected by crosstalk when this coupler is active.
    """
    qubit1_id: str = ""
    qubit2_id: str = ""
    coupling_strength: float = 1.0
    is_active: bool = False
    gate_fidelity: float = 0.99
    crosstalk_qubits: Set[str] = field(default_factory=set)
    
    def __init__(
        self,
        id: str,
        qubit1_id: str,
        qubit2_id: str,
        coupling_strength: float = 1.0,
        gate_fidelity: float = 0.99,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(id, "coupler", metadata)
        self.qubit1_id = qubit1_id
        self.qubit2_id = qubit2_id
        self.coupling_strength = coupling_strength
        self.is_active = False
        self.gate_fidelity = gate_fidelity
        self.crosstalk_qubits = set()
    
    def reset(self) -> None:
        """Reset coupler to inactive state."""
        self.is_active = False
    
    def activate(self) -> None:
        """Activate the coupler for a two-qubit gate."""
        self.is_active = True
    
    def deactivate(self) -> None:
        """Deactivate the coupler."""
        self.is_active = False
    
    def connects(self, qubit_id: str) -> bool:
        """Check if this coupler connects to a given qubit."""
        return qubit_id in {self.qubit1_id, self.qubit2_id}
    
    def other_qubit(self, qubit_id: str) -> Optional[str]:
        """Get the other qubit connected by this coupler."""
        if qubit_id == self.qubit1_id:
            return self.qubit2_id
        elif qubit_id == self.qubit2_id:
            return self.qubit1_id
        return None
    
    def __hash__(self) -> int:
        return hash(self.id)


class QubitRegister:
    """Collection of physical qubits.
    
    Manages a set of physical qubits and provides lookup methods.
    
    Attributes
    ----------
    qubits : Dict[str, PhysicalQubit]
        Qubits by ID.
    by_index : Dict[int, PhysicalQubit]
        Qubits by integer index.
    by_zone : Dict[str, List[PhysicalQubit]]
        Qubits grouped by zone.
    """
    
    def __init__(self):
        self.qubits: Dict[str, PhysicalQubit] = {}
        self.by_index: Dict[int, PhysicalQubit] = {}
        self.by_zone: Dict[str, List[PhysicalQubit]] = {}
    
    def add(self, qubit: PhysicalQubit) -> None:
        """Add a qubit to the register."""
        self.qubits[qubit.id] = qubit
        self.by_index[qubit.index] = qubit
        if qubit.zone_id:
            if qubit.zone_id not in self.by_zone:
                self.by_zone[qubit.zone_id] = []
            self.by_zone[qubit.zone_id].append(qubit)
    
    def get(self, id_or_index: str | int) -> Optional[PhysicalQubit]:
        """Get a qubit by ID or index."""
        if isinstance(id_or_index, int):
            return self.by_index.get(id_or_index)
        return self.qubits.get(id_or_index)
    
    def in_zone(self, zone_id: str) -> List[PhysicalQubit]:
        """Get all qubits in a zone."""
        return self.by_zone.get(zone_id, [])
    
    def available(self) -> List[PhysicalQubit]:
        """Get all available (idle) qubits."""
        return [q for q in self.qubits.values() if q.is_available]
    
    def reset_all(self) -> None:
        """Reset all qubits."""
        for qubit in self.qubits.values():
            qubit.reset()
    
    def __len__(self) -> int:
        return len(self.qubits)
    
    def __iter__(self):
        return iter(self.qubits.values())
    
    def __getitem__(self, key: str | int) -> PhysicalQubit:
        qubit = self.get(key)
        if qubit is None:
            raise KeyError(f"Qubit {key!r} not found")
        return qubit


# Platform-specific component bases (to be extended)

class TrappedIonComponent(HardwareComponent):
    """Base class for trapped ion components.
    
    Extended by Ion, Trap, Junction, etc. in trapped_ion implementation.
    """
    pass


class SuperconductingComponent(HardwareComponent):
    """Base class for superconducting components.
    
    Extended by TransmonQubit, TunableCoupler, etc.
    """
    pass


class NeutralAtomComponent(HardwareComponent):
    """Base class for neutral atom components.
    
    Extended by Atom, TweezerSite, etc.
    """
    pass
