"""
Trapped Ion QCCD Hardware Components.

This module defines the physical components of a trapped ion QCCD system:
- Ions (qubits, cooling, spectator)
- Traps (manipulation, storage)
- Junctions (routing nodes)
- Crossings (connections between nodes)

These components are used to model the physical layout and state of
a QCCD architecture for simulation and compilation.

Ported and refactored from the original qccd_nodes.py implementation.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np


# =============================================================================
# Operation Types
# =============================================================================

class QCCDOperation(enum.Enum):
    """Operations available in a trapped ion QCCD system.
    
    These operations represent the primitive actions that can be performed
    on ions in a QCCD architecture.
    """
    # Transport operations
    SPLIT = enum.auto()           # Split ion chain
    MERGE = enum.auto()           # Merge ion chains
    MOVE = enum.auto()            # Move ion chain
    JUNCTION_CROSSING = enum.auto()  # Cross a junction
    
    # Crystal manipulation
    CRYSTAL_ROTATION = enum.auto()  # Rotate ion crystal
    GATE_SWAP = enum.auto()         # Swap ion positions via gates
    
    # Quantum operations
    ONE_QUBIT_GATE = enum.auto()    # Single-qubit rotation
    TWO_QUBIT_MS_GATE = enum.auto() # Mølmer-Sørensen gate
    MEASUREMENT = enum.auto()       # State readout
    QUBIT_RESET = enum.auto()       # Reset to |0⟩
    
    # Auxiliary operations
    RECOOLING = enum.auto()         # Sympathetic cooling
    GLOBAL_RECONFIG = enum.auto()   # Batch reconfiguration (WISE)
    PARALLEL = enum.auto()          # Parallel operation marker


# =============================================================================
# Physical Constants
# =============================================================================

@dataclass(frozen=True)
class TrappedIonTimings:
    """Timing constants for trapped ion operations (microseconds).
    
    These are typical values from the literature. Actual values
    depend on the specific ion species and trap design.
    """
    # Transport timings
    splitting_time: float = 80.0      # Ion chain split
    merging_time: float = 80.0        # Ion chain merge  
    junction_time: float = 100.0      # Junction crossing
    linear_move_time: float = 10.0    # Linear transport per segment
    
    # Gate timings
    one_qubit_gate: float = 5.0       # Single-qubit rotation
    ms_gate: float = 40.0             # Mølmer-Sørensen gate
    measurement: float = 400.0        # State readout
    reset: float = 50.0               # Qubit reset
    
    # Auxiliary
    recooling_time: float = 200.0     # Sympathetic cooling


@dataclass(frozen=True)
class HeatingRates:
    """Heating rates for different operations (quanta/operation).
    
    Motional heating degrades gate fidelity and must be managed
    through sympathetic cooling.
    """
    split: float = 0.1
    merge: float = 0.1
    junction_crossing: float = 0.5
    linear_move: float = 0.05
    background_rate: float = 4.0  # quanta/second


# Default timing and heating rate instances
DEFAULT_TIMINGS = TrappedIonTimings()
DEFAULT_HEATING_RATES = HeatingRates()


# =============================================================================
# Abstract Base Classes
# =============================================================================

class QCCDComponent(abc.ABC):
    """Abstract base class for all QCCD components.
    
    All components have:
    - A unique identifier (idx)
    - A physical position
    - A set of allowed operations
    """
    
    @property
    @abc.abstractmethod
    def idx(self) -> int:
        """Unique identifier for this component."""
        ...
    
    @property
    @abc.abstractmethod
    def position(self) -> Tuple[float, float]:
        """Physical (x, y) position in the trap layout."""
        ...
    
    @property
    @abc.abstractmethod
    def allowed_operations(self) -> Set[QCCDOperation]:
        """Operations that can be performed with/on this component."""
        ...


# =============================================================================
# Ion Classes
# =============================================================================

class Ion(QCCDComponent):
    """Base class for all ion types.
    
    An ion is the fundamental qubit carrier in a trapped ion system.
    Ions can be moved between traps and used for quantum operations.
    
    Attributes
    ----------
    idx : int
        Unique ion identifier.
    position : Tuple[float, float]
        Current (x, y) position.
    parent : Optional[QCCDNode]
        The trap or crossing currently holding this ion.
    motional_energy : float
        Current motional excitation (quanta above ground).
    label : str
        Display label prefix (e.g., "Q" for qubit).
    """
    
    # Base operations available to all ions
    _BASE_OPERATIONS: Set[QCCDOperation] = {
        QCCDOperation.CRYSTAL_ROTATION,
        QCCDOperation.GATE_SWAP,
        QCCDOperation.JUNCTION_CROSSING,
        QCCDOperation.MERGE,
        QCCDOperation.SPLIT,
        QCCDOperation.MOVE,
        QCCDOperation.RECOOLING,
    }
    
    def __init__(
        self,
        idx: int = 0,
        label: str = "I",
        color: str = "lightblue",
    ) -> None:
        """Initialize an ion.
        
        Parameters
        ----------
        idx : int
            Unique ion identifier.
        label : str
            Display label prefix.
        color : str
            Visualization color.
        """
        self._idx = idx
        self._position_x: float = 0.0
        self._position_y: float = 0.0
        self._parent: Optional[Union[QCCDNode, Crossing]] = None
        self._label = label
        self._color = color
        self._motional_energy: float = 0.0
    
    @property
    def idx(self) -> int:
        return self._idx
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self._position_x, self._position_y)
    
    @property
    def allowed_operations(self) -> Set[QCCDOperation]:
        return self._BASE_OPERATIONS.copy()
    
    @property
    def parent(self) -> Optional[Union[QCCDNode, Crossing]]:
        """Get the component currently holding this ion."""
        return self._parent
    
    @property
    def motional_energy(self) -> float:
        """Current motional excitation in quanta."""
        return self._motional_energy
    
    @property
    def label(self) -> str:
        """Display label including index."""
        return f"{self._label}{self._idx}"
    
    @property
    def color(self) -> str:
        """Visualization color."""
        return self._color
    
    def set_state(
        self,
        idx: Optional[int] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        parent: Optional[Union[QCCDNode, Crossing]] = None,
    ) -> None:
        """Update ion state.
        
        Parameters
        ----------
        idx : Optional[int]
            New index (if changing).
        x, y : Optional[float]
            New position coordinates.
        parent : Optional[QCCDNode | Crossing]
            New parent component.
        """
        if idx is not None:
            self._idx = idx
        if x is not None:
            self._position_x = x
        if y is not None:
            self._position_y = y
        if parent is not None:
            self._parent = parent
    
    def add_motional_energy(self, quanta: float) -> None:
        """Add motional energy (heating)."""
        self._motional_energy += quanta
    
    def cool(self) -> float:
        """Cool ion to ground state, return energy removed."""
        energy = self._motional_energy
        self._motional_energy = 0.0
        return energy
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.label}, pos={self.position})"


class QubitIon(Ion):
    """Ion used as a qubit for quantum computation.
    
    Qubit ions support quantum operations in addition to transport.
    """
    
    _QUBIT_OPERATIONS: Set[QCCDOperation] = {
        QCCDOperation.MEASUREMENT,
        QCCDOperation.ONE_QUBIT_GATE,
        QCCDOperation.TWO_QUBIT_MS_GATE,
        QCCDOperation.QUBIT_RESET,
    }
    
    def __init__(
        self,
        idx: int = 0,
        label: str = "Q",
        color: str = "blue",
    ) -> None:
        super().__init__(idx, label, color)
    
    @property
    def allowed_operations(self) -> Set[QCCDOperation]:
        return super().allowed_operations | self._QUBIT_OPERATIONS


class CoolingIon(Ion):
    """Ion used for sympathetic cooling.
    
    Cooling ions absorb motional energy from qubit ions via
    shared motional modes, then dissipate it through laser cooling.
    They do not participate in quantum operations.
    """
    
    def __init__(
        self,
        idx: int = 0,
        label: str = "C",
        color: str = "cyan",
    ) -> None:
        super().__init__(idx, label, color)


class SpectatorIon(QubitIon):
    """Ion used as an ancilla or spectator.
    
    Spectator ions are qubit ions not directly involved in the
    main computation, often used for error detection.
    """
    
    def __init__(
        self,
        idx: int = 0,
        label: str = "S",
        color: str = "green",
    ) -> None:
        super().__init__(idx, label, color)


# =============================================================================
# Node Classes (Traps and Junctions)
# =============================================================================

class QCCDNode(QCCDComponent):
    """Base class for trap segments and junctions.
    
    A node is a location in the QCCD architecture that can hold ions.
    Nodes are connected by crossings to form the trap network.
    
    Attributes
    ----------
    idx : int
        Unique node identifier.
    position : Tuple[float, float]
        Node center position.
    ions : List[Ion]
        Ions currently in this node.
    capacity : int
        Maximum number of ions this node can hold.
    """
    
    def __init__(
        self,
        idx: int,
        x: float,
        y: float,
        capacity: int,
        label: str = "N",
        color: str = "gray",
        ions: Optional[Sequence[Ion]] = None,
    ) -> None:
        """Initialize a node.
        
        Parameters
        ----------
        idx : int
            Unique node identifier.
        x, y : float
            Position coordinates.
        capacity : int
            Maximum ion capacity.
        label : str
            Display label prefix.
        color : str
            Visualization color.
        ions : Optional[Sequence[Ion]]
            Initial ions in the node.
        """
        self._idx = idx
        self._position_x = x
        self._position_y = y
        self._capacity = capacity
        self._label = label
        self._color = color
        self._ions: List[Ion] = list(ions) if ions else []
        
        # Set parent reference for initial ions
        for ion in self._ions:
            ion.set_state(parent=self)
    
    @property
    def idx(self) -> int:
        return self._idx
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self._position_x, self._position_y)
    
    @property
    def allowed_operations(self) -> Set[QCCDOperation]:
        # Base node operations - subclasses add more
        return {
            QCCDOperation.SPLIT,
            QCCDOperation.MOVE,
            QCCDOperation.MERGE,
        }
    
    @property
    def ions(self) -> List[Ion]:
        """Ions currently in this node."""
        return self._ions
    
    @property
    def num_ions(self) -> int:
        """Current ion count."""
        return len(self._ions)
    
    @property
    def capacity(self) -> int:
        """Maximum ion capacity."""
        return self._capacity
    
    @property
    def is_full(self) -> bool:
        """Check if node is at capacity."""
        return self.num_ions >= self._capacity
    
    @property
    def is_empty(self) -> bool:
        """Check if node has no ions."""
        return self.num_ions == 0
    
    @property
    def label(self) -> str:
        """Display label with index."""
        return f"{self._label}{self._idx}"
    
    @property
    def color(self) -> str:
        """Visualization color."""
        return self._color
    
    @property
    def motional_energy(self) -> float:
        """Total motional energy of non-cooling ions."""
        return sum(
            ion.motional_energy
            for ion in self._ions
            if not isinstance(ion, CoolingIon)
        )
    
    @property
    def qubit_ions(self) -> List[QubitIon]:
        """Get only qubit ions in this node."""
        return [ion for ion in self._ions if isinstance(ion, QubitIon)]
    
    @property
    def cooling_ions(self) -> List[CoolingIon]:
        """Get only cooling ions in this node."""
        return [ion for ion in self._ions if isinstance(ion, CoolingIon)]
    
    def add_ion(
        self,
        ion: Ion,
        position: int = -1,
    ) -> None:
        """Add an ion to this node.
        
        Parameters
        ----------
        ion : Ion
            Ion to add.
        position : int
            Position in ion list (-1 for end).
            
        Raises
        ------
        ValueError
            If node is at capacity.
        """
        if self.is_full:
            raise ValueError(
                f"Cannot add ion to {self.label}: at capacity ({self._capacity})"
            )
        
        if position < 0:
            self._ions.append(ion)
        else:
            self._ions.insert(position, ion)
        
        ion.set_state(x=self._position_x, y=self._position_y, parent=self)
        self._arrange_ions()
    
    def remove_ion(self, ion: Optional[Ion] = None) -> Ion:
        """Remove an ion from this node.
        
        Parameters
        ----------
        ion : Optional[Ion]
            Ion to remove (first ion if None).
            
        Returns
        -------
        Ion
            The removed ion.
            
        Raises
        ------
        ValueError
            If node is empty or ion not found.
        """
        if self.is_empty:
            raise ValueError(f"Cannot remove ion from {self.label}: empty")
        
        if ion is None:
            ion = self._ions[0]
        
        self._ions.remove(ion)
        self._arrange_ions()
        return ion
    
    def _arrange_ions(self) -> None:
        """Rearrange ion positions within node.
        
        Override in subclasses for specific arrangements.
        """
        for ion in self._ions:
            ion.set_state(x=self._position_x, y=self._position_y)
    
    def distribute_heating(self, quanta: float) -> None:
        """Distribute heating energy among ions."""
        if self._ions:
            per_ion = quanta / len(self._ions)
            for ion in self._ions:
                ion.add_motional_energy(per_ion)
    
    def cool(self) -> float:
        """Perform sympathetic cooling.
        
        Transfers motional energy from qubit ions to cooling ions.
        
        Returns
        -------
        float
            Total energy transferred.
        """
        cooling = self.cooling_ions
        if not cooling:
            return 0.0
        
        # Collect energy from qubit ions
        total_energy = 0.0
        for ion in self.qubit_ions:
            total_energy += ion.cool()
        
        # Distribute to cooling ions
        per_cooling = total_energy / len(cooling)
        for ion in cooling:
            ion.add_motional_energy(per_cooling)
        
        return total_energy
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.label}, ions={self.num_ions}/{self._capacity})"


class Junction(QCCDNode):
    """Junction node for routing ions.
    
    Junctions connect multiple trap segments and allow ions to change
    direction during transport. They typically have lower capacity
    than traps.
    """
    
    DEFAULT_CAPACITY = 1
    DEFAULT_COLOR = "orange"
    DEFAULT_LABEL = "J"
    
    def __init__(
        self,
        idx: int,
        x: float,
        y: float,
        capacity: int = DEFAULT_CAPACITY,
        label: str = DEFAULT_LABEL,
        color: str = DEFAULT_COLOR,
    ) -> None:
        super().__init__(idx, x, y, capacity, label, color)
    
    @property
    def allowed_operations(self) -> Set[QCCDOperation]:
        return {
            QCCDOperation.JUNCTION_CROSSING,
            QCCDOperation.SPLIT,
            QCCDOperation.MOVE,
            QCCDOperation.MERGE,
        }


class Trap(QCCDNode):
    """Linear trap segment for holding ion chains.
    
    Traps are the primary containers for ions, arranged as linear
    chains along either horizontal or vertical axes.
    
    Attributes
    ----------
    is_horizontal : bool
        True if ions arranged horizontally.
    spacing : float
        Distance between adjacent ions.
    """
    
    # Background heating rate in quanta/second
    BACKGROUND_HEATING_RATE = 4.0
    
    def __init__(
        self,
        idx: int,
        x: float,
        y: float,
        capacity: int,
        is_horizontal: bool = False,
        spacing: float = 10.0,
        label: str = "T",
        color: str = "lightgray",
        ions: Optional[Sequence[Ion]] = None,
    ) -> None:
        """Initialize a trap.
        
        Parameters
        ----------
        idx : int
            Unique trap identifier.
        x, y : float
            Center position.
        capacity : int
            Maximum ions.
        is_horizontal : bool
            Ion chain orientation.
        spacing : float
            Inter-ion spacing.
        label : str
            Display label.
        color : str
            Visualization color.
        ions : Optional[Sequence[Ion]]
            Initial ions.
        """
        self._is_horizontal = is_horizontal
        self._spacing = spacing
        super().__init__(idx, x, y, capacity, label, color, ions)
        self._arrange_ions()
    
    @property
    def allowed_operations(self) -> Set[QCCDOperation]:
        return {
            QCCDOperation.CRYSTAL_ROTATION,
            QCCDOperation.SPLIT,
            QCCDOperation.MOVE,
            QCCDOperation.MERGE,
            QCCDOperation.QUBIT_RESET,
            QCCDOperation.RECOOLING,
        }
    
    @property
    def is_horizontal(self) -> bool:
        """True if ions arranged horizontally."""
        return self._is_horizontal
    
    @property
    def spacing(self) -> float:
        """Distance between adjacent ions."""
        return self._spacing
    
    @property
    def has_cooling_ion(self) -> bool:
        """Check if trap has a cooling ion."""
        return bool(self.cooling_ions)
    
    def _arrange_ions(self) -> None:
        """Arrange ions in a linear chain centered on the trap."""
        n = len(self._ions)
        for i, ion in enumerate(self._ions):
            offset = (i - n / 2 + 0.5) * self._spacing
            if self._is_horizontal:
                ion.set_state(
                    x=self._position_x + offset,
                    y=self._position_y,
                )
            else:
                ion.set_state(
                    x=self._position_x,
                    y=self._position_y + offset,
                )
    
    def get_edge_ion(self, end: int = 0) -> Optional[Ion]:
        """Get ion at either end of the chain.
        
        Parameters
        ----------
        end : int
            0 for first ion, -1 for last ion.
            
        Returns
        -------
        Optional[Ion]
            Ion at the specified end, or None if empty.
        """
        if not self._ions:
            return None
        return self._ions[0] if end == 0 else self._ions[-1]


class ManipulationTrap(Trap):
    """Trap with gate execution capability.
    
    Manipulation traps have tighter confinement and laser access
    for performing quantum gates. They typically have lower capacity
    than storage traps.
    """
    
    DEFAULT_CAPACITY = 3
    DEFAULT_COLOR = "lightyellow"
    DEFAULT_LABEL = "MT"
    
    def __init__(
        self,
        idx: int,
        x: float,
        y: float,
        capacity: int = DEFAULT_CAPACITY,
        is_horizontal: bool = False,
        spacing: float = 10.0,
        label: str = DEFAULT_LABEL,
        color: str = DEFAULT_COLOR,
        ions: Optional[Sequence[Ion]] = None,
    ) -> None:
        super().__init__(
            idx, x, y, capacity, is_horizontal, spacing, label, color, ions
        )
    
    @property
    def allowed_operations(self) -> Set[QCCDOperation]:
        base = super().allowed_operations
        return base | {
            QCCDOperation.ONE_QUBIT_GATE,
            QCCDOperation.TWO_QUBIT_MS_GATE,
            QCCDOperation.MEASUREMENT,
            QCCDOperation.GATE_SWAP,
        }


class StorageTrap(Trap):
    """Trap for storing idle qubits.
    
    Storage traps have higher capacity but cannot perform gates.
    They are used to hold qubits not currently needed for computation.
    """
    
    DEFAULT_CAPACITY = 5
    DEFAULT_COLOR = "lightgray"
    DEFAULT_LABEL = "ST"
    
    def __init__(
        self,
        idx: int,
        x: float,
        y: float,
        capacity: int = DEFAULT_CAPACITY,
        is_horizontal: bool = False,
        spacing: float = 10.0,
        label: str = DEFAULT_LABEL,
        color: str = DEFAULT_COLOR,
        ions: Optional[Sequence[Ion]] = None,
    ) -> None:
        super().__init__(
            idx, x, y, capacity, is_horizontal, spacing, label, color, ions
        )


# =============================================================================
# Crossing (Connection between nodes)
# =============================================================================

class Crossing:
    """Connection between two QCCD nodes.
    
    A crossing represents the physical channel between two trap segments
    or between a trap and a junction. Ions are moved through crossings
    during transport operations.
    
    Attributes
    ----------
    idx : int
        Unique crossing identifier.
    source : QCCDNode
        One endpoint of the crossing.
    target : QCCDNode
        Other endpoint of the crossing.
    ion : Optional[Ion]
        Ion currently in transit (if any).
    """
    
    DEFAULT_LABEL = "X"
    
    def __init__(
        self,
        idx: int,
        source: QCCDNode,
        target: QCCDNode,
        label: str = DEFAULT_LABEL,
    ) -> None:
        """Initialize a crossing.
        
        Parameters
        ----------
        idx : int
            Unique crossing identifier.
        source : QCCDNode
            First endpoint.
        target : QCCDNode
            Second endpoint.
        label : str
            Display label.
        """
        self._idx = idx
        self._source = source
        self._target = target
        self._label = label
        self._ion: Optional[Ion] = None
        self._ion_at_source: bool = True
    
    @property
    def idx(self) -> int:
        """Crossing identifier."""
        return self._idx
    
    @property
    def label(self) -> str:
        """Display label."""
        return f"{self._label}{self._idx}: {self._source.label}↔{self._target.label}"
    
    @property
    def source(self) -> QCCDNode:
        """First endpoint."""
        return self._source
    
    @property
    def target(self) -> QCCDNode:
        """Second endpoint."""
        return self._target
    
    @property
    def endpoints(self) -> Tuple[QCCDNode, QCCDNode]:
        """Both endpoints."""
        return (self._source, self._target)
    
    @property
    def ion(self) -> Optional[Ion]:
        """Ion currently in transit."""
        return self._ion
    
    @property
    def is_occupied(self) -> bool:
        """Check if crossing has an ion in transit."""
        return self._ion is not None
    
    @property
    def position(self) -> Tuple[float, float]:
        """Center position of the crossing."""
        sx, sy = self._source.position
        tx, ty = self._target.position
        return ((sx + tx) / 2, (sy + ty) / 2)
    
    @property
    def allowed_operations(self) -> Set[QCCDOperation]:
        """Operations available at crossings."""
        return {
            QCCDOperation.SPLIT,
            QCCDOperation.MOVE,
            QCCDOperation.MERGE,
            QCCDOperation.JUNCTION_CROSSING,
        }
    
    def ion_location(self) -> Optional[QCCDNode]:
        """Get which endpoint the ion is closer to.
        
        Returns
        -------
        Optional[QCCDNode]
            The node the ion is near, or None if no ion.
        """
        if self._ion is None:
            return None
        return self._source if self._ion_at_source else self._target
    
    def connects(self, node: QCCDNode) -> bool:
        """Check if this crossing connects to a node."""
        return node is self._source or node is self._target
    
    def other_endpoint(self, node: QCCDNode) -> QCCDNode:
        """Get the endpoint opposite to the given node.
        
        Parameters
        ----------
        node : QCCDNode
            One endpoint.
            
        Returns
        -------
        QCCDNode
            The other endpoint.
            
        Raises
        ------
        ValueError
            If node is not an endpoint.
        """
        if node is self._source:
            return self._target
        elif node is self._target:
            return self._source
        else:
            raise ValueError(f"Node {node.label} is not an endpoint of {self.label}")
    
    def set_ion(self, ion: Ion, from_node: QCCDNode) -> None:
        """Place an ion in the crossing.
        
        Parameters
        ----------
        ion : Ion
            Ion entering the crossing.
        from_node : QCCDNode
            Node the ion is coming from.
            
        Raises
        ------
        ValueError
            If crossing already occupied or node invalid.
        """
        if self._ion is not None:
            raise ValueError(f"Crossing {self.label} already occupied")
        if not self.connects(from_node):
            raise ValueError(f"Node {from_node.label} not connected to {self.label}")
        
        self._ion = ion
        self._ion_at_source = (from_node is self._source)
        
        # Position ion in crossing
        x, y = self.position
        ion.set_state(x=x, y=y, parent=self)
    
    def move_ion(self) -> QCCDNode:
        """Move ion to the opposite endpoint.
        
        Returns
        -------
        QCCDNode
            The endpoint the ion moved toward.
            
        Raises
        ------
        ValueError
            If no ion in crossing.
        """
        if self._ion is None:
            raise ValueError(f"No ion in crossing {self.label}")
        
        self._ion_at_source = not self._ion_at_source
        return self.ion_location()
    
    def remove_ion(self) -> Ion:
        """Remove ion from crossing.
        
        Returns
        -------
        Ion
            The removed ion.
            
        Raises
        ------
        ValueError
            If no ion in crossing.
        """
        if self._ion is None:
            raise ValueError(f"No ion in crossing {self.label}")
        
        ion = self._ion
        self._ion = None
        return ion
    
    def __repr__(self) -> str:
        status = f", ion={self._ion.label}" if self._ion else ""
        return f"Crossing({self.label}{status})"


# =============================================================================
# Architecture Configuration Dataclass
# =============================================================================

@dataclass
class QCCDWISEConfig:
    """Configuration for a WISE (Width-Independent Scalable Entanglement) architecture.
    
    WISE is an m×n grid of manipulation traps connected through junctions,
    optimized for efficient routing via SAT-based solvers.
    
    Attributes
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    k : int
        Maximum ions per trap.
    """
    m: int
    n: int
    k: int = 3
    
    @property
    def num_traps(self) -> int:
        """Total number of traps in the grid."""
        return self.m * self.n
    
    @property
    def num_junctions(self) -> int:
        """Total number of junctions."""
        return (self.m - 1) * self.n + self.m * (self.n - 1)


@dataclass
class LinearChainConfig:
    """Configuration for a linear chain architecture.
    
    A simple architecture with a single trap holding all ions,
    providing all-to-all connectivity within the chain.
    
    Attributes
    ----------
    num_ions : int
        Total number of ions.
    ion_spacing : float
        Distance between adjacent ions (μm).
    """
    num_ions: int
    ion_spacing: float = 5.0


# =============================================================================
# Factory Functions
# =============================================================================

def create_qubit_ion(idx: int) -> QubitIon:
    """Create a qubit ion with standard settings."""
    return QubitIon(idx=idx)


def create_cooling_ion(idx: int) -> CoolingIon:
    """Create a cooling ion with standard settings."""
    return CoolingIon(idx=idx)


def create_manipulation_trap(
    idx: int,
    x: float,
    y: float,
    num_qubits: int = 0,
    include_cooling: bool = False,
    **kwargs: Any,
) -> ManipulationTrap:
    """Create a manipulation trap with optional initial ions.
    
    Parameters
    ----------
    idx : int
        Trap identifier.
    x, y : float
        Position.
    num_qubits : int
        Number of qubit ions to create.
    include_cooling : bool
        Whether to add a cooling ion.
    **kwargs
        Additional trap parameters.
        
    Returns
    -------
    ManipulationTrap
        Configured manipulation trap.
    """
    ions: List[Ion] = []
    ion_idx = idx * 100  # Space out ion indices
    
    for i in range(num_qubits):
        ions.append(create_qubit_ion(ion_idx + i))
    
    if include_cooling:
        ions.append(create_cooling_ion(ion_idx + num_qubits))
    
    return ManipulationTrap(idx=idx, x=x, y=y, ions=ions, **kwargs)


def create_storage_trap(
    idx: int,
    x: float,
    y: float,
    num_qubits: int = 0,
    **kwargs: Any,
) -> StorageTrap:
    """Create a storage trap with optional initial ions.
    
    Parameters
    ----------
    idx : int
        Trap identifier.
    x, y : float
        Position.
    num_qubits : int
        Number of qubit ions to create.
    **kwargs
        Additional trap parameters.
        
    Returns
    -------
    StorageTrap
        Configured storage trap.
    """
    ions: List[Ion] = []
    ion_idx = idx * 100
    
    for i in range(num_qubits):
        ions.append(create_qubit_ion(ion_idx + i))
    
    return StorageTrap(idx=idx, x=x, y=y, ions=ions, **kwargs)
