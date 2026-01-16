# src/qectostim/experiments/hardware_simulation/core/architecture.py
"""
Abstract hardware architecture definitions.

Defines the topology and constraints of quantum hardware platforms.
Each platform (trapped ion, superconducting, neutral atom) implements
these interfaces with platform-specific details.
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
    Iterator,
    Any,
    FrozenSet,
)

import networkx as nx
from typing import Protocol, runtime_checkable


# =============================================================================
# Position Model and Platform Capabilities
# =============================================================================

class PositionModel(Enum):
    """How qubit positions are modeled in an architecture.
    
    Different quantum computing platforms have fundamentally different
    approaches to qubit positioning:
    
    - FIXED: Qubits don't move (superconducting, some neutral atom)
    - DISCRETE: Qubits move between discrete zones (QCCD trapped ion)
    - CONTINUOUS: Qubits can be at arbitrary positions (tweezer arrays)
    """
    FIXED = auto()            # Qubits are stationary (e.g., superconducting)
    DISCRETE = auto()         # Qubits move between discrete zones (e.g., QCCD)
    CONTINUOUS = auto()       # Qubits have continuous positions (e.g., tweezers)


@runtime_checkable
class PlatformCapabilities(Protocol):
    """Protocol defining queryable hardware capabilities.
    
    Core code queries capabilities instead of checking platform types.
    This enables technology-agnostic algorithms.
    
    Example
    -------
    >>> if arch.supports_qubit_movement():
    ...     use_transport_routing()
    ... else:
    ...     use_swap_routing()
    """
    
    def supports_qubit_movement(self) -> bool:
        """Whether qubits can physically move between zones."""
        ...
    
    def supports_parallel_2q_gates(self) -> bool:
        """Whether multiple 2Q gates can execute simultaneously."""
        ...
    
    def supports_global_operations(self) -> bool:
        """Whether global operations (all qubits at once) are native."""
        ...
    
    def native_2q_gate_type(self) -> str:
        """Name of the native two-qubit gate (e.g., 'CZ', 'MS', 'CX')."""
        ...
    
    def position_model(self) -> PositionModel:
        """How qubit positions are modeled."""
        ...
    
    def max_interaction_distance(self) -> Optional[float]:
        """Maximum distance for qubit interaction, or None if unlimited."""
        ...


@runtime_checkable
class InteractionModel(Protocol):
    """Protocol for distance-dependent qubit interactions.
    
    Different platforms have different interaction physics:
    - Neutral atoms: Rydberg blockade (1/r^6 dipole-dipole)
    - Trapped ions: Coulomb interaction (long-range in chain)
    - Superconducting: Direct capacitive/inductive (nearest-neighbor)
    
    This protocol abstracts interaction strength/fidelity calculations
    so algorithms can reason about interaction quality without
    platform-specific knowledge.
    
    Example Implementation
    ----------------------
    >>> class RydbergInteractionModel:
    ...     def __init__(self, blockade_radius: float):
    ...         self.blockade_radius = blockade_radius
    ...     
    ...     def interaction_strength(self, distance: float, **context) -> float:
    ...         # Rydberg blockade: strong within radius, weak outside
    ...         if distance <= self.blockade_radius:
    ...             return 1.0
    ...         return (self.blockade_radius / distance) ** 6
    """
    
    def interaction_strength(
        self,
        distance: float,
        **context,
    ) -> float:
        """Calculate relative interaction strength at a distance.
        
        Parameters
        ----------
        distance : float
            Distance between qubits (in platform units).
        **context
            Platform-specific parameters (e.g., laser power).
            
        Returns
        -------
        float
            Interaction strength in [0, 1], where 1 is maximum.
        """
        ...
    
    def interaction_fidelity(
        self,
        distance: float,
        gate_type: str = "CZ",
        **context,
    ) -> float:
        """Calculate expected gate fidelity at a distance.
        
        Parameters
        ----------
        distance : float
            Distance between qubits.
        gate_type : str
            Type of gate being performed.
        **context
            Platform-specific parameters.
            
        Returns
        -------
        float
            Expected gate fidelity in [0, 1].
        """
        ...
    
    def max_interaction_distance(self, min_fidelity: float = 0.99) -> float:
        """Maximum distance where interaction fidelity exceeds threshold.
        
        Parameters
        ----------
        min_fidelity : float
            Minimum acceptable fidelity.
            
        Returns
        -------
        float
            Maximum interaction distance.
        """
        ...
    
    def is_blockaded(
        self,
        qubit_positions: List[Tuple[float, ...]],
        target_qubit: int,
        **context,
    ) -> bool:
        """Check if a qubit is blockaded by nearby qubits.
        
        Relevant for Rydberg-based neutral atom systems where
        nearby atoms in Rydberg states prevent excitation.
        
        Returns False for platforms without blockade effects.
        """
        ...


@runtime_checkable
class CalibrationData(Protocol):
    """Protocol for accessing hardware calibration data.
    
    Real hardware has calibration data that affects gate fidelities,
    frequencies, timings, etc. This protocol provides a technology-agnostic
    interface for querying calibration information.
    
    Example Implementation
    ----------------------
    >>> class IBMCalibrationData:
    ...     def gate_fidelity(self, gate: str, qubits: Tuple[int, ...]) -> float:
    ...         return self._backend_properties.gate_error(gate, qubits)
    """
    
    def gate_fidelity(
        self,
        gate: str,
        qubits: Tuple[int, ...],
    ) -> float:
        """Get calibrated fidelity for a gate on specific qubits."""
        ...
    
    def t1_time(self, qubit: int) -> float:
        """Get T1 relaxation time for a qubit in microseconds."""
        ...
    
    def t2_time(self, qubit: int) -> float:
        """Get T2 dephasing time for a qubit in microseconds."""
        ...
    
    def readout_fidelity(self, qubit: int) -> float:
        """Get measurement readout fidelity for a qubit."""
        ...
    
    def calibration_timestamp(self) -> Optional[float]:
        """Get timestamp of last calibration (Unix time)."""
        ...


class ZoneType(Enum):
    """Types of zones in hardware architectures.
    
    These are generic, technology-agnostic zone types.
    Platform-specific zone types should extend this in their modules.
    
    Note: Platform modules may define their own zone type enums that
    map to these core types for compatibility.
    """
    # Core generic types - technology independent
    GATE = auto()             # Zone where gate operations occur
    STORAGE = auto()          # Zone where qubits wait (idle)
    ROUTING = auto()          # Zone for qubit transport/routing
    READOUT = auto()          # Zone where measurement occurs
    ANCILLA = auto()          # Zone for ancilla/auxiliary qubits
    
    # Extended generic types
    MULTI_QUBIT = auto()      # Zone supporting multi-qubit interactions
    SINGLE_QUBIT = auto()     # Zone supporting only single-qubit ops
    BUFFER = auto()           # Intermediate/buffer zone
    
    # Additional generic types for various platforms
    COMPUTATION = auto()      # General computation zone (superconducting/trapped ion)
    INTERACTION = auto()      # Dedicated interaction zone (Rydberg, MS gates)
    LOADING = auto()          # Qubit loading/initialization zone (neutral atom, ion traps)
    JUNCTION = auto()         # Junction/routing intersection (QCCD T-junctions)
    RESERVOIR = auto()        # Reservoir for spare qubits (neutral atom reload)


@dataclass(frozen=True)
class Zone:
    """A zone in the hardware architecture.
    
    Zones are locations where qubits can reside and operations can occur.
    The allowed operations and capacity depend on the zone type.
    
    Attributes
    ----------
    id : str
        Unique identifier for the zone.
    zone_type : ZoneType
        Type of zone determining allowed operations.
    capacity : int
        Maximum number of qubits that can occupy this zone.
    position : Optional[Tuple[float, ...]]
        Physical coordinates (for visualization and distance calculations).
    allowed_operations : FrozenSet[str]
        Set of operation names allowed in this zone.
    metadata : Dict[str, Any]
        Platform-specific additional properties.
    """
    id: str
    zone_type: ZoneType
    capacity: int = 1
    position: Optional[Tuple[float, ...]] = None
    allowed_operations: FrozenSet[str] = field(default_factory=frozenset)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Zone):
            return NotImplemented
        return self.id == other.id


@dataclass
class ConnectivityGraph:
    """Graph representation of hardware connectivity.
    
    Wraps a networkx graph with hardware-specific semantics.
    Nodes are zones, edges represent possible qubit transport paths.
    
    Attributes
    ----------
    graph : nx.Graph
        The underlying networkx graph.
    zones : Dict[str, Zone]
        Mapping from zone ID to Zone object.
    """
    graph: nx.Graph = field(default_factory=nx.Graph)
    zones: Dict[str, Zone] = field(default_factory=dict)
    
    def add_zone(self, zone: Zone) -> None:
        """Add a zone to the connectivity graph."""
        self.zones[zone.id] = zone
        self.graph.add_node(zone.id, zone=zone)
    
    def add_connection(
        self,
        zone1_id: str,
        zone2_id: str,
        weight: float = 1.0,
        bidirectional: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a connection (edge) between two zones.
        
        Parameters
        ----------
        zone1_id, zone2_id : str
            IDs of the zones to connect.
        weight : float
            Edge weight (typically transport time or distance).
        bidirectional : bool
            If True, connection works both ways.
        metadata : Optional[Dict]
            Additional edge properties (e.g., transport fidelity).
        """
        edge_data = {"weight": weight, **(metadata or {})}
        if bidirectional:
            self.graph.add_edge(zone1_id, zone2_id, **edge_data)
        else:
            # Convert to directed graph if needed
            if not self.graph.is_directed():
                self.graph = self.graph.to_directed()
            self.graph.add_edge(zone1_id, zone2_id, **edge_data)
    
    def get_zone(self, zone_id: str) -> Optional[Zone]:
        """Get a zone by ID."""
        return self.zones.get(zone_id)
    
    def neighbors(self, zone_id: str) -> Iterator[str]:
        """Get IDs of neighboring zones."""
        return self.graph.neighbors(zone_id)
    
    def shortest_path(
        self,
        source: str,
        target: str,
        weight: str = "weight",
    ) -> List[str]:
        """Find shortest path between two zones."""
        return nx.shortest_path(self.graph, source, target, weight=weight)
    
    def path_length(
        self,
        source: str,
        target: str,
        weight: str = "weight",
    ) -> float:
        """Get the weighted path length between two zones."""
        return nx.shortest_path_length(self.graph, source, target, weight=weight)
    
    def computation_zones(self) -> List[Zone]:
        """Get all zones where computation (gates) can occur."""
        return [
            z for z in self.zones.values()
            if z.zone_type in {ZoneType.GATE, ZoneType.MULTI_QUBIT, ZoneType.SINGLE_QUBIT}
        ]
    
    def storage_zones(self) -> List[Zone]:
        """Get all storage zones."""
        return [
            z for z in self.zones.values()
            if z.zone_type == ZoneType.STORAGE
        ]
    
    def routing_zones(self) -> List[Zone]:
        """Get all zones used for qubit transport/routing."""
        return [
            z for z in self.zones.values()
            if z.zone_type in {ZoneType.ROUTING, ZoneType.BUFFER}
        ]


@dataclass
class LayoutSnapshot:
    """Snapshot of layout state for backtracking.
    
    Used by SAT-based routing to checkpoint and restore
    qubit positions during search.
    
    Attributes
    ----------
    qubit_to_zone : Dict[int, str]
        Snapshot of qubit positions.
    snapshot_id : int
        Unique identifier for this snapshot.
    timestamp : Optional[float]
        When the snapshot was taken.
    """
    qubit_to_zone: Dict[int, str]
    snapshot_id: int = 0
    timestamp: Optional[float] = None


class LayoutTracker:
    """Tracks the dynamic layout of qubits across zones.
    
    For reconfigurable architectures (trapped ion, neutral atom),
    qubits move between zones during computation. LayoutTracker
    maintains the current qubit positions and supports:
    - Position queries and updates
    - Snapshots for backtracking (SAT-based routing)
    - Conflict detection (capacity, colocation constraints)
    
    This is separated from the architecture to allow:
    - Multiple independent layout states (for parallel search)
    - Lightweight snapshots without copying entire architecture
    - Easy rollback in optimization algorithms
    
    Attributes
    ----------
    qubit_to_zone : Dict[int, str]
        Maps qubit ID to current zone ID.
    zone_to_qubits : Dict[str, Set[int]]
        Maps zone ID to set of qubit IDs.
    zone_capacities : Dict[str, int]
        Maximum capacity of each zone.
    """
    
    def __init__(
        self,
        zone_capacities: Optional[Dict[str, int]] = None,
    ):
        """Initialize the layout tracker.
        
        Parameters
        ----------
        zone_capacities : Optional[Dict[str, int]]
            Zone capacities (zone_id -> max qubits).
        """
        self.qubit_to_zone: Dict[int, str] = {}
        self.zone_to_qubits: Dict[str, Set[int]] = {}
        self.zone_capacities: Dict[str, int] = zone_capacities or {}
        self._history: List[Dict[int, str]] = []  # For snapshots
    
    def place_qubit(self, qubit_id: int, zone_id: str) -> bool:
        """Place a qubit in a zone.
        
        Parameters
        ----------
        qubit_id : int
            Qubit to place.
        zone_id : str
            Target zone.
            
        Returns
        -------
        bool
            True if placement succeeded, False if zone at capacity.
        """
        # Check capacity
        capacity = self.zone_capacities.get(zone_id, float('inf'))
        current = len(self.zone_to_qubits.get(zone_id, set()))
        
        # If qubit already in this zone, no change needed
        if self.qubit_to_zone.get(qubit_id) == zone_id:
            return True
        
        # If qubit was elsewhere, remove from old zone
        old_zone = self.qubit_to_zone.get(qubit_id)
        if old_zone is not None and old_zone in self.zone_to_qubits:
            self.zone_to_qubits[old_zone].discard(qubit_id)
        
        # Check capacity (don't count the qubit we're moving)
        if current >= capacity:
            return False
        
        # Place in new zone
        self.qubit_to_zone[qubit_id] = zone_id
        if zone_id not in self.zone_to_qubits:
            self.zone_to_qubits[zone_id] = set()
        self.zone_to_qubits[zone_id].add(qubit_id)
        
        return True
    
    def remove_qubit(self, qubit_id: int) -> Optional[str]:
        """Remove a qubit from its zone.
        
        Returns
        -------
        Optional[str]
            The zone the qubit was in, or None if not placed.
        """
        zone_id = self.qubit_to_zone.pop(qubit_id, None)
        if zone_id is not None and zone_id in self.zone_to_qubits:
            self.zone_to_qubits[zone_id].discard(qubit_id)
        return zone_id
    
    def move_qubit(self, qubit_id: int, target_zone: str) -> bool:
        """Move a qubit from current zone to target zone.
        
        Parameters
        ----------
        qubit_id : int
            Qubit to move.
        target_zone : str
            Destination zone.
            
        Returns
        -------
        bool
            True if move succeeded.
        """
        return self.place_qubit(qubit_id, target_zone)
    
    def get_zone(self, qubit_id: int) -> Optional[str]:
        """Get the current zone of a qubit."""
        return self.qubit_to_zone.get(qubit_id)
    
    def get_qubits(self, zone_id: str) -> Set[int]:
        """Get all qubits in a zone."""
        return self.zone_to_qubits.get(zone_id, set()).copy()
    
    def zone_occupancy(self, zone_id: str) -> int:
        """Get number of qubits in a zone."""
        return len(self.zone_to_qubits.get(zone_id, set()))
    
    def zone_available_capacity(self, zone_id: str) -> int:
        """Get remaining capacity in a zone."""
        capacity = self.zone_capacities.get(zone_id, float('inf'))
        current = self.zone_occupancy(zone_id)
        return max(0, capacity - current)
    
    def can_colocate(self, qubit_ids: List[int], zone_id: str) -> bool:
        """Check if qubits can all be placed in a zone.
        
        Parameters
        ----------
        qubit_ids : List[int]
            Qubits to colocate.
        zone_id : str
            Target zone.
            
        Returns
        -------
        bool
            True if zone can hold all qubits.
        """
        capacity = self.zone_capacities.get(zone_id, float('inf'))
        current = self.zone_to_qubits.get(zone_id, set())
        
        # Count how many of the requested qubits are NOT already in this zone
        new_qubits = sum(1 for q in qubit_ids if q not in current)
        
        return len(current) + new_qubits <= capacity
    
    def are_colocated(self, qubit_ids: List[int]) -> bool:
        """Check if qubits are all in the same zone."""
        if not qubit_ids:
            return True
        zones = [self.qubit_to_zone.get(q) for q in qubit_ids]
        return all(z == zones[0] and z is not None for z in zones)
    
    def snapshot(self) -> int:
        """Create a snapshot of current layout.
        
        Returns
        -------
        int
            Snapshot ID for later restoration.
        """
        self._history.append(dict(self.qubit_to_zone))
        return len(self._history) - 1
    
    def restore(self, snapshot_id: int) -> None:
        """Restore layout to a previous snapshot.
        
        Parameters
        ----------
        snapshot_id : int
            ID from a previous snapshot() call.
        """
        if snapshot_id < 0 or snapshot_id >= len(self._history):
            raise ValueError(f"Invalid snapshot ID: {snapshot_id}")
        
        # Restore qubit_to_zone
        self.qubit_to_zone = dict(self._history[snapshot_id])
        
        # Rebuild zone_to_qubits
        self.zone_to_qubits = {}
        for qubit_id, zone_id in self.qubit_to_zone.items():
            if zone_id not in self.zone_to_qubits:
                self.zone_to_qubits[zone_id] = set()
            self.zone_to_qubits[zone_id].add(qubit_id)
        
        # Truncate history to this point
        self._history = self._history[:snapshot_id + 1]
    
    def copy(self) -> "LayoutTracker":
        """Create a deep copy of this tracker."""
        tracker = LayoutTracker(dict(self.zone_capacities))
        tracker.qubit_to_zone = dict(self.qubit_to_zone)
        tracker.zone_to_qubits = {
            zone: set(qubits)
            for zone, qubits in self.zone_to_qubits.items()
        }
        return tracker
    
    def as_dict(self) -> Dict[int, str]:
        """Get layout as a dictionary."""
        return dict(self.qubit_to_zone)
    
    @classmethod
    def from_dict(
        cls,
        layout: Dict[int, str],
        zone_capacities: Optional[Dict[str, int]] = None,
    ) -> "LayoutTracker":
        """Create tracker from a dictionary layout."""
        tracker = cls(zone_capacities)
        for qubit_id, zone_id in layout.items():
            tracker.place_qubit(qubit_id, zone_id)
        return tracker
    
    def __repr__(self) -> str:
        return f"LayoutTracker({len(self.qubit_to_zone)} qubits in {len(self.zone_to_qubits)} zones)"


# =============================================================================
# Discrete Layout Abstraction
# =============================================================================

@dataclass
class GridPosition:
    """A position on a discrete grid.
    
    Technology-agnostic representation of a grid position.
    Works for WISE grids, heavy-hex lattices, tweezer arrays, etc.
    
    Attributes
    ----------
    row : int
        Row index (0-indexed).
    col : int
        Column index (0-indexed).
    layer : int
        Optional layer for 3D grids (default 0).
    """
    row: int
    col: int
    layer: int = 0
    
    def __hash__(self) -> int:
        return hash((self.row, self.col, self.layer))
    
    def manhattan_distance(self, other: "GridPosition") -> int:
        """Manhattan distance to another position."""
        return (
            abs(self.row - other.row) +
            abs(self.col - other.col) +
            abs(self.layer - other.layer)
        )
    
    def neighbors(self, include_diagonal: bool = False) -> List["GridPosition"]:
        """Get adjacent grid positions."""
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if include_diagonal:
            offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        return [
            GridPosition(self.row + dr, self.col + dc, self.layer)
            for dr, dc in offsets
        ]
    
    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.row, self.col, self.layer)


class DiscreteLayout:
    """Technology-agnostic discrete layout for grid-based architectures.
    
    Manages qubit positions on a discrete grid. Works for:
    - WISE grids (trapped ion)
    - Heavy-hex lattices (superconducting)
    - Tweezer arrays (neutral atom)
    - Square grids (various)
    
    This is a higher-level abstraction than LayoutTracker that
    understands grid structure and can compute distances, paths,
    and adjacency relationships.
    
    Attributes
    ----------
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    qubit_positions : Dict[int, GridPosition]
        Current qubit positions on the grid.
    """
    
    def __init__(
        self,
        rows: int,
        cols: int,
        layers: int = 1,
    ):
        """Initialize a discrete layout.
        
        Parameters
        ----------
        rows : int
            Number of rows.
        cols : int
            Number of columns.
        layers : int
            Number of layers (for 3D grids).
        """
        self.rows = rows
        self.cols = cols
        self.layers = layers
        self.qubit_positions: Dict[int, GridPosition] = {}
        self._position_to_qubit: Dict[Tuple[int, int, int], int] = {}
        self._history: List[Dict[int, GridPosition]] = []
    
    @property
    def num_sites(self) -> int:
        """Total number of grid sites."""
        return self.rows * self.cols * self.layers
    
    def is_valid_position(self, pos: GridPosition) -> bool:
        """Check if a position is within the grid bounds."""
        return (
            0 <= pos.row < self.rows and
            0 <= pos.col < self.cols and
            0 <= pos.layer < self.layers
        )
    
    def is_occupied(self, pos: GridPosition) -> bool:
        """Check if a position is occupied by a qubit."""
        return pos.as_tuple() in self._position_to_qubit
    
    def place_qubit(self, qubit_id: int, pos: GridPosition) -> bool:
        """Place a qubit at a grid position.
        
        Returns
        -------
        bool
            True if placement succeeded, False if position invalid or occupied.
        """
        if not self.is_valid_position(pos):
            return False
        
        # Remove from old position if exists
        old_pos = self.qubit_positions.get(qubit_id)
        if old_pos is not None:
            del self._position_to_qubit[old_pos.as_tuple()]
        
        # Check if new position is occupied by another qubit
        existing = self._position_to_qubit.get(pos.as_tuple())
        if existing is not None and existing != qubit_id:
            return False
        
        # Place at new position
        self.qubit_positions[qubit_id] = pos
        self._position_to_qubit[pos.as_tuple()] = qubit_id
        return True
    
    def remove_qubit(self, qubit_id: int) -> Optional[GridPosition]:
        """Remove a qubit from the grid.
        
        Returns
        -------
        Optional[GridPosition]
            The position the qubit was at, or None if not on grid.
        """
        pos = self.qubit_positions.pop(qubit_id, None)
        if pos is not None:
            self._position_to_qubit.pop(pos.as_tuple(), None)
        return pos
    
    def get_position(self, qubit_id: int) -> Optional[GridPosition]:
        """Get the position of a qubit."""
        return self.qubit_positions.get(qubit_id)
    
    def get_qubit_at(self, pos: GridPosition) -> Optional[int]:
        """Get the qubit at a position, if any."""
        return self._position_to_qubit.get(pos.as_tuple())
    
    def move_qubit(self, qubit_id: int, new_pos: GridPosition) -> bool:
        """Move a qubit to a new position."""
        return self.place_qubit(qubit_id, new_pos)
    
    def swap_qubits(self, qubit_a: int, qubit_b: int) -> bool:
        """Swap positions of two qubits.
        
        Returns
        -------
        bool
            True if swap succeeded.
        """
        pos_a = self.qubit_positions.get(qubit_a)
        pos_b = self.qubit_positions.get(qubit_b)
        
        if pos_a is None or pos_b is None:
            return False
        
        # Swap
        self.qubit_positions[qubit_a] = pos_b
        self.qubit_positions[qubit_b] = pos_a
        self._position_to_qubit[pos_a.as_tuple()] = qubit_b
        self._position_to_qubit[pos_b.as_tuple()] = qubit_a
        return True
    
    def distance(self, qubit_a: int, qubit_b: int) -> Optional[int]:
        """Manhattan distance between two qubits.
        
        Returns
        -------
        Optional[int]
            Distance, or None if either qubit not on grid.
        """
        pos_a = self.qubit_positions.get(qubit_a)
        pos_b = self.qubit_positions.get(qubit_b)
        
        if pos_a is None or pos_b is None:
            return None
        
        return pos_a.manhattan_distance(pos_b)
    
    def are_adjacent(self, qubit_a: int, qubit_b: int) -> bool:
        """Check if two qubits are adjacent (distance 1)."""
        dist = self.distance(qubit_a, qubit_b)
        return dist == 1 if dist is not None else False
    
    def get_neighbors(self, qubit_id: int) -> List[int]:
        """Get qubits adjacent to a given qubit."""
        pos = self.qubit_positions.get(qubit_id)
        if pos is None:
            return []
        
        neighbors = []
        for neighbor_pos in pos.neighbors():
            if self.is_valid_position(neighbor_pos):
                neighbor_qubit = self.get_qubit_at(neighbor_pos)
                if neighbor_qubit is not None:
                    neighbors.append(neighbor_qubit)
        return neighbors
    
    def get_row(self, row: int) -> List[int]:
        """Get all qubits in a row."""
        return [
            q for q, pos in self.qubit_positions.items()
            if pos.row == row
        ]
    
    def get_column(self, col: int) -> List[int]:
        """Get all qubits in a column."""
        return [
            q for q, pos in self.qubit_positions.items()
            if pos.col == col
        ]
    
    def snapshot(self) -> int:
        """Create a snapshot for backtracking.
        
        Returns
        -------
        int
            Snapshot ID.
        """
        self._history.append(dict(self.qubit_positions))
        return len(self._history) - 1
    
    def restore(self, snapshot_id: int) -> None:
        """Restore to a previous snapshot."""
        if snapshot_id < 0 or snapshot_id >= len(self._history):
            raise ValueError(f"Invalid snapshot ID: {snapshot_id}")
        
        self.qubit_positions = dict(self._history[snapshot_id])
        self._position_to_qubit = {
            pos.as_tuple(): qubit
            for qubit, pos in self.qubit_positions.items()
        }
        self._history = self._history[:snapshot_id + 1]
    
    def copy(self) -> "DiscreteLayout":
        """Create a deep copy."""
        layout = DiscreteLayout(self.rows, self.cols, self.layers)
        layout.qubit_positions = dict(self.qubit_positions)
        layout._position_to_qubit = dict(self._position_to_qubit)
        return layout
    
    def __repr__(self) -> str:
        return f"DiscreteLayout({self.rows}x{self.cols}, {len(self.qubit_positions)} qubits)"


@dataclass
class PhysicalConstraints:
    """Physical constraints of the hardware.
    
    Encapsulates timing, fidelity, and operational constraints
    that affect compilation and simulation.
    
    Attributes
    ----------
    max_parallel_1q_gates : Optional[int]
        Maximum 1-qubit gates that can run simultaneously.
    max_parallel_2q_gates : Optional[int]
        Maximum 2-qubit gates that can run simultaneously.
    max_qubits : int
        Total qubit capacity of the hardware.
    t1_time : Optional[float]
        T1 relaxation time in microseconds.
    t2_time : Optional[float]
        T2 dephasing time in microseconds.
    gate_times : Dict[str, float]
        Typical gate duration in microseconds by gate name.
    readout_time : float
        Measurement duration in microseconds.
    reset_time : float
        Qubit reset duration in microseconds.
    transport_time : Optional[float]
        Typical qubit transport time (for mobile architectures).
    crosstalk_pairs : Optional[Set[Tuple[int, int]]]
        Pairs of qubits with significant crosstalk.
    """
    max_parallel_1q_gates: Optional[int] = None
    max_parallel_2q_gates: Optional[int] = None
    max_qubits: int = 100
    t1_time: Optional[float] = None  # microseconds
    t2_time: Optional[float] = None  # microseconds
    gate_times: Dict[str, float] = field(default_factory=dict)
    readout_time: float = 1.0  # microseconds
    reset_time: float = 1.0   # microseconds
    transport_time: Optional[float] = None  # microseconds
    crosstalk_pairs: Optional[Set[Tuple[int, int]]] = None


class HardwareArchitecture(ABC):
    """Abstract base class for hardware architectures.
    
    Defines the interface that all platform-specific architectures must implement.
    An architecture describes:
    - The native gate set (what operations the hardware can perform)
    - The connectivity (which qubits can interact)
    - Physical constraints (timing, parallelism, fidelity limits)
    - Zone structure (for reconfigurable architectures)
    
    Subclasses implement platform-specific details:
    - TrappedIonArchitecture: QCCD, linear chains
    - SuperconductingArchitecture: fixed coupling, tunable coupling
    - NeutralAtomArchitecture: tweezer arrays, Rydberg interactions
    """
    
    def __init__(
        self,
        name: str,
        num_qubits: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the architecture.
        
        Parameters
        ----------
        name : str
            Human-readable name for the architecture.
        num_qubits : int
            Number of qubits in the system.
        metadata : Optional[Dict]
            Additional platform-specific metadata.
        """
        self.name = name
        self.num_qubits = num_qubits
        self.metadata = metadata or {}
        self._connectivity: Optional[ConnectivityGraph] = None
        self._constraints: Optional[PhysicalConstraints] = None
    
    @abstractmethod
    def native_gate_set(self) -> "NativeGateSet":
        """Return the native gate set for this architecture.
        
        Returns
        -------
        NativeGateSet
            The set of gates natively supported by the hardware.
        """
        ...
    
    @abstractmethod
    def connectivity_graph(self) -> ConnectivityGraph:
        """Return the connectivity graph of the architecture.
        
        Returns
        -------
        ConnectivityGraph
            Graph describing which qubits can interact and transport paths.
        """
        ...
    
    @abstractmethod
    def physical_constraints(self) -> PhysicalConstraints:
        """Return physical constraints of the hardware.
        
        Returns
        -------
        PhysicalConstraints
            Timing, parallelism, and fidelity constraints.
        """
        ...
    
    @abstractmethod
    def zone_types(self) -> List[ZoneType]:
        """Return the zone types present in this architecture.
        
        Returns
        -------
        List[ZoneType]
            Zone types used by this platform.
        """
        ...
    
    @abstractmethod
    def can_interact(self, qubit1: int, qubit2: int) -> bool:
        """Check if two qubits can directly interact (2-qubit gate).
        
        Parameters
        ----------
        qubit1, qubit2 : int
            Physical qubit indices.
            
        Returns
        -------
        bool
            True if a native 2-qubit gate can be applied between them.
        """
        ...
    
    @abstractmethod
    def interaction_distance(self, qubit1: int, qubit2: int) -> int:
        """Get the interaction distance between two qubits.
        
        Returns the minimum number of SWAP operations (or equivalent)
        needed to bring two qubits into interaction range.
        
        Parameters
        ----------
        qubit1, qubit2 : int
            Physical qubit indices.
            
        Returns
        -------
        int
            Number of routing operations needed (0 if directly connected).
        """
        ...
    
    def is_reconfigurable(self) -> bool:
        """Check if the architecture supports qubit reconfiguration.
        
        Reconfigurable architectures (trapped ions, neutral atoms) can
        physically move qubits to change connectivity. Fixed architectures
        (most superconducting) cannot.
        
        Returns
        -------
        bool
            True if qubits can be physically moved.
        """
        return False
    
    def supports_all_to_all(self) -> bool:
        """Check if any qubit pair can eventually interact.
        
        Returns
        -------
        bool
            True if the connectivity graph is fully connected
            (possibly via transport/reconfiguration).
        """
        graph = self.connectivity_graph()
        return nx.is_connected(graph.graph)
    
    def max_zone_capacity(self) -> int:
        """Get the maximum capacity of any single zone."""
        graph = self.connectivity_graph()
        if not graph.zones:
            return self.num_qubits
        return max(z.capacity for z in graph.zones.values())
    
    # =========================================================================
    # PlatformCapabilities Protocol Implementation
    # =========================================================================
    # These provide default implementations. Subclasses should override
    # to provide accurate platform-specific information.
    
    def supports_qubit_movement(self) -> bool:
        """Whether qubits can physically move between zones.
        
        Default: same as is_reconfigurable().
        """
        return self.is_reconfigurable()
    
    def supports_parallel_2q_gates(self) -> bool:
        """Whether multiple 2Q gates can execute simultaneously.
        
        Default: True (most platforms support some parallelism).
        """
        constraints = self.physical_constraints()
        if constraints.max_parallel_2q_gates is not None:
            return constraints.max_parallel_2q_gates > 1
        return True
    
    def supports_global_operations(self) -> bool:
        """Whether global operations (all qubits at once) are native.
        
        Default: False. Override for platforms with global gates
        (e.g., neutral atoms with global Rydberg addressing).
        """
        return False
    
    def native_2q_gate_type(self) -> str:
        """Name of the native two-qubit gate.
        
        Default: tries to determine from native gate set.
        """
        gate_set = self.native_gate_set()
        # Look for common 2Q gate names
        for name in ["CZ", "CX", "CNOT", "MS", "iSWAP", "SWAP", "ZZ"]:
            if gate_set.has_gate(name):
                return name
        return "UNKNOWN"
    
    def position_model(self) -> PositionModel:
        """How qubit positions are modeled.
        
        Default: FIXED for non-reconfigurable, DISCRETE for reconfigurable.
        """
        if self.is_reconfigurable():
            return PositionModel.DISCRETE
        return PositionModel.FIXED
    
    def max_interaction_distance(self) -> Optional[float]:
        """Maximum distance for qubit interaction.
        
        Default: None (unlimited - qubits can eventually interact).
        Override for platforms with limited interaction range.
        """
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, num_qubits={self.num_qubits})"


class ReconfigurableArchitecture(HardwareArchitecture):
    """Abstract base class for architectures with movable qubits.
    
    Reconfigurable architectures can physically transport qubits to change
    connectivity dynamically. This includes:
    - Trapped ion QCCD: ions moved through junction networks
    - Neutral atom tweezers: atoms repositioned by optical tweezers
    
    These architectures share common concepts:
    - Transport cost: time/fidelity penalty for moving qubits
    - Layout tracking: current physical positions of qubits
    - Batch reconfiguration: moving multiple qubits simultaneously
    - Routing: finding optimal paths for qubit movement
    
    Platform-specific implementations handle the actual transport mechanics:
    - QCCD: split/merge/shuttle operations with junction crossings
    - WISE: SAT-based optimal routing on grid architectures  
    - Tweezers: parallel moves with collision avoidance
    
    Attributes
    ----------
    qubit_positions : Dict[int, str]
        Maps logical qubit IDs to current zone IDs.
    """
    
    def __init__(
        self,
        name: str,
        num_qubits: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, num_qubits, metadata)
        # Track current position of each qubit
        self.qubit_positions: Dict[int, str] = {}
    
    def is_reconfigurable(self) -> bool:
        """Reconfigurable architectures always return True."""
        return True
    
    @abstractmethod
    def transport_cost(
        self,
        source_zone: str,
        target_zone: str,
        num_qubits: int = 1,
    ) -> "TransportCost":
        """Calculate the cost of transporting qubits between zones.
        
        Parameters
        ----------
        source_zone : str
            Starting zone ID.
        target_zone : str
            Destination zone ID.
        num_qubits : int
            Number of qubits being transported together.
            
        Returns
        -------
        TransportCost
            Cost metrics including time, fidelity loss, and operations.
        """
        ...
    
    @abstractmethod
    def reconfiguration_plan(
        self,
        target_layout: Dict[int, str],
    ) -> "ReconfigurationPlan":
        """Plan a batch reconfiguration to achieve target layout.
        
        Given the current qubit positions and a target layout, compute
        an optimal (or near-optimal) sequence of transport operations
        to achieve the target.
        
        Parameters
        ----------
        target_layout : Dict[int, str]
            Desired mapping of qubit IDs to zone IDs.
            
        Returns
        -------
        ReconfigurationPlan
            Ordered list of transport operations with timing.
        """
        ...
    
    @abstractmethod
    def can_colocate(self, qubit_ids: List[int], zone_id: str) -> bool:
        """Check if qubits can be placed together in a zone.
        
        Parameters
        ----------
        qubit_ids : List[int]
            Qubits to place together.
        zone_id : str
            Target zone.
            
        Returns
        -------
        bool
            True if zone has capacity and allows the operation.
        """
        ...
    
    @abstractmethod
    def current_zone(self, qubit_id: int) -> str:
        """Get the current zone of a qubit.
        
        Parameters
        ----------
        qubit_id : int
            Logical qubit ID.
            
        Returns
        -------
        str
            Zone ID where the qubit is currently located.
        """
        ...
    
    @abstractmethod
    def qubits_in_zone(self, zone_id: str) -> List[int]:
        """Get all qubits currently in a zone.
        
        Parameters
        ----------
        zone_id : str
            Zone ID to query.
            
        Returns
        -------
        List[int]
            List of qubit IDs in the zone.
        """
        ...
    
    def interaction_distance(self, qubit1: int, qubit2: int) -> int:
        """Get interaction distance via transport.
        
        For reconfigurable architectures, this is the number of
        transport steps needed to bring qubits together.
        """
        zone1 = self.current_zone(qubit1)
        zone2 = self.current_zone(qubit2)
        if zone1 == zone2:
            return 0
        # Use connectivity graph path length
        graph = self.connectivity_graph()
        try:
            path = graph.shortest_path(zone1, zone2)
            return len(path) - 1
        except nx.NetworkXNoPath:
            return -1  # No path exists


@dataclass
class TransportCost:
    """Cost metrics for a transport operation.
    
    Attributes
    ----------
    time_us : float
        Total time in microseconds.
    fidelity_loss : float
        Estimated fidelity reduction (1.0 - final_fidelity).
    num_operations : int
        Number of primitive operations (moves, splits, etc.).
    heating_added : float
        Motional excitation added (for trapped ions).
    path : List[str]
        Sequence of zones traversed.
    """
    time_us: float
    fidelity_loss: float
    num_operations: int
    heating_added: float = 0.0
    path: List[str] = field(default_factory=list)


@dataclass
class ReconfigurationPlan:
    """Plan for batch qubit reconfiguration.
    
    Attributes
    ----------
    operations : List[TransportOperation]
        Ordered sequence of transport operations.
    total_time_us : float
        Total time for full reconfiguration.
    total_fidelity_loss : float
        Cumulative fidelity loss.
    parallelism : int
        Maximum operations happening simultaneously.
    """
    operations: List[Any]  # Will be TransportOperation
    total_time_us: float
    total_fidelity_loss: float
    parallelism: int = 1


# Import NativeGateSet here to avoid circular imports
# (will be defined in gates.py)
from qectostim.experiments.hardware_simulation.core.gates import NativeGateSet
