# src/qectostim/experiments/hardware_simulation/trapped_ion/architecture.py
"""
Trapped ion architecture definitions.

Defines hardware architectures for trapped ion quantum computers.
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Any,
    Union,
)

import networkx as nx

from qectostim.experiments.hardware_simulation.core.architecture import (
    HardwareArchitecture,
    ConnectivityGraph,
    Zone,
    ZoneType,
    PhysicalConstraints,
)
from qectostim.experiments.hardware_simulation.core.gates import (
    NativeGateSet,
    GateSpec,
    GateType,
)
from qectostim.experiments.hardware_simulation.trapped_ion.gates import (
    TRAPPED_ION_NATIVE_GATES,
)


# =============================================================================
# QCCD Node Types
# =============================================================================

class QCCDOperationType(Enum):
    """Operations available in QCCD architectures."""
    SPLIT = auto()
    MOVE = auto()
    MERGE = auto()
    GATE_SWAP = auto()
    CRYSTAL_ROTATION = auto()
    ONE_QUBIT_GATE = auto()
    TWO_QUBIT_MS_GATE = auto()
    JUNCTION_CROSSING = auto()
    MEASUREMENT = auto()
    QUBIT_RESET = auto()
    RECOOLING = auto()
    PARALLEL = auto()
    GLOBAL_RECONFIG = auto()


@dataclass
class Ion:
    """Represents a single trapped ion in a QCCD system.
    
    Attributes
    ----------
    idx : int
        Unique identifier for this ion.
    position : Tuple[float, float]
        Current (x, y) position in the trap layout.
    label : str
        Display label (e.g., "Q0", "Q1").
    is_cooling : bool
        Whether this is a cooling ion (sympathetic cooling).
    motional_energy : float
        Current motional energy (heating accumulation).
    """
    idx: int
    position: Tuple[float, float] = (0.0, 0.0)
    label: str = "Q"
    is_cooling: bool = False
    motional_energy: float = 0.0
    
    @property
    def display_label(self) -> str:
        return f"{self.label}{self.idx}"
    
    def add_motional_energy(self, energy: float) -> None:
        """Add motional energy (heating)."""
        self.motional_energy += energy
    
    def reset_motional_energy(self) -> None:
        """Reset motional energy (after cooling)."""
        self.motional_energy = 0.0


@dataclass
class QCCDNode:
    """Base class for QCCD network nodes (traps, junctions).
    
    Attributes
    ----------
    idx : int
        Unique identifier for this node.
    position : Tuple[float, float]
        Position in the layout.
    capacity : int
        Maximum number of ions this node can hold.
    ions : List[Ion]
        Current ions in this node.
    """
    idx: int
    position: Tuple[float, float]
    capacity: int
    label: str = ""
    ions: List[Ion] = field(default_factory=list)
    
    @property
    def num_ions(self) -> int:
        return len(self.ions)
    
    @property
    def is_full(self) -> bool:
        return self.num_ions >= self.capacity
    
    @property
    def is_empty(self) -> bool:
        return self.num_ions == 0
    
    @property
    def display_label(self) -> str:
        return f"{self.label}{self.idx}"
    
    def add_ion(self, ion: Ion, position_idx: int = -1) -> None:
        """Add an ion to this node."""
        if self.is_full:
            raise ValueError(f"Node {self.idx} is at capacity {self.capacity}")
        if position_idx < 0:
            self.ions.append(ion)
        else:
            self.ions.insert(position_idx, ion)
    
    def remove_ion(self, ion: Optional[Ion] = None) -> Ion:
        """Remove and return an ion from this node."""
        if self.is_empty:
            raise ValueError(f"Node {self.idx} has no ions")
        if ion is None:
            return self.ions.pop(0)
        self.ions.remove(ion)
        return ion
    
    @property
    def allowed_operations(self) -> List[QCCDOperationType]:
        """Operations allowed at this node."""
        return []


@dataclass
class ManipulationTrap(QCCDNode):
    """Trap zone for gate operations.
    
    Manipulation traps are where quantum gates (1Q and 2Q) are performed.
    """
    is_horizontal: bool = True
    spacing: float = 1.0
    label: str = "MT"
    
    @property
    def allowed_operations(self) -> List[QCCDOperationType]:
        return [
            QCCDOperationType.SPLIT,
            QCCDOperationType.MERGE,
            QCCDOperationType.MOVE,
            QCCDOperationType.CRYSTAL_ROTATION,
            QCCDOperationType.GATE_SWAP,
            QCCDOperationType.ONE_QUBIT_GATE,
            QCCDOperationType.TWO_QUBIT_MS_GATE,
            QCCDOperationType.MEASUREMENT,
            QCCDOperationType.QUBIT_RESET,
            QCCDOperationType.RECOOLING,
        ]


@dataclass
class StorageTrap(QCCDNode):
    """Trap zone for storing idle ions."""
    is_horizontal: bool = True
    spacing: float = 1.0
    label: str = "ST"
    
    @property
    def allowed_operations(self) -> List[QCCDOperationType]:
        return [
            QCCDOperationType.SPLIT,
            QCCDOperationType.MERGE,
            QCCDOperationType.MOVE,
            QCCDOperationType.CRYSTAL_ROTATION,
        ]


@dataclass
class Junction(QCCDNode):
    """Junction node for routing between traps."""
    label: str = "J"
    
    @property
    def allowed_operations(self) -> List[QCCDOperationType]:
        return [
            QCCDOperationType.JUNCTION_CROSSING,
            QCCDOperationType.SPLIT,
            QCCDOperationType.MERGE,
            QCCDOperationType.MOVE,
        ]


@dataclass
class Crossing:
    """Represents an edge (crossing) between two QCCD nodes.
    
    Crossings are the paths ions take when shuttling between traps/junctions.
    """
    idx: int
    source: QCCDNode
    target: QCCDNode
    ion: Optional[Ion] = None  # Ion currently in transit
    label: str = "C"
    
    @property
    def position(self) -> Tuple[float, float]:
        """Midpoint position of the crossing."""
        x = (self.source.position[0] + self.target.position[0]) / 2
        y = (self.source.position[1] + self.target.position[1]) / 2
        return (x, y)
    
    @property
    def connection(self) -> Tuple[QCCDNode, QCCDNode]:
        return (self.source, self.target)
    
    @property
    def display_label(self) -> str:
        return f"{self.label}{self.idx}"


# =============================================================================
# QCCD Graph Builder
# =============================================================================

class QCCDGraph:
    """Graph representation of a QCCD architecture.
    
    Manages the network of traps, junctions, and crossings.
    Provides routing table computation and visualization.
    """
    
    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[int, QCCDNode] = {}
        self._crossings: Dict[int, Crossing] = {}
        self._crossing_edges: Dict[Tuple[int, int], Crossing] = {}
        self._next_idx: int = 0
        self._routing_table: Dict[int, Dict[int, List[int]]] = {}
    
    @property
    def graph(self) -> nx.DiGraph:
        """The underlying NetworkX directed graph."""
        return self._graph
    
    @property
    def nodes(self) -> Dict[int, QCCDNode]:
        """All nodes (traps and junctions) by index."""
        return self._nodes
    
    @property
    def crossings(self) -> Dict[int, Crossing]:
        """All crossings by index."""
        return self._crossings
    
    @property
    def ions(self) -> Dict[int, Ion]:
        """All ions across all nodes."""
        all_ions = {}
        for node in self._nodes.values():
            for ion in node.ions:
                all_ions[ion.idx] = ion
        for crossing in self._crossings.values():
            if crossing.ion is not None:
                all_ions[crossing.ion.idx] = crossing.ion
        return all_ions
    
    def add_manipulation_trap(
        self,
        position: Tuple[float, float],
        ions: List[Ion],
        capacity: int = 4,
        is_horizontal: bool = True,
        spacing: float = 1.0,
    ) -> ManipulationTrap:
        """Add a manipulation trap to the graph."""
        trap = ManipulationTrap(
            idx=self._next_idx,
            position=position,
            capacity=capacity,
            ions=ions,
            is_horizontal=is_horizontal,
            spacing=spacing,
        )
        self._nodes[trap.idx] = trap
        self._next_idx += 1
        return trap
    
    def add_storage_trap(
        self,
        position: Tuple[float, float],
        ions: List[Ion],
        capacity: int = 8,
        is_horizontal: bool = True,
        spacing: float = 1.0,
    ) -> StorageTrap:
        """Add a storage trap to the graph."""
        trap = StorageTrap(
            idx=self._next_idx,
            position=position,
            capacity=capacity,
            ions=ions,
            is_horizontal=is_horizontal,
            spacing=spacing,
        )
        self._nodes[trap.idx] = trap
        self._next_idx += 1
        return trap
    
    def add_junction(
        self,
        position: Tuple[float, float],
        capacity: int = 1,
    ) -> Junction:
        """Add a junction to the graph."""
        junction = Junction(
            idx=self._next_idx,
            position=position,
            capacity=capacity,
        )
        self._nodes[junction.idx] = junction
        self._next_idx += 1
        return junction
    
    def add_crossing(
        self,
        source: QCCDNode,
        target: QCCDNode,
    ) -> Crossing:
        """Add a crossing (edge) between two nodes."""
        crossing = Crossing(
            idx=self._next_idx,
            source=source,
            target=target,
        )
        self._crossings[crossing.idx] = crossing
        self._crossing_edges[(source.idx, target.idx)] = crossing
        self._crossing_edges[(target.idx, source.idx)] = crossing
        self._next_idx += 1
        return crossing
    
    def build_networkx_graph(self) -> nx.DiGraph:
        """Build/refresh the NetworkX graph from nodes and crossings."""
        g = nx.DiGraph()
        
        # Add nodes
        for node in self._nodes.values():
            g.add_node(node.idx, pos=node.position, node=node)
        
        # Add edges from crossings
        for crossing in self._crossings.values():
            src, tgt = crossing.source.idx, crossing.target.idx
            g.add_edge(src, tgt, crossing=crossing, weight=1)
            g.add_edge(tgt, src, crossing=crossing, weight=1)
        
        # Add intra-trap edges (ions within same trap can interact)
        for node in self._nodes.values():
            if isinstance(node, (ManipulationTrap, StorageTrap)):
                for i, ion1 in enumerate(node.ions):
                    for j, ion2 in enumerate(node.ions):
                        if i != j:
                            # Weight 0 for same-trap interactions
                            g.add_edge(ion1.idx, ion2.idx, weight=0, same_trap=True)
        
        self._graph = g
        return g
    
    def compute_routing_table(self) -> Dict[int, Dict[int, List[int]]]:
        """Compute shortest paths between all node pairs."""
        if not self._graph:
            self.build_networkx_graph()
        
        self._routing_table = {}
        node_indices = list(self._nodes.keys())
        
        for source in node_indices:
            self._routing_table[source] = {}
            for target in node_indices:
                if source != target:
                    try:
                        path = nx.shortest_path(
                            self._graph, source, target, weight='weight'
                        )
                        self._routing_table[source][target] = path
                    except nx.NetworkXNoPath:
                        self._routing_table[source][target] = []
        
        return self._routing_table
    
    def get_routing_path(self, source: int, target: int) -> List[int]:
        """Get routing path between two nodes."""
        if not self._routing_table:
            self.compute_routing_table()
        return self._routing_table.get(source, {}).get(target, [])
    
    def routing_distance(self, source: int, target: int) -> int:
        """Get routing distance (number of hops) between nodes."""
        path = self.get_routing_path(source, target)
        return len(path) - 1 if path else -1


class TrappedIonArchitecture(HardwareArchitecture):
    """Abstract base class for trapped ion architectures.
    
    Common properties:
    - Native MS (Mølmer-Sørensen) entangling gate
    - All-to-all connectivity within chains
    - Single-qubit rotations via focused lasers
    
    Subclasses define specific trap geometries:
    - QCCDArchitecture: Multi-zone with shuttling
    - LinearChainArchitecture: Single linear trap
    """
    
    def __init__(
        self,
        name: str,
        num_qubits: int,
        ms_gate_time: float = 40.0,  # μs
        single_qubit_time: float = 5.0,  # μs
        measurement_time: float = 400.0,  # μs
        t2_time: float = 2.2e6,  # μs (2.2 seconds)
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, num_qubits, metadata)
        self.ms_gate_time = ms_gate_time
        self.single_qubit_time = single_qubit_time
        self.measurement_time = measurement_time
        self.t2_time = t2_time
    
    def native_gate_set(self) -> NativeGateSet:
        """Trapped ion native gates: MS + single-qubit rotations."""
        gate_set = NativeGateSet("trapped_ion")
        
        # Add trapped ion native gates
        for name, spec in TRAPPED_ION_NATIVE_GATES.items():
            gate_set.add_gate(spec)
        
        # Standard single-qubit Clifford gates (implemented via rotations)
        gate_set.add_gate(GateSpec("H", GateType.SINGLE_QUBIT, 1, is_clifford=True))
        gate_set.add_gate(GateSpec("S", GateType.SINGLE_QUBIT, 1, is_clifford=True))
        gate_set.add_gate(GateSpec("X", GateType.SINGLE_QUBIT, 1, is_clifford=True))
        gate_set.add_gate(GateSpec("Y", GateType.SINGLE_QUBIT, 1, is_clifford=True))
        gate_set.add_gate(GateSpec("Z", GateType.SINGLE_QUBIT, 1, is_clifford=True))
        
        return gate_set
    
    def physical_constraints(self) -> PhysicalConstraints:
        """Physical constraints for trapped ions."""
        return PhysicalConstraints(
            max_qubits=self.num_qubits,
            t2_time=self.t2_time,
            gate_times={
                "MS": self.ms_gate_time,
                "R": self.single_qubit_time,
                "H": self.single_qubit_time,
                "S": self.single_qubit_time,
                "X": self.single_qubit_time,
                "Y": self.single_qubit_time,
                "Z": self.single_qubit_time,
            },
            readout_time=self.measurement_time,
            reset_time=50.0,  # μs
        )
    
    def is_reconfigurable(self) -> bool:
        """Trapped ions support reconfiguration via shuttling."""
        return True


class QCCDArchitecture(TrappedIonArchitecture):
    """QCCD (Quantum Charge-Coupled Device) architecture.
    
    Multi-zone trap network where ions are shuttled between zones.
    Uses QCCDGraph for representing the trap network.
    
    Features:
    - Manipulation zones for gate operations
    - Storage zones for holding idle ions
    - Junctions for routing between zones
    - Ion transport via split/merge/shuttle operations
    
    Parameters
    ----------
    rows : int
        Number of rows in the trap grid.
    cols : int
        Number of columns in the trap grid.
    ions_per_trap : int
        Maximum ions per manipulation trap.
    trap_spacing : float
        Distance between traps in the grid.
    """
    
    def __init__(
        self,
        rows: int = 2,
        cols: int = 2,
        ions_per_trap: int = 4,
        trap_spacing: float = 10.0,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        num_qubits = rows * cols * ions_per_trap
        super().__init__(
            name=name or f"QCCD_{rows}x{cols}",
            num_qubits=num_qubits,
            metadata=metadata,
        )
        self.rows = rows
        self.cols = cols
        self.ions_per_trap = ions_per_trap
        self.trap_spacing = trap_spacing
        
        # Build the QCCD graph
        self._qccd_graph = QCCDGraph()
        self._build_grid_topology()
    
    def _build_grid_topology(self) -> None:
        """Build a grid-based QCCD topology.
        
        Creates a grid of manipulation traps connected via junctions.
        """
        # Create ions
        ion_idx = 0
        trap_grid: List[List[ManipulationTrap]] = []
        junction_grid: List[List[Optional[Junction]]] = []
        
        # Create manipulation traps at grid positions
        for r in range(self.rows):
            trap_row = []
            for c in range(self.cols):
                # Create ions for this trap
                ions = [
                    Ion(idx=ion_idx + i, label="Q")
                    for i in range(self.ions_per_trap)
                ]
                ion_idx += self.ions_per_trap
                
                # Position traps at grid locations with spacing for junctions
                x = c * self.trap_spacing * 2
                y = r * self.trap_spacing * 2
                
                trap = self._qccd_graph.add_manipulation_trap(
                    position=(x, y),
                    ions=ions,
                    capacity=self.ions_per_trap,
                    is_horizontal=(r % 2 == 0),  # Alternate orientation
                )
                trap_row.append(trap)
            trap_grid.append(trap_row)
        
        # Create junctions between traps
        for r in range(self.rows):
            junction_row = []
            for c in range(self.cols):
                # Junction to the right
                if c < self.cols - 1:
                    jx = (c * 2 + 1) * self.trap_spacing
                    jy = r * self.trap_spacing * 2
                    j_right = self._qccd_graph.add_junction(position=(jx, jy))
                    
                    # Connect trap to junction to next trap
                    self._qccd_graph.add_crossing(trap_grid[r][c], j_right)
                    self._qccd_graph.add_crossing(j_right, trap_grid[r][c + 1])
                
                # Junction below
                if r < self.rows - 1:
                    jx = c * self.trap_spacing * 2
                    jy = (r * 2 + 1) * self.trap_spacing
                    j_below = self._qccd_graph.add_junction(position=(jx, jy))
                    
                    # Connect trap to junction to trap below
                    self._qccd_graph.add_crossing(trap_grid[r][c], j_below)
                    self._qccd_graph.add_crossing(j_below, trap_grid[r + 1][c])
            
            junction_row.append(j_right if c < self.cols - 1 else None)  # type: ignore
            junction_grid.append(junction_row)
        
        # Build the NetworkX graph
        self._qccd_graph.build_networkx_graph()
        self._qccd_graph.compute_routing_table()
    
    @property
    def qccd_graph(self) -> QCCDGraph:
        """The underlying QCCD graph."""
        return self._qccd_graph
    
    def connectivity_graph(self) -> ConnectivityGraph:
        """Build QCCD connectivity graph."""
        # Create connectivity from the QCCD graph
        graph = ConnectivityGraph()
        
        # Add all qubits
        for i in range(self.num_qubits):
            graph.add_node(i)
        
        # Within each trap, all ions can interact
        for node in self._qccd_graph.nodes.values():
            if isinstance(node, ManipulationTrap):
                ions = node.ions
                for i, ion1 in enumerate(ions):
                    for j, ion2 in enumerate(ions):
                        if i < j:
                            graph.add_edge(ion1.idx, ion2.idx)
        
        return graph
    
    def zone_types(self) -> List[ZoneType]:
        """Zone types in QCCD architecture."""
        return [
            ZoneType.MANIPULATION,
            ZoneType.STORAGE,
            ZoneType.JUNCTION,
            ZoneType.CROSSING,
        ]
    
    def can_interact(self, qubit1: int, qubit2: int) -> bool:
        """Check if qubits can interact (same trap or via shuttling).
        
        In QCCD, any pair can eventually interact via shuttling.
        """
        if not (0 <= qubit1 < self.num_qubits and 0 <= qubit2 < self.num_qubits):
            return False
        
        # Check if in same trap (direct interaction)
        for node in self._qccd_graph.nodes.values():
            ion_indices = [ion.idx for ion in node.ions]
            if qubit1 in ion_indices and qubit2 in ion_indices:
                return True
        
        # Any pair can interact via shuttling
        return True
    
    def interaction_distance(self, qubit1: int, qubit2: int) -> int:
        """Get routing distance between qubits.
        
        Returns:
            0 if in same trap
            Number of hops if reachable via shuttling
            -1 if invalid qubits
        """
        if not (0 <= qubit1 < self.num_qubits and 0 <= qubit2 < self.num_qubits):
            return -1
        
        # Find which traps contain the qubits
        trap1, trap2 = None, None
        for node in self._qccd_graph.nodes.values():
            ion_indices = [ion.idx for ion in node.ions]
            if qubit1 in ion_indices:
                trap1 = node
            if qubit2 in ion_indices:
                trap2 = node
        
        if trap1 is None or trap2 is None:
            return -1
        
        if trap1 == trap2:
            return 0  # Same trap
        
        # Get routing distance between traps
        return self._qccd_graph.routing_distance(trap1.idx, trap2.idx)
    
    def get_trap_for_qubit(self, qubit: int) -> Optional[QCCDNode]:
        """Get the trap containing a specific qubit."""
        for node in self._qccd_graph.nodes.values():
            ion_indices = [ion.idx for ion in node.ions]
            if qubit in ion_indices:
                return node
        return None


class LinearChainArchitecture(TrappedIonArchitecture):
    """Linear ion chain architecture.
    
    Single linear trap with all-to-all connectivity within the chain.
    Simpler than QCCD but limited scalability.
    
    Parameters
    ----------
    num_ions : int
        Number of ions in the chain.
    """
    
    def __init__(
        self,
        num_ions: int,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name or f"LinearChain_{num_ions}",
            num_qubits=num_ions,
            metadata=metadata,
        )
        self._num_ions = num_ions
    
    def connectivity_graph(self) -> ConnectivityGraph:
        """Build linear chain connectivity (single computation zone).
        
        Linear chain is a single trap zone with all-to-all connectivity.
        """
        from ..core.architecture import Zone
        
        graph = ConnectivityGraph()
        # Single computation zone holding all ions
        zone = Zone(
            id="chain",
            zone_type=ZoneType.COMPUTATION,
            capacity=self._num_ions,
            position=(0.0,),
            allowed_operations=frozenset({
                "R", "Rz", "MS", "XX", "M",  # Trapped ion gates
            }),
        )
        graph.add_zone(zone)
        return graph
    
    def zone_types(self) -> List[ZoneType]:
        """Single computation zone for linear chain."""
        return [ZoneType.COMPUTATION]
    
    def can_interact(self, qubit1: int, qubit2: int) -> bool:
        """All ions in chain can directly interact."""
        return 0 <= qubit1 < self.num_qubits and 0 <= qubit2 < self.num_qubits
    
    def interaction_distance(self, qubit1: int, qubit2: int) -> int:
        """All-to-all: distance is always 0."""
        if self.can_interact(qubit1, qubit2):
            return 0
        return -1  # Invalid qubits
