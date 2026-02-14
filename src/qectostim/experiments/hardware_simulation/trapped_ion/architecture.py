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
from qectostim.experiments.hardware_simulation.core.components import (
    PhysicalQubit,
    QubitState,
)
from qectostim.experiments.hardware_simulation.core.gates import (
    NativeGateSet,
    GateSpec,
    GateType,
)
# TRAPPED_ION_NATIVE_GATES is imported lazily in native_gate_set()
# to avoid circular import (operations.py imports architecture.py).
from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
    ModeSnapshot,
    ModeStructure,
    DEFAULT_CALIBRATION as _CAL,
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


# Alias for backward compatibility with old routing code
Operations = QCCDOperationType


@dataclass
class Ion(PhysicalQubit):
    """Represents a single trapped ion in a QCCD system.
    
    Inherits from core.components.PhysicalQubit with trapped-ion specific
    extensions for motional energy tracking and cooling ion support.
    
    Attributes
    ----------
    idx : int
        Unique identifier for this ion (alias for index).
    position : Tuple[float, float]
        Current (x, y) position in the trap layout.
    label : str
        Display label (e.g., "Q0", "Q1").
    is_cooling : bool
        Whether this is a cooling ion (sympathetic cooling).
    motional_energy : float
        Current motional energy (heating accumulation).
        This is the key trapped-ion specific extension.
    parent : Optional["QCCDNode"]
        The trap/node currently containing this ion.
        Set automatically by QCCDNode.add_ion() / remove_ion().
    """
    # Trapped-ion specific fields
    label: str = "Q"
    is_cooling: bool = False
    motional_energy: float = 0.0
    parent: Optional["QCCDNode"] = None
    _color: str = "lightblue"
    
    def __init__(
        self,
        idx: int,
        position: Tuple[float, float] = (0.0, 0.0),
        label: str = "Q",
        is_cooling: bool = False,
        motional_energy: float = 0.0,
        parent: Optional["QCCDNode"] = None,
        metadata: Optional[Dict[str, Any]] = None,
        color: str = "lightblue",
    ):
        # Initialize parent PhysicalQubit
        super().__init__(
            id=f"{label}{idx}",
            index=idx,
            position=position,
            metadata=metadata,
        )
        self.label = label
        self.is_cooling = is_cooling
        self.motional_energy = motional_energy
        self.parent = parent
        self._color = color
    
    @property
    def idx(self) -> int:
        """Alias for index (trapped-ion convention)."""
        return self.index
    
    @property
    def pos(self) -> Tuple[float, float]:
        """Alias for position (old API compatibility)."""
        return self.position
    
    @property
    def color(self) -> str:
        """Ion color for visualization."""
        return self._color
    
    @property
    def display_label(self) -> str:
        return f"{self.label}{self.idx}"
    
    @property
    def allowedOperations(self) -> List[QCCDOperationType]:
        """Operations allowed for this ion (old API compatibility)."""
        return [
            QCCDOperationType.CRYSTAL_ROTATION,
            QCCDOperationType.GATE_SWAP,
            QCCDOperationType.JUNCTION_CROSSING,
            QCCDOperationType.MERGE,
            QCCDOperationType.SPLIT,
            QCCDOperationType.MOVE,
            QCCDOperationType.RECOOLING,
            QCCDOperationType.MEASUREMENT,
            QCCDOperationType.ONE_QUBIT_GATE,
            QCCDOperationType.TWO_QUBIT_MS_GATE,
            QCCDOperationType.QUBIT_RESET,
        ]
    
    def set(
        self,
        idx: int,
        x: float,
        y: float,
        parent: Optional["QCCDNode"] = None,
    ) -> None:
        """Set ion position and parent (old API compatibility)."""
        self.index = idx
        self.id = f"{self.label}{idx}"
        self.position = (float(x), float(y))
        self.parent = parent
    
    @property
    def motionalMode(self) -> float:
        """Alias for motional_energy (old API compatibility)."""
        return self.motional_energy
    
    def add_motional_energy(self, energy: float) -> None:
        """Add motional energy (heating)."""
        self.motional_energy += energy
    
    def addMotionalEnergy(self, energy: float) -> None:
        """Old API compatibility alias."""
        self.add_motional_energy(energy)
    
    def reset_motional_energy(self) -> None:
        """Reset motional energy (after cooling)."""
        self.motional_energy = 0.0
    
    def reset(self) -> None:
        """Reset ion to initial state including motional energy."""
        super().reset()
        self.motional_energy = 0.0
    
    def __hash__(self) -> int:
        """Hash based on unique ion index for use as dict keys."""
        return hash(self.idx)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on ion index."""
        if isinstance(other, Ion):
            return self.idx == other.idx
        return False


# =============================================================================
# Ion Type Aliases and Factory Functions
# =============================================================================

# Type aliases for different ion roles
QubitIon = Ion  # Standard qubit ion
CoolingIon = Ion  # Sympathetic cooling ion (is_cooling=True)
SpectatorIon = Ion  # Spectator ion (for state readout)


def create_qubit_ion(
    idx: int,
    position: Tuple[float, float] = (0.0, 0.0),
    label: str = "Q",
) -> Ion:
    """Create a qubit ion."""
    return Ion(idx=idx, position=position, label=label, is_cooling=False)


def create_cooling_ion(
    idx: int,
    position: Tuple[float, float] = (0.0, 0.0),
    label: str = "C",
) -> Ion:
    """Create a cooling ion for sympathetic cooling."""
    return Ion(idx=idx, position=position, label=label, is_cooling=True)


# =============================================================================
# Abstract Base for QCCD Components (Old API Compatibility)
# =============================================================================

class QCCDComponent:
    """Abstract base class for QCCD components (old API compatibility).
    
    Provides the interface expected by old routing code.
    """
    
    @property
    def pos(self) -> Tuple[float, float]:
        """Component position."""
        raise NotImplementedError
    
    @property
    def idx(self) -> int:
        """Component index."""
        raise NotImplementedError
    
    @property
    def allowedOperations(self) -> Sequence[QCCDOperationType]:
        """Allowed operations for this component."""
        return []


# =============================================================================
# Architecture Configuration Dataclasses
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


# Old API alias
QCCDWiseArch = QCCDWISEConfig


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
# QCCD Network Nodes
# =============================================================================

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
    _color: str = "lightgray"
    
    # Mutable counter for routing (old API compatibility)
    numIons: int = field(default=0, repr=False)
    
    def __post_init__(self):
        self.numIons = len(self.ions)
    
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
    
    @property
    def pos(self) -> Tuple[float, float]:
        """Alias for position (old API compatibility)."""
        return self.position
    
    @property
    def color(self) -> str:
        """Node color for visualization."""
        return self._color
    
    @property
    def nodes(self) -> List[int]:
        """List of node indices (self + ions) for old API compatibility."""
        n = [self.idx]
        if self.ions:
            n += [i.idx for i in self.ions]
        return n
    
    @property
    def positions(self) -> List[Tuple[float, float]]:
        """List of positions (self + ions) for old API compatibility."""
        n = [self.pos]
        if self.ions:
            n += [i.pos for i in self.ions]
        return n
    
    def add_ion(self, ion: Ion, position_idx: int = -1) -> None:
        """Add an ion to this node.

        ``numIons`` is **not** auto-updated here — the greedy routing
        algorithm manages the counter manually via direct assignment.
        """
        if position_idx < 0:
            self.ions.append(ion)
        else:
            self.ions.insert(position_idx, ion)
        ion.parent = self
    
    def addIon(
        self, ion: Ion, adjacentIon: Optional[Ion] = None, offset: int = 0
    ) -> None:
        """Old API compatibility: add ion next to adjacentIon."""
        pos_idx = (self.ions.index(adjacentIon) + offset if adjacentIon else offset)
        self.ions.insert(pos_idx, ion)
        ion.set(ion.idx, *self.pos, parent=self)
    
    def remove_ion(self, ion: Optional[Ion] = None) -> Ion:
        """Remove and return an ion from this node."""
        if self.is_empty:
            raise ValueError(f"Node {self.idx} has no ions")
        if ion is None:
            removed = self.ions.pop(0)
        else:
            self.ions.remove(ion)
            removed = ion
        removed.parent = None
        return removed
    
    def removeIon(self, ion: Optional[Ion] = None) -> Ion:
        """Old API compatibility alias."""
        return self.remove_ion(ion)
    
    @property
    def allowed_operations(self) -> List[QCCDOperationType]:
        """Operations allowed at this node."""
        return []
    
    @property
    def allowedOperations(self) -> Sequence[QCCDOperationType]:
        """Old API compatibility alias."""
        return self.allowed_operations
    
    def subgraph(self, graph: "nx.Graph") -> "nx.Graph":
        """Add this node to a networkx graph (old API compatibility)."""
        for n, p in zip(self.nodes, self.positions):
            graph.add_node(n, pos=p)
        return graph
    
    def addMotionalEnergy(self, energy: float) -> None:
        """Add motional energy to all ions (old API compatibility)."""
        if self.ions:
            energy_per_ion = energy / len(self.ions)
            for ion in self.ions:
                ion.addMotionalEnergy(energy_per_ion)
    
    @property
    def motionalMode(self) -> float:
        """Total motional energy of non-cooling ions (old API compatibility)."""
        return sum(
            ion.motionalMode for ion in self.ions
            if not getattr(ion, 'is_cooling', False)
        )
    
    @property
    def motional_energy(self) -> float:
        """Total motional energy of ions in this node (alias for motionalMode)."""
        return self.motionalMode
    
    def __hash__(self) -> int:
        """Hash based on unique node index for use as dict keys/set members."""
        return hash(self.idx)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on node index."""
        if isinstance(other, QCCDNode):
            return self.idx == other.idx
        return False


@dataclass
class ManipulationTrap(QCCDNode):
    """Trap zone for gate operations.
    
    Manipulation traps are where quantum gates (1Q and 2Q) are performed.
    When ions are added or removed, the 3N normal-mode structure is
    automatically recomputed (see :class:`ModeStructure` in physics.py).

    Parameters
    ----------
    secular_frequencies : Tuple[float, float, float]
        Secular trap frequencies (ω_z, ω_x, ω_y) in Hz.
        Default (1 MHz, 5 MHz, 5 MHz).
    """
    is_horizontal: bool = True
    spacing: float = 1.0
    label: str = "MT"
    secular_frequencies: Tuple[float, float, float] = (1.0e6, 5.0e6, 5.0e6)
    _color: str = "lightyellow"
    
    # Constants for old API compatibility (from physics.py)
    BACKGROUND_HEATING_RATE: float = _CAL.heating_rate
    CAPACITY_SCALING: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.mode_structure: Optional[ModeStructure] = None
        self._desired_capacity = self.capacity
        self._spacing = self.spacing
        self._isHorizontal = self.is_horizontal
        self._arrangeIons()

    # ------ Override add_ion / remove_ion to recompute modes ------

    def add_ion(self, ion: Ion, position_idx: int = -1) -> None:
        """Add an ion and recompute the 3N mode structure.

        ``numIons`` is not updated — managed by routing.
        """
        if position_idx < 0:
            self.ions.append(ion)
        else:
            self.ions.insert(position_idx, ion)
        ion.parent = self
        self._arrangeIons()
        self._recompute_modes()

    def addIon(
        self, ion: Ion, adjacentIon: Optional[Ion] = None, offset: int = 0
    ) -> None:
        """Old API compatibility: add ion next to adjacentIon."""
        pos_idx = (self.ions.index(adjacentIon) + offset if adjacentIon else offset)
        self.ions.insert(pos_idx, ion)
        ion.set(ion.idx, *self.pos, parent=self)
        self._arrangeIons()
        self._recompute_modes()

    def remove_ion(self, ion: Optional[Ion] = None) -> Ion:
        """Remove an ion and recompute the 3N mode structure."""
        if self.is_empty:
            raise ValueError(f"Node {self.idx} has no ions")
        if ion is None:
            removed = self.ions.pop(0)
        else:
            self.ions.remove(ion)
            removed = ion
        removed.parent = None
        self._arrangeIons()
        self._recompute_modes()
        return removed

    def removeIon(self, ion: Optional[Ion] = None) -> Ion:
        """Old API compatibility alias."""
        return self.remove_ion(ion)

    def _arrangeIons(self) -> None:
        """Arrange ion positions within the trap (old API compatibility)."""
        for i, ion in enumerate(self.ions):
            o = i - len(self.ions) / 2
            x = self.pos[0] + o * self._spacing * self._isHorizontal
            y = self.pos[1] + o * self._spacing * (1 - self._isHorizontal)
            ion.set(ion.idx, x, y, parent=self)

    def _recompute_modes(self) -> None:
        """Recompute mode structure for the current ion crystal."""
        n = len(self.ions)
        if n == 0:
            self.mode_structure = None
            return
        wz, wx, wy = self.secular_frequencies
        self.mode_structure = ModeStructure.compute(
            n, axial_freq=wz, radial_freqs=(wx, wy),
        )

    @property
    def backgroundHeatingRate(self) -> float:
        """Background heating rate (old API compatibility)."""
        return self.BACKGROUND_HEATING_RATE

    @property
    def desiredCapacity(self) -> int:
        """Desired capacity (old API compatibility)."""
        return self._desired_capacity

    @property
    def hasCoolingIon(self) -> bool:
        """Check if trap has a cooling ion (old API compatibility)."""
        return any(getattr(ion, 'is_cooling', False) for ion in self.ions)

    def coolTrap(self) -> bool:
        """Old API compatibility: reset motional energy via cooling ions."""
        cooling_ions = [ion for ion in self.ions if getattr(ion, 'is_cooling', False)]
        qubit_ions = [ion for ion in self.ions if not getattr(ion, 'is_cooling', False)]
        if not cooling_ions:
            return False
        energy = 0.0
        for ion in qubit_ions:
            energy += ion.motionalMode
            ion.motional_energy = 0.0
        energy_per_cooling = energy / len(cooling_ions)
        for ion in cooling_ions:
            ion.addMotionalEnergy(energy_per_cooling)
        return True

    def cool_trap(self) -> None:
        """Sympathetic recooling: reset all mode occupancies to zero.

        Only performs cooling if a cooling ion is present in the trap.
        """
        has_cooling = any(getattr(ion, 'is_cooling', False) for ion in self.ions)
        if has_cooling and self.mode_structure is not None:
            self.mode_structure.cool_to_ground()
    
    def __hash__(self) -> int:
        """Hash based on unique node index for use as dict keys/set members."""
        return hash(self.idx)
    
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


# Old API alias
Trap = ManipulationTrap


@dataclass
class StorageTrap(QCCDNode):
    """Trap zone for storing idle ions."""
    is_horizontal: bool = True
    spacing: float = 1.0
    label: str = "ST"
    _color: str = "grey"
    
    # Constants for old API compatibility (from physics.py)
    BACKGROUND_HEATING_RATE: float = _CAL.heating_rate
    CAPACITY_SCALING: int = 1
    
    def __post_init__(self):
        super().__post_init__()
        self._desired_capacity = self.capacity
        self._spacing = self.spacing
        self._isHorizontal = self.is_horizontal
    
    @property
    def backgroundHeatingRate(self) -> float:
        """Background heating rate (old API compatibility)."""
        return self.BACKGROUND_HEATING_RATE
    
    @property
    def allowed_operations(self) -> List[QCCDOperationType]:
        return [
            QCCDOperationType.SPLIT,
            QCCDOperationType.MERGE,
            QCCDOperationType.MOVE,
            QCCDOperationType.CRYSTAL_ROTATION,
        ]
    
    def __hash__(self) -> int:
        """Hash based on unique node index for use as dict keys/set members."""
        return hash(self.idx)


@dataclass
class Junction(QCCDNode):
    """Junction node for routing between traps."""
    label: str = "J"
    _color: str = "orange"
    
    # Default constants for old API compatibility
    DEFAULT_COLOR: str = "orange"
    DEFAULT_LABEL: str = "J"
    DEFAULT_CAPACITY: int = 1
    
    def __post_init__(self):
        super().__post_init__()
    
    @property
    def allowed_operations(self) -> List[QCCDOperationType]:
        return [
            QCCDOperationType.JUNCTION_CROSSING,
            QCCDOperationType.SPLIT,
            QCCDOperationType.MERGE,
            QCCDOperationType.MOVE,
        ]
    
    def __hash__(self) -> int:
        """Hash based on unique node index for use as dict keys/set members."""
        return hash(self.idx)


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
    _ionAtSource: bool = False
    
    # Constants for old API compatibility
    DEFAULT_LABEL: str = "C"
    MOVE_AMOUNT: int = 8
    
    @property
    def pos(self) -> Tuple[float, float]:
        """Alias for position (old API compatibility)."""
        return self.position
    
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
    
    @property
    def allowedOperations(self) -> Sequence[QCCDOperationType]:
        """Allowed operations (old API compatibility)."""
        return [
            QCCDOperationType.SPLIT,
            QCCDOperationType.MOVE,
            QCCDOperationType.MERGE,
            QCCDOperationType.JUNCTION_CROSSING,
        ]
    
    def ionAt(self) -> QCCDNode:
        """Return node where ion currently is (old API compatibility)."""
        if not self.ion:
            raise ValueError(f"ionAt: no ion for crossing {self.idx}")
        return self.source if self._ionAtSource else self.target
    
    def hasTrap(self, trap: "Trap") -> bool:
        """Check if this crossing connects to a trap (old API compatibility)."""
        return trap == self.source or trap == self.target
    
    def hasJunction(self, junction: Junction) -> bool:
        """Check if this crossing connects to a junction (old API compatibility)."""
        return junction == self.source or junction == self.target
    
    def _getEdgeIdxs(self) -> Tuple[int, int]:
        """Get edge indices for visualization (old API compatibility)."""
        import numpy as np
        permutations = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        if not self.source.ions:
            permutations[0][0] = 0
            permutations[1][0] = 0
        if not self.target.ions:
            permutations[0][1] = 0
            permutations[2][1] = 0
        pairs_distances = [
            (self.source.positions[i][0] - self.target.positions[j][0]) ** 2
            + (self.source.positions[i][1] - self.target.positions[j][1]) ** 2
            for (i, j) in permutations
        ]
        return permutations[np.argmin(pairs_distances)]
    
    def graphEdge(self) -> Tuple[int, int]:
        """Get graph edge tuple (old API compatibility)."""
        idx1, idx2 = self._getEdgeIdxs()
        return (self.source.nodes[idx1], self.target.nodes[idx2])
    
    def getEdgeIon(self, node: QCCDNode) -> Ion:
        """Get edge ion for a node (old API compatibility)."""
        if not node.ions:
            raise ValueError("getEdgeIon: no edge ions")
        ionIdx = node.nodes[self._getEdgeIdxs()[1 - (node == self.source)]]
        return [i for i in node.ions if i.idx == ionIdx][0]
    
    def setIon(self, ion: Ion, node: QCCDNode) -> None:
        """Set ion in crossing (old API compatibility)."""
        if self.ion is not None:
            raise ValueError(f"setIon: crossing has not been cleared")
        self.ion = ion
        self._ionAtSource = (self.source == node)
        edgeIdxs = self._getEdgeIdxs()
        w = (1 + (self.MOVE_AMOUNT - 2) * (self.source == node)) / self.MOVE_AMOUNT
        x = self.source.positions[edgeIdxs[0]][0] * w + self.target.positions[
            edgeIdxs[1]
        ][0] * (1 - w)
        y = self.source.positions[edgeIdxs[0]][1] * w + self.target.positions[
            edgeIdxs[1]
        ][1] * (1 - w)
        self.ion.set(self.ion.idx, x, y, parent=self)
    
    def moveIon(self) -> None:
        """Move ion to other side of crossing (old API compatibility)."""
        if self.ion is None:
            raise ValueError(f"moveIon: no ion to move in crossing")
        node = self.target if self._ionAtSource else self.source
        ion = self.ion
        self.clearIon()
        self.setIon(ion, node)
    
    def clearIon(self) -> None:
        """Clear ion from crossing (old API compatibility)."""
        self.ion = None
    
    def __hash__(self) -> int:
        """Hash based on unique crossing index for use as dict keys/set members."""
        return hash(self.idx)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on crossing index."""
        if isinstance(other, Crossing):
            return self.idx == other.idx
        return False


# =============================================================================
# Lightweight edge-operation helpers (used by QCCDGraph.build_networkx_graph)
# =============================================================================

class _EdgeOp:
    """Minimal transport operation for graph-edge routing.

    ``greedy_routing.ionRouting`` reads ``graph.edges[n1,n2]["operations"]``
    and calls ``.run()`` on each entry.  This class provides just enough
    interface to satisfy that loop (and the ``isinstance`` checks that
    follow it).  It is intentionally **not** a subclass of the legacy
    ``routing.reconfiguration.Operation`` to avoid circular imports.
    """

    def __init__(
        self,
        run_fn,
        label: str = "transport",
        involved_components=None,
    ):
        self._run_fn = run_fn
        self._label_str = label
        self._involvedComponents = list(involved_components or [])
        self._involvedIonsForLabel: list = []
        self._addOns = ""
        self._fidelity: float = 1.0
        self._dephasingFidelity: float = 1.0
        self._operationTime: float = 0.0

    # ------ greedy_routing interface ------
    def run(self) -> None:  # noqa: D401
        self._run_fn()

    @property
    def label(self) -> str:
        return self._label_str + self._addOns

    @property
    def involvedComponents(self):
        return self._involvedComponents

    @property
    def involvedIonsForLabel(self):
        return self._involvedIonsForLabel

    # Timing stubs (used by Operation base but not by greedy loop)
    def calculateOperationTime(self) -> None:
        pass

    def calculateFidelity(self) -> None:
        pass

    def calculateDephasingFidelity(self) -> None:
        pass

    def _generateLabelAddOns(self) -> None:
        pass


def _make_crossing_ops(
    src_node: "QCCDNode",
    tgt_node: "QCCDNode",
    crossing: "Crossing",
):
    """Build directed transport ``_EdgeOp`` list for one crossing edge.

    The operation sequence depends on the types of *src_node* and
    *tgt_node*, faithfully matching the old ``QCCDArch.addEdge()``
    logic:

    * **Trap → Junction**: Split + Move + JunctionCrossing
    * **Junction → Trap**: JunctionCrossing + Move + Merge
    * **Junction → Junction**: JunctionCrossing + Move + JunctionCrossing
    * **Trap → Trap**: Split + Move + Merge

    Additionally, when the source is a Trap containing exactly one ion
    a "rotation" edge-op (a self-``GateSwap``) is prepended so the
    routing code can hit its ``isinstance(m, GateSwap) and
    m._ions[0] == m._ions[1]`` skip-check.

    The returned closures capture *references* to the architecture
    objects so that the actual state mutation happens at ``.run()``-time.
    """
    from .operations import GateSwap as _GS

    src_is_trap = isinstance(src_node, (ManipulationTrap, StorageTrap, Trap))
    tgt_is_trap = isinstance(tgt_node, (ManipulationTrap, StorageTrap, Trap))

    # ---- closure factories ------------------------------------------------

    def _rotation():
        """Self-GateSwap 'rotation' for single-ion source trap."""
        pass  # no-op; greedy routing skips it via identity check

    def _split():
        edge_ion = crossing.getEdgeIon(src_node)
        src_node.remove_ion(edge_ion)
        crossing.setIon(edge_ion, src_node)

    def _move():
        crossing.moveIon()

    def _merge():
        ion = crossing.ion
        crossing.clearIon()
        tgt_node.add_ion(ion)

    def _junction_enter():
        """Ion enters a junction (from crossing side)."""
        ion = crossing.ion
        crossing.clearIon()
        tgt_node.add_ion(ion)

    def _junction_leave():
        """Ion leaves a junction (into crossing)."""
        if src_node.ions:
            edge_ion = src_node.ions[0]
        else:
            edge_ion = crossing.getEdgeIon(src_node)
        src_node.remove_ion(edge_ion)
        crossing.setIon(edge_ion, src_node)

    # ---- build sequence ---------------------------------------------------
    ops: list = []

    if src_is_trap and not tgt_is_trap:
        # Trap → Junction
        ops.append(_EdgeOp(_split, f"Split({src_node.label}{src_node.idx})",
                           [src_node, crossing]))
        ops.append(_EdgeOp(_move,  f"Move(C{crossing.idx})", [crossing]))
        ops.append(_EdgeOp(_junction_enter,
                           f"JCross({tgt_node.label}{tgt_node.idx})",
                           [tgt_node, crossing]))

    elif not src_is_trap and tgt_is_trap:
        # Junction → Trap
        ops.append(_EdgeOp(_junction_leave,
                           f"JCross({src_node.label}{src_node.idx})",
                           [src_node, crossing]))
        ops.append(_EdgeOp(_move,  f"Move(C{crossing.idx})", [crossing]))
        ops.append(_EdgeOp(_merge, f"Merge({tgt_node.label}{tgt_node.idx})",
                           [tgt_node, crossing]))

    elif not src_is_trap and not tgt_is_trap:
        # Junction → Junction
        ops.append(_EdgeOp(_junction_leave,
                           f"JCross({src_node.label}{src_node.idx})",
                           [src_node, crossing]))
        ops.append(_EdgeOp(_move,  f"Move(C{crossing.idx})", [crossing]))
        ops.append(_EdgeOp(_junction_enter,
                           f"JCross({tgt_node.label}{tgt_node.idx})",
                           [tgt_node, crossing]))

    else:
        # Trap → Trap (direct, no junction)
        ops.append(_EdgeOp(_split, f"Split({src_node.label}{src_node.idx})",
                           [src_node, crossing]))
        ops.append(_EdgeOp(_move,  f"Move(C{crossing.idx})", [crossing]))
        ops.append(_EdgeOp(_merge, f"Merge({tgt_node.label}{tgt_node.idx})",
                           [tgt_node, crossing]))

    return ops


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
        # Ordered list of all ions (logical qubit 0 → first ion added, etc.)
        self._qubit_ions: List[Ion] = []
    
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
    def crossingEdges(self) -> Dict[Tuple[int, int], Crossing]:
        """Crossing edges (old API compatibility)."""
        return self._crossing_edges
    
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

    @property
    def qubit_ions(self) -> List[Ion]:
        """Ordered list mapping logical qubit index → Ion object.

        ``qubit_ions[i]`` is the Ion corresponding to logical qubit *i*
        (in the order ions were added to traps).
        """
        return self._qubit_ions
    
    def add_manipulation_trap(
        self,
        position: Tuple[float, float],
        ions: List[Ion],
        capacity: int = 4,
        is_horizontal: bool = True,
        spacing: float = 1.0,
    ) -> ManipulationTrap:
        """Add a manipulation trap to the graph.

        Index allocation follows the old ``QCCDArch`` convention:
        ``_next_idx`` is used for the trap node, then each ion gets
        indices ``_next_idx + 1 .. _next_idx + len(ions)``, and the
        counter advances by ``len(ions) + 1``.  This guarantees
        ``node.nodes == [trap_idx, ion0_idx, ion1_idx, ...]`` which
        the greedy routing algorithm relies on.
        """
        trap_idx = self._next_idx
        # Reassign ion indices to be contiguous with the trap
        for i, ion in enumerate(ions):
            ion.index = trap_idx + 1 + i
            ion.id = f"{ion.label}{ion.index}"
        trap = ManipulationTrap(
            idx=trap_idx,
            position=position,
            capacity=capacity,
            ions=ions,
            is_horizontal=is_horizontal,
            spacing=spacing,
        )
        self._nodes[trap.idx] = trap
        self._qubit_ions.extend(ions)
        self._next_idx += len(ions) + 1
        return trap
    
    def add_storage_trap(
        self,
        position: Tuple[float, float],
        ions: List[Ion],
        capacity: int = 8,
        is_horizontal: bool = True,
        spacing: float = 1.0,
    ) -> StorageTrap:
        """Add a storage trap to the graph (block index allocation)."""
        trap_idx = self._next_idx
        for i, ion in enumerate(ions):
            ion.index = trap_idx + 1 + i
            ion.id = f"{ion.label}{ion.index}"
        trap = StorageTrap(
            idx=trap_idx,
            position=position,
            capacity=capacity,
            ions=ions,
            is_horizontal=is_horizontal,
            spacing=spacing,
        )
        self._nodes[trap.idx] = trap
        self._qubit_ions.extend(ions)
        self._next_idx += len(ions) + 1
        return trap
    
    def add_junction(
        self,
        position: Tuple[float, float],
        capacity: int = 1,
    ) -> Junction:
        """Add a junction to the graph.

        Default capacity is 1.  The greedy routing algorithm uses the
        ``numIons`` counter (not actual ion count) to track junction
        occupancy and checks ``isinstance(node, Junction) and
        node.numIons == 1`` to mark a junction as full.
        """
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
    
    def build_networkx_graph(self, include_intra_trap: bool = True) -> nx.DiGraph:
        """Build/refresh the NetworkX graph from nodes and crossings.

        Faithfully reproduces the old ``QCCDArch.refreshGraph()``
        topology so that :func:`routing.greedy_routing.ionRouting`
        works correctly:

        1. **Ion + trap/junction nodes** — via ``node.subgraph(g)``.
        2. **Ion → parent-trap edges** — weight-0, empty operations.
        3. **Intra-trap GateSwap edges** — every ion pair in each trap.
        4. **Crossing edges** — endpoints are *ion-level* indices
           obtained from ``crossing.graphEdge()``.  The
           ``_crossing_edges`` dict is re-keyed by those ion indices.
        5. **``numIons`` reset** — set to ``len(node.ions)`` for every
           node so that the routing's manual counter starts correct.

        Parameters
        ----------
        include_intra_trap : bool
            If True (default), add ion↔ion GateSwap edges inside traps.
        """
        from .operations import GateSwap as _GS

        g = nx.DiGraph()

        # ---- 1. nodes (trap/junction + ions) via subgraph ----
        for node in self._nodes.values():
            node.subgraph(g)
            # Reset numIons to actual count so routing counters start fresh
            node.numIons = len(node.ions)

        # ---- 2. ion → parent-trap edges ----
        for node in self._nodes.values():
            if isinstance(node, (ManipulationTrap, StorageTrap)):
                for ion in node.ions:
                    g.add_edge(ion.idx, node.idx, operations=[], weight=0)

        # ---- 3. intra-trap GateSwap edges (ion ↔ ion) ----
        if include_intra_trap:
            for node in self._nodes.values():
                if isinstance(node, (ManipulationTrap, StorageTrap)):
                    for i, ion1 in enumerate(node.ions):
                        for j, ion2 in enumerate(node.ions):
                            if i != j:
                                swap = _GS.qubitOperation(ion1, ion2, trap=node)
                                g.add_edge(
                                    ion1.idx, ion2.idx,
                                    weight=0, same_trap=True,
                                    operations=[swap],
                                )

        # ---- 4. crossing edges (ion-level endpoints) ----
        # Re-key _crossing_edges by ion-level indices as the old code did
        self._crossing_edges.clear()
        for crossing in self._crossings.values():
            src_node, tgt_node = crossing.source, crossing.target
            try:
                n1Idx, n2Idx = crossing.graphEdge()
            except (ValueError, IndexError):
                # Fallback: use node-level indices if no ions (e.g. junction↔junction)
                n1Idx, n2Idx = src_node.idx, tgt_node.idx

            ops_fwd = _make_crossing_ops(src_node, tgt_node, crossing)
            ops_rev = _make_crossing_ops(tgt_node, src_node, crossing)

            g.add_edge(n1Idx, n2Idx, crossing=crossing, weight=1,
                       operations=ops_fwd)
            g.add_edge(n2Idx, n1Idx, crossing=crossing, weight=1,
                       operations=ops_rev)

            self._crossing_edges[(n1Idx, n2Idx)] = crossing
            self._crossing_edges[(n2Idx, n1Idx)] = crossing

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
        ms_gate_time: float = _CAL.ms_gate_time * 1e6,  # μs (from physics.py)
        single_qubit_time: float = _CAL.single_qubit_gate_time * 1e6,  # μs
        measurement_time: float = _CAL.measurement_time * 1e6,  # μs
        t2_time: float = _CAL.t2_time * 1e6,  # μs
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, num_qubits, metadata)
        self.ms_gate_time = ms_gate_time
        self.single_qubit_time = single_qubit_time
        self.measurement_time = measurement_time
        self.t2_time = t2_time
    
    def native_gate_set(self) -> NativeGateSet:
        """Trapped ion native gates: MS + single-qubit rotations."""
        # Lazy import to avoid circular dependency (operations imports architecture)
        from qectostim.experiments.hardware_simulation.trapped_ion.operations import TRAPPED_ION_NATIVE_GATES
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
            reset_time=_CAL.reset_time * 1e6,  # μs (from physics.py)
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

    # ── Routing-compatible delegation properties ─────────────────
    # These let routing code use QCCDArchitecture / WISEArchitecture
    # directly via duck-typed .graph, .ions, .nodes, etc.

    @property
    def graph(self) -> nx.DiGraph:
        """The underlying NetworkX graph (routing-compatible)."""
        return self._qccd_graph.graph

    @property
    def ions(self) -> Dict[int, Ion]:
        """All ions across all nodes (routing-compatible)."""
        return self._qccd_graph.ions

    @property
    def qubit_ions(self) -> List[Ion]:
        """Ordered list: ``qubit_ions[logical_idx]`` → Ion."""
        return self._qccd_graph.qubit_ions

    @property
    def nodes(self) -> Dict[int, QCCDNode]:
        """All nodes (traps and junctions) (routing-compatible)."""
        return self._qccd_graph.nodes

    @property
    def crossingEdges(self) -> Dict[Tuple[int, int], Crossing]:
        """Crossing edges dict (routing-compatible)."""
        return self._qccd_graph.crossingEdges

    @property
    def _manipulationTraps(self) -> List[ManipulationTrap]:
        """All manipulation traps (routing-compatible)."""
        return [n for n in self._qccd_graph.nodes.values()
                if isinstance(n, ManipulationTrap)]

    def refreshGraph(self) -> None:
        """Rebuild the NetworkX graph and routing table (routing-compatible)."""
        self._qccd_graph.build_networkx_graph()
        self._qccd_graph.compute_routing_table()

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


# =============================================================================
# WISE Architecture
# =============================================================================

class WISEArchitecture(QCCDArchitecture):
    """WISE (Wired Ion Scalable Entanglement) grid architecture.

    A 2-D grid of **m × n** ion traps (column-groups × rows), each
    containing **k** ions.  Adjacent column-groups in the same row are
    connected via junctions; junctions in adjacent rows are connected
    vertically so that ions can be routed across the full grid.

    Parameters
    ----------
    col_groups : int
        Number of column groups (blocks), *m*.
    rows : int
        Number of rows, *n*.
    ions_per_segment : int
        Number of ions per trap segment, *k*.
    trap_spacing : float
        Physical spacing between traps (µm).
    name : str, optional
        Human-readable architecture name.
    metadata : dict, optional
        Arbitrary extra metadata.

    Attributes
    ----------
    col_groups : int
        Number of column groups (*m*).
    rows : int
        Number of rows (*n*).  Inherited from ``QCCDArchitecture``.
    ions_per_segment : int
        Ions per segment (*k*).
    num_qubits : int
        Total qubit count = *m × n × k*.
    total_columns : int
        Absolute grid width = *m × k*.
    grid_shape : Tuple[int, int]
        ``(n_rows, n_cols)`` = ``(rows, total_columns)`` — the shape
        used by the SAT router.
    k : int
        Alias for ``ions_per_segment`` (trap capacity).
    capacity : int
        Alias for ``ions_per_segment`` (block capacity).
    n_rows : int
        Alias for ``rows``.
    n_cols : int
        Alias for ``total_columns``.
    traps : Dict[Tuple[int, int], ManipulationTrap]
        Traps keyed by ``(block, row)``.
    junctions : Dict[Tuple[int, int], Junction]
        Junctions keyed by ``(block, row)``.

    Notes
    -----
    Ion indexing follows row-major order::

        ion_idx = row * total_columns + block * k + slot

    where ``slot ∈ [0, k)``.

    The SAT-based router and compiler access the architecture through
    ``grid_shape``, ``capacity``, ``n_rows``, ``n_cols``.  The
    visualisation layer uses ``col_groups``, ``rows``,
    ``ions_per_segment``, ``total_columns``.
    """

    def __init__(
        self,
        col_groups: int = 2,
        rows: int = 2,
        ions_per_segment: int = 2,
        trap_spacing: float = 10.0,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Store WISE-specific attributes *before* super().__init__
        # because it calls _build_grid_topology() immediately.
        self.col_groups = col_groups
        self.ions_per_segment = ions_per_segment
        self._traps: Dict[Tuple[int, int], ManipulationTrap] = {}
        self._junctions: Dict[Tuple[int, int], Junction] = {}
        self._ion_index: Dict[int, Ion] = {}

        super().__init__(
            rows=rows,
            cols=col_groups,
            ions_per_trap=ions_per_segment,
            trap_spacing=trap_spacing,
            name=name or f"WISE_{col_groups}x{rows}x{ions_per_segment}",
            metadata=metadata,
        )

    # --------------------------------------------------------------------- #
    #  Topology builder  (overrides QCCDArchitecture._build_grid_topology)   #
    # --------------------------------------------------------------------- #

    def _build_grid_topology(self) -> None:
        """Build the WISE *m × n × k* grid topology.

        Creates:

        * **m × n** manipulation traps, each holding *k* ions.
        * **(m − 1) × n** horizontal junctions connecting adjacent
          column-groups within each row.
        * Vertical crossings linking junctions of neighbouring rows
          so that ions can be routed between rows.

        Called automatically by ``QCCDArchitecture.__init__``.
        """
        self._qccd_graph = QCCDGraph()

        # Defensive re-init (handles construction ordering)
        self._traps = getattr(self, "_traps", {})
        self._junctions = getattr(self, "_junctions", {})
        self._ion_index = getattr(self, "_ion_index", {})
        self._traps.clear()
        self._junctions.clear()
        self._ion_index.clear()

        m = self.col_groups
        n = self.rows
        k = self.ions_per_segment
        total_cols = m * k

        # ---- 1. Create manipulation traps --------------------------------
        for r in range(n):
            for b in range(m):
                ions: List[Ion] = []
                for s in range(k):
                    idx = r * total_cols + b * k + s
                    ion = Ion(
                        idx=idx,
                        label="Q",
                        position=(float(b * k + s), float(r)),
                    )
                    ions.append(ion)
                    self._ion_index[idx] = ion

                x = b * self.trap_spacing * 2
                y = r * self.trap_spacing * 2

                trap = self._qccd_graph.add_manipulation_trap(
                    position=(x, y),
                    ions=ions,
                    capacity=k,
                    is_horizontal=True,
                )
                self._traps[(b, r)] = trap

        # ---- 2. Create horizontal junctions ------------------------------
        for r in range(n):
            for b in range(m - 1):
                jx = (b * 2 + 1) * self.trap_spacing
                jy = r * self.trap_spacing * 2
                junction = self._qccd_graph.add_junction(position=(jx, jy))
                self._junctions[(b, r)] = junction

                # trap(b, r)  ↔  junction  ↔  trap(b+1, r)
                self._qccd_graph.add_crossing(
                    self._traps[(b, r)], junction
                )
                self._qccd_graph.add_crossing(
                    junction, self._traps[(b + 1, r)]
                )

        # ---- 3. Vertical crossings between junctions --------------------
        for r in range(n - 1):
            for b in range(m - 1):
                self._qccd_graph.add_crossing(
                    self._junctions[(b, r)],
                    self._junctions[(b, r + 1)],
                )

        # ---- 4. Finalise graph -------------------------------------------
        self._qccd_graph.build_networkx_graph()
        self._qccd_graph.compute_routing_table()

    # --------------------------------------------------------------------- #
    #  Properties                                                            #
    # --------------------------------------------------------------------- #

    @property
    def k(self) -> int:
        """Trap capacity (alias for ``ions_per_segment``)."""
        return self.ions_per_segment

    @property
    def capacity(self) -> int:
        """Block capacity (alias for ``ions_per_segment``)."""
        return self.ions_per_segment

    @property
    def total_columns(self) -> int:
        """Total grid columns = ``col_groups × ions_per_segment``."""
        return self.col_groups * self.ions_per_segment

    @property
    def n_rows(self) -> int:
        """Number of rows (alias for ``rows``)."""
        return self.rows

    @property
    def n_cols(self) -> int:
        """Number of grid columns (alias for ``total_columns``)."""
        return self.total_columns

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """``(n_rows, n_cols)`` grid dimensions for the SAT router."""
        return (self.rows, self.total_columns)

    @property
    def m(self) -> int:
        """Number of column groups (QCCDWiseArch-compatible alias for ``col_groups``)."""
        return self.col_groups

    @property
    def n(self) -> int:
        """Number of rows (QCCDWiseArch-compatible alias for ``rows``)."""
        return self.rows

    @property
    def traps(self) -> Dict[Tuple[int, int], ManipulationTrap]:
        """Trap dict keyed by ``(block, row)``."""
        return self._traps

    @property
    def junctions(self) -> Dict[Tuple[int, int], Junction]:
        """Junction dict keyed by ``(block, row)``."""
        return self._junctions

    @property
    def wise_config(self) -> "QCCDWISEConfig":
        """Return a QCCDWISEConfig for this architecture.
        
        Used by the adapter layer to build old-style routing objects.
        """
        return QCCDWISEConfig(m=self.col_groups, n=self.rows, k=self.ions_per_segment)

    # --------------------------------------------------------------------- #
    #  Ion / trap lookup                                                     #
    # --------------------------------------------------------------------- #

    def get_ion(self, idx: int) -> Optional[Ion]:
        """Look up an :class:`Ion` by its global index.

        Parameters
        ----------
        idx : int
            Ion index (0-based, row-major).

        Returns
        -------
        Ion or None
            The ``Ion`` object, or ``None`` if *idx* is out of range.
        """
        return self._ion_index.get(idx)

    def get_trap_for_ion(self, idx: int) -> Optional[ManipulationTrap]:
        """Return the trap containing the ion with the given index.

        Parameters
        ----------
        idx : int
            Ion index.

        Returns
        -------
        ManipulationTrap or None
        """
        tc = self.total_columns
        k_val = self.ions_per_segment
        if tc == 0 or k_val == 0 or idx < 0 or idx >= self.num_qubits:
            return None
        row = idx // tc
        block = (idx % tc) // k_val
        return self._traps.get((block, row))

    # --------------------------------------------------------------------- #
    #  Overrides                                                             #
    # --------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"WISEArchitecture(col_groups={self.col_groups}, "
            f"rows={self.rows}, ions_per_segment={self.ions_per_segment})"
        )


# =============================================================================
# Augmented Grid Architecture
# =============================================================================

class AugmentedGridArchitecture(QCCDArchitecture):
    """Augmented grid QCCD architecture.

    A 2-D chequerboard layout of manipulation traps where **even** rows
    use column positions ``(2c, 2r)`` and **odd** interleaved rows use
    offset positions ``(2c+1, 2r+1)``.  Vertical neighbours in the
    even-column grid are connected via junctions; horizontal edges link
    the junctions to the diagonal interleaved traps.

    This topology offers richer connectivity than a simple rectangular
    grid while remaining physically implementable with T-junction
    segments.

    Parameters
    ----------
    rows : int
        Number of main rows.
    cols : int
        Number of main columns.
    ions_per_trap : int
        Maximum ions per manipulation trap.
    padding : int
        Extra empty trap rows/cols around the populated core.
    trap_spacing : float
        Physical spacing multiplier.

    Notes
    -----
    Ported from ``old/src/simulator/qccd_circuit.py``
    → ``processCircuitAugmentedGrid()``.
    """

    def __init__(
        self,
        rows: int = 3,
        cols: int = 5,
        ions_per_trap: int = 3,
        padding: int = 1,
        trap_spacing: float = 10.0,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.padding = padding
        self._traps_dict: Dict[Tuple[int, int], ManipulationTrap] = {}
        self._junctions_dict: Dict[Tuple[int, int], Junction] = {}

        # QCCDArchitecture.__init__ calls _build_grid_topology()
        super().__init__(
            rows=rows,
            cols=cols,
            ions_per_trap=ions_per_trap,
            trap_spacing=trap_spacing,
            name=name or f"AugGrid_{rows}x{cols}",
            metadata=metadata,
        )

    # --------------------------------------------------------------------- #

    def _build_grid_topology(  # noqa: C901
        self,
        initial_ions: Optional[Dict[Tuple[int, int], List[Ion]]] = None,
    ) -> None:
        """Build the augmented (chequerboard) grid topology.

        Grid positions:
        * Even rows, even cols: ``(2c, 2r)``       — main traps
        * Interleaved:          ``(2c+1, 2r+1)``   — diagonal traps
        * Junctions sit midway between vertical main-trap neighbours.
        * Horizontal edges connect junctions to diagonal traps.

        Parameters
        ----------
        initial_ions : dict, optional
            Mapping ``{(grid_col, grid_row): [Ion, ...]}`` specifying
            which ions to place in each trap.  If *None* (the default),
            every trap is filled with ``ions_per_trap`` generic "Q" ions.
            When supplied, traps whose grid key is absent receive **no**
            ions (empty traps).  Ion indices are reassigned by
            :meth:`QCCDGraph.add_manipulation_trap` regardless of the
            values passed in.
        """
        self._qccd_graph = QCCDGraph()
        self._traps_dict = {}
        self._junctions_dict = {}

        rows = self.rows
        cols = self.cols
        k = self.ions_per_trap

        ion_idx = 0

        # ---- 1. Create main traps at (2c, 2r) ----------------------------
        for r in range(rows):
            for c in range(cols):
                grid_key = (2 * c, 2 * r)
                if initial_ions is not None:
                    ions = initial_ions.get(grid_key, [])
                else:
                    ions = [
                        Ion(idx=ion_idx + i, label="Q",
                            position=(float(2 * c), float(2 * r)))
                        for i in range(k)
                    ]
                ion_idx += len(ions)
                pos = (2 * c * self.trap_spacing, 2 * r * self.trap_spacing)
                trap = self._qccd_graph.add_manipulation_trap(
                    position=pos,
                    ions=ions,
                    capacity=k + 1,  # Extra slot for routing
                    is_horizontal=(rows == 1),
                )
                self._traps_dict[grid_key] = trap

            # ---- 1b. Interleaved diagonal traps at (2c+1, 2r+1) ----------
            if r < rows - 1:
                for c in range(cols - 1):
                    grid_key = (2 * c + 1, 2 * r + 1)
                    if initial_ions is not None:
                        ions = initial_ions.get(grid_key, [])
                    else:
                        ions = [
                            Ion(idx=ion_idx + i, label="Q",
                                position=(float(2 * c + 1), float(2 * r + 1)))
                            for i in range(k)
                        ]
                    ion_idx += len(ions)
                    pos = (
                        (2 * c + 1) * self.trap_spacing,
                        (2 * r + 1) * self.trap_spacing,
                    )
                    trap = self._qccd_graph.add_manipulation_trap(
                        position=pos,
                        ions=ions,
                        capacity=k + 1,  # Extra slot for routing
                        is_horizontal=True,
                    )
                    self._traps_dict[grid_key] = trap

        # ---- 2. Edges / junctions -----------------------------------------
        if rows == 1:
            # Simple linear chain of even traps
            for (col, r), trap in self._traps_dict.items():
                if (col + 2, r) in self._traps_dict:
                    self._qccd_graph.add_crossing(trap, self._traps_dict[(col + 2, r)])
        else:
            # Vertical junctions between even-column traps
            for (col, row), trap in list(self._traps_dict.items()):
                if col % 2 == 0 and (col, row + 2) in self._traps_dict:
                    mid_pos = (
                        col * self.trap_spacing,
                        (row + 1) * self.trap_spacing,
                    )
                    junction = self._qccd_graph.add_junction(position=mid_pos)
                    self._junctions_dict[(col, row + 1)] = junction
                    self._qccd_graph.add_crossing(trap, junction)
                    self._qccd_graph.add_crossing(junction, self._traps_dict[(col, row + 2)])

            # Horizontal edges: junction ↔ diagonal trap
            for r in range(rows - 1):
                for c in range(cols - 1):
                    jkey = (2 * c, 2 * r + 1)
                    tkey = (2 * c + 1, 2 * r + 1)
                    if jkey in self._junctions_dict and tkey in self._traps_dict:
                        self._qccd_graph.add_crossing(
                            self._junctions_dict[jkey], self._traps_dict[tkey]
                        )
                    tkey2 = (2 * c + 1, 2 * r + 1)
                    jkey2 = (2 * c + 2, 2 * r + 1)
                    if tkey2 in self._traps_dict and jkey2 in self._junctions_dict:
                        self._qccd_graph.add_crossing(
                            self._traps_dict[tkey2], self._junctions_dict[jkey2]
                        )

        # ---- 3. Finalise ---------------------------------------------------
        self._qccd_graph.build_networkx_graph()
        self._qccd_graph.compute_routing_table()

    # ---- Properties --------------------------------------------------------

    @property
    def traps_dict(self) -> Dict[Tuple[int, int], ManipulationTrap]:
        return self._traps_dict

    @property
    def junctions_dict(self) -> Dict[Tuple[int, int], Junction]:
        return self._junctions_dict

    def __repr__(self) -> str:
        return (
            f"AugmentedGridArchitecture(rows={self.rows}, cols={self.cols}, "
            f"ions_per_trap={self.ions_per_trap}, padding={self.padding})"
        )


# =============================================================================
# Networked Grid Architecture
# =============================================================================

class NetworkedGridArchitecture(QCCDArchitecture):
    """Networked (fully-connected) QCCD architecture.

    A linear chain of manipulation traps, each connected through a
    junction sub-chain.  All junction endpoints are cross-connected,
    giving an all-to-all routing topology between traps.

    Parameters
    ----------
    num_traps : int
        Number of manipulation traps.
    ions_per_trap : int
        Maximum ions per trap.
    trap_spacing : float
        Physical spacing multiplier.

    Notes
    -----
    Ported from ``old/src/simulator/qccd_circuit.py``
    → ``processCircuitNetworkedGrid()``.
    """

    def __init__(
        self,
        num_traps: int = 5,
        ions_per_trap: int = 3,
        trap_spacing: float = 10.0,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.num_traps = num_traps
        self._traps_dict: Dict[int, ManipulationTrap] = {}
        self._junctions_dict: Dict[Tuple[int, int], Junction] = {}

        super().__init__(
            rows=num_traps,
            cols=1,
            ions_per_trap=ions_per_trap,
            trap_spacing=trap_spacing,
            name=name or f"Networked_{num_traps}",
            metadata=metadata,
        )

    # --------------------------------------------------------------------- #

    def _build_grid_topology(self) -> None:
        """Build a fully-connected networked topology.

        Each trap has a junction chain of length 1 to its right.
        All junction endpoints are cross-connected to every other
        junction endpoint, giving all-to-all connectivity.
        """
        self._qccd_graph = QCCDGraph()
        self._traps_dict = {}
        self._junctions_dict = {}

        k = self.ions_per_trap
        n_traps = self.num_traps
        switch_cost = 1  # junction-chain length per trap
        ion_idx = 0

        # ---- 1. Create traps ------------------------------------------------
        for row in range(n_traps):
            ions = [
                Ion(idx=ion_idx + i, label="Q", position=(0.0, float(row)))
                for i in range(k)
            ]
            ion_idx += k
            pos = (0.0, row * self.trap_spacing)
            trap = self._qccd_graph.add_manipulation_trap(
                position=pos, ions=ions, capacity=k, is_horizontal=True,
            )
            self._traps_dict[row] = trap

        # ---- 2. Create junction chain per trap ------------------------------
        for row, trap in self._traps_dict.items():
            for i in range(switch_cost):
                jpos = ((i + 1) * self.trap_spacing, row * self.trap_spacing)
                junction = self._qccd_graph.add_junction(position=jpos)
                self._junctions_dict[(i + 1, row)] = junction
                if i == 0:
                    self._qccd_graph.add_crossing(trap, junction)
                else:
                    self._qccd_graph.add_crossing(
                        self._junctions_dict[(i, row)], junction,
                    )

        # ---- 3. Cross-connect all junction endpoints (all-to-all) -----------
        for row1 in range(n_traps):
            j1 = self._junctions_dict[(switch_cost, row1)]
            for row2 in range(n_traps):
                if row1 == row2:
                    continue
                j2 = self._junctions_dict[(switch_cost, row2)]
                self._qccd_graph.add_crossing(j1, j2)

        # ---- 4. Finalise ----------------------------------------------------
        self._qccd_graph.build_networkx_graph()
        self._qccd_graph.compute_routing_table()

    # ---- Properties ---------------------------------------------------------

    @property
    def traps(self) -> Dict[int, ManipulationTrap]:
        return self._traps_dict

    def __repr__(self) -> str:
        return (
            f"NetworkedGridArchitecture(num_traps={self.num_traps}, "
            f"ions_per_trap={self.ions_per_trap})"
        )


# =============================================================================
# Re-export qubit operations for old API compatibility
# =============================================================================
# These imports must come at the end to avoid circular imports
def _load_qubit_ops():
    """Lazy load qubit operations to avoid circular imports."""
    from qectostim.experiments.hardware_simulation.trapped_ion.operations import (
        QubitOperation,
        OneQubitGate,
        XRotation,
        YRotation,
        Measurement,
        QubitReset,
        TwoQubitMSGate,
        GateSwap,
    )
    return {
        'QubitOperation': QubitOperation,
        'OneQubitGate': OneQubitGate,
        'XRotation': XRotation,
        'YRotation': YRotation,
        'Measurement': Measurement,
        'QubitReset': QubitReset,
        'TwoQubitMSGate': TwoQubitMSGate,
        'GateSwap': GateSwap,
    }

# Lazy load and export
_qubit_ops = None

def __getattr__(name):
    global _qubit_ops
    qubit_op_names = {
        'QubitOperation', 'OneQubitGate', 'XRotation', 'YRotation',
        'Measurement', 'QubitReset', 'TwoQubitMSGate', 'GateSwap'
    }
    if name in qubit_op_names:
        if _qubit_ops is None:
            _qubit_ops = _load_qubit_ops()
        return _qubit_ops[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
