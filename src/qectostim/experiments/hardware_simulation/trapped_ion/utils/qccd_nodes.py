import numpy as np
from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Union,
    Mapping
)
import networkx as nx
import enum
import abc
from dataclasses import dataclass
from .physics import DEFAULT_CALIBRATION, ModeStructure


class Operations(enum.Enum):
    """
    Trapped Ion QIP ToolBox
    """

    SPLIT = enum.auto()
    MOVE = enum.auto()
    MERGE = enum.auto()
    GATE_SWAP = enum.auto()
    CRYSTAL_ROTATION = enum.auto()
    ONE_QUBIT_GATE = enum.auto()
    TWO_QUBIT_MS_GATE = enum.auto()
    JUNCTION_CROSSING = enum.auto()
    MEASUREMENT = enum.auto()
    QUBIT_RESET = enum.auto()
    # TODO need to add recooling operation
    RECOOLING = enum.auto()
    PARALLEL = enum.auto()
    GLOBAL_RECONFIG = enum.auto()

class QCCDComponent:
    @property
    @abc.abstractmethod
    def pos(self) -> Tuple[float, float]: ...

    @property
    @abc.abstractmethod
    def idx(self) -> int: ...

    @property
    @abc.abstractmethod
    def allowedOperations(self) -> Sequence[Operations]:
        ...


class Ion(QCCDComponent):
    def __init__(self, color: str = "lightblue", label: str = "Q") -> None:
        self._idx: int = 0
        self._positionX: int = 0
        self._positionY: int = 0
        self._parent: Optional[Union["QCCDNode", "Crossing"]] = None
        self._color = color
        self._label = label
        self._motionalEnergy = 0.0

    def addMotionalEnergy(self, energy: float) -> None:
        self._motionalEnergy += energy

    @property
    def motionalMode(self) -> float:
        return self._motionalEnergy

    def set(
        self,
        idx: int,
        x: int,
        y: int,
        parent: Optional[Union["QCCDNode", "Crossing"]] = None,
    ) -> None:
        self._idx = idx
        self._positionX: int = x
        self._positionY: int = y
        self._parent = parent

    @property
    def parent(self) -> Optional[Union["QCCDNode", "Crossing"]]:
        return self._parent

    @property
    def pos(self) -> Tuple[float, float]:
        return (self._positionX, self._positionY)

    @property
    def color(self) -> str:
        return self._color

    @property
    def label(self) -> str:
        return self._label + str(int(self._idx))

    @property
    def idx(self) -> int:
        return self._idx
    
    @property
    def allowedOperations(self) -> Sequence[Operations]:
        return [Operations.CRYSTAL_ROTATION, Operations.GATE_SWAP, Operations.JUNCTION_CROSSING, Operations.MERGE, Operations.SPLIT, Operations.MOVE, Operations.RECOOLING]

class CoolingIon(Ion):
    ...

class QubitIon(Ion):
    @property
    def allowedOperations(self) -> Sequence[Operations]:
        return super().allowedOperations+[Operations.MEASUREMENT, Operations.ONE_QUBIT_GATE, Operations.TWO_QUBIT_MS_GATE, Operations.QUBIT_RESET]

class SpectatorIon(QubitIon):
    ...


class QCCDNode(QCCDComponent):
    # TODO only allow defined set of operations
    def __init__(
        self,
        idx: int,
        x: int,
        y: int,
        color: str,
        capacity: int,
        label: str,
        ions: Sequence[Ion] = [],
    ) -> None:
        self._idx: int = idx
        self._positionX: int = x
        self._positionY: int = y
        self._color = color
        self._ions: List[Ion] = list(ions)
        self._capacity: int = capacity
        self._label: str = label
        self.numIons: int = len(ions)

    @property
    def label(self) -> int:
        return self._label + str(self._idx)

    @property
    def ions(self) -> Sequence[Ion]:
        return self._ions

    @property
    def color(self) -> str:
        return self._color

    @property
    def pos(self) -> Tuple[float, float]:
        return (self._positionX, self._positionY)

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def nodes(self) -> Sequence[int]:
        n = [self._idx]
        if self.ions:
            n += [i.idx for i in self.ions]
        return n

    @property
    def positions(self) -> Sequence[Tuple[int, int]]:
        n = [self.pos]
        if self.ions:
            n += [i.pos for i in self.ions]
        return n

    def addMotionalEnergy(self, energy: float) -> None:
        for ion in self._ions:
            ion.addMotionalEnergy(energy / len(self._ions))

    @property
    def motionalMode(self) -> float:
        return sum(ion.motionalMode for ion in self._ions if not isinstance(ion, CoolingIon))

    def subgraph(self, graph: nx.Graph) -> nx.Graph:
        for n, p in zip(self.nodes, self.positions):
            graph.add_node(n, pos=p)
        return graph

    def addIon(
        self, ion: Ion, adjacentIon: Optional[Ion] = None, offset: int = 0
    ) -> None:
        if len(self.ions) == self._capacity:
            raise ValueError(
                f"addIon: QCCDNode {self.idx} {self} is at capacity {self._capacity}"
            )
        self._ions.insert(
            (self._ions.index(adjacentIon) + offset if adjacentIon else offset), ion
        )
        ion.set(ion.idx, *self.pos, parent=self)
        self.numIons = len(self._ions)

    def removeIon(self, ion: Optional[Ion] = None) -> Ion:
        if ion is None:
            if len(self.ions) == 0:
                raise ValueError(
                    f"removeIon: QCCDNode {self.idx} does not have any ions"
                )
            ion = self.ions[0]
        self._ions.remove(ion)
        self.numIons = len(self._ions)
        return ion


class Junction(QCCDNode):
    DEFAULT_COLOR = "orange"
    DEFAULT_LABEL = "J"
    DEFAULT_CAPACITY = 1

    def __init__(
        self,
        idx: int,
        x: int,
        y: int,
        color: str = DEFAULT_COLOR,
        label: str = DEFAULT_LABEL,
        capacity: int = DEFAULT_CAPACITY,
    ) -> None:
        super().__init__(idx, x, y, color=color, capacity=capacity, label=label)

    @property
    def allowedOperations(self) -> Sequence[Operations]:
        return [Operations.JUNCTION_CROSSING, Operations.SPLIT, Operations.MOVE, Operations.MERGE]


class Trap(QCCDNode):
    # BACKGROUND_HEATING_RATE now centralized in physics.py
    BACKGROUND_HEATING_RATE = DEFAULT_CALIBRATION.heating_rate
    CAPACITY_SCALING = 1

    def __init__(
        self,
        idx: int,
        x: int,
        y: int,
        ions: Sequence[Ion],
        color: str,
        isHorizontal: bool,
        spacing: int,
        capacity: int,
        label: str,
    ) -> None:
        super().__init__(
            idx, x, y, color=color, capacity=capacity*self.CAPACITY_SCALING, ions=ions, label=label
        )
        self._desired_capacity = capacity
        self._spacing = spacing
        self._isHorizontal = isHorizontal
        self._coolingIons: Sequence[CoolingIon] = []
        for i, ion in enumerate(self._ions):
            ion.set(self._idx + i + 1, 0, 0, parent=self)
        self._arrangeIons()

    @property
    def desiredCapacity(self) -> int:
        return self._desired_capacity

    @property
    def hasCoolingIon(self) -> bool:
        return any(isinstance(ion, CoolingIon) for ion in self.ions)
    
    def coolTrap(self) -> bool:
        coolingIons = [ion for ion in self.ions if isinstance(ion, CoolingIon)]
        qubitIons = [ion for ion in self.ions if isinstance(ion, QubitIon)]
        energy = 0.0
        for ion in qubitIons:
            energy += ion.motionalMode
            ion.addMotionalEnergy(-ion.motionalMode)
        energy /= len(coolingIons)
        for ion in coolingIons:
            ion.addMotionalEnergy(energy)

    @property
    def backgroundHeatingRate(self) -> float:
        return self.BACKGROUND_HEATING_RATE

    def _arrangeIons(self) -> None:
        n = len(self._ions)
        for i, ion in enumerate(self._ions):
            # Centre ions symmetrically around the trap position.
            # Previous formula ``i - n/2`` produced offsets [-1, 0] for
            # two ions, placing the second ion exactly at the trap centre.
            # With large spacing this made _getEdgeIdxs pick that ion as
            # the "edge" for *both* crossings, forcing a spurious GateSwap.
            o = i - (n - 1) / 2
            ion.set(
                ion.idx,
                self.pos[0] + o * self._spacing * self._isHorizontal,
                self.pos[1] + o * self._spacing * (1 - self._isHorizontal),
                parent=self,
            )

    def addIon(
        self, ion: Ion, adjacentIon: Optional[Ion] = None, offset: int = 0
    ) -> None:
        super().addIon(ion, adjacentIon, offset)
        self._arrangeIons()

    def removeIon(self, ion: Optional[Ion] = None) -> Ion:
        ion = super().removeIon(ion)
        self._arrangeIons()
        return ion
    
    @property
    def allowedOperations(self) -> Sequence[Operations]:
        return [Operations.CRYSTAL_ROTATION, Operations.SPLIT, Operations.MOVE, Operations.MERGE, Operations.QUBIT_RESET, Operations.RECOOLING]


class ManipulationTrap(Trap):
    """Trap zone for gate operations with mode tracking.
    
    ManipulationTrap extends Trap to support 3N normal-mode structure tracking.
    When ions are added or removed, the mode structure is automatically recomputed
    (see ModeStructure in physics.py).
    
    Parameters
    ----------
    secular_frequencies : Tuple[float, float, float], optional
        Secular trap frequencies (ω_z, ω_x, ω_y) in Hz.
        Default (1 MHz, 5 MHz, 5 MHz).
    """
    DEFAULT_COLOR = "lightyellow"
    DEFAULT_ORIENTATION = False
    DEFAULT_SPACING = 10
    DEFAULT_CAPACITY = 3
    DEFAULT_LABEL = "MT"
    DEFAULT_SECULAR_FREQUENCIES = (1.0e6, 5.0e6, 5.0e6)

    def __init__(
        self,
        idx: int,
        x: int,
        y: int,
        ions: Sequence[Ion],
        color: str = DEFAULT_COLOR,
        isHorizontal: bool = DEFAULT_ORIENTATION,
        spacing=DEFAULT_SPACING,
        capacity: int = DEFAULT_CAPACITY,
        label: str = DEFAULT_LABEL,
        secular_frequencies: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        super().__init__(idx, x, y, ions, color, isHorizontal, spacing, capacity, label)
        self._secular_frequencies = secular_frequencies or self.DEFAULT_SECULAR_FREQUENCIES
        self._mode_structure: Optional[ModeStructure] = None
        # Compute initial mode structure if ions present
        self._recompute_modes()

    @property
    def secular_frequencies(self) -> Tuple[float, float, float]:
        """Secular trap frequencies (ω_z, ω_x, ω_y) in Hz."""
        return self._secular_frequencies

    @property
    def mode_structure(self) -> Optional[ModeStructure]:
        """Current 3N normal-mode structure for the ion crystal."""
        return self._mode_structure

    def _recompute_modes(self, old_modes: Optional[ModeStructure] = None,
                         kept_ion_indices: Optional[list] = None) -> None:
        """Recompute mode structure for the current ion crystal.

        If ``old_modes`` and ``kept_ion_indices`` are provided, phonon
        occupancies are redistributed via eigenvector overlap
        (see :meth:`ModeStructure.remap_after_split`).  Otherwise a
        fresh ground-state structure is created.
        """
        n = len(self.ions)
        if n == 0:
            self._mode_structure = None
            return
        wz, wx, wy = self._secular_frequencies
        if old_modes is not None and kept_ion_indices is not None:
            self._mode_structure = ModeStructure.remap_after_split(
                old_structure=old_modes,
                new_n_ions=n,
                kept_ion_indices=kept_ion_indices,
                axial_freq=wz,
                radial_freqs=(wx, wy),
            )
        else:
            self._mode_structure = ModeStructure.compute(
                n, axial_freq=wz, radial_freqs=(wx, wy),
            )

    def addIon(
        self, ion: Ion, adjacentIon: Optional[Ion] = None, offset: int = 0
    ) -> None:
        """Add an ion and recompute the 3N mode structure.

        Preserves phonon occupancy from the old crystal by passing
        all previous ion indices as ``kept_ion_indices``.
        """
        old_modes = self._mode_structure
        old_n = len(self.ions)
        super().addIon(ion, adjacentIon, offset)
        # All old ions are kept; they map to indices [0..old_n-1]
        kept = list(range(old_n)) if old_modes is not None else None
        self._recompute_modes(old_modes=old_modes, kept_ion_indices=kept)

    def removeIon(self, ion: Optional[Ion] = None) -> Ion:
        """Remove an ion and recompute the 3N mode structure.

        Preserves phonon occupancy from the old crystal.  The removed
        ion's index is excluded from ``kept_ion_indices``.
        """
        old_modes = self._mode_structure
        # Find the index of the ion being removed
        if ion is not None and ion in self.ions:
            removed_idx = self.ions.index(ion)
        else:
            # Default removal is the last ion
            removed_idx = len(self.ions) - 1 if self.ions else None
        old_n = len(self.ions)
        removed = super().removeIon(ion)
        if old_modes is not None and removed_idx is not None:
            kept = [i for i in range(old_n) if i != removed_idx]
            self._recompute_modes(old_modes=old_modes, kept_ion_indices=kept)
        else:
            self._recompute_modes()
        return removed

    def cool_trap(self) -> None:
        """Sympathetic recooling: reset all mode occupancies to zero.
        
        Only performs cooling if a cooling ion is present in the trap
        and mode tracking is enabled.
        """
        if self.hasCoolingIon and self._mode_structure is not None:
            self._mode_structure.cool_to_ground()
        # Also call parent coolTrap for backward compatibility
        self.coolTrap()

    @property
    def allowedOperations(self) -> Sequence[Operations]:
        return super().allowedOperations+[Operations.ONE_QUBIT_GATE, Operations.MEASUREMENT, Operations.TWO_QUBIT_MS_GATE, Operations.GATE_SWAP]

class StorageTrap(Trap):
    DEFAULT_COLOR = "grey"
    DEFAULT_ORIENTATION = False  # 0 for vertical, 1 for horizontal
    DEFAULT_SPACING = 10
    DEFAULT_CAPACITY = 5
    DEFAULT_LABEL = "ST"

    def __init__(self, idx: int, x: int, y: int, ions: Sequence[Ion], color: str = DEFAULT_COLOR, isHorizontal: bool = DEFAULT_ORIENTATION, spacing=DEFAULT_SPACING, capacity: int = DEFAULT_CAPACITY, label: str = DEFAULT_LABEL) -> None:
        super().__init__(idx, x, y, ions, color, isHorizontal, spacing, capacity, label)


class Crossing:
    DEFAULT_LABEL = "C"
    MOVE_AMOUNT = 8

    def __init__(
        self, idx: int, source: QCCDNode, target: QCCDNode, label: str = DEFAULT_LABEL
    ) -> None:
        self._idx: int = idx
        self._source: QCCDNode = source
        self._target: QCCDNode = target
        self._ion: Optional[Ion] = None
        self._ionAtSource: bool = False
        self._label = label

    def ionAt(self) -> QCCDNode:
        if not self._ion:
            raise ValueError(f"ionAt: no ion for crossing {self.idx}")
        return self._source if self._ionAtSource else self._target

    @property
    def pos(self) -> Tuple[int, int]:
        i1, i2 = self._getEdgeIdxs()
        x1, y1 = self._source.positions[i1]
        x2, y2 = self._target.positions[i2]
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def label(self) -> int:
        return (
            self._label
            + str(self._idx)
            + f" {self._source.label} to {self._target.label}"
        )

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def connection(self) -> Tuple[QCCDNode, QCCDNode]:
        return self._source, self._target

    def _getEdgeIdxs(self) -> Tuple[int, int]:
        permutations = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        if not self._source.ions:
            permutations[0][0] = 0
            permutations[1][0] = 0
        if not self._target.ions:
            permutations[0][1] = 0
            permutations[2][1] = 0
        pairs_distances = [
            (self._source.positions[i][0] - self._target.positions[j][0]) ** 2
            + (self._source.positions[i][1] - self._target.positions[j][1]) ** 2
            for (i, j) in permutations
        ]
        return permutations[np.argmin(pairs_distances)]

    def graphEdge(self) -> Tuple[int, int]:
        idx1, idx2 = self._getEdgeIdxs()
        return (self._source.nodes[idx1], self._target.nodes[idx2])

    def hasTrap(self, trap: Trap) -> None:
        if trap == self._source:
            return True
        elif trap == self._target:
            return True
        else:
            return False

    def hasJunction(self, junction: Junction) -> None:
        if junction == self._source:
            return True
        elif junction == self._target:
            return True
        else:
            return False

    def getEdgeIon(self, node: QCCDNode) -> Ion:
        if not node.ions:
            raise ValueError("getEdgeIon: no edge ions")
        ionIdx = node.nodes[self._getEdgeIdxs()[1 - (node == self._source)]]
        return [i for i in node.ions if i.idx == ionIdx][0]

    def setIon(self, ion: Ion, node: QCCDNode) -> None:
        if self._ion is not None:
            raise ValueError(f"setIon: crossing has not been cleared")
        self._ion = ion
        self._ionAtSource = self._source == node
        edgeIdxs = self._getEdgeIdxs()
        w = (1 + (self.MOVE_AMOUNT - 2) * (self._source == node)) / self.MOVE_AMOUNT
        x = self._source.positions[edgeIdxs[0]][0] * w + self._target.positions[
            edgeIdxs[1]
        ][0] * (1 - w)
        y = self._source.positions[edgeIdxs[0]][1] * w + self._target.positions[
            edgeIdxs[1]
        ][1] * (1 - w)
        self._ion.set(self._ion.idx, x, y, parent=self)

    def moveIon(self) -> None:
        if self.ion is None:
            raise ValueError(f"moveIon: no ion to move in crossing")
        node = self._target if self._ionAtSource else self._source
        ion = self.ion
        self.clearIon()
        self.setIon(ion, node)

    def clearIon(self) -> None:
        self._ion = None

    @property
    def ion(self) -> Optional[Ion]:
        return self._ion
    
    @property
    def allowedOperations(self) -> Sequence[Operations]:
        return [Operations.SPLIT, Operations.MOVE, Operations.MERGE, Operations.JUNCTION_CROSSING]


@dataclass
class QCCDWiseArch:
    m: int
    n: int
    k: int