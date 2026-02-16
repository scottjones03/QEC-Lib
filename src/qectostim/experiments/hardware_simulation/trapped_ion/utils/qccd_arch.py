
from typing import (
    Any,
    Sequence,
    List,
    Tuple,
    Optional,
    Mapping,
    Dict,
    TYPE_CHECKING,
)
import numpy as np
import numpy.typing as npt
import networkx as nx
from dataclasses import dataclass

if TYPE_CHECKING:
    from .trapped_ion_compiler import TrappedIonCompiler

from .qccd_nodes import *
from .qccd_operations import *
from .qccd_operations_on_qubits import *
from .physics import DEFAULT_CALIBRATION
from qectostim.experiments.hardware_simulation.core.architecture import (
    ReconfigurableArchitecture,
    ConnectivityGraph,
    PhysicalConstraints,
    TransportCost,
    ReconfigurationPlan,
    Zone,
    ZoneType,
)
from qectostim.experiments.hardware_simulation.core.gates import (
    NativeGateSet,
    GateSpec,
    GateType,
)


class QCCDArch(ReconfigurableArchitecture):
    """QCCD trapped-ion architecture.

    Manages the junction-based topology (traps, junctions, crossings)
    and the routing graph used by compilers.

    Inherits from :class:`ReconfigurableArchitecture` so that core
    algorithms can query zones, connectivity, and transport costs in a
    platform-agnostic way.

    All visualisation-specific code has been moved to ``trapped_ion.viz``.
    """

    def __init__(self, spacing: float = 1.0):
        super().__init__(name="QCCD", num_qubits=0)
        self.spacing = spacing
        self._trapEdges: Mapping[int, Sequence[Tuple[int, int]]] = {}
        self._crossingEdges: Mapping[Tuple[int, int], Crossing] = {}
        self._crossings: List[Crossing] = []
        self._manipulationTraps: List[ManipulationTrap] = []
        self._junctions: List[Junction] = []
        self._nextIdx = 0
        self._routingTable: Mapping[int, Mapping[int, Sequence[Operation]]] = {}
        self._graph: nx.DiGraph = nx.DiGraph()
        self._inActiveEdges: List[int] = []
        self._centralities: Mapping
        self._originalArrangement: Dict = {}  # mixed: Trap→[Ion] and Ion→Trap

    @property
    def graph(self):
        return self._graph
    
    @property
    def crossingEdges(self):
        return self._crossingEdges

    @property
    def routingTable(self):
        return self._routingTable

    @property
    def ions(self) -> Mapping[int, Ion]:
        ions = {}
        for t in self._manipulationTraps:
            ions.update([(ion.idx, ion) for ion in t.ions])
        for j in self._junctions:
            ions.update([(ion.idx, ion) for ion in j.ions])
        for c in self._crossings:
            if c.ion:
                ions[c.ion.idx] = c.ion
        return ions

    @property
    def nodes(self) -> Mapping[int, QCCDNode]:
        cs = {}
        for t in self._manipulationTraps:
            cs[t.idx] = t
        for j in self._junctions:
            cs[j.idx] = j
        return cs

    def addEdge(self, source: QCCDNode, target: QCCDNode) -> Crossing:
        crossing = Crossing(self._nextIdx, source, target)
        self._crossings.append(crossing)
        self._nextIdx += 1
        return crossing

    def addManipulationTrap(
        self,
        x: int,
        y: int,
        ions: Sequence[Ion],
        color: str = ManipulationTrap.DEFAULT_COLOR,
        isHorizontal: bool = ManipulationTrap.DEFAULT_ORIENTATION,
        spacing: int = ManipulationTrap.DEFAULT_SPACING,
        capacity: int = ManipulationTrap.DEFAULT_CAPACITY,
    ) -> Trap:
        trap = ManipulationTrap(
            self._nextIdx,
            x,
            y,
            ions,
            color=color,
            isHorizontal=isHorizontal,
            spacing=spacing,
            capacity=capacity,
        )
        for ion in trap.ions:
            self._originalArrangement[ion] = trap
        self._manipulationTraps.append(trap)
        self._nextIdx += len(ions) + 1
        self.num_qubits = len(self.ions)
        return trap

    def addJunction(
        self,
        x: int,
        y: int,
        color: str = Junction.DEFAULT_COLOR,
        label: str = Junction.DEFAULT_LABEL,
        capacity: int = Junction.DEFAULT_CAPACITY,
    ) -> Junction:
        junction = Junction(
            self._nextIdx, x, y, color=color, label=label, capacity=capacity
        )
        self._junctions.append(junction)
        self._nextIdx += 1
        return junction

    def refreshGraph(self) -> None:
        g = nx.DiGraph()

        for j in self._junctions:
            j.subgraph(g)
            j.numIons = len(j.ions)

        for t in self._manipulationTraps:
            t.subgraph(g)
            t.numIons = len(t.ions)

        for trap in self._manipulationTraps:
            trapEdges = []
            for ion1 in trap.ions:
                g.add_edge(ion1.idx, trap.idx, operations=[])
                for ion2 in trap.ions:
                    if ion1 == ion2:
                        continue
                    trapEdges.append((ion1.idx, ion2.idx))
                    trapEdges.append((ion2.idx, ion1.idx))
                    g.add_edge(
                        ion1.idx,
                        ion2.idx,
                        operations=[GateSwap.physicalOperation(trap=trap, ion1=ion1, ion2=ion2)],
                    )
                    g.add_edge(
                        ion2.idx,
                        ion1.idx,
                        operations=[GateSwap.physicalOperation(trap=trap, ion1=ion2, ion2=ion1)],
                    )
            self._trapEdges[trap.idx] = trapEdges

        crossingEdges = {}
        for crossing in self._crossings:
            n1, n2 = crossing.connection
            n1Idx = crossing.getEdgeIon(n1).idx if n1.ions else n1.idx
            n2Idx = crossing.getEdgeIon(n2).idx if n2.ions else n2.idx
            crossingEdges[(n1Idx, n2Idx)] = crossing
            crossingEdges[(n2Idx, n1Idx)] = crossing
            ion1 = crossing.getEdgeIon(n1) if n1.ions else None
            ion2 = crossing.getEdgeIon(n2) if n2.ions else None
            doRotation1 = [GateSwap.physicalOperation(trap=n1,ion1=ion1,ion2=ion1)] if len(n1.ions)==1 else []
            doRotation2 = [GateSwap.physicalOperation(trap=n2,ion1=ion2,ion2=ion2)] if len(n2.ions)==1 else []
            if isinstance(n1, Trap) and isinstance(n2, Junction):
                ops1 = doRotation1+[
                    Split.physicalOperation(n1, crossing, ion1),
                    Move.physicalOperation(crossing, ion1),
                    JunctionCrossing.physicalOperation(n2, crossing, ion1),
                ]
                ops2 = [
                    JunctionCrossing.physicalOperation(n2, crossing, ion2),
                    Move.physicalOperation(crossing, ion2),
                    Merge.physicalOperation(n1, crossing, ion2),
                ]
            elif isinstance(n1, Junction) and isinstance(n2, Trap):
                ops1 = [
                    JunctionCrossing.physicalOperation(n1, crossing, ion1),
                    Move.physicalOperation(crossing, ion1),
                    Merge.physicalOperation(n2, crossing, ion1),
                ]
                ops2 = doRotation2+[
                    Split.physicalOperation(n2, crossing, ion2),
                    Move.physicalOperation(crossing, ion2),
                    JunctionCrossing.physicalOperation(n1, crossing, ion2),
                ]
            elif isinstance(n1, Junction) and isinstance(n2, Junction):
                ops1 = [
                    JunctionCrossing.physicalOperation(n1, crossing, ion1),
                    Move.physicalOperation(crossing, ion1),
                    JunctionCrossing.physicalOperation(n2, crossing, ion1),
                ]
                ops2 = [
                    JunctionCrossing.physicalOperation(n2, crossing, ion2),
                    Move.physicalOperation(crossing, ion2),
                    JunctionCrossing.physicalOperation(n1, crossing, ion2),
                ]
            else:
                ops1 = doRotation1+[
                    Split.physicalOperation(n1, crossing, ion1),
                    Move.physicalOperation(crossing, ion1),
                    Merge.physicalOperation(n2, crossing, ion1),
                ]
                ops2 = doRotation2+[
                    Split.physicalOperation(n2, crossing, ion2),
                    Move.physicalOperation(crossing, ion2),
                    Merge.physicalOperation(n1, crossing, ion2),
                ]
            g.add_edge(n1Idx, n2Idx, operations=ops1)
            g.add_edge(n2Idx, n1Idx, operations=ops2)
            if crossing.ion:
                g.add_node(crossing.ion.idx, pos=crossing.ion.pos)
        self._crossingEdges = crossingEdges

        for n2Idx in self._inActiveEdges:
            graphEdges = [
                (u, v)
                for (u, v), crossing in self._crossingEdges.items()
                if self.nodes[n2Idx] in crossing.connection
                and (v in self.nodes[n2Idx].nodes)
            ]
            g.remove_edges_from(graphEdges)
        self._graph = g
        self._centralities = None
        self._routingTable = {ion.idx: {} for ion in self.ions.values()}
        self.num_qubits = len(self.ions)

    # =========================================================================
    # ReconfigurableArchitecture / HardwareArchitecture abstract methods
    # =========================================================================

    def native_gate_set(self) -> NativeGateSet:
        """Trapped-ion native gate set: MS, RX, RY, M, R."""
        cal = DEFAULT_CALIBRATION
        gates = {
            "MS": GateSpec(
                "MS", GateType.TWO_QUBIT, 2,
                is_clifford=False, is_native=True,
                metadata={"time_s": cal.ms_gate_time},
            ),
            "RX": GateSpec(
                "RX", GateType.SINGLE_QUBIT, 1,
                parameters=("theta",), is_clifford=False, is_native=True,
                metadata={"time_s": cal.single_qubit_gate_time},
            ),
            "RY": GateSpec(
                "RY", GateType.SINGLE_QUBIT, 1,
                parameters=("theta",), is_clifford=False, is_native=True,
                metadata={"time_s": cal.single_qubit_gate_time},
            ),
        }
        return NativeGateSet(platform="trapped_ion", gates=gates)

    def connectivity_graph(self) -> ConnectivityGraph:
        """Build a :class:`ConnectivityGraph` from the current topology."""
        cg = ConnectivityGraph()
        for trap in self._manipulationTraps:
            cg.add_zone(Zone(
                id=str(trap.idx),
                zone_type=ZoneType.GATE,
                capacity=trap.capacity,
                position=trap.pos,
            ))
        for junction in self._junctions:
            cg.add_zone(Zone(
                id=str(junction.idx),
                zone_type=ZoneType.JUNCTION,
                capacity=junction.capacity,
                position=junction.pos,
            ))
        for crossing in self._crossings:
            n1, n2 = crossing.connection
            cg.add_connection(str(n1.idx), str(n2.idx))
        return cg

    def physical_constraints(self) -> PhysicalConstraints:
        """Physical constraints derived from :data:`DEFAULT_CALIBRATION`."""
        cal = DEFAULT_CALIBRATION
        return PhysicalConstraints(
            max_parallel_2q_gates=1,
            max_qubits=self.num_qubits,
            t2_time=cal.t2_time * 1e6,  # convert s → μs
            gate_times=cal.gate_times_us(),
            readout_time=cal.measurement_time * 1e6,
            reset_time=cal.reset_time * 1e6,
            transport_time=cal.shuttle_time * 1e6,
        )

    def zone_types(self) -> List[ZoneType]:
        """Zone types present in a QCCD architecture."""
        return [ZoneType.GATE, ZoneType.JUNCTION]

    def can_interact(self, qubit1: int, qubit2: int) -> bool:
        """Two ions can interact if they share the same manipulation trap."""
        all_ions = self.ions
        ion1 = all_ions.get(qubit1)
        ion2 = all_ions.get(qubit2)
        if ion1 is None or ion2 is None:
            return False
        return (
            ion1.parent is not None
            and isinstance(ion1.parent, Trap)
            and ion1.parent is ion2.parent
        )

    # --- ReconfigurableArchitecture abstract methods ---

    def transport_cost(
        self,
        source_zone: str,
        target_zone: str,
        num_qubits: int = 1,
    ) -> TransportCost:
        """Estimate transport cost between two zones via graph shortest-path."""
        cg = self.connectivity_graph()
        try:
            path = cg.shortest_path(source_zone, target_zone)
        except nx.NetworkXNoPath:
            return TransportCost(
                time_us=float("inf"),
                fidelity_loss=1.0,
                num_operations=0,
                path=[],
            )
        cal = DEFAULT_CALIBRATION
        hops = len(path) - 1
        # Each hop involves split + shuttle + junction crossing + merge
        time_per_hop_us = (
            cal.split_time + cal.shuttle_time + cal.junction_time + cal.merge_time
        ) * 1e6
        total_time = hops * time_per_hop_us * num_qubits
        fidelity_loss = 1.0 - (1.0 - cal.heating_rate * cal.shuttle_time) ** hops
        return TransportCost(
            time_us=total_time,
            fidelity_loss=fidelity_loss,
            num_operations=hops * 4,
            path=path,
        )

    def reconfiguration_plan(
        self,
        target_layout: Dict[int, str],
    ) -> ReconfigurationPlan:
        """Compute a reconfiguration plan.

        For simple cases this computes individual transport costs.
        Full SAT-based reconfiguration is handled by
        :class:`GlobalReconfigurations`.
        """
        operations: List[Any] = []
        total_time = 0.0
        total_fidelity_loss = 0.0
        for qubit_id, target_zone in target_layout.items():
            current = self.current_zone(qubit_id)
            if current == target_zone:
                continue
            cost = self.transport_cost(current, target_zone)
            operations.append({
                "qubit": qubit_id,
                "from": current,
                "to": target_zone,
                "cost": cost,
            })
            total_time += cost.time_us
            total_fidelity_loss += cost.fidelity_loss
        return ReconfigurationPlan(
            operations=operations,
            total_time_us=total_time,
            total_fidelity_loss=total_fidelity_loss,
        )

    def can_colocate(self, qubit_ids: List[int], zone_id: str) -> bool:
        """Check if *qubit_ids* can all fit in *zone_id*."""
        node_idx = int(zone_id)
        node = self.nodes.get(node_idx)
        if node is None:
            return False
        return len(qubit_ids) <= node.capacity

    def current_zone(self, qubit_id: int) -> str:
        """Return the zone (trap/junction) string-id for *qubit_id*."""
        ion = self.ions.get(qubit_id)
        if ion is None:
            raise ValueError(f"Unknown qubit id {qubit_id}")
        parent = ion.parent
        if parent is None:
            raise ValueError(f"Ion {qubit_id} has no parent zone")
        return str(parent.idx)

    def qubits_in_zone(self, zone_id: str) -> List[int]:
        """Return qubit ids currently in *zone_id*."""
        node_idx = int(zone_id)
        node = self.nodes.get(node_idx)
        if node is None:
            return []
        return [ion.idx for ion in node.ions]

    # =========================================================================
    # Shared helpers for architecture subclasses
    # =========================================================================

    def _gridToCoordinate(
        self, pos: Tuple[int, int], trapCapacity: int
    ) -> npt.NDArray[np.float64]:
        """Convert a grid position to (x, y) coordinates."""
        return np.array(pos) * (trapCapacity + 1) * self.spacing

    def resetArrangement(self) -> "QCCDArch":
        """Clear all nodes/crossings and restore the original ion arrangement."""
        for node in self.nodes.values():
            while node.ions:
                node.removeIon(node.ions[0])

        for crossing in self._crossings:
            if crossing.ion:
                crossing.clearIon()

        # _originalArrangement has Trap→[Ion] entries (from architectures)
        # and Ion→Trap entries (from addManipulationTrap). Use Trap→[Ion].
        for key, val in self._originalArrangement.items():
            if not isinstance(val, list):
                continue  # skip Ion→Trap entries
            for i, ion in enumerate(val):
                key.addIon(ion, offset=i)
                ion.addMotionalEnergy(-ion.motionalMode)
        return self

    def create_default_compiler(self) -> "TrappedIonCompiler":
        """Create the default trapped-ion compiler."""
        from .trapped_ion_compiler import TrappedIonCompiler
        
        is_wise = hasattr(self, "wise_config")
        wise_config = getattr(self, "wise_config", None)
        return TrappedIonCompiler(
            self,
            is_wise=is_wise,
            wise_config=wise_config,
        )