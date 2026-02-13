
from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Mapping,
    Dict,
)
from matplotlib import pyplot as plt
import networkx as nx
from dataclasses import dataclass
from matplotlib.patches import Ellipse
from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *

class QCCDArch:
    SIZING = 1
    JUNCTION_SIZE = 800 * SIZING
    ION_SIZE = 800 * SIZING
    FONT_SIZE = 14 * SIZING
    WINDOW_SIZE = 30 * SIZING, 24 * SIZING
    TRAP_WIDTH = 15 * SIZING
    EDGE_WIDTH = SIZING

    HIGHLIGHT_COLOR = "yellow"
    HIGHLIGHT_NODE_SIZE = 4000 * SIZING
    JUNCTION_SHAPE = "s"
    ION_SHAPE = "o"
    DEFAULT_ALPHA = 0.5
    PADDING = 0.6

    N_ITERS = 5000

    def __init__(self):
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
        self._originalArrangement: Dict[Ion, Trap] = {}

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
            spacing=spacing * self.SIZING,
            capacity=capacity,
        )
        for ion in trap.ions:
            self._originalArrangement[ion] = trap
        self._manipulationTraps.append(trap)
        self._nextIdx += len(ions) + 1
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

    def display(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        title: str = "",
        operation: Optional[Operation] = None,
        show_junction: bool = True,
        showEdges: bool = True,
        showIons: bool = True,
        showLabels: bool = True,
        runOps: bool = False,
    ) -> None:
        pos = {}
        labels = {}
        operationNodes: List[List[int]] = []
        involvedIons: List[Sequence[Ion]] = []

        if operation is None:
            operations = []
        elif isinstance(operation, ParallelOperation):
            operations = operation.operations
            if runOps:
                operation.run()
                self.refreshGraph()
        else:
            operations = [operation]
            if runOps:
                for op in operations:
                    op.run()

                self.refreshGraph()

        for op in operations:
            operationNodes.append([])
            involvedIons.append(op.involvedIonsForLabel)

        for junction in self._junctions:
            pos[junction.nodes[0]] = junction.pos
            labels[junction.nodes[0]] = ""
            if show_junction:
                nx.draw_networkx_nodes(
                    self._graph,
                    pos,
                    ax=ax,
                    nodelist=[junction.nodes[0]],
                    node_color=[junction.color],
                    node_shape=self.JUNCTION_SHAPE,
                    node_size=self.JUNCTION_SIZE,
                )
            for n, ion in zip(junction.nodes[1:], junction.ions):
                pos[n] = ion.pos
                labels[n] = ion.label
                if showIons:
                    nx.draw_networkx_nodes(
                        self._graph,
                        pos,
                        ax=ax,
                        nodelist=[n],
                        node_color=[ion.color],
                        node_shape=self.ION_SHAPE,
                        node_size=self.ION_SIZE,
                    )
                for nodes, ions in zip(operationNodes, involvedIons):
                    if ion in ions:
                        nodes.append(n)
            if showLabels:
                x = junction.pos[0]
                y = junction.pos[1]
                ax.text(
                    x,
                    y,
                    junction.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        for c in self._crossings:
            if c.ion:
                pos[c.ion.idx] = c.ion.pos
                labels[c.ion.idx] = c.ion.label
                if showIons:
                    nx.draw_networkx_nodes(
                        self._graph,
                        pos,
                        ax=ax,
                        nodelist=[c.ion.idx],
                        node_color=[c.ion.color],
                        node_shape=self.ION_SHAPE,
                        node_size=self.ION_SIZE,
                    )
                for nodes, ions in zip(operationNodes, involvedIons):
                    if c.ion in ions:
                        nodes.append(c.ion.idx)

        for t in self._manipulationTraps:
            if not isinstance(t, Trap):
                continue
            pos[t.nodes[0]] = t.pos
            labels[t.nodes[0]] = ""
            colors = {}
            for n, ion in zip(t.nodes[1:], t.ions):
                pos[n] = ion.pos
                labels[n] = ion.label
                colors[n] = ion.color
                for nodes, ions in zip(operationNodes, involvedIons):
                    if ion in ions:
                        nodes.append(n)
            if showIons:
                nx.draw_networkx_nodes(
                    self._graph,
                    pos,
                    ax=ax,
                    nodelist=t.nodes[1:],
                    node_color=colors.values(),
                    node_shape=self.ION_SHAPE,
                    node_size=self.ION_SIZE,
                )

        for trap in self._manipulationTraps:
            if not isinstance(trap, Trap):
                nx.draw_networkx_edges(
                    self._graph,
                    pos,
                    edgelist=trap[0],
                    ax=ax,
                    alpha=self.DEFAULT_ALPHA,
                    edge_color='red',
                    width=trap[1],
                )
                continue
            if showIons:
                nx.draw_networkx_edges(
                    self._graph,
                    pos,
                    edgelist=self._trapEdges[trap.idx],
                    ax=ax,
                    alpha=self.DEFAULT_ALPHA,
                    edge_color=trap.color,
                    width=self.TRAP_WIDTH,
                )
            if showLabels:
                x = trap.pos[0]
                y = trap.pos[1]
                ax.text(
                    x,
                    y,
                    trap.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        if showEdges:
            nx.draw_networkx_edges(
                self._graph,
                pos,
                edgelist=self._crossingEdges.keys(),
                ax=ax,
                alpha=self.DEFAULT_ALPHA,
                width=self.EDGE_WIDTH,
            )
        if showLabels:
            for e in self._crossings:
                ax.text(
                    *e.pos,
                    e.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        if showIons:
            nx.draw_networkx_labels(
                self._graph, pos, ax=ax, labels=labels, font_size=self.FONT_SIZE
            )

        for nodes, op in zip(operationNodes, operations):
            if nodes:
                xVals = [pos[node][0] for node in nodes]
                yVals = [pos[node][1] for node in nodes]
                padding = self.SIZING * self.PADDING
                xMin, xMax = min(xVals) - padding, max(xVals) + padding
                yMin, yMax = min(yVals) - padding, max(yVals) + padding
                width = xMax - xMin
                height = yMax - yMin
                xLabel = (xMin + xMax) / 2
                yLabel = (yMin + yMax) / 2
                ellip = Ellipse(
                    (xLabel, yLabel),
                    width,
                    height,
                    edgecolor=op.color,
                    alpha=self.DEFAULT_ALPHA,
                    facecolor=op.color,
                )
                ax.add_patch(ellip)
                xLabel = (xMin + xMax) / 2
                yLabel = (yMin + yMax) / 2
                ax.text(
                    xLabel,
                    yLabel,
                    op.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        ax.set_title(title, fontsize=self.FONT_SIZE*5)
        n = len(fig.axes)
        fig.set_size_inches(self.WINDOW_SIZE[0]*n, self.WINDOW_SIZE[1])

