
from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Set,
    Dict,
)
import networkx as nx
from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *
from src.utils.qccd_arch import *


def ionRouting(
    qccdArch: QCCDArch,
    operations: Sequence[QubitOperation],
    trapCapacity: int
) -> Tuple[Sequence[Operation], Sequence[int]]:
    twoqubitGatesForAncillaIons: Dict[int, List[TwoQubitMSGate]] = {}
    for op in operations:
        if isinstance(op, TwoQubitMSGate):
            ion1, ion2 = op.ions
            ancilla, data = sorted(
                (ion1, ion2), key=lambda ion: ion.label[0]=='D'
            )
            trap = ancilla.parent
            if ancilla.idx in twoqubitGatesForAncillaIons:
                twoqubitGatesForAncillaIons[ancilla.idx].append(op)
            else:
                twoqubitGatesForAncillaIons[ancilla.idx] = [op]
    
    opPriorities: Dict[Operation, int] = {op: i for i, op in enumerate(operations)}

    allOps: List[Operation] = []
    barriers: List[int] = []
    operationsLeft = list(operations)
    toMoveCandidates: Dict[int, TwoQubitMSGate] = {}

    while operationsLeft:


        # Run the operations that do not need routing
        
        while True:
            toRemove: List[Operation] = []
            ionsInvolved: Set[Ion] = set()
            for op in operationsLeft:
                trap = op.getTrapForIons()
                if ionsInvolved.isdisjoint(op.ions) and trap:
                    op.setTrap(trap)
                    toRemove.append(op)
                ionsInvolved = ionsInvolved.union(op.ions)

            for op in toRemove:
                op.run()
                allOps.append(op)
                operationsLeft.remove(op)
                if isinstance(op, TwoQubitMSGate):
                    ion1, ion2 = op.ions
                    if ion1.idx in twoqubitGatesForAncillaIons:
                        twoqubitGatesForAncillaIons[ion1.idx].remove(op)
                    else:
                        twoqubitGatesForAncillaIons[ion2.idx].remove(op)
            
            if len(toRemove) == 0:
                break

        # Determine the operations that need routing
        for ancillaIdx in twoqubitGatesForAncillaIons.keys():
            if ancillaIdx in toMoveCandidates:
                continue
            if len(twoqubitGatesForAncillaIons[ancillaIdx]) == 0:
                continue
            gate = twoqubitGatesForAncillaIons[ancillaIdx][0]
            trap = gate.getTrapForIons()
            if trap:
                continue
            toMoveCandidates[ancillaIdx] = twoqubitGatesForAncillaIons[ancillaIdx].pop(0)

        # move operations with priority according to the original happens before
        toMove = sorted([(k,o) for k,o in toMoveCandidates.items()], key=lambda ko: opPriorities[ko[1]])

        crossingsUsed: Set[Crossing] = set()
        qccdNodesFull: Set[QCCDNode] = set()
        # ancillaIdx: op, pathChosen, destTrap, goBack
        movements: Dict[int, Tuple[TwoQubitMSGate, List[QCCDNode], Trap]] = {}


        # ionsInvolved = set()
        for ancillaIdx, op in toMove:
            ion1, ion2 = op.ions

            ancilla, data = (ion1, ion2) if ion1.idx == ancillaIdx else (ion2, ion1)
            trap = data.parent
            if not isinstance(trap, Trap):
                raise ValueError(f"Data Ion not in a trap {trap}")
            
            src = ancillaIdx
            dest = trap.idx
            paths = list(nx.all_shortest_paths(qccdArch.graph, src, dest))
            
            qccdNodesChosen: List[QCCDNode] = []
            crossingsChosen: List[Crossing] = []

            for path in paths:
                crossingsInPath: List[Crossing] = []
                for n1, n2 in zip(path[:-1], path[1:]):
                    if (n1, n2) not in qccdArch.crossingEdges:
                        continue
                    crossingsInPath.append(qccdArch.crossingEdges[(n1,n2)])

                qccdNodesInPath: List[QCCDNode] = []
                for n in path: 
                    nd = qccdArch.nodes[n] if n in qccdArch.nodes else qccdArch.ions[n].parent
                    if nd not in qccdNodesInPath:
                        qccdNodesInPath.append(nd)   

                qccdNodesInPathFull: List[QCCDNode] = []
                # Do not include source since the source is going to decrease in ions or stay the same at all points
                for qccdNode in qccdNodesInPath[1:]:
                    if isinstance(qccdNode, Junction) and qccdNode.numIons==1:
                        qccdNodesInPathFull.append(qccdNode)      
                    elif qccdNode.numIons == trapCapacity:
                        qccdNodesInPathFull.append(qccdNode)        
                            
                if crossingsUsed.isdisjoint(crossingsInPath) and qccdNodesFull.isdisjoint(qccdNodesInPathFull):
                    qccdNodesChosen = qccdNodesInPath
                    crossingsChosen = crossingsInPath
                    break 

            # unable to complete move operation this time round
            if len(qccdNodesChosen) == 0:
                continue 

            # able to complete move operation 
            toMoveCandidates.pop(ancillaIdx)
            movements[ancillaIdx]=(op, qccdNodesChosen, trap)
            # remove crossings that are reserved
            crossingsUsed = crossingsUsed.union(crossingsChosen)
            # increment traps and junctions number of ions by 1 EXCEPT the source
            # remove traps and junctions if currently at capacity EXCEPT the source 
            for qccdNode in qccdNodesChosen[1:]:
                qccdNode.numIons+=1
                if isinstance(qccdNode, Junction) and qccdNode.numIons==1:
                    qccdNodesFull.add(qccdNode)
                elif qccdNode.numIons == trapCapacity:
                    qccdNodesFull.add(qccdNode)


        # if destination trap is at capacity then we need to send the ancilla back to original trap at start of barrier to maintain invariant
        toForward: Dict[TwoQubitMSGate, Tuple[int, List[QCCDNode], Trap, Optional[Trap]]] = {}
        for ancillaIdx, (op, qccdNodes, destTrap) in movements.items():
            if destTrap.numIons == trapCapacity:
                goBackTrap=qccdNodes[0]
                for nd in qccdNodes[::-1][:-1]:
                    # note we have already reserved the goBackTrap (qccdNode.numIons+=1) so no need to increment again
                    if nd.numIons <= trapCapacity-1 and isinstance(nd, Trap):
                        goBackTrap = nd
                destTrap.numIons -= 1
                # no need to increment srcTrap.numIons because we never decremented it in the first place
            else: 
                goBackTrap=None
            toForward[op] = (ancillaIdx, qccdNodes, goBackTrap)

        startedGoingBack = {op: False for op in toForward.keys()}
        while toForward:
            ionsInvolved = set()
            orderedToForward = sorted([(o, rc) for o, rc in toForward.items()], key=lambda orc: opPriorities[orc[0]])
            for op, (ancillaIdx, qccdNodes, goBackTrap) in orderedToForward:
                if not ionsInvolved.isdisjoint(op.ions):
                    continue 

                ionsInvolvedNow = [qccdArch.ions[ancillaIdx]]

                n1 = ancillaIdx
                n1Idx = qccdArch.ions[ancillaIdx].parent.idx
                trap = qccdNodes[-1]
                srcTrap = goBackTrap
            
                if n1Idx == trap.idx and not startedGoingBack[op]:
                    op.setTrap(trap)
                    op.run()
                    allOps.append(op)
                    operationsLeft.remove(op)
                    ionsInvolved = ionsInvolved.union(op.ions)
                    if goBackTrap is not None:
                        startedGoingBack[op] = True
                    else:
                        toForward.pop(op)
                        continue
                elif startedGoingBack[op] and n1Idx == srcTrap.idx:
                    toForward.pop(op)
                    continue


                if startedGoingBack[op]:
                    n2Idx = [dn.idx for sn, dn in zip(qccdNodes[::-1][:-1], qccdNodes[::-1][1:]) if sn.idx == n1Idx][0]
                else:
                    n2Idx = [dn.idx for sn, dn in zip(qccdNodes[:-1], qccdNodes[1:]) if sn.idx == n1Idx][0]
                forwardingPath = nx.shortest_path(qccdArch.graph, n1, n2Idx)
                ms: List[Operation] = []
                for n1, n2 in zip(forwardingPath[:-1], forwardingPath[1:]):
                    ms.extend(qccdArch.graph.edges[n1, n2]["operations"])
                    
                for m in ms:
                    if isinstance(m, CrystalOperation):
                        ionsInvolvedNow.extend(m.ionsInfluenced)
                    # Hack FIXME
                    if isinstance(m, GateSwap) and m._ion1==m._ion2:
                        continue
                    m.run()
                    allOps.append(m)
                
                qccdArch.refreshGraph()
                ionsInvolved = ionsInvolved.union(ionsInvolvedNow)

        barriers.append(len(allOps))

    return allOps, barriers


