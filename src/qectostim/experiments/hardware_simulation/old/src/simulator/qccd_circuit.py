
import numpy as np
import numpy.typing as npt
from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Mapping,
    Dict,
)
import stim
import pymatching
from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *
from src.utils.qccd_arch import *
from src.compiler.qccd_parallelisation import *
from src.compiler.qccd_qubits_to_ions import *
from src.compiler.qccd_ion_routing import *
from src.compiler.qccd_WISE_ion_route import *
# from src.Moveless_QCCDSimWork.machine import Machine, machine_graph_to_labeled_coordinates
# from src.Moveless_QCCDSimWork.parse import InputParse
# from src.Moveless_QCCDSimWork.mappers import QubitMapGreedy
# from src.Moveless_QCCDSimWork import customScheduler
from src.Moveless_QCCDSimWork.analyzer import *
import logging
from multiprocessing import get_logger

OPEN_QASM_VERSION = 2
class QCCDCircuit(stim.Circuit):
    DATA_QUBIT_COLOR = "lightblue"
    MEASUREMENT_QUBIT_COLOR = "red"
    PLACEMENT_ION = ("grey", "P")
    TRAP_COLOR = "grey"
    JUNCTION_COLOR = "orange"
    SPACING = 20


    start_score: int = 1
    score_delta: int = 2
    joinDisjointClusters: bool = False
    minIters: int = 1_000
    maxIters: int = 10_000


    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ionMapping: Dict[int, Tuple[Ion, Tuple[int, int]]] = {}
        self._measurementIons: List[Ion] = []
        self._dataIons: List[Ion] = []
        self._originalArrangement: Dict[Trap, Sequence[Ion]] = {}
        self._arch: QCCDArch

    @classmethod
    def generated(cls, *args, **kwargs) -> "QCCDCircuit":
        return QCCDCircuit(stim.Circuit.generated(*args, **kwargs).__str__())

    def circuitString(self, include_annotation: bool = False) -> Tuple[Sequence[str], Sequence[Sequence[Tuple[int, int]]]]:
        instructions = (
            self.flattened().decomposed().without_noise().__str__().splitlines()
        )
        newInstructions = []
        toMoves = []
        for i in instructions:
            qubits = i.rsplit(" ")[1:]
            if i.startswith("DETECTOR") or i.startswith("TICK") or i.startswith("OBSERVABLE"):
                if include_annotation:
                    newInstructions.append(i)
                continue
            elif i[0] in ("R", "H", "M"):
                for qubit in qubits:
                    newInstructions.append(f"{i[0]} {qubit}")
                # newInstructions.append("BARRIER")
            elif any(i.startswith(s) for s in stim.gate_data("cnot").aliases):
                toMove = []
                for i in range(int(len(qubits) / 2)):
                    newInstructions.append(f"CNOT {qubits[2*i]} {qubits[2*i+1]}")
                    toMove.append((qubits[2*i], qubits[2*i+1]))
                toMoves.append(toMove)
                newInstructions.append("BARRIER")
            elif any(i.startswith(s) for s in stim.gate_data("cz").aliases):
                toMove = []
                for i in range(int(len(qubits) / 2)):
                    newInstructions.append(f"CZ {qubits[2*i]} {qubits[2*i+1]}")
                    toMove.append((qubits[2*i], qubits[2*i+1]))
                toMoves.append(toMove)
                newInstructions.append("BARRIER")
            else:
                newInstructions.append(i)
        return newInstructions, toMoves

    @property
    def ionMapping(self) -> Mapping[int, Tuple[Ion, Tuple[int, int]]]:
        return self._ionMapping

    def _parseCircuitString(self, dataQubitsIdxs: Optional[Sequence[int]]=None) -> Tuple[Sequence[QubitOperation], Sequence[int], Sequence[Sequence[TwoQubitMSGate]]]:
        instructions, toMoves = self.circuitString()

        self._measurementIons = []
        self._ionMapping = {}
        self._dataIons = []

        for j, i in enumerate(instructions):
            if not i.startswith("QUBIT_COORDS"):
                break
            coords = tuple(
                map(int, i.removeprefix("QUBIT_COORDS(").split(")")[0].split(","))
            )
            idx = int(i.split(" ")[-1])
            if dataQubitsIdxs is not None or (coords[0]%2)==0:
                ion = QubitIon(self.MEASUREMENT_QUBIT_COLOR, label="M")
                ion.set(ion.idx, *coords)
                self._ionMapping[idx] = ion, coords
                self._measurementIons.append(ion)
            else:
                ion = QubitIon(self.DATA_QUBIT_COLOR, label="D")
                ion.set(ion.idx, *coords)
                self._ionMapping[idx] = ion, coords
                self._dataIons.append(ion)


        instructions = instructions[j:]
        operations = []
        barriers = []
        dataQubits = []
        toMoveOps = [[] for _ in range(len(toMoves))]
        toMoveIdx = 0
        # TODO establish correct mapping of qubit operations from QIP toolkit with references
        for j, i in enumerate(instructions):
            if i.startswith("BARRIER"):
                barriers.append(len(operations))
                continue
            if not ( i[0] in ("M", "H", "R") or i.startswith("CNOT")):
                continue
            idx = int(i.split(" ")[1])
            ion = self._ionMapping[idx][0]
            if i[0] == "M":
                operations.append(Measurement.qubitOperation(ion))
                # if dataQubitsIdxs is None:
                #     # dataQubits.append(ion) # data qubits are the ones measured at the end
                #     dataQubits.clear()
            elif i[0] == "H":
                # page 80 https://iontrap.umd.edu/wp-content/uploads/2013/10/FiggattThesis.pdf
                operations.extend([
                    YRotation.qubitOperation(ion),
                    XRotation.qubitOperation(ion)
                ])
                # if dataQubitsIdxs is None:
                #     # dataQubits.clear()
                #     dataQubits.append(ion) # data qubits are the ones with hadamard at the end
            elif i[0] == "R":
                operations.append(QubitReset.qubitOperation(ion))
                if dataQubitsIdxs is None:
                    dataQubits.clear()
            elif i.startswith("CNOT"):
                idx2 = int(i.split(" ")[2])
                ion2 = self._ionMapping[idx2][0]
                # Fig 4. https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
                operations.extend([
                    YRotation.qubitOperation(ion),
                    XRotation.qubitOperation(ion),
                    XRotation.qubitOperation(ion2),
                    TwoQubitMSGate.qubitOperation(
                        ion, ion2
                    ),
                    YRotation.qubitOperation(ion)
                ])
                # if dataQubitsIdxs is None:
                #     dataQubits.clear()
                toMoveOps[toMoveIdx].append(operations[-2])
                if len(toMoveOps[toMoveIdx])==len(toMoves[toMoveIdx]):
                    toMoveIdx+=1
            elif i.startswith("CZ"):
                idx2 = int(i.split(" ")[2])
                ion2 = self._ionMapping[idx2][0]
                # Fig 4. https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
                # Adding I H = I (RY RX) before and after CNOT then cancelling (IH CNOT IH)
                operations.extend([
                    YRotation.qubitOperation(ion),
                    XRotation.qubitOperation(ion),
                    YRotation.qubitOperation(ion2),
                    XRotation.qubitOperation(ion2),
                    TwoQubitMSGate.qubitOperation(
                        ion, ion2
                    ),
                    YRotation.qubitOperation(ion),
                    YRotation.qubitOperation(ion2),
                    XRotation.qubitOperation(ion2),
                ])
                # if dataQubitsIdxs is None:
                #     dataQubits.clear()
                toMoveOps[toMoveIdx].append(operations[-4])
                if len(toMoveOps[toMoveIdx])==len(toMoves[toMoveIdx]):
                    toMoveIdx+=1
        if dataQubitsIdxs is not None:
            dataQubits = [self._ionMapping[j][0] for j in dataQubitsIdxs]
        # TODO use cooling ions? probs not here since architecture dependent
        for d in dataQubits:
            d._color = self.DATA_QUBIT_COLOR
            d._label = "D"
            self._dataIons.append(d)
            self._measurementIons.remove(d)
        return operations, barriers, toMoveOps

    def _gridToCoordinate(
        self, pos: Tuple[int, int], trapCapacity: int
    ) -> npt.NDArray[np.float64]:
        return np.array(pos) * (trapCapacity + 1) * self.SPACING

    def resetArch(
        self
    ) -> QCCDArch:
        for node in self._arch.nodes.values():
            while node.ions:
                node.removeIon(node.ions[0])

        for crossing in self._arch._crossings:
            if crossing.ion:
                crossing.clearIon()

        for trap, ions in self._originalArrangement.items():
            for i, ion in enumerate(ions):
                trap.addIon(ion, offset=i)
                ion.addMotionalEnergy(-ion.motionalMode)
        return self._arch
    
    def simulate(self, operations: Sequence[Operation], num_shots: int = 100_000, error_scaling: float = 1.0, decode: bool = True, isWISEArch: bool =True) -> Tuple[float, float, float]:
        # TODO add the effect of dephasing noise from idling qubits involved in splits and merges into this simulation (see notability notes)
        # TODO add importance subset sampling (see notability notes)
        # TODO speed up with sinter (see stim/getting_started)
        stimInstructions, _ = self.circuitString(include_annotation=True)
        
        stimIdxs: List[int] = []
        ions: List[Ion] = []
        for stimIdx, (ion, _) in self._ionMapping.items():
            if isinstance(ion, SpectatorIon):
                continue
            stimIdxs.append(stimIdx)
            ions.append(ion)

        operationsForIons: Dict[int, List[QubitOperation]] = {stimIdx: [] for stimIdx in stimIdxs}
        gateSwapsForIons: Dict[int, List[Tuple[int, GateSwap]]] = {stimIdx: [] for stimIdx in stimIdxs}
        qubitOps = []
        for op in operations:
            if isinstance(op, GateSwap):
                ion = op.ions[0] # first ion is the source ion in the gate swap
                # assert ion.label[0] == "M", "only swap ancillas"
                stimIdxForIon = stimIdxs[ions.index(ion)]
                opForIonIdx = len(operationsForIons[stimIdxForIon])
                gateSwapsForIons[stimIdxForIon].append(( opForIonIdx, op))
            elif isinstance(op, QubitOperation):
                for ion in op.ions:
                    operationsForIons[stimIdxs[ions.index(ion)]].append(op)
                qubitOps.append(op)

        gateSwapsForOperations: Dict[QubitOperation, List[GateSwap]] = {op: [] for op in qubitOps}
        for stimIdx, gateSwaps in gateSwapsForIons.items():
            for (opForIonIdx, op) in gateSwaps:
                gateSwapsForOperations[operationsForIons[stimIdx][opForIonIdx]].append(op)

        meanPhysicalZError = 0.0
        meanPhysicalXError = 0.0
        dephasingSchedule  = calculateDephasingFromIdling(operations, isWISEArch)
        dephasingSchedule = dict(dephasingSchedule)

        numZGates = 0
        numXGates = 0
        circuitString = ''
        for i in stimInstructions:
            if i.startswith("BARRIER"):
                continue
            idx = int(i.split(" ")[1]) if ( i[0] in ("M", "H", "R") or i.startswith("CNOT") or i.startswith("CZ")) else -1
            doNoiseAfter = False if i[0]=="M" else True
            if i[0] == "M" or i[0] == "R":
                ops = operationsForIons[idx][:1]
                operationsForIons[idx].pop(0)
            elif i[0] == "H":
                ops = operationsForIons[idx][:2]
                operationsForIons[idx].pop(0)
                operationsForIons[idx].pop(0)
            elif i.startswith("CNOT"):
                idx2 = int(i.split(" ")[2])
                # Do not duplicate the two qubit gate
                ops = operationsForIons[idx][:4] + operationsForIons[idx2][:1]
                operationsForIons[idx].pop(0)
                operationsForIons[idx].pop(0)
                operationsForIons[idx].pop(0)
                operationsForIons[idx2].pop(0)
                operationsForIons[idx2].pop(0)
            elif i.startswith("CZ"):
                idx2 = int(i.split(" ")[2])
                # Do not duplicate the two qubit gate
                ops = operationsForIons[idx][:4] + operationsForIons[idx2][:2]+operationsForIons[idx2][3:5]
                operationsForIons[idx].pop(0)
                operationsForIons[idx].pop(0)
                operationsForIons[idx].pop(0)
                operationsForIons[idx2].pop(0)
                operationsForIons[idx2].pop(0)
                operationsForIons[idx2].pop(0)
                operationsForIons[idx2].pop(0)
                operationsForIons[idx2].pop(0)
            else:
                ops = []

            physicalZError = 0.0
            physicalXError = 0.0
            for op in ops:
                if op in qubitOps:
                    for ion in op.ions:
                        if len(dephasingSchedule[ion])>0:
                            dephasing = [dephasingFidelity for opAtEndOfIdle, dephasingFidelity in dephasingSchedule[ion] if opAtEndOfIdle==op]
                            if dephasing:
                                dephasingInFidelity = min((1-dephasing[0])/error_scaling, 0.5)
                                physicalZError += dephasingInFidelity
                                circuitString+=f"Z_ERROR({dephasingInFidelity}) {stimIdxs[ions.index(ion)]}\n"
                    for gs in gateSwapsForOperations[op]:
                        gsInfidelity = min((1-gs.fidelity())/error_scaling, 0.5)
                        physicalXError += gsInfidelity/2
                        physicalZError += gsInfidelity/2
                        circuitString+=f"DEPOLARIZE2({gsInfidelity}) {stimIdxs[ions.index(gs.ions[0])]} {stimIdxs[ions.index(gs.ions[1])]}\n"
                      
                        
            if doNoiseAfter:
                circuitString+=f'{i}\n'

            for op in ops:
                opInfidelity = min((1-op.fidelity())/error_scaling, 0.5)
                if len(op.ions)==1: 
                    if isinstance(op, QubitReset) or isinstance(op, Measurement):
                        physicalXError+=opInfidelity
                        circuitString+=f"X_ERROR({opInfidelity}) {stimIdxs[ions.index(op.ions[0])]}\n"
                    else:
                        physicalXError+=opInfidelity/2
                        physicalZError+=opInfidelity/2
                        circuitString+=f"DEPOLARIZE1({opInfidelity}) {stimIdxs[ions.index(op.ions[0])]}\n"
                elif len(op.ions)==2:
                    physicalXError+=opInfidelity/2
                    physicalZError+=opInfidelity/2
                    circuitString+=f"DEPOLARIZE2({opInfidelity}) {stimIdxs[ions.index(op.ions[0])]} {stimIdxs[ions.index(op.ions[1])]}\n"
                else:
                    raise ValueError(f"simulate: {op} contains {len(op.ions)} ions.")
            numZGates+=physicalZError>0
            numXGates+=physicalXError>0
            meanPhysicalZError += physicalZError
            meanPhysicalXError += physicalXError
            if not doNoiseAfter:
                circuitString+=f'{i}\n'
        meanPhysicalZError /= numZGates
        meanPhysicalXError /= numXGates
        circuit = stim.Circuit(circuitString)
        if not decode:
            return 1, meanPhysicalXError, meanPhysicalZError
        # Sample the circuit, by using the fast circuit stabilizer tableau simulator provided by Stim.
        sampler = circuit.compile_detector_sampler()
        sample =sampler.sample(num_shots, separate_observables=True)
        detection_events, observable_flips = sample
        detection_events = np.array(detection_events, order='C')

        # Construct a Tanner graph, by translating the detector error model using the circuit.
        detector_error_model = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

        # Determine the predicted logical observable, by running the MWPM decoding algorithm on the Tanner graph
        predictions = []
        for i in range(num_shots):
            predictions.append(matcher.decode(detection_events[i]))
        predictions=np.array(predictions)

        # Count the mistakes.
        num_errors = 0
        for shot in range(num_shots):
            actual_for_shot = observable_flips[shot]
            predicted_for_shot = predictions[shot]
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors += 1
        logicalError = num_errors / num_shots 
        return logicalError, meanPhysicalXError, meanPhysicalZError



    # def processCircuitWithQCCDSimMachine(
    #         self,
    #         machine: Machine,
    #         dataQubitIdxs: Optional[Sequence[int]]=None,
    # ) -> Tuple[QCCDArch, Tuple[Sequence[QubitOperation], Sequence[int]]]: 
    #     instructions, barriers = self._parseCircuitString(dataQubitsIdxs=dataQubitIdxs)
    #     if machine.num_ions < len(self._ionMapping):
    #         raise ValueError("processCircuit: not enough traps")
        
    #     qasmCircuit = self.without_noise().to_qasm(open_qasm_version=OPEN_QASM_VERSION, skip_dets_and_obs=True)
    #     trapCapacity = 0
    #     #Parse the input program DAG
    #     ip = InputParse()
    #     ip.parse_from_string(qasmCircuit)

    #     # qm = QubitMapGreedy(ip, machine)
    #     # mapping = qm.compute_mapping()  
        
    #     mapping = customScheduler.get_custom_mapping(qasmCircuit, machine, "", len(self._measurementIons), len(self._ionMapping))

    #     labelled_coords = machine_graph_to_labeled_coordinates(machine)

    #     self._arch = QCCDArch()
        
    #     coordsToNd = {}
    #     nToNd = {}
    #     for (l, n, cx, cy) in labelled_coords:
    #         if l=='T':
    #             trapCapacity = n.capacity
    #             nd = self._arch.addManipulationTrap(
    #                 *self._gridToCoordinate((cx, cy), trapCapacity), [self._ionMapping[idx][0] for idx in mapping[n.id]], isHorizontal=False, capacity=n.capacity,  color=self.TRAP_COLOR,
    #             )
    #         else:
    #             nd = self._arch.addJunction(
    #                 *self._gridToCoordinate((cx, cy), trapCapacity)
    #             )
    #         coordsToNd[(cx, cy)] = nd, l
    #         nToNd[n] = nd
    #     # minX, maxX = min(lc[2] for lc in labelled_coords), max(lc[2] for lc in labelled_coords)
    #     # minY, maxY = min(lc[3] for lc in labelled_coords), max(lc[3] for lc in labelled_coords)
    #     # for x in range(minX, maxX+1, 1):
    #     #     for y in range(minY, maxY+1, 1):
    #     #         if (x,y) in coordsToNd:
    #     #             nd1, l1 = coordsToNd[(x,y)]
    #     #             if (x+1, y) in coordsToNd:
    #     #                 nd2, l2 = coordsToNd[(x+1,y)]
    #     #                 if l1 == 'J' or l2 == 'J':   
    #     #                     self._arch.addEdge(nd1, nd2)
    #     #             if (x, y+1) in coordsToNd:
    #     #                 nd2, l2 = coordsToNd[(x, y+1)]
    #     #                 if l1 == 'J' or l2 == 'J':   
    #     #                     self._arch.addEdge(nd1, nd2)

    #     for edge in machine.graph.edges:
    #         n1, n2 = edge[0], edge[1]
    #         self._arch.addEdge(nToNd[n1], nToNd[n2])

        

    #     trap_ions = {}
    #     for i in machine.traps:
    #         if mapping[i.id]:
    #             trap_ions[i.id] = mapping[i.id][:]

    #     return self._arch, (instructions, barriers)


    def processCircuitAugmentedGrid(
        self,
        trapCapacity: int = 2,
        rows: int = 1,
        cols: int = 5,
        padding: int = 1,
        dataQubitIdxs: Optional[Sequence[int]]=None,
    ) -> Tuple[QCCDArch, Tuple[Sequence[QubitOperation], Sequence[int]]]:        
        instructions, barriers, _ = self._parseCircuitString(dataQubitsIdxs=dataQubitIdxs)
        if (trapCapacity-1) * ((rows-1) * (2*cols-1)+cols) < len(self._ionMapping):
            raise ValueError("processCircuit: not enough traps")
           
        clusters=regularPartition(self._measurementIons, self._dataIons, trapCapacity)

        cs, rs = cols, rows
        allGridPos = []
        for r in range(rs):
            for c in range(cs):
                allGridPos.append((2*c, 2*r))
                if c < cs-1 and r<rs-1:
                    allGridPos.append((2*c+1, 2*r+1)) 

        gridPositions = hillClimbOnArrangeClusters(clusters, allGridPos=allGridPos)
        gridPositions = [(c+padding, r+padding) for (c, r) in gridPositions]
        rows = rows+2*padding
        cols = cols+2*padding
        trap_for_grid = {
            (col, row): clusters[trapIdx]
            for trapIdx, (col, row) in enumerate(gridPositions)
        }
        self._originalArrangement = {}

        self._arch = QCCDArch()
        traps_dict = {}
        for row in range(rows):
            for col in range(cols):
                if (2*col, 2*row) in trap_for_grid:
                    ions = trap_for_grid[(2*col, 2*row)][0]
                else:
                    ions = []
                traps_dict[(2*col, 2*row)] = self._arch.addManipulationTrap(
                    *self._gridToCoordinate((2*col, 2*row), trapCapacity),
                    ions,
                    color=self.TRAP_COLOR,
                    isHorizontal=(rows==1),
                    capacity=trapCapacity
                )
                self._originalArrangement[traps_dict[(2*col, 2*row)]] = ions

            if row == rows-1:
                break

            for col in range(cols-1):
                if (2*col+1, 2*row+1) in trap_for_grid:
                    ions = trap_for_grid[(2*col+1, 2*row+1)][0]
                else:
                    ions = []
                traps_dict[(2*col+1, 2*row+1)] = self._arch.addManipulationTrap(
                    *self._gridToCoordinate((2*col+1, 2*row+1), trapCapacity),
                    ions,
                    color=self.TRAP_COLOR,
                    isHorizontal=True,
                    capacity=trapCapacity
                )
                self._originalArrangement[traps_dict[(2*col+1, 2*row+1)]] = ions
            
        if rows == 1:
            for (col, r), trap_node in traps_dict.items():
                if (col + 2, r) in traps_dict:
                    self._arch.addEdge(trap_node, traps_dict[(col + 2, r)])
        else:
            junctions_dict = {}
            for (col, row), trap_node in traps_dict.items():
                # Add vertical edges (between even rows)
                if col % 2 == 0 and (col, row + 2) in traps_dict:
                    junction = self._arch.addJunction(
                        *(
                            (
                                self._gridToCoordinate((col, row), trapCapacity)
                                + self._gridToCoordinate((col, row + 2), trapCapacity)
                            )
                            / 2
                        ),
                        color=self.JUNCTION_COLOR,
                    )
                    junctions_dict[(col, row+1)] = junction
                    self._arch.addEdge(trap_node, junction)
                    self._arch.addEdge(junction, traps_dict[(col, row + 2)])

            # Add horizontal edges between traps and junctions in the same row
            for row in range(rows-1):
                for col in range(cols - 1):
                    if (2*col, 2*row+1) in junctions_dict and (
                        2*col + 1,
                        2*row + 1,
                    ) in traps_dict:
                        self._arch.addEdge(
                            junctions_dict[(2*col, 2*row+1)], traps_dict[(2*col + 1, 2*row+1)]
                        )
                    if (2*col+1, 2*row+1) in traps_dict and (
                        2*col + 2,
                        2*row + 1,
                    ) in junctions_dict:
                        self._arch.addEdge(
                            traps_dict[(2*col+1, 2*row+1)], junctions_dict[(2*col + 2, 2*row+1)]
                        )

        if any(i.parent is None for i in self._arch.ions.values()):
            raise ValueError(f"Ions not in traps for {trapCapacity} and {len(self._measurementIons)+len(self._dataIons)}")
        return self._arch, (instructions, barriers)
    


    def processCircuitWiseArch(
        self,
        wiseArch: QCCDWiseArch,
        dataQubitIdxs: Optional[Sequence[int]]=None,
        addSpectators: bool = True,
        compactClustering: bool = True
    ) -> Tuple[QCCDArch, Tuple[Sequence[QubitOperation], Sequence[int], Sequence[Sequence[TwoQubitMSGate]]]]:        
        instructions, barriers, toMoveOps = self._parseCircuitString(dataQubitsIdxs=dataQubitIdxs)
        if compactClustering and wiseArch.m*wiseArch.n*wiseArch.k < len(self._ionMapping):
            raise ValueError("processCircuit: not enough traps")
        # if not compactClustering and (wiseArch.k-1) * ((rows-1) * (cols-1)+cols) < len(self._ionMapping):
        #     raise ValueError("processCircuit: not enough traps")
           
        
        clusters=regularPartition(self._measurementIons, self._dataIons, wiseArch.k, isWISEArch=( compactClustering), maxClusters=wiseArch.m*wiseArch.n if compactClustering else None)

        cs, rs = wiseArch.m, wiseArch.n
        allGridPos = []
        for r in range(rs):
            for c in range(cs):
                allGridPos.append((c, r))
        gridPositions = hillClimbOnArrangeClusters(clusters, allGridPos=allGridPos)
        gridPositions = [(c, r) for (c, r) in gridPositions]
        rows = wiseArch.n
        cols = wiseArch.m
        trap_for_grid = {
            (2*col, row): clusters[trapIdx]
            for trapIdx, (col, row) in enumerate(gridPositions)
        }
        self._originalArrangement = {}


        self._arch = QCCDArch()
        traps_dict = {}
        for row in range(rows):
            for col in range(cols):
                if (2*col, row) in trap_for_grid:
                    ions = trap_for_grid[(2*col, row)][0]
                else:
                    ions = []
                if addSpectators:
                    maxIdx=max(self._ionMapping.keys())
                    nplaceholds = wiseArch.k-len(ions)
                    for i in range(nplaceholds):
                        ion = SpectatorIon(*self.PLACEMENT_ION)
                        idx = maxIdx+1+i
                        ion.set(idx, *ion.pos)
                        self._ionMapping[idx] = ion, ion.pos
                        ions.append(ion)
                traps_dict[(2*col, row)] = self._arch.addManipulationTrap(
                    *self._gridToCoordinate((2*col, row), wiseArch.k),
                    ions,
                    color=self.TRAP_COLOR,
                    isHorizontal=True,
                    capacity=wiseArch.k
                )
                self._originalArrangement[traps_dict[(2*col, row)]] = ions
            
        if rows == 1:
            for (col, r), trap_node in traps_dict.items():
                if (col + 2, r) in traps_dict:
                    self._arch.addEdge(trap_node, traps_dict[(col + 2, r)])
        else:
            junctions_dict = {}
            for (col, row), trap_node in traps_dict.items():
                if (col, row + 1) in traps_dict:
                    if (col+1, row) not in junctions_dict:
                        junctionTop = self._arch.addJunction(
                            *self._gridToCoordinate((col+1, row), wiseArch.k),
                            color=self.JUNCTION_COLOR,
                        )
                        junctions_dict[(col+1, row)] = junctionTop
                    if (col+1, row+1) not in junctions_dict:
                    
                        junctionBottom = self._arch.addJunction(
                            *self._gridToCoordinate((col+1, row+1), wiseArch.k),
                            color=self.JUNCTION_COLOR,
                         )
                        
                        junctions_dict[(col+1, row+1)] = junctionBottom
                
            for (col, row), junctionTop in junctions_dict.items():
                if (col, row+1) in junctions_dict:
                    self._arch.addEdge(
                        junctions_dict[(col,row+1)], junctionTop
                    )
                
            # Add horizontal edges between traps and junctions in the same row
            for row in range(rows):
                for col in range(cols):
                    if (2*col+1, row) in junctions_dict and (
                        2*col,
                        row,
                    ) in traps_dict:
                        self._arch.addEdge(
                            junctions_dict[(2*col+1, row)], traps_dict[(2*col, row)]
                        )
                    if (2*col+2, row) in traps_dict and (
                        2*col + 1,
                        row,
                    ) in junctions_dict:
                        self._arch.addEdge(
                            traps_dict[(2*col+2, row)], junctions_dict[(2*col + 1, row)]
                        )

        if any(i.parent is None for i in self._arch.ions.values()):
            raise ValueError(f"Ions not in traps for {wiseArch.k} and {len(self._measurementIons)+len(self._dataIons)}")
        return self._arch, (instructions, barriers, toMoveOps)
    

    def processCircuitNetworkedGrid(self,
        trapCapacity: int = 2,
        traps: int = 1,
        dataQubitIdxs: Optional[Sequence[int]]=None
        # capacityIsInTermsOfDataIons: bool = False
    ) -> Tuple[QCCDArch, Tuple[Sequence[QubitOperation], Sequence[int]]]:        
        instructions, barriers, _ = self._parseCircuitString(dataQubitsIdxs=dataQubitIdxs)
        if (trapCapacity-1) * traps< len(self._ionMapping):
            raise ValueError("processCircuit: not enough traps")
           
        clusters=regularPartition(self._measurementIons, self._dataIons, trapCapacity)

        allGridPos = []
        for r in range(traps):
            allGridPos.append((0, r))

        gridPositions = arrangeClusters(clusters, allGridPos=allGridPos)

        trap_for_grid = {
            row: clusters[trapIdx]
            for trapIdx, (_, row) in enumerate(gridPositions)
        }
        self._originalArrangement = {}

        self._arch = QCCDArch()
        traps_dict = {}
        for row in range(traps):
            if row in trap_for_grid:
                ions = trap_for_grid[row][0]
            else:
                ions = []
            traps_dict[row] = self._arch.addManipulationTrap(
                *self._gridToCoordinate((0, row), trapCapacity),
                ions,
                color=self.TRAP_COLOR,
                isHorizontal=True,
                capacity=trapCapacity
            )
            self._originalArrangement[traps_dict[row]] = ions


        switch_cost = 1
        junctions_dict = {}
        
        for row, trap_node in traps_dict.items():
            for i in range(switch_cost):
                junction2 = self._arch.addJunction(
                    *self._gridToCoordinate((i+1, row), trapCapacity),
                    color=self.JUNCTION_COLOR,
                )
                junctions_dict[(i+1, row)] = junction2
                if i==0:
                    self._arch.addEdge(trap_node, junction2)
                else:
                    self._arch.addEdge(junctions_dict[(i, row)], junction2)

        for row, trap_node in traps_dict.items(): 
            junction2 = junctions_dict[(switch_cost, row)]
            for row2 in range(traps):
                if row==row2:
                    continue
               
                junction1 = junctions_dict[(switch_cost, row2)]
                self._arch.addEdge(junction1, junction2)

        return self._arch, (instructions, barriers)




NDE_LZ = 10
NDE_JZ = 20
NSE_Z = 10

def process_circuit(distance, capacity, gate_improvements, num_shots):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("process_log.txt")
    formatter = logging.Formatter('%(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"Starting circuit generation for distance {distance} and capacity {capacity}")
  
    circuit = QCCDCircuit.generated(
        "surface_code:rotated_memory_z",
        rounds=1,
        distance=distance,
    )
    nqubitsNeeded = 2 * distance**2 - 1

    nrowsNeeded = int(np.sqrt(nqubitsNeeded))+2

    logger.info(f"Processing circuit with {nqubitsNeeded} qubits and {nrowsNeeded} rows")

    arch, (instructions, _) = circuit.processCircuitAugmentedGrid(rows=nrowsNeeded, cols=nrowsNeeded, trapCapacity=capacity)
    # arch, (instructions, _) = circuit.processCircuitNetworkedGrid(traps=nqubitsNeeded, trapCapacity=capacity)
    arch.refreshGraph()

    results = {"ElapsedTime": {}, "Operations": {}, "MeanConcurrency": {}, "QubitOperations": {}, "LogicalErrorRates": {}, "PhysicalZErrorRates": {}, "PhysicalXErrorRates": {}, "Electrodes": {}, "DACs": {}}

    # FIXME legacy formatting!
    label ="Forwarding"

    logger.info(f"Processing operations using {label} for distance {distance} and capacity {capacity}")
    allOps, barriers = ionRouting(arch, instructions, capacity)
 
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
    logicalErrors = []
    physicalZErrors = []
    physicalXErrors = []
    
    for gate_improvement in gate_improvements:
        logicalError, physicalXError, physicalZError = circuit.simulate(allOps, num_shots=num_shots, error_scaling=gate_improvement, isWISEArch=False)
        logicalErrors.append(logicalError)
        physicalZErrors.append(physicalZError)
        physicalXErrors.append(physicalXError)

    logger.info(f"Simulated {label} method with gate improvements for distance {distance}, capacity {capacity}")
    
    
    for op in parallelOpsMap.values():
        op.calculateOperationTime()
        op.calculateFidelity()

    circuit.resetArch()
    arch.refreshGraph()

    results["Capacity"] = capacity
    results["Distance"] = distance
    results["ElapsedTime"][label] = max(parallelOpsMap.keys())
    results["Operations"][label] = len(allOps)
    results["MeanConcurrency"][label] = np.mean([len(op.operations) for op in parallelOpsMap.values()])
    results["QubitOperations"][label] = len(instructions)
    results["LogicalErrorRates"][label] = logicalErrors
    results["PhysicalZErrorRates"][label] = physicalZErrors
    results["PhysicalXErrorRates"][label] = physicalXErrors


    trapSet = set()
    junctionSet = set()
    for op in allOps:
        for c in op.involvedComponents:
            if isinstance(c, Trap):
                trapSet.add(c)
            elif isinstance(c, Junction):
                junctionSet.add(c)

    # Every zone can contain up to two qubits
    # Njz = 1*Nj, Nlz = Nl*k
    # The number of zones  N = 1*Nj+k*Nl
    Njz = len(junctionSet) # each junction is one zone
    Nlz = len(trapSet)*capacity # each trap is k zones

    # Njz = int(np.ceil(nqubitsNeeded / (2*(capacity-1))) )# 2 traps per junction
    # Nlz = nqubitsNeeded-Njz
    Nde = NDE_LZ*Nlz+NDE_JZ*Njz
    Nse = NSE_Z*(Njz+Nlz)

    Num_electrodes = Nde+Nse
    Num_DACs = Num_electrodes
    results["DACs"][label] = Num_DACs
    results["Electrodes"][label] = Num_electrodes

    logger.info(f"{distance} {capacity} {label} = {results}")
    
    logger.info(f"Finished processing for distance {distance} and capacity {capacity}")
    return results




def process_circuit_wise_arch(distance, capacity, gate_improvements, num_shots):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("process_log_wise.txt")
    formatter = logging.Formatter('%(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"Starting circuit generation for distance {distance} and capacity {capacity}")
  
    circuit = QCCDCircuit.generated(
        "surface_code:rotated_memory_z",
        rounds=1,
        distance=distance,
    )
    nqubitsNeeded = 2 * distance**2 - 1

    nrowsNeeded = int(np.sqrt(nqubitsNeeded))+2

    logger.info(f"Processing circuit with {nqubitsNeeded} qubits and {nrowsNeeded} rows")

    wiseArch = QCCDWiseArch(m=int(np.sqrt(capacity*nqubitsNeeded/2))+1, n=int(np.sqrt(2*nqubitsNeeded/capacity))+1, k=capacity)
    arch, (instructions, _) = circuit.processCircuitWiseArch(wiseArch=wiseArch)
    
    arch.refreshGraph()

    results = {"ElapsedTime": {}, "Operations": {}, "MeanConcurrency": {}, "QubitOperations": {}, "LogicalErrorRates": {}, "PhysicalZErrorRates": {}, "PhysicalXErrorRates": {}, "Electrodes": {}, "DACs": {}}

    # FIXME legacy formatting!
    label ="Forwarding"

    logger.info(f"Processing operations using {label} for distance {distance} and capacity {capacity}")
    allOps, barriers = ionRoutingWISEArch(arch, wiseArch, instructions)
 
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
    logicalErrors = []
    physicalZErrors = []
    physicalXErrors = []
    
    for gate_improvement in gate_improvements:
        logicalError, physicalXError, physicalZError = circuit.simulate(allOps, num_shots=num_shots, error_scaling=gate_improvement, isWISEArch=True)
        logicalErrors.append(logicalError)
        physicalZErrors.append(physicalZError)
        physicalXErrors.append(physicalXError)

    logger.info(f"Simulated {label} method with gate improvements for distance {distance}, capacity {capacity}")
    
    
    for op in parallelOpsMap.values():
        op.calculateOperationTime()
        op.calculateFidelity()

    circuit.resetArch()
    arch.refreshGraph()

    results["Capacity"] = capacity
    results["Distance"] = distance
    results["ElapsedTime"][label] = max(parallelOpsMap.keys())
    results["Operations"][label] = len(allOps)
    results["MeanConcurrency"][label] = np.mean([len(op.operations) for op in parallelOpsMap.values()])
    results["QubitOperations"][label] = len(instructions)
    results["LogicalErrorRates"][label] = logicalErrors
    results["PhysicalZErrorRates"][label] = physicalZErrors
    results["PhysicalXErrorRates"][label] = physicalXErrors

    Njz = np.ceil(nqubitsNeeded / capacity)
    Nlz = nqubitsNeeded - Njz # note the difference because we do not have vertical traps

    Nde = NDE_LZ*Nlz+NDE_JZ*Njz
    Nse = NSE_Z*(Njz+Nlz)

    Num_electrodes =int( Nde+Nse)
    Num_DACs = int(min(100, Nde)+np.ceil(Nse/100))
    results["DACs"][label] = Num_DACs
    results["Electrodes"][label] = Num_electrodes


    logger.info(f"{distance} {capacity} {label} = {results}")
    
    logger.info(f"Finished processing for distance {distance} and capacity {capacity}")
    return results