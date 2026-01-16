
# # from typing import (
# #     Sequence,
# #     List,
# #     Tuple,
# #     Optional,
# #     Set,
# #     Dict,
# # )
# # import networkx as nx
# # from src.utils.qccd_nodes import *
# # from src.utils.qccd_operations import *
# # from src.utils.qccd_operations_on_qubits import *
# # from src.utils.qccd_arch import *

# # import numpy as np
# # import numpy.typing as npt
# # from scipy.spatial import distance_matrix
# # from scipy.optimize import linear_sum_assignment
# # from typing import (
# #     Sequence,
# #     Tuple,
# # )
# # import itertools
# # from src.utils.qccd_nodes import *
# # from src.utils.qccd_operations import *
# # from src.utils.qccd_operations_on_qubits import *
# # from src.utils.qccd_arch import *
# # from src.compiler.qccd_parallelisation import *


# # def _minWeightPerfectMatch(wiseLayoutInit: Mapping[int, Tuple[int,int]], wiseLayoutTarget: Mapping[int, Tuple[int,int]]) -> Tuple[float, Sequence[int]]:
    
# #     cols = {}
# #     rowsDups = {}
# #     for idx in wiseLayoutInit.keys():
# #         (ir, ic) = wiseLayoutInit[idx]
# #         (tr, tc) = wiseLayoutTarget[idx]

# #         if ic in cols and tr in cols[ic]:
# #             if ir in rowsDups:
# #                 rowsDups[ir].append(idx)
# #             else: 
# #                 rowsDups[ir] = idx
# #         if ic in cols:
# #             cols[ic].append(tr)
# #         else:
# #             cols[ic] = [tr]


    
    



# #     cost_matrix =distance_matrix(A, B)
# #     # the Hungarian algorithm 
# #     row_ind, col_ind = linear_sum_assignment(cost_matrix)
# #     total_cost = cost_matrix[row_ind, col_ind].sum()
# #     return total_cost, col_ind



# from scipy.optimize import linear_sum_assignment
# from typing import Mapping, Tuple
# # wiseLayoutInit = {
# #     0: (1,1), 1: (1,2), 2: (1,3),
# #     3: (2,1), 4: (2,2), 5: (2,3),
# #     6: (3,1), 7: (3,2), 8: (3,3),
# # }

# # wiseLayoutTarget = {
# #     5: (1,1), 6: (1,2), 2: (1,3),
# #     3: (2,1), 0: (2,2), 7: (2,3),
# #     1: (3,1), 8: (3,2), 4: (3,3),
# # }

# # rows = 3
# # cols = 3

# # def pprint(_Tinit):

#     # for r in range(3):
#     #     print(r+1, [_Tinit[(r+1,c+1)] for c in range(3)])

#     # print('\n')
            



# def firstRowSort(wiseLayoutInit: Mapping[int, Tuple[int, int]], wiseLayoutTarget: Mapping[int, Tuple[int, int]], rows: int, cols: int) -> Mapping[int, Tuple[int, int]]:
#     Tinit = {}
#     for idx in wiseLayoutInit.keys():
#         Tinit[wiseLayoutInit[idx]]=wiseLayoutTarget[idx]


#     # pprint(Tinit)

#     targRowByinitCol = [ [ False for _ in range(cols)] for _ in range(rows)]

#     for r in range(rows):
#         ir = r+1
#         lhnodes = []
#         rhnodes = []
#         for c in range(cols):
#             ic = c+1
#             if (ir, ic) in Tinit:
#                 tr, tc = Tinit[ir, ic]
#                 lhnodes.append((tr, tc))
#             rhnodes.append(ic)

#         if not lhnodes:
#             continue
#         cost_matrix = [[targRowByinitCol[tr-1][ic-1] for ic in rhnodes] for (tr,_) in lhnodes]
#         _, col_ind = linear_sum_assignment(cost_matrix)

#         for (tr, tc), c in zip(lhnodes, col_ind):
#             ic = rhnodes[c]
#             Tinit[ir, ic] = (tr, tc)
#             targRowByinitCol[tr-1][ic-1]=True

#         # pprint(Tinit)


#     wiseLayoutTargRev = {(tr, tc): k for k, (tr, tc) in wiseLayoutTarget.items()}

#     wiseLayout = {wiseLayoutTargRev[(tr, tc)]: (ir, ic) for (ir, ic), (tr, tc) in Tinit.items()}
#     return wiseLayout

# def oddEvenColSort(wiseLayoutInit: Mapping[int, Tuple[int, int]], wiseLayoutTarget: Mapping[int, Tuple[int, int]], k:int, i: int) -> int:
#     columnsInit ={}
#     columnsDesired={}
#     for k, (ir, ic) in wiseLayoutInit.items():
#         (tr, tc) = wiseLayoutTarget[k]
#         if ic != tc:
#             raise ValueError('oddEvenColSort: not a column')
#         if ic % k != i:
#             continue
#         if ic in columnsInit:
#             columnsInit[ic].append((ir, k))
#         else:
#             columnsInit[ic]=[(ir,k)]
#         if tc in columnsDesired:
#             columnsDesired[tc].append((tr, k))
#         else:
#             columnsDesired[tc]=[(tr,k)]

#     max_num_swaps = 0
#     for ic, colInitList in columnsInit.items():
#         colInitListSort = sorted(colInitList, key=lambda x:x[0])
#         colDesListSort = sorted(columnsDesired[ic], key=lambda x:x[0])
#         swaps = []
#         l = int(len(colDesListSort)/2)
#         while True:
#             doneswap = False
#             for i in range(l):
#                 if colDesListSort[2*i+1][1]>colDesListSort[2*i+2][1]:
#                     v = colInitListSort[2*i+1]
#                     colInitListSort[2*i+1]=colInitListSort[2*i+2]
#                     colInitListSort[2*i+2]=v
#                     swaps.append((v[1], colInitListSort[2*i+1][1]))
#                     doneswap=True 
#             for i in range(l):
#                 if colDesListSort[2*i][1]>colDesListSort[2*i+1][1]:
#                     v = colInitListSort[2*i]
#                     colInitListSort[2*i]=colInitListSort[2*i+1]
#                     colInitListSort[2*i+1]=v
#                     doneswap=True 
#                     swaps.append((v[1], colInitListSort[2*i][1]))
                
#             if not doneswap:
#                 break

#         nswaps = len(swaps)
#         if nswaps > max_num_swaps:
#             nswaps = max_num_swaps

#     return max_num_swaps



# def oddEvenRowSort(wiseLayoutInit: Mapping[int, Tuple[int, int]], wiseLayoutTarget: Mapping[int, Tuple[int, int]]) -> int:
#     rowsInit ={}
#     rowsDesired={}
#     for k, (ir, ic) in wiseLayoutInit.items():
#         (tr, tc) = wiseLayoutTarget[k]
#         if tr != ir:
#             raise ValueError('oddEvenColSort: not a column')
#         if ir in rowsInit:
#             rowsInit[ir].append((ic, k))
#         else:
#             rowsInit[ir]=[(ic,k)]
#         if tr in rowsDesired:
#             rowsDesired[tr].append((tc, k))
#         else:
#             rowsDesired[tr]=[(tc,k)]

#     max_num_swaps = 0
#     for ir, rowInitList in rowsInit.items():
#         rowInitListSort = sorted(rowInitList, key=lambda x:x[0])
#         rowDesListSort = sorted(rowsDesired[ir], key=lambda x:x[0])
#         swaps = []
#         l = int(len(rowInitListSort)/2)
#         while True:
#             doneswap = False
#             for i in range(l):
#                 if rowDesListSort[2*i+1][1]>rowDesListSort[2*i+2][1]:
#                     v = rowInitListSort[2*i+1]
#                     rowInitListSort[2*i+1]=rowInitListSort[2*i+2]
#                     rowInitListSort[2*i+2]=v
#                     swaps.append((v[1], rowInitListSort[2*i+1][1]))
#                     doneswap=True 
#             for i in range(l):
#                 if rowDesListSort[2*i][1]>rowDesListSort[2*i+1][1]:
#                     v = rowInitListSort[2*i]
#                     rowInitListSort[2*i]=rowInitListSort[2*i+1]
#                     rowInitListSort[2*i+1]=v
#                     doneswap=True 
#                     swaps.append((v[1], rowInitListSort[2*i][1]))
                
#             if not doneswap:
#                 break

#         nswaps = len(swaps)
#         if nswaps > max_num_swaps:
#             nswaps = max_num_swaps

#     return max_num_swaps

        

# class QCCDWISECircuit(stim.Circuit):
#     DATA_QUBIT_COLOR = "lightblue"
#     MEASUREMENT_QUBIT_COLOR = "red"
#     PLACEMENT_ION = ("grey", "P")
#     TRAP_COLOR = "grey"
#     JUNCTION_COLOR = "orange"
#     SPACING = 20


#     start_score: int = 1
#     score_delta: int = 2
#     joinDisjointClusters: bool = False
#     minIters: int = 1_000
#     maxIters: int = 10_000


#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self._ionMapping: Dict[int, Tuple[Ion, Tuple[int, int]]] = {}
#         self._measurementIons: List[Ion] = []
#         self._dataIons: List[Ion] = []
#         self._arch: QCCDWiseArch

#     @classmethod
#     def generated(cls, *args, **kwargs) -> "QCCDWISECircuit":
#         return QCCDWISECircuit(stim.Circuit.generated(*args, **kwargs).__str__())

#     def circuitString(self, include_annotation: bool = False) -> Sequence[str]:
#         instructions = (
#             self.flattened().decomposed().without_noise().__str__().splitlines()
#         )
#         newInstructions = []
#         for i in instructions:
#             qubits = i.rsplit(" ")[1:]
#             if i.startswith("DETECTOR") or i.startswith("TICK") or i.startswith("OBSERVABLE"):
#                 if include_annotation:
#                     newInstructions.append(i)
#                 continue
#             elif i[0] in ("R", "H", "M"):
#                 for qubit in qubits:
#                     newInstructions.append(f"{i[0]} {qubit}")
#                 # newInstructions.append("BARRIER")
#             elif any(i.startswith(s) for s in stim.gate_data("cnot").aliases):
#                 for i in range(int(len(qubits) / 2)):
#                     newInstructions.append(f"CNOT {qubits[2*i]} {qubits[2*i+1]}")
#                 newInstructions.append("BARRIER")
#             else:
#                 newInstructions.append(i)
#         return newInstructions

#     @property
#     def ionMapping(self) -> Mapping[int, Tuple[Ion, Tuple[int, int]]]:
#         return self._ionMapping

#     def _parseCircuitString(self, dataQubitsIdxs: Optional[Sequence[int]]=None) -> Tuple[Sequence[QubitOperation], Sequence[int]]:
#         instructions = self.circuitString()

#         self._measurementIons = []
#         self._ionMapping = {}
#         self._dataIons = []

#         for j, i in enumerate(instructions):
#             if not i.startswith("QUBIT_COORDS"):
#                 break
#             coords = tuple(
#                 map(int, i.removeprefix("QUBIT_COORDS(").split(")")[0].split(","))
#             )
#             idx = int(i.split(" ")[-1])
#             ion = QubitIon(self.MEASUREMENT_QUBIT_COLOR, label="M")
#             ion.set(idx, *coords)
#             self._ionMapping[idx] = ion, coords
#             self._measurementIons.append(ion)

#         instructions = instructions[j:]
#         operations = []
#         barriers = []
#         dataQubits = []
#         # TODO establish correct mapping of qubit operations from QIP toolkit with references
#         for j, i in enumerate(instructions):
#             if i.startswith("BARRIER"):
#                 barriers.append(len(operations))
#                 continue
#             if not ( i[0] in ("M", "H", "R") or i.startswith("CNOT")):
#                 continue
#             idx = int(i.split(" ")[1])
#             ion = self._ionMapping[idx][0]
#             if i[0] == "M":
#                 operations.append(Measurement.qubitOperation(ion))
#                 if dataQubitsIdxs is None:
#                     dataQubits.append(ion) # data qubits are the ones measured at the end
#             elif i[0] == "H":
#                 # page 80 https://iontrap.umd.edu/wp-content/uploads/2013/10/FiggattThesis.pdf
#                 operations.extend([
#                     YRotation.qubitOperation(ion),
#                     XRotation.qubitOperation(ion)
#                 ])
#                 if dataQubitsIdxs is None:
#                     dataQubits.clear()
#             elif i[0] == "R":
#                 operations.append(QubitReset.qubitOperation(ion))
#                 if dataQubitsIdxs is None:
#                     dataQubits.clear()
#             elif i.startswith("CNOT"):
#                 idx2 = int(i.split(" ")[2])
#                 ion2 = self._ionMapping[idx2][0]
#                 # Fig 4. https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
#                 operations.extend([
#                     YRotation.qubitOperation(ion),
#                     XRotation.qubitOperation(ion),
#                     XRotation.qubitOperation(ion2),
#                     TwoQubitMSGate.qubitOperation(
#                         ion, ion2
#                     ),
#                     YRotation.qubitOperation(ion)
#                 ])
#                 if dataQubitsIdxs is None:
#                     dataQubits.clear()
#         if dataQubitsIdxs is not None:
#             dataQubits = [self._ionMapping[j][0] for j in dataQubitsIdxs]
#         # TODO use cooling ions? probs not here since architecture dependent
#         for d in dataQubits:
#             d._color = self.DATA_QUBIT_COLOR
#             d._label = "D"
#             self._dataIons.append(d)
#             self._measurementIons.remove(d)
#         return operations, barriers


#     def simulate(self, operations: Sequence[Operation], barriers: Mapping[int, float], num_shots: int = 100_000, error_scaling: float = 1.0, decode: bool = True) -> Tuple[float, float, float]:
#         # TODO add the effect of dephasing noise from idling qubits involved in splits and merges into this simulation (see notability notes)
#         # TODO add importance subset sampling (see notability notes)
#         # TODO speed up with sinter (see stim/getting_started)
#         stimInstructions = self.circuitString(include_annotation=True)
#         operations = list(operations)
#         stimIdxs: List[int] = []
#         ions: List[Ion] = []
#         for stimIdx, (ion, _) in self._ionMapping.items():
#             stimIdxs.append(stimIdx)
#             ions.append(ion)

#         operationsForIons: Dict[int, List[QubitOperation]] = {stimIdx: [] for stimIdx in stimIdxs}
#         qubitOps = []
#         for op in operations:
#             if isinstance(op, QubitOperation):
#                 for ion in op.ions:
#                     operationsForIons[stimIdxs[ions.index(ion)]].append(op)
#                 qubitOps.append(op)

#         meanPhysicalZError = 0.0
#         meanPhysicalXError = 0.0

#         numZGates = 0
#         numXGates = 0
#         circuitString = ''
#         for i in stimInstructions:
#             if i.startswith("BARRIER"):
#                 continue
#             idx = int(i.split(" ")[1]) if ( i[0] in ("M", "H", "R") or i.startswith("CNOT")) else -1
#             doNoiseAfter = False if i[0]=="M" else True
#             if i[0] == "M" or i[0] == "R":
#                 ops = operationsForIons[idx][:1]
#                 operationsForIons[idx].pop(0)
#             elif i[0] == "H":
#                 ops = operationsForIons[idx][:2]
#                 operationsForIons[idx].pop(0)
#                 operationsForIons[idx].pop(0)
#             elif i.startswith("CNOT"):
#                 idx2 = int(i.split(" ")[2])
#                 # Do not duplicate the two qubit gate
#                 ops = operationsForIons[idx][:3] + operationsForIons[idx2][:1]
#                 operationsForIons[idx].pop(0)
#                 operationsForIons[idx].pop(0)
#                 operationsForIons[idx].pop(0)
#                 operationsForIons[idx2].pop(0)
#                 operationsForIons[idx2].pop(0)
#             else:
#                 ops = []

#             physicalZError = 0.0
#             physicalXError = 0.0

#             for op in ops:
#                 if operations.index(op) in barriers:
#                     dephasingTime = barriers[operations.index(op)]
#                     # log(error) = m*log(delay)+c
#                     m = (np.log(0.008)-np.log(0.00001))/((np.log(1)-np.log(0.01)))
#                     c = np.log(0.00001)-m*np.log(0.01)
#                     dephasingInFidelity = np.exp(m*np.log(dephasingTime)+c)
#                     physicalXError+=dephasingInFidelity/2
#                     physicalZError+=dephasingInFidelity/2
#                     for idx in stimIdxs:
#                         circuitString+=f"DEPOLARIZE1({dephasingInFidelity}) {idx}\n"
                        
#             if doNoiseAfter:
#                 circuitString+=f'{i}\n'

#             # for op in ops:
#             #     op.calculateOperationTime()
#             #     op.calculateFidelity()
#             #     opInfidelity = min((1-op.fidelity())/error_scaling, 0.5)
#             #     if len(op.ions)==1: 
#             #         if isinstance(op, QubitReset) or isinstance(op, Measurement):
#             #             physicalXError+=opInfidelity
#             #             circuitString+=f"X_ERROR({opInfidelity}) {stimIdxs[ions.index(op.ions[0])]}\n"
#             #         else:
#             #             physicalXError+=opInfidelity/2
#             #             physicalZError+=opInfidelity/2
#             #             circuitString+=f"DEPOLARIZE1({opInfidelity}) {stimIdxs[ions.index(op.ions[0])]}\n"
#             #     elif len(op.ions)==2:
#             #         physicalXError+=opInfidelity/2
#             #         physicalZError+=opInfidelity/2
#             #         circuitString+=f"DEPOLARIZE2({opInfidelity}) {stimIdxs[ions.index(op.ions[0])]} {stimIdxs[ions.index(op.ions[1])]}\n"
#             #     else:
#             #         raise ValueError(f"simulate: {op} contains {len(op.ions)} ions.")
#             numZGates+=physicalZError>0
#             numXGates+=physicalXError>0
#             meanPhysicalZError += physicalZError
#             meanPhysicalXError += physicalXError
#             if not doNoiseAfter:
#                 circuitString+=f'{i}\n'
#         meanPhysicalZError /= numZGates
#         meanPhysicalXError /= numXGates
#         circuit = stim.Circuit(circuitString)
#         if not decode:
#             return 1, meanPhysicalXError, meanPhysicalZError
#         # Sample the circuit, by using the fast circuit stabilizer tableau simulator provided by Stim.
#         sampler = circuit.compile_detector_sampler()
#         sample =sampler.sample(num_shots, separate_observables=True)
#         detection_events, observable_flips = sample
#         detection_events = np.array(detection_events, order='C')

#         # Construct a Tanner graph, by translating the detector error model using the circuit.
#         detector_error_model = circuit.detector_error_model(decompose_errors=True)
#         matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

#         # Determine the predicted logical observable, by running the MWPM decoding algorithm on the Tanner graph
#         predictions = []
#         for i in range(num_shots):
#             predictions.append(matcher.decode(detection_events[i]))
#         predictions=np.array(predictions)

#         # Count the mistakes.
#         num_errors = 0
#         for shot in range(num_shots):
#             actual_for_shot = observable_flips[shot]
#             predicted_for_shot = predictions[shot]
#             if not np.array_equal(actual_for_shot, predicted_for_shot):
#                 num_errors += 1
#         logicalError = num_errors / num_shots 
#         return logicalError, meanPhysicalXError, meanPhysicalZError


#     def processCircuitWISEArch(
#         self,
#         trapCapacity: int = 2,
#         rows: int = 1,
#         cols: int = 5,
#         dataQubitIdxs: Optional[Sequence[int]]=None,
#     ) -> Tuple[QCCDWiseArch, Sequence[QubitOperation]]:
#         instructions, opBarriers = self._parseCircuitString(dataQubitsIdxs=dataQubitIdxs)
#         if (trapCapacity) * (rows) * cols < len(self._ionMapping):
#             raise ValueError("processCircuit: not enough traps")
        
#         # if trapCapacity % 2!=0:
#         #     raise ValueError("processCircuit: trap capacity must be divisible by 2")
           
#         clusters=regularPartitionWiseArch(self._measurementIons, self._dataIons, trapCapacity)

#         cs, rs = cols, rows
#         allGridPos = []
#         for r in range(rs):
#             for c in range(cs):
#                 allGridPos.append((c, r))
        
#         gridPositions = hillClimbOnArrangeClusters(clusters, allGridPos=allGridPos)

#         wiseArch = {}
#         for trapIdx, (col, row) in enumerate(gridPositions):
#             ions = clusters[trapIdx][0]
#             for i, ion in enumerate(ions):
#                 wiseArch[ion.idx]=(row,col*trapCapacity+i)
#             # numDIons = sum(ion in self._dataIons for ion in ions)
#             # if numDIons > trapCapacity / 2:
#             #     raise ValueError("processCircuit: too many data ions in a trap")
#             # if len(ions)-numDIons > trapCapacity / 2:
#             #     raise ValueError("processCircuit: too many ancilla ions in a trap")
            
#             # dIdx = 0
#             # aIdx = 0
#             # for ion in ions:
#             #     if ion in self._dataIons:
#             #         wiseArch[ion.idx]=(row,col*trapCapacity+2*dIdx)
#             #         dIdx+=1
#             #     else:
#             #         wiseArch[ion.idx]=(row,col*trapCapacity+2*aIdx+1)
#             #         aIdx+=1

#         return QCCDWiseArch(layout=wiseArch, measurementIons=self._measurementIons, dataIons=self._dataIons, rows=rows, cols=cols*trapCapacity, k=trapCapacity), (instructions, opBarriers)
                    
