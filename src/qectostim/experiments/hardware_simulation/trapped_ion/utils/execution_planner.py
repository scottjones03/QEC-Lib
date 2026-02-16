"""
Execution planner for trapped-ion QCCD hardware.

Builds a :class:`~core.execution.ExecutionPlan` from a set of
compiled operations and calibration constants.  The plan captures
per-gate timing, idle intervals, and gate-swap metadata so that
downstream noise injection can account for real hardware physics.
"""

from __future__ import annotations

import stim
from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Dict,
    Any,
)

from qectostim.experiments.hardware_simulation.core.execution import (
    ExecutionPlan,
    OperationTiming,
    GateSwapInfo,
    IdleInterval,
)
from ...core.pipeline import CompiledCircuit
from .qccd_nodes import Ion, SpectatorIon
from .qccd_operations import Operation, GateSwap
from .qccd_operations_on_qubits import QubitOperation
from .physics import CalibrationConstants, DEFAULT_CALIBRATION
from ..compiler.qccd_parallelisation import calculateDephasingFromIdling


class TrappedIonExecutionPlanner:
    """Build a physics-aware :class:`ExecutionPlan` from compiled operations.

    Takes the same inputs that ``QCCDCircuit._simulate_with_timing_aware_noise``
    used to consume and packages the timing + gate-swap information into the
    core ``ExecutionPlan`` dataclass.

    Parameters
    ----------
    calibration : CalibrationConstants, optional
        Hardware calibration data.  Uses ``DEFAULT_CALIBRATION`` when
        ``None``.
    """

    def __init__(
        self,
        calibration: Optional[CalibrationConstants] = None,
    ):
        self.cal = calibration or DEFAULT_CALIBRATION

    def create_plan(
        self,
        compiled_circuit: CompiledCircuit,
    ) -> ExecutionPlan:
        """Build an ExecutionPlan from a CompiledCircuit.

        Extracts the ion mapping, routed operations, dephasing schedule,
        and per-qubit stim instruction stream, packing everything into
        ``ExecutionPlan.metadata`` for use by
        :meth:`TrappedIonNoiseModel.apply_with_plan`.
        """
        # ----------------------------------------------------------
        # 1. Navigate the compilation chain to extract data
        # ----------------------------------------------------------
        routed = compiled_circuit.scheduled.routed_circuit
        mapped = routed.mapped_circuit
        native = mapped.native_circuit

        ion_mapping = native.metadata["ion_mapping"]
        operations = routed.metadata.get(
            "all_operations", list(routed.operations)
        )
        is_wise = routed.metadata.get("is_wise", False)

        # ----------------------------------------------------------
        # 2. Build stimIdx / ion lists (skip spectators)
        # ----------------------------------------------------------
        stimIdxs: List[int] = []
        ions: List[Ion] = []
        for stimIdx, (ion, _) in ion_mapping.items():
            if isinstance(ion, SpectatorIon):
                continue
            stimIdxs.append(stimIdx)
            ions.append(ion)

        # ----------------------------------------------------------
        # 3. Build per-ion operation queues + gate-swap associations
        # ----------------------------------------------------------
        operationsForIons: Dict[int, List[QubitOperation]] = {
            stimIdx: [] for stimIdx in stimIdxs
        }
        gateSwapsForIons: Dict[int, List[Tuple[int, GateSwap]]] = {
            stimIdx: [] for stimIdx in stimIdxs
        }
        qubitOps: List[QubitOperation] = []
        for op in operations:
            if isinstance(op, GateSwap):
                ion = op.ions[0]  # source ion in the gate swap
                stimIdxForIon = stimIdxs[ions.index(ion)]
                opForIonIdx = len(operationsForIons[stimIdxForIon])
                gateSwapsForIons[stimIdxForIon].append((opForIonIdx, op))
            elif isinstance(op, QubitOperation):
                for ion in op.ions:
                    operationsForIons[stimIdxs[ions.index(ion)]].append(op)
                qubitOps.append(op)

        gateSwapsForOperations: Dict[QubitOperation, List[GateSwap]] = {
            op: [] for op in qubitOps
        }
        for stimIdx, gateSwaps in gateSwapsForIons.items():
            for (opForIonIdx, op) in gateSwaps:
                gateSwapsForOperations[
                    operationsForIons[stimIdx][opForIonIdx]
                ].append(op)

        # ----------------------------------------------------------
        # 4. Dephasing schedule
        # ----------------------------------------------------------
        dephasingSchedule = dict(
            calculateDephasingFromIdling(operations, is_wise)
        )

        # ----------------------------------------------------------
        # 5. Build circuitString (per-qubit stim instructions)
        #    Replicates QCCDCircuit.circuitString(include_annotation=True)
        # ----------------------------------------------------------
        ideal = compiled_circuit.original_circuit
        raw = (
            ideal.flattened()
            .decomposed()
            .without_noise()
            .__str__()
            .splitlines()
        )

        stim_instructions: List[str] = []
        for line in raw:
            qubits = line.rsplit(" ")[1:]
            if (
                line.startswith("DETECTOR")
                or line.startswith("TICK")
                or line.startswith("OBSERVABLE")
            ):
                stim_instructions.append(line)
            elif line and line[0] in ("R", "H", "M"):
                for qubit in qubits:
                    stim_instructions.append(f"{line[0]} {qubit}")
            elif any(
                line.startswith(s)
                for s in stim.gate_data("cnot").aliases
            ):
                for i in range(len(qubits) // 2):
                    stim_instructions.append(
                        f"CNOT {qubits[2*i]} {qubits[2*i+1]}"
                    )
                stim_instructions.append("BARRIER")
            elif any(
                line.startswith(s)
                for s in stim.gate_data("cz").aliases
            ):
                for i in range(len(qubits) // 2):
                    stim_instructions.append(
                        f"CZ {qubits[2*i]} {qubits[2*i+1]}"
                    )
                stim_instructions.append("BARRIER")
            else:
                stim_instructions.append(line)

        # ----------------------------------------------------------
        # 6. Pack into ExecutionPlan.metadata
        # ----------------------------------------------------------
        return ExecutionPlan(
            metadata={
                "operationsForIons": operationsForIons,
                "gateSwapsForOperations": gateSwapsForOperations,
                "dephasingSchedule": dephasingSchedule,
                "stimIdxs": stimIdxs,
                "ions": ions,
                "qubitOps": qubitOps,
                "stim_instructions": stim_instructions,
            }
        )


# Alias for convenience (experiment.py imports ExecutionPlanner)
ExecutionPlanner = TrappedIonExecutionPlanner