"""
Trapped-ion hardware noise model.

Implements :class:`~qectostim.noise.hardware.base.HardwareNoiseModel` for
QCCD trapped-ion hardware.  The key method is :meth:`apply_with_plan`, which
walks the stim instruction stream and injects:

1. **Dephasing** (``Z_ERROR``) from idle intervals computed by
   :func:`calculateDephasingFromIdling`.
2. **Gate-swap depolarisation** (``DEPOLARIZE2``) when ions are
   physically swapped for routing.
3. **Per-gate infidelity** — ``X_ERROR`` for measurement/reset,
   ``DEPOLARIZE1`` for 1-qubit gates, ``DEPOLARIZE2`` for 2-qubit
   gates.

Measurement instructions receive noise *before* the gate; all other
instructions receive noise *after*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Any,
    TYPE_CHECKING,
)

import copy

import stim

from qectostim.noise.hardware.base import (
    CalibrationData,
    HardwareNoiseModel,
    NoiseChannel,
    NoiseChannelType,
)

from .qccd_nodes import Ion, SpectatorIon
from .qccd_operations import Operation, GateSwap
from .qccd_operations_on_qubits import (
    QubitOperation,
    QubitReset,
    Measurement,
)
from .physics import CalibrationConstants, DEFAULT_CALIBRATION

from ..compiler.qccd_parallelisation import calculateDephasingFromIdling

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.execution import (
        ExecutionPlan,
    )
    from qectostim.experiments.hardware_simulation.core.operations import (
        PhysicalOperation,
    )


# ---------------------------------------------------------------------------
# CalibrationData adapter
# ---------------------------------------------------------------------------

@dataclass
class TrappedIonCalibration(CalibrationData):
    """Adapter that wraps :class:`CalibrationConstants` as a :class:`CalibrationData`.

    The framework's generic :class:`CalibrationData` stores per-qubit
    fidelities keyed by ``(qubit, gate)`` while the trapped-ion code
    derives fidelities from a chain-length formula.  This adapter
    exposes the CalibrationConstants through the CalibrationData API
    so the ``HardwareNoiseModel`` base class can use either pathway.
    """

    constants: CalibrationConstants = field(
        default_factory=CalibrationConstants
    )

    def __post_init__(self):
        # Populate CalibrationData fields from CalibrationConstants
        gt = self.constants.gate_times_us()
        self.gate_times = {k: v for k, v in gt.items()}
        self.heating_rate = self.constants.heating_rate
        self.metadata["platform"] = "trapped_ion"

    # Convenience delegate to CalibrationConstants
    def get_1q_fidelity(
        self, qubit: int, gate: str, default: float = 0.999
    ) -> float:
        fids = self.constants.gate_fidelities()
        return fids.get(gate, default)

    def get_2q_fidelity(
        self,
        qubit1: int,
        qubit2: int,
        gate: str,
        default: float = 0.99,
    ) -> float:
        fids = self.constants.gate_fidelities()
        return fids.get(gate, default)

    def get_readout_fidelity(
        self,
        qubit: int,
        default: Tuple[float, float] = (0.999, 0.999),
    ) -> Tuple[float, float]:
        mf = 1.0 - self.constants.measurement_infidelity
        return (mf, mf)


# ---------------------------------------------------------------------------
# TrappedIonNoiseModel
# ---------------------------------------------------------------------------

class TrappedIonNoiseModel(HardwareNoiseModel):
    """QCCD trapped-ion noise model.

    The main entry point is :meth:`apply_with_plan`, which uses
    QCCD-specific operation lists, ion mappings, and dephasing
    schedules stored in ``plan.metadata`` to inject realistic noise.

    Falls back to :meth:`apply` (the calibration-data-based default
    from the base class) when no execution plan is available.

    Parameters
    ----------
    calibration : CalibrationConstants, optional
        Device calibration constants.  Uses ``DEFAULT_CALIBRATION``
        when ``None``.
    error_scaling : float
        Scale factor for all error probabilities (for studies).
    """

    def __init__(
        self,
        calibration: Optional[CalibrationConstants] = None,
        error_scaling: float = 1.0,
    ):
        self._cal = calibration or DEFAULT_CALIBRATION
        ti_cal = TrappedIonCalibration(constants=self._cal)
        super().__init__(
            calibration=ti_cal,
            error_scaling=error_scaling,
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def apply_to_operation(
        self,
        operation: "PhysicalOperation",
    ) -> List[NoiseChannel]:
        """Per-operation noise channels (used by base `apply()`)."""
        channels: List[NoiseChannel] = []
        name = operation.gate_name.upper()
        qubits = operation.qubits

        fids = self._cal.gate_fidelities()
        fid = fids.get(name, 0.999)
        error_prob = min((1.0 - fid) / self.error_scaling, 0.5)

        if error_prob <= 0:
            return channels

        if name in ("M", "MX", "MY", "MZ", "MR", "R"):
            for q in qubits:
                channels.append(
                    NoiseChannel(
                        channel_type=NoiseChannelType.BIT_FLIP,
                        probability=error_prob,
                        qubits=(q,),
                    )
                )
        elif len(qubits) == 1:
            channels.append(
                NoiseChannel(
                    channel_type=NoiseChannelType.DEPOLARIZING_1Q,
                    probability=error_prob,
                    qubits=tuple(qubits),
                )
            )
        elif len(qubits) >= 2:
            for i in range(0, len(qubits) - 1, 2):
                channels.append(
                    NoiseChannel(
                        channel_type=NoiseChannelType.DEPOLARIZING_2Q,
                        probability=error_prob,
                        qubits=(qubits[i], qubits[i + 1]),
                    )
                )
        return channels

    def idle_noise(
        self,
        qubit: int,
        duration: float,
    ) -> List[NoiseChannel]:
        """Dephasing noise for idle time (T2 decay)."""
        if duration <= 0:
            return []
        import math

        t2 = self._cal.t2_time  # seconds
        # duration is in microseconds from the plan
        duration_s = duration * 1e-6
        # Dephasing probability: p = 0.5 * (1 - exp(-t/T2))
        p = 0.5 * (1 - math.exp(-duration_s / t2))
        p = min(p / self.error_scaling, 0.5)
        if p <= 0:
            return []
        return [
            NoiseChannel(
                channel_type=NoiseChannelType.DEPHASING,
                probability=p,
                qubits=(qubit,),
            )
        ]

    # ------------------------------------------------------------------
    # QCCD-specific noise injection (the "inline" walk)
    # ------------------------------------------------------------------

    def apply_with_plan(
        self,
        circuit: stim.Circuit,
        plan: ExecutionPlan,
    ) -> stim.Circuit:
        """Apply QCCD noise using the operation-level dephasing schedule.
        """
 
        stim_instructions = plan.metadata.get("stim_instructions") or circuit.__str__().splitlines()

        # Extract pre-built data structures from the execution plan
        operationsForIons = copy.deepcopy(plan.metadata["operationsForIons"])
        gateSwapsForOperations = plan.metadata["gateSwapsForOperations"]
        dephasingSchedule = plan.metadata["dephasingSchedule"]
        stimIdxs = plan.metadata["stimIdxs"]
        ions = plan.metadata["ions"]
        qubitOps = plan.metadata["qubitOps"]
        error_scaling = self.error_scaling

        # Build an idx-based lookup so we aren't reliant on Ion identity.
        # After deep-copy (above) or architecture resets, Ion references in
        # operations may differ from those in `ions`.
        _ion_idx_to_pos = {ion.idx: pos for pos, ion in enumerate(ions)}

        def _ion_pos(ion_obj):
            """Return position of *ion_obj* in the ``ions`` list."""
            pos = _ion_idx_to_pos.get(ion_obj.idx)
            if pos is not None:
                return pos
            # Fallback: identity search (original behaviour)
            return ions.index(ion_obj)

        meanPhysicalZError = 0.0
        meanPhysicalXError = 0.0
        numZGates = 0
        numXGates = 0
        circuitString = ""

        for instr in stim_instructions:
            if instr.startswith("BARRIER"):
                continue
            idx = (
                int(instr.split(" ")[1])
                if (
                    instr[0] in ("M", "H", "R")
                    or instr.startswith("CNOT")
                    or instr.startswith("CZ")
                )
                else -1
            )
            doNoiseAfter = instr[0] != "M"

            # --- Pop matching operations per instruction type ---
            if instr[0] == "M" or instr[0] == "R":
                ops = operationsForIons[idx][:1]
                operationsForIons[idx].pop(0)
            elif instr[0] == "H":
                ops = operationsForIons[idx][:2]
                operationsForIons[idx].pop(0)
                operationsForIons[idx].pop(0)
            elif instr.startswith("CNOT"):
                idx2 = int(instr.split(" ")[2])
                ops = (
                    operationsForIons[idx][:4]
                    + operationsForIons[idx2][:1]
                )
                operationsForIons[idx].pop(0)
                operationsForIons[idx].pop(0)
                operationsForIons[idx].pop(0)
                operationsForIons[idx2].pop(0)
                operationsForIons[idx2].pop(0)
            elif instr.startswith("CZ"):
                idx2 = int(instr.split(" ")[2])
                ops = (
                    operationsForIons[idx][:4]
                    + operationsForIons[idx2][:2]
                    + operationsForIons[idx2][3:5]
                )
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

            # --- Dephasing + gate-swap noise ---
            physicalZError = 0.0
            physicalXError = 0.0
            for op in ops:
                if op in qubitOps:
                    for ion_obj in op.ions:
                        if len(dephasingSchedule.get(ion_obj, [])) > 0:
                            dephasing = [
                                dephasingFidelity
                                for opAtEndOfIdle, dephasingFidelity
                                in dephasingSchedule[ion_obj]
                                if opAtEndOfIdle == op
                            ]
                            if dephasing:
                                dephasingInFidelity = min(
                                    (1 - dephasing[0]) / error_scaling,
                                    0.5,
                                )
                                physicalZError += dephasingInFidelity
                                circuitString += (
                                    f"Z_ERROR({dephasingInFidelity}) "
                                    f"{stimIdxs[_ion_pos(ion_obj)]}\n"
                                )
                    for gs in gateSwapsForOperations.get(op, []):
                        gsInfidelity = min(
                            (1 - gs.fidelity()) / error_scaling, 0.5
                        )
                        physicalXError += gsInfidelity / 2
                        physicalZError += gsInfidelity / 2
                        circuitString += (
                            f"DEPOLARIZE2({gsInfidelity}) "
                            f"{stimIdxs[_ion_pos(gs.ions[0])]} "
                            f"{stimIdxs[_ion_pos(gs.ions[1])]}\n"
                        )

            # --- Emit instruction (before or after noise) ---
            if doNoiseAfter:
                circuitString += f"{instr}\n"

            # --- Per-gate infidelity noise ---
            for op in ops:
                opInfidelity = min(
                    (1 - op.fidelity()) / error_scaling, 0.5
                )
                if len(op.ions) == 1:
                    if isinstance(op, (QubitReset, Measurement)):
                        physicalXError += opInfidelity
                        circuitString += (
                            f"X_ERROR({opInfidelity}) "
                            f"{stimIdxs[_ion_pos(op.ions[0])]}\n"
                        )
                    else:
                        physicalXError += opInfidelity / 2
                        physicalZError += opInfidelity / 2
                        circuitString += (
                            f"DEPOLARIZE1({opInfidelity}) "
                            f"{stimIdxs[_ion_pos(op.ions[0])]}\n"
                        )
                elif len(op.ions) == 2:
                    physicalXError += opInfidelity / 2
                    physicalZError += opInfidelity / 2
                    circuitString += (
                        f"DEPOLARIZE2({opInfidelity}) "
                        f"{stimIdxs[_ion_pos(op.ions[0])]} "
                        f"{stimIdxs[_ion_pos(op.ions[1])]}\n"
                    )
                else:
                    raise ValueError(
                        f"_inject_inline_noise: {op} contains "
                        f"{len(op.ions)} ions."
                    )

            numZGates += physicalZError > 0
            numXGates += physicalXError > 0
            meanPhysicalZError += physicalZError
            meanPhysicalXError += physicalXError

            if not doNoiseAfter:
                circuitString += f"{instr}\n"

        meanPhysicalZError /= max(numZGates, 1)
        meanPhysicalXError /= max(numXGates, 1)

        noisy_circuit = stim.Circuit(circuitString)
        return noisy_circuit
