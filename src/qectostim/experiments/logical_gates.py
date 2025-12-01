# src/src.qectostim/experiments/logical_gates.py
from __future__ import annotations
from typing import Any, Dict, Optional, List

import stim

from src.qectostim.experiments import Experiment
from src.qectostim.codes.abstract_code import Code
from src.qectostim.noise.models import NoiseModel
from src.qectostim.noise.stim_noise import apply_noise_to_circuit


class LogicalGateExperiment(Experiment):
    """
    Runs a logical gate (within a single code block) and checks if it succeeded.

    Example: apply logical H, S, or logical CNOT between two logical qubits
    of the SAME code block using transversal or teleportation gadgets that stay inside
    the code.
    """

    def __init__(self,
                 code: Code,
                 noise_model: NoiseModel,
                 gate_name: str,
                 logical_qubits: List[int],
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(code, noise_model, metadata)
        self.gate_name = gate_name
        self.logical_qubits = logical_qubits

    def to_stim(self) -> stim.Circuit:
        c = stim.Circuit()

        # TODO:
        # 1) Prepare logical input state(s)
        # 2) Apply fault-tolerant implementation of gate_name
        #    (e.g. transversal, in-code teleportation, etc.)
        # 3) Measure outputs and define OBSERVABLE_INCLUDE.

        # Insert noise as we go or globally via apply_noise_to_circuit
        c = apply_noise_to_circuit(c, self.noise_model)
        return c