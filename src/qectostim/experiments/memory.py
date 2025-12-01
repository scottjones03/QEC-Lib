# src/src.qectostim/experiments/memory.py
from __future__ import annotations
from typing import Any, Dict, Optional

import stim

from src.qectostim.experiments import Experiment
from src.qectostim.codes.abstract_code import Code
from src.qectostim.noise.models import NoiseModel
from src.qectostim.noise.stim_noise import apply_noise_to_circuit


class MemoryExperiment(Experiment):
    """
    Repeated stabilizer measurement (memory) experiment for a single logical qubit
    in the given code.
    """

    def __init__(self,
                 code: Code,
                 noise_model: NoiseModel,
                 rounds: int,
                 logical_qubit: int = 0,
                 initial_state: str = "0",
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(code, noise_model, metadata)
        self.rounds = rounds
        self.logical_qubit = logical_qubit
        self.initial_state = initial_state

    def to_stim(self) -> stim.Circuit:
        c = stim.Circuit()

        # 1) Reset all physical qubits
        c.append("R", range(self.code.n))

        # 2) Prepare logical state (e.g. |0_L>, |+_L> etc.)
        # TODO: use code.logical_x_ops() / logical_z_ops() to build encoder or rely
        # on repeated stabilizer measurement to project.
        # For now: placeholder.

        # 3) REPEAT { stabilizer measurement + noise }
        body = stim.Circuit()

        # TODO: build 1-round stabilizer measurement circuit from code.stabilizers()
        # and maybe geometry / ancilla allocation.

        # apply noise according to noise_model
        body = apply_noise_to_circuit(body, self.noise_model)

        c.append("REPEAT", self.rounds, body)

        # 4) Final measurement of data qubits
        c.append("MZ", range(self.code.n))

        # TODO: add DETECTORS and OBSERVABLE_INCLUDE based on checks + logicals.

        return c