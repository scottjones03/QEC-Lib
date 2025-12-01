# src/src.qectostim/experiments/gadget_experiment.py
from __future__ import annotations
from typing import Any, Dict, Optional, List

import stim

from src.qectostim.experiments import Experiment
from src.qectostim.codes.abstract_code import Code
from src.qectostim.noise.models import NoiseModel
from src.qectostim.noise.stim_noise import apply_noise_to_circuit
from src.qectostim.gadgets.base import Gadget  # we'll define this base


class GadgetExperiment(Experiment):
    """
    Experiment that uses a Gadget (multi-code logical operation) to implement
    a logical gate between several codes, e.g. CSS-surgery CNOT between
    (codeA, codeB).
    """

    def __init__(self,
                 codes: List[Code],
                 noise_model: NoiseModel,
                 gadget: Gadget,
                 metadata: Optional[Dict[str, Any]] = None):
        # For now: treat the first code as the 'primary' for run_decode; metadata can
        # indicate multi-code results.
        super().__init__(codes[0], noise_model, metadata)
        self.codes = codes
        self.gadget = gadget

    def to_stim(self) -> stim.Circuit:
        """
        Delegate circuit construction to the gadget.
        Gadget sees the codes and noise model and returns a Stim circuit that:
         - prepares inputs,
         - runs gadget,
         - measures,
         - defines detectors/observables.
        """
        c = self.gadget.to_stim(self.codes, self.noise_model)
        c = apply_noise_to_circuit(c, self.noise_model)
        return c