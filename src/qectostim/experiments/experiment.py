# src/src.qectostim/experiments/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import stim

from src.qectostim.codes.abstract_code import Code
from src.qectostim.noise.models import NoiseModel
from src.qectostim.decoders.decoder_selector import select_decoder


class Experiment(ABC):
    """
    Abstract experiment: takes a Code and a NoiseModel, can:
      - build a Stim circuit (to_stim)
      - run decoding and return results (run_decode)
    """

    def __init__(self,
                 code: Code,
                 noise_model: NoiseModel,
                 metadata: Optional[Dict[str, Any]] = None):
        self.code = code
        self.noise_model = noise_model
        self.metadata = metadata or {}

    @abstractmethod
    def to_stim(self) -> stim.Circuit:
        """Build the Stim circuit representing this experiment."""

    def run_decode(self,
                   shots: int = 10_000,
                   decoder_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Compile circuit, simulate with Stim, decode using an appropriate decoder.
        Returns a result dict (e.g. logical error rates, raw samples, etc.).
        """
        circuit = self.to_stim()

        # Build DEM from circuit
        dem = circuit.detector_error_model(decompose_errors=True)

        # Select decoder based on DEM + optional override
        decoder = select_decoder(dem, preferred=decoder_name)

        # Compile detector sampler
        sampler = circuit.compile_detector_sampler()

        dets, obs = sampler.sample(
            shots=shots,
            separate_observables=True
        )

        # Decode each shot
        logical_errors = decoder.decode_batch(dets, obs)

        return {
            "shots": shots,
            "logical_errors": logical_errors,
            "logical_error_rate": float(sum(logical_errors) / shots),
        }