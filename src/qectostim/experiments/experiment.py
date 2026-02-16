"""Abstract experiment wrapper: compiles circuits and runs decoding."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from time import time
from typing import Any, Dict, Optional

from dataclasses import dataclass
import numpy as np
import stim

from qectostim.codes.abstract_code import Code
from qectostim.decoders.decoder_selector import select_decoder
from qectostim.decoders.base import Decoder
from qectostim.noise.models import NoiseModel

logger = logging.getLogger(__name__)

@dataclass
class LERResult:
    """One logical error rate measurement."""
    mean_px: float
    mean_pz: float
    logical_error_rate: float
    num_shots: int
    num_errors: int
    num_detectors: int
    num_observables: int
    wall_seconds: float

class Experiment(ABC):
    def __init__(
        self,
        code: Code,
        noise_model: Optional[NoiseModel] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.noise_model = noise_model
        self.metadata = metadata or {}

    @abstractmethod
    def to_stim(self) -> stim.Circuit:
        ...

    def run_no_decode(
        self,
        shots: int = 10_000,
    ) -> LERResult:
        """
        Logical error estimator WITHOUT decoding (ideal measurement).

        Strategy:
          * Build circuit via self.to_stim()
          * Apply noise model (which inserts DEPOLARIZE gates, etc.)
          * Sample the full measurement record from circuit.compile_sampler()
          * Extract the OBSERVABLE_INCLUDE targets from the circuit
          * Use those targets to compute the logical observable from measurement results
          * Compare to ideal (0) to get logical error rate

        This uses the same observable definition as the DEM-based decoder,
        but doesn't attempt to correct detector errors. It's essentially
        "perfect syndrome extraction" without error correction.

        This is useful for understanding the "uncorrected" error rate when
        syndrome information is perfect but not used for correction.
        """
        ...


    def run_decode(
        self,
        noisy_circuit: stim.Circuit,
        shots: int = 100,
        decoder: Optional[Decoder] = None,
    ) -> Dict[str, Any]:
        """
        Distance-aware logical error estimator.

        Common return keys: shots, logical_errors, logical_error_rate
        """
        if decoder is None:
            decoder = select_decoder(self.code)
        # ── Data structures ──────────────────────────────────────────────────
        def _build_dem(circuit):
            """Build DEM with graceful decomposition fallback."""
            for strategy in ['decompose', 'ignore_failures', 'no_decompose']:
                try:
                    if strategy == 'decompose':
                        return circuit.detector_error_model(decompose_errors=True)
                    elif strategy == 'ignore_failures':
                        return circuit.detector_error_model(
                            decompose_errors=True,
                            ignore_decomposition_failures=True,
                        )
                    else:
                        return circuit.detector_error_model(decompose_errors=False)
                except Exception:
                    continue
            raise RuntimeError('Cannot build DEM with any strategy')
        
        def _calculate_mean_physical_z_x_error(circuit):
            """Calculate mean physical Z and X error probabiliies from circuit."""
            return 0.0, 0.0  # Placeholder: implement by scanning stim circuit string
        
        dem = _build_dem(noisy_circuit)
        sampler = dem.compile_sampler()

        t0 = time.time()
        det_samples, obs_samples, _ = sampler.sample(shots=shots)
        corrections = decoder.decode_batch(det_samples)
        predicted_obs = corrections % 2
        actual_obs = obs_samples.astype(np.uint8)
        errors = np.any(predicted_obs != actual_obs, axis=1)
        num_errors = int(errors.sum())
        ler = num_errors / shots
        dt = time.time() - t0
        mean_px, mean_pz = _calculate_mean_physical_z_x_error(noisy_circuit)
        return LERResult(
            mean_px=mean_px, mean_pz=mean_pz, logical_error_rate=ler,
            num_shots=shots, num_errors=num_errors,
            num_detectors=noisy_circuit.num_detectors,
            num_observables=noisy_circuit.num_observables,
            wall_seconds=dt,
        )


            