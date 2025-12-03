# src/qectostim/decoders/union_find_decoder.py
from __future__ import annotations

from typing import Any

import numpy as np
import stim

from qectostim.decoders.base import Decoder
from qectostim.decoders.pymatching_decoder import PyMatchingDecoder


class UnionFindDecoder(Decoder):
    """
    Placeholder union-find decoder that currently delegates to PyMatching.

    This keeps the interface stable and lets you swap in a genuine union-find
    implementation later without touching Experiment.run_decode.
    """

    def __init__(self, dem: stim.DetectorErrorModel):
        # inside __init__
        try:
            import some_union_find_lib as uf  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "UnionFindDecoder requires 'some_union_find_lib'. "
                "Install it or stick with the PyMatching fallback."
            ) from exc

        self._uf = uf.build_from_stim_dem(dem)  # hypothetical helper
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """
        Decode a batch of detector outcomes via the fallback PyMatching decoder.

        Parameters
        ----------
        dets : np.ndarray
            Boolean / {0,1} array of shape (shots, num_detectors).

        Returns
        -------
        np.ndarray
            Array of predicted logical flips of shape (shots, num_observables).
        """
        return self._fallback.decode_batch(dets)