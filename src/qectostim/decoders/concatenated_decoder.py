# src/qectostim/decoders/concatenated_decoder.py
from __future__ import annotations

from typing import Any

import numpy as np
import stim

from qectostim.codes.composite.concatenated import ConcatenatedCode
from qectostim.decoders.base import Decoder
from qectostim.decoders.pymatching_decoder import PyMatchingDecoder


class ConcatenatedDecoder(Decoder):
    """
    Simple concatenated-code decoder.

    Currently this is a thin wrapper that just runs PyMatching on the global
    DEM. Later you can replace this with genuine two-level decoding that
    uses `code.inner` and `code.outer` to decode hierarchically.
    """

    def __init__(self, code: ConcatenatedCode, dem: stim.DetectorErrorModel):
        """
        Parameters
        ----------
        code : ConcatenatedCode
            The composite code object, giving access to inner/outer structure.
        dem : stim.DetectorErrorModel
            Detector error model corresponding to the *full* concatenated code.
        """
        self.code = code
        self.dem = dem
        self._fallback = PyMatchingDecoder(dem=dem)

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """
        Decode a batch of detector outcomes using the fallback PyMatching
        decoder.

        Parameters
        ----------
        dets : np.ndarray
            Boolean or {0,1} array of shape (shots, num_detectors).

        Returns
        -------
        np.ndarray
            Array of predicted logical flips of shape (shots, num_logicals).
            For now num_logicals == 1, consistent with the DEM.
        """
        return self._fallback.decode_batch(dets)