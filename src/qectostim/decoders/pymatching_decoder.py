# qectostim/decoders/pymatching_decoder.py

from __future__ import annotations

from typing import Any

import numpy as np
import stim
import pymatching
from qectostim.decoders.base import Decoder
from dataclasses import dataclass

@dataclass
class PyMatchingDecoder(Decoder):
    dem: stim.DetectorErrorModel

    def __post_init__(self) -> None:
        if pymatching is None:
            raise RuntimeError(
                "pymatching is not installed. Install it with `pip install "
                "pymatching` or select a different decoder."
            ) 

        self._matching = pymatching.Matching.from_detector_error_model(self.dem)
        self._num_observables = self.dem.num_observables

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.int8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        out = self._matching.decode_batch(dets)
        out = np.asarray(out, dtype=np.uint8)

        if out.ndim == 1:
            out = out.reshape(-1, self._num_observables)

        return out