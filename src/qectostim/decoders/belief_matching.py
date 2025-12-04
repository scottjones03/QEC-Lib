# src/qectostim/decoders/beliefmatching_decoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import stim

from qectostim.decoders.base import Decoder


@dataclass
class BeliefMatchingDecoder(Decoder):
    """Decoder using the `beliefmatching` package on Stim DEMs."""

    dem: stim.DetectorErrorModel
    p: float
    max_iters: int = 50
    damping: float = 0.0  # if supported

    def __post_init__(self) -> None:
        try:
            import beliefmatching as bm  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "BeliefMatchingDecoder requires the `beliefmatching` package. "
                "Install it via `pip install beliefmatching`."
            ) from exc

        self._bm = bm
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables

        # Convert DEM into the internal graph representation used by beliefmatching.
        # The library typically has something like:
        #   graph = bm.StimGraph.from_dem(dem)
        self._graph = self._bm.StimGraph.from_dem(self.dem)

        # Create a decoder object.
        self._decoder = self._bm.Decoder(
            graph=self._graph,
            p=self.p,
            max_iters=self.max_iters,
            damping=self.damping,
        )

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        if dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"BeliefMatchingDecoder: expected dets.shape[1]={self.num_detectors}, "
                f"got {dets.shape[1]}"
            )

        corrections = self._decoder.decode_batch(dets)
        corrections = np.asarray(corrections, dtype=np.uint8)
        if corrections.ndim == 1:
            corrections = corrections.reshape(-1, self.num_observables)
        return corrections