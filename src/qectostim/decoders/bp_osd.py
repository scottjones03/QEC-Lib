# src/qectostim/decoders/bposd_decoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import stim

from qectostim.decoders.base import Decoder


@dataclass
class BPOSDDecoder(Decoder):
    """Belief-propagation + OSD decoder using stimbposd.

    This expects the `stimbposd` package to be installed.
    """

    dem: stim.DetectorErrorModel
    p: float  # physical error rate / prior
    max_bp_iters: int = 50
    osd_order: int = 2
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        try:
            import stimbposd  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "BPOSDDecoder requires the `stimbposd` package. "
                "Install it via `pip install stimbposd`."
            ) from exc

        self._stimbposd = stimbposd
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables

        # Build an internal decoder object from the DEM.
        # stimbposd typically has helpers like:
        #   stimbposd.Decoder.from_detector_error_model(dem, p, ...)
        # Adjust to the actual API in your version.
        self._decoder = self._stimbposd.Decoder.from_detector_error_model(
            self.dem,
            p=self.p,
            max_bp_iters=self.max_bp_iters,
            osd_order=self.osd_order,
            seed=self.seed,
        )

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        if dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"BPOSDDecoder: expected dets.shape[1]={self.num_detectors}, "
                f"got {dets.shape[1]}"
            )

        # stimbposd typically offers a batch decode function; if not, loop.
        corrections = self._decoder.decode_batch(dets)
        corrections = np.asarray(corrections, dtype=np.uint8)
        if corrections.ndim == 1:
            corrections = corrections.reshape(-1, self.num_observables)
        return corrections