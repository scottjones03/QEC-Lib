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

    Uses the stimbposd package which directly accepts Stim DetectorErrorModels.
    BP+OSD is particularly effective for LDPC and QLDPC codes.
    
    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decode.
    max_bp_iters : int, default=30
        Maximum number of belief propagation iterations.
    bp_method : str, default='product_sum'
        BP algorithm variant. Options: 'product_sum', 'minimum_sum'.
    osd_order : int, default=60
        Order of OSD post-processing.
    osd_method : str, default='osd_cs'
        OSD algorithm variant.
    """

    dem: stim.DetectorErrorModel
    max_bp_iters: int = 30
    bp_method: str = "product_sum"
    osd_order: int = 60
    osd_method: str = "osd_cs"

    def __post_init__(self) -> None:
        try:
            from stimbposd import BPOSD  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "BPOSDDecoder requires the `stimbposd` package. "
                "Install it via `pip install stimbposd`."
            ) from exc

        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables

        # Clamp osd_order to valid range [0, num_detectors - 1]
        effective_osd_order = max(0, min(self.osd_order, self.num_detectors - 1))

        # BPOSD accepts the DEM directly
        self._decoder = BPOSD(
            self.dem,
            max_bp_iters=self.max_bp_iters,
            bp_method=self.bp_method,
            osd_order=effective_osd_order,
            osd_method=self.osd_method,
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

        corrections = self._decoder.decode_batch(dets)
        corrections = np.asarray(corrections, dtype=np.uint8)
        if corrections.ndim == 1:
            corrections = corrections.reshape(-1, self.num_observables)
        return corrections