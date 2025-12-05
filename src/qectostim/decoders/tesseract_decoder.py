# src/qectostim/decoders/tesseract_decoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import stim

from qectostim.decoders.base import Decoder


@dataclass
class TesseractDecoder(Decoder):
    """Wrapper for the tesseract tensor-network decoder on Stim DEMs.

    Uses the tesseract_decoder package which provides efficient tensor-network
    based decoding. Particularly effective for codes with complex error structures.
    
    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decode.
    det_beam : int, default=5
        Beam search width for detector ordering.
    merge_errors : bool, default=True
        Whether to merge similar error mechanisms.
    """

    dem: stim.DetectorErrorModel
    det_beam: int = 5
    merge_errors: bool = True

    def __post_init__(self) -> None:
        try:
            import tesseract_decoder as tdec  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "TesseractDecoder requires the `tesseract-decoder` package. "
                "Install it via `pip install tesseract-decoder`."
            ) from exc

        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables

        # Build decoder via TesseractConfig
        config = tdec.tesseract.TesseractConfig()
        config.det_beam = self.det_beam
        config.merge_errors = 1 if self.merge_errors else 0
        
        self._decoder = config.compile_decoder_for_dem(self.dem)

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        if dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"TesseractDecoder: expected dets.shape[1]={self.num_detectors}, "
                f"got {dets.shape[1]}"
            )

        shots = dets.shape[0]
        corrections = np.zeros((shots, self.num_observables), dtype=np.uint8)
        
        for i in range(shots):
            # decode returns a list of bools for each observable
            corr = self._decoder.decode(dets[i].astype(bool))
            corrections[i, :] = np.asarray(corr, dtype=np.uint8)
            
        return corrections