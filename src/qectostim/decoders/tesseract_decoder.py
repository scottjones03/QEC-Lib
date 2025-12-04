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

    Requires the `tesseract-decoder` Python package (or whatever name it uses
    in your environment). The exact import may need tweaking based on the
    installed package.
    """

    dem: stim.DetectorErrorModel
    max_bond_dim: int = 16
    error_model: str = "phenomenological"

    def __post_init__(self) -> None:
        try:
            import tesseract as tdec  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "TesseractDecoder requires the `tesseract` package. "
                "Install it (e.g. `pip install tesseract-decoder`), "
                "or choose a different decoder."
            ) from exc

        self._tdec = tdec
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables

        # Build a tesseract model from the DEM. The exact API may vary slightly
        # depending on the version of tesseract you install; adjust as needed.
        #
        # Typical pattern (from the repo) is something like:
        #   model = tdec.Decoder.from_stim_dem(dem, max_bond_dim=..., ...)
        self._decoder = self._tdec.Decoder.from_stim_dem(
            self.dem,
            max_bond_dim=self.max_bond_dim,
            error_model=self.error_model,
        )

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
            s = dets[i].astype(bool)
            # Most tesseract APIs return a correction bitstring over observables.
            corr = self._decoder.decode(s)
            corr = np.asarray(corr, dtype=np.uint8).reshape(-1)
            if corr.size != self.num_observables:
                raise ValueError(
                    "TesseractDecoder: decoder returned size "
                    f"{corr.size}, expected {self.num_observables}."
                )
            corrections[i, :] = corr
        return corrections