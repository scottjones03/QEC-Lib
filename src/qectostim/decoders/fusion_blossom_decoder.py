# src/qectostim/decoders/fusion_blossom_decoder.py
from __future__ import annotations

from typing import Any

import numpy as np
import stim

from qectostim.decoders.base import Decoder


class FusionBlossomDecoder(Decoder):
    """
    Wrapper around fusion_blossom's matching decoder built from a Stim DEM.

    This expects fusion_blossom to be installed. If it isn't, construction will
    fail with a clear error.
    """

    def __init__(self, dem: stim.DetectorErrorModel):
        """
        Parameters
        ----------
        dem : stim.DetectorErrorModel
            Detector error model describing correlations between detectors and
            logical observables.
        """
        try:
            import fusion_blossom as fb  # type: ignore
            from fusion_blossom import stim_util  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "FusionBlossomDecoder requires the 'fusion_blossom' package. "
                "Install it via `pip install fusion-blossom`."
            ) from exc

        self.dem = dem
        self._fb = fb
        self._stim_util = stim_util

        # Build the fusion-blossom decoder object from the DEM.
        # stim_util.stim_dem_to_fusion_blossom is the standard helper.
        self._graph, self._decoder = self._stim_util.stim_dem_to_fusion_blossom(
            dem
        )

        # Cache detector and observable counts for sanity checks.
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """
        Decode a batch of detector outcomes.

        Parameters
        ----------
        dets : np.ndarray
            Boolean / {0,1} array of shape (shots, num_detectors).

        Returns
        -------
        np.ndarray
            Array of predicted logical flips of shape (shots, num_observables).

        Notes
        -----
        We assume `dets.shape[1] == dem.num_detectors`. If not, a ValueError is
        raised.
        """
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim != 2:
            raise ValueError(
                f"FusionBlossomDecoder.decode_batch expected 2D array, got shape {dets.shape}"
            )
        if dets.shape[1] != self.num_detectors:
            raise ValueError(
                "FusionBlossomDecoder.decode_batch: dets has num_detectors "
                f"{dets.shape[1]}, but DEM has {self.num_detectors}."
            )

        shots = dets.shape[0]
        # fusion-blossom expects a list of syndrome vectors
        corrections = np.zeros((shots, self.num_observables), dtype=np.uint8)

        for i in range(shots):
            syndrome = dets[i].astype(bool)
            # decode returns a correction bitstring over observables
            # depending on fusion-blossom's API, this may need conversion
            corr = self._decoder.decode(syndrome)
            corr = np.asarray(corr, dtype=np.uint8).reshape(-1)
            if corr.size != self.num_observables:
                raise ValueError(
                    "FusionBlossomDecoder: decoder returned correction of size "
                    f"{corr.size}, expected {self.num_observables}."
                )
            corrections[i, :] = corr

        return corrections