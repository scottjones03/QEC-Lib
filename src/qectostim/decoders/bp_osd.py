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
            from stimbposd.dem_to_matrices import detector_error_model_to_check_matrices  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "BPOSDDecoder requires the `stimbposd` package. "
                "Install it via `pip install stimbposd`."
            ) from exc

        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables
        self.num_errors = self.dem.num_errors

        # Work around a stimbposd bug: BPOSD.__init__ computes
        #   max_osd = ncols - nrows   (= num_errors - num_detectors)
        # and passes min(osd_order, max_osd) to the ldpc BpOsdDecoder.
        # When num_errors < num_detectors, max_osd is negative and ldpc
        # raises ValueError.  We bypass BPOSD and call ldpc directly.
        #
        # IMPORTANT: The correct max OSD order is (n_err - rank(H)),
        # NOT (n_err - n_det).  For concatenated codes, H is highly
        # overdetermined (many linearly dependent rows), so rank(H) can
        # be much less than n_det.  We let ldpc compute the rank
        # internally rather than clamping to 0.
        matrices = detector_error_model_to_check_matrices(
            self.dem, allow_undecomposed_hyperedges=True
        )
        self._matrices = matrices

        # Let ldpc handle OSD order clamping internally — it will
        # compute rank and limit the search space appropriately.
        effective_osd_order = self.osd_order
        effective_method = self.osd_method

        try:
            from ldpc import BpOsdDecoder as LdpcBpOsd  # type: ignore
            self._decoder_raw = LdpcBpOsd(
                matrices.check_matrix,
                max_iter=self.max_bp_iters,
                bp_method=self.bp_method,
                error_channel=list(matrices.priors),
                osd_order=effective_osd_order,
                osd_method=effective_method,
                input_vector_type="syndrome",
            )
            self._use_raw = True
        except ImportError:
            # Fallback: old stimbposd path (will raise if max_osd < 0)
            self._decoder = BPOSD(
                self.dem,
                max_bp_iters=self.max_bp_iters,
                bp_method=self.bp_method,
                osd_order=effective_osd_order,
                osd_method=effective_method,
            )
            self._use_raw = False

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        if dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"BPOSDDecoder: expected dets.shape[1]={self.num_detectors}, "
                f"got {dets.shape[1]}"
            )

        if not self._use_raw:
            corrections = self._decoder.decode_batch(dets)
            corrections = np.asarray(corrections, dtype=np.uint8)
            if corrections.ndim == 1:
                corrections = corrections.reshape(-1, self.num_observables)
            return corrections

        # Raw ldpc decoder path: decode each shot individually, then map
        # error predictions → observable corrections via the observables
        # matrix from the DEM.  obs_matrix shape is (num_obs, num_errors).
        obs_matrix = self._matrices.observables_matrix
        # Convert sparse to dense for reliable matmul with 1-D vectors
        obs_dense = np.asarray(obs_matrix.todense(), dtype=np.uint8) if hasattr(obs_matrix, 'todense') else np.asarray(obs_matrix, dtype=np.uint8)
        n_shots = dets.shape[0]
        corrections = np.zeros((n_shots, self.num_observables), dtype=np.uint8)
        for i in range(n_shots):
            error_pred = self._decoder_raw.decode(dets[i])
            error_pred = np.asarray(error_pred, dtype=np.uint8)
            # obs_dense: (num_obs, num_errors), error_pred: (num_errors,)
            obs_flip = (obs_dense @ error_pred) % 2
            corrections[i] = obs_flip.astype(np.uint8)
        return corrections