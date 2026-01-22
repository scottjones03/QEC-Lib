# src/qectostim/decoders/beliefmatching_decoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import stim

from qectostim.decoders.base import Decoder


class BeliefMatchingIncompatibleError(Exception):
    """Raised when BeliefMatching cannot handle the DEM structure.
    
    This typically occurs with:
    - QLDPC codes with complex hyperedge structures
    - 4D+ toric codes with high-weight error mechanisms  
    - Codes where BP fails to converge
    """
    pass


@dataclass
class BeliefMatchingDecoder(Decoder):
    """Decoder using the `beliefmatching` package on Stim DEMs.
    
    Uses belief propagation combined with minimum-weight perfect matching.
    The beliefmatching package directly accepts Stim DetectorErrorModels
    and handles hyperedge decomposition internally.
    
    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decode.
    max_bp_iters : int, default=20
        Maximum number of belief propagation iterations.
    bp_method : str, default='product_sum'
        BP algorithm variant. Options: 'product_sum', 'minimum_sum'.
    """

    dem: stim.DetectorErrorModel
    max_bp_iters: int = 20
    bp_method: str = "product_sum"

    def __post_init__(self) -> None:
        try:
            from beliefmatching import BeliefMatching  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "BeliefMatchingDecoder requires the `beliefmatching` package. "
                "Install it via `pip install beliefmatching`."
            ) from exc

        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables

        # BeliefMatching accepts the DEM directly and handles hyperedges internally.
        # For some complex codes (QLDPC, high-dimensional toric), BP may fail.
        # We try multiple strategies to maximize compatibility.
        
        init_errors = []
        self._decoder = None
        
        # Strategy 1: Try with original DEM
        try:
            self._decoder = BeliefMatching(
                self.dem,
                max_bp_iters=self.max_bp_iters,
                bp_method=self.bp_method,
            )
            return  # Success
        except Exception as e:
            init_errors.append(f"Original DEM: {e}")
        
        # Strategy 2: Try with rounded probabilities (helps numerical stability)
        try:
            rounded_dem = self.dem.rounded(3)
            self._decoder = BeliefMatching(
                rounded_dem,
                max_bp_iters=self.max_bp_iters,
                bp_method=self.bp_method,
            )
            return  # Success
        except Exception as e:
            init_errors.append(f"Rounded DEM: {e}")
        
        # Strategy 3: Try with minimum_sum BP method (sometimes more stable)
        if self.bp_method != 'minimum_sum':
            try:
                self._decoder = BeliefMatching(
                    self.dem,
                    max_bp_iters=self.max_bp_iters,
                    bp_method='minimum_sum',
                )
                return  # Success
            except Exception as e:
                init_errors.append(f"minimum_sum method: {e}")
        
        # All strategies failed - raise informative error
        raise BeliefMatchingIncompatibleError(
            f"BeliefMatchingDecoder cannot handle this DEM. "
            f"This code likely has hyperedges or error mechanisms incompatible with BP. "
            f"Consider using BPOSD or Tesseract decoder instead. "
            f"Errors: {'; '.join(init_errors[:2])}"
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