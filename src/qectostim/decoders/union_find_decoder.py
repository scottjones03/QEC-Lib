# src/qectostim/decoders/union_find_decoder.py
"""Union-Find decoder wrapper.

The Union-Find decoder is a fast decoder that runs in nearly-linear time.
However, there is currently no widely available UF decoder that works directly
with Stim's DetectorErrorModel.

Available backends:
- PanQEC: Has UnionFindDecoder but only works with PanQEC's Toric2DCode
- fusion_blossom: Uses MWPM, not Union-Find
- PyMatching v2: Uses MWPM, not Union-Find

This module provides a placeholder that can be configured with a custom
backend, or falls back to PyMatching for compatibility (with a warning).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
import warnings

import numpy as np
import stim

from qectostim.decoders.base import Decoder


UFBackend = Callable[[np.ndarray], np.ndarray]


@dataclass
class UnionFindDecoder(Decoder):
    """Union-Find style decoder wrapper.

    Note: There is currently no widely available Union-Find implementation
    that works directly with Stim's DetectorErrorModel. Available options:
    
    - PanQEC's UnionFindDecoder: Only works with their Toric2DCode class
    - fusion_blossom: Uses MWPM (Blossom algorithm), not Union-Find
    - PyMatching: Uses MWPM, not Union-Find
    
    If no UF backend is found, this falls back to PyMatching for compatibility,
    but emits a warning since it's not true Union-Find.
    
    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decode.
    backend_decode : callable, optional
        Custom decode function: (syndromes: np.ndarray) -> corrections: np.ndarray
        If not provided, falls back to PyMatching.
    use_pymatching_fallback : bool
        If True and no UF backend found, use PyMatching as fallback.
        Default True for compatibility.
    """

    dem: stim.DetectorErrorModel
    backend_decode: Optional[UFBackend] = None
    use_pymatching_fallback: bool = True

    def __post_init__(self) -> None:
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables
        self._is_fallback = False

        if self.backend_decode is not None:
            # User supplied their own UF decoder function.
            return

        # Try to find a true UF backend
        # (Currently none available that work with stim DEM)
        
        # Fallback to PyMatching if enabled
        if self.use_pymatching_fallback:
            try:
                import pymatching
                matcher = pymatching.Matching.from_detector_error_model(self.dem)
                
                def pymatching_backend(dets: np.ndarray) -> np.ndarray:
                    return np.asarray(matcher.decode_batch(dets), dtype=np.uint8)
                
                self.backend_decode = pymatching_backend
                self._is_fallback = True
                warnings.warn(
                    "UnionFindDecoder: No true Union-Find backend available. "
                    "Using PyMatching (MWPM) as fallback. For true UF decoding, "
                    "provide a custom backend_decode function.",
                    UserWarning,
                    stacklevel=2
                )
                return
            except ImportError:
                pass

        raise ImportError(
            "UnionFindDecoder: No union-find backend available and PyMatching "
            "fallback not available. Options:\n"
            "  1. Install pymatching for MWPM fallback\n"
            "  2. Provide a custom backend_decode function\n"
            "  3. Use a different decoder (PyMatchingDecoder, FusionBlossomDecoder)"
        )

    @property
    def is_fallback(self) -> bool:
        """True if using PyMatching fallback instead of true Union-Find."""
        return self._is_fallback

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        if self.backend_decode is None:
            raise RuntimeError("UnionFindDecoder has no backend_decode set.")

        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        if dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"UnionFindDecoder: expected dets.shape[1]={self.num_detectors}, "
                f"got {dets.shape[1]}"
            )

        corrections = self.backend_decode(dets)
        corrections = np.asarray(corrections, dtype=np.uint8)
        if corrections.ndim == 1:
            corrections = corrections.reshape(-1, self.num_observables)
        return corrections