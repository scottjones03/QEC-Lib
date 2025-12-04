# src/qectostim/decoders/union_find_decoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import stim

from qectostim.decoders.base import Decoder


UFBackend = Callable[[np.ndarray], np.ndarray]


@dataclass
class UnionFindDecoder(Decoder):
    """Union-Find style decoder.

    This class is a thin wrapper around any union-find implementation that
    provides a Stim-DEM-aware constructor.

    You can either:
      - let it try to import a known UF library (e.g. `panqec` or `qecc`), or
      - pass a custom `backend_decode` callable which takes a syndrome batch
        and returns logical corrections.
    """

    dem: stim.DetectorErrorModel
    backend_decode: Optional[UFBackend] = None

    def __post_init__(self) -> None:
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables

        if self.backend_decode is not None:
            # User supplied their own UF decoder function.
            return

        # Try to auto-detect a UF library.
        # You can customise this block depending on what you actually install.
        try:
            import panqec  # type: ignore
            from panqec.decoders import UnionFindDecoder as PanQECUFDecoder  # type: ignore
        except ImportError:
            panqec = None
            PanQECUFDecoder = None  # type: ignore

        if panqec is not None and PanQECUFDecoder is not None:
            # Example: assume PanQEC has a helper to build from DEM.
            uf_decoder = PanQECUFDecoder.from_stim_dem(self.dem)

            def backend(dets: np.ndarray) -> np.ndarray:
                return np.asarray(uf_decoder.decode_batch(dets), dtype=np.uint8)

            self.backend_decode = backend
            return

        raise ImportError(
            "UnionFindDecoder: no union-find backend available. Install a "
            "UF decoder (e.g. PanQEC) and/or pass `backend_decode` explicitly."
        )

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