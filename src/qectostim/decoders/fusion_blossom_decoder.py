from qectostim.decoders.base import Decoder
import stim 
import numpy as np
from typing import Any  

class FusionBlossomDecoder(Decoder):
    """
    Uses the fusion-blossom library to perform MWPM decoding on a Stim DEM.
    """
    def __init__(self, dem: stim.DetectorErrorModel):
        try:
            import fusion_blossom as fb
            from fusion_blossom import stim_util
        except ImportError as exc:
            raise ImportError("FusionBlossomDecoder requires `fusion-blossom` package. "
                              "Install it via `pip install fusion-blossom`.") from exc
        self.dem = dem
        # Build Fusion Blossom decoding graph from Stim DEM
        self._graph, self._decoder = stim_util.stim_dem_to_fusion_blossom(dem)
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim != 2 or dets.shape[1] != self.num_detectors:
            raise ValueError(f"UnionFindDecoder expected input shape (shots, {self.num_detectors}).")
        shots = dets.shape[0]
        corrections = np.zeros((shots, self.num_observables), dtype=np.uint8)
        for i in range(shots):
            syndrome = dets[i].astype(bool)
            # Decode returns a correction bitstring over `num_observables` logicals
            corr = self._decoder.decode(syndrome)  # bool list of length num_observables
            corr = np.asarray(corr, dtype=np.uint8).reshape(-1)
            if corr.size != self.num_observables:
                raise RuntimeError("FusionBlossomDecoder: unexpected correction size.")
            corrections[i, :] = corr
        return corrections