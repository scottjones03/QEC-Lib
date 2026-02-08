# src/qectostim/decoders/concat_mle_decoder.py
"""
Near-optimal "perfect" decoder for concatenated CSS codes.

This module provides a reference decoder that achieves the best
achievable logical error rate on concatenated code DEMs.  It cascades
through decoders of decreasing optimality until one is available:

1. **Tesseract** (tensor-network contraction) — provably near-MLE
   for moderate-size DEMs.  Handles hyperedges natively.
2. **BP+OSD-CS with maximum order** — combination-sweep OSD with a
   very large search depth (``osd_order=200``) and many BP iterations.
   This is the most practical "near-perfect" decoder when tesseract
   is unavailable.
3. **Exact MLE** (syndrome lookup) — exhaustive enumeration up to
   ``max_weight`` faults.  Only feasible for very small DEMs
   (num_detectors ≤ 30).

The decoder automatically selects the best available backend at
construction time.

Usage
-----
>>> from qectostim.decoders.concat_mle_decoder import ConcatMLEDecoder
>>> decoder = ConcatMLEDecoder(dem)            # auto-select backend
>>> decoder = ConcatMLEDecoder(dem, backend="bposd")  # force BP+OSD
>>> corrections = decoder.decode_batch(det_samples)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import stim

from qectostim.decoders.base import Decoder


# ──────────────────────────────────────────────────────────────────────
# Backend implementation: exact MLE via syndrome lookup table
# ──────────────────────────────────────────────────────────────────────

class _ExactMLEBackend:
    """Exhaustive weight-enumeration MLE for small DEMs.

    Builds a syndrome → (max_probability, observable_correction) lookup
    table by enumerating all fault combinations up to ``max_weight``.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        max_weight: int = 4,
    ) -> None:
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

        # Parse error mechanisms from DEM
        self._errors: List[Tuple[float, np.ndarray, np.ndarray]] = []
        for instr in dem.flattened():
            if instr.type != "error":
                continue
            prob = instr.args_copy()[0]
            det_mask = np.zeros(self.num_detectors, dtype=np.uint8)
            obs_mask = np.zeros(self.num_observables, dtype=np.uint8)
            for t in instr.targets_copy():
                if t.is_relative_detector_id():
                    det_mask[t.val] = 1
                elif t.is_logical_observable_id():
                    obs_mask[t.val] ^= 1
            self._errors.append((prob, det_mask, obs_mask))

        self._lookup: Dict[bytes, Tuple[float, np.ndarray]] = {}
        self._build_lookup(max_weight)

    # ---- lookup construction ----

    def _insert(self, syndrome: np.ndarray, prob: float, obs: np.ndarray) -> None:
        key = syndrome.tobytes()
        if key not in self._lookup or prob > self._lookup[key][0]:
            self._lookup[key] = (prob, obs.copy())

    def _build_lookup(self, max_weight: int) -> None:
        n = len(self._errors)
        # No-error
        self._insert(
            np.zeros(self.num_detectors, dtype=np.uint8),
            1.0,
            np.zeros(self.num_observables, dtype=np.uint8),
        )

        # Weight 1
        for i, (pi, di, oi) in enumerate(self._errors):
            self._insert(di, pi, oi)

        # Weight 2
        if max_weight >= 2 and n <= 500:
            for i in range(n):
                pi, di, oi = self._errors[i]
                for j in range(i + 1, n):
                    pj, dj, oj = self._errors[j]
                    self._insert(di ^ dj, pi * pj, oi ^ oj)

        # Weight 3
        if max_weight >= 3 and n <= 100:
            for i in range(n):
                pi, di, oi = self._errors[i]
                for j in range(i + 1, n):
                    pj, dj, oj = self._errors[j]
                    d_ij, o_ij = di ^ dj, oi ^ oj
                    p_ij = pi * pj
                    for k in range(j + 1, n):
                        pk, dk, ok = self._errors[k]
                        self._insert(d_ij ^ dk, p_ij * pk, o_ij ^ ok)

        # Weight 4
        if max_weight >= 4 and n <= 50:
            for i in range(n):
                pi, di, oi = self._errors[i]
                for j in range(i + 1, n):
                    pj, dj, oj = self._errors[j]
                    d_ij, o_ij, p_ij = di ^ dj, oi ^ oj, pi * pj
                    for k in range(j + 1, n):
                        pk, dk, ok = self._errors[k]
                        d_ijk, o_ijk, p_ijk = d_ij ^ dk, o_ij ^ ok, p_ij * pk
                        for l_ in range(k + 1, n):
                            pl, dl, ol = self._errors[l_]
                            self._insert(d_ijk ^ dl, p_ijk * pl, o_ijk ^ ol)

    # ---- decode ----

    def decode(self, det: np.ndarray) -> np.ndarray:
        key = np.asarray(det, dtype=np.uint8).tobytes()
        if key in self._lookup:
            return self._lookup[key][1].copy()
        return np.zeros(self.num_observables, dtype=np.uint8)

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        out = np.zeros((dets.shape[0], self.num_observables), dtype=np.uint8)
        for i in range(dets.shape[0]):
            out[i] = self.decode(dets[i])
        return out


# ──────────────────────────────────────────────────────────────────────
# Backend implementation: BP+OSD-CS reference
# ──────────────────────────────────────────────────────────────────────

class _BPOSDReferenceBackend:
    """BP+OSD-CS with maximum search depth (reference quality)."""

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        osd_order: int = 200,
        max_bp_iters: int = 100,
        bp_method: str = "product_sum",
        osd_method: str = "osd_cs",
    ) -> None:
        from stimbposd.dem_to_matrices import (
            detector_error_model_to_check_matrices,
        )
        from ldpc import BpOsdDecoder as LdpcBpOsd

        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

        matrices = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True
        )
        H = matrices.check_matrix

        self._raw = LdpcBpOsd(
            H,
            max_iter=max_bp_iters,
            bp_method=bp_method,
            error_channel=list(matrices.priors),
            osd_order=osd_order,
            osd_method=osd_method,
            input_vector_type="syndrome",
        )

        obs_matrix = matrices.observables_matrix
        self._obs_dense = (
            np.asarray(obs_matrix.todense(), dtype=np.uint8)
            if hasattr(obs_matrix, "todense")
            else np.asarray(obs_matrix, dtype=np.uint8)
        )

    def decode(self, det: np.ndarray) -> np.ndarray:
        err = np.asarray(self._raw.decode(det), dtype=np.uint8)
        return (self._obs_dense @ err) % 2

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        out = np.zeros((dets.shape[0], self.num_observables), dtype=np.uint8)
        for i in range(dets.shape[0]):
            out[i] = self.decode(dets[i])
        return out


# ──────────────────────────────────────────────────────────────────────
# Backend implementation: Tesseract tensor-network decoder
# ──────────────────────────────────────────────────────────────────────

class _TesseractBackend:
    """Wrapper around the tesseract tensor-network decoder."""

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        det_beam: int = 8,
        merge_errors: bool = True,
    ) -> None:
        import tesseract_decoder as tdec  # type: ignore

        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

        config = tdec.tesseract.TesseractConfig()
        config.det_beam = det_beam
        config.merge_errors = 1 if merge_errors else 0
        self._decoder = config.compile_decoder_for_dem(dem)

    def decode(self, det: np.ndarray) -> np.ndarray:
        corr = self._decoder.decode(np.asarray(det, dtype=bool))
        return np.asarray(corr, dtype=np.uint8)

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        out = np.zeros((dets.shape[0], self.num_observables), dtype=np.uint8)
        for i in range(dets.shape[0]):
            out[i] = self.decode(dets[i])
        return out


# ──────────────────────────────────────────────────────────────────────
# Main decoder class
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ConcatMLEDecoder(Decoder):
    """Near-optimal reference decoder for concatenated CSS codes.

    Automatically selects the best available backend:
      ``"tesseract"`` → ``"bposd"`` → ``"exact_mle"``

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        Detector error model (may contain undecomposed hyperedges).
    backend : str or None
        Force a specific backend: ``"tesseract"``, ``"bposd"``, or
        ``"exact_mle"``.  If ``None`` (default), auto-selects.
    osd_order : int
        OSD order for ``"bposd"`` backend.  200 gives near-MLE quality
        at the cost of speed.
    max_bp_iters : int
        Max BP iterations for ``"bposd"`` backend.
    max_weight : int
        Max error weight for ``"exact_mle"`` backend.
    det_beam : int
        Beam width for ``"tesseract"`` backend.
    """

    dem: stim.DetectorErrorModel
    backend: Optional[str] = None
    osd_order: int = 200
    max_bp_iters: int = 100
    max_weight: int = 4
    det_beam: int = 8

    _backend_impl: Any = field(default=None, init=False, repr=False)
    _backend_name: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables

        if self.backend is not None:
            self._init_backend(self.backend)
        else:
            self._auto_select()

    def _auto_select(self) -> None:
        """Try backends in order of quality."""
        # 1. Tesseract (best quality, may not be installed)
        try:
            self._init_backend("tesseract")
            return
        except (ImportError, Exception):
            pass

        # 2. BP+OSD-CS (robust, always available with ldpc)
        try:
            self._init_backend("bposd")
            return
        except (ImportError, Exception):
            pass

        # 3. Exact MLE (only for small DEMs)
        if self.num_detectors <= 30:
            self._init_backend("exact_mle")
            return

        raise RuntimeError(
            "ConcatMLEDecoder: no backend available. Install `ldpc` + `stimbposd` "
            "(for BP+OSD) or `tesseract-decoder` (for tensor-network decoding)."
        )

    def _init_backend(self, name: str) -> None:
        if name == "tesseract":
            self._backend_impl = _TesseractBackend(
                self.dem, det_beam=self.det_beam
            )
        elif name == "bposd":
            self._backend_impl = _BPOSDReferenceBackend(
                self.dem,
                osd_order=self.osd_order,
                max_bp_iters=self.max_bp_iters,
            )
        elif name == "exact_mle":
            if self.num_detectors > 30:
                raise ValueError(
                    f"exact_mle backend limited to ≤30 detectors, "
                    f"got {self.num_detectors}"
                )
            self._backend_impl = _ExactMLEBackend(
                self.dem, max_weight=self.max_weight
            )
        else:
            raise ValueError(f"Unknown backend: {name!r}")
        self._backend_name = name

    @property
    def active_backend(self) -> str:
        """Name of the backend in use."""
        return self._backend_name

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """Decode a batch of detector samples.

        Parameters
        ----------
        dets : np.ndarray, shape (n_shots, num_detectors)
            Binary detector outcomes.

        Returns
        -------
        np.ndarray, shape (n_shots, num_observables)
            Predicted observable flips.
        """
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        if dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"Expected {self.num_detectors} detectors, got {dets.shape[1]}"
            )
        return self._backend_impl.decode_batch(dets)
