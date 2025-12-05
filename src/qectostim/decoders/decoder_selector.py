# src/qectostim/decoders/decoder_selector.py
from __future__ import annotations

from typing import Optional

import stim

from qectostim.codes.composite.concatenated import ConcatenatedCode
from qectostim.decoders.base import Decoder
from qectostim.decoders.concatenated_decoder import ConcatenatedDecoder
from qectostim.decoders.pymatching_decoder import PyMatchingDecoder
from qectostim.decoders.fusion_blossom_decoder import FusionBlossomDecoder
from qectostim.decoders.union_find_decoder import UnionFindDecoder
from qectostim.decoders.tesseract_decoder import TesseractDecoder
from qectostim.decoders.bp_osd import BPOSDDecoder
from qectostim.decoders.belief_matching import BeliefMatchingDecoder

def select_decoder(
    dem: stim.DetectorErrorModel,
    preferred: Optional[str] = None,
    code=None,
    *,
    max_bp_iters: int = 30,
    osd_order: int = 60,
    tesseract_det_beam: int = 5,
) -> Decoder:
    """Factory for constructing a decoder from a Stim DEM.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        Detector error model to be decoded.
    preferred : Optional[str]
        Decoder name hint (case-insensitive). Examples:
          - "pymatching", "matching", "mwpm"
          - "fusion", "fusionblossom"
          - "uf", "union-find"
          - "tesseract"
          - "bposd", "bp-osd"
          - "beliefmatching", "belief"
    code : Optional[Code]
        Code object. ConcatenatedCode triggers the special concatenated decoder.
    max_bp_iters : int
        Maximum BP iterations for BP-based decoders.
    osd_order : int
        OSD order for BP+OSD decoder.
    tesseract_det_beam : int
        Detector beam width for Tesseract decoder.
    """

    # Concatenated codes -> dedicated wrapper
    if isinstance(code, ConcatenatedCode):
        return ConcatenatedDecoder(
            code=code,
            dem=dem,
            preferred=preferred,
            max_bp_iters=max_bp_iters,
            osd_order=osd_order,
            tesseract_bond_dim=tesseract_det_beam,
        )

    name = (preferred or "pymatching").lower()

    # MWPM family
    if name in {"pymatching", "matching", "mwpm", "mwpm2"}:
        return PyMatchingDecoder(dem)

    if name in {"fb", "fusion", "fusionblossom", "fusion-blossom"}:
        return FusionBlossomDecoder(dem)

    # Union-Find
    if name in {"uf", "unionfind", "union-find"}:
        return UnionFindDecoder(dem)

    # Tensor network (tesseract)
    if name in {"tesseract", "tn"}:
        return TesseractDecoder(dem, det_beam=tesseract_det_beam)

    # BP+OSD (stimbposd)
    if name in {"bposd", "bp-osd", "bp_osd"}:
        return BPOSDDecoder(dem, max_bp_iters=max_bp_iters, osd_order=osd_order)

    # Belief-matching
    if name in {"beliefmatching", "belief-matching", "belief"}:
        return BeliefMatchingDecoder(dem, max_bp_iters=max_bp_iters)

    # Fallback: PyMatching.
    return PyMatchingDecoder(dem)