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
# add:
from qectostim.decoders.tesseract_decoder import TesseractDecoder
from qectostim.decoders.bp_osd import BPOSDDecoder
from qectostim.decoders.belief_matching import BeliefMatchingDecoder

def select_decoder(
    dem: stim.DetectorErrorModel,
    preferred: Optional[str] = None,
    code=None,
    *,
    physical_error_rate: float = 1e-3,
    max_bp_iters: int = 50,
    osd_order: int = 2,
    tesseract_bond_dim: int = 16,
    belief_damping: float = 0.0,
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
    physical_error_rate : float
        Prior error rate for BP/OSD/belief-matching decoders.
    """

    # Concatenated codes -> dedicated wrapper (see below).
    if isinstance(code, ConcatenatedCode):
        return ConcatenatedDecoder(
            code=code,
            dem=dem,
            preferred=preferred,
            physical_error_rate=physical_error_rate,
            max_bp_iters=max_bp_iters,
            osd_order=osd_order,
            tesseract_bond_dim=tesseract_bond_dim,
            belief_damping=belief_damping,
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
        return TesseractDecoder(
            dem,
            max_bond_dim=tesseract_bond_dim,
        )

    # BP+OSD (stimbposd)
    if name in {"bposd", "bp-osd", "bp_osd"}:
        return BPOSDDecoder(
            dem,
            p=physical_error_rate,
            max_bp_iters=max_bp_iters,
            osd_order=osd_order,
        )

    # Belief-matching
    if name in {"beliefmatching", "belief-matching", "belief"}:
        return BeliefMatchingDecoder(
            dem,
            p=physical_error_rate,
            max_iters=max_bp_iters,
            damping=belief_damping,
        )

    # Fallback: PyMatching.
    return PyMatchingDecoder(dem)