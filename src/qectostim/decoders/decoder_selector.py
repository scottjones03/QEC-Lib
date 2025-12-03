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

def select_decoder(
    dem: stim.DetectorErrorModel,
    preferred: Optional[str] = None,
    code=None,
):
    # Concatenated codes -> dedicated wrapper (currently PyMatching inside).
    if isinstance(code, ConcatenatedCode):
        return ConcatenatedDecoder(code=code, dem=dem)

    name = (preferred or "pymatching").lower()

    if name in {"pymatching", "matching", "mwpm", "mwpm2"}:
        return PyMatchingDecoder(dem)

    if name in {"uf", "unionfind", "union-find"}:
        return UnionFindDecoder(dem)

    if name in {"fb", "fusion", "fusionblossom", "fusion-blossom"}:
        return FusionBlossomDecoder(dem)

    # Fallback: PyMatching.
    return PyMatchingDecoder(dem)