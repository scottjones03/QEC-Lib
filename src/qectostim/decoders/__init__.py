# src/qectostim/decoders/__init__.py
"""
Decoder implementations for QEC codes.

Available decoders:
- PyMatchingDecoder: MWPM decoder using PyMatching
- FusionBlossomDecoder: Fast MWPM decoder using Fusion Blossom
- BeliefMatchingDecoder: BP + matching decoder
- BPOSDDecoder: Belief propagation + OSD decoder
- TesseractDecoder: General-purpose decoder
- UnionFindDecoder: Union-find based decoder
- ChromobiusDecoder: Hyperedge-aware decoder for non-CSS codes
- MLEDecoder: Exact maximum likelihood decoder for small codes
- HypergraphDecoder: Matching + boundary L0 correction
"""

from qectostim.decoders.base import Decoder

# Core decoders - always available
__all__ = ["Decoder"]

# Optional decoders - imported if available
try:
    from qectostim.decoders.pymatching_decoder import PyMatchingDecoder
    __all__.append("PyMatchingDecoder")
except ImportError:
    pass

try:
    from qectostim.decoders.fusion_blossom_decoder import FusionBlossomDecoder
    __all__.append("FusionBlossomDecoder")
except ImportError:
    pass

try:
    from qectostim.decoders.belief_matching import BeliefMatchingDecoder
    __all__.append("BeliefMatchingDecoder")
except ImportError:
    pass

try:
    from qectostim.decoders.bp_osd import BPOSDDecoder
    __all__.append("BPOSDDecoder")
except ImportError:
    pass

try:
    from qectostim.decoders.tesseract_decoder import TesseractDecoder
    __all__.append("TesseractDecoder")
except ImportError:
    pass

try:
    from qectostim.decoders.union_find_decoder import UnionFindDecoder
    __all__.append("UnionFindDecoder")
except ImportError:
    pass

# New decoders for non-CSS codes
try:
    from qectostim.decoders.chromobius_decoder import ChromobiusDecoder
    __all__.append("ChromobiusDecoder")
except ImportError:
    pass

try:
    from qectostim.decoders.mle_decoder import MLEDecoder, HypergraphDecoder
    __all__.extend(["MLEDecoder", "HypergraphDecoder"])
except ImportError:
    pass

# Concatenated code decoders
try:
    from qectostim.decoders.concatenated_decoder import (
        ConcatenatedDecoder,
        FlatConcatenatedDecoder,
        ConcatenatedDecoderIncompatibleError,
    )
    __all__.extend(["ConcatenatedDecoder", "FlatConcatenatedDecoder", "ConcatenatedDecoderIncompatibleError"])
except ImportError:
    pass
