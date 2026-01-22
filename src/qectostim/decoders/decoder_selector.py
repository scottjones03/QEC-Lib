# src/qectostim/decoders/decoder_selector.py
from __future__ import annotations

from typing import Optional

import stim

from qectostim.codes.composite.concatenated import ConcatenatedCode
from qectostim.codes.composite.homological_product import HomologicalProductCode
from qectostim.decoders.base import Decoder
from qectostim.decoders._ignore.enhanced_concatenated_decoder import EnhancedConcatenatedDecoder as ConcatenatedDecoder
from qectostim.decoders.pymatching_decoder import PyMatchingDecoder
from qectostim.decoders.fusion_blossom_decoder import FusionBlossomDecoder
from qectostim.decoders.union_find_decoder import UnionFindDecoder
from qectostim.decoders.tesseract_decoder import TesseractDecoder
from qectostim.decoders.bp_osd import BPOSDDecoder
from qectostim.decoders.belief_matching import BeliefMatchingDecoder
from qectostim.decoders.mle_decoder import MLEDecoder, HypergraphDecoder
from qectostim.decoders.chromobius_decoder import ChromobiusDecoder


class DecoderIncompatibleError(Exception):
    """Raised when a decoder is not compatible with the given code/DEM."""
    pass

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
          - "mle", "maximum-likelihood" (exact, for small codes only)
          - "hypergraph", "hyper" (PyMatching + boundary L0 correction)
          - "chromobius", "chromo" (for hyperedge DEMs)
    code : Optional[Code]
        Code object. ConcatenatedCode triggers the special concatenated decoder.
    max_bp_iters : int
        Maximum BP iterations for BP-based decoders.
    osd_order : int
        OSD order for BP+OSD decoder.
    tesseract_det_beam : int
        Detector beam width for Tesseract decoder.
    """

    # =========================================================================
    # AUTO-DETECTION: Select decoder based on code type (if no preference given)
    # =========================================================================
    
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
    
    # Homological product codes -> BPOSD (produces hyperedge DEMs)
    # These codes have errors affecting >2 detectors, which breaks PyMatching
    if isinstance(code, HomologicalProductCode):
        return BPOSDDecoder(dem, max_bp_iters=max_bp_iters, osd_order=osd_order)
    
    # If no explicit preference, try intelligent selection based on code properties
    if preferred is None and code is not None:
        # Color codes -> Chromobius (if compatible)
        code_meta = getattr(code, 'metadata', {}) if hasattr(code, 'metadata') else {}
        if code_meta.get('is_chromobius_compatible', False):
            try:
                return ChromobiusDecoder(dem)
            except Exception:
                pass  # Fall through to default
        
        # QLDPC codes -> BP-OSD (handles hypergraph structure well)
        # QLDPC codes often have hyperedges that break matching-based decoders
        code_type = code_meta.get('type', '')
        is_qldpc = code_meta.get('is_qldpc', False)
        if is_qldpc or code_type in ('qldpc', 'hgp', 'bivariate_bicycle', 'lifted_product', 'balanced_product'):
            return BPOSDDecoder(dem, max_bp_iters=max_bp_iters, osd_order=osd_order)
        
        # 5-chain codes with metachecks -> BP-OSD (good for LDPC-like structure)
        # SingleShot decoder would be ideal here when implemented
        if hasattr(code, 'chain_length') and code.chain_length >= 5:
            has_meta = (
                (hasattr(code, 'meta_x') and code.meta_x is not None and code.meta_x.size > 0) or
                (hasattr(code, 'meta_z') and code.meta_z is not None and code.meta_z.size > 0)
            )
            if has_meta:
                # For now, use BP-OSD for 5-chain codes with metachecks
                # TODO: Use SingleShotDecoder when available
                return BPOSDDecoder(dem, max_bp_iters=max_bp_iters, osd_order=osd_order)
        
        # Homological product codes (4D+) -> BP-OSD handles hypergraph structure well
        if hasattr(code, 'chain_length') and code.chain_length >= 4:
            return BPOSDDecoder(dem, max_bp_iters=max_bp_iters, osd_order=osd_order)
        
        # Small codes (≤30 detectors) -> MLE is exact and fast
        if dem.num_detectors <= 30:
            try:
                return MLEDecoder(dem)
            except Exception:
                pass  # Fall through to default

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

    # MLE decoder (exact, for small codes)
    if name in {"mle", "maximum-likelihood", "lookup"}:
        return MLEDecoder(dem)

    # Hypergraph decoder (PyMatching + boundary L0 correction)
    if name in {"hypergraph", "hyper"}:
        return HypergraphDecoder(dem)

    # Chromobius decoder (for hyperedge DEMs)
    if name in {"chromobius", "chromo"}:
        return ChromobiusDecoder(dem)

    # Hierarchical decoder (for concatenated codes only)
    if name in {"hierarchical", "hier", "concatenated-hierarchical"}:
        if not isinstance(code, ConcatenatedCode):
            raise DecoderIncompatibleError(
                "HierarchicalConcatenatedDecoder requires a ConcatenatedCode instance. "
                f"Got {type(code).__name__ if code else 'None'}."
            )
        from qectostim.decoders.hard_hierarchical_decoder import HierarchicalConcatenatedDecoder
        # Infer rounds from DEM structure if possible
        rounds = getattr(code, '_rounds', 3)
        return HierarchicalConcatenatedDecoder(code=code, dem=dem, rounds=rounds, basis="Z")

    # Turbo decoder for concatenated codes with soft information passing
    if name in {"turbo", "turbo-concatenated", "soft-hierarchical", "soft-turbo"}:
        if not isinstance(code, ConcatenatedCode):
            raise DecoderIncompatibleError(
                "ConcatenatedTurboDecoder requires a ConcatenatedCode instance. "
                f"Got {type(code).__name__ if code else 'None'}."
            )
        from qectostim.decoders.concatenated_turbo_decoder import (
            ConcatenatedTurboDecoder,
            TurboDecoderIncompatibleError,
        )
        rounds = getattr(code, '_rounds', 3)
        try:
            return ConcatenatedTurboDecoder(code=code, dem=dem, rounds=rounds, basis="Z")
        except TurboDecoderIncompatibleError as e:
            raise DecoderIncompatibleError(str(e))

    # SingleShot decoder (for codes with metachecks only)
    if name in {"singleshot", "single-shot", "single_shot", "ss"}:
        # Check for metachecks
        def _has_valid_meta(c, attr):
            if not hasattr(c, attr):
                return False
            m = getattr(c, attr)
            if m is None:
                return False
            if not hasattr(m, 'size') or not hasattr(m, 'shape'):
                return False
            return m.size > 0 and m.shape[0] > 0
        
        has_meta_x = _has_valid_meta(code, 'meta_x')
        has_meta_z = _has_valid_meta(code, 'meta_z')
        
        if not (has_meta_x or has_meta_z):
            raise DecoderIncompatibleError(
                "SingleShotDecoder requires a code with metachecks (meta_x or meta_z). "
                "Use codes from 4D/5D chain complexes (e.g., ToricCode4D, HomologicalProductCode)."
            )
        from qectostim.decoders.single_shot_decoder import SingleShotDecoder
        rounds = getattr(code, '_rounds', 1)
        return SingleShotDecoder(code=code, dem=dem, rounds=rounds, basis="Z")

    # Fallback: Tesseract (tensor network).
    return TesseractDecoder(dem, det_beam=tesseract_det_beam)