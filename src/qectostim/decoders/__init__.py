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
- SoftMessagePassingDecoder: Hierarchical soft decoder for concatenated codes (recommended)
- TurboDecoderV2: Iterative turbo decoder with code-based inner decoding
- ExtrinsicTurboDecoder: True turbo decoder with extrinsic information exchange

Shared Utilities:
- block_extraction: Code structure analysis and detector slicing for hierarchical decoders
"""

from qectostim.decoders.base import Decoder

# Core decoders - always available
__all__ = ["Decoder"]

# Block extraction utilities for hierarchical decoders
try:
    from qectostim.decoders.block_extraction import (
        BlockExtractor,
        CodeStructure,
        DetectorSlices,
        extract_code_structure,
        get_logical_support,
        compute_detector_slices,
        collapse_block_syndrome,
        soft_xor_over_support,
        prob_to_llr,
        llr_to_prob,
    )
    __all__.extend([
        "BlockExtractor",
        "CodeStructure", 
        "DetectorSlices",
        "extract_code_structure",
        "get_logical_support",
        "compute_detector_slices",
        "collapse_block_syndrome",
        "soft_xor_over_support",
        "prob_to_llr",
        "llr_to_prob",
    ])
except ImportError:
    pass

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

# New hierarchical decoder for concatenated codes
try:
    from qectostim.decoders.hard_hierarchical_decoder import (
        HierarchicalConcatenatedDecoder,
        create_hierarchical_decoder,
    )
    __all__.extend(["HierarchicalConcatenatedDecoder", "create_hierarchical_decoder"])
except ImportError:
    pass

# Turbo decoder for concatenated codes with soft information passing
try:
    from qectostim.decoders.concatenated_turbo_decoder import (
        ConcatenatedTurboDecoder,
        TurboDecoderIncompatibleError,
        create_turbo_decoder,
    )
    __all__.extend(["ConcatenatedTurboDecoder", "TurboDecoderIncompatibleError", "create_turbo_decoder"])
except ImportError:
    pass

# Extrinsic turbo decoder with proper turbo iteration
try:
    from qectostim.decoders.extrinsic_turbo_decoder import ExtrinsicTurboDecoder
    __all__.append("ExtrinsicTurboDecoder")
except ImportError:
    pass

# Single-shot decoder with syndrome repair for 4D codes
try:
    from qectostim.decoders.single_shot_decoder import (
        SingleShotDecoder,
        SingleShotDecoderIncompatibleError,
    )
    __all__.extend(["SingleShotDecoder", "SingleShotDecoderIncompatibleError"])
except ImportError:
    pass

# Decoder selector and generic incompatibility error
try:
    from qectostim.decoders.decoder_selector import (
        select_decoder,
        DecoderIncompatibleError,
    )
    __all__.extend(["select_decoder", "DecoderIncompatibleError"])
except ImportError:
    pass

# Soft hierarchical decoder with per-shot renormalization (gold standard)
try:
    from qectostim.decoders.soft_hierarchical_decoder import SoftHierarchicalDecoder
    __all__.append("SoftHierarchicalDecoder")
except ImportError:
    pass

# TurboDecoderV2 - minimal iterative turbo decoder with code-based inner decoding
try:
    from qectostim.decoders.turbo_decoder_v2 import TurboDecoderV2
    __all__.append("TurboDecoderV2")
except ImportError:
    pass

# Soft Message-Passing Decoder - proper hierarchical without inner observables
# Uses BOTH hx AND hz matrices for full CSS soft information
try:
    from qectostim.decoders.soft_message_passing_decoder import SoftMessagePassingDecoder
    __all__.append("SoftMessagePassingDecoder")
except ImportError:
    pass

# New raw-measurement hierarchical scaffold decoder
try:
    from qectostim.decoders.new_hierarchical_decoder import NewHierarchicalDecoder
    __all__.append("NewHierarchicalDecoder")
except ImportError:
    pass

# Hierarchical decoders for HierarchicalConcatenatedMemoryExperiment
# Use explicit detector structure from experiment metadata
try:
    from qectostim.decoders.hard_hierarchical_concatenated_decoder import (
        HardHierarchicalConcatenatedDecoder,
    )
    __all__.append("HardHierarchicalConcatenatedDecoder")
except ImportError:
    pass

try:
    from qectostim.decoders.soft_hierarchical_concatenated_decoder import (
        SoftHierarchicalConcatenatedDecoder,
    )
    __all__.append("SoftHierarchicalConcatenatedDecoder")
except ImportError:
    pass

# HierarchicalMemoryV2 decoder - correctly handles inner→outer decoding flow
try:
    from qectostim.decoders.hierarchical_memory_v2_decoder import (
        HierarchicalMemoryV2Decoder,
    )
    __all__.append("HierarchicalMemoryV2Decoder")
except ImportError:
    pass

# HierarchicalV3Decoder - clean two-phase hierarchical decoder
try:
    from qectostim.decoders.hierarchical_v3_decoder import HierarchicalV3Decoder
    __all__.append("HierarchicalV3Decoder")
except ImportError:
    pass

# L0-only hierarchical decoder - simple and correct
try:
    from qectostim.decoders.hierarchical_l0_decoder import (
        HierarchicalL0Decoder,
        SimpleL0Decoder,
        strip_ancilla_observables,
    )
    __all__.extend(["HierarchicalL0Decoder", "SimpleL0Decoder", "strip_ancilla_observables"])
except ImportError:
    pass

# Adaptive Soft Decoder - optimal for concatenated surface codes
# Automatically selects V5 (inner-only) or V6 (inner+outer) based on noise level
try:
    from qectostim.decoders.adaptive_soft_decoder import (
        AdaptiveSoftDecoderV3,
        SoftHierarchicalDecoderV5NoOuter,
        SoftHierarchicalDecoderV6,
        MajorityVoteHierarchicalDecoder,
    )
    __all__.extend([
        "AdaptiveSoftDecoderV3",
        "SoftHierarchicalDecoderV5NoOuter",
        "SoftHierarchicalDecoderV6",
        "MajorityVoteHierarchicalDecoder",
    ])
except ImportError:
    pass

# TrueHierarchicalDecoder - 3-phase hierarchical with proper inner decoding
# Phase 1: Inner-only (baseline), Phase 2: Soft-weighted, Phase 3: Outer-gated
try:
    from qectostim.decoders.true_hierarchical_decoder import (
        TrueHierarchicalDecoder,
        TilePosterior,
        DecoderMode,
        create_decoder as create_true_hierarchical_decoder,
    )
    __all__.extend([
        "TrueHierarchicalDecoder",
        "TilePosterior",
        "DecoderMode",
        "create_true_hierarchical_decoder",
    ])
except ImportError:
    pass

# SyndromeHierarchicalDecoder - TRUE syndrome-based hierarchical decoder
# This decoder uses syndrome CHANGES to detect/correct errors, NOT just final data XOR
# It differs from RAW-XOR baseline when syndrome detection catches measurement errors
try:
    from qectostim.decoders.syndrome_hierarchical_decoder import (
        SyndromeHierarchicalDecoder,
        TilePosterior as SyndromeTilePosterior,
        DecoderMode as SyndromeDecoderMode,
        create_syndrome_decoder,
    )
    __all__.extend([
        "SyndromeHierarchicalDecoder",
        "SyndromeTilePosterior",
        "SyndromeDecoderMode",
        "create_syndrome_decoder",
    ])
except ImportError:
    pass

# ProperHierarchicalDecoder - TRUE hierarchical with MANDATORY syndrome-based inner correction
# This decoder implements the 3-layer architecture required by the spec:
# Layer 1: Inner syndrome-based decode (computes inferred_flip from MWPM)
# Layer 2: Tile → Outer interface (converts to outer checks)
# Layer 3: Outer soft/gated decode
# The inner decoder MUST differ from RAW-XOR when syndrome catches errors
try:
    from qectostim.decoders.proper_hierarchical_decoder import (
        ProperHierarchicalDecoder,
        InnerTilePosterior,
        OuterCheckPosterior,
        DecoderMode as ProperDecoderMode,
        create_proper_decoder,
    )
    __all__.extend([
        "ProperHierarchicalDecoder",
        "InnerTilePosterior",
        "OuterCheckPosterior",
        "ProperDecoderMode",
        "create_proper_decoder",
    ])
except ImportError:
    pass

# SpacetimeHierarchicalDecoder - TRUE space-time MWPM hierarchical decoder
# This decoder CORRECTLY implements syndrome-based decoding to BEAT RAW-XOR
# Key insight: decoding objective is to recover PREPARED logical (0), not final state
try:
    from qectostim.decoders.spacetime_hierarchical_decoder import (
        SpacetimeHierarchicalDecoder,
        InnerBlockDecoder,
        TilePosterior as SpacetimeTilePosterior,
        DecoderMode as SpacetimeDecoderMode,
        create_spacetime_decoder,
    )
    __all__.extend([
        "SpacetimeHierarchicalDecoder",
        "InnerBlockDecoder",
        "SpacetimeTilePosterior",
        "SpacetimeDecoderMode",
        "create_spacetime_decoder",
    ])
except ImportError:
    pass

# HierarchicalV5Variants - Slimmed-down decoder variants
# Temporal DEM or Majority Vote inner decoding
try:
    from qectostim.decoders.hierarchical_v5_variants import (
        BaseHierarchicalV5Decoder,
        TemporalHierarchicalV5Decoder,
        MajorityVoteHierarchicalV5Decoder,
        create_temporal_decoder,
        create_majority_vote_decoder,
    )
    __all__.extend([
        "BaseHierarchicalV5Decoder",
        "TemporalHierarchicalV5Decoder",
        "MajorityVoteHierarchicalV5Decoder",
        "create_temporal_decoder",
        "create_majority_vote_decoder",
    ])
except ImportError:
    pass

# HierarchicalV6Variants - Complete inner→outer pipeline decoder
# Fixes the critical flaw where outer stabilizer measurements were not decoded
# Implements 4-phase architecture:
#   Phase 1: Inner error inference per segment
#   Phase 2: Correct outer syndrome using ancilla inner corrections
#   Phase 3: Correct data logicals using ALL inner corrections
#   Phase 4: Outer temporal DEM decode
try:
    from qectostim.decoders.hierarchical_v6_variants import (
        BaseHierarchicalV6Decoder,
        TemporalHierarchicalV6Decoder,
        MajorityVoteHierarchicalV6Decoder,
        DecoderConfig as V6DecoderConfig,
        InnerDecodingMode,
        Segment,
        InnerCorrection,
        create_temporal_v6_decoder,
        create_majority_vote_v6_decoder,
    )
    __all__.extend([
        "BaseHierarchicalV6Decoder",
        "TemporalHierarchicalV6Decoder",
        "MajorityVoteHierarchicalV6Decoder",
        "V6DecoderConfig",
        "InnerDecodingMode",
        "Segment",
        "InnerCorrection",
        "create_temporal_v6_decoder",
        "create_majority_vote_v6_decoder",
    ])
except ImportError:
    pass

# UnifiedHierarchicalDecoder - BEST OF ALL WORLDS decoder
# Combines proven components from proper, spacetime, and true_v2 decoders:
# - Epoch boundary hardening (never crosses segment boundaries)
# - Clean spacetime MWPM construction
# - Proper consecutive-round XOR for detectors
# - Explicit boundary detector for final readout
# - Mandatory syndrome-based correction (differs from RAW-XOR)
# - Reliability-gated outer correction
try:
    from qectostim.decoders.unified_hierarchical_decoder import (
        UnifiedHierarchicalDecoder,
        EpochHardenedInnerDecoder,
        EpochSyndromeData,
        TilePosterior as UnifiedTilePosterior,
        DecoderMode as UnifiedDecoderMode,
        create_unified_hierarchical_decoder,
        BestOfAllWorldsDecoder,
        create_best_decoder,
    )
    __all__.extend([
        "UnifiedHierarchicalDecoder",
        "EpochHardenedInnerDecoder",
        "EpochSyndromeData",
        "UnifiedTilePosterior",
        "UnifiedDecoderMode",
        "create_unified_hierarchical_decoder",
        "BestOfAllWorldsDecoder",
        "create_best_decoder",
    ])
except ImportError:
    pass

# Decoder strategies for multi-level hierarchical decoding
try:
    from qectostim.decoders.strategies import (
        DecoderStrategy,
        StrategyOutput,
        SyndromeLookupStrategy,
        MWPMStrategy,
        BeliefPropagationStrategy,
        MLStrategy,
        MajorityVoteStrategy,
        TemporalMajorityStrategy,
    )
    __all__.extend([
        "DecoderStrategy",
        "StrategyOutput",
        "SyndromeLookupStrategy",
        "MWPMStrategy",
        "BeliefPropagationStrategy",
        "MLStrategy",
        "MajorityVoteStrategy",
        "TemporalMajorityStrategy",
    ])
except ImportError:
    pass

# RecursiveHierarchicalDecoder for multi-level concatenated codes
try:
    from qectostim.decoders.recursive_hierarchical_decoder import (
        RecursiveHierarchicalDecoder,
        LevelDecodeResult,
        ECSyndromeData,
        create_recursive_decoder,
    )
    __all__.extend([
        "RecursiveHierarchicalDecoder",
        "LevelDecodeResult",
        "ECSyndromeData",
        "create_recursive_decoder",
    ])
except ImportError:
    pass

# ConcatenatedECDecoder for multi-level codes with EC rounds and Pauli frame tracking
try:
    from qectostim.decoders.concatenated_ec_decoder import (
        ConcatenatedECDecoder,
        PauliFrame,
        create_ec_decoder,
    )
    __all__.extend([
        "ConcatenatedECDecoder",
        "PauliFrame",
        "create_ec_decoder",
    ])
except ImportError:
    pass

# FlagAwareDecoder - uses verification flags for improved FT decoding
# Supports post-selection, soft-weighting, and flag-gated strategies
try:
    from qectostim.decoders.flag_aware_decoder import (
        FlagAwareDecoder,
        FlagAwareConfig,
        FlagMode,
        DecodeResult,
        create_flag_aware_decoder,
        create_post_selecting_decoder,
    )
    __all__.extend([
        "FlagAwareDecoder",
        "FlagAwareConfig",
        "FlagMode",
        "DecodeResult",
        "create_flag_aware_decoder",
        "create_post_selecting_decoder",
    ])
except ImportError:
    pass

# EnhancedConcatenatedDecoder - Full FT decoder with:
# Phase 2: Temporal multi-round decoding (distinguishes meas vs data errors)
# Phase 3: Outer-level EC processing (full hierarchical correction)
# Phase 4: Soft information propagation (LLR-based confidence)
# Now supports BP-OSD for hyperedge DEMs
# NEW: EC syndrome extraction for proper hierarchical decoding
# NEW: Post-selection support for verified ancilla
# NEW: Optimal ML decoder for true p^d scaling with any noise model
try:
    from qectostim.decoders.enhanced_concatenated_decoder import (
        EnhancedConcatenatedDecoder,
        EnhancedDecoderConfig,
        DecoderPhase,
        FullDecodeResult,
        InnerBlockResult,
        OuterDecodeResult,
        TemporalDetectors,
        SyndromeRound,
        # New enums for decoder selection
        FlagMode,
        InnerDecoderType,
        PauliFrame,
        # Full DEM decoder (bypasses hierarchical)
        FullDEMDecoder,
        # Soft info utilities
        prob_to_llr,
        llr_to_prob,
        soft_xor,
        multi_soft_xor,
        # Factory functions
        create_enhanced_decoder,
        create_full_ft_decoder,
        create_bposd_decoder,
        create_full_dem_decoder,
    )
    __all__.extend([
        "EnhancedConcatenatedDecoder",
        "EnhancedDecoderConfig",
        "DecoderPhase",
        "FullDecodeResult",
        "InnerBlockResult",
        "OuterDecodeResult",
        "TemporalDetectors",
        "SyndromeRound",
        "FlagMode",
        "InnerDecoderType",
        "PauliFrame",
        "FullDEMDecoder",
        "prob_to_llr",
        "llr_to_prob",
        "soft_xor",
        "multi_soft_xor",
        "create_enhanced_decoder",
        "create_full_ft_decoder",
        "create_bposd_decoder",
        "create_full_dem_decoder",
    ])
except ImportError:
    pass

# OptimalConcatenatedDecoder - Generalized ML decoder for concatenated CSS codes
# - Uses exact ML at inner level (tractable for small codes)
# - Propagates soft information (LLRs) to outer level
# - Handles ANY i.i.d. noise model (bit-flip, depolarizing, biased, etc.)
# - No code-specific lookup tables - generalizes to any concatenated CSS code
try:
    from qectostim.decoders.optimal_concatenated_decoder import (
        OptimalConcatenatedDecoder,
        OptimalDecoderConfig,
        OptimalDecodeResult,
        NoiseModel,
        InnerCodeML,
        OuterCodeML,
        InnerMLResult,
        create_steane_steane_optimal,
        create_concatenated_decoder,
    )
    __all__.extend([
        "OptimalConcatenatedDecoder",
        "OptimalDecoderConfig",
        "OptimalDecodeResult",
        "NoiseModel",
        "InnerCodeML",
        "OuterCodeML",
        "InnerMLResult",
        "create_steane_steane_optimal",
        "create_concatenated_decoder",
    ])
except ImportError:
    pass

# CodeStructureHandler - General utility for extracting code structure
try:
    from qectostim.decoders.code_structure_handler import (
        CodeStructureHandler,
        CodeInfo,
        extract_code_structure,
        build_syndrome_lookup_from_matrix,
    )
    __all__.extend([
        "CodeStructureHandler",
        "CodeInfo",
        "extract_code_structure",
        "build_syndrome_lookup_from_matrix",
    ])
except ImportError:
    pass

# JointMLDecoder - Optimal decoder for concatenated codes
# Achieves true p^d scaling by treating full code as single entity
# For [[49,1,9]] = [[7,1,3]] ⊗ [[7,1,3]]: achieves p^5 scaling
# (Hierarchical decoder is limited to p^4)
try:
    from qectostim.decoders.joint_ml_decoder import (
        JointMLDecoder,
        JointMLConfig,
        build_concatenated_Hz,
        build_full_ZL,
        create_steane_steane_decoder,
    )
    __all__.extend([
        "JointMLDecoder",
        "JointMLConfig",
        "build_concatenated_Hz",
        "build_full_ZL",
        "create_steane_steane_decoder",
    ])
except ImportError:
    pass

# JointMinWeightDecoder - Production-ready minimum-weight decoder
# Noise-agnostic decoder that integrates with MultiLevelConcatenatedCode
# Achieves p^((d+1)/2) scaling via joint minimum-weight decoding
try:
    from qectostim.decoders.joint_minimum_weight_decoder import (
        JointMinWeightDecoder,
        JointDecoderConfig,
        JointDecodeResult,
        extract_final_data,
        extract_final_data_by_block,
    )
    __all__.extend([
        "JointMinWeightDecoder",
        "JointDecoderConfig",
        "JointDecodeResult",
        "extract_final_data",
        "extract_final_data_by_block",
    ])
except ImportError:
    pass

# CircuitLevelDecoder - Full circuit measurement record decoder
# Uses syndrome history from EC rounds + final data for circuit-level decoding
# Handles Pauli frame corrections for teleportation-based EC (Knill)
# Supports flag-aware decoding for hook error detection
try:
    from qectostim.decoders.circuit_level_decoder import (
        CircuitLevelDecoder,
        CircuitDecodeConfig,
        CircuitDecodeResult,
        decode_with_metadata,
        FlagDecodingMode,
        FlagConditionedEntry,
    )
    __all__.extend([
        "CircuitLevelDecoder",
        "CircuitDecodeConfig",
        "CircuitDecodeResult",
        "decode_with_metadata",
        "FlagDecodingMode",
        "FlagConditionedEntry",
    ])
except ImportError:
    pass

# NEW: Detector-based decoders (local hard-decision, no MWPM)
# These decoders work on detector sampler output (not measurement sampler)
# They use per-gadget local decoding with code-specific lookup tables
try:
    from qectostim.decoders.local_decoders import (
        LocalDecoder,
        LocalDecoderResult,
        SteaneDecoder,
        ShorDecoder,
        RepetitionDecoder,
        HierarchicalDecoder,
        post_select_verification,
        filter_shots_by_verification,
        get_decoder_for_code,
    )
    __all__.extend([
        "LocalDecoder",
        "LocalDecoderResult",
        "SteaneDecoder",
        "ShorDecoder",
        "RepetitionDecoder",
        "HierarchicalDecoder",
        "post_select_verification",
        "filter_shots_by_verification",
        "get_decoder_for_code",
    ])
except ImportError:
    pass

# Detector-based decoder (main decoder for detector-based architecture)
try:
    from qectostim.decoders.detector_decoder import (
        DetectorBasedDecoder,
        HierarchicalDetectorDecoder,
        DecodingResults,
        ShotResult,
        RejectionReason,
        decode_detector_samples,
    )
    __all__.extend([
        "DetectorBasedDecoder",
        "HierarchicalDetectorDecoder",
        "DecodingResults",
        "ShotResult",
        "RejectionReason",
        "decode_detector_samples",
    ])
except ImportError:
    pass

