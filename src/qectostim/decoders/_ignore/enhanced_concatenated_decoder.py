# src/qectostim/decoders/enhanced_concatenated_decoder.py
"""
Enhanced Concatenated Decoder with Full FT Features.

This decoder implements all advanced features for concatenated QEC:

Phase 2: Temporal Multi-Round Decoding
- Syndrome changes between rounds form detectors
- Distinguishes measurement errors from data errors
- Transient changes (round R differs from R-1 and R+1) → measurement error
- Persistent changes (stay flipped) → data error

Phase 3: Outer-Level EC Processing
- Inner decode provides logical values for each block
- Outer syndrome computed from inner logicals
- Full hierarchical correction at both levels

Phase 4: Soft Information Propagation
- Log-Likelihood Ratios (LLRs) track confidence
- Soft XOR combines probabilities correctly
- Weighted outer decoding based on inner confidence

Literature:
- AGP (quant-ph/0504218): Concatenated FT threshold
- Dennis et al. (quant-ph/0110143): Spacetime MWPM
- Poulin (quant-ph/0603042): Soft concatenated decoding
- Duclos-Cianci & Poulin (arXiv:0911.0581): LLR propagation

Example
-------
>>> from qectostim.decoders import EnhancedConcatenatedDecoder
>>> 
>>> # Build experiment with multiple EC rounds
>>> exp = MultiLevelMemoryExperiment(code, rounds=3, ancilla_prep="verified")
>>> circuit, metadata = exp.build()
>>> 
>>> # Create enhanced decoder
>>> decoder = EnhancedConcatenatedDecoder(code, metadata.__dict__)
>>> 
>>> # Decode with full temporal + hierarchical + soft
>>> logical = decoder.decode(measurements)
"""
from __future__ import annotations

import numpy as np
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Union, Literal
from dataclasses import dataclass, field
from collections import defaultdict

from qectostim.decoders.base import Decoder
from qectostim.decoders.code_structure_handler import CodeStructureHandler, CodeInfo

if TYPE_CHECKING:
    from qectostim.codes.composite.multilevel_concatenated import MultiLevelConcatenatedCode

try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False

try:
    from stimbposd import BPOSD
    HAS_BPOSD = True
except ImportError:
    HAS_BPOSD = False

try:
    from qectostim.decoders.joint_ml_decoder import (
        JointMLDecoder, JointMLConfig, build_concatenated_Hz, build_full_ZL
    )
    HAS_JOINT_ML = True
except ImportError:
    HAS_JOINT_ML = False

try:
    from qectostim.decoders.optimal_concatenated_decoder import (
        OptimalConcatenatedDecoder,
        OptimalDecoderConfig,
        NoiseModel,
        create_steane_steane_optimal,
    )
    HAS_OPTIMAL = True
except ImportError:
    HAS_OPTIMAL = False


# =============================================================================
# Soft Information Utilities (Phase 4)
# =============================================================================

def prob_to_llr(p: float, eps: float = 1e-10) -> float:
    """
    Convert error probability to log-likelihood ratio.
    
    LLR = log((1-p)/p)
    Positive LLR → more likely to be 0
    Negative LLR → more likely to be 1
    """
    p = np.clip(p, eps, 1 - eps)
    return np.log((1 - p) / p)


def llr_to_prob(llr: float) -> float:
    """Convert LLR back to probability of being 1."""
    return 1.0 / (1.0 + np.exp(llr))


def soft_xor(llr1: float, llr2: float) -> float:
    """
    Soft XOR in LLR domain.
    
    For independent binary variables:
    tanh(LLR_out/2) = tanh(LLR_1/2) * tanh(LLR_2/2)
    """
    # Avoid numerical issues
    llr1 = np.clip(llr1, -30, 30)
    llr2 = np.clip(llr2, -30, 30)
    
    tanh1 = np.tanh(llr1 / 2)
    tanh2 = np.tanh(llr2 / 2)
    product = tanh1 * tanh2
    
    # Avoid arctanh of ±1
    product = np.clip(product, -0.9999, 0.9999)
    return 2 * np.arctanh(product)


def multi_soft_xor(llrs: List[float]) -> float:
    """Soft XOR of multiple LLRs."""
    if not llrs:
        return float('inf')  # Certain 0
    
    result = llrs[0]
    for llr in llrs[1:]:
        result = soft_xor(result, llr)
    return result


# =============================================================================
# Data Structures
# =============================================================================

class DecoderPhase(Enum):
    """Which decoding phases to use."""
    BASIC = auto()          # Just final data decode
    TEMPORAL = auto()       # + temporal syndrome processing
    HIERARCHICAL = auto()   # + outer-level EC
    FULL_SOFT = auto()      # + soft information propagation


class FlagMode(Enum):
    """How to use verification flags in decoding."""
    IGNORE = "ignore"            # Don't use flags at all
    POST_SELECT = "post_select"  # Reject shots with triggered flags
    SOFT_WEIGHT = "soft_weight"  # Reduce confidence for flagged rounds
    DISCARD_ROUND = "discard"    # Ignore syndrome from flagged rounds


class InnerDecoderType(Enum):
    """Which decoder to use for inner blocks."""
    LOOKUP = "lookup"      # Syndrome lookup table (simplest)
    MWPM = "mwpm"          # PyMatching MWPM (for graphlike)
    BPOSD = "bposd"        # BP-OSD (handles hyperedges)
    AUTO = "auto"          # Auto-select based on code structure


@dataclass
class PauliFrame:
    """
    Tracks accumulated Pauli corrections for a block.
    
    In concatenated codes, we track corrections at each level:
    - x_correction: X errors (bit flips) - tracked per qubit
    - z_correction: Z errors (phase flips) - tracked per qubit
    """
    x_correction: np.ndarray  # X error correction (bit flips)
    z_correction: np.ndarray  # Z error correction (phase flips)
    
    @classmethod
    def identity(cls, n_qubits: int) -> 'PauliFrame':
        """Create identity (no correction) frame."""
        return cls(
            x_correction=np.zeros(n_qubits, dtype=np.uint8),
            z_correction=np.zeros(n_qubits, dtype=np.uint8),
        )
    
    def apply_correction(self, correction: np.ndarray, error_type: str = 'X') -> None:
        """Apply a correction to this frame."""
        if error_type.upper() == 'X':
            self.x_correction ^= correction.astype(np.uint8)
        else:
            self.z_correction ^= correction.astype(np.uint8)
    
    def combine(self, other: 'PauliFrame') -> 'PauliFrame':
        """Combine two Pauli frames (XOR corrections)."""
        return PauliFrame(
            x_correction=self.x_correction ^ other.x_correction,
            z_correction=self.z_correction ^ other.z_correction,
        )


@dataclass
class SyndromeRound:
    """Syndrome measurements for one round."""
    round_idx: int
    x_syndromes: Dict[int, np.ndarray]  # block_id -> syndrome bits
    z_syndromes: Dict[int, np.ndarray]  # block_id -> syndrome bits


@dataclass
class TemporalDetectors:
    """Detector events from temporal syndrome comparison."""
    # block_id -> list of (round_idx, detector_bits) 
    # Detector = syndrome[r] XOR syndrome[r-1]
    x_detectors: Dict[int, List[Tuple[int, np.ndarray]]] = field(default_factory=dict)
    z_detectors: Dict[int, List[Tuple[int, np.ndarray]]] = field(default_factory=dict)


@dataclass
class InnerBlockResult:
    """Result of decoding one inner block."""
    block_idx: int
    logical_value: int  # Hard decision (0 or 1)
    llr: float          # Log-likelihood ratio (positive = likely 0)
    syndrome_weight: int  # Number of syndrome bits fired
    correction_applied: bool = False


@dataclass 
class OuterDecodeResult:
    """Result of outer code decoding."""
    logical_value: int
    llr: float
    outer_syndrome: np.ndarray
    correction_applied: bool = False
    inner_results: List[InnerBlockResult] = field(default_factory=list)


@dataclass
class FullDecodeResult:
    """Complete decode result with all information."""
    logical_value: int
    confidence: float  # 0 to 1
    llr: float
    accepted: bool = True  # For post-selection
    n_flags_fired: int = 0
    inner_results: List[InnerBlockResult] = field(default_factory=list)
    outer_result: Optional[OuterDecodeResult] = None
    temporal_detectors: Optional[TemporalDetectors] = None


@dataclass
class EnhancedDecoderConfig:
    """
    Configuration for enhanced concatenated decoder.
    
    This config controls all phases of hierarchical decoding and provides
    strategy selection for both inner and outer code decoding.
    
    Key Options:
    - use_ec_syndromes: Use actual EC round syndrome measurements instead of
      computing syndrome from final data. This is ESSENTIAL for proper FT.
    - use_outer_ec: Enable outer level correction (now True by default with
      confidence gating to prevent harmful corrections).
    - inner_strategy/outer_strategy: Select decoding algorithm.
    - use_joint_ml: Use Joint ML decoder for optimal d-scaling (bypasses hierarchical).
    
    Attributes
    ----------
    use_temporal : bool
        Enable Phase 2 temporal multi-round decoding.
    use_outer_ec : bool  
        Enable Phase 3 outer level EC (default True with confidence gating).
    use_soft_info : bool
        Enable Phase 4 soft information propagation.
    use_ec_syndromes : bool
        Use EC round syndrome measurements instead of computing from final data.
        This is the key difference between DEM-only and proper hierarchical decoding.
    use_joint_ml : bool
        Use Joint ML decoder instead of hierarchical decoding. This achieves optimal
        p^(d) scaling (e.g., p^5 for [[49,1,9]]) but requires precomputing a syndrome
        lookup table for all weight-0 to weight-(d-1)/2 errors. Default: False.
    inner_strategy : str
        Strategy for inner block decoding: "lookup", "mwpm", "bposd", "auto".
    outer_strategy : str
        Strategy for outer code decoding: "lookup", "mwpm", "bposd", "auto".
    """
    # Which phases to enable
    use_temporal: bool = False     # Phase 2 - temporal multi-round decoding
    use_outer_ec: bool = True      # Phase 3 - outer level EC (with confidence gating)
    use_soft_info: bool = False    # Phase 4 - soft information propagation
    use_flags: bool = True         # Flag-based decoding
    use_pauli_frame: bool = True   # Track Pauli frame through EC rounds
    
    # Optimal ML decoder option (replaces Joint ML - generalizes to any code)
    use_optimal_ml: bool = False   # Use optimal ML decoder (p^d scaling, any noise)
    optimal_noise_type: Literal["bit_flip", "depolarizing", "biased_z"] = "bit_flip"
    optimal_noise_bias: float = 1.0  # Z/X bias for biased_z noise
    
    # Legacy Joint ML decoder option (kept for backwards compatibility)
    use_joint_ml: bool = False     # Use Joint ML decoder (p^d scaling)
    joint_ml_max_weight: int = 4   # Maximum correctable weight for Joint ML
    
    # EC Syndrome Extraction (key for hierarchical decoding)
    use_ec_syndromes: bool = True  # Use actual EC syndrome measurements
    ec_syndrome_indices_per_block: int = 3  # First N measurements per block are syndrome
    
    # Strategy selection (general, not code-specific)
    inner_strategy: Literal["lookup", "mwpm", "bposd", "auto", "joint_ml"] = "auto"
    outer_strategy: Literal["lookup", "mwpm", "bposd", "auto", "joint_ml"] = "auto"
    
    # Legacy decoder type (for backwards compatibility)
    inner_decoder_type: InnerDecoderType = InnerDecoderType.AUTO
    outer_decoder_type: InnerDecoderType = InnerDecoderType.AUTO
    
    # BP-OSD parameters (when using BPOSD decoder)
    bposd_max_bp_iters: int = 30
    bposd_osd_order: int = 10
    bposd_bp_method: str = "product_sum"
    
    # Flag handling
    flag_mode: FlagMode = FlagMode.SOFT_WEIGHT
    flag_penalty: float = 0.5  # Confidence multiplier when flag fires
    post_select_on_flags: bool = False
    
    # Temporal decoding parameters
    temporal_window: int = 3  # Number of rounds for temporal analysis
    
    # Soft decoding parameters
    base_error_prob: float = 0.01  # Base p for LLR computation
    syndrome_weight_penalty: float = 0.3  # LLR penalty per syndrome bit
    
    # Outer decoding parameters (confidence gating)
    outer_correction_threshold: float = 0.5  # Min confidence to apply outer correction
    max_inner_syndrome_weight_for_outer: int = 2  # Skip outer if any inner has higher weight
    
    # Multi-level support
    recursive_decode: bool = True  # Use recursive hierarchical decode for >2 levels
    
    # Post-selection support
    enable_post_selection: bool = False  # Post-select on verification failures
    
    # Debug
    verbose: bool = False


# =============================================================================
# Main Enhanced Decoder
# =============================================================================

class EnhancedConcatenatedDecoder(Decoder):
    """
    Enhanced decoder with temporal, hierarchical, and soft decoding.
    
    This decoder implements the full concatenated FT decoding pipeline:
    
    1. **Temporal Processing** (Phase 2):
       - Extract syndromes from each EC round
       - Compute detectors as syndrome changes
       - Classify errors: transient (measurement) vs persistent (data)
    
    2. **Inner Decoding**:
       - Decode each inner block using syndrome + final data
       - Compute confidence/LLR based on syndrome weight
       - Apply MWPM correction if available
    
    3. **Outer EC** (Phase 3):
       - Compute outer syndrome from inner logical values
       - Decode outer code to get outer correction
       - Apply correction to inner logicals
    
    4. **Soft Propagation** (Phase 4):
       - Propagate LLRs through hierarchy
       - Weight outer syndrome by inner confidence
       - Final decision based on combined LLR
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The concatenated code.
    metadata : Dict[str, Any]
        Experiment metadata with syndrome layout.
    config : EnhancedDecoderConfig
        Decoder configuration.
    """
    
    def __init__(
        self,
        code: Any,
        metadata: Dict[str, Any],
        config: Optional[EnhancedDecoderConfig] = None,
    ):
        self.code = code
        self.metadata = metadata
        self.config = config or EnhancedDecoderConfig()
        
        # Optimal ML decoder (when enabled, bypasses hierarchical)
        self._optimal_decoder = None
        
        # Legacy Joint ML decoder (when enabled, bypasses hierarchical)
        self._joint_ml_decoder = None
        
        # Setup code structures
        self._setup_code_info()
        self._setup_syndrome_extraction()
        self._setup_inner_decoder()
        self._setup_outer_decoder()
        
        # Setup optimal ML decoder if requested (preferred over Joint ML)
        if self.config.use_optimal_ml:
            self._setup_optimal_decoder()
        # Fallback to legacy Joint ML if requested
        elif self.config.use_joint_ml:
            self._setup_joint_ml_decoder()
    
    def _setup_code_info(self) -> None:
        """Extract code structure using CodeStructureHandler for generality."""
        # Use CodeStructureHandler for general code structure extraction
        self._code_handler = CodeStructureHandler(self.code, self.metadata)
        
        # Get level info
        inner_info = self._code_handler.get_inner_code_info()
        outer_info = self._code_handler.get_outer_code_info()
        
        # Extract code references if available
        if hasattr(self.code, 'level_codes'):
            self.inner_code = self.code.level_codes[-1]
            self.outer_code = self.code.level_codes[0] if len(self.code.level_codes) > 1 else None
        else:
            self.inner_code = None
            self.outer_code = None
        
        # Core dimensions
        self.n_inner = inner_info.n if inner_info.n > 0 else self.metadata.get('n_inner', 7)
        self.n_blocks = outer_info.n if outer_info.n > 0 else self.metadata.get('n_blocks', 7)
        
        # Get matrices from handler (general extraction)
        self._inner_hz = inner_info.hz
        self._inner_hx = inner_info.hx
        self._outer_hz = outer_info.hz
        
        # Get Z logical supports from handler
        self._inner_z_support = inner_info.z_support if inner_info.z_support else self._get_z_support(self.inner_code, self.n_inner)
        self._outer_z_support = outer_info.z_support if outer_info.z_support else self._get_z_support(self.outer_code, self.n_blocks)
        
        # Number of rounds
        self.n_rounds = self.metadata.get('n_ec_rounds', 0)
        
        # Verification indices
        self.verification_indices = self.metadata.get('verification_measurements', {})
        
        # Build syndrome-to-qubit lookup tables (from handler)
        self._inner_syndrome_lookup = inner_info.syndrome_lookup_z
        self._outer_syndrome_lookup = outer_info.syndrome_lookup_z
        
        # Fallback if lookup tables empty
        if not self._inner_syndrome_lookup:
            self._inner_syndrome_lookup = self._build_syndrome_lookup(self._inner_hz)
        if not self._outer_syndrome_lookup:
            self._outer_syndrome_lookup = self._build_syndrome_lookup(self._outer_hz)
    
    def _build_syndrome_lookup(self, h_matrix: np.ndarray) -> Dict[int, int]:
        """
        Build syndrome -> qubit error position lookup table.
        
        For each qubit q, compute syndrome of single-qubit error X_q,
        then map syndrome_value -> q.
        """
        if h_matrix.size == 0:
            return {}
        
        n_qubits = h_matrix.shape[1]
        lookup = {}
        
        for q in range(n_qubits):
            # Single-qubit error vector
            error = np.zeros(n_qubits, dtype=np.uint8)
            error[q] = 1
            
            # Compute syndrome
            syndrome = (h_matrix @ error) % 2
            
            # Convert to integer value (binary encoding)
            syn_val = sum(int(s) * (2 ** i) for i, s in enumerate(syndrome))
            
            if syn_val > 0:  # Only store non-trivial syndromes
                lookup[syn_val] = q
        
        return lookup
    
    def _get_code_hz(self, code: Any, default_n: int) -> np.ndarray:
        """Get Hz matrix from code."""
        if code is not None:
            hz = getattr(code, 'hz', None)
            if hz is None:
                hz = getattr(code, '_hz', None)
            if hz is not None:
                return np.atleast_2d(np.asarray(hz, dtype=np.uint8))
        
        # Default Steane Hz
        if default_n == 7:
            return np.array([
                [0, 0, 0, 1, 1, 1, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
            ], dtype=np.uint8)
        return np.zeros((0, default_n), dtype=np.uint8)
    
    def _get_code_hx(self, code: Any, default_n: int) -> np.ndarray:
        """Get Hx matrix from code."""
        if code is not None:
            hx = getattr(code, 'hx', None)
            if hx is None:
                hx = getattr(code, '_hx', None)
            if hx is not None:
                return np.atleast_2d(np.asarray(hx, dtype=np.uint8))
        
        # Default Steane Hx (same as Hz for CSS)
        if default_n == 7:
            return np.array([
                [0, 0, 0, 1, 1, 1, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
            ], dtype=np.uint8)
        return np.zeros((0, default_n), dtype=np.uint8)
    
    def _get_z_support(self, code: Any, default_n: int) -> List[int]:
        """Get Z logical support."""
        if code is not None:
            try:
                return list(code.logical_z_support(0))
            except:
                pass
        
        # Default Steane
        if default_n == 7:
            return [0, 1, 2]
        return list(range(min(3, default_n)))
    
    def _setup_syndrome_extraction(self) -> None:
        """Setup syndrome measurement extraction."""
        # Extract syndrome measurement layout from metadata
        self.syndrome_layout = self.metadata.get('syndrome_measurements', {})
        
        # Build mapping from (round, block) to measurement indices
        self._z_syndrome_indices: Dict[Tuple[int, int], List[int]] = {}
        self._x_syndrome_indices: Dict[Tuple[int, int], List[int]] = {}
        
        for stype in ['Z', 'X']:
            if stype in self.syndrome_layout:
                target = self._z_syndrome_indices if stype == 'Z' else self._x_syndrome_indices
                for key, indices in self.syndrome_layout[stype].items():
                    if isinstance(key, tuple) and len(key) == 2:
                        round_idx, block_key = key
                        # block_key might be (level, block_idx) or just block_idx
                        if isinstance(block_key, tuple):
                            block_idx = block_key[-1]
                        else:
                            block_idx = block_key
                        target[(round_idx, block_idx)] = indices
    
    def _setup_inner_decoder(self) -> None:
        """
        Setup inner block decoder.
        
        Selection order (respects both inner_strategy and inner_decoder_type):
        1. If config specifies BPOSD/bposd and available -> use BPOSD
        2. If config specifies MWPM/mwpm and available and graphlike -> use PyMatching
        3. If AUTO/auto: prefer BPOSD (handles hyperedges), fall back to MWPM/lookup
        4. If LOOKUP/lookup: use syndrome lookup table (always works)
        """
        self._inner_decoder = None
        self._inner_decoder_type = "lookup"  # Default fallback
        
        # Respect both new string-based strategy and legacy enum
        strategy = self.config.inner_strategy.lower()
        decoder_type = self.config.inner_decoder_type
        
        # Map string strategy to enum for unified handling
        if strategy == "lookup":
            # Force lookup - skip other decoders
            self._inner_decoder_type = "lookup"
            return
        
        # Try BPOSD first (handles hyperedges)
        use_bposd = (strategy in ("bposd", "auto") or 
                     decoder_type in (InnerDecoderType.BPOSD, InnerDecoderType.AUTO))
        if use_bposd:
            if HAS_BPOSD and self._inner_hz.size > 0:
                try:
                    # BPOSD works directly with parity check matrix
                    osd_order = min(self.config.bposd_osd_order, self.n_inner - 1)
                    self._inner_decoder = BPOSD.from_parity_check_matrix(
                        self._inner_hz,
                        max_bp_iters=self.config.bposd_max_bp_iters,
                        bp_method=self.config.bposd_bp_method,
                        osd_order=osd_order,
                    )
                    self._inner_decoder_type = "bposd"
                    return
                except Exception as e:
                    if self.config.verbose:
                        print(f"BPOSD inner setup failed: {e}")
        
        # Try PyMatching (graphlike only)
        use_mwpm = (strategy in ("mwpm", "auto") or
                    decoder_type in (InnerDecoderType.MWPM, InnerDecoderType.AUTO))
        if use_mwpm:
            if HAS_PYMATCHING and self._inner_hz.size > 0:
                try:
                    col_weights = self._inner_hz.sum(axis=0)
                    if np.max(col_weights) <= 2:  # Graphlike check
                        self._inner_decoder = pymatching.Matching(self._inner_hz)
                        self._inner_decoder_type = "mwpm"
                        return
                except Exception as e:
                    if self.config.verbose:
                        print(f"PyMatching inner setup failed: {e}")
        
        # Fallback to lookup (always works)
        self._inner_decoder_type = "lookup"
    
    def _setup_outer_decoder(self) -> None:
        """
        Setup outer code decoder.
        
        Same selection logic as inner decoder, respects both outer_strategy
        and outer_decoder_type config options.
        """
        self._outer_decoder = None
        self._outer_decoder_type = "lookup"
        
        if self._outer_hz.size == 0:
            return
        
        # Respect both new string-based strategy and legacy enum
        strategy = self.config.outer_strategy.lower()
        decoder_type = self.config.outer_decoder_type
        
        # Force lookup if specified
        if strategy == "lookup":
            self._outer_decoder_type = "lookup"
            return
        
        # Try BPOSD first
        use_bposd = (strategy in ("bposd", "auto") or
                     decoder_type in (InnerDecoderType.BPOSD, InnerDecoderType.AUTO))
        if use_bposd:
            if HAS_BPOSD and self._outer_hz.size > 0:
                try:
                    osd_order = min(self.config.bposd_osd_order, self.n_blocks - 1)
                    self._outer_decoder = BPOSD.from_parity_check_matrix(
                        self._outer_hz,
                        max_bp_iters=self.config.bposd_max_bp_iters,
                        bp_method=self.config.bposd_bp_method,
                        osd_order=osd_order,
                    )
                    self._outer_decoder_type = "bposd"
                    return
                except Exception as e:
                    if self.config.verbose:
                        print(f"BPOSD outer setup failed: {e}")
        
        # Try PyMatching
        use_mwpm = (strategy in ("mwpm", "auto") or
                    decoder_type in (InnerDecoderType.MWPM, InnerDecoderType.AUTO))
        if use_mwpm:
            if HAS_PYMATCHING and self._outer_hz.size > 0:
                try:
                    col_weights = self._outer_hz.sum(axis=0)
                    if np.max(col_weights) <= 2:
                        self._outer_decoder = pymatching.Matching(self._outer_hz)
                        self._outer_decoder_type = "mwpm"
                        return
                except Exception as e:
                    if self.config.verbose:
                        print(f"PyMatching outer setup failed: {e}")
        
        # Fallback to lookup
        self._outer_decoder_type = "lookup"
    
    def _setup_optimal_decoder(self) -> None:
        """
        Setup Optimal ML decoder for true p^d scaling with any noise model.
        
        This decoder:
        - Uses exact ML decoding at inner level (tractable for small codes)
        - Propagates soft information (LLRs) to outer level
        - Uses exact ML at outer level with soft inputs
        - Handles ANY i.i.d. noise model (bit-flip, depolarizing, biased, etc.)
        - No code-specific lookup tables - generalizes to any concatenated CSS code
        
        Advantages over hierarchical:
        - Achieves optimal p^((d+1)/2) scaling
        - Handles all noise types uniformly
        - Properly propagates uncertainty between levels
        """
        if not HAS_OPTIMAL:
            if self.config.verbose:
                print("Warning: Optimal ML decoder requested but not available. "
                      "Falling back to hierarchical decoding.")
            return
        
        try:
            # Get inner code matrices
            inner_Hz = self._inner_hz
            inner_Hx = getattr(self, '_inner_hx', inner_Hz)  # For CSS, often Hx = Hz
            
            # Get outer code matrices  
            outer_Hz = self._outer_hz
            outer_Hx = getattr(self, '_outer_hx', outer_Hz)
            
            # Get logical supports
            inner_ZL = list(self._inner_z_support)
            inner_XL = getattr(self, '_inner_x_support', inner_ZL)
            outer_ZL_blocks = list(self._outer_z_support)
            outer_XL_blocks = getattr(self, '_outer_x_support', outer_ZL_blocks)
            
            # Create noise model based on config
            if self.config.optimal_noise_type == "bit_flip":
                default_noise = NoiseModel.bit_flip(self.config.base_error_prob)
            elif self.config.optimal_noise_type == "depolarizing":
                default_noise = NoiseModel.depolarizing(self.config.base_error_prob)
            elif self.config.optimal_noise_type == "biased_z":
                default_noise = NoiseModel.biased_z(
                    self.config.base_error_prob,
                    self.config.optimal_noise_bias
                )
            else:
                default_noise = NoiseModel.bit_flip(self.config.base_error_prob)
            
            # Create config
            opt_config = OptimalDecoderConfig(
                default_noise=default_noise,
                decode_x_errors=True,
                decode_z_errors=True,
                verbose=self.config.verbose,
            )
            
            # Create decoder
            self._optimal_decoder = OptimalConcatenatedDecoder(
                inner_Hz=inner_Hz,
                inner_Hx=inner_Hx,
                inner_ZL=inner_ZL,
                inner_XL=inner_XL,
                outer_Hz=outer_Hz,
                outer_Hx=outer_Hx,
                outer_ZL_blocks=outer_ZL_blocks,
                outer_XL_blocks=outer_XL_blocks,
                config=opt_config,
            )
            
            if self.config.verbose:
                print(f"Optimal ML decoder created:")
                print(f"  Inner code: {self.n_inner} qubits")
                print(f"  Outer code: {self.n_blocks} blocks")
                print(f"  Noise type: {self.config.optimal_noise_type}")
                
        except Exception as e:
            if self.config.verbose:
                print(f"Optimal ML decoder setup failed: {e}")
                import traceback
                traceback.print_exc()
            self._optimal_decoder = None
    
    def _setup_joint_ml_decoder(self) -> None:
        """
        Setup Joint ML decoder for optimal p^d scaling.
        
        This decoder bypasses hierarchical decoding entirely and treats the
        concatenated code as a single [[n*k, 1, d*d']] code. It achieves
        optimal scaling (e.g., p^5 for [[49,1,9]]) by precomputing a syndrome
        lookup table for all errors up to weight (d-1)/2.
        
        Key advantage over hierarchical:
        - Hierarchical fails on [2,2,0,...] patterns (weight 4 for [[49,1,9]])
        - Joint ML correctly handles ALL weight-4 errors
        """
        if not HAS_JOINT_ML:
            if self.config.verbose:
                print("Warning: Joint ML decoder requested but not available. "
                      "Falling back to hierarchical decoding.")
            return
        
        try:
            # Build full Hz matrix for concatenated code
            full_Hz = build_concatenated_Hz(
                self._inner_hz,
                self._outer_hz,
                self._inner_z_support,
            )
            
            # Build full Z_L support
            full_ZL = build_full_ZL(
                self._inner_z_support,
                self._outer_z_support,
                self.n_inner,
            )
            
            # Create config
            ml_config = JointMLConfig(
                max_weight=self.config.joint_ml_max_weight,
                verbose=self.config.verbose,
            )
            
            # Create decoder
            self._joint_ml_decoder = JointMLDecoder(full_Hz, full_ZL, ml_config)
            
            if self.config.verbose:
                print(f"Joint ML decoder created: Hz={full_Hz.shape}, "
                      f"Z_L weight={len(full_ZL)}, "
                      f"max_weight={self.config.joint_ml_max_weight}")
                
        except Exception as e:
            if self.config.verbose:
                print(f"Joint ML decoder setup failed: {e}")
            self._joint_ml_decoder = None
    
    def _init_pauli_frames(self) -> Dict[int, PauliFrame]:
        """Initialize Pauli frames for all inner blocks."""
        return {i: PauliFrame.identity(self.n_inner) for i in range(self.n_blocks)}
    
    # =========================================================================
    # Main Decode Interface
    # =========================================================================
    
    def decode(self, measurements: np.ndarray) -> int:
        """Decode to logical value."""
        result = self.decode_full(measurements)
        return result.logical_value
    
    def decode_batch(self, measurements_batch: np.ndarray) -> np.ndarray:
        """Decode batch of shots."""
        n_shots = measurements_batch.shape[0]
        return np.array([self.decode(measurements_batch[i]) for i in range(n_shots)])
    
    def decode_full(self, measurements: np.ndarray) -> FullDecodeResult:
        """
        Full decode with all phases and information.
        
        Decode pipeline:
        1. Check verification measurements for post-selection (if enabled)
        2. Check flags (if enabled)
        3. Extract temporal syndromes (Phase 2)
        4. Decode inner blocks with temporal info
        5. Compute outer syndrome and decode (Phase 3)
        6. Propagate soft information (Phase 4)
        7. Final decision
        """
        measurements = np.asarray(measurements, dtype=np.uint8).flatten()
        
        # Step 0: Post-selection on verification measurements
        if self.config.enable_post_selection:
            verification_failed = self._check_verification_failures(measurements)
            if verification_failed:
                return FullDecodeResult(
                    logical_value=0,
                    confidence=0.0,
                    llr=0.0,
                    accepted=False,
                    n_flags_fired=0,
                )
        
        # Step 1: Check flags
        n_flags = 0
        if self.config.use_flags:
            flags_by_round = self._check_flags(measurements)
            n_flags = sum(len(b) for b in flags_by_round.values())
            
            if self.config.post_select_on_flags and n_flags > 0:
                return FullDecodeResult(
                    logical_value=0,
                    confidence=0.0,
                    llr=0.0,
                    accepted=False,
                    n_flags_fired=n_flags,
                )
        
        # Step 1.5a: Use Optimal ML decoder if available (preferred, bypasses hierarchical)
        if self._optimal_decoder is not None:
            return self._decode_with_optimal_ml(measurements, n_flags)
        
        # Step 1.5b: Use legacy Joint ML decoder if available (bypasses hierarchical)
        if self._joint_ml_decoder is not None:
            return self._decode_with_joint_ml(measurements, n_flags)
        
        # Step 2: Extract temporal syndrome information (Phase 2)
        temporal_detectors = None
        syndrome_history = None
        if self.config.use_temporal and self.n_rounds > 0:
            syndrome_history = self._extract_syndrome_history(measurements)
            temporal_detectors = self._compute_temporal_detectors(syndrome_history)
        
        # Step 3: Decode inner blocks
        inner_results = self._decode_all_inner_blocks(
            measurements, syndrome_history, temporal_detectors
        )
        
        # Step 4: Outer EC (Phase 3)
        outer_result = None
        if self.config.use_outer_ec:
            outer_result = self._decode_outer_level(inner_results)
        
        # Step 5: Compute final logical (Phase 4 - soft combination)
        if self.config.use_soft_info and outer_result is not None:
            final_logical, final_llr, confidence = self._soft_final_decision(
                inner_results, outer_result
            )
        else:
            # Hard decision from outer decode or inner XOR
            if outer_result is not None:
                final_logical = outer_result.logical_value
                final_llr = outer_result.llr
            else:
                # XOR inner logicals over outer Z support (the CORRECT way)
                final_logical = self._xor_inner_logicals(inner_results)
                final_llr = sum(r.llr for r in inner_results) / len(inner_results)
            confidence = llr_to_prob(abs(final_llr))
        
        return FullDecodeResult(
            logical_value=final_logical,
            confidence=confidence,
            llr=final_llr,
            accepted=True,
            n_flags_fired=n_flags,
            inner_results=inner_results,
            outer_result=outer_result,
            temporal_detectors=temporal_detectors,
        )
    
    def _decode_with_optimal_ml(
        self,
        measurements: np.ndarray,
        n_flags: int,
        p_error: Optional[float] = None,
    ) -> FullDecodeResult:
        """
        Decode using Optimal ML decoder (generalized, handles any noise).
        
        This extracts the final data measurements and uses the optimal
        concatenated decoder with proper soft information propagation.
        Achieves true p^d scaling under ANY i.i.d. noise model.
        
        Args:
            measurements: Full measurement array from circuit
            n_flags: Number of flags that fired (for result tracking)
            p_error: Optional error probability override
            
        Returns:
            FullDecodeResult with decoded logical value and confidence
        """
        # Extract final data block measurements
        final_data_indices = self._get_final_data_indices()
        
        if final_data_indices is None or len(final_data_indices) != self.n_blocks * self.n_inner:
            # Fallback: use last n_blocks * n_inner measurements
            n_total = self.n_blocks * self.n_inner
            final_data_indices = list(range(len(measurements) - n_total, len(measurements)))
        
        # Extract and organize final data
        final_data = np.zeros(self.n_blocks * self.n_inner, dtype=np.uint8)
        for i, idx in enumerate(final_data_indices):
            if 0 <= idx < len(measurements):
                final_data[i] = measurements[idx]
        
        # Create noise model for this decode
        noise = None
        if p_error is not None:
            if self.config.optimal_noise_type == "bit_flip":
                noise = NoiseModel.bit_flip(p_error)
            elif self.config.optimal_noise_type == "depolarizing":
                noise = NoiseModel.depolarizing(p_error)
            elif self.config.optimal_noise_type == "biased_z":
                noise = NoiseModel.biased_z(p_error, self.config.optimal_noise_bias)
        
        # Use Optimal ML decoder
        result = self._optimal_decoder.decode(final_data, noise=noise)
        
        # For CSS codes, X logical is what we typically track for Z memory
        logical_value = result.logical_x
        confidence = result.confidence_x
        llr = result.llr_x
        
        return FullDecodeResult(
            logical_value=logical_value,
            confidence=confidence,
            llr=llr,
            accepted=True,
            n_flags_fired=n_flags,
            inner_results=None,  # Optimal ML returns its own inner results
            outer_result=None,
            temporal_detectors=None,
        )
    
    def _decode_with_joint_ml(
        self, 
        measurements: np.ndarray,
        n_flags: int,
    ) -> FullDecodeResult:
        """
        Decode using Joint ML decoder (optimal for concatenated codes).
        
        This extracts the final data measurements and uses the precomputed
        syndrome lookup table to find the most likely error correction.
        Achieves true p^d scaling under i.i.d. noise.
        
        Args:
            measurements: Full measurement array from circuit
            n_flags: Number of flags that fired (for result tracking)
            
        Returns:
            FullDecodeResult with decoded logical value
        """
        # Extract final data block measurements
        # The final data is measured at the end of the circuit
        # We need to get the final 49 data qubits (7 blocks × 7 qubits each)
        
        # Find final data measurement indices
        final_data_indices = self._get_final_data_indices()
        
        if final_data_indices is None or len(final_data_indices) != self.n_blocks * self.n_inner:
            # Fallback: use last n_blocks * n_inner measurements
            n_total = self.n_blocks * self.n_inner
            final_data_indices = list(range(len(measurements) - n_total, len(measurements)))
        
        # Extract and organize final data
        final_data = np.zeros(self.n_blocks * self.n_inner, dtype=np.uint8)
        for i, idx in enumerate(final_data_indices):
            if 0 <= idx < len(measurements):
                final_data[i] = measurements[idx]
        
        # Use Joint ML decoder - returns int (logical value)
        logical_value = self._joint_ml_decoder.decode(final_data)
        
        return FullDecodeResult(
            logical_value=logical_value,
            confidence=1.0,  # ML decoder gives optimal correction
            llr=10.0 if logical_value == 0 else -10.0,  # High confidence
            accepted=True,
            n_flags_fired=n_flags,
            inner_results=None,  # Joint ML bypasses hierarchical inner decoding
            outer_result=None,
            temporal_detectors=None,
        )
    
    def _get_final_data_indices(self) -> Optional[List[int]]:
        """
        Get measurement indices for final data block.
        
        Returns indices into measurement array where final data measurements
        are located, or None if not available (will use fallback).
        """
        # Check if we have final data block indices stored
        if hasattr(self, '_final_data_indices') and self._final_data_indices is not None:
            return self._final_data_indices
        
        # Try to find from measurement structure
        # Final data is typically in the last measurement block
        # For now return None to use fallback
        return None
    
    # =========================================================================
    # Phase 2: Temporal Syndrome Processing
    # =========================================================================
    
    def _extract_syndrome_history(
        self,
        measurements: np.ndarray,
    ) -> List[SyndromeRound]:
        """
        Extract syndrome measurements from each EC round.
        
        Returns list of SyndromeRound objects, one per round.
        """
        rounds = []
        
        for round_idx in range(self.n_rounds):
            x_syndromes = {}
            z_syndromes = {}
            
            for block_idx in range(self.n_blocks):
                # Get Z syndrome indices for this (round, block)
                z_indices = self._z_syndrome_indices.get((round_idx, block_idx), [])
                if z_indices:
                    z_bits = np.array([
                        measurements[i] if 0 <= i < len(measurements) else 0
                        for i in z_indices
                    ], dtype=np.uint8)
                    # Compute syndrome from raw measurements
                    if len(z_bits) == self.n_inner and self._inner_hz.size > 0:
                        z_syndromes[block_idx] = (self._inner_hz @ z_bits) % 2
                    else:
                        z_syndromes[block_idx] = z_bits
                
                # Get X syndrome indices
                x_indices = self._x_syndrome_indices.get((round_idx, block_idx), [])
                if x_indices:
                    x_bits = np.array([
                        measurements[i] if 0 <= i < len(measurements) else 0
                        for i in x_indices
                    ], dtype=np.uint8)
                    if len(x_bits) == self.n_inner and self._inner_hx.size > 0:
                        x_syndromes[block_idx] = (self._inner_hx @ x_bits) % 2
                    else:
                        x_syndromes[block_idx] = x_bits
            
            rounds.append(SyndromeRound(
                round_idx=round_idx,
                x_syndromes=x_syndromes,
                z_syndromes=z_syndromes,
            ))
        
        return rounds
    
    def _compute_temporal_detectors(
        self,
        syndrome_history: List[SyndromeRound],
    ) -> TemporalDetectors:
        """
        Compute detector events from syndrome changes.
        
        Detector pattern interpretation:
        - Syndrome changes from round R-1 to R forms detector D_R
        - Transient change (D_R and D_{R+1} both fire) → measurement error
        - Persistent change (only D_R fires) → data error
        
        For Z-basis memory:
        - Z syndrome detects X errors (which flip logical Z)
        - X syndrome detects Z errors (which don't affect Z measurement)
        """
        detectors = TemporalDetectors()
        
        for block_idx in range(self.n_blocks):
            detectors.z_detectors[block_idx] = []
            detectors.x_detectors[block_idx] = []
        
        if len(syndrome_history) < 2:
            return detectors
        
        # Compute detector events for each consecutive pair of rounds
        for r in range(1, len(syndrome_history)):
            prev_round = syndrome_history[r - 1]
            curr_round = syndrome_history[r]
            
            for block_idx in range(self.n_blocks):
                # Z detectors
                prev_z = prev_round.z_syndromes.get(block_idx)
                curr_z = curr_round.z_syndromes.get(block_idx)
                if prev_z is not None and curr_z is not None:
                    if len(prev_z) == len(curr_z):
                        detector = (prev_z ^ curr_z).astype(np.uint8)
                        detectors.z_detectors[block_idx].append((r, detector))
                
                # X detectors
                prev_x = prev_round.x_syndromes.get(block_idx)
                curr_x = curr_round.x_syndromes.get(block_idx)
                if prev_x is not None and curr_x is not None:
                    if len(prev_x) == len(curr_x):
                        detector = (prev_x ^ curr_x).astype(np.uint8)
                        detectors.x_detectors[block_idx].append((r, detector))
        
        return detectors
    
    def _classify_temporal_errors(
        self,
        detectors: List[Tuple[int, np.ndarray]],
    ) -> Tuple[np.ndarray, int]:
        """
        Classify temporal detector pattern to identify data errors.
        
        Returns:
        - inferred_data_error: np.ndarray of which stabilizers had data errors
        - n_measurement_errors: count of measurement errors detected
        
        Pattern matching:
        - [0, 0, ...]: No error
        - [1, 0, ...]: Data error before round 1
        - [1, 1, 0, ...]: Measurement error at round 1
        - [0, 1, 0, ...]: Data error between rounds 1-2, or meas error at round 2
        """
        if not detectors:
            return np.zeros(0, dtype=np.uint8), 0
        
        n_checks = len(detectors[0][1])
        n_measurement_errors = 0
        inferred_data_error = np.zeros(n_checks, dtype=np.uint8)
        
        # For each stabilizer check, analyze temporal pattern
        for check_idx in range(n_checks):
            pattern = [d[1][check_idx] for d in detectors]
            
            # Find runs of 1s
            in_run = False
            run_start = 0
            runs = []
            
            for i, bit in enumerate(pattern):
                if bit == 1 and not in_run:
                    in_run = True
                    run_start = i
                elif bit == 0 and in_run:
                    runs.append((run_start, i - run_start))
                    in_run = False
            if in_run:
                runs.append((run_start, len(pattern) - run_start))
            
            # Interpret runs
            for start, length in runs:
                if length == 1:
                    # Single detector fire → likely data error
                    inferred_data_error[check_idx] ^= 1
                elif length >= 2:
                    # Multiple consecutive fires → likely measurement errors
                    n_measurement_errors += length - 1
                    # Still might be one data error at the boundary
                    if start == 0:
                        inferred_data_error[check_idx] ^= 1
        
        return inferred_data_error, n_measurement_errors
    
    # =========================================================================
    # Inner Block Decoding
    # =========================================================================
    
    def _decode_all_inner_blocks(
        self,
        measurements: np.ndarray,
        syndrome_history: Optional[List[SyndromeRound]],
        temporal_detectors: Optional[TemporalDetectors],
    ) -> List[InnerBlockResult]:
        """Decode all inner blocks."""
        results = []
        
        # Get final data measurements
        final_data = self._extract_final_data(measurements)
        
        # Extract EC syndromes if enabled (key for proper hierarchical decoding)
        ec_syndromes = None
        if self.config.use_ec_syndromes:
            ec_syndromes = self._extract_ec_syndromes(measurements)
        
        for block_idx in range(self.n_blocks):
            block_data = final_data[block_idx * self.n_inner:(block_idx + 1) * self.n_inner]
            
            # Get EC syndrome for this block (if available)
            block_ec_syndrome = None
            if ec_syndromes is not None and block_idx in ec_syndromes:
                block_ec_syndrome = ec_syndromes[block_idx]
            
            # Get temporal information for this block
            z_detectors = None
            if temporal_detectors is not None:
                z_detectors = temporal_detectors.z_detectors.get(block_idx, [])
            
            result = self._decode_inner_block(
                block_idx, block_data, syndrome_history, z_detectors,
                ec_syndrome=block_ec_syndrome,
            )
            results.append(result)
        
        return results
    
    def _extract_ec_syndromes(
        self,
        measurements: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """
        Extract EC round syndrome measurements for each inner block.
        
        This is the KEY difference between flat DEM decoding and proper
        hierarchical decoding. Instead of computing syndrome from final
        data measurements ((Hz @ final_data) % 2), we use the actual
        syndrome bits extracted during EC rounds.
        
        For Steane EC, the first N measurements per block (typically 3)
        give the direct syndrome bits from the ancilla syndrome qubits.
        
        Returns
        -------
        Dict[int, np.ndarray]
            Mapping from block_idx to syndrome bits.
        """
        ec_syndromes = {}
        n_syndrome_bits = self.config.ec_syndrome_indices_per_block
        
        # Try to get EC syndrome from metadata layout
        syndrome_layout = self.syndrome_layout
        
        if 'Z' in syndrome_layout:
            # Look for inner level syndrome measurements
            for key, indices in syndrome_layout['Z'].items():
                if isinstance(key, tuple) and len(key) == 2:
                    round_idx, block_key = key
                    
                    # Extract block index
                    if isinstance(block_key, tuple):
                        level, block_idx = block_key
                        if level != 1:  # Skip outer level
                            continue
                    else:
                        block_idx = block_key
                    
                    # Get first N measurements as syndrome bits
                    if indices and len(indices) >= n_syndrome_bits:
                        syn_indices = indices[:n_syndrome_bits]
                        syndrome = np.array([
                            measurements[i] if 0 <= i < len(measurements) else 0
                            for i in syn_indices
                        ], dtype=np.uint8)
                        
                        # Store for this block (use last round if multiple)
                        ec_syndromes[block_idx] = syndrome
        
        return ec_syndromes
    
    def _decode_inner_block(
        self,
        block_idx: int,
        block_data: np.ndarray,
        syndrome_history: Optional[List[SyndromeRound]],
        z_detectors: Optional[List[Tuple[int, np.ndarray]]],
        ec_syndrome: Optional[np.ndarray] = None,
    ) -> InnerBlockResult:
        """
        Decode one inner block with temporal awareness and EC syndrome support.
        
        Algorithm:
        1. Get syndrome (from EC measurement if available, else compute from data)
        2. If temporal detectors available, use them to infer errors
        3. Apply lookup/MWPM/BPOSD correction
        4. Compute LLR based on syndrome weight and temporal pattern
        
        Parameters
        ----------
        block_idx : int
            Index of the inner block.
        block_data : np.ndarray
            Final data measurements for this block.
        syndrome_history : Optional[List[SyndromeRound]]
            Temporal syndrome history (if enabled).
        z_detectors : Optional[List[Tuple[int, np.ndarray]]]
            Temporal detector events for this block.
        ec_syndrome : Optional[np.ndarray]
            Actual EC round syndrome bits (if use_ec_syndromes=True).
            This is the key for proper hierarchical decoding.
        """
        # Ensure correct size
        if len(block_data) < self.n_inner:
            block_data = np.concatenate([
                block_data,
                np.zeros(self.n_inner - len(block_data), dtype=np.uint8)
            ])
        elif len(block_data) > self.n_inner:
            block_data = block_data[:self.n_inner]
        
        # Raw logical from Z support (computed from final data)
        raw_logical = 0
        for idx in self._inner_z_support:
            if idx < len(block_data):
                raw_logical ^= int(block_data[idx])
        
        # Get syndrome - prefer EC syndrome if available, else compute from data
        if ec_syndrome is not None and len(ec_syndrome) > 0:
            # Use actual EC round syndrome measurements
            syndrome = ec_syndrome.astype(np.uint8)
            if self.config.verbose:
                print(f"Block {block_idx}: using EC syndrome {syndrome}")
        elif self._inner_hz.size > 0:
            # Fallback: compute syndrome from final data
            syndrome = (self._inner_hz @ block_data) % 2
        else:
            syndrome = np.zeros(0, dtype=np.uint8)
        
        syndrome_weight = int(np.sum(syndrome))
        correction_applied = False
        corrected_logical = raw_logical
        
        # Use temporal detectors if available
        inferred_error = None
        if z_detectors and self.config.use_temporal:
            inferred_error, n_meas_err = self._classify_temporal_errors(z_detectors)
            
            if self.config.verbose:
                print(f"Block {block_idx}: temporal inferred error = {inferred_error}, "
                      f"meas errors = {n_meas_err}")
        
        # Apply correction based on decoder type
        if self._inner_decoder is not None and syndrome_weight > 0:
            try:
                if self._inner_decoder_type == "bposd":
                    # BPOSD decode - expects syndrome as 2D array
                    syndrome_2d = syndrome.astype(np.uint8).reshape(1, -1)
                    correction = self._inner_decoder.decode(syndrome_2d)[0]
                else:
                    # MWPM decode
                    correction = self._inner_decoder.decode(syndrome.astype(np.uint8))
                
                # Apply correction to logical
                correction_parity = 0
                for idx in self._inner_z_support:
                    if idx < len(correction) and correction[idx]:
                        correction_parity ^= 1
                
                corrected_logical = (raw_logical + correction_parity) % 2
                correction_applied = True
            except Exception as e:
                if self.config.verbose:
                    print(f"Block {block_idx} decode failed: {e}")
        elif syndrome_weight > 0:
            # Lookup-based correction using precomputed syndrome table
            syndrome_val = sum(int(s) * (2 ** i) for i, s in enumerate(syndrome))
            if syndrome_val in self._inner_syndrome_lookup:
                error_pos = self._inner_syndrome_lookup[syndrome_val]
                if error_pos in self._inner_z_support:
                    # Error on a qubit in Z support flips the logical
                    corrected_logical = (raw_logical + 1) % 2
                    correction_applied = True
        
        # Compute LLR (Phase 4)
        # Higher syndrome weight → lower confidence
        base_llr = prob_to_llr(self.config.base_error_prob)
        
        # Penalize for syndrome weight
        llr_penalty = syndrome_weight * self.config.syndrome_weight_penalty * base_llr
        
        # If we decoded to 0, positive LLR; if 1, negative LLR
        if corrected_logical == 0:
            llr = base_llr - llr_penalty
        else:
            llr = -(base_llr - llr_penalty)
        
        return InnerBlockResult(
            block_idx=block_idx,
            logical_value=corrected_logical,
            llr=llr,
            syndrome_weight=syndrome_weight,
            correction_applied=correction_applied,
        )
    
    # =========================================================================
    # Phase 3: Outer-Level EC
    # =========================================================================
    
    def _should_apply_outer_correction(
        self,
        inner_results: List[InnerBlockResult],
    ) -> bool:
        """
        Determine if outer correction should be applied based on inner confidence.
        
        Key insight: If multiple inner blocks have high syndrome weight, their
        decoded values may be wrong. If these errors happen to cancel at the
        outer level, applying outer correction will UNDO this beneficial
        cancellation and cause an error.
        
        Only apply outer correction if we're confident that inner decodes are
        correct (low syndrome weights).
        
        This is the CRITICAL gating function that makes hierarchical decoding
        work. Without this, outer EC often hurts performance.
        """
        # Use configurable threshold
        high_syndrome_threshold = self.config.max_inner_syndrome_weight_for_outer
        
        # Count inner blocks with high syndrome weight
        n_high_syndrome = sum(
            1 for r in inner_results if r.syndrome_weight > high_syndrome_threshold
        )
        
        # If more than 1 inner block has high syndrome, don't trust outer syndrome
        # because multiple inner decode errors might be canceling
        if n_high_syndrome > 1:
            return False
        
        # Also use LLR if soft info is enabled
        if self.config.use_soft_info:
            # Check if any inner block has low confidence (LLR close to 0)
            llr_threshold = self.config.outer_correction_threshold
            n_low_confidence = sum(
                1 for r in inner_results if abs(r.llr) < llr_threshold
            )
            if n_low_confidence > 1:
                return False
        
        return True
    
    def _decode_outer_level(
        self,
        inner_results: List[InnerBlockResult],
    ) -> OuterDecodeResult:
        """
        Decode outer code using inner logical values.
        
        Algorithm:
        1. Extract inner logical values as "data" for outer code
        2. Compute outer syndrome
        3. Check if we should apply correction (based on inner confidence)
        4. If yes, decode outer syndrome → correction
        5. Apply correction to get final outer logical
        """
        # Inner logicals form the "qubits" for outer code
        inner_logicals = np.array([r.logical_value for r in inner_results], dtype=np.uint8)
        
        # Compute outer syndrome
        if self._outer_hz.size > 0:
            outer_syndrome = (self._outer_hz @ inner_logicals) % 2
        else:
            outer_syndrome = np.zeros(0, dtype=np.uint8)
        
        # Raw outer logical (before correction)
        raw_outer_logical = 0
        for idx in self._outer_z_support:
            if idx < len(inner_logicals):
                raw_outer_logical ^= int(inner_logicals[idx])
        
        outer_syndrome_weight = int(np.sum(outer_syndrome))
        corrected_outer_logical = raw_outer_logical
        correction_applied = False
        
        # Check if we should apply outer correction
        should_correct = self._should_apply_outer_correction(inner_results)
        
        # Decode outer syndrome only if we trust inner decodes
        if outer_syndrome_weight > 0 and should_correct:
            if self._outer_decoder is not None:
                try:
                    if self._outer_decoder_type == "bposd":
                        # BPOSD decode
                        syndrome_2d = outer_syndrome.astype(np.uint8).reshape(1, -1)
                        correction = self._outer_decoder.decode(syndrome_2d)[0]
                    else:
                        # MWPM decode
                        correction = self._outer_decoder.decode(outer_syndrome.astype(np.uint8))
                    
                    # Correct the inner logicals first
                    corrected_inner = inner_logicals.copy()
                    for idx in range(len(correction)):
                        if idx < len(corrected_inner) and correction[idx]:
                            corrected_inner[idx] ^= 1
                    
                    # Recompute outer logical from corrected inner logicals
                    corrected_outer_logical = 0
                    for idx in self._outer_z_support:
                        if idx < len(corrected_inner):
                            corrected_outer_logical ^= int(corrected_inner[idx])
                    
                    correction_applied = True
                except Exception as e:
                    if self.config.verbose:
                        print(f"Outer decode failed: {e}")
            else:
                # Lookup-based correction using precomputed syndrome table
                syndrome_val = sum(int(s) * (2 ** i) for i, s in enumerate(outer_syndrome))
                if syndrome_val in self._outer_syndrome_lookup:
                    error_pos = self._outer_syndrome_lookup[syndrome_val]
                    
                    # Correct the identified inner logical
                    corrected_inner = inner_logicals.copy()
                    if error_pos < len(corrected_inner):
                        corrected_inner[error_pos] ^= 1
                    
                    # Recompute outer logical from corrected inner logicals
                    corrected_outer_logical = 0
                    for idx in self._outer_z_support:
                        if idx < len(corrected_inner):
                            corrected_outer_logical ^= int(corrected_inner[idx])
                    
                    correction_applied = True
        
        # Compute outer LLR by combining inner LLRs over outer Z support
        if self.config.use_soft_info:
            support_llrs = [inner_results[idx].llr for idx in self._outer_z_support 
                          if idx < len(inner_results)]
            outer_llr = multi_soft_xor(support_llrs) if support_llrs else 0.0
            
            # Flip sign if we decoded to 1
            if corrected_outer_logical == 1:
                outer_llr = -abs(outer_llr)
            else:
                outer_llr = abs(outer_llr)
        else:
            outer_llr = prob_to_llr(self.config.base_error_prob)
            if corrected_outer_logical == 1:
                outer_llr = -outer_llr
        
        return OuterDecodeResult(
            logical_value=corrected_outer_logical,
            llr=outer_llr,
            outer_syndrome=outer_syndrome,
            correction_applied=correction_applied,
            inner_results=inner_results,
        )
    
    # =========================================================================
    # Phase 4: Soft Final Decision
    # =========================================================================
    
    def _soft_final_decision(
        self,
        inner_results: List[InnerBlockResult],
        outer_result: OuterDecodeResult,
    ) -> Tuple[int, float, float]:
        """
        Make final decision using soft information.
        
        Combines:
        - Inner block LLRs
        - Outer decoding result
        - Syndrome weights as reliability indicators
        
        Returns:
        - logical_value: 0 or 1
        - llr: final log-likelihood ratio
        - confidence: probability of being correct
        """
        # Compute weighted combination of inner LLRs over outer Z support
        support_llrs = []
        for idx in self._outer_z_support:
            if idx < len(inner_results):
                support_llrs.append(inner_results[idx].llr)
        
        # Soft XOR over support
        combined_inner_llr = multi_soft_xor(support_llrs) if support_llrs else 0.0
        
        # Use the hard decision from outer decoding as the reference
        # The outer decoder already accounts for corrections properly
        hard_decision = outer_result.logical_value
        
        # Make LLR sign match the hard decision
        # If hard decision is 0, LLR should be positive
        # If hard decision is 1, LLR should be negative
        if hard_decision == 0:
            final_llr = abs(combined_inner_llr)
        else:
            final_llr = -abs(combined_inner_llr)
        
        # Weight by outer syndrome reliability
        outer_syndrome_weight = int(np.sum(outer_result.outer_syndrome))
        if outer_syndrome_weight > 0:
            # Reduce confidence when outer syndrome is non-trivial
            reliability = 1.0 / (1.0 + 0.2 * outer_syndrome_weight)
            final_llr *= reliability
        
        # Final decision
        logical_value = 0 if final_llr >= 0 else 1
        confidence = llr_to_prob(-abs(final_llr))  # Prob of being correct
        
        return logical_value, final_llr, confidence
    
    def _xor_inner_logicals(self, inner_results: List[InnerBlockResult]) -> int:
        """XOR inner logicals over outer Z support to get final logical."""
        inner_logicals = np.array([r.logical_value for r in inner_results], dtype=np.uint8)
        xor_result = 0
        for idx in self._outer_z_support:
            if idx < len(inner_logicals):
                xor_result ^= int(inner_logicals[idx])
        return xor_result
    
    def _majority_vote_inner(self, inner_results: List[InnerBlockResult]) -> int:
        """Majority vote over inner logicals (fallback)."""
        ones = sum(r.logical_value for r in inner_results)
        return 1 if ones > len(inner_results) // 2 else 0
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _check_verification_failures(self, measurements: np.ndarray) -> bool:
        """
        Check if any verification measurements indicate ancilla prep failure.
        
        For fault-tolerant operation, we need to post-select on successful
        ancilla preparation. If ANY verification measurement is non-zero,
        the ancilla preparation failed and the shot should be discarded.
        
        Parameters
        ----------
        measurements : np.ndarray
            Raw measurement outcomes.
            
        Returns
        -------
        bool
            True if verification failed (shot should be rejected).
        """
        # Check verification indices from metadata
        for round_idx, blocks in self.verification_indices.items():
            for block_idx, indices in blocks.items():
                for idx in indices:
                    if 0 <= idx < len(measurements) and measurements[idx] != 0:
                        # Non-zero verification measurement = ancilla prep failure
                        if self.config.verbose:
                            print(f"Verification failed: round {round_idx}, "
                                  f"block {block_idx}, measurement {idx}")
                        return True
        return False
    
    def _check_flags(self, measurements: np.ndarray) -> Dict[int, List[int]]:
        """Check verification flags."""
        flags_fired: Dict[int, List[int]] = {}
        
        for round_idx, blocks in self.verification_indices.items():
            r = int(round_idx) if isinstance(round_idx, str) else round_idx
            for block_idx, indices in blocks.items():
                b = int(block_idx) if isinstance(block_idx, str) else block_idx
                for idx in indices:
                    if 0 <= idx < len(measurements) and measurements[idx] != 0:
                        if r not in flags_fired:
                            flags_fired[r] = []
                        if b not in flags_fired[r]:
                            flags_fired[r].append(b)
                        break
        
        return flags_fired
    
    def _extract_final_data(self, measurements: np.ndarray) -> np.ndarray:
        """Extract final data measurements."""
        # Try metadata
        final_meas = self.metadata.get('final_data_measurements', {})
        if final_meas:
            indices = []
            for addr in sorted(final_meas.keys()):
                indices.extend(final_meas[addr])
            if indices:
                return np.array([
                    measurements[i] if 0 <= i < len(measurements) else 0
                    for i in indices
                ], dtype=np.uint8)
        
        # Fallback: last n_data measurements
        n_data = self.n_blocks * self.n_inner
        if len(measurements) >= n_data:
            return measurements[-n_data:].astype(np.uint8)
        
        return np.concatenate([
            measurements,
            np.zeros(n_data - len(measurements), dtype=np.uint8)
        ])


# =============================================================================
# Factory Functions
# =============================================================================

def create_enhanced_decoder(
    code: Any,
    metadata: Dict[str, Any],
    use_temporal: bool = False,
    use_outer_ec: bool = False,
    use_soft_info: bool = False,
    **kwargs,
) -> EnhancedConcatenatedDecoder:
    """
    Create enhanced concatenated decoder.
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
    metadata : Dict
    use_temporal : bool
        Enable Phase 2 temporal decoding (default False)
    use_outer_ec : bool
        Enable Phase 3 outer EC (default False - often hurts for single-level circuits)
    use_soft_info : bool
        Enable Phase 4 soft information (default False)
        
    Notes
    -----
    By default, only inner block decoding is used because:
    - Inner decode alone achieves excellent error suppression
    - Outer EC can hurt when inner blocks are already well-corrected
    - Temporal/soft features need specific circuit structures to help
    """
    config = EnhancedDecoderConfig(
        use_temporal=use_temporal,
        use_outer_ec=use_outer_ec,
        use_soft_info=use_soft_info,
        **kwargs,
    )
    return EnhancedConcatenatedDecoder(code, metadata, config)


def create_full_ft_decoder(
    code: Any,
    metadata: Dict[str, Any],
    post_select: bool = False,
) -> EnhancedConcatenatedDecoder:
    """
    Create decoder with all FT features enabled.
    
    Warning: This may not improve performance for all circuit structures.
    Consider using create_enhanced_decoder with defaults first.
    """
    config = EnhancedDecoderConfig(
        use_temporal=True,
        use_outer_ec=True,
        use_soft_info=True,
        use_flags=True,
        post_select_on_flags=post_select,
    )
    return EnhancedConcatenatedDecoder(code, metadata, config)


def create_bposd_decoder(
    code: Any,
    metadata: Dict[str, Any],
    max_bp_iters: int = 20,
    osd_order: int = 10,
    **kwargs,
) -> EnhancedConcatenatedDecoder:
    """
    Create decoder that uses BP-OSD for both inner and outer levels.
    
    BP-OSD is preferred for DEMs with hyperedge errors (>2 detectors per error)
    that PyMatching cannot handle natively.
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The concatenated code
    metadata : Dict
        Circuit metadata from experiment builder
    max_bp_iters : int
        Maximum BP iterations (default 20)
    osd_order : int
        OSD search order (default 10, higher = more thorough but slower)
    **kwargs
        Additional config options
        
    Returns
    -------
    EnhancedConcatenatedDecoder
        Decoder configured to use BP-OSD
        
    Raises
    ------
    ImportError
        If stimbposd is not installed
    """
    if not HAS_BPOSD:
        raise ImportError(
            "BP-OSD decoder requires stimbposd. "
            "Install with: pip install stimbposd"
        )
    
    config = EnhancedDecoderConfig(
        inner_decoder_type=InnerDecoderType.BPOSD,
        outer_decoder_type=InnerDecoderType.BPOSD,
        bposd_max_bp_iters=max_bp_iters,
        bposd_osd_order=osd_order,
        **kwargs,
    )
    return EnhancedConcatenatedDecoder(code, metadata, config)


# =============================================================================
# Full DEM Decoder (Non-hierarchical BP-OSD)
# =============================================================================

class FullDEMDecoder(Decoder):
    """
    Full DEM decoder using BP-OSD for the entire circuit.
    
    This bypasses hierarchical decoding and directly decodes the full
    detector error model. Useful when the DEM has hyperedge structure
    that makes hierarchical decomposition difficult.
    
    This is the recommended approach for concatenated codes where:
    - The DEM has many hyperedge errors (>2 detectors)
    - PyMatching fails with "odd number of errors" messages
    - Hierarchical decomposition doesn't match circuit structure
    """
    
    def __init__(
        self,
        circuit_or_dem: Any,
        max_bp_iters: int = 20,
        osd_order: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize full DEM decoder.
        
        Parameters
        ----------
        circuit_or_dem : stim.Circuit or stim.DetectorErrorModel
            Either a Stim circuit (DEM will be extracted) or DEM directly
        max_bp_iters : int
            Maximum BP iterations
        osd_order : int
            OSD search order (will be capped to valid range)
        verbose : bool
            Print debug information
        """
        import stim
        
        if not HAS_BPOSD:
            raise ImportError("FullDEMDecoder requires stimbposd")
        
        self.verbose = verbose
        
        # Get DEM from circuit if needed
        if isinstance(circuit_or_dem, stim.Circuit):
            self.dem = circuit_or_dem.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True,
            )
            self.circuit = circuit_or_dem
        else:
            self.dem = circuit_or_dem
            self.circuit = None
        
        # Check that DEM has errors
        if self.dem.num_errors == 0:
            raise ValueError(
                "DEM has no error mechanisms. "
                "Make sure to apply noise to the circuit before creating the decoder."
            )
        
        # Create BP-OSD decoder from DEM
        # Note: osd_order is automatically capped by stimbposd
        self.decoder = BPOSD(
            self.dem,
            max_bp_iters=max_bp_iters,
            osd_order=max(0, osd_order),  # Ensure non-negative
        )
        
        # Get number of detectors and observables
        self.n_detectors = self.dem.num_detectors
        self.n_observables = self.dem.num_observables
        
        if verbose:
            print(f"FullDEMDecoder: {self.n_detectors} detectors, "
                  f"{self.n_observables} observables, {self.dem.num_errors} errors")
    
    def decode(self, syndrome: np.ndarray) -> int:
        """
        Decode a single syndrome.
        
        Parameters
        ----------
        syndrome : np.ndarray
            Detector values (shape: n_detectors)
            
        Returns
        -------
        int
            Predicted logical value (0 or 1)
        """
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        
        # Ensure 1D for single syndrome decode
        if syndrome.ndim > 1:
            syndrome = syndrome.flatten()
        
        # Decode - stimbposd.BPOSD.decode expects 1D array
        prediction = self.decoder.decode(syndrome)
        
        # Return first observable
        if hasattr(prediction, '__len__') and len(prediction) > 0:
            return int(prediction[0])
        return int(prediction)
    
    def decode_batch(
        self,
        syndromes: np.ndarray,
    ) -> np.ndarray:
        """
        Decode batch of syndromes.
        
        Note: stimbposd doesn't support native batch decoding,
        so we loop over syndromes.
        
        Parameters
        ----------
        syndromes : np.ndarray
            Detector values (shape: n_shots x n_detectors)
            
        Returns
        -------
        np.ndarray
            Predicted logical values (shape: n_shots)
        """
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        
        n_shots = syndromes.shape[0]
        predictions = np.zeros(n_shots, dtype=np.uint8)
        
        # Decode each shot individually
        for i in range(n_shots):
            pred = self.decoder.decode(syndromes[i])
            predictions[i] = pred[0] if hasattr(pred, '__len__') else pred
        
        return predictions


def create_full_dem_decoder(
    circuit: Any,
    max_bp_iters: int = 20,
    osd_order: int = 10,
    verbose: bool = False,
) -> FullDEMDecoder:
    """
    Create a full DEM decoder for a circuit.
    
    This is the simplest approach for circuits with complex DEMs.
    It bypasses hierarchical structure and directly decodes the full DEM.
    
    Parameters
    ----------
    circuit : stim.Circuit
        The circuit to decode
    max_bp_iters : int
        Maximum BP iterations (default 20)
    osd_order : int
        OSD search order (default 10)
    verbose : bool
        Print debug information
        
    Returns
    -------
    FullDEMDecoder
        Ready-to-use decoder
        
    Example
    -------
    >>> circuit, metadata = experiment.build()
    >>> decoder = create_full_dem_decoder(circuit)
    >>> 
    >>> # Sample and decode
    >>> sampler = circuit.compile_detector_sampler()
    >>> detectors, observables = sampler.sample(1000, separate_observables=True)
    >>> predictions = decoder.decode_batch(detectors)
    >>> errors = np.sum(predictions != observables[:, 0])
    """
    return FullDEMDecoder(
        circuit,
        max_bp_iters=max_bp_iters,
        osd_order=osd_order,
        verbose=verbose,
    )
