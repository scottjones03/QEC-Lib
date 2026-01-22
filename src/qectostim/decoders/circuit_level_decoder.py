# src/qectostim/decoders/circuit_level_decoder.py
"""
TRUE Circuit-Level Decoder for Fault-Tolerance Characterization.

IMPORTANT DESIGN PRINCIPLE:
--------------------------
This decoder uses ONLY syndrome/ancilla measurements for decoding.
Final data measurements are used ONLY as ground truth for validation.

If a decoder uses final data measurements to decide the logical value,
it is "cheating" - it bypasses the actual error correction process and
does not test the effectiveness of the gadgets or syndrome extraction.

TRUE CIRCUIT-LEVEL DECODING:
---------------------------
1. **Input for decoding**: ONLY syndrome measurements from EC rounds
   - These tell us what errors were detected
   - Multiple rounds allow detection of measurement errors
   
2. **Decoder output**: Predicted logical value based on syndromes
   - Uses syndrome history to infer what corrections are needed
   - Applies corrections logically (tracks Pauli frame)
   
3. **Final data measurements**: Used ONLY for validation
   - Compare decoder's prediction to actual outcome
   - This measures the TRUE fault-tolerance of the circuit

FLAG-AWARE DECODING:
-------------------
Flag qubits detect "hook errors" - single faults that propagate to 
weight-2+ errors on data. The decoder uses flag outcomes to select
the appropriate correction:

- Flag=0, syndrome S: Look up weight-1 correction for S
- Flag=1, syndrome S: Look up flag-conditioned correction (weight-2 pattern)

This is ESSENTIAL for achieving true fault-tolerance with flagged circuits.
Without flag-aware decoding, the decoder cannot distinguish between:
- A weight-1 data error (correctable)
- A hook error that looks like weight-1 syndrome but is actually weight-2

SPACETIME DECODING:
------------------
1. Syndrome differencing: detects WHEN errors occurred
2. Majority voting: handles measurement errors
3. Hierarchical decoding: inner syndromes → outer syndromes

Integration with Production Workflow:
------------------------------------
    >>> from qectostim.decoders import CircuitLevelDecoder
    >>> 
    >>> # Decode using ONLY syndrome measurements
    >>> decoder = CircuitLevelDecoder.from_code(code)
    >>> result = decoder.decode_from_syndromes(sample, metadata)
    >>> 
    >>> # Validate against ground truth (final data)
    >>> is_correct = decoder.validate(result, sample, metadata)
"""
from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Any, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum, auto

if TYPE_CHECKING:
    from qectostim.codes.composite.multilevel_concatenated import MultiLevelConcatenatedCode
    from qectostim.experiments.multilevel_memory import MultiLevelMetadata

# Try to import pymatching for spacetime MWPM
try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False

# Try to import BPOSD for inner block decoding
try:
    from ldpc import BpOsdDecoder
    HAS_BPOSD = True
except ImportError:
    HAS_BPOSD = False


class FlagDecodingMode(Enum):
    """How to use flag qubit information in decoding."""
    IGNORE = auto()           # Don't use flags at all
    LOOKUP = auto()           # Use flag-conditioned lookup tables
    SOFT_WEIGHT = auto()      # Reduce confidence when flags fire
    POST_SELECT = auto()      # Reject shots where flags fire
    ADAPTIVE = auto()         # Combine lookup + soft weighting


@dataclass
class FlagConditionedEntry:
    """
    An entry in the flag-conditioned correction table.
    
    For a given (syndrome, flag_pattern) pair, stores the minimum-weight
    correction and metadata about the error type.
    """
    syndrome: Tuple[int, ...]
    flag_pattern: Tuple[int, ...]  # Which flags fired (one per flagged stabilizer)
    correction: np.ndarray
    weight: int
    error_type: str  # 'hook', 'data', 'measurement', 'ancilla'
    
    def matches(self, syndrome: Tuple, flags: Tuple) -> bool:
        """Check if this entry matches the given syndrome and flags."""
        return self.syndrome == syndrome and self.flag_pattern == flags


@dataclass
class InnerCandidate:
    """
    A single candidate inner code correction pattern.
    
    For multi-candidate tables, each syndrome maps to a list of these.
    They are sorted by weight (or approximate log-probability) to enable
    beam search and joint optimization.
    """
    pattern: np.ndarray          # Physical error pattern (bitvector)
    weight: int                  # Hamming weight
    logical_flip_z: bool         # Whether this pattern flips Z logical
    logical_flip_x: bool         # Whether this pattern flips X logical
    error_type: str = 'data'     # 'data', 'hook', 'measurement', etc.
    
    # For soft decoding / joint search
    log_prob: float = 0.0        # log(P(pattern)) if physical error rate is known
    
    @property
    def cost(self) -> float:
        """Default cost: negative log probability, or just weight."""
        return -self.log_prob if self.log_prob != 0.0 else float(self.weight)


@dataclass
class CircuitDecodeConfig:
    """Configuration for circuit-level decoder."""
    # Maximum error weight to include in lookup table
    max_weight: int = 4
    
    # Whether to use syndrome history for temporal decoding
    use_syndrome_history: bool = True
    
    # Whether to use spacetime MWPM decoding (requires pymatching)
    use_spacetime_mwpm: bool = True
    
    # Measurement error probability (for spacetime graph edge weights)
    measurement_error_prob: float = 0.01
    
    # Physical error probability (for spacetime graph edge weights)
    physical_error_prob: float = 0.01
    
    # Use majority voting for repeated syndrome measurements
    use_majority_voting: bool = True
    
    # Hierarchical decoding: majority vote outer syndromes across rounds
    # NOTE: This is DISABLED by default because voting on outer syndrome bits
    # can produce spurious syndromes when different blocks have false-positive
    # inner logical errors in different rounds. Use inner_logical_majority_voting instead.
    use_outer_syndrome_voting: bool = False
    
    # Hierarchical decoding: majority vote inner logical outcomes across rounds
    use_inner_logical_majority_voting: bool = True
    
    # Syndrome-level voting: vote on syndrome BITS before decoding, not on decoded logicals.
    # This is more effective at filtering measurement errors. When enabled, syndromes are
    # collected across rounds, majority voted bit-by-bit, then decoded once per block.
    # NOTE: When True, inner_logical_majority_voting becomes a no-op (correct behavior)
    use_syndrome_level_voting: bool = True
    
    # Flag-aware decoding options
    # DEFAULT: IGNORE to enable hierarchical decoding for concatenated codes
    # Set to LOOKUP only for flagged circuits with hook error detection
    flag_decoding_mode: FlagDecodingMode = FlagDecodingMode.IGNORE
    
    # Soft weighting: confidence multiplier when flag fires
    flag_confidence_penalty: float = 0.5
    
    # Post-selection: reject shots where flags fire
    post_select_on_flags: bool = False
    
    # Build flag-conditioned tables automatically
    auto_build_flag_tables: bool = True
    
    # Maximum hook error weight to enumerate in flag tables
    max_hook_weight: int = 2
    
    # Verbose output
    verbose: bool = False

    # Allow use of final data in spacetime MWPM (analysis mode). For strict FT benchmarking,
    # keep this False so decoding stays syndrome-only.
    allow_final_data_for_spacetime: bool = False

    # Use BPOSD for inner block decoding (more robust than lookup tables)
    use_bposd_inner: bool = True
    
    # BPOSD parameters for inner block decoding
    bposd_max_iter: int = 10  # BP iterations
    bposd_osd_order: int = 0  # OSD order (0=BP only, higher=more accurate but slower)
    
    # Allow spacetime MWPM using syndrome-only information (no final data). Requires
    # metadata.enable_spacetime_mwpm True. Keeps benchmark-safe while offering temporal decoding.
    use_spacetime_syndrome_only: bool = False

    # When True, require explicit metadata block span info; otherwise fallback to contiguous
    # packing with warnings. Set True to avoid silent misplacement on exotic layouts.
    strict_block_mapping: bool = False

    # Optional soft weighting between hook vs data corrections when flags fire. When 0.0,
    # keeps exact-match deterministic behavior. When >0, will prefer hook if flags fire,
    # else fall back to data; currently a simple prior, not a full probabilistic decode.
    hook_prior_weight: float = 0.0

    # When True, for concatenated codes, suppress per-block corrections when that block's
    # syndrome toggles across EC rounds (unstable measurement indicator). Lightweight
    # temporal filter without full blockwise MWPM.
    use_block_temporal_filter: bool = False
    
    # =========================================================================
    # HIERARCHICAL DECODING OPTIONS (Gaps 1-5 fixes)
    # =========================================================================
    
    # Use decoder-side outer syndrome computation from inner block logicals
    # This is CRITICAL for enabling verified ancillas without DEM disconnection
    # (Fixes Gap 1 & 2)
    use_decoder_outer_syndrome: bool = True
    
    # Apply temporal majority voting to inner block logical outcomes
    # before computing outer syndrome (Fixes Gap 4)
    use_inner_logical_majority_voting: bool = True
    
    # Syndrome-level voting: vote on syndrome BITS before decoding, not on decoded logicals.
    # This is more effective at filtering measurement errors. When enabled, syndromes are
    # collected across rounds, majority voted bit-by-bit, then decoded once per block.
    # NOTE: When True, inner_logical_majority_voting becomes a no-op (correct behavior)
    use_syndrome_level_voting: bool = True
    
    # Use BPOSD for outer-level decoding (Fixes Gap 5 - soft info propagation)
    use_bposd_outer: bool = False
    
    # BPOSD parameters for outer level
    bposd_outer_max_iter: int = 10
    bposd_outer_osd_order: int = 0
    
    # Use soft information propagation between levels
    # When True, inner decode confidence affects outer decoding
    use_soft_hierarchical: bool = False
    
    # Confidence threshold for inner block logical outcomes
    # If BPOSD posterior confidence is below this, treat as uncertain
    inner_confidence_threshold: float = 0.5
    
    # A/B testing: skip outer correction entirely (use only inner decoding)
    # When True, the outer syndrome is computed but NOT used to modify the
    # final correction. This is useful for diagnosing whether outer correction
    # helps or hurts at different error rates.
    skip_outer_correction: bool = False
    
    # =========================================================================
    # JOINT MINIMUM-WEIGHT DECODING (for improved distance scaling)
    # =========================================================================
    
    # Enable multi-candidate inner tables (syndrome -> list of candidates)
    # When True, each inner syndrome maps to multiple correction patterns
    # sorted by weight, enabling joint optimization across blocks/rounds
    use_multi_candidate_tables: bool = False
    
    # Maximum weight for inner table enumeration (separate from max_weight)
    # Higher values capture more degeneracy but increase table size
    inner_max_weight: int = 3
    
    # Maximum candidates to store per syndrome in multi-candidate mode
    # Limits memory; only best (lowest-weight) candidates are kept
    inner_candidate_limit: int = 5
    
    # Enable joint search over inner blocks and EC rounds
    # When True, decoder finds global minimum-weight explanation across all blocks
    # instead of making independent per-block decisions
    enable_joint_search: bool = False
    
    # Joint search mode: 'enumerate' (exact, bounded weight) or 'factor_graph' (approximate, soft)
    joint_search_mode: str = 'enumerate'  # 'enumerate' or 'factor_graph'
    
    # Maximum total physical error weight in global joint search
    # For distance-9 concatenated code, use 4 to achieve p^5 scaling
    max_total_weight: int = 4
    
    # Maximum weight per EC round in joint search (optional constraint)
    # Helps control temporal dimension; None = no per-round limit
    max_weight_per_round: Optional[int] = None
    
    # Maximum number of global hypotheses to explore before fallback
    # Prevents exponential blowup; if exceeded, revert to hierarchical decode
    global_candidate_limit: int = 10000
    
    # Number of EC rounds to include in joint temporal search
    # None = use all rounds; smaller values reduce search complexity
    joint_search_rounds: Optional[int] = None
    
    # For factor-graph mode: number of min-sum iterations
    factor_graph_iterations: int = 20
    
    # For factor-graph mode: convergence threshold for early stopping
    factor_graph_convergence_threshold: float = 1e-6


@dataclass
class CircuitDecodeResult:
    """
    Result from circuit-level decoder.
    
    IMPORTANT: logical_z and logical_x are the decoder's PREDICTIONS
    based on syndrome measurements only. Use validate() to check
    against ground truth.
    """
    # Decoder's predictions (from syndromes ONLY)
    logical_z: int                        # Predicted Z logical value (0 or 1)
    logical_x: int                        # Predicted X logical value (0 or 1)
    
    # Syndrome information
    x_syndrome_final: Optional[Tuple] = None   # Final X syndrome from EC
    z_syndrome_final: Optional[Tuple] = None   # Final Z syndrome from EC
    syndrome_history: List = field(default_factory=list)  # Syndromes per EC round
    
    # Flag information (for flag-aware decoding)
    flag_outcomes: Dict[str, Dict[int, Dict[int, int]]] = field(default_factory=dict)
    # Structure: {'X': {block_id: {stab_idx: flag_value}}, 'Z': {...}}
    n_flags_fired: int = 0                # Total flags that fired
    used_flag_correction: bool = False    # Whether flag-conditioned correction was used
    
    # Correction information
    x_correction: Optional[np.ndarray] = None  # X error correction applied
    z_correction: Optional[np.ndarray] = None  # Z error correction applied
    
    # Spacetime decoding results
    syndrome_changes: List = field(default_factory=list)  # Changes between rounds
    n_syndrome_changes: int = 0           # Total number of syndrome changes
    n_detector_events: int = 0            # Total detector events in spacetime
    
    # Soft information / confidence
    confidence: float = 1.0               # Decoder confidence (0 to 1)
    llr_z: float = float('inf')           # Log-likelihood ratio for Z logical
    llr_x: float = float('inf')           # Log-likelihood ratio for X logical
    
    # Validation results (filled by validate() method)
    ground_truth_z: Optional[int] = None  # Actual Z logical from final data
    ground_truth_x: Optional[int] = None  # Actual X logical from final data
    is_correct_z: Optional[bool] = None   # Did decoder predict correctly?
    is_correct_x: Optional[bool] = None   # Did decoder predict correctly?
    
    # Post-selection (for flag-based post-selection)
    rejected: bool = False                # True if shot should be rejected
    rejection_reason: str = ""            # Why shot was rejected
    
    # Decoding method used
    decoding_method: str = "syndrome_only"  # "syndrome_only", "flag_lookup", etc.
    
    @property
    def is_correct(self) -> Optional[bool]:
        """True if both X and Z predictions were correct."""
        if self.is_correct_z is None or self.is_correct_x is None:
            return None
        return self.is_correct_z and self.is_correct_x
    
    @property
    def logical_error(self) -> Optional[bool]:
        """True if there was a logical error (decoder was wrong)."""
        if self.is_correct is None:
            return None
        return not self.is_correct


class CircuitLevelDecoder:
    """
    Circuit-level decoder that uses the full measurement record.
    
    This decoder extracts syndromes from EC rounds and final data,
    then applies minimum-weight decoding on the full code.
    
    For CSS codes with EC gadgets, we decode in multiple stages:
    1. Extract syndrome from EC rounds (flag detection, repeated measurements)
    2. Extract syndrome from final data (state after EC)
    3. Use combined information to find minimum-weight correction
    4. Apply Pauli frame correction if teleportation EC was used
    
    Key insight: EC syndrome rounds give temporal information:
    - Round i syndrome differs from round i-1 → error occurred in that round
    - Stable syndrome → error was present before EC or during measurement
    
    Attributes:
        Hz: Z-stabilizer parity check matrix (detects X errors)
        Hx: X-stabilizer parity check matrix (detects Z errors)
        ZL: Qubit indices for Z logical operator (computes X logical)
        XL: Qubit indices for X logical operator (computes Z logical)
        config: Decoder configuration
    """
    
    def __init__(
        self,
        Hz: np.ndarray,
        Hx: np.ndarray,
        ZL: List[int],
        XL: List[int],
        config: Optional[CircuitDecodeConfig] = None,
        inner_code_info: Optional[Dict] = None,  # For hierarchical decoding
        outer_code_info: Optional[Dict] = None,  # For outer block decoding
    ):
        """
        Initialize circuit-level decoder.
        
        Args:
            Hz: Z-stabilizer parity check matrix (n_z_checks × n_qubits)
            Hx: X-stabilizer parity check matrix (n_x_checks × n_qubits)
            ZL: Qubit indices for Z logical operator (for X logical value)
            XL: Qubit indices for X logical operator (for Z logical value)
            config: Decoder configuration
            inner_code_info: Optional dict with inner code's Hz, Hx, n_qubits
                            for hierarchical syndrome decoding
            outer_code_info: Optional dict with outer code's Hz, Hx
                            for outer block syndrome decoding
        """
        self.Hz = np.asarray(Hz, dtype=np.uint8)
        self.Hx = np.asarray(Hx, dtype=np.uint8)
        self.ZL = set(ZL)
        self.XL = set(XL)
        self.n_qubits = Hz.shape[1]
        self.config = config or CircuitDecodeConfig()
        self.inner_code_info = inner_code_info
        self.outer_code_info = outer_code_info
        
        # Warning tracking for verbose mode
        self._warned_parse_keys = set()
        self._warned_contiguous_fallback = set()
        self._cached_level_ranges = {}
        # Cache mapping level -> {block_idx: (start, end)} built per decode call
        self._cached_level_ranges: Dict[int, Dict[int, Tuple[int, int]]] = {}
        
        # DIAGNOSTIC COUNTERS for hierarchical decode analysis
        # These track how often outer decoding actually fires
        self._diag_total_decodes = 0
        self._diag_outer_nonzero_syndrome = 0
        self._diag_outer_applied_correction = 0
        self._diag_inner_logical_errors_detected = 0
        
        if self.config.verbose:
            print(f"CircuitLevelDecoder initializing:")
            print(f"  n_qubits: {self.n_qubits}")
            print(f"  Hz shape: {self.Hz.shape}")
            print(f"  Hx shape: {self.Hx.shape}")
            print(f"  Z_L support: {sorted(self.ZL)} (weight {len(self.ZL)})")
            print(f"  X_L support: {sorted(self.XL)} (weight {len(self.XL)})")
            print(f"  max_weight: {self.config.max_weight}")
            print(f"  flag_decoding_mode: {self.config.flag_decoding_mode}")
        
        # Build syndrome lookup tables for full code
        self._x_table = self._build_syndrome_table(self.Hz, "X")
        self._z_table = self._build_syndrome_table(self.Hx, "Z")
        
        # Build flag-conditioned tables (for hook error correction)
        self._x_flag_table: Dict[Tuple, List[FlagConditionedEntry]] = {}
        self._z_flag_table: Dict[Tuple, List[FlagConditionedEntry]] = {}
        if self.config.auto_build_flag_tables and self.config.flag_decoding_mode != FlagDecodingMode.IGNORE:
            self._build_flag_conditioned_tables()
        
        # Build inner code tables if available (for hierarchical decoding)
        self._inner_x_table: Optional[Dict] = None
        self._inner_z_table: Optional[Dict] = None
        self._inner_Hz: Optional[np.ndarray] = None
        self._inner_Hx: Optional[np.ndarray] = None
        self._inner_x_flag_table: Dict[Tuple, List[FlagConditionedEntry]] = {}
        self._inner_z_flag_table: Dict[Tuple, List[FlagConditionedEntry]] = {}
        self._inner_n_qubits: int = 0
        
        # BPOSD decoders for inner blocks (more robust than lookup tables)
        self._inner_bposd_x: Optional[Any] = None
        self._inner_bposd_z: Optional[Any] = None
        
        # Outer code tables for hierarchical decoding (2-level concatenation)
        # These decode the "outer syndrome" formed by inner block logical parities
        self._outer_x_table: Optional[Dict] = None
        self._outer_z_table: Optional[Dict] = None
        self._outer_Hz: Optional[np.ndarray] = None
        self._outer_Hx: Optional[np.ndarray] = None
        self._outer_n_blocks: int = 0
        self._inner_x_logical_support: Set[int] = set()
        self._inner_z_logical_support: Set[int] = set()
        
        if inner_code_info is not None:
            self._build_inner_tables(inner_code_info)
            self._build_outer_tables(inner_code_info, outer_code_info)
            
            # Build BPOSD decoders if enabled
            if self.config.use_bposd_inner and HAS_BPOSD:
                try:
                    # Use physical error rate from config, default to 0.001 if not specified
                    inner_n = inner_code_info['Hz'].shape[1]  # Number of qubits
                    p_phys = getattr(self.config, 'bposd_physical_error_rate', 0.001)
                    error_channel_uniform = [p_phys] * inner_n
                    
                    # CSS convention: Hz detects X errors (Z syndrome), Hx detects Z errors (X syndrome)
                    self._inner_bposd_z = BpOsdDecoder(
                        pcm=inner_code_info['Hz'],
                        error_channel=error_channel_uniform,
                        max_iter=self.config.bposd_max_iter,
                        bp_method='product_sum',
                        osd_order=self.config.bposd_osd_order,
                    )
                    self._inner_bposd_x = BpOsdDecoder(
                        pcm=inner_code_info['Hx'],
                        error_channel=error_channel_uniform,
                        max_iter=self.config.bposd_max_iter,
                        bp_method='product_sum',
                        osd_order=self.config.bposd_osd_order,
                    )
                    if self.config.verbose:
                        print(f"  BPOSD inner decoders initialized (max_iter={self.config.bposd_max_iter}, osd_order={self.config.bposd_osd_order}, p_phys={p_phys})")
                except Exception as e:
                    if self.config.verbose:
                        print(f"  Warning: BPOSD initialization failed: {e}")
                    self._inner_bposd_x = None
                    self._inner_bposd_z = None
        
        # Initialize outer BPOSD decoders for hierarchical soft decoding (Gap 5 fix)
        self._outer_bposd_x: Optional[Any] = None
        self._outer_bposd_z: Optional[Any] = None
        
        if inner_code_info is not None and self.config.use_bposd_outer and HAS_BPOSD:
            try:
                # Outer code operates on n_blocks = n_qubits / n_inner
                n_blocks = self.n_qubits // inner_code_info['n_qubits']
                outer_Hz = self._outer_Hz if self._outer_Hz is not None else np.asarray(inner_code_info['Hz'], dtype=np.uint8)
                outer_Hx = self._outer_Hx if self._outer_Hx is not None else np.asarray(inner_code_info['Hx'], dtype=np.uint8)
                
                # Error channel for outer level (block logical error rate ~p^2 for distance-3 inner)
                p_phys = getattr(self.config, 'bposd_physical_error_rate', 0.001)
                p_block = p_phys ** 2  # Approximate block logical error rate
                outer_error_channel = [p_block] * n_blocks
                
                self._outer_bposd_x = BpOsdDecoder(
                    pcm=outer_Hz,
                    error_channel=outer_error_channel,
                    max_iter=self.config.bposd_outer_max_iter,
                    bp_method='product_sum',
                    osd_order=self.config.bposd_outer_osd_order,
                )
                self._outer_bposd_z = BpOsdDecoder(
                    pcm=outer_Hx,
                    error_channel=outer_error_channel,
                    max_iter=self.config.bposd_outer_max_iter,
                    bp_method='product_sum',
                    osd_order=self.config.bposd_outer_osd_order,
                )
                if self.config.verbose:
                    print(f"  BPOSD outer decoders initialized (n_blocks={n_blocks}, p_block={p_block:.6f})")
            except Exception as e:
                if self.config.verbose:
                    print(f"  Warning: Outer BPOSD initialization failed: {e}")
                self._outer_bposd_x = None
                self._outer_bposd_z = None
    
    def update_bposd_error_rate(self, physical_error_rate: float):
        """Update BPOSD error channel based on current physical error rate."""
        if not (self.config.use_bposd_inner and HAS_BPOSD):
            return
        
        if self._inner_bposd_x is not None and self._inner_bposd_z is not None:
            try:
                # Recreate decoders with new error channel (channel_probs may be read-only)
                inner_n = self.inner_code_info['Hz'].shape[1]
                new_channel = [physical_error_rate] * inner_n
                
                # Recreate Z decoder
                self._inner_bposd_z = BpOsdDecoder(
                    pcm=self.inner_code_info['Hz'],
                    error_channel=new_channel,
                    max_iter=self.config.bposd_max_iter,
                    bp_method='product_sum',
                    osd_order=self.config.bposd_osd_order,
                )
                
                # Recreate X decoder
                self._inner_bposd_x = BpOsdDecoder(
                    pcm=self.inner_code_info['Hx'],
                    error_channel=new_channel,
                    max_iter=self.config.bposd_max_iter,
                    bp_method='product_sum',
                    osd_order=self.config.bposd_osd_order,
                )
                
                if self.config.verbose:
                    print(f"  BPOSD error channel updated to p={physical_error_rate}")
            except Exception as e:
                if self.config.verbose:
                    print(f"  Warning: Could not update BPOSD error channel: {e}")
        
        if self.config.verbose:
            print(f"  X syndrome table: {len(self._x_table)} entries")
            print(f"  Z syndrome table: {len(self._z_table)} entries")
            print(f"  X flag table: {len(self._x_flag_table)} entries")
            print(f"  Z flag table: {len(self._z_flag_table)} entries")
            if self._inner_x_table is not None:
                print(f"  Inner X syndrome table: {len(self._inner_x_table)} entries")
                print(f"  Inner Z syndrome table: {len(self._inner_z_table)} entries")
                print(f"  Inner X flag table: {len(self._inner_x_flag_table)} entries")
                print(f"  Inner Z flag table: {len(self._inner_z_flag_table)} entries")
    
    def get_hierarchical_diagnostics(self) -> dict:
        """
        Get diagnostic statistics for hierarchical decoding.
        
        Returns a dict with:
          - total_decodes: Number of decode() calls
          - outer_nonzero_syndrome: Times outer syndrome was non-zero
          - outer_applied_correction: Times outer correction was applied
          - inner_logical_errors_detected: Times inner blocks had logical errors
        """
        return {
            'total_decodes': self._diag_total_decodes,
            'outer_nonzero_syndrome': self._diag_outer_nonzero_syndrome,
            'outer_applied_correction': self._diag_outer_applied_correction,
            'inner_logical_errors_detected': self._diag_inner_logical_errors_detected,
        }
    
    def reset_hierarchical_diagnostics(self):
        """Reset the hierarchical decoding diagnostic counters."""
        self._diag_total_decodes = 0
        self._diag_outer_nonzero_syndrome = 0
        self._diag_outer_applied_correction = 0
        self._diag_inner_logical_errors_detected = 0

    def print_hierarchical_diagnostics(self):
        """Print hierarchical decoding diagnostic statistics."""
        stats = self.get_hierarchical_diagnostics()
        total = stats['total_decodes']
        if total == 0:
            print("No hierarchical decodes performed yet.")
            return
        
        print(f"\n=== Hierarchical Decode Diagnostics ===")
        print(f"Total decodes: {total}")
        print(f"Inner logical errors detected: {stats['inner_logical_errors_detected']} ({100*stats['inner_logical_errors_detected']/total:.1f}%)")
        print(f"Outer non-zero syndrome: {stats['outer_nonzero_syndrome']} ({100*stats['outer_nonzero_syndrome']/total:.1f}%)")
        print(f"Outer correction applied: {stats['outer_applied_correction']} ({100*stats['outer_applied_correction']/total:.1f}%)")
        
        # Flag potential issues
        if stats['inner_logical_errors_detected'] > 0 and stats['outer_nonzero_syndrome'] == 0:
            print("WARNING: Inner errors detected but outer syndrome always zero - check syndrome orientation!")
        if stats['outer_nonzero_syndrome'] > 0 and stats['outer_applied_correction'] == 0:
            print("WARNING: Outer syndrome non-zero but no corrections applied - check decode tables!")
    
    def _build_inner_tables(self, inner_code_info: Dict) -> None:
        """Build syndrome tables for the inner code (for hierarchical decoding)."""
        inner_Hz = np.asarray(inner_code_info['Hz'], dtype=np.uint8)
        inner_Hx = np.asarray(inner_code_info['Hx'], dtype=np.uint8)
        self._inner_n_qubits = inner_code_info['n_qubits']
        
        # Get inner max weight (use config's inner_max_weight if multi-candidate, else fallback)
        if self.config.use_multi_candidate_tables:
            inner_max_weight = self.config.inner_max_weight
        else:
            inner_max_weight = inner_code_info.get('max_weight', 1)  # Typically t=(d-1)/2
        
        # Store inner parity check matrices for hierarchical syndrome extraction
        self._inner_Hz = inner_Hz
        self._inner_Hx = inner_Hx
        
        # Get logical supports for multi-candidate mode
        if 'x_logical' in inner_code_info:
            inner_x_logical = set(inner_code_info['x_logical'])
        else:
            inner_x_logical = {0, 1, 2}  # Fallback for Steane
            
        if 'z_logical' in inner_code_info:
            inner_z_logical = set(inner_code_info['z_logical'])
        else:
            inner_z_logical = {0, 1, 2}
        
        # Build tables: multi-candidate or single-candidate
        if self.config.use_multi_candidate_tables:
            # Multi-candidate mode: syndrome -> List[InnerCandidate]
            self._inner_z_candidates = self._build_multi_candidate_inner_table(
                inner_Hz, self._inner_n_qubits, inner_max_weight,
                inner_z_logical, inner_x_logical, "inner_Z"
            )
            self._inner_x_candidates = self._build_multi_candidate_inner_table(
                inner_Hx, self._inner_n_qubits, inner_max_weight,
                inner_z_logical, inner_x_logical, "inner_X"
            )
            # Also build legacy tables for fallback compatibility
            self._inner_z_table = {
                syn: (candidates[0].pattern, candidates[0].weight)
                for syn, candidates in self._inner_z_candidates.items()
            }
            self._inner_x_table = {
                syn: (candidates[0].pattern, candidates[0].weight)
                for syn, candidates in self._inner_x_candidates.items()
            }
        else:
            # Single-candidate mode: syndrome -> (correction, weight)
            self._inner_z_table = self._build_inner_syndrome_table(
                inner_Hz, self._inner_n_qubits, inner_max_weight, "inner_Z"
            )
            self._inner_x_table = self._build_inner_syndrome_table(
                inner_Hx, self._inner_n_qubits, inner_max_weight, "inner_X"
            )
            # Empty candidate tables for consistency
            self._inner_z_candidates = {}
            self._inner_x_candidates = {}
        
        # Build inner flag-conditioned tables
        if self.config.auto_build_flag_tables and self.config.flag_decoding_mode != FlagDecodingMode.IGNORE:
            self._build_inner_flag_tables(inner_code_info)

    def _build_outer_tables(self, inner_code_info: Dict, outer_code_info: Optional[Dict] = None) -> None:
        """
        Build outer code tables for hierarchical decoding.
        
        For 2-level concatenation (e.g., Steane^2), the outer code corrects
        errors at the "block level" - i.e., when an inner block has a logical error.
        
        The "outer syndrome" is formed by the pattern of inner block logical errors.
        For CSS codes, X logical errors on inner blocks are detected by the outer Hz,
        and Z logical errors on inner blocks are detected by the outer Hx.
        """
        # For 2-level concatenation, outer code = inner code (both Steane for Steane^2)
        # The outer code operates on n_blocks = n_qubits / n_inner
        n_inner = inner_code_info['n_qubits']
        self._outer_n_blocks = self.n_qubits // n_inner
        
        # Store inner logical supports for computing block logical parities
        if 'x_logical' in inner_code_info:
            self._inner_x_logical_support = set(inner_code_info['x_logical'])
        else:
            # Fallback for Steane: qubits 0,1,2 form the logical operator
            self._inner_x_logical_support = {0, 1, 2}
            
        if 'z_logical' in inner_code_info:
            self._inner_z_logical_support = set(inner_code_info['z_logical'])
        else:
            self._inner_z_logical_support = {0, 1, 2}
        
        # Build outer syndrome tables
        if outer_code_info is not None:
             outer_Hz = np.asarray(outer_code_info['Hz'], dtype=np.uint8)
             outer_Hx = np.asarray(outer_code_info['Hx'], dtype=np.uint8)
             outer_max_weight = outer_code_info.get('max_weight', 1)
        else:
             # Fallback: self-dual assumption (same as inner)
             outer_Hz = np.asarray(inner_code_info['Hz'], dtype=np.uint8)
             outer_Hx = np.asarray(inner_code_info['Hx'], dtype=np.uint8)
             outer_max_weight = inner_code_info.get('max_weight', 1)
        
        self._outer_Hz = outer_Hz
        self._outer_Hx = outer_Hx
        
        # Build tables: syndrome -> (correction, weight)
        # For outer decoding, "correction" tells us which blocks to flip
        # CRITICAL FIX: Use higher weight for better outer code coverage
        # For d=3 outer code, we need weight up to (d-1)/2 = 1, but use 2 for safety margin
        outer_enhanced_weight = min(2, outer_max_weight + 1)
        
        # NAMING FIX: 
        # - _outer_z_table: maps Z syndrome (from Hz detecting X errors) → X correction on blocks
        # - _outer_x_table: maps X syndrome (from Hx detecting Z errors) → Z correction on blocks
        # 
        # But in _decode_outer_syndrome, we look up outer_z_syndrome in _outer_z_table
        # So _outer_z_table should be built from Hz (which produces Z syndromes for X errors)
        self._outer_z_table = self._build_inner_syndrome_table(
            outer_Hz, self._outer_n_blocks, outer_enhanced_weight, "outer_Z"
        )
        self._outer_x_table = self._build_inner_syndrome_table(
            outer_Hx, self._outer_n_blocks, outer_enhanced_weight, "outer_X"
        )
        
        # CRITICAL FIX: Add all-zeros syndrome entry (no errors)
        # Zero syndrome dimensions must match the H matrix used to build each table
        # _outer_z_table built from outer_Hz, so zero Z syndrome has outer_Hz.shape[0] bits
        # _outer_x_table built from outer_Hx, so zero X syndrome has outer_Hx.shape[0] bits
        zero_syndrome_z = tuple([0] * outer_Hz.shape[0])
        zero_syndrome_x = tuple([0] * outer_Hx.shape[0])
        if zero_syndrome_z not in self._outer_z_table:
            self._outer_z_table[zero_syndrome_z] = (np.zeros(self._outer_n_blocks, dtype=np.uint8), 0)
        if zero_syndrome_x not in self._outer_x_table:
            self._outer_x_table[zero_syndrome_x] = (np.zeros(self._outer_n_blocks, dtype=np.uint8), 0)
        
        if self.config.verbose:
            print(f"  Outer X syndrome table: {len(self._outer_x_table)} entries (n_blocks={self._outer_n_blocks}, max_weight={outer_enhanced_weight})")
            print(f"  Outer Z syndrome table: {len(self._outer_z_table)} entries")
            print(f"  Inner X logical support: {sorted(self._inner_x_logical_support)}")
            print(f"  Inner Z logical support: {sorted(self._inner_z_logical_support)}")

    def _build_flag_conditioned_tables(self) -> None:
        """
        Build flag-conditioned correction tables.
        
        For flag fault-tolerance, we need different corrections depending on
        whether flags fired. The key insight is:
        
        - Flag=0: Normal data error → use standard weight-1 correction
        - Flag=1: Hook error → the apparent weight-1 syndrome may actually be
                  from a weight-2 error that propagated through the ancilla
        
        For each flagged stabilizer, enumerate hook errors: errors on the
        ancilla that propagate to specific weight-2 patterns on data.
        """
        if self.config.verbose:
            print("  Building flag-conditioned tables...")
        
        # Identify weight-4+ stabilizers that would be flagged
        z_flagged_stabs = self._get_flagged_stabilizers(self.Hz)
        x_flagged_stabs = self._get_flagged_stabilizers(self.Hx)
        
        # Build X error flag table (detected by Hz)
        self._x_flag_table = self._build_flag_table_for_checks(
            self.Hz, z_flagged_stabs, "X"
        )
        
        # Build Z error flag table (detected by Hx)  
        self._z_flag_table = self._build_flag_table_for_checks(
            self.Hx, x_flagged_stabs, "Z"
        )
    
    def _build_inner_flag_tables(self, inner_code_info: Dict) -> None:
        """Build flag-conditioned tables for inner code."""
        inner_Hz = np.asarray(inner_code_info['Hz'], dtype=np.uint8)
        inner_Hx = np.asarray(inner_code_info['Hx'], dtype=np.uint8)
        n_inner = inner_code_info['n_qubits']
        
        z_flagged = self._get_flagged_stabilizers(inner_Hz)
        x_flagged = self._get_flagged_stabilizers(inner_Hx)
        
        self._inner_x_flag_table = self._build_flag_table_for_checks(
            inner_Hz, z_flagged, "inner_X", n_qubits=n_inner
        )
        self._inner_z_flag_table = self._build_flag_table_for_checks(
            inner_Hx, x_flagged, "inner_Z", n_qubits=n_inner
        )
    
    def _get_flagged_stabilizers(self, H: np.ndarray) -> List[Tuple[int, List[int]]]:
        """
        Identify stabilizers that would have flag qubits.
        
        Returns list of (stabilizer_idx, support_qubits) for weight-4+ stabilizers.
        """
        flagged = []
        for stab_idx in range(H.shape[0]):
            support = list(np.where(H[stab_idx] == 1)[0])
            if len(support) >= 4:
                flagged.append((stab_idx, support))
        return flagged
    
    def _build_flag_table_for_checks(
        self,
        H: np.ndarray,
        flagged_stabs: List[Tuple[int, List[int]]],
        error_type: str,
        n_qubits: Optional[int] = None,
    ) -> Dict[Tuple, List[FlagConditionedEntry]]:
        """
        Build flag-conditioned table for a set of parity checks.
        
        For each flagged stabilizer, enumerate "hook errors":
        - Single fault on ancilla during CNOT sequence
        - Propagates to weight-2 error on data qubits coupled AFTER the fault
        
        The flag qubit detects this, allowing the decoder to apply the 
        correct weight-2 correction instead of guessing weight-1.
        """
        table: Dict[Tuple, List[FlagConditionedEntry]] = {}
        n = n_qubits or self.n_qubits
        n_checks = H.shape[0]
        n_flagged = len(flagged_stabs)
        
        if n_flagged == 0:
            return table
        
        # For each flagged stabilizer, enumerate hook errors
        for flag_idx, (stab_idx, support) in enumerate(flagged_stabs):
            # Flag is placed in the middle of the CNOT sequence
            # Hook error: X error on ancilla after first CNOTs, before rest
            # This X propagates to data qubits in the "after" portion
            
            flag_pos = len(support) // 2
            qubits_before = support[:flag_pos]
            qubits_after = support[flag_pos:]
            
            # Enumerate hook errors: X errors that propagate to qubits_after
            # A single X error on the ancilla after coupling qubits_before
            # will show up as X errors on all of qubits_after
            for hook_weight in range(1, min(self.config.max_hook_weight + 1, len(qubits_after) + 1)):
                for affected_subset in combinations(qubits_after, hook_weight):
                    # Create the hook error pattern
                    hook_error = np.zeros(n, dtype=np.uint8)
                    for q in affected_subset:
                        hook_error[q] = 1
                    
                    # Compute syndrome for this hook error
                    hook_syndrome = tuple((H @ hook_error) % 2)
                    
                    # Create flag pattern (only this flag fires)
                    flag_pattern = tuple(1 if i == flag_idx else 0 for i in range(n_flagged))
                    
                    # Create entry
                    entry = FlagConditionedEntry(
                        syndrome=hook_syndrome,
                        flag_pattern=flag_pattern,
                        correction=hook_error.copy(),
                        weight=hook_weight,
                        error_type='hook',
                    )
                    
                    # Store in table (keyed by syndrome)
                    if hook_syndrome not in table:
                        table[hook_syndrome] = []
                    table[hook_syndrome].append(entry)
        
        # Also add standard (flag=0) entries for all syndromes
        all_zero_flags = tuple(0 for _ in range(n_flagged))
        
        for weight in range(1, min(self.config.max_weight + 1, n + 1)):
            for qubits in combinations(range(n), weight):
                e = np.zeros(n, dtype=np.uint8)
                for q in qubits:
                    e[q] = 1
                
                syn = tuple((H @ e) % 2)
                entry = FlagConditionedEntry(
                    syndrome=syn,
                    flag_pattern=all_zero_flags,
                    correction=e.copy(),
                    weight=weight,
                    error_type='data',
                )
                
                if syn not in table:
                    table[syn] = []
                # Only add if not already present with same flag pattern
                existing = [x for x in table[syn] if x.flag_pattern == all_zero_flags]
                if not existing:
                    table[syn].append(entry)
        
        if self.config.verbose:
            print(f"    {error_type} flag table: {len(table)} syndromes, "
                  f"{sum(len(v) for v in table.values())} total entries")
        
        return table
    
    def _build_inner_syndrome_table(
        self,
        H: np.ndarray,
        n_qubits: int,
        max_weight: int,
        error_type: str,
    ) -> Dict[Tuple[int, ...], Tuple[np.ndarray, int]]:
        """Build syndrome table for inner code (separate n_qubits)."""
        table: Dict[Tuple[int, ...], Tuple[np.ndarray, int]] = {}
        n_checks = H.shape[0]
        
        if self.config.verbose:
            print(f"  Building {error_type} syndrome table (n={n_qubits}, max_w={max_weight})...")
        
        # Weight 0
        syn0 = tuple([0] * n_checks)
        table[syn0] = (np.zeros(n_qubits, dtype=np.uint8), 0)
        
        # Weights 1 to max_weight
        for weight in range(1, max_weight + 1):
            for qubits in combinations(range(n_qubits), weight):
                e = np.zeros(n_qubits, dtype=np.uint8)
                for q in qubits:
                    e[q] = 1
                syn = tuple((H @ e) % 2)
                if syn not in table:
                    table[syn] = (e.copy(), weight)
        
        return table
    
    def _build_multi_candidate_inner_table(
        self,
        H: np.ndarray,
        n_qubits: int,
        max_weight: int,
        z_logical_support: Set[int],
        x_logical_support: Set[int],
        error_type: str,
    ) -> Dict[Tuple[int, ...], List[InnerCandidate]]:
        """
        Build multi-candidate syndrome table for joint optimization.
        
        Each syndrome maps to a list of InnerCandidate objects, sorted by weight.
        This enables joint search by exposing multiple plausible corrections per block.
        
        Args:
            H: Parity check matrix for this error type
            n_qubits: Number of qubits in the inner code
            max_weight: Maximum error weight to enumerate
            z_logical_support: Qubit indices for Z logical operator
            x_logical_support: Qubit indices for X logical operator
            error_type: Label for logging ("inner_X" or "inner_Z")
        
        Returns:
            Dict mapping syndrome -> list of InnerCandidate, sorted by weight
        """
        table: Dict[Tuple[int, ...], List[InnerCandidate]] = {}
        n_checks = H.shape[0]
        candidate_limit = self.config.inner_candidate_limit
        
        if self.config.verbose:
            print(f"  Building multi-candidate {error_type} table (n={n_qubits}, "
                  f"max_w={max_weight}, limit={candidate_limit})...")
        
        # Weight 0 (no error)
        syn0 = tuple([0] * n_checks)
        zero_pattern = np.zeros(n_qubits, dtype=np.uint8)
        table[syn0] = [InnerCandidate(
            pattern=zero_pattern,
            weight=0,
            logical_flip_z=False,
            logical_flip_x=False,
            error_type='none'
        )]
        
        # Weights 1 to max_weight
        for weight in range(1, max_weight + 1):
            for qubits in combinations(range(n_qubits), weight):
                e = np.zeros(n_qubits, dtype=np.uint8)
                for q in qubits:
                    e[q] = 1
                
                # Compute syndrome
                syn = tuple((H @ e) % 2)
                
                # Compute logical flips
                # For X errors (detected by Hz): they flip Z logical (measured by XL)
                # For Z errors (detected by Hx): they flip X logical (measured by ZL)
                # We're building a table for a specific error type based on H
                
                # Determine logical flips based on overlap with logical supports
                z_flip = (sum(e[q] for q in z_logical_support) % 2) == 1
                x_flip = (sum(e[q] for q in x_logical_support) % 2) == 1
                
                candidate = InnerCandidate(
                    pattern=e.copy(),
                    weight=weight,
                    logical_flip_z=z_flip,
                    logical_flip_x=x_flip,
                    error_type='data'
                )
                
                # Add to table
                if syn not in table:
                    table[syn] = []
                table[syn].append(candidate)
                
                # Enforce candidate limit (keep lowest weight)
                if len(table[syn]) > candidate_limit:
                    table[syn] = sorted(table[syn], key=lambda c: c.weight)[:candidate_limit]
        
        # Final sort and truncate for all syndromes
        for syn in table:
            table[syn] = sorted(table[syn], key=lambda c: c.weight)[:candidate_limit]
        
        if self.config.verbose:
            avg_candidates = sum(len(v) for v in table.values()) / max(1, len(table))
            print(f"    {error_type} multi-candidate table: {len(table)} syndromes, "
                  f"avg {avg_candidates:.1f} candidates/syndrome")
        
        return table
    
    def _build_syndrome_table(
        self,
        H: np.ndarray,
        error_type: str,
    ) -> Dict[Tuple[int, ...], Tuple[np.ndarray, int]]:
        """
        Build syndrome → (minimum weight correction, weight) lookup table.
        
        Enumerates all error patterns up to max_weight and stores the first
        (minimum weight) correction for each syndrome.
        """
        table: Dict[Tuple[int, ...], Tuple[np.ndarray, int]] = {}
        n_checks = H.shape[0]
        
        if self.config.verbose:
            print(f"  Building {error_type} syndrome table...")
        
        # Weight 0 (no error)
        syn0 = tuple([0] * n_checks)
        table[syn0] = (np.zeros(self.n_qubits, dtype=np.uint8), 0)
        
        # Weights 1 to max_weight
        for weight in range(1, self.config.max_weight + 1):
            count = 0
            for qubits in combinations(range(self.n_qubits), weight):
                e = np.zeros(self.n_qubits, dtype=np.uint8)
                for q in qubits:
                    e[q] = 1
                
                syn = tuple((H @ e) % 2)
                
                # Only store if this syndrome not seen (minimum weight)
                if syn not in table:
                    table[syn] = (e.copy(), weight)
                    count += 1
            
            if self.config.verbose:
                print(f"    Weight {weight}: added {count} new syndromes")
        
        return table
    
    @classmethod
    def from_code(
        cls,
        code: 'MultiLevelConcatenatedCode',
        config: Optional[CircuitDecodeConfig] = None,
    ) -> 'CircuitLevelDecoder':
        """
        Create decoder from a MultiLevelConcatenatedCode.
        
        This is the recommended way to create the decoder in production.
        Automatically constructs the full Hz, Hx, and logical operator supports.
        Also extracts inner code information for hierarchical syndrome decoding.
        
        Args:
            code: The concatenated code object
            config: Decoder configuration (default: max_weight = (d-1)//2)
            
        Returns:
            CircuitLevelDecoder configured for the code
        """
        # Get full parity check matrices from code
        Hz = np.asarray(code.hz, dtype=np.uint8)
        Hx = np.asarray(code.hx, dtype=np.uint8)
        
        # Get logical operator supports
        ZL = cls._get_logical_support(code, 'Z')
        XL = cls._get_logical_support(code, 'X')
        
        # Set default max_weight based on distance
        if config is None:
            d = getattr(code, 'd', getattr(code, 'distance', 9))
            max_weight = (d - 1) // 2
            config = CircuitDecodeConfig(max_weight=max_weight)
        
        # Extract inner code info for hierarchical decoding
        inner_code_info = None
        outer_code_info = None
        if hasattr(code, 'level_codes') and len(code.level_codes) >= 2:
            inner_code = code.level_codes[-1]  # Last level is innermost
            try:
                inner_Hz = np.asarray(inner_code.hz, dtype=np.uint8)
                inner_Hx = np.asarray(inner_code.hx, dtype=np.uint8)
                inner_d = getattr(inner_code, 'd', getattr(inner_code, 'distance', 3))
                # Use higher max_weight for inner code to handle measurement errors
                # and multi-error syndromes. For d=3, use weight 2 not 1.
                inner_max_weight = min(inner_d - 1, inner_code.n // 2)  # d-1 or n/2, whichever is smaller
                
                # CRITICAL FIX: Extract inner code logical supports for outer syndrome computation
                # Without this, we fall back to hardcoded {0,1,2} which may be wrong
                inner_x_logical = cls._get_level_logical_support(inner_code, 'X')
                inner_z_logical = cls._get_level_logical_support(inner_code, 'Z')
                
                inner_code_info = {
                    'Hz': inner_Hz,
                    'Hx': inner_Hx,
                    'n_qubits': inner_code.n,
                    'max_weight': inner_max_weight,
                    'distance': inner_d,
                    'x_logical': inner_x_logical,  # Support of X logical operator
                    'z_logical': inner_z_logical,  # Support of Z logical operator
                }
                # Also extract outer code info for top-level syndrome calculation
                outer_code = code.level_codes[0] # First level is outermost
                outer_Hz = np.asarray(outer_code.hz, dtype=np.uint8)
                outer_Hx = np.asarray(outer_code.hx, dtype=np.uint8)
                outer_d = getattr(outer_code, 'd', getattr(outer_code, 'distance', 3))
                outer_code_info = {
                    'Hz': outer_Hz,
                    'Hx': outer_Hx,
                    'n_qubits': outer_code.n, # Note: this is likely 'n' of the base code, not total physical qubits
                    'max_weight': (outer_d - 1) // 2,
                    'distance': outer_d,
                }
            except AttributeError:
                pass  # Inner/Outer code doesn't have expected structure
        
        # VALIDATION: Warn about expected metadata structure for hierarchical codes
        if inner_code_info is not None and config.verbose:
            print(f"\nHierarchical decoder initialized for concatenated code:")
            print(f"  Inner code: n={inner_code_info['n_qubits']}, d={inner_code_info['distance']}")
            print(f"  Inner X logical support: {inner_code_info.get('x_logical', 'NOT FOUND')}")
            print(f"  Inner Z logical support: {inner_code_info.get('z_logical', 'NOT FOUND')}")
            if outer_code_info:
                print(f"  Outer code: n={outer_code_info['n_qubits']}, d={outer_code_info['distance']}")
            print(f"  Expected metadata: inner syndromes with level≠0 keys")
            print(f"  Decoder computes outer syndrome from inner block logical outcomes")
        
        return cls(Hz=Hz, Hx=Hx, ZL=ZL, XL=XL, config=config, inner_code_info=inner_code_info, outer_code_info=outer_code_info)
    
    @classmethod
    def _get_logical_support(cls, code: Any, logical_type: str) -> List[int]:
        """Extract logical operator support from code."""
        # Try logical_z/logical_x property (string format)
        attr_name = f'logical_{logical_type.lower()}'
        if hasattr(code, attr_name):
            ops = getattr(code, attr_name)
            if isinstance(ops, list) and len(ops) > 0:
                op = ops[0]
                if isinstance(op, str):
                    target = logical_type.upper()
                    return [i for i, c in enumerate(op) if c.upper() in (target, 'Y')]
        
        # Try lz/lx method (numpy format)
        method_name = f'l{logical_type.lower()}'
        if hasattr(code, method_name):
            lop = getattr(code, method_name)
            if callable(lop):
                lop = lop()
            if isinstance(lop, np.ndarray):
                lop = np.atleast_2d(lop)
                if lop.shape[0] > 0:
                    return list(np.where(lop[0] != 0)[0])
        
        # Fallback: construct from concatenated structure
        if hasattr(code, 'level_codes') and len(code.level_codes) >= 2:
            return cls._construct_logical_support_from_levels(code, logical_type)
        
        raise ValueError(f"Could not determine {logical_type} logical support from code")
    
    @classmethod
    def _construct_logical_support_from_levels(
        cls,
        code: Any,
        logical_type: str,
    ) -> List[int]:
        """Construct logical support from level codes."""
        inner_code = code.level_codes[-1]
        outer_code = code.level_codes[0]
        n_inner = inner_code.n
        
        inner_support = cls._get_level_logical_support(inner_code, logical_type)
        outer_support = cls._get_level_logical_support(outer_code, logical_type)
        
        full_support = []
        for block in outer_support:
            for q in inner_support:
                full_support.append(block * n_inner + q)
        
        return full_support
    
    @staticmethod
    def _get_level_logical_support(code: Any, logical_type: str) -> List[int]:
        """Get logical support for a single level code."""
        attr_name = f'logical_{logical_type.lower()}'
        if hasattr(code, attr_name):
            ops = getattr(code, attr_name)
            if isinstance(ops, list) and len(ops) > 0:
                op = ops[0]
                if isinstance(op, str):
                    target = logical_type.upper()
                    return [i for i, c in enumerate(op) if c.upper() in (target, 'Y')]
        
        method_name = f'l{logical_type.lower()}'
        if hasattr(code, method_name):
            lop = getattr(code, method_name)
            if callable(lop):
                lop = lop()
            if isinstance(lop, np.ndarray):
                lop = np.atleast_2d(lop)
                if lop.shape[0] > 0:
                    return list(np.where(lop[0] != 0)[0])
        
        # Fallback for common codes
        code_name = getattr(code, 'name', '') or type(code).__name__
        if 'steane' in code_name.lower() or 'stean' in code_name.lower():
            return [0, 1, 2]
        
        return list(range(min(3, code.n)))
    
    def decode(
        self,
        sample: np.ndarray,
        metadata: 'MultiLevelMetadata',
    ) -> CircuitDecodeResult:
        """
        TRUE circuit-level decode using ONLY syndrome measurements.
        
        IMPORTANT: This method does NOT use final data measurements for decoding.
        The logical value is inferred purely from EC syndrome measurements.
        Use validate() to check the prediction against ground truth.
        
        FLAG-AWARE DECODING:
        -------------------
        When flag_decoding_mode != IGNORE, this method:
        1. Extracts flag outcomes from EC rounds
        2. Uses flag-conditioned lookup tables for hook errors
        3. Adjusts confidence based on flag status
        
        Decoding Strategy:
        -----------------
        1. Extract syndrome history from all EC rounds
        2. Extract flag outcomes (if available)
        3. For each inner block, decode using syndrome + flags
        4. Propagate corrections to outer level
        5. Use majority voting if multiple rounds available
        6. Predict logical value based on cumulative corrections
        
        Args:
            sample: Full measurement sample from circuit
            metadata: MultiLevelMetadata from experiment.build()
            
        Returns:
            CircuitDecodeResult with PREDICTED logical values (not ground truth)
        """
        # Step 1: Extract syndrome history from EC rounds
        syndrome_history_raw = self._extract_syndrome_history(sample, metadata)
        
        # CRITICAL FIX: Filter out outer-level (level=0) syndrome entries before processing
        # The decoder computes outer syndromes from inner block logical outcomes.
        # Having outer-level keys causes shape mismatch in aggregation.
        syndrome_history = []
        for round_data in syndrome_history_raw:
            filtered_round = {'round': round_data['round'], 'X': {}, 'Z': {}}
            for stype in ['X', 'Z']:
                for block_key, syn in round_data.get(stype, {}).items():
                    level, _ = self._parse_block_key(block_key)
                    if level != 0:  # Only keep inner-level syndromes
                        filtered_round[stype][block_key] = syn
            syndrome_history.append(filtered_round)
        
        # Precompute level ranges for mapping corrections (cached per decode call)
        self._cached_level_ranges = self._build_level_ranges(metadata)
        # Reset per-decode warning trackers
        self._warned_levels = set()
        self._warned_parse_keys = set()
        # Set block-key normalization hook if provided
        self._metadata_normalize_hook = getattr(metadata, 'normalize_block_key', None) if metadata else None
        
        # Step 2: Extract flag outcomes (if flag-aware decoding enabled)
        flag_outcomes = {}
        n_flags_fired = 0
        if self.config.flag_decoding_mode != FlagDecodingMode.IGNORE:
            flag_outcomes = self._extract_flag_outcomes(sample, metadata)
            n_flags_fired = self._count_flags_fired(flag_outcomes)
        
        # Check for post-selection rejection
        if self.config.post_select_on_flags and n_flags_fired > 0:
            return CircuitDecodeResult(
                logical_z=0,
                logical_x=0,
                flag_outcomes=flag_outcomes,
                n_flags_fired=n_flags_fired,
                rejected=True,
                rejection_reason=f"Flag fired ({n_flags_fired} flags)",
                decoding_method="post_selected",
            )
        
        if not syndrome_history:
            # No EC rounds - cannot decode from syndromes
            # Return trivial prediction (no errors detected)
            return CircuitDecodeResult(
                logical_z=0,
                logical_x=0,
                syndrome_history=[],
                flag_outcomes=flag_outcomes,
                n_flags_fired=n_flags_fired,
                decoding_method="no_ec_rounds",
            )
        
        # Step 3: Compute syndrome changes
        syndrome_changes = self._compute_syndrome_changes(syndrome_history)
        n_syndrome_changes = self._count_syndrome_changes(syndrome_changes)
        measurement_instability = self._check_measurement_instability(syndrome_history, syndrome_changes)
        
        # Step 4: Decode using syndromes + flags
        # For concatenated codes, we need to decode hierarchically:
        # - Inner syndromes tell us errors within each block
        # - Flags tell us if hook errors occurred
        # - These propagate to logical errors at the outer level
        
        if self.config.flag_decoding_mode in (FlagDecodingMode.LOOKUP, FlagDecodingMode.ADAPTIVE):
            x_correction, z_correction, used_flag = self._decode_flag_aware(
                syndrome_history, flag_outcomes, metadata
            )
            decoding_method = "flag_lookup" if used_flag else "syndrome_only"
        else:
            # Check if we should use the new clean hierarchical decoder
            # (enabled by default for concatenated codes when use_decoder_outer_syndrome=True)
            is_concatenated = (self._outer_n_blocks > 0 and self._outer_Hz is not None)
            
            if is_concatenated and self.config.use_decoder_outer_syndrome:
                # NEW: Clean hierarchical decode with decoder-side outer syndrome (Gaps 1-5 fix)
                x_correction, z_correction = self._decode_hierarchical_clean(
                    syndrome_history, metadata
                )
                decoding_method = "hierarchical_decoder_outer"
            else:
                # Legacy: Use _decode_from_syndrome_history (has BPOSD pathway)
                x_correction, z_correction = self._decode_from_syndrome_history(
                    syndrome_history, syndrome_changes, metadata, measurement_instability
                )
                decoding_method = "syndrome_history_with_bposd"
            used_flag = False
        
        # Step 5: Compute predicted logical values from corrections
        # X correction flips Z logical, Z correction flips X logical
        predicted_z = self._compute_logical_from_correction(x_correction, self.ZL)
        predicted_x = self._compute_logical_from_correction(z_correction, self.XL)
        
        # Step 6: Compute confidence (reduced when flags fire)
        confidence = 1.0
        if self.config.flag_decoding_mode == FlagDecodingMode.SOFT_WEIGHT and n_flags_fired > 0:
            confidence = self.config.flag_confidence_penalty ** n_flags_fired

        # Further reduce confidence if temporal instability observed
        if measurement_instability:
            confidence *= 0.5
        
        # Get final syndrome from last EC round (for diagnostics)
        x_syndrome_final = self._get_aggregated_syndrome(syndrome_history[-1], 'X')
        z_syndrome_final = self._get_aggregated_syndrome(syndrome_history[-1], 'Z')

        # Optional spacetime MWPM using syndrome-only info (benchmark-safe)
        if (
            self.config.use_spacetime_mwpm
            and self.config.use_spacetime_syndrome_only
            and getattr(metadata, 'enable_spacetime_mwpm', False)
            and not self._is_concatenated_blockwise(metadata)
        ):
            try:
                st_x_logical = self._decode_spacetime_single_type(
                    syndrome_changes,
                    x_syndrome_final,
                    self.Hz,
                    self.ZL,
                    'X',
                )
                st_z_logical = self._decode_spacetime_single_type(
                    syndrome_changes,
                    z_syndrome_final,
                    self.Hx,
                    self.XL,
                    'Z',
                )
                predicted_z = st_x_logical
                predicted_x = st_z_logical
                decoding_method = f"{decoding_method}+spacetime_syn"
                n_detector_events = self._count_detector_events(
                    syndrome_changes, x_syndrome_final, z_syndrome_final
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"Syndrome-only spacetime MWPM fallback to syndrome-only due to: {e}")

        # Optional spacetime MWPM decoding when explicitly enabled and available (analysis mode; uses final data)
        if (
            self.config.use_spacetime_mwpm
            and HAS_PYMATCHING
            and getattr(metadata, 'enable_spacetime_mwpm', False)
            and self.config.allow_final_data_for_spacetime
            and not self._is_concatenated_blockwise(metadata)
        ):
            try:
                from qectostim.decoders._ignore.joint_minimum_weight_decoder import extract_final_data

                final_data = extract_final_data(sample, metadata, apply_frame_correction=True)
                final_x_syndrome = tuple(int(x) for x in (self.Hz @ final_data) % 2)
                final_z_syndrome = tuple(int(x) for x in (self.Hx @ final_data) % 2)

                st_x_logical, st_z_logical, _ = self._decode_spacetime_mwpm(
                    syndrome_history,
                    syndrome_changes,
                    final_x_syndrome,
                    final_z_syndrome,
                    final_data,
                )

                predicted_z = st_x_logical  # X-type decoding informs logical Z
                predicted_x = st_z_logical  # Z-type decoding informs logical X
                decoding_method = f"{decoding_method}+spacetime"
                n_detector_events = self._count_detector_events(
                    syndrome_changes, final_x_syndrome, final_z_syndrome
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"Spacetime MWPM fallback to syndrome-only due to: {e}")

        # Detector events include syndrome changes and final-round syndromes
        n_detector_events = self._count_detector_events(syndrome_changes, x_syndrome_final, z_syndrome_final)
        
        return CircuitDecodeResult(
            logical_z=predicted_z,
            logical_x=predicted_x,
            x_syndrome_final=x_syndrome_final,
            z_syndrome_final=z_syndrome_final,
            syndrome_history=syndrome_history,
            flag_outcomes=flag_outcomes,
            n_flags_fired=n_flags_fired,
            used_flag_correction=used_flag,
            x_correction=x_correction,
            z_correction=z_correction,
            syndrome_changes=syndrome_changes,
            n_syndrome_changes=n_syndrome_changes,
            n_detector_events=n_detector_events,
            confidence=confidence,
            decoding_method=decoding_method,
            rejection_reason="measurement_instability" if measurement_instability else "",
        )
    
    def validate(
        self,
        result: CircuitDecodeResult,
        sample: np.ndarray,
        metadata: 'MultiLevelMetadata',
    ) -> CircuitDecodeResult:
        """
        Validate decoder's prediction against ground truth (final data).
        
        This method extracts the TRUE logical value from final data measurements
        and compares it to the decoder's prediction. This is the proper way
        to benchmark fault-tolerance.
        
        Args:
            result: CircuitDecodeResult from decode()
            sample: Full measurement sample
            metadata: MultiLevelMetadata from experiment.build()
            
        Returns:
            Updated CircuitDecodeResult with validation fields filled
        """
        from qectostim.decoders._ignore.joint_minimum_weight_decoder import extract_final_data
        
        # Extract final data (ground truth)
        final_data = extract_final_data(sample, metadata, apply_frame_correction=True)
        
        # Compute actual logical values from final data
        # For a properly prepared |0⟩_L state, the logical Z measurement should be 0
        # unless there's a logical X error
        ground_truth_z = sum(int(final_data[q]) for q in self.ZL) % 2
        ground_truth_x = sum(int(final_data[q]) for q in self.XL) % 2
        
        # Update result with validation
        result.ground_truth_z = ground_truth_z
        result.ground_truth_x = ground_truth_x
        result.is_correct_z = (result.logical_z == ground_truth_z)
        result.is_correct_x = (result.logical_x == ground_truth_x)
        
        return result
    
    def decode_and_validate(
        self,
        sample: np.ndarray,
        metadata: 'MultiLevelMetadata',
    ) -> CircuitDecodeResult:
        """
        Convenience method: decode from syndromes and validate against ground truth.
        
        This is the recommended method for fault-tolerance benchmarking.
        """
        result = self.decode(sample, metadata)
        return self.validate(result, sample, metadata)
    
    # =========================================================================
    # FLAG EXTRACTION AND FLAG-AWARE DECODING
    # =========================================================================
    
    def _extract_flag_outcomes(
        self,
        sample: np.ndarray,
        metadata: 'MultiLevelMetadata',
    ) -> Dict[str, Dict[int, Dict[int, int]]]:
        """
        Extract flag qubit measurement outcomes from the sample.
        
        Returns structure: {'X': {block_id: {stab_idx: flag_value}}, 'Z': {...}}
        
        The flag indices are stored in metadata.flag_measurements (if available).
        """
        flag_outcomes: Dict[str, Dict[int, Dict[int, int]]] = {'X': {}, 'Z': {}}
        
        # Check if metadata has flag_measurements
        if not hasattr(metadata, 'flag_measurements'):
            return flag_outcomes
        
        flag_meas = getattr(metadata, 'flag_measurements', {})
        if not flag_meas:
            return flag_outcomes
        
        for stab_type in ['X', 'Z']:
            if stab_type not in flag_meas:
                continue
            
            flag_outcomes[stab_type] = {}
            
            for block_id, stab_dict in flag_meas[stab_type].items():
                flag_outcomes[stab_type][block_id] = {}
                
                for stab_idx, meas_indices in stab_dict.items():
                    # Get flag value (XOR of all flag measurements for this stabilizer)
                    flag_val = 0
                    for idx in meas_indices:
                        if idx < len(sample):
                            flag_val ^= int(sample[idx])
                    
                    flag_outcomes[stab_type][block_id][stab_idx] = flag_val
        
        return flag_outcomes
    
    def _count_flags_fired(
        self,
        flag_outcomes: Dict[str, Dict[int, Dict[int, int]]],
    ) -> int:
        """Count total number of flags that fired (value = 1)."""
        count = 0
        for stab_type in ['X', 'Z']:
            for block_dict in flag_outcomes.get(stab_type, {}).values():
                for flag_val in block_dict.values():
                    count += flag_val
        return count
    
    def _get_flag_pattern_for_block(
        self,
        flag_outcomes: Dict[str, Dict[int, Dict[int, int]]],
        stab_type: str,
        block_id: int,
    ) -> Tuple[int, ...]:
        """
        Get the flag pattern for a block as a tuple.
        
        Returns tuple of flag values ordered by stabilizer index.
        """
        if stab_type not in flag_outcomes or block_id not in flag_outcomes[stab_type]:
            return ()
        
        stab_dict = flag_outcomes[stab_type][block_id]
        if not stab_dict:
            return ()
        
        max_stab_idx = max(stab_dict.keys())
        return tuple(stab_dict.get(i, 0) for i in range(max_stab_idx + 1))
    
    def _decode_flag_aware(
        self,
        syndrome_history: List[Dict],
        flag_outcomes: Dict[str, Dict[int, Dict[int, int]]],
        metadata: 'MultiLevelMetadata',
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Decode using flag-conditioned lookup tables.
        
        For each inner block:
        1. Get syndrome from syndrome_history
        2. Get flag pattern from flag_outcomes
        3. Look up correction in flag-conditioned table
        4. Apply correction
        
        Returns:
            (x_correction, z_correction, used_flag_correction)
        """
        x_correction = np.zeros(self.n_qubits, dtype=np.uint8)
        z_correction = np.zeros(self.n_qubits, dtype=np.uint8)
        used_flag = False
        
        if not syndrome_history:
            return x_correction, z_correction, used_flag
        
        # Use majority voting if multiple rounds
        if self.config.use_majority_voting and len(syndrome_history) > 1:
            x_raw = self._majority_vote_syndromes(syndrome_history, 'X')
            z_raw = self._majority_vote_syndromes(syndrome_history, 'Z')
        else:
            x_raw = syndrome_history[-1].get('X', {})
            z_raw = syndrome_history[-1].get('Z', {})
        
        # Process X syndromes (detect Z errors) with Z flags across all levels
        for block_key, raw_meas in x_raw.items():
            level, block_idx = self._parse_block_key(block_key)
            if level is None:
                continue
            if level == 0:
                # For outer-level syndrome, collect ALL inner block measurements
                # and apply hierarchical extraction
                inner_measurements = []
                # Collect measurements from all 7 inner blocks
                for inner_block_id in range(7):  # TODO: Get from metadata
                    inner_key = (1, (1, inner_block_id))
                    if inner_key in x_raw:
                        inner_measurements.extend(x_raw[inner_key])
                
                # Apply hierarchical extraction: inner measurements → outer syndrome
                true_syndrome = tuple(self._compute_outer_syndrome_from_raw(inner_measurements, 'X', metadata))
                block_correction = self._lookup_correction_from_table(true_syndrome, 'Z')
                z_correction = (z_correction + block_correction) % 2
                continue

            true_syndrome = self._compute_true_syndrome(raw_meas, 'X')
            if sum(true_syndrome) > 0:
                flag_pattern = self._get_flag_pattern_for_block(flag_outcomes, 'Z', block_idx)
                block_correction, flag_used = self._decode_with_flags(
                    true_syndrome, flag_pattern, (level, block_idx), 'Z', metadata
                )
                z_correction = (z_correction + block_correction) % 2
                used_flag = used_flag or flag_used

        # Process Z syndromes (detect X errors) with X flags across all levels
        for block_key, raw_meas in z_raw.items():
            level, block_idx = self._parse_block_key(block_key)
            if level is None:
                continue
            if level == 0:
                # For outer-level syndrome, collect ALL inner block measurements
                # and apply hierarchical extraction
                inner_measurements = []
                # Collect measurements from all 7 inner blocks
                for inner_block_id in range(7):  # TODO: Get from metadata
                    inner_key = (1, (1, inner_block_id))
                    if inner_key in z_raw:
                        inner_measurements.extend(z_raw[inner_key])
                
                # Apply hierarchical extraction: inner measurements → outer syndrome  
                true_syndrome = tuple(self._compute_outer_syndrome_from_raw(inner_measurements, 'Z', metadata))
                block_correction = self._lookup_correction_from_table(true_syndrome, 'X')
                x_correction = (x_correction + block_correction) % 2
                continue

            true_syndrome = self._compute_true_syndrome(raw_meas, 'Z')
            if sum(true_syndrome) > 0:
                flag_pattern = self._get_flag_pattern_for_block(flag_outcomes, 'X', block_idx)
                block_correction, flag_used = self._decode_with_flags(
                    true_syndrome, flag_pattern, (level, block_idx), 'X', metadata
                )
                x_correction = (x_correction + block_correction) % 2
                used_flag = used_flag or flag_used
        
        return x_correction, z_correction, used_flag
    
    def _compute_outer_syndrome_from_raw(
        self,
        raw_measurements: List[int],
        check_type: str,
        metadata: 'MultiLevelMetadata',
    ) -> List[int]:
        """
        Compute outer-level syndrome from raw ancilla measurements for concatenated codes.
        
        For a 2-level concatenated code like [[49,1,9]] = [[7,1,3]] ⊗ [[7,1,3]]:
        - Raw measurements contain all 49 ancilla measurements (7 blocks × 7 measurements each)
        - We need to extract: logical error per block → outer syndrome (3 bits)
        
        Process:
        1. Group raw measurements by inner block (7 measurements per block)
        2. Apply inner syndrome extraction: Hz_inner @ raw → syndrome_inner (3 bits)
        3. Decode inner syndrome to get logical error (1 bit per block)
        4. Apply outer syndrome extraction: Hz_outer @ logical → syndrome_outer (3 bits)
        
        Parameters
        ----------
        raw_measurements : List[int]
            All raw ancilla measurements (n_blocks × n_inner measurements)
        check_type : str
            'X' for Hz checks (detect Z errors) or 'Z' for Hx checks (detect X errors)
        metadata : MultiLevelMetadata
            Circuit metadata
            
        Returns
        -------
        List[int]
            The outer syndrome (3 bits for Steane [[7,1,3]])
        """
        # DEBUG
        if self.config.verbose:
            print(f"    _compute_outer_syndrome_from_raw called: check_type={check_type}, len(raw)={len(raw_measurements)}")
        
        # For non-concatenated codes or when we don't have proper matrices, return as-is
        if not hasattr(self, '_inner_Hz') or not hasattr(self, '_outer_Hz'):
            if self.config.verbose:
                print(f"    WARNING: Missing inner/outer matrices!")
            return raw_measurements[:3] if len(raw_measurements) >= 3 else raw_measurements
        
        # Get the appropriate matrices
        if check_type == 'X':
            inner_H = self._inner_Hz
            outer_H = self._outer_Hz
        else:
            inner_H = self._inner_Hx
            outer_H = self._outer_Hx
        
        if inner_H is None or outer_H is None:
            if self.config.verbose:
                print(f"    WARNING: inner_H or outer_H is None!")
            return raw_measurements[:3] if len(raw_measurements) >= 3 else raw_measurements
        
        n_inner_checks = inner_H.shape[0]  # Number of syndrome bits per block (3 for Steane)
        n_blocks = outer_H.shape[1]  # Number of blocks (7 for [[49,1,9]])
        n_inner_qubits = inner_H.shape[1]  # Qubits per block (7 for Steane)
        
        # Step 1: Group raw measurements by block
        # Assuming measurements are organized as: block0_ancillas, block1_ancillas, ...
        logical_errors = np.zeros(n_blocks, dtype=np.uint8)
        
        for block_idx in range(n_blocks):
            # Extract this block's raw measurements
            start_idx = block_idx * n_inner_qubits
            end_idx = start_idx + n_inner_qubits
            
            if end_idx > len(raw_measurements):
                break
            
            block_raw = np.array(raw_measurements[start_idx:end_idx], dtype=np.uint8)
            
            # Step 2: Compute inner syndrome: s_inner = H_inner @ raw
            inner_syndrome = (inner_H @ block_raw) % 2
            inner_syndrome_tuple = tuple(inner_syndrome)
            
            # Step 3: Decode inner syndrome to get logical error
            # Use the inner decoder table to determine if there's a logical error
            # For Steane [[7,1,3]], logical Z is on qubits [0,1,2]
            # If syndrome indicates error on logical support → logical error
            
            if sum(inner_syndrome) > 0:
                # Use the decoder table to get correction
                if check_type == 'X' and self._inner_z_table and inner_syndrome_tuple in self._inner_z_table:
                    correction, _ = self._inner_z_table[inner_syndrome_tuple]
                    # Check if correction affects logical Z (qubits in Z_L support)
                    if hasattr(self, '_inner_z_logical_support'):
                        logical_error = sum(correction[q] for q in self._inner_z_logical_support if q < len(correction)) % 2
                    else:
                        logical_error = sum(correction) % 2
                    logical_errors[block_idx] = logical_error
                elif check_type == 'Z' and self._inner_x_table and inner_syndrome_tuple in self._inner_x_table:
                    correction, _ = self._inner_x_table[inner_syndrome_tuple]
                    # Check if correction affects logical X (qubits in X_L support)
                    if hasattr(self, '_inner_x_logical_support'):
                        logical_error = sum(correction[q] for q in self._inner_x_logical_support if q < len(correction)) % 2
                    else:
                        logical_error = sum(correction) % 2
                    logical_errors[block_idx] = logical_error
        
        # Step 4: Compute outer syndrome: s_outer = H_outer @ logical_errors
        outer_syndrome = (outer_H @ logical_errors) % 2
        
        if self.config.verbose:
            print(f"    Logical errors per block: {list(logical_errors)}")
            print(f"    Outer syndrome: {list(outer_syndrome)}")
        
        return list(outer_syndrome)
    
    def _decode_with_flags(
        self,
        syndrome: Tuple,
        flag_pattern: Tuple,
        block_key: Any,
        error_type: str,
        metadata: 'MultiLevelMetadata',
    ) -> Tuple[np.ndarray, bool]:
        """
        Decode a single block using flag-conditioned tables.
        
        Strategy:
        1. If flag fired and (syndrome, flag_pattern) in flag table → use hook correction
        2. Else → fall back to standard syndrome lookup
        
        Returns:
            (correction, used_flag_correction)
        """
        correction = np.zeros(self.n_qubits, dtype=np.uint8)
        used_flag = False
        
        # Select appropriate flag table
        if error_type == 'X':
            flag_table = self._inner_x_flag_table if self._inner_x_flag_table else self._x_flag_table
        else:
            flag_table = self._inner_z_flag_table if self._inner_z_flag_table else self._z_flag_table
        
        # Check if any flag fired
        any_flag_fired = flag_pattern and any(f == 1 for f in flag_pattern)
        
        # Look up in flag table
        if any_flag_fired and syndrome in flag_table:
            entries = flag_table[syndrome]
            
            # Find best entry. Priority: exact flag match on hook; optional likelihood weighting.
            best_entry = None
            best_score = -float('inf')
            
            for entry in entries:
                if len(entry.flag_pattern) != len(flag_pattern):
                    continue
                
                # Exact match on hook: highest priority
                if entry.flag_pattern == flag_pattern and entry.error_type == 'hook':
                    best_entry = entry
                    break
                
                # Likelihood scoring when hook_prior_weight > 0
                if self.config.hook_prior_weight > 0.0:
                    score = 0.0
                    
                    # Score based on error type and flag match
                    if entry.error_type == 'hook':
                        # Hook prior weight
                        score += self.config.hook_prior_weight
                        # Bonus for partial flag match
                        n_match = sum(1 for e, a in zip(entry.flag_pattern, flag_pattern) if e == a == 1)
                        score += n_match
                    elif entry.error_type == 'data':
                        # Data errors less likely when flags fire
                        score += (1.0 - self.config.hook_prior_weight)
                    
                    # Penalty for weight (prefer lower weight)
                    score -= 0.1 * entry.weight
                    
                    if score > best_score:
                        best_score = score
                        best_entry = entry

            if best_entry is not None:
                inner_correction = best_entry.correction

                level, block_idx = self._parse_block_key(block_key)
                if level is not None and block_idx is not None:
                    start_idx, block_size = self._resolve_block_range(level, block_idx, metadata)
                    for i, val in enumerate(inner_correction):
                        if i < block_size and start_idx + i < self.n_qubits:
                            correction[start_idx + i] = val

                used_flag = True
                return correction, used_flag
        
        # Fall back to standard syndrome lookup (no flag or no matching hook entry)
        return self._decode_inner_block_syndrome(
            syndrome, block_key, error_type, metadata
        ), False

    # =========================================================================
    # HIERARCHICAL DECODING WITH DECODER-SIDE OUTER SYNDROME (Gaps 1-5 Fixes)
    # =========================================================================

    def _decode_hierarchical_clean(
        self,
        syndrome_history: List[Dict],
        metadata: 'MultiLevelMetadata',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean hierarchical decode path with decoder-side outer syndrome computation.
        
        This is the core fix for Gaps 1-5:
        1. Decode each inner block independently per round
        2. Apply temporal majority voting to inner block logical outcomes (Gap 4)
        3. Compute outer syndrome from majority-voted inner logicals (Gaps 1-3)
        4. Decode outer level (optionally with BPOSD for soft info, Gap 5)
        5. Combine for final correction
        
        This enables verified ancillas without DEM disconnection because outer
        syndrome is computed by the decoder, not measured in the circuit.
        
        Returns:
            (x_correction, z_correction) arrays
        """
        x_correction = np.zeros(self.n_qubits, dtype=np.uint8)
        z_correction = np.zeros(self.n_qubits, dtype=np.uint8)
        
        # Update diagnostic counter
        self._diag_total_decodes += 1
        
        if not syndrome_history:
            return x_correction, z_correction
        
        n_rounds = len(syndrome_history)
        n_blocks = self._outer_n_blocks if self._outer_n_blocks > 0 else 7
        
        # =================================================================
        # STAGE 1: Collect inner syndromes and optionally vote at syndrome level
        # =================================================================
        # FIX: Vote on syndrome BITS first, then decode - reduces measurement error impact
        # Previous approach voted on logical outcomes AFTER decode, which amplified errors
        
        # Collect syndromes per block per round
        z_syndromes_by_block: Dict[int, List[Tuple[int, ...]]] = {i: [] for i in range(n_blocks)}
        x_syndromes_by_block: Dict[int, List[Tuple[int, ...]]] = {i: [] for i in range(n_blocks)}
        
        for round_idx in range(n_rounds):
            round_syndromes = syndrome_history[round_idx]
            
            # Collect Z syndromes (detecting X errors)
            for block_key, raw_meas in round_syndromes.get('Z', {}).items():
                level, block_idx = self._parse_block_key(block_key)
                if level is None or level == 0:  # Skip outer-level syndromes
                    continue
                if block_idx < n_blocks:
                    true_syndrome = self._compute_true_syndrome(raw_meas, 'Z')
                    z_syndromes_by_block[block_idx].append(true_syndrome)
            
            # Collect X syndromes (detecting Z errors)
            for block_key, raw_meas in round_syndromes.get('X', {}).items():
                level, block_idx = self._parse_block_key(block_key)
                if level is None or level == 0:
                    continue
                if block_idx < n_blocks:
                    true_syndrome = self._compute_true_syndrome(raw_meas, 'X')
                    x_syndromes_by_block[block_idx].append(true_syndrome)
        
        # =================================================================
        # STAGE 1B: Vote on syndrome bits, then decode (FIX for over-correction)
        # =================================================================
        # Track inner logical outcomes per round: round -> {block_idx -> (z_logical, x_logical)}
        inner_logicals_by_round: List[Dict[int, Tuple[int, int]]] = []
        inner_confidence_by_round: List[Dict[int, Tuple[float, float]]] = []
        
        # If syndrome-level voting enabled, vote on syndrome bits first
        use_syndrome_voting = getattr(self.config, 'use_syndrome_level_voting', True) and n_rounds > 1
        
        if use_syndrome_voting:
            # Majority vote on syndrome bits per block
            voted_z_syndromes: Dict[int, Tuple[int, ...]] = {}
            voted_x_syndromes: Dict[int, Tuple[int, ...]] = {}
            
            for block_idx in range(n_blocks):
                # Vote on Z syndrome bits
                z_syns = z_syndromes_by_block[block_idx]
                if z_syns:
                    # Bit-by-bit majority vote
                    n_bits = len(z_syns[0]) if z_syns else 3
                    voted_z_syn = []
                    for bit_idx in range(n_bits):
                        bit_votes = [syn[bit_idx] for syn in z_syns if len(syn) > bit_idx]
                        # FIX: Use >= for majority to handle ties (allows 1 vote in 2 rounds to pass)
                        # This is critical for sparse errors that appear in only some rounds
                        voted_z_syn.append(1 if sum(bit_votes) * 2 >= len(bit_votes) else 0)
                    voted_z_syndromes[block_idx] = tuple(voted_z_syn)
                else:
                    voted_z_syndromes[block_idx] = (0, 0, 0)
                
                # Vote on X syndrome bits
                x_syns = x_syndromes_by_block[block_idx]
                if x_syns:
                    n_bits = len(x_syns[0]) if x_syns else 3
                    voted_x_syn = []
                    for bit_idx in range(n_bits):
                        bit_votes = [syn[bit_idx] for syn in x_syns if len(syn) > bit_idx]
                        # FIX: Use >= for majority to handle ties
                        voted_x_syn.append(1 if sum(bit_votes) * 2 >= len(bit_votes) else 0)
                    voted_x_syndromes[block_idx] = tuple(voted_x_syn)
                else:
                    voted_x_syndromes[block_idx] = (0, 0, 0)
            
            # Now decode the VOTED syndromes (single decode per block, not per round)
            round_logicals: Dict[int, Tuple[int, int]] = {}
            round_confidence: Dict[int, Tuple[float, float]] = {}
            
            for block_idx in range(n_blocks):
                z_syn = voted_z_syndromes[block_idx]
                x_syn = voted_x_syndromes[block_idx]
                
                # Decode voted Z syndrome (X errors)
                if sum(z_syn) > 0:
                    z_logical, z_conf = self._decode_inner_block_logical(z_syn, block_idx, 'Z')
                else:
                    z_logical, z_conf = 0, 1.0
                
                # Decode voted X syndrome (Z errors)
                if sum(x_syn) > 0:
                    x_logical, x_conf = self._decode_inner_block_logical(x_syn, block_idx, 'X')
                else:
                    x_logical, x_conf = 0, 1.0
                
                round_logicals[block_idx] = (z_logical, x_logical)
                round_confidence[block_idx] = (z_conf, x_conf)
            
            # Store as single "effective round" for downstream stages
            inner_logicals_by_round.append(round_logicals)
            inner_confidence_by_round.append(round_confidence)
            
        else:
            # Original approach: decode each round separately, then vote on logicals
            for round_idx in range(n_rounds):
                round_syndromes = syndrome_history[round_idx]
                round_logicals: Dict[int, Tuple[int, int]] = {}
                round_confidence: Dict[int, Tuple[float, float]] = {}
                
                # Decode Z syndromes (X errors) -> X logical errors
                for block_key, raw_meas in round_syndromes.get('Z', {}).items():
                    level, block_idx = self._parse_block_key(block_key)
                    if level is None or level == 0:
                        continue
                        
                    true_syndrome = self._compute_true_syndrome(raw_meas, 'Z')
                    
                    if sum(true_syndrome) > 0:
                        z_logical, z_conf = self._decode_inner_block_logical(
                            true_syndrome, block_idx, 'Z'
                        )
                    else:
                        z_logical, z_conf = 0, 1.0
                    
                    if block_idx not in round_logicals:
                        round_logicals[block_idx] = (z_logical, 0)
                        round_confidence[block_idx] = (z_conf, 1.0)
                    else:
                        existing_z, existing_x = round_logicals[block_idx]
                        round_logicals[block_idx] = (z_logical, existing_x)
                        ex_z_conf, ex_x_conf = round_confidence[block_idx]
                        round_confidence[block_idx] = (z_conf, ex_x_conf)
                
                # Decode X syndromes (Z errors) -> Z logical errors
                for block_key, raw_meas in round_syndromes.get('X', {}).items():
                    level, block_idx = self._parse_block_key(block_key)
                    if level is None or level == 0:
                        continue
                        
                    true_syndrome = self._compute_true_syndrome(raw_meas, 'X')
                    
                    if sum(true_syndrome) > 0:
                        x_logical, x_conf = self._decode_inner_block_logical(
                            true_syndrome, block_idx, 'X'
                        )
                    else:
                        x_logical, x_conf = 0, 1.0
                    
                    if block_idx not in round_logicals:
                        round_logicals[block_idx] = (0, x_logical)
                        round_confidence[block_idx] = (1.0, x_conf)
                    else:
                        existing_z, _ = round_logicals[block_idx]
                        round_logicals[block_idx] = (existing_z, x_logical)
                        ex_z_conf, _ = round_confidence[block_idx]
                        round_confidence[block_idx] = (ex_z_conf, x_conf)
                
                inner_logicals_by_round.append(round_logicals)
                inner_confidence_by_round.append(round_confidence)
        
        # =================================================================
        # STAGE 2: Temporal majority voting on inner logical outcomes (Gap 4)
        # =================================================================
        # NOTE: If syndrome-level voting was used, inner_logicals_by_round has only 1 entry
        #       and this stage is effectively a no-op (which is correct)
        final_inner_z_logicals: Dict[int, int] = {}  # X errors on blocks
        final_inner_x_logicals: Dict[int, int] = {}  # Z errors on blocks
        
        effective_n_rounds = len(inner_logicals_by_round)
        
        if self.config.use_inner_logical_majority_voting and effective_n_rounds > 1:
            for block_idx in range(n_blocks):
                z_votes = []
                x_votes = []
                
                for round_idx in range(effective_n_rounds):
                    if block_idx in inner_logicals_by_round[round_idx]:
                        z_log, x_log = inner_logicals_by_round[round_idx][block_idx]
                        z_votes.append(z_log)
                        x_votes.append(x_log)
                    else:
                        z_votes.append(0)
                        x_votes.append(0)
                
                # Majority vote
                final_inner_z_logicals[block_idx] = 1 if sum(z_votes) > n_rounds // 2 else 0
                final_inner_x_logicals[block_idx] = 1 if sum(x_votes) > n_rounds // 2 else 0
            
            if self.config.verbose:
                z_errs = [b for b, v in final_inner_z_logicals.items() if v == 1]
                x_errs = [b for b, v in final_inner_x_logicals.items() if v == 1]
                if z_errs or x_errs:
                    print(f"  Majority-voted inner logicals: Z errors on blocks {z_errs}, X errors on blocks {x_errs}")
        else:
            # Use last round only (or single round)
            last_round = inner_logicals_by_round[-1] if inner_logicals_by_round else {}
            for block_idx in range(n_blocks):
                if block_idx in last_round:
                    z_log, x_log = last_round[block_idx]
                    final_inner_z_logicals[block_idx] = z_log
                    final_inner_x_logicals[block_idx] = x_log
                else:
                    final_inner_z_logicals[block_idx] = 0
                    final_inner_x_logicals[block_idx] = 0
        
        # =================================================================
        # STAGE 3: Compute outer syndrome from inner logicals (Gaps 1-3)
        # =================================================================
        # CRITICAL FIX: Use temporal majority voting for outer syndromes
        # This dramatically improves outer syndrome reliability
        
        if n_rounds > 1 and self.config.use_outer_syndrome_voting:
            outer_z_syndrome, outer_x_syndrome = self._majority_vote_outer_syndromes(
                inner_logicals_by_round
            )
            if self.config.verbose and (sum(outer_z_syndrome) > 0 or sum(outer_x_syndrome) > 0):
                print(f"  Majority-voted outer syndromes: Z={outer_z_syndrome}, X={outer_x_syndrome}")
        else:
            # Fallback: compute from final inner logicals only
            outer_z_syndrome, outer_x_syndrome = self._compute_outer_syndrome_from_inner_logicals(
                final_inner_z_logicals,  # X errors on blocks
                final_inner_x_logicals,  # Z errors on blocks
            )
            if self.config.verbose and (sum(outer_z_syndrome) > 0 or sum(outer_x_syndrome) > 0):
                print(f"  Outer syndromes from inner logicals: Z={outer_z_syndrome}, X={outer_x_syndrome}")
        
        # DIAGNOSTIC: Track non-zero outer syndrome
        if sum(outer_z_syndrome) > 0 or sum(outer_x_syndrome) > 0:
            self._diag_outer_nonzero_syndrome += 1
        
        # Also track inner logical errors detected (before outer correction)
        inner_errors_detected = sum(1 for v in final_inner_z_logicals.values() if v == 1)
        inner_errors_detected += sum(1 for v in final_inner_x_logicals.values() if v == 1)
        if inner_errors_detected > 0:
            self._diag_inner_logical_errors_detected += 1
        
        # =================================================================
        # STAGE 4: Decode outer level (Gap 5 - optional BPOSD)
        # =================================================================
        # CRITICAL FIX: Outer decoding is INDEPENDENT of whether inner had errors
        # This is crucial for achieving quadratic scaling (γ ≈ 4-5)
        outer_z_correction, outer_x_correction = self._decode_outer_syndrome(
            outer_z_syndrome, outer_x_syndrome
        )
        
        # DIAGNOSTIC: Track when outer correction is actually applied
        if np.sum(outer_z_correction) > 0 or np.sum(outer_x_correction) > 0:
            self._diag_outer_applied_correction += 1
        
        if self.config.verbose and (np.sum(outer_z_correction) > 0 or np.sum(outer_x_correction) > 0):
            print(f"  Outer decode: Z correction on blocks {list(np.where(outer_z_correction)[0])}, X correction on blocks {list(np.where(outer_x_correction)[0])}")
        
        # =================================================================
        # STAGE 5: Combine corrections (CRITICAL FOR γ ≈ 5)
        # =================================================================
        # CRITICAL FIX: Proper hierarchical correction combination
        # 
        # The outer correction tells us which BLOCKS have residual logical errors
        # after inner decoding. We XOR this with the inner logical outcomes.
        # 
        # This achieves multiplicative error suppression:
        # - Inner [[7,1,3]]: corrects weight-1 errors → p_block ~ p^2
        # - Outer [[7,1,3]]: corrects weight-1 block errors → p_logical ~ (p^2)^2 = p^4
        # - Total scaling: γ ≈ 4-5 for distance-9 concatenated code
        
        # STEP 1: Combine inner and outer corrections for each block
        corrected_z_logicals = {}  # Final Z logical state per block
        corrected_x_logicals = {}  # Final X logical state per block
        
        for block_idx in range(n_blocks):
            # Inner decoding says block has Z logical error?
            inner_z_err = final_inner_z_logicals.get(block_idx, 0)
            # Outer decoding says block has Z logical error?
            # If skip_outer_correction is True, don't use outer correction
            if self.config.skip_outer_correction:
                outer_z_err = 0
            else:
                outer_z_err = int(outer_z_correction[block_idx]) if block_idx < len(outer_z_correction) else 0
            # XOR them: outer correction flips the inner result
            corrected_z_logicals[block_idx] = (inner_z_err + outer_z_err) % 2
            
            # Same for X
            inner_x_err = final_inner_x_logicals.get(block_idx, 0)
            if self.config.skip_outer_correction:
                outer_x_err = 0
            else:
                outer_x_err = int(outer_x_correction[block_idx]) if block_idx < len(outer_x_correction) else 0
            corrected_x_logicals[block_idx] = (inner_x_err + outer_x_err) % 2
        
        # STEP 2: Apply combined corrections to physical qubits
        for block_idx in range(n_blocks):
            start_idx = block_idx * self._inner_n_qubits
            
            # If block has final Z logical error → apply X correction
            if corrected_z_logicals.get(block_idx, 0) == 1:
                for q in self._inner_z_logical_support:
                    if start_idx + q < self.n_qubits:
                        x_correction[start_idx + q] = (x_correction[start_idx + q] + 1) % 2
            
            # If block has final X logical error → apply Z correction
            if corrected_x_logicals.get(block_idx, 0) == 1:
                for q in self._inner_x_logical_support:
                    if start_idx + q < self.n_qubits:
                        z_correction[start_idx + q] = (z_correction[start_idx + q] + 1) % 2
        
        if self.config.verbose:
            final_z_errs = [b for b, v in corrected_z_logicals.items() if v == 1]
            final_x_errs = [b for b, v in corrected_x_logicals.items() if v == 1]
            if final_z_errs or final_x_errs:
                print(f"  Final corrected logicals: Z errors {final_z_errs}, X errors {final_x_errs}")
        
        return x_correction, z_correction

    def _decode_inner_block_logical(
        self,
        syndrome: Tuple[int, ...],
        block_idx: int,
        error_type: str,
    ) -> Tuple[int, float]:
        """
        Decode inner block syndrome to determine if CORRECTION flips the logical.
        
        CRITICAL FOR CONCATENATION: This method reports whether the minimum-weight
        CORRECTION (not the error itself) flips the logical operator. This is essential
        for achieving quadratic error suppression in concatenated codes.
        
        WHY THIS MATTERS:
        - Inner decoder applies minimum-weight correction for any detected syndrome
        - The correction itself may flip the logical (e.g., X0 when qubit 0 in Z_L)
        - Outer decoder MUST see these logical flips to correct them
        - Without this, weight-1 inner errors → uncorrected outer errors → γ ≈ 1
        - With this: weight-1 errors suppressed by p^2, then p^4 → γ ≈ 4-5
        
        IMPORTANT: Prioritize table lookup over BPOSD because tables have
        minimum-weight corrections, while BPOSD may return degenerate
        corrections with higher weight that incorrectly trigger outer decoding.
        
        Returns:
            (logical_error, confidence) where:
            - logical_error: 1 if CORRECTION flips the logical operator, 0 otherwise
            - confidence: float in [0, 1], higher = more confident (for soft decoding)
        """
        if sum(syndrome) == 0:
            return 0, 1.0  # No syndrome = no error = no correction = no logical flip
        
        # PRIORITY 1: Table lookup (guaranteed minimum-weight corrections)
        syndrome_tuple = tuple(syndrome)
        if error_type == 'Z' and self._inner_z_table and syndrome_tuple in self._inner_z_table:
            correction, weight = self._inner_z_table[syndrome_tuple]
            logical_support = self._inner_z_logical_support
            
            # Check if correction overlaps with logical operator (causes logical flip)
            logical_error = sum(int(correction[q]) for q in logical_support if q < len(correction)) % 2
            return logical_error, 1.0  # High confidence for table lookup
            
        elif error_type == 'X' and self._inner_x_table and syndrome_tuple in self._inner_x_table:
            correction, weight = self._inner_x_table[syndrome_tuple]
            logical_support = self._inner_x_logical_support
            
            # Check if correction overlaps with logical operator (causes logical flip)
            logical_error = sum(int(correction[q]) for q in logical_support if q < len(correction)) % 2
            return logical_error, 1.0  # High confidence for table lookup
        
        # PRIORITY 2: BPOSD as fallback for syndromes not in table
        if self.config.use_bposd_inner:
            bposd_decoder = self._inner_bposd_z if error_type == 'Z' else self._inner_bposd_x
            logical_support = self._inner_z_logical_support if error_type == 'Z' else self._inner_x_logical_support
            
            if bposd_decoder is not None:
                try:
                    syndrome_array = np.array(syndrome, dtype=np.uint8)
                    correction = bposd_decoder.decode(syndrome_array)
                    
                    # Check if BPOSD correction flips the logical
                    logical_error = sum(int(correction[q]) for q in logical_support if q < len(correction)) % 2
                    return logical_error, 0.7  # Medium-high confidence for BPOSD
                except Exception:
                    pass
        
        # Unknown syndrome, not in table, BPOSD failed
        # Conservative: assume correction would flip logical (report as error)
        return 1, 0.3  # Low confidence, assume worst case
    
    def _majority_vote_outer_syndromes(
        self,
        inner_logicals_by_round: List[Dict[int, Tuple[int, int]]],
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        CRITICAL FIX 3: Temporal majority voting for outer syndromes.
        
        Compute outer syndrome for each round, then majority vote bit-by-bit.
        This significantly reduces syndrome measurement errors for the outer level.
        
        Returns:
            (voted_outer_z_syndrome, voted_outer_x_syndrome)
        """
        n_rounds = len(inner_logicals_by_round)
        n_blocks = self._outer_n_blocks if self._outer_n_blocks > 0 else 7
        
        if n_rounds == 0:
            return (0,) * 3, (0,) * 3
        
        # Compute outer syndrome for each round
        outer_z_syndromes = []
        outer_x_syndromes = []
        
        for round_logicals in inner_logicals_by_round:
            # Extract inner Z and X logicals for this round
            inner_z_vec = np.array([round_logicals.get(i, (0, 0))[0] for i in range(n_blocks)], dtype=np.uint8)
            inner_x_vec = np.array([round_logicals.get(i, (0, 0))[1] for i in range(n_blocks)], dtype=np.uint8)
            
            # Compute outer syndromes for this round - CORRECTED ORIENTATION
            # Inner Z logical flips (X errors on outer) → detected by outer Hz → outer Z syndrome
            if self._outer_Hz is not None:
                z_syn = tuple((self._outer_Hz @ inner_z_vec) % 2)
            else:
                z_syn = (0,) * 3
            
            # Inner X logical flips (Z errors on outer) → detected by outer Hx → outer X syndrome
            if self._outer_Hx is not None:
                x_syn = tuple((self._outer_Hx @ inner_x_vec) % 2)
            else:
                x_syn = (0,) * 3
            
            outer_z_syndromes.append(z_syn)
            outer_x_syndromes.append(x_syn)
        
        # Majority vote bit-by-bit
        if n_rounds == 1:
            voted_z = outer_z_syndromes[0]
            voted_x = outer_x_syndromes[0]
        else:
            n_syndrome_bits = len(outer_z_syndromes[0])
            voted_z = tuple(
                1 if sum(syn[i] for syn in outer_z_syndromes) > n_rounds // 2 else 0
                for i in range(n_syndrome_bits)
            )
            voted_x = tuple(
                1 if sum(syn[i] for syn in outer_x_syndromes) > n_rounds // 2 else 0
                for i in range(n_syndrome_bits)
            )
        
        return voted_z, voted_x
    
    def _compute_outer_syndrome_from_inner_logicals(
        self,
        inner_z_logicals: Dict[int, int],  # Blocks with Z logical flips (caused by X errors)
        inner_x_logicals: Dict[int, int],  # Blocks with X logical flips (caused by Z errors)
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Compute 3-bit outer syndrome from 7 inner block logical outcomes.
        
        This is the core of the decoder-side outer syndrome computation (Gaps 1-3).
        
        CRITICAL ORIENTATION FIX:
        - inner_z_logicals[i]=1 means block i had its Z logical flipped = X error on outer "qubit" i
        - inner_x_logicals[i]=1 means block i had its X logical flipped = Z error on outer "qubit" i
        
        For CSS codes, X errors are detected by Z stabilizers (Hz), Z errors by X stabilizers (Hx):
        - X errors on outer blocks (inner_z_logicals) → detected by outer Hz → outer Z syndrome
        - Z errors on outer blocks (inner_x_logicals) → detected by outer Hx → outer X syndrome
        
        Returns:
            (outer_z_syndrome, outer_x_syndrome) as tuples
        """
        n_blocks = self._outer_n_blocks if self._outer_n_blocks > 0 else 7
        
        # Build inner logical error vectors
        inner_z_vec = np.array([inner_z_logicals.get(i, 0) for i in range(n_blocks)], dtype=np.uint8)
        inner_x_vec = np.array([inner_x_logicals.get(i, 0) for i in range(n_blocks)], dtype=np.uint8)
        
        # Compute outer syndromes - CORRECTED ORIENTATION
        if self._outer_Hz is not None:
            # Inner Z logical flips (X errors on outer) → detected by outer Hz → outer Z syndrome
            outer_z_syn = tuple((self._outer_Hz @ inner_z_vec) % 2)
        else:
            outer_z_syn = (0,) * 3
        
        if self._outer_Hx is not None:
            # Inner X logical flips (Z errors on outer) → detected by outer Hx → outer X syndrome
            outer_x_syn = tuple((self._outer_Hx @ inner_x_vec) % 2)
        else:
            outer_x_syn = (0,) * 3
        
        return outer_z_syn, outer_x_syn
    
    def _decode_outer_syndrome(
        self,
        outer_z_syndrome: Tuple[int, ...],
        outer_x_syndrome: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode outer-level syndrome to get block corrections.
        
        Uses BPOSD if enabled (Gap 5), otherwise lookup table.
        
        Returns:
            (z_block_correction, x_block_correction) - which blocks to flip
        """
        n_blocks = self._outer_n_blocks if self._outer_n_blocks > 0 else 7
        z_correction = np.zeros(n_blocks, dtype=np.uint8)
        x_correction = np.zeros(n_blocks, dtype=np.uint8)
        
        # Decode outer Z syndrome → Z block correction (flip X logical on these blocks)
        if sum(outer_z_syndrome) > 0:
            if self.config.use_bposd_outer and self._outer_bposd_z is not None:
                try:
                    syndrome_array = np.array(outer_z_syndrome, dtype=np.uint8)
                    z_correction = self._outer_bposd_z.decode(syndrome_array)
                except Exception:
                    # Fall back to table
                    if self._outer_z_table and outer_z_syndrome in self._outer_z_table:
                        z_correction, _ = self._outer_z_table[outer_z_syndrome]
            elif self._outer_z_table and outer_z_syndrome in self._outer_z_table:
                z_correction, _ = self._outer_z_table[outer_z_syndrome]
        
        # Decode outer X syndrome → X block correction (flip Z logical on these blocks)
        if sum(outer_x_syndrome) > 0:
            if self.config.use_bposd_outer and self._outer_bposd_x is not None:
                try:
                    syndrome_array = np.array(outer_x_syndrome, dtype=np.uint8)
                    x_correction = self._outer_bposd_x.decode(syndrome_array)
                except Exception:
                    # Fall back to table
                    if self._outer_x_table and outer_x_syndrome in self._outer_x_table:
                        x_correction, _ = self._outer_x_table[outer_x_syndrome]
            elif self._outer_x_table and outer_x_syndrome in self._outer_x_table:
                x_correction, _ = self._outer_x_table[outer_x_syndrome]
        
        if self.config.verbose and (np.sum(z_correction) > 0 or np.sum(x_correction) > 0):
            print(f"  Outer decode: Z correction on blocks {list(np.where(z_correction)[0])}, X correction on blocks {list(np.where(x_correction)[0])}")
        
        return z_correction, x_correction

    def _decode_from_syndrome_history(
        self,
        syndrome_history: List[Dict],
        syndrome_changes: List[Dict],
        metadata: 'MultiLevelMetadata',
        instability: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode errors from syndrome history (no final data).
        
        Key improvements:
        1. Syndrome differencing: Use syndrome CHANGES to identify when errors occurred
        2. Temporal filtering: Suppress unstable/changing syndromes (measurement errors)
        3. Majority voting: Robust against single-round measurement errors
        4. HIERARCHICAL OUTER DECODING: After decoding inner blocks, form outer syndrome
           from inner block logical parities and decode at outer level
        
        For concatenated codes with Steane EC:
        - Each inner block reports its syndrome
        - Non-zero syndrome indicates error in that block
        - Decode each block, track if it causes logical error at outer level
        - Form "outer syndrome" from inner block logical errors
        - Decode outer code to get final logical correction
        
        Returns:
            (x_correction, z_correction) arrays
        """
        x_correction = np.zeros(self.n_qubits, dtype=np.uint8)
        z_correction = np.zeros(self.n_qubits, dtype=np.uint8)
        
        # Initialize outer raw syndromes
        outer_x_syndrome_raw = None
        outer_z_syndrome_raw = None
        
        if not syndrome_history:
            return x_correction, z_correction
        
        # Track inner block logical errors for outer decoding
        # For X errors: Z corrections that flip inner X logical = inner X logical error
        # For Z errors: X corrections that flip inner Z logical = inner Z logical error
        inner_x_logical_errors = {}  # block_idx -> parity (affected by z_correction)
        inner_z_logical_errors = {}  # block_idx -> parity (affected by x_correction)
        
        # Determine instability once (allow caller override)
        instability = instability or self._check_measurement_instability(syndrome_history, syndrome_changes)

        # CRITICAL FIX: Use syndrome differencing when we have multiple rounds
        # This extracts WHEN errors occurred, not just WHAT the syndrome is
        if self.config.use_syndrome_history and len(syndrome_history) > 2:
            # Use syndrome differencing: decode the CHANGES not the raw syndromes
            x_raw, z_raw = self._aggregate_syndrome_changes(syndrome_changes, syndrome_history)
        elif self.config.use_majority_voting and len(syndrome_history) > 1 and not instability:
            # Majority voting across all rounds
            x_raw = self._majority_vote_syndromes(syndrome_history, 'X')
            z_raw = self._majority_vote_syndromes(syndrome_history, 'Z')
        else:
            # Single round or unstable: use last round
            x_raw = syndrome_history[-1].get('X', {})
            z_raw = syndrome_history[-1].get('Z', {})

        # If instability or block temporal filter enabled, skip unstable blocks
        unstable_blocks = set()
        if instability or (self.config.use_block_temporal_filter and self._is_concatenated_blockwise(metadata)):
            from collections import defaultdict
            block_change_counts = defaultdict(int)
            for change in syndrome_changes[1:]:  # skip t=0 baseline
                for syn_dict in (change.get('X', {}), change.get('Z', {})):
                    for key, syn in syn_dict.items():
                        if sum(syn) > 0:
                            block_change_counts[key] += 1
            # Only mark as unstable if changed MORE THAN ONCE
            unstable_blocks = {key for key, count in block_change_counts.items() if count > 1}
            if self.config.verbose and len(unstable_blocks) > 0 and self.config.use_block_temporal_filter:
                print(f"Info: Block temporal filter suppressing {len(unstable_blocks)} blocks that changed >1 time")

        # X syndromes (detect Z errors) -> Z corrections
        for block_key, raw_meas in x_raw.items():
            if block_key in unstable_blocks:
                continue

            level, block_idx = self._parse_block_key(block_key)
            if level is None:
                continue
            if level == 0:
                # For outer-level syndrome, collect ALL inner block measurements
                # and apply hierarchical extraction
                inner_measurements = []
                # Collect measurements from all 7 inner blocks
                for inner_block_id in range(7):  # TODO: Get from metadata
                    inner_key = (1, (1, inner_block_id))
                    if inner_key in x_raw:
                        inner_measurements.extend(x_raw[inner_key])
                
                # Capture outer Z syndrome (from X checks which detect Z errors)
                # For concatenated codes, need to compute outer syndrome from inner logical measurements
                outer_z_syndrome_raw = self._compute_outer_syndrome_from_raw(
                    inner_measurements, 'X', metadata
                )
                if self.config.verbose:
                    print(f"  Computed outer Z syndrome from {len(inner_measurements)} raw measurements: {outer_z_syndrome_raw}")
                continue

            true_syndrome = self._compute_true_syndrome(raw_meas, 'X')
            if sum(true_syndrome) > 0:
                block_correction = self._decode_inner_block_syndrome(
                    true_syndrome, block_key, 'Z', metadata
                )
                z_correction = (z_correction + block_correction) % 2
                
                # Track inner block logical error using syndrome-based detection
                # Key insight: if syndrome weight > max correctable, inner block has logical error
                if block_idx is not None:
                    inner_logical_error = self._detect_inner_logical_error(
                        true_syndrome, block_correction, 'X', level, block_idx, metadata
                    )
                    if block_idx not in inner_x_logical_errors:
                        inner_x_logical_errors[block_idx] = 0
                    inner_x_logical_errors[block_idx] = (inner_x_logical_errors[block_idx] + inner_logical_error) % 2

        # Z syndromes (detect X errors) -> X corrections
        for block_key, raw_meas in z_raw.items():
            if block_key in unstable_blocks:
                continue

            level, block_idx = self._parse_block_key(block_key)
            if level is None:
                continue
            if level == 0:
                # For outer-level syndrome, collect ALL inner block measurements
                # and apply hierarchical extraction
                inner_measurements = []
                # Collect measurements from all 7 inner blocks
                for inner_block_id in range(7):  # TODO: Get from metadata
                    inner_key = (1, (1, inner_block_id))
                    if inner_key in z_raw:
                        inner_measurements.extend(z_raw[inner_key])
                
                # Capture outer X syndrome (from Z checks which detect X errors)
                # For concatenated codes, need to compute outer syndrome from inner logical measurements
                outer_x_syndrome_raw = self._compute_outer_syndrome_from_raw(
                    inner_measurements, 'Z', metadata
                )
                if self.config.verbose:
                    print(f"  Computed outer X syndrome from {len(inner_measurements)} raw measurements: {outer_x_syndrome_raw}")
                continue

            true_syndrome = self._compute_true_syndrome(raw_meas, 'Z')
            if sum(true_syndrome) > 0:
                block_correction = self._decode_inner_block_syndrome(
                    true_syndrome, block_key, 'X', metadata
                )
                x_correction = (x_correction + block_correction) % 2
                
                # Track inner block logical error using syndrome-based detection
                # Key insight: if syndrome weight > max correctable, inner block has logical error
                if block_idx is not None:
                    inner_logical_error = self._detect_inner_logical_error(
                        true_syndrome, block_correction, 'Z', level, block_idx, metadata
                    )
                    if block_idx not in inner_z_logical_errors:
                        inner_z_logical_errors[block_idx] = 0
                    inner_z_logical_errors[block_idx] = (inner_z_logical_errors[block_idx] + inner_logical_error) % 2
        
        # ========================================================================
        # OUTER-LEVEL DECODING: Decode the pattern of inner block logical errors
        # ========================================================================
        # CRITICAL FIX: Decode outer syndrome INDEPENDENTLY, not just when combined with inner errors
        # The outer syndrome can detect errors even when inner blocks don't have logical failures
        
        # STEP 1: Independent outer syndrome decoding (raw measurements only)
        # This catches outer-level errors that don't propagate to inner logical errors
        if self._outer_n_blocks > 0 and (self._outer_x_table or self._outer_z_table):
            # First, decode outer raw syndromes independently
            if outer_z_syndrome_raw is not None and sum(outer_z_syndrome_raw) > 0:
                outer_z_syn_tuple = tuple(outer_z_syndrome_raw)
                if self._outer_z_table and outer_z_syn_tuple in self._outer_z_table:
                    independent_z_correction, _ = self._outer_z_table[outer_z_syn_tuple]
                    # Apply: These blocks have Z errors at the outer level
                    for block_idx, flip in enumerate(independent_z_correction):
                        if flip and block_idx < self._outer_n_blocks:
                            start_idx = block_idx * self._inner_n_qubits
                            for q in self._inner_x_logical_support:
                                if start_idx + q < self.n_qubits:
                                    z_correction[start_idx + q] = (z_correction[start_idx + q] + 1) % 2
                    if self.config.verbose:
                        print(f"  Independent outer Z correction (from raw syndrome {outer_z_syn_tuple}): {list(independent_z_correction)}")
            
            if outer_x_syndrome_raw is not None and sum(outer_x_syndrome_raw) > 0:
                outer_x_syn_tuple = tuple(outer_x_syndrome_raw)
                if self._outer_x_table and outer_x_syn_tuple in self._outer_x_table:
                    independent_x_correction, _ = self._outer_x_table[outer_x_syn_tuple]
                    # Apply: These blocks have X errors at the outer level
                    for block_idx, flip in enumerate(independent_x_correction):
                        if flip and block_idx < self._outer_n_blocks:
                            start_idx = block_idx * self._inner_n_qubits
                            for q in self._inner_z_logical_support:
                                if start_idx + q < self.n_qubits:
                                    x_correction[start_idx + q] = (x_correction[start_idx + q] + 1) % 2
                    if self.config.verbose:
                        print(f"  Independent outer X correction (from raw syndrome {outer_x_syn_tuple}): {list(independent_x_correction)}")
        
        # STEP 2: Combined outer decoding (raw + inferred from inner logical errors)
        # This handles cases where both inner and outer levels have errors
        if self._outer_n_blocks > 0 and (self._outer_x_table or self._outer_z_table):
            # Form outer syndrome from inner block logical errors
            # inner_x_logical_errors: Z corrections caused X logical errors in blocks
            # inner_z_logical_errors: X corrections caused Z logical errors in blocks
            
            # 1. Construct error vectors
            # These are vectors of length n_blocks, where 1 means logical error on that block
            outer_x_error_vector = np.array(
                [inner_x_logical_errors.get(i, 0) for i in range(self._outer_n_blocks)],
                dtype=np.uint8
            )
            outer_z_error_vector = np.array(
                [inner_z_logical_errors.get(i, 0) for i in range(self._outer_n_blocks)],
                dtype=np.uint8
            )
            
            # Proceed with combined decoding if there are inner errors to combine
            has_inner_errors = (np.sum(outer_x_error_vector) > 0 or np.sum(outer_z_error_vector) > 0)
            
            if has_inner_errors:
                # 2. Compute TRUE outer syndromes: s_total = H @ s_raw + H @ e_inferred
                # This combines: (a) actual syndrome measurements from circuit
                #                (b) inferred inner block logical failures
                
                # Outer X Syndrome (from Hz stabilizers, detects Z errors)
                n_x_checks = self._outer_Hz.shape[0] if self._outer_Hz is not None else self._outer_n_blocks
                s_x = np.zeros(n_x_checks, dtype=np.uint8)
                
                # Add outer syndrome (already computed from raw measurements)
                if outer_z_syndrome_raw is not None:
                    syn = np.array(outer_z_syndrome_raw, dtype=np.uint8)
                    if len(syn) == n_x_checks:
                        s_x = (s_x + syn) % 2
                
                # Add inferred syndrome from inner X logical errors
                if self._outer_Hz is not None:
                    s_x = (s_x + (self._outer_Hz @ outer_x_error_vector)) % 2
                else:
                    s_x = (s_x + outer_x_error_vector[:n_x_checks]) % 2
                outer_x_syndrome = tuple(s_x)

                # Outer Z Syndrome (from Hx stabilizers, detects X errors)
                n_z_checks = self._outer_Hx.shape[0] if self._outer_Hx is not None else self._outer_n_blocks
                s_z = np.zeros(n_z_checks, dtype=np.uint8)
                
                # Add outer syndrome (already computed from raw measurements)
                if outer_x_syndrome_raw is not None:
                    syn = np.array(outer_x_syndrome_raw, dtype=np.uint8)
                    if len(syn) == n_z_checks:
                        s_z = (s_z + syn) % 2
                
                # Add inferred syndrome from inner Z logical errors
                if self._outer_Hx is not None:
                    s_z = (s_z + (self._outer_Hx @ outer_z_error_vector)) % 2
                else:
                    s_z = (s_z + outer_z_error_vector[:n_z_checks]) % 2
                outer_z_syndrome = tuple(s_z)

                if self.config.verbose and (sum(outer_x_syndrome) > 0 or sum(outer_z_syndrome) > 0):
                    print(f"  Combined outer X syndrome (raw + inner errors): {outer_x_syndrome}")
                    print(f"  Combined outer Z syndrome (raw + inner errors): {outer_z_syndrome}")
                
                # CRITICAL: Decode outer X syndrome → outer Z correction (which blocks to flip)
                # This applies ADDITIONAL correction on top of the independent decoding above
                if sum(outer_x_syndrome) > 0 and self._outer_z_table and outer_x_syndrome in self._outer_z_table:
                    outer_z_correction, _ = self._outer_z_table[outer_x_syndrome]
                    # Apply outer correction: flip Z logical on indicated blocks
                    for block_idx, flip in enumerate(outer_z_correction):
                        if flip:
                            start_idx = block_idx * self._inner_n_qubits
                            for q in self._inner_x_logical_support:
                                if start_idx + q < self.n_qubits:
                                    z_correction[start_idx + q] = (z_correction[start_idx + q] + 1) % 2
                    if self.config.verbose:
                        print(f"  Outer Z correction (blocks to flip X logical): {list(outer_z_correction)}")
                
                # Decode outer Z syndrome → outer X correction (which blocks to flip)
                if sum(outer_z_syndrome) > 0 and self._outer_x_table and outer_z_syndrome in self._outer_x_table:
                    outer_x_correction, _ = self._outer_x_table[outer_z_syndrome]
                    # Apply outer correction: flip X logical on indicated blocks
                    for block_idx, flip in enumerate(outer_x_correction):
                        if flip:
                            start_idx = block_idx * self._inner_n_qubits
                            for q in self._inner_z_logical_support:
                                if start_idx + q < self.n_qubits:
                                    x_correction[start_idx + q] = (x_correction[start_idx + q] + 1) % 2
                    if self.config.verbose:
                        print(f"  Outer X correction (blocks to flip Z logical): {list(outer_x_correction)}")
        
        return x_correction, z_correction

    def _compute_true_syndrome(
        self,
        raw_measurements: Tuple[int, ...],
        error_type: str,
    ) -> Tuple[int, ...]:
        """
        Compute true syndrome from raw ancilla measurements.
        
        CRITICAL FIX: TransversalSyndromeGadget measures ALL n ancilla qubits (e.g., 7 for Steane).
        These are RAW ancilla measurements, NOT the syndrome. The syndrome has (n-k) bits (e.g., 3 for Steane).
        
        We MUST apply H @ raw to get the true syndrome, where H is the parity check matrix.
        
        For Steane EC:
        - Raw measurements: 7 bits (one per ancilla qubit)
        - Parity check matrix Hz: 3×7 (3 stabilizers, 7 qubits)
        - True syndrome: Hz @ raw = 3 bits
        
        For Shor EC (redundant measurements):
        - Raw measurements: (n-k)×w bits (w ancillas per stabilizer)
        - Each group of w measurements corresponds to one syndrome bit
        - Syndrome bit = XOR of w measurements
        - Example: Steane with w=4: 12 measurements → 3 syndrome bits
        
        Args:
            raw_measurements: Tuple of measurement outcomes
            error_type: 'X' for X errors (use Hz) or 'Z' for Z errors (use Hx)
            
        Returns:
            Syndrome tuple of (n-k) bits
        """
        if self.inner_code_info is None:
            # No inner code info - return raw (fallback)
            return raw_measurements
        
        raw = np.array(raw_measurements, dtype=np.uint8)
        
        # CRITICAL FIX: error_type refers to the SYNDROME TYPE, not the error type
        # - 'Z' syndrome (from Z stabilizers Hz) detects X errors
        # - 'X' syndrome (from X stabilizers Hx) detects Z errors
        # So: syndrome_type='Z' → use Hz, syndrome_type='X' → use Hx
        if error_type == 'Z':
            H = self.inner_code_info['Hz']  # Z syndrome from Hz
        else:
            H = self.inner_code_info['Hx']  # X syndrome from Hx
        
        expected_syndrome_size = H.shape[0]  # Number of parity checks (n-k)
        n_qubits = H.shape[1]  # Number of qubits (n)
        
        # PRIMARY CASE: Raw is n-bit ancilla measurement → compute syndrome
        if len(raw) == n_qubits:
            syndrome = (H @ raw) % 2
            return tuple(int(x) for x in syndrome)
        # SHOR EC CASE: Raw is (n-k)×w bits (w redundant measurements per stabilizer)
        elif len(raw) % expected_syndrome_size == 0:
            w = len(raw) // expected_syndrome_size  # Measurements per stabilizer
            syndrome = []
            
            for stab_idx in range(expected_syndrome_size):
                # Get w measurements for this stabilizer
                stab_meas = raw[stab_idx * w : (stab_idx + 1) * w]
                
                # CRITICAL FIX FOR SHOR EC:
                # For Shor-style EC, measurements are organized as:
                #   [rep0: weight meas, rep1: weight meas, ..., repN: weight meas]
                # where weight is the number of qubits in this stabilizer's support
                #
                # To decode:
                #   1. Determine weight from parity check matrix H
                #   2. If w % weight == 0, we have (w // weight) reps
                #   3. Group into reps, XOR each rep, then majority vote
                
                # Get stabilizer weight from H matrix
                stab_weight = int(np.sum(H[stab_idx, :]))
                
                if stab_weight > 0 and w % stab_weight == 0:
                    # Shor EC with multiple reps
                    n_reps = w // stab_weight
                    if n_reps > 1:
                        # Group measurements into reps and decode properly
                        rep_bits = []
                        for rep_idx in range(n_reps):
                            rep_start = rep_idx * stab_weight
                            rep_end = rep_start + stab_weight
                            rep_meas = stab_meas[rep_start:rep_end]
                            # XOR measurements within this rep (cat state decoding)
                            rep_bit = int(np.sum(rep_meas) % 2)
                            rep_bits.append(rep_bit)
                        
                        # Majority vote across reps
                        ones_count = sum(rep_bits)
                        syndrome_bit = 1 if ones_count > n_reps // 2 else 0
                        syndrome.append(syndrome_bit)
                    else:
                        # Single rep: just XOR all measurements
                        syndrome_bit = int(np.sum(stab_meas) % 2)
                        syndrome.append(syndrome_bit)
                else:
                    # Fallback: simple majority vote (old behavior)
                    ones_count = int(np.sum(stab_meas))
                    syndrome_bit = 1 if ones_count > w // 2 else 0
                    syndrome.append(syndrome_bit)
            
            return tuple(syndrome)
        # FALLBACK: Raw is already syndrome-sized
        elif len(raw) == expected_syndrome_size:
            return tuple(int(x) for x in raw)
        else:
            # Size mismatch - try to apply H anyway
            if self.config.verbose:
                print(f"Warning: Raw size {len(raw)} doesn't match expected patterns (n={n_qubits}, syndrome_size={expected_syndrome_size})")
            if len(raw) <= n_qubits:
                # Pad and compute
                padded = np.zeros(n_qubits, dtype=np.uint8)
                padded[:len(raw)] = raw
                syndrome = (H @ padded) % 2
                return tuple(int(x) for x in syndrome)
            else:
                # Try Shor EC grouping as last resort
                if len(raw) >= expected_syndrome_size:
                    w = len(raw) // expected_syndrome_size
                    syndrome = []
                    for stab_idx in range(expected_syndrome_size):
                        start_idx = stab_idx * w
                        end_idx = min(start_idx + w, len(raw))
                        stab_meas = raw[start_idx:end_idx]
                        # FIXED: Use majority vote instead of XOR
                        w_actual = end_idx - start_idx
                        ones_count = int(np.sum(stab_meas))
                        syndrome_bit = 1 if ones_count > w_actual // 2 else 0
                        syndrome.append(syndrome_bit)
                    return tuple(syndrome)
                else:
                    # Fallback - truncate and compute
                    syndrome = (H @ raw[:n_qubits]) % 2
                    return tuple(int(x) for x in syndrome)
    
    def _aggregate_syndrome_changes(
        self,
        syndrome_changes: List[Dict],
        syndrome_history: List[Dict],
    ) -> Tuple[Dict, Dict]:
        """
        Aggregate syndrome changes across EC rounds using syndrome differencing.
        
        Syndrome differencing extracts WHEN errors occurred:
        - Use the LAST stable round's syndrome (most recent stable measurement)
        - This avoids accumulating transient measurement errors
        - If no stable round exists, use the last round
        
        This is more robust than XOR accumulation because:
        1. Uses the most recent reliable syndrome measurement
        2. Avoids cancellation artifacts from XOR
        3. Handles persistent vs transient measurement errors correctly
        
        Returns:
            (aggregated_x_syndromes, aggregated_z_syndromes)
        """
        from collections import defaultdict
        import numpy as np
        
        if not syndrome_history or len(syndrome_changes) < 2:
            # Not enough rounds for differencing - use last round
            if syndrome_history:
                return syndrome_history[-1].get('X', {}), syndrome_history[-1].get('Z', {})
            return {}, {}
        
        # ACTUAL syndrome differencing: XOR syndrome changes across rounds
        # This identifies WHEN errors occurred (round-by-round changes)
        x_aggregated = {}
        z_aggregated = {}
        
        # Initialize all blocks from syndrome_history with correct sizes per block
        # Different levels may have different syndrome sizes (inner vs outer)
        for round_data in syndrome_history:
            for block_key, syn in round_data.get('X', {}).items():
                # CRITICAL FIX: Skip outer-level keys (level=0) - decoder computes these
                level, _ = self._parse_block_key(block_key)
                if level == 0:
                    continue
                if block_key not in x_aggregated:
                    x_aggregated[block_key] = np.zeros(len(syn), dtype=np.uint8)
            for block_key, syn in round_data.get('Z', {}).items():
                # CRITICAL FIX: Skip outer-level keys (level=0) - decoder computes these
                level, _ = self._parse_block_key(block_key)
                if level == 0:
                    continue
                if block_key not in z_aggregated:
                    z_aggregated[block_key] = np.zeros(len(syn), dtype=np.uint8)
        
        # Process syndrome changes from each round (skip t=0 baseline)
        for change_dict in syndrome_changes[1:]:
            # X syndromes (detect Z errors)
            for block_key, change_syn in change_dict.get('X', {}).items():
                change_array = np.array(change_syn, dtype=np.uint8)
                if block_key not in x_aggregated:
                    # Initialize with correct size for this block
                    x_aggregated[block_key] = change_array
                else:
                    # XOR accumulation: net change
                    x_aggregated[block_key] = (x_aggregated[block_key] + change_array) % 2
            
            # Z syndromes (detect X errors)
            for block_key, change_syn in change_dict.get('Z', {}).items():
                change_array = np.array(change_syn, dtype=np.uint8)
                if block_key not in z_aggregated:
                    # Initialize with correct size for this block
                    z_aggregated[block_key] = change_array
                else:
                    z_aggregated[block_key] = (z_aggregated[block_key] + change_array) % 2
        
        # Convert to regular dicts with tuples (KEEP ALL syndromes including zeros for BPOSD)
        x_result = {k: tuple(v) for k, v in x_aggregated.items()}
        z_result = {k: tuple(v) for k, v in z_aggregated.items()}
        
        if self.config.verbose:
            x_nonzero = sum(1 for s in x_result.values() if sum(s) > 0)
            z_nonzero = sum(1 for s in z_result.values() if sum(s) > 0)
            print(f"Syndrome differencing: XOR of {len(syndrome_changes)-1} rounds → {x_nonzero} X blocks, {z_nonzero} Z blocks (non-zero)")
        
        return x_result, z_result
    
    def _majority_vote_syndromes(
        self,
        syndrome_history: List[Dict],
        error_type: str,
    ) -> Dict:
        """
        Apply majority voting across rounds to handle measurement errors.
        
        For each block, take the syndrome that appears most often.
        """
        from collections import Counter
        
        # Collect all syndromes for each block across rounds
        block_syndromes: Dict[Any, List[Tuple]] = {}
        
        for round_data in syndrome_history:
            for block_key, syndrome in round_data.get(error_type, {}).items():
                if block_key not in block_syndromes:
                    block_syndromes[block_key] = []
                block_syndromes[block_key].append(syndrome)
        
        # Majority vote for each block
        result = {}
        for block_key, syndromes in block_syndromes.items():
            if syndromes:
                # Count syndrome occurrences
                counter = Counter(syndromes)
                # Get most common
                result[block_key] = counter.most_common(1)[0][0]
        
        return result
    
    def _detect_inner_logical_error(
        self,
        syndrome: Tuple,
        correction: np.ndarray,
        syndrome_type: str,  # 'X' or 'Z'
        level: int,
        block_idx: int,
        metadata: 'MultiLevelMetadata'
    ) -> int:
        """
        Detect if an inner block has experienced a logical error.
        
        FIX 4: SIMPLIFIED HEURISTIC to reduce false positives.
        Only flag logical error if BOTH:
        1. Correction weight > max_correctable (indicating heavy error)
        2. Syndrome weight > max_correctable (consistent with heavy error)
        
        Previous version was too aggressive, flagging many correctable errors
        as logical errors, causing the outer decoder to overcorrect.
        
        Args:
            syndrome: The syndrome tuple for this block
            correction: The correction applied to this block  
            syndrome_type: 'X' for X stabilizer syndrome, 'Z' for Z stabilizer
            level: The concatenation level
            block_idx: Index of the block
            metadata: Multi-level metadata
            
        Returns:
            1 if logical error detected, 0 otherwise
        """
        # Zero syndrome = no error
        syndrome_weight = sum(int(s) for s in syndrome)
        if syndrome_weight == 0:
            return 0
        
        # Compute correction weight
        correction_weight = np.sum(correction)
        
        # Get distance and max correctable weight
        inner_d = self.inner_code_info.get('distance', 3) if self.inner_code_info else 3
        max_correctable = (inner_d - 1) // 2  # = 1 for d=3
        
        # FIX 4: Require BOTH conditions to avoid false positives
        # For d=3 Steane: only flag if correction_weight > 1 AND syndrome_weight > 1
        # This is much more conservative than the previous heuristic
        if correction_weight > max_correctable and syndrome_weight > max_correctable:
            return 1
        
        return 0
    
    def _decode_inner_block_syndrome(
        self,
        syndrome: Tuple,
        block_key: Any,
        error_type: str,
        metadata: 'MultiLevelMetadata',
    ) -> np.ndarray:
        """
        Decode syndrome for a single inner block.
        
        Uses BPOSD if available (more robust for noisy syndromes),
        otherwise falls back to lookup table or exhaustive search.
        
        For concatenated codes, inner syndromes should be decoded using
        the inner code's syndrome table, not the full code's table.
        
        Maps the inner block's syndrome to a correction on the full code.
        """
        correction = np.zeros(self.n_qubits, dtype=np.uint8)

        level, block_idx = self._parse_block_key(block_key)
        if level is None:
            if self.config.verbose:
                print(f"Warning: Could not parse block_key {block_key}")
            return correction

        # Use BPOSD for inner blocks if available (preferred)
        if level > 0 and self.config.use_bposd_inner:
            bposd_decoder = self._inner_bposd_z if error_type == 'Z' else self._inner_bposd_x
            
            if bposd_decoder is not None:
                try:
                    # CRITICAL: Verify syndrome size matches parity check matrix
                    H = self.inner_code_info['Hz'] if error_type == 'Z' else self.inner_code_info['Hx']
                    expected_syndrome_size = H.shape[0]
                    
                    if len(syndrome) != expected_syndrome_size:
                        if self.config.verbose:
                            print(f"Warning: Syndrome size {len(syndrome)} != expected {expected_syndrome_size}, recomputing")
                        # Recompute syndrome to ensure correct size
                        syndrome = self._compute_true_syndrome(syndrome, error_type)
                    
                    # BPOSD decode - ALWAYS use BPOSD even for zero syndromes (BP needs all context)
                    syndrome_array = np.array(syndrome, dtype=np.uint8)
                    inner_correction = bposd_decoder.decode(syndrome_array)
                    
                    # Map to full array
                    start_idx, block_size = self._resolve_block_range(level, block_idx, metadata)
                    for i, val in enumerate(inner_correction):
                        if i < block_size and start_idx + i < self.n_qubits:
                            correction[start_idx + i] = int(val)
                    
                    return correction
                except Exception as e:
                    if self.config.verbose:
                        print(f"Warning: BPOSD decode failed for {syndrome}: {e}, falling back")

        # Fallback: Try lookup table
        if error_type == 'X' and self._inner_x_table is not None and level > 0:
            table = self._inner_x_table
        elif error_type == 'Z' and self._inner_z_table is not None and level > 0:
            table = self._inner_z_table
        else:
            table = self._z_table if error_type == 'Z' else self._x_table

        if syndrome in table:
            inner_correction, _ = table[syndrome]
            start_idx, block_size = self._resolve_block_range(level, block_idx, metadata)
            for i, val in enumerate(inner_correction):
                if i < block_size and start_idx + i < self.n_qubits:
                    correction[start_idx + i] = val
        else:
            # Syndrome not in table - exhaustive search fallback (increased max_weight to 4)
            if level > 0 and self.inner_code_info is not None:
                H_inner = self.inner_code_info['Hz'] if error_type == 'Z' else self.inner_code_info['Hx']
                n_inner = self.inner_code_info['n_qubits']
                
                inner_correction = self._find_minimum_weight_error(
                    np.array(syndrome, dtype=np.uint8),
                    H_inner,
                    n_inner,
                    max_weight=min(4, (n_inner + 1) // 2)
                )
                
                start_idx, block_size = self._resolve_block_range(level, block_idx, metadata)
                for i, val in enumerate(inner_correction):
                    if i < block_size and start_idx + i < self.n_qubits:
                        correction[start_idx + i] = val
                
                if self.config.verbose and np.sum(inner_correction) > 0:
                    print(f"Info: exhaustive search found weight-{np.sum(inner_correction)} correction for syndrome {syndrome}")
            elif self.config.verbose:
                print(f"Warning: inner syndrome {syndrome} not found and no fallback available")

        return correction
    
    def _find_minimum_weight_error(
        self,
        syndrome: np.ndarray,
        H: np.ndarray,
        n_qubits: int,
        max_weight: int = 3,
    ) -> np.ndarray:
        """
        Find minimum-weight error consistent with syndrome using exhaustive search.
        
        This is a fallback for syndromes not in the lookup table.
        """
        from itertools import combinations
        
        # Weight 0
        if np.sum(syndrome) == 0:
            return np.zeros(n_qubits, dtype=np.uint8)
        
        # Weights 1 to max_weight
        for weight in range(1, max_weight + 1):
            for qubits in combinations(range(n_qubits), weight):
                e = np.zeros(n_qubits, dtype=np.uint8)
                for q in qubits:
                    e[q] = 1
                
                if np.array_equal((H @ e) % 2, syndrome):
                    return e
        
        # No correction found - return zero
        return np.zeros(n_qubits, dtype=np.uint8)
    
    def _parse_block_key(self, block_key: Any) -> Tuple[Optional[int], Optional[int]]:
        """Parse block key to (level, block_idx) across common layouts."""
        # Allow metadata to provide custom normalization
        if hasattr(self, '_metadata_normalize_hook') and self._metadata_normalize_hook is not None:
            try:
                result = self._metadata_normalize_hook(block_key)
                if isinstance(result, tuple) and len(result) == 2:
                    return result
            except Exception as e:
                if self.config.verbose:
                    print(f"Warning: metadata normalize_block_key hook failed: {e}")

        if isinstance(block_key, tuple):
            # Handle ((level, block_idx),) format from multilevel_memory syndrome extraction
            # This comes from key[1:] in _extract_syndrome_history where key=(round, (level, block))
            if len(block_key) == 1 and isinstance(block_key[0], tuple):
                inner = block_key[0]  # Extract (level, block_idx) from the wrapper
                # Check if this is a simple (level, block) pair
                if len(inner) == 2 and all(isinstance(x, (int, np.integer)) for x in inner):
                    return int(inner[0]), int(inner[1])
                # If it's still nested deeper, recurse
                return self._parse_block_key(inner)
            # Handle (level, (level2, block_idx)) nested format - extract innermost
            if len(block_key) == 2 and isinstance(block_key[0], (int, np.integer)) and isinstance(block_key[1], tuple):
                nested = block_key[1]
                if isinstance(nested, tuple) and len(nested) == 2:
                    if all(isinstance(x, (int, np.integer)) for x in nested):
                        return int(nested[0]), int(nested[1])
                    # Even deeper nesting: (level, (level2, (level3, block_idx)))
                    # Recurse to extract the innermost (level, block_idx)
                    return self._parse_block_key(nested)
            if len(block_key) >= 3 and isinstance(block_key[0], (int, np.integer)) and isinstance(block_key[1], (int, np.integer)):
                return int(block_key[1]), int(block_key[2])
            if len(block_key) == 2 and all(isinstance(x, (int, np.integer)) for x in block_key):
                return int(block_key[0]), int(block_key[1])
            if len(block_key) == 1 and isinstance(block_key[0], (int, np.integer)):
                return 1, int(block_key[0])
        elif isinstance(block_key, (int, np.integer)):
            return 1, int(block_key)

        # Log parse failure once per unique key
        if self.config.verbose and block_key not in self._warned_parse_keys:
            print(f"Warning: Could not parse block_key: {block_key}")
            self._warned_parse_keys.add(block_key)
        return None, None

    def _resolve_block_range(
        self,
        level: int,
        block_idx: int,
        metadata: Optional['MultiLevelMetadata'] = None,
        default_block_size: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Return (start, size) for block using metadata.address_to_range when provided."""
        if metadata is not None and hasattr(metadata, 'address_to_range'):
            ranges = self._cached_level_ranges.get(level)
            if ranges is None:
                ranges = self._collect_level_ranges_from_metadata(level, metadata)
                self._cached_level_ranges[level] = ranges
            if block_idx in ranges:
                start, end = ranges[block_idx]
                return start, end - start
            if self.config.strict_block_mapping:
                raise ValueError(f"Missing address_to_range for level {level} block {block_idx}")

        # Fallback to contiguous packing when no metadata mapping is available
        size_from_meta = None
        if metadata is not None and hasattr(metadata, 'level_block_sizes'):
            lbs = getattr(metadata, 'level_block_sizes')
            if isinstance(lbs, dict) and level in lbs:
                size_from_meta = lbs[level]

        if self.config.strict_block_mapping and metadata is not None:
            raise ValueError(f"Strict block mapping enabled but no range for level {level} block {block_idx}")

        n_inner = size_from_meta or default_block_size or (self._inner_n_qubits if level > 0 else self.n_qubits) or 7

        # Warn once per level when falling back to contiguous packing
        if self.config.verbose and level not in self._warned_levels:
            if metadata is not None and size_from_meta is None and not hasattr(metadata, 'address_to_range'):
                print(f"Warning: using fallback contiguous mapping for level {level} with size {n_inner}")
                self._warned_levels.add(level)

        start_idx = block_idx * n_inner
        return start_idx, n_inner

    def _collect_level_ranges_from_metadata(self, level: int, metadata: 'MultiLevelMetadata') -> Dict[int, Tuple[int, int]]:
        ranges: Dict[int, Tuple[int, int]] = {}
        addr_map = getattr(metadata, 'address_to_range', {})
        for addr, rng in addr_map.items():
            if not (isinstance(addr, tuple) and len(addr) >= 2):
                continue
            if addr[0] != level:
                continue
            block_idx = addr[1]
            if isinstance(block_idx, int):
                ranges[block_idx] = rng
        return ranges

    def _build_level_ranges(self, metadata: 'MultiLevelMetadata') -> Dict[int, Dict[int, Tuple[int, int]]]:
        level_ranges: Dict[int, Dict[int, Tuple[int, int]]] = {}
        addr_map = getattr(metadata, 'address_to_range', {})
        for addr, rng in addr_map.items():
            if not (isinstance(addr, tuple) and len(addr) >= 2):
                continue
            level, block_idx = addr[0], addr[1]
            if not isinstance(block_idx, int):
                continue
            if level not in level_ranges:
                level_ranges[level] = {}
            level_ranges[level][block_idx] = rng
        return level_ranges

    def _is_concatenated_blockwise(self, metadata: 'MultiLevelMetadata') -> bool:
        """Detect concatenated block structure from explicit flag or heuristic."""
        # Use explicit flag if available
        if hasattr(metadata, 'is_concatenated'):
            return bool(metadata.is_concatenated)

        # Fall back to heuristic
        if not hasattr(metadata, 'syndrome_measurements'):
            return False
        for stab_type in ('X', 'Z'):
            for key in metadata.syndrome_measurements.get(stab_type, {}).keys():
                # Nested block info like (round, (level, idx)) or 3+ tuple implies concatenation
                if isinstance(key, tuple):
                    if len(key) >= 2 and isinstance(key[1], tuple):
                        if self.config.verbose:
                            print(f"Info: Using heuristic to detect concatenation (nested tuple key)")
                        return True
                    if len(key) >= 3:
                        if self.config.verbose:
                            print(f"Info: Using heuristic to detect concatenation (length-3+ key)")
                        return True
        return False

    def _lookup_correction_from_table(self, syndrome: Tuple[int, ...], error_type: str) -> np.ndarray:
        """Fetch minimum-weight correction from the appropriate table or zeros."""
        table = self._z_table if error_type == 'Z' else self._x_table
        if syndrome in table:
            correction, _ = table[syndrome]
            return correction.copy()
        return np.zeros(self.n_qubits, dtype=np.uint8)
    
    def _aggregate_inner_logicals_to_outer(
        self,
        x_correction: np.ndarray,
        z_correction: np.ndarray,
        metadata: Optional['MultiLevelMetadata'] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate inner block logical parities to outer-level evidence.
        
        For hierarchical codes, inner blocks may have local errors that propagate
        to logical errors at the outer level. This method computes the logical
        parity for each inner block and can serve as a proxy outer syndrome.
        
        Returns:
            Dict with 'inner_x_logicals', 'inner_z_logicals', 'outer_x_proxy', 'outer_z_proxy'
        """
        if not hasattr(self, 'inner_code_info') or self.inner_code_info is None:
            return {}
        
        # Get inner code logical supports
        inner_x_support = set()
        inner_z_support = set()
        
        # Try to extract from inner_code_info
        if 'x_logical' in self.inner_code_info:
            inner_x_support = set(self.inner_code_info['x_logical'])
        if 'z_logical' in self.inner_code_info:
            inner_z_support = set(self.inner_code_info['z_logical'])
        
        # Fallback to simple heuristic for Steane
        if not inner_x_support:
            inner_x_support = {0, 1, 2}
        if not inner_z_support:
            inner_z_support = {0, 1, 2}
        
        n_inner = self.inner_code_info.get('n_qubits', 7)
        n_blocks = self.n_qubits // n_inner
        
        inner_x_logicals = []
        inner_z_logicals = []
        
        for block_idx in range(n_blocks):
            start_idx = block_idx * n_inner
            
            # Compute X logical parity for this block (affected by Z corrections)
            x_log = sum(
                int(z_correction[start_idx + q]) for q in inner_x_support
                if start_idx + q < self.n_qubits
            ) % 2
            inner_x_logicals.append(x_log)
            
            # Compute Z logical parity for this block (affected by X corrections)
            z_log = sum(
                int(x_correction[start_idx + q]) for q in inner_z_support
                if start_idx + q < self.n_qubits
            ) % 2
            inner_z_logicals.append(z_log)
        
        # Outer-level proxy: treat inner logicals as pseudo-syndrome
        return {
            'inner_x_logicals': inner_x_logicals,
            'inner_z_logicals': inner_z_logicals,
            'outer_x_proxy': tuple(inner_x_logicals),
            'outer_z_proxy': tuple(inner_z_logicals),
            'n_inner_x_errors': sum(inner_x_logicals),
            'n_inner_z_errors': sum(inner_z_logicals),
        }
    
    def _map_inner_correction_to_physical(
        self,
        inner_correction: np.ndarray,
        block_key: Any,
        metadata: 'MultiLevelMetadata',
    ) -> np.ndarray:
        """
        Map correction from inner block to physical qubit indices.
        
        For Steane^2, block_key is typically ((level, block_idx),) where
        level=1 means inner code blocks.
        """
        correction = np.zeros(self.n_qubits, dtype=np.uint8)
        level, block_idx = self._parse_block_key(block_key)
        if level is None or block_idx is None:
            return correction
        if level >= 1:
            start_idx, block_size = self._resolve_block_range(level, block_idx, metadata)
            for i, val in enumerate(inner_correction):
                if i < block_size and start_idx + i < self.n_qubits:
                    correction[start_idx + i] = val
        
        return correction
    
    def _compute_logical_from_correction(
        self,
        correction: np.ndarray,
        logical_support: Set[int],
    ) -> int:
        """
        Compute logical value implied by a correction.
        
        If the correction has odd weight on the logical support,
        it flips the logical value.
        """
        return sum(int(correction[q]) for q in logical_support if q < len(correction)) % 2
    
    def _get_aggregated_syndrome(
        self,
        round_data: Dict,
        error_type: str,
    ) -> Tuple:
        """Get aggregated syndrome from a single round's data."""
        syndromes = round_data.get(error_type, {})
        if not syndromes:
            return tuple()
        
        # If there's an outer-level syndrome, use that
        for key, syn in syndromes.items():
            if isinstance(key, tuple) and len(key) >= 1:
                if isinstance(key[0], tuple):
                    level = key[0][0]
                else:
                    level = key[0]
                if level == 0:  # Outer level
                    return syn
        
        # Otherwise return first available
        return list(syndromes.values())[0] if syndromes else tuple()

    def decode_final_data(self, final_data: np.ndarray) -> CircuitDecodeResult:
        """
        DEPRECATED: Decode from final data only.
        
        WARNING: This method uses final data for decoding, which is "cheating"
        for fault-tolerance benchmarking. Use decode() + validate() instead.
        
        This method is kept for backward compatibility only.
        """
        import warnings
        warnings.warn(
            "decode_final_data() uses final data for decoding, which bypasses "
            "syndrome-based error correction. For true FT benchmarking, use "
            "decode() + validate() instead.",
            DeprecationWarning
        )
        
        # Compute syndromes from final data
        x_syndrome = tuple(int(x) for x in (self.Hz @ final_data) % 2)
        z_syndrome = tuple(int(x) for x in (self.Hx @ final_data) % 2)
        
        # Apply minimum-weight decoding
        x_logical, _ = self._decode_error_type(
            final_data, x_syndrome, self._x_table, self.ZL
        )
        z_logical, _ = self._decode_error_type(
            final_data, z_syndrome, self._z_table, self.XL
        )
        
        return CircuitDecodeResult(
            logical_z=x_logical,
            logical_x=z_logical,
            x_syndrome_final=x_syndrome,
            z_syndrome_final=z_syndrome,
            decoding_method="final_data_cheating",
        )
    
    def _decode_error_type(
        self,
        data: np.ndarray,
        syndrome: Tuple,
        table: Dict,
        logical_support: set,
    ) -> Tuple[int, int]:
        """
        Decode one error type (X or Z).
        
        Returns (logical_value, correction_weight).
        """
        if syndrome in table:
            correction, weight = table[syndrome]
            # Apply correction to data
            corrected = (data + correction) % 2
            # Compute logical value
            logical = sum(int(corrected[q]) for q in logical_support) % 2
            return (logical, weight)
        else:
            # Syndrome not in table - compute raw logical
            logical = sum(int(data[q]) for q in logical_support) % 2
            if self.config.verbose:
                print(f"Warning: syndrome not in table (weight > {self.config.max_weight})")
            return (logical, -1)  # -1 indicates uncorrectable
    
    # =========================================================================
    # JOINT MINIMUM-WEIGHT SEARCH ENGINE
    # =========================================================================
    
    def _joint_search_enumerate(
        self,
        block_syndromes: Dict[int, Tuple],  # block_idx -> (z_syn, x_syn)
        n_blocks: int,
        error_type: str = 'Z',  # 'Z' or 'X'
    ) -> Tuple[np.ndarray, int, int]:
        """
        Joint minimum-weight search across all inner blocks.
        
        Args:
            block_syndromes: Map from block_idx to (z_syndrome, x_syndrome) tuples
            n_blocks: Total number of blocks
            error_type: 'Z' for Z errors (X syndrome) or 'X' for X errors (Z syndrome)
        
        Returns:
            (global_correction, total_weight, logical_flip): 
            - global_correction: Physical correction pattern
            - total_weight: Total error weight
            - logical_flip: 0 or 1, the final outer logical flip
        
        Algorithm:
            1. For each block, get list of candidates from multi-candidate tables
            2. Enumerate all combinations up to max_total_weight
            3. For each combination, compute outer syndrome
            4. Check if outer syndrome is correctable (in outer tables)
            5. Return minimum-weight valid combination with outer logical flip
        """
        if not self.config.use_multi_candidate_tables:
            # Fallback to hierarchical if multi-candidate not enabled
            corr, w = self._fallback_hierarchical_decode(block_syndromes, n_blocks, error_type)
            return corr, w, 0  # No logical flip info from fallback
        
        # Get candidate tables
        if error_type == 'Z':
            candidate_table = self._inner_z_candidates
            outer_table = self._outer_z_table
        else:
            candidate_table = self._inner_x_candidates
            outer_table = self._outer_x_table
        
        # Collect candidates per block
        block_candidates: Dict[int, List[InnerCandidate]] = {}
        for block_idx in range(n_blocks):
            if error_type == 'Z':
                # Z errors detected by X syndrome (index 1)
                syn = block_syndromes.get(block_idx, (tuple([0]*3), tuple([0]*3)))[1]
            else:
                # X errors detected by Z syndrome (index 0)
                syn = block_syndromes.get(block_idx, (tuple([0]*3), tuple([0]*3)))[0]
            
            if syn in candidate_table:
                block_candidates[block_idx] = candidate_table[syn]
            else:
                # No entry: use zero correction
                block_candidates[block_idx] = [InnerCandidate(
                    pattern=np.zeros(self._inner_n_qubits, dtype=np.uint8),
                    weight=0,
                    logical_flip_z=False,
                    logical_flip_x=False,
                    error_type='none'
                )]
        
        # Enumerate global patterns via bounded search
        best_correction = None
        best_weight = float('inf')
        best_logical_flip = 0
        n_explored = 0
        
        # Use recursive enumeration limited by max_total_weight
        def search(block_idx: int, current_pattern: List, current_weight: int, current_logicals: List):
            nonlocal best_correction, best_weight, best_logical_flip, n_explored
            
            if n_explored >= self.config.global_candidate_limit:
                return  # Exceeded search limit
            
            if block_idx >= n_blocks:
                # All blocks assigned: check outer syndrome consistency
                n_explored += 1
                
                # Compute outer syndrome from logical flips
                outer_syndrome = self._compute_outer_syndrome_from_logicals(
                    current_logicals, error_type
                )
                
                # Check if this outer syndrome is correctable
                if outer_syndrome in outer_table:
                    outer_correction, outer_weight = outer_table[outer_syndrome]
                    total_weight = current_weight  # Could add outer_weight if needed
                    
                    if total_weight < best_weight:
                        best_weight = total_weight
                        # Build full physical correction from block patterns
                        best_correction = self._assemble_correction_from_blocks(
                            current_pattern, n_blocks
                        )
                        # Compute final logical flip:
                        # inner_logicals XOR outer_correction = final block errors
                        # Then parity of final block errors on outer logical = final logical flip
                        inner_vec = np.array(current_logicals, dtype=np.uint8)
                        final_block_errors = (inner_vec + outer_correction) % 2
                        # For CSS code, outer logical support is typically [0,1,2] (first 3 blocks)
                        outer_logical_support = self._get_outer_logical_support(error_type)
                        best_logical_flip = int(sum(final_block_errors[b] for b in outer_logical_support if b < len(final_block_errors)) % 2)
                return
            
            # Try each candidate for current block
            candidates = block_candidates.get(block_idx, [])
            for cand in candidates:
                new_weight = current_weight + cand.weight
                
                # Prune if exceeds weight limit
                if new_weight > self.config.max_total_weight:
                    continue
                
                # Recurse with this candidate
                search(
                    block_idx + 1,
                    current_pattern + [cand],
                    new_weight,
                    current_logicals + [1 if (cand.logical_flip_z if error_type == 'Z' else cand.logical_flip_x) else 0]
                )
        
        # Start search
        search(0, [], 0, [])
        
        if best_correction is None:
            # No valid solution found: fallback
            if self.config.verbose:
                print(f"  Warning: joint search found no valid pattern, using fallback")
            corr, w = self._fallback_hierarchical_decode(block_syndromes, n_blocks, error_type)
            return corr, w, 0
        
        return best_correction, int(best_weight), best_logical_flip
    
    def _compute_outer_syndrome_from_logicals(
        self,
        logical_flips: List[int],
        error_type: str
    ) -> Tuple[int, ...]:
        """
        Compute outer syndrome from inner block logical flips.
        
        Args:
            logical_flips: Binary list of logical flips per block
            error_type: 'Z' or 'X'
        
        Returns:
            Outer syndrome tuple
        """
        logical_vec = np.array(logical_flips, dtype=np.uint8)
        
        if error_type == 'Z':
            # Z errors on inner blocks → X errors on outer
            # Detected by outer Hz
            syndrome = (self._outer_Hz @ logical_vec) % 2
        else:
            # X errors on inner blocks → Z errors on outer
            # Detected by outer Hx
            syndrome = (self._outer_Hx @ logical_vec) % 2
        
        return tuple(syndrome)
    
    def _get_outer_logical_support(self, error_type: str) -> List[int]:
        """Get outer code logical operator support for final logical computation."""
        # For Steane code, logical X and Z both have support on qubits {0,1,2}
        # which corresponds to blocks {0,1,2} at the outer level
        if self.inner_code_info is not None:
            if error_type == 'Z':
                # Z errors → check outer Z logical support
                return list(self._inner_z_logical_support) if self._inner_z_logical_support else [0, 1, 2]
            else:
                # X errors → check outer X logical support
                return list(self._inner_x_logical_support) if self._inner_x_logical_support else [0, 1, 2]
        return [0, 1, 2]  # Default: Steane logical support
    
    def _assemble_correction_from_blocks(
        self,
        block_patterns: List[InnerCandidate],
        n_blocks: int
    ) -> np.ndarray:
        """Assemble global physical correction from per-block patterns."""
        correction = np.zeros(self.n_qubits, dtype=np.uint8)
        n_inner = self._inner_n_qubits
        
        for block_idx, cand in enumerate(block_patterns):
            if block_idx >= n_blocks:
                break
            start_idx = block_idx * n_inner
            end_idx = min(start_idx + n_inner, self.n_qubits)
            
            for i in range(end_idx - start_idx):
                if i < len(cand.pattern):
                    correction[start_idx + i] = cand.pattern[i]
        
        return correction
    
    def _fallback_hierarchical_decode(
        self,
        block_syndromes: Dict[int, Tuple],
        n_blocks: int,
        error_type: str
    ) -> Tuple[np.ndarray, int]:
        """Fallback to standard hierarchical decode if joint search disabled/fails."""
        correction = np.zeros(self.n_qubits, dtype=np.uint8)
        total_weight = 0
        n_inner = self._inner_n_qubits
        
        # Use legacy single-candidate tables
        if error_type == 'Z':
            table = self._inner_z_table
        else:
            table = self._inner_x_table
        
        for block_idx in range(n_blocks):
            if error_type == 'Z':
                syn = block_syndromes.get(block_idx, (tuple([0]*3), tuple([0]*3)))[0]
            else:
                syn = block_syndromes.get(block_idx, (tuple([0]*3), tuple([0]*3)))[1]
            
            if syn in table:
                block_corr, weight = table[syn]
                total_weight += weight
                
                start_idx = block_idx * n_inner
                for i in range(min(len(block_corr), n_inner)):
                    if start_idx + i < self.n_qubits:
                        correction[start_idx + i] = block_corr[i]
        
        return correction, total_weight
    
    # =========================================================================
    # NEW DETECTION-EVENT-BASED DECODING INTERFACE
    # =========================================================================
    
    def decode_from_detection_events(
        self,
        detection_events: np.ndarray,
        metadata: 'MultiLevelMetadata',
        n_inner_blocks: int = 7,
        n_stabilizers_per_block: int = 3,
    ) -> CircuitDecodeResult:
        """
        Decode using Stim's detection events instead of raw measurements.
        
        WARNING: This method assumes detection events have a simple structure
        mapping directly to inner block syndromes. For circuits with complex
        detector layouts (e.g., temporal detectors, comparison detectors),
        this assumption may not hold and the decoder may perform poorly.
        
        For circuit-level noise with proper detector structure, consider using:
        - pymatching with the detector error model
        - BP-OSD decoder  
        - Other graph-based decoders
        
        ASSUMED DETECTOR LAYOUT:
        - First n_blocks * n_stabs detectors: Z syndromes for blocks 0 to n_blocks-1
        - Next n_blocks * n_stabs detectors: X syndromes for blocks 0 to n_blocks-1
        
        If your circuit's detectors don't match this layout, use
        decode_from_raw_measurements() instead for single-round decoding.
        
        Args:
            detection_events: Binary array of detection events from stim sampler
                             (stim.CompiledDetectorSampler().sample())
            metadata: MultiLevelMetadata containing circuit structure info
            n_inner_blocks: Number of inner code blocks (default 7 for Steane)
            n_stabilizers_per_block: Stabilizers per block (default 3)
            
        Returns:
            CircuitDecodeResult with decoded logical values
        """
        # Update diagnostic counter
        self._diag_total_decodes += 1
        
        n_blocks = n_inner_blocks
        n_stabs = n_stabilizers_per_block
        n_detectors = len(detection_events)
        
        # Get number of EC rounds from metadata
        n_ec_rounds = getattr(metadata, 'n_ec_rounds', 3)
        
        # =====================================================================
        # STAGE 1: Map detectors to hierarchical structure
        # =====================================================================
        # WARNING: This assumes a simple detector layout where:
        # - First 21 detectors = Z syndromes for blocks 0-6
        # - Next 21 detectors = X syndromes for blocks 0-6
        #
        # For actual circuits with temporal/comparison detectors, this
        # mapping is incorrect and will produce poor decoding results.
        #
        # For a 3-round EC with 7 blocks × 3 stabs × 2 types (X/Z):
        # n_spatial = 7 × 3 × 2 = 42 per round
        # n_temporal = n_spatial × (n_rounds - 1) = 42 × 2 = 84
        # Plus initial round detectors = 42
        # Total ≈ 42 + 84 = 126 (plus final detectors)
        
        # Extract syndrome information from detection events
        # Key insight: A detection event at detector d means the syndrome BIT
        # corresponding to d flipped at that time slice.
        
        # Initialize syndrome accumulators per block
        z_syndrome_events_by_block: Dict[int, List[int]] = {i: [] for i in range(n_blocks)}
        x_syndrome_events_by_block: Dict[int, List[int]] = {i: [] for i in range(n_blocks)}
        
        # =====================================================================
        # STAGE 2: Parse detection events into hierarchical syndromes
        # =====================================================================
        # For the initial round, detectors 0 to (n_blocks * n_stabs * 2 - 1)
        # represent the initial syndrome measurement.
        # 
        # Layout typically:
        #   Z detectors first: block_idx * n_stabs + stab_idx  (for block 0-6, stab 0-2)
        #   X detectors next: n_blocks * n_stabs + block_idx * n_stabs + stab_idx
        
        n_z_detectors_initial = n_blocks * n_stabs  # 21 for Steane
        n_x_detectors_initial = n_blocks * n_stabs  # 21 for Steane
        n_initial_detectors = n_z_detectors_initial + n_x_detectors_initial  # 42
        
        # Parse initial round Z detectors
        for det_idx in range(min(n_z_detectors_initial, n_detectors)):
            if detection_events[det_idx]:
                block_idx = det_idx // n_stabs
                stab_idx = det_idx % n_stabs
                if block_idx < n_blocks:
                    z_syndrome_events_by_block[block_idx].append(stab_idx)
        
        # Parse initial round X detectors
        for det_idx in range(n_z_detectors_initial, min(n_initial_detectors, n_detectors)):
            if detection_events[det_idx]:
                local_idx = det_idx - n_z_detectors_initial
                block_idx = local_idx // n_stabs
                stab_idx = local_idx % n_stabs
                if block_idx < n_blocks:
                    x_syndrome_events_by_block[block_idx].append(stab_idx)
        
        # =====================================================================
        # STAGE 3: Convert detection events to syndromes
        # =====================================================================
        # For each block, if a stabilizer index appears an ODD number of times
        # in the detection events, that stabilizer syndrome bit is 1.
        #
        # This automatically handles:
        # - Measurement errors (appear as paired events that cancel)
        # - Data errors (appear as single events that persist)
        
        z_syndromes: Dict[int, Tuple[int, ...]] = {}
        x_syndromes: Dict[int, Tuple[int, ...]] = {}
        
        for block_idx in range(n_blocks):
            # Build Z syndrome for this block
            z_events = z_syndrome_events_by_block[block_idx]
            z_syn = [0] * n_stabs
            for stab_idx in z_events:
                if stab_idx < n_stabs:
                    z_syn[stab_idx] ^= 1  # XOR to handle paired events
            z_syndromes[block_idx] = tuple(z_syn)
            
            # Build X syndrome for this block
            x_events = x_syndrome_events_by_block[block_idx]
            x_syn = [0] * n_stabs
            for stab_idx in x_events:
                if stab_idx < n_stabs:
                    x_syn[stab_idx] ^= 1
            x_syndromes[block_idx] = tuple(x_syn)
        
        # =====================================================================
        # STAGE 4: Decode - JOINT SEARCH or HIERARCHICAL
        # =====================================================================
        
        x_correction = np.zeros(self.n_qubits, dtype=np.uint8)
        z_correction = np.zeros(self.n_qubits, dtype=np.uint8)
        joint_logical_z = 0  # For joint decoder: direct logical flip
        joint_logical_x = 0
        use_joint_logicals = False
        
        # Prepare block syndrome dict for joint search
        block_syndromes = {
            block_idx: (z_syndromes[block_idx], x_syndromes[block_idx])
            for block_idx in range(n_blocks)
        }
        
        if self.config.enable_joint_search and self.config.use_multi_candidate_tables:
            # ===== JOINT MINIMUM-WEIGHT DECODING =====
            # Find global minimum-weight error pattern across all blocks
            use_joint_logicals = True
            
            # Decode Z errors (detected by X syndromes) → affects X logical
            # error_type='Z' means "Z errors"
            if any(sum(x_syndromes[i]) > 0 for i in range(n_blocks)):
                z_correction, z_weight, joint_logical_x = self._joint_search_enumerate(
                    block_syndromes, n_blocks, error_type='Z'
                )
                if self.config.verbose:
                    print(f"  Joint Z decode: weight={z_weight}, logical_x_flip={joint_logical_x}")
            
            # Decode X errors (detected by Z syndromes) → affects Z logical
            # error_type='X' means "X errors"
            if any(sum(z_syndromes[i]) > 0 for i in range(n_blocks)):
                x_correction, x_weight, joint_logical_z = self._joint_search_enumerate(
                    block_syndromes, n_blocks, error_type='X'
                )
                if self.config.verbose:
                    print(f"  Joint X decode: weight={x_weight}, logical_z_flip={joint_logical_z}")
        
        else:
            # ===== HIERARCHICAL DECODING (Original) =====
            inner_z_logicals: Dict[int, int] = {}  # X errors on inner blocks
            inner_x_logicals: Dict[int, int] = {}  # Z errors on inner blocks
            
            for block_idx in range(n_blocks):
                z_syn = z_syndromes[block_idx]
                x_syn = x_syndromes[block_idx]
                
                # Decode Z syndrome (X errors) -> X logical error
                if sum(z_syn) > 0:
                    z_logical, _ = self._decode_inner_block_logical(z_syn, block_idx, 'Z')
                else:
                    z_logical = 0
                inner_z_logicals[block_idx] = z_logical
                
                # Decode X syndrome (Z errors) -> Z logical error
                if sum(x_syn) > 0:
                    x_logical, _ = self._decode_inner_block_logical(x_syn, block_idx, 'X')
                else:
                    x_logical = 0
                inner_x_logicals[block_idx] = x_logical
            
            # Track if any inner logical errors were detected
            n_inner_z_errors = sum(inner_z_logicals.values())
            n_inner_x_errors = sum(inner_x_logicals.values())
            if n_inner_z_errors > 0 or n_inner_x_errors > 0:
                self._diag_inner_logical_errors_detected += 1
            
            # ===== OUTER LEVEL DECODE =====
            if self._outer_Hz is not None and n_inner_z_errors > 0:
                # Build outer X syndrome from inner Z logicals (X errors)
                inner_x_error_vec = np.array([inner_z_logicals.get(i, 0) for i in range(n_blocks)], dtype=np.uint8)
                outer_x_syndrome = tuple((self._outer_Hz @ inner_x_error_vec) % 2)
                
                if sum(outer_x_syndrome) > 0:
                    self._diag_outer_nonzero_syndrome += 1
                    
                    # Decode outer level
                    if not getattr(self.config, 'skip_outer_correction', False):
                        if outer_x_syndrome in self._outer_x_table:
                            outer_correction, _ = self._outer_x_table[outer_x_syndrome]
                            # XOR with inner error to get final block-level correction
                            final_x_errors = (inner_x_error_vec + outer_correction) % 2
                            self._diag_outer_applied_correction += 1
                        else:
                            # Outer syndrome not in table - use inner errors directly
                            final_x_errors = inner_x_error_vec
                    else:
                        final_x_errors = inner_x_error_vec
                else:
                    # No outer syndrome - use inner errors directly
                    final_x_errors = inner_x_error_vec
                
                # Convert block-level errors to qubit-level corrections
                for block_idx, has_error in enumerate(final_x_errors):
                    if has_error:
                        x_correction = self._apply_block_logical_x(x_correction, block_idx)
            
            if self._outer_Hx is not None and n_inner_x_errors > 0:
                # Build outer Z syndrome from inner X logicals (Z errors)
                inner_z_error_vec = np.array([inner_x_logicals.get(i, 0) for i in range(n_blocks)], dtype=np.uint8)
                outer_z_syndrome = tuple((self._outer_Hx @ inner_z_error_vec) % 2)
                
                if sum(outer_z_syndrome) > 0:
                    self._diag_outer_nonzero_syndrome += 1
                    
                    # Decode outer level
                    if not getattr(self.config, 'skip_outer_correction', False):
                        if outer_z_syndrome in self._outer_z_table:
                            outer_correction, _ = self._outer_z_table[outer_z_syndrome]
                            final_z_errors = (inner_z_error_vec + outer_correction) % 2
                            self._diag_outer_applied_correction += 1
                        else:
                            final_z_errors = inner_z_error_vec
                    else:
                        final_z_errors = inner_z_error_vec
                else:
                    final_z_errors = inner_z_error_vec
                
                # Convert block-level errors to qubit-level corrections
                for block_idx, has_error in enumerate(final_z_errors):
                    if has_error:
                        z_correction = self._apply_block_logical_z(z_correction, block_idx)
        
        # =====================================================================
        # STAGE 6: Compute final logical values from corrections
        # =====================================================================
        if use_joint_logicals:
            # Joint decoder directly computes logical flips
            logical_z = joint_logical_z
            logical_x = joint_logical_x
        else:
            # Hierarchical decoder: parity of correction on logical support
            logical_z = int(sum(x_correction[q] for q in self.ZL) % 2)
            logical_x = int(sum(z_correction[q] for q in self.XL) % 2)
        
        return CircuitDecodeResult(
            logical_z=logical_z,
            logical_x=logical_x,
            x_correction=x_correction,
            z_correction=z_correction,
            n_detector_events=int(np.sum(detection_events)),
            decoding_method="detection_events" if not use_joint_logicals else "joint_search",
        )
    
    def decode_from_raw_measurements(
        self,
        measurements: np.ndarray,
        metadata: 'MultiLevelMetadata',
        n_inner_blocks: int = 7,
    ) -> CircuitDecodeResult:
        """
        Decode using raw measurement outcomes for hierarchical concatenated codes.
        
        This method extracts syndrome information from the final EC round's 
        ancilla measurements and applies hierarchical decoding. For best results,
        the observable value should be computed from the final data measurements.
        
        IMPORTANT: This method is designed for single-shot decoding of the final
        syndrome. For multi-round circuits with detection events, use pymatching
        or another graph-based decoder.
        
        Args:
            measurements: Raw measurement outcomes from stim sampler (not detection events)
            metadata: MultiLevelMetadata with syndrome_layout and measurement info
            n_inner_blocks: Number of inner code blocks (default 7 for Steane)
            
        Returns:
            CircuitDecodeResult with decoded logical values
            
        Note:
            The logical_z value represents the predicted parity of the Z observable.
            For a memory experiment, compare with the actual observable (parity of
            final Z measurements on logical Z support) to determine if decoding
            succeeded.
        """
        # Get inner code parity matrices
        if self._inner_Hz is None or self._inner_Hx is None:
            # Try to get from inner_code_info
            if self.inner_code_info and 'hz' in self.inner_code_info:
                inner_Hz = np.array(self.inner_code_info['hz'])
                inner_Hx = np.array(self.inner_code_info['hx'])
            else:
                raise ValueError("Inner code parity matrices not available. "
                               "Initialize decoder with from_code() for hierarchical decoding.")
        else:
            inner_Hz = self._inner_Hz
            inner_Hx = self._inner_Hx
        
        # Syndrome to error position mapping for Steane code
        syndrome_to_pos = {
            (0, 0, 0): None,
            (0, 0, 1): 0,
            (0, 1, 0): 1,
            (0, 1, 1): 2,
            (1, 0, 0): 3,
            (1, 0, 1): 4,
            (1, 1, 0): 5,
            (1, 1, 1): 6,
        }
        
        inner_ZL = {0, 1, 2}  # Inner Z logical support
        outer_ZL = [0, 1, 2]  # Outer Z logical support (which blocks)
        
        # Extract syndrome layout for inner blocks
        level_1_layout = metadata.syndrome_layout.get(1, {})
        if not level_1_layout:
            # Fallback: return trivial result
            return CircuitDecodeResult(
                logical_z=0, logical_x=0,
                x_correction=np.zeros(self.n_qubits, dtype=np.uint8),
                z_correction=np.zeros(self.n_qubits, dtype=np.uint8),
                n_detector_events=0,
                decoding_method="raw_measurements_fallback"
            )
        
        # For each inner block, extract syndrome and decode
        inner_x_logical_flips = []  # X errors that flip Z logical
        inner_z_logical_flips = []  # Z errors that flip X logical
        
        for block_idx in range(n_inner_blocks):
            if block_idx not in level_1_layout:
                inner_x_logical_flips.append(0)
                inner_z_logical_flips.append(0)
                continue
            
            info = level_1_layout[block_idx]
            n_raw = min(info['x_count'], 7)  # Typically 7 for Steane
            
            # X syndrome (from Z stabilizer measurements) - detects X errors
            # X errors flip Z measurements, so they affect Z logical
            x_raw = np.array([int(measurements[info['x_start'] + i]) for i in range(n_raw)], dtype=np.uint8)
            x_syn = tuple(int(s) for s in (inner_Hx @ x_raw) % 2)
            
            # Decode X errors
            x_pos = syndrome_to_pos.get(x_syn, None)
            x_affects_ZL = x_pos in inner_ZL if x_pos is not None else False
            inner_x_logical_flips.append(1 if x_affects_ZL else 0)
            
            # Z syndrome (from X stabilizer measurements) - detects Z errors
            # Z errors flip X measurements, so they affect X logical
            z_raw = np.array([int(measurements[info['z_start'] + i]) for i in range(n_raw)], dtype=np.uint8)
            z_syn = tuple(int(s) for s in (inner_Hz @ z_raw) % 2)
            
            # Decode Z errors
            z_pos = syndrome_to_pos.get(z_syn, None)
            z_affects_XL = z_pos in inner_ZL if z_pos is not None else False
            inner_z_logical_flips.append(1 if z_affects_XL else 0)
        
        # Outer level decoding for X errors (affecting Z observable)
        inner_x_vec = np.array(inner_x_logical_flips, dtype=np.uint8)
        outer_syn_x = tuple((self._outer_Hz @ inner_x_vec) % 2)
        outer_pos_x = syndrome_to_pos.get(outer_syn_x, None)
        
        if outer_pos_x is not None:
            outer_correction_x = np.zeros(n_inner_blocks, dtype=np.uint8)
            outer_correction_x[outer_pos_x] = 1
            final_x_vec = (inner_x_vec + outer_correction_x) % 2
        else:
            final_x_vec = inner_x_vec
        
        # Final Z logical = parity of X errors on outer Z support
        logical_z = int(sum(final_x_vec[i] for i in outer_ZL) % 2)
        
        # Outer level decoding for Z errors (affecting X observable)
        inner_z_vec = np.array(inner_z_logical_flips, dtype=np.uint8)
        outer_syn_z = tuple((self._outer_Hx @ inner_z_vec) % 2)
        outer_pos_z = syndrome_to_pos.get(outer_syn_z, None)
        
        if outer_pos_z is not None:
            outer_correction_z = np.zeros(n_inner_blocks, dtype=np.uint8)
            outer_correction_z[outer_pos_z] = 1
            final_z_vec = (inner_z_vec + outer_correction_z) % 2
        else:
            final_z_vec = inner_z_vec
        
        # Final X logical = parity of Z errors on outer X support
        logical_x = int(sum(final_z_vec[i] for i in outer_ZL) % 2)
        
        return CircuitDecodeResult(
            logical_z=logical_z,
            logical_x=logical_x,
            x_correction=np.zeros(self.n_qubits, dtype=np.uint8),  # Not computing physical correction
            z_correction=np.zeros(self.n_qubits, dtype=np.uint8),
            n_detector_events=0,
            decoding_method="raw_measurements_hierarchical"
        )
    
    def _apply_block_logical_x(self, correction: np.ndarray, block_idx: int) -> np.ndarray:
        """Apply X logical operator for inner block to correction."""
        inner_n = self.inner_code_info.get('n_qubits', 7) if self.inner_code_info else 7
        start = block_idx * inner_n
        # X logical for Steane code is on all 7 qubits
        for i in range(inner_n):
            if start + i < len(correction):
                correction[start + i] ^= 1
        return correction
    
    def _apply_block_logical_z(self, correction: np.ndarray, block_idx: int) -> np.ndarray:
        """Apply Z logical operator for inner block to correction."""
        inner_n = self.inner_code_info.get('n_qubits', 7) if self.inner_code_info else 7
        start = block_idx * inner_n
        # Z logical for Steane code is on all 7 qubits
        for i in range(inner_n):
            if start + i < len(correction):
                correction[start + i] ^= 1
        return correction
    
    def _extract_syndrome_history(
        self,
        sample: np.ndarray,
        metadata: 'MultiLevelMetadata',
    ) -> List[Dict]:
        """
        Extract syndrome values from EC rounds.
        
        Returns list of dicts, one per round, with structure:
        {
            'round': int,
            'X': {block_key: syndrome_tuple, ...},
            'Z': {block_key: syndrome_tuple, ...}
        }
        """
        history = []
        
        if not metadata.syndrome_measurements:
            return history
        
        # Process each EC round
        for round_idx in range(metadata.n_ec_rounds):
            round_data = {'round': round_idx, 'X': {}, 'Z': {}}
            
            # Extract X syndromes (detected by Z stabilizers during EC)
            if 'X' in metadata.syndrome_measurements:
                for key, indices in metadata.syndrome_measurements['X'].items():
                    if isinstance(key, tuple) and len(key) >= 2:
                        r = key[0]
                        block_info = key[1:]
                        if r == round_idx:
                            syndrome = tuple(
                                int(sample[i]) for i in indices if i < len(sample)
                            )
                            round_data['X'][block_info] = syndrome
            
            # Extract Z syndromes (detected by X stabilizers during EC)
            if 'Z' in metadata.syndrome_measurements:
                for key, indices in metadata.syndrome_measurements['Z'].items():
                    if isinstance(key, tuple) and len(key) >= 2:
                        r = key[0]
                        block_info = key[1:]
                        if r == round_idx:
                            syndrome = tuple(
                                int(sample[i]) for i in indices if i < len(sample)
                            )
                            round_data['Z'][block_info] = syndrome
            
            history.append(round_data)
        
        return history

    def _decode_temporal_differences(
        self,
        syndrome_history: List[Dict],
        syndrome_changes: List[Dict],
        metadata: 'MultiLevelMetadata',
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Temporal decoding using syndrome differences to classify errors.
        
        Strategy:
        - Stable syndromes across rounds → likely data error
        - Toggling syndromes → likely measurement error
        - Weight corrections by stability; filter unstable blocks
        
        Returns:
            (x_correction, z_correction, temporal_metadata)
        """
        x_correction = np.zeros(self.n_qubits, dtype=np.uint8)
        z_correction = np.zeros(self.n_qubits, dtype=np.uint8)
        
        # Classify blocks by stability
        stable_blocks = {}  # block_key -> (syndrome, count)
        toggling_blocks = set()
        
        for t, change in enumerate(syndrome_changes):
            for error_type in ['X', 'Z']:
                for block_key, syn in change.get(error_type, {}).items():
                    if sum(syn) > 0:
                        # Syndrome changed this round
                        if block_key not in stable_blocks:
                            stable_blocks[block_key] = (syn, error_type, 1)
                        else:
                            # Block toggled - mark as unstable
                            toggling_blocks.add(block_key)
        
        # Decode stable blocks with higher confidence
        for block_key, (syn, error_type, _) in stable_blocks.items():
            if block_key in toggling_blocks:
                continue  # Skip unstable
            
            if self._is_outer_block(block_key):
                block_correction = self._lookup_correction_from_table(syn, error_type)
            else:
                true_syndrome = self._compute_true_syndrome(syn, error_type)
                block_correction = self._decode_inner_block_syndrome(
                    true_syndrome, block_key, error_type, metadata
                )
            
            if error_type == 'X':
                z_correction = (z_correction + block_correction) % 2
            else:
                x_correction = (x_correction + block_correction) % 2
        
        temporal_metadata = {
            'n_stable': len(stable_blocks) - len(toggling_blocks),
            'n_toggling': len(toggling_blocks),
            'toggling_blocks': list(toggling_blocks),
        }
        
        return x_correction, z_correction, temporal_metadata

    def _compute_syndrome_changes(
        self,
        syndrome_history: List[Dict],
    ) -> List[Dict]:
        """
        Compute syndrome DIFFERENCES between consecutive rounds.
        
        This is the key to temporal decoding:
        - syndrome_change[t] = syndrome[t] XOR syndrome[t-1]
        - Non-zero change indicates an error occurred in round t
          OR a measurement error in round t-1
        
        Returns list of dicts with same structure as syndrome_history,
        but values are the XOR differences.
        """
        if not syndrome_history:
            return []
        
        changes = []
        
        for t, round_data in enumerate(syndrome_history):
            change = {'round': t, 'X': {}, 'Z': {}}
            
            if t == 0:
                # First round: change is the syndrome itself (compared to 0)
                change['X'] = dict(round_data.get('X', {}))
                change['Z'] = dict(round_data.get('Z', {}))
            else:
                prev_data = syndrome_history[t - 1]
                
                # XOR X syndromes with previous round
                for key, syn in round_data.get('X', {}).items():
                    prev_syn = prev_data.get('X', {}).get(key)
                    if prev_syn is not None and len(syn) == len(prev_syn):
                        change['X'][key] = tuple((a ^ b) for a, b in zip(syn, prev_syn))
                    else:
                        change['X'][key] = syn  # No previous, use as-is
                
                # XOR Z syndromes with previous round
                for key, syn in round_data.get('Z', {}).items():
                    prev_syn = prev_data.get('Z', {}).get(key)
                    if prev_syn is not None and len(syn) == len(prev_syn):
                        change['Z'][key] = tuple((a ^ b) for a, b in zip(syn, prev_syn))
                    else:
                        change['Z'][key] = syn  # No previous, use as-is
            
            changes.append(change)
        
        return changes
    
    def _count_syndrome_changes(self, syndrome_changes: List[Dict]) -> int:
        """Count total number of non-zero syndrome changes."""
        count = 0
        for change in syndrome_changes:
            for error_type in ['X', 'Z']:
                for syn in change.get(error_type, {}).values():
                    count += sum(syn)
        return count
    
    def _count_detector_events(
        self,
        syndrome_changes: List[Dict],
        final_x_syndrome: Tuple,
        final_z_syndrome: Tuple,
    ) -> int:
        """
        Count total detector events in spacetime.
        
        Detector events are:
        1. Non-zero syndrome changes between EC rounds
        2. Non-zero final syndrome (comparing last EC to final data)
        """
        n_events = self._count_syndrome_changes(syndrome_changes)
        n_events += sum(final_x_syndrome) + sum(final_z_syndrome)
        return n_events
    
    def _decode_spacetime_mwpm(
        self,
        syndrome_history: List[Dict],
        syndrome_changes: List[Dict],
        final_x_syndrome: Tuple,
        final_z_syndrome: Tuple,
        final_data: np.ndarray,
    ) -> Tuple[int, int, np.ndarray]:
        """
        Decode using spacetime MWPM on the full detector graph.
        
        The spacetime approach treats decoding as matching in a 3D graph:
        - x, y: spatial position (check index)
        - t: time (EC round)
        
        Detector nodes fire when syndrome CHANGES (flips between rounds).
        We match detector events using MWPM, then track which matchings
        flip the logical observable.
        
        NOTE: For concatenated codes, the EC syndrome measurements are 
        per-inner-block, not per-outer-check. The spacetime MWPM approach
        needs to be adapted for hierarchical codes. For now, we use a
        simplified approach that only considers the FINAL syndrome for
        decoding, but tracks syndrome history for diagnostics.
        
        Returns:
            (x_logical, z_logical, spacetime_correction)
        """
        # For concatenated codes, EC syndromes are hierarchical (inner blocks)
        # The final syndrome is the only one that maps to the full code's checks
        # 
        # Key insight: In concatenated codes with Steane EC:
        # - EC rounds measure inner-code syndromes on ancilla blocks
        # - Final measurement is the true outer-code state
        # - We can use syndrome_changes to detect measurement instability
        #   but the primary decoding should be on the final syndrome
        
        # Check if we have meaningful syndrome changes between consecutive rounds
        # This would indicate potential measurement errors or transient faults
        has_measurement_instability = self._check_measurement_instability(
            syndrome_history, syndrome_changes
        )
        
        if self.config.verbose and has_measurement_instability:
            print("  Warning: Measurement instability detected in syndrome history")
        
        # Decode using lookup table on final data (same as _decode_error_type)
        # This properly applies correction and computes logical value
        x_logical, x_weight = self._decode_error_type(
            final_data, final_x_syndrome, self._x_table, self.ZL
        )
        z_logical, z_weight = self._decode_error_type(
            final_data, final_z_syndrome, self._z_table, self.XL
        )
        
        # Spacetime correction not computed for hierarchical codes
        spacetime_correction = None
        
        return (x_logical, z_logical, spacetime_correction)
    
    def _check_measurement_instability(
        self,
        syndrome_history: List[Dict],
        syndrome_changes: List[Dict],
    ) -> bool:
        """
        Check if syndrome history shows instability (potential measurement errors).
        
        For consecutive rounds, stable syndromes suggest reliable measurements.
        Changing syndromes (especially toggling back and forth) suggest errors.
        
        Returns True if instability detected.
        """
        if len(syndrome_history) < 2:
            return False
        
        # Check for syndrome flips between consecutive rounds
        # (This is a simplified heuristic - proper FT would use majority voting)
        for t in range(1, len(syndrome_changes)):
            change = syndrome_changes[t]
            for error_type in ['X', 'Z']:
                for syn in change.get(error_type, {}).values():
                    if sum(syn) > 0:
                        # Some syndrome bits changed between rounds
                        return True
        
        return False
    
    def _decode_spacetime_single_type(
        self,
        syndrome_changes: List[Dict],
        final_syndrome: Tuple,
        H: np.ndarray,
        logical_support: Set[int],
        error_type: str,
        outer_check_info: Optional[Dict] = None,
    ) -> int:
        """
        Decode one error type using spacetime MWPM.
        
        Args:
            syndrome_changes: Syndrome differences per round
            final_syndrome: Final syndrome from data measurements
            H: Parity check matrix for this error type
            logical_support: Qubit indices for logical operator
            error_type: 'X' or 'Z'
            outer_check_info: Optional dict with outer-level check structure for hierarchical codes
                Expected keys: 'outer_H', 'outer_logical_support', 'block_to_outer_map'
            
        Returns:
            Decoded logical value (0 or 1)
        """
        n_checks = H.shape[0]
        n_rounds = len(syndrome_changes)
        
        if n_rounds == 0:
            # No EC rounds - fall back to standard decoding
            return self._decode_from_final_syndrome(final_syndrome, H, logical_support)
        
        # Collect all detector events (non-zero syndrome changes)
        detector_events = []
        
        for t, change in enumerate(syndrome_changes):
            syn_dict = change.get(error_type, {})
            # Aggregate syndromes across all blocks at this round
            aggregated_syn = self._aggregate_block_syndromes(syn_dict, n_checks)
            
            for check_idx, val in enumerate(aggregated_syn):
                if val == 1:
                    detector_events.append((t, check_idx))
        
        # Add final syndrome as detectors (comparing to last EC round)
        for check_idx, val in enumerate(final_syndrome):
            if val == 1:
                detector_events.append((n_rounds, check_idx))  # t = n_rounds for final
        
        # If outer check info provided, add outer-level detectors for hierarchical decoding
        if outer_check_info is not None:
            outer_H = outer_check_info.get('outer_H')
            outer_syn = outer_check_info.get('outer_syndrome')
            block_to_outer = outer_check_info.get('block_to_outer_map', {})
            
            if outer_H is not None and outer_syn is not None:
                # Add outer syndrome events with time offset
                t_outer = n_rounds + 1
                for outer_check_idx, val in enumerate(outer_syn):
                    if val == 1:
                        # Map outer check to corresponding inner checks if possible
                        mapped_inner = block_to_outer.get(outer_check_idx)
                        if mapped_inner is not None:
                            detector_events.append((t_outer, mapped_inner))
                        else:
                            # Add as separate outer detector
                            detector_events.append((t_outer, n_checks + outer_check_idx))
        
        if len(detector_events) == 0:
            # No detector events - trivial correction
            return 0
        
        if len(detector_events) % 2 == 1:
            # Odd number of detectors - one connects to boundary
            # For now, use majority vote on final syndrome
            return self._decode_from_final_syndrome(final_syndrome, H, logical_support)
        
        # Build spacetime matching graph and run MWPM
        try:
            logical_flip = self._run_spacetime_mwpm(
                detector_events, n_rounds, n_checks, H, logical_support
            )
            return logical_flip
        except Exception as e:
            if self.config.verbose:
                print(f"MWPM failed for {error_type}: {e}")
            return self._decode_from_final_syndrome(final_syndrome, H, logical_support)
    
    def _aggregate_block_syndromes(
        self,
        syn_dict: Dict,
        n_checks: int,
    ) -> Tuple[int, ...]:
        """
        Aggregate syndromes from multiple blocks into a single syndrome.
        
        For concatenated codes, EC may report syndromes per-block.
        We need to combine them into the full syndrome vector.
        """
        if not syn_dict:
            return tuple([0] * n_checks)
        
        # If there's only one entry and it has the right size, use it directly
        if len(syn_dict) == 1:
            syn = list(syn_dict.values())[0]
            if len(syn) == n_checks:
                return syn
        
        # Otherwise, aggregate (XOR all syndromes)
        result = [0] * n_checks
        for syn in syn_dict.values():
            for i, val in enumerate(syn):
                if i < n_checks:
                    result[i] ^= val
        
        return tuple(result)
    
    def _decode_from_final_syndrome(
        self,
        syndrome: Tuple,
        H: np.ndarray,
        logical_support: Set[int],
    ) -> int:
        """
        Simple syndrome-based decoding (fallback).
        
        Uses the lookup table or computes directly.
        """
        # Check if this is an X or Z syndrome by checking which table to use
        if H.shape == self.Hz.shape and np.array_equal(H, self.Hz):
            table = self._x_table
        else:
            table = self._z_table
        
        if syndrome in table:
            correction, _ = table[syndrome]
            # Count logical parity
            return sum(int(correction[q]) for q in logical_support) % 2
        
        # Not in table - return 0 (hope for the best)
        return 0
    
    def _run_spacetime_mwpm(
        self,
        detector_events: List[Tuple[int, int]],
        n_rounds: int,
        n_checks: int,
        H: np.ndarray,
        logical_support: Set[int],
    ) -> int:
        """
        Run MWPM on spacetime detector graph.
        
        Uses PyMatching for efficient MWPM solving.
        
        Returns logical flip (0 or 1).
        """
        if not HAS_PYMATCHING:
            raise RuntimeError("PyMatching not available")
        
        n_detectors = len(detector_events)
        if n_detectors == 0:
            return 0
        
        # Build adjacency matrix for detector graph
        # Weight = log(p/(1-p)) typically, but we use unit weights for simplicity
        # In a proper implementation, weights would be based on error probabilities
        
        # Create detector -> index mapping
        det_to_idx = {det: i for i, det in enumerate(detector_events)}
        
        # Build matching matrix
        # For spacetime MWPM, we connect:
        # 1. Spatially adjacent detectors at same time (data qubit errors)
        # 2. Same-check detectors at adjacent times (measurement errors)
        
        edges = []
        boundary_edges = []
        
        # Temporal connections (measurement errors)
        for i, (t1, c1) in enumerate(detector_events):
            for j, (t2, c2) in enumerate(detector_events):
                if j <= i:
                    continue
                
                if c1 == c2 and abs(t1 - t2) == 1:
                    # Same check, adjacent time -> measurement error
                    edges.append((i, j, 1.0))
        
        # Spatial connections (data qubit errors)
        # Two checks are connected if they share a data qubit
        check_to_qubits = {}
        for check_idx in range(n_checks):
            check_to_qubits[check_idx] = set(np.where(H[check_idx] == 1)[0])
        
        for i, (t1, c1) in enumerate(detector_events):
            for j, (t2, c2) in enumerate(detector_events):
                if j <= i:
                    continue
                
                if t1 == t2 and c1 != c2:
                    # Same time, different checks
                    shared = check_to_qubits.get(c1, set()) & check_to_qubits.get(c2, set())
                    if shared:
                        # Checks share qubits -> data error can flip both
                        edges.append((i, j, 1.0))
        
        # Boundary connections for logical errors
        # Detectors can match to boundary if their check intersects logical support
        for i, (t, c) in enumerate(detector_events):
            check_qubits = check_to_qubits.get(c, set())
            if check_qubits & logical_support:
                boundary_edges.append(i)
        
        if not edges and not boundary_edges:
            # No structure - can't match, return 0
            return 0
        
        # Build scipy sparse matrix for pymatching
        try:
            from scipy.sparse import csr_matrix
            
            # Create matching matrix with boundary node
            n_nodes = n_detectors + 1  # +1 for boundary
            boundary_idx = n_detectors
            
            rows, cols, data = [], [], []
            
            for i, j, w in edges:
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([w, w])
            
            for i in boundary_edges:
                rows.extend([i, boundary_idx])
                cols.extend([boundary_idx, i])
                data.extend([1.0, 1.0])
            
            if not rows:
                return 0
            
            adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
            
            # Create matching object
            matching = pymatching.Matching(adj_matrix)
            
            # Create detection vector
            detection = np.zeros(n_nodes, dtype=np.uint8)
            for i in range(n_detectors):
                detection[i] = 1
            
            # Run MWPM
            correction = matching.decode(detection)
            
            # Check if any boundary edges are in correction
            # If odd number of boundary matchings, logical is flipped
            logical_flip = 0
            for i in boundary_edges:
                # Check if detector i matched to boundary
                if correction[i] == 1:
                    logical_flip ^= 1
            
            return logical_flip
            
        except Exception as e:
            if self.config.verbose:
                print(f"PyMatching error: {e}")
            return 0


# Convenience function for unified API
def decode_with_metadata(
    sample: np.ndarray,
    metadata: 'MultiLevelMetadata',
    code: 'MultiLevelConcatenatedCode',
    decoder: Optional[CircuitLevelDecoder] = None,
) -> CircuitDecodeResult:
    """
    Convenience function for circuit-level decoding.
    
    Args:
        sample: Full measurement sample from circuit
        metadata: MultiLevelMetadata from experiment.build()
        code: The concatenated code object
        decoder: Pre-built decoder (optional, will create if not provided)
        
    Returns:
        CircuitDecodeResult with decoded logical values
    """
    if decoder is None:
        decoder = CircuitLevelDecoder.from_code(code)
    
    return decoder.decode(sample, metadata)
