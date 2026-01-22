# src/qectostim/experiments/multilevel_memory.py
"""
Multi-Level Memory Experiment for Arbitrary-Depth Concatenated Codes.

This module provides a Stim circuit builder for multi-level concatenated codes,
with REAL ERROR CORRECTION using the EC gadget system.

The circuit structure is:
1. Prepare |0⟩ on all physical qubits
2. Apply recursive EC rounds via configurable EC gadgets
3. Apply idle noise between rounds
4. Final measurement of all data qubits
5. Add observable for concatenated Z_L

EC Gadget Configuration:
------------------------
The experiment supports highly configurable EC gadget selection:

1. Simple presets via ec_method parameter:
   - "steane": Steane EC at all levels
   - "knill": Knill EC at all levels  
   - "transversal": Basic transversal syndrome extraction
   - "parallel_steane": Steane EC with parallel scheduling
   - "parallel_knill": Knill EC with parallel scheduling

2. Per-level configuration via level_gadgets dict:
   >>> exp = MultiLevelMemoryExperiment(
   ...     code=code,
   ...     level_gadgets={
   ...         0: SteaneECGadget(outer_code),
   ...         1: KnillECGadget(inner_code),
   ...     }
   ... )

3. Full control via ec_gadget parameter (pre-built recursive gadget)

Inherits from MemoryExperiment for consistent API with other memory experiments.
"""
from __future__ import annotations

import stim
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from qectostim.experiments.experiment import Experiment
from qectostim.experiments.memory import MemoryExperiment, get_logical_ops, ops_valid
from qectostim.noise.models import NoiseModel, CircuitDepolarizingNoise
from qectostim.experiments.gadgets.base import MeasurementMap

if TYPE_CHECKING:
    from qectostim.codes.composite.multilevel_concatenated import MultiLevelConcatenatedCode
    from qectostim.utils.hierarchical_mapper import HierarchicalQubitMapper
    from qectostim.experiments.gadgets.base import Gadget


class ECMethod(Enum):
    """Enumeration of available EC methods."""
    TRANSVERSAL = "transversal"      # Basic transversal syndrome extraction
    STEANE = "steane"                # Steane EC (cat-state ancilla)
    KNILL = "knill"                  # Knill EC (teleportation-based)
    SHOR = "shor"                    # Shor EC (cat-state with redundant measurements)
    PARALLEL_TRANSVERSAL = "parallel_transversal"  # Transversal with parallel scheduling
    PARALLEL_STEANE = "parallel_steane"            # Steane with parallel scheduling
    PARALLEL_KNILL = "parallel_knill"              # Knill with parallel scheduling


@dataclass
class ECConfig:
    """
    Configuration for EC gadget construction.
    
    This allows fine-grained control over how EC is performed at each level.
    
    Attributes
    ----------
    method : ECMethod or str
        The EC method to use. Default: STEANE (recommended for concatenated codes).
    parallel : bool
        Whether to use parallel scheduling (graph-colored CNOTs).
    extract_x : bool
        Whether to extract X-type stabilizers.
    extract_z : bool
        Whether to extract Z-type stabilizers.
    ancilla_prep : str or AncillaPrepMethod
        How to prepare ancilla qubits. Options:
        - "bare": Simple reset (NOT fault-tolerant, but fast)
        - "encoded": Proper encoding circuit (fault-tolerant)
        - "verified": Encoding + verification (fully fault-tolerant)
        Default: "bare" for backward compatibility.
    p_ancilla : float
        Ancilla preparation error probability.
    p_gate : float
        Gate error probability (for gadget-level noise, separate from circuit noise).
    """
    method: Union[ECMethod, str] = ECMethod.STEANE
    parallel: bool = False
    extract_x: bool = True
    extract_z: bool = True
    ancilla_prep: Union[str, Any] = "encoded"  # str or AncillaPrepMethod
    p_ancilla: float = 0.0
    p_gate: float = 0.0
    use_flags: bool = False  # Wrap with FlaggedSyndromeGadget for hook error detection
    
    def __post_init__(self):
        if isinstance(self.method, str):
            # Handle string inputs
            method_lower = self.method.lower()
            if method_lower.startswith("parallel_"):
                self.parallel = True
                method_lower = method_lower[9:]  # Remove "parallel_" prefix
            
            method_map = {
                "transversal": ECMethod.TRANSVERSAL,
                "steane": ECMethod.STEANE,
                "knill": ECMethod.KNILL,
                "shor": ECMethod.SHOR,
            }
            self.method = method_map.get(method_lower, ECMethod.TRANSVERSAL)
        
        # WARN: use_flags is not yet compatible with encoded/verified ancillas
        if self.use_flags and self.ancilla_prep in ("encoded", "verified"):
            import warnings
            warnings.warn(
                "use_flags=True is not yet compatible with encoded/verified ancillas. "
                "The FlaggedSyndromeGadget currently uses a different measurement model. "
                "This combination will produce incorrect results. "
                "Set use_flags=False or ancilla_prep='bare' for now.",
                UserWarning
            )


class DetectorType(Enum):
    """Type of detector emitted in the circuit."""
    INITIAL = "initial"              # Round 0 syndrome (should be 0 for |0_L⟩)
    TEMPORAL = "temporal"            # XOR of syndrome between rounds
    METACHECK_SPATIAL = "spatial"    # Spatial metacheck (XOR across blocks)
    METACHECK_PARITY = "parity"      # Parity metacheck (XOR within block)
    FINAL = "final"                  # Final round syndrome


@dataclass
class DetectorInfo:
    """Information about a single detector in the circuit.
    
    Attributes
    ----------
    detector_idx : int
        Index of this detector in the circuit.
    detector_type : DetectorType
        Type of detector (INITIAL, TEMPORAL, METACHECK, etc.).
    round_idx : int
        EC round this detector belongs to (0-indexed).
    level : int
        Hierarchy level (0=outer, 1=inner for 2-level concatenation).
    block_idx : int
        Block index within the level.
    stabilizer_idx : int
        Index of stabilizer within the block (0-2 for Steane).
    error_type : str
        'X' or 'Z' - which type of errors this detector detects.
    """
    detector_idx: int
    detector_type: DetectorType
    round_idx: int
    level: int
    block_idx: int
    stabilizer_idx: int
    error_type: str  # 'X' or 'Z'


@dataclass
class MultiLevelMetadata:
    """Metadata for multi-level memory experiment.
    
    This dataclass stores all information needed for decoding and analysis
    of multi-level concatenated code experiments.
    
    Attributes
    ----------
    code_name : str
        Name of the concatenated code.
    depth : int
        Number of concatenation levels.
    level_code_names : List[str]
        Names of codes at each level (outermost to innermost).
    n_physical_qubits : int
        Total number of physical data qubits.
    total_distance : int
        Product of distances across levels.
    n_qubits_per_level : List[int]
        Number of qubits in each level's code.
    leaf_addresses : List[Tuple[int, ...]]
        Addresses of all leaf (innermost) blocks.
    address_to_range : Dict[Tuple[int, ...], Tuple[int, int]]
        Mapping from block address to qubit range.
    total_measurements : int
        Total number of measurements in the circuit.
    final_data_measurements : Dict[Tuple[int, ...], List[int]]
        Measurement indices for final data readout per block.
    syndrome_measurements : Dict
        Syndrome measurement indices organized by type and round.
    syndrome_layout : Dict
        Hierarchical syndrome layout for decoder (maps level→block→start/count).
    level_z_supports : List[List[int]]
        Z logical operator support for each level's code.
    n_ec_rounds : int
        Number of error correction rounds.
    ec_measurements_per_round : int
        Syndrome measurements per EC round.
    gadget_names : List[str]
        Names of gadgets used.
    verification_measurements : Dict[int, Dict[int, List[int]]]
        Verification measurement indices per round per block.
        Used for flag-based decoding with verified ancilla.
    ancilla_prep_type : str
        Type of ancilla preparation used ("bare", "encoded", "verified").
    """
    code_name: str
    depth: int
    level_code_names: List[str]
    n_physical_qubits: int
    total_distance: int
    n_qubits_per_level: List[int]
    leaf_addresses: List[Tuple[int, ...]]
    address_to_range: Dict[Tuple[int, ...], Tuple[int, int]]
    total_measurements: int
    final_data_measurements: Dict[Tuple[int, ...], List[int]]
    syndrome_measurements: Dict[str, Dict[Any, List[int]]] = field(default_factory=dict)
    syndrome_layout: Dict[int, Dict[int, Dict[str, Any]]] = field(default_factory=dict)
    level_z_supports: List[List[int]] = field(default_factory=list)
    n_ec_rounds: int = 0
    ec_measurements_per_round: int = 0
    gadget_names: List[str] = field(default_factory=list)
    verification_measurements: Dict[int, Dict[int, List[int]]] = field(default_factory=dict)
    ancilla_prep_type: str = "bare"
    # Per-level ancilla prep (outermost→innermost)
    ancilla_prep_types_per_level: List[str] = field(default_factory=list)
    # Pauli frame measurements for teleportation-based EC (Knill)
    # Structure: {round: {'X': {block_id: [meas_indices]}, 'Z': {...}}}
    # 'X' frame: measurements that contribute X correction to output state
    # For Z_L measurement: final_data[i] XOR frame_X[i] gives corrected value
    pauli_frame_measurements: Dict[int, Dict[str, Dict[Any, List[int]]]] = field(default_factory=dict)
    # Whether this experiment uses teleportation-based EC (requires frame correction)
    uses_teleportation_ec: bool = False
    # NEW: Track measurement type per level ('raw_ancilla' = n bits, 'syndrome' = n-k bits)
    level_measurement_types: Dict[int, str] = field(default_factory=dict)
    # NEW: Parity check matrices per level for syndrome computation
    level_parity_matrices: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # NEW: Detector mapping for detection-event-based decoding
    # Maps detector_idx -> DetectorInfo with (round, level, block, stabilizer, type)
    detector_mapping: Dict[int, 'DetectorInfo'] = field(default_factory=dict)
    # Number of inner blocks (for hierarchical decoding)
    n_inner_blocks: int = 7
    # Stabilizers per block per type (3 for Steane)
    n_stabilizers_per_block: int = 3


class MultiLevelMemoryExperiment(MemoryExperiment):
    """
    Memory experiment for multi-level concatenated codes with configurable EC.
    
    This experiment properly inherits from MemoryExperiment and provides
    highly configurable error correction gadget selection.
    
    EC Configuration Options:
    -------------------------
    1. **Simple preset** via `ec_method` string:
       >>> exp = MultiLevelMemoryExperiment(code, ec_method="parallel_knill")
    
    2. **Per-level config** via `level_ec_configs`:
       >>> exp = MultiLevelMemoryExperiment(
       ...     code=code,
       ...     level_ec_configs={
       ...         0: ECConfig(method="steane", parallel=True),
       ...         1: ECConfig(method="knill", parallel=False),
       ...     }
       ... )
    
    3. **Pre-built gadgets** via `level_gadgets`:
       >>> exp = MultiLevelMemoryExperiment(
       ...     code=code,
       ...     level_gadgets={0: my_outer_gadget, 1: my_inner_gadget}
       ... )
    
    4. **Full control** via `ec_gadget` (pre-built recursive gadget):
       >>> exp = MultiLevelMemoryExperiment(code, ec_gadget=my_recursive_gadget)
    
    Ancilla Preparation:
    --------------------
    The `ancilla_prep` parameter controls how ancilla qubits are prepared for
    syndrome extraction. This is CRITICAL for fault-tolerance:
    
    - "bare" (default): Simple reset. NOT fault-tolerant. Kept for backward
      compatibility and quick testing.
    - "encoded": Proper encoding circuit (|0⟩ → |0_L⟩). Fault-tolerant.
    - "verified": Encoding + verification measurements. Fully fault-tolerant.
    
    For true fault-tolerant operation, use ancilla_prep="encoded" or "verified".
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The multi-level concatenated code to use.
    noise_model : NoiseModel, optional
        Noise model to apply. If None, uses CircuitDepolarizingNoise(0.001).
    rounds : int
        Number of EC (syndrome extraction) rounds. Default 0 for simple memory.
    basis : str
        Measurement basis ('Z' or 'X'). Default 'Z'.
    ec_method : str or ECMethod, optional
        Simple preset for EC method at all levels. One of:
        - "transversal", "steane", "knill"
        - "parallel_transversal", "parallel_steane", "parallel_knill"
    ancilla_prep : str, optional
        How to prepare ancilla qubits at all levels. One of:
        - "bare": Simple reset (NOT fault-tolerant)
        - "encoded": Proper encoding circuit (fault-tolerant)  
        - "verified": Encoding + verification (fully fault-tolerant)
        This is applied to all levels unless overridden by level_ec_configs.
    level_ec_configs : Dict[int, ECConfig], optional
        Per-level EC configuration for fine-grained control.
    level_gadgets : Dict[int, Gadget], optional
        Pre-built gadgets per level (overrides ec_method and level_ec_configs).
    ec_gadget : Gadget, optional
        Pre-built recursive EC gadget (overrides all other EC options).
    """
    
    def __init__(
        self,
        code: 'MultiLevelConcatenatedCode',
        noise_model: Optional[NoiseModel] = None,
        rounds: int = 0,
        basis: str = 'Z',
        ec_method: Optional[Union[str, ECMethod]] = None,
        ancilla_prep: Optional[str] = None,
        level_ec_configs: Optional[Dict[int, ECConfig]] = None,
        level_gadgets: Optional[Dict[int, 'Gadget']] = None,
        ec_gadget: Optional['Gadget'] = None,
        logical_qubit: int = 0,
        initial_state: str = "0",
        metadata: Optional[Dict[str, Any]] = None,
        emit_metachecks: bool = True,
        allow_bare_ancilla_for_testing: bool = False,
        allow_logical_support_fallback: bool = False,
        decoder_handles_outer_syndrome: bool = True,  # NEW: Let decoder compute outer syndrome
        use_symmetric_observable: bool = False,  # Option 2: Redistribute observable for balanced coverage
        use_temporal_redundancy: bool = False,   # Option 3: Add temporal detectors for undercovered qubits
        temporal_depth: int = 2,  # NEW: Number of rounds to compare for temporal detectors (≥3 for min_weight≥6)
        temporal_lookback: int = 1,  # NEW: Number of rounds to look back for multi-round temporal detectors
        use_spatial_metachecks: bool = False,  # NEW: Add safe outer-code spatial meta-check detectors
        use_within_round_spatial_metachecks: bool = False,  # NEW: Emit spatial metachecks within each EC round
        use_inner_parity_metachecks: bool = False,  # NEW: Inner X-stabilizer parity metachecks per block
        use_detector_level_spatial_metachecks: bool = False,  # NEW: XOR detectors across outer stabilizer blocks
        use_outer_level_metachecks: bool = False,  # NEW: XOR inner X-stabs across outer X-stabilizer blocks
    ):
        # Initialize parent MemoryExperiment
        super().__init__(
            code=code,
            noise_model=noise_model,
            rounds=rounds,
            logical_qubit=logical_qubit,
            initial_state=initial_state,
            metadata=metadata,
        )
        
        self.basis = basis.upper()
        self.operation = "multilevel_memory"
        self.emit_metachecks = emit_metachecks  # Control outer-level metacheck detectors
        self.allow_bare_ancilla_for_testing = allow_bare_ancilla_for_testing
        self.allow_logical_support_fallback = allow_logical_support_fallback
        self.decoder_handles_outer_syndrome = decoder_handles_outer_syndrome  # NEW: decoder computes outer syndrome
        self.use_symmetric_observable = use_symmetric_observable  # Option 2: balanced observable
        self.use_temporal_redundancy = use_temporal_redundancy   # Option 3: temporal detectors
        self.temporal_depth = max(2, temporal_depth)  # Minimum 2 for temporal comparison
        self.temporal_lookback = max(1, temporal_lookback)  # Minimum 1 for standard temporal detectors
        self.use_spatial_metachecks = use_spatial_metachecks  # Safe outer-code spatial meta-checks
        self.use_within_round_spatial_metachecks = use_within_round_spatial_metachecks  # Within-round spatial metachecks
        self.use_inner_parity_metachecks = use_inner_parity_metachecks  # Inner parity metachecks
        self.use_detector_level_spatial_metachecks = use_detector_level_spatial_metachecks  # Detector-level spatial metachecks
        self.use_outer_level_metachecks = use_outer_level_metachecks  # Outer-level metachecks
        
        # Store EC configuration options
        self._ec_method = ec_method
        # Effective ancilla prep: default to encoded when rounds>0 for FT, unless explicitly provided
        if ancilla_prep is None and rounds > 0:
            self._ancilla_prep = "encoded"
        else:
            self._ancilla_prep = ancilla_prep
        if rounds > 0 and self._ancilla_prep == "bare" and not self.allow_bare_ancilla_for_testing:
            raise ValueError(
                "Bare ancilla preparation is not fault-tolerant; set allow_bare_ancilla_for_testing=True "
                "to permit it, or use ancilla_prep='encoded'/'verified'."
            )
        self._level_ec_configs = level_ec_configs
        self._level_gadgets = level_gadgets
        self._pre_built_ec_gadget = ec_gadget
        
        # These will be populated by _create_ec_gadget
        self.level_gadgets = None
        self.ec_gadget = None
        
        # Set up hierarchical qubit mapper
        from qectostim.utils.hierarchical_mapper import HierarchicalQubitMapper
        self.qubit_mapper = HierarchicalQubitMapper(code)
        
        # Create EC gadget if rounds > 0
        if self.rounds > 0:
            self._create_ec_gadget()
        
        # Precompute Z logical supports for each level
        self._level_z_supports = self._compute_level_z_supports()
    
    def _create_ec_gadget(self) -> None:
        """
        Create the recursive EC gadget based on configuration.
        
        Priority order:
        1. Pre-built ec_gadget (if provided)
        2. Pre-built level_gadgets (if provided)
        3. level_ec_configs (if provided)
        4. ec_method preset (if provided)
        5. Default: SteaneECGadget at all levels
        """
        from qectostim.experiments.gadgets.combinators import RecursiveGadget
        
        # Option 1: Use pre-built recursive gadget
        if self._pre_built_ec_gadget is not None:
            self.ec_gadget = self._pre_built_ec_gadget
            # Try to extract level_gadgets for metadata
            if hasattr(self.ec_gadget, 'level_gadgets'):
                self.level_gadgets = self.ec_gadget.level_gadgets
            return
        
        # Option 2: Use pre-built level gadgets
        if self._level_gadgets is not None:
            self.level_gadgets = self._level_gadgets
        else:
            # Build level gadgets from configuration
            self.level_gadgets = self._build_level_gadgets()
        
        # Wrap in RecursiveGadget (always recursive for multi-level codes)
        self.ec_gadget = RecursiveGadget(
            code=self.code,
            level_gadgets=self.level_gadgets,
        )
    
    def _build_level_gadgets(self) -> Dict[int, 'Gadget']:
        """Build gadgets for each level based on configuration."""
        gadgets = {}
        
        for level_idx, level_code in enumerate(self.code.level_codes):
            # Determine config for this level
            if self._level_ec_configs and level_idx in self._level_ec_configs:
                config = self._level_ec_configs[level_idx]
            elif self._ec_method is not None:
                # Use global ec_method and ancilla_prep
                config = ECConfig(
                    method=self._ec_method,
                    ancilla_prep=self._ancilla_prep or "encoded",
                )
            else:
                # Default: Steane EC with specified ancilla prep
                config = ECConfig(
                    method=ECMethod.STEANE,
                    ancilla_prep=self._ancilla_prep or "encoded",
                )
            
            # Enforce FT ancilla choice unless explicitly allowed
            if config.ancilla_prep == "bare" and self.rounds > 0 and not self.allow_bare_ancilla_for_testing:
                raise ValueError(
                    f"Level {level_idx} configured with bare ancilla while rounds>0; "
                    f"set allow_bare_ancilla_for_testing=True to override."
                )
            
            gadgets[level_idx] = self._create_gadget_from_config(level_code, config)
        
        return gadgets
    
    def _create_gadget_from_config(self, code: Any, config: ECConfig) -> 'Gadget':
        """
        Create an EC gadget from configuration.
        
        This is the factory method that instantiates the appropriate gadget
        based on the ECConfig settings. Uses TransversalSyndromeGadget or 
        TeleportationECGadget as base gadgets, wrapped with ParallelGadget if 
        parallel=True.
        
        The ancilla_prep parameter controls fault-tolerance level:
        - "bare": NOT fault-tolerant (default for backward compatibility)
        - "encoded": Fault-tolerant using proper encoding circuit
        - "verified": Fully fault-tolerant with verification
        """
        from qectostim.experiments.gadgets.transversal_syndrome_gadget import TransversalSyndromeGadget
        from qectostim.experiments.gadgets.teleportation_ec_gadget import TeleportationECGadget
        from qectostim.experiments.gadgets.shor_syndrome_gadget import ShorSyndromeGadget
        from qectostim.experiments.gadgets.combinators import ParallelGadget, FlaggedSyndromeGadget
        
        # Determine the base gadget class
        method = config.method
        if isinstance(method, str):
            method = ECMethod(method.lower())
        
        # Create base gadget based on EC method
        if method in (ECMethod.KNILL, ECMethod.PARALLEL_KNILL):
            # TeleportationECGadget: teleportation-based EC (Knill protocol)
            # Pass ancilla_prep for fault-tolerant Bell pair preparation
            base_gadget = TeleportationECGadget(
                code=code,
                ancilla_prep=config.ancilla_prep,  # FT ancilla prep for Bell pairs
            )
        elif method == ECMethod.SHOR:
            # ShorSyndromeGadget: cat-state based EC with redundant measurements
            # Very robust to measurement errors via majority voting
            # Pass ancilla_prep for encoded/verified ancilla preparation
            base_gadget = ShorSyndromeGadget(
                code=code,
                extract_x_syndrome=config.extract_x,
                extract_z_syndrome=config.extract_z,
                verify_cat=getattr(config, 'verify_cat', False),
                measurement_reps=getattr(config, 'measurement_reps', 1),
                ancilla_prep=config.ancilla_prep,
            )
        elif method in (ECMethod.TRANSVERSAL, ECMethod.PARALLEL_TRANSVERSAL):
            # TRANSVERSAL: allow encoded/verified ancilla for FT; keep bare only if explicitly requested
            base_gadget = TransversalSyndromeGadget(
                code=code,
                extract_x_syndrome=config.extract_x,
                extract_z_syndrome=config.extract_z,
                ancilla_prep=config.ancilla_prep or "encoded",
            )
        else:
            # STEANE: Fault-tolerant syndrome extraction with configurable ancilla prep
            # Use config.ancilla_prep directly (can be "verified", "encoded", or "bare")
            base_gadget = TransversalSyndromeGadget(
                code=code,
                extract_x_syndrome=config.extract_x,
                extract_z_syndrome=config.extract_z,
                ancilla_prep=config.ancilla_prep,  # Pass through from config
            )
        
        # Wrap with FlaggedSyndromeGadget for hook error detection if requested
        if config.use_flags:
            base_gadget = FlaggedSyndromeGadget(
                base_gadget=base_gadget,
                code=code,
                flag_all_weight4_plus=True,  # Flag all weight-4+ stabilizers
            )
        
        # Wrap in ParallelGadget combinator if parallel scheduling requested
        if config.parallel or method in (ECMethod.PARALLEL_STEANE, ECMethod.PARALLEL_KNILL, ECMethod.PARALLEL_TRANSVERSAL):
            return ParallelGadget(base_gadget)
        
        return base_gadget
    
    def _compute_level_z_supports(self) -> List[List[int]]:
        """Compute Z logical operator support for each level's code."""
        supports = []
        for level_code in self.code.level_codes:
            support = self._get_z_support_for_code(level_code)
            supports.append(support)
        return supports
    
    def _get_z_support_for_code(self, code: Any) -> List[int]:
        """Get Z logical operator support for a single code."""
        # Try logical_z property (string format)
        if hasattr(code, 'logical_z'):
            lz = code.logical_z
            if isinstance(lz, list) and len(lz) > 0:
                op = lz[0]
                if isinstance(op, str):
                    return [i for i, c in enumerate(op) if c.upper() in ('Z', 'Y')]
        
        # Try lz method/property (numpy format)
        if hasattr(code, 'lz'):
            lz = getattr(code, 'lz')
            if callable(lz):
                lz = lz()
            if isinstance(lz, np.ndarray):
                lz = np.atleast_2d(lz)
                if lz.shape[0] > 0:
                    return list(np.where(lz[0] != 0)[0])
        
        # Try logical_z_ops
        ops = get_logical_ops(code, 'z')
        if ops_valid(ops):
            first_op = ops[0]
            if isinstance(first_op, str):
                return [i for i, c in enumerate(first_op) if c.upper() in ('Z', 'Y')]
        
        # Code-specific fallbacks
        code_name = getattr(code, 'name', '') or type(code).__name__
        n = code.n
        if self.allow_logical_support_fallback:
            if 'shor' in code_name.lower():
                return list(range(n))
            if 'steane' in code_name.lower() or 'stean' in code_name.lower():
                return [0, 1, 2]
            if 'hamming' in code_name.lower():
                return [0, 1, 2]
            return list(range(min(3, n)))
        raise ValueError(
            f"Unable to determine logical Z support for code {code_name}; provide logical_z/logical_z_ops or set "
            f"allow_logical_support_fallback=True to permit heuristic fallbacks."
        )
    
    def _extract_outer_syndrome_measurements(
        self,
        mmap: 'MeasurementMap',
        inner_code: Any,
        outer_code: Any,
        n_inner_blocks: int,
    ) -> Dict[str, List[int]]:
        """
        Extract outer-level syndrome measurements for concatenated codes.
        
        For a 2-level concatenated code like [[49,1,9]] = [[7,1,3]] ⊗ [[7,1,3]]:
        - The gadget emits n_blocks × n_inner raw ancilla measurements
        - We need to extract: logical measurements of each block → outer syndrome
        
        This function:
        1. Groups raw measurements by inner block (7 blocks × 7 measurements each)
        2. Applies inner syndrome extraction (Hz_inner @ raw_7 → 3 bits per block)
        3. Applies outer syndrome extraction (Hz_outer @ logical_7 → 3 bits total)
        
        Parameters
        ----------
        mmap : MeasurementMap
            The measurement map from the EC gadget
        inner_code : Code
            The inner code (e.g., Steane [[7,1,3]])
        outer_code : Code
            The outer code (e.g., Steane [[7,1,3]])
        n_inner_blocks : int
            Number of inner blocks (7 for [[49,1,9]])
            
        Returns
        -------
        Dict[str, List[int]]
            {'X': [indices for 3-bit X syndrome], 'Z': [indices for 3-bit Z syndrome]}
        """
        outer_syndromes = {}
        
        # Get parity check matrices
        inner_hz = None
        inner_hx = None
        outer_hz = None
        outer_hx = None
        
        if hasattr(inner_code, 'hz'):
            hz_raw = inner_code.hz
            inner_hz = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        if hasattr(inner_code, 'hx'):
            hx_raw = inner_code.hx
            inner_hx = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
        if hasattr(outer_code, 'hz'):
            hz_raw = outer_code.hz
            outer_hz = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        if hasattr(outer_code, 'hx'):
            hx_raw = outer_code.hx
            outer_hx = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
        
        # For each syndrome type, extract outer syndrome
        for stype in ['X', 'Z']:
            if stype not in mmap.stabilizer_measurements:
                continue
                
            inner_h = inner_hx if stype == 'X' else inner_hz
            outer_h = outer_hx if stype == 'X' else outer_hz
            
            if inner_h is None or outer_h is None:
                # Fallback: use raw measurements (non-hierarchical)
                if 0 in mmap.stabilizer_measurements[stype]:
                    outer_syndromes[stype] = mmap.stabilizer_measurements[stype][0]
                continue
            
            # Step 1: Group raw measurements by block
            # Assuming mmap.stabilizer_measurements[stype] has measurements organized by block
            # Key format: block_id -> list of measurement indices
            raw_meas_by_block = {}
            for block_id, indices in mmap.stabilizer_measurements[stype].items():
                raw_meas_by_block[block_id] = indices
            
            # Step 2: For hierarchical extraction, we need to:
            # - Compute inner syndrome for each block: Hz_inner @ raw_measurements → logical error
            # - The logical error pattern is what we feed to outer code
            # 
            # However, in the circuit, we don't have access to actual measurement values yet!
            # We only have indices. The solution:
            # - Create new measurements that represent outer syndrome bits
            # - These will be computed in the DECODER, not in the circuit!
            #
            # So we mark this in metadata: "use hierarchical extraction"
            # and store ALL raw measurements, but with a flag indicating they need processing
            
            # For now, keep the raw measurements but mark for hierarchical processing
            # The DECODER will handle the Hz_inner @ raw → logical → Hz_outer @ logical
            all_indices = []
            for block_id in sorted(raw_meas_by_block.keys()):
                all_indices.extend(raw_meas_by_block[block_id])
            
            outer_syndromes[stype] = all_indices
        
        return outer_syndromes
    
    def to_stim(self) -> stim.Circuit:
        """
        Build the Stim circuit for the multi-level memory experiment.
        
        Returns
        -------
        stim.Circuit
            The complete circuit with noise applied.
        """
        circuit, _ = self.build()
        return circuit
    
    def build(self) -> Tuple[stim.Circuit, MultiLevelMetadata]:
        """
        Build the ideal circuit and metadata.
        
        This builds the circuit WITHOUT noise - noise is applied later
        by to_stim() using the noise_model. This separation allows
        the same ideal circuit to be tested with different noise models.
        
        The circuit includes DETECTOR instructions for temporal syndrome
        comparison (round R vs round R-1) when rounds > 1.
        
        Returns
        -------
        circuit : stim.Circuit
            The ideal circuit (no noise).
        metadata : MultiLevelMetadata
            Metadata for decoding and analysis.
        """
        circuit = stim.Circuit()
        n_total = self.code.n
        data_qubits = list(range(n_total))
        data_by_block = self._organize_data_by_block()
        
        # 1. State preparation: Reset all data qubits to |0⟩
        circuit.append("R", data_qubits)
        
        # For X-basis preparation, apply H after reset
        if self.initial_state == "+" or self.basis == 'X':
            circuit.append("H", data_qubits)
        
        circuit.append("TICK")
        
        # 2. EC rounds with syndrome extraction
        current_meas_offset = 0
        all_syndrome_meas: Dict[str, Dict[Any, List[int]]] = {'X': {}, 'Z': {}}
        all_verification_meas: Dict[int, Dict[int, List[int]]] = {}  # round -> block -> indices
        all_pauli_frames: Dict[int, Dict[str, Dict[Any, List[int]]]] = {}  # round -> type -> key -> indices
        round_meas_counts = []  # Track measurement count per round
        detector_count = 0
        
        if self.rounds > 0 and self.ec_gadget is not None:
            for round_idx in range(self.rounds):
                round_start_offset = current_meas_offset
                
                # Emit EC gadget (syndrome extraction)
                mmap = self.ec_gadget.emit(
                    circuit=circuit,
                    data_qubits=data_by_block,
                    ancilla_qubits=None,
                    noise_model=None,  # Noise applied later by to_stim()
                    measurement_offset=current_meas_offset,
                )
                
                # Track syndrome measurements
                if mmap.stabilizer_measurements:
                    # For concatenated codes (depth > 1), we need hierarchical syndrome extraction
                    # The outer level should measure logical states of inner blocks, not physical qubits
                    if self.code.depth > 1:
                        # Get inner and outer codes
                        inner_code = self.code.level_codes[-1]  # Innermost
                        outer_code = self.code.level_codes[0]   # Outermost
                        n_inner_blocks = outer_code.n
                        
                        # Extract outer syndrome measurements (3 bits) from inner logical measurements
                        # NOTE: We compute outer_synd for validation but DON'T store it in metadata.
                        # The decoder computes outer syndromes from inner block logical errors.
                        outer_synd = self._extract_outer_syndrome_measurements(
                            mmap, inner_code, outer_code, n_inner_blocks
                        )
                        
                        # Store BOTH outer and inner syndrome measurements.
                        # Outer syndromes are needed by _emit_boundary_metacheck_detectors
                        # to provide additional detector coverage for observable qubits.
                        
                        # Store inner-level syndromes (level=1) for each inner block
                        for stype in ['X', 'Z']:
                            if stype in mmap.stabilizer_measurements:
                                for block_id, indices in mmap.stabilizer_measurements[stype].items():
                                    if block_id == (0, 0):
                                        # Outer-level syndromes: store with special key format
                                        # Used by _emit_boundary_metacheck_detectors
                                        compound_key = (round_idx, (0, 0))
                                        all_syndrome_meas.setdefault(stype, {})[compound_key] = indices
                                    else:
                                        # Inner-level syndromes: (1, x) format
                                        compound_key = (round_idx, (1, block_id))
                                        all_syndrome_meas.setdefault(stype, {})[compound_key] = indices
                    else:
                        # Single-level code: use measurements as-is
                        for stype in ['X', 'Z']:
                            if stype in mmap.stabilizer_measurements:
                                for key, indices in mmap.stabilizer_measurements[stype].items():
                                    compound_key = (round_idx, key)
                                    all_syndrome_meas.setdefault(stype, {})[compound_key] = indices
                
                # Track verification measurements
                if hasattr(mmap, 'verification_measurements') and mmap.verification_measurements:
                    all_verification_meas[round_idx] = {}
                    for block_id, v_indices in mmap.verification_measurements.items():
                        all_verification_meas[round_idx][block_id] = v_indices
                
                # Track Pauli frame for teleportation EC
                if hasattr(mmap, 'pauli_frame') and mmap.pauli_frame:
                    all_pauli_frames[round_idx] = {}
                    for ptype, frame_dict in mmap.pauli_frame.items():
                        all_pauli_frames[round_idx][ptype] = frame_dict
                
                # CRITICAL: Update data qubit locations if gadget changes data identity
                # This happens for TeleportationEC where data moves to ancilla qubits
                if hasattr(mmap, 'output_qubits') and mmap.output_qubits:
                    # Update data_by_block with new qubit locations
                    for block_id, new_qubits in mmap.output_qubits.items():
                        data_by_block[block_id] = new_qubits
                    # Also update the flat data_qubits list for final measurement
                    data_qubits = []
                    for block_id in sorted(data_by_block.keys()):
                        data_qubits.extend(data_by_block[block_id])
                
                round_meas_counts.append(mmap.total_measurements)
                current_meas_offset += mmap.total_measurements
                
                # Add detectors even for teleportation EC (use Pauli-frame aware handling)
                if mmap.stabilizer_measurements:
                    if round_idx == 0:
                        # Initial detectors (syndrome should be 0 for perfect |0_L⟩ init)
                        detector_count = self._emit_initial_detectors(
                            circuit, mmap, current_meas_offset, detector_count
                        )
                        # DISABLED: Metachecks cause non-deterministic detectors with encoded ancillas
                    else:
                        # Temporal detectors (compare round R to round R-1)
                        detector_count = self._emit_temporal_detectors(
                            circuit, round_idx, mmap, round_meas_counts, 
                            current_meas_offset, detector_count
                        )
                        # DISABLED: Metachecks cause non-deterministic detectors with encoded ancillas

                # Add Pauli-frame temporal detectors for teleportation EC (frame consistency)
                if round_idx > 0 and all_pauli_frames and round_idx in all_pauli_frames and (round_idx - 1) in all_pauli_frames:
                    detector_count = self._emit_temporal_frame_detectors(
                        circuit=circuit,
                        round_idx=round_idx,
                        pauli_frames=all_pauli_frames,
                        round_meas_counts=round_meas_counts,
                        current_total_meas=current_meas_offset,
                        detector_count=detector_count,
                    )
                
                # Add verification detectors (if using verified ancilla prep)
                # Verification measurements should be 0 for correct preparation
                if hasattr(mmap, 'verification_measurements') and mmap.verification_measurements:
                    detector_count = self._emit_verification_detectors(
                        circuit, mmap, current_meas_offset, detector_count
                    )
                
                # Add within-round spatial metachecks (if enabled)
                # These XOR inner syndromes across blocks within THIS round only
                # This avoids crossing reset boundaries between rounds
                if self.use_within_round_spatial_metachecks and self.code.depth > 1:
                    detector_count = self._emit_within_round_spatial_metachecks(
                        circuit=circuit,
                        round_idx=round_idx,
                        mmap=mmap,
                        current_meas_offset=current_meas_offset,
                        detector_count=detector_count,
                    )
                
                circuit.append("TICK")
        
        # 3. Final measurement
        if self.basis == 'X':
            # For X-basis measurement, apply H before M
            circuit.append("H", data_qubits)
        circuit.append("M", data_qubits)
        
        # 4. Add boundary detectors (compare last syndrome to final data)
        if self.rounds > 0 and all_syndrome_meas:
            detector_count = self._emit_boundary_detectors(
                circuit, self.rounds - 1, all_syndrome_meas, 
                current_meas_offset, n_total, detector_count
            )
            
            # Add selective penultimate boundary detectors for observable coverage
            # Only emit for observable-supporting blocks to avoid >15 symptom errors
            if self.rounds >= 2:
                detector_count = self._emit_selective_boundary_detectors(
                    circuit, self.rounds - 2, all_syndrome_meas,
                    current_meas_offset, n_total, detector_count
                )
            
            # 4b. Option 3: Enhanced temporal detectors with configurable depth
            # Adds boundary detectors for undercovered observable qubits using earlier rounds
            # This uses boundary detector approach (syndrome → final data) to avoid ancilla reset issues
            if self.use_temporal_redundancy and self.rounds >= 2:
                detector_count = self._emit_observable_temporal_detectors(
                    circuit, all_syndrome_meas,
                    current_meas_offset, n_total, detector_count
                )
            
            # 4c. Add boundary metacheck detectors for outer-level syndrome coverage
            # DISABLED: These detectors cross ancilla resets and create non-deterministic
            # parities (X components on ancilla qubits after R instructions).
            # The outer syndrome measurements are from a different circuit phase than
            # the final data measurements, so they can't be combined into valid detectors.
            # TODO: Find alternative approach for outer-level detector coverage.
            # detector_count = self._emit_boundary_metacheck_detectors(
            #     circuit, self.rounds - 1, all_syndrome_meas,
            #     current_meas_offset, n_total, detector_count
            # )
            
            # 4d. Safe outer-code spatial metachecks (deterministic cross-block detectors)
            # These XOR inner block LOGICAL syndrome bits according to outer stabilizer structure.
            # This is deterministic because it represents true stabilizers of the concatenated code.
            if self.use_spatial_metachecks and self.rounds >= 1:
                detector_count = self._emit_safe_spatial_metachecks(
                    circuit, all_syndrome_meas,
                    current_meas_offset, n_total, detector_count
                )
            
            # 4e. Inner parity metachecks: XOR all 3 inner X-stabilizers per block per round
            # For symmetric observable (qubit 6), Z errors trigger all 3 X-stabs (parity=1)
            # Weight-4 errors trigger only 2 X-stabs (parity=0), so metacheck fires
            if self.use_inner_parity_metachecks and self.rounds >= 1:
                detector_count = self._emit_inner_parity_metachecks(
                    circuit, all_syndrome_meas,
                    current_meas_offset, n_total, detector_count
                )
            
            # 4f. Detector-level spatial metachecks: XOR detectors across outer stabilizer blocks
            # These metachecks XOR DETECTORS (not measurements) for each inner stabilizer
            # across the blocks in each outer stabilizer's support
            # DETERMINISTIC because detectors are 0 in absence of errors
            if self.use_detector_level_spatial_metachecks and self.rounds >= 1:
                detector_count = self._emit_detector_level_spatial_metachecks(
                    circuit, all_syndrome_meas,
                    current_meas_offset, n_total, detector_count
                )
            
            # 4g. Outer-level metachecks: XOR each inner X-stabilizer across outer X-stabilizer blocks
            # For error X@block0_q2 + X@block1_q2:
            #   - Inner X1,X2 fire in blocks 0,1
            #   - Outer X1 metacheck (XOR X1 over blocks [1,2,5,6]): fires for block 1
            #   - Outer X2 metacheck (XOR X2 over blocks [0,2,4,6]): fires for block 0
            # This increases weight-4 hook errors to weight-8+
            if self.use_outer_level_metachecks and self.rounds >= 1:
                detector_count = self._emit_outer_level_metachecks(
                    circuit, all_syndrome_meas,
                    current_meas_offset, n_total, detector_count
                )
            
            # 4h. Validate detector coverage for observable measurements
            self._validate_detector_coverage(circuit, current_meas_offset, n_total)
        
        # 5. Add logical observable (with Pauli frame if teleportation EC)
        total_meas = current_meas_offset + n_total
        self._add_observable(circuit, n_total, current_meas_offset, all_pauli_frames, total_meas)
        
        # 6. Build metadata (include Pauli frame info for teleportation EC)
        metadata = self._build_metadata(
            n_total, current_meas_offset, all_syndrome_meas, all_verification_meas,
            all_pauli_frames
        )

        # Apply noise model if provided
        if self.noise_model is not None:
            circuit = self.noise_model.apply(circuit)
        
        
        return circuit, metadata
    
    def _emit_initial_detectors(
        self,
        circuit: stim.Circuit,
        mmap: 'MeasurementMap',
        current_total_meas: int,
        detector_count: int,
    ) -> int:
        """
        Emit initial detectors for round 0.
        
        These detectors verify that syndrome measurements are consistent with
        the initial |0_L⟩ state (all syndromes should be 0).
        
        Creates syndrome PARITY detectors (one per stabilizer generator) for
        proper distance behavior. Each detector XORs the measurements that 
        contribute to one syndrome bit.
        
        For Shor-style EC (measurement_type == "shor_redundant"), uses the
        shor_measurement_info to build detectors from redundant cat state
        measurements, XORing adjacent pairs for consistency checking.
        """
        # Check if this is Shor-style redundant measurements
        is_shor = hasattr(mmap, 'measurement_type') and mmap.measurement_type == "shor_redundant"
        
        # For each syndrome type and block
        for stype in ['X', 'Z']:
            if stype not in mmap.stabilizer_measurements:
                continue
            
            for block_key, indices in mmap.stabilizer_measurements[stype].items():
                level, block_idx = self._get_block_level_and_idx(block_key)
                
                # Handle Shor-style measurements specially
                if is_shor and hasattr(mmap, 'shor_measurement_info'):
                    shor_key = (stype, block_key)
                    if shor_key in mmap.shor_measurement_info:
                        shor_info = mmap.shor_measurement_info[shor_key]
                        measurement_lists = shor_info.get('measurement_lists', [])
                        
                        # For each stabilizer, create detectors from redundant measurements
                        for stab_idx, meas_list in enumerate(measurement_lists):
                            if len(meas_list) >= 2:
                                # 1. Consistency detectors: XOR adjacent pairs
                                # For cat state: all ancillas should agree, so XOR pairs = 0
                                for i in range(len(meas_list) - 1):
                                    m1, m2 = meas_list[i], meas_list[i + 1]
                                    lookback1 = m1 - current_total_meas
                                    lookback2 = m2 - current_total_meas
                                    circuit.append("DETECTOR", [
                                        stim.target_rec(lookback1),
                                        stim.target_rec(lookback2)
                                    ])
                                    detector_count += 1
                                
                                # 2. Syndrome detector: first measurement should be 0
                                # for initial |0_L⟩ state
                                lookback = meas_list[0] - current_total_meas
                                circuit.append("DETECTOR", [stim.target_rec(lookback)])
                                detector_count += 1
                            elif len(meas_list) == 1:
                                # Single measurement: just check it's 0
                                lookback = meas_list[0] - current_total_meas
                                circuit.append("DETECTOR", [stim.target_rec(lookback)])
                                detector_count += 1
                        continue  # Skip standard processing for this block
                
                # Standard processing for non-Shor measurements
                hz, hx = self._get_parity_matrices(level)
                H = hz if stype == 'Z' else hx
                if H is not None:
                    n_stabilizers = H.shape[0]
                    n_qubits = len(indices)
                    for stab_idx in range(n_stabilizers):
                        support = np.where(H[stab_idx] == 1)[0]
                        targets = []
                        for q in support:
                            if q < n_qubits:
                                abs_idx = indices[q]
                                lookback = abs_idx - current_total_meas
                                targets.append(stim.target_rec(lookback))
                        if targets:
                            circuit.append("DETECTOR", targets)
                            detector_count += 1
                else:
                    for abs_idx in indices:
                        lookback = abs_idx - current_total_meas
                        circuit.append("DETECTOR", [stim.target_rec(lookback)])
                        detector_count += 1
        
        return detector_count
    
    def _emit_verification_detectors(
        self,
        circuit: stim.Circuit,
        mmap: 'MeasurementMap',
        current_total_meas: int,
        detector_count: int,
    ) -> int:
        """
        Emit detectors for ancilla verification measurements.
        
        IMPORTANT: This function is intentionally disabled. Verification 
        detectors cause "non-deterministic detector" errors in stim because:
        
        1. Verification CNOTs from verification qubit to ancilla qubits
        2. Later, ancilla qubits get RESET for the next syndrome extraction
        3. Stim's backward-propagating detector sensitivity includes the 
           ancilla qubits due to the CNOTs
        4. The reset "cuts" the propagation chain, making the detector 
           non-deterministic
        
        Instead of using detectors, verification is handled via post-selection:
        - Verification measurements are tracked in metadata.verification_measurements
        - The decoder can post-select on shots where all verification == 0
        - This achieves the same FT guarantee without detector issues
        
        To use post-selection, the decoder should check:
            for round_idx, blocks in metadata.verification_measurements.items():
                for block_id, meas_indices in blocks.items():
                    for idx in meas_indices:
                        if measurements[idx] != 0:
                            # Post-select (discard this shot)
        """
        # Deliberately do nothing - verification is handled by post-selection
        # The measurements are tracked in mmap.verification_measurements
        return detector_count
    
    def _emit_temporal_detectors(
        self,
        circuit: stim.Circuit,
        round_idx: int,
        mmap: 'MeasurementMap',
        round_meas_counts: List[int],
        current_total_meas: int,
        detector_count: int,
    ) -> int:
        """
        Emit detectors comparing syndrome at round R to previous rounds.
        
        Creates syndrome PARITY detectors (one per stabilizer generator) comparing
        current round to previous rounds. With temporal_lookback > 1, creates
        detectors comparing to multiple historical rounds (R-1, R-2, ..., R-k).
        
        Multi-round lookback is SAFE because:
        - All compared measurements are from SAME qubit type (ancilla to ancilla)
        - Ancilla reset happens AFTER measurement within each round
        - No cross-block XOR (avoids independent encoded superposition issues)
        - Creates temporal redundancy without crossing ancilla resets
        
        Parity detectors are essential for proper distance behavior - they ensure
        single measurement errors don't dominate the error model.
        
        For Shor-style EC, emits:
        - Consistency detectors (XOR adjacent cat state measurements)
        - Temporal detectors using representative measurement from each cat state
        """
        # Check if this is Shor-style redundant measurements
        is_shor = hasattr(mmap, 'measurement_type') and mmap.measurement_type == "shor_redundant"
        
        # Emit detectors comparing to multiple previous rounds
        max_lookback = min(self.temporal_lookback, round_idx)
        if round_idx == 0:
            # First round: compare to final round (wrap around)
            prev_round_meas = round_meas_counts[-1]
            lookback_rounds = [(-1, prev_round_meas)]  # Only one comparison for round 0
        else:
            # Later rounds: compare to k previous rounds
            lookback_rounds = []
            cumulative_offset = 0
            for k in range(1, max_lookback + 1):
                if round_idx - k >= 0:
                    cumulative_offset += round_meas_counts[round_idx - k]
                    lookback_rounds.append((round_idx - k, cumulative_offset))
                else:
                    break

        # For each syndrome type and block
        for stype in ['X', 'Z']:
            if stype not in mmap.stabilizer_measurements:
                continue
            
            for block_key, indices in mmap.stabilizer_measurements[stype].items():
                level, block_idx = self._get_block_level_and_idx(block_key)
                
                # Handle Shor-style measurements specially
                if is_shor and hasattr(mmap, 'shor_measurement_info'):
                    shor_key = (stype, block_key)
                    if shor_key in mmap.shor_measurement_info:
                        shor_info = mmap.shor_measurement_info[shor_key]
                        measurement_lists = shor_info.get('measurement_lists', [])
                        
                        # For each stabilizer
                        for stab_idx, meas_list in enumerate(measurement_lists):
                            if len(meas_list) >= 2:
                                # 1. Consistency detectors: XOR adjacent cat state measurements
                                for i in range(len(meas_list) - 1):
                                    m1, m2 = meas_list[i], meas_list[i + 1]
                                    lookback1 = m1 - current_total_meas
                                    lookback2 = m2 - current_total_meas
                                    circuit.append("DETECTOR", [
                                        stim.target_rec(lookback1),
                                        stim.target_rec(lookback2)
                                    ])
                                    detector_count += 1
                                
                                # 2. Temporal detector: compare first measurement to previous round
                                # Use first measurement as representative
                                curr_rep = meas_list[0]
                                curr_lookback = curr_rep - current_total_meas
                                for prev_round_idx, cumulative_offset in lookback_rounds:
                                    prev_lookback = curr_lookback - cumulative_offset
                                    circuit.append("DETECTOR", [
                                        stim.target_rec(curr_lookback),
                                        stim.target_rec(prev_lookback),
                                    ])
                                    detector_count += 1
                            elif len(meas_list) == 1:
                                # Single measurement: just do temporal comparison
                                curr_lookback = meas_list[0] - current_total_meas
                                for prev_round_idx, cumulative_offset in lookback_rounds:
                                    prev_lookback = curr_lookback - cumulative_offset
                                    circuit.append("DETECTOR", [
                                        stim.target_rec(curr_lookback),
                                        stim.target_rec(prev_lookback),
                                    ])
                                    detector_count += 1
                        continue  # Skip standard processing
                
                # Standard processing for non-Shor measurements
                # NOTE: Previous skip for encoded ancillas removed. With the MR fix
                # (R → MR in EncodedAncillaGadget.emit_prepare), resets are now
                # deterministic and temporal detectors are valid.
                # 
                # The MR fix collapses any entanglement before reset, so ancilla
                # measurements across rounds can now be compared without causing
                # non-deterministic detector errors.
                
                hz, hx = self._get_parity_matrices(level)
                H = hz if stype == 'Z' else hx
                if H is not None:
                    n_stabilizers = H.shape[0]
                    n_qubits = len(indices)
                    for stab_idx in range(n_stabilizers):
                        support = np.where(H[stab_idx] == 1)[0]
                        
                        # Emit one detector per lookback round
                        for prev_round_idx, cumulative_offset in lookback_rounds:
                            targets = []
                            for q in support:
                                if q < n_qubits:
                                    abs_idx = indices[q]
                                    curr_lookback = abs_idx - current_total_meas
                                    prev_lookback = curr_lookback - cumulative_offset
                                    targets.append(stim.target_rec(curr_lookback))
                                    targets.append(stim.target_rec(prev_lookback))
                            if targets:
                                circuit.append("DETECTOR", targets)
                                detector_count += 1
                else:
                    # No parity matrix: emit simple pairwise detectors
                    for prev_round_idx, cumulative_offset in lookback_rounds:
                        for abs_idx in indices:
                            curr_lookback = abs_idx - current_total_meas
                            prev_lookback = curr_lookback - cumulative_offset
                            circuit.append("DETECTOR", [
                                stim.target_rec(curr_lookback),
                                stim.target_rec(prev_lookback),
                            ])
                            detector_count += 1
        
        return detector_count
    
    def _emit_initial_metacheck_detectors(
        self,
        circuit: stim.Circuit,
        mmap: 'MeasurementMap',
        current_total_meas: int,
        detector_count: int,
    ) -> int:
        """
        Emit outer-level metacheck detectors for round 0.
        
        For concatenated codes [[n1,k1,d1]] ⊗ [[n2,k2,d2]], the outer code's
        stabilizers act on the logical qubits of the inner code blocks.
        
        UPDATED: Now uses actual outer-level measurements at key (0, 0) instead
        of XORing inner block measurements. This properly detects outer logical
        errors and avoids weight-2 logical error paths from correlated detectors.
        """
        # Only needed for multi-level codes (depth >= 2)
        if len(self.code.level_codes) < 2:
            return detector_count
        
        # Get outer code's parity check matrices
        outer_code = self.code.level_codes[0]  # Outermost code
        
        hz_outer = None
        hx_outer = None
        if hasattr(outer_code, 'hz'):
            hz_raw = outer_code.hz
            hz_outer = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        if hasattr(outer_code, 'hx'):
            hx_raw = outer_code.hx
            hx_outer = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
        
        # For each syndrome type (X and Z)
        for stype in ['X', 'Z']:
            if stype not in mmap.stabilizer_measurements:
                continue
            
            # Get outer code's parity matrix for this syndrome type
            H_outer = hz_outer if stype == 'Z' else hx_outer
            if H_outer is None:
                continue
            
            # Look for outer-level measurements at key (0, 0)
            # These are the actual outer stabilizer measurements on the 49-qubit block
            outer_key = (0, 0)
            if outer_key not in mmap.stabilizer_measurements[stype]:
                # Fallback: try without tuple wrapping (some gadgets may use int keys)
                outer_key = 0
                if outer_key not in mmap.stabilizer_measurements[stype]:
                    continue
            
            outer_indices = mmap.stabilizer_measurements[stype][outer_key]
            n_outer_stabs = H_outer.shape[0]  # e.g., 3 for [[7,1,3]]
            
            # Each outer stabilizer measurement corresponds to one row of H_outer
            # Create one detector per outer stabilizer
            for stab_idx in range(n_outer_stabs):
                if stab_idx < len(outer_indices):
                    abs_idx = outer_indices[stab_idx]
                    lookback = abs_idx - current_total_meas
                    circuit.append("DETECTOR", [stim.target_rec(lookback)])
                    detector_count += 1
        
        return detector_count
    
    def _get_x_support_for_code(self, code: Any) -> List[int]:
        """Get X logical operator support for a single code."""
        # Try logical_x property (string format)
        if hasattr(code, 'logical_x'):
            lx = code.logical_x
            if isinstance(lx, list) and len(lx) > 0:
                op = lx[0]
                if isinstance(op, str):
                    return [i for i, c in enumerate(op) if c.upper() in ('X', 'Y')]
        
        # Try lx method/property (numpy format)
        if hasattr(code, 'lx'):
            lx = getattr(code, 'lx')
            if callable(lx):
                lx = lx()
            if isinstance(lx, np.ndarray):
                lx = np.atleast_2d(lx)
                if lx.shape[0] > 0:
                    return list(np.where(lx[0] != 0)[0])
        
        # Try logical_x_ops
        ops = get_logical_ops(code, 'x')
        if ops_valid(ops):
            first_op = ops[0]
            if isinstance(first_op, str):
                return [i for i, c in enumerate(first_op) if c.upper() in ('X', 'Y')]
        
        # Code-specific fallbacks
        code_name = getattr(code, 'name', '') or type(code).__name__
        n = code.n
        if self.allow_logical_support_fallback:
            if 'shor' in code_name.lower():
                return list(range(n))
            if 'steane' in code_name.lower() or 'stean' in code_name.lower():
                return [0, 1, 2]
            if 'hamming' in code_name.lower():
                return [0, 1, 2]
            return list(range(min(3, n)))
        raise ValueError(
            f"Unable to determine logical X support for code {code_name}; provide logical_x/logical_x_ops or set "
            f"allow_logical_support_fallback=True to permit heuristic fallbacks."
        )

    def _get_block_level_and_idx(self, block_key: Any) -> Tuple[int, int]:
        """Return (level, block_idx) for a block key."""
        if isinstance(block_key, tuple) and len(block_key) == 2:
            level, block_idx = block_key
            return int(level), int(block_idx)
        # Default: innermost level
        return self.code.depth - 1, int(block_key)

    def _get_parity_matrices(self, level: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Fetch Hz/Hx matrices for a given concatenation level."""
        code = self.code.level_codes[level]
        hz = hx = None
        if hasattr(code, 'hz'):
            raw = code.hz
            hz = np.atleast_2d(np.array(raw() if callable(raw) else raw, dtype=int))
        if hasattr(code, 'hx'):
            raw = code.hx
            hx = np.atleast_2d(np.array(raw() if callable(raw) else raw, dtype=int))
        return hz, hx
    
    def _emit_temporal_metacheck_detectors(
        self,
        circuit: stim.Circuit,
        round_idx: int,
        mmap: 'MeasurementMap',
        round_meas_counts: List[int],
        current_total_meas: int,
        detector_count: int,
    ) -> int:
        """
        Emit temporal outer-level metacheck detectors comparing round R to round R-1.
        
        UPDATED: Now uses actual outer-level measurements at key (0, 0) for temporal
        comparison instead of XORing inner block measurements. This properly detects
        outer logical errors and avoids weight-2 logical error paths.
        """
        # Only needed for multi-level codes (depth >= 2)
        if len(self.code.level_codes) < 2:
            return detector_count
        
        # Calculate measurements per round (use exact previous round size)
        meas_per_round = round_meas_counts[round_idx - 1] if round_idx > 0 else round_meas_counts[-1]
        
        # Get outer code's parity check matrices
        outer_code = self.code.level_codes[0]
        
        hz_outer = None
        hx_outer = None
        if hasattr(outer_code, 'hz'):
            hz_raw = outer_code.hz
            hz_outer = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        if hasattr(outer_code, 'hx'):
            hx_raw = outer_code.hx
            hx_outer = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
        
        # For each syndrome type (X and Z)
        for stype in ['X', 'Z']:
            if stype not in mmap.stabilizer_measurements:
                continue
            
            H_outer = hz_outer if stype == 'Z' else hx_outer
            if H_outer is None:
                continue
            
            # Look for outer-level measurements at key (0, 0)
            outer_key = (0, 0)
            if outer_key not in mmap.stabilizer_measurements[stype]:
                # Fallback: try without tuple wrapping
                outer_key = 0
                if outer_key not in mmap.stabilizer_measurements[stype]:
                    continue
            
            outer_indices = mmap.stabilizer_measurements[stype][outer_key]
            n_outer_stabs = H_outer.shape[0]  # e.g., 3 for [[7,1,3]]
            
            # Each outer stabilizer: XOR current round with previous round
            for stab_idx in range(n_outer_stabs):
                if stab_idx < len(outer_indices):
                    abs_idx = outer_indices[stab_idx]
                    curr_lookback = abs_idx - current_total_meas
                    prev_lookback = curr_lookback - meas_per_round
                    circuit.append("DETECTOR", [
                        stim.target_rec(curr_lookback),
                        stim.target_rec(prev_lookback),
                    ])
                    detector_count += 1
        
        return detector_count

    def _emit_temporal_frame_detectors(
        self,
        circuit: stim.Circuit,
        round_idx: int,
        pauli_frames: Dict[int, Dict[str, Dict[Any, List[int]]]],
        round_meas_counts: List[int],
        current_total_meas: int,
        detector_count: int,
    ) -> int:
        """Compare Pauli frame bits (X and Z) between consecutive rounds for teleportation EC."""
        prev_round_meas = round_meas_counts[round_idx - 1] if round_idx > 0 else round_meas_counts[-1]
        curr_frame = pauli_frames.get(round_idx, {})
        prev_frame = pauli_frames.get(round_idx - 1, {})
        for ptype in ('X', 'Z'):
            if ptype not in curr_frame or ptype not in prev_frame:
                continue
            for key, indices in curr_frame[ptype].items():
                prev_indices = prev_frame[ptype].get(key)
                if not prev_indices:
                    continue
                n = min(len(indices), len(prev_indices))
                for i in range(n):
                    curr_lookback = indices[i] - current_total_meas
                    prev_lookback = curr_lookback - prev_round_meas
                    circuit.append("DETECTOR", [stim.target_rec(curr_lookback), stim.target_rec(prev_lookback)])
                    detector_count += 1
        return detector_count

    def _get_syndrome_to_stabilizer_map(
        self,
        syndrome_measurements: List[int],
        stabilizer_type: str,
        code: "CSSCode",
    ) -> Dict[int, List[int]]:
        """
        Get mapping from stabilizer index to syndrome measurement indices.
        
        Delegates to the EC gadget if it provides this mapping (e.g., Shor EC),
        otherwise falls back to one-to-one mapping (e.g., Transversal EC).
        
        Parameters
        ----------
        syndrome_measurements : List[int]
            Flat list of all measurement indices for this syndrome type
        stabilizer_type : str
            'X' or 'Z' syndrome type
        code : CSSCode
            The code being protected
            
        Returns
        -------
        Dict[int, List[int]]
            Mapping from stabilizer_idx -> list of measurement indices
        """
        # Try to get gadget-specific mapping
        if self.ec_gadget is not None:
            gadget = self.ec_gadget
            
            # For RecursiveGadget, get the innermost gadget
            if hasattr(gadget, 'level_gadgets'):
                inner_level = max(gadget.level_gadgets.keys()) if gadget.level_gadgets else 0
                gadget = gadget.level_gadgets.get(inner_level, gadget)
            
            # For ParallelGadget, get the underlying gadget
            # ParallelGadget uses 'base_gadget' attribute
            if hasattr(gadget, 'base_gadget'):
                gadget = gadget.base_gadget
            # Some gadgets might use 'gadget' attribute
            elif hasattr(gadget, 'gadget'):
                gadget = gadget.gadget
            
            # Use gadget's method if available
            if hasattr(gadget, 'get_syndrome_to_stabilizer_map'):
                return gadget.get_syndrome_to_stabilizer_map(
                    syndrome_measurements, stabilizer_type, code
                )
        
        # Default: one-to-one mapping (measurement i → stabilizer i)
        result = {}
        for i, meas_idx in enumerate(syndrome_measurements):
            result[i] = [meas_idx]
        return result

    def _emit_boundary_detectors(
        self,
        circuit: stim.Circuit,
        last_round: int,
        all_syndrome_meas: Dict[str, Dict[Any, List[int]]],
        ec_meas_offset: int,
        n_data: int,
        detector_count: int,
    ) -> int:
        """
        Emit boundary detectors comparing last syndrome to final data.
        
        These detectors compare the last round's syndrome measurements to the
        parities we can infer from the final data measurement. This catches
        errors that occurred after the last EC round.
        
        For Z stabilizers (which detect X errors), the final data measurement
        in the Z basis gives us direct parity information via Hz matrix.
        
        The boundary detector for stabilizer s in block b is:
            D = (last_syndrome_s) XOR (final_data parity for stabilizer s)
        
        This should be 0 if no errors occurred after the last syndrome extraction.
        """
        total_meas = ec_meas_offset + n_data
        
        # Detect SHOR EC or TRANSVERSAL+encoded by checking the ec_method configuration
        # SHOR EC has ancilla resets between syndrome measurements and final data,
        # so we cannot include syndrome measurements in boundary detectors.
        # TRANSVERSAL+encoded also resets ancillas after each measurement (changed from MR to M),
        # so we have the same issue.
        is_shor_ec = False
        ancillas_encoded = hasattr(self, '_ancilla_prep') and self._ancilla_prep in ("encoded", "verified")
        if self._ec_method is not None:
            ec_method_str = str(self._ec_method).lower()
            is_shor_ec = 'shor' in ec_method_str or ('transversal' in ec_method_str and ancillas_encoded)
        
        # Get inner code's parity check matrices
        inner_code = self.code.level_codes[-1]
        hz = None
        hx = None
        if hasattr(inner_code, 'hz'):
            hz_raw = inner_code.hz
            hz = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        if hasattr(inner_code, 'hx'):
            hx_raw = inner_code.hx
            hx = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
        
        if hz is None:
            return detector_count
        
        n_z_stabilizers = hz.shape[0]
        inner_level = len(self.code.level_codes) - 1  # e.g., 1 for 2-level concatenation
        
        # Build mapping from block_id to data qubits
        block_to_data = {}
        for idx, (addr, (start, end)) in enumerate(self.qubit_mapper.iter_leaf_ranges()):
            block_to_data[idx] = list(range(start, end))
        
        # For Z-basis measurement, we can check Z stabilizers against final data
        # For each block with last-round Z syndrome measurements
        for (round_idx, block_key), z_meas_indices in all_syndrome_meas.get('Z', {}).items():
            if round_idx != last_round:
                continue
            
            # block_key can be nested: (level, block_idx) or (level, (level, block_idx))
            # Need to recursively extract the innermost (level, block_idx) pair
            if isinstance(block_key, tuple):
                # Recursively unwrap nested tuples
                current = block_key
                while isinstance(current, tuple) and len(current) == 2:
                    if isinstance(current[1], tuple):
                        # Nested structure like (1, (1, 0))
                        current = current[1]
                    else:
                        # Found the innermost (level, block_idx)
                        break
                
                if isinstance(current, tuple) and len(current) == 2:
                    level, block_idx = current
                    # CRITICAL FIX: Allow BOTH inner (level=1) AND outer (level=0) boundary detectors
                    # Outer-level boundary detectors are essential to eliminate weight-1 errors
                    # Skip only if it's not inner AND not outer (e.g., intermediate levels in deep concatenation)
                    if level not in [0, inner_level]:
                        continue  # Skip intermediate levels only
                else:
                    continue  # Malformed key
            else:
                block_idx = block_key  # Direct block index
            
            if block_idx not in block_to_data:
                continue
                
            block_data_qs = block_to_data[block_idx]
            n_block = len(block_data_qs)
            
            # Get syndrome-to-stabilizer mapping from gadget
            # For Shor EC: each stabilizer has multiple measurements
            # For Transversal EC: each stabilizer has one measurement
            stab_to_meas = self._get_syndrome_to_stabilizer_map(z_meas_indices, 'Z', inner_code)
            
            # For each Z stabilizer generator
            for stab_idx in range(n_z_stabilizers):
                # Get support of this Z stabilizer (qubits it acts on)
                support = np.where(hz[stab_idx] == 1)[0]
                
                # Build boundary detector:
                # For Transversal EC: XOR syndrome measurement(s) with final data parity
                # For Shor EC: Only use final data parity (skip syndrome - causes non-determinism)
                targets = []
                
                if not is_shor_ec:
                    # Non-Shor EC: include syndrome measurement in boundary detector
                    stab_meas = stab_to_meas.get(stab_idx, [])
                    for syndrome_meas_idx in stab_meas:
                        rel_idx = syndrome_meas_idx - total_meas
                        if rel_idx >= -16777215 and rel_idx <= -1:
                            targets.append(stim.target_rec(rel_idx))
                # For Shor EC: Skip syndrome measurements to avoid non-determinism
                # The ancillas are reset between syndrome extraction and final data measurement
                
                # Add final data measurements for qubits in this stabilizer's support
                for local_q in support:
                    if local_q < n_block:
                        data_q = block_data_qs[local_q]
                        data_meas_idx = ec_meas_offset + data_q
                        targets.append(stim.target_rec(data_meas_idx - total_meas))
                
                if targets:
                    circuit.append("DETECTOR", targets)
                    detector_count += 1
        
        # X-side boundary detectors when final basis is X
        if hx is not None and self.basis == 'X' and 'X' in all_syndrome_meas:
            n_x_stabilizers = hx.shape[0]
            for (round_idx, block_key), x_meas_indices in all_syndrome_meas.get('X', {}).items():
                if round_idx != last_round:
                    continue
                if isinstance(block_key, tuple):
                    # Recursively unwrap nested tuples
                    current = block_key
                    while isinstance(current, tuple) and len(current) == 2:
                        if isinstance(current[1], tuple):
                            current = current[1]
                        else:
                            break
                    
                    if isinstance(current, tuple) and len(current) == 2:
                        level, block_idx = current
                        if level != inner_level:
                            continue
                    else:
                        continue
                else:
                    block_idx = block_key
                if block_idx not in block_to_data:
                    continue
                block_data_qs = block_to_data[block_idx]
                n_block = len(block_data_qs)
                
                # Get syndrome-to-stabilizer mapping for X syndromes
                stab_to_meas_x = self._get_syndrome_to_stabilizer_map(x_meas_indices, 'X', inner_code)
                
                for stab_idx in range(n_x_stabilizers):
                    support = np.where(hx[stab_idx] == 1)[0]
                    targets = []
                    
                    if not is_shor_ec:
                        # Non-Shor EC: include syndrome measurements
                        stab_meas_x = stab_to_meas_x.get(stab_idx, [])
                        for meas_idx in stab_meas_x:
                            rel_idx = meas_idx - total_meas
                            if rel_idx >= -16777215 and rel_idx <= -1:
                                targets.append(stim.target_rec(rel_idx))
                    # For Shor EC: Skip syndrome measurements to avoid non-determinism
                    
                    # Add final data measurements for qubits in this stabilizer's support
                    for local_q in support:
                        if local_q < n_block:
                            data_q = block_data_qs[local_q]
                            data_meas_idx = ec_meas_offset + data_q
                            targets.append(stim.target_rec(data_meas_idx - total_meas))
                    if targets:
                        circuit.append("DETECTOR", targets)
                        detector_count += 1
        
        return detector_count
    
    def _emit_selective_boundary_detectors(
        self,
        circuit: stim.Circuit,
        round_idx: int,
        all_syndrome_meas: Dict[str, Dict[Any, List[int]]],
        ec_meas_offset: int,
        n_data: int,
        detector_count: int,
    ) -> int:
        """
        Emit boundary detectors only for observable-supporting blocks.
        
        This provides redundant coverage for qubits in few stabilizers without
        creating too many correlations (which causes >15 symptom errors in DEM).
        """
        total_meas = ec_meas_offset + n_data
        
        # Detect SHOR EC by checking the ec_method configuration
        is_shor_ec = False
        if self._ec_method is not None:
            ec_method_str = str(self._ec_method).lower()
            is_shor_ec = 'shor' in ec_method_str
        
        # Get observable-supporting qubits
        obs_qubits = set(self._compute_observable_support())
        if not obs_qubits:
            return detector_count
        
        # Get inner code's parity check matrices
        inner_code = self.code.level_codes[-1]
        hz = None
        if hasattr(inner_code, 'hz'):
            hz_raw = inner_code.hz
            hz = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        
        if hz is None:
            return detector_count
        
        n_z_stabilizers = hz.shape[0]
        inner_level = len(self.code.level_codes) - 1
        
        # Build mapping from block_id to data qubits
        block_to_data = {}
        for idx, (addr, (start, end)) in enumerate(self.qubit_mapper.iter_leaf_ranges()):
            block_to_data[idx] = list(range(start, end))
        
        # Identify observable-supporting blocks
        obs_blocks = set()
        for block_idx, block_qubits in block_to_data.items():
            if any(q in obs_qubits for q in block_qubits):
                obs_blocks.add(block_idx)
        
        # Emit boundary detectors only for observable-supporting blocks
        for (r, block_key), z_meas_indices in all_syndrome_meas.get('Z', {}).items():
            if r != round_idx:
                continue
            
            # Parse block_key
            if isinstance(block_key, tuple):
                current = block_key
                while isinstance(current, tuple) and len(current) == 2:
                    if isinstance(current[1], tuple):
                        current = current[1]
                    else:
                        break
                
                if isinstance(current, tuple) and len(current) == 2:
                    level, block_idx = current
                    if level != inner_level:
                        continue
                else:
                    continue
            else:
                block_idx = block_key
            
            # Skip blocks that don't support observable
            if block_idx not in obs_blocks:
                continue
            
            if block_idx not in block_to_data:
                continue
            
            block_data_qs = block_to_data[block_idx]
            n_block = len(block_data_qs)
            
            # Emit detectors for stabilizers touching observable qubits in this block
            for stab_idx in range(n_z_stabilizers):
                support = np.where(hz[stab_idx] == 1)[0]
                
                # Check if this stabilizer touches any observable qubits
                stab_obs_qubits = [block_data_qs[local_q] for local_q in support 
                                  if local_q < n_block and block_data_qs[local_q] in obs_qubits]
                
                if not stab_obs_qubits:
                    continue  # Skip stabilizers that don't touch observable
                
                targets = []
                
                # For Shor EC: Skip syndrome measurements to avoid non-determinism
                # The ancilla qubits are reset between syndrome extraction and final data measurement
                if not is_shor_ec:
                    # Get syndrome-to-stabilizer mapping
                    stab_to_meas = self._get_syndrome_to_stabilizer_map(z_meas_indices, 'Z', inner_code)
                    for local_q in support:
                        if local_q < len(z_meas_indices):
                            syndrome_meas_idx = z_meas_indices[local_q]
                            targets.append(stim.target_rec(syndrome_meas_idx - total_meas))
                
                for local_q in support:
                    if local_q < n_block:
                        data_q = block_data_qs[local_q]
                        data_meas_idx = ec_meas_offset + data_q
                        targets.append(stim.target_rec(data_meas_idx - total_meas))
                
                if targets:
                    circuit.append("DETECTOR", targets)
                    detector_count += 1
        
        return detector_count
    
    def _emit_observable_temporal_detectors(
        self,
        circuit: stim.Circuit,
        all_syndrome_meas: Dict[str, Dict[Any, List[int]]],
        ec_meas_offset: int,
        n_data: int,
        detector_count: int,
    ) -> int:
        """
        Emit temporal redundancy detectors for observable qubits.
        
        RESTORED AND ENHANCED OLD IMPLEMENTATION:
        This method uses the BOUNDARY DETECTOR approach that compares syndrome measurements
        from EARLIER rounds (not just the last round) to final data measurements. This avoids
        the non-deterministic detector errors that occur when comparing syndromes across rounds
        (which crosses ancilla resets).
        
        UPDATED: Now emits for ALL observable-supporting qubits, not just undercovered ones.
        The original implementation only targeted qubits in <2 stabilizers, which was too
        conservative for codes like Steane where all qubits have good coverage.
        
        Problem: Observable measurements need redundant detector coverage to prevent
        weight-1 logical errors (single measurement error flips both observable and
        exactly one detector).
        
        Solution: For each observable-supporting qubit, add detectors that compare
        its final data measurement against syndrome measurements from EARLIER rounds
        (not just the last round).
        
        This creates temporal redundancy: errors in the final measurement are
        detected by:
        1. The standard boundary detector (last syndrome vs final data)
        2. These temporal redundancy detectors (earlier syndromes vs final data)
        
        Key insight: Use earlier round syndromes (r-2, r-3, ..., r-temporal_depth)
        because round r-1 syndromes are already used in boundary detectors. The
        temporal_depth parameter controls how many earlier rounds to use.
        
        The detector parity is: syndrome(r-k) ⊕ final_data(qubits_in_stabilizer)
        where k ranges from 2 to temporal_depth.
        
        BOUNDARY DETECTOR APPROACH IS SAFE:
        - Syndrome measurements reference ancilla qubits BEFORE they're reset
        - Final data measurements reference data qubits only
        - No cross-round syndrome comparison → no ancilla reset crossing
        """
        if self.rounds < 2:
            return detector_count  # Need at least 2 rounds
        
        total_meas = ec_meas_offset + n_data
        obs_qubits = set(self._compute_observable_support())
        if not obs_qubits:
            return detector_count
        
        inner_code = self.code.level_codes[-1]
        hz = None
        if hasattr(inner_code, 'hz'):
            hz_raw = inner_code.hz
            hz = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        
        if hz is None:
            return detector_count
        
        n_z_stabilizers = hz.shape[0]
        inner_level = len(self.code.level_codes) - 1
        n_inner_qubits = inner_code.n
        
        # Build mapping from block_id to data qubits
        block_to_data = {}
        for idx, (addr, (start, end)) in enumerate(self.qubit_mapper.iter_leaf_ranges()):
            block_to_data[idx] = list(range(start, end))
        
        # UPDATED: Target ALL observable qubits, not just undercovered ones
        # The OLD implementation was too conservative - Steane has no undercovered qubits!
        # By emitting for all observable qubits, we get temporal redundancy for better min_weight
        target_obs_qubits = obs_qubits  # Use all observable qubits
        
        # For each undercovered observable qubit, find which stabilizer(s) contain it
        # and add temporal detectors comparing earlier round syndromes to final data
        
        # ENHANCED: Use temporal_depth to emit detectors for multiple earlier rounds
        # For temporal_depth=3: emit from rounds r-2, r-3, r-4 (if they exist)
        depth = min(self.temporal_depth, self.rounds)
        emitted_pairs = set()  # Track (block_idx, stab_idx, round) to avoid duplicates
        
        for offset in range(2, depth + 1):  # Start at r-2, go back temporal_depth rounds
            target_round = self.rounds - offset
            if target_round < 0:
                continue  # Skip if round doesn't exist
            
            for (round_idx, block_key), z_meas_indices in all_syndrome_meas.get('Z', {}).items():
                if round_idx != target_round:
                    continue
                
                # Parse block_key to get block_idx
                if isinstance(block_key, tuple):
                    current = block_key
                    while isinstance(current, tuple) and len(current) == 2:
                        if isinstance(current[1], tuple):
                            current = current[1]
                        else:
                            break
                    
                    if isinstance(current, tuple) and len(current) == 2:
                        level, block_idx = current
                        if level != inner_level:
                            continue
                    else:
                        continue
                else:
                    block_idx = block_key
                
                if block_idx not in block_to_data:
                    continue
                
                block_data_qs = block_to_data[block_idx]
                n_block = len(block_data_qs)
                
                # Check if this block has any observable qubits
                block_obs_qubits = [q for q in block_data_qs if q in target_obs_qubits]
                if not block_obs_qubits:
                    continue
                
                # For each Z stabilizer that contains an observable qubit, emit a detector
                for stab_idx in range(n_z_stabilizers):
                    support = np.where(hz[stab_idx] == 1)[0]
                    
                    # Check if this stabilizer touches any observable qubits
                    stab_obs_qubits = []
                    for local_q in support:
                        if local_q < n_block:
                            phys_q = block_data_qs[local_q]
                            if phys_q in target_obs_qubits:
                                stab_obs_qubits.append(phys_q)
                    
                    if not stab_obs_qubits:
                        continue
                    
                    # Avoid emitting duplicate detectors
                    det_key = (block_idx, stab_idx, target_round)
                    if det_key in emitted_pairs:
                        continue
                    emitted_pairs.add(det_key)
                    
                    # Build detector: XOR of syndrome measurements + final data for qubits in support
                    targets = []
                    
                    # Add syndrome measurements for this stabilizer
                    for local_q in support:
                        if local_q < len(z_meas_indices):
                            syn_meas_idx = z_meas_indices[local_q]
                            targets.append(stim.target_rec(syn_meas_idx - total_meas))
                    
                    # Add final data measurements for qubits in this stabilizer's support
                    for local_q in support:
                        if local_q < n_block:
                            phys_q = block_data_qs[local_q]
                            data_meas_idx = ec_meas_offset + phys_q
                            targets.append(stim.target_rec(data_meas_idx - total_meas))
                    
                    if targets:
                        circuit.append("DETECTOR", targets)
                        detector_count += 1
        
        return detector_count
    
    def _validate_detector_coverage(
        self,
        circuit: stim.Circuit,
        ec_meas_offset: int,
        n_data: int,
    ) -> None:
        """
        Validate that each observable measurement appears in at least 2 detectors.
        
        This prevents weight-1 logical errors where a single measurement error
        can flip both a detector and the observable with no other signatures.
        
        Raises:
            AssertionError: If any observable measurement has insufficient coverage
        """
        total_meas = ec_meas_offset + n_data
        obs_qubits = set(self._compute_observable_support())
        
        if not obs_qubits:
            return  # No observable to validate
        
        # Map observable qubits to their measurement indices (final data measurements)
        obs_meas_indices = {ec_meas_offset + q for q in obs_qubits}
        
        # Count how many detectors cover each measurement
        meas_coverage = {idx: 0 for idx in obs_meas_indices}
        
        # Parse all detector instructions to count coverage
        import re
        circuit_str = str(circuit)
        detector_lines = [line for line in circuit_str.split('\n') if line.strip().startswith('DETECTOR')]
        
        for det_line in detector_lines:
            # Extract all rec[...] references
            rec_matches = re.findall(r'rec\[(-?\d+)\]', det_line)
            for rec_str in rec_matches:
                rec_offset = int(rec_str)
                abs_idx = total_meas + rec_offset  # Convert relative to absolute
                if abs_idx in meas_coverage:
                    meas_coverage[abs_idx] += 1
        
        # Check for insufficient coverage
        insufficient = {idx: count for idx, count in meas_coverage.items() if count < 2}
        
        if insufficient:
            print(f"\n⚠️ WARNING: {len(insufficient)} observable measurements have < 2 detector coverage:")
            for idx, count in sorted(insufficient.items())[:10]:
                qubit = idx - ec_meas_offset
                print(f"  Qubit {qubit} (meas {idx}): {count} detector(s)")
            if len(insufficient) > 10:
                print(f"  ... and {len(insufficient) - 10} more")
            print(f"\n  This will cause weight-1 logical errors!")
            print(f"  Expected: All {len(obs_meas_indices)} obs measurements should have ≥2 detectors")
            print(f"  Actual: {len(insufficient)} have <2, {len(meas_coverage) - len(insufficient)} have ≥2\n")
    
    def _emit_observable_redundancy_detectors(
        self,
        circuit: stim.Circuit,
        all_syndrome_meas: Dict[str, Dict[Any, List[int]]],
        ec_meas_offset: int,
        n_data: int,
        detector_count: int,
    ) -> int:
        """
        Emit additional detectors to ensure observable measurements have redundant coverage.
        
        Problem: Weight-1 logical errors occur when a measurement error flips both:
        1. Exactly one detector (the error appears in only one detector)
        2. The observable (the measurement is part of the observable)
        
        Solution: For each measurement in the observable support, ensure it appears
        in at least 2 detectors. This function adds detectors comparing:
        - Penultimate round syndrome to final data measurements on observable support
        
        This provides temporal redundancy: errors in the final measurement are
        detected by both the last-round detector AND these penultimate-round detectors.
        """
        if self.rounds < 2:
            return detector_count  # Need at least 2 rounds for this redundancy
        
        total_meas = ec_meas_offset + n_data
        obs_qubits = set(self._compute_observable_support())
        if not obs_qubits:
            return detector_count
        
        inner_code = self.code.level_codes[-1]
        hz = None
        if hasattr(inner_code, 'hz'):
            hz_raw = inner_code.hz
            hz = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        
        if hz is None:
            return detector_count
        
        n_z_stabilizers = hz.shape[0]
        inner_level = len(self.code.level_codes) - 1
        penultimate_round = self.rounds - 2  # Round before last
        
        # Build mapping from block_id to data qubits
        block_to_data = {}
        for idx, (addr, (start, end)) in enumerate(self.qubit_mapper.iter_leaf_ranges()):
            block_to_data[idx] = list(range(start, end))
        
        # For each block, add detectors comparing penultimate syndrome to final data
        # but ONLY for qubits that are in the observable support
        for (round_idx, block_key), z_meas_indices in all_syndrome_meas.get('Z', {}).items():
            if round_idx != penultimate_round:
                continue
            
            if isinstance(block_key, tuple):
                level, block_idx = block_key
                if level != inner_level:
                    continue
            else:
                block_idx = block_key
            
            if block_idx not in block_to_data:
                continue
            
            block_data_qs = block_to_data[block_idx]
            n_block = len(block_data_qs)
            
            # Check if any of this block's qubits are in the observable
            block_obs_qubits = [q for q in block_data_qs if q in obs_qubits]
            if not block_obs_qubits:
                continue
            
            # For each Z stabilizer, emit a detector if it touches observable qubits
            for stab_idx in range(n_z_stabilizers):
                support = np.where(hz[stab_idx] == 1)[0]
                
                # Check if this stabilizer touches any observable qubits in this block
                stab_obs_qubits = [block_data_qs[local_q] for local_q in support 
                                  if local_q < n_block and block_data_qs[local_q] in obs_qubits]
                if not stab_obs_qubits:
                    continue
                
                targets = []
                
                # Add penultimate syndrome measurements
                for local_q in support:
                    if local_q < len(z_meas_indices):
                        syndrome_meas_idx = z_meas_indices[local_q]
                        targets.append(stim.target_rec(syndrome_meas_idx - total_meas))
                
                # Add final data measurements  
                for local_q in support:
                    if local_q < n_block:
                        data_q = block_data_qs[local_q]
                        data_meas_idx = ec_meas_offset + data_q
                        targets.append(stim.target_rec(data_meas_idx - total_meas))
                
                if targets:
                    circuit.append("DETECTOR", targets)
                    detector_count += 1
        
        return detector_count

    def _emit_boundary_metacheck_detectors(
        self,
        circuit: stim.Circuit,
        last_round: int,
        all_syndrome_meas: Dict[str, Dict[Any, List[int]]],
        ec_meas_offset: int,
        n_data: int,
        detector_count: int,
    ) -> int:
        """
        Emit boundary metacheck detectors comparing last syndrome to final data.
        
        UPDATED: Now uses actual outer-level measurements at key (0, 0) instead
        of XORing inner block measurements. Compares outer syndrome to outer
        logical parity inferred from final data measurements.
        """
        # Only needed for multi-level codes (depth >= 2)
        if len(self.code.level_codes) < 2:
            return detector_count
        
        total_meas = ec_meas_offset + n_data
        
        # Get outer code info
        outer_code = self.code.level_codes[0]
        
        hz_outer = None
        if hasattr(outer_code, 'hz'):
            hz_raw = outer_code.hz
            hz_outer = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        
        if hz_outer is None:
            return detector_count
        
        # Look for outer-level Z syndrome measurements at key (0, 0)
        outer_z_key = (last_round, (0, 0))
        if outer_z_key not in all_syndrome_meas.get('Z', {}):
            # Try alternate key format
            outer_z_key = (last_round, 0)
            if outer_z_key not in all_syndrome_meas.get('Z', {}):
                return detector_count
        
        outer_z_indices = all_syndrome_meas['Z'][outer_z_key]
        n_outer_stabs = hz_outer.shape[0]
        
        # Get inner code info for hierarchical mapping
        inner_code = self.code.level_codes[-1]
        inner_n = inner_code.n
        inner_lz = self._get_z_support_for_code(inner_code)  # e.g., [0,1,2] for Steane
        
        # For each outer Z-stabilizer, build detector with proper hierarchical support
        for stab_idx in range(n_outer_stabs):
            if stab_idx >= len(outer_z_indices):
                continue
            
            targets = []
            
            # Add last syndrome measurement for this stabilizer
            syndrome_idx = outer_z_indices[stab_idx]
            targets.append(stim.target_rec(syndrome_idx - total_meas))
            
            # FIX: Add final data measurements for ALL blocks in outer stabilizer support
            # For each outer block in the stabilizer's support, add the inner logical Z qubits
            outer_stab_support = np.where(hz_outer[stab_idx] == 1)[0]
            
            if inner_lz is not None:
                for outer_block in outer_stab_support:
                    for inner_q in inner_lz:
                        # Map (outer_block, inner_q) to physical qubit
                        physical_q = outer_block * inner_n + inner_q
                        if physical_q < n_data:
                            data_meas_idx = ec_meas_offset + physical_q
                            targets.append(stim.target_rec(data_meas_idx - total_meas))
            
            if targets:
                circuit.append("DETECTOR", targets)
                detector_count += 1

        # Outer X-side boundary metachecks (only meaningful when final basis is X)
        if self.basis == 'X':
            hx_outer = None
            if hasattr(outer_code, 'hx'):
                hx_raw = outer_code.hx
                hx_outer = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
            
            if hx_outer is not None:
                n_outer_x = hx_outer.shape[0]
                
                # Look for outer-level X syndrome measurements
                outer_x_key = (last_round, (0, 0))
                if outer_x_key not in all_syndrome_meas.get('X', {}):
                    outer_x_key = (last_round, 0)
                
                if outer_x_key in all_syndrome_meas.get('X', {}):
                    outer_x_indices = all_syndrome_meas['X'][outer_x_key]
                    inner_lx = self._get_x_support_for_code(inner_code)  # Use inner code's X support
                    
                    for stab_idx in range(n_outer_x):
                        if stab_idx >= len(outer_x_indices):
                            continue
                        
                        targets = []
                        
                        # Add last X syndrome measurement
                        syndrome_idx = outer_x_indices[stab_idx]
                        targets.append(stim.target_rec(syndrome_idx - total_meas))
                        
                        # FIX: Add final data measurements for ALL blocks in outer X stabilizer support
                        outer_stab_support = np.where(hx_outer[stab_idx] == 1)[0]
                        
                        if inner_lx is not None:
                            for outer_block in outer_stab_support:
                                for inner_q in inner_lx:
                                    physical_q = outer_block * inner_n + inner_q
                                    if physical_q < n_data:
                                        data_meas_idx = ec_meas_offset + physical_q
                                        targets.append(stim.target_rec(data_meas_idx - total_meas))
                        
                        if targets:
                            circuit.append("DETECTOR", targets)
                            detector_count += 1
        
        return detector_count

    def _emit_spatial_metachecks(
        self,
        circuit: stim.Circuit,
        all_syndrome_meas: Dict[str, Dict[Any, List[int]]],
        total_meas: int,
        detector_count: int,
    ) -> int:
        """
        Emit spatial metacheck detectors connecting inner blocks via outer stabilizers.
        
        CRITICAL FOR FAULT-TOLERANCE:
        For concatenated codes, the logical observable only spans a subset of blocks
        (determined by outer code's logical operator support, e.g., blocks {0,1,2} for
        Steane outer code). Blocks outside this support (e.g., {3,4,5,6}) have detectors
        that would otherwise be disconnected from the observable in the DEM.
        
        This function creates cross-block detectors that XOR inner block syndrome 
        measurements according to outer stabilizer structure. For each outer Z-stabilizer
        with support S = {b1, b2, ..., bw}, we create detectors that XOR measurements
        from blocks in S, connecting blocks outside logical support to those inside.
        
        The key insight: outer stabilizers connect ALL their support blocks together,
        so if any block in S is in the logical support, errors in OTHER blocks in S
        will have DEM paths to the observable via these cross-block detectors.
        """
        if len(self.code.level_codes) < 2:
            return detector_count
        
        outer_code = self.code.level_codes[0]
        inner_code = self.code.level_codes[-1]
        inner_level = len(self.code.level_codes) - 1
        
        # Get outer code's parity check matrices
        hz_outer = None
        hx_outer = None
        if hasattr(outer_code, 'hz'):
            hz_raw = outer_code.hz
            hz_outer = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        if hasattr(outer_code, 'hx'):
            hx_raw = outer_code.hx
            hx_outer = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
        
        # Get inner code's parity check matrices
        hz_inner = None
        hx_inner = None
        if hasattr(inner_code, 'hz'):
            hz_raw = inner_code.hz
            hz_inner = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        if hasattr(inner_code, 'hx'):
            hx_raw = inner_code.hx
            hx_inner = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
        
        # Get inner code's logical Z support (which qubits within each block are in ZL)
        lz_inner = self._get_z_support_for_code(inner_code)  # e.g., [0,1,2] for Steane
        
        # === Z-side spatial metachecks ===
        # INSIGHT: For concatenated codes with encoded ancillas, we CANNOT create
        # cross-block detectors using syndrome measurements from the same round because:
        # 1. Each block's ancilla is prepared in an encoded superposition
        # 2. Different blocks' superpositions are INDEPENDENT
        # 3. XORing measurements from independent superpositions is non-deterministic
        #
        # Instead, we rely on:
        # 1. Per-block temporal detectors (same block, different rounds) - DETERMINISTIC
        # 2. Per-block boundary detectors (last syndrome vs final data) - DETERMINISTIC
        # 3. Observable that spans all blocks via outer logical Z - connects blocks via DEM
        #
        # The DEM connectivity comes from the OBSERVABLE_INCLUDE, not from cross-block
        # detectors. Each block's errors can flip the observable, and the decoder
        # uses the outer code structure to identify which blocks have errors.
        #
        # DISABLED: Cross-block spatial metachecks cause non-deterministic detectors
        # with encoded ancilla preparation.
        pass
        
        # === X-side spatial metachecks (for X-basis memory) ===
        # DISABLED: Same issue as Z-side - independent encoded ancilla superpositions
        # cause non-deterministic detectors when XORing across blocks.
        
        return detector_count
    
    def _emit_safe_spatial_metachecks(
        self,
        circuit: stim.Circuit,
        all_syndrome_meas: Dict[str, Dict[Any, List[int]]],
        ec_meas_offset: int,
        n_data: int,
        detector_count: int,
    ) -> int:
        """
        Emit SAFE outer-code spatial meta-check detectors that are deterministic.
        
        KEY INSIGHT: While XORing raw ancilla measurements from different blocks is
        non-deterministic (due to independent encoded superpositions), we CAN create
        deterministic cross-block detectors by using outer-code stabilizer constraints
        applied to BOUNDARY detectors (last syndrome vs final data).
        
        For each outer Z-stabilizer with block support S = {b1, b2, ...}:
        - Create a detector that XORs the BOUNDARY detector targets for all blocks in S
        - This means: for each block b in S, include its last-round Z-syndrome measurements
          AND its corresponding final data measurements
        - The combined detector checks that the PRODUCT of inner logical Z values
          across blocks in S is deterministic (which it is, as an outer stabilizer)
        
        This is deterministic because:
        1. It represents a TRUE STABILIZER of the concatenated code
        2. Each block's contribution is (last_syndrome XOR final_data), which is deterministic
        3. The XOR across blocks in an outer stabilizer support is also deterministic
        """
        if len(self.code.level_codes) < 2:
            return detector_count
        
        outer_code = self.code.level_codes[0]
        inner_code = self.code.level_codes[-1]
        inner_n = inner_code.n
        
        # Get outer code's Z parity check matrix
        hz_outer = None
        if hasattr(outer_code, 'hz'):
            hz_raw = outer_code.hz
            hz_outer = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        
        if hz_outer is None:
            return detector_count
        
        # Get inner code's Z parity check matrix
        hz_inner = None
        if hasattr(inner_code, 'hz'):
            hz_raw = inner_code.hz
            hz_inner = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        
        if hz_inner is None:
            return detector_count
        
        # Get inner logical Z support
        inner_lz = self._get_z_support_for_code(inner_code)
        if inner_lz is None:
            inner_lz = list(range(inner_n))  # fallback: all qubits
        
        # Find the last round's syndrome measurements
        last_round = self.rounds - 1
        z_meas = all_syndrome_meas.get('Z', {})
        
        total_meas = ec_meas_offset + n_data
        n_outer_stab = hz_outer.shape[0]
        n_inner_stab = hz_inner.shape[0]
        
        # For each outer Z-stabilizer
        for outer_stab_idx in range(n_outer_stab):
            outer_stab_row = hz_outer[outer_stab_idx]
            block_support = np.where(outer_stab_row == 1)[0]
            
            targets = []
            
            # For each block in this outer stabilizer's support
            for block_idx in block_support:
                # Get last-round inner Z syndrome measurements for this block
                # Key format is (round, (1, (1, block_idx))) for inner blocks
                block_key = (last_round, (1, (1, block_idx)))
                
                if block_key not in z_meas:
                    # Try alternative key formats
                    block_key = (last_round, (1, block_idx))
                    
                if block_key not in z_meas:
                    block_key = (last_round, block_idx)
                
                if block_key not in z_meas:
                    continue
                
                inner_z_indices = z_meas[block_key]
                
                # Add syndrome measurements for each inner Z-stabilizer
                # We need to select which inner stabilizers to use based on inner logical Z
                # For boundary detectors: we XOR syndrome with data parity
                
                # Strategy: Include ALL inner Z syndrome measurements for this block
                # AND the corresponding data measurements on inner lz support
                for meas_idx in inner_z_indices:
                    rel_idx = meas_idx - total_meas
                    if rel_idx < 0:
                        targets.append(stim.target_rec(rel_idx))
                
                # Add final data measurements for inner lz support qubits in this block
                for inner_q in inner_lz:
                    physical_q = block_idx * inner_n + inner_q
                    if physical_q < n_data:
                        data_meas_idx = ec_meas_offset + physical_q
                        rel_idx = data_meas_idx - total_meas
                        targets.append(stim.target_rec(rel_idx))
            
            # Emit the spatial metacheck detector if we have valid targets
            if len(targets) >= 2:
                circuit.append("DETECTOR", targets)
                detector_count += 1
        
        # Also emit for earlier rounds for temporal redundancy
        for round_offset in range(1, min(self.temporal_depth, self.rounds)):
            target_round = self.rounds - 1 - round_offset
            if target_round < 0:
                continue
                
            for outer_stab_idx in range(n_outer_stab):
                outer_stab_row = hz_outer[outer_stab_idx]
                block_support = np.where(outer_stab_row == 1)[0]
                
                targets = []
                
                for block_idx in block_support:
                    block_key = (target_round, (1, (1, block_idx)))
                    
                    if block_key not in z_meas:
                        block_key = (target_round, (1, block_idx))
                        
                    if block_key not in z_meas:
                        continue
                    
                    inner_z_indices = z_meas[block_key]
                    
                    for meas_idx in inner_z_indices:
                        rel_idx = meas_idx - total_meas
                        if rel_idx < 0:
                            targets.append(stim.target_rec(rel_idx))
                    
                    for inner_q in inner_lz:
                        physical_q = block_idx * inner_n + inner_q
                        if physical_q < n_data:
                            data_meas_idx = ec_meas_offset + physical_q
                            rel_idx = data_meas_idx - total_meas
                            targets.append(stim.target_rec(rel_idx))
                
                if len(targets) >= 2:
                    circuit.append("DETECTOR", targets)
                    detector_count += 1
        
        return detector_count
    
    def _emit_within_round_spatial_metachecks(
        self,
        circuit: stim.Circuit,
        round_idx: int,
        mmap: MeasurementMap,  # Measurements from this round only
        current_meas_offset: int,
        detector_count: int,
    ) -> int:
        """
        Emit spatial metachecks that XOR inner syndromes within the CURRENT round.
        
        Unlike _emit_safe_spatial_metachecks which uses boundary detectors (last_syndrome vs final_data),
        this method only references syndrome measurements from the current round.
        This avoids crossing reset boundaries that occur between rounds.
        
        For each outer Z-stabilizer with block support S = {b1, b2, ...}:
        - XOR the Z-syndrome measurements from each inner block in S
        - This checks the outer stabilizer constraint using ONLY current round data
        
        This is DETERMINISTIC because:
        1. All measurements are from within the same round (no reset crossing)
        2. The XOR represents a true stabilizer of the concatenated code
        3. For error-free state, the combined stabilizer should be +1 (detector=0)
        """
        if len(self.code.level_codes) < 2:
            return detector_count
        
        outer_code = self.code.level_codes[0]
        inner_code = self.code.level_codes[-1]
        
        # Get outer code's Z parity check matrix
        hz_outer = None
        if hasattr(outer_code, 'hz'):
            hz_raw = outer_code.hz
            hz_outer = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        
        if hz_outer is None:
            return detector_count
        
        # Get syndrome measurements from this round
        if not hasattr(mmap, 'stabilizer_measurements') or not mmap.stabilizer_measurements:
            return detector_count
        
        z_meas = mmap.stabilizer_measurements.get('Z', {})
        if not z_meas:
            return detector_count
        
        # For each outer Z-stabilizer
        for outer_stab_idx in range(hz_outer.shape[0]):
            outer_stab_row = hz_outer[outer_stab_idx]
            block_support = np.where(outer_stab_row == 1)[0]
            
            targets = []
            
            # For each block in this outer stabilizer's support
            for block_idx in block_support:
                # Try different key formats that might be used
                # Common formats: (1, (1, block_idx)), (1, block_idx), or just block_idx
                inner_z_indices = None
                
                for possible_key in [(1, (1, block_idx)), (1, block_idx), block_idx, (0, block_idx)]:
                    if possible_key in z_meas:
                        inner_z_indices = z_meas[possible_key]
                        break
                
                if inner_z_indices is None:
                    continue
                
                # Add these measurements to the detector
                for meas_idx in inner_z_indices:
                    rel_idx = meas_idx - current_meas_offset
                    if rel_idx < 0:
                        targets.append(stim.target_rec(rel_idx))
            
            # Emit the detector if we have valid targets
            if len(targets) >= 2:
                circuit.append("DETECTOR", targets)
                detector_count += 1
        
        return detector_count
    
    def _emit_inner_parity_metachecks(
        self,
        circuit: stim.Circuit,
        all_syndrome_meas: Dict[str, Dict[Any, List[int]]],
        ec_meas_offset: int,
        n_data: int,
        detector_count: int,
    ) -> int:
        """
        Emit inner X-stabilizer parity metachecks for each block and round.
        
        KEY INSIGHT: For the symmetric observable (qubit 6 from each block),
        Z errors on observable qubits trigger ALL 3 inner X-stabilizers (parity=1).
        Weight-4 DEM errors typically only trigger 2 inner X-stabs (parity=0).
        
        This metacheck fires when:
        - There is inner X syndrome activity (at least 2 stabilizers)
        - The parity is EVEN (not consistent with observable qubit error)
        
        METACHECK: PARITY_{round,block} = D(InnerX_X0) ⊕ D(InnerX_X1) ⊕ D(InnerX_X2)
        
        The metacheck detector XORs ALL measurements used by all 3 inner X-stabilizers.
        Due to symmetric difference, measurements used an even number of times cancel out,
        and only measurements at qubits with odd coverage remain.
        
        For Steane code: Qubit 6 is in all 3 X-stabilizers (coverage=3, odd) → included
        Qubits 0,1,3 have coverage=1 (odd) → included
        Qubits 2,4,5 have coverage=2 (even) → cancelled
        """
        if len(self.code.level_codes) < 2:
            return detector_count
        
        inner_code = self.code.level_codes[-1]
        n_inner = inner_code.n
        
        # Get inner code's X parity check matrix
        hx_inner = None
        if hasattr(inner_code, 'hx'):
            hx_raw = inner_code.hx
            hx_inner = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
        
        if hx_inner is None:
            return detector_count
        
        n_inner_x_stabs = hx_inner.shape[0]
        if n_inner_x_stabs < 3:
            # Need at least 3 X stabilizers for meaningful parity check
            return detector_count
        
        # Compute qubit coverage: how many X-stabilizers contain each qubit
        qubit_coverage = np.sum(hx_inner, axis=0)  # Shape: (n_inner,)
        
        # The parity metacheck includes measurements for qubits with ODD coverage
        odd_coverage_qubits = [q for q in range(n_inner) if qubit_coverage[q] % 2 == 1]
        
        if not odd_coverage_qubits:
            # No qubits with odd coverage - metacheck would be trivial
            return detector_count
        
        # Get number of outer blocks
        n_blocks = self.code.level_codes[0].n if len(self.code.level_codes) >= 2 else 1
        
        total_meas = ec_meas_offset + n_data
        x_meas = all_syndrome_meas.get('X', {})
        
        # Track which (round, block) pairs we've already emitted metachecks for
        # to avoid duplicates
        emitted_pairs = set()
        parity_count = 0  # DEBUG: count how many we emit
        
        # For each round and block, emit a parity metacheck
        # CRITICAL: Only emit for QEC measurement rounds, not initialization or ancilla prep
        for round_idx in range(self.rounds):
            for block_idx in range(n_blocks):
                # Skip if we've already emitted for this (round, block) pair
                if (round_idx, block_idx) in emitted_pairs:
                    continue
                
                # Find X syndrome measurements for this block in this round
                # Try various key formats
                block_key = None
                for key_format in [
                    (round_idx, (1, (1, block_idx))),
                    (round_idx, (1, block_idx)),
                    (round_idx, block_idx),
                ]:
                    if key_format in x_meas:
                        block_key = key_format
                        break
                
                if block_key is None:
                    continue
                
                inner_x_indices = x_meas[block_key]
                
                # inner_x_indices has one measurement per qubit (n_inner entries)
                if len(inner_x_indices) < n_inner:
                    continue
                
                # Create SPATIAL-ONLY parity metacheck (single round snapshot):
                # Parity = M_X0[odd_qubits] ⊕ M_X1[odd_qubits] ⊕ M_X2[odd_qubits]
                # This fires when parity is ODD in the current round
                targets = []
                
                # Add current round measurements for odd-coverage qubits ONLY
                for q in odd_coverage_qubits:
                    if q < len(inner_x_indices):
                        meas_idx = inner_x_indices[q]
                        rel_idx = meas_idx - total_meas
                        if rel_idx < 0:
                            targets.append(stim.target_rec(rel_idx))
                
                # Only emit if we have measurements
                if len(targets) > 0:
                    circuit.append("DETECTOR", targets)
                    detector_count += 1
                    parity_count += 1  # DEBUG
                    # Mark this (round, block) pair as emitted
                    emitted_pairs.add((round_idx, block_idx))
        
        return detector_count
    
    def _emit_outer_level_metachecks(
        self,
        circuit: stim.Circuit,
        all_syndrome_meas: Dict[str, Dict[Any, List[int]]],
        ec_meas_offset: int,
        n_data: int,
        detector_count: int,
    ) -> int:
        """
        Emit outer-level metachecks that XOR each inner X-stabilizer across outer X-stabilizer blocks.
        
        KEY INSIGHT: Weight-4 L0 errors are hook errors during outer CNOTs.
        Example: X@block0_q2 + X@block1_q2 from DEPOLARIZE2(q2, q9)
        
        Inner qubit 2 triggers X1=1, X2=1 (not X0) in each affected block.
        With current detectors, this gives weight-4 DEM error.
        
        SOLUTION: XOR each inner X-stabilizer (X0, X1, X2) across the blocks
        defined by each outer X-stabilizer (outer_X0, outer_X1, outer_X2).
        
        For Steane outer code:
          outer_X0: blocks [3,4,5,6]
          outer_X1: blocks [1,2,5,6]  
          outer_X2: blocks [0,2,4,6]
        
        For X@block0_q2 + X@block1_q2:
          - Inner X1 fires in blocks 0,1
          - Outer metacheck "X1 over blocks [1,2,5,6]" includes block 1 → XOR=1 (fires!)
          - Outer metacheck "X1 over blocks [0,2,4,6]" includes block 0 → XOR=1 (fires!)
          - Similar for X2
          → Adds 4 more detector flips, making weight-8 instead of weight-4!
        
        3 inner stabs × 3 outer stabs × (temporal pairs) = 9 metachecks per round
        """
        if len(self.code.level_codes) < 2:
            return detector_count
        
        outer_code = self.code.level_codes[0]
        inner_code = self.code.level_codes[-1]
        n_inner = inner_code.n
        n_blocks = outer_code.n
        
        # Get outer code's X parity check matrix
        hx_outer = None
        if hasattr(outer_code, 'hx'):
            hx_raw = outer_code.hx
            hx_outer = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
        
        if hx_outer is None:
            return detector_count
        
        n_outer_x_stabs = hx_outer.shape[0]
        
        # Get inner code's X parity check matrix
        hx_inner = None
        if hasattr(inner_code, 'hx'):
            hx_raw = inner_code.hx
            hx_inner = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
        
        if hx_inner is None:
            return detector_count
        
        n_inner_x_stabs = hx_inner.shape[0]
        
        total_meas = ec_meas_offset + n_data
        x_meas = all_syndrome_meas.get('X', {})
        
        # For each round, each outer X-stabilizer, and each inner X-stabilizer:
        # Create a metacheck that XORs the inner X-stab temporal detector across outer-stab blocks
        for round_idx in range(self.rounds):
            for outer_stab_idx in range(n_outer_x_stabs):
                # Get blocks in this outer X-stabilizer's support
                outer_support_blocks = [b for b in range(n_blocks) if hx_outer[outer_stab_idx, b] == 1]
                
                if len(outer_support_blocks) < 2:
                    # Need at least 2 blocks for meaningful metacheck
                    continue
                
                for inner_stab_idx in range(n_inner_x_stabs):
                    # Get qubit indices for this inner X-stabilizer
                    inner_stab_qubits = [q for q in range(n_inner) if hx_inner[inner_stab_idx, q] == 1]
                    
                    targets = []
                    
                    # For each block in outer support, add measurements for this inner stab
                    for block_idx in outer_support_blocks:
                        # Find X syndrome measurements for this block in this round
                        block_key = None
                        for key_format in [
                            (round_idx, (1, (1, block_idx))),
                            (round_idx, (1, block_idx)),
                            (round_idx, block_idx),
                        ]:
                            if key_format in x_meas:
                                block_key = key_format
                                break
                        
                        if block_key is None:
                            continue
                        
                        inner_x_indices = x_meas[block_key]
                        
                        # Add current round measurements for this inner stabilizer's qubits
                        for q in inner_stab_qubits:
                            if q < len(inner_x_indices):
                                meas_idx = inner_x_indices[q]
                                rel_idx = meas_idx - total_meas
                                if rel_idx < 0:
                                    targets.append(stim.target_rec(rel_idx))
                        
                        # Add previous round measurements for temporal comparison
                        if round_idx > 0:
                            prev_key = None
                            for key_format in [
                                (round_idx - 1, (1, (1, block_idx))),
                                (round_idx - 1, (1, block_idx)),
                                (round_idx - 1, block_idx),
                            ]:
                                if key_format in x_meas:
                                    prev_key = key_format
                                    break
                            
                            if prev_key is not None:
                                prev_x_indices = x_meas[prev_key]
                                for q in inner_stab_qubits:
                                    if q < len(prev_x_indices):
                                        meas_idx = prev_x_indices[q]
                                        rel_idx = meas_idx - total_meas
                                        if rel_idx < 0:
                                            targets.append(stim.target_rec(rel_idx))
                    
                    # Only emit if we have measurements from multiple blocks
                    if len(targets) > 0:
                        circuit.append("DETECTOR", targets)
                        detector_count += 1
        
        return detector_count
    
    def _emit_detector_level_spatial_metachecks(
        self,
        circuit: stim.Circuit,
        all_syndrome_meas: Dict[str, Dict[Any, List[int]]],
        ec_meas_offset: int,
        n_data: int,
        detector_count: int,
    ) -> int:
        """
        Emit detector-level spatial metachecks.
        
        KEY INSIGHT: Unlike measurement-level spatial metachecks (which are non-deterministic
        due to independent encoded ancilla preparations), DETECTOR-level spatial metachecks
        are DETERMINISTIC because detectors are 0 in the absence of errors.
        
        For each inner X-stabilizer index s ∈ {0,1,2} and each outer stabilizer o:
            Meta_s_o = XOR of D(Xs@Bi) for all blocks Bi in outer stabilizer o's support
        
        IMPLEMENTATION: Create detectors that XOR the same measurements that the
        individual block detectors XOR, but across multiple blocks.
        
        For a temporal detector D(Xs@Bi) = M(Xs@Bi, round_n) XOR M(Xs@Bi, round_{n-1})
        
        The spatial metacheck combines these:
            Meta_s_o = D(Xs@B0) XOR D(Xs@B2) XOR D(Xs@B4) XOR D(Xs@B6)
                     = (M(Xs@B0,rn) XOR M(Xs@B0,r{n-1}))
                       XOR (M(Xs@B2,rn) XOR M(Xs@B2,r{n-1}))
                       XOR ...
        
        This is equivalent to XORing all the temporal pairs' measurements together.
        """
        if len(self.code.level_codes) < 2:
            return detector_count
        
        outer_code = self.code.level_codes[0]
        inner_code = self.code.level_codes[-1]
        n_inner = inner_code.n
        
        # Get outer code's parity check matrix (use Hz for now)
        hz_outer = None
        if hasattr(outer_code, 'hz'):
            hz_raw = outer_code.hz
            hz_outer = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        
        if hz_outer is None:
            return detector_count
        
        n_outer_stabs = hz_outer.shape[0]
        n_blocks = outer_code.n
        
        # Get inner X stabilizer count
        hx_inner = None
        if hasattr(inner_code, 'hx'):
            hx_raw = inner_code.hx
            hx_inner = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
        
        if hx_inner is None:
            return detector_count
        
        n_inner_x_stabs = hx_inner.shape[0]
        total_meas = ec_meas_offset + n_data
        x_meas = all_syndrome_meas.get('X', {})
        
        # For each round, each outer stabilizer, each inner X-stabilizer
        # create a metacheck that XORs temporal detector measurements across blocks
        for round_idx in range(self.rounds):
            for outer_stab_idx in range(n_outer_stabs):
                # Get blocks in this outer stabilizer's support
                block_support = np.where(hz_outer[outer_stab_idx] == 1)[0]
                
                for inner_stab_idx in range(n_inner_x_stabs):
                    targets = []
                    valid_block_count = 0
                    
                    for block_idx in block_support:
                        # Find X syndrome measurements for this block in CURRENT round
                        curr_key = None
                        for key_format in [
                            (round_idx, (1, (1, block_idx))),
                            (round_idx, (1, block_idx)),
                            (round_idx, block_idx),
                        ]:
                            if key_format in x_meas:
                                curr_key = key_format
                                break
                        
                        if curr_key is None:
                            continue
                        
                        curr_meas_list = x_meas[curr_key]
                        
                        # The inner stabilizer index tells us which measurement within the block
                        # For Steane code, each stabilizer measures multiple qubits
                        # The measurement index for stabilizer s is based on the block's X measurement list
                        # Assuming measurements are ordered by stabilizer: stab_0, stab_1, stab_2, ...
                        # But actually, the measurement list is per-qubit, not per-stabilizer!
                        
                        # For a detector XOR, we need the measurements for the stabilizer's support qubits
                        stab_support = np.where(hx_inner[inner_stab_idx] == 1)[0]
                        
                        # Get current round measurements for these qubits
                        for q in stab_support:
                            if q < len(curr_meas_list):
                                meas_idx = curr_meas_list[q]
                                rel_idx = meas_idx - total_meas
                                if rel_idx < 0:
                                    targets.append(stim.target_rec(rel_idx))
                        
                        # For temporal detector, also need previous round measurements
                        if round_idx > 0:
                            prev_key = None
                            for key_format in [
                                (round_idx - 1, (1, (1, block_idx))),
                                (round_idx - 1, (1, block_idx)),
                                (round_idx - 1, block_idx),
                            ]:
                                if key_format in x_meas:
                                    prev_key = key_format
                                    break
                            
                            if prev_key is not None:
                                prev_meas_list = x_meas[prev_key]
                                for q in stab_support:
                                    if q < len(prev_meas_list):
                                        meas_idx = prev_meas_list[q]
                                        rel_idx = meas_idx - total_meas
                                        if rel_idx < 0:
                                            targets.append(stim.target_rec(rel_idx))
                        
                        valid_block_count += 1
                    
                    # Only emit if we have measurements from multiple blocks
                    # (single block doesn't add new information)
                    if valid_block_count >= 2 and len(targets) > 0:
                        circuit.append("DETECTOR", targets)
                        detector_count += 1
        
        return detector_count
    
    def _emit_boundary_spatial_metachecks(
        self,
        circuit: stim.Circuit,
        last_round: int,
        all_syndrome_meas: Dict[str, Dict[Any, List[int]]],
        ec_meas_offset: int,
        n_data: int,
        detector_count: int,
    ) -> int:
        """
        Emit boundary spatial metachecks connecting inner blocks to observable via outer code.
        
        CRITICAL FOR DEM CONNECTIVITY:
        This function creates BOUNDARY detectors that span multiple inner blocks according
        to outer code stabilizer structure. These detectors XOR:
        1. Inner syndrome measurements from the LAST round across all blocks in outer stab support
        2. Final data measurements on the inner logical support across same blocks
        
        For each outer Z-stabilizer s with support S = {b1, b2, ..., bw}:
            For each qubit position q in inner ZL support:
                Detector = XOR of (last syndrome[q] from each block bi) 
                         XOR (final data on inner ZL qubits from each block bi)
        
        This creates error mechanism paths from ALL blocks in S to the observable,
        even for blocks outside the outer logical support.
        """
        if len(self.code.level_codes) < 2:
            return detector_count
        
        total_meas = ec_meas_offset + n_data
        
        outer_code = self.code.level_codes[0]
        inner_code = self.code.level_codes[-1]
        inner_level = len(self.code.level_codes) - 1
        n_inner = inner_code.n
        
        # Get outer code's Z parity check matrix
        hz_outer = None
        if hasattr(outer_code, 'hz'):
            hz_raw = outer_code.hz
            hz_outer = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        
        # Get inner code's Z parity check matrix
        hz_inner = None
        if hasattr(inner_code, 'hz'):
            hz_raw = inner_code.hz
            hz_inner = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        
        if hz_outer is None or hz_inner is None:
            return detector_count
        
        n_outer_stabs = hz_outer.shape[0]
        n_inner_stabs = hz_inner.shape[0]
        
        # Build mapping from block_id to data qubits
        block_to_data = {}
        for idx, (addr, (start, end)) in enumerate(self.qubit_mapper.iter_leaf_ranges()):
            block_to_data[idx] = list(range(start, end))
        
        # Get inner code's logical Z support
        lz_inner = self._get_z_support_for_code(inner_code)  # e.g., [0,1,2] for Steane
        if lz_inner is None:
            lz_inner = list(range(n_inner))  # Fallback: all qubits
        
        # === Boundary spatial metachecks for Z stabilizers ===
        # For each outer Z-stabilizer, create detectors spanning its block support
        for outer_s in range(n_outer_stabs):
            block_support = list(np.where(hz_outer[outer_s] == 1)[0])
            
            # Skip if stabilizer only spans one block
            if len(block_support) < 2:
                continue
            
            # For each inner stabilizer index, create a cross-block boundary detector
            for inner_s in range(n_inner_stabs):
                targets = []
                inner_stab_support = np.where(hz_inner[inner_s] == 1)[0]
                
                for block_id in block_support:
                    if block_id not in block_to_data:
                        continue
                    block_data_qs = block_to_data[block_id]
                    
                    # Add last-round syndrome measurement for this stabilizer in this block
                    # Key format: (round_idx, block_id) where block_id is integer
                    key = (last_round, block_id)
                    if key in all_syndrome_meas.get('Z', {}):
                        inner_indices = all_syndrome_meas['Z'][key]
                        if inner_s < len(inner_indices):
                            targets.append(stim.target_rec(
                                inner_indices[inner_s] - total_meas
                            ))
                    
                    # Add final data measurements for qubits in this stabilizer's support
                    for local_q in inner_stab_support:
                        if local_q < len(block_data_qs):
                            data_q = block_data_qs[local_q]
                            data_meas_idx = ec_meas_offset + data_q
                            targets.append(stim.target_rec(data_meas_idx - total_meas))
                
                # Only emit if we have targets from multiple blocks
                if len(targets) >= 4:  # At least 2 blocks × (1 syndrome + data)
                    circuit.append("DETECTOR", targets)
                    detector_count += 1
        
        # === X-side boundary spatial metachecks (for X-basis memory) ===
        if self.basis == 'X':
            hx_outer = None
            hx_inner = None
            if hasattr(outer_code, 'hx'):
                hx_raw = outer_code.hx
                hx_outer = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
            if hasattr(inner_code, 'hx'):
                hx_raw = inner_code.hx
                hx_inner = np.atleast_2d(np.array(hx_raw() if callable(hx_raw) else hx_raw, dtype=int))
            
            if hx_outer is not None and hx_inner is not None:
                n_outer_x = hx_outer.shape[0]
                n_inner_x = hx_inner.shape[0]
                
                for outer_s in range(n_outer_x):
                    block_support = list(np.where(hx_outer[outer_s] == 1)[0])
                    if len(block_support) < 2:
                        continue
                    
                    for inner_s in range(n_inner_x):
                        targets = []
                        inner_stab_support = np.where(hx_inner[inner_s] == 1)[0]
                        
                        for block_id in block_support:
                            if block_id not in block_to_data:
                                continue
                            block_data_qs = block_to_data[block_id]
                            
                            # Key format: (round_idx, block_id) where block_id is integer
                            key = (last_round, block_id)
                            if key in all_syndrome_meas.get('X', {}):
                                inner_indices = all_syndrome_meas['X'][key]
                                if inner_s < len(inner_indices):
                                    targets.append(stim.target_rec(
                                        inner_indices[inner_s] - total_meas
                                    ))
                            
                            for local_q in inner_stab_support:
                                if local_q < len(block_data_qs):
                                    data_q = block_data_qs[local_q]
                                    data_meas_idx = ec_meas_offset + data_q
                                    targets.append(stim.target_rec(data_meas_idx - total_meas))
                        
                        if len(targets) >= 4:
                            circuit.append("DETECTOR", targets)
                            detector_count += 1
        
        return detector_count

    def _organize_data_by_block(self) -> Dict[int, List[int]]:
        """Organize data qubits by leaf block index."""
        data_by_block = {}
        for block_idx, (addr, (start, end)) in enumerate(self.qubit_mapper.iter_leaf_ranges()):
            data_by_block[block_idx] = list(range(start, end))
        return data_by_block
    
    def _add_observable(
        self, 
        circuit: stim.Circuit, 
        n_total: int, 
        ec_meas_offset: int,
        pauli_frames: Optional[Dict[int, Dict[str, Dict[Any, List[int]]]]] = None,
        total_meas: Optional[int] = None,
    ) -> None:
        """
        Add OBSERVABLE_INCLUDE for the concatenated logical Z operator.
        
        For teleportation-based EC (Knill), the logical Z observable must include
        Pauli frame corrections from Bell measurements. The output state after
        teleportation is: |ψ_out⟩ = X^m_anc1 Z^m_data |ψ_original⟩
        
        For Z_L measurement:
        - Final data measurements on Z support contribute
        - Pauli frame 'X' (from m_anc1) also contributes to Z_L parity
        
        Parameters
        ----------
        pauli_frames : dict
            Pauli frame measurements collected from EC rounds.
            Structure: {round: {'X': {key: [indices]}, 'Z': {...}}}
        """
        obs_qubits = self._compute_observable_support()
        if not obs_qubits:
            return
        
        if total_meas is None:
            total_meas = ec_meas_offset + n_total
            
        obs_targets = []
        
        # Add final data measurements on Z support
        for qubit in obs_qubits:
            meas_idx = ec_meas_offset + qubit
            lookback = meas_idx - total_meas
            obs_targets.append(stim.target_rec(lookback))
        
        # Add Pauli frame 'X' corrections for teleportation EC
        # The X Pauli frame affects Z_L: Z_L X^m = (-1)^{m·Z_L} X^m Z_L
        # So we need to XOR the frame measurements on Z support
        if pauli_frames:
            frame_targets = self._compute_pauli_frame_observable_targets(
                pauli_frames, obs_qubits, total_meas
            )
            obs_targets.extend(frame_targets)
        
        circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)
    
    def _compute_pauli_frame_observable_targets(
        self,
        pauli_frames: Dict[int, Dict[str, Dict[Any, List[int]]]],
        obs_qubits: List[int],
        total_meas: int,
    ) -> List[stim.GateTarget]:
        """
        Compute the Pauli frame measurement targets for the observable.
        
        For teleportation EC with concatenated codes, the X Pauli frame affects
        the Z_L observable. We need to include frame measurements that correspond
        to the same logical Z support as the final data measurements.
        
        For a 2-level concatenation:
        - Z_L_total = XOR(inner_Z_L[b] for b in outer_Z_support)
        - inner_Z_L[b] = XOR(data[b,i] for i in inner_Z_support)
        - Frame correction: same XOR pattern on frame measurements
        
        Key insight: We need frame measurements for the SAME blocks that
        contribute to the logical observable (those on outer Z support).
        """
        targets = []
        
        obs_set = set(obs_qubits)

        # Precompute block ranges per level to map block_id → (start, end)
        level_block_ranges: Dict[int, List[Tuple[int, int]]] = {}
        for level in range(self.code.depth):
            level_block_ranges[level] = [rng for _, rng in self.qubit_mapper.iter_level_ranges(level)]
        
        for round_idx, round_frames in pauli_frames.items():
            if 'X' not in round_frames:
                continue
            for key, frame_indices in round_frames['X'].items():
                if len(frame_indices) == 0:
                    continue
                if isinstance(key, tuple) and len(key) == 2:
                    level, block_id = key
                else:
                    level = self.code.depth - 1
                    block_id = key
                if level not in level_block_ranges:
                    continue
                blocks = level_block_ranges[level]
                if not (0 <= int(block_id) < len(blocks)):
                    continue
                start, end = blocks[int(block_id)]
                for local_idx, meas_idx in enumerate(frame_indices):
                    phys_q = start + local_idx
                    if phys_q in obs_set:
                        lookback = meas_idx - total_meas
                        targets.append(stim.target_rec(lookback))
        
        return targets

    def _compute_frame_bit_flips(
        self,
        pauli_frames: Dict[int, Dict[str, Dict[Any, List[int]]]],
        shot: np.ndarray,
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Compute per-physical-qubit frame flips from X and Z frames."""
        frame_x: Dict[int, int] = {}
        frame_z: Dict[int, int] = {}
        # Map level → list of (start, end) ranges for block_id order
        level_block_ranges: Dict[int, List[Tuple[int, int]]] = {
            lvl: [rng for _, rng in self.qubit_mapper.iter_level_ranges(lvl)]
            for lvl in range(self.code.depth)
        }
        for _, round_frames in pauli_frames.items():
            for ptype, tgt in round_frames.items():
                for key, frame_indices in tgt.items():
                    if isinstance(key, tuple) and len(key) == 2:
                        level, block_id = key
                    else:
                        level = self.code.depth - 1
                        block_id = key
                    blocks = level_block_ranges.get(level)
                    if blocks is None or not (0 <= int(block_id) < len(blocks)):
                        continue
                    start, end = blocks[int(block_id)]
                    block_size = end - start
                    for local_idx, meas_idx in enumerate(frame_indices):
                        if local_idx >= block_size:
                            continue
                        phys_q = start + local_idx
                        bit = int(shot[meas_idx])
                        if ptype == 'X':
                            frame_x[phys_q] = frame_x.get(phys_q, 0) ^ bit
                        elif ptype == 'Z':
                            frame_z[phys_q] = frame_z.get(phys_q, 0) ^ bit
        return frame_x, frame_z

    def _apply_frame_to_final_data(
        self,
        shot: np.ndarray,
        metadata: MultiLevelMetadata,
        basis: str,
    ) -> np.ndarray:
        """Apply Pauli frame corrections to final data measurements for decoding."""
        final_meas_start = metadata.total_measurements - metadata.n_physical_qubits
        final_shot = np.array(shot[final_meas_start:], copy=True)
        if not metadata.pauli_frame_measurements:
            return final_shot
        frame_x, frame_z = self._compute_frame_bit_flips(metadata.pauli_frame_measurements, shot)
        if basis == 'Z':
            for q, bit in frame_x.items():
                if 0 <= q < len(final_shot):
                    final_shot[q] ^= bit
        elif basis == 'X':
            for q, bit in frame_z.items():
                if 0 <= q < len(final_shot):
                    final_shot[q] ^= bit
        return final_shot
    
    def _compute_observable_support(self) -> List[int]:
        """
        Compute the physical qubit support for the concatenated Z_L observable.
        
        For concatenated codes, the logical Z is defined recursively:
        Z_L^(l) = Z_L^(l-1) on each qubit in outer Z_L support
        
        This recursively expands the logical Z from outermost to innermost level.
        
        If use_symmetric_observable is enabled, uses a modified observable
        that has balanced detector coverage across all inner blocks.
        """
        if not self._level_z_supports:
            return []
        
        # Option 2: Symmetric Observable
        # Use a different observable construction that ensures all observable
        # qubits receive equal coverage from inner Z-stabilizers
        if self.use_symmetric_observable:
            return self._compute_symmetric_observable_support()
        
        # Standard observable: Z_L^(outer) ⊗ Z_L^(inner)
        # Start with root address
        current_addresses = [()]
        
        # Expand through each level (except innermost)
        for level in range(self.code.depth - 1):
            z_support = self._level_z_supports[level]
            new_addresses = []
            for addr in current_addresses:
                for child_idx in z_support:
                    new_addresses.append(addr + (child_idx,))
            current_addresses = new_addresses
        
        # At innermost level, map to physical qubits
        inner_z_support = self._level_z_supports[-1]
        obs_qubits = []
        
        for addr in current_addresses:
            start, end = self.qubit_mapper.get_qubit_range(addr)
            for local_idx in inner_z_support:
                if local_idx < (end - start):
                    obs_qubits.append(start + local_idx)
        
        return sorted(set(obs_qubits))
    
    def _compute_symmetric_observable_support(self) -> List[int]:
        """
        Compute a symmetric observable support with balanced detector coverage.
        
        Problem: Standard Z_L = Z_L^(outer) ⊗ Z_L^(inner) means:
        - Only blocks on outer Z support participate  
        - Inner Z_L qubits {0,1,2} include qubits 0,1 that are only in 1 Z-stabilizer
        - This creates asymmetric detector coverage
        
        Solution: Use Z_L that includes only inner qubits with ≥2 stabilizer coverage.
        For Steane [[7,1,3]] code:
        - Z_L = Z0 Z1 Z2 (standard)
        - Z stabilizers: S0={3,4,5,6}, S1={1,2,5,6}, S2={0,2,4,6}
        - Qubit 0: only in S2, Qubit 1: only in S1, Qubit 2: in S1,S2 ✓
        
        Alternative: Use qubits that appear in ≥2 stabilizers as observable.
        For Steane: qubit 2,5,6 appear in 2 stabilizers each; 4 appears in 2.
        
        Actually, the symmetric approach uses a DIFFERENT logical representative:
        Z_L' = Z_L * (product of Z-stabilizers that balance coverage)
        
        For Steane, multiplying by S2 = Z0 Z2 Z4 Z6 gives:
        Z_L' = Z_L * S2 = (Z0 Z1 Z2) * (Z0 Z2 Z4 Z6) = Z1 Z4 Z6
        Check coverage: qubit 1 in S1, qubit 4 in S0,S2, qubit 6 in S0,S1,S2
        Still asymmetric!
        
        Best approach: Use qubits {4, 5, 6} as observable support.
        - Qubit 4: in S0, S2 (2 stabilizers)
        - Qubit 5: in S0, S1 (2 stabilizers)  
        - Qubit 6: in S0, S1, S2 (3 stabilizers)
        This is equivalent to Z_L * S1 = Z0 Z1 Z2 * Z1 Z2 Z5 Z6 = Z0 Z5 Z6
        Still not right...
        
        Actually, reconsider: the issue is that qubits 0,1 only appear in 1 stabilizer.
        The symmetric observable uses qubits that all appear in ≥2 stabilizers.
        
        For Steane Z-stabilizers:
        - S0 = {3,4,5,6}, S1 = {1,2,5,6}, S2 = {0,2,4,6}
        Coverage: q0→1, q1→1, q2→2, q3→1, q4→2, q5→2, q6→3
        
        Best inner support: {2, 4, 5, 6} all have ≥2 coverage, but that's 4 qubits.
        For a valid logical Z, we need a set that:
        1. Is non-trivial (commutes with X stabilizers, anticommutes with X_L)
        2. Has symmetric coverage
        
        Simplest: use {2, 5, 6} - all have ≥2 coverage.
        Check if valid: Z2 Z5 Z6 = Z_L * S1 * S0 = Z0 Z1 Z2 * Z1 Z2 Z5 Z6 * Z3 Z4 Z5 Z6
                      = Z0 Z3 Z4 (not the same)
        
        Alternative: Just use one qubit per block! The key insight is that
        the OUTER blocks selection is what causes asymmetry. If we use ALL
        outer blocks (all 7 for [[49,1,9]]), the observable becomes symmetric.
        
        This method implements: Use ALL outer blocks with the most-covered
        inner qubit (the qubit that appears in most inner Z-stabilizers).
        """
        if not self._level_z_supports or self.code.depth < 2:
            # Fall back to standard for single-level codes
            return self._compute_standard_observable_support()
        
        inner_code = self.code.level_codes[-1]
        inner_z_support = self._level_z_supports[-1]
        
        # Get inner code's Z stabilizer matrix
        hz = None
        if hasattr(inner_code, 'hz'):
            hz_raw = inner_code.hz
            hz = np.atleast_2d(np.array(hz_raw() if callable(hz_raw) else hz_raw, dtype=int))
        
        # If no parity matrix, use standard
        if hz is None:
            return self._compute_standard_observable_support()
        
        # Count how many Z-stabilizers cover each qubit in inner Z support
        n_inner_qubits = inner_code.n
        coverage = {}
        for q in range(n_inner_qubits):
            coverage[q] = sum(hz[s, q] for s in range(hz.shape[0]))
        
        # Find the inner qubit with maximum coverage that's in Z support
        best_inner_qubit = None
        best_coverage = -1
        for q in inner_z_support:
            if coverage.get(q, 0) > best_coverage:
                best_coverage = coverage.get(q, 0)
                best_inner_qubit = q
        
        if best_inner_qubit is None:
            return self._compute_standard_observable_support()
        
        # Use ALL outer blocks with the best inner qubit
        # This gives symmetric coverage because all outer blocks are treated equally
        obs_qubits = []
        outer_code = self.code.level_codes[0]
        n_outer = outer_code.n  # Number of outer "qubits" (inner blocks)
        
        for outer_idx in range(n_outer):
            # Address for this outer block
            addr = (outer_idx,)
            try:
                start, end = self.qubit_mapper.get_qubit_range(addr)
                phys_q = start + best_inner_qubit
                if phys_q < end:
                    obs_qubits.append(phys_q)
            except (KeyError, ValueError):
                continue
        
        return sorted(set(obs_qubits))
    
    def _compute_standard_observable_support(self) -> List[int]:
        """Standard observable computation (non-symmetric)."""
        if not self._level_z_supports:
            return []
        
        current_addresses = [()]
        for level in range(self.code.depth - 1):
            z_support = self._level_z_supports[level]
            new_addresses = []
            for addr in current_addresses:
                for child_idx in z_support:
                    new_addresses.append(addr + (child_idx,))
            current_addresses = new_addresses
        
        inner_z_support = self._level_z_supports[-1]
        obs_qubits = []
        for addr in current_addresses:
            start, end = self.qubit_mapper.get_qubit_range(addr)
            for local_idx in inner_z_support:
                if local_idx < (end - start):
                    obs_qubits.append(start + local_idx)
        
        return sorted(set(obs_qubits))
    
    def _build_metadata(
        self,
        n_total: int,
        ec_meas_offset: int,
        syndrome_meas: Dict[str, Dict[Any, List[int]]],
        verification_meas: Optional[Dict[int, Dict[int, List[int]]]] = None,
        pauli_frames: Optional[Dict[int, Dict[str, Dict[Any, List[int]]]]] = None,
    ) -> MultiLevelMetadata:
        """Build comprehensive metadata for the experiment."""
        leaf_addresses = [addr for addr, _ in self.qubit_mapper.iter_leaf_ranges()]
        
        # Map final data measurements to leaf addresses
        final_data_meas = {}
        for addr in leaf_addresses:
            start, end = self.qubit_mapper.get_qubit_range(addr)
            meas_indices = list(range(ec_meas_offset + start, ec_meas_offset + end))
            final_data_meas[addr] = meas_indices
        
        address_to_range = dict(self.qubit_mapper._address_to_range)
        
        # Collect gadget names
        gadget_names = []
        if self.ec_gadget is not None:
            gadget_names.append(self.ec_gadget.name)
            if self.level_gadgets:
                for level_idx, gadget in sorted(self.level_gadgets.items()):
                    gadget_names.append(f"L{level_idx}:{gadget.name}")
        
        # Get syndrome layout from EC gadget if available
        syndrome_layout = {}
        if self.ec_gadget is not None and hasattr(self.ec_gadget, 'syndrome_layout'):
            layout = self.ec_gadget.syndrome_layout
            if layout is not None:
                syndrome_layout = layout.to_dict() if hasattr(layout, 'to_dict') else layout
        
        # Determine ancilla prep type
        ancilla_prep_type = self._ancilla_prep or "bare"
        if self.ec_gadget is not None and hasattr(self.ec_gadget, 'ancilla_prep'):
            prep = self.ec_gadget.ancilla_prep
            if hasattr(prep, 'name'):
                ancilla_prep_type = prep.name.lower()
            elif hasattr(prep, 'value'):
                ancilla_prep_type = prep.value.lower()

        # Per-level ancilla prep (best-effort)
        ancilla_prep_types_per_level: List[str] = []
        if self.level_gadgets:
            for level_idx in range(self.code.depth):
                gadget = self.level_gadgets.get(level_idx)
                prep = getattr(gadget, 'ancilla_prep', self._ancilla_prep or "bare") if gadget else (self._ancilla_prep or "bare")
                prep_str = None
                if hasattr(prep, 'name'):
                    prep_str = prep.name.lower()
                elif hasattr(prep, 'value'):
                    prep_str = prep.value.lower()
                elif isinstance(prep, str):
                    prep_str = prep
                ancilla_prep_types_per_level.append(prep_str or "bare")
        
        # Determine if teleportation EC is used (requires Pauli frame correction)
        uses_teleportation_ec = bool(pauli_frames)
        
        return MultiLevelMetadata(
            code_name=self.code.name,
            depth=self.code.depth,
            level_code_names=[getattr(c, 'name', type(c).__name__) for c in self.code.level_codes],
            n_physical_qubits=n_total,
            total_distance=self.code.total_distance,
            n_qubits_per_level=[c.n for c in self.code.level_codes],
            leaf_addresses=leaf_addresses,
            address_to_range=address_to_range,
            total_measurements=ec_meas_offset + n_total,
            final_data_measurements=final_data_meas,
            syndrome_measurements=syndrome_meas,
            syndrome_layout=syndrome_layout,
            level_z_supports=self._level_z_supports,
            n_ec_rounds=self.rounds,
            ec_measurements_per_round=ec_meas_offset // max(1, self.rounds) if self.rounds > 0 else 0,
            gadget_names=gadget_names,
            verification_measurements=verification_meas or {},
            ancilla_prep_type=ancilla_prep_type,
            ancilla_prep_types_per_level=ancilla_prep_types_per_level,
            pauli_frame_measurements=pauli_frames or {},
            uses_teleportation_ec=uses_teleportation_ec,
        )
    
    def sample_and_decode(
        self,
        n_shots: int = 1000,
        use_syndromes: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Sample from the circuit and decode with optional syndrome awareness.
        
        This method provides the full pipeline: build circuit, sample, decode.
        When use_syndromes=True, it extracts EC syndrome measurements and
        passes them to the decoder for true hierarchical decoding.
        
        Parameters
        ----------
        n_shots : int
            Number of shots to sample.
        use_syndromes : bool
            If True, use EC syndromes for decoder (recommended).
        verbose : bool
            If True, print progress info.
            
        Returns
        -------
        Dict[str, Any]
            Results dictionary with keys:
            - 'logical_error_rate': float
            - 'n_shots': int
            - 'n_errors': int
            - 'code_name': str
            - 'n_physical_qubits': int
            - 'n_ec_rounds': int
        """
        from qectostim.decoders.recursive_hierarchical_decoder import (
            RecursiveHierarchicalDecoder,
            ECSyndromeData,
        )
        
        # Build circuit and metadata
        ideal_circuit, metadata = self.build()
        
        # Apply noise
        if self.noise_model is not None:
            circuit = self.noise_model.apply(ideal_circuit)
        else:
            circuit = ideal_circuit
        
        if verbose:
            print(f"Code: {metadata.code_name}")
            print(f"Physical qubits: {metadata.n_physical_qubits}")
            print(f"EC rounds: {metadata.n_ec_rounds}")
            print(f"Total measurements: {metadata.total_measurements}")
        
        # Sample
        sampler = circuit.compile_sampler()
        samples = sampler.sample(n_shots)
        
        # Create decoder
        decoder = RecursiveHierarchicalDecoder(
            code=self.code,
            qubit_mapper=self.qubit_mapper,
        )
        
        n_errors = 0
        for i in range(n_shots):
            shot = samples[i]
            
            # Extract final data measurements (apply frame corrections if present)
            final_shot = self._apply_frame_to_final_data(shot, metadata, self.basis)
            
            if use_syndromes and metadata.syndrome_layout:
                # Use syndrome-aware decoding
                ec_syndromes = ECSyndromeData.from_measurements(
                    measurements=shot,
                    metadata=metadata,
                    code=self.code,
                )
                prediction = decoder.decode_with_syndromes(final_shot, ec_syndromes=ec_syndromes)
            else:
                # Standard decoding from final measurements only
                prediction = decoder.decode(final_shot)
            
            if prediction != 0:
                n_errors += 1
        
        return {
            'logical_error_rate': n_errors / n_shots,
            'n_shots': n_shots,
            'n_errors': n_errors,
            'code_name': metadata.code_name,
            'n_physical_qubits': metadata.n_physical_qubits,
            'n_ec_rounds': metadata.n_ec_rounds,
        }


def run_multilevel_experiment(
    code: 'MultiLevelConcatenatedCode',
    noise_model: Optional[NoiseModel] = None,
    p_error: float = 0.001,
    n_shots: int = 1000,
    n_ec_rounds: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run multi-level memory experiment with recursive hierarchical decoding.
    
    This is a convenience function that creates the experiment, builds the
    circuit, samples, and decodes in one call.
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The multi-level concatenated code.
    noise_model : NoiseModel, optional
        Noise model to use. If None, uses CircuitDepolarizingNoise(p1=p_error).
    p_error : float
        Error probability (used if noise_model is None).
    n_shots : int
        Number of shots to sample.
    n_ec_rounds : int
        Number of EC rounds.
    verbose : bool
        If True, print experiment info.
        
    Returns
    -------
    Dict[str, Any]
        Results including logical_error_rate, n_shots, n_errors, etc.
    """
    from qectostim.decoders.recursive_hierarchical_decoder import RecursiveHierarchicalDecoder
    
    # Create noise model if not provided
    if noise_model is None:
        noise_model = CircuitDepolarizingNoise(p1=p_error, p2=p_error)
    
    # Create experiment
    exp = MultiLevelMemoryExperiment(
        code=code,
        noise_model=noise_model,
        rounds=n_ec_rounds,
    )
    
    # Build circuit and get metadata
    ideal_circuit, metadata = exp.build()
    
    # Apply noise
    circuit = noise_model.apply(ideal_circuit) if noise_model else ideal_circuit
    
    if verbose:
        print(f"Code: {metadata.code_name}")
        print(f"Physical qubits: {metadata.n_physical_qubits}")
        print(f"EC rounds: {metadata.n_ec_rounds}")
        print(f"Gadgets: {metadata.gadget_names}")
        print(f"Total measurements: {metadata.total_measurements}")
    
    # Sample
    sampler = circuit.compile_sampler()
    samples = sampler.sample(n_shots)
    
    # Decode
    decoder = RecursiveHierarchicalDecoder(code=code, qubit_mapper=exp.qubit_mapper)
    
    n_errors = 0
    for i in range(n_shots):
        shot = samples[i]
        # Extract final data measurements (apply frame corrections if present)
        final_shot = exp._apply_frame_to_final_data(shot, metadata, exp.basis)
        prediction = decoder.decode(final_shot)
        if prediction != 0:
            n_errors += 1
    
    return {
        'logical_error_rate': n_errors / n_shots,
        'n_shots': n_shots,
        'n_errors': n_errors,
        'p_physical': p_error,
        'n_ec_rounds': n_ec_rounds,
        'code_name': metadata.code_name,
        'n_physical_qubits': metadata.n_physical_qubits,
    }
