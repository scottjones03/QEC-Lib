# src/qectostim/decoders/hierarchical_v6_variants.py
"""
Hierarchical V6 Decoder - Measurement-Centric Model (MCM) Architecture.

This module implements the hierarchical decoder for concatenated codes using
the Measurement-Centric Model (MCM), which provides a direct mapping from
circuit measurements to the 4-phase decoding pipeline.

Measurement-Centric Model (MCM) Architecture
=============================================
The MCM consists of three data structures that directly map to decoder needs:

  InnerECInstance:
    - One instance per (block, segment) with n_rounds syndrome measurements
    - Contains x_anc_measurements and z_anc_measurements (flat lists)
    - The decoder uses n_rounds to reshape for majority vote or temporal DEM
    - Each instance_id maps directly to decoder's inner_corrections lookup

  OuterSyndromeValue:
    - One per outer stabilizer measurement per round
    - Contains ancilla_ec_instance_id: direct link to inner correction
    - Contains transversal_meas_indices: raw measurement indices
    - Decoder looks up correction via: inner_corrections[ancilla_ec_instance_id]

  FinalDataMeasurement:
    - Per-block final transversal measurement indices
    - Decoder applies cumulative XOR of all inner corrections for that block

This architecture eliminates the ambiguity of segment-based correction routing.

4-Phase Decoding Pipeline
=========================

Phase 1: Inner Error Inference (MCM-based)
    - Iterate over all InnerECInstance objects
    - For each instance with n_rounds measurements:
      - Reshape to (n_rounds, n_stabs) for processing
      - Apply majority vote across rounds (most common outcome per stabilizer)
      - OR apply temporal DEM (consecutive round comparison + MWPM)
    - Output: inner_corrections[instance_id][block_id] = InnerCorrection(x, z)
    - Note: X ancillas detect Z errors → z_correction
            Z ancillas detect X errors → x_correction

Phase 2: Correct Outer Syndrome (MCM-based)
    - For each OuterSyndromeValue:
      - Compute raw logical from transversal_meas_indices (XOR over Z_L support)
      - Look up correction via ancilla_ec_instance_id
      - X stabilizer (|+⟩ ancilla, X-basis measure): use z_correction
      - Z stabilizer (|0⟩ ancilla, Z-basis measure): use x_correction
    - Output: corrected_syn[round][stab_type][stab_idx] = corrected_bit

Phase 3: Correct Data Logicals (MCM-based)
    - For each data block:
      - Get all ec_instance_ids from block_to_ec_instances
      - Compute cumulative XOR of inner corrections
      - Apply to raw final measurement
    - Output: corrected_logicals[block_id] = (x_val, z_val)

Phase 4: Outer Temporal DEM Decode
    - Build outer DEM: D[r,s] = OuterSyn[r+1,s] XOR OuterSyn[r,s]
    - Add final boundary detectors from corrected data logicals
    - Decode using PyMatching
    - Apply correction for final logical prediction

CNOT Layer-by-Layer TICK Placement
==================================
For accurate DEM generation, inner EC rounds now emit TICKs between CNOT layers:

  X stabilizer round (layer-by-layer):
    H(ancillas) → TICK → CNOT_layer_0 → TICK → CNOT_layer_1 → TICK → ... → H(ancillas) → MR

  Z stabilizer round (layer-by-layer):
    CNOT_layer_0 → TICK → CNOT_layer_1 → TICK → ... → MR

This enables the noise model to insert distinguishable errors between CNOT layers,
which is critical for temporal error detection within a single syndrome round.

exRec Structure
===============
Each outer stabilizer measurement uses the extended rectangle (exRec) structure
from Aliferis-Gottesman-Preskill (AGP):

    [Leading EC] → [Outer CNOT layers] → [Trailing EC] → [Measure]

Inner EC is applied before AND after outer CNOTs, creating multiple InnerECInstance
objects per outer gadget. The OuterSyndromeValue links to the trailing EC instance
for correction application.

Literature Foundation
=====================
- Inner DEM: Dennis et al. (quant-ph/0110143), Fowler et al. (arXiv:1208.0928)
- exRec Structure: AGP (quant-ph/0504218) Sections 3-5
- Outer Code Decoding: Chamberland & Jochym-O'Connor (arXiv:1710.06505)
"""
from __future__ import annotations

import numpy as np
import stim
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto

if TYPE_CHECKING:
    from qectostim.codes.abstract_code import Code

try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False

from .base import Decoder

# Import measurement-centric model
from qectostim.experiments.measurement_model import (
    InnerECInstance,
    OuterSyndromeValue,
    FinalDataMeasurement,
    MeasurementCentricMetadata,
)


# =============================================================================
# Data Structures
# =============================================================================

class InnerDecodingMode(Enum):
    """Inner decoding strategy."""
    TEMPORAL_DEM = auto()
    MAJORITY_VOTE = auto()


@dataclass
class DecoderConfig:
    """Configuration for hierarchical decoder."""
    inner_mode: InnerDecodingMode = InnerDecodingMode.TEMPORAL_DEM
    error_prob: float = 0.01
    verbose: bool = False
    skip_inner_corrections: bool = False  # If True, skip Phase 1 inner corrections


@dataclass
class InnerCorrection:
    """Inner correction for a block in an EC instance."""
    x_correction: int = 0
    z_correction: int = 0


# =============================================================================
# Base Class
# =============================================================================

class BaseHierarchicalV6Decoder(Decoder, ABC):
    """
    Base class for hierarchical V6 decoders with complete inner→outer pipeline.
    
    Implements the 4-phase MCM architecture:
    1. Inner error inference per InnerECInstance
    2. Correct outer syndrome using ancilla inner corrections
    3. Correct data logicals using ALL inner corrections
    4. Outer temporal DEM decode
    
    MCM is REQUIRED - the decoder will raise ValueError without it.
    
    Subclasses implement inner decoding strategy via:
    - _compute_inner_correction(block_id, x_anc_meas, z_anc_meas, ec_instance)
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        config: Optional[DecoderConfig] = None,
        basis: str = "Z",
    ):
        self.metadata = metadata
        self.config = config or DecoderConfig()
        self.basis = basis.upper()
        self._shot_count = 0
        
        # Extract code objects
        self.inner_code = metadata.get('inner_code')
        self.outer_code = metadata.get('outer_code')
        
        # Block structure
        self.n_data_blocks = metadata.get('n_data_blocks', 7)
        self.n_x_ancilla_blocks = metadata.get('n_x_ancilla_blocks', 3)
        self.n_z_ancilla_blocks = metadata.get('n_z_ancilla_blocks', 3)
        self.qubits_per_block = metadata.get('qubits_per_block', 13)
        
        # Ancilla block ID ranges
        self._x_ancilla_offset = self.n_data_blocks
        self._z_ancilla_offset = self.n_data_blocks + self.n_x_ancilla_blocks
        
        # Stabilizer to ancilla block mapping (from outer stabilizer engine)
        # Key: (stab_type, stab_idx) -> ancilla_block_id
        self._stab_to_ancilla_block = metadata.get('stab_to_ancilla_block', {})
        
        # Ancilla logical measurements (transversal measurements)
        self._ancilla_logical_measurements = metadata.get('ancilla_logical_measurements', {})
        
        # Final data measurements
        self._final_data_measurements = metadata.get('final_data_measurements', {})
        
        # Outer logical support
        self._outer_logical_support = metadata.get('outer_logical_support', {})
        
        # Outer stabilizer support (which data blocks each stabilizer touches)
        self._outer_stab_support = metadata.get('outer_stab_support', {})
        
        # =================================================================
        # Measurement-Centric Metadata (MCM) - REQUIRED
        # =================================================================
        # MCM provides direct mapping from circuit measurements to decoder phases.
        # Without MCM, the decoder cannot function correctly.
        self._mcm: Optional[MeasurementCentricMetadata] = None
        mcm_dict = metadata.get('measurement_centric_metadata')
        if mcm_dict:
            self._mcm = MeasurementCentricMetadata.from_dict(mcm_dict)
        else:
            raise ValueError(
                "HierarchicalV6Decoder: measurement_centric_metadata is REQUIRED. "
                "MCM provides direct mapping from circuit measurements to decoder phases. "
                "Use emit_inner_ec_segment() from LogicalBlockManagerV2 to generate MCM."
            )
        
        # Setup code structures
        self._setup_inner_code_structure()
        self._setup_outer_code_structure()
        
        # PyMatching decoders
        self._inner_x_matcher = None
        self._inner_z_matcher = None
        self._outer_matcher = None
        
        if HAS_PYMATCHING:
            self._build_inner_matchers()
            self._build_outer_matcher()
    
    # =========================================================================
    # Initialization Helpers
    # =========================================================================
    
    def _setup_inner_code_structure(self) -> None:
        """Setup inner code matrices and logical supports."""
        if self.inner_code is None:
            # Default to Steane [[7,1,3]] code
            # H_X = H_Z for CSS self-dual code
            self._inner_hx = np.array([
                [0, 0, 0, 1, 1, 1, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
            ], dtype=np.uint8)
            self._inner_hz = np.array([
                [0, 0, 0, 1, 1, 1, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
            ], dtype=np.uint8)
            self._inner_z_logical_support = {0, 1, 2}
            self._inner_x_logical_support = {0, 1, 2}
            self._n_inner = 7
            self._n_x_stabs = 3
            self._n_z_stabs = 3
            return
        
        hx = getattr(self.inner_code, 'hx', None)
        hz = getattr(self.inner_code, 'hz', None)
        
        self._inner_hx = np.asarray(hx, dtype=np.uint8) if hx is not None and np.size(hx) > 0 else np.zeros((0, 7), dtype=np.uint8)
        self._inner_hz = np.asarray(hz, dtype=np.uint8) if hz is not None and np.size(hz) > 0 else np.zeros((0, 7), dtype=np.uint8)
        
        self._n_inner = self.inner_code.n
        self._n_x_stabs = self._inner_hx.shape[0] if self._inner_hx.size > 0 else 0
        self._n_z_stabs = self._inner_hz.shape[0] if self._inner_hz.size > 0 else 0
        
        self._inner_z_logical_support = self._get_logical_support(self.inner_code, 'Z')
        self._inner_x_logical_support = self._get_logical_support(self.inner_code, 'X')
    
    def _setup_outer_code_structure(self) -> None:
        """Setup outer code matrices and logical supports."""
        if self.outer_code is None:
            # Default [[7,1,3]] Steane code
            self._outer_hx = np.array([
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 0],
                [1, 0, 1, 0, 1, 0, 1],
            ], dtype=np.uint8)
            self._outer_hz = self._outer_hx.copy()
            self._outer_z_logical_support = {0, 1, 2}
            self._outer_x_logical_support = {0, 1, 2}
            self._n_outer_x_stabs = 3
            self._n_outer_z_stabs = 3
            return
        
        hx = getattr(self.outer_code, 'hx', None)
        hz = getattr(self.outer_code, 'hz', None)
        
        self._outer_hx = np.asarray(hx, dtype=np.uint8) if hx is not None and np.size(hx) > 0 else np.zeros((0, self.n_data_blocks), dtype=np.uint8)
        self._outer_hz = np.asarray(hz, dtype=np.uint8) if hz is not None and np.size(hz) > 0 else np.zeros((0, self.n_data_blocks), dtype=np.uint8)
        
        self._n_outer_x_stabs = self._outer_hx.shape[0] if self._outer_hx.size > 0 else 0
        self._n_outer_z_stabs = self._outer_hz.shape[0] if self._outer_hz.size > 0 else 0
        
        self._outer_z_logical_support = self._get_logical_support(self.outer_code, 'Z')
        self._outer_x_logical_support = self._get_logical_support(self.outer_code, 'X')
    
    def _get_logical_support(self, code: Any, basis: str) -> Set[int]:
        """Get qubit/block indices in logical operator support."""
        try:
            from qectostim.experiments.stabilizer_rounds.utils import get_logical_support
            return set(get_logical_support(code, basis, 0))
        except Exception:
            pass
        
        if basis == 'Z' and hasattr(code, 'lz'):
            lz = np.asarray(code.lz)
            if lz.ndim == 1:
                return set(np.where(lz != 0)[0])
            return set(np.where(lz[0] != 0)[0])
        elif basis == 'X' and hasattr(code, 'lx'):
            lx = np.asarray(code.lx)
            if lx.ndim == 1:
                return set(np.where(lx != 0)[0])
            return set(np.where(lx[0] != 0)[0])
        
        n = getattr(code, 'n', 7)
        d = getattr(code, 'd', 3) if hasattr(code, 'd') else 3
        return set(range(min(d, n)))
    
    def _build_inner_matchers(self) -> None:
        """Build PyMatching decoders for inner code."""
        if not HAS_PYMATCHING:
            return
        try:
            if self._inner_hz.size > 0 and self._inner_hz.shape[0] > 0:
                self._inner_x_matcher = pymatching.Matching(self._inner_hz)
            if self._inner_hx.size > 0 and self._inner_hx.shape[0] > 0:
                self._inner_z_matcher = pymatching.Matching(self._inner_hx)
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not build inner matchers: {e}")
    
    def _build_outer_matcher(self) -> None:
        """Build PyMatching decoder for outer code."""
        if not HAS_PYMATCHING:
            return
        try:
            # Use outer Hz for Z logical (or Hx based on measurement basis)
            if self.basis == "Z":
                if self._outer_hx.size > 0 and self._outer_hx.shape[0] > 0:
                    self._outer_matcher = pymatching.Matching(self._outer_hx)
            else:
                if self._outer_hz.size > 0 and self._outer_hz.shape[0] > 0:
                    self._outer_matcher = pymatching.Matching(self._outer_hz)
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not build outer matcher: {e}")
    
    # =========================================================================
    # Main Decoding Interface
    # =========================================================================
    
    def decode(self, measurements: np.ndarray) -> int:
        """
        Decode raw measurements using 4-phase pipeline.
        
        Phase 1: Inner error inference for each segment
        Phase 2: Correct outer syndrome using ancilla inner corrections
        Phase 3: Correct data logicals using ALL inner corrections + X stabilizer backaction
        Phase 4: Outer temporal DEM decode
        """
        self._shot_count += 1
        
        # Phase 1: Inner error inference
        inner_corrections = self._phase1_inner_error_inference(measurements)
        
        # Phase 2: Correct outer syndrome
        corrected_outer_syndrome = self._phase2_correct_outer_syndrome(
            measurements, inner_corrections
        )
        
        # Phase 3: Correct data logicals (including X stabilizer measurement backaction)
        # Returns both raw and corrected logicals
        raw_data_logicals, corrected_data_logicals = self._phase3_correct_data_logicals(
            measurements, inner_corrections, corrected_outer_syndrome
        )
        
        # Phase 4: Outer temporal DEM decode
        # Use raw logicals for boundary detector, corrected for final logical
        final_logical = self._phase4_outer_decode(
            corrected_outer_syndrome, raw_data_logicals, corrected_data_logicals
        )
        
        return final_logical
    
    def decode_batch(self, measurements_batch: np.ndarray) -> np.ndarray:
        """Decode a batch of measurement samples."""
        n_shots = measurements_batch.shape[0]
        results = np.zeros(n_shots, dtype=np.int32)
        for i in range(n_shots):
            results[i] = self.decode(measurements_batch[i])
        return results
    
    # =========================================================================
    # Phase 1: Inner Error Inference
    # =========================================================================
    
    def _phase1_inner_error_inference(
        self,
        measurements: np.ndarray
    ) -> Dict[int, Dict[int, InnerCorrection]]:
        """
        Phase 1: Compute inner corrections for each EC instance or segment.
        
        NEW MODEL (measurement-centric):
        - Iterates over InnerECInstance objects
        - Each instance maps directly to one (block, EC-application)
        - Returns: instance_id -> block_id -> InnerCorrection
        
        LEGACY MODEL (segment-based):
        - Processes BOTH inner_ec segments AND outer_stab segments
        - Returns: segment_id -> block_id -> InnerCorrection
        
        The new model is simpler because every EC application creates an instance,
        regardless of whether it's in an inner_ec or outer_stab segment.
        
        If config.skip_inner_corrections is True, returns empty corrections.
        This is useful for concatenated codes where inner corrections may be
        unreliable due to high error density per block.
        """
        # Check if we should skip inner corrections entirely
        if self.config.skip_inner_corrections:
            if self.config.verbose:
                print("Phase 1: Skipping inner corrections (skip_inner_corrections=True)")
            return {}
        
        # MCM is required
        if self._mcm is None or not self._mcm.inner_ec_instances:
            import warnings
            warnings.warn(
                "Phase 1: No MCM inner_ec_instances available. "
                "Returning empty corrections. Use emit_inner_ec_segment() to generate MCM.",
                UserWarning
            )
            return {}
        
        return self._phase1_inner_error_inference_mcm(measurements)
    
    def _phase1_inner_error_inference_mcm(
        self,
        measurements: np.ndarray
    ) -> Dict[int, Dict[int, InnerCorrection]]:
        """
        Phase 1 using new measurement-centric model.
        
        Iterates over all InnerECInstance objects directly.
        Each instance has:
        - instance_id: unique ID
        - block_id: which block
        - x_anc_measurements: all X ancilla measurements (n_x_stabs * n_rounds)
        - z_anc_measurements: all Z ancilla measurements (n_z_stabs * n_rounds)
        - n_rounds: number of syndrome extraction rounds
        
        The decoder uses n_rounds to:
        - Validate sufficient rounds for temporal DEM (>= 2)
        - Validate sufficient rounds for majority vote (>= 3)
        - Properly reshape measurements for decoding
        """
        import warnings
        
        corrections: Dict[int, Dict[int, InnerCorrection]] = {}
        
        # Validate n_rounds on first instance
        first_instance = next(iter(self._mcm.inner_ec_instances.values()), None)
        if first_instance is not None:
            n_rounds = first_instance.n_rounds
            
            if n_rounds < 2 and self.config.inner_mode == InnerDecodingMode.TEMPORAL_DEM:
                warnings.warn(
                    f"InnerECInstance has n_rounds={n_rounds} < 2. "
                    f"Temporal DEM requires >= 2 rounds to compute syndrome changes. "
                    f"Inner decoding will return trivial corrections (0).",
                    UserWarning
                )
            
            if n_rounds < 3 and self.config.inner_mode == InnerDecodingMode.MAJORITY_VOTE:
                warnings.warn(
                    f"InnerECInstance has n_rounds={n_rounds} < 3. "
                    f"Majority vote requires >= 3 rounds for robust decoding. "
                    f"Results may be unreliable.",
                    UserWarning
                )
        
        for instance_id, ec_instance in self._mcm.inner_ec_instances.items():
            block_id = ec_instance.block_id
            
            # Get measurements for this EC instance
            x_anc_meas = np.array([
                measurements[i] for i in ec_instance.x_anc_measurements 
                if 0 <= i < len(measurements)
            ], dtype=np.uint8)
            
            z_anc_meas = np.array([
                measurements[i] for i in ec_instance.z_anc_measurements 
                if 0 <= i < len(measurements)
            ], dtype=np.uint8)
            
            # Compute correction using instance-aware method
            x_corr, z_corr = self._compute_inner_correction_for_instance(
                block_id, x_anc_meas, z_anc_meas, ec_instance
            )
            
            if instance_id not in corrections:
                corrections[instance_id] = {}
            
            corrections[instance_id][block_id] = InnerCorrection(
                x_correction=x_corr,
                z_correction=z_corr
            )
        
        return corrections
    
    def _compute_inner_correction_for_instance(
        self,
        block_id: int,
        x_anc_meas: np.ndarray,
        z_anc_meas: np.ndarray,
        ec_instance: InnerECInstance,
    ) -> Tuple[int, int]:
        """
        Compute inner correction for an EC instance.
        
        Subclasses implement the actual decoding logic (temporal DEM or majority vote).
        
        Parameters
        ----------
        block_id : int
            The block ID being corrected
        x_anc_meas : np.ndarray
            X ancilla measurements for this instance
        z_anc_meas : np.ndarray
            Z ancilla measurements for this instance
        ec_instance : InnerECInstance
            The EC instance metadata (includes n_rounds, context, etc.)
            
        Returns
        -------
        Tuple[int, int]
            (x_correction, z_correction) - each 0 or 1
        """
        return self._compute_inner_correction(
            block_id, x_anc_meas, z_anc_meas, ec_instance
        )
    
    @abstractmethod
    def _compute_inner_correction(
        self,
        block_id: int,
        x_anc_meas: np.ndarray,
        z_anc_meas: np.ndarray,
        ec_instance: InnerECInstance,
    ) -> Tuple[int, int]:
        """
        Compute inner X and Z corrections for a block.
        
        Implemented by subclasses for temporal DEM or majority vote.
        
        Parameters
        ----------
        block_id : int
            The block ID being corrected
        x_anc_meas : np.ndarray
            X ancilla measurements, shape [n_rounds * n_x_stabs]
        z_anc_meas : np.ndarray
            Z ancilla measurements, shape [n_rounds * n_z_stabs]
        ec_instance : InnerECInstance
            The EC instance metadata with n_rounds for proper reshaping
        
        Returns
        -------
        Tuple[int, int]
            (x_correction, z_correction) - each 0 or 1
        """
        pass
    
    # =========================================================================
    # Phase 2: Correct Outer Syndrome
    # =========================================================================
    
    def _phase2_correct_outer_syndrome(
        self,
        measurements: np.ndarray,
        inner_corrections: Dict[int, Dict[int, InnerCorrection]]
    ) -> Dict[int, Dict[str, Dict[int, int]]]:
        """
        Phase 2: Apply inner corrections to outer syndrome measurements.
        
        NEW MODEL (measurement-centric):
        - Iterates over OuterSyndromeValue objects
        - Each has ancilla_ec_instance_id linking directly to inner correction
        - Simple lookup: correction = inner_corrections[instance_id][ancilla_block_id]
        
        LEGACY MODEL (segment-based):
        - Looks up corrections from outer_stab segment
        - Requires segment.id to find ancilla corrections
        
        Returns:
            round_idx -> stab_type -> stab_idx -> corrected_value
        """
        # MCM is required
        if self._mcm is None or not self._mcm.outer_syndrome_values:
            import warnings
            warnings.warn(
                "Phase 2: No MCM outer_syndrome_values available. "
                "Returning empty corrected syndrome. Outer syndrome extraction requires MCM.",
                UserWarning
            )
            return {}
        
        return self._phase2_correct_outer_syndrome_mcm(measurements, inner_corrections)
    
    def _phase2_correct_outer_syndrome_mcm(
        self,
        measurements: np.ndarray,
        inner_corrections: Dict[int, Dict[int, InnerCorrection]]
    ) -> Dict[int, Dict[str, Dict[int, int]]]:
        """
        Phase 2 using new measurement-centric model.
        
        Each OuterSyndromeValue has:
        - ancilla_ec_instance_id: directly links to the EC instance
        - transversal_meas_indices: for computing raw logical value
        """
        corrected_syn: Dict[int, Dict[str, Dict[int, int]]] = {}
        
        for outer_syn in self._mcm.outer_syndrome_values:
            round_idx = outer_syn.outer_round
            stab_type = outer_syn.stab_type
            stab_idx = outer_syn.stab_idx
            ancilla_block_id = outer_syn.ancilla_block_id
            ec_instance_id = outer_syn.ancilla_ec_instance_id
            
            # Initialize round dict if needed
            if round_idx not in corrected_syn:
                corrected_syn[round_idx] = {'X': {}, 'Z': {}}
            
            # Get raw logical value from transversal measurement
            raw_value = self._compute_raw_logical_from_indices(
                measurements, outer_syn.transversal_meas_indices
            )
            
            # Get inner correction directly from the linked EC instance
            # The key is the EC instance ID (not segment ID in new model)
            total_correction = 0
            if ec_instance_id in inner_corrections:
                ec_corrections = inner_corrections[ec_instance_id]
                if ancilla_block_id in ec_corrections:
                    # X stabilizer: Z errors flip X measurement → use z_correction
                    # Z stabilizer: X errors flip Z measurement → use x_correction
                    if stab_type == 'X':
                        total_correction = ec_corrections[ancilla_block_id].z_correction
                    else:  # Z stabilizer
                        total_correction = ec_corrections[ancilla_block_id].x_correction
            
            corrected_syn[round_idx][stab_type][stab_idx] = (raw_value + total_correction) % 2
        
        return corrected_syn
    
    def _compute_raw_logical_from_indices(
        self,
        measurements: np.ndarray,
        meas_indices: List[int],
    ) -> int:
        """Compute raw logical value from transversal measurement indices."""
        if not meas_indices:
            return 0
        
        raw_meas = np.array([
            measurements[i] for i in meas_indices if 0 <= i < len(measurements)
        ], dtype=np.uint8)
        
        # XOR over Z logical support
        logical_value = 0
        for idx in self._inner_z_logical_support:
            if idx < len(raw_meas):
                logical_value ^= int(raw_meas[idx])
        
        return logical_value
    
    def _get_raw_ancilla_logical(
        self,
        measurements: np.ndarray,
        round_idx: int,
        stab_type: str,
        stab_idx: int,
        ancilla_block_id: int,
    ) -> int:
        """Get raw logical measurement from ancilla transversal measurement."""
        # Lookup from ancilla_logical_measurements
        meas_key = (round_idx, stab_type, stab_idx)
        meas_indices = self._ancilla_logical_measurements.get(meas_key, [])
        
        if not meas_indices:
            # Try string key
            for key, indices in self._ancilla_logical_measurements.items():
                if (key[0] == round_idx and key[1] == stab_type and key[2] == stab_idx):
                    meas_indices = indices
                    break
        
        if not meas_indices:
            return 0
        
        # Decode raw measurements to logical value
        raw_meas = np.array([measurements[i] for i in meas_indices if 0 <= i < len(measurements)], dtype=np.uint8)
        
        # Always use Z logical support because measurement is always MR (Z-basis)
        # For X stabilizers: H applied before MR, so Z measurement gives X_L value
        # For Z stabilizers: MR directly gives Z_L value
        # In both cases, we XOR over Z logical support of the physical qubits
        logical_support = self._inner_z_logical_support
        
        logical_value = 0
        for idx in logical_support:
            if idx < len(raw_meas):
                logical_value ^= int(raw_meas[idx])
        
        return logical_value
    
    # =========================================================================
    # Phase 3: Correct Data Logicals
    # =========================================================================
    
    def _phase3_correct_data_logicals(
        self,
        measurements: np.ndarray,
        inner_corrections: Dict[int, Dict[int, InnerCorrection]],
        corrected_outer_syndrome: Dict[int, Dict[str, Dict[int, int]]] = None
    ) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, Tuple[int, int]]]:
        """
        Phase 3: Apply ALL inner corrections to data block final measurements,
        plus X stabilizer measurement backaction correction.
        
        For each data block:
        1. Get raw logical from final measurement
        2. Find ALL inner EC instances that touched this block
        3. XOR cumulative inner corrections
        4. **NEW**: XOR X stabilizer measurement backaction (for Z logical only)
        
        X Stabilizer Backaction:
        ========================
        The X stabilizer measurement protocol uses CNOT(ancilla→data) which
        applies X̄ to data blocks based on measurement outcome. When the ancilla
        is prepared in |+̄⟩ and measured, outcome=1 means X̄ was applied to all
        data blocks in the stabilizer support. This flips the Z logical value.
        
        We must correct for this by XORing the cumulative X stabilizer outcomes
        (over all rounds) for each block's Z logical readout.
        
        Returns:
            Tuple of (raw_data_logicals, corrected_data_logicals)
            - raw_data_logicals: Before inner corrections (for boundary detector)
            - corrected_data_logicals: After inner corrections (for final logical)
        """
        # MCM is required - this was validated in __init__
        if self._mcm is None:
            raise RuntimeError(
                "Phase 3: MCM (measurement-centric metadata) is required but missing. "
                "The experiment must provide mcm_metadata. "
                "Did the circuit generation complete successfully?"
            )
        
        return self._phase3_correct_data_logicals_mcm(
            measurements, inner_corrections, corrected_outer_syndrome
        )
    
    def _phase3_correct_data_logicals_mcm(
        self,
        measurements: np.ndarray,
        inner_corrections: Dict[int, Dict[int, InnerCorrection]],
        corrected_outer_syndrome: Dict[int, Dict[str, Dict[int, int]]] = None
    ) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, Tuple[int, int]]]:
        """
        Phase 3 using measurement-centric model.
        
        Uses block_to_ec_instances from MCM to find all EC instances
        that touched each data block.
        
        NOTE: With the CORRECT Steane-style protocol (CNOT data→ancilla for BOTH
        X and Z stabilizers), there is NO backaction. The data qubits are always
        the control qubits, so they are never modified by the CNOTs.
        
        Previously there was backaction code for the incorrect CNOT(ancilla→data)
        protocol. That has been removed.
        
        Returns:
            Tuple of (raw_data_logicals, corrected_data_logicals)
            - raw_data_logicals: Before inner corrections (for boundary detector)
            - corrected_data_logicals: After inner corrections (for final logical)
        """
        raw_logicals: Dict[int, Tuple[int, int]] = {}
        corrected_logicals: Dict[int, Tuple[int, int]] = {}
        
        # NO backaction with correct protocol (CNOT data→ancilla for both X and Z)
        # Data is always control qubit, never modified
        
        for block_id in range(self.n_data_blocks):
            # Get raw logicals from final measurement
            raw_x, raw_z = self._get_raw_data_logicals_mcm(measurements, block_id)
            raw_logicals[block_id] = (raw_x, raw_z)
            
            # Accumulate all inner corrections for this block
            total_x_corr = 0
            total_z_corr = 0
            
            # Get all EC instance IDs that touched this block
            ec_instance_ids = self._mcm.block_to_ec_instances.get(block_id, [])
            for ec_instance_id in ec_instance_ids:
                if ec_instance_id in inner_corrections:
                    if block_id in inner_corrections[ec_instance_id]:
                        total_x_corr ^= inner_corrections[ec_instance_id][block_id].x_correction
                        total_z_corr ^= inner_corrections[ec_instance_id][block_id].z_correction
            
            corrected_logicals[block_id] = (
                (raw_x + total_x_corr) % 2,
                (raw_z + total_z_corr) % 2  # No backaction with correct protocol
            )
            
            if self.config.verbose:
                print(f"  Block {block_id}: raw=({raw_x},{raw_z}), "
                      f"corr=({total_x_corr},{total_z_corr}), "
                      f"from {len(ec_instance_ids)} EC instances, "
                      f"final=({corrected_logicals[block_id][0]},{corrected_logicals[block_id][1]})")
        
        return raw_logicals, corrected_logicals
    
    def _compute_x_stabilizer_backaction(
        self,
        corrected_outer_syndrome: Dict[int, Dict[str, Dict[int, int]]] = None
    ) -> Dict[int, int]:
        """
        DEPRECATED: No backaction with correct Steane-style protocol.
        
        With CNOT(data→ancilla) for BOTH X and Z stabilizers, data qubits
        are always the control and never modified. This function is kept
        for backward compatibility but always returns zeros.
        """
        # Always return zeros - no backaction with correct protocol
        return {block_id: 0 for block_id in range(self.n_data_blocks)}
    
    def _get_raw_data_logicals_mcm(
        self,
        measurements: np.ndarray,
        block_id: int
    ) -> Tuple[int, int]:
        """Get raw X and Z logicals from final data measurement using MCM."""
        # Get FinalDataMeasurement for this block directly from dict
        final_meas = self._mcm.final_data_measurements.get(block_id)
        
        if final_meas is None or not final_meas.transversal_meas_indices:
            return (0, 0)
        
        meas_indices = final_meas.transversal_meas_indices
        raw_meas = np.array([measurements[i] for i in meas_indices if 0 <= i < len(measurements)], dtype=np.uint8)
        
        # Compute raw logicals
        raw_z = 0
        for idx in self._inner_z_logical_support:
            if idx < len(raw_meas):
                raw_z ^= int(raw_meas[idx])
        
        raw_x = 0
        for idx in self._inner_x_logical_support:
            if idx < len(raw_meas):
                raw_x ^= int(raw_meas[idx])
        
        return (raw_x, raw_z)
    
    # =========================================================================
    # Phase 4: Outer Temporal DEM Decode
    # =========================================================================
    
    def _phase4_outer_decode(
        self,
        corrected_outer_syndrome: Dict[int, Dict[str, Dict[int, int]]],
        raw_data_logicals: Dict[int, Tuple[int, int]],
        corrected_data_logicals: Dict[int, Tuple[int, int]]
    ) -> int:
        """
        Phase 4: Decode outer code using temporal DEM.
        
        1. Build detector bits from consecutive syndrome comparison
        2. Add final boundary detectors from RAW data logicals (not corrected!)
        3. Decode using PyMatching
        4. Apply correction to CORRECTED data logicals for final logical
        
        Key insight: The boundary detector compares last syndrome to data state.
        Inner corrections on data blocks would create artificial mismatches if
        we used corrected values. The boundary detector should use RAW values
        to detect errors that occurred AFTER the last syndrome measurement.
        Inner corrections are applied separately to the final logical.
        """
        if self.config.verbose:
            print(f"Phase 4: Outer decode")
            print(f"  Corrected outer syndrome: {corrected_outer_syndrome}")
            print(f"  Raw data logicals: {raw_data_logicals}")
            print(f"  Corrected data logicals: {corrected_data_logicals}")
        
        # Get sorted round indices
        round_indices = sorted(corrected_outer_syndrome.keys())
        n_rounds = len(round_indices)
        
        if n_rounds < 1:
            # No syndrome data, use corrected data logicals directly
            return self._compute_logical_from_data(corrected_data_logicals)
        
        # Use appropriate stabilizer type based on measurement basis
        stab_type = 'X' if self.basis == 'Z' else 'Z'
        n_stabs = self._n_outer_x_stabs if stab_type == 'X' else self._n_outer_z_stabs
        
        if n_stabs == 0:
            return self._compute_logical_from_data(corrected_data_logicals)
        
        # Build outer temporal DEM
        dem = self._build_outer_temporal_dem(n_rounds, n_stabs)
        
        if dem.num_detectors == 0:
            return self._compute_logical_from_data(corrected_data_logicals)
        
        # Extract detector bits using RAW data logicals for boundary detector
        # This prevents inner corrections from creating artificial detector firings
        detector_bits = self._extract_outer_detector_bits(
            corrected_outer_syndrome, raw_data_logicals,
            round_indices, stab_type, n_stabs
        )
        
        if self.config.verbose:
            print(f"  Outer detector bits: {detector_bits}")
        
        # Decode
        if not HAS_PYMATCHING:
            # Fallback: just use corrected data
            return self._compute_logical_from_data(corrected_data_logicals)
        
        try:
            matcher = pymatching.Matching.from_detector_error_model(dem)
            prediction = matcher.decode(detector_bits)
            
            # Apply outer correction to CORRECTED data logicals
            logical_value = self._compute_logical_from_data(corrected_data_logicals)
            if len(prediction) > 0 and prediction[0]:
                logical_value ^= 1
            
            return logical_value
        except Exception as e:
            if self.config.verbose:
                print(f"  Outer decode error: {e}")
            return self._compute_logical_from_data(corrected_data_logicals)
    
    def _build_outer_temporal_dem(
        self,
        n_rounds: int,
        n_stabs: int,
    ) -> stim.DetectorErrorModel:
        """
        Build temporal DEM for outer code with BLOCK-LEVEL correlated error model.
        
        Key insight: An inner Z error on block b triggers ALL X stabilizers that
        have support on block b. This creates correlated detector firings that
        the decoder must model correctly.
        
        For the [[7,1,3]] Steane code with Z_L = {0,1,2}:
        - Block 0 error: triggers X0,X1,X2 AND flips Z_L
        - Block 1 error: triggers X0,X1 AND flips Z_L
        - Block 2 error: triggers X0,X2 AND flips Z_L
        - Block 3 error: triggers X0 only (no Z_L flip)
        - Block 4 error: triggers X1,X2 (no Z_L flip)
        - Block 5 error: triggers X1 only (no Z_L flip)
        - Block 6 error: triggers X2 only (no Z_L flip)
        
        PLUS measurement errors for each stabilizer (paired consecutive detectors).
        
        Detectors: D[r,s] = OuterSyn[r+1,s] XOR OuterSyn[r,s] for r in [0,n_rounds-2]
                   D[n_comparison*n_stabs + s] = boundary detector for stabilizer s
        """
        # Number of comparison detectors (between rounds)
        n_comparison = max(0, n_rounds - 1)
        # Final boundary detectors (one per stabilizer)
        n_boundary = n_stabs
        n_detectors = n_comparison * n_stabs + n_boundary
        
        if n_detectors == 0:
            return stim.DetectorErrorModel()
        
        dem_lines = []
        
        # Get check matrix and logical support
        if self.basis == 'Z':
            logical_support = self._outer_z_logical_support
            check_matrix = self._outer_hx
        else:
            logical_support = self._outer_x_logical_support
            check_matrix = self._outer_hz
        
        # Model BLOCK-LEVEL errors with correlated detector triggers
        # A Z error on block b triggers all X stabilizers with support on b
        for block in range(self.n_data_blocks):
            # Find which stabilizers have support on this block
            triggered_stabs = []
            for s in range(n_stabs):
                if s < check_matrix.shape[0] and block < check_matrix.shape[1]:
                    if check_matrix[s][block]:
                        triggered_stabs.append(s)
            
            if not triggered_stabs:
                continue
            
            # Does this block flip the logical?
            flips_logical = block in logical_support
            logical_suffix = " L0" if flips_logical else ""
            
            # For each round transition, this block can have a DATA error
            for r in range(n_comparison):
                # Detector indices for all triggered stabilizers at this round
                det_strs = [f"D{r * n_stabs + s}" for s in triggered_stabs]
                dem_lines.append(f"error({self.config.error_prob}) {' '.join(det_strs)}{logical_suffix}")
            
            # Also for boundary (between last syndrome and data measurement)
            boundary_det_strs = [f"D{n_comparison * n_stabs + s}" for s in triggered_stabs]
            dem_lines.append(f"error({self.config.error_prob}) {' '.join(boundary_det_strs)}{logical_suffix}")
        
        # Model MEASUREMENT errors (per stabilizer, no correlated triggers)
        # A measurement error causes a transient syndrome flip: D[r,s] and D[r+1,s] both fire
        for s in range(n_stabs):
            # Detector indices for this stabilizer across rounds
            all_det_indices = [r * n_stabs + s for r in range(n_comparison)]
            all_det_indices.append(n_comparison * n_stabs + s)  # boundary
            
            # Measurement errors are paired consecutive detectors, NO logical flip
            for i in range(len(all_det_indices) - 1):
                det1 = f"D{all_det_indices[i]}"
                det2 = f"D{all_det_indices[i + 1]}"
                dem_lines.append(f"error({self.config.error_prob}) {det1} {det2}")
        
        dem_str = "\n".join(dem_lines)
        return stim.DetectorErrorModel(dem_str)
    
    def _extract_outer_detector_bits(
        self,
        corrected_outer_syndrome: Dict[int, Dict[str, Dict[int, int]]],
        corrected_data_logicals: Dict[int, Tuple[int, int]],
        round_indices: List[int],
        stab_type: str,
        n_stabs: int,
    ) -> np.ndarray:
        """Extract detector bits for outer temporal DEM."""
        n_rounds = len(round_indices)
        n_comparison = max(0, n_rounds - 1)
        n_boundary = n_stabs
        n_detectors = n_comparison * n_stabs + n_boundary
        
        detector_bits = np.zeros(n_detectors, dtype=np.uint8)
        
        # Comparison detectors
        for r in range(n_comparison):
            round_a = round_indices[r]
            round_b = round_indices[r + 1]
            
            syn_a = corrected_outer_syndrome.get(round_a, {}).get(stab_type, {})
            syn_b = corrected_outer_syndrome.get(round_b, {}).get(stab_type, {})
            
            for s in range(n_stabs):
                val_a = syn_a.get(s, 0)
                val_b = syn_b.get(s, 0)
                detector_bits[r * n_stabs + s] = (val_a ^ val_b)
        
        # Boundary detectors: compare last syndrome to data logical parity
        if n_rounds > 0:
            last_round = round_indices[-1]
            last_syn = corrected_outer_syndrome.get(last_round, {}).get(stab_type, {})
            
            # Get check matrix for syndrome computation from data
            if stab_type == 'X':
                check_matrix = self._outer_hx
            else:
                check_matrix = self._outer_hz
            
            # Compute expected syndrome from corrected data logicals
            data_logical_array = np.zeros(self.n_data_blocks, dtype=np.uint8)
            for block_id in range(self.n_data_blocks):
                if block_id in corrected_data_logicals:
                    x_val, z_val = corrected_data_logicals[block_id]
                    # Use Z logical for X syndrome, X logical for Z syndrome
                    if stab_type == 'X':
                        data_logical_array[block_id] = z_val
                    else:
                        data_logical_array[block_id] = x_val
            
            for s in range(n_stabs):
                syn_from_data = 0
                if s < check_matrix.shape[0]:
                    row = check_matrix[s]
                    for q in range(min(len(row), self.n_data_blocks)):
                        if row[q]:
                            syn_from_data ^= int(data_logical_array[q])
                
                last_syn_val = last_syn.get(s, 0)
                boundary_det_idx = n_comparison * n_stabs + s
                detector_bits[boundary_det_idx] = (last_syn_val ^ syn_from_data)
        
        return detector_bits
    
    def _compute_logical_from_data(
        self,
        corrected_data_logicals: Dict[int, Tuple[int, int]]
    ) -> int:
        """Compute final logical from corrected data logicals."""
        support = self._outer_z_logical_support if self.basis == "Z" else self._outer_x_logical_support
        
        result = 0
        for block_id in support:
            if block_id in corrected_data_logicals:
                x_val, z_val = corrected_data_logicals[block_id]
                if self.basis == "Z":
                    result ^= z_val
                else:
                    result ^= x_val
        
        return result


# =============================================================================
# Temporal DEM Decoder
# =============================================================================

class TemporalHierarchicalV6Decoder(BaseHierarchicalV6Decoder):
    """
    Hierarchical V6 decoder with temporal DEM for inner decoding.
    
    Inner decoding uses consecutive round comparison:
        D[r,s] = S[r+1,s] XOR S[r,s]
    
    Outer decoding always uses temporal DEM (from base class).
    
    For concatenated codes, set skip_inner_corrections=True to disable
    inner corrections. This is recommended when the inner code's error
    correction is unreliable due to high error density per block.
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        basis: str = "Z",
        verbose: bool = False,
        error_prob: float = 0.01,
        skip_inner_corrections: bool = False,
    ):
        config = DecoderConfig(
            inner_mode=InnerDecodingMode.TEMPORAL_DEM,
            error_prob=error_prob,
            verbose=verbose,
            skip_inner_corrections=skip_inner_corrections,
        )
        super().__init__(metadata, config, basis)
    
    def _compute_inner_correction(
        self,
        block_id: int,
        x_anc_meas: np.ndarray,
        z_anc_meas: np.ndarray,
        ec_instance: InnerECInstance,
    ) -> Tuple[int, int]:
        """
        Compute inner corrections using temporal DEM.
        
        Returns (x_correction, z_correction) where:
        - x_correction: Flips X logical (corrects X errors detected by Z stabilizers)
        - z_correction: Flips Z logical (corrects Z errors detected by X stabilizers)
        """
        # X ancillas measure X stabilizers which detect Z errors
        z_error_correction = self._temporal_correction(
            x_anc_meas, self._n_x_stabs, 'X', block_id
        )
        # Z ancillas measure Z stabilizers which detect X errors
        x_error_correction = self._temporal_correction(
            z_anc_meas, self._n_z_stabs, 'Z', block_id
        )
        
        # Return (x_correction, z_correction)
        # x_correction = correction for X errors (from Z stabilizer decode)
        # z_correction = correction for Z errors (from X stabilizer decode)
        return (x_error_correction, z_error_correction)
    
    def _temporal_correction(
        self,
        measurements: np.ndarray,
        n_stabs: int,
        stab_type: str,
        block_id: int,
    ) -> int:
        """
        Compute correction using temporal DEM for one stabilizer type.
        
        Uses COMPARISON DETECTORS ONLY (no boundary detectors).
        
        Detector pattern:
        - D[r,s] = S[r+1,s] XOR S[r,s] for r in [0, n_rounds-2]
        
        Error model:
        - Single detector = data error between rounds → may flip L0
        - Adjacent pair = measurement error → no L0 flip
        
        NOTE: Boundary detectors are NOT used because:
        - D_boundary = S[final] XOR S[initial] = XOR of all comparison detectors
        - This is mathematically redundant and causes spurious pairings in MWPM
        - Without boundary, we can't distinguish first/last round errors, but this
          is acceptable when n_rounds >> 1 (boundary errors are rare)
        """
        if len(measurements) < n_stabs * 2:
            return 0
        
        n_rounds = len(measurements) // n_stabs
        if n_rounds < 2:
            return 0
        
        # Build temporal DEM (comparison detectors only)
        dem = self._build_inner_temporal_dem(n_rounds, n_stabs, stab_type)
        
        if dem.num_detectors == 0:
            return 0
        
        # Comparison detectors only: n_comparison * n_stabs
        n_comparison = n_rounds - 1
        n_detectors = n_comparison * n_stabs
        
        detector_bits = np.zeros(n_detectors, dtype=np.uint8)
        
        # Reshape measurements to (n_rounds, n_stabs)
        meas_by_round = measurements[:n_rounds * n_stabs].reshape(n_rounds, n_stabs)
        
        # Comparison detectors: D[r*n_stabs + s] = S[r+1,s] XOR S[r,s]
        for r in range(n_rounds - 1):
            for s in range(n_stabs):
                detector_bits[r * n_stabs + s] = int(meas_by_round[r+1, s]) ^ int(meas_by_round[r, s])
        
        # Decode
        if not HAS_PYMATCHING:
            return int(np.any(detector_bits))
        
        try:
            matcher = pymatching.Matching.from_detector_error_model(dem)
            prediction = matcher.decode(detector_bits)
            return int(prediction[0]) if len(prediction) > 0 else 0
        except Exception:
            return int(np.any(detector_bits))
    
    def _build_inner_temporal_dem(
        self,
        n_rounds: int,
        n_stabs: int,
        stab_type: str,
    ) -> stim.DetectorErrorModel:
        """
        Build CONSERVATIVE temporal DEM for inner code.
        
        Key insight: Single-detector patterns are AMBIGUOUS:
        - Could be data error (would flip logical)
        - Could be measurement error at boundary round (no logical flip)
        
        The inner decoder should NOT flip L0 for any single-detector pattern
        because we cannot distinguish data errors from measurement errors.
        
        Only paired detector patterns (D[r,s] and D[r+1,s]) are unambiguous:
        - These are caused by measurement errors at middle rounds
        - They never flip the logical
        
        The outer decoder will use the CORRECTED data logical measurements
        to make the final decision. By being conservative at the inner level,
        we avoid over-correction.
        
        Detector layout:
        - D[r*n_stabs + s] = S[r+1,s] XOR S[r,s] for r in 0..n_rounds-2
        """
        n_comparison = n_rounds - 1
        n_total_detectors = n_comparison * n_stabs
        
        if n_total_detectors == 0:
            return stim.DetectorErrorModel()
        
        dem_lines = []
        
        # All single-detector terms: NO L0 flip (conservative approach)
        # This is because single-detector patterns are ambiguous:
        # - Data error at round r → D[r] only
        # - Measurement error at round 0 → D[0] only  
        # - Measurement error at final round → D[n-1] only
        for s in range(n_stabs):
            for r in range(n_comparison):
                det = f"D{r * n_stabs + s}"
                dem_lines.append(f"error({self.config.error_prob}) {det}")
            
            # MEASUREMENT errors at middle rounds: paired detectors → no L0 flip
            # These are unambiguous and should be used for syndrome correction
            for r in range(n_comparison - 1):
                det1 = f"D{r * n_stabs + s}"
                det2 = f"D{(r + 1) * n_stabs + s}"
                dem_lines.append(f"error({self.config.error_prob}) {det1} {det2}")
        
        dem_str = "\n".join(dem_lines)
        return stim.DetectorErrorModel(dem_str)


# =============================================================================
# Majority Vote Decoder  
# =============================================================================

class MajorityVoteHierarchicalV6Decoder(BaseHierarchicalV6Decoder):
    """
    Hierarchical V6 decoder with majority vote for inner decoding.
    
    Inner decoding uses consensus from d rounds:
    - First round is baseline
    - Majority vote determines if error occurred
    - Decode consensus syndrome for correction
    
    Outer decoding always uses temporal DEM (from base class).
    """
    
    def __init__(
        self,
        metadata: Dict[str, Any],
        basis: str = "Z",
        verbose: bool = False,
        error_prob: float = 0.01,
    ):
        config = DecoderConfig(
            inner_mode=InnerDecodingMode.MAJORITY_VOTE,
            error_prob=error_prob,
            verbose=verbose,
        )
        super().__init__(metadata, config, basis)
    
    def _compute_inner_correction(
        self,
        block_id: int,
        x_anc_meas: np.ndarray,
        z_anc_meas: np.ndarray,
        ec_instance: InnerECInstance,
    ) -> Tuple[int, int]:
        """
        Compute inner corrections using majority vote.
        
        Returns (x_correction, z_correction) where:
        - x_correction: Flips X logical (corrects X errors detected by Z stabilizers)
        - z_correction: Flips Z logical (corrects Z errors detected by X stabilizers)
        """
        # X ancillas measure X stabilizers which detect Z errors
        z_error_correction = self._majority_vote_correction(
            x_anc_meas, self._n_x_stabs, 'X'
        )
        # Z ancillas measure Z stabilizers which detect X errors
        x_error_correction = self._majority_vote_correction(
            z_anc_meas, self._n_z_stabs, 'Z'
        )
        
        # Return (x_correction, z_correction)
        # x_correction = correction for X errors (from Z stabilizer decode)
        # z_correction = correction for Z errors (from X stabilizer decode)
        return (x_error_correction, z_error_correction)
    
    def _majority_vote_correction(
        self,
        measurements: np.ndarray,
        n_stabs: int,
        stab_type: str,
    ) -> int:
        """
        Compute correction using majority vote on CHANGES from first round.
        
        Key insight: The first round syndrome may be non-zero due to:
        - Random X stabilizer outcomes when measuring Z-basis state
        - Non-trivial expected syndrome in concatenated codes
        
        We vote on whether each stabilizer CHANGED from the first round,
        not on its absolute value. This correctly handles:
        - Constant non-zero syndrome → no change → no correction
        - Persistent change → data error → correction needed
        - Transient change → measurement error → no correction (by vote)
        
        Per AGP (quant-ph/0504218) adapted for non-zero expected syndromes.
        """
        if len(measurements) < n_stabs:
            return 0
        
        n_rounds = len(measurements) // n_stabs
        if n_rounds < 2:
            # Need at least 2 rounds to detect changes
            return 0
        
        # Get first round as reference
        reference = np.array([measurements[s] for s in range(n_stabs)], dtype=np.uint8)
        
        # Vote on changes from reference for each stabilizer
        # change_votes[s] = number of rounds (after first) where syndrome[s] differs from reference
        consensus_change = np.zeros(n_stabs, dtype=np.uint8)
        threshold = (n_rounds - 1) / 2.0  # Threshold for remaining rounds
        
        for s in range(n_stabs):
            # Count how many rounds (after first) show a change
            change_count = 0
            for r in range(1, n_rounds):
                idx = r * n_stabs + s
                if idx < len(measurements):
                    if int(measurements[idx]) != int(reference[s]):
                        change_count += 1
            
            # Consensus: 1 if majority of rounds show change from reference
            consensus_change[s] = 1 if change_count > threshold else 0
        
        # If no persistent changes detected, no correction needed
        if not np.any(consensus_change):
            return 0
        
        # Decode the consensus CHANGE pattern
        # This tells us which stabilizer was persistently flipped
        return self._decode_syndrome(consensus_change, stab_type)
    
    def _decode_syndrome(self, syndrome: np.ndarray, stab_type: str) -> int:
        """Decode syndrome to get logical correction."""
        if stab_type == 'X':
            matcher = self._inner_x_matcher
            logical_support = self._inner_z_logical_support
        else:
            matcher = self._inner_z_matcher
            logical_support = self._inner_x_logical_support
        
        if matcher is not None and HAS_PYMATCHING:
            try:
                correction = matcher.decode(syndrome)
                # Check if correction overlaps logical
                result = 0
                for idx in logical_support:
                    if idx < len(correction):
                        result ^= int(correction[idx])
                return result
            except Exception:
                pass
        
        # Fallback: any syndrome deviation implies correction
        return int(np.any(syndrome))


# =============================================================================
# Convenience Functions
# =============================================================================

def create_temporal_v6_decoder(metadata: Dict[str, Any], **kwargs) -> TemporalHierarchicalV6Decoder:
    """Create a temporal DEM-based hierarchical V6 decoder."""
    return TemporalHierarchicalV6Decoder(metadata, **kwargs)


def create_majority_vote_v6_decoder(metadata: Dict[str, Any], **kwargs) -> MajorityVoteHierarchicalV6Decoder:
    """Create a majority vote hierarchical V6 decoder."""
    return MajorityVoteHierarchicalV6Decoder(metadata, **kwargs)
