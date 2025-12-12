# src/qectostim/decoders/concatenated_decoder.py
"""
Decoders for concatenated quantum error-correcting codes.

This module provides two decoder implementations:

1. FlatConcatenatedDecoder: Simple decoder that uses PyMatching on the full 
   concatenated DEM. This is the recommended decoder for most use cases as
   it's simpler and achieves good logical error rates.

2. ConcatenatedDecoder: Hierarchical decoder with cross-correlation handling.
   This implements proper hierarchical decoding:
   - Extracts per-block sub-DEMs that PRESERVE correlations with outer detectors
     by mapping outer correlations to virtual observables
   - Decodes each inner block to get both logical corrections AND outer-syndrome
     contributions (via virtual observable predictions)
   - Constructs effective outer syndrome = raw_outer XOR (H_outer * inner_logicals)
   - Decodes outer code with the effective syndrome
   - Final correction = outer decoder output (after syndrome mapping)

The key insight is that inner logical errors are "physical errors on the outer code",
so they must modify the outer syndrome before decoding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, TYPE_CHECKING

import numpy as np
import stim

from qectostim.codes.composite.concatenated import ConcatenatedCode
from qectostim.decoders.base import Decoder

if TYPE_CHECKING:
    pass


class ConcatenatedDecoderIncompatibleError(Exception):
    """Raised when a concatenated decoder is used with a non-concatenated code."""
    pass


@dataclass
class FlatConcatenatedDecoder(Decoder):
    """Simple decoder for concatenated codes using PyMatching on the full DEM.
    
    This decoder treats the concatenated code as a single monolithic code and
    applies PyMatching directly to the full detector error model. While this
    doesn't exploit the hierarchical structure of concatenated codes, it:
    
    - Is simple and robust
    - Achieves good logical error rates (comparable to PyMatchingDecoder)
    - Works correctly for any concatenation structure
    - Provides a reliable baseline for comparing hierarchical decoders
    
    For large concatenated codes where hierarchical decoding might be faster,
    consider ConcatenatedDecoder instead.
    
    Parameters
    ----------
    code : ConcatenatedCode
        The concatenated code instance.
    dem : stim.DetectorErrorModel
        The detector error model from the circuit.
    rounds : int
        Number of syndrome measurement rounds (stored for metadata).
    basis : str
        Measurement basis ('X' or 'Z', stored for metadata).
    """
    
    code: ConcatenatedCode
    dem: stim.DetectorErrorModel
    rounds: int = 3
    basis: str = "Z"
    
    # Internal state
    _matching: Any = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize PyMatching decoder on full DEM."""
        import pymatching
        
        # Validate concatenated code (but we'll accept any code)
        if not isinstance(self.code, ConcatenatedCode):
            # Check if it has concatenation metadata
            meta = getattr(self.code, "metadata", getattr(self.code, "_metadata", {}))
            if "concatenation" not in meta and not hasattr(self.code, '_outer_code'):
                raise ConcatenatedDecoderIncompatibleError(
                    f"FlatConcatenatedDecoder is designed for ConcatenatedCode or codes "
                    f"with concatenation structure. Got {type(self.code).__name__}. "
                    f"Use PyMatchingDecoder for non-concatenated codes."
                )
        
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables
        
        # Build PyMatching decoder on full DEM
        self._matching = pymatching.Matching.from_detector_error_model(self.dem)
    
    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """Decode a batch of detector samples using PyMatching on full DEM.
        
        Parameters
        ----------
        dets : np.ndarray
            Shape (shots, num_detectors) array of detector outcomes.
            
        Returns
        -------
        np.ndarray
            Shape (shots, num_observables) array of logical corrections.
        """
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        
        if dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"FlatConcatenatedDecoder: expected {self.num_detectors} detectors, "
                f"got {dets.shape[1]}."
            )
        
        corrections = self._matching.decode_batch(dets)
        return np.asarray(corrections, dtype=np.uint8).reshape(-1, self.num_observables)
    
    def decode(self, dets: np.ndarray) -> np.ndarray:
        """Decode a single shot of detector samples.
        
        Parameters
        ----------
        dets : np.ndarray
            Shape (num_detectors,) array of detector outcomes.
            
        Returns
        -------
        np.ndarray
            Shape (num_observables,) array of logical corrections.
        """
        return self.decode_batch(dets.reshape(1, -1)).flatten()


def _extract_block_dem_with_outer_correlations(
    global_dem: stim.DetectorErrorModel,
    det_start: int,
    det_stop: int,
    outer_start: int,
    outer_stop: int,
) -> Tuple[stim.DetectorErrorModel, Dict[int, int]]:
    """
    Extract a sub-DEM for a block that PRESERVES correlations with outer detectors.
    
    For edges that touch both inner and outer detectors, we:
    1. Keep the inner detector parts (remapped to local indices)
    2. Map outer detector correlations to virtual observables (obs 1, 2, ...)
    3. Preserve the actual logical observable (obs 0)
    
    This allows the inner decoder to predict which outer detectors would be
    affected by the same error, enabling proper syndrome mapping.
    
    Parameters
    ----------
    global_dem : stim.DetectorErrorModel
        The full DEM from the concatenated circuit.
    det_start : int
        Start of this block's detector range (inclusive).
    det_stop : int
        End of this block's detector range (exclusive).
    outer_start : int
        Start of outer detector range.
    outer_stop : int
        End of outer detector range.
        
    Returns
    -------
    Tuple[stim.DetectorErrorModel, Dict[int, int]]
        (block_dem, outer_correlation_map) where outer_correlation_map[outer_det_idx] 
        maps to virtual observable index (1, 2, ...).
    """
    block_dem = stim.DetectorErrorModel()
    n_inner_dets = det_stop - det_start
    
    # Map outer detector global index -> virtual observable index (starting at 1)
    outer_to_virtual: Dict[int, int] = {}
    next_virtual = 1  # Observable 0 is reserved for actual logical
    
    for instruction in global_dem.flattened():
        if instruction.type == "error":
            prob = instruction.args_copy()[0]
            targets = instruction.targets_copy()
            
            inner_dets = []
            outer_dets_affected = []  # Global indices of outer detectors
            logical_targets = []
            other_block_dets = False
            
            for t in targets:
                if t.is_relative_detector_id():
                    det_id = t.val
                    if det_start <= det_id < det_stop:
                        # In this block - remap to local
                        local_id = det_id - det_start
                        inner_dets.append(stim.target_relative_detector_id(local_id))
                    elif outer_start <= det_id < outer_stop:
                        # Outer detector - record for virtual observable
                        outer_dets_affected.append(det_id)
                    else:
                        # Another inner block - skip this edge for this block's decoder
                        other_block_dets = True
                elif t.is_logical_observable_id():
                    logical_targets.append(t)
            
            # Include edge if it affects this block (and doesn't span other blocks)
            if inner_dets and not other_block_dets:
                # Build virtual observable targets for outer correlations
                virtual_targets = []
                for outer_det in outer_dets_affected:
                    if outer_det not in outer_to_virtual:
                        outer_to_virtual[outer_det] = next_virtual
                        next_virtual += 1
                    virtual_targets.append(
                        stim.target_logical_observable_id(outer_to_virtual[outer_det])
                    )
                
                all_targets = inner_dets + logical_targets + virtual_targets
                block_dem.append("error", prob, all_targets)
    
    return block_dem, outer_to_virtual


def _build_outer_dem_from_code(
    outer_code: Any,
    basis: str,
    n_blocks: int,
) -> Tuple[stim.DetectorErrorModel, np.ndarray]:
    """
    Build a simple outer DEM directly from the outer code's parity check matrix.
    
    This creates a phenomenological DEM where each "qubit" (inner block) can have
    an independent error that triggers the appropriate stabilizers.
    
    Parameters
    ----------
    outer_code : CSSCode
        The outer code in the concatenation.
    basis : str
        'X' or 'Z' for which stabilizers to use.
    n_blocks : int
        Number of inner blocks (= outer code n).
        
    Returns
    -------
    Tuple[stim.DetectorErrorModel, np.ndarray]
        (outer_dem, parity_check_matrix)
    """
    # Get appropriate parity check matrix
    if basis.upper() == "Z":
        H = np.array(outer_code.hz, dtype=np.uint8)
    else:
        H = np.array(outer_code.hx, dtype=np.uint8)
    
    n_stabs, n_qubits = H.shape
    
    # Create DEM with one error per "qubit" (inner block)
    outer_dem = stim.DetectorErrorModel()
    p_err = 0.01  # Nominal error rate (actual probabilities come from inner decoders)
    
    for q in range(min(n_qubits, n_blocks)):
        # Find which stabilizers this qubit participates in
        affected_stabs = np.where(H[:, q] == 1)[0]
        
        if len(affected_stabs) > 0:
            targets = [stim.target_relative_detector_id(s) for s in affected_stabs]
            # Add logical observable if this qubit is in the logical operator
            # For now, assume all qubits contribute to logical
            targets.append(stim.target_logical_observable_id(0))
            outer_dem.append("error", p_err, targets)
    
    return outer_dem, H


def _extract_pure_outer_dem(
    global_dem: stim.DetectorErrorModel,
    outer_start: int,
    outer_stop: int,
) -> stim.DetectorErrorModel:
    """
    Extract edges that involve ONLY outer detectors.
    
    Parameters
    ----------
    global_dem : stim.DetectorErrorModel
        The full DEM.
    outer_start : int
        Start of outer detector range.
    outer_stop : int
        End of outer detector range.
        
    Returns
    -------
    stim.DetectorErrorModel
        DEM with only pure-outer edges, remapped to [0, n_outer_dets).
    """
    outer_dem = stim.DetectorErrorModel()
    
    for instruction in global_dem.flattened():
        if instruction.type == "error":
            prob = instruction.args_copy()[0]
            targets = instruction.targets_copy()
            
            outer_dets = []
            has_non_outer = False
            logical_targets = []
            
            for t in targets:
                if t.is_relative_detector_id():
                    det_id = t.val
                    if outer_start <= det_id < outer_stop:
                        local_id = det_id - outer_start
                        outer_dets.append(stim.target_relative_detector_id(local_id))
                    else:
                        has_non_outer = True
                        break
                elif t.is_logical_observable_id():
                    logical_targets.append(t)
            
            # Only include pure-outer edges
            if outer_dets and not has_non_outer:
                all_targets = outer_dets + logical_targets
                outer_dem.append("error", prob, all_targets)
    
    return outer_dem


@dataclass
class ConcatenatedDecoder(Decoder):
    """Hierarchical decoder for concatenated codes with proper cross-correlation handling.

    This decoder implements proper hierarchical decoding:
    1. Extract per-block sub-DEMs that PRESERVE correlations with outer detectors
       using virtual observables to track which outer syndrome bits are affected
    2. Decode each inner block to get logical corrections AND outer-syndrome predictions
    3. Construct effective outer syndrome = raw_outer XOR (H_outer @ inner_logicals)
    4. Decode outer code with effective syndrome
    5. Final correction is the outer decoder's output
    
    The key insight is that an inner logical error is equivalent to a "physical error"
    on the outer code, which flips specific outer syndrome bits according to H_outer.
    
    Parameters
    ----------
    code : ConcatenatedCode
        The concatenated code instance.
    dem : stim.DetectorErrorModel
        The detector error model from the circuit.
    rounds : int
        Number of syndrome measurement rounds.
    basis : str
        Measurement basis ('X' or 'Z').
    """

    code: ConcatenatedCode
    dem: stim.DetectorErrorModel
    rounds: int = 3
    basis: str = "Z"
    
    # Legacy parameters for compatibility
    preferred: Optional[str] = None
    max_bp_iters: int = 30
    osd_order: int = 60
    tesseract_bond_dim: int = 5
    
    # Internal state (set in __post_init__)
    _block_decoders: Dict[int, Any] = field(default_factory=dict, repr=False)
    _block_outer_maps: Dict[int, Dict[int, int]] = field(default_factory=dict, repr=False)
    _outer_decoder: Any = field(default=None, repr=False)
    _outer_parity_check: Optional[np.ndarray] = field(default=None, repr=False)
    _inner_slices: Dict[int, Tuple[int, int]] = field(default_factory=dict, repr=False)
    _outer_slices: Dict[int, Tuple[int, int]] = field(default_factory=dict, repr=False)
    _n_blocks: int = field(default=0, repr=False)
    _outer_n_dets: int = field(default=0, repr=False)
    _use_fallback: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize hierarchical decoder structure with cross-correlation handling."""
        import pymatching
        
        # Validate concatenated code
        if not isinstance(self.code, ConcatenatedCode):
            meta = getattr(self.code, "metadata", getattr(self.code, "_metadata", {}))
            if "concatenation" not in meta and not hasattr(self.code, 'build_concatenation_decoder_metadata'):
                raise ConcatenatedDecoderIncompatibleError(
                    f"ConcatenatedDecoder requires a ConcatenatedCode or a code with "
                    f"concatenation metadata. Got {type(self.code).__name__}."
                )
        
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables
        
        # Get concatenation metadata
        concat_meta = self._get_concat_metadata()
        
        if concat_meta is None:
            self._use_fallback = True
            self._matching = pymatching.Matching.from_detector_error_model(self.dem)
            return
        
        # Extract slice information
        self._inner_slices = concat_meta.get('dem_slices', [{}])[0] if concat_meta.get('dem_slices') else {}
        self._outer_slices = concat_meta.get('dem_slices', [{}, {}])[1] if len(concat_meta.get('dem_slices', [])) > 1 else {}
        self._n_blocks = concat_meta.get('n_inner_blocks', len(self._inner_slices))
        
        if not self._inner_slices or not self._outer_slices:
            self._use_fallback = True
            self._matching = pymatching.Matching.from_detector_error_model(self.dem)
            return
        
        # Get outer detector range
        outer_start, outer_stop = self._outer_slices[0]
        self._outer_n_dets = outer_stop - outer_start
        
        # Store outer parity check matrix for syndrome mapping
        self._store_outer_parity_check()
        
        # Build per-block decoders WITH cross-correlation tracking
        self._build_block_decoders_with_correlations(pymatching, outer_start, outer_stop)
        
        # Build outer decoder from pure-outer edges
        self._build_outer_decoder(pymatching, outer_start, outer_stop)
        
        # For compatibility
        self.levels = 2
        self._expected_internal_detectors = self.num_detectors
    
    def _get_concat_metadata(self) -> Optional[Dict[str, Any]]:
        """Get concatenation metadata from the code."""
        meta = getattr(self.code, "_metadata", {})
        if "concatenation" in meta:
            return meta["concatenation"]
        
        if hasattr(self.code, 'build_concatenation_decoder_metadata'):
            try:
                return self.code.build_concatenation_decoder_metadata(
                    rounds=self.rounds,
                    basis=self.basis
                )
            except Exception:
                pass
        
        return None
    
    def _store_outer_parity_check(self) -> None:
        """Store outer code's parity check matrix for syndrome mapping."""
        if hasattr(self.code, '_outer_code'):
            outer_code = self.code._outer_code
            if self.basis.upper() == "Z":
                if hasattr(outer_code, 'hz') and outer_code.hz is not None:
                    self._outer_parity_check = np.array(outer_code.hz, dtype=np.uint8)
            else:
                if hasattr(outer_code, 'hx') and outer_code.hx is not None:
                    self._outer_parity_check = np.array(outer_code.hx, dtype=np.uint8)
    
    def _build_block_decoders_with_correlations(
        self, pymatching, outer_start: int, outer_stop: int
    ) -> None:
        """Build per-block decoders that track correlations with outer detectors."""
        self._block_decoders = {}
        self._block_outer_maps = {}
        
        for block_id, (start, stop) in self._inner_slices.items():
            # Extract DEM with outer correlations mapped to virtual observables
            block_dem, outer_map = _extract_block_dem_with_outer_correlations(
                self.dem, start, stop, outer_start, outer_stop
            )
            
            self._block_outer_maps[block_id] = outer_map
            
            if block_dem.num_errors > 0:
                try:
                    self._block_decoders[block_id] = pymatching.Matching.from_detector_error_model(block_dem)
                except Exception:
                    self._block_decoders[block_id] = None
            else:
                self._block_decoders[block_id] = None
    
    def _build_outer_decoder(self, pymatching, outer_start: int, outer_stop: int) -> None:
        """Build outer decoder from pure-outer edges."""
        outer_dem = _extract_pure_outer_dem(self.dem, outer_start, outer_stop)
        
        if outer_dem.num_errors > 0:
            try:
                self._outer_decoder = pymatching.Matching.from_detector_error_model(outer_dem)
            except Exception:
                self._outer_decoder = None
        else:
            self._outer_decoder = None

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """Decode using hierarchical approach with proper inner-outer coupling.

        The decoding proceeds as:
        1. Decode each inner block to get inner logical predictions
        2. Compute effective outer syndrome by XORing raw outer syndrome with
           the syndrome contribution from inner logical errors
        3. Decode outer code with effective syndrome
        4. Return outer decoder's logical output as final correction

        Parameters
        ----------
        dets : np.ndarray
            Shape (shots, num_detectors) array of detector outcomes.

        Returns
        -------
        np.ndarray
            Shape (shots, num_observables) array of logical corrections.
        """
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        if dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"ConcatenatedDecoder: expected {self.num_detectors} detectors, "
                f"got {dets.shape[1]}."
            )
        
        # Fallback mode
        if self._use_fallback:
            corrections = self._matching.decode_batch(dets)
            return np.asarray(corrections, dtype=np.uint8).reshape(-1, self.num_observables)
        
        n_shots = dets.shape[0]
        
        # =================================================================
        # Step 1: Decode each inner block
        # =================================================================
        # inner_logicals[block_id] = (n_shots,) array of inner logical corrections
        inner_logicals = np.zeros((n_shots, self._n_blocks), dtype=np.uint8)
        
        # outer_syndrome_contrib tracks syndrome bits flipped by inner-outer correlations
        outer_syndrome_contrib = np.zeros((n_shots, self._outer_n_dets), dtype=np.uint8)
        
        for block_id in range(self._n_blocks):
            if block_id not in self._inner_slices:
                continue
                
            start, stop = self._inner_slices[block_id]
            block_dets = dets[:, start:stop]
            
            decoder = self._block_decoders.get(block_id)
            outer_map = self._block_outer_maps.get(block_id, {})
            
            if decoder is not None:
                try:
                    # Decode block - may return multiple observables (logical + virtuals)
                    block_result = decoder.decode_batch(block_dets)
                    block_result = np.atleast_2d(block_result)
                    
                    # Observable 0 is the inner logical
                    if block_result.shape[1] > 0:
                        inner_logicals[:, block_id] = block_result[:, 0]
                    
                    # Observables 1+ are virtual outer correlations
                    # When virtual observable i fires, it means the correlated
                    # outer detector should be flipped in the effective syndrome
                    for outer_det_global, virtual_obs in outer_map.items():
                        if virtual_obs < block_result.shape[1]:
                            outer_det_local = outer_det_global - self._outer_slices[0][0]
                            if 0 <= outer_det_local < self._outer_n_dets:
                                outer_syndrome_contrib[:, outer_det_local] ^= block_result[:, virtual_obs]
                                
                except Exception:
                    pass  # Block decoding failed - assume no errors
        
        # =================================================================
        # Step 2: Construct effective outer syndrome
        # =================================================================
        outer_start, outer_stop = self._outer_slices[0]
        raw_outer_syndrome = dets[:, outer_start:outer_stop]
        
        # Effective syndrome = raw XOR contribution from inner-outer correlations
        effective_outer_syndrome = (raw_outer_syndrome ^ outer_syndrome_contrib).astype(np.uint8)
        
        # Additionally, inner logical errors act as "physical errors" on outer code
        # This flips outer syndrome according to outer parity check matrix
        if self._outer_parity_check is not None:
            H = self._outer_parity_check
            n_stabs, n_qubits = H.shape
            n_blocks = min(n_qubits, self._n_blocks)
            
            # inner_syndrome_contrib[shot, stab] = XOR over blocks of (inner_logical[block] AND H[stab, block])
            inner_contrib = (inner_logicals[:, :n_blocks] @ H[:, :n_blocks].T) % 2
            inner_contrib = inner_contrib.astype(np.uint8)
            
            # Apply to effective syndrome (size may differ from n_stabs)
            contrib_size = min(inner_contrib.shape[1], effective_outer_syndrome.shape[1])
            effective_outer_syndrome[:, :contrib_size] ^= inner_contrib[:, :contrib_size]
        
        # =================================================================
        # Step 3: Decode outer code with effective syndrome
        # =================================================================
        if self._outer_decoder is not None:
            try:
                outer_result = self._outer_decoder.decode_batch(effective_outer_syndrome)
                outer_result = np.atleast_2d(outer_result)
                if outer_result.shape[1] > 0:
                    outer_logical = outer_result[:, 0]
                else:
                    outer_logical = np.zeros(n_shots, dtype=np.uint8)
            except Exception:
                outer_logical = np.zeros(n_shots, dtype=np.uint8)
        else:
            # No outer decoder - use majority vote on inner logicals as fallback
            inner_sum = np.sum(inner_logicals, axis=1)
            outer_logical = (inner_sum > self._n_blocks // 2).astype(np.uint8)
        
        # =================================================================
        # Step 4: Return final correction
        # =================================================================
        # The outer decoder output after syndrome mapping IS the final answer
        final_correction = np.zeros((n_shots, self.num_observables), dtype=np.uint8)
        if self.num_observables > 0:
            final_correction[:, 0] = outer_logical
        
        return final_correction

    def decode(self, dets: np.ndarray) -> np.ndarray:
        """Decode a single shot of detector samples.
        
        Parameters
        ----------
        dets : np.ndarray
            Shape (num_detectors,) array of detector outcomes.
            
        Returns
        -------
        np.ndarray
            Shape (num_observables,) array of logical corrections.
        """
        return self.decode_batch(dets.reshape(1, -1)).flatten()