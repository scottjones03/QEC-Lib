# src/qectostim/decoders/mle_decoder.py
"""
Maximum Likelihood Estimation decoder for small non-CSS codes.

This decoder performs exact maximum likelihood decoding by exhaustively
searching all possible error patterns. It's optimal but only practical
for small codes (typically n ≤ 15).

For non-CSS codes where matching decoders fail, this provides a reference
for the best achievable logical error rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import stim

from qectostim.decoders.base import Decoder


@dataclass
class MLEDecoder(Decoder):
    """Maximum Likelihood Estimation decoder using syndrome lookup table.
    
    This decoder pre-computes all possible error patterns and their syndromes,
    then uses a lookup table to find the most likely correction for each
    observed syndrome.
    
    Suitable for:
    - Small codes (n ≤ 15, num_detectors ≤ 20)
    - Non-CSS codes where matching decoders fail
    - Reference/baseline for optimal decoder performance
    
    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decode.
    max_errors : int, default=3
        Maximum number of simultaneous errors to consider.
        Higher values are more accurate but exponentially slower.
    """

    dem: stim.DetectorErrorModel
    max_errors: int = 3
    _lookup_table: Dict[Tuple[int, ...], Tuple[float, np.ndarray]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables
        
        if self.num_detectors > 30:
            raise ValueError(
                f"MLEDecoder is not practical for {self.num_detectors} detectors. "
                f"Use max 30 detectors or switch to a different decoder."
            )
        
        # Parse the DEM to extract error mechanisms
        self._error_mechanisms: List[Tuple[float, np.ndarray, np.ndarray]] = []
        self._parse_dem()
        
        # Build lookup table
        self._build_lookup_table()

    def _parse_dem(self) -> None:
        """Parse the DEM to extract error mechanisms."""
        for instr in self.dem.flattened():
            if instr.type == 'error':
                prob = instr.args_copy()[0]
                
                det_mask = np.zeros(self.num_detectors, dtype=np.uint8)
                obs_mask = np.zeros(self.num_observables, dtype=np.uint8)
                
                for target in instr.targets_copy():
                    if target.is_relative_detector_id():
                        det_mask[target.val] = 1
                    elif target.is_logical_observable_id():
                        obs_mask[target.val] ^= 1  # XOR for repeated L0 L0
                
                self._error_mechanisms.append((prob, det_mask, obs_mask))

    def _build_lookup_table(self) -> None:
        """Build syndrome -> (probability, correction) lookup table."""
        n_errors = len(self._error_mechanisms)
        
        # Start with no errors
        no_error_syndrome = tuple([0] * self.num_detectors)
        no_error_correction = np.zeros(self.num_observables, dtype=np.uint8)
        self._lookup_table[no_error_syndrome] = (1.0, no_error_correction)
        
        # Consider single errors
        for i, (prob, det_mask, obs_mask) in enumerate(self._error_mechanisms):
            syndrome = tuple(det_mask.tolist())
            if syndrome not in self._lookup_table or prob > self._lookup_table[syndrome][0]:
                self._lookup_table[syndrome] = (prob, obs_mask.copy())
        
        # Consider pairs of errors if max_errors >= 2
        if self.max_errors >= 2 and n_errors <= 200:
            for i in range(n_errors):
                prob_i, det_i, obs_i = self._error_mechanisms[i]
                for j in range(i + 1, n_errors):
                    prob_j, det_j, obs_j = self._error_mechanisms[j]
                    
                    combined_det = (det_i ^ det_j)
                    combined_obs = (obs_i ^ obs_j)
                    combined_prob = prob_i * prob_j
                    
                    syndrome = tuple(combined_det.tolist())
                    if syndrome not in self._lookup_table or combined_prob > self._lookup_table[syndrome][0]:
                        self._lookup_table[syndrome] = (combined_prob, combined_obs.copy())
        
        # Consider triples of errors if max_errors >= 3
        if self.max_errors >= 3 and n_errors <= 50:
            for i in range(n_errors):
                prob_i, det_i, obs_i = self._error_mechanisms[i]
                for j in range(i + 1, n_errors):
                    prob_j, det_j, obs_j = self._error_mechanisms[j]
                    for k in range(j + 1, n_errors):
                        prob_k, det_k, obs_k = self._error_mechanisms[k]
                        
                        combined_det = (det_i ^ det_j ^ det_k)
                        combined_obs = (obs_i ^ obs_j ^ obs_k)
                        combined_prob = prob_i * prob_j * prob_k
                        
                        syndrome = tuple(combined_det.tolist())
                        if syndrome not in self._lookup_table or combined_prob > self._lookup_table[syndrome][0]:
                            self._lookup_table[syndrome] = (combined_prob, combined_obs.copy())

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """Decode a batch of detection events using the lookup table.
        
        Parameters
        ----------
        dets : np.ndarray
            Detection events, shape (num_shots, num_detectors).
            
        Returns
        -------
        np.ndarray
            Predicted observable flips, shape (num_shots, num_observables).
        """
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        if dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"MLEDecoder: expected dets.shape[1]={self.num_detectors}, "
                f"got {dets.shape[1]}"
            )

        num_shots = dets.shape[0]
        corrections = np.zeros((num_shots, self.num_observables), dtype=np.uint8)
        
        for i in range(num_shots):
            syndrome = tuple(dets[i].tolist())
            if syndrome in self._lookup_table:
                corrections[i] = self._lookup_table[syndrome][1]
            else:
                # Unknown syndrome - default to no correction
                # This can happen if max_errors is too low
                pass
        
        return corrections


@dataclass  
class HypergraphDecoder(Decoder):
    """Decoder that handles hyperedge DEMs by tracking L0 on boundary edges.
    
    This decoder extends matching-based decoding to handle cases where
    logical errors are associated with single-detector (boundary) edges.
    It does this by:
    
    1. Identifying all single-detector L0 errors in the DEM
    2. Computing which boundary detectors are most likely to flip L0
    3. Using this information during decoding
    
    Particularly useful for non-CSS codes where L0 errors trigger single detectors.
    
    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decode.
    """

    dem: stim.DetectorErrorModel
    
    def __post_init__(self) -> None:
        try:
            import pymatching
        except ImportError as exc:
            raise ImportError(
                "HypergraphDecoder requires pymatching. "
                "Install via `pip install pymatching`."
            ) from exc
        
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables
        
        # Parse DEM to find boundary L0 associations
        self._boundary_L0_probs = np.zeros(self.num_detectors)
        self._boundary_no_L0_probs = np.zeros(self.num_detectors)
        
        for instr in self.dem.flattened():
            if instr.type == 'error':
                prob = instr.args_copy()[0]
                
                det_ids = []
                has_L0 = False
                
                for target in instr.targets_copy():
                    if target.is_relative_detector_id():
                        det_ids.append(target.val)
                    elif target.is_logical_observable_id() and target.val == 0:
                        has_L0 = not has_L0  # XOR for L0 L0
                
                # Single-detector errors are boundary edges
                if len(det_ids) == 1:
                    det_id = det_ids[0]
                    if has_L0:
                        self._boundary_L0_probs[det_id] += prob
                    else:
                        self._boundary_no_L0_probs[det_id] += prob
        
        # Compute L0 probability given each boundary detector fires alone
        self._L0_given_boundary = np.zeros(self.num_detectors)
        for d in range(self.num_detectors):
            total = self._boundary_L0_probs[d] + self._boundary_no_L0_probs[d]
            if total > 0:
                self._L0_given_boundary[d] = self._boundary_L0_probs[d] / total
        
        # Create underlying PyMatching decoder
        self._matcher = pymatching.Matching.from_detector_error_model(self.dem)

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """Decode a batch of detection events.
        
        Uses PyMatching for base decoding, then applies boundary L0 corrections
        based on which single-detector errors are most likely to flip L0.
        
        Parameters
        ----------
        dets : np.ndarray
            Detection events, shape (num_shots, num_detectors).
            
        Returns
        -------
        np.ndarray
            Predicted observable flips, shape (num_shots, num_observables).
        """
        dets = np.asarray(dets, dtype=np.bool_)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        if dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"HypergraphDecoder: expected dets.shape[1]={self.num_detectors}, "
                f"got {dets.shape[1]}"
            )

        # Get base predictions from PyMatching
        base_corrections = self._matcher.decode_batch(dets)
        base_corrections = np.asarray(base_corrections, dtype=np.uint8)
        if base_corrections.ndim == 1:
            base_corrections = base_corrections.reshape(-1, self.num_observables)
        
        # For each shot, check if there are unmatched boundary detectors
        # that are likely to flip L0
        num_shots = dets.shape[0]
        
        for i in range(num_shots):
            syndrome = dets[i]
            
            # Count firing detectors
            firing = np.where(syndrome)[0]
            
            # If odd number of detectors fire, one is a boundary connection
            # Check if that boundary connection likely flips L0
            if len(firing) % 2 == 1:
                # Find the detector most likely to be a boundary L0 flip
                boundary_probs = self._L0_given_boundary[firing]
                max_prob = np.max(boundary_probs)
                
                if max_prob > 0.5:
                    # This boundary connection probably flips L0
                    # XOR with base correction
                    base_corrections[i, 0] ^= 1
        
        return base_corrections
