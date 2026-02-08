# qectostim/decoders/hybrid_teleportation_decoder.py
"""
Hybrid decoder for teleportation-based gadgets.

This decoder combines DEM-based error correction with classical Pauli frame tracking.
The key insight is that teleportation has RANDOM measurements that can't be included
in Stim's DEM observables, but we can:

1. Use the DEM for error correction (detectors compare stabilizer rounds)
2. Emit "clean" observables (just final data measurements, no frame measurements)
3. Apply Pauli frame corrections CLASSICALLY after decoding

For teleported Hadamard:
- OBSERVABLE_0: Data block Z_L (teleportation measurement outcome)
- OBSERVABLE_1: Ancilla block X_L (final logical state after H)

Final logical error = (decoded_obs_1) XOR (projection_frame) XOR (decoded_obs_0)

Where projection_frame = XOR of projection X measurements with odd overlap with X_L
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import stim

from qectostim.decoders.base import Decoder
from qectostim.decoders.pymatching_decoder import PyMatchingDecoder


@dataclass
class HybridDecodingResult:
    """Result from hybrid teleportation decoding.
    
    Attributes:
        logical_errors: Boolean array of logical errors (after frame correction)
        decoded_obs_0: Decoded observable 0 (data Z_L)
        decoded_obs_1: Decoded observable 1 (ancilla X_L)
        projection_frames: Projection frame corrections applied
        num_shots: Number of shots decoded
    """
    logical_errors: np.ndarray
    decoded_obs_0: np.ndarray
    decoded_obs_1: np.ndarray
    projection_frames: np.ndarray
    num_shots: int


@dataclass
class HybridTeleportationDecoder(Decoder):
    """
    Hybrid decoder combining DEM-based error correction with classical frame tracking.
    
    This decoder handles teleportation gadgets where:
    1. Projection measurements are RANDOM (define the Pauli frame)
    2. Teleportation measurements are RANDOM (encode the Bell outcome)
    3. These random outcomes can't be included in Stim's deterministic DEM observables
    
    Solution:
    - Use PyMatchingDecoder for DEM-based error correction on TWO observables:
      - OBSERVABLE_0: Data block Z_L (teleportation measurement)
      - OBSERVABLE_1: Ancilla block X_L (final measurement)
    - Apply projection frame correction CLASSICALLY using raw measurement sampling
    
    The projection frame is the XOR of all X stabilizer measurements that have
    odd overlap with X_L on the ancilla block.
    
    Usage:
        circuit = experiment.to_stim()
        dem = circuit.detector_error_model()
        
        # Get frame tracking info from experiment
        frame_info = experiment.get_frame_tracking_info()
        
        decoder = HybridTeleportationDecoder(
            dem=dem,
            frame_meas_indices=frame_info['projection_frame_meas'],
        )
        
        # Sample raw measurements (needed for frame tracking)
        sampler = circuit.compile_sampler()
        raw_samples = sampler.sample(num_shots)
        
        # Sample detectors
        det_sampler = circuit.compile_detector_sampler()
        det_samples = det_sampler.sample(num_shots, separate_observables=True)
        
        result = decoder.decode_with_frame(det_samples[0], raw_samples)
    """
    
    dem: stim.DetectorErrorModel
    frame_meas_indices: List[int] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Initialize the underlying PyMatching decoder."""
        self._inner_decoder = PyMatchingDecoder(self.dem)
        self._num_observables = self.dem.num_observables
    
    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """
        Standard decode_batch interface (without frame correction).
        
        This just delegates to the inner PyMatchingDecoder.
        For full hybrid decoding with frame correction, use decode_with_frame().
        """
        return self._inner_decoder.decode_batch(dets)
    
    def decode_with_frame(
        self,
        detector_samples: np.ndarray,
        raw_samples: np.ndarray,
    ) -> HybridDecodingResult:
        """
        Decode with classical Pauli frame tracking.
        
        Parameters
        ----------
        detector_samples : np.ndarray
            Detector samples from circuit.compile_detector_sampler().sample(n)
            Shape: (num_shots, num_detectors)
        raw_samples : np.ndarray
            Raw measurement samples from circuit.compile_sampler().sample(n)
            Shape: (num_shots, num_measurements)
            
        Returns
        -------
        HybridDecodingResult
            Result containing logical errors after frame correction.
        """
        num_shots = detector_samples.shape[0]
        
        # Decode using PyMatching
        decoded = self._inner_decoder.decode_batch(detector_samples)
        # decoded shape: (num_shots, num_observables)
        
        if decoded.ndim == 1:
            decoded = decoded.reshape(-1, self._num_observables)
        
        # Extract decoded observables
        decoded_obs_0 = decoded[:, 0] if self._num_observables > 0 else np.zeros(num_shots, dtype=np.uint8)
        decoded_obs_1 = decoded[:, 1] if self._num_observables > 1 else np.zeros(num_shots, dtype=np.uint8)
        
        # Compute projection frame from raw samples
        projection_frames = np.zeros(num_shots, dtype=np.uint8)
        for meas_idx in self.frame_meas_indices:
            if meas_idx < raw_samples.shape[1]:
                projection_frames ^= raw_samples[:, meas_idx].astype(np.uint8)
        
        # Final logical error:
        # For Hadamard: output is H|input⟩
        # If input is |0⟩_L, output should be |+⟩_L (X_L eigenstate with eigenvalue +1)
        # Error if X_L measurement gives -1 (value 1)
        #
        # But we need to account for:
        # 1. Projection frame: random outcome that defines the logical state
        # 2. Teleportation frame: decoded Z_L on data block
        #
        # Final = decoded_obs_1 XOR projection_frame XOR decoded_obs_0
        logical_errors = (decoded_obs_1 ^ projection_frames ^ decoded_obs_0).astype(np.uint8)
        
        return HybridDecodingResult(
            logical_errors=logical_errors,
            decoded_obs_0=decoded_obs_0,
            decoded_obs_1=decoded_obs_1,
            projection_frames=projection_frames,
            num_shots=num_shots,
        )
    
    def decode_simple(
        self,
        detector_samples: np.ndarray,
        raw_samples: np.ndarray,
    ) -> np.ndarray:
        """
        Simple interface returning just logical errors.
        
        Parameters
        ----------
        detector_samples : np.ndarray
            Detector samples from circuit.compile_detector_sampler().sample(n)
        raw_samples : np.ndarray
            Raw measurement samples from circuit.compile_sampler().sample(n)
            
        Returns
        -------
        np.ndarray
            Boolean array of logical errors, shape (num_shots,)
        """
        result = self.decode_with_frame(detector_samples, raw_samples)
        return result.logical_errors


def compute_frame_meas_indices(code, projection_x_meas: List[int], meas_base: int = 0) -> List[int]:
    """
    Compute which projection X measurements contribute to the frame.
    
    For |+⟩_L preparation, X stabilizers with ODD overlap with X_L contribute.
    
    Parameters
    ----------
    code : Code
        The code (for X stabilizer and X_L support info)
    projection_x_meas : List[int]
        Local measurement indices for X stabilizer measurements
    meas_base : int
        Global measurement index offset
        
    Returns
    -------
    List[int]
        Global measurement indices that contribute to the projection frame
    """
    # Get X_L support
    lx = getattr(code, 'Lx', None)
    if lx is None:
        lx = getattr(code, 'lx', None)
    if lx is None:
        return []
    
    lx = np.atleast_2d(lx)
    x_l_support = set(np.where(lx[0])[0])
    
    # Get X stabilizer matrix
    hx = getattr(code, 'hx', None)
    if hx is None:
        return []
    
    # Find stabilizers with odd overlap
    frame_meas = []
    for stab_idx, local_meas_idx in enumerate(projection_x_meas):
        if stab_idx < hx.shape[0]:
            stab_support = set(np.where(hx[stab_idx])[0])
            overlap = len(stab_support & x_l_support)
            if overlap % 2 == 1:
                frame_meas.append(meas_base + local_meas_idx)
    
    return frame_meas
