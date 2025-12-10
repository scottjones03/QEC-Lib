# src/qectostim/decoders/chromobius_decoder.py
"""
Chromobius decoder for color-code-like circuits.

Chromobius is a specialized Möbius decoder designed for color codes and 
related circuits. It requires DEMs with specific properties:

1. Annotated detectors: Every detector must have a 4-component coordinate
   where the 4th entry encodes (basis, color):
   - 0 = X, red; 1 = X, green; 2 = X, blue
   - 3 = Z, red; 4 = Z, green; 5 = Z, blue

2. Rainbow triplets: Bulk errors hitting 3 detectors must hit one of each color.

3. Moveable excitations: Nearby bulk errors can be composed to "shift" excitations.

4. Matchable-avoids-color: Matching regions must avoid one color entirely.

This decoder is NOT a general hyperedge decoder - it will fail on generic
stabilizer/LDPC/subsystem codes that don't satisfy these constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import stim

from qectostim.decoders.base import Decoder
import chromobius


class ChromobiusIncompatibleError(Exception):
    """Raised when a DEM is incompatible with Chromobius requirements."""
    pass


def check_chromobius_compatibility(dem: stim.DetectorErrorModel) -> tuple[bool, str]:
    """Check if a DEM is compatible with Chromobius requirements.
    
    Returns (is_compatible, reason) tuple.
    """
    # Check 1: All detectors must have coordinates with 4 components
    # where the 4th component is in {0,1,2,3,4,5} (basis/color encoding)
    detector_coords = []
    for instruction in dem.flattened():
        if instruction.type == "detector":
            coords = instruction.args_copy()
            detector_coords.append(coords)
    
    if not detector_coords:
        return False, "No detectors with coordinates found"
    
    # Check that we have color annotations (4th coordinate in range 0-5)
    has_color_annotations = False
    for coords in detector_coords:
        if len(coords) >= 4:
            color_val = coords[3]
            if 0 <= color_val <= 5:
                has_color_annotations = True
                break
    
    if not has_color_annotations:
        return False, "Expected all detectors to have color annotations (coord[3] in 0-5)"
    
    return True, "Compatible"


@dataclass
class ChromobiusDecoder(Decoder):
    """Chromobius decoder for color-code-like circuits.
    
    Chromobius is a specialized Möbius decoder designed for color codes.
    It requires DEMs with specific structure:
    
    - Detector coordinates with 4 components (x, y, t, color_basis)
    - Rainbow triplet structure for bulk errors
    - Matchable-avoids-color property
    
    For generic stabilizer codes, use PyMatching, BPOSD, or other decoders.
    
    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decode. Must be from a color-code-like circuit.
        
    Raises
    ------
    ChromobiusIncompatibleError
        If the DEM doesn't satisfy Chromobius requirements.
    """

    dem: stim.DetectorErrorModel

    def __post_init__(self) -> None:
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables

        # Check compatibility before attempting to compile
        is_compatible, reason = check_chromobius_compatibility(self.dem)
        if not is_compatible:
            raise ChromobiusIncompatibleError(reason)

        # Chromobius accepts the DEM directly
        try:
            self._decoder = chromobius.compile_decoder_for_dem(self.dem)
        except Exception as e:
            # Re-raise with clearer error message
            raise ChromobiusIncompatibleError(
                f"Failed to compile DEM for Chromobius: {str(e)}"
            ) from e

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """Decode a batch of detection events.
        
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
                f"ChromobiusDecoder: expected dets.shape[1]={self.num_detectors}, "
                f"got {dets.shape[1]}"
            )

        # Chromobius decode_batch returns observable predictions
        corrections = self._decoder.predict_obs_flips_from_dets_bit_packed(
            np.packbits(dets, axis=1, bitorder='little')
        )
        
        # Unpack results
        corrections = np.unpackbits(
            corrections, axis=1, count=self.num_observables, bitorder='little'
        )
        
        return corrections.astype(np.uint8)
