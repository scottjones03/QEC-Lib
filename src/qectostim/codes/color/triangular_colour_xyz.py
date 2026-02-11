"""Triangular Colour Code with Joint XYZ Measurements

This variant of the triangular color code uses joint XYZ measurements
instead of separate X and Z stabilizer measurements. This matches Stim's
built-in color_code:memory_xyz circuit structure.

Key differences from TriangularColourCode:
- Uses C_XYZ gate on data qubits to rotate measurement basis
- Single ancilla per face (not separate X and Z ancillas)
- Half the number of detectors per round
- Compatible with Chromobius decoder using color annotations

The joint XYZ measurement approach:
1. Apply C_XYZ to data qubits (X→Y→Z→X rotation)
2. Apply CNOTs from data to ancilla
3. Measure ancilla in computational basis
4. This effectively measures the product of X, Y, Z on data qubits

Usage:
------
Use XYZColorCodeMemoryExperiment to build circuits:

    from qectostim.codes.color import TriangularColourCodeXYZ
    from qectostim.experiments import XYZColorCodeMemoryExperiment
    
    code = TriangularColourCodeXYZ(d=3)
    exp = XYZColorCodeMemoryExperiment(code, rounds=3)
    circuit = exp.to_stim()

Code Parameters:
    [[n, 1, d]] on a 6.6.6 lattice, where n grows quadratically with the
    distance d.  Only odd d ≥ 3 are supported.

Stabiliser Structure:
    Weight-6 mixed XYZ stabilisers.  Each face of the 6.6.6 lattice
    contributes a single joint XYZ measurement (as opposed to the
    separate X and Z plaquette measurements used by the standard variant).

Raises:
    ValueError
        If *distance* is even or less than 3.

Reference: Stim's color_code:memory_xyz implementation
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import math
import stim

from qectostim.codes.color.triangular_colour import TriangularColourCode


class TriangularColourCodeXYZ(TriangularColourCode):
    """
    Triangular colour code with joint XYZ measurements.
    
    This class provides metadata for XYZ color code circuits that use
    C_XYZ gates for basis rotation, matching Stim's color_code:memory_xyz.
    
    To build a memory experiment circuit, use XYZColorCodeMemoryExperiment:
    
        from qectostim.experiments import XYZColorCodeMemoryExperiment
        
        code = TriangularColourCodeXYZ(d=5)
        exp = XYZColorCodeMemoryExperiment(code, rounds=3)
        circuit = exp.to_stim()
    
    Parameters
    ----------
    distance : int
        Code distance (must be odd, >= 3).
        
    Attributes
    ----------
    metadata : dict
        Contains XYZ-specific metadata:
        - measurement_style: "XYZ"
        - faces: list of data qubit indices per stabilizer face
        - stab_colors: color (0-2) per face
        - ancilla_coords: (x, y) coordinates for ancilla qubits
        - n_ancillas: number of ancilla qubits (one per face)
        - is_chromobius_compatible: True
    """

    def __init__(self, distance: int = 3, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(distance=distance, metadata=metadata)
        
        # Update metadata to indicate XYZ measurement style
        self._metadata["measurement_style"] = "XYZ"
        self._metadata["name"] = f"TriangularColourXYZ_d{distance}"
        
        # Build ancilla coordinates (one per face, at face centers)
        self._build_ancilla_layout()
    
    def _build_ancilla_layout(self):
        """Build ancilla qubit layout - one ancilla per stabilizer face."""
        faces = self._faces
        coords = self._coords
        
        # Ancilla coordinates are at face centers
        ancilla_coords = []
        for face in faces:
            cx = sum(coords[q][0] for q in face) / len(face)
            cy = sum(coords[q][1] for q in face) / len(face)
            ancilla_coords.append((cx, cy))
        
        self._ancilla_coords = ancilla_coords
        self._n_ancillas = len(faces)
        
        # Update metadata
        self._metadata["ancilla_coords"] = ancilla_coords
        self._metadata["n_ancillas"] = self._n_ancillas
        self._metadata["total_qubits"] = self.n + self._n_ancillas
    
    @property
    def num_detectors_per_round(self) -> int:
        """Number of detectors per syndrome round (one per face)."""
        return self._n_ancillas
    
    @property
    def total_qubits(self) -> int:
        """Total number of qubits (data + ancilla)."""
        return self.n + self._n_ancillas


# Pre-built instances
TriangularColourXYZ3 = lambda: TriangularColourCodeXYZ(distance=3)
TriangularColourXYZ5 = lambda: TriangularColourCodeXYZ(distance=5)
TriangularColourXYZ7 = lambda: TriangularColourCodeXYZ(distance=7)
