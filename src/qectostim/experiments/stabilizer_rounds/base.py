# src/qectostim/experiments/stabilizer_rounds/base.py
"""
Base classes for stabilizer round builders.

This module provides the StabilizerBasis enum and BaseStabilizerRoundBuilder
abstract base class that all concrete builders extend.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import stim

if TYPE_CHECKING:
    from qectostim.codes.abstract_code import Code

from .context import DetectorContext


class StabilizerBasis(Enum):
    """Which stabilizers to measure in a round."""
    X = "x"
    Z = "z"
    BOTH = "both"


class BaseStabilizerRoundBuilder:
    """
    Base class for stabilizer measurement round builders.
    
    Provides common functionality for tracking measurements, emitting
    detectors, and managing qubit coordinates. Subclasses specialize
    for CSS codes, general stabilizer codes, and color codes.
    
    The builder pattern allows granular control for FT gadget experiments
    while enabling simple full-circuit generation for memory experiments.
    
    Parameters
    ----------
    code : Code
        The quantum error correcting code.
    ctx : DetectorContext
        Context for tracking measurements and detectors.
    block_name : str
        Name for this code block (for multi-code gadgets).
    data_offset : int
        Offset for data qubit indices in the global circuit.
    ancilla_offset : int
        Offset for ancilla qubit indices.
    measurement_basis : str
        The basis for memory experiment ("Z" or "X"). Used to determine
        which first-round detectors are valid.
    """
    
    def __init__(
        self,
        code: "Code",
        ctx: DetectorContext,
        block_name: str = "main",
        data_offset: int = 0,
        ancilla_offset: Optional[int] = None,
        measurement_basis: str = "Z",
    ):
        self.code = code
        self.ctx = ctx
        self.block_name = block_name
        self.data_offset = data_offset
        self.measurement_basis = measurement_basis.upper()
        
        # Compute ancilla offset if not provided
        n = code.n
        if ancilla_offset is None:
            self.ancilla_offset = data_offset + n
        else:
            self.ancilla_offset = ancilla_offset
        
        # Cache metadata using Code ABC property
        self._meta = code.metadata or {}
        self._data_coords = self._meta.get('data_coords', [])
        
        # Periodic boundary support for toric codes
        self._lattice_size = self._meta.get('lattice_size')
        
        # Build coordinate lookup for geometric scheduling
        # Use N-dimensional keys to support 2D, 3D, and higher-dimensional codes
        self._coord_to_data: Dict[Tuple[float, ...], int] = {}
        for local_idx, coord in enumerate(self._data_coords):
            if len(coord) >= 2:
                # Use full N-dimensional coordinate as key
                key = tuple(float(c) for c in coord)
                self._coord_to_data[key] = data_offset + local_idx
                # Also store 2D projection for backward compatibility
                key_2d = (float(coord[0]), float(coord[1]))
                if key_2d not in self._coord_to_data:
                    self._coord_to_data[key_2d] = data_offset + local_idx
        
        # Track round number for detector emission
        self._round_number = 0
    
    @property
    def has_metachecks(self) -> bool:
        """Whether this builder supports metacheck detectors.
        
        Default: False. CSSStabilizerRoundBuilder overrides when code
        has meta_x/meta_z check matrices.
        """
        return False

    @property
    def x_ancillas(self) -> List[int]:
        """X-type stabilizer ancilla qubit indices. Override in subclasses."""
        return []

    @property
    def z_ancillas(self) -> List[int]:
        """Z-type stabilizer ancilla qubit indices. Override in subclasses."""
        return []

    @property
    def qubit_coords(self) -> Dict[int, Tuple[float, ...]]:
        """Mapping of qubit index to coordinates. Override in subclasses."""
        return {}

    def reset_stabilizer_history(
        self,
        swap_xz: bool = False,
        skip_first_round: bool = False,
        clear_history: bool = False,
    ) -> None:
        """Reset stabilizer history after a gate transform.
        
        Override in subclasses for proper history management.
        Default: reset round counter.
        
        Parameters
        ----------
        swap_xz : bool
            If True, X and Z stabilizers were swapped by the gate.
        skip_first_round : bool
            If True, skip first-round anchor detectors after reset.
        clear_history : bool
            If True, clear all measurement history (teleportation).
        """
        self._round_number = 0
    
    def _wrap_coord(self, coord: Tuple[float, ...]) -> Tuple[float, ...]:
        """Wrap a coordinate for periodic boundary conditions.
        
        For toric codes with `lattice_size` metadata, coordinates may need
        to wrap around the torus. This method handles that wrapping.
        
        The coordinate system for toric codes typically has:
        - Integer coordinates for data qubits on edges
        - Half-integer coordinates (x+0.5, y) or (x, y+0.5) for edge midpoints
        
        We wrap so that e.g. coord -0.5 wraps to lattice_size - 0.5.
        
        Supports N-dimensional coordinates for 3D toric codes and beyond.
        """
        if self._lattice_size is None:
            return coord
        
        L = float(self._lattice_size)
        
        # Wrap all dimensions to [0, L) range
        wrapped = tuple(c % L for c in coord)
        return wrapped
    
    def _lookup_data_qubit(self, coord: Tuple[float, ...]) -> Optional[int]:
        """Look up a data qubit by coordinate, handling periodic boundaries.
        
        First tries direct lookup with the full N-dimensional coordinate.
        If that fails, tries 2D projection for backward compatibility.
        If that fails and lattice_size is set, tries wrapped coordinates.
        
        Parameters
        ----------
        coord : Tuple[float, ...]
            N-dimensional coordinate to look up (typically 2D or 3D).
            
        Returns
        -------
        Optional[int]
            Global index of the data qubit at this coordinate, or None if not found.
        """
        # Direct lookup with full coordinate
        result = self._coord_to_data.get(coord)
        if result is not None:
            return result
        
        # Try 2D projection for backward compatibility
        if len(coord) >= 2:
            coord_2d = (coord[0], coord[1])
            result = self._coord_to_data.get(coord_2d)
            if result is not None:
                return result
        
        # Try wrapped coordinate for toric codes (both full and 2D)
        if self._lattice_size is not None:
            wrapped = self._wrap_coord(coord)
            result = self._coord_to_data.get(wrapped)
            if result is not None:
                return result
            
            # Try 2D projection of wrapped coordinate
            if len(wrapped) >= 2:
                wrapped_2d = (wrapped[0], wrapped[1])
                result = self._coord_to_data.get(wrapped_2d)
                if result is not None:
                    return result
        
        return None
    
    @property
    def data_qubits(self) -> List[int]:
        """Global indices of data qubits."""
        return list(range(self.data_offset, self.data_offset + self.code.n))
    
    @property
    def total_qubits(self) -> int:
        """Total qubits used by this block (data + ancillas)."""
        raise NotImplementedError("Subclass must implement total_qubits")
    
    def emit_qubit_coords(self, circuit: stim.Circuit) -> None:
        """Emit QUBIT_COORDS for all qubits in this block."""
        # Data qubits
        for local_idx, coord in enumerate(self._data_coords):
            if len(coord) >= 2:
                global_idx = self.data_offset + local_idx
                circuit.append("QUBIT_COORDS", [global_idx], [float(coord[0]), float(coord[1])])
    
    def emit_reset_all(self, circuit: stim.Circuit) -> None:
        """Reset all data and ancilla qubits."""
        raise NotImplementedError("Subclass must implement emit_reset_all")
    
    def emit_prepare_logical_state(
        self,
        circuit: stim.Circuit,
        state: str = "0",
        logical_idx: int = 0,
    ) -> None:
        """
        Prepare a logical eigenstate.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        state : str
            "0" for |0⟩_L, "1" for |1⟩_L, "+" for |+⟩_L, "-" for |-⟩_L.
        logical_idx : int
            Which logical qubit.
        """
        raise NotImplementedError("Subclass must implement emit_prepare_logical_state")
    
    def emit_round(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_detectors: bool = True,
        **kwargs,
    ) -> None:
        """
        Emit one complete stabilizer measurement round.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        stab_type : StabilizerBasis
            Which stabilizers to measure.
        emit_detectors : bool
            Whether to emit time-like detectors.
        **kwargs
            Additional parameters for subclass-specific features:
            - emit_z_anchors: bool - Emit Z anchor detectors (CSS codes)
            - emit_x_anchors: bool - Emit X anchor detectors (CSS codes)
            - explicit_anchor_mode: bool - Only use explicit anchor flags
            - first_basis: StabilizerBasis - Which basis to measure first
        """
        raise NotImplementedError("Subclass must implement emit_round")
    
    def emit_round_with_transform(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_detectors: bool = True,
        swap_xz: bool = False,
    ) -> None:
        """
        Emit a stabilizer round with transform applied.
        
        Default implementation just calls emit_round. CSS codes override this
        to handle stabilizer swapping after Hadamard gates.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        stab_type : StabilizerBasis
            Which stabilizers to measure.
        emit_detectors : bool
            Whether to emit time-like detectors.
        swap_xz : bool
            If True, apply X↔Z swap transform (for Hadamard gates).
        """
        # Default: ignore transform and emit normally
        self.emit_round(circuit, stab_type, emit_detectors)
    
    def emit_rounds(
        self,
        circuit: stim.Circuit,
        num_rounds: int,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
    ) -> None:
        """Emit multiple stabilizer measurement rounds."""
        for _ in range(num_rounds):
            self.emit_round(circuit, stab_type, emit_detectors=True)
    
    def emit_z_layer(
        self,
        circuit: stim.Circuit,
        emit_detectors: bool = True,
        emit_z_anchors: bool = False,
    ) -> None:
        """
        Emit only Z stabilizer measurements (for parallel extraction).
        
        Default implementation falls back to emit_round with Z basis.
        CSS codes override this for true layer-by-layer emission.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        emit_detectors : bool
            Whether to emit detectors.
        emit_z_anchors : bool
            Whether to emit anchor detectors for Z stabilizers.
        """
        self.emit_round(
            circuit,
            StabilizerBasis.Z,
            emit_detectors=emit_detectors,
            emit_z_anchors=emit_z_anchors,
            explicit_anchor_mode=True,
        )
    
    def emit_x_layer(
        self,
        circuit: stim.Circuit,
        emit_detectors: bool = True,
        emit_x_anchors: bool = False,
    ) -> None:
        """
        Emit only X stabilizer measurements (for parallel extraction).
        
        Default implementation falls back to emit_round with X basis.
        CSS codes override this for true layer-by-layer emission.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        emit_detectors : bool
            Whether to emit detectors.
        emit_x_anchors : bool
            Whether to emit anchor detectors for X stabilizers.
        """
        self.emit_round(
            circuit,
            StabilizerBasis.X,
            emit_detectors=emit_detectors,
            emit_x_anchors=emit_x_anchors,
            explicit_anchor_mode=True,
        )
    
    def finalize_parallel_round(self, circuit: stim.Circuit) -> None:
        """
        Finalize a round after parallel extraction of both layers.
        
        Called after both emit_z_layer and emit_x_layer to handle any
        cleanup needed (e.g., emitting TICKs, advancing round counter).
        
        Default implementation does nothing. CSS codes override this
        to emit final TICK and update round tracking.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        """
        # Default: no finalization needed
        pass

    def emit_final_measurement(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
        logical_idx: int = 0,
    ) -> List[int]:
        """Emit final data qubit measurements with space-like detectors."""
        raise NotImplementedError("Subclass must implement emit_final_measurement")
    
    def _get_detector_coord(
        self,
        stab_coords: Optional[List],
        s_idx: int,
        t: float,
    ) -> Tuple[float, ...]:
        """
        Get detector coordinates for a stabilizer.
        
        Subclasses can override to add additional coordinates (e.g., color).
        
        Returns
        -------
        Tuple[float, ...]
            Detector coordinates (x, y, t) or extended.
        """
        if stab_coords is None or s_idx >= len(stab_coords):
            return (0.0, 0.0, t)
        coord = stab_coords[s_idx]
        if len(coord) >= 2:
            return (float(coord[0]), float(coord[1]), t)
        return (0.0, 0.0, t)
    
    def get_last_measurement_indices(self) -> Dict[str, List[int]]:
        """
        Get the measurement indices from the last round.
        
        Used for crossing detectors that need to compare pre-gate and
        post-gate syndrome measurements.
        
        Returns
        -------
        Dict[str, List[int]]
            Mapping from stabilizer type ("X" or "Z") to list of
            absolute measurement indices from the last round.
        """
        # Default implementation returns empty - CSS codes override this
        return {"X": [], "Z": []}
    
    def get_last_x_syndrome_indices(self) -> List[int]:
        """Get X syndrome measurement indices from the last round."""
        return self.get_last_measurement_indices().get("X", [])
    
    def get_last_z_syndrome_indices(self) -> List[int]:
        """Get Z syndrome measurement indices from the last round."""
        return self.get_last_measurement_indices().get("Z", [])
    
    def _emit_shift_coords(self, circuit: stim.Circuit) -> None:
        """Emit SHIFT_COORDS for time advancement."""
        if self._data_coords:
            circuit.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
