# src/qectostim/experiments/stabilizer_rounds.py
"""
Reusable Stabilizer Round Builder and Detector Context.

Provides utilities for building fault-tolerant circuits with proper
detector continuity across memory → gadget → memory phases.

Key components:
- DetectorContext: Tracks measurement indices and stabilizer state across phases
- StabilizerRoundBuilder: Emits stabilizer measurement rounds with proper scheduling
- Observable tracking: Computes logical observable support through transformations

Based on TQEC's approach: memory experiments are the fundamental building block,
with gadgets sandwiched between memory rounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, TYPE_CHECKING
from enum import Enum

import numpy as np
import stim

from qectostim.codes.abstract_code import FTGadgetCodeConfig, ScheduleMode
from qectostim.utils.scheduling_core import (
    CodeMetadataCache,
    graph_coloring_cnots,
    schedule_stabilizer_cnots,
)

if TYPE_CHECKING:
    from qectostim.codes.abstract_code import Code, StabilizerCode
    from qectostim.codes.abstract_css import CSSCode


class StabilizerBasis(Enum):
    """Basis for stabilizer measurement."""
    X = "x"
    Z = "z"
    BOTH = "both"


@dataclass
class DetectorContext:
    """
    Tracks detector state across circuit phases for proper continuity.
    
    This is the key to connecting memory rounds before and after a gadget.
    The context maintains:
    - Measurement index counter
    - Last measurement index for each stabilizer
    - Detector coordinates (spatial + temporal)
    - Observable accumulation
    
    Using a DetectorContext ensures that:
    1. Time-like detectors compare measurements across phase boundaries
    2. The observable correctly tracks through transformations
    3. No measurement indices are double-counted or missed
    
    Example usage:
    ```python
    ctx = DetectorContext()
    
    # Phase 1: Pre-gadget memory
    builder = StabilizerRoundBuilder(code, ctx)
    builder.emit_rounds(circuit, num_rounds=5)
    
    # Phase 2: Gadget (passes ctx to maintain continuity)
    gadget.emit_with_context(circuit, ctx)
    
    # Phase 3: Post-gadget memory
    builder.emit_rounds(circuit, num_rounds=5)
    
    # Phase 4: Final measurement
    builder.emit_final_measurement(circuit)
    ctx.emit_observable(circuit)
    ```
    """
    
    # Measurement tracking
    measurement_index: int = 0
    
    # Stabilizer state: maps (block_name, stab_type, stab_idx) -> last measurement index
    # stab_type is "x" or "z"
    last_stabilizer_meas: Dict[Tuple[str, str, int], int] = field(default_factory=dict)
    
    # Time coordinate for detectors
    current_time: float = 0.0
    time_step: float = 1.0
    
    # Observable tracking: maps (observable_idx,) -> list of measurement indices
    observable_measurements: Dict[int, List[int]] = field(default_factory=dict)
    
    # Track observable transformations (from gate applications)
    observable_transforms: Dict[int, Dict[str, str]] = field(default_factory=dict)
    
    # Track which stabilizers have been measured at least once
    # (first round establishes baseline, not a detector)
    stabilizer_initialized: Set[Tuple[str, str, int]] = field(default_factory=set)
    
    def add_measurement(self, n: int = 1) -> int:
        """
        Record n measurements and return the starting index.
        
        Returns
        -------
        int
            The measurement index before adding (i.e., first of the new measurements).
        """
        start = self.measurement_index
        self.measurement_index += n
        return start
    
    def record_stabilizer_measurement(
        self,
        block_name: str,
        stab_type: str,
        stab_idx: int,
        meas_idx: int,
    ) -> Optional[int]:
        """
        Record a stabilizer measurement and return the previous measurement index.
        
        For time-like detectors, we need to compare with the previous measurement
        of the same stabilizer. This method updates the tracking and returns
        the previous index (or None if this is the first measurement).
        
        Parameters
        ----------
        block_name : str
            Name of the code block.
        stab_type : str
            "x" or "z".
        stab_idx : int
            Index of the stabilizer.
        meas_idx : int
            Current measurement index.
            
        Returns
        -------
        Optional[int]
            Previous measurement index, or None if first measurement.
        """
        key = (block_name, stab_type, stab_idx)
        prev = self.last_stabilizer_meas.get(key)
        self.last_stabilizer_meas[key] = meas_idx
        
        # Mark as initialized
        self.stabilizer_initialized.add(key)
        
        return prev
    
    def advance_time(self, delta: Optional[float] = None) -> None:
        """Advance the time coordinate."""
        self.current_time += delta if delta is not None else self.time_step
    
    def add_observable_measurement(
        self,
        observable_idx: int,
        meas_indices: List[int],
    ) -> None:
        """
        Add measurement indices to an observable's accumulator.
        
        For observables that span multiple phases (e.g., measurements
        before and after a gadget), this accumulates all contributing
        measurement indices.
        """
        if observable_idx not in self.observable_measurements:
            self.observable_measurements[observable_idx] = []
        self.observable_measurements[observable_idx].extend(meas_indices)
    
    def record_observable_transform(
        self,
        observable_idx: int,
        transform: Dict[str, str],
    ) -> None:
        """
        Record how a gate transforms the observable.
        
        For tracking how logical X/Z change through the circuit.
        E.g., after Hadamard: {'X': 'Z', 'Z': 'X'}
        """
        if observable_idx not in self.observable_transforms:
            self.observable_transforms[observable_idx] = {'X': 'X', 'Z': 'Z', 'Y': 'Y'}
        
        # Compose transformations
        current = self.observable_transforms[observable_idx]
        new_transform = {}
        for pauli, result in current.items():
            # Strip sign for lookup, preserve for result
            lookup = result.lstrip('-')
            sign = '-' if result.startswith('-') else ''
            if lookup in transform:
                new_result = transform[lookup]
                # Compose signs
                new_sign = '-' if (sign == '-') != new_result.startswith('-') else ''
                new_transform[pauli] = new_sign + new_result.lstrip('-')
            else:
                new_transform[pauli] = result
        
        self.observable_transforms[observable_idx] = new_transform
    
    def get_transformed_basis(self, observable_idx: int, original_basis: str) -> str:
        """Get the current basis of an observable after all transformations."""
        if observable_idx not in self.observable_transforms:
            return original_basis
        transform = self.observable_transforms[observable_idx]
        result = transform.get(original_basis, original_basis)
        return result.lstrip('-')  # Return just the basis, not the sign
    
    def emit_detector(
        self,
        circuit: stim.Circuit,
        meas_indices: List[int],
        coord: Tuple[float, ...],
    ) -> None:
        """
        Emit a DETECTOR instruction.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        meas_indices : List[int]
            Absolute measurement indices to include.
        coord : Tuple[float, ...]
            Detector coordinates (x, y, t) or (x, y, t, basis).
        """
        if not meas_indices:
            return
        
        lookbacks = [idx - self.measurement_index for idx in meas_indices]
        targets = [stim.target_rec(lb) for lb in lookbacks]
        circuit.append("DETECTOR", targets, list(coord))
    
    def emit_observable(
        self,
        circuit: stim.Circuit,
        observable_idx: int = 0,
    ) -> None:
        """
        Emit OBSERVABLE_INCLUDE for accumulated measurements.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        observable_idx : int
            Which logical observable.
        """
        meas_indices = self.observable_measurements.get(observable_idx, [])
        if not meas_indices:
            return
        
        lookbacks = [idx - self.measurement_index for idx in meas_indices]
        targets = [stim.target_rec(lb) for lb in lookbacks]
        circuit.append("OBSERVABLE_INCLUDE", targets, observable_idx)
    
    def clone(self) -> "DetectorContext":
        """Create a copy of the context."""
        return DetectorContext(
            measurement_index=self.measurement_index,
            last_stabilizer_meas=dict(self.last_stabilizer_meas),
            current_time=self.current_time,
            time_step=self.time_step,
            observable_measurements={k: list(v) for k, v in self.observable_measurements.items()},
            observable_transforms={k: dict(v) for k, v in self.observable_transforms.items()},
            stabilizer_initialized=set(self.stabilizer_initialized),
        )
    
    def update_for_gate(self, gate_name: str) -> None:
        """
        Update observable tracking for a gate transformation.
        
        This records how the logical observables transform through
        the gate, which affects how final measurements should be
        interpreted for the logical observable.
        
        Parameters
        ----------
        gate_name : str
            Name of the gate (H, S, T, CNOT, CZ, etc.)
        """
        # Gate transformation rules for Pauli observables
        gate_transforms = {
            # Hadamard: swaps X and Z
            "H": {"X": "Z", "Y": "-Y", "Z": "X"},
            # S gate: X -> Y, Y -> -X, Z -> Z
            "S": {"X": "Y", "Y": "-X", "Z": "Z"},
            "S_DAG": {"X": "-Y", "Y": "X", "Z": "Z"},
            # T gate: X -> (X+Y)/√2, but for logical tracking we use X -> X
            # (exact phase not needed for detector matching)
            "T": {"X": "X", "Y": "Y", "Z": "Z"},
            "T_DAG": {"X": "X", "Y": "Y", "Z": "Z"},
            # Pauli gates: just apply phases
            "X": {"X": "X", "Y": "-Y", "Z": "-Z"},
            "Y": {"X": "-X", "Y": "Y", "Z": "-Z"},
            "Z": {"X": "-X", "Y": "-Y", "Z": "Z"},
            # CNOT: X_c -> X_c X_t, Z_t -> Z_c Z_t
            # (Handled specially for two-qubit)
            "CNOT": {"X": "X", "Y": "Y", "Z": "Z"},  # Placeholder
            "CX": {"X": "X", "Y": "Y", "Z": "Z"},
            "CZ": {"X": "X", "Y": "Y", "Z": "Z"},  # Symmetric
        }
        
        transform = gate_transforms.get(gate_name.upper(), {"X": "X", "Y": "Y", "Z": "Z"})
        
        # Apply to observable 0 (can extend to multiple observables)
        self.record_observable_transform(0, transform)
    
    def clear_stabilizer_history(
        self,
        block_name: Optional[str] = None,
        swap_xz: bool = False,
    ) -> None:
        """
        Clear stabilizer measurement history at gate boundaries.
        
        This is necessary when a gate changes the interpretation of stabilizers,
        such as a Hadamard which swaps X and Z stabilizers. After clearing,
        the next stabilizer round will establish a new baseline rather than
        creating invalid time-like detectors.
        
        Parameters
        ----------
        block_name : str, optional
            If provided, only clear history for this block.
            If None, clear all history.
        swap_xz : bool
            If True, swap X and Z keys in the history instead of clearing.
            This preserves continuity when X stabilizers become Z stabilizers.
        """
        if swap_xz:
            # Swap X and Z stabilizer entries instead of clearing
            new_meas = {}
            new_init = set()
            for key, val in self.last_stabilizer_meas.items():
                bname, stab_type, stab_idx = key
                if block_name is None or bname == block_name:
                    # Swap x <-> z
                    new_type = "z" if stab_type == "x" else "x" if stab_type == "z" else stab_type
                    new_key = (bname, new_type, stab_idx)
                    new_meas[new_key] = val
                else:
                    new_meas[key] = val
            
            for key in self.stabilizer_initialized:
                bname, stab_type, stab_idx = key
                if block_name is None or bname == block_name:
                    new_type = "z" if stab_type == "x" else "x" if stab_type == "z" else stab_type
                    new_init.add((bname, new_type, stab_idx))
                else:
                    new_init.add(key)
            
            self.last_stabilizer_meas = new_meas
            self.stabilizer_initialized = new_init
        else:
            if block_name is None:
                self.last_stabilizer_meas.clear()
                self.stabilizer_initialized.clear()
            else:
                keys_to_remove = [k for k in self.last_stabilizer_meas if k[0] == block_name]
                for k in keys_to_remove:
                    del self.last_stabilizer_meas[k]
                self.stabilizer_initialized = {
                    k for k in self.stabilizer_initialized if k[0] != block_name
                }


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
        
        # Cache metadata - try both _metadata and metadata attributes
        self._meta = getattr(code, '_metadata', None) or getattr(code, 'metadata', None) or {}
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
        """
        raise NotImplementedError("Subclass must implement emit_round")
    
    def emit_rounds(
        self,
        circuit: stim.Circuit,
        num_rounds: int,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
    ) -> None:
        """Emit multiple stabilizer measurement rounds."""
        for _ in range(num_rounds):
            self.emit_round(circuit, stab_type, emit_detectors=True)
    
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
    
    def _emit_shift_coords(self, circuit: stim.Circuit) -> None:
        """Emit SHIFT_COORDS for time advancement."""
        if self._data_coords:
            circuit.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])


class CSSStabilizerRoundBuilder(BaseStabilizerRoundBuilder):
    """
    Stabilizer round builder for CSS codes.
    
    Handles separate X and Z stabilizer measurements with proper scheduling
    and first-round detector logic. This is the main builder for CSS memory
    experiments and FT gadget experiments.
    
    Key features:
    - First-round X detectors only for X-basis memory (|+⟩ is X eigenstate)
    - First-round Z detectors only for Z-basis memory (|0⟩ is Z eigenstate)
    - Geometric scheduling when metadata provides coordinates
    - Graph-coloring fallback for conflict-free CNOT layers
    - SHIFT_COORDS for time advancement
    - Interleaved X/Z scheduling for surface codes
    """
    
    def __init__(
        self,
        code: "CSSCode",
        ctx: DetectorContext,
        block_name: str = "main",
        data_offset: int = 0,
        ancilla_offset: Optional[int] = None,
        measurement_basis: str = "Z",
        enable_metachecks: bool = False,
        single_shot_metachecks: bool = True,
    ):
        super().__init__(code, ctx, block_name, data_offset, ancilla_offset, measurement_basis)
        
        # Get code's FT gadget config for code-specific behavior
        self._ft_config = code.get_ft_gadget_config()
        
        # Single-shot metacheck mode: emit spatial constraint detectors from round 1
        # This allows metachecks to detect measurement errors within a single round,
        # rather than requiring time-like comparison across rounds.
        self._single_shot_metachecks = single_shot_metachecks
        
        # Cache stabilizer info - CSS codes define hx/hz as @property returning numpy arrays
        _hx_raw = getattr(code, 'hx', None)
        _hz_raw = getattr(code, 'hz', None)
        
        # Only use if it's actually a numpy array (has .shape attribute)
        self._hx = _hx_raw if _hx_raw is not None and hasattr(_hx_raw, 'shape') else None
        self._hz = _hz_raw if _hz_raw is not None and hasattr(_hz_raw, 'shape') else None
        self._n_x = self._hx.shape[0] if self._hx is not None and self._hx.size > 0 else 0
        self._n_z = self._hz.shape[0] if self._hz is not None and self._hz.size > 0 else 0
        
        # Detect self-dual codes (color codes where hx = hz)
        # For self-dual codes, |0⟩^⊗n is only deterministic for Z stabilizers,
        # and |+⟩^⊗n is only deterministic for X stabilizers.
        # We cannot have both first-round X and Z detectors in the same basis.
        self._is_self_dual = (
            self._hx is not None and self._hz is not None and
            self._hx.shape == self._hz.shape and
            np.array_equal(self._hx, self._hz)
        )
        
        # Get stabilizer coordinates - try code's hooks first, then metadata
        self._x_stab_coords = self._get_stabilizer_coords('x')
        self._z_stab_coords = self._get_stabilizer_coords('z')
        
        # Get schedules - try code's hooks first, then metadata
        self._x_schedule = self._get_schedule('x')
        self._z_schedule = self._get_schedule('z')
        
        # Track last measurements for each stabilizer (for time-like detectors)
        self._last_x_meas: List[Optional[int]] = [None] * self._n_x
        self._last_z_meas: List[Optional[int]] = [None] * self._n_z
        
        # Flag to skip first-round detectors (used after teleportation)
        self._skip_first_round: bool = False
        
        # Flag to track if stabilizer types have been swapped (after Hadamard)
        # When True, physical X ancillas measure logical Z, and vice versa
        self._stabilizer_swapped: bool = False
        
        # Metacheck support for 4D/5D codes (single-shot error correction)
        # Use code's config setting, but can be overridden by parameter
        self._enable_metachecks = enable_metachecks or self._ft_config.enable_metachecks
        _meta_x_raw = getattr(code, 'meta_x', None)
        _meta_z_raw = getattr(code, 'meta_z', None)
        self._meta_x = _meta_x_raw if _meta_x_raw is not None and hasattr(_meta_x_raw, 'shape') else None
        self._meta_z = _meta_z_raw if _meta_z_raw is not None and hasattr(_meta_z_raw, 'shape') else None
        self._n_meta_x = self._meta_x.shape[0] if self._meta_x is not None and self._meta_x.size > 0 else 0
        self._n_meta_z = self._meta_z.shape[0] if self._meta_z is not None and self._meta_z.size > 0 else 0
        
        # Track metacheck measurements (for detecting syndrome measurement errors)
        self._last_meta_x_meas: List[Optional[int]] = [None] * self._n_meta_x
        self._last_meta_z_meas: List[Optional[int]] = [None] * self._n_meta_z
    
    def _get_stabilizer_coords(self, basis: str) -> List[Tuple[float, ...]]:
        """Get stabilizer coordinates using code hooks or metadata.
        
        Parameters
        ----------
        basis : str
            'x' or 'z' for X-type or Z-type stabilizers.
            
        Returns
        -------
        List[Tuple[float, ...]]
            Coordinates for each stabilizer of the given type.
        """
        # Try code's hook method first (allows code-specific customization)
        if hasattr(self.code, f'get_{basis}_stabilizer_coords'):
            coords = getattr(self.code, f'get_{basis}_stabilizer_coords')()
            if coords is not None:
                return coords
        
        # Fall back to metadata
        key = f'{basis}_stab_coords'
        return self._meta.get(key, [])
    
    def _get_schedule(self, basis: str) -> Optional[List[Tuple[float, float]]]:
        """Get CNOT schedule using code hooks or metadata.
        
        Parameters
        ----------
        basis : str
            'x' or 'z' for X-type or Z-type stabilizers.
            
        Returns
        -------
        Optional[List[Tuple[float, float]]]
            Schedule offsets or None for default scheduling.
        """
        # Try code's hook method first
        if hasattr(self.code, 'get_stabilizer_schedule'):
            schedule = self.code.get_stabilizer_schedule(basis)
            if schedule is not None:
                return schedule
        
        # Fall back to metadata
        key = f'{basis}_schedule'
        return self._meta.get(key)
    
    def reset_stabilizer_history(self, swap_xz: bool = False, skip_first_round: bool = False) -> None:
        """
        Reset the builder's internal stabilizer measurement history.
        
        This is necessary after gates like Hadamard that change the stabilizer
        basis. After a Hadamard, the physical stabilizer measurements have 
        different logical interpretations, so we cannot compare post-gate 
        measurements with pre-gate measurements.
        
        Parameters
        ----------
        swap_xz : bool
            If True, also swap the effective measurement basis and track that
            stabilizer types are swapped. This is needed after Hadamard because:
            - Before H: |0⟩ is Z eigenstate → first-round Z detectors OK
            - After H: |+⟩ is X eigenstate → first-round X detectors OK
            - Physical X ancillas now measure logical Z (and vice versa)
        skip_first_round : bool
            If True, skip emitting first-round detectors entirely. This is needed
            for teleportation where the block's stabilizers are entangled with
            Bell measurements and not independently deterministic.
            
        Note
        ----
        We always clear the history (not swap) because hx and hz are different
        physical operators. But we DO swap the measurement_basis because the
        logical state has changed basis.
        
        For teleportation, use skip_first_round=True because the block's stabilizers
        are entangled with Bell measurements and not independently deterministic.
        """
        # Always clear measurement history - establish new baseline
        self._last_x_meas = [None] * self._n_x
        self._last_z_meas = [None] * self._n_z
        self._last_meta_x_meas = [None] * self._n_meta_x
        self._last_meta_z_meas = [None] * self._n_meta_z
        
        # Swap effective measurement basis if requested (for Hadamard)
        if swap_xz:
            self.measurement_basis = "X" if self.measurement_basis == "Z" else "Z"
            # Toggle the stabilizer swapped flag
            self._stabilizer_swapped = not self._stabilizer_swapped
        
        # Set flag to skip first-round detectors (for teleportation)
        self._skip_first_round = skip_first_round

    @property
    def x_ancillas(self) -> List[int]:
        """Global indices of X stabilizer ancillas."""
        return list(range(self.ancilla_offset, self.ancilla_offset + self._n_x))
    
    @property
    def z_ancillas(self) -> List[int]:
        """Global indices of Z stabilizer ancillas."""
        return list(range(
            self.ancilla_offset + self._n_x,
            self.ancilla_offset + self._n_x + self._n_z
        ))
    
    @property
    def meta_x_ancillas(self) -> List[int]:
        """Global indices of meta-X check ancillas (check Z syndrome)."""
        if self._n_meta_x == 0:
            return []
        base = self.ancilla_offset + self._n_x + self._n_z
        return list(range(base, base + self._n_meta_x))
    
    @property
    def meta_z_ancillas(self) -> List[int]:
        """Global indices of meta-Z check ancillas (check X syndrome)."""
        if self._n_meta_z == 0:
            return []
        base = self.ancilla_offset + self._n_x + self._n_z + self._n_meta_x
        return list(range(base, base + self._n_meta_z))
    
    @property
    def has_metachecks(self) -> bool:
        """Whether this builder has metacheck support enabled."""
        return self._enable_metachecks and (self._n_meta_x > 0 or self._n_meta_z > 0)
    
    @property
    def total_qubits(self) -> int:
        """Total qubits used by this block (data + ancillas + metacheck ancillas)."""
        return self.code.n + self._n_x + self._n_z + self._n_meta_x + self._n_meta_z
    
    def emit_qubit_coords(self, circuit: stim.Circuit) -> None:
        """Emit QUBIT_COORDS for all qubits in this block."""
        # Data qubits
        for local_idx, coord in enumerate(self._data_coords):
            if len(coord) >= 2:
                global_idx = self.data_offset + local_idx
                circuit.append("QUBIT_COORDS", [global_idx], [float(coord[0]), float(coord[1])])
        
        # X ancillas
        for local_idx, coord in enumerate(self._x_stab_coords):
            if len(coord) >= 2 and local_idx < self._n_x:
                global_idx = self.ancilla_offset + local_idx
                circuit.append("QUBIT_COORDS", [global_idx], [float(coord[0]), float(coord[1])])
        
        # Z ancillas
        for local_idx, coord in enumerate(self._z_stab_coords):
            if len(coord) >= 2 and local_idx < self._n_z:
                global_idx = self.ancilla_offset + self._n_x + local_idx
                circuit.append("QUBIT_COORDS", [global_idx], [float(coord[0]), float(coord[1])])
    
    def emit_reset_all(self, circuit: stim.Circuit) -> None:
        """Reset all data, ancilla, and metacheck ancilla qubits."""
        all_qubits = self.data_qubits + self.x_ancillas + self.z_ancillas
        if self._enable_metachecks:
            all_qubits = all_qubits + self.meta_x_ancillas + self.meta_z_ancillas
        if all_qubits:
            circuit.append("R", all_qubits)
    
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
        # Get code interface
        code = self.code
        
        # Determine which qubits need preparation based on state
        if state in ("0", "1"):
            # Z-basis eigenstate
            # |0⟩_L: prepare all data in |0⟩ (already done by reset)
            # |1⟩_L: apply logical X
            if state == "1" and hasattr(code, 'logical_x_support'):
                support = code.logical_x_support(logical_idx)
                for q in support:
                    circuit.append("X", [self.data_offset + q])
        
        elif state in ("+", "-"):
            # X-basis eigenstate
            # For CSS codes: |+⟩_L = H^⊗n|0⟩_L (uniform superposition over Z code space)
            # Apply H to ALL data qubits to prepare |+⟩^⊗n which is +1 eigenstate of all X stabilizers
            # For |-⟩_L, additionally apply Z to the logical X support to get the -1 eigenvalue
            if state == "-" and hasattr(code, 'logical_x_support'):
                support = code.logical_x_support(logical_idx)
                for q in support:
                    circuit.append("Z", [self.data_offset + q])
            # Apply H to all data qubits
            circuit.append("H", self.data_qubits)
        
        circuit.append("TICK")
    
    def _can_interleave(self) -> bool:
        """Check if interleaved X/Z scheduling is possible.
        
        Interleaved scheduling requires:
        1. Both X and Z stabilizers present
        2. Both have geometric schedules
        3. Schedules have the same length (so phases can be paired)
        """
        if self._n_x == 0 or self._n_z == 0:
            return False
        if not self._use_geometric_x() or not self._use_geometric_z():
            return False
        if self._x_schedule is None or self._z_schedule is None:
            return False
        # Schedules must have same number of phases for pairing
        return len(self._x_schedule) == len(self._z_schedule)
    
    def emit_round(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_detectors: bool = True,
        emit_metachecks: bool = False,
    ) -> None:
        """
        Emit one complete stabilizer measurement round.
        
        Uses interleaved X/Z scheduling when both geometric schedules are
        available with matching phases. This reduces circuit depth and
        improves error correction performance by having X and Z CNOTs
        happen in parallel within each TICK layer.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        stab_type : StabilizerBasis
            Which stabilizers to measure.
        emit_detectors : bool
            Whether to emit time-like detectors.
        emit_metachecks : bool
            Whether to emit metacheck detectors for single-shot correction.
            Only applies if the code has metachecks (4D/5D surface codes).
        """
        # Track measurement indices for metachecks
        x_meas_start = self.ctx.measurement_index
        z_meas_start = x_meas_start + self._n_x  # Z measurements follow X
        
        # Use interleaved scheduling for BOTH when possible (surface codes, etc.)
        if stab_type == StabilizerBasis.BOTH and self._can_interleave():
            self._emit_interleaved_round(circuit, emit_detectors)
        else:
            # Sequential fallback for single-type or non-geometric codes
            if stab_type in (StabilizerBasis.X, StabilizerBasis.BOTH):
                self._emit_x_round(circuit, emit_detectors)
            
            if stab_type in (StabilizerBasis.Z, StabilizerBasis.BOTH):
                self._emit_z_round(circuit, emit_detectors)
        
        # Emit metachecks if enabled and available
        if emit_metachecks and self.has_metachecks and stab_type == StabilizerBasis.BOTH:
            self._emit_metacheck_round(
                circuit, x_meas_start, z_meas_start, emit_detectors
            )
        
        # Advance time and emit SHIFT_COORDS for visualization
        self.ctx.advance_time()
        self._emit_shift_coords(circuit)
        self._round_number += 1
        
        # Clear skip_first_round flag after the first round completes
        # Subsequent rounds should emit normal time-like detectors
        self._skip_first_round = False
    
    def _emit_interleaved_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """Emit interleaved X/Z stabilizer measurements.
        
        This is the optimal scheduling for surface codes and similar CSS codes
        where X and Z stabilizers have matching geometric schedules. X and Z
        CNOTs happen in parallel within each phase, reducing circuit depth.
        
        Circuit structure (normal mode):
        1. H on X ancillas (prepare in |+⟩)
        2. For each phase: X and Z CNOTs together in same TICK
        3. H on X ancillas (rotate back to computational basis)
        4. Measure all ancillas
        
        After Hadamard (_stabilizer_swapped=True):
        - X ancillas now measure Z-type stabilizers → no H gates, data controls
        - Z ancillas now measure X-type stabilizers → H gates, ancilla controls
        """
        x_anc = self.x_ancillas
        z_anc = self.z_ancillas
        
        if self._stabilizer_swapped:
            # After Hadamard: swap the circuit patterns
            # Z ancillas measure X-type → H-CNOT(anc→data)-H-M
            # X ancillas measure Z-type → CNOT(data→anc)-M
            
            # Step 1: Prepare Z ancillas with H (they now measure X-type)
            circuit.append("H", z_anc)
            circuit.append("TICK")
            
            # Step 2: Interleaved CNOT phases - swapped control/target
            for phase_idx, ((dx_x, dy_x), (dx_z, dy_z)) in enumerate(
                zip(self._x_schedule, self._z_schedule)
            ):
                if phase_idx > 0:
                    circuit.append("TICK")
                
                # X CNOTs: now measuring Z-type → data controls ancilla
                for s_idx, (sx, sy) in enumerate(self._x_stab_coords):
                    if s_idx >= len(x_anc):
                        continue
                    anc = x_anc[s_idx]
                    nbr = (float(sx) + dx_x, float(sy) + dy_x)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CNOT", [dq, anc])  # data controls ancilla
                
                # Z CNOTs: now measuring X-type → ancilla controls data
                for s_idx, (sx, sy) in enumerate(self._z_stab_coords):
                    if s_idx >= len(z_anc):
                        continue
                    anc = z_anc[s_idx]
                    nbr = (float(sx) + dx_z, float(sy) + dy_z)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CNOT", [anc, dq])  # ancilla controls data
            
            circuit.append("TICK")
            
            # Step 3: Final H on Z ancillas (they measure X-type)
            circuit.append("H", z_anc)
            circuit.append("TICK")
            
        else:
            # Normal mode: standard surface code interleaving
            # Step 1: Prepare X ancillas with H
            circuit.append("H", x_anc)
            circuit.append("TICK")
            
            # Step 2: Interleaved CNOT phases - X and Z in parallel
            for phase_idx, ((dx_x, dy_x), (dx_z, dy_z)) in enumerate(
                zip(self._x_schedule, self._z_schedule)
            ):
                if phase_idx > 0:
                    circuit.append("TICK")
                
                # X CNOTs for this phase
                # For X stabilizers: H-CNOT-H pattern means ancilla is control
                # (H converts Z-basis measurement to X-basis parity check)
                for s_idx, (sx, sy) in enumerate(self._x_stab_coords):
                    if s_idx >= len(x_anc):
                        continue
                    anc = x_anc[s_idx]
                    nbr = (float(sx) + dx_x, float(sy) + dy_x)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CNOT", [anc, dq])  # ancilla controls data
                
                # Z CNOTs for this phase (SAME TICK - parallel with X)
                # For Z stabilizers: data qubits are control (standard parity check)
                for s_idx, (sx, sy) in enumerate(self._z_stab_coords):
                    if s_idx >= len(z_anc):
                        continue
                    anc = z_anc[s_idx]
                    nbr = (float(sx) + dx_z, float(sy) + dy_z)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CNOT", [dq, anc])  # data controls ancilla
            
            # Separate final CNOT layer from H gates for proper DEM error granularity
            circuit.append("TICK")
            
            # Step 3: Final H on X ancillas
            circuit.append("H", x_anc)
            circuit.append("TICK")
        
        # Step 4: Measure all ancillas
        # X ancillas first, then Z ancillas
        x_meas_start = self.ctx.add_measurement(self._n_x)
        circuit.append("MR", x_anc)
        
        z_meas_start = self.ctx.add_measurement(self._n_z)
        circuit.append("MR", z_anc)
        
        # Emit detectors - after Hadamard, first-round logic changes
        # X ancillas now measure Z-type → first-round OK if basis == "Z"
        # Z ancillas now measure X-type → first-round OK if basis == "X"
        x_effective_basis = "Z" if self._stabilizer_swapped else "X"
        z_effective_basis = "X" if self._stabilizer_swapped else "Z"
        
        # Emit detectors for X stabilizers
        if emit_detectors:
            for s_idx in range(self._n_x):
                cur_meas = x_meas_start + s_idx
                prev_meas = self._last_x_meas[s_idx]
                
                if prev_meas is None:
                    # First round: only emit if measurement_basis matches effective type
                    # For self-dual codes (hx=hz), X first-round only if measurement_basis=="X"
                    if self._is_self_dual:
                        should_emit = not self._skip_first_round and self.measurement_basis == "X"
                    else:
                        should_emit = not self._skip_first_round and self.measurement_basis == x_effective_basis
                    if should_emit:
                        coord = self._get_stab_coord("x", s_idx)
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                else:
                    coord = self._get_stab_coord("x", s_idx)
                    self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
                
                self._last_x_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(self.block_name, "x", s_idx, cur_meas)
        else:
            for s_idx in range(self._n_x):
                cur_meas = x_meas_start + s_idx
                self._last_x_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(self.block_name, "x", s_idx, cur_meas)
        
        # Emit detectors for Z stabilizers
        if emit_detectors:
            for s_idx in range(self._n_z):
                cur_meas = z_meas_start + s_idx
                prev_meas = self._last_z_meas[s_idx]
                
                if prev_meas is None:
                    # First round: only emit if measurement_basis matches effective type
                    # For self-dual codes (hx=hz), Z first-round only if measurement_basis=="Z"
                    if self._is_self_dual:
                        should_emit = not self._skip_first_round and self.measurement_basis == "Z"
                    else:
                        should_emit = not self._skip_first_round and self.measurement_basis == z_effective_basis
                    if should_emit:
                        coord = self._get_stab_coord("z", s_idx)
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                else:
                    coord = self._get_stab_coord("z", s_idx)
                    self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
                
                self._last_z_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(self.block_name, "z", s_idx, cur_meas)
        else:
            for s_idx in range(self._n_z):
                cur_meas = z_meas_start + s_idx
                self._last_z_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(self.block_name, "z", s_idx, cur_meas)
        
        # Add final TICK after measurements to separate from next round
        circuit.append("TICK")

    def _emit_x_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """Emit X stabilizer measurements.
        
        After Hadamard (when _stabilizer_swapped=True), the X ancillas measure
        what are now Z stabilizers, so we use a Z-style circuit (no H gates).
        """
        if self._hx is None or self._n_x == 0:
            return
        
        x_anc = self.x_ancillas
        
        if self._stabilizer_swapped:
            # After Hadamard: X ancillas measure Z-type stabilizers
            # Use Z syndrome circuit: CNOT - M (no H gates)
            # But we still use _hx matrix because that describes the qubit support
            if self._use_geometric_x():
                self._emit_geometric_cnots(circuit, "x")
            else:
                # Use is_x_type=False to use CZ-like CNOT pattern (control on ancilla)
                self._emit_graph_coloring_cnots(circuit, self._hx, self.data_qubits, x_anc, is_x_type=False)
            
            circuit.append("TICK")
        else:
            # Normal: X ancillas measure X-type stabilizers
            # Use X syndrome circuit: H - CNOT - H - M
            circuit.append("H", x_anc)
            circuit.append("TICK")
            
            if self._use_geometric_x():
                self._emit_geometric_cnots(circuit, "x")
            else:
                self._emit_graph_coloring_cnots(circuit, self._hx, self.data_qubits, x_anc, is_x_type=True)
            
            circuit.append("TICK")
            circuit.append("H", x_anc)
            circuit.append("TICK")
        
        # Measure X ancillas
        meas_start = self.ctx.add_measurement(self._n_x)
        circuit.append("MR", x_anc)
        
        # Time-like detectors with proper first-round logic
        # After swap: X ancillas measure Z, so first-round OK if measurement_basis == "Z"
        # Before swap: X ancillas measure X, so first-round OK if measurement_basis == "X"
        effective_first_round_basis = "Z" if self._stabilizer_swapped else "X"
        
        if emit_detectors:
            for s_idx in range(self._n_x):
                cur_meas = meas_start + s_idx
                prev_meas = self._last_x_meas[s_idx]
                
                if prev_meas is None:
                    # First round: only create detector if appropriate
                    # For self-dual codes (hx=hz, e.g. color codes), |0⟩^⊗n is only
                    # deterministic for Z stabilizers, and |+⟩^⊗n only for X.
                    # So for self-dual: X first-round only if measurement_basis=="X"
                    if self._is_self_dual:
                        should_emit = (
                            not self._skip_first_round and
                            self._ft_config.first_round_x_detectors and
                            self.measurement_basis == "X"
                        )
                    else:
                        should_emit = (
                            not self._skip_first_round and
                            self._ft_config.first_round_x_detectors and
                            self.measurement_basis == effective_first_round_basis
                        )
                    if should_emit:
                        coord = self._get_stab_coord("x", s_idx)
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                else:
                    coord = self._get_stab_coord("x", s_idx)
                    self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
                
                self._last_x_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "x", s_idx, cur_meas
                )
        else:
            for s_idx in range(self._n_x):
                cur_meas = meas_start + s_idx
                self._last_x_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "x", s_idx, cur_meas
                )
    
    def _emit_z_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """Emit Z stabilizer measurements.
        
        After Hadamard (when _stabilizer_swapped=True), the Z ancillas measure
        what are now X stabilizers, so we use an X-style circuit (with H gates).
        """
        if self._hz is None or self._n_z == 0:
            return
        
        z_anc = self.z_ancillas
        
        if self._stabilizer_swapped:
            # After Hadamard: Z ancillas measure X-type stabilizers  
            # Use X syndrome circuit: H - CNOT - H - M
            circuit.append("H", z_anc)
            circuit.append("TICK")
            
            if self._use_geometric_z():
                self._emit_geometric_cnots(circuit, "z")
            else:
                # Use is_x_type=True to use X-style CNOT pattern (control on data)
                self._emit_graph_coloring_cnots(circuit, self._hz, self.data_qubits, z_anc, is_x_type=True)
            
            circuit.append("TICK")
            circuit.append("H", z_anc)
            circuit.append("TICK")
        else:
            # Normal: Z ancillas measure Z-type stabilizers
            # Use Z syndrome circuit: CNOT - M (no H gates)
            if self._use_geometric_z():
                self._emit_geometric_cnots(circuit, "z")
            else:
                self._emit_graph_coloring_cnots(circuit, self._hz, self.data_qubits, z_anc, is_x_type=False)
            
            circuit.append("TICK")
        
        # Measure Z ancillas
        meas_start = self.ctx.add_measurement(self._n_z)
        circuit.append("MR", z_anc)
        
        # Time-like detectors with proper first-round logic
        # After swap: Z ancillas measure X, so first-round OK if measurement_basis == "X"
        # Before swap: Z ancillas measure Z, so first-round OK if measurement_basis == "Z"
        effective_first_round_basis = "X" if self._stabilizer_swapped else "Z"
        
        if emit_detectors:
            for s_idx in range(self._n_z):
                cur_meas = meas_start + s_idx
                prev_meas = self._last_z_meas[s_idx]
                
                if prev_meas is None:
                    # First round: only create detector if appropriate
                    # For self-dual codes (hx=hz, e.g. color codes), |0⟩^⊗n is only
                    # deterministic for Z stabilizers, and |+⟩^⊗n only for X.
                    # So for self-dual: Z first-round only if measurement_basis=="Z"
                    if self._is_self_dual:
                        should_emit = (
                            not self._skip_first_round and
                            self._ft_config.first_round_z_detectors and
                            self.measurement_basis == "Z"
                        )
                    else:
                        should_emit = (
                            not self._skip_first_round and
                            self._ft_config.first_round_z_detectors and
                            self.measurement_basis == effective_first_round_basis
                        )
                    if should_emit:
                        coord = self._get_stab_coord("z", s_idx)
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                else:
                    coord = self._get_stab_coord("z", s_idx)
                    self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
                
                self._last_z_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "z", s_idx, cur_meas
                )
        else:
            for s_idx in range(self._n_z):
                cur_meas = meas_start + s_idx
                self._last_z_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "z", s_idx, cur_meas
                )
    
    def _emit_metacheck_round(
        self,
        circuit: stim.Circuit,
        x_syndrome_meas_start: int,
        z_syndrome_meas_start: int,
        emit_detectors: bool = True,
    ) -> None:
        """
        Emit metacheck measurements for single-shot error correction.
        
        Metachecks verify the consistency of syndrome measurements:
        - meta_x: checks the parity of Z syndrome bits (meta_x @ hz = 0)
        - meta_z: checks the parity of X syndrome bits (meta_z @ hx = 0)
        
        For 4D surface codes (5-chain), both meta_x and meta_z exist.
        For 3D toric codes (4-chain), only one type exists depending on qubit grade.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        x_syndrome_meas_start : int
            Starting measurement index for X syndrome bits (needed for meta_z).
        z_syndrome_meas_start : int
            Starting measurement index for Z syndrome bits (needed for meta_x).
        emit_detectors : bool
            Whether to emit metacheck detectors.
        
        Note
        ----
        Metachecks measure parities of SYNDROME BITS (ancilla measurements),
        not data qubits. This is done by tracking which syndrome measurements
        feed into each metacheck, then emitting detectors that compare
        the metacheck result with the expected parity from syndrome bits.
        
        IMPORTANT: Metacheck spatial detectors (round 1) are linearly dependent
        on regular stabilizer detectors due to the chain condition meta @ H = 0.
        Time-like metachecks (round 2+) provide useful measurement error detection.
        With only 1 round, metachecks provide no additional information and may
        actually hurt decoder performance by adding redundant hyperedges to the DEM.
        """
        if not self._enable_metachecks:
            return
        
        # Meta-X checks (check Z syndrome parity)
        if self._n_meta_x > 0 and self._meta_x is not None:
            self._emit_meta_x_checks(
                circuit, z_syndrome_meas_start, emit_detectors
            )
        
        # Meta-Z checks (check X syndrome parity)
        if self._n_meta_z > 0 and self._meta_z is not None:
            self._emit_meta_z_checks(
                circuit, x_syndrome_meas_start, emit_detectors
            )
    
    def _emit_meta_x_checks(
        self,
        circuit: stim.Circuit,
        z_syndrome_meas_start: int,
        emit_detectors: bool,
    ) -> None:
        """
        Emit meta-X checks that verify Z syndrome consistency.
        
        Each meta-X check computes: ⊕_{j where meta_x[i,j]=1} z_syndrome[j]
        
        Two modes of operation:
        
        1. SINGLE-SHOT MODE (default, self._single_shot_metachecks=True):
           - For Z-basis memory: Z syndrome should be 0 (|0⟩ is Z eigenstate)
           - First round: emit SPATIAL detector on current syndrome bits only
             This fires if XOR(syndrome_bits) ≠ 0 (constraint violation)
           - Later rounds: emit TIME-LIKE detector comparing rounds
           - Benefit: Can detect measurement errors from round 1
        
        2. TIME-LIKE ONLY MODE (self._single_shot_metachecks=False):
           - First round: skip (no baseline)
           - Later rounds: compare XOR(prev) with XOR(cur)
           - This is redundant with regular stabilizer detectors
        
        The chain condition meta_x @ hz = 0 ensures that valid syndromes
        (from data errors) have even parity under each metacheck row.
        Measurement errors break this constraint.
        
        NOTE: First-round spatial metacheck detectors are linearly dependent on
        regular stabilizer detectors (due to chain condition meta_x @ hz = 0).
        They provide no additional error-correcting information and can hurt
        decoder performance by adding redundant hyperedges to the DEM.
        We now ONLY emit time-like metacheck detectors (round 2+).
        """
        # Build current round's syndrome indices for each metacheck
        cur_syndrome_indices_list = []
        for meta_idx in range(self._n_meta_x):
            row = self._meta_x[meta_idx]
            cur_indices = [
                z_syndrome_meas_start + j
                for j in range(len(row)) if row[j] != 0
            ]
            cur_syndrome_indices_list.append(cur_indices)
        
        if emit_detectors:
            for meta_idx in range(self._n_meta_x):
                cur_indices = cur_syndrome_indices_list[meta_idx]
                prev_indices = self._last_meta_x_meas[meta_idx]
                
                if not cur_indices:
                    continue
                
                # Compute detector coordinate (use centroid of covered syndrome ancillas)
                coord = self._get_metacheck_coord("x", meta_idx)
                
                if prev_indices is None:
                    # First round: emit spatial metacheck detector if in single-shot mode
                    # These check the constraint that valid syndrome satisfies metacheck.
                    # For standard decoders, this is redundant (chain condition meta @ H.T = 0)
                    # but for syndrome-repair decoders like SingleShotDecoder, they enable
                    # detection and correction of measurement errors.
                    # Account for stabilizer swap after Hadamard gadgets
                    effective_z_basis = "X" if self._stabilizer_swapped else "Z"
                    if self._single_shot_metachecks and self.measurement_basis == effective_z_basis:
                        # Only emit for Z-basis (meta_x checks Z syndrome)
                        self.ctx.emit_detector(circuit, cur_indices, coord)
                else:
                    # TIME-LIKE DETECTOR: Compare XOR(prev) with XOR(cur)
                    # XOR(prev_indices) XOR XOR(cur_indices) = 0 if no measurement errors
                    all_indices = prev_indices + cur_indices
                    self.ctx.emit_detector(circuit, all_indices, coord)
        
        # Update tracking for next round
        for meta_idx in range(self._n_meta_x):
            self._last_meta_x_meas[meta_idx] = cur_syndrome_indices_list[meta_idx]
    
    def _emit_meta_z_checks(
        self,
        circuit: stim.Circuit,
        x_syndrome_meas_start: int,
        emit_detectors: bool,
    ) -> None:
        """
        Emit meta-Z checks that verify X syndrome consistency.
        
        Each meta-Z check computes: ⊕_{j where meta_z[i,j]=1} x_syndrome[j]
        
        Two modes of operation:
        
        1. SINGLE-SHOT MODE (default, self._single_shot_metachecks=True):
           - For X-basis memory: X syndrome should be 0 (|+⟩ is X eigenstate)
           - First round: emit SPATIAL detector on current syndrome bits only
             This fires if XOR(syndrome_bits) ≠ 0 (constraint violation)
           - Later rounds: emit TIME-LIKE detector comparing rounds
           - Benefit: Can detect measurement errors from round 1
        
        2. TIME-LIKE ONLY MODE (self._single_shot_metachecks=False):
           - First round: skip (no baseline)
           - Later rounds: compare XOR(prev) with XOR(cur)
           - This is redundant with regular stabilizer detectors
        
        The chain condition meta_z @ hx = 0 ensures that valid syndromes
        (from data errors) have even parity under each metacheck row.
        Measurement errors break this constraint.
        
        NOTE: First-round spatial metacheck detectors are linearly dependent on
        regular stabilizer detectors (due to chain condition meta_z @ hx = 0).
        They provide no additional error-correcting information and can hurt
        decoder performance by adding redundant hyperedges to the DEM.
        We now ONLY emit time-like metacheck detectors (round 2+).
        """
        # Build current round's syndrome indices for each metacheck
        cur_syndrome_indices_list = []
        for meta_idx in range(self._n_meta_z):
            row = self._meta_z[meta_idx]
            cur_indices = [
                x_syndrome_meas_start + j
                for j in range(len(row)) if row[j] != 0
            ]
            cur_syndrome_indices_list.append(cur_indices)
        
        if emit_detectors:
            for meta_idx in range(self._n_meta_z):
                cur_indices = cur_syndrome_indices_list[meta_idx]
                prev_indices = self._last_meta_z_meas[meta_idx]
                
                if not cur_indices:
                    continue
                
                # Compute detector coordinate (use centroid of covered syndrome ancillas)
                coord = self._get_metacheck_coord("z", meta_idx)
                
                if prev_indices is None:
                    # First round: emit spatial metacheck detector if in single-shot mode
                    # These check the constraint that valid syndrome satisfies metacheck.
                    # For standard decoders, this is redundant (chain condition meta @ H.T = 0)
                    # but for syndrome-repair decoders like SingleShotDecoder, they enable
                    # detection and correction of measurement errors.
                    # Account for stabilizer swap after Hadamard gadgets
                    effective_x_basis = "Z" if self._stabilizer_swapped else "X"
                    if self._single_shot_metachecks and self.measurement_basis == effective_x_basis:
                        # Only emit for X-basis (meta_z checks X syndrome)
                        self.ctx.emit_detector(circuit, cur_indices, coord)
                else:
                    # TIME-LIKE DETECTOR: Compare XOR(prev) with XOR(cur)
                    # XOR(prev_indices) XOR XOR(cur_indices) = 0 if no measurement errors
                    all_indices = prev_indices + cur_indices
                    self.ctx.emit_detector(circuit, all_indices, coord)
        
        # Update tracking for next round
        for meta_idx in range(self._n_meta_z):
            self._last_meta_z_meas[meta_idx] = cur_syndrome_indices_list[meta_idx]

    def _use_geometric_x(self) -> bool:
        """Check if geometric scheduling should be used for X stabilizers.
        
        Respects the code's schedule_mode configuration:
        - GRAPH_COLORING: Always returns False (force graph coloring)
        - GEOMETRIC: Returns True if geometric is available (may fail if coords missing)
        - AUTO: Returns True only if ALL connections can be resolved geometrically
        """
        # Check code's schedule mode preference
        mode = self._ft_config.schedule_mode
        if mode == ScheduleMode.GRAPH_COLORING:
            return False
        
        # Check if geometric scheduling is even possible
        if not (
            self._x_schedule is not None and
            self._data_coords and
            len(self._x_stab_coords) == self._n_x
        ):
            if mode == ScheduleMode.GEOMETRIC:
                # Geometric required but not available - will fail
                # (caller can handle this or fall back)
                pass
            return False
        
        # For AUTO mode, validate that ALL schedule lookups will succeed
        return self._validate_geometric_schedule(
            self._x_schedule, self._x_stab_coords, self._hx
        )
    
    def _use_geometric_z(self) -> bool:
        """Check if geometric scheduling should be used for Z stabilizers.
        
        Respects the code's schedule_mode configuration:
        - GRAPH_COLORING: Always returns False (force graph coloring)
        - GEOMETRIC: Returns True if geometric is available (may fail if coords missing)
        - AUTO: Returns True only if ALL connections can be resolved geometrically
        """
        # Check code's schedule mode preference
        mode = self._ft_config.schedule_mode
        if mode == ScheduleMode.GRAPH_COLORING:
            return False
        
        # Check if geometric scheduling is even possible
        if not (
            self._z_schedule is not None and
            self._data_coords and
            len(self._z_stab_coords) == self._n_z
        ):
            if mode == ScheduleMode.GEOMETRIC:
                # Geometric required but not available - will fail
                pass
            return False
        
        # For AUTO mode, validate that ALL schedule lookups will succeed
        return self._validate_geometric_schedule(
            self._z_schedule, self._z_stab_coords, self._hz
        )
    
    def _validate_geometric_schedule(
        self,
        schedule: List[Tuple[float, float]],
        stab_coords: List[Tuple[float, ...]],
        stab_matrix: Optional[np.ndarray],
    ) -> bool:
        """Validate that geometric scheduling will cover all stabilizer-data connections.
        
        For geometric scheduling to be valid, every non-zero entry in the stabilizer
        matrix must have a corresponding coordinate lookup that succeeds. If any
        lookup fails, we should fall back to graph coloring.
        
        Parameters
        ----------
        schedule : List[Tuple[float, float]]
            The geometric schedule offsets (dx, dy) per layer.
        stab_coords : List[Tuple[float, ...]]
            Coordinates of stabilizer ancillas.
        stab_matrix : Optional[np.ndarray]
            The stabilizer matrix (hx or hz) defining which data qubits are involved.
            
        Returns
        -------
        bool
            True if geometric scheduling will cover all connections, False otherwise.
        """
        if stab_matrix is None or stab_matrix.size == 0:
            return False
        
        n_stabs = stab_matrix.shape[0]
        n_data = stab_matrix.shape[1] if stab_matrix.ndim > 1 else 1
        
        # Count how many data qubits each stabilizer should touch
        # and track which ones we can reach via geometric schedule
        for s_idx in range(min(n_stabs, len(stab_coords))):
            sx, sy = stab_coords[s_idx][:2] if len(stab_coords[s_idx]) >= 2 else (0, 0)
            
            # Count expected connections from matrix
            if stab_matrix.ndim == 1:
                expected = int(stab_matrix[s_idx]) if s_idx < len(stab_matrix) else 0
            else:
                expected = int(np.sum(stab_matrix[s_idx] != 0))
            
            # Count reachable connections from schedule
            found = 0
            for dx, dy in schedule:
                nbr = (float(sx) + dx, float(sy) + dy)
                if self._lookup_data_qubit(nbr) is not None:
                    found += 1
            
            # If we can't reach all expected data qubits, geometric fails
            if found < expected:
                return False
        
        return True
    
    def _emit_geometric_cnots(self, circuit: stim.Circuit, stab_type: str) -> None:
        """Emit CNOTs using geometric scheduling.
        
        CNOT direction convention for CSS stabilizer measurement (matches Stim):
        - X-type stabilizers: CNOT(ancilla, data) - ancilla controls data
        - Z-type stabilizers: CNOT(data, ancilla) - data controls ancilla
        
        Why different directions?
        - X-type with H-CNOT(a→d)-H: X on ancilla propagates to data, then H
          rotates back. Ancilla measures X parity of data qubits.
        - Z-type with CNOT(d→a): Z on data propagates to ancilla.
          Ancilla measures Z parity of data qubits.
        
        The key insight is that CNOT propagates:
        - Z from control to target: CNOT|Z⊗I⟩ = |Z⊗Z⟩
        - X from target to control: CNOT|I⊗X⟩ = |X⊗X⟩
        
        For Z-stabilizers: data controls, Z propagates to ancilla
        For X-stabilizers: ancilla controls, X propagates from data to ancilla via target→control
        """
        if stab_type == "x":
            schedule = self._x_schedule
            stab_coords = self._x_stab_coords
            ancillas = self.x_ancillas
            is_x_type = True
        else:
            schedule = self._z_schedule
            stab_coords = self._z_stab_coords
            ancillas = self.z_ancillas
            is_x_type = False
        
        for layer_idx, (dx, dy) in enumerate(schedule):
            if layer_idx > 0:
                circuit.append("TICK")
            for s_idx, (sx, sy) in enumerate(stab_coords):
                if s_idx >= len(ancillas):
                    continue
                anc = ancillas[s_idx]
                nbr = (float(sx) + dx, float(sy) + dy)
                dq = self._lookup_data_qubit(nbr)
                if dq is not None:
                    if is_x_type:
                        # X-type: ancilla controls data (matches Stim convention)
                        circuit.append("CNOT", [anc, dq])
                    else:
                        # Z-type: data controls ancilla
                        circuit.append("CNOT", [dq, anc])
    
    def _emit_graph_coloring_cnots(
        self,
        circuit: stim.Circuit,
        stab_matrix: np.ndarray,
        data_qubits: List[int],
        ancilla_qubits: List[int],
        is_x_type: bool = True,
    ) -> None:
        """Emit CNOTs using greedy graph coloring.
        
        CNOT direction convention for CSS stabilizer measurement (matches Stim):
        - X-type stabilizers (is_x_type=True): CNOT(ancilla, data) - ancilla controls
        - Z-type stabilizers (is_x_type=False): CNOT(data, ancilla) - data controls
        
        Why different directions?
        - X-type with H-CNOT(a→d)-H: ancilla controls, X propagates from data→ancilla
        - Z-type with CNOT(d→a): data controls, Z propagates from data→ancilla
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        stab_matrix : np.ndarray
            Parity check matrix.
        data_qubits : List[int]
            Data qubit indices.
        ancilla_qubits : List[int]
            Ancilla qubit indices.
        is_x_type : bool
            True for X-type stabilizers (ancilla→data), False for Z-type (data→ancilla).
        """
        if stab_matrix is None or stab_matrix.size == 0:
            return
        
        n_stabs, n_data = stab_matrix.shape
        
        # Collect all CNOT pairs with correct direction based on stabilizer type
        all_cnots: List[Tuple[int, int]] = []
        for s_idx in range(min(n_stabs, len(ancilla_qubits))):
            anc = ancilla_qubits[s_idx]
            for d_idx in range(min(n_data, len(data_qubits))):
                if stab_matrix[s_idx, d_idx]:
                    dq = data_qubits[d_idx]
                    if is_x_type:
                        # X-type: ancilla controls data (matches Stim)
                        all_cnots.append((anc, dq))
                    else:
                        # Z-type: data controls ancilla
                        all_cnots.append((dq, anc))
        
        if not all_cnots:
            return
        
        # Use shared graph coloring algorithm
        layers = graph_coloring_cnots(all_cnots)
        
        # Emit layers with TICKs (between layers, not after last)
        for layer_idx, layer in enumerate(layers):
            if layer_idx > 0:
                circuit.append("TICK")
            for ctrl, tgt in layer:
                circuit.append("CNOT", [ctrl, tgt])
    
    def _get_stab_coord(self, stab_type: str, s_idx: int) -> Tuple[float, float, float]:
        """Get detector coordinate for a stabilizer."""
        if stab_type == "x":
            coords = self._x_stab_coords
        else:
            coords = self._z_stab_coords
        
        if s_idx < len(coords):
            x, y = coords[s_idx][:2]
            return (float(x), float(y), self.ctx.current_time)
        return (0.0, 0.0, self.ctx.current_time)
    
    def _get_metacheck_coord(self, meta_type: str, meta_idx: int) -> Tuple[float, float, float, float]:
        """
        Get detector coordinate for a metacheck.
        
        Computes the centroid of the syndrome ancillas covered by this metacheck,
        which helps the decoder correlate metacheck detectors with nearby errors.
        
        Parameters
        ----------
        meta_type : str
            'x' for meta-X check (covers Z stabilizers), 'z' for meta-Z check (covers X stabilizers)
        meta_idx : int
            Index of the metacheck row
            
        Returns
        -------
        Tuple[float, float, float, float]
            (x, y, t, meta_flag) where meta_flag distinguishes from regular detectors
        """
        # Get the metacheck matrix and corresponding stabilizer coordinates
        if meta_type == "x":
            meta_matrix = self._meta_x
            coords = self._z_stab_coords  # meta_x covers Z stabilizers
        else:
            meta_matrix = self._meta_z
            coords = self._x_stab_coords  # meta_z covers X stabilizers
        
        if meta_matrix is None or meta_idx >= meta_matrix.shape[0]:
            return (0.0, 0.0, self.ctx.current_time, 1.0)
        
        # Get which stabilizers this metacheck covers
        row = meta_matrix[meta_idx]
        covered_indices = np.where(row)[0]
        
        if len(covered_indices) == 0 or len(coords) == 0:
            return (0.0, 0.0, self.ctx.current_time, 1.0)
        
        # Compute centroid of covered stabilizer positions
        x_sum, y_sum, count = 0.0, 0.0, 0
        for s_idx in covered_indices:
            if s_idx < len(coords):
                x_sum += coords[s_idx][0]
                y_sum += coords[s_idx][1]
                count += 1
        
        if count > 0:
            return (x_sum / count, y_sum / count, self.ctx.current_time, 1.0)
        return (0.0, 0.0, self.ctx.current_time, 1.0)
    
    def emit_rounds(
        self,
        circuit: stim.Circuit,
        num_rounds: int,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_metachecks: bool = False,
    ) -> None:
        """
        Emit multiple stabilizer measurement rounds.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        num_rounds : int
            Number of rounds.
        stab_type : StabilizerBasis
            Which stabilizers to measure.
        emit_metachecks : bool
            Whether to emit metacheck detectors for single-shot correction.
        """
        for _ in range(num_rounds):
            self.emit_round(circuit, stab_type, emit_detectors=True, emit_metachecks=emit_metachecks)
    
    def emit_final_measurement(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
        logical_idx: int = 0,
    ) -> List[int]:
        """
        Emit final data qubit measurements with space-like detectors.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        basis : str
            Measurement basis ("Z" or "X").
        logical_idx : int
            Which logical qubit to track.
            
        Returns
        -------
        List[int]
            Measurement indices for the logical observable.
        """
        data = self.data_qubits
        n = len(data)
        basis = basis.upper()
        
        # Basis change if needed
        if basis == "X":
            circuit.append("H", data)
        
        # Measure all data qubits (not just logical support)
        meas_start = self.ctx.add_measurement(n)
        circuit.append("M", data)
        
        # Build data measurement lookup
        data_meas = {q: meas_start + i for i, q in enumerate(range(n))}
        
        # Space-like detectors: pair final data measurements with last stabilizer round
        # For Z-basis: use Z stabilizers (hz) and last_z_meas
        # For X-basis: use X stabilizers (hx) and last_x_meas
        if basis == "Z" and self._hz is not None:
            for s_idx in range(self._n_z):
                last_meas = self._last_z_meas[s_idx]
                if last_meas is None:
                    continue
                
                # Get data qubits in this stabilizer
                row = self._hz[s_idx]
                data_idxs = [meas_start + d_idx for d_idx in range(min(n, len(row))) if row[d_idx]]
                
                if data_idxs:
                    recs = data_idxs + [last_meas]
                    # Space-like detectors: get stabilizer coordinate and update time
                    coord = self._get_stab_coord("z", s_idx)
                    # Preserve all dimensions (3D or 4D for color codes)
                    coord = (coord[0], coord[1], self.ctx.current_time) + coord[3:]
                    self.ctx.emit_detector(circuit, recs, coord)
        
        elif basis == "X" and self._hx is not None:
            for s_idx in range(self._n_x):
                last_meas = self._last_x_meas[s_idx]
                if last_meas is None:
                    continue
                
                row = self._hx[s_idx]
                data_idxs = [meas_start + d_idx for d_idx in range(min(n, len(row))) if row[d_idx]]
                
                if data_idxs:
                    recs = data_idxs + [last_meas]
                    # Space-like detectors: get stabilizer coordinate and update time
                    coord = self._get_stab_coord("x", s_idx)
                    # Preserve all dimensions (3D or 4D for color codes)
                    coord = (coord[0], coord[1], self.ctx.current_time) + coord[3:]
                    self.ctx.emit_detector(circuit, recs, coord)
        
        # Compute logical observable measurements
        logical_meas = []
        code = self.code
        
        # Get transformed basis (in case of gates applied in FT experiments)
        effective_basis = self.ctx.get_transformed_basis(logical_idx, basis)
        
        # Get logical operator support
        if effective_basis == "Z":
            logical_support = get_logical_support(code, "Z", logical_idx)
        else:
            logical_support = get_logical_support(code, "X", logical_idx)
        
        if logical_support:
            logical_meas = [meas_start + q for q in logical_support if q < n]
        else:
            # No valid logical operator found - use first qubit as minimal fallback
            # This avoids non-deterministic detectors from XORing all qubits
            import warnings
            warnings.warn(
                f"No valid logical {effective_basis} operator found for {type(code).__name__}. "
                f"Observable will track qubit 0 only - decoding may not work correctly.",
                RuntimeWarning,
                stacklevel=2
            )
            logical_meas = [meas_start]  # Single qubit fallback
        
        # Add to observable accumulator
        self.ctx.add_observable_measurement(logical_idx, logical_meas)
        
        return logical_meas
    
    def emit_space_like_detectors(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
        data_meas_start: Optional[int] = None,
    ) -> None:
        """
        Emit space-like detectors comparing final data measurements with last stabilizer round.
        
        This is called after data qubits have been measured. It compares the
        data qubit measurements (which should match stabilizer eigenvalues)
        with the last stabilizer measurement round.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        basis : str
            Measurement basis ("Z" or "X"). This is the LOGICAL basis - if
            stabilizers have been swapped (after Hadamard), this method
            automatically uses the correct physical stabilizer type.
        data_meas_start : int, optional
            The measurement index where this block's data qubit measurements start.
            If not provided, assumes this block's data was measured last and computes
            the start from ctx.measurement_index - n. For multi-block experiments,
            this parameter should be provided to correctly identify each block's
            data measurements within a combined final measurement.
        """
        n = len(self.data_qubits)
        
        # Get the measurement start for this block's data qubits
        if data_meas_start is not None:
            meas_start = data_meas_start
        else:
            # Fallback: assume this block's data was measured last
            meas_start = self.ctx.measurement_index - n
        
        # Determine the physical stabilizer type to use
        # After Hadamard (swap_xz=True), X↔Z are swapped:
        #   - Logical X basis should use physical Z stabilizers
        #   - Logical Z basis should use physical X stabilizers
        logical_basis = basis.upper()
        if self._stabilizer_swapped:
            # Swap logical to physical mapping
            physical_type = "x" if logical_basis == "Z" else "z"
        else:
            physical_type = "z" if logical_basis == "Z" else "x"
        
        # Use the correct matrix and measurement history based on physical type
        if physical_type == "z" and self._hz is not None:
            for s_idx in range(self._n_z):
                last_meas = self.ctx.last_stabilizer_meas.get(
                    (self.block_name, "z", s_idx)
                )
                if last_meas is None:
                    continue
                
                # Get data qubits in this stabilizer
                support = []
                for d_idx in range(min(n, self._hz.shape[1])):
                    if self._hz[s_idx, d_idx]:
                        support.append(meas_start + d_idx)
                
                if support:
                    coord = self._get_stab_coord("z", s_idx)
                    self.ctx.emit_detector(circuit, [last_meas] + support, coord)
        
        elif physical_type == "x" and self._hx is not None:
            for s_idx in range(self._n_x):
                last_meas = self.ctx.last_stabilizer_meas.get(
                    (self.block_name, "x", s_idx)
                )
                if last_meas is None:
                    continue
                
                support = []
                for d_idx in range(min(n, self._hx.shape[1])):
                    if self._hx[s_idx, d_idx]:
                        support.append(meas_start + d_idx)
                
                if support:
                    coord = self._get_stab_coord("x", s_idx)
                    self.ctx.emit_detector(circuit, [last_meas] + support, coord)


def get_logical_support(code: "Code", basis: str, logical_idx: int = 0) -> List[int]:
    """
    Get logical operator support from a code.
    
    Handles both CSSCode (with logical_x_support/logical_z_support methods)
    and general codes.
    
    Parameters
    ----------
    code : Code
        The quantum error correcting code.
    basis : str
        "X" or "Z".
    logical_idx : int
        Which logical qubit.
        
    Returns
    -------
    List[int]
        Qubit indices in the logical operator support.
    """
    basis = basis.upper()
    
    # Try CSSCode methods first
    if basis == "X" and hasattr(code, 'logical_x_support'):
        return code.logical_x_support(logical_idx)
    if basis == "Z" and hasattr(code, 'logical_z_support'):
        return code.logical_z_support(logical_idx)
    
    # Helper to safely get logical ops - handles both properties and methods
    def _get_ops(attr_name):
        attr = getattr(code, attr_name, None)
        if attr is None:
            return None
        # If it's callable (a method), call it; otherwise assume it's a property/list
        if callable(attr):
            return attr()
        return attr
    
    # Fallback to parsing logical ops
    if hasattr(code, 'logical_x_ops') and basis == "X":
        ops = _get_ops('logical_x_ops')
        if ops and logical_idx < len(ops):
            support = _parse_pauli_support(ops[logical_idx], ('X', 'Y'), code.n)
            # Check for placeholder logical (all qubits = likely invalid)
            if support and len(support) < code.n:
                return support
            # Placeholder detected - return empty to signal unknown
            return []
    
    if hasattr(code, 'logical_z_ops') and basis == "Z":
        ops = _get_ops('logical_z_ops')
        if ops and logical_idx < len(ops):
            support = _parse_pauli_support(ops[logical_idx], ('Z', 'Y'), code.n)
            # Check for placeholder logical (all qubits = likely invalid)
            if support and len(support) < code.n:
                return support
            # Placeholder detected - return empty to signal unknown
            return []
    
    # Ultimate fallback - return empty to signal unknown logical operator
    # Using all qubits causes non-deterministic detectors
    return []


def _parse_pauli_support(
    pauli_op: Any,
    paulis: Tuple[str, ...],
    n: int,
) -> List[int]:
    """Parse Pauli operator support."""
    support = []
    if isinstance(pauli_op, str):
        for q, p in enumerate(pauli_op):
            if p in paulis:
                support.append(q)
    elif isinstance(pauli_op, dict):
        for q, p in pauli_op.items():
            if p in paulis:
                support.append(q)
    elif isinstance(pauli_op, np.ndarray):
        half = len(pauli_op) // 2
        for q in range(min(n, half)):
            has_x = bool(pauli_op[q])
            has_z = bool(pauli_op[half + q]) if half + q < len(pauli_op) else False
            if has_x and has_z:
                p = 'Y'
            elif has_x:
                p = 'X'
            elif has_z:
                p = 'Z'
            else:
                continue
            if p in paulis:
                support.append(q)
    return support


class ColorCodeStabilizerRoundBuilder(CSSStabilizerRoundBuilder):
    """
    Stabilizer round builder for color codes with Chromobius-compatible detectors.
    
    Extends CSSStabilizerRoundBuilder to emit detector coordinates with a 4th
    component encoding (basis, color) as required by the Chromobius decoder:
    
    - coord[3] = 0, 1, 2 for X-type red, green, blue stabilizers
    - coord[3] = 3, 4, 5 for Z-type red, green, blue stabilizers
    
    The code must provide:
    - metadata["stab_colors"]: list of colors (0=red, 1=green, 2=blue) per stabilizer
    - metadata["is_chromobius_compatible"]: True
    """
    
    def __init__(
        self,
        code: "CSSCode",
        ctx: DetectorContext,
        block_name: str = "main",
        data_offset: int = 0,
        ancilla_offset: Optional[int] = None,
        measurement_basis: str = "Z",
    ):
        super().__init__(code, ctx, block_name, data_offset, ancilla_offset, measurement_basis)
        
        # Validate and cache color metadata
        if not self._meta.get("is_chromobius_compatible", False):
            raise ValueError(
                "ColorCodeStabilizerRoundBuilder requires a code with "
                "metadata['is_chromobius_compatible'] = True"
            )
        if "stab_colors" not in self._meta:
            raise ValueError(
                "ColorCodeStabilizerRoundBuilder requires code.metadata['stab_colors'] "
                "to be a list of colors (0=red, 1=green, 2=blue) per stabilizer"
            )
        
        self._stab_colors = self._meta["stab_colors"]
    
    def _get_color(self, s_idx: int, is_x_type: bool) -> int:
        """
        Get Chromobius color encoding for stabilizer.
        
        For X-type: color in {0, 1, 2}
        For Z-type: color + 3 in {3, 4, 5}
        """
        base_color = self._stab_colors[s_idx % len(self._stab_colors)] if self._stab_colors else 0
        return base_color if is_x_type else base_color + 3
    
    def _get_stab_coord(self, stab_type: str, s_idx: int) -> Tuple[float, float, float, float]:
        """Get 4D detector coordinate for a stabilizer (x, y, t, color)."""
        is_x_type = (stab_type == "x")
        coords = self._x_stab_coords if is_x_type else self._z_stab_coords
        color = float(self._get_color(s_idx, is_x_type))
        
        if s_idx < len(coords):
            x, y = coords[s_idx][:2]
            return (float(x), float(y), self.ctx.current_time, color)
        return (0.0, 0.0, self.ctx.current_time, color)


class GeneralStabilizerRoundBuilder(BaseStabilizerRoundBuilder):
    """
    Stabilizer round builder for general (non-CSS) stabilizer codes.
    
    Handles stabilizers with mixed X, Y, and Z components using the
    symplectic stabilizer_matrix representation [X|Z]. Each stabilizer
    is measured by entangling an ancilla with data qubits, with
    appropriate basis rotations:
    
    - X component: H - CNOT - H
    - Y component: S† - H - CNOT - H - S
    - Z component: CNOT
    
    Parameters
    ----------
    code : StabilizerCode
        The quantum error correcting code with stabilizer_matrix attribute.
    ctx : DetectorContext
        Context for tracking measurements and detectors.
    block_name : str
        Name for this code block.
    data_offset : int
        Offset for data qubit indices.
    ancilla_offset : int, optional
        Offset for ancilla qubit indices.
    """
    
    def __init__(
        self,
        code: "StabilizerCode",
        ctx: DetectorContext,
        block_name: str = "main",
        data_offset: int = 0,
        ancilla_offset: Optional[int] = None,
        measurement_basis: str = "Z",
    ):
        # Pass measurement_basis to parent for proper first-round detector handling
        super().__init__(code, ctx, block_name, data_offset, ancilla_offset, measurement_basis)
        
        # Get symplectic stabilizer matrix
        self._stab_mat = getattr(code, 'stabilizer_matrix', None)
        if self._stab_mat is None or self._stab_mat.size == 0:
            raise ValueError("GeneralStabilizerRoundBuilder requires code.stabilizer_matrix")
        
        self._n_stabs = self._stab_mat.shape[0]
        self._n = code.n
        
        # Tracking for measurements
        self._last_stab_meas: List[Optional[int]] = [None] * self._n_stabs
        self._round_number = 0
    
    @property
    def data_qubits(self) -> List[int]:
        """Global indices of data qubits."""
        return list(range(self.data_offset, self.data_offset + self._n))
    
    @property
    def ancilla_qubits(self) -> List[int]:
        """Global indices of ancilla qubits."""
        return list(range(self.ancilla_offset, self.ancilla_offset + self._n_stabs))
    
    @property
    def total_qubits(self) -> int:
        """Total qubits used by this block."""
        return self._n + self._n_stabs
    
    def emit_qubit_coords(self, circuit: stim.Circuit) -> None:
        """Emit QUBIT_COORDS for all qubits."""
        for local_idx, coord in enumerate(self._data_coords):
            if len(coord) >= 2:
                global_idx = self.data_offset + local_idx
                circuit.append("QUBIT_COORDS", [global_idx], [float(coord[0]), float(coord[1])])
    
    def emit_reset_all(self, circuit: stim.Circuit) -> None:
        """Reset all data and ancilla qubits."""
        all_qubits = self.data_qubits + self.ancilla_qubits
        if all_qubits:
            circuit.append("R", all_qubits)
    
    def emit_prepare_logical_state(
        self,
        circuit: stim.Circuit,
        state: str = "0",
        logical_idx: int = 0,
    ) -> None:
        """
        Prepare a logical eigenstate.
        
        For non-CSS codes, we prepare based on the logical operator structure.
        """
        code = self.code
        
        # Get logical operators
        logical_x = None
        logical_z = None
        if hasattr(code, 'logical_x') and code.logical_x:
            ops = code.logical_x
            if logical_idx < len(ops):
                logical_x = ops[logical_idx]
        if hasattr(code, 'logical_z') and code.logical_z:
            ops = code.logical_z
            if logical_idx < len(ops):
                logical_z = ops[logical_idx]
        
        if state in ("0", "1"):
            # Z-basis eigenstate - prepare |0⟩_L or |1⟩_L
            if state == "1" and logical_x is not None:
                support = _parse_pauli_support(logical_x, ('X', 'Y', 'Z'), self._n)
                for q in support:
                    circuit.append("X", [self.data_offset + q])
        
        elif state in ("+", "-"):
            # X-basis eigenstate - apply H to all data qubits
            # Then apply Z to logical X support for |-⟩
            circuit.append("H", self.data_qubits)
            if state == "-" and logical_x is not None:
                support = _parse_pauli_support(logical_x, ('X', 'Y', 'Z'), self._n)
                for q in support:
                    circuit.append("Z", [self.data_offset + q])
        
        circuit.append("TICK")
    
    def emit_round(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_detectors: bool = True,
        emit_metachecks: bool = False,
    ) -> None:
        """
        Emit one complete stabilizer measurement round.
        
        For non-CSS codes, all stabilizers are measured together using
        basis rotations appropriate to each Pauli component.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        stab_type : StabilizerBasis
            Ignored for non-CSS codes (always measures all stabilizers).
        emit_detectors : bool
            Whether to emit time-like detectors.
        emit_metachecks : bool
            Ignored for non-CSS codes.
        """
        if self._n_stabs == 0:
            return
        
        anc = self.ancilla_qubits
        
        # Reset ancillas at start of round
        circuit.append("R", anc)
        circuit.append("TICK")
        
        # Apply stabilizer gates with graph-coloring scheduling
        self._emit_general_stabilizer_cnots(circuit)
        
        circuit.append("TICK")
        
        # Measure all ancillas
        meas_start = self.ctx.add_measurement(self._n_stabs)
        circuit.append("MR", anc)
        
        # Time-like detectors - for non-CSS codes, skip first round
        # (initial state is generally not eigenstate of all stabilizers)
        if emit_detectors:
            for s_idx in range(self._n_stabs):
                cur_meas = meas_start + s_idx
                prev_meas = self._last_stab_meas[s_idx]
                
                if prev_meas is not None:
                    # Compare with previous round
                    coord = self._get_stab_coord(s_idx)
                    self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
                # Skip first-round detectors for non-CSS codes
                
                self._last_stab_meas[s_idx] = cur_meas
        else:
            for s_idx in range(self._n_stabs):
                self._last_stab_meas[s_idx] = meas_start + s_idx
        
        # Advance time
        self.ctx.advance_time()
        self._round_number += 1
    
    def _emit_general_stabilizer_cnots(self, circuit: stim.Circuit) -> None:
        """
        Emit CNOTs for general stabilizers with basis rotations.
        
        Handles X, Y, Z components with appropriate pre/post rotations.
        
        CNOT direction convention (matches Stim and CSS builder):
        - X-type component: CNOT(ancilla, data) - ancilla controls, data is target
        - Y-type component: CNOT(ancilla, data) - same as X (Y = iXZ)
        - Z-type component: CNOT(data, ancilla) - data controls, ancilla is target
        
        Why different directions?
        - X-type with H-CNOT(a→d)-H: X propagates target→control, so ancilla must
          control for X to propagate from data (target) to ancilla (control)
        - Z-type with CNOT(d→a): Z propagates control→target, so data must control
          for Z to propagate from data (control) to ancilla (target)
        """
        stab_mat = self._stab_mat
        data = self.data_qubits
        anc = self.ancilla_qubits
        n = self._n
        
        # Collect operations: (ctrl_idx, tgt_idx, pauli_type)
        # Direction depends on Pauli type!
        all_ops: List[Tuple[int, int, str]] = []
        
        for s_idx in range(self._n_stabs):
            x_part = stab_mat[s_idx, :n]
            z_part = stab_mat[s_idx, n:2*n] if stab_mat.shape[1] >= 2*n else np.zeros(n)
            
            for q in range(n):
                x_bit = x_part[q] if q < len(x_part) else 0
                z_bit = z_part[q] if q < len(z_part) else 0
                
                if x_bit and z_bit:
                    # Y-type: ancilla controls, data is target (like X)
                    all_ops.append((anc[s_idx], data[q], 'Y'))
                elif x_bit:
                    # X-type: ancilla controls, data is target
                    all_ops.append((anc[s_idx], data[q], 'X'))
                elif z_bit:
                    # Z-type: data controls, ancilla is target
                    all_ops.append((data[q], anc[s_idx], 'Z'))
        
        if not all_ops:
            return
        
        # Extract CNOT pairs (ctrl, tgt) and build lookup
        cnot_pairs = [(ctrl, tgt) for ctrl, tgt, _ in all_ops]
        op_pauli = {(ctrl, tgt): p for ctrl, tgt, p in all_ops}
        
        # Schedule into conflict-free layers
        layers = graph_coloring_cnots(cnot_pairs)
        
        # Emit each layer with basis rotations
        for layer_idx, layer_cnots in enumerate(layers):
            if layer_idx > 0:
                circuit.append("TICK")
            
            # Group by Pauli type - identify data qubits for rotations
            # For X/Y: data is TARGET (second element), for Z: data is CTRL (first element)
            x_ops = [(ctrl, tgt) for ctrl, tgt in layer_cnots if op_pauli.get((ctrl, tgt)) == 'X']
            y_ops = [(ctrl, tgt) for ctrl, tgt in layer_cnots if op_pauli.get((ctrl, tgt)) == 'Y']
            z_ops = [(ctrl, tgt) for ctrl, tgt in layer_cnots if op_pauli.get((ctrl, tgt)) == 'Z']
            
            # Pre-rotation for X: H on data qubits (which are TARGETS for X-type)
            if x_ops:
                x_data = list(set(tgt for _, tgt in x_ops))
                circuit.append("H", x_data)
            
            # Pre-rotation for Y: S† then H on data qubits (which are TARGETS for Y-type)
            if y_ops:
                y_data = list(set(tgt for _, tgt in y_ops))
                circuit.append("S_DAG", y_data)
                circuit.append("H", y_data)
            
            # Apply all CNOTs with correct direction per Pauli type
            for ctrl, tgt in layer_cnots:
                circuit.append("CX", [ctrl, tgt])
            
            # Post-rotation for X: H on data qubits
            if x_ops:
                circuit.append("H", x_data)
            
            # Post-rotation for Y: H then S on data qubits
            if y_ops:
                circuit.append("H", y_data)
                circuit.append("S", y_data)
    
    def _get_stab_coord(self, s_idx: int) -> Tuple[float, float, float]:
        """Get detector coordinate for a stabilizer."""
        # For non-CSS codes, use generic coordinates
        return (0.0, float(s_idx), self.ctx.current_time)
    
    def reset_stabilizer_history(self, swap_xz: bool = False, skip_first_round: bool = False, clear_history: bool = False) -> None:
        """
        Reset the builder's internal stabilizer measurement history.
        
        For non-CSS codes, this clears all stabilizer measurement history.
        
        Parameters
        ----------
        swap_xz : bool
            Ignored for non-CSS codes.
        skip_first_round : bool  
            Ignored for non-CSS codes.
        clear_history : bool
            If True, clear all measurement history.
        """
        # Clear measurement history
        self._last_stab_meas = [None] * self._n_stabs
        self._round_number = 0
    
    def emit_space_like_detectors(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
    ) -> None:
        """
        Emit space-like detectors comparing final data measurements with last stabilizer round.
        
        For non-CSS codes, this is more complex since stabilizers have mixed components.
        We can only emit space-like detectors for stabilizers whose components
        match the measurement basis.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        basis : str
            Measurement basis ("Z" or "X").
        """
        n = len(self.data_qubits)
        
        # Get the measurement start from context
        # Assumes we're called right after data qubits were measured
        meas_start = self.ctx.measurement_index - n
        
        stab_mat = self._stab_mat
        basis = basis.upper()
        
        for s_idx in range(self._n_stabs):
            last_meas = self._last_stab_meas[s_idx]
            if last_meas is None:
                continue
            
            x_part = stab_mat[s_idx, :n]
            z_part = stab_mat[s_idx, n:2*n] if stab_mat.shape[1] >= 2*n else np.zeros(n)
            
            # For space-like detectors to work, stabilizer components must match measurement
            # Z measurement: Z components contribute, X components are traced out
            # X measurement: X components contribute, Z components are traced out
            support = []
            if basis == "Z":
                for q in range(n):
                    z_bit = z_part[q] if q < len(z_part) else 0
                    x_bit = x_part[q] if q < len(x_part) else 0
                    if z_bit and not x_bit:  # Pure Z component
                        support.append(q)
            else:  # X basis
                for q in range(n):
                    z_bit = z_part[q] if q < len(z_part) else 0
                    x_bit = x_part[q] if q < len(x_part) else 0
                    if x_bit and not z_bit:  # Pure X component
                        support.append(q)
            
            if support:
                data_idxs = [meas_start + q for q in support]
                recs = data_idxs + [last_meas]
                coord = self._get_stab_coord(s_idx)
                self.ctx.emit_detector(circuit, recs, coord)

    def emit_final_measurement(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
        logical_idx: int = 0,
    ) -> List[int]:
        """
        Emit final data qubit measurements with space-like detectors.
        
        For non-CSS codes, we measure in the basis matching the logical operator.
        """
        data = self.data_qubits
        n = len(data)
        basis = basis.upper()
        
        # Get logical operator to determine measurement bases
        code = self.code
        if basis == "Z" and hasattr(code, 'logical_z') and code.logical_z:
            logical_op = code.logical_z[logical_idx] if logical_idx < len(code.logical_z) else None
        elif hasattr(code, 'logical_x') and code.logical_x:
            logical_op = code.logical_x[logical_idx] if logical_idx < len(code.logical_x) else None
        else:
            logical_op = None
        
        # Determine per-qubit measurement basis from logical operator
        qubit_basis = {q: 'Z' for q in range(n)}  # Default
        logical_support = []
        
        if logical_op is not None:
            support = _parse_pauli_support(logical_op, ('X', 'Y', 'Z'), n)
            logical_support = support
            
            # Parse the actual Pauli type for each qubit
            if isinstance(logical_op, str):
                for i, p in enumerate(logical_op):
                    if p in ('X', 'Y', 'Z'):
                        qubit_basis[i] = p
            elif isinstance(logical_op, dict):
                for q, p in logical_op.items():
                    if p in ('X', 'Y', 'Z'):
                        qubit_basis[q] = p
        
        # Apply basis change gates
        x_qubits = [data[q] for q in range(n) if qubit_basis.get(q) == 'X']
        y_qubits = [data[q] for q in range(n) if qubit_basis.get(q) == 'Y']
        
        if x_qubits:
            circuit.append("H", x_qubits)
        for q in y_qubits:
            circuit.append("S_DAG", [q])
            circuit.append("H", [q])
        
        # Measure all data qubits
        meas_start = self.ctx.add_measurement(n)
        circuit.append("M", data)
        
        # Space-like detectors (simplified for non-CSS)
        # Only emit if we can verify stabilizers with final measurements
        stab_mat = self._stab_mat
        for s_idx in range(self._n_stabs):
            last_meas = self._last_stab_meas[s_idx]
            if last_meas is None:
                continue
            
            x_part = stab_mat[s_idx, :n]
            z_part = stab_mat[s_idx, n:2*n] if stab_mat.shape[1] >= 2*n else np.zeros(n)
            
            # Check if stabilizer can be verified with current measurement bases
            support = []
            can_measure = True
            
            for q in range(n):
                x_bit = x_part[q] if q < len(x_part) else 0
                z_bit = z_part[q] if q < len(z_part) else 0
                meas_b = qubit_basis.get(q, 'Z')
                
                if x_bit and z_bit:  # Y component
                    if meas_b != 'Y':
                        can_measure = False
                        break
                    support.append(q)
                elif x_bit:  # X component
                    if meas_b not in ('X', 'Y'):
                        can_measure = False
                        break
                    support.append(q)
                elif z_bit:  # Z component
                    if meas_b not in ('Z', 'Y'):
                        can_measure = False
                        break
                    support.append(q)
            
            if not can_measure:
                continue
            
            data_idxs = [meas_start + q for q in support]
            recs = data_idxs + [last_meas]
            
            if recs:
                coord = self._get_stab_coord(s_idx)
                self.ctx.emit_detector(circuit, recs, coord)
        
        # Logical observable
        if logical_support:
            logical_meas = [meas_start + q for q in logical_support if q < n]
        else:
            # No valid logical operator found - use first qubit as minimal fallback
            import warnings
            warnings.warn(
                f"No valid logical operator found for {type(code).__name__}. "
                f"Observable will track qubit 0 only - decoding may not work correctly.",
                RuntimeWarning,
                stacklevel=2
            )
            logical_meas = [meas_start]  # Single qubit fallback
        
        self.ctx.add_observable_measurement(logical_idx, logical_meas)
        
        return logical_meas


class XYZColorCodeStabilizerRoundBuilder(BaseStabilizerRoundBuilder):
    """
    Stabilizer round builder for XYZ color codes with C_XYZ basis cycling.
    
    Unlike CSS codes, XYZ color codes use:
    - One ancilla per face (not separate X and Z ancillas)
    - C_XYZ gate on data qubits to cycle measurement basis (X→Y→Z→X)
    - Single detector per face (measuring joint XYZ observable)
    - Color encoding 0-2 (no X/Z distinction since it's a joint measurement)
    
    The C_XYZ gate cycles the measurement basis each round:
    - Round 1: effectively measures X component
    - Round 2: effectively measures Y component  
    - Round 3: effectively measures Z component
    - Round 4: back to X, etc.
    
    For Z-basis memory (|0⟩ initial state), the Z-like rounds give
    deterministic outcomes. However, since the cycling is implicit,
    we use time-like detectors comparing consecutive rounds after
    the first two rounds establish a baseline.
    
    Chromobius compatibility:
    - 4D coordinates: (x, y, t, color) where color ∈ {0, 1, 2}
    - Uses single basis encoding (0-2) since XYZ is a joint measurement
    
    Parameters
    ----------
    code : Code
        An XYZ color code with metadata containing:
        - faces: list of data qubit indices per stabilizer face
        - stab_colors: color (0-2) per face
        - ancilla_coords: (x, y) coordinates for ancilla qubits
    ctx : DetectorContext
        Context for tracking measurements and detectors.
    block_name : str
        Name for this code block.
    data_offset : int
        Offset for data qubit indices.
    ancilla_offset : int, optional
        Offset for ancilla qubit indices.
    measurement_basis : str
        Memory basis ("Z" or "X").
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
        super().__init__(code, ctx, block_name, data_offset, ancilla_offset, measurement_basis)
        
        # Validate XYZ color code metadata
        if "faces" not in self._meta:
            raise ValueError(
                "XYZColorCodeStabilizerRoundBuilder requires code.metadata['faces'] "
                "to be a list of data qubit indices per stabilizer face"
            )
        if "stab_colors" not in self._meta:
            raise ValueError(
                "XYZColorCodeStabilizerRoundBuilder requires code.metadata['stab_colors'] "
                "to be a list of colors (0=red, 1=green, 2=blue) per face"
            )
        if "ancilla_coords" not in self._meta:
            raise ValueError(
                "XYZColorCodeStabilizerRoundBuilder requires code.metadata['ancilla_coords'] "
                "to be a list of (x, y) coordinates for ancilla qubits"
            )
        
        # Cache XYZ-specific info
        self._faces: List[List[int]] = self._meta["faces"]
        self._stab_colors: List[int] = self._meta["stab_colors"]
        self._ancilla_coords: List[Tuple[float, float]] = self._meta["ancilla_coords"]
        self._n_faces = len(self._faces)
        
        # Build CNOT schedule (data → ancilla)
        self._cnot_schedule = self._build_cnot_schedule()
        
        # Track last measurement for each face (for time-like detectors)
        self._last_face_meas: List[Optional[int]] = [None] * self._n_faces
        
        # Track round count for first-round detector logic
        self._round_count = 0
    
    @property
    def ancilla_qubits(self) -> List[int]:
        """Global indices of ancilla qubits (one per face)."""
        return list(range(self.ancilla_offset, self.ancilla_offset + self._n_faces))
    
    @property
    def total_qubits(self) -> int:
        """Total qubits used by this block (data + ancillas)."""
        return self.code.n + self._n_faces
    
    def _build_cnot_schedule(self) -> List[List[Tuple[int, int]]]:
        """
        Build CNOT schedule in layers to avoid qubit conflicts.
        
        Uses greedy graph coloring to minimize circuit depth while ensuring
        no two CNOTs in the same layer share a data qubit or ancilla.
        
        Returns
        -------
        List[List[Tuple[int, int]]]
            List of layers, each layer is list of (data_q, face_idx) pairs.
        """
        # Collect all CNOTs needed: (data_qubit, face_index)
        all_cnots = []
        for face_idx, face in enumerate(self._faces):
            for data_q in face:
                all_cnots.append((data_q, face_idx))
        
        # Greedy scheduling - assign to first non-conflicting layer
        layers: List[List[Tuple[int, int]]] = []
        for data_q, face_idx in all_cnots:
            placed = False
            for layer in layers:
                # Check if this CNOT conflicts with any in this layer
                conflict = False
                for d, f in layer:
                    if d == data_q or f == face_idx:
                        conflict = True
                        break
                if not conflict:
                    layer.append((data_q, face_idx))
                    placed = True
                    break
            if not placed:
                layers.append([(data_q, face_idx)])
        
        return layers
    
    def emit_qubit_coords(self, circuit: stim.Circuit) -> None:
        """Emit QUBIT_COORDS for all data and ancilla qubits."""
        # Data qubits
        for local_idx, coord in enumerate(self._data_coords):
            if len(coord) >= 2:
                global_idx = self.data_offset + local_idx
                circuit.append("QUBIT_COORDS", [global_idx], [float(coord[0]), float(coord[1])])
        
        # Ancilla qubits (at face centers)
        for local_idx, (x, y) in enumerate(self._ancilla_coords):
            global_idx = self.ancilla_offset + local_idx
            circuit.append("QUBIT_COORDS", [global_idx], [float(x), float(y)])
    
    def emit_reset_all(self, circuit: stim.Circuit) -> None:
        """Reset all data and ancilla qubits."""
        all_qubits = self.data_qubits + self.ancilla_qubits
        if all_qubits:
            circuit.append("R", all_qubits)
    
    def emit_prepare_logical_state(
        self,
        circuit: stim.Circuit,
        state: str = "0",
        logical_idx: int = 0,
    ) -> None:
        """
        Prepare a logical eigenstate.
        
        For XYZ color codes, preparation is similar to CSS codes:
        - |0⟩_L: all data qubits in |0⟩ (reset)
        - |+⟩_L: apply H to all data qubits
        """
        if state in ("0", "1"):
            # Z-basis eigenstate
            if state == "1":
                logical_support = self._meta.get("logical_x_support", [])
                for q in logical_support:
                    circuit.append("X", [self.data_offset + q])
        
        elif state in ("+", "-"):
            # X-basis eigenstate
            if state == "-":
                logical_support = self._meta.get("logical_x_support", [])
                for q in logical_support:
                    circuit.append("Z", [self.data_offset + q])
            circuit.append("H", self.data_qubits)
        
        circuit.append("TICK")
    
    def emit_round(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_detectors: bool = True,
    ) -> None:
        """
        Emit one XYZ stabilizer measurement round.
        
        Structure:
        1. C_XYZ on all data qubits (basis rotation)
        2. CNOT layers (data → ancilla)
        3. MR on ancillas
        4. Time-like detectors comparing with previous round
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        stab_type : StabilizerBasis
            Ignored for XYZ codes (always measures joint XYZ).
        emit_detectors : bool
            Whether to emit time-like detectors.
        """
        data = self.data_qubits
        anc = self.ancilla_qubits
        
        # Step 1: Apply C_XYZ to all data qubits
        circuit.append("TICK")
        circuit.append("C_XYZ", data)
        circuit.append("TICK")
        
        # Step 2: CNOT layers (data controls ancilla)
        for layer_idx, layer in enumerate(self._cnot_schedule):
            if layer_idx > 0:
                circuit.append("TICK")
            for data_q, face_idx in layer:
                global_data = self.data_offset + data_q
                global_anc = self.ancilla_offset + face_idx
                circuit.append("CX", [global_data, global_anc])
        
        circuit.append("TICK")
        
        # Step 3: Measure and reset ancillas
        meas_start = self.ctx.add_measurement(self._n_faces)
        circuit.append("MR", anc)
        
        # Step 4: Time-like detectors
        if emit_detectors:
            for face_idx in range(self._n_faces):
                cur_meas = meas_start + face_idx
                prev_meas = self._last_face_meas[face_idx]
                
                coord = self._get_face_coord(face_idx)
                
                if prev_meas is None:
                    # First round - skip detector (no baseline yet)
                    # XYZ cycling means we need at least 2 rounds for comparison
                    pass
                else:
                    # Compare with previous round
                    self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
                
                self._last_face_meas[face_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "xyz", face_idx, cur_meas
                )
        else:
            for face_idx in range(self._n_faces):
                cur_meas = meas_start + face_idx
                self._last_face_meas[face_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "xyz", face_idx, cur_meas
                )
        
        # Advance time
        self.ctx.advance_time()
        self._emit_shift_coords(circuit)
        self._round_count += 1
    
    def _get_face_coord(self, face_idx: int) -> Tuple[float, float, float, float]:
        """
        Get 4D detector coordinate for a face.
        
        Returns (x, y, t, color) where color ∈ {0, 1, 2}.
        """
        if face_idx < len(self._ancilla_coords):
            x, y = self._ancilla_coords[face_idx]
        else:
            x, y = 0.0, 0.0
        
        color = float(self._stab_colors[face_idx]) if face_idx < len(self._stab_colors) else 0.0
        
        return (float(x), float(y), self.ctx.current_time, color)
    
    def emit_final_measurement(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
        logical_idx: int = 0,
    ) -> List[int]:
        """
        Emit final data qubit measurements with space-like detectors.
        
        For XYZ codes, final detectors compare last syndrome measurement
        with data qubit product for each face.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        basis : str
            Measurement basis ("Z" or "X").
        logical_idx : int
            Which logical qubit to track.
            
        Returns
        -------
        List[int]
            Measurement indices for the logical observable.
        """
        data = self.data_qubits
        n = len(data)
        basis = basis.upper()
        
        # Basis change if needed
        if basis == "X":
            circuit.append("H", data)
        
        # Measure all data qubits
        meas_start = self.ctx.add_measurement(n)
        circuit.append("M", data)
        
        # Space-like detectors: compare last syndrome with data product
        for face_idx, face in enumerate(self._faces):
            last_meas = self._last_face_meas[face_idx]
            if last_meas is None:
                continue
            
            # Get data measurement indices for this face
            data_idxs = [meas_start + dq for dq in face if dq < n]
            
            if data_idxs:
                recs = data_idxs + [last_meas]
                # Use current time for space-like detectors
                coord = self._get_face_coord(face_idx)
                coord = (coord[0], coord[1], self.ctx.current_time, coord[3])
                self.ctx.emit_detector(circuit, recs, coord)
        
        # Compute logical observable
        if basis == "Z":
            logical_support = self._meta.get("logical_z_support", [])
        else:
            logical_support = self._meta.get("logical_x_support", [])
        
        # Check for placeholder logical (all qubits = likely invalid)
        if logical_support and len(logical_support) >= n:
            logical_support = []  # Treat as unknown
        
        logical_meas = [meas_start + q for q in logical_support if q < n]
        if not logical_meas:
            # No valid logical operator found - use first qubit as minimal fallback
            import warnings
            warnings.warn(
                f"No valid logical {basis} operator found for XYZ color code. "
                f"Observable will track qubit 0 only - decoding may not work correctly.",
                RuntimeWarning,
                stacklevel=2
            )
            logical_meas = [meas_start]  # Single qubit fallback
        
        self.ctx.add_observable_measurement(logical_idx, logical_meas)
        
        return logical_meas
    
    def emit_rounds(
        self,
        circuit: stim.Circuit,
        num_rounds: int,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
    ) -> None:
        """Emit multiple XYZ stabilizer measurement rounds."""
        for _ in range(num_rounds):
            self.emit_round(circuit, stab_type, emit_detectors=True)


# Backwards compatibility alias
StabilizerRoundBuilder = CSSStabilizerRoundBuilder
