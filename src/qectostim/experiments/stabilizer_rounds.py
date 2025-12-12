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
        
        # Build coordinate lookup for geometric scheduling
        self._coord_to_data: Dict[Tuple[float, float], int] = {}
        for local_idx, coord in enumerate(self._data_coords):
            if len(coord) >= 2:
                key = (float(coord[0]), float(coord[1]))
                self._coord_to_data[key] = data_offset + local_idx
        
        # Track round number for detector emission
        self._round_number = 0
    
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
    ):
        super().__init__(code, ctx, block_name, data_offset, ancilla_offset, measurement_basis)
        
        # Cache stabilizer info - CSS codes define hx/hz as @property returning numpy arrays
        _hx_raw = getattr(code, 'hx', None)
        _hz_raw = getattr(code, 'hz', None)
        
        # Only use if it's actually a numpy array (has .shape attribute)
        self._hx = _hx_raw if _hx_raw is not None and hasattr(_hx_raw, 'shape') else None
        self._hz = _hz_raw if _hz_raw is not None and hasattr(_hz_raw, 'shape') else None
        self._n_x = self._hx.shape[0] if self._hx is not None and self._hx.size > 0 else 0
        self._n_z = self._hz.shape[0] if self._hz is not None and self._hz.size > 0 else 0
        
        # Cache CSS-specific stabilizer coordinates
        self._x_stab_coords = self._meta.get('x_stab_coords', [])
        self._z_stab_coords = self._meta.get('z_stab_coords', [])
        self._x_schedule = self._meta.get('x_schedule')
        self._z_schedule = self._meta.get('z_schedule')
        
        # Track last measurements for each stabilizer (for time-like detectors)
        self._last_x_meas: List[Optional[int]] = [None] * self._n_x
        self._last_z_meas: List[Optional[int]] = [None] * self._n_z
    
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
    def total_qubits(self) -> int:
        """Total qubits used by this block (data + ancillas)."""
        return self.code.n + self._n_x + self._n_z
    
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
        """Reset all data and ancilla qubits."""
        all_qubits = self.data_qubits + self.x_ancillas + self.z_ancillas
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
        # Measure X stabilizers
        if stab_type in (StabilizerBasis.X, StabilizerBasis.BOTH):
            self._emit_x_round(circuit, emit_detectors)
        
        # Measure Z stabilizers
        if stab_type in (StabilizerBasis.Z, StabilizerBasis.BOTH):
            self._emit_z_round(circuit, emit_detectors)
        
        # Advance time and emit SHIFT_COORDS for visualization
        self.ctx.advance_time()
        self._emit_shift_coords(circuit)
        self._round_number += 1
    
    def _emit_x_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """Emit X stabilizer measurements."""
        if self._hx is None or self._n_x == 0:
            return
        
        x_anc = self.x_ancillas
        
        # Prepare X ancillas with H
        circuit.append("H", x_anc)
        circuit.append("TICK")
        
        # Apply CNOTs using geometric or graph-coloring schedule
        if self._use_geometric_x():
            self._emit_geometric_cnots(circuit, "x")
        else:
            self._emit_graph_coloring_cnots(circuit, self._hx, self.data_qubits, x_anc, is_x_type=True)
        
        # TICK to separate CNOTs from final H
        circuit.append("TICK")
        
        # Final H on X ancillas
        circuit.append("H", x_anc)
        circuit.append("TICK")
        
        # Measure X ancillas
        meas_start = self.ctx.add_measurement(self._n_x)
        circuit.append("MR", x_anc)
        
        # Time-like detectors with proper first-round logic
        # For Z-basis memory: X stabilizers have RANDOM first-round outcomes (|0⟩ is not X eigenstate)
        # For X-basis memory: X stabilizers have DETERMINISTIC first-round outcomes (|+⟩ is X eigenstate)
        if emit_detectors:
            for s_idx in range(self._n_x):
                cur_meas = meas_start + s_idx
                prev_meas = self._last_x_meas[s_idx]
                
                if prev_meas is None:
                    # First round: only create detector if basis matches
                    if self.measurement_basis == "X":
                        coord = self._get_stab_coord("x", s_idx)
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                    # else: skip first-round detector for X stabilizers in Z-basis memory
                else:
                    # Compare with previous round
                    coord = self._get_stab_coord("x", s_idx)
                    self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
                
                self._last_x_meas[s_idx] = cur_meas
                # Also record in context for FT gadget experiments
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "x", s_idx, cur_meas
                )
        else:
            # Still record measurements even without detectors
            for s_idx in range(self._n_x):
                cur_meas = meas_start + s_idx
                self._last_x_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "x", s_idx, cur_meas
                )
    
    def _emit_z_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """Emit Z stabilizer measurements."""
        if self._hz is None or self._n_z == 0:
            return
        
        z_anc = self.z_ancillas
        
        # Apply CNOTs using geometric or graph-coloring schedule
        if self._use_geometric_z():
            self._emit_geometric_cnots(circuit, "z")
        else:
            self._emit_graph_coloring_cnots(circuit, self._hz, self.data_qubits, z_anc, is_x_type=False)
        
        circuit.append("TICK")
        
        # Measure Z ancillas
        meas_start = self.ctx.add_measurement(self._n_z)
        circuit.append("MR", z_anc)
        
        # Time-like detectors with proper first-round logic
        # For Z-basis memory: Z stabilizers have DETERMINISTIC first-round outcomes (|0⟩ is Z eigenstate)
        # For X-basis memory: Z stabilizers have RANDOM first-round outcomes (|+⟩ is not Z eigenstate)
        if emit_detectors:
            for s_idx in range(self._n_z):
                cur_meas = meas_start + s_idx
                prev_meas = self._last_z_meas[s_idx]
                
                if prev_meas is None:
                    # First round: only create detector if basis matches
                    if self.measurement_basis == "Z":
                        coord = self._get_stab_coord("z", s_idx)
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                    # else: skip first-round detector for Z stabilizers in X-basis memory
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
    
    def _use_geometric_x(self) -> bool:
        """Check if geometric scheduling is available for X stabilizers."""
        return (
            self._x_schedule is not None and
            self._data_coords and
            len(self._x_stab_coords) == self._n_x
        )
    
    def _use_geometric_z(self) -> bool:
        """Check if geometric scheduling is available for Z stabilizers."""
        return (
            self._z_schedule is not None and
            self._data_coords and
            len(self._z_stab_coords) == self._n_z
        )
    
    def _emit_geometric_cnots(self, circuit: stim.Circuit, stab_type: str) -> None:
        """Emit CNOTs using geometric scheduling.
        
        CNOT direction convention:
        - X-type stabilizers: CNOT(data, ancilla) - data controls
        - Z-type stabilizers: CNOT(ancilla, data) - ancilla controls
        """
        if stab_type == "x":
            schedule = self._x_schedule
            stab_coords = self._x_stab_coords
            ancillas = self.x_ancillas
        else:
            schedule = self._z_schedule
            stab_coords = self._z_stab_coords
            ancillas = self.z_ancillas
        
        for layer_idx, (dx, dy) in enumerate(schedule):
            if layer_idx > 0:
                circuit.append("TICK")
            for s_idx, (sx, sy) in enumerate(stab_coords):
                if s_idx >= len(ancillas):
                    continue
                anc = ancillas[s_idx]
                nbr = (float(sx) + dx, float(sy) + dy)
                dq = self._coord_to_data.get(nbr)
                if dq is not None:
                    if stab_type == "x":
                        # X-type: CNOT from data to ancilla
                        circuit.append("CNOT", [dq, anc])
                    else:
                        # Z-type: CNOT from ancilla to data
                        circuit.append("CNOT", [anc, dq])
    
    def _emit_graph_coloring_cnots(
        self,
        circuit: stim.Circuit,
        stab_matrix: np.ndarray,
        data_qubits: List[int],
        ancilla_qubits: List[int],
        is_x_type: bool = True,
    ) -> None:
        """Emit CNOTs using greedy graph coloring.
        
        CNOT direction convention:
        - X-type stabilizers (is_x_type=True): CNOT(data, ancilla)
        - Z-type stabilizers (is_x_type=False): CNOT(ancilla, data)
        
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
            True for X-type stabilizers, False for Z-type.
        """
        if stab_matrix is None or stab_matrix.size == 0:
            return
        
        n_stabs, n_data = stab_matrix.shape
        
        # Collect all CNOT pairs with correct direction
        all_cnots: List[Tuple[int, int]] = []
        for s_idx in range(min(n_stabs, len(ancilla_qubits))):
            anc = ancilla_qubits[s_idx]
            for d_idx in range(min(n_data, len(data_qubits))):
                if stab_matrix[s_idx, d_idx]:
                    dq = data_qubits[d_idx]
                    if is_x_type:
                        # X-type: CNOT from data to ancilla
                        all_cnots.append((dq, anc))
                    else:
                        # Z-type: CNOT from ancilla to data
                        all_cnots.append((anc, dq))
        
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
    
    def emit_rounds(
        self,
        circuit: stim.Circuit,
        num_rounds: int,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
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
        """
        for _ in range(num_rounds):
            self.emit_round(circuit, stab_type, emit_detectors=True)
    
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
            # Fallback: all data qubits
            logical_meas = list(range(meas_start, meas_start + n))
        
        # Add to observable accumulator
        self.ctx.add_observable_measurement(logical_idx, logical_meas)
        
        return logical_meas
    
    def emit_space_like_detectors(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
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
            Measurement basis ("Z" or "X").
        """
        n = len(self.data_qubits)
        
        # Get the measurement start from context
        # Assumes we're called right after data qubits were measured
        meas_start = self.ctx.measurement_index - n
        
        if basis.upper() == "Z" and self._hz is not None:
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
        
        elif basis.upper() == "X" and self._hx is not None:
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
            return _parse_pauli_support(ops[logical_idx], ('X', 'Y'), code.n)
    
    if hasattr(code, 'logical_z_ops') and basis == "Z":
        ops = _get_ops('logical_z_ops')
        if ops and logical_idx < len(ops):
            return _parse_pauli_support(ops[logical_idx], ('Z', 'Y'), code.n)
    
    # Ultimate fallback
    return list(range(code.n))


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
    ):
        # Don't pass measurement_basis - non-CSS codes don't have deterministic first rounds
        super().__init__(code, ctx, block_name, data_offset, ancilla_offset, measurement_basis="Z")
        
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
        emit_detectors: bool = True,
    ) -> None:
        """
        Emit one complete stabilizer measurement round.
        
        For non-CSS codes, all stabilizers are measured together using
        basis rotations appropriate to each Pauli component.
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
        """
        stab_mat = self._stab_mat
        data = self.data_qubits
        anc = self.ancilla_qubits
        n = self._n
        
        # Collect operations: (data_idx, anc_idx, pauli_type)
        all_ops: List[Tuple[int, int, str]] = []
        
        for s_idx in range(self._n_stabs):
            x_part = stab_mat[s_idx, :n]
            z_part = stab_mat[s_idx, n:2*n] if stab_mat.shape[1] >= 2*n else np.zeros(n)
            
            for q in range(n):
                x_bit = x_part[q] if q < len(x_part) else 0
                z_bit = z_part[q] if q < len(z_part) else 0
                
                if x_bit and z_bit:
                    all_ops.append((data[q], anc[s_idx], 'Y'))
                elif x_bit:
                    all_ops.append((data[q], anc[s_idx], 'X'))
                elif z_bit:
                    all_ops.append((data[q], anc[s_idx], 'Z'))
        
        if not all_ops:
            return
        
        # Extract CNOT pairs and build lookup
        cnot_pairs = [(dq, a) for dq, a, _ in all_ops]
        op_pauli = {(dq, a): p for dq, a, p in all_ops}
        
        # Schedule into conflict-free layers
        layers = graph_coloring_cnots(cnot_pairs)
        
        # Emit each layer with basis rotations
        for layer_idx, layer_cnots in enumerate(layers):
            if layer_idx > 0:
                circuit.append("TICK")
            
            # Group by Pauli type
            x_ops = [(dq, a) for dq, a in layer_cnots if op_pauli.get((dq, a)) == 'X']
            y_ops = [(dq, a) for dq, a in layer_cnots if op_pauli.get((dq, a)) == 'Y']
            z_ops = [(dq, a) for dq, a in layer_cnots if op_pauli.get((dq, a)) == 'Z']
            
            # Pre-rotation for X: H
            if x_ops:
                x_data = list(set(dq for dq, _ in x_ops))
                circuit.append("H", x_data)
            
            # Pre-rotation for Y: S† then H
            if y_ops:
                y_data = list(set(dq for dq, _ in y_ops))
                circuit.append("S_DAG", y_data)
                circuit.append("H", y_data)
            
            # Apply all CNOTs (data controls ancilla)
            for dq, a in layer_cnots:
                circuit.append("CX", [dq, a])
            
            # Post-rotation for X: H
            if x_ops:
                circuit.append("H", x_data)
            
            # Post-rotation for Y: H then S
            if y_ops:
                circuit.append("H", y_data)
                circuit.append("S", y_data)
    
    def _get_stab_coord(self, s_idx: int) -> Tuple[float, float, float]:
        """Get detector coordinate for a stabilizer."""
        # For non-CSS codes, use generic coordinates
        return (0.0, float(s_idx), self.ctx.current_time)
    
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
            logical_meas = list(range(meas_start, meas_start + n))
        
        self.ctx.add_observable_measurement(logical_idx, logical_meas)
        
        return logical_meas


# Backwards compatibility alias
StabilizerRoundBuilder = CSSStabilizerRoundBuilder
