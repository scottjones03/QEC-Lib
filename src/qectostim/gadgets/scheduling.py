# src/qectostim/gadgets/scheduling.py
"""
Parallel Gate Scheduler for Gadget Circuits.

Schedules gates for maximum parallelism while respecting qubit constraints.
Supports both geometric scheduling (for topological codes with explicit schedules)
and graph-coloring-based scheduling (fallback for arbitrary codes).

Key features:
- CircuitLayer abstraction for time slices
- Geometric scheduling using x_schedule/z_schedule from code metadata
- Greedy graph coloring for conflict-free CNOT layers
- Stim circuit generation with proper TICK and SHIFT_COORDS
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set
from collections import defaultdict
import numpy as np
import stim

from qectostim.gadgets.layout import GadgetLayout
from qectostim.gadgets.coordinates import (
    CoordND,
    emit_qubit_coords_nd,
    emit_detector_nd,
    pad_coord_to_dim,
)
from qectostim.utils.scheduling_core import graph_coloring_cnots


@dataclass
class CircuitLayer:
    """
    A single time slice of parallelizable operations.
    
    All operations within a layer can be executed simultaneously
    (no qubit conflicts).
    """
    
    time: float = 0.0
    resets: List[int] = field(default_factory=list)
    single_qubit_gates: Dict[str, List[int]] = field(default_factory=dict)
    two_qubit_gates: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    measurements: List[int] = field(default_factory=list)
    measurement_reset: bool = False  # Use MR instead of M
    
    def add_single_gate(self, gate_name: str, qubit: int) -> None:
        """Add a single-qubit gate."""
        if gate_name not in self.single_qubit_gates:
            self.single_qubit_gates[gate_name] = []
        self.single_qubit_gates[gate_name].append(qubit)
    
    def add_two_qubit_gate(self, gate_name: str, q1: int, q2: int) -> None:
        """Add a two-qubit gate."""
        if gate_name not in self.two_qubit_gates:
            self.two_qubit_gates[gate_name] = []
        self.two_qubit_gates[gate_name].append((q1, q2))
    
    def get_all_qubits(self) -> Set[int]:
        """Get all qubits involved in this layer."""
        qubits = set(self.resets)
        for gate_qubits in self.single_qubit_gates.values():
            qubits.update(gate_qubits)
        for gate_pairs in self.two_qubit_gates.values():
            for q1, q2 in gate_pairs:
                qubits.add(q1)
                qubits.add(q2)
        qubits.update(self.measurements)
        return qubits
    
    def is_empty(self) -> bool:
        """Check if layer has no operations."""
        return (
            not self.resets and 
            not self.single_qubit_gates and 
            not self.two_qubit_gates and 
            not self.measurements
        )
    
    def to_stim(self, circuit: stim.Circuit) -> None:
        """Append this layer's operations to a Stim circuit."""
        # Resets first
        if self.resets:
            circuit.append("R", self.resets)
        
        # Single-qubit gates
        for gate_name, qubits in self.single_qubit_gates.items():
            if qubits:
                circuit.append(gate_name, qubits)
        
        # Two-qubit gates
        for gate_name, pairs in self.two_qubit_gates.items():
            for q1, q2 in pairs:
                circuit.append(gate_name, [q1, q2])
        
        # Measurements
        if self.measurements:
            if self.measurement_reset:
                circuit.append("MR", self.measurements)
            else:
                circuit.append("M", self.measurements)


class GadgetScheduler:
    """
    Schedules operations for a gadget circuit with maximum parallelism.
    
    Uses geometric scheduling when available (codes with x_schedule/z_schedule),
    otherwise falls back to greedy graph coloring.
    
    Parameters
    ----------
    layout : GadgetLayout
        The spatial layout of code blocks.
        
    Examples
    --------
    >>> scheduler = GadgetScheduler(layout)
    >>> scheduler.schedule_transversal_gate("H", "control")
    >>> scheduler.schedule_stabilizer_round("control", "x")
    >>> circuit = scheduler.to_stim()
    """
    
    def __init__(self, layout: GadgetLayout):
        self.layout = layout
        self.layers: List[CircuitLayer] = []
        self.current_time: float = 0.0
        self._measurement_index = 0
        self._detector_info: List[Tuple[CoordND, float, List[int]]] = []
        self._observable_info: List[Tuple[int, List[int]]] = []
    
    def add_layer(self, layer: Optional[CircuitLayer] = None) -> CircuitLayer:
        """Add a new layer (time slice) to the schedule."""
        if layer is None:
            layer = CircuitLayer(time=self.current_time)
        else:
            layer.time = self.current_time
        self.layers.append(layer)
        return layer
    
    def advance_time(self, delta: float = 1.0) -> None:
        """Advance the current time."""
        self.current_time += delta
    
    def schedule_resets(self, qubits: List[int]) -> CircuitLayer:
        """Schedule reset operations on given qubits."""
        layer = self.add_layer()
        layer.resets = list(qubits)
        self.advance_time()
        return layer
    
    def schedule_transversal_gate(
        self,
        gate_name: str,
        block_name: str,
        target_qubits: Optional[List[int]] = None,
    ) -> CircuitLayer:
        """
        Schedule a transversal (parallel) single-qubit gate on a block.
        
        Parameters
        ----------
        gate_name : str
            Gate name (e.g., 'H', 'S', 'T').
        block_name : str
            Name of the code block.
        target_qubits : List[int], optional
            Specific global qubit indices. If None, applies to all data qubits.
            
        Returns
        -------
        CircuitLayer
            The layer containing the transversal gate.
        """
        if target_qubits is None:
            target_qubits = self.layout.get_block_data_qubits(block_name)
        
        layer = self.add_layer()
        layer.single_qubit_gates[gate_name] = list(target_qubits)
        self.advance_time()
        return layer
    
    def schedule_transversal_two_qubit_gate(
        self,
        gate_name: str,
        block_name_a: str,
        block_name_b: str,
    ) -> CircuitLayer:
        """
        Schedule a transversal two-qubit gate between two blocks.
        
        Pairs qubits by index: qubit i of block A with qubit i of block B.
        Blocks must have the same number of data qubits.
        
        Parameters
        ----------
        gate_name : str
            Gate name (e.g., 'CNOT', 'CZ').
        block_name_a : str
            First block (control for CNOT).
        block_name_b : str
            Second block (target for CNOT).
            
        Returns
        -------
        CircuitLayer
            The layer containing the transversal two-qubit gate.
        """
        qubits_a = self.layout.get_block_data_qubits(block_name_a)
        qubits_b = self.layout.get_block_data_qubits(block_name_b)
        
        if len(qubits_a) != len(qubits_b):
            raise ValueError(
                f"Blocks have different sizes: {block_name_a}={len(qubits_a)}, "
                f"{block_name_b}={len(qubits_b)}"
            )
        
        layer = self.add_layer()
        layer.two_qubit_gates[gate_name] = list(zip(qubits_a, qubits_b))
        self.advance_time()
        return layer
    
    def schedule_stabilizer_round(
        self,
        block_name: str,
        stab_type: str = "both",  # "x", "z", or "both"
        use_measurement_reset: bool = True,
    ) -> List[CircuitLayer]:
        """
        Schedule a full stabilizer measurement round for a block.
        
        Uses geometric scheduling if the code has x_schedule/z_schedule metadata,
        otherwise falls back to greedy graph coloring.
        
        Parameters
        ----------
        block_name : str
            Name of the code block.
        stab_type : str
            Which stabilizers to measure: "x", "z", or "both".
        use_measurement_reset : bool
            Whether to use MR (measure+reset) instead of M.
            
        Returns
        -------
        List[CircuitLayer]
            The layers comprising the stabilizer round.
        """
        block = self.layout.blocks.get(block_name)
        if block is None:
            return []
        
        code = block.code
        layers = []
        
        # Get stabilizer matrices - safely handle non-CSS codes
        # CSS codes define hx/hz as @property returning numpy arrays
        # Non-CSS codes may not have these or may have them as methods
        hx_raw = getattr(code, 'hx', None)
        hz_raw = getattr(code, 'hz', None)
        # Only use if it's actually a numpy array (has .shape attribute)
        hx = hx_raw if hx_raw is not None and hasattr(hx_raw, 'shape') else None
        hz = hz_raw if hz_raw is not None and hasattr(hz_raw, 'shape') else None
        
        # Get scheduling info from metadata
        meta = getattr(code, '_metadata', {}) or {}
        x_schedule = meta.get('x_schedule')
        z_schedule = meta.get('z_schedule')
        data_coords = meta.get('data_coords', [])
        x_stab_coords = meta.get('x_stab_coords', [])
        z_stab_coords = meta.get('z_stab_coords', [])
        
        data_qubits = self.layout.get_block_data_qubits(block_name)
        x_ancillas = self.layout.get_block_x_ancillas(block_name)
        z_ancillas = self.layout.get_block_z_ancillas(block_name)
        
        # Check if we can use geometric scheduling
        use_geo_x = (
            x_schedule is not None and 
            data_coords and 
            x_stab_coords and 
            len(x_stab_coords) == len(x_ancillas)
        )
        use_geo_z = (
            z_schedule is not None and 
            data_coords and 
            z_stab_coords and 
            len(z_stab_coords) == len(z_ancillas)
        )
        
        # Build coordinate lookup for geometric scheduling
        coord_to_data = {}
        if data_coords:
            for local_idx, coord in enumerate(data_coords):
                coord_to_data[tuple(coord)] = data_qubits[local_idx] if local_idx < len(data_qubits) else None
        
        # Schedule X stabilizers
        if stab_type in ("x", "both") and hx is not None and len(x_ancillas) > 0:
            # Prepare X ancillas (H gate)
            h_layer = self.add_layer()
            h_layer.single_qubit_gates["H"] = list(x_ancillas)
            self.advance_time()
            layers.append(h_layer)
            
            if use_geo_x:
                # Geometric scheduling: one TICK per phase
                for dx, dy in x_schedule:
                    cnot_layer = self.add_layer()
                    for s_idx, (sx, sy) in enumerate(x_stab_coords):
                        if s_idx >= len(x_ancillas):
                            continue
                        anc = x_ancillas[s_idx]
                        nbr = (float(sx) + dx, float(sy) + dy)
                        dq = coord_to_data.get(nbr)
                        if dq is not None:
                            cnot_layer.add_two_qubit_gate("CNOT", dq, anc)
                    self.advance_time()
                    layers.append(cnot_layer)
            else:
                # Fallback: use graph coloring for conflict-free CNOT layers
                layers.extend(self._schedule_stabilizers_graph_coloring(
                    hx, data_qubits, x_ancillas
                ))
            
            # Final H on X ancillas
            h_layer2 = self.add_layer()
            h_layer2.single_qubit_gates["H"] = list(x_ancillas)
            self.advance_time()
            layers.append(h_layer2)
            
            # Measure X ancillas
            meas_layer = self.add_layer()
            meas_layer.measurements = list(x_ancillas)
            meas_layer.measurement_reset = use_measurement_reset
            self.advance_time()
            layers.append(meas_layer)
        
        # Schedule Z stabilizers
        if stab_type in ("z", "both") and hz is not None and len(z_ancillas) > 0:
            if use_geo_z:
                # Geometric scheduling
                for dx, dy in z_schedule:
                    cnot_layer = self.add_layer()
                    for s_idx, (sx, sy) in enumerate(z_stab_coords):
                        if s_idx >= len(z_ancillas):
                            continue
                        anc = z_ancillas[s_idx]
                        nbr = (float(sx) + dx, float(sy) + dy)
                        dq = coord_to_data.get(nbr)
                        if dq is not None:
                            cnot_layer.add_two_qubit_gate("CNOT", dq, anc)
                    self.advance_time()
                    layers.append(cnot_layer)
            else:
                # Fallback: graph coloring
                layers.extend(self._schedule_stabilizers_graph_coloring(
                    hz, data_qubits, z_ancillas
                ))
            
            # Measure Z ancillas
            meas_layer = self.add_layer()
            meas_layer.measurements = list(z_ancillas)
            meas_layer.measurement_reset = use_measurement_reset
            self.advance_time()
            layers.append(meas_layer)
        
        return layers
    
    def _schedule_stabilizers_graph_coloring(
        self,
        stab_matrix: np.ndarray,
        data_qubits: List[int],
        ancilla_qubits: List[int],
    ) -> List[CircuitLayer]:
        """
        Schedule CNOT gates using greedy graph coloring.
        
        Creates conflict-free layers where no qubit is used twice.
        
        Parameters
        ----------
        stab_matrix : np.ndarray
            Stabilizer matrix (n_stabs x n_data).
        data_qubits : List[int]
            Global indices of data qubits.
        ancilla_qubits : List[int]
            Global indices of ancilla qubits.
            
        Returns
        -------
        List[CircuitLayer]
            CNOT layers with no conflicts.
        """
        if stab_matrix is None or stab_matrix.size == 0:
            return []
        
        n_stabs, n_data = stab_matrix.shape
        
        # Collect all CNOT operations: (data_qubit, ancilla_qubit)
        all_cnots = []
        for s_idx in range(min(n_stabs, len(ancilla_qubits))):
            anc = ancilla_qubits[s_idx]
            for d_idx in range(min(n_data, len(data_qubits))):
                if stab_matrix[s_idx, d_idx]:
                    dq = data_qubits[d_idx]
                    all_cnots.append((dq, anc))
        
        if not all_cnots:
            return []
        
        # Use shared graph coloring algorithm
        layers_data = graph_coloring_cnots(all_cnots)
        
        # Convert to CircuitLayer objects
        result = []
        for layer_cnots in layers_data:
            layer = self.add_layer()
            layer.two_qubit_gates["CNOT"] = layer_cnots
            self.advance_time()
            result.append(layer)
        
        return result
    
    def schedule_measurements(
        self,
        qubits: List[int],
        use_reset: bool = False,
    ) -> CircuitLayer:
        """Schedule measurement operations."""
        layer = self.add_layer()
        layer.measurements = list(qubits)
        layer.measurement_reset = use_reset
        self.advance_time()
        return layer
    
    def add_detector(
        self,
        coord: CoordND,
        measurement_indices: List[int],
    ) -> None:
        """
        Register a detector to be emitted.
        
        Parameters
        ----------
        coord : CoordND
            Spatial coordinate for the detector.
        measurement_indices : List[int]
            Absolute measurement record indices.
        """
        self._detector_info.append((coord, self.current_time, measurement_indices))
    
    def add_observable(
        self,
        observable_idx: int,
        measurement_indices: List[int],
    ) -> None:
        """
        Register an observable to be emitted.
        
        Parameters
        ----------
        observable_idx : int
            Logical observable index.
        measurement_indices : List[int]
            Absolute measurement record indices.
        """
        self._observable_info.append((observable_idx, measurement_indices))
    
    def to_stim(self, include_coords: bool = True) -> stim.Circuit:
        """
        Convert the scheduled layers to a Stim circuit.
        
        Parameters
        ----------
        include_coords : bool
            Whether to emit QUBIT_COORDS instructions.
            
        Returns
        -------
        stim.Circuit
            The complete Stim circuit.
        """
        circuit = stim.Circuit()
        
        # Emit QUBIT_COORDS
        if include_coords:
            for global_idx, coord in sorted(self.layout.qubit_map.global_coords.items()):
                emit_qubit_coords_nd(circuit, global_idx, coord)
        
        # Emit layers with TICKs
        for i, layer in enumerate(self.layers):
            if not layer.is_empty():
                layer.to_stim(circuit)
                # Track measurements
                n_meas = len(layer.measurements)
                self._measurement_index += n_meas
            
            # Add TICK between layers (except after last)
            if i < len(self.layers) - 1:
                circuit.append("TICK")
        
        # Emit detectors
        for coord, time, meas_indices in self._detector_info:
            emit_detector_nd(circuit, meas_indices, coord, time, self._measurement_index)
        
        # Emit observables
        for obs_idx, meas_indices in self._observable_info:
            if meas_indices:
                lookbacks = [idx - self._measurement_index for idx in meas_indices]
                targets = [stim.target_rec(lb) for lb in lookbacks]
                circuit.append("OBSERVABLE_INCLUDE", targets, obs_idx)
        
        return circuit
    
    def get_measurement_count(self) -> int:
        """Get total number of measurements scheduled."""
        return sum(len(layer.measurements) for layer in self.layers)
