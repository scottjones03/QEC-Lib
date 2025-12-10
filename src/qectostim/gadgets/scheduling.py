"""
Parallel gate scheduling for fault-tolerant gadget circuits.

This module provides infrastructure for scheduling quantum gates in parallel
layers, minimizing circuit depth while respecting qubit dependencies.

Two scheduling strategies are supported:
1. Geometric scheduling: Uses code's cnot_schedule when available
2. Graph coloring: Greedy coloring fallback for arbitrary gate patterns
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum

from .coordinates import CoordND


class GateType(Enum):
    """Supported gate types for scheduling."""
    # Single qubit gates
    I = "I"
    X = "X"
    Y = "Y"
    Z = "Z"
    H = "H"
    S = "S"
    S_DAG = "S_DAG"
    SQRT_X = "SQRT_X"
    SQRT_X_DAG = "SQRT_X_DAG"
    
    # Two qubit gates
    CX = "CX"
    CY = "CY"
    CZ = "CZ"
    
    # Special operations
    R = "R"  # Reset
    M = "M"  # Measure Z
    MX = "MX"  # Measure X
    MR = "MR"  # Measure and reset
    MRX = "MRX"  # Measure X and reset


@dataclass
class ScheduledGate:
    """A gate scheduled in the circuit."""
    gate_type: GateType
    targets: Tuple[int, ...]  # Global qubit indices
    
    def __post_init__(self):
        # Ensure targets is a tuple
        if not isinstance(self.targets, tuple):
            self.targets = tuple(self.targets)


@dataclass
class CircuitLayer:
    """
    A single time-slice of parallel operations.
    
    All operations in a layer execute simultaneously and must not
    share any qubit targets.
    """
    time: float
    resets: List[int] = field(default_factory=list)
    gates: List[ScheduledGate] = field(default_factory=list)
    measurements: List[Tuple[int, GateType]] = field(default_factory=list)
    
    def get_all_qubits(self) -> Set[int]:
        """Return all qubits used in this layer."""
        qubits = set(self.resets)
        for gate in self.gates:
            qubits.update(gate.targets)
        for qubit, _ in self.measurements:
            qubits.add(qubit)
        return qubits
    
    def can_add_gate(self, gate: ScheduledGate) -> bool:
        """Check if gate can be added without conflicts."""
        used = self.get_all_qubits()
        return all(q not in used for q in gate.targets)
    
    def can_add_reset(self, qubit: int) -> bool:
        """Check if reset can be added without conflicts."""
        return qubit not in self.get_all_qubits()
    
    def can_add_measurement(self, qubit: int) -> bool:
        """Check if measurement can be added without conflicts."""
        return qubit not in self.get_all_qubits()


class GadgetScheduler:
    """
    Scheduler for fault-tolerant gadget circuits.
    
    Manages parallel gate scheduling with two strategies:
    1. Geometric: Uses code's cnot_schedule for natural parallelism
    2. Graph coloring: Greedy coloring for arbitrary patterns
    
    Parameters
    ----------
    layout : GadgetLayout
        The layout manager providing qubit indices and coordinates.
    time_start : float
        Starting time coordinate for the schedule.
    """
    
    def __init__(self, layout: Any, time_start: float = 0.0):
        self.layout = layout
        self.current_time = time_start
        self.layers: List[CircuitLayer] = []
        self._measurement_records: List[Tuple[int, float, str]] = []  # (qubit, time, type)
    
    def _get_or_create_layer(self, time: float) -> CircuitLayer:
        """Get layer at time, creating if necessary."""
        for layer in self.layers:
            if abs(layer.time - time) < 1e-9:
                return layer
        layer = CircuitLayer(time=time)
        self.layers.append(layer)
        self.layers.sort(key=lambda l: l.time)
        return layer
    
    def advance_time(self, delta: float = 1.0) -> float:
        """Advance the current time and return new time."""
        self.current_time += delta
        return self.current_time
    
    def schedule_resets(self, qubits: List[int], time: Optional[float] = None) -> float:
        """
        Schedule reset operations on qubits.
        
        Parameters
        ----------
        qubits : List[int]
            Global qubit indices to reset.
        time : Optional[float]
            Time to schedule at. Uses current_time if None.
            
        Returns
        -------
        float
            Time at which resets are scheduled.
        """
        t = time if time is not None else self.current_time
        layer = self._get_or_create_layer(t)
        
        for q in qubits:
            if layer.can_add_reset(q):
                layer.resets.append(q)
            else:
                # Create new layer if conflict
                t = self.advance_time()
                layer = self._get_or_create_layer(t)
                layer.resets.append(q)
        
        return t
    
    def schedule_single_qubit_gate(
        self,
        gate_type: GateType,
        qubits: List[int],
        time: Optional[float] = None
    ) -> float:
        """
        Schedule single-qubit gates in parallel.
        
        Parameters
        ----------
        gate_type : GateType
            The gate to apply.
        qubits : List[int]
            Global qubit indices.
        time : Optional[float]
            Time to schedule at.
            
        Returns
        -------
        float
            Time at which gates are scheduled.
        """
        t = time if time is not None else self.current_time
        layer = self._get_or_create_layer(t)
        
        for q in qubits:
            gate = ScheduledGate(gate_type, (q,))
            if layer.can_add_gate(gate):
                layer.gates.append(gate)
            else:
                # Should not happen for single-qubit gates on distinct qubits
                t = self.advance_time()
                layer = self._get_or_create_layer(t)
                layer.gates.append(gate)
        
        return t
    
    def schedule_two_qubit_gates_geometric(
        self,
        gate_type: GateType,
        pairs: List[Tuple[int, int]],
        schedule: Optional[List[List[int]]] = None
    ) -> float:
        """
        Schedule two-qubit gates using geometric schedule.
        
        Uses the code's cnot_schedule if provided, otherwise falls back
        to graph coloring.
        
        Parameters
        ----------
        gate_type : GateType
            Two-qubit gate type (CX, CY, CZ).
        pairs : List[Tuple[int, int]]
            List of (control, target) pairs as global indices.
        schedule : Optional[List[List[int]]]
            Pre-computed schedule from code. Each sublist is a layer
            containing pair indices that can execute in parallel.
            
        Returns
        -------
        float
            Time after all gates scheduled.
        """
        if not pairs:
            return self.current_time
            
        if schedule is not None:
            return self._apply_geometric_schedule(gate_type, pairs, schedule)
        else:
            return self._apply_graph_coloring_schedule(gate_type, pairs)
    
    def _apply_geometric_schedule(
        self,
        gate_type: GateType,
        pairs: List[Tuple[int, int]],
        schedule: List[List[int]]
    ) -> float:
        """Apply pre-computed geometric schedule."""
        for layer_indices in schedule:
            layer = self._get_or_create_layer(self.current_time)
            
            for idx in layer_indices:
                if idx < len(pairs):
                    ctrl, tgt = pairs[idx]
                    gate = ScheduledGate(gate_type, (ctrl, tgt))
                    layer.gates.append(gate)
            
            self.advance_time()
        
        return self.current_time
    
    def _apply_graph_coloring_schedule(
        self,
        gate_type: GateType,
        pairs: List[Tuple[int, int]]
    ) -> float:
        """
        Apply greedy graph coloring for scheduling.
        
        Build conflict graph where edges connect gates sharing a qubit,
        then greedily color to find parallel layers.
        """
        n = len(pairs)
        if n == 0:
            return self.current_time
            
        # Build conflict graph as adjacency list
        conflicts: Dict[int, Set[int]] = {i: set() for i in range(n)}
        
        for i in range(n):
            ctrl_i, tgt_i = pairs[i]
            qubits_i = {ctrl_i, tgt_i}
            
            for j in range(i + 1, n):
                ctrl_j, tgt_j = pairs[j]
                qubits_j = {ctrl_j, tgt_j}
                
                if qubits_i & qubits_j:  # Overlap
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        
        # Greedy graph coloring
        colors: Dict[int, int] = {}
        
        for gate_idx in range(n):
            # Find colors used by neighbors
            neighbor_colors = {colors[nb] for nb in conflicts[gate_idx] if nb in colors}
            
            # Find smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1
            
            colors[gate_idx] = color
        
        # Group by color
        max_color = max(colors.values()) + 1 if colors else 0
        color_groups: List[List[int]] = [[] for _ in range(max_color)]
        
        for gate_idx, color in colors.items():
            color_groups[color].append(gate_idx)
        
        # Schedule each color group as a layer
        for group in color_groups:
            layer = self._get_or_create_layer(self.current_time)
            
            for gate_idx in group:
                ctrl, tgt = pairs[gate_idx]
                gate = ScheduledGate(gate_type, (ctrl, tgt))
                layer.gates.append(gate)
            
            self.advance_time()
        
        return self.current_time
    
    def schedule_measurements(
        self,
        qubits: List[int],
        basis: GateType = GateType.M,
        time: Optional[float] = None,
        labels: Optional[List[str]] = None
    ) -> float:
        """
        Schedule measurement operations.
        
        Parameters
        ----------
        qubits : List[int]
            Global qubit indices to measure.
        basis : GateType
            Measurement basis (M, MX, MR, MRX).
        time : Optional[float]
            Time to schedule at.
        labels : Optional[List[str]]
            Labels for measurement records.
            
        Returns
        -------
        float
            Time at which measurements are scheduled.
        """
        t = time if time is not None else self.current_time
        layer = self._get_or_create_layer(t)
        
        for i, q in enumerate(qubits):
            if layer.can_add_measurement(q):
                layer.measurements.append((q, basis))
                label = labels[i] if labels and i < len(labels) else f"meas_{q}"
                self._measurement_records.append((q, t, label))
            else:
                t = self.advance_time()
                layer = self._get_or_create_layer(t)
                layer.measurements.append((q, basis))
                label = labels[i] if labels and i < len(labels) else f"meas_{q}"
                self._measurement_records.append((q, t, label))
        
        return t
    
    def schedule_stabilizer_round(
        self,
        block_name: str,
        stabilizer_type: str,  # "X" or "Z"
        data_qubits: List[int],
        ancilla_qubits: List[int],
        cnot_pairs: List[Tuple[int, int]],
        cnot_schedule: Optional[List[List[int]]] = None
    ) -> float:
        """
        Schedule a complete stabilizer measurement round.
        
        Parameters
        ----------
        block_name : str
            Name of the block (for labeling).
        stabilizer_type : str
            "X" or "Z" for stabilizer type.
        data_qubits : List[int]
            Data qubit global indices.
        ancilla_qubits : List[int]
            Ancilla qubit global indices.
        cnot_pairs : List[Tuple[int, int]]
            CNOT (control, target) pairs.
        cnot_schedule : Optional[List[List[int]]]
            Pre-computed CNOT schedule.
            
        Returns
        -------
        float
            Time after round completes.
        """
        # Reset ancillas
        self.schedule_resets(ancilla_qubits)
        self.advance_time()
        
        # Hadamard on ancillas for X stabilizers
        if stabilizer_type == "X":
            self.schedule_single_qubit_gate(GateType.H, ancilla_qubits)
            self.advance_time()
        
        # CNOTs
        self.schedule_two_qubit_gates_geometric(
            GateType.CX, cnot_pairs, cnot_schedule
        )
        
        # Hadamard on ancillas for X stabilizers
        if stabilizer_type == "X":
            self.schedule_single_qubit_gate(GateType.H, ancilla_qubits)
            self.advance_time()
        
        # Measure ancillas
        labels = [f"{block_name}_{stabilizer_type}_{i}" for i in range(len(ancilla_qubits))]
        self.schedule_measurements(ancilla_qubits, labels=labels)
        self.advance_time()
        
        return self.current_time
    
    def schedule_transversal_gate(
        self,
        gate_type: GateType,
        block_qubits: List[int]
    ) -> float:
        """
        Schedule a transversal single-qubit gate on a block.
        
        Parameters
        ----------
        gate_type : GateType
            The gate to apply transversally.
        block_qubits : List[int]
            All data qubits in the block.
            
        Returns
        -------
        float
            Time after gate applied.
        """
        self.schedule_single_qubit_gate(gate_type, block_qubits)
        self.advance_time()
        return self.current_time
    
    def schedule_transversal_two_qubit(
        self,
        gate_type: GateType,
        control_qubits: List[int],
        target_qubits: List[int]
    ) -> float:
        """
        Schedule a transversal two-qubit gate between blocks.
        
        Parameters
        ----------
        gate_type : GateType
            Two-qubit gate (CX, CZ).
        control_qubits : List[int]
            Control qubits from one block.
        target_qubits : List[int]
            Target qubits from another block.
            
        Returns
        -------
        float
            Time after gates applied.
        """
        if len(control_qubits) != len(target_qubits):
            raise ValueError("Control and target qubit lists must have same length")
        
        pairs = list(zip(control_qubits, target_qubits))
        self.schedule_two_qubit_gates_geometric(gate_type, pairs)
        
        return self.current_time
    
    def schedule_joint_measurement(
        self,
        bridge_ancilla: int,
        target_qubits: List[int],
        measurement_type: str,  # "XX" or "ZZ"
        cnot_schedule: Optional[List[List[int]]] = None
    ) -> float:
        """
        Schedule a joint parity measurement via bridge ancilla.
        
        Parameters
        ----------
        bridge_ancilla : int
            Global index of bridge ancilla.
        target_qubits : List[int]
            Data qubits to measure parity of.
        measurement_type : str
            "XX" or "ZZ" for measurement type.
        cnot_schedule : Optional[List[List[int]]]
            Pre-computed CNOT schedule.
            
        Returns
        -------
        float
            Time after measurement.
        """
        # Reset bridge ancilla
        self.schedule_resets([bridge_ancilla])
        self.advance_time()
        
        # For XX: H on ancilla, CNOTs from ancilla to targets, H on ancilla
        # For ZZ: CNOTs from targets to ancilla
        
        if measurement_type == "XX":
            self.schedule_single_qubit_gate(GateType.H, [bridge_ancilla])
            self.advance_time()
            
            pairs = [(bridge_ancilla, t) for t in target_qubits]
            self.schedule_two_qubit_gates_geometric(GateType.CX, pairs, cnot_schedule)
            
            self.schedule_single_qubit_gate(GateType.H, [bridge_ancilla])
            self.advance_time()
        else:  # ZZ
            pairs = [(t, bridge_ancilla) for t in target_qubits]
            self.schedule_two_qubit_gates_geometric(GateType.CX, pairs, cnot_schedule)
        
        # Measure bridge ancilla
        self.schedule_measurements([bridge_ancilla], labels=["joint_parity"])
        self.advance_time()
        
        return self.current_time
    
    def get_measurement_record_index(
        self,
        qubit: int,
        time: float
    ) -> Optional[int]:
        """
        Get the measurement record index for a measurement.
        
        Returns negative index (as used in Stim rec syntax).
        """
        # Find matching measurement
        for i, (q, t, _) in enumerate(self._measurement_records):
            if q == qubit and abs(t - time) < 1e-9:
                # Convert to negative index from end
                return i - len(self._measurement_records)
        return None
    
    def to_stim_instructions(self) -> List[str]:
        """
        Convert schedule to Stim circuit instructions.
        
        Returns
        -------
        List[str]
            List of Stim instruction strings.
        """
        instructions = []
        
        # Sort layers by time
        sorted_layers = sorted(self.layers, key=lambda l: l.time)
        
        for layer in sorted_layers:
            # Emit TICK for time boundary
            instructions.append("TICK")
            
            # Resets
            if layer.resets:
                qubits = " ".join(str(q) for q in layer.resets)
                instructions.append(f"R {qubits}")
            
            # Gates by type
            gates_by_type: Dict[GateType, List[ScheduledGate]] = {}
            for gate in layer.gates:
                if gate.gate_type not in gates_by_type:
                    gates_by_type[gate.gate_type] = []
                gates_by_type[gate.gate_type].append(gate)
            
            for gate_type, gates in gates_by_type.items():
                if len(gates[0].targets) == 1:
                    # Single qubit gates
                    qubits = " ".join(str(g.targets[0]) for g in gates)
                    instructions.append(f"{gate_type.value} {qubits}")
                else:
                    # Two qubit gates
                    pairs = []
                    for g in gates:
                        pairs.extend([str(g.targets[0]), str(g.targets[1])])
                    qubits = " ".join(pairs)
                    instructions.append(f"{gate_type.value} {qubits}")
            
            # Measurements
            if layer.measurements:
                # Group by measurement type
                by_type: Dict[GateType, List[int]] = {}
                for qubit, mtype in layer.measurements:
                    if mtype not in by_type:
                        by_type[mtype] = []
                    by_type[mtype].append(qubit)
                
                for mtype, qubits in by_type.items():
                    qs = " ".join(str(q) for q in qubits)
                    instructions.append(f"{mtype.value} {qs}")
        
        return instructions
    
    def get_circuit_depth(self) -> int:
        """Return the number of time layers in the schedule."""
        return len(self.layers)
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the schedule.
        
        Returns
        -------
        Dict[str, int]
            Statistics including gate counts and depth.
        """
        stats = {
            "depth": len(self.layers),
            "total_gates": 0,
            "single_qubit_gates": 0,
            "two_qubit_gates": 0,
            "measurements": 0,
            "resets": 0,
        }
        
        for layer in self.layers:
            stats["resets"] += len(layer.resets)
            stats["measurements"] += len(layer.measurements)
            
            for gate in layer.gates:
                stats["total_gates"] += 1
                if len(gate.targets) == 1:
                    stats["single_qubit_gates"] += 1
                else:
                    stats["two_qubit_gates"] += 1
        
        return stats


def merge_schedules(
    schedules: List['GadgetScheduler'],
    time_gap: float = 1.0
) -> 'GadgetScheduler':
    """
    Merge multiple schedules sequentially.
    
    Parameters
    ----------
    schedules : List[GadgetScheduler]
        Schedules to merge.
    time_gap : float
        Time gap between schedules.
        
    Returns
    -------
    GadgetScheduler
        Merged schedule.
    """
    if not schedules:
        raise ValueError("No schedules to merge")
    
    # Use first schedule's layout (they should be compatible)
    merged = GadgetScheduler(schedules[0].layout, time_start=0.0)
    
    current_offset = 0.0
    
    for sched in schedules:
        # Copy layers with time offset
        for layer in sched.layers:
            new_layer = CircuitLayer(
                time=layer.time + current_offset,
                resets=list(layer.resets),
                gates=list(layer.gates),
                measurements=list(layer.measurements)
            )
            merged.layers.append(new_layer)
        
        # Copy measurement records with offset
        for qubit, time, label in sched._measurement_records:
            merged._measurement_records.append((qubit, time + current_offset, label))
        
        # Update offset for next schedule
        if sched.layers:
            max_time = max(l.time for l in sched.layers)
            current_offset = max_time + time_gap
    
    merged.current_time = current_offset
    return merged
