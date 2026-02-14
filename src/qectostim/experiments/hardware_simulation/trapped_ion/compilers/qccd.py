# src/qectostim/experiments/hardware_simulation/trapped_ion/compilers/qccd.py
"""
QCCD (Quantum Charge-Coupled Device) architecture compiler.

QCCD uses multiple trap zones connected by junctions, with ions
transported via split/merge/shuttle operations.
"""
from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    TYPE_CHECKING,
)

from qectostim.experiments.hardware_simulation.core.pipeline import (
    NativeCircuit,
    MappedCircuit,
    RoutedCircuit,
    ScheduledCircuit,
    QubitMapping,
)
from qectostim.experiments.hardware_simulation.trapped_ion.compilers.base import (
    TrappedIonCompiler,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
        TrappedIonArchitecture,
    )


_logger = logging.getLogger(__name__)


class QCCDCompiler(TrappedIonCompiler):
    """Compiler for general QCCD (Quantum Charge-Coupled Device) architecture.
    
    QCCD uses multiple trap zones connected by junctions, with ions
    transported via split/merge/shuttle operations.
    """
    
    def __init__(
        self,
        architecture: "TrappedIonArchitecture",
        optimization_level: int = 1,
        use_wise_routing: bool = False,
    ):
        super().__init__(architecture, optimization_level)
        self.use_wise_routing = use_wise_routing
    
    def _setup_passes(self) -> None:
        """Set up QCCD compilation passes."""
        pass
    
    def decompose_to_native(self, circuit) -> NativeCircuit:
        """Decompose circuit to native MS + rotation gates.
        
        Builds a stim_instruction_map that maps each stim instruction
        index (in the flattened circuit) to the native op indices it
        produced.  Annotations are preserved in stim_source.
        """
        native_ops = []
        stim_instruction_map: Dict[int, List[int]] = {}
        stim_idx = 0
        
        for instruction in circuit.flattened():
            gate_name = instruction.name
            
            # Skip annotations — they stay in the original circuit
            if gate_name in ("TICK", "QUBIT_COORDS", "DETECTOR",
                             "OBSERVABLE_INCLUDE", "SHIFT_COORDS"):
                stim_idx += 1
                continue
            
            targets = instruction.targets_copy()
            qubits = tuple(t.value for t in targets if t.is_qubit_target)
            if not qubits:
                stim_idx += 1
                continue
            
            start_native_idx = len(native_ops)
            decomposed = self.decompose_stim_gate(gate_name, qubits)
            native_ops.extend(decomposed)
            end_native_idx = len(native_ops)
            
            stim_instruction_map[stim_idx] = list(
                range(start_native_idx, end_native_idx)
            )
            stim_idx += 1
        
        return NativeCircuit(
            operations=native_ops,
            num_qubits=circuit.num_qubits,
            metadata={"source": "stim", "compiler": "QCCDCompiler"},
            stim_instruction_map=stim_instruction_map,
            stim_source=circuit,
        )
    
    def map_qubits(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map logical qubits to ions in QCCD zones."""
        mapping = QubitMapping()
        for logical_q in range(circuit.num_qubits):
            mapping.assign(logical_q, logical_q)
        
        return MappedCircuit(
            native_circuit=circuit,
            mapping=mapping,
            metadata={"mapping_strategy": "sequential"},
        )
    
    def route(self, circuit: MappedCircuit) -> RoutedCircuit:
        """Route ions between zones using split/merge/shuttle.

        Uses junction-based routing via ``route_ions_junction()`` when the
        architecture provides a QCCD graph.  Falls back to identity routing
        (no transport ops) otherwise.
        """
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            QCCDArchitecture,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.operations import (
            MSGate,
        )
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            ionRouting,
            old_ops_to_transport_list,
        )

        physical_ops = self._native_ops_to_physical(
            circuit.native_circuit.operations,
            circuit.mapping,
        )

        # --- attempt junction-based routing if architecture supports it ---
        routing_metadata: Dict[str, Any] = {"routing_strategy": "qccd_junction"}
        routing_overhead = 0
        ops_list: list = []

        arch = self.architecture
        if isinstance(arch, QCCDArchitecture):
            # Use the architecture directly (no adapter needed)
            # qubit_ions maps logical qubit index → Ion object
            qi = arch.qubit_ions
            old_operations = []
            for op in physical_ops:
                qs = getattr(op, "qubits", ())
                if len(qs) == 2:
                    q0, q1 = qs[0], qs[1]
                    if 0 <= q0 < len(qi) and 0 <= q1 < len(qi):
                        gate = MSGate.qubitOperation(qi[q0], qi[q1])
                        old_operations.append(gate)

            if old_operations:
                # ── Capture pre-routing ion arrangement ──────────────
                # ionRouting() mutates the architecture: ions physically
                # move between traps.  We save a snapshot so the animation
                # can *replay* transport ops from the initial positions.
                _pre_routing_arrangement: Dict[Any, list] = {}
                for _node in arch.qccd_graph.nodes.values():
                    _pre_routing_arrangement[_node] = list(_node.ions)
                for _cx in arch.qccd_graph.crossings.values():
                    if _cx.ion is not None:
                        _pre_routing_arrangement[_cx] = [_cx.ion]

                # Trap capacity for routing matches ions_per_trap
                trap_cap = arch.ions_per_trap
                all_ops, barriers = ionRouting(
                    arch, old_operations, trap_cap,
                )
                ops_list, barrier_list, meta = old_ops_to_transport_list(
                    all_ops, barriers,
                )
                # Filter out gate objects (MSGate, GateSwap) from routing ops —
                # they're already in physical_ops and would cause double-counting.
                # Keep only transport ops (_EdgeOp, Split, Merge, Move, etc.).
                _GATE_CLASSES = {"MSGate", "GateSwap", "QubitGate"}
                ops_list = [
                    o for o in ops_list
                    if type(o).__name__ not in _GATE_CLASSES
                ]
                routing_overhead = len(ops_list)
                routing_metadata["barriers"] = barrier_list
                routing_metadata["old_operations"] = ops_list
                routing_metadata["old_barriers"] = barrier_list

                # ── Restore ions to pre-routing positions ───────────
                # This lets the animation replay ops from the start.
                # 1) Clear all nodes and crossings
                for _node in arch.qccd_graph.nodes.values():
                    while _node.ions:
                        _node.remove_ion(_node.ions[0])
                for _cx in arch.qccd_graph.crossings.values():
                    if getattr(_cx, 'ion', None) is not None:
                        try:
                            _cx.clearIon()
                        except Exception:
                            try:
                                _cx.clear_ion()
                            except Exception:
                                pass
                # 2) Put ions back in their original traps
                for _container, _ions in _pre_routing_arrangement.items():
                    _ctype = type(_container).__name__
                    if 'Crossing' in _ctype:
                        if _ions:
                            try:
                                _container.setIon(_ions[0], None)
                            except Exception:
                                try:
                                    _container.set_ion(_ions[0], None)
                                except Exception:
                                    pass
                    else:
                        for _ion in _ions:
                            try:
                                _container.add_ion(_ion)
                            except Exception:
                                try:
                                    _container.addIon(_ion)
                                except Exception:
                                    pass
                # 3) Rebuild the graph
                if hasattr(arch, 'refreshGraph'):
                    arch.refreshGraph()
                elif hasattr(arch, 'refresh_graph'):
                    arch.refresh_graph()

                routing_metadata["pre_routing_arrangement"] = _pre_routing_arrangement
        else:
            routing_metadata = {"routing_strategy": "qccd_placeholder"}

        return RoutedCircuit(
            operations=physical_ops,
            routing_operations=ops_list,
            final_mapping=circuit.mapping.copy(),
            routing_overhead=routing_overhead,
            mapped_circuit=circuit,
            metadata=routing_metadata,
        )
    
    def schedule(self, circuit: RoutedCircuit) -> ScheduledCircuit:
        """Schedule operations for QCCD architecture."""
        from qectostim.experiments.hardware_simulation.core.operations import (
            GreedyBatchScheduler,
        )
        
        scheduler = GreedyBatchScheduler()
        batches = scheduler.schedule(circuit.operations)
        scheduled_ops, layers, total_dur = self._batches_to_scheduled(batches)
        
        return ScheduledCircuit(
            layers=layers,
            scheduled_ops=scheduled_ops,
            routed_circuit=circuit,
            batches=batches,
            total_duration=total_dur,
            metadata={"scheduler": "GreedyBatchScheduler"},
        )
    
    def plan_ion_routing(
        self,
        gate_sequence: List[Any],
        current_positions: Dict[int, str],
    ) -> List[Any]:
        """Plan ion routing for a gate sequence."""
        raise NotImplementedError(
            "QCCDCompiler.plan_ion_routing() not yet implemented. "
            "For WISE grids, use WISECompiler instead."
        )
