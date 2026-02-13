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
        from qectostim.experiments.hardware_simulation.trapped_ion.routing.greedy import (
            route_ions_junction,
            GateRequest,
        )

        physical_ops = self._native_ops_to_physical(
            circuit.native_circuit.operations,
            circuit.mapping,
        )

        # --- attempt junction-based routing if architecture supports it ---
        routing_metadata: Dict[str, Any] = {"routing_strategy": "qccd_junction"}
        routing_ops: List[Any] = []
        routing_overhead = 0

        arch = self.architecture
        if isinstance(arch, QCCDArchitecture):
            qccd_graph = arch.qccd_graph

            # Build gate requests for two-qubit gates
            gate_requests: List[GateRequest] = []
            for gid, op in enumerate(physical_ops):
                if len(getattr(op, "qubits", ())) == 2:
                    q0, q1 = op.qubits[0], op.qubits[1]
                    # Find ion objects
                    all_ions = qccd_graph.ions
                    ion0 = all_ions.get(q0)
                    ion1 = all_ions.get(q1)
                    if ion0 is not None and ion1 is not None:
                        gate_requests.append(GateRequest(
                            ancilla_ion=ion0,
                            data_ion=ion1,
                            priority=gid,
                            gate_id=gid,
                        ))

            if gate_requests:
                trap_cap = arch.ions_per_trap
                result = route_ions_junction(qccd_graph, gate_requests, trap_cap)
                routing_ops = result.transport_ops
                routing_overhead = len(routing_ops)
                routing_metadata["barriers"] = result.barriers
                routing_metadata["total_transport_time_s"] = result.total_time_s
                routing_metadata["total_transport_heating"] = result.total_heating
                routing_metadata["gate_execution_order"] = result.gate_execution_order
        else:
            routing_metadata = {"routing_strategy": "qccd_placeholder"}

        return RoutedCircuit(
            operations=physical_ops,
            final_mapping=circuit.mapping.copy(),
            routing_overhead=routing_overhead,
            mapped_circuit=circuit,
            routing_operations=routing_ops,
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
