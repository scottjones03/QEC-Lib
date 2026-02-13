# src/qectostim/experiments/hardware_simulation/trapped_ion/compilers/linear.py
"""
Linear chain trapped ion compiler.

Linear chains have all-to-all connectivity via the MS gate,
so NO ROUTING is needed.
"""
from __future__ import annotations

import logging
from typing import (
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


class LinearChainCompiler(TrappedIonCompiler):
    """Compiler for linear chain trapped ion architecture.
    
    Linear chains have all-to-all connectivity via the MS gate,
    so NO ROUTING is needed.
    """
    
    def __init__(
        self,
        architecture: "TrappedIonArchitecture",
        optimization_level: int = 1,
    ):
        super().__init__(architecture, optimization_level, use_global_rotations=True)
    
    def _setup_passes(self) -> None:
        """Set up linear chain compilation passes."""
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
            metadata={"source": "stim", "compiler": "LinearChainCompiler"},
            stim_instruction_map=stim_instruction_map,
            stim_source=circuit,
        )
    
    def map_qubits(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map logical qubits to ions in the chain."""
        mapping = QubitMapping()
        for logical_q in range(circuit.num_qubits):
            mapping.assign(logical_q, logical_q)
        
        return MappedCircuit(
            native_circuit=circuit,
            mapping=mapping,
            metadata={"mapping_strategy": "identity"},
        )
    
    def route(self, circuit: MappedCircuit) -> RoutedCircuit:
        """No routing needed for linear chain - all-to-all connectivity.

        Converts native ops to PhysicalOperation objects so that the
        scheduler receives concrete typed operations.
        """
        physical_ops = self._native_ops_to_physical(
            circuit.native_circuit.operations,
            circuit.mapping,
        )
        return RoutedCircuit(
            operations=physical_ops,
            final_mapping=circuit.mapping.copy(),
            routing_overhead=0,
            mapped_circuit=circuit,
            routing_operations=[],
            metadata={"routing_strategy": "none", "reason": "all-to-all connectivity"},
        )
    
    def schedule(self, circuit: RoutedCircuit) -> ScheduledCircuit:
        """Schedule operations for linear chain."""
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
