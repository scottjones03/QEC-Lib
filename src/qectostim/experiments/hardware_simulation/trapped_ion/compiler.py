# src/qectostim/experiments/hardware_simulation/trapped_ion/compiler.py
"""
Trapped ion circuit compiler.

Compiles logical circuits to trapped ion hardware with:
- Gate decomposition to MS + rotations
- Ion-to-qubit mapping
- Ion routing for multi-zone architectures
- Operation scheduling

Architecture hierarchy:
- TrappedIonCompiler (ABC): Technology-specific base (MS gates, ion physics)
  - WISECompiler: WISE grid architecture with SAT-based routing
  - LinearChainCompiler: Single linear chain, all-to-all connectivity
  - QCCDCompiler: General QCCD with split/merge/shuttle routing

Each concrete compiler handles the specific constraints of its architecture.
The base class provides common trapped-ion functionality (gate decompositions).
"""
from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    TYPE_CHECKING,
)

import stim

from qectostim.experiments.hardware_simulation.core.compiler import (
    HardwareCompiler,
    CompilationPass,
    CompilationResult,
    CompilationStage,
    DecompositionPass,
    MappingPass,
    RoutingPass,
    SchedulingPass,
)
from qectostim.experiments.hardware_simulation.core.pipeline import (
    NativeCircuit,
    MappedCircuit,
    RoutedCircuit,
    ScheduledCircuit,
    QubitMapping,
)
from qectostim.experiments.hardware_simulation.core.gates import (
    GateSpec,
    GateType,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
        TrappedIonArchitecture,
        WISEArchitecture,
    )
    from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
        WISERoutingConfig,
    )


# =============================================================================
# Gate Decomposition Data
# =============================================================================

# Standard angles
PI = math.pi
PI_2 = math.pi / 2
PI_4 = math.pi / 4


@dataclass
class DecomposedGate:
    """A gate in a decomposition sequence.
    
    Attributes
    ----------
    name : str
        Gate name (MS, RX, RY, RZ).
    qubits : Tuple[int, ...]
        Target qubits.
    params : Dict[str, float]
        Gate parameters (e.g., angle).
    """
    name: str
    qubits: Tuple[int, ...]
    params: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Abstract Base: TrappedIonCompiler
# =============================================================================

class TrappedIonCompiler(HardwareCompiler):
    """Abstract base compiler for trapped ion architectures.
    
    This class provides trapped-ion-specific functionality common to
    ALL trapped ion architectures:
    - MS gate decomposition recipes
    - Ion physics constraints
    - Native gate set definition
    
    Subclasses implement architecture-specific compilation:
    - WISECompiler: WISE grid with SAT-based optimal routing
    - LinearChainCompiler: Single chain, all-to-all connectivity
    - QCCDCompiler: General QCCD with T-junctions and shuttling
    
    This class is ABSTRACT - instantiate a concrete subclass.
    """
    
    # Native gate set for trapped ions
    NATIVE_GATES = ["MS", "RX", "RY", "RZ"]
    
    def __init__(
        self,
        architecture: "TrappedIonArchitecture",
        optimization_level: int = 1,
        use_global_rotations: bool = False,
    ):
        """Initialize the trapped ion compiler.
        
        Parameters
        ----------
        architecture : TrappedIonArchitecture
            Target trapped ion architecture.
        optimization_level : int
            0 = no optimization, 1 = basic, 2 = aggressive.
        use_global_rotations : bool
            If True, use global rotations where possible (all ions rotate).
        """
        super().__init__(architecture, optimization_level)
        self.use_global_rotations = use_global_rotations
        self._decomposition_cache: Dict[str, List[DecomposedGate]] = {}
    
    # =========================================================================
    # Common trapped-ion gate decompositions
    # =========================================================================
    
    def _decompose_cnot(self, control: int, target: int) -> List[DecomposedGate]:
        """Decompose CNOT to MS + single-qubit rotations.
        
        Uses the standard 2-MS decomposition.
        """
        return [
            DecomposedGate("RY", (target,), {"angle": -PI_2}),
            DecomposedGate("MS", (control, target), {"angle": PI_4}),
            DecomposedGate("RX", (control,), {"angle": PI_2}),
            DecomposedGate("RX", (target,), {"angle": PI_2}),
            DecomposedGate("MS", (control, target), {"angle": PI_4}),
            DecomposedGate("RY", (target,), {"angle": PI_2}),
        ]
    
    def _decompose_h(self, qubit: int) -> List[DecomposedGate]:
        """Decompose Hadamard to rotations. H = Ry(pi/2) . Rz(pi)"""
        return [
            DecomposedGate("RY", (qubit,), {"angle": PI_2}),
            DecomposedGate("RZ", (qubit,), {"angle": PI}),
        ]
    
    def _decompose_s(self, qubit: int) -> List[DecomposedGate]:
        """Decompose S gate. S = Rz(pi/2)"""
        return [DecomposedGate("RZ", (qubit,), {"angle": PI_2})]
    
    def _decompose_s_dag(self, qubit: int) -> List[DecomposedGate]:
        """Decompose S dagger gate. S_dag = Rz(-pi/2)"""
        return [DecomposedGate("RZ", (qubit,), {"angle": -PI_2})]
    
    def _decompose_t(self, qubit: int) -> List[DecomposedGate]:
        """Decompose T gate. T = Rz(pi/4)"""
        return [DecomposedGate("RZ", (qubit,), {"angle": PI_4})]
    
    def _decompose_t_dag(self, qubit: int) -> List[DecomposedGate]:
        """Decompose T dagger gate. T_dag = Rz(-pi/4)"""
        return [DecomposedGate("RZ", (qubit,), {"angle": -PI_4})]
    
    def _decompose_x(self, qubit: int) -> List[DecomposedGate]:
        """Decompose X gate. X = Rx(pi)"""
        return [DecomposedGate("RX", (qubit,), {"angle": PI})]
    
    def _decompose_y(self, qubit: int) -> List[DecomposedGate]:
        """Decompose Y gate. Y = Ry(pi)"""
        return [DecomposedGate("RY", (qubit,), {"angle": PI})]
    
    def _decompose_z(self, qubit: int) -> List[DecomposedGate]:
        """Decompose Z gate. Z = Rz(pi)"""
        return [DecomposedGate("RZ", (qubit,), {"angle": PI})]
    
    def _decompose_cz(self, control: int, target: int) -> List[DecomposedGate]:
        """Decompose CZ to CNOT + H. CZ = H_t . CNOT . H_t"""
        ops = []
        ops.extend(self._decompose_h(target))
        ops.extend(self._decompose_cnot(control, target))
        ops.extend(self._decompose_h(target))
        return ops
    
    def _decompose_swap(self, q1: int, q2: int) -> List[DecomposedGate]:
        """Decompose SWAP to 3 CNOTs. SWAP = CNOT_12 . CNOT_21 . CNOT_12"""
        ops = []
        ops.extend(self._decompose_cnot(q1, q2))
        ops.extend(self._decompose_cnot(q2, q1))
        ops.extend(self._decompose_cnot(q1, q2))
        return ops
    
    def _decompose_iswap(self, q1: int, q2: int) -> List[DecomposedGate]:
        """Decompose iSWAP. iSWAP can be done with 2 MS gates."""
        return [
            DecomposedGate("MS", (q1, q2), {"angle": PI_4}),
            DecomposedGate("MS", (q1, q2), {"angle": PI_4}),
        ]
    
    def decompose_stim_gate(
        self,
        gate_name: str,
        qubits: Tuple[int, ...],
    ) -> List[DecomposedGate]:
        """Decompose a Stim gate to native trapped ion gates."""
        # Check cache
        cache_key = f"{gate_name}_{qubits}"
        if cache_key in self._decomposition_cache:
            return self._decomposition_cache[cache_key]
        
        # Two-qubit gates
        if gate_name in ("CX", "CNOT", "ZCX"):
            result = self._decompose_cnot(qubits[0], qubits[1])
        elif gate_name in ("CZ", "ZCZ"):
            result = self._decompose_cz(qubits[0], qubits[1])
        elif gate_name == "SWAP":
            result = self._decompose_swap(qubits[0], qubits[1])
        elif gate_name == "ISWAP":
            result = self._decompose_iswap(qubits[0], qubits[1])
        # Single-qubit gates
        elif gate_name == "H":
            result = self._decompose_h(qubits[0])
        elif gate_name == "X":
            result = self._decompose_x(qubits[0])
        elif gate_name == "Y":
            result = self._decompose_y(qubits[0])
        elif gate_name == "Z":
            result = self._decompose_z(qubits[0])
        elif gate_name == "S":
            result = self._decompose_s(qubits[0])
        elif gate_name == "S_DAG":
            result = self._decompose_s_dag(qubits[0])
        elif gate_name == "T":
            result = self._decompose_t(qubits[0])
        elif gate_name == "T_DAG":
            result = self._decompose_t_dag(qubits[0])
        elif gate_name == "SQRT_X":
            result = [DecomposedGate("RX", (qubits[0],), {"angle": PI_2})]
        elif gate_name == "SQRT_X_DAG":
            result = [DecomposedGate("RX", (qubits[0],), {"angle": -PI_2})]
        elif gate_name == "SQRT_Y":
            result = [DecomposedGate("RY", (qubits[0],), {"angle": PI_2})]
        elif gate_name == "SQRT_Y_DAG":
            result = [DecomposedGate("RY", (qubits[0],), {"angle": -PI_2})]
        # Identity / reset / measure
        elif gate_name == "I":
            result = []
        elif gate_name in ("R", "RX", "RY", "RZ"):
            result = [DecomposedGate(gate_name, qubits, {})]
        elif gate_name in ("M", "MX", "MY", "MZ", "MR"):
            result = [DecomposedGate(gate_name, qubits, {})]
        else:
            raise ValueError(f"Unknown gate for decomposition: {gate_name}")
        
        self._decomposition_cache[cache_key] = result
        return result
    
    def get_native_gate_set(self) -> List[str]:
        """Return the native gate set for trapped ions."""
        return self.NATIVE_GATES.copy()
    
    # =========================================================================
    # Abstract methods - MUST be implemented by architecture-specific compilers
    # =========================================================================
    
    @abstractmethod
    def _setup_passes(self) -> None:
        """Set up architecture-specific compilation passes."""
        ...
    
    @abstractmethod
    def decompose_to_native(self, circuit: stim.Circuit) -> NativeCircuit:
        """Decompose input circuit to native MS + rotation gates."""
        ...
    
    @abstractmethod
    def map_qubits(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map logical qubits to physical ions."""
        ...
    
    @abstractmethod  
    def route(self, circuit: MappedCircuit) -> RoutedCircuit:
        """Route ions to satisfy gate locality constraints."""
        ...
    
    @abstractmethod
    def schedule(self, circuit: RoutedCircuit) -> ScheduledCircuit:
        """Schedule operations with timing and parallelization."""
        ...


# =============================================================================
# Concrete Compiler: WISECompiler
# =============================================================================

class WISECompiler(TrappedIonCompiler):
    """Compiler for WISE grid architecture.
    
    WISE uses a 2D grid of ion traps with SAT-based optimal routing.
    """
    
    def __init__(
        self,
        architecture: "WISEArchitecture",
        optimization_level: int = 1,
        use_global_rotations: bool = True,
        routing_config: Optional["WISERoutingConfig"] = None,
    ):
        super().__init__(architecture, optimization_level, use_global_rotations)
        self.routing_config = routing_config
        self._routing_pass: Optional[RoutingPass] = None
    
    def _setup_passes(self) -> None:
        """Set up WISE compilation passes."""
        pass
    
    def decompose_to_native(self, circuit: stim.Circuit) -> NativeCircuit:
        """Decompose circuit to native MS + rotation gates."""
        native_ops = []
        
        for instruction in circuit:
            gate_name = instruction.name
            targets = instruction.targets_copy()
            
            if gate_name.startswith("DETECTOR") or gate_name.startswith("OBSERVABLE"):
                continue
            if gate_name in ("TICK", "QUBIT_COORDS"):
                continue
            
            qubits = tuple(t.value for t in targets if t.is_qubit_target)
            if not qubits:
                continue
            
            decomposed = self.decompose_stim_gate(gate_name, qubits)
            native_ops.extend(decomposed)
        
        return NativeCircuit(
            operations=native_ops,
            num_qubits=circuit.num_qubits,
            metadata={"source": "stim", "compiler": "WISECompiler"},
        )
    
    def map_qubits(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map logical qubits to WISE grid positions."""
        num_qubits = circuit.num_qubits
        mapping = QubitMapping()
        for logical_q in range(num_qubits):
            mapping.assign(logical_q, logical_q)
        
        return MappedCircuit(
            native_circuit=circuit,
            mapping=mapping,
            metadata={"mapping_strategy": "sequential"},
        )
    
    def route(self, circuit: MappedCircuit) -> RoutedCircuit:
        """Route ions using SAT-based WISE routing."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WISERoutingPass,
            WISERoutingConfig,
        )
        
        config = self.routing_config or WISERoutingConfig()
        routing_pass = WISERoutingPass(
            architecture=self.architecture,
            config=config,
        )
        
        return routing_pass.route(circuit, self.architecture)
    
    def schedule(self, circuit: RoutedCircuit) -> ScheduledCircuit:
        """Schedule operations with WISE-specific parallelization."""
        from qectostim.experiments.hardware_simulation.trapped_ion.scheduling import (
            WISEBatchScheduler,
        )
        
        scheduler = WISEBatchScheduler()
        batches = scheduler.schedule(circuit.operations)
        
        return ScheduledCircuit(
            routed_circuit=circuit,
            batches=batches,
            total_duration=sum(b.duration for b in batches),
            metadata={"scheduler": "WISEBatchScheduler"},
        )


# =============================================================================
# Concrete Compiler: LinearChainCompiler
# =============================================================================

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
    
    def decompose_to_native(self, circuit: stim.Circuit) -> NativeCircuit:
        """Decompose circuit to native MS + rotation gates."""
        native_ops = []
        
        for instruction in circuit:
            gate_name = instruction.name
            targets = instruction.targets_copy()
            
            if gate_name in ("TICK", "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"):
                continue
            
            qubits = tuple(t.value for t in targets if t.is_qubit_target)
            if not qubits:
                continue
            
            decomposed = self.decompose_stim_gate(gate_name, qubits)
            native_ops.extend(decomposed)
        
        return NativeCircuit(
            operations=native_ops,
            num_qubits=circuit.num_qubits,
            metadata={"source": "stim", "compiler": "LinearChainCompiler"},
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
        """No routing needed for linear chain - all-to-all connectivity."""
        return RoutedCircuit(
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
        
        return ScheduledCircuit(
            routed_circuit=circuit,
            batches=batches,
            total_duration=sum(b.duration for b in batches),
            metadata={"scheduler": "GreedyBatchScheduler"},
        )


# =============================================================================
# Concrete Compiler: QCCDCompiler
# =============================================================================

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
    
    def decompose_to_native(self, circuit: stim.Circuit) -> NativeCircuit:
        """Decompose circuit to native MS + rotation gates."""
        native_ops = []
        
        for instruction in circuit:
            gate_name = instruction.name
            targets = instruction.targets_copy()
            
            if gate_name in ("TICK", "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"):
                continue
            
            qubits = tuple(t.value for t in targets if t.is_qubit_target)
            if not qubits:
                continue
            
            decomposed = self.decompose_stim_gate(gate_name, qubits)
            native_ops.extend(decomposed)
        
        return NativeCircuit(
            operations=native_ops,
            num_qubits=circuit.num_qubits,
            metadata={"source": "stim", "compiler": "QCCDCompiler"},
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
        """Route ions between zones using split/merge/shuttle."""
        return RoutedCircuit(
            mapped_circuit=circuit,
            routing_operations=[],
            metadata={"routing_strategy": "qccd_placeholder"},
        )
    
    def schedule(self, circuit: RoutedCircuit) -> ScheduledCircuit:
        """Schedule operations for QCCD architecture."""
        from qectostim.experiments.hardware_simulation.core.operations import (
            GreedyBatchScheduler,
        )
        
        scheduler = GreedyBatchScheduler()
        batches = scheduler.schedule(circuit.operations)
        
        return ScheduledCircuit(
            routed_circuit=circuit,
            batches=batches,
            total_duration=sum(b.duration for b in batches),
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
