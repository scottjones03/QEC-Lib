# src/qectostim/experiments/hardware_simulation/core/compiler.py
"""
Abstract hardware compiler interfaces.

Defines the compilation pipeline that transforms logical circuits
into hardware-executable sequences of physical operations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    Callable,
    Type,
    TYPE_CHECKING,
)

import stim

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.architecture import HardwareArchitecture
    from qectostim.experiments.hardware_simulation.core.pipeline import (
        NativeCircuit,
        MappedCircuit,
        RoutedCircuit,
        ScheduledCircuit,
        CompiledCircuit,
        QubitMapping,
    )
    from qectostim.experiments.hardware_simulation.core.gates import NativeGateSet


@dataclass
class CompilationConfig:
    """Configuration for compilation.
    
    Attributes
    ----------
    optimization_level : int
        Optimization aggressiveness (0=none, 1=basic, 2=aggressive).
    allow_approximations : bool
        Allow gate approximations (e.g., for non-Clifford gates).
    max_routing_depth : int
        Maximum routing iterations before failing.
    preserve_barriers : bool
        Keep barrier instructions through compilation.
    target_gate_set : Optional[List[str]]
        Target gate set for decomposition (uses native if None).
    """
    optimization_level: int = 1
    allow_approximations: bool = True
    max_routing_depth: int = 100
    preserve_barriers: bool = True
    target_gate_set: Optional[List[str]] = None


class CompilationStage(Enum):
    """Stages of the compilation pipeline."""
    PARSE = auto()          # Parse input circuit
    DECOMPOSE = auto()      # Decompose to native gates
    MAP = auto()            # Map logical to physical qubits
    ROUTE = auto()          # Insert routing operations
    SCHEDULE = auto()       # Assign timing and parallelism
    OPTIMIZE = auto()       # Post-compilation optimization
    GENERATE = auto()       # Generate output (Stim circuit)


@dataclass
class CompilationResult:
    """Result of a compilation pass.
    
    Attributes
    ----------
    success : bool
        Whether compilation succeeded.
    stage : CompilationStage
        Stage that produced this result.
    data : Any
        Stage-specific output data.
    metrics : Dict[str, Any]
        Compilation metrics from this stage.
    warnings : List[str]
        Non-fatal warnings.
    errors : List[str]
        Error messages (if success=False).
    """
    success: bool = True
    stage: CompilationStage = CompilationStage.PARSE
    data: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class CompilationPass(ABC):
    """Abstract base class for compilation passes.
    
    A compilation pass transforms the circuit representation
    from one form to another as part of the compilation pipeline.
    """
    
    def __init__(self, name: str, stage: CompilationStage):
        self.name = name
        self.stage = stage
    
    @abstractmethod
    def run(self, input_data: Any, context: "CompilationContext") -> CompilationResult:
        """Execute this compilation pass.
        
        Parameters
        ----------
        input_data : Any
            Input from previous pass.
        context : CompilationContext
            Compilation context with architecture and settings.
            
        Returns
        -------
        CompilationResult
            Result including transformed data.
        """
        ...
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, stage={self.stage})"


class DecompositionPass(CompilationPass):
    """Abstract pass for decomposing gates to native set.
    
    Transforms a Stim circuit into a NativeCircuit where all
    gates are from the hardware's native gate set.
    """
    
    def __init__(self, name: str = "decomposition"):
        super().__init__(name, CompilationStage.DECOMPOSE)
    
    @abstractmethod
    def decompose_gate(
        self,
        gate_name: str,
        qubits: Tuple[int, ...],
        native_gates: "NativeGateSet",
    ) -> List[Tuple[str, Tuple[int, ...]]]:
        """Decompose a single gate to native gates.
        
        Parameters
        ----------
        gate_name : str
            Name of the gate to decompose.
        qubits : Tuple[int, ...]
            Qubits the gate acts on.
        native_gates : NativeGateSet
            Available native gates.
            
        Returns
        -------
        List[Tuple[str, Tuple[int, ...]]]
            Sequence of (native_gate_name, qubits) pairs.
        """
        ...


class MappingPass(CompilationPass):
    """Abstract pass for logical-to-physical qubit mapping.
    
    Assigns logical qubits to physical qubits on the hardware,
    considering connectivity and code structure.
    """
    
    def __init__(self, name: str = "mapping"):
        super().__init__(name, CompilationStage.MAP)
    
    @abstractmethod
    def compute_initial_mapping(
        self,
        circuit: "NativeCircuit",
        architecture: "HardwareArchitecture",
    ) -> "QubitMapping":
        """Compute initial qubit placement.
        
        Parameters
        ----------
        circuit : NativeCircuit
            Circuit to map.
        architecture : HardwareArchitecture
            Target hardware.
            
        Returns
        -------
        QubitMapping
            Initial logical-to-physical mapping.
        """
        ...


class RoutingPass(CompilationPass):
    """Abstract pass for qubit routing.
    
    Inserts SWAP or transport operations to satisfy connectivity
    constraints of the hardware.
    """
    
    def __init__(self, name: str = "routing"):
        super().__init__(name, CompilationStage.ROUTE)
    
    @abstractmethod
    def route(
        self,
        circuit: "MappedCircuit",
        architecture: "HardwareArchitecture",
    ) -> "RoutedCircuit":
        """Route qubits to satisfy connectivity.
        
        Parameters
        ----------
        circuit : MappedCircuit
            Mapped circuit needing routing.
        architecture : HardwareArchitecture
            Target hardware with connectivity constraints.
            
        Returns
        -------
        RoutedCircuit
            Circuit with routing operations inserted.
        """
        ...


class SchedulingPass(CompilationPass):
    """Abstract pass for operation scheduling.
    
    Assigns timing and parallelization to operations,
    respecting hardware constraints.
    """
    
    def __init__(self, name: str = "scheduling"):
        super().__init__(name, CompilationStage.SCHEDULE)
    
    @abstractmethod
    def schedule(
        self,
        circuit: "RoutedCircuit",
        architecture: "HardwareArchitecture",
    ) -> "ScheduledCircuit":
        """Schedule operations with timing info.
        
        Parameters
        ----------
        circuit : RoutedCircuit
            Circuit to schedule.
        architecture : HardwareArchitecture
            Hardware with timing constraints.
            
        Returns
        -------
        ScheduledCircuit
            Circuit with timing and parallel groups.
        """
        ...


@dataclass
class CompilationContext:
    """Context for compilation passes.
    
    Carries architecture, settings, and intermediate state
    through the compilation pipeline.
    
    Attributes
    ----------
    architecture : HardwareArchitecture
        Target hardware architecture.
    optimization_level : int
        Optimization aggressiveness (0=none, 1=basic, 2=full).
    settings : Dict[str, Any]
        Additional compilation settings.
    intermediate_results : Dict[CompilationStage, CompilationResult]
        Results from completed stages.
    """
    architecture: "HardwareArchitecture"
    optimization_level: int = 1
    settings: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[CompilationStage, CompilationResult] = field(default_factory=dict)
    
    def get_result(self, stage: CompilationStage) -> Optional[CompilationResult]:
        """Get result from a completed stage."""
        return self.intermediate_results.get(stage)
    
    def set_result(self, result: CompilationResult) -> None:
        """Store result from a stage."""
        self.intermediate_results[result.stage] = result


class RoutingStrategy(Enum):
    """Strategies for routing qubits to satisfy connectivity.
    
    Technology-agnostic routing strategies. The actual implementation
    depends on the platform capabilities (fixed vs. mobile qubits).
    
    Strategies:
    - LOCAL_EXCHANGE: Exchange-based routing (SWAPs on fixed, local moves on mobile)
    - GLOBAL_OPTIMIZATION: SAT/ILP-based optimal routing
    - GREEDY: Fast greedy heuristic
    - LOOKAHEAD: Consider future gates for better routing decisions
    - HIERARCHICAL: Multi-level routing (coarse then fine)
    """
    LOCAL_EXCHANGE = auto()    # Local qubit exchanges (SWAP or transport)
    GLOBAL_OPTIMIZATION = auto()  # SAT/ILP-based global optimization
    GREEDY = auto()            # Fast greedy routing
    LOOKAHEAD = auto()         # Consider future gates
    HIERARCHICAL = auto()      # Multi-level (coarse-to-fine) routing


@dataclass
class RoutingResult:
    """Result of a routing operation.
    
    Attributes
    ----------
    success : bool
        Whether routing succeeded.
    operations : List[Any]
        Routing operations (SWAPs or transports).
    cost : float
        Total routing cost (time, gate count, etc.).
    final_mapping : Optional[QubitMapping]
        Qubit mapping after routing.
    metrics : Dict[str, Any]
        Routing metrics (depth, swap count, etc.).
    """
    success: bool = True
    operations: List[Any] = field(default_factory=list)
    cost: float = 0.0
    final_mapping: Optional["QubitMapping"] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Cost Model Abstraction
# =============================================================================

class CostModel(ABC):
    """Abstract cost model for technology-agnostic routing optimization.
    
    Different hardware platforms have different notions of "cost":
    - Superconducting: SWAP count, circuit depth
    - Trapped ion: Transport time, motional heating
    - Neutral atom: Atom move distance, atom loss probability
    
    By abstracting cost, routing algorithms can optimize without
    knowing platform-specific details.
    
    Example
    -------
    >>> class TrappedIonCostModel(CostModel):
    ...     def operation_cost(self, op_type, qubits, params):
    ...         if op_type == "transport":
    ...             return params.get("distance", 1.0) * self.transport_time_per_unit
    ...         return self.gate_times.get(op_type, 0.0)
    """
    
    @abstractmethod
    def operation_cost(
        self,
        operation_type: str,
        qubits: Tuple[int, ...],
        params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Estimate cost of a single operation.
        
        Parameters
        ----------
        operation_type : str
            Type of operation (e.g., "swap", "transport", "gate").
        qubits : Tuple[int, ...]
            Qubits involved in the operation.
        params : Optional[Dict]
            Operation-specific parameters.
            
        Returns
        -------
        float
            Cost value (interpretation depends on platform).
        """
        ...
    
    @abstractmethod
    def sequence_cost(
        self,
        operations: List[Tuple[str, Tuple[int, ...], Optional[Dict[str, Any]]]],
    ) -> float:
        """Estimate total cost of an operation sequence.
        
        May account for parallelism, overhead, etc.
        
        Parameters
        ----------
        operations : List
            List of (operation_type, qubits, params) tuples.
            
        Returns
        -------
        float
            Total cost.
        """
        ...
    
    def compare(self, cost_a: float, cost_b: float) -> int:
        """Compare two costs.
        
        Returns
        -------
        int
            -1 if a < b (a is better), 0 if equal, 1 if a > b.
        """
        if cost_a < cost_b:
            return -1
        elif cost_a > cost_b:
            return 1
        return 0
    
    def is_acceptable(self, cost: float) -> bool:
        """Check if a cost is acceptable (not too expensive).
        
        Can be overridden to implement cost thresholds.
        """
        return True


class UniformCostModel(CostModel):
    """Simple cost model where all operations have unit cost.
    
    Useful for testing and when minimizing operation count.
    """
    
    def operation_cost(
        self,
        operation_type: str,
        qubits: Tuple[int, ...],
        params: Optional[Dict[str, Any]] = None,
    ) -> float:
        return 1.0
    
    def sequence_cost(
        self,
        operations: List[Tuple[str, Tuple[int, ...], Optional[Dict[str, Any]]]],
    ) -> float:
        return float(len(operations))


class Router(ABC):
    """Abstract base class for qubit routing algorithms.
    
    Routers compute sequences of operations to move qubits into
    positions where required gates can be executed.
    
    This is separated from RoutingPass to allow:
    - Multiple routing strategies per platform
    - Strategy selection at runtime
    - Easy experimentation with new routing algorithms
    
    Technology-agnostic: uses CostModel for cost estimation.
    Platform implementations provide their own cost models.
    """
    
    def __init__(
        self,
        strategy: RoutingStrategy,
        cost_model: Optional[CostModel] = None,
        name: str = "router",
    ):
        self.strategy = strategy
        self.cost_model = cost_model or UniformCostModel()
        self.name = name
    
    @abstractmethod
    def route_gate(
        self,
        gate_qubits: Tuple[int, ...],
        current_mapping: "QubitMapping",
        architecture: "HardwareArchitecture",
    ) -> RoutingResult:
        """Route qubits for a single gate.
        
        Parameters
        ----------
        gate_qubits : Tuple[int, ...]
            Logical qubits that need to interact.
        current_mapping : QubitMapping
            Current logical-to-physical mapping.
        architecture : HardwareArchitecture
            Target hardware.
            
        Returns
        -------
        RoutingResult
            Operations to bring qubits together.
        """
        ...
    
    def route_batch(
        self,
        gate_pairs: List[Tuple[int, int]],
        current_mapping: "QubitMapping",
        architecture: "HardwareArchitecture",
    ) -> RoutingResult:
        """Route qubits for a batch of two-qubit gates.
        
        Default implementation routes gates sequentially.
        Override for batch-optimized routing (e.g., SAT-based).
        
        Parameters
        ----------
        gate_pairs : List[Tuple[int, int]]
            Pairs of logical qubits that need to interact.
        current_mapping : QubitMapping
            Current logical-to-physical mapping.
        architecture : HardwareArchitecture
            Target hardware.
            
        Returns
        -------
        RoutingResult
            Combined routing operations.
        """
        all_operations = []
        total_cost = 0.0
        mapping = current_mapping.copy()
        
        for pair in gate_pairs:
            result = self.route_gate(pair, mapping, architecture)
            if not result.success:
                return RoutingResult(
                    success=False,
                    metrics={"failed_at": pair},
                )
            all_operations.extend(result.operations)
            total_cost += result.cost
            if result.final_mapping:
                mapping = result.final_mapping
        
        return RoutingResult(
            success=True,
            operations=all_operations,
            cost=total_cost,
            final_mapping=mapping,
        )
    
    def supports_batch_routing(self) -> bool:
        """Check if this router has optimized batch routing.
        
        Returns True if route_batch is overridden with a better
        algorithm than sequential routing.
        """
        return False
    
    def estimate_cost(
        self,
        gate_qubits: Tuple[int, ...],
        current_mapping: "QubitMapping",
        architecture: "HardwareArchitecture",
    ) -> float:
        """Estimate routing cost without computing full solution.
        
        Used for lookahead and heuristic routing decisions.
        """
        # Default: compute full route and return cost
        result = self.route_gate(gate_qubits, current_mapping, architecture)
        return result.cost if result.success else float('inf')
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(strategy={self.strategy.name})"


class SwapRouter(Router):
    """Router that inserts SWAP gates for fixed connectivity.
    
    Used primarily for superconducting architectures where qubits
    cannot move and connectivity is determined by physical couplers.
    """
    
    def __init__(self, name: str = "swap_router"):
        super().__init__(RoutingStrategy.SWAP_BASED, name)
    
    def route_gate(
        self,
        gate_qubits: Tuple[int, ...],
        current_mapping: "QubitMapping",
        architecture: "HardwareArchitecture",
    ) -> RoutingResult:
        """Route using SWAP insertion along shortest path."""
        if len(gate_qubits) != 2:
            # Single qubit gates don't need routing
            return RoutingResult(success=True, final_mapping=current_mapping)
        
        q1, q2 = gate_qubits
        p1 = current_mapping.get_physical(q1)
        p2 = current_mapping.get_physical(q2)
        
        if p1 is None or p2 is None:
            return RoutingResult(
                success=False,
                metrics={"error": "Unmapped qubits"},
            )
        
        # Check if already connected
        if architecture.can_interact(p1, p2):
            return RoutingResult(success=True, final_mapping=current_mapping)
        
        # Find shortest path and insert SWAPs
        graph = architecture.connectivity_graph()
        try:
            path = graph.shortest_path(str(p1), str(p2))
        except Exception:
            return RoutingResult(
                success=False,
                metrics={"error": f"No path between {p1} and {p2}"},
            )
        
        # Insert SWAPs along path
        swaps = []
        mapping = current_mapping.copy()
        
        for i in range(len(path) - 2):
            swap_p1 = int(path[i])
            swap_p2 = int(path[i + 1])
            swaps.append(("SWAP", (swap_p1, swap_p2)))
            mapping.swap_physical(swap_p1, swap_p2)
        
        return RoutingResult(
            success=True,
            operations=swaps,
            cost=len(swaps),
            final_mapping=mapping,
            metrics={"swap_count": len(swaps), "path_length": len(path)},
        )


class TransportRouter(Router):
    """Router for reconfigurable architectures with qubit transport.
    
    Used for trapped ion and neutral atom architectures where
    qubits can be physically moved between zones.
    """
    
    def __init__(self, name: str = "transport_router"):
        super().__init__(RoutingStrategy.TRANSPORT_BASED, name)
    
    def route_gate(
        self,
        gate_qubits: Tuple[int, ...],
        current_mapping: "QubitMapping",
        architecture: "HardwareArchitecture",
    ) -> RoutingResult:
        """Route by transporting qubits to a common zone."""
        from qectostim.experiments.hardware_simulation.core.architecture import (
            ReconfigurableArchitecture,
        )
        
        if not isinstance(architecture, ReconfigurableArchitecture):
            return RoutingResult(
                success=False,
                metrics={"error": "Transport routing requires ReconfigurableArchitecture"},
            )
        
        if len(gate_qubits) != 2:
            return RoutingResult(success=True, final_mapping=current_mapping)
        
        q1, q2 = gate_qubits
        zone1 = architecture.current_zone(q1)
        zone2 = architecture.current_zone(q2)
        
        if zone1 == zone2:
            # Already colocated
            return RoutingResult(success=True, final_mapping=current_mapping)
        
        # Compute transport cost to bring q2 to q1's zone
        cost = architecture.transport_cost(zone2, zone1, num_qubits=1)
        
        # Generate transport plan
        target_layout = {q1: zone1, q2: zone1}
        plan = architecture.reconfiguration_plan(target_layout)
        
        return RoutingResult(
            success=True,
            operations=plan.operations,
            cost=cost.time_us,
            final_mapping=current_mapping,  # Logical mapping unchanged
            metrics={
                "transport_time_us": cost.time_us,
                "heating_added": cost.heating_added,
            },
        )


class HardwareCompiler(ABC):
    """Abstract base class for hardware compilers.
    
    Orchestrates the compilation pipeline to transform logical
    circuits into hardware-executable sequences.
    
    Subclasses implement platform-specific compilation logic:
    - TrappedIonCompiler: Ion routing, chain management
    - SuperconductingCompiler: SWAP routing on fixed connectivity
    - NeutralAtomCompiler: Atom rearrangement, zone-based compilation
    """
    
    def __init__(
        self,
        architecture: "HardwareArchitecture",
        optimization_level: int = 1,
    ):
        """Initialize the compiler.
        
        Parameters
        ----------
        architecture : HardwareArchitecture
            Target hardware architecture.
        optimization_level : int
            Optimization level (0-2).
        """
        self.architecture = architecture
        self.optimization_level = optimization_level
        self._passes: List[CompilationPass] = []
        
        # Set up default compilation passes
        self._setup_passes()
    
    @abstractmethod
    def _setup_passes(self) -> None:
        """Set up the compilation passes.
        
        Subclasses should populate self._passes with appropriate
        DecompositionPass, MappingPass, RoutingPass, SchedulingPass.
        """
        ...
    
    @abstractmethod
    def decompose_to_native(self, circuit: stim.Circuit) -> "NativeCircuit":
        """Decompose circuit to native gates.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Input Stim circuit.
            
        Returns
        -------
        NativeCircuit
            Circuit with native gates only.
        """
        ...
    
    @abstractmethod
    def map_qubits(self, circuit: "NativeCircuit") -> "MappedCircuit":
        """Map logical to physical qubits.
        
        Parameters
        ----------
        circuit : NativeCircuit
            Native circuit.
            
        Returns
        -------
        MappedCircuit
            Circuit with qubit mapping.
        """
        ...
    
    @abstractmethod
    def route(self, circuit: "MappedCircuit") -> "RoutedCircuit":
        """Insert routing operations.
        
        Parameters
        ----------
        circuit : MappedCircuit
            Mapped circuit.
            
        Returns
        -------
        RoutedCircuit
            Circuit with routing inserted.
        """
        ...
    
    @abstractmethod
    def schedule(self, circuit: "RoutedCircuit") -> "ScheduledCircuit":
        """Schedule operations with timing.
        
        Parameters
        ----------
        circuit : RoutedCircuit
            Routed circuit.
            
        Returns
        -------
        ScheduledCircuit
            Scheduled circuit.
        """
        ...
    
    def compile(self, circuit: stim.Circuit) -> "CompiledCircuit":
        """Run the full compilation pipeline.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Input Stim circuit to compile.
            
        Returns
        -------
        CompiledCircuit
            Fully compiled circuit ready for simulation.
        """
        from qectostim.experiments.hardware_simulation.core.pipeline import (
            CompiledCircuit,
        )
        
        # Create compilation context
        context = CompilationContext(
            architecture=self.architecture,
            optimization_level=self.optimization_level,
        )
        
        # Run pipeline stages
        native = self.decompose_to_native(circuit)
        mapped = self.map_qubits(native)
        routed = self.route(mapped)
        scheduled = self.schedule(routed)
        
        # Build final compiled circuit
        compiled = CompiledCircuit(
            scheduled=scheduled,
            mapping=routed.final_mapping or mapped.mapping,
        )
        
        # Compute metrics
        compiled.compute_metrics()
        
        return compiled
    
    def add_pass(self, pass_: CompilationPass) -> None:
        """Add a compilation pass to the pipeline."""
        self._passes.append(pass_)
    
    def get_passes(self) -> List[CompilationPass]:
        """Get all compilation passes."""
        return list(self._passes)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(architecture={self.architecture.name!r})"


# Convenience type aliases
CompilerFactory = Callable[["HardwareArchitecture"], HardwareCompiler]
