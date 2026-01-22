# src/qectostim/experiments/hardware_simulation/core/__init__.py
"""
Core abstract interfaces for hardware simulation.

This module defines the abstract base classes that all platform-specific
implementations must inherit from. The design separates concerns into:

1. Architecture: Hardware topology, native gates, physical constraints
2. Components: Physical elements (qubits, zones, couplers)
3. Operations: Physical actions with timing and fidelity models
4. Compiler: Circuit transformation pipeline
5. Pipeline: Data structures for compilation stages

IMPORTANT: This module is technology-agnostic. No platform-specific
terminology (ions, atoms, transmons, etc.) should appear here.
"""

from qectostim.experiments.hardware_simulation.core.architecture import (
    HardwareArchitecture,
    ReconfigurableArchitecture,
    ConnectivityGraph,
    Zone,
    ZoneType,
    PhysicalConstraints,
    TransportCost,
    ReconfigurationPlan,
    LayoutTracker,
    LayoutSnapshot,
    # Technology-agnostic capability protocols
    PositionModel,
    PlatformCapabilities,
    InteractionModel,
    CalibrationData,
    # Grid layout abstractions
    GridPosition,
    DiscreteLayout,
)
from qectostim.experiments.hardware_simulation.core.components import (
    HardwareComponent,
    PhysicalQubit,
    QubitState,
    Coupler,
)
from qectostim.experiments.hardware_simulation.core.operations import (
    # Base operation classes
    PhysicalOperation,
    GateOperation,
    TransportOperation,
    MeasurementOperation,
    IdleOperation,
    ResetOperation,
    BarrierOperation,
    # New platform-agnostic operation types
    GlobalOperation,
    LossOperation,
    RearrangementOperation,
    # Result and type enums
    OperationResult,
    OperationType,
    # Batching and scheduling
    OperationBatch,
    BatchScheduler,
    GreedyBatchScheduler,
    DependencyEdge,
    # Error model protocol
    TransportErrorModel,
)
from qectostim.experiments.hardware_simulation.core.compiler import (
    HardwareCompiler,
    CompilationPass,
    CompilationConfig,
    DecompositionPass,
    MappingPass,
    RoutingPass,
    SchedulingPass,
    Router,
    RoutingStrategy,
    RoutingResult,
    # Technology-agnostic cost model
    CostModel,
    UniformCostModel,
)
from qectostim.experiments.hardware_simulation.core.pipeline import (
    CompiledCircuit,
    NativeCircuit,
    MappedCircuit,
    RoutedCircuit,
    ScheduledCircuit,
    ScheduledOperation,
    QubitMapping,
)
from qectostim.experiments.hardware_simulation.core.gates import (
    GateSpec,
    GateType,
    NativeGateSet,
    GateDecomposition,
)
from qectostim.experiments.hardware_simulation.core.execution import (
    ExecutionPlan,
    OperationTiming,
    GateSwapInfo,
    IdleInterval,
    build_execution_plan_from_compiled,
)
from qectostim.experiments.hardware_simulation.core.sat_interface import (
    # Core configuration
    SATRoutingConfig,
    PatchRoutingConfig,
    # Core types
    ConstraintType,
    Constraint,
    SATSolution,
    PlacementRequirement,
    # Abstract encoders
    SATEncoder,
    GridSATEncoder,
    # Solver protocol
    SATSolverProtocol,
    # Utilities
    manhattan_distance,
    are_adjacent,
    grid_neighbors,
)

__all__ = [
    # Architecture
    "HardwareArchitecture",
    "ReconfigurableArchitecture",
    "ConnectivityGraph", 
    "Zone",
    "ZoneType",
    "PhysicalConstraints",
    "TransportCost",
    "ReconfigurationPlan",
    "LayoutTracker",
    "LayoutSnapshot",
    # Technology-agnostic capability protocols
    "PositionModel",
    "PlatformCapabilities",
    "InteractionModel",
    "CalibrationData",
    # Grid layout
    "GridPosition",
    "DiscreteLayout",
    # Components
    "HardwareComponent",
    "PhysicalQubit",
    "QubitState",
    "Coupler",
    # Operations - base classes
    "PhysicalOperation",
    "GateOperation",
    "TransportOperation",
    "MeasurementOperation",
    "IdleOperation",
    "ResetOperation",
    "BarrierOperation",
    # Operations - platform-agnostic types
    "GlobalOperation",
    "LossOperation",
    "RearrangementOperation",
    # Operations - results and enums
    "OperationResult",
    "OperationType",
    # Operations - batching
    "OperationBatch",
    "BatchScheduler",
    "GreedyBatchScheduler",
    "DependencyEdge",
    # Operations - error models
    "TransportErrorModel",
    # Compiler
    "HardwareCompiler",
    "CompilationPass",
    "CompilationConfig",
    "DecompositionPass",
    "MappingPass",
    "RoutingPass",
    "SchedulingPass",
    "Router",
    "RoutingStrategy",
    "RoutingResult",
    # Cost model
    "CostModel",
    "UniformCostModel",
    # Pipeline
    "CompiledCircuit",
    "NativeCircuit",
    "MappedCircuit",
    "RoutedCircuit",
    "ScheduledCircuit",
    "ScheduledOperation",
    "QubitMapping",
    # Gates
    "GateSpec",
    "GateType",
    "NativeGateSet",
    "GateDecomposition",
    # Execution planning
    "ExecutionPlan",
    "OperationTiming",
    "GateSwapInfo",
    "IdleInterval",
    "build_execution_plan_from_compiled",
    # SAT interface - configuration
    "SATRoutingConfig",
    "PatchRoutingConfig",
    # SAT interface - types
    "ConstraintType",
    "Constraint",
    "SATSolution",
    "PlacementRequirement",
    "SATEncoder",
    "GridSATEncoder",
    "SATSolverProtocol",
    "manhattan_distance",
    "are_adjacent",
    "grid_neighbors",
]
