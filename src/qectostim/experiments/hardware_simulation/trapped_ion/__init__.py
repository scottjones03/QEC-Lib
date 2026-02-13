# src/qectostim/experiments/hardware_simulation/trapped_ion/__init__.py
"""
Trapped ion hardware simulation.

Provides simulation capabilities for trapped ion quantum computers,
including QCCD (Quantum Charge-Coupled Device) and linear chain architectures.

Platform Characteristics:
- High two-qubit gate fidelity (via Mølmer-Sørensen gates)
- All-to-all connectivity within ion chains
- Qubit transport via shuttling between zones
- Long coherence times (T2 ~ seconds)
- Motional heating affects gate fidelity

Architectures:
- QCCDArchitecture: Multi-zone trap with ion shuttling
- LinearChainArchitecture: Single linear trap

Components:
- Ion types: QubitIon, CoolingIon, SpectatorIon
- Trap types: ManipulationTrap, StorageTrap, Junction
- Transport: Crossing (connection between nodes)

Operations:
- Transport: Split, Merge, Move, JunctionCrossing
- Crystal: CrystalRotation, SympatheticCooling
- Quantum: SingleQubitGate, MSGate, Measurement, QubitReset

Gates:
- MS: Mølmer-Sørensen entangling gate
- R: Single-ion rotation
- GR: Global rotation (all ions)
"""

from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
    TrappedIonArchitecture,
    QCCDArchitecture,
    WISEArchitecture,
    LinearChainArchitecture,
    AugmentedGridArchitecture,
    NetworkedGridArchitecture,
)
from qectostim.experiments.hardware_simulation.trapped_ion.compiler import (
    TrappedIonCompiler,
    WISECompiler,
    QCCDCompiler,
    LinearChainCompiler,
    AugmentedGridCompiler,
    NetworkedGridCompiler,
    DecomposedGate,
)
from qectostim.experiments.hardware_simulation.trapped_ion.simulator import (
    TrappedIonSimulator,
)
from qectostim.experiments.hardware_simulation.trapped_ion.experiments import (
    TrappedIonExperiment,
    TrappedIonGadgetExperiment,
)
from qectostim.experiments.hardware_simulation.trapped_ion.noise import (
    TrappedIonNoiseModel,
    TrappedIonCalibration,
)
from qectostim.experiments.hardware_simulation.trapped_ion.execution import (
    TrappedIonExecutionPlanner,
    create_simple_execution_plan,
    DEFAULT_GATE_TIMES,
    DEFAULT_FIDELITIES,
)
from qectostim.experiments.hardware_simulation.trapped_ion.gates import (
    TrappedIonGateSet,
)
from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
    # Operation types
    QCCDOperationType as QCCDOperation,
    # Ion types
    Ion,
    QubitIon,
    CoolingIon,
    SpectatorIon,
    # Node types
    QCCDNode,
    Junction,
    ManipulationTrap,
    StorageTrap,
    # Connection
    Crossing,
    # Configuration
    QCCDWISEConfig,
    LinearChainConfig,
    # Factories
    create_qubit_ion,
    create_cooling_ion,
)
from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
    # Physical constants & species
    PhysicalConstants,
    CONSTANTS,
    IonSpecies,
    YB171,
    BA137,
    CA40,
    TrapParameters,
    DEFAULT_TRAP,
    # Calibration
    CalibrationConstants,
    DEFAULT_CALIBRATION,
    # Fidelity model
    IonChainFidelityModel,
    DEFAULT_FIDELITY_MODEL,
    # Mode structure
    ModeStructure,
    ModeSnapshot,
    # Legacy timing/heating constants (deprecated — use CalibrationConstants)
    TrappedIonTimings,
    HeatingRates,
    DEFAULT_TIMINGS,
    DEFAULT_HEATING_RATES,
)

# Backward compatibility aliases
Trap = ManipulationTrap
create_manipulation_trap = ManipulationTrap
create_storage_trap = StorageTrap

from qectostim.experiments.hardware_simulation.trapped_ion.operations import (
    # Base classes
    QCCDOperationBase,
    TransportOperation,
    CrystalOperation,
    QubitOperation,
    OperationResult,
    # Transport operations
    Split,
    Merge,
    Move,
    JunctionCrossing,
    # Crystal operations
    CrystalRotation,
    SympatheticCooling,
    # Qubit operations
    SingleQubitGate,
    MSGate,
    GateSwap,
    Measurement,
    QubitReset,
    # Batch reconfiguration
    ReconfigurationStep,
    GlobalReconfiguration,
    # Factories
    create_split,
    create_merge,
    create_ms_gate,
    create_single_qubit_gate,
    create_measurement,
)
from qectostim.experiments.hardware_simulation.trapped_ion.scheduling import (
    # Schedulers
    WISEBatchScheduler,
    WISECriticalPathScheduler,
    BarrierAwareScheduler,
    # Scheduling functions
    schedule_operations_wise,
    schedule_operations_with_barriers,
    build_happens_before_dag,
    # Dephasing
    calculate_dephasing_fidelity,
    calculate_dephasing_from_idling,
    # Data structures
    ParallelOperationGroup,
    # Constants
    T2_DEPHASING_TIME,
)
from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
    # SAT Encoder
    WISESATEncoder,
    WISESATContext,
    # Routers
    WiseSatRouter,
    WisePatchRouter,
    GreedyIonRouter,
    # Compilation Pass
    WISERoutingPass,
    # Cost Model
    WISECostModel,
    # Configuration
    WISERoutingConfig,
    # Data structures
    GridLayout,
    RoutingPass,
    RoutingSchedule,
    GatePairRequirement,
    # Utilities
    compute_target_positions,
)
from qectostim.experiments.hardware_simulation.trapped_ion.clustering import (
    regular_partition,
    arrange_clusters,
    hill_climb_on_arrange_clusters,
    merge_clusters_to_limit,
)
from qectostim.experiments.hardware_simulation.trapped_ion.viz import (
    display_architecture,
    animate_transport,
    visualize_reconfiguration,
)

__all__ = [
    # Architecture
    "TrappedIonArchitecture",
    "QCCDArchitecture",
    "WISEArchitecture",
    "LinearChainArchitecture",
    "AugmentedGridArchitecture",
    "NetworkedGridArchitecture",
    # Compiler
    "TrappedIonCompiler",
    "WISECompiler",
    "QCCDCompiler",
    "LinearChainCompiler",
    "AugmentedGridCompiler",
    "NetworkedGridCompiler",
    "DecomposedGate",
    # Simulator
    "TrappedIonSimulator",
    # Experiments
    "TrappedIonExperiment",
    "TrappedIonGadgetExperiment",
    # Noise
    "TrappedIonNoiseModel",
    "TrappedIonCalibration",
    # Execution Planning
    "TrappedIonExecutionPlanner",
    "create_simple_execution_plan",
    "DEFAULT_GATE_TIMES",
    "DEFAULT_FIDELITIES",
    # Gates
    "TrappedIonGateSet",
    # Components - Operation enum
    "QCCDOperation",
    # Components - Constants
    "TrappedIonTimings",
    "HeatingRates",
    "DEFAULT_TIMINGS",
    "DEFAULT_HEATING_RATES",
    # Components - Ions
    "Ion",
    "QubitIon",
    "CoolingIon",
    "SpectatorIon",
    # Components - Nodes
    "QCCDNode",
    "Junction",
    "Trap",
    "ManipulationTrap",
    "StorageTrap",
    "Crossing",
    # Components - Config
    "QCCDWISEConfig",
    "LinearChainConfig",
    # Components - Factories
    "create_qubit_ion",
    "create_cooling_ion",
    "create_manipulation_trap",
    "create_storage_trap",
    # Operations - Base
    "QCCDOperationBase",
    "TransportOperation",
    "CrystalOperation",
    "QubitOperation",
    "OperationResult",
    # Operations - Transport
    "Split",
    "Merge",
    "Move",
    "JunctionCrossing",
    # Operations - Crystal
    "CrystalRotation",
    "SympatheticCooling",
    # Operations - Quantum
    "SingleQubitGate",
    "MSGate",
    "GateSwap",
    "Measurement",
    "QubitReset",
    # Operations - Batch
    "ReconfigurationStep",
    "GlobalReconfiguration",
    # Operations - Factories
    "create_split",
    "create_merge",
    "create_ms_gate",
    "create_single_qubit_gate",
    "create_measurement",
    # Scheduling - Schedulers
    "WISEBatchScheduler",
    "WISECriticalPathScheduler",
    "BarrierAwareScheduler",
    # Scheduling - Functions
    "schedule_operations_wise",
    "schedule_operations_with_barriers",
    "build_happens_before_dag",
    "calculate_dephasing_fidelity",
    "calculate_dephasing_from_idling",
    # Scheduling - Data structures
    "ParallelOperationGroup",
    "T2_DEPHASING_TIME",
    # Routing - SAT Encoder
    "WISESATEncoder",
    "WISESATContext",
    # Routing - Routers
    "WiseSatRouter",
    "WisePatchRouter",
    "GreedyIonRouter",
    # Routing - Compilation Pass
    "WISERoutingPass",
    # Routing - Cost Model
    "WISECostModel",
    # Routing - Config
    "WISERoutingConfig",
    # Routing - Data structures
    "GridLayout",
    "RoutingPass",
    "RoutingSchedule",
    "GatePairRequirement",
    "compute_target_positions",
    # Clustering
    "regular_partition",
    "arrange_clusters",
    "hill_climb_on_arrange_clusters",
    "merge_clusters_to_limit",
    # Visualization
    "display_architecture",
    "animate_transport",
    "visualize_reconfiguration",
]
