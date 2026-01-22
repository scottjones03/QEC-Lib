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
    LinearChainArchitecture,
)
from qectostim.experiments.hardware_simulation.trapped_ion.compiler import (
    TrappedIonCompiler,
    WISECompiler,
    QCCDCompiler,
    LinearChainCompiler,
    DecomposedGate,
)
from qectostim.experiments.hardware_simulation.trapped_ion.simulator import (
    TrappedIonSimulator,
)
from qectostim.experiments.hardware_simulation.trapped_ion.experiments import (
    TrappedIonMemoryExperiment,
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
from qectostim.experiments.hardware_simulation.trapped_ion.components import (
    # Operation types
    QCCDOperation,
    # Timing/heating constants
    TrappedIonTimings,
    HeatingRates,
    DEFAULT_TIMINGS,
    DEFAULT_HEATING_RATES,
    # Ion types
    Ion,
    QubitIon,
    CoolingIon,
    SpectatorIon,
    # Node types
    QCCDNode,
    Junction,
    Trap,
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
    create_manipulation_trap,
    create_storage_trap,
)
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

__all__ = [
    # Architecture
    "TrappedIonArchitecture",
    "QCCDArchitecture",
    "LinearChainArchitecture",
    # Compiler
    "TrappedIonCompiler",
    "WISECompiler",
    "QCCDCompiler",
    "LinearChainCompiler",
    "DecomposedGate",
    # Simulator
    "TrappedIonSimulator",
    # Experiments
    "TrappedIonMemoryExperiment",
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
]
