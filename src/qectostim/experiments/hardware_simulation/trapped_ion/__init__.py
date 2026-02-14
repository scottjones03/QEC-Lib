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
from qectostim.experiments.hardware_simulation.trapped_ion.operations import (
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
    QubitGate,
    QubitGateBase,
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
    XRotation,
    YRotation,
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
# ── Routing (old code, copied verbatim from old/) ────────────
from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
    # Config
    WISERoutingConfig,
    RoutingProgress,
    ProgressCallback,
    make_tqdm_progress_callback,
    # Utilities
    old_ops_to_transport_list,
    # WISE SAT routing
    ionRoutingWISEArch,
    # Greedy junction routing (augmented grid)
    ionRouting,
    # Qubit-to-ion mapping
    regularPartition,
    arrangeClusters,
    hillClimbOnArrangeClusters,
    # Scheduling / parallelisation
    paralleliseOperations,
    paralleliseOperationsSimple,
    paralleliseOperationsWithBarriers,
    calculateDephasingFromIdling,
    calculateDephasingFidelity,
    NoFeasibleLayoutError,
    ReconfigurationPlanner,
    GlobalReconfigurations,  # Alias for backwards compat
    ParallelOperation,
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
    "QubitGate",
    "QubitGateBase",
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
    "XRotation",
    "YRotation",
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
    # Routing - Config
    "WISERoutingConfig",
    "RoutingProgress",
    "ProgressCallback",
    "make_tqdm_progress_callback",
    # Routing - Utilities
    "old_ops_to_transport_list",
    # Routing - WISE SAT
    "ionRoutingWISEArch",
    # Routing - Greedy junction
    "ionRouting",
    # Routing - Qubit mapping
    "regularPartition",
    "arrangeClusters",
    "hillClimbOnArrangeClusters",
    # Routing - Scheduling
    "paralleliseOperations",
    "paralleliseOperationsSimple",
    "paralleliseOperationsWithBarriers",
    "calculateDephasingFromIdling",
    "calculateDephasingFidelity",
    # Routing - Operation types
    "NoFeasibleLayoutError",
    "ReconfigurationPlanner",
    "GlobalReconfigurations",
    "ParallelOperation",
    # Visualization
    "display_architecture",
    "animate_transport",
    "visualize_reconfiguration",
]
