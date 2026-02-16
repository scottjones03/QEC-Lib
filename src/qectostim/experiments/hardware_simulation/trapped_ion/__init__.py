"""
Trapped-ion QCCD hardware simulation package.

This package provides a complete simulation framework for trapped-ion
Quantum Charge-Coupled Device (QCCD) architectures, including:

- **Architecture models**: Augmented Grid, WISE (Wavelength-multiplexed Ion Shuttling Engine)
- **Compilation pipeline**: Decomposition, mapping, routing, scheduling
- **Noise modeling**: Physics-based noise models with calibration parameters
- **Visualization**: Animation of ion transport, gate execution, and reconfiguration

Main Components
---------------

### Architectures (`utils.architectures`)
- :class:`AugmentedGridArchitecture`: Standard junction-based QCCD grid
- :class:`WISEArchitecture`: Wavelength-multiplexed architecture with patch-based routing
- :class:`NetworkedGridArchitecture`: Multi-grid networked architecture

### Compiler (`utils.trapped_ion_compiler`)
- :class:`TrappedIonCompiler`: Main compilation pipeline (decompose → map → route → schedule)

### Experiment (`utils.experiment`)
- :class:`TrappedIonExperiment`: High-level experiment interface with noise simulation

### Visualization (`viz`)
- :func:`display_architecture`: Render QCCD topology
- :func:`animate_transport`: Create transport animation with ion movements
- :func:`render_animation`: Export animation to video file

Example
-------
>>> from qectostim.experiments.hardware_simulation.trapped_ion import (
...     AugmentedGridArchitecture,
...     TrappedIonCompiler,
...     TrappedIonExperiment,
... )
>>> arch = AugmentedGridArchitecture(rows=4, cols=4, ions_per_trap=2)
>>> compiler = TrappedIonCompiler(arch)
>>> # ... define code and experiment ...
"""

# Main architecture classes
from .utils import (
    # Architectures
    AugmentedGridArchitecture,
    WISEArchitecture,
    NetworkedGridArchitecture,
    # Base classes
    QCCDArch,
    QCCDWiseArch,
    # Operations
    Operation,
    QubitOperation,
    TwoQubitMSGate,
    # Compiler
    TrappedIonCompiler,
    # Experiment
    TrappedIonExperiment,
    process_circuit,
    process_circuit_wise_arch,
    # Noise
    TrappedIonNoiseModel,
    TrappedIonCalibration,
    # Execution planning
    ExecutionPlanner,
    TrappedIonExecutionPlanner,
    # Configuration
    WISERoutingConfig,
    RoutingProgress,
    # Constants
    NDE_LZ,
    NDE_JZ,
    NSE_Z,
)

# Visualization (import submodule, not individual functions)
from . import viz

# Compiler submodule (import as module to avoid circular imports)
from . import compiler

__all__ = [
    # Architectures
    "AugmentedGridArchitecture",
    "WISEArchitecture",
    "NetworkedGridArchitecture",
    # Base classes
    "QCCDArch",
    "QCCDWiseArch",
    # Operations
    "Operation",
    "QubitOperation",
    "TwoQubitMSGate",
    # Compiler
    "TrappedIonCompiler",
    # Experiment
    "TrappedIonExperiment",
    "process_circuit",
    "process_circuit_wise_arch",
    # Noise
    "TrappedIonNoiseModel",
    "TrappedIonCalibration",
    # Execution planning
    "ExecutionPlanner",
    "TrappedIonExecutionPlanner",
    # Configuration
    "WISERoutingConfig",
    "RoutingProgress",
    # Constants
    "NDE_LZ",
    "NDE_JZ",
    "NSE_Z",
    # Submodules
    "viz",
    "compiler",
]

