# src/qectostim/experiments/hardware_simulation/__init__.py
"""
Hardware simulation framework for QEC experiments.

This module provides hardware-aware simulation of quantum error correction,
modeling realistic noise from specific qubit platforms and architectures.

Supported Platforms
-------------------
- **Trapped Ion**: QCCD and linear chain architectures
- **Superconducting**: Fixed grid and tunable coupler architectures  
- **Neutral Atom**: Tweezer array and Rydberg lattice architectures

Architecture
------------
The framework follows a compilation pipeline:

1. **Decomposition**: Abstract circuit → native gates
2. **Mapping**: Logical qubits → physical qubits
3. **Routing**: Insert transport/SWAP for connectivity
4. **Scheduling**: Time-order operations with constraints
5. **Noise injection**: Apply hardware-specific noise model

Example
-------
>>> from qectostim.experiments.hardware_simulation import (
...     TweezerArrayArchitecture,
...     NeutralAtomSimulator,
...     get_platform_registry,
... )
>>> # Create architecture
>>> arch = TweezerArrayArchitecture(rows=4, cols=4)
>>> # Create simulator
>>> sim = NeutralAtomSimulator(arch)
>>> # Run experiment (NOT YET IMPLEMENTED)
>>> # results = sim.run(code, shots=1000)

NOT IMPLEMENTED: This framework defines interfaces only.
Concrete implementations will follow.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type, TYPE_CHECKING

# Core interfaces
from qectostim.experiments.hardware_simulation.core import (
    # Architecture
    HardwareArchitecture,
    ConnectivityGraph,
    Zone,
    ZoneType,
    PhysicalConstraints,
    # Gates
    NativeGateSet,
    GateSpec,
    GateType,
    GateDecomposition,
    # Components
    HardwareComponent,
    PhysicalQubit,
    Coupler,
    # Operations
    PhysicalOperation,
    TransportOperation,
    GateOperation,
    MeasurementOperation,
    # Pipeline
    NativeCircuit,
    MappedCircuit,
    RoutedCircuit,
    ScheduledCircuit,
    CompiledCircuit,
    # Compiler
    HardwareCompiler,
    CompilationPass,
    CompilationConfig,
)

# Base simulator
from qectostim.experiments.hardware_simulation.base import HardwareSimulator

# Trapped Ion platform
from qectostim.experiments.hardware_simulation.trapped_ion import (
    TrappedIonArchitecture,
    QCCDArchitecture,
    LinearChainArchitecture,
    TrappedIonCompiler,
    QCCDCompiler,
    LinearChainCompiler,
    TrappedIonSimulator,
    TrappedIonNoiseModel,
)

# Superconducting platform
from qectostim.experiments.hardware_simulation.superconducting import (
    SuperconductingArchitecture,
    FixedCouplingArchitecture,
    TunableCouplerArchitecture,
    HeavyHexArchitecture,
    SquareLatticeArchitecture,
    SuperconductingCompiler,
    SuperconductingSimulator,
    SuperconductingNoiseModel,
)

# Neutral Atom platform
from qectostim.experiments.hardware_simulation.neutral_atom import (
    NeutralAtomArchitecture,
    TweezerArrayArchitecture,
    RydbergLatticeArchitecture,
    NeutralAtomCompiler,
    NeutralAtomSimulator,
    NeutralAtomNoiseModel,
)

# Noise models (from main noise module)
from qectostim.noise.hardware import (
    HardwareNoiseModel,
    CalibrationData,
)


# =============================================================================
# Platform Registry
# =============================================================================

class PlatformInfo:
    """Information about a hardware platform."""
    
    def __init__(
        self,
        name: str,
        architecture_classes: List[Type[HardwareArchitecture]],
        compiler_classes: List[Type[HardwareCompiler]],
        simulator_class: Type[HardwareSimulator],
        noise_model_class: Type[HardwareNoiseModel],
        description: str = "",
    ):
        self.name = name
        self.architecture_classes = architecture_classes
        self.compiler_classes = compiler_classes
        self.simulator_class = simulator_class
        self.noise_model_class = noise_model_class
        self.description = description
    
    def __repr__(self) -> str:
        return f"PlatformInfo({self.name!r})"


# Registry of available platforms
_PLATFORM_REGISTRY: Dict[str, PlatformInfo] = {
    "trapped_ion": PlatformInfo(
        name="trapped_ion",
        architecture_classes=[
            TrappedIonArchitecture,
            QCCDArchitecture,
            LinearChainArchitecture,
        ],
        compiler_classes=[
            TrappedIonCompiler,
            QCCDCompiler,
            LinearChainCompiler,
        ],
        simulator_class=TrappedIonSimulator,
        noise_model_class=TrappedIonNoiseModel,
        description="Trapped ion quantum computers (QCCD, linear chain)",
    ),
    "superconducting": PlatformInfo(
        name="superconducting",
        architecture_classes=[
            SuperconductingArchitecture,
            FixedCouplingArchitecture,
            TunableCouplerArchitecture,
            HeavyHexArchitecture,
            SquareLatticeArchitecture,
        ],
        compiler_classes=[
            SuperconductingCompiler,
        ],
        simulator_class=SuperconductingSimulator,
        noise_model_class=SuperconductingNoiseModel,
        description="Superconducting transmon quantum computers",
    ),
    "neutral_atom": PlatformInfo(
        name="neutral_atom",
        architecture_classes=[
            NeutralAtomArchitecture,
            TweezerArrayArchitecture,
            RydbergLatticeArchitecture,
        ],
        compiler_classes=[
            NeutralAtomCompiler,
        ],
        simulator_class=NeutralAtomSimulator,
        noise_model_class=NeutralAtomNoiseModel,
        description="Neutral atom quantum computers (tweezer arrays, Rydberg lattices)",
    ),
}


def get_platform_registry() -> Dict[str, PlatformInfo]:
    """Get the registry of available hardware platforms.
    
    Returns
    -------
    Dict[str, PlatformInfo]
        Mapping from platform name to platform info.
    
    Example
    -------
    >>> registry = get_platform_registry()
    >>> print(registry.keys())
    dict_keys(['trapped_ion', 'superconducting', 'neutral_atom'])
    """
    return _PLATFORM_REGISTRY.copy()


def get_platform(name: str) -> PlatformInfo:
    """Get information about a specific platform.
    
    Parameters
    ----------
    name : str
        Platform name ('trapped_ion', 'superconducting', 'neutral_atom').
    
    Returns
    -------
    PlatformInfo
        Information about the platform.
    
    Raises
    ------
    KeyError
        If platform not found.
    """
    if name not in _PLATFORM_REGISTRY:
        available = list(_PLATFORM_REGISTRY.keys())
        raise KeyError(f"Platform {name!r} not found. Available: {available}")
    return _PLATFORM_REGISTRY[name]


def list_platforms() -> List[str]:
    """List available hardware platforms.
    
    Returns
    -------
    List[str]
        Names of available platforms.
    """
    return list(_PLATFORM_REGISTRY.keys())


def list_architectures(platform: Optional[str] = None) -> List[Type[HardwareArchitecture]]:
    """List available architectures.
    
    Parameters
    ----------
    platform : str, optional
        Filter by platform name.
    
    Returns
    -------
    List[Type[HardwareArchitecture]]
        Available architecture classes.
    """
    if platform is not None:
        info = get_platform(platform)
        return info.architecture_classes.copy()
    
    all_archs = []
    for info in _PLATFORM_REGISTRY.values():
        all_archs.extend(info.architecture_classes)
    return all_archs


# =============================================================================
# Convenience factory functions
# =============================================================================

def create_simulator(
    platform: str,
    architecture_type: Optional[str] = None,
    **architecture_kwargs,
) -> HardwareSimulator:
    """Create a hardware simulator for a platform.
    
    Parameters
    ----------
    platform : str
        Platform name ('trapped_ion', 'superconducting', 'neutral_atom').
    architecture_type : str, optional
        Specific architecture type (e.g., 'qccd', 'tweezer_array').
    **architecture_kwargs
        Arguments passed to architecture constructor.
    
    Returns
    -------
    HardwareSimulator
        Configured simulator instance.
    
    Raises
    ------
    NotImplementedError
        Simulators are not yet implemented.
    
    Example
    -------
    >>> sim = create_simulator(
    ...     "neutral_atom",
    ...     architecture_type="tweezer_array",
    ...     rows=4, cols=4,
    ... )
    """
    raise NotImplementedError(
        "create_simulator() not yet implemented. "
        "Use platform-specific classes directly."
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core interfaces
    "HardwareArchitecture",
    "ConnectivityGraph",
    "Zone",
    "ZoneType",
    "PhysicalConstraints",
    "NativeGateSet",
    "GateSpec",
    "GateType",
    "GateDecomposition",
    "HardwareComponent",
    "PhysicalQubit",
    "Coupler",
    "PhysicalOperation",
    "TransportOperation",
    "GateOperation",
    "MeasurementOperation",
    "NativeCircuit",
    "MappedCircuit",
    "RoutedCircuit",
    "ScheduledCircuit",
    "CompiledCircuit",
    "HardwareCompiler",
    "CompilationPass",
    "CompilationConfig",
    # Base
    "HardwareSimulator",
    "HardwareNoiseModel",
    "CalibrationData",
    # Trapped Ion
    "TrappedIonArchitecture",
    "QCCDArchitecture",
    "LinearChainArchitecture",
    "TrappedIonCompiler",
    "QCCDCompiler",
    "LinearChainCompiler",
    "TrappedIonSimulator",
    "TrappedIonNoiseModel",
    # Superconducting
    "SuperconductingArchitecture",
    "FixedCouplingArchitecture",
    "TunableCouplerArchitecture",
    "HeavyHexArchitecture",
    "SquareLatticeArchitecture",
    "SuperconductingCompiler",
    "SuperconductingSimulator",
    "SuperconductingNoiseModel",
    # Neutral Atom
    "NeutralAtomArchitecture",
    "TweezerArrayArchitecture",
    "RydbergLatticeArchitecture",
    "NeutralAtomCompiler",
    "NeutralAtomSimulator",
    "NeutralAtomNoiseModel",
    # Registry
    "PlatformInfo",
    "get_platform_registry",
    "get_platform",
    "list_platforms",
    "list_architectures",
    "create_simulator",
]
