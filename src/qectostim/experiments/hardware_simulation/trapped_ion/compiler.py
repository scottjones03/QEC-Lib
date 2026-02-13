# src/qectostim/experiments/hardware_simulation/trapped_ion/compiler.py
"""
Trapped ion circuit compiler.

This module re-exports all compiler classes from the compilers/ package
for backwards compatibility. New code should import directly from the
compilers subpackage:

>>> from qectostim.experiments.hardware_simulation.trapped_ion.compilers import (
...     WISECompiler,
...     LinearChainCompiler,
... )

Architecture hierarchy:
- TrappedIonCompiler (ABC): Technology-specific base (MS gates, ion physics)
  - WISECompiler: WISE grid architecture with SAT-based routing
  - LinearChainCompiler: Single linear chain, all-to-all connectivity
  - QCCDCompiler: General QCCD with split/merge/shuttle routing

Each concrete compiler handles the specific constraints of its architecture.
The base class provides common trapped-ion functionality (gate decompositions).
"""
from __future__ import annotations

# Re-export everything from compilers package for backwards compatibility
from qectostim.experiments.hardware_simulation.trapped_ion.compilers import (
    # Constants
    PI,
    PI_2,
    PI_4,
    # Data classes
    DecomposedGate,
    # Base class
    TrappedIonCompiler,
    # Concrete compilers
    WISECompiler,
    LinearChainCompiler,
    QCCDCompiler,
    AugmentedGridCompiler,
    NetworkedGridCompiler,
)

__all__ = [
    # Constants
    "PI",
    "PI_2",
    "PI_4",
    # Data classes
    "DecomposedGate",
    # Base class
    "TrappedIonCompiler",
    # Concrete compilers
    "WISECompiler",
    "LinearChainCompiler",
    "QCCDCompiler",
    "AugmentedGridCompiler",
    "NetworkedGridCompiler",
]
