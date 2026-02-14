# src/qectostim/experiments/hardware_simulation/trapped_ion/compilers/__init__.py
"""
Trapped ion compilers package.

Provides architecture-specific compilers for different trapped ion platforms:
- TrappedIonCompiler: Abstract base with shared gate decompositions
- WISECompiler: WISE grid architecture with SAT-based routing
- QCCDCompiler: General QCCD with split/merge/shuttle routing

Example
-------
>>> from qectostim.experiments.hardware_simulation.trapped_ion.compilers import (
...     WISECompiler,
... )
>>> compiler = WISECompiler(architecture, routing_config=config)
>>> result = compiler.compile(circuit)
"""
from __future__ import annotations

from qectostim.experiments.hardware_simulation.trapped_ion.compilers.base import (
    DecomposedGate,
    TrappedIonCompiler,
    PI,
    PI_2,
    PI_4,
)
from qectostim.experiments.hardware_simulation.trapped_ion.compilers.wise import (
    WISECompiler,
)
from qectostim.experiments.hardware_simulation.trapped_ion.compilers.qccd import (
    QCCDCompiler,
)
from qectostim.experiments.hardware_simulation.trapped_ion.compilers.augmented_grid import (
    AugmentedGridCompiler,
)
from qectostim.experiments.hardware_simulation.trapped_ion.compilers.networked import (
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
    "QCCDCompiler",
    "AugmentedGridCompiler",
    "NetworkedGridCompiler",
]
