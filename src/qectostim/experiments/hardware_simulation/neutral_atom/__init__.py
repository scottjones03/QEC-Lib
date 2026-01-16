# src/qectostim/experiments/hardware_simulation/neutral_atom/__init__.py
"""
Neutral atom hardware simulation.

Provides simulation capabilities for neutral atom quantum computers,
including optical tweezer arrays with Rydberg interactions.

Platform Characteristics:
- Dynamic reconfiguration (atoms can be moved)
- All-to-all connectivity within interaction zones
- High-fidelity Rydberg entangling gates
- Global addressing mode (all atoms in zone)
- Atom loss during operation

Architectures:
- TweezerArrayArchitecture: Optical tweezer array
- RydbergLatticeArchitecture: Fixed lattice with Rydberg gates

Gate Sets:
- NeutralAtomGateSet: Rydberg CZ, global rotations
- TweezerGateSet: Parallel multi-zone operations
- RydbergLatticeGateSet: Nearest-neighbor interactions
"""

from qectostim.experiments.hardware_simulation.neutral_atom.architecture import (
    NeutralAtomArchitecture,
    TweezerArrayArchitecture,
    RydbergLatticeArchitecture,
)
from qectostim.experiments.hardware_simulation.neutral_atom.compiler import (
    NeutralAtomCompiler,
)
from qectostim.experiments.hardware_simulation.neutral_atom.simulator import (
    NeutralAtomSimulator,
)
from qectostim.experiments.hardware_simulation.neutral_atom.noise import (
    NeutralAtomNoiseModel,
)
from qectostim.experiments.hardware_simulation.neutral_atom.gates import (
    NeutralAtomGateSet,
)

__all__ = [
    # Architecture
    "NeutralAtomArchitecture",
    "TweezerArrayArchitecture",
    "RydbergLatticeArchitecture",
    # Compiler
    "NeutralAtomCompiler",
    # Simulator
    "NeutralAtomSimulator",
    # Noise
    "NeutralAtomNoiseModel",
    # Gates
    "NeutralAtomGateSet",
]
