# src/qectostim/experiments/hardware_simulation/superconducting/__init__.py
"""
Superconducting hardware simulation.

Provides simulation capabilities for superconducting quantum computers,
including fixed-coupling and tunable-coupler architectures.

Platform Characteristics:
- Fixed connectivity (heavy-hex, square lattice, etc.)
- Fast gates (~100ns for 2Q gates)
- T1/T2 limited coherence (~100μs)
- Crosstalk between neighboring qubits
- Native gates: CX/CZ (IBM), iSWAP/CZ (Google), XY/CZ (Rigetti)

Architectures:
- FixedCouplingArchitecture: Heavy-hex, square lattice
- TunableCouplerArchitecture: Adjustable coupling strength

Gate Sets:
- IBMGateSet: CR, CX native gates
- GoogleGateSet: SQRT_ISWAP, FSIM native gates
- RigettiGateSet: XY, CZ native gates
"""

from qectostim.experiments.hardware_simulation.superconducting.architecture import (
    SuperconductingArchitecture,
    FixedCouplingArchitecture,
    TunableCouplerArchitecture,
    HeavyHexArchitecture,
    SquareLatticeArchitecture,
)
from qectostim.experiments.hardware_simulation.superconducting.compiler import (
    SuperconductingCompiler,
)
from qectostim.experiments.hardware_simulation.superconducting.simulator import (
    SuperconductingSimulator,
)
from qectostim.experiments.hardware_simulation.superconducting.noise import (
    SuperconductingNoiseModel,
)
from qectostim.experiments.hardware_simulation.superconducting.gates import (
    SuperconductingGateSet,
    IBMGateSet,
    GoogleGateSet,
    TunableCouplerGateSet,
)

__all__ = [
    # Architecture
    "SuperconductingArchitecture",
    "FixedCouplingArchitecture",
    "TunableCouplerArchitecture",
    "HeavyHexArchitecture",
    "SquareLatticeArchitecture",
    # Compiler
    "SuperconductingCompiler",
    # Simulator
    "SuperconductingSimulator",
    # Noise
    "SuperconductingNoiseModel",
    # Gates
    "SuperconductingGateSet",
    "IBMGateSet",
    "GoogleGateSet",
    "TunableCouplerGateSet",
]
