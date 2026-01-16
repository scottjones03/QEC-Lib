# src/qectostim/experiments/hardware_simulation/superconducting/simulator.py
"""
Superconducting hardware simulator.

NOT IMPLEMENTED: This is a stub defining the interface.
"""
from __future__ import annotations

from typing import (
    Dict,
    Optional,
    Any,
    TYPE_CHECKING,
)

import stim

from qectostim.experiments.hardware_simulation.base import HardwareSimulator
from qectostim.codes.abstract_code import Code
from qectostim.noise.models import NoiseModel

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.superconducting.architecture import (
        SuperconductingArchitecture,
    )
    from qectostim.experiments.hardware_simulation.superconducting.compiler import (
        SuperconductingCompiler,
    )
    from qectostim.experiments.hardware_simulation.superconducting.noise import (
        SuperconductingNoiseModel,
    )


class SuperconductingSimulator(HardwareSimulator):
    """Simulator for superconducting quantum computers.
    
    NOT IMPLEMENTED: This is a stub defining the interface.
    """
    
    def __init__(
        self,
        code: Code,
        architecture: "SuperconductingArchitecture",
        compiler: Optional["SuperconductingCompiler"] = None,
        hardware_noise: Optional["SuperconductingNoiseModel"] = None,
        noise_model: Optional[NoiseModel] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            code=code,
            architecture=architecture,
            compiler=compiler,
            hardware_noise=hardware_noise,
            noise_model=noise_model,
            metadata=metadata,
        )
    
    def _create_default_compiler(self) -> "SuperconductingCompiler":
        """Create default superconducting compiler."""
        from qectostim.experiments.hardware_simulation.superconducting.compiler import (
            SuperconductingCompiler,
        )
        return SuperconductingCompiler(self.architecture)
    
    def build_ideal_circuit(self) -> stim.Circuit:
        """Build ideal circuit for the QEC experiment.
        
        NOT IMPLEMENTED.
        """
        raise NotImplementedError(
            "SuperconductingSimulator.build_ideal_circuit() not yet implemented."
        )
