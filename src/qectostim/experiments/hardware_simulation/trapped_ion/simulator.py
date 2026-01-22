# src/qectostim/experiments/hardware_simulation/trapped_ion/simulator.py
"""
Trapped ion hardware simulator.

Main simulator class for trapped ion quantum computers.
Integrates with TrappedIonExecutionPlanner for timing-aware noise.
"""
from __future__ import annotations

from typing import (
    Dict,
    List,
    Optional,
    Any,
    TYPE_CHECKING,
)

import stim

from qectostim.experiments.hardware_simulation.base import HardwareSimulator
from qectostim.codes.abstract_code import Code
from qectostim.noise.models import NoiseModel

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
        TrappedIonArchitecture,
    )
    from qectostim.experiments.hardware_simulation.trapped_ion.compiler import (
        TrappedIonCompiler,
    )
    from qectostim.experiments.hardware_simulation.trapped_ion.noise import (
        TrappedIonNoiseModel,
    )
    from qectostim.experiments.hardware_simulation.core.execution import ExecutionPlan


class TrappedIonSimulator(HardwareSimulator):
    """Simulator for trapped ion quantum computers.
    
    Simulates QEC experiments on trapped ion hardware with:
    - QCCD or linear chain architectures
    - MS gate + rotation native gate set
    - Ion shuttling and routing (QCCD)
    - Timing-aware noise model with idle dephasing
    
    The simulator uses TrappedIonExecutionPlanner to extract timing
    and routing metadata from compilation, enabling accurate noise
    injection that accounts for:
    - Idle dephasing (T2 decay between operations)
    - Gate swap errors from ion transport
    - Per-qubit calibrated gate fidelities
    
    Parameters
    ----------
    code : Code
        The quantum error correction code.
    architecture : TrappedIonArchitecture
        Hardware architecture (QCCD or linear chain).
    compiler : Optional[TrappedIonCompiler]
        Compiler for this architecture.
    hardware_noise : Optional[TrappedIonNoiseModel]
        Trapped ion noise model.
    noise_model : Optional[NoiseModel]
        Additional circuit-level noise.
    metadata : Optional[Dict[str, Any]]
        Additional experiment metadata.
    
    Examples
    --------
    >>> from qectostim.codes.surface import SurfaceCode
    >>> from qectostim.experiments.hardware_simulation.trapped_ion import (
    ...     TrappedIonSimulator,
    ...     TrappedIonNoiseModel,
    ...     WISECompiler,
    ...     QCCDArchitecture,
    ... )
    >>> 
    >>> code = SurfaceCode(distance=3)
    >>> arch = QCCDArchitecture(num_zones=4, ions_per_zone=8)
    >>> compiler = WISECompiler(arch)
    >>> noise = TrappedIonNoiseModel()
    >>> 
    >>> simulator = TrappedIonSimulator(
    ...     code=code,
    ...     architecture=arch,
    ...     compiler=compiler,
    ...     hardware_noise=noise,
    ... )
    >>> result = simulator.simulate(num_shots=10000)
    >>> print(f"Logical error rate: {result.logical_error_rate:.2e}")
    """
    
    def __init__(
        self,
        code: Code,
        architecture: "TrappedIonArchitecture",
        compiler: Optional["TrappedIonCompiler"] = None,
        hardware_noise: Optional["TrappedIonNoiseModel"] = None,
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
        self._execution_plan: Optional["ExecutionPlan"] = None
    
    def _create_default_compiler(self) -> "TrappedIonCompiler":
        """Create default trapped ion compiler."""
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler import (
            LinearChainCompiler,
        )
        # Default to linear chain compiler (simpler, works for any architecture)
        return LinearChainCompiler(self.architecture)
    
    def apply_hardware_noise(self, circuit: stim.Circuit) -> stim.Circuit:
        """Apply timing-aware trapped ion noise.
        
        Uses TrappedIonExecutionPlanner to create an ExecutionPlan
        from the compilation result, then applies noise using
        apply_with_plan() for timing-accurate injection.
        
        The noise injection order is:
        1. Idle dephasing (Z_ERROR) from T2 decay
        2. Gate swap noise (DEPOLARIZE2) from transport
        3. Original instruction
        4. Gate infidelity (DEPOLARIZE1/2)
        
        Parameters
        ----------
        circuit : stim.Circuit
            The Stim circuit to apply noise to.
            
        Returns
        -------
        stim.Circuit
            Circuit with timing-aware noise applied.
        """
        if self.hardware_noise is None:
            return circuit
        
        from qectostim.experiments.hardware_simulation.trapped_ion.execution import (
            TrappedIonExecutionPlanner,
        )
        
        # Create execution planner
        calibration = None
        if hasattr(self.hardware_noise, 'calibration'):
            calibration = self.hardware_noise.calibration
        
        planner = TrappedIonExecutionPlanner(
            compiler=self.compiler,
            calibration=calibration,
        )
        
        # Generate execution plan
        self._execution_plan = planner.plan_execution(
            circuit=circuit,
            compiled=self._compiled,
        )
        
        # Apply timing-aware noise
        if hasattr(self.hardware_noise, 'apply_with_plan'):
            return self.hardware_noise.apply_with_plan(circuit, self._execution_plan)
        else:
            # Fallback to basic noise application
            return self.hardware_noise.apply(circuit)
    
    @property
    def execution_plan(self) -> Optional["ExecutionPlan"]:
        """Get the execution plan from the last noise application."""
        return self._execution_plan
    
    def build_ideal_circuit(self) -> stim.Circuit:
        """Build ideal circuit for the QEC experiment.
        
        NOTE: This method should be overridden by experiment-specific
        subclasses (e.g., TrappedIonMemoryExperiment).
        """
        raise NotImplementedError(
            "TrappedIonSimulator.build_ideal_circuit() not yet implemented. "
            "Use a specific experiment class like TrappedIonMemoryExperiment, "
            "or override this method in a subclass."
        )
