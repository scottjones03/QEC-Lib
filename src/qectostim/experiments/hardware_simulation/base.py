# src/qectostim/experiments/hardware_simulation/base.py
"""
Hardware simulator base class.

Provides the abstract base for platform-specific hardware simulators
that integrate with the QECToStim experiment framework.
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    TYPE_CHECKING,
)

from qectostim.decoders.base import Decoder
import stim
import numpy as np

from qectostim.experiments.experiment import Experiment
from qectostim.codes.abstract_code import Code
from qectostim.noise.models import NoiseModel

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.architecture import HardwareArchitecture
    from qectostim.experiments.hardware_simulation.core.compiler import HardwareCompiler
    from qectostim.experiments.hardware_simulation.core.pipeline import CompiledCircuit
    from qectostim.noise.hardware.base import HardwareNoiseModel


@dataclass
class HardwareSimulationResult:
    """Result of hardware simulation.
    
    Attributes
    ----------
    logical_error_rate : float
        Observed logical error rate.
    num_shots : int
        Number of shots simulated.
    num_errors : int
        Number of logical errors.
    compilation_metrics : Dict[str, Any]
        Metrics from compilation (depth, gate count, etc.).
    simulation_metrics : Dict[str, Any]
        Metrics from simulation (duration, fidelity, etc.).
    decoder_used : str
        Decoder used for error correction.
    """
    logical_error_rate: float
    num_shots: int
    num_errors: int
    compilation_metrics: Dict[str, Any] = field(default_factory=dict)
    simulation_metrics: Dict[str, Any] = field(default_factory=dict)
    decoder_used: str = "auto"


class HardwareSimulator(Experiment):
    """Abstract base class for hardware simulators.
    
    Integrates with the QECToStim Experiment framework while adding
    hardware-specific compilation and noise modeling.
    
    The simulation pipeline:
    1. Build ideal circuit from QEC experiment
    2. Compile to hardware (decompose, map, route, schedule)
    3. Apply hardware-specific noise model
    4. Run decoding and return results
    
    Subclasses implement platform-specific logic:
    - TrappedIonSimulator: QCCD architecture
    - SuperconductingSimulator: Fixed-grid architectures
    - NeutralAtomSimulator: Tweezer array architectures
    
    Parameters
    ----------
    code : Code
        The quantum error correction code.
    architecture : HardwareArchitecture
        Hardware architecture specification.
    compiler : Optional[HardwareCompiler]
        Compiler for this architecture (created from architecture if None).
    hardware_noise : Optional[HardwareNoiseModel]
        Hardware-specific noise model.
    noise_model : Optional[NoiseModel]
        Additional circuit-level noise (applied after hardware noise).
    metadata : Optional[Dict[str, Any]]
        Additional experiment metadata.
    """
    
    def __init__(
        self,
        code: Code,
        architecture: "HardwareArchitecture",
        compiler: Optional["HardwareCompiler"] = None,
        hardware_noise: Optional["HardwareNoiseModel"] = None,
        noise_model: Optional[NoiseModel] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(code, noise_model, metadata)
        
        self.architecture = architecture
        self._compiler = compiler
        self.hardware_noise = hardware_noise
        
        # Cached compiled circuit
        self._compiled: Optional["CompiledCircuit"] = None
    
    @property
    def compiler(self) -> "HardwareCompiler":
        """Get the compiler, creating default if needed."""
        if self._compiler is None:
            self._compiler = self._create_default_compiler()
        return self._compiler
    
    @abstractmethod
    def _create_default_compiler(self) -> "HardwareCompiler":
        """Create the default compiler for this platform.
        
        Returns
        -------
        HardwareCompiler
            Compiler configured for the architecture.
        """
        ...
    
    @abstractmethod
    def build_ideal_circuit(self) -> stim.Circuit:
        """Build the ideal (noise-free) circuit for this experiment.
        
        This is the logical circuit before hardware compilation.
        Typically includes stabilizer measurements, logical operations,
        and final measurements.
        
        Returns
        -------
        stim.Circuit
            Ideal Stim circuit.
        """
        ...
    
    def compile(self, circuit: Optional[stim.Circuit] = None) -> "CompiledCircuit":
        """Compile a circuit to hardware.
        
        Parameters
        ----------
        circuit : Optional[stim.Circuit]
            Circuit to compile. If None, uses build_ideal_circuit().
            
        Returns
        -------
        CompiledCircuit
            Fully compiled circuit.
        """
        if circuit is None:
            circuit = self.build_ideal_circuit()
        
        # Apply pre-compile hook
        circuit = self.pre_compile(circuit)
        
        # Run compilation — pass QEC metadata if available
        _qec_meta = getattr(self, '_qec_metadata', None)
        self._compiled = self.compiler.compile(circuit, qec_metadata=_qec_meta)
        
        # Apply post-compile hook
        self._compiled = self.post_compile(self._compiled)
        
        return self._compiled
    
    def to_stim(self) -> stim.Circuit:
        """Generate the final Stim circuit with hardware compilation and noise.
        
        Implements the Experiment interface.
        
        Returns
        -------
        stim.Circuit
            Hardware-compiled circuit with noise.
        """
        # Compile if not already done
        if self._compiled is None:
            self.compile()
        
        # Generate Stim circuit from compilation
        circuit = self._compiled.to_stim()

        return circuit
    
    @abstractmethod
    def simulate(
        self,
        num_shots: int = 10000,
        decoder: Optional[Decoder] = None,
    ) -> HardwareSimulationResult:
        """Run hardware simulation with decoding.
        
        Parameters
        ----------
        num_shots : int
            Number of shots to simulate.
        decoder : Optional[Decoder]
            Decoder to use. If None, auto-selects.
            
        Returns
        -------
        HardwareSimulationResult
            Simulation results.
        """

    
    def get_compilation_metrics(self) -> Dict[str, Any]:
        """Get metrics from the last compilation.
        
        Returns
        -------
        Dict[str, Any]
            Compilation metrics (gate count, depth, duration, etc.).
        """
        if self._compiled is None:
            return {}
        return self._compiled.compute_metrics()
    
    # Hooks for subclass customization
    
    def pre_compile(self, circuit: stim.Circuit) -> stim.Circuit:
        """Hook called before compilation.
        
        Override to preprocess the circuit before hardware compilation.
        """
        return circuit
    
    def post_compile(self, compiled: "CompiledCircuit") -> "CompiledCircuit":
        """Hook called after compilation.
        
        Override to postprocess the compiled circuit.
        """
        return compiled
    
    @abstractmethod
    def apply_hardware_noise(self) -> stim.Circuit:
        """Apply hardware-specific noise to the compiled circuit.

        Subclasses use ``self._compiled`` internally to access the
        compiled circuit data and build a noisy ``stim.Circuit``.

        Returns
        -------
        stim.Circuit
            Noisy circuit.
        """
        ...
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"code={self.code.__class__.__name__}, "
            f"architecture={self.architecture.name!r})"
        )


# Legacy alias for backwards compatibility
AbstractHardwareSimulator = HardwareSimulator