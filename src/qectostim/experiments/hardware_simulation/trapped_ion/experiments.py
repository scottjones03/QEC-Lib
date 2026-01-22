# src/qectostim/experiments/hardware_simulation/trapped_ion/experiments.py
"""
Trapped ion specific experiments.

Provides experiment classes for trapped ion hardware simulation,
integrating QEC codes with hardware-aware compilation and noise.
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
import numpy as np

from qectostim.experiments.hardware_simulation.trapped_ion.simulator import (
    TrappedIonSimulator,
)
from qectostim.experiments.hardware_simulation.base import HardwareSimulationResult
from qectostim.codes.abstract_code import Code
from qectostim.codes.abstract_css import CSSCode
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


class TrappedIonMemoryExperiment(TrappedIonSimulator):
    """Memory experiment for trapped ion hardware.
    
    Simulates a QEC memory experiment (repeated stabilizer measurements)
    on trapped ion hardware with realistic noise modeling.
    
    This experiment:
    1. Initializes logical qubit in a chosen state
    2. Performs `rounds` of stabilizer measurements
    3. Measures data qubits to extract syndrome and logical outcome
    4. Applies hardware-aware noise (idle dephasing, gate errors, transport)
    
    Parameters
    ----------
    code : Code
        The quantum error correction code.
    architecture : TrappedIonArchitecture
        Hardware architecture (QCCD, linear chain, etc.).
    rounds : int
        Number of stabilizer measurement rounds.
    compiler : Optional[TrappedIonCompiler]
        Compiler for hardware. Uses LinearChainCompiler if None.
    hardware_noise : Optional[TrappedIonNoiseModel]
        Trapped ion noise model.
    noise_model : Optional[NoiseModel]
        Additional circuit-level noise.
    logical_qubit : int
        Which logical qubit to use (for multi-qubit codes).
    initial_state : str
        Initial logical state: "0", "1", "+", "-".
    metadata : Optional[Dict[str, Any]]
        Additional experiment metadata.
        
    Examples
    --------
    >>> from qectostim.codes.surface import SurfaceCode
    >>> from qectostim.experiments.hardware_simulation.trapped_ion import (
    ...     TrappedIonMemoryExperiment,
    ...     TrappedIonNoiseModel,
    ...     LinearChainArchitecture,
    ... )
    >>> 
    >>> code = SurfaceCode(distance=3)
    >>> arch = LinearChainArchitecture(num_ions=17)
    >>> noise = TrappedIonNoiseModel()
    >>> 
    >>> exp = TrappedIonMemoryExperiment(
    ...     code=code,
    ...     architecture=arch,
    ...     rounds=3,
    ...     hardware_noise=noise,
    ... )
    >>> result = exp.simulate(num_shots=10000)
    >>> print(f"Logical error rate: {result.logical_error_rate:.2e}")
    """
    
    def __init__(
        self,
        code: Code,
        architecture: "TrappedIonArchitecture",
        rounds: int = 1,
        compiler: Optional["TrappedIonCompiler"] = None,
        hardware_noise: Optional["TrappedIonNoiseModel"] = None,
        noise_model: Optional[NoiseModel] = None,
        logical_qubit: int = 0,
        initial_state: str = "0",
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
        self.rounds = rounds
        self.logical_qubit = logical_qubit
        self.initial_state = initial_state
        self.operation = "memory"
        
        # Cache for ideal circuit
        self._ideal_circuit: Optional[stim.Circuit] = None
    
    def build_ideal_circuit(self) -> stim.Circuit:
        """Build the ideal memory experiment circuit.
        
        Attempts to use the code's built-in circuit generation,
        falling back to a generic CSS memory circuit if needed.
        
        Returns
        -------
        stim.Circuit
            Ideal Stim circuit for memory experiment.
        """
        if self._ideal_circuit is not None:
            return self._ideal_circuit
        
        # Try code's to_stim() method first
        if hasattr(self.code, 'to_stim'):
            try:
                self._ideal_circuit = self.code.to_stim(
                    rounds=self.rounds,
                    after_clifford_depolarization=0.0,  # No noise in ideal
                    before_measure_flip_probability=0.0,
                )
                return self._ideal_circuit
            except (TypeError, ValueError):
                pass
        
        # Try memory_experiment method
        if hasattr(self.code, 'memory_experiment'):
            try:
                self._ideal_circuit = self.code.memory_experiment(
                    rounds=self.rounds,
                )
                return self._ideal_circuit
            except (TypeError, ValueError, AttributeError):
                pass
        
        # Build generic CSS memory circuit
        if isinstance(self.code, CSSCode):
            self._ideal_circuit = self._build_css_memory_circuit()
            return self._ideal_circuit
        
        # Last resort: try to get any circuit from the code
        raise NotImplementedError(
            f"Cannot build ideal circuit for {self.code.__class__.__name__}. "
            f"Code must implement to_stim() or memory_experiment() method."
        )
    
    def _build_css_memory_circuit(self) -> stim.Circuit:
        """Build a generic CSS memory experiment circuit.
        
        This builds a simple memory circuit for CSS codes using
        the stabilizer generators from hx and hz matrices.
        """
        from qectostim.experiments.memory import CSSMemoryExperiment
        
        # Map our initial_state to basis for CSSMemoryExperiment
        # initial_state=0 typically means |0> logical (Z basis encoding)
        # initial_state=1 typically means |+> logical (X basis encoding)
        basis = "Z" if self.initial_state == 0 else "X"
        
        # Delegate to the existing CSSMemoryExperiment
        css_exp = CSSMemoryExperiment(
            code=self.code,
            noise_model=None,  # We'll apply noise separately
            rounds=self.rounds,
            basis=basis,
        )
        return css_exp.to_stim()
    
    def to_stim(self) -> stim.Circuit:
        """Generate Stim circuit with hardware noise.
        
        For the initial implementation, this bypasses complex compilation
        and applies timing-aware noise directly to the ideal circuit.
        
        Returns
        -------
        stim.Circuit
            Circuit with hardware noise applied.
        """
        # Get ideal circuit
        ideal = self.build_ideal_circuit()
        
        # Apply hardware noise using execution plan
        if self.hardware_noise is not None:
            from qectostim.experiments.hardware_simulation.trapped_ion.execution import (
                create_simple_execution_plan,
            )
            
            # Create execution plan (estimates timing without full compilation)
            plan = create_simple_execution_plan(ideal)
            
            # Apply timing-aware noise
            if hasattr(self.hardware_noise, 'apply_with_plan'):
                circuit = self.hardware_noise.apply_with_plan(ideal, plan)
            else:
                circuit = self.hardware_noise.apply(ideal)
        else:
            circuit = ideal
        
        # Apply additional circuit-level noise
        if self.noise_model is not None:
            circuit = self.noise_model.apply(circuit)
        
        return circuit
    
    def simulate(
        self,
        num_shots: int = 10000,
        decoder_name: Optional[str] = None,
    ) -> HardwareSimulationResult:
        """Run hardware simulation with decoding.
        
        Parameters
        ----------
        num_shots : int
            Number of shots to simulate.
        decoder_name : Optional[str]
            Decoder to use. If None, auto-selects.
            
        Returns
        -------
        HardwareSimulationResult
            Simulation results with logical error rate.
        """
        # Run decoding through parent Experiment class
        decode_result = self.run_decode(shots=num_shots, decoder_name=decoder_name)
        
        # Get execution plan for metrics
        plan = None
        if self.hardware_noise is not None:
            from qectostim.experiments.hardware_simulation.trapped_ion.execution import (
                create_simple_execution_plan,
            )
            plan = create_simple_execution_plan(self.build_ideal_circuit())
        
        # Build result
        return HardwareSimulationResult(
            logical_error_rate=decode_result['logical_error_rate'],
            num_shots=decode_result['shots'],
            num_errors=int(np.sum(decode_result['logical_errors'])),
            compilation_metrics={
                "rounds": self.rounds,
                "num_qubits": self.code.n,
            },
            simulation_metrics={
                "total_duration_us": plan.total_duration if plan else 0.0,
                "num_operations": len(plan.operations) if plan else 0,
            },
            decoder_used=decoder_name or "auto",
        )
    
    def __repr__(self) -> str:
        return (
            f"TrappedIonMemoryExperiment("
            f"code={self.code.__class__.__name__}, "
            f"rounds={self.rounds}, "
            f"architecture={self.architecture.name!r})"
        )


class TrappedIonGadgetExperiment(TrappedIonSimulator):
    """Fault-tolerant gadget experiment for trapped ions.
    
    NOT YET IMPLEMENTED - placeholder for future gadget experiments
    (transversal gates, lattice surgery, etc.)
    """
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "TrappedIonGadgetExperiment not yet implemented. "
            "Use TrappedIonMemoryExperiment for memory experiments."
        )
    
    def build_ideal_circuit(self) -> stim.Circuit:
        raise NotImplementedError()
