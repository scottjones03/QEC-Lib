# src/qectostim/noise/hardware/base.py
"""
Base classes for hardware-specific noise models.

Defines the abstract interfaces for hardware noise that captures
the physical characteristics of real quantum devices.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    Union,
    Callable,
    TYPE_CHECKING,
)

import numpy as np
import stim

from qectostim.noise.models import NoiseModel

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.operations import PhysicalOperation
    from qectostim.experiments.hardware_simulation.core.execution import ExecutionPlan


class NoiseChannelType(Enum):
    """Types of noise channels."""
    DEPOLARIZING_1Q = auto()    # Single-qubit depolarizing
    DEPOLARIZING_2Q = auto()    # Two-qubit depolarizing
    DEPHASING = auto()          # Z-error (T2 decay)
    AMPLITUDE_DAMPING = auto()  # T1 decay
    BIT_FLIP = auto()           # X-error
    PHASE_FLIP = auto()         # Z-error
    MEASUREMENT = auto()        # Measurement error
    LEAKAGE = auto()            # Leakage to non-computational states
    CROSSTALK = auto()          # Crosstalk-induced error


@dataclass
class NoiseChannel:
    """A quantum noise channel specification.
    
    Attributes
    ----------
    channel_type : NoiseChannelType
        Type of noise channel.
    probability : float
        Error probability.
    qubits : Tuple[int, ...]
        Affected qubits.
    stim_instruction : str
        Stim noise instruction string.
    metadata : Dict[str, Any]
        Additional channel parameters.
    """
    channel_type: NoiseChannelType
    probability: float
    qubits: Tuple[int, ...]
    stim_instruction: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.stim_instruction:
            self.stim_instruction = self._generate_stim_instruction()
    
    def _generate_stim_instruction(self) -> str:
        """Generate Stim instruction for this channel."""
        qubit_str = " ".join(str(q) for q in self.qubits)
        
        channel_map = {
            NoiseChannelType.DEPOLARIZING_1Q: "DEPOLARIZE1",
            NoiseChannelType.DEPOLARIZING_2Q: "DEPOLARIZE2",
            NoiseChannelType.DEPHASING: "Z_ERROR",
            NoiseChannelType.BIT_FLIP: "X_ERROR",
            NoiseChannelType.PHASE_FLIP: "Z_ERROR",
            NoiseChannelType.MEASUREMENT: "X_ERROR",
        }
        
        stim_name = channel_map.get(self.channel_type, "DEPOLARIZE1")
        return f"{stim_name}({self.probability}) {qubit_str}"
    
    def to_stim(self) -> str:
        """Get the Stim instruction string."""
        return self.stim_instruction


@dataclass
class CalibrationData:
    """Hardware calibration data.
    
    Stores measured properties of a quantum device that affect
    noise modeling. Can be loaded from device calibration files.
    
    Attributes
    ----------
    t1_times : Dict[int, float]
        T1 relaxation time per qubit (microseconds).
    t2_times : Dict[int, float]
        T2 dephasing time per qubit (microseconds).
    single_qubit_gate_fidelities : Dict[Tuple[int, str], float]
        Gate fidelity by (qubit, gate_name).
    two_qubit_gate_fidelities : Dict[Tuple[int, int, str], float]
        Gate fidelity by (qubit1, qubit2, gate_name).
    readout_fidelities : Dict[int, Tuple[float, float]]
        Readout fidelity per qubit: (P(0|0), P(1|1)).
    crosstalk_matrix : Optional[np.ndarray]
        Crosstalk probability matrix between qubits.
    gate_times : Dict[str, float]
        Gate duration by name (microseconds).
    transport_fidelity : Optional[float]
        Fidelity of qubit transport (for mobile architectures).
    heating_rate : Optional[float]
        Motional heating rate (for trapped ions).
    metadata : Dict[str, Any]
        Additional calibration data.
    """
    t1_times: Dict[int, float] = field(default_factory=dict)
    t2_times: Dict[int, float] = field(default_factory=dict)
    single_qubit_gate_fidelities: Dict[Tuple[int, str], float] = field(default_factory=dict)
    two_qubit_gate_fidelities: Dict[Tuple[int, int, str], float] = field(default_factory=dict)
    readout_fidelities: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    crosstalk_matrix: Optional[np.ndarray] = None
    gate_times: Dict[str, float] = field(default_factory=dict)
    transport_fidelity: Optional[float] = None
    heating_rate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_t1(self, qubit: int, default: float = 100.0) -> float:
        """Get T1 time for a qubit."""
        return self.t1_times.get(qubit, default)
    
    def get_t2(self, qubit: int, default: float = 50.0) -> float:
        """Get T2 time for a qubit."""
        return self.t2_times.get(qubit, default)
    
    def get_1q_fidelity(self, qubit: int, gate: str, default: float = 0.999) -> float:
        """Get single-qubit gate fidelity."""
        return self.single_qubit_gate_fidelities.get((qubit, gate), default)
    
    def get_2q_fidelity(
        self, qubit1: int, qubit2: int, gate: str, default: float = 0.99
    ) -> float:
        """Get two-qubit gate fidelity."""
        # Try both orderings
        key1 = (qubit1, qubit2, gate)
        key2 = (qubit2, qubit1, gate)
        if key1 in self.two_qubit_gate_fidelities:
            return self.two_qubit_gate_fidelities[key1]
        if key2 in self.two_qubit_gate_fidelities:
            return self.two_qubit_gate_fidelities[key2]
        return default
    
    def get_readout_fidelity(
        self, qubit: int, default: Tuple[float, float] = (0.99, 0.99)
    ) -> Tuple[float, float]:
        """Get readout fidelity (P(0|0), P(1|1))."""
        return self.readout_fidelities.get(qubit, default)
    
    def get_gate_time(self, gate: str, default: float = 1.0) -> float:
        """Get gate duration in microseconds."""
        return self.gate_times.get(gate, default)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationData":
        """Create from dictionary (e.g., loaded from JSON)."""
        return cls(
            t1_times={int(k): v for k, v in data.get("t1_times", {}).items()},
            t2_times={int(k): v for k, v in data.get("t2_times", {}).items()},
            gate_times=data.get("gate_times", {}),
            metadata=data.get("metadata", {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "t1_times": {str(k): v for k, v in self.t1_times.items()},
            "t2_times": {str(k): v for k, v in self.t2_times.items()},
            "gate_times": self.gate_times,
            "metadata": self.metadata,
        }


class HardwareNoiseModel(NoiseModel):
    """Abstract base class for hardware-specific noise models.
    
    Extends NoiseModel to support:
    - Per-operation noise (not just per-gate)
    - Idle noise based on timing
    - Calibration data integration
    - Platform-specific noise characteristics
    
    Subclasses implement platform-specific noise:
    - TrappedIonNoiseModel: Heating, MS gate errors, transport
    - SuperconductingNoiseModel: T1/T2 decay, crosstalk, leakage
    - NeutralAtomNoiseModel: Atom loss, Rydberg errors, global addressing
    """
    
    def __init__(
        self,
        calibration: Optional[CalibrationData] = None,
        error_scaling: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the hardware noise model.
        
        Parameters
        ----------
        calibration : Optional[CalibrationData]
            Hardware calibration data.
        error_scaling : float
            Scale factor for all errors (for studies).
        metadata : Optional[Dict[str, Any]]
            Additional model metadata.
        """
        self.calibration = calibration or CalibrationData()
        self.error_scaling = error_scaling
        self.metadata = metadata or {}
    
    @abstractmethod
    def apply_to_operation(
        self,
        operation: "PhysicalOperation",
    ) -> List[NoiseChannel]:
        """Get noise channels for a physical operation.
        
        Parameters
        ----------
        operation : PhysicalOperation
            The operation being executed.
            
        Returns
        -------
        List[NoiseChannel]
            Noise channels to apply.
        """
        ...
    
    @abstractmethod
    def idle_noise(
        self,
        qubit: int,
        duration: float,
    ) -> List[NoiseChannel]:
        """Get noise channels for idle time.
        
        Parameters
        ----------
        qubit : int
            The idling qubit.
        duration : float
            Idle time in microseconds.
            
        Returns
        -------
        List[NoiseChannel]
            Noise channels for decoherence.
        """
        ...
    
    def apply_with_plan(
        self,
        circuit: stim.Circuit,
        plan: "ExecutionPlan",
    ) -> stim.Circuit:
        """Apply hardware noise using an execution plan for timing-aware injection.
        
        This method uses the ExecutionPlan to inject noise that accounts for:
        - Idle dephasing based on actual timing gaps
        - Gate swap errors from ion transport
        - Per-qubit calibrated fidelities
        - Architecture-specific noise characteristics
        
        The circuit structure is NEVER modified - only noise instructions
        are appended after existing gates.
        
        Parameters
        ----------
        circuit : stim.Circuit
            The Stim circuit to apply noise to.
        plan : ExecutionPlan
            Execution plan with timing/routing metadata.
            
        Returns
        -------
        stim.Circuit
            Circuit with timing-aware noise applied.
            
        Notes
        -----
        Subclasses should override this for platform-specific noise.
        The default implementation delegates to apply_to_operation()
        and idle_noise() abstract methods.
        """
        from qectostim.experiments.hardware_simulation.core.execution import (
            ExecutionPlan,
            OperationTiming,
        )
        
        noisy = stim.Circuit()
        instruction_index = 0
        
        for inst in circuit:
            if isinstance(inst, stim.CircuitRepeatBlock):
                # For repeat blocks, we apply basic noise (no timing info)
                noisy_body = self.apply(inst.body_copy())
                noisy.append(stim.CircuitRepeatBlock(inst.repeat_count, noisy_body))
                continue
            
            name = inst.name.upper()
            qubit_targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            
            # Skip non-gate instructions for noise injection
            if name in {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS"}:
                noisy.append(inst)
                continue
            
            # 1. Apply idle dephasing BEFORE this instruction
            idle_intervals = plan.get_idle_before(instruction_index)
            for idle in idle_intervals:
                channels = self.idle_noise(idle.qubit, idle.duration)
                for channel in channels:
                    if channel.probability > 0:
                        noisy.append_from_stim_program_text(channel.to_stim())
            
            # 2. Apply gate swap noise (if any transport was required)
            swaps = plan.get_swaps_for_instruction(instruction_index)
            for swap_info in swaps:
                if swap_info.num_swaps > 0:
                    error_prob = min(swap_info.error_probability * self.error_scaling, 0.5)
                    if error_prob > 0:
                        q1, q2 = swap_info.qubits
                        noisy.append_from_stim_program_text(
                            f"DEPOLARIZE2({error_prob}) {q1} {q2}"
                        )
            
            # 3. Append the original instruction
            noisy.append(inst)
            
            # 4. Apply operation noise AFTER the gate
            op_timing = plan.get_operation(instruction_index)
            if op_timing is not None and qubit_targets:
                # Get gate infidelity noise
                error_prob = (1.0 - op_timing.fidelity) * self.error_scaling
                error_prob = min(error_prob, 0.5)  # Clamp to valid range
                
                if error_prob > 0:
                    if len(qubit_targets) == 1:
                        noisy.append_from_stim_program_text(
                            f"DEPOLARIZE1({error_prob}) {qubit_targets[0]}"
                        )
                    elif len(qubit_targets) >= 2:
                        # Apply to pairs
                        for i in range(0, len(qubit_targets) - 1, 2):
                            q1, q2 = qubit_targets[i], qubit_targets[i + 1]
                            noisy.append_from_stim_program_text(
                                f"DEPOLARIZE2({error_prob}) {q1} {q2}"
                            )
            
            instruction_index += 1
        
        return noisy
    
    def apply(self, circuit: stim.Circuit) -> stim.Circuit:
        """Apply noise to a Stim circuit.
        
        Default implementation adds depolarizing noise based on
        calibration data. Override for platform-specific behavior.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Input circuit.
            
        Returns
        -------
        stim.Circuit
            Circuit with noise applied.
        """
        noisy = stim.Circuit()
        
        for inst in circuit:
            if isinstance(inst, stim.CircuitRepeatBlock):
                # Recursively apply to repeat blocks
                noisy_body = self.apply(inst.body_copy())
                noisy.append(stim.CircuitRepeatBlock(inst.repeat_count, noisy_body))
                continue
            
            noisy.append(inst)
            
            # Add noise after gates
            name = inst.name.upper()
            qubit_targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            
            if not qubit_targets:
                continue
            
            # Get noise based on gate type
            noise_channels = self._get_gate_noise(name, tuple(qubit_targets))
            
            for channel in noise_channels:
                if channel.probability > 0:
                    noisy.append_from_stim_program_text(channel.to_stim())
        
        return noisy
    
    def _get_gate_noise(
        self,
        gate_name: str,
        qubits: Tuple[int, ...],
    ) -> List[NoiseChannel]:
        """Get noise channels for a gate from calibration data.
        
        Override for platform-specific noise models.
        """
        channels = []
        
        # Skip non-gate instructions
        if gate_name in {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}:
            return channels
        
        # Single-qubit gates
        single_qubit_gates = {
            "H", "X", "Y", "Z", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG",
            "SQRT_Y", "SQRT_Y_DAG", "R", "RX", "RY", "RZ",
        }
        
        # Two-qubit gates
        two_qubit_gates = {"CX", "CNOT", "CZ", "CY", "SWAP", "ISWAP", "ISWAP_DAG"}
        
        if gate_name in single_qubit_gates:
            for q in qubits:
                fidelity = self.calibration.get_1q_fidelity(q, gate_name)
                error_prob = (1.0 - fidelity) * self.error_scaling
                if error_prob > 0:
                    channels.append(NoiseChannel(
                        channel_type=NoiseChannelType.DEPOLARIZING_1Q,
                        probability=error_prob,
                        qubits=(q,),
                    ))
        
        elif gate_name in two_qubit_gates and len(qubits) >= 2:
            for i in range(0, len(qubits), 2):
                if i + 1 < len(qubits):
                    q1, q2 = qubits[i], qubits[i + 1]
                    fidelity = self.calibration.get_2q_fidelity(q1, q2, gate_name)
                    error_prob = (1.0 - fidelity) * self.error_scaling
                    if error_prob > 0:
                        channels.append(NoiseChannel(
                            channel_type=NoiseChannelType.DEPOLARIZING_2Q,
                            probability=error_prob,
                            qubits=(q1, q2),
                        ))
        
        elif gate_name in {"M", "MX", "MY", "MZ", "MR"}:
            for q in qubits:
                p0_0, p1_1 = self.calibration.get_readout_fidelity(q)
                # Symmetric readout error approximation
                error_prob = ((1 - p0_0) + (1 - p1_1)) / 2 * self.error_scaling
                if error_prob > 0:
                    channels.append(NoiseChannel(
                        channel_type=NoiseChannelType.MEASUREMENT,
                        probability=error_prob,
                        qubits=(q,),
                        stim_instruction=f"X_ERROR({error_prob}) {q}",
                    ))
        
        return channels


class OperationNoiseModel(ABC):
    """Abstract base class for per-operation noise models.
    
    Provides fine-grained noise modeling at the physical operation
    level, allowing different noise for the same gate depending on
    context (which qubits, timing, etc.).
    """
    
    @abstractmethod
    def noise_for_operation(
        self,
        operation: "PhysicalOperation",
        context: Dict[str, Any],
    ) -> List[NoiseChannel]:
        """Get noise channels for an operation given context.
        
        Parameters
        ----------
        operation : PhysicalOperation
            The operation.
        context : Dict[str, Any]
            Context including timing, qubit state, etc.
            
        Returns
        -------
        List[NoiseChannel]
            Noise channels to apply.
        """
        ...


class IdleNoiseModel(ABC):
    """Abstract base class for idle noise (decoherence during waiting).
    
    Models T1/T2 decay and other decoherence effects while qubits
    wait for operations on other qubits.
    """
    
    def __init__(
        self,
        t1_default: float = 100.0,
        t2_default: float = 50.0,
    ):
        self.t1_default = t1_default
        self.t2_default = t2_default
    
    @abstractmethod
    def noise_for_idle(
        self,
        qubit: int,
        duration: float,
        t1: Optional[float] = None,
        t2: Optional[float] = None,
    ) -> List[NoiseChannel]:
        """Get noise channels for idle time.
        
        Parameters
        ----------
        qubit : int
            The idling qubit.
        duration : float
            Idle time in microseconds.
        t1 : Optional[float]
            T1 time (uses default if None).
        t2 : Optional[float]
            T2 time (uses default if None).
            
        Returns
        -------
        List[NoiseChannel]
            Decoherence noise channels.
        """
        ...
    
    def t1_error_probability(self, duration: float, t1: float) -> float:
        """Calculate T1 error probability."""
        if t1 <= 0:
            return 0.0
        return (1.0 - np.exp(-duration / t1)) / 2
    
    def t2_error_probability(self, duration: float, t2: float) -> float:
        """Calculate T2 (dephasing) error probability."""
        if t2 <= 0:
            return 0.0
        return (1.0 - np.exp(-duration / t2)) / 2
