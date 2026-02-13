# src/qectostim/experiments/hardware_simulation/core/operations.py
"""
Physical operation abstractions.

Defines operations that can be performed on hardware, including
gates, transport, measurement, and idle time. Each operation
carries timing and fidelity information for noise modeling.
"""
from __future__ import annotations

import logging
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
    Sequence,
    Set,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.gates import GateSpec, ParameterizedGate
    from qectostim.experiments.hardware_simulation.core.components import PhysicalQubit


_logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Categories of physical operations."""
    GATE_1Q = auto()        # Single-qubit gate
    GATE_2Q = auto()        # Two-qubit gate  
    GATE_MULTI = auto()     # Multi-qubit gate
    GATE_GLOBAL = auto()    # Global gate (all qubits simultaneously)
    MEASUREMENT = auto()    # Qubit measurement
    RESET = auto()          # Qubit reset/initialization
    TRANSPORT = auto()      # Qubit movement (mobile architectures)
    REARRANGEMENT = auto()  # Parallel position changes (neutral atom)
    IDLE = auto()           # Waiting/decoherence
    RECOVERY = auto()       # Platform-specific state recovery (cooling, T1 reset, etc.)
    LOSS = auto()           # Particle loss event (neutral atom, ion loss)
    BARRIER = auto()        # Synchronization barrier


@dataclass
class OperationResult:
    """Result of executing a physical operation.
    
    Attributes
    ----------
    success : bool
        Whether the operation completed successfully.
    duration : float
        Actual duration in microseconds.
    fidelity : float
        Achieved fidelity (1.0 = perfect).
    error_channel : Optional[str]
        Type of error channel to apply in simulation.
    error_probability : float
        Probability of error for the noise model.
    measurement_result : Optional[int]
        Result if this was a measurement (0 or 1).
    metadata : Dict[str, Any]
        Additional operation-specific results.
    """
    success: bool = True
    duration: float = 0.0
    fidelity: float = 1.0
    error_channel: Optional[str] = None
    error_probability: float = 0.0
    measurement_result: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PhysicalOperation(ABC):
    """Abstract base class for physical operations.
    
    All operations that can be performed on the hardware inherit from this.
    Operations know their duration, fidelity model, and affected qubits.
    
    Inspired by the Operation classes in qccd_operations.py.
    
    Attributes
    ----------
    operation_type : OperationType
        Category of operation.
    qubits : Tuple[int, ...]
        Physical qubit indices affected by this operation.
    duration : float
        Nominal duration in microseconds.
    """
    
    def __init__(
        self,
        operation_type: OperationType,
        qubits: Tuple[int, ...],
        duration: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.operation_type = operation_type
        self.qubits = qubits
        self._duration = duration
        self.metadata = metadata or {}
    
    @property
    def duration(self) -> float:
        """Get the operation duration in microseconds."""
        return self._duration
    
    @abstractmethod
    def fidelity(self, **context) -> float:
        """Calculate the operation fidelity given context.
        
        Parameters
        ----------
        **context
            Platform-specific context (e.g., heating rate, chain length).
            
        Returns
        -------
        float
            Fidelity in [0, 1] where 1 is perfect.
        """
        ...
    
    @abstractmethod
    def error_channel(self) -> str:
        """Get the Stim error channel name for this operation.
        
        Returns
        -------
        str
            Stim noise channel (e.g., "DEPOLARIZE1", "DEPOLARIZE2", "Z_ERROR").
        """
        ...
    
    @abstractmethod
    def to_stim_instructions(self) -> List[str]:
        """Convert to Stim circuit instructions.
        
        Returns
        -------
        List[str]
            Stim instruction strings (gate + noise).
        """
        ...
    
    def affected_qubits(self) -> Tuple[int, ...]:
        """Get the qubits affected by this operation."""
        return self.qubits
    
    def is_applicable(self, **context) -> bool:
        """Check if this operation can be applied in current context.
        
        Override in subclasses for platform-specific constraints.
        """
        return True
    
    def execute(self, **context) -> OperationResult:
        """Execute the operation and return results.
        
        Default implementation returns nominal values.
        Override for platform-specific execution logic.
        """
        fid = self.fidelity(**context)
        return OperationResult(
            success=True,
            duration=self.duration,
            fidelity=fid,
            error_channel=self.error_channel(),
            error_probability=1.0 - fid,
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(qubits={self.qubits}, duration={self.duration})"


class GateOperation(PhysicalOperation):
    """A quantum gate operation.
    
    Wraps a gate specification with physical execution properties.
    
    Attributes
    ----------
    gate : GateSpec or ParameterizedGate
        The gate being applied.
    qubits : Tuple[int, ...]
        Target qubit indices.
    base_fidelity : float
        Baseline gate fidelity (before decoherence effects).
    """
    
    def __init__(
        self,
        gate: Union["GateSpec", "ParameterizedGate"],
        qubits: Tuple[int, ...],
        duration: float = 1.0,
        base_fidelity: float = 0.999,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Determine operation type from gate
        num_qubits = gate.num_qubits
        if num_qubits == 1:
            op_type = OperationType.GATE_1Q
        elif num_qubits == 2:
            op_type = OperationType.GATE_2Q
        else:
            op_type = OperationType.GATE_MULTI
        
        super().__init__(op_type, qubits, duration, metadata)
        self.gate = gate
        self.base_fidelity = base_fidelity
    
    @property
    def gate_name(self) -> str:
        """Get the gate name."""
        from qectostim.experiments.hardware_simulation.core.gates import GateSpec
        return self.gate.name if isinstance(self.gate, GateSpec) else self.gate.spec.name
    
    def fidelity(self, **context) -> float:
        """Calculate gate fidelity including decoherence effects."""
        # Base implementation - subclasses add platform-specific effects
        return self.base_fidelity
    
    def error_channel(self) -> str:
        """Get error channel based on gate type."""
        if self.operation_type == OperationType.GATE_1Q:
            return "DEPOLARIZE1"
        elif self.operation_type == OperationType.GATE_2Q:
            return "DEPOLARIZE2"
        else:
            return "DEPOLARIZE1"  # Approximate for multi-qubit
    
    def to_stim_instructions(self) -> List[str]:
        """Convert to Stim instructions."""
        from qectostim.experiments.hardware_simulation.core.gates import GateSpec
        
        gate_name = self.gate.to_stim_name() if isinstance(self.gate, GateSpec) else self.gate.spec.to_stim_name()
        qubit_str = " ".join(str(q) for q in self.qubits)
        
        instructions = [f"{gate_name} {qubit_str}"]
        
        # Add noise if fidelity < 1
        error_prob = 1.0 - self.base_fidelity
        if error_prob > 0:
            channel = self.error_channel()
            instructions.append(f"{channel}({error_prob}) {qubit_str}")
        
        return instructions


class TransportOperation(PhysicalOperation):
    """Qubit transport operation (for mobile/reconfigurable architectures).
    
    Models physical movement of qubits between zones.
    
    Attributes
    ----------
    qubit : int
        The qubit being transported.
    source_zone : str
        Starting zone ID.
    target_zone : str
        Destination zone ID.
    path : List[str]
        Sequence of zone IDs traversed.
    error_rate : float
        Generic error rate per microsecond of transport.
        Platforms override fidelity() for physics-specific models.
    """
    
    def __init__(
        self,
        qubit: int,
        source_zone: str,
        target_zone: str,
        duration: float = 10.0,
        path: Optional[List[str]] = None,
        error_rate: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(OperationType.TRANSPORT, (qubit,), duration, metadata)
        self.source_zone = source_zone
        self.target_zone = target_zone
        self.path = path or [source_zone, target_zone]
        self.error_rate = error_rate
    
    def fidelity(self, **context) -> float:
        """Base fidelity: generic error_rate * duration model.
        
        Platforms override for physics-specific models (e.g. heating-based
        dephasing for trapped ions, T1 decay for superconducting, etc.).
        """
        if self.error_rate <= 0:
            return 1.0
        return max(0.0, 1.0 - self.error_rate * self.duration)
    
    def error_channel(self) -> str:
        """Transport primarily causes dephasing."""
        return "Z_ERROR"
    
    def to_stim_instructions(self) -> List[str]:
        """Transport is modeled as noise only."""
        instructions = []
        error_prob = 1.0 - self.fidelity()
        if error_prob > 0:
            instructions.append(f"Z_ERROR({error_prob}) {self.qubits[0]}")
        return instructions


class MeasurementOperation(PhysicalOperation):
    """Qubit measurement operation.
    
    Attributes
    ----------
    qubit : int
        The qubit being measured.
    basis : str
        Measurement basis ("Z", "X", "Y").
    readout_fidelity : float
        Probability of correct readout.
    destructive : bool
        Whether measurement destroys the qubit state.
    """
    
    def __init__(
        self,
        qubit: int,
        basis: str = "Z",
        duration: float = 1.0,
        readout_fidelity: float = 0.99,
        destructive: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(OperationType.MEASUREMENT, (qubit,), duration, metadata)
        self.basis = basis.upper()
        self.readout_fidelity = readout_fidelity
        self.destructive = destructive
    
    def fidelity(self, **context) -> float:
        """Measurement fidelity is the readout fidelity."""
        return self.readout_fidelity
    
    def error_channel(self) -> str:
        """Measurement error is a bit-flip on the classical result."""
        return "X_ERROR"  # Applied before measurement in Stim
    
    def to_stim_instructions(self) -> List[str]:
        """Convert to Stim measurement instruction."""
        instructions = []
        
        # Add measurement error
        error_prob = 1.0 - self.readout_fidelity
        if error_prob > 0:
            instructions.append(f"X_ERROR({error_prob}) {self.qubits[0]}")
        
        # Add measurement
        basis_map = {"Z": "M", "X": "MX", "Y": "MY"}
        meas_gate = basis_map.get(self.basis, "M")
        instructions.append(f"{meas_gate} {self.qubits[0]}")
        
        return instructions


class IdleOperation(PhysicalOperation):
    """Idle time operation (decoherence during waiting).
    
    Models T1/T2 decay while qubit waits for other operations.
    
    Attributes
    ----------
    qubit : int
        The idling qubit.
    t1_time : Optional[float]
        T1 relaxation time in microseconds.
    t2_time : Optional[float]
        T2 dephasing time in microseconds.
    """
    
    def __init__(
        self,
        qubit: int,
        duration: float,
        t1_time: Optional[float] = None,
        t2_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(OperationType.IDLE, (qubit,), duration, metadata)
        self.t1_time = t1_time
        self.t2_time = t2_time
    
    def fidelity(self, **context) -> float:
        """Calculate idle fidelity from T1/T2 decay."""
        fid = 1.0
        
        # T1 decay (amplitude damping)
        if self.t1_time and self.t1_time > 0:
            t1_decay = 1.0 - (1.0 - np.exp(-self.duration / self.t1_time)) / 2
            fid *= t1_decay
        
        # T2 decay (dephasing)
        if self.t2_time and self.t2_time > 0:
            t2_decay = (1.0 + np.exp(-self.duration / self.t2_time)) / 2
            fid *= t2_decay
        
        return fid
    
    def error_channel(self) -> str:
        """Idle errors are primarily dephasing."""
        return "Z_ERROR"
    
    def to_stim_instructions(self) -> List[str]:
        """Convert idle to Stim noise instructions."""
        instructions = []
        
        # T2 dephasing
        if self.t2_time and self.t2_time > 0:
            dephasing_prob = (1.0 - np.exp(-self.duration / self.t2_time)) / 2
            if dephasing_prob > 0:
                instructions.append(f"Z_ERROR({dephasing_prob}) {self.qubits[0]}")
        
        # T1 decay (approximated as depolarizing)
        if self.t1_time and self.t1_time > 0:
            decay_prob = (1.0 - np.exp(-self.duration / self.t1_time)) / 2
            if decay_prob > 0:
                instructions.append(f"DEPOLARIZE1({decay_prob}) {self.qubits[0]}")
        
        return instructions


class ResetOperation(PhysicalOperation):
    """Qubit reset/initialization operation.
    
    Attributes
    ----------
    qubit : int
        The qubit being reset.
    target_state : str
        Target state ("|0⟩", "|1⟩", "|+⟩", etc.).
    reset_fidelity : float
        Probability of correct initialization.
    """
    
    def __init__(
        self,
        qubit: int,
        target_state: str = "|0⟩",
        duration: float = 1.0,
        reset_fidelity: float = 0.999,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(OperationType.RESET, (qubit,), duration, metadata)
        self.target_state = target_state
        self.reset_fidelity = reset_fidelity
    
    def fidelity(self, **context) -> float:
        return self.reset_fidelity
    
    def error_channel(self) -> str:
        return "X_ERROR"
    
    def to_stim_instructions(self) -> List[str]:
        instructions = ["R " + str(self.qubits[0])]
        
        error_prob = 1.0 - self.reset_fidelity
        if error_prob > 0:
            instructions.append(f"X_ERROR({error_prob}) {self.qubits[0]}")
        
        return instructions


class BarrierOperation(PhysicalOperation):
    """Synchronization barrier.
    
    Forces all specified qubits to synchronize at this point.
    No physical effect, but affects scheduling.
    """
    
    def __init__(
        self,
        qubits: Tuple[int, ...],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(OperationType.BARRIER, qubits, 0.0, metadata)
    
    def fidelity(self, **context) -> float:
        return 1.0  # No error
    
    def error_channel(self) -> str:
        return ""  # No noise
    
    def to_stim_instructions(self) -> List[str]:
        return ["TICK"]  # Stim's barrier equivalent


class GlobalOperation(PhysicalOperation):
    """Global operation acting on all (or many) qubits simultaneously.
    
    Many quantum platforms support truly global operations that act on
    all qubits at once with the same duration as a single-qubit gate:
    
    - Neutral atoms: Global Rydberg pulses for parallel entanglement
    - Trapped ions: Global microwave/optical addressing
    - NMR: RF pulses
    
    This is distinct from GATE_MULTI which acts on a specific subset.
    GlobalOperation indicates the operation is inherently parallel.
    
    Attributes
    ----------
    gate_name : str
        Name of the global gate (e.g., "GLOBAL_RZ", "GLOBAL_R").
    qubits : Tuple[int, ...]
        All qubits affected (may be all qubits in the system).
    angle : Optional[float]
        Rotation angle if this is a rotation gate.
    axis : Optional[str]
        Rotation axis ("X", "Y", "Z") if applicable.
    base_fidelity : float
        Baseline fidelity of the global operation.
    """
    
    def __init__(
        self,
        gate_name: str,
        qubits: Tuple[int, ...],
        duration: float = 1.0,
        angle: Optional[float] = None,
        axis: Optional[str] = None,
        base_fidelity: float = 0.999,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(OperationType.GATE_GLOBAL, qubits, duration, metadata)
        self._gate_name = gate_name
        self.angle = angle
        self.axis = axis
        self.base_fidelity = base_fidelity
    
    @property
    def gate_name(self) -> str:
        return self._gate_name
    
    def fidelity(self, **context) -> float:
        """Calculate fidelity - may decrease with number of qubits."""
        # Simple model: fidelity drops slightly per qubit
        per_qubit_factor = context.get("per_qubit_fidelity_factor", 1.0)
        return self.base_fidelity * (per_qubit_factor ** len(self.qubits))
    
    def error_channel(self) -> str:
        """Global operations typically cause correlated errors."""
        return "DEPOLARIZE1"  # Applied to each qubit
    
    def to_stim_instructions(self) -> List[str]:
        """Convert to Stim instructions."""
        instructions = []
        
        # Map common global gates to Stim equivalents
        stim_name = self._gate_name
        if self.angle is not None and self.axis:
            # Parameterized rotation - approximate or use exact if supported
            if self.axis == "Z":
                stim_name = "RZ" if "RZ" in dir() else "Z"
            elif self.axis == "X":
                stim_name = "RX" if "RX" in dir() else "H"
            elif self.axis == "Y":
                stim_name = "RY" if "RY" in dir() else "Y"
        
        qubit_str = " ".join(str(q) for q in self.qubits)
        instructions.append(f"{stim_name} {qubit_str}")
        
        # Add noise
        error_prob = 1.0 - self.base_fidelity
        if error_prob > 0:
            for q in self.qubits:
                instructions.append(f"DEPOLARIZE1({error_prob}) {q}")
        
        return instructions


class LossOperation(PhysicalOperation):
    """Models particle loss events.
    
    In many physical systems, qubits can be lost entirely:
    - Neutral atoms: Escape from optical traps
    - Trapped ions: Ion loss from heating or collisions
    - Photonic: Photon loss in transmission
    
    This operation represents the loss event itself and triggers
    appropriate error handling (e.g., erasure error, circuit abort).
    
    Unlike other errors, loss is detectable (heralded) in many systems,
    making it an erasure error rather than a Pauli error.
    
    Attributes
    ----------
    qubit : int
        The qubit that was lost.
    loss_probability : float
        Probability that loss occurred (for modeling).
    heralded : bool
        Whether the loss is detectable (heralded erasure).
    replacement_possible : bool
        Whether the qubit can be replaced (e.g., reload atom).
    replacement_time : float
        Time to replace the qubit if possible.
    """
    
    def __init__(
        self,
        qubit: int,
        loss_probability: float = 0.0,
        heralded: bool = True,
        replacement_possible: bool = False,
        replacement_time: float = 0.0,
        duration: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(OperationType.LOSS, (qubit,), duration, metadata)
        self.loss_probability = loss_probability
        self.heralded = heralded
        self.replacement_possible = replacement_possible
        self.replacement_time = replacement_time
    
    def fidelity(self, **context) -> float:
        """Loss fidelity is 1 - loss_probability."""
        return 1.0 - self.loss_probability
    
    def error_channel(self) -> str:
        """Loss is an erasure error if heralded, else depolarizing."""
        if self.heralded:
            return "E"  # Erasure (Stim supports PAULI_CHANNEL for this)
        return "DEPOLARIZE1"
    
    def to_stim_instructions(self) -> List[str]:
        """Convert to Stim instructions."""
        instructions = []
        
        if self.loss_probability > 0:
            if self.heralded:
                # Model erasure as PAULI_CHANNEL_1 with specific probs
                # In Stim, we can model heralded loss specially
                # For now, use depolarizing as approximation
                instructions.append(
                    f"DEPOLARIZE1({self.loss_probability}) {self.qubits[0]}"
                )
            else:
                instructions.append(
                    f"DEPOLARIZE1({self.loss_probability}) {self.qubits[0]}"
                )
        
        return instructions


class RearrangementOperation(PhysicalOperation):
    """Parallel position rearrangement operation.
    
    In reconfigurable systems, multiple particles can be moved
    simultaneously in a single operation cycle:
    
    - Neutral atoms: AOD-based parallel atom moves
    - Trapped ions: Parallel ion shuttling in WISE architecture
    
    This is more general than TransportOperation which moves one qubit.
    RearrangementOperation describes a permutation applied in parallel.
    
    Attributes
    ----------
    moves : Dict[int, Tuple[int, int]]
        Mapping from qubit_id to (from_position, to_position).
    collision_free : bool
        Whether the moves are verified collision-free.
    """
    
    def __init__(
        self,
        moves: Dict[int, Tuple[Any, Any]],
        duration: float = 10.0,
        base_fidelity: float = 0.999,
        collision_free: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        qubits = tuple(moves.keys())
        super().__init__(OperationType.REARRANGEMENT, qubits, duration, metadata)
        self.moves = moves
        self.base_fidelity = base_fidelity
        self.collision_free = collision_free
    
    def fidelity(self, **context) -> float:
        """Fidelity depends on number of moves and distances."""
        # Simple model: fidelity decreases per move
        per_move_fidelity = context.get("per_move_fidelity", 0.9999)
        return self.base_fidelity * (per_move_fidelity ** len(self.moves))
    
    def error_channel(self) -> str:
        """Rearrangement primarily causes dephasing."""
        return "Z_ERROR"
    
    def to_stim_instructions(self) -> List[str]:
        """Convert to Stim noise instructions (moves don't change gates)."""
        instructions = []
        error_prob = 1.0 - self.fidelity()
        if error_prob > 0:
            for q in self.qubits:
                instructions.append(f"Z_ERROR({error_prob}) {q}")
        return instructions


# =============================================================================
# Transport Error Model Protocol
# =============================================================================

@runtime_checkable
class TransportErrorModel(Protocol):
    """Protocol for modeling errors during qubit transport/movement.
    
    Different platforms have different transport error mechanisms:
    - Trapped ions: Motional heating during shuttling
    - Neutral atoms: Atom loss probability during moves
    - Superconducting (if mobile): Flux noise during frequency tuning
    
    This protocol abstracts the error model so routing algorithms
    can query transport costs without platform knowledge.
    
    Example Implementation
    ----------------------
    >>> class TrappedIonTransportError:
    ...     def __init__(self, heating_rate: float):
    ...         self.heating_rate = heating_rate
    ...     
    ...     def transport_error_probability(self, distance, duration):
    ...         quanta = self.heating_rate * duration / 1e6
    ...         return 1.0 - math.exp(-quanta * 0.01)
    """
    
    def transport_error_probability(
        self,
        distance: float,
        duration: float,
        **context,
    ) -> float:
        """Calculate error probability for a transport operation.
        
        Parameters
        ----------
        distance : float
            Distance traveled (in platform units).
        duration : float
            Duration of transport in microseconds.
        **context
            Platform-specific parameters.
            
        Returns
        -------
        float
            Probability of error during transport.
        """
        ...
    
    def transport_fidelity(
        self,
        distance: float,
        duration: float,
        **context,
    ) -> float:
        """Calculate fidelity of a transport operation.
        
        Default implementation: 1 - error_probability.
        """
        ...
    
    def loss_probability(
        self,
        distance: float,
        duration: float,
        **context,
    ) -> float:
        """Calculate probability of qubit loss during transport.
        
        For platforms where loss is possible (neutral atom, some ion traps).
        Returns 0.0 if loss is not applicable.
        """
        ...


# =============================================================================
# Operation Batching and Scheduling
# =============================================================================

@dataclass
class OperationBatch:
    """A batch of operations that can be executed together.
    
    Used for:
    - Parallel gate execution
    - WISE global reconfigurations (all transports in one batch)
    - Grouped measurements
    
    Attributes
    ----------
    operations : List[PhysicalOperation]
        Operations in this batch.
    batch_type : str
        Type of batch ("parallel_gates", "transport", "measurement", etc.).
    barrier_before : bool
        Require synchronization before batch.
    barrier_after : bool
        Require synchronization after batch.
    metadata : Dict[str, Any]
        Additional batch properties.
    """
    operations: List[PhysicalOperation] = field(default_factory=list)
    batch_type: str = "parallel"
    barrier_before: bool = False
    barrier_after: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Duration is the max of all operations (parallel execution)."""
        if not self.operations:
            return 0.0
        return max(op.duration for op in self.operations)
    
    @property
    def total_duration(self) -> float:
        """Total duration if executed sequentially."""
        return sum(op.duration for op in self.operations)
    
    @property
    def parallelism(self) -> int:
        """Number of operations in the batch."""
        return len(self.operations)
    
    def combined_fidelity(self, **context) -> float:
        """Calculate combined fidelity of all operations."""
        fid = 1.0
        for op in self.operations:
            fid *= op.fidelity(**context)
        return fid
    
    def affected_qubits(self) -> Set[int]:
        """Get all qubits affected by this batch."""
        qubits = set()
        for op in self.operations:
            qubits.update(op.qubits)
        return qubits
    
    def add_operation(self, op: PhysicalOperation) -> None:
        """Add an operation to the batch."""
        self.operations.append(op)
    
    def can_add(self, op: PhysicalOperation) -> bool:
        """Check if operation can be added without conflicts.
        
        No qubit can be used by more than one operation.
        """
        op_qubits = set(op.qubits)
        return not op_qubits.intersection(self.affected_qubits())
    
    def to_stim_instructions(self) -> List[str]:
        """Convert batch to Stim instructions."""
        instructions = []
        if self.barrier_before:
            instructions.append("TICK")
        
        for op in self.operations:
            instructions.extend(op.to_stim_instructions())
        
        if self.barrier_after:
            instructions.append("TICK")
        
        return instructions
    
    def __len__(self) -> int:
        return len(self.operations)
    
    def __iter__(self):
        return iter(self.operations)


@dataclass
class DependencyEdge:
    """An edge in the operation dependency DAG.
    
    Attributes
    ----------
    source : int
        Index of source operation (must complete first).
    target : int
        Index of target operation (depends on source).
    dependency_type : str
        Type of dependency ("qubit", "zone", "barrier", etc.).
    """
    source: int
    target: int
    dependency_type: str = "qubit"


class BatchScheduler(ABC):
    """Abstract scheduler for batching and ordering operations.
    
    Builds a happens-before DAG from operation conflicts and
    computes optimal parallel scheduling.
    
    Used by:
    - WISE scheduler: batch by operation type, global barriers
    - Standard scheduler: maximize parallelism
    - Critical-path scheduler: minimize total time
    
    Subclasses implement different scheduling strategies.
    """
    
    def __init__(self, name: str = "scheduler"):
        self.name = name
    
    @abstractmethod
    def build_dependency_dag(
        self,
        operations: List[PhysicalOperation],
    ) -> List[DependencyEdge]:
        """Build the happens-before dependency graph.
        
        Operations on the same qubit or zone must be ordered.
        
        Parameters
        ----------
        operations : List[PhysicalOperation]
            Operations to schedule.
            
        Returns
        -------
        List[DependencyEdge]
            Dependency edges between operations.
        """
        ...
    
    @abstractmethod
    def schedule(
        self,
        operations: List[PhysicalOperation],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[OperationBatch]:
        """Schedule operations into batches.
        
        Parameters
        ----------
        operations : List[PhysicalOperation]
            Operations to schedule.
        constraints : Optional[Dict]
            Scheduling constraints (max parallelism, etc.).
            
        Returns
        -------
        List[OperationBatch]
            Ordered batches of operations.
        """
        ...
    
    def compute_critical_path(
        self,
        operations: List[PhysicalOperation],
        dependencies: List[DependencyEdge],
    ) -> float:
        """Compute the critical path length (minimum total time).
        
        Parameters
        ----------
        operations : List[PhysicalOperation]
            Operations to analyze.
        dependencies : List[DependencyEdge]
            Dependency edges.
            
        Returns
        -------
        float
            Critical path duration in microseconds.
        """
        n = len(operations)
        if n == 0:
            return 0.0
        
        # Build adjacency list
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        in_degree = [0] * n
        
        for edge in dependencies:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1
        
        # Topological sort with longest path
        earliest_end = [0.0] * n
        
        # Find sources (no dependencies)
        queue = [i for i in range(n) if in_degree[i] == 0]
        
        while queue:
            node = queue.pop(0)
            earliest_end[node] = earliest_end[node] + operations[node].duration
            
            for neighbor in adj[node]:
                earliest_end[neighbor] = max(
                    earliest_end[neighbor],
                    earliest_end[node],
                )
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return max(earliest_end) if earliest_end else 0.0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class GreedyBatchScheduler(BatchScheduler):
    """Greedy scheduler that maximizes parallelism.
    
    Schedules operations as early as possible, batching all
    operations that have no conflicts.
    """
    
    def __init__(self, name: str = "greedy_scheduler"):
        super().__init__(name)
    
    def build_dependency_dag(
        self,
        operations: List[PhysicalOperation],
    ) -> List[DependencyEdge]:
        """Build DAG based on qubit conflicts."""
        edges = []
        n = len(operations)
        
        for i in range(n):
            qubits_i = set(operations[i].qubits)
            for j in range(i + 1, n):
                qubits_j = set(operations[j].qubits)
                if qubits_i.intersection(qubits_j):
                    edges.append(DependencyEdge(i, j, "qubit"))
        
        return edges
    
    def schedule(
        self,
        operations: List[PhysicalOperation],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[OperationBatch]:
        """Schedule greedily, maximizing parallelism."""
        if not operations:
            return []
        
        n = len(operations)
        constraints = constraints or {}
        max_parallel = constraints.get("max_parallel", float('inf'))
        
        # Build dependency info
        dependencies = self.build_dependency_dag(operations)
        
        # Track which operations are scheduled
        scheduled = [False] * n
        dep_count = [0] * n  # Number of unmet dependencies
        
        for edge in dependencies:
            dep_count[edge.target] += 1
        
        batches = []
        
        while not all(scheduled):
            # Find all operations that can be scheduled
            ready = [
                i for i in range(n)
                if not scheduled[i] and dep_count[i] == 0
            ]
            
            if not ready:
                # Shouldn't happen with valid DAG
                break
            
            # Create batch with non-conflicting operations
            batch = OperationBatch()
            just_scheduled = []
            
            for idx in ready:
                if len(batch) >= max_parallel:
                    break
                if batch.can_add(operations[idx]):
                    batch.add_operation(operations[idx])
                    scheduled[idx] = True
                    just_scheduled.append(idx)
            
            # Update dependencies only for ops scheduled THIS round
            for idx in just_scheduled:
                for edge in dependencies:
                    if edge.source == idx:
                        dep_count[edge.target] -= 1
            
            if batch.operations:
                batches.append(batch)
        
        return batches


class BarrierAwareScheduler(BatchScheduler):
    """Scheduler that respects synchronization barriers.

    Splits operations into segments at barrier boundaries, then
    schedules each segment independently using a delegate scheduler.
    This ensures that the parallelized schedule never reorders
    operations across barriers, preserving circuit semantics.

    Parameters
    ----------
    delegate : Optional[BatchScheduler]
        Inner scheduler used within each barrier segment.
        Defaults to :class:`GreedyBatchScheduler`.
    name : str
        Scheduler identifier.

    Example
    -------
    >>> scheduler = BarrierAwareScheduler()
    >>> batches = scheduler.schedule(operations, constraints={"barriers": [5, 12, 20]})
    """

    def __init__(
        self,
        delegate: Optional[BatchScheduler] = None,
        name: str = "barrier_aware_scheduler",
    ) -> None:
        super().__init__(name)
        self.delegate = delegate or GreedyBatchScheduler()

    def build_dependency_dag(
        self,
        operations: List[PhysicalOperation],
    ) -> List[DependencyEdge]:
        """Delegate DAG building to the inner scheduler."""
        return self.delegate.build_dependency_dag(operations)

    def schedule(
        self,
        operations: List[PhysicalOperation],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[OperationBatch]:
        """Schedule with explicit barrier positions.

        Barrier positions are passed via ``constraints["barriers"]`` as
        a list of operation indices.  Operations are split at those
        indices, each segment is scheduled independently, and a barrier
        flag separates the segments in the output.

        Parameters
        ----------
        operations : List[PhysicalOperation]
            Operations to schedule.
        constraints : Optional[Dict]
            Must contain ``"barriers"`` key with a list of int indices.

        Returns
        -------
        List[OperationBatch]
            Ordered batches, with barrier flags between segments.
        """
        if not operations:
            return []

        constraints = constraints or {}
        barrier_positions = sorted(set(constraints.get("barriers", [])))

        if not barrier_positions:
            return self.delegate.schedule(operations, constraints)

        # Split into segments
        segments: List[List[PhysicalOperation]] = []
        prev = 0
        for bp in barrier_positions:
            if bp > prev:
                segments.append(operations[prev:bp])
            prev = bp
        if prev < len(operations):
            segments.append(operations[prev:])

        # Schedule each segment independently
        all_batches: List[OperationBatch] = []
        for i, segment in enumerate(segments):
            if not segment:
                continue
            seg_batches = self.delegate.schedule(segment, constraints)
            if i > 0 and seg_batches:
                seg_batches[0] = OperationBatch(
                    operations=seg_batches[0].operations,
                    batch_type=seg_batches[0].batch_type,
                    barrier_before=True,
                    barrier_after=seg_batches[0].barrier_after,
                )
            all_batches.extend(seg_batches)

        return all_batches


class TypeOrderedBatchScheduler(BatchScheduler):
    """Scheduler that groups operations by type with configurable ordering.
    
    This is a technology-agnostic scheduler that groups operations by their
    type (gate, transport, measurement, etc.) and executes groups in a
    specified order with barriers between them.
    
    The type ordering is fully configurable, making this scheduler suitable
    for any platform that benefits from type-based batching:
    - QCCD trapped ions: transport → gates → measurement
    - Neutral atoms: rearrangement → gates → measurement
    - Superconducting: gates by locality for crosstalk
    
    Within each type group, operations on the same qubit are serialized
    while non-conflicting operations run in parallel.
    
    Parameters
    ----------
    name : str
        Scheduler identifier.
    type_order : Optional[List[OperationType]]
        Ordered list of operation types. Operations are scheduled in this
        order with barriers between type groups. Types not in the list
        are scheduled last.
        
    Example
    -------
    >>> # Trapped ion ordering
    >>> ion_order = [OperationType.TRANSPORT, OperationType.GATE_2Q, 
    ...              OperationType.GATE_1Q, OperationType.MEASUREMENT]
    >>> scheduler = TypeOrderedBatchScheduler(type_order=ion_order)
    >>> batches = scheduler.schedule(operations)
    """
    
    # Default type ordering (generic, suitable for most platforms)
    DEFAULT_TYPE_ORDER = [
        OperationType.TRANSPORT,
        OperationType.REARRANGEMENT,
        OperationType.GATE_2Q,
        OperationType.GATE_MULTI,
        OperationType.GATE_1Q,
        OperationType.MEASUREMENT,
        OperationType.RESET,
        OperationType.RECOVERY,
    ]
    
    def __init__(
        self,
        name: str = "type_ordered_scheduler",
        type_order: Optional[List[OperationType]] = None,
    ) -> None:
        """Initialize the type-ordered batch scheduler.
        
        Parameters
        ----------
        name : str
            Scheduler name for identification.
        type_order : Optional[List[OperationType]]
            Custom order of operation types. Uses DEFAULT_TYPE_ORDER if None.
        """
        super().__init__(name)
        self.type_order = type_order or self.DEFAULT_TYPE_ORDER
    
    def build_dependency_dag(
        self,
        operations: List[PhysicalOperation],
    ) -> List[DependencyEdge]:
        """Build DAG with type-based grouping.
        
        Only intra-group qubit conflicts create dependencies since
        type groups are separated by barriers.
        """
        edges = []
        
        # Group operations by type
        type_groups: Dict[OperationType, List[int]] = {}
        for i, op in enumerate(operations):
            if op.operation_type not in type_groups:
                type_groups[op.operation_type] = []
            type_groups[op.operation_type].append(i)
        
        # Add dependencies within each type group (same-qubit conflicts)
        for op_type, indices in type_groups.items():
            for i, idx_i in enumerate(indices):
                qubits_i = set(operations[idx_i].qubits)
                for idx_j in indices[i + 1:]:
                    qubits_j = set(operations[idx_j].qubits)
                    if qubits_i.intersection(qubits_j):
                        edges.append(DependencyEdge(idx_i, idx_j, "qubit"))
        
        return edges
    
    def schedule(
        self,
        operations: List[PhysicalOperation],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[OperationBatch]:
        """Schedule with type-based batching and global barriers.
        
        Parameters
        ----------
        operations : List[PhysicalOperation]
            Operations to schedule.
        constraints : Optional[Dict]
            Scheduling constraints:
            - max_parallel: Maximum ops per batch
            - type_order: Override the instance's type order
            
        Returns
        -------
        List[OperationBatch]
            Ordered batches with barriers between type groups.
        """
        if not operations:
            return []
        
        constraints = constraints or {}
        type_order = constraints.get("type_order", self.type_order)
        
        # Group operations by type
        type_groups: Dict[OperationType, List[PhysicalOperation]] = {}
        other_ops: List[PhysicalOperation] = []
        
        for op in operations:
            if op.operation_type in type_order:
                if op.operation_type not in type_groups:
                    type_groups[op.operation_type] = []
                type_groups[op.operation_type].append(op)
            else:
                other_ops.append(op)
        
        batches = []
        
        # Create batches in type order with barriers
        for op_type in type_order:
            if op_type in type_groups:
                ops_of_type = type_groups[op_type]
                
                # Split based on qubit conflicts
                sub_batches = self._split_conflicting_ops(
                    ops_of_type,
                    constraints.get("max_parallel", float("inf")),
                )
                
                for i, sub_ops in enumerate(sub_batches):
                    batch = OperationBatch(
                        operations=sub_ops,
                        batch_type=op_type.name.lower(),
                        barrier_before=(i == 0),  # First sub-batch gets barrier
                        barrier_after=(i == len(sub_batches) - 1),
                    )
                    batches.append(batch)
        
        # Add other operations at the end
        if other_ops:
            batch = OperationBatch(
                operations=other_ops,
                batch_type="other",
                barrier_before=True,
            )
            batches.append(batch)
        
        return batches
    
    def _split_conflicting_ops(
        self,
        operations: List[PhysicalOperation],
        max_parallel: float,
    ) -> List[List[PhysicalOperation]]:
        """Split operations with qubit conflicts into separate batches.
        
        Operations that share qubits cannot run in parallel.
        """
        if not operations:
            return []
        
        batches: List[List[PhysicalOperation]] = []
        remaining = list(operations)
        
        while remaining:
            current_batch: List[PhysicalOperation] = []
            used_qubits: Set[int] = set()
            still_remaining = []
            
            for op in remaining:
                op_qubits = set(op.qubits)
                if (not op_qubits.intersection(used_qubits) and
                        len(current_batch) < max_parallel):
                    current_batch.append(op)
                    used_qubits.update(op_qubits)
                else:
                    still_remaining.append(op)
            
            if current_batch:
                batches.append(current_batch)
            remaining = still_remaining
        
        return batches


# WISEBatchScheduler is in trapped_ion/scheduling.py as it uses WISE-specific
# type ordering. It inherits from TypeOrderedBatchScheduler.


# Import numpy for idle fidelity calculations
import numpy as np
