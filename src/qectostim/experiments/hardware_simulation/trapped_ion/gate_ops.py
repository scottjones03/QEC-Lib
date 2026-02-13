# src/qectostim/experiments/hardware_simulation/trapped_ion/gate_ops.py
"""
Qubit-level gate operations for trapped ion QCCD systems.

Extracted from operations.py — these are the quantum gate operations
(as opposed to transport operations, which live in transport.py).

Classes
-------
OperationResult   — result of executing an operation
QCCDOperationBase — abstract base for all QCCD operations
QubitOperation    — base for qubit-level gate operations
SingleQubitGate   — single-qubit rotation
MSGate            — Mølmer-Sørensen two-qubit entangling gate
GateSwap          — physical SWAP via three MS gates
Measurement       — qubit measurement
QubitReset        — qubit reset to |0⟩
ReconfigurationStep / GlobalReconfiguration — batch reconfiguration helpers

Physical parameters from:
* arXiv:2004.04706 (Honeywell QCCD) — Table I, pages 6–7
* PRA 99, 022330 (Bermudez et al.) — Table IV
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from qectostim.experiments.hardware_simulation.core.operations import (
    PhysicalOperation,
    OperationType,
    OperationResult as CoreOperationResult,
)
from .architecture import (
    Crossing,
    Ion,
    Junction,
    ManipulationTrap,
    QCCDNode as QCCDComponent,
    QCCDOperationType as QCCDOperation,
    QCCDWISEConfig,
    QubitIon,
    CoolingIon,
)
from .physics import (
    DEFAULT_CALIBRATION as _CAL,
    DEFAULT_FIDELITY_MODEL as _FIDELITY,
    DEFAULT_TIMINGS,
    DEFAULT_HEATING_RATES,
)

# Trap is a union type for type hints
Trap = ManipulationTrap  # For backward compatibility


_logger = logging.getLogger(__name__)


# =============================================================================
# Operation Result
# =============================================================================

@dataclass
class OperationResult:
    """Result of executing an operation.

    Attributes
    ----------
    success : bool
        Whether the operation completed successfully.
    time_us : float
        Duration of the operation in microseconds.
    fidelity : float
        Operation fidelity (1.0 = perfect).
    dephasing_fidelity : float
        Fidelity loss due to dephasing.
    heating_added : float
        Motional quanta added.
    error_message : Optional[str]
        Error description if operation failed.
    """
    success: bool = True
    time_us: float = 0.0
    fidelity: float = 1.0
    dephasing_fidelity: float = 1.0
    heating_added: float = 0.0
    error_message: Optional[str] = None


# =============================================================================
# Operation Base Classes
# =============================================================================

class QCCDOperationBase(PhysicalOperation):
    """Abstract base class for all QCCD operations.

    Extends core.operations.PhysicalOperation with trapped-ion specific
    features like heating tracking and component-based applicability checks.
    
    Operations represent physical actions performed on the trapped ion system.
    Each operation has:
    - Timing: How long it takes
    - Fidelity: How accurately it performs the intended action
    - Heating: Motional energy added to ions
    - Applicability: Conditions under which it can be performed

    Subclasses implement specific operations like Split, Merge, gates, etc.
    
    Bridge to PhysicalOperation
    ---------------------------
    - fidelity() → calls calculate_fidelity()
    - error_channel() → returns appropriate Stim channel
    - to_stim_instructions() → generates Stim instructions
    - duration property → calls calculate_time()
    """

    # Class-level operation type
    OPERATION_TYPE: QCCDOperation

    # Default timing constants (can be overridden)
    DEFAULT_TIME_US: float = 0.0
    DEFAULT_HEATING_RATE: float = 0.0

    def __init__(
        self,
        involved_components: Optional[List[QCCDComponent]] = None,
        config: Optional[QCCDWISEConfig] = None,
        operation_type: OperationType = OperationType.GATE_1Q,
    ) -> None:
        self._involved_components: List[QCCDComponent] = involved_components or []
        self._config = config
        
        # Extract qubit indices from involved components (ions)
        qubit_indices = tuple(
            c.idx for c in self._involved_components 
            if hasattr(c, 'idx') and isinstance(c, Ion)
        )
        
        # Initialize parent PhysicalOperation
        super().__init__(
            operation_type=operation_type,
            qubits=qubit_indices,
            duration=0.0,  # Will be computed dynamically
        )

    @property
    def involved_components(self) -> List[QCCDComponent]:
        """Get all components involved in this operation."""
        return self._involved_components
    
    @property
    def duration(self) -> float:
        """Override to compute duration dynamically."""
        return self.calculate_time()

    @abc.abstractmethod
    def calculate_time(self) -> float:
        """Calculate operation duration in microseconds."""
        ...

    @abc.abstractmethod
    def calculate_heating(self) -> float:
        """Calculate motional quanta added by this operation."""
        ...

    def calculate_fidelity(self) -> float:
        """Calculate operation fidelity (override in subclasses)."""
        return 1.0

    def calculate_dephasing_fidelity(self) -> float:
        """Calculate dephasing fidelity contribution."""
        return 1.0

    # --- PhysicalOperation abstract method implementations ---
    
    def fidelity(self, **context) -> float:
        """Calculate the operation fidelity given context.
        
        Bridges to calculate_fidelity() with optional context handling.
        """
        return self.calculate_fidelity() * self.calculate_dephasing_fidelity()
    
    def error_channel(self) -> str:
        """Get the Stim error channel name for this operation.
        
        Returns appropriate channel based on operation type.
        """
        if self.operation_type == OperationType.GATE_2Q:
            return "DEPOLARIZE2"
        elif self.operation_type == OperationType.MEASUREMENT:
            return "X_ERROR"  # Measurement error
        elif self.operation_type == OperationType.TRANSPORT:
            return "DEPOLARIZE1"  # Transport dephasing
        else:
            return "DEPOLARIZE1"  # Default for 1Q gates
    
    def to_stim_instructions(self) -> List[str]:
        """Convert to Stim circuit instructions.
        
        Override in subclasses for specific gate representations.
        """
        # Default: just return empty (subclasses provide specifics)
        return []

    def is_applicable(self) -> bool:
        """Check if the operation can currently be performed."""
        return True

    def validate(self) -> Optional[str]:
        """Validate operation preconditions.

        Returns
        -------
        Optional[str]
            Error message if invalid, None if valid.
        """
        return None

    def execute(self) -> OperationResult:
        """Execute the operation.

        Returns
        -------
        OperationResult
            Result of execution.
        """
        error = self.validate()
        if error:
            return OperationResult(success=False, error_message=error)
        if not self.is_applicable():
            return OperationResult(
                success=False,
                error_message="Operation not applicable in current state",
            )

        time_us = self.calculate_time()
        fidelity = self.calculate_fidelity()
        dephasing = self.calculate_dephasing_fidelity()
        heating = self.calculate_heating()

        self._execute()

        return OperationResult(
            success=True,
            time_us=time_us,
            fidelity=fidelity,
            dephasing_fidelity=dephasing,
            heating_added=heating,
        )

    @abc.abstractmethod
    def _execute(self) -> None:
        """Internal execution logic (state mutation)."""
        ...

    @property
    @abc.abstractmethod
    def label(self) -> str:
        """Human-readable label for this operation."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.label})"


# =============================================================================
# Qubit Operation Base
# =============================================================================

class QubitOperation(QCCDOperationBase):
    """Base class for qubit-level quantum operations.

    These operate on specific qubit ions within a trap.

    Attributes
    ----------
    FIDELITY_SCALE_A : float
        Scaling constant for chain-length-dependent infidelity,
        from arXiv:2004.04706, page 7.
    """

    FIDELITY_SCALE_A: float = _CAL.fidelity_scaling_A

    def __init__(
        self,
        ions: Sequence[QubitIon],
        trap: Optional[Trap] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(involved_components=list(ions), **kwargs)
        self._ions = list(ions)
        self._trap = trap

    @property
    def ions(self) -> List[QubitIon]:
        return self._ions

    @property
    def trap(self) -> Optional[Trap]:
        return self._trap

    def set_trap(self, trap: Trap) -> None:
        """Set the trap context for this operation."""
        self._trap = trap
        if trap not in self._involved_components:
            self._involved_components.append(trap)

    def is_applicable(self) -> bool:
        if self._trap is None:
            return False
        # All ions must be in the same trap
        for ion in self._ions:
            if ion.parent != self._trap:
                return False
        return super().is_applicable()

    def validate(self) -> Optional[str]:
        if self._trap is None:
            return "Trap not set for qubit operation"
        for ion in self._ions:
            if ion.parent != self._trap:
                return f"Ion {ion.label} not in trap {self._trap.label}"
        return super().validate()


# =============================================================================
# Qubit Operations
# =============================================================================

class SingleQubitGate(QubitOperation):
    """Single-qubit rotation gate.

    Physical parameters from TABLE IV, PRA 99, 022330:
    - Duration: 5 μs
    - Infidelity: ~3e-3 (depends on chain length and heating)
    """

    OPERATION_TYPE = QCCDOperation.ONE_QUBIT_GATE
    DEFAULT_TIME_US = _CAL.single_qubit_gate_time * 1e6
    DEFAULT_HEATING_RATE = 0.0  # No direct heating

    def __init__(
        self,
        ion: QubitIon,
        gate_type: str = "R",
        theta: float = 0.0,
        phi: float = 0.0,
        trap: Optional[Trap] = None,
    ) -> None:
        super().__init__(ions=[ion], trap=trap)
        self._gate_type = gate_type
        self._theta = theta
        self._phi = phi

    @property
    def gate_type(self) -> str:
        return self._gate_type

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def phi(self) -> float:
        return self._phi

    def calculate_time(self) -> float:
        return self.DEFAULT_TIME_US

    def calculate_heating(self) -> float:
        return self.DEFAULT_HEATING_RATE

    def calculate_fidelity(self) -> float:
        """Calculate gate fidelity based on chain length and heating.

        From arXiv:2004.04706, Page 7.
        Delegates to IonChainFidelityModel for consistency.
        """
        if self._trap is None:
            return _FIDELITY.single_qubit_gate_fidelity(2, 0.0)

        n = len(self._trap.ions)
        heating = self._trap.motional_energy
        return _FIDELITY.single_qubit_gate_fidelity(n, heating)

    def _execute(self) -> None:
        # Gate execution is handled by the simulator
        pass

    @property
    def label(self) -> str:
        return f"{self._gate_type}({self._ions[0].label}, θ={self._theta:.3f}, φ={self._phi:.3f})"


class MSGate(QubitOperation):
    """Mølmer-Sørensen two-qubit entangling gate.

    The native two-qubit gate for trapped ions, creating entanglement
    via shared motional modes.

    Physical parameters from arXiv:2004.04706:
    - Duration: 40 μs (can vary with gate type)
    - Infidelity: ~2e-3 (depends on chain and heating)
    """

    OPERATION_TYPE = QCCDOperation.TWO_QUBIT_MS_GATE
    DEFAULT_TIME_US = _CAL.ms_gate_time * 1e6
    DEFAULT_HEATING_RATE = 0.0

    def __init__(
        self,
        ion1: QubitIon,
        ion2: QubitIon,
        theta: float = np.pi / 4,
        phi: float = 0.0,
        gate_variant: str = "AM2",
        trap: Optional[Trap] = None,
    ) -> None:
        super().__init__(ions=[ion1, ion2], trap=trap)
        self._theta = theta
        self._phi = phi
        self._gate_variant = gate_variant

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def phi(self) -> float:
        return self._phi

    @property
    def gate_variant(self) -> str:
        return self._gate_variant

    def calculate_time(self) -> float:
        return self.DEFAULT_TIME_US

    def calculate_heating(self) -> float:
        return self.DEFAULT_HEATING_RATE

    def calculate_fidelity(self) -> float:
        """Calculate MS gate fidelity.

        From arXiv:2004.04706, Page 7.
        Delegates to IonChainFidelityModel for consistency.
        """
        if self._trap is None:
            return _FIDELITY.ms_gate_fidelity(2, 0.0)

        n = len(self._trap.ions)
        heating = self._trap.motional_energy
        return _FIDELITY.ms_gate_fidelity(n, heating)

    def _execute(self) -> None:
        pass

    @property
    def label(self) -> str:
        return f"MS({self._ions[0].label}, {self._ions[1].label}, θ={self._theta:.3f})"


class GateSwap(QubitOperation):
    """Swap two ion positions using MS gates.

    Performs a physical swap of two ion positions using three MS gates,
    following the protocol from arXiv:2004.04706, Fig. 5.
    """

    OPERATION_TYPE = QCCDOperation.GATE_SWAP

    def __init__(
        self,
        ion1: QubitIon,
        ion2: QubitIon,
        trap: Optional[Trap] = None,
    ) -> None:
        super().__init__(ions=[ion1, ion2], trap=trap)
        self._ms_gates = [
            MSGate(ion1, ion2, trap=trap) for _ in range(3)
        ]

    def calculate_time(self) -> float:
        return sum(g.calculate_time() for g in self._ms_gates)

    def calculate_heating(self) -> float:
        return sum(g.calculate_heating() for g in self._ms_gates)

    def calculate_fidelity(self) -> float:
        return float(np.prod([g.calculate_fidelity() for g in self._ms_gates]))

    def calculate_dephasing_fidelity(self) -> float:
        return float(np.prod([g.calculate_dephasing_fidelity() for g in self._ms_gates]))

    def set_trap(self, trap: Trap) -> None:
        super().set_trap(trap)
        for g in self._ms_gates:
            g.set_trap(trap)

    def _execute(self) -> None:
        if self._trap is None:
            raise ValueError("Trap not set")

        ion1, ion2 = self._ions
        if ion1 is ion2:
            return

        idx1 = self._trap.ions.index(ion1)

        self._trap.remove_ion(ion1)
        self._trap.add_ion(ion1, position=self._trap.ions.index(ion2) + 1)
        self._trap.remove_ion(ion2)
        self._trap.add_ion(ion2, position=idx1)

    @property
    def label(self) -> str:
        return f"GateSwap({self._ions[0].label}, {self._ions[1].label})"


class Measurement(QubitOperation):
    """Measure a qubit in the computational basis.

    Physical parameters from TABLE IV, PRA 99, 022330:
    - Duration: 400 μs
    - Infidelity: 1e-3
    """

    OPERATION_TYPE = QCCDOperation.MEASUREMENT
    DEFAULT_TIME_US = _CAL.measurement_time * 1e6
    INFIDELITY = _CAL.measurement_infidelity

    def __init__(
        self,
        ion: QubitIon,
        trap: Optional[Trap] = None,
    ) -> None:
        super().__init__(ions=[ion], trap=trap)

    def calculate_time(self) -> float:
        return self.DEFAULT_TIME_US

    def calculate_heating(self) -> float:
        return 0.0

    def calculate_fidelity(self) -> float:
        return 1.0 - self.INFIDELITY

    def calculate_dephasing_fidelity(self) -> float:
        return 1.0

    def _execute(self) -> None:
        pass

    @property
    def label(self) -> str:
        return f"Measure({self._ions[0].label})"


class QubitReset(QubitOperation):
    """Reset a qubit to the |0⟩ state.

    Physical parameters from TABLE IV, PRA 99, 022330:
    - Duration: 50 μs
    - Infidelity: 5e-3
    """

    OPERATION_TYPE = QCCDOperation.QUBIT_RESET
    DEFAULT_TIME_US = _CAL.reset_time * 1e6
    INFIDELITY = _CAL.reset_infidelity

    def __init__(
        self,
        ion: QubitIon,
        trap: Optional[Trap] = None,
    ) -> None:
        super().__init__(ions=[ion], trap=trap)

    def calculate_time(self) -> float:
        return self.DEFAULT_TIME_US

    def calculate_heating(self) -> float:
        return 0.0

    def calculate_fidelity(self) -> float:
        return 1.0 - self.INFIDELITY

    def _execute(self) -> None:
        pass

    @property
    def label(self) -> str:
        return f"Reset({self._ions[0].label})"


# =============================================================================
# Batch Reconfiguration (for WISE architecture)
# =============================================================================

@dataclass
class ReconfigurationStep:
    """A single step in a batch reconfiguration.

    Attributes
    ----------
    operations : List[QCCDOperationBase]
        Operations to execute in parallel.
    time_us : float
        Duration of this step.
    """
    operations: List[QCCDOperationBase] = field(default_factory=list)
    time_us: float = 0.0


@dataclass
class GlobalReconfiguration:
    """Batch reconfiguration of ions across the architecture.

    Used by WISE architectures to move multiple ions simultaneously
    to achieve a target layout.

    Attributes
    ----------
    steps : List[ReconfigurationStep]
        Sequence of parallel operation steps.
    total_time_us : float
        Total reconfiguration time.
    total_fidelity : float
        Combined fidelity of all operations.
    """
    steps: List[ReconfigurationStep] = field(default_factory=list)
    total_time_us: float = 0.0
    total_fidelity: float = 1.0

    def add_step(self, step: ReconfigurationStep) -> None:
        """Add a step to the reconfiguration."""
        self.steps.append(step)
        self.total_time_us += step.time_us

    @property
    def num_steps(self) -> int:
        return len(self.steps)


# =============================================================================
# Operation Factories
# =============================================================================

def create_ms_gate(
    ion1: QubitIon,
    ion2: QubitIon,
    trap: Trap,
    theta: float = np.pi / 4,
) -> MSGate:
    """Create an MS gate operation."""
    return MSGate(ion1, ion2, theta=theta, trap=trap)


def create_single_qubit_gate(
    ion: QubitIon,
    trap: Trap,
    gate_type: str = "R",
    theta: float = 0.0,
    phi: float = 0.0,
) -> SingleQubitGate:
    """Create a single-qubit gate operation."""
    return SingleQubitGate(ion, gate_type, theta, phi, trap)


def create_measurement(ion: QubitIon, trap: Trap) -> Measurement:
    """Create a measurement operation."""
    return Measurement(ion, trap)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base classes
    "OperationResult",
    "QCCDOperationBase",
    "QubitOperation",
    # Gate operations
    "SingleQubitGate",
    "MSGate",
    "GateSwap",
    "Measurement",
    "QubitReset",
    # Reconfiguration
    "ReconfigurationStep",
    "GlobalReconfiguration",
    # Factories
    "create_ms_gate",
    "create_single_qubit_gate",
    "create_measurement",
]
