"""
Trapped Ion QCCD Operations.

This module defines the operations that can be performed in a trapped ion
QCCD system, including:
- Transport operations (Split, Merge, Move, JunctionCrossing)
- Crystal manipulation (CrystalRotation, Cooling)
- Quantum operations (SingleQubitGate, MSGate, Measurement, Reset)
- Batch reconfiguration (GlobalReconfiguration)

Each operation has timing models, fidelity calculations, and heating effects
based on published experimental data.

Ported and refactored from the original qccd_operations.py implementation.
"""

from __future__ import annotations

import abc
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

from .components import (
    Crossing,
    Ion,
    Junction,
    ManipulationTrap,
    QCCDComponent,
    QCCDOperation,
    QCCDWISEConfig,
    QubitIon,
    Trap,
    CoolingIon,
    DEFAULT_TIMINGS,
    DEFAULT_HEATING_RATES,
)


# =============================================================================
# Operation Base Classes
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


class QCCDOperationBase(abc.ABC):
    """Abstract base class for all QCCD operations.
    
    Operations represent physical actions performed on the trapped ion system.
    Each operation has:
    - Timing: How long it takes
    - Fidelity: How accurately it performs the intended action
    - Heating: Motional energy added to ions
    - Applicability: Conditions under which it can be performed
    
    Subclasses implement specific operations like Split, Merge, gates, etc.
    """
    
    # Class-level operation type
    OPERATION_TYPE: QCCDOperation
    
    # Default timing constants (can be overridden)
    DEFAULT_TIME_US: float = 0.0
    DEFAULT_HEATING_RATE: float = 0.0
    
    # Dephasing time constant (T2) in seconds
    T2_TIME: float = 2.2
    
    def __init__(
        self,
        involved_components: Sequence[QCCDComponent],
        **kwargs: Any,
    ) -> None:
        """Initialize an operation.
        
        Parameters
        ----------
        involved_components : Sequence[QCCDComponent]
            Components participating in this operation.
        **kwargs
            Additional operation-specific parameters.
        """
        self._involved_components = list(involved_components)
        self._kwargs = dict(kwargs)
        self._result: Optional[OperationResult] = None
    
    @property
    def involved_components(self) -> List[QCCDComponent]:
        """Components involved in this operation."""
        return self._involved_components
    
    @property
    def operation_type(self) -> QCCDOperation:
        """The type of this operation."""
        return self.OPERATION_TYPE
    
    @property
    def result(self) -> Optional[OperationResult]:
        """Result of the last execution, or None if not executed."""
        return self._result
    
    @abc.abstractmethod
    def is_applicable(self) -> bool:
        """Check if this operation can be performed.
        
        Returns
        -------
        bool
            True if all preconditions are met.
        """
        # Check that all components allow this operation type
        for component in self._involved_components:
            if self.OPERATION_TYPE not in component.allowed_operations:
                return False
        return True
    
    @abc.abstractmethod
    def validate(self) -> Optional[str]:
        """Validate operation preconditions.
        
        Returns
        -------
        Optional[str]
            Error message if validation fails, None if valid.
        """
        for component in self._involved_components:
            if self.OPERATION_TYPE not in component.allowed_operations:
                return (
                    f"Component {component} does not allow "
                    f"{self.OPERATION_TYPE.name}"
                )
        return None
    
    @abc.abstractmethod
    def calculate_time(self) -> float:
        """Calculate operation duration in microseconds."""
        return self.DEFAULT_TIME_US
    
    @abc.abstractmethod
    def calculate_fidelity(self) -> float:
        """Calculate operation fidelity."""
        return 1.0
    
    def calculate_dephasing_fidelity(self) -> float:
        """Calculate fidelity loss due to dephasing.
        
        Uses exponential decay model: F = 1 - (1 - exp(-t/T2))/2
        """
        t = self.calculate_time() * 1e-6  # Convert to seconds
        return 1.0 - (1.0 - np.exp(-t / self.T2_TIME)) / 2.0
    
    @abc.abstractmethod
    def calculate_heating(self) -> float:
        """Calculate motional quanta added."""
        return self.DEFAULT_HEATING_RATE * self.calculate_time()
    
    @abc.abstractmethod
    def _execute(self) -> None:
        """Execute the operation (modify state)."""
        ...
    
    def execute(self) -> OperationResult:
        """Execute the operation with full validation.
        
        Returns
        -------
        OperationResult
            Result of the operation including timing and fidelity.
        """
        # Validate preconditions
        error = self.validate()
        if error:
            self._result = OperationResult(
                success=False,
                error_message=error,
            )
            return self._result
        
        # Calculate metrics
        time_us = self.calculate_time()
        fidelity = self.calculate_fidelity()
        dephasing = self.calculate_dephasing_fidelity()
        heating = self.calculate_heating()
        
        # Execute the operation
        try:
            self._execute()
            self._result = OperationResult(
                success=True,
                time_us=time_us,
                fidelity=fidelity,
                dephasing_fidelity=dephasing,
                heating_added=heating,
            )
        except Exception as e:
            self._result = OperationResult(
                success=False,
                time_us=time_us,
                error_message=str(e),
            )
        
        return self._result
    
    @property
    @abc.abstractmethod
    def label(self) -> str:
        """Human-readable label for this operation."""
        ...
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.label})"


class TransportOperation(QCCDOperationBase):
    """Base class for ion transport operations.
    
    Transport operations move ions between locations in the trap network.
    They typically add heating but don't directly affect quantum state.
    """
    
    def calculate_fidelity(self) -> float:
        """Transport fidelity is unity (noise via heating model)."""
        return 1.0


class CrystalOperation(QCCDOperationBase):
    """Base class for operations on ion crystals in a trap.
    
    Crystal operations manipulate the arrangement of ions within
    a single trap segment.
    """
    
    def __init__(
        self,
        trap: Trap,
        **kwargs: Any,
    ) -> None:
        super().__init__(involved_components=[trap], **kwargs)
        self._trap = trap
    
    @property
    def trap(self) -> Trap:
        return self._trap
    
    @property
    def ions(self) -> List[Ion]:
        """Ions affected by this operation."""
        return self._trap.ions
    
    def calculate_fidelity(self) -> float:
        """Crystal operation fidelity (noise via heating model)."""
        return 1.0


class QubitOperation(QCCDOperationBase):
    """Base class for quantum operations on qubit ions.
    
    Qubit operations affect the quantum state of ions and must
    account for gate errors, dephasing, and motional heating.
    """
    
    # Scaling factor for fidelity calculation from arXiv:2004.04706
    FIDELITY_SCALE_A: float = 0.003680029
    
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
# Transport Operations
# =============================================================================

class Split(TransportOperation):
    """Split an ion from a trap into a crossing.
    
    Removes the edge ion from a trap and places it in the adjacent crossing.
    
    Physical parameters from TABLE I, arXiv:2004.04706:
    - Duration: 80 μs
    - Heating: 6 quanta
    """
    
    OPERATION_TYPE = QCCDOperation.SPLIT
    DEFAULT_TIME_US = 80.0  # microseconds
    DEFAULT_HEATING_RATE = 6.0  # quanta per operation
    
    def __init__(
        self,
        trap: Trap,
        crossing: Crossing,
        ion: Optional[Ion] = None,
    ) -> None:
        super().__init__(involved_components=[trap, crossing])
        self._trap = trap
        self._crossing = crossing
        self._ion = ion
    
    @property
    def trap(self) -> Trap:
        return self._trap
    
    @property
    def crossing(self) -> Crossing:
        return self._crossing
    
    @property
    def ion(self) -> Optional[Ion]:
        return self._ion
    
    def is_applicable(self) -> bool:
        if not self._crossing.connects(self._trap):
            return False
        if self._crossing.is_occupied:
            return False
        if self._trap.is_empty:
            return False
        return super().is_applicable()
    
    def validate(self) -> Optional[str]:
        if not self._crossing.connects(self._trap):
            return f"Crossing {self._crossing.label} not connected to {self._trap.label}"
        if self._crossing.is_occupied:
            return f"Crossing {self._crossing.label} already occupied"
        if self._trap.is_empty:
            return f"Trap {self._trap.label} has no ions to split"
        return super().validate()
    
    def calculate_time(self) -> float:
        return self.DEFAULT_TIME_US
    
    def calculate_heating(self) -> float:
        return self.DEFAULT_HEATING_RATE
    
    def _execute(self) -> None:
        # Get edge ion to split
        ion = self._ion or self._trap.get_edge_ion(-1)
        if ion is None:
            raise ValueError("No ion to split from trap")
        
        # Remove from trap and place in crossing
        self._trap.remove_ion(ion)
        self._crossing.set_ion(ion, self._trap)
        
        # Add heating to both trap and ion
        heating = self.calculate_heating()
        self._trap.distribute_heating(heating)
        ion.add_motional_energy(heating)
    
    @property
    def label(self) -> str:
        ion_label = self._ion.label if self._ion else "edge"
        return f"Split({self._trap.label}→{self._crossing.label}, {ion_label})"


class Merge(TransportOperation):
    """Merge an ion from a crossing into a trap.
    
    Takes an ion from a crossing and adds it to the adjacent trap.
    
    Physical parameters from TABLE I, arXiv:2004.04706:
    - Duration: 80 μs
    - Heating: 6 quanta
    """
    
    OPERATION_TYPE = QCCDOperation.MERGE
    DEFAULT_TIME_US = 80.0
    DEFAULT_HEATING_RATE = 6.0
    
    def __init__(
        self,
        trap: Trap,
        crossing: Crossing,
    ) -> None:
        super().__init__(involved_components=[trap, crossing])
        self._trap = trap
        self._crossing = crossing
    
    @property
    def trap(self) -> Trap:
        return self._trap
    
    @property
    def crossing(self) -> Crossing:
        return self._crossing
    
    def is_applicable(self) -> bool:
        if not self._crossing.connects(self._trap):
            return False
        if not self._crossing.is_occupied:
            return False
        if self._trap.is_full:
            return False
        return super().is_applicable()
    
    def validate(self) -> Optional[str]:
        if not self._crossing.connects(self._trap):
            return f"Crossing {self._crossing.label} not connected to {self._trap.label}"
        if not self._crossing.is_occupied:
            return f"Crossing {self._crossing.label} has no ion to merge"
        if self._trap.is_full:
            return f"Trap {self._trap.label} is at capacity"
        return super().validate()
    
    def calculate_time(self) -> float:
        return self.DEFAULT_TIME_US
    
    def calculate_heating(self) -> float:
        return self.DEFAULT_HEATING_RATE
    
    def _execute(self) -> None:
        # Get ion from crossing
        ion = self._crossing.remove_ion()
        
        # Add to trap
        self._trap.add_ion(ion)
        
        # Add heating
        self._trap.distribute_heating(self.calculate_heating())
    
    @property
    def label(self) -> str:
        ion_label = self._crossing.ion.label if self._crossing.ion else "?"
        return f"Merge({self._crossing.label}→{self._trap.label}, {ion_label})"


class Move(TransportOperation):
    """Move an ion through a crossing.
    
    Advances an ion from one end of a crossing toward the other.
    
    Physical parameters from TABLE I, arXiv:2004.04706:
    - Duration: 5 μs
    - Heating: 0.1 quanta
    """
    
    OPERATION_TYPE = QCCDOperation.MOVE
    DEFAULT_TIME_US = 5.0
    DEFAULT_HEATING_RATE = 0.1
    
    def __init__(
        self,
        crossing: Crossing,
    ) -> None:
        super().__init__(involved_components=[crossing])
        self._crossing = crossing
    
    @property
    def crossing(self) -> Crossing:
        return self._crossing
    
    def is_applicable(self) -> bool:
        return self._crossing.is_occupied and super().is_applicable()
    
    def validate(self) -> Optional[str]:
        if not self._crossing.is_occupied:
            return f"Crossing {self._crossing.label} has no ion to move"
        return super().validate()
    
    def calculate_time(self) -> float:
        return self.DEFAULT_TIME_US
    
    def calculate_heating(self) -> float:
        return self.DEFAULT_HEATING_RATE
    
    def _execute(self) -> None:
        # Move ion to opposite end
        self._crossing.move_ion()
        
        # Add heating
        if self._crossing.ion:
            self._crossing.ion.add_motional_energy(self.calculate_heating())
    
    @property
    def label(self) -> str:
        ion_label = self._crossing.ion.label if self._crossing.ion else "?"
        return f"Move({self._crossing.label}, {ion_label})"


class JunctionCrossing(TransportOperation):
    """Cross a junction between crossings.
    
    Moves an ion between a crossing and an adjacent junction.
    
    Physical parameters from TABLE I, arXiv:2004.04706:
    - Duration: 50 μs
    - Heating: 3 quanta
    """
    
    OPERATION_TYPE = QCCDOperation.JUNCTION_CROSSING
    DEFAULT_TIME_US = 50.0
    DEFAULT_HEATING_RATE = 3.0
    
    def __init__(
        self,
        junction: Junction,
        crossing: Crossing,
    ) -> None:
        super().__init__(involved_components=[junction, crossing])
        self._junction = junction
        self._crossing = crossing
    
    @property
    def junction(self) -> Junction:
        return self._junction
    
    @property
    def crossing(self) -> Crossing:
        return self._crossing
    
    def is_applicable(self) -> bool:
        if not self._crossing.connects(self._junction):
            return False
        # Need ion in crossing XOR in junction
        has_crossing_ion = self._crossing.is_occupied
        has_junction_ion = not self._junction.is_empty
        if not (has_crossing_ion or has_junction_ion):
            return False
        if has_crossing_ion and self._junction.is_full:
            return False
        return super().is_applicable()
    
    def validate(self) -> Optional[str]:
        if not self._crossing.connects(self._junction):
            return f"Crossing {self._crossing.label} not connected to {self._junction.label}"
        has_crossing_ion = self._crossing.is_occupied
        has_junction_ion = not self._junction.is_empty
        if not (has_crossing_ion or has_junction_ion):
            return "Neither crossing nor junction has an ion"
        if has_crossing_ion and self._junction.is_full:
            return f"Junction {self._junction.label} is at capacity"
        return super().validate()
    
    def calculate_time(self) -> float:
        return self.DEFAULT_TIME_US
    
    def calculate_heating(self) -> float:
        return self.DEFAULT_HEATING_RATE
    
    def _execute(self) -> None:
        if self._crossing.is_occupied:
            # Move from crossing to junction
            ion = self._crossing.remove_ion()
            self._junction.add_ion(ion)
        else:
            # Move from junction to crossing
            ion = self._junction.remove_ion()
            self._crossing.set_ion(ion, self._junction)
        
        # Add heating
        ion.add_motional_energy(self.calculate_heating())
    
    @property
    def label(self) -> str:
        if self._crossing.is_occupied:
            return f"JunctionCrossing({self._crossing.label}→{self._junction.label})"
        else:
            return f"JunctionCrossing({self._junction.label}→{self._crossing.label})"


# =============================================================================
# Crystal Operations
# =============================================================================

class CrystalRotation(CrystalOperation):
    """Rotate the order of ions in a trap.
    
    Reverses the order of ions in the trap (first becomes last, etc.).
    
    Physical parameters from TABLE IV, PRA 99, 022330:
    - Duration: 42 μs
    - Heating: 0.3 quanta
    """
    
    OPERATION_TYPE = QCCDOperation.CRYSTAL_ROTATION
    DEFAULT_TIME_US = 42.0
    DEFAULT_HEATING_RATE = 0.3
    
    def is_applicable(self) -> bool:
        return len(self._trap.ions) >= 2 and super().is_applicable()
    
    def validate(self) -> Optional[str]:
        if len(self._trap.ions) < 2:
            return "Need at least 2 ions to rotate"
        return super().validate()
    
    def calculate_time(self) -> float:
        return self.DEFAULT_TIME_US
    
    def calculate_heating(self) -> float:
        return self.DEFAULT_HEATING_RATE
    
    def _execute(self) -> None:
        # Reverse ion order
        ions = list(self._trap.ions)
        for ion in ions:
            self._trap.remove_ion(ion)
        for ion in reversed(ions):
            self._trap.add_ion(ion)
        
        # Add heating
        self._trap.distribute_heating(self.calculate_heating())
    
    @property
    def label(self) -> str:
        return f"CrystalRotation({self._trap.label})"


class SympatheticCooling(CrystalOperation):
    """Cool ions via sympathetic cooling.
    
    Transfers motional energy from qubit ions to cooling ions.
    
    Physical parameters from TABLE IV, PRA 99, 022330:
    - Duration: 400 μs
    - Heating: 0.1 quanta (net cooling effect is larger)
    """
    
    OPERATION_TYPE = QCCDOperation.RECOOLING
    DEFAULT_TIME_US = 400.0
    DEFAULT_HEATING_RATE = 0.1
    
    def is_applicable(self) -> bool:
        return self._trap.has_cooling_ion and super().is_applicable()
    
    def validate(self) -> Optional[str]:
        if not self._trap.has_cooling_ion:
            return f"Trap {self._trap.label} has no cooling ion"
        return super().validate()
    
    def calculate_time(self) -> float:
        return self.DEFAULT_TIME_US
    
    def calculate_heating(self) -> float:
        return self.DEFAULT_HEATING_RATE
    
    def _execute(self) -> None:
        # Cool the trap (transfers energy to cooling ions)
        self._trap.cool()
        
        # Small heating from cooling process
        self._trap.distribute_heating(self.calculate_heating())
    
    @property
    def label(self) -> str:
        return f"SympatheticCooling({self._trap.label})"


# =============================================================================
# Qubit Operations
# =============================================================================

class SingleQubitGate(QubitOperation):
    """Single-qubit rotation gate.
    
    Performs a rotation on a single qubit ion.
    
    Physical parameters from TABLE IV, PRA 99, 022330:
    - Duration: 5 μs
    - Infidelity: ~3e-3 (depends on chain length and heating)
    """
    
    OPERATION_TYPE = QCCDOperation.ONE_QUBIT_GATE
    DEFAULT_TIME_US = 5.0
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
        """
        if self._trap is None:
            return 1.0 - 3e-3  # Default infidelity
        
        n = len(self._trap.ions)
        if n >= 2:
            n_eff = n / np.log(n)
        else:
            n_eff = 1
        
        heating = self._trap.motional_energy
        time_s = self.calculate_time() * 1e-6
        background_heating = self._trap.BACKGROUND_HEATING_RATE * time_s
        
        infidelity = (
            background_heating +
            self.FIDELITY_SCALE_A * n_eff * (2 * heating + 1)
        )
        
        return max(0.0, 1.0 - infidelity)
    
    def _execute(self) -> None:
        # Gate execution is handled by the simulator
        # This just validates and tracks metrics
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
    DEFAULT_TIME_US = 40.0
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
        """Calculate gate time based on variant and ion distance."""
        # Simplified model - actual timing depends on ion separation
        return self.DEFAULT_TIME_US
    
    def calculate_heating(self) -> float:
        return self.DEFAULT_HEATING_RATE
    
    def calculate_fidelity(self) -> float:
        """Calculate MS gate fidelity.
        
        From arXiv:2004.04706, Page 7.
        """
        if self._trap is None:
            return 1.0 - 2e-3
        
        n = len(self._trap.ions)
        if n >= 2:
            n_eff = n / np.log(n)
        else:
            n_eff = 1
        
        heating = self._trap.motional_energy
        time_s = self.calculate_time() * 1e-6
        background_heating = self._trap.BACKGROUND_HEATING_RATE * time_s
        
        infidelity = (
            background_heating +
            self.FIDELITY_SCALE_A * n_eff * (2 * heating + 1)
        )
        
        return max(0.0, 1.0 - infidelity)
    
    def _execute(self) -> None:
        # Gate execution is handled by the simulator
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
        # Three MS gates for the swap
        self._ms_gates = [
            MSGate(ion1, ion2, trap=trap) for _ in range(3)
        ]
    
    def calculate_time(self) -> float:
        return sum(g.calculate_time() for g in self._ms_gates)
    
    def calculate_heating(self) -> float:
        return sum(g.calculate_heating() for g in self._ms_gates)
    
    def calculate_fidelity(self) -> float:
        return np.prod([g.calculate_fidelity() for g in self._ms_gates])
    
    def calculate_dephasing_fidelity(self) -> float:
        return np.prod([g.calculate_dephasing_fidelity() for g in self._ms_gates])
    
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
        
        # Get current positions
        idx1 = self._trap.ions.index(ion1)
        
        # Remove and re-add in swapped positions
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
    DEFAULT_TIME_US = 400.0
    INFIDELITY = 1e-3
    
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
        # Measurements can be scheduled at end, minimal dephasing impact
        return 1.0
    
    def _execute(self) -> None:
        # Measurement handled by simulator
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
    DEFAULT_TIME_US = 50.0
    INFIDELITY = 5e-3
    
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
        # Reset handled by simulator
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

def create_split(trap: Trap, crossing: Crossing, ion: Optional[Ion] = None) -> Split:
    """Create a split operation."""
    return Split(trap, crossing, ion)


def create_merge(trap: Trap, crossing: Crossing) -> Merge:
    """Create a merge operation."""
    return Merge(trap, crossing)


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
