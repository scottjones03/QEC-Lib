"""
Trapped Ion QCCD Operations — backward-compatibility shim.

**This module is deprecated for new code.**  Canonical definitions:

* **gate_ops.py** — Qubit-level quantum operations (QCCDOperationBase,
  QubitOperation, MSGate, SingleQubitGate, Measurement, QubitReset,
  GateSwap, ReconfigurationStep, GlobalReconfiguration, OperationResult)
* **transport.py** — Physics-only transport data objects (no state mutation)

This file retains the state-mutating transport + crystal operations
(Split, Merge, Move, JunctionCrossing, CrystalRotation,
SympatheticCooling) because they have no equivalent elsewhere.
All qubit operations are re-exported from gate_ops.py.
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
    DEFAULT_TIMINGS,
    DEFAULT_HEATING_RATES,
)

# Trap is an alias for backward compatibility
Trap = ManipulationTrap

# ---- Re-exports from gate_ops.py (canonical home) ----
from .gate_ops import (  # noqa: F401
    OperationResult,
    QCCDOperationBase,
    QubitOperation,
    SingleQubitGate,
    MSGate,
    GateSwap,
    Measurement,
    QubitReset,
    ReconfigurationStep,
    GlobalReconfiguration,
    create_ms_gate,
    create_single_qubit_gate,
    create_measurement,
)


_logger = logging.getLogger(__name__)


# =============================================================================
# Transport / Crystal Operation Base Classes (kept here — not in gate_ops)
# =============================================================================

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


# =============================================================================
# Transport Operations (state-mutating — unique to this module)
# =============================================================================

class Split(TransportOperation):
    """Split an ion from a trap into a crossing.

    Removes the edge ion from a trap and places it in the adjacent crossing.

    Physical parameters from TABLE I, arXiv:2004.04706:
    - Duration: 80 μs
    - Heating: 6 quanta
    """

    OPERATION_TYPE = QCCDOperation.SPLIT
    DEFAULT_TIME_US = _CAL.split_time * 1e6
    DEFAULT_HEATING_RATE = _CAL.split_heating_rate

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
        ion = self._ion or self._trap.get_edge_ion(-1)
        if ion is None:
            raise ValueError("No ion to split from trap")
        self._trap.remove_ion(ion)
        self._crossing.set_ion(ion, self._trap)
        heating = self.calculate_heating()
        self._trap.distribute_heating(heating)
        ion.add_motional_energy(heating)

    @property
    def label(self) -> str:
        ion_label = self._ion.label if self._ion else "edge"
        return f"Split({self._trap.label}→{self._crossing.label}, {ion_label})"


class Merge(TransportOperation):
    """Merge an ion from a crossing into a trap.

    Physical parameters from TABLE I, arXiv:2004.04706:
    - Duration: 80 μs
    - Heating: 6 quanta
    """

    OPERATION_TYPE = QCCDOperation.MERGE
    DEFAULT_TIME_US = _CAL.merge_time * 1e6
    DEFAULT_HEATING_RATE = _CAL.merge_heating_rate

    def __init__(self, trap: Trap, crossing: Crossing) -> None:
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
        ion = self._crossing.remove_ion()
        self._trap.add_ion(ion)
        self._trap.distribute_heating(self.calculate_heating())

    @property
    def label(self) -> str:
        ion_label = self._crossing.ion.label if self._crossing.ion else "?"
        return f"Merge({self._crossing.label}→{self._trap.label}, {ion_label})"


class Move(TransportOperation):
    """Move an ion through a crossing.

    Physical parameters from TABLE I, arXiv:2004.04706:
    - Duration: 5 μs  - Heating: 0.1 quanta
    """

    OPERATION_TYPE = QCCDOperation.MOVE
    DEFAULT_TIME_US = _CAL.shuttle_time * 1e6
    DEFAULT_HEATING_RATE = _CAL.shuttle_heating_rate

    def __init__(self, crossing: Crossing) -> None:
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
        self._crossing.move_ion()
        if self._crossing.ion:
            self._crossing.ion.add_motional_energy(self.calculate_heating())

    @property
    def label(self) -> str:
        ion_label = self._crossing.ion.label if self._crossing.ion else "?"
        return f"Move({self._crossing.label}, {ion_label})"


class JunctionCrossing(TransportOperation):
    """Cross a junction between crossings.

    Physical parameters from TABLE I, arXiv:2004.04706:
    - Duration: 50 μs  - Heating: 3 quanta
    """

    OPERATION_TYPE = QCCDOperation.JUNCTION_CROSSING
    DEFAULT_TIME_US = _CAL.junction_time * 1e6
    DEFAULT_HEATING_RATE = _CAL.junction_heating_rate

    def __init__(self, junction: Junction, crossing: Crossing) -> None:
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
            ion = self._crossing.remove_ion()
            self._junction.add_ion(ion)
        else:
            ion = self._junction.remove_ion()
            self._crossing.set_ion(ion, self._junction)
        ion.add_motional_energy(self.calculate_heating())

    @property
    def label(self) -> str:
        if self._crossing.is_occupied:
            return f"JunctionCrossing({self._crossing.label}→{self._junction.label})"
        return f"JunctionCrossing({self._junction.label}→{self._crossing.label})"


# =============================================================================
# Crystal Operations (state-mutating — unique to this module)
# =============================================================================

class CrystalRotation(CrystalOperation):
    """Rotate the order of ions in a trap.

    Physical parameters from TABLE IV, PRA 99, 022330:
    - Duration: 42 μs  - Heating: 0.3 quanta
    """

    OPERATION_TYPE = QCCDOperation.CRYSTAL_ROTATION
    DEFAULT_TIME_US = _CAL.rotation_time * 1e6
    DEFAULT_HEATING_RATE = _CAL.rotation_heating_rate

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
        ions = list(self._trap.ions)
        for ion in ions:
            self._trap.remove_ion(ion)
        for ion in reversed(ions):
            self._trap.add_ion(ion)
        self._trap.distribute_heating(self.calculate_heating())

    @property
    def label(self) -> str:
        return f"CrystalRotation({self._trap.label})"


class SympatheticCooling(CrystalOperation):
    """Cool ions via sympathetic cooling.

    Physical parameters from TABLE IV, PRA 99, 022330:
    - Duration: 400 μs  - Heating: 0.1 quanta (net cooling is larger)
    """

    OPERATION_TYPE = QCCDOperation.RECOOLING
    DEFAULT_TIME_US = _CAL.recool_time * 1e6
    DEFAULT_HEATING_RATE = _CAL.cooling_heating_rate

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
        self._trap.cool()
        self._trap.distribute_heating(self.calculate_heating())

    @property
    def label(self) -> str:
        return f"SympatheticCooling({self._trap.label})"


# =============================================================================
# Transport Operation Factories (kept here — transport-specific)
# =============================================================================

def create_split(trap: Trap, crossing: Crossing, ion: Optional[Ion] = None) -> Split:
    """Create a split operation."""
    return Split(trap, crossing, ion)


def create_merge(trap: Trap, crossing: Crossing) -> Merge:
    """Create a merge operation."""
    return Merge(trap, crossing)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Re-exported from gate_ops.py
    "OperationResult",
    "QCCDOperationBase",
    "QubitOperation",
    "SingleQubitGate",
    "MSGate",
    "GateSwap",
    "Measurement",
    "QubitReset",
    "ReconfigurationStep",
    "GlobalReconfiguration",
    "create_ms_gate",
    "create_single_qubit_gate",
    "create_measurement",
    # Transport / Crystal bases (defined here)
    "TransportOperation",
    "CrystalOperation",
    # Concrete transport ops
    "Split",
    "Merge",
    "Move",
    "JunctionCrossing",
    # Concrete crystal ops
    "CrystalRotation",
    "SympatheticCooling",
    # Transport factories
    "create_split",
    "create_merge",
]
