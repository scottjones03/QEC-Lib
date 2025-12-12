# src/qectostim/gadgets/pauli_frame.py
"""
Pauli frame tracking for fault-tolerant quantum computation.

In fault-tolerant QEC, many operations (like teleportation and surgery)
produce classical measurement outcomes that determine Pauli corrections.
Instead of applying these corrections as physical gates, we track them
in a classical "Pauli frame" and propagate them through subsequent
operations.

Key concepts:
- Pauli frame: Classical record of pending X and Z corrections per qubit
- Frame propagation: How corrections transform through Clifford gates
- Frame collapse: Applying accumulated corrections at measurement

This approach:
- Avoids extra noisy gate operations
- Simplifies circuit structure
- Naturally handles measurement-dependent corrections

Example usage:
    >>> from qectostim.gadgets.pauli_frame import PauliFrame, PauliTracker
    >>> 
    >>> tracker = PauliTracker(num_logical_qubits=2)
    >>> 
    >>> # Record a Z correction on qubit 0
    >>> tracker.apply_z(0)
    >>> 
    >>> # Propagate through CNOT(0, 1)
    >>> tracker.propagate_cnot(control=0, target=1)
    >>> 
    >>> # Get current frame
    >>> frame = tracker.get_frame()
    >>> print(frame)  # Shows X/Z corrections on each qubit
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Tuple,
    Set,
)
from enum import Enum
import numpy as np


class PauliType(Enum):
    """Types of Pauli operators."""
    I = "I"
    X = "X"
    Y = "Y"
    Z = "Z"


@dataclass
class PauliFrame:
    """
    Classical record of Pauli corrections for a set of logical qubits.
    
    Each qubit has two bits: one for X correction, one for Z correction.
    The actual Pauli is determined by (X, Z):
    - (0, 0) → I
    - (1, 0) → X
    - (0, 1) → Z
    - (1, 1) → Y (= iXZ)
    
    Attributes:
        num_qubits: Number of logical qubits tracked
        x_frame: Bit vector of X corrections (True = X applied)
        z_frame: Bit vector of Z corrections (True = Z applied)
    """
    num_qubits: int
    x_frame: List[bool] = field(default_factory=list)
    z_frame: List[bool] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize frames if not provided."""
        if not self.x_frame:
            self.x_frame = [False] * self.num_qubits
        if not self.z_frame:
            self.z_frame = [False] * self.num_qubits
    
    def get_pauli(self, qubit: int) -> PauliType:
        """Get the Pauli operator for a specific qubit."""
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {self.num_qubits})")
        
        x = self.x_frame[qubit]
        z = self.z_frame[qubit]
        
        if not x and not z:
            return PauliType.I
        elif x and not z:
            return PauliType.X
        elif not x and z:
            return PauliType.Z
        else:
            return PauliType.Y
    
    def set_pauli(self, qubit: int, pauli: PauliType) -> None:
        """Set the Pauli operator for a specific qubit."""
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {self.num_qubits})")
        
        if pauli == PauliType.I:
            self.x_frame[qubit] = False
            self.z_frame[qubit] = False
        elif pauli == PauliType.X:
            self.x_frame[qubit] = True
            self.z_frame[qubit] = False
        elif pauli == PauliType.Z:
            self.x_frame[qubit] = False
            self.z_frame[qubit] = True
        elif pauli == PauliType.Y:
            self.x_frame[qubit] = True
            self.z_frame[qubit] = True
    
    def copy(self) -> "PauliFrame":
        """Create a copy of this frame."""
        return PauliFrame(
            num_qubits=self.num_qubits,
            x_frame=self.x_frame.copy(),
            z_frame=self.z_frame.copy(),
        )
    
    def reset(self) -> None:
        """Reset all corrections to identity."""
        self.x_frame = [False] * self.num_qubits
        self.z_frame = [False] * self.num_qubits
    
    def __str__(self) -> str:
        """String representation of the frame."""
        paulis = [self.get_pauli(i).value for i in range(self.num_qubits)]
        return f"PauliFrame({' '.join(paulis)})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_qubits": self.num_qubits,
            "x_frame": self.x_frame,
            "z_frame": self.z_frame,
            "paulis": [self.get_pauli(i).value for i in range(self.num_qubits)],
        }


class PauliTracker:
    """
    Tracks Pauli frame through quantum operations.
    
    Provides methods to:
    - Apply Pauli corrections (from measurements)
    - Propagate frame through Clifford gates
    - Query current frame state
    - Handle conditional operations based on measurement outcomes
    
    Attributes:
        num_qubits: Number of logical qubits being tracked
        frame: Current PauliFrame
        history: Optional history of frame states
    """
    
    def __init__(
        self,
        num_logical_qubits: int,
        track_history: bool = False,
    ):
        """
        Initialize Pauli tracker.
        
        Args:
            num_logical_qubits: Number of logical qubits to track
            track_history: Whether to record frame history
        """
        self.num_qubits = num_logical_qubits
        self.frame = PauliFrame(num_qubits=num_logical_qubits)
        self.track_history = track_history
        self.history: List[Tuple[str, PauliFrame]] = []
    
    def _record_history(self, operation: str) -> None:
        """Record current state to history."""
        if self.track_history:
            self.history.append((operation, self.frame.copy()))
    
    # =========================================================================
    # Pauli application methods
    # =========================================================================
    
    def apply_x(self, qubit: int) -> None:
        """Apply X correction to qubit (XOR with current X frame)."""
        self.frame.x_frame[qubit] = not self.frame.x_frame[qubit]
        self._record_history(f"X({qubit})")
    
    def apply_z(self, qubit: int) -> None:
        """Apply Z correction to qubit (XOR with current Z frame)."""
        self.frame.z_frame[qubit] = not self.frame.z_frame[qubit]
        self._record_history(f"Z({qubit})")
    
    def apply_y(self, qubit: int) -> None:
        """Apply Y correction to qubit (XOR both X and Z frames)."""
        self.apply_x(qubit)
        self.apply_z(qubit)
        self._record_history(f"Y({qubit})")
    
    def apply_pauli(self, qubit: int, pauli: PauliType) -> None:
        """Apply arbitrary Pauli correction."""
        if pauli == PauliType.X:
            self.apply_x(qubit)
        elif pauli == PauliType.Z:
            self.apply_z(qubit)
        elif pauli == PauliType.Y:
            self.apply_y(qubit)
        # I does nothing
    
    def apply_from_measurement(
        self,
        qubit: int,
        measurement_outcome: int,
        correction_type: str = "Z",
    ) -> None:
        """
        Apply Pauli correction based on measurement outcome.
        
        Args:
            qubit: Qubit to correct
            measurement_outcome: 0 or 1
            correction_type: "X", "Y", or "Z"
        """
        if measurement_outcome == 1:
            if correction_type == "X":
                self.apply_x(qubit)
            elif correction_type == "Z":
                self.apply_z(qubit)
            elif correction_type == "Y":
                self.apply_y(qubit)
    
    # =========================================================================
    # Frame propagation through Clifford gates
    # =========================================================================
    
    def propagate_h(self, qubit: int) -> None:
        """
        Propagate frame through Hadamard gate.
        
        H transforms Paulis as:
        - X → Z
        - Z → X
        - Y → -Y (tracked as Y)
        """
        # Swap X and Z frames for this qubit
        self.frame.x_frame[qubit], self.frame.z_frame[qubit] = \
            self.frame.z_frame[qubit], self.frame.x_frame[qubit]
        self._record_history(f"H({qubit})")
    
    def propagate_s(self, qubit: int) -> None:
        """
        Propagate frame through S gate.
        
        S transforms Paulis as:
        - X → Y = XZ
        - Z → Z
        - Y → -X
        
        For frame tracking: X gets Z added, Z stays same
        """
        if self.frame.x_frame[qubit]:
            # X → XZ (add Z)
            self.frame.z_frame[qubit] = not self.frame.z_frame[qubit]
        self._record_history(f"S({qubit})")
    
    def propagate_s_dag(self, qubit: int) -> None:
        """
        Propagate frame through S† gate.
        
        S† transforms Paulis as:
        - X → -Y = -XZ
        - Z → Z
        - Y → X
        """
        if self.frame.x_frame[qubit]:
            self.frame.z_frame[qubit] = not self.frame.z_frame[qubit]
        self._record_history(f"S_DAG({qubit})")
    
    def propagate_cnot(self, control: int, target: int) -> None:
        """
        Propagate frame through CNOT gate.
        
        CNOT transforms Paulis as:
        - X_c → X_c X_t
        - Z_c → Z_c
        - X_t → X_t
        - Z_t → Z_c Z_t
        
        For frame: X on control spreads to target,
                   Z on target spreads to control
        """
        # X on control → X on both
        if self.frame.x_frame[control]:
            self.frame.x_frame[target] = not self.frame.x_frame[target]
        
        # Z on target → Z on both
        if self.frame.z_frame[target]:
            self.frame.z_frame[control] = not self.frame.z_frame[control]
        
        self._record_history(f"CNOT({control},{target})")
    
    def propagate_cz(self, qubit1: int, qubit2: int) -> None:
        """
        Propagate frame through CZ gate.
        
        CZ transforms Paulis as:
        - X_1 → X_1 Z_2
        - X_2 → Z_1 X_2
        - Z_1, Z_2 → unchanged
        
        For frame: X on one adds Z to other
        """
        # X on qubit1 → add Z to qubit2
        if self.frame.x_frame[qubit1]:
            self.frame.z_frame[qubit2] = not self.frame.z_frame[qubit2]
        
        # X on qubit2 → add Z to qubit1
        if self.frame.x_frame[qubit2]:
            self.frame.z_frame[qubit1] = not self.frame.z_frame[qubit1]
        
        self._record_history(f"CZ({qubit1},{qubit2})")
    
    def propagate_swap(self, qubit1: int, qubit2: int) -> None:
        """
        Propagate frame through SWAP gate.
        
        Simply swaps the frame entries.
        """
        self.frame.x_frame[qubit1], self.frame.x_frame[qubit2] = \
            self.frame.x_frame[qubit2], self.frame.x_frame[qubit1]
        self.frame.z_frame[qubit1], self.frame.z_frame[qubit2] = \
            self.frame.z_frame[qubit2], self.frame.z_frame[qubit1]
        self._record_history(f"SWAP({qubit1},{qubit2})")
    
    # =========================================================================
    # Teleportation-specific methods
    # =========================================================================
    
    def process_teleportation_outcome(
        self,
        source_qubit: int,
        target_qubit: int,
        x_measurement: int,
        z_measurement: int,
    ) -> None:
        """
        Process teleportation measurement outcomes.
        
        Standard teleportation corrections:
        - m_x=1 → Z on target
        - m_z=1 → X on target
        
        Also transfers existing frame from source to target.
        
        Args:
            source_qubit: Original data qubit (measured)
            target_qubit: Teleported output qubit
            x_measurement: X-basis measurement result
            z_measurement: Z-basis measurement result
        """
        # Transfer frame from source to target
        self.frame.x_frame[target_qubit] = self.frame.x_frame[source_qubit]
        self.frame.z_frame[target_qubit] = self.frame.z_frame[source_qubit]
        
        # Clear source frame (qubit is measured out)
        self.frame.x_frame[source_qubit] = False
        self.frame.z_frame[source_qubit] = False
        
        # Apply corrections based on measurement outcomes
        if x_measurement == 1:
            self.frame.z_frame[target_qubit] = not self.frame.z_frame[target_qubit]
        if z_measurement == 1:
            self.frame.x_frame[target_qubit] = not self.frame.x_frame[target_qubit]
        
        self._record_history(
            f"TELEPORT({source_qubit}→{target_qubit}, mx={x_measurement}, mz={z_measurement})"
        )
    
    # =========================================================================
    # Surgery-specific methods
    # =========================================================================
    
    def process_surgery_outcome(
        self,
        qubit1: int,
        qubit2: int,
        merge_measurement: int,
        operator_type: str = "Z",
    ) -> None:
        """
        Process surgery merge measurement outcome.
        
        For ZZ merge: odd outcome indicates Z correction needed
        For XX merge: odd outcome indicates X correction needed
        
        Args:
            qubit1: First qubit in merge
            qubit2: Second qubit in merge
            merge_measurement: Parity measurement result
            operator_type: "X" or "Z" for merge type
        """
        if merge_measurement == 1:
            if operator_type == "Z":
                # ZZ merge with -1 outcome: need Z on one qubit
                self.frame.z_frame[qubit1] = not self.frame.z_frame[qubit1]
            else:
                # XX merge with -1 outcome: need X on one qubit
                self.frame.x_frame[qubit1] = not self.frame.x_frame[qubit1]
        
        self._record_history(
            f"SURGERY({qubit1},{qubit2}, m={merge_measurement}, type={operator_type})"
        )
    
    # =========================================================================
    # Query methods
    # =========================================================================
    
    def get_frame(self) -> PauliFrame:
        """Get current Pauli frame."""
        return self.frame.copy()
    
    def get_correction(self, qubit: int) -> Tuple[bool, bool]:
        """
        Get X and Z correction bits for a qubit.
        
        Returns:
            (x_correction, z_correction) booleans
        """
        return (self.frame.x_frame[qubit], self.frame.z_frame[qubit])
    
    def needs_correction(self, qubit: int) -> bool:
        """Check if qubit has any pending correction."""
        return self.frame.x_frame[qubit] or self.frame.z_frame[qubit]
    
    def reset(self) -> None:
        """Reset all corrections to identity."""
        self.frame.reset()
        self.history.clear()
    
    def __str__(self) -> str:
        return f"PauliTracker({self.frame})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_qubits": self.num_qubits,
            "frame": self.frame.to_dict(),
            "history_length": len(self.history),
        }


def pauli_product(p1: PauliType, p2: PauliType) -> Tuple[PauliType, int]:
    """
    Compute product of two Pauli operators.
    
    Returns (result_pauli, phase) where phase is 0, 1, 2, 3
    representing i^phase.
    
    Args:
        p1: First Pauli
        p2: Second Pauli
        
    Returns:
        (product_pauli, phase_exponent)
    """
    # Multiplication table for Paulis (ignoring phase)
    # I*X=X, I*Y=Y, I*Z=Z
    # X*X=I, X*Y=Z, X*Z=Y
    # Y*X=Z, Y*Y=I, Y*Z=X
    # Z*X=Y, Z*Y=X, Z*Z=I
    
    mult_table = {
        (PauliType.I, PauliType.I): (PauliType.I, 0),
        (PauliType.I, PauliType.X): (PauliType.X, 0),
        (PauliType.I, PauliType.Y): (PauliType.Y, 0),
        (PauliType.I, PauliType.Z): (PauliType.Z, 0),
        (PauliType.X, PauliType.I): (PauliType.X, 0),
        (PauliType.X, PauliType.X): (PauliType.I, 0),
        (PauliType.X, PauliType.Y): (PauliType.Z, 1),  # XY = iZ
        (PauliType.X, PauliType.Z): (PauliType.Y, 3),  # XZ = -iY
        (PauliType.Y, PauliType.I): (PauliType.Y, 0),
        (PauliType.Y, PauliType.X): (PauliType.Z, 3),  # YX = -iZ
        (PauliType.Y, PauliType.Y): (PauliType.I, 0),
        (PauliType.Y, PauliType.Z): (PauliType.X, 1),  # YZ = iX
        (PauliType.Z, PauliType.I): (PauliType.Z, 0),
        (PauliType.Z, PauliType.X): (PauliType.Y, 1),  # ZX = iY
        (PauliType.Z, PauliType.Y): (PauliType.X, 3),  # ZY = -iX
        (PauliType.Z, PauliType.Z): (PauliType.I, 0),
    }
    
    return mult_table[(p1, p2)]
