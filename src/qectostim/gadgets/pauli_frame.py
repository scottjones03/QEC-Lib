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
    Tuple,
)
from enum import Enum


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
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize frame state for gadget chaining.
        
        Returns a dictionary that can be used with deserialize() to
        restore the exact frame state. This is essential for chaining
        gadgets where the Pauli frame must persist between gadgets.
        
        Returns
        -------
        Dict[str, Any]
            Serialized frame state with keys:
            - num_qubits: int
            - x_frame: List[bool]
            - z_frame: List[bool]
        """
        return {
            "num_qubits": self.num_qubits,
            "x_frame": list(self.x_frame),
            "z_frame": list(self.z_frame),
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "PauliFrame":
        """
        Restore frame state from serialized data.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Serialized frame state from serialize().
            
        Returns
        -------
        PauliFrame
            Restored frame with exact state.
        """
        return cls(
            num_qubits=data["num_qubits"],
            x_frame=list(data["x_frame"]),
            z_frame=list(data["z_frame"]),
        )


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
    # Query methods
    # =========================================================================
    
    def get_frame(self) -> PauliFrame:
        """Get current Pauli frame."""
        return self.frame.copy()
    
    def __str__(self) -> str:
        return f"PauliTracker({self.frame})"
    
    # =========================================================================
    # Observable computation methods
    # =========================================================================
    
    def compute_observable_parity(
        self,
        logical_support: List[int],
        basis: str,
    ) -> int:
        """
        Compute the parity of Pauli frame corrections on a logical operator.
        
        For a logical Z (X) observable on qubits in logical_support,
        this computes whether the Pauli frame introduces a sign flip.
        
        - Z observable: X frame corrections flip the sign
        - X observable: Z frame corrections flip the sign
        
        Parameters
        ----------
        logical_support : List[int]
            Qubit indices in the logical operator support.
        basis : str
            "X" or "Z" for the observable basis.
            
        Returns
        -------
        int
            0 if no sign flip, 1 if sign flipped by frame.
        """
        parity = 0
        for qubit in logical_support:
            if qubit >= self.num_qubits:
                continue
            if basis.upper() == "Z":
                # Z observable anticommutes with X corrections
                if self.frame.x_frame[qubit]:
                    parity ^= 1
            else:  # X observable
                # X observable anticommutes with Z corrections
                if self.frame.z_frame[qubit]:
                    parity ^= 1
        return parity
    
    def get_frame_correction_indices(
        self,
        logical_support: List[int],
        basis: str,
    ) -> List[int]:
        """
        Get indices of qubits that need frame correction for an observable.
        
        Parameters
        ----------
        logical_support : List[int]
            Qubit indices in the logical operator support.
        basis : str
            "X" or "Z" for the observable basis.
            
        Returns
        -------
        List[int]
            Indices of qubits with frame corrections affecting this observable.
        """
        correction_indices = []
        for qubit in logical_support:
            if qubit >= self.num_qubits:
                continue
            if basis.upper() == "Z":
                if self.frame.x_frame[qubit]:
                    correction_indices.append(qubit)
            else:  # X
                if self.frame.z_frame[qubit]:
                    correction_indices.append(qubit)
        return correction_indices


class MultiBlockPauliTracker:
    """
    Tracks Pauli frames across multiple named code blocks.
    
    For multi-block gadgets (teleportation, CNOT, surgery), we need to
    track frames per block and handle inter-block operations.
    
    Attributes
    ----------
    block_trackers : Dict[str, PauliTracker]
        Per-block Pauli trackers.
    """
    
    def __init__(self, block_sizes: Dict[str, int]):
        """
        Initialize multi-block tracker.
        
        Parameters
        ----------
        block_sizes : Dict[str, int]
            Mapping from block name to number of logical qubits.
        """
        self.block_trackers: Dict[str, PauliTracker] = {}
        for block_name, size in block_sizes.items():
            self.block_trackers[block_name] = PauliTracker(size)
    
    def get_tracker(self, block_name: str) -> PauliTracker:
        """Get tracker for a specific block."""
        return self.block_trackers.get(block_name)
    
    def propagate_inter_block_cnot(
        self,
        control_block: str,
        target_block: str,
        control_qubit: int = 0,
        target_qubit: int = 0,
    ) -> None:
        """
        Propagate frame through inter-block CNOT.
        
        CNOT(ctrl→tgt) transforms:
        - X_ctrl → X_ctrl ⊗ X_tgt (X on control spreads to target)
        - Z_tgt → Z_ctrl ⊗ Z_tgt (Z on target spreads to control)
        """
        ctrl_tracker = self.block_trackers.get(control_block)
        tgt_tracker = self.block_trackers.get(target_block)
        
        if ctrl_tracker is None or tgt_tracker is None:
            return
        
        # X on control spreads to target
        if ctrl_tracker.frame.x_frame[control_qubit]:
            tgt_tracker.apply_x(target_qubit)
        
        # Z on target spreads to control
        if tgt_tracker.frame.z_frame[target_qubit]:
            ctrl_tracker.apply_z(control_qubit)
    
    def propagate_inter_block_cz(
        self,
        block1: str,
        block2: str,
        qubit1: int = 0,
        qubit2: int = 0,
    ) -> None:
        """
        Propagate frame through inter-block CZ.
        
        CZ transforms:
        - X_1 → X_1 ⊗ Z_2 (X on qubit1 adds Z to qubit2)
        - X_2 → Z_1 ⊗ X_2 (X on qubit2 adds Z to qubit1)
        """
        tracker1 = self.block_trackers.get(block1)
        tracker2 = self.block_trackers.get(block2)
        
        if tracker1 is None or tracker2 is None:
            return
        
        # X on qubit1 adds Z to qubit2
        if tracker1.frame.x_frame[qubit1]:
            tracker2.apply_z(qubit2)
        
        # X on qubit2 adds Z to qubit1
        if tracker2.frame.x_frame[qubit2]:
            tracker1.apply_z(qubit1)
    
    def process_teleportation(
        self,
        source_block: str,
        target_block: str,
        x_measurement: int,
        z_measurement: int,
        source_qubit: int = 0,
        target_qubit: int = 0,
    ) -> None:
        """
        Process teleportation measurement outcomes across blocks.
        
        Transfers frame from source to target and applies corrections.
        """
        src_tracker = self.block_trackers.get(source_block)
        tgt_tracker = self.block_trackers.get(target_block)
        
        if src_tracker is None or tgt_tracker is None:
            return
        
        # Transfer existing frame
        tgt_tracker.frame.x_frame[target_qubit] = src_tracker.frame.x_frame[source_qubit]
        tgt_tracker.frame.z_frame[target_qubit] = src_tracker.frame.z_frame[source_qubit]
        
        # Clear source
        src_tracker.frame.x_frame[source_qubit] = False
        src_tracker.frame.z_frame[source_qubit] = False
        
        # Apply measurement-based corrections
        if x_measurement == 1:
            tgt_tracker.frame.z_frame[target_qubit] = not tgt_tracker.frame.z_frame[target_qubit]
        if z_measurement == 1:
            tgt_tracker.frame.x_frame[target_qubit] = not tgt_tracker.frame.x_frame[target_qubit]
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize multi-block tracker state for gadget chaining.
        
        Returns
        -------
        Dict[str, Any]
            Serialized state with per-block frame data.
        """
        return {
            "blocks": {
                name: tracker.frame.serialize()
                for name, tracker in self.block_trackers.items()
            }
        }
    
    def compute_block_observable_parity(
        self,
        block_name: str,
        logical_support: List[int],
        basis: str,
    ) -> int:
        """
        Compute the Pauli-frame parity on a logical operator for one block.

        This is the multi-block entry-point for PauliTracker.compute_observable_parity().
        It returns 0 or 1 indicating whether the current frame on the given block
        would flip the sign of the specified logical observable.

        Parameters
        ----------
        block_name : str
            Name of the code block.
        logical_support : List[int]
            Local qubit indices in the logical operator support.
        basis : str
            "X" or "Z" for the observable basis.

        Returns
        -------
        int
            0 if no sign flip, 1 if the frame flips the observable.
        """
        tracker = self.block_trackers.get(block_name)
        if tracker is None:
            return 0
        return tracker.compute_observable_parity(logical_support, basis)

    def get_block_frame_correction_indices(
        self,
        block_name: str,
        logical_support: List[int],
        basis: str,
    ) -> List[int]:
        """
        Get qubit indices within a block that need frame correction.

        Multi-block entry-point for PauliTracker.get_frame_correction_indices().

        Parameters
        ----------
        block_name : str
            Name of the code block.
        logical_support : List[int]
            Local qubit indices in the logical operator support.
        basis : str
            "X" or "Z" for the observable basis.

        Returns
        -------
        List[int]
            Local qubit indices with frame corrections affecting this observable.
        """
        tracker = self.block_trackers.get(block_name)
        if tracker is None:
            return []
        return tracker.get_frame_correction_indices(logical_support, basis)
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "MultiBlockPauliTracker":
        """
        Restore multi-block tracker state from serialized data.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Serialized state from serialize().
            
        Returns
        -------
        MultiBlockPauliTracker
            Restored tracker with exact state.
        """
        block_sizes = {
            name: frame_data["num_qubits"]
            for name, frame_data in data["blocks"].items()
        }
        tracker = cls(block_sizes)
        for name, frame_data in data["blocks"].items():
            restored_frame = PauliFrame.deserialize(frame_data)
            tracker.block_trackers[name].frame = restored_frame
        return tracker
    

