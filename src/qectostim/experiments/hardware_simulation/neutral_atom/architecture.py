# src/qectostim/experiments/hardware_simulation/neutral_atom/architecture.py
"""
Neutral atom architecture definitions.

Defines hardware architectures for neutral atom quantum computers.

NOT IMPLEMENTED: These are stubs defining the interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Set,
    Any,
)

from qectostim.experiments.hardware_simulation.core.architecture import (
    HardwareArchitecture,
    ConnectivityGraph,
    Zone,
    ZoneType,
    PhysicalConstraints,
)
from qectostim.experiments.hardware_simulation.core.gates import (
    NativeGateSet,
    GateSpec,
    GateType,
)
from qectostim.experiments.hardware_simulation.neutral_atom.gates import (
    NEUTRAL_ATOM_NATIVE_GATES,
)


class NeutralAtomArchitecture(HardwareArchitecture):
    """Abstract base class for neutral atom architectures.
    
    Common properties:
    - Rydberg blockade entangling gates
    - Optical tweezer atom positioning
    - Global single-qubit rotations
    - Long coherence (seconds scale)
    
    Subclasses define specific geometries:
    - TweezerArrayArchitecture: Dynamically reconfigurable array
    - RydbergLatticeArchitecture: Fixed optical lattice
    """
    
    def __init__(
        self,
        name: str,
        num_qubits: int,
        blockade_radius: float = 10.0,  # μm
        rydberg_gate_time: float = 0.5,  # μs
        single_qubit_time: float = 0.1,  # μs (global rotation)
        measurement_time: float = 10.0,  # μs
        atom_move_time: float = 100.0,  # μs
        t2_time: float = 1e6,  # μs (1 second)
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, num_qubits, metadata)
        self.blockade_radius = blockade_radius
        self.rydberg_gate_time = rydberg_gate_time
        self.single_qubit_time = single_qubit_time
        self.measurement_time = measurement_time
        self.atom_move_time = atom_move_time
        self.t2_time = t2_time
    
    def native_gate_set(self) -> NativeGateSet:
        """Neutral atom native gates."""
        gate_set = NativeGateSet("neutral_atom")
        
        # Add neutral atom native gates
        for name, spec in NEUTRAL_ATOM_NATIVE_GATES.items():
            gate_set.add_gate(spec)
        
        # Global rotations (act on all atoms)
        gate_set.add_gate(GateSpec(
            "GLOBAL_X", GateType.MULTI_QUBIT, -1,
            parameters=("theta",), is_clifford=False, is_native=True
        ))
        gate_set.add_gate(GateSpec(
            "GLOBAL_Y", GateType.MULTI_QUBIT, -1,
            parameters=("theta",), is_clifford=False, is_native=True
        ))
        
        # Standard single-qubit (via local addressing)
        for name in ["H", "X", "Y", "Z", "S"]:
            gate_set.add_gate(GateSpec(name, GateType.SINGLE_QUBIT, 1, is_clifford=True))
        
        # CZ via Rydberg blockade
        gate_set.add_gate(GateSpec("CZ", GateType.TWO_QUBIT, 2, is_clifford=True, is_native=True))
        
        return gate_set
    
    def physical_constraints(self) -> PhysicalConstraints:
        """Physical constraints for neutral atoms."""
        return PhysicalConstraints(
            max_qubits=self.num_qubits,
            t2_time=self.t2_time,
            gate_times={
                "CZ": self.rydberg_gate_time,
                "RYDBERG_CZ": self.rydberg_gate_time,
                "GLOBAL_R": self.single_qubit_time,
                "1Q": self.single_qubit_time,
            },
            readout_time=self.measurement_time,
            reset_time=self.measurement_time,  # Re-preparation
            transport_time=self.atom_move_time,
        )
    
    def zone_types(self) -> List[ZoneType]:
        """Zone types in neutral atom architecture."""
        return [
            ZoneType.COMPUTATION,
            ZoneType.STORAGE,
            ZoneType.INTERACTION,  # Rydberg interaction zone
        ]


class TweezerArrayArchitecture(NeutralAtomArchitecture):
    """Optical tweezer array architecture.
    
    Atoms held in optical tweezers that can be rearranged.
    Enables dynamic connectivity via atom movement.
    
    Parameters
    ----------
    rows : int
        Number of rows in the array.
    cols : int
        Number of columns in the array.
    site_spacing : float
        Distance between sites (μm).
    
    NOT IMPLEMENTED: This is a stub for future implementation.
    """
    
    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        site_spacing: float = 5.0,  # μm
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        num_qubits = rows * cols
        super().__init__(
            name=name or f"TweezerArray_{rows}x{cols}",
            num_qubits=num_qubits,
            metadata=metadata,
        )
        self.rows = rows
        self.cols = cols
        self.site_spacing = site_spacing
    
    def connectivity_graph(self) -> ConnectivityGraph:
        """Build tweezer array connectivity.
        
        Initially: connectivity based on blockade radius.
        Can be reconfigured by moving atoms.
        
        NOT IMPLEMENTED.
        """
        raise NotImplementedError(
            "TweezerArrayArchitecture.connectivity_graph() not yet implemented."
        )
    
    def can_interact(self, qubit1: int, qubit2: int) -> bool:
        """Check if atoms are within blockade radius.
        
        NOT IMPLEMENTED.
        """
        raise NotImplementedError(
            "TweezerArrayArchitecture.can_interact() not yet implemented."
        )
    
    def interaction_distance(self, qubit1: int, qubit2: int) -> int:
        """Get number of moves needed to bring atoms together.
        
        NOT IMPLEMENTED.
        """
        raise NotImplementedError(
            "TweezerArrayArchitecture.interaction_distance() not yet implemented."
        )
    
    def is_reconfigurable(self) -> bool:
        """Tweezer arrays support atom movement."""
        return True


class RydbergLatticeArchitecture(NeutralAtomArchitecture):
    """Fixed Rydberg lattice architecture.
    
    Atoms in fixed optical lattice positions.
    Connectivity determined by lattice geometry and blockade radius.
    
    NOT IMPLEMENTED: This is a stub for future implementation.
    """
    
    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        lattice_spacing: float = 5.0,  # μm
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        num_qubits = rows * cols
        super().__init__(
            name=name or f"RydbergLattice_{rows}x{cols}",
            num_qubits=num_qubits,
            metadata=metadata,
        )
        self.rows = rows
        self.cols = cols
        self.lattice_spacing = lattice_spacing
    
    def connectivity_graph(self) -> ConnectivityGraph:
        """Build lattice connectivity based on blockade radius."""
        raise NotImplementedError(
            "RydbergLatticeArchitecture.connectivity_graph() not yet implemented."
        )
    
    def can_interact(self, qubit1: int, qubit2: int) -> bool:
        """Check if atoms are within blockade radius."""
        raise NotImplementedError(
            "RydbergLatticeArchitecture.can_interact() not yet implemented."
        )
    
    def interaction_distance(self, qubit1: int, qubit2: int) -> int:
        """Get SWAP-like distance (fixed lattice = no reconfiguration)."""
        raise NotImplementedError(
            "RydbergLatticeArchitecture.interaction_distance() not yet implemented."
        )
    
    def is_reconfigurable(self) -> bool:
        """Fixed lattice cannot be reconfigured."""
        return False
