# src/qectostim/experiments/hardware_simulation/superconducting/architecture.py
"""
Superconducting architecture definitions.

Defines hardware architectures for superconducting quantum computers.

NOT IMPLEMENTED: These are stubs defining the interfaces.
"""
from __future__ import annotations

from abc import abstractmethod
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
from qectostim.experiments.hardware_simulation.superconducting.gates import (
    SUPERCONDUCTING_NATIVE_GATES,
)


class SuperconductingArchitecture(HardwareArchitecture):
    """Abstract base class for superconducting architectures.
    
    Common properties:
    - Fixed connectivity (can't move qubits)
    - Fast gates but limited coherence
    - Various native 2Q gates (CX, CZ, iSWAP)
    
    Subclasses define specific connectivity:
    - HeavyHexArchitecture: IBM-style heavy-hex lattice
    - SquareLatticeArchitecture: Google-style square grid
    - TunableCouplerArchitecture: Adjustable coupling
    """
    
    def __init__(
        self,
        name: str,
        num_qubits: int,
        native_2q_gate: str = "CX",
        single_qubit_time: float = 0.05,  # μs (50ns)
        two_qubit_time: float = 0.3,  # μs (300ns)
        measurement_time: float = 1.0,  # μs
        t1_time: float = 100.0,  # μs
        t2_time: float = 100.0,  # μs
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, num_qubits, metadata)
        self.native_2q_gate = native_2q_gate
        self.single_qubit_time = single_qubit_time
        self.two_qubit_time = two_qubit_time
        self.measurement_time = measurement_time
        self.t1_time = t1_time
        self.t2_time = t2_time
    
    def native_gate_set(self) -> NativeGateSet:
        """Superconducting native gates."""
        gate_set = NativeGateSet("superconducting")
        
        # Single-qubit gates (all platforms)
        for name in ["H", "X", "Y", "Z", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG"]:
            gate_set.add_gate(GateSpec(name, GateType.SINGLE_QUBIT, 1, is_clifford=True))
        
        # Virtual Z (typically free)
        gate_set.add_gate(GateSpec(
            "RZ", GateType.SINGLE_QUBIT, 1,
            parameters=("theta",), is_clifford=False, is_native=True
        ))
        
        # Native 2Q gate depends on platform
        if self.native_2q_gate == "CX":
            gate_set.add_gate(GateSpec("CX", GateType.TWO_QUBIT, 2, is_clifford=True, is_native=True))
        elif self.native_2q_gate == "CZ":
            gate_set.add_gate(GateSpec("CZ", GateType.TWO_QUBIT, 2, is_clifford=True, is_native=True))
        elif self.native_2q_gate == "ISWAP":
            gate_set.add_gate(GateSpec("ISWAP", GateType.TWO_QUBIT, 2, is_clifford=True, is_native=True))
        
        return gate_set
    
    def physical_constraints(self) -> PhysicalConstraints:
        """Physical constraints for superconducting qubits."""
        return PhysicalConstraints(
            max_qubits=self.num_qubits,
            t1_time=self.t1_time,
            t2_time=self.t2_time,
            gate_times={
                "1Q": self.single_qubit_time,
                "2Q": self.two_qubit_time,
                self.native_2q_gate: self.two_qubit_time,
            },
            readout_time=self.measurement_time,
            reset_time=self.measurement_time,  # Active reset ~ measurement time
        )
    
    def zone_types(self) -> List[ZoneType]:
        """Single zone type for fixed architecture."""
        return [ZoneType.COMPUTATION]
    
    def is_reconfigurable(self) -> bool:
        """Superconducting qubits cannot be moved."""
        return False


class FixedCouplingArchitecture(SuperconductingArchitecture):
    """Fixed coupling superconducting architecture.
    
    Qubits have fixed neighbors defined by chip layout.
    Two-qubit gates only between connected pairs.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    edges : List[Tuple[int, int]]
        List of connected qubit pairs.
    
    NOT IMPLEMENTED: This is a stub for future implementation.
    """
    
    def __init__(
        self,
        num_qubits: int,
        edges: List[Tuple[int, int]],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            name=name or f"FixedCoupling_{num_qubits}Q",
            num_qubits=num_qubits,
            metadata=metadata,
            **kwargs,
        )
        self.edges = edges
        self._edge_set: Set[Tuple[int, int]] = set()
        for q1, q2 in edges:
            self._edge_set.add((min(q1, q2), max(q1, q2)))
    
    def connectivity_graph(self) -> ConnectivityGraph:
        """Build connectivity graph from edges."""
        graph = ConnectivityGraph()
        
        # Add qubit nodes
        for q in range(self.num_qubits):
            zone = Zone(
                id=f"q{q}",
                zone_type=ZoneType.COMPUTATION,
                capacity=1,
                position=(q, 0),
            )
            graph.add_zone(zone)
        
        # Add edges
        for q1, q2 in self.edges:
            graph.add_connection(
                f"q{q1}", f"q{q2}",
                weight=self.two_qubit_time,
            )
        
        return graph
    
    def can_interact(self, qubit1: int, qubit2: int) -> bool:
        """Check if qubits are directly connected."""
        key = (min(qubit1, qubit2), max(qubit1, qubit2))
        return key in self._edge_set
    
    def interaction_distance(self, qubit1: int, qubit2: int) -> int:
        """Get SWAP distance between qubits."""
        if self.can_interact(qubit1, qubit2):
            return 0
        
        # BFS for shortest path
        graph = self.connectivity_graph()
        try:
            path = graph.shortest_path(f"q{qubit1}", f"q{qubit2}")
            return len(path) - 2  # Number of SWAPs needed
        except Exception:
            return -1  # Not connected


class HeavyHexArchitecture(FixedCouplingArchitecture):
    """IBM Heavy-Hex lattice architecture.
    
    Heavy-hex topology used in IBM Quantum systems.
    
    NOT IMPLEMENTED: This is a stub for future implementation.
    """
    
    def __init__(
        self,
        rows: int = 3,
        cols: int = 3,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Build heavy-hex edges
        # TODO: Implement proper heavy-hex topology
        num_qubits = rows * cols  # Simplified
        edges = []  # TODO: Generate heavy-hex edges
        
        super().__init__(
            num_qubits=num_qubits,
            edges=edges,
            name=name or f"HeavyHex_{rows}x{cols}",
            native_2q_gate="CX",
            metadata=metadata,
        )
        self.rows = rows
        self.cols = cols
    
    def connectivity_graph(self) -> ConnectivityGraph:
        """Build heavy-hex connectivity."""
        raise NotImplementedError(
            "HeavyHexArchitecture.connectivity_graph() not yet implemented."
        )


class SquareLatticeArchitecture(FixedCouplingArchitecture):
    """Square lattice architecture (Google Sycamore style).
    
    Regular square grid with nearest-neighbor connectivity.
    
    NOT IMPLEMENTED: This is a stub for future implementation.
    """
    
    def __init__(
        self,
        rows: int = 3,
        cols: int = 3,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        num_qubits = rows * cols
        edges = []
        
        # Build square lattice edges
        for r in range(rows):
            for c in range(cols):
                q = r * cols + c
                # Right neighbor
                if c < cols - 1:
                    edges.append((q, q + 1))
                # Bottom neighbor
                if r < rows - 1:
                    edges.append((q, q + cols))
        
        super().__init__(
            num_qubits=num_qubits,
            edges=edges,
            name=name or f"SquareLattice_{rows}x{cols}",
            native_2q_gate="CZ",  # Google style
            metadata=metadata,
        )
        self.rows = rows
        self.cols = cols


class TunableCouplerArchitecture(SuperconductingArchitecture):
    """Tunable coupler architecture.
    
    Coupling strength can be adjusted, allowing different effective
    connectivities and reduced crosstalk.
    
    NOT IMPLEMENTED: This is a stub for future implementation.
    """
    
    def __init__(
        self,
        num_qubits: int,
        base_connectivity: List[Tuple[int, int]],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name or f"TunableCoupler_{num_qubits}Q",
            num_qubits=num_qubits,
            metadata=metadata,
        )
        self.base_connectivity = base_connectivity
    
    def connectivity_graph(self) -> ConnectivityGraph:
        """Build connectivity graph."""
        raise NotImplementedError(
            "TunableCouplerArchitecture.connectivity_graph() not yet implemented."
        )
    
    def can_interact(self, qubit1: int, qubit2: int) -> bool:
        """Check if qubits can be coupled."""
        raise NotImplementedError(
            "TunableCouplerArchitecture.can_interact() not yet implemented."
        )
    
    def interaction_distance(self, qubit1: int, qubit2: int) -> int:
        """Get routing distance."""
        raise NotImplementedError(
            "TunableCouplerArchitecture.interaction_distance() not yet implemented."
        )
