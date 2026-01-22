# src/qectostim/experiments/hardware_simulation/neutral_atom/gates.py
"""
Neutral atom native gates.

Defines the native gate set for neutral atom quantum computers,
including Rydberg blockade gates and global rotations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from qectostim.experiments.hardware_simulation.core.gates import (
    GateSpec,
    GateType,
    NativeGateSet,
    GateDecomposer,
    GateDecomposition,
    ParameterizedGate,
    STANDARD_GATES,
)


# =============================================================================
# Neutral Atom Native Gate Definitions
# =============================================================================

# Rydberg blockade CZ gate
RYDBERG_CZ_GATE = GateSpec(
    name="RYDBERG_CZ",
    gate_type=GateType.TWO_QUBIT,
    num_qubits=2,
    is_clifford=True,
    is_native=True,
    metadata={
        "platform": "neutral_atom",
        "description": "CZ gate via Rydberg blockade",
        "typical_duration_us": 0.5,
        "typical_fidelity": 0.995,
        "requires_blockade": True,
    },
)

# Multi-qubit Rydberg gate (CCZ, etc.)
RYDBERG_CCZ_GATE = GateSpec(
    name="RYDBERG_CCZ",
    gate_type=GateType.MULTI_QUBIT,
    num_qubits=3,
    is_clifford=True,
    is_native=True,
    metadata={
        "platform": "neutral_atom",
        "description": "CCZ gate via multi-body Rydberg interaction",
        "typical_duration_us": 1.0,
    },
)

# Global rotation (all atoms in addressing zone)
GLOBAL_ROTATION_NA = GateSpec(
    name="GLOBAL_R",
    gate_type=GateType.MULTI_QUBIT,
    num_qubits=-1,  # Variable: all atoms
    parameters=("theta", "phi"),
    is_clifford=False,
    is_native=True,
    metadata={
        "platform": "neutral_atom",
        "description": "Global rotation via uniform addressing",
        "typical_duration_us": 0.1,
        "typical_fidelity": 0.9999,
    },
)

# Global X rotation
GLOBAL_X = GateSpec(
    name="GLOBAL_X",
    gate_type=GateType.MULTI_QUBIT,
    num_qubits=-1,
    parameters=("theta",),
    is_clifford=False,
    is_native=True,
    metadata={"platform": "neutral_atom"},
)

# Global Y rotation
GLOBAL_Y = GateSpec(
    name="GLOBAL_Y",
    gate_type=GateType.MULTI_QUBIT,
    num_qubits=-1,
    parameters=("theta",),
    is_clifford=False,
    is_native=True,
    metadata={"platform": "neutral_atom"},
)

# Local single-qubit rotation (via focused beam)
LOCAL_ROTATION = GateSpec(
    name="LOCAL_R",
    gate_type=GateType.SINGLE_QUBIT,
    num_qubits=1,
    parameters=("theta", "phi"),
    is_clifford=False,
    is_native=True,
    metadata={
        "platform": "neutral_atom",
        "description": "Local rotation via focused addressing beam",
        "typical_duration_us": 0.5,
        "has_crosstalk": True,
    },
)

# Atom movement operation (for tweezer arrays)
ATOM_MOVE = GateSpec(
    name="ATOM_MOVE",
    gate_type=GateType.BARRIER,  # Uses BARRIER as placeholder
    num_qubits=1,
    is_native=True,
    metadata={
        "platform": "neutral_atom",
        "description": "Move atom to new position",
        "typical_duration_us": 100.0,
    },
)

# Fluorescence measurement
NA_MEASUREMENT = GateSpec(
    name="M_NA",
    gate_type=GateType.MEASUREMENT,
    num_qubits=1,
    is_native=True,
    metadata={
        "platform": "neutral_atom",
        "description": "Fluorescence measurement (destructive)",
        "typical_duration_us": 10.0,
        "typical_fidelity": 0.99,
        "is_destructive": True,  # Atom loss possible
    },
)

# Dictionary of all neutral atom native gates
NEUTRAL_ATOM_NATIVE_GATES: Dict[str, GateSpec] = {
    "RYDBERG_CZ": RYDBERG_CZ_GATE,
    "RYDBERG_CCZ": RYDBERG_CCZ_GATE,
    "GLOBAL_R": GLOBAL_ROTATION_NA,
    "GLOBAL_X": GLOBAL_X,
    "GLOBAL_Y": GLOBAL_Y,
    "LOCAL_R": LOCAL_ROTATION,
    "ATOM_MOVE": ATOM_MOVE,
    "M_NA": NA_MEASUREMENT,
}


# =============================================================================
# Neutral Atom Gate Set
# =============================================================================

class NeutralAtomGateSet(NativeGateSet):
    """Native gate set for neutral atom hardware.
    
    Includes:
    - Rydberg blockade CZ gate
    - Global rotations (all atoms)
    - Local rotations (focused beam)
    - Fluorescence measurement
    
    Example
    -------
    >>> gate_set = NeutralAtomGateSet()
    >>> gate_set.has_gate("RYDBERG_CZ")
    True
    """
    
    def __init__(self):
        super().__init__(platform="neutral_atom")
        
        # Add neutral atom native gates
        for name, spec in NEUTRAL_ATOM_NATIVE_GATES.items():
            self.add_gate(spec)
        
        # Add standard single-qubit Clifford gates
        for name in ["H", "S", "S_DAG", "X", "Y", "Z"]:
            if name in STANDARD_GATES:
                self.add_gate(STANDARD_GATES[name])
    
    def entangling_gate(self) -> GateSpec:
        """Get the native entangling gate (Rydberg CZ)."""
        return self.get_gate("RYDBERG_CZ")
    
    def has_global_operations(self) -> bool:
        """Neutral atoms support global rotations."""
        return True
    
    def supports_multi_qubit_gates(self) -> bool:
        """Neutral atoms can do multi-qubit Rydberg gates."""
        return True


# =============================================================================
# Gate Decomposer for Neutral Atoms
# =============================================================================

class NeutralAtomGateDecomposer(GateDecomposer):
    """Decomposes gates to neutral atom native operations.
    
    Standard decompositions:
    - CNOT → H + CZ + H
    - Uses global rotations where possible
    
    NOT IMPLEMENTED: This is a stub for the decomposition logic.
    """
    
    def __init__(self):
        super().__init__(NeutralAtomGateSet())
    
    def decompose(
        self,
        gate: GateSpec | ParameterizedGate,
        qubits: Tuple[int, ...],
    ) -> GateDecomposition:
        """Decompose a gate to neutral atom natives.
        
        NOT IMPLEMENTED.
        """
        raise NotImplementedError(
            "NeutralAtomGateDecomposer.decompose() not yet implemented."
        )


# =============================================================================
# Gate Timing Models
# =============================================================================

@dataclass
class NeutralAtomGateTiming:
    """Timing parameters for neutral atom gates.
    
    Attributes
    ----------
    rydberg_gate_time : float
        Rydberg CZ gate duration in seconds.
    global_rotation_time : float
        Global rotation duration in seconds.
    local_rotation_time : float
        Local addressing rotation in seconds.
    measurement_time : float
        Fluorescence measurement duration in seconds.
    atom_move_time : float
        Atom rearrangement time in seconds.
    """
    rydberg_gate_time: float = 0.5e-6  # 500 ns
    global_rotation_time: float = 0.1e-6  # 100 ns
    local_rotation_time: float = 0.5e-6  # 500 ns
    measurement_time: float = 10e-6  # 10 μs
    atom_move_time: float = 100e-6  # 100 μs


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Gate specifications
    "RYDBERG_CZ_GATE",
    "RYDBERG_CCZ_GATE",
    "GLOBAL_ROTATION_NA",
    "GLOBAL_X",
    "GLOBAL_Y",
    "LOCAL_ROTATION",
    "ATOM_MOVE",
    "NA_MEASUREMENT",
    "NEUTRAL_ATOM_NATIVE_GATES",
    # Classes
    "NeutralAtomGateSet",
    "NeutralAtomGateDecomposer",
    "NeutralAtomGateTiming",
]
