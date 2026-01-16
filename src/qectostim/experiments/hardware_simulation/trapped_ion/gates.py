# src/qectostim/experiments/hardware_simulation/trapped_ion/gates.py
"""
Trapped ion native gates.

Defines the native gate set for trapped ion quantum computers,
including Mølmer-Sørensen gates and single-ion rotations.
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
# Trapped Ion Native Gate Definitions
# =============================================================================

# Mølmer-Sørensen gate (native 2-qubit entangling gate)
MS_GATE = GateSpec(
    name="MS",
    gate_type=GateType.TWO_QUBIT,
    num_qubits=2,
    parameters=("theta", "phi"),
    is_clifford=False,
    is_native=True,
    metadata={
        "platform": "trapped_ion",
        "description": "Mølmer-Sørensen entangling gate",
        "typical_duration_us": 40.0,
        "typical_fidelity": 0.995,
    },
)

# XX gate (MS at specific angle) - Clifford version
XX_GATE = GateSpec(
    name="XX",
    gate_type=GateType.TWO_QUBIT,
    num_qubits=2,
    parameters=("theta",),
    is_clifford=False,
    is_native=True,
    metadata={
        "platform": "trapped_ion",
        "description": "XX rotation gate (parameterized MS)",
    },
)

# Global rotation (all ions in the chain)
GLOBAL_ROTATION = GateSpec(
    name="GR",
    gate_type=GateType.MULTI_QUBIT,
    num_qubits=-1,  # Variable: all ions
    parameters=("theta", "phi"),
    is_clifford=False,
    is_native=True,
    metadata={
        "platform": "trapped_ion",
        "description": "Global rotation - acts on all ions simultaneously",
        "typical_duration_us": 5.0,
    },
)

# Single-ion rotation (addressed beam)
SINGLE_ION_ROTATION = GateSpec(
    name="R",
    gate_type=GateType.SINGLE_QUBIT,
    num_qubits=1,
    parameters=("theta", "phi"),
    is_clifford=False,
    is_native=True,
    metadata={
        "platform": "trapped_ion",
        "description": "Single-ion rotation with addressed beam",
        "typical_duration_us": 5.0,
        "typical_fidelity": 0.9997,
    },
)

# X rotation (special case of R)
X_ROTATION = GateSpec(
    name="RX",
    gate_type=GateType.SINGLE_QUBIT,
    num_qubits=1,
    parameters=("theta",),
    is_clifford=False,
    is_native=True,
    metadata={"platform": "trapped_ion"},
)

# Y rotation (special case of R)
Y_ROTATION = GateSpec(
    name="RY",
    gate_type=GateType.SINGLE_QUBIT,
    num_qubits=1,
    parameters=("theta",),
    is_clifford=False,
    is_native=True,
    metadata={"platform": "trapped_ion"},
)

# Z rotation (virtual, zero duration)
Z_ROTATION = GateSpec(
    name="RZ",
    gate_type=GateType.SINGLE_QUBIT,
    num_qubits=1,
    parameters=("theta",),
    is_clifford=False,
    is_native=True,
    metadata={
        "platform": "trapped_ion",
        "description": "Virtual Z rotation (phase tracking)",
        "typical_duration_us": 0.0,  # Virtual gate
    },
)

# Trapped ion measurement (fluorescence detection)
ION_MEASUREMENT = GateSpec(
    name="M_ION",
    gate_type=GateType.MEASUREMENT,
    num_qubits=1,
    is_native=True,
    metadata={
        "platform": "trapped_ion",
        "description": "Fluorescence measurement",
        "typical_duration_us": 400.0,
        "typical_fidelity": 0.999,
    },
)

# Dictionary of all trapped ion native gates
TRAPPED_ION_NATIVE_GATES: Dict[str, GateSpec] = {
    "MS": MS_GATE,
    "XX": XX_GATE,
    "GR": GLOBAL_ROTATION,
    "R": SINGLE_ION_ROTATION,
    "RX": X_ROTATION,
    "RY": Y_ROTATION,
    "RZ": Z_ROTATION,
    "M_ION": ION_MEASUREMENT,
}


# =============================================================================
# Trapped Ion Gate Set
# =============================================================================

class TrappedIonGateSet(NativeGateSet):
    """Native gate set for trapped ion hardware.
    
    Includes:
    - MS (Mølmer-Sørensen) two-qubit entangling gate
    - Single-ion rotations (RX, RY, RZ)
    - Global rotations (all ions)
    - Measurement via fluorescence
    
    Example
    -------
    >>> gate_set = TrappedIonGateSet()
    >>> gate_set.has_gate("MS")
    True
    >>> ms = gate_set.get_gate("MS")
    >>> ms.parameters
    ('theta', 'phi')
    """
    
    def __init__(self):
        super().__init__(platform="trapped_ion")
        
        # Add trapped ion native gates
        for name, spec in TRAPPED_ION_NATIVE_GATES.items():
            self.add_gate(spec)
        
        # Add standard single-qubit Clifford gates (implemented via rotations)
        for name in ["H", "S", "S_DAG", "X", "Y", "Z"]:
            if name in STANDARD_GATES:
                self.add_gate(STANDARD_GATES[name])
    
    def entangling_gate(self) -> GateSpec:
        """Get the native entangling gate (MS)."""
        return self.get_gate("MS")
    
    def has_global_operations(self) -> bool:
        """Trapped ions support global rotations."""
        return True


# =============================================================================
# Gate Decomposer for Trapped Ions
# =============================================================================

class TrappedIonGateDecomposer(GateDecomposer):
    """Decomposes gates to trapped ion native operations.
    
    Standard decompositions:
    - CNOT → MS + single-qubit rotations
    - H → RY(π/2) + RZ(π)
    - CZ → H + CNOT + H
    
    NOT IMPLEMENTED: This is a stub for the decomposition logic.
    """
    
    def __init__(self):
        super().__init__(TrappedIonGateSet())
    
    def decompose(
        self,
        gate: GateSpec | ParameterizedGate,
        qubits: Tuple[int, ...],
    ) -> GateDecomposition:
        """Decompose a gate into MS + rotations.
        
        NOT IMPLEMENTED.
        """
        name = gate.name if isinstance(gate, GateSpec) else gate.spec.name
        
        if self.is_native(gate):
            # Already native
            return GateDecomposition(
                original=gate if isinstance(gate, GateSpec) else gate.spec,
                sequence=[(gate, qubits)],
                cost=0.0,
            )
        
        # Decomposition logic to be implemented
        raise NotImplementedError(
            f"TrappedIonGateDecomposer: decomposition for {name!r} not yet implemented."
        )
    
    def decompose_cnot(self, control: int, target: int) -> GateDecomposition:
        """Decompose CNOT to MS + rotations.
        
        CNOT = (I ⊗ RY(-π/2)) · MS(π/4) · (RX(π/2) ⊗ RX(π/2)) · MS(π/4) · (I ⊗ RY(π/2))
        
        NOT IMPLEMENTED.
        """
        raise NotImplementedError(
            "TrappedIonGateDecomposer.decompose_cnot() not yet implemented."
        )


# =============================================================================
# Gate Timing Models
# =============================================================================

@dataclass
class TrappedIonGateTiming:
    """Timing parameters for trapped ion gates.
    
    Based on experimental values from:
    - https://arxiv.org/pdf/2004.04706 (Honeywell QCCD)
    - https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    
    Attributes
    ----------
    ms_gate_time : float
        MS gate duration in seconds.
    single_qubit_time : float
        Single-qubit rotation duration in seconds.
    measurement_time : float
        Fluorescence measurement duration in seconds.
    reset_time : float
        Qubit reset duration in seconds.
    """
    ms_gate_time: float = 40e-6  # 40 μs
    single_qubit_time: float = 5e-6  # 5 μs
    measurement_time: float = 400e-6  # 400 μs
    reset_time: float = 50e-6  # 50 μs
    
    # AM gate scaling (from paper)
    am1_base: float = 100e-6  # μs per ion distance - 22
    am2_base: float = 38e-6  # μs per ion distance + 10
    pm_base: float = 5e-6  # μs per ion distance + 160
    
    def ms_duration(
        self,
        ion_distance: int = 1,
        gate_type: str = "AM2",
        chain_length: int = 2,
    ) -> float:
        """Calculate MS gate duration based on ion positions.
        
        Parameters
        ----------
        ion_distance : int
            Distance between ions in the chain.
        gate_type : str
            Gate type: "AM1", "AM2", "PM", or "FM".
        chain_length : int
            Number of ions in the chain.
            
        Returns
        -------
        float
            Gate duration in seconds.
        """
        # Simplified model - actual implementation would use chain length
        if gate_type == "AM1":
            return max(self.am1_base * ion_distance - 22e-6, 0)
        elif gate_type == "AM2":
            return max(self.am2_base * ion_distance + 10e-6, 0)
        elif gate_type == "PM":
            return max(self.pm_base * ion_distance + 160e-6, 0)
        else:
            return self.ms_gate_time


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Gate specifications
    "MS_GATE",
    "XX_GATE",
    "GLOBAL_ROTATION",
    "SINGLE_ION_ROTATION",
    "X_ROTATION",
    "Y_ROTATION",
    "Z_ROTATION",
    "ION_MEASUREMENT",
    "TRAPPED_ION_NATIVE_GATES",
    # Classes
    "TrappedIonGateSet",
    "TrappedIonGateDecomposer",
    "TrappedIonGateTiming",
]
