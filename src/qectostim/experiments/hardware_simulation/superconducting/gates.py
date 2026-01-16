# src/qectostim/experiments/hardware_simulation/superconducting/gates.py
"""
Superconducting native gates.

Defines the native gate set for superconducting quantum computers,
including vendor-specific variants (IBM, Google, Rigetti).
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
# Superconducting Native Gate Definitions
# =============================================================================

# IBM-style gates

# Cross-resonance gate (IBM native)
CR_GATE = GateSpec(
    name="CR",
    gate_type=GateType.TWO_QUBIT,
    num_qubits=2,
    parameters=("theta",),
    is_clifford=False,
    is_native=True,
    metadata={
        "platform": "superconducting",
        "variant": "ibm",
        "description": "Cross-resonance gate",
        "typical_duration_ns": 300,
    },
)

# Echoed cross-resonance (more common in practice)
ECR_GATE = GateSpec(
    name="ECR",
    gate_type=GateType.TWO_QUBIT,
    num_qubits=2,
    is_clifford=True,
    is_native=True,
    metadata={
        "platform": "superconducting",
        "variant": "ibm",
        "description": "Echoed cross-resonance gate",
        "typical_duration_ns": 600,
    },
)

# Google-style gates

# Sqrt-iSWAP (Google Sycamore native)
SQRT_ISWAP_GATE = GateSpec(
    name="SQRT_ISWAP",
    gate_type=GateType.TWO_QUBIT,
    num_qubits=2,
    is_clifford=True,
    is_native=True,
    metadata={
        "platform": "superconducting",
        "variant": "google",
        "description": "Square root of iSWAP gate",
        "typical_duration_ns": 32,
    },
)

# Fermionic simulation gate (Google)
FSIM_GATE = GateSpec(
    name="FSIM",
    gate_type=GateType.TWO_QUBIT,
    num_qubits=2,
    parameters=("theta", "phi"),
    is_clifford=False,
    is_native=True,
    metadata={
        "platform": "superconducting",
        "variant": "google",
        "description": "Fermionic simulation gate",
        "typical_duration_ns": 32,
    },
)

# Sycamore gate (specific FSIM)
SYCAMORE_GATE = GateSpec(
    name="SYC",
    gate_type=GateType.TWO_QUBIT,
    num_qubits=2,
    is_clifford=False,
    is_native=True,
    metadata={
        "platform": "superconducting",
        "variant": "google",
        "description": "Sycamore gate (FSIM at specific angles)",
    },
)

# Generic superconducting gates

# CZ gate (common native gate)
CZ_SC_GATE = GateSpec(
    name="CZ",
    gate_type=GateType.TWO_QUBIT,
    num_qubits=2,
    is_clifford=True,
    is_native=True,
    metadata={
        "platform": "superconducting",
        "description": "Controlled-Z via flux tuning",
        "typical_duration_ns": 40,
    },
)

# Single-qubit gates (microwave pulses)
X90_GATE = GateSpec(
    name="X90",
    gate_type=GateType.SINGLE_QUBIT,
    num_qubits=1,
    is_clifford=True,
    is_native=True,
    stim_name="SQRT_X",
    metadata={
        "platform": "superconducting",
        "description": "π/2 rotation around X",
        "typical_duration_ns": 20,
    },
)

# Virtual Z rotation (frame change)
VZ_GATE = GateSpec(
    name="VZ",
    gate_type=GateType.SINGLE_QUBIT,
    num_qubits=1,
    parameters=("theta",),
    is_clifford=False,
    is_native=True,
    metadata={
        "platform": "superconducting",
        "description": "Virtual Z rotation (phase tracking, zero duration)",
        "typical_duration_ns": 0,
    },
)

# Superconducting measurement
SC_MEASUREMENT = GateSpec(
    name="M_SC",
    gate_type=GateType.MEASUREMENT,
    num_qubits=1,
    is_native=True,
    metadata={
        "platform": "superconducting",
        "description": "Dispersive readout",
        "typical_duration_ns": 1000,
        "typical_fidelity": 0.99,
    },
)

# Dictionary of all superconducting native gates
SUPERCONDUCTING_NATIVE_GATES: Dict[str, GateSpec] = {
    # IBM
    "CR": CR_GATE,
    "ECR": ECR_GATE,
    # Google
    "SQRT_ISWAP": SQRT_ISWAP_GATE,
    "FSIM": FSIM_GATE,
    "SYC": SYCAMORE_GATE,
    # Generic
    "CZ": CZ_SC_GATE,
    "X90": X90_GATE,
    "VZ": VZ_GATE,
    "M_SC": SC_MEASUREMENT,
}


# =============================================================================
# Superconducting Gate Sets (vendor-specific)
# =============================================================================

class SuperconductingGateSet(NativeGateSet):
    """Base native gate set for superconducting hardware."""
    
    def __init__(self, variant: str = "generic"):
        super().__init__(platform="superconducting")
        self.variant = variant
        
        # Add common gates
        self.add_gate(X90_GATE)
        self.add_gate(VZ_GATE)
        self.add_gate(SC_MEASUREMENT)
        
        # Add standard single-qubit gates
        for name in ["X", "Y", "Z", "H", "S"]:
            if name in STANDARD_GATES:
                self.add_gate(STANDARD_GATES[name])


class IBMGateSet(SuperconductingGateSet):
    """IBM Quantum native gate set.
    
    Native gates: ECR, X, √X, RZ
    """
    
    def __init__(self):
        super().__init__(variant="ibm")
        self.add_gate(ECR_GATE)
        self.add_gate(CR_GATE)
    
    def entangling_gate(self) -> GateSpec:
        """IBM uses ECR as the native entangling gate."""
        return self.get_gate("ECR")


class GoogleGateSet(SuperconductingGateSet):
    """Google Sycamore native gate set.
    
    Native gates: √iSWAP, FSIM, Phased XZ
    """
    
    def __init__(self):
        super().__init__(variant="google")
        self.add_gate(SQRT_ISWAP_GATE)
        self.add_gate(FSIM_GATE)
        self.add_gate(SYCAMORE_GATE)
    
    def entangling_gate(self) -> GateSpec:
        """Google uses √iSWAP as the native entangling gate."""
        return self.get_gate("SQRT_ISWAP")


class TunableCouplerGateSet(SuperconductingGateSet):
    """Gate set for tunable coupler architectures.
    
    Native gates: CZ (via flux tuning), single-qubit rotations
    """
    
    def __init__(self):
        super().__init__(variant="tunable_coupler")
        self.add_gate(CZ_SC_GATE)
    
    def entangling_gate(self) -> GateSpec:
        """Tunable coupler uses CZ."""
        return self.get_gate("CZ")


# =============================================================================
# Gate Decomposer for Superconducting
# =============================================================================

class SuperconductingGateDecomposer(GateDecomposer):
    """Decomposes gates to superconducting native operations.
    
    NOT IMPLEMENTED: This is a stub for the decomposition logic.
    """
    
    def __init__(self, gate_set: Optional[SuperconductingGateSet] = None):
        super().__init__(gate_set or SuperconductingGateSet())
    
    def decompose(
        self,
        gate: GateSpec | ParameterizedGate,
        qubits: Tuple[int, ...],
    ) -> GateDecomposition:
        """Decompose a gate to superconducting natives.
        
        NOT IMPLEMENTED.
        """
        raise NotImplementedError(
            "SuperconductingGateDecomposer.decompose() not yet implemented."
        )


# =============================================================================
# Gate Timing Models
# =============================================================================

@dataclass
class SuperconductingGateTiming:
    """Timing parameters for superconducting gates.
    
    Attributes
    ----------
    single_qubit_time : float
        Single-qubit gate duration in seconds.
    two_qubit_time : float
        Two-qubit gate duration in seconds.
    measurement_time : float
        Measurement duration in seconds.
    reset_time : float
        Active reset duration in seconds.
    """
    single_qubit_time: float = 20e-9  # 20 ns
    two_qubit_time: float = 200e-9  # 200 ns (varies by gate type)
    measurement_time: float = 1000e-9  # 1 μs
    reset_time: float = 500e-9  # 500 ns (active reset)
    
    # Gate-specific timings
    ecr_time: float = 600e-9  # 600 ns
    sqrt_iswap_time: float = 32e-9  # 32 ns
    cz_time: float = 40e-9  # 40 ns


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Gate specifications
    "CR_GATE",
    "ECR_GATE",
    "SQRT_ISWAP_GATE",
    "FSIM_GATE",
    "SYCAMORE_GATE",
    "CZ_SC_GATE",
    "X90_GATE",
    "VZ_GATE",
    "SC_MEASUREMENT",
    "SUPERCONDUCTING_NATIVE_GATES",
    # Classes
    "SuperconductingGateSet",
    "IBMGateSet",
    "GoogleGateSet",
    "TunableCouplerGateSet",
    "SuperconductingGateDecomposer",
    "SuperconductingGateTiming",
]
