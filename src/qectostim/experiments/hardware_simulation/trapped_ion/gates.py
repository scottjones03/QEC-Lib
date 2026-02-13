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
from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
    DEFAULT_CALIBRATION as _CAL,
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
        "typical_duration_us": _CAL.ms_gate_time * 1e6,
        "typical_fidelity": _CAL.gate_fidelities().get("MS", 0.995),
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
        "typical_duration_us": _CAL.single_qubit_gate_time * 1e6,
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
        "typical_duration_us": _CAL.single_qubit_gate_time * 1e6,
        "typical_fidelity": _CAL.gate_fidelities().get("RX", 0.9997),
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
        "typical_duration_us": _CAL.measurement_time * 1e6,
        "typical_fidelity": 1.0 - _CAL.measurement_infidelity,
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
    - CNOT → MS(π/4) + single-qubit rotations  (2-qubit cost = 1 MS)
    - CZ   → H(target) + CNOT + H(target)      (2-qubit cost = 1 MS)
    - H    → RY(π/2) · RZ(π)                   (virtual Z is free)
    - SWAP → 3× CNOT  → 3 MS gates

    Reference: arXiv:2004.04706, Fig. 5; PRA 99, 022330.
    """

    def __init__(self):
        super().__init__(TrappedIonGateSet())
        # Build a decomposition lookup table
        self._table = self._build_decomposition_table()

    # -----------------------------------------------------------------
    # Internal: pre-built table of (gate_name → factory)
    # -----------------------------------------------------------------
    def _build_decomposition_table(self):
        """Return { gate_name: callable(qubits) -> GateDecomposition }."""
        import math
        pi = math.pi

        table = {}

        # --- H → RY(π/2) · RZ(π) --------------------------------
        def _decompose_h(qubits):
            q = qubits[0]
            ry = Y_ROTATION.with_parameters(theta=pi / 2)
            rz = Z_ROTATION.with_parameters(theta=pi)
            return GateDecomposition(
                original=STANDARD_GATES["H"],
                sequence=[(ry, (q,)), (rz, (q,))],
                cost=_CAL.single_qubit_gate_time * 1e6,  # single-qubit gate (RZ is virtual)
            )
        table["H"] = _decompose_h

        # --- CNOT → (I⊗RY(-π/2)) · MS(π/4) · (RX(-π/2)⊗RX(-π/2)) · MS(π/4) · (I⊗RY(π/2))
        #     Simplified: 2 MS + 3 single-qubit rotations
        #     Using the compact form: RY(-π/2)_t · MS · RX(-π/2)_c · RX(-π/2)_t · RY(π/2)_t
        def _decompose_cnot(qubits):
            c, t = qubits
            ms = MS_GATE.with_parameters(theta=pi / 4, phi=0.0)
            ry_pos = Y_ROTATION.with_parameters(theta=pi / 2)
            ry_neg = Y_ROTATION.with_parameters(theta=-pi / 2)
            rx_neg = X_ROTATION.with_parameters(theta=-pi / 2)
            return GateDecomposition(
                original=STANDARD_GATES.get("CNOT", STANDARD_GATES.get("CX")),
                sequence=[
                    (ry_neg, (t,)),
                    (ms, (c, t)),
                    (rx_neg, (c,)),
                    (rx_neg, (t,)),
                    (ry_pos, (t,)),
                ],
                cost=_CAL.ms_gate_time * 1e6 + 3 * _CAL.single_qubit_gate_time * 1e6,  # 1 MS + 3 rotations
            )
        table["CNOT"] = _decompose_cnot
        table["CX"] = _decompose_cnot

        # --- CZ → H_t · CNOT · H_t ---------------------------------
        def _decompose_cz(qubits):
            c, t = qubits
            # Inline H as RY(π/2)·RZ(π)
            ry_h = Y_ROTATION.with_parameters(theta=pi / 2)
            rz_h = Z_ROTATION.with_parameters(theta=pi)
            # CNOT decomposition pieces
            ms = MS_GATE.with_parameters(theta=pi / 4, phi=0.0)
            ry_pos = Y_ROTATION.with_parameters(theta=pi / 2)
            ry_neg = Y_ROTATION.with_parameters(theta=-pi / 2)
            rx_neg = X_ROTATION.with_parameters(theta=-pi / 2)
            return GateDecomposition(
                original=STANDARD_GATES["CZ"],
                sequence=[
                    # H on target
                    (ry_h, (t,)), (rz_h, (t,)),
                    # CNOT
                    (ry_neg, (t,)),
                    (ms, (c, t)),
                    (rx_neg, (c,)),
                    (rx_neg, (t,)),
                    (ry_pos, (t,)),
                    # H on target
                    (ry_h, (t,)), (rz_h, (t,)),
                ],
                cost=_CAL.ms_gate_time * 1e6 + 7 * _CAL.single_qubit_gate_time * 1e6,  # 1 MS + 7 rotations
            )
        table["CZ"] = _decompose_cz

        # --- SWAP → 3× CNOT → 3 MS gates ----------------------------
        def _decompose_swap(qubits):
            c, t = qubits
            cnot_ct = _decompose_cnot((c, t))
            cnot_tc = _decompose_cnot((t, c))
            seq = cnot_ct.sequence + cnot_tc.sequence + cnot_ct.sequence
            return GateDecomposition(
                original=STANDARD_GATES["SWAP"],
                sequence=seq,
                cost=cnot_ct.cost * 3,
            )
        table["SWAP"] = _decompose_swap

        # --- S → RZ(π/2) -------------------------------------------
        def _decompose_s(qubits):
            q = qubits[0]
            rz = Z_ROTATION.with_parameters(theta=pi / 2)
            return GateDecomposition(
                original=STANDARD_GATES["S"],
                sequence=[(rz, (q,))],
                cost=0.0,  # virtual Z
            )
        table["S"] = _decompose_s

        # --- S_DAG → RZ(-π/2) --------------------------------------
        def _decompose_s_dag(qubits):
            q = qubits[0]
            rz = Z_ROTATION.with_parameters(theta=-pi / 2)
            return GateDecomposition(
                original=STANDARD_GATES["S_DAG"],
                sequence=[(rz, (q,))],
                cost=0.0,
            )
        table["S_DAG"] = _decompose_s_dag

        # --- T → RZ(π/4) -------------------------------------------
        def _decompose_t(qubits):
            q = qubits[0]
            rz = Z_ROTATION.with_parameters(theta=pi / 4)
            return GateDecomposition(
                original=STANDARD_GATES["T"],
                sequence=[(rz, (q,))],
                cost=0.0,
            )
        table["T"] = _decompose_t

        # --- T_DAG → RZ(-π/4) --------------------------------------
        def _decompose_t_dag(qubits):
            q = qubits[0]
            rz = Z_ROTATION.with_parameters(theta=-pi / 4)
            return GateDecomposition(
                original=STANDARD_GATES["T_DAG"],
                sequence=[(rz, (q,))],
                cost=0.0,
            )
        table["T_DAG"] = _decompose_t_dag

        return table

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def decompose(
        self,
        gate: GateSpec | ParameterizedGate,
        qubits: Tuple[int, ...],
    ) -> GateDecomposition:
        """Decompose a gate into MS + rotations."""
        name = gate.name if isinstance(gate, GateSpec) else gate.spec.name

        if self.is_native(gate):
            return GateDecomposition(
                original=gate if isinstance(gate, GateSpec) else gate.spec,
                sequence=[(gate, qubits)],
                cost=0.0,
            )

        factory = self._table.get(name)
        if factory is not None:
            return factory(qubits)

        raise NotImplementedError(
            f"TrappedIonGateDecomposer: decomposition for {name!r} not yet implemented."
        )

    def decompose_cnot(self, control: int, target: int) -> GateDecomposition:
        """Decompose CNOT to MS + rotations.

        CNOT = (I ⊗ RY(-π/2)) · MS(π/4) · (RX(-π/2) ⊗ RX(-π/2)) · (I ⊗ RY(π/2))
        """
        return self.decompose(STANDARD_GATES.get("CNOT", STANDARD_GATES.get("CX")), (control, target))


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
    ms_gate_time: float = _CAL.ms_gate_time        # 40 μs
    single_qubit_time: float = _CAL.single_qubit_gate_time  # 5 μs
    measurement_time: float = _CAL.measurement_time  # 400 μs
    reset_time: float = _CAL.reset_time              # 50 μs
    
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
