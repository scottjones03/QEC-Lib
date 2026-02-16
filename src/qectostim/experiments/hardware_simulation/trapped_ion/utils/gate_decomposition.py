"""Standalone gate decomposition for trapped-ion native gate set.

Decomposes standard quantum gates into the trapped-ion native gate set:
    MS  (Molmer-Sorensen 2-qubit entangling gate)
    RX  (rotation about X)
    RY  (rotation about Y)
    RZ  (rotation about Z, virtual — near-zero duration)
    M   (measurement in Z basis)
    R   (reset to |0>)

Ported from ``trapped_ion/compilers/base.py`` for use in the ``old/``
module without requiring the ``core/`` framework or any ABC inheritance.

References
----------
* PRA 99, 022330, Fig 4 — CNOT decomposition.
* Figgatt thesis p.80 — Hadamard decomposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PI = math.pi
PI_2 = math.pi / 2
PI_4 = math.pi / 4

# Native gate set for trapped ions
NATIVE_GATES = frozenset({"MS", "RX", "RY", "RZ", "M", "MX", "MY", "R"})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DecomposedGate:
    """A single gate in a decomposition sequence.

    Attributes
    ----------
    name : str
        Gate name (MS, RX, RY, RZ, M, R, etc.).
    qubits : Tuple[int, ...]
        Target qubit indices.
    params : Dict[str, float]
        Gate parameters (e.g., ``{"angle": pi/2}``).
    """
    name: str
    qubits: Tuple[int, ...]
    params: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v:.4f}" for k, v in self.params.items())
        qs = ",".join(str(q) for q in self.qubits)
        return f"{self.name}({qs}{'; ' + p if p else ''})"


# ---------------------------------------------------------------------------
# Elementary decompositions
# ---------------------------------------------------------------------------

def _decompose_cnot(control: int, target: int) -> List[DecomposedGate]:
    """CNOT → 1 MS + 4 rotations (PRA 99, 022330, Fig 4).

    RY(ctl)  RX(ctl)  RX(tgt)  MS(ctl,tgt)  RY(ctl)
    """
    return [
        DecomposedGate("RY", (control,), {"angle": PI_2}),
        DecomposedGate("RX", (control,), {"angle": PI_2}),
        DecomposedGate("RX", (target,),  {"angle": PI_2}),
        DecomposedGate("MS", (control, target), {"angle": PI_4}),
        DecomposedGate("RY", (control,), {"angle": PI_2}),
    ]


def _decompose_h(qubit: int) -> List[DecomposedGate]:
    """Hadamard → RY(π/2) · RX(π/2) (Figgatt thesis p.80)."""
    return [
        DecomposedGate("RY", (qubit,), {"angle": PI_2}),
        DecomposedGate("RX", (qubit,), {"angle": PI_2}),
    ]


def _decompose_s(qubit: int) -> List[DecomposedGate]:
    """S gate → RZ(π/2)."""
    return [DecomposedGate("RZ", (qubit,), {"angle": PI_2})]


def _decompose_s_dag(qubit: int) -> List[DecomposedGate]:
    """S† → RZ(−π/2)."""
    return [DecomposedGate("RZ", (qubit,), {"angle": -PI_2})]


def _decompose_t(qubit: int) -> List[DecomposedGate]:
    """T gate → RZ(π/4)."""
    return [DecomposedGate("RZ", (qubit,), {"angle": PI_4})]


def _decompose_t_dag(qubit: int) -> List[DecomposedGate]:
    """T† → RZ(−π/4)."""
    return [DecomposedGate("RZ", (qubit,), {"angle": -PI_4})]


def _decompose_x(qubit: int) -> List[DecomposedGate]:
    """X gate → RX(π)."""
    return [DecomposedGate("RX", (qubit,), {"angle": PI})]


def _decompose_y(qubit: int) -> List[DecomposedGate]:
    """Y gate → RY(π)."""
    return [DecomposedGate("RY", (qubit,), {"angle": PI})]


def _decompose_z(qubit: int) -> List[DecomposedGate]:
    """Z gate → RZ(π)."""
    return [DecomposedGate("RZ", (qubit,), {"angle": PI})]


def _decompose_cz(control: int, target: int) -> List[DecomposedGate]:
    """CZ → 1 MS + 7 rotations (IH·CNOT·IH cancellation, PRA 99, 022330)."""
    return [
        DecomposedGate("RY", (control,), {"angle": PI_2}),
        DecomposedGate("RX", (control,), {"angle": PI_2}),
        DecomposedGate("RY", (target,),  {"angle": PI_2}),
        DecomposedGate("RX", (target,),  {"angle": PI_2}),
        DecomposedGate("MS", (control, target), {"angle": PI_4}),
        DecomposedGate("RY", (control,), {"angle": PI_2}),
        DecomposedGate("RY", (target,),  {"angle": PI_2}),
        DecomposedGate("RX", (target,),  {"angle": PI_2}),
    ]


def _decompose_swap(q1: int, q2: int) -> List[DecomposedGate]:
    """SWAP → 3 CNOTs = 3 MS + 12 rotations."""
    ops: List[DecomposedGate] = []
    ops.extend(_decompose_cnot(q1, q2))
    ops.extend(_decompose_cnot(q2, q1))
    ops.extend(_decompose_cnot(q1, q2))
    return ops


def _decompose_iswap(q1: int, q2: int) -> List[DecomposedGate]:
    """iSWAP → 2 MS gates."""
    return [
        DecomposedGate("MS", (q1, q2), {"angle": PI_4}),
        DecomposedGate("MS", (q1, q2), {"angle": PI_4}),
    ]


# ---------------------------------------------------------------------------
# Two-qubit gate names (applied pair-wise on targets)
# ---------------------------------------------------------------------------

_TWO_Q_GATE_NAMES = frozenset(
    ("CX", "CNOT", "ZCX", "CZ", "ZCZ", "SWAP", "ISWAP",
     "ISWAP_DAG", "SQRT_XX", "SQRT_XX_DAG")
)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

_CACHE: Dict[str, List[DecomposedGate]] = {}


def decompose_stim_gate(
    gate_name: str,
    qubits: Tuple[int, ...],
    *,
    use_cache: bool = True,
) -> List[DecomposedGate]:
    """Decompose a Stim gate to native trapped-ion gates.

    Multi-target 2Q instructions (e.g. ``CX 0 1 2 3 4 5``) are split
    into consecutive qubit pairs and each pair is decomposed
    independently.

    Parameters
    ----------
    gate_name : str
        Stim gate name (e.g. ``"CX"``, ``"H"``, ``"M"``).
    qubits : Tuple[int, ...]
        Target qubit indices.
    use_cache : bool
        If True, cache decomposition results for reuse.

    Returns
    -------
    List[DecomposedGate]
        Sequence of native gates implementing the input gate.

    Raises
    ------
    ValueError
        If ``gate_name`` is not recognized.
    """
    if use_cache:
        cache_key = f"{gate_name}_{qubits}"
        if cache_key in _CACHE:
            return _CACHE[cache_key]

    result: List[DecomposedGate]

    # --- Two-qubit gates (handle multi-target pairs) ---
    if gate_name in _TWO_Q_GATE_NAMES and len(qubits) >= 2:
        result = []
        for pi in range(0, len(qubits) - 1, 2):
            q0, q1 = qubits[pi], qubits[pi + 1]
            if gate_name in ("CX", "CNOT", "ZCX"):
                result.extend(_decompose_cnot(q0, q1))
            elif gate_name in ("CZ", "ZCZ"):
                result.extend(_decompose_cz(q0, q1))
            elif gate_name == "SWAP":
                result.extend(_decompose_swap(q0, q1))
            elif gate_name in ("ISWAP", "ISWAP_DAG"):
                result.extend(_decompose_iswap(q0, q1))
            elif gate_name in ("SQRT_XX", "SQRT_XX_DAG"):
                # SQRT_XX ≈ single MS gate
                result.append(DecomposedGate("MS", (q0, q1), {"angle": PI_4}))

    # --- Single-qubit gates (may target multiple qubits) ---
    elif gate_name == "H":
        result = []
        for q in qubits:
            result.extend(_decompose_h(q))

    elif gate_name == "X":
        result = [DecomposedGate("RX", (q,), {"angle": PI}) for q in qubits]

    elif gate_name == "Y":
        result = [DecomposedGate("RY", (q,), {"angle": PI}) for q in qubits]

    elif gate_name == "Z":
        result = [DecomposedGate("RZ", (q,), {"angle": PI}) for q in qubits]

    elif gate_name == "S":
        result = [DecomposedGate("RZ", (q,), {"angle": PI_2}) for q in qubits]

    elif gate_name == "S_DAG":
        result = [DecomposedGate("RZ", (q,), {"angle": -PI_2}) for q in qubits]

    elif gate_name == "T":
        result = [DecomposedGate("RZ", (q,), {"angle": PI_4}) for q in qubits]

    elif gate_name == "T_DAG":
        result = [DecomposedGate("RZ", (q,), {"angle": -PI_4}) for q in qubits]

    elif gate_name == "SQRT_X":
        result = [DecomposedGate("RX", (q,), {"angle": PI_2}) for q in qubits]

    elif gate_name == "SQRT_X_DAG":
        result = [DecomposedGate("RX", (q,), {"angle": -PI_2}) for q in qubits]

    elif gate_name == "SQRT_Y":
        result = [DecomposedGate("RY", (q,), {"angle": PI_2}) for q in qubits]

    elif gate_name == "SQRT_Y_DAG":
        result = [DecomposedGate("RY", (q,), {"angle": -PI_2}) for q in qubits]

    # --- Identity (no-op) ---
    elif gate_name == "I":
        result = []

    # --- Reset operations ---
    elif gate_name in ("R", "RZ"):
        # Stim R / RZ = reset in Z basis (prepare |0>)
        result = [DecomposedGate("R", (q,), {}) for q in qubits]

    elif gate_name == "RX":
        # Stim RX = reset in X basis (prepare |+>)
        # Decompose: R (reset |0>) then H (|0> -> |+>)
        result = []
        for q in qubits:
            result.append(DecomposedGate("R", (q,), {}))
            result.extend(_decompose_h(q))

    elif gate_name == "RY":
        # Stim RY = reset in Y basis (prepare |+i>)
        # Decompose: R (reset |0>) then S then H
        result = []
        for q in qubits:
            result.append(DecomposedGate("R", (q,), {}))
            result.extend(_decompose_s(q))
            result.extend(_decompose_h(q))

    # --- Measure + Reset ---
    elif gate_name == "MR":
        # MR = measure + reset in Z basis
        result = []
        for q in qubits:
            result.append(DecomposedGate("M", (q,), {}))
        for q in qubits:
            result.append(DecomposedGate("R", (q,), {}))

    elif gate_name == "MRX":
        # MRX = measure in X then reset in X
        result = []
        for q in qubits:
            result.append(DecomposedGate("MX", (q,), {}))
            result.append(DecomposedGate("R", (q,), {}))
            result.extend(_decompose_h(q))

    elif gate_name == "MRY":
        # MRY = measure in Y then reset in Y
        result = []
        for q in qubits:
            result.append(DecomposedGate("MY", (q,), {}))
            result.append(DecomposedGate("R", (q,), {}))
            result.extend(_decompose_s(q))
            result.extend(_decompose_h(q))

    # --- Measurements ---
    elif gate_name in ("M", "MX", "MY", "MZ"):
        # Split multi-qubit measurement into per-qubit ops
        result = [DecomposedGate(gate_name, (q,), {}) for q in qubits]

    else:
        raise ValueError(f"Unknown gate for decomposition: {gate_name}")

    if use_cache:
        _CACHE[f"{gate_name}_{qubits}"] = result

    return result


def clear_cache() -> None:
    """Clear the decomposition cache."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Counting helpers
# ---------------------------------------------------------------------------

def count_native_gates(
    decomposed: List[DecomposedGate],
) -> Dict[str, int]:
    """Count native gates by type in a decomposition.

    Returns
    -------
    Dict[str, int]
        Mapping from gate name to count.
    """
    counts: Dict[str, int] = {}
    for gate in decomposed:
        counts[gate.name] = counts.get(gate.name, 0) + 1
    return counts


def count_ms_gates(decomposed: List[DecomposedGate]) -> int:
    """Count the number of MS (entangling) gates in a decomposition."""
    return sum(1 for g in decomposed if g.name == "MS")
