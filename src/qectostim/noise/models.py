from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import stim


class NoiseModel(ABC):
    """Abstract noise model.

    A NoiseModel knows how to take an *ideal* Stim circuit and return a new
    circuit with noise channels inserted. Experiments are responsible for
    constructing the ideal circuit; noise models are responsible for decorating
    it with noise.
    """

    @abstractmethod
    def apply(self, circuit: stim.Circuit) -> stim.Circuit:
        """Return a *new* circuit with noise inserted.

        Implementations are free to mutate `circuit` in-place and return it,
        but callers should not rely on this.
        """
        raise NotImplementedError


class CircuitDepolarizingNoise(NoiseModel):
    """Insert depolarizing noise after 1q/2q gates, *except* on final data gates.

    This avoids some DEM error mechanisms that flip only the logical observable
    L0 with no detectors (pure logical faults at the readout layer), by not
    injecting noise on the last gate touching each qubit.
    """

    def __init__(self, p1: float = 0.0, p2: float = 0.0):
        self.p1 = float(p1)
        self.p2 = float(p2)

    def apply(self, circuit: stim.Circuit) -> stim.Circuit:
        """Apply circuit depolarizing noise.
        
        Handles REPEAT blocks by recursively applying noise to their body.
        """
        return self._apply_impl(circuit)
    
    def _apply_impl(self, circuit: stim.Circuit) -> stim.Circuit:
        """Implementation of noise application."""
        # First pass: find index of the last non-measurement/non-reset gate
        # acting on each qubit. We do *not* treat M/DETECTOR/OBSERVABLE_INCLUDE
        # as gates for this purpose.
        last_gate_on_qubit: Dict[int, int] = {}

        instructions: List = list(circuit)
        for idx, inst in enumerate(instructions):
            # Handle REPEAT blocks - don't track their internal structure for last_gate
            if isinstance(inst, stim.CircuitRepeatBlock):
                continue
            name = inst.name.upper()
            if name in {"M", "MR", "R", "MRX", "MX", "MY", "MZ",
                        "DETECTOR", "OBSERVABLE_INCLUDE",
                        "TICK", "SHIFT_COORDS"}:
                continue

            for t in inst.targets_copy():
                if t.is_qubit_target:
                    q = t.value
                    last_gate_on_qubit[q] = idx

        noisy = stim.Circuit()

        for idx, inst in enumerate(instructions):
            # Handle REPEAT blocks recursively
            if isinstance(inst, stim.CircuitRepeatBlock):
                noisy_body = self._apply_impl(inst.body_copy())
                noisy.append(stim.CircuitRepeatBlock(inst.repeat_count, noisy_body))
                continue
            
            noisy.append(inst)
            name = inst.name.upper()

            # Extract only qubit targets.
            qubit_targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if not qubit_targets:
                continue

            # If this is the last gate touching *all* of these qubits,
            # skip injecting noise here. This suppresses many "naked L0"
            # error terms that correspond to pure logical readout faults.
            is_final_touch = all(last_gate_on_qubit.get(q, -1) == idx for q in qubit_targets)

            # 1-qubit gates: add DEPOLARIZE1 on each target unless it's the last touch.
            if name in {
                "H", "X", "Y", "Z", "S", "S_DAG",
                "SQRT_X", "SQRT_X_DAG",
                "SQRT_Y", "SQRT_Y_DAG",
            } and self.p1 > 0 and not is_final_touch:
                noisy.append("DEPOLARIZE1", qubit_targets, self.p1)

            # 2-qubit gates: add DEPOLARIZE2 in pairs unless they're all final touches.
            if name in {"CX", "CNOT", "CZ", "ISWAP", "SWAP"} and self.p2 > 0 and not is_final_touch:
                if len(qubit_targets) % 2 != 0:
                    raise ValueError(
                        f"Gate {name} has odd number of qubit targets: {qubit_targets}"
                    )
                noisy.append("DEPOLARIZE2", qubit_targets, self.p2)

        return noisy


class StimStyleDepolarizingNoise(NoiseModel):
    """Insert depolarizing noise after ALL Clifford gates.
    
    This exactly matches Stim's `after_clifford_depolarization` parameter behavior
    in `stim.Circuit.generated()`. Unlike CircuitDepolarizingNoise, this does NOT
    skip noise on the final gate touching each qubit.
    
    Use this noise model when comparing our circuits against Stim's reference
    circuits to ensure fair comparison.
    
    Parameters
    ----------
    p : float
        Depolarization probability. Applied as DEPOLARIZE1(p) after 1-qubit
        Clifford gates and DEPOLARIZE2(p) after 2-qubit Clifford gates.
    """

    def __init__(self, p: float = 0.0):
        self.p = float(p)

    def apply(self, circuit: stim.Circuit) -> stim.Circuit:
        """Apply Stim-style depolarizing noise after every Clifford gate.
        
        Handles REPEAT blocks by recursively applying noise to their body.
        """
        noisy = stim.Circuit()

        for inst in circuit:
            # Handle REPEAT blocks recursively
            if isinstance(inst, stim.CircuitRepeatBlock):
                noisy_body = self.apply(inst.body_copy())
                noisy.append(stim.CircuitRepeatBlock(inst.repeat_count, noisy_body))
                continue
            
            noisy.append(inst)
            name = inst.name.upper()

            # Extract only qubit targets.
            qubit_targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if not qubit_targets or self.p <= 0:
                continue

            # 1-qubit Clifford gates: add DEPOLARIZE1
            # This matches the gates Stim adds noise after with after_clifford_depolarization
            if name in {
                "H", "X", "Y", "Z", "S", "S_DAG",
                "SQRT_X", "SQRT_X_DAG",
                "SQRT_Y", "SQRT_Y_DAG",
                "C_XYZ", "C_ZYX",  # Additional Clifford gates
                "H_XY", "H_XZ", "H_YZ",  # Hadamard variants
            }:
                noisy.append("DEPOLARIZE1", qubit_targets, self.p)

            # 2-qubit Clifford gates: add DEPOLARIZE2
            if name in {"CX", "CNOT", "CZ", "CY", "ISWAP", "ISWAP_DAG", 
                        "SWAP", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ",
                        "ZCX", "ZCY", "ZCZ", "SQRT_XX", "SQRT_XX_DAG",
                        "SQRT_YY", "SQRT_YY_DAG", "SQRT_ZZ", "SQRT_ZZ_DAG"}:
                if len(qubit_targets) % 2 != 0:
                    raise ValueError(
                        f"Gate {name} has odd number of qubit targets: {qubit_targets}"
                    )
                noisy.append("DEPOLARIZE2", qubit_targets, self.p)

        return noisy