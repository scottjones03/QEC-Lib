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
    """Insert depolarizing noise after 1q/2q gates and before measurements.

    By default, includes noise on all gates including the final gate touching
    each qubit, plus X_ERROR before measurements. This is required for proper 
    decoding since the DEM needs error mechanisms that can flip the observable.
    
    Parameters
    ----------
    p1 : float
        Single-qubit depolarization probability after 1q Clifford gates.
    p2 : float  
        Two-qubit depolarization probability after 2q Clifford gates.
    include_final_noise : bool
        If True (default), include noise on final gates (before measurement).
    before_measure_flip : float
        Probability of X_ERROR before each measurement (default=p1). 
        This is critical for proper decoding as it creates error mechanisms
        that can flip the logical observable.
    """

    def __init__(self, p1: float = 0.0, p2: float = 0.0, include_final_noise: bool = True,
                 before_measure_flip: float = None):
        self.p1 = float(p1)
        self.p2 = float(p2)
        self.include_final_noise = include_final_noise
        self.before_measure_flip = before_measure_flip if before_measure_flip is not None else p1

    def apply(self, circuit: stim.Circuit) -> stim.Circuit:
        """Apply circuit depolarizing noise.
        
        Handles REPEAT blocks by recursively applying noise to their body.
        Also adds X_ERROR before measurements if before_measure_flip > 0.
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
            if name in {"R", "MRX", "MX", "MY", "MZ",
                        "DETECTOR", "OBSERVABLE_INCLUDE",
                        "TICK", "SHIFT_COORDS"}:
                continue
            # Skip MR for last_gate tracking but we'll handle M specially
            if name == "MR":
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
            
            name = inst.name.upper()
            
            # Add X_ERROR before ALL measurement instructions (measurement noise)
            # This includes M, MR (measure-reset), and basis-specific variants
            # CRITICAL: MR is used for syndrome measurements, so missing this
            # means no measurement errors on ancilla qubits!
            if name in {"M", "MR", "MX", "MY", "MZ", "MRX", "MRY", "MRZ"} and self.before_measure_flip > 0:
                qubit_targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
                if qubit_targets:
                    noisy.append("X_ERROR", qubit_targets, self.before_measure_flip)
            
            noisy.append(inst)

            # Extract only qubit targets.
            qubit_targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if not qubit_targets:
                continue

            # If this is the last gate touching *all* of these qubits,
            # optionally skip injecting noise here.
            is_final_touch = all(last_gate_on_qubit.get(q, -1) == idx for q in qubit_targets)
            skip_noise = is_final_touch and not self.include_final_noise

            # 1-qubit gates: add DEPOLARIZE1 on each target unless skipped.
            if name in {
                "H", "X", "Y", "Z", "S", "S_DAG",
                "SQRT_X", "SQRT_X_DAG",
                "SQRT_Y", "SQRT_Y_DAG",
            } and self.p1 > 0 and not skip_noise:
                noisy.append("DEPOLARIZE1", qubit_targets, self.p1)

            # 2-qubit gates: add DEPOLARIZE2 in pairs unless skipped.
            if name in {"CX", "CNOT", "CZ", "ISWAP", "SWAP"} and self.p2 > 0 and not skip_noise:
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