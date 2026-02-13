#!/usr/bin/env python3
"""Count noisy locations and test very low noise rates."""
import math, time
import numpy as np
import stim

from qectostim.codes import SteaneCode713
from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

flat_code = SteaneCode713()
hier_code = ConcatenatedCSSCode(flat_code, flat_code)

print("=" * 70)
print("CIRCUIT STATISTICS")
print("=" * 70)

for code_label, code_obj in [("Flat [[7,1,3]]", flat_code), ("Hier [[49,1,9]]", hier_code)]:
    for gad_label, gadget in [("CNOT-HTEL 0", CNOTHTeleportGadget(input_state="0")),
                                ("KnillEC 0", KnillECGadget(input_state="0"))]:
        exp = FaultTolerantGadgetExperiment(
            codes=[code_obj], gadget=gadget, noise_model=None,
            num_rounds_before=3, num_rounds_after=3,
        )
        circ = exp.to_stim()
        
        # Count operations in flattened circuit
        n_1q_gates = 0
        n_2q_gates = 0
        n_meas = 0
        n_det = 0
        n_tick = 0
        one_q = {"H", "X", "Y", "Z", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG",
                 "SQRT_Y", "SQRT_Y_DAG", "C_XYZ", "C_ZYX", "H_XY", "H_XZ", "H_YZ"}
        two_q = {"CX", "CNOT", "CZ", "CY", "ISWAP", "ISWAP_DAG", "SWAP",
                 "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "ZCX", "ZCY", "ZCZ"}
        
        for inst in circ.flattened():
            name = inst.name.upper()
            if name in one_q:
                n_1q_gates += len([t for t in inst.targets_copy() if t.is_qubit_target])
            elif name in two_q:
                n_2q_gates += len([t for t in inst.targets_copy() if t.is_qubit_target]) // 2
            elif name == "M" or name.startswith("M"):
                n_meas += len([t for t in inst.targets_copy() if t.is_qubit_target])
            elif name == "DETECTOR":
                n_det += 1
            elif name == "TICK":
                n_tick += 1
        
        n_qubits = circ.num_qubits
        print(f"\n{code_label} + {gad_label}:")
        print(f"  qubits: {n_qubits}")
        print(f"  1-qubit gates: {n_1q_gates}")
        print(f"  2-qubit gates: {n_2q_gates}")
        print(f"  measurements: {n_meas}")
        print(f"  detectors: {n_det}")
        print(f"  ticks: {n_tick}")
        print(f"  total noisy locations: {n_1q_gates + n_2q_gates}")

# Test with very low noise
print("\n" + "=" * 70)
print("VERY LOW NOISE TEST (flat Steane, PyMatching)")
print("=" * 70)

import pymatching

gadget = CNOTHTeleportGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(
    codes=[flat_code], gadget=gadget, noise_model=None,
    num_rounds_before=3, num_rounds_after=3,
)
base = exp.to_stim()

SHOTS = 10000
for p in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
    noise = StimStyleDepolarizingNoise(p=p)
    noisy = noise.apply(base)
    dem = noisy.detector_error_model(decompose_errors=True)
    sampler = noisy.compile_detector_sampler()
    raw = sampler.sample(shots=SHOTS, separate_observables=True)
    det_s = np.asarray(raw[0], dtype=bool)
    obs_s = np.asarray(raw[1], dtype=bool)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    preds = matcher.decode_batch(det_s)
    n_err = np.sum(preds[:, 0] != obs_s[:, 0])
    p_L = n_err / SHOTS
    if p_L > 0 and p_L < 1:
        a = math.log(p_L) / math.log(p)
    else:
        a = float("nan")
    print(f"  p={p:.1e}  p_L={p_L:.6f} ({n_err}/{SHOTS})  a={a:.2f}")

# Also test KnillEC
print("\n" + "=" * 70)
print("VERY LOW NOISE TEST (flat Steane, KnillEC, PyMatching)")
print("=" * 70)

gadget2 = KnillECGadget(input_state="0")
exp2 = FaultTolerantGadgetExperiment(
    codes=[flat_code], gadget=gadget2, noise_model=None,
    num_rounds_before=3, num_rounds_after=3,
)
base2 = exp2.to_stim()

for p in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
    noise = StimStyleDepolarizingNoise(p=p)
    noisy = noise.apply(base2)
    dem = noisy.detector_error_model(decompose_errors=True)
    sampler = noisy.compile_detector_sampler()
    raw = sampler.sample(shots=SHOTS, separate_observables=True)
    det_s = np.asarray(raw[0], dtype=bool)
    obs_s = np.asarray(raw[1], dtype=bool)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    preds = matcher.decode_batch(det_s)
    n_err = np.sum(preds[:, 0] != obs_s[:, 0])
    p_L = n_err / SHOTS
    if p_L > 0 and p_L < 1:
        a = math.log(p_L) / math.log(p)
    else:
        a = float("nan")
    print(f"  p={p:.1e}  p_L={p_L:.6f} ({n_err}/{SHOTS})  a={a:.2f}")

print("\nDone.")
