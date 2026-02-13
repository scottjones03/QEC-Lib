#!/usr/bin/env python3
"""Sanity check: alpha for Stim's reference repetition code and surface code."""
import math
import numpy as np
import stim
import pymatching

SHOTS = 10000

def test_alpha(label, circuit, p_values):
    print(f"\n{label}")
    print(f"  detectors: {sum(1 for i in circuit.flattened() if i.name=='DETECTOR')}")
    for p in p_values:
        noisy = circuit.copy()
        # This circuit already has noise parameterized, 
        # so we need a different approach
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

# Test Stim's reference surface code
print("=" * 60)
print("STIM REFERENCE CIRCUITS (sanity check)")
print("=" * 60)

for p in [1e-4, 5e-4, 1e-3]:
    # Surface code d=3, rounds=3
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=p,
    )
    dem = circ.detector_error_model(decompose_errors=True)
    sampler = circ.compile_detector_sampler()
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
    n_det = sum(1 for i in circ.flattened() if i.name == "DETECTOR")
    print(f"Surface d=3 r=3: p={p:.1e}  p_L={p_L:.6f}  a={a:.2f}  ({n_det} det)")

print()

for p in [1e-4, 5e-4, 1e-3]:
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=5,
        rounds=5,
        after_clifford_depolarization=p,
    )
    dem = circ.detector_error_model(decompose_errors=True)
    sampler = circ.compile_detector_sampler()
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
    n_det = sum(1 for i in circ.flattened() if i.name == "DETECTOR")
    print(f"Surface d=5 r=5: p={p:.1e}  p_L={p_L:.6f}  a={a:.2f}  ({n_det} det)")

# Now test our Steane code with the same noise model but MEMORY ONLY
# (no teleportation gadget) to isolate the gadget effect
print()
print("=" * 60)
print("STEANE CODE MEMORY (no gadget, direct Stim construction)")
print("=" * 60)

# Build a minimal Steane [[7,1,3]] memory circuit manually using Stim
# to verify that the code itself has proper α=2 scaling
from qectostim.codes import SteaneCode713
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()

# Try building a circuit with just memory rounds (identity gadget?)
# Actually, there's no identity gadget. Let's check if we can use
# the experiment with just num_rounds_before and no gadget at all.
# We need a "trivial" gadget — let's try TransversalCNOT with rounds only.

# Actually, let's just build a KnillEC gadget with the same noise model
# and see if the issue is the noise model or the circuit
from qectostim.gadgets.knill_ec import KnillECGadget

gad = KnillECGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(
    codes=[code], gadget=gad, noise_model=None,
    num_rounds_before=3, num_rounds_after=3,
)
base = exp.to_stim()

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
    print(f"Steane KnillEC: p={p:.1e}  p_L={p_L:.6f}  a={a:.2f}")

print("\nDone.")
