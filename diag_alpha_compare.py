#!/usr/bin/env python3
"""Quick alpha comparison: flat [[7,1,3]] vs hierarchical [[49,1,9]]
CNOT-HTEL input=0 only, 3 p-values, 100 shots.
"""
import logging, math, sys, time
import numpy as np
import stim

from qectostim.codes import SteaneCode713
from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SHOTS = 200
P_VALUES = [0.0005, 0.001, 0.002, 0.005]

def compute_alpha(p_L, p):
    if p_L <= 0 or p_L >= 1 or p <= 0 or p >= 1:
        return float("nan")
    return math.log(p_L) / math.log(p)

def run_with_pymatching(noisy_circuit, shots):
    """Use PyMatching for flat codes (works with decomposed DEM)."""
    import pymatching
    dem = noisy_circuit.detector_error_model(decompose_errors=True)
    logger.info("  DEM: %d det, %d err (decomposed)", dem.num_detectors, dem.num_errors)
    sampler = noisy_circuit.compile_detector_sampler()
    raw = sampler.sample(shots=shots, separate_observables=True)
    det_samples = np.asarray(raw[0], dtype=bool)
    obs_samples = np.asarray(raw[1], dtype=bool)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    predictions = matcher.decode_batch(det_samples)
    n_err = np.sum(predictions[:, 0] != obs_samples[:, 0])
    return n_err / shots

def run_with_bposd(noisy_circuit, shots, osd_order=10):
    """Use BP-OSD for hierarchical codes (hyperedge DEM)."""
    from stimbposd.dem_to_matrices import detector_error_model_to_check_matrices
    from ldpc import BpOsdDecoder as LdpcBpOsd
    
    # Use decompose_errors=False (cleaner for BP-OSD)
    dem = noisy_circuit.detector_error_model(decompose_errors=False)
    logger.info("  DEM: %d det, %d err (non-decomposed)", dem.num_detectors, dem.num_errors)
    
    sampler = noisy_circuit.compile_detector_sampler()
    raw = sampler.sample(shots=shots, separate_observables=True)
    det_samples = np.asarray(raw[0], dtype=np.uint8)
    obs_samples = np.asarray(raw[1], dtype=np.uint8)
    
    matrices = detector_error_model_to_check_matrices(
        dem, allow_undecomposed_hyperedges=True
    )
    H = matrices.check_matrix
    decoder = LdpcBpOsd(
        H, max_iter=20, bp_method="product_sum",
        error_channel=list(matrices.priors),
        osd_order=osd_order, osd_method="osd_cs",
        input_vector_type="syndrome",
    )
    obs_matrix = matrices.observables_matrix
    obs_dense = (
        np.asarray(obs_matrix.todense(), dtype=np.uint8)
        if hasattr(obs_matrix, "todense")
        else np.asarray(obs_matrix, dtype=np.uint8)
    )
    
    n_err = 0
    for i in range(det_samples.shape[0]):
        err = np.asarray(decoder.decode(det_samples[i]), dtype=np.uint8)
        pred = (obs_dense @ err) % 2
        if pred[0] != obs_samples[i, 0]:
            n_err += 1
    return n_err / shots


# =========================================================================
# Test flat Steane [[7,1,3]]
# =========================================================================
print("=" * 60)
print("FLAT Steane [[7,1,3]] — CNOT-HTEL input=0")
print("=" * 60)

flat_code = SteaneCode713()
gadget = CNOTHTeleportGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(
    codes=[flat_code], gadget=gadget, noise_model=None,
    num_rounds_before=3, num_rounds_after=3,
)
base_flat = exp.to_stim()
det_count = sum(1 for inst in base_flat.flattened() if inst.name == "DETECTOR")
print(f"Circuit: {det_count} detectors")

for p in P_VALUES:
    noise = StimStyleDepolarizingNoise(p=p)
    noisy = noise.apply(base_flat)
    t0 = time.time()
    try:
        p_L = run_with_pymatching(noisy, SHOTS)
        a = compute_alpha(p_L, p)
        print(f"  p={p:.4f}  p_L={p_L:.5f}  a={a:.2f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  p={p:.4f}  ERR: {e}")

# =========================================================================
# Test hierarchical Steane×Steane [[49,1,9]]
# =========================================================================
print("\n" + "=" * 60)
print("HIERARCHICAL [[49,1,9]] — CNOT-HTEL input=0")
print("=" * 60)

inner = SteaneCode713()
hier_code = ConcatenatedCSSCode(inner, inner)
gadget2 = CNOTHTeleportGadget(input_state="0")
exp2 = FaultTolerantGadgetExperiment(
    codes=[hier_code], gadget=gadget2, noise_model=None,
    num_rounds_before=3, num_rounds_after=3,
)
base_hier = exp2.to_stim()
det_count2 = sum(1 for inst in base_hier.flattened() if inst.name == "DETECTOR")
print(f"Circuit: {det_count2} detectors")

for p in P_VALUES:
    noise = StimStyleDepolarizingNoise(p=p)
    noisy = noise.apply(base_hier)
    t0 = time.time()
    try:
        p_L = run_with_bposd(noisy, SHOTS, osd_order=10)
        a = compute_alpha(p_L, p)
        print(f"  p={p:.4f}  p_L={p_L:.5f}  a={a:.2f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  p={p:.4f}  ERR: {e}")

# =========================================================================
# Also try hierarchical with PyMatching (ignore_decomposition_failures)
# =========================================================================
print("\n" + "=" * 60)
print("HIERARCHICAL [[49,1,9]] — CNOT-HTEL input=0 (PyMatching, approx)")
print("=" * 60)

for p in P_VALUES:
    noise = StimStyleDepolarizingNoise(p=p)
    noisy = noise.apply(base_hier)
    t0 = time.time()
    try:
        # Build DEM with ignore_decomposition_failures
        dem = noisy.detector_error_model(
            decompose_errors=True, ignore_decomposition_failures=True
        )
        logger.info("  DEM: %d det, %d err (ignore_failures)", dem.num_detectors, dem.num_errors)
        
        import pymatching
        sampler = noisy.compile_detector_sampler()
        raw = sampler.sample(shots=SHOTS, separate_observables=True)
        det_samples = np.asarray(raw[0], dtype=bool)
        obs_samples = np.asarray(raw[1], dtype=bool)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        predictions = matcher.decode_batch(det_samples)
        n_err = np.sum(predictions[:, 0] != obs_samples[:, 0])
        p_L = n_err / SHOTS
        a = compute_alpha(p_L, p)
        print(f"  p={p:.4f}  p_L={p_L:.5f}  a={a:.2f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  p={p:.4f}  ERR: {e}")

print("\nDone.")
