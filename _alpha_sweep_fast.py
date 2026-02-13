#!/usr/bin/env python3
"""Fast alpha sweep for hierarchical Steane x Steane [[49,1,9]] code.

Optimized: reduced osd_order (10 → fast), fewer shots (100), CNOT-HTEL only.
This gives a quick directional estimate of alpha.
"""
from __future__ import annotations

import logging
import math
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import stim

# -- codes -----------------------------------------------------------------
from qectostim.codes import SteaneCode713
from qectostim.codes.composite.concatenated import ConcatenatedCSSCode

# -- gadgets ---------------------------------------------------------------
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget

# -- experiment + noise + decoder ------------------------------------------
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

# -- quiet the chatty loggers a bit ---------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =========================================================================
# CONFIG
# =========================================================================
SHOTS       = 100
P_VALUES    = [0.0005, 0.001, 0.002]
ROUNDS_B    = 3
ROUNDS_A    = 3

# =========================================================================
# Build the [[49,1,9]] code
# =========================================================================
inner = SteaneCode713()
code  = ConcatenatedCSSCode(inner, inner)
print(f"Code: {code.name}  n={code.n}  distance={code.distance}")

# =========================================================================
# Define gadget configurations -- CNOT-HTEL only for speed
# =========================================================================
GadgetConfig = Tuple[str, object]

CONFIGS: List[GadgetConfig] = []

for inp in ("0", "+"):
    label = f"CNOT-HTEL input={inp}"
    gadget = CNOTHTeleportGadget(input_state=inp)
    CONFIGS.append((label, gadget))


# =========================================================================
# Helper functions
# =========================================================================
def compute_alpha(p_L: float, p: float) -> float:
    """a = log(p_L) / log(p)."""
    if p_L <= 0 or p_L >= 1 or p <= 0 or p >= 1:
        return float("nan")
    return math.log(p_L) / math.log(p)


def build_dem(circuit: stim.Circuit):
    """Build DEM with fallback for hypergraph errors. Returns (dem, is_hyper)."""
    try:
        return circuit.detector_error_model(decompose_errors=True), False
    except ValueError:
        pass
    try:
        dem = circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True,
        )
        return dem, True
    except ValueError:
        pass
    dem = circuit.detector_error_model(decompose_errors=False)
    return dem, True


def run_circuit_level(noisy_circuit: stim.Circuit, shots: int) -> float:
    """Circuit-level sampling + decoding with fast BP-OSD."""
    from stimbposd.dem_to_matrices import detector_error_model_to_check_matrices
    from ldpc import BpOsdDecoder as LdpcBpOsd

    # Build DEM
    dem, is_hyper = build_dem(noisy_circuit)
    logger.info("  DEM: %d det, %d err, %d obs, hyper=%s",
                dem.num_detectors, dem.num_errors, dem.num_observables, is_hyper)

    # Sample
    sampler = noisy_circuit.compile_detector_sampler()
    raw = sampler.sample(shots=shots, separate_observables=True)
    det_samples = np.asarray(raw[0], dtype=np.uint8)
    obs_samples = np.asarray(raw[1], dtype=np.uint8)

    # Build fast BP-OSD decoder directly (osd_order=10 instead of 200)
    matrices = detector_error_model_to_check_matrices(
        dem, allow_undecomposed_hyperedges=True
    )
    H = matrices.check_matrix

    decoder = LdpcBpOsd(
        H,
        max_iter=20,          # fast: 20 BP iters (not 100)
        bp_method="product_sum",
        error_channel=list(matrices.priors),
        osd_order=10,         # fast: order 10 (not 200)
        osd_method="osd_cs",
        input_vector_type="syndrome",
    )

    obs_matrix = matrices.observables_matrix
    obs_dense = (
        np.asarray(obs_matrix.todense(), dtype=np.uint8)
        if hasattr(obs_matrix, "todense")
        else np.asarray(obs_matrix, dtype=np.uint8)
    )

    # Decode shot by shot
    n_err = 0
    for i in range(det_samples.shape[0]):
        err = np.asarray(decoder.decode(det_samples[i]), dtype=np.uint8)
        pred = (obs_dense @ err) % 2
        if pred[0] != obs_samples[i, 0]:
            n_err += 1

    return n_err / shots


# =========================================================================
# Main sweep
# =========================================================================
results: Dict[str, Dict[float, Dict]] = {}

total_configs = len(CONFIGS) * len(P_VALUES)
done = 0

for label, gadget in CONFIGS:
    results[label] = {}

    logger.info("Building base circuit for %s ...", label)
    t_build = time.time()
    exp = FaultTolerantGadgetExperiment(
        codes=[code],
        gadget=gadget,
        noise_model=None,
        num_rounds_before=ROUNDS_B,
        num_rounds_after=ROUNDS_A,
    )
    base_circuit = exp.to_stim()
    build_time = time.time() - t_build
    logger.info("  base circuit: %d instr (%.1fs)", len(base_circuit), build_time)

    for p in P_VALUES:
        done += 1
        tag = f"[{done}/{total_configs}]"
        logger.info("%s  %s  p=%.4f  -- running %d shots ...", tag, label, p, SHOTS)
        t0 = time.time()
        try:
            noise = StimStyleDepolarizingNoise(p=p)
            noisy = noise.apply(base_circuit)
            p_L = run_circuit_level(noisy, SHOTS)
            elapsed = time.time() - t0
            a = compute_alpha(p_L, p)
            results[label][p] = {"p_L": p_L, "alpha": a, "time": elapsed, "error": None}
            logger.info("%s  p_L=%.5f  a=%.2f  (%.1fs)", tag, p_L, a, elapsed)
        except Exception as exc:
            elapsed = time.time() - t0
            results[label][p] = {"p_L": float("nan"), "alpha": float("nan"),
                                 "time": elapsed, "error": str(exc)[:120]}
            logger.error("%s  FAILED (%.1fs): %s", tag, elapsed, exc)

    print(f"\n--- {label} ---")
    for p in P_VALUES:
        d = results[label][p]
        if d["error"]:
            print(f"  p={p:.4f}  ERR: {d['error'][:80]}")
        else:
            print(f"  p={p:.4f}  p_L={d['p_L']:.5f}  a={d['alpha']:.2f}  ({d['time']:.1f}s)")
    sys.stdout.flush()

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print(f"FAST ALPHA SWEEP   (shots={SHOTS}, osd_order=10)")
print("=" * 70)
header = f"{'Gadget':<30}"
for p in P_VALUES:
    header += f"  p={p:.4f}"
print(header)
print("-" * 70)

for label in results:
    row = f"{label:<30}"
    for p in P_VALUES:
        d = results[label][p]
        if d["error"]:
            row += f"  {'ERR':>10}"
        elif math.isnan(d["alpha"]):
            pL_str = f'{d["p_L"]:.4f}' if not math.isnan(d["p_L"]) else "NaN"
            row += f"  {pL_str:>10}"
        else:
            row += f"  {d['alpha']:>10.2f}"
    print(row)

print("-" * 70)

all_alphas = []
for label in results:
    for p in P_VALUES:
        a = results[label][p]["alpha"]
        if not math.isnan(a):
            all_alphas.append(a)

if all_alphas:
    print(f"\nMean alpha: {np.mean(all_alphas):.2f} +/- {np.std(all_alphas):.2f}")

print("\n\nLogical error rates (p_L):")
print(f"{'Gadget':<30}", end="")
for p in P_VALUES:
    print(f"  p={p:.4f}", end="")
print()
print("-" * 70)
for label in results:
    row = f"{label:<30}"
    for p in P_VALUES:
        d = results[label][p]
        if d["error"]:
            row += f"  {'ERR':>10}"
        else:
            row += f"  {d['p_L']:>10.5f}"
    print(row)
print("-" * 70)
print("Done.")
