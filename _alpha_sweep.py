#!/usr/bin/env python3
"""Alpha sweep for hierarchical Steane x Steane [[49,1,9]] code.

Optimized: builds base circuit once per gadget, then varies noise.

Gadgets tested:
  - CNOTHTeleportGadget   (2 input-state combos)  [fast]
  - CSSSurgeryCNOTGadget  (4 input-state combos)  [slow]

Parameters:
  shots   = 400
  decoder = concat_mle
  p in {0.0005, 0.001, 0.0012, 0.0015, 0.0017}
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
from qectostim.decoders.concat_mle_decoder import ConcatMLEDecoder

# -- quiet the chatty loggers a bit ---------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =========================================================================
# CONFIG
# =========================================================================
SHOTS       = 400
DECODER     = "concat_mle"
P_VALUES    = [0.0005, 0.001, 0.0012, 0.0015, 0.0017]
ROUNDS_B    = 3
ROUNDS_A    = 3

# =========================================================================
# Build the [[49,1,9]] code
# =========================================================================
inner = SteaneCode713()
code  = ConcatenatedCSSCode(inner, inner)
print(f"Code: {code.name}  n={code.n}  distance={code.distance}")

# =========================================================================
# Define gadget configurations -- CNOT-HTEL first (fast), then Surgery CNOT
# =========================================================================
GadgetConfig = Tuple[str, object]   # (label, gadget instance)

CONFIGS: List[GadgetConfig] = []

# -- CNOT-H Teleportation (2 state combos) --------------------------------
for inp in ("0", "+"):
    label = f"CNOT-HTEL input={inp}"
    gadget = CNOTHTeleportGadget(input_state=inp)
    CONFIGS.append((label, gadget))

# -- Surgery CNOT (4 state combos) ----------------------------------------
for ctrl in ("0", "+"):
    for tgt in ("0", "+"):
        label = f"SurgeryCNOT ctrl={ctrl} tgt={tgt}"
        gadget = CSSSurgeryCNOTGadget(control_state=ctrl, target_state=tgt)
        CONFIGS.append((label, gadget))


# =========================================================================
# Helper functions
# =========================================================================
def compute_alpha(p_L: float, p: float) -> float:
    """a = log(p_L) / log(p).  Returns NaN on bad inputs."""
    if p_L <= 0 or p_L >= 1 or p <= 0 or p >= 1:
        return float("nan")
    return math.log(p_L) / math.log(p)


def build_dem(circuit: stim.Circuit):
    """Build DEM with fallback for hypergraph errors. Returns (dem, is_hyper)."""
    # 1. Try standard decomposition
    try:
        return circuit.detector_error_model(decompose_errors=True), False
    except ValueError:
        pass
    # 2. Try ignoring decomposition failures
    try:
        dem = circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True,
        )
        return dem, True
    except ValueError:
        pass
    # 3. Last resort: non-decomposed DEM (hyperedges, needs BP-OSD/tesseract)
    dem = circuit.detector_error_model(decompose_errors=False)
    return dem, True


def run_circuit_level(noisy_circuit: stim.Circuit, shots: int) -> float:
    """Circuit-level sampling + decoding.

    This avoids DEM decomposition issues by:
      1. Building a DEM for the *decoder* (with decompose_errors=False fallback)
      2. Sampling detector+observable outcomes from the *circuit* directly
      3. Decoding with ConcatMLEDecoder
    """
    # Build DEM (only for decoder construction, not for sampling)
    dem, is_hyper = build_dem(noisy_circuit)
    logger.info("  DEM: %d det, %d err, %d obs, hyper=%s",
                dem.num_detectors, dem.num_errors, dem.num_observables, is_hyper)

    # Circuit-level sampling: sample det+obs directly from the noisy circuit
    sampler = noisy_circuit.compile_detector_sampler()
    raw = sampler.sample(shots=shots, separate_observables=True)
    det_samples = np.asarray(raw[0], dtype=np.uint8)
    obs_samples = np.asarray(raw[1], dtype=np.uint8)

    # Decode — force bposd backend (handles hypergraph DEMs much better
    # than tesseract, which is designed for 2D topological codes)
    decoder = ConcatMLEDecoder(dem, backend="bposd")
    logger.info("  decoder backend: %s", decoder.active_backend)
    corrections = decoder.decode_batch(det_samples)
    corrections = np.asarray(corrections, dtype=np.uint8)
    if corrections.ndim == 1:
        corrections = corrections.reshape(-1, dem.num_observables)
    true_log = obs_samples[:, 0]
    pred_log = corrections[:, 0]
    logical_errors = (pred_log ^ true_log).astype(np.uint8)
    return float(logical_errors.mean())


# =========================================================================
# Main sweep -- build base circuit ONCE per gadget, then vary noise
# =========================================================================
results: Dict[str, Dict[float, Dict]] = {}   # label -> {p -> result_dict}

total_configs = len(CONFIGS) * len(P_VALUES)
done = 0

for label, gadget in CONFIGS:
    results[label] = {}

    # Build noiseless base circuit once (to_stim applies noise_model internally)
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

    # Print intermediate results after each gadget finishes
    print(f"\n--- {label} ---")
    for p in P_VALUES:
        d = results[label][p]
        if d["error"]:
            print(f"  p={p:.4f}  ERR: {d['error'][:80]}")
        else:
            print(f"  p={p:.4f}  p_L={d['p_L']:.5f}  a={d['alpha']:.2f}  ({d['time']:.1f}s)")
    sys.stdout.flush()

# =========================================================================
# Pretty-print summary table
# =========================================================================
print("\n" + "=" * 90)
print(f"ALPHA SWEEP RESULTS   (shots={SHOTS}, decoder=concat_mle)")
print("=" * 90)
header = f"{'Gadget':<38}"
for p in P_VALUES:
    header += f"  p={p:.4f}"
print(header)
print("-" * 90)

for label in results:
    row = f"{label:<38}"
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

print("-" * 90)

# Average alpha per gadget family
families = {"SurgeryCNOT": [], "CNOT-HTEL": []}
for label in results:
    fam = "SurgeryCNOT" if "SurgeryCNOT" in label else "CNOT-HTEL"
    for p in P_VALUES:
        a = results[label][p]["alpha"]
        if not math.isnan(a):
            families[fam].append(a)

print("\nAverage alpha per family:")
for fam, vals in families.items():
    if vals:
        mean_a = np.mean(vals)
        std_a  = np.std(vals)
        print(f"  {fam:<20}  a = {mean_a:.2f} +/- {std_a:.2f}  (n={len(vals)} pts)")
    else:
        print(f"  {fam:<20}  NO DATA")

# Also print per-p error rate table
print("\n\nLogical error rates (p_L):")
print(f"{'Gadget':<38}", end="")
for p in P_VALUES:
    print(f"  p={p:.4f}", end="")
print()
print("-" * 90)
for label in results:
    row = f"{label:<38}"
    for p in P_VALUES:
        d = results[label][p]
        if d["error"]:
            row += f"  {'ERR':>10}"
        else:
            row += f"  {d['p_L']:>10.5f}"
    print(row)
print("-" * 90)
print("Done.")
