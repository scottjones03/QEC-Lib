#!/usr/bin/env python3
"""Alpha comparison: with and without d_inner=3.
Uses flat Steane for quick comparison (PyMatching).
"""
import math
import numpy as np
import stim
import pymatching

from qectostim.codes import SteaneCode713
from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

SHOTS = 10000

# Flat Steane first as reference
code = SteaneCode713()
gadget = CNOTHTeleportGadget(input_state="0")

configs = [
    ("Flat r=3, no d_inner", dict(codes=[code], gadget=gadget, noise_model=None,
                                    num_rounds_before=3, num_rounds_after=3)),
    ("Flat r=2, d_inner=3", dict(codes=[code], gadget=gadget, noise_model=None,
                                   num_rounds_before=2, num_rounds_after=2, d_inner=3)),
]

# Also hierarchical
inner = SteaneCode713()
hcode = ConcatenatedCSSCode(inner, inner)
gadget2 = CNOTHTeleportGadget(input_state="0")

configs.extend([
    ("Hier r=3, no d_inner", dict(codes=[hcode], gadget=CNOTHTeleportGadget(input_state="0"),
                                    noise_model=None,
                                    num_rounds_before=3, num_rounds_after=3)),
    ("Hier r=2, d_inner=3", dict(codes=[hcode], gadget=CNOTHTeleportGadget(input_state="0"),
                                   noise_model=None,
                                   num_rounds_before=2, num_rounds_after=2, d_inner=3)),
])

for label, kw in configs:
    exp = FaultTolerantGadgetExperiment(**kw)
    base = exp.to_stim()
    n_det = base.num_detectors
    
    is_hier = "Hier" in label
    print(f"\n{label}: {n_det} detectors")
    
    for p in [1e-5, 1e-4, 5e-4]:
        noise = StimStyleDepolarizingNoise(p=p)
        noisy = noise.apply(base)
        
        if not is_hier:
            # Flat: use PyMatching
            dem = noisy.detector_error_model(decompose_errors=True)
            sampler = noisy.compile_detector_sampler()
            raw = sampler.sample(shots=SHOTS, separate_observables=True)
            det_s = np.asarray(raw[0], dtype=bool)
            obs_s = np.asarray(raw[1], dtype=bool)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            preds = matcher.decode_batch(det_s)
            n_err = np.sum(preds[:, 0] != obs_s[:, 0])
        else:
            # Hierarchical: use BP-OSD  
            from stimbposd.dem_to_matrices import detector_error_model_to_check_matrices
            from ldpc import BpOsdDecoder as LdpcBpOsd
            
            dem = noisy.detector_error_model(decompose_errors=False)
            sampler = noisy.compile_detector_sampler()
            raw = sampler.sample(shots=SHOTS, separate_observables=True)
            det_s = np.asarray(raw[0], dtype=np.uint8)
            obs_s = np.asarray(raw[1], dtype=np.uint8)
            
            matrices = detector_error_model_to_check_matrices(
                dem, allow_undecomposed_hyperedges=True
            )
            H = matrices.check_matrix
            decoder = LdpcBpOsd(
                H, max_iter=30, bp_method="product_sum",
                error_channel=list(matrices.priors),
                osd_order=15, osd_method="osd_cs",
                input_vector_type="syndrome",
            )
            obs_matrix = matrices.observables_matrix
            obs_dense = (
                np.asarray(obs_matrix.todense(), dtype=np.uint8)
                if hasattr(obs_matrix, "todense")
                else np.asarray(obs_matrix, dtype=np.uint8)
            )
            n_err = 0
            for i in range(min(SHOTS, 200)):  # Limit to 200 for speed
                err = np.asarray(decoder.decode(det_s[i]), dtype=np.uint8)
                pred = (obs_dense @ err) % 2
                if pred[0] != obs_s[i, 0]:
                    n_err += 1
            # Adjust for sub-sampling
            shots_used = min(SHOTS, 200)
        
        shots_used = SHOTS if not is_hier else min(SHOTS, 200)
        p_L = n_err / shots_used
        a = math.log(p_L) / math.log(p) if 0 < p_L < 1 else float("nan")
        print(f"  p={p:.1e}  p_L={p_L:.5f} ({n_err}/{shots_used})  a={a:.2f}  DEM:{dem.num_detectors}det,{dem.num_errors}err")

print("\nDone.")
