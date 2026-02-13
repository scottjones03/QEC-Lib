#!/usr/bin/env python3
"""Quick check: how many detectors in circuit vs DEM for hierarchical."""
import stim
from qectostim.codes import SteaneCode713
from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

inner = SteaneCode713()
code = ConcatenatedCSSCode(inner, inner)
gadget = CNOTHTeleportGadget(input_state="0")
exp = FaultTolerantGadgetExperiment(codes=[code], gadget=gadget, noise_model=None, num_rounds_before=3, num_rounds_after=3)
base = exp.to_stim()

det_count = sum(1 for inst in base.flattened() if inst.name == "DETECTOR")
print(f"Circuit DETECTOR instructions: {det_count}")

dem_nl = base.detector_error_model(decompose_errors=False)
print(f"Noiseless DEM: num_detectors={dem_nl.num_detectors}")

noise = StimStyleDepolarizingNoise(p=1e-4)
noisy = noise.apply(base)
dem_n = noisy.detector_error_model(decompose_errors=False)
print(f"Noisy DEM (decompose=False): num_detectors={dem_n.num_detectors}, num_errors={dem_n.num_errors}")

dem_ig = noisy.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
print(f"Noisy DEM (ignore_failures): num_detectors={dem_ig.num_detectors}, num_errors={dem_ig.num_errors}")
