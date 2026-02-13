#!/usr/bin/env python3
"""Test DEM decomposition modes for hierarchical codes with noise."""
import logging
import stim
from qectostim.codes import SteaneCode713
from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.noise.models import StimStyleDepolarizingNoise

logging.basicConfig(level=logging.WARNING)

steane = SteaneCode713()
concat = ConcatenatedCSSCode(steane, steane)

configs = [
    ("CNOT-HTEL 0", CNOTHTeleportGadget(input_state="0"), [concat]),
    ("Surgery 0", CSSSurgeryCNOTGadget(control_state="0"), [concat]*3),
]

for p in [0.001, 0.0005]:
    noise = StimStyleDepolarizingNoise(p=p)
    print(f"\n{'='*60}")
    print(f"  Noise p={p}")
    print(f"{'='*60}")
    
    for label, gadget, codes_list in configs:
        exp = FaultTolerantGadgetExperiment(
            codes=codes_list, gadget=gadget, noise_model=noise,
            num_rounds_before=2, num_rounds_after=2, d_inner=3,
        )
        circuit = exp.to_stim()
        n_det = circuit.num_detectors
        
        modes = [
            ("decompose", dict(decompose_errors=True)),
            ("decompose+approx", dict(decompose_errors=True, approximate_disjoint_errors=True)),
            ("ignore_failures", dict(decompose_errors=True, ignore_decomposition_failures=True)),
            ("ignore+approx", dict(decompose_errors=True, ignore_decomposition_failures=True, approximate_disjoint_errors=True)),
            ("no_decompose", dict(decompose_errors=False)),
            ("no_decompose+approx", dict(decompose_errors=False, approximate_disjoint_errors=True)),
        ]
        
        for mode_label, kw in modes:
            try:
                dem = circuit.detector_error_model(**kw)
                print(f"  [{label}] DEM({mode_label}): OK → {dem.num_detectors} det, {dem.num_errors} err")
                break
            except Exception as e:
                msg = str(e)[:80]
                print(f"  [{label}] DEM({mode_label}): FAIL — {msg}")
