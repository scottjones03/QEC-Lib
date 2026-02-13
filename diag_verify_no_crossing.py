#!/usr/bin/env python3
"""Verify: no-crossing-detector approach works with noise for hierarchical codes."""
import logging
import stim
from qectostim.codes import SteaneCode713
from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget, CZHTeleportGadget
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.noise.models import StimStyleDepolarizingNoise

logging.basicConfig(level=logging.WARNING)

steane = SteaneCode713()
concat = ConcatenatedCSSCode(steane, steane)
noise = StimStyleDepolarizingNoise(p=0.001)

configs = [
    ("CNOT-HTEL 0", CNOTHTeleportGadget(input_state="0")),
    ("CNOT-HTEL +", CNOTHTeleportGadget(input_state="+")),
    ("CZ-HTEL 0", CZHTeleportGadget(input_state="0")),
    ("CZ-HTEL +", CZHTeleportGadget(input_state="+")),
    ("KnillEC 0", KnillECGadget(input_state="0")),
    ("KnillEC +", KnillECGadget(input_state="+")),
    ("Surgery CNOT 0", CSSSurgeryCNOTGadget(control_state="0")),
    ("Surgery CNOT +", CSSSurgeryCNOTGadget(control_state="+")),
]

print("Testing with crossing detectors disabled (return None) and noise p=0.001\n")

for label, gadget in configs:
    # Disable crossing detectors
    gadget.get_crossing_detector_config = lambda: None
    
    try:
        exp = FaultTolerantGadgetExperiment(
            codes=[concat], gadget=gadget, noise_model=noise,
            num_rounds_before=2, num_rounds_after=2, d_inner=3,
        )
        circuit = exp.to_stim()
        n_det = circuit.num_detectors
        n_obs = circuit.num_observables
        
        # Try DEM
        dem_ok = False
        dem_mode = None
        for mode, kw in [
            ("decompose", dict(decompose_errors=True, approximate_disjoint_errors=True)),
            ("ignore", dict(decompose_errors=True, ignore_decomposition_failures=True, approximate_disjoint_errors=True)),
            ("no_decompose", dict(decompose_errors=False, approximate_disjoint_errors=True)),
        ]:
            try:
                dem = circuit.detector_error_model(**kw)
                print(f"  [{label}] {n_det} det, {n_obs} obs → DEM({mode}) OK: {dem.num_detectors} det, {dem.num_errors} err")
                dem_ok = True
                dem_mode = mode
                break
            except Exception as e:
                continue
        
        if not dem_ok:
            print(f"  [{label}] {n_det} det, {n_obs} obs → ALL DEM modes FAILED")
    except Exception as e:
        print(f"  [{label}] CIRCUIT BUILD FAILED: {str(e)[:100]}")
