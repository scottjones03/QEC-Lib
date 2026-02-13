#!/usr/bin/env python3
"""Quick circuit-level distance check for all KnillEC and CNOT-HTEL configs.
Checks: undetectable errors, shortest graphlike error distance."""
import stim
from qectostim.codes import SteaneCode713
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()
noise = StimStyleDepolarizingNoise(p=1e-8)

configs = [
    ("CNOT-HTEL 0", CNOTHTeleportGadget(input_state="0")),
    ("CNOT-HTEL +", CNOTHTeleportGadget(input_state="+")),
    ("KnillEC 0", KnillECGadget(input_state="0")),
    ("KnillEC +", KnillECGadget(input_state="+")),
]

print(f"{'Config':<18} {'n_det':<8} {'n_err':<8} {'undet':<8} {'shortest':<10}")
print("-" * 55)

for label, gadget in configs:
    exp = FaultTolerantGadgetExperiment(
        codes=[code], gadget=gadget, noise_model=noise,
        num_rounds_before=3, num_rounds_after=3,
    )
    circ = exp.to_stim()
    dem = circ.detector_error_model(decompose_errors=False)
    
    n_det = circ.num_detectors
    n_err = sum(1 for e in dem if e.type == "error")
    
    undet = 0
    shortest = 999
    for error in dem:
        if error.type != "error":
            continue
        targets = error.targets_copy()
        dets = [t for t in targets if t.is_relative_detector_id()]
        obs = [t for t in targets if t.is_logical_observable_id()]
        
        if len(obs) > 0:
            if len(dets) == 0:
                undet += 1
            else:
                shortest = min(shortest, len(dets))
    
    if shortest == 999:
        shortest = "N/A"
    
    print(f"{label:<18} {n_det:<8} {n_err:<8} {undet:<8} {str(shortest):<10}")
