#!/usr/bin/env python3
"""Final check: compare all gadgets after fixes."""
import stim
from qectostim.codes import SteaneCode713
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()
noise = StimStyleDepolarizingNoise(p=1e-8)

configs = [
    ("KnillEC 0", KnillECGadget(input_state="0")),
    ("KnillEC +", KnillECGadget(input_state="+")),
    ("CNOT-HTEL 0", CNOTHTeleportGadget(input_state="0")),
    ("CNOT-HTEL +", CNOTHTeleportGadget(input_state="+")),
]

for label, gadget in configs:
    exp = FaultTolerantGadgetExperiment(
        codes=[code], gadget=gadget, noise_model=None,
        num_rounds_before=3, num_rounds_after=3,
    )
    base = exp.to_stim()
    noisy = noise.apply(base)
    dem = noisy.detector_error_model(decompose_errors=False)
    
    undet = 0
    for instr in dem.flattened():
        if instr.type != "error":
            continue
        targets = instr.targets_copy()
        has_det = any(t.is_relative_detector_id() for t in targets)
        has_obs = any(t.is_logical_observable_id() for t in targets)
        if has_obs and not has_det:
            undet += 1
    
    try:
        sp = dem.shortest_graphlike_error(ignore_ungraphlike_errors=True)
        sp_len = len(sp)
    except:
        sp_len = "ERR"
    
    noiseless = base.detector_error_model()
    
    print(f"{label:15s}: {dem.num_detectors:3d} det, {dem.num_errors:5d} err, "
          f"{undet} undet, shortest={sp_len}, noiseless_err={noiseless.num_errors}")

print("\nDone.")
