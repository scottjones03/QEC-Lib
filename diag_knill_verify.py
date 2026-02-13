#!/usr/bin/env python3
"""Verify the KnillEC boundary detector fix."""
import stim
from qectostim.codes import SteaneCode713
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()

for inp in ['0', '+']:
    gadget = KnillECGadget(input_state=inp)
    exp = FaultTolerantGadgetExperiment(
        codes=[code], gadget=gadget, noise_model=None,
        num_rounds_before=3, num_rounds_after=3,
    )
    base = exp.to_stim()
    noise = StimStyleDepolarizingNoise(p=1e-8)
    noisy = noise.apply(base)
    dem = noisy.detector_error_model(decompose_errors=False)
    
    undet = 0
    for instr in dem.flattened():
        if instr.type != 'error':
            continue
        targets = instr.targets_copy()
        has_det = any(t.is_relative_detector_id() for t in targets)
        has_obs = any(t.is_logical_observable_id() for t in targets)
        if has_obs and not has_det:
            undet += 1
    
    try:
        sp = dem.shortest_graphlike_error(ignore_ungraphlike_errors=True)
        sp_len = len(sp)
    except Exception as e:
        sp_len = f'ERROR: {e}'
    
    print(f'KnillEC input={inp}: {dem.num_detectors} det, {dem.num_errors} err, {undet} undetectable, shortest_path={sp_len}')

    # Also check noiseless DEM for non-deterministic detectors
    noiseless_dem = base.detector_error_model()
    n_noiseless_err = noiseless_dem.num_errors
    print(f'  Noiseless DEM: {noiseless_dem.num_detectors} det, {n_noiseless_err} errors (should be 0)')

print("\nDone.")
