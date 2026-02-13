#!/usr/bin/env python3
"""Deep diagnostic: check if observable tracking is correct.

If the observable is wrong, any single error that flips the observable
but doesn't trigger any detector would be undetectable, giving α→1.

Also check: how many single-fault errors flip the observable?
"""
import numpy as np
import stim

from qectostim.codes import SteaneCode713
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()

for gad_label, gadget in [("CNOT-HTEL 0", CNOTHTeleportGadget(input_state="0")),
                           ("KnillEC 0", KnillECGadget(input_state="0"))]:
    exp = FaultTolerantGadgetExperiment(
        codes=[code], gadget=gadget, noise_model=None,
        num_rounds_before=3, num_rounds_after=3,
    )
    base = exp.to_stim()
    
    # Add very low noise for analysis
    noise = StimStyleDepolarizingNoise(p=1e-6)
    noisy = noise.apply(base)
    
    dem = noisy.detector_error_model(decompose_errors=True)
    
    print(f"\n{'='*60}")
    print(f"{gad_label}")
    print(f"{'='*60}")
    print(f"Detectors: {dem.num_detectors}")
    print(f"Observables: {dem.num_observables}")
    print(f"Error mechanisms: {dem.num_errors}")
    
    # Count errors by observable involvement
    n_det_only = 0
    n_obs_only = 0
    n_both = 0
    n_neither = 0
    max_det_weight = 0
    obs_only_errors = []
    
    for instr in dem.flattened():
        if instr.type != "error":
            continue
        targets = instr.targets_copy()
        prob = instr.args_copy()[0]
        
        has_det = any(t.is_relative_detector_id() for t in targets)
        has_obs = any(t.is_logical_observable_id() for t in targets)
        n_det = sum(1 for t in targets if t.is_relative_detector_id())
        max_det_weight = max(max_det_weight, n_det)
        
        if has_det and has_obs:
            n_both += 1
        elif has_det:
            n_det_only += 1
        elif has_obs:
            n_obs_only += 1
            obs_only_errors.append((prob, n_det, targets))
        else:
            n_neither += 1
    
    print(f"\nError classification:")
    print(f"  detector-only: {n_det_only}")
    print(f"  observable-only (UNDETECTABLE!): {n_obs_only}")
    print(f"  both det+obs: {n_both}")
    print(f"  neither: {n_neither}")
    print(f"  max detector weight: {max_det_weight}")
    
    if obs_only_errors:
        print(f"\n  ⚠️  {n_obs_only} UNDETECTABLE errors that flip observable:")
        for prob, nd, tgts in obs_only_errors[:10]:
            print(f"    prob={prob:.2e}  det_weight={nd}  targets={[str(t) for t in tgts]}")
    
    # Check the shortest error path that flips the observable
    print(f"\n  Checking has_flow...")
    try:
        # Use Stim's built-in check
        explain = dem.shortest_graphlike_error(
            ignore_ungraphlike_errors=True
        )
        print(f"  Shortest graphlike error path (weight={len(explain)})")
        for e in explain[:5]:
            print(f"    {e}")
    except Exception as ex:
        print(f"  shortest_graphlike_error failed: {ex}")

print("\nDone.")
