#!/usr/bin/env python3
"""Deep investigation of the KnillEC undetectable weight-1 error.

Finding: KnillEC input=0 has 1 undetectable error (L0 only, no detectors)
with prob=6.533e-08 at p=1e-8.  This is a CRITICAL issue.
"""
import numpy as np
import stim

from qectostim.codes import SteaneCode713
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()

for inp in ["0", "+"]:
    gadget = KnillECGadget(input_state=inp)
    exp = FaultTolerantGadgetExperiment(
        codes=[code], gadget=gadget, noise_model=None,
        num_rounds_before=3, num_rounds_after=3,
    )
    base = exp.to_stim()

    noise = StimStyleDepolarizingNoise(p=1e-8)
    noisy = noise.apply(base)

    # Use decompose_errors=False to avoid decomposition failures
    dem = noisy.detector_error_model(decompose_errors=False)

    print(f"\n{'='*70}")
    print(f"KnillEC input={inp}")
    print(f"{'='*70}")
    print(f"Circuit qubits: {base.num_qubits}")
    print(f"Detectors: {dem.num_detectors}, Errors: {dem.num_errors}, Obs: {dem.num_observables}")

    # Find undetectable errors (flip observable but no detectors)
    undetectable = []
    for instr in dem.flattened():
        if instr.type != "error":
            continue
        targets = instr.targets_copy()
        prob = instr.args_copy()[0]
        has_det = any(t.is_relative_detector_id() for t in targets)
        has_obs = any(t.is_logical_observable_id() for t in targets)
        if has_obs and not has_det:
            undetectable.append((prob, targets))

    if not undetectable:
        print("  ✓ No undetectable errors - GOOD!")
    else:
        print(f"  ⚠️  {len(undetectable)} UNDETECTABLE errors:")
        for prob, tgts in undetectable:
            obs_ids = [t.val for t in tgts if t.is_logical_observable_id()]
            print(f"    prob={prob:.3e}  obs={obs_ids}")

    # Also check weight-1 errors (any error with <=1 detector and the observable)
    weight1_obs = []
    for instr in dem.flattened():
        if instr.type != "error":
            continue
        targets = instr.targets_copy()
        prob = instr.args_copy()[0]
        n_det = sum(1 for t in targets if t.is_relative_detector_id())
        has_obs = any(t.is_logical_observable_id() for t in targets)
        if has_obs and n_det <= 1:
            det_ids = [t.val for t in targets if t.is_relative_detector_id()]
            weight1_obs.append((prob, n_det, det_ids))

    if weight1_obs:
        print(f"  Weight ≤ 1 errors flipping observable: {len(weight1_obs)}")
        for prob, nd, dids in weight1_obs[:10]:
            print(f"    prob={prob:.3e}  dets={nd}  det_ids={dids}")

    # Shortest graphlike error path
    try:
        sp = dem.shortest_graphlike_error(ignore_ungraphlike_errors=True)
        print(f"  Shortest graphlike error: weight {len(sp)}")
    except Exception as ex:
        print(f"  shortest_graphlike_error: {ex}")

    # Now TRACE the undetectable error back to the circuit
    if undetectable:
        print("\n  --- Tracing undetectable error to circuit ---")
        
        # Build a filter DEM containing just the undetectable error
        for prob, tgts in undetectable:
            filter_dem = stim.DetectorErrorModel()
            filter_dem.append(
                "error",
                [prob],
                tgts
            )
            
            try:
                explanations = noisy.explain_detector_error_model_errors(
                    dem_filter=filter_dem
                )
                print(f"\n  Explanation for undetectable error (prob={prob:.3e}):")
                for exp_item in explanations:
                    print(f"    {exp_item}")
            except Exception as e:
                print(f"  explain failed: {e}")
                
                # Alternative: manual fault injection
                print("\n  Alternative: manual search via circuit inspection")
                
    # Check the OBSERVABLE_INCLUDE instructions in the circuit
    print(f"\n  OBSERVABLE_INCLUDE instructions in circuit:")
    flat = list(base.flattened())
    total_meas = 0
    obs_includes = []
    for i, instr in enumerate(flat):
        if instr.name in ("M", "MX", "MR", "MRX", "MY"):
            total_meas += len(instr.targets_copy())
        elif instr.name == "OBSERVABLE_INCLUDE":
            targets = instr.targets_copy()
            args = instr.args_copy()
            recs = [str(t) for t in targets]
            obs_includes.append((i, args, recs, total_meas))
    
    for idx, args, recs, meas_before in obs_includes:
        print(f"    [{idx}] OBS({int(args[0]) if args else '?'}) recs={recs} (meas_so_far={meas_before})")
    
    print(f"\n  Total measurements: {total_meas}")

    # Also check: CNOT-HTEL with same code for comparison
    if inp == "0":
        from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
        gadget2 = CNOTHTeleportGadget(input_state="0")
        exp2 = FaultTolerantGadgetExperiment(
            codes=[code], gadget=gadget2, noise_model=None,
            num_rounds_before=3, num_rounds_after=3,
        )
        base2 = exp2.to_stim()
        noisy2 = noise.apply(base2)
        dem2 = noisy2.detector_error_model(decompose_errors=False)
        
        undet2 = 0
        for instr in dem2.flattened():
            if instr.type != "error":
                continue
            targets = instr.targets_copy()
            has_det = any(t.is_relative_detector_id() for t in targets)
            has_obs = any(t.is_logical_observable_id() for t in targets)
            if has_obs and not has_det:
                undet2 += 1
        
        print(f"\n  Comparison CNOT-HTEL input=0: {dem2.num_detectors} det, {dem2.num_errors} err, {undet2} undetectable")

print("\nDone.")
