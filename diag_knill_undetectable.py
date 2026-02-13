#!/usr/bin/env python3
"""Deep investigation of the KnillEC undetectable weight-1 error.

A single fault that flips the observable with no detector signature
is a CRITICAL bug — it means the circuit-level distance is 1.

We need to find:
1. Which physical error channel causes this
2. Where in the circuit it occurs
3. Why no detector catches it
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

    # Very low noise so each error mechanism is isolated
    noise = StimStyleDepolarizingNoise(p=1e-8)
    noisy = noise.apply(base)

    dem = noisy.detector_error_model(decompose_errors=True)

    print(f"\n{'='*70}")
    print(f"KnillEC input={inp}")
    print(f"{'='*70}")
    print(f"Detectors: {dem.num_detectors}, Errors: {dem.num_errors}")

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
        continue

    print(f"  ⚠️  {len(undetectable)} UNDETECTABLE errors:")
    for prob, tgts in undetectable:
        obs_ids = [t.val for t in tgts if t.is_logical_observable_id()]
        print(f"    prob={prob:.3e}  obs={obs_ids}  targets={[str(t) for t in tgts]}")

    # Now use explain_detector_error_model to trace this back to the circuit
    print("\n  Tracing back to circuit errors...")
    noisy_explain = noisy.detector_error_model(
        decompose_errors=True,
        block_decomposition_from_introducing_remnant_edges=True,
    )

    # Also try the non-decomposed version
    dem_raw = noisy.detector_error_model(decompose_errors=False)
    print(f"\n  Raw DEM (no decompose): {dem_raw.num_errors} errors")

    undetectable_raw = []
    for instr in dem_raw.flattened():
        if instr.type != "error":
            continue
        targets = instr.targets_copy()
        prob = instr.args_copy()[0]
        has_det = any(t.is_relative_detector_id() for t in targets)
        has_obs = any(t.is_logical_observable_id() for t in targets)
        if has_obs and not has_det:
            undetectable_raw.append((prob, targets))

    if undetectable_raw:
        print(f"  Raw DEM also has {len(undetectable_raw)} undetectable errors:")
        for prob, tgts in undetectable_raw[:5]:
            print(f"    prob={prob:.3e}")

    # Use Stim's error analysis to find which circuit errors produce this
    print("\n  Using explain_detector_error_model_errors()...")
    try:
        explanations = noisy.explain_detector_error_model_errors(
            dem_filter=stim.DetectorErrorModel()
        )
        # This won't work with empty filter, try another approach
    except Exception as e:
        print(f"    explain failed: {e}")

    # Alternative: manually inject single errors and check
    print("\n  Manual single-fault injection analysis...")
    
    # Get the noiseless circuit
    noiseless = base.copy()
    
    # Count circuit operations to understand structure
    tick_count = 0
    gate_count = 0
    meas_count = 0
    for instr in noiseless.flattened():
        if instr.name == "TICK":
            tick_count += 1
        elif instr.name in ("M", "MX", "MR", "MRX"):
            meas_count += len(instr.targets_copy())
        elif instr.name not in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"):
            gate_count += 1
    
    print(f"    Circuit: {tick_count} ticks, {gate_count} gates, {meas_count} measurements")
    
    # Check: does shortest error path have weight 1?
    try:
        sp = dem.shortest_graphlike_error(ignore_ungraphlike_errors=True)
        print(f"    Shortest graphlike error path: weight {len(sp)}")
        for e in sp:
            print(f"      {e}")
    except Exception as ex:
        print(f"    shortest_graphlike_error: {ex}")

    # Now let's look at the circuit structure around the Bell measurement
    print("\n  Circuit structure (last 30 instructions before final measurements):")
    flat = list(noiseless.flattened())
    
    # Find measurement indices
    meas_idx = []
    for i, instr in enumerate(flat):
        if instr.name in ("M", "MX", "MR", "MRX"):
            meas_idx.append((i, instr.name, len(instr.targets_copy())))
    
    print(f"    Total measurement groups: {len(meas_idx)}")
    for i, (idx, name, count) in enumerate(meas_idx):
        print(f"      group {i}: instr[{idx}] {name} × {count} qubits")

    # Look at the DETECTOR and OBSERVABLE_INCLUDE instructions
    print("\n  DETECTOR and OBSERVABLE_INCLUDE instructions:")
    for i, instr in enumerate(flat):
        if instr.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            targets = instr.targets_copy()
            args = instr.args_copy()
            recs = [str(t) for t in targets]
            if instr.name == "OBSERVABLE_INCLUDE":
                print(f"    [{i}] {instr.name}({args[0] if args else '?'}) targets: {recs}")
            else:
                coords = args if args else []
                print(f"    [{i}] {instr.name} coords={coords} targets: {recs}")

print("\n\nDone.")
