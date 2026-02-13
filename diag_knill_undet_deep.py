#!/usr/bin/env python3
"""Deep analysis of the remaining undetectable error in KnillEC.

We know:
- bell_a Z boundary detectors fixed: 3 new detectors  
- data_block X boundary detectors impossible (protocol limitation)
- 1 undetectable error remains

This script traces exactly which fault locations produce undetectable L errors,
and categorizes them to understand if they can be fixed.
"""
import stim
from qectostim.codes import SteaneCode713
from qectostim.gadgets.knill_ec import KnillECGadget
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.models import StimStyleDepolarizingNoise

code = SteaneCode713()
noise = StimStyleDepolarizingNoise(p=1e-8)

for inp in ["0", "+"]:
    gadget = KnillECGadget(input_state=inp)
    exp = FaultTolerantGadgetExperiment(
        codes=[code], gadget=gadget, noise_model=noise,
        num_rounds_before=3, num_rounds_after=3,
    )
    circ = exp.to_stim()
    dem = circ.detector_error_model(decompose_errors=False)
    
    # Find all undetectable errors (no detector symptoms, only L_ flips)
    undetectable = []
    for error in dem:
        if error.type != "error":
            continue
        targets = error.targets_copy()
        det_targets = [t for t in targets if t.is_relative_detector_id()]
        obs_targets = [t for t in targets if t.is_logical_observable_id()]
        if len(det_targets) == 0 and len(obs_targets) > 0:
            undetectable.append(error)
    
    print(f"=== KnillEC input={inp} ===")
    print(f"Detectors: {circ.num_detectors}")
    print(f"DEM errors: {sum(1 for e in dem if e.type == 'error')}")
    print(f"Undetectable errors: {len(undetectable)}")
    
    for ue in undetectable:
        obs = [t.val for t in ue.targets_copy() if t.is_logical_observable_id()]
        print(f"  prob={ue.args_copy()[0]:.3e}, L={obs}")
    
    # Trace back to fault locations using explain
    if undetectable:
        print(f"\n  --- Tracing undetectable errors ---")
        explained = circ.explain_detector_error_model_errors(
            dem_filter=stim.DetectorErrorModel(str(undetectable[0])),
            reduce_to_one_representative_error=False
        )
        
        print(f"  {len(explained)} fault location(s) produce this error:")
        for i, exp_err in enumerate(explained[:15]):
            for loc in exp_err.circuit_error_locations:
                print(f"\n  Fault {i}: tick={loc.tick_offset}")
                print(f"    flipped_pauli: {loc.flipped_pauli_product}")
                inst = loc.instruction_targets
                print(f"    gate: {inst.gate}")
                tgts = inst.targets_copy()
                for t in tgts:
                    print(f"      target: {t}")
                print(f"    args: {inst.args}")
        if len(explained) > 15:
            print(f"  ... and {len(explained) - 15} more")
    print()
