#!/usr/bin/env python3
"""Diagnose crossing detector has_flow failures.

Current state: compensated pre-gadget meas, raw post-gadget meas → 21/24 fail.
"""
import logging
import stim
from qectostim.codes import SteaneCode713
from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget

logging.basicConfig(level=logging.WARNING)
# Enable DEBUG for crossing detector emission
logging.getLogger('qectostim.experiments.detector_emission').setLevel(logging.WARNING)

steane = SteaneCode713()
concat = ConcatenatedCSSCode(steane, steane)

def test_config(label, gadget, rounds=2):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    exp = FaultTolerantGadgetExperiment(
        codes=[concat], gadget=gadget, noise_model=None,
        num_rounds_before=rounds, num_rounds_after=rounds,
        d_inner=3,
    )
    circuit = exp.to_stim()
    print(f"  Qubits: {circuit.num_qubits}, Meas: {circuit.num_measurements}")
    print(f"  Detectors: {circuit.num_detectors}, Obs: {circuit.num_observables}")
    
    # Check DEM
    for mode in ["decompose", "ignore_failures", "no_decompose"]:
        try:
            if mode == "decompose":
                dem = circuit.detector_error_model(decompose_errors=True)
            elif mode == "ignore_failures":
                dem = circuit.detector_error_model(
                    decompose_errors=True,
                    ignore_decomposition_failures=True,
                )
            else:
                dem = circuit.detector_error_model(decompose_errors=False)
            print(f"  DEM ({mode}): OK → {dem.num_detectors} det, {dem.num_errors} errors, {dem.num_observables} obs")
            break
        except Exception as e:
            msg = str(e)[:120]
            print(f"  DEM ({mode}): FAIL — {msg}")

# CNOT-HTEL
for state in ["0", "+"]:
    test_config(f"CNOT-HTEL input={state}", CNOTHTeleportGadget(input_state=state))

# Surgery CNOT
for state in ["0", "+"]:
    test_config(f"Surgery CNOT ctrl={state}", CSSSurgeryCNOTGadget(control_state=state))
