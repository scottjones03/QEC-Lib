#!/usr/bin/env python3
"""Investigate: Do hierarchical builders emit their own crossing detectors?

If the builder already handles temporal detectors correctly through its 
normal emit_round mechanism, then the explicit crossing detector emission
might be redundant and harmful.

Key question: What happens if we disable crossing detectors entirely 
and rely on the builder's native temporal detectors?
"""
import logging
import stim
from qectostim.codes import SteaneCode713
from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.gadgets.teleportation_h_gadgets import CNOTHTeleportGadget
from qectostim.gadgets.css_surgery_cnot import CSSSurgeryCNOTGadget

logging.basicConfig(level=logging.WARNING)

steane = SteaneCode713()
concat = ConcatenatedCSSCode(steane, steane)

def test_no_crossing(label, gadget, rounds=2):
    """Test with crossing detectors disabled."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    # Option 1: Override get_crossing_detector_config to return None
    orig_config = gadget.get_crossing_detector_config
    gadget.get_crossing_detector_config = lambda: None
    
    exp = FaultTolerantGadgetExperiment(
        codes=[concat], gadget=gadget, noise_model=None,
        num_rounds_before=2, num_rounds_after=2, d_inner=3,
    )
    circuit = exp.to_stim()
    
    gadget.get_crossing_detector_config = orig_config
    
    print(f"  Qubits: {circuit.num_qubits}, Meas: {circuit.num_measurements}")
    print(f"  Detectors: {circuit.num_detectors}, Obs: {circuit.num_observables}")
    
    # Check DEM
    try:
        dem = circuit.detector_error_model(decompose_errors=True)
        print(f"  DEM (decompose=True): OK → {dem.num_detectors} det, {dem.num_errors} err")
    except Exception as e:
        msg = str(e)[:120]
        print(f"  DEM (decompose=True): FAIL — {msg}")
        try:
            dem = circuit.detector_error_model(decompose_errors=False)
            print(f"  DEM (decompose=False): OK → {dem.num_detectors} det, {dem.num_errors} err")
        except Exception as e2:
            print(f"  DEM (decompose=False): FAIL — {str(e2)[:120]}")
    
    # Now test with crossing detectors enabled (normal)
    print(f"\n  --- With crossing detectors (normal) ---")
    exp2 = FaultTolerantGadgetExperiment(
        codes=[concat], gadget=gadget, noise_model=None,
        num_rounds_before=2, num_rounds_after=2, d_inner=3,
    )
    circuit2 = exp2.to_stim()
    print(f"  Detectors: {circuit2.num_detectors}, Obs: {circuit2.num_observables}")
    print(f"  Difference: {circuit2.num_detectors - circuit.num_detectors} crossing dets emitted")
    
    # Now test with auto_detectors 
    print(f"\n  --- With auto_detectors=True ---")
    exp3 = FaultTolerantGadgetExperiment(
        codes=[concat], gadget=gadget, noise_model=None,
        num_rounds_before=2, num_rounds_after=2, d_inner=3,
        auto_detectors=True,
    )
    circuit3 = exp3.to_stim()
    print(f"  Detectors: {circuit3.num_detectors}, Obs: {circuit3.num_observables}")
    
    try:
        dem3 = circuit3.detector_error_model(decompose_errors=True)
        print(f"  DEM (decompose=True): OK → {dem3.num_detectors} det, {dem3.num_errors} err")
    except Exception as e:
        print(f"  DEM (decompose=True): FAIL — {str(e)[:120]}")

test_no_crossing("CNOT-HTEL input=0", CNOTHTeleportGadget(input_state="0"))
test_no_crossing("CNOT-HTEL input=+", CNOTHTeleportGadget(input_state="+"))
test_no_crossing("Surgery CNOT ctrl=0", CSSSurgeryCNOTGadget(control_state="0"))
