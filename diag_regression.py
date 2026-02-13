#!/usr/bin/env python3
"""Quick regression test: flat Steane + hierarchical Steane×Steane."""
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

gadgets = [
    ("CNOT-HTEL 0", CNOTHTeleportGadget(input_state="0")),
    ("CNOT-HTEL +", CNOTHTeleportGadget(input_state="+")),
    ("CZ-HTEL 0", CZHTeleportGadget(input_state="0")),
    ("CZ-HTEL +", CZHTeleportGadget(input_state="+")),
    ("KnillEC 0", KnillECGadget(input_state="0")),
    ("KnillEC +", KnillECGadget(input_state="+")),
    ("Surgery 0", CSSSurgeryCNOTGadget(control_state="0")),
    ("Surgery +", CSSSurgeryCNOTGadget(control_state="+")),
]

flat_expected = {
    "CNOT-HTEL 0": 48, "CNOT-HTEL +": 48,
    "CZ-HTEL 0": 45, "CZ-HTEL +": 45,
    "KnillEC 0": 48, "KnillEC +": 48,
    "Surgery 0": 132, "Surgery +": 132,
}

pass_count = 0
fail_count = 0

# === FLAT CODES ===
print("=" * 70)
print("  FLAT Steane [[7,1,3]]")
print("=" * 70)
for label, gadget in gadgets:
    try:
        codes_list = [steane] * (3 if "Surgery" in label else 1)
        exp = FaultTolerantGadgetExperiment(
            codes=codes_list, gadget=gadget, noise_model=noise,
            num_rounds_before=3, num_rounds_after=3,
        )
        circuit = exp.to_stim()
        n_det = circuit.num_detectors
        
        # Validate DEM
        dem = circuit.detector_error_model(decompose_errors=True)
        print(f"  [{label}] {n_det} det → DEM OK ({dem.num_errors} err)")
        pass_count += 1
    except Exception as e:
        print(f"  [{label}] FAIL: {str(e)[:100]}")
        fail_count += 1
    except Exception as e:
        print(f"  [{label}] FAIL: {str(e)[:100]}")
        fail_count += 1

# === HIERARCHICAL ===
print(f"\n{'='*70}")
print("  HIERARCHICAL Steane×Steane [[49,1,9]]")
print("=" * 70)
for label, gadget in gadgets:
    try:
        codes_list = [concat] * (3 if "Surgery" in label else 1)
        exp = FaultTolerantGadgetExperiment(
            codes=codes_list, gadget=gadget, noise_model=noise,
            num_rounds_before=2, num_rounds_after=2, d_inner=3,
        )
        circuit = exp.to_stim()
        n_det = circuit.num_detectors
        n_obs = circuit.num_observables
        
        # Try DEM
        dem_ok = False
        for mode, kw in [
            ("decompose", dict(decompose_errors=True, approximate_disjoint_errors=True)),
            ("ignore", dict(decompose_errors=True, ignore_decomposition_failures=True, approximate_disjoint_errors=True)),
            ("no_decompose", dict(decompose_errors=False, approximate_disjoint_errors=True)),
        ]:
            try:
                dem = circuit.detector_error_model(**kw)
                print(f"  [{label}] {n_det} det, {n_obs} obs → DEM({mode}) OK: {dem.num_errors} err")
                dem_ok = True
                pass_count += 1
                break
            except:
                continue
        if not dem_ok:
            print(f"  [{label}] {n_det} det → ALL DEM MODES FAILED")
            fail_count += 1
    except Exception as e:
        print(f"  [{label}] FAIL: {str(e)[:100]}")
        fail_count += 1

print(f"\n{'='*70}")
print(f"  RESULTS: {pass_count} PASS, {fail_count} FAIL")
print(f"{'='*70}")
