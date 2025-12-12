#!/usr/bin/env python3
"""Comprehensive test script for TransversalH fix across multiple codes."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Clear any cached qectostim modules
modules_to_remove = [key for key in list(sys.modules.keys()) if 'qectostim' in key]
for mod in modules_to_remove:
    del sys.modules[mod]

# Now import everything fresh
from qectostim.codes.small import SteanCode713, FourQubit422Code, PerfectCode513
from qectostim.codes.small.non_css_codes import BareAncillaCode713
from qectostim.gadgets.transversal import TransversalHadamard
from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
from qectostim.noise.circuit_noise import CircuitDepolarizingNoise

# Test configuration
codes_to_test = [
    ("SteanCode713", SteanCode713),
    ("FourQubit422Code", FourQubit422Code),
    ("BareAncillaCode713", BareAncillaCode713),
    ("PerfectCode513", PerfectCode513),
]

results = []
print("=" * 70)
print("TransversalHadamard Fix - Comprehensive Test")
print("=" * 70)

for code_name, code_class in codes_to_test:
    print(f"\n{'='*70}")
    print(f"Testing: {code_name}")
    print("=" * 70)
    
    try:
        # Create code instance
        code = code_class()
        print(f"  ✓ Created code instance")
        print(f"    - n={code.n}, k={code.k}, d={code.d}")
        
        # Create gadget
        gadget = TransversalHadamard()
        print(f"  ✓ Created TransversalHadamard gadget")
        
        # Create noise model
        noise = CircuitDepolarizingNoise(p1=1e-3, p2=1e-3)
        print(f"  ✓ Created CircuitDepolarizingNoise(p1=1e-3, p2=1e-3)")
        
        # Create experiment with codes as list
        exp = FaultTolerantGadgetExperiment(
            codes=[code],
            gadget=gadget,
            noise_model=noise,
            num_rounds_before=2,
            num_rounds_after=2
        )
        print(f"  ✓ Created FaultTolerantGadgetExperiment")
        print(f"    - num_rounds_before=2, num_rounds_after=2")
        
        # Convert to stim circuit
        circuit = exp.to_stim()
        print(f"  ✓ Generated Stim circuit")
        print(f"    - Circuit has {circuit.num_qubits} qubits")
        print(f"    - Circuit has {circuit.num_detectors} detectors")
        print(f"    - Circuit has {circuit.num_observables} observables")
        
        # Generate detector error model (this is where non-deterministic errors would fail)
        dem = circuit.detector_error_model()
        
        # Count error mechanisms
        num_errors = len([inst for inst in dem.flattened() if inst.type == "error"])
        num_detectors = circuit.num_detectors
        
        print(f"  ✓ Generated Detector Error Model")
        print(f"    - {num_detectors} detectors")
        print(f"    - {num_errors} error mechanisms")
        
        results.append({
            "code": code_name,
            "status": "PASSED",
            "detectors": num_detectors,
            "error_mechanisms": num_errors,
            "n": code.n,
            "k": code.k,
            "d": code.d
        })
        print(f"\n  ★ PASSED ★")
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        tb = traceback.format_exc()
        results.append({
            "code": code_name,
            "status": "FAILED",
            "error": error_msg,
            "traceback": tb
        })
        print(f"\n  ✗ FAILED: {error_msg}")
        print(f"\n  Traceback:\n{tb}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

passed = [r for r in results if r["status"] == "PASSED"]
failed = [r for r in results if r["status"] == "FAILED"]

print(f"\nTotal: {len(results)} codes tested")
print(f"Passed: {len(passed)}")
print(f"Failed: {len(failed)}")

if passed:
    print("\n✓ PASSED CODES:")
    print("-" * 70)
    print(f"{'Code':<25} {'[n,k,d]':<12} {'Detectors':<12} {'Errors':<12}")
    print("-" * 70)
    for r in passed:
        nkd = f"[{r['n']},{r['k']},{r['d']}]"
        print(f"{r['code']:<25} {nkd:<12} {r['detectors']:<12} {r['error_mechanisms']:<12}")

if failed:
    print("\n✗ FAILED CODES:")
    print("-" * 70)
    for r in failed:
        print(f"  {r['code']}: {r['error']}")

print("\n" + "=" * 70)
if len(passed) == len(results):
    print("🎉 ALL TESTS PASSED - TransversalHadamard fix is working correctly!")
else:
    print(f"⚠️  {len(failed)} test(s) failed - review errors above")
print("=" * 70)
