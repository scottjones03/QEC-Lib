#!/usr/bin/env python3
"""Quick test script for TransversalH fix."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from qectostim.gadgets import TransversalHadamard
from qectostim.experiments import FaultTolerantGadgetExperiment
from qectostim.noise.models import CircuitDepolarizingNoise
from qectostim.codes.small import SteaneCode

# Create test
code = SteaneCode()
gadget = TransversalHadamard()
noise = CircuitDepolarizingNoise(p=1e-3)

exp = FaultTolerantGadgetExperiment(
    codes=[code],
    gadget=gadget,
    noise_model=noise,
    pre_gadget_rounds=2,
    post_gadget_rounds=2
)

try:
    circuit = exp.to_stim()
    dem = circuit.detector_error_model()
    print("✓ TransversalH now generates valid DEM!")
    print(f"  Circuit has {circuit.num_detectors} detectors")
    print(f"  DEM has {dem.num_detectors} detectors, {dem.num_errors} error mechanisms")
except Exception as e:
    print(f"✗ Still failing: {e}")
    print("\nCircuit structure:")
    for i, line in enumerate(str(circuit).split('\n')[:100]):
        print(f"{i+1:3d}: {line}")
