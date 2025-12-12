#!/usr/bin/env python3
"""Quick test to verify transversal gate refactoring works correctly."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Clear cached modules
for mod in [k for k in sys.modules if 'qectostim' in k]:
    del sys.modules[mod]

print("Testing Transversal Gate Refactoring")
print("=" * 50)

# Test 1: Import the new TwoQubitObservableTransform
print("\n1. Testing TwoQubitObservableTransform import...")
try:
    from qectostim.gadgets.base import (
        StabilizerTransform,
        ObservableTransform,
        TwoQubitObservableTransform,
    )
    print("   ✓ Import successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test TwoQubitObservableTransform factory methods
print("\n2. Testing TwoQubitObservableTransform factory methods...")
try:
    cnot_transform = TwoQubitObservableTransform.cnot()
    print(f"   ✓ CNOT transform: ctrl_x={cnot_transform.control_x_to}, ctrl_z={cnot_transform.control_z_to}")
    print(f"                     tgt_x={cnot_transform.target_x_to}, tgt_z={cnot_transform.target_z_to}")
    
    cz_transform = TwoQubitObservableTransform.cz()
    print(f"   ✓ CZ transform: ctrl_x={cz_transform.control_x_to}, ctrl_z={cz_transform.control_z_to}")
    
    swap_transform = TwoQubitObservableTransform.swap()
    print(f"   ✓ SWAP transform: ctrl_x={swap_transform.control_x_to}, ctrl_z={swap_transform.control_z_to}")
except Exception as e:
    print(f"   ✗ Factory methods failed: {e}")
    sys.exit(1)

# Test 3: Test ObservableTransform.to_stabilizer_transform()
print("\n3. Testing ObservableTransform.to_stabilizer_transform()...")
try:
    h_obs = ObservableTransform.hadamard()
    h_stab = h_obs.to_stabilizer_transform()
    print(f"   ✓ Hadamard: x_becomes={h_stab.x_becomes}, z_becomes={h_stab.z_becomes}, swap_xz={h_stab.swap_xz}")
    
    s_obs = ObservableTransform.s_gate()
    s_stab = s_obs.to_stabilizer_transform()
    print(f"   ✓ S-gate: x_becomes={s_stab.x_becomes}, z_becomes={s_stab.z_becomes}")
except Exception as e:
    print(f"   ✗ to_stabilizer_transform failed: {e}")
    sys.exit(1)

# Test 4: Test that subclasses properly override get_stabilizer_transform
print("\n4. Testing TransversalGate subclasses...")
try:
    from qectostim.gadgets.transversal import (
        TransversalHadamard,
        TransversalS,
        TransversalCNOT,
        TransversalCZ,
        TransversalSWAP,
    )
    
    h_gadget = TransversalHadamard()
    h_stab = h_gadget.get_stabilizer_transform()
    print(f"   ✓ TransversalHadamard: x_becomes={h_stab.x_becomes}, z_becomes={h_stab.z_becomes}")
    
    s_gadget = TransversalS()
    s_stab = s_gadget.get_stabilizer_transform()
    print(f"   ✓ TransversalS: x_becomes={s_stab.x_becomes}, z_becomes={s_stab.z_becomes}")
    
    cnot_gadget = TransversalCNOT()
    cnot_stab = cnot_gadget.get_stabilizer_transform()
    print(f"   ✓ TransversalCNOT: x_becomes={cnot_stab.x_becomes}, z_becomes={cnot_stab.z_becomes}")
    
    # Test two-qubit observable transform
    cnot_2q = cnot_gadget.get_two_qubit_observable_transform()
    print(f"   ✓ TransversalCNOT two-qubit transform: ctrl_z_to={cnot_2q.control_z_to}, tgt_x_to={cnot_2q.target_x_to}")
    
    cz_gadget = TransversalCZ()
    cz_2q = cz_gadget.get_two_qubit_observable_transform()
    print(f"   ✓ TransversalCZ two-qubit transform: ctrl_x_to={cz_2q.control_x_to}")
    
    swap_gadget = TransversalSWAP()
    swap_2q = swap_gadget.get_two_qubit_observable_transform()
    print(f"   ✓ TransversalSWAP two-qubit transform: ctrl_x_to={swap_2q.control_x_to}")
    
except Exception as e:
    import traceback
    print(f"   ✗ Subclass test failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test get_metadata fallback
print("\n5. Testing get_metadata fallback...")
try:
    h_gadget = TransversalHadamard()
    metadata = h_gadget.get_metadata()
    print(f"   ✓ get_metadata returned: gadget_type={metadata.gadget_type}, logical_operation={metadata.logical_operation}")
except Exception as e:
    print(f"   ✗ get_metadata failed: {e}")
    sys.exit(1)

# Test 6: Test full circuit generation with a simple code
print("\n6. Testing full circuit generation...")
try:
    from qectostim.codes.small import SteanCode713
    from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
    from qectostim.noise.models import CircuitDepolarizingNoise
    
    code = SteanCode713()
    noise = CircuitDepolarizingNoise(p=0.001)
    gadget = TransversalHadamard()
    
    experiment = FaultTolerantGadgetExperiment(
        codes=[code],
        gadget=gadget,
        noise_model=noise,
        num_rounds_before=2,
        num_rounds_after=2,
    )
    
    circuit = experiment.to_stim()
    print(f"   ✓ Generated circuit with {circuit.num_qubits} qubits")
    
    # Check metadata is available after circuit generation
    metadata = gadget.get_metadata()
    print(f"   ✓ Metadata after generation: gadget_type={metadata.gadget_type}")
    
except Exception as e:
    import traceback
    print(f"   ✗ Circuit generation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("All tests passed! ✓")
print("=" * 50)
