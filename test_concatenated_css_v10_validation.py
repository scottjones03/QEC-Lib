"""
Side-by-side circuit comparison test between original concatenated_steane.py
and generalized concatenated_css_v10.py.

This test builds identical circuits with both implementations and compares:
1. Gate counts by type
2. Gate order
3. Detector structure
4. Qubit indices used
5. Memory logical error rate estimation (L2)
"""

import sys
import os

# Add paths for imports
base_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(base_dir, 'src', 'qectostim', 'experiments'))
sys.path.insert(0, os.path.join(base_dir, 'src', 'qectostim', 'noise'))
sys.path.insert(0, os.path.join(base_dir, 'src', 'qectostim'))
sys.path.insert(0, os.path.join(base_dir, 'src'))

# Create a mock noise_model module to satisfy the import in concatenated_css_v10.py
# The actual NoiseModel isn't needed for circuit comparison tests
import types
noise_model_mock = types.ModuleType('noise_model')
class MockNoiseModel:
    pass
noise_model_mock.NoiseModel = MockNoiseModel
sys.modules['noise_model'] = noise_model_mock

import stim
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Import from original concatenated_steane.py
import qectostim.experiments.concatenated_steane as orig_steane


# =============================================================================
# Original Implementation Wrapper
# =============================================================================

class OriginalImplementation:
    """
    Wrapper around the original concatenated_steane.py module.
    Uses the module's global variables and functions directly.
    """
    
    def __init__(self):
        self.N_steane = 7
        
        # Initialize global variables in the original module
        orig_steane.N_steane = 7
        orig_steane.gamma = 0.0001
        orig_steane.detector_now = 0
        
        # Set up the check matrix and logical operators
        orig_steane.check_matrix = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ])
        orig_steane.logical_op = np.array([1, 1, 1, 0, 0, 0, 0])
        
        # Propagation tables for level-2
        orig_steane.propagation_l2_0prep_X = [
            [0,2,4,6],[1,2,5,6],[2],[3,4,5,6],[4,6],[5],
            [0,2],[1,5,6],[2],[3,4,6],[4,6],[5],[6],
            [0],[1,5],[2],[3],[4,6],[5],[6],[],
            [0],[1],[2],[3],[4],[5],[6],[],
            [0],[1],[2],[3],[4],[5],[6],[],
            [0],[1],[2],[3],[4],[5],[6],[]
        ]
        
        orig_steane.propagation_l2_0prep_Z = [
            [0],[1],[2,0,1],[3],[4,3,0],[5,1,3],
            [0],[1],[2,0],[3],[4,3],[5,1],[6,1,4],
            [0],[1],[2],[3],[4],[5,1],[6,4],[2,4,5],
            [0],[1],[2],[3],[4],[5],[6],[4,5],
            [0],[1],[2],[3],[4],[5],[6],[5],
            [0],[1],[2],[3],[4],[5],[6],[]
        ]
        
        orig_steane.propagation_l2_0prep_m = [2,4,5,6,7,8,9,10,11,14,15,17,18,20,25,26,28,34,36,44]
        orig_steane.num_ec_0prep = 45
    
    def reset(self):
        """Reset detector counter."""
        orig_steane.detector_now = 0
    
    def set_gamma(self, p):
        """Set gamma for error model 'a'."""
        orig_steane.gamma = p / 10
    
    def build_l1_cnot_test_circuit(self, p, Q=10):
        """Build level-1 CNOT test circuit using original implementation."""
        self.reset()
        self.set_gamma(p)
        
        N_prev = 1
        N_now = self.N_steane
        NN = 2 * N_now
        
        circuit = stim.Circuit()
        
        # Initialize global lists
        orig_steane.list_detector_0prep = []
        orig_steane.list_detector_X = []
        orig_steane.list_detector_Z = []
        orig_steane.list_detector_m = []
        orig_steane.Q = Q
        orig_steane.no_post_EC = True
        orig_steane.isCZ = False
        
        list_detector_0prep = []
        list_detector_X = []
        list_detector_Z = []
        list_detector_m = []
        
        # Prepare ideal Bell pairs
        orig_steane.append_0prep(circuit, 0, N_prev, N_now)
        orig_steane.append_0prep(circuit, NN, N_prev, N_now)
        orig_steane.append_0prep(circuit, 2 * NN, N_prev, N_now)
        orig_steane.append_0prep(circuit, 3 * NN, N_prev, N_now)
        
        orig_steane.append_h(circuit, 0, N_prev, N_now)
        orig_steane.append_h(circuit, 2 * NN, N_prev, N_now)
        
        orig_steane.append_cnot(circuit, 0, NN, N_prev, N_now)
        orig_steane.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, N_now)
        
        # Q rounds of CNOT + EC
        for q in range(Q):
            orig_steane.append_cnot(circuit, 0, 2 * NN, N_prev, N_now)
            orig_steane.append_noisy_cnot(circuit, 0, 2 * NN, N_prev, N_now, p)
            
            result = orig_steane.append_noisy_ec(circuit, 0, 4 * NN, 5 * NN, 6 * NN, N_prev, N_now, p)
            list_detector_0prep.extend(result[0])
            list_detector_X.append(result[2])
            list_detector_Z.append(result[1])
            
            result = orig_steane.append_noisy_ec(circuit, 2 * NN, 4 * NN, 5 * NN, 6 * NN, N_prev, N_now, p)
            list_detector_0prep.extend(result[0])
            list_detector_X.append(result[2])
            list_detector_Z.append(result[1])
        
        # Undo Bell pairs
        orig_steane.append_cnot(circuit, 0, NN, N_prev, N_now)
        orig_steane.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, N_now)
        
        orig_steane.append_h(circuit, 0, N_prev, N_now)
        orig_steane.append_h(circuit, 2 * NN, N_prev, N_now)
        
        # Measure
        list_detector_m.append(orig_steane.append_m(circuit, 0, N_prev, N_now))
        list_detector_m.append(orig_steane.append_m(circuit, NN, N_prev, N_now))
        list_detector_m.append(orig_steane.append_m(circuit, 2 * NN, N_prev, N_now))
        list_detector_m.append(orig_steane.append_m(circuit, 3 * NN, N_prev, N_now))
        
        return circuit, {
            'detector_0prep': list_detector_0prep,
            'detector_X': list_detector_X,
            'detector_Z': list_detector_Z,
            'detector_m': list_detector_m
        }


# =============================================================================
# New Implementation Wrapper (v10)
# =============================================================================

class NewImplementationV10:
    """Wrapper for the new concatenated_css_v10_steane.py implementation."""
    
    def __init__(self):
        # Import Steane-specific components from the new steane module
        from qectostim.experiments.concatenated_css_v10_steane import (
            create_concatenated_steane,
            SteanePreparationStrategy,
            SteaneECGadget,
            SteaneDecoder,
        )
        from qectostim.experiments.concatenated_css_v10 import (
            TransversalOps,
            PostSelector,
        )
        
        self.concat_code = create_concatenated_steane(2)  # 2 levels for L2
        self.ops = TransversalOps(self.concat_code)
        self.prep = SteanePreparationStrategy(self.concat_code, self.ops)
        self.ec = SteaneECGadget(self.concat_code, self.ops)
        self.decoder = SteaneDecoder(self.concat_code)
        self.post_selector = PostSelector(self.concat_code, self.decoder)
        
        # Wire up circular dependencies
        self.ec.set_prep(self.prep)
        self.prep.set_ec_gadget(self.ec)
        
        # Store classes for later use
        self._create_concatenated_steane = create_concatenated_steane
    
    def build_l1_cnot_test_circuit(self, p, Q=10):
        """Build level-1 CNOT test circuit using new v10 implementation."""
        # Use single level for L1
        from qectostim.experiments.concatenated_css_v10_steane import (
            create_concatenated_steane,
            SteanePreparationStrategy,
            SteaneECGadget,
        )
        from qectostim.experiments.concatenated_css_v10 import TransversalOps
        
        concat_code = create_concatenated_steane(1)
        ops = TransversalOps(concat_code)
        prep = SteanePreparationStrategy(concat_code, ops)
        ec = SteaneECGadget(concat_code, ops)
        ec.set_prep(prep)
        prep.set_ec_gadget(ec)
        
        N_prev = 1
        N_now = 7
        NN = 2 * N_now
        
        circuit = stim.Circuit()
        detector_counter = [0]
        
        list_detector_0prep = []
        list_detector_X = []
        list_detector_Z = []
        list_detector_m = []
        
        # Prepare ideal Bell pairs
        prep.append_0prep(circuit, 0, N_prev, N_now)
        prep.append_0prep(circuit, NN, N_prev, N_now)
        prep.append_0prep(circuit, 2 * NN, N_prev, N_now)
        prep.append_0prep(circuit, 3 * NN, N_prev, N_now)
        
        ops.append_h(circuit, 0, N_prev, N_now)
        ops.append_h(circuit, 2 * NN, N_prev, N_now)
        
        ops.append_cnot(circuit, 0, NN, N_prev, N_now)
        ops.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, N_now)
        
        # Q rounds of CNOT + EC
        for q in range(Q):
            ops.append_cnot(circuit, 0, 2 * NN, N_prev, N_now)
            ops.append_noisy_cnot(circuit, 0, 2 * NN, N_prev, N_now, p)
            
            result = ec.append_noisy_ec(
                circuit, 0, 4 * NN, 5 * NN, 6 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_X.append(result[2])
            list_detector_Z.append(result[1])
            
            result = ec.append_noisy_ec(
                circuit, 2 * NN, 4 * NN, 5 * NN, 6 * NN, N_prev, N_now, p, detector_counter
            )
            list_detector_0prep.extend(result[0])
            list_detector_X.append(result[2])
            list_detector_Z.append(result[1])
        
        # Undo Bell pairs
        ops.append_cnot(circuit, 0, NN, N_prev, N_now)
        ops.append_cnot(circuit, 2 * NN, 3 * NN, N_prev, N_now)
        
        ops.append_h(circuit, 0, N_prev, N_now)
        ops.append_h(circuit, 2 * NN, N_prev, N_now)
        
        # Measure
        list_detector_m.append(ops.append_m(circuit, 0, N_prev, N_now, detector_counter))
        list_detector_m.append(ops.append_m(circuit, NN, N_prev, N_now, detector_counter))
        list_detector_m.append(ops.append_m(circuit, 2 * NN, N_prev, N_now, detector_counter))
        list_detector_m.append(ops.append_m(circuit, 3 * NN, N_prev, N_now, detector_counter))
        
        return circuit, {
            'detector_0prep': list_detector_0prep,
            'detector_X': list_detector_X,
            'detector_Z': list_detector_Z,
            'detector_m': list_detector_m
        }


# =============================================================================
# Comparison Utilities
# =============================================================================

@dataclass
class GateInfo:
    """Information about a single gate."""
    name: str
    targets: Tuple
    args: Tuple = ()
    
    def __hash__(self):
        return hash((self.name, self.targets, self.args))


@dataclass
class CircuitAnalysis:
    """Analysis of a circuit."""
    gate_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    gate_sequence: List[GateInfo] = field(default_factory=list)
    qubit_usage: Dict[int, List[str]] = field(default_factory=lambda: defaultdict(list))
    detector_count: int = 0
    measurement_count: int = 0


def analyze_circuit(circuit: stim.Circuit) -> CircuitAnalysis:
    """Analyze a circuit and extract gate information."""
    analysis = CircuitAnalysis()
    
    for instruction in circuit:
        name = instruction.name
        
        # Get targets
        targets = []
        for t in instruction.targets_copy():
            if t.is_qubit_target:
                targets.append(t.qubit_value)
            elif t.is_measurement_record_target:
                targets.append(f"rec[{t.value}]")
        
        # Get args
        args = tuple(instruction.gate_args_copy())
        
        # Update counts
        analysis.gate_counts[name] += 1
        
        # Track gate sequence
        gate_info = GateInfo(name, tuple(targets), args)
        analysis.gate_sequence.append(gate_info)
        
        # Track qubit usage
        for t in targets:
            if isinstance(t, int):
                analysis.qubit_usage[t].append(name)
        
        # Special counters
        if name == "DETECTOR":
            analysis.detector_count += 1
        elif name == "M":
            analysis.measurement_count += 1
    
    return analysis


def compare_gate_counts(orig: CircuitAnalysis, new: CircuitAnalysis) -> List[str]:
    """Compare gate counts between two circuits."""
    differences = []
    
    all_gates = set(orig.gate_counts.keys()) | set(new.gate_counts.keys())
    
    for gate in sorted(all_gates):
        orig_count = orig.gate_counts.get(gate, 0)
        new_count = new.gate_counts.get(gate, 0)
        
        if orig_count != new_count:
            differences.append(f"  {gate}: original={orig_count}, new={new_count}, diff={new_count - orig_count}")
    
    return differences


def compare_gate_sequence(orig: CircuitAnalysis, new: CircuitAnalysis,
                          max_diffs: int = 20) -> List[str]:
    """Compare gate sequences."""
    differences = []
    
    min_len = min(len(orig.gate_sequence), len(new.gate_sequence))
    max_len = max(len(orig.gate_sequence), len(new.gate_sequence))
    
    if len(orig.gate_sequence) != len(new.gate_sequence):
        differences.append(f"  Sequence length differs: original={len(orig.gate_sequence)}, new={len(new.gate_sequence)}")
    
    diff_count = 0
    for i in range(min_len):
        orig_gate = orig.gate_sequence[i]
        new_gate = new.gate_sequence[i]
        
        if orig_gate.name != new_gate.name or orig_gate.targets != new_gate.targets:
            if diff_count < max_diffs:
                differences.append(
                    f"  Position {i}: original={orig_gate.name}{list(orig_gate.targets)}, "
                    f"new={new_gate.name}{list(new_gate.targets)}"
                )
            diff_count += 1
    
    if diff_count > max_diffs:
        differences.append(f"  ... and {diff_count - max_diffs} more differences")
    
    return differences


def compare_detector_structure(orig_info: Dict, new_info: Dict) -> List[str]:
    """Compare detector structures."""
    differences = []
    
    # Compare detector_0prep
    orig_0prep = orig_info.get('detector_0prep', [])
    new_0prep = new_info.get('detector_0prep', [])
    
    if len(orig_0prep) != len(new_0prep):
        differences.append(f"  detector_0prep length: original={len(orig_0prep)}, new={len(new_0prep)}")
    else:
        for i, (o, n) in enumerate(zip(orig_0prep, new_0prep)):
            if o != n:
                differences.append(f"  detector_0prep[{i}]: original={o}, new={n}")
                if len(differences) > 10:
                    differences.append("  ... (truncated)")
                    break
    
    # Compare detector_m
    orig_m = orig_info.get('detector_m', [])
    new_m = new_info.get('detector_m', [])
    
    if len(orig_m) != len(new_m):
        differences.append(f"  detector_m length: original={len(orig_m)}, new={len(new_m)}")
    else:
        for i, (o, n) in enumerate(zip(orig_m, new_m)):
            if o != n:
                differences.append(f"  detector_m[{i}]: original={o}, new={n}")
    
    # Compare detector_X structure
    orig_X = orig_info.get('detector_X', [])
    new_X = new_info.get('detector_X', [])
    
    if len(orig_X) != len(new_X):
        differences.append(f"  detector_X length: original={len(orig_X)}, new={len(new_X)}")
    
    # Compare detector_Z structure
    orig_Z = orig_info.get('detector_Z', [])
    new_Z = new_info.get('detector_Z', [])
    
    if len(orig_Z) != len(new_Z):
        differences.append(f"  detector_Z length: original={len(orig_Z)}, new={len(new_Z)}")
    
    return differences


def find_first_difference(orig: CircuitAnalysis, new: CircuitAnalysis) -> Tuple[int, GateInfo, GateInfo]:
    """Find the first position where circuits differ."""
    min_len = min(len(orig.gate_sequence), len(new.gate_sequence))
    
    for i in range(min_len):
        orig_gate = orig.gate_sequence[i]
        new_gate = new.gate_sequence[i]
        
        if orig_gate.name != new_gate.name or orig_gate.targets != new_gate.targets:
            return i, orig_gate, new_gate
    
    if len(orig.gate_sequence) != len(new.gate_sequence):
        return min_len, None, None
    
    return -1, None, None


def print_circuit_context(analysis: CircuitAnalysis, position: int, context: int = 5):
    """Print circuit gates around a specific position."""
    start = max(0, position - context)
    end = min(len(analysis.gate_sequence), position + context + 1)
    
    for i in range(start, end):
        gate = analysis.gate_sequence[i]
        marker = " >>> " if i == position else "     "
        print(f"{marker}[{i:4d}] {gate.name} {list(gate.targets)} {gate.args if gate.args else ''}")


# =============================================================================
# Main Comparison Tests
# =============================================================================

def run_comparison_tests():
    """Run all comparison tests."""
    print("=" * 70)
    print("CIRCUIT COMPARISON: concatenated_steane.py vs concatenated_css_v10.py")
    print("=" * 70)
    print()
    
    # Test parameters
    p = 0.001
    Q = 1  # Start with just 1 round for easier debugging
    
    print(f"Test parameters: p={p}, Q={Q}")
    print()
    
    # Build circuits
    print("Building circuits...")
    
    orig_impl = OriginalImplementation()
    orig_circuit, orig_info = orig_impl.build_l1_cnot_test_circuit(p, Q)
    print(f"  Original circuit: {len(orig_circuit)} instructions")
    
    try:
        new_impl = NewImplementationV10()
        new_circuit, new_info = new_impl.build_l1_cnot_test_circuit(p, Q)
        print(f"  New circuit (v10): {len(new_circuit)} instructions")
    except Exception as e:
        print(f"  ERROR building new circuit: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Analyze circuits
    print("Analyzing circuits...")
    orig_analysis = analyze_circuit(orig_circuit)
    new_analysis = analyze_circuit(new_circuit)
    print()
    
    all_passed = True
    
    # Test 1: Gate counts
    print("-" * 70)
    print("TEST 1: Gate Count Comparison")
    print("-" * 70)
    
    count_diffs = compare_gate_counts(orig_analysis, new_analysis)
    if count_diffs:
        print("DIFFERENCES FOUND:")
        for diff in count_diffs:
            print(diff)
        all_passed = False
    else:
        print("✓ All gate counts match!")
    
    print()
    print("Gate counts (original):")
    for gate, count in sorted(orig_analysis.gate_counts.items()):
        print(f"  {gate}: {count}")
    print()
    
    # Test 2: Gate sequence
    print("-" * 70)
    print("TEST 2: Gate Sequence Comparison")
    print("-" * 70)
    
    seq_diffs = compare_gate_sequence(orig_analysis, new_analysis)
    if seq_diffs:
        print("DIFFERENCES FOUND:")
        for diff in seq_diffs:
            print(diff)
        all_passed = False
        
        # Find and show context around first difference
        pos, orig_gate, new_gate = find_first_difference(orig_analysis, new_analysis)
        if pos >= 0:
            print()
            print(f"First difference at position {pos}:")
            print()
            print("ORIGINAL circuit context:")
            print_circuit_context(orig_analysis, pos)
            print()
            print("NEW circuit context:")
            print_circuit_context(new_analysis, pos)
    else:
        print("✓ Gate sequences match!")
    print()
    
    # Test 3: Detector structure
    print("-" * 70)
    print("TEST 3: Detector Structure Comparison")
    print("-" * 70)
    
    det_diffs = compare_detector_structure(orig_info, new_info)
    if det_diffs:
        print("DIFFERENCES FOUND:")
        for diff in det_diffs:
            print(diff)
        all_passed = False
    else:
        print("✓ Detector structures match!")
    print()
    
    # Test 4: Circuit validity
    print("-" * 70)
    print("TEST 4: Circuit Validity")
    print("-" * 70)
    
    try:
        orig_sampler = orig_circuit.compile_detector_sampler()
        print("✓ Original circuit compiles successfully")
    except Exception as e:
        print(f"✗ Original circuit failed to compile: {e}")
        all_passed = False
    
    try:
        new_sampler = new_circuit.compile_detector_sampler()
        print("✓ New circuit (v10) compiles successfully")
    except Exception as e:
        print(f"✗ New circuit (v10) failed to compile: {e}")
        all_passed = False
    print()
    
    # Test 5: Sample comparison (if both compile)
    print("-" * 70)
    print("TEST 5: Sampling Comparison (statistical)")
    print("-" * 70)
    
    try:
        np.random.seed(42)
        orig_samples = orig_circuit.compile_detector_sampler().sample(shots=100)
        new_samples = new_circuit.compile_detector_sampler().sample(shots=100)
        
        print(f"  Original samples shape: {orig_samples.shape}")
        print(f"  New samples shape: {new_samples.shape}")
        
        if orig_samples.shape == new_samples.shape:
            print("✓ Sample shapes match!")
        else:
            print("✗ Sample shapes differ!")
            all_passed = False
    except Exception as e:
        print(f"  Sampling failed: {e}")
        all_passed = False
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_passed:
        print("✓ ALL TESTS PASSED - Circuits are equivalent!")
    else:
        print("✗ SOME TESTS FAILED - See details above")
    
    return all_passed


def run_minimal_comparison():
    """Run minimal comparison for just preparation circuit."""
    print("=" * 70)
    print("MINIMAL COMPARISON: Just State Preparation (v10)")
    print("=" * 70)
    print()
    
    # Original
    orig = OriginalImplementation()
    orig.reset()
    orig_circuit = stim.Circuit()
    orig_steane.append_0prep(orig_circuit, 0, 1, 7)
    
    print("Original 0prep circuit:")
    orig_analysis = analyze_circuit(orig_circuit)
    for gate, count in sorted(orig_analysis.gate_counts.items()):
        print(f"  {gate}: {count}")
    print()
    print("Gate sequence:")
    for i, gate in enumerate(orig_analysis.gate_sequence):
        print(f"  [{i:2d}] {gate.name} {list(gate.targets)}")
    print()
    
    # New v10
    try:
        from qectostim.experiments.concatenated_css_v10_steane import (
            create_concatenated_steane,
            SteanePreparationStrategy,
            SteaneECGadget,
        )
        from qectostim.experiments.concatenated_css_v10 import TransversalOps
        
        concat_code = create_concatenated_steane(1)
        ops = TransversalOps(concat_code)
        prep = SteanePreparationStrategy(concat_code, ops)
        ec = SteaneECGadget(concat_code, ops)
        ec.set_prep(prep)
        prep.set_ec_gadget(ec)
        
        new_circuit = stim.Circuit()
        prep.append_0prep(new_circuit, 0, 1, 7)
        
        print("New 0prep circuit (v10):")
        new_analysis = analyze_circuit(new_circuit)
        for gate, count in sorted(new_analysis.gate_counts.items()):
            print(f"  {gate}: {count}")
        print()
        print("Gate sequence:")
        for i, gate in enumerate(new_analysis.gate_sequence):
            print(f"  [{i:2d}] {gate.name} {list(gate.targets)}")
        print()
        
        # Compare
        print("Comparison:")
        diffs = compare_gate_counts(orig_analysis, new_analysis)
        if diffs:
            print("  DIFFERENCES:")
            for d in diffs:
                print(f"    {d}")
        else:
            print("  ✓ Gate counts match!")
        
        seq_diffs = compare_gate_sequence(orig_analysis, new_analysis)
        if seq_diffs:
            print("  SEQUENCE DIFFERENCES:")
            for d in seq_diffs:
                print(f"    {d}")
        else:
            print("  ✓ Sequences match!")
            
    except Exception as e:
        print(f"Error with new implementation (v10): {e}")
        import traceback
        traceback.print_exc()


def run_noisy_prep_comparison():
    """Compare noisy preparation circuits."""
    print("=" * 70)
    print("NOISY PREPARATION COMPARISON (L1) - v10")
    print("=" * 70)
    print()
    
    p = 0.001
    
    # Original
    orig = OriginalImplementation()
    orig.reset()
    orig.set_gamma(p)
    orig_circuit = stim.Circuit()
    orig_detector_info = orig_steane.append_noisy_0prep(orig_circuit, 0, 7, 1, 7, p)
    
    print(f"Original noisy_0prep_l1 circuit ({len(orig_circuit)} instructions):")
    orig_analysis = analyze_circuit(orig_circuit)
    for gate, count in sorted(orig_analysis.gate_counts.items()):
        print(f"  {gate}: {count}")
    print(f"  Detector info: {orig_detector_info}")
    print()
    
    # New v10
    try:
        from qectostim.experiments.concatenated_css_v10_steane import (
            create_concatenated_steane,
            SteanePreparationStrategy,
            SteaneECGadget,
        )
        from qectostim.experiments.concatenated_css_v10 import TransversalOps
        
        concat_code = create_concatenated_steane(1)
        ops = TransversalOps(concat_code)
        prep = SteanePreparationStrategy(concat_code, ops)
        ec = SteaneECGadget(concat_code, ops)
        ec.set_prep(prep)
        prep.set_ec_gadget(ec)
        
        new_circuit = stim.Circuit()
        detector_counter = [0]
        new_detector_info = prep.append_noisy_0prep(new_circuit, 0, 7, 1, 7, p, detector_counter)
        
        print(f"New noisy_0prep circuit v10 ({len(new_circuit)} instructions):")
        new_analysis = analyze_circuit(new_circuit)
        for gate, count in sorted(new_analysis.gate_counts.items()):
            print(f"  {gate}: {count}")
        print(f"  Detector info: {new_detector_info}")
        print()
        
        # Compare
        print("Comparison:")
        diffs = compare_gate_counts(orig_analysis, new_analysis)
        if diffs:
            print("  GATE COUNT DIFFERENCES:")
            for d in diffs:
                print(f"    {d}")
        else:
            print("  ✓ Gate counts match!")
        
        seq_diffs = compare_gate_sequence(orig_analysis, new_analysis, max_diffs=10)
        if seq_diffs:
            print("  SEQUENCE DIFFERENCES:")
            for d in seq_diffs:
                print(f"    {d}")
            
            # Show context
            pos, _, _ = find_first_difference(orig_analysis, new_analysis)
            if pos >= 0:
                print()
                print(f"  Context around first difference (position {pos}):")
                print()
                print("  ORIGINAL:")
                print_circuit_context(orig_analysis, pos, context=3)
                print()
                print("  NEW (v10):")
                print_circuit_context(new_analysis, pos, context=3)
        else:
            print("  ✓ Sequences match!")
            
    except Exception as e:
        print(f"Error with new implementation (v10): {e}")
        import traceback
        traceback.print_exc()


def run_ec_comparison():
    """Compare EC gadget circuits."""
    print("=" * 70)
    print("EC GADGET COMPARISON (L1) - v10")
    print("=" * 70)
    print()
    
    p = 0.001
    
    # Original
    orig = OriginalImplementation()
    orig.reset()
    orig.set_gamma(p)
    orig_circuit = stim.Circuit()
    orig_result = orig_steane.append_noisy_ec(orig_circuit, 0, 14, 28, 42, 1, 7, p)
    
    print(f"Original EC circuit ({len(orig_circuit)} instructions):")
    orig_analysis = analyze_circuit(orig_circuit)
    for gate, count in sorted(orig_analysis.gate_counts.items()):
        print(f"  {gate}: {count}")
    print(f"  detector_0prep: {len(orig_result[0])} items")
    print(f"  detector_Z: {len(orig_result[1])} items")
    print(f"  detector_X: {len(orig_result[2])} items")
    print()
    
    # New v10
    try:
        from qectostim.experiments.concatenated_css_v10_steane import (
            create_concatenated_steane,
            SteanePreparationStrategy,
            SteaneECGadget,
        )
        from qectostim.experiments.concatenated_css_v10 import TransversalOps
        
        concat_code = create_concatenated_steane(1)
        ops = TransversalOps(concat_code)
        prep = SteanePreparationStrategy(concat_code, ops)
        ec = SteaneECGadget(concat_code, ops)
        ec.set_prep(prep)
        prep.set_ec_gadget(ec)
        
        new_circuit = stim.Circuit()
        detector_counter = [0]
        new_result = ec.append_noisy_ec(new_circuit, 0, 14, 28, 42, 1, 7, p, detector_counter)
        
        print(f"New EC circuit v10 ({len(new_circuit)} instructions):")
        new_analysis = analyze_circuit(new_circuit)
        for gate, count in sorted(new_analysis.gate_counts.items()):
            print(f"  {gate}: {count}")
        print(f"  detector_0prep: {len(new_result[0])} items")
        print(f"  detector_Z: {len(new_result[1])} items")
        print(f"  detector_X: {len(new_result[2])} items")
        print()
        
        # Compare
        print("Comparison:")
        diffs = compare_gate_counts(orig_analysis, new_analysis)
        if diffs:
            print("  GATE COUNT DIFFERENCES:")
            for d in diffs:
                print(f"    {d}")
        else:
            print("  ✓ Gate counts match!")
        
        seq_diffs = compare_gate_sequence(orig_analysis, new_analysis, max_diffs=10)
        if seq_diffs:
            print("  SEQUENCE DIFFERENCES:")
            for d in seq_diffs:
                print(f"    {d}")
        else:
            print("  ✓ Sequences match!")
        
        # Compare detector structures
        if len(orig_result[0]) != len(new_result[0]):
            print(f"  ✗ detector_0prep lengths differ: {len(orig_result[0])} vs {len(new_result[0])}")
        else:
            print(f"  ✓ detector_0prep lengths match: {len(orig_result[0])}")
            
    except Exception as e:
        print(f"Error with new implementation (v10): {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# CNOT L1 Comparison
# =============================================================================

def run_cnot_l1_comparison():
    """Compare estimate_logical_cnot_error_l1 between original and v10."""
    print("=" * 70)
    print("CNOT L1 LOGICAL ERROR COMPARISON")
    print("=" * 70)
    print()
    
    p = 0.001
    num_shots = 5000
    
    print(f"Test parameters: p={p}, num_shots={num_shots}")
    print()
    
    # Initialize the original module's required global variables
    orig_steane.N_steane = 7
    orig_steane.gamma = p / 10
    orig_steane.check_matrix = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])
    orig_steane.logical_op = np.array([1, 1, 1, 0, 0, 0, 0])
    
    # Original implementation
    print("Running original implementation...")
    try:
        orig_error, orig_var = orig_steane.estimate_logical_cnot_error_l1(p, num_shots)
        print(f"  Original: logical_error={orig_error:.6f}, variance={orig_var:.6e}")
    except Exception as e:
        print(f"  Original FAILED: {e}")
        import traceback
        traceback.print_exc()
        orig_error = None
    
    print()
    
    # V10 implementation
    print("Running v10 implementation...")
    try:
        from qectostim.experiments.concatenated_css_v10_steane import create_steane_simulator
        from qectostim.noise.models import CircuitDepolarizingNoise
        
        # Create noise model (error_model 'a' uses gamma = p/10)
        noise_model = CircuitDepolarizingNoise(p, p / 10)
        
        # Create simulator for level-1
        simulator = create_steane_simulator(1, noise_model=noise_model)
        new_error, new_var = simulator.estimate_logical_cnot_error_l1(p, num_shots)
        print(f"  V10: logical_error={new_error:.6f}, variance={new_var:.6e}")
    except Exception as e:
        print(f"  V10 FAILED: {e}")
        import traceback
        traceback.print_exc()
        new_error = None
    
    print()
    
    # Compare results
    print("-" * 70)
    print("COMPARISON")
    print("-" * 70)
    
    if orig_error is not None and new_error is not None:
        diff = abs(orig_error - new_error)
        max_err = max(orig_error, new_error, 1e-10)
        rel_diff = diff / max_err * 100
        
        print(f"  Absolute difference: {diff:.6f}")
        print(f"  Relative difference: {rel_diff:.2f}%")
        
        if rel_diff < 50:  # Allow up to 50% due to stochastic nature
            print("✓ Results are within expected statistical variance!")
        else:
            print("✗ Results differ significantly - may indicate implementation mismatch")
    else:
        print("✗ Cannot compare - one or both implementations failed")
    
    return orig_error, new_error


# =============================================================================
# CNOT L2 Comparison  
# =============================================================================

def run_cnot_l2_comparison():
    """Compare estimate_logical_cnot_error_l2 between original and v10."""
    print("=" * 70)
    print("CNOT L2 LOGICAL ERROR COMPARISON")
    print("=" * 70)
    print()
    
    p = 0.001
    num_shots = 5000
    
    print(f"Test parameters: p={p}, num_shots={num_shots}")
    print()
    
    # Initialize the original module's required global variables
    orig_steane.N_steane = 7
    orig_steane.gamma = p / 10
    orig_steane.check_matrix = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])
    orig_steane.logical_op = np.array([1, 1, 1, 0, 0, 0, 0])
    
    # Propagation tables for level-2
    orig_steane.propagation_l2_0prep_X = [
        [0,2,4,6],[1,2,5,6],[2],[3,4,5,6],[4,6],[5],
        [0,2],[1,5,6],[2],[3,4,6],[4,6],[5],[6],
        [0],[1,5],[2],[3],[4,6],[5],[6],[],
        [0],[1],[2],[3],[4],[5],[6],[],
        [0],[1],[2],[3],[4],[5],[6],[],
        [0],[1],[2],[3],[4],[5],[6],[]
    ]
    
    orig_steane.propagation_l2_0prep_Z = [
        [0],[1],[2,0,1],[3],[4,3,0],[5,1,3],
        [0],[1],[2,0],[3],[4,3],[5,1],[6,1,4],
        [0],[1],[2],[3],[4],[5,1],[6,4],[2,4,5],
        [0],[1],[2],[3],[4],[5],[6],[4,5],
        [0],[1],[2],[3],[4],[5],[6],[5],
        [0],[1],[2],[3],[4],[5],[6],[]
    ]
    
    orig_steane.propagation_l2_0prep_m = [2,4,5,6,7,8,9,10,11,14,15,17,18,20,25,26,28,34,36,44]
    orig_steane.num_ec_0prep = 45
    
    # Original implementation
    print("Running original implementation...")
    try:
        orig_error, orig_var = orig_steane.estimate_logical_cnot_error_l2(p, num_shots)
        print(f"  Original: logical_error={orig_error:.6f}, variance={orig_var:.6e}")
    except Exception as e:
        print(f"  Original FAILED: {e}")
        import traceback
        traceback.print_exc()
        orig_error = None
    
    print()
    
    # V10 implementation
    print("Running v10 implementation...")
    try:
        from qectostim.experiments.concatenated_css_v10_steane import create_steane_simulator
        from qectostim.noise.models import CircuitDepolarizingNoise
        
        # Create noise model (error_model 'a' uses gamma = p/10)
        noise_model = CircuitDepolarizingNoise(p, p / 10)
        
        # Create simulator for level-2
        simulator = create_steane_simulator(2, noise_model=noise_model)
        new_error, new_var = simulator.estimate_logical_cnot_error_l2(p, num_shots)
        print(f"  V10: logical_error={new_error:.6f}, variance={new_var:.6e}")
    except Exception as e:
        print(f"  V10 FAILED: {e}")
        import traceback
        traceback.print_exc()
        new_error = None
    
    print()
    
    # Compare results
    print("-" * 70)
    print("COMPARISON")
    print("-" * 70)
    
    if orig_error is not None and new_error is not None:
        diff = abs(orig_error - new_error)
        max_err = max(orig_error, new_error, 1e-10)
        rel_diff = diff / max_err * 100
        
        print(f"  Absolute difference: {diff:.6f}")
        print(f"  Relative difference: {rel_diff:.2f}%")
        
        if rel_diff < 50:  # Allow up to 50% due to stochastic nature
            print("✓ Results are within expected statistical variance!")
        else:
            print("✗ Results differ significantly - may indicate implementation mismatch")
    else:
        print("✗ Cannot compare - one or both implementations failed")
    
    return orig_error, new_error


# =============================================================================
# Memory L1 Comparison
# =============================================================================

def run_memory_l1_comparison():
    """Compare estimate_memory_logical_error_l1 between v10 implementations.
    
    Note: The original concatenated_steane.py doesn't have an L1 memory function,
    so this just validates the v10 implementation runs correctly.
    """
    print("=" * 70)
    print("MEMORY L1 LOGICAL ERROR - V10 VALIDATION")
    print("=" * 70)
    print()
    
    p = 0.001
    num_shots = 5000
    
    print(f"Test parameters: p={p}, num_shots={num_shots}")
    print("Note: Original implementation doesn't have memory L1, so this is a V10-only validation")
    print()
    
    # V10 implementation
    print("Running v10 implementation...")
    try:
        from qectostim.experiments.concatenated_css_v10_steane import create_steane_simulator
        from qectostim.noise.models import CircuitDepolarizingNoise
        
        # Create noise model (error_model 'a' uses gamma = p/10)
        noise_model = CircuitDepolarizingNoise(p, p / 10)
        
        # Create simulator for level-1
        simulator = create_steane_simulator(1, noise_model=noise_model)
        new_error, new_var = simulator.estimate_memory_logical_error_l1(p, num_shots)
        print(f"  V10: logical_error={new_error:.6f}, variance={new_var:.6e}")
        print("✓ V10 memory L1 estimation completed successfully!")
        return new_error
    except Exception as e:
        print(f"  V10 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None



def run_memory_l2_comparison():
    """Compare estimate_memory_logical_error_l2 between original and v10."""
    print("=" * 70)
    print("MEMORY L2 LOGICAL ERROR COMPARISON")
    print("=" * 70)
    print()
    
    p = 0.001
    num_shots = 500_000  # More shots for better statistics
    
    print(f"Test parameters: p={p}, num_shots={num_shots}")
    print()
    
    # Initialize the original module's required global variables
    # These are normally set in the if __name__ == '__main__' block
    orig_steane.N_steane = 7
    orig_steane.gamma = p / 10  # Error model 'a'
    orig_steane.check_matrix = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])
    orig_steane.logical_op = np.array([1, 1, 1, 0, 0, 0, 0])
    
    # Propagation tables for level-2
    orig_steane.propagation_l2_0prep_X = [
        [0,2,4,6],[1,2,5,6],[2],[3,4,5,6],[4,6],[5],
        [0,2],[1,5,6],[2],[3,4,6],[4,6],[5],[6],
        [0],[1,5],[2],[3],[4,6],[5],[6],[],
        [0],[1],[2],[3],[4],[5],[6],[],
        [0],[1],[2],[3],[4],[5],[6],[],
        [0],[1],[2],[3],[4],[5],[6],[]
    ]
    
    orig_steane.propagation_l2_0prep_Z = [
        [0],[1],[2,0,1],[3],[4,3,0],[5,1,3],
        [0],[1],[2,0],[3],[4,3],[5,1],[6,1,4],
        [0],[1],[2],[3],[4],[5,1],[6,4],[2,4,5],
        [0],[1],[2],[3],[4],[5],[6],[4,5],
        [0],[1],[2],[3],[4],[5],[6],[5],
        [0],[1],[2],[3],[4],[5],[6],[]
    ]
    
    orig_steane.propagation_l2_0prep_m = [2,4,5,6,7,8,9,10,11,14,15,17,18,20,25,26,28,34,36,44]
    orig_steane.num_ec_0prep = 45
    
    # Original implementation
    print("Running original implementation...")
    try:
        orig_error, orig_var = orig_steane.estimate_memory_logical_error_l2(p, num_shots)
        print(f"  Original: logical_error={orig_error:.6f}, variance={orig_var:.6e}")
    except Exception as e:
        print(f"  Original FAILED: {e}")
        import traceback
        traceback.print_exc()
        orig_error = None
    
    print()
    
    # V10 implementation
    print("Running v10 implementation...")
    try:
        from qectostim.experiments.concatenated_css_v10_steane import create_steane_simulator
        from qectostim.noise.models import CircuitDepolarizingNoise
        
        # Create noise model (error_model 'a' uses gamma = p/10)
        noise_model = CircuitDepolarizingNoise(p, p / 10)
        
        # Create simulator for level-2 (L2 memory test requires level 2!)
        simulator = create_steane_simulator(2, noise_model=noise_model)
        new_error, new_var = simulator.estimate_memory_logical_error_l2(p, num_shots)
        print(f"  V10: logical_error={new_error:.6f}, variance={new_var:.6e}")
        print("✓ V10 memory L2 estimation completed successfully!")
    except Exception as e:
        print(f"  V10 FAILED: {e}")
        import traceback
        traceback.print_exc()
        new_error = None
    
    print()
    
    # Compare results
    print("-" * 70)
    print("COMPARISON")
    print("-" * 70)
    
    if orig_error is not None and new_error is not None:
        diff = abs(orig_error - new_error)
        rel_diff = diff / max(orig_error, new_error, 1e-10) * 100
        
        print(f"  Absolute difference: {diff:.6f}")
        print(f"  Relative difference: {rel_diff:.2f}%")
        
        # Due to stochastic nature, allow some tolerance
        if rel_diff < 50:  # Allow up to 50% relative difference due to random sampling
            print("✓ Results are within expected statistical variance!")
        else:
            print("✗ Results differ significantly - may indicate implementation mismatch")
    else:
        print("✗ Cannot compare - one or both implementations failed")
    
    return orig_error, new_error





if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--minimal':
            run_minimal_comparison()
        elif sys.argv[1] == '--noisy-prep':
            run_noisy_prep_comparison()
        elif sys.argv[1] == '--ec':
            run_ec_comparison()
        elif sys.argv[1] == '--memory-l2':
            run_memory_l2_comparison()
        elif sys.argv[1] == '--cnot-l1':
            run_cnot_l1_comparison()
        elif sys.argv[1] == '--cnot-l2':
            run_cnot_l2_comparison()
        elif sys.argv[1] == '--memory-l1':
            run_memory_l1_comparison()
        elif sys.argv[1] == '--all':
            run_minimal_comparison()
            print("\n" * 2)
            run_noisy_prep_comparison()
            print("\n" * 2)
            run_ec_comparison()
            print("\n" * 2)
            run_comparison_tests()
            print("\n" * 2)
            run_memory_l2_comparison()
            print("\n" * 2)
            run_cnot_l1_comparison()
            print("\n" * 2)
            run_cnot_l2_comparison()
            print("\n" * 2)
            run_memory_l1_comparison()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Options: --minimal, --noisy-prep, --ec, --memory-l2, --cnot-l1, --cnot-l2, --memory-l1, --all")
    else:
        run_comparison_tests()
