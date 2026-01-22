"""
Shor x Steane Concatenated Code Memory Experiment Template.

This template demonstrates how to use the detector-based architecture to measure
the logical error rate of a Shor [[9,1,3]] ⊗ Steane [[7,1,3]] concatenated code
memory experiment.

Architecture Summary (from reference code):
------------------------------------------
1. **Gadgets emit DETECTORs** directly after each measurement block
2. **No OBSERVABLE_INCLUDE** - all logical outcomes computed in Python
3. **Detector sampler** - use circuit.compile_detector_sampler().sample()
4. **Local hard-decision decoding** - per-gadget lookup tables, no global MWPM
5. **Post-selection via verification** - reject shots with verification parity ≠ 0
6. **Ambiguous handling** - decoder returns success=False, confidence=0.5 for
   uncorrectable patterns; either reject or count as 0.5 error contribution
7. **Corrections accumulated in software** - no feed-forward in circuit

Code Structure:
--------------
- Outer code: Shor [[9,1,3]] - 9 logical qubits from inner level
- Inner code: Steane [[7,1,3]] - 7 physical qubits per inner block
- Total physical qubits: 9 × 7 = 63
- Total distance: 3 × 3 = 9

Usage:
------
    python shor_x_steane_template.py

    # Or import and use directly:
    from shor_x_steane_template import run_shor_x_steane_experiment
    results = run_shor_x_steane_experiment(p=1e-3, rounds=3, shots=10000)
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Core imports
from qectostim.codes.small import SteaneCode713, ShorCode91
from qectostim.codes.composite.multilevel_concatenated import MultiLevelConcatenatedCode
from qectostim.experiments.detector_multilevel_memory import (
    DetectorMultiLevelMemory,
    ExperimentResult,
    estimate_logical_error_rate,
    run_memory_sweep,
)
from qectostim.noise.models import CircuitDepolarizingNoise


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default experiment parameters
DEFAULT_ROUNDS = 3       # Number of EC rounds (Q)
DEFAULT_SHOTS = 10000    # Monte Carlo samples
DEFAULT_P = 1e-3         # Physical error probability

# Noise sweep range
P_MIN = 1e-4
P_MAX = 5e-2
N_POINTS = 10


# =============================================================================
# CODE CONSTRUCTION
# =============================================================================

def make_shor_x_steane_code() -> MultiLevelConcatenatedCode:
    """
    Construct Shor [[9,1,3]] ⊗ Steane [[7,1,3]] concatenated code.
    
    Structure:
    - Outer code: Shor [[9,1,3]] - protects against X and Z errors at logical level
    - Inner code: Steane [[7,1,3]] - provides physical error correction
    
    The concatenation order is:
        Shor (outer) ⊗ Steane (inner)
    
    This means:
    - 9 inner Steane blocks (one for each Shor data qubit)
    - Each inner block has 7 physical qubits
    - Total: 63 physical qubits
    - Distance: d = 3 × 3 = 9
    
    Returns:
        MultiLevelConcatenatedCode: The concatenated code
    """
    shor = ShorCode91()
    steane = SteaneCode713()
    
    # Order: [outer, inner] = [Shor, Steane]
    return MultiLevelConcatenatedCode([shor, steane])


def make_steane_x_shor_code() -> MultiLevelConcatenatedCode:
    """
    Construct Steane [[7,1,3]] ⊗ Shor [[9,1,3]] concatenated code (alternative order).
    
    Structure:
    - Outer code: Steane [[7,1,3]]
    - Inner code: Shor [[9,1,3]]
    
    Total: 7 × 9 = 63 physical qubits (same as above, but different structure)
    
    Returns:
        MultiLevelConcatenatedCode: The concatenated code
    """
    steane = SteaneCode713()
    shor = ShorCode91()
    
    return MultiLevelConcatenatedCode([steane, shor])


# =============================================================================
# EXPERIMENT SETUP
# =============================================================================

def create_experiment(
    code: MultiLevelConcatenatedCode,
    p: float,
    rounds: int = DEFAULT_ROUNDS,
    basis: str = 'Z',
    ancilla_prep: str = 'encoded',
    reject_on_ambiguity: bool = False,
) -> DetectorMultiLevelMemory:
    """
    Create a detector-based memory experiment for the concatenated code.
    
    Args:
        code: The concatenated code
        p: Physical error probability (depolarizing noise)
        rounds: Number of EC rounds
        basis: Measurement basis ('Z' for |0_L⟩, 'X' for |+_L⟩)
        ancilla_prep: Ancilla preparation method
            - 'bare': Simple reset (NOT fault-tolerant)
            - 'encoded': Proper encoding (fault-tolerant)
            - 'verified': Encoding + verification (fully FT with post-selection)
        reject_on_ambiguity: If True, reject ambiguous shots; if False, count as 0.5 error
        
    Returns:
        Configured DetectorMultiLevelMemory experiment
    """
    return DetectorMultiLevelMemory(
        code=code,
        rounds=rounds,
        p=p,
        basis=basis,
        initial_state='0' if basis == 'Z' else '+',
        ancilla_prep=ancilla_prep,
        reject_on_ambiguity=reject_on_ambiguity,
    )


# =============================================================================
# RUNNING EXPERIMENTS
# =============================================================================

def run_shor_x_steane_experiment(
    p: float = DEFAULT_P,
    rounds: int = DEFAULT_ROUNDS,
    shots: int = DEFAULT_SHOTS,
    basis: str = 'Z',
    ancilla_prep: str = 'encoded',
    verbose: bool = True,
) -> ExperimentResult:
    """
    Run a single Shor ⊗ Steane memory experiment.
    
    This is the main API for running memory experiments with the detector-based
    architecture.
    
    Args:
        p: Physical error probability
        rounds: Number of EC rounds
        shots: Number of Monte Carlo samples
        basis: Measurement basis ('Z' or 'X')
        ancilla_prep: Ancilla preparation method
        verbose: Print progress and results
        
    Returns:
        ExperimentResult with logical error rate and statistics
        
    Example:
        >>> result = run_shor_x_steane_experiment(p=1e-3, rounds=3, shots=10000)
        >>> print(f"Logical error rate: {result.logical_error_rate:.4e}")
    """
    if verbose:
        print("=" * 60)
        print("Shor ⊗ Steane Concatenated Code Memory Experiment")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Physical error rate (p): {p:.2e}")
        print(f"  EC rounds (Q): {rounds}")
        print(f"  Shots: {shots}")
        print(f"  Basis: {basis}")
        print(f"  Ancilla prep: {ancilla_prep}")
        print("-" * 60)
    
    # Build code
    code = make_shor_x_steane_code()
    
    if verbose:
        print(f"Code structure:")
        print(f"  Outer: Shor [[9,1,3]]")
        print(f"  Inner: Steane [[7,1,3]]")
        print(f"  Physical qubits: {code.n}")
        print(f"  Distance: {code.total_distance}")
        print("-" * 60)
    
    # Create experiment
    exp = create_experiment(code, p=p, rounds=rounds, basis=basis, ancilla_prep=ancilla_prep)
    
    if verbose:
        print("Building circuit...")
    
    # Run experiment
    result = exp.run(shots)
    
    if verbose:
        print("-" * 60)
        print("Results:")
        print(f"  Accepted shots: {result.n_accepted}/{result.n_shots}")
        print(f"  Rejected shots: {result.n_rejected}")
        print(f"  Acceptance rate: {result.acceptance_rate:.2%}")
        print(f"  Logical errors: {result.n_logical_errors}")
        print(f"  Ambiguous shots: {result.n_ambiguous}")
        print(f"  Logical error rate: {result.logical_error_rate:.4e} ± {result.logical_error_rate_std:.4e}")
        print("=" * 60)
    
    return result


def run_noise_sweep(
    p_values: list = None,
    rounds: int = DEFAULT_ROUNDS,
    shots_per_point: int = DEFAULT_SHOTS,
    verbose: bool = True,
) -> dict:
    """
    Sweep over physical error probabilities.
    
    This is useful for finding the pseudo-threshold where the concatenated
    code's logical error rate equals the physical error rate.
    
    Args:
        p_values: List of physical error probabilities (or auto-generate)
        rounds: Number of EC rounds
        shots_per_point: Samples per error probability
        verbose: Print progress
        
    Returns:
        Dict mapping p -> ExperimentResult
        
    Example:
        >>> results = run_noise_sweep(rounds=3, shots_per_point=5000)
        >>> for p, res in results.items():
        ...     print(f"p={p:.4f}: p_L={res.logical_error_rate:.4e}")
    """
    if p_values is None:
        p_values = np.logspace(np.log10(P_MIN), np.log10(P_MAX), N_POINTS)
    
    if verbose:
        print("=" * 60)
        print("Noise Sweep: Shor ⊗ Steane Concatenated Code")
        print("=" * 60)
        print(f"EC rounds: {rounds}")
        print(f"Shots per point: {shots_per_point}")
        print(f"Error probabilities: {len(p_values)} points from {p_values[0]:.2e} to {p_values[-1]:.2e}")
        print("-" * 60)
    
    code = make_shor_x_steane_code()
    results = run_memory_sweep(code, list(p_values), rounds, shots_per_point)
    
    if verbose:
        print("\nResults:")
        print("-" * 60)
        print(f"{'p':>10} | {'p_L':>12} | {'std':>10} | {'accept%':>8}")
        print("-" * 60)
        for p in sorted(results.keys()):
            res = results[p]
            print(f"{p:10.4e} | {res.logical_error_rate:12.4e} | {res.logical_error_rate_std:10.4e} | {res.acceptance_rate:8.1%}")
        print("=" * 60)
    
    return results


def find_pseudo_threshold(
    rounds: int = DEFAULT_ROUNDS,
    shots_per_point: int = DEFAULT_SHOTS,
    tolerance: float = 0.1,
) -> float:
    """
    Find the pseudo-threshold where p_L ≈ p.
    
    The pseudo-threshold is the physical error rate where the concatenated
    code's logical error rate equals the physical error rate. Below this
    threshold, the code provides benefit.
    
    Args:
        rounds: Number of EC rounds
        shots_per_point: Samples per point
        tolerance: Relative tolerance for convergence
        
    Returns:
        Estimated pseudo-threshold
    """
    print("=" * 60)
    print("Finding Pseudo-Threshold")
    print("=" * 60)
    
    # Binary search
    p_low, p_high = P_MIN, P_MAX
    code = make_shor_x_steane_code()
    
    for iteration in range(10):
        p_mid = np.sqrt(p_low * p_high)  # Geometric midpoint
        
        exp = create_experiment(code, p=p_mid, rounds=rounds)
        result = exp.run(shots_per_point)
        
        p_L = result.logical_error_rate
        ratio = p_L / p_mid
        
        print(f"  Iteration {iteration+1}: p={p_mid:.4e}, p_L={p_L:.4e}, ratio={ratio:.2f}")
        
        if abs(ratio - 1.0) < tolerance:
            print(f"\nPseudo-threshold found: p* ≈ {p_mid:.4e}")
            return p_mid
        
        if p_L > p_mid:
            p_high = p_mid
        else:
            p_low = p_mid
    
    p_threshold = np.sqrt(p_low * p_high)
    print(f"\nPseudo-threshold estimate: p* ≈ {p_threshold:.4e}")
    return p_threshold


# =============================================================================
# DETAILED ANALYSIS
# =============================================================================

def analyze_single_shot(p: float = DEFAULT_P, rounds: int = 1, verbose: bool = True):
    """
    Build circuit and analyze detector structure for a single configuration.
    
    This is useful for understanding the circuit structure and debugging.
    
    Args:
        p: Physical error probability
        rounds: Number of EC rounds
        verbose: Print detailed analysis
    """
    print("=" * 60)
    print("Circuit Structure Analysis")
    print("=" * 60)
    
    code = make_shor_x_steane_code()
    exp = create_experiment(code, p=p, rounds=rounds)
    
    circuit, metadata = exp.build()
    
    print(f"\nCode: {metadata.code_name}")
    print(f"Levels: {metadata.n_levels}")
    print(f"Level codes: {metadata.level_code_names}")
    print(f"Physical qubits: {metadata.n_physical_qubits}")
    print(f"Distance: {metadata.total_distance}")
    
    print(f"\nCircuit statistics:")
    print(f"  Total detectors: {metadata.total_detectors}")
    print(f"  Total measurements: {metadata.total_measurements}")
    print(f"  EC rounds: {len(metadata.ec_rounds)}")
    print(f"  Verification ranges: {len(metadata.verification_ranges)}")
    
    print(f"\nLogical operator support:")
    print(f"  Z_L on qubits: {metadata.z_logical_support}")
    print(f"  X_L on qubits: {metadata.x_logical_support}")
    
    if verbose:
        print(f"\nCircuit preview (first 50 lines):")
        circuit_str = str(circuit)
        lines = circuit_str.split('\n')[:50]
        for line in lines:
            print(f"  {line}")
        if len(circuit_str.split('\n')) > 50:
            print("  ...")
    
    # Sample a few shots to verify
    print(f"\nSampling 10 test shots...")
    detector_sampler = circuit.compile_detector_sampler()
    samples = detector_sampler.sample(10)
    
    print(f"  Sample shape: {samples.shape}")
    print(f"  Expected: (10, {metadata.total_detectors})")
    print(f"  Non-zero detectors per shot (avg): {np.mean(np.sum(samples, axis=1)):.1f}")
    
    return circuit, metadata


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Run example experiments demonstrating the detector-based architecture.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Shor x Steane Concatenated Code Memory Experiment"
    )
    parser.add_argument("--mode", choices=["single", "sweep", "threshold", "analyze"],
                       default="single", help="Experiment mode")
    parser.add_argument("--p", type=float, default=DEFAULT_P,
                       help="Physical error probability")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                       help="Number of EC rounds")
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS,
                       help="Number of Monte Carlo samples")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if args.mode == "single":
        result = run_shor_x_steane_experiment(
            p=args.p, rounds=args.rounds, shots=args.shots, verbose=verbose
        )
        
    elif args.mode == "sweep":
        results = run_noise_sweep(
            rounds=args.rounds, shots_per_point=args.shots, verbose=verbose
        )
        
    elif args.mode == "threshold":
        threshold = find_pseudo_threshold(
            rounds=args.rounds, shots_per_point=args.shots
        )
        print(f"\nPseudo-threshold: {threshold:.4e}")
        
    elif args.mode == "analyze":
        circuit, metadata = analyze_single_shot(
            p=args.p, rounds=args.rounds, verbose=verbose
        )


if __name__ == "__main__":
    main()
