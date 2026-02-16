"""
Example: Using OldTrappedIonExperiment adapter with QECToStim codes.

This example demonstrates how to use the old/ trapped ion simulator
with the QECToStim framework via the OldTrappedIonExperiment adapter.
"""
import sys
sys.path.insert(0, '/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/QECToStim/src')

# Import QECToStim code - Steane [[7,1,3]] code
from qectostim.codes.small.steane_713 import SteaneCode713

# Import the adapter
from qectostim.experiments.hardware_simulation.old.trapped_ion.simulator.trapped_ion_experiment import (
    OldTrappedIonExperiment,
    WISEArchConfig,
    create_wise_experiment,
)


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic trapped ion experiment")
    print("=" * 60)
    
    # Create a Steane [[7,1,3]] code
    code = SteaneCode713()
    
    # Configure WISE architecture
    config = WISEArchConfig(
        n=2,      # 2 rows
        m=2,      # 2 columns  
        k=5,      # 5 ions per manipulation zone
        lookahead=2,
        subgridsize=(6, 4, 1),
    )
    
    # Create experiment
    exp = OldTrappedIonExperiment(
        code=code,
        arch_config=config,
        rounds=1,
        error_scaling=1.0,
    )
    
    print(f"Code: {code.__class__.__name__}")
    print(f"Architecture: WISE {config.n}x{config.m}, k={config.k}")
    print()
    
    # Get the stim circuit
    try:
        stim_circuit = exp.to_stim()
        print(f"Circuit generated: {len(stim_circuit)} instructions")
    except Exception as e:
        print(f"Circuit generation skipped: {e}")
    
    print()


def example_convenience_function():
    """Example using convenience function."""
    print("=" * 60)
    print("Example 2: Using create_wise_experiment()")
    print("=" * 60)
    
    code = SteaneCode713()
    
    # Simple one-liner
    exp = create_wise_experiment(
        code=code,
        n=2, m=2, k=5,
        rounds=1,
        error_scaling=1.0,
    )
    
    print(f"Experiment created with default WISE config")
    print()


def example_simulation():
    """Example running simulation."""
    print("=" * 60)
    print("Example 3: Running simulation")
    print("=" * 60)
    
    code = SteaneCode713()
    config = WISEArchConfig(n=2, m=2, k=5)
    
    exp = OldTrappedIonExperiment(
        code=code,
        arch_config=config,
        rounds=1,
    )
    
    # Run simulation
    print("Running simulation with 1000 shots...")
    try:
        result = exp.simulate(
            num_shots=1000,
            decode=True,
            use_timing_aware_noise=True,
        )
        
        print(f"Logical error rate: {result.logical_error_rate:.4f}")
        print(f"Physical X error:   {result.mean_physical_x_error:.4f}")
        print(f"Physical Z error:   {result.mean_physical_z_error:.4f}")
        print(f"Operations:         {result.compilation_metrics.get('num_operations', 0)}")
        print(f"Decoder used:       {result.decoder_used}")
    except Exception as e:
        print(f"Simulation skipped: {e}")
    
    print()


def example_architecture_inspection():
    """Example inspecting compiled architecture."""
    print("=" * 60)
    print("Example 4: Architecture inspection")
    print("=" * 60)
    
    code = SteaneCode713()
    config = WISEArchConfig(n=2, m=2, k=5)
    
    exp = OldTrappedIonExperiment(
        code=code,
        arch_config=config,
        rounds=1,
    )
    
    # Access architecture after compilation
    try:
        _ = exp.to_stim()  # Triggers compilation
        
        print(f"Data qubits:    {sorted(exp.data_qubit_indices)}")
        print(f"Ancilla qubits: {sorted(exp.ancilla_qubit_indices)}")
        print(f"Qubit roles:    {exp.qubit_roles}")
    except Exception as e:
        print(f"Architecture inspection skipped: {e}")
    
    print()


if __name__ == "__main__":
    print("\nOldTrappedIonExperiment Adapter Examples")
    print("=" * 60)
    print()
    
    example_basic_usage()
    example_convenience_function()
    example_simulation()
    example_architecture_inspection()
    
    print("Done!")
