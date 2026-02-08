"""
Bell State XX Correlation Test (Diagnostic Gadget).

This module implements a Bell state verification experiment using XX measurements.
It serves as a diagnostic tool for checking entanglement generation and stabilizer
propagation, but DOES NOT implement a teleported Hadamard gate.

PROTOCOL:
1. Prepare Data in |0⟩_L, Ancilla in |+⟩_L
2. Apply transversal CNOT(Ancilla -> Data)
   - Creates Bell state |Φ+⟩_L = (|00⟩ + |11⟩)/√2
3. Measure both patches in X basis
4. Check XX correlation: X_L(Data) ⊗ X_L(Ancilla) = +1

This allows using standard Stim OBSERVABLE_INCLUDE because the correlation
operator is deterministic for the Bell state.

NOTE: This is NOT H-teleportation because it acts as the identity channel
(measure X, prepare X) rather than the Hadamard channel (measure X, prepare Z).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Tuple,
    TYPE_CHECKING,
)
from enum import Enum

import numpy as np
import stim

from qectostim.codes.abstract_code import Code
from qectostim.codes.abstract_css import CSSCode
from qectostim.gadgets.teleportation_two_dem import (
    TwoDEMCircuitBuilder,
    TwoDEMExperimentResult,
)

# Reuse the circuit builder since it already defaults to XX measurement logic
# in the current codebase state. We just wrapper it for clarity.

class BellXXCircuitBuilder(TwoDEMCircuitBuilder):
    """
    Circuit builder specifically for Bell State XX correlation tests.
    Inherits from TwoDEMCircuitBuilder which currently implements the XX logic.
    """
    pass

def run_xx_bell_experiment(
    code: CSSCode,
    p: float,
    num_shots: int = 10_000,
    num_ec_rounds: int = 1,
) -> TwoDEMExperimentResult:
    """
    Run the Bell State XX correlation experiment.
    
    Verifies that CNOT(Ancilla->Data) creates a Bell state |Φ+⟩ by checking
    the stabilizer X_L ⊗ X_L = +1.
    """
    import pymatching

    # 1. Build circuit
    builder = BellXXCircuitBuilder(code, p=p, num_ec_rounds=num_ec_rounds)
    circuit = builder.build_correlated_circuit()

    # 2. Build DEM (includes OBSERVABLE_INCLUDE for X_L ⊗ X_L)
    dem = circuit.detector_error_model(
        decompose_errors=True,
        ignore_decomposition_failures=True,
    )

    # 3. Decode
    # Handle p=0 case
    if dem.num_errors == 0:
        sampler = circuit.compile_sampler()
        samples = sampler.sample(shots=num_shots)
        
        # Check raw correlation directly
        raw_X_data = np.zeros(num_shots, dtype=np.uint8)
        raw_X_anc = np.zeros(num_shots, dtype=np.uint8)
        
        for q in builder.x_logical:
            raw_X_data ^= samples[:, builder._data_x_final_start + q].astype(np.uint8)
            raw_X_anc ^= samples[:, builder._ancilla_x_final_start + q].astype(np.uint8)
            
        naive_error = float(np.mean(raw_X_data ^ raw_X_anc))
        return TwoDEMExperimentResult(
            dem_d_error_rate=0.0,
            dem_a_error_rate=0.0,
            combined_error_rate=naive_error,
            naive_error_rate=naive_error,
            improvement_pct=0.0,
            num_shots=num_shots,
            code_distance=code.distance,
            physical_error_rate=p,
        )

    # Standard decoding
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    det_samples, obs_samples = sampler.sample(num_shots, separate_observables=True)
    
    predictions = matcher.decode_batch(det_samples)
    
    actual_obs = obs_samples[:, 0].astype(int)
    predicted_obs = predictions[:, 0]
    
    logical_errors = actual_obs != predicted_obs
    logical_error_rate = float(np.mean(logical_errors))

    return TwoDEMExperimentResult(
        dem_d_error_rate=float(np.mean(predictions[:, 0])),
        dem_a_error_rate=float(np.mean(actual_obs)),
        combined_error_rate=logical_error_rate,
        naive_error_rate=0.0, # Placeholder
        improvement_pct=0.0,
        num_shots=num_shots,
        code_distance=code.distance,
        physical_error_rate=p,
    )
