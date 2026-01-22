"""
Simple Concatenated Memory Experiment.

This implements the paper's approach:
1. Prepare |0_L⟩ for concatenated code
2. Apply idle noise
3. Measure all data qubits in Z basis
4. Decode hierarchically (inner first, then outer)

NO mid-circuit syndrome extraction - just prepare → wait → measure.
This is designed to work with HardDecisionHierarchicalDecoder.
"""
from __future__ import annotations

import stim
import numpy as np
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode
    from qectostim.noise.models import NoiseModel


@dataclass
class SimpleConcatenatedMetadata:
    """Metadata for simple concatenated memory experiment."""
    inner_code_name: str
    outer_code_name: str
    n_inner: int
    n_outer: int
    n_total_data: int
    inner_z_support: List[int]
    outer_z_support: List[int]
    final_data_measurements: Dict[int, List[int]]  # block_id -> measurement indices
    total_measurements: int
    

class SimpleConcatenatedMemoryExperiment:
    """
    Simple memory experiment for concatenated codes.
    
    Structure:
    - 7 inner blocks (for Steane outer)
    - Each block has n_inner data qubits
    - Total: 7 × n_inner data qubits
    
    Circuit flow:
    1. Prepare |0⟩ on all data qubits (inner code encodes this to |0_L⟩)
    2. Apply depolarizing noise (simulates idle errors)
    3. Measure all data qubits in Z basis
    
    The decoder then:
    - Extracts measurements for each inner block
    - Decodes each inner block → 7 logical values
    - Decodes outer Steane using those 7 values
    """
    
    def __init__(
        self,
        inner_code: "CSSCode",
        outer_code: "CSSCode",
        noise_model: Optional["NoiseModel"] = None,
        p_error: float = 0.001,
    ):
        """
        Initialize simple concatenated memory experiment.
        
        Parameters
        ----------
        inner_code : CSSCode
            Inner code (e.g., Shor [[9,1,3]] or Steane [[7,1,3]])
        outer_code : CSSCode
            Outer code (typically Steane [[7,1,3]])
        noise_model : NoiseModel, optional
            Noise model to apply. If None, uses simple depolarizing.
        p_error : float
            Error probability for simple depolarizing noise
        """
        self.inner_code = inner_code
        self.outer_code = outer_code
        self.noise_model = noise_model
        self.p_error = p_error
        
        # Code parameters
        self.n_inner = inner_code.n
        self.n_outer = outer_code.n  # Number of blocks
        self.n_total_data = self.n_inner * self.n_outer
        
        # Get logical supports
        self.inner_z_support = self._get_z_support(inner_code)
        self.outer_z_support = self._get_z_support(outer_code)
        
        # Qubit allocation
        self._data_qubits = list(range(self.n_total_data))
        
    def _get_z_support(self, code: "CSSCode") -> List[int]:
        """Get Z logical operator support."""
        if hasattr(code, 'logical_z_support'):
            try:
                return list(code.logical_z_support(0))
            except:
                pass
        
        # Try parsing logical_z_ops
        if hasattr(code, 'logical_z_ops'):
            ops = code.logical_z_ops
            if callable(ops):
                ops = ops()
            if ops and len(ops) > 0:
                op = ops[0]
                if isinstance(op, str):
                    return [i for i, c in enumerate(op) if c in ('Z', 'Y')]
        
        # For Shor code [[9,1,3]], all 9 qubits are in Z_L support
        code_name = getattr(code, 'name', '') or str(type(code).__name__)
        if 'shor' in code_name.lower() or 'Shor' in code_name:
            return list(range(code.n))
        
        # For Steane code, Z_L = {0,1,2}
        if 'steane' in code_name.lower() or 'Stean' in code_name:
            return [0, 1, 2]
        
        # Fallback
        return list(range(min(3, code.n)))
    
    def build(self) -> Tuple[stim.Circuit, SimpleConcatenatedMetadata]:
        """
        Build the Stim circuit and metadata.
        
        Returns
        -------
        circuit : stim.Circuit
            The Stim circuit
        metadata : SimpleConcatenatedMetadata
            Metadata for decoder
        """
        circuit = stim.Circuit()
        
        # 1. Prepare |0⟩ on all data qubits
        # (For CSS code in Z basis, |0_L⟩ is prepared by starting in |0⟩^n
        #  and the stabilizers project onto the code space)
        circuit.append("R", self._data_qubits)
        
        # 2. Apply noise (idle errors)
        if self.noise_model is not None:
            # Use provided noise model
            circuit = self.noise_model.apply(circuit)
        elif self.p_error > 0:
            # Simple depolarizing noise on each qubit
            circuit.append("DEPOLARIZE1", self._data_qubits, self.p_error)
        
        # 3. Measure all data qubits in Z basis
        circuit.append("M", self._data_qubits)
        
        # 4. Add observable (outer Z_L)
        self._add_observable(circuit)
        
        # Build metadata
        final_data_meas = {}
        for block_id in range(self.n_outer):
            start_idx = block_id * self.n_inner
            final_data_meas[block_id] = list(range(start_idx, start_idx + self.n_inner))
        
        metadata = SimpleConcatenatedMetadata(
            inner_code_name=getattr(self.inner_code, 'name', 'unknown'),
            outer_code_name=getattr(self.outer_code, 'name', 'unknown'),
            n_inner=self.n_inner,
            n_outer=self.n_outer,
            n_total_data=self.n_total_data,
            inner_z_support=self.inner_z_support,
            outer_z_support=self.outer_z_support,
            final_data_measurements=final_data_meas,
            total_measurements=self.n_total_data,
        )
        
        return circuit, metadata
    
    def _add_observable(self, circuit: stim.Circuit) -> None:
        """
        Add OBSERVABLE_INCLUDE for concatenated Z_L.
        
        For concatenated code:
        - Outer Z_L acts on blocks in outer_z_support (e.g., {0,1,2})
        - Inner Z_L acts on qubits in inner_z_support within each block
        - Observable = XOR of all these measurements
        """
        obs_targets = []
        
        for block_id in self.outer_z_support:
            # For each block in outer logical support
            block_start = block_id * self.n_inner
            
            for inner_qubit in self.inner_z_support:
                # For each qubit in inner logical support
                meas_idx = block_start + inner_qubit
                # Lookback from end of measurements
                lookback = meas_idx - self.n_total_data
                obs_targets.append(stim.target_rec(lookback))
        
        if obs_targets:
            circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)
    
    def to_stim(self) -> stim.Circuit:
        """Build and return just the circuit."""
        circuit, _ = self.build()
        return circuit
    
    def get_decoder_metadata(self) -> Dict[str, Any]:
        """Get metadata as dict for decoder."""
        _, metadata = self.build()
        return {
            'inner_code': self.inner_code,
            'outer_code': self.outer_code,
            'n_inner': metadata.n_inner,
            'n_outer': metadata.n_outer,
            'n_data_blocks': metadata.n_outer,
            'inner_z_support': metadata.inner_z_support,
            'outer_z_support': metadata.outer_z_support,
            'final_data_measurements': metadata.final_data_measurements,
            'total_measurements': metadata.total_measurements,
        }


class PaperStyleConcatenatedMemory(SimpleConcatenatedMemoryExperiment):
    """
    Memory experiment matching the paper's "Concatenated codes, save qubits" setup.
    
    Uses Steane outer code with configurable inner code.
    Designed to work with HardDecisionHierarchicalDecoder.
    """
    
    def __init__(
        self,
        inner_code: "CSSCode",
        p_error: float = 0.001,
        outer_code: Optional["CSSCode"] = None,
    ):
        """
        Initialize paper-style memory experiment.
        
        Parameters
        ----------
        inner_code : CSSCode
            Inner code (Shor [[9,1,3]] recommended for PyMatching compatibility)
        p_error : float
            Physical error probability
        outer_code : CSSCode, optional
            Outer code. Defaults to Steane [[7,1,3]].
        """
        if outer_code is None:
            # Import Steane code
            try:
                from qectostim.codes.small.steane_713 import SteaneCode713
                outer_code = SteaneCode713()
            except ImportError:
                raise ValueError("Could not import SteaneCode713. Please provide outer_code.")
        
        super().__init__(
            inner_code=inner_code,
            outer_code=outer_code,
            noise_model=None,
            p_error=p_error,
        )


def run_simple_concatenated_experiment(
    inner_code: "CSSCode",
    outer_code: "CSSCode",
    p_error: float = 0.001,
    n_shots: int = 1000,
    use_inner_correction: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a simple concatenated memory experiment with hard-decision decoding.
    
    Parameters
    ----------
    inner_code : CSSCode
        Inner code
    outer_code : CSSCode  
        Outer code
    p_error : float
        Physical error probability
    n_shots : int
        Number of shots to sample
    use_inner_correction : bool
        Whether to apply inner code error correction
    verbose : bool
        Print debug info
        
    Returns
    -------
    dict with:
        - logical_error_rate: fraction of shots with wrong logical value
        - raw_error_rate: error rate without outer correction
        - n_shots: number of shots
        - n_errors: number of logical errors
    """
    from qectostim.decoders.hard_decision_hierarchical_decoder import (
        HardDecisionHierarchicalDecoder,
        HardDecisionConfig,
    )
    
    # Build experiment
    exp = SimpleConcatenatedMemoryExperiment(
        inner_code=inner_code,
        outer_code=outer_code,
        p_error=p_error,
    )
    circuit, metadata = exp.build()
    decoder_metadata = exp.get_decoder_metadata()
    
    if verbose:
        print(f"Circuit: {circuit.num_qubits} qubits, {circuit.num_measurements} measurements")
        print(f"Inner code: {decoder_metadata['n_inner']} qubits")
        print(f"Outer code: {decoder_metadata['n_outer']} blocks")
        print(f"Inner Z_L support: {decoder_metadata['inner_z_support']}")
        print(f"Outer Z_L support: {decoder_metadata['outer_z_support']}")
    
    # Sample
    sampler = circuit.compile_sampler()
    samples = sampler.sample(n_shots)
    
    # Build decoder
    config = HardDecisionConfig(
        verbose=False,
        use_inner_correction=use_inner_correction,
    )
    decoder = HardDecisionHierarchicalDecoder(decoder_metadata, config)
    
    # Decode
    n_errors = 0
    n_raw_errors = 0
    
    for i in range(n_shots):
        shot = samples[i]
        
        # Decode
        prediction = decoder.decode(shot)
        
        # Expected value is 0 (prepared |0_L⟩)
        if prediction != 0:
            n_errors += 1
        
        # Also compute raw error (no outer correction)
        # Raw = XOR of inner logicals in outer support
        inner_logicals = decoder._decode_inner_blocks(
            decoder._extract_block_measurements(shot)
        )
        raw_logical = 0
        for b in decoder_metadata['outer_z_support']:
            raw_logical ^= inner_logicals.get(b, 0)
        if raw_logical != 0:
            n_raw_errors += 1
    
    return {
        'logical_error_rate': n_errors / n_shots,
        'raw_error_rate': n_raw_errors / n_shots,
        'n_shots': n_shots,
        'n_errors': n_errors,
        'n_raw_errors': n_raw_errors,
    }
