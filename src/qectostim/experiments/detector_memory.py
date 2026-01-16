"""
Detector-Based Memory Experiment.

This module provides the main experiment class for the detector-based architecture.
Key design principles:

1. **Gadgets emit DETECTORs** - syndrome info captured as detector outcomes
2. **No OBSERVABLE_INCLUDE** - logical outcomes computed entirely in Python
3. **Detector sampler** - use circuit.compile_detector_sampler().sample()
4. **Local hard-decision decoding** - per-gadget decoders, no global MWPM
5. **Post-selection via detector groups** - verification detectors for FT

Circuit Structure:
-----------------
1. Prepare |0⟩^⊗n on all data qubits
2. Encode to |0_L⟩ (optional, or assume perfect prep)
3. For each EC round:
   - Apply EC gadget (emits syndrome DETECTORs + verification DETECTORs)
   - Apply idle noise between rounds
4. Final measurement of all data qubits
5. Emit final data DETECTORs

Sampling Flow:
-------------
    >>> circuit, metadata = experiment.build()
    >>> detector_sampler = circuit.compile_detector_sampler()
    >>> detector_samples = detector_sampler.sample(shots)
    >>> # detector_samples shape: (shots, n_detectors)
    >>> results = decode_samples(detector_samples, metadata)

Decoding Flow:
-------------
    >>> for shot in detector_samples:
    ...     # 1. Post-selection: reject if verification detectors fire
    ...     if not post_select(shot, metadata.verification_groups):
    ...         continue
    ...     # 2. Decode each EC round independently
    ...     corrections = []
    ...     for round_info in metadata.ec_rounds:
    ...         result = decoder.decode_syndrome(shot, round_info)
    ...         if result == -1:  # ambiguous
    ...             # Handle as 0.5 contribution or reject
    ...         corrections.append(result)
    ...     # 3. Decode final measurement
    ...     final = decoder.decode_final(shot, metadata.final_measurement)
    ...     # 4. Apply accumulated corrections in software
    ...     logical = apply_corrections(final, corrections)
    ...     # 5. Compare to expected (|0_L⟩ or |+_L⟩)
    ...     error = (logical != expected)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Union

import stim
import numpy as np

from qectostim.experiments.gadgets.detector_primitives import (
    DetectorMap, DetectorGroup, DetectorType, DetectorCounter,
    CircuitMetadata, ECRoundResult,
)
from qectostim.experiments.gadgets.detector_gadgets import (
    DetectorTransversalSyndrome, DetectorNoOp,
)
from qectostim.decoders.local_decoders import (
    LocalDecoder, LocalDecoderResult, HierarchicalDecoder,
    SteaneDecoder, ShorDecoder, get_decoder_for_code,
    post_select_verification, filter_shots_by_verification,
)

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode
    from qectostim.codes.composite.multilevel_concatenated import MultiLevelConcatenatedCode
    from qectostim.noise.models import NoiseModel


@dataclass
class ExperimentResult:
    """
    Result from running a detector-based experiment.
    
    Contains both raw data and computed statistics.
    """
    # Raw data
    n_shots: int
    n_accepted: int  # After post-selection
    n_rejected: int
    
    # Logical error statistics
    n_logical_errors: int
    logical_error_rate: float
    logical_error_rate_std: float
    
    # Fractional error contribution (for ambiguous shots)
    fractional_error_contribution: float = 0.0
    
    # Post-selection statistics
    rejection_rate: float = 0.0
    rejection_by_group: Dict[str, int] = field(default_factory=dict)
    
    # Ambiguity statistics  
    n_ambiguous: int = 0
    ambiguity_rate: float = 0.0
    
    # Decoder diagnostics
    decoder_info: Dict[str, Any] = field(default_factory=dict)


class DetectorMemoryExperiment:
    """
    Detector-based memory experiment for CSS codes.
    
    This experiment builds a Stim circuit that:
    1. Prepares logical |0_L⟩ or |+_L⟩
    2. Applies EC rounds (each emits DETECTORs)
    3. Measures all data qubits
    4. Does NOT use OBSERVABLE_INCLUDE
    
    Decoding is performed entirely in Python using local hard-decision decoders.
    
    Parameters
    ----------
    code : CSSCode
        The CSS code to use
    noise_model : NoiseModel, optional
        Noise model to apply (if None, uses noiseless)
    rounds : int
        Number of EC rounds (0 for simple memory with no EC)
    basis : str
        Measurement basis ('Z' or 'X')
    ancilla_prep : str
        Ancilla preparation method ('bare', 'encoded', 'verified')
    initial_state : str
        Initial logical state ('0' or '+')
    """
    
    def __init__(
        self,
        code: 'CSSCode',
        noise_model: Optional['NoiseModel'] = None,
        rounds: int = 0,
        basis: str = 'Z',
        ancilla_prep: str = 'encoded',
        initial_state: str = '0',
    ):
        self.code = code
        self.noise_model = noise_model
        self.rounds = rounds
        self.basis = basis.upper()
        self.ancilla_prep = ancilla_prep
        self.initial_state = initial_state
        
        # Create decoder
        self._decoder = self._create_decoder()
        
        # Create EC gadget
        self._ec_gadget = DetectorTransversalSyndrome(
            code=code,
            extract_x_syndrome=True,
            extract_z_syndrome=True,
            ancilla_prep=ancilla_prep,
            emit_individual_detectors=True,  # Emit syndrome-level detectors
        )
    
    def _create_decoder(self) -> LocalDecoder:
        """Create appropriate decoder for the code."""
        code_name = getattr(self.code, 'name', '').lower()
        
        if 'steane' in code_name or (hasattr(self.code, 'n') and self.code.n == 7):
            return SteaneDecoder()
        elif 'shor' in code_name or (hasattr(self.code, 'n') and self.code.n == 9):
            return ShorDecoder()
        else:
            # Try to get decoder by name
            try:
                return get_decoder_for_code(code_name)
            except ValueError:
                # Default to Steane
                return SteaneDecoder()
    
    def build(self) -> Tuple[stim.Circuit, CircuitMetadata]:
        """
        Build the Stim circuit and metadata.
        
        Returns:
            (circuit, metadata) where:
            - circuit: stim.Circuit with DETECTOR instructions
            - metadata: CircuitMetadata with detector group info for decoding
        """
        circuit = stim.Circuit()
        counter = DetectorCounter()
        metadata = CircuitMetadata()
        
        n = self.code.n
        data_qubits = {0: list(range(n))}  # Single block for non-concatenated
        
        # Track measurements
        current_meas = 0
        
        # === State Preparation ===
        circuit.append("R", list(range(n)))
        
        # Apply encoding circuit (simple for |0_L⟩, add H for |+_L⟩)
        if self.initial_state == '+':
            circuit.append("H", list(range(n)))
        
        # Apply encoding gates (for Steane code)
        self._apply_encoding(circuit, list(range(n)))
        
        # === EC Rounds ===
        for round_idx in range(self.rounds):
            # Add idle noise before EC
            if self.noise_model is not None and round_idx > 0:
                self._apply_idle_noise(circuit, list(range(n)))
            
            # Emit EC gadget
            dmap = self._ec_gadget.emit(
                circuit=circuit,
                data_qubits=data_qubits,
                counter=counter,
                noise_model=self.noise_model,
                measurement_offset=current_meas,
                round_idx=round_idx,
            )
            
            current_meas += dmap.total_measurements
            
            # Record EC round
            ec_round = ECRoundResult(
                round_idx=round_idx,
                detector_map=dmap,
            )
            metadata.add_ec_round(ec_round)
        
        # === Final Measurement ===
        # NOTE: These detectors compare final data measurements to 0, which is
        # not correct for encoded states (e.g., Steane |0_L⟩ is a superposition).
        # For accurate logical error detection, should use OBSERVABLE_INCLUDE.
        # Current implementation gives high false positive rate.
        circuit.append("M", list(range(n)))
        
        final_meas_indices = list(range(current_meas, current_meas + n))
        current_meas += n
        
        # Emit final data detectors
        final_dmap = DetectorMap(offset=counter.peek(), measurement_offset=current_meas - n)
        det_start = counter.peek()
        
        for i, meas_idx in enumerate(final_meas_indices):
            lookback = meas_idx - current_meas
            circuit.append("DETECTOR", [stim.target_rec(lookback)])
            counter.advance(1)
        
        det_end = counter.peek()
        
        final_group = DetectorGroup(
            start=det_start,
            end=det_end,
            dtype=DetectorType.FINAL_DATA,
            block_id=0,
            level=0,
        )
        final_dmap.add_group(final_group)
        final_dmap.total_detectors = det_end - det_start
        
        metadata.final_measurement = final_dmap
        
        # === Populate Metadata ===
        metadata.total_detectors = counter.current
        metadata.total_measurements = current_meas
        metadata.code_name = getattr(self.code, 'name', 'unknown')
        metadata.n_data_qubits = n
        metadata.initial_state = self.initial_state
        metadata.basis = self.basis
        
        # Logical operator support
        metadata.z_logical_support = self._get_z_support()
        metadata.x_logical_support = self._get_x_support()
        
        # Apply noise model if provided
        if self.noise_model is not None:
            circuit = self.noise_model.apply(circuit)
        
        return circuit, metadata
    
    def _apply_encoding(self, circuit: stim.Circuit, qubits: List[int]) -> None:
        """Apply encoding circuit for the code."""
        n = len(qubits)
        
        if n == 7:
            # Steane code encoding
            h_qubits = [qubits[i] for i in [0, 1, 3]]
            circuit.append("H", h_qubits)
            
            cnots = [
                (1, 2), (3, 5), (0, 4),
                (1, 6), (0, 2), (3, 4),
                (1, 5), (4, 6),
            ]
            for c, t in cnots:
                circuit.append("CNOT", [qubits[c], qubits[t]])
        
        elif n == 9:
            # Shor code encoding
            # |0⟩ → (|000⟩ + |111⟩)^⊗3 / 2√2
            circuit.append("H", [qubits[0], qubits[3], qubits[6]])
            circuit.append("CNOT", [qubits[0], qubits[1]])
            circuit.append("CNOT", [qubits[0], qubits[2]])
            circuit.append("CNOT", [qubits[3], qubits[4]])
            circuit.append("CNOT", [qubits[3], qubits[5]])
            circuit.append("CNOT", [qubits[6], qubits[7]])
            circuit.append("CNOT", [qubits[6], qubits[8]])
    
    def _apply_idle_noise(self, circuit: stim.Circuit, qubits: List[int]) -> None:
        """Apply idle noise to qubits."""
        if self.noise_model is not None:
            if hasattr(self.noise_model, 'idle_error_rate'):
                p = self.noise_model.idle_error_rate
                for q in qubits:
                    circuit.append("DEPOLARIZE1", [q], p)
    
    def _get_z_support(self) -> List[int]:
        """Get Z logical operator support."""
        if hasattr(self.code, 'z_logical') and self.code.z_logical:
            return list(np.where(self.code.z_logical[0])[0])
        # Default: all qubits for CSS codes
        return list(range(self.code.n))
    
    def _get_x_support(self) -> List[int]:
        """Get X logical operator support."""
        if hasattr(self.code, 'x_logical') and self.code.x_logical:
            return list(np.where(self.code.x_logical[0])[0])
        return list(range(self.code.n))
    
    def run(
        self,
        shots: int = 1000,
        verbose: bool = False,
    ) -> ExperimentResult:
        """
        Run the experiment and decode results.
        
        Args:
            shots: Number of shots to sample
            verbose: Print progress info
            
        Returns:
            ExperimentResult with error statistics
        """
        # Build circuit
        circuit, metadata = self.build()
        
        if verbose:
            print(f"Circuit built: {metadata.total_detectors} detectors, "
                  f"{metadata.total_measurements} measurements")
        
        # Sample using detector sampler
        detector_sampler = circuit.compile_detector_sampler()
        detector_samples = detector_sampler.sample(shots)
        
        if verbose:
            print(f"Sampled {shots} shots")
        
        # Decode
        return self._decode_samples(detector_samples, metadata, verbose)
    
    def _decode_samples(
        self,
        detector_samples: np.ndarray,
        metadata: CircuitMetadata,
        verbose: bool = False,
    ) -> ExperimentResult:
        """
        Decode detector samples and compute error statistics.
        
        Implements the local hard-decision decoding flow:
        1. Post-selection on verification detectors
        2. Per-round syndrome decoding
        3. Final measurement decoding
        4. Correction application
        5. Logical error computation
        """
        n_shots = len(detector_samples)
        n_accepted = 0
        n_rejected = 0
        n_logical_errors = 0
        n_ambiguous = 0
        fractional_error = 0.0
        
        # Get verification ranges for post-selection
        ver_ranges = metadata.get_all_verification_ranges()
        
        for shot_idx, shot in enumerate(detector_samples):
            # === Step 1: Post-selection ===
            if ver_ranges and not post_select_verification(shot, ver_ranges):
                n_rejected += 1
                continue
            
            n_accepted += 1
            
            # === Step 2: Decode EC rounds ===
            accumulated_correction_x = np.zeros(metadata.n_data_qubits, dtype=np.uint8)
            accumulated_correction_z = np.zeros(metadata.n_data_qubits, dtype=np.uint8)
            shot_ambiguous = False
            shot_confidence = 1.0
            
            for ec_round in metadata.ec_rounds:
                dmap = ec_round.detector_map
                
                # Extract Z syndrome detectors
                z_group = dmap.syndrome_z.get(0)
                x_group = dmap.syndrome_x.get(0)
                
                if z_group is not None and x_group is not None:
                    z_det = shot[z_group.start:z_group.end].astype(np.uint8)
                    x_det = shot[x_group.start:x_group.end].astype(np.uint8)
                    
                    # Decode this round
                    result = self._decoder.decode_syndrome(z_det, x_det)
                    
                    if not result.success:
                        shot_ambiguous = True
                        shot_confidence *= result.confidence
                    else:
                        # Accumulate corrections
                        accumulated_correction_x ^= result.correction_x
                        accumulated_correction_z ^= result.correction_z
            
            # === Step 3: Decode final measurement ===
            if metadata.final_measurement is not None:
                final_group = metadata.final_measurement.final_data
                if final_group is not None:
                    final_det = shot[final_group.start:final_group.end].astype(np.uint8)
                    
                    # Final measurement gives us raw data qubit values
                    # For detector-based: these are already the measurement outcomes
                    # (not syndrome - actual data values)
                    final_data = final_det
                else:
                    final_data = np.zeros(metadata.n_data_qubits, dtype=np.uint8)
            else:
                final_data = np.zeros(metadata.n_data_qubits, dtype=np.uint8)
            
            # === Step 4: Apply corrections and compute logical ===
            corrected_data = (final_data ^ accumulated_correction_x) % 2
            
            # Compute logical value
            if self.basis == 'Z':
                # Z basis: logical Z = parity on Z support
                logical_value = int(np.sum(corrected_data[metadata.z_logical_support]) % 2)
                expected = 0 if self.initial_state == '0' else 1
            else:
                # X basis: logical X = parity on X support (after corrections)
                corrected_x = (final_data ^ accumulated_correction_z) % 2
                logical_value = int(np.sum(corrected_x[metadata.x_logical_support]) % 2)
                expected = 0 if self.initial_state == '+' else 1
            
            # === Step 5: Check for error ===
            if shot_ambiguous:
                # Fractional contribution for ambiguous shots
                n_ambiguous += 1
                if logical_value != expected:
                    fractional_error += shot_confidence  # 0.5 for single ambiguous
                else:
                    fractional_error += (1.0 - shot_confidence)
            else:
                if logical_value != expected:
                    n_logical_errors += 1
        
        # Compute statistics
        if n_accepted > 0:
            logical_error_rate = (n_logical_errors + fractional_error) / n_accepted
            # Binomial std estimate
            logical_error_rate_std = np.sqrt(
                logical_error_rate * (1 - logical_error_rate) / n_accepted
            )
        else:
            logical_error_rate = 0.0
            logical_error_rate_std = 0.0
        
        return ExperimentResult(
            n_shots=n_shots,
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            n_logical_errors=n_logical_errors,
            logical_error_rate=logical_error_rate,
            logical_error_rate_std=logical_error_rate_std,
            fractional_error_contribution=fractional_error,
            rejection_rate=n_rejected / n_shots if n_shots > 0 else 0.0,
            n_ambiguous=n_ambiguous,
            ambiguity_rate=n_ambiguous / n_accepted if n_accepted > 0 else 0.0,
        )


class DetectorConcatenatedMemoryExperiment:
    """
    Detector-based memory experiment for concatenated codes.
    
    Supports multi-level concatenation with hierarchical decoding.
    Uses local hard-decision decoders at each level.
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The concatenated code
    noise_model : NoiseModel, optional
        Noise model
    rounds : int
        Number of EC rounds per level
    basis : str
        Measurement basis
    ancilla_prep : str
        Ancilla preparation method at all levels
    """
    
    def __init__(
        self,
        code: 'MultiLevelConcatenatedCode',
        noise_model: Optional['NoiseModel'] = None,
        rounds: int = 0,
        basis: str = 'Z',
        ancilla_prep: str = 'encoded',
        initial_state: str = '0',
    ):
        self.code = code
        self.noise_model = noise_model
        self.rounds = rounds
        self.basis = basis.upper()
        self.ancilla_prep = ancilla_prep
        self.initial_state = initial_state
        
        # Get inner and outer codes
        self.inner_code = code.codes[-1] if hasattr(code, 'codes') else code
        self.outer_code = code.codes[0] if hasattr(code, 'codes') and len(code.codes) > 1 else code
        self.n_inner_blocks = self.outer_code.n if hasattr(self.outer_code, 'n') else 1
        
        # Create hierarchical decoder
        self._decoder = self._create_hierarchical_decoder()
        
        # Create EC gadgets per level
        self._inner_gadget = DetectorTransversalSyndrome(
            code=self.inner_code,
            ancilla_prep=ancilla_prep,
        )
        self._outer_gadget = DetectorTransversalSyndrome(
            code=self.outer_code,
            ancilla_prep=ancilla_prep,
        )
    
    def _create_hierarchical_decoder(self) -> HierarchicalDecoder:
        """Create hierarchical decoder for concatenated code."""
        inner_decoder = self._get_decoder_for_code(self.inner_code)
        outer_decoder = self._get_decoder_for_code(self.outer_code)
        
        return HierarchicalDecoder(
            inner_decoder=inner_decoder,
            outer_decoder=outer_decoder,
            n_inner_blocks=self.n_inner_blocks,
        )
    
    def _get_decoder_for_code(self, code) -> LocalDecoder:
        """Get decoder for a specific code."""
        n = code.n if hasattr(code, 'n') else 7
        
        if n == 7:
            return SteaneDecoder()
        elif n == 9:
            return ShorDecoder()
        else:
            return SteaneDecoder()  # Default
    
    def build(self) -> Tuple[stim.Circuit, CircuitMetadata]:
        """
        Build circuit for concatenated code memory experiment.
        
        Structure:
        1. Prepare all physical qubits
        2. Encode at each level
        3. EC rounds with hierarchical gadget emission
        4. Final measurement of all data
        5. DETECTORs for syndrome and final data
        """
        circuit = stim.Circuit()
        counter = DetectorCounter()
        metadata = CircuitMetadata()
        
        # Total qubits: inner_n * outer_n
        inner_n = self.inner_code.n
        outer_n = self.outer_code.n  
        total_n = inner_n * outer_n
        
        # Data qubits organized by inner block
        data_qubits = {}
        for block_id in range(outer_n):
            start = block_id * inner_n
            data_qubits[block_id] = list(range(start, start + inner_n))
        
        current_meas = 0
        
        # === State Preparation ===
        circuit.append("R", list(range(total_n)))
        
        # Encode at inner level (each block)
        for block_id, qubits in data_qubits.items():
            self._apply_inner_encoding(circuit, qubits)
        
        # Encode at outer level
        self._apply_outer_encoding(circuit, data_qubits)
        
        # === EC Rounds ===
        for round_idx in range(self.rounds):
            if self.noise_model is not None and round_idx > 0:
                self._apply_idle_noise(circuit, list(range(total_n)))
            
            # Emit EC for each inner block
            round_dmap = DetectorMap(offset=counter.peek(), measurement_offset=current_meas)
            
            for block_id, qubits in data_qubits.items():
                inner_dmap = self._inner_gadget.emit(
                    circuit=circuit,
                    data_qubits={0: qubits},
                    counter=counter,
                    noise_model=self.noise_model,
                    measurement_offset=current_meas,
                    round_idx=round_idx,
                    level=1,  # Inner level
                )
                current_meas += inner_dmap.total_measurements
                
                # Track inner syndrome groups with block_id
                for group in inner_dmap.all_groups:
                    group.block_id = block_id
                    round_dmap.add_group(group)
            
            round_dmap.total_measurements = current_meas - round_dmap.measurement_offset
            
            ec_round = ECRoundResult(
                round_idx=round_idx,
                detector_map=round_dmap,
            )
            metadata.add_ec_round(ec_round)
        
        # === Final Measurement ===
        circuit.append("M", list(range(total_n)))
        
        final_meas_start = current_meas
        current_meas += total_n
        
        # Emit final data detectors (one per data qubit)
        final_dmap = DetectorMap(offset=counter.peek(), measurement_offset=final_meas_start)
        
        for block_id, qubits in data_qubits.items():
            det_start = counter.peek()
            
            for i, q in enumerate(qubits):
                meas_idx = final_meas_start + q
                lookback = meas_idx - current_meas
                circuit.append("DETECTOR", [stim.target_rec(lookback)])
                counter.advance(1)
            
            det_end = counter.peek()
            
            final_group = DetectorGroup(
                start=det_start,
                end=det_end,
                dtype=DetectorType.FINAL_DATA,
                block_id=block_id,
                level=1,
            )
            final_dmap.add_group(final_group)
        
        metadata.final_measurement = final_dmap
        
        # === Populate Metadata ===
        metadata.total_detectors = counter.current
        metadata.total_measurements = current_meas
        metadata.code_name = getattr(self.code, 'name', 'concatenated')
        metadata.n_levels = 2
        metadata.inner_code_name = getattr(self.inner_code, 'name', 'inner')
        metadata.outer_code_name = getattr(self.outer_code, 'name', 'outer')
        metadata.n_inner_blocks = outer_n
        metadata.n_data_qubits = total_n
        metadata.initial_state = self.initial_state
        metadata.basis = self.basis
        
        # Logical operator support for concatenated code
        # Z_L = Z_L_outer composed with Z_L_inner
        metadata.z_logical_support = self._get_concatenated_z_support()
        metadata.x_logical_support = self._get_concatenated_x_support()
        
        # Apply noise model if provided
        if self.noise_model is not None:
            circuit = self.noise_model.apply(circuit)
        
        return circuit, metadata
    
    def _apply_inner_encoding(self, circuit: stim.Circuit, qubits: List[int]) -> None:
        """Apply encoding for one inner block."""
        n = len(qubits)
        
        if n == 7:
            # Steane encoding
            h_qubits = [qubits[i] for i in [0, 1, 3]]
            circuit.append("H", h_qubits)
            
            cnots = [(1, 2), (3, 5), (0, 4), (1, 6), (0, 2), (3, 4), (1, 5), (4, 6)]
            for c, t in cnots:
                circuit.append("CNOT", [qubits[c], qubits[t]])
    
    def _apply_outer_encoding(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
    ) -> None:
        """Apply outer-level encoding (transversal on inner blocks)."""
        # For Steane-in-Steane: outer encoding is transversal
        n_blocks = len(data_qubits)
        
        if n_blocks == 7:
            # Steane outer encoding applied transversally
            # H on blocks 0, 1, 3
            for block_id in [0, 1, 3]:
                for q in data_qubits[block_id]:
                    circuit.append("H", [q])
            
            # CNOTs applied block-wise
            cnots = [(1, 2), (3, 5), (0, 4), (1, 6), (0, 2), (3, 4), (1, 5), (4, 6)]
            for c, t in cnots:
                for i in range(len(data_qubits[c])):
                    circuit.append("CNOT", [data_qubits[c][i], data_qubits[t][i]])
    
    def _apply_idle_noise(self, circuit: stim.Circuit, qubits: List[int]) -> None:
        """Apply idle noise."""
        if self.noise_model is not None and hasattr(self.noise_model, 'idle_error_rate'):
            p = self.noise_model.idle_error_rate
            for q in qubits:
                circuit.append("DEPOLARIZE1", [q], p)
    
    def _get_concatenated_z_support(self) -> List[int]:
        """Get Z logical support for concatenated code."""
        # Z_L = Z_L_outer ⊗ Z_L_inner
        inner_z = list(range(self.inner_code.n))  # All qubits for Steane
        outer_z = list(range(self.outer_code.n))  # All blocks for Steane
        
        support = []
        for block_id in outer_z:
            base = block_id * self.inner_code.n
            for q in inner_z:
                support.append(base + q)
        return support
    
    def _get_concatenated_x_support(self) -> List[int]:
        """Get X logical support for concatenated code."""
        # Same as Z for Steane code
        return self._get_concatenated_z_support()
    
    def run(
        self,
        shots: int = 1000,
        verbose: bool = False,
    ) -> ExperimentResult:
        """Run experiment and decode."""
        circuit, metadata = self.build()
        
        if verbose:
            print(f"Concatenated circuit: {metadata.total_detectors} detectors, "
                  f"{metadata.n_inner_blocks} inner blocks")
        
        detector_sampler = circuit.compile_detector_sampler()
        detector_samples = detector_sampler.sample(shots)
        
        return self._decode_samples(detector_samples, metadata, verbose)
    
    def _decode_samples(
        self,
        detector_samples: np.ndarray,
        metadata: CircuitMetadata,
        verbose: bool = False,
    ) -> ExperimentResult:
        """Decode samples using hierarchical decoder."""
        n_shots = len(detector_samples)
        n_accepted = 0
        n_rejected = 0
        n_logical_errors = 0
        n_ambiguous = 0
        fractional_error = 0.0
        
        ver_ranges = metadata.get_all_verification_ranges()
        
        for shot in detector_samples:
            # Post-selection
            if ver_ranges and not post_select_verification(shot, ver_ranges):
                n_rejected += 1
                continue
            
            n_accepted += 1
            
            # Hierarchical decoding
            # For each EC round, decode inner blocks then outer
            accumulated_correction = np.zeros(metadata.n_data_qubits, dtype=np.uint8)
            shot_confidence = 1.0
            shot_ambiguous = False
            
            for ec_round in metadata.ec_rounds:
                dmap = ec_round.detector_map
                
                # Build inner syndrome groups
                inner_z_groups = []
                inner_x_groups = []
                
                for block_id in range(metadata.n_inner_blocks):
                    z_key = (1, block_id)  # level=1 (inner), block_id
                    x_key = (1, block_id)
                    
                    if 1 in dmap.hierarchical and block_id in dmap.hierarchical[1]:
                        block_map = dmap.hierarchical[1][block_id]
                        
                        if 'syndrome_z' in block_map:
                            g = block_map['syndrome_z']
                            inner_z_groups.append((g.start, g.end))
                        else:
                            inner_z_groups.append((0, 0))
                        
                        if 'syndrome_x' in block_map:
                            g = block_map['syndrome_x']
                            inner_x_groups.append((g.start, g.end))
                        else:
                            inner_x_groups.append((0, 0))
                    else:
                        inner_z_groups.append((0, 0))
                        inner_x_groups.append((0, 0))
                
                # Decode hierarchically
                if inner_z_groups and inner_x_groups:
                    result = self._decoder.decode(
                        shot,
                        inner_z_groups,
                        inner_x_groups,
                    )
                    
                    if not result.success:
                        shot_ambiguous = True
                        shot_confidence *= result.confidence
                    else:
                        accumulated_correction ^= result.correction_x
            
            # Decode final measurement
            if metadata.final_measurement is not None:
                final_data = np.zeros(metadata.n_data_qubits, dtype=np.uint8)
                
                for group in metadata.final_measurement.all_groups:
                    if group.dtype == DetectorType.FINAL_DATA:
                        block_data = shot[group.start:group.end]
                        base = group.block_id * self.inner_code.n
                        final_data[base:base + len(block_data)] = block_data
            else:
                final_data = np.zeros(metadata.n_data_qubits, dtype=np.uint8)
            
            # Apply corrections
            corrected = (final_data ^ accumulated_correction) % 2
            
            # Compute logical value
            logical_value = int(np.sum(corrected[metadata.z_logical_support]) % 2)
            expected = 0 if self.initial_state == '0' else 1
            
            # Check error
            if shot_ambiguous:
                n_ambiguous += 1
                if logical_value != expected:
                    fractional_error += shot_confidence
                else:
                    fractional_error += (1.0 - shot_confidence)
            else:
                if logical_value != expected:
                    n_logical_errors += 1
        
        if n_accepted > 0:
            logical_error_rate = (n_logical_errors + fractional_error) / n_accepted
            logical_error_rate_std = np.sqrt(
                logical_error_rate * (1 - logical_error_rate) / n_accepted
            )
        else:
            logical_error_rate = 0.0
            logical_error_rate_std = 0.0
        
        return ExperimentResult(
            n_shots=n_shots,
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            n_logical_errors=n_logical_errors,
            logical_error_rate=logical_error_rate,
            logical_error_rate_std=logical_error_rate_std,
            fractional_error_contribution=fractional_error,
            rejection_rate=n_rejected / n_shots if n_shots > 0 else 0.0,
            n_ambiguous=n_ambiguous,
            ambiguity_rate=n_ambiguous / n_accepted if n_accepted > 0 else 0.0,
        )
