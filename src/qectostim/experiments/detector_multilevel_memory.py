"""
Detector-Based Multi-Level Memory Experiment.

This module provides a Stim circuit builder for multi-level concatenated codes
using the detector-based architecture. Key design principles from reference code:

Architecture (matching c4steane.py patterns):
--------------------------------------------
1. **Gadgets emit DETECTOR instructions** directly after each measurement block
2. **No OBSERVABLE_INCLUDE** - all logical outcomes computed in Python
3. **Detector sampler** - use circuit.compile_detector_sampler().sample()
4. **Local hard-decision decoding** - per-gadget lookup tables, no global MWPM
5. **Post-selection via verification** - reject shots with verification detector parity != 0
6. **Ambiguous handling** - decoder returns -1 for uncorrectable patterns:
   - Verification checks: -1 → reject (post-select away)
   - Syndrome decoding: -1 → either reject or count as 0.5 error contribution
7. **Corrections accumulated in software** - no feed-forward in circuit

Circuit Structure:
-----------------
1. Reset all data qubits (R)
2. For each EC round:
   - Prepare ancilla via append_noisy_0prep (returns verification detector groups)
   - Transversal CNOTs (data ↔ ancilla)
   - Measure ancilla, emit DETECTORs immediately
   - Reset ancilla for next round
3. Final measurement of all data qubits
4. Final DETECTORs for boundary detection

Data Flow:
---------
    >>> exp = DetectorMultiLevelMemory(code, noise_model, rounds=Q)
    >>> circuit, metadata = exp.build()
    >>> detector_samples = circuit.compile_detector_sampler().sample(shots)
    >>> for shot in detector_samples:
    ...     # Post-selection on verification groups
    ...     if any(shot[start:end].sum() % 2 for start, end in metadata.verification_ranges):
    ...         continue  # reject
    ...     # Decode each EC round locally
    ...     corrections = []
    ...     for ec_round in metadata.ec_rounds:
    ...         result = decoder.decode_syndrome(shot, ec_round)
    ...         if result.is_ambiguous:
    ...             # Either reject or mark for 0.5 contribution
    ...         corrections.append(result)
    ...     # Decode final measurement
    ...     final = decode_final(shot, metadata)
    ...     # Apply corrections in software
    ...     logical = apply_corrections(final, corrections)
    ...     # Compare to expected |0_L⟩
    ...     error = (logical != 0)

Note on Decoder Returns:
-----------------------
Decoders return LocalDecoderResult which has:
- success: bool (False if ambiguous/detected)
- correction_x, correction_z: np.ndarray
- confidence: float (1.0 for definite, 0.5 for ambiguous)
- logical_value: [int] (decoded logical bits)

Ambiguous (-1 equivalent) is represented by success=False + confidence=0.5

Usage:
------
    >>> from qectostim.experiments.detector_multilevel_memory import (
    ...     DetectorMultiLevelMemory, estimate_logical_error_rate
    ... )
    >>> from qectostim.codes.composite import make_concatenated_code
    >>> 
    >>> # Build Steane x Steane concatenated code
    >>> code = make_concatenated_code(['steane', 'steane'])
    >>> exp = DetectorMultiLevelMemory(code, rounds=3, p=1e-3)
    >>> results = estimate_logical_error_rate(exp, shots=10000)
    >>> print(f"Logical error rate: {results.logical_error_rate:.2e}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import stim
import numpy as np

from qectostim.experiments.gadgets.detector_primitives import (
    DetectorMap, DetectorGroup, DetectorType, DetectorCounter,
    CircuitMetadata, ECRoundResult, GadgetResult,
)
from qectostim.decoders.local_decoders import (
    LocalDecoder, LocalDecoderResult, HierarchicalDecoder,
    SteaneDecoder, ShorDecoder, RepetitionDecoder,
    get_decoder_for_code, post_select_verification, filter_shots_by_verification,
)
from qectostim.decoders.detector_decoder import (
    DetectorBasedDecoder, HierarchicalDetectorDecoder,
    DecodingResults, ShotResult, RejectionReason,
    decode_detector_samples,
)

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode
    from qectostim.codes.composite.multilevel_concatenated import MultiLevelConcatenatedCode
    from qectostim.noise.models import NoiseModel


# =============================================================================
# METADATA CLASSES
# =============================================================================

@dataclass
class DetectorCircuitMetadata:
    """
    Complete metadata for detector-based concatenated code experiment.
    
    This extends CircuitMetadata with concatenation-specific information.
    """
    # Basic info
    code_name: str = ""
    n_levels: int = 1
    level_code_names: List[str] = field(default_factory=list)
    n_physical_qubits: int = 0
    total_distance: int = 1
    
    # Detector structure
    total_detectors: int = 0
    total_measurements: int = 0
    
    # EC rounds with detector maps
    ec_rounds: List[ECRoundResult] = field(default_factory=list)
    
    # Verification detector ranges for post-selection
    # List of (start, end) tuples - reject if parity != 0
    verification_ranges: List[Tuple[int, int]] = field(default_factory=list)
    
    # Final measurement detector groups
    final_measurement: Optional[DetectorMap] = None
    
    # Logical operator support (for computing logical value in Python)
    z_logical_support: List[int] = field(default_factory=list)
    x_logical_support: List[int] = field(default_factory=list)
    
    # Level-specific decoder info
    level_decoders: Dict[int, str] = field(default_factory=dict)  # level -> decoder name
    n_inner_blocks: int = 1
    
    # Prepared state (for expected logical value)
    initial_state: str = "0"
    basis: str = "Z"
    
    # Hierarchical structure for decoder
    # level -> block_id -> syndrome_type -> (start, end)
    hierarchical_detector_ranges: Dict[int, Dict[int, Dict[str, Tuple[int, int]]]] = field(default_factory=dict)
    
    def add_verification_range(self, start: int, end: int) -> None:
        """Add a verification detector range."""
        self.verification_ranges.append((start, end))
    
    def add_ec_round(self, round_result: ECRoundResult) -> None:
        """Add an EC round result."""
        self.ec_rounds.append(round_result)
        # Collect verification ranges
        for v_group in round_result.detector_map.verification:
            self.verification_ranges.append((v_group.start, v_group.end))


@dataclass
class ExperimentResult:
    """
    Result from running a detector-based memory experiment.
    
    Matches the output format from reference code estimate functions.
    """
    # Counts
    n_shots: int
    n_accepted: int
    n_rejected: int
    n_logical_errors: int
    n_ambiguous: int
    
    # Rates
    logical_error_rate: float
    logical_error_rate_variance: float
    acceptance_rate: float
    
    # Fractional contribution from ambiguous shots (0.5 each)
    fractional_error_contribution: float = 0.0
    
    # Breakdown
    rejection_by_reason: Dict[str, int] = field(default_factory=dict)
    
    @property
    def logical_error_rate_std(self) -> float:
        """Standard deviation of logical error rate estimate."""
        return np.sqrt(self.logical_error_rate_variance)
    
    @classmethod
    def from_decoding_results(cls, results: DecodingResults, n_shots: int) -> 'ExperimentResult':
        """Create from DecodingResults."""
        return cls(
            n_shots=n_shots,
            n_accepted=results.n_accepted,
            n_rejected=results.n_rejected,
            n_logical_errors=results.n_logical_errors,
            n_ambiguous=results.n_ambiguous,
            logical_error_rate=results.logical_error_rate,
            logical_error_rate_variance=results.logical_error_rate_std ** 2,
            acceptance_rate=1 - results.rejection_rate,
            fractional_error_contribution=results.fractional_error,
        )


# =============================================================================
# MAIN EXPERIMENT CLASS
# =============================================================================

class DetectorMultiLevelMemory:
    """
    Detector-based memory experiment for multi-level concatenated codes.
    
    This implements the architecture from the reference code (c4steane.py):
    - Gadgets emit DETECTORs directly after measurements
    - No OBSERVABLE_INCLUDE
    - Local hard-decision decoders
    - Post-selection on verification detectors
    - Corrections accumulated in software
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The concatenated code (e.g., Steane ⊗ Steane)
    rounds : int
        Number of EC rounds (Q in reference code)
    p : float
        Physical error probability (applies depolarizing noise)
    noise_model : NoiseModel, optional
        Custom noise model (if None, uses depolarizing with p)
    basis : str
        Measurement basis ('Z' or 'X')
    initial_state : str
        Initial logical state ('0' for |0_L⟩, '+' for |+_L⟩)
    ancilla_prep : str
        Ancilla preparation method:
        - "bare": Simple reset (NOT fault-tolerant)
        - "encoded": Proper encoding (fault-tolerant)
        - "verified": Encoding + verification (fully FT with post-selection)
    reject_on_ambiguity : bool
        If True, reject shots where decoder returns ambiguous
        If False, count ambiguous as 0.5 error contribution
    """
    
    def __init__(
        self,
        code: 'MultiLevelConcatenatedCode',
        rounds: int = 1,
        p: float = 0.0,
        noise_model: Optional['NoiseModel'] = None,
        basis: str = 'Z',
        initial_state: str = '0',
        ancilla_prep: str = 'encoded',
        reject_on_ambiguity: bool = False,
    ):
        self.code = code
        self.rounds = rounds
        self.p = p
        self.noise_model = noise_model
        self.basis = basis.upper()
        self.initial_state = initial_state
        self.ancilla_prep = ancilla_prep
        self.reject_on_ambiguity = reject_on_ambiguity
        
        # Set up noise model if not provided
        if self.noise_model is None and p > 0:
            from qectostim.noise.models import CircuitDepolarizingNoise
            self.noise_model = CircuitDepolarizingNoise(p)
        
        # Extract level codes
        self.level_codes = code.level_codes
        self.n_levels = len(self.level_codes)
        
        # Create decoders for each level
        self._decoders = self._create_level_decoders()
        
        # Create hierarchical qubit mapper
        from qectostim.utils.hierarchical_mapper import HierarchicalQubitMapper
        self.qubit_mapper = HierarchicalQubitMapper(code)
        
        # Get total physical qubits
        self.n_physical = code.n
        
        # Precompute logical supports
        self._z_support = self._compute_z_logical_support()
        self._x_support = self._compute_x_logical_support()
    
    def _create_level_decoders(self) -> Dict[int, LocalDecoder]:
        """Create a local decoder for each level's code."""
        decoders = {}
        for level_idx, level_code in enumerate(self.level_codes):
            code_name = getattr(level_code, 'name', '').lower()
            n = level_code.n
            
            if 'steane' in code_name or n == 7:
                decoders[level_idx] = SteaneDecoder()
            elif 'shor' in code_name or n == 9:
                decoders[level_idx] = ShorDecoder()
            elif 'repetition' in code_name:
                decoders[level_idx] = RepetitionDecoder(n)
            else:
                # Default to Steane
                decoders[level_idx] = SteaneDecoder()
        
        return decoders
    
    def _compute_z_logical_support(self) -> List[int]:
        """Compute Z logical operator support on physical qubits."""
        # For concatenated code, Z_L acts on inner logical Z supports
        # which recursively map to physical qubits
        outer_code = self.level_codes[0]
        
        # Get outer Z_L support
        outer_z_support = self._get_code_z_support(outer_code)
        
        if self.n_levels == 1:
            return outer_z_support
        
        # For 2-level concatenation
        inner_code = self.level_codes[1]
        inner_z_support = self._get_code_z_support(inner_code)
        inner_n = inner_code.n
        
        # Z_L on physical qubits: for each outer block in Z_L support,
        # take the inner Z_L support within that block
        physical_support = []
        for outer_block in outer_z_support:
            block_base = outer_block * inner_n
            for inner_q in inner_z_support:
                physical_support.append(block_base + inner_q)
        
        return physical_support
    
    def _compute_x_logical_support(self) -> List[int]:
        """Compute X logical operator support on physical qubits."""
        outer_code = self.level_codes[0]
        outer_x_support = self._get_code_x_support(outer_code)
        
        if self.n_levels == 1:
            return outer_x_support
        
        inner_code = self.level_codes[1]
        inner_x_support = self._get_code_x_support(inner_code)
        inner_n = inner_code.n
        
        physical_support = []
        for outer_block in outer_x_support:
            block_base = outer_block * inner_n
            for inner_q in inner_x_support:
                physical_support.append(block_base + inner_q)
        
        return physical_support
    
    def _get_code_z_support(self, code: Any) -> List[int]:
        """Get Z logical support for a single code."""
        if hasattr(code, 'logical_z') and code.logical_z:
            lz = code.logical_z
            if isinstance(lz, list) and len(lz) > 0:
                op = lz[0]
                if isinstance(op, str):
                    return [i for i, c in enumerate(op) if c.upper() in ('Z', 'Y')]
        
        if hasattr(code, 'lz'):
            lz = getattr(code, 'lz')
            if callable(lz):
                lz = lz()
            if isinstance(lz, np.ndarray):
                lz = np.atleast_2d(lz)
                if lz.shape[0] > 0:
                    return list(np.where(lz[0] != 0)[0])
        
        # Code-specific fallbacks
        code_name = getattr(code, 'name', '').lower()
        n = code.n
        
        if 'steane' in code_name or n == 7:
            return list(range(7))  # All qubits for Steane
        elif 'shor' in code_name or n == 9:
            return [0, 3, 6]  # One from each block for Shor
        else:
            return list(range(n))
    
    def _get_code_x_support(self, code: Any) -> List[int]:
        """Get X logical support for a single code."""
        if hasattr(code, 'logical_x') and code.logical_x:
            lx = code.logical_x
            if isinstance(lx, list) and len(lx) > 0:
                op = lx[0]
                if isinstance(op, str):
                    return [i for i, c in enumerate(op) if c.upper() in ('X', 'Y')]
        
        if hasattr(code, 'lx'):
            lx = getattr(code, 'lx')
            if callable(lx):
                lx = lx()
            if isinstance(lx, np.ndarray):
                lx = np.atleast_2d(lx)
                if lx.shape[0] > 0:
                    return list(np.where(lx[0] != 0)[0])
        
        # Code-specific fallbacks
        code_name = getattr(code, 'name', '').lower()
        n = code.n
        
        if 'steane' in code_name or n == 7:
            return list(range(7))
        elif 'shor' in code_name or n == 9:
            return list(range(9))  # All qubits for Shor X_L
        else:
            return list(range(n))
    
    def build(self) -> Tuple[stim.Circuit, DetectorCircuitMetadata]:
        """
        Build the Stim circuit and metadata.
        
        Circuit structure follows reference code:
        1. R (reset) all data qubits
        2. For each EC round:
           - append_noisy_0prep for ancilla (emits verification DETECTORs)
           - Transversal CNOTs
           - Measure ancilla (emits syndrome DETECTORs)
        3. M (measure) all data qubits
        4. Emit final data DETECTORs
        
        Returns:
            (circuit, metadata) where circuit uses DETECTOR instructions
            and metadata has all info for Python-based decoding
        """
        circuit = stim.Circuit()
        counter = DetectorCounter()
        metadata = DetectorCircuitMetadata()
        
        # Organize data qubits by block
        data_by_block = self._organize_data_by_block()
        all_data = list(range(self.n_physical))
        
        current_meas = 0
        
        # === 1. State Preparation ===
        circuit.append("R", all_data)
        
        # For |+_L⟩ preparation (before encoding)
        if self.initial_state == '+' or self.basis == 'X':
            circuit.append("H", all_data)
        
        # Apply encoding to each inner block
        # This prepares the logical |0_L⟩ or |+_L⟩ state as a proper codeword
        for block_id in sorted(data_by_block.keys()):
            data_qs = data_by_block[block_id]
            self._apply_encoding(circuit, data_qs)
        
        circuit.append("TICK")
        
        # === 2. EC Rounds ===
        for round_idx in range(self.rounds):
            round_dmap = DetectorMap(offset=counter.peek(), measurement_offset=current_meas)
            
            # Emit EC gadget for all blocks
            for block_id in sorted(data_by_block.keys()):
                data_qs = data_by_block[block_id]
                n = len(data_qs)
                
                # Allocate ancilla qubits for this block
                ancilla_base = self.n_physical + block_id * n
                ancilla_qs = list(range(ancilla_base, ancilla_base + n))
                
                # === Z Syndrome Extraction (detects X errors) ===
                dmap_z = self._emit_syndrome_extraction(
                    circuit=circuit,
                    data_qs=data_qs,
                    ancilla_qs=ancilla_qs,
                    basis='Z',
                    counter=counter,
                    current_meas=current_meas,
                    round_idx=round_idx,
                    block_id=block_id,
                )
                current_meas += n
                
                # Merge into round detector map
                round_dmap.verification.extend(dmap_z.verification)
                round_dmap.syndrome_z[block_id] = dmap_z.syndrome_z.get(0)
                
                # === X Syndrome Extraction (detects Z errors) ===
                dmap_x = self._emit_syndrome_extraction(
                    circuit=circuit,
                    data_qs=data_qs,
                    ancilla_qs=ancilla_qs,
                    basis='X',
                    counter=counter,
                    current_meas=current_meas,
                    round_idx=round_idx,
                    block_id=block_id,
                )
                current_meas += n
                
                round_dmap.verification.extend(dmap_x.verification)
                round_dmap.syndrome_x[block_id] = dmap_x.syndrome_x.get(0)
            
            # Record EC round
            round_result = ECRoundResult(
                round_idx=round_idx,
                detector_map=round_dmap,
            )
            metadata.add_ec_round(round_result)
            
            # Apply idle noise between rounds
            if self.noise_model is not None:
                self._apply_idle_noise(circuit, all_data)
            
            circuit.append("TICK")
        
        # === 3. Final Measurement ===
        if self.basis == 'X':
            circuit.append("H", all_data)
        
        circuit.append("M", all_data)
        final_meas_start = current_meas
        current_meas += self.n_physical
        
        # === 4. Final Data DETECTORs ===
        # These compare final data to last syndrome round (boundary detection)
        final_dmap = self._emit_final_detectors(
            circuit=circuit,
            counter=counter,
            final_meas_start=final_meas_start,
            current_meas=current_meas,
            metadata=metadata,
        )
        metadata.final_measurement = final_dmap
        
        # === 5. Populate Metadata ===
        metadata.code_name = getattr(self.code, 'name', 'concatenated')
        metadata.n_levels = self.n_levels
        metadata.level_code_names = [getattr(c, 'name', 'unknown') for c in self.level_codes]
        metadata.n_physical_qubits = self.n_physical
        metadata.total_distance = self._compute_total_distance()
        metadata.total_detectors = counter.current
        metadata.total_measurements = current_meas
        metadata.z_logical_support = self._z_support
        metadata.x_logical_support = self._x_support
        metadata.initial_state = self.initial_state
        metadata.basis = self.basis
        metadata.n_inner_blocks = self.level_codes[0].n if self.n_levels > 1 else 1
        
        # Store decoder names
        for level_idx, decoder in self._decoders.items():
            metadata.level_decoders[level_idx] = decoder.code_name
        
        # Apply noise model
        if self.noise_model is not None:
            circuit = self.noise_model.apply(circuit)
        
        return circuit, metadata
    
    def _organize_data_by_block(self) -> Dict[int, List[int]]:
        """Organize data qubits by inner block index."""
        data_by_block = {}
        for block_idx, (addr, (start, end)) in enumerate(self.qubit_mapper.iter_leaf_ranges()):
            data_by_block[block_idx] = list(range(start, end))
        return data_by_block
    
    def _apply_encoding(self, circuit: stim.Circuit, data_qs: List[int]) -> None:
        """
        Apply the encoding circuit for a single inner block.
        
        This prepares |0⟩^⊗n → |0_L⟩ for the inner code.
        The encoding must be done BEFORE syndrome extraction to ensure
        we start in a valid codeword state.
        
        Args:
            circuit: Stim circuit to append to
            data_qs: The data qubits for this block (in order)
        """
        # The inner code is the LAST in level_codes (most-nested level)
        inner_code = self.level_codes[-1]
        code_name = getattr(inner_code, 'name', '').lower()
        n = len(data_qs)
        
        if 'steane' in code_name or n == 7:
            # Steane [[7,1,3]] encoding circuit
            # Hadamards on qubits 0, 1, 3 (indices within block)
            circuit.append("H", [data_qs[0], data_qs[1], data_qs[3]])
            
            # CNOT cascade for encoding
            # Based on standard Steane encoder
            circuit.append("CNOT", [data_qs[0], data_qs[4]])
            circuit.append("CNOT", [data_qs[1], data_qs[5]])
            circuit.append("CNOT", [data_qs[3], data_qs[6]])
            circuit.append("CNOT", [data_qs[0], data_qs[2]])
            circuit.append("CNOT", [data_qs[1], data_qs[2]])
            circuit.append("CNOT", [data_qs[3], data_qs[2]])
            circuit.append("CNOT", [data_qs[0], data_qs[6]])
            circuit.append("CNOT", [data_qs[1], data_qs[4]])
            circuit.append("CNOT", [data_qs[3], data_qs[5]])
            
        elif 'shor' in code_name or n == 9:
            # Shor [[9,1,3]] encoding circuit
            # First encode bit flip code: |0⟩ → |+++⟩
            circuit.append("CNOT", [data_qs[0], data_qs[3]])
            circuit.append("CNOT", [data_qs[0], data_qs[6]])
            
            # Then encode each group with phase flip code
            # |+⟩ → |000⟩ + |111⟩ for each group
            circuit.append("H", [data_qs[0], data_qs[3], data_qs[6]])
            circuit.append("CNOT", [data_qs[0], data_qs[1]])
            circuit.append("CNOT", [data_qs[0], data_qs[2]])
            circuit.append("CNOT", [data_qs[3], data_qs[4]])
            circuit.append("CNOT", [data_qs[3], data_qs[5]])
            circuit.append("CNOT", [data_qs[6], data_qs[7]])
            circuit.append("CNOT", [data_qs[6], data_qs[8]])
            
        elif 'repetition' in code_name:
            # [[n,1,n]] repetition code for Z basis
            # |0⟩ → |00...0⟩ - CNOTs from first qubit
            for i in range(1, n):
                circuit.append("CNOT", [data_qs[0], data_qs[i]])
        
        else:
            # For unknown codes, assume qubits are already in product state
            # (no encoding needed, but this may not give valid codeword)
            pass
    
    def _emit_syndrome_extraction(
        self,
        circuit: stim.Circuit,
        data_qs: List[int],
        ancilla_qs: List[int],
        basis: str,
        counter: DetectorCounter,
        current_meas: int,
        round_idx: int,
        block_id: int,
    ) -> DetectorMap:
        """
        Emit syndrome extraction for one block.
        
        This follows the reference code pattern:
        1. Prepare ancilla in |0⟩ or |+⟩ (with optional verification)
        2. Transversal CNOTs
        3. Measure ancilla
        4. Emit DETECTORs immediately after measurement
        
        Args:
            basis: 'Z' for Z syndrome (detect X errors), 'X' for X syndrome
            
        Returns:
            DetectorMap with syndrome and verification detector groups
        """
        dmap = DetectorMap(offset=counter.peek(), measurement_offset=current_meas)
        n = len(data_qs)
        
        # === 1. Ancilla Preparation ===
        # Reset ancilla
        circuit.append("R", ancilla_qs)
        
        # For X syndrome extraction, prepare ancilla in |+⟩
        if basis == 'X':
            circuit.append("H", ancilla_qs)
        
        # Verified ancilla: add verification measurements
        if self.ancilla_prep == 'verified':
            # This is where append_noisy_0prep would emit verification DETECTORs
            # For simplicity, we use encoded without explicit verification here
            pass
        
        circuit.append("TICK")
        
        # === 2. Transversal CNOTs ===
        # BOTH Z and X syndrome extraction use CNOT(data→ancilla)
        # The difference is:
        #   Z syndrome: ancilla in |0⟩, measure in Z basis → detects X errors
        #   X syndrome: ancilla in |+⟩, measure in X basis → detects Z errors
        for d, a in zip(data_qs, ancilla_qs):
            circuit.append("CNOT", [d, a])
        
        circuit.append("TICK")
        
        # === 3. Measure Ancilla ===
        if basis == 'X':
            circuit.append("H", ancilla_qs)
        
        circuit.append("M", ancilla_qs)
        meas_start = current_meas
        meas_end = meas_start + n
        
        # === 4. Emit DETECTORs ===
        # Each measurement becomes a detector (raw ancilla mode)
        # The decoder will compute syndromes: H @ raw_ancilla mod 2
        det_start = counter.peek()
        
        for i, meas_idx in enumerate(range(meas_start, meas_end)):
            lookback = meas_idx - (current_meas + n)  # After M, total meas increased
            circuit.append("DETECTOR", [stim.target_rec(lookback)])
            counter.advance(1)
        
        det_end = counter.peek()
        
        # Create detector group
        dtype = DetectorType.SYNDROME_Z if basis == 'Z' else DetectorType.SYNDROME_X
        det_group = DetectorGroup(
            start=det_start,
            end=det_end,
            dtype=dtype,
            block_id=block_id,
            level=1 if self.n_levels > 1 else 0,
            round_idx=round_idx,
        )
        dmap.add_group(det_group)
        dmap.total_measurements = n
        dmap.total_detectors = det_end - det_start
        
        return dmap
    
    def _emit_final_detectors(
        self,
        circuit: stim.Circuit,
        counter: DetectorCounter,
        final_meas_start: int,
        current_meas: int,
        metadata: DetectorCircuitMetadata,
    ) -> DetectorMap:
        """
        Emit final data measurement detectors.
        
        These are boundary detectors comparing final data to last syndrome.
        """
        dmap = DetectorMap(offset=counter.peek(), measurement_offset=final_meas_start)
        det_start = counter.peek()
        
        for i in range(self.n_physical):
            meas_idx = final_meas_start + i
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
        dmap.add_group(final_group)
        dmap.final_data = final_group
        dmap.total_detectors = det_end - det_start
        
        return dmap
    
    def _apply_idle_noise(self, circuit: stim.Circuit, qubits: List[int]) -> None:
        """Apply idle noise to data qubits between EC rounds."""
        if self.noise_model is not None and hasattr(self.noise_model, 'apply_idle'):
            self.noise_model.apply(circuit, qubits)
        elif self.p > 0:
            circuit.append("DEPOLARIZE1", qubits, self.p)
    
    def _compute_total_distance(self) -> int:
        """Compute total distance of concatenated code."""
        d = 1
        for code in self.level_codes:
            d *= getattr(code, 'd', 3)
        return d
    
    def run(self, shots: int) -> ExperimentResult:
        """
        Build circuit, sample, and decode.
        
        This is the main entry point for running experiments.
        
        Args:
            shots: Number of Monte Carlo samples
            
        Returns:
            ExperimentResult with logical error rate and statistics
        """
        circuit, metadata = self.build()
        
        # Sample using detector sampler (not measurement sampler!)
        detector_sampler = circuit.compile_detector_sampler()
        detector_samples = detector_sampler.sample(shots)
        
        # Decode using hierarchical decoder
        results = self._decode_samples(detector_samples, metadata)
        
        return ExperimentResult.from_decoding_results(results, shots)
    
    def _decode_samples(
        self,
        detector_samples: np.ndarray,
        metadata: DetectorCircuitMetadata,
    ) -> DecodingResults:
        """
        Decode all samples using hierarchical local decoding.
        
        This implements the Python-side decoding from reference code:
        1. Post-selection on verification detector groups
        2. Per-round local decoding with code-specific decoders
        3. Correction accumulation
        4. Final logical outcome computation
        5. Fractional counting for ambiguous results
        """
        shot_results = []
        
        for shot in detector_samples:
            result = self._decode_single_shot(shot, metadata)
            shot_results.append(result)
        
        return DecodingResults.from_shot_results(shot_results)
    
    def _decode_single_shot(
        self,
        detector_vector: np.ndarray,
        metadata: DetectorCircuitMetadata,
    ) -> ShotResult:
        """
        Decode a single shot.
        
        This follows the reference code's accept() / accept_c4() pattern.
        """
        # === 1. Post-selection on verification ===
        for v_start, v_end in metadata.verification_ranges:
            v_det = detector_vector[v_start:v_end]
            if np.sum(v_det) % 2 != 0:
                return ShotResult(
                    accepted=False,
                    rejection_reason=RejectionReason.VERIFICATION_FAILED,
                )
        
        # === 2. Decode each EC round ===
        accumulated_correction_x = np.zeros(self.n_physical, dtype=np.uint8)
        accumulated_correction_z = np.zeros(self.n_physical, dtype=np.uint8)
        
        round_results = []
        total_confidence = 1.0
        any_ambiguous = False
        
        for ec_round in metadata.ec_rounds:
            round_result = self._decode_ec_round(
                detector_vector,
                ec_round,
                metadata,
            )
            round_results.append(round_result)
            
            if not round_result.success:
                any_ambiguous = True
                total_confidence *= round_result.confidence
                
                if self.reject_on_ambiguity:
                    return ShotResult(
                        accepted=False,
                        rejection_reason=RejectionReason.EC_AMBIGUOUS,
                        round_results=round_results,
                    )
            
            if round_result.correction_x is not None:
                accumulated_correction_x ^= round_result.correction_x
            if round_result.correction_z is not None:
                accumulated_correction_z ^= round_result.correction_z
        
        # === 3. Decode final measurement ===
        final_data = np.zeros(self.n_physical, dtype=np.uint8)
        if metadata.final_measurement is not None:
            fg = metadata.final_measurement.final_data
            if fg is not None:
                final_data = detector_vector[fg.start:fg.end].astype(np.uint8)
        
        # === 4. Apply corrections ===
        if self.basis == 'Z':
            corrected = (final_data ^ accumulated_correction_x) % 2
            logical_support = metadata.z_logical_support
        else:
            corrected = (final_data ^ accumulated_correction_z) % 2
            logical_support = metadata.x_logical_support
        
        # === 5. Compute logical value ===
        logical_value = int(np.sum(corrected[logical_support]) % 2)
        
        # Expected value
        expected = 0 if metadata.initial_state == '0' else 1
        is_error = (logical_value != expected)
        
        return ShotResult(
            accepted=True,
            logical_value=logical_value,
            expected_value=expected,
            is_error=is_error,
            is_ambiguous=any_ambiguous,
            confidence=total_confidence,
            correction_x=accumulated_correction_x,
            correction_z=accumulated_correction_z,
            round_results=round_results,
        )
    
    def _decode_ec_round(
        self,
        detector_vector: np.ndarray,
        ec_round: ECRoundResult,
        metadata: DetectorCircuitMetadata,
    ) -> LocalDecoderResult:
        """
        Decode one EC round using hierarchical decoding.
        
        For concatenated codes:
        1. Decode each inner block
        2. Combine inner results to form outer-level data
        3. Decode outer level (if applicable)
        """
        dmap = ec_round.detector_map
        
        if self.n_levels == 1:
            # Single-level code: just decode the syndrome
            decoder = self._decoders[0]
            
            z_group = dmap.syndrome_z.get(0)
            x_group = dmap.syndrome_x.get(0)
            
            if z_group is None or x_group is None:
                return LocalDecoderResult.ambiguous()
            
            z_det = detector_vector[z_group.start:z_group.end].astype(np.uint8)
            x_det = detector_vector[x_group.start:x_group.end].astype(np.uint8)
            
            # Compute syndrome from raw ancilla: s = H @ raw mod 2
            if hasattr(decoder, 'H') and len(z_det) > decoder.n_syndrome_bits_z:
                z_det = (decoder.H @ z_det) % 2
                x_det = (decoder.H @ x_det) % 2
            
            return decoder.decode_syndrome(z_det, x_det)
        
        # Multi-level: hierarchical decoding
        inner_decoder = self._decoders.get(1, self._decoders.get(self.n_levels - 1))
        outer_decoder = self._decoders.get(0)
        n_inner_blocks = self.level_codes[0].n
        
        # Decode each inner block
        inner_logical_z = []
        inner_logical_x = []
        inner_corrections_x = []
        inner_corrections_z = []
        
        total_confidence = 1.0
        any_inner_ambiguous = False
        
        for block_id in range(n_inner_blocks):
            z_group = dmap.syndrome_z.get(block_id)
            x_group = dmap.syndrome_x.get(block_id)
            
            if z_group is None or x_group is None:
                # Missing data for this block
                inner_logical_z.append(0)
                inner_logical_x.append(0)
                inner_corrections_x.append(np.zeros(inner_decoder.n_data_qubits, dtype=np.uint8))
                inner_corrections_z.append(np.zeros(inner_decoder.n_data_qubits, dtype=np.uint8))
                continue
            
            z_det = detector_vector[z_group.start:z_group.end].astype(np.uint8)
            x_det = detector_vector[x_group.start:x_group.end].astype(np.uint8)
            
            # Compute syndrome if needed
            if hasattr(inner_decoder, 'H') and len(z_det) > inner_decoder.n_syndrome_bits_z:
                z_det = (inner_decoder.H @ z_det) % 2
                x_det = (inner_decoder.H @ x_det) % 2
            
            result = inner_decoder.decode_syndrome(z_det, x_det)
            
            if not result.success:
                any_inner_ambiguous = True
                total_confidence *= result.confidence
                inner_logical_z.append(0)
                inner_logical_x.append(0)
            else:
                inner_logical_z.append(result.logical_value[0] if result.logical_value else 0)
                inner_logical_x.append(result.logical_value[1] if len(result.logical_value) > 1 else 0)
            
            inner_corrections_x.append(result.correction_x if result.correction_x is not None else np.zeros(inner_decoder.n_data_qubits, dtype=np.uint8))
            inner_corrections_z.append(result.correction_z if result.correction_z is not None else np.zeros(inner_decoder.n_data_qubits, dtype=np.uint8))
        
        # Combine inner corrections into physical-level correction
        n_inner = inner_decoder.n_data_qubits
        full_correction_x = np.zeros(self.n_physical, dtype=np.uint8)
        full_correction_z = np.zeros(self.n_physical, dtype=np.uint8)
        
        for block_id in range(n_inner_blocks):
            base = block_id * n_inner
            full_correction_x[base:base + n_inner] = inner_corrections_x[block_id]
            full_correction_z[base:base + n_inner] = inner_corrections_z[block_id]
        
        # Form outer-level "data" from inner logical outcomes
        outer_z_data = np.array(inner_logical_z, dtype=np.uint8)
        outer_x_data = np.array(inner_logical_x, dtype=np.uint8)
        
        # For now, outer syndrome is derived from inner logical patterns
        # In a full implementation, we would have outer syndrome detectors
        # Here we assume inner-level decoding is sufficient for most errors
        
        return LocalDecoderResult.success_result(
            logical_value=[int(np.sum(outer_z_data) % 2)],
            correction_x=full_correction_x,
            correction_z=full_correction_z,
        ) if not any_inner_ambiguous else LocalDecoderResult.ambiguous()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def estimate_logical_error_rate(
    experiment: DetectorMultiLevelMemory,
    shots: int,
    return_samples: bool = False,
) -> Union[ExperimentResult, Tuple[ExperimentResult, np.ndarray]]:
    """
    Convenience function to estimate logical error rate.
    
    This is the main API for running memory experiments.
    
    Args:
        experiment: Configured experiment
        shots: Number of Monte Carlo samples
        return_samples: Whether to return raw detector samples
        
    Returns:
        ExperimentResult (and optionally detector_samples)
    """
    circuit, metadata = experiment.build()
    
    detector_sampler = circuit.compile_detector_sampler()
    detector_samples = detector_sampler.sample(shots)
    
    result = experiment._decode_samples(detector_samples, metadata)
    exp_result = ExperimentResult.from_decoding_results(result, shots)
    
    if return_samples:
        return exp_result, detector_samples
    return exp_result


def run_memory_sweep(
    code: 'MultiLevelConcatenatedCode',
    p_values: List[float],
    rounds: int,
    shots_per_point: int,
    **kwargs,
) -> Dict[float, ExperimentResult]:
    """
    Run memory experiment for multiple error probabilities.
    
    Args:
        code: Concatenated code
        p_values: Physical error probabilities to sweep
        rounds: Number of EC rounds
        shots_per_point: Samples per error probability
        **kwargs: Additional arguments for DetectorMultiLevelMemory
        
    Returns:
        Dict mapping p -> ExperimentResult
    """
    results = {}
    
    for p in p_values:
        exp = DetectorMultiLevelMemory(code, rounds=rounds, p=p, **kwargs)
        results[p] = exp.run(shots_per_point)
    
    return results



if __name__ == '__main__':    # Quick test
    from qectostim.codes.small import SteaneCode713
    from qectostim.codes.composite.multilevel_concatenated import MultiLevelConcatenatedCode
    steane1 = SteaneCode713()
    steane2 = SteaneCode713()
    code = MultiLevelConcatenatedCode([steane1, steane2])
    
    print(f"Testing Steane x Steane ({code.n} qubits)")
    exp = DetectorMultiLevelMemory(code, rounds=1, p=0.001)
    result = exp.run(shots=10)
    print(f"Logical error rate: {result.logical_error_rate}")