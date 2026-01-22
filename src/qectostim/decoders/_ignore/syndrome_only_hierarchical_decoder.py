# src/qectostim/decoders/syndrome_only_hierarchical_decoder.py
"""
Syndrome-Only Hierarchical Decoder for Concatenated Codes.

This decoder processes ONLY syndrome/ancilla measurements, NOT final data measurements.
This is required for true fault-tolerant operation where we cannot afford to measure
data qubits destructively mid-circuit.

Algorithm for [[49,1,9]] = Steane⊗Steane:
=========================================
Each EC round provides:
- 7 inner blocks × 3 Z-syndrome bits = 21 inner Z syndrome bits
- 7 inner blocks × 3 X-syndrome bits = 21 inner X syndrome bits

Step 1: Inner Block Decoding
----------------------------
For each inner block b ∈ {0..6}:
- Take the 3 Z-syndrome bits s₀, s₁, s₂
- Compute error location: i = s₀ + 2*s₁ + 4*s₂
- If i ≠ 0: inner block b has a logical Z flip (X error) with probability
  proportional to whether error is in logical support

Step 2: Outer Code Syndrome
---------------------------
The outer code's X stabilizers detect Z errors on the outer logical qubits.
Each outer X stabilizer spans 4 inner blocks.
If an inner block has a logical Z flip, it contributes to the outer X syndrome.

outer_syndrome[k] = XOR of (inner_has_logical_z_error[b] for b in outer_Hx[k])

Step 3: Outer Decoding
----------------------
Decode the outer syndrome to identify which inner blocks had logical errors.
Apply correction if the error is in the outer logical support {0,1,2}.

Final Output:
- 0 if we predict no logical flip
- 1 if we predict a logical flip
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings

if TYPE_CHECKING:
    import stim
    from qectostim.codes.abstract_code import Code

from .base import Decoder


@dataclass
class SyndromeOnlyConfig:
    """Configuration for syndrome-only hierarchical decoder."""
    verbose: bool = False
    # Use soft decision (probability) vs hard decision (0/1)
    use_soft_inner: bool = False
    # Inner code distance (for confidence weighting)
    inner_distance: int = 3
    # Number of EC rounds
    rounds: int = 1


class SyndromeOnlyHierarchicalDecoder(Decoder):
    """
    Hierarchical decoder using ONLY syndrome measurements (no final data measurements).
    
    For [[49,1,9]] concatenated Steane code:
    - 49 data qubits (7 blocks × 7 qubits)
    - 49 ancilla qubits for syndrome extraction
    
    Per EC round:
    - 7 × 3 = 21 Z-syndrome measurements (one per inner Z stabilizer)
    - 7 × 3 = 21 X-syndrome measurements (one per inner X stabilizer)
    
    The decoder:
    1. Processes inner syndromes to infer inner logical errors
    2. Maps inner logical errors to outer syndrome
    3. Decodes outer syndrome to predict logical flip
    
    CRITICAL: This decoder does NOT use final data measurements.
    """
    
    def __init__(
        self,
        code: Any,
        dem: "stim.DetectorErrorModel",
        rounds: int = 1,
        basis: str = "Z",
        config: Optional[SyndromeOnlyConfig] = None,
        **kwargs,
    ):
        """
        Initialize syndrome-only hierarchical decoder.
        
        Parameters
        ----------
        code : MultiLevelConcatenatedCode
            The concatenated code structure.
        dem : stim.DetectorErrorModel
            The detector error model from the circuit.
        rounds : int
            Number of EC rounds in the experiment.
        basis : str
            Measurement basis ("Z" or "X").
        config : SyndromeOnlyConfig, optional
            Decoder configuration.
        """
        self.code = code
        self.dem = dem
        self.rounds = rounds
        self.basis = basis.upper()
        self.config = config or SyndromeOnlyConfig(rounds=rounds)
        
        # Validate code structure
        self._validate_code()
        
        # Setup inner and outer code matrices
        self._setup_code_structure()
        
        # Build detector-to-syndrome mapping from DEM
        self._build_detector_mapping()
        
        # Track if we're using final data measurements (should be False!)
        self._uses_final_data = False
    
    def _validate_code(self) -> None:
        """Validate that we have a concatenated code."""
        if not hasattr(self.code, 'level_codes') or len(self.code.level_codes) < 2:
            raise ValueError(
                "SyndromeOnlyHierarchicalDecoder requires a concatenated code "
                "with at least 2 levels"
            )
    
    def _setup_code_structure(self) -> None:
        """Setup inner and outer code parity check matrices."""
        outer_code = self.code.level_codes[0]
        inner_code = self.code.level_codes[-1]
        
        # Inner code Hz (for Z syndrome)
        if hasattr(inner_code, 'hz'):
            hz = inner_code.hz
            self._inner_hz = np.atleast_2d(np.array(hz() if callable(hz) else hz, dtype=np.uint8))
        else:
            # Default Steane Hz
            self._inner_hz = np.array([
                [0, 0, 0, 1, 1, 1, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
            ], dtype=np.uint8)
        
        # Inner code Hx (for X syndrome)
        if hasattr(inner_code, 'hx'):
            hx = inner_code.hx
            self._inner_hx = np.atleast_2d(np.array(hx() if callable(hx) else hx, dtype=np.uint8))
        else:
            # Default Steane Hx (same as Hz for self-dual code)
            self._inner_hx = self._inner_hz.copy()
        
        # Outer code matrices
        if hasattr(outer_code, 'hz'):
            hz = outer_code.hz
            self._outer_hz = np.atleast_2d(np.array(hz() if callable(hz) else hz, dtype=np.uint8))
        else:
            self._outer_hz = self._inner_hz.copy()
        
        if hasattr(outer_code, 'hx'):
            hx = outer_code.hx
            self._outer_hx = np.atleast_2d(np.array(hx() if callable(hx) else hx, dtype=np.uint8))
        else:
            self._outer_hx = self._inner_hx.copy()
        
        # Inner logical support
        self._inner_z_support = self._get_logical_support(inner_code, 'Z')
        self._inner_x_support = self._get_logical_support(inner_code, 'X')
        
        # Outer logical support
        self._outer_z_support = set(self._get_logical_support(outer_code, 'Z'))
        self._outer_x_support = set(self._get_logical_support(outer_code, 'X'))
        
        # Number of inner blocks
        self._n_inner_blocks = outer_code.n  # 7 for Steane
        self._n_inner_qubits = inner_code.n  # 7 for Steane
        self._n_inner_z_stabs = self._inner_hz.shape[0]  # 3 for Steane
        self._n_inner_x_stabs = self._inner_hx.shape[0]  # 3 for Steane
    
    def _get_logical_support(self, code: Any, basis: str) -> List[int]:
        """Get qubit indices in logical operator support."""
        basis = basis.upper()
        
        # Try lz/lx properties
        if basis == 'Z' and hasattr(code, 'lz'):
            lz = getattr(code, 'lz')
            if callable(lz):
                lz = lz()
            if isinstance(lz, np.ndarray):
                lz = np.atleast_2d(lz)
                if lz.shape[0] > 0:
                    return list(np.where(lz[0] != 0)[0])
        
        if basis == 'X' and hasattr(code, 'lx'):
            lx = getattr(code, 'lx')
            if callable(lx):
                lx = lx()
            if isinstance(lx, np.ndarray):
                lx = np.atleast_2d(lx)
                if lx.shape[0] > 0:
                    return list(np.where(lx[0] != 0)[0])
        
        # Try logical_z/logical_x string format
        if basis == 'Z' and hasattr(code, 'logical_z'):
            ops = code.logical_z
            if isinstance(ops, list) and len(ops) > 0:
                op = ops[0]
                if isinstance(op, str):
                    return [i for i, c in enumerate(op) if c.upper() in ('Z', 'Y')]
        
        if basis == 'X' and hasattr(code, 'logical_x'):
            ops = code.logical_x
            if isinstance(ops, list) and len(ops) > 0:
                op = ops[0]
                if isinstance(op, str):
                    return [i for i, c in enumerate(op) if c.upper() in ('X', 'Y')]
        
        # Default: first 3 qubits (Steane code)
        return [0, 1, 2]
    
    def _build_detector_mapping(self) -> None:
        """
        Build mapping from detector indices to syndrome positions.
        
        For [[49,1,9]], the DEM has detectors for:
        - 7 blocks × 3 inner Z stabilizers = 21 inner Z detectors
        - 7 blocks × 3 inner X stabilizers = 21 inner X detectors
        - 3 outer Z metachecks
        - 3 outer X metachecks
        
        Total = 48 detectors per round (for standard memory)
        Plus time-like detectors comparing rounds.
        """
        # For now, use a simple mapping based on detector ordering
        # TODO: Parse detector coordinates from DEM for proper mapping
        
        self._n_detectors = self.dem.num_detectors
        
        # Each round has:
        # - 21 inner Z syndrome detectors (blocks 0-6, stabs 0-2)
        # - 21 inner X syndrome detectors (blocks 0-6, stabs 0-2)
        # - 6 outer metachecks (3 Z + 3 X)
        self._detectors_per_round = 48  # Approximate
        
        # Build block -> detector mapping
        # This is approximate; proper implementation should use detector coords
        self._inner_z_detectors = {}  # block_id -> [det_indices]
        self._inner_x_detectors = {}  # block_id -> [det_indices]
        self._outer_z_detectors = []  # [det_indices]
        self._outer_x_detectors = []  # [det_indices]
        
        # Simple linear mapping (needs refinement based on actual circuit structure)
        for block in range(self._n_inner_blocks):
            self._inner_z_detectors[block] = list(range(
                block * self._n_inner_z_stabs,
                (block + 1) * self._n_inner_z_stabs
            ))
            self._inner_x_detectors[block] = list(range(
                21 + block * self._n_inner_x_stabs,
                21 + (block + 1) * self._n_inner_x_stabs
            ))
        
        # Outer metachecks (approximate position)
        self._outer_z_detectors = list(range(42, 45))
        self._outer_x_detectors = list(range(45, 48))
    
    def decode(self, detector_events: np.ndarray) -> np.ndarray:
        """
        Decode using syndrome-only hierarchical approach.
        
        Parameters
        ----------
        detector_events : np.ndarray
            Array of shape (n_shots, n_detectors) with detector outcomes.
            
        Returns
        -------
        np.ndarray
            Array of shape (n_shots, n_observables) with predicted corrections.
        """
        # Ensure 2D
        if detector_events.ndim == 1:
            detector_events = detector_events.reshape(1, -1)
        
        n_shots = detector_events.shape[0]
        predictions = np.zeros((n_shots, 1), dtype=np.uint8)
        
        for shot_idx in range(n_shots):
            events = detector_events[shot_idx]
            prediction = self._decode_single_shot(events)
            predictions[shot_idx, 0] = prediction
        
        return predictions
    
    def _decode_single_shot(self, detector_events: np.ndarray) -> int:
        """
        Decode a single shot using hierarchical syndrome processing.
        
        Steps:
        1. Extract inner block syndromes
        2. Decode each inner block → logical error flag
        3. Compute outer syndrome from inner logical errors
        4. Decode outer syndrome → final prediction
        """
        # Step 1: Extract inner syndromes
        inner_z_syndromes = {}
        inner_x_syndromes = {}
        
        for block_id in range(self._n_inner_blocks):
            z_dets = self._inner_z_detectors.get(block_id, [])
            x_dets = self._inner_x_detectors.get(block_id, [])
            
            z_syndrome = np.array([
                detector_events[d] if d < len(detector_events) else 0
                for d in z_dets
            ], dtype=np.uint8)
            
            x_syndrome = np.array([
                detector_events[d] if d < len(detector_events) else 0
                for d in x_dets
            ], dtype=np.uint8)
            
            inner_z_syndromes[block_id] = z_syndrome
            inner_x_syndromes[block_id] = x_syndrome
        
        # Step 2: Decode each inner block
        inner_z_errors = np.zeros(self._n_inner_blocks, dtype=np.uint8)  # Inner logical Z errors
        inner_x_errors = np.zeros(self._n_inner_blocks, dtype=np.uint8)  # Inner logical X errors
        
        for block_id in range(self._n_inner_blocks):
            # Decode X errors (detected by Z syndrome)
            z_syndrome = inner_z_syndromes[block_id]
            x_error_location = self._decode_inner_syndrome(z_syndrome)
            
            # Check if error is in logical Z support
            if x_error_location > 0:  # Error detected
                error_qubit = x_error_location - 1
                if error_qubit in self._inner_z_support:
                    inner_z_errors[block_id] = 1  # X error on Z_L support → Z_L flip
            
            # Decode Z errors (detected by X syndrome)
            x_syndrome = inner_x_syndromes[block_id]
            z_error_location = self._decode_inner_syndrome(x_syndrome)
            
            # Check if error is in logical X support
            if z_error_location > 0:  # Error detected
                error_qubit = z_error_location - 1
                if error_qubit in self._inner_x_support:
                    inner_x_errors[block_id] = 1  # Z error on X_L support → X_L flip
        
        # Step 3: Compute outer syndrome from inner logical errors
        # For Z-basis memory:
        # - X errors on inner blocks cause Z_L flips
        # - Outer Hz detects these as outer Z syndrome
        outer_z_syndrome = (self._outer_hz @ inner_z_errors) % 2
        
        # Step 4: Decode outer code
        outer_error_location = self._decode_outer_syndrome(outer_z_syndrome)
        
        # Step 5: Compute final prediction
        # The logical Z observable is XOR of Z_L on outer logical support blocks
        logical_parity = np.sum(inner_z_errors[list(self._outer_z_support)]) % 2
        
        # If outer code detected an error, apply correction
        if outer_error_location > 0:
            error_block = outer_error_location - 1
            if error_block in self._outer_z_support:
                # Flip the prediction
                logical_parity = (logical_parity + 1) % 2
        
        return int(logical_parity)
    
    def _decode_inner_syndrome(self, syndrome: np.ndarray) -> int:
        """
        Decode a 3-bit Steane syndrome to error location.
        
        For Steane [[7,1,3]]:
        - syndrome = [s0, s1, s2]
        - error_location = s0 + 2*s1 + 4*s2
        - Returns 0 if no error, 1-7 if error on qubit 0-6
        
        Parameters
        ----------
        syndrome : np.ndarray
            3-bit syndrome vector.
            
        Returns
        -------
        int
            Error location: 0=no error, 1-7=error on qubit i-1
        """
        if len(syndrome) < 3:
            syndrome = np.pad(syndrome, (0, 3 - len(syndrome)))
        
        # Steane syndrome decoding
        location = int(syndrome[0]) + 2 * int(syndrome[1]) + 4 * int(syndrome[2])
        return location
    
    def _decode_outer_syndrome(self, syndrome: np.ndarray) -> int:
        """
        Decode the outer code syndrome.
        
        Same as inner for Steane outer code.
        """
        return self._decode_inner_syndrome(syndrome)
    
    def decode_batch(
        self,
        detector_events: np.ndarray,
        num_shots: Optional[int] = None,
    ) -> np.ndarray:
        """Batch decode interface."""
        return self.decode(detector_events)
    
    @property
    def uses_final_data_measurements(self) -> bool:
        """Return True if decoder uses final data measurements (should be False!)."""
        return self._uses_final_data


class SyndromeOnlyDecoderIncompatibleError(Exception):
    """Raised when decoder is incompatible with code/experiment structure."""
    pass
