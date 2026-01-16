# src/qectostim/decoders/joint_minimum_weight_decoder.py
"""
Joint Minimum-Weight Decoder for Concatenated CSS Codes.

This decoder achieves true p^((d+1)/2) scaling for concatenated codes by
doing minimum-weight decoding on the FULL concatenated code, rather than
hierarchical inner/outer decoding.

KEY INSIGHT: Why Joint Decoding is Necessary
--------------------------------------------
Hierarchical decoding (decode inner blocks, then outer code) CANNOT achieve
optimal scaling because inner decoders make hard decisions that lose information.

Example: [[49,1,9]] = [[7,1,3]] ⊗ [[7,1,3]] with error pattern [2,2,0,0,0,0,0]
- Physical weight: 4 (correctable, since d=9 handles up to 4)
- Hierarchical view: Blocks 0 and 1 each have 2 errors
  - Inner decoder for block 0: 2 errors → wrong syndrome match → wrong logical
  - Inner decoder for block 1: 2 errors → wrong syndrome match → wrong logical  
  - Outer decoder sees 2 "logical" errors → can only correct 1 → FAILURE
- Joint view: Weight-4 physical error with specific syndrome
  - Find minimum-weight correction (4 qubits) → exact match → SUCCESS

Integration with Production Workflow:
------------------------------------
    >>> from qectostim.codes.composite import MultiLevelConcatenatedCode
    >>> from qectostim.codes.small import SteaneCode713
    >>> from qectostim.experiments import MultiLevelMemoryExperiment
    >>> from qectostim.noise import CircuitDepolarizingNoise
    >>> from qectostim.decoders import JointMinWeightDecoder
    >>> 
    >>> # 1. Create concatenated code
    >>> code = MultiLevelConcatenatedCode([SteaneCode713(), SteaneCode713()])
    >>> 
    >>> # 2. Create experiment with EC gadgets
    >>> exp = MultiLevelMemoryExperiment(code, rounds=1, ec_method="steane")
    >>> circuit, metadata = exp.build()
    >>> 
    >>> # 3. Apply noise (experiment concern, NOT decoder's)
    >>> noise = CircuitDepolarizingNoise(p1=0.01, p2=0.01)
    >>> noisy_circuit = noise.apply(circuit)
    >>> 
    >>> # 4. Sample circuit
    >>> sample = noisy_circuit.compile_sampler().sample(1)[0]
    >>> 
    >>> # 5. Extract final data from sample using metadata
    >>> final_data = extract_final_data(sample, metadata)
    >>> 
    >>> # 6. Decode (noise-agnostic - only uses code structure)
    >>> decoder = JointMinWeightDecoder.from_code(code)
    >>> logical = decoder.decode(final_data)

Theoretical Foundation:
----------------------
For CSS code with distance d, minimum-weight decoding is optimal ML when p < 0.5.
This is because lower weight error patterns have higher probability.

For [[49,1,9]] = [[7,1,3]] ⊗ [[7,1,3]]:
- Distance: 9 → corrects up to 4 errors
- Lookup table size: ~232K entries (all weight 0-4 syndromes)
- Achieves p^5 scaling (first failures at weight 5)
"""
from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from qectostim.codes.composite.multilevel_concatenated import MultiLevelConcatenatedCode
    from qectostim.experiments.multilevel_memory import MultiLevelMetadata


@dataclass
class JointDecoderConfig:
    """Configuration for Joint Minimum-Weight Decoder."""
    # Maximum error weight to include in lookup table
    # Should be floor((d-1)/2) for distance d code
    max_weight: int = 4
    
    # Output options
    verbose: bool = False
    return_correction: bool = False


@dataclass 
class JointDecodeResult:
    """Result from joint decoder."""
    logical_x: int  # Decoded X logical value (0 or 1)
    logical_z: int  # Decoded Z logical value (0 or 1)
    x_correction_weight: int = 0  # Weight of X correction applied
    z_correction_weight: int = 0  # Weight of Z correction applied


class JointMinWeightDecoder:
    """
    Joint minimum-weight decoder for concatenated CSS codes.
    
    This decoder is NOISE-AGNOSTIC:
    - Only receives measurement data
    - Only uses code structure (Hz, Hx, logical operators)
    - Minimum-weight decoding IS optimal ML for any i.i.d. noise with p < 0.5
    
    For [[49,1,9]], this achieves true p^5 scaling by:
    1. Building lookup table for ALL weight 0-4 error syndromes (~232K entries)
    2. Finding minimum-weight correction for any syndrome
    3. Correctly handling cross-block error patterns that hierarchical fails on
    
    Attributes:
        Hz: Full Z-stabilizer parity check matrix (detects X errors)
        Hx: Full X-stabilizer parity check matrix (detects Z errors)
        ZL: Qubit indices for Z logical operator (computes X logical)
        XL: Qubit indices for X logical operator (computes Z logical)
    """
    
    def __init__(
        self,
        Hz: np.ndarray,
        Hx: np.ndarray,
        ZL: List[int],
        XL: List[int],
        config: Optional[JointDecoderConfig] = None,
    ):
        """
        Initialize joint minimum-weight decoder.
        
        Args:
            Hz: Z-stabilizer parity check matrix (n_z_checks × n_qubits)
            Hx: X-stabilizer parity check matrix (n_x_checks × n_qubits)
            ZL: Qubit indices for Z logical operator (for X logical value)
            XL: Qubit indices for X logical operator (for Z logical value)
            config: Decoder configuration
        """
        self.Hz = np.asarray(Hz, dtype=np.uint8)
        self.Hx = np.asarray(Hx, dtype=np.uint8)
        self.ZL = set(ZL)
        self.XL = set(XL)
        self.n_qubits = Hz.shape[1]
        self.config = config or JointDecoderConfig()
        
        if self.config.verbose:
            print(f"JointMinWeightDecoder initializing:")
            print(f"  n_qubits: {self.n_qubits}")
            print(f"  Hz shape: {self.Hz.shape}")
            print(f"  Hx shape: {self.Hx.shape}")
            print(f"  Z_L support: {sorted(self.ZL)} (weight {len(self.ZL)})")
            print(f"  X_L support: {sorted(self.XL)} (weight {len(self.XL)})")
            print(f"  max_weight: {self.config.max_weight}")
        
        # Build syndrome lookup tables
        self._x_table = self._build_syndrome_table(self.Hz, "X")
        self._z_table = self._build_syndrome_table(self.Hx, "Z")
        
        if self.config.verbose:
            print(f"  X syndrome table: {len(self._x_table)} entries")
            print(f"  Z syndrome table: {len(self._z_table)} entries")
    
    def _build_syndrome_table(
        self, 
        H: np.ndarray,
        error_type: str,
    ) -> Dict[Tuple[int, ...], Tuple[np.ndarray, int]]:
        """
        Build syndrome → (minimum weight correction, weight) lookup table.
        
        Enumerates all error patterns up to max_weight and stores the first
        (minimum weight) correction for each syndrome.
        """
        table: Dict[Tuple[int, ...], Tuple[np.ndarray, int]] = {}
        n_checks = H.shape[0]
        
        if self.config.verbose:
            print(f"  Building {error_type} syndrome table...")
        
        # Weight 0 (no error)
        syn0 = tuple([0] * n_checks)
        table[syn0] = (np.zeros(self.n_qubits, dtype=np.uint8), 0)
        
        # Weights 1 to max_weight
        for weight in range(1, self.config.max_weight + 1):
            count = 0
            for qubits in combinations(range(self.n_qubits), weight):
                e = np.zeros(self.n_qubits, dtype=np.uint8)
                for q in qubits:
                    e[q] = 1
                
                syn = tuple((H @ e) % 2)
                
                # Only store if this syndrome not seen (minimum weight)
                if syn not in table:
                    table[syn] = (e.copy(), weight)
                    count += 1
            
            if self.config.verbose:
                print(f"    Weight {weight}: added {count} new syndromes")
        
        return table
    
    @classmethod
    def from_code(
        cls,
        code: 'MultiLevelConcatenatedCode',
        config: Optional[JointDecoderConfig] = None,
    ) -> 'JointMinWeightDecoder':
        """
        Create decoder from a MultiLevelConcatenatedCode.
        
        This is the recommended way to create the decoder in production.
        Automatically constructs the full Hz, Hx, and logical operator supports.
        
        Args:
            code: The concatenated code object
            config: Decoder configuration (default: max_weight = (d-1)//2)
            
        Returns:
            JointMinWeightDecoder configured for the code
        """
        # Get full parity check matrices from code
        Hz = np.asarray(code.hz, dtype=np.uint8)
        Hx = np.asarray(code.hx, dtype=np.uint8)
        
        # Get logical operator supports
        ZL = cls._get_logical_support(code, 'Z')
        XL = cls._get_logical_support(code, 'X')
        
        # Set default max_weight based on distance
        if config is None:
            d = getattr(code, 'd', getattr(code, 'distance', 9))
            max_weight = (d - 1) // 2
            config = JointDecoderConfig(max_weight=max_weight)
        
        return cls(Hz=Hz, Hx=Hx, ZL=ZL, XL=XL, config=config)
    
    @classmethod
    def _get_logical_support(cls, code: Any, logical_type: str) -> List[int]:
        """Extract logical operator support from code."""
        # Try logical_z/logical_x property (string format)
        attr_name = f'logical_{logical_type.lower()}'
        if hasattr(code, attr_name):
            ops = getattr(code, attr_name)
            if isinstance(ops, list) and len(ops) > 0:
                op = ops[0]
                if isinstance(op, str):
                    target = logical_type.upper()
                    return [i for i, c in enumerate(op) if c.upper() in (target, 'Y')]
        
        # Try lz/lx method (numpy format)
        method_name = f'l{logical_type.lower()}'
        if hasattr(code, method_name):
            lop = getattr(code, method_name)
            if callable(lop):
                lop = lop()
            if isinstance(lop, np.ndarray):
                lop = np.atleast_2d(lop)
                if lop.shape[0] > 0:
                    return list(np.where(lop[0] != 0)[0])
        
        # Try logical_z_ops/logical_x_ops (can be string or dict format)
        ops_name = f'logical_{logical_type.lower()}_ops'
        if hasattr(code, ops_name):
            ops = getattr(code, ops_name)
            if ops and len(ops) > 0:
                op = ops[0]
                if isinstance(op, str):
                    target = logical_type.upper()
                    return [i for i, c in enumerate(op) if c.upper() in (target, 'Y')]
                elif isinstance(op, dict):
                    # Dict format: {qubit_idx: 'Z', ...}
                    target = logical_type.upper()
                    return [k for k, v in op.items() if v.upper() in (target, 'Y')]
        
        # Fallback: construct from concatenated structure
        # For [[n₁,1,d₁]] ⊗ [[n₂,1,d₂]], Z_L is inner_ZL ⊗ outer_ZL
        if hasattr(code, 'level_codes') and len(code.level_codes) >= 2:
            return cls._construct_logical_support_from_levels(code, logical_type)
        
        raise ValueError(f"Could not determine {logical_type} logical support from code")
    
    @classmethod
    def _construct_logical_support_from_levels(
        cls,
        code: Any, 
        logical_type: str,
    ) -> List[int]:
        """Construct logical support from level codes."""
        # Get inner and outer code logical supports
        inner_code = code.level_codes[-1]  # Innermost
        outer_code = code.level_codes[0]   # Outermost
        n_inner = inner_code.n
        
        # Get inner logical support
        inner_support = cls._get_level_logical_support(
            inner_code, logical_type
        )
        
        # Get outer logical blocks
        outer_support = cls._get_level_logical_support(
            outer_code, logical_type
        )
        
        # Full support is inner_support positions within outer_support blocks
        full_support = []
        for block in outer_support:
            for q in inner_support:
                full_support.append(block * n_inner + q)
        
        return full_support
    
    @staticmethod
    def _get_level_logical_support(code: Any, logical_type: str) -> List[int]:
        """Get logical support for a single level code."""
        # Try string format
        attr_name = f'logical_{logical_type.lower()}'
        if hasattr(code, attr_name):
            ops = getattr(code, attr_name)
            if isinstance(ops, list) and len(ops) > 0:
                op = ops[0]
                if isinstance(op, str):
                    target = logical_type.upper()
                    return [i for i, c in enumerate(op) if c.upper() in (target, 'Y')]
        
        # Try numpy format
        method_name = f'l{logical_type.lower()}'
        if hasattr(code, method_name):
            lop = getattr(code, method_name)
            if callable(lop):
                lop = lop()
            if isinstance(lop, np.ndarray):
                lop = np.atleast_2d(lop)
                if lop.shape[0] > 0:
                    return list(np.where(lop[0] != 0)[0])
        
        # Code-specific fallbacks
        code_name = getattr(code, 'name', '') or type(code).__name__
        n = code.n
        
        if 'steane' in code_name.lower() or 'stean' in code_name.lower():
            return [0, 1, 2]  # Standard Steane Z_L = X_L
        if 'shor' in code_name.lower():
            return list(range(n))  # All qubits
        if 'hamming' in code_name.lower():
            return [0, 1, 2]
        
        # Generic: first few qubits
        return list(range(min(3, n)))
    
    @classmethod
    def for_steane_steane(
        cls,
        config: Optional[JointDecoderConfig] = None,
    ) -> 'JointMinWeightDecoder':
        """
        Factory method for [[49,1,9]] = [[7,1,3]] ⊗ [[7,1,3]].
        
        Returns decoder with precomputed Hz/Hx for Steane ⊗ Steane.
        Default max_weight=4 for p^5 scaling.
        """
        if config is None:
            config = JointDecoderConfig(max_weight=4)
        
        # Steane code parity check matrix (3×7)
        steane_Hz = np.array([
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
        ], dtype=np.uint8)
        
        # Build full 49-qubit Hz (block diagonal inner + lifted outer)
        n_inner = 7
        n_blocks = 7
        n_inner_checks = 3
        n_outer_checks = 3
        inner_ZL = [0, 1, 2]
        outer_ZL_blocks = [0, 1, 2]
        
        # Total checks = inner_checks × n_blocks + outer_checks
        n_total_checks = n_inner_checks * n_blocks + n_outer_checks
        full_Hz = np.zeros((n_total_checks, n_inner * n_blocks), dtype=np.uint8)
        
        # Inner stabilizers (block diagonal)
        for block in range(n_blocks):
            for stab in range(n_inner_checks):
                row = block * n_inner_checks + stab
                for q in range(n_inner):
                    if steane_Hz[stab, q] == 1:
                        full_Hz[row, block * n_inner + q] = 1
        
        # Outer stabilizers (act on inner Z_L positions)
        for stab in range(n_outer_checks):
            row = n_inner_checks * n_blocks + stab
            for block in range(n_blocks):
                if steane_Hz[stab, block] == 1:
                    for q in inner_ZL:
                        full_Hz[row, block * n_inner + q] = 1
        
        # For CSS Steane code, Hx = Hz (self-dual)
        full_Hx = full_Hz.copy()
        
        # Z logical support
        full_ZL = []
        for block in outer_ZL_blocks:
            for q in inner_ZL:
                full_ZL.append(block * n_inner + q)
        
        # X logical support (same for self-dual)
        full_XL = full_ZL.copy()
        
        return cls(Hz=full_Hz, Hx=full_Hx, ZL=full_ZL, XL=full_XL, config=config)
    
    def decode(self, final_data: np.ndarray) -> JointDecodeResult:
        """
        Decode measurement data to logical values.
        
        This decoder is NOISE-AGNOSTIC. It only uses:
        1. The measurement data (final_data)
        2. The code structure (Hz, Hx, logical operators)
        
        Minimum-weight decoding is optimal for any i.i.d. noise with p < 0.5.
        
        Args:
            final_data: Final measurement data (n_qubits bits)
            
        Returns:
            JointDecodeResult with logical values
        """
        data = np.asarray(final_data, dtype=np.uint8).flatten()
        
        if len(data) != self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} data bits, got {len(data)}"
            )
        
        # Decode X errors (using Hz)
        x_syn = tuple((self.Hz @ data) % 2)
        if x_syn in self._x_table:
            x_correction, x_weight = self._x_table[x_syn]
        else:
            x_correction = np.zeros(self.n_qubits, dtype=np.uint8)
            x_weight = 0
        
        x_corrected = (data + x_correction) % 2
        logical_x = sum(x_corrected[q] for q in self.ZL) % 2
        
        # Decode Z errors (using Hx)
        z_syn = tuple((self.Hx @ data) % 2)
        if z_syn in self._z_table:
            z_correction, z_weight = self._z_table[z_syn]
        else:
            z_correction = np.zeros(self.n_qubits, dtype=np.uint8)
            z_weight = 0
        
        z_corrected = (data + z_correction) % 2
        logical_z = sum(z_corrected[q] for q in self.XL) % 2
        
        return JointDecodeResult(
            logical_x=int(logical_x),
            logical_z=int(logical_z),
            x_correction_weight=x_weight,
            z_correction_weight=z_weight,
        )
    
    def decode_x_only(self, final_data: np.ndarray) -> int:
        """
        Decode X errors only, return X logical value.
        
        Args:
            final_data: Final measurement data
            
        Returns:
            X logical value (0 or 1)
        """
        return self.decode(final_data).logical_x
    
    def decode_batch(
        self, 
        final_data_batch: np.ndarray,
    ) -> np.ndarray:
        """
        Decode batch of shots.
        
        Args:
            final_data_batch: Array of shape (n_shots, n_qubits)
            
        Returns:
            Array of n_shots X logical values
        """
        n_shots = final_data_batch.shape[0]
        results = np.zeros(n_shots, dtype=np.uint8)
        
        for i in range(n_shots):
            results[i] = self.decode(final_data_batch[i]).logical_x
        
        return results


# =============================================================================
# Utility functions for production workflow integration
# =============================================================================

def extract_final_data(
    sample: np.ndarray,
    metadata: 'MultiLevelMetadata',
    apply_frame_correction: bool = True,
) -> np.ndarray:
    """
    Extract final data measurements from circuit sample.
    
    For teleportation-based EC (Knill), the output state has a Pauli frame:
        |ψ_out⟩ = X^{m_anc1} Z^{m_data} |ψ_original⟩
    
    When measuring in Z basis to compute Z_L, the X frame affects the result:
        <Z_L> → X^m Z_L = (-1)^{m·Z_L} Z_L
    
    So we need to XOR the frame measurements on Z_L support with the final data.
    This is done automatically when apply_frame_correction=True (default).
    
    Args:
        sample: Full measurement sample from circuit
        metadata: MultiLevelMetadata from experiment.build()
        apply_frame_correction: If True (default), apply Pauli frame correction
            for teleportation-based EC. Set to False for raw data extraction.
        
    Returns:
        Array of n_physical_qubits final data measurements (corrected if applicable)
    """
    n_physical = metadata.n_physical_qubits
    total_meas = metadata.total_measurements
    
    # Final data is the last n_physical measurements
    # (after all syndrome measurements)
    final_start = total_meas - n_physical
    
    final_data = np.asarray(sample[final_start:final_start + n_physical], dtype=np.uint8)
    
    # Apply Pauli frame correction for teleportation-based EC
    if apply_frame_correction and metadata.uses_teleportation_ec:
        final_data = _apply_pauli_frame_correction(sample, metadata, final_data)
    
    return final_data


def _apply_pauli_frame_correction(
    sample: np.ndarray,
    metadata: 'MultiLevelMetadata',
    final_data: np.ndarray,
) -> np.ndarray:
    """
    Apply Pauli frame correction to final data for teleportation-based EC.
    
    The teleportation protocol produces output: |ψ_out⟩ = X^{m_anc1} |ψ⟩
    where m_anc1 are the Bell measurement outcomes on ancilla1.
    
    For Z_L measurement, we need: corrected[i] = final_data[i] ⊕ frame_X[i]
    where frame_X[i] is the accumulated X Pauli frame on qubit i.
    
    For concatenated codes with recursive teleportation EC:
    - Outer level teleportation: frame on full Z_L support (physical qubits)
    - Inner level teleportation: frame on inner Z_L within each outer Z_L block
    
    Both levels contribute independently and must be XORed.
    
    Args:
        sample: Full measurement sample
        metadata: MultiLevelMetadata with pauli_frame_measurements
        final_data: Raw final data measurements
        
    Returns:
        Corrected final data with Pauli frame applied
    """
    corrected = final_data.copy()
    
    if not metadata.pauli_frame_measurements:
        return corrected
    
    # Get Z_L support for each level
    level_z_supports = metadata.level_z_supports
    if not level_z_supports:
        return corrected
    
    inner_z_support = level_z_supports[-1] if level_z_supports else []
    outer_z_support = level_z_supports[0] if len(level_z_supports) > 1 else list(range(metadata.n_qubits_per_level[0]))
    n_inner = metadata.n_qubits_per_level[-1] if metadata.n_qubits_per_level else 7
    depth = metadata.depth
    
    # Compute full Z_L support (physical qubit indices)
    # For 2-level: Z_L = inner_z_support within each outer_z_support block
    full_zl_support = []
    for block in outer_z_support:
        for q in inner_z_support:
            full_zl_support.append(block * n_inner + q)
    
    # Accumulate frame correction across all rounds
    for round_idx, round_frames in metadata.pauli_frame_measurements.items():
        if 'X' not in round_frames:
            continue
            
        for key, frame_indices in round_frames['X'].items():
            if len(frame_indices) == 0:
                continue
            
            # Determine level and block_id from key
            if isinstance(key, tuple) and len(key) == 2:
                level, block_id = key
            else:
                level = depth - 1  # Assume innermost
                block_id = key
            
            # Handle based on level
            if level == 0:
                # OUTER level frame: applies to full Z_L support
                # frame_indices[phys_q] is the frame measurement for physical qubit phys_q
                for phys_q in full_zl_support:
                    if phys_q < len(frame_indices):
                        frame_meas_idx = frame_indices[phys_q]
                        if frame_meas_idx < len(sample) and phys_q < len(corrected):
                            corrected[phys_q] ^= sample[frame_meas_idx]
            
            elif level == depth - 1:
                # INNER level frame: only blocks on outer Z support
                if block_id not in outer_z_support:
                    continue
                
                # Apply frame correction on inner Z_L support within this block
                for local_idx in inner_z_support:
                    if local_idx < len(frame_indices):
                        frame_meas_idx = frame_indices[local_idx]
                        if frame_meas_idx < len(sample):
                            phys_q = block_id * n_inner + local_idx
                            if phys_q < len(corrected):
                                corrected[phys_q] ^= sample[frame_meas_idx]
    
    return corrected


def extract_final_data_by_block(
    sample: np.ndarray,
    metadata: 'MultiLevelMetadata',
    apply_frame_correction: bool = True,
) -> Dict[Tuple[int, ...], np.ndarray]:
    """
    Extract final data measurements organized by block.
    
    Args:
        sample: Full measurement sample from circuit
        metadata: MultiLevelMetadata from experiment.build()
        apply_frame_correction: If True (default), apply Pauli frame correction
            for teleportation-based EC.
        
    Returns:
        Dict mapping block address → array of final measurements for that block
    """
    # Get corrected full data first
    full_data = extract_final_data(sample, metadata, apply_frame_correction)
    
    # Organize by block
    result = {}
    for address, indices in metadata.final_data_measurements.items():
        # Map measurement indices back to physical qubit positions
        # indices are absolute measurement positions in the circuit
        n_physical = metadata.n_physical_qubits
        total_meas = metadata.total_measurements
        final_start = total_meas - n_physical
        
        block_data = []
        for meas_idx in indices:
            phys_q = meas_idx - final_start
            if 0 <= phys_q < len(full_data):
                block_data.append(full_data[phys_q])
        result[address] = np.asarray(block_data, dtype=np.uint8)
    
    return result
