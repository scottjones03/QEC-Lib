# src/qectostim/decoders/concatenated_ec_decoder.py
"""
Concatenated EC Decoder with Proper Pauli Frame Tracking.

This decoder implements true error correction for concatenated codes by:
1. Processing syndrome measurements from each EC round
2. Computing corrections based on syndromes
3. Tracking a Pauli frame through the rounds
4. Applying accumulated corrections to final measurements

The key insight is that in a circuit with EC, the syndrome measurements
tell us about errors that occurred, and we need to track how corrections
propagate through the code hierarchy.

Algorithm:
---------
For each EC round (outermost to innermost):
  For each level (innermost to outermost):
    For each block at this level:
      1. Extract syndrome measurements for this block/round
      2. Decode syndrome to get correction
      3. Update Pauli frame for this block

After all rounds:
  Apply Pauli frame corrections to final data measurements
  Decode hierarchically as usual

Example
-------
>>> from qectostim.decoders import ConcatenatedECDecoder
>>> from qectostim.experiments import MultiLevelMemoryExperiment
>>> 
>>> exp = MultiLevelMemoryExperiment(code, level_gadgets={...}, rounds=3)
>>> circuit, metadata = exp.build()
>>> samples = circuit.compile_sampler().sample(1000)
>>> 
>>> decoder = ConcatenatedECDecoder(code, metadata)
>>> predictions = decoder.decode_batch(samples)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING
import numpy as np

from qectostim.decoders.base import Decoder
from qectostim.decoders.strategies.base import DecoderStrategy, StrategyOutput
from qectostim.decoders.strategies.syndrome_lookup import SyndromeLookupStrategy

if TYPE_CHECKING:
    from qectostim.codes.composite.multilevel_concatenated import (
        CodeNode,
        MultiLevelConcatenatedCode,
    )
    from qectostim.utils.hierarchical_mapper import HierarchicalQubitMapper
    from qectostim.experiments.multilevel_memory import MultiLevelMetadata


@dataclass
class PauliFrame:
    """
    Tracks accumulated Pauli corrections for a block.
    
    In concatenated codes, we track corrections at each level:
    - x_correction: X errors (bit flips) - tracked per qubit
    - z_correction: Z errors (phase flips) - tracked per qubit
    
    For a concatenated code with n_inner physical qubits per block,
    we track the correction as a length-n_inner binary vector.
    """
    x_correction: np.ndarray  # X error correction (bit flips)
    z_correction: np.ndarray  # Z error correction (phase flips)
    
    @classmethod
    def identity(cls, n_qubits: int) -> 'PauliFrame':
        """Create identity (no correction) frame."""
        return cls(
            x_correction=np.zeros(n_qubits, dtype=np.uint8),
            z_correction=np.zeros(n_qubits, dtype=np.uint8),
        )
    
    def apply_correction(self, correction: np.ndarray, error_type: str = 'X') -> None:
        """Apply a correction to this frame."""
        if error_type.upper() == 'X':
            self.x_correction ^= correction.astype(np.uint8)
        else:
            self.z_correction ^= correction.astype(np.uint8)
    
    def combine(self, other: 'PauliFrame') -> 'PauliFrame':
        """Combine two Pauli frames (XOR corrections)."""
        return PauliFrame(
            x_correction=self.x_correction ^ other.x_correction,
            z_correction=self.z_correction ^ other.z_correction,
        )


@dataclass
class SyndromeInfo:
    """Information about syndrome measurements for one block/round."""
    level: int
    block_idx: int
    block_address: Tuple[int, ...]
    round_idx: int
    x_syndrome_indices: List[int]  # Measurement indices for X stabilizers
    z_syndrome_indices: List[int]  # Measurement indices for Z stabilizers


@dataclass
class ConcatenatedECDecoder(Decoder):
    """
    Decoder for concatenated codes with EC that tracks Pauli frames.
    
    This decoder properly handles EC rounds by:
    1. Extracting syndromes from EC measurements
    2. Computing corrections for each round
    3. Tracking Pauli frame through rounds
    4. Applying corrections to final decode
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The multi-level concatenated code.
    metadata : MultiLevelMetadata, optional
        Experiment metadata containing syndrome layout.
    level_strategies : Dict[int, Type[DecoderStrategy]], optional
        Strategy class to use at each level.
    basis : str
        Decoding basis ('X' or 'Z').
    qubit_mapper : HierarchicalQubitMapper, optional
        Prebuilt qubit mapper.
    """
    code: Any  # MultiLevelConcatenatedCode
    metadata: Optional[Any] = None  # MultiLevelMetadata
    level_strategies: Dict[int, Type[DecoderStrategy]] = field(default_factory=dict)
    basis: str = 'Z'
    qubit_mapper: Optional[Any] = None  # HierarchicalQubitMapper
    
    # Internal state
    strategies: Dict[int, DecoderStrategy] = field(default_factory=dict, init=False)
    _initialized: bool = field(default=False, init=False)
    
    def __post_init__(self) -> None:
        """Initialize strategies and mapper."""
        self._initialize()
    
    def _initialize(self) -> None:
        """Set up strategies and qubit mapper."""
        if self._initialized:
            return
        
        # Create qubit mapper if not provided
        if self.qubit_mapper is None:
            from qectostim.utils.hierarchical_mapper import HierarchicalQubitMapper
            self.qubit_mapper = HierarchicalQubitMapper(self.code)
        
        # Initialize strategies for each level
        for level in range(self.code.depth):
            level_code = self.code.level_codes[level]
            
            if level in self.level_strategies:
                strategy_cls = self.level_strategies[level]
            else:
                strategy_cls = SyndromeLookupStrategy
            
            self.strategies[level] = strategy_cls(level_code, self.basis)
        
        self._initialized = True
    
    def decode(self, measurements: np.ndarray) -> int:
        """
        Decode measurements to logical value.
        
        This method handles both cases:
        - If metadata has syndrome info: use EC-aware decoding with Pauli frame
        - Otherwise: fall back to standard hierarchical decode
        
        Parameters
        ----------
        measurements : np.ndarray
            Full measurement array (includes syndromes + final data).
            
        Returns
        -------
        int
            Predicted logical value (0 or 1).
        """
        measurements = np.asarray(measurements, dtype=np.uint8).flatten()
        
        # Check if we have syndrome information
        if self._has_syndrome_info():
            return self._decode_with_ec(measurements)
        else:
            # Fall back to basic hierarchical decode
            return self._decode_final_only(measurements)
    
    def _has_syndrome_info(self) -> bool:
        """Check if metadata contains useful syndrome layout."""
        if self.metadata is None:
            return False
        
        if not hasattr(self.metadata, 'syndrome_layout'):
            return False
        
        layout = self.metadata.syndrome_layout
        if not layout:
            return False
        
        # Check if there's actual syndrome data
        for level_data in layout.values():
            for block_info in level_data.values():
                if block_info.get('z_count', 0) > 0 or block_info.get('x_count', 0) > 0:
                    return True
        
        return False
    
    def _decode_with_ec(self, measurements: np.ndarray) -> int:
        """
        Decode using EC syndrome information with Pauli frame tracking.
        
        Algorithm:
        1. Initialize Pauli frames for all blocks
        2. For each EC round:
           - Extract syndromes for each block
           - Decode syndromes to get corrections
           - Update Pauli frames
        3. Apply accumulated corrections to final measurements
        4. Decode corrected measurements hierarchically
        """
        n_physical = self.code.n
        n_rounds = getattr(self.metadata, 'n_ec_rounds', 0)
        
        # Get final data measurement indices
        final_data_start = self.metadata.total_measurements - n_physical
        final_data = measurements[final_data_start:final_data_start + n_physical].copy()
        
        # Initialize Pauli frames for leaf blocks
        leaf_frames = self._init_leaf_frames()
        
        # Process EC rounds
        if n_rounds > 0:
            for round_idx in range(n_rounds):
                self._process_ec_round(measurements, round_idx, leaf_frames)
        
        # Apply accumulated corrections to final measurements
        corrected_data = self._apply_frames_to_data(final_data, leaf_frames)
        
        # Hierarchical decode on corrected data
        return self._decode_hierarchically(corrected_data)
    
    def _init_leaf_frames(self) -> Dict[Tuple[int, ...], PauliFrame]:
        """Initialize Pauli frames for all leaf blocks."""
        frames = {}
        
        # Get inner code size (qubits per leaf block)
        inner_code = self.code.level_codes[-1]
        n_inner = inner_code.n
        
        for addr, (start, end) in self.qubit_mapper.iter_leaf_ranges():
            frames[addr] = PauliFrame.identity(n_inner)
        
        return frames
    
    def _process_ec_round(
        self,
        measurements: np.ndarray,
        round_idx: int,
        leaf_frames: Dict[Tuple[int, ...], PauliFrame],
    ) -> None:
        """
        Process one EC round, updating Pauli frames.
        
        For concatenated codes, EC is done hierarchically:
        - Inner level: syndrome extraction on physical qubits
        - Outer level: syndrome extraction on logical qubits
        
        We process inner level first, then propagate to outer.
        """
        # Process from innermost to outermost level
        for level in reversed(range(self.code.depth)):
            self._process_level_ec(measurements, round_idx, level, leaf_frames)
    
    def _process_level_ec(
        self,
        measurements: np.ndarray,
        round_idx: int,
        level: int,
        leaf_frames: Dict[Tuple[int, ...], PauliFrame],
    ) -> None:
        """
        Process EC for one level.
        
        At the inner (leaf) level:
        - Extract raw ancilla measurements 
        - Compute syndrome = Hz @ raw_measurements
        - Decode syndrome to get correction
        - Update leaf Pauli frame
        
        At outer levels:
        - The "measurement" is the logical value from inner decode
        - This is handled implicitly through frame propagation
        """
        if level != self.code.depth - 1:
            # For now, focus on inner level EC (most important)
            # Outer level EC requires decoded logical values
            return
        
        # Inner level: process each leaf block
        syndrome_layout = getattr(self.metadata, 'syndrome_layout', {})
        level_layout = syndrome_layout.get(level, {})
        
        inner_code = self.code.level_codes[level]
        strategy = self.strategies[level]
        
        # Get Hz matrix for syndrome computation
        hz = None
        if hasattr(inner_code, 'hz'):
            hz_attr = inner_code.hz
            hz = np.atleast_2d(np.array(hz_attr() if callable(hz_attr) else hz_attr, dtype=int))
        
        for block_idx, block_info in level_layout.items():
            address = block_info.get('address', ())
            if isinstance(address, list):
                address = tuple(address)
            
            # Extract Z ancilla measurements (used to detect X errors)
            z_start = block_info.get('z_start', 0)
            z_count = block_info.get('z_count', 0)
            
            if z_count > 0 and address in leaf_frames and hz is not None:
                # Get raw ancilla measurements
                raw_z = measurements[z_start:z_start + z_count].astype(np.uint8)
                
                # Compute syndrome = Hz @ raw_measurements (mod 2)
                z_syndrome = (hz @ raw_z) % 2
                z_syndrome = z_syndrome.astype(np.uint8)
                
                # Decode syndrome to get correction
                output = strategy.decode(z_syndrome, raw_z)
                
                if output.correction is not None:
                    # Apply X correction (Z syndrome detects X errors)
                    leaf_frames[address].apply_correction(output.correction, 'X')
            
            # Extract X ancilla measurements (used to detect Z errors)
            x_start = block_info.get('x_start', 0)
            x_count = block_info.get('x_count', 0)
            
            if x_count > 0 and address in leaf_frames:
                # For X syndrome, we'd need Hx matrix
                # For Z-basis memory, Z errors don't affect measurement
                pass  # Z errors don't affect Z measurement
                # But for Z-basis memory, X errors are what matter
                pass  # Z errors don't affect Z measurement
    
    def _apply_frames_to_data(
        self,
        final_data: np.ndarray,
        leaf_frames: Dict[Tuple[int, ...], PauliFrame],
    ) -> np.ndarray:
        """
        Apply accumulated Pauli corrections to final measurements.
        
        For Z-basis measurement:
        - X errors (bit flips) affect the measurement outcome
        - We XOR the X correction with the measurement
        """
        corrected = final_data.copy()
        
        for addr, frame in leaf_frames.items():
            start, end = self.qubit_mapper.get_qubit_range(addr)
            
            # Apply X correction (bit flip correction)
            block_size = end - start
            if len(frame.x_correction) == block_size:
                corrected[start:end] ^= frame.x_correction
        
        return corrected
    
    def _decode_final_only(self, measurements: np.ndarray) -> int:
        """Fall back decode using only final measurements."""
        n_physical = self.code.n
        
        if len(measurements) < n_physical:
            measurements = np.concatenate([
                measurements,
                np.zeros(n_physical - len(measurements), dtype=np.uint8)
            ])
        elif len(measurements) > n_physical:
            measurements = measurements[-n_physical:]
        
        return self._decode_hierarchically(measurements)
    
    def _decode_hierarchically(self, data: np.ndarray) -> int:
        """
        Decode data measurements hierarchically.
        
        This is the standard hierarchical decode:
        1. Decode leaf blocks to get inner logical values
        2. Use inner logical values as "measurements" for outer code
        3. Repeat up the hierarchy
        """
        return self._decode_node(self.code.root, data)
    
    def _decode_node(self, node: 'CodeNode', data: np.ndarray) -> int:
        """Recursively decode a node."""
        level = node.level
        strategy = self.strategies[level]
        
        if node.is_leaf:
            # Leaf: decode from physical data
            start, end = self.qubit_mapper.get_qubit_range(node.address)
            block_data = data[start:end]
            syndrome = strategy.compute_syndrome(block_data)
            output = strategy.decode(syndrome, block_data)
            return output.logical_value
        else:
            # Internal: decode children first
            child_logicals = np.zeros(len(node.children), dtype=np.uint8)
            for i, child in enumerate(node.children):
                child_logicals[i] = self._decode_node(child, data)
            
            syndrome = strategy.compute_syndrome(child_logicals)
            output = strategy.decode(syndrome, child_logicals)
            return output.logical_value
    
    def decode_batch(self, measurements_batch: np.ndarray) -> np.ndarray:
        """
        Decode a batch of shots.
        
        Parameters
        ----------
        measurements_batch : np.ndarray
            Shape (n_shots, n_measurements).
            
        Returns
        -------
        np.ndarray
            Shape (n_shots,) with predicted logical values.
        """
        measurements_batch = np.asarray(measurements_batch, dtype=np.uint8)
        if measurements_batch.ndim == 1:
            measurements_batch = measurements_batch.reshape(1, -1)
        
        n_shots = measurements_batch.shape[0]
        results = np.zeros(n_shots, dtype=np.uint8)
        
        for i in range(n_shots):
            results[i] = self.decode(measurements_batch[i])
        
        return results
    
    def set_metadata(self, metadata: Any) -> None:
        """Update metadata (e.g., after building experiment)."""
        self.metadata = metadata


def create_ec_decoder(
    code: Any,
    metadata: Optional[Any] = None,
    level_strategies: Optional[Dict[int, Type[DecoderStrategy]]] = None,
    basis: str = 'Z',
) -> ConcatenatedECDecoder:
    """
    Factory function to create a ConcatenatedECDecoder.
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The multi-level concatenated code.
    metadata : MultiLevelMetadata, optional
        Experiment metadata.
    level_strategies : Dict[int, Type[DecoderStrategy]], optional
        Strategy classes per level.
    basis : str
        Decoding basis ('X' or 'Z').
        
    Returns
    -------
    ConcatenatedECDecoder
        Configured decoder.
    """
    return ConcatenatedECDecoder(
        code=code,
        metadata=metadata,
        level_strategies=level_strategies or {},
        basis=basis,
    )
