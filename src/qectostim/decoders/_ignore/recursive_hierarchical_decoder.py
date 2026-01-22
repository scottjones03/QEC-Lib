# src/qectostim/decoders/recursive_hierarchical_decoder.py
"""
Recursive Hierarchical Decoder for Multi-Level Concatenated Codes.

This decoder implements a bottom-up hierarchical decoding strategy:
1. Decode innermost (leaf) blocks from physical measurements → logical bits
2. Propagate logical values upward through the tree
3. At each level, apply the configured strategy to decode that level's code

The decoder supports configurable strategies per level, allowing different
decoding algorithms to be used at different levels of the concatenation.

It also supports using EC syndrome measurements from gadgets for improved
decoding when syndrome information is available.

Example
-------
>>> from qectostim.codes.composite import MultiLevelConcatenatedCode, ConcatenatedCodeBuilder
>>> from qectostim.codes.small import SteaneCode713, ShorCode91
>>> from qectostim.decoders import RecursiveHierarchicalDecoder
>>> from qectostim.decoders.strategies import SyndromeLookupStrategy
>>> 
>>> code = ConcatenatedCodeBuilder().add_level(SteaneCode713()).add_level(ShorCode91()).build()
>>> decoder = RecursiveHierarchicalDecoder(code)
>>> 
>>> # Decode measurements
>>> measurements = np.zeros(63, dtype=np.uint8)  # No errors
>>> result = decoder.decode(measurements)
>>> print(result)  # Should be 0
0

>>> # With EC syndrome information
>>> decoder_with_syndromes = RecursiveHierarchicalDecoder(code)
>>> result = decoder_with_syndromes.decode_with_syndromes(
...     measurements, ec_syndromes=syndrome_data
... )
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
    from qectostim.experiments.gadgets.base import MeasurementMap


@dataclass
class LevelDecodeResult:
    """Result from decoding a single level/block."""
    logical_value: int
    confidence: float = 1.0
    correction: Optional[np.ndarray] = None
    syndrome: Optional[np.ndarray] = None
    child_results: Dict[int, 'LevelDecodeResult'] = field(default_factory=dict)


@dataclass
class RecursiveHierarchicalDecoder(Decoder):
    """
    Recursive hierarchical decoder for multi-level concatenated codes.
    
    Decodes bottom-up through the concatenation tree:
    1. At leaf level: Extract physical measurements → decode to logical value
    2. At each parent level: Collect child logical values → decode outer code
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The multi-level concatenated code to decode.
    level_strategies : Dict[int, Type[DecoderStrategy]], optional
        Strategy class to use at each level. If not specified for a level,
        defaults to SyndromeLookupStrategy.
    basis : str
        Decoding basis ('X' or 'Z'). Default 'Z'.
    qubit_mapper : HierarchicalQubitMapper, optional
        Prebuilt qubit mapper. If None, will be created.
        
    Attributes
    ----------
    code : MultiLevelConcatenatedCode
        Reference to the code.
    strategies : Dict[int, DecoderStrategy]
        Instantiated strategy objects per level.
    qubit_mapper : HierarchicalQubitMapper
        Mapper for address → qubit range conversion.
    """
    code: Any  # MultiLevelConcatenatedCode
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
                # Default to syndrome lookup
                strategy_cls = SyndromeLookupStrategy
            
            self.strategies[level] = strategy_cls(level_code, self.basis)
        
        self._initialized = True
    
    def decode(self, measurements: np.ndarray) -> int:
        """
        Decode measurements to logical value.
        
        Parameters
        ----------
        measurements : np.ndarray
            Physical qubit measurements (length = code.n).
            
        Returns
        -------
        int
            Predicted logical value (0 or 1).
        """
        measurements = np.asarray(measurements, dtype=np.uint8).flatten()
        
        if len(measurements) < self.code.n:
            # Pad if needed
            measurements = np.concatenate([
                measurements,
                np.zeros(self.code.n - len(measurements), dtype=np.uint8)
            ])
        elif len(measurements) > self.code.n:
            # Take last n measurements (final data measurement)
            measurements = measurements[-self.code.n:]
        
        # Decode recursively from root
        result = self._decode_node(self.code.root, measurements)
        return result.logical_value
    
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
    
    def _decode_node(
        self,
        node: 'CodeNode',
        measurements: np.ndarray
    ) -> LevelDecodeResult:
        """
        Recursively decode a node in the code tree.
        
        For leaf nodes: Extract measurements and decode directly.
        For non-leaf nodes: Decode children first, then decode this level.
        
        Parameters
        ----------
        node : CodeNode
            Current node to decode.
        measurements : np.ndarray
            Full measurement array (all physical qubits).
            
        Returns
        -------
        LevelDecodeResult
            Decoding result for this node.
        """
        level = node.level
        address = node.address
        strategy = self.strategies[level]
        
        if node.is_leaf:
            # Leaf node: decode from physical measurements
            return self._decode_leaf(node, measurements, strategy)
        else:
            # Non-leaf: decode children first, then this level
            return self._decode_internal(node, measurements, strategy)
    
    def _decode_leaf(
        self,
        node: 'CodeNode',
        measurements: np.ndarray,
        strategy: DecoderStrategy
    ) -> LevelDecodeResult:
        """
        Decode a leaf node from physical measurements.
        
        Parameters
        ----------
        node : CodeNode
            Leaf node.
        measurements : np.ndarray
            Full measurement array.
        strategy : DecoderStrategy
            Strategy to use for this level.
            
        Returns
        -------
        LevelDecodeResult
            Decoded logical value and metadata.
        """
        # Get qubit range for this leaf
        start, end = self.qubit_mapper.get_qubit_range(node.address)
        block_meas = measurements[start:end]
        
        # Compute syndrome
        syndrome = strategy.compute_syndrome(block_meas)
        
        # Decode
        output = strategy.decode(syndrome, block_meas)
        
        return LevelDecodeResult(
            logical_value=output.logical_value,
            confidence=output.confidence,
            correction=output.correction,
            syndrome=syndrome,
        )
    
    def _decode_internal(
        self,
        node: 'CodeNode',
        measurements: np.ndarray,
        strategy: DecoderStrategy
    ) -> LevelDecodeResult:
        """
        Decode an internal (non-leaf) node.
        
        1. Recursively decode all children → get logical values
        2. Treat child logical values as "measurements" for this level
        3. Apply strategy to decode this level's code
        
        Parameters
        ----------
        node : CodeNode
            Internal node.
        measurements : np.ndarray
            Full measurement array.
        strategy : DecoderStrategy
            Strategy for this level.
            
        Returns
        -------
        LevelDecodeResult
            Decoded logical value and child results.
        """
        # Decode all children
        child_results = {}
        child_logicals = np.zeros(len(node.children), dtype=np.uint8)
        
        for i, child in enumerate(node.children):
            child_result = self._decode_node(child, measurements)
            child_results[i] = child_result
            child_logicals[i] = child_result.logical_value
        
        # The child logical values become the "measurements" for this level
        # Compute syndrome from child logicals
        syndrome = strategy.compute_syndrome(child_logicals)
        
        # Decode this level
        output = strategy.decode(syndrome, child_logicals)
        
        return LevelDecodeResult(
            logical_value=output.logical_value,
            confidence=output.confidence,
            correction=output.correction,
            syndrome=syndrome,
            child_results=child_results,
        )
    
    def decode_with_trace(
        self,
        measurements: np.ndarray
    ) -> Tuple[int, LevelDecodeResult]:
        """
        Decode with full trace information.
        
        Returns both the final logical value and the complete decode tree
        with results at each level.
        
        Parameters
        ----------
        measurements : np.ndarray
            Physical qubit measurements.
            
        Returns
        -------
        Tuple[int, LevelDecodeResult]
            (logical_value, full_trace)
        """
        measurements = np.asarray(measurements, dtype=np.uint8).flatten()
        
        if len(measurements) < self.code.n:
            measurements = np.concatenate([
                measurements,
                np.zeros(self.code.n - len(measurements), dtype=np.uint8)
            ])
        elif len(measurements) > self.code.n:
            measurements = measurements[-self.code.n:]
        
        result = self._decode_node(self.code.root, measurements)
        return result.logical_value, result
    
    def decode_with_syndromes(
        self,
        measurements: np.ndarray,
        ec_syndromes: Optional[ECSyndromeData] = None,
        metadata: Optional[Any] = None,
    ) -> int:
        """
        Decode using EC syndrome information when available.
        
        This method uses syndrome measurements from EC gadgets (if available)
        instead of computing syndromes from final measurements. This is more
        accurate when EC syndromes are measured mid-circuit.
        
        Parameters
        ----------
        measurements : np.ndarray
            Physical qubit measurements.
        ec_syndromes : ECSyndromeData, optional
            Pre-extracted EC syndrome data.
        metadata : Any, optional
            Experiment metadata (used if ec_syndromes not provided).
            
        Returns
        -------
        int
            Predicted logical value (0 or 1).
        """
        measurements = np.asarray(measurements, dtype=np.uint8).flatten()
        
        # Extract syndrome data if not provided
        if ec_syndromes is None and metadata is not None:
            ec_syndromes = ECSyndromeData.from_measurements(
                measurements, metadata, self.code
            )
        
        # Use final measurements only
        if len(measurements) > self.code.n:
            final_meas = measurements[-self.code.n:]
        else:
            final_meas = measurements
            if len(final_meas) < self.code.n:
                final_meas = np.concatenate([
                    final_meas,
                    np.zeros(self.code.n - len(final_meas), dtype=np.uint8)
                ])
        
        # Decode with syndrome awareness
        result = self._decode_node_with_syndromes(
            self.code.root, final_meas, ec_syndromes
        )
        return result.logical_value
    
    def _decode_node_with_syndromes(
        self,
        node: 'CodeNode',
        measurements: np.ndarray,
        ec_syndromes: Optional[ECSyndromeData],
    ) -> LevelDecodeResult:
        """
        Recursively decode a node, using EC syndromes when available.
        
        Parameters
        ----------
        node : CodeNode
            Current node to decode.
        measurements : np.ndarray
            Full measurement array.
        ec_syndromes : ECSyndromeData, optional
            EC syndrome data for this decode.
            
        Returns
        -------
        LevelDecodeResult
            Decoding result.
        """
        level = node.level
        strategy = self.strategies[level]
        
        # Try to get pre-measured syndrome
        pre_syndrome = None
        if ec_syndromes is not None:
            pre_syndrome = ec_syndromes.get_syndrome(level, node.address, self.basis)
        
        if node.is_leaf:
            return self._decode_leaf_with_syndrome(
                node, measurements, strategy, pre_syndrome
            )
        else:
            return self._decode_internal_with_syndromes(
                node, measurements, strategy, ec_syndromes, pre_syndrome
            )
    
    def _decode_leaf_with_syndrome(
        self,
        node: 'CodeNode',
        measurements: np.ndarray,
        strategy: DecoderStrategy,
        pre_syndrome: Optional[np.ndarray],
    ) -> LevelDecodeResult:
        """
        Decode leaf with optional pre-computed syndrome.
        
        Parameters
        ----------
        node : CodeNode
            Leaf node.
        measurements : np.ndarray
            Full measurement array.
        strategy : DecoderStrategy
            Strategy for this level.
        pre_syndrome : np.ndarray, optional
            Pre-measured syndrome from EC.
            
        Returns
        -------
        LevelDecodeResult
            Decoded result.
        """
        start, end = self.qubit_mapper.get_qubit_range(node.address)
        block_meas = measurements[start:end]
        
        # Use pre-syndrome if available, otherwise compute
        if pre_syndrome is not None:
            syndrome = pre_syndrome
        else:
            syndrome = strategy.compute_syndrome(block_meas)
        
        # Decode
        output = strategy.decode(syndrome, block_meas)
        
        return LevelDecodeResult(
            logical_value=output.logical_value,
            confidence=output.confidence,
            correction=output.correction,
            syndrome=syndrome,
        )
    
    def _decode_internal_with_syndromes(
        self,
        node: 'CodeNode',
        measurements: np.ndarray,
        strategy: DecoderStrategy,
        ec_syndromes: Optional[ECSyndromeData],
        pre_syndrome: Optional[np.ndarray],
    ) -> LevelDecodeResult:
        """
        Decode internal node with syndrome awareness.
        
        Parameters
        ----------
        node : CodeNode
            Internal node.
        measurements : np.ndarray
            Full measurement array.
        strategy : DecoderStrategy
            Strategy for this level.
        ec_syndromes : ECSyndromeData, optional
            EC syndrome data.
        pre_syndrome : np.ndarray, optional
            Pre-measured syndrome for this node.
            
        Returns
        -------
        LevelDecodeResult
            Decoded result.
        """
        # Decode all children
        child_results = {}
        child_logicals = np.zeros(len(node.children), dtype=np.uint8)
        
        for i, child in enumerate(node.children):
            child_result = self._decode_node_with_syndromes(
                child, measurements, ec_syndromes
            )
            child_results[i] = child_result
            child_logicals[i] = child_result.logical_value
        
        # Use pre-syndrome if available
        if pre_syndrome is not None:
            syndrome = pre_syndrome
        else:
            syndrome = strategy.compute_syndrome(child_logicals)
        
        # Decode this level
        output = strategy.decode(syndrome, child_logicals)
        
        return LevelDecodeResult(
            logical_value=output.logical_value,
            confidence=output.confidence,
            correction=output.correction,
            syndrome=syndrome,
            child_results=child_results,
        )


def create_recursive_decoder(
    code: Any,
    level_strategies: Optional[Dict[int, Type[DecoderStrategy]]] = None,
    basis: str = 'Z',
) -> RecursiveHierarchicalDecoder:
    """
    Factory function to create a RecursiveHierarchicalDecoder.
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The multi-level concatenated code.
    level_strategies : Dict[int, Type[DecoderStrategy]], optional
        Strategy classes per level.
    basis : str
        Decoding basis ('X' or 'Z').
        
    Returns
    -------
    RecursiveHierarchicalDecoder
        Configured decoder.
    """
    return RecursiveHierarchicalDecoder(
        code=code,
        level_strategies=level_strategies or {},
        basis=basis,
    )


@dataclass
class ECSyndromeData:
    """
    Container for EC syndrome data extracted from experiment circuits.
    
    This class holds syndrome measurement results organized by level and block,
    enabling the decoder to use EC syndromes rather than just final measurements.
    
    Attributes
    ----------
    syndromes_by_level : Dict[int, Dict[tuple, np.ndarray]]
        Mapping: level → {block_address → syndrome_array}
    measurement_map : Optional[MeasurementMap]
        Reference to measurement map from EC gadget.
    n_rounds : int
        Number of EC rounds.
    """
    syndromes_by_level: Dict[int, Dict[tuple, np.ndarray]] = field(default_factory=dict)
    measurement_map: Optional[Any] = None  # MeasurementMap
    n_rounds: int = 1
    
    @classmethod
    def from_measurements(
        cls,
        measurements: np.ndarray,
        metadata: Any,
        code: Any,
    ) -> 'ECSyndromeData':
        """
        Extract EC syndrome data from raw measurements and metadata.
        
        Supports two metadata formats:
        1. Old format: metadata.syndrome_layout with nested dicts
        2. New format: metadata.syndrome_layout from HierarchicalSyndromeLayout
        
        Parameters
        ----------
        measurements : np.ndarray
            Full measurement array from sampling.
        metadata : MultiLevelMetadata or similar
            Metadata containing syndrome layout information.
        code : MultiLevelConcatenatedCode
            The code structure for address mapping.
            
        Returns
        -------
        ECSyndromeData
            Extracted syndrome information.
        """
        syndromes_by_level: Dict[int, Dict[tuple, np.ndarray]] = {}
        
        # Try to extract from metadata.syndrome_layout (new format from HierarchicalSyndromeLayout)
        if hasattr(metadata, 'syndrome_layout') and metadata.syndrome_layout:
            layout = metadata.syndrome_layout
            
            for level, level_data in layout.items():
                level_int = int(level) if not isinstance(level, int) else level
                syndromes_by_level[level_int] = {}
                
                for block_idx, info in level_data.items():
                    # New format has z_start, z_count, x_start, x_count, address
                    z_start = info.get('z_start', 0)
                    z_count = info.get('z_count', 0)
                    address = info.get('address', ())
                    
                    if z_count > 0 and z_start + z_count <= len(measurements):
                        syndrome = measurements[z_start:z_start + z_count]
                        # Store by address tuple
                        addr_tuple = tuple(address) if not isinstance(address, tuple) else address
                        syndromes_by_level[level_int][addr_tuple] = np.asarray(syndrome, dtype=np.uint8)
        
        # Fallback: try old format with 'start' and 'count' keys
        elif hasattr(metadata, 'syndrome_layout'):
            layout = getattr(metadata, 'syndrome_layout', {})
            
            for level, level_data in layout.items():
                syndromes_by_level[level] = {}
                
                for block_addr, info in level_data.items():
                    start = info.get('start', 0)
                    count = info.get('count', 0)
                    
                    if count > 0 and start + count <= len(measurements):
                        syndrome = measurements[start:start + count]
                        syndromes_by_level[level][block_addr] = np.asarray(syndrome, dtype=np.uint8)
        
        return cls(
            syndromes_by_level=syndromes_by_level,
            n_rounds=getattr(metadata, 'n_ec_rounds', 1),
        )
    
    def get_syndrome(
        self,
        level: int,
        address: tuple,
        basis: str = 'Z'
    ) -> Optional[np.ndarray]:
        """
        Get syndrome for a specific block.
        
        Parameters
        ----------
        level : int
            Level in the hierarchy.
        address : tuple
            Block address (e.g., (0,), (1, 2)).
        basis : str
            'Z' or 'X' syndrome.
            
        Returns
        -------
        np.ndarray or None
            Syndrome array if available, None otherwise.
        """
        if level not in self.syndromes_by_level:
            return None
        
        level_syns = self.syndromes_by_level[level]
        
        if address in level_syns:
            return level_syns[address]
        
        # Try tuple conversion
        if not isinstance(address, tuple):
            address = tuple(address) if hasattr(address, '__iter__') else (address,)
            if address in level_syns:
                return level_syns[address]
        
        return None
