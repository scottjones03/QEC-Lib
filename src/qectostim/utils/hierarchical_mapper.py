# src/qectostim/utils/hierarchical_mapper.py
"""
Hierarchical Qubit and Measurement Mappers for Multi-Level Concatenation.

This module provides address-based mapping between hierarchical block structures
and flat physical indices for both qubits and measurements.

Key Classes
-----------
HierarchicalQubitMapper
    Maps hierarchical addresses to physical qubit ranges.
    
HierarchicalMeasurementMapper
    Records and retrieves measurements with hierarchical metadata.

MeasurementRecord
    Dataclass holding measurement metadata.

Example
-------
>>> from qectostim.codes.composite import MultiLevelConcatenatedCode, ConcatenatedCodeBuilder
>>> from qectostim.codes.small import SteaneCode713, ShorCode91
>>> 
>>> code = ConcatenatedCodeBuilder().add_level(SteaneCode713()).add_level(ShorCode91()).build()
>>> mapper = HierarchicalQubitMapper(code)
>>> 
>>> # Get physical qubit range for a specific block
>>> start, end = mapper.get_qubit_range((0,))  # First outer block
>>> print(f"Block (0,) spans qubits {start}:{end}")
Block (0,) spans qubits 0:9
>>> 
>>> # Get all leaf ranges
>>> for addr, (start, end) in mapper.iter_leaf_ranges():
...     print(f"Leaf {addr}: qubits {start}:{end}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from qectostim.codes.composite.multilevel_concatenated import (
        CodeNode,
        MultiLevelConcatenatedCode,
    )


class MeasurementType(Enum):
    """Types of measurements in a concatenated code circuit."""
    DATA = auto()           # Final data qubit measurement
    X_SYNDROME = auto()     # X stabilizer syndrome measurement
    Z_SYNDROME = auto()     # Z stabilizer syndrome measurement
    LOGICAL_X = auto()      # Logical X measurement
    LOGICAL_Z = auto()      # Logical Z measurement
    FLAG = auto()           # Flag qubit measurement
    ANCILLA = auto()        # Generic ancilla measurement


@dataclass
class MeasurementRecord:
    """
    Record of a single measurement with hierarchical metadata.
    
    Attributes
    ----------
    index : int
        Global measurement index in the circuit.
    round : int
        EC round number (0-indexed).
    level : int
        Hierarchical level (0 = outermost, depth-1 = innermost).
    address : Tuple[int, ...]
        Hierarchical block address.
    mtype : MeasurementType
        Type of measurement.
    qubit_within_block : int
        Qubit index relative to the block (0 to block_size-1).
    physical_qubit : Optional[int]
        Global physical qubit index if applicable.
    check_index : Optional[int]
        Index of the stabilizer check (for syndrome measurements).
    """
    index: int
    round: int
    level: int
    address: Tuple[int, ...]
    mtype: MeasurementType
    qubit_within_block: int = 0
    physical_qubit: Optional[int] = None
    check_index: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'index': self.index,
            'round': self.round,
            'level': self.level,
            'address': self.address,
            'type': self.mtype.name,
            'qubit_within_block': self.qubit_within_block,
            'physical_qubit': self.physical_qubit,
            'check_index': self.check_index,
        }


class HierarchicalQubitMapper:
    """
    Maps hierarchical addresses to physical qubit ranges.
    
    For a multi-level concatenated code, provides bidirectional mapping:
    - Address → Physical qubit range
    - Physical qubit → (Address, local_index)
    
    The mapping is computed recursively based on the code tree structure,
    ensuring that all leaf blocks cover [0, n_total) exactly once with
    no overlaps.
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The multi-level concatenated code to build mappings for.
        
    Attributes
    ----------
    code : MultiLevelConcatenatedCode
        Reference to the code structure.
    _address_to_range : Dict[Tuple[int, ...], Tuple[int, int]]
        Cached address → (start, end) mapping.
    _qubit_to_address : Dict[int, Tuple[Tuple[int, ...], int]]
        Cached qubit → (address, local_idx) mapping.
    
    Example
    -------
    >>> code = MultiLevelConcatenatedCode([SteaneCode(), ShorCode()])
    >>> mapper = HierarchicalQubitMapper(code)
    >>> 
    >>> # Get range for root (entire code)
    >>> mapper.get_qubit_range(())  # (0, 63)
    >>> 
    >>> # Get range for first outer block
    >>> mapper.get_qubit_range((0,))  # (0, 9)
    >>> 
    >>> # Find which block contains qubit 50
    >>> mapper.get_address_for_qubit(50)  # ((5,), 5)
    """
    
    def __init__(self, code: MultiLevelConcatenatedCode) -> None:
        """
        Initialize the qubit mapper.
        
        Parameters
        ----------
        code : MultiLevelConcatenatedCode
            The multi-level code to map.
        """
        self.code = code
        self._address_to_range: Dict[Tuple[int, ...], Tuple[int, int]] = {}
        self._qubit_to_address: Dict[int, Tuple[Tuple[int, ...], int]] = {}
        
        # Build the mappings
        self._build_mappings()
    
    def _build_mappings(self) -> None:
        """Build all address ↔ qubit mappings recursively."""
        self._build_node_mapping(self.code.root, 0)
    
    def _build_node_mapping(self, node: CodeNode, offset: int) -> int:
        """
        Recursively build mapping for a node and its children.
        
        Parameters
        ----------
        node : CodeNode
            Current node in the tree.
        offset : int
            Starting qubit index for this node.
            
        Returns
        -------
        int
            The next qubit index after this node's range.
        """
        if node.is_leaf:
            # Leaf node: maps to physical qubits
            end = offset + node.code.n
            self._address_to_range[node.address] = (offset, end)
            
            # Build reverse mapping: qubit → (address, local_idx)
            for local_idx in range(node.code.n):
                self._qubit_to_address[offset + local_idx] = (node.address, local_idx)
            
            return end
        else:
            # Internal node: aggregate children
            start = offset
            current = offset
            for child in node.children:
                current = self._build_node_mapping(child, current)
            
            self._address_to_range[node.address] = (start, current)
            return current
    
    def get_qubit_range(self, address: Tuple[int, ...]) -> Tuple[int, int]:
        """
        Get the physical qubit range for a block address.
        
        Parameters
        ----------
        address : Tuple[int, ...]
            Hierarchical address. Empty tuple for root (entire code).
            
        Returns
        -------
        Tuple[int, int]
            (start, end) indices where end is exclusive.
            
        Raises
        ------
        KeyError
            If address is not valid.
            
        Example
        -------
        >>> mapper.get_qubit_range(())      # (0, 63) for Steane⊗Shor
        >>> mapper.get_qubit_range((0,))    # (0, 9) first block
        >>> mapper.get_qubit_range((6,))    # (54, 63) last block
        """
        return self._address_to_range[address]
    
    def get_qubit_indices(self, address: Tuple[int, ...]) -> range:
        """
        Get the physical qubit indices for a block as a range object.
        
        Parameters
        ----------
        address : Tuple[int, ...]
            Hierarchical address.
            
        Returns
        -------
        range
            Range object for the qubit indices.
        """
        start, end = self.get_qubit_range(address)
        return range(start, end)
    
    def get_address_for_qubit(self, qubit: int) -> Tuple[Tuple[int, ...], int]:
        """
        Find the leaf block address containing a physical qubit.
        
        Parameters
        ----------
        qubit : int
            Physical qubit index.
            
        Returns
        -------
        Tuple[Tuple[int, ...], int]
            (leaf_address, local_index_within_block)
            
        Raises
        ------
        KeyError
            If qubit index is out of range.
        """
        return self._qubit_to_address[qubit]
    
    def iter_leaf_ranges(self) -> Iterator[Tuple[Tuple[int, ...], Tuple[int, int]]]:
        """
        Iterate over all leaf block ranges.
        
        Yields
        ------
        Tuple[Tuple[int, ...], Tuple[int, int]]
            (address, (start, end)) for each leaf block.
        """
        for node in self.code.iter_leaves():
            yield node.address, self._address_to_range[node.address]
    
    def iter_level_ranges(self, level: int) -> Iterator[Tuple[Tuple[int, ...], Tuple[int, int]]]:
        """
        Iterate over all block ranges at a given level.
        
        Parameters
        ----------
        level : int
            Level to iterate (0 = root level).
            
        Yields
        ------
        Tuple[Tuple[int, ...], Tuple[int, int]]
            (address, (start, end)) for each block at the level.
        """
        for node in self.code.iter_level(level):
            yield node.address, self._address_to_range[node.address]
    
    def get_block_size(self, address: Tuple[int, ...]) -> int:
        """
        Get the number of physical qubits in a block.
        
        Parameters
        ----------
        address : Tuple[int, ...]
            Hierarchical address.
            
        Returns
        -------
        int
            Number of physical qubits in the block.
        """
        start, end = self.get_qubit_range(address)
        return end - start
    
    def validate_coverage(self) -> bool:
        """
        Validate that leaf blocks cover all qubits exactly once.
        
        Returns
        -------
        bool
            True if coverage is valid.
            
        Raises
        ------
        AssertionError
            If coverage is invalid (overlap or gap).
        """
        total_qubits = self.code.n
        covered = np.zeros(total_qubits, dtype=bool)
        
        for node in self.code.iter_leaves():
            start, end = self._address_to_range[node.address]
            
            # Check for overlap
            if np.any(covered[start:end]):
                raise AssertionError(f"Overlap detected at address {node.address}")
            
            covered[start:end] = True
        
        # Check for gaps
        if not np.all(covered):
            uncovered = np.where(~covered)[0]
            raise AssertionError(f"Gap detected at qubits {uncovered}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export mappings as dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with address_to_range entries.
        """
        return {
            'n_total': self.code.n,
            'depth': self.code.depth,
            'n_leaves': self.code.count_leaves(),
            'ranges': {
                str(addr): {'start': start, 'end': end}
                for addr, (start, end) in self._address_to_range.items()
            },
        }


class HierarchicalMeasurementMapper:
    """
    Records and retrieves measurements with hierarchical metadata.
    
    This mapper tracks all measurements in a multi-level concatenated code
    circuit, recording their hierarchical context (level, address, type).
    
    It supports:
    - Recording measurements during circuit construction
    - Querying measurements by round, level, address, or type
    - Extracting syndrome vectors for specific blocks
    - Computing block-level syndromes from raw measurement data
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The multi-level concatenated code.
    qubit_mapper : Optional[HierarchicalQubitMapper]
        Qubit mapper (created automatically if not provided).
        
    Attributes
    ----------
    code : MultiLevelConcatenatedCode
        Reference to the code structure.
    qubit_mapper : HierarchicalQubitMapper
        Associated qubit mapper.
    _records : List[MeasurementRecord]
        All recorded measurements.
    _by_round : Dict[int, List[int]]
        Measurement indices grouped by round.
    _by_level : Dict[int, List[int]]
        Measurement indices grouped by level.
    _by_address : Dict[Tuple[int, ...], List[int]]
        Measurement indices grouped by address.
    _by_type : Dict[MeasurementType, List[int]]
        Measurement indices grouped by type.
        
    Example
    -------
    >>> mm = HierarchicalMeasurementMapper(code)
    >>> 
    >>> # Record measurements during circuit construction
    >>> mm.record(index=0, round=0, level=1, address=(0,), 
    ...           mtype=MeasurementType.Z_SYNDROME, check_index=0)
    >>> 
    >>> # Query measurements
    >>> round0_meas = mm.get_measurements_by_round(0)
    >>> syndrome_meas = mm.get_measurements_by_type(MeasurementType.Z_SYNDROME)
    """
    
    def __init__(
        self,
        code: MultiLevelConcatenatedCode,
        qubit_mapper: Optional[HierarchicalQubitMapper] = None,
    ) -> None:
        """
        Initialize the measurement mapper.
        
        Parameters
        ----------
        code : MultiLevelConcatenatedCode
            The multi-level code.
        qubit_mapper : Optional[HierarchicalQubitMapper]
            Qubit mapper. Created if not provided.
        """
        self.code = code
        self.qubit_mapper = qubit_mapper or HierarchicalQubitMapper(code)
        
        # Storage
        self._records: List[MeasurementRecord] = []
        self._next_index = 0
        
        # Indices for fast lookup
        self._by_round: Dict[int, List[int]] = {}
        self._by_level: Dict[int, List[int]] = {}
        self._by_address: Dict[Tuple[int, ...], List[int]] = {}
        self._by_type: Dict[MeasurementType, List[int]] = {}
    
    def record(
        self,
        round: int,
        level: int,
        address: Tuple[int, ...],
        mtype: MeasurementType,
        qubit_within_block: int = 0,
        physical_qubit: Optional[int] = None,
        check_index: Optional[int] = None,
        index: Optional[int] = None,
    ) -> int:
        """
        Record a measurement.
        
        Parameters
        ----------
        round : int
            EC round number.
        level : int
            Hierarchical level.
        address : Tuple[int, ...]
            Block address.
        mtype : MeasurementType
            Measurement type.
        qubit_within_block : int
            Local qubit index within block.
        physical_qubit : Optional[int]
            Global physical qubit index.
        check_index : Optional[int]
            Stabilizer check index for syndrome measurements.
        index : Optional[int]
            Explicit measurement index. If None, auto-incremented.
            
        Returns
        -------
        int
            The measurement index.
        """
        if index is None:
            index = self._next_index
            self._next_index += 1
        else:
            self._next_index = max(self._next_index, index + 1)
        
        record = MeasurementRecord(
            index=index,
            round=round,
            level=level,
            address=address,
            mtype=mtype,
            qubit_within_block=qubit_within_block,
            physical_qubit=physical_qubit,
            check_index=check_index,
        )
        
        # Store the record
        self._records.append(record)
        
        # Update indices
        if round not in self._by_round:
            self._by_round[round] = []
        self._by_round[round].append(len(self._records) - 1)
        
        if level not in self._by_level:
            self._by_level[level] = []
        self._by_level[level].append(len(self._records) - 1)
        
        if address not in self._by_address:
            self._by_address[address] = []
        self._by_address[address].append(len(self._records) - 1)
        
        if mtype not in self._by_type:
            self._by_type[mtype] = []
        self._by_type[mtype].append(len(self._records) - 1)
        
        return index
    
    def record_syndrome_round(
        self,
        round: int,
        level: int,
        address: Tuple[int, ...],
        n_x_checks: int,
        n_z_checks: int,
        base_index: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Record a complete syndrome measurement round for a block.
        
        Parameters
        ----------
        round : int
            EC round number.
        level : int
            Hierarchical level.
        address : Tuple[int, ...]
            Block address.
        n_x_checks : int
            Number of X stabilizer checks.
        n_z_checks : int
            Number of Z stabilizer checks.
        base_index : Optional[int]
            Starting measurement index.
            
        Returns
        -------
        Tuple[List[int], List[int]]
            (x_check_indices, z_check_indices) - measurement indices.
        """
        x_indices = []
        z_indices = []
        
        if base_index is not None:
            self._next_index = base_index
        
        # Record X syndrome measurements
        for i in range(n_x_checks):
            idx = self.record(
                round=round,
                level=level,
                address=address,
                mtype=MeasurementType.X_SYNDROME,
                check_index=i,
            )
            x_indices.append(idx)
        
        # Record Z syndrome measurements
        for i in range(n_z_checks):
            idx = self.record(
                round=round,
                level=level,
                address=address,
                mtype=MeasurementType.Z_SYNDROME,
                check_index=i,
            )
            z_indices.append(idx)
        
        return x_indices, z_indices
    
    def record_data_measurements(
        self,
        round: int,
        level: int,
        address: Tuple[int, ...],
        n_qubits: int,
        base_index: Optional[int] = None,
    ) -> List[int]:
        """
        Record data qubit measurements for a block.
        
        Parameters
        ----------
        round : int
            Measurement round.
        level : int
            Hierarchical level.
        address : Tuple[int, ...]
            Block address.
        n_qubits : int
            Number of data qubits.
        base_index : Optional[int]
            Starting measurement index.
            
        Returns
        -------
        List[int]
            Measurement indices.
        """
        if base_index is not None:
            self._next_index = base_index
        
        # Get physical qubit range for this block
        try:
            start_qubit, _ = self.qubit_mapper.get_qubit_range(address)
        except KeyError:
            start_qubit = 0
        
        indices = []
        for i in range(n_qubits):
            idx = self.record(
                round=round,
                level=level,
                address=address,
                mtype=MeasurementType.DATA,
                qubit_within_block=i,
                physical_qubit=start_qubit + i,
            )
            indices.append(idx)
        
        return indices
    
    def get_record(self, index: int) -> MeasurementRecord:
        """
        Get measurement record by index.
        
        Parameters
        ----------
        index : int
            Measurement index.
            
        Returns
        -------
        MeasurementRecord
            The measurement record.
        """
        for record in self._records:
            if record.index == index:
                return record
        raise KeyError(f"No measurement with index {index}")
    
    def get_measurements_by_round(self, round: int) -> List[MeasurementRecord]:
        """Get all measurements from a specific round."""
        indices = self._by_round.get(round, [])
        return [self._records[i] for i in indices]
    
    def get_measurements_by_level(self, level: int) -> List[MeasurementRecord]:
        """Get all measurements at a specific level."""
        indices = self._by_level.get(level, [])
        return [self._records[i] for i in indices]
    
    def get_measurements_by_address(
        self, 
        address: Tuple[int, ...]
    ) -> List[MeasurementRecord]:
        """Get all measurements for a specific block address."""
        indices = self._by_address.get(address, [])
        return [self._records[i] for i in indices]
    
    def get_measurements_by_type(
        self, 
        mtype: MeasurementType
    ) -> List[MeasurementRecord]:
        """Get all measurements of a specific type."""
        indices = self._by_type.get(mtype, [])
        return [self._records[i] for i in indices]
    
    def get_block_syndrome_indices(
        self,
        round: int,
        address: Tuple[int, ...],
    ) -> Tuple[List[int], List[int]]:
        """
        Get syndrome measurement indices for a specific block and round.
        
        Parameters
        ----------
        round : int
            EC round number.
        address : Tuple[int, ...]
            Block address.
            
        Returns
        -------
        Tuple[List[int], List[int]]
            (x_syndrome_indices, z_syndrome_indices)
        """
        x_indices = []
        z_indices = []
        
        for record in self._records:
            if record.round == round and record.address == address:
                if record.mtype == MeasurementType.X_SYNDROME:
                    x_indices.append(record.index)
                elif record.mtype == MeasurementType.Z_SYNDROME:
                    z_indices.append(record.index)
        
        # Sort by check_index for consistent ordering
        x_indices.sort(key=lambda i: self.get_record(i).check_index or 0)
        z_indices.sort(key=lambda i: self.get_record(i).check_index or 0)
        
        return x_indices, z_indices
    
    def extract_block_syndrome(
        self,
        measurements: np.ndarray,
        round: int,
        address: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract X and Z syndrome vectors for a block from raw measurements.
        
        Parameters
        ----------
        measurements : np.ndarray
            Raw measurement vector from Stim sampling.
        round : int
            EC round.
        address : Tuple[int, ...]
            Block address.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (x_syndrome, z_syndrome) as binary vectors.
        """
        x_indices, z_indices = self.get_block_syndrome_indices(round, address)
        
        x_syndrome = np.array([measurements[i] for i in x_indices], dtype=np.uint8)
        z_syndrome = np.array([measurements[i] for i in z_indices], dtype=np.uint8)
        
        return x_syndrome, z_syndrome
    
    def get_data_measurement_indices(
        self,
        round: int,
        address: Tuple[int, ...],
    ) -> List[int]:
        """
        Get data measurement indices for a block.
        
        Parameters
        ----------
        round : int
            Measurement round.
        address : Tuple[int, ...]
            Block address.
            
        Returns
        -------
        List[int]
            Data measurement indices in qubit order.
        """
        indices = []
        for record in self._records:
            if (record.round == round and 
                record.address == address and
                record.mtype == MeasurementType.DATA):
                indices.append((record.qubit_within_block, record.index))
        
        # Sort by qubit_within_block
        indices.sort(key=lambda x: x[0])
        return [idx for _, idx in indices]
    
    def get_total_measurements(self) -> int:
        """Get total number of recorded measurements."""
        return len(self._records)
    
    def get_rounds(self) -> List[int]:
        """Get list of all recorded rounds."""
        return sorted(self._by_round.keys())
    
    def clear(self) -> None:
        """Clear all recorded measurements."""
        self._records.clear()
        self._next_index = 0
        self._by_round.clear()
        self._by_level.clear()
        self._by_address.clear()
        self._by_type.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all records as dictionary."""
        return {
            'n_total': self.get_total_measurements(),
            'rounds': self.get_rounds(),
            'records': [r.to_dict() for r in self._records],
        }
    
    def __len__(self) -> int:
        return len(self._records)
    
    def __iter__(self) -> Iterator[MeasurementRecord]:
        return iter(self._records)


def create_mappers(
    code: MultiLevelConcatenatedCode,
) -> Tuple[HierarchicalQubitMapper, HierarchicalMeasurementMapper]:
    """
    Create both qubit and measurement mappers for a code.
    
    Convenience function that creates a shared qubit mapper.
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The multi-level code.
        
    Returns
    -------
    Tuple[HierarchicalQubitMapper, HierarchicalMeasurementMapper]
        (qubit_mapper, measurement_mapper)
    """
    qubit_mapper = HierarchicalQubitMapper(code)
    measurement_mapper = HierarchicalMeasurementMapper(code, qubit_mapper)
    return qubit_mapper, measurement_mapper
