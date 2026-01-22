"""
Detector-Based Primitives for Fault-Tolerant Gadgets.

This module provides the core data structures for detector-based circuit
construction, matching the architecture where:

1. Gadgets emit DETECTOR instructions directly (not raw measurements)
2. Detector indices are tracked globally during circuit assembly
3. Decoders work on detector sampler output (binary detector vectors)
4. Post-selection uses verification detector groups
5. Logical outcomes computed entirely in Python (no OBSERVABLE_INCLUDE)

Key Types:
----------
- DetectorGroup: A [start, end) range of detector indices
- DetectorMap: Return type from gadget.emit() - tracks all detector groups
- GadgetResult: Extended return with detector map + metadata
- DetectorCounter: Global counter for tracking detector positions

Architecture Pattern:
--------------------
Each gadget appends measurements and immediately adds DETECTOR instructions.
The detector index is tracked globally (like `detector_now` in reference code).

    >>> counter = DetectorCounter()
    >>> gadget.emit(circuit, data_qubits, counter=counter)
    >>> # counter.current now reflects all detectors emitted
    >>> # Returns DetectorMap with [start, end) ranges for each group

Decoder Pattern:
---------------
    >>> detector_samples = circuit.compile_detector_sampler().sample(shots)
    >>> for shot in detector_samples:
    ...     if not post_select(shot, dmap.verification_groups):
    ...         continue  # reject shot
    ...     result = decode_gadget(shot, dmap.syndrome_groups)
    ...     if result == -1:
    ...         # ambiguous - handle as 0.5 contribution or reject
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union


class DetectorType(Enum):
    """Type of detector group in the circuit."""
    VERIFICATION = auto()      # Ancilla prep verification (post-select on these)
    SYNDROME_Z = auto()        # Z syndrome extraction (detects X errors)
    SYNDROME_X = auto()        # X syndrome extraction (detects Z errors)
    FINAL_DATA = auto()        # Final data qubit measurements
    LOGICAL = auto()           # Decoded logical measurement


@dataclass
class DetectorGroup:
    """
    A contiguous range of detector indices [start, end).
    
    Attributes:
        start: First detector index (inclusive)
        end: Last detector index (exclusive)
        dtype: Type of detector group
        block_id: Which code block this belongs to (for concatenated codes)
        level: Concatenation level (0=outer, 1=inner, etc.)
        round_idx: EC round index (if applicable)
        metadata: Additional info (stabilizer indices, etc.)
    """
    start: int
    end: int
    dtype: DetectorType
    block_id: int = 0
    level: int = 0
    round_idx: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        """Number of detectors in this group."""
        return self.end - self.start
    
    def slice_from(self, detector_vector) -> Any:
        """Extract this group's detectors from a full detector vector."""
        return detector_vector[self.start:self.end]
    
    def __repr__(self) -> str:
        return f"DetectorGroup([{self.start}:{self.end}), {self.dtype.name}, block={self.block_id})"


@dataclass
class DetectorMap:
    """
    Complete detector layout from a gadget emission.
    
    This replaces MeasurementMap with detector-centric tracking.
    All detector groups use [start, end) convention.
    
    Usage:
        >>> dmap = gadget.emit(circuit, data_qubits, counter)
        >>> # Post-select on verification
        >>> if any(shot[g.start:g.end].sum() % 2 for g in dmap.verification):
        ...     continue  # reject
        >>> # Decode syndromes
        >>> z_syndrome = shot[dmap.syndrome_z[0].start:dmap.syndrome_z[0].end]
    
    Attributes:
        verification: Verification detector groups (for post-selection)
        syndrome_z: Z syndrome detector groups (per block)
        syndrome_x: X syndrome detector groups (per block)
        final_data: Final data measurement detectors (if applicable)
        all_groups: All detector groups in emission order
        total_detectors: Total number of detectors emitted
        offset: Starting detector index for this gadget
    """
    verification: List[DetectorGroup] = field(default_factory=list)
    syndrome_z: Dict[int, DetectorGroup] = field(default_factory=dict)  # block_id -> group
    syndrome_x: Dict[int, DetectorGroup] = field(default_factory=dict)  # block_id -> group
    final_data: Optional[DetectorGroup] = None
    
    # Hierarchical structure for concatenated codes
    # level -> block_id -> syndrome_type -> DetectorGroup
    hierarchical: Dict[int, Dict[int, Dict[str, DetectorGroup]]] = field(default_factory=dict)
    
    all_groups: List[DetectorGroup] = field(default_factory=list)
    total_detectors: int = 0
    offset: int = 0
    
    # For tracking raw measurement indices (needed for some operations)
    measurement_offset: int = 0
    total_measurements: int = 0
    
    def add_group(self, group: DetectorGroup) -> None:
        """Add a detector group and update tracking."""
        self.all_groups.append(group)
        
        if group.dtype == DetectorType.VERIFICATION:
            self.verification.append(group)
        elif group.dtype == DetectorType.SYNDROME_Z:
            self.syndrome_z[group.block_id] = group
        elif group.dtype == DetectorType.SYNDROME_X:
            self.syndrome_x[group.block_id] = group
        elif group.dtype == DetectorType.FINAL_DATA:
            self.final_data = group
            
        # Update hierarchical view
        if group.level not in self.hierarchical:
            self.hierarchical[group.level] = {}
        if group.block_id not in self.hierarchical[group.level]:
            self.hierarchical[group.level][group.block_id] = {}
        
        type_key = group.dtype.name.lower()
        self.hierarchical[group.level][group.block_id][type_key] = group
        
        # Update total
        self.total_detectors = max(self.total_detectors, group.end - self.offset)
    
    def get_verification_ranges(self) -> List[Tuple[int, int]]:
        """Get all verification detector ranges for post-selection."""
        return [(g.start, g.end) for g in self.verification]
    
    def get_syndrome_groups(self, syndrome_type: str = 'Z') -> Dict[int, DetectorGroup]:
        """Get syndrome detector groups by type."""
        if syndrome_type.upper() == 'Z':
            return self.syndrome_z
        elif syndrome_type.upper() == 'X':
            return self.syndrome_x
        else:
            raise ValueError(f"Unknown syndrome type: {syndrome_type}")


class DetectorCounter:
    """
    Global counter for detector indices during circuit assembly.
    
    Mimics the `detector_now` global from reference code but as explicit state.
    
    Usage:
        >>> counter = DetectorCounter()
        >>> gadget1.emit(circuit, qubits, counter=counter)  # emits detectors 0-5
        >>> gadget2.emit(circuit, qubits, counter=counter)  # emits detectors 6-10
        >>> print(counter.current)  # 11
    """
    
    def __init__(self, start: int = 0):
        self._current = start
        self._history: List[Tuple[str, int, int]] = []  # (gadget_name, start, end)
    
    @property
    def current(self) -> int:
        """Current detector index (next detector will have this index)."""
        return self._current
    
    def allocate(self, count: int, gadget_name: str = "unknown") -> Tuple[int, int]:
        """
        Allocate a range of detector indices.
        
        Returns:
            (start, end) tuple where start is inclusive, end is exclusive
        """
        start = self._current
        self._current += count
        end = self._current
        self._history.append((gadget_name, start, end))
        return start, end
    
    def advance(self, count: int = 1) -> int:
        """Advance counter by count, return the starting index."""
        start = self._current
        self._current += count
        return start
    
    def peek(self) -> int:
        """Get current index without advancing."""
        return self._current
    
    def reset(self, value: int = 0) -> None:
        """Reset counter to a specific value."""
        self._current = value
        self._history.clear()
    
    @property
    def history(self) -> List[Tuple[str, int, int]]:
        """Get allocation history."""
        return self._history.copy()


@dataclass
class GadgetResult:
    """
    Complete result from a gadget emission.
    
    Includes detector map plus additional metadata needed for decoding.
    """
    detector_map: DetectorMap
    
    # Code information for decoder
    code_name: str = ""
    n_data_qubits: int = 0
    n_syndrome_bits: int = 0
    
    # For correction propagation
    x_propagate: Optional[List[int]] = None  # Which logical qubits X correction affects
    z_propagate: Optional[List[int]] = None  # Which logical qubits Z correction affects
    
    # Pauli frame from teleportation EC
    pauli_frame_x: Dict[int, int] = field(default_factory=dict)  # block -> measurement idx
    pauli_frame_z: Dict[int, int] = field(default_factory=dict)
    
    # Output qubit mapping (if gadget moves qubits)
    output_qubits: Dict[int, List[int]] = field(default_factory=dict)


@dataclass 
class ECRoundResult:
    """
    Result from one EC round, containing all detector groups and corrections.
    
    Used to accumulate corrections across rounds in Python.
    """
    round_idx: int
    detector_map: DetectorMap
    
    # Corrections computed for this round (set during decoding)
    correction_x: Optional[List[int]] = None
    correction_z: Optional[List[int]] = None
    
    # Whether this round's decoder returned ambiguous (-1)
    is_ambiguous: bool = False
    
    # Verification status
    verification_passed: bool = True


@dataclass
class CircuitMetadata:
    """
    Complete metadata for a detector-based circuit.
    
    This is returned alongside the stim.Circuit and contains all
    information needed for sampling and decoding.
    """
    # Detector groups per EC round
    ec_rounds: List[ECRoundResult] = field(default_factory=list)
    
    # Final measurement detector groups
    final_measurement: Optional[DetectorMap] = None
    
    # Verification detector groups (for post-selection filtering)
    verification_groups: List[DetectorGroup] = field(default_factory=list)
    
    # Total detectors in circuit
    total_detectors: int = 0
    
    # Total measurements in circuit (for reference)
    total_measurements: int = 0
    
    # Code structure info
    code_name: str = ""
    n_levels: int = 1
    inner_code_name: str = ""
    outer_code_name: str = ""
    n_inner_blocks: int = 1
    n_data_qubits: int = 0
    
    # Logical operator support (qubit indices)
    z_logical_support: List[int] = field(default_factory=list)
    x_logical_support: List[int] = field(default_factory=list)
    
    # Prepared state (for computing logical error)
    initial_state: str = "0"
    basis: str = "Z"
    
    # Correction propagation arrays
    x_propagate: List[int] = field(default_factory=list)
    z_propagate: List[int] = field(default_factory=list)
    
    def add_ec_round(self, result: ECRoundResult) -> None:
        """Add an EC round result."""
        self.ec_rounds.append(result)
        # Collect verification groups
        self.verification_groups.extend(result.detector_map.verification)
        
    def get_all_verification_ranges(self) -> List[Tuple[int, int]]:
        """Get all verification ranges for post-selection."""
        return [(g.start, g.end) for g in self.verification_groups]


# =============================================================================
# Helper functions for detector emission
# =============================================================================

def emit_detector(
    circuit,  # stim.Circuit
    measurement_indices: List[int],
    counter: DetectorCounter,
    total_measurements_so_far: int,
) -> int:
    """
    Emit a single DETECTOR instruction referencing measurement indices.
    
    Args:
        circuit: Stim circuit to append to
        measurement_indices: Indices into the measurement record (absolute)
        counter: Detector counter to track position
        total_measurements_so_far: Total measurements in circuit so far
        
    Returns:
        Detector index that was assigned
    """
    import stim
    
    # Convert absolute indices to relative (negative lookback)
    targets = []
    for idx in measurement_indices:
        lookback = idx - total_measurements_so_far
        targets.append(stim.target_rec(lookback))
    
    circuit.append("DETECTOR", targets)
    return counter.advance(1)


def emit_detector_group(
    circuit,  # stim.Circuit
    measurement_groups: List[List[int]],
    counter: DetectorCounter,
    total_measurements_so_far: int,
    dtype: DetectorType,
    block_id: int = 0,
    level: int = 0,
    round_idx: int = 0,
) -> DetectorGroup:
    """
    Emit multiple DETECTOR instructions for a group of measurement patterns.
    
    Each entry in measurement_groups becomes one DETECTOR instruction.
    
    Args:
        circuit: Stim circuit to append to
        measurement_groups: List of measurement index lists (one per detector)
        counter: Detector counter
        total_measurements_so_far: Total measurements in circuit
        dtype: Type of detector group
        block_id: Code block ID
        level: Concatenation level
        round_idx: EC round index
        
    Returns:
        DetectorGroup covering all emitted detectors
    """
    import stim
    
    start = counter.peek()
    
    for meas_indices in measurement_groups:
        targets = []
        for idx in meas_indices:
            lookback = idx - total_measurements_so_far
            targets.append(stim.target_rec(lookback))
        circuit.append("DETECTOR", targets)
        counter.advance(1)
    
    end = counter.peek()
    
    return DetectorGroup(
        start=start,
        end=end,
        dtype=dtype,
        block_id=block_id,
        level=level,
        round_idx=round_idx,
    )


def emit_parity_detector(
    circuit,  # stim.Circuit
    measurement_indices: List[int],
    counter: DetectorCounter,
    total_measurements_so_far: int,
) -> int:
    """
    Emit a DETECTOR that computes XOR (parity) of multiple measurements.
    
    This is used for syndrome extraction where syndrome bit = XOR of ancilla measurements.
    """
    import stim
    
    targets = []
    for idx in measurement_indices:
        lookback = idx - total_measurements_so_far
        targets.append(stim.target_rec(lookback))
    
    circuit.append("DETECTOR", targets)
    return counter.advance(1)
