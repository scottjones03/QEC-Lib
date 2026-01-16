"""
Base Gadget Interface for Concatenated Code EC Operations.

Gadgets are modular building blocks that can be composed to create
different fault-tolerant memory experiments.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple, Set

import stim
import numpy as np

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode
    from qectostim.noise.models import NoiseModel


@dataclass
class MeasurementEntry:
    """
    A single measurement record for tracking purposes.
    
    Used by some gadgets to provide detailed measurement info.
    """
    index: int                          # Global measurement index in circuit
    qubit: int                          # Physical qubit measured
    basis: str = 'Z'                    # 'X' or 'Z' basis
    block_idx: int = 0                  # Which code block
    level: int = 0                      # Concatenation level
    role: str = 'data'                  # 'data', 'x_syndrome', 'z_syndrome', etc.
    round_idx: int = 0                  # Round number for repeated measurements


@dataclass
class MeasurementMap:
    """Tracks measurement indices for a gadget's output."""
    # Block ID -> list of measurement indices for data qubits
    data_measurements: Dict[int, List[int]] = field(default_factory=dict)
    # Block ID -> list of measurement indices for syndrome ancillas
    syndrome_measurements: Dict[int, List[int]] = field(default_factory=dict)
    # Stabilizer type -> block ID -> measurement indices
    stabilizer_measurements: Dict[str, Dict[int, List[int]]] = field(default_factory=dict)
    # Total measurements emitted by this gadget
    total_measurements: int = 0
    # Starting offset in the overall circuit
    offset: int = 0
    # Optional detailed entries
    entries: List[MeasurementEntry] = field(default_factory=list)
    # Block ID -> output qubit indices (for gadgets that change qubit identity)
    output_qubits: Dict[int, List[int]] = field(default_factory=dict)
    # Pauli frame tracking: 'X' or 'Z' -> block ID -> measurement indices
    # Used by teleportation-based gadgets where output has a Pauli frame
    # For TrueKnillEC: pauli_frame['X'] contains indices of m_anc1
    pauli_frame: Dict[str, Dict[int, List[int]]] = field(default_factory=dict)
    # Block ID -> verification measurement indices (for verified ancilla prep)
    # Used by VerifiedAncillaGadget to track which measurements are verification checks
    verification_measurements: Dict[int, List[int]] = field(default_factory=dict)
    # Flag qubit measurements: stabilizer_type -> block_id -> stabilizer_idx -> flag_meas_indices
    # Used by FlaggedSyndromeGadget to track flag outcomes for flag-aware decoding
    # Structure: {'X': {block_id: {stab_idx: [flag_meas_indices], ...}, ...}, 'Z': {...}}
    flag_measurements: Dict[str, Dict[int, Dict[int, List[int]]]] = field(default_factory=dict)
    # Whether downstream processing must post-select on verification outcomes to remain FT
    requires_post_selection: bool = False
    # Whether DETECTORs were added to enforce post-selection in-circuit
    post_selection_detectors_added: bool = False
    # For verified ancillas: block_id -> verification measurement indices that have DETECTORs attached
    verification_detectors: Dict[int, List[int]] = field(default_factory=dict)
    # Type of measurements: "raw_ancilla" = n bits needing H @ raw, "syndrome" = (n-k) bits ready
    measurement_type: str = "raw_ancilla"  # Default for Steane-style EC


@dataclass 
class SyndromeSchedule:
    """Describes the syndrome extraction schedule for a gadget."""
    # Which stabilizer types are measured (e.g., ['X', 'Z'])
    stabilizer_types: List[str] = field(default_factory=list)
    # Number of rounds per type
    rounds_per_type: Dict[str, int] = field(default_factory=dict)
    # Order of operations
    schedule: List[Tuple[str, int]] = field(default_factory=list)  # (type, round_idx)


@dataclass
class LogicalMeasurementMap:
    """Maps from measurement indices to logical operator components."""
    # Observable index -> list of (measurement_idx, coefficient) pairs
    observable_components: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    # Block ID -> logical Z support qubits (after inner decode)
    inner_z_support: Dict[int, List[int]] = field(default_factory=dict)
    # Outer logical Z support (which blocks contribute)
    outer_z_support: List[int] = field(default_factory=list)


class Gadget(ABC):
    """
    Abstract base class for EC gadgets.
    
    A gadget is a self-contained circuit component that performs some
    operation on a concatenated code and tracks its measurements.
    
    Examples:
    - NoOpGadget: Just idle noise, no EC
    - KnillECGadget: Knill's teleportation EC
    - SteaneECGadget: Steane's cat-state EC
    - ExRecECGadget: Extended rectangle FT gadget
    """
    
    @abstractmethod
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],  # block_id -> qubit indices
        ancilla_qubits: Optional[Dict[int, List[int]]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> MeasurementMap:
        """
        Emit this gadget's operations to the circuit.
        
        Parameters
        ----------
        circuit : stim.Circuit
            The Stim circuit to append operations to
        data_qubits : dict
            Mapping from block_id to list of data qubit indices
        ancilla_qubits : dict, optional
            Mapping from block_id to list of ancilla qubit indices
        noise_model : NoiseModel, optional
            Noise model to apply to operations
        measurement_offset : int
            Starting measurement index for this gadget
            
        Returns
        -------
        MeasurementMap
            Tracking information for all measurements emitted
        """
        pass
    
    @abstractmethod
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        """
        Get the syndrome extraction schedule for this gadget.
        
        Returns
        -------
        SyndromeSchedule
            Description of what syndromes are measured and in what order
        """
        pass
    
    @abstractmethod
    def get_logical_map(
        self,
        inner_code: "CSSCode",
        outer_code: "CSSCode",
    ) -> LogicalMeasurementMap:
        """
        Get the logical measurement mapping for this gadget.
        
        Parameters
        ----------
        inner_code : CSSCode
            The inner code
        outer_code : CSSCode
            The outer code
            
        Returns
        -------
        LogicalMeasurementMap
            Mapping from measurements to logical operator components
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this gadget."""
        pass
    
    def get_syndrome_to_stabilizer_map(
        self,
        syndrome_measurements: List[int],
        stabilizer_type: str,
        code: "CSSCode",
    ) -> Dict[int, List[int]]:
        """
        Map syndrome measurements to stabilizer indices.
        
        For standard transversal EC, measurement i corresponds to stabilizer i.
        For Shor EC with redundant measurements, multiple measurements map to each stabilizer.
        
        Parameters
        ----------
        syndrome_measurements : List[int]
            List of measurement indices for this syndrome type
        stabilizer_type : str
            'X' or 'Z' syndrome type
        code : CSSCode
            The code being protected
            
        Returns
        -------
        Dict[int, List[int]]
            Mapping from stabilizer_idx -> list of measurement indices
            
        Default implementation assumes one-to-one mapping (transversal EC).
        """
        # Default: one-to-one mapping (measurement i → stabilizer i)
        result = {}
        for i, meas_idx in enumerate(syndrome_measurements):
            result[i] = [meas_idx]
        return result
    
    @property
    def requires_ancillas(self) -> bool:
        """Whether this gadget requires ancilla qubits."""
        return False
    
    @property
    def ancillas_per_block(self) -> int:
        """Number of ancilla qubits needed per block."""
        return 0
    
    def get_qubit_requirements(
        self,
        inner_code: "CSSCode",
        outer_code: "CSSCode",
    ) -> Dict[str, int]:
        """
        Get qubit requirements for this gadget.
        
        Returns
        -------
        dict with:
            - n_data_qubits: total data qubits
            - n_ancilla_qubits: total ancilla qubits  
            - n_total_qubits: total qubits needed
        """
        n_inner = inner_code.n
        n_outer = outer_code.n
        n_data = n_inner * n_outer
        n_ancilla = self.ancillas_per_block * n_outer if self.requires_ancillas else 0
        
        return {
            'n_data_qubits': n_data,
            'n_ancilla_qubits': n_ancilla,
            'n_total_qubits': n_data + n_ancilla,
        }


class CompositeGadget(Gadget):
    """
    A gadget composed of multiple sub-gadgets executed in sequence.
    
    Useful for building complex EC protocols from simpler components.
    """
    
    def __init__(self, gadgets: List[Gadget], name: str = "composite"):
        self._gadgets = gadgets
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def requires_ancillas(self) -> bool:
        return any(g.requires_ancillas for g in self._gadgets)
    
    @property
    def ancillas_per_block(self) -> int:
        return max((g.ancillas_per_block for g in self._gadgets), default=0)
    
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        ancilla_qubits: Optional[Dict[int, List[int]]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> MeasurementMap:
        """Emit all sub-gadgets in sequence."""
        combined_map = MeasurementMap(offset=measurement_offset)
        current_offset = measurement_offset
        
        for gadget in self._gadgets:
            mmap = gadget.emit(
                circuit, data_qubits, ancilla_qubits,
                noise_model, current_offset
            )
            
            # Merge measurement maps
            for block_id, indices in mmap.data_measurements.items():
                if block_id not in combined_map.data_measurements:
                    combined_map.data_measurements[block_id] = []
                combined_map.data_measurements[block_id].extend(indices)
            
            for block_id, indices in mmap.syndrome_measurements.items():
                if block_id not in combined_map.syndrome_measurements:
                    combined_map.syndrome_measurements[block_id] = []
                combined_map.syndrome_measurements[block_id].extend(indices)
            
            current_offset += mmap.total_measurements
        
        combined_map.total_measurements = current_offset - measurement_offset
        return combined_map
    
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        """Combine syndrome schedules from all sub-gadgets."""
        combined = SyndromeSchedule()
        
        for gadget in self._gadgets:
            schedule = gadget.get_syndrome_schedule()
            combined.stabilizer_types.extend(schedule.stabilizer_types)
            combined.schedule.extend(schedule.schedule)
            for stype, rounds in schedule.rounds_per_type.items():
                combined.rounds_per_type[stype] = (
                    combined.rounds_per_type.get(stype, 0) + rounds
                )
        
        return combined
    
    def get_logical_map(
        self,
        inner_code: "CSSCode",
        outer_code: "CSSCode",
    ) -> LogicalMeasurementMap:
        """Get logical map from last sub-gadget (which has final measurements)."""
        if self._gadgets:
            return self._gadgets[-1].get_logical_map(inner_code, outer_code)
        return LogicalMeasurementMap()
