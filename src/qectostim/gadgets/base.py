# src/qectostim/gadgets/base.py
"""
Enhanced Gadget base class for fault-tolerant logical gate implementations.

This module provides the abstract base class for all gadgets (logical operations),
with support for:
- N-dimensional topology tracking (2D, 3D, 4D + temporal)
- Phase-based circuit emission for fault-tolerant experiments
- Detector coordinate emission for decoder integration
- Metadata storage both in circuit and as separate dict

Architecture:
- Gadgets define WHAT they do via emit_next_phase() (multi-phase interface)
- Experiments (FaultTolerantGadgetExperiment) handle HOW to build the full circuit
- Experiment handles stabilizer rounds between phases for fault tolerance
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Tuple,
    Set,
    TYPE_CHECKING,
)

import stim

from qectostim.codes.abstract_code import Code
from qectostim.gadgets.coordinates import CoordND, get_code_dimension
from qectostim.gadgets.layout import GadgetLayout, QubitIndexMap, QubitAllocation

if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds import DetectorContext
    from qectostim.gadgets.scheduling import GadgetScheduler


class PhaseType(Enum):
    """Types of gadget phases that require different handling."""
    GATE = "gate"              # Pure gate operations (H, CNOT, etc.)
    MEASUREMENT = "measurement" # Mid-circuit measurement (teleportation, surgery)
    PREPARATION = "preparation" # Ancilla/state preparation
    COMPLETE = "complete"       # Gadget is finished


@dataclass
class PhaseResult:
    """
    Result from emitting a gadget phase.
    
    This tells the experiment what was emitted and what handling is needed
    before the next phase.
    
    Attributes:
        phase_type: What kind of phase was just emitted
        is_final: True if this was the last phase
        needs_stabilizer_rounds: Number of stabilizer rounds to run before next phase
        stabilizer_transform: How stabilizers changed during this phase
        measurement_count: Number of measurements emitted (for tracking)
        measurement_qubits: Which qubits were measured (for context updates)
        measured_blocks: Which blocks had qubits measured
        pauli_frame_update: Any Pauli frame corrections to apply
        extra: Additional phase-specific metadata
    """
    phase_type: PhaseType
    is_final: bool = False
    needs_stabilizer_rounds: int = 0
    stabilizer_transform: Optional["StabilizerTransform"] = None
    measurement_count: int = 0
    measurement_qubits: List[int] = field(default_factory=list)
    measured_blocks: List[str] = field(default_factory=list)
    pauli_frame_update: Optional[Dict[str, str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def gate_phase(cls, is_final: bool = True, 
                   transform: Optional["StabilizerTransform"] = None,
                   needs_stabilizer_rounds: int = 0) -> "PhaseResult":
        """Create a simple gate phase result."""
        return cls(
            phase_type=PhaseType.GATE,
            is_final=is_final,
            stabilizer_transform=transform,
            needs_stabilizer_rounds=needs_stabilizer_rounds,
        )
    
    @classmethod
    def measurement_phase(cls, count: int, qubits: List[int], 
                         blocks: List[str], is_final: bool = False) -> "PhaseResult":
        """Create a measurement phase result."""
        return cls(
            phase_type=PhaseType.MEASUREMENT,
            is_final=is_final,
            measurement_count=count,
            measurement_qubits=qubits,
            measured_blocks=blocks,
        )
    
    @classmethod
    def complete(cls) -> "PhaseResult":
        """Create a completion result."""
        return cls(
            phase_type=PhaseType.COMPLETE,
            is_final=True,
        )


@dataclass
class StabilizerTransform:
    """
    Describes how a gadget transforms stabilizer types.
    
    For detectors to be properly connected across a gadget, we need to know
    how X and Z stabilizers are affected by the gate.
    
    Attributes:
        x_becomes: What X stabilizers become ("X", "Z", or "Y")
        z_becomes: What Z stabilizers become ("X", "Z", or "Y")
        clear_history: Whether to clear stabilizer history (can't compare across gate)
        swap_xz: Whether to swap X and Z in the history (for H-like gates)
    """
    x_becomes: str = "X"
    z_becomes: str = "Z"
    clear_history: bool = False
    swap_xz: bool = False
    
    @classmethod
    def identity(cls) -> "StabilizerTransform":
        """No transformation."""
        return cls(x_becomes="X", z_becomes="Z")
    
    @classmethod
    def hadamard(cls) -> "StabilizerTransform":
        """Hadamard swaps X and Z."""
        return cls(x_becomes="Z", z_becomes="X", clear_history=True, swap_xz=True)
    
    @classmethod
    def pauli(cls) -> "StabilizerTransform":
        """Pauli gates don't change stabilizer types."""
        return cls(x_becomes="X", z_becomes="Z")


@dataclass
class ObservableTransform:
    """
    Describes how a gadget transforms the logical observable.
    
    Used to compute the correct OBSERVABLE_INCLUDE at the end.
    
    Attributes:
        transform: Dict mapping Pauli -> Pauli (e.g., {"X": "Z", "Z": "X"} for H)
    """
    transform: Dict[str, str] = field(default_factory=lambda: {"X": "X", "Z": "Z", "Y": "Y"})
    
    @classmethod
    def identity(cls) -> "ObservableTransform":
        return cls(transform={"X": "X", "Z": "Z", "Y": "Y"})
    
    @classmethod
    def hadamard(cls) -> "ObservableTransform":
        return cls(transform={"X": "Z", "Z": "X", "Y": "-Y"})
    
    @classmethod
    def s_gate(cls) -> "ObservableTransform":
        return cls(transform={"X": "Y", "Z": "Z", "Y": "-X"})
    
    @classmethod
    def from_gate_name(cls, gate_name: str) -> "ObservableTransform":
        """Create transform from gate name."""
        gate_transforms = {
            "H": cls.hadamard(),
            "S": cls.s_gate(),
            "S_DAG": cls(transform={"X": "-Y", "Z": "Z", "Y": "X"}),
            "X": cls(transform={"X": "X", "Z": "-Z", "Y": "-Y"}),
            "Y": cls(transform={"X": "-X", "Z": "-Z", "Y": "Y"}),
            "Z": cls(transform={"X": "-X", "Z": "Z", "Y": "-Y"}),
            "T": cls.identity(),  # T doesn't change Paulis (just adds phase)
            "T_DAG": cls.identity(),
            "CNOT": cls.identity(),  # For now, simplified
            "CZ": cls.identity(),
            "SWAP": cls.identity(),
        }
        return gate_transforms.get(gate_name.upper(), cls.identity())
    
    def to_stabilizer_transform(self) -> "StabilizerTransform":
        """
        Convert to StabilizerTransform for detector tracking.
        
        Maps the observable transform to a stabilizer transform by extracting
        the Pauli type (ignoring signs).
        """
        x_becomes = self.transform.get("X", "X").lstrip("-")
        z_becomes = self.transform.get("Z", "Z").lstrip("-")
        
        # Check for swap (Hadamard-like)
        swap_xz = (x_becomes == "Z" and z_becomes == "X")
        clear_history = swap_xz  # Clear history when X/Z swap
        
        return StabilizerTransform(
            x_becomes=x_becomes,
            z_becomes=z_becomes,
            clear_history=clear_history,
            swap_xz=swap_xz,
        )


@dataclass
class TwoQubitObservableTransform:
    """
    Describes how a two-qubit gate transforms logical observables across two blocks.
    
    For two-qubit gates like CNOT, CZ, SWAP, the logical observables on each
    block can be affected by the gate:
    
    CNOT (control=block_0, target=block_1):
        X_ctrl → X_ctrl               (unchanged)
        Z_ctrl → Z_ctrl ⊗ Z_tgt       (spreads to target)
        X_tgt → X_ctrl ⊗ X_tgt        (picks up control)
        Z_tgt → Z_tgt                 (unchanged)
    
    CZ (symmetric on both blocks):
        X_ctrl → X_ctrl ⊗ Z_tgt       (picks up Z from other)
        Z_ctrl → Z_ctrl               (unchanged)
        X_tgt → Z_ctrl ⊗ X_tgt        (picks up Z from other)
        Z_tgt → Z_tgt                 (unchanged)
    
    SWAP:
        X_ctrl ↔ X_tgt                (swapped)
        Z_ctrl ↔ Z_tgt                (swapped)
    
    Attributes:
        control_x_to: What control X becomes (e.g., ("X", None) means X_ctrl only)
        control_z_to: What control Z becomes (e.g., ("Z", "Z") means Z_ctrl ⊗ Z_tgt)
        target_x_to: What target X becomes
        target_z_to: What target Z becomes
        
    Each tuple is (block_0_component, block_1_component) where None means
    the observable doesn't involve that block.
    """
    control_x_to: Tuple[Optional[str], Optional[str]] = ("X", None)
    control_z_to: Tuple[Optional[str], Optional[str]] = ("Z", None)
    target_x_to: Tuple[Optional[str], Optional[str]] = (None, "X")
    target_z_to: Tuple[Optional[str], Optional[str]] = (None, "Z")
    
    @classmethod
    def identity(cls) -> "TwoQubitObservableTransform":
        """Identity transform - no changes to observables."""
        return cls(
            control_x_to=("X", None),
            control_z_to=("Z", None),
            target_x_to=(None, "X"),
            target_z_to=(None, "Z"),
        )
    
    @classmethod
    def cnot(cls) -> "TwoQubitObservableTransform":
        """
        CNOT transform (control=block_0, target=block_1).
        
        CNOT propagates X from target to control and Z from control to target:
            X_ctrl → X_ctrl
            Z_ctrl → Z_ctrl ⊗ Z_tgt
            X_tgt → X_ctrl ⊗ X_tgt
            Z_tgt → Z_tgt
        """
        return cls(
            control_x_to=("X", None),       # X_ctrl unchanged
            control_z_to=("Z", "Z"),        # Z_ctrl spreads to Z_tgt
            target_x_to=("X", "X"),         # X_tgt picks up X_ctrl
            target_z_to=(None, "Z"),        # Z_tgt unchanged
        )
    
    @classmethod
    def cz(cls) -> "TwoQubitObservableTransform":
        """
        CZ transform (symmetric).
        
        CZ propagates Z across blocks when X is applied:
            X_ctrl → X_ctrl ⊗ Z_tgt
            Z_ctrl → Z_ctrl
            X_tgt → Z_ctrl ⊗ X_tgt
            Z_tgt → Z_tgt
        """
        return cls(
            control_x_to=("X", "Z"),        # X_ctrl picks up Z_tgt
            control_z_to=("Z", None),       # Z_ctrl unchanged
            target_x_to=("Z", "X"),         # X_tgt picks up Z_ctrl
            target_z_to=(None, "Z"),        # Z_tgt unchanged
        )
    
    @classmethod
    def swap(cls) -> "TwoQubitObservableTransform":
        """
        SWAP transform.
        
        SWAP exchanges observables between blocks:
            X_ctrl → X_tgt
            Z_ctrl → Z_tgt
            X_tgt → X_ctrl
            Z_tgt → Z_ctrl
        """
        return cls(
            control_x_to=(None, "X"),       # X_ctrl becomes X_tgt
            control_z_to=(None, "Z"),       # Z_ctrl becomes Z_tgt
            target_x_to=("X", None),        # X_tgt becomes X_ctrl
            target_z_to=("Z", None),        # Z_tgt becomes Z_ctrl
        )
    
    def to_single_block_transforms(self) -> Tuple[ObservableTransform, ObservableTransform]:
        """
        Convert to per-block transforms (for compatibility with existing code).
        
        Note: This loses the cross-block entanglement information but provides
        a simplified view for single-block analysis.
        
        Returns:
            Tuple of (control_transform, target_transform) as ObservableTransform
        """
        # Extract diagonal components only
        ctrl_x = self.control_x_to[0] if self.control_x_to[0] else "X"
        ctrl_z = self.control_z_to[0] if self.control_z_to[0] else "Z"
        tgt_x = self.target_x_to[1] if self.target_x_to[1] else "X"
        tgt_z = self.target_z_to[1] if self.target_z_to[1] else "Z"
        
        ctrl_transform = ObservableTransform(transform={"X": ctrl_x, "Z": ctrl_z, "Y": "Y"})
        tgt_transform = ObservableTransform(transform={"X": tgt_x, "Z": tgt_z, "Y": "Y"})
        
        return ctrl_transform, tgt_transform


@dataclass
class GadgetMetadata:
    """
    Metadata for a gadget execution, stored separately for decoder access.
    
    Attributes:
        gadget_type: Type of gadget ("transversal", "teleportation", "surgery")
        logical_operation: Description of logical operation (e.g., "CNOT", "H", "T")
        input_codes: List of input code names/types
        output_codes: List of output code names/types
        qubit_index_map: Mapping from local block indices to global circuit indices
        detector_coords: Dict mapping detector index to (spatial_coords, time_coord)
        logical_observable_coords: Dict mapping observable index to coordinates
        ancilla_info: Information about bridge/measurement ancillas
        timing_info: Dict with layer counts, depths, etc.
        extra: Additional gadget-specific metadata
    """
    gadget_type: str
    logical_operation: str
    input_codes: List[str] = field(default_factory=list)
    output_codes: List[str] = field(default_factory=list)
    qubit_index_map: Optional[QubitIndexMap] = None
    detector_coords: Dict[int, Tuple[CoordND, float]] = field(default_factory=dict)
    logical_observable_coords: Dict[int, CoordND] = field(default_factory=dict)
    ancilla_info: Dict[str, Any] = field(default_factory=dict)
    timing_info: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gadget_type": self.gadget_type,
            "logical_operation": self.logical_operation,
            "input_codes": self.input_codes,
            "output_codes": self.output_codes,
            "qubit_index_map": {
                "block_to_global": self.qubit_index_map.block_to_global if self.qubit_index_map else {},
                "global_to_block": self.qubit_index_map.global_to_block if self.qubit_index_map else {},
                "global_coords": {k: list(v) for k, v in (self.qubit_index_map.global_coords.items() if self.qubit_index_map else {})}
            },
            "detector_coords": {k: (list(v[0]), v[1]) for k, v in self.detector_coords.items()},
            "logical_observable_coords": {k: list(v) for k, v in self.logical_observable_coords.items()},
            "ancilla_info": self.ancilla_info,
            "timing_info": self.timing_info,
            "extra": self.extra,
        }


class Gadget(ABC):
    """
    Abstract base for gadgets: logical operations implemented by specific protocols
    (teleportation, CSS surgery, transversal wrappers, etc.).
    
    A Gadget encapsulates:
    1. Phase-based gate emission: Multi-phase interface for complex operations
    2. Transform specification: How stabilizers/observables change through the gate
    3. Metadata generation: Producing decoder-accessible metadata
    
    PRIMARY INTERFACE (required for FaultTolerantGadgetExperiment):
    - num_phases: Number of phases in this gadget
    - emit_next_phase(): Emit the next phase, return PhaseResult
    - get_stabilizer_transform(): Overall stabilizer transformation
    - get_observable_transform(): Overall observable transformation
    
    The experiment calls emit_next_phase() repeatedly, handling stabilizer
    rounds between phases as indicated by PhaseResult.needs_stabilizer_rounds.
    
    Phase-based approach supports:
    - Simple transversal gates: 1 phase (just gates)
    - Teleportation: 3 phases (prep → CNOTs → measurement)
    - Surgery: Multiple phases (merge rounds with intermediate stabilizers)
    
    The workflow:
    1. experiment = FaultTolerantGadgetExperiment(codes, gadget, noise)
    2. circuit = experiment.to_stim()  # Handles everything
    
    Internally, experiment does:
        gadget.reset_phases()
        while not done:
            result = gadget.emit_next_phase(circuit, alloc, ctx)
            if result.needs_stabilizer_rounds > 0:
                emit_stabilizer_rounds(result.needs_stabilizer_rounds)
            apply_transform(result.stabilizer_transform)
            if result.is_final:
                break
    """
    
    def __init__(self):
        """Initialize gadget with empty cached state."""
        self._cached_layout: Optional[GadgetLayout] = None
        self._cached_scheduler: Optional["GadgetScheduler"] = None
        self._cached_metadata: Optional[GadgetMetadata] = None
        self._dimension: int = 2  # Default to 2D
        self._current_phase: int = 0  # Phase counter for multi-phase gadgets
    
    # =========================================================================
    # PRIMARY INTERFACE - Phase-based emission
    # =========================================================================
    
    @property
    def num_phases(self) -> int:
        """
        Return the number of phases in this gadget.
        
        Override in subclasses for multi-phase gadgets.
        Default is 1 (single-phase gadget like transversal gates).
        """
        return 1
    
    def reset_phases(self) -> None:
        """Reset phase counter for a new circuit generation."""
        self._current_phase = 0
    
    @abstractmethod
    def emit_next_phase(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """
        Emit the next phase of this gadget.
        
        This is the primary interface for gadgets. Each call emits one phase
        and returns a PhaseResult describing what was emitted and what the
        experiment should do before the next phase.
        
        Phases should emit:
        - Physical gate operations (H, CNOT, etc.)
        - Measurements (if this phase requires them)
        - TICKs as needed for timing
        
        Phases should NOT emit:
        - Stabilizer rounds (experiment handles based on PhaseResult)
        - QUBIT_COORDS (handled by experiment at start)
        - Detectors (experiment handles based on context)
        - OBSERVABLE_INCLUDE (experiment handles at end)
        
        Parameters
        ----------
        circuit : stim.Circuit
            The circuit to emit operations into.
        alloc : QubitAllocation
            Qubit allocation with block-to-global index mappings.
        ctx : DetectorContext
            Detector context for measurement tracking.
            
        Returns
        -------
        PhaseResult
            Describes what was emitted and what handling is needed.
        """
        ...
    
    @abstractmethod
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """
        Return how this gadget transforms stabilizer types.
        
        This is the OVERALL transformation after all phases complete.
        Individual phase transforms are returned in PhaseResult.
        
        Returns
        -------
        StabilizerTransform
            Description of how X/Z stabilizers change.
        """
        ...
    
    @abstractmethod
    def get_observable_transform(self) -> ObservableTransform:
        """
        Return how this gadget transforms the logical observable.
        
        This is used by the experiment to correctly emit OBSERVABLE_INCLUDE.
        
        Returns
        -------
        ObservableTransform
            Description of how logical X/Z/Y change.
        """
        ...
    
    def validate_codes(self, codes: List[Code]) -> None:
        """
        Validate that codes are compatible with this gadget.
        
        Override in subclasses to add specific validation.
        Default implementation checks that codes list is not empty.
        
        Raises:
            ValueError: If codes are incompatible
        """
        if not codes:
            raise ValueError("At least one code is required")
    
    def supports_code(self, code: Code) -> bool:
        """
        Check if this gadget supports a given code type.
        
        Override in subclasses for specific compatibility checks.
        Default returns True.
        """
        return True
    
    @abstractmethod
    def compute_layout(self, codes: List[Code]) -> GadgetLayout:
        """
        Compute the spatial layout for this gadget's code blocks.
        
        Each gadget MUST implement this to define how code blocks are
        positioned relative to each other. This is gadget-specific:
        
        - Transversal gates: Place blocks side-by-side with spacing
        - Surgery: Arrange blocks for merge boundaries  
        - Teleportation: Place data and ancilla blocks
        
        The layout is used by the experiment to:
        1. Emit correct QUBIT_COORDS
        2. Compute non-overlapping offsets
        3. Map block-local indices to global circuit indices
        
        Parameters
        ----------
        codes : List[Code]
            The codes involved in this gadget.
            
        Returns
        -------
        GadgetLayout
            Layout with block positions and offsets.
        """
        ...
    
    @abstractmethod
    def get_metadata(self) -> GadgetMetadata:
        """
        Get metadata for decoder integration.
        
        Returns metadata that can be passed to decoders, including:
        - Qubit index mappings
        - Detector coordinates
        - Logical observable information
        - Timing and layer information
        
        Returns:
            GadgetMetadata with decoder-relevant information
        """
        ...
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    @property
    def dimension(self) -> int:
        """Get the spatial dimension of this gadget (2, 3, or 4)."""
        return self._dimension
    
    @dimension.setter
    def dimension(self, value: int):
        """Set the spatial dimension (2, 3, or 4)."""
        if value not in (2, 3, 4):
            raise ValueError(f"Dimension must be 2, 3, or 4, got {value}")
        self._dimension = value
    
    def infer_dimension(self, codes: List[Code]) -> int:
        """
        Infer the spatial dimension from the codes' qubit coordinates.
        
        Returns the maximum dimension found across all codes.
        """
        max_dim = 2
        for code in codes:
            dim = get_code_dimension(code)
            max_dim = max(max_dim, dim)
        return max_dim
    
    def clear_cache(self) -> None:
        """Clear cached layout, scheduler, and metadata."""
        self._cached_layout = None
        self._cached_scheduler = None
        self._cached_metadata = None
    
    def get_detector_coords(
        self,
        layout: GadgetLayout,
        time: float = 0.0,
    ) -> Dict[int, Tuple[CoordND, float]]:
        """
        Compute detector coordinates for all detectors in the gadget.
        
        Detectors are identified by their stabilizer origin in the
        spatial coordinates, plus a temporal coordinate.
        
        Args:
            layout: The GadgetLayout with block positions
            time: Base time coordinate for this round
            
        Returns:
            Dict mapping detector index to (spatial_coord, time)
        """
        detector_coords: Dict[int, Tuple[CoordND, float]] = {}
        detector_idx = 0
        
        for block_name, block_info in layout.blocks.items():
            code = block_info.code
            offset = block_info.offset
            
            # Get stabilizer positions if available
            if hasattr(code, 'stabilizer_coords'):
                stab_coords = code.stabilizer_coords()
                for local_coord in stab_coords:
                    # Translate to global coordinates
                    global_coord = tuple(
                        local_coord[i] + offset[i] if i < len(local_coord) else offset[i]
                        for i in range(len(offset))
                    )
                    detector_coords[detector_idx] = (global_coord, time)
                    detector_idx += 1
        
        return detector_coords


class TransversalGadgetMixin:
    """
    Mixin for gadgets that apply gates transversally (qubit-by-qubit).
    
    Provides helper methods for transversal gate application where
    the logical gate is implemented by applying physical gates to
    each qubit in parallel.
    """
    
    def get_transversal_gate_pairs(
        self,
        code: Code,
        gate_name: str,
    ) -> List[Tuple[int, ...]]:
        """
        Get qubit indices for transversal gate application.
        
        For single-qubit gates, returns list of (qubit_idx,) tuples.
        For two-qubit gates like CZ, returns list of (ctrl, tgt) pairs.
        
        Args:
            code: The code to apply gate to
            gate_name: Gate name (H, S, T, CZ, CNOT, etc.)
            
        Returns:
            List of qubit index tuples for gate application
        """
        n = code.n  # Number of data qubits
        
        if gate_name in ("H", "S", "T", "X", "Y", "Z", "S_DAG", "T_DAG"):
            # Single-qubit gates: apply to all data qubits
            return [(i,) for i in range(n)]
        elif gate_name == "CZ":
            # CZ between corresponding qubits of two codes
            # Caller should handle multi-code case
            return [(i,) for i in range(n)]
        elif gate_name == "CNOT":
            # CNOT between corresponding qubits
            return [(i,) for i in range(n)]
        else:
            raise ValueError(f"Unknown transversal gate: {gate_name}")
    
    def check_transversal_support(self, code: Code, gate_name: str) -> bool:
        """
        Check if a code supports a transversal implementation of a gate.
        
        Queries the code's transversal_gates property if available.
        
        Args:
            code: The code to check
            gate_name: The gate name to check
            
        Returns:
            True if gate is supported transversally
        """
        if hasattr(code, 'transversal_gates'):
            supported = code.transversal_gates()
            return gate_name in supported
        # Default: assume H, CZ, CNOT are available for CSS codes
        if hasattr(code, 'is_css') and code.is_css:
            return gate_name in ("H", "CZ", "CNOT")
        return False


class SurgeryGadgetMixin:
    """
    Mixin for gadgets that use CSS code surgery techniques.
    
    Provides helper methods for lattice surgery operations including
    merge/split operations and logical operator measurement.
    """
    
    def get_merge_boundary(
        self,
        code1: Code,
        code2: Code,
        operator_type: str = "Z",
    ) -> List[Tuple[int, int]]:
        """
        Get qubit pairs on the boundary for merge operation.
        
        Args:
            code1: First code block
            code2: Second code block
            operator_type: "X" or "Z" for operator type
            
        Returns:
            List of (qubit_from_code1, qubit_from_code2) pairs
        """
        # Default: return empty, subclasses should override
        return []
    
    def compute_surgery_ancillas(
        self,
        boundary_pairs: List[Tuple[int, int]],
        layout: GadgetLayout,
    ) -> List[Tuple[int, CoordND]]:
        """
        Compute ancilla positions for surgery measurement.
        
        Args:
            boundary_pairs: Pairs of qubits to measure
            layout: Current layout with block positions
            
        Returns:
            List of (ancilla_idx, coord) tuples
        """
        ancillas = []
        for idx, (q1, q2) in enumerate(boundary_pairs):
            # Place ancilla between the two qubits
            coord1 = layout.qubit_map.global_coords.get(q1, (0.0,) * self._dimension)
            coord2 = layout.qubit_map.global_coords.get(q2, (0.0,) * self._dimension)
            mid_coord = tuple((c1 + c2) / 2 for c1, c2 in zip(coord1, coord2))
            ancillas.append((layout.total_qubits + idx, mid_coord))
        return ancillas


class TeleportationGadgetMixin:
    """
    Mixin for gadgets that use teleportation protocols.
    
    Provides helper methods for teleportation-based logical gate
    implementations using Bell pairs and measurements.
    """
    
    def get_bell_pair_qubits(
        self,
        code: Code,
        layout: GadgetLayout,
    ) -> List[Tuple[int, int]]:
        """
        Get qubit pairs for Bell pair preparation.
        
        Args:
            code: The code for teleportation
            layout: Current layout
            
        Returns:
            List of (data_qubit, ancilla_qubit) pairs
        """
        # Default: pair each data qubit with corresponding ancilla
        n = code.n
        return [(i, n + i) for i in range(n)]
    
    def compute_correction_pauli(
        self,
        x_measurement: int,
        z_measurement: int,
    ) -> str:
        """
        Compute Pauli correction based on measurement outcomes.
        
        Standard teleportation corrections:
        - m_x=0, m_z=0: I
        - m_x=1, m_z=0: Z
        - m_x=0, m_z=1: X
        - m_x=1, m_z=1: Y
        
        Args:
            x_measurement: X-basis measurement result (0 or 1)
            z_measurement: Z-basis measurement result (0 or 1)
            
        Returns:
            Pauli operator name ("I", "X", "Y", "Z")
        """
        if x_measurement == 0 and z_measurement == 0:
            return "I"
        elif x_measurement == 1 and z_measurement == 0:
            return "Z"
        elif x_measurement == 0 and z_measurement == 1:
            return "X"
        else:
            return "Y"
