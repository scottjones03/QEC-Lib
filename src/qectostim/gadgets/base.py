# src/qectostim/gadgets/base.py
"""
Gadget base class and core types for fault-tolerant logical gate implementations.

This module provides:

* The abstract base class :class:`Gadget` for all logical operations.
* Core value types: :class:`PhaseType`, :class:`PhaseResult`, :class:`FrameUpdate`,
  :class:`StabilizerTransform`, :class:`ObservableTransform`,
  :class:`TwoQubitObservableTransform`, :class:`GadgetMetadata`.
* Block-name normalisation utilities.

Configuration data classes (``ObservableConfig``, ``PreparationConfig``, etc.)
live in :mod:`qectostim.gadgets.configs` and are **re-exported** here for
backward compatibility.

Mixin classes (``AutoCSSGadgetMixin``, ``TransversalGadgetMixin``,
``TeleportationGadgetMixin``) live in :mod:`qectostim.gadgets.mixins` and are
likewise **re-exported** from this module.
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
    Union,
    TYPE_CHECKING,
)

import stim

from qectostim.codes.abstract_code import Code
from qectostim.gadgets.coordinates import CoordND, get_code_dimension
from qectostim.gadgets.layout import GadgetLayout, QubitIndexMap, QubitAllocation

# ── Re-exports from configs.py (backward compatibility) ─────────────────
from qectostim.gadgets.configs import (  # noqa: F401
    HeisenbergFrame,
    ObservableTerm,
    ObservableConfig,
    BlockPreparationConfig,
    PreparationConfig,
    MeasurementConfig,
    CrossingDetectorTerm,
    CrossingDetectorFormula,
    CrossingDetectorConfig,
    BoundaryDetectorConfig,
)

# ── Re-exports from mixins.py (backward compatibility) ──────────────────
from qectostim.gadgets.mixins import (  # noqa: F401
    AutoCSSGadgetMixin,
    TransversalGadgetMixin,
    TeleportationGadgetMixin,
)

if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds import DetectorContext


# =============================================================================
# Block Name Normalization
# =============================================================================
# These mappings allow the experiment to handle different naming conventions
# for blocks across gadgets. This eliminates hardcoded name checks.

BLOCK_NAME_ALIASES = {
    # Data block aliases
    "data_block": "block_0",
    "data": "block_0",
    "control": "block_0",
    "control_block": "block_0",
    
    # First ancilla block aliases
    "ancilla_block": "block_1",
    "ancilla": "block_1",
    "target": "block_1",
    "target_block": "block_1",
    "ancilla1": "block_1",
    
    # Second ancilla block aliases (for 3-block gadgets)
    "ancilla2": "block_2",
    "output": "block_2",
    "output_block": "block_2",
}


def normalize_block_name(name: str) -> str:
    """
    Normalize block name to canonical form (block_0, block_1, block_2...).
    
    This eliminates hardcoded block name mappings throughout the codebase
    by providing a single point of normalization.
    
    Parameters
    ----------
    name : str
        Block name in any convention (e.g., "data_block", "ancilla_block").
        
    Returns
    -------
    str
        Canonical block name (e.g., "block_0", "block_1").
        
    Examples
    --------
    >>> normalize_block_name("data_block")
    "block_0"
    >>> normalize_block_name("ancilla_block")  
    "block_1"
    >>> normalize_block_name("block_0")
    "block_0"
    """
    return BLOCK_NAME_ALIASES.get(name, name)


def blocks_match(name1: str, name2: str) -> bool:
    """
    Check if two block names refer to the same block.
    
    Handles different naming conventions by normalizing before comparison.
    
    Parameters
    ----------
    name1 : str
        First block name.
    name2 : str
        Second block name.
        
    Returns
    -------
    bool
        True if both names refer to the same block.
        
    Examples
    --------
    >>> blocks_match("data_block", "block_0")
    True
    >>> blocks_match("ancilla_block", "block_1")
    True
    >>> blocks_match("block_0", "block_1")
    False
    """
    return normalize_block_name(name1) == normalize_block_name(name2)


def get_canonical_block_names(num_blocks: int) -> List[str]:
    """
    Get canonical block names for a given number of blocks.
    
    Parameters
    ----------
    num_blocks : int
        Number of blocks.
        
    Returns
    -------
    List[str]
        List of canonical block names ["block_0", "block_1", ...].
    """
    return [f"block_{i}" for i in range(num_blocks)]


class PhaseType(Enum):
    """Types of gadget phases that require different handling."""
    GATE = "gate"              # Pure gate operations (H, CNOT, etc.)
    MEASUREMENT = "measurement" # Mid-circuit measurement (teleportation, surgery)
    PREPARATION = "preparation" # Ancilla/state preparation
    COMPLETE = "complete"       # Gadget is finished


@dataclass
class FrameUpdate:
    """
    Typed Pauli frame update from a gadget measurement phase.
    
    This replaces the untyped Dict[str, Any] previously used for frame updates,
    providing compile-time checking and clear documentation of the contract
    between gadgets and the phase orchestrator.
    
    Frame updates describe how measurement outcomes affect the logical
    Pauli frame (classical corrections that will be applied at readout).
    
    Attributes
    ----------
    block_name : str
        Which block's frame is affected by these corrections.
    x_meas : List[int]
        Measurement indices (relative to phase) whose XOR determines X correction.
    z_meas : List[int]
        Measurement indices (relative to phase) whose XOR determines Z correction.
    teleport : bool
        If True, this is a teleportation where logical info transfers between blocks.
        The source block is destroyed, and corrections apply to the target.
    source_block : Optional[str]
        For teleportation, the block being measured out (source of logical info).
    """
    block_name: str
    x_meas: List[int] = field(default_factory=list)
    z_meas: List[int] = field(default_factory=list)
    teleport: bool = False
    source_block: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for backwards compatibility with existing code."""
        result = {
            'block_name': self.block_name,
            'x_meas': self.x_meas,
            'z_meas': self.z_meas,
            'teleport': self.teleport,
        }
        if self.source_block is not None:
            result['source_block'] = self.source_block
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FrameUpdate":
        """Create from dict for backwards compatibility."""
        return cls(
            block_name=d.get('block_name', 'main'),
            x_meas=list(d.get('x_meas', d.get('x_stab_meas_indices', []))),
            z_meas=list(d.get('z_meas', d.get('z_stab_meas_indices', []))),
            teleport=d.get('teleport', False),
            source_block=d.get('source_block'),
        )


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
        destroyed_blocks: Block names that are destroyed (measured out) during this phase
        pauli_frame_update: Pauli frame corrections to apply from measurements.
            Can be either a FrameUpdate dataclass (preferred) or legacy Dict[str, Any].
            See FrameUpdate for the typed structure.
        extra: Additional phase-specific metadata
    """
    phase_type: PhaseType
    is_final: bool = False
    needs_stabilizer_rounds: int = 0
    stabilizer_transform: Optional["StabilizerTransform"] = None
    measurement_count: int = 0
    measurement_qubits: List[int] = field(default_factory=list)
    measured_blocks: List[str] = field(default_factory=list)
    destroyed_blocks: Set[str] = field(default_factory=set)
    pauli_frame_update: Optional[Union[FrameUpdate, Dict[str, Any]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    # Per-block stabilizer basis constraints for inter-phase EC rounds.
    # Maps block_name -> "X" | "Z" | "BOTH".  Blocks not listed default to
    # "BOTH".  E.g. during XX merge, blocks 1 & 2 must be X-only because
    # bridge CX corrupts their Z parity.
    stab_constraints: Optional[Dict[str, str]] = None
    
    def get_frame_update(self) -> Optional[FrameUpdate]:
        """Get frame update as typed FrameUpdate, converting from dict if needed."""
        if self.pauli_frame_update is None:
            return None
        if isinstance(self.pauli_frame_update, FrameUpdate):
            return self.pauli_frame_update
        return FrameUpdate.from_dict(self.pauli_frame_update)
    
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
        skip_first_round: Whether to skip first-round detectors (for teleportation)
    """
    x_becomes: str = "X"
    z_becomes: str = "Z"
    clear_history: bool = False
    swap_xz: bool = False
    skip_first_round: bool = False
    # Per-block override for skip_first_round and clear_history.
    # Maps block_name -> dict with optional keys:
    #   "skip_first_round": bool  (override the global skip_first_round)
    #   "clear_history": bool     (override the global clear_history)
    #   "skip_z": bool            (skip first-round Z detectors only)
    #   "skip_x": bool            (skip first-round X detectors only)
    # Blocks not listed use the global settings.
    per_block: Optional[Dict[str, Dict[str, bool]]] = None
    
    @classmethod
    def identity(
        cls,
        clear_history: bool = False,
        skip_first_round: bool = False,
        per_block: Optional[Dict[str, Dict[str, bool]]] = None,
    ) -> "StabilizerTransform":
        """No transformation of stabilizer types.
        
        Parameters
        ----------
        clear_history : bool
            If True, clear measurement history. This is needed for two-qubit gates
            (CNOT, CZ, SWAP) where blocks become entangled and measurements from
            before the gate cannot be compared to measurements after.
        skip_first_round : bool
            If True, skip emitting first-round detectors after the gate. This is
            needed for partial entangling gates (like lattice surgery XX merge)
            where the blocks become entangled such that individual stabilizer
            measurements are non-deterministic.
        per_block : dict or None
            Per-block overrides for skip_first_round, clear_history, skip_z, skip_x.
        """
        return cls(
            x_becomes="X",
            z_becomes="Z",
            clear_history=clear_history,
            skip_first_round=skip_first_round,
            per_block=per_block,
        )
    
    @classmethod
    def hadamard(cls) -> "StabilizerTransform":
        """Hadamard swaps X and Z.
        
        For CSS codes, Hadamard swaps X↔Z stabilizer types. After the gate:
        - What were Z stabilizers become X stabilizers (and vice versa)
        - The deterministic basis swaps: if measuring in Z basis before H,
          Z stabilizers were deterministic. After H, X stabilizers are deterministic.
        
        We use swap_xz=True to swap the measurement_basis, which correctly:
        - Emits first-round detectors for the now-deterministic type
        - Skips first-round detectors for the now-random type
        
        We do NOT use skip_first_round=True because one stabilizer type IS
        deterministic after the gate - the swapped measurement basis handles this.
        """
        return cls(x_becomes="Z", z_becomes="X", clear_history=True, swap_xz=True, skip_first_round=False)
    
    @classmethod
    def pauli(cls) -> "StabilizerTransform":
        """Pauli gates don't change stabilizer types."""
        return cls(x_becomes="X", z_becomes="Z")
    
    @classmethod
    def teleportation(cls, swap_xz: bool = False) -> "StabilizerTransform":
        """Teleportation: clear history and skip first-round detectors."""
        return cls(clear_history=True, swap_xz=swap_xz, skip_first_round=True)


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
    
    def __init__(self, input_state: str = "0"):
        """Initialize gadget with empty cached state.
        
        Parameters
        ----------
        input_state : str
            Logical input state ("0" or "+"). Determines measurement basis:
            - "0" → measurement in Z basis
            - "+" → measurement in X basis
        """
        self._cached_layout: Optional[GadgetLayout] = None
        self._cached_metadata: Optional[GadgetMetadata] = None
        self._dimension: int = 2  # Default to 2D
        self._current_phase: int = 0  # Phase counter for multi-phase gadgets
        self._prep_meas_start: Optional[int] = None  # Set by set_prep_meas_start()
        self._input_state: str = input_state
    
    # =========================================================================
    # PRIMARY INTERFACE - Phase-based emission
    # =========================================================================
    
    @property
    def gate_name(self) -> str:
        """
        Return the gate name for this gadget (e.g., "H", "CZ", "CNOT").
        
        Used for observable transforms and metadata. Override in subclasses
        either by defining a @property or setting self._gate_name.
        Default: class name stripped of common suffixes.
        """
        if hasattr(self, '_gate_name'):
            return self._gate_name
        name = self.__class__.__name__
        for suffix in ("Gadget", "Teleport", "Transversal"):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name
    
    @gate_name.setter
    def gate_name(self, value: str) -> None:
        """Allow subclasses to set gate_name as an attribute."""
        self._gate_name = value
    
    def get_measurement_basis(self) -> str:
        """
        Return the measurement basis for this gadget, derived from input state.
        
        The measurement basis is determined by the gadget's input state(s):
        - |0⟩ input → Z basis measurement
        - |+⟩ input → X basis measurement
        
        Multi-block gadgets (e.g., TransversalCNOT) override this to derive
        per-block bases from their output states.
        
        Returns
        -------
        str
            "Z" or "X"
        """
        if hasattr(self, 'input_state'):
            return "X" if self.input_state == "+" else "Z"
        return "X" if self._input_state == "+" else "Z"
    
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
        
        # Check code-type compatibility
        req = self.required_code_properties
        for i, code in enumerate(codes):
            if "css" in req and not code.is_css:
                raise ValueError(
                    f"{self.__class__.__name__} requires CSS codes, but "
                    f"codes[{i}] ({code.name}) is not CSS."
                )
            if "stabilizer" in req and not code.is_stabilizer:
                raise ValueError(
                    f"{self.__class__.__name__} requires stabilizer codes, but "
                    f"codes[{i}] ({code.name}) is not a stabilizer code."
                )
    
    @property
    def required_code_properties(self) -> Set[str]:
        """Code properties required by this gadget.

        Return a set of strings describing what the gadget needs.
        Recognised values:

        * ``"css"`` — codes must satisfy ``code.is_css``
        * ``"stabilizer"`` — codes must satisfy ``code.is_stabilizer``

        The default is ``{"stabilizer"}`` (works with any
        :class:`StabilizerCode`, CSS or not).  Gadgets that require
        CSS structure (e.g. surgery CNOT with separate X/Z merge
        rounds) should override and return ``{"css"}``.

        Returns
        -------
        Set[str]
        """
        return {"stabilizer"}
    
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
        self._cached_metadata = None
    
    # =========================================================================
    # GENERIC INTERFACE METHODS - for clean experiment abstraction
    # =========================================================================
    # These methods allow the experiment to query gadget behavior generically
    # without type-checking (avoiding anti-patterns like `is_teleportation`).
    
    def get_blocks_to_skip_preparation(self) -> Set[str]:
        """
        Return block names that the experiment should NOT prepare.
        
        The gadget will prepare these blocks in its own specific state
        during emit_next_phase(). Used for teleportation ancilla blocks
        or surgery ancilla blocks.
        
        Default: empty set (experiment prepares all blocks)
        
        Returns
        -------
        Set[str]
            Block names to skip preparation.
        """
        return set()
    
    def get_blocks_to_skip_pre_rounds(self) -> Set[str]:
        """
        Return block names to skip during pre-gadget EC rounds.
        
        For teleportation, ancilla blocks are prepared by the gadget
        and shouldn't have pre-gadget stabilizer rounds.
        
        Default: empty set (EC on all blocks)
        
        Returns
        -------
        Set[str]
            Block names to skip pre-gadget rounds.
        """
        return set()
    
    def get_blocks_to_skip_post_rounds(self) -> Set[str]:
        """
        Return block names to skip during post-gadget EC rounds.
        
        Used for destroyed blocks or blocks that are measured immediately.
        
        Default: empty set (EC on all blocks)
        
        Returns
        -------
        Set[str]
            Block names to skip post-gadget rounds.
        """
        return set()
    
    def get_default_stabilizer_ordering(self) -> Optional[str]:
        """
        Return the default stabilizer ordering for this gadget's rounds.
        
        Some protocols require a specific ordering of stabilizer bases in
        every round to produce correct crossing detectors. For example:
        
        - CZ H-teleportation: Z-first ordering (Z stabilizers before X)
        - CNOT H-teleportation: X-first ordering (X stabilizers before Z)
        
        When ``None`` (default), the scheduler uses its standard heuristic
        based on prep/meas basis and anchor/boundary determinism rules.
        
        Returns
        -------
        Optional[str]
            ``"Z_FIRST"`` or ``"X_FIRST"`` to override middle-round ordering,
            or ``None`` to use the scheduler default.
        """
        return None
    
    def get_destroyed_blocks(self) -> Set[str]:
        """
        Return block names that are destroyed (measured out) during the gadget.
        
        Destroyed blocks should not be measured at the end.
        For teleportation, the data block is measured and destroyed.
        
        Default: empty set (no blocks destroyed)
        
        Returns
        -------
        Set[str]
            Block names that are destroyed.
        """
        return set()
    
    def get_initial_state_for_block(self, block_name: str, requested_state: str) -> str:
        """
        Get the initial state to use for a specific block.
        
        Some gadgets need specific initial states for certain blocks
        (e.g., teleportation ancilla in |+⟩ for CZ).
        
        Default: return the requested state unchanged.
        
        Parameters
        ----------
        block_name : str
            Name of the block.
        requested_state : str
            The state requested by the experiment ("0" or "+").
            
        Returns
        -------
        str
            The actual initial state to use.
        """
        return requested_state
    
    def should_skip_state_preparation(self) -> bool:
        """
        Return True if the gadget handles ALL state preparation internally.
        
        When True, the experiment skips emit_prepare_logical_state entirely.
        Used for teleportation where the gadget manages both data and ancilla
        preparation in a coordinated way.
        
        Default: False (experiment handles preparation)
        
        Returns
        -------
        bool
            True to skip all experiment state preparation.
        """
        return False
    
    def use_auto_detectors(self) -> bool:
        """
        Return True if this gadget should use automatic detector/observable
        emission via flow matching instead of manual config-based emission.

        When True, ``FaultTolerantGadgetExperiment.to_stim()`` will:

        1. Build the circuit normally (with whatever manual detectors the
           config methods produce).
        2. Strip all DETECTOR and OBSERVABLE_INCLUDE instructions.
        3. Call ``discover_detectors()`` and ``discover_observables()`` from
           :mod:`qectostim.experiments.auto_detector_emission`.
        4. Emit the discovered annotations into the circuit.

        This replaces the previous pattern of calling auto-discovery
        *after* ``to_stim()`` in test harnesses.

        Default: False (use manual config-based emission)

        Returns
        -------
        bool
            True to use automatic detector/observable emission.
        """
        return False

    def should_emit_space_like_detectors(self) -> bool:
        """
        Return True if the experiment should emit space-like detectors at the end.
        
        Space-like detectors compare final data measurement to last stabilizer round.
        For teleportation, the Pauli frame from measurements already determines
        the observable, so space-like detectors are not needed.
        
        Default: True (emit space-like detectors)
        
        Returns
        -------
        bool
            True to emit space-like detectors.
        """
        return True
    
    def get_space_like_detector_config(self) -> Dict[str, Optional[str]]:
        """
        Return per-block space-like detector configuration.
        
        This method specifies which stabilizer type to use for space-like
        detectors for each block. Space-like detectors compare final data
        measurements with the last stabilizer round of the specified type.
        
        For transversal gates that transform stabilizers, some blocks may
        need to skip space-like detectors (return None) or use a different
        stabilizer type than the measurement basis.
        
        Default: Returns empty dict, meaning use the measurement basis for all
        blocks (standard behavior).
        
        Returns
        -------
        Dict[str, Optional[str]]
            Mapping from block_name to stabilizer type ("X", "Z") or None to skip.
            Empty dict means use default (measurement basis for all blocks).
            
        Examples
        --------
        For CZ H-teleportation with |+⟩ input:
        - Data block measured in X, but X_D → X_D ⊗ Z_A (transformed)
        - Ancilla block measured in Z, Z_A → Z_A (unchanged)
        
        Returns {"data_block": None, "ancilla_block": "Z"} to:
        - Skip space-like detectors for data block (X stabilizers transformed)
        - Emit Z space-like detectors for ancilla block
        """
        return {}
    
    def get_ancilla_block_names(self) -> Set[str]:
        """
        Return names of ancilla blocks used by this gadget.
        
        Ancilla blocks are auxiliary code blocks used for teleportation,
        surgery, or other multi-block protocols. They typically need
        special preparation and may be measured in a different basis.
        
        Default: empty set (no ancilla blocks)
        
        Returns
        -------
        Set[str]
            Names of ancilla blocks.
        """
        return set()
    
    def is_teleportation_gadget(self) -> bool:
        """
        Return True if this is a teleportation-based gadget.
        
        This is a convenience method for compatibility. Prefer using
        the more specific methods (get_destroyed_blocks, etc.) for
        querying gadget behavior.
        
        Default: False
        
        Returns
        -------
        bool
            True if teleportation-based.
        """
        return False
    
    def requires_three_blocks(self) -> bool:
        """
        Return True if this gadget requires three code blocks.
        
        Three-block gadgets typically involve Bell-state teleportation
        with a shared ancilla pair. Examples:
        - Bell-state based CNOT: data1 + ancilla(Bell) + data2
        - Distillation protocols
        
        Default: False (most gadgets use 1-2 blocks)
        
        Returns
        -------
        bool
            True if three blocks required.
        """
        return False
    
    def requires_parallel_extraction(self) -> bool:
        """
        Return True if this gadget requires parallel syndrome extraction.
        
        Parallel extraction means both blocks are measured together in each round,
        with measurement ordering: [D_Z, A_Z, D_X, A_X] per round.
        
        This is required for teleportation gadgets because:
        1. 3-term crossing detectors need measurements from both blocks in same round
        2. Ground truth builders use this ordering
        3. Temporal detector chains must align across blocks
        
        Default: False (sequential per-block extraction)
        
        Returns
        -------
        bool
            True if parallel extraction required.
        """
        return False
    
    def get_x_stabilizer_mode(self) -> str:
        """
        Return the gate type to use for X stabilizer measurement.
        
        For most gadgets, CZ-based X stabilizer measurement works correctly.
        However, teleportation gadgets require CX (CNOT) for X stabilizers
        to match the ground truth and ensure correct backward error propagation.
        
        The modes are:
        - "cz": Use H-CZ-H circuit (default, symmetric, good for memory experiments)
        - "cx": Use H-CX-H circuit (required for teleportation gadgets)
        
        The difference is in backward Pauli propagation:
        - CZ: X_syndrome → X_syndrome ⊗ Z_data (couples to data Z)
        - CX: X_syndrome → X_syndrome (stays local to syndrome)
        
        For teleportation, CZ would make X anchors non-deterministic because
        the Z_data term propagates through H to become X_data on |0⟩.
        
        Returns
        -------
        str
            "cz" or "cx"
        """
        return "cz"  # Default for most gadgets
    
    def get_observable_config(self) -> ObservableConfig:
        """
        Return configuration for how observables should be constructed.
        
        This is the PRIMARY interface for gadgets to declare their observable
        requirements. The experiment calls this method and handles observable
        emission generically based on the returned config.
        
        By having gadgets return an ObservableConfig, the experiment avoids
        type-specific branching (e.g., `if is_teleportation_gadget()`).
        
        Returns
        -------
        ObservableConfig
            Configuration specifying output blocks, frame corrections,
            and other observable requirements.
            
        Default Implementation
        ----------------------
        Returns a simple config for single-block transversal gates:
        - Output on "data_block" or "block_0"
        - No frame corrections
        """
        # Check for two-qubit transform (provided by subclasses that override
        # get_two_qubit_observable_transform to return non-None)
        transform = self.get_two_qubit_observable_transform()
        if transform is not None:
            return ObservableConfig.transversal_two_qubit(transform)
        
        # Default: single-block transversal gate
        return ObservableConfig.transversal_single_qubit()
    
    def get_two_qubit_observable_transform(self) -> Optional["TwoQubitObservableTransform"]:
        """
        Return the two-qubit observable transform for this gadget.
        
        For two-qubit gates (CNOT, CZ, etc.), this describes how observables
        propagate through the gate. Used for constructing correct logical
        observables after the gate.
        
        Single-qubit gates and teleportation gadgets should return None.
        Two-qubit transversal gates override this to return the appropriate
        transform.
        
        Returns
        -------
        Optional[TwoQubitObservableTransform]
            The transform if this is a two-qubit gate, None otherwise.
        """
        return None  # Default: not a two-qubit gate
    
    def get_preparation_config(self, input_state: str = "0") -> PreparationConfig:
        """
        Return configuration for state preparation across all blocks.
        
        This is the PRIMARY interface for gadgets to declare per-block
        preparation requirements. The experiment calls this method and
        handles preparation generically based on the returned config.
        
        Parameters
        ----------
        input_state : str
            The logical input state ("0", "1", "+", "-").
            
        Returns
        -------
        PreparationConfig
            Configuration specifying per-block initial states and determinism.
            
        Default Implementation
        ----------------------
        Returns config for single-block with experiment-provided state.
        """
        return PreparationConfig.single_block(input_state)
    
    def get_measurement_config(self) -> MeasurementConfig:
        """
        Return configuration for final measurements across all blocks.
        
        This is the PRIMARY interface for gadgets to declare per-block
        measurement bases. The experiment calls this method to determine
        which basis to measure each block.
        
        Returns
        -------
        MeasurementConfig
            Configuration specifying per-block measurement bases.
            
        Default Implementation
        ----------------------
        Returns config for single-block with gadget's measurement basis.
        """
        return MeasurementConfig.single_block(self.get_measurement_basis())
    
    def get_crossing_detector_config(self) -> Optional[CrossingDetectorConfig]:
        """
        Return configuration for crossing detectors around the gate.
        
        Crossing detectors compare stabilizer measurements before and after
        a transversal gate to detect errors during the gate.
        
        Returns
        -------
        Optional[CrossingDetectorConfig]
            Configuration specifying crossing detector formulas.
            Returns None if no crossing detectors needed.
            
        Default Implementation
        ----------------------
        Returns None (simple gates don't need special crossing detectors).
        """
        return None
    
    def get_boundary_detector_config(self) -> BoundaryDetectorConfig:
        """
        Return configuration for boundary (space-like) detectors.
        
        Boundary detectors compare final data measurements to the last
        syndrome round to detect errors in the final measurement.
        
        Returns
        -------
        BoundaryDetectorConfig
            Configuration specifying which boundary detectors to emit.
            
        Default Implementation
        ----------------------
        Returns config for single-block boundary matching gadget's measurement basis.
        """
        return BoundaryDetectorConfig.single_block(self.get_measurement_basis())
    
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
            
            # Get stabilizer positions from Code ABC
            stab_coords = code.stabilizer_coords()
            if stab_coords is not None:
                for local_coord in stab_coords:
                    # Translate to global coordinates
                    global_coord = tuple(
                        local_coord[i] + offset[i] if i < len(local_coord) else offset[i]
                        for i in range(len(offset))
                    )
                    detector_coords[detector_idx] = (global_coord, time)
                    detector_idx += 1
        
        return detector_coords

    def use_hybrid_decoding(self) -> bool:
        """
        Return True if this gadget uses hybrid DEM-based decoding.
        
        Hybrid decoding is used for teleportation gadgets where the
        observable depends on measurement outcomes during the gadget.
        
        Default: False (standard DEM decoding)
        
        Returns
        -------
        bool
            True if hybrid decoding should be used.
        """
        return False
    
    def requires_raw_sampling(self) -> bool:
        """
        Return True if this gadget requires raw sampling for decoding.
        
        Some gadgets need access to raw measurement outcomes for their
        classical processing or observable determination.
        
        Default: False
        """
        return False
    
    def get_output_block_name(self) -> str:
        """
        Return the name of the block that carries the output logical qubit.
        
        For transversal gates, this is typically "block_0" (unchanged).
        For teleportation, this is the ancilla block (data is consumed).
        
        Default: "block_0" or "data_block"
        
        Returns
        -------
        str
            Name of the output block.
        """
        return "block_0"
    
    def get_input_block_name(self) -> str:
        """
        Return the name of the block that carries the input logical qubit.
        
        For transversal gates, this is typically "block_0".
        For teleportation, this is the data block (consumed during gate).
        
        Default: "block_0"
        
        Returns
        -------
        str
            Name of the input/data block.
        """
        return "block_0"
    
    def get_frame_correction_info(self) -> Optional[Dict[str, Any]]:
        """
        Return Pauli frame correction information from mid-gadget measurements.
        
        For teleportation gadgets, the Bell measurement outcomes determine
        Pauli corrections that must be applied to the observable.
        
        Default: None (no frame corrections)
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Frame correction info with keys:
            - 'x_meas_indices': Measurement indices determining X correction
            - 'z_meas_indices': Measurement indices determining Z correction
            - 'output_block': Which block receives the corrections
            Returns None if no corrections needed.
        """
        return None
    
    def get_prep_meas_start(self) -> Optional[int]:
        """
        Return the global measurement index where preparation measurements start.
        
        For teleportation gadgets, this is the index of the first X stabilizer
        measurement from ancilla preparation. These measurements are used for
        Pauli frame correction in the observable.
        
        Default: None (no preparation measurements tracked)
        
        Returns
        -------
        Optional[int]
            Global measurement index of first prep measurement, or None.
        """
        return self._prep_meas_start
    
    def set_prep_meas_start(self, meas_start: int) -> None:
        """
        Set the global measurement index where preparation measurements start.
        
        Called by the phase orchestrator after emitting preparation phase.
        
        Parameters
        ----------
        meas_start : int
            Global measurement index.
        """
        self._prep_meas_start = meas_start
    
    def get_input_block_config(self) -> Dict[str, str]:
        """
        Return input state configuration for each block.
        
        This specifies what state each input block should be in.
        For transversal gates, both blocks get the experiment's requested state.
        For teleportation, the data block gets the input state, ancilla gets
        a specific state (|+⟩ for CZ, |0⟩ for CNOT).
        
        Returns
        -------
        Dict[str, str]
            Mapping from block_name to initial state ("0", "+", etc.)
        """
        return {}  # Default: experiment decides
    
    def get_ancilla_preparation_config(self) -> Optional[Dict[str, Any]]:
        """
        Return configuration for ancilla block preparation.
        
        This is used by gadgets that internally prepare ancilla blocks
        (teleportation, surgery). The config specifies how to prepare
        each ancilla block during the gadget's preparation phase.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Ancilla prep config with keys:
            - 'block_name': Name of the ancilla block
            - 'initial_state': State to prepare ("0", "+", etc.)
            - 'num_rounds': Number of stabilizer rounds for preparation
            Returns None if no ancilla preparation needed.
        """
        return None

