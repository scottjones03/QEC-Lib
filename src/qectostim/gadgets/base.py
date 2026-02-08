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
    Union,
    TYPE_CHECKING,
)

import stim

from qectostim.codes.abstract_code import Code
from qectostim.gadgets.coordinates import CoordND, get_code_dimension
from qectostim.gadgets.layout import GadgetLayout, QubitIndexMap, QubitAllocation

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
    
    @classmethod
    def identity(cls, clear_history: bool = False, skip_first_round: bool = False) -> "StabilizerTransform":
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
        """
        return cls(x_becomes="X", z_becomes="Z", clear_history=clear_history, skip_first_round=skip_first_round)
    
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
class ObservableTerm:
    """
    A single term in an observable formula.
    
    Used to specify multi-block correlation observables like X_L(D) ⊕ Z_L(A).
    
    Attributes
    ----------
    block : str
        Block name ("data_block", "ancilla_block").
    basis : str
        Pauli basis for this term ("X", "Z", or "Y").
    """
    block: str
    basis: str  # "X", "Z", or "Y"


@dataclass
class ObservableConfig:
    """
    Configuration for how observables should be constructed for a gadget.
    
    This provides a DECLARATIVE interface for gadgets to specify their
    observable requirements, allowing the experiment to handle observable
    emission generically without type-specific branching.
    
    The key insight is that different gadgets have different observable structures:
    - Transversal gates: Output on same block(s), may have basis transforms
    - Teleportation: Output on ancilla block, may need frame corrections from measurements
    - Two-qubit gates: Observable may spread across blocks (CNOT Z→Z⊗Z)
    - Surgery: Similar to teleportation, output on merged/split blocks
    
    By having gadgets return an ObservableConfig, the experiment can handle
    all cases uniformly without `isinstance` checks.
    
    Attributes
    ----------
    output_blocks : List[str]
        Which blocks contribute to the output observable.
        For transversal single-qubit gates: ["data_block"] or ["block_0"]
        For teleportation: ["ancilla_block"] or ["block_1"]
        For two-qubit gates: ["block_0", "block_1"] (if observable spreads)
        
    block_bases : Dict[str, str]
        The Pauli basis to use for each output block.
        Keys are block names, values are "X", "Z", or "Y".
        If a block is not in this dict, the experiment's measurement_basis is used.
        Example for CNOT Z observable: {"block_0": "Z", "block_1": "Z"}
        
    frame_correction_blocks : List[str]
        Blocks whose measurements contribute to frame correction.
        For teleportation, this is typically the destroyed data block.
        The experiment will include these measurements in OBSERVABLE_INCLUDE.
        
    frame_correction_basis : str
        The Pauli type of the frame correction ("X", "Z", or "XZ").
        - "Z": Include Z logical measurements for Z correction
        - "X": Include X logical measurements for X correction  
        - "XZ": Include both (for Bell-state teleportation)
        
    requires_raw_sampling : bool
        If True, skip OBSERVABLE_INCLUDE entirely. The gadget needs raw
        measurement sampling with classical post-processing instead.
        This is for gadgets where measurement outcomes are random and
        cannot be expressed as deterministic Stim observables.
        
    two_qubit_transform : Optional[TwoQubitObservableTransform]
        For two-qubit gates, how observables transform across blocks.
        If provided, the experiment uses this to determine which blocks
        contribute to the observable and in which basis.
        
    use_hybrid_decoding : bool
        If True, emit clean observables without frame measurements for
        DEM-based decoding. Frame corrections applied classically after.
    """
    output_blocks: List[str] = field(default_factory=lambda: ["data_block"])
    block_bases: Dict[str, str] = field(default_factory=dict)
    frame_correction_blocks: List[str] = field(default_factory=list)
    frame_correction_basis: str = "Z"
    requires_raw_sampling: bool = False
    two_qubit_transform: Optional[TwoQubitObservableTransform] = None
    use_hybrid_decoding: bool = False
    
    # NEW: Support multi-block correlation observables
    # e.g., X_L(D) ⊕ Z_L(A) for CZ |+⟩ teleportation
    correlation_terms: List[ObservableTerm] = field(default_factory=list)
    
    @classmethod
    def transversal_single_qubit(cls, basis_transform: Optional[Dict[str, str]] = None) -> "ObservableConfig":
        """
        Config for transversal single-qubit gates (H, S, T, X, Y, Z).
        
        Output is on the same data block, optionally with basis transform.
        
        Parameters
        ----------
        basis_transform : Optional[Dict[str, str]]
            How measurement bases transform, e.g. {"X": "Z", "Z": "X"} for H.
            If None, no transformation (identity gate).
        """
        return cls(
            output_blocks=["data_block"],
            block_bases={},  # Use experiment's measurement_basis, possibly transformed
        )
    
    @classmethod
    def transversal_two_qubit(cls, transform: TwoQubitObservableTransform) -> "ObservableConfig":
        """
        Config for transversal two-qubit gates (CNOT, CZ, SWAP).
        
        Observable may spread across blocks based on the gate's transform.
        
        Parameters
        ----------
        transform : TwoQubitObservableTransform
            How observables transform across the two blocks.
        """
        return cls(
            output_blocks=["block_0", "block_1"],  # May use both
            two_qubit_transform=transform,
        )
    
    @classmethod
    def teleportation(
        cls,
        output_block: str = "ancilla_block",
        destroyed_block: str = "data_block",
        frame_basis: str = "Z",
        use_hybrid: bool = False,
        requires_raw: bool = False,
    ) -> "ObservableConfig":
        """
        Config for teleportation gadgets.
        
        Output is on the ancilla block. Frame corrections come from
        the destroyed (measured out) data block.
        
        Parameters
        ----------
        output_block : str
            Block containing the output logical state.
        destroyed_block : str
            Block that was measured out (provides frame correction).
        frame_basis : str
            Which Pauli frame corrections to include ("Z", "X", or "XZ").
        use_hybrid : bool
            Use hybrid decoding mode (DEM + classical frame).
        requires_raw : bool
            Require raw sampling (skip OBSERVABLE_INCLUDE entirely).
        """
        return cls(
            output_blocks=[output_block],
            frame_correction_blocks=[destroyed_block],
            frame_correction_basis=frame_basis,
            requires_raw_sampling=requires_raw,
            use_hybrid_decoding=use_hybrid,
        )
    
    @classmethod
    def bell_teleportation(
        cls,
        output_block: str = "block_2",
        bell_blocks: List[str] = None,
        frame_basis: str = "XZ",
    ) -> "ObservableConfig":
        """
        Config for Bell-state teleportation (3-block protocol).
        
        Output is on ancilla2 (block_2). Frame corrections come from
        the Bell measurement on data + ancilla1.
        
        Parameters
        ----------
        output_block : str
            Block containing the output (typically "block_2").
        bell_blocks : List[str]
            Blocks involved in Bell measurement (typically ["block_0", "block_1"]).
        frame_basis : str
            Frame correction basis ("XZ" for both X and Z corrections).
        """
        if bell_blocks is None:
            bell_blocks = ["block_0", "block_1"]
        return cls(
            output_blocks=[output_block],
            frame_correction_blocks=bell_blocks,
            frame_correction_basis=frame_basis,
        )
    
    @classmethod
    def cz_teleportation_zero(cls) -> "ObservableConfig":
        """
        CZ H-teleportation with |0⟩ input: Observable = X_L(A).
        
        The output is on the ancilla block, measured in X basis.
        No frame correction needed (deterministic at p=0).
        """
        return cls(
            output_blocks=["ancilla_block"],
            correlation_terms=[
                ObservableTerm(block="ancilla_block", basis="X"),
            ],
        )
    
    @classmethod
    def cz_teleportation_plus(cls) -> "ObservableConfig":
        """
        CZ H-teleportation with |+⟩ input: Observable = X_L(D) ⊕ Z_L(A).
        
        Bell correlation: MX on data, MZ on ancilla.
        Both blocks contribute to the observable.
        """
        return cls(
            output_blocks=["data_block", "ancilla_block"],
            correlation_terms=[
                ObservableTerm(block="data_block", basis="X"),
                ObservableTerm(block="ancilla_block", basis="Z"),
            ],
        )
    
    @classmethod
    def cnot_teleportation_zero(cls) -> "ObservableConfig":
        """
        CNOT H-teleportation with |0⟩ input: Observable = Z_L(A).
        
        The output is on the ancilla block, measured in Z basis.
        """
        return cls(
            output_blocks=["ancilla_block"],
            correlation_terms=[
                ObservableTerm(block="ancilla_block", basis="Z"),
            ],
        )
    
    @classmethod
    def cnot_teleportation_plus(cls) -> "ObservableConfig":
        """
        CNOT H-teleportation with |+⟩ input: Observable = X_L(D) ⊕ X_L(A).
        
        Bell correlation: MX on both blocks.
        """
        return cls(
            output_blocks=["data_block", "ancilla_block"],
            correlation_terms=[
                ObservableTerm(block="data_block", basis="X"),
                ObservableTerm(block="ancilla_block", basis="X"),
            ],
        )


@dataclass
class BlockPreparationConfig:
    """
    Configuration for preparing a single code block.
    
    Attributes
    ----------
    initial_state : str
        The logical initial state: "0", "1", "+", "-".
    z_deterministic : bool
        Whether Z stabilizers are deterministic after preparation.
        True for |0⟩ and |1⟩ inputs (Z eigenstate).
    x_deterministic : bool
        Whether X stabilizers are deterministic after preparation.
        True for |+⟩ and |-⟩ inputs (X eigenstate).
    skip_experiment_prep : bool
        If True, the gadget handles this block's preparation itself.
        The experiment should not emit preparation for this block.
    """
    initial_state: str = "0"
    z_deterministic: bool = True
    x_deterministic: bool = False
    skip_experiment_prep: bool = False
    
    @classmethod
    def zero_state(cls, skip_prep: bool = False) -> "BlockPreparationConfig":
        """|0⟩ state: Z deterministic, X random."""
        return cls(initial_state="0", z_deterministic=True, x_deterministic=False, 
                   skip_experiment_prep=skip_prep)
    
    @classmethod
    def plus_state(cls, skip_prep: bool = False) -> "BlockPreparationConfig":
        """|+⟩ state: X deterministic, Z random."""
        return cls(initial_state="+", z_deterministic=False, x_deterministic=True,
                   skip_experiment_prep=skip_prep)


@dataclass
class PreparationConfig:
    """
    Configuration for state preparation across all blocks.
    
    Gadgets return this to declare per-block initial states and determinism.
    The experiment and library modules (preparation.py, StabilizerRoundBuilder)
    consume this config generically.
    
    Block Name Normalization
    ------------------------
    Teleportation gadgets use semantic names ("data_block", "ancilla_block")
    while the experiment's QubitAllocation may use generic names ("block_0", "block_1").
    
    The get_block_config() method handles this normalization automatically:
    - "block_0" → "data_block" (first block is data)
    - "block_1" → "ancilla_block" (second block is ancilla)
    
    Attributes
    ----------
    blocks : Dict[str, BlockPreparationConfig]
        Per-block preparation configuration.
        Keys are block names ("data_block", "ancilla_block", etc.)
    """
    blocks: Dict[str, BlockPreparationConfig] = field(default_factory=dict)
    
    # Class-level block name aliases for normalization
    BLOCK_ALIASES: Dict[str, str] = field(default_factory=lambda: {
        "block_0": "data_block",
        "block_1": "ancilla_block",
        "block_2": "ancilla_block_2",
    }, repr=False, init=False)
    
    def __post_init__(self):
        """Initialize block aliases after dataclass init."""
        # Use class-level aliases
        object.__setattr__(self, 'BLOCK_ALIASES', {
            "block_0": "data_block",
            "block_1": "ancilla_block",
            "block_2": "ancilla_block_2",
        })
    
    def get_block_config(self, block_name: str) -> Optional[BlockPreparationConfig]:
        """
        Get configuration for a block with name normalization.
        
        Handles the mapping between generic names (block_0, block_1) and
        semantic names (data_block, ancilla_block).
        
        Parameters
        ----------
        block_name : str
            The block name (can be generic or semantic).
            
        Returns
        -------
        Optional[BlockPreparationConfig]
            The block configuration, or None if not found.
        """
        # Try direct lookup first
        if block_name in self.blocks:
            return self.blocks[block_name]
        
        # Try alias lookup (block_0 → data_block, etc.)
        alias = self.BLOCK_ALIASES.get(block_name)
        if alias and alias in self.blocks:
            return self.blocks[alias]
        
        # Try reverse alias (data_block → block_0)
        for generic, semantic in self.BLOCK_ALIASES.items():
            if block_name == semantic and generic in self.blocks:
                return self.blocks[generic]
        
        return None
    
    def get_normalized_block_name(self, block_name: str) -> str:
        """
        Normalize a block name to the name used in this config.
        
        Parameters
        ----------
        block_name : str
            The block name to normalize.
            
        Returns
        -------
        str
            The normalized name that exists in self.blocks, or the original
            name if no normalization is possible.
        """
        if block_name in self.blocks:
            return block_name
        
        alias = self.BLOCK_ALIASES.get(block_name)
        if alias and alias in self.blocks:
            return alias
        
        for generic, semantic in self.BLOCK_ALIASES.items():
            if block_name == semantic and generic in self.blocks:
                return generic
        
        return block_name
    
    @classmethod
    def single_block(cls, state: str = "0") -> "PreparationConfig":
        """Config for single-block gadget (transversal gates)."""
        if state in ("0", "1"):
            return cls(blocks={"data_block": BlockPreparationConfig.zero_state()})
        else:  # "+", "-"
            return cls(blocks={"data_block": BlockPreparationConfig.plus_state()})
    
    @classmethod
    def cz_teleportation(cls, input_state: str = "0") -> "PreparationConfig":
        """
        Config for CZ H-teleportation: data=input, ancilla=|+⟩.
        
        For CZ protocol:
        - Data block: input state (|0⟩ has Z deterministic, |+⟩ has X deterministic)
        - Ancilla block: |+⟩ (X deterministic, Z indeterminate)
        
        This enables anchor detectors on the first prep round for:
        - Z_D (|0⟩ input) or X_D (|+⟩ input)
        - X_A (ancilla always |+⟩)
        """
        if input_state in ("0", "1"):
            # |0⟩ input: Z_D deterministic
            data_config = BlockPreparationConfig.zero_state()
        else:
            # |+⟩ input: X_D deterministic
            data_config = BlockPreparationConfig.plus_state()
        
        # Ancilla |+⟩: X_A deterministic
        # Ancilla prep is declared here by the gadget (gadgets/preparation.py
        # principle) and executed by experiment's preparation module at the
        # correct time (before any syndrome rounds).
        ancilla_config = BlockPreparationConfig.plus_state()
        
        return cls(blocks={
            "data_block": data_config,
            "ancilla_block": ancilla_config,
        })
    
    @classmethod
    def cnot_teleportation(cls, input_state: str = "0") -> "PreparationConfig":
        """
        Config for CNOT H-teleportation: data=input, ancilla=|0⟩.
        
        For CNOT protocol:
        - Data block: input state (|0⟩ has Z deterministic, |+⟩ has X deterministic)
        - Ancilla block: |0⟩ (Z deterministic, X indeterminate)
        
        This enables anchor detectors on the first prep round for:
        - Z_D (|0⟩ input) or X_D (|+⟩ input)
        - Z_A (ancilla always |0⟩)
        """
        if input_state in ("0", "1"):
            # |0⟩ input: Z_D deterministic
            data_config = BlockPreparationConfig.zero_state()
        else:
            # |+⟩ input: X_D deterministic
            data_config = BlockPreparationConfig.plus_state()
        
        # Ancilla |0⟩: Z_A deterministic
        # Ancilla prep is declared here by the gadget (gadgets/preparation.py
        # principle) and executed by experiment's preparation module at the
        # correct time (before any syndrome rounds).
        ancilla_config = BlockPreparationConfig.zero_state()
        
        return cls(blocks={
            "data_block": data_config,
            "ancilla_block": ancilla_config,
        })


@dataclass
class MeasurementConfig:
    """
    Configuration for final measurements across all blocks.
    
    Gadgets return this to declare per-block measurement bases.
    The experiment uses this in _emit_final_measurement() to apply
    correct basis rotations before measuring.
    
    Attributes
    ----------
    block_bases : Dict[str, str]
        Per-block measurement basis ("X", "Z", or "Y").
        Keys are block names.
    destroyed_blocks : Set[str]
        Blocks that are destroyed (measured out) during the gadget.
        These blocks are NOT measured at the end.
    """
    block_bases: Dict[str, str] = field(default_factory=dict)
    destroyed_blocks: Set[str] = field(default_factory=set)
    
    @classmethod
    def single_block(cls, basis: str = "Z") -> "MeasurementConfig":
        """Config for single-block gadget."""
        return cls(block_bases={"data_block": basis})
    
    @classmethod
    def cz_teleportation(cls, input_state: str = "0") -> "MeasurementConfig":
        """
        Config for CZ H-teleportation measurements.
        
        Data is always measured in X (Bell-like measurement).
        Ancilla basis depends on input state:
        - |0⟩ input: ancilla measured X (output is X_L)
        - |+⟩ input: ancilla measured Z (output is Z_L after H)
        
        NOTE: data_block is NOT destroyed - it gets EC rounds after CZ
        and participates in the final measurement. The logical information
        is transferred to ancilla, but the data block is still measured
        (in X basis) to complete the Bell measurement.
        """
        ancilla_basis = "X" if input_state in ("0", "1") else "Z"
        return cls(
            block_bases={"data_block": "X", "ancilla_block": ancilla_basis},
            # No destroyed_blocks - both blocks are measured
        )
    
    @classmethod
    def cnot_teleportation(cls, input_state: str = "0") -> "MeasurementConfig":
        """
        Config for CNOT H-teleportation measurements.
        
        Data is always measured in X.
        Ancilla basis depends on input state:
        - |0⟩ input: ancilla measured Z
        - |+⟩ input: ancilla measured X
        
        NOTE: data_block is NOT destroyed - it gets EC rounds after CNOT
        and participates in the final measurement.
        """
        ancilla_basis = "Z" if input_state in ("0", "1") else "X"
        return cls(
            block_bases={"data_block": "X", "ancilla_block": ancilla_basis},
            # No destroyed_blocks - both blocks are measured
        )


@dataclass
class CrossingDetectorTerm:
    """
    A single term in a crossing detector formula.
    
    Attributes
    ----------
    block : str
        Block name ("data_block", "ancilla_block").
    stabilizer_type : str
        "X" or "Z" stabilizer.
    timing : str
        "pre" (before gate) or "post" (after gate).
    """
    block: str
    stabilizer_type: str  # "X" or "Z"
    timing: str  # "pre" or "post"


@dataclass
class CrossingDetectorFormula:
    """
    Formula for a crossing detector.
    
    A crossing detector compares stabilizer measurements before and after
    a transversal gate. The formula specifies which measurements to XOR.
    
    Example for CZ X_D crossing (3-term):
        terms = [
            CrossingDetectorTerm("data_block", "X", "pre"),
            CrossingDetectorTerm("data_block", "X", "post"),
            CrossingDetectorTerm("ancilla_block", "Z", "post"),
        ]
    
    Attributes
    ----------
    name : str
        Human-readable name (e.g., "X_D", "Z_A").
    terms : List[CrossingDetectorTerm]
        Terms to XOR for this detector.
    num_stabilizers : Optional[int]
        Number of stabilizers of this type (if None, infer from code).
    """
    name: str
    terms: List[CrossingDetectorTerm]
    num_stabilizers: Optional[int] = None


@dataclass
class CrossingDetectorConfig:
    """
    Configuration for crossing detectors across a transversal gate.
    
    Gadgets return this to declare the crossing detector formulas.
    The detector_tracking module emits detectors from this config.
    
    Attributes
    ----------
    formulas : List[CrossingDetectorFormula]
        List of crossing detector formulas to emit.
    """
    formulas: List[CrossingDetectorFormula] = field(default_factory=list)
    
    @classmethod
    def identity(cls) -> "CrossingDetectorConfig":
        """No crossing (identity gate) - simple 2-term temporal detectors."""
        return cls(formulas=[
            CrossingDetectorFormula("Z_D", [
                CrossingDetectorTerm("data_block", "Z", "pre"),
                CrossingDetectorTerm("data_block", "Z", "post"),
            ]),
            CrossingDetectorFormula("X_D", [
                CrossingDetectorTerm("data_block", "X", "pre"),
                CrossingDetectorTerm("data_block", "X", "post"),
            ]),
        ])
    
    @classmethod
    def cz_teleportation(cls) -> "CrossingDetectorConfig":
        """
        Crossing detectors for CZ H-teleportation.
        
        CZ preserves Z stabilizers but mixes X with Z:
        - Z_D, Z_A: 2-term (unchanged through CZ)
        - X_D: 3-term (pre_X_D ⊕ post_X_D ⊕ post_Z_A)
        - X_A: 3-term (pre_X_A ⊕ post_Z_D ⊕ post_X_A)
        """
        return cls(formulas=[
            # Z stabilizers: 2-term (unchanged through CZ)
            CrossingDetectorFormula("Z_D", [
                CrossingDetectorTerm("data_block", "Z", "pre"),
                CrossingDetectorTerm("data_block", "Z", "post"),
            ]),
            CrossingDetectorFormula("Z_A", [
                CrossingDetectorTerm("ancilla_block", "Z", "pre"),
                CrossingDetectorTerm("ancilla_block", "Z", "post"),
            ]),
            # X stabilizers: 3-term (X picks up Z from other block)
            CrossingDetectorFormula("X_D", [
                CrossingDetectorTerm("data_block", "X", "pre"),
                CrossingDetectorTerm("data_block", "X", "post"),
                CrossingDetectorTerm("ancilla_block", "Z", "post"),
            ]),
            CrossingDetectorFormula("X_A", [
                CrossingDetectorTerm("ancilla_block", "X", "pre"),
                CrossingDetectorTerm("data_block", "Z", "post"),
                CrossingDetectorTerm("ancilla_block", "X", "post"),
            ]),
        ])
    
    @classmethod
    def cnot_teleportation(cls, input_state: str = "0") -> "CrossingDetectorConfig":
        """
        Crossing detectors for CNOT H-teleportation.
        
        CNOT (data=control, ancilla=target) Heisenberg transformation:
        - Z_D → Z_D              (control Z unchanged)
        - Z_A → Z_D ⊗ Z_A        (target Z picks up control Z)
        - X_D → X_D ⊗ X_A        (control X picks up target X)
        - X_A → X_A              (target X unchanged)
        
        Crossing formulas:
        - Z_D: ALWAYS 2-term (Z_D unchanged through CNOT)
        - Z_A: 2-term for |0⟩ (Z_D deterministic = +1), 3-term for |+⟩
        - X_D: ALWAYS 3-term (X_D_pre = X_D_post ⊕ X_A_post)
        - X_A: ALWAYS 2-term (X_A unchanged)
        """
        formulas = []
        
        # Z_D crossing: ALWAYS 2-term (Z_D is unchanged through CNOT)
        formulas.append(CrossingDetectorFormula("Z_D", [
            CrossingDetectorTerm("data_block", "Z", "pre"),
            CrossingDetectorTerm("data_block", "Z", "post"),
        ]))
        
        # Z_A crossing: Z_A → Z_D ⊗ Z_A
        if input_state in ("0", "1"):
            # |0⟩: Z_D deterministic (+1), so Z_A crossing is 2-term
            formulas.append(CrossingDetectorFormula("Z_A", [
                CrossingDetectorTerm("ancilla_block", "Z", "pre"),
                CrossingDetectorTerm("ancilla_block", "Z", "post"),
            ]))
        else:
            # |+⟩: Z_D random, need 3-term: Z_A_pre ⊕ Z_D_post ⊕ Z_A_post = 0
            formulas.append(CrossingDetectorFormula("Z_A", [
                CrossingDetectorTerm("ancilla_block", "Z", "pre"),
                CrossingDetectorTerm("data_block", "Z", "post"),
                CrossingDetectorTerm("ancilla_block", "Z", "post"),
            ]))
        
        # X_D crossing: ALWAYS 3-term (X_D → X_D ⊗ X_A)
        formulas.append(CrossingDetectorFormula("X_D", [
            CrossingDetectorTerm("data_block", "X", "pre"),
            CrossingDetectorTerm("data_block", "X", "post"),
            CrossingDetectorTerm("ancilla_block", "X", "post"),
        ]))
        
        # X_A crossing: ALWAYS 2-term (X_A unchanged)
        formulas.append(CrossingDetectorFormula("X_A", [
            CrossingDetectorTerm("ancilla_block", "X", "pre"),
            CrossingDetectorTerm("ancilla_block", "X", "post"),
        ]))
        
        return cls(formulas=formulas)


@dataclass
class BoundaryDetectorConfig:
    """
    Configuration for boundary (space-like) detectors.
    
    Boundary detectors compare final data measurements to last syndrome round.
    
    Attributes
    ----------
    block_configs : Dict[str, Dict[str, bool]]
        Per-block configuration: {block_name: {"X": emit_x_boundary, "Z": emit_z_boundary}}
        Only emit boundary detector if the measurement basis is compatible
        (e.g., MX data can have X_D boundary but NOT Z_D boundary).
    """
    block_configs: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    
    @classmethod
    def single_block(cls, measurement_basis: str = "Z") -> "BoundaryDetectorConfig":
        """Config for single-block gadget."""
        # Only emit boundary for compatible basis
        return cls(block_configs={
            "data_block": {
                "X": measurement_basis == "X",
                "Z": measurement_basis == "Z",
            }
        })
    
    @classmethod
    def cz_teleportation(cls, input_state: str = "0") -> "BoundaryDetectorConfig":
        """
        Boundary detectors for CZ H-teleportation.
        
        Data measured MX:
        - X_D boundary: NO — CZ transforms X_D → X_D ⊗ Z_A, so the
          post-CZ X_D syndrome is X_D ⊗ Z_A. A simple 2-term boundary
          comparing last X_D syndrome to MX(data) = X_D would leave
          the Z_A contribution unmatched.
        - Z_D boundary: NO — MX anticommutes with Z
        
        Ancilla:
        - |0⟩ input: MX → X_A boundary YES (X_A → Z_D ⊗ X_A through CZ,
          but the post-CZ X_A syndrome tracks the transformed stabilizer,
          and MX on ancilla matches X_A)
        - |+⟩ input: MZ → Z_A boundary YES (Z_A unchanged through CZ)
        """
        ancilla_basis = "X" if input_state in ("0", "1") else "Z"
        return cls(block_configs={
            "data_block": {"X": False, "Z": False},  # No data boundary for CZ
            "ancilla_block": {
                "X": ancilla_basis == "X",
                "Z": ancilla_basis == "Z",
            },
        })
    
    @classmethod
    def cnot_teleportation(cls, input_state: str = "0") -> "BoundaryDetectorConfig":
        """
        Boundary detectors for CNOT H-teleportation.
        
        Data measured MX → X_D boundary YES, Z_D boundary NO
        Ancilla:
        - |0⟩ input: MZ → Z_A boundary YES
        - |+⟩ input: MX → X_A boundary YES
        """
        ancilla_basis = "Z" if input_state in ("0", "1") else "X"
        return cls(block_configs={
            "data_block": {"X": True, "Z": False},  # MX data
            "ancilla_block": {
                "X": ancilla_basis == "X",
                "Z": ancilla_basis == "Z",
            },
        })


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
        
        Uses Code ABC transversal_gates() method directly.
        
        Args:
            code: The code to check
            gate_name: The gate name to check
            
        Returns:
            True if gate is supported transversally
        """
        supported = code.transversal_gates()
        return gate_name in supported



class TeleportationGadgetMixin:
    """
    Mixin for gadgets that use teleportation protocols.
    
    Provides helper methods and default implementations of the Gadget
    generic interface methods for teleportation-based logical gate
    implementations using Bell pairs and measurements.
    
    Subclasses should set:
        self._data_block_name: str = "data_block"
        self._ancilla_block_name: str = "ancilla_block"
        self._ancilla_initial_state: str = "+" or "0"
    """
    
    # Subclasses should set these
    _data_block_name: str = "data_block"
    _ancilla_block_name: str = "ancilla_block"
    _ancilla_initial_state: str = "+"  # Override in CNOT gadget to "0"
    input_state: str = "0"  # Set by subclass __init__
    _use_hybrid_decoding: bool = False  # Set by subclass __init__
    
    # =========================================================================
    # Generic interface implementations for teleportation
    # =========================================================================
    
    def is_teleportation_gadget(self) -> bool:
        """Return True - this is a teleportation gadget."""
        return True
    
    def get_input_block_name(self) -> str:
        """Return data block name (consumed during teleportation)."""
        return self._data_block_name
    
    def get_output_block_name(self) -> str:
        """Return ancilla block name (carries output after teleportation)."""
        return self._ancilla_block_name
    
    def get_x_stabilizer_mode(self) -> str:
        """
        Return 'cx' for teleportation gadgets.
        
        Teleportation requires CX (CNOT) for X stabilizer measurement to match
        the ground truth builder and ensure X anchor detectors are deterministic.
        
        With CZ, backward Pauli trace goes: X_syndrome → X_syndrome ⊗ Z_data,
        and Z_data through H on data becomes X_data on |0⟩ (non-deterministic).
        
        With CX, backward trace goes: X_syndrome → X_syndrome (no data coupling),
        so the X anchor traces back cleanly to syndrome |0⟩ (deterministic).
        """
        return "cx"
    
    def requires_parallel_extraction(self) -> bool:
        """
        Teleportation gadgets require parallel syndrome extraction.
        
        This ensures:
        1. Both blocks measured together per round: [D_Z, A_Z, D_X, A_X]
        2. 3-term crossing detectors can reference both blocks in same round
        3. Matches ground truth CZHTeleportationBuilder/CNOTHTeleportationBuilder
        """
        return True
    
    def get_blocks_to_skip_preparation(self) -> Set[str]:
        """Return ancilla blocks - gadget prepares them."""
        return {self._ancilla_block_name}
    
    def get_blocks_to_skip_pre_rounds(self) -> Set[str]:
        """
        Return blocks to skip in pre-gadget EC rounds.
        
        For teleportation with crossing detectors, we MUST measure BOTH
        blocks before the gate so that crossing detectors can reference
        both pre-gate X and Z measurements.
        
        The ground truth builder (CZHTeleportationBuilder) measures BOTH
        blocks in pre-CZ rounds. We match that behavior by returning an
        empty set.
        """
        return set()  # Measure BOTH blocks before the gate
    
    def get_blocks_to_skip_post_rounds(self) -> Set[str]:
        """
        Return blocks to skip in post-gadget EC rounds.
        
        The data block is destroyed (measured MX) during teleportation,
        but we need to run post-gadget rounds on both blocks BEFORE the
        destructive measurement to get crossing detector coverage.
        
        IMPORTANT: The data block's post-CZ rounds provide X_D(post)
        measurements needed for X_D crossing detectors.
        """
        return set()  # Measure BOTH blocks after the gate (before final measurement)
    
    def get_destroyed_blocks(self) -> Set[str]:
        """Return data block - measured and destroyed during teleportation."""
        return {self._data_block_name}
    
    def get_ancilla_block_names(self) -> Set[str]:
        """Return ancilla block name."""
        return {self._ancilla_block_name}
    
    def get_initial_state_for_block(self, block_name: str, requested_state: str) -> str:
        """
        Get initial state for a block.
        
        For data block: use requested state
        For ancilla block: use _ancilla_initial_state (|+⟩ for CZ, |0⟩ for CNOT)
        """
        if block_name == self._ancilla_block_name:
            return self._ancilla_initial_state
        return requested_state
    
    def should_skip_state_preparation(self) -> bool:
        """
        Return False - experiment handles preparation using PreparationConfig.
        
        The gadget declares preparation requirements via get_preparation_config().
        The experiment handles initial state prep (R, H) based on that config.
        """
        return False
    
    def should_emit_space_like_detectors(self) -> bool:
        """
        Return True - space-like detectors provide additional error detection.
        
        Teleportation uses space-like detectors for boundary detection when
        the measurement basis matches the stabilizer type (e.g., Z_A boundary
        when ancilla is measured in Z basis for |+⟩ input CZ).
        """
        return True
    
    def get_observable_config(self) -> ObservableConfig:
        """
        Return observable configuration for teleportation gadgets.
        
        Uses the specific factory methods for each gate type and input state:
        - CZ |0⟩: X_L(A)
        - CZ |+⟩: X_L(D) ⊕ Z_L(A)
        - CNOT |0⟩: Z_L(A)
        - CNOT |+⟩: X_L(D) ⊕ X_L(A)
        
        Returns
        -------
        ObservableConfig
            Teleportation-specific observable configuration with correlation_terms.
        """
        # Use gadget's explicit input_state, not inference from measurement_basis
        input_state = self.input_state
        
        # Detect gadget type from ancilla initial state
        ancilla_state = self._ancilla_initial_state
        
        # Check for hybrid decoding mode
        use_hybrid = self._use_hybrid_decoding
        
        # Check for raw sampling requirement
        requires_raw = self.requires_raw_sampling()
        
        if ancilla_state == '+':
            # CZ teleportation
            if input_state == "0":
                config = ObservableConfig.cz_teleportation_zero()
            else:
                config = ObservableConfig.cz_teleportation_plus()
        else:
            # CNOT teleportation
            if input_state == "0":
                config = ObservableConfig.cnot_teleportation_zero()
            else:
                config = ObservableConfig.cnot_teleportation_plus()
        
        # Update config with mode flags
        config.use_hybrid_decoding = use_hybrid
        config.requires_raw_sampling = requires_raw
        
        return config
    
    def get_preparation_config(self, input_state: str = "0") -> PreparationConfig:
        """
        Return preparation configuration for teleportation gadgets.
        
        Teleportation gadgets handle preparation internally with proper
        detector placement. They prepare:
        - Data block: input state (|0⟩ or |+⟩)
        - Ancilla block: gadget-specific state (|+⟩ for CZ, |0⟩ for CNOT)
        
        Parameters
        ----------
        input_state : str
            The logical input state from experiment (ignored for teleportation -
            we use self.input_state directly since the gadget knows its input).
            
        Returns
        -------
        PreparationConfig
            Teleportation-specific preparation configuration.
        """
        # Use gadget's explicit input_state, not inference from experiment
        # The gadget knows what input state it was constructed with
        actual_input_state = self.input_state
        
        # Detect gadget type from ancilla initial state
        ancilla_state = self._ancilla_initial_state
        
        if ancilla_state == '+':
            # CZ teleportation: ancilla |+⟩
            return PreparationConfig.cz_teleportation(actual_input_state)
        else:
            # CNOT teleportation: ancilla |0⟩
            return PreparationConfig.cnot_teleportation(actual_input_state)
    
    def get_measurement_config(self) -> MeasurementConfig:
        """
        Return measurement configuration for teleportation gadgets.
        
        Teleportation gadgets have block-dependent measurements:
        - Data block: always MX (destroyed in Bell-like measurement)
        - Ancilla block: depends on input state and gadget type
        
        Returns
        -------
        MeasurementConfig
            Teleportation-specific measurement configuration.
        """
        # Use gadget's explicit input_state, not inference from measurement_basis
        # The gadget knows what input state it was constructed with
        input_state = self.input_state
        
        # Detect gadget type from ancilla initial state
        ancilla_state = self._ancilla_initial_state
        
        if ancilla_state == '+':
            # CZ teleportation
            return MeasurementConfig.cz_teleportation(input_state)
        else:
            # CNOT teleportation
            return MeasurementConfig.cnot_teleportation(input_state)
    
    def get_crossing_detector_config(self) -> Optional[CrossingDetectorConfig]:
        """
        Return crossing detector configuration for teleportation gadgets.
        
        Teleportation gadgets have specific crossing detector requirements
        based on how the entangling gate (CZ or CNOT) transforms stabilizers.
        
        Returns
        -------
        CrossingDetectorConfig
            Configuration for crossing detectors around the gate.
        """
        input_state = self.input_state
        
        # Detect gadget type from ancilla initial state
        ancilla_state = self._ancilla_initial_state
        
        if ancilla_state == '+':
            # CZ teleportation - same crossing formulas for both input states
            return CrossingDetectorConfig.cz_teleportation()
        else:
            # CNOT teleportation - Z_D formula depends on input state
            return CrossingDetectorConfig.cnot_teleportation(input_state)
    
    def get_boundary_detector_config(self) -> BoundaryDetectorConfig:
        """
        Return boundary detector configuration for teleportation gadgets.
        
        Teleportation gadgets need specific boundary detectors based on
        which blocks are measured in which basis.
        
        Returns
        -------
        BoundaryDetectorConfig
            Teleportation-specific boundary detector configuration.
        """
        # Use gadget's explicit input_state, not inference from measurement_basis
        input_state = self.input_state
        
        # Detect gadget type from ancilla initial state
        ancilla_state = self._ancilla_initial_state
        
        if ancilla_state == '+':
            # CZ teleportation
            return BoundaryDetectorConfig.cz_teleportation(input_state)
        else:
            # CNOT teleportation
            return BoundaryDetectorConfig.cnot_teleportation(input_state)
    
    # =========================================================================
    # Teleportation-specific helper methods
    # =========================================================================
    
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

