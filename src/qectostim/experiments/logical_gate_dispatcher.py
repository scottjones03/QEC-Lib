# src/qectostim/experiments/logical_gate_dispatcher.py
"""
Logical Gate Dispatcher for Hierarchical Concatenated Code Experiments.

This module provides LogicalGateDispatcher, which selects and emits appropriate
logical gate implementations for use in hierarchical concatenated codes.

Integration with HierarchicalV6Decoder
======================================
The dispatcher emits raw gate operations only - no DETECTOR instructions or
transform tracking. The HierarchicalV6Decoder implements a 4-phase pipeline:

  Phase 1: Inner Error Inference
    - Decode inner syndromes per segment (temporal or majority_vote mode)
    - Output: inner_z_correction[seg_id], inner_x_correction[seg_id]
    
  Phase 2: Correct Outer Syndrome
    - Apply ancilla inner corrections to outer syndrome measurements
    - X stabilizers: use inner_x_correction (Z anc detects X errors)
    - Z stabilizers: use inner_z_correction (X anc detects Z errors)
    
  Phase 3: Correct Data Logicals  
    - Apply ALL inner corrections to final data measurements
    - Cumulative XOR across all EC segments per data block
    
  Phase 4: Outer Temporal DEM Decode
    - Build spacetime DEM with consecutive round comparison
    - Decode with PyMatching for final logical prediction

Gate Implementations
====================

1. **Transversal gates** (preferred when available):
   - Apply physical gates qubit-by-qubit across blocks
   - Simplest and most fault-tolerant
   - Available for CSS codes with transversal CNOT (e.g., Steane code)
   
2. **Lattice surgery** (for topological codes):
   - Merge/split operations on code boundaries
   - Requires code geometry support
   - Standard for surface codes
   
3. **Teleportation-based gates** (universal fallback):
   - Uses ancilla blocks and measurement-based protocols
   - Works for any code but requires more resources
   - Necessary when transversal gates don't exist

The dispatcher provides a uniform interface for emitting logical gates,
automatically selecting the appropriate implementation based on:
- Code properties (is_css, has_transversal_cnot, etc.)
- User-specified preferences (gate_method parameter)
- Availability of implementations
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import stim

if TYPE_CHECKING:
    from qectostim.codes.abstract_code import Code
    from qectostim.codes.abstract_css import CSSCode
    from qectostim.experiments.logical_block_manager_v2 import LogicalBlockManager, BlockInfo

from qectostim.experiments.stabilizer_rounds import (
    StabilizerBasis,
    get_logical_support,
)
from qectostim.gadgets.layout import BlockAllocation, QubitAllocation
from qectostim.gadgets.transversal import (
    TransversalCNOT,
    TransversalCZ,
    TransversalHadamard,
    TransversalX,
    TransversalZ,
    TransversalGate,
)
from qectostim.gadgets.css_surgery import SurgeryCNOT
from qectostim.gadgets.teleportation import (
    TeleportedGate,
    TeleportedHadamard,
)
from qectostim.gadgets.base import StabilizerTransform, Gadget


class GateMethod(Enum):
    """
    Available methods for implementing logical gates.
    
    AUTO: Automatically select best available method
    TRANSVERSAL: Use transversal gates (qubit-by-qubit)
    SURGERY: Use lattice surgery (merge/split operations)
    TELEPORTATION: Use measurement-based teleportation
    """
    AUTO = auto()
    TRANSVERSAL = auto()
    SURGERY = auto()
    TELEPORTATION = auto()


class GateType(Enum):
    """Types of logical gates."""
    CNOT = "CNOT"
    CZ = "CZ"
    HADAMARD = "H"
    X = "X"
    Z = "Z"


@dataclass
class GateEmissionResult:
    """
    Result of emitting a logical gate.
    
    Attributes
    ----------
    gate_type : GateType
        The type of gate emitted.
    method_used : GateMethod
        The method used to implement the gate.
    requires_correction : bool
        Whether the gate requires Pauli frame correction.
    correction_info : Dict[str, Any]
        Information needed for correction (measurement outcomes, etc.).
    stabilizer_transform : StabilizerTransform
        How stabilizers are transformed by this gate.
    """
    gate_type: GateType
    method_used: GateMethod
    requires_correction: bool = False
    correction_info: Dict[str, Any] = field(default_factory=dict)
    stabilizer_transform: Optional[StabilizerTransform] = None


def _check_transversal_cnot_support(code: "Code") -> bool:
    """
    Check if a code supports transversal CNOT.
    
    For CSS codes, CNOT is transversal when:
    1. The code is self-dual CSS (hx and hz have compatible structure)
    2. Or explicitly marked with has_transversal_cnot metadata
    
    Parameters
    ----------
    code : Code
        Code to check.
        
    Returns
    -------
    bool
        True if transversal CNOT is supported.
    """
    # Check explicit metadata
    if hasattr(code, 'has_transversal_cnot') and code.has_transversal_cnot:
        return True
    
    if hasattr(code, '_metadata'):
        if code._metadata.get('has_transversal_cnot', False):
            return True
    
    # Check for known code families with transversal CNOT
    code_name = getattr(code, 'name', '').lower()
    known_transversal_cnot = [
        'steane', '7_qubit', 'hamming', 'golay',
        'repetition', 'toric', 'surface',  # Surface/toric have transversal CNOT
    ]
    
    for name in known_transversal_cnot:
        if name in code_name:
            return True
    
    # CSS codes generally have transversal CNOT between identical copies
    if hasattr(code, 'is_css') and code.is_css:
        return True
    
    return False


def _check_surgery_support(code: "Code") -> bool:
    """
    Check if a code supports lattice surgery.
    
    Surgery requires:
    1. Code is CSS
    2. Code has geometric structure (coordinates)
    3. Boundary operators can be identified
    
    Parameters
    ----------
    code : Code
        Code to check.
        
    Returns
    -------
    bool
        True if lattice surgery is supported.
    """
    # Must be CSS
    if not (hasattr(code, 'is_css') and code.is_css):
        return False
    
    # Check for explicit metadata
    if hasattr(code, '_metadata'):
        if code._metadata.get('supports_surgery', False):
            return True
    
    # Check for known topological codes
    code_name = getattr(code, 'name', '').lower()
    known_surgery_codes = ['surface', 'toric', 'color', 'rotated']
    
    for name in known_surgery_codes:
        if name in code_name:
            return True
    
    # Check for coordinate support (needed for surgery)
    has_coords = (
        hasattr(code, 'qubit_coords') and code.qubit_coords is not None
    ) or (
        hasattr(code, '_metadata') and 
        code._metadata.get('qubit_coords') is not None
    )
    
    return has_coords


class LogicalGateDispatcher:
    """
    Dispatches logical gate implementations for hierarchical concatenated codes.
    
    Provides a uniform interface for emitting logical gates between inner code
    blocks. Automatically selects the appropriate implementation based on code
    properties and user preferences.
    
    Segment-Based Architecture for HierarchicalV6Decoder
    =====================================================
    Emits raw gate operations only - no DETECTOR instructions or transform
    tracking. Gates are emitted within outer stabilizer gadgets, and the
    HierarchicalV6Decoder extracts syndrome information from segment metadata:
    
    - Each outer stabilizer gadget contains `ancilla_ec_segment_ids`
      identifying which inner EC segments protect the ancilla
    - Data block corrections accumulate via `data_block_to_all_ec_segments`
    - Consecutive round comparison is used for detectors: D[r,s] = S[r+1,s] XOR S[r,s]
    
    Supported Methods:
    - Transversal gates (preferred for most CSS codes)
    - Lattice surgery (for topological codes)  
    - Teleportation-based gates (universal fallback)
    
    Parameters
    ----------
    inner_code : Code
        The code used for inner blocks.
    default_method : GateMethod
        Default method for gate implementation.
    fallback_to_surgery : bool
        If transversal not available, try surgery before teleportation.
    inner_rounds_per_gate : int
        Number of inner EC rounds after each gate (0 = none).
        
    Examples
    --------
    >>> from qectostim.codes.small import SteaneCode
    >>> from qectostim.experiments.logical_gate_dispatcher import LogicalGateDispatcher
    >>>
    >>> code = SteaneCode()
    >>> dispatcher = LogicalGateDispatcher(code)
    >>> 
    >>> # Check available methods
    >>> print(dispatcher.available_cnot_methods)  # [GateMethod.TRANSVERSAL, ...]
    >>>
    >>> # Emit a logical CNOT
    >>> circuit = stim.Circuit()
    >>> result = dispatcher.emit_logical_cnot(
    ...     circuit, control_alloc, target_alloc
    ... )
    """
    
    def __init__(
        self,
        inner_code: "Code",
        default_method: GateMethod = GateMethod.AUTO,
        fallback_to_surgery: bool = True,
        inner_rounds_per_gate: int = 0,
    ):
        self._inner_code = inner_code
        self._default_method = default_method
        self._fallback_to_surgery = fallback_to_surgery
        self._inner_rounds_per_gate = inner_rounds_per_gate
        
        # Analyze code capabilities
        self._has_transversal_cnot = _check_transversal_cnot_support(inner_code)
        self._has_surgery = _check_surgery_support(inner_code)
        
        # Build available methods list
        self._available_cnot_methods: List[GateMethod] = []
        if self._has_transversal_cnot:
            self._available_cnot_methods.append(GateMethod.TRANSVERSAL)
        if self._has_surgery:
            self._available_cnot_methods.append(GateMethod.SURGERY)
        # Teleportation is always available as fallback
        self._available_cnot_methods.append(GateMethod.TELEPORTATION)
        
        # Cache gadget instances - use existing library gadgets
        self._transversal_cnot: Optional[TransversalCNOT] = None
        self._transversal_cz: Optional[TransversalCZ] = None
        self._transversal_h: Optional[TransversalHadamard] = None
        self._transversal_x: Optional[TransversalX] = None
        self._transversal_z: Optional[TransversalZ] = None
        self._surgery_cnot: Optional[SurgeryCNOT] = None
        # Note: TeleportedCNOT not yet implemented in teleportation.py
        # Full teleportation-based CNOT would be added there
    
    @property
    def inner_code(self) -> "Code":
        """The inner code."""
        return self._inner_code
    
    @property
    def available_cnot_methods(self) -> List[GateMethod]:
        """List of available methods for CNOT implementation."""
        return list(self._available_cnot_methods)
    
    @property
    def has_transversal_cnot(self) -> bool:
        """Whether transversal CNOT is available."""
        return self._has_transversal_cnot
    
    @property
    def inner_rounds_per_gate(self) -> int:
        """Number of inner EC rounds to run after each logical gate."""
        return self._inner_rounds_per_gate
    
    def supports_transversal_cnot(self) -> bool:
        """
        Check if inner code supports transversal CNOT.
        
        Returns
        -------
        bool
            True if transversal CNOT is available.
        """
        return self._has_transversal_cnot
    
    def supports_transversal_cz(self) -> bool:
        """
        Check if inner code supports transversal CZ.
        
        For CSS codes, CZ is transversal when CNOT is transversal.
        
        Returns
        -------
        bool
            True if transversal CZ is available.
        """
        # For CSS codes, CZ availability typically matches CNOT
        return self._has_transversal_cnot
    
    def set_default_method(self, method: GateMethod) -> None:
        """
        Change the default gate method.
        
        Parameters
        ----------
        method : GateMethod
            New default method.
        """
        self._default_method = method
    
    def set_inner_rounds_per_gate(self, rounds: int) -> None:
        """
        Change the number of inner EC rounds per gate.
        
        Parameters
        ----------
        rounds : int
            Number of inner rounds to run after each logical gate.
        """
        self._inner_rounds_per_gate = rounds
    
    @property
    def has_surgery(self) -> bool:
        """Whether lattice surgery is available."""
        return self._has_surgery
    
    def _select_method(
        self,
        gate_type: GateType,
        method: GateMethod = GateMethod.AUTO,
    ) -> GateMethod:
        """
        Select the best available method for a gate.
        
        Parameters
        ----------
        gate_type : GateType
            Type of gate to implement.
        method : GateMethod
            Requested method (AUTO for automatic selection).
            
        Returns
        -------
        GateMethod
            Selected implementation method.
            
        Raises
        ------
        ValueError
            If requested method is not available.
        """
        if method == GateMethod.AUTO:
            # Automatic selection: prefer transversal > surgery > teleportation
            if gate_type in (GateType.CNOT, GateType.CZ):
                if self._has_transversal_cnot:
                    return GateMethod.TRANSVERSAL
                elif self._has_surgery and self._fallback_to_surgery:
                    return GateMethod.SURGERY
                else:
                    return GateMethod.TELEPORTATION
            else:
                # Single-qubit gates: transversal is always available for CSS
                return GateMethod.TRANSVERSAL
        else:
            # Validate requested method
            if method == GateMethod.TRANSVERSAL and not self._has_transversal_cnot:
                raise ValueError(
                    f"Transversal {gate_type.value} not available for {self._inner_code}"
                )
            if method == GateMethod.SURGERY and not self._has_surgery:
                raise ValueError(
                    f"Surgery not available for {self._inner_code}"
                )
            return method
    
    # =========================================================================
    # Logical CNOT
    # =========================================================================
    
    def emit_logical_cnot(
        self,
        circuit: stim.Circuit,
        control_alloc: BlockAllocation,
        target_alloc: BlockAllocation,
        method: GateMethod = GateMethod.AUTO,
    ) -> GateEmissionResult:
        """
        Emit a logical CNOT between two inner code blocks.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        control_alloc : BlockAllocation
            Allocation for control block.
        target_alloc : BlockAllocation
            Allocation for target block.
        method : GateMethod
            Method to use (AUTO for automatic selection).
            
        Returns
        -------
        GateEmissionResult
            Result describing what was emitted.
        """
        selected_method = self._select_method(GateType.CNOT, method)
        
        if selected_method == GateMethod.TRANSVERSAL:
            return self._emit_transversal_cnot(circuit, control_alloc, target_alloc)
        elif selected_method == GateMethod.SURGERY:
            return self._emit_surgery_cnot(circuit, control_alloc, target_alloc)
        else:
            return self._emit_teleported_cnot(circuit, control_alloc, target_alloc)
    
    def _emit_transversal_cnot(
        self,
        circuit: stim.Circuit,
        control_alloc: BlockAllocation,
        target_alloc: BlockAllocation,
    ) -> GateEmissionResult:
        """
        Emit transversal CNOT using the TransversalCNOT gadget.
        
        Emits raw CNOT gates qubit-by-qubit between control and target blocks.
        No DETECTOR instructions or transform tracking - decoder handles syndrome.
        """
        # Lazily create the gadget (without stabilizer rounds - we handle those separately)
        if self._transversal_cnot is None:
            self._transversal_cnot = TransversalCNOT(include_stabilizer_rounds=False)
        
        # Create a QubitAllocation that maps to our blocks
        alloc = self._create_two_block_allocation(control_alloc, target_alloc)
        
        # Reset gadget phase counter for fresh emission
        self._transversal_cnot._current_phase = 0
        
        # Use the gadget's emit_next_phase method (ctx not used for transversal)
        phase_result = self._transversal_cnot.emit_next_phase(circuit, alloc, None)
        
        return GateEmissionResult(
            gate_type=GateType.CNOT,
            method_used=GateMethod.TRANSVERSAL,
            requires_correction=False,
            stabilizer_transform=phase_result.stabilizer_transform if phase_result.stabilizer_transform else StabilizerTransform.identity(clear_history=True),
        )
    
    def _create_two_block_allocation(
        self,
        block0: BlockAllocation,
        block1: BlockAllocation,
    ) -> QubitAllocation:
        """
        Create a QubitAllocation from two BlockAllocations.
        
        Maps block0 -> "block_0" and block1 -> "block_1" as expected by gadgets.
        """
        alloc = QubitAllocation()
        
        # Create block_0 from control allocation
        alloc.blocks["block_0"] = BlockAllocation(
            block_name="block_0",
            code=block0.code,
            data_start=block0.data_start,
            data_count=block0.data_count,
            x_anc_start=block0.x_anc_start,
            x_anc_count=block0.x_anc_count,
            z_anc_start=block0.z_anc_start,
            z_anc_count=block0.z_anc_count,
            offset=block0.offset,
        )
        
        # Create block_1 from target allocation
        alloc.blocks["block_1"] = BlockAllocation(
            block_name="block_1",
            code=block1.code,
            data_start=block1.data_start,
            data_count=block1.data_count,
            x_anc_start=block1.x_anc_start,
            x_anc_count=block1.x_anc_count,
            z_anc_start=block1.z_anc_start,
            z_anc_count=block1.z_anc_count,
            offset=block1.offset,
        )
        
        return alloc
    
    def _create_single_block_allocation(self, block: BlockAllocation) -> QubitAllocation:
        """Create a QubitAllocation from a single BlockAllocation."""
        alloc = QubitAllocation()
        alloc.blocks["block_0"] = BlockAllocation(
            block_name="block_0",
            code=block.code,
            data_start=block.data_start,
            data_count=block.data_count,
            x_anc_start=block.x_anc_start,
            x_anc_count=block.x_anc_count,
            z_anc_start=block.z_anc_start,
            z_anc_count=block.z_anc_count,
            offset=block.offset,
        )
        return alloc
    
    def _emit_surgery_cnot(
        self,
        circuit: stim.Circuit,
        control_alloc: BlockAllocation,
        target_alloc: BlockAllocation,
    ) -> GateEmissionResult:
        """
        Emit surgery-based CNOT using the SurgeryCNOT gadget.
        
        Uses ZZ+XX merge protocol for logical CNOT via the existing
        css_surgery module. Emits raw operations only.
        """
        # Lazily create the gadget (minimal rounds - we handle EC separately)
        if self._surgery_cnot is None:
            self._surgery_cnot = SurgeryCNOT(
                num_rounds_before=0,
                num_rounds_after=0,
                num_merge_rounds=1,
            )
        
        # Create allocation for the gadget
        alloc = self._create_two_block_allocation(control_alloc, target_alloc)
        
        # Reset gadget phase counter
        self._surgery_cnot._current_phase = 0
        
        # Surgery CNOT has 2 phases: ZZ merge then XX merge
        # Emit both phases
        phase_result = None
        for _ in range(self._surgery_cnot.num_phases):
            phase_result = self._surgery_cnot.emit_next_phase(circuit, alloc, None)
        
        return GateEmissionResult(
            gate_type=GateType.CNOT,
            method_used=GateMethod.SURGERY,
            requires_correction=False,  # Surgery handles corrections internally
            stabilizer_transform=phase_result.stabilizer_transform if phase_result and phase_result.stabilizer_transform else StabilizerTransform.identity(clear_history=True),
        )
    
    def _emit_teleported_cnot(
        self,
        circuit: stim.Circuit,
        control_alloc: BlockAllocation,
        target_alloc: BlockAllocation,
    ) -> GateEmissionResult:
        """
        Emit teleportation-based CNOT.
        
        Uses measurement-based protocol via ancilla code blocks.
        
        NOTE: The teleportation.py module currently provides single-qubit
        teleported gates (TeleportedHadamard, TeleportedS, etc.). A full
        TeleportedCNOT would require:
        1. Ancilla block prepared in Bell state
        2. Bell measurement between control and ancilla
        3. Classically-controlled correction on target
        
        For CSS codes (our focus), transversal CNOT is available and preferred.
        For non-CSS codes, surgery via SurgeryCNOT is the recommended approach.
        
        Current implementation: Falls back to transversal for CSS codes.
        TODO: Implement full teleported CNOT using teleportation.py infrastructure.
        """
        # For CSS codes, transversal CNOT is available
        if self._has_transversal_cnot:
            return self._emit_transversal_cnot(circuit, control_alloc, target_alloc)
        
        # For non-CSS codes with surgery support, use that
        if self._has_surgery:
            return self._emit_surgery_cnot(circuit, control_alloc, target_alloc)
        
        # Ultimate fallback: emit transversal pattern (may not be fault-tolerant)
        # A proper implementation would use teleportation infrastructure
        return self._emit_transversal_cnot(circuit, control_alloc, target_alloc)
    
    # =========================================================================
    # Logical CZ
    # =========================================================================
    
    def emit_logical_cz(
        self,
        circuit: stim.Circuit,
        block1_alloc: BlockAllocation,
        block2_alloc: BlockAllocation,
        method: GateMethod = GateMethod.AUTO,
    ) -> GateEmissionResult:
        """
        Emit a logical CZ between two inner code blocks.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        block1_alloc : BlockAllocation
            Allocation for first block.
        block2_alloc : BlockAllocation
            Allocation for second block.
        method : GateMethod
            Method to use (AUTO for automatic selection).
            
        Returns
        -------
        GateEmissionResult
            Result describing what was emitted.
        """
        selected_method = self._select_method(GateType.CZ, method)
        
        if selected_method == GateMethod.TRANSVERSAL:
            return self._emit_transversal_cz(circuit, block1_alloc, block2_alloc)
        else:
            # CZ = H_tgt CNOT H_tgt
            # For non-transversal, decompose into H + CNOT + H
            self.emit_logical_hadamard(circuit, block2_alloc)
            result = self.emit_logical_cnot(circuit, block1_alloc, block2_alloc, method)
            self.emit_logical_hadamard(circuit, block2_alloc)
            result.gate_type = GateType.CZ
            return result
    
    def _emit_transversal_cz(
        self,
        circuit: stim.Circuit,
        block1_alloc: BlockAllocation,
        block2_alloc: BlockAllocation,
    ) -> GateEmissionResult:
        """
        Emit transversal CZ using the TransversalCZ gadget.
        
        Emits raw CZ gates qubit-by-qubit between the two blocks.
        No DETECTOR instructions or transform tracking.
        """
        # Lazily create the gadget
        if self._transversal_cz is None:
            self._transversal_cz = TransversalCZ(include_stabilizer_rounds=False)
        
        # Create allocation for the gadget
        alloc = self._create_two_block_allocation(block1_alloc, block2_alloc)
        
        # Reset and emit
        self._transversal_cz._current_phase = 0
        phase_result = self._transversal_cz.emit_next_phase(circuit, alloc, None)
        
        return GateEmissionResult(
            gate_type=GateType.CZ,
            method_used=GateMethod.TRANSVERSAL,
            requires_correction=False,
            stabilizer_transform=phase_result.stabilizer_transform if phase_result.stabilizer_transform else StabilizerTransform.identity(clear_history=True),
        )
    
    # =========================================================================
    # Single-Qubit Logical Gates
    # =========================================================================
    
    def emit_logical_hadamard(
        self,
        circuit: stim.Circuit,
        block_alloc: BlockAllocation,
    ) -> GateEmissionResult:
        """
        Emit logical Hadamard using the TransversalHadamard gadget.
        
        For CSS codes, applies H to all data qubits (swaps X/Z bases).
        """
        # Lazily create the gadget
        if self._transversal_h is None:
            self._transversal_h = TransversalHadamard(include_stabilizer_rounds=False)
        
        # Create allocation
        alloc = self._create_single_block_allocation(block_alloc)
        
        # Reset and emit
        self._transversal_h._current_phase = 0
        phase_result = self._transversal_h.emit_next_phase(circuit, alloc, None)
        
        return GateEmissionResult(
            gate_type=GateType.HADAMARD,
            method_used=GateMethod.TRANSVERSAL,
            requires_correction=False,
            stabilizer_transform=phase_result.stabilizer_transform if phase_result.stabilizer_transform else StabilizerTransform.hadamard(),
        )
    
    def emit_logical_x(
        self,
        circuit: stim.Circuit,
        block_alloc: BlockAllocation,
        logical_idx: int = 0,
    ) -> GateEmissionResult:
        """
        Emit logical X using the TransversalX gadget.
        
        Applies X to all qubits in the logical X support.
        """
        # Lazily create the gadget
        if self._transversal_x is None:
            self._transversal_x = TransversalX(include_stabilizer_rounds=False)
        
        # Create allocation
        alloc = self._create_single_block_allocation(block_alloc)
        
        # Reset and emit
        self._transversal_x._current_phase = 0
        phase_result = self._transversal_x.emit_next_phase(circuit, alloc, None)
        
        return GateEmissionResult(
            gate_type=GateType.X,
            method_used=GateMethod.TRANSVERSAL,
            requires_correction=False,
            stabilizer_transform=phase_result.stabilizer_transform if phase_result.stabilizer_transform else StabilizerTransform.pauli(),
        )
    
    def emit_logical_z(
        self,
        circuit: stim.Circuit,
        block_alloc: BlockAllocation,
        logical_idx: int = 0,
    ) -> GateEmissionResult:
        """
        Emit logical Z using the TransversalZ gadget.
        
        Applies Z to all qubits in the logical Z support.
        """
        # Lazily create the gadget
        if self._transversal_z is None:
            self._transversal_z = TransversalZ(include_stabilizer_rounds=False)
        
        # Create allocation
        alloc = self._create_single_block_allocation(block_alloc)
        
        # Reset and emit
        self._transversal_z._current_phase = 0
        phase_result = self._transversal_z.emit_next_phase(circuit, alloc, None)
        
        return GateEmissionResult(
            gate_type=GateType.Z,
            method_used=GateMethod.TRANSVERSAL,
            requires_correction=False,
            stabilizer_transform=phase_result.stabilizer_transform if phase_result.stabilizer_transform else StabilizerTransform.pauli(),
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_recommended_method(self, gate_type: GateType) -> GateMethod:
        """
        Get the recommended method for a gate type.
        
        Parameters
        ----------
        gate_type : GateType
            Type of gate.
            
        Returns
        -------
        GateMethod
            Recommended implementation method.
        """
        return self._select_method(gate_type, GateMethod.AUTO)
    
    def describe_capabilities(self) -> Dict[str, Any]:
        """
        Describe the capabilities of this dispatcher.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary describing available methods and code properties.
        """
        return {
            "inner_code": str(self._inner_code),
            "has_transversal_cnot": self._has_transversal_cnot,
            "has_surgery": self._has_surgery,
            "available_cnot_methods": [m.name for m in self._available_cnot_methods],
            "recommended_cnot_method": self.get_recommended_method(GateType.CNOT).name,
            "is_css": getattr(self._inner_code, 'is_css', False),
        }
    
    def supports_code_type(self, code_type: str) -> bool:
        """Check if this dispatcher supports a given code type."""
        supported_types = {"css", "color", "xyz_color"}
        return code_type in supported_types
    
    def emit_logical_gate_generic(
        self,
        circuit: stim.Circuit,
        gate_name: str,
        allocations: List[BlockAllocation],
        method: GateMethod = GateMethod.AUTO,
    ) -> GateEmissionResult:
        """
        Generic gate emission interface for extensibility.
        
        This provides a single entry point for emitting any logical gate,
        which can be extended to support new gate types.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        gate_name : str
            Name of the gate ("CNOT", "CZ", "H", "X", "Z", etc.).
        allocations : List[BlockAllocation]
            Block allocations (1 for single-qubit, 2 for two-qubit gates).
        method : GateMethod
            Method to use.
            
        Returns
        -------
        GateEmissionResult
            Result of the gate emission.
        """
        gate_name = gate_name.upper()
        
        if gate_name in ("CNOT", "CX"):
            if len(allocations) != 2:
                raise ValueError("CNOT requires exactly 2 block allocations")
            return self.emit_logical_cnot(circuit, allocations[0], allocations[1], method)
        
        elif gate_name == "CZ":
            if len(allocations) != 2:
                raise ValueError("CZ requires exactly 2 block allocations")
            return self.emit_logical_cz(circuit, allocations[0], allocations[1], method)
        
        elif gate_name == "H":
            if len(allocations) != 1:
                raise ValueError("H requires exactly 1 block allocation")
            return self.emit_logical_hadamard(circuit, allocations[0])
        
        elif gate_name == "X":
            if len(allocations) != 1:
                raise ValueError("X requires exactly 1 block allocation")
            return self.emit_logical_x(circuit, allocations[0])
        
        elif gate_name == "Z":
            if len(allocations) != 1:
                raise ValueError("Z requires exactly 1 block allocation")
            return self.emit_logical_z(circuit, allocations[0])
        
        else:
            raise ValueError(f"Unknown gate: {gate_name}")
    
    def get_non_css_fallback_method(self, gate_type: GateType) -> Optional[GateMethod]:
        """
        Get the fallback method for non-CSS codes.
        
        For non-CSS codes, not all gate methods are available.
        This returns the best available fallback.
        
        Parameters
        ----------
        gate_type : GateType
            Type of gate.
            
        Returns
        -------
        Optional[GateMethod]
            Fallback method, or None if no method is available.
        """
        is_css = getattr(self._inner_code, 'is_css', False)
        
        if is_css:
            # CSS codes have full support
            return self._select_method(gate_type, GateMethod.AUTO)
        
        # Non-CSS codes: teleportation is the universal fallback
        if gate_type in (GateType.CNOT, GateType.CZ):
            return GateMethod.TELEPORTATION
        
        # Single-qubit gates are usually available
        if gate_type in (GateType.X, GateType.Z):
            return GateMethod.TRANSVERSAL
        
        # Hadamard might not be transversal for non-CSS
        if gate_type == GateType.HADAMARD:
            return GateMethod.TELEPORTATION
        
        return None