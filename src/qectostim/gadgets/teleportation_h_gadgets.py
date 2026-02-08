#!/usr/bin/env python3
"""
H-Teleportation Gadgets for FaultTolerantGadgetExperiment Integration.

════════════════════════════════════════════════════════════════════════════════
OVERVIEW
════════════════════════════════════════════════════════════════════════════════

This module provides `Gadget` subclasses that implement H-teleportation using
either CZ or CNOT transversal gates. These gadgets integrate with the
`FaultTolerantGadgetExperiment` framework to produce circuits with the same
detector coverage and observables as the proven ground truth builders in
`cz_h_teleportation.py`.

Design Goals:
1. Match detector coverage of ground truth `CZHTeleportationBuilder.to_stim()`
   and `CNOTHTeleportationBuilder.to_stim()` 
2. Work seamlessly with `FaultTolerantGadgetExperiment` framework
3. Support both |0⟩ and |+⟩ input states with correct observables

════════════════════════════════════════════════════════════════════════════════
PROTOCOL SUMMARY
════════════════════════════════════════════════════════════════════════════════

CZ H-Teleportation (self-dual codes):
    |ψ⟩_D ⊗ |+⟩_A → [EC] → [CZ] → [EC] → MX(D), MX/MZ(A)
    
    |0⟩ input: Observable = X_L(A)
    |+⟩ input: Observable = X_L(D) ⊕ Z_L(A)
    
    Crossing detectors: 3-term for X stabilizers (X_pre ⊕ X_post ⊕ Z_post)

CNOT H-Teleportation (any CSS code):
    |ψ⟩_D ⊗ |0⟩_A → [EC] → [CNOT(D→A)] → [EC] → MX(D), MZ/MX(A)
    
    |0⟩ input: Observable = Z_L(A)
    |+⟩ input: Observable = X_L(D) ⊕ X_L(A)
    
    Crossing detectors: 2-term for most, 3-term for Z_D when |+⟩ input

════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional, Dict, Any, Set, TYPE_CHECKING

import numpy as np
import stim

from qectostim.codes.abstract_code import Code
from qectostim.codes.abstract_css import CSSCode
from qectostim.gadgets.base import (
    Gadget,
    GadgetMetadata,
    StabilizerTransform,
    ObservableTransform,
    TwoQubitObservableTransform,
    TeleportationGadgetMixin,
    PhaseResult,
    PhaseType,
    FrameUpdate,
)
from qectostim.gadgets.layout import (
    GadgetLayout,
    QubitAllocation,
)

if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds import DetectorContext


# ════════════════════════════════════════════════════════════════════════════════
# CZ H-TELEPORTATION GADGET
# ════════════════════════════════════════════════════════════════════════════════

class CZHTeleportGadget(TeleportationGadgetMixin, Gadget):
    """
    CZ-based H-Teleportation Gadget for self-dual CSS codes.
    
    This gadget implements the H gate via teleportation using a transversal CZ
    between the data block and an ancilla block prepared in |+⟩.
    
    Protocol:
        1. Data block prepared in |ψ⟩_L (|0⟩ or |+⟩)
        2. Ancilla block prepared in |+⟩_L
        3. EC rounds on both blocks
        4. Transversal CZ
        5. EC rounds on both blocks
        6. MX on data, MX or MZ on ancilla (depends on input)
    
    Observable:
        |0⟩ input: X_L(A)
        |+⟩ input: X_L(D) ⊕ Z_L(A)
    
    Requirements:
        - Self-dual code (hx[i] = hz[i]) for 3-term X stabilizer crossing detectors
    """
    
    # TeleportationGadgetMixin configuration
    _data_block_name = "data_block"
    _ancilla_block_name = "ancilla_block"
    _ancilla_initial_state = "+"  # CZ needs |+⟩ on ancilla
    
    @property
    def gate_name(self) -> str:
        """CZ teleportation uses transversal CZ gate."""
        return "CZ"
    
    def __init__(
        self,
        input_state: Literal["0", "+"] = "0",
        num_ec_rounds: int = 1,
    ):
        """
        Initialize CZ H-Teleportation gadget.
        
        Parameters
        ----------
        input_state : Literal["0", "+"]
            Input logical state. "0" for |0⟩_L, "+" for |+⟩_L.
        num_ec_rounds : int
            Number of EC rounds (typically = distance).
        """
        super().__init__(input_state=input_state)
        self.input_state = input_state
        self.num_ec_rounds = num_ec_rounds
        
        # Cache for code info
        self._code: Optional[CSSCode] = None
        self._z_stabilizers: List[List[int]] = []
        self._x_stabilizers: List[List[int]] = []
        self._z_logical: List[int] = []
        self._x_logical: List[int] = []
        
        # Multi-phase tracking
        self._current_phase = 0
    
    # =========================================================================
    # TeleportationGadgetMixin - Override with specific values
    # =========================================================================
    # Note: Most methods inherited from TeleportationGadgetMixin using
    # _data_block_name, _ancilla_block_name, _ancilla_initial_state
    
    def get_output_block_name(self) -> str:
        return "ancilla_block"
    
    def get_teleportation_correction_info(self) -> Dict[str, Any]:
        return {
            "input_state": self.input_state,
            "gate": "CZ",
            "data_measurement_basis": "X",
            "ancilla_measurement_basis": "X" if self.input_state == "0" else "Z",
        }
    
    # =========================================================================
    # Gadget interface implementation
    # =========================================================================
    
    @property
    def num_phases(self) -> int:
        """
        CZ H-teleportation has 3 phases:
        1. PREPARATION: Prepare ancilla block in |+⟩
        2. GATE: Apply transversal CZ
        3. MEASUREMENT: Bell measurement on data block (MX)
        
        The experiment emits EC rounds between phases.
        """
        return 3
    
    def reset_phases(self) -> None:
        """Reset phase counter for new circuit generation."""
        self._current_phase = 0
        self._ancilla_meas_start: Optional[int] = None  # Track for frame corrections
        self._data_meas_start: Optional[int] = None

    def emit_next_phase(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """
        Emit the next phase of CZ H-teleportation.
        
        Phase 1 (PREPARATION): Prepare ancilla block in |+⟩_L
        Phase 2 (GATE): Apply transversal CZ
        Phase 3 (MEASUREMENT): Bell measurement on data block (MX)
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        alloc : QubitAllocation
            Qubit allocation with data_block and ancilla_block.
        ctx : DetectorContext
            Detector context for Pauli frame tracking.
            
        Returns
        -------
        PhaseResult
            Result describing what was emitted and next steps.
        """
        self._current_phase += 1
        
        # Get block allocations
        data_block = alloc.get_block("data_block")
        ancilla_block = alloc.get_block("ancilla_block")
        
        if data_block is None:
            data_block = alloc.get_block("block_0")
        if ancilla_block is None:
            ancilla_block = alloc.get_block("block_1")
        
        if data_block is None or ancilla_block is None:
            raise ValueError("CZ H-teleportation requires data_block and ancilla_block")
        
        data_qubits = data_block.get_data_qubits()
        ancilla_qubits = ancilla_block.get_data_qubits()
        n = min(len(data_qubits), len(ancilla_qubits))
        
        if self._current_phase == 1:
            # ═══════════════════════════════════════════════════════════
            # PHASE 1: PREPARATION - Ancilla block in |+⟩_L
            # ═══════════════════════════════════════════════════════════
            # Ancilla prep requirements are declared via get_preparation_config()
            # (PreparationConfig from gadgets/) which specifies:
            #   ancilla_block: |+⟩, X deterministic, Z indeterminate
            # The experiment's preparation module (experiments/preparation.py)
            # executes the actual RX instruction at the correct time (before
            # any syndrome rounds), ensuring proper detector chain.
            #
            # Request inter-phase rounds for pre-gate EC on both blocks.
            return PhaseResult(
                phase_type=PhaseType.PREPARATION,
                is_final=False,
                needs_stabilizer_rounds=self.num_ec_rounds,
                stabilizer_transform=None,
                pauli_frame_update=None,
            )
            
        elif self._current_phase == 2:
            # ═══════════════════════════════════════════════════════════
            # PHASE 2: GATE - Apply transversal CZ
            # ═══════════════════════════════════════════════════════════
            for i in range(n):
                circuit.append("CZ", [data_qubits[i], ancilla_qubits[i]])
            circuit.append("TICK")
            
            # CZ transformation:
            # Z_D → Z_D (unchanged)
            # Z_A → Z_A (unchanged)
            # X_D → X_D ⊗ Z_A (X stabilizer picks up Z from partner)
            # X_A → Z_D ⊗ X_A (X stabilizer picks up Z from partner)
            return PhaseResult.gate_phase(
                is_final=False,
                transform=self.get_stabilizer_transform(),
                needs_stabilizer_rounds=self.num_ec_rounds,  # EC after gate
            )
            
        else:
            # ═══════════════════════════════════════════════════════════
            # PHASE 3: MEASUREMENT - Bell measurement (MX on data)
            # ═══════════════════════════════════════════════════════════
            # Record measurement start for frame correction
            self._data_meas_start = circuit.num_measurements
            
            # MX on all data qubits (Bell measurement)
            circuit.append("MX", data_qubits)
            circuit.append("TICK")
            
            # Frame update: MX outcomes determine Pauli corrections
            # For CZ teleportation: X measurement on data → Z correction on ancilla
            frame_update = FrameUpdate(
                block_name='ancilla_block',
                x_meas=list(range(n)),  # Relative indices within this measurement
                z_meas=[],  # No Z correction needed
                teleport=True,
                source_block='data_block',
            )
            
            return PhaseResult(
                phase_type=PhaseType.MEASUREMENT,
                is_final=True,
                needs_stabilizer_rounds=0,
                stabilizer_transform=None,
                destroyed_blocks={'data_block'},  # Data block is consumed
                pauli_frame_update=frame_update,
                measurement_count=n,  # Track the MX measurements in ctx
            )
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """
        Get stabilizer transform for CZ teleportation.
        
        CZ transforms X stabilizers to X⊗Z products.
        
        For self-dual codes with matching X/Z stabilizer supports,
        this requires 3-term crossing detectors.
        """
        return StabilizerTransform.teleportation()
    
    def get_space_like_detector_config(self) -> Dict[str, Optional[str]]:
        """
        Return per-block space-like detector configuration for CZ teleportation.
        
        Space-like (boundary) detectors correlate final syndrome with final
        data qubit measurements. They work when measuring in the same basis
        as the stabilizer type being checked.
        
        After CZ:
        - X_D → X_D ⊗ Z_A (transformed - but data_block is destroyed)
        - Z_D → Z_D (unchanged)
        - X_A → Z_D ⊗ X_A (transformed - but Z_D projected by data MX)
        - Z_A → Z_A (unchanged)
        
        For |+⟩ input (data MX, ancilla MZ):
        - Data block: destroyed (MX), skip space-like
        - Ancilla block: Emit Z space-like (Z_A unchanged, measured in Z)
        
        For |0⟩ input (data MX, ancilla MX):
        - Data block: destroyed (MX), skip space-like
        - Ancilla block: Emit X space-like
          (X_A(post) = Z_D ⊗ X_A, but Z_D is projected by data MX,
           so effectively X_A at final syndrome matches ancilla MX boundary)
        """
        if self.input_state == "+":
            # |+⟩ input: ancilla measured in Z basis
            # Z_A stabilizers unchanged, so Z space-like works
            return {
                "data_block": None,  # Destroyed - skip
                "ancilla_block": "Z",  # Z space-like - Z_A unchanged
            }
        else:
            # |0⟩ input: ancilla measured in X basis
            # X_A syndrome correlates with ancilla data MX (boundary)
            return {
                "data_block": None,  # Destroyed - skip
                "ancilla_block": "X",  # X space-like - boundary detection
            }

    def get_observable_transform(self) -> TwoQubitObservableTransform:
        """
        Return the two-qubit observable transform for CZ.
        
        CZ transforms:
            X_0 → X_0 ⊗ Z_1
            Z_0 → Z_0
            X_1 → Z_0 ⊗ X_1
            Z_1 → Z_1
        """
        return TwoQubitObservableTransform.cz()
    
    def compute_layout(self, codes: List[Code]) -> GadgetLayout:
        """
        Compute layout for two code blocks (data and ancilla).
        
        Parameters
        ----------
        codes : List[Code]
            List containing exactly one code (used for both blocks).
            
        Returns
        -------
        GadgetLayout
            Layout with data_block and ancilla_block.
        """
        if len(codes) != 1:
            raise ValueError(
                f"CZ H-teleportation requires exactly 1 code (for both blocks), "
                f"got {len(codes)}"
            )
        
        code = codes[0]
        self._code = code
        
        # CZ gadget requires self-dual CSS code (hx=hz)
        if not code.is_self_dual:
            raise ValueError(
                f"CZ H-teleportation requires a self-dual CSS code (hx=hz), "
                f"but {code.name} is not self-dual. Use CNOTHTeleportGadget for "
                f"non-self-dual codes."
            )
        
        self._extract_code_info(code)
        
        layout = GadgetLayout(target_dim=2)
        
        # Data block at origin
        layout.add_block(
            name="data_block",
            code=code,
            offset=(0.0, 0.0),
        )
        
        # Dynamic margin based on code distance
        margin = max(3.0, (code.distance or 3) * 1.5)
        
        # Ancilla block offset to the right
        layout.add_block(
            name="ancilla_block", 
            code=code,
            auto_offset=True,
            margin=margin,
        )
        
        self._cached_layout = layout
        return layout
    
    def _extract_code_info(self, code: Code) -> None:
        """
        Extract stabilizer and logical operator info from code.
        
        Uses Code ABC methods directly — no hasattr guards needed.
        """
        self._z_stabilizers = code.get_z_stabilizers()
        self._x_stabilizers = code.get_x_stabilizers()
        self._z_logical = code.get_logical_z_support(0)
        self._x_logical = code.get_logical_x_support(0)
    
    def get_metadata(self) -> GadgetMetadata:
        """Return metadata about this gadget."""
        return GadgetMetadata(
            gadget_type="h_teleportation",
            logical_operation="H",
            extra={
                "gate": "CZ",
                "input_state": self.input_state,
                "num_ec_rounds": self.num_ec_rounds,
                "is_teleportation": True,
                # NOTE: data_block is NOT destroyed - it gets EC rounds and final measurement
                "output_block": "ancilla_block",
            },
        )


# ════════════════════════════════════════════════════════════════════════════════
# CNOT H-TELEPORTATION GADGET
# ════════════════════════════════════════════════════════════════════════════════

class CNOTHTeleportGadget(TeleportationGadgetMixin, Gadget):
    """
    CNOT-based H-Teleportation Gadget for any CSS code.
    
    This gadget implements the H gate via teleportation using a transversal
    CNOT(D→A) between the data block (control) and an ancilla block (target)
    prepared in |0⟩.
    
    Protocol:
        1. Data block prepared in |ψ⟩_L (|0⟩ or |+⟩)
        2. Ancilla block prepared in |0⟩_L
        3. EC rounds on both blocks
        4. Transversal CNOT (data=control, ancilla=target)
        5. EC rounds on both blocks
        6. MX on data, MZ or MX on ancilla (depends on input)
    
    Observable:
        |0⟩ input: Z_L(A)
        |+⟩ input: X_L(D) ⊕ X_L(A)
    
    Advantages over CZ:
        - Works for ANY CSS code (not just self-dual)
        - X stabilizers have 2-term crossing (X unchanged through CNOT)
    """
    
    # TeleportationGadgetMixin configuration
    _data_block_name = "data_block"
    _ancilla_block_name = "ancilla_block"
    _ancilla_initial_state = "0"  # CNOT needs |0⟩ on ancilla (target)
    
    @property
    def gate_name(self) -> str:
        """CNOT teleportation uses transversal CNOT gate."""
        return "CNOT"
    
    def __init__(
        self,
        input_state: Literal["0", "+"] = "0",
        num_ec_rounds: int = 1,
    ):
        """
        Initialize CNOT H-Teleportation gadget.
        
        Parameters
        ----------
        input_state : Literal["0", "+"]
            Input logical state. "0" for |0⟩_L, "+" for |+⟩_L.
        num_ec_rounds : int
            Number of EC rounds (typically = distance).
        """
        super().__init__(input_state=input_state)
        self.input_state = input_state
        self.num_ec_rounds = num_ec_rounds
        
        # Cache for code info
        self._code: Optional[CSSCode] = None
        self._z_stabilizers: List[List[int]] = []
        self._x_stabilizers: List[List[int]] = []
        self._z_logical: List[int] = []
        self._x_logical: List[int] = []
        
        # Multi-phase tracking
        self._current_phase = 0
    
    # =========================================================================
    # TeleportationGadgetMixin - Override with specific values
    # =========================================================================
    # Note: Most methods inherited from TeleportationGadgetMixin using
    # _data_block_name, _ancilla_block_name, _ancilla_initial_state
    
    def get_output_block_name(self) -> str:
        return "ancilla_block"
    
    def get_default_stabilizer_ordering(self) -> Optional[str]:
        """CNOT ground truth uses X-first ordering in all rounds."""
        return "X_FIRST"
    
    def get_teleportation_correction_info(self) -> Dict[str, Any]:
        return {
            "input_state": self.input_state,
            "gate": "CNOT",
            "data_measurement_basis": "X",
            "ancilla_measurement_basis": "Z" if self.input_state == "0" else "X",
        }
    
    # =========================================================================
    # Gadget interface implementation
    # =========================================================================
    
    @property
    def num_phases(self) -> int:
        """
        CNOT H-teleportation has 3 phases:
        1. PREPARATION: Prepare ancilla block in |0⟩_L
        2. GATE: Apply transversal CNOT (data=control, ancilla=target)
        3. MEASUREMENT: Bell measurement on data block (MX)
        
        The experiment emits EC rounds between phases.
        """
        return 3
    
    def reset_phases(self) -> None:
        """Reset phase counter for new circuit generation."""
        self._current_phase = 0
        self._ancilla_meas_start: Optional[int] = None  # Track for frame corrections
        self._data_meas_start: Optional[int] = None
    
    def emit_next_phase(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """
        Emit the next phase of CNOT H-teleportation.
        
        Phase 1 (PREPARATION): Prepare ancilla block in |0⟩_L
        Phase 2 (GATE): Apply transversal CNOT (data=control, ancilla=target)
        Phase 3 (MEASUREMENT): Bell measurement on data block (MX)
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        alloc : QubitAllocation
            Qubit allocation with data_block and ancilla_block.
        ctx : DetectorContext
            Detector context for Pauli frame tracking.
            
        Returns
        -------
        PhaseResult
            Result describing what was emitted and next steps.
        """
        self._current_phase += 1
        
        # Get block allocations
        data_block = alloc.get_block("data_block")
        ancilla_block = alloc.get_block("ancilla_block")
        
        if data_block is None:
            data_block = alloc.get_block("block_0")
        if ancilla_block is None:
            ancilla_block = alloc.get_block("block_1")
        
        if data_block is None or ancilla_block is None:
            raise ValueError("CNOT H-teleportation requires data_block and ancilla_block")
        
        data_qubits = data_block.get_data_qubits()
        ancilla_qubits = ancilla_block.get_data_qubits()
        n = min(len(data_qubits), len(ancilla_qubits))
        
        if self._current_phase == 1:
            # ═══════════════════════════════════════════════════════════
            # PHASE 1: PREPARATION - Ancilla block in |0⟩_L
            # ═══════════════════════════════════════════════════════════
            # Ancilla prep requirements are declared via get_preparation_config()
            # (PreparationConfig from gadgets/) which specifies:
            #   ancilla_block: |0⟩, Z deterministic, X indeterminate
            # The experiment's preparation module (experiments/preparation.py)
            # executes the actual R instruction at the correct time (before
            # any syndrome rounds), ensuring proper detector chain.
            #
            # Request inter-phase rounds for pre-gate EC on both blocks.
            return PhaseResult(
                phase_type=PhaseType.PREPARATION,
                is_final=False,
                needs_stabilizer_rounds=self.num_ec_rounds,
                stabilizer_transform=None,
                pauli_frame_update=None,
            )
            
        elif self._current_phase == 2:
            # ═══════════════════════════════════════════════════════════
            # PHASE 2: GATE - Apply transversal CNOT
            # ═══════════════════════════════════════════════════════════
            # Emit transversal CNOT (data=control, ancilla=target)
            for i in range(n):
                circuit.append("CNOT", [data_qubits[i], ancilla_qubits[i]])
            circuit.append("TICK")
            
            # CNOT(D→A) transformation:
            # X_D → X_D (control X unchanged)
            # Z_D → Z_D ⊗ Z_A (control Z picks up target Z)
            # X_A → X_D ⊗ X_A (target X picks up control X)
            # Z_A → Z_A (target Z unchanged)
            return PhaseResult.gate_phase(
                is_final=False,
                transform=self.get_stabilizer_transform(),
                needs_stabilizer_rounds=self.num_ec_rounds,  # EC after gate
            )
            
        else:
            # ═══════════════════════════════════════════════════════════
            # PHASE 3: MEASUREMENT - Bell measurement (MX on data)
            # ═══════════════════════════════════════════════════════════
            # Record measurement start for frame correction
            self._data_meas_start = circuit.num_measurements
            
            # MX on all data qubits (Bell measurement)
            circuit.append("MX", data_qubits)
            circuit.append("TICK")
            
            # Frame update: MX outcomes determine Pauli corrections
            # For CNOT teleportation: X measurement on data → X correction on ancilla
            frame_update = FrameUpdate(
                block_name='ancilla_block',
                x_meas=list(range(n)),  # Relative indices within this measurement
                z_meas=[],  # No Z correction needed from MX
                teleport=True,
                source_block='data_block',
            )
            
            return PhaseResult(
                phase_type=PhaseType.MEASUREMENT,
                is_final=True,
                needs_stabilizer_rounds=0,
                stabilizer_transform=None,
                destroyed_blocks={'data_block'},  # Data block is consumed
                pauli_frame_update=frame_update,
                measurement_count=n,  # Track the MX measurements in ctx
            )
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """
        CNOT stabilizer transform.
        
        For CNOT(D→A):
        - X stabilizers are unchanged (control X invariant, target X picks up control)
        - Z stabilizers: control picks up target Z
        """
        return StabilizerTransform.teleportation()
    
    def get_space_like_detector_config(self) -> Dict[str, Optional[str]]:
        """
        Return per-block space-like detector configuration for CNOT teleportation.
        
        After CNOT(D→A):
        - X_D → X_D (unchanged)
        - Z_D → Z_D ⊗ Z_A (transformed, cannot use for space-like alone)
        - X_A → X_D ⊗ X_A (transformed)
        - Z_A → Z_A (unchanged)
        
        For |0⟩ input (data MX, ancilla MZ):
        - Data block: Emit X space-like (X stabilizers unchanged)
        - Ancilla block: Emit Z space-like (Z stabilizers unchanged)
        
        For |+⟩ input (data MX, ancilla MX):
        - Data block: Emit X space-like (X stabilizers unchanged)
        - Ancilla block: Emit X space-like
          Although X_A is transformed (X_A → X_D ⊗ X_A), at the boundary
          when both blocks are measured in X basis, X_D is projected and
          the boundary detector XORing last X syndrome with final X data
          measurements is still deterministic.
        """
        if self.input_state == "0":
            # |0⟩ input: ancilla measured in Z basis
            return {
                "data_block": "X",  # X space-like - X stabilizers unchanged
                "ancilla_block": "Z",  # Z space-like - Z stabilizers unchanged
            }
        else:
            # |+⟩ input: ancilla measured in X basis
            # X_A boundary detectors are valid because X_D is projected by data MX
            return {
                "data_block": "X",  # X space-like - X stabilizers unchanged
                "ancilla_block": "X",  # X space-like - boundary still works
            }
    
    def get_observable_transform(self) -> TwoQubitObservableTransform:
        """
        Return the two-qubit observable transform for CNOT.
        
        CNOT(D→A) transforms:
            X_D → X_D (control X unchanged)
            Z_D → Z_D ⊗ Z_A (control Z picks up target Z)
            X_A → X_D ⊗ X_A (target X picks up control X)
            Z_A → Z_A (target Z unchanged)
        """
        return TwoQubitObservableTransform.cnot()
    
    def compute_layout(self, codes: List[Code]) -> GadgetLayout:
        """
        Compute layout for two code blocks (data and ancilla).
        
        Parameters
        ----------
        codes : List[Code]
            List containing exactly one code (used for both blocks).
            
        Returns
        -------
        GadgetLayout
            Layout with data_block and ancilla_block.
        """
        if len(codes) != 1:
            raise ValueError(
                f"CNOT H-teleportation requires exactly 1 code (for both blocks), "
                f"got {len(codes)}"
            )
        
        code = codes[0]
        self._code = code
        self._extract_code_info(code)
        
        layout = GadgetLayout(target_dim=2)
        
        # Data block at origin
        layout.add_block(
            name="data_block",
            code=code,
            offset=(0.0, 0.0),
        )
        
        # Dynamic margin based on code distance
        margin = max(3.0, (code.distance or 3) * 1.5)
        
        # Ancilla block offset to the right
        layout.add_block(
            name="ancilla_block",
            code=code,
            auto_offset=True,
            margin=margin,
        )
        
        self._cached_layout = layout
        return layout
    
    def _extract_code_info(self, code: Code) -> None:
        """
        Extract stabilizer and logical operator info from code.
        
        Uses Code ABC methods directly — no hasattr guards needed.
        """
        self._z_stabilizers = code.get_z_stabilizers()
        self._x_stabilizers = code.get_x_stabilizers()
        self._z_logical = code.get_logical_z_support(0)
        self._x_logical = code.get_logical_x_support(0)
    
    def get_metadata(self) -> GadgetMetadata:
        """Return metadata about this gadget."""
        return GadgetMetadata(
            gadget_type="h_teleportation",
            logical_operation="H",
            extra={
                "gate": "CNOT",
                "input_state": self.input_state,
                "num_ec_rounds": self.num_ec_rounds,
                "is_teleportation": True,
                # NOTE: data_block is NOT destroyed - it gets EC rounds and final measurement
                "output_block": "ancilla_block",
            },
        )


# ════════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ════════════════════════════════════════════════════════════════════════════════

def create_h_teleport_gadget(
    gate: Literal["CZ", "CNOT"] = "CZ",
    input_state: Literal["0", "+"] = "0",
    num_ec_rounds: int = 1,
) -> Gadget:
    """
    Factory function to create an H-teleportation gadget.
    
    Parameters
    ----------
    gate : Literal["CZ", "CNOT"]
        Which transversal gate to use for teleportation.
        "CZ" requires self-dual codes, "CNOT" works for any CSS code.
    input_state : Literal["0", "+"]
        Input logical state. "0" for |0⟩_L, "+" for |+⟩_L.
    num_ec_rounds : int
        Number of EC rounds (typically = distance).
        
    Returns
    -------
    Gadget
        Either CZHTeleportGadget or CNOTHTeleportGadget.
    """
    if gate.upper() == "CZ":
        return CZHTeleportGadget(input_state=input_state, num_ec_rounds=num_ec_rounds)
    elif gate.upper() == "CNOT":
        return CNOTHTeleportGadget(input_state=input_state, num_ec_rounds=num_ec_rounds)
    else:
        raise ValueError(f"Unknown gate '{gate}'. Supported: 'CZ', 'CNOT'")
