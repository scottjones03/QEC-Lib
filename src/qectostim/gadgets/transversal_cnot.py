#!/usr/bin/env python3
"""
Transversal CNOT Gadget for FaultTolerantGadgetExperiment.

════════════════════════════════════════════════════════════════════════════════
OVERVIEW
════════════════════════════════════════════════════════════════════════════════

This module provides a `Gadget` subclass for transversal CNOT gates between
two independent code blocks. Unlike CNOT H-teleportation, this gadget:

1. Does NOT destroy either block (both survive the gate)
2. Does NOT apply H (just the CNOT)
3. Entangles the two blocks (stabilizers spread across boundaries)

This is useful for:
- Multi-qubit logical circuits with intermediate gates
- Testing the gadget framework with simpler operations
- Understanding CNOT stabilizer transformations

════════════════════════════════════════════════════════════════════════════════
STABILIZER TRANSFORMATIONS (Standard Heisenberg picture)
════════════════════════════════════════════════════════════════════════════════

CNOT(control → target) transforms:
    X_ctrl → X_ctrl ⊗ X_tgt       (X spreads from control to target)
    Z_ctrl → Z_ctrl               (unchanged)
    X_tgt  → X_tgt                (unchanged)
    Z_tgt  → Z_ctrl ⊗ Z_tgt       (Z back-propagates from target to control)

Crossing Detector Analysis:
- X_ctrl: 3-term (X_ctrl(pre) ⊕ X_ctrl(post) ⊕ X_tgt(post)) - spreads
- Z_ctrl: 2-term (Z_ctrl(pre) ⊕ Z_ctrl(post)) - unchanged
- X_tgt:  2-term (X_tgt(pre) ⊕ X_tgt(post)) - unchanged
- Z_tgt:  3-term (Z_tgt(pre) ⊕ Z_ctrl(post) ⊕ Z_tgt(post)) - picks up

All four crossing detectors are unconditionally deterministic for any input
state because the algebra always XORs to 0 regardless of stabilizer eigenvalues.

════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any, TYPE_CHECKING

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
    PhaseResult,
    PhaseType,
    ObservableConfig,
    ObservableTerm,
    PreparationConfig,
    BlockPreparationConfig,
    MeasurementConfig,
    BoundaryDetectorConfig,
    CrossingDetectorConfig,
    CrossingDetectorFormula,
    CrossingDetectorTerm,
)
from qectostim.gadgets.layout import GadgetLayout, QubitAllocation

if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds import DetectorContext


class TransversalCNOTGadget(Gadget):
    """
    Transversal CNOT Gadget between two code blocks.
    
    Applies CNOT(control_block → target_block) transversally.
    Both blocks survive - neither is destroyed.
    
    This is a simpler gadget than teleportation-based gadgets and
    demonstrates the basic gadget interface without mid-circuit measurements.
    
    Protocol:
        1. Control block prepared in |ψ₀⟩
        2. Target block prepared in |ψ₁⟩  
        3. EC rounds on both blocks
        4. Transversal CNOT
        5. EC rounds on both blocks
        6. Final measurement on both blocks (per-block basis)
    
    Observable (CNOT truth table — each block measured in its OUTPUT state basis):
        - |0⟩|0⟩ → |0⟩|0⟩: MZ(ctrl) ⊕ MZ(tgt) = 0 deterministic
        - |+⟩|0⟩ → |+⟩|+⟩: MX(ctrl) ⊕ MX(tgt) = 0 deterministic
        - |0⟩|+⟩ → |0⟩|+⟩: MZ(ctrl) ⊕ MX(tgt) = 0 deterministic
        - |+⟩|+⟩ → |+⟩|0⟩: MX(ctrl) = 0 deterministic
    """
    
    @property
    def gate_name(self) -> str:
        """Transversal CNOT gate."""
        return "CNOT"
    
    def __init__(
        self,
        control_state: Literal["0", "+"] = "0",
        target_state: Literal["0", "+"] = "0",
    ):
        """
        Initialize Transversal CNOT gadget.
        
        Parameters
        ----------
        control_state : Literal["0", "+"]
            Initial state of control block.
        target_state : Literal["0", "+"]
            Initial state of target block.
        """
        super().__init__()
        self.control_state = control_state
        self.target_state = target_state
        
        # Cache
        self._code: Optional[CSSCode] = None
        self._current_phase = 0
    
    # =========================================================================
    # Gadget interface implementation
    # =========================================================================
    
    @property
    def num_phases(self) -> int:
        """Transversal CNOT is single-phase - the experiment handles EC rounds."""
        return 1
    
    @property
    def num_blocks(self) -> int:
        """Two blocks: control and target."""
        return 2
    
    def reset_phases(self) -> None:
        """Reset phase counter for new circuit generation."""
        self._current_phase = 0
    
    def is_teleportation_gadget(self) -> bool:
        """Not a teleportation gadget - both blocks survive."""
        return False
    
    def get_measurement_basis(self) -> str:
        """
        Return the primary measurement basis for CNOT, derived from output states.
        
        For TransversalCNOT, the per-block bases are determined by the output
        states. The 'primary' basis returned here is used by the experiment
        for global decisions (e.g., fallback basis). The per-block bases are
        provided by get_measurement_config().
        
        Returns the control block's output basis as the primary basis.
        """
        bases = self._get_output_bases()
        return bases["block_0"]
    
    def get_destroyed_blocks(self) -> List[str]:
        """No blocks are destroyed."""
        return []
    
    def get_x_stabilizer_mode(self) -> str:
        """
        Return the gate type to use for X stabilizer measurement.
        
        For |+⟩ state preparation, CZ-based X-stabilizer measurement gives
        non-deterministic results because the H-CZ-H circuit doesn't properly
        measure X-parity on product |+⟩ states.
        
        Using CX (CNOT ancilla→data) with H-CX-H correctly measures X-parity:
        - The ancilla measures the XOR of X eigenvalues of data qubits
        - For |+⟩^⊗n, all X eigenvalues are +1, so measurement is deterministic 0
        
        Returns
        -------
        str
            "cx" if any block uses |+⟩ preparation, "cz" otherwise.
        """
        # Use CX mode if either block is prepared in |+⟩
        if self.control_state == "+" or self.target_state == "+":
            return "cx"
        return "cz"
    
    def emit_next_phase(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """
        Emit the transversal CNOT gate.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        alloc : QubitAllocation
            Qubit allocation with block_0 (control) and block_1 (target).
        ctx : DetectorContext
            Detector context for Pauli frame tracking.
            
        Returns
        -------
        PhaseResult
            Gate phase result with CNOT stabilizer transform.
        """
        self._current_phase += 1
        
        # Get block allocations
        ctrl_block = alloc.get_block("block_0")
        tgt_block = alloc.get_block("block_1")
        
        if ctrl_block is None or tgt_block is None:
            raise ValueError("Transversal CNOT requires block_0 (control) and block_1 (target)")
        
        ctrl_qubits = ctrl_block.get_data_qubits()
        tgt_qubits = tgt_block.get_data_qubits()
        n = min(len(ctrl_qubits), len(tgt_qubits))
        
        # Emit transversal CNOT (control → target)
        for i in range(n):
            circuit.append("CNOT", [ctrl_qubits[i], tgt_qubits[i]])
        
        circuit.append("TICK")
        
        return PhaseResult.gate_phase(
            is_final=True,
            transform=self.get_stabilizer_transform(),
        )
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """
        CNOT entangles blocks but doesn't swap X/Z.
        
        Clear history because stabilizers spread across blocks,
        but don't swap X/Z since CNOT doesn't do that.
        """
        return StabilizerTransform.identity(clear_history=True)
    
    def get_observable_transform(self) -> TwoQubitObservableTransform:
        """
        Return the two-qubit observable transform for CNOT.
        
        CNOT(ctrl→tgt) Heisenberg picture:
            X_ctrl → X_ctrl ⊗ X_tgt      (X spreads from control to target)
            Z_ctrl → Z_ctrl               (unchanged)
            X_tgt  → X_tgt                (unchanged)
            Z_tgt  → Z_ctrl ⊗ Z_tgt       (Z back-propagates from target to control)
        """
        return TwoQubitObservableTransform.cnot()
    
    def get_preparation_config(self, input_state: str) -> PreparationConfig:
        """
        Get preparation configuration for both blocks.
        
        Parameters
        ----------
        input_state : str
            Overall input state indicator (from experiment measurement_basis).
            
        Returns
        -------
        PreparationConfig
            Configuration for preparing both blocks.
        """
        blocks = {}
        
        # Control block
        ctrl_z_det = (self.control_state == "0")
        ctrl_x_det = (self.control_state == "+")
        blocks["block_0"] = BlockPreparationConfig(
            initial_state=self.control_state,
            z_deterministic=ctrl_z_det,
            x_deterministic=ctrl_x_det,
            skip_experiment_prep=False,  # Experiment handles H if needed
        )
        
        # Target block
        tgt_z_det = (self.target_state == "0")
        tgt_x_det = (self.target_state == "+")
        blocks["block_1"] = BlockPreparationConfig(
            initial_state=self.target_state,
            z_deterministic=tgt_z_det,
            x_deterministic=tgt_x_det,
            skip_experiment_prep=False,
        )
        
        return PreparationConfig(blocks=blocks)
    
    def get_crossing_detector_config(self) -> CrossingDetectorConfig:
        """
        Get crossing detector configuration for CNOT.
        
        Standard Heisenberg picture for CNOT(ctrl→tgt):
        - X_ctrl → X_ctrl ⊗ X_tgt: 3-term crossing (X spreads)
        - Z_ctrl → Z_ctrl: 2-term crossing (unchanged)
        - X_tgt → X_tgt: 2-term crossing (unchanged)
        - Z_tgt → Z_ctrl ⊗ Z_tgt: 3-term crossing (Z picks up)
        
        All four crossing detectors are unconditionally deterministic for any
        input state because the XOR algebra always evaluates to 0:
        
        For 2-term (unchanged): pre_S ⊕ post_S = s ⊕ s = 0  ✓
        For 3-term (spreads):   pre_S_a ⊕ post_S_a ⊕ post_S_b
                                = s_a ⊕ (s_a ⊕ s_b) ⊕ s_b = 0  ✓  (where S_a → S_a ⊗ S_b)
        
        Returns
        -------
        CrossingDetectorConfig
            Configuration for crossing detector emission.
        """
        formulas = []
        
        # X_ctrl: 3-term (X_ctrl → X_ctrl ⊗ X_tgt, X spreads from control to target)
        # pre_X_ctrl ⊕ post_X_ctrl ⊕ post_X_tgt = x_c ⊕ (x_c ⊕ x_t) ⊕ x_t = 0
        formulas.append(CrossingDetectorFormula(
            name="X_ctrl",
            terms=[
                CrossingDetectorTerm(block="block_0", stabilizer_type="X", timing="pre"),
                CrossingDetectorTerm(block="block_0", stabilizer_type="X", timing="post"),
                CrossingDetectorTerm(block="block_1", stabilizer_type="X", timing="post"),
            ],
        ))
        
        # Z_ctrl: 2-term (Z_ctrl → Z_ctrl, unchanged)
        # pre_Z_ctrl ⊕ post_Z_ctrl = z_c ⊕ z_c = 0
        formulas.append(CrossingDetectorFormula(
            name="Z_ctrl",
            terms=[
                CrossingDetectorTerm(block="block_0", stabilizer_type="Z", timing="pre"),
                CrossingDetectorTerm(block="block_0", stabilizer_type="Z", timing="post"),
            ],
        ))
        
        # X_tgt: 2-term (X_tgt → X_tgt, unchanged)
        # pre_X_tgt ⊕ post_X_tgt = x_t ⊕ x_t = 0
        formulas.append(CrossingDetectorFormula(
            name="X_tgt",
            terms=[
                CrossingDetectorTerm(block="block_1", stabilizer_type="X", timing="pre"),
                CrossingDetectorTerm(block="block_1", stabilizer_type="X", timing="post"),
            ],
        ))
        
        # Z_tgt: 3-term (Z_tgt → Z_ctrl ⊗ Z_tgt, Z picks up from control)
        # pre_Z_tgt ⊕ post_Z_ctrl ⊕ post_Z_tgt = z_t ⊕ z_c ⊕ (z_c ⊕ z_t) = 0
        formulas.append(CrossingDetectorFormula(
            name="Z_tgt",
            terms=[
                CrossingDetectorTerm(block="block_1", stabilizer_type="Z", timing="pre"),
                CrossingDetectorTerm(block="block_0", stabilizer_type="Z", timing="post"),
                CrossingDetectorTerm(block="block_1", stabilizer_type="Z", timing="post"),
            ],
        ))
        
        return CrossingDetectorConfig(formulas=formulas)
    
    def _get_output_bases(self) -> Dict[str, str]:
        """
        Compute per-block measurement basis from the CNOT output state.
        
        CNOT truth table:
            |00⟩ → |00⟩: ctrl=Z, tgt=Z
            |+0⟩ → |++⟩: ctrl=X, tgt=X
            |0+⟩ → |0+⟩: ctrl=Z, tgt=X
            |++⟩ → |+0⟩: ctrl=X, tgt=Z
            
        The measurement basis for each block is the basis of its OUTPUT state:
            - |0⟩ output → MZ
            - |+⟩ output → MX
            
        Returns
        -------
        Dict[str, str]
            {"block_0": "Z" or "X", "block_1": "Z" or "X"}
        """
        # Control output state: CNOT doesn't change control
        # |0⟩ → |0⟩ (MZ), |+⟩ → |+⟩ (MX)
        ctrl_basis = "X" if self.control_state == "+" else "Z"
        
        # Target output state: CNOT(|c⟩|t⟩) = |c⟩|c⊕t⟩
        # |0⟩|0⟩ → |0⟩ (MZ), |+⟩|0⟩ → |+⟩ (MX)
        # |0⟩|+⟩ → |+⟩ (MX), |+⟩|+⟩ → |0⟩ (MZ)
        if self.control_state == "0" and self.target_state == "0":
            tgt_basis = "Z"
        elif self.control_state == "+" and self.target_state == "0":
            tgt_basis = "X"
        elif self.control_state == "0" and self.target_state == "+":
            tgt_basis = "X"
        else:  # "+" and "+"
            tgt_basis = "Z"
        
        return {"block_0": ctrl_basis, "block_1": tgt_basis}
    
    def get_observable_config(self) -> ObservableConfig:
        """
        Get observable configuration based on CNOT output states.
        
        For most input states, the observable is a two-block XOR:
            Observable = B_L(ctrl) ⊕ B'_L(tgt) = 0
        where B, B' are the measurement bases of each block's output state.
        
        CNOT truth table:
            |00⟩ → |00⟩: Z_L(ctrl) ⊕ Z_L(tgt) = 0  (two-block)
            |+0⟩ → |++⟩: X_L(ctrl) ⊕ X_L(tgt) = 0  (two-block)
            |0+⟩ → |0+⟩: Z_L(ctrl) ⊕ X_L(tgt) = 0  (two-block)
            |++⟩ → |+0⟩: X_L(ctrl) = 0              (single-block)
        
        For |++⟩, the ideal two-block observable X_L(ctrl) ⊕ Z_L(tgt) is
        physically deterministic but incompatible with stim's DEM construction.
        The backward Pauli frame propagation through CNOT creates Y on both
        blocks (X_ctrl spreads → X on tgt, Z_tgt picks up → Z on ctrl,
        combined = Y on both), which anti-commutes with the RX preparation
        of both blocks. No cancellation occurs because both the X and Z
        terms spread through the CNOT.
        
        For the other three inputs, the two-block observable works because
        one term's backward propagation cancels the other's cross-block
        contribution:
        - |00⟩ ZZ: Z_ctrl backward stays, Z_tgt backward picks up Z_ctrl
          → Z_ctrl × Z_ctrl = I on ctrl → commutes with R ✓
        - |+0⟩ XX: X_ctrl backward spreads X_tgt, X_tgt backward stays
          → X_tgt × X_tgt = I on tgt → commutes with R ✓
        - |0+⟩ ZX: Z_ctrl stays on ctrl, X_tgt stays on tgt
          → no cross terms → each commutes with its own prep ✓
        
        For |++⟩ we use X_L(ctrl) alone, which backward-propagates to
        X_ctrl × X_tgt (CNOT spreads X), commuting with RX on both blocks.
        Error coverage: Z errors on either block propagate through CNOT
        and flip X_L(ctrl); X errors on ctrl flip X_L(ctrl) directly.
        Tgt-only X errors don't flip the observable but are still correctable
        via the detector graph (temporal, crossing, boundary detectors).
        
        Returns
        -------
        ObservableConfig
            Observable configuration (two-block or single-block).
        """
        bases = self._get_output_bases()
        
        # Check if the two-block observable's backward Pauli frame commutes
        # with both blocks' preparations. This fails when BOTH observable
        # terms spread through the CNOT (X on ctrl spreads, Z on tgt picks
        # up), creating Y on both blocks. This happens exactly when the
        # output bases are X(ctrl) and Z(tgt) — i.e., the |++⟩ input.
        ctrl_output_spreads = (bases["block_0"] == "X")  # X_ctrl → X_ctrl ⊗ X_tgt
        tgt_output_spreads = (bases["block_1"] == "Z")   # Z_tgt → Z_ctrl ⊗ Z_tgt
        
        if ctrl_output_spreads and tgt_output_spreads:
            # Both terms spread → backward frame is Y on both blocks.
            # Fall back to single-block observable using the ctrl block,
            # whose X_L backward frame (X on both blocks) commutes with
            # RX on both blocks.
            return ObservableConfig(
                output_blocks=["block_0"],
                block_bases=bases,
                correlation_terms=[
                    ObservableTerm(block="block_0", basis=bases["block_0"]),
                ],
            )
        
        # Standard two-block observable — backward frame cancels on one block
        return ObservableConfig(
            output_blocks=["block_0", "block_1"],
            block_bases=bases,
            correlation_terms=[
                ObservableTerm(block="block_0", basis=bases["block_0"]),
                ObservableTerm(block="block_1", basis=bases["block_1"]),
            ],
        )
    
    def get_measurement_config(self) -> MeasurementConfig:
        """
        Return per-block measurement basis derived from CNOT output states.
        
        Each block is measured in the basis of its output state, NOT a
        global measurement basis. This ensures the observable correctly
        verifies the CNOT gate.
        
        Returns
        -------
        MeasurementConfig
            Per-block measurement bases.
        """
        bases = self._get_output_bases()
        return MeasurementConfig(block_bases=bases)
    
    def get_boundary_detector_config(self) -> BoundaryDetectorConfig:
        """
        Return per-block boundary detector configuration.
        
        Boundary detectors are emitted for each block in that block's
        measurement basis (matching the output state basis).
        
        For CNOT with both blocks surviving:
            |00⟩ → ctrl: Z boundary, tgt: Z boundary
            |+0⟩ → ctrl: X boundary, tgt: X boundary
            |0+⟩ → ctrl: Z boundary, tgt: X boundary
            |++⟩ → ctrl: X boundary, tgt: Z boundary
        
        Returns
        -------
        BoundaryDetectorConfig
            Per-block boundary detector configuration with correct block names.
        """
        bases = self._get_output_bases()
        return BoundaryDetectorConfig(block_configs={
            "block_0": {
                "X": bases["block_0"] == "X",
                "Z": bases["block_0"] == "Z",
            },
            "block_1": {
                "X": bases["block_1"] == "X",
                "Z": bases["block_1"] == "Z",
            },
        })
    
    def get_space_like_detector_config(self) -> Dict[str, Optional[str]]:
        """
        Return per-block space-like detector configuration.
        
        Each block's boundary basis matches its output state basis.
        """
        bases = self._get_output_bases()
        return bases
    
    def requires_parallel_extraction(self) -> bool:
        """
        Transversal CNOT requires parallel extraction since blocks are entangled.
        
        After CNOT, Z_ctrl measurements depend on Z_tgt (and vice versa for X).
        Both blocks must be measured together for consistent detector formulas.
        """
        return True
    
    def compute_layout(self, codes: List[Code]) -> GadgetLayout:
        """
        Compute layout for two code blocks.
        
        Parameters
        ----------
        codes : List[Code]
            List containing 1 or 2 codes.
            If 1, both blocks use the same code.
            If 2, block_0 uses codes[0], block_1 uses codes[1].
            
        Returns
        -------
        GadgetLayout
            Layout with block_0 (control) and block_1 (target).
        """
        if len(codes) == 1:
            ctrl_code = tgt_code = codes[0]
        elif len(codes) == 2:
            ctrl_code, tgt_code = codes
        else:
            raise ValueError(
                f"Transversal CNOT requires 1 or 2 codes, got {len(codes)}"
            )
        
        self._code = ctrl_code
        
        layout = GadgetLayout(target_dim=2)
        
        # Control block at origin
        layout.add_block(
            name="block_0",
            code=ctrl_code,
            offset=(0.0, 0.0),
        )
        
        # Margin based on code distance (d is on the Code ABC)
        margin = max(3.0, ctrl_code.d * 1.5)
        
        # Target block offset
        layout.add_block(
            name="block_1",
            code=tgt_code,
            auto_offset=True,
            margin=margin,
        )
        
        return layout
    
    def get_metadata(self) -> GadgetMetadata:
        """
        Return metadata about the gadget.
        
        Returns
        -------
        GadgetMetadata
            Metadata including gate type, blocks, and observable structure.
        """
        return GadgetMetadata(
            gadget_type="transversal_two_qubit",
            logical_operation="CNOT",
            extra={
                "num_blocks": 2,
                "block_names": ["block_0", "block_1"],
                "control_state": self.control_state,
                "target_state": self.target_state,
                "is_teleportation": False,
                "destroyed_blocks": [],
            },
        )


# ════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FACTORY
# ════════════════════════════════════════════════════════════════════════════════

def create_transversal_cnot(
    control_state: str = "0",
    target_state: str = "0",
) -> TransversalCNOTGadget:
    """
    Create a transversal CNOT gadget.
    
    Parameters
    ----------
    control_state : str
        Initial state of control block ("0" or "+").
    target_state : str
        Initial state of target block ("0" or "+").
        
    Returns
    -------
    TransversalCNOTGadget
        Configured CNOT gadget.
    """
    return TransversalCNOTGadget(
        control_state=control_state,
        target_state=target_state,
    )
