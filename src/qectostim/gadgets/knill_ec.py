#!/usr/bin/env python3
"""
Knill Error Correction Gadget for FaultTolerantGadgetExperiment.

════════════════════════════════════════════════════════════════════════════════
OVERVIEW
════════════════════════════════════════════════════════════════════════════════

Knill Error Correction (Knill EC) is a teleportation-based error correction
scheme that uses Bell-state measurements to detect and correct errors while
teleporting the logical state to a fresh ancilla block.

This is distinct from standard stabilizer-based EC and provides:
- Fault-tolerant error correction via teleportation
- Built-in state transfer to fresh qubits
- Natural interface for fault-tolerant circuits

════════════════════════════════════════════════════════════════════════════════
PROTOCOL
════════════════════════════════════════════════════════════════════════════════

Three-block protocol:
    Block 0: Data block (input state |ψ⟩)
    Block 1: Bell pair qubit A (half of |Φ⁺⟩)
    Block 2: Bell pair qubit B (half of |Φ⁺⟩)

Steps:
    1. Prepare data block in |ψ⟩
    2. Prepare Bell pair |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 across blocks 1 and 2
    3. EC rounds on all three blocks
    4. Bell measurement: CNOT(data→Bell_A), then measure data(X), Bell_A(Z)
    5. Apply corrections to Bell_B based on measurement outcomes
    6. Output is on Block 2 (Bell_B)

Bell State Preparation:
    |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    
    Transversal: H on block_1, then CNOT(block_1 → block_2)
    
    For self-dual CSS codes, this creates the logical Bell state:
    |Φ⁺⟩_L = (|0⟩_L|0⟩_L + |1⟩_L|1⟩_L)/√2

Bell Measurement:
    CNOT(data → bell_A) followed by MX(data), MZ(bell_A)
    
    Outcomes:
    - MX(data) = ±1 → Z correction on output
    - MZ(bell_A) = ±1 → X correction on output

════════════════════════════════════════════════════════════════════════════════
STABILIZER TRANSFORMATIONS
════════════════════════════════════════════════════════════════════════════════

Bell Prep (H on block_1, CNOT(1→2)):
    After H: X_1 ↔ Z_1
    After CNOT: 
        X_1 → X_1
        Z_1 → Z_1 ⊗ Z_2
        X_2 → X_1 ⊗ X_2
        Z_2 → Z_2

    Combined stabilizer: X_1 ⊗ X_2 and Z_1 ⊗ Z_2 are the Bell stabilizers
    
Bell Measurement (CNOT(0→1), MX(0), MZ(1)):
    Before measurement:
        X_0 → X_0              (unchanged)
        Z_0 → Z_0 ⊗ Z_1        (spreads to bell_A)
        X_1 → X_0 ⊗ X_1        (picks up data)
        Z_1 → Z_1              (unchanged)
        
    After measuring MX(0), MZ(1):
        - Block 0 destroyed (measured out)
        - Block 1 destroyed (measured out)
        - Block 2 retains output with Pauli frame corrections

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
    TwoQubitObservableTransform,
    PhaseResult,
    PhaseType,
    ObservableConfig,
    PreparationConfig,
    BlockPreparationConfig,
    CrossingDetectorConfig,
    CrossingDetectorFormula,
    CrossingDetectorTerm,
)
from qectostim.gadgets.layout import GadgetLayout, QubitAllocation

if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds import DetectorContext


class KnillECGadget(Gadget):
    """
    Knill Error Correction Gadget using Bell-state teleportation.
    
    This gadget implements fault-tolerant error correction via teleportation.
    It uses three code blocks:
    - Block 0: Input data block
    - Block 1: Bell pair qubit A (measured out)
    - Block 2: Bell pair qubit B (output)
    
    Protocol:
        1. Prepare Bell pair across blocks 1 and 2
        2. Bell measurement: CNOT(0→1), MX(0), MZ(1)
        3. Apply corrections to block 2 based on outcomes
        4. Output on block 2
    
    This is a 2-phase gadget:
        Phase 1: Bell pair preparation (H on block_1, CNOT(1→2))
        Phase 2: Bell measurement and teleportation
    """
    
    def __init__(
        self,
        input_state: Literal["0", "+"] = "0",
        num_ec_rounds: int = 1,
    ):
        """
        Initialize Knill EC gadget.
        
        Parameters
        ----------
        input_state : Literal["0", "+"]
            Input logical state. "0" for |0⟩_L, "+" for |+⟩_L.
        num_ec_rounds : int
            Number of EC rounds between phases.
        """
        super().__init__(input_state=input_state)
        self.input_state = input_state
        self.num_ec_rounds = num_ec_rounds
        
        # Cache
        self._code: Optional[CSSCode] = None
        self._current_phase = 0
    
    # =========================================================================
    # Gadget interface implementation
    # =========================================================================
    
    @property
    def gate_name(self) -> str:
        """Knill EC uses transversal CNOT for Bell measurement."""
        return "CNOT"
    
    @property
    def num_phases(self) -> int:
        """Knill EC has 2 phases: Bell prep and Bell measurement."""
        return 2
    
    @property
    def num_blocks(self) -> int:
        """Three blocks: data, bell_A, bell_B."""
        return 3
    
    def reset_phases(self) -> None:
        """Reset phase counter for new circuit generation."""
        self._current_phase = 0
    
    def is_teleportation_gadget(self) -> bool:
        """Yes - this is a teleportation gadget."""
        return True
    
    def get_destroyed_blocks(self) -> List[str]:
        """Blocks 0 and 1 are destroyed (measured out)."""
        return ["block_0", "block_1"]
    
    def get_output_block_name(self) -> str:
        """Output is on block 2 (Bell pair qubit B)."""
        return "block_2"
    
    def emit_next_phase(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """
        Emit the next phase of the Knill EC protocol.
        
        Phase 1: Bell pair preparation
        Phase 2: Bell measurement and teleportation
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        alloc : QubitAllocation
            Qubit allocation with block_0, block_1, block_2.
        ctx : DetectorContext
            Detector context for Pauli frame tracking.
            
        Returns
        -------
        PhaseResult
            Result from the phase.
        """
        self._current_phase += 1
        
        if self._current_phase == 1:
            return self._emit_bell_preparation(circuit, alloc, ctx)
        elif self._current_phase == 2:
            return self._emit_bell_measurement(circuit, alloc, ctx)
        else:
            return PhaseResult.complete()
    
    def _emit_bell_preparation(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """
        Emit Bell pair preparation: H on block_1, CNOT(1→2).
        
        Creates |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 across blocks 1 and 2.
        """
        block_1 = alloc.get_block("block_1")
        block_2 = alloc.get_block("block_2")
        
        if block_1 is None or block_2 is None:
            raise ValueError("Knill EC requires block_1 and block_2")
        
        qubits_1 = block_1.get_data_qubits()
        qubits_2 = block_2.get_data_qubits()
        n = min(len(qubits_1), len(qubits_2))
        
        # H on block_1 (creates |+⟩ from |0⟩)
        circuit.append("H", qubits_1[:n])
        circuit.append("TICK")
        
        # CNOT(block_1 → block_2) creates Bell pair
        for i in range(n):
            circuit.append("CNOT", [qubits_1[i], qubits_2[i]])
        circuit.append("TICK")
        
        # After Bell prep, blocks 1 and 2 are entangled.
        # X_1, X_2 become joint stabilizer X_1 ⊗ X_2
        # Z_1, Z_2 become joint stabilizer Z_1 ⊗ Z_2
        
        return PhaseResult(
            phase_type=PhaseType.PREPARATION,
            is_final=False,
            needs_stabilizer_rounds=self.num_ec_rounds,
            stabilizer_transform=StabilizerTransform.identity(clear_history=True),
        )
    
    def _emit_bell_measurement(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """
        Emit Bell measurement: CNOT(0→1), MX(0), MZ(1).
        
        Teleports the data state to block_2 with Pauli corrections.
        Uses MX directly instead of H+M for cleaner determinism.
        """
        block_0 = alloc.get_block("block_0")
        block_1 = alloc.get_block("block_1")
        
        if block_0 is None or block_1 is None:
            raise ValueError("Knill EC requires block_0 and block_1")
        
        qubits_0 = block_0.get_data_qubits()
        qubits_1 = block_1.get_data_qubits()
        n = min(len(qubits_0), len(qubits_1))
        
        # CNOT(data → bell_A)
        for i in range(n):
            circuit.append("CNOT", [qubits_0[i], qubits_1[i]])
        circuit.append("TICK")
        
        # Measure data in X-basis directly (MX) - cleaner than H+M
        circuit.append("MX", qubits_0[:n])
        
        # Measure bell_A in Z-basis
        circuit.append("M", qubits_1[:n])
        
        circuit.append("TICK")
        
        # Track measurements for Pauli frame
        total_meas = 2 * n
        all_measured = qubits_0[:n] + qubits_1[:n]
        
        return PhaseResult(
            phase_type=PhaseType.MEASUREMENT,
            is_final=True,
            measurement_count=total_meas,
            measurement_qubits=all_measured,
            measured_blocks=["block_0", "block_1"],
            pauli_frame_update={
                "block_name": "block_2",
                "x_meas": list(range(n)),  # MX outcomes for Z correction
                "z_meas": list(range(n, total_meas)),  # MZ outcomes for X correction
                "teleport": True,
            },
        )
    
    def get_stabilizer_transform(self) -> StabilizerTransform:
        """
        Overall stabilizer transform for Knill EC.
        
        The output block has the same stabilizer structure as input
        (modulo Pauli corrections), so no swap_xz needed.
        """
        return StabilizerTransform.teleportation(swap_xz=False)
    
    def get_observable_transform(self) -> TwoQubitObservableTransform:
        """
        Return the observable transform for Knill EC.
        
        For teleportation, the observable transforms depend on the
        combined Bell measurement and correction process. Simplified
        here as identity since output has same logical content as input.
        """
        # Knill EC is teleportation - observable passes through unchanged
        # The frame corrections handle the actual transform
        return TwoQubitObservableTransform.identity()
    
    def get_preparation_config(self, input_state: str) -> PreparationConfig:
        """
        Get preparation configuration for all three blocks.
        
        - Block 0: Data block, prepared based on input_state
        - Block 1: Bell pair A, prepared in |0⟩ (will get H for Bell prep)
        - Block 2: Bell pair B, prepared in |0⟩
        """
        blocks = {}
        
        # Data block
        data_z_det = (self.input_state == "0")
        data_x_det = (self.input_state == "+")
        blocks["block_0"] = BlockPreparationConfig(
            initial_state=self.input_state,
            z_deterministic=data_z_det,
            x_deterministic=data_x_det,
            skip_experiment_prep=False,
        )
        
        # Bell pair block A - starts in |0⟩, gets H during gadget
        blocks["block_1"] = BlockPreparationConfig(
            initial_state="0",
            z_deterministic=True,
            x_deterministic=False,
            skip_experiment_prep=True,  # Gadget handles Bell prep
        )
        
        # Bell pair block B - starts in |0⟩
        blocks["block_2"] = BlockPreparationConfig(
            initial_state="0",
            z_deterministic=True,
            x_deterministic=False,
            skip_experiment_prep=True,  # Part of Bell prep
        )
        
        return PreparationConfig(blocks=blocks)
    
    def get_crossing_detector_config(self) -> CrossingDetectorConfig:
        """
        Get crossing detector configuration for Knill EC.
        
        Knill EC has TWO entangling operations that need crossing detectors:
        
        1. Bell Prep (H on block_1, CNOT(1→2)):
           - X_1: H transforms to Z_1, then CNOT keeps it Z_1
           - Z_1: H transforms to X_1, CNOT spreads to X_1 ⊗ X_2
           - X_2: CNOT spreads to X_1 ⊗ X_2
           - Z_2: Unchanged
           
        2. Bell Measurement (CNOT(0→1)):
           - X_0: Unchanged
           - Z_0: Spreads to Z_0 ⊗ Z_1
           - X_1: Spreads to X_0 ⊗ X_1
           - Z_1: Unchanged
           
        For the Bell measurement CNOT(0→1):
        - X_0: 2-term (unchanged)
        - Z_0: 3-term (spreads to Z_1)
        - X_1: 3-term (picks up X_0)
        - Z_1: 2-term (unchanged)
        
        Note: block_1 stabilizers after Bell prep are joint (X_1⊗X_2, Z_1⊗Z_2).
        The Bell measurement CNOT further transforms these correlations.
        """
        formulas = []
        
        # For the Bell measurement CNOT(data=block_0 → bell_A=block_1):
        
        # X_0 (data X): 2-term, unchanged through CNOT
        formulas.append(CrossingDetectorFormula(
            name="X_data",
            terms=[
                CrossingDetectorTerm(block="block_0", stabilizer_type="X", timing="pre"),
                CrossingDetectorTerm(block="block_0", stabilizer_type="X", timing="post"),
            ],
        ))
        
        # Z_0 (data Z): 3-term, spreads to bell_A (block_1)
        # Z_0(pre) ⊕ Z_0(post) ⊕ Z_1(post)
        formulas.append(CrossingDetectorFormula(
            name="Z_data",
            terms=[
                CrossingDetectorTerm(block="block_0", stabilizer_type="Z", timing="pre"),
                CrossingDetectorTerm(block="block_0", stabilizer_type="Z", timing="post"),
                CrossingDetectorTerm(block="block_1", stabilizer_type="Z", timing="post"),
            ],
        ))
        
        # X_1 (bell_A X): 3-term, picks up data X
        # After Bell prep, X_1 is part of joint stabilizer X_1⊗X_2
        # Through CNOT: X_1 → X_0 ⊗ X_1
        # So: X_1(pre) ⊕ X_0(post) ⊕ X_1(post)
        formulas.append(CrossingDetectorFormula(
            name="X_bellA",
            terms=[
                CrossingDetectorTerm(block="block_1", stabilizer_type="X", timing="pre"),
                CrossingDetectorTerm(block="block_0", stabilizer_type="X", timing="post"),
                CrossingDetectorTerm(block="block_1", stabilizer_type="X", timing="post"),
            ],
        ))
        
        # Z_1 (bell_A Z): 2-term, unchanged through CNOT
        # (Note: after Bell prep, Z_1 is part of joint Z_1⊗Z_2, but the local
        # measurement on block_1 still gives a deterministic crossing detector)
        formulas.append(CrossingDetectorFormula(
            name="Z_bellA",
            terms=[
                CrossingDetectorTerm(block="block_1", stabilizer_type="Z", timing="pre"),
                CrossingDetectorTerm(block="block_1", stabilizer_type="Z", timing="post"),
            ],
        ))
        
        # Block 2 (bell_B): No crossing detectors needed (not involved in CNOT)
        # But for completeness, emit temporal detectors
        formulas.append(CrossingDetectorFormula(
            name="X_bellB",
            terms=[
                CrossingDetectorTerm(block="block_2", stabilizer_type="X", timing="pre"),
                CrossingDetectorTerm(block="block_2", stabilizer_type="X", timing="post"),
            ],
        ))
        
        formulas.append(CrossingDetectorFormula(
            name="Z_bellB",
            terms=[
                CrossingDetectorTerm(block="block_2", stabilizer_type="Z", timing="pre"),
                CrossingDetectorTerm(block="block_2", stabilizer_type="Z", timing="post"),
            ],
        ))
        
        return CrossingDetectorConfig(formulas=formulas)
    
    def get_observable_config(self) -> ObservableConfig:
        """
        Get observable configuration for Knill EC.
        
        The output is on block_2. The observable depends on input state
        and measurement basis, with frame corrections from the Bell measurement.
        
        |0⟩ input → measure Z on block_2, no frame correction needed
        |+⟩ input → measure X on block_2, Z frame from MX(block_0)
        """
        return ObservableConfig.bell_teleportation(
            output_block="block_2",
            bell_blocks=["block_0", "block_1"],
            frame_basis="XZ",
        )
    
    def get_space_like_detector_config(self) -> Dict[str, Optional[str]]:
        """
        Return per-block space-like detector configuration.
        
        - Blocks 0 and 1 are destroyed, so no space-like detectors
        - Block 2 survives, emit based on measurement basis
        """
        if self.input_state == "0":
            # |0⟩ input → measure Z on output
            return {
                "block_0": None,  # Destroyed
                "block_1": None,  # Destroyed
                "block_2": "Z",   # Output measured in Z
            }
        else:
            # |+⟩ input → measure X on output
            return {
                "block_0": None,  # Destroyed
                "block_1": None,  # Destroyed
                "block_2": "X",   # Output measured in X
            }
    
    def requires_parallel_extraction(self) -> bool:
        """
        Knill EC requires parallel extraction due to Bell entanglement.
        
        After Bell prep, blocks 1 and 2 are entangled. After Bell measurement,
        the corrections depend on joint outcomes.
        """
        return True
    
    def compute_layout(self, codes: List[Code]) -> GadgetLayout:
        """
        Compute layout for three code blocks.
        
        Parameters
        ----------
        codes : List[Code]
            List containing 1 code (used for all three blocks).
            
        Returns
        -------
        GadgetLayout
            Layout with block_0, block_1, block_2.
        """
        if len(codes) != 1:
            raise ValueError(
                f"Knill EC requires exactly 1 code (for all blocks), got {len(codes)}"
            )
        
        code = codes[0]
        self._code = code
        
        layout = GadgetLayout(target_dim=2)
        
        # Margin based on code distance
        margin = max(3.0, code.d * 1.5)
        
        # Block 0 (data) at origin
        layout.add_block(
            name="block_0",
            code=code,
            offset=(0.0, 0.0),
        )
        
        # Block 1 (Bell A) to the right
        layout.add_block(
            name="block_1",
            code=code,
            auto_offset=True,
            margin=margin,
        )
        
        # Block 2 (Bell B, output) further right
        layout.add_block(
            name="block_2",
            code=code,
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
            Metadata including gate type, blocks, and protocol info.
        """
        return GadgetMetadata(
            gadget_type="teleportation_ec",
            logical_operation="KnillEC",
            extra={
                "num_blocks": 3,
                "block_names": ["block_0", "block_1", "block_2"],
                "input_state": self.input_state,
                "is_teleportation": True,
                "destroyed_blocks": ["block_0", "block_1"],
                "output_block": "block_2",
            },
        )


# ════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FACTORY
# ════════════════════════════════════════════════════════════════════════════════

def create_knill_ec(
    input_state: str = "0",
    num_ec_rounds: int = 1,
) -> KnillECGadget:
    """
    Create a Knill Error Correction gadget.
    
    Parameters
    ----------
    input_state : str
        Initial state of data block ("0" or "+").
    num_ec_rounds : int
        Number of EC rounds between phases.
        
    Returns
    -------
    KnillECGadget
        Configured Knill EC gadget.
    """
    return KnillECGadget(
        input_state=input_state,
        num_ec_rounds=num_ec_rounds,
    )
