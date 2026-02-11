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
    data_block:  Input state |ψ⟩
    bell_a:      Bell pair qubit A (half of |Φ⁺⟩)
    bell_b:      Bell pair qubit B (half of |Φ⁺⟩, carries output)

Steps:
    1. Prepare data block in |ψ⟩ (outside gadget, via experiments/preparation.py)
    2. Prepare bell_a in |+⟩ via RX, bell_b in |0⟩ via R (inside gadget, Phase 1)
    3. EC rounds on all three blocks
    4. Bell pair creation: CNOT(bell_a → bell_b)   [bell_a already |+⟩]
    5. EC rounds on all three blocks
    6. Bell measurement: CNOT(data → bell_a), MX(data), M(bell_a)
    7. Apply corrections to bell_b based on measurement outcomes
    8. Output is on bell_b

Bell State Preparation:
    |Φ⁺⟩ = (|00⟩ + |11⟩)/√2

    bell_a prepared in |+⟩ via RX (subsumes H into preparation)
    then CNOT(bell_a → bell_b) creates Bell pair from |+⟩|0⟩

    For self-dual CSS codes, this creates the logical Bell state:
    |Φ⁺⟩_L = (|0⟩_L|0⟩_L + |1⟩_L|1⟩_L)/√2

Bell Measurement:
    CNOT(data → bell_a) followed by MX(data), MZ(bell_a)

    Outcomes:
    - MX(data) = ±1 → Z correction on output
    - MZ(bell_a) = ±1 → X correction on output

════════════════════════════════════════════════════════════════════════════════
ARCHITECTURE NOTES
════════════════════════════════════════════════════════════════════════════════

This gadget is designed for AUTOMATIC detector and observable emission:
- Crossing detectors: discovered via discover_detectors()
- Boundary detectors: discovered via discover_detectors()
- Observables: discovered via discover_observables()

The gadget returns None/minimal configs for crossing/boundary/observable
to let the experiment produce a bare circuit. Auto discovery is then
applied post-hoc by the user or test harness (as in test_end_to_end.py).

Preparation architecture:
    - data_block: prepared OUTSIDE gadget (experiments/preparation.py)
    - bell_a, bell_b: prepared INSIDE gadget (RX/R in Phase 1)
    - data_block + bell_a: measured INSIDE gadget (Phase 3)
    - bell_b: measured OUTSIDE gadget (experiment final measurement)

3-phase structure (matching TeleportationGadgetMixin pattern):
    Phase 1 (PREPARATION): Prepare bell blocks: RX(bell_a) + R(bell_b), request EC
    Phase 2 (GATE):        Bell pair entangle: CNOT(bell_a → bell_b)
    Phase 3 (MEASUREMENT): Bell measurement: CNOT(data → bell_a), MX(data), M(bell_a)

════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import List, Literal, Optional, Dict, Any, Set, TYPE_CHECKING

import stim

from qectostim.codes.abstract_code import Code
from qectostim.codes.abstract_css import CSSCode
from qectostim.gadgets.base import (
    Gadget,
    GadgetMetadata,
    HeisenbergFrame,
    StabilizerTransform,
    TwoQubitObservableTransform,
    PhaseResult,
    PhaseType,
    FrameUpdate,
    ObservableConfig,
    PreparationConfig,
    BlockPreparationConfig,
    MeasurementConfig,
    BoundaryDetectorConfig,
    CrossingDetectorConfig,
)
from qectostim.gadgets.layout import GadgetLayout, QubitAllocation

if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds import DetectorContext


class KnillECGadget(Gadget):
    """
    Knill Error Correction Gadget using Bell-state teleportation.

    This gadget implements fault-tolerant error correction via teleportation.
    It uses three code blocks:
    - data_block:  Input data block (measured out)
    - bell_a:      Bell pair qubit A (measured out)
    - bell_b:      Bell pair qubit B (output)

    Protocol (3 phases):
        Phase 1 (PREPARATION): Prepare bell blocks (RX on bell_a, R on bell_b)
        Phase 2 (GATE):        Entangle Bell pair: CNOT(bell_a → bell_b)
        Phase 3 (MEASUREMENT): CNOT(data → bell_a), MX(data), M(bell_a)

    Preparation split:
        - data_block: prepared by experiment (outside gadget)
        - bell_a/bell_b: prepared by gadget internally (inside, Phase 1)

    Designed for automatic detector/observable emission.
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
        self._code: Optional[CSSCode] = None
        self._current_phase = 0

    # =========================================================================
    # Core Gadget interface
    # =========================================================================

    @property
    def gate_name(self) -> str:
        return "CNOT"

    @property
    def num_phases(self) -> int:
        """3 phases: PREPARATION → GATE → MEASUREMENT."""
        return 3

    @property
    def num_blocks(self) -> int:
        return 3

    def reset_phases(self) -> None:
        self._current_phase = 0

    def is_teleportation_gadget(self) -> bool:
        return True

    def use_auto_detectors(self) -> bool:
        """Knill EC always uses automatic detector/observable emission."""
        return True

    def get_destroyed_blocks(self) -> Set[str]:
        """data_block and bell_a are measured out."""
        return {"data_block", "bell_a"}

    def get_output_block_name(self) -> str:
        return "bell_b"

    def requires_parallel_extraction(self) -> bool:
        """All 3 blocks need coordinated syndrome extraction."""
        return True

    # =========================================================================
    # Phase emission (3-phase pattern)
    # =========================================================================

    def emit_next_phase(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """
        Emit the next phase of the Knill EC protocol.

        Phase 1 (PREPARATION): Prepare bell blocks (RX+R), request EC rounds.
        Phase 2 (GATE): Entangle Bell pair: CNOT(bell_a → bell_b).
        Phase 3 (MEASUREMENT): CNOT(data→bell_a), MX(data), M(bell_a).
        """
        self._current_phase += 1

        if self._current_phase == 1:
            return self._emit_preparation(circuit, alloc)
        elif self._current_phase == 2:
            return self._emit_bell_entangle(circuit, alloc)
        elif self._current_phase == 3:
            return self._emit_bell_measurement(circuit, alloc)
        else:
            return PhaseResult.complete()

    def _emit_preparation(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
    ) -> PhaseResult:
        """
        Phase 1: Prepare bell blocks internally and request EC rounds.

        bell_a is prepared in |+⟩ via RX (subsumes H into preparation).
        bell_b is prepared in |0⟩ via R.
        data_block is NOT prepared here — the experiment handles it.
        """
        bell_a = alloc.get_block("bell_a")
        bell_b = alloc.get_block("bell_b")

        if bell_a is None or bell_b is None:
            raise ValueError("Knill EC requires bell_a and bell_b blocks")

        qubits_a = bell_a.get_data_qubits()
        qubits_b = bell_b.get_data_qubits()

        # Prepare bell_a in |+⟩ via RX (atomic X-basis preparation)
        circuit.append("RX", qubits_a)

        # Prepare bell_b in |0⟩ via R (atomic Z-basis reset)
        circuit.append("R", qubits_b)

        circuit.append("TICK")

        return PhaseResult(
            phase_type=PhaseType.PREPARATION,
            is_final=False,
            needs_stabilizer_rounds=self.num_ec_rounds,
            stabilizer_transform=None,
            pauli_frame_update=None,
        )

    def _emit_bell_entangle(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
    ) -> PhaseResult:
        """
        Phase 2 (GATE): Bell pair entanglement.

        CNOT(bell_a → bell_b) entangles the pair.
        bell_a is already in |+⟩ from Phase 1 (RX), bell_b is |0⟩ from R.
        Result: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 across bell_a and bell_b.

        No H gates needed — the |+⟩ state is subsumed into RX preparation.
        """
        bell_a = alloc.get_block("bell_a")
        bell_b = alloc.get_block("bell_b")

        if bell_a is None or bell_b is None:
            raise ValueError("Knill EC requires bell_a and bell_b blocks")

        qubits_a = bell_a.get_data_qubits()
        qubits_b = bell_b.get_data_qubits()
        n = min(len(qubits_a), len(qubits_b))

        # CNOT(bell_a → bell_b) creates Bell pair from |+⟩|0⟩
        for i in range(n):
            circuit.append("CNOT", [qubits_a[i], qubits_b[i]])
        circuit.append("TICK")

        return PhaseResult.gate_phase(
            is_final=False,
            transform=self.get_stabilizer_transform(),
            needs_stabilizer_rounds=self.num_ec_rounds,
        )

    def _emit_bell_measurement(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
    ) -> PhaseResult:
        """
        Phase 3 (MEASUREMENT): Bell measurement.

        CNOT(data → bell_a), then MX(data), M(bell_a).
        Teleports the data state to bell_b with Pauli corrections.
        """
        data_block = alloc.get_block("data_block")
        bell_a = alloc.get_block("bell_a")

        if data_block is None or bell_a is None:
            raise ValueError("Knill EC requires data_block and bell_a")

        data_qubits = data_block.get_data_qubits()
        qubits_a = bell_a.get_data_qubits()
        n = min(len(data_qubits), len(qubits_a))

        # CNOT(data → bell_a)
        for i in range(n):
            circuit.append("CNOT", [data_qubits[i], qubits_a[i]])
        circuit.append("TICK")

        # MX on data (Bell measurement, X-basis)
        circuit.append("MX", data_qubits[:n])

        # M on bell_a (Z-basis)
        circuit.append("M", qubits_a[:n])

        circuit.append("TICK")

        total_meas = 2 * n

        # Frame update: teleportation corrections propagate to bell_b
        frame_update = FrameUpdate(
            block_name="bell_b",
            x_meas=list(range(n)),              # MX(data) outcomes
            z_meas=list(range(n, total_meas)),   # M(bell_a) outcomes
            teleport=True,
            source_block="data_block",
        )

        return PhaseResult(
            phase_type=PhaseType.MEASUREMENT,
            is_final=True,
            needs_stabilizer_rounds=0,
            stabilizer_transform=None,
            destroyed_blocks={"data_block", "bell_a"},
            pauli_frame_update=frame_update,
            measurement_count=total_meas,
        )

    # =========================================================================
    # Stabilizer / observable transforms
    # =========================================================================

    def get_stabilizer_transform(self) -> StabilizerTransform:
        """
        Overall stabilizer transform for Knill EC.

        Output on bell_b has the same stabilizer structure as the input
        (modulo Pauli corrections), so this is a teleportation identity.
        """
        return StabilizerTransform.teleportation(swap_xz=False)

    def get_observable_transform(self) -> TwoQubitObservableTransform:
        """
        Observable transform for Knill EC.

        Teleportation passes observable through unchanged (corrections
        handled by Pauli frame).
        """
        return TwoQubitObservableTransform.identity()

    # =========================================================================
    # Configuration methods
    # =========================================================================

    def get_preparation_config(self, input_state: str = "0") -> PreparationConfig:
        """
        Preparation config for all three blocks.

        - data_block: prepared OUTSIDE gadget (experiments/preparation.py)
        - bell_a: prepared INSIDE gadget in |+⟩ via RX (Phase 1)
        - bell_b: prepared INSIDE gadget in |0⟩ via R (Phase 1)
        """
        actual_input = self.input_state

        data_z_det = (actual_input == "0")
        data_x_det = (actual_input == "+")

        blocks = {
            "data_block": BlockPreparationConfig(
                initial_state=actual_input,
                z_deterministic=data_z_det,
                x_deterministic=data_x_det,
                skip_experiment_prep=False,  # Experiment prepares data block
            ),
            "bell_a": BlockPreparationConfig(
                initial_state="+",       # Prepared in |+⟩ via RX inside gadget
                z_deterministic=False,
                x_deterministic=True,    # |+⟩ → X stabilizers deterministic
                skip_experiment_prep=True,  # Gadget prepares bell_a
            ),
            "bell_b": BlockPreparationConfig(
                initial_state="0",       # Prepared in |0⟩ via R inside gadget
                z_deterministic=True,    # |0⟩ → Z stabilizers deterministic
                x_deterministic=False,
                skip_experiment_prep=True,  # Gadget prepares bell_b
            ),
        }

        return PreparationConfig(blocks=blocks)

    def get_measurement_config(self) -> MeasurementConfig:
        """
        Final measurement config.

        Only bell_b survives — the experiment measures it.
        data_block and bell_a are destroyed during the gadget.
        """
        basis = "X" if self.input_state == "+" else "Z"
        return MeasurementConfig(
            block_bases={"bell_b": basis},
            destroyed_blocks={"data_block", "bell_a"},
        )

    def get_observable_config(self) -> ObservableConfig:
        """
        Observable configuration via universal Heisenberg derivation.

        Uses ``HeisenbergFrame.knill_ec()`` to automatically derive
        the correct observable for any input state.

        Knill EC output (auto-derived)::

            |0⟩: Z_L(bell_b)   (eigenstate determinism)
            |+⟩: X_L(bell_b)   (trivially deterministic)
        """
        frame = HeisenbergFrame.knill_ec()
        return ObservableConfig.from_heisenberg(frame, self.input_state)

    def get_crossing_detector_config(self) -> Optional[CrossingDetectorConfig]:
        """
        Return None — auto detection via discover_detectors() handles this.

        The Knill EC protocol has complex crossing detector requirements
        across 3 blocks, which are better discovered automatically than
        manually specified.
        """
        return None

    def get_boundary_detector_config(self) -> BoundaryDetectorConfig:
        """
        Boundary detector config for surviving block (bell_b).

        Only bell_b survives; data_block and bell_a are destroyed.
        """
        basis = "X" if self.input_state == "+" else "Z"
        return BoundaryDetectorConfig(
            block_configs={
                "data_block": {"X": False, "Z": False},  # Destroyed
                "bell_a": {"X": False, "Z": False},       # Destroyed
                "bell_b": {
                    "X": basis == "X",
                    "Z": basis == "Z",
                },
            },
        )

    def get_space_like_detector_config(self) -> Dict[str, Optional[str]]:
        """Per-block space-like detector config."""
        basis = "X" if self.input_state == "+" else "Z"
        return {
            "data_block": None,   # Destroyed
            "bell_a": None,       # Destroyed
            "bell_b": basis,      # Output block
        }

    # =========================================================================
    # Teleportation-specific interface methods
    # =========================================================================

    def get_x_stabilizer_mode(self) -> str:
        """Use CX for X stabilizer measurement (matches teleportation pattern)."""
        return "cx"

    def get_blocks_to_skip_preparation(self) -> Set[str]:
        """Bell blocks are prepared inside the gadget (Phase 1)."""
        return {"bell_a", "bell_b"}

    def get_blocks_to_skip_pre_rounds(self) -> Set[str]:
        """Bell blocks skip pre-gadget rounds (prepared inside gadget)."""
        return {"bell_a", "bell_b"}

    def get_blocks_to_skip_post_rounds(self) -> Set[str]:
        """All blocks participate in post-gadget EC rounds."""
        return set()

    def get_ancilla_block_names(self) -> Set[str]:
        """Return the Bell pair block names."""
        return {"bell_a", "bell_b"}

    def get_initial_state_for_block(self, block_name: str, requested_state: str) -> str:
        """Initial state per block."""
        if block_name == "bell_a":
            return "+"  # Prepared in |+⟩ via RX inside gadget
        if block_name == "bell_b":
            return "0"  # Prepared in |0⟩ via R inside gadget
        return requested_state

    def should_skip_state_preparation(self) -> bool:
        return False

    def should_emit_space_like_detectors(self) -> bool:
        return True

    # =========================================================================
    # Layout
    # =========================================================================

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
            Layout with data_block, bell_a, bell_b.
        """
        if len(codes) != 1:
            raise ValueError(
                f"Knill EC requires exactly 1 code (for all blocks), got {len(codes)}"
            )

        code = codes[0]
        self._code = code

        layout = GadgetLayout(target_dim=2)

        margin = max(3.0, code.d * 1.5)

        # data_block at origin
        layout.add_block(
            name="data_block",
            code=code,
            offset=(0.0, 0.0),
        )

        # bell_a to the right
        layout.add_block(
            name="bell_a",
            code=code,
            auto_offset=True,
            margin=margin,
        )

        # bell_b further right
        layout.add_block(
            name="bell_b",
            code=code,
            auto_offset=True,
            margin=margin,
        )

        return layout

    def get_metadata(self) -> GadgetMetadata:
        return GadgetMetadata(
            gadget_type="teleportation_ec",
            logical_operation="KnillEC",
            extra={
                "num_blocks": 3,
                "block_names": ["data_block", "bell_a", "bell_b"],
                "input_state": self.input_state,
                "is_teleportation": True,
                "destroyed_blocks": ["data_block", "bell_a"],
                "output_block": "bell_b",
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
