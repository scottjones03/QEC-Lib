#!/usr/bin/env python3
"""
CSS Surgery CNOT Gadget -- 5-phase lattice-surgery protocol.

Implements a logical CNOT between two CSS code blocks via measurement-based
lattice surgery using a mediating ancilla code block.

Protocol (5 phases):
    Phase 1 (ZZ merge):   d rounds of bridge Z-parity + interleaved EC on all blocks.
    Phase 2 (ZZ split):   Destructive M on ZZ bridge qubits -> s_zz.
    Phase 3 (XX merge):   d rounds of bridge X-parity + interleaved EC (X-only on blocks 1,2).
    Phase 4 (XX split):   Destructive MX on XX bridge qubits -> s_xx.
    Phase 5 (Anc MX):     MX on ancilla data -> project out block_1.

CRITICAL: This gadget uses set_builders() to receive block builders from the
orchestrator, then calls builder.emit_round() directly during merge phases to
achieve proper interleaving of bridge measurements and EC rounds. This is
necessary because bridge operations and EC must happen in the SAME round, not
sequentially.

Observable Corrections:
    |00> -> |00>:  Z(ctrl) XOR Z(tgt) XOR s_zz
    |+0> -> |++>:  X(ctrl)             XOR m_anc_XL
    |0+> -> |0+>:  Z(ctrl) XOR X(tgt)  (no correction - both commute)
    |++> -> |+0>:  X(ctrl)             XOR m_anc_XL
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Any,
    TYPE_CHECKING,
)

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
    FrameUpdate,
    CrossingDetectorConfig,
    CrossingDetectorFormula,
    CrossingDetectorTerm,
    HeisenbergFrame,
    ObservableConfig,
    ObservableTerm,
    PreparationConfig,
    BlockPreparationConfig,
    MeasurementConfig,
    BoundaryDetectorConfig,
)
from qectostim.gadgets.layout import (
    GadgetLayout,
    QubitAllocation,
    BlockAllocation,
)

if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds import DetectorContext
    from qectostim.experiments.stabilizer_rounds.css_builder import CSSStabilizerRoundBuilder

from qectostim.experiments.stabilizer_rounds.base import StabilizerBasis


class CSSSurgeryCNOTGadget(Gadget):
    """
    CSS lattice-surgery CNOT gadget using a 5-phase merge/split protocol.
    
    The gadget receives block builders via set_builders() and directly calls
    builder.emit_round() during merge phases to achieve proper interleaving.
    """

    @property
    def gate_name(self) -> str:
        """Return SURGERY_CNOT to prevent auto frame propagation."""
        return "SURGERY_CNOT"

    def __init__(
        self,
        control_state: Literal["0", "+"] = "0",
        target_state: Literal["0", "+"] = "0",
        merge_rounds: Optional[int] = None,
    ):
        super().__init__()
        self.control_state = control_state
        self.target_state = target_state
        self.merge_rounds = merge_rounds

        # Caches
        self._code: Optional[CSSCode] = None
        self._current_phase: int = 0
        self._builders: Optional[List["CSSStabilizerRoundBuilder"]] = None
        
        # Bridge measurement tracking
        self._zz_bridge_meas: List[List[int]] = []
        self._xx_bridge_meas: List[List[int]] = []
        self._zz_split_meas: List[int] = []
        self._xx_split_meas: List[int] = []
        
        # Ancilla MX tracking
        self._anc_meas_indices: List[int] = []
        
        # Data offsets for observable correction
        self._ctrl_data_offset: int = 0
        self._anc_data_offset: int = 0
        self._tgt_data_offset: int = 0

    # ------------------------------------------------------------------
    # Builder injection (called by orchestrator)
    # ------------------------------------------------------------------

    def set_builders(self, builders: List["CSSStabilizerRoundBuilder"]) -> None:
        """
        Receive block builders from orchestrator for interleaved EC emission.
        
        The builders are ordered: [block_0, block_1, block_2].
        """
        self._builders = builders

    # ------------------------------------------------------------------
    # Gadget ABC
    # ------------------------------------------------------------------

    @property
    def num_phases(self) -> int:
        return 5

    @property
    def num_blocks(self) -> int:
        return 3

    def reset_phases(self) -> None:
        self._current_phase = 0
        self._zz_bridge_meas.clear()
        self._xx_bridge_meas.clear()
        self._zz_split_meas.clear()
        self._xx_split_meas.clear()
        self._anc_meas_indices.clear()

    def is_teleportation_gadget(self) -> bool:
        return False

    def requires_three_blocks(self) -> bool:
        return True

    def requires_parallel_extraction(self) -> bool:
        return True

    @property
    def required_code_properties(self) -> set:
        """Surgery CNOT requires CSS codes for separate X/Z merge rounds."""
        return {"css"}

    # ------------------------------------------------------------------
    # emit_next_phase
    # ------------------------------------------------------------------

    def emit_next_phase(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        self._current_phase += 1
        
        if self._code is None:
            raise RuntimeError("compute_layout() must be called first")
        if self._builders is None:
            raise RuntimeError("set_builders() must be called first")

        ctrl = alloc.get_block("block_0")
        anc = alloc.get_block("block_1")
        tgt = alloc.get_block("block_2")
        
        # Cache data offsets
        self._ctrl_data_offset = ctrl.get_data_qubits()[0]
        self._anc_data_offset = anc.get_data_qubits()[0]
        self._tgt_data_offset = tgt.get_data_qubits()[0]

        if self._current_phase == 1:
            return self._emit_zz_merge(circuit, alloc, ctx, ctrl, anc, tgt)
        elif self._current_phase == 2:
            return self._emit_zz_split(circuit, alloc, ctx)
        elif self._current_phase == 3:
            return self._emit_xx_merge(circuit, alloc, ctx, ctrl, anc, tgt)
        elif self._current_phase == 4:
            return self._emit_xx_split(circuit, alloc, ctx)
        else:
            return self._emit_ancilla_measurement(circuit, alloc, ctx, anc)

    # ------------------------------------------------------------------
    # Phase 1: ZZ Merge
    # ------------------------------------------------------------------

    def _emit_zz_merge(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
        ctrl: BlockAllocation,
        anc: BlockAllocation,
        tgt: BlockAllocation,
    ) -> PhaseResult:
        """
        ZZ merge: d rounds of bridge Z-parity + interleaved EC on all blocks.
        
        Couples control (block_0) and ancilla (block_1) to measure Z_ctrl ⊗ Z_anc.
        
        All stabilizers (X and Z) are preserved during ZZ merge because CSS
        stabilizer weights are even, ensuring X dressing from CNOT(data→bridge)
        cancels at the stabilizer level. Full EC (BOTH types) with detectors.
        
        Each round:
            1. R(bridge_zz)
            2. CX(ctrl_data[support] -> bridge)  [ctrl is CONTROL]
            3. CX(anc_data[support] -> bridge)   [anc is CONTROL]
            4. M(bridge_zz)
            5. EC round on all 3 blocks (BOTH X+Z, full detectors)
        
        Bridge temporal detectors for round >= 1.
        Block anchors based on prep state (round 0 only).
        """
        code = self._code
        d = self.merge_rounds if self.merge_rounds is not None else code.d

        # Get bridge qubit indices
        bridge_zz = [
            gi for gi, _, purpose in alloc.bridge_ancillas
            if purpose.startswith("zz_merge")
        ]
        n_z = len(bridge_zz)

        # Z-stabiliser supports from code.hz
        # Use min(n_z, hz.shape[0]) to handle boundary-based layouts
        n_stab = min(n_z, code.hz.shape[0])
        z_supports = []
        for s_idx in range(n_stab):
            z_supports.append(list(np.where(code.hz[s_idx])[0]))
        # If we have more bridges than stabilisers, use single-qubit supports
        # (boundary layout case)
        while len(z_supports) < n_z:
            z_supports.append([])  # Will be handled by boundary pairs below

        ctrl_data = ctrl.get_data_qubits()
        anc_data = anc.get_data_qubits()

        # Get prep config for anchor decisions
        prep = self.get_preparation_config()
        
        self._zz_bridge_meas.clear()

        for rnd in range(d):
            # --- Bridge ZZ measurement ---
            circuit.append("R", bridge_zz)
            circuit.append("TICK")

            # CX ctrl_data[support] -> bridge (ctrl is CONTROL)
            for b_idx in range(n_z):
                if b_idx < len(z_supports) and z_supports[b_idx]:
                    for local_idx in z_supports[b_idx]:
                        circuit.append("CNOT", [ctrl_data[local_idx], bridge_zz[b_idx]])
            circuit.append("TICK")

            # CX anc_data[support] -> bridge (anc is CONTROL)
            # ZZ merge couples control and ancilla to measure Z_ctrl ⊗ Z_anc
            for b_idx in range(n_z):
                if b_idx < len(z_supports) and z_supports[b_idx]:
                    for local_idx in z_supports[b_idx]:
                        circuit.append("CNOT", [anc_data[local_idx], bridge_zz[b_idx]])
            circuit.append("TICK")

            # Measure bridge
            meas_start = ctx.add_measurement(n_z)
            circuit.append("M", bridge_zz)
            self._zz_bridge_meas.append(list(range(meas_start, meas_start + n_z)))
            circuit.append("TICK")

            # Bridge temporal detectors (round >= 1)
            if rnd >= 1:
                for b_idx in range(n_z):
                    curr = self._zz_bridge_meas[rnd][b_idx] - ctx.measurement_index
                    prev = self._zz_bridge_meas[rnd - 1][b_idx] - ctx.measurement_index
                    # Add coordinate for DEM visualization (bridge between block_0 and block_1)
                    coord = (b_idx + 0.5, -0.5, rnd)  # Offset to distinguish from block ancillas
                    circuit.append("DETECTOR", [stim.target_rec(curr), stim.target_rec(prev)], coord)

            # --- Interleaved EC on all blocks ---
            for builder in self._builders:
                block_name = builder.block_name
                block_prep = prep.blocks.get(block_name)
                
                if block_name == "block_1":
                    # Ancilla (|+⟩ prep): CNOT(anc_data→bridge) with data as CONTROL.
                    # Individual qubit X is dressed: X_d → X_d ⊗ X_bridge, BUT
                    # at the STABILIZER level, all X stabilizers have EVEN overlap
                    # with Z-bridge supports (CSS weight parity), so X_stab is
                    # actually PRESERVED. Z is also stable (CNOT control preserves Z).
                    # |+⟩ prep: X anchors OK (R0), Z random → no Z anchor but
                    # Z temporals from R1+.
                    if rnd == 0:
                        # R0: X anchors (|+⟩ prep), Z baseline (no anchor)
                        builder.emit_round(
                            circuit,
                            emit_detectors=True,
                            emit_x_anchors=True,   # X deterministic from |+⟩
                            emit_z_anchors=False,   # Z random from |+⟩
                            explicit_anchor_mode=True,
                        )
                    else:
                        # R1+: BOTH temporals (X preserved by even overlap,
                        # Z stable round-to-round)
                        builder.emit_round(
                            circuit,
                            emit_detectors=True,
                        )
                elif block_name == "block_0":
                    # Ctrl block: CNOT(ctrl_data→bridge) with data as CONTROL.
                    # Individual X dressed but X STABILIZERS preserved (even overlap).
                    # Z unaffected. Anchors depend on prep state.
                    z_det = block_prep.z_deterministic if block_prep else False
                    x_det = block_prep.x_deterministic if block_prep else False
                    if rnd == 0:
                        builder.emit_round(
                            circuit,
                            emit_detectors=True,
                            emit_z_anchors=z_det,
                            emit_x_anchors=x_det,  # X stabs preserved by even overlap
                            explicit_anchor_mode=True,
                        )
                    else:
                        # R1+: BOTH temporal detectors (X and Z both stable)
                        builder.emit_round(
                            circuit,
                            emit_detectors=True,
                        )
                else:
                    # block_2: Not involved in ZZ merge, both X and Z are valid
                    if rnd == 0:
                        z_det = block_prep.z_deterministic if block_prep else False
                        x_det = block_prep.x_deterministic if block_prep else False
                        builder.emit_round(
                            circuit,
                            emit_detectors=True,
                            emit_z_anchors=z_det,
                            emit_x_anchors=x_det,
                            explicit_anchor_mode=True,
                        )
                    else:
                        # Subsequent rounds: temporal detectors
                        builder.emit_round(circuit, emit_detectors=True)

        # ZZ merge done. EC already interleaved, so no additional rounds needed.
        # NOTE: Do NOT set stabilizer_transform here - let get_stabilizer_transform()
        # apply the final transform after all phases complete.
        return PhaseResult(
            phase_type=PhaseType.GATE,
            is_final=False,
            needs_stabilizer_rounds=0,  # EC already emitted
        )

    # ------------------------------------------------------------------
    # Phase 2: ZZ Split
    # ------------------------------------------------------------------

    def _emit_zz_split(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """
        ZZ split: destructive Z measurement of bridge qubits.
        
        Emits M(bridge_zz) and boundary detectors (split vs last bridge round).
        Stores split outcomes for observable correction.
        """
        bridge_zz = [
            gi for gi, _, purpose in alloc.bridge_ancillas
            if purpose.startswith("zz_merge")
        ]
        n_z = len(bridge_zz)

        # M on bridge (Z basis)
        meas_start = ctx.add_measurement(n_z)
        circuit.append("M", bridge_zz)
        self._zz_split_meas = list(range(meas_start, meas_start + n_z))
        circuit.append("TICK")

        # Boundary detectors: split vs last bridge syndrome
        if self._zz_bridge_meas:
            last_round = self._zz_bridge_meas[-1]
            for b_idx in range(n_z):
                split_rec = self._zz_split_meas[b_idx] - ctx.measurement_index
                last_rec = last_round[b_idx] - ctx.measurement_index
                circuit.append("DETECTOR", [stim.target_rec(split_rec), stim.target_rec(last_rec)])

        return PhaseResult(
            phase_type=PhaseType.MEASUREMENT,
            is_final=False,
            needs_stabilizer_rounds=0,
        )

    # ------------------------------------------------------------------
    # Phase 3: XX Merge
    # ------------------------------------------------------------------

    def _emit_xx_merge(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
        ctrl: BlockAllocation,
        anc: BlockAllocation,
        tgt: BlockAllocation,
    ) -> PhaseResult:
        """
        XX merge: d rounds of bridge X-parity + interleaved EC.
        
        All stabilizers (X and Z) are preserved during XX merge because CSS
        stabilizer weights are even, ensuring dressing from CNOT(bridge→data)
        cancels at the stabilizer level. Full EC (BOTH types) on all blocks.
        
        Each round:
            1. RX(bridge_xx)               [prepare |+>]
            2. CX(bridge -> anc_data)      [bridge is CONTROL]
            3. CX(bridge -> tgt_data)      [bridge is CONTROL]
            4. MX(bridge_xx)               [X-basis measurement]
            5. EC: all blocks BOTH (X + Z) with full detectors
        """
        code = self._code
        d = self.merge_rounds if self.merge_rounds is not None else code.d

        # NOTE: We do NOT reset stabilizer history for block_1/2 here.
        # The ZZ merge phase preserved ALL stabilizers (both X and Z) because
        # the CSS weight parity ensures all stabilizer-bridge overlaps are EVEN.
        # After ZZ split (destructive M on bridge qubits, which doesn't touch
        # data), both X and Z eigenvalues on data are unchanged. So temporal
        # detectors from XX merge R0 vs ZZ merge R[last] are valid for BOTH types.
        # This gives us cross-phase temporal detectors "for free".

        bridge_xx = [
            gi for gi, _, purpose in alloc.bridge_ancillas
            if purpose.startswith("xx_merge")
        ]
        n_x = len(bridge_xx)

        # X-stabiliser supports from code.hx
        n_stab = min(n_x, code.hx.shape[0])
        x_supports = []
        for s_idx in range(n_stab):
            x_supports.append(list(np.where(code.hx[s_idx])[0]))
        while len(x_supports) < n_x:
            x_supports.append([])

        anc_data = anc.get_data_qubits()
        tgt_data = tgt.get_data_qubits()

        self._xx_bridge_meas.clear()

        for rnd in range(d):
            # --- Bridge XX measurement ---
            circuit.append("RX", bridge_xx)
            circuit.append("TICK")

            # CX bridge -> anc_data (bridge is CONTROL)
            for b_idx in range(n_x):
                if b_idx < len(x_supports) and x_supports[b_idx]:
                    for local_idx in x_supports[b_idx]:
                        circuit.append("CNOT", [bridge_xx[b_idx], anc_data[local_idx]])
            circuit.append("TICK")

            # CX bridge -> tgt_data (bridge is CONTROL)
            for b_idx in range(n_x):
                if b_idx < len(x_supports) and x_supports[b_idx]:
                    for local_idx in x_supports[b_idx]:
                        circuit.append("CNOT", [bridge_xx[b_idx], tgt_data[local_idx]])
            circuit.append("TICK")

            # Measure bridge in X basis
            meas_start = ctx.add_measurement(n_x)
            circuit.append("MX", bridge_xx)
            self._xx_bridge_meas.append(list(range(meas_start, meas_start + n_x)))
            circuit.append("TICK")

            # XX bridge temporal detectors (round >= 1).
            # The bridge measures X_anc ⊗ X_tgt parity. Individual X values
            # may be random (from ZZ merge corruption), but the PARITY is
            # stable round-to-round because nothing flips X on data between
            # XX merge rounds. So consecutive bridge MX outcomes match.
            if rnd >= 1:
                for b_idx in range(n_x):
                    curr = self._xx_bridge_meas[rnd][b_idx] - ctx.measurement_index
                    prev = self._xx_bridge_meas[rnd - 1][b_idx] - ctx.measurement_index
                    coord = (b_idx + 0.5, -0.5, rnd + 100)  # Offset to distinguish from ZZ bridge
                    circuit.append("DETECTOR", [stim.target_rec(curr), stim.target_rec(prev)], coord)

            # --- Interleaved EC ---
            # XX bridge CNOT is CNOT(bridge→data) with bridge as CONTROL.
            # Individual qubit Z is dressed: Z_d → Z_d ⊗ Z_bridge, BUT at the
            # STABILIZER level, all Z stabilizers have EVEN overlap with X-bridge
            # supports (CSS weight parity), so Z_stab is PRESERVED.
            # X_data → X_data (unchanged by CNOT on target).
            # Both X and Z stabilizers are stable → emit BOTH with full detectors.
            for builder in self._builders:
                block_name = builder.block_name
                if block_name == "block_0":
                    # Ctrl: not coupled to XX bridge, full EC
                    builder.emit_round(circuit, emit_detectors=True)
                else:
                    # Anc, Tgt: BOTH X and Z are preserved by even overlap.
                    # No history reset needed — temporals trace back to ZZ merge
                    # last round, which is valid (data unchanged by ZZ split).
                    builder.emit_round(
                        circuit,
                        emit_detectors=True,  # Full temporal detectors
                    )

        return PhaseResult(
            phase_type=PhaseType.GATE,
            is_final=False,
            needs_stabilizer_rounds=0,
        )

    # ------------------------------------------------------------------
    # Phase 4: XX Split
    # ------------------------------------------------------------------

    def _emit_xx_split(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
    ) -> PhaseResult:
        """
        XX split: destructive X measurement of bridge qubits.
        """
        bridge_xx = [
            gi for gi, _, purpose in alloc.bridge_ancillas
            if purpose.startswith("xx_merge")
        ]
        n_x = len(bridge_xx)

        meas_start = ctx.add_measurement(n_x)
        circuit.append("MX", bridge_xx)
        self._xx_split_meas = list(range(meas_start, meas_start + n_x))
        circuit.append("TICK")

        # Boundary detectors
        if self._xx_bridge_meas:
            last_round = self._xx_bridge_meas[-1]
            for b_idx in range(n_x):
                split_rec = self._xx_split_meas[b_idx] - ctx.measurement_index
                last_rec = last_round[b_idx] - ctx.measurement_index
                circuit.append("DETECTOR", [stim.target_rec(split_rec), stim.target_rec(last_rec)])

        return PhaseResult(
            phase_type=PhaseType.MEASUREMENT,
            is_final=False,
            needs_stabilizer_rounds=0,
        )

    # ------------------------------------------------------------------
    # Phase 5: Ancilla MX
    # ------------------------------------------------------------------

    def _emit_ancilla_measurement(
        self,
        circuit: stim.Circuit,
        alloc: QubitAllocation,
        ctx: "DetectorContext",
        anc: BlockAllocation,
    ) -> PhaseResult:
        """
        Destructive MX on ancilla data.
        """
        anc_data = anc.get_data_qubits()
        n_anc = len(anc_data)

        meas_start = ctx.add_measurement(n_anc)
        circuit.append("MX", anc_data)
        self._anc_meas_indices = list(range(meas_start, meas_start + n_anc))
        circuit.append("TICK")

        # X-stabiliser boundary detectors for ancilla.
        # MX on all data qubits gives direct access to X eigenvalues.
        # For each X stabilizer S_X = ⊗_{d in support} X_d, the boundary
        # detector compares: ⊕_{d in support} MX(d) ⊕ last_x_stab_meas.
        # This is deterministic because:
        #   1. X stabilizers are PRESERVED during XX merge (even overlap with
        #      Z-bridge supports means dressing cancels at stabilizer level)
        #   2. XX split (MX on bridge) doesn't touch data qubits
        #   3. MX on data directly measures X eigenvalues
        anc_builder = self._builders[1]  # block_1 builder
        hx = self._code.hx
        n_x_stab = hx.shape[0]
        
        for s_idx in range(n_x_stab):
            last_x = anc_builder._last_x_meas[s_idx]
            if last_x is None:
                continue  # No previous measurement to compare against
            
            support = list(np.where(hx[s_idx])[0])
            rec_targets = []
            
            # MX data outcomes for this stabilizer's support
            for local_idx in support:
                mx_global = self._anc_meas_indices[local_idx]
                rec_targets.append(stim.target_rec(mx_global - ctx.measurement_index))
            
            # Last X stabilizer measurement from XX merge
            rec_targets.append(stim.target_rec(last_x - ctx.measurement_index))
            
            coord = (s_idx + 0.5, 0.5, 200)  # Distinctive coord for anc boundary
            circuit.append("DETECTOR", rec_targets, coord)

        return PhaseResult(
            phase_type=PhaseType.MEASUREMENT,
            is_final=True,
            needs_stabilizer_rounds=0,
            destroyed_blocks={"block_1"},
            extra={"destroyed_block_meas_starts": {"block_1": meas_start}},
        )

    # ------------------------------------------------------------------
    # Stabiliser / Observable transforms
    # ------------------------------------------------------------------

    def get_stabilizer_transform(self) -> StabilizerTransform:
        """
        Post-gadget transform for surviving blocks.
        
        CRITICAL: The measurement history from merge phases includes bridge-coupled
        measurements. Post-gadget rounds must NOT emit temporal detectors comparing
        against those old measurements, as they would trace through the bridges.
        
        block_0: Clear history, do NOT skip first round. After all 5 phases,
                 block_0 data is in a definite state. The builder's
                 measurement_basis logic will emit correct anchors:
                   ctrl=0 → measurement_basis="Z" → Z anchors
                   ctrl=+ → measurement_basis="X" → X anchors
        
        block_2: Clear history, do NOT skip first round. After all 5 phases
                 (including XX split + ancilla MX), the XX bridge is fully
                 disentangled. Both X and Z stabilizers on block_2 are back
                 to definite eigenvalues. The builder's measurement_basis
                 logic will emit correct anchors:
                   tgt=0 → measurement_basis="Z" → Z anchors
                   tgt=+ → measurement_basis="X" → X anchors
        """
        return StabilizerTransform.identity(
            clear_history=True,  # Clear measurement history from merge phases
            skip_first_round=True,  # Default: skip (overridden per-block below)
            per_block={
                "block_0": {"clear_history": True, "skip_first_round": False},
                "block_2": {"clear_history": True, "skip_first_round": False},
            },
        )

    def get_observable_transform(self) -> TwoQubitObservableTransform:
        return TwoQubitObservableTransform.cnot()

    def get_two_qubit_observable_transform(self):
        return self.get_observable_transform()

    # ------------------------------------------------------------------
    # Observable config
    # ------------------------------------------------------------------

    def get_observable_config(self) -> ObservableConfig:
        """
        Observable configuration for lattice surgery CNOT via HeisenbergFrame.

        The universal Heisenberg-picture operators (input-independent)::

            Z_ctrl_out = Z_ctrl                          (no correction)
            X_ctrl_out = X_ctrl ⊕ m_anc_XL               (X correction)
            Z_tgt_out  = Z_tgt ⊕ Z_ctrl ⊕ s_zz_log      (Z correction)
            X_tgt_out  = X_tgt ⊕ m_anc_XL ⊕ s_xx_log    (X correction)

        The HeisenbergFrame handles all input-state-dependent logic:
        determinism checks, cross-block correlations, and GF(2)
        deduplication of bridge corrections.
        """
        def _correction_provider(basis: str) -> List[int]:
            """Return protocol correction indices for the given basis."""
            if basis == "X":
                return (
                    self._get_anc_x_logical_meas()
                    + self._get_xx_logical_bridge_meas()
                )
            elif basis == "Z":
                return self._get_zz_logical_bridge_meas()
            return []

        input_state = self.control_state + self.target_state
        frame = HeisenbergFrame.surgery_cnot(
            correction_provider=_correction_provider,
        )
        return ObservableConfig.from_heisenberg(frame, input_state)

    def _get_output_bases(self) -> Dict[str, str]:
        """
        Return the measurement basis for each surviving block.
        
        The ctrl basis matches ctrl input: Z for |0⟩, X for |+⟩.
        The tgt basis matches tgt input: Z for |0⟩, X for |+⟩.
        
        This is because after CNOT:
        - Z_ctrl_out = Z_ctrl (preserved) → measure Z if ctrl was |0⟩
        - X_ctrl_out = X_ctrl ⊕ correction (dressed) → measure X if ctrl was |+⟩
        - Z_tgt_out = Z_tgt ⊕ Z_ctrl ⊕ s_zz (dressed) → measure Z if tgt was |0⟩
        - X_tgt_out = X_tgt (preserved) → measure X if tgt was |+⟩
        """
        ctrl_basis = "X" if self.control_state == "+" else "Z"
        tgt_basis = "X" if self.target_state == "+" else "Z"
        return {"block_0": ctrl_basis, "block_2": tgt_basis}

    def _get_frame_corrections(self) -> List[int]:
        """
        Observable frame corrections (bridge measurement parity).
        
        Derived uniformly from the universal Heisenberg operators:
        
            For each X-measured block → include X corrections
            For each Z-measured block → include Z corrections
        
        The same logic as get_observable_config().bridge_frame_meas_indices,
        with GF(2) deduplication (m_anc_XL cancels when both blocks are
        X-measured).
        
        Note: These corrections are integrated directly into
        get_observable_config() via bridge_frame_meas_indices.
        """
        from collections import Counter

        bases = self._get_output_bases()
        ctrl_basis = bases["block_0"]
        tgt_basis = bases["block_2"]

        m_anc_xl = self._get_anc_x_logical_meas()
        s_xx_log = self._get_xx_logical_bridge_meas()
        s_zz_log = self._get_zz_logical_bridge_meas()

        bridge_raw: List[int] = []

        # ctrl X correction
        if ctrl_basis == "X":
            bridge_raw.extend(m_anc_xl)

        # tgt corrections
        if tgt_basis == "X":
            bridge_raw.extend(m_anc_xl)
            bridge_raw.extend(s_xx_log)
        elif tgt_basis == "Z" and ctrl_basis == "Z":
            bridge_raw.extend(s_zz_log)

        # GF(2) dedup
        counts = Counter(bridge_raw)
        return [idx for idx, cnt in counts.items() if cnt % 2 == 1]

    def _get_far_x_logical_support(self) -> List[int]:
        """
        Get X-logical support on the column FARTHEST from the ZZ merge boundary.

        The default X̄ representative sits on the leftmost column (x_min) of the
        ancilla block, which is directly adjacent to the ZZ merge bridges.  A
        single depolarisation fault on a bridge CX can then flip both a
        boundary detector AND the observable, creating a weight-2 graphlike
        error and reducing d_eff to d − 1.

        Because any vertical column is a valid representative of the same
        homology class (columns differ by products of X-stabilisers), we can
        equivalently use the rightmost column (x_max).  This column is d − 1
        data qubits away from the merge boundary, restoring full distance
        protection.

        Returns
        -------
        List[int]
            Sorted local data-qubit indices forming the far-edge X̄ representative.
        """
        if self._code is None:
            return []
        data_coords = self._code.metadata.get("data_coords", [])
        if not data_coords:
            # Fallback: try code's get_data_coords() method
            css = self._code.as_css() if hasattr(self._code, 'as_css') else None
            if css is not None and hasattr(css, 'get_data_coords'):
                dc = css.get_data_coords()
                if dc:
                    data_coords = dc
        if not data_coords:
            return self._code.get_logical_x_support(0)

        # Find the rightmost column (max x among data qubits)
        x_max = max(c[0] for c in data_coords)
        far_support = sorted(
            i for i, c in enumerate(data_coords) if c[0] == x_max
        )
        return far_support if far_support else self._code.get_logical_x_support(0)

    def _get_anc_x_logical_meas(self) -> List[int]:
        """Get ancilla MX indices for logical X support (far-edge representative)."""
        if self._code is None or not self._anc_meas_indices:
            return []
        x_support = self._get_far_x_logical_support()
        result = []
        for local_idx in x_support:
            if local_idx < len(self._anc_meas_indices):
                result.append(self._anc_meas_indices[local_idx])
        return result

    def _get_xx_logical_bridge_meas(self) -> List[int]:
        """
        Get XX split measurements whose X-stabilizer has odd overlap with X_L.
        
        For each XX bridge stabilizer s, if |supp(s) ∩ X_L| is odd, then the
        bridge measurement carries logical X parity and must be included in the
        observable correction for X_tgt_out.

        Uses the far-edge X̄ representative (rightmost column) so that the
        observable avoids the merge boundary.
        """
        if self._code is None or not self._xx_split_meas:
            return []
        import numpy as np
        hx = self._code.hx
        x_logical = set(self._get_far_x_logical_support())
        n_bridges = len(self._xx_split_meas)
        result = []
        for s_idx in range(min(n_bridges, hx.shape[0])):
            stab_support = set(np.where(hx[s_idx])[0])
            if len(stab_support & x_logical) % 2 == 1:
                result.append(self._xx_split_meas[s_idx])
        return result

    def _get_zz_logical_bridge_meas(self) -> List[int]:
        """
        Get ZZ split measurements whose Z-stabilizer has odd overlap with Z_L.
        
        For each ZZ bridge stabilizer s, if |supp(s) ∩ Z_L| is odd, then the
        bridge measurement carries logical Z parity and must be included in the
        observable correction for Z_tgt_out.
        
        At all tested distances (d=3,5,7), no Z-stabilizer has odd overlap
        with Z_L (the boundary stabilizer with odd overlap is the (d+1)th
        stabilizer which is NOT bridged), so this always returns [].
        
        We include it for conceptual completeness: the uniform derivation
        of the observable should reference s_zz_log even when it's empty.
        """
        if self._code is None or not self._zz_split_meas:
            return []
        import numpy as np
        hz = self._code.hz
        z_logical = set(self._code.get_logical_z_support(0))
        n_bridges = len(self._zz_split_meas)
        result = []
        for s_idx in range(min(n_bridges, hz.shape[0])):
            stab_support = set(np.where(hz[s_idx])[0])
            if len(stab_support & z_logical) % 2 == 1:
                result.append(self._zz_split_meas[s_idx])
        return result

    # ------------------------------------------------------------------
    # Destroyed/surviving blocks
    # ------------------------------------------------------------------

    def get_destroyed_blocks(self) -> Set[str]:
        return {"block_1"}

    def get_ancilla_block_names(self) -> Set[str]:
        return {"block_1"}

    def get_blocks_to_skip_pre_rounds(self) -> Set[str]:
        """Skip all pre-gadget EC - the gadget handles EC during merge phases."""
        return {"block_0", "block_1", "block_2"}

    def get_blocks_to_skip_post_rounds(self) -> Set[str]:
        return {"block_1"}

    def get_x_stabilizer_mode(self) -> str:
        """
        Return the gate type to use for X stabilizer measurement.
        
        For |+⟩ state preparation, CZ-based X-stabilizer measurement gives
        non-deterministic results. Using CX (CNOT ancilla→data) with H-CX-H
        correctly measures X-parity on |+⟩ states.
        
        Block_1 (ancilla) is always |+⟩, so we always need CX mode.
        """
        return "cx"

    # ------------------------------------------------------------------
    # Preparation / measurement configs
    # ------------------------------------------------------------------

    def get_measurement_basis(self) -> str:
        return self._get_output_bases()["block_0"]

    def get_preparation_config(self, input_state: str = "0") -> PreparationConfig:
        return PreparationConfig(blocks={
            "block_0": BlockPreparationConfig(
                initial_state=self.control_state,
                z_deterministic=(self.control_state == "0"),
                x_deterministic=(self.control_state == "+"),
            ),
            "block_1": BlockPreparationConfig(
                initial_state="+",
                z_deterministic=False,
                x_deterministic=True,
            ),
            "block_2": BlockPreparationConfig(
                initial_state=self.target_state,
                z_deterministic=(self.target_state == "0"),
                x_deterministic=(self.target_state == "+"),
            ),
        })

    def get_measurement_config(self) -> MeasurementConfig:
        bases = self._get_output_bases()
        return MeasurementConfig(
            block_bases={"block_0": bases["block_0"], "block_2": bases["block_2"]},
            destroyed_blocks={"block_1"},
        )

    def get_boundary_detector_config(self) -> BoundaryDetectorConfig:
        bases = self._get_output_bases()
        return BoundaryDetectorConfig(block_configs={
            "block_0": {"X": bases["block_0"] == "X", "Z": bases["block_0"] == "Z"},
            # Skip block_1 X boundary detectors - the last X-stab measurements
            # from XX merge are entangled with the XX bridge, so boundary detectors
            # comparing final MX with those measurements would trace through the
            # bridge and anti-commute with the bridge RX resets.
            "block_1": {"X": False, "Z": False},
            "block_2": {"X": bases["block_2"] == "X", "Z": bases["block_2"] == "Z"},
        })

    def get_crossing_detector_config(self) -> Optional[CrossingDetectorConfig]:
        """
        Return None - surgery gadget handles all detectors internally.
        
        Returning None (not empty config) ensures the orchestrator emits normal
        temporal detectors for post-gadget rounds. An empty CrossingDetectorConfig
        would cause emit_auto_detectors=False with no formulas to compensate.
        """
        return None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compute_layout(self, codes: List[Code]) -> GadgetLayout:
        if len(codes) == 1:
            ctrl_code = anc_code = tgt_code = codes[0]
        elif len(codes) == 3:
            ctrl_code, anc_code, tgt_code = codes
        else:
            raise ValueError(f"Expected 1 or 3 codes, got {len(codes)}")

        self._code = ctrl_code
        layout = GadgetLayout(target_dim=2)
        
        surgery_gap = 1.0
        layout.add_block("block_0", ctrl_code, offset=(0.0, 0.0))

        has_boundaries = (
            hasattr(ctrl_code, 'has_physical_boundaries') and
            ctrl_code.has_physical_boundaries()
        )

        if has_boundaries:
            layout.add_block_adjacent("block_1", anc_code, reference_block="block_0",
                                      my_edge="left", their_edge="right", gap=surgery_gap)
            layout.add_block_adjacent("block_2", tgt_code, reference_block="block_1",
                                      my_edge="top", their_edge="bottom", gap=surgery_gap)
            layout.add_boundary_bridges("block_0", "block_1", "right", "left", "zz_merge", 0)
            layout.add_boundary_bridges("block_1", "block_2", "bottom", "top", "xx_merge", 0)
        else:
            margin = max(3.0, (ctrl_code.d or 3) * 1.5)
            layout.add_block("block_1", anc_code, auto_offset=True, margin=margin)
            layout.add_block("block_2", tgt_code, auto_offset=True, margin=margin)
            n_z = ctrl_code.hz.shape[0] if ctrl_code.hz is not None else 0
            for s in range(n_z):
                layout.add_bridge_ancilla(purpose=f"zz_merge_{s}", connected_blocks=["block_0", "block_1"])
            n_x = ctrl_code.hx.shape[0] if ctrl_code.hx is not None else 0
            for s in range(n_x):
                layout.add_bridge_ancilla(purpose=f"xx_merge_{s}", connected_blocks=["block_1", "block_2"])

        return layout

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> GadgetMetadata:
        return GadgetMetadata(
            gadget_type="css_surgery",
            logical_operation="CNOT",
            extra={
                "num_blocks": 3,
                "control_state": self.control_state,
                "target_state": self.target_state,
                "protocol": "5_phase_interleaved",
            },
        )


def create_surgery_cnot(
    control_state: str = "0",
    target_state: str = "0",
    merge_rounds: Optional[int] = None,
) -> CSSSurgeryCNOTGadget:
    return CSSSurgeryCNOTGadget(
        control_state=control_state,
        target_state=target_state,
        merge_rounds=merge_rounds,
    )
