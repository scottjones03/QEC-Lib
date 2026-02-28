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
from qectostim.codes.abstract_css import CSSCode, MergeStabilizerInfo
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
    BlockInfo,
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
        
        # Merge stabilizer info (seam + grown stabs for each merge)
        self._zz_merge_info: Optional[MergeStabilizerInfo] = None
        self._xx_merge_info: Optional[MergeStabilizerInfo] = None
        self._zz_seam_ancillas: List[int] = []
        self._xx_seam_ancillas: List[int] = []
        
        # Bridge measurement tracking (seam stab measurements per round)
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
        self._zz_seam_ancillas.clear()
        self._xx_seam_ancillas.clear()
        self._anc_meas_indices.clear()

    def get_block_names(self) -> List[str]:
        """block_0 = control, block_1 = ancilla, block_2 = target."""
        return ["block_0", "block_1", "block_2"]

    def get_phase_active_blocks(self, phase_index: int) -> List[str]:
        """Per-phase active blocks for CSS Surgery CNOT.

        Merge phases (0, 2) return ALL blocks because interleaved EC
        is emitted on every block during the merge loop — the routing
        layer must cover the full grid so every block's ions can be
        shuffled.

        Phase 0 (ZZ merge):  all blocks (bridge on block_0 ↔ block_1, EC on all)
        Phase 1 (ZZ split):  block_0, block_1
        Phase 2 (XX merge):  all blocks (bridge on block_1 ↔ block_2, EC on all)
        Phase 3 (XX split):  block_1, block_2
        Phase 4 (Anc MX):    block_1
        """
        if phase_index in (0, 2):
            # Merge phases emit interleaved EC on ALL blocks.
            return self.get_block_names()
        elif phase_index == 1:
            return ["block_0", "block_1"]
        elif phase_index == 3:
            return ["block_1", "block_2"]
        elif phase_index == 4:
            return ["block_1"]
        return self.get_block_names()

    def get_phase_pairs(self, phase_index, alloc):
        """CSS Surgery phase pairs for hardware routing.

        During merge phases, the seam stabilizer CX gates must be routed
        alongside the EC CX gates.  Each merge round produces 4 CX-phase
        layers (one per geometric CX phase), where each layer contains all
        seam+grown CX pairs for that phase.  The EC CX pairs are handled
        by the block builders and routed separately; only the cross-block
        seam/grown pairs appear here.

        Phase 0 (ZZ merge): ``d`` rounds × 4 CX phases.
        Phase 1 (ZZ split): no MS pairs.
        Phase 2 (XX merge): ``d`` rounds × 4 CX phases.
        Phase 3 (XX split): no MS pairs.
        Phase 4 (Anc MX):   no MS pairs.
        """
        if phase_index in (1, 3, 4):
            return []

        code = self._code
        if code is None:
            return []
        d = self.merge_rounds if self.merge_rounds is not None else (code.d or 3)

        merge_info = (
            self._zz_merge_info if phase_index == 0
            else self._xx_merge_info
        )
        if merge_info is None or not merge_info.seam_stabs:
            return []

        n_phases = merge_info.num_cx_phases
        layers: List[List[Tuple[int, int]]] = []

        for _rnd in range(d):
            for ph in range(n_phases):
                phase_pairs: List[Tuple[int, int]] = []
                for seam_stab in merge_info.seam_stabs:
                    for pair in seam_stab.cx_per_phase[ph]:
                        phase_pairs.append(pair)
                # Grown stab cross-block CX pairs for this phase
                for grown_stab in merge_info.grown_stabs:
                    if grown_stab.existing_ancilla_global >= 0:
                        for pair in grown_stab.new_cx_per_phase[ph]:
                            phase_pairs.append(pair)
                if phase_pairs:
                    layers.append(phase_pairs)
        return layers

    # ------------------------------------------------------------------
    # Grown-stabilizer ancilla resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_grown_stabs(
        merge_info: MergeStabilizerInfo,
        code: CSSCode,
        block_info: "BlockInfo",
        block_name: str,
    ) -> None:
        """Resolve placeholder indices in grown stabilizers.

        Each :class:`GrownStabilizer` produced by
        ``code.get_merge_stabilizers()`` has ``existing_ancilla_global = -1``
        and ``belongs_to_block = ""`` because the code doesn't know the
        global allocation.  This method matches each grown stab's
        ``lattice_position`` against the code's stabilizer coordinate
        list and fills in the actual global ancilla index from the
        block's allocation ranges.

        Once resolved, ``grown.new_cx_per_phase`` entries also have their
        ``-1`` placeholders replaced with the resolved ancilla global.
        """
        if not merge_info.grown_stabs:
            return

        grown_type = merge_info.grown_type  # "X" or "Z"
        if grown_type == "X":
            stab_coords = code.get_x_stabilizer_coords()
            anc_range = block_info.x_ancilla_range
        else:
            stab_coords = code.get_z_stabilizer_coords()
            anc_range = block_info.z_ancilla_range

        if stab_coords is None:
            return  # Code doesn't provide coords — can't resolve

        # Build a lookup: (rounded coord) → local index
        coord_to_local: dict = {}
        for local_idx, c in enumerate(stab_coords):
            key = (round(c[0]), round(c[1])) if len(c) >= 2 else (round(c[0]),)
            coord_to_local[key] = local_idx

        for gs in merge_info.grown_stabs:
            lp = gs.lattice_position
            key = (round(lp[0]), round(lp[1])) if len(lp) >= 2 else (round(lp[0]),)
            local_idx = coord_to_local.get(key, -1)
            if local_idx < 0:
                continue  # No match found

            g_anc = anc_range.start + local_idx
            gs.existing_ancilla_global = g_anc
            gs.belongs_to_block = block_name

            # Replace -1 placeholders in new_cx_per_phase
            for ph in range(len(gs.new_cx_per_phase)):
                gs.new_cx_per_phase[ph] = [
                    (g_anc, pair[1]) if pair[0] == -1 else
                    (pair[0], g_anc) if pair[1] == -1 else
                    pair
                    for pair in gs.new_cx_per_phase[ph]
                ]

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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_ec_round(builder, circuit, **kwargs):
        """Emit an EC round during merge phases (single-builder fallback).

        For hierarchical builders, uses inner-only rounds to avoid
        intermediate outer CX operations.  These outer operations would
        introduce sensitivities on outer ancilla qubits at each merge
        round, making post-gadget inner crossing detectors non-deterministic
        (they cannot compensate for *all* intermediate outer preparations).

        For flat builders, uses the standard full round with anchor control.
        """
        if hasattr(builder, 'emit_inner_only_round'):
            # Hierarchical: inner-only preserves crossing detector flow.
            builder.emit_inner_only_round(
                circuit,
                emit_detectors=kwargs.get('emit_detectors', True),
            )
        else:
            # Flat: full round with anchor control.
            builder.emit_round(circuit, **kwargs)

    @staticmethod
    def _emit_parallel_ec_round(
        builders,
        circuit: "stim.Circuit",
        per_builder_kwargs: dict,
    ) -> None:
        """Emit one EC round with all builders sharing the same TICK layers.

        This produces the same structure as
        ``MemoryRoundEmitter._emit_parallel_interleaved``:

        1. Reset all ancillas (all builders) → TICK
        2. For each geometric CX phase:
              all builders emit their X+Z CX gates → TICK
        3. Measure all ancillas + emit detectors (all builders) → TICK

        This achieves cross-block CX parallelism: gates from block_0,
        block_1 and block_2 land in the *same* TICK layers instead of
        being serialised into separate per-builder TICK groups.

        Parameters
        ----------
        builders : list
            CSSStabilizerRoundBuilder instances (one per block).
        circuit : stim.Circuit
            Target circuit to append instructions to.
        per_builder_kwargs : dict
            Maps ``builder.block_name`` → dict of keyword arguments that
            would have been passed to ``_emit_ec_round``.  Recognised
            keys: ``emit_detectors``, ``emit_z_anchors``,
            ``emit_x_anchors``, ``explicit_anchor_mode``.
        """
        # Decide whether we can use the phase-decomposed path for
        # ALL builders.  If any builder is hierarchical or doesn't
        # support interleaving, fall back to sequential emission.
        all_flat_interleave = all(
            not hasattr(b, 'emit_inner_only_round')
            and hasattr(b, 'emit_cx_for_phase')
            and hasattr(b, 'n_interleave_phases')
            and b.n_interleave_phases() > 0
            for b in builders
        )
        if not all_flat_interleave:
            # Fallback: sequential per-builder
            for b in builders:
                kw = per_builder_kwargs.get(b.block_name, {})
                CSSSurgeryCNOTGadget._emit_ec_round(b, circuit, **kw)
            return

        # Determine the maximum number of CX phases across builders
        n_phases = max(b.n_interleave_phases() for b in builders)

        # --- 1. Reset ancillas for all builders ---
        for b in builders:
            kw = per_builder_kwargs.get(b.block_name, {})
            b.emit_ancilla_reset(
                circuit,
                emit_z_anchors=kw.get('emit_z_anchors', False),
                emit_x_anchors=kw.get('emit_x_anchors', False),
            )
        circuit.append("TICK")

        # --- 2. Interleaved CX phases ---
        for phase_idx in range(n_phases):
            if phase_idx > 0:
                circuit.append("TICK")
            for b in builders:
                if phase_idx < b.n_interleave_phases():
                    b.emit_cx_for_phase(circuit, phase_idx)

        circuit.append("TICK")

        # --- 3. Measure ancillas + detectors ---
        for b in builders:
            kw = per_builder_kwargs.get(b.block_name, {})
            b.emit_ancilla_measure_and_detectors(
                circuit,
                emit_detectors=kw.get('emit_detectors', True),
            )

        # Final TICK separating this round from the next
        circuit.append("TICK")

    # ------------------------------------------------------------------
    # Joint merge round — used by ZZ and XX merge phases
    # ------------------------------------------------------------------

    def _emit_joint_merge_round(
        self,
        builders,
        circuit: "stim.Circuit",
        per_builder_kwargs: dict,
        seam_ancillas: list,
        seam_cx_per_phase: "List[List[Tuple[int,int]]]",
        grown_cx_per_phase: "List[List[Tuple[int,int]]]",
        seam_reset_op: str,
        seam_measure_op: str,
        ctx: "DetectorContext",
        rnd: int,
        prev_seam_meas: "list | None",
        seam_meas_list: list,
        det_coord_offset: float = 0.0,
    ) -> None:
        """Emit one JOINT merge round: EC + seam stabs in the SAME tick window.

        Proper Horsman/Fowler lattice surgery requires all stabilisers of the
        joint patch (internal EC stabilisers AND new seam stabilisers) to be
        measured in a single round using a unified 4-phase CX schedule.

        Structure per round (no extra TICK phases for seam CX):

        1. Reset EC ancillas (all builders) + seam ancillas → TICK
        2. CX Phase 0: all EC CX (builders) + seam CX (phase 0) + grown CX (phase 0) → TICK
        3. CX Phase 1: ~same~ → TICK
        4. CX Phase 2: ~same~ → TICK
        5. CX Phase 3: ~same~ → TICK
        6. Measure EC ancillas (builders) + seam ancillas → detectors → TICK
        """
        n_seam = len(seam_ancillas)

        # Check interleaving capability
        all_flat_interleave = all(
            not hasattr(b, 'emit_inner_only_round')
            and hasattr(b, 'emit_cx_for_phase')
            and hasattr(b, 'n_interleave_phases')
            and b.n_interleave_phases() > 0
            for b in builders
        )

        if not all_flat_interleave:
            # Fallback: sequential per-builder + separate seam
            # Reset seam
            if n_seam > 0:
                circuit.append(seam_reset_op, seam_ancillas)
                circuit.append("TICK")
            # Seam CX (separate ticks as fallback)
            for ph in range(len(seam_cx_per_phase)):
                cx_flat = [q for pair in seam_cx_per_phase[ph] for q in pair]
                gcx_flat = [q for pair in grown_cx_per_phase[ph] for q in pair]
                all_cx = cx_flat + gcx_flat
                if all_cx:
                    circuit.append("CNOT", all_cx)
                    circuit.append("TICK")
            # Measure seam
            if n_seam > 0:
                meas_start = ctx.add_measurement(n_seam)
                circuit.append(seam_measure_op, seam_ancillas)
                seam_meas_list.append(list(range(meas_start, meas_start + n_seam)))
                circuit.append("TICK")
                if rnd >= 1 and prev_seam_meas is not None:
                    curr_meas = seam_meas_list[-1]
                    for s_idx in range(n_seam):
                        curr = curr_meas[s_idx] - ctx.measurement_index
                        prev = prev_seam_meas[s_idx] - ctx.measurement_index
                        coord = (s_idx + 0.5, -0.5, rnd + det_coord_offset)
                        circuit.append("DETECTOR",
                                       [stim.target_rec(curr), stim.target_rec(prev)],
                                       coord)
            # EC sequentially
            for b in builders:
                kw = per_builder_kwargs.get(b.block_name, {})
                CSSSurgeryCNOTGadget._emit_ec_round(b, circuit, **kw)
            return

        n_phases = max(b.n_interleave_phases() for b in builders)

        # --- 1. Reset EC ancillas + seam ancillas ---
        for b in builders:
            kw = per_builder_kwargs.get(b.block_name, {})
            b.emit_ancilla_reset(
                circuit,
                emit_z_anchors=kw.get('emit_z_anchors', False),
                emit_x_anchors=kw.get('emit_x_anchors', False),
            )
        if n_seam > 0:
            circuit.append(seam_reset_op, seam_ancillas)
        circuit.append("TICK")

        # --- 2. Interleaved CX phases: EC + seam + grown, all in same TICK ---
        for phase_idx in range(n_phases):
            if phase_idx > 0:
                circuit.append("TICK")
            # EC CX from all builders
            for b in builders:
                if phase_idx < b.n_interleave_phases():
                    b.emit_cx_for_phase(circuit, phase_idx)
            # Seam stabilizer CX for this phase
            if phase_idx < len(seam_cx_per_phase):
                cx_flat = [q for pair in seam_cx_per_phase[phase_idx] for q in pair]
                if cx_flat:
                    circuit.append("CNOT", cx_flat)
            # Grown stabilizer additional CX for this phase
            if phase_idx < len(grown_cx_per_phase):
                gcx_flat = [q for pair in grown_cx_per_phase[phase_idx] for q in pair]
                if gcx_flat:
                    circuit.append("CNOT", gcx_flat)

        circuit.append("TICK")

        # --- 3. Measure seam ancillas FIRST (so record indices are known) ---
        if n_seam > 0:
            meas_start = ctx.add_measurement(n_seam)
            circuit.append(seam_measure_op, seam_ancillas)
            seam_meas_list.append(list(range(meas_start, meas_start + n_seam)))

        # --- 4. Measure EC ancillas + emit detectors ---
        for b in builders:
            kw = per_builder_kwargs.get(b.block_name, {})
            b.emit_ancilla_measure_and_detectors(
                circuit,
                emit_detectors=kw.get('emit_detectors', True),
            )

        circuit.append("TICK")

        # --- 5. Seam temporal detectors (round >= 1) ---
        if rnd >= 1 and prev_seam_meas is not None and n_seam > 0:
            curr_meas = seam_meas_list[-1]
            for s_idx in range(n_seam):
                curr = curr_meas[s_idx] - ctx.measurement_index
                prev = prev_seam_meas[s_idx] - ctx.measurement_index
                coord = (s_idx + 0.5, -0.5, rnd + det_coord_offset)
                circuit.append("DETECTOR",
                               [stim.target_rec(curr), stim.target_rec(prev)],
                               coord)

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
        ZZ merge: d rounds of joint stabiliser measurement using seam Z-stabs.
        
        Proper Horsman/Fowler lattice surgery: the seam Z-stabilisers are
        weight-2/4 plaquettes at even–even lattice positions between block_0
        (control) and block_1 (ancilla).  Their CX gates are integrated into
        the same 4-phase schedule as the internal EC stabilisers.
        
        The product of all seam Z-stab eigenvalues = Z_L_ctrl ⊗ Z_L_anc.
        """
        code = self._code
        d = self.merge_rounds if self.merge_rounds is not None else (code.d or 3)

        # Seam ancillas were allocated during compute_layout()
        seam_zz = [
            gi for gi, _, purpose in alloc.bridge_ancillas
            if purpose.startswith("zz_merge")
        ]
        self._zz_seam_ancillas = seam_zz
        n_seam = len(seam_zz)
        if n_seam == 0:
            return PhaseResult(
                phase_type=PhaseType.GATE, is_final=False,
                needs_stabilizer_rounds=0,
            )

        # Use precomputed merge info from compute_layout (required)
        merge_info = self._zz_merge_info
        if merge_info is None or not merge_info.seam_stabs:
            raise RuntimeError(
                "ZZ merge info not available — compute_layout() must be called "
                "first with a code that implements get_merge_stabilizers()."
            )

        # Build per-phase CX lists from seam stabs
        n_phases = merge_info.num_cx_phases
        seam_cx_per_phase = [[] for _ in range(n_phases)]
        for seam_stab in merge_info.seam_stabs:
            for ph in range(n_phases):
                if ph < len(seam_stab.cx_per_phase):
                    seam_cx_per_phase[ph].extend(seam_stab.cx_per_phase[ph])

        # Grown stab extra CX: boundary stabs whose weight increases
        # during merge gain additional data neighbors from the other block.
        grown_cx_per_phase = [[] for _ in range(n_phases)]
        for gs in merge_info.grown_stabs:
            if gs.existing_ancilla_global >= 0:
                for ph in range(n_phases):
                    if ph < len(gs.new_cx_per_phase):
                        grown_cx_per_phase[ph].extend(gs.new_cx_per_phase[ph])

        prep = self.get_preparation_config()
        self._zz_bridge_meas.clear()

        for rnd in range(d):
            per_builder_kw: dict = {}
            for builder in self._builders:
                block_name = builder.block_name
                block_prep = prep.blocks.get(block_name)

                if rnd == 0:
                    if block_name == "block_1":
                        per_builder_kw[block_name] = dict(
                            emit_detectors=True,
                            emit_x_anchors=True,
                            emit_z_anchors=False,
                            explicit_anchor_mode=True,
                        )
                    else:
                        z_det = block_prep.z_deterministic if block_prep else False
                        x_det = block_prep.x_deterministic if block_prep else False
                        per_builder_kw[block_name] = dict(
                            emit_detectors=True,
                            emit_z_anchors=z_det,
                            emit_x_anchors=x_det,
                            explicit_anchor_mode=True,
                        )
                else:
                    per_builder_kw[block_name] = dict(emit_detectors=True)

            prev_meas = self._zz_bridge_meas[rnd - 1] if rnd >= 1 else None
            self._emit_joint_merge_round(
                self._builders, circuit, per_builder_kw,
                seam_ancillas=seam_zz,
                seam_cx_per_phase=seam_cx_per_phase,
                grown_cx_per_phase=grown_cx_per_phase,
                seam_reset_op="R",
                seam_measure_op="M",
                ctx=ctx,
                rnd=rnd,
                prev_seam_meas=prev_meas,
                seam_meas_list=self._zz_bridge_meas,
                det_coord_offset=0.0,
            )

        return PhaseResult(
            phase_type=PhaseType.GATE,
            is_final=False,
            needs_stabilizer_rounds=0,
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
        XX merge: d rounds of joint stabiliser measurement using seam X-stabs.
        
        Proper Horsman/Fowler lattice surgery: seam X-stabilisers are
        weight-2/4 plaquettes at even–even lattice positions between block_1
        (ancilla) and block_2 (target).  Their CX gates are integrated into
        the same 4-phase schedule as the internal EC stabilisers.
        
        The product of all seam X-stab eigenvalues = X_L_anc ⊗ X_L_tgt.
        """
        code = self._code
        d = self.merge_rounds if self.merge_rounds is not None else (code.d or 3)

        seam_xx = [
            gi for gi, _, purpose in alloc.bridge_ancillas
            if purpose.startswith("xx_merge")
        ]
        self._xx_seam_ancillas = seam_xx
        n_seam = len(seam_xx)
        if n_seam == 0:
            return PhaseResult(
                phase_type=PhaseType.GATE, is_final=False,
                needs_stabilizer_rounds=0,
            )

        merge_info = self._xx_merge_info
        if merge_info is None or not merge_info.seam_stabs:
            raise RuntimeError(
                "XX merge info not available — compute_layout() must be called "
                "first with a code that implements get_merge_stabilizers()."
            )

        n_phases = merge_info.num_cx_phases
        seam_cx_per_phase = [[] for _ in range(n_phases)]
        for seam_stab in merge_info.seam_stabs:
            for ph in range(n_phases):
                if ph < len(seam_stab.cx_per_phase):
                    seam_cx_per_phase[ph].extend(seam_stab.cx_per_phase[ph])

        # Grown stab extra CX for XX merge
        grown_cx_per_phase = [[] for _ in range(n_phases)]
        for gs in merge_info.grown_stabs:
            if gs.existing_ancilla_global >= 0:
                for ph in range(n_phases):
                    if ph < len(gs.new_cx_per_phase):
                        grown_cx_per_phase[ph].extend(gs.new_cx_per_phase[ph])

        self._xx_bridge_meas.clear()

        for rnd in range(d):
            per_builder_kw: dict = {}
            for builder in self._builders:
                per_builder_kw[builder.block_name] = dict(emit_detectors=True)

            prev_meas = self._xx_bridge_meas[rnd - 1] if rnd >= 1 else None
            self._emit_joint_merge_round(
                self._builders, circuit, per_builder_kw,
                seam_ancillas=seam_xx,
                seam_cx_per_phase=seam_cx_per_phase,
                grown_cx_per_phase=grown_cx_per_phase,
                seam_reset_op="RX",
                seam_measure_op="MX",
                ctx=ctx,
                rnd=rnd,
                prev_seam_meas=prev_meas,
                seam_meas_list=self._xx_bridge_meas,
                det_coord_offset=100.0,
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
        #
        # NOTE: This manual boundary detector emission only works for flat
        # CSS builders that expose _last_x_meas as a list.  Hierarchical
        # builders use a different multi-level measurement tracking structure
        # and their boundary detectors are handled by auto_detectors instead.
        anc_builder = self._builders[1]  # block_1 builder
        if hasattr(anc_builder, '_last_x_meas'):
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
        
        Flat codes: block_0/block_2 do NOT skip first round — post-gadget
        anchor detectors are deterministic and provide proper boundary coverage.
        
        Hierarchical codes: block_0/block_2 skip first round — outer-ancilla
        prep measurements conflict with bridge-coupled merge-phase history,
        making anchors non-deterministic. Crossing detectors cover the gap.
        """
        _hier = self._code is not None and self._code.d is None
        return StabilizerTransform.identity(
            clear_history=True,
            skip_first_round=True,  # Default for block_1
            per_block={
                "block_0": {"clear_history": True, "skip_first_round": _hier},
                "block_2": {"clear_history": True, "skip_first_round": _hier},
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
        if not far_support:
            return self._code.get_logical_x_support(0)

        # Validate that far_support is in ker(H_Z) — required for a valid
        # X_L representative.  For surface codes every column works, but for
        # general CSS codes (e.g. Steane [[7,1,3]]) a single column may
        # anti-commute with Z stabilizers.
        if hasattr(self._code, 'hz') and self._code.hz is not None:
            vec = np.zeros(self._code.n, dtype=int)
            for idx in far_support:
                vec[idx] = 1
            if np.any((self._code.hz @ vec) % 2 != 0):
                return self._code.get_logical_x_support(0)

        return far_support

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
        Get XX split measurements for the observable correction.
        
        In proper lattice surgery, the product of ALL seam X-stab
        eigenvalues gives X_L_anc ⊗ X_L_tgt.  Therefore the XOR of ALL
        seam X-stab split measurements is the correction for X_tgt_out.
        """
        return list(self._xx_split_meas)

    def _get_zz_logical_bridge_meas(self) -> List[int]:
        """
        Get ZZ split measurements for the observable correction.
        
        In proper lattice surgery, the product of ALL seam Z-stab
        eigenvalues gives Z_L_ctrl ⊗ Z_L_anc.  Therefore the XOR of ALL
        seam Z-stab split measurements is the correction for Z_tgt_out.
        """
        return list(self._zz_split_meas)

    # ------------------------------------------------------------------
    # Destroyed/surviving blocks
    # ------------------------------------------------------------------

    def get_destroyed_blocks(self) -> Set[str]:
        return {"block_1"}

    def get_ancilla_block_names(self) -> Set[str]:
        return {"block_1"}

    def get_blocks_to_skip_pre_rounds(self) -> Set[str]:
        """Blocks to skip during pre-gadget EC rounds.

        CSS Surgery requires d rounds of EC on ALL blocks for state
        preparation (establishing stabilizer eigenvalues) before the
        first merge phase begins.

        - block_0 (control): prepared in |0⟩, needs Z-basis EC to
          establish Z-stabilizer eigenvalues.
        - block_1 (ancilla): prepared in |+⟩, needs X-basis EC.
        - block_2 (target): prepared in |0⟩, needs Z-basis EC.

        All blocks participate in pre-gadget rounds.  The anchor
        detectors emitted during round 0 use the prep config from
        get_preparation_config().
        """
        return set()  # all blocks get pre-gadget EC

    def get_blocks_to_skip_post_rounds(self) -> Set[str]:
        return {"block_1"}

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
        Crossing detectors for the net CNOT(block_0 → block_2) operation.

        Return ``None`` for all code types.

        - **Flat codes**: merge-phase temporal detectors plus post-gadget
          anchor detectors provide full coverage.
        - **Hierarchical codes**: the builder's native temporal detector
          emission on the first post-gadget round already handles inner
          and outer stabiliser comparisons correctly.  Explicit crossing
          formulas fail ``has_flow`` for inner stabilisers because they
          reference raw inner measurement indices across the gadget
          boundary where the concatenated stabiliser structure changes.
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
        
        surgery_gap = 2.0
        layout.add_block("block_0", ctrl_code, offset=(0.0, 0.0))

        has_boundaries = (
            hasattr(ctrl_code, 'has_physical_boundaries') and
            ctrl_code.has_physical_boundaries()
        )

        if has_boundaries:
            # L-shaped layout following rotated surface code boundary
            # conventions (§Horsman et al. 2012, Litinski 2019):
            #
            #   [block_0 control]
            #   [block_1 ancilla] [block_2 target]
            #
            # Boundary types (rotated surface code):
            #   left/right  → smooth (X-type)
            #   top/bottom  → rough  (Z-type)
            #
            # ZZ merge (rough↔rough): block_0 bottom ↔ block_1 top
            # XX merge (smooth↔smooth): block_1 right ↔ block_2 left
            #
            # block_1 below block_0 (ZZ merge boundary)
            layout.add_block_adjacent("block_1", anc_code, reference_block="block_0",
                                      my_edge="top", their_edge="bottom", gap=surgery_gap)
            # block_2 to the RIGHT of block_1 (XX merge boundary)
            layout.add_block_adjacent("block_2", tgt_code, reference_block="block_1",
                                      my_edge="left", their_edge="right", gap=surgery_gap)

            # ---- Seam stabiliser allocation (proper lattice surgery) ----
            # Build local→global data maps for each block.
            blk0 = layout.blocks["block_0"]
            blk1 = layout.blocks["block_1"]
            blk2 = layout.blocks["block_2"]

            ctrl_data_map = {
                i: blk0.data_qubit_range.start + i
                for i in range(len(blk0.data_qubit_range))
            }
            anc_data_map = {
                i: blk1.data_qubit_range.start + i
                for i in range(len(blk1.data_qubit_range))
            }
            tgt_data_map = {
                i: blk2.data_qubit_range.start + i
                for i in range(len(blk2.data_qubit_range))
            }

            # ZZ merge: seam between block_0 bottom and block_1 top.
            # Seam Z-stabilisers are weight-2/4 plaquettes at even–even
            # lattice positions along the shared rough boundary.
            zz_seam_offset = layout._next_global_idx
            zz_info = ctrl_code.get_merge_stabilizers(
                "ZZ", "bottom", anc_code, "top",
                ctrl_data_map, anc_data_map, zz_seam_offset,
            )
            if not zz_info.seam_stabs:
                raise ValueError(
                    f"{type(ctrl_code).__name__}.get_merge_stabilizers('ZZ', ...) "
                    "returned empty seam stabilizers. Lattice surgery requires "
                    "the code to produce proper seam plaquettes for the ZZ merge."
                )
            layout.add_seam_stabilizers(zz_info, "block_0", "block_1",
                                        "zz_merge")
            self._resolve_grown_stabs(zz_info, ctrl_code, blk0, "block_0")
            self._zz_merge_info = zz_info

            # XX merge: seam between block_1 right and block_2 left.
            xx_seam_offset = layout._next_global_idx
            xx_info = anc_code.get_merge_stabilizers(
                "XX", "right", tgt_code, "left",
                anc_data_map, tgt_data_map, xx_seam_offset,
            )
            if not xx_info.seam_stabs:
                raise ValueError(
                    f"{type(anc_code).__name__}.get_merge_stabilizers('XX', ...) "
                    "returned empty seam stabilizers. Lattice surgery requires "
                    "the code to produce proper seam plaquettes for the XX merge."
                )
            layout.add_seam_stabilizers(xx_info, "block_1", "block_2",
                                        "xx_merge")
            self._resolve_grown_stabs(xx_info, anc_code, blk1, "block_1")
            self._xx_merge_info = xx_info
        else:
            raise NotImplementedError(
                f"{type(ctrl_code).__name__} does not support lattice surgery: "
                "the code must have physical boundaries (has_physical_boundaries()) "
                "and implement get_merge_stabilizers(). Currently supported: "
                "RotatedSurfaceCode."
            )

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
