# src/qectostim/gadgets/configs.py
"""

This module contains all declarative configuration types used by gadgets
and experiments to communicate preparation, measurement, detector, and
observable requirements.  Only **gadget-independent** factory methods live
here; gadget-specific factories (``cz_teleportation``, ``cnot_teleportation``,
etc.) are defined in the gadget modules that need them and are attached to
the config classes via ``_register_factory``.

Classes
-------
ObservableTerm, ObservableConfig
    Observable specification (output blocks, correlation terms, frame corrections).
BlockPreparationConfig, PreparationConfig
    Per-block state preparation configuration.
MeasurementConfig
    Per-block final measurement basis.
CrossingDetectorTerm, CrossingDetectorFormula, CrossingDetectorConfig
    Crossing detector specification across a transversal gate.
BoundaryDetectorConfig
    Boundary (space-like) detector specification.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from qectostim.gadgets.base import TwoQubitObservableTransform


# ═════════════════════════════════════════════════════════════════════════════
# Heisenberg Observable Derivation
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class HeisenbergTerm:
    """A single term in a Heisenberg-picture output operator.

    Represents a Pauli factor from a specific block that appears in
    the output operator after conjugation through the gate.

    Attributes
    ----------
    block : str
        Block name ("block_0", "data_block", "ancilla_block", etc.).
    basis : str
        Pauli basis of this factor ("X" or "Z").
    """
    block: str
    basis: str  # "X" or "Z"


@dataclass
class HeisenbergOperator:
    """A Heisenberg-picture output operator for one logical Pauli.

    After conjugation through the gate, a logical Pauli $P_{\\text{block}}$
    becomes a product of Pauli factors on (possibly multiple) blocks, plus
    optional protocol correction terms (mid-circuit measurements).

    Example: CNOT maps $Z_{\\text{tgt}} \\to Z_{\\text{ctrl}} \\otimes Z_{\\text{tgt}}$.
    This would be represented as::

        HeisenbergOperator(
            block="target",
            basis="Z",
            output_terms=[
                HeisenbergTerm("control", "Z"),
                HeisenbergTerm("target", "Z"),
            ],
        )

    Attributes
    ----------
    block : str
        The block this operator originates from.
    basis : str
        The Pauli basis of the input operator ("X" or "Z").
    output_terms : List[HeisenbergTerm]
        Pauli factors in the output (after gate conjugation).
        These are the "bare" operator terms (code-block logicals).
    correction_basis : str
        Basis of the protocol corrections ("X", "Z", or "" for none).
        This is the basis of the dressing that comes from mid-circuit
        measurements (e.g., ancilla MX for teleportation, bridge
        measurements for surgery).
    """
    block: str
    basis: str  # "X" or "Z" — the INPUT operator
    output_terms: List[HeisenbergTerm] = field(default_factory=list)
    correction_basis: str = ""  # "" means no protocol corrections

    @property
    def is_identity(self) -> bool:
        """True if operator maps to itself (single term, same block+basis)."""
        return (
            len(self.output_terms) == 1
            and self.output_terms[0].block == self.block
            and self.output_terms[0].basis == self.basis
        )


@dataclass
class HeisenbergFrame:
    """Complete Heisenberg frame for a gate: all output operators.

    Gadgets declare one ``HeisenbergFrame`` describing how ALL logical
    Paulis transform through the gate.  The universal derivation in
    ``ObservableConfig.from_heisenberg()`` then automatically determines
    which blocks/terms to include in the observable for any input state.

    The ``block_bases`` callback maps input states to per-block measurement
    bases, keeping the Heisenberg operators universal (input-independent).

    Attributes
    ----------
    operators : List[HeisenbergOperator]
        All Heisenberg output operators (typically 2 per block: X and Z).
    get_block_bases : Callable
        Function ``(input_state: str) -> Dict[str, str]`` returning
        per-block measurement basis for a given input state.
        For multi-block gates, input_state may encode multiple blocks
        (e.g., "00", "+0", "0+", "++").
    get_deterministic_bases : Optional[Callable]
        Function ``(input_state: str) -> Dict[str, Set[str]]`` returning
        per-block set of Pauli bases that are deterministic due to the
        input eigenstate.  For example, |0⟩ makes Z deterministic (+1),
        |+⟩ makes X deterministic (+1).  This allows cross-block terms
        to be treated as deterministic even when the other block is
        measured in a different basis.
    correction_provider : Optional[Callable]
        Optional function ``(basis: str) -> List[int]`` returning raw
        measurement indices for protocol corrections of the given basis.
        Used by surgery gadgets to supply bridge measurement indices.
        Returns empty list if no corrections for that basis.
    """
    operators: List[HeisenbergOperator] = field(default_factory=list)
    get_block_bases: Optional[Callable[..., Dict[str, str]]] = None
    get_deterministic_bases: Optional[Callable[..., Dict[str, Set[str]]]] = None
    correction_provider: Optional[Callable[[str], List[int]]] = None

    def get_operators_for_block(
        self, block: str, basis: str
    ) -> Optional[HeisenbergOperator]:
        """Get the Heisenberg operator for a specific block and basis."""
        for op in self.operators:
            if op.block == block and op.basis == basis:
                return op
        return None

    def derive_observable(
        self,
        input_state: str,
    ) -> "ObservableConfig":
        """Derive ObservableConfig from this Heisenberg frame.

        Universal algorithm:

        1. Get per-block measurement bases from input state.
        2. For each block, find the Heisenberg operator matching the
           measurement basis (the operator being "checked").
        3. Check determinism: every output_term must be deterministic via:
           (a) measurement determinism — other block measured in same basis,
           (b) eigenstate determinism — other block in eigenstate of that basis,
           (c) correction determinism — operator has a correction_basis that
               matches measurement_basis, meaning the cross-block random term
               is compensated by including the other block's measurement as a
               correlation/frame-correction term.
        4. If deterministic, include the block in the observable.
           For case (c), also include the correcting block as a correlation term.
        5. Collect same-basis protocol corrections (bridge indices).
        6. GF(2)-deduplicate bridge corrections.

        Parameters
        ----------
        input_state : str
            Input state string (e.g., "0", "+", "00", "+0").

        Returns
        -------
        ObservableConfig
            Fully derived observable configuration.
        """
        if self.get_block_bases is None:
            raise ValueError("HeisenbergFrame.get_block_bases must be set")

        bases = self.get_block_bases(input_state)
        det_bases = (
            self.get_deterministic_bases(input_state)
            if self.get_deterministic_bases is not None
            else {}
        )

        # Collect all blocks
        all_blocks = list(bases.keys())

        correlation_terms: List[ObservableTerm] = []
        output_blocks: List[str] = []
        bridge_raw: List[int] = []

        for block in all_blocks:
            meas_basis = bases[block]

            # Find the Heisenberg operator for this block+basis
            op = self.get_operators_for_block(block, meas_basis)
            if op is None:
                # No operator declared → skip this block
                continue

            # Check determinism of every cross-block output term
            deterministic = True
            # correction_blocks: blocks whose measurements are needed
            # as correlation terms.  Tuple of (block, basis, is_frame_corr).
            # is_frame_corr=True means it's a frame correction from a 
            # destroyed block (should be prepended); False means it's a
            # peer block in a multi-block observable.
            correction_blocks: List[Tuple[str, str, bool]] = []
            for term in op.output_terms:
                if term.block == block:
                    # Same block, same basis → always deterministic
                    continue

                # (a) Measurement determinism: other block measured in same basis
                other_meas = bases.get(term.block)
                if other_meas == term.basis:
                    # Other block measured in the right basis → deterministic
                    # This is a peer block, not a frame correction
                    correction_blocks.append((term.block, term.basis, False))
                    continue

                # (b) Eigenstate determinism: other block is eigenstate
                block_det = det_bases.get(term.block, set())
                if term.basis in block_det:
                    # Eigenstate makes this term +1 → deterministic, no corr needed
                    continue

                # (c) Correction determinism (same-basis rule):
                # If this operator has a correction_basis matching the
                # measurement basis, the random cross-block term is
                # compensated by including the other block's measurement
                # as a frame correction.
                if (op.correction_basis == meas_basis
                        and term.block not in bases):
                    # Destroyed block → frame correction (prepend)
                    correction_blocks.append((term.block, "X", True))
                    continue

                # None of the above → term is random → block not observable
                deterministic = False
                break

            if not deterministic:
                continue

            # Same-basis frame correction rule (teleportation |+⟩ cases):
            # When the measured operator is trivially deterministic (no cross-block
            # terms), check if the OTHER operator for this block has a 
            # correction_basis that matches our measurement basis.  If so, the
            # protocol's frame correction from the destroyed block must be
            # included in the observable.
            if not correction_blocks:
                for other_op in self.operators:
                    if other_op.block == block and other_op.basis != meas_basis:
                        if other_op.correction_basis == meas_basis:
                            for t in other_op.output_terms:
                                if t.block != block and t.block not in bases:
                                    correction_blocks.append((t.block, "X", True))

            # Block is observable.
            # Frame corrections (from destroyed blocks) go BEFORE the main block.
            # Peer corrections (from measured blocks) are just noted; they'll be
            # added when those blocks are processed in the main loop.
            for cb, cb_basis, is_frame in correction_blocks:
                if is_frame and cb not in output_blocks:
                    output_blocks.append(cb)
                    correlation_terms.append(ObservableTerm(block=cb, basis=cb_basis))
            if block not in output_blocks:
                output_blocks.append(block)
                correlation_terms.append(ObservableTerm(block=block, basis=meas_basis))

            # Collect same-basis protocol corrections (bridge indices)
            if op.correction_basis and self.correction_provider:
                corrections = self.correction_provider(op.correction_basis)
                bridge_raw.extend(corrections)

        # GF(2) deduplication — cancel even multiplicities
        counts = Counter(bridge_raw)
        bridge_deduped = [idx for idx, cnt in counts.items() if cnt % 2 == 1]

        return ObservableConfig(
            output_blocks=output_blocks,
            block_bases=bases,
            correlation_terms=correlation_terms,
            bridge_frame_meas_indices=bridge_deduped,
        )

    # ── Pre-built frames for common gate types ───────────────────────────

    @classmethod
    def identity(cls) -> "HeisenbergFrame":
        """Identity gate: all operators unchanged.

        X_D → X_D, Z_D → Z_D.
        """
        def _bases(input_state: str) -> Dict[str, str]:
            return {"data_block": "X" if input_state in ("+", "-") else "Z"}

        return cls(
            operators=[
                HeisenbergOperator("data_block", "X", [
                    HeisenbergTerm("data_block", "X"),
                ]),
                HeisenbergOperator("data_block", "Z", [
                    HeisenbergTerm("data_block", "Z"),
                ]),
            ],
            get_block_bases=_bases,
        )

    @classmethod
    def transversal_cnot(cls) -> "HeisenbergFrame":
        """Transversal CNOT (block_0=control, block_1=target).

        Heisenberg operators::

            X_ctrl → X_ctrl ⊗ X_tgt   (control X spreads)
            Z_ctrl → Z_ctrl            (control Z unchanged)
            X_tgt  → X_tgt             (target X unchanged)
            Z_tgt  → Z_ctrl ⊗ Z_tgt   (target Z picks up control Z)
        """
        def _bases(input_state: str) -> Dict[str, str]:
            # input_state encodes ctrl+tgt: "00", "+0", "0+", "++"
            ctrl_in = input_state[0] if len(input_state) > 1 else input_state
            tgt_in = input_state[1] if len(input_state) > 1 else "0"

            ctrl_basis = "X" if ctrl_in == "+" else "Z"

            # CNOT output state truth table
            if ctrl_in == "0" and tgt_in == "0":
                tgt_basis = "Z"
            elif ctrl_in == "+" and tgt_in == "0":
                tgt_basis = "X"
            elif ctrl_in == "0" and tgt_in == "+":
                tgt_basis = "X"
            else:  # "++"
                tgt_basis = "Z"

            return {"block_0": ctrl_basis, "block_1": tgt_basis}

        def _det_bases(input_state: str) -> Dict[str, Set[str]]:
            ctrl_in = input_state[0] if len(input_state) > 1 else input_state
            tgt_in = input_state[1] if len(input_state) > 1 else "0"
            result: Dict[str, Set[str]] = {}
            result["block_0"] = {"X"} if ctrl_in == "+" else {"Z"}
            result["block_1"] = {"X"} if tgt_in == "+" else {"Z"}
            return result

        return cls(
            operators=[
                # X_ctrl → X_ctrl ⊗ X_tgt
                HeisenbergOperator("block_0", "X", [
                    HeisenbergTerm("block_0", "X"),
                    HeisenbergTerm("block_1", "X"),
                ]),
                # Z_ctrl → Z_ctrl
                HeisenbergOperator("block_0", "Z", [
                    HeisenbergTerm("block_0", "Z"),
                ]),
                # X_tgt → X_tgt
                HeisenbergOperator("block_1", "X", [
                    HeisenbergTerm("block_1", "X"),
                ]),
                # Z_tgt → Z_ctrl ⊗ Z_tgt
                HeisenbergOperator("block_1", "Z", [
                    HeisenbergTerm("block_0", "Z"),
                    HeisenbergTerm("block_1", "Z"),
                ]),
            ],
            get_block_bases=_bases,
            get_deterministic_bases=_det_bases,
        )

    @classmethod
    def transversal_cz(cls) -> "HeisenbergFrame":
        """Transversal CZ (symmetric on block_0, block_1).

        Heisenberg operators::

            X_0 → X_0 ⊗ Z_1   (X picks up Z from other block)
            Z_0 → Z_0          (Z unchanged)
            X_1 → Z_0 ⊗ X_1   (X picks up Z from other block)
            Z_1 → Z_1          (Z unchanged)
        """
        def _bases(input_state: str) -> Dict[str, str]:
            b0_in = input_state[0] if len(input_state) > 1 else input_state
            b1_in = input_state[1] if len(input_state) > 1 else "0"
            return {
                "block_0": "X" if b0_in == "+" else "Z",
                "block_1": "X" if b1_in == "+" else "Z",
            }

        def _det_bases(input_state: str) -> Dict[str, Set[str]]:
            b0_in = input_state[0] if len(input_state) > 1 else input_state
            b1_in = input_state[1] if len(input_state) > 1 else "0"
            result: Dict[str, Set[str]] = {}
            result["block_0"] = {"X"} if b0_in == "+" else {"Z"}
            result["block_1"] = {"X"} if b1_in == "+" else {"Z"}
            return result

        return cls(
            operators=[
                HeisenbergOperator("block_0", "X", [
                    HeisenbergTerm("block_0", "X"),
                    HeisenbergTerm("block_1", "Z"),
                ]),
                HeisenbergOperator("block_0", "Z", [
                    HeisenbergTerm("block_0", "Z"),
                ]),
                HeisenbergOperator("block_1", "X", [
                    HeisenbergTerm("block_0", "Z"),
                    HeisenbergTerm("block_1", "X"),
                ]),
                HeisenbergOperator("block_1", "Z", [
                    HeisenbergTerm("block_1", "Z"),
                ]),
            ],
            get_block_bases=_bases,
            get_deterministic_bases=_det_bases,
        )

    @classmethod
    def cz_teleportation(cls) -> "HeisenbergFrame":
        """CZ H-teleportation (data_block → ancilla_block).

        Heisenberg operators on the ancilla (output block)::

            X_A^out = Z_D ⊗ X_A   → correction_basis = "Z"
            Z_A^out = Z_A          → no correction

        Data is always measured MX.  Ancilla measurement basis depends
        on input state: |0⟩ → MX, |+⟩ → MZ.

        Eigenstate determinism: |0⟩ makes Z_D deterministic (+1), so
        X_A^out = (+1) ⊗ X_A = X_A is deterministic for |0⟩.
        """
        def _bases(input_state: str) -> Dict[str, str]:
            ancilla_basis = "X" if input_state in ("0", "1") else "Z"
            return {
                "ancilla_block": ancilla_basis,
            }

        def _det_bases(input_state: str) -> Dict[str, Set[str]]:
            # |0⟩/|1⟩: data_block Z is deterministic (Z eigenstate)
            # |+⟩/|-⟩: data_block X is deterministic (X eigenstate)
            if input_state in ("0", "1"):
                return {"data_block": {"Z"}}
            else:
                return {"data_block": {"X"}}

        return cls(
            operators=[
                # X_A^out = Z_D ⊗ X_A — correction is Z-type (data MX = Z corr)
                HeisenbergOperator("ancilla_block", "X", [
                    HeisenbergTerm("data_block", "Z"),
                    HeisenbergTerm("ancilla_block", "X"),
                ], correction_basis="Z"),
                # Z_A^out = Z_A — no correction
                HeisenbergOperator("ancilla_block", "Z", [
                    HeisenbergTerm("ancilla_block", "Z"),
                ]),
            ],
            get_block_bases=_bases,
            get_deterministic_bases=_det_bases,
        )

    @classmethod
    def cnot_teleportation(cls) -> "HeisenbergFrame":
        """CNOT H-teleportation (data=control → ancilla=target).

        Heisenberg operators on the ancilla (output block)::

            X_A^out = X_A              → no correction
            Z_A^out = Z_D ⊗ Z_A       → correction_basis = "X"

        Data is always measured MX.  Ancilla measurement basis depends
        on input state: |0⟩ → MZ, |+⟩ → MX.

        Eigenstate determinism: |0⟩ makes Z_D deterministic (+1), so
        Z_A^out = (+1) ⊗ Z_A = Z_A is deterministic for |0⟩.
        """
        def _bases(input_state: str) -> Dict[str, str]:
            ancilla_basis = "Z" if input_state in ("0", "1") else "X"
            return {
                "ancilla_block": ancilla_basis,
            }

        def _det_bases(input_state: str) -> Dict[str, Set[str]]:
            if input_state in ("0", "1"):
                return {"data_block": {"Z"}}
            else:
                return {"data_block": {"X"}}

        return cls(
            operators=[
                # X_A^out = X_A — no correction
                HeisenbergOperator("ancilla_block", "X", [
                    HeisenbergTerm("ancilla_block", "X"),
                ]),
                # Z_A^out = Z_D ⊗ Z_A — correction is X-type (data MX = X corr)
                HeisenbergOperator("ancilla_block", "Z", [
                    HeisenbergTerm("data_block", "Z"),
                    HeisenbergTerm("ancilla_block", "Z"),
                ], correction_basis="X"),
            ],
            get_block_bases=_bases,
            get_deterministic_bases=_det_bases,
        )

    @classmethod
    def surgery_cnot(
        cls,
        correction_provider: Optional[Callable[[str], List[int]]] = None,
    ) -> "HeisenbergFrame":
        """Lattice surgery CNOT (block_0=ctrl, block_1=ancilla, block_2=tgt).

        Heisenberg operators (block_1 is destroyed during surgery)::

            Z_ctrl_out = Z_ctrl                                (no correction)
            X_ctrl_out = X_ctrl ⊕ m_anc_XL                    (X correction)
            Z_tgt_out  = Z_tgt ⊕ Z_ctrl ⊕ s_zz_log           (Z correction)
            X_tgt_out  = X_tgt ⊕ m_anc_XL ⊕ s_xx_log         (X correction)

        Parameters
        ----------
        correction_provider : callable, optional
            ``(basis: str) -> List[int]`` returning raw measurement indices
            for protocol corrections.  "X" returns m_anc_XL + s_xx_log;
            "Z" returns s_zz_log.
        """
        def _bases(input_state: str) -> Dict[str, str]:
            # input_state encodes ctrl+tgt: "00", "+0", "0+", "++"
            ctrl_in = input_state[0] if len(input_state) > 1 else input_state
            tgt_in = input_state[1] if len(input_state) > 1 else "0"
            return {
                "block_0": "X" if ctrl_in == "+" else "Z",
                "block_2": "X" if tgt_in == "+" else "Z",
            }

        return cls(
            operators=[
                # Z_ctrl_out = Z_ctrl (no correction)
                HeisenbergOperator("block_0", "Z", [
                    HeisenbergTerm("block_0", "Z"),
                ]),
                # X_ctrl_out = X_ctrl (X correction from ancilla)
                HeisenbergOperator("block_0", "X", [
                    HeisenbergTerm("block_0", "X"),
                ], correction_basis="X"),
                # Z_tgt_out = Z_tgt ⊕ Z_ctrl (Z correction from bridges)
                HeisenbergOperator("block_2", "Z", [
                    HeisenbergTerm("block_0", "Z"),
                    HeisenbergTerm("block_2", "Z"),
                ], correction_basis="Z"),
                # X_tgt_out = X_tgt (X correction from ancilla + bridges)
                HeisenbergOperator("block_2", "X", [
                    HeisenbergTerm("block_2", "X"),
                ], correction_basis="X"),
            ],
            get_block_bases=_bases,
            correction_provider=correction_provider,
        )

    @classmethod
    def knill_ec(cls) -> "HeisenbergFrame":
        """Knill EC (Bell-state teleportation, 3-block protocol).

        Blocks: data_block (input), bell_a, bell_b (output).

        Protocol::

            1. bell_a = |+⟩, bell_b = |0⟩
            2. CNOT(bell_a → bell_b) creates Bell pair
            3. CNOT(data → bell_a), MX(data), MZ(bell_a)
            4. Output on bell_b with Pauli corrections

        Heisenberg operators on the output block (bell_b)::

            X_B^out = X_B                  (no cross-block)
            Z_B^out = Z_D ⊗ Z_A ⊗ Z_B     (cross-block, corrected by MZ(bell_a))

        Both data_block and bell_a are destroyed (measured).
        Corrections are handled by the Pauli frame update, so the
        observable is simply the logical operator on bell_b.

        For |0⟩ input: Z_D is deterministic (+1) and Z_A is corrected
        by frame → measure Z on bell_b.
        For |+⟩ input: X_B is trivially deterministic → measure X on bell_b.
        """
        def _bases(input_state: str) -> Dict[str, str]:
            basis = "X" if input_state in ("+", "-") else "Z"
            return {"bell_b": basis}

        def _det_bases(input_state: str) -> Dict[str, Set[str]]:
            if input_state in ("0", "1"):
                return {"data_block": {"Z"}, "bell_a": {"X"}, "bell_b": {"Z"}}
            else:
                return {"data_block": {"X"}, "bell_a": {"X"}, "bell_b": {"Z"}}

        return cls(
            operators=[
                # X_B^out = X_B (trivial, no cross-block)
                HeisenbergOperator("bell_b", "X", [
                    HeisenbergTerm("bell_b", "X"),
                ]),
                # Z_B^out = Z_D ⊗ Z_A ⊗ Z_B
                # Z_D and Z_A are on destroyed blocks; their corrections
                # are handled by the Pauli frame (not protocol corrections).
                # For the HeisenbergFrame to resolve determinism:
                # |0⟩: Z_D is eigenstate-det, Z_A is eigenstate-det (bell_a prep |+⟩ → X det, but Z is random)
                # Actually, bell_a is prep |+⟩ so Z_A is random. But the Pauli frame
                # corrects for it, and the experiment handles it via FrameUpdate.
                # So for the observable, bell_b alone is sufficient — the frame
                # update includes MX(data) and MZ(bell_a) corrections.
                #
                # We model this as: Z_B^out = Z_B (after frame correction absorbs
                # the cross-block terms). This matches the physical reality that
                # the FrameUpdate in Phase 3 already handles the teleportation
                # corrections, and the observable just needs to check bell_b.
                HeisenbergOperator("bell_b", "Z", [
                    HeisenbergTerm("bell_b", "Z"),
                ]),
            ],
            get_block_bases=_bases,
            get_deterministic_bases=_det_bases,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Observable Configuration
# ═════════════════════════════════════════════════════════════════════════════

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
    all cases uniformly without ``isinstance`` checks.

    Attributes
    ----------
    output_blocks : List[str]
        Which blocks contribute to the output observable.
    block_bases : Dict[str, str]
        The Pauli basis to use for each output block.
    frame_correction_blocks : List[str]
        Blocks whose measurements contribute to frame correction.
    frame_correction_basis : str
        The Pauli type of the frame correction ("X", "Z", or "XZ").
    requires_raw_sampling : bool
        If True, skip OBSERVABLE_INCLUDE entirely.
    two_qubit_transform : Optional[TwoQubitObservableTransform]
        For two-qubit gates, how observables transform across blocks.
    use_hybrid_decoding : bool
        If True, emit clean observables without frame measurements.
    correlation_terms : List[ObservableTerm]
        Multi-block correlation observable terms.
    bridge_frame_meas_indices : List[int]
        Bridge measurement frame corrections for lattice surgery.
    """
    output_blocks: List[str] = field(default_factory=lambda: ["data_block"])
    block_bases: Dict[str, str] = field(default_factory=dict)
    frame_correction_blocks: List[str] = field(default_factory=list)
    frame_correction_basis: str = "Z"
    requires_raw_sampling: bool = False
    two_qubit_transform: Optional["TwoQubitObservableTransform"] = None
    use_hybrid_decoding: bool = False

    # Multi-block correlation observables
    correlation_terms: List[ObservableTerm] = field(default_factory=list)

    # Bridge measurement frame corrections for lattice surgery
    bridge_frame_meas_indices: List[int] = field(default_factory=list)

    # ── Generic factory methods ──────────────────────────────────────────

    @classmethod
    def transversal_single_qubit(
        cls,
        basis_transform: Optional[Dict[str, str]] = None,
    ) -> "ObservableConfig":
        """Config for transversal single-qubit gates (H, S, T, X, Y, Z)."""
        return cls(
            output_blocks=["data_block"],
            block_bases={},
        )

    @classmethod
    def transversal_two_qubit(
        cls,
        transform: "TwoQubitObservableTransform",
    ) -> "ObservableConfig":
        """Config for transversal two-qubit gates (CNOT, CZ, SWAP)."""
        return cls(
            output_blocks=["block_0", "block_1"],
            two_qubit_transform=transform,
        )

    @classmethod
    def teleportation(
        cls,
        gate: str,
        input_state: str,
    ) -> "ObservableConfig":
        """
        Universal Heisenberg derivation for teleportation H-gadgets.

        Delegates to ``HeisenbergFrame.cz_teleportation()`` or
        ``HeisenbergFrame.cnot_teleportation()`` for automatic derivation.

        Parameters
        ----------
        gate : str
            "CZ" or "CNOT".
        input_state : str
            "0" or "+".
        """
        if gate == "CZ":
            frame = HeisenbergFrame.cz_teleportation()
        else:
            frame = HeisenbergFrame.cnot_teleportation()
        return cls.from_heisenberg(frame, input_state)

    @classmethod
    def bell_teleportation(
        cls,
        output_block: str = "block_2",
        bell_blocks: Optional[List[str]] = None,
        frame_basis: str = "XZ",
    ) -> "ObservableConfig":
        """
        Config for Bell-state teleportation (3-block protocol).

        Output is on ancilla2 (block_2). Frame corrections come from
        the Bell measurement on data + ancilla1.
        """
        if bell_blocks is None:
            bell_blocks = ["block_0", "block_1"]
        return cls(
            output_blocks=[output_block],
            frame_correction_blocks=bell_blocks,
            frame_correction_basis=frame_basis,
        )

    @classmethod
    def from_heisenberg(
        cls,
        frame: "HeisenbergFrame",
        input_state: str,
        **overrides,
    ) -> "ObservableConfig":
        """Derive an ObservableConfig from a Heisenberg frame.

        This is the **universal** entry point: declare the gate's Heisenberg
        operators once, and this method handles any input state automatically.

        Parameters
        ----------
        frame : HeisenbergFrame
            Complete Heisenberg frame for the gate.
        input_state : str
            Input state (e.g., "0", "+", "00", "+0").
        **overrides
            Extra fields to set on the resulting ObservableConfig
            (e.g., ``use_hybrid_decoding=True``).

        Returns
        -------
        ObservableConfig
            Fully derived observable configuration.
        """
        config = frame.derive_observable(input_state)
        for key, val in overrides.items():
            if hasattr(config, key):
                setattr(config, key, val)
        return config


# ═════════════════════════════════════════════════════════════════════════════
# Preparation Configuration
# ═════════════════════════════════════════════════════════════════════════════

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
    x_deterministic : bool
        Whether X stabilizers are deterministic after preparation.
    skip_experiment_prep : bool
        If True, the gadget handles this block's preparation itself.
    """
    initial_state: str = "0"
    z_deterministic: bool = True
    x_deterministic: bool = False
    skip_experiment_prep: bool = False

    @classmethod
    def zero_state(cls, skip_prep: bool = False) -> "BlockPreparationConfig":
        """|0⟩ state: Z deterministic, X random."""
        return cls(
            initial_state="0",
            z_deterministic=True,
            x_deterministic=False,
            skip_experiment_prep=skip_prep,
        )

    @classmethod
    def plus_state(cls, skip_prep: bool = False) -> "BlockPreparationConfig":
        """|+⟩ state: X deterministic, Z random."""
        return cls(
            initial_state="+",
            z_deterministic=False,
            x_deterministic=True,
            skip_experiment_prep=skip_prep,
        )


@dataclass
class PreparationConfig:
    """
    Configuration for state preparation across all blocks.

    Gadgets return this to declare per-block initial states and determinism.
    The experiment and library modules consume this config generically.

    Block Name Normalization
    ------------------------
    Teleportation gadgets use semantic names ("data_block", "ancilla_block")
    while the experiment's QubitAllocation may use generic names ("block_0",
    "block_1").  ``get_block_config()`` normalises automatically.

    Attributes
    ----------
    blocks : Dict[str, BlockPreparationConfig]
        Per-block preparation configuration.
    """
    blocks: Dict[str, BlockPreparationConfig] = field(default_factory=dict)

    # Class-level block name aliases for normalization
    BLOCK_ALIASES: Dict[str, str] = field(
        default_factory=lambda: {
            "block_0": "data_block",
            "block_1": "ancilla_block",
            "block_2": "ancilla_block_2",
        },
        repr=False,
        init=False,
    )

    def __post_init__(self):
        """Initialize block aliases after dataclass init."""
        object.__setattr__(self, "BLOCK_ALIASES", {
            "block_0": "data_block",
            "block_1": "ancilla_block",
            "block_2": "ancilla_block_2",
        })

    def get_block_config(
        self, block_name: str
    ) -> Optional[BlockPreparationConfig]:
        """
        Get configuration for a block with name normalization.

        Handles the mapping between generic names (block_0, block_1) and
        semantic names (data_block, ancilla_block).
        """
        # Direct lookup
        if block_name in self.blocks:
            return self.blocks[block_name]

        # Alias lookup (block_0 → data_block, etc.)
        alias = self.BLOCK_ALIASES.get(block_name)
        if alias and alias in self.blocks:
            return self.blocks[alias]

        # Reverse alias (data_block → block_0)
        for generic, semantic in self.BLOCK_ALIASES.items():
            if block_name == semantic and generic in self.blocks:
                return self.blocks[generic]

        return None

    def get_normalized_block_name(self, block_name: str) -> str:
        """
        Normalize a block name to the name used in this config.
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

    # ── Generic factory methods ──────────────────────────────────────────

    @classmethod
    def single_block(cls, state: str = "0") -> "PreparationConfig":
        """Config for single-block gadget (transversal gates)."""
        if state in ("0", "1"):
            return cls(blocks={"data_block": BlockPreparationConfig.zero_state()})
        else:
            return cls(blocks={"data_block": BlockPreparationConfig.plus_state()})

    @classmethod
    def cz_teleportation(cls, input_state: str = "0") -> "PreparationConfig":
        """
        Config for CZ H-teleportation: data=input, ancilla=|+⟩.

        - Data block: prepared OUTSIDE gadget (experiments/preparation.py)
        - Ancilla block: prepared INSIDE gadget in |+⟩ via RX (Phase 1)
        """
        if input_state in ("0", "1"):
            data_config = BlockPreparationConfig.zero_state()
        else:
            data_config = BlockPreparationConfig.plus_state()

        ancilla_config = BlockPreparationConfig.plus_state(skip_prep=True)

        return cls(blocks={
            "data_block": data_config,
            "ancilla_block": ancilla_config,
        })

    @classmethod
    def cnot_teleportation(cls, input_state: str = "0") -> "PreparationConfig":
        """
        Config for CNOT H-teleportation: data=input, ancilla=|0⟩.

        - Data block: prepared OUTSIDE gadget (experiments/preparation.py)
        - Ancilla block: prepared INSIDE gadget in |0⟩ via R (Phase 1)
        """
        if input_state in ("0", "1"):
            data_config = BlockPreparationConfig.zero_state()
        else:
            data_config = BlockPreparationConfig.plus_state()

        ancilla_config = BlockPreparationConfig.zero_state(skip_prep=True)

        return cls(blocks={
            "data_block": data_config,
            "ancilla_block": ancilla_config,
        })


# ═════════════════════════════════════════════════════════════════════════════
# Measurement Configuration
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MeasurementConfig:
    """
    Configuration for final measurements across all blocks.

    Attributes
    ----------
    block_bases : Dict[str, str]
        Per-block measurement basis ("X", "Z", or "Y").
    destroyed_blocks : Set[str]
        Blocks destroyed (measured out) during the gadget.
    """
    block_bases: Dict[str, str] = field(default_factory=dict)
    destroyed_blocks: Set[str] = field(default_factory=set)

    # ── Generic factory methods ──────────────────────────────────────────

    @classmethod
    def single_block(cls, basis: str = "Z") -> "MeasurementConfig":
        """Config for single-block gadget."""
        return cls(block_bases={"data_block": basis})

    @classmethod
    def cz_teleportation(cls, input_state: str = "0") -> "MeasurementConfig":
        """
        Config for CZ H-teleportation measurements.

        Data always measured in X; ancilla basis depends on input state.
        """
        ancilla_basis = "X" if input_state in ("0", "1") else "Z"
        return cls(
            block_bases={"data_block": "X", "ancilla_block": ancilla_basis},
        )

    @classmethod
    def cnot_teleportation(cls, input_state: str = "0") -> "MeasurementConfig":
        """
        Config for CNOT H-teleportation measurements.

        Data always measured in X; ancilla basis depends on input state.
        """
        ancilla_basis = "Z" if input_state in ("0", "1") else "X"
        return cls(
            block_bases={"data_block": "X", "ancilla_block": ancilla_basis},
        )


# ═════════════════════════════════════════════════════════════════════════════
# Crossing Detector Configuration
# ═════════════════════════════════════════════════════════════════════════════

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
    a transversal gate.  The formula specifies which measurements to XOR.

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

    Attributes
    ----------
    formulas : List[CrossingDetectorFormula]
        List of crossing detector formulas to emit.
    """
    formulas: List[CrossingDetectorFormula] = field(default_factory=list)

    # ── Generic factory methods ──────────────────────────────────────────

    @classmethod
    def identity(cls) -> "CrossingDetectorConfig":
        """No crossing (identity gate) — simple 2-term temporal detectors."""
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
            CrossingDetectorFormula("Z_D", [
                CrossingDetectorTerm("data_block", "Z", "pre"),
                CrossingDetectorTerm("data_block", "Z", "post"),
            ]),
            CrossingDetectorFormula("Z_A", [
                CrossingDetectorTerm("ancilla_block", "Z", "pre"),
                CrossingDetectorTerm("ancilla_block", "Z", "post"),
            ]),
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
    def cnot_teleportation(
        cls, input_state: str = "0"
    ) -> "CrossingDetectorConfig":
        """
        Crossing detectors for CNOT H-teleportation.

        CNOT (data=control, ancilla=target) Heisenberg transformation:
        - Z_D → Z_D              (control Z unchanged)
        - Z_A → Z_D ⊗ Z_A        (target Z picks up control Z)
        - X_D → X_D ⊗ X_A        (control X picks up target X)
        - X_A → X_A              (target X unchanged)
        """
        formulas = []

        formulas.append(CrossingDetectorFormula("Z_D", [
            CrossingDetectorTerm("data_block", "Z", "pre"),
            CrossingDetectorTerm("data_block", "Z", "post"),
        ]))

        if input_state in ("0", "1"):
            formulas.append(CrossingDetectorFormula("Z_A", [
                CrossingDetectorTerm("ancilla_block", "Z", "pre"),
                CrossingDetectorTerm("ancilla_block", "Z", "post"),
            ]))
        else:
            formulas.append(CrossingDetectorFormula("Z_A", [
                CrossingDetectorTerm("ancilla_block", "Z", "pre"),
                CrossingDetectorTerm("data_block", "Z", "post"),
                CrossingDetectorTerm("ancilla_block", "Z", "post"),
            ]))

        formulas.append(CrossingDetectorFormula("X_D", [
            CrossingDetectorTerm("data_block", "X", "pre"),
            CrossingDetectorTerm("data_block", "X", "post"),
            CrossingDetectorTerm("ancilla_block", "X", "post"),
        ]))

        formulas.append(CrossingDetectorFormula("X_A", [
            CrossingDetectorTerm("ancilla_block", "X", "pre"),
            CrossingDetectorTerm("ancilla_block", "X", "post"),
        ]))

        return cls(formulas=formulas)


# ═════════════════════════════════════════════════════════════════════════════
# Boundary Detector Configuration
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BoundaryDetectorConfig:
    """
    Configuration for boundary (space-like) detectors.

    Boundary detectors compare final data measurements to last syndrome round.

    Attributes
    ----------
    block_configs : Dict[str, Dict[str, bool]]
        Per-block configuration mapping block_name to
        ``{"X": emit_x_boundary, "Z": emit_z_boundary}``.
    """
    block_configs: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    # ── Generic factory methods ──────────────────────────────────────────

    @classmethod
    def single_block(
        cls, measurement_basis: str = "Z"
    ) -> "BoundaryDetectorConfig":
        """Config for single-block gadget."""
        return cls(block_configs={
            "data_block": {
                "X": measurement_basis == "X",
                "Z": measurement_basis == "Z",
            }
        })

    @classmethod
    def cz_teleportation(
        cls, input_state: str = "0"
    ) -> "BoundaryDetectorConfig":
        """
        Boundary detectors for CZ H-teleportation.

        Data measured MX: no data boundary (CZ transforms X_D → X_D ⊗ Z_A).
        Ancilla: |0⟩ → MX → X_A boundary; |+⟩ → MZ → Z_A boundary.
        """
        ancilla_basis = "X" if input_state in ("0", "1") else "Z"
        return cls(block_configs={
            "data_block": {"X": False, "Z": False},
            "ancilla_block": {
                "X": ancilla_basis == "X",
                "Z": ancilla_basis == "Z",
            },
        })

    @classmethod
    def cnot_teleportation(
        cls, input_state: str = "0"
    ) -> "BoundaryDetectorConfig":
        """
        Boundary detectors for CNOT H-teleportation.

        Data measured MX → X_D boundary YES, Z_D boundary NO.
        Ancilla: |0⟩ → MZ → Z_A boundary; |+⟩ → MX → X_A boundary.
        """
        ancilla_basis = "Z" if input_state in ("0", "1") else "X"
        return cls(block_configs={
            "data_block": {"X": True, "Z": False},
            "ancilla_block": {
                "X": ancilla_basis == "X",
                "Z": ancilla_basis == "Z",
            },
        })
