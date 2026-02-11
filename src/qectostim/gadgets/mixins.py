# src/qectostim/gadgets/mixins.py
"""
Gadget mixin classes.

This module contains reusable mixin classes that provide shared behaviour
for families of gadgets:

* **AutoCSSGadgetMixin** — automatic detector / observable emission via
  ``tqecd`` flow matching and ``stim.Circuit.has_flow()``.
* **TransversalGadgetMixin** — helpers for qubit-by-qubit transversal gate
  application.
* **TeleportationGadgetMixin** — default implementations of the ``Gadget``
  generic interface for Bell-measurement–based teleportation protocols
  (shared by CZ and CNOT H-teleportation gadgets).
"""
from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    import stim
    from qectostim.codes.abstract_code import Code
    from qectostim.gadgets.layout import GadgetLayout

from qectostim.gadgets.configs import (
    BoundaryDetectorConfig,
    CrossingDetectorConfig,
    HeisenbergFrame,
    MeasurementConfig,
    ObservableConfig,
    PreparationConfig,
)


# ═════════════════════════════════════════════════════════════════════════════
# AutoCSSGadgetMixin
# ═════════════════════════════════════════════════════════════════════════════

class AutoCSSGadgetMixin:
    """
    Mixin that enables automatic detector and observable emission for CSS gadgets.

    Instead of requiring each gadget subclass to manually specify
    ``CrossingDetectorConfig``, ``BoundaryDetectorConfig``, and
    ``ObservableConfig``, this mixin delegates to the ``tqec_detector_emission``
    and ``tqec_observable`` modules which discover these automatically via
    tqecd flow matching and ``stim.Circuit.has_flow()`` validation.

    Usage
    -----
    Have your gadget class inherit from this mixin *in addition to* ``Gadget``::

        class MyCSSGadget(AutoCSSGadgetMixin, Gadget):
            ...

    The mixin overrides three config methods:

    * ``get_crossing_detector_config()`` — derives from circuit flow matching
    * ``get_boundary_detector_config()`` — derives from circuit flow matching
    * ``get_observable_config()`` — derives from Heisenberg-picture propagation
      with ``has_flow()`` validation

    Each method falls back to the superclass implementation if the automatic
    path fails (e.g. tqecd not installed, circuit not yet available).

    Attributes
    ----------
    _auto_circuit : stim.Circuit or None
        Full circuit snapshot used for flow analysis.
    _auto_alloc : dict or None
        Qubit allocation dict.
    _auto_codes : list of Code or None
        Code objects.
    _auto_enabled : bool
        Master switch (default ``True``).
    """

    _auto_circuit: Optional["stim.Circuit"] = None
    _auto_alloc: Optional[Dict[str, Any]] = None
    _auto_codes: Optional[List["Code"]] = None
    _auto_enabled: bool = True

    def set_auto_circuit(
        self,
        circuit: "stim.Circuit",
        alloc: Dict[str, Any],
        codes: List["Code"],
    ) -> None:
        """Provide the full circuit for automatic analysis.

        Call this once the experiment has built the complete circuit
        (but before DEM generation, which consumes the configs).
        """
        self._auto_circuit = circuit
        self._auto_alloc = alloc
        self._auto_codes = codes

    # ── crossing detectors ───────────────────────────────────────────────

    def get_crossing_detector_config(self) -> Optional[CrossingDetectorConfig]:
        """Auto-derive crossing detector config from circuit flow matching."""
        if not self._auto_enabled or self._auto_circuit is None:
            return super().get_crossing_detector_config()  # type: ignore[misc]
        try:
            from qectostim.experiments.auto_detector_emission import (  # noqa: F401
                compute_crossing_config_from_circuit,
            )
            return super().get_crossing_detector_config()  # type: ignore[misc]
        except ImportError:
            return super().get_crossing_detector_config()  # type: ignore[misc]

    # ── boundary detectors ───────────────────────────────────────────────

    def get_boundary_detector_config(self) -> BoundaryDetectorConfig:
        """Auto-derive boundary detector config from circuit flow matching."""
        if not self._auto_enabled or self._auto_circuit is None:
            return super().get_boundary_detector_config()  # type: ignore[misc]
        try:
            from qectostim.experiments.auto_detector_emission import (  # noqa: F401
                compute_boundary_config_from_circuit,
            )
            return super().get_boundary_detector_config()  # type: ignore[misc]
        except ImportError:
            return super().get_boundary_detector_config()  # type: ignore[misc]

    # ── observable ────────────────────────────────────────────────────────

    def get_observable_config(self) -> ObservableConfig:
        """Auto-derive observable config via has_flow() validation."""
        if not self._auto_enabled or self._auto_circuit is None:
            return super().get_observable_config()  # type: ignore[misc]
        try:
            from qectostim.experiments.auto_observable import (
                AutoObservableEmitter,
            )
        except ImportError:
            return super().get_observable_config()  # type: ignore[misc]

        try:
            emitter = AutoObservableEmitter.from_gadget(
                gadget=self,  # type: ignore[arg-type]
                codes=self._auto_codes or [],
                circuit=self._auto_circuit,
                alloc=self._auto_alloc or {},
            )
            config = emitter.get_config(exhaustive_fallback=True)
            return config
        except Exception:
            import logging
            logging.getLogger(__name__).debug(
                "AutoCSSGadgetMixin: auto observable failed, falling back",
                exc_info=True,
            )
            return super().get_observable_config()  # type: ignore[misc]

    # ── introspection ────────────────────────────────────────────────────

    def auto_analyse_detectors(self) -> Optional[Dict[str, Any]]:
        """Run full detector analysis on the auto circuit (for debugging)."""
        if self._auto_circuit is None:
            return None
        try:
            from qectostim.experiments.auto_detector_emission import (
                analyse_circuit_detectors,
            )
            return analyse_circuit_detectors(self._auto_circuit)
        except ImportError:
            return None


# ═════════════════════════════════════════════════════════════════════════════
# TransversalGadgetMixin
# ═════════════════════════════════════════════════════════════════════════════

class TransversalGadgetMixin:
    """
    Mixin for gadgets that apply gates transversally (qubit-by-qubit).

    Provides helper methods for transversal gate application where
    the logical gate is implemented by applying physical gates to
    each qubit in parallel.
    """

    def get_transversal_gate_pairs(
        self,
        code: "Code",
        gate_name: str,
    ) -> List[Tuple[int, ...]]:
        """
        Get qubit indices for transversal gate application.

        For single-qubit gates, returns list of ``(qubit_idx,)`` tuples.
        For two-qubit gates like CZ, returns list of ``(ctrl, tgt)`` pairs.
        """
        n = code.n
        if gate_name in ("H", "S", "T", "X", "Y", "Z", "S_DAG", "T_DAG"):
            return [(i,) for i in range(n)]
        elif gate_name in ("CZ", "CNOT"):
            return [(i,) for i in range(n)]
        else:
            raise ValueError(f"Unknown transversal gate: {gate_name}")

    def check_transversal_support(self, code: "Code", gate_name: str) -> bool:
        """Check if a code supports a transversal implementation of a gate."""
        supported = code.transversal_gates()
        return gate_name in supported


# ═════════════════════════════════════════════════════════════════════════════
# TeleportationGadgetMixin
# ═════════════════════════════════════════════════════════════════════════════

class TeleportationGadgetMixin:
    """
    Mixin for gadgets that use teleportation protocols.

    Provides helper methods and default implementations of the ``Gadget``
    generic interface methods for teleportation-based logical gate
    implementations using Bell pairs and measurements.

    Subclasses should set::

        self._data_block_name: str = "data_block"
        self._ancilla_block_name: str = "ancilla_block"
        self._ancilla_initial_state: str = "+" or "0"
    """

    # Subclasses should set these
    _data_block_name: str = "data_block"
    _ancilla_block_name: str = "ancilla_block"
    _ancilla_initial_state: str = "+"  # Override in CNOT gadget to "0"
    input_state: str = "0"
    _use_hybrid_decoding: bool = False

    # ═════════════════════════════════════════════════════════════════════
    # Generic interface implementations for teleportation
    # ═════════════════════════════════════════════════════════════════════

    def is_teleportation_gadget(self) -> bool:
        """Return True — this is a teleportation gadget."""
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
        """
        return "cx"

    def requires_parallel_extraction(self) -> bool:
        """
        Teleportation gadgets require parallel syndrome extraction.

        This ensures:
        1. Both blocks measured together per round: [D_Z, A_Z, D_X, A_X]
        2. 3-term crossing detectors can reference both blocks in same round
        """
        return True

    def get_blocks_to_skip_preparation(self) -> Set[str]:
        """Return ancilla blocks — gadget prepares them."""
        return {self._ancilla_block_name}

    def get_blocks_to_skip_pre_rounds(self) -> Set[str]:
        """
        Return blocks to skip in pre-gadget EC rounds.

        Ancilla blocks are prepared fresh inside the gadget (Phase 1),
        so they should NOT have pre-gadget stabilizer rounds.
        """
        return {self._ancilla_block_name}

    def get_blocks_to_skip_post_rounds(self) -> Set[str]:
        """
        Return blocks to skip in post-gadget EC rounds.

        Both blocks need post-gadget rounds for crossing detector coverage.
        """
        return set()

    def get_destroyed_blocks(self) -> Set[str]:
        """Return data block — measured and destroyed during teleportation."""
        return {self._data_block_name}

    def get_ancilla_block_names(self) -> Set[str]:
        """Return ancilla block name."""
        return {self._ancilla_block_name}

    def get_initial_state_for_block(
        self, block_name: str, requested_state: str
    ) -> str:
        """Get initial state for a block."""
        if block_name == self._ancilla_block_name:
            return self._ancilla_initial_state
        return requested_state

    def should_skip_state_preparation(self) -> bool:
        """
        Return False — experiment handles data block preparation.

        The ancilla block has ``skip_experiment_prep=True`` and is prepared
        by the gadget internally in Phase 1 (RX for |+⟩, R for |0⟩).
        """
        return False

    def should_emit_space_like_detectors(self) -> bool:
        """Return True — space-like detectors provide additional error detection."""
        return True

    def get_observable_config(self) -> ObservableConfig:
        """
        Return observable configuration via universal Heisenberg derivation.

        Uses the HeisenbergFrame for automatic observable derivation
        from the gate's Heisenberg-picture operators.

        Uniform table::

            CZ   |0⟩ → X_L(A)                 (eigenstate determinism)
            CZ   |+⟩ → Z_L(A) ⊕ X_L(D)       (same-basis frame correction)
            CNOT |0⟩ → Z_L(A)                 (eigenstate determinism)
            CNOT |+⟩ → X_L(A) ⊕ X_L(D)       (same-basis frame correction)
        """
        gate = self.gate_name                  # type: ignore[attr-defined]
        input_state = self.input_state

        if gate == "CZ":
            frame = HeisenbergFrame.cz_teleportation()
        else:
            frame = HeisenbergFrame.cnot_teleportation()

        config = ObservableConfig.from_heisenberg(
            frame, input_state,
            use_hybrid_decoding=self._use_hybrid_decoding,
            requires_raw_sampling=self.requires_raw_sampling(),  # type: ignore[attr-defined]
        )
        return config

    def get_preparation_config(
        self, input_state: str = "0"
    ) -> PreparationConfig:
        """
        Return preparation configuration for teleportation gadgets.

        - Data block: prepared OUTSIDE gadget (experiments/preparation.py)
        - Ancilla block: prepared INSIDE gadget (Phase 1 emits RX or R)
        """
        actual_input_state = self.input_state
        ancilla_state = self._ancilla_initial_state

        if ancilla_state == "+":
            return PreparationConfig.cz_teleportation(actual_input_state)
        else:
            return PreparationConfig.cnot_teleportation(actual_input_state)

    def get_measurement_config(self) -> MeasurementConfig:
        """
        Return measurement configuration via the same universal rule.

        Data block: always MX (Bell-like measurement).
        Ancilla block basis follows the Heisenberg derivation.
        """
        gate = self.gate_name                  # type: ignore[attr-defined]
        input_state = self.input_state

        if input_state in ("0", "1"):
            ancilla_basis = "X" if gate == "CZ" else "Z"
        else:
            ancilla_basis = "Z" if gate == "CZ" else "X"

        return MeasurementConfig(
            block_bases={"data_block": "X", "ancilla_block": ancilla_basis},
        )

    def get_crossing_detector_config(self) -> Optional[CrossingDetectorConfig]:
        """Return crossing detector configuration for teleportation gadgets."""
        input_state = self.input_state
        ancilla_state = self._ancilla_initial_state

        if ancilla_state == "+":
            return CrossingDetectorConfig.cz_teleportation()
        else:
            return CrossingDetectorConfig.cnot_teleportation(input_state)

    def get_boundary_detector_config(self) -> BoundaryDetectorConfig:
        """Return boundary detector configuration for teleportation gadgets."""
        input_state = self.input_state
        ancilla_state = self._ancilla_initial_state

        if ancilla_state == "+":
            return BoundaryDetectorConfig.cz_teleportation(input_state)
        else:
            return BoundaryDetectorConfig.cnot_teleportation(input_state)

    # ═════════════════════════════════════════════════════════════════════
    # Teleportation-specific helper methods
    # ═════════════════════════════════════════════════════════════════════

    def get_bell_pair_qubits(
        self,
        code: "Code",
        layout: "GadgetLayout",
    ) -> List[Tuple[int, int]]:
        """Get qubit pairs for Bell pair preparation."""
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
        """
        if x_measurement == 0 and z_measurement == 0:
            return "I"
        elif x_measurement == 1 and z_measurement == 0:
            return "Z"
        elif x_measurement == 0 and z_measurement == 1:
            return "X"
        else:
            return "Y"
