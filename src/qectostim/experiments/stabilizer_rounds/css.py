# src/qectostim/experiments/stabilizer_rounds/css.py
"""
CSS stabilizer round builder for CSS codes.

This module provides CSSStabilizerRoundBuilder which handles separate X and Z
stabilizer measurements with proper scheduling and first-round detector logic.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import stim

from qectostim.codes.abstract_code import ScheduleMode
from qectostim.utils.scheduling_core import graph_coloring_cnots

from .context import DetectorContext
from .base import BaseStabilizerRoundBuilder, StabilizerBasis

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode


class CSSStabilizerRoundBuilder(BaseStabilizerRoundBuilder):
    """
    Stabilizer round builder for CSS codes.
    
    Handles separate X and Z stabilizer measurements with proper scheduling
    and first-round detector logic. This is the main builder for CSS memory
    experiments and FT gadget experiments.
    
    Key features:
    - First-round X detectors only for X-basis memory (|+⟩ is X eigenstate)
    - First-round Z detectors only for Z-basis memory (|0⟩ is Z eigenstate)
    - Geometric scheduling when metadata provides coordinates
    - Graph-coloring fallback for conflict-free CNOT layers
    - SHIFT_COORDS for time advancement
    - Interleaved X/Z scheduling for surface codes
    
    X Stabilizer Measurement
    ------------------------
    Uses CX-based measurement with native RX/MRX instructions:
    - RX resets ancillas to |+⟩ directly (no H gate needed)
    - CX(ancilla → data) for parity extraction  
    - MRX measures in X basis directly (no H gate needed)
    
    This approach is simpler and avoids the error propagation issues
    that CZ-based measurement has with certain initial states.
    """
    
    def __init__(
        self,
        code: "CSSCode",
        ctx: DetectorContext,
        block_name: str = "main",
        data_offset: int = 0,
        ancilla_offset: Optional[int] = None,
        measurement_basis: str = "Z",
        enable_metachecks: bool = False,
        single_shot_metachecks: bool = True,
        coord_offset: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__(code, ctx, block_name, data_offset, ancilla_offset, measurement_basis, coord_offset=coord_offset)
        
        # Get code's FT gadget config for code-specific behavior
        self._ft_config = code.get_ft_gadget_config()
        
        # Single-shot metacheck mode: emit spatial constraint detectors from round 1
        # This allows metachecks to detect measurement errors within a single round,
        # rather than requiring time-like comparison across rounds.
        self._single_shot_metachecks = single_shot_metachecks
        
        # Cache stabilizer info - CSS codes define hx/hz as @property returning numpy arrays
        _hx_raw = getattr(code, 'hx', None)
        _hz_raw = getattr(code, 'hz', None)
        
        # Only use if it's actually a numpy array (has .shape attribute)
        self._hx = _hx_raw if _hx_raw is not None and hasattr(_hx_raw, 'shape') else None
        self._hz = _hz_raw if _hz_raw is not None and hasattr(_hz_raw, 'shape') else None
        self._n_x = self._hx.shape[0] if self._hx is not None and self._hx.size > 0 else 0
        self._n_z = self._hz.shape[0] if self._hz is not None and self._hz.size > 0 else 0
        
        # Detect self-dual codes (color codes where hx = hz)
        # For self-dual codes, |0⟩^⊗n is only deterministic for Z stabilizers,
        # and |+⟩^⊗n is only deterministic for X stabilizers.
        # We cannot have both first-round X and Z detectors in the same basis.
        self._is_self_dual = (
            self._hx is not None and self._hz is not None and
            self._hx.shape == self._hz.shape and
            np.array_equal(self._hx, self._hz)
        )
        
        # Get stabilizer coordinates - try code's hooks first, then metadata
        self._x_stab_coords = self._get_stabilizer_coords('x')
        self._z_stab_coords = self._get_stabilizer_coords('z')
        
        # Get schedules - try code's hooks first, then metadata
        self._x_schedule = self._get_schedule('x')
        self._z_schedule = self._get_schedule('z')
        
        # Track last measurements for each stabilizer (for time-like detectors)
        self._last_x_meas: List[Optional[int]] = [None] * self._n_x
        self._last_z_meas: List[Optional[int]] = [None] * self._n_z
        
        # Flag to skip first-round detectors (used after teleportation)
        self._skip_first_round: bool = False
        
        # Temporary flags for explicit anchor detector emission
        # These are set per-round by emit_round() and consumed by _emit_x_round/_emit_z_round
        self._emit_z_anchors: bool = False
        self._emit_x_anchors: bool = False
        
        # Flag to track if stabilizer types have been swapped (after Hadamard)
        # When True, physical X ancillas measure logical Z, and vice versa
        self._stabilizer_swapped: bool = False
        
        # Metacheck support for 4D/5D codes (single-shot error correction)
        # Use code's config setting, but can be overridden by parameter
        self._enable_metachecks = enable_metachecks or self._ft_config.enable_metachecks
        _meta_x_raw = self._meta.get('meta_x', None)
        _meta_z_raw = self._meta.get('meta_z', None)
        # meta_x and meta_z should be numpy arrays or None
        self._meta_x = _meta_x_raw if _meta_x_raw is not None and hasattr(_meta_x_raw, 'shape') else None
        self._meta_z = _meta_z_raw if _meta_z_raw is not None and hasattr(_meta_z_raw, 'shape') else None
        self._n_meta_x = self._meta_x.shape[0] if self._meta_x is not None and self._meta_x.size > 0 else 0
        self._n_meta_z = self._meta_z.shape[0] if self._meta_z is not None and self._meta_z.size > 0 else 0
        
        # Track metacheck measurements (for detecting syndrome measurement errors)
        self._last_meta_x_meas: List[Optional[int]] = [None] * self._n_meta_x
        self._last_meta_z_meas: List[Optional[int]] = [None] * self._n_meta_z

        # Track whether the previous round ended with MR/MRX so that the
        # next round can skip the redundant R/RX on ancillas.  MR already
        # resets to |0⟩ and MRX already resets to |+⟩, so a subsequent R/RX
        # on the same qubit is a no-op that wastes hardware cycles.
        self._ancilla_already_reset: bool = False
    
    def _get_stabilizer_coords(self, basis: str) -> List[Tuple[float, ...]]:
        """Get stabilizer coordinates using code hooks or metadata.
        
        Parameters
        ----------
        basis : str
            'x' or 'z' for X-type or Z-type stabilizers.
            
        Returns
        -------
        List[Tuple[float, ...]]
            Coordinates for each stabilizer of the given type.
        """
        # Use CSSCode hook if available (code.as_css() returns non-None for CSS codes)
        css = self.code.as_css()
        if css is not None:
            if basis == 'x':
                coords = css.get_x_stabilizer_coords()
            else:
                coords = css.get_z_stabilizer_coords()
            if coords is not None:
                return coords
        
        # Fall back to metadata
        key = f'{basis}_stab_coords'
        return self._meta.get(key, [])
    
    def _get_schedule(self, basis: str) -> Optional[List[Tuple[float, float]]]:
        """Get CNOT schedule using code hooks or metadata.
        
        Parameters
        ----------
        basis : str
            'x' or 'z' for X-type or Z-type stabilizers.
            
        Returns
        -------
        Optional[List[Tuple[float, float]]]
            Schedule offsets or None for default scheduling.
        """
        # Use CSSCode hook if available
        css = self.code.as_css()
        if css is not None:
            schedule = css.get_stabilizer_schedule(basis)
            if schedule is not None:
                return schedule
        
        # Fall back to metadata
        key = f'{basis}_schedule'
        return self._meta.get(key)
    
    def reset_stabilizer_history(self, swap_xz: bool = False, skip_first_round: bool = False, clear_history: bool = False) -> None:
        """
        Reset the builder's internal stabilizer measurement history.
        
        This is necessary after gates like Hadamard that change the stabilizer
        basis. After a Hadamard, the physical stabilizer measurements have 
        different logical interpretations, so we cannot compare post-gate 
        measurements with pre-gate measurements.
        
        Parameters
        ----------
        swap_xz : bool
            If True, also swap the effective measurement basis and track that
            stabilizer types are swapped. This is needed after Hadamard because:
            - Before H: |0⟩ is Z eigenstate → first-round Z detectors OK
            - After H: |+⟩ is X eigenstate → first-round X detectors OK
            - Physical X ancillas now measure logical Z (and vice versa)
        skip_first_round : bool
            If True, skip emitting first-round detectors entirely. This is needed
            for teleportation where the block's stabilizers are entangled with
            Bell measurements and not independently deterministic.
        clear_history : bool
            If True, clear all measurement history (teleportation-style
            transforms where the logical state is teleported).
            
        Note
        ----
        We always clear the history (not swap) because hx and hz are different
        physical operators. But we DO swap the measurement_basis because the
        logical state has changed basis.
        
        For teleportation, use skip_first_round=True because the block's stabilizers
        are entangled with Bell measurements and not independently deterministic.
        """
        # Always clear measurement history - establish new baseline
        self._last_x_meas = [None] * self._n_x
        self._last_z_meas = [None] * self._n_z
        self._last_meta_x_meas = [None] * self._n_meta_x
        self._last_meta_z_meas = [None] * self._n_meta_z
        
        # Swap effective measurement basis if requested (for Hadamard)
        if swap_xz:
            self.measurement_basis = "X" if self.measurement_basis == "Z" else "Z"
            # Toggle the stabilizer swapped flag
            self._stabilizer_swapped = not self._stabilizer_swapped
        
        # Set flag to skip first-round detectors (for teleportation)
        self._skip_first_round = skip_first_round

        # After a stabiliser transform that swaps X↔Z (Hadamard), the
        # ancilla preparation basis changes, so the next round must emit
        # its own R/RX.  When only clearing measurement history
        # (clear_history without swap_xz), the physical ancilla state is
        # unchanged — MR/MRX already left them correctly reset.
        if swap_xz:
            self._ancilla_already_reset = False

    @property
    def x_ancillas(self) -> List[int]:
        """Global indices of X stabilizer ancillas."""
        return list(range(self.ancilla_offset, self.ancilla_offset + self._n_x))
    
    @property
    def z_ancillas(self) -> List[int]:
        """Global indices of Z stabilizer ancillas."""
        return list(range(
            self.ancilla_offset + self._n_x,
            self.ancilla_offset + self._n_x + self._n_z
        ))
    
    @property
    def meta_x_ancillas(self) -> List[int]:
        """Global indices of meta-X check ancillas (check Z syndrome)."""
        if self._n_meta_x == 0:
            return []
        base = self.ancilla_offset + self._n_x + self._n_z
        return list(range(base, base + self._n_meta_x))
    
    @property
    def meta_z_ancillas(self) -> List[int]:
        """Global indices of meta-Z check ancillas (check X syndrome)."""
        if self._n_meta_z == 0:
            return []
        base = self.ancilla_offset + self._n_x + self._n_z + self._n_meta_x
        return list(range(base, base + self._n_meta_z))
    
    @property
    def has_metachecks(self) -> bool:
        """Whether this builder has metacheck support enabled."""
        return self._enable_metachecks and (self._n_meta_x > 0 or self._n_meta_z > 0)
    
    @property
    def total_qubits(self) -> int:
        """Total qubits used by this block (data + ancillas + metacheck ancillas)."""
        return self.code.n + self._n_x + self._n_z + self._n_meta_x + self._n_meta_z

    @property
    def qubit_roles(self) -> Dict[int, str]:
        """Authoritative mapping of qubit index → role string.

        Returns a dict keyed by global qubit index with values:

        - ``'D'`` — data qubit
        - ``'X'`` — X-stabilizer ancilla
        - ``'Z'`` — Z-stabilizer ancilla
        - ``'MX'`` — metacheck ancilla for X syndrome
        - ``'MZ'`` — metacheck ancilla for Z syndrome

        This is the *source of truth* for qubit roles; downstream
        consumers (hardware compilers, visualisation) should use this
        rather than reverse-engineering roles from MR vs M instructions.
        """
        roles: Dict[int, str] = {}
        for q in self.data_qubits:
            roles[q] = "D"
        for q in self.x_ancillas:
            roles[q] = "X"
        for q in self.z_ancillas:
            roles[q] = "Z"
        for q in self.meta_x_ancillas:
            roles[q] = "MX"
        for q in self.meta_z_ancillas:
            roles[q] = "MZ"
        return roles

    # ------------------------------------------------------------------
    # CNOT schedule export (for QECMetadata → routing)
    # ------------------------------------------------------------------

    def get_cnot_layers(self, stab_type: str) -> Optional[List[List[Tuple[int, int]]]]:
        """Return parallel CNOT layers for the given stabilizer type.

        Each layer is a list of ``(ctrl, tgt)`` pairs that can execute
        simultaneously (no qubit conflicts within a layer).  The layer
        ordering matches the actual circuit emission order.

        Parameters
        ----------
        stab_type : str
            ``"x"`` or ``"z"``.

        Returns
        -------
        list or None
            ``None`` when the code has no stabilizers of this type.
        """
        if stab_type == "x":
            return self._get_cnot_layers_for_type(
                self._hx, self._x_schedule, self._x_stab_coords,
                self.x_ancillas, is_x_type=True,
            )
        else:
            return self._get_cnot_layers_for_type(
                self._hz, self._z_schedule, self._z_stab_coords,
                self.z_ancillas, is_x_type=False,
            )

    def get_all_cnot_layers(self) -> Dict[str, Optional[List[List[Tuple[int, int]]]]]:
        """Return CNOT layers for both X and Z stabilizers."""
        return {"x": self.get_cnot_layers("x"), "z": self.get_cnot_layers("z")}

    def _get_cnot_layers_for_type(
        self,
        stab_matrix,
        schedule,
        stab_coords,
        ancillas: List[int],
        is_x_type: bool,
    ) -> Optional[List[List[Tuple[int, int]]]]:
        """Build parallel CNOT layers using geometric or graph-coloring."""
        if stab_matrix is None or not hasattr(stab_matrix, "shape") or stab_matrix.size == 0:
            return None

        if schedule is not None and stab_coords:
            # Geometric scheduling — mirror _emit_geometric_cnots logic
            layers: List[List[Tuple[int, int]]] = []
            for dx, dy in schedule:
                layer: List[Tuple[int, int]] = []
                for s_idx, (sx, sy) in enumerate(stab_coords):
                    if s_idx >= len(ancillas):
                        continue
                    anc = ancillas[s_idx]
                    nbr = (float(sx) + dx, float(sy) + dy)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        if is_x_type:
                            layer.append((anc, dq))
                        else:
                            layer.append((dq, anc))
                if layer:
                    layers.append(layer)
            return layers if layers else None

        # Graph-coloring fallback — mirror _emit_graph_coloring_cnots logic
        from qectostim.utils.scheduling_core import graph_coloring_cnots

        n_stabs, n_data = stab_matrix.shape
        all_gates: List[Tuple[int, int]] = []
        data_qubits = self.data_qubits
        for s_idx in range(min(n_stabs, len(ancillas))):
            anc = ancillas[s_idx]
            for d_idx in range(min(n_data, len(data_qubits))):
                if stab_matrix[s_idx, d_idx]:
                    dq = data_qubits[d_idx]
                    if is_x_type:
                        all_gates.append((anc, dq))
                    else:
                        all_gates.append((dq, anc))
        if not all_gates:
            return None
        return graph_coloring_cnots(all_gates)

    def emit_qubit_coords(self, circuit: stim.Circuit) -> None:
        """Emit QUBIT_COORDS for all qubits in this block.

        Covers data qubits, X/Z stabiliser ancillas, and metacheck ancillas.
        Every qubit that may appear in the circuit gets coordinates so
        that downstream consumers (compilers, visualisation) can always
        look up a position.
        """
        used_positions: set = set()

        # Data qubits
        for local_idx, coord in enumerate(self._data_coords):
            if len(coord) >= 2:
                global_idx = self.data_offset + local_idx
                pos = (float(coord[0]), float(coord[1]))
                circuit.append("QUBIT_COORDS", [global_idx], list(pos))
                used_positions.add(pos)
        
        # X ancillas
        for local_idx, coord in enumerate(self._x_stab_coords):
            if len(coord) >= 2 and local_idx < self._n_x:
                global_idx = self.ancilla_offset + local_idx
                pos = (float(coord[0]), float(coord[1]))
                circuit.append("QUBIT_COORDS", [global_idx], list(pos))
                used_positions.add(pos)
        
        # Z ancillas
        for local_idx, coord in enumerate(self._z_stab_coords):
            if len(coord) >= 2 and local_idx < self._n_z:
                global_idx = self.ancilla_offset + self._n_x + local_idx
                pos = (float(coord[0]), float(coord[1]))
                circuit.append("QUBIT_COORDS", [global_idx], list(pos))
                used_positions.add(pos)

        # Metacheck ancillas (if any)
        meta_base = self.ancilla_offset + self._n_x + self._n_z
        for i in range(self._n_meta_x + self._n_meta_z):
            global_idx = meta_base + i
            # Place at a unique position outside the code footprint
            pos = (-1.0, float(i))
            while pos in used_positions:
                pos = (pos[0] - 0.01, pos[1])
            circuit.append("QUBIT_COORDS", [global_idx], list(pos))
            used_positions.add(pos)
    
    def emit_reset_all(self, circuit: stim.Circuit, *, skip_data: bool = False) -> None:
        """Reset data, ancilla, and metacheck ancilla qubits.

        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        skip_data : bool
            If ``True``, only reset ancilla (and metacheck) qubits.
            Use when the caller will prepare data qubits with ``RX``
            so that the redundant ``R`` → ``RX`` is avoided.
        """
        if skip_data:
            all_qubits = self.x_ancillas + self.z_ancillas
        else:
            all_qubits = self.data_qubits + self.x_ancillas + self.z_ancillas
        if self._enable_metachecks:
            all_qubits = all_qubits + self.meta_x_ancillas + self.meta_z_ancillas
        if all_qubits:
            circuit.append("R", all_qubits)
    
    def emit_prepare_logical_state(
        self,
        circuit: stim.Circuit,
        state: str = "0",
        logical_idx: int = 0,
        use_projection: bool = False,
    ) -> Dict[str, List[int]]:
        """
        Prepare a logical eigenstate via data-qubit initialisation.

        For CSS codes the product state |0⟩^⊗n (or |+⟩^⊗n) already satisfies
        one set of stabilisers deterministically:

        - |0⟩^⊗n  →  Z stabilisers  = +1  (deterministic)
        - |+⟩^⊗n  →  X stabilisers  = +1  (deterministic)

        Projection into the code-space happens automatically when the first
        syndrome round is measured by ``emit_round()``, which also establishes
        anchor detectors for the deterministic stabiliser type.  No extra
        projection ancilla or serial stabiliser measurements are needed.

        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        state : str
            ``"0"`` for |0⟩_L, ``"1"`` for |1⟩_L,
            ``"+"`` for |+⟩_L, ``"-"`` for |−⟩_L.
        logical_idx : int
            Which logical qubit.
        use_projection : bool
            Ignored (kept for API compatibility).  Serial projection is no
            longer performed; the first ``emit_round()`` call handles it.

        Returns
        -------
        Dict[str, List[int]]
            Always ``{'x_stab_meas': [], 'z_stab_meas': [], 'logical_meas': []}``.
        """
        code = self.code

        if state in ("0", "1"):
            # |0⟩^⊗n is the reset default; only apply X for |1⟩_L
            if state == "1" and hasattr(code, 'logical_x_support'):
                support = code.logical_x_support(logical_idx)
                for q in support:
                    circuit.append("X", [self.data_offset + q])
        elif state in ("+", "-"):
            # RX atomically prepares |+⟩ (no H gate needed).
            circuit.append("RX", self.data_qubits)
            # Z|+⟩ = |−⟩  →  apply Z *after* RX for |−⟩_L
            if state == "-" and hasattr(code, 'logical_z_support'):
                support = code.logical_z_support(logical_idx)
                for q in support:
                    circuit.append("Z", [self.data_offset + q])

        circuit.append("TICK")

        return {
            'x_stab_meas': [],
            'z_stab_meas': [],
            'logical_meas': [],
        }
    
    def _can_interleave(self) -> bool:
        """Check if interleaved X/Z scheduling is possible.
        
        Interleaved scheduling requires:
        1. Both X and Z stabilizers present
        2. Both have geometric schedules
        3. Schedules have the same length (so phases can be paired)
        """
        if self._n_x == 0 or self._n_z == 0:
            return False
        if not self._use_geometric_x() or not self._use_geometric_z():
            return False
        if self._x_schedule is None or self._z_schedule is None:
            return False
        # Schedules must have same number of phases for pairing
        return len(self._x_schedule) == len(self._z_schedule)
    
    def emit_z_layer(
        self,
        circuit: stim.Circuit,
        emit_detectors: bool = True,
        emit_z_anchors: bool = False,
    ) -> None:
        """
        Emit Z stabilizer measurement layer only (no time advance).
        
        This is used for parallel extraction in teleportation gadgets where
        multiple blocks need to emit their Z measurements together before
        any block emits X measurements.
        
        Unlike emit_round(), this does NOT:
        - Advance time (ctx.advance_time)
        - Emit SHIFT_COORDS
        - Increment round number
        
        Call finalize_parallel_round() after all layers are emitted.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        emit_detectors : bool
            Whether to emit temporal detectors.
        emit_z_anchors : bool
            If True, emit single-term anchor detectors for Z stabilizers
            (use for first round when Z stabilizers are deterministic).
        """
        self._emit_z_anchors = emit_z_anchors
        self._explicit_anchor_mode = True  # Always use explicit mode for layer methods
        
        self._emit_z_round(circuit, emit_detectors)
        
        self._emit_z_anchors = False
        self._explicit_anchor_mode = False
    
    def emit_x_layer(
        self,
        circuit: stim.Circuit,
        emit_detectors: bool = True,
        emit_x_anchors: bool = False,
    ) -> None:
        """
        Emit X stabilizer measurement layer only (no time advance).
        
        This is used for parallel extraction in teleportation gadgets where
        multiple blocks need to emit their X measurements together after
        all blocks emitted Z measurements.
        
        Unlike emit_round(), this does NOT:
        - Advance time (ctx.advance_time)
        - Emit SHIFT_COORDS
        - Increment round number
        
        Call finalize_parallel_round() after all layers are emitted.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        emit_detectors : bool
            Whether to emit temporal detectors.
        emit_x_anchors : bool
            If True, emit single-term anchor detectors for X stabilizers
            (use for first round when X stabilizers are deterministic).
        """
        self._emit_x_anchors = emit_x_anchors
        self._explicit_anchor_mode = True  # Always use explicit mode for layer methods
        
        self._emit_x_round(circuit, emit_detectors)
        
        self._emit_x_anchors = False
        self._explicit_anchor_mode = False
    
    def finalize_parallel_round(self, circuit: stim.Circuit) -> None:
        """
        Finalize a parallel round after emit_z_layer and emit_x_layer calls.
        
        This advances time, emits SHIFT_COORDS, and increments the round number.
        Call this once per logical round after all blocks have emitted their
        Z and X layers.
        """
        self.ctx.advance_time()
        self._emit_shift_coords(circuit)
        self._round_number += 1
        self._skip_first_round = False

    # ------------------------------------------------------------------
    # Cross-builder interleaved emission helpers
    # ------------------------------------------------------------------
    # These are used by MemoryRoundEmitter._emit_parallel_interleaved()
    # to emit all builders' X+Z CX gates in shared TICK layers.

    def n_interleave_phases(self) -> int:
        """Number of geometric CX phases available for interleaving.

        Returns 0 if interleaving is not possible for this builder.
        """
        if not self._can_interleave():
            return 0
        return len(self._x_schedule)

    def emit_ancilla_reset(self, circuit: stim.Circuit,
                           emit_z_anchors: bool = False,
                           emit_x_anchors: bool = False) -> None:
        """Emit ancilla resets for both X and Z types (no TICK).

        Handles ``_stabilizer_swapped`` state transparently.
        The caller must emit a TICK *after* all builders have reset.

        Parameters
        ----------
        emit_z_anchors, emit_x_anchors : bool
            Store anchor flags for later use by ``emit_ancilla_measure_and_detectors``.
        """
        self._emit_z_anchors = emit_z_anchors
        self._emit_x_anchors = emit_x_anchors
        self._explicit_anchor_mode = True

        # Skip redundant resets if the previous round's MR/MRX already
        # reset the ancillas (same optimisation as _emit_interleaved_round).
        if self._ancilla_already_reset:
            self._ancilla_already_reset = False   # consumed
            return

        x_anc = self.x_ancillas
        z_anc = self.z_ancillas

        if self._stabilizer_swapped:
            # After Hadamard: X ancillas measure Z-type (R), Z measure X-type (RX)
            if x_anc:
                circuit.append("R", x_anc)
            if z_anc:
                circuit.append("RX", z_anc)
        else:
            # Normal: X ancillas → |+⟩ (RX), Z ancillas → |0⟩ (R)
            if x_anc:
                circuit.append("RX", x_anc)
            if z_anc:
                circuit.append("R", z_anc)

    def emit_cx_for_phase(self, circuit: stim.Circuit, phase_idx: int) -> None:
        """Emit X+Z CX gates for one geometric phase (no TICK, no reset, no measure).

        The caller manages TICK boundaries between phases and across builders.

        Parameters
        ----------
        phase_idx : int
            Index into the geometric schedule (0-based).
        """
        x_anc = self.x_ancillas
        z_anc = self.z_ancillas

        # X schedule CX gates for this phase
        if (self._x_schedule is not None
                and phase_idx < len(self._x_schedule)
                and x_anc):
            dx_x, dy_x = self._x_schedule[phase_idx]
            for s_idx, (sx, sy) in enumerate(self._x_stab_coords):
                if s_idx >= len(x_anc):
                    continue
                anc = x_anc[s_idx]
                nbr = (float(sx) + dx_x, float(sy) + dy_x)
                dq = self._lookup_data_qubit(nbr)
                if dq is not None:
                    if self._stabilizer_swapped:
                        # X ancillas now measure Z-type: data controls ancilla
                        circuit.append("CX", [dq, anc])
                    else:
                        # Normal X-type: ancilla controls data
                        circuit.append("CX", [anc, dq])

        # Z schedule CX gates for this phase
        if (self._z_schedule is not None
                and phase_idx < len(self._z_schedule)
                and z_anc):
            dx_z, dy_z = self._z_schedule[phase_idx]
            for s_idx, (sx, sy) in enumerate(self._z_stab_coords):
                if s_idx >= len(z_anc):
                    continue
                anc = z_anc[s_idx]
                nbr = (float(sx) + dx_z, float(sy) + dy_z)
                dq = self._lookup_data_qubit(nbr)
                if dq is not None:
                    if self._stabilizer_swapped:
                        # Z ancillas now measure X-type: ancilla controls data
                        circuit.append("CX", [anc, dq])
                    else:
                        # Normal Z-type: data controls ancilla
                        circuit.append("CX", [dq, anc])

    def emit_ancilla_measure_and_detectors(
        self, circuit: stim.Circuit, emit_detectors: bool = True,
    ) -> None:
        """Emit ancilla measurements and detectors (no TICK before/after).

        The caller must emit a TICK *before* calling this (after the last
        CX phase) and a TICK *after* all builders have measured.
        Handles ``_stabilizer_swapped`` transparently.
        """
        x_anc = self.x_ancillas
        z_anc = self.z_ancillas

        if self._stabilizer_swapped:
            # X ancillas measured Z-type → MR
            if x_anc:
                x_meas_start = self.ctx.add_measurement(self._n_x)
                circuit.append("MR", x_anc)
            else:
                x_meas_start = self.ctx.measurement_index
            # Z ancillas measured X-type → MRX
            if z_anc:
                z_meas_start = self.ctx.add_measurement(self._n_z)
                circuit.append("MRX", z_anc)
            else:
                z_meas_start = self.ctx.measurement_index
        else:
            # X ancillas → MRX, Z ancillas → MR
            if x_anc:
                x_meas_start = self.ctx.add_measurement(self._n_x)
                circuit.append("MRX", x_anc)
            else:
                x_meas_start = self.ctx.measurement_index
            if z_anc:
                z_meas_start = self.ctx.add_measurement(self._n_z)
                circuit.append("MR", z_anc)
            else:
                z_meas_start = self.ctx.measurement_index

        # Detectors
        x_eff = "Z" if self._stabilizer_swapped else "X"
        z_eff = "X" if self._stabilizer_swapped else "Z"
        self._emit_detectors_for_type(circuit, "x", x_meas_start, emit_detectors, x_eff)
        self._emit_detectors_for_type(circuit, "z", z_meas_start, emit_detectors, z_eff)

        # Clear anchor flags
        self._emit_z_anchors = False
        self._emit_x_anchors = False
        self._explicit_anchor_mode = False

        # MR/MRX already resets ancillas — flag so the next round
        # can skip an explicit R/RX (applies to both sequential and
        # parallel emission paths).
        self._ancilla_already_reset = True

    def emit_round(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_detectors: bool = True,
        emit_metachecks: bool = False,
        emit_z_anchors: bool = False,
        emit_x_anchors: bool = False,
        explicit_anchor_mode: Optional[bool] = None,
        first_basis: Optional[StabilizerBasis] = None,
    ) -> None:
        """
        Emit one complete stabilizer measurement round.
        
        Uses interleaved X/Z scheduling when both geometric schedules are
        available with matching phases. This reduces circuit depth and
        improves error correction performance by having X and Z CNOTs
        happen in parallel within each TICK layer.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        stab_type : StabilizerBasis
            Which stabilizers to measure.
        emit_detectors : bool
            Whether to emit time-like detectors.
        emit_metachecks : bool
            Whether to emit metacheck detectors for single-shot correction.
            Only applies if the code has metachecks (4D/5D surface codes).
        emit_z_anchors : bool
            If True, emit single-term anchor detectors for Z stabilizers.
            Use this for first round when Z stabilizers are deterministic
            (e.g., after |0⟩ preparation).
        emit_x_anchors : bool
            If True, emit single-term anchor detectors for X stabilizers.
            Use this for first round when X stabilizers are deterministic
            (e.g., after |+⟩ preparation).
        explicit_anchor_mode : Optional[bool]
            If True, ONLY use emit_z_anchors/emit_x_anchors flags for anchor
            emission - do NOT also check measurement_basis. This ensures the
            caller has full control over which anchors are emitted.
            If None (default), auto-detect: use explicit mode if either
            anchor flag was set in previous rounds for this builder.
        first_basis : Optional[StabilizerBasis]
            Which stabilizer type to measure first in this round.
            - StabilizerBasis.Z: Measure Z first, then X
            - StabilizerBasis.X: Measure X first, then Z
            - None (default): Use default order (X first, then Z)
            
            For deterministic anchor/boundary detectors:
            - Anchor round (first round): Measure deterministic basis FIRST
              (Z first for |0⟩ prep, X first for |+⟩ prep)
            - Boundary round (last round): Measure deterministic basis LAST
              (Z last for MZ, X last for MX)
        """
        # Track measurement indices for metachecks
        x_meas_start = self.ctx.measurement_index
        z_meas_start = x_meas_start + self._n_x  # Z measurements follow X
        
        # Store anchor flags for internal methods to access
        self._emit_z_anchors = emit_z_anchors
        self._emit_x_anchors = emit_x_anchors
        
        # Determine explicit anchor mode
        # If caller explicitly passed anchor flags, use ONLY those flags
        # (don't also check measurement_basis which could enable unwanted anchors)
        if explicit_anchor_mode is None:
            # Auto-detect: if any anchor flag is True, we're in explicit mode
            self._explicit_anchor_mode = emit_z_anchors or emit_x_anchors
        else:
            self._explicit_anchor_mode = explicit_anchor_mode
        
        # Use interleaved scheduling for BOTH when possible (surface codes, etc.)
        if stab_type == StabilizerBasis.BOTH and self._can_interleave():
            self._emit_interleaved_round(circuit, emit_detectors)
        else:
            # Sequential fallback for single-type or non-geometric codes
            # Respect first_basis ordering for deterministic anchors/boundaries
            if first_basis == StabilizerBasis.Z:
                # Z first, then X
                if stab_type in (StabilizerBasis.Z, StabilizerBasis.BOTH):
                    self._emit_z_round(circuit, emit_detectors)
                if stab_type in (StabilizerBasis.X, StabilizerBasis.BOTH):
                    self._emit_x_round(circuit, emit_detectors)
            else:
                # Default: X first, then Z
                if stab_type in (StabilizerBasis.X, StabilizerBasis.BOTH):
                    self._emit_x_round(circuit, emit_detectors)
                if stab_type in (StabilizerBasis.Z, StabilizerBasis.BOTH):
                    self._emit_z_round(circuit, emit_detectors)
        
        # Clear anchor flags after use
        self._emit_z_anchors = False
        self._emit_x_anchors = False
        
        # Emit metachecks if enabled and available
        if emit_metachecks and self.has_metachecks and stab_type == StabilizerBasis.BOTH:
            self._emit_metacheck_round(
                circuit, x_meas_start, z_meas_start, emit_detectors
            )
        
        # Advance time and emit SHIFT_COORDS for visualization
        self.ctx.advance_time()
        self._emit_shift_coords(circuit)
        self._round_number += 1
        
        # Clear skip_first_round flag after the first round completes
        # Subsequent rounds should emit normal time-like detectors
        self._skip_first_round = False
    
    def emit_round_with_transform(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_detectors: bool = True,
        swap_xz: bool = False,
    ) -> None:
        """
        Emit a stabilizer round with stabilizer transform applied.
        
        This is used after gates like Hadamard that swap X↔Z stabilizers.
        The transform is applied temporarily for this round only.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        stab_type : StabilizerBasis
            Which stabilizers to measure.
        emit_detectors : bool
            Whether to emit time-like detectors.
        swap_xz : bool
            If True, measure stabilizers as if H was applied (X stabs measure Z, Z stabs measure X).
            This should be True for post-gadget rounds after Hadamard gates.
        """
        if swap_xz:
            # Temporarily set swap state to swapped
            saved_swap_state = self._stabilizer_swapped
            self._stabilizer_swapped = True  # Set to swapped (not toggle!)
            
            # Emit the round with swapped stabilizers
            self.emit_round(circuit, stab_type, emit_detectors)
            
            # Restore original swap state
            self._stabilizer_swapped = saved_swap_state
        else:
            # No transform: emit round normally
            self.emit_round(circuit, stab_type, emit_detectors)
    
    def _emit_interleaved_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """Emit interleaved X/Z stabilizer measurements.
        
        This is the optimal scheduling for surface codes and similar CSS codes
        where X and Z stabilizers have matching geometric schedules. X and Z
        CNOTs happen in parallel within each phase, reducing circuit depth.
        
        Circuit structure (normal mode):
        1. RX on X ancillas, R on Z ancillas (prepare |+⟩ and |0⟩)
        2. For each phase: X and Z CNOTs together in same TICK
        3. MRX on X ancillas, MR on Z ancillas
        
        After Hadamard (_stabilizer_swapped=True):
        - X ancillas now measure Z-type stabilizers → R, data controls, MR
        - Z ancillas now measure X-type stabilizers → RX, ancilla controls, MRX
        """
        x_anc = self.x_ancillas
        z_anc = self.z_ancillas
        
        if self._stabilizer_swapped:
            # After Hadamard: swap the circuit patterns
            # Z ancillas measure X-type → RX - CX(anc→data) - MRX
            # X ancillas measure Z-type → R - CNOT(data→anc) - MR
            
            # Step 1: Prepare ancillas (skip if MR/MRX already reset them)
            if not self._ancilla_already_reset:
                circuit.append("R", x_anc)    # X ancillas: |0⟩ for Z-type measurement
                circuit.append("RX", z_anc)   # Z ancillas: |+⟩ for X-type measurement
            circuit.append("TICK")
            
            # Step 2: Interleaved CNOT phases - swapped control/target
            for phase_idx, ((dx_x, dy_x), (dx_z, dy_z)) in enumerate(
                zip(self._x_schedule, self._z_schedule)
            ):
                if phase_idx > 0:
                    circuit.append("TICK")
                
                # X ancillas: now measuring Z-type → data controls ancilla
                for s_idx, (sx, sy) in enumerate(self._x_stab_coords):
                    if s_idx >= len(x_anc):
                        continue
                    anc = x_anc[s_idx]
                    nbr = (float(sx) + dx_x, float(sy) + dy_x)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CX", [dq, anc])  # data controls ancilla
                
                # Z ancillas: now measuring X-type → ancilla controls data
                for s_idx, (sx, sy) in enumerate(self._z_stab_coords):
                    if s_idx >= len(z_anc):
                        continue
                    anc = z_anc[s_idx]
                    nbr = (float(sx) + dx_z, float(sy) + dy_z)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CX", [anc, dq])  # ancilla controls data
            
            circuit.append("TICK")
            
        else:
            # Normal mode: standard surface code interleaving
            # Step 1: Prepare ancillas - RX for X-type, R for Z-type
            # (skip if MR/MRX from the previous round already reset them)
            if not self._ancilla_already_reset:
                circuit.append("RX", x_anc)  # X ancillas: |+⟩ for X-type measurement
                circuit.append("R", z_anc)   # Z ancillas: |0⟩ for Z-type measurement
            circuit.append("TICK")
            
            # Step 2: Interleaved CNOT phases - X and Z in parallel
            for phase_idx, ((dx_x, dy_x), (dx_z, dy_z)) in enumerate(
                zip(self._x_schedule, self._z_schedule)
            ):
                if phase_idx > 0:
                    circuit.append("TICK")
                
                # X stabilizer measurement: CX(ancilla → data)
                for s_idx, (sx, sy) in enumerate(self._x_stab_coords):
                    if s_idx >= len(x_anc):
                        continue
                    anc = x_anc[s_idx]
                    nbr = (float(sx) + dx_x, float(sy) + dy_x)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CX", [anc, dq])  # ancilla controls data
                
                # Z CNOTs for this phase (SAME TICK - parallel with X)
                # For Z stabilizers: data qubits control ancilla (standard parity check)
                for s_idx, (sx, sy) in enumerate(self._z_stab_coords):
                    if s_idx >= len(z_anc):
                        continue
                    anc = z_anc[s_idx]
                    nbr = (float(sx) + dx_z, float(sy) + dy_z)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CX", [dq, anc])  # data controls ancilla
            
            circuit.append("TICK")
        
        # Step 3: Measure all ancillas with appropriate basis
        if self._stabilizer_swapped:
            # X ancillas measured Z-type, Z ancillas measured X-type
            x_meas_start = self.ctx.add_measurement(self._n_x)
            circuit.append("MR", x_anc)
            
            z_meas_start = self.ctx.add_measurement(self._n_z)
            circuit.append("MRX", z_anc)
        else:
            # X ancillas measure X-type, Z ancillas measure Z-type
            x_meas_start = self.ctx.add_measurement(self._n_x)
            circuit.append("MRX", x_anc)
            
            z_meas_start = self.ctx.add_measurement(self._n_z)
            circuit.append("MR", z_anc)
        
        # Emit detectors using the unified helper
        # X effective basis: after swap X ancillas measure Z, before swap they measure X
        # Z effective basis: after swap Z ancillas measure X, before swap they measure Z
        x_effective_basis = "Z" if self._stabilizer_swapped else "X"
        z_effective_basis = "X" if self._stabilizer_swapped else "Z"
        
        self._emit_detectors_for_type(circuit, "x", x_meas_start, emit_detectors, x_effective_basis)
        self._emit_detectors_for_type(circuit, "z", z_meas_start, emit_detectors, z_effective_basis)

        # MR/MRX already reset the ancillas; mark so the next round
        # can skip the redundant R/RX.
        self._ancilla_already_reset = True
        
        # Add final TICK after measurements to separate from next round
        circuit.append("TICK")

    def _emit_x_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """Emit X stabilizer measurements.
        
        Uses CX-based measurement with native RX/MRX instructions:
        - RX resets ancillas to |+⟩ directly
        - CX(ancilla → data) for parity extraction
        - MRX measures in X basis directly
        
        After Hadamard (when _stabilizer_swapped=True), the X ancillas measure
        what are now Z stabilizers, so we use standard Z-style R/CNOT/MR circuit.
        """
        if self._hx is None or self._n_x == 0:
            return
        
        x_anc = self.x_ancillas
        
        if self._stabilizer_swapped:
            # After Hadamard: X ancillas measure Z-type stabilizers
            # Use Z syndrome circuit: R - CNOT(data→anc) - MR
            if not self._ancilla_already_reset:
                circuit.append("R", x_anc)
            circuit.append("TICK")
            
            if self._use_geometric_x():
                self._emit_geometric_cnots(circuit, "x")
            else:
                self._emit_graph_coloring_cnots(circuit, self._hx, self.data_qubits, x_anc, is_x_type=False)
            
            circuit.append("TICK")
            meas_start = self.ctx.add_measurement(self._n_x)
            circuit.append("MR", x_anc)
        else:
            # Normal: X ancillas measure X-type stabilizers
            # Use X syndrome circuit: RX - CX(anc→data) - MRX
            if not self._ancilla_already_reset:
                circuit.append("RX", x_anc)
            circuit.append("TICK")
            
            if self._use_geometric_x():
                self._emit_geometric_cnots(circuit, "x")
            else:
                self._emit_graph_coloring_cnots(circuit, self._hx, self.data_qubits, x_anc, is_x_type=True)
            
            circuit.append("TICK")
            meas_start = self.ctx.add_measurement(self._n_x)
            circuit.append("MRX", x_anc)
        
        # Effective basis: after swap X ancillas measure Z, before swap they measure X
        effective_basis = "Z" if self._stabilizer_swapped else "X"
        self._emit_detectors_for_type(circuit, "x", meas_start, emit_detectors, effective_basis)

        # MR/MRX already reset ancillas
        self._ancilla_already_reset = True
    
    def _emit_z_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """Emit Z stabilizer measurements.
        
        Uses standard Z measurement: R - CNOT(data→anc) - MR
        
        After Hadamard (when _stabilizer_swapped=True), the Z ancillas measure
        what are now X stabilizers, so we use RX/CX/MRX pattern.
        """
        if self._hz is None or self._n_z == 0:
            return
        
        z_anc = self.z_ancillas
        
        if self._stabilizer_swapped:
            # After Hadamard: Z ancillas measure X-type stabilizers  
            # Use X syndrome circuit: RX - CX(anc→data) - MRX
            if not self._ancilla_already_reset:
                circuit.append("RX", z_anc)
            circuit.append("TICK")
            
            if self._use_geometric_z():
                self._emit_geometric_cnots(circuit, "z")
            else:
                self._emit_graph_coloring_cnots(circuit, self._hz, self.data_qubits, z_anc, is_x_type=True)
            
            circuit.append("TICK")
            meas_start = self.ctx.add_measurement(self._n_z)
            circuit.append("MRX", z_anc)
        else:
            # Normal: Z ancillas measure Z-type stabilizers
            # Use Z syndrome circuit: R - CNOT(data→anc) - MR
            if not self._ancilla_already_reset:
                circuit.append("R", z_anc)
            circuit.append("TICK")
            
            if self._use_geometric_z():
                self._emit_geometric_cnots(circuit, "z")
            else:
                self._emit_graph_coloring_cnots(circuit, self._hz, self.data_qubits, z_anc, is_x_type=False)
            
            circuit.append("TICK")
            meas_start = self.ctx.add_measurement(self._n_z)
            circuit.append("MR", z_anc)
        
        # Effective basis: after swap Z ancillas measure X, before swap they measure Z
        effective_basis = "X" if self._stabilizer_swapped else "Z"
        self._emit_detectors_for_type(circuit, "z", meas_start, emit_detectors, effective_basis)

        # MR/MRX already reset ancillas
        self._ancilla_already_reset = True
    
    def _emit_metacheck_round(
        self,
        circuit: stim.Circuit,
        x_syndrome_meas_start: int,
        z_syndrome_meas_start: int,
        emit_detectors: bool = True,
    ) -> None:
        """
        Emit metacheck measurements for single-shot error correction.
        
        Metachecks verify the consistency of syndrome measurements:
        - meta_x: checks the parity of Z syndrome bits (meta_x @ hz = 0)
        - meta_z: checks the parity of X syndrome bits (meta_z @ hx = 0)
        
        For 4D surface codes (5-chain), both meta_x and meta_z exist.
        For 3D toric codes (4-chain), only one type exists depending on qubit grade.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        x_syndrome_meas_start : int
            Starting measurement index for X syndrome bits (needed for meta_z).
        z_syndrome_meas_start : int
            Starting measurement index for Z syndrome bits (needed for meta_x).
        emit_detectors : bool
            Whether to emit metacheck detectors.
        """
        if not self._enable_metachecks:
            return
        
        # Meta-X checks (check Z syndrome parity)
        if self._n_meta_x > 0 and self._meta_x is not None:
            self._emit_meta_x_checks(
                circuit, z_syndrome_meas_start, emit_detectors
            )
        
        # Meta-Z checks (check X syndrome parity)
        if self._n_meta_z > 0 and self._meta_z is not None:
            self._emit_meta_z_checks(
                circuit, x_syndrome_meas_start, emit_detectors
            )
    
    def _emit_meta_x_checks(
        self,
        circuit: stim.Circuit,
        z_syndrome_meas_start: int,
        emit_detectors: bool,
    ) -> None:
        """
        Emit meta-X checks that verify Z syndrome consistency.
        
        Each meta-X check computes: ⊕_{j where meta_x[i,j]=1} z_syndrome[j]
        """
        # Build current round's syndrome indices for each metacheck
        cur_syndrome_indices_list = []
        for meta_idx in range(self._n_meta_x):
            row = self._meta_x[meta_idx]
            cur_indices = [
                z_syndrome_meas_start + j
                for j in range(len(row)) if row[j] != 0
            ]
            cur_syndrome_indices_list.append(cur_indices)
        
        if emit_detectors:
            for meta_idx in range(self._n_meta_x):
                cur_indices = cur_syndrome_indices_list[meta_idx]
                prev_indices = self._last_meta_x_meas[meta_idx]
                
                if not cur_indices:
                    continue
                
                # Compute detector coordinate (use centroid of covered syndrome ancillas)
                coord = self._get_metacheck_coord("x", meta_idx)
                
                if prev_indices is None:
                    # First round: emit spatial metacheck detector if in single-shot mode
                    effective_z_basis = "X" if self._stabilizer_swapped else "Z"
                    if self._single_shot_metachecks and self.measurement_basis == effective_z_basis:
                        self.ctx.emit_detector(circuit, cur_indices, coord)
                else:
                    # TIME-LIKE DETECTOR: Compare XOR(prev) with XOR(cur)
                    all_indices = prev_indices + cur_indices
                    self.ctx.emit_detector(circuit, all_indices, coord)
        
        # Update tracking for next round
        for meta_idx in range(self._n_meta_x):
            self._last_meta_x_meas[meta_idx] = cur_syndrome_indices_list[meta_idx]
    
    def _emit_meta_z_checks(
        self,
        circuit: stim.Circuit,
        x_syndrome_meas_start: int,
        emit_detectors: bool,
    ) -> None:
        """
        Emit meta-Z checks that verify X syndrome consistency.
        
        Each meta-Z check computes: ⊕_{j where meta_z[i,j]=1} x_syndrome[j]
        """
        # Build current round's syndrome indices for each metacheck
        cur_syndrome_indices_list = []
        for meta_idx in range(self._n_meta_z):
            row = self._meta_z[meta_idx]
            cur_indices = [
                x_syndrome_meas_start + j
                for j in range(len(row)) if row[j] != 0
            ]
            cur_syndrome_indices_list.append(cur_indices)
        
        if emit_detectors:
            for meta_idx in range(self._n_meta_z):
                cur_indices = cur_syndrome_indices_list[meta_idx]
                prev_indices = self._last_meta_z_meas[meta_idx]
                
                if not cur_indices:
                    continue
                
                # Compute detector coordinate (use centroid of covered syndrome ancillas)
                coord = self._get_metacheck_coord("z", meta_idx)
                
                if prev_indices is None:
                    # First round: emit spatial metacheck detector if in single-shot mode
                    effective_x_basis = "Z" if self._stabilizer_swapped else "X"
                    if self._single_shot_metachecks and self.measurement_basis == effective_x_basis:
                        self.ctx.emit_detector(circuit, cur_indices, coord)
                else:
                    # TIME-LIKE DETECTOR: Compare XOR(prev) with XOR(cur)
                    all_indices = prev_indices + cur_indices
                    self.ctx.emit_detector(circuit, all_indices, coord)
        
        # Update tracking for next round
        for meta_idx in range(self._n_meta_z):
            self._last_meta_z_meas[meta_idx] = cur_syndrome_indices_list[meta_idx]

    def _emit_detectors_for_type(
        self,
        circuit: stim.Circuit,
        stab_type: str,
        meas_start: int,
        emit_detectors: bool,
        effective_first_round_basis: str,
    ) -> None:
        """Emit detectors for one stabilizer type (X or Z).
        
        This is a unified helper that handles all detector emission logic:
        - First-round anchor detectors (based on measurement basis)
        - Time-like detectors (comparing current to previous round)
        - Transform-aware measurements (after CZ/CNOT gates)
        - Measurement recording in the context
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit to emit detectors into.
        stab_type : str
            'x' for X stabilizers or 'z' for Z stabilizers.
        meas_start : int
            Starting measurement index for this stabilizer type.
        emit_detectors : bool
            Whether to emit detector instructions (False just records measurements).
        effective_first_round_basis : str
            The basis that matches this stabilizer type for first-round detectors.
            E.g. "X" for X-type (|+⟩ state), "Z" for Z-type (|0⟩ state).
        """
        if stab_type == "x":
            n_stab = self._n_x
            last_meas = self._last_x_meas
            emit_anchors = self._emit_x_anchors
            first_round_config = self._ft_config.first_round_x_detectors
            get_detector_meas = self.ctx.get_x_detector_measurements
            consume_transforms = self.ctx.consume_x_transforms
        else:
            n_stab = self._n_z
            last_meas = self._last_z_meas
            emit_anchors = self._emit_z_anchors
            first_round_config = self._ft_config.first_round_z_detectors
            get_detector_meas = self.ctx.get_z_detector_measurements
            consume_transforms = self.ctx.consume_z_transforms
        
        if emit_detectors:
            for s_idx in range(n_stab):
                cur_meas = meas_start + s_idx
                prev_meas = last_meas[s_idx]
                
                # Check for pending transforms (from CZ/CNOT gates)
                has_transforms = self.ctx.has_pending_transforms(self.block_name, stab_type, s_idx)
                
                if prev_meas is None and not has_transforms:
                    # First round: emit anchor detector if appropriate
                    #
                    # In explicit_anchor_mode, ONLY use the explicit flag:
                    # - emit_anchors=True => emit anchor
                    # - emit_anchors=False => no anchor
                    #
                    # In legacy mode (no explicit flags set), use measurement_basis
                    if getattr(self, '_explicit_anchor_mode', False):
                        should_emit = not self._skip_first_round and emit_anchors
                    else:
                        should_emit = (
                            not self._skip_first_round and (
                                emit_anchors or  # Explicit anchor request
                                (first_round_config and
                                 self.measurement_basis == effective_first_round_basis)
                            )
                        )
                    if should_emit:
                        coord = self._get_stab_coord(stab_type, s_idx)
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                elif not self._skip_first_round:
                    # Time-like detector: use transform-aware detector measurements
                    meas_indices = get_detector_meas(self.block_name, s_idx, cur_meas)
                    coord = self._get_stab_coord(stab_type, s_idx)
                    self.ctx.emit_detector(circuit, meas_indices, coord)
                    # Consume the transforms after emitting
                    consume_transforms(self.block_name, s_idx)
                
                last_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(self.block_name, stab_type, s_idx, cur_meas)
        else:
            for s_idx in range(n_stab):
                cur_meas = meas_start + s_idx
                last_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(self.block_name, stab_type, s_idx, cur_meas)

    def _use_geometric(self, stab_type: str) -> bool:
        """Check if geometric scheduling should be used for the given stabilizer type.
        
        Parameters
        ----------
        stab_type : str
            'x' for X stabilizers or 'z' for Z stabilizers.
            
        Returns
        -------
        bool
            True if geometric scheduling can be used, False otherwise.
        """
        mode = self._ft_config.schedule_mode
        if mode == ScheduleMode.GRAPH_COLORING:
            return False
        
        if stab_type == "x":
            schedule, stab_coords, matrix, n_stab = (
                self._x_schedule, self._x_stab_coords, self._hx, self._n_x
            )
        else:
            schedule, stab_coords, matrix, n_stab = (
                self._z_schedule, self._z_stab_coords, self._hz, self._n_z
            )
        
        if not (schedule is not None and self._data_coords and len(stab_coords) == n_stab):
            return False
        
        return self._validate_geometric_schedule(schedule, stab_coords, matrix)
    
    def _use_geometric_x(self) -> bool:
        """Check if geometric scheduling should be used for X stabilizers."""
        return self._use_geometric("x")
    
    def _use_geometric_z(self) -> bool:
        """Check if geometric scheduling should be used for Z stabilizers."""
        return self._use_geometric("z")
    
    def _validate_geometric_schedule(
        self,
        schedule: List[Tuple[float, float]],
        stab_coords: List[Tuple[float, ...]],
        stab_matrix: Optional[np.ndarray],
    ) -> bool:
        """Validate that geometric scheduling will cover all stabilizer-data connections."""
        if stab_matrix is None or stab_matrix.size == 0:
            return False
        
        n_stabs = stab_matrix.shape[0]
        
        for s_idx in range(min(n_stabs, len(stab_coords))):
            sx, sy = stab_coords[s_idx][:2] if len(stab_coords[s_idx]) >= 2 else (0, 0)
            
            if stab_matrix.ndim == 1:
                expected = int(stab_matrix[s_idx]) if s_idx < len(stab_matrix) else 0
            else:
                expected = int(np.sum(stab_matrix[s_idx] != 0))
            
            found = 0
            for dx, dy in schedule:
                nbr = (float(sx) + dx, float(sy) + dy)
                if self._lookup_data_qubit(nbr) is not None:
                    found += 1
            
            if found < expected:
                return False
        
        return True
    
    def _emit_geometric_cnots(self, circuit: stim.Circuit, stab_type: str) -> None:
        """Emit CNOTs using geometric scheduling.
        
        Gate directions:
        - X-type stabilizers: CX(ancilla → data) - ancilla controls data
        - Z-type stabilizers: CX(data → ancilla) - data controls ancilla
        """
        if stab_type == "x":
            schedule = self._x_schedule
            stab_coords = self._x_stab_coords
            ancillas = self.x_ancillas
            is_x_type = True
        else:
            schedule = self._z_schedule
            stab_coords = self._z_stab_coords
            ancillas = self.z_ancillas
            is_x_type = False
        
        for layer_idx, (dx, dy) in enumerate(schedule):
            if layer_idx > 0:
                circuit.append("TICK")
            for s_idx, (sx, sy) in enumerate(stab_coords):
                if s_idx >= len(ancillas):
                    continue
                anc = ancillas[s_idx]
                nbr = (float(sx) + dx, float(sy) + dy)
                dq = self._lookup_data_qubit(nbr)
                if dq is not None:
                    if is_x_type:
                        # X-type: ancilla controls data
                        circuit.append("CX", [anc, dq])
                    else:
                        # Z-type: data controls ancilla
                        circuit.append("CX", [dq, anc])

    def _emit_graph_coloring_cnots(
        self,
        circuit: stim.Circuit,
        stab_matrix: np.ndarray,
        data_qubits: List[int],
        ancilla_qubits: List[int],
        is_x_type: bool = True,
    ) -> None:
        """Emit CNOTs using greedy graph coloring.
        
        Gate directions:
        - X-type stabilizers (is_x_type=True): CX(ancilla → data)
        - Z-type stabilizers (is_x_type=False): CX(data → ancilla)
        """
        if stab_matrix is None or stab_matrix.size == 0:
            return
        
        n_stabs, n_data = stab_matrix.shape
        
        # Build list of (ctrl, tgt) pairs for graph coloring
        all_gates: List[Tuple[int, int]] = []
        for s_idx in range(min(n_stabs, len(ancilla_qubits))):
            anc = ancilla_qubits[s_idx]
            for d_idx in range(min(n_data, len(data_qubits))):
                if stab_matrix[s_idx, d_idx]:
                    dq = data_qubits[d_idx]
                    if is_x_type:
                        # X-type: ancilla controls data
                        all_gates.append((anc, dq))
                    else:
                        # Z-type: data controls ancilla
                        all_gates.append((dq, anc))
        
        if not all_gates:
            return
        
        layers = graph_coloring_cnots(all_gates)
        
        for layer_idx, layer in enumerate(layers):
            if layer_idx > 0:
                circuit.append("TICK")
            for ctrl, tgt in layer:
                circuit.append("CX", [ctrl, tgt])
    
    def _get_stab_coord(self, stab_type: str, s_idx: int) -> Tuple[float, float, float]:
        """Get detector coordinate for a stabilizer.
        
        Applies the block's spatial coord_offset so detector coordinates
        are in the global (layout-level) reference frame.
        """
        if stab_type == "x":
            coords = self._x_stab_coords
        else:
            coords = self._z_stab_coords
        
        ox = self._coord_offset[0] if len(self._coord_offset) > 0 else 0.0
        oy = self._coord_offset[1] if len(self._coord_offset) > 1 else 0.0
        
        if s_idx < len(coords):
            x, y = coords[s_idx][:2]
            return (float(x) + ox, float(y) + oy, self.ctx.current_time)
        return (ox, oy, self.ctx.current_time)
    
    def _get_metacheck_coord(self, meta_type: str, meta_idx: int) -> Tuple[float, float, float, float]:
        """Get detector coordinate for a metacheck.
        
        Applies the block's spatial coord_offset.
        """
        if meta_type == "x":
            meta_matrix = self._meta_x
            coords = self._z_stab_coords
        else:
            meta_matrix = self._meta_z
            coords = self._x_stab_coords
        
        ox = self._coord_offset[0] if len(self._coord_offset) > 0 else 0.0
        oy = self._coord_offset[1] if len(self._coord_offset) > 1 else 0.0
        
        if meta_matrix is None or meta_idx >= meta_matrix.shape[0]:
            return (ox, oy, self.ctx.current_time, 1.0)
        
        row = meta_matrix[meta_idx]
        covered_indices = np.where(row)[0]
        
        if len(covered_indices) == 0 or len(coords) == 0:
            return (ox, oy, self.ctx.current_time, 1.0)
        
        x_sum, y_sum, count = 0.0, 0.0, 0
        for s_idx in covered_indices:
            if s_idx < len(coords):
                x_sum += coords[s_idx][0]
                y_sum += coords[s_idx][1]
                count += 1
        
        if count > 0:
            return (x_sum / count + ox, y_sum / count + oy, self.ctx.current_time, 1.0)
        return (ox, oy, self.ctx.current_time, 1.0)
    
    def get_last_measurement_indices(self) -> dict:
        """Get the last measurement indices for each stabilizer type.
        
        Returns a dict with:
        - 'x': list of measurement indices for X stabilizers (or None if not measured)
        - 'z': list of measurement indices for Z stabilizers (or None if not measured)
        - 'round': current round number
        
        This allows external code to construct custom detectors after calling
        emit_round() with emit_detectors=False.
        """
        return {
            'x': list(self._last_x_meas),
            'z': list(self._last_z_meas),
            'round': self._round_number,
        }
    
    def emit_rounds(
        self,
        circuit: stim.Circuit,
        num_rounds: int,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_metachecks: bool = False,
        emit_detectors: bool = True,
    ) -> None:
        """Emit multiple stabilizer measurement rounds.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        num_rounds : int
            Number of rounds to emit.
        stab_type : StabilizerBasis
            Which stabilizers to measure.
        emit_metachecks : bool
            Whether to emit metacheck detectors.
        emit_detectors : bool
            Whether to emit time-like detectors. Set to False when you need
            to construct custom detectors (e.g., for cross-block correlations
            in teleportation gadgets). Use get_last_measurement_indices() to
            retrieve measurement info for manual detector construction.
        """
        for _ in range(num_rounds):
            self.emit_round(circuit, stab_type, emit_detectors=emit_detectors, emit_metachecks=emit_metachecks)
    
    def emit_scheduled_rounds(
        self,
        circuit: stim.Circuit,
        num_rounds: int,
        prep_basis: str,
        meas_basis: str,
        emit_detectors: bool = True,
    ) -> None:
        """
        Emit rounds with optimal scheduling for deterministic detectors.
        
        This uses StabilizerScheduler to determine optimal measurement ordering:
        - First round: Deterministic basis measured FIRST for anchor detectors
        - Last round: Deterministic basis measured LAST for boundary detectors
        - Middle rounds: Standard ordering
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        num_rounds : int
            Number of syndrome rounds.
        prep_basis : str
            Preparation basis: "0" for |0⟩, "+" for |+⟩.
        meas_basis : str
            Final measurement basis: "Z" or "X".
        emit_detectors : bool
            Whether to emit detectors.
            
        Example
        -------
        For |0⟩ prep → MZ measurement:
        - Round 1: Z first (Z has anchors), X second
        - Round N: X first, Z last (Z has boundaries)
        
        For |+⟩ prep → MX measurement:
        - Round 1: X first (X has anchors), Z second
        - Round N: Z first, X last (X has boundaries)
        """
        from qectostim.gadgets.scheduling import StabilizerScheduler
        
        scheduler = StabilizerScheduler()
        
        # Compute which basis is deterministic for each detector type
        anchor_basis = scheduler.get_anchor_deterministic_basis(prep_basis)
        boundary_basis = scheduler.get_boundary_deterministic_basis(meas_basis)
        
        for round_idx in range(num_rounds):
            is_first = (round_idx == 0)
            is_last = (round_idx == num_rounds - 1)
            
            # Determine first_basis for this round
            if is_first and is_last:
                # Single round: anchor takes priority
                first_basis = anchor_basis
            elif is_first:
                # First round: deterministic basis first for anchors
                first_basis = anchor_basis
            elif is_last:
                # Last round: deterministic basis last for boundaries
                # So the OTHER basis is first
                other_basis = StabilizerBasis.X if boundary_basis == StabilizerBasis.Z else StabilizerBasis.Z
                first_basis = other_basis
            else:
                # Middle rounds: standard order (Z first is typical)
                first_basis = StabilizerBasis.Z
            
            # Determine anchor flags
            emit_z_anchors = is_first and anchor_basis == StabilizerBasis.Z
            emit_x_anchors = is_first and anchor_basis == StabilizerBasis.X
            
            self.emit_round(
                circuit,
                emit_detectors=emit_detectors,
                emit_z_anchors=emit_z_anchors,
                emit_x_anchors=emit_x_anchors,
                first_basis=first_basis,
            )
    
    def emit_final_measurement(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
        logical_idx: int = 0,
    ) -> List[int]:
        """Emit final data qubit measurements with space-like detectors."""
        from .utils import get_logical_support
        
        data = self.data_qubits
        n = len(data)
        basis = basis.upper()
        
        if basis == "X":
            circuit.append("H", data)
        
        meas_start = self.ctx.add_measurement(n)
        circuit.append("M", data)
        
        data_meas = {q: meas_start + i for i, q in enumerate(range(n))}
        
        if basis == "Z" and self._hz is not None:
            for s_idx in range(self._n_z):
                last_meas = self._last_z_meas[s_idx]
                if last_meas is None:
                    continue
                
                row = self._hz[s_idx]
                data_idxs = [meas_start + d_idx for d_idx in range(min(n, len(row))) if row[d_idx]]
                
                if data_idxs:
                    recs = data_idxs + [last_meas]
                    coord = self._get_stab_coord("z", s_idx)
                    coord = (coord[0], coord[1], self.ctx.current_time) + coord[3:]
                    self.ctx.emit_detector(circuit, recs, coord)
        
        elif basis == "X" and self._hx is not None:
            for s_idx in range(self._n_x):
                last_meas = self._last_x_meas[s_idx]
                if last_meas is None:
                    continue
                
                row = self._hx[s_idx]
                data_idxs = [meas_start + d_idx for d_idx in range(min(n, len(row))) if row[d_idx]]
                
                if data_idxs:
                    recs = data_idxs + [last_meas]
                    coord = self._get_stab_coord("x", s_idx)
                    coord = (coord[0], coord[1], self.ctx.current_time) + coord[3:]
                    self.ctx.emit_detector(circuit, recs, coord)
        
        logical_meas = []
        code = self.code
        
        effective_basis = self.ctx.get_transformed_basis(logical_idx, basis)
        
        if effective_basis == "Z":
            logical_support = get_logical_support(code, "Z", logical_idx)
        else:
            logical_support = get_logical_support(code, "X", logical_idx)
        
        if logical_support:
            logical_meas = [meas_start + q for q in logical_support if q < n]
        else:
            import warnings
            warnings.warn(
                f"No valid logical {effective_basis} operator found for {type(code).__name__}. "
                f"Observable will track qubit 0 only - decoding may not work correctly.",
                RuntimeWarning,
                stacklevel=2
            )
            logical_meas = [meas_start]
        
        self.ctx.add_observable_measurement(logical_idx, logical_meas)
        
        return logical_meas
    
    def emit_space_like_detectors(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
        data_meas_start: Optional[int] = None,
    ) -> None:
        """Emit space-like detectors comparing final data measurements with last stabilizer round."""
        n = len(self.data_qubits)
        
        if data_meas_start is not None:
            meas_start = data_meas_start
        else:
            meas_start = self.ctx.measurement_index - n
        
        logical_basis = basis.upper()
        if self._stabilizer_swapped:
            physical_type = "x" if logical_basis == "Z" else "z"
        else:
            physical_type = "z" if logical_basis == "Z" else "x"
        
        if physical_type == "z" and self._hz is not None:
            for s_idx in range(self._n_z):
                last_meas = self.ctx.last_stabilizer_meas.get(
                    (self.block_name, "z", s_idx)
                )
                if last_meas is None:
                    continue
                
                support = []
                for d_idx in range(min(n, self._hz.shape[1])):
                    if self._hz[s_idx, d_idx]:
                        support.append(meas_start + d_idx)
                
                if support:
                    coord = self._get_stab_coord("z", s_idx)
                    self.ctx.emit_detector(circuit, [last_meas] + support, coord)
        
        elif physical_type == "x" and self._hx is not None:
            for s_idx in range(self._n_x):
                last_meas = self.ctx.last_stabilizer_meas.get(
                    (self.block_name, "x", s_idx)
                )
                if last_meas is None:
                    continue
                
                support = []
                for d_idx in range(min(n, self._hx.shape[1])):
                    if self._hx[s_idx, d_idx]:
                        support.append(meas_start + d_idx)
                
                if support:
                    coord = self._get_stab_coord("x", s_idx)
                    self.ctx.emit_detector(circuit, [last_meas] + support, coord)
