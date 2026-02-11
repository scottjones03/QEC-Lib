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
    from qectostim.gadgets.preparation import CSSStatePreparation, LogicalBasis


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
        x_stabilizer_mode: str = "cz",
        coord_offset: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__(code, ctx, block_name, data_offset, ancilla_offset, measurement_basis, coord_offset=coord_offset)
        
        # X-stabilizer extraction mode:
        # - "cz": Use CZ gates (default, safe for memory experiments)
        #         H-CZ-H-Measure measures X parity without disturbing Z-basis state
        # - "cx": Use CX gates (required for teleportation gadgets)
        #         H-CX-H-Measure: after teleportation CNOT, post-CNOT X_D syndrome
        #         remains local to the data block, enabling deterministic detectors
        # 
        # IMPORTANT: Teleportation requires "cx" mode because CZ-based extraction
        # couples X_D measurements to Z_A after the teleportation CNOT, making
        # the boundary detectors non-deterministic.
        if x_stabilizer_mode not in ("cz", "cx"):
            raise ValueError(f"x_stabilizer_mode must be 'cz' or 'cx', got {x_stabilizer_mode!r}")
        self._x_stabilizer_mode = x_stabilizer_mode
        
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
    
    def reset_stabilizer_history(self, swap_xz: bool = False, skip_first_round: bool = False) -> None:
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
    
    def emit_qubit_coords(self, circuit: stim.Circuit) -> None:
        """Emit QUBIT_COORDS for all qubits in this block."""
        # Data qubits
        for local_idx, coord in enumerate(self._data_coords):
            if len(coord) >= 2:
                global_idx = self.data_offset + local_idx
                circuit.append("QUBIT_COORDS", [global_idx], [float(coord[0]), float(coord[1])])
        
        # X ancillas
        for local_idx, coord in enumerate(self._x_stab_coords):
            if len(coord) >= 2 and local_idx < self._n_x:
                global_idx = self.ancilla_offset + local_idx
                circuit.append("QUBIT_COORDS", [global_idx], [float(coord[0]), float(coord[1])])
        
        # Z ancillas
        for local_idx, coord in enumerate(self._z_stab_coords):
            if len(coord) >= 2 and local_idx < self._n_z:
                global_idx = self.ancilla_offset + self._n_x + local_idx
                circuit.append("QUBIT_COORDS", [global_idx], [float(coord[0]), float(coord[1])])
    
    def emit_reset_all(self, circuit: stim.Circuit) -> None:
        """Reset all data, ancilla, and metacheck ancilla qubits."""
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
        use_projection: bool = True,
    ) -> Dict[str, List[int]]:
        """
        Prepare a logical eigenstate using Pauli frame projection.
        
        This implements the correct Pauli frame approach for CSS code preparation
        using the CSSStatePreparation strategy:
        - For |0⟩_L: Reset to |0⟩^⊗n, measure stabilizers to project into codespace
        - For |+⟩_L: Reset to |0⟩^⊗n, H on all, measure stabilizers to project
        - For |1⟩_L: Same as |0⟩_L + logical X
        - For |-⟩_L: Same as |+⟩_L + logical Z
        
        The stabilizer measurements PROJECT into the codespace. The eigenvalues
        may be random (±1), but they are CONSISTENT - this is the Pauli frame.
        The frame information is returned so it can be tracked and applied
        at final measurement.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        state : str
            "0" for |0⟩_L, "1" for |1⟩_L, "+" for |+⟩_L, "-" for |-⟩_L.
        logical_idx : int
            Which logical qubit.
        use_projection : bool
            If True (default), use stabilizer projection for proper Pauli frame
            encoding. If False, use the simple (incorrect for general CSS) approach
            of assuming |0⟩^⊗n is already in codespace.
            
        Returns
        -------
        Dict[str, List[int]]
            Measurement indices from projection:
            - 'x_stab_meas': X stabilizer measurement indices
            - 'z_stab_meas': Z stabilizer measurement indices
            - 'logical_meas': Logical operator measurement index (if measured)
            Empty dict if use_projection=False.
        """
        code = self.code
        result: Dict[str, List[int]] = {
            'x_stab_meas': [],
            'z_stab_meas': [],
            'logical_meas': [],
        }
        
        if not use_projection:
            # Legacy behavior - simple but incorrect for general CSS codes
            # Kept for backward compatibility and testing
            if state in ("0", "1"):
                if state == "1" and hasattr(code, 'logical_x_support'):
                    support = code.logical_x_support(logical_idx)
                    for q in support:
                        circuit.append("X", [self.data_offset + q])
            elif state in ("+", "-"):
                if state == "-" and hasattr(code, 'logical_x_support'):
                    support = code.logical_x_support(logical_idx)
                    for q in support:
                        circuit.append("Z", [self.data_offset + q])
                circuit.append("H", self.data_qubits)
            circuit.append("TICK")
            return result
        
        # === USE CSSStatePreparation STRATEGY ===
        # This ensures consistent state preparation across all CSS components
        # Import lazily to avoid circular imports
        from qectostim.gadgets.preparation import CSSStatePreparation
        css_prep = CSSStatePreparation(code)
        
        # Get the projection ancilla (after all stabilizer ancillas)
        proj_ancilla = self.ancilla_offset + self._n_x + self._n_z
        if self._enable_metachecks:
            proj_ancilla += self._n_meta_x + self._n_meta_z
        
        # Record where measurements will start in the DetectorContext
        meas_start = self.ctx.measurement_index
        
        # Prepare the state using CSSStatePreparation
        if state in ("0", "1"):
            prep_result = css_prep.prepare_zero(
                circuit, self.data_qubits, proj_ancilla,
                num_rounds=1, emit_detectors=False
            )
        else:  # "+" or "-"
            prep_result = css_prep.prepare_plus(
                circuit, self.data_qubits, proj_ancilla,
                num_rounds=1, emit_detectors=False
            )
        
        # Map the local measurement indices from CSSStatePreparation to global DetectorContext indices
        # CSSStatePreparation uses local indices starting from 0, we need to add meas_start
        num_meas = len(prep_result.all_measurements)
        for _ in range(num_meas):
            self.ctx.add_measurement(1)  # Register each measurement with context
        
        # Convert local indices to global indices
        result['z_stab_meas'] = [m + meas_start for m in prep_result.final_z_meas]
        result['x_stab_meas'] = [m + meas_start for m in prep_result.final_x_meas]
        
        # Apply logical operator for |1⟩_L or |-⟩_L
        if state == "1" and hasattr(code, 'logical_x_support'):
            support = code.logical_x_support(logical_idx)
            for q in support:
                circuit.append("X", [self.data_offset + q])
            circuit.append("TICK")
        elif state == "-" and hasattr(code, 'logical_z_support'):
            support = code.logical_z_support(logical_idx)
            for q in support:
                circuit.append("Z", [self.data_offset + q])
            circuit.append("TICK")
        
        # Initialize stabilizer history from projection measurements
        self._initialize_stabilizer_history_from_projection(result)
        
        # CRITICAL: Skip first-round detectors after projection!
        # Projection measurements give random ±1 outcomes, so detectors comparing
        # first round to projection would be non-deterministic. We establish the
        # stabilizer history for measurement index tracking, but don't emit detectors.
        # Second and subsequent rounds will have deterministic time-like detectors.
        self._skip_first_round = True
        
        return result
    
    def _emit_projection_measurements(
        self,
        circuit: stim.Circuit,
        state_type: str,
    ) -> Dict[str, List[int]]:
        """
        Emit stabilizer measurements to project into the codespace.
        
        DEPRECATED: This method is kept for backward compatibility.
        New code should use emit_prepare_logical_state() which delegates
        to CSSStatePreparation.
        
        This is the core of Pauli frame encoding. The measurements project
        |0⟩^⊗n or |+⟩^⊗n into the code space. The measurement outcomes are
        random (±1) but define a consistent Pauli frame.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        state_type : str
            "0" or "1" for Z-basis, "+" or "-" for X-basis.
            
        Returns
        -------
        Dict[str, List[int]]
            Measurement indices for each stabilizer type.
        """
        n = self.code.n
        data_qubits = self.data_qubits
        
        # Use a dedicated projection ancilla (after all stabilizer ancillas)
        # This avoids conflicts with the parallel stabilizer ancillas
        proj_ancilla = self.ancilla_offset + self._n_x + self._n_z
        if self._enable_metachecks:
            proj_ancilla += self._n_meta_x + self._n_meta_z
        
        result = {
            'x_stab_meas': [],
            'z_stab_meas': [],
            'logical_meas': [],
        }
        
        # === Measure Z-type stabilizers ===
        # Circuit: R-CX(data→anc)-M (NO H gates!)
        # CNOT propagates Z: measures Z parity directly
        if self._hz is not None and self._n_z > 0:
            for row_idx in range(self._n_z):
                # Reset ancilla to |0⟩
                circuit.append("R", [proj_ancilla])
                
                # CNOT from each data qubit in stabilizer support to ancilla
                # data controls ancilla → measures Z parity
                for q in range(n):
                    if self._hz[row_idx, q]:
                        circuit.append("CX", [data_qubits[q], proj_ancilla])
                
                # Measure in Z-basis
                meas_idx = self.ctx.add_measurement(1)
                circuit.append("M", [proj_ancilla])
                result['z_stab_meas'].append(meas_idx)
            
            circuit.append("TICK")
        
        # === Measure X-type stabilizers ===
        # Circuit: R-H-CX[anc→data]-H-M
        # Ancilla in |+⟩, CNOT applies X to data, measures X parity
        if self._hx is not None and self._n_x > 0:
            for row_idx in range(self._n_x):
                # Reset and prepare ancilla in |+⟩
                circuit.append("R", [proj_ancilla])
                circuit.append("H", [proj_ancilla])
                
                # CNOT from ancilla to each data qubit in stabilizer support
                # This measures X parity non-destructively for X eigenstates
                for q in range(n):
                    if self._hx[row_idx, q]:
                        circuit.append("CX", [proj_ancilla, data_qubits[q]])  # anc controls data
                
                # Measure in X-basis to get parity
                circuit.append("H", [proj_ancilla])
                meas_idx = self.ctx.add_measurement(1)
                circuit.append("M", [proj_ancilla])
                result['x_stab_meas'].append(meas_idx)
            
            circuit.append("TICK")
        
        return result
    
    def _initialize_stabilizer_history_from_projection(
        self,
        projection_result: Dict[str, List[int]],
    ) -> None:
        """
        Initialize stabilizer measurement history from projection.
        
        After projection, the stabilizer measurements establish a baseline
        for subsequent time-like detectors. We record these so emit_round()
        can compare against them.
        
        Parameters
        ----------
        projection_result : Dict[str, List[int]]
            Result from _emit_projection_measurements.
        """
        # Record X stabilizer measurements
        x_meas = projection_result.get('x_stab_meas', [])
        for s_idx, meas_idx in enumerate(x_meas):
            if s_idx < self._n_x:
                self._last_x_meas[s_idx] = meas_idx
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "x", s_idx, meas_idx
                )
        
        # Record Z stabilizer measurements
        z_meas = projection_result.get('z_stab_meas', [])
        for s_idx, meas_idx in enumerate(z_meas):
            if s_idx < self._n_z:
                self._last_z_meas[s_idx] = meas_idx
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "z", s_idx, meas_idx
                )
    
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
        1. H on X ancillas (prepare in |+⟩)
        2. For each phase: X and Z CNOTs together in same TICK
        3. H on X ancillas (rotate back to computational basis)
        4. Measure all ancillas
        
        After Hadamard (_stabilizer_swapped=True):
        - X ancillas now measure Z-type stabilizers → no H gates, data controls
        - Z ancillas now measure X-type stabilizers → H gates, ancilla controls
        """
        x_anc = self.x_ancillas
        z_anc = self.z_ancillas
        
        if self._stabilizer_swapped:
            # After Hadamard: swap the circuit patterns
            # Z ancillas measure X-type → H-CZ-H-M (use CZ to not disturb data!)
            # X ancillas measure Z-type → CNOT(data→anc)-M
            
            # Step 1: Prepare Z ancillas with H (they now measure X-type)
            circuit.append("H", z_anc)
            circuit.append("TICK")
            
            # Step 2: Interleaved CZ/CNOT phases - swapped control/target
            for phase_idx, ((dx_x, dy_x), (dx_z, dy_z)) in enumerate(
                zip(self._x_schedule, self._z_schedule)
            ):
                if phase_idx > 0:
                    circuit.append("TICK")
                
                # X ancillas: now measuring Z-type → data controls ancilla (CNOT)
                for s_idx, (sx, sy) in enumerate(self._x_stab_coords):
                    if s_idx >= len(x_anc):
                        continue
                    anc = x_anc[s_idx]
                    nbr = (float(sx) + dx_x, float(sy) + dy_x)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CNOT", [dq, anc])  # data controls ancilla
                
                # Z ancillas: now measuring X-type → use CZ (same fix as normal mode)
                # CRITICAL: Must use CZ, not CNOT[anc,data], to avoid flipping data!
                for s_idx, (sx, sy) in enumerate(self._z_stab_coords):
                    if s_idx >= len(z_anc):
                        continue
                    anc = z_anc[s_idx]
                    nbr = (float(sx) + dx_z, float(sy) + dy_z)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CZ", [anc, dq])  # CZ measures X parity
            
            circuit.append("TICK")
            
            # Step 3: Final H on Z ancillas (they measure X-type)
            circuit.append("H", z_anc)
            circuit.append("TICK")
            
        else:
            # Normal mode: standard surface code interleaving
            # Step 1: Prepare X ancillas with H
            circuit.append("H", x_anc)
            circuit.append("TICK")
            
            # Step 2: Interleaved CNOT/CZ phases - X and Z in parallel
            for phase_idx, ((dx_x, dy_x), (dx_z, dy_z)) in enumerate(
                zip(self._x_schedule, self._z_schedule)
            ):
                if phase_idx > 0:
                    circuit.append("TICK")
                
                # X stabilizer measurement for this phase
                # Gate choice depends on x_stabilizer_mode:
                # - "cz": Use CZ (default for memory experiments)
                #         CZ is symmetric and measures X parity without disturbing Z-basis
                # - "cx": Use CX(ancilla → data) (required for teleportation)
                #         After teleportation CNOT, CX keeps X_D syndromes local
                x_gate = "CZ" if self._x_stabilizer_mode == "cz" else "CNOT"
                for s_idx, (sx, sy) in enumerate(self._x_stab_coords):
                    if s_idx >= len(x_anc):
                        continue
                    anc = x_anc[s_idx]
                    nbr = (float(sx) + dx_x, float(sy) + dy_x)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        if self._x_stabilizer_mode == "cz":
                            circuit.append("CZ", [anc, dq])
                        else:
                            circuit.append("CNOT", [anc, dq])  # CX(anc → data)
                
                # Z CNOTs for this phase (SAME TICK - parallel with X)
                # For Z stabilizers: data qubits control ancilla (standard parity check)
                # This doesn't change data because target is ancilla.
                for s_idx, (sx, sy) in enumerate(self._z_stab_coords):
                    if s_idx >= len(z_anc):
                        continue
                    anc = z_anc[s_idx]
                    nbr = (float(sx) + dx_z, float(sy) + dy_z)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CNOT", [dq, anc])  # data controls ancilla
            
            # Separate final CNOT layer from H gates for proper DEM error granularity
            circuit.append("TICK")
            
            # Step 3: Final H on X ancillas
            circuit.append("H", x_anc)
            circuit.append("TICK")
        
        # Step 4: Measure all ancillas
        # X ancillas first, then Z ancillas
        x_meas_start = self.ctx.add_measurement(self._n_x)
        circuit.append("MR", x_anc)
        
        z_meas_start = self.ctx.add_measurement(self._n_z)
        circuit.append("MR", z_anc)
        
        # Emit detectors - after Hadamard, first-round logic changes
        # X ancillas now measure Z-type → first-round OK if basis == "Z"
        # Z ancillas now measure X-type → first-round OK if basis == "X"
        x_effective_basis = "Z" if self._stabilizer_swapped else "X"
        z_effective_basis = "X" if self._stabilizer_swapped else "Z"
        
        # Emit detectors for X stabilizers
        if emit_detectors:
            for s_idx in range(self._n_x):
                cur_meas = x_meas_start + s_idx
                prev_meas = self._last_x_meas[s_idx]
                
                # Check for pending transforms (from CZ/CNOT gates)
                has_transforms = self.ctx.has_pending_transforms(self.block_name, "x", s_idx)
                
                if prev_meas is None and not has_transforms:
                    # First round: emit anchor detector if appropriate
                    # 
                    # In explicit_anchor_mode, ONLY use the explicit flag:
                    # - emit_x_anchors=True => emit anchor
                    # - emit_x_anchors=False => no anchor
                    #
                    # In legacy mode (no explicit flags set), use measurement_basis:
                    # After swap, X ancillas measure Z-type, so first-round OK if state is |0⟩
                    # Before swap, X ancillas measure X-type, so first-round OK if state is |+⟩
                    if getattr(self, '_explicit_anchor_mode', False):
                        should_emit = not self._skip_first_round and self._emit_x_anchors
                    else:
                        should_emit = (
                            not self._skip_first_round and (
                                self._emit_x_anchors or
                                self.measurement_basis == x_effective_basis
                            )
                        )
                    if should_emit:
                        coord = self._get_stab_coord("x", s_idx)
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                elif not self._skip_first_round:
                    # Only emit time-like detector if not skipping first round
                    # (After projection, first round should skip detectors since
                    # projection outcomes are random - frame is established but
                    # first comparison would be non-deterministic)
                    meas_indices = self.ctx.get_x_detector_measurements(
                        self.block_name, s_idx, cur_meas
                    )
                    coord = self._get_stab_coord("x", s_idx)
                    self.ctx.emit_detector(circuit, meas_indices, coord)
                    # Consume the transforms after emitting
                    self.ctx.consume_x_transforms(self.block_name, s_idx)
                
                self._last_x_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(self.block_name, "x", s_idx, cur_meas)
        else:
            for s_idx in range(self._n_x):
                cur_meas = x_meas_start + s_idx
                self._last_x_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(self.block_name, "x", s_idx, cur_meas)
        
        # Emit detectors for Z stabilizers
        if emit_detectors:
            for s_idx in range(self._n_z):
                cur_meas = z_meas_start + s_idx
                prev_meas = self._last_z_meas[s_idx]
                
                # Check for pending transforms (from CNOT gates)
                has_transforms = self.ctx.has_pending_transforms(self.block_name, "z", s_idx)
                
                if prev_meas is None and not has_transforms:
                    # First round: emit anchor detector if appropriate
                    #
                    # In explicit_anchor_mode, ONLY use the explicit flag:
                    # - emit_z_anchors=True => emit anchor
                    # - emit_z_anchors=False => no anchor
                    #
                    # In legacy mode (no explicit flags set), use measurement_basis:
                    # After swap, Z ancillas measure X-type, so first-round OK if state is |+⟩
                    # Before swap, Z ancillas measure Z-type, so first-round OK if state is |0⟩
                    if getattr(self, '_explicit_anchor_mode', False):
                        should_emit = not self._skip_first_round and self._emit_z_anchors
                    else:
                        should_emit = (
                            not self._skip_first_round and (
                                self._emit_z_anchors or
                                self.measurement_basis == z_effective_basis
                            )
                        )
                    if should_emit:
                        coord = self._get_stab_coord("z", s_idx)
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                elif not self._skip_first_round:
                    # Only emit time-like detector if not skipping first round
                    # (After projection, first round should skip detectors since
                    # projection outcomes are random - frame is established but
                    # first comparison would be non-deterministic)
                    # Use transform-aware detector measurements
                    meas_indices = self.ctx.get_z_detector_measurements(
                        self.block_name, s_idx, cur_meas
                    )
                    coord = self._get_stab_coord("z", s_idx)
                    self.ctx.emit_detector(circuit, meas_indices, coord)
                    # Consume the transforms after emitting
                    self.ctx.consume_z_transforms(self.block_name, s_idx)
                
                self._last_z_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(self.block_name, "z", s_idx, cur_meas)
        else:
            for s_idx in range(self._n_z):
                cur_meas = z_meas_start + s_idx
                self._last_z_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(self.block_name, "z", s_idx, cur_meas)
        
        # Add final TICK after measurements to separate from next round
        circuit.append("TICK")

    def _emit_x_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """Emit X stabilizer measurements.
        
        After Hadamard (when _stabilizer_swapped=True), the X ancillas measure
        what are now Z stabilizers, so we use a Z-style circuit (no H gates).
        """
        if self._hx is None or self._n_x == 0:
            return
        
        x_anc = self.x_ancillas
        
        if self._stabilizer_swapped:
            # After Hadamard: X ancillas measure Z-type stabilizers
            # Use Z syndrome circuit: CNOT - M (no H gates)
            # But we still use _hx matrix because that describes the qubit support
            if self._use_geometric_x():
                self._emit_geometric_cnots(circuit, "x")
            else:
                # Use is_x_type=False to use CZ-like CNOT pattern (control on ancilla)
                self._emit_graph_coloring_cnots(circuit, self._hx, self.data_qubits, x_anc, is_x_type=False)
            
            circuit.append("TICK")
        else:
            # Normal: X ancillas measure X-type stabilizers
            # Use X syndrome circuit: H - CNOT - H - M
            circuit.append("H", x_anc)
            circuit.append("TICK")
            
            if self._use_geometric_x():
                self._emit_geometric_cnots(circuit, "x")
            else:
                self._emit_graph_coloring_cnots(circuit, self._hx, self.data_qubits, x_anc, is_x_type=True)
            
            circuit.append("TICK")
            circuit.append("H", x_anc)
            circuit.append("TICK")
        
        # Measure X ancillas
        meas_start = self.ctx.add_measurement(self._n_x)
        circuit.append("MR", x_anc)
        
        # Time-like detectors with proper first-round logic
        # After swap: X ancillas measure Z, so first-round OK if measurement_basis == "Z"
        # Before swap: X ancillas measure X, so first-round OK if measurement_basis == "X"
        effective_first_round_basis = "Z" if self._stabilizer_swapped else "X"
        
        if emit_detectors:
            for s_idx in range(self._n_x):
                cur_meas = meas_start + s_idx
                prev_meas = self._last_x_meas[s_idx]
                
                # Check for pending transforms (from CZ/CNOT gates)
                has_transforms = self.ctx.has_pending_transforms(self.block_name, "x", s_idx)
                
                if prev_meas is None and not has_transforms:
                    # First round: emit anchor detector if appropriate
                    #
                    # In explicit_anchor_mode, ONLY use the explicit flag:
                    # - emit_x_anchors=True => emit anchor
                    # - emit_x_anchors=False => no anchor
                    #
                    # In legacy mode (no explicit flags set), use measurement_basis:
                    # - Before swap: X ancillas measure X → first-round OK if state is |+⟩ (measurement_basis=="X")
                    # - After swap: X ancillas measure Z → first-round OK if state is |0⟩ (measurement_basis=="Z")
                    if getattr(self, '_explicit_anchor_mode', False):
                        should_emit = not self._skip_first_round and self._emit_x_anchors
                    else:
                        should_emit = (
                            not self._skip_first_round and (
                                self._emit_x_anchors or  # Explicit anchor request
                                (self._ft_config.first_round_x_detectors and
                                 self.measurement_basis == effective_first_round_basis)
                            )
                        )
                    if should_emit:
                        coord = self._get_stab_coord("x", s_idx)
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                else:
                    # Use transform-aware detector measurements
                    meas_indices = self.ctx.get_x_detector_measurements(
                        self.block_name, s_idx, cur_meas
                    )
                    coord = self._get_stab_coord("x", s_idx)
                    self.ctx.emit_detector(circuit, meas_indices, coord)
                    # Consume the transforms after emitting
                    self.ctx.consume_x_transforms(self.block_name, s_idx)
                
                self._last_x_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "x", s_idx, cur_meas
                )
        else:
            for s_idx in range(self._n_x):
                cur_meas = meas_start + s_idx
                self._last_x_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "x", s_idx, cur_meas
                )
    
    def _emit_z_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """Emit Z stabilizer measurements.
        
        After Hadamard (when _stabilizer_swapped=True), the Z ancillas measure
        what are now X stabilizers, so we use an X-style circuit (with H gates).
        """
        if self._hz is None or self._n_z == 0:
            return
        
        z_anc = self.z_ancillas
        
        if self._stabilizer_swapped:
            # After Hadamard: Z ancillas measure X-type stabilizers  
            # Use X syndrome circuit: H - CNOT - H - M
            circuit.append("H", z_anc)
            circuit.append("TICK")
            
            if self._use_geometric_z():
                self._emit_geometric_cnots(circuit, "z")
            else:
                # Use is_x_type=True to use X-style CNOT pattern (control on data)
                self._emit_graph_coloring_cnots(circuit, self._hz, self.data_qubits, z_anc, is_x_type=True)
            
            circuit.append("TICK")
            circuit.append("H", z_anc)
            circuit.append("TICK")
        else:
            # Normal: Z ancillas measure Z-type stabilizers
            # Use Z syndrome circuit: CNOT - M (no H gates)
            if self._use_geometric_z():
                self._emit_geometric_cnots(circuit, "z")
            else:
                self._emit_graph_coloring_cnots(circuit, self._hz, self.data_qubits, z_anc, is_x_type=False)
            
            circuit.append("TICK")
        
        # Measure Z ancillas
        meas_start = self.ctx.add_measurement(self._n_z)
        circuit.append("MR", z_anc)
        
        # Time-like detectors with proper first-round logic
        # After swap: Z ancillas measure X, so first-round OK if measurement_basis == "X"
        # Before swap: Z ancillas measure Z, so first-round OK if measurement_basis == "Z"
        effective_first_round_basis = "X" if self._stabilizer_swapped else "Z"
        
        if emit_detectors:
            for s_idx in range(self._n_z):
                cur_meas = meas_start + s_idx
                prev_meas = self._last_z_meas[s_idx]
                
                # Check for pending transforms (from CNOT gates)
                has_transforms = self.ctx.has_pending_transforms(self.block_name, "z", s_idx)
                
                if prev_meas is None and not has_transforms:
                    # First round: emit anchor detector if appropriate
                    #
                    # In explicit_anchor_mode, ONLY use the explicit flag:
                    # - emit_z_anchors=True => emit anchor
                    # - emit_z_anchors=False => no anchor
                    #
                    # In legacy mode (no explicit flags set), use measurement_basis:
                    # - Before swap: Z ancillas measure Z → first-round OK if state is |0⟩ (measurement_basis=="Z")
                    # - After swap: Z ancillas measure X → first-round OK if state is |+⟩ (measurement_basis=="X")
                    if getattr(self, '_explicit_anchor_mode', False):
                        should_emit = not self._skip_first_round and self._emit_z_anchors
                    else:
                        should_emit = (
                            not self._skip_first_round and (
                                self._emit_z_anchors or  # Explicit anchor request
                                (self._ft_config.first_round_z_detectors and
                                 self.measurement_basis == effective_first_round_basis)
                            )
                        )
                    if should_emit:
                        coord = self._get_stab_coord("z", s_idx)
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                else:
                    # Use transform-aware detector measurements
                    meas_indices = self.ctx.get_z_detector_measurements(
                        self.block_name, s_idx, cur_meas
                    )
                    coord = self._get_stab_coord("z", s_idx)
                    self.ctx.emit_detector(circuit, meas_indices, coord)
                    # Consume the transforms after emitting
                    self.ctx.consume_z_transforms(self.block_name, s_idx)
                
                self._last_z_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "z", s_idx, cur_meas
                )
        else:
            for s_idx in range(self._n_z):
                cur_meas = meas_start + s_idx
                self._last_z_meas[s_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "z", s_idx, cur_meas
                )
    
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

    def _use_geometric_x(self) -> bool:
        """Check if geometric scheduling should be used for X stabilizers."""
        mode = self._ft_config.schedule_mode
        if mode == ScheduleMode.GRAPH_COLORING:
            return False
        
        if not (
            self._x_schedule is not None and
            self._data_coords and
            len(self._x_stab_coords) == self._n_x
        ):
            return False
        
        return self._validate_geometric_schedule(
            self._x_schedule, self._x_stab_coords, self._hx
        )
    
    def _use_geometric_z(self) -> bool:
        """Check if geometric scheduling should be used for Z stabilizers."""
        mode = self._ft_config.schedule_mode
        if mode == ScheduleMode.GRAPH_COLORING:
            return False
        
        if not (
            self._z_schedule is not None and
            self._data_coords and
            len(self._z_stab_coords) == self._n_z
        ):
            return False
        
        return self._validate_geometric_schedule(
            self._z_schedule, self._z_stab_coords, self._hz
        )
    
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
        """Emit controlled gates using geometric scheduling.
        
        Gate Choice for X-Type Stabilizers
        -----------------------------------
        The gate type depends on self._x_stabilizer_mode:
        
        - "cz" (default): Use CZ gates. CZ is symmetric and measures X parity
          correctly with H-CZ-H circuit. Good for memory experiments.
          
        - "cx": Use CNOT[ancilla → data]. Required for teleportation gadgets to
          match ground truth and ensure deterministic X anchor detectors.
          
        The difference is in backward Pauli propagation:
        - CZ: X_ancilla → X_ancilla ⊗ Z_data (Z_data on |0⟩ prep = random)
        - CX: X_ancilla → X_ancilla (stays local, deterministic on |0⟩ prep)
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
        
        # Choose gate based on x_stabilizer_mode for X stabilizers
        if is_x_type:
            x_gate = "CZ" if self._x_stabilizer_mode == "cz" else "CNOT"
        
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
                        # Use x_stabilizer_mode to determine gate (CZ or CNOT)
                        circuit.append(x_gate, [anc, dq])
                    else:
                        circuit.append("CNOT", [dq, anc])

    def _emit_graph_coloring_cnots(
        self,
        circuit: stim.Circuit,
        stab_matrix: np.ndarray,
        data_qubits: List[int],
        ancilla_qubits: List[int],
        is_x_type: bool = True,
    ) -> None:
        """Emit controlled gates using greedy graph coloring.
        
        Gate Choice for X-Type Stabilizers
        -----------------------------------
        The gate type depends on self._x_stabilizer_mode:
        
        - "cz" (default): Use CZ gates. CZ is symmetric and measures X parity
          correctly with H-CZ-H circuit. Good for memory experiments.
          
        - "cx": Use CNOT[ancilla → data]. Required for teleportation gadgets to
          match ground truth and ensure deterministic X anchor detectors.
          
        The difference is in backward Pauli propagation:
        - CZ: X_ancilla → X_ancilla ⊗ Z_data (Z_data → H → X_data on |0⟩ = random)
        - CX: X_ancilla → X_ancilla (stays local, deterministic on |0⟩)
        """
        if stab_matrix is None or stab_matrix.size == 0:
            return
        
        n_stabs, n_data = stab_matrix.shape
        
        # Build list of (ancilla, data) pairs for graph coloring
        all_gates: List[Tuple[int, int]] = []
        for s_idx in range(min(n_stabs, len(ancilla_qubits))):
            anc = ancilla_qubits[s_idx]
            for d_idx in range(min(n_data, len(data_qubits))):
                if stab_matrix[s_idx, d_idx]:
                    dq = data_qubits[d_idx]
                    if is_x_type:
                        # Ancilla is control for both CZ and CNOT
                        all_gates.append((anc, dq))
                    else:
                        # Z-type: data controls ancilla
                        all_gates.append((dq, anc))
        
        if not all_gates:
            return
        
        layers = graph_coloring_cnots(all_gates)
        
        # Choose gate type based on is_x_type and x_stabilizer_mode
        if is_x_type:
            # Use x_stabilizer_mode to choose CZ vs CNOT
            gate_name = "CZ" if self._x_stabilizer_mode == "cz" else "CNOT"
        else:
            # Z-type always uses CNOT (data controls ancilla)
            gate_name = "CNOT"
        
        for layer_idx, layer in enumerate(layers):
            if layer_idx > 0:
                circuit.append("TICK")
            for ctrl, tgt in layer:
                circuit.append(gate_name, [ctrl, tgt])
    
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
