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
    ):
        super().__init__(code, ctx, block_name, data_offset, ancilla_offset, measurement_basis)
        
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
        
        # Flag to track if stabilizer types have been swapped (after Hadamard)
        # When True, physical X ancillas measure logical Z, and vice versa
        self._stabilizer_swapped: bool = False
        
        # Metacheck support for 4D/5D codes (single-shot error correction)
        # Use code's config setting, but can be overridden by parameter
        self._enable_metachecks = enable_metachecks or self._ft_config.enable_metachecks
        _meta_x_raw = getattr(code, 'meta_x', None)
        _meta_z_raw = getattr(code, 'meta_z', None)
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
        # Try code's hook method first (allows code-specific customization)
        if hasattr(self.code, f'get_{basis}_stabilizer_coords'):
            coords = getattr(self.code, f'get_{basis}_stabilizer_coords')()
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
        # Try code's hook method first
        if hasattr(self.code, 'get_stabilizer_schedule'):
            schedule = self.code.get_stabilizer_schedule(basis)
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
    ) -> None:
        """
        Prepare a logical eigenstate.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        state : str
            "0" for |0⟩_L, "1" for |1⟩_L, "+" for |+⟩_L, "-" for |-⟩_L.
        logical_idx : int
            Which logical qubit.
        """
        # Get code interface
        code = self.code
        
        # Determine which qubits need preparation based on state
        if state in ("0", "1"):
            # Z-basis eigenstate
            # |0⟩_L: prepare all data in |0⟩ (already done by reset)
            # |1⟩_L: apply logical X
            if state == "1" and hasattr(code, 'logical_x_support'):
                support = code.logical_x_support(logical_idx)
                for q in support:
                    circuit.append("X", [self.data_offset + q])
        
        elif state in ("+", "-"):
            # X-basis eigenstate
            # For CSS codes: |+⟩_L = H^⊗n|0⟩_L (uniform superposition over Z code space)
            # Apply H to ALL data qubits to prepare |+⟩^⊗n which is +1 eigenstate of all X stabilizers
            # For |-⟩_L, additionally apply Z to the logical X support to get the -1 eigenvalue
            if state == "-" and hasattr(code, 'logical_x_support'):
                support = code.logical_x_support(logical_idx)
                for q in support:
                    circuit.append("Z", [self.data_offset + q])
            # Apply H to all data qubits
            circuit.append("H", self.data_qubits)
        
        circuit.append("TICK")
    
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
    
    def emit_round(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_detectors: bool = True,
        emit_metachecks: bool = False,
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
        """
        # Track measurement indices for metachecks
        x_meas_start = self.ctx.measurement_index
        z_meas_start = x_meas_start + self._n_x  # Z measurements follow X
        
        # Use interleaved scheduling for BOTH when possible (surface codes, etc.)
        if stab_type == StabilizerBasis.BOTH and self._can_interleave():
            self._emit_interleaved_round(circuit, emit_detectors)
        else:
            # Sequential fallback for single-type or non-geometric codes
            if stab_type in (StabilizerBasis.X, StabilizerBasis.BOTH):
                self._emit_x_round(circuit, emit_detectors)
            
            if stab_type in (StabilizerBasis.Z, StabilizerBasis.BOTH):
                self._emit_z_round(circuit, emit_detectors)
        
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
            # Z ancillas measure X-type → H-CNOT(anc→data)-H-M
            # X ancillas measure Z-type → CNOT(data→anc)-M
            
            # Step 1: Prepare Z ancillas with H (they now measure X-type)
            circuit.append("H", z_anc)
            circuit.append("TICK")
            
            # Step 2: Interleaved CNOT phases - swapped control/target
            for phase_idx, ((dx_x, dy_x), (dx_z, dy_z)) in enumerate(
                zip(self._x_schedule, self._z_schedule)
            ):
                if phase_idx > 0:
                    circuit.append("TICK")
                
                # X CNOTs: now measuring Z-type → data controls ancilla
                for s_idx, (sx, sy) in enumerate(self._x_stab_coords):
                    if s_idx >= len(x_anc):
                        continue
                    anc = x_anc[s_idx]
                    nbr = (float(sx) + dx_x, float(sy) + dy_x)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CNOT", [dq, anc])  # data controls ancilla
                
                # Z CNOTs: now measuring X-type → ancilla controls data
                for s_idx, (sx, sy) in enumerate(self._z_stab_coords):
                    if s_idx >= len(z_anc):
                        continue
                    anc = z_anc[s_idx]
                    nbr = (float(sx) + dx_z, float(sy) + dy_z)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CNOT", [anc, dq])  # ancilla controls data
            
            circuit.append("TICK")
            
            # Step 3: Final H on Z ancillas (they measure X-type)
            circuit.append("H", z_anc)
            circuit.append("TICK")
            
        else:
            # Normal mode: standard surface code interleaving
            # Step 1: Prepare X ancillas with H
            circuit.append("H", x_anc)
            circuit.append("TICK")
            
            # Step 2: Interleaved CNOT phases - X and Z in parallel
            for phase_idx, ((dx_x, dy_x), (dx_z, dy_z)) in enumerate(
                zip(self._x_schedule, self._z_schedule)
            ):
                if phase_idx > 0:
                    circuit.append("TICK")
                
                # X CNOTs for this phase
                # For X stabilizers: H-CNOT-H pattern means ancilla is control
                # (H converts Z-basis measurement to X-basis parity check)
                for s_idx, (sx, sy) in enumerate(self._x_stab_coords):
                    if s_idx >= len(x_anc):
                        continue
                    anc = x_anc[s_idx]
                    nbr = (float(sx) + dx_x, float(sy) + dy_x)
                    dq = self._lookup_data_qubit(nbr)
                    if dq is not None:
                        circuit.append("CNOT", [anc, dq])  # ancilla controls data
                
                # Z CNOTs for this phase (SAME TICK - parallel with X)
                # For Z stabilizers: data qubits are control (standard parity check)
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
                    # First round: only emit if measurement_basis matches effective type
                    # After swap, X ancillas measure Z-type, so first-round OK if state is |0⟩
                    # Before swap, X ancillas measure X-type, so first-round OK if state is |+⟩
                    should_emit = not self._skip_first_round and self.measurement_basis == x_effective_basis
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
                    # First round: only emit if measurement_basis matches effective type
                    # After swap, Z ancillas measure X-type, so first-round OK if state is |+⟩
                    # Before swap, Z ancillas measure Z-type, so first-round OK if state is |0⟩
                    should_emit = not self._skip_first_round and self.measurement_basis == z_effective_basis
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
                    # First round: only create detector if appropriate
                    # 
                    # For CSS codes, first-round detectors can only be emitted if the
                    # measurement result is deterministic. This depends on:
                    # 1. The prepared state (|0⟩ is Z eigenstate, |+⟩ is X eigenstate)
                    # 2. What type of parity the ancilla actually measures
                    #
                    # After Hadamard (_stabilizer_swapped=True), X ancillas use Z-type
                    # circuit (data→anc) so they measure Z parity. Z parity of |+⟩ is
                    # random, so X first-round should NOT be emitted after H.
                    #
                    # The effective_first_round_basis accounts for the swap:
                    # - Before swap: X ancillas measure X → first-round OK if state is |+⟩ (measurement_basis=="X")
                    # - After swap: X ancillas measure Z → first-round OK if state is |0⟩ (measurement_basis=="Z")
                    should_emit = (
                        not self._skip_first_round and
                        self._ft_config.first_round_x_detectors and
                        self.measurement_basis == effective_first_round_basis
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
                    # First round: only create detector if appropriate
                    # 
                    # Similar to X stabilizers, we need to check what the Z ancillas
                    # actually measure after potential swap:
                    # - Before swap: Z ancillas measure Z → first-round OK if state is |0⟩ (measurement_basis=="Z")
                    # - After swap: Z ancillas measure X → first-round OK if state is |+⟩ (measurement_basis=="X")
                    should_emit = (
                        not self._skip_first_round and
                        self._ft_config.first_round_z_detectors and
                        self.measurement_basis == effective_first_round_basis
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
        """Emit CNOTs using geometric scheduling."""
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
                        circuit.append("CNOT", [anc, dq])
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
        """Emit CNOTs using greedy graph coloring."""
        if stab_matrix is None or stab_matrix.size == 0:
            return
        
        n_stabs, n_data = stab_matrix.shape
        
        all_cnots: List[Tuple[int, int]] = []
        for s_idx in range(min(n_stabs, len(ancilla_qubits))):
            anc = ancilla_qubits[s_idx]
            for d_idx in range(min(n_data, len(data_qubits))):
                if stab_matrix[s_idx, d_idx]:
                    dq = data_qubits[d_idx]
                    if is_x_type:
                        all_cnots.append((anc, dq))
                    else:
                        all_cnots.append((dq, anc))
        
        if not all_cnots:
            return
        
        layers = graph_coloring_cnots(all_cnots)
        
        for layer_idx, layer in enumerate(layers):
            if layer_idx > 0:
                circuit.append("TICK")
            for ctrl, tgt in layer:
                circuit.append("CNOT", [ctrl, tgt])
    
    def _get_stab_coord(self, stab_type: str, s_idx: int) -> Tuple[float, float, float]:
        """Get detector coordinate for a stabilizer."""
        if stab_type == "x":
            coords = self._x_stab_coords
        else:
            coords = self._z_stab_coords
        
        if s_idx < len(coords):
            x, y = coords[s_idx][:2]
            return (float(x), float(y), self.ctx.current_time)
        return (0.0, 0.0, self.ctx.current_time)
    
    def _get_metacheck_coord(self, meta_type: str, meta_idx: int) -> Tuple[float, float, float, float]:
        """Get detector coordinate for a metacheck."""
        if meta_type == "x":
            meta_matrix = self._meta_x
            coords = self._z_stab_coords
        else:
            meta_matrix = self._meta_z
            coords = self._x_stab_coords
        
        if meta_matrix is None or meta_idx >= meta_matrix.shape[0]:
            return (0.0, 0.0, self.ctx.current_time, 1.0)
        
        row = meta_matrix[meta_idx]
        covered_indices = np.where(row)[0]
        
        if len(covered_indices) == 0 or len(coords) == 0:
            return (0.0, 0.0, self.ctx.current_time, 1.0)
        
        x_sum, y_sum, count = 0.0, 0.0, 0
        for s_idx in covered_indices:
            if s_idx < len(coords):
                x_sum += coords[s_idx][0]
                y_sum += coords[s_idx][1]
                count += 1
        
        if count > 0:
            return (x_sum / count, y_sum / count, self.ctx.current_time, 1.0)
        return (0.0, 0.0, self.ctx.current_time, 1.0)
    
    def emit_rounds(
        self,
        circuit: stim.Circuit,
        num_rounds: int,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_metachecks: bool = False,
    ) -> None:
        """Emit multiple stabilizer measurement rounds."""
        for _ in range(num_rounds):
            self.emit_round(circuit, stab_type, emit_detectors=True, emit_metachecks=emit_metachecks)
    
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
