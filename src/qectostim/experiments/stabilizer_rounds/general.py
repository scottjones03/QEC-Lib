# src/qectostim/experiments/stabilizer_rounds/general.py
"""
General stabilizer round builder for non-CSS stabilizer codes.

This module provides GeneralStabilizerRoundBuilder which handles stabilizers
with mixed X, Y, and Z components using the symplectic stabilizer_matrix
representation [X|Z].
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import stim

from qectostim.utils.scheduling_core import graph_coloring_cnots

from .context import DetectorContext
from .base import BaseStabilizerRoundBuilder, StabilizerBasis
from .utils import _parse_pauli_support

if TYPE_CHECKING:
    from qectostim.codes.abstract_code import StabilizerCode


class GeneralStabilizerRoundBuilder(BaseStabilizerRoundBuilder):
    """
    Stabilizer round builder for general (non-CSS) stabilizer codes.
    
    Handles stabilizers with mixed X, Y, and Z components using the
    symplectic stabilizer_matrix representation [X|Z]. Each stabilizer
    is measured by entangling an ancilla with data qubits, with
    appropriate basis rotations:
    
    - X component: H - CNOT - H
    - Y component: S† - H - CNOT - H - S
    - Z component: CNOT
    
    Parameters
    ----------
    code : StabilizerCode
        The quantum error correcting code with stabilizer_matrix attribute.
    ctx : DetectorContext
        Context for tracking measurements and detectors.
    block_name : str
        Name for this code block.
    data_offset : int
        Offset for data qubit indices.
    ancilla_offset : int, optional
        Offset for ancilla qubit indices.
    """
    
    def __init__(
        self,
        code: "StabilizerCode",
        ctx: DetectorContext,
        block_name: str = "main",
        data_offset: int = 0,
        ancilla_offset: Optional[int] = None,
        measurement_basis: str = "Z",
        coord_offset: Optional[Tuple[float, ...]] = None,
    ):
        # Pass measurement_basis and coord_offset to parent
        super().__init__(code, ctx, block_name, data_offset, ancilla_offset, measurement_basis, coord_offset=coord_offset)
        
        # Get symplectic stabilizer matrix
        self._stab_mat = getattr(code, 'stabilizer_matrix', None)
        if self._stab_mat is None or self._stab_mat.size == 0:
            raise ValueError("GeneralStabilizerRoundBuilder requires code.stabilizer_matrix")
        
        self._n_stabs = self._stab_mat.shape[0]
        self._n = code.n
        
        # Tracking for measurements
        self._last_stab_meas: List[Optional[int]] = [None] * self._n_stabs
        self._round_number = 0
    
    @property
    def data_qubits(self) -> List[int]:
        """Global indices of data qubits."""
        return list(range(self.data_offset, self.data_offset + self._n))
    
    @property
    def ancilla_qubits(self) -> List[int]:
        """Global indices of ancilla qubits."""
        return list(range(self.ancilla_offset, self.ancilla_offset + self._n_stabs))

    @property
    def all_ancillas(self) -> List[int]:
        """All ancilla qubits — unified pool for non-CSS codes."""
        return self.ancilla_qubits

    @property
    def x_ancillas(self) -> List[int]:
        """For non-CSS codes, all ancillas are reported as 'x_ancillas'.

        This ensures that callers using ``builder.x_ancillas + builder.z_ancillas``
        still get the correct total set of ancillas.
        """
        return self.ancilla_qubits

    @property
    def total_qubits(self) -> int:
        """Total qubits used by this block."""
        return self._n + self._n_stabs
    
    def emit_qubit_coords(self, circuit: stim.Circuit) -> None:
        """Emit QUBIT_COORDS for all qubits."""
        for local_idx, coord in enumerate(self._data_coords):
            if len(coord) >= 2:
                global_idx = self.data_offset + local_idx
                circuit.append("QUBIT_COORDS", [global_idx], [float(coord[0]), float(coord[1])])
    
    def emit_reset_all(self, circuit: stim.Circuit) -> None:
        """Reset all data and ancilla qubits."""
        all_qubits = self.data_qubits + self.ancilla_qubits
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
        
        For non-CSS codes, we prepare based on the logical operator structure.
        """
        code = self.code
        
        # Get logical operators
        logical_x = None
        logical_z = None
        if hasattr(code, 'logical_x') and code.logical_x:
            ops = code.logical_x
            if logical_idx < len(ops):
                logical_x = ops[logical_idx]
        if hasattr(code, 'logical_z') and code.logical_z:
            ops = code.logical_z
            if logical_idx < len(ops):
                logical_z = ops[logical_idx]
        
        if state in ("0", "1"):
            # Z-basis eigenstate - prepare |0⟩_L or |1⟩_L
            if state == "1" and logical_x is not None:
                support = _parse_pauli_support(logical_x, ('X', 'Y', 'Z'), self._n)
                for q in support:
                    circuit.append("X", [self.data_offset + q])
        
        elif state in ("+", "-"):
            # X-basis eigenstate - apply H to all data qubits
            # Then apply Z to logical X support for |-⟩
            circuit.append("H", self.data_qubits)
            if state == "-" and logical_x is not None:
                support = _parse_pauli_support(logical_x, ('X', 'Y', 'Z'), self._n)
                for q in support:
                    circuit.append("Z", [self.data_offset + q])
        
        circuit.append("TICK")
    
    def emit_round(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_detectors: bool = True,
        emit_metachecks: bool = False,
    ) -> None:
        """
        Emit one complete stabilizer measurement round.
        
        For non-CSS codes, all stabilizers are measured together using
        basis rotations appropriate to each Pauli component.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        stab_type : StabilizerBasis
            Ignored for non-CSS codes (always measures all stabilizers).
        emit_detectors : bool
            Whether to emit time-like detectors.
        emit_metachecks : bool
            Ignored for non-CSS codes.
        """
        if self._n_stabs == 0:
            return
        
        anc = self.ancilla_qubits
        
        # Reset ancillas at start of round
        circuit.append("R", anc)
        circuit.append("TICK")
        
        # Apply stabilizer gates with graph-coloring scheduling
        self._emit_general_stabilizer_cnots(circuit)
        
        circuit.append("TICK")
        
        # Measure all ancillas
        meas_start = self.ctx.add_measurement(self._n_stabs)
        circuit.append("MR", anc)
        
        # Time-like detectors - for non-CSS codes, skip first round
        # (initial state is generally not eigenstate of all stabilizers)
        if emit_detectors:
            for s_idx in range(self._n_stabs):
                cur_meas = meas_start + s_idx
                prev_meas = self._last_stab_meas[s_idx]
                
                if prev_meas is not None:
                    # Compare with previous round
                    coord = self._get_stab_coord(s_idx)
                    self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
                # Skip first-round detectors for non-CSS codes
                
                self._last_stab_meas[s_idx] = cur_meas
        else:
            for s_idx in range(self._n_stabs):
                self._last_stab_meas[s_idx] = meas_start + s_idx
        
        # Advance time
        self.ctx.advance_time()
        self._round_number += 1
    
    def _emit_general_stabilizer_cnots(self, circuit: stim.Circuit) -> None:
        """
        Emit CNOTs for general stabilizers with basis rotations.
        
        Handles X, Y, Z components with appropriate pre/post rotations.
        
        CNOT direction convention (matches Stim and CSS builder):
        - X-type component: CNOT(ancilla, data) - ancilla controls, data is target
        - Y-type component: CNOT(ancilla, data) - same as X (Y = iXZ)
        - Z-type component: CNOT(data, ancilla) - data controls, ancilla is target
        """
        stab_mat = self._stab_mat
        data = self.data_qubits
        anc = self.ancilla_qubits
        n = self._n
        
        # Collect operations: (ctrl_idx, tgt_idx, pauli_type)
        all_ops: List[Tuple[int, int, str]] = []
        
        for s_idx in range(self._n_stabs):
            x_part = stab_mat[s_idx, :n]
            z_part = stab_mat[s_idx, n:2*n] if stab_mat.shape[1] >= 2*n else np.zeros(n)
            
            for q in range(n):
                x_bit = x_part[q] if q < len(x_part) else 0
                z_bit = z_part[q] if q < len(z_part) else 0
                
                if x_bit and z_bit:
                    # Y-type: ancilla controls, data is target (like X)
                    all_ops.append((anc[s_idx], data[q], 'Y'))
                elif x_bit:
                    # X-type: ancilla controls, data is target
                    all_ops.append((anc[s_idx], data[q], 'X'))
                elif z_bit:
                    # Z-type: data controls, ancilla is target
                    all_ops.append((data[q], anc[s_idx], 'Z'))
        
        if not all_ops:
            return
        
        # Extract CNOT pairs (ctrl, tgt) and build lookup
        cnot_pairs = [(ctrl, tgt) for ctrl, tgt, _ in all_ops]
        op_pauli = {(ctrl, tgt): p for ctrl, tgt, p in all_ops}
        
        # Schedule into conflict-free layers
        layers = graph_coloring_cnots(cnot_pairs)
        
        # Emit each layer with basis rotations
        for layer_idx, layer_cnots in enumerate(layers):
            if layer_idx > 0:
                circuit.append("TICK")
            
            # Group by Pauli type - identify data qubits for rotations
            x_ops = [(ctrl, tgt) for ctrl, tgt in layer_cnots if op_pauli.get((ctrl, tgt)) == 'X']
            y_ops = [(ctrl, tgt) for ctrl, tgt in layer_cnots if op_pauli.get((ctrl, tgt)) == 'Y']
            z_ops = [(ctrl, tgt) for ctrl, tgt in layer_cnots if op_pauli.get((ctrl, tgt)) == 'Z']
            
            # Pre-rotation for X: H on data qubits (which are TARGETS for X-type)
            if x_ops:
                x_data = list(set(tgt for _, tgt in x_ops))
                circuit.append("H", x_data)
            
            # Pre-rotation for Y: S† then H on data qubits (which are TARGETS for Y-type)
            if y_ops:
                y_data = list(set(tgt for _, tgt in y_ops))
                circuit.append("S_DAG", y_data)
                circuit.append("H", y_data)
            
            # Apply all CNOTs with correct direction per Pauli type
            for ctrl, tgt in layer_cnots:
                circuit.append("CX", [ctrl, tgt])
            
            # Post-rotation for X: H on data qubits
            if x_ops:
                circuit.append("H", x_data)
            
            # Post-rotation for Y: H then S on data qubits
            if y_ops:
                circuit.append("H", y_data)
                circuit.append("S", y_data)
    
    def _get_stab_coord(self, s_idx: int) -> Tuple[float, float, float]:
        """Get detector coordinate for a stabilizer."""
        ox = self._coord_offset[0] if len(self._coord_offset) > 0 else 0.0
        oy = self._coord_offset[1] if len(self._coord_offset) > 1 else 0.0
        # For non-CSS codes, use generic coordinates
        return (ox, float(s_idx) + oy, self.ctx.current_time)
    
    def reset_stabilizer_history(self, swap_xz: bool = False, skip_first_round: bool = False, clear_history: bool = False) -> None:
        """
        Reset the builder's internal stabilizer measurement history.
        
        For non-CSS codes, this clears all stabilizer measurement history.
        """
        # Clear measurement history
        self._last_stab_meas = [None] * self._n_stabs
        self._round_number = 0

    def get_last_measurement_indices(self) -> Dict[str, List[int]]:
        """Return last-round measurement indices.

        For non-CSS codes all stabilizers are measured together.  We report
        them under the ``"X"`` key so that crossing-detector logic (which
        iterates ``"X"`` and ``"Z"`` keys) can find them.

        Returns
        -------
        Dict[str, List[int]]
            ``{"X": [...], "Z": []}`` with all stabilizer measurements in X.
        """
        indices = [m for m in self._last_stab_meas if m is not None]
        return {"X": indices, "Z": []}
    
    def emit_space_like_detectors(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
    ) -> None:
        """
        Emit space-like detectors comparing final data measurements with last stabilizer round.
        
        For non-CSS codes, this is more complex since stabilizers have mixed components.
        """
        n = len(self.data_qubits)
        meas_start = self.ctx.measurement_index - n
        
        stab_mat = self._stab_mat
        basis = basis.upper()
        
        for s_idx in range(self._n_stabs):
            last_meas = self._last_stab_meas[s_idx]
            if last_meas is None:
                continue
            
            x_part = stab_mat[s_idx, :n]
            z_part = stab_mat[s_idx, n:2*n] if stab_mat.shape[1] >= 2*n else np.zeros(n)
            
            support = []
            if basis == "Z":
                for q in range(n):
                    z_bit = z_part[q] if q < len(z_part) else 0
                    x_bit = x_part[q] if q < len(x_part) else 0
                    if z_bit and not x_bit:
                        support.append(q)
            else:
                for q in range(n):
                    z_bit = z_part[q] if q < len(z_part) else 0
                    x_bit = x_part[q] if q < len(x_part) else 0
                    if x_bit and not z_bit:
                        support.append(q)
            
            if support:
                data_idxs = [meas_start + q for q in support]
                recs = data_idxs + [last_meas]
                coord = self._get_stab_coord(s_idx)
                self.ctx.emit_detector(circuit, recs, coord)

    def emit_final_measurement(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
        logical_idx: int = 0,
    ) -> List[int]:
        """Emit final data qubit measurements with space-like detectors."""
        data = self.data_qubits
        n = len(data)
        basis = basis.upper()
        
        # Get logical operator to determine measurement bases
        code = self.code
        if basis == "Z" and hasattr(code, 'logical_z') and code.logical_z:
            logical_op = code.logical_z[logical_idx] if logical_idx < len(code.logical_z) else None
        elif hasattr(code, 'logical_x') and code.logical_x:
            logical_op = code.logical_x[logical_idx] if logical_idx < len(code.logical_x) else None
        else:
            logical_op = None
        
        # Determine per-qubit measurement basis from logical operator
        qubit_basis = {q: 'Z' for q in range(n)}
        logical_support = []
        
        if logical_op is not None:
            support = _parse_pauli_support(logical_op, ('X', 'Y', 'Z'), n)
            logical_support = support
            
            if isinstance(logical_op, str):
                for i, p in enumerate(logical_op):
                    if p in ('X', 'Y', 'Z'):
                        qubit_basis[i] = p
            elif isinstance(logical_op, dict):
                for q, p in logical_op.items():
                    if p in ('X', 'Y', 'Z'):
                        qubit_basis[q] = p
        
        # Apply basis change gates
        x_qubits = [data[q] for q in range(n) if qubit_basis.get(q) == 'X']
        y_qubits = [data[q] for q in range(n) if qubit_basis.get(q) == 'Y']
        
        if x_qubits:
            circuit.append("H", x_qubits)
        for q in y_qubits:
            circuit.append("S_DAG", [q])
            circuit.append("H", [q])
        
        # Measure all data qubits
        meas_start = self.ctx.add_measurement(n)
        circuit.append("M", data)
        
        # Space-like detectors (simplified for non-CSS)
        stab_mat = self._stab_mat
        for s_idx in range(self._n_stabs):
            last_meas = self._last_stab_meas[s_idx]
            if last_meas is None:
                continue
            
            x_part = stab_mat[s_idx, :n]
            z_part = stab_mat[s_idx, n:2*n] if stab_mat.shape[1] >= 2*n else np.zeros(n)
            
            support = []
            can_measure = True
            
            for q in range(n):
                x_bit = x_part[q] if q < len(x_part) else 0
                z_bit = z_part[q] if q < len(z_part) else 0
                meas_b = qubit_basis.get(q, 'Z')
                
                if x_bit and z_bit:
                    if meas_b != 'Y':
                        can_measure = False
                        break
                    support.append(q)
                elif x_bit:
                    if meas_b not in ('X', 'Y'):
                        can_measure = False
                        break
                    support.append(q)
                elif z_bit:
                    if meas_b not in ('Z', 'Y'):
                        can_measure = False
                        break
                    support.append(q)
            
            if not can_measure:
                continue
            
            data_idxs = [meas_start + q for q in support]
            recs = data_idxs + [last_meas]
            
            if recs:
                coord = self._get_stab_coord(s_idx)
                self.ctx.emit_detector(circuit, recs, coord)
        
        # Logical observable
        if logical_support:
            logical_meas = [meas_start + q for q in logical_support if q < n]
        else:
            import warnings
            warnings.warn(
                f"No valid logical operator found for {type(code).__name__}. "
                f"Observable will track qubit 0 only - decoding may not work correctly.",
                RuntimeWarning,
                stacklevel=2
            )
            logical_meas = [meas_start]
        
        self.ctx.add_observable_measurement(logical_idx, logical_meas)
        
        return logical_meas
