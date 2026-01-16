# src/qectostim/experiments/stabilizer_rounds/xyz_color.py
"""
XYZ color code stabilizer round builder with C_XYZ basis cycling.

This module provides XYZColorCodeStabilizerRoundBuilder for XYZ color codes
that use a single ancilla per face and C_XYZ gates to cycle measurement basis.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

import stim

from .context import DetectorContext
from .base import BaseStabilizerRoundBuilder, StabilizerBasis

if TYPE_CHECKING:
    from qectostim.codes.abstract_code import Code


class XYZColorCodeStabilizerRoundBuilder(BaseStabilizerRoundBuilder):
    """
    Stabilizer round builder for XYZ color codes with C_XYZ basis cycling.
    
    Unlike CSS codes, XYZ color codes use:
    - One ancilla per face (not separate X and Z ancillas)
    - C_XYZ gate on data qubits to cycle measurement basis (X→Y→Z→X)
    - Single detector per face (measuring joint XYZ observable)
    - Color encoding 0-2 (no X/Z distinction since it's a joint measurement)
    
    The C_XYZ gate cycles the measurement basis each round:
    - Round 1: effectively measures X component
    - Round 2: effectively measures Y component  
    - Round 3: effectively measures Z component
    - Round 4: back to X, etc.
    
    For Z-basis memory (|0⟩ initial state), the Z-like rounds give
    deterministic outcomes. However, since the cycling is implicit,
    we use time-like detectors comparing consecutive rounds after
    the first two rounds establish a baseline.
    
    Chromobius compatibility:
    - 4D coordinates: (x, y, t, color) where color ∈ {0, 1, 2}
    - Uses single basis encoding (0-2) since XYZ is a joint measurement
    
    Parameters
    ----------
    code : Code
        An XYZ color code with metadata containing:
        - faces: list of data qubit indices per stabilizer face
        - stab_colors: color (0-2) per face
        - ancilla_coords: (x, y) coordinates for ancilla qubits
    ctx : DetectorContext
        Context for tracking measurements and detectors.
    block_name : str
        Name for this code block.
    data_offset : int
        Offset for data qubit indices.
    ancilla_offset : int, optional
        Offset for ancilla qubit indices.
    measurement_basis : str
        Memory basis ("Z" or "X").
    """
    
    def __init__(
        self,
        code: "Code",
        ctx: DetectorContext,
        block_name: str = "main",
        data_offset: int = 0,
        ancilla_offset: Optional[int] = None,
        measurement_basis: str = "Z",
    ):
        super().__init__(code, ctx, block_name, data_offset, ancilla_offset, measurement_basis)
        
        # Validate XYZ color code metadata
        if "faces" not in self._meta:
            raise ValueError(
                "XYZColorCodeStabilizerRoundBuilder requires code.metadata['faces'] "
                "to be a list of data qubit indices per stabilizer face"
            )
        if "stab_colors" not in self._meta:
            raise ValueError(
                "XYZColorCodeStabilizerRoundBuilder requires code.metadata['stab_colors'] "
                "to be a list of colors (0=red, 1=green, 2=blue) per face"
            )
        if "ancilla_coords" not in self._meta:
            raise ValueError(
                "XYZColorCodeStabilizerRoundBuilder requires code.metadata['ancilla_coords'] "
                "to be a list of (x, y) coordinates for ancilla qubits"
            )
        
        # Cache XYZ-specific info
        self._faces: List[List[int]] = self._meta["faces"]
        self._stab_colors: List[int] = self._meta["stab_colors"]
        self._ancilla_coords: List[Tuple[float, float]] = self._meta["ancilla_coords"]
        self._n_faces = len(self._faces)
        
        # Build CNOT schedule (data → ancilla)
        self._cnot_schedule = self._build_cnot_schedule()
        
        # Track last measurement for each face (for time-like detectors)
        self._last_face_meas: List[Optional[int]] = [None] * self._n_faces
        
        # Track round count for first-round detector logic
        self._round_count = 0
    
    @property
    def ancilla_qubits(self) -> List[int]:
        """Global indices of ancilla qubits (one per face)."""
        return list(range(self.ancilla_offset, self.ancilla_offset + self._n_faces))
    
    @property
    def total_qubits(self) -> int:
        """Total qubits used by this block (data + ancillas)."""
        return self.code.n + self._n_faces
    
    def _build_cnot_schedule(self) -> List[List[Tuple[int, int]]]:
        """
        Build CNOT schedule in layers to avoid qubit conflicts.
        
        Uses greedy graph coloring to minimize circuit depth while ensuring
        no two CNOTs in the same layer share a data qubit or ancilla.
        
        Returns
        -------
        List[List[Tuple[int, int]]]
            List of layers, each layer is list of (data_q, face_idx) pairs.
        """
        # Collect all CNOTs needed: (data_qubit, face_index)
        all_cnots = []
        for face_idx, face in enumerate(self._faces):
            for data_q in face:
                all_cnots.append((data_q, face_idx))
        
        # Greedy scheduling - assign to first non-conflicting layer
        layers: List[List[Tuple[int, int]]] = []
        for data_q, face_idx in all_cnots:
            placed = False
            for layer in layers:
                # Check if this CNOT conflicts with any in this layer
                conflict = False
                for d, f in layer:
                    if d == data_q or f == face_idx:
                        conflict = True
                        break
                if not conflict:
                    layer.append((data_q, face_idx))
                    placed = True
                    break
            if not placed:
                layers.append([(data_q, face_idx)])
        
        return layers
    
    def emit_qubit_coords(self, circuit: stim.Circuit) -> None:
        """Emit QUBIT_COORDS for all data and ancilla qubits."""
        # Data qubits
        for local_idx, coord in enumerate(self._data_coords):
            if len(coord) >= 2:
                global_idx = self.data_offset + local_idx
                circuit.append("QUBIT_COORDS", [global_idx], [float(coord[0]), float(coord[1])])
        
        # Ancilla qubits (at face centers)
        for local_idx, (x, y) in enumerate(self._ancilla_coords):
            global_idx = self.ancilla_offset + local_idx
            circuit.append("QUBIT_COORDS", [global_idx], [float(x), float(y)])
    
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
        
        For XYZ color codes, preparation is similar to CSS codes:
        - |0⟩_L: all data qubits in |0⟩ (reset)
        - |+⟩_L: apply H to all data qubits
        """
        if state in ("0", "1"):
            # Z-basis eigenstate
            if state == "1":
                logical_support = self._meta.get("logical_x_support", [])
                for q in logical_support:
                    circuit.append("X", [self.data_offset + q])
        
        elif state in ("+", "-"):
            # X-basis eigenstate
            if state == "-":
                logical_support = self._meta.get("logical_x_support", [])
                for q in logical_support:
                    circuit.append("Z", [self.data_offset + q])
            circuit.append("H", self.data_qubits)
        
        circuit.append("TICK")
    
    def emit_round(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_detectors: bool = True,
    ) -> None:
        """
        Emit one XYZ stabilizer measurement round.
        
        Structure:
        1. C_XYZ on all data qubits (basis rotation)
        2. CNOT layers (data → ancilla)
        3. MR on ancillas
        4. Time-like detectors comparing with previous round
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        stab_type : StabilizerBasis
            Ignored for XYZ codes (always measures joint XYZ).
        emit_detectors : bool
            Whether to emit time-like detectors.
        """
        data = self.data_qubits
        anc = self.ancilla_qubits
        
        # Step 1: Apply C_XYZ to all data qubits
        circuit.append("TICK")
        circuit.append("C_XYZ", data)
        circuit.append("TICK")
        
        # Step 2: CNOT layers (data controls ancilla)
        for layer_idx, layer in enumerate(self._cnot_schedule):
            if layer_idx > 0:
                circuit.append("TICK")
            for data_q, face_idx in layer:
                global_data = self.data_offset + data_q
                global_anc = self.ancilla_offset + face_idx
                circuit.append("CX", [global_data, global_anc])
        
        circuit.append("TICK")
        
        # Step 3: Measure and reset ancillas
        meas_start = self.ctx.add_measurement(self._n_faces)
        circuit.append("MR", anc)
        
        # Step 4: Time-like detectors
        if emit_detectors:
            for face_idx in range(self._n_faces):
                cur_meas = meas_start + face_idx
                prev_meas = self._last_face_meas[face_idx]
                
                coord = self._get_face_coord(face_idx)
                
                if prev_meas is None:
                    # First round - skip detector (no baseline yet)
                    # XYZ cycling means we need at least 2 rounds for comparison
                    pass
                else:
                    # Compare with previous round
                    self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
                
                self._last_face_meas[face_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "xyz", face_idx, cur_meas
                )
        else:
            for face_idx in range(self._n_faces):
                cur_meas = meas_start + face_idx
                self._last_face_meas[face_idx] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, "xyz", face_idx, cur_meas
                )
        
        # Advance time
        self.ctx.advance_time()
        self._emit_shift_coords(circuit)
        self._round_count += 1
    
    def _get_face_coord(self, face_idx: int) -> Tuple[float, float, float, float]:
        """
        Get 4D detector coordinate for a face.
        
        Returns (x, y, t, color) where color ∈ {0, 1, 2}.
        """
        if face_idx < len(self._ancilla_coords):
            x, y = self._ancilla_coords[face_idx]
        else:
            x, y = 0.0, 0.0
        
        color = float(self._stab_colors[face_idx]) if face_idx < len(self._stab_colors) else 0.0
        
        return (float(x), float(y), self.ctx.current_time, color)
    
    def emit_final_measurement(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
        logical_idx: int = 0,
    ) -> List[int]:
        """
        Emit final data qubit measurements with space-like detectors.
        
        For XYZ codes, final detectors compare last syndrome measurement
        with data qubit product for each face.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        basis : str
            Measurement basis ("Z" or "X").
        logical_idx : int
            Which logical qubit to track.
            
        Returns
        -------
        List[int]
            Measurement indices for the logical observable.
        """
        data = self.data_qubits
        n = len(data)
        basis = basis.upper()
        
        # Basis change if needed
        if basis == "X":
            circuit.append("H", data)
        
        # Measure all data qubits
        meas_start = self.ctx.add_measurement(n)
        circuit.append("M", data)
        
        # Space-like detectors: compare last syndrome with data product
        for face_idx, face in enumerate(self._faces):
            last_meas = self._last_face_meas[face_idx]
            if last_meas is None:
                continue
            
            # Get data measurement indices for this face
            data_idxs = [meas_start + dq for dq in face if dq < n]
            
            if data_idxs:
                recs = data_idxs + [last_meas]
                # Use current time for space-like detectors
                coord = self._get_face_coord(face_idx)
                coord = (coord[0], coord[1], self.ctx.current_time, coord[3])
                self.ctx.emit_detector(circuit, recs, coord)
        
        # Compute logical observable
        if basis == "Z":
            logical_support = self._meta.get("logical_z_support", [])
        else:
            logical_support = self._meta.get("logical_x_support", [])
        
        # Check for placeholder logical (all qubits = likely invalid)
        if logical_support and len(logical_support) >= n:
            logical_support = []
        
        logical_meas = [meas_start + q for q in logical_support if q < n]
        if not logical_meas:
            import warnings
            warnings.warn(
                f"No valid logical {basis} operator found for XYZ color code. "
                f"Observable will track qubit 0 only - decoding may not work correctly.",
                RuntimeWarning,
                stacklevel=2
            )
            logical_meas = [meas_start]
        
        self.ctx.add_observable_measurement(logical_idx, logical_meas)
        
        return logical_meas
    
    def emit_rounds(
        self,
        circuit: stim.Circuit,
        num_rounds: int,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
    ) -> None:
        """Emit multiple XYZ stabilizer measurement rounds."""
        for _ in range(num_rounds):
            self.emit_round(circuit, stab_type, emit_detectors=True)
