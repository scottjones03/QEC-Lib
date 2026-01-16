# src/qectostim/experiments/stabilizer_rounds/concatenated.py
"""
Flat concatenated code stabilizer round builder.

This module provides FlatConcatenatedStabilizerRoundBuilder (formerly 
ConcatenatedStabilizerRoundBuilder) which handles stabilizer measurements 
for concatenated CSS codes with support for block-grouped detector emission 
enabling hierarchical decoding.

IMPORTANT: FLAT CONCATENATION MODEL
===================================
This module implements a SIMPLIFIED "flat" model of concatenated code syndrome
extraction where ALL stabilizers (inner and lifted outer) are measured using 
direct physical syndrome extraction circuits with physical CNOT gates.

In a true hierarchical concatenated code:
- Outer stabilizer measurements would involve LOGICAL CNOTs between inner code blocks
- Inner codes would run their own error correction during outer operations
- Error suppression would cascade from inner to outer levels

This flat approach:
✓ Correctly implements the stabilizer group measurements
✓ Produces detector structure compatible with hierarchical decoders
✓ Is useful for code structure validation and testing
✗ Does NOT use fault-tolerant logical gadgets for outer operations
✗ Does NOT capture the error suppression hierarchy of true concatenation

For proper hierarchical concatenation, the outer stabilizer measurements should
use LogicalCNOT gadgets (transversal or surgery) operating on inner code blocks.
See HierarchicalConcatenatedMemoryExperiment for this approach.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import stim

from qectostim.utils.scheduling_core import graph_coloring_cnots

from .context import DetectorContext
from .base import BaseStabilizerRoundBuilder, StabilizerBasis

if TYPE_CHECKING:
    from qectostim.codes.composite.concatenated import ConcatenatedCode


class FlatConcatenatedStabilizerRoundBuilder(BaseStabilizerRoundBuilder):
    """
    Flat stabilizer round builder for concatenated CSS codes.
    
    IMPORTANT: FLAT CONCATENATION MODEL
    ===================================
    This builder implements a SIMPLIFIED "flat" model where ALL stabilizers
    (inner and lifted outer) are measured using direct physical syndrome 
    extraction circuits. This does NOT use fault-tolerant logical gadgets
    for outer code operations.
    
    In this flat model:
    - Outer stabilizers are measured via physical CNOTs to ancilla qubits
    - No logical operations are used - everything is at the physical level
    - The concatenated code is treated as one large stabilizer code
    
    For proper hierarchical concatenation where outer stabilizers are measured
    using logical CNOTs on inner code blocks, see the hierarchical experiment
    infrastructure (HierarchicalConcatenatedMemoryExperiment).
    
    Features
    --------
    - Supports both standard (round-by-round) and block-contiguous detector emission
    - Tracks inner and outer stabilizer measurements separately
    - Supports multi-level (deeply nested) concatenation
    - Compatible with FT gadget experiments (non-contiguous mode)
    - Block-contiguous mode for hierarchical decoder compatibility
    
    The detector layout when using block_contiguous mode is:
        [Block 0 detectors] [Block 1 detectors] ... [Block N-1 detectors] [Outer detectors]
    
    This matches the structure expected by ConcatenatedDecoder's decode_batch() method.
    
    Parameters
    ----------
    code : ConcatenatedCode
        A concatenated CSS code.
    ctx : DetectorContext
        Context for tracking measurements and detectors.
    block_name : str
        Name for this code block.
    data_offset : int
        Offset for data qubit indices.
    ancilla_offset : int, optional
        Offset for ancilla qubit indices.
    measurement_basis : str
        The basis for memory experiment ("Z" or "X").
    block_contiguous : bool
        If True, defer detector emission and emit in block-contiguous order
        when emit_deferred_detectors() is called. Default False for
        compatibility with FT gadget experiments.
        
    See Also
    --------
    HierarchicalConcatenatedMemoryExperiment : Proper hierarchical implementation
        using fault-tolerant logical gadgets for outer code operations.
    """
    
    def __init__(
        self,
        code: "ConcatenatedCode",
        ctx: DetectorContext,
        block_name: str = "main",
        data_offset: int = 0,
        ancilla_offset: Optional[int] = None,
        measurement_basis: str = "Z",
        block_contiguous: bool = False,
    ):
        super().__init__(code, ctx, block_name, data_offset, ancilla_offset, measurement_basis)
        
        # Validate code is concatenated
        from qectostim.codes.composite.concatenated import ConcatenatedCode
        if not isinstance(code, ConcatenatedCode):
            raise TypeError(
                f"FlatConcatenatedStabilizerRoundBuilder requires a ConcatenatedCode, "
                f"got {type(code).__name__}"
            )
        
        # Store concatenation structure
        if hasattr(code, '_inner_code') and hasattr(code, '_outer_code'):
            self._inner_code = code._inner_code
            self._outer_code = code._outer_code
        elif hasattr(code, 'inner') and hasattr(code, 'outer'):
            self._inner_code = code.inner
            self._outer_code = code.outer
        else:
            raise ValueError("Code must have inner/outer code attributes")
        
        self._n_outer = code._n_outer  # Number of inner code blocks
        
        # Cache CSS matrices
        self._hx = code.hx
        self._hz = code.hz
        self._n_x = self._hx.shape[0] if self._hx is not None and self._hx.size > 0 else 0
        self._n_z = self._hz.shape[0] if self._hz is not None and self._hz.size > 0 else 0
        
        # Inner code stabilizer counts
        self._n_inner_x = len(self._inner_code.hx)
        self._n_inner_z = len(self._inner_code.hz)
        
        # Outer code stabilizer counts (lifted)
        self._n_outer_x = len(self._outer_code.hx)
        self._n_outer_z = len(self._outer_code.hz)
        
        # Track last measurements for time-like detectors
        # Inner: indexed by (block_id, local_idx)
        self._last_inner_x_meas: Dict[Tuple[int, int], int] = {}
        self._last_inner_z_meas: Dict[Tuple[int, int], int] = {}
        # Outer: indexed by local_idx
        self._last_outer_x_meas: Dict[int, int] = {}
        self._last_outer_z_meas: Dict[int, int] = {}
        
        # Block-contiguous mode for hierarchical decoding
        self._block_contiguous = block_contiguous
        
        # Storage for deferred detector emission (block-contiguous mode)
        # inner_x_meas[block_id][local_idx] = list of (round, meas_idx)
        self._deferred_inner_x: List[List[List[Tuple[int, int]]]] = [
            [[] for _ in range(self._n_inner_x)] for _ in range(self._n_outer)
        ]
        self._deferred_inner_z: List[List[List[Tuple[int, int]]]] = [
            [[] for _ in range(self._n_inner_z)] for _ in range(self._n_outer)
        ]
        self._deferred_outer_x: List[List[Tuple[int, int]]] = [
            [] for _ in range(self._n_outer_x)
        ]
        self._deferred_outer_z: List[List[Tuple[int, int]]] = [
            [] for _ in range(self._n_outer_z)
        ]
        
        # Track round number
        self._round_number = 0
    
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
    def total_qubits(self) -> int:
        """Total qubits used by this block (data + ancillas)."""
        return self.code.n + self._n_x + self._n_z
    
    def emit_qubit_coords(self, circuit: stim.Circuit) -> None:
        """Emit QUBIT_COORDS for all qubits in this block."""
        # Data qubits
        for local_idx, coord in enumerate(self._data_coords):
            if len(coord) >= 2:
                global_idx = self.data_offset + local_idx
                circuit.append("QUBIT_COORDS", [global_idx], [float(coord[0]), float(coord[1])])
    
    def emit_reset_all(self, circuit: stim.Circuit) -> None:
        """Reset all data and ancilla qubits."""
        all_qubits = self.data_qubits + self.x_ancillas + self.z_ancillas
        if all_qubits:
            circuit.append("R", all_qubits)
    
    def emit_prepare_logical_state(
        self,
        circuit: stim.Circuit,
        state: str = "0",
        logical_idx: int = 0,
    ) -> None:
        """Prepare a logical eigenstate."""
        code = self.code
        
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
    
    def emit_round(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_detectors: bool = True,
        emit_metachecks: bool = False,
    ) -> None:
        """
        Emit one complete stabilizer measurement round.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        stab_type : StabilizerBasis
            Which stabilizers to measure.
        emit_detectors : bool
            Whether to emit detectors. If block_contiguous mode, detectors
            are deferred regardless of this setting.
        emit_metachecks : bool
            Ignored for concatenated codes (no metachecks).
        """
        data = self.data_qubits
        x_anc = self.x_ancillas
        z_anc = self.z_ancillas
        
        # X stabilizer round
        if stab_type in (StabilizerBasis.X, StabilizerBasis.BOTH) and self._n_x > 0:
            # Apply H to X ancillas
            circuit.append("H", x_anc)
            circuit.append("TICK")
            
            # Apply CNOTs
            self._emit_stabilizer_cnots(circuit, self._hx, data, x_anc, is_x_type=True)
            
            # Final H on X ancillas
            circuit.append("H", x_anc)
            circuit.append("TICK")
            
            # Measure X ancillas
            x_meas_start = self.ctx.add_measurement(self._n_x)
            circuit.append("MR", x_anc)
            
            # Process X measurements
            self._process_x_measurements(circuit, x_meas_start, emit_detectors)
        
        # Z stabilizer round
        if stab_type in (StabilizerBasis.Z, StabilizerBasis.BOTH) and self._n_z > 0:
            # Apply CNOTs (no H for Z-type)
            self._emit_stabilizer_cnots(circuit, self._hz, data, z_anc, is_x_type=False)
            circuit.append("TICK")
            
            # Measure Z ancillas
            z_meas_start = self.ctx.add_measurement(self._n_z)
            circuit.append("MR", z_anc)
            
            # Process Z measurements
            self._process_z_measurements(circuit, z_meas_start, emit_detectors)
        
        # Advance time
        self.ctx.advance_time()
        self._emit_shift_coords(circuit)
        self._round_number += 1
    
    def _emit_stabilizer_cnots(
        self,
        circuit: stim.Circuit,
        parity_check: np.ndarray,
        data_qubits: List[int],
        ancilla_qubits: List[int],
        is_x_type: bool,
    ) -> None:
        """Emit CNOTs for stabilizer measurements using graph coloring."""
        if parity_check is None or parity_check.size == 0:
            return
        
        n_stabs = parity_check.shape[0]
        n = len(data_qubits)
        
        # Collect all CNOT pairs
        cnot_pairs: List[Tuple[int, int]] = []
        
        for s_idx in range(min(n_stabs, len(ancilla_qubits))):
            anc = ancilla_qubits[s_idx]
            row = parity_check[s_idx]
            
            for q in range(min(n, len(row))):
                if row[q]:
                    dq = data_qubits[q]
                    if is_x_type:
                        # X-type: ancilla controls data
                        cnot_pairs.append((anc, dq))
                    else:
                        # Z-type: data controls ancilla
                        cnot_pairs.append((dq, anc))
        
        if not cnot_pairs:
            return
        
        # Schedule into conflict-free layers
        layers = graph_coloring_cnots(cnot_pairs)
        
        for layer_idx, layer_cnots in enumerate(layers):
            if layer_idx > 0:
                circuit.append("TICK")
            for ctrl, targ in layer_cnots:
                circuit.append("CX", [ctrl, targ])
    
    def _process_x_measurements(
        self,
        circuit: stim.Circuit,
        meas_start: int,
        emit_detectors: bool,
    ) -> None:
        """Process X stabilizer measurements and emit/defer detectors."""
        # Process inner X stabilizers
        for block_id in range(self._n_outer):
            for local_idx in range(self._n_inner_x):
                global_idx = block_id * self._n_inner_x + local_idx
                cur_meas = meas_start + global_idx
                
                key = (block_id, local_idx)
                prev_meas = self._last_inner_x_meas.get(key)
                
                if self._block_contiguous:
                    # Defer detector emission
                    self._deferred_inner_x[block_id][local_idx].append(
                        (self._round_number, cur_meas)
                    )
                elif emit_detectors:
                    # Emit detector immediately
                    self._emit_x_detector(circuit, cur_meas, prev_meas, block_id, local_idx)
                
                self._last_inner_x_meas[key] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, f"inner_x_{block_id}", local_idx, cur_meas
                )
        
        # Process outer X stabilizers
        for local_idx in range(self._n_outer_x):
            global_idx = self._n_outer * self._n_inner_x + local_idx
            cur_meas = meas_start + global_idx
            
            prev_meas = self._last_outer_x_meas.get(local_idx)
            
            if self._block_contiguous:
                self._deferred_outer_x[local_idx].append(
                    (self._round_number, cur_meas)
                )
            elif emit_detectors:
                self._emit_outer_x_detector(circuit, cur_meas, prev_meas, local_idx)
            
            self._last_outer_x_meas[local_idx] = cur_meas
            self.ctx.record_stabilizer_measurement(
                self.block_name, "outer_x", local_idx, cur_meas
            )
    
    def _process_z_measurements(
        self,
        circuit: stim.Circuit,
        meas_start: int,
        emit_detectors: bool,
    ) -> None:
        """Process Z stabilizer measurements and emit/defer detectors."""
        # Process inner Z stabilizers
        for block_id in range(self._n_outer):
            for local_idx in range(self._n_inner_z):
                global_idx = block_id * self._n_inner_z + local_idx
                cur_meas = meas_start + global_idx
                
                key = (block_id, local_idx)
                prev_meas = self._last_inner_z_meas.get(key)
                
                if self._block_contiguous:
                    self._deferred_inner_z[block_id][local_idx].append(
                        (self._round_number, cur_meas)
                    )
                elif emit_detectors:
                    self._emit_z_detector(circuit, cur_meas, prev_meas, block_id, local_idx)
                
                self._last_inner_z_meas[key] = cur_meas
                self.ctx.record_stabilizer_measurement(
                    self.block_name, f"inner_z_{block_id}", local_idx, cur_meas
                )
        
        # Process outer Z stabilizers
        for local_idx in range(self._n_outer_z):
            global_idx = self._n_outer * self._n_inner_z + local_idx
            cur_meas = meas_start + global_idx
            
            prev_meas = self._last_outer_z_meas.get(local_idx)
            
            if self._block_contiguous:
                self._deferred_outer_z[local_idx].append(
                    (self._round_number, cur_meas)
                )
            elif emit_detectors:
                self._emit_outer_z_detector(circuit, cur_meas, prev_meas, local_idx)
            
            self._last_outer_z_meas[local_idx] = cur_meas
            self.ctx.record_stabilizer_measurement(
                self.block_name, "outer_z", local_idx, cur_meas
            )
    
    def _emit_x_detector(
        self,
        circuit: stim.Circuit,
        cur_meas: int,
        prev_meas: Optional[int],
        block_id: int,
        local_idx: int,
    ) -> None:
        """Emit detector for inner X stabilizer."""
        coord = (float(block_id), float(local_idx), self.ctx.current_time)
        
        if prev_meas is None:
            # First round: only emit if X-basis memory
            if self.measurement_basis == "X":
                self.ctx.emit_detector(circuit, [cur_meas], coord)
        else:
            self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
    
    def _emit_z_detector(
        self,
        circuit: stim.Circuit,
        cur_meas: int,
        prev_meas: Optional[int],
        block_id: int,
        local_idx: int,
    ) -> None:
        """Emit detector for inner Z stabilizer."""
        coord = (float(block_id), float(local_idx), self.ctx.current_time)
        
        if prev_meas is None:
            # First round: only emit if Z-basis memory
            if self.measurement_basis == "Z":
                self.ctx.emit_detector(circuit, [cur_meas], coord)
        else:
            self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
    
    def _emit_outer_x_detector(
        self,
        circuit: stim.Circuit,
        cur_meas: int,
        prev_meas: Optional[int],
        local_idx: int,
    ) -> None:
        """Emit detector for outer X stabilizer."""
        coord = (float(self._n_outer), float(local_idx), self.ctx.current_time)
        
        if prev_meas is None:
            if self.measurement_basis == "X":
                self.ctx.emit_detector(circuit, [cur_meas], coord)
        else:
            self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
    
    def _emit_outer_z_detector(
        self,
        circuit: stim.Circuit,
        cur_meas: int,
        prev_meas: Optional[int],
        local_idx: int,
    ) -> None:
        """Emit detector for outer Z stabilizer."""
        coord = (float(self._n_outer), float(local_idx), self.ctx.current_time)
        
        if prev_meas is None:
            if self.measurement_basis == "Z":
                self.ctx.emit_detector(circuit, [cur_meas], coord)
        else:
            self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
    
    def emit_deferred_detectors(
        self,
        circuit: stim.Circuit,
        data_meas: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Emit all deferred detectors in block-contiguous order.
        
        This should be called after all rounds and final measurement when
        using block_contiguous mode. Detectors are emitted in the order:
        
            [Block 0 detectors] [Block 1 detectors] ... [Outer detectors]
        
        Within each block: X time-like, Z time-like, then space-like.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        data_meas : dict, optional
            Mapping from data qubit index to measurement index.
            Required for space-like detectors.
        """
        if not self._block_contiguous:
            return
        
        basis = self.measurement_basis
        
        # Inner blocks first (block-contiguous order)
        for block_id in range(self._n_outer):
            # X stabilizer detectors for this block
            for local_idx in range(self._n_inner_x):
                meas_list = self._deferred_inner_x[block_id][local_idx]
                coord_base = (float(block_id), float(local_idx))
                
                for i, (r, cur_meas) in enumerate(meas_list):
                    coord = coord_base + (float(r),)
                    if i == 0:
                        if basis == "X":
                            self.ctx.emit_detector(circuit, [cur_meas], coord)
                    else:
                        prev_meas = meas_list[i - 1][1]
                        self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
            
            # Z stabilizer detectors for this block
            for local_idx in range(self._n_inner_z):
                meas_list = self._deferred_inner_z[block_id][local_idx]
                coord_base = (float(block_id), float(local_idx))
                
                for i, (r, cur_meas) in enumerate(meas_list):
                    coord = coord_base + (float(r),)
                    if i == 0:
                        if basis == "Z":
                            self.ctx.emit_detector(circuit, [cur_meas], coord)
                    else:
                        prev_meas = meas_list[i - 1][1]
                        self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
            
            # Space-like detectors for this block
            if data_meas:
                self._emit_block_space_like_detectors(circuit, block_id, data_meas)
        
        # Outer stabilizer detectors
        for local_idx in range(self._n_outer_x):
            meas_list = self._deferred_outer_x[local_idx]
            coord_base = (float(self._n_outer), float(local_idx))
            
            for i, (r, cur_meas) in enumerate(meas_list):
                coord = coord_base + (float(r),)
                if i == 0:
                    if basis == "X":
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                else:
                    prev_meas = meas_list[i - 1][1]
                    self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
        
        for local_idx in range(self._n_outer_z):
            meas_list = self._deferred_outer_z[local_idx]
            coord_base = (float(self._n_outer), float(local_idx))
            
            for i, (r, cur_meas) in enumerate(meas_list):
                coord = coord_base + (float(r),)
                if i == 0:
                    if basis == "Z":
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                else:
                    prev_meas = meas_list[i - 1][1]
                    self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
        
        # Outer space-like detectors
        if data_meas:
            self._emit_outer_space_like_detectors(circuit, data_meas)
    
    def _emit_block_space_like_detectors(
        self,
        circuit: stim.Circuit,
        block_id: int,
        data_meas: Dict[int, int],
    ) -> None:
        """Emit space-like detectors for an inner block."""
        basis = self.measurement_basis
        n_rounds = self._round_number
        
        if basis == "Z":
            # Z space-like detectors
            for local_idx in range(self._n_inner_z):
                global_idx = block_id * self._n_inner_z + local_idx
                if global_idx >= len(self._hz):
                    continue
                
                row = self._hz[global_idx]
                data_idxs = [data_meas[q] for q in np.where(row == 1)[0] if q in data_meas]
                
                recs = list(data_idxs)
                meas_list = self._deferred_inner_z[block_id][local_idx]
                if meas_list:
                    recs.append(meas_list[-1][1])  # Last round measurement
                
                if recs:
                    coord = (float(block_id), float(local_idx), float(n_rounds))
                    self.ctx.emit_detector(circuit, recs, coord)
        else:
            # X space-like detectors
            for local_idx in range(self._n_inner_x):
                global_idx = block_id * self._n_inner_x + local_idx
                if global_idx >= len(self._hx):
                    continue
                
                row = self._hx[global_idx]
                data_idxs = [data_meas[q] for q in np.where(row == 1)[0] if q in data_meas]
                
                recs = list(data_idxs)
                meas_list = self._deferred_inner_x[block_id][local_idx]
                if meas_list:
                    recs.append(meas_list[-1][1])
                
                if recs:
                    coord = (float(block_id), float(local_idx), float(n_rounds))
                    self.ctx.emit_detector(circuit, recs, coord)
    
    def _emit_outer_space_like_detectors(
        self,
        circuit: stim.Circuit,
        data_meas: Dict[int, int],
    ) -> None:
        """Emit space-like detectors for outer stabilizers."""
        basis = self.measurement_basis
        n_rounds = self._round_number
        
        if basis == "Z":
            for local_idx in range(self._n_outer_z):
                global_idx = self._n_outer * self._n_inner_z + local_idx
                if global_idx >= len(self._hz):
                    continue
                
                row = self._hz[global_idx]
                data_idxs = [data_meas[q] for q in np.where(row == 1)[0] if q in data_meas]
                
                recs = list(data_idxs)
                meas_list = self._deferred_outer_z[local_idx]
                if meas_list:
                    recs.append(meas_list[-1][1])
                
                if recs:
                    coord = (float(self._n_outer), float(local_idx), float(n_rounds))
                    self.ctx.emit_detector(circuit, recs, coord)
        else:
            for local_idx in range(self._n_outer_x):
                global_idx = self._n_outer * self._n_inner_x + local_idx
                if global_idx >= len(self._hx):
                    continue
                
                row = self._hx[global_idx]
                data_idxs = [data_meas[q] for q in np.where(row == 1)[0] if q in data_meas]
                
                recs = list(data_idxs)
                meas_list = self._deferred_outer_x[local_idx]
                if meas_list:
                    recs.append(meas_list[-1][1])
                
                if recs:
                    coord = (float(self._n_outer), float(local_idx), float(n_rounds))
                    self.ctx.emit_detector(circuit, recs, coord)
    
    def emit_final_measurement(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
        logical_idx: int = 0,
    ) -> List[int]:
        """
        Emit final data qubit measurements.
        
        In block_contiguous mode, space-like detectors are deferred and
        emitted via emit_deferred_detectors().
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        basis : str
            Measurement basis ("Z" or "X").
        logical_idx : int
            Which logical qubit.
            
        Returns
        -------
        List[int]
            Measurement indices for the logical observable.
        """
        from .utils import get_logical_support
        
        data = self.data_qubits
        n = len(data)
        basis = basis.upper()
        
        if basis == "X":
            circuit.append("H", data)
        
        meas_start = self.ctx.add_measurement(n)
        circuit.append("M", data)
        
        # Build data measurement lookup for space-like detectors
        data_meas = {q: meas_start + i for i, q in enumerate(range(n))}
        
        # Emit space-like detectors (or defer if block_contiguous)
        if not self._block_contiguous:
            self._emit_all_space_like_detectors(circuit, basis, data_meas)
        
        # Compute logical observable
        effective_basis = self.ctx.get_transformed_basis(logical_idx, basis)
        logical_support = get_logical_support(self.code, effective_basis, logical_idx)
        
        if logical_support:
            logical_meas = [meas_start + q for q in logical_support if q < n]
        else:
            logical_meas = [meas_start]  # Fallback
        
        self.ctx.add_observable_measurement(logical_idx, logical_meas)
        
        return logical_meas
    
    def _emit_all_space_like_detectors(
        self,
        circuit: stim.Circuit,
        basis: str,
        data_meas: Dict[int, int],
    ) -> None:
        """Emit all space-like detectors (non-block-contiguous mode)."""
        basis = basis.upper()
        
        if basis == "Z":
            # Z space-like for all stabilizers
            for s_idx in range(self._n_z):
                # Determine if inner or outer
                if s_idx < self._n_outer * self._n_inner_z:
                    block_id = s_idx // self._n_inner_z
                    local_idx = s_idx % self._n_inner_z
                    last_meas = self._last_inner_z_meas.get((block_id, local_idx))
                else:
                    local_idx = s_idx - self._n_outer * self._n_inner_z
                    last_meas = self._last_outer_z_meas.get(local_idx)
                
                if last_meas is None:
                    continue
                
                row = self._hz[s_idx]
                data_idxs = [data_meas[q] for q in np.where(row == 1)[0] if q in data_meas]
                
                if data_idxs:
                    recs = data_idxs + [last_meas]
                    coord = (0.0, float(s_idx), self.ctx.current_time)
                    self.ctx.emit_detector(circuit, recs, coord)
        else:
            # X space-like
            for s_idx in range(self._n_x):
                if s_idx < self._n_outer * self._n_inner_x:
                    block_id = s_idx // self._n_inner_x
                    local_idx = s_idx % self._n_inner_x
                    last_meas = self._last_inner_x_meas.get((block_id, local_idx))
                else:
                    local_idx = s_idx - self._n_outer * self._n_inner_x
                    last_meas = self._last_outer_x_meas.get(local_idx)
                
                if last_meas is None:
                    continue
                
                row = self._hx[s_idx]
                data_idxs = [data_meas[q] for q in np.where(row == 1)[0] if q in data_meas]
                
                if data_idxs:
                    recs = data_idxs + [last_meas]
                    coord = (0.0, float(s_idx), self.ctx.current_time)
                    self.ctx.emit_detector(circuit, recs, coord)
    
    def emit_rounds(
        self,
        circuit: stim.Circuit,
        num_rounds: int,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
    ) -> None:
        """Emit multiple stabilizer measurement rounds."""
        for _ in range(num_rounds):
            self.emit_round(circuit, stab_type, emit_detectors=True)
    
    def reset_stabilizer_history(
        self,
        swap_xz: bool = False,
        skip_first_round: bool = False,
    ) -> None:
        """Reset stabilizer measurement history."""
        self._last_inner_x_meas.clear()
        self._last_inner_z_meas.clear()
        self._last_outer_x_meas.clear()
        self._last_outer_z_meas.clear()
        
        if swap_xz:
            self.measurement_basis = "X" if self.measurement_basis == "Z" else "Z"
    
    def get_data_meas_mapping(self, meas_start: int) -> Dict[int, int]:
        """
        Get mapping from data qubit index to measurement index.
        
        Useful for emit_deferred_detectors() after final measurement.
        """
        n = self.code.n
        return {q: meas_start + i for i, q in enumerate(range(n))}
    
    # Properties for hierarchical decoding
    @property
    def n_outer_blocks(self) -> int:
        """Number of inner code blocks."""
        return self._n_outer
    
    @property
    def n_inner_x_stabs(self) -> int:
        """Number of X stabilizers per inner block."""
        return self._n_inner_x
    
    @property
    def n_inner_z_stabs(self) -> int:
        """Number of Z stabilizers per inner block."""
        return self._n_inner_z
    
    @property
    def n_outer_x_stabs(self) -> int:
        """Number of outer X stabilizers."""
        return self._n_outer_x
    
    @property
    def n_outer_z_stabs(self) -> int:
        """Number of outer Z stabilizers."""
        return self._n_outer_z
    
    @property
    def inner_code(self):
        """The inner code."""
        return self._inner_code
    
    @property
    def outer_code(self):
        """The outer code."""
        return self._outer_code
    
    def get_syndrome_meas_history(self) -> Dict[str, Any]:
        """
        Get the syndrome measurement history for observable determinism fix.
        
        Returns a dictionary containing measurement indices for all stabilizers
        organized by block and type, enabling the experiment to include necessary
        syndrome measurements in inner observable definitions to make them
        deterministic.
        
        Returns
        -------
        dict
            Dictionary with structure:
            {
                'inner_x': List[List[List[int]]], # [block][stab][meas_indices_per_round]
                'inner_z': List[List[List[int]]],
                'outer_x': List[List[int]],       # [stab][meas_indices_per_round]
                'outer_z': List[List[int]],
                'last_inner_x': Dict[(block, stab), int],  # Last measurement index
                'last_inner_z': Dict[(block, stab), int],
            }
        """
        # Extract measurement indices from deferred lists (they store (round, meas_idx))
        inner_x_meas = [
            [[m[1] for m in stab_list] for stab_list in block_list]
            for block_list in self._deferred_inner_x
        ]
        inner_z_meas = [
            [[m[1] for m in stab_list] for stab_list in block_list]
            for block_list in self._deferred_inner_z
        ]
        outer_x_meas = [[m[1] for m in stab_list] for stab_list in self._deferred_outer_x]
        outer_z_meas = [[m[1] for m in stab_list] for stab_list in self._deferred_outer_z]
        
        # Get last measurement indices
        last_inner_x = dict(self._last_inner_x_meas)
        last_inner_z = dict(self._last_inner_z_meas)
        
        return {
            'inner_x': inner_x_meas,
            'inner_z': inner_z_meas,
            'outer_x': outer_x_meas,
            'outer_z': outer_z_meas,
            'last_inner_x': last_inner_x,
            'last_inner_z': last_inner_z,
        }


# =============================================================================
# Backward Compatibility Alias (Deprecated)
# =============================================================================

class ConcatenatedStabilizerRoundBuilder(FlatConcatenatedStabilizerRoundBuilder):
    """
    DEPRECATED: Use FlatConcatenatedStabilizerRoundBuilder instead.
    
    This alias exists for backward compatibility and will be removed in a future version.
    
    For flat concatenation model: FlatConcatenatedStabilizerRoundBuilder
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ConcatenatedStabilizerRoundBuilder is deprecated and will be removed in a "
            "future version. Use FlatConcatenatedStabilizerRoundBuilder for flat "
            "concatenation model.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
