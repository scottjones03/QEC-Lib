# src/qectostim/experiments/stabilizer_rounds/hierarchical_concatenated.py
"""
Hierarchical Concatenated Stabilizer Round Builder.

This module implements a HIERARCHICAL model for concatenated CSS code
syndrome extraction using logical ancilla blocks for fault-tolerant outer
stabilizer measurement.

Architecture
------------
For concatenated code outer ∘ inner:
- **Inner stabilizers**: Standard parallel extraction on each of n_out data blocks
- **Outer X stabilizers**: Measured via logical |+_L⟩ ancilla blocks + transversal CNOTs
- **Outer Z stabilizers**: Measured via logical |0_L⟩ ancilla blocks + transversal CNOTs

This provides proper fault tolerance at the outer level, unlike the flat model
which uses direct physical CNOTs.

Qubit Allocation
----------------
Total qubits = N_data + N_inner_anc + N_outer_anc_blocks

Where:
- N_data = n_out × n_in  (data qubits)
- N_inner_anc = n_out × (r_x_in + r_z_in)  (inner syndrome ancillas)
- N_outer_anc_blocks = max(r_x_out, r_z_out) × (n_in + r_x_in + r_z_in)
  (logical ancilla blocks for outer stabilizers, reused between X and Z phases)

Three-Phase Round Structure
---------------------------
Phase 1: Inner Stabilizers
    - Parallel X/Z syndrome extraction on all n_out data blocks
    - Uses standard CSS ancillas for each block

Phase 2: Outer X Stabilizers (via |+_L⟩ ancillas)
    - Prepare r_x_out logical ancilla blocks in |+_L⟩
    - Transversal CNOT: ancilla[j,i] → data[k,i] for each outer_x[j,k]=1
    - Measure inner Z-stabs on ancillas (deterministic after CNOT)
    - Measure inner X-stabs on ancillas (random outcome is outer X syndrome)
    - Measure data qubits on ancillas in X basis (recovers |+_L⟩ outcome)

Phase 3: Outer Z Stabilizers (via |0_L⟩ ancillas)  
    - Prepare r_z_out logical ancilla blocks in |0_L⟩
    - Transversal CNOT: data[k,i] → ancilla[j,i] for each outer_z[j,k]=1
    - Measure inner X-stabs on ancillas (deterministic after CNOT)
    - Measure inner Z-stabs on ancillas (random outcome is outer Z syndrome)
    - Measure data qubits on ancillas in Z basis (recovers |0_L⟩ outcome)

Detector Coverage
-----------------
For Z-basis memory (|0⟩_L initial state):
- Inner Z anchors: Round 1 inner Z-stabs (deterministic on |0⟩^⊗n)
- Inner X temporal: XOR consecutive inner X rounds (random first round)
- Inner Z temporal: XOR consecutive inner Z rounds
- Outer Z temporal: XOR consecutive outer Z phases (via |0_L⟩ ancilla Z-stabs)
- Outer X temporal: XOR consecutive outer X phases (via |+_L⟩ ancilla X-stabs)
- Boundary detectors: Final data measurement vs last syndrome round

For X-basis memory (|+⟩_L initial state), swap X↔Z in above.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

import numpy as np
import stim

from qectostim.utils.scheduling_core import graph_coloring_cnots

from .context import DetectorContext
from .base import BaseStabilizerRoundBuilder, StabilizerBasis
from .css import CSSStabilizerRoundBuilder

if TYPE_CHECKING:
    from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
    from qectostim.codes.abstract_css import CSSCode


class DetectorRole(Enum):
    """Role tag for the 4th detector coordinate.

    Used by hierarchical decoders to partition the full DEM into
    inner-per-block and outer sub-DEMs.
    """
    INNER_TEMPORAL = 0   # Inner stab temporal or anchor (pure per-block)
    INNER_CROSSING = 1   # Inner stab crossing (multi-term across outer boundary)
    INNER_BOUNDARY = 2   # Inner stab space-like (final-measurement boundary)
    OUTER_TEMPORAL = 3   # Outer stab temporal, anchor, or boundary
    OAB_BOUNDARY   = 4   # Outer ancilla block inner-stab boundary detector


class InnerBlockType(Enum):
    """Type of inner block for distinguishing detector behavior."""
    DATA = 0             # Data blocks (prepared once at start)
    OUTER_X_ANCILLA = 1  # Ancilla blocks for outer X stab (prep in |+_L⟩ each outer round)
    OUTER_Z_ANCILLA = 2  # Ancilla blocks for outer Z stab (prep in |0_L⟩ each outer round)


class HierarchicalConcatenatedStabilizerRoundBuilder(BaseStabilizerRoundBuilder):
    """
    Hierarchical stabilizer round builder for concatenated CSS codes.
    
    Implements fault-tolerant outer stabilizer extraction using logical ancilla
    blocks prepared in |+_L⟩ or |0_L⟩ states.
    
    Parameters
    ----------
    code : ConcatenatedCSSCode
        The concatenated CSS code.
    ctx : DetectorContext
        Detector tracking context.
    block_name : str
        Name prefix for this block.
    data_offset : int
        Starting qubit index for data qubits.
    measurement_basis : str
        "Z" or "X" - determines which stabilizers are deterministic initially.
    block_contiguous : bool
        If True, defer detector emission for block-contiguous DEM ordering.
        This groups detectors by logical block for hierarchical decoding.
    
    Attributes
    ----------
    outer_code : CSSCode
        The outer code (acts on logical qubits of inner code).
    inner_code : CSSCode
        The inner code (provides physical encoding).
    n_out : int
        Number of physical qubits in outer code (= number of inner blocks).
    n_in : int
        Number of physical qubits in inner code.
    r_x_out, r_z_out : int
        Number of outer X/Z stabilizers.
    r_x_in, r_z_in : int
        Number of inner X/Z stabilizers.
    """
    
    def __init__(
        self,
        code: "ConcatenatedCSSCode",
        ctx: DetectorContext,
        block_name: str = "concat",
        data_offset: int = 0,
        measurement_basis: str = "Z",
        block_contiguous: bool = False,
        d_inner: int = 1,
    ):
        """
        Initialize hierarchical concatenated round builder.
        
        Parameters
        ----------
        code : ConcatenatedCSSCode
            The concatenated CSS code.
        ctx : DetectorContext
            Detector tracking context.
        block_name : str
            Name prefix for this block.
        data_offset : int
            Starting qubit index for data qubits.
        measurement_basis : str
            "Z" or "X" - determines which stabilizers are deterministic initially.
        block_contiguous : bool
            If True, defer detector emission for block-contiguous DEM ordering.
        d_inner : int
            Number of inner syndrome rounds per outer syndrome round.
            This enables inner temporal detectors within each block of d_inner rounds.
            Structure: (d_outer) × { (d_inner) × [inner X, inner Z], outer X, outer Z }
        """
        super().__init__(code, ctx, block_name, data_offset, None, measurement_basis)
        
        # Inner rounds per outer round
        self.d_inner = d_inner
        
        # Counter for position within current inner block (0 to d_inner-1)
        self._inner_round_in_block = 0
        
        # Counter for outer rounds completed
        self._outer_round_number = 0
        
        # Extract outer and inner codes
        self.outer_code: "CSSCode" = code.outer
        self.inner_code: "CSSCode" = code.inner
        
        # Code dimensions
        self.n_out = code.n_outer  # Number of inner blocks (outer code physical qubits)
        self.n_in = code.n_inner   # Qubits per inner block
        
        # Inner stabilizer counts
        self.r_x_in = self.inner_code.hx.shape[0] if self.inner_code.hx.size > 0 else 0
        self.r_z_in = self.inner_code.hz.shape[0] if self.inner_code.hz.size > 0 else 0
        
        # Outer stabilizer counts
        self.r_x_out = self.outer_code.hx.shape[0] if self.outer_code.hx.size > 0 else 0
        self.r_z_out = self.outer_code.hz.shape[0] if self.outer_code.hz.size > 0 else 0
        
        # Total data qubits
        self.n_data = self.n_out * self.n_in
        
        # Qubit allocation
        self._compute_qubit_allocation()
        
        # Inner code parity check matrices (cached)
        self._hx_in = self.inner_code.hx
        self._hz_in = self.inner_code.hz
        
        # Outer code parity check matrices (cached)
        self._hx_out = self.outer_code.hx
        self._hz_out = self.outer_code.hz
        
        # Inner logical operator supports (for outer syndrome decoding)
        self._inner_logical_x_support = self.inner_code.get_logical_x_support(0)
        self._inner_logical_z_support = self.inner_code.get_logical_z_support(0)
        
        # Block-contiguous mode for hierarchical DEM
        self._block_contiguous = block_contiguous
        
        # Measurement history tracking
        self._init_measurement_history()
        
        # Round counter
        self._round_number = 0

        # Skip-first-round flag: when True, suppress all anchor detectors
        # in the next round after a reset_stabilizer_history call (e.g. after
        # a transversal gate).  The flag auto-clears after the first inner
        # round is emitted.
        self._skip_first_round: bool = False
        
        # Gadget block index for unique detector coordinates in multi-block gadgets.
        # Extract from block_name (e.g., "block_0" -> 0, "block_1" -> 1).
        self._gadget_block_index = self._extract_gadget_block_index(block_name)
    
    def _extract_gadget_block_index(self, block_name: str) -> int:
        """Extract numeric index from block name for unique detector coordinates."""
        import re
        match = re.search(r'_(\d+)$', block_name)
        if match:
            return int(match.group(1))
        return 0  # Default for single-block or unrecognized patterns
    
    def _make_detector_coord(
        self,
        block_id: float,
        stab_id: float,
        time: float,
        role: float,
    ) -> Tuple[float, float, float, float, float]:
        """Create detector coordinate tuple with gadget block index.
        
        All detectors in multi-block gadgets need a 5th coordinate to
        distinguish detectors from different gadget blocks (e.g., control
        vs target in a transversal CNOT).
        """
        return (block_id, stab_id, time, role, float(self._gadget_block_index))
    
    def _compute_qubit_allocation(self) -> None:
        """Compute qubit index ranges for all regions."""
        # Data qubits: [0, N_data)
        # Layout: block 0 qubits [0, n_in), block 1 qubits [n_in, 2*n_in), etc.
        self._data_start = self.data_offset
        self._data_end = self._data_start + self.n_data
        
        # Inner syndrome ancillas: [N_data, N_data + n_out*(r_x_in + r_z_in))
        # Layout: block 0 X-ancillas, block 0 Z-ancillas, block 1 X-ancillas, ...
        self._inner_anc_start = self._data_end
        self._inner_anc_per_block = self.r_x_in + self.r_z_in
        self._inner_anc_end = self._inner_anc_start + self.n_out * self._inner_anc_per_block
        
        # Outer ancilla blocks: [inner_anc_end, inner_anc_end + max(r_x_out, r_z_out) * block_size)
        # Each outer ancilla block has: n_in data + r_x_in X-anc + r_z_in Z-anc
        self._outer_anc_block_size = self.n_in + self.r_x_in + self.r_z_in
        self._n_outer_anc_blocks = max(self.r_x_out, self.r_z_out)
        self._outer_anc_start = self._inner_anc_end
        self._outer_anc_end = self._outer_anc_start + self._n_outer_anc_blocks * self._outer_anc_block_size
        
        # Total qubits
        self._total_qubits = self._outer_anc_end - self.data_offset
    
    def _init_measurement_history(self) -> None:
        """Initialize measurement tracking structures."""
        # Inner stabilizer measurements on DATA blocks: (block_id, stab_idx) -> last_meas_idx
        self._last_inner_x_meas: Dict[Tuple[int, int], int] = {}
        self._last_inner_z_meas: Dict[Tuple[int, int], int] = {}
        
        # Inner stabilizer measurements on ANCILLA blocks during prep
        # These are used for boundary detectors after destructive measurement
        # Key: (anc_block_idx, stab_idx) -> meas_idx from most recent prep
        self._last_anc_inner_x_meas_prep: Dict[Tuple[int, int], int] = {}
        self._last_anc_inner_z_meas_prep: Dict[Tuple[int, int], int] = {}
        
        # Outer stabilizer measurements: stab_idx -> [meas_indices for logical decode]
        # Each outer measurement stores ALL measurements in the logical support
        self._last_outer_x_meas: Dict[int, List[int]] = {}
        self._last_outer_z_meas: Dict[int, List[int]] = {}
        
        # Deferred detectors for block-contiguous mode
        # Structure: block_id -> stab_idx -> [(round, meas_idx), ...]
        self._deferred_inner_x: List[List[List[Tuple[int, int]]]] = [
            [[] for _ in range(self.r_x_in)] for _ in range(self.n_out)
        ]
        self._deferred_inner_z: List[List[List[Tuple[int, int]]]] = [
            [[] for _ in range(self.r_z_in)] for _ in range(self.n_out)
        ]
        self._deferred_outer_x: List[List[Tuple[int, int]]] = [
            [] for _ in range(self.r_x_out)
        ]
        self._deferred_outer_z: List[List[Tuple[int, int]]] = [
            [] for _ in range(self.r_z_out)
        ]
        
        # Pre-outer measurements: saved before outer ops for crossing detectors
        # These hold the LAST inner measurements before outer X/Z transversal CNOTs
        self._pre_outer_inner_x_meas: Dict[Tuple[int, int], int] = {}
        self._pre_outer_inner_z_meas: Dict[Tuple[int, int], int] = {}
        
        # Ancilla prep measurements for crossing detector multi-term formulas:
        # - Outer X ancilla Z-stab prep: needed for inner Z crossing on data blocks
        # - Outer Z ancilla X-stab prep: needed for inner X crossing on data blocks
        # Key: (anc_block_idx, stab_idx) -> meas_idx
        self._outer_x_anc_z_prep: Dict[Tuple[int, int], int] = {}
        self._outer_z_anc_x_prep: Dict[Tuple[int, int], int] = {}
        
        # Destructive measurement storage for multi-term boundary detectors:
        # - Outer X ancilla destructive MX: for inner X boundary on outer X ancilla
        #   AND for inner X boundary on data blocks (X-basis memory)
        # - Outer Z ancilla destructive MZ: for inner Z boundary on data blocks (Z-basis memory)
        # Key: (anc_block_idx, stab_idx) -> List[meas_idx] (measurements in stab support)
        self._outer_x_anc_destruct_x: Dict[Tuple[int, int], List[int]] = {}
        self._outer_z_anc_destruct_z: Dict[Tuple[int, int], List[int]] = {}
    
    def _save_pre_outer_inner_meas(self) -> None:
        """Save current inner measurements for crossing detector emission."""
        # Copy current inner measurements to pre-outer storage
        self._pre_outer_inner_x_meas = dict(self._last_inner_x_meas)
        self._pre_outer_inner_z_meas = dict(self._last_inner_z_meas)
    
    def _outer_x_stabs_for_block(self, block_id: int) -> List[int]:
        """Return list of outer X stabilizer indices that include data block block_id."""
        result = []
        for j in range(self.r_x_out):
            if self._hx_out[j, block_id]:
                result.append(j)
        return result
    
    def _outer_z_stabs_for_block(self, block_id: int) -> List[int]:
        """Return list of outer Z stabilizer indices that include data block block_id."""
        result = []
        for j in range(self.r_z_out):
            if self._hz_out[j, block_id]:
                result.append(j)
        return result
    
    def _data_blocks_in_outer_z_stab(self, stab_idx: int) -> List[int]:
        """Return list of data block indices that participate in outer Z stabilizer stab_idx."""
        result = []
        for k in range(self.n_out):
            if self._hz_out[stab_idx, k]:
                result.append(k)
        return result
    
    def _data_blocks_in_outer_x_stab(self, stab_idx: int) -> List[int]:
        """Return list of data block indices that participate in outer X stabilizer stab_idx."""
        result = []
        for k in range(self.n_out):
            if self._hx_out[stab_idx, k]:
                result.append(k)
        return result
    
    # =========================================================================
    # Qubit Index Accessors
    # =========================================================================
    
    @property
    def data_qubits(self) -> List[int]:
        """All data qubit indices."""
        return list(range(self._data_start, self._data_end))
    
    def data_block_qubits(self, block_id: int) -> List[int]:
        """Data qubit indices for a specific inner block."""
        start = self._data_start + block_id * self.n_in
        return list(range(start, start + self.n_in))
    
    def inner_x_ancillas(self, block_id: int) -> List[int]:
        """X syndrome ancilla indices for a specific inner block."""
        block_start = self._inner_anc_start + block_id * self._inner_anc_per_block
        return list(range(block_start, block_start + self.r_x_in))
    
    def inner_z_ancillas(self, block_id: int) -> List[int]:
        """Z syndrome ancilla indices for a specific inner block."""
        block_start = self._inner_anc_start + block_id * self._inner_anc_per_block + self.r_x_in
        return list(range(block_start, block_start + self.r_z_in))
    
    def outer_anc_block_data(self, anc_block_id: int) -> List[int]:
        """Data qubit indices within an outer ancilla block."""
        block_start = self._outer_anc_start + anc_block_id * self._outer_anc_block_size
        return list(range(block_start, block_start + self.n_in))
    
    def outer_anc_block_x_ancillas(self, anc_block_id: int) -> List[int]:
        """X ancilla indices within an outer ancilla block."""
        block_start = self._outer_anc_start + anc_block_id * self._outer_anc_block_size + self.n_in
        return list(range(block_start, block_start + self.r_x_in))
    
    def outer_anc_block_z_ancillas(self, anc_block_id: int) -> List[int]:
        """Z ancilla indices within an outer ancilla block."""
        block_start = (self._outer_anc_start + anc_block_id * self._outer_anc_block_size 
                       + self.n_in + self.r_x_in)
        return list(range(block_start, block_start + self.r_z_in))
    
    @property
    def x_ancillas(self) -> List[int]:
        """All X syndrome ancilla indices (inner only)."""
        result = []
        for block_id in range(self.n_out):
            result.extend(self.inner_x_ancillas(block_id))
        return result
    
    @property
    def z_ancillas(self) -> List[int]:
        """All Z syndrome ancilla indices (inner only)."""
        result = []
        for block_id in range(self.n_out):
            result.extend(self.inner_z_ancillas(block_id))
        return result
    
    @property
    def total_qubits(self) -> int:
        """Total number of qubits used."""
        return self._total_qubits

    @property
    def all_ancillas(self) -> List[int]:
        """All ancilla qubit indices including outer ancilla blocks.

        Returns inner X/Z ancillas for all data blocks **plus** all qubits
        in the outer ancilla blocks (data + ancillas within each block).
        This is needed for ancilla reset between gadget phases.
        """
        result = self.x_ancillas + self.z_ancillas
        # Add all outer ancilla block qubits
        result.extend(range(self._outer_anc_start, self._outer_anc_end))
        return result

    @property
    def qubit_coords(self) -> Dict[int, Tuple[float, ...]]:
        """Mapping of qubit index to coordinates.

        Returns coordinates for data qubits derived from concatenated code
        metadata.  Inner/outer ancilla qubits are given simple synthetic
        coordinates so that crossing detector emission has fallback values.
        """
        coords: Dict[int, Tuple[float, ...]] = {}
        data_coords = self._meta.get('data_coords', [])
        for local_idx, coord in enumerate(data_coords):
            if len(coord) >= 2 and local_idx < self.n_data:
                global_idx = self._data_start + local_idx
                coords[global_idx] = tuple(float(c) for c in coord)
        return coords

    def get_last_measurement_indices(self) -> Dict[str, list]:
        """Return measurement indices from the last completed round.

        Returns a dict with lowercase keys ``'x'`` and ``'z'`` matching
        the format expected by :func:`emit_crossing_detectors` and
        :class:`MemoryRoundEmitter`.  Also includes ``'round'`` for
        compatibility with the CSS builder interface.

        For the hierarchical builder this returns **all** stabilizer
        measurements — inner AND outer — so that crossing detectors
        cover the full stabilizer group of the concatenated code.

        Inner entries are plain ``int`` (single ancilla measurement).
        Outer entries are ``List[int]`` (logical parity decoded from
        multiple qubit measurements on the inner logical support).
        The crossing-detector emitter handles both formats: a plain
        ``int`` becomes one ``target_rec`` while a ``List[int]``
        becomes several (Stim XORs them).
        """
        from typing import Union
        x_indices: List[Union[int, List[int]]] = []
        z_indices: List[Union[int, List[int]]] = []

        # Collect inner X measurements: iterate blocks × stabilizers
        for block_id in range(self.n_out):
            for s_idx in range(self.r_x_in):
                meas_idx = self._last_inner_x_meas.get((block_id, s_idx))
                if meas_idx is not None:
                    x_indices.append(meas_idx)

        # Collect inner Z measurements
        for block_id in range(self.n_out):
            for s_idx in range(self.r_z_in):
                meas_idx = self._last_inner_z_meas.get((block_id, s_idx))
                if meas_idx is not None:
                    z_indices.append(meas_idx)

        # Append outer X stabilizer measurements (List[int] each)
        for stab_idx in range(self.r_x_out):
            outer_meas = self._last_outer_x_meas.get(stab_idx)
            if outer_meas is not None:
                x_indices.append(list(outer_meas))

        # Append outer Z stabilizer measurements (List[int] each)
        for stab_idx in range(self.r_z_out):
            outer_meas = self._last_outer_z_meas.get(stab_idx)
            if outer_meas is not None:
                z_indices.append(list(outer_meas))

        return {
            'x': x_indices,
            'z': z_indices,
            'round': self._round_number,
        }

    # =========================================================================
    # Circuit Emission Methods
    # =========================================================================
    
    def emit_qubit_coords(self, circuit: stim.Circuit) -> None:
        """Emit QUBIT_COORDS for all qubits."""
        # Data qubits - use concatenated code coordinates if available
        data_coords = self._meta.get('data_coords', [])
        for local_idx, coord in enumerate(data_coords):
            if len(coord) >= 2 and local_idx < self.n_data:
                global_idx = self._data_start + local_idx
                circuit.append("QUBIT_COORDS", [global_idx], [float(coord[0]), float(coord[1])])
        
        # Inner ancilla coordinates could be derived from inner code's stab coords
        # For now, we skip detailed ancilla coordinates
    
    def emit_reset_all(self, circuit: stim.Circuit, *, skip_data: bool = False) -> None:
        """Reset all data, inner ancilla, and outer ancilla qubits.

        Parameters
        ----------
        skip_data : bool
            If True, skip resetting data qubits (caller will use RX).
        """
        if skip_data:
            anc_qubits = list(range(self._inner_anc_start, self._outer_anc_end))
            if anc_qubits:
                circuit.append("R", anc_qubits)
        else:
            all_qubits = list(range(self._data_start, self._outer_anc_end))
            if all_qubits:
                circuit.append("R", all_qubits)
    
    def emit_prepare_logical_state(
        self,
        circuit: stim.Circuit,
        state: str = "0",
        logical_idx: int = 0,
    ) -> None:
        """
        Prepare a logical eigenstate of the concatenated code.
        
        For Z-basis (|0⟩_L or |1⟩_L):
            - Reset all data to |0⟩
            - Measure inner Z-stabs FIRST (deterministic +1 on |0⟩^⊗n) → ANCHORS
            - Measure inner X-stabs (random, projects to code space)
            - For |1⟩_L, apply logical X
            
        For X-basis (|+⟩_L or |-⟩_L):
            - Reset all data to |+⟩ using RX (avoids H gates)
            - Measure inner X-stabs FIRST (deterministic +1 on |+⟩^⊗n) → ANCHORS
            - Measure inner Z-stabs (random, projects to code space)
            - For |-⟩_L, apply logical Z
            
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        state : str
            "0", "1", "+", or "-".
        logical_idx : int
            Which logical qubit to prepare.
        """
        data_qubits = self.data_qubits
        
        if state in ("0", "1"):
            # Z-basis preparation: |0⟩^⊗n is +1 eigenstate of Z-stabs
            # Order: R → Z-stabs (anchor) → X-stabs (random)
            
            # Measure inner Z-stabs FIRST (deterministic on |0⟩)
            self._emit_inner_z_projection(circuit, emit_anchors=True)
            
            # Measure inner X-stabs (random, establishes frame)
            self._emit_inner_x_projection(circuit, emit_anchors=False)
            
            if state == "1":
                # Apply logical X
                if hasattr(self.code, 'logical_x_support'):
                    support = self.code.logical_x_support(logical_idx)
                    for q in support:
                        circuit.append("X", [self._data_start + q])
                    circuit.append("TICK")
        
        elif state in ("+", "-"):
            # X-basis preparation: |+⟩^⊗n is +1 eigenstate of X-stabs
            # Use RX to prepare |+⟩ directly (avoids H gate)
            circuit.append("RX", data_qubits)
            circuit.append("TICK")
            
            # Measure inner X-stabs FIRST (deterministic on |+⟩)
            self._emit_inner_x_projection(circuit, emit_anchors=True)
            
            # Measure inner Z-stabs (random, establishes frame)
            self._emit_inner_z_projection(circuit, emit_anchors=False)
            
            if state == "-":
                # Apply logical Z
                if hasattr(self.code, 'logical_z_support'):
                    support = self.code.logical_z_support(logical_idx)
                    for q in support:
                        circuit.append("Z", [self._data_start + q])
                    circuit.append("TICK")
        
        # Advance time coordinate after initialization projection so that
        # subsequent stabilizer round detectors have distinct time coordinates.
        self.ctx.advance_time()
        self._emit_shift_coords(circuit)
        self._round_number += 1

    def _emit_inner_x_projection(
        self, 
        circuit: stim.Circuit, 
        emit_anchors: bool,
    ) -> None:
        """Emit inner X-stabilizer measurements for all blocks (projection phase)."""
        for block_id in range(self.n_out):
            data = self.data_block_qubits(block_id)
            x_anc = self.inner_x_ancillas(block_id)
            
            # Reset X ancillas to |+⟩ using RX (avoids H gate)
            circuit.append("RX", x_anc)
            circuit.append("TICK")
            
            # CNOTs: ancilla -> data for X-type (ancilla is control)
            self._emit_inner_x_cnots(circuit, block_id, data, x_anc)
            
            # Measure in X basis using MRX (avoids H gates)
            meas_start = self.ctx.add_measurement(self.r_x_in)
            circuit.append("MRX", x_anc)
            
            # Record measurements and emit anchors if requested
            for s_idx in range(self.r_x_in):
                meas_idx = meas_start + s_idx
                key = (block_id, s_idx)
                
                if emit_anchors:
                    coord = self._make_detector_coord(float(block_id), float(s_idx), float(self._round_number), float(DetectorRole.INNER_TEMPORAL.value))
                    self.ctx.emit_detector(circuit, [meas_idx], coord)
                
                self._last_inner_x_meas[key] = meas_idx
                
                if self._block_contiguous:
                    self._deferred_inner_x[block_id][s_idx].append(
                        (self._round_number, meas_idx)
                    )
    
    def _emit_inner_z_projection(
        self, 
        circuit: stim.Circuit, 
        emit_anchors: bool,
    ) -> None:
        """Emit inner Z-stabilizer measurements for all blocks."""
        for block_id in range(self.n_out):
            data = self.data_block_qubits(block_id)
            z_anc = self.inner_z_ancillas(block_id)
            
            # CNOTs: data -> ancilla for Z-type
            self._emit_inner_z_cnots(circuit, block_id, data, z_anc)
            circuit.append("TICK")
            
            # Measure
            meas_start = self.ctx.add_measurement(self.r_z_in)
            circuit.append("MR", z_anc)
            
            # Record measurements and emit anchors if requested
            for s_idx in range(self.r_z_in):
                meas_idx = meas_start + s_idx
                key = (block_id, s_idx)
                
                if emit_anchors:
                    coord = self._make_detector_coord(float(block_id), float(s_idx), float(self._round_number), float(DetectorRole.INNER_TEMPORAL.value))
                    self.ctx.emit_detector(circuit, [meas_idx], coord)
                
                self._last_inner_z_meas[key] = meas_idx
                
                if self._block_contiguous:
                    self._deferred_inner_z[block_id][s_idx].append(
                        (self._round_number, meas_idx)
                    )
    
    def _emit_inner_x_cnots(
        self,
        circuit: stim.Circuit,
        block_id: int,
        data_qubits: List[int],
        x_ancillas: List[int],
    ) -> None:
        """Emit CNOTs for inner X-stabilizer measurement on one block."""
        if self._hx_in is None or self._hx_in.size == 0:
            return
        
        cnot_pairs: List[Tuple[int, int]] = []
        for s_idx in range(min(self.r_x_in, len(x_ancillas))):
            anc = x_ancillas[s_idx]
            row = self._hx_in[s_idx]
            for q in range(min(self.n_in, len(row))):
                if row[q]:
                    # X-type: ancilla controls data
                    cnot_pairs.append((anc, data_qubits[q]))
        
        if cnot_pairs:
            layers = graph_coloring_cnots(cnot_pairs)
            for layer_idx, layer_cnots in enumerate(layers):
                if layer_idx > 0:
                    circuit.append("TICK")
                for ctrl, targ in layer_cnots:
                    circuit.append("CX", [ctrl, targ])
    
    def _emit_inner_z_cnots(
        self,
        circuit: stim.Circuit,
        block_id: int,
        data_qubits: List[int],
        z_ancillas: List[int],
    ) -> None:
        """Emit CNOTs for inner Z-stabilizer measurement on one block."""
        if self._hz_in is None or self._hz_in.size == 0:
            return
        
        cnot_pairs: List[Tuple[int, int]] = []
        for s_idx in range(min(self.r_z_in, len(z_ancillas))):
            anc = z_ancillas[s_idx]
            row = self._hz_in[s_idx]
            for q in range(min(self.n_in, len(row))):
                if row[q]:
                    # Z-type: data controls ancilla
                    cnot_pairs.append((data_qubits[q], anc))
        
        if cnot_pairs:
            layers = graph_coloring_cnots(cnot_pairs)
            for layer_idx, layer_cnots in enumerate(layers):
                if layer_idx > 0:
                    circuit.append("TICK")
                for ctrl, targ in layer_cnots:
                    circuit.append("CX", [ctrl, targ])
    
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
        Emit one full stabilizer measurement round (inner + outer).

        Accepts the same keyword arguments as :class:`CSSStabilizerRoundBuilder`
        so that :class:`MemoryRoundEmitter` can call this through the unified
        pipeline.  The ``emit_z_anchors``, ``emit_x_anchors``,
        ``explicit_anchor_mode``, and ``first_basis`` parameters are accepted
        for interface compatibility but are handled internally by the
        hierarchical builder's own anchor logic.

        For d_inner=1 (default), this behaves as before: inner stabs + outer stabs.
        For d_inner>1, use emit_outer_round() which handles the nested structure.
        """
        if self.d_inner == 1:
            # Legacy behavior: inner + outer in single round
            self._emit_inner_round(circuit, stab_type, emit_detectors)
            
            # Save pre-outer measurements for crossing detectors in next round
            self._save_pre_outer_inner_meas()
            
            # Outer stabilizers
            if stab_type in (StabilizerBasis.X, StabilizerBasis.BOTH) and self.r_x_out > 0:
                self._emit_outer_x_round(circuit, emit_detectors)
            if stab_type in (StabilizerBasis.Z, StabilizerBasis.BOTH) and self.r_z_out > 0:
                self._emit_outer_z_round(circuit, emit_detectors)
            
            # Track outer round for crossing detector logic
            self._outer_round_number += 1
            
            # Now that inner + outer are done, clear _skip_first_round
            self._skip_first_round = False

            self.ctx.advance_time()
            self._emit_shift_coords(circuit)
            self._round_number += 1
        else:
            # With d_inner > 1, emit a full outer round (d_inner inner rounds
            # plus outer X/Z stabilizers) so that a single call to emit_round()
            # from MemoryRoundEmitter produces one complete round.
            self.emit_outer_round(circuit, stab_type, emit_detectors)
    
    def emit_outer_round(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_detectors: bool = True,
    ) -> None:
        """
        Emit one complete outer round: d_inner inner rounds + outer X/Z.
        
        Structure: (d_inner) × [inner X, inner Z], outer X (destructive), outer Z (destructive)
        
        This is the primary method for hierarchical concatenated memory when d_inner > 1.
        
        Detector emission:
        - Inner X/Z: temporal within d_inner block, crossing at end
        - Outer X: temporal (no anchor for Z-basis)
        - Outer Z: anchor on first outer round, temporal thereafter
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        stab_type : StabilizerBasis
            Which stabilizers to measure.
        emit_detectors : bool
            Whether to emit detectors.
        """
        # Reset inner round counter for this outer round
        self._inner_round_in_block = 0
        
        # Phase 1: d_inner inner stabilizer rounds
        for inner_idx in range(self.d_inner):
            self._emit_inner_round(circuit, stab_type, emit_detectors)
            self._inner_round_in_block += 1
            
            # Advance time coordinate within inner block
            self.ctx.advance_time()
            self._emit_shift_coords(circuit)
            self._round_number += 1
        
        # Crossing detectors will be emitted AFTER outer operations complete
        # (they connect last inner round of this block to first inner round of next block)
        # Store the current inner measurements for crossing detector emission later
        self._save_pre_outer_inner_meas()
        
        # Phase 2: Outer X stabilizers via |+_L⟩ ancillas (destructive)
        if stab_type in (StabilizerBasis.X, StabilizerBasis.BOTH) and self.r_x_out > 0:
            self._emit_outer_x_round_destructive(circuit, emit_detectors)
        
        # Phase 3: Outer Z stabilizers via |0_L⟩ ancillas (destructive)
        if stab_type in (StabilizerBasis.Z, StabilizerBasis.BOTH) and self.r_z_out > 0:
            self._emit_outer_z_round_destructive(circuit, emit_detectors)
        
        # Increment outer round counter
        self._outer_round_number += 1
        
        # Now that all inner + outer rounds are done, clear _skip_first_round
        self._skip_first_round = False

        # Reset inner round counter for next outer round
        self._inner_round_in_block = 0
    
    def emit_inner_only_round(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis = StabilizerBasis.BOTH,
        emit_detectors: bool = True,
    ) -> None:
        """
        Emit a single inner-only syndrome round (no outer operations).

        Used for trailing inner rounds before final measurement to provide
        temporal depth at the temporal boundary.  The first trailing round
        after an outer round correctly gets crossing detectors (via
        _inner_round_in_block == 0 and _outer_round_number > 0).  Subsequent
        trailing rounds get temporal detectors.
        """
        self._emit_inner_round(circuit, stab_type, emit_detectors)
        self._inner_round_in_block += 1
        # Inner-only rounds should also clear _skip_first_round
        self._skip_first_round = False
        self.ctx.advance_time()
        self._emit_shift_coords(circuit)
        self._round_number += 1

    def _emit_inner_round(
        self,
        circuit: stim.Circuit,
        stab_type: StabilizerBasis,
        emit_detectors: bool,
    ) -> None:
        """
        Emit inner stabilizers on all data blocks.
        
        Order depends on measurement_basis for proper scheduling:
        - Z-basis: X first, Z second (Z measured right before outer Z ops)
        - X-basis: Z first, X second (X measured right before outer X ops)
        
        This ensures that the stabilizer matching the measurement basis is
        measured closest to the destructive measurement in that basis.
        """
        if self.measurement_basis == "Z":
            # X first, Z last - Z is closer to outer Z destructive MZ
            if stab_type in (StabilizerBasis.X, StabilizerBasis.BOTH):
                self._emit_inner_x_round(circuit, emit_detectors)
            if stab_type in (StabilizerBasis.Z, StabilizerBasis.BOTH):
                self._emit_inner_z_round(circuit, emit_detectors)
        else:
            # Z first, X last - X is closer to outer X destructive MX
            if stab_type in (StabilizerBasis.Z, StabilizerBasis.BOTH):
                self._emit_inner_z_round(circuit, emit_detectors)
            if stab_type in (StabilizerBasis.X, StabilizerBasis.BOTH):
                self._emit_inner_x_round(circuit, emit_detectors)

        # NOTE: _skip_first_round is NOT cleared here.  It is cleared at the
        # end of a *complete* stabiliser round (inner + outer) so that outer
        # anchor detectors are also properly suppressed.  The clearing
        # happens in emit_round / emit_outer_round after the outer rounds
        # have been emitted.
    
    def _emit_inner_x_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """Emit inner X-stabilizer measurements for all blocks."""
        for block_id in range(self.n_out):
            data = self.data_block_qubits(block_id)
            x_anc = self.inner_x_ancillas(block_id)
            
            # Reset X ancillas to |+⟩ using RX (avoids H gate)
            circuit.append("RX", x_anc)
            circuit.append("TICK")
            
            # CNOTs: ancilla controls data for X-type
            self._emit_inner_x_cnots(circuit, block_id, data, x_anc)
            
            # Measure in X basis using MRX (avoids H gates)
            meas_start = self.ctx.add_measurement(self.r_x_in)
            circuit.append("MRX", x_anc)
            
            # Process measurements
            for s_idx in range(self.r_x_in):
                meas_idx = meas_start + s_idx
                key = (block_id, s_idx)
                prev_meas = self._last_inner_x_meas.get(key)
                
                if self._block_contiguous:
                    self._deferred_inner_x[block_id][s_idx].append(
                        (self._round_number, meas_idx)
                    )
                elif emit_detectors:
                    self._emit_inner_x_detector(circuit, meas_idx, prev_meas, block_id, s_idx)
                
                self._last_inner_x_meas[key] = meas_idx
                self.ctx.record_stabilizer_measurement(
                    self.block_name, f"inner_x_{block_id}", s_idx, meas_idx
                )
    
    def _emit_inner_z_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """Emit inner Z-stabilizer measurements for all blocks."""
        for block_id in range(self.n_out):
            data = self.data_block_qubits(block_id)
            z_anc = self.inner_z_ancillas(block_id)
            
            # CNOTs
            self._emit_inner_z_cnots(circuit, block_id, data, z_anc)
            circuit.append("TICK")
            
            # Measure
            meas_start = self.ctx.add_measurement(self.r_z_in)
            circuit.append("MR", z_anc)
            
            # Process measurements
            for s_idx in range(self.r_z_in):
                meas_idx = meas_start + s_idx
                key = (block_id, s_idx)
                prev_meas = self._last_inner_z_meas.get(key)
                
                if self._block_contiguous:
                    self._deferred_inner_z[block_id][s_idx].append(
                        (self._round_number, meas_idx)
                    )
                elif emit_detectors:
                    self._emit_inner_z_detector(circuit, meas_idx, prev_meas, block_id, s_idx)
                
                self._last_inner_z_meas[key] = meas_idx
                self.ctx.record_stabilizer_measurement(
                    self.block_name, f"inner_z_{block_id}", s_idx, meas_idx
                )
    
    def _emit_outer_x_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """
        Emit outer X-stabilizer measurements via logical |+_L⟩ ancilla blocks.
        
        For each outer X-stabilizer j:
        1. Prepare ancilla block j in |+_L⟩ (RX, Z-stabs random, X-stabs anchor)
        2. Transversal CNOT: ancilla[j,i] -> data[k,i] for each outer_hx[j,k]=1
        3. Measure logical X on ancilla (MX on all qubits) to get outer X syndrome
        
        The outer X syndrome is XOR of measurements on inner logical X support.
        """
        # Prepare logical |+_L⟩ states on outer ancilla blocks
        for anc_idx in range(self.r_x_out):
            self._prepare_ancilla_block_plus_L(circuit, anc_idx, emit_inner_detectors=emit_detectors)
        
        # Transversal CNOTs for outer X stabilizers
        # outer_hx[j, k] = 1 means outer X-stab j acts on outer qubit k
        # For X-stab measurement via |+_L⟩: ancilla → data (ancilla is control)
        for stab_idx in range(self.r_x_out):
            row = self._hx_out[stab_idx]
            anc_data = self.outer_anc_block_data(stab_idx)
            
            for outer_q in range(self.n_out):
                if row[outer_q]:
                    # Transversal CNOT: anc[stab_idx, i] -> data[outer_q, i]
                    data_block = self.data_block_qubits(outer_q)
                    for i in range(self.n_in):
                        circuit.append("CX", [anc_data[i], data_block[i]])
            
            circuit.append("TICK")
        
        # Measure ancilla blocks to extract outer X syndrome
        for stab_idx in range(self.r_x_out):
            meas_indices = self._measure_ancilla_block_x(circuit, stab_idx)
            prev_meas = self._last_outer_x_meas.get(stab_idx)
            
            if self._block_contiguous:
                self._deferred_outer_x[stab_idx].append((self._round_number, meas_indices))
            elif emit_detectors:
                self._emit_outer_x_detector(circuit, meas_indices, prev_meas, stab_idx)
            
            self._last_outer_x_meas[stab_idx] = meas_indices
            # Record using first measurement for simple lookups
            self.ctx.record_stabilizer_measurement(
                self.block_name, "outer_x", stab_idx, meas_indices[0] if meas_indices else -1
            )
    
    def _emit_outer_z_round(self, circuit: stim.Circuit, emit_detectors: bool) -> None:
        """
        Emit outer Z-stabilizer measurements via logical |0_L⟩ ancilla blocks.
        
        For each outer Z-stabilizer j:
        1. Prepare ancilla block j in |0_L⟩ (R, X-stabs random, Z-stabs anchor)
        2. Transversal CNOT: data[k,i] -> ancilla[j,i] for each outer_hz[j,k]=1
        3. Measure logical Z on ancilla (MZ on all qubits) to get outer Z syndrome
        
        The outer Z syndrome is XOR of measurements on inner logical Z support.
        """
        # Prepare logical |0_L⟩ states on outer ancilla blocks
        for anc_idx in range(self.r_z_out):
            self._prepare_ancilla_block_zero_L(circuit, anc_idx, emit_inner_detectors=emit_detectors)
        
        # Transversal CNOTs for outer Z stabilizers
        # outer_hz[j, k] = 1 means outer Z-stab j acts on outer qubit k
        # For Z-stab measurement via |0_L⟩: data → ancilla (data is control)
        for stab_idx in range(self.r_z_out):
            row = self._hz_out[stab_idx]
            anc_data = self.outer_anc_block_data(stab_idx)
            
            for outer_q in range(self.n_out):
                if row[outer_q]:
                    # Transversal CNOT: data[outer_q, i] -> anc[stab_idx, i]
                    data_block = self.data_block_qubits(outer_q)
                    for i in range(self.n_in):
                        circuit.append("CX", [data_block[i], anc_data[i]])
            
            circuit.append("TICK")
        
        # Measure ancilla blocks to extract outer Z syndrome
        for stab_idx in range(self.r_z_out):
            meas_indices = self._measure_ancilla_block_z(circuit, stab_idx)
            prev_meas = self._last_outer_z_meas.get(stab_idx)
            
            if self._block_contiguous:
                self._deferred_outer_z[stab_idx].append((self._round_number, meas_indices))
            elif emit_detectors:
                self._emit_outer_z_detector(circuit, meas_indices, prev_meas, stab_idx)
            
            self._last_outer_z_meas[stab_idx] = meas_indices
            # Record using first measurement for simple lookups
            self.ctx.record_stabilizer_measurement(
                self.block_name, "outer_z", stab_idx, meas_indices[0] if meas_indices else -1
            )
    
    def _emit_outer_x_round_destructive(
        self, 
        circuit: stim.Circuit, 
        emit_detectors: bool
    ) -> None:
        """
        Emit outer X-stabilizer measurements via destructive |+_L⟩ ancilla blocks.
        
        For each outer X-stabilizer j:
        1. Prepare ancilla block j in |+_L⟩ (RX data, Z-stabs random, X-stabs anchor)
        2. Transversal CNOT: ancilla[j,i] -> data[k,i] for each outer_hx[j,k]=1
        3. MX ALL qubits on ancilla (not just logical support) - destructive
        4. Emit inner X boundary detectors on ancilla blocks
        
        The outer X syndrome is XOR of measurements on inner logical X support.
        Inner X boundary detectors compare MX vs last inner X-stab measurement.
        """
        # Prepare logical |+_L⟩ states on outer ancilla blocks
        # Inner X-stabs are anchors during prep (deterministic on |+⟩^⊗n)
        for anc_idx in range(self.r_x_out):
            self._prepare_ancilla_block_plus_L(circuit, anc_idx, emit_inner_detectors=emit_detectors)
        
        # Transversal CNOTs for outer X stabilizers
        # outer_hx[j, k] = 1 means outer X-stab j acts on outer qubit k
        # For X-stab measurement via |+_L⟩: ancilla → data (ancilla is control)
        for stab_idx in range(self.r_x_out):
            row = self._hx_out[stab_idx]
            anc_data = self.outer_anc_block_data(stab_idx)
            
            for outer_q in range(self.n_out):
                if row[outer_q]:
                    # Transversal CNOT: anc[stab_idx, i] -> data[outer_q, i]
                    data_block = self.data_block_qubits(outer_q)
                    for i in range(self.n_in):
                        circuit.append("CX", [anc_data[i], data_block[i]])
            
            circuit.append("TICK")
        
        # DESTRUCTIVE MEASUREMENT: MX on ALL qubits of each ancilla block
        for stab_idx in range(self.r_x_out):
            anc_data = self.outer_anc_block_data(stab_idx)
            x_anc = self.outer_anc_block_x_ancillas(stab_idx)
            z_anc = self.outer_anc_block_z_ancillas(stab_idx)
            
            # Measure ALL ancilla qubits in X basis
            all_anc_qubits = anc_data + x_anc + z_anc
            meas_start = self.ctx.add_measurement(len(all_anc_qubits))
            circuit.append("MX", all_anc_qubits)
            
            # Store destructive MX measurements for inner X-stab boundary detectors
            # These are needed for: (1) inner X boundary on outer X ancilla blocks
            #                       (2) inner X boundary on data blocks (X-basis memory)
            for s_idx in range(self.r_x_in):
                row = self._hx_in[s_idx]
                destruct_meas_indices = [meas_start + q for q in range(self.n_in) if row[q]]
                if destruct_meas_indices:
                    self._outer_x_anc_destruct_x[(stab_idx, s_idx)] = destruct_meas_indices
            
            # Extract outer X syndrome from logical X support
            # Outer X syndrome = XOR of MX on inner logical X support qubits
            logical_meas = [meas_start + q for q in self._inner_logical_x_support]
            
            # Emit inner X boundary detectors on ancilla data qubits
            # Each data qubit MX should match its expectation from last inner X-stab
            if emit_detectors:
                self._emit_outer_anc_x_boundary_detectors(
                    circuit, stab_idx, meas_start, len(anc_data)
                )
            
            # Outer X detector: temporal comparison
            prev_meas = self._last_outer_x_meas.get(stab_idx)
            if emit_detectors:
                self._emit_outer_x_detector(circuit, logical_meas, prev_meas, stab_idx)
            
            self._last_outer_x_meas[stab_idx] = logical_meas
            self.ctx.record_stabilizer_measurement(
                self.block_name, "outer_x", stab_idx, logical_meas[0] if logical_meas else -1
            )
    
    def _emit_outer_z_round_destructive(
        self, 
        circuit: stim.Circuit, 
        emit_detectors: bool
    ) -> None:
        """
        Emit outer Z-stabilizer measurements via destructive |0_L⟩ ancilla blocks.
        
        For each outer Z-stabilizer j:
        1. Prepare ancilla block j in |0_L⟩ (R data, X-stabs random, Z-stabs anchor)
        2. Transversal CNOT: data[k,i] -> ancilla[j,i] for each outer_hz[j,k]=1
        3. MZ ALL qubits on ancilla (not just logical support) - destructive
        4. Emit inner Z boundary detectors on ancilla blocks
        
        The outer Z syndrome is XOR of measurements on inner logical Z support.
        Inner Z boundary detectors compare MZ vs last inner Z-stab measurement.
        """
        # Prepare logical |0_L⟩ states on outer ancilla blocks
        # Inner Z-stabs are anchors during prep (deterministic on |0⟩^⊗n)
        for anc_idx in range(self.r_z_out):
            self._prepare_ancilla_block_zero_L(circuit, anc_idx, emit_inner_detectors=emit_detectors)
        
        # Transversal CNOTs for outer Z stabilizers
        # outer_hz[j, k] = 1 means outer Z-stab j acts on outer qubit k
        # For Z-stab measurement via |0_L⟩: data → ancilla (data is control)
        for stab_idx in range(self.r_z_out):
            row = self._hz_out[stab_idx]
            anc_data = self.outer_anc_block_data(stab_idx)
            
            for outer_q in range(self.n_out):
                if row[outer_q]:
                    # Transversal CNOT: data[outer_q, i] -> anc[stab_idx, i]
                    data_block = self.data_block_qubits(outer_q)
                    for i in range(self.n_in):
                        circuit.append("CX", [data_block[i], anc_data[i]])
            
            circuit.append("TICK")
        
        # DESTRUCTIVE MEASUREMENT: MZ on ALL qubits of each ancilla block
        for stab_idx in range(self.r_z_out):
            anc_data = self.outer_anc_block_data(stab_idx)
            x_anc = self.outer_anc_block_x_ancillas(stab_idx)
            z_anc = self.outer_anc_block_z_ancillas(stab_idx)
            
            # Measure ALL ancilla qubits in Z basis
            all_anc_qubits = anc_data + x_anc + z_anc
            meas_start = self.ctx.add_measurement(len(all_anc_qubits))
            circuit.append("M", all_anc_qubits)
            
            # Store destructive MZ measurements for inner Z-stab boundary detectors
            # These are needed for inner Z boundary on data blocks (Z-basis memory)
            for s_idx in range(self.r_z_in):
                row = self._hz_in[s_idx]
                destruct_meas_indices = [meas_start + q for q in range(self.n_in) if row[q]]
                if destruct_meas_indices:
                    self._outer_z_anc_destruct_z[(stab_idx, s_idx)] = destruct_meas_indices
            
            # Extract outer Z syndrome from logical Z support
            # Outer Z syndrome = XOR of MZ on inner logical Z support qubits
            logical_meas = [meas_start + q for q in self._inner_logical_z_support]
            
            # Emit inner Z boundary detectors on ancilla data qubits
            if emit_detectors:
                self._emit_outer_anc_z_boundary_detectors(
                    circuit, stab_idx, meas_start, len(anc_data)
                )
            
            # Outer Z detector: anchor on first outer round, temporal thereafter
            prev_meas = self._last_outer_z_meas.get(stab_idx)
            if emit_detectors:
                if self._outer_round_number == 0:
                    # First outer round: anchor detector for Z-basis memory
                    # |0_L⟩ data → outer Z-stab should be deterministic +1
                    # Skip when _skip_first_round is set (post-gadget: the
                    # data qubits are no longer in a fresh eigenstate after a
                    # transversal gate, so the 1-term anchor would be
                    # non-deterministic).
                    if self.measurement_basis == "Z" and not self._skip_first_round:
                        coord = self._make_detector_coord(float(self.n_out + 10), float(stab_idx), float(self._round_number), float(DetectorRole.OUTER_TEMPORAL.value))
                        self.ctx.emit_detector(circuit, logical_meas, coord)
                    elif prev_meas is not None:
                        self._emit_outer_z_detector(circuit, logical_meas, prev_meas, stab_idx)
                else:
                    self._emit_outer_z_detector(circuit, logical_meas, prev_meas, stab_idx)
            
            self._last_outer_z_meas[stab_idx] = logical_meas
            self.ctx.record_stabilizer_measurement(
                self.block_name, "outer_z", stab_idx, logical_meas[0] if logical_meas else -1
            )
    
    def _emit_outer_anc_x_boundary_detectors(
        self,
        circuit: stim.Circuit,
        anc_idx: int,
        meas_start: int,
        n_data: int,
    ) -> None:
        """
        Emit inner X boundary detectors on an outer ancilla block after destructive MX.
        
        For ancilla blocks prepared in |+_L⟩:
        - Inner X-stabs were deterministic +1 during prep (anchors emitted then)
        - CX[anc→data] propagates X FROM ancilla TO main data blocks:
          X_anc → X_anc ⊗ X_main_data
        - Destructive MX now measures X_anc ⊗ X_main_data
        
        Multi-term formula:
        X_prep(anc, s) ⊕ Σ_{k: outer_hx[anc,k]=1} pre_outer_X(data_k, s) ⊕ X_destruct(anc, s) = 0
        
        Where:
        - X_prep: inner X-stab measurement during ancilla prep
        - pre_outer_X: last inner X-stab on each data block before outer operations
        - X_destruct: XOR of MX on ancilla data qubits in inner X-stab support
        """
        # Get the data blocks that participate in this outer X stabilizer
        data_blocks = self._data_blocks_in_outer_x_stab(anc_idx)
        
        for s_idx in range(self.r_x_in):
            # Get prep measurement for this inner X stabilizer
            prep_meas = self._last_anc_inner_x_meas_prep.get((anc_idx, s_idx))
            if prep_meas is None:
                continue
            
            # Get destructive MX measurements for this stabilizer's support
            row = self._hx_in[s_idx]
            destruct_meas = [meas_start + q for q in range(self.n_in) if row[q]]
            
            if not destruct_meas:
                continue
            
            # Collect all measurement terms
            all_meas = [prep_meas] + destruct_meas
            
            # Add pre-outer inner X measurements from each data block in the outer X stab
            for data_block in data_blocks:
                data_x_meas = self._pre_outer_inner_x_meas.get((data_block, s_idx))
                if data_x_meas is not None:
                    all_meas.append(data_x_meas)
            
            # Boundary detector: multi-term
            coord = self._make_detector_coord(float(self.n_out + 20 + anc_idx), float(s_idx), self.ctx.current_time, float(DetectorRole.OAB_BOUNDARY.value))
            self.ctx.emit_detector(circuit, all_meas, coord)
    
    def _emit_outer_anc_z_boundary_detectors(
        self,
        circuit: stim.Circuit,
        anc_idx: int,
        meas_start: int,
        n_data: int,
    ) -> None:
        """
        Emit inner Z boundary detectors on an outer ancilla block after destructive MZ.
        
        For ancilla blocks prepared in |0_L⟩:
        - Inner Z-stabs were deterministic +1 during prep (anchors emitted then)
        - CX[data→anc] propagates Z FROM data TO ancilla via backward propagation:
          Z_anc → Z_anc ⊗ Z_data (for each data block in outer Z stab support)
        - Destructive MZ measures Z on all qubits
        
        IMPORTANT: The outer X CX (CX[outer_x_anc → data]) happens BEFORE the
        outer Z CX (CX[data → outer_z_anc]). The outer X CX back-propagates Z:
          Z_data → Z_data ⊗ Z_outer_x_anc
        So when outer Z CX then propagates Z_data to the outer Z ancilla, the
        Z_outer_x_anc component comes along. The boundary detector must include
        the outer X ancilla's Z-prep measurements to cancel this extra term.
        
        Formula:
          prep_Z(j,s) ⊕ Σ_{k: outer_hz[j,k]=1} [ pre_outer_Z(data_k, s)
                        ⊕ Σ_{m: outer_hx[m,k]=1} Z_prep(outer_x_anc_m, s) ]
                       ⊕ destruct_Z(j,s) = 0
        
        This is a multi-term detector:
        - prep Z-stab on ancilla
        - pre-outer inner Z on each data block in the outer Z stab support
        - outer X ancilla Z-prep for each outer X stab touching each data block
        - destructive MZ parity on inner Z support
        """
        # Get the data blocks that participate in this outer Z stabilizer
        data_blocks = self._data_blocks_in_outer_z_stab(anc_idx)
        
        for s_idx in range(self.r_z_in):
            # Get prep measurement for this inner Z stabilizer
            prep_meas = self._last_anc_inner_z_meas_prep.get((anc_idx, s_idx))
            if prep_meas is None:
                continue
            
            # Get destructive MZ measurements for this stabilizer's support
            row = self._hz_in[s_idx]
            destruct_meas = [meas_start + q for q in range(self.n_in) if row[q]]
            
            if not destruct_meas:
                continue
            
            # Collect all measurement terms
            all_meas = [prep_meas] + destruct_meas
            
            # Add pre-outer inner Z measurements from each data block in the outer Z stab
            for data_block in data_blocks:
                data_z_meas = self._pre_outer_inner_z_meas.get((data_block, s_idx))
                if data_z_meas is not None:
                    all_meas.append(data_z_meas)
                
                # Add outer X ancilla Z-prep correction for this data block.
                # The outer X CX (CX[anc→data]) back-propagates Z onto outer X
                # ancilla qubits. When outer Z CX then propagates Z_data to the
                # outer Z ancilla, the extra Z_outer_x_anc comes along and must
                # be cancelled by including the Z-stab prep measurement from the
                # outer X ancilla block.
                for m in self._outer_x_stabs_for_block(data_block):
                    anc_z_prep = self._outer_x_anc_z_prep.get((m, s_idx))
                    if anc_z_prep is not None:
                        all_meas.append(anc_z_prep)
            
            # Boundary detector: multi-term
            coord = self._make_detector_coord(float(self.n_out + 30 + anc_idx), float(s_idx), self.ctx.current_time, float(DetectorRole.OAB_BOUNDARY.value))
            self.ctx.emit_detector(circuit, all_meas, coord)


    def _prepare_ancilla_block_plus_L(
        self, 
        circuit: stim.Circuit, 
        anc_idx: int,
        emit_inner_detectors: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """
        Prepare an outer ancilla block in logical |+_L⟩ state.
        
        |+_L⟩ preparation for X-basis logical ancilla:
        1. Reset all data qubits to |+⟩ using RX (now |+⟩^⊗n, avoids H gate)
        2. Measure Z-stabilizers (RANDOM, projects into codespace)
        3. Measure X-stabilizers (DETERMINISTIC +1 on |+⟩^⊗n after projection)
           → emit anchor detectors for X-stabs
        
        Returns
        -------
        Tuple[List[int], List[int]]
            (x_stab_meas_indices, z_stab_meas_indices) for syndrome tracking
        """
        data = self.outer_anc_block_data(anc_idx)
        x_anc = self.outer_anc_block_x_ancillas(anc_idx)
        z_anc = self.outer_anc_block_z_ancillas(anc_idx)
        
        # Reset data to |+⟩ using RX (avoids H gate)
        circuit.append("RX", data)
        # Reset ancillas to |0⟩
        circuit.append("R", x_anc + z_anc)
        circuit.append("TICK")
        
        # Measure Z-stabs FIRST (random, establishes frame - NO detector)
        z_meas = self._emit_z_stabs_on_block(circuit, data, z_anc, emit_detectors=False, anc_block_idx=anc_idx)
        
        # Measure X-stabs SECOND (deterministic +1 on |+⟩^⊗n - anchor detectors)
        x_meas = self._emit_x_stabs_on_block(circuit, data, x_anc, emit_detectors=emit_inner_detectors, anc_block_idx=anc_idx)
        
        # Store X-stab measurements for boundary detectors after destructive MX
        for s_idx, m_idx in enumerate(x_meas):
            self._last_anc_inner_x_meas_prep[(anc_idx, s_idx)] = m_idx
        
        # Store Z-stab measurements for crossing detectors
        # Inner Z crossing on data blocks needs these to account for CX propagation
        for s_idx, m_idx in enumerate(z_meas):
            self._outer_x_anc_z_prep[(anc_idx, s_idx)] = m_idx
        
        return x_meas, z_meas
    
    def _prepare_ancilla_block_zero_L(
        self, 
        circuit: stim.Circuit, 
        anc_idx: int,
        emit_inner_detectors: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """
        Prepare an outer ancilla block in logical |0_L⟩ state.
        
        |0_L⟩ preparation for Z-basis logical ancilla:
        1. Reset all qubits to |0⟩ (now |0⟩^⊗n)
        2. Measure X-stabilizers (RANDOM, projects into codespace)
        3. Measure Z-stabilizers (DETERMINISTIC +1 on |0⟩^⊗n after projection)
           → emit anchor detectors for Z-stabs
        
        Returns
        -------
        Tuple[List[int], List[int]]
            (x_stab_meas_indices, z_stab_meas_indices) for syndrome tracking
        """
        data = self.outer_anc_block_data(anc_idx)
        x_anc = self.outer_anc_block_x_ancillas(anc_idx)
        z_anc = self.outer_anc_block_z_ancillas(anc_idx)
        
        # Reset all to |0⟩
        circuit.append("R", data + x_anc + z_anc)
        circuit.append("TICK")
        
        # Measure X-stabs FIRST (random, establishes frame - NO detector)
        x_meas = self._emit_x_stabs_on_block(circuit, data, x_anc, emit_detectors=False, anc_block_idx=anc_idx)
        
        # Measure Z-stabs SECOND (deterministic +1 on |0⟩^⊗n - anchor detectors)
        z_meas = self._emit_z_stabs_on_block(circuit, data, z_anc, emit_detectors=emit_inner_detectors, anc_block_idx=anc_idx)
        
        # Store Z-stab measurements for boundary detectors after destructive MZ
        for s_idx, m_idx in enumerate(z_meas):
            self._last_anc_inner_z_meas_prep[(anc_idx, s_idx)] = m_idx
        
        # Store X-stab measurements for crossing detectors
        # Inner X crossing on data blocks needs these to account for CX propagation
        for s_idx, m_idx in enumerate(x_meas):
            self._outer_z_anc_x_prep[(anc_idx, s_idx)] = m_idx
        
        return x_meas, z_meas
    
    def _emit_x_stabs_on_block(
        self,
        circuit: stim.Circuit,
        data: List[int],
        x_anc: List[int],
        emit_detectors: bool,
        anc_block_idx: Optional[int] = None,
    ) -> List[int]:
        """
        Emit X-stabilizer measurements on a logical block.
        
        Uses RX/MRX to avoid H gates.
        If emit_detectors=True, emits anchor detectors (for deterministic measurements).
        
        Parameters
        ----------
        anc_block_idx : int, optional
            Index of the ancilla block when called for outer ancilla preparation.
            Used to create unique coordinates for different ancilla blocks.
        """
        if self._hx_in is None or self.r_x_in == 0:
            return []
        
        # Reset X ancillas to |+⟩ using RX (avoids H gate)
        circuit.append("RX", x_anc)
        circuit.append("TICK")
        
        # CNOTs: ancilla controls data for X-type
        cnot_pairs: List[Tuple[int, int]] = []
        for s_idx in range(self.r_x_in):
            anc = x_anc[s_idx]
            row = self._hx_in[s_idx]
            for q in range(self.n_in):
                if row[q]:
                    cnot_pairs.append((anc, data[q]))
        
        if cnot_pairs:
            layers = graph_coloring_cnots(cnot_pairs)
            for layer_idx, layer_cnots in enumerate(layers):
                if layer_idx > 0:
                    circuit.append("TICK")
                for ctrl, targ in layer_cnots:
                    circuit.append("CX", [ctrl, targ])
        
        circuit.append("TICK")
        
        # Measure in X basis using MRX (avoids H gates)
        meas_start = self.ctx.add_measurement(self.r_x_in)
        circuit.append("MRX", x_anc)
        
        # Emit anchor detectors if requested (for deterministic |+⟩^⊗n state)
        if emit_detectors:
            for s_idx in range(self.r_x_in):
                meas_idx = meas_start + s_idx
                # Coordinate: use ancilla block index + stabilizer index
                # For outer ancilla blocks, use n_out + 1 + anc_block_idx to differentiate
                anc_offset = 0 if anc_block_idx is None else anc_block_idx
                coord = self._make_detector_coord(float(self.n_out + 1 + anc_offset), float(s_idx), float(self._round_number), float(DetectorRole.OAB_BOUNDARY.value))
                self.ctx.emit_detector(circuit, [meas_idx], coord)
        
        return list(range(meas_start, meas_start + self.r_x_in))
    
    def _emit_z_stabs_on_block(
        self,
        circuit: stim.Circuit,
        data: List[int],
        z_anc: List[int],
        emit_detectors: bool,
        anc_block_idx: Optional[int] = None,
    ) -> List[int]:
        """
        Emit Z-stabilizer measurements on a logical block.
        
        If emit_detectors=True, emits anchor detectors (for deterministic measurements).
        
        Parameters
        ----------
        anc_block_idx : int, optional
            Index of the ancilla block when called for outer ancilla preparation.
            Used to create unique coordinates for different ancilla blocks.
        """
        if self._hz_in is None or self.r_z_in == 0:
            return []
        
        # Reset Z ancillas to |0⟩ (already in |0⟩ from block reset, but explicit)
        # CNOTs: data controls ancilla for Z-type
        cnot_pairs: List[Tuple[int, int]] = []
        for s_idx in range(self.r_z_in):
            anc = z_anc[s_idx]
            row = self._hz_in[s_idx]
            for q in range(self.n_in):
                if row[q]:
                    cnot_pairs.append((data[q], anc))
        
        if cnot_pairs:
            layers = graph_coloring_cnots(cnot_pairs)
            for layer_idx, layer_cnots in enumerate(layers):
                if layer_idx > 0:
                    circuit.append("TICK")
                for ctrl, targ in layer_cnots:
                    circuit.append("CX", [ctrl, targ])
        
        circuit.append("TICK")
        
        # Measure in Z basis
        meas_start = self.ctx.add_measurement(self.r_z_in)
        circuit.append("MR", z_anc)
        
        # Emit anchor detectors if requested (for deterministic |0⟩^⊗n state)
        if emit_detectors:
            # Offset Z stabilizer indices by r_x_in to avoid coord collision with X stabilizers
            z_offset = self.r_x_in
            for s_idx in range(self.r_z_in):
                meas_idx = meas_start + s_idx
                # Coordinate: use ancilla block index + stabilizer index
                # For outer ancilla blocks, use n_out + 1 + anc_block_idx to differentiate
                anc_offset = 0 if anc_block_idx is None else anc_block_idx
                coord = self._make_detector_coord(float(self.n_out + 1 + anc_offset), float(s_idx + z_offset), float(self._round_number), float(DetectorRole.OAB_BOUNDARY.value))
                self.ctx.emit_detector(circuit, [meas_idx], coord)
        
        return list(range(meas_start, meas_start + self.r_z_in))
    
    def _measure_ancilla_block_x(self, circuit: stim.Circuit, anc_idx: int) -> List[int]:
        """
        Measure logical X on an ancilla block (for outer X syndrome).
        
        After transversal CNOT, the ancilla's logical X parity encodes the
        outer X syndrome. We measure all ancilla data qubits in X basis.
        
        The outer X syndrome is the XOR of measurements on the inner logical X support.
        This returns ALL measurement indices in the logical support so the detector
        can XOR them properly (not decoded - flat model doesn't need decode).
        
        Returns
        -------
        List[int]
            Measurement indices for the inner logical X support.
            XOR of these gives the outer X syndrome bit.
        """
        data = self.outer_anc_block_data(anc_idx)
        
        # Measure in X basis using MX (avoids H gate)
        meas_start = self.ctx.add_measurement(self.n_in)
        circuit.append("MX", data)
        
        # Return measurement indices for qubits in the inner logical X support
        # The outer X syndrome = XOR of these measurements
        logical_meas = [meas_start + q for q in self._inner_logical_x_support]
        return logical_meas
    
    def _measure_ancilla_block_z(self, circuit: stim.Circuit, anc_idx: int) -> List[int]:
        """
        Measure logical Z on an ancilla block (for outer Z syndrome).
        
        After transversal CNOT, the ancilla's logical Z parity encodes the
        outer Z syndrome. We measure all ancilla data qubits in Z basis.
        
        The outer Z syndrome is the XOR of measurements on the inner logical Z support.
        This returns ALL measurement indices in the logical support so the detector
        can XOR them properly (not decoded - flat model doesn't need decode).
        
        Returns
        -------
        List[int]
            Measurement indices for the inner logical Z support.
            XOR of these gives the outer Z syndrome bit.
        """
        data = self.outer_anc_block_data(anc_idx)
        
        # Measure in Z basis
        meas_start = self.ctx.add_measurement(self.n_in)
        circuit.append("M", data)
        
        # Return measurement indices for qubits in the inner logical Z support
        # The outer Z syndrome = XOR of these measurements
        logical_meas = [meas_start + q for q in self._inner_logical_z_support]
        return logical_meas
    
    # =========================================================================
    # Detector Emission
    # =========================================================================
    
    def _emit_inner_x_detector(
        self,
        circuit: stim.Circuit,
        cur_meas: int,
        prev_meas: Optional[int],
        block_id: int,
        local_idx: int,
    ) -> None:
        """
        Emit detector for inner X stabilizer on data block.
        
        For Z-basis memory (|0⟩ prep):
        - Round 0: NO anchor (X-stab outcome is RANDOM on |0⟩^⊗n)
        - Rounds 1+: TEMPORAL if d_inner > 1 and within same outer block
        - Crossing: Multi-term across outer block boundary
        - Boundary: NO (X anti-commutes with final MZ)
        
        For X-basis memory (|+⟩ prep):
        - Round 0: ANCHOR (X-stab is +1 eigenstate of |+⟩^⊗n)
        - Rounds 1+: TEMPORAL if d_inner > 1 and within same outer block
        - Crossing: Multi-term across outer block boundary
        - Boundary: YES (X commutes with final MX)
        
        Crossing detector formula:
        X propagates through CX[data→outer_z_anc] control to target.
        To cancel: pre_X(b,s) ⊕ Σ_{j: outer_hz[j,b]=1} X_prep(outer_z_anc_j, s) ⊕ post_X(b,s) = 0
        """
        R_T = float(DetectorRole.INNER_TEMPORAL.value)
        R_C = float(DetectorRole.INNER_CROSSING.value)
        coord = self._make_detector_coord(float(block_id), float(local_idx), self.ctx.current_time, R_T)
        
        if prev_meas is None:
            # First round: only emit anchor if X-basis memory (|+⟩ prep)
            # BUT suppress if _skip_first_round (post-gadget: not a fresh eigenstate)
            if self.measurement_basis == "X" and not self._skip_first_round:
                self.ctx.emit_detector(circuit, [cur_meas], coord)
            # For Z-basis: no anchor (X-stab is random on |0⟩)
        else:
            # Temporal or crossing detectors
            if self.d_inner > 1:
                if self._inner_round_in_block > 0:
                    # Within d_inner block: simple 2-term temporal
                    self.ctx.emit_detector(circuit, [cur_meas, prev_meas], coord)
                else:
                    # First inner round of new outer block (_inner_round_in_block == 0)
                    # Need crossing detector with multi-term formula
                    # prev_meas is from last inner round of previous outer block
                    pre_meas = self._pre_outer_inner_x_meas.get((block_id, local_idx))
                    if pre_meas is not None and self._outer_round_number > 0:
                        # Multi-term crossing: pre ⊕ Σ(outer_z_anc_x_prep) ⊕ post
                        all_meas = [cur_meas, pre_meas]
                        
                        # Add X-stab prep from each outer Z ancilla block that couples to this data block
                        for j in self._outer_z_stabs_for_block(block_id):
                            anc_x_prep = self._outer_z_anc_x_prep.get((j, local_idx))
                            if anc_x_prep is not None:
                                all_meas.append(anc_x_prep)
                        
                        # Use distinct coord for crossing detector
                        cross_coord = self._make_detector_coord(float(block_id), float(local_idx + 1000), self.ctx.current_time, R_C)
                        self.ctx.emit_detector(circuit, all_meas, cross_coord)
                    elif self._outer_round_number == 0 and prev_meas is not None:
                        # First outer round, first inner round: simple temporal from init projection.
                        # No outer CX has occurred yet, so no crossing correction needed.
                        self.ctx.emit_detector(circuit, [cur_meas, prev_meas], coord)
            else:
                # d_inner = 1: every inner measurement spans an outer operation
                if self._outer_round_number == 0 and prev_meas is not None:
                    # First round: simple temporal from init projection (no outer ops yet)
                    self.ctx.emit_detector(circuit, [cur_meas, prev_meas], coord)
                elif self._outer_round_number >= 1 and prev_meas is not None:
                    # Crossing detector: outer ops occurred between prev and cur
                    pre_meas = self._pre_outer_inner_x_meas.get((block_id, local_idx))
                    if pre_meas is not None:
                        all_meas = [cur_meas, pre_meas]
                        for j in self._outer_z_stabs_for_block(block_id):
                            anc_x_prep = self._outer_z_anc_x_prep.get((j, local_idx))
                            if anc_x_prep is not None:
                                all_meas.append(anc_x_prep)
                        cross_coord = self._make_detector_coord(float(block_id), float(local_idx + 1000), self.ctx.current_time, R_C)
                        self.ctx.emit_detector(circuit, all_meas, cross_coord)
    
    def _emit_inner_z_detector(
        self,
        circuit: stim.Circuit,
        cur_meas: int,
        prev_meas: Optional[int],
        block_id: int,
        local_idx: int,
    ) -> None:
        """
        Emit detector for inner Z stabilizer on data block.
        
        For Z-basis memory (|0⟩ prep):
        - Round 0: ANCHOR (Z-stab is +1 eigenstate of |0⟩^⊗n)
        - Rounds 1+: TEMPORAL if d_inner > 1 and within same outer block
        - Crossing: Multi-term across outer block boundary
        - Boundary: CONDITIONAL (handled in space-like detectors)
        
        For X-basis memory (|+⟩ prep):
        - Round 0: NO anchor (Z-stab outcome is RANDOM on |+⟩^⊗n)
        - Rounds 1+: TEMPORAL if d_inner > 1 and within same outer block
        - Crossing: Multi-term across outer block boundary
        - Boundary: NO (Z anti-commutes with final MX)
        
        Crossing detector formula:
        Z_data backward-propagates through CX[outer_x_anc→data] to Z_outer_x_anc.
        To cancel: pre_Z(b,s) ⊕ Σ_{j: outer_hx[j,b]=1} Z_prep(outer_x_anc_j, s) ⊕ post_Z(b,s) = 0
        """
        R_T = float(DetectorRole.INNER_TEMPORAL.value)
        R_C = float(DetectorRole.INNER_CROSSING.value)
        # Offset Z stabilizer indices by r_x_in to avoid coord collision with X stabilizers
        z_offset = self.r_x_in
        coord = self._make_detector_coord(float(block_id), float(local_idx + z_offset), self.ctx.current_time, R_T)
        
        if prev_meas is None:
            # First round: only emit anchor if Z-basis memory (|0⟩ prep)
            # BUT suppress if _skip_first_round (post-gadget: not a fresh eigenstate)
            if self.measurement_basis == "Z" and not self._skip_first_round:
                self.ctx.emit_detector(circuit, [cur_meas], coord)
            # For X-basis: no anchor (Z-stab is random on |+⟩)
        else:
            # Temporal or crossing detectors
            if self.d_inner > 1:
                if self._inner_round_in_block > 0:
                    # Within d_inner block: simple 2-term temporal
                    self.ctx.emit_detector(circuit, [cur_meas, prev_meas], coord)
                else:
                    # First inner round of new outer block (_inner_round_in_block == 0)
                    # Need crossing detector with multi-term formula
                    pre_meas = self._pre_outer_inner_z_meas.get((block_id, local_idx))
                    if pre_meas is not None and self._outer_round_number > 0:
                        # Multi-term crossing: pre ⊕ Σ(outer_x_anc_z_prep) ⊕ post
                        all_meas = [cur_meas, pre_meas]
                        
                        # Add Z-stab prep from each outer X ancilla block that couples to this data block
                        for j in self._outer_x_stabs_for_block(block_id):
                            anc_z_prep = self._outer_x_anc_z_prep.get((j, local_idx))
                            if anc_z_prep is not None:
                                all_meas.append(anc_z_prep)
                        
                        # Use distinct coord for crossing detector (offset by z_offset + 1000)
                        cross_coord = self._make_detector_coord(float(block_id), float(local_idx + z_offset + 1000), self.ctx.current_time, R_C)
                        self.ctx.emit_detector(circuit, all_meas, cross_coord)
                    elif self._outer_round_number == 0 and prev_meas is not None:
                        # First outer round, first inner round: simple temporal from init projection.
                        # No outer CX has occurred yet, so no crossing correction needed.
                        self.ctx.emit_detector(circuit, [cur_meas, prev_meas], coord)
            else:
                # d_inner = 1: every inner measurement spans an outer operation
                if self._outer_round_number == 0 and prev_meas is not None:
                    # First round: simple temporal from init projection (no outer ops yet)
                    self.ctx.emit_detector(circuit, [cur_meas, prev_meas], coord)
                elif self._outer_round_number >= 1 and prev_meas is not None:
                    # Crossing detector: outer ops occurred between prev and cur
                    pre_meas = self._pre_outer_inner_z_meas.get((block_id, local_idx))
                    if pre_meas is not None:
                        all_meas = [cur_meas, pre_meas]
                        for j in self._outer_x_stabs_for_block(block_id):
                            anc_z_prep = self._outer_x_anc_z_prep.get((j, local_idx))
                            if anc_z_prep is not None:
                                all_meas.append(anc_z_prep)
                        cross_coord = self._make_detector_coord(float(block_id), float(local_idx + z_offset + 1000), self.ctx.current_time, R_C)
                        self.ctx.emit_detector(circuit, all_meas, cross_coord)
    
    def _emit_outer_x_detector(
        self,
        circuit: stim.Circuit,
        cur_meas: List[int],
        prev_meas: Optional[List[int]],
        local_idx: int,
    ) -> None:
        """
        Emit detector for outer X stabilizer.
        
        The outer X syndrome is XOR of all measurements in cur_meas (logical X support).
        For temporal detectors, we XOR current syndrome with previous syndrome.
        """
        R_O = float(DetectorRole.OUTER_TEMPORAL.value)
        # Use n_out as block coord to distinguish from inner blocks
        coord = self._make_detector_coord(float(self.n_out), float(local_idx), self.ctx.current_time, R_O)
        
        if prev_meas is None:
            # First round (anchor): only emit if X-basis memory
            # Suppress if _skip_first_round (post-gadget)
            if self.measurement_basis == "X" and not self._skip_first_round:
                # Anchor detector: syndrome should be 0 for |+⟩_L state
                self.ctx.emit_detector(circuit, cur_meas, coord)
        else:
            # Temporal detector: XOR current and previous syndromes
            # All measurements in both lists contribute to the XOR
            all_meas = list(cur_meas) + list(prev_meas)
            self.ctx.emit_detector(circuit, all_meas, coord)
    
    def _emit_outer_z_detector(
        self,
        circuit: stim.Circuit,
        cur_meas: List[int],
        prev_meas: Optional[List[int]],
        local_idx: int,
    ) -> None:
        """
        Emit detector for outer Z stabilizer.
        
        The outer Z syndrome is XOR of all measurements in cur_meas (logical Z support).
        For temporal detectors, we XOR current syndrome with previous syndrome.
        """
        R_O = float(DetectorRole.OUTER_TEMPORAL.value)
        # Offset Z stabilizer indices by r_x_out to avoid coord collision with X stabilizers
        z_offset = self.r_x_out
        coord = self._make_detector_coord(float(self.n_out), float(local_idx + z_offset), self.ctx.current_time, R_O)
        
        if prev_meas is None:
            # First round (anchor): only emit if Z-basis memory
            # Suppress if _skip_first_round (post-gadget)
            if self.measurement_basis == "Z" and not self._skip_first_round:
                # Anchor detector: syndrome should be 0 for |0⟩_L state
                self.ctx.emit_detector(circuit, cur_meas, coord)
        else:
            # Temporal detector: XOR current and previous syndromes
            all_meas = list(cur_meas) + list(prev_meas)
            self.ctx.emit_detector(circuit, all_meas, coord)
    
    def _emit_shift_coords(self, circuit: stim.Circuit) -> None:
        """Emit SHIFT_COORDS to advance time coordinate."""
        circuit.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0, 0.0, 0.0])
    
    # =========================================================================
    # Final Measurement
    # =========================================================================
    
    def emit_final_measurement(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
        logical_idx: int = 0,
        emit_inner_observables: bool = False,
        skip_observable: bool = False,
        defer_boundary_detectors: bool = False,
    ) -> List[int]:
        """
        Emit final data qubit measurements with proper boundary detectors.
        
        For Z-basis memory:
        - Last syndrome round should have Z-stabs measured last
        - Final MZ on data qubits
        - Emit inner Z boundary detectors (last Z-syndrome XOR data measurements)
        - Emit outer Z boundary detectors (last outer Z XOR decoded logical Z from blocks)
        
        For X-basis memory:
        - Last syndrome round should have X-stabs measured last  
        - Final MX on data qubits
        - Emit inner X boundary detectors
        - Emit outer X boundary detectors
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        basis : str
            Measurement basis ("Z" or "X").
        logical_idx : int
            Which logical qubit.
        emit_inner_observables : bool
            If True, also register per-block inner logical observables
            at indices 1..n_out.  Observable ``1+b`` tracks the inner
            logical of block *b*, enabling hierarchical decoding with
            per-block sub-DEMs that each have their own observable.
        skip_observable : bool
            If True, skip registering this block's measurements for the
            logical observable (used when ObservableConfig says this block
            should not contribute to the observable).
        defer_boundary_detectors : bool
            If True, do not emit boundary detectors now. Instead, save the
            data_meas mapping and call ``emit_deferred_boundary_detectors()``
            later after all blocks have finished their final measurements.
            This is required for multi-block gadgets where lookbacks must be
            computed after ALL blocks have measured.
            
        Returns
        -------
        List[int]
            Measurement indices for the logical observable.
        """
        from .utils import get_logical_support
        
        # Update pre-outer measurements to reference the most recent inner
        # measurements.  When trailing inner rounds have been emitted after
        # the last outer round, this points the boundary detectors at the
        # trailing measurements so no crossing corrections are needed.
        self._save_pre_outer_inner_meas()
        
        data = self.data_qubits
        n = len(data)
        basis = basis.upper()
        
        # Use MX for X-basis measurement (avoids H gate)
        if basis == "X":
            meas_start = self.ctx.add_measurement(n)
            circuit.append("MX", data)
        else:
            meas_start = self.ctx.add_measurement(n)
            circuit.append("M", data)
        
        # Build data measurement lookup for space-like detectors
        data_meas = {q: meas_start + i for i, q in enumerate(range(n))}
        
        # Emit space-like (boundary) detectors or defer for later
        if defer_boundary_detectors:
            # Save state for deferred emission after all blocks measured
            self._deferred_boundary_basis = basis
            self._deferred_boundary_data_meas = data_meas
            self._deferred_boundary_meas_start = meas_start
        elif not self._block_contiguous:
            self._emit_all_space_like_detectors(circuit, basis, data_meas, meas_start)
        
        # Compute logical observable
        if not skip_observable:
            effective_basis = self.ctx.get_transformed_basis(logical_idx, basis)
            logical_support = get_logical_support(self.code, effective_basis, logical_idx)
            
            if logical_support:
                logical_meas = [meas_start + q for q in logical_support if q < n]
            else:
                logical_meas = [meas_start]
            
            self.ctx.add_observable_measurement(logical_idx, logical_meas)
        else:
            logical_meas = []
        
        # ---- per-block inner logical observables ----
        if emit_inner_observables:
            # Choose inner logical support based on measurement basis.
            if basis == "Z":
                inner_support = self._inner_logical_z_support
            else:
                inner_support = self._inner_logical_x_support
            for b in range(self.n_out):
                block_meas = [
                    meas_start + b * self.n_in + q for q in inner_support
                ]
                # Observable index 1+b (index 0 is the global observable)
                self.ctx.add_observable_measurement(1 + b, block_meas)
        
        return logical_meas
    
    def emit_deferred_boundary_detectors(self, circuit: stim.Circuit) -> None:
        """
        Emit boundary detectors that were deferred during emit_final_measurement.
        
        This must be called AFTER all blocks have completed their final
        measurements so that the lookbacks are computed relative to the
        correct total measurement count.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Target circuit.
        """
        if not hasattr(self, '_deferred_boundary_basis'):
            return  # Nothing deferred
        
        if not self._block_contiguous:
            self._emit_all_space_like_detectors(
                circuit,
                self._deferred_boundary_basis,
                self._deferred_boundary_data_meas,
                self._deferred_boundary_meas_start,
            )
        
        # Clear deferred state
        del self._deferred_boundary_basis
        del self._deferred_boundary_data_meas
        del self._deferred_boundary_meas_start
    
    def _emit_all_space_like_detectors(
        self,
        circuit: stim.Circuit,
        basis: str,
        data_meas: Dict[int, int],
        meas_start: int = 0,
    ) -> None:
        """
        Emit inner space-like (boundary) detectors on data blocks.

        After trailing inner rounds have been emitted,
        ``_pre_outer_inner_z/x_meas`` points to the last trailing inner
        measurement.  Because no transversal outer CNOTs occur between the
        trailing rounds and the final destructive measurement, the boundary
        detector is a simple 2-term comparison::

            last_inner_Z(b,s) ⊕ final_MZ_parity(b,s) = 0   (Z-basis)
            last_inner_X(b,s) ⊕ final_MX_parity(b,s) = 0   (X-basis)

        Outer boundary detectors are intentionally **not** emitted here.
        They would share the same final-measurement data as the inner
        boundary detectors, creating a hub-detector that couples all blocks
        in the outer stabiliser's support.  A single Z error on a qubit
        that is in both an inner Z stabiliser and the inner logical-Z
        support would then trigger only 2 detectors (1 inner boundary +
        1 outer boundary) plus the observable — giving a weight-2 DEM
        error that collapses the effective code distance to 2.

        Without the outer boundary detector, the inner boundary detectors
        are *independent per block*, exactly like boundary detectors in a
        standard surface code.  The outer stabiliser detection chain is
        closed by the temporal detectors from the ``d_outer`` outer rounds
        (anchor at round 0, then ``d_outer − 1`` temporal comparisons).
        """
        basis = basis.upper()
        
        # Use gadget block index for unique coords in multi-block gadgets
        gadget_idx = self._gadget_block_index

        if basis == "Z":
            # Inner Z boundary detectors on each data block
            # Offset Z stabilizer indices by r_x_in to avoid coord collision with X stabilizers
            z_offset = self.r_x_in
            for block_id in range(self.n_out):
                for s_idx in range(self.r_z_in):
                    # Last inner Z measurement (from trailing rounds or pre-outer)
                    pre_meas = self._pre_outer_inner_z_meas.get((block_id, s_idx))
                    if pre_meas is None:
                        continue

                    # Final MZ parity for this inner Z stabilizer on this block
                    row = self._hz_in[s_idx]
                    block_data_start = block_id * self.n_in
                    final_parity_meas = []
                    for q in range(self.n_in):
                        if row[q]:
                            global_q = block_data_start + q
                            if global_q in data_meas:
                                final_parity_meas.append(data_meas[global_q])

                    if not final_parity_meas:
                        continue

                    # Simple 2-term boundary: last_inner_Z ⊕ final_parity
                    all_meas = [pre_meas] + final_parity_meas

                    # Coord includes gadget_idx to differentiate blocks in multi-block gadgets
                    coord = (float(block_id), float(s_idx + z_offset + 2000), self.ctx.current_time, float(DetectorRole.INNER_BOUNDARY.value), float(gadget_idx))
                    self.ctx.emit_detector(circuit, all_meas, coord)
        else:
            # Inner X boundary detectors on each data block
            for block_id in range(self.n_out):
                for s_idx in range(self.r_x_in):
                    # Last inner X measurement (from trailing rounds or pre-outer)
                    pre_meas = self._pre_outer_inner_x_meas.get((block_id, s_idx))
                    if pre_meas is None:
                        continue

                    # Final MX parity for this inner X stabilizer on this block
                    row = self._hx_in[s_idx]
                    block_data_start = block_id * self.n_in
                    final_parity_meas = []
                    for q in range(self.n_in):
                        if row[q]:
                            global_q = block_data_start + q
                            if global_q in data_meas:
                                final_parity_meas.append(data_meas[global_q])

                    if not final_parity_meas:
                        continue

                    # Simple 2-term boundary: last_inner_X ⊕ final_parity
                    all_meas = [pre_meas] + final_parity_meas

                    # Coord includes gadget_idx to differentiate blocks in multi-block gadgets
                    coord = (float(block_id), float(s_idx + 2000), self.ctx.current_time, float(DetectorRole.INNER_BOUNDARY.value), float(gadget_idx))
                    self.ctx.emit_detector(circuit, all_meas, coord)
        
        # Outer space-like (boundary) detectors
        # For outer Z-stab j with block support B_j:
        #   Boundary = last_outer_z_meas[j] XOR (XOR_{b in B_j} decoded_logical_Z(block_b))
        # The decoded logical Z from each block = XOR of final measurements
        # on inner logical Z support.
        #
        # These detectors close the outer stabiliser detection chain against
        # the final destructive measurement.  They are necessary: without
        # them, a single Z error on a qubit in the inner logical-Z support
        # would trigger only the inner Z boundary detector (weight 1 + L0),
        # which is worse than the weight-2 case.
        #
        # With trailing inner rounds, the temporal depth between the last
        # outer round and the final measurement is d_inner, which restores
        # the inner code's distance protection for errors *during* that
        # window.  Only errors at the very last instant (between the last
        # trailing round and the final MZ) produce weight-2 DEM errors.
        if basis == "Z":
            for stab_idx in range(self.r_z_out):
                last_outer_meas = self._last_outer_z_meas.get(stab_idx)
                if last_outer_meas is None:
                    continue

                # Get blocks in outer Z-stab support
                row = self._hz_out[stab_idx]
                block_support = [b for b in range(self.n_out) if row[b]]

                # Collect final measurements for logical Z of each block
                boundary_meas: List[int] = list(last_outer_meas)
                for block_id in block_support:
                    block_data_start = block_id * self.n_in
                    for local_q in self._inner_logical_z_support:
                        global_q = block_data_start + local_q
                        if global_q in data_meas:
                            boundary_meas.append(data_meas[global_q])

                if boundary_meas:
                    coord = self._make_detector_coord(float(self.n_out), float(stab_idx), self.ctx.current_time, float(DetectorRole.OUTER_TEMPORAL.value))
                    self.ctx.emit_detector(circuit, boundary_meas, coord)
        else:
            # X-basis: outer X boundary detectors
            for stab_idx in range(self.r_x_out):
                last_outer_meas = self._last_outer_x_meas.get(stab_idx)
                if last_outer_meas is None:
                    continue

                # Get blocks in outer X-stab support
                row = self._hx_out[stab_idx]
                block_support = [b for b in range(self.n_out) if row[b]]

                # Collect final measurements for logical X of each block
                boundary_meas: List[int] = list(last_outer_meas)
                for block_id in block_support:
                    block_data_start = block_id * self.n_in
                    for local_q in self._inner_logical_x_support:
                        global_q = block_data_start + local_q
                        if global_q in data_meas:
                            boundary_meas.append(data_meas[global_q])

                if boundary_meas:
                    coord = self._make_detector_coord(float(self.n_out), float(stab_idx), self.ctx.current_time, float(DetectorRole.OUTER_TEMPORAL.value))
                    self.ctx.emit_detector(circuit, boundary_meas, coord)
    
    def emit_deferred_detectors(
        self,
        circuit: stim.Circuit,
        data_meas: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Emit all deferred detectors in block-contiguous order.
        
        For hierarchical decoding compatibility.
        """
        if not self._block_contiguous:
            return
        
        basis = self.measurement_basis
        
        # Inner blocks first (block-contiguous order)
        for block_id in range(self.n_out):
            # X stabilizer detectors for this block
            for local_idx in range(self.r_x_in):
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
            for local_idx in range(self.r_z_in):
                meas_list = self._deferred_inner_z[block_id][local_idx]
                # Offset Z stabilizer indices by r_x_in to avoid coord collision with X stabilizers
                z_offset = self.r_x_in
                coord_base = (float(block_id), float(local_idx + z_offset))
                
                for i, (r, cur_meas) in enumerate(meas_list):
                    coord = coord_base + (float(r),)
                    if i == 0:
                        if basis == "Z":
                            self.ctx.emit_detector(circuit, [cur_meas], coord)
                    else:
                        prev_meas = meas_list[i - 1][1]
                        self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
            
            # Block space-like detectors
            if data_meas:
                self._emit_block_space_like_detectors(circuit, block_id, data_meas)
        
        # Outer stabilizer detectors
        for local_idx in range(self.r_x_out):
            meas_list = self._deferred_outer_x[local_idx]
            coord_base = (float(self.n_out), float(local_idx))
            
            for i, (r, cur_meas) in enumerate(meas_list):
                coord = coord_base + (float(r),)
                if i == 0:
                    if basis == "X":
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                else:
                    prev_meas = meas_list[i - 1][1]
                    self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
        
        for local_idx in range(self.r_z_out):
            meas_list = self._deferred_outer_z[local_idx]
            coord_base = (float(self.n_out), float(local_idx))
            
            for i, (r, cur_meas) in enumerate(meas_list):
                coord = coord_base + (float(r),)
                if i == 0:
                    if basis == "Z":
                        self.ctx.emit_detector(circuit, [cur_meas], coord)
                else:
                    prev_meas = meas_list[i - 1][1]
                    self.ctx.emit_detector(circuit, [prev_meas, cur_meas], coord)
    
    def _emit_block_space_like_detectors(
        self,
        circuit: stim.Circuit,
        block_id: int,
        data_meas: Dict[int, int],
    ) -> None:
        """Emit space-like detectors for an inner block."""
        basis = self.measurement_basis
        n_rounds = self._round_number
        
        block_data = self.data_block_qubits(block_id)
        block_start = block_data[0]
        
        if basis == "Z":
            for local_idx in range(self.r_z_in):
                row = self._hz_in[local_idx]
                data_idxs = [
                    data_meas[q - self._data_start]
                    for q_local, q in enumerate(block_data)
                    if q_local < len(row) and row[q_local] and (q - self._data_start) in data_meas
                ]
                
                recs = list(data_idxs)
                meas_list = self._deferred_inner_z[block_id][local_idx]
                if meas_list:
                    recs.append(meas_list[-1][1])
                
                if recs:
                    # Offset Z stabilizer indices by r_x_in to avoid coord collision with X stabilizers
                    z_offset = self.r_x_in
                    coord = (float(block_id), float(local_idx + z_offset), float(n_rounds), float(DetectorRole.INNER_BOUNDARY.value))
                    self.ctx.emit_detector(circuit, recs, coord)
        else:
            for local_idx in range(self.r_x_in):
                row = self._hx_in[local_idx]
                data_idxs = [
                    data_meas[q - self._data_start]
                    for q_local, q in enumerate(block_data)
                    if q_local < len(row) and row[q_local] and (q - self._data_start) in data_meas
                ]
                
                recs = list(data_idxs)
                meas_list = self._deferred_inner_x[block_id][local_idx]
                if meas_list:
                    recs.append(meas_list[-1][1])
                
                if recs:
                    coord = (float(block_id), float(local_idx), float(n_rounds), float(DetectorRole.INNER_BOUNDARY.value))
                    self.ctx.emit_detector(circuit, recs, coord)
    
    def reset_stabilizer_history(
        self,
        swap_xz: bool = False,
        skip_first_round: bool = False,
        clear_history: bool = False,
    ) -> None:
        """Reset stabilizer measurement history after a transversal gate.

        Parameters
        ----------
        swap_xz : bool
            If True, swap the effective measurement basis (needed after
            Hadamard where X↔Z).
        skip_first_round : bool
            If True, suppress ALL anchor detectors in the next emitted
            round.  After a transversal gate the data qubits are no longer
            in a fresh eigenstate, so first-round anchor detectors would
            be non-deterministic.  The flag auto-clears after the first
            inner round so that subsequent rounds emit normal temporal
            detectors.
        clear_history : bool
            If True, clear all measurement history (teleportation-style
            transforms where the logical state is teleported).
        """
        self._last_inner_x_meas.clear()
        self._last_inner_z_meas.clear()
        self._last_outer_x_meas.clear()
        self._last_outer_z_meas.clear()
        
        # Reset round counters so post-gadget rounds start fresh
        self._outer_round_number = 0
        self._inner_round_in_block = 0
        self._round_number = 0
        
        # Clear pre-outer measurement tracking (used for crossing detectors)
        self._pre_outer_inner_x_meas.clear()
        self._pre_outer_inner_z_meas.clear()
        
        # Clear outer ancilla prep measurements
        self._outer_x_anc_z_prep.clear()
        self._outer_z_anc_x_prep.clear()
        
        # Clear destructive measurement storage
        self._outer_x_anc_destruct_x.clear()
        self._outer_z_anc_destruct_z.clear()
        
        # Clear ancilla inner measurements from prep
        self._last_anc_inner_x_meas_prep.clear()
        self._last_anc_inner_z_meas_prep.clear()
        
        # Reset deferred detectors
        self._deferred_inner_x = [
            [[] for _ in range(self.r_x_in)] for _ in range(self.n_out)
        ]
        self._deferred_inner_z = [
            [[] for _ in range(self.r_z_in)] for _ in range(self.n_out)
        ]
        self._deferred_outer_x = [
            [] for _ in range(self.r_x_out)
        ]
        self._deferred_outer_z = [
            [] for _ in range(self.r_z_out)
        ]
        
        if swap_xz:
            self.measurement_basis = "X" if self.measurement_basis == "Z" else "Z"

        self._skip_first_round = skip_first_round
        
        # NOTE: We respect the caller's skip_first_round even when
        # clear_history=True.  Teleportation gadgets set skip_first_round=False
        # because the surviving block IS in the code space after teleportation
        # (all stabilizer eigenvalues are +1), making first-round anchors
        # deterministic and critical for α scaling.  Surgery-CNOT sets
        # skip_first_round=True because bridge-induced dressing makes
        # inner eigenvalues non-deterministic.  The flat CSS builder also
        # respects skip_first_round without override.
    
    def get_data_meas_mapping(self, meas_start: int) -> Dict[int, int]:
        """Get mapping from data qubit index to measurement index."""
        n = self.n_data
        return {q: meas_start + q for q in range(n)}
