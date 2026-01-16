# src/qectostim/experiments/logical_block_manager_v2.py
"""
Logical Block Manager V2 - Measurement-Centric Model for HierarchicalV6Decoder.

This module provides LogicalBlockManagerV2 for managing inner code blocks
in hierarchical concatenated code experiments.

Measurement-Centric Model (MCM) Architecture
=============================================
The MCM provides direct mapping from circuit measurements to decoder phases:

  InnerECInstance
    - One per (block, segment) with `n_rounds` field
    - Contains x_meas_indices_by_round & z_meas_indices_by_round
    - Decoder computes inner_z/x_correction per instance via majority vote

  OuterSyndromeValue  
    - One per outer stabilizer measurement
    - Links to ancilla_ec_instance_id for correction application
    - Raw measurement index stored for syndrome extraction

  FinalDataMeasurement
    - Per-block final measurement indices
    - Decoder applies cumulative inner corrections

HierarchicalV6Decoder 4-Phase Pipeline
======================================

  Phase 1: Inner Error Inference
    - For each InnerECInstance, majority vote across n_rounds
    - X ancillas detect Z errors → Z correction
    - Z ancillas detect X errors → X correction
    - Output: inner_corrections[instance_id] → {'z': [...], 'x': [...]}

  Phase 2: Correct Outer Syndrome  
    - For each OuterSyndromeValue, apply ancilla correction
    - syndrome_corrected = raw_syndrome XOR correction[ancilla_ec_instance_id]
    - X outer stabs: use inner_x_correction
    - Z outer stabs: use inner_z_correction

  Phase 3: Correct Data Logicals
    - Final measurements corrected by cumulative inner corrections
    - All EC instances for each data block XORed together

  Phase 4: Outer Temporal DEM Decode
    - Compare corrected syndromes across rounds
    - PyMatching decode for logical prediction

Key Features:
=============

1. **Doubled Ancilla Pools**: Separate X and Z ancilla pools for parallel
   measurement of X-type and Z-type outer stabilizers.

2. **Block Management**: Allocate/release ancilla blocks as needed for
   outer stabilizer measurement protocols.

3. **MCM Tracking**: InnerECInstance with n_rounds field aggregates all 
   measurements from one (block, segment) for decoder majority vote.

4. **Raw Circuit Emission**: All emit_* methods produce raw gates and
   measurements only - no detectors, no transform tracking.

Architecture:
=============

┌─────────────────────────────────────────────────────────────────────────────┐
│                         LogicalBlockManagerV2                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Block Management:                                                          │
│    • Data blocks (permanent, represent outer qubits)                        │
│    • X ancilla pool (for X outer stabilizers)                               │
│    • Z ancilla pool (for Z outer stabilizers)                               │
│                                                                             │
│  MCM Tracking (for HierarchicalV6Decoder):                                  │
│    • InnerECInstance: (block_id, context, n_rounds, measurements)           │
│    • OuterSyndromeValue: (stab_type, stab_idx, ancilla_ec_id)               │
│    • FinalDataMeasurement: (block_id, basis, indices)                       │
│                                                                             │
│  Operations:                                                                │
│    • emit_inner_ec_segment() - emits d rounds as ONE InnerECInstance        │
│    • emit_inner_ec_round() - single round (internal use)                    │
│    • emit_reset_block() - reset qubits in a block                           │
│    • emit_prepare_logical_*() - logical state preparation                   │
│    • emit_measure_logical_*() - logical measurement                         │
│    • allocate/release_ancilla() - pool management                           │
│                                                                             │
│  Output:                                                                    │
│    • MeasurementCentricMetadata: Direct decoder input                       │
│    • Raw circuit with measurements indexed for decoder to parse             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum, auto

import numpy as np
import stim

if TYPE_CHECKING:
    from qectostim.codes.abstract_code import Code

from qectostim.experiments.stabilizer_rounds import (
    StabilizerBasis,
    get_logical_support,
)
from qectostim.gadgets.layout import BlockAllocation
from qectostim.experiments.measurement_model import (
    InnerECInstance,
    MeasurementCentricMetadata,
)


class BlockType(Enum):
    """Type of logical block."""
    DATA = auto()
    X_ANCILLA = auto()
    Z_ANCILLA = auto()


@dataclass
class BlockInfoV2:
    """
    Information about a single inner code block.
    
    Attributes
    ----------
    block_id : int
        Unique identifier for this block.
    block_type : BlockType
        Type of block: DATA, X_ANCILLA, or Z_ANCILLA.
    name : str
        Human-readable name for debugging.
    allocation : BlockAllocation
        Qubit index allocation for this block.
    data_qubits : List[int]
        Indices of data qubits in this block.
    x_ancilla_qubits : List[int]
        Indices of X stabilizer ancilla qubits.
    z_ancilla_qubits : List[int]
        Indices of Z stabilizer ancilla qubits.
    is_allocated : bool
        For ancilla blocks: whether currently in use.
    """
    block_id: int
    block_type: BlockType
    name: str
    allocation: BlockAllocation
    data_qubits: List[int] = field(default_factory=list)
    x_ancilla_qubits: List[int] = field(default_factory=list)
    z_ancilla_qubits: List[int] = field(default_factory=list)
    is_allocated: bool = False
    
    @property
    def all_qubits(self) -> List[int]:
        """All qubit indices in this block."""
        return self.data_qubits + self.x_ancilla_qubits + self.z_ancilla_qubits


class LogicalBlockManagerV2:
    """
    Logical block manager for segment-based hierarchical experiments.
    
    Manages inner code blocks for hierarchical concatenated codes with support for:
    - Doubled ancilla pools (separate X and Z pools for parallel outer stabilizers)
    - Segment tracking for HierarchicalV6Decoder 4-phase pipeline
    - Raw circuit emission (no DETECTOR instructions)
    - Block allocation metadata for decoder
    
    Segment-Based Architecture for HierarchicalV6Decoder
    =====================================================
    All emit_* methods produce raw operations only. The HierarchicalV6Decoder
    extracts syndrome information from segment metadata:
    
    Segment Types:
    - 'inner_ec': Inner error correction round(s) on a block
      Contains: {block_id, round, measurements: {r: {'x_anc': [...], 'z_anc': [...]}}}
      
    - 'outer_x_stab' / 'outer_z_stab': Outer stabilizer gadget
      Contains: {outer_round, stab_idx, ancilla_block, data_blocks,
                 ancilla_ec_segment_ids, ancilla_meas_indices}
                 
    - 'final_measurement': Final transversal data measurements
      Contains: {final_meas_by_block: {block_id: [meas_indices]}}
    
    Key Mappings Built for Decoder:
    - ancilla_ec_segment_ids: Per outer stab gadget, which inner EC segs protect ancilla
    - data_block_to_all_ec_segments: For each data block, ALL inner EC seg_ids
    
    Parameters
    ----------
    inner_code : Code
        The code used for each inner block.
    n_data_blocks : int
        Number of data blocks (= outer code n).
    n_x_ancilla_blocks : int
        Number of X ancilla blocks in pool.
    n_z_ancilla_blocks : int
        Number of Z ancilla blocks in pool.
    measurement_basis : str
        Default measurement basis ("Z" or "X").
    """
    
    def __init__(
        self,
        inner_code: "Code",
        n_data_blocks: int,
        n_x_ancilla_blocks: int = 1,
        n_z_ancilla_blocks: int = 1,
        measurement_basis: str = "Z",
    ):
        self._inner_code = inner_code
        self._n_data_blocks = n_data_blocks
        self._n_x_ancilla_blocks = n_x_ancilla_blocks
        self._n_z_ancilla_blocks = n_z_ancilla_blocks
        self._measurement_basis = measurement_basis.upper()
        
        # Inner code properties
        self._n_inner = inner_code.n
        self._k_inner = inner_code.k
        
        # Get stabilizer counts
        hx = getattr(inner_code, 'hx', None)
        hz = getattr(inner_code, 'hz', None)
        self._n_x_inner = hx.shape[0] if hx is not None and hasattr(hx, 'shape') and hx.size > 0 else 0
        self._n_z_inner = hz.shape[0] if hz is not None and hasattr(hz, 'shape') and hz.size > 0 else 0
        self._qubits_per_block = self._n_inner + self._n_x_inner + self._n_z_inner
        
        # Storage
        self._blocks: Dict[int, BlockInfoV2] = {}
        self._x_ancilla_pool: List[int] = []  # Available X ancilla block IDs
        self._z_ancilla_pool: List[int] = []  # Available Z ancilla block IDs
        
        # Measurement tracking
        self._measurement_index = 0
        # Track stabilizer measurement indices by block: block_id -> {'x_anc': [...], 'z_anc': [...]}
        self._inner_stabilizer_measurements: Dict[int, Dict[str, List[int]]] = {}
        
        # =====================================================================
        # NEW: Measurement-Centric Model for InnerECInstance tracking
        # =====================================================================
        # Every inner EC application creates an InnerECInstance.
        # This is more direct than segment-based tracking because:
        # - Each EC instance maps 1:1 with decoder correction computation
        # - OuterSyndromeValue can directly reference its ancilla's EC instance
        # - No ambiguity about where ancilla corrections come from
        # =====================================================================
        self._ec_instance_counter = 0
        self._measurement_centric_metadata = MeasurementCentricMetadata()
        self._measurement_centric_metadata.n_x_stabs_inner = self._n_x_inner
        self._measurement_centric_metadata.n_z_stabs_inner = self._n_z_inner
        self._current_ec_context = "unknown"  # Set by caller: 'init', 'pre_outer', etc
        self._current_outer_round = -1  # Set by caller
        
        # Block ID offsets
        self._x_ancilla_offset = n_data_blocks
        self._z_ancilla_offset = n_data_blocks + n_x_ancilla_blocks
        
        # Initialize blocks
        self._initialize_blocks()
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def inner_code(self) -> "Code":
        return self._inner_code
    
    @property
    def n_data_blocks(self) -> int:
        return self._n_data_blocks
    
    @property
    def n_x_ancilla_blocks(self) -> int:
        return self._n_x_ancilla_blocks
    
    @property
    def n_z_ancilla_blocks(self) -> int:
        return self._n_z_ancilla_blocks
    
    @property
    def total_blocks(self) -> int:
        return self._n_data_blocks + self._n_x_ancilla_blocks + self._n_z_ancilla_blocks
    
    @property
    def total_qubits(self) -> int:
        return self.total_blocks * self._qubits_per_block
    
    @property
    def qubits_per_block(self) -> int:
        return self._qubits_per_block
    
    @property
    def measurement_index(self) -> int:
        return self._measurement_index
    
    @property
    def x_ancilla_offset(self) -> int:
        return self._x_ancilla_offset
    
    @property
    def z_ancilla_offset(self) -> int:
        return self._z_ancilla_offset
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def _initialize_blocks(self) -> None:
        """Initialize all data and ancilla blocks."""
        idx = 0
        
        # Data blocks
        for i in range(self._n_data_blocks):
            block_id = i
            name = f"data_{i}"
            self._create_block(
                block_id=block_id,
                block_type=BlockType.DATA,
                name=name,
                qubit_start=idx,
                offset=(i * 10.0, 0.0),
            )
            idx += self._qubits_per_block
        
        # X ancilla pool
        for i in range(self._n_x_ancilla_blocks):
            block_id = self._x_ancilla_offset + i
            name = f"x_anc_{i}"
            self._create_block(
                block_id=block_id,
                block_type=BlockType.X_ANCILLA,
                name=name,
                qubit_start=idx,
                offset=(i * 10.0, 5.0),
            )
            self._x_ancilla_pool.append(block_id)
            idx += self._qubits_per_block
        
        # Z ancilla pool
        for i in range(self._n_z_ancilla_blocks):
            block_id = self._z_ancilla_offset + i
            name = f"z_anc_{i}"
            self._create_block(
                block_id=block_id,
                block_type=BlockType.Z_ANCILLA,
                name=name,
                qubit_start=idx,
                offset=(i * 10.0, 10.0),
            )
            self._z_ancilla_pool.append(block_id)
            idx += self._qubits_per_block
    
    def _create_block(
        self,
        block_id: int,
        block_type: BlockType,
        name: str,
        qubit_start: int,
        offset: Tuple[float, float],
    ) -> None:
        """Create a single block."""
        allocation = BlockAllocation(
            block_name=name,
            code=self._inner_code,
            data_start=qubit_start,
            data_count=self._n_inner,
            x_anc_start=qubit_start + self._n_inner,
            x_anc_count=self._n_x_inner,
            z_anc_start=qubit_start + self._n_inner + self._n_x_inner,
            z_anc_count=self._n_z_inner,
            offset=offset,
        )
        
        self._blocks[block_id] = BlockInfoV2(
            block_id=block_id,
            block_type=block_type,
            name=name,
            allocation=allocation,
            data_qubits=list(range(qubit_start, qubit_start + self._n_inner)),
            x_ancilla_qubits=list(range(
                qubit_start + self._n_inner,
                qubit_start + self._n_inner + self._n_x_inner
            )),
            z_ancilla_qubits=list(range(
                qubit_start + self._n_inner + self._n_x_inner,
                qubit_start + self._n_inner + self._n_x_inner + self._n_z_inner
            )),
            is_allocated=False,
        )
    
    # =========================================================================
    # Block Access
    # =========================================================================
    
    def get_block(self, block_id: int) -> BlockInfoV2:
        """Get block info by ID."""
        return self._blocks[block_id]
    
    def get_block_by_name(self, name: str) -> Optional[BlockInfoV2]:
        """Get block info by name."""
        for block in self._blocks.values():
            if block.name == name:
                return block
        return None
    
    def get_data_blocks(self) -> List[BlockInfoV2]:
        """Get all data blocks."""
        return [b for b in self._blocks.values() if b.block_type == BlockType.DATA]
    
    def get_all_qubits(self) -> List[int]:
        """Get all qubit indices."""
        qubits = []
        for block in self._blocks.values():
            qubits.extend(block.all_qubits)
        return sorted(qubits)
    
    def get_allocation(self, block_id: int) -> BlockAllocation:
        """Get block allocation."""
        return self._blocks[block_id].allocation
    
    # =========================================================================
    # Ancilla Pool Management
    # =========================================================================
    
    def allocate_x_ancilla(self) -> Optional[int]:
        """Allocate an X ancilla block from the pool."""
        if not self._x_ancilla_pool:
            return None
        block_id = self._x_ancilla_pool.pop(0)
        self._blocks[block_id].is_allocated = True
        return block_id
    
    def allocate_z_ancilla(self) -> Optional[int]:
        """Allocate a Z ancilla block from the pool."""
        if not self._z_ancilla_pool:
            return None
        block_id = self._z_ancilla_pool.pop(0)
        self._blocks[block_id].is_allocated = True
        return block_id
    
    def release_x_ancilla(self, block_id: int) -> None:
        """Release an X ancilla block back to the pool."""
        if block_id not in self._x_ancilla_pool:
            self._x_ancilla_pool.append(block_id)
            self._blocks[block_id].is_allocated = False
    
    def release_z_ancilla(self, block_id: int) -> None:
        """Release a Z ancilla block back to the pool."""
        if block_id not in self._z_ancilla_pool:
            self._z_ancilla_pool.append(block_id)
            self._blocks[block_id].is_allocated = False
    
    def allocate_x_ancillas(self, count: int) -> List[int]:
        """Allocate multiple X ancilla blocks."""
        allocated = []
        for _ in range(count):
            block_id = self.allocate_x_ancilla()
            if block_id is not None:
                allocated.append(block_id)
        return allocated
    
    def allocate_z_ancillas(self, count: int) -> List[int]:
        """Allocate multiple Z ancilla blocks."""
        allocated = []
        for _ in range(count):
            block_id = self.allocate_z_ancilla()
            if block_id is not None:
                allocated.append(block_id)
        return allocated
    
    # =========================================================================
    # Circuit Emission - Coordinates
    # =========================================================================
    
    def emit_qubit_coords(self, circuit: stim.Circuit) -> None:
        """Emit QUBIT_COORDS for all qubits."""
        for block in self._blocks.values():
            offset = block.allocation.offset
            # Data qubits
            for i, q in enumerate(block.data_qubits):
                circuit.append("QUBIT_COORDS", [q], [offset[0] + i % 3, offset[1] + i // 3])
            # X ancilla qubits
            for i, q in enumerate(block.x_ancilla_qubits):
                circuit.append("QUBIT_COORDS", [q], [offset[0] + i + 0.5, offset[1] - 0.5])
            # Z ancilla qubits
            for i, q in enumerate(block.z_ancilla_qubits):
                circuit.append("QUBIT_COORDS", [q], [offset[0] + i + 0.5, offset[1] + 3.5])
    
    # =========================================================================
    # Circuit Emission - Reset
    # =========================================================================
    
    def emit_reset_all(self, circuit: stim.Circuit) -> None:
        """Reset all qubits in all blocks."""
        circuit.append("R", self.get_all_qubits())
    
    def emit_reset_block(self, circuit: stim.Circuit, block_id: int) -> None:
        """
        Reset all qubits in a block.
        
        Emits R gates on all data and ancilla qubits in the specified block.
        """
        block = self._blocks[block_id]
        circuit.append("R", block.all_qubits)
    
    def emit_reset_blocks(self, circuit: stim.Circuit, block_ids: List[int]) -> None:
        """
        Reset all qubits in multiple blocks.
        
        Emits R gates on all data and ancilla qubits in the specified blocks.
        """
        qubits = []
        for block_id in block_ids:
            block = self._blocks[block_id]
            qubits.extend(block.all_qubits)
        if qubits:
            circuit.append("R", sorted(qubits))
    
    # =========================================================================
    # Circuit Emission - Logical State Preparation
    # =========================================================================
    
    def emit_prepare_logical_zero(self, circuit: stim.Circuit, block_id: int) -> None:
        """
        Prepare block in logical |0⟩_L via stabilizer projection.
        
        PREPARATION METHOD (per Steane 1997, AGP quant-ph/0504218):
        ===========================================================
        For CSS codes, the correct preparation is:
        1. Reset gives |0⟩^⊗n (product state of physical zeros)
        2. Measure X stabilizers to project onto code space
        3. This gives |0⟩_L (or Z-shifted |0⟩_L depending on syndrome)
        
        Why X stabilizers (not Z)?
        --------------------------
        - |0⟩^⊗n is already a Z stabilizer eigenstate with eigenvalue +1
        - But |0⟩^⊗n is NOT in the code space (mixture of codewords + non-codewords)
        - X stabilizers project onto code space without disturbing Z eigenvalue
        - Result: |0⟩_L with Z_L = +1 (deterministic)
        
        CRITICAL: Without this projection, data blocks are NOT valid codewords!
        The inner EC and outer syndrome measurements will see garbage.
        """
        block = self._blocks[block_id]
        
        # |0⟩^⊗n is already prepared by reset
        # Measure X stabilizers to project onto code space
        self._emit_x_stabilizer_projection(circuit, block)
    
    def emit_prepare_logical_plus(self, circuit: stim.Circuit, block_id: int) -> None:
        """
        Prepare block in logical |+⟩_L via stabilizer projection.
        
        PREPARATION METHOD (per Steane 1997, AGP quant-ph/0504218):
        ===========================================================
        For CSS codes, the correct preparation is:
        1. Apply H^⊗n to get |+⟩^⊗n (superposition of all 2^n computational states)
        2. Measure Z stabilizers to project onto code space
        3. This gives |+⟩_L (or |-⟩_L depending on syndrome)
        
        Why Z stabilizers (not X)?
        --------------------------
        - |+⟩^⊗n is already an X stabilizer eigenstate with eigenvalue +1
        - But |+⟩^⊗n is NOT in the code space (mixture of codewords + non-codewords)
        - Z stabilizers project onto code space without disturbing X eigenvalue
        - Result: |+⟩_L with X_L = +1 (deterministic)
        
        VERIFICATION (Fix #4 - per AGP Section 3.1):
        ============================================
        For full fault-tolerance, the AGP protocol requires VERIFIED ancilla
        preparation. This means:
        
        1. After projection, measure X stabilizers to verify X syndrome = 0
        2. If X syndrome ≠ 0, discard and re-prepare (or apply correction)
        3. Alternative: Use flag qubits during preparation to detect errors
        
        Current implementation: Basic projection WITHOUT verification.
        The decoder compensates by using X syndrome on ancilla blocks to
        correct Z errors that would corrupt the X_L measurement.
        
        For improved fidelity, implement verified preparation:
        - Check X syndrome after Z projection
        - If non-zero, either correct or re-prepare
        - This reduces burden on inner decoder
        
        ERROR PROPAGATION ANALYSIS:
        ---------------------------
        - X errors during preparation don't affect X_L (X commutes with X_L)
        - Z errors during preparation flip X_L (detected by X syndrome)
        - Inner EC after preparation catches most Z errors
        - CNOT to data can propagate errors but weight-1 → weight-1 (transversal)
        """
        block = self._blocks[block_id]
        
        # Step 1: H on all data qubits → |+⟩^⊗n
        circuit.append("H", block.data_qubits)
        
        # Step 2: Measure Z stabilizers to project onto code space
        # This is non-destructive and gives |+⟩_L
        self._emit_z_stabilizer_projection(circuit, block)
    
    def emit_prepare_logical_one(self, circuit: stim.Circuit, block_id: int) -> None:
        """
        Prepare block in logical |1⟩ = X|0⟩.
        
        For CSS codes, logical X is the X operators on the logical X support.
        |1⟩ is an eigenstate of logical Z with eigenvalue -1.
        """
        block = self._blocks[block_id]
        # Get logical X operator support from inner code
        lx_support = get_logical_support(self._inner_code, "X", 0)
        
        # Apply X on logical X support qubits
        for idx in lx_support:
            if idx < len(block.data_qubits):
                circuit.append("X", [block.data_qubits[idx]])
    
    def emit_prepare_logical_minus(self, circuit: stim.Circuit, block_id: int) -> None:
        """
        Prepare block in logical |-⟩ = H|1⟩ = Z|+⟩.
        
        Applies H on all data qubits then applies logical Z.
        """
        block = self._blocks[block_id]
        
        # First: H on all data qubits to get |+⟩
        circuit.append("H", block.data_qubits)
        
        # Then: Apply logical Z to get |-⟩ (Z|+⟩ = |-⟩)
        lz_support = get_logical_support(self._inner_code, "Z", 0)
        for idx in lz_support:
            if idx < len(block.data_qubits):
                circuit.append("Z", [block.data_qubits[idx]])
    
    # =========================================================================
    # Circuit Emission - Z Stabilizer Projection for Initialization
    # =========================================================================

    def _emit_z_stabilizer_projection(
        self,
        circuit: stim.Circuit,
        block: BlockInfoV2,
    ) -> None:
        """
        Emit Z stabilizer measurements to project |+⟩^⊗n onto |+⟩_L.
        
        Per AGP (quant-ph/0504218) Section 3.1, Steane (1997):
        =====================================================
        After H^⊗n, we have |+⟩^⊗n = equal superposition of ALL 2^n basis states.
        This is NOT in the code space - it's a mixture of codewords and non-codewords.
        
        Measuring Z stabilizers:
        1. Projects onto states with correct Z parities
        2. These states are exactly the CSS codewords  
        3. The X eigenvalue is preserved (H^⊗n gave +1, projection doesn't change it)
        4. Result: |+⟩_L with X_L = +1 deterministically
        
        The measurement outcomes determine WHICH codeword basis state is projected:
        - Syndrome = 0 → |+⟩_L directly
        - Syndrome ≠ 0 → shifted |+⟩_L (Pauli frame, still valid X eigenstate)
        
        We use MR (measure-reset) to reset ancillas for reuse. The measurement
        outcomes are intentionally not tracked as syndromes since this is for
        state preparation, not error detection.
        """
        if not block.z_ancilla_qubits:
            return
        
        # Get Z stabilizer matrix (hz defines Z stabilizers for CSS codes)
        # hz[i] gives support of i-th Z stabilizer (positions of Z operators)
        hz = getattr(self._inner_code, 'hz', None)
        if hz is None or not hasattr(hz, 'shape') or hz.size == 0:
            return
        
        # Z ancillas start in |0⟩ (default after reset, no H needed)
        
        # Apply CNOTs for Z stabilizers: CNOT(data→ancilla) computes Z parity
        # Z stabilizer = ⊗_i Z_i on support qubits
        # CNOT copies Z parity to ancilla: if data has Z_i=1, ancilla flips
        for stab_idx in range(min(hz.shape[0], len(block.z_ancilla_qubits))):
            anc = block.z_ancilla_qubits[stab_idx]
            support = np.where(hz[stab_idx] != 0)[0]
            for data_idx in support:
                if data_idx < len(block.data_qubits):
                    circuit.append("CX", [block.data_qubits[data_idx], anc])
        
        # Measure Z ancillas in Z basis (no H needed) to project
        # MR resets ancillas for reuse - measurements are for projection only
        circuit.append("MR", block.z_ancilla_qubits)
        self._measurement_index += len(block.z_ancilla_qubits)

    def _emit_x_stabilizer_projection(
        self,
        circuit: stim.Circuit,
        block: BlockInfoV2,
    ) -> None:
        """
        Emit X stabilizer measurements to project |0⟩^⊗n onto |0⟩_L.
        
        Per AGP (quant-ph/0504218) Section 3.1, Steane (1997):
        =====================================================
        After reset, we have |0⟩^⊗n = all zeros in computational basis.
        This is NOT in the code space - it's a mixture of codewords and non-codewords.
        
        Measuring X stabilizers:
        1. Projects onto states with correct X parities
        2. These states are exactly the CSS codewords  
        3. The Z eigenvalue is preserved (reset gave +1, projection doesn't change it)
        4. Result: |0⟩_L with Z_L = +1 deterministically
        
        The measurement outcomes determine WHICH codeword basis state is projected:
        - Syndrome = 0 → |0⟩_L directly
        - Syndrome ≠ 0 → shifted |0⟩_L (Pauli frame, still valid Z eigenstate)
        
        We use MR (measure-reset) to reset ancillas for reuse. The measurement
        outcomes are intentionally not tracked as syndromes since this is for
        state preparation, not error detection.
        """
        if not block.x_ancilla_qubits:
            return
        
        # Get X stabilizer matrix (hx defines X stabilizers for CSS codes)
        # hx[i] gives support of i-th X stabilizer (positions of X operators)
        hx = getattr(self._inner_code, 'hx', None)
        if hx is None or not hasattr(hx, 'shape') or hx.size == 0:
            return
        
        # Prepare X ancillas in |+⟩ (H on |0⟩)
        circuit.append("H", block.x_ancilla_qubits)
        
        # Apply CNOTs for X stabilizers: CNOT(ancilla→data) computes X parity
        # X stabilizer = ⊗_i X_i on support qubits
        # CNOT propagates X from ancilla to data; measuring ancilla gives parity
        for stab_idx in range(min(hx.shape[0], len(block.x_ancilla_qubits))):
            anc = block.x_ancilla_qubits[stab_idx]
            support = np.where(hx[stab_idx] != 0)[0]
            for data_idx in support:
                if data_idx < len(block.data_qubits):
                    circuit.append("CX", [anc, block.data_qubits[data_idx]])
        
        # Measure X ancillas in X basis (H then MZ)
        circuit.append("H", block.x_ancilla_qubits)
        circuit.append("MR", block.x_ancilla_qubits)
        self._measurement_index += len(block.x_ancilla_qubits)
    
    def _emit_x_stabilizers(
        self,
        circuit: stim.Circuit,
        block: BlockInfoV2,
    ) -> List[int]:
        """
        Emit X stabilizer measurements for a block with layer-by-layer TICKs.
        
        Emits raw measurement circuit with TICKs between CNOT layers:
            H → TICK → [CNOT layer 0] → TICK → [CNOT layer 1] → ... → H → MR
        
        Layer-by-layer emission enables the noise model to distinguish errors
        in different CNOT layers, which is critical for accurate DEM generation.
        
        CNOT Schedule: Each layer contains at most one CNOT per ancilla.
        For X stabilizers, CNOT direction is ancilla → data (control on ancilla).
        
        Returns
        -------
        List[int]
            The measurement indices for X ancilla measurements.
        """
        if not block.x_ancilla_qubits:
            return []
        
        # Prepare X ancillas in |+⟩
        circuit.append("H", block.x_ancilla_qubits)
        circuit.append("TICK")
        
        # Build CNOT schedule: layer_idx -> [(ancilla, data), ...]
        hx = getattr(self._inner_code, 'hx', None)
        if hx is not None and hasattr(hx, 'shape') and hx.size > 0:
            # Compute max weight to determine number of layers
            max_weight = 0
            for stab_idx in range(hx.shape[0]):
                weight = np.sum(hx[stab_idx] != 0)
                max_weight = max(max_weight, weight)
            
            # For each layer, emit CNOTs for the layer_idx-th qubit in each stabilizer's support
            for layer_idx in range(max_weight):
                cnot_pairs = []
                for stab_idx in range(hx.shape[0]):
                    if stab_idx < len(block.x_ancilla_qubits):
                        anc = block.x_ancilla_qubits[stab_idx]
                        support = np.where(hx[stab_idx] != 0)[0]
                        if layer_idx < len(support):
                            data_idx = support[layer_idx]
                            if data_idx < len(block.data_qubits):
                                cnot_pairs.append((anc, block.data_qubits[data_idx]))
                
                # Emit all CNOTs for this layer
                for anc, data in cnot_pairs:
                    circuit.append("CX", [anc, data])
                
                # TICK after each CNOT layer
                if cnot_pairs:
                    circuit.append("TICK")
        
        # Measure X ancillas
        circuit.append("H", block.x_ancilla_qubits)
        
        # Record measurement indices
        base_meas_idx = self._measurement_index
        circuit.append("MR", block.x_ancilla_qubits)
        self._measurement_index += len(block.x_ancilla_qubits)
        
        # Track for decoder (flat list for backward compatibility)
        if block.block_id not in self._inner_stabilizer_measurements:
            self._inner_stabilizer_measurements[block.block_id] = {'x_anc': [], 'z_anc': []}
        meas_indices = [base_meas_idx + i for i in range(len(block.x_ancilla_qubits))]
        self._inner_stabilizer_measurements[block.block_id]['x_anc'].extend(meas_indices)
        
        return meas_indices
    
    def _emit_z_stabilizers(
        self,
        circuit: stim.Circuit,
        block: BlockInfoV2,
    ) -> List[int]:
        """
        Emit Z stabilizer measurements for a block with layer-by-layer TICKs.
        
        Emits raw measurement circuit with TICKs between CNOT layers:
            [CNOT layer 0] → TICK → [CNOT layer 1] → TICK → ... → MR
        
        Layer-by-layer emission enables the noise model to distinguish errors
        in different CNOT layers, which is critical for accurate DEM generation.
        
        CNOT Schedule: Each layer contains at most one CNOT per ancilla.
        For Z stabilizers, CNOT direction is data → ancilla (control on data).
        
        Returns
        -------
        List[int]
            The measurement indices for Z ancilla measurements.
        """
        if not block.z_ancilla_qubits:
            return []
        
        # Build CNOT schedule: layer_idx -> [(data, ancilla), ...]
        hz = getattr(self._inner_code, 'hz', None)
        if hz is not None and hasattr(hz, 'shape') and hz.size > 0:
            # Compute max weight to determine number of layers
            max_weight = 0
            for stab_idx in range(hz.shape[0]):
                weight = np.sum(hz[stab_idx] != 0)
                max_weight = max(max_weight, weight)
            
            # For each layer, emit CNOTs for the layer_idx-th qubit in each stabilizer's support
            for layer_idx in range(max_weight):
                cnot_pairs = []
                for stab_idx in range(hz.shape[0]):
                    if stab_idx < len(block.z_ancilla_qubits):
                        anc = block.z_ancilla_qubits[stab_idx]
                        support = np.where(hz[stab_idx] != 0)[0]
                        if layer_idx < len(support):
                            data_idx = support[layer_idx]
                            if data_idx < len(block.data_qubits):
                                cnot_pairs.append((block.data_qubits[data_idx], anc))
                
                # Emit all CNOTs for this layer
                for data, anc in cnot_pairs:
                    circuit.append("CX", [data, anc])
                
                # TICK after each CNOT layer
                if cnot_pairs:
                    circuit.append("TICK")
        
        # Record measurement indices
        base_meas_idx = self._measurement_index
        circuit.append("MR", block.z_ancilla_qubits)
        self._measurement_index += len(block.z_ancilla_qubits)
        
        # Track for decoder (flat list for backward compatibility)
        if block.block_id not in self._inner_stabilizer_measurements:
            self._inner_stabilizer_measurements[block.block_id] = {'x_anc': [], 'z_anc': []}
        meas_indices = [base_meas_idx + i for i in range(len(block.z_ancilla_qubits))]
        self._inner_stabilizer_measurements[block.block_id]['z_anc'].extend(meas_indices)
        
        return meas_indices
    
    # =========================================================================
    # Circuit Emission - Inner EC Segment (MCM Architecture)
    # =========================================================================

    def emit_inner_ec_segment(
        self,
        circuit: stim.Circuit,
        block_ids: List[int],
        n_rounds: int,
        context: str,
        outer_round: int = -1,
    ) -> Dict[int, int]:
        """
        Emit a complete inner EC segment with proper TICK placement.
        
        This is the core method for aggregating multiple syndrome extraction
        rounds into a single InnerECInstance per block. Each round is separated
        by a TICK instruction to enable temporal error detection.
        
        Architecture:
        =============
        For n_rounds=d (inner code distance), this emits:
        
            [Round 1 all blocks] → TICK → [Round 2 all blocks] → TICK → ... → [Round d all blocks] → TICK
        
        Each block gets ONE InnerECInstance containing all d*n_stabs measurements.
        The decoder can then:
        - Compare rounds for temporal DEM: D[r,s] = S[r+1,s] XOR S[r,s]
        - Use majority vote across rounds
        
        CRITICAL: TICKs between rounds are required for the noise model to
        insert errors that affect different rounds differently. Without TICKs,
        circuit noise affects all rounds identically and temporal DEM fails.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into
        block_ids : List[int]
            Block IDs to apply EC to
        n_rounds : int
            Number of syndrome extraction rounds (should be >= inner code distance)
        context : str
            EC context: 'init', 'pre_outer', 'intra_gadget', 'post_outer', 'final'
        outer_round : int
            Outer round index (-1 for init/final)
            
        Returns
        -------
        Dict[int, int]
            Mapping of block_id -> InnerECInstance ID created for that block
        """
        import warnings
        
        # Validate n_rounds for decoder requirements
        if n_rounds < 2:
            warnings.warn(
                f"emit_inner_ec_segment: n_rounds={n_rounds} < 2. "
                f"Temporal DEM requires at least 2 rounds to compute syndrome changes. "
                f"Consider using n_rounds >= inner code distance.",
                UserWarning
            )
        if n_rounds < 3:
            warnings.warn(
                f"emit_inner_ec_segment: n_rounds={n_rounds} < 3. "
                f"Majority vote decoding requires at least 3 rounds. "
                f"Consider using n_rounds >= 3 for robust decoding.",
                UserWarning
            )
        
        # Create InnerECInstance for each block (one instance per block per segment)
        block_to_instance_id: Dict[int, int] = {}
        block_instances: Dict[int, InnerECInstance] = {}
        
        for block_id in block_ids:
            instance_id = self._ec_instance_counter
            self._ec_instance_counter += 1
            
            ec_instance = InnerECInstance(
                instance_id=instance_id,
                block_id=block_id,
                x_anc_measurements=[],
                z_anc_measurements=[],
                context=context,
                outer_round=outer_round,
                n_rounds=n_rounds,
            )
            block_instances[block_id] = ec_instance
            block_to_instance_id[block_id] = instance_id
        
        # Emit n_rounds of syndrome extraction with TICKs between rounds
        for round_idx in range(n_rounds):
            # Emit one complete round of stabilizer measurements for all blocks
            for block_id in block_ids:
                block = self._blocks[block_id]
                
                # X stabilizer measurement (detects Z errors)
                x_meas_indices = self._emit_x_stabilizers(circuit, block)
                block_instances[block_id].x_anc_measurements.extend(x_meas_indices)
                
                # Z stabilizer measurement (detects X errors)
                z_meas_indices = self._emit_z_stabilizers(circuit, block)
                block_instances[block_id].z_anc_measurements.extend(z_meas_indices)
            
            # TICK between rounds (critical for temporal error detection)
            circuit.append("TICK")
        
        # Add all instances to measurement-centric metadata
        for block_id, ec_instance in block_instances.items():
            self._measurement_centric_metadata.add_ec_instance(ec_instance)
        
        return block_to_instance_id
    
    def emit_inner_ec_segment_data_blocks(
        self,
        circuit: stim.Circuit,
        n_rounds: int,
        context: str,
        outer_round: int = -1,
    ) -> Dict[int, int]:
        """
        Emit inner EC segment on all data blocks.
        
        Convenience method that calls emit_inner_ec_segment with all data block IDs.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into
        n_rounds : int
            Number of syndrome extraction rounds
        context : str
            EC context: 'init', 'pre_outer', 'post_outer', 'final'
        outer_round : int
            Outer round index
            
        Returns
        -------
        Dict[int, int]
            Mapping of block_id -> InnerECInstance ID
        """
        data_block_ids = [
            block.block_id for block in self._blocks.values()
            if block.block_type == BlockType.DATA
        ]
        return self.emit_inner_ec_segment(
            circuit, data_block_ids, n_rounds, context, outer_round
        )
    
    def emit_inner_ec_segment_all_blocks(
        self,
        circuit: stim.Circuit,
        n_rounds: int,
        context: str,
        outer_round: int = -1,
    ) -> Dict[int, int]:
        """
        Emit inner EC segment on all blocks (data + ancilla).
        
        Convenience method that calls emit_inner_ec_segment with all block IDs.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into
        n_rounds : int
            Number of syndrome extraction rounds
        context : str
            EC context: 'init', 'pre_outer', 'intra_gadget', 'post_outer', 'final'
        outer_round : int
            Outer round index
            
        Returns
        -------
        Dict[int, int]
            Mapping of block_id -> InnerECInstance ID
        """
        all_block_ids = list(self._blocks.keys())
        return self.emit_inner_ec_segment(
            circuit, all_block_ids, n_rounds, context, outer_round
        )
    
    # =========================================================================
    # Circuit Emission - Logical Measurement
    # =========================================================================
    
    def emit_measure_logical_z(
        self,
        circuit: stim.Circuit,
        block_id: int,
    ) -> List[int]:
        """Measure block in logical Z basis, return measurement indices."""
        block = self._blocks[block_id]
        
        base_meas_idx = self._measurement_index
        circuit.append("MR", block.data_qubits)
        self._measurement_index += len(block.data_qubits)
        
        return list(range(base_meas_idx, base_meas_idx + len(block.data_qubits)))
    
    def emit_measure_logical_x(
        self,
        circuit: stim.Circuit,
        block_id: int,
    ) -> List[int]:
        """Measure block in logical X basis, return measurement indices."""
        block = self._blocks[block_id]
        
        # Apply H to measure in X basis
        circuit.append("H", block.data_qubits)
        
        base_meas_idx = self._measurement_index
        circuit.append("MR", block.data_qubits)
        self._measurement_index += len(block.data_qubits)
        
        return list(range(base_meas_idx, base_meas_idx + len(block.data_qubits)))
    
    def emit_final_data_measurement(
        self,
        circuit: stim.Circuit,
        basis: str = "Z",
    ) -> Dict[int, List[int]]:
        """
        Measure all data blocks in specified basis.
        
        Returns mapping of block_id -> measurement indices for decoder.
        No DETECTOR instructions are emitted - decoder extracts syndrome
        from raw measurements.
        """
        final_measurements: Dict[int, List[int]] = {}
        
        for block in self._blocks.values():
            if block.block_type == BlockType.DATA:
                if basis.upper() == "X":
                    meas = self.emit_measure_logical_x(circuit, block.block_id)
                else:
                    meas = self.emit_measure_logical_z(circuit, block.block_id)
                final_measurements[block.block_id] = meas
        
        return final_measurements
    
    # =========================================================================
    # Observable Emission
    # =========================================================================
    
    def emit_logical_observable(
        self,
        circuit: stim.Circuit,
        outer_code: "Code",
        final_measurements: Dict[int, List[int]],
        basis: str = "Z",
        observable_idx: int = 0,
    ) -> None:
        """Emit OBSERVABLE_INCLUDE for the concatenated logical."""
        # Get outer logical support
        outer_support = get_logical_support(outer_code, basis, 0)
        inner_support = get_logical_support(self._inner_code, basis, 0)
        
        obs_targets = []
        for block_id in outer_support:
            if block_id in final_measurements:
                block_meas = final_measurements[block_id]
                for inner_idx in inner_support:
                    if inner_idx < len(block_meas):
                        meas_idx = block_meas[inner_idx]
                        obs_targets.append(stim.target_rec(meas_idx - self._measurement_index))
        
        if obs_targets:
            circuit.append("OBSERVABLE_INCLUDE", obs_targets, [observable_idx])
    
    # =========================================================================
    # Metadata
    # =========================================================================
    
    def get_block_info_dict(self) -> Dict[int, Dict]:
        """Get block info as serializable dict."""
        return {
            block_id: {
                'block_id': block.block_id,
                'block_type': block.block_type.name,
                'name': block.name,
                'data_qubits': block.data_qubits,
                'x_ancilla_qubits': block.x_ancilla_qubits,
                'z_ancilla_qubits': block.z_ancilla_qubits,
            }
            for block_id, block in self._blocks.items()
        }
    
    def get_inner_stabilizer_measurements(self) -> Dict[int, Dict[str, List[int]]]:
        """
        Get inner stabilizer measurement indices organized by block ID.
        
        Returns measurement indices recorded during emit_inner_ec_round calls.
        Used by HierarchicalV5Decoder to extract inner syndrome from raw measurements.
        
        Returns
        -------
        Dict[int, Dict[str, List[int]]]
            block_id -> {
                'x_anc': [meas_idx, ...],  # All X ancilla measurement indices
                'z_anc': [meas_idx, ...],  # All Z ancilla measurement indices
            }
        """
        # Ensure all blocks have entries
        result: Dict[int, Dict[str, List[int]]] = {}
        for block_id in self._blocks.keys():
            if block_id in self._inner_stabilizer_measurements:
                result[block_id] = self._inner_stabilizer_measurements[block_id]
            else:
                result[block_id] = {'x_anc': [], 'z_anc': []}
        return result
    

    # =========================================================================
    # Measurement-Centric Model API
    # =========================================================================
    
    def set_ec_context(self, context: str, outer_round: int = -1) -> None:
        """
        Set the context for subsequent emit_inner_ec_round() calls.
        
        This context is stored in each InnerECInstance created.
        
        Parameters
        ----------
        context : str
            One of: 'init', 'pre_outer', 'intra_gadget', 'post_outer', 'final'
        outer_round : int
            Which outer round this belongs to (-1 for init/final)
        """
        self._current_ec_context = context
        self._current_outer_round = outer_round
    
    def get_measurement_centric_metadata(self) -> MeasurementCentricMetadata:
        """
        Get measurement-centric metadata for decoder.
        
        Returns the new format metadata that directly maps to decoder needs.
        
        Returns
        -------
        MeasurementCentricMetadata
            Contains:
            - inner_ec_instances: All EC instances
            - block_to_ec_instances: Block -> instance_id mapping
            - outer_syndrome_values: Outer measurements (populated by outer engine)
            - final_data_measurements: Final measurements
        """
        return self._measurement_centric_metadata
    
    def get_last_ec_instance_id(self, block_id: int) -> int:
        """
        Get the most recent EC instance ID for a block.
        
        Used by OuterStabilizerEngine to link ancilla EC to outer syndrome.
        
        Parameters
        ----------
        block_id : int
            Block to query
            
        Returns
        -------
        int
            The most recent EC instance ID, or -1 if none
        """
        ec_ids = self._measurement_centric_metadata.block_to_ec_instances.get(block_id, [])
        return ec_ids[-1] if ec_ids else -1
