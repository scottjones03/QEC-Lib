# src/qectostim/experiments/outer_stabilizer_engine_v3.py
"""
Outer Stabilizer Engine V3: Fault-Tolerant exRec-Based Architecture.

This module implements the CORRECT fault-tolerant protocol for outer stabilizer
measurement in concatenated codes, following the Extended Rectangle (exRec)
formalism from Aliferis-Gottesman-Preskill (quant-ph/0504218).

HierarchicalV6Decoder Integration
=================================

The engine emits outer stabilizer gadgets with segment metadata for the
HierarchicalV6Decoder 4-phase pipeline:

  Phase 2 (Correct Outer Syndrome) uses this engine's metadata:
    - Each outer_stab segment has ancilla_ec_segment_ids
    - X stabilizers: ancilla measured in Z basis → X errors flip result
      → Use inner_x_correction (from z_anc detectors)
    - Z stabilizers: ancilla measured in X basis → Z errors flip result  
      → Use inner_z_correction (from x_anc detectors)
    - corrected_syn = raw_logical XOR gadget_correction

  CNOT Direction (Steane-style):
    - Both X and Z stabilizers use CNOT(data → ancilla)
    - Z stabilizer: ancilla in |+⟩, measure X basis
    - X stabilizer: ancilla in |0⟩, measure Z basis

Literature Foundation
=====================

**Aliferis-Gottesman-Preskill (AGP) Threshold Theorem** (quant-ph/0504218):
    "Quantum accuracy threshold for concatenated distance-3 codes"
    
The AGP paper establishes that fault-tolerant quantum computation is possible
when physical error rates are below a threshold. The key construct is the
**extended rectangle (exRec)**, which ensures that a single fault anywhere
produces at most one error at the output boundary.

**Extended Rectangle (exRec) Structure**:
    
    ┌────────────────────────────────────────────────────────────┐
    │                         exRec                               │
    │  ┌─────────────┐    ┌───────────────┐    ┌─────────────┐   │
    │  │  Leading    │ →  │   Logical     │ →  │  Trailing   │   │
    │  │     EC      │    │     Gate      │    │     EC      │   │
    │  └─────────────┘    └───────────────┘    └─────────────┘   │
    └────────────────────────────────────────────────────────────┘

For a sequence of logical gates, trailing EC of exRec_n merges with leading EC
of exRec_{n+1}, giving the optimized structure:

    [EC] → [Gate₁] → [EC] → [Gate₂] → [EC] → ... → [Gate_k] → [EC]

**FT Property**: A single fault in the exRec produces at most one error at the
output. For distance-3 codes, the trailing EC corrects this single error.

**Steane-Style Encoded Ancilla** (Steane, PRS 1996):
For outer stabilizer Z̄₁Z̄₂...Z̄_w measurement:
    1. Prepare ancilla BLOCK in |+̄⟩ (n-qubit logical plus state)
    2. Transversal CNOT: data_block[i] → ancilla[i] for each qubit i
    3. Measure ancilla block in X basis (n physical measurements)
    4. DECODE inner code on measurements → logical X̄ = outer syndrome bit

This is fault-tolerant because:
    - Single fault affects one physical qubit per block
    - Inner code (distance-3) corrects single errors
    - Outer syndrome is obtained by decoding, not direct measurement

Architecture
============

This engine implements the exRec-based FT protocol with segment tracking:

┌─────────────────────────────────────────────────────────────────────────────┐
│              EXREC-BASED FT OUTER STABILIZER MEASUREMENT                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE A: FT STATE PREPARATION                                              │
│    A1. Reset ancilla blocks                                                 │
│    A2. Prepare logical basis state:                                         │
│        - X stabilizers: |+̄⟩ (H on all data qubits)                         │
│        - Z stabilizers: |0̄⟩ (from reset)                                   │
│                                                                             │
│  PHASE B: LEADING INNER EC (exRec start) → creates inner_ec segment         │
│    B1. Inner EC on ALL blocks (data + ancilla)                              │
│    B2. Track ancilla_ec_segment_ids for this gadget                         │
│                                                                             │
│  PHASE C: GATE LAYERS WITH INTERLEAVED EC (exRec core)                      │
│    For each gate layer:                                                     │
│      C1. Apply logical CNOTs via LogicalGateDispatcher                      │
│          - Both X and Z stabilizers: CNOT(data → ancilla)                   │
│      C2. Inner EC on ALL blocks → additional inner_ec segments              │
│      C3. Accumulate ancilla_ec_segment_ids                                  │
│                                                                             │
│  PHASE D: TRANSVERSAL MEASUREMENT                                           │
│    D1. Apply H for X stabilizers (Z→X basis change)                         │
│    D2. Measure all ancilla block data qubits (M or MR)                      │
│    D3. Record ancilla_meas_indices for decoder                              │
│                                                                             │
│  PHASE E: OUTER SYNDROME EXTRACTION (HierarchicalV6Decoder Phase 2)         │
│    E1. For each outer_stab segment, decoder gets ancilla_ec_segment_ids     │
│    E2. X stab (anc Z-meas): gadget_corr = XOR(inner_x_correction[seg_ids])  │
│    E3. Z stab (anc X-meas): gadget_corr = XOR(inner_z_correction[seg_ids])  │
│    E4. corrected_syn = raw_logical XOR gadget_corr                          │
│                                                                             │
│  SEGMENT METADATA OUTPUT:                                                   │
│  - outer_stab segments with ancilla_ec_segment_ids, ancilla_meas_indices    │
│  - stab_to_ancilla_block mapping                                            │
│  - Enables HierarchicalV6Decoder Phase 2 correction                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

This separation of concerns is consistent with AGP's hierarchical structure:
- Level 0 (Stim): Inner syndrome decoded  within each block
- Level 1 (Decoder): Outer syndrome from decoded ancilla block logicals

References
==========
- Aliferis, Gottesman, Preskill: "Quantum accuracy threshold for concatenated
  distance-3 codes" (quant-ph/0504218)
- Steane: "Multiple-particle interference and quantum error correction" 
  (Proc. R. Soc. Lond. A 452, 1996)
- Knill: "Quantum computing with realistically noisy devices" (Nature 2005)
- Chamberland et al: "Flag fault-tolerant error correction" (Quantum 2018)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING
from collections import defaultdict

import numpy as np
import stim

if TYPE_CHECKING:
    from qectostim.codes.abstract_code import Code

from qectostim.experiments.stabilizer_rounds import (
    DetectorContext,
    get_logical_support,
)
from qectostim.experiments.logical_block_manager_v2 import (
    LogicalBlockManagerV2, 
    BlockInfoV2,
    BlockType,
)
from qectostim.experiments.logical_gate_dispatcher import (
    LogicalGateDispatcher,
    GateMethod,
    GateType,
)
from qectostim.experiments.measurement_model import OuterSyndromeValue


@dataclass
class OuterStabInfo:
    """
    Information about a single outer stabilizer.
    
    Per AGP terminology, this represents one generator of the outer code's
    stabilizer group that we need to measure fault-tolerantly.
    """
    stab_idx: int
    stab_type: str  # "X" or "Z"
    support: List[int]  # Data block indices in stabilizer support
    coord: Optional[Tuple[float, ...]] = None
    
    @property
    def weight(self) -> int:
        """Number of data blocks in support (weight of outer stabilizer)."""
        return len(self.support)


@dataclass
class OuterStabSet:
    """
    A set of outer stabilizers that can be measured in parallel.
    
    Parallel measurement is possible when stabilizers:
    1. Don't share data blocks (no ancilla conflicts)
    2. Have sufficient ancilla blocks available
    
    Each stabilizer in the set gets its own ancilla block.
    """
    set_idx: int
    x_stabilizers: List[OuterStabInfo] = field(default_factory=list)
    z_stabilizers: List[OuterStabInfo] = field(default_factory=list)
    x_ancilla_assignments: Dict[int, int] = field(default_factory=dict)
    z_ancilla_assignments: Dict[int, int] = field(default_factory=dict)
    
    @property
    def all_x_ancillas(self) -> Set[int]:
        return set(self.x_ancilla_assignments.values())
    
    @property
    def all_z_ancillas(self) -> Set[int]:
        return set(self.z_ancilla_assignments.values())
    
    @property
    def all_ancillas(self) -> Set[int]:
        return self.all_x_ancillas | self.all_z_ancillas


@dataclass
class OuterMeasResult:
    """Result of measuring one outer stabilizer."""
    stab_info: OuterStabInfo
    ancilla_block_id: int
    measurement_indices: List[int]  # Physical measurement indices in ancilla block


class OuterStabilizerEngineV3:
    """
    Fault-Tolerant Outer Stabilizer Engine using exRec Architecture.
    
    This engine implements the Aliferis-Gottesman-Preskill (AGP) extended
    rectangle formalism for fault-tolerant outer stabilizer measurement
    in concatenated codes.
    
    Key Features (per AGP):
    =======================
    
    1. **exRec Structure**: Every logical gate is sandwiched between inner EC
       rounds, ensuring single faults produce at most one error at boundaries.
       
    2. **LogicalGateDispatcher**: Uses dispatcher for logical CNOTs, supporting
       transversal gates, lattice surgery, or teleportation-based gates.
       
    3. **Segment-Based Metadata for HierarchicalV6Decoder**:
       - Each outer_stab segment contains ancilla_ec_segment_ids
       - ancilla_meas_indices for raw transversal measurement
       - stab_to_ancilla_block mapping
       
    4. **Correction Type Mapping (Phase 2 of decoder)**:
       - X stabilizers: ancilla measured in Z basis
         → X errors on ancilla flip measurement
         → Use inner_x_correction (from z_anc detectors)
       - Z stabilizers: ancilla measured in X basis
         → Z errors on ancilla flip measurement
         → Use inner_z_correction (from x_anc detectors)
    
    Parameters
    ----------
    outer_code : Code
        The outer code whose stabilizers we measure.
    block_manager : LogicalBlockManagerV2
        Manager for inner code blocks with segment tracking.
    gate_dispatcher : LogicalGateDispatcher
        Dispatcher for logical gate implementations.
    gate_method : GateMethod
        Method for logical CNOTs (AUTO, TRANSVERSAL, SURGERY, etc.).
    inner_rounds : int, optional
        Number of inner syndrome extraction rounds per EC segment.
        Should be >= inner code distance for proper error correction.
        Default is 1 for backward compatibility.
    """
    
    def __init__(
        self,
        outer_code: "Code",
        block_manager: LogicalBlockManagerV2,
        gate_dispatcher: LogicalGateDispatcher,
        gate_method: GateMethod = GateMethod.AUTO,
        inner_rounds: int = 1,
    ):
        self._outer_code = outer_code
        self._block_manager = block_manager
        self._gate_dispatcher = gate_dispatcher
        self._gate_method = gate_method
        self._inner_rounds = inner_rounds
        
        # Parse outer code stabilizers
        self._x_stabilizers: List[OuterStabInfo] = []
        self._z_stabilizers: List[OuterStabInfo] = []
        self._parse_outer_stabilizers()
        
        # Build parallel sets using graph coloring
        self._x_sets: List[OuterStabSet] = []
        self._z_sets: List[OuterStabSet] = []
        self._mixed_sets: List[OuterStabSet] = []
        self._build_parallel_sets()
        
        # Track ancilla logical measurements for decoder metadata
        # Key: (round_idx, stab_type, stab_idx) -> List[measurement_indices]
        self._ancilla_logical_measurements: Dict[Tuple[int, str, int], List[int]] = {}
        
        # Track PRE-CNOT ancilla measurements for tight syndrome comparison
        # Key: (round_idx, stab_type, stab_idx, layer_idx) -> List[measurement_indices]
        self._pre_cnot_measurements: Dict[Tuple[int, str, int, int], List[int]] = {}
        
        # Track POST-CNOT ancilla measurements (paired with pre-CNOT)
        # Key: (round_idx, stab_type, stab_idx, layer_idx) -> List[measurement_indices]
        self._post_cnot_measurements: Dict[Tuple[int, str, int, int], List[int]] = {}
        
        self._current_round = 0
    
    # =========================================================================
    # Outer Stabilizer Parsing
    # =========================================================================
    
    def _parse_outer_stabilizers(self) -> None:
        """
        Parse outer code's stabilizer generators from hx/hz matrices.
        
        Per CSS code structure, X stabilizers come from hx matrix rows,
        Z stabilizers from hz matrix rows.
        """
        outer = self._outer_code
        
        # X stabilizers from hx matrix
        hx = getattr(outer, 'hx', None)
        if hx is not None and hasattr(hx, 'shape') and hx.size > 0:
            hx_arr = np.asarray(hx)
            for i in range(hx_arr.shape[0]):
                support = list(np.where(hx_arr[i] != 0)[0])
                self._x_stabilizers.append(OuterStabInfo(
                    stab_idx=i, stab_type="X", support=support
                ))
        
        # Z stabilizers from hz matrix  
        hz = getattr(outer, 'hz', None)
        if hz is not None and hasattr(hz, 'shape') and hz.size > 0:
            hz_arr = np.asarray(hz)
            for i in range(hz_arr.shape[0]):
                support = list(np.where(hz_arr[i] != 0)[0])
                self._z_stabilizers.append(OuterStabInfo(
                    stab_idx=i, stab_type="Z", support=support
                ))
    
    # =========================================================================
    # Parallel Set Building (Graph Coloring per AGP Section 4)
    # =========================================================================
    
    def _build_parallel_sets(self) -> None:
        """
        Build parallel stabilizer sets using graph coloring.
        
        Per AGP, stabilizers can be measured in parallel if they don't
        share data blocks (which would create ancilla conflicts).
        """
        self._x_sets = self._build_sets_for_type(self._x_stabilizers, "X")
        self._z_sets = self._build_sets_for_type(self._z_stabilizers, "Z")
        self._build_mixed_sets()
    
    def _build_sets_for_type(
        self,
        stabilizers: List[OuterStabInfo],
        stab_type: str,
    ) -> List[OuterStabSet]:
        """
        Build parallel sets for one stabilizer type using greedy coloring.
        
        Per AGP (quant-ph/0504218), each extended rectangle (exRec) must be
        self-contained - ancilla blocks used in one exRec cannot be reused
        until that exRec completes. This means:
        
        1. Stabilizers that CONFLICT (share data blocks) must be in DIFFERENT sets
           (they would interfere if measured simultaneously)
           
        2. Each stabilizer needs its OWN dedicated ancilla block
           (ancillas are reset at the start of each exRec, so reusing an
           ancilla across sets would destroy detector coherence)
        
        The greedy coloring assigns stabilizers to sets such that no two
        conflicting stabilizers share a set. Each stabilizer then gets a
        UNIQUE ancilla block via global counting across all sets.
        
        Architecture:
        -------------
        For Steane [[7,1,3]] outer code with 3 X and 3 Z stabilizers:
        - X stabilizers: X0, X1, X2 (all pairwise overlap → 3 separate sets)
        - Z stabilizers: Z0, Z1, Z2 (all pairwise overlap → 3 separate sets)
        - X ancillas: blocks 7, 8, 9 (one per X stabilizer)
        - Z ancillas: blocks 10, 11, 12 (one per Z stabilizer)
        
        Each exRec measures ONE X and ONE Z stabilizer using dedicated ancillas,
        ensuring no cross-contamination between exRecs.
        """
        if not stabilizers:
            return []
        
        # Determine ancilla pool
        n_ancilla = (self._block_manager.n_x_ancilla_blocks if stab_type == "X" 
                     else self._block_manager.n_z_ancilla_blocks)
        ancilla_offset = (self._block_manager.x_ancilla_offset if stab_type == "X"
                         else self._block_manager.z_ancilla_offset)
        
        # Build conflict graph (stabilizers conflict if they share data blocks)
        n = len(stabilizers)
        conflicts = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if set(stabilizers[i].support) & set(stabilizers[j].support):
                    conflicts[i][j] = True
                    conflicts[j][i] = True
        
        # Greedy graph coloring
        colors = [-1] * n
        sets: List[OuterStabSet] = []
        
        for i in range(n):
            used_colors = set()
            for j in range(n):
                if conflicts[i][j] and colors[j] >= 0:
                    used_colors.add(colors[j])
            
            color = 0
            while True:
                if color not in used_colors:
                    count_in_color = sum(1 for c in colors if c == color)
                    if count_in_color < n_ancilla:
                        break
                color += 1
            
            colors[i] = color
            
            while len(sets) <= color:
                sets.append(OuterStabSet(set_idx=len(sets)))
        
        # Assign stabilizers to sets with GLOBALLY UNIQUE ancilla block assignments
        # Each stabilizer gets its own dedicated ancilla to prevent cross-exRec interference
        global_ancilla_counter = 0
        
        for i, stab in enumerate(stabilizers):
            color = colors[i]
            stab_set = sets[color]
            
            # Assign unique ancilla block using global counter (not per-set counter)
            ancilla_block_id = ancilla_offset + global_ancilla_counter
            global_ancilla_counter += 1
            
            if stab_type == "X":
                stab_set.x_stabilizers.append(stab)
                stab_set.x_ancilla_assignments[stab.stab_idx] = ancilla_block_id
            else:
                stab_set.z_stabilizers.append(stab)
                stab_set.z_ancilla_assignments[stab.stab_idx] = ancilla_block_id
        
        return sets
    
    def _build_mixed_sets(self) -> None:
        """Build mixed X+Z sets for interleaved measurement."""
        n_x = len(self._x_sets)
        n_z = len(self._z_sets)
        n_mixed = min(n_x, n_z)
        
        self._mixed_sets = []
        
        for i in range(n_mixed):
            x_set = self._x_sets[i]
            z_set = self._z_sets[i]
            
            mixed = OuterStabSet(
                set_idx=len(self._mixed_sets),
                x_stabilizers=list(x_set.x_stabilizers),
                z_stabilizers=list(z_set.z_stabilizers),
                x_ancilla_assignments=dict(x_set.x_ancilla_assignments),
                z_ancilla_assignments=dict(z_set.z_ancilla_assignments),
            )
            self._mixed_sets.append(mixed)
        
        for i in range(n_mixed, n_x):
            x_set = self._x_sets[i]
            mixed = OuterStabSet(
                set_idx=len(self._mixed_sets),
                x_stabilizers=list(x_set.x_stabilizers),
                x_ancilla_assignments=dict(x_set.x_ancilla_assignments),
            )
            self._mixed_sets.append(mixed)
        
        for i in range(n_mixed, n_z):
            z_set = self._z_sets[i]
            mixed = OuterStabSet(
                set_idx=len(self._mixed_sets),
                z_stabilizers=list(z_set.z_stabilizers),
                z_ancilla_assignments=dict(z_set.z_ancilla_assignments),
            )
            self._mixed_sets.append(mixed)
    
    def can_interleave(self) -> bool:
        """Check if interleaved X/Z scheduling is possible."""
        if len(self._x_stabilizers) == 0 or len(self._z_stabilizers) == 0:
            return False
        
        max_parallel_x = max((len(s.x_stabilizers) for s in self._x_sets), default=0)
        max_parallel_z = max((len(s.z_stabilizers) for s in self._z_sets), default=0)
        
        if max_parallel_x > self._block_manager.n_x_ancilla_blocks:
            return False
        if max_parallel_z > self._block_manager.n_z_ancilla_blocks:
            return False
        
        return True
    
    # =========================================================================
    # Main Entry Point
    # =========================================================================
    
    def emit_outer_round(
        self,
        circuit: stim.Circuit,
        round_idx: int,
        parallel: bool = True,
    ) -> List[OuterMeasResult]:
        """
        Emit one complete round of outer stabilizer measurement.
        
        Implements the exRec-based FT protocol from AGP:
        - Leading EC before first gate
        - EC between each gate layer  
        - Trailing EC after last gate


        ┌─────────────────────────────────────────────────────────────────────────┐
        │ PHASE A: FT STATE PREPARATION                                           │
        │ ─────────────────────────────────────────────────────────────────────── │
        │ A1. Reset ancilla blocks (→ |0̄⟩)                                        │
        │ A2. H on X ancillas (→ |+̄⟩ for X stabilizer measurement)                │
        │                                                                         │
        │ At this point: Ancilla is freshly prepared, data blocks have            │
        │ established inner syndrome baseline from previous round.                │
        └─────────────────────────────────────────────────────────────────────────┘
                                            ↓
        ┌─────────────────────────────────────────────────────────────────────────┐
        │ PHASE B: LEADING INNER EC                                               │
        │ ─────────────────────────────────────────────────────────────────────── │
        │ B1. Inner EC on ALL participating blocks                                │
        │                                                                         │
        │ Justification: Per AGP, leading EC catches errors from previous ops     │
        │ and establishes baseline before the gate. Data blocks have valid        │
        │ comparison with previous measurements. Ancilla is new each exRec.       │
        └─────────────────────────────────────────────────────────────────────────┘
                                            ↓
        ┌─────────────────────────────────────────────────────────────────────────┐
        │ PHASE C: GATE LAYERS + POST-GATE EC                                     │
        │ ─────────────────────────────────────────────────────────────────────── │
        │ For each gate layer:                                                    │
        │   C1. Outer CNOTs (data ↔ ancilla)                                      │
        │       - X stab: CNOT(x_anc → data) → mark data.Z baseline broken        │
        │       - Z stab: CNOT(data → z_anc) → mark data.X baseline broken        │
        │                                                                         │
        │   C2. Inner EC on participating blocks                                  │
        │       - After this EC, inner baseline is re-established for data blocks │
        │                                                                         │
        │ DETECTOR BOUNDARY PRINCIPLE (Critical Insight):                         │
        │ ─────────────────────────────────────────────────────────────────────── │
        │ The outer CNOT creates a "detector boundary":                           │
        │ - After CNOT(data → z_anc): data's inner X measurements have            │
        │   X sensitivity on z_anc DATA qubits (which were reset before CNOT)     │                              
        └─────────────────────────────────────────────────────────────────────────┘
                                            ↓
        ┌─────────────────────────────────────────────────────────────────────────┐
        │ PHASE D: TRANSVERSAL MEASUREMENT                                        │
        │ ─────────────────────────────────────────────────────────────────────── │
        │ D1. H on X ancillas (X → Z basis)                                       │
        │ D2. Measure all ancilla blocks (MR)                                     │
        │ D3. Decode inner syndrome to get logical measurement result             │
        │                                                                         │
        │ The ancilla measurement extracts the outer syndrome.                    │
        └─────────────────────────────────────────────────────────────────────────┘
                                            ↓
        ┌─────────────────────────────────────────────────────────────────────────┐
        │ PHASE E: COHERENCE UPDATE                                               │
        │ ─────────────────────────────────────────────────────────────────────── │
        │ Mark appropriate coherence flags for affected blocks.                   │
        └─────────────────────────────────────────────────────────────────────────┘
                
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        round_idx : int
            Current round index.
        parallel : bool
            If True and can_interleave(), use parallel X+Z mode.
            
        Returns
        -------
        List[OuterMeasResult]
            Results of all outer stabilizer measurements.
        """
        self._current_round = round_idx
        
        if parallel and self.can_interleave():
            return self._emit_interleaved_round(circuit)
        else:
            return self._emit_sequential_round(circuit)
    
    def _emit_interleaved_round(self, circuit: stim.Circuit) -> List[OuterMeasResult]:
        """Emit interleaved X+Z measurement (X and Z share EC rounds)."""
        all_results = []
        
        for stab_set in self._mixed_sets:
            results = self._emit_stabilizer_set_exrec(circuit, stab_set)
            all_results.extend(results)
        
        return all_results
    
    def _emit_sequential_round(self, circuit: stim.Circuit) -> List[OuterMeasResult]:
        """Emit sequential X then Z measurement."""
        all_results = []
        
        for x_set in self._x_sets:
            results = self._emit_stabilizer_set_exrec(circuit, x_set)
            all_results.extend(results)
        
        for z_set in self._z_sets:
            results = self._emit_stabilizer_set_exrec(circuit, z_set)
            all_results.extend(results)
        
        return all_results
    
    # =========================================================================
    # NEW: Parallel Outer Stabilizer Emission
    # =========================================================================
    
    def emit_parallel_outer_stabilizers(
        self,
        circuit: stim.Circuit,
        round_idx: int,
        inner_rounds: int,
    ) -> List[OuterMeasResult]:
        """
        Emit ALL outer stabilizers (X and Z) in parallel.
        
        This is the new architecture that executes all 6 outer stabilizers
        simultaneously, rather than sequentially per set. For Steane [[7,1,3]]:
        
            X stabilizers (3): X0, X1, X2 → ancilla blocks 7, 8, 9
            Z stabilizers (3): Z0, Z1, Z2 → ancilla blocks 10, 11, 12
        
        PARALLEL EXECUTION STRUCTURE:
        =============================
        
        Phase 1: Prepare ALL ancilla blocks
            - Reset all 6 ancilla blocks
            - H on X ancillas (prepare |+⟩)
            - Z ancillas remain |0⟩
            
        Phase 2: Leading inner EC on ALL 13 blocks
            - emit_inner_ec_segment on blocks [0..12]
            - Creates ONE InnerECInstance per block with inner_rounds × n_stabs meas
            
        Phase 3: Transversal CNOTs in parallel layers
            - Compute conflict-free CNOT schedule
            - Each layer has non-conflicting block pairs
            - TICK between layers
            
        Phase 4: Trailing inner EC on ALL 13 blocks
            - Creates InnerECInstances used for outer syndrome correction
            
        Phase 5: Measure ALL ancilla blocks
            - H on X ancillas, then MR
            - MR on Z ancillas
            - Create OuterSyndromeValue for each stabilizer
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into
        round_idx : int
            Outer round index
        inner_rounds : int
            Number of inner syndrome extraction rounds per EC segment
            
        Returns
        -------
        List[OuterMeasResult]
            Results for all 6 outer stabilizer measurements
        """
        self._current_round = round_idx
        results: List[OuterMeasResult] = []
        
        # Always measure BOTH X and Z stabilizers
        # X stabilizers detect Z errors, Z stabilizers detect X errors
        # Both are needed regardless of measurement basis
        all_x_stabs = self._x_stabilizers
        all_z_stabs = self._z_stabilizers
        
        # Build unified ancilla assignment from all sets
        x_ancilla_map: Dict[int, int] = {}  # stab_idx -> ancilla_block_id
        z_ancilla_map: Dict[int, int] = {}
        
        if all_x_stabs:
            for stab_set in self._x_sets:
                x_ancilla_map.update(stab_set.x_ancilla_assignments)
        if all_z_stabs:
            for stab_set in self._z_sets:
                z_ancilla_map.update(stab_set.z_ancilla_assignments)
        
        x_ancillas = set(x_ancilla_map.values())
        z_ancillas = set(z_ancilla_map.values())
        all_ancillas = x_ancillas | z_ancillas
        
        if not all_ancillas:
            return results
        
        # All participating blocks (data + ancilla)
        data_blocks = set(range(self._block_manager.n_data_blocks))
        all_blocks = data_blocks | all_ancillas
        
        # =====================================================================
        # PHASE 1: Prepare ALL ancilla blocks in parallel
        # =====================================================================
        
        # Reset all ancilla blocks
        self._block_manager.emit_reset_blocks(circuit, list(all_ancillas))
        circuit.append("TICK")
        
        # CORRECT Steane-style ancilla preparation:
        # - X stabilizers: ancilla in |0⟩, CNOT(data→anc), measure Z basis (detects Z errors)
        # - Z stabilizers: ancilla in |+⟩, CNOT(data→anc), measure X basis (detects X errors)
        # X ancillas: already |0⟩ from reset - no additional prep needed
        for anc_id in z_ancillas:
            self._block_manager.emit_prepare_logical_plus(circuit, anc_id)
        circuit.append("TICK")
        
        # =====================================================================
        # PHASE 2: Leading inner EC on ALL blocks
        # =====================================================================
        
        self._block_manager.set_ec_context('gadget_leading_ec', round_idx)
        leading_ec_ids = self._block_manager.emit_inner_ec_segment(
            circuit,
            block_ids=sorted(all_blocks),
            n_rounds=inner_rounds,
            context='gadget_leading_ec',
            outer_round=round_idx,
        )
        
        # =====================================================================
        # PHASE 3: Transversal CNOTs in parallel layers
        # =====================================================================
        
        # Compute parallel CNOT schedule
        cnot_schedule = self._compute_parallel_cnot_schedule(
            all_x_stabs, all_z_stabs, x_ancilla_map, z_ancilla_map
        )
        
        for layer in cnot_schedule:
            for ctrl_block_id, targ_block_id, stab_type in layer:
                # Get block allocations from block manager
                ctrl_block = self._block_manager.get_block(ctrl_block_id)
                targ_block = self._block_manager.get_block(targ_block_id)
                
                # Emit transversal CNOT via dispatcher
                self._gate_dispatcher.emit_logical_cnot(
                    circuit, 
                    ctrl_block.allocation, 
                    targ_block.allocation, 
                    self._gate_method
                )
            circuit.append("TICK")
        
        # =====================================================================
        # PHASE 4: Trailing inner EC on ALL blocks
        # =====================================================================
        
        self._block_manager.set_ec_context('gadget_trailing_ec', round_idx)
        trailing_ec_ids = self._block_manager.emit_inner_ec_segment(
            circuit,
            block_ids=sorted(all_blocks),
            n_rounds=inner_rounds,
            context='gadget_trailing_ec',
            outer_round=round_idx,
        )
        
        # =====================================================================
        # PHASE 5: Measure ALL ancilla blocks and create OuterSyndromeValues
        # =====================================================================
        
        # Get MCM for adding outer syndrome values
        mcm = self._block_manager.get_measurement_centric_metadata()
        
        # Measure X ancillas (Z-basis measurement for X stabilizers)
        # CORRECT: X stabilizer uses ancilla in |0⟩, CNOT(data→anc), measure Z basis
        for stab in all_x_stabs:
            anc_block_id = x_ancilla_map.get(stab.stab_idx)
            if anc_block_id is None:
                continue
            
            block = self._block_manager.get_block(anc_block_id)
            
            # Direct Z-basis measurement (ancilla was in |0⟩, no H needed)
            base_meas_idx = self._block_manager.measurement_index
            circuit.append("MR", block.data_qubits)
            self._block_manager._measurement_index += len(block.data_qubits)
            meas_indices = [base_meas_idx + i for i in range(len(block.data_qubits))]
            
            # Store for decoder metadata
            self._ancilla_logical_measurements[(round_idx, 'X', stab.stab_idx)] = meas_indices
            
            # Create OuterSyndromeValue
            outer_syn = OuterSyndromeValue(
                outer_round=round_idx,
                stab_type='X',
                stab_idx=stab.stab_idx,
                ancilla_block_id=anc_block_id,
                ancilla_ec_instance_id=trailing_ec_ids.get(anc_block_id, -1),
                transversal_meas_indices=meas_indices,
            )
            mcm.add_outer_syndrome(outer_syn)
            
            results.append(OuterMeasResult(
                stab_info=stab,
                ancilla_block_id=anc_block_id,
                measurement_indices=meas_indices,
            ))
        
        # Measure Z ancillas (X-basis measurement for Z stabilizers)
        # CORRECT: Z stabilizer uses ancilla in |+⟩, CNOT(data→anc), measure X basis
        for stab in all_z_stabs:
            anc_block_id = z_ancilla_map.get(stab.stab_idx)
            if anc_block_id is None:
                continue
            
            block = self._block_manager.get_block(anc_block_id)
            
            # H to convert to X-basis measurement (ancilla was in |+⟩)
            circuit.append("H", block.data_qubits)
            
            # Z-basis measurement after H = X-basis measurement
            base_meas_idx = self._block_manager.measurement_index
            circuit.append("MR", block.data_qubits)
            self._block_manager._measurement_index += len(block.data_qubits)
            meas_indices = [base_meas_idx + i for i in range(len(block.data_qubits))]
            
            # Store for decoder metadata
            self._ancilla_logical_measurements[(round_idx, 'Z', stab.stab_idx)] = meas_indices
            
            # Create OuterSyndromeValue
            outer_syn = OuterSyndromeValue(
                outer_round=round_idx,
                stab_type='Z',
                stab_idx=stab.stab_idx,
                ancilla_block_id=anc_block_id,
                ancilla_ec_instance_id=trailing_ec_ids.get(anc_block_id, -1),
                transversal_meas_indices=meas_indices,
            )
            mcm.add_outer_syndrome(outer_syn)
            
            results.append(OuterMeasResult(
                stab_info=stab,
                ancilla_block_id=anc_block_id,
                measurement_indices=meas_indices,
            ))
        
        circuit.append("TICK")
        return results
    
    def _compute_parallel_cnot_schedule(
        self,
        x_stabs: List[OuterStabInfo],
        z_stabs: List[OuterStabInfo],
        x_ancilla_map: Dict[int, int],
        z_ancilla_map: Dict[int, int],
    ) -> List[List[Tuple[int, int, str]]]:
        """
        Compute a parallel CNOT schedule for all stabilizers.
        
        For Steane code with weight-4 stabilizers, we need 4 CNOT layers.
        Each layer contains non-conflicting block pairs.
        
        CORRECT Steane-style syndrome extraction (BOTH use data→ancilla):
        
        X stabilizers (detect Z errors):
        - Ancilla prepared in |0⟩ (logical zero)
        - CNOT(data → ancilla) for each data block
        - Measure ancilla in Z basis (direct MR)
        - Z errors on data propagate to ancilla via CNOT control XOR
        - NO backaction on data (data is control qubit, never modified)
        
        Z stabilizers (detect X errors):
        - Ancilla prepared in |+⟩ (logical plus)
        - CNOT(data → ancilla) for each data block
        - Measure ancilla in X basis (H then MR)
        - X errors on data propagate to ancilla via CNOT phase kickback
        - NO backaction on data (data is control qubit, never modified)
        
        Returns list of layers, where each layer is a list of
        (ctrl_block, targ_block, stab_type) tuples.
        """
        # Find maximum stabilizer weight
        max_weight = 0
        for stab in x_stabs + z_stabs:
            max_weight = max(max_weight, stab.weight)
        
        schedule: List[List[Tuple[int, int, str]]] = []
        
        for layer_idx in range(max_weight):
            layer: List[Tuple[int, int, str]] = []
            
            # X stabilizers: CNOT(data → ancilla) - copies Z values to ancilla
            # Data is control, ancilla is target - NO backaction on data
            for stab in x_stabs:
                if layer_idx < len(stab.support):
                    anc_block = x_ancilla_map.get(stab.stab_idx)
                    data_block = stab.support[layer_idx]
                    if anc_block is not None:
                        layer.append((data_block, anc_block, 'X'))  # data→ancilla
            
            # Z stabilizers: CNOT(data → ancilla) - copies Z values to ancilla
            # Data is control, ancilla is target - NO backaction
            for stab in z_stabs:
                if layer_idx < len(stab.support):
                    anc_block = z_ancilla_map.get(stab.stab_idx)
                    data_block = stab.support[layer_idx]
                    if anc_block is not None:
                        layer.append((data_block, anc_block, 'Z'))  # data→ancilla
            
            if layer:
                schedule.append(layer)
        
        return schedule

    # =========================================================================
    # exRec-Based FT Protocol (Core Implementation)
    # =========================================================================
    
    def _emit_stabilizer_set_exrec(
        self,
        circuit: stim.Circuit,
        stab_set: OuterStabSet,
    ) -> List[OuterMeasResult]:
        """
        Emit measurement of a stabilizer set using exRec structure.
        
        Per Aliferis-Gottesman-Preskill (AGP), the extended rectangle (exRec)
        ensures fault-tolerance by sandwiching each logical gate with error
        correction. The structure is:
        
            [Prepare] → [Leading EC] → [Gate₁] → [EC] → [Gate₂] → [EC] → ... 
                      → [Gate_k] → [Trailing EC] → [Measure]
        
        This ensures a single fault anywhere produces at most one error at the
        output boundary, which the trailing EC can correct (for distance-3 codes).
        
        """
        results = []
        
        x_ancillas = stab_set.all_x_ancillas
        all_ancillas = stab_set.all_ancillas
        
        if not all_ancillas:
            return results
        
        # Track participating blocks for this set
        participating_blocks: Set[int] = set()
        for stab in stab_set.x_stabilizers:
            participating_blocks.update(stab.support)
        for stab in stab_set.z_stabilizers:
            participating_blocks.update(stab.support)
        participating_blocks.update(all_ancillas)
        
        # =====================================================================
        # PHASE A: FT STATE PREPARATION (Steane-style encoded ancilla)
        # =====================================================================
        
        # A1. Reset ALL ancilla blocks (gives |0̄⟩ for CSS codes)
        self._block_manager.emit_reset_blocks(circuit, list(all_ancillas))
        circuit.append("TICK")
        
        # A2. Prepare basis states  
        # X stabilizers need |+̄⟩, Z stabilizers use |0̄⟩ (from reset)
        for anc_id in x_ancillas:
            self._block_manager.emit_prepare_logical_plus(circuit, anc_id)
        circuit.append("TICK")
        
        # =====================================================================
        # PHASE B: LEADING INNER EC (exRec start)
        # =====================================================================
        
        self._emit_inner_ec_selected_blocks(
            circuit, 
            participating_blocks,
            phase_name="leading_ec",
            ancilla_blocks=all_ancillas,
            n_rounds=self._inner_rounds,
        )
        
        # =====================================================================
        # PHASE C: GATE LAYERS WITH INTERLEAVED EC (exRec core)
        # Per AGP Section 4: EC after each gate catches propagated errors
        # =====================================================================
        
        gate_layers = self._get_gate_layers(stab_set)
        
        for layer_idx, layer in enumerate(gate_layers):
            # C0. PRE-CNOT MEASUREMENT (baseline for tight syndrome comparison)
            # This enables: syndrome_change = pre_cnot XOR post_cnot
            self._emit_pre_cnot_measurement(circuit, stab_set, layer_idx)
            
            # C1. Apply logical CNOTs for this layer via dispatcher
            # Also record entanglement events for decoder correction
            for stab_type, stab_idx, data_block_id, anc_block_id in layer:
                self._emit_logical_cnot_for_outer_stab(
                    circuit, stab_type, data_block_id, anc_block_id
                )
                
            
            circuit.append("TICK")
            
            # C2. POST-CNOT MEASUREMENT (MUST be before inner EC!)
            # Inner EC includes Z stabilizer measurements which kick X phase.
            # We need to measure X̄ IMMEDIATELY after CNOT, before any Z measurements.
            self._emit_post_cnot_measurement(circuit, stab_set, layer_idx)
        
            # C3. Inner EC after gate layer (exRec: catches propagated errors)
            last_ancilla_ec_ids = self._emit_inner_ec_selected_blocks(
                circuit,
                participating_blocks,  
                phase_name=f"post_gate_layer_{layer_idx}",
                ancilla_blocks=all_ancillas,
                n_rounds=self._inner_rounds,
            )
        
        # =====================================================================
        # PHASE D: TRANSVERSAL MEASUREMENT  
        # Per Steane: Measure encoded ancilla, decode to get logical value
        # We pass the last EC instance IDs to link outer syndromes to ancilla EC
        # =====================================================================
        
        results = self._measure_ancilla_blocks_transversal(
            circuit, stab_set, last_ancilla_ec_ids if gate_layers else {}
        )
        
        circuit.append("TICK")
        return results
    
    def _emit_logical_cnot_for_outer_stab(
        self,
        circuit: stim.Circuit,
        stab_type: str,
        data_block_id: int,
        anc_block_id: int,
    ) -> None:
        """
        Emit logical CNOT for outer stabilizer measurement via dispatcher.
        
        CRITICAL: Both X and Z stabilizers use CNOT(data → ancilla).
        This is the standard CSS syndrome extraction direction that:
        - For X stabilizers: extracts X parity via X_anc → X_anc ⊗ X_data
          while preserving Z_data (not corrupted by random ancilla Z)
        - For Z stabilizers: extracts Z parity via Z_anc → Z_anc ⊗ Z_data
          while preserving X_data
        
        Uses LogicalGateDispatcher for gate implementation flexibility:
        - Transversal CNOT (default for CSS codes like Steane)
        - Lattice surgery CNOT (for surface codes)
        - Teleportation-based CNOT (universal fallback)
    
        This separation is consistent with AGP's hierarchical FT structure:
          - Level 0 (Stim): Inner syndrome detection WITHIN each block
          - Level 1 (Decoder): Outer syndrome from decoded ancilla logicals
        """
        data_block = self._block_manager.get_block(data_block_id)
        anc_block = self._block_manager.get_block(anc_block_id)
        
        # Determine CNOT direction based on stabilizer type
        # CRITICAL FIX: Both X and Z stabilizers use CNOT(data→ancilla)
        # 
        # For X stabilizer (ancilla in |+⟩_L, measure X):
        #   - CNOT(data→anc): X_anc → X_anc ⊗ X_data (syndrome extraction)
        #   - Z_data → Z_data (PRESERVED - no random Z from ancilla)
        #   - Previously used CNOT(anc→data) which corrupted Z_data
        #
        # For Z stabilizer (ancilla in |0⟩_L, measure Z):
        #   - CNOT(data→anc): Z_anc → Z_anc ⊗ Z_data (syndrome extraction)
        #   - X_data → X_data (preserved)
        #
        # Both cases: data controls, ancilla is target
        control_alloc = data_block.allocation
        target_alloc = anc_block.allocation
        
        # Emit logical CNOT via dispatcher (supports transversal/surgery/teleportation)
        self._gate_dispatcher.emit_logical_cnot(
            circuit,
            control_alloc,
            target_alloc,
            method=self._gate_method,
        )
        

        # =====================================================================
        # NOTE ON ENTANGLEMENT (For reference)
        # =====================================================================
        # With the corrected CNOT direction (data→ancilla for both X and Z):
        #   - X stab: CNOT(data → x_anc) causes X_anc to get X_data (syndrome)
        #     Z_data is PRESERVED (no entanglement with ancilla Z)
        #   - Z stab: CNOT(data → z_anc) causes Z_anc to get Z_data (syndrome)
        #     X_data is preserved (no entanglement with ancilla X)
        #
        # This means the logical observable is now deterministic!
        # The decoder no longer needs to track entanglement corrections.
        # =====================================================================
    
    def _emit_inner_ec_selected_blocks(
        self,
        circuit: stim.Circuit,
        participating_blocks: Set[int],
        phase_name: str = "",
        ancilla_blocks: Optional[Set[int]] = None,
        n_rounds: int = 1,
    ) -> Dict[int, int]:
        """
        Emit inner EC segment on selected participating blocks.
        
        Per AGP, inner EC is compulsory at exRec boundaries.
        
        This method uses the new emit_inner_ec_segment() which properly
        aggregates n_rounds with TICKs between them for temporal error detection.
 
        Parameters
        ----------
        participating_blocks : Set[int]
            Block IDs to run inner EC on.
        phase_name : str
            Debug label for this EC phase.
        ancilla_blocks : Set[int], optional
            Block IDs that are ancilla blocks.
        n_rounds : int
            Number of syndrome extraction rounds (should be >= inner code distance).
            
        Returns
        -------
        Dict[int, int]
            Maps block_id -> InnerECInstance ID created for that block
        """
        if ancilla_blocks is None:
            ancilla_blocks = set()
        
        ec_instance_ids: Dict[int, int] = {}
        
        # Separate data and ancilla blocks
        data_blocks = participating_blocks - ancilla_blocks
        ancilla_in_participating = ancilla_blocks & participating_blocks
        
        # Emit EC segment for data blocks
        if data_blocks:
            data_ids = self._block_manager.emit_inner_ec_segment(
                circuit,
                block_ids=sorted(data_blocks),
                n_rounds=n_rounds,
                context='intra_gadget',
                outer_round=self._current_round,
            )
            ec_instance_ids.update(data_ids)
        
        # Emit EC segment for ancilla blocks
        if ancilla_in_participating:
            ancilla_ids = self._block_manager.emit_inner_ec_segment(
                circuit,
                block_ids=sorted(ancilla_in_participating),
                n_rounds=n_rounds,
                context='intra_gadget',
                outer_round=self._current_round,
            )
            ec_instance_ids.update(ancilla_ids)
        
        return ec_instance_ids
    
    def _get_gate_layers(
        self, 
        stab_set: OuterStabSet
    ) -> List[List[Tuple[str, int, int, int]]]:
        """
        Get gate layers for a stabilizer set.
        
        For weight-w stabilizers, we need w CNOT layers. Each layer
        connects one data block to its ancilla.
        
        Returns list of layers, each containing tuples of:
        (stab_type, stab_idx, data_block_id, ancilla_block_id)
        """
        max_weight = 0
        for stab in stab_set.x_stabilizers + stab_set.z_stabilizers:
            max_weight = max(max_weight, stab.weight)
        
        layers = []
        for layer_idx in range(max_weight):
            layer = []
            
            for stab in stab_set.x_stabilizers:
                if layer_idx < len(stab.support):
                    data_block = stab.support[layer_idx]
                    anc_block = stab_set.x_ancilla_assignments.get(stab.stab_idx)
                    if anc_block is not None:
                        layer.append(("X", stab.stab_idx, data_block, anc_block))
            
            for stab in stab_set.z_stabilizers:
                if layer_idx < len(stab.support):
                    data_block = stab.support[layer_idx]
                    anc_block = stab_set.z_ancilla_assignments.get(stab.stab_idx)
                    if anc_block is not None:
                        layer.append(("Z", stab.stab_idx, data_block, anc_block))
            
            if layer:
                layers.append(layer)
        
        return layers
    
    def _measure_ancilla_blocks_transversal(
        self,
        circuit: stim.Circuit,
        stab_set: OuterStabSet,
        last_ancilla_ec_ids: Optional[Dict[int, int]] = None,
    ) -> List[OuterMeasResult]:
        """
        Measure ancilla blocks transversally to extract outer syndrome.
        
        Per Steane's encoded ancilla method:
        1. Apply H for X stabilizers (Z→X basis change)
        2. Measure all n physical qubits of ancilla block
        3. Record measurement indices for decoder
        4. Decoder will decode inner code → logical measurement value
        
        The outer syndrome bit is the DECODED logical value, not raw measurements.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        stab_set : OuterStabSet
            Stabilizer set being measured.
        last_ancilla_ec_ids : Dict[int, int], optional
            Mapping from ancilla block_id to the InnerECInstance ID 
            that protects this measurement.
        """
        results = []
        if last_ancilla_ec_ids is None:
            last_ancilla_ec_ids = {}
        
        # Process X stabilizers (measure in X basis via H then Z measurement)
        for stab in stab_set.x_stabilizers:
            anc_id = stab_set.x_ancilla_assignments.get(stab.stab_idx)
            if anc_id is None:
                continue
            
            anc_block = self._block_manager.get_block(anc_id)
            
            # H for basis change (Z→X measurement)
            circuit.append("H", anc_block.data_qubits)
        
        circuit.append("TICK")
        
        # Measure all ancilla blocks
        # IMPORTANT: Use separate lookups for X and Z ancilla assignments
        # because stab_idx can be the same for X and Z stabilizers (e.g., both
        # X stab 0 and Z stab 0 have stab_idx=0). Merging dicts would cause
        # Z to overwrite X, leaving X ancilla blocks unmeasured!
        all_stabs = list(stab_set.x_stabilizers) + list(stab_set.z_stabilizers)
        
        for stab in all_stabs:
            # Look up ancilla assignment based on stabilizer TYPE to avoid key collision
            if stab.stab_type == "X":
                anc_id = stab_set.x_ancilla_assignments.get(stab.stab_idx)
            else:
                anc_id = stab_set.z_ancilla_assignments.get(stab.stab_idx)
            if anc_id is None:
                continue
            
            anc_block = self._block_manager.get_block(anc_id)
            
            # Record measurement indices BEFORE emitting measurement
            base_meas_idx = self._block_manager._measurement_index
            
            # Transversal measurement (MR resets for reuse)
            circuit.append("MR", anc_block.data_qubits)
            self._block_manager._measurement_index += len(anc_block.data_qubits)
            
            # All physical measurements for this ancilla block
            all_meas_indices = list(range(
                base_meas_idx, 
                base_meas_idx + len(anc_block.data_qubits)
            ))
            
            # Track for decoder metadata (legacy format)
            meas_key = (self._current_round, stab.stab_type, stab.stab_idx)
            self._ancilla_logical_measurements[meas_key] = all_meas_indices
            
            # NEW: Create OuterSyndromeValue for measurement-centric model
            # Get the EC instance ID that protects this ancilla measurement
            ancilla_ec_instance_id = last_ancilla_ec_ids.get(anc_id, -1)
            
            outer_syndrome = OuterSyndromeValue(
                outer_round=self._current_round,
                stab_type=stab.stab_type,
                stab_idx=stab.stab_idx,
                ancilla_block_id=anc_id,
                transversal_meas_indices=all_meas_indices,
                ancilla_ec_instance_id=ancilla_ec_instance_id,
            )
            
            # Add to block manager's measurement-centric metadata
            metadata = self._block_manager.get_measurement_centric_metadata()
            metadata.add_outer_syndrome(outer_syndrome)
   
            results.append(OuterMeasResult(
                stab_info=stab,
                ancilla_block_id=anc_id,
                measurement_indices=all_meas_indices,
            ))
        
        return results
    
    def _emit_pre_cnot_measurement(
        self, 
        circuit: stim.Circuit, 
        stab_set: OuterStabSet, 
        layer_idx: int
    ) -> None:
        """
        Measure ancilla X̄ BEFORE outer CNOT layer for tight syndrome comparison.
        
        This gives us a baseline for comparing with post-CNOT measurement.
        The XOR (pre ⊕ post) should be deterministic (0 if no error propagated).
        
        Key insight: We need non-destructive measurement to preserve ancilla state.
        Use MX (measure X basis) which is non-destructive for |+̄⟩ state.
        """
        
        # Measure X ancillas (these are in |+̄⟩ state, measure X̄)
        for stab in stab_set.x_stabilizers:
            anc_id = stab_set.x_ancilla_assignments.get(stab.stab_idx)
            if anc_id is None:
                continue
            
            anc_block = self._block_manager.get_block(anc_id)
            
            # Record measurement indices BEFORE emitting
            base_meas_idx = self._block_manager._measurement_index
            
            # Non-destructive X measurement (preserves |+⟩ eigenstates)
            circuit.append("MX", anc_block.data_qubits)
            self._block_manager._measurement_index += len(anc_block.data_qubits)
            
            all_meas_indices = list(range(
                base_meas_idx,
                base_meas_idx + len(anc_block.data_qubits)
            ))
            
            # Store with key: (round, stab_type, stab_idx, layer_idx)
            meas_key = (self._current_round, "X", stab.stab_idx, layer_idx)
            self._pre_cnot_measurements[meas_key] = all_meas_indices
        
        circuit.append("TICK")
    
    def _emit_post_cnot_measurement(
        self,
        circuit: stim.Circuit,
        stab_set: OuterStabSet,
        layer_idx: int
    ) -> None:
        """
        Measure ancilla X̄ AFTER outer CNOT layer for tight syndrome comparison.
        
        Compare with pre-CNOT measurement to detect errors propagated during CNOT.
        """
        
        # Measure X ancillas
        for stab in stab_set.x_stabilizers:
            anc_id = stab_set.x_ancilla_assignments.get(stab.stab_idx)
            if anc_id is None:
                continue
            
            anc_block = self._block_manager.get_block(anc_id)
            
            base_meas_idx = self._block_manager._measurement_index
            
            # Non-destructive X measurement
            circuit.append("MX", anc_block.data_qubits)
            self._block_manager._measurement_index += len(anc_block.data_qubits)
            
            all_meas_indices = list(range(
                base_meas_idx,
                base_meas_idx + len(anc_block.data_qubits)
            ))
            
            meas_key = (self._current_round, "X", stab.stab_idx, layer_idx)
            self._post_cnot_measurements[meas_key] = all_meas_indices
        
        circuit.append("TICK")
    
    # =========================================================================
    # Metadata for Decoder
    # =========================================================================
    
    @property
    def n_x_stabilizers(self) -> int:
        return len(self._x_stabilizers)
    
    @property
    def n_z_stabilizers(self) -> int:
        return len(self._z_stabilizers)
    
    @property
    def inner_rounds(self) -> int:
        """Number of inner syndrome extraction rounds per EC segment."""
        return self._inner_rounds
    
    @property
    def ancilla_logical_measurements(self) -> Dict[Tuple[int, str, int], List[int]]:
        """
        Get ancilla logical measurements: (round, type, idx) -> [meas_indices].
        
        The decoder uses these to:
        1. Extract inner syndrome for ancilla block
        2. Decode inner code → predicted logical value
        3. Compute outer syndrome from decoded values
        """
        return dict(self._ancilla_logical_measurements)
    
    @property
    def pre_cnot_measurements(self) -> Dict[Tuple[int, str, int, int], List[int]]:
        """
        Get pre-CNOT ancilla measurements: (round, type, stab_idx, layer_idx) -> [meas_indices].
        
        For tight syndrome comparison: pre ⊕ post should be 0 if no error.
        """
        return dict(self._pre_cnot_measurements)
    
    @property
    def post_cnot_measurements(self) -> Dict[Tuple[int, str, int, int], List[int]]:
        """
        Get post-CNOT ancilla measurements: (round, type, stab_idx, layer_idx) -> [meas_indices].
        
        For tight syndrome comparison: pre ⊕ post should be 0 if no error.
        """
        return dict(self._post_cnot_measurements)
    
    def get_outer_stab_support(self) -> Dict[Tuple[str, int], List[int]]:
        """Get support of each outer stabilizer: (type, idx) -> [data_blocks]."""
        support = {}
        for stab in self._x_stabilizers:
            support[("X", stab.stab_idx)] = stab.support
        for stab in self._z_stabilizers:
            support[("Z", stab.stab_idx)] = stab.support
        return support
    
    def get_stab_to_ancilla_block(self) -> Dict[Tuple[str, int], int]:
        """Get mapping from (stab_type, stab_idx) to ancilla block ID."""
        mapping = {}
        
        for stab_set in self._mixed_sets:
            for stab_idx, anc_block in stab_set.x_ancilla_assignments.items():
                mapping[("X", stab_idx)] = anc_block
            for stab_idx, anc_block in stab_set.z_ancilla_assignments.items():
                mapping[("Z", stab_idx)] = anc_block
        
        for stab_set in self._x_sets:
            for stab_idx, anc_block in stab_set.x_ancilla_assignments.items():
                if ("X", stab_idx) not in mapping:
                    mapping[("X", stab_idx)] = anc_block
                    
        for stab_set in self._z_sets:
            for stab_idx, anc_block in stab_set.z_ancilla_assignments.items():
                if ("Z", stab_idx) not in mapping:
                    mapping[("Z", stab_idx)] = anc_block
        
        return mapping

    def get_schedule_summary(self) -> str:
        """Get human-readable schedule summary."""
        lines = [
            f"OuterStabilizerEngineV3 (exRec-Based FT Protocol)",
            f"  Literature: AGP (quant-ph/0504218), Steane EC",
            f"  Outer code: {self._outer_code}",
            f"  X stabilizers: {self.n_x_stabilizers} in {len(self._x_sets)} sets",
            f"  Z stabilizers: {self.n_z_stabilizers} in {len(self._z_sets)} sets",
            f"  Mixed sets: {len(self._mixed_sets)}",
            f"  Can interleave: {self.can_interleave()}",
            f"  Gate method: {self._gate_method}",
            f"  Inner EC: COMPULSORY (per AGP exRec structure)",
        ]
        return "\n".join(lines)
