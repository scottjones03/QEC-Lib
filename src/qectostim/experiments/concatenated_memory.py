# src/qectostim/experiments/concatenated_memory.py
"""
Memory experiment for concatenated codes with block-grouped detectors.

This module provides ConcatenatedMemoryExperiment, which generates circuits
where detectors are organized by inner code blocks, enabling hierarchical
decoding with ConcatenatedDecoder.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import stim

if TYPE_CHECKING:
    from qectostim.codes.composite.concatenated import ConcatenatedCode

from qectostim.experiments.memory import (
    CSSMemoryExperiment,
    get_logical_ops,
    ops_valid,
    ops_len,
    pauli_at,
)
from qectostim.utils.scheduling_core import graph_coloring_cnots




def apply_stabilizer_cnots_with_ticks(
    circuit: stim.Circuit,
    parity_check: np.ndarray,
    data_qubits: List[int],
    ancilla_qubits: List[int],
    is_x_type: bool = False,
) -> None:
    """
    Apply stabilizer CNOT gates for CSS codes with TICK scheduling.
    
    This function applies CNOTs for either X-type or Z-type stabilizers,
    scheduling them into conflict-free layers using graph coloring.
    
    CNOT direction convention for CSS stabilizer measurement:
    - X-type stabilizers: CNOT(data, ancilla) - data controls
    - Z-type stabilizers: CNOT(data, ancilla) - data controls (SAME direction!)
    
    Why the same direction for both?
    - X-type: H-CNOT(d→a)-H measures ⊗σ^x by propagating X from data to ancilla
    - Z-type: CNOT(d→a) measures ⊗σ^z by propagating Z from data to ancilla
    
    The key insight is that CNOT propagates Z on control and X on target:
    - CNOT|Z⊗I⟩ = |Z⊗Z⟩ (Z copies from control to target)
    - CNOT|I⊗X⟩ = |X⊗X⟩ (X copies from target to control)
    
    Parameters
    ----------
    circuit : stim.Circuit
        The Stim circuit to append gates to.
    parity_check : np.ndarray
        Parity check matrix (n_stabs x n_data) - either Hx or Hz.
    data_qubits : List[int]
        Global indices of data qubits.
    ancilla_qubits : List[int]
        Global indices of ancilla qubits.
    is_x_type : bool
        True for X-type stabilizers (H applied before/after), False for Z-type.
        Note: Both use the same CNOT direction (data→ancilla).
    """
    if parity_check is None or parity_check.size == 0:
        return
    
    n_stabs = parity_check.shape[0]
    n = len(data_qubits)
    
    # Collect all CNOT pairs - both X and Z use data→ancilla direction
    cnot_pairs: List[Tuple[int, int]] = []
    
    for s_idx in range(min(n_stabs, len(ancilla_qubits))):
        anc = ancilla_qubits[s_idx]
        row = parity_check[s_idx]
        
        for q in range(min(n, len(row))):
            if row[q]:
                dq = data_qubits[q]
                # Both X-type and Z-type: CNOT from data to ancilla
                cnot_pairs.append((dq, anc))
    
    if not cnot_pairs:
        return
    
    # Schedule into conflict-free layers
    layers = graph_coloring_cnots(cnot_pairs)
    
    # Apply each layer with TICK separation
    # NOTE: We add TICK after every layer (not just between) for proper DEM error granularity
    for layer_idx, layer_cnots in enumerate(layers):
        if layer_idx > 0:
            circuit.append("TICK")
        
        for ctrl, targ in layer_cnots:
            circuit.append("CX", [ctrl, targ])
    
    # Add final TICK after all CNOT layers
    if layers:
        circuit.append("TICK")

class ConcatenatedMemoryExperiment(CSSMemoryExperiment):
    """
    Memory experiment for concatenated CSS codes with block-grouped detectors.
    
    This class generates circuits where detectors are organized by inner code blocks,
    enabling hierarchical decoding with ConcatenatedDecoder. The detector layout is:
    
        [Block 0 detectors] [Block 1 detectors] ... [Block N-1 detectors] [Outer detectors]
    
    Each inner block's detectors are grouped together, matching the structure expected
    by ConcatenatedDecoder's `decode_batch()` method.
    
    Supports multi-level (deeply nested) concatenation - if the inner code is itself
    a concatenated code, the detector grouping is applied recursively.
    
    Parameters
    ----------
    code : ConcatenatedCode
        A concatenated CSS code (can be multi-level nested).
    rounds : int
        Number of syndrome measurement rounds.
    noise_model : optional
        Noise model to apply.
    basis : str
        Measurement basis ('X' or 'Z').
    metadata : optional
        Additional metadata.
    """
    
    def __init__(
        self,
        code,  # ConcatenatedCode or ConcatenatedTopologicalCSSCode
        rounds: int,
        noise_model: Dict[str, Any] | None = None,
        basis: str = "Z",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Import here to avoid circular imports
        from qectostim.codes.composite.concatenated import ConcatenatedCode
        
        if not isinstance(code, ConcatenatedCode):
            raise TypeError(
                f"ConcatenatedMemoryExperiment requires a ConcatenatedCode, "
                f"got {type(code).__name__}"
            )
        
        super().__init__(
            code=code,
            rounds=rounds,
            noise_model=noise_model,
            basis=basis,
            metadata=metadata,
        )
        
        # Store concatenation structure
        # ConcatenatedCSSCode uses .outer and .inner, not ._inner_code/_outer_code
        if hasattr(code, '_inner_code') and hasattr(code, '_outer_code'):
            self._inner_code = code._inner_code
            self._outer_code = code._outer_code
        elif hasattr(code, 'inner') and hasattr(code, 'outer'):
            self._inner_code = code.inner
            self._outer_code = code.outer
        else:
            raise ValueError("Code must have inner/outer code attributes")
        
        self._n_outer = code._n_outer  # Number of inner code blocks
        
        # Inner code stabilizer counts
        self._n_inner_x = len(self._inner_code.hx)
        self._n_inner_z = len(self._inner_code.hz)
        self._n_inner_stabs = self._n_inner_x + self._n_inner_z
        
        # Outer code stabilizer counts (lifted)
        self._n_outer_x = len(self._outer_code.hx)
        self._n_outer_z = len(self._outer_code.hz)
        self._n_outer_stabs = self._n_outer_x + self._n_outer_z
        
        # Total counts in concatenated code
        self._total_x_stabs = self._n_outer * self._n_inner_x + self._n_outer_x
        self._total_z_stabs = self._n_outer * self._n_inner_z + self._n_outer_z
        
        # Check if inner code is also concatenated (for multi-level support)
        self._inner_is_concatenated = isinstance(self._inner_code, ConcatenatedCode)
        
        # Compute hierarchical structure
        self._hierarchy = self._compute_hierarchy()
    
    def _compute_hierarchy(self) -> Dict[str, Any]:
        """
        Compute the hierarchical structure of the concatenated code.
        
        For multi-level concatenation, this recursively computes the
        detector structure at each level.
        """
        from qectostim.codes.composite.concatenated import ConcatenatedCode
        
        def _get_inner(code):
            """Helper to get inner code from either attribute name."""
            if hasattr(code, '_inner_code'):
                return code._inner_code
            return code.inner
        
        def _get_outer(code):
            """Helper to get outer code from either attribute name."""
            if hasattr(code, '_outer_code'):
                return code._outer_code
            return code.outer
        
        def get_level_info(code, level=0):
            """Recursively extract level information."""
            if not isinstance(code, ConcatenatedCode):
                # Base case: not concatenated
                return {
                    'level': level,
                    'is_concatenated': False,
                    'n': code.n,
                    'k': code.k,
                    'n_x_stabs': len(code.hx),
                    'n_z_stabs': len(code.hz),
                    'inner': None,
                    'outer': None,
                    'n_blocks': 1,
                }
            
            # Recursive case: concatenated code
            inner_code = _get_inner(code)
            outer_code = _get_outer(code)
            inner_info = get_level_info(inner_code, level + 1)
            outer_info = get_level_info(outer_code, level + 1)
            
            return {
                'level': level,
                'is_concatenated': True,
                'n': code.n,
                'k': code.k,
                'n_x_stabs': len(code.hx),
                'n_z_stabs': len(code.hz),
                'inner': inner_info,
                'outer': outer_info,
                'n_blocks': code._n_outer,
                'inner_code': inner_code,
                'outer_code': outer_code,
            }
        
        return get_level_info(self.code)
    
    def _get_concatenation_depth(self) -> int:
        """Get the depth of concatenation nesting."""
        depth = 1
        info = self._hierarchy
        while info.get('inner') and info['inner'].get('is_concatenated'):
            depth += 1
            info = info['inner']
        return depth
    
    def to_stim(self) -> stim.Circuit:
        """
        Build a memory experiment with block-grouped detectors for hierarchical decoding.
        
        IMPORTANT: This method emits detectors in BLOCK-CONTIGUOUS order, meaning
        all detectors for block 0 come first (across all rounds), then all for block 1,
        etc., and finally all outer detectors. This is critical for ConcatenatedDecoder
        which uses simple slicing to extract per-block syndromes.
        
        Detector organization (for Z-basis memory):
            [Block 0: all rounds time-like + space-like]
            [Block 1: all rounds time-like + space-like]
            ...
            [Block N-1: all rounds time-like + space-like]
            [Outer: all rounds time-like + space-like]
        
        Within each block, detectors are ordered by round, then by stabilizer type
        (X before Z within each round).
        """
        code = self.code
        n = code.n
        hx = code.hx
        hz = code.hz
        basis = self.basis.upper()
        
        n_x = hx.shape[0]
        n_z = hz.shape[0]
        
        data_qubits = list(range(n))
        anc_x = list(range(n, n + n_x))
        anc_z = list(range(n + n_x, n + n_x + n_z))
        
        c = stim.Circuit()
        
        # Initial preparation
        total_qubits = n + n_x + n_z
        if total_qubits:
            c.append("R", range(total_qubits))
        
        if basis == "X" and n > 0:
            c.append("H", data_qubits)
        
        c.append("TICK")
        
        # =====================================================================
        # PHASE 1: Syndrome measurement rounds (NO detector emission yet)
        # =====================================================================
        # We'll collect all measurement indices and emit detectors at the end
        # in block-contiguous order.
        
        m_index = 0
        
        # Track measurement indices for each stabilizer per round
        # inner_x_meas[block_id][local_idx][round] = measurement index
        inner_x_meas: List[List[List[int]]] = [
            [[] for _ in range(self._n_inner_x)] for _ in range(self._n_outer)
        ]
        inner_z_meas: List[List[List[int]]] = [
            [[] for _ in range(self._n_inner_z)] for _ in range(self._n_outer)
        ]
        outer_x_meas: List[List[int]] = [[] for _ in range(self._n_outer_x)]
        outer_z_meas: List[List[int]] = [[] for _ in range(self._n_outer_z)]
        
        # Syndrome rounds
        for r in range(self.rounds):
            # Apply H to X ancillas at start of round
            if n_x:
                for a in anc_x:
                    c.append("H", [a])
            
            # Apply stabilizer CNOTs
            if n_x:
                apply_stabilizer_cnots_with_ticks(
                    c, hx, list(range(n)), anc_x, is_x_type=True
                )
                for a in anc_x:
                    c.append("H", [a])
            
            c.append("TICK")
            
            if n_z:
                apply_stabilizer_cnots_with_ticks(
                    c, hz, list(range(n)), anc_z, is_x_type=False
                )
            
            # Measure all ancillas
            x_meas_start = m_index
            if n_x:
                c.append("MR", anc_x)
                m_index += n_x
            
            z_meas_start = m_index
            if n_z:
                c.append("MR", anc_z)
                m_index += n_z
            
            # Record measurement indices (NO detector emission yet)
            for block_id in range(self._n_outer):
                for local_idx in range(self._n_inner_x):
                    global_x_idx = block_id * self._n_inner_x + local_idx
                    inner_x_meas[block_id][local_idx].append(x_meas_start + global_x_idx)
                
                for local_idx in range(self._n_inner_z):
                    global_z_idx = block_id * self._n_inner_z + local_idx
                    inner_z_meas[block_id][local_idx].append(z_meas_start + global_z_idx)
            
            for local_idx in range(self._n_outer_x):
                global_x_idx = self._n_outer * self._n_inner_x + local_idx
                outer_x_meas[local_idx].append(x_meas_start + global_x_idx)
            
            for local_idx in range(self._n_outer_z):
                global_z_idx = self._n_outer * self._n_inner_z + local_idx
                outer_z_meas[local_idx].append(z_meas_start + global_z_idx)
        
        # Final data measurement
        data_meas = {}
        if n:
            if basis == "X":
                c.append("H", data_qubits)
            c.append("M", data_qubits)
            
            first_data_idx = m_index
            data_meas = {q: first_data_idx + i for i, q in enumerate(data_qubits)}
            m_index += n
        
        # =====================================================================
        # PHASE 2: Emit detectors in BLOCK-CONTIGUOUS order
        # =====================================================================
        # Now emit all detectors in block-grouped order. This ensures detector
        # indices are contiguous per block, enabling simple slicing in decoder.
        
        def add_detector_at_end(rec_indices: list[int], coord: tuple = (0.0, 0.0, 0.0)) -> None:
            """Emit a DETECTOR with rec lookbacks from current m_index."""
            if not rec_indices:
                return
            lookbacks = [idx - m_index for idx in rec_indices]
            c.append(
                "DETECTOR",
                [stim.target_rec(lb) for lb in lookbacks],
                list(coord),
            )
        
        # Inner blocks first (block-contiguous order)
        for block_id in range(self._n_outer):
            # X stabilizer detectors for this block (all rounds)
            for local_idx in range(self._n_inner_x):
                meas_list = inner_x_meas[block_id][local_idx]
                
                for r, cur_meas in enumerate(meas_list):
                    if r == 0:
                        # First round: only create detector if X-basis
                        if basis == "X":
                            add_detector_at_end([cur_meas], (float(block_id), float(local_idx), float(r)))
                    else:
                        prev_meas = meas_list[r - 1]
                        add_detector_at_end([prev_meas, cur_meas], (float(block_id), float(local_idx), float(r)))
            
            # Z stabilizer detectors for this block (all rounds + space-like)
            for local_idx in range(self._n_inner_z):
                meas_list = inner_z_meas[block_id][local_idx]
                
                for r, cur_meas in enumerate(meas_list):
                    if r == 0:
                        # First round: only create detector if Z-basis
                        if basis == "Z":
                            add_detector_at_end([cur_meas], (float(block_id), float(local_idx), float(r)))
                    else:
                        prev_meas = meas_list[r - 1]
                        add_detector_at_end([prev_meas, cur_meas], (float(block_id), float(local_idx), float(r)))
            
            # Space-like detectors for this block (only matching basis)
            if basis == "Z" and data_meas:
                # Z space-like detectors
                for local_idx in range(self._n_inner_z):
                    global_idx = block_id * self._n_inner_z + local_idx
                    row = hz[global_idx]
                    
                    data_idxs = [data_meas[q] for q in np.where(row == 1)[0] if q in data_meas]
                    recs = list(data_idxs)
                    
                    if inner_z_meas[block_id][local_idx]:
                        recs.append(inner_z_meas[block_id][local_idx][-1])  # Last round
                    
                    if recs:
                        add_detector_at_end(recs, (float(block_id), float(local_idx), float(self.rounds)))
            
            elif basis == "X" and data_meas:
                # X space-like detectors
                for local_idx in range(self._n_inner_x):
                    global_idx = block_id * self._n_inner_x + local_idx
                    row = hx[global_idx]
                    
                    data_idxs = [data_meas[q] for q in np.where(row == 1)[0] if q in data_meas]
                    recs = list(data_idxs)
                    
                    if inner_x_meas[block_id][local_idx]:
                        recs.append(inner_x_meas[block_id][local_idx][-1])  # Last round
                    
                    if recs:
                        add_detector_at_end(recs, (float(block_id), float(local_idx), float(self.rounds)))
        
        # Outer stabilizer detectors (all rounds + space-like)
        # X stabilizers
        for local_idx in range(self._n_outer_x):
            meas_list = outer_x_meas[local_idx]
            
            for r, cur_meas in enumerate(meas_list):
                if r == 0:
                    if basis == "X":
                        add_detector_at_end([cur_meas], (float(self._n_outer), float(local_idx), float(r)))
                else:
                    prev_meas = meas_list[r - 1]
                    add_detector_at_end([prev_meas, cur_meas], (float(self._n_outer), float(local_idx), float(r)))
        
        # Z stabilizers
        for local_idx in range(self._n_outer_z):
            meas_list = outer_z_meas[local_idx]
            
            for r, cur_meas in enumerate(meas_list):
                if r == 0:
                    if basis == "Z":
                        add_detector_at_end([cur_meas], (float(self._n_outer), float(local_idx), float(r)))
                else:
                    prev_meas = meas_list[r - 1]
                    add_detector_at_end([prev_meas, cur_meas], (float(self._n_outer), float(local_idx), float(r)))
        
        # Outer space-like detectors
        if data_meas:
            if basis == "Z":
                # Outer Z space-like
                for local_idx in range(self._n_outer_z):
                    global_idx = self._n_outer * self._n_inner_z + local_idx
                    row = hz[global_idx]
                    
                    data_idxs = [data_meas[q] for q in np.where(row == 1)[0] if q in data_meas]
                    recs = list(data_idxs)
                    
                    if outer_z_meas[local_idx]:
                        recs.append(outer_z_meas[local_idx][-1])
                    
                    if recs:
                        add_detector_at_end(recs, (float(self._n_outer), float(local_idx), float(self.rounds)))
            
            else:  # X-basis
                # Outer X space-like
                for local_idx in range(self._n_outer_x):
                    global_idx = self._n_outer * self._n_inner_x + local_idx
                    row = hx[global_idx]
                    
                    data_idxs = [data_meas[q] for q in np.where(row == 1)[0] if q in data_meas]
                    recs = list(data_idxs)
                    
                    if outer_x_meas[local_idx]:
                        recs.append(outer_x_meas[local_idx][-1])
                    
                    if recs:
                        add_detector_at_end(recs, (float(self._n_outer), float(local_idx), float(self.rounds)))
        
        # Logical observable
        z_ops = get_logical_ops(code, 'z')
        x_ops = get_logical_ops(code, 'x')
        logical_support: list[int] = []
        
        if basis == "Z" and ops_valid(z_ops) and self.logical_qubit < ops_len(z_ops):
            L = z_ops[self.logical_qubit]
            logical_support = [q for q in range(n) if pauli_at(L, q) in ("Z", "Y")]
        elif basis == "X" and ops_valid(x_ops) and self.logical_qubit < ops_len(x_ops):
            L = x_ops[self.logical_qubit]
            logical_support = [q for q in range(n) if pauli_at(L, q) in ("X", "Y")]
        
        if not logical_support:
            logical_support = data_qubits
        
        obs_rec_indices = [data_meas[q] for q in logical_support if q in data_meas]
        if obs_rec_indices:
            lookbacks = [idx - m_index for idx in obs_rec_indices]
            c.append("OBSERVABLE_INCLUDE", [stim.target_rec(lb) for lb in lookbacks], 0)
        
        # =====================================================================
        # PHASE 3: Emit INNER LOGICAL OBSERVABLES for hierarchical decoding
        # =====================================================================
        # For each inner block, emit an observable that tracks whether an inner
        # logical error occurred. This enables the hierarchical decoder to:
        # 1. Decode each inner block's syndrome to get inner logical prediction
        # 2. Use inner logical predictions to build effective outer syndrome
        # 3. Decode outer code with corrected syndrome
        #
        # Observable 0: Main concatenated logical (already emitted above)
        # Observable 1: Inner block 0's logical error
        # Observable 2: Inner block 1's logical error
        # ... etc.
        self._emit_inner_logical_observables(c, data_meas, m_index, basis)
        
        return c
    
    def _emit_inner_logical_observables(
        self,
        circuit: stim.Circuit,
        data_meas: Dict[int, int],
        m_index: int,
        basis: str
    ) -> None:
        """
        Emit OBSERVABLE_INCLUDE for each inner block's logical operator.
        
        For hierarchical decoding, we need to track inner logical errors separately
        from the main concatenated logical. Each inner block gets its own observable.
        
        Supports multi-level concatenation: if the inner code is itself concatenated,
        recursively emit observables for nested inner blocks.
        
        Parameters
        ----------
        circuit : stim.Circuit
            The circuit being built.
        data_meas : dict
            Mapping from data qubit index to measurement index.
        m_index : int
            Current measurement index (for computing rec lookbacks).
        basis : str
            Measurement basis ('X' or 'Z').
        """
        from qectostim.codes.composite.concatenated import ConcatenatedCode
        
        # Check if inner code is itself concatenated (multi-level)
        inner_is_concatenated = isinstance(self._inner_code, ConcatenatedCode)
        
        # Get inner code's logical operator
        inner_z_ops = get_logical_ops(self._inner_code, 'z')
        inner_x_ops = get_logical_ops(self._inner_code, 'x')
        
        n_inner = self._inner_code.n
        
        # Determine which inner logical to use based on basis
        if basis == "Z" and ops_valid(inner_z_ops) and ops_len(inner_z_ops) > 0:
            inner_logical = inner_z_ops[0]  # First Z logical of inner code
            pauli_type = ("Z", "Y")
        elif basis == "X" and ops_valid(inner_x_ops) and ops_len(inner_x_ops) > 0:
            inner_logical = inner_x_ops[0]  # First X logical of inner code
            pauli_type = ("X", "Y")
        else:
            # Fallback: can't determine inner logical
            return
        
        # Get the support of the inner logical operator
        inner_logical_support = [
            q for q in range(n_inner) 
            if pauli_at(inner_logical, q) in pauli_type
        ]
        
        if not inner_logical_support:
            return
        
        # Observable counter starts at 1 (0 is main concatenated logical)
        obs_counter = 1
        
        # Emit an observable for each inner block
        for block_id in range(self._n_outer):
            block_offset = block_id * n_inner
            
            # Map inner logical support to global qubit indices for this block
            block_logical_qubits = [q + block_offset for q in inner_logical_support]
            
            # Get measurement indices
            obs_rec_indices = [
                data_meas[q] for q in block_logical_qubits 
                if q in data_meas
            ]
            
            if obs_rec_indices:
                lookbacks = [idx - m_index for idx in obs_rec_indices]
                # Observable obs_counter: inner logical for block block_id
                circuit.append(
                    "OBSERVABLE_INCLUDE", 
                    [stim.target_rec(lb) for lb in lookbacks], 
                    obs_counter
                )
            
            obs_counter += 1
            
            # For multi-level concatenation, also emit nested inner logicals
            if inner_is_concatenated:
                obs_counter = self._emit_nested_inner_logicals(
                    circuit, data_meas, m_index, basis,
                    self._inner_code, block_offset, obs_counter
                )
    
    def _emit_nested_inner_logicals(
        self,
        circuit: stim.Circuit,
        data_meas: Dict[int, int],
        m_index: int,
        basis: str,
        nested_code: 'ConcatenatedCode',
        global_offset: int,
        obs_counter: int
    ) -> int:
        """
        Recursively emit observables for nested inner blocks in multi-level concatenation.
        
        For a 3-level code A ∘ (B ∘ C):
        - Level 1: n_A blocks of (B ∘ C)
        - Level 2: Each (B ∘ C) block has n_B sub-blocks of C
        
        This method handles level 2+ recursively.
        
        Parameters
        ----------
        circuit : stim.Circuit
            The circuit being built.
        data_meas : dict
            Mapping from data qubit index to measurement index.
        m_index : int
            Current measurement index.
        basis : str
            Measurement basis.
        nested_code : ConcatenatedCode
            The nested concatenated code to emit observables for.
        global_offset : int
            Offset to add to qubit indices.
        obs_counter : int
            Current observable number to use.
            
        Returns
        -------
        int
            Updated observable counter after emitting all nested observables.
        """
        from qectostim.codes.composite.concatenated import ConcatenatedCode
        
        # Get nested code's inner and outer
        if hasattr(nested_code, '_inner_code') and hasattr(nested_code, '_outer_code'):
            nested_inner = nested_code._inner_code
            nested_outer = nested_code._outer_code
        elif hasattr(nested_code, 'inner') and hasattr(nested_code, 'outer'):
            nested_inner = nested_code.inner
            nested_outer = nested_code.outer
        else:
            return obs_counter
        
        n_nested_outer = nested_outer.n
        n_nested_inner = nested_inner.n
        
        # Get nested inner logical support
        inner_z_ops = get_logical_ops(nested_inner, 'z')
        inner_x_ops = get_logical_ops(nested_inner, 'x')
        
        if basis == "Z" and ops_valid(inner_z_ops) and ops_len(inner_z_ops) > 0:
            inner_logical = inner_z_ops[0]
            pauli_type = ("Z", "Y")
        elif basis == "X" and ops_valid(inner_x_ops) and ops_len(inner_x_ops) > 0:
            inner_logical = inner_x_ops[0]
            pauli_type = ("X", "Y")
        else:
            return obs_counter
        
        inner_logical_support = [
            q for q in range(n_nested_inner) 
            if pauli_at(inner_logical, q) in pauli_type
        ]
        
        if not inner_logical_support:
            return obs_counter
        
        # Emit observables for each sub-block
        for sub_block_id in range(n_nested_outer):
            sub_block_offset = global_offset + sub_block_id * n_nested_inner
            
            # Map inner logical support to global qubit indices
            block_logical_qubits = [q + sub_block_offset for q in inner_logical_support]
            
            obs_rec_indices = [
                data_meas[q] for q in block_logical_qubits 
                if q in data_meas
            ]
            
            if obs_rec_indices:
                lookbacks = [idx - m_index for idx in obs_rec_indices]
                circuit.append(
                    "OBSERVABLE_INCLUDE", 
                    [stim.target_rec(lb) for lb in lookbacks], 
                    obs_counter
                )
            
            obs_counter += 1
            
            # Recurse if nested_inner is also concatenated
            if isinstance(nested_inner, ConcatenatedCode):
                obs_counter = self._emit_nested_inner_logicals(
                    circuit, data_meas, m_index, basis,
                    nested_inner, sub_block_offset, obs_counter
                )
        
        return obs_counter

    def get_detector_slices(self) -> Dict[str, Any]:
        """
        Get detector slice information for ConcatenatedDecoder.
        
        Returns a dict with detector index ranges for each inner block
        and the outer code, compatible with build_concatenation_decoder_metadata().
        
        The detector counts match CSSMemoryExperiment's behavior:
        
        For Z-basis:
        - X stabilizers: (rounds-1) time-like detectors, NO space-like
        - Z stabilizers: rounds time-like + 1 space-like = rounds+1 detectors per stabilizer
        
        For X-basis:
        - X stabilizers: rounds time-like + 1 space-like = rounds+1 detectors per stabilizer
        - Z stabilizers: (rounds-1) time-like detectors, NO space-like
        """
        basis = self.basis.upper()
        
        # Count detectors per inner block (X then Z, accounting for CSSMemoryExperiment behavior)
        if basis == "Z":
            # X stabilizers: skip first round, no space-like → rounds-1
            # Z stabilizers: all rounds + space-like → rounds+1
            inner_x_dets = (self.rounds - 1) * self._n_inner_x
            inner_z_dets = (self.rounds + 1) * self._n_inner_z
            outer_x_dets = (self.rounds - 1) * self._n_outer_x
            outer_z_dets = (self.rounds + 1) * self._n_outer_z
        else:  # X-basis
            # X stabilizers: all rounds + space-like → rounds+1
            # Z stabilizers: skip first round, no space-like → rounds-1
            inner_x_dets = (self.rounds + 1) * self._n_inner_x
            inner_z_dets = (self.rounds - 1) * self._n_inner_z
            outer_x_dets = (self.rounds + 1) * self._n_outer_x
            outer_z_dets = (self.rounds - 1) * self._n_outer_z
        
        inner_dets_per_block = inner_x_dets + inner_z_dets
        outer_dets = outer_x_dets + outer_z_dets
        
        inner_slices = {}
        for block_id in range(self._n_outer):
            start = block_id * inner_dets_per_block
            stop = start + inner_dets_per_block
            inner_slices[block_id] = (start, stop)
        
        outer_start = self._n_outer * inner_dets_per_block
        outer_slices = {0: (outer_start, outer_start + outer_dets)}
        
        total_detectors = self._n_outer * inner_dets_per_block + outer_dets
        
        return {
            'inner_slices': inner_slices,
            'outer_slices': outer_slices,
            'inner_dets_per_block': inner_dets_per_block,
            'outer_dets': outer_dets,
            'n_inner_blocks': self._n_outer,
            'total_detectors': total_detectors,
            'rounds': self.rounds,
            'basis': self.basis,
        }
    
    def build_decoder_metadata(self) -> Dict[str, Any]:
        """
        Build complete metadata for ConcatenatedDecoder.
        
        This generates the 'concatenation' metadata dict that ConcatenatedDecoder
        expects, including per-level DEMs and detector slice mappings.
        
        For multi-level concatenation, this method recursively builds metadata
        for nested inner codes.
        """
        from qectostim.codes.composite.concatenated import ConcatenatedCode
        from qectostim.noise.models import CircuitDepolarizingNoise
        
        noise_model = CircuitDepolarizingNoise(p1=0.001, p2=0.001)
        
        # Generate inner code DEM using standard CSSMemoryExperiment
        inner_exp = CSSMemoryExperiment(
            code=self._inner_code,
            rounds=self.rounds,
            noise_model=noise_model,
            basis=self.basis,
        )
        inner_circuit = noise_model.apply(inner_exp.to_stim())
        try:
            inner_dem = inner_circuit.detector_error_model(decompose_errors=True)
        except Exception:
            inner_dem = inner_circuit.detector_error_model(
                decompose_errors=True,
                ignore_decomposition_failures=True
            )
        
        # Generate outer code DEM
        outer_exp = CSSMemoryExperiment(
            code=self._outer_code,
            rounds=self.rounds,
            noise_model=noise_model,
            basis=self.basis,
        )
        outer_circuit = noise_model.apply(outer_exp.to_stim())
        try:
            outer_dem = outer_circuit.detector_error_model(decompose_errors=True)
        except Exception:
            outer_dem = outer_circuit.detector_error_model(
                decompose_errors=True,
                ignore_decomposition_failures=True
            )
        
        # Build detector slices
        inner_n_detectors = inner_dem.num_detectors
        
        inner_slices = {}
        for block_id in range(self._n_outer):
            start = block_id * inner_n_detectors
            stop = start + inner_n_detectors
            inner_slices[block_id] = (start, stop)
        
        outer_start = self._n_outer * inner_n_detectors
        outer_slices = {0: (outer_start, outer_start + outer_dem.num_detectors)}
        
        # For 2-level concatenation
        concat_meta = {
            'dem_per_level': [inner_dem, outer_dem],
            'dem_slices': [inner_slices, outer_slices],
            'logicals_per_level': [self._inner_code.k, self._outer_code.k],
            'inner_dem': inner_dem,
            'outer_dem': outer_dem,
            'n_inner_blocks': self._n_outer,
            'inner_n_detectors': inner_n_detectors,
            'outer_n_detectors': outer_dem.num_detectors,
            'total_detectors': self._n_outer * inner_n_detectors + outer_dem.num_detectors,
        }
        
        # Handle multi-level: if inner code is concatenated, add its metadata
        if isinstance(self._inner_code, ConcatenatedCode):
            inner_concat_exp = ConcatenatedMemoryExperiment(
                code=self._inner_code,
                rounds=self.rounds,
                noise_model=noise_model,
                basis=self.basis,
            )
            concat_meta['inner_concatenation'] = inner_concat_exp.build_decoder_metadata()
            concat_meta['depth'] = 1 + concat_meta['inner_concatenation'].get('depth', 1)
        else:
            concat_meta['depth'] = 1
        
        return concat_meta
