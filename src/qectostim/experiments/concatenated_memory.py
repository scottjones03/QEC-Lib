# src/qectostim/experiments/concatenated_memory.py
"""
Flat memory experiment for concatenated codes with block-grouped detectors.

This module provides FlatConcatenatedMemoryExperiment (formerly ConcatenatedMemoryExperiment),
which generates circuits where detectors are organized by inner code blocks, enabling 
hierarchical decoding with ConcatenatedDecoder.

IMPORTANT: FLAT CONCATENATION MODEL
===================================
This module implements a SIMPLIFIED "flat" model of concatenated codes where the 
concatenated code is treated as a single large stabilizer code. All stabilizers 
(inner and lifted outer) are measured using direct physical syndrome extraction 
circuits with physical CNOT gates.

This approach:
✓ Correctly computes the stabilizer group of the concatenated code
✓ Is useful for code structure validation and testing
✓ Works with hierarchical decoders that expect block-grouped syndromes
✗ Does NOT use fault-tolerant logical gadgets for outer operations
✗ Does NOT capture the error suppression hierarchy of true concatenation
✗ Does NOT accurately predict concatenated code threshold behavior

For a proper hierarchical implementation using fault-tolerant logical 
operations (where outer stabilizers are measured via logical CNOTs on 
inner code blocks), see HierarchicalConcatenatedMemoryExperiment.

See Also
--------
HierarchicalConcatenatedMemoryExperiment : Proper hierarchical implementation
    using fault-tolerant logical gadgets for outer code operations.
"""
from __future__ import annotations

import warnings
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
from qectostim.experiments.stabilizer_rounds import (
    FlatConcatenatedStabilizerRoundBuilder,
    DetectorContext,
    StabilizerBasis,
)


class FlatConcatenatedMemoryExperiment(CSSMemoryExperiment):
    """
    Flat memory experiment for concatenated CSS codes with block-grouped detectors.
    
    IMPORTANT: FLAT CONCATENATION MODEL
    ===================================
    This class implements a SIMPLIFIED "flat" model of concatenated codes where
    the concatenated code is treated as a single large stabilizer code. All 
    stabilizers (inner and lifted outer) are measured using direct physical 
    syndrome extraction circuits.
    
    This approach:
    ✓ Correctly computes the stabilizer group of the concatenated code
    ✓ Is useful for code structure validation and testing
    ✓ Works with hierarchical decoders that expect block-grouped syndromes
    ✗ Does NOT use fault-tolerant logical gadgets for outer operations
    ✗ Does NOT capture the error suppression hierarchy of true concatenation
    ✗ Does NOT accurately predict concatenated code threshold behavior
    
    For a proper hierarchical implementation using fault-tolerant logical 
    operations, see HierarchicalConcatenatedMemoryExperiment.
    
    Circuit Structure
    -----------------
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
        
    See Also
    --------
    HierarchicalConcatenatedMemoryExperiment : Proper hierarchical implementation
        using fault-tolerant logical gadgets for outer code operations.
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
                f"FlatConcatenatedMemoryExperiment requires a ConcatenatedCode, "
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
        
        Uses ConcatenatedStabilizerRoundBuilder for consistent circuit construction
        with block-contiguous detector ordering required by ConcatenatedDecoder.
        
        IMPORTANT: Detectors are emitted in BLOCK-CONTIGUOUS order, meaning
        all detectors for block 0 come first (across all rounds), then all for block 1,
        etc., and finally all outer detectors. This enables simple slicing for
        per-block syndrome extraction in hierarchical decoding.
        
        Detector organization (for Z-basis memory):
            [Block 0: all rounds time-like + space-like]
            [Block 1: all rounds time-like + space-like]
            ...
            [Block N-1: all rounds time-like + space-like]
            [Outer: all rounds time-like + space-like]
        """
        basis = self.basis.upper()
        
        # Create detector context for tracking
        ctx = DetectorContext()
        
        # Create concatenated round builder with block_contiguous mode
        # for hierarchical decoder compatibility
        builder = FlatConcatenatedStabilizerRoundBuilder(
            self.code, ctx,
            block_name="main",
            measurement_basis=basis,
            block_contiguous=True,  # Enable deferred detector emission
        )
        
        c = stim.Circuit()
        
        # Emit qubit coordinates
        builder.emit_qubit_coords(c)
        
        # Reset all qubits
        builder.emit_reset_all(c)
        
        # Prepare logical state
        initial_state = "+" if basis == "X" else "0"
        builder.emit_prepare_logical_state(c, state=initial_state, logical_idx=self.logical_qubit)
        
        # Emit all stabilizer measurement rounds (detectors deferred)
        for _ in range(self.rounds):
            builder.emit_round(c, stab_type=StabilizerBasis.BOTH, emit_detectors=True)
        
        # Final data measurement
        logical_meas = builder.emit_final_measurement(c, basis=basis, logical_idx=self.logical_qubit)
        
        # Get data measurement mapping for space-like detectors
        n = self.code.n
        data_meas_start = ctx.measurement_index - n
        data_meas = builder.get_data_meas_mapping(data_meas_start)
        
        # Emit all deferred detectors in block-contiguous order
        builder.emit_deferred_detectors(c, data_meas)
        
        # Get syndrome measurement history for deterministic observable construction
        syndrome_history = builder.get_syndrome_meas_history()
        
        # Emit main logical observable
        ctx.emit_observable(c, observable_idx=0)
        
        # NOTE: Inner logical observables are NOT emitted.
        # They cause "non-deterministic observable" errors for any outer code
        # with cross-block stabilizers (i.e., all except repetition codes).
        # Hierarchical decoders compute inner logical probabilities directly
        # from block syndromes via _compute_inner_logical_prob(), so inner
        # observables in the circuit are unnecessary.
        
        return c
    
    def _can_emit_deterministic_inner_observables(self, basis: str) -> bool:
        """
        Check if inner observables can be made deterministic for this code/basis combination.
        
        Inner observables become non-deterministic when outer code syndrome extraction
        uses transversal CNOTs that spread Z-operators across blocks. Specifically:
        
        For Z-basis memory with inner Z-logicals:
        - If outer code has X-stabilizers touching multiple blocks, Z propagates
          through those X-stabilizer CNOTs to other blocks' X-ancillas
        - This cannot be fixed by including syndrome measurements because the
          Hadamard gates on X-ancillas transform Z→X, which then spreads further
          
        For X-basis memory with inner X-logicals:
        - Same issue but with outer Z-stabilizers
        
        Returns True if inner observables can be deterministic (outer code has no
        cross-block stabilizers of the relevant type), False otherwise.
        """
        import numpy as np
        
        # Get the relevant outer stabilizer matrix
        if basis.upper() == "Z":
            # Z-basis: check outer X-stabilizers
            outer_stab = self._outer_code.hx if hasattr(self._outer_code, 'hx') else None
        else:
            # X-basis: check outer Z-stabilizers
            outer_stab = self._outer_code.hz if hasattr(self._outer_code, 'hz') else None
        
        if outer_stab is None or outer_stab.size == 0:
            # No cross-type stabilizers → safe
            return True
        
        if hasattr(outer_stab, 'toarray'):
            outer_stab = outer_stab.toarray()
        outer_stab = np.atleast_2d(outer_stab)
        
        # Check if any stabilizer touches more than one block
        for row_idx in range(outer_stab.shape[0]):
            row = outer_stab[row_idx]
            support = np.where(row != 0)[0]
            if len(support) > 1:
                # This stabilizer connects multiple blocks → non-deterministic
                return False
        
        # All stabilizers are single-block (or no stabilizers) → safe
        return True
    
    def _emit_inner_logical_observables(
        self,
        circuit: stim.Circuit,
        data_meas: Dict[int, int],
        m_index: int,
        basis: str,
        syndrome_history: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Emit OBSERVABLE_INCLUDE for each inner block's logical operator.
        
        For hierarchical decoding, we need to track inner logical errors separately
        from the main concatenated logical. Each inner block gets its own observable.
        
        To make observables deterministic, we include syndrome measurements that
        share support with the inner logical operator. This cancels the non-determinism
        caused by CNOT gates propagating Z-observables to X-ancillas during syndrome
        extraction.
        
        NOTE: Inner observables cannot always be made deterministic. If the outer code
        has X-stabilizers (for Z-basis) or Z-stabilizers (for X-basis) that connect
        multiple blocks, the transversal syndrome extraction will spread Pauli operators
        across blocks in a way that cannot be cancelled. In such cases, this method
        will emit a warning and skip inner observable emission.
        
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
        syndrome_history : dict, optional
            Syndrome measurement history from builder.get_syndrome_meas_history().
            Currently unused - inner observables in deterministic cases don't need
            syndrome measurements. Kept for potential future use.
        """
        from qectostim.codes.composite.concatenated import ConcatenatedCode
        
        # Always emit inner observables - Stim handles non-determinism via 
        # gauge detectors / error decomposition. This enables hierarchical and
        # turbo decoding for all concatenated codes.
        # (Previously we checked _can_emit_deterministic_inner_observables and
        # skipped emission if False, but that prevented turbo decoding.)
        
        # Check if inner code is itself concatenated (multi-level)
        inner_is_concatenated = isinstance(self._inner_code, ConcatenatedCode)
        
        n_inner = self._inner_code.n
        
        # Get inner logical support using robust method with fallback
        inner_logical_support = self._get_inner_logical_support(self._inner_code, basis)
        
        if not inner_logical_support:
            # Cannot emit inner observables without logical support
            import warnings
            warnings.warn(
                f"FlatConcatenatedMemoryExperiment: Could not determine inner logical "
                f"support for {type(self._inner_code).__name__} in {basis}-basis. "
                f"Inner observables will not be emitted (hierarchical/turbo decoding unavailable)."
            )
            return
        
        # Note: For deterministic cases (outer code has no cross-block X/Z-stabs),
        # we don't need to include syndrome measurements in the observable.
        # The inner observables are naturally deterministic because:
        # 1. The inner X/Z-stab MR operations properly absorb the backward-propagated
        #    operators through their reset operation
        # 2. No cross-block spreading occurs through outer transversal gates
        
        # Observable counter starts at 1 (0 is main concatenated logical)
        obs_counter = 1
        
        # Emit an observable for each inner block
        for block_id in range(self._n_outer):
            block_offset = block_id * n_inner
            
            # Map inner logical support to global qubit indices for this block
            block_logical_qubits = [q + block_offset for q in inner_logical_support]
            
            # Get data measurement indices - ONLY include data measurements
            # No syndrome measurements needed for deterministic cases
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
                    self._inner_code, block_offset, obs_counter,
                    syndrome_history
                )
    
    def _get_inner_logical_support(self, code, basis: str) -> List[int]:
        """
        Get the qubit indices that support the inner logical operator.
        
        Uses multiple fallback methods to robustly extract logical support:
        1. Try get_logical_ops() with pauli_at parsing
        2. Fall back to lz/lx matrix row (CSS logical representation)
        3. Fall back to checking code's logical_z/logical_x attributes
        
        Parameters
        ----------
        code : CSSCode
            The code to extract logical support from.
        basis : str
            Measurement basis ('X' or 'Z').
            
        Returns
        -------
        List[int]
            Qubit indices in the support of the logical operator.
        """
        from scipy import sparse
        
        n = code.n
        
        # Method 1: Try get_logical_ops() (handles Pauli string format)
        if basis == "Z":
            ops = get_logical_ops(code, 'z')
            pauli_type = ("Z", "Y")
        else:
            ops = get_logical_ops(code, 'x')
            pauli_type = ("X", "Y")
        
        if ops_valid(ops) and ops_len(ops) > 0:
            logical_op = ops[0]
            support = [q for q in range(n) if pauli_at(logical_op, q) in pauli_type]
            if support:
                return support
        
        # Method 2: Try lz/lx matrices (CSS code representation)
        # For Z-basis memory, we measure Z and track Z-type logicals
        # Z logical errors are detected by X stabilizers and tracked by lz
        if basis == "Z" and hasattr(code, 'lz') and code.lz is not None:
            lz = code.lz
            if sparse.issparse(lz):
                lz = lz.toarray()
            lz = np.atleast_2d(lz)
            if lz.shape[0] > 0 and lz.shape[1] == n:
                # First logical Z operator - nonzero entries are support
                support = list(np.where(lz[0] != 0)[0])
                if support:
                    return support
        
        if basis == "X" and hasattr(code, 'lx') and code.lx is not None:
            lx = code.lx
            if sparse.issparse(lx):
                lx = lx.toarray()
            lx = np.atleast_2d(lx)
            if lx.shape[0] > 0 and lx.shape[1] == n:
                # First logical X operator - nonzero entries are support
                support = list(np.where(lx[0] != 0)[0])
                if support:
                    return support
        
        # Method 3: Try direct logical_z/logical_x attributes
        # Handle multiple formats: numpy array, list-of-dicts, list-of-strings
        # TopologicalCSSCode uses _logical_z/_logical_x (with underscore)
        attr_names = ['logical_z', '_logical_z'] if basis == "Z" else ['logical_x', '_logical_x']
        pauli_type = ('Z', 'Y') if basis == "Z" else ('X', 'Y')
        
        for attr_name in attr_names:
            if not hasattr(code, attr_name):
                continue
            logical = getattr(code, attr_name, None)
            if logical is None:
                continue
            
            # Handle list-of-dicts format (e.g., [{0: 'Z', 2: 'Z'}, ...])
            # Used by TopologicalCSSCode subclasses like FourQubit422Code
            if isinstance(logical, list) and len(logical) > 0:
                first_op = logical[0]
                if isinstance(first_op, dict):
                    # Dict format: {qubit_idx: pauli_char}
                    support = [q for q, p in first_op.items() if p in pauli_type]
                    if support:
                        return support
                elif isinstance(first_op, str):
                    # String format: "ZIZI"
                    support = [q for q, p in enumerate(first_op) if p in pauli_type]
                    if support:
                        return support
            
            # Handle numpy array / sparse matrix format (original logic)
            if sparse.issparse(logical):
                logical = logical.toarray()
            try:
                logical_arr = np.atleast_2d(np.asarray(logical))
                # Only process if it's a numeric array (not object array from list-of-dicts)
                if logical_arr.dtype.kind in ('i', 'u', 'f', 'b'):  # int, uint, float, bool
                    if logical_arr.shape[0] > 0:
                        support = list(np.where(logical_arr[0] != 0)[0])
                        if support:
                            return support
            except (ValueError, TypeError):
                # Conversion failed, try next attribute
                pass
        
        # No support found
        return []

    def _emit_nested_inner_logicals(
        self,
        circuit: stim.Circuit,
        data_meas: Dict[int, int],
        m_index: int,
        basis: str,
        nested_code: 'ConcatenatedCode',
        global_offset: int,
        obs_counter: int,
        syndrome_history: Optional[Dict[str, Any]] = None,
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
        
        # Get nested inner logical support using robust method with fallback
        inner_logical_support = self._get_inner_logical_support(nested_inner, basis)
        
        if not inner_logical_support:
            # Can't emit nested observables without logical support
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
            
            # Note: For nested inner blocks, we don't yet include syndrome measurements
            # because the syndrome_history structure is organized for the top level.
            # This is a limitation - nested inner observables may still be non-deterministic.
            # TODO: Track per-level syndrome history for full multi-level support.
            
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
                    nested_inner, sub_block_offset, obs_counter,
                    syndrome_history  # Pass through (not used yet for nested)
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
            inner_concat_exp = FlatConcatenatedMemoryExperiment(
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


# =============================================================================
# Backward Compatibility Alias (Deprecated)
# =============================================================================

def _deprecated_concatenated_memory_experiment(*args, **kwargs):
    """Deprecated: Use FlatConcatenatedMemoryExperiment instead."""
    warnings.warn(
        "ConcatenatedMemoryExperiment is deprecated and will be removed in a future "
        "version. Use FlatConcatenatedMemoryExperiment for flat concatenation model, "
        "or HierarchicalConcatenatedMemoryExperiment for proper hierarchical "
        "fault-tolerant concatenation.",
        DeprecationWarning,
        stacklevel=2
    )
    return FlatConcatenatedMemoryExperiment(*args, **kwargs)


class ConcatenatedMemoryExperiment(FlatConcatenatedMemoryExperiment):
    """
    DEPRECATED: Use FlatConcatenatedMemoryExperiment instead.
    
    This alias exists for backward compatibility and will be removed in a future version.
    
    For flat concatenation model: FlatConcatenatedMemoryExperiment
    For hierarchical FT concatenation: HierarchicalConcatenatedMemoryExperiment
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ConcatenatedMemoryExperiment is deprecated and will be removed in a future "
            "version. Use FlatConcatenatedMemoryExperiment for flat concatenation model, "
            "or HierarchicalConcatenatedMemoryExperiment for proper hierarchical "
            "fault-tolerant concatenation.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
