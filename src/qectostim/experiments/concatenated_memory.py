# src/qectostim/experiments/concatenated_memory.py
"""
Concatenated CSS Memory Experiments.

This module provides memory experiment classes for concatenated CSS codes,
supporting both flat and hierarchical models.

Classes
-------

HierarchicalConcatenatedMemoryExperiment  
    Hierarchical model using logical ancilla blocks for outer stabilizer 
    measurement. Provides proper fault tolerance at both levels.

ConcatenatedCSSMemoryExperiment
    Extended CSS memory experiment for concatenated codes with support
    for both flat and hierarchical round builders.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import stim

from qectostim.experiments.memory import CSSMemoryExperiment
from qectostim.experiments.stabilizer_rounds import (
    DetectorContext,
    StabilizerBasis,
)

if TYPE_CHECKING:
    from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
    from qectostim.noise.models import NoiseModel


class ConcatenatedCSSMemoryExperiment(CSSMemoryExperiment):
    """
    Memory experiment for concatenated CSS codes.
    
    Extends CSSMemoryExperiment with support for hierarchical structure
    and proper detector coverage for concatenated codes.
    
    Parameters
    ----------
    code : ConcatenatedCSSCode
        The concatenated CSS code.
    rounds : int
        Number of syndrome measurement rounds.
    noise_model : NoiseModel | Dict | None
        Noise model to apply. If None, no noise is applied.
    basis : str
        Measurement basis ("Z" or "X").
    use_hierarchical : bool
        If True, use hierarchical round builder with logical ancilla blocks
        for outer stabilizer measurement (fault-tolerant).
        If False (default), use flat round builder treating the concatenated
        code as a single large CSS code (simpler but not FT at outer level).
    block_contiguous : bool
        If True, emit detectors in block-contiguous order for hierarchical
        decoding. Inner block detectors are grouped together before outer
        stabilizer detectors.
    metadata : Dict | None
        Additional experiment metadata.
        
    Attributes
    ----------
    outer_code : CSSCode
        The outer code (acts on logical qubits of inner code).
    inner_code : CSSCode  
        The inner code (provides physical encoding).
    n_out : int
        Number of inner blocks (outer code physical qubits).
    n_in : int
        Number of physical qubits per inner block.
    
    Examples
    --------
    >>> from qectostim.codes.stabilizer.steane import SteaneCode
    >>> from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
    >>> 
    >>> # Steane ∘ Steane concatenation
    >>> outer = SteaneCode()
    >>> inner = SteaneCode()
    >>> concat = ConcatenatedCSSCode(outer, inner)
    >>> 
    >>> # Flat model (simple)
    >>> exp_flat = ConcatenatedCSSMemoryExperiment(
    ...     concat, rounds=3, basis="Z", use_hierarchical=False
    ... )
    >>> 
    >>> # Hierarchical model (fault-tolerant)
    >>> exp_hier = ConcatenatedCSSMemoryExperiment(
    ...     concat, rounds=3, basis="Z", use_hierarchical=True
    ... )
    """
    
    def __init__(
        self,
        code: "ConcatenatedCSSCode",
        rounds: int,
        noise_model: Optional[Any] = None,
        basis: str = "Z",
        use_hierarchical: bool = False,
        block_contiguous: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        d_inner: int = 1,
        emit_inner_observables: bool = False,
    ):
        """
        Initialize concatenated CSS memory experiment.
        
        Parameters
        ----------
        code : ConcatenatedCSSCode
            The concatenated CSS code.
        rounds : int
            For d_inner=1: total syndrome measurement rounds (legacy behavior).
            For d_inner>1: number of outer syndrome rounds (d_outer).
        noise_model : NoiseModel | Dict | None
            Noise model to apply.
        basis : str
            Measurement basis ("Z" or "X").
        use_hierarchical : bool
            If True, use hierarchical round builder.
        block_contiguous : bool
            If True, emit detectors in block-contiguous order.
        metadata : Dict | None
            Additional experiment metadata.
        d_inner : int
            Number of inner syndrome rounds per outer round.
            - d_inner=1 (default): Legacy behavior, inner+outer in each round.
            - d_inner>1: New structure with d_inner inner rounds before each
              destructive outer measurement.
              
        Notes
        -----
        With d_inner > 1, the circuit structure is:
        
            (d_outer) × { (d_inner) × [inner X, inner Z], outer X, outer Z }
            
        where d_outer = rounds. This enables inner temporal detectors within
        each block of d_inner rounds without spanning outer transversal ops.
        """
        # Initialize parent CSSMemoryExperiment
        super().__init__(
            code=code,
            rounds=rounds,
            noise_model=noise_model,
            basis=basis,
            metadata=metadata
        )
        
        self.use_hierarchical = use_hierarchical
        self.block_contiguous = block_contiguous
        self.d_inner = d_inner
        self.emit_inner_observables = emit_inner_observables
        
        # d_outer is just 'rounds' - the number of outer syndrome rounds
        self.d_outer = rounds
        
        self.use_hierarchical = use_hierarchical
        self.block_contiguous = block_contiguous
        
        # Store concatenated code references
        self.outer_code = code.outer
        self.inner_code = code.inner
        self.n_out = code.n_outer
        self.n_in = code.n_inner
        
        # Inner stabilizer counts
        self.r_x_in = self.inner_code.hx.shape[0] if self.inner_code.hx.size > 0 else 0
        self.r_z_in = self.inner_code.hz.shape[0] if self.inner_code.hz.size > 0 else 0
        
        # Outer stabilizer counts
        self.r_x_out = self.outer_code.hx.shape[0] if self.outer_code.hx.size > 0 else 0
        self.r_z_out = self.outer_code.hz.shape[0] if self.outer_code.hz.size > 0 else 0
    
    def to_stim(self) -> stim.Circuit:
        """
        Build a memory experiment circuit for the concatenated CSS code.
        
        Uses either flat or hierarchical round builder based on use_hierarchical.
        
        Returns
        -------
        stim.Circuit
            The complete memory experiment circuit.
        """
        return self._to_stim_hierarchical()
    
    
    def _to_stim_hierarchical(self) -> stim.Circuit:
        """Build circuit using hierarchical round builder with logical ancillas."""
        # Import here to avoid circular imports
        from qectostim.experiments.stabilizer_rounds.hierarchical_concatenated import (
            HierarchicalConcatenatedStabilizerRoundBuilder
        )
        
        basis = self.basis.upper()
        
        # Create detector context for tracking
        ctx = DetectorContext()
        
        # Create hierarchical round builder with d_inner
        builder = HierarchicalConcatenatedStabilizerRoundBuilder(
            self.code, ctx,
            block_name="concat",
            measurement_basis=basis,
            block_contiguous=self.block_contiguous,
            d_inner=self.d_inner,
        )
        
        c = stim.Circuit()
        
        # Emit qubit coordinates
        builder.emit_qubit_coords(c)
        
        # Reset all qubits
        builder.emit_reset_all(c)
        
        # Prepare logical state
        initial_state = "+" if basis == "X" else "0"
        builder.emit_prepare_logical_state(c, state=initial_state, logical_idx=self.logical_qubit)
        
        if self.d_inner > 1:
            # New structure: d_outer outer rounds, each containing d_inner inner rounds
            # Structure: (d_outer) × { (d_inner) × [inner X, inner Z], outer X, outer Z }
            
            # First outer round
            builder.emit_outer_round(c, stab_type=StabilizerBasis.BOTH, emit_detectors=True)
            
            # Subsequent outer rounds in REPEAT block
            if self.d_outer > 1:
                repeat_body = stim.Circuit()
                builder.emit_outer_round(repeat_body, stab_type=StabilizerBasis.BOTH, emit_detectors=True)
                c.append(stim.CircuitRepeatBlock(self.d_outer - 1, repeat_body))
        else:
            # Legacy behavior: d_inner=1, each round has inner + outer
            # First stabilizer round
            builder.emit_round(c, stab_type=StabilizerBasis.BOTH, emit_detectors=True)
            
            # Subsequent rounds in REPEAT block
            if self.rounds > 1:
                repeat_body = stim.Circuit()
                builder.emit_round(repeat_body, stab_type=StabilizerBasis.BOTH, emit_detectors=True)
                c.append(stim.CircuitRepeatBlock(self.rounds - 1, repeat_body))
        
        # Add trailing inner-only rounds for temporal boundary depth.
        # This provides d_inner layers of inner temporal detectors between
        # the last outer round and the final measurement.  The first
        # trailing round gets crossing detectors (accounting for the
        # preceding outer transversal CNOTs); subsequent trailing rounds
        # get temporal detectors.  This decouples the inner boundary
        # detectors from the outer stabiliser chain at the final
        # measurement, preventing low-weight DEM errors.
        for _ in range(self.d_inner):
            builder.emit_inner_only_round(
                c, stab_type=StabilizerBasis.BOTH, emit_detectors=True
            )
        
        # Final measurement
        logical_meas = builder.emit_final_measurement(
            c, basis=basis, logical_idx=self.logical_qubit,
            emit_inner_observables=self.emit_inner_observables,
        )
        
        # If block-contiguous mode, emit deferred detectors
        if self.block_contiguous:
            data_meas = builder.get_data_meas_mapping(ctx.measurement_index - self.code.n)
            builder.emit_deferred_detectors(c, data_meas)
        
        # Emit observable(s)
        ctx.emit_observable(c, observable_idx=0)
        if self.emit_inner_observables:
            for b in range(builder.n_out):
                ctx.emit_observable(c, observable_idx=1 + b)
        
        return c
    
    def get_detector_slices(self) -> Dict[str, Any]:
        """
        Get detector index slices for hierarchical decoding.
        
        Returns detector ranges grouped by inner block and outer stabilizers.
        This is useful for constructing hierarchical decoders that process
        inner syndromes separately from outer syndromes.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with:
            - 'inner_slices': List of (start, end) tuples for each inner block
            - 'outer_slices': (start, end) tuple for outer stabilizer detectors
            - 'inner_dets_per_block': Number of detectors per inner block
            - 'outer_dets': Number of outer stabilizer detectors
        """
        # Compute detector counts per block
        inner_x_dets_per_block = self.r_x_in * self.rounds
        inner_z_dets_per_block = self.r_z_in * self.rounds
        inner_dets_per_block = inner_x_dets_per_block + inner_z_dets_per_block
        
        # Add first-round anchor detectors based on measurement basis
        if self.basis.upper() == "Z":
            inner_dets_per_block += self.r_z_in  # Z anchors
        else:
            inner_dets_per_block += self.r_x_in  # X anchors
        
        # Add space-like (boundary) detectors
        if self.basis.upper() == "Z":
            inner_dets_per_block += self.r_z_in  # Z boundary
        else:
            inner_dets_per_block += self.r_x_in  # X boundary
        
        # Outer detector counts
        outer_x_dets = self.r_x_out * self.rounds
        outer_z_dets = self.r_z_out * self.rounds
        outer_dets = outer_x_dets + outer_z_dets
        
        # Add outer anchors and boundaries
        if self.basis.upper() == "Z":
            outer_dets += self.r_z_out * 2  # anchors + boundary
        else:
            outer_dets += self.r_x_out * 2  # anchors + boundary
        
        # Build slices
        inner_slices = []
        offset = 0
        for block_id in range(self.n_out):
            inner_slices.append((offset, offset + inner_dets_per_block))
            offset += inner_dets_per_block
        
        outer_slices = (offset, offset + outer_dets)
        
        return {
            'inner_slices': inner_slices,
            'outer_slices': outer_slices,
            'inner_dets_per_block': inner_dets_per_block,
            'outer_dets': outer_dets,
        }




class HierarchicalConcatenatedMemoryExperiment(ConcatenatedCSSMemoryExperiment):
    """
    Hierarchical model memory experiment for concatenated CSS codes.
    
    This is equivalent to ConcatenatedCSSMemoryExperiment with use_hierarchical=True.
    Uses logical ancilla blocks for outer stabilizer measurement, providing proper
    fault tolerance at both inner and outer levels.
    
    The circuit structure follows a three-phase round:
    1. Inner X/Z stabilizers measured in parallel on all data blocks
    2. Outer X stabilizers via |+_L⟩ ancilla blocks + transversal CNOTs
    3. Outer Z stabilizers via |0_L⟩ ancilla blocks + transversal CNOTs
    
    Parameters
    ----------
    code : ConcatenatedCSSCode
        The concatenated CSS code.
    rounds : int
        Number of syndrome measurement rounds.
    noise_model : NoiseModel | Dict | None
        Noise model to apply.
    basis : str
        Measurement basis ("Z" or "X").
    block_contiguous : bool
        If True, emit detectors in block-contiguous order.
    metadata : Dict | None
        Additional experiment metadata.
    """
    
    def __init__(
        self,
        code: "ConcatenatedCSSCode",
        rounds: int,
        noise_model: Optional[Any] = None,
        basis: str = "Z",
        block_contiguous: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            code=code,
            rounds=rounds,
            noise_model=noise_model,
            basis=basis,
            use_hierarchical=True,
            block_contiguous=block_contiguous,
            metadata=metadata,
        )


# Aliases for naming consistency
FlatConcatenatedMemoryExperiment = ConcatenatedCSSMemoryExperiment
ConcatenatedMemoryExperiment = ConcatenatedCSSMemoryExperiment  # Deprecated
