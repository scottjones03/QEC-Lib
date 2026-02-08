# src/qectostim/experiments/detector_emission.py
"""
Detector Emission Module for FT Gadget Experiments.

This module extracts detector-related emission logic from FaultTolerantGadgetExperiment
into dedicated, reusable components. It handles:

1. Crossing detectors: Compare pre-gate and post-gate stabilizer measurements
2. Boundary detectors: Compare final data measurements with last syndrome round
3. Detector coordinate computation

The key insight is that detector emission depends on:
- CrossingDetectorConfig: Formulas for crossing detectors
- BoundaryDetectorConfig: Which boundary detectors to emit
- StabilizerTransform: How stabilizers change through the gate

By centralizing this logic, we enable:
- Clean separation of concerns from the experiment class
- Reuse across different gadget types
- Consistent detector coverage matching ground truth builders
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import stim

from qectostim.experiments.detector_tracking import compute_detector_coords

if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds import DetectorContext
    from qectostim.experiments.stabilizer_rounds.base import (
        BaseStabilizerRoundBuilder,
        StabilizerBasis,
    )
    from qectostim.gadgets.base import (
        CrossingDetectorConfig,
        CrossingDetectorFormula,
        CrossingDetectorTerm,
        BoundaryDetectorConfig,
    )
    from qectostim.gadgets.scheduling import (
        BlockSchedule,
        RoundSchedule,
    )


@dataclass
class DetectorEmissionContext:
    """
    Context for detector emission operations.
    
    Tracks the state needed for emitting various detector types.
    
    Attributes
    ----------
    builders : Dict[str, BaseStabilizerRoundBuilder]
        Block name to builder mapping.
    pre_gadget_meas : Dict[str, Dict[str, List[int]]]
        Pre-gadget measurement indices per block.
    post_gadget_meas : Dict[str, Dict[str, List[int]]]
        Post-gadget measurement indices per block.
    destroyed_blocks : Set[str]
        Blocks destroyed during the gadget.
    total_measurements : int
        Total measurements in circuit (for lookback calculation).
    """
    builders: Dict[str, "BaseStabilizerRoundBuilder"] = field(default_factory=dict)
    pre_gadget_meas: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    post_gadget_meas: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    destroyed_blocks: Set[str] = field(default_factory=set)
    total_measurements: int = 0


class CrossingDetectorEmitter:
    """
    Emitter for crossing detectors around a transversal gate.
    
    Crossing detectors compare stabilizer measurements before and after
    a transversal gate. They are essential for detecting errors that
    occur during the gate application.
    
    For different gates, crossing detectors have different formulas:
    
    - Identity/unchanged stabilizers: 2-term (pre ⊕ post)
    - CZ X stabilizers: 3-term (X_pre ⊕ X_post ⊕ Z_post)
    - CNOT Z_control: 3-term for |+⟩ input (Z_pre ⊕ Z_post_ctrl ⊕ Z_post_tgt)
    
    This class extracts and centralizes the logic from:
    - FaultTolerantGadgetExperiment._emit_crossing_detectors()
    - FaultTolerantGadgetExperiment._get_detector_coords()
    """
    
    def __init__(self, ctx: DetectorEmissionContext):
        """
        Initialize the emitter.
        
        Parameters
        ----------
        ctx : DetectorEmissionContext
            Context with builder and measurement information.
        """
        self.ctx = ctx
    
    def emit_crossing_detectors(
        self,
        circuit: stim.Circuit,
        config: "CrossingDetectorConfig",
    ) -> int:
        """
        Emit crossing detectors based on configuration.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        config : CrossingDetectorConfig
            Configuration specifying crossing detector formulas.
            
        Returns
        -------
        int
            Number of detectors emitted.
        """
        if config is None:
            return 0
        
        total_emitted = 0
        
        for formula in config.formulas:
            emitted = self._emit_formula(circuit, formula)
            total_emitted += emitted
        
        return total_emitted
    
    def _emit_formula(
        self,
        circuit: stim.Circuit,
        formula: "CrossingDetectorFormula",
    ) -> int:
        """
        Emit detectors for a single crossing formula.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        formula : CrossingDetectorFormula
            The crossing detector formula to emit.
            
        Returns
        -------
        int
            Number of detectors emitted.
        """
        if not formula.terms:
            return 0
        
        # Get first term to determine stabilizer count
        first_term = formula.terms[0]
        first_block = first_term.block
        first_basis = first_term.stabilizer_type.upper()
        
        # Get stabilizer count from builder
        builder = self.ctx.builders.get(first_block)
        if builder is None:
            return 0
        
        if first_basis == 'X':
            num_stab = len(builder.x_ancillas)
        else:
            num_stab = len(builder.z_ancillas)
        
        # Use explicit stabilizer count if provided
        if formula.num_stabilizers is not None:
            num_stab = min(num_stab, formula.num_stabilizers)
        
        emitted = 0
        
        # Emit detector for each stabilizer
        for stab_idx in range(num_stab):
            targets = self._build_detector_targets(formula.terms, stab_idx)
            
            if targets:
                coords = self._get_detector_coords(builder, first_basis, stab_idx)
                circuit.append("DETECTOR", targets, coords)
                emitted += 1
        
        return emitted
    
    def _build_detector_targets(
        self,
        terms: List["CrossingDetectorTerm"],
        stab_idx: int,
    ) -> List[stim.GateTarget]:
        """
        Build detector targets for a single stabilizer.
        
        Parameters
        ----------
        terms : List[CrossingDetectorTerm]
            Terms in the crossing detector formula.
        stab_idx : int
            Index of the stabilizer.
            
        Returns
        -------
        List[stim.GateTarget]
            Targets for the DETECTOR instruction.
        """
        targets = []
        
        for term in terms:
            block = term.block
            basis = term.stabilizer_type.lower()
            phase = term.timing
            
            # Get measurement indices based on timing
            if phase == 'pre':
                meas_dict = self.ctx.pre_gadget_meas.get(block, {})
            else:  # 'post'
                meas_dict = self.ctx.post_gadget_meas.get(block, {})
            
            meas_list = meas_dict.get(basis, [])
            
            if stab_idx < len(meas_list):
                abs_idx = meas_list[stab_idx]
                if abs_idx is not None:
                    lookback = abs_idx - self.ctx.total_measurements
                    targets.append(stim.target_rec(lookback))
        
        return targets
    
    def _get_detector_coords(
        self,
        builder: "BaseStabilizerRoundBuilder",
        basis: str,
        stab_idx: int,
    ) -> List[float]:
        """
        Get detector coordinates for a stabilizer.
        
        Delegates to detector_tracking.compute_detector_coords() for the
        coordinate computation, using builder's qubit_coords for offset.
        
        Parameters
        ----------
        builder : BaseStabilizerRoundBuilder
            Builder for the block.
        basis : str
            'X' or 'Z' stabilizer type.
        stab_idx : int
            Index of the stabilizer.
            
        Returns
        -------
        List[float]
            Coordinates for the detector.
        """
        if basis.upper() == 'X':
            ancillas = builder.x_ancillas
        else:
            ancillas = builder.z_ancillas
        
        if stab_idx < len(ancillas):
            qubit_idx = ancillas[stab_idx]
            if qubit_idx in builder.qubit_coords:
                coords = list(builder.qubit_coords[qubit_idx])
                coords.append(float(builder._round_number))
                return coords
        
        # Fallback to canonical coordinate computation
        coords = compute_detector_coords(
            block_name=builder.block_name,
            stabilizer_type=basis.upper(),
            stabilizer_index=stab_idx,
            time_coord=float(builder._round_number),
        )
        return list(coords)


class BoundaryDetectorEmitter:
    """
    Emitter for boundary (space-like) detectors.
    
    Boundary detectors compare final data qubit measurements with the
    last syndrome measurement. They work when the measurement basis
    matches the stabilizer type being checked.
    
    For example:
    - MZ data with Z stabilizers: Boundary detector possible
    - MX data with X stabilizers: Boundary detector possible
    - MX data with Z stabilizers: No boundary (basis mismatch)
    """
    
    def __init__(
        self,
        builders: Dict[str, "BaseStabilizerRoundBuilder"],
    ):
        """
        Initialize the emitter.
        
        Parameters
        ----------
        builders : Dict[str, BaseStabilizerRoundBuilder]
            Block name to builder mapping.
        """
        self.builders = builders
    
    def emit_boundary_detectors(
        self,
        circuit: stim.Circuit,
        config: "BoundaryDetectorConfig",
        block_meas_offsets: Dict[str, int],
        total_measurements: int,
    ) -> int:
        """
        Emit boundary detectors based on configuration.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        config : BoundaryDetectorConfig
            Configuration specifying which boundary detectors to emit.
        block_meas_offsets : Dict[str, int]
            Starting measurement index for each block's data qubits.
        total_measurements : int
            Total measurements in circuit (for lookback calculation).
            
        Returns
        -------
        int
            Number of detectors emitted.
        """
        if config is None:
            return 0
        
        total_emitted = 0
        
        for block_name, block_config in config.block_configs.items():
            builder = self.builders.get(block_name)
            if builder is None:
                continue
            
            meas_offset = block_meas_offsets.get(block_name)
            if meas_offset is None:
                continue
            
            # Emit X boundary detectors if configured
            if block_config.get("X", False):
                emitted = builder.emit_space_like_detectors(
                    circuit, "X", data_meas_start=meas_offset
                )
                total_emitted += emitted if isinstance(emitted, int) else 0
            
            # Emit Z boundary detectors if configured
            if block_config.get("Z", False):
                emitted = builder.emit_space_like_detectors(
                    circuit, "Z", data_meas_start=meas_offset
                )
                total_emitted += emitted if isinstance(emitted, int) else 0
        
        return total_emitted


def emit_crossing_detectors(
    circuit: stim.Circuit,
    builders: List["BaseStabilizerRoundBuilder"],
    pre_gadget_meas: Dict[str, Dict[str, List[int]]],
    crossing_config: "CrossingDetectorConfig",
    destroyed_blocks: Set[str],
) -> int:
    """
    Convenience function to emit crossing detectors.
    
    Parameters
    ----------
    circuit : stim.Circuit
        Circuit to emit into.
    builders : List[BaseStabilizerRoundBuilder]
        Round builders for each block.
    pre_gadget_meas : Dict[str, Dict[str, List[int]]]
        Pre-gadget measurement indices.
    crossing_config : CrossingDetectorConfig
        Crossing detector configuration.
    destroyed_blocks : Set[str]
        Blocks destroyed during the gadget.
        
    Returns
    -------
    int
        Number of detectors emitted.
    """
    # Build block lookup
    block_builders = {b.block_name: b for b in builders}
    
    # Get post-gadget measurements
    post_gadget_meas = {}
    for builder in builders:
        if builder.block_name not in destroyed_blocks:
            post_gadget_meas[builder.block_name] = builder.get_last_measurement_indices()
    
    # Create context
    ctx = DetectorEmissionContext(
        builders=block_builders,
        pre_gadget_meas=pre_gadget_meas,
        post_gadget_meas=post_gadget_meas,
        destroyed_blocks=destroyed_blocks,
        total_measurements=circuit.num_measurements,
    )
    
    # Emit
    emitter = CrossingDetectorEmitter(ctx)
    return emitter.emit_crossing_detectors(circuit, crossing_config)


def emit_boundary_detectors(
    circuit: stim.Circuit,
    builders: List["BaseStabilizerRoundBuilder"],
    config: "BoundaryDetectorConfig",
    block_meas_offsets: Dict[str, int],
) -> int:
    """
    Convenience function to emit boundary detectors.
    
    Parameters
    ----------
    circuit : stim.Circuit
        Circuit to emit into.
    builders : List[BaseStabilizerRoundBuilder]
        Round builders for each block.
    config : BoundaryDetectorConfig
        Boundary detector configuration.
    block_meas_offsets : Dict[str, int]
        Starting measurement index for each block's data qubits.
        
    Returns
    -------
    int
        Number of detectors emitted.
    """
    block_builders = {b.block_name: b for b in builders}
    emitter = BoundaryDetectorEmitter(block_builders)
    return emitter.emit_boundary_detectors(
        circuit, config, block_meas_offsets, circuit.num_measurements
    )


class MemoryRoundEmitter:
    """
    Emitter for stabilizer memory rounds (pre/post-gadget and inter-phase).
    
    Encapsulates the shared logic for emitting stabilizer measurement rounds
    with proper scheduling-driven basis ordering and anchor/temporal detectors.
    
    This eliminates duplication between:
    - FaultTolerantGadgetExperiment._emit_sequential_round / _emit_parallel_round
    - PhaseOrchestrator._emit_inter_phase_rounds
    
    Both sequential and parallel extraction modes are supported:
    - Sequential: each builder emits a full round (emit_round)
    - Parallel: all builders share Z/X layers (emit_z_layer/emit_x_layer)
    
    Usage
    -----
    >>> emitter = MemoryRoundEmitter(builders)
    >>> emitter.emit_round(circuit, round_schedules, parallel=False, emit_detectors=True)
    """
    
    def __init__(
        self,
        builders: List["BaseStabilizerRoundBuilder"],
    ):
        """
        Initialize the emitter.
        
        Parameters
        ----------
        builders : List[BaseStabilizerRoundBuilder]
            Round builders for active code blocks.
        """
        self.builders = builders
    
    def emit_round(
        self,
        circuit: stim.Circuit,
        round_info: Dict[str, "RoundSchedule"],
        parallel: bool = False,
        emit_detectors: bool = True,
        stab_type: Optional["StabilizerBasis"] = None,
    ) -> None:
        """
        Emit one stabilizer measurement round using schedule-driven ordering.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        round_info : Dict[str, RoundSchedule]
            Per-block scheduling decisions for this round.
        parallel : bool
            If True, use parallel extraction (shared Z/X layers).
            If False, each builder emits a full sequential round.
        emit_detectors : bool
            Whether to emit temporal/anchor detectors.
        stab_type : Optional[StabilizerBasis]
            Which stabilizers to measure (for hybrid decoding).
            None or BOTH means measure all.
        """
        if parallel:
            self._emit_parallel(circuit, round_info, emit_detectors)
        else:
            self._emit_sequential(circuit, round_info, emit_detectors, stab_type)
    
    def _emit_sequential(
        self,
        circuit: stim.Circuit,
        round_info: Dict[str, "RoundSchedule"],
        emit_detectors: bool = True,
        stab_type: Optional["StabilizerBasis"] = None,
    ) -> None:
        """
        Emit one round sequentially — each builder emits a full round.
        
        Uses RoundSchedule to set anchor flags and first_basis ordering.
        """
        from qectostim.experiments.stabilizer_rounds import StabilizerBasis as RoundBasis
        from qectostim.gadgets.scheduling import StabilizerBasis as SchedulerBasis
        
        # Default stab_type to BOTH
        if stab_type is None:
            stab_type = RoundBasis.BOTH
        
        for builder in self.builders:
            rs = round_info[builder.block_name]
            
            emit_z_anchors = rs.is_anchor_round and rs.anchor_basis == SchedulerBasis.Z
            emit_x_anchors = rs.is_anchor_round and rs.anchor_basis == SchedulerBasis.X
            
            # Map scheduler basis to stabilizer_rounds basis for first_basis
            first_basis_mapped = None
            if rs.first_basis == SchedulerBasis.Z:
                first_basis_mapped = RoundBasis.Z
            elif rs.first_basis == SchedulerBasis.X:
                first_basis_mapped = RoundBasis.X
            
            builder.emit_round(
                circuit,
                stab_type,
                emit_detectors=emit_detectors,
                emit_z_anchors=emit_z_anchors,
                emit_x_anchors=emit_x_anchors,
                explicit_anchor_mode=rs.is_anchor_round,
                first_basis=first_basis_mapped,
            )
    
    def _emit_parallel(
        self,
        circuit: stim.Circuit,
        round_info: Dict[str, "RoundSchedule"],
        emit_detectors: bool = True,
    ) -> None:
        """
        Emit one round in parallel — all builders share Z/X layers.
        
        For anchor rounds, blocks are grouped by first_basis:
        - X-first blocks (|+⟩ prep): emit X layer first, then Z
        - Z-first blocks (|0⟩ prep): emit Z layer first, then X
        
        For non-anchor rounds, standard Z-then-X order is used.
        """
        from qectostim.gadgets.scheduling import StabilizerBasis as SchedulerBasis
        
        is_anchor_round = any(rs.is_anchor_round for rs in round_info.values())
        
        if is_anchor_round:
            # Group builders by which basis needs to be measured first
            x_first = [b for b in self.builders
                       if round_info[b.block_name].first_basis == SchedulerBasis.X]
            z_first = [b for b in self.builders
                       if round_info[b.block_name].first_basis == SchedulerBasis.Z]
            other = [b for b in self.builders
                     if b not in x_first and b not in z_first]
            
            # X-first blocks: X layer (with anchors), then Z layer
            for b in x_first:
                rs = round_info[b.block_name]
                b.emit_x_layer(circuit, emit_detectors=emit_detectors,
                               emit_x_anchors=(rs.anchor_basis == SchedulerBasis.X))
            for b in z_first:
                rs = round_info[b.block_name]
                b.emit_z_layer(circuit, emit_detectors=emit_detectors,
                               emit_z_anchors=(rs.anchor_basis == SchedulerBasis.Z))
            
            # Second layer for each group
            for b in x_first:
                b.emit_z_layer(circuit, emit_detectors=emit_detectors, emit_z_anchors=False)
            for b in z_first:
                b.emit_x_layer(circuit, emit_detectors=emit_detectors, emit_x_anchors=False)
            
            # Non-anchor blocks: standard Z then X
            for b in other:
                b.emit_z_layer(circuit, emit_detectors=emit_detectors, emit_z_anchors=False)
            for b in other:
                b.emit_x_layer(circuit, emit_detectors=emit_detectors, emit_x_anchors=False)
        else:
            # Standard ordering: Z layer for all, then X layer for all
            for b in self.builders:
                b.emit_z_layer(circuit, emit_detectors=emit_detectors, emit_z_anchors=False)
            for b in self.builders:
                b.emit_x_layer(circuit, emit_detectors=emit_detectors, emit_x_anchors=False)
        
        # Finalize round for all builders
        for b in self.builders:
            b.finalize_parallel_round(circuit)


def emit_scheduled_rounds(
    circuit: stim.Circuit,
    builders: List["BaseStabilizerRoundBuilder"],
    schedules: Dict[str, "BlockSchedule"],
    num_rounds: int,
    parallel: bool = False,
    stab_type: Optional["StabilizerBasis"] = None,
    crossing_config: Optional["CrossingDetectorConfig"] = None,
    pre_gadget_meas: Optional[Dict[str, Dict[str, List[int]]]] = None,
    destroyed_blocks: Optional[Set[str]] = None,
    all_builders: Optional[List["BaseStabilizerRoundBuilder"]] = None,
) -> None:
    """
    Convenience function to emit multiple scheduled rounds with optional crossing detectors.
    
    This is the unified entry point for emitting stabilizer memory rounds,
    replacing duplicated logic across ft_gadget_experiment and phase_orchestrator.
    
    Parameters
    ----------
    circuit : stim.Circuit
        Circuit to emit into.
    builders : List[BaseStabilizerRoundBuilder]
        Active round builders (surviving blocks only).
    schedules : Dict[str, BlockSchedule]
        Per-block scheduling decisions.
    num_rounds : int
        Number of rounds to emit.
    parallel : bool
        If True, use parallel extraction.
    stab_type : Optional[StabilizerBasis]
        Which stabilizers to measure (for hybrid decoding).
    crossing_config : Optional[CrossingDetectorConfig]
        If provided along with pre_gadget_meas, crossing detectors are
        emitted after the first round.
    pre_gadget_meas : Optional[Dict]
        Pre-gadget measurement indices for crossing detector emission.
    destroyed_blocks : Optional[Set[str]]
        Blocks destroyed during gadget (for crossing detector context).
    all_builders : Optional[List[BaseStabilizerRoundBuilder]]
        All builders (including destroyed blocks) needed for crossing detector
        emission. If None, uses builders.
    """
    if not builders or num_rounds == 0:
        return
    
    emitter = MemoryRoundEmitter(builders)
    
    for round_idx in range(num_rounds):
        # Extract per-round schedule
        round_info: Dict[str, "RoundSchedule"] = {}
        for builder in builders:
            block_sched = schedules[builder.block_name]
            round_info[builder.block_name] = block_sched.round_schedules[round_idx]
        
        # On the first post-gate round, suppress auto-detectors for crossing
        emit_crossing = (round_idx == 0 and crossing_config is not None
                         and pre_gadget_meas is not None)
        emit_auto_detectors = not emit_crossing
        
        emitter.emit_round(
            circuit, round_info,
            parallel=parallel,
            emit_detectors=emit_auto_detectors,
            stab_type=stab_type,
        )
        
        # Emit crossing detectors after the first post-gate round
        if emit_crossing:
            crossing_builders = all_builders if all_builders is not None else builders
            emit_crossing_detectors(
                circuit,
                crossing_builders,
                pre_gadget_meas,
                crossing_config,
                destroyed_blocks or set(),
            )
