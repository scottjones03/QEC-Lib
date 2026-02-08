# src/qectostim/experiments/phase_orchestrator.py
"""
Phase Orchestrator for FT Gadget Experiments.

This module extracts the phase execution logic from FaultTolerantGadgetExperiment
into a dedicated orchestrator. The orchestrator handles:

1. Iterating through gadget phases via emit_next_phase()
2. Handling PhaseResult types (GATE, MEASUREMENT, PREPARATION)
3. Applying stabilizer transforms between phases
4. Managing Pauli frame updates from measurements
5. Emitting crossing detectors around transversal gates

The key insight is that phase orchestration is independent of:
- Specific gadget types (teleportation, transversal, surgery)
- Memory round emission (handled by MemoryRoundEmitter)
- Detector emission (handled by detector_emission module)

By centralizing this logic, we enable:
- Clean separation of concerns from the experiment class
- Consistent phase handling across gadget types
- Easy addition of new gadget types without modifying orchestration
- Foundation for gadget chaining

Integration with Other Modules:
------------------------------
- scheduling.py: Used to determine stabilizer ordering for determinism
- detector_tracking.py: Used for anchor/temporal/boundary detector emission
- pauli_frame.py: Used for classical correction tracking
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union, TYPE_CHECKING

import stim

from qectostim.experiments.detector_tracking import GadgetChainState
from qectostim.gadgets.pauli_frame import MultiBlockPauliTracker

if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds import DetectorContext
    from qectostim.experiments.stabilizer_rounds.base import BaseStabilizerRoundBuilder
    from qectostim.gadgets.base import (
        Gadget,
        PhaseResult,
        PhaseType,
        StabilizerTransform,
        FrameUpdate,
        PreparationConfig,
    )
    from qectostim.gadgets.layout import QubitAllocation


@dataclass
class PhaseExecutionResult:
    """
    Result of executing all gadget phases.
    
    Attributes
    ----------
    destroyed_blocks : Set[str]
        Blocks that were destroyed (measured) during phase execution.
    transform_applied : bool
        Whether a stabilizer transform was applied.
    total_measurements : int
        Total measurements emitted during phase execution.
    final_phase_result : Optional[PhaseResult]
        The result from the final phase.
    pre_gate_meas : Dict[str, Dict[str, List[int]]]
        Pre-gate measurement indices (for crossing detectors).
    post_gate_meas : Dict[str, Dict[str, List[int]]]
        Post-gate measurement indices (for crossing detectors).
    crossing_detectors_emitted : int
        Number of crossing detectors emitted.
    crossing_handled : bool
        Whether crossing detectors were handled by inter-phase rounds.
        If False, the post-gadget memory must handle them.
    pauli_frame : Optional[Dict[str, Any]]
        Serialized Pauli frame state for gadget chaining.
    """
    destroyed_blocks: Set[str] = field(default_factory=set)
    transform_applied: bool = False
    total_measurements: int = 0
    final_phase_result: Optional["PhaseResult"] = None
    pre_gate_meas: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    post_gate_meas: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    crossing_detectors_emitted: int = 0
    crossing_handled: bool = False
    pauli_frame: Optional[Dict[str, Any]] = None
    destroyed_block_meas_starts: Dict[str, int] = field(default_factory=dict)

    def to_chain_state(self) -> GadgetChainState:
        """
        Convert to GadgetChainState for gadget chaining.
        
        Captures the essential information from this completed gadget
        that's needed to emit inter-gadget crossing detectors when the
        next gadget begins.
        
        Returns
        -------
        GadgetChainState
            State needed for chaining to the next gadget.
        """
        # Get the crossing config from the final phase if available
        crossing_config = None
        if self.final_phase_result is not None:
            from qectostim.gadgets.base import PhaseType
            if self.final_phase_result.phase_type == PhaseType.GATE:
                # The outgoing correlations are the post-gate measurements
                pass  # crossing_config determined by next gadget
        
        return GadgetChainState(
            final_syndrome_meas=self.post_gate_meas,
            outgoing_stabilizer_correlations=crossing_config,
            measurement_index=self.total_measurements,
            pauli_frame_state=self.pauli_frame,
        )


class PhaseOrchestrator:
    """
    Orchestrator for executing gadget phases.
    
    Handles the phase execution loop from FaultTolerantGadgetExperiment,
    including:
    - Calling gadget.emit_next_phase() repeatedly
    - Processing PhaseResult from each phase
    - Applying stabilizer transforms
    - Managing Pauli frame updates
    - Tracking destroyed blocks
    
    This class extracts and centralizes the logic from the to_stim()
    phase loop in FaultTolerantGadgetExperiment.
    """
    
    def __init__(
        self,
        gadget: "Gadget",
        ctx: "DetectorContext",
        builders: List["BaseStabilizerRoundBuilder"],
        alloc_dict: Dict[str, Any],
        initial_frame: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the orchestrator.
        
        Parameters
        ----------
        gadget : Gadget
            The gadget to execute.
        ctx : DetectorContext
            Detector context for measurement tracking.
        builders : List[BaseStabilizerRoundBuilder]
            Round builders for stabilizer transform application.
        alloc_dict : Dict[str, Any]
            Allocation dictionary for block information.
        initial_frame : Optional[Dict[str, Any]]
            Serialized Pauli frame state from previous gadget.
            Used for gadget chaining.
        """
        from qectostim.gadgets.scheduling import StabilizerScheduler
        
        self.gadget = gadget
        self.ctx = ctx
        self.builders = builders
        self.alloc_dict = alloc_dict
        self._block_builders = {b.block_name: b for b in builders}
        
        # Initialize StabilizerScheduler for deterministic round ordering
        self._scheduler = StabilizerScheduler()
        
        # Initialize MultiBlockPauliTracker for frame tracking across blocks
        # Each block tracks 1 logical qubit (standard for code blocks)
        block_sizes = {
            block_name: 1
            for block_name in alloc_dict
            if block_name != "total"
        }
        if initial_frame is not None:
            self._pauli_tracker = MultiBlockPauliTracker.deserialize(initial_frame)
        else:
            self._pauli_tracker = MultiBlockPauliTracker(block_sizes)
    
    def execute_phases(
        self,
        circuit: stim.Circuit,
        unified_alloc: "QubitAllocation",
        use_hybrid_decoding: bool = False,
        pre_gadget_meas: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> PhaseExecutionResult:
        """
        Execute all phases of the gadget.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        unified_alloc : QubitAllocation
            Unified qubit allocation.
        use_hybrid_decoding : bool
            If True, emit hybrid decoding observables for teleportation.
        pre_gadget_meas : Optional[Dict]
            Pre-gadget measurement indices per block for crossing detector
            emission. When provided, crossing detectors are emitted after
            the first post-gate inter-phase round (correct timing).
            
        Returns
        -------
        PhaseExecutionResult
            Result of phase execution.
        """
        from qectostim.gadgets.base import PhaseType
        from qectostim.experiments.stabilizer_rounds import StabilizerBasis
        from qectostim.experiments.observable import get_logical_support
        
        result = PhaseExecutionResult()
        
        # Reset gadget phase counter
        self.gadget.reset_phases()
        
        # Track prep measurement start for hybrid decoding
        prep_meas_start = None
        
        # Execute all phases
        for phase_idx in range(self.gadget.num_phases):
            phase_result = self.gadget.emit_next_phase(circuit, unified_alloc, self.ctx)
            
            # Track measured/destroyed blocks from BOTH fields
            if phase_result.measured_blocks:
                result.destroyed_blocks.update(phase_result.measured_blocks)
            if phase_result.destroyed_blocks:
                result.destroyed_blocks.update(phase_result.destroyed_blocks)
            
            # Track measurements
            meas_start = None
            if phase_result.measurement_count > 0:
                meas_start = self.ctx.add_measurement(phase_result.measurement_count)
                result.total_measurements += phase_result.measurement_count
                
                # Track measurement starts for destroyed blocks so boundary
                # detectors can be emitted for them in _emit_final_measurement.
                # E.g. teleportation Phase 3 emits MX on data before destroying it.
                if phase_result.destroyed_blocks:
                    for block_name in phase_result.destroyed_blocks:
                        result.destroyed_block_meas_starts[block_name] = meas_start
                
                # For hybrid decoding, track prep measurement start
                if use_hybrid_decoding and phase_result.phase_type == PhaseType.PREPARATION:
                    prep_meas_start = meas_start
                    self.gadget.set_prep_meas_start(meas_start)
                
                # For hybrid decoding with teleportation measurement,
                # emit OBSERVABLE_0 for data Z_L
                if (use_hybrid_decoding and
                    phase_result.phase_type == PhaseType.MEASUREMENT and
                    phase_result.pauli_frame_update is not None):
                    # Check teleport flag on typed FrameUpdate or legacy dict
                    fu = phase_result.get_frame_update()
                    if fu is not None and fu.teleport:
                        self._emit_hybrid_observable(
                            circuit, phase_result, meas_start
                        )
            
            # Apply stabilizer transform
            if phase_result.stabilizer_transform is not None:
                self._apply_transform(
                    phase_result.stabilizer_transform,
                    result.destroyed_blocks,
                )
                result.transform_applied = True
            
            # Propagate Pauli frame through gate operation
            if phase_result.phase_type == PhaseType.GATE:
                self._propagate_frame_through_gate()
            
            # Process Pauli frame updates
            if phase_result.pauli_frame_update is not None:
                self._process_frame_update(phase_result.pauli_frame_update, meas_start)
            
            # Handle inter-phase stabilizer rounds if needed
            if phase_result.needs_stabilizer_rounds > 0 and not phase_result.is_final:
                # After the GATE phase, pass pre_gadget_meas so crossing
                # detectors are emitted after the first post-gate round
                gate_pre_meas = None
                if phase_result.phase_type == PhaseType.GATE and pre_gadget_meas is not None:
                    gate_pre_meas = pre_gadget_meas
                self._emit_inter_phase_rounds(
                    circuit,
                    phase_result.needs_stabilizer_rounds,
                    result.destroyed_blocks,
                    pre_gadget_meas=gate_pre_meas,
                )
                if gate_pre_meas is not None:
                    result.crossing_handled = True
            
            # Store final result
            if phase_result.is_final:
                result.final_phase_result = phase_result
                break
        
        # Apply overall transform if not already applied
        if not result.transform_applied:
            overall_transform = self.gadget.get_stabilizer_transform()
            if (overall_transform.clear_history or 
                overall_transform.swap_xz or 
                overall_transform.skip_first_round):
                self._apply_transform(overall_transform, result.destroyed_blocks)
        
        # Apply observable transform for the gate
        self._apply_observable_transform()
        
        # Serialize Pauli frame state for gadget chaining
        result.pauli_frame = self._pauli_tracker.serialize()
        
        return result
    
    def _propagate_frame_through_gate(self) -> None:
        """
        Propagate Pauli frame through the gadget's transversal gate.
        
        Reads the gadget's gate_name property and calls the corresponding
        propagation on MultiBlockPauliTracker:
        - "CZ" → propagate_inter_block_cz(block_0, block_1)
        - "CNOT" → propagate_inter_block_cnot(block_0, block_1)
        - "H" and single-qubit gates → propagate_h on each block
        
        This ensures the classical Pauli frame is correctly updated
        through gate phases, which is essential for gadget chaining.
        """
        gate = self.gadget.gate_name
        block_names = [b for b in self.alloc_dict if b != "total"]
        
        if gate == "CZ" and len(block_names) >= 2:
            self._pauli_tracker.propagate_inter_block_cz(
                block1=block_names[0],
                block2=block_names[1],
            )
        elif gate == "CNOT" and len(block_names) >= 2:
            self._pauli_tracker.propagate_inter_block_cnot(
                control_block=block_names[0],
                target_block=block_names[1],
            )
        elif gate in ("H", "S", "T", "X", "Y", "Z"):
            # Single-qubit gate: propagate on each block's logical qubit
            for block_name in block_names:
                tracker = self._pauli_tracker.get_tracker(block_name)
                if tracker is not None:
                    if gate == "H":
                        tracker.propagate_h(0)
                    elif gate == "S":
                        tracker.propagate_s(0)
                    # X, Y, Z are Pauli gates — they commute with frame
    
    def _emit_hybrid_observable(
        self,
        circuit: stim.Circuit,
        phase_result: "PhaseResult",
        meas_start: int,
    ) -> None:
        """
        Emit OBSERVABLE_0 for hybrid decoding teleportation.
        
        This emits the data Z_L measurement for OBSERVABLE_0.
        """
        from qectostim.experiments.observable import get_logical_support
        from qectostim.gadgets.base import blocks_match
        
        # Get data block code for Z_L support
        data_code = None
        data_data_start = None
        for block_name, block_info in self.alloc_dict.items():
            if block_name == "total":
                continue
            # Use blocks_match() for flexible block name matching
            if blocks_match(block_name, "block_0"):
                data_code = block_info.get("code")
                data_range = block_info.get("data", (0, 0))
                data_data_start = data_range[0] if isinstance(data_range, tuple) else 0
                break
        
        if data_code is None:
            return
        
        z_support = get_logical_support(data_code, "Z", 0)
        observable_0_meas = []
        for local_idx in z_support:
            if phase_result.measurement_qubits:
                global_qubit_idx = data_data_start + local_idx
                if global_qubit_idx in phase_result.measurement_qubits:
                    meas_offset = phase_result.measurement_qubits.index(global_qubit_idx)
                    observable_0_meas.append(meas_start + meas_offset)
        
        if observable_0_meas:
            lookbacks = [idx - self.ctx.measurement_index for idx in observable_0_meas]
            targets = [stim.target_rec(lb) for lb in lookbacks]
            circuit.append("OBSERVABLE_INCLUDE", targets, 0)
    
    def _apply_observable_transform(self) -> None:
        """
        Apply observable transform for the gate.
        Uses gadget.gate_name property (defined on Gadget base class).
        """
        self.ctx.update_for_gate(self.gadget.gate_name)
    
    def _apply_transform(
        self,
        transform: "StabilizerTransform",
        destroyed_blocks: Set[str],
    ) -> None:
        """
        Apply stabilizer transform to surviving blocks.
        
        Parameters
        ----------
        transform : StabilizerTransform
            The transform to apply.
        destroyed_blocks : Set[str]
            Blocks to skip (already destroyed).
        """
        if not (transform.clear_history or transform.swap_xz or transform.skip_first_round):
            return
        
        # Clear history in detector context
        for block_name in self.alloc_dict:
            if block_name != "total" and block_name not in destroyed_blocks:
                self.ctx.clear_stabilizer_history(
                    block_name=block_name,
                    swap_xz=transform.swap_xz,
                )
        
        # Reset builder history
        for builder in self.builders:
            if builder.block_name not in destroyed_blocks:
                builder.reset_stabilizer_history(
                    swap_xz=transform.swap_xz,
                    skip_first_round=transform.skip_first_round,
                )
    
    def _process_frame_update(
        self,
        frame_update: "Union[FrameUpdate, Dict[str, Any]]",
        meas_base: Optional[int],
    ) -> None:
        """
        Process Pauli frame update from measurement phase.
        
        Updates both the detector context (for observable construction)
        and the MultiBlockPauliTracker (for classical correction tracking).
        
        For teleportation, this records the frame transfer from source
        to target block in both tracking systems.
        
        Parameters
        ----------
        frame_update : Union[FrameUpdate, Dict[str, Any]]
            Frame update information. Can be a typed FrameUpdate dataclass
            (preferred) or legacy Dict for backwards compatibility.
        meas_base : Optional[int]
            Base measurement index for relative index conversion.
        """
        from qectostim.gadgets.base import FrameUpdate
        
        # Convert to typed FrameUpdate if dict
        if isinstance(frame_update, dict):
            fu = FrameUpdate.from_dict(frame_update)
        else:
            fu = frame_update
        
        # Convert relative indices to absolute
        if meas_base is not None:
            x_meas_abs = [meas_base + i for i in fu.x_meas]
            z_meas_abs = [meas_base + i for i in fu.z_meas]
        else:
            x_meas_abs = fu.x_meas
            z_meas_abs = fu.z_meas
        
        # Update detector context (existing behavior)
        if fu.teleport:
            self.ctx.record_frame_from_teleportation(fu.block_name, x_meas_abs, z_meas_abs)
        else:
            self.ctx.record_projection_frame(fu.block_name, x_meas_abs, z_meas_abs)
        
        # Also propagate through MultiBlockPauliTracker
        # For teleportation: transfer frame from source to target block
        if fu.teleport and fu.source_block:
            self._pauli_tracker.process_teleportation(
                source_block=fu.source_block,
                target_block=fu.block_name,
                x_measurement=1 if x_meas_abs else 0,
                z_measurement=1 if z_meas_abs else 0,
            )
    
    def _emit_inter_phase_rounds(
        self,
        circuit: stim.Circuit,
        num_rounds: int,
        destroyed_blocks: Set[str],
        pre_gadget_meas: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Emit stabilizer rounds between gadget phases with proper scheduling.
        
        Delegates to emit_scheduled_rounds() from detector_emission for
        the actual round emission, computing schedules from prep config.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        num_rounds : int
            Number of rounds to emit.
        destroyed_blocks : Set[str]
            Blocks to skip (already measured out).
        pre_gadget_meas : Optional[Dict]
            Pre-gadget measurement indices for crossing detector emission.
            When provided, crossing detectors are emitted after the first round.
        """
        from qectostim.experiments.detector_emission import emit_scheduled_rounds
        from qectostim.gadgets.scheduling import BlockSchedule
        
        active_builders = [
            b for b in self.builders
            if b.block_name not in destroyed_blocks
        ]
        
        if not active_builders or num_rounds == 0:
            return
        
        # Get prep config from gadget to determine block prep basis
        prep_config = self.gadget.get_preparation_config("0")
        default_ordering = self.gadget.get_default_stabilizer_ordering()
        
        # Compute schedules for each active block using the scheduler
        schedules: Dict[str, BlockSchedule] = {}
        for builder in active_builders:
            prep_basis = "0"  # default
            block_config = prep_config.get_block_config(builder.block_name)
            if block_config is not None:
                prep_basis = block_config.initial_state
            
            schedule = self._scheduler.compute_block_schedule(
                block_name=builder.block_name,
                prep_basis=prep_basis,
                meas_basis="Z",  # Default to Z-basis meas for inter-phase
                num_rounds=num_rounds,
                default_ordering=default_ordering,
            )
            schedules[builder.block_name] = schedule
        
        # Get crossing detector config if we have pre-gadget measurements
        crossing_config = None
        if pre_gadget_meas is not None:
            crossing_config = self.gadget.get_crossing_detector_config()
        
        # Delegate to emit_scheduled_rounds — eliminates duplicated round loop
        emit_scheduled_rounds(
            circuit, active_builders, schedules, num_rounds,
            parallel=False,  # Inter-phase rounds use sequential-per-builder emission
            crossing_config=crossing_config,
            pre_gadget_meas=pre_gadget_meas,
            destroyed_blocks=destroyed_blocks,
            all_builders=self.builders,
        )


def execute_gadget_phases(
    gadget: "Gadget",
    circuit: stim.Circuit,
    unified_alloc: "QubitAllocation",
    ctx: "DetectorContext",
    builders: List["BaseStabilizerRoundBuilder"],
    alloc_dict: Dict[str, Any],
    use_hybrid_decoding: bool = False,
    initial_frame: Optional[Dict[str, Any]] = None,
    pre_gadget_meas: Optional[Dict[str, Dict[str, Any]]] = None,
) -> PhaseExecutionResult:
    """
    Convenience function to execute gadget phases.
    
    Parameters
    ----------
    gadget : Gadget
        The gadget to execute.
    circuit : stim.Circuit
        Circuit to emit into.
    unified_alloc : QubitAllocation
        Unified qubit allocation.
    ctx : DetectorContext
        Detector context.
    builders : List[BaseStabilizerRoundBuilder]
        Round builders.
    alloc_dict : Dict[str, Any]
        Allocation dictionary.
    use_hybrid_decoding : bool
        If True, emit hybrid decoding observables for teleportation.
    initial_frame : Optional[Dict[str, Any]]
        Serialized Pauli frame state from previous gadget for chaining.
    pre_gadget_meas : Optional[Dict]
        Pre-gadget measurement indices for crossing detector emission.
        When provided, crossing detectors are emitted at the correct time
        (after the first post-gate inter-phase round).
        
    Returns
    -------
    PhaseExecutionResult
        Result of phase execution (includes serialized pauli_frame for chaining).
    """
    orchestrator = PhaseOrchestrator(gadget, ctx, builders, alloc_dict, initial_frame)
    return orchestrator.execute_phases(circuit, unified_alloc, use_hybrid_decoding, pre_gadget_meas)
