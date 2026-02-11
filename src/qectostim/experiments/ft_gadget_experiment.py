# src/qectostim/experiments/ft_gadget_experiment.py
"""
Fault-Tolerant Gadget Experiment.

Implements the TQEC-style pattern for measuring logical gate error rates:

    Memory → Gadget → Memory → Measure

This ensures proper fault-tolerance by:
1. Running stabilizer rounds before the gadget (establish baseline)
2. Applying the logical gate via the gadget
3. Running stabilizer rounds after the gadget (verify stabilizer continuity)
4. Measuring all data qubits (extract logical result)

The detector network spans all phases, enabling decoding of the full
experiment including the gadget operation.

Example usage:
    >>> from qectostim.experiments.ft_gadget_experiment import FaultTolerantGadgetExperiment
    >>> from qectostim.codes.surface import RotatedSurfaceCode
    >>> from qectostim.gadgets import TransversalHadamard
    >>> from qectostim.noise.models import UniformDepolarizing
    >>>
    >>> code = RotatedSurfaceCode(distance=3)
    >>> gadget = TransversalHadamard()
    >>> noise = UniformDepolarizing(p=0.001)
    >>> 
    >>> exp = FaultTolerantGadgetExperiment(
    ...     codes=[code],
    ...     gadget=gadget,
    ...     noise_model=noise,
    ...     num_rounds_before=3,
    ...     num_rounds_after=3,
    ... )
    >>> circuit = exp.to_stim()
"""
from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple, Set
from dataclasses import dataclass, field

import numpy as np
import stim

from qectostim.codes.abstract_code import Code
from qectostim.experiments.experiment import Experiment
from qectostim.noise.models import NoiseModel
from qectostim.gadgets.base import (
    Gadget, 
    GadgetMetadata, 
    PreparationConfig,
    StabilizerTransform,
)
from qectostim.gadgets.layout import QubitAllocation
from qectostim.experiments.stabilizer_rounds import (
    DetectorContext,
    CSSStabilizerRoundBuilder,
    GeneralStabilizerRoundBuilder,
    BaseStabilizerRoundBuilder,
    StabilizerBasis,
)
from qectostim.experiments.preparation import (
    emit_initial_states,
    get_qubits_to_skip_reset,
)
from qectostim.experiments.observable import emit_observable
from qectostim.gadgets.scheduling import (
    StabilizerScheduler,
    BlockSchedule,
)
from qectostim.experiments.detector_emission import (
    emit_boundary_detectors as emit_boundary_detectors_func,
    emit_scheduled_rounds,
)
from qectostim.experiments.detector_tracking import (
    get_stabilizer_supports,
    MeasurementRecord,
    DetectorCoverageResolver,
)
from qectostim.experiments.phase_orchestrator import (
    execute_gadget_phases,
    PhaseExecutionResult,
)


def validate_circuit_detectors(circuit: stim.Circuit) -> Tuple[bool, str]:
    """Validate that all detectors in a circuit are deterministic.
    
    A deterministic detector has an XOR of its referenced measurements that
    equals 0 for noiseless simulation. Non-deterministic detectors indicate
    bugs in the circuit construction (missing CNOTs, wrong stabilizer transforms, etc.)
    
    Parameters
    ----------
    circuit : stim.Circuit
        The circuit to validate.
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message) - is_valid is True if all detectors are deterministic.
    """
    try:
        # Try to generate the detector error model - this will fail if detectors are non-deterministic
        dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        return True, ""
    except ValueError as e:
        error_msg = str(e)
        if "non-deterministic" in error_msg.lower() or "detector" in error_msg.lower():
            return False, f"Non-deterministic detector(s) found: {error_msg}"
        # Re-raise other ValueError types
        raise


def apply_noise_to_circuit(circuit: stim.Circuit, noise_model: Optional[NoiseModel]) -> stim.Circuit:
    """Apply noise model to circuit if provided."""
    if noise_model is None:
        return circuit
    return noise_model.apply(circuit)


@dataclass
class FTGadgetExperimentResult:
    """
    Result of a fault-tolerant gadget experiment.
    
    Attributes:
        logical_error_rate: Measured logical error rate.
        num_shots: Number of shots sampled.
        num_errors: Number of logical errors observed.
        gadget_metadata: Metadata from the gadget.
        decoder_used: Name of decoder used.
    """
    logical_error_rate: float
    num_shots: int
    num_errors: int
    gadget_metadata: Optional[GadgetMetadata] = None
    decoder_used: str = "unknown"
    extra: Dict[str, Any] = field(default_factory=dict)


class FaultTolerantGadgetExperiment(Experiment):
    """
    Fault-tolerant experiment for measuring logical gate error rates.
    
    This experiment implements the TQEC pattern:
    
    1. **Initialization**: Reset all qubits
    2. **Pre-gadget memory**: num_rounds_before stabilizer rounds
       - Establishes baseline syndrome measurements
       - Creates detectors comparing consecutive rounds
    3. **Gadget execution**: Logical gate operation
       - Uses gadget.to_stim() which should emit internal detectors
    4. **Post-gadget memory**: num_rounds_after stabilizer rounds  
       - Verifies stabilizer continuity after gate
       - Creates detectors linking back to pre-gadget rounds
    5. **Final measurement**: Measure all data qubits
       - Creates space-like detectors
       - Emits OBSERVABLE_INCLUDE for logical operator
    
    The detector graph spans all phases, enabling unified decoding.
    
    Parameters
    ----------
    codes : List[Code]
        Code(s) to apply gadget to.
    gadget : Gadget
        The logical gate gadget to test.
    noise_model : NoiseModel
        Noise model for the experiment.
    num_rounds_before : int
        Number of stabilizer rounds before gadget.
    num_rounds_after : int
        Number of stabilizer rounds after gadget.
    auto_detectors : bool or None
        If True, enable automatic detector/observable emission via flow
        matching (replaces all manual DETECTOR/OBSERVABLE_INCLUDE with
        auto-discovered ones).  If None (default), defers to
        ``gadget.use_auto_detectors()``.  Explicit True/False overrides
        the gadget's preference.
    metadata : Optional[Dict]
        Additional experiment metadata.
    """
    
    def __init__(
        self,
        codes: List[Code],
        gadget: Gadget,
        noise_model: NoiseModel,
        num_rounds_before: int = 3,
        num_rounds_after: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
        auto_detectors: Optional[bool] = None,
    ):
        # Use first code as primary for base class
        super().__init__(codes[0], noise_model, metadata)
        
        self.codes = codes
        self.gadget = gadget
        self.num_rounds_before = num_rounds_before
        self.num_rounds_after = num_rounds_after
        # Infer auto_detectors from gadget when not explicitly set
        if auto_detectors is None:
            self.auto_detectors = gadget.use_auto_detectors()
        else:
            self.auto_detectors = auto_detectors
        
        # Check for placeholder logicals (QLDPC codes without proper logical operators)
        self._validate_codes(codes)
        
        # Validate gadget-code compatibility (CSS requirements, etc.)
        gadget.validate_codes(codes)
        
        # Initialize stabilizer scheduler and detector coverage resolver
        self._scheduler = StabilizerScheduler()
        self._coverage_resolver = DetectorCoverageResolver(self._scheduler)
        
        # Cached state
        self._ctx: Optional[DetectorContext] = None
        self._builders: List[BaseStabilizerRoundBuilder] = []
        self._qubit_allocation: Optional[Dict[str, Any]] = None
        self._prep_config: Optional[PreparationConfig] = None  # Set in to_stim()
        self._use_rx_prep: bool = True  # Use RX instead of H for |+⟩ prep
    
    def _validate_codes(self, codes: List[Code]) -> None:
        """
        Validate that codes are compatible with FT gadget experiments.
        
        Raises
        ------
        ValueError
            If a code has placeholder logical operators or is otherwise unsupported.
        """
        for i, code in enumerate(codes):
            # Check for placeholder logicals via get_ft_gadget_config
            # All Code subclasses implement get_ft_gadget_config() (defined in Code ABC)
            config = code.get_ft_gadget_config()
            extra = config.extra or {}
            
            if extra.get('has_placeholder_logicals', False):
                code_name = code.name or code.metadata.get('name', f'codes[{i}]')
                reason = extra.get('unsupported_reason', 'placeholder logical operators')
                raise ValueError(
                    f"Code '{code_name}' is not supported for FT gadget experiments: {reason}. "
                    f"QLDPC codes require proper logical operator computation. "
                    f"Consider using a code with explicit logical operators like RotatedSurfaceCode or SteaneCode."
                )
        
    @staticmethod
    def _get_stabilizer_counts(code: Code) -> Tuple[int, int]:
        """
        Get X and Z stabilizer counts using the Code interface.
        
        Uses code.is_css / CSSCode.hx/hz for CSS codes, or falls back
        to get_stabilizer_supports() from detector_tracking for others.
        
        Returns
        -------
        Tuple[int, int]
            (num_x_stabilizers, num_z_stabilizers)
        """
        if code.is_css:
            css_code = code.as_css()
            if css_code is not None:
                return css_code.hx.shape[0], css_code.hz.shape[0]
        
        # Use detector_tracking's canonical utility
        try:
            nx = len(get_stabilizer_supports(code, "X"))
            nz = len(get_stabilizer_supports(code, "Z"))
            return nx, nz
        except NotImplementedError:
            return 0, 0

    def _allocation_to_dict(self, unified: QubitAllocation) -> Dict[str, Any]:
        """
        Convert QubitAllocation to dict format for backwards compatibility.
        
        Used during the transition period while internal methods are being
        refactored to use QubitAllocation directly.
        """
        alloc = {}
        
        for block_name, block in unified.blocks.items():
            # Reserve projection ancilla index (after z ancillas)
            proj_anc_idx = block.z_anc_start + block.z_anc_count
            
            alloc[block_name] = {
                "code": block.code,
                "data": (block.data_start, block.data_count),
                "x_anc": (block.x_anc_start, block.x_anc_count),
                "z_anc": (block.z_anc_start, block.z_anc_count),
                "proj_anc": proj_anc_idx,
            }
        
        # GOAL 10: Use QubitAllocation.total_qubits as source of truth
        alloc["total"] = unified.total_qubits
        return alloc
    
    def _emit_qubit_coords(
        self,
        circuit: stim.Circuit,
        unified_alloc: QubitAllocation,
    ) -> None:
        """
        Emit QUBIT_COORDS for all qubits.
        
        Delegates to QubitAllocation.emit_qubit_coords() which uses
        topology-aware placement from code coordinate metadata.
        """
        unified_alloc.emit_qubit_coords(circuit)
    
    def _create_builders(
        self,
        alloc: Dict[str, Any],
        ctx: DetectorContext,
        unified_alloc: Optional["QubitAllocation"] = None,
    ) -> List[BaseStabilizerRoundBuilder]:
        """Create appropriate StabilizerRoundBuilder for each code block.
        
        Selects the builder type based on code properties:
        - CSSCode with hx/hz: CSSStabilizerRoundBuilder
        - StabilizerCode with stabilizer_matrix: GeneralStabilizerRoundBuilder
        
        X Stabilizer Mode
        -----------------
        The gadget determines which gate to use for X stabilizer measurement
        via get_x_stabilizer_mode(). This is CRITICAL for teleportation gadgets:
        
        - Most gadgets use "cz" (H-CZ-H circuit) which is symmetric and efficient
        - Teleportation gadgets MUST use "cx" (H-CX-H circuit) to match ground truth
        
        The difference is in backward Pauli propagation through the syndrome circuit:
        - CZ: X_syndrome → X_syndrome ⊗ Z_data (couples to data Z, breaks determinism)
        - CX: X_syndrome → X_syndrome (stays local, preserves determinism)
        """
        builders = []
        
        # Get X stabilizer mode from gadget (critical for teleportation)
        x_stabilizer_mode = self.gadget.get_x_stabilizer_mode()
        
        for block_name, block_info in alloc.items():
            if block_name == "total":
                continue
            
            code = block_info["code"]
            data_start, _ = block_info["data"]
            x_anc_start, _ = block_info["x_anc"]
            
            # Get block spatial offset for detector coordinates
            block_coord_offset = None
            if unified_alloc is not None:
                block_alloc = unified_alloc.blocks.get(block_name)
                if block_alloc is not None:
                    block_coord_offset = block_alloc.offset
            
            # Use per-block measurement basis if available from MeasurementConfig,
            # otherwise fall back to gadget's measurement_basis.
            # This is important for gadgets like TransversalCNOT where each block
            # may be measured in a different basis.
            meas_config = self.gadget.get_measurement_config()
            block_meas_basis = meas_config.block_bases.get(
                block_name, self.gadget.get_measurement_basis()
            )
            
            # Determine which builder to use based on code properties:
            # - code.is_css → CSSStabilizerRoundBuilder (uses hx/hz)
            # - code.is_stabilizer → GeneralStabilizerRoundBuilder (uses stabilizer_matrix)
            # - otherwise → CSS builder fallback (produces 0 detectors)
            if code.is_css:
                builder = CSSStabilizerRoundBuilder(
                    code=code,
                    ctx=ctx,
                    block_name=block_name,
                    data_offset=data_start,
                    ancilla_offset=x_anc_start,
                    measurement_basis=block_meas_basis,
                    x_stabilizer_mode=x_stabilizer_mode,
                    coord_offset=block_coord_offset,
                )
            elif code.is_stabilizer:
                builder = GeneralStabilizerRoundBuilder(
                    code=code,
                    ctx=ctx,
                    block_name=block_name,
                    data_offset=data_start,
                    ancilla_offset=x_anc_start,
                    measurement_basis=block_meas_basis,
                    coord_offset=block_coord_offset,
                )
            else:
                # Code base class only: CSS builder fallback
                builder = CSSStabilizerRoundBuilder(
                    code=code,
                    ctx=ctx,
                    block_name=block_name,
                    data_offset=data_start,
                    ancilla_offset=x_anc_start,
                    measurement_basis=block_meas_basis,
                    x_stabilizer_mode=x_stabilizer_mode,
                    coord_offset=block_coord_offset,
                )
            
            builders.append(builder)
        
        return builders

    def _emit_prepare_logical_states(
        self,
        circuit: stim.Circuit,
        builders: List[BaseStabilizerRoundBuilder],
    ) -> None:
        """
        Prepare logical states for all code blocks.
        
        Uses PreparationConfig from the gadget to determine:
        - Initial state per block (|0⟩ or |+⟩)
        - Which stabilizers are deterministic (for anchor detectors)
        
        The initial state determines which stabilizers can emit anchor detectors:
        - |0⟩_L preparation → Z stabilizers deterministic → Z anchors
        - |+⟩_L preparation → X stabilizers deterministic → X anchors
        
        This method emits:
        1. Initial state preparation (H for |+⟩ blocks)
        2. DOES NOT emit first syndrome round - that's done in _emit_pre_gadget_memory
        
        NOTE: prep_config is computed and stored in to_stim() before the reset,
        so we use the stored config here.
        """
        # Use PreparationConfig already computed in to_stim()
        if self._prep_config is None:
            # Compute if not already done
            initial_state = "+" if self.gadget.get_measurement_basis() == "X" else "0"
            self._prep_config = self.gadget.get_preparation_config(initial_state)
        prep_config = self._prep_config
        
        # Skip if gadget handles all preparation
        if self.gadget.should_skip_state_preparation():
            circuit.append("TICK")
            return
        
        # Emit initial states (RX for |+⟩, nothing for |0⟩)
        # Build block_data_qubits mapping
        block_data_qubits: Dict[str, List[int]] = {}
        for builder in builders:
            block_name = builder.block_name
            if block_name in prep_config.blocks:
                block_config = prep_config.blocks[block_name]
                if not block_config.skip_experiment_prep:
                    block_data_qubits[block_name] = builder.data_qubits
        
        # Use preparation module to emit initial states
        # use_rx=True means we use RX instead of H for |+⟩ preparation
        use_rx = self._use_rx_prep
        emit_initial_states(circuit, prep_config, block_data_qubits, use_rx=use_rx)
    
    def _compute_block_schedules(
        self,
        builders: List[BaseStabilizerRoundBuilder],
        num_rounds: int,
    ) -> Dict[str, BlockSchedule]:
        """Compute scheduling for all active blocks.
        
        Delegates to DetectorCoverageResolver for the determinism logic.
        Uses per-block measurement basis from MeasurementConfig when available.
        """
        # Get per-block measurement bases from gadget's MeasurementConfig
        meas_config = self.gadget.get_measurement_config()
        block_meas_bases = meas_config.block_bases if meas_config.block_bases else None
        
        return self._coverage_resolver.compute_pre_gadget_coverage(
            builders=builders,
            prep_config=self._prep_config,
            meas_basis=self.gadget.get_measurement_basis(),
            num_rounds=num_rounds,
            default_ordering=self.gadget.get_default_stabilizer_ordering(),
            block_meas_bases=block_meas_bases,
        )

    def _compute_post_gadget_schedules(
        self,
        builders: List[BaseStabilizerRoundBuilder],
        num_rounds: int,
        stab_transform: StabilizerTransform,
    ) -> Dict[str, BlockSchedule]:
        """Compute scheduling for post-gadget rounds on surviving blocks.
        
        Delegates to DetectorCoverageResolver which handles swap_xz
        transformation for the effective prep basis.
        Uses per-block measurement basis from MeasurementConfig when available.
        """
        # Get per-block measurement bases from gadget's MeasurementConfig
        meas_config = self.gadget.get_measurement_config()
        block_meas_bases = meas_config.block_bases if meas_config.block_bases else None
        
        return self._coverage_resolver.compute_post_gadget_coverage(
            builders=builders,
            prep_config=self._prep_config,
            meas_basis=self.gadget.get_measurement_basis(),
            num_rounds=num_rounds,
            stab_transform=stab_transform,
            default_ordering=self.gadget.get_default_stabilizer_ordering(),
            block_meas_bases=block_meas_bases,
        )

    def _emit_pre_gadget_memory(
        self,
        circuit: stim.Circuit,
        builders: List[BaseStabilizerRoundBuilder],
    ) -> None:
        """
        Emit pre-gadget stabilizer rounds with anchor detectors.
        
        Uses StabilizerScheduler.compute_block_schedule() to determine
        per-round anchor flags and basis ordering for each block.
        
        Supports both sequential and parallel extraction:
        - Sequential: each builder emits a full round (emit_round)
        - Parallel: all builders share a round (emit_z_layer/emit_x_layer)
        
        The scheduling decisions are identical — only the emission API differs.
        
        First round emits anchor detectors for the deterministic basis:
        - |0⟩ prep → Z stabilizers deterministic → emit_z_anchors=True
        - |+⟩ prep → X stabilizers deterministic → emit_x_anchors=True
        
        Subsequent rounds emit normal temporal detectors.
        """
        blocks_to_skip = self.gadget.get_blocks_to_skip_pre_rounds()
        parallel = self.gadget.requires_parallel_extraction()
        
        # Filter to active builders
        active_builders = [
            b for b in builders
            if b.block_name not in blocks_to_skip
        ]
        
        if not active_builders:
            return
        
        # Compute schedules and delegate to emit_scheduled_rounds
        schedules = self._compute_block_schedules(active_builders, self.num_rounds_before)
        emit_scheduled_rounds(
            circuit, active_builders, schedules, self.num_rounds_before,
            parallel=parallel,
        )
    
    def _emit_post_gadget_memory(
        self,
        circuit: stim.Circuit,
        builders: List[BaseStabilizerRoundBuilder],
        destroyed_blocks: Optional[Set[str]] = None,
        pre_gadget_meas: Optional[Dict[str, Dict[str, List[int]]]] = None,
    ) -> None:
        """Emit post-gadget stabilizer rounds with crossing detectors.
        
        Delegates to emit_scheduled_rounds() from detector_emission for the
        actual round emission, handling ancilla reset and schedule computation.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        builders : List[BaseStabilizerRoundBuilder]
            Round builders for each code block.
        destroyed_blocks : Optional[Set[str]]
            Blocks that are destroyed (measured) during the gadget.
        pre_gadget_meas : Optional[Dict[str, Dict[str, List[int]]]]
            Pre-gadget measurement indices per block, keyed by block_name.
            If provided, crossing detectors will be emitted on first post-round.
        """
        if destroyed_blocks is None:
            destroyed_blocks = set()
        
        parallel = self.gadget.requires_parallel_extraction()
        use_hybrid = self.gadget.use_hybrid_decoding()
        blocks_to_skip_post = self.gadget.get_blocks_to_skip_post_rounds()
        
        # Filter to active (surviving) builders
        active_builders = [
            b for b in builders
            if b.block_name not in destroyed_blocks
            and b.block_name not in blocks_to_skip_post
        ]
        
        if not active_builders:
            return
        
        # Check if stabilizer transform cleared history (teleportation)
        stab_transform = self.gadget.get_stabilizer_transform()
        skip_ancilla_reset = stab_transform.clear_history
        
        # Reset ancillas for surviving blocks before first post-gadget round
        # Skip for teleportation-style transforms where history is cleared
        if not skip_ancilla_reset:
            for builder in active_builders:
                anc = builder.all_ancillas
                if builder.has_metachecks:
                    anc = anc + builder.meta_x_ancillas + builder.meta_z_ancillas
                if anc:
                    circuit.append("R", anc)
            if active_builders:
                circuit.append("TICK")
        
        # Get crossing detector config if we have pre-gadget measurements
        crossing_config = None
        if pre_gadget_meas is not None:
            crossing_config = self.gadget.get_crossing_detector_config()
        
        # Determine stabilizer type for hybrid decoding
        stab_type = StabilizerBasis.Z if use_hybrid else StabilizerBasis.BOTH
        
        # Compute post-gadget scheduling
        post_schedules = self._compute_post_gadget_schedules(
            active_builders, self.num_rounds_after, stab_transform
        )
        
        # Delegate to emit_scheduled_rounds for the actual emission
        emit_scheduled_rounds(
            circuit, active_builders, post_schedules, self.num_rounds_after,
            parallel=parallel, stab_type=stab_type,
            crossing_config=crossing_config,
            pre_gadget_meas=pre_gadget_meas,
            destroyed_blocks=destroyed_blocks,
            all_builders=builders,
        )

    def _emit_final_measurement(
        self,
        circuit: stim.Circuit,
        builders: List[BaseStabilizerRoundBuilder],
        alloc: Dict[str, Any],
        ctx: DetectorContext,
        destroyed_blocks: Optional[Set[str]] = None,
        destroyed_block_meas_starts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Emit final data measurement, boundary detectors, and observable."""
        if destroyed_blocks is None:
            destroyed_blocks = set()
        if destroyed_block_meas_starts is None:
            destroyed_block_meas_starts = {}

        measurement_config = self.gadget.get_measurement_config()
        if measurement_config.destroyed_blocks:
            destroyed_blocks = destroyed_blocks | measurement_config.destroyed_blocks

        # Phase A: Destructive data measurement
        surviving_alloc, meas_start = self._emit_destructive_data_measurement(
            circuit, alloc, ctx, measurement_config, destroyed_blocks,
        )
        if meas_start is None:
            return  # All blocks destroyed

        # Phase B: Boundary detectors
        self._emit_boundary_detectors(
            circuit, builders, surviving_alloc, meas_start,
            destroyed_blocks, destroyed_block_meas_starts,
        )

        # Phase C: Observable
        # Use full alloc (not just surviving_alloc) so that correlation terms
        # referencing destroyed blocks (e.g., data_block in teleportation) can
        # resolve their qubit→measurement mappings via destroyed_block_meas_starts.
        obs_config = self.gadget.get_observable_config()
        emit_observable(
            circuit=circuit,
            obs_config=obs_config,
            alloc=alloc,
            ctx=ctx,
            meas_start=meas_start,
            measurement_basis=self.gadget.get_measurement_basis(),
            gadget=self.gadget,
            destroyed_block_meas_starts=destroyed_block_meas_starts,
        )

    def _emit_destructive_data_measurement(
        self,
        circuit: stim.Circuit,
        alloc: Dict[str, Any],
        ctx: DetectorContext,
        measurement_config: Any,
        destroyed_blocks: Set[str],
    ) -> Tuple[Dict[str, Any], Optional[int]]:
        """Emit MX/M on surviving data qubits, one block at a time in block order.
        
        Measurements are emitted per-block in allocation order so that
        downstream helpers (_build_qubit_to_meas, _emit_boundary_detectors)
        can assume measurement indices follow block iteration order.
        
        Returns (surviving_alloc, meas_start) or (empty, None) if all destroyed.
        """
        surviving_alloc: Dict[str, Any] = {}
        block_qubit_basis: list = []  # [(block_name, qubits, basis), ...]

        for block_name, block_info in alloc.items():
            if block_name == "total" or block_name in destroyed_blocks:
                continue
            data_start, n = block_info["data"]
            block_qubits = list(range(data_start, data_start + n))
            surviving_alloc[block_name] = block_info
            block_basis = measurement_config.block_bases.get(
                block_name, self.gadget.get_measurement_basis()
            )
            block_qubit_basis.append((block_name, block_qubits, block_basis))

        total_data = sum(len(qs) for _, qs, _ in block_qubit_basis)
        if total_data == 0:
            return {}, None

        meas_start = ctx.add_measurement(total_data)
        
        # Emit measurements block-by-block in allocation order
        for _, block_qubits, basis in block_qubit_basis:
            if basis == "X":
                circuit.append("MX", block_qubits)
            else:
                circuit.append("M", block_qubits)

        return surviving_alloc, meas_start

    def _emit_boundary_detectors(
        self,
        circuit: stim.Circuit,
        builders: List[BaseStabilizerRoundBuilder],
        surviving_alloc: Dict[str, Any],
        meas_start: int,
        destroyed_blocks: Set[str],
        destroyed_block_meas_starts: Dict[str, int],
    ) -> None:
        """Emit space-like boundary detectors for surviving and destroyed blocks."""
        if not self.gadget.should_emit_space_like_detectors():
            return

        surviving_builders = [
            b for b in builders if b.block_name not in destroyed_blocks
        ]
        boundary_config = self.gadget.get_boundary_detector_config()

        # Surviving block offsets
        block_meas_offsets: Dict[str, int] = {}
        offset = meas_start
        for block_name, block_info in surviving_alloc.items():
            _, n = block_info["data"]
            block_meas_offsets[block_name] = offset
            offset += n

        emit_boundary_detectors_func(
            circuit, surviving_builders, boundary_config, block_meas_offsets,
        )

        # Destroyed blocks whose measurements were emitted by the gadget
        if destroyed_block_meas_starts:
            destroyed_builders = [
                b for b in builders
                if b.block_name in destroyed_block_meas_starts
            ]
            destroyed_offsets = {
                b.block_name: destroyed_block_meas_starts[b.block_name]
                for b in destroyed_builders
            }
            if destroyed_builders and destroyed_offsets:
                emit_boundary_detectors_func(
                    circuit, destroyed_builders, boundary_config, destroyed_offsets,
                )
    
    def to_stim(self) -> stim.Circuit:
        """
        Generate complete fault-tolerant Stim circuit.
        
        The circuit includes:
        - QUBIT_COORDS for visualization
        - Pre-gadget memory rounds with DETECTOR
        - Gadget circuit (from gadget.to_stim, but we use our own detector tracking)
        - Post-gadget memory rounds with DETECTOR
        - Final measurement with OBSERVABLE_INCLUDE
        
        Returns
        -------
        stim.Circuit
            Complete Stim circuit for the experiment.
        """
        circuit = stim.Circuit()
        
        # Initialize context
        ctx = DetectorContext()
        self._ctx = ctx
        
        # =====================================================================
        # GOAL 6 & 10: Use GadgetLayout.allocate_qubits() and QubitAllocation.total_qubits
        # =====================================================================
        # Get layout from gadget and use library's allocation
        layout = self.gadget.compute_layout(self.codes)
        unified_alloc = layout.allocate_qubits()
        
        # Store for later use by other methods
        self._unified_allocation = unified_alloc
        
        # Convert to dict format for backwards compatibility with internal methods
        # TODO: Refactor internal methods to use QubitAllocation directly
        alloc = self._allocation_to_dict(unified_alloc)
        self._qubit_allocation = alloc
        
        # =====================================================================
        # Initialize Pauli Frame Tracker
        # =====================================================================
        # Count the number of logical qubits (code blocks)
        num_logical = len(unified_alloc.blocks)
        if num_logical > 0:
            ctx.init_pauli_tracker(max(1, num_logical))
        
        # =====================================================================
        # Phase 1: QUBIT_COORDS and Reset
        # =====================================================================
        self._emit_qubit_coords(circuit, unified_alloc)
        
        # =====================================================================
        # Phase 1.5: Create builders BEFORE reset to know which qubits need H
        # =====================================================================
        # We need builders to know which data qubits belong to which block
        builders = self._create_builders(alloc, ctx, unified_alloc)
        self._builders = builders
        
        # =====================================================================
        # Phase 1.6: Compute preparation config and gate optimization
        # =====================================================================
        # We use RX instead of H for |+⟩ preparation because:
        # - RX is a single atomic operation (prepare in X-basis eigenstate)
        # - H requires relying on Stim's implicit |0⟩ and careful reset skipping
        # - RX produces cleaner detector backward-tracking through the circuit
        initial_state = "+" if self.gadget.get_measurement_basis() == "X" else "0"
        prep_config = self.gadget.get_preparation_config(initial_state)
        self._prep_config = prep_config
        
        # Build mapping of block names to data qubits
        block_data_qubits: Dict[str, List[int]] = {}
        for builder in builders:
            block_name = builder.block_name
            if block_name in prep_config.blocks:
                block_config = prep_config.blocks[block_name]
                if not block_config.skip_experiment_prep:
                    block_data_qubits[block_name] = builder.data_qubits
        
        # Use RX for |+⟩ preparation (cleaner than H)
        # With RX, we don't need to skip reset - RX is atomic
        use_rx_prep = True
        skip_reset_qubits = get_qubits_to_skip_reset(prep_config, block_data_qubits, use_rx=use_rx_prep)
        self._use_rx_prep = use_rx_prep  # Store for use by _emit_prepare_logical_states
        
        # =====================================================================
        # Phase 1.7: Emit Reset on qubits that need it
        # =====================================================================
        # GOAL 10: Use QubitAllocation.total_qubits instead of alloc["total"]
        total = unified_alloc.total_qubits
        if total > 0:
            # Only reset qubits NOT in the skip set
            reset_qubits = [q for q in range(total) if q not in skip_reset_qubits]
            if reset_qubits:
                circuit.append("R", reset_qubits)
            circuit.append("TICK")
        
        # =====================================================================
        # Phase 2: (Builders already created above)
        # =====================================================================
        
        # =====================================================================
        # Phase 2.5: Prepare logical states for all blocks
        # =====================================================================
        # This is CRITICAL for correct first-round detectors. Without state
        # preparation, detectors comparing against |0⟩^⊗n will be non-deterministic
        # for X-type stabilizers (when measuring in Z basis) or Z-type stabilizers
        # (when measuring in X basis).
        self._emit_prepare_logical_states(circuit, builders)
        
        # =====================================================================
        # Phase 3: Pre-gadget memory rounds
        # =====================================================================
        self._emit_pre_gadget_memory(circuit, builders)
        
        # Save pre-gadget measurement indices for crossing detectors
        # This MUST be done AFTER pre-gadget rounds and BEFORE gadget execution
        # Save pre-gadget measurement indices for crossing detectors
        # This MUST be done AFTER pre-gadget rounds and BEFORE gadget execution
        # Use MeasurementRecord from detector_tracking for structured tracking
        pre_gadget_record = MeasurementRecord()
        pre_gadget_meas = {}
        for builder in builders:
            last_meas = builder.get_last_measurement_indices()
            pre_gadget_meas[builder.block_name] = last_meas
            # Also store in structured record for future chaining
            for basis, indices in last_meas.items():
                pre_gadget_record.add_measurement(
                    builder.block_name, basis, "pre", indices
                )
        
        # =====================================================================
        # Phase 4: Gadget execution via PhaseOrchestrator
        # =====================================================================
        # Determine if hybrid decoding is requested
        use_hybrid = self.gadget.use_hybrid_decoding()
        
        # Execute all gadget phases using the orchestrator
        phase_result = execute_gadget_phases(
            gadget=self.gadget,
            circuit=circuit,
            unified_alloc=unified_alloc,
            ctx=ctx,
            builders=builders,
            alloc_dict=alloc,
            use_hybrid_decoding=use_hybrid,
            pre_gadget_meas=pre_gadget_meas,
        )
        
        # Extract results from orchestrator
        destroyed_blocks = phase_result.destroyed_blocks
        
        # =====================================================================
        # Phase 5: Post-gadget memory rounds (skip destroyed blocks)
        # =====================================================================
        # If the orchestrator handled crossing detectors via inter-phase rounds
        # (multi-phase gadgets like teleportation), pass None to avoid
        # double-emission.  For single-phase gadgets (e.g. TransversalCNOT)
        # where is_final=True on the gate phase, the orchestrator did NOT
        # emit inter-phase rounds, so we must pass pre_gadget_meas here so
        # that emit_scheduled_rounds can suppress auto-detectors on the first
        # post-gate round and emit crossing detectors instead.
        post_gadget_pre_meas = None if phase_result.crossing_handled else pre_gadget_meas
        self._emit_post_gadget_memory(
            circuit, builders, destroyed_blocks, post_gadget_pre_meas
        )
        
        # =====================================================================
        # Phase 6: Final measurement (skip destroyed blocks)
        # =====================================================================
        self._emit_final_measurement(
            circuit, builders, alloc, ctx, destroyed_blocks,
            destroyed_block_meas_starts=phase_result.destroyed_block_meas_starts,
        )

        # =====================================================================
        # Phase 7: Auto detector / observable emission
        # =====================================================================
        # When auto_detectors is enabled (explicitly or via
        # gadget.use_auto_detectors()), replace all manual DETECTOR and
        # OBSERVABLE_INCLUDE instructions with flow-matched auto-discovered
        # ones.  This runs BEFORE noise so the bare circuit is analysed.
        if self.auto_detectors:
            circuit = self._apply_auto_detectors(circuit)

        # Apply noise model
        circuit = apply_noise_to_circuit(circuit, self.noise_model)
        
        return circuit

    # -----------------------------------------------------------------
    # Auto detector / observable pipeline
    # -----------------------------------------------------------------

    def _apply_auto_detectors(self, circuit: stim.Circuit) -> stim.Circuit:
        """Replace manual DETECTOR / OBSERVABLE_INCLUDE with auto-discovered ones.

        1. Strip all existing DETECTOR and OBSERVABLE_INCLUDE instructions.
        2. Call ``discover_detectors()`` to find all deterministic detectors
           via Stim flow matching + GF(2) pruning.
        3. Call ``discover_observables()`` to find valid logical observables.
        4. Emit the discovered annotations into the bare circuit.

        Parameters
        ----------
        circuit : stim.Circuit
            The fully built (but pre-noise) circuit with manual annotations.

        Returns
        -------
        stim.Circuit
            Circuit with auto-discovered DETECTOR and OBSERVABLE_INCLUDE.
        """
        from qectostim.experiments.auto_detector_emission import (
            discover_detectors,
            discover_observables,
        )

        # Strip existing annotations
        bare = stim.Circuit()
        for inst in circuit.flattened():
            if inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                bare.append(inst)

        # Discover detectors
        det_result = discover_detectors(bare, use_cache=False)
        n_meas = det_result.num_measurements

        # Emit auto detectors
        for det in det_result.detectors:
            targets = [
                stim.target_rec(-(n_meas - idx))
                for idx in sorted(det.measurement_indices)
            ]
            bare.append("DETECTOR", targets, det.coordinates)

        # Discover observables
        obs_list = discover_observables(bare)

        if obs_list:
            # Pick best: prefer most blocks spanned, then lowest weight
            best = max(obs_list, key=lambda o: (o.n_blocks, -o.weight))
            targets = [
                stim.target_rec(-(n_meas - idx))
                for idx in sorted(best.measurement_indices)
            ]
            bare.append("OBSERVABLE_INCLUDE", targets, [0])

        return bare

    def run_decode(
        self,
        decoder_name: Optional[str] = None,
        num_shots: int = 10000,
        **kwargs,
    ) -> FTGadgetExperimentResult:
        """
        Run the experiment and decode results.
        
        Uses the base class `Experiment.run_decode()` for the core decoding logic,
        then wraps the result in `FTGadgetExperimentResult` for gadget-specific metadata.
        
        Parameters
        ----------
        decoder_name : Optional[str]
            Decoder to use. If None, uses intelligent selection from `select_decoder`.
        num_shots : int
            Number of shots to sample.
        **kwargs
            Additional decoder arguments.
            
        Returns
        -------
        FTGadgetExperimentResult
            Experiment results including logical error rate.
        """
        # Use base class run_decode which handles:
        # - Distance-aware routing (detection vs correction)
        # - Intelligent decoder selection via select_decoder
        # - Hypergraph DEM handling
        base_result = super().run_decode(shots=num_shots, decoder_name=decoder_name)
        
        # Get gadget metadata
        gadget_metadata = None
        try:
            gadget_metadata = self.gadget.get_metadata()
        except RuntimeError:
            pass
        
        # Get circuit info for extra metadata
        circuit = self.to_stim()
        
        return FTGadgetExperimentResult(
            logical_error_rate=base_result['logical_error_rate'],
            num_shots=base_result['shots'],
            num_errors=int(np.sum(base_result['logical_errors'])),
            gadget_metadata=gadget_metadata,
            decoder_used=decoder_name or "auto",
            extra={
                "num_detectors": circuit.num_detectors,
                "num_observables": circuit.num_observables,
                "rounds_before": self.num_rounds_before,
                "rounds_after": self.num_rounds_after,
                **{k: v for k, v in base_result.items() 
                   if k not in ('logical_error_rate', 'shots', 'logical_errors')},
            },
        )
    
    def verify_zero_noise_ler(
        self,
        num_shots: int = 1000,
        tolerance: float = 0.0,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Verify that the circuit has zero logical error rate with no noise.
        
        This is a critical sanity check: if the gadget correctly implements
        the logical gate and the observable is properly tracked, then with
        no noise the logical error rate MUST be zero. Any non-zero LER
        indicates a bug in:
        - Observable tracking (wrong qubits included in OBSERVABLE_INCLUDE)
        - Stabilizer transform application (detectors reference wrong measurements)
        - Gate implementation (wrong physical gates emitted)
        
        Parameters
        ----------
        num_shots : int
            Number of shots to sample (default 1000).
        tolerance : float
            Acceptable error rate (default 0.0 for strict verification).
            
        Returns
        -------
        Tuple[bool, float, Optional[str]]
            (passed, ler, error_message)
            - passed: True if LER <= tolerance
            - ler: Measured logical error rate
            - error_message: Description of failure if any
        """
        # Create a copy of the experiment with no noise
        no_noise_exp = FaultTolerantGadgetExperiment(
            codes=self.codes,
            gadget=self.gadget,
            noise_model=None,  # No noise!
            num_rounds_before=self.num_rounds_before,
            num_rounds_after=self.num_rounds_after,
        )
        
        try:
            circuit = no_noise_exp.to_stim()
        except Exception as e:
            return False, 1.0, f"Circuit generation failed: {e}"
        
        # Check for required components
        circuit_str = str(circuit)
        if "DETECTOR" not in circuit_str:
            return False, 1.0, "No DETECTOR instructions in circuit"
        if "OBSERVABLE_INCLUDE" not in circuit_str:
            return False, 1.0, "No OBSERVABLE_INCLUDE in circuit"
        
        try:
            # Sample from circuit (no noise, so should be deterministic)
            sampler = circuit.compile_detector_sampler()
            samples = sampler.sample(num_shots, append_observables=True)
            
            num_detectors = circuit.num_detectors
            num_observables = circuit.num_observables
            
            if num_observables == 0:
                return False, 1.0, "Circuit has 0 observables"
            
            # Check detector values (should all be 0 with no noise)
            detector_shots = samples[:, :num_detectors]
            observable_shots = samples[:, num_detectors:num_detectors + num_observables]
            
            # Detector check: any non-zero detector indicates a problem
            total_detector_flips = np.sum(detector_shots)
            if total_detector_flips > 0:
                flip_rate = total_detector_flips / (num_shots * num_detectors)
                return False, 1.0, f"Detectors firing with no noise (flip_rate={flip_rate:.4f})"
            
            # Observable check: should all be 0 (identity result)
            ler = float(np.mean(observable_shots[:, 0]))
            
            if ler > tolerance:
                return False, ler, f"Zero-noise LER = {ler:.4f} > {tolerance} (observable tracking bug)"
            
            return True, ler, None
            
        except Exception as e:
            return False, 1.0, f"Verification failed: {e}"


def run_ft_gadget_experiment(
    codes: List[Code],
    gadget: Gadget,
    noise_model: NoiseModel,
    num_rounds_before: int = 3,
    num_rounds_after: int = 3,
    decoder_name: str = "pymatching",
    num_shots: int = 10000,
) -> FTGadgetExperimentResult:
    """
    Convenience function to run a fault-tolerant gadget experiment.
    
    Parameters
    ----------
    codes : List[Code]
        Code(s) to apply gadget to.
    gadget : Gadget
        The logical gate gadget.
    noise_model : NoiseModel
        Noise model.
    num_rounds_before : int
        Stabilizer rounds before gadget.
    num_rounds_after : int
        Stabilizer rounds after gadget.
    decoder_name : str
        Decoder to use.
    num_shots : int
        Number of shots.
        
    Returns
    -------
    FTGadgetExperimentResult
        Experiment results.
    """
    exp = FaultTolerantGadgetExperiment(
        codes=codes,
        gadget=gadget,
        noise_model=noise_model,
        num_rounds_before=num_rounds_before,
        num_rounds_after=num_rounds_after,
    )
    return exp.run_decode(decoder_name=decoder_name, num_shots=num_shots)
