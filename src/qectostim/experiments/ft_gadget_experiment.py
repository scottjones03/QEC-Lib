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
    HierarchicalConcatenatedStabilizerRoundBuilder,
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
        d_inner: int = 1,
    ):
        # Use first code as primary for base class
        super().__init__(codes[0], noise_model, metadata)
        
        self.codes = codes
        self.gadget = gadget
        self.num_rounds_before = num_rounds_before
        self.num_rounds_after = num_rounds_after
        self.d_inner = d_inner
        # Infer auto_detectors from gadget when not explicitly set
        if auto_detectors is None:
            self.auto_detectors = gadget.use_auto_detectors()
        else:
            self.auto_detectors = auto_detectors
        
        # Detect concatenated CSS codes — must check BEFORE is_css
        # since ConcatenatedCSSCode inherits CSSCode
        self._is_hierarchical = self._detect_concatenated_codes(codes)
        
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
        self._qec_metadata: Optional[Any] = None
        self._prep_config: Optional[PreparationConfig] = None  # Set in to_stim()
        self._use_rx_prep: bool = True  # Use RX instead of H for |+⟩ prep
    
    @property
    def qec_metadata(self) -> Optional[Any]:
        """Rich QEC metadata for the hardware compiler.

        Available after ``to_stim()`` has been called.  Returns a
        :class:`QECMetadata` instance populated from the builders,
        allocation, and gadget.
        """
        return self._qec_metadata

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
    def _detect_concatenated_codes(codes: List[Code]) -> bool:
        """Detect if any code is a concatenated CSS code needing hierarchical treatment.

        A concatenated CSS code (ConcatenatedCSSCode) inherits from CSSCode,
        so ``code.is_css`` returns True.  Without explicit detection the
        experiment would use a flat CSSStabilizerRoundBuilder, losing all
        hierarchical structure (outer ancilla blocks, crossing detectors,
        nested inner rounds).

        Single-code experiments (``codes=[concat]``) are fully supported in
        hierarchical mode.  The gadget's ``compute_layout()`` determines
        how many blocks are needed and all blocks share the same
        concatenated code.  This covers TransversalCNOT, teleportation
        gadgets, Knill EC, and CSS surgery CNOT.

        Multi-code experiments (``codes=[concat_a, concat_b]``) with
        *different* concatenated codes are not yet supported.

        Returns
        -------
        bool
            True if hierarchical mode should be used.
        """
        try:
            from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
        except ImportError:
            return False

        has_concat = any(isinstance(c, ConcatenatedCSSCode) for c in codes)
        if not has_concat:
            return False

        # Single-code: the gadget may create multiple blocks from the
        # same code (e.g. TransversalCNOT uses codes=[c] for both
        # control and target).  This is fully supported.
        if len(codes) == 1:
            return True

        # Multi-code: all must be the same concatenated code.
        # Different concatenated codes would need heterogeneous builders.
        if all(isinstance(c, ConcatenatedCSSCode) for c in codes):
            return True

        # Mix of concatenated and non-concatenated codes is not supported.
        import warnings
        warnings.warn(
            "Mixed concatenated/non-concatenated codes are not yet "
            "supported in hierarchical mode. Falling back to flat "
            "(non-FT outer) round builder.",
            stacklevel=3,
        )
        return False

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
    
    def _compute_hierarchical_allocation(
        self,
    ) -> Tuple[QubitAllocation, Dict[str, Any]]:
        """Compute qubit allocation for hierarchical concatenated codes.

        Hierarchical concatenated codes require more qubits than a flat
        CSS layout because each gadget-level block uses outer ancilla
        blocks (logical ancillas for fault-tolerant outer stabilizer
        measurement).  The standard ``layout.allocate_qubits()`` does not
        know about these extra qubits, so we compute the allocation here
        using the hierarchical builder's own layout logic.

        Returns
        -------
        Tuple[QubitAllocation, Dict[str, Any]]
            ``(unified_alloc, alloc_dict)`` — the QubitAllocation and its
            dict representation for backwards compatibility.
        """
        from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
        from qectostim.gadgets.layout import BlockAllocation

        layout = self.gadget.compute_layout(self.codes)
        block_names = list(layout.blocks.keys())

        # Map block names to codes
        block_codes: Dict[str, ConcatenatedCSSCode] = {}
        for i, bname in enumerate(block_names):
            code = self.codes[min(i, len(self.codes) - 1)]
            assert isinstance(code, ConcatenatedCSSCode), (
                f"Block {bname}: expected ConcatenatedCSSCode, got {type(code).__name__}"
            )
            block_codes[bname] = code

        # Lay out blocks contiguously — each block's qubit range starts
        # where the previous one ended.  We use a temporary builder just
        # to query the qubit range sizes.
        running_offset = 0
        block_offsets: Dict[str, Tuple[int, int]] = {}  # bname -> (data_start, outer_anc_end)
        for bname in block_names:
            code = block_codes[bname]
            temp = HierarchicalConcatenatedStabilizerRoundBuilder(
                code, DetectorContext(),
                block_name=bname,
                data_offset=running_offset,
                d_inner=self.d_inner,
            )
            block_offsets[bname] = (running_offset, temp._outer_anc_end)
            running_offset = temp._outer_anc_end

        total_qubits = running_offset

        # Build QubitAllocation
        hier_alloc = QubitAllocation()
        for bname in block_names:
            code = block_codes[bname]
            data_start, outer_end = block_offsets[bname]
            n_out = code.n_outer
            n_in = code.n_inner
            r_x_in = code.inner.hx.shape[0] if code.inner.hx.size > 0 else 0
            r_z_in = code.inner.hz.shape[0] if code.inner.hz.size > 0 else 0
            n_data = n_out * n_in
            inner_anc_per_block = r_x_in + r_z_in
            inner_anc_start = data_start + n_data
            inner_anc_end = inner_anc_start + n_out * inner_anc_per_block
            outer_anc_start = inner_anc_end

            hier_alloc.blocks[bname] = BlockAllocation(
                block_name=bname,
                code=code,
                data_start=data_start,
                data_count=n_data,
                x_anc_start=inner_anc_start,
                x_anc_count=inner_anc_end - inner_anc_start,
                z_anc_start=outer_anc_start,
                z_anc_count=outer_end - outer_anc_start,
            )
        hier_alloc._total_qubits = total_qubits

        alloc_dict = self._allocation_to_dict(hier_alloc)
        return hier_alloc, alloc_dict

    def _create_hierarchical_builders(
        self,
        alloc: Dict[str, Any],
        ctx: DetectorContext,
    ) -> List[BaseStabilizerRoundBuilder]:
        """Create HierarchicalConcatenatedStabilizerRoundBuilder per block.

        Uses the same per-block measurement basis logic as ``_create_builders``
        but instantiates hierarchical builders with proper ``data_offset``
        and ``d_inner``.
        """
        from qectostim.codes.composite.concatenated import ConcatenatedCSSCode

        meas_config = self.gadget.get_measurement_config()
        global_basis = self.gadget.get_measurement_basis()
        # Determine initial basis from preparation config
        initial_state = "+" if global_basis == "X" else "0"
        prep_config = self.gadget.get_preparation_config(initial_state)

        builders: List[BaseStabilizerRoundBuilder] = []
        for block_name, block_info in alloc.items():
            if block_name == "total":
                continue

            code = block_info["code"]
            assert isinstance(code, ConcatenatedCSSCode)
            data_start = block_info["data"][0]

            # Initial basis from prep state: |0⟩ → "Z", |+⟩ → "X"
            block_prep = prep_config.blocks.get(block_name)
            if block_prep is not None:
                init_basis = "X" if block_prep.initial_state == "+" else "Z"
            else:
                init_basis = meas_config.block_bases.get(block_name, global_basis)

            builder = HierarchicalConcatenatedStabilizerRoundBuilder(
                code, ctx,
                block_name=block_name,
                data_offset=data_start,
                measurement_basis=init_basis,
                d_inner=self.d_inner,
            )
            builders.append(builder)

        return builders
    
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
        - ConcatenatedCSSCode (hierarchical): HierarchicalConcatenatedStabilizerRoundBuilder
        - CSSCode with hx/hz: CSSStabilizerRoundBuilder
        - StabilizerCode with stabilizer_matrix: GeneralStabilizerRoundBuilder
        
        X stabiliser measurement now always uses CX-based extraction
        (RX/CX/MRX), so there is no per-gadget mode selection.
        """
        # For hierarchical concatenated codes, use dedicated builder creation
        if self._is_hierarchical:
            return self._create_hierarchical_builders(alloc, ctx)

        builders = []
        
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
        
        # ─── Hierarchical concatenated codes ───
        # Hierarchical builders have their own emit_prepare_logical_state()
        # that measures stabilizers in the correct order:
        #   |0⟩ prep → Z-stabs FIRST (deterministic on pristine data) → X-stabs
        #   |+⟩ prep → X-stabs FIRST (deterministic on |+⟩^⊗n) → Z-stabs
        # This establishes anchor detectors and populates measurement history.
        # The generic emit_initial_states() only does RX for |+⟩ blocks and
        # does NOT measure stabilizers, leaving the builder with empty history
        # so the first emit_round() produces non-deterministic anchors.
        if self._is_hierarchical:
            for builder in builders:
                block_prep = prep_config.blocks.get(builder.block_name)
                if block_prep is not None and block_prep.skip_experiment_prep:
                    # Gadget will prepare this block (e.g. ancilla in teleportation)
                    continue
                # Determine state: from prep_config or fall back to basis-implied
                initial_state_str = "0"
                if block_prep is not None:
                    initial_state_str = block_prep.initial_state
                elif self.gadget.get_measurement_basis() == "X":
                    initial_state_str = "+"
                builder.emit_prepare_logical_state(
                    circuit, state=initial_state_str, logical_idx=0,
                )
            return
        
        # ─── Flat (non-hierarchical) codes ───
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
        
        # Hierarchical builders handle both X and Z in a single emit_round()
        # call (inner stabs + outer stabs are tightly coupled).  Parallel
        # extraction would split them into separate Z-only / X-only calls,
        # doubling the outer operations and corrupting the data.
        if self._is_hierarchical:
            parallel = False
        
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
    
    @staticmethod
    def _compensate_hierarchical_pre_meas(
        builder: "BaseStabilizerRoundBuilder",
        raw_meas: Dict[str, list],
    ) -> Dict[str, list]:
        """Add outer-ancilla compensation to pre-gadget inner measurements.

        For hierarchical concatenated codes, crossing detectors that compare
        pre-gadget inner measurements with post-gadget inner measurements
        need *compensation targets* from the pre-gadget outer ancilla
        preparation.

        **Why this is needed:**
        Backward propagation from a post-gadget inner X measurement
        traverses the pre-gadget outer Z round's ``CX(data → outer_z_anc)``
        gates, spreading X sensitivity onto outer Z ancilla qubits.  That
        sensitivity then anti-commutes with the ``R`` (Z-basis reset) used
        to prepare the outer Z ancilla in |0_L⟩.  The pre-gadget inner X
        measurement does *not* traverse these outer operations (it was
        measured before them), so the sensitivities are asymmetric and the
        raw 2-term crossing detector is non-deterministic.

        Including the inner X stabiliser measurement from the outer Z
        ancilla |0_L⟩ preparation (stored in ``_outer_z_anc_x_prep``)
        absorbs the asymmetric X sensitivity at the same ``R``, restoring
        determinism.  The analogous argument applies to inner Z stabilisers
        and outer X ancilla |+_L⟩ preparation (``_outer_x_anc_z_prep``).

        This is the same mechanism the builder uses internally for crossing
        detectors between consecutive outer blocks within a single epoch
        (see ``_emit_inner_x_detector`` / ``_emit_inner_z_detector``).

        Parameters
        ----------
        builder : HierarchicalConcatenatedRoundBuilder
            Builder with outer-ancilla prep measurement data.
        raw_meas : Dict[str, list]
            Raw measurement indices from ``get_last_measurement_indices()``.

        Returns
        -------
        Dict[str, list]
            Copy of *raw_meas* where inner ``int`` entries that need
            compensation have been replaced by ``List[int]`` entries of the
            form ``[inner_meas, comp_1, comp_2, …]``.
        """
        # Guard: only hierarchical builders carry the required attributes
        if not hasattr(builder, '_outer_z_anc_x_prep'):
            return raw_meas

        n_out = builder.n_out
        r_x_in = builder.r_x_in
        r_z_in = builder.r_z_in

        result = {}
        for k, v in raw_meas.items():
            if isinstance(v, list):
                result[k] = list(v)
            else:
                result[k] = v

        # --- Inner X: compensate with outer Z ancilla X-prep ---
        n_inner_x = n_out * r_x_in
        x_list = result.get('x', [])
        for idx in range(min(n_inner_x, len(x_list))):
            entry = x_list[idx]
            if entry is None or isinstance(entry, list):
                continue  # skip missing / already-compound entries
            block_id = idx // r_x_in
            local_idx = idx % r_x_in
            comp: list = []
            for j in builder._outer_z_stabs_for_block(block_id):
                c = builder._outer_z_anc_x_prep.get((j, local_idx))
                if c is not None:
                    comp.append(c)
            if comp:
                x_list[idx] = [entry] + comp

        # --- Inner Z: compensate with outer X ancilla Z-prep ---
        n_inner_z = n_out * r_z_in
        z_list = result.get('z', [])
        for idx in range(min(n_inner_z, len(z_list))):
            entry = z_list[idx]
            if entry is None or isinstance(entry, list):
                continue
            block_id = idx // r_z_in
            local_idx = idx % r_z_in
            comp = []
            for j in builder._outer_x_stabs_for_block(block_id):
                c = builder._outer_x_anc_z_prep.get((j, local_idx))
                if c is not None:
                    comp.append(c)
            if comp:
                z_list[idx] = [entry] + comp

        return result

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
        
        # Hierarchical builders handle both X and Z in a single emit_round()
        # call.  Parallel extraction would split them, corrupting outer ops.
        if self._is_hierarchical:
            parallel = False
        
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
        
        # ── Post-gadget rounds with crossing detectors ──
        # For the first post-gadget round, emit_scheduled_rounds will:
        #   1. Suppress auto-detectors (emit_detectors=False)
        #   2. Emit the full round (inner + outer for hierarchical)
        #   3. Emit crossing detectors comparing pre- and post-gadget
        #      stabilizer measurements
        #
        # For hierarchical codes, _skip_first_round=True (set by
        # reset_stabilizer_history) already suppresses anchor/temporal
        # detectors that the builder would normally emit.  Crossing
        # detectors cover ALL stabilizers (inner + outer) since
        # get_last_measurement_indices() now returns both.
        #
        # Determine destruction bases for destroyed blocks so crossing
        # detector emission can skip formulas whose "post" terms
        # anti-commute with the destructive measurement (e.g., skip
        # post-Z on a block destroyed by MX).
        destroyed_block_bases: Dict[str, str] = {}
        if destroyed_blocks:
            meas_config = self.gadget.get_measurement_config()
            global_basis = self.gadget.get_measurement_basis()
            for block_name in destroyed_blocks:
                basis = meas_config.block_bases.get(block_name, global_basis)
                destroyed_block_bases[block_name] = basis
        emit_scheduled_rounds(
            circuit, active_builders, post_schedules, self.num_rounds_after,
            parallel=parallel, stab_type=stab_type,
            crossing_config=crossing_config,
            pre_gadget_meas=pre_gadget_meas,
            destroyed_blocks=destroyed_blocks,
            all_builders=builders,
            destroyed_block_bases=destroyed_block_bases,
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

        if self._is_hierarchical:
            self._emit_hierarchical_final(
                circuit, builders, destroyed_blocks,
                destroyed_block_meas_starts=destroyed_block_meas_starts,
            )
            return

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
    
    # -----------------------------------------------------------------
    # Hierarchical concatenated code path — final measurement
    # -----------------------------------------------------------------

    def _emit_hierarchical_final(
        self,
        circuit: stim.Circuit,
        builders: List[BaseStabilizerRoundBuilder],
        destroyed_blocks: Set[str],
        destroyed_block_meas_starts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Emit trailing inner rounds + final measurement for hierarchical codes.

        Hierarchical builders handle data measurement, boundary detectors,
        and observable terms in their own ``emit_final_measurement`` call,
        which correctly handles the multi-term inner/outer boundary formulas.

        This replaces the flat ``_emit_final_measurement`` pipeline
        (destructive measurement → boundary detectors → emit_observable)
        when ``_is_hierarchical`` is True.

        For gadgets with correlation_terms referencing destroyed blocks
        (e.g. KnillEC: Z_B ⊕ MZ(bell_a)), the destroyed block's mid-circuit
        measurements are added to the observable after surviving blocks
        have emitted their final measurements.
        """
        if destroyed_block_meas_starts is None:
            destroyed_block_meas_starts = {}

        meas_config = self.gadget.get_measurement_config()
        obs_config = self.gadget.get_observable_config()
        obs_blocks = set(obs_config.output_blocks)
        prep_config = self._prep_config

        active_builders = [
            b for b in builders
            if b.block_name not in destroyed_blocks
        ]

        # Trailing inner rounds for temporal depth at the boundary
        for b in active_builders:
            if hasattr(b, 'd_inner'):
                for _ in range(b.d_inner):
                    b.emit_inner_only_round(
                        circuit, stab_type=StabilizerBasis.BOTH, emit_detectors=True,
                    )

        # Determine which SURVIVING blocks contribute to the observable
        active_block_names = {b.block_name for b in active_builders}
        obs_block_builders: set = set()
        for ob in obs_blocks:
            if ob in active_block_names:
                obs_block_builders.add(ob)
            else:
                # Check alias mapping
                for b in active_builders:
                    if prep_config is not None:
                        norm = prep_config.get_normalized_block_name(ob)
                        if norm == b.block_name:
                            obs_block_builders.add(b.block_name)

        # Emit final measurement per surviving block.
        # For multi-block gadgets, we MUST defer boundary detector emission
        # until all blocks have measured. Otherwise, lookbacks computed
        # during block_0's measurement will be wrong after block_1 adds
        # more measurements.
        need_deferred = len(active_builders) > 1
        for b in active_builders:
            final_basis = meas_config.block_bases.get(
                b.block_name, self.gadget.get_measurement_basis(),
            )
            skip_obs = b.block_name not in obs_block_builders
            b.emit_final_measurement(
                circuit, basis=final_basis, logical_idx=0,
                skip_observable=skip_obs,
                defer_boundary_detectors=need_deferred,
            )

        # Emit deferred boundary detectors now that all blocks have measured
        if need_deferred:
            for b in active_builders:
                b.emit_deferred_boundary_detectors(circuit)

        # Add correlation terms from destroyed blocks to the observable.
        # For teleportation gadgets (KnillEC, CNOT-HTel), the observable
        # includes mid-circuit measurements from blocks that were destroyed
        # during the gadget.  These measurements are tracked via
        # destroyed_block_meas_starts.
        if obs_config.correlation_terms and destroyed_block_meas_starts:
            from qectostim.experiments.observable import get_logical_support
            # Build a map from block_name → (code, builder) for destroyed blocks
            destroyed_builder_map = {
                b.block_name: b for b in builders
                if b.block_name in destroyed_blocks
            }
            for term in obs_config.correlation_terms:
                # Skip terms for surviving blocks (already handled by builders)
                if term.block in active_block_names:
                    continue
                # This term references a destroyed block
                if term.block not in destroyed_block_meas_starts:
                    continue
                block_meas_start = destroyed_block_meas_starts[term.block]
                # Get the code for this block from its builder
                builder = destroyed_builder_map.get(term.block)
                if builder is None:
                    continue
                code = builder.code
                support = get_logical_support(code, term.basis, 0)
                if support:
                    meas_indices = [block_meas_start + q for q in support]
                    self._ctx.add_observable_measurement(0, meas_indices)

        # Emit observable(s) — only observable 0 has been populated
        self._ctx.emit_observable(circuit, observable_idx=0)

    # -----------------------------------------------------------------
    # Hierarchical concatenated code path (LEGACY — to be removed)
    # -----------------------------------------------------------------

    def _to_stim_hierarchical(self) -> stim.Circuit:
        """Build circuit using hierarchical round builders for concatenated codes.

        This mirrors the approach in
        :class:`ConcatenatedCSSMemoryExperiment._to_stim_hierarchical` but wraps
        gadget phases between pre- and post-gadget hierarchical memory rounds.

        Supports multi-block gadgets (TransversalCNOT, teleportation, Knill EC,
        CSS surgery CNOT) by creating one
        :class:`HierarchicalConcatenatedStabilizerRoundBuilder` per block with
        non-overlapping qubit regions.

        Structure
        ---------
        1. Qubit coords & reset for all blocks
        2. Logical state preparation per block (initial states from gadget)
        3. Pre-gadget memory: ``num_rounds_before`` outer rounds (skipped blocks per gadget)
        4. Gadget execution via ``gadget.emit_next_phase``
        5. Post-gadget memory: ``num_rounds_after`` outer rounds (skip destroyed blocks)
        6. Trailing inner rounds per surviving block
        7. Final measurement + boundary detectors + observables
        """
        from qectostim.codes.composite.concatenated import ConcatenatedCSSCode
        from qectostim.gadgets.layout import BlockAllocation

        # ── Resolve codes and block names from gadget layout ──
        layout = self.gadget.compute_layout(self.codes)
        block_names = list(layout.blocks.keys())
        num_blocks = len(block_names)

        # Map each block name to its ConcatenatedCSSCode.
        # If codes has 1 entry, every block uses it.  Otherwise match by index.
        block_codes: Dict[str, ConcatenatedCSSCode] = {}
        for i, bname in enumerate(block_names):
            code = self.codes[min(i, len(self.codes) - 1)]
            assert isinstance(code, ConcatenatedCSSCode), (
                f"Block {bname}: expected ConcatenatedCSSCode, got {type(code).__name__}"
            )
            block_codes[bname] = code

        # Determine per-block measurement bases and initial states.
        meas_config = self.gadget.get_measurement_config()
        global_basis = self.gadget.get_measurement_basis()

        # Preparation config determines which stabilizers are deterministic
        # in the first round (anchor detectors).  For |0⟩ prep → Z anchors
        # (measurement_basis="Z"), for |+⟩ prep → X anchors ("X").
        initial_state = "+" if global_basis == "X" else "0"
        prep_config = self.gadget.get_preparation_config(initial_state)

        ctx = DetectorContext()
        self._ctx = ctx
        d_inner = self.d_inner

        # ── Create one hierarchical builder per block ──
        # Builders are laid out contiguously: block_0 qubits, then block_1, etc.
        # The *initial* measurement_basis is determined by the preparation state
        # (not the final measurement basis) so that first-round anchor detectors
        # are emitted for the correct (deterministic) stabiliser type.
        builders: Dict[str, HierarchicalConcatenatedStabilizerRoundBuilder] = {}
        running_offset = 0
        for bname in block_names:
            code = block_codes[bname]
            # Initial basis from prep state: |0⟩ → "Z", |+⟩ → "X"
            block_prep = prep_config.blocks.get(bname)
            if block_prep is not None:
                init_basis = "X" if block_prep.initial_state == "+" else "Z"
            else:
                init_basis = meas_config.block_bases.get(bname, global_basis)
            b = HierarchicalConcatenatedStabilizerRoundBuilder(
                code, ctx,
                block_name=bname,
                data_offset=running_offset,
                measurement_basis=init_basis,
                d_inner=d_inner,
            )
            builders[bname] = b
            running_offset = b._outer_anc_end  # next block starts after this one

        total_qubits = running_offset

        # ── Build QubitAllocation matching gadget expectations ──
        hier_alloc = QubitAllocation()
        for bname, b in builders.items():
            hier_alloc.blocks[bname] = BlockAllocation(
                block_name=bname,
                code=block_codes[bname],
                data_start=b._data_start,
                data_count=b.n_data,
                x_anc_start=b._inner_anc_start,
                x_anc_count=b._inner_anc_end - b._inner_anc_start,
                z_anc_start=b._outer_anc_start,
                z_anc_count=b._outer_anc_end - b._outer_anc_start,
            )
        hier_alloc._total_qubits = total_qubits

        c = stim.Circuit()

        # ── Phase 1: Coords + reset ──
        for b in builders.values():
            b.emit_qubit_coords(c)
        # Reset all qubits at once
        c.append("R", list(range(total_qubits)))
        c.append("TICK")

        # ── Phase 2: State preparation ──
        # Get preparation config from gadget to determine per-block initial states
        initial_state = "+" if global_basis == "X" else "0"
        prep_config = self.gadget.get_preparation_config(initial_state)

        blocks_skip_pre = self.gadget.get_blocks_to_skip_pre_rounds()

        for bname, b in builders.items():
            block_prep = prep_config.blocks.get(bname)
            if block_prep is not None and block_prep.skip_experiment_prep:
                # Gadget will prepare this block (e.g. ancilla in teleportation)
                continue
            # Determine state: from prep_config or fall back to basis-implied
            state = initial_state
            if block_prep is not None:
                state = block_prep.initial_state
            b.emit_prepare_logical_state(c, state=state, logical_idx=0)

        # ── Phase 3: Pre-gadget memory rounds ──
        active_pre = [
            b for bname, b in builders.items()
            if bname not in blocks_skip_pre
        ]
        for _r in range(self.num_rounds_before):
            for b in active_pre:
                if d_inner > 1:
                    b.emit_outer_round(c, stab_type=StabilizerBasis.BOTH, emit_detectors=True)
                else:
                    b.emit_round(c, stab_type=StabilizerBasis.BOTH, emit_detectors=True)

        # ── Phase 3b: Save pre-gadget measurements for crossing detectors ──
        pre_gadget_meas: Dict[str, Dict[str, List[int]]] = {}
        for bname, b in builders.items():
            if hasattr(b, 'get_last_measurement_indices'):
                pre_gadget_meas[bname] = b.get_last_measurement_indices()

        # ── Phase 4: Gadget execution ──
        self.gadget.reset_phases()
        n_phases = self.gadget.num_phases

        for phase_idx in range(n_phases):
            phase_result = self.gadget.emit_next_phase(c, hier_alloc, ctx)

            # Between non-final phases (multi-phase gadgets), run inter-phase
            # EC rounds on blocks the gadget requests.
            if not phase_result.is_final:
                ec_blocks = set()
                if hasattr(phase_result, 'request_ec_blocks') and phase_result.request_ec_blocks:
                    ec_blocks = set(phase_result.request_ec_blocks)
                elif hasattr(phase_result, 'request_ec') and phase_result.request_ec:
                    ec_blocks = set(builders.keys())

                for bname in ec_blocks:
                    if bname in builders:
                        b = builders[bname]
                        if d_inner > 1:
                            b.emit_outer_round(c, stab_type=StabilizerBasis.BOTH, emit_detectors=True)
                        else:
                            b.emit_round(c, stab_type=StabilizerBasis.BOTH, emit_detectors=True)

        # ── Phase 4b: Apply stabilizer transform to surviving builders ──
        # Two types of basis swap may be needed:
        # (a) Global swap_xz from the stabiliser transform (e.g. Hadamard)
        # (b) Per-block swap when the final measurement basis differs from the
        #     initial preparation basis (e.g. CNOT(+,0): block_1 starts Z, ends X)
        stab_transform = self.gadget.get_stabilizer_transform()
        destroyed = set(self.gadget.get_destroyed_blocks())
        blocks_skip_post = set(self.gadget.get_blocks_to_skip_post_rounds()) if hasattr(self.gadget, 'get_blocks_to_skip_post_rounds') else set()

        for bname, b in builders.items():
            if bname in destroyed:
                continue
            # Determine if this block needs an X↔Z swap.
            # Start with the global transform, then check per-block basis mismatch.
            swap_xz = stab_transform.swap_xz

            # Per-block basis mismatch: if the post-gadget measurement basis
            # differs from the builder's current (initial) basis, we need a swap.
            final_basis = meas_config.block_bases.get(bname, global_basis)
            if final_basis != b.measurement_basis and not swap_xz:
                swap_xz = True

            b.reset_stabilizer_history(
                swap_xz=swap_xz,
                skip_first_round=True,
            )

        # ── Phase 5: Post-gadget memory rounds ──
        active_post = [
            b for bname, b in builders.items()
            if bname not in destroyed and bname not in blocks_skip_post
        ]
        crossing_config = self.gadget.get_crossing_detector_config()
        for _r in range(self.num_rounds_after):
            for b in active_post:
                if d_inner > 1:
                    b.emit_outer_round(c, stab_type=StabilizerBasis.BOTH, emit_detectors=True)
                else:
                    b.emit_round(c, stab_type=StabilizerBasis.BOTH, emit_detectors=True)

            # After the first post-gadget round, emit crossing detectors.
            # The first round's auto-detectors were suppressed by
            # _skip_first_round, so we emit the crossing detectors manually
            # using the pre-gadget and now-available post-gadget measurements.
            if _r == 0 and crossing_config is not None and pre_gadget_meas:
                from qectostim.experiments.detector_emission import emit_crossing_detectors
                emit_crossing_detectors(
                    c,
                    active_post,
                    pre_gadget_meas,
                    crossing_config,
                    destroyed,
                )

        # ── Phase 6: Trailing inner rounds ──
        for b in active_post:
            for _ in range(d_inner):
                b.emit_inner_only_round(c, stab_type=StabilizerBasis.BOTH, emit_detectors=True)

        # ── Phase 7: Final measurement + boundary + observable ──
        # Use ObservableConfig to determine which blocks contribute to the
        # observable and in which basis.
        obs_config = self.gadget.get_observable_config()
        obs_blocks = set(obs_config.output_blocks)

        # Map gadget block names to builder block names for observable lookup.
        # Gadget block names may use aliases (block_0, data_block, etc.)
        obs_block_builders: set = set()
        for ob in obs_blocks:
            # Check direct name
            if ob in builders and ob not in destroyed:
                obs_block_builders.add(ob)
            # Check alias mapping (data_block → block_0, etc.)
            else:
                for bname in builders:
                    if bname not in destroyed:
                        norm = prep_config.get_normalized_block_name(ob)
                        if norm == bname or prep_config.get_normalized_block_name(bname) == ob:
                            obs_block_builders.add(bname)

        # Emit final measurement for ALL active blocks (for boundary detectors),
        # but only add observable measurements for blocks in obs_block_builders.
        for b_info in active_post:
            bname = b_info.block_name
            final_basis = b_info.measurement_basis
            skip_obs = bname not in obs_block_builders
            b_info.emit_final_measurement(
                c, basis=final_basis, logical_idx=0,
                skip_observable=skip_obs,
            )

        # Emit observable(s) — only observable 0 has been populated
        ctx.emit_observable(c, observable_idx=0)

        # Auto detectors if requested
        if self.auto_detectors:
            c = self._apply_auto_detectors(c)

        # Apply noise
        c = apply_noise_to_circuit(c, self.noise_model)
        return c

    def to_stim(self) -> stim.Circuit:
        """
        Generate complete fault-tolerant Stim circuit.
        
        For concatenated CSS codes with hierarchical mode enabled, delegates
        to :meth:`_to_stim_hierarchical` which uses a single
        :class:`HierarchicalConcatenatedStabilizerRoundBuilder` that manages
        qubit allocation, round emission, and detector emission internally.

        For all other codes, uses the standard flat builder pipeline.
        
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
        if self._is_hierarchical:
            # Hierarchical concatenated codes need a custom allocation that
            # accounts for outer ancilla blocks (not part of flat layout).
            unified_alloc, alloc = self._compute_hierarchical_allocation()
        else:
            # Standard flat allocation from gadget layout
            layout = self.gadget.compute_layout(self.codes)
            unified_alloc = layout.allocate_qubits()
            alloc = self._allocation_to_dict(unified_alloc)
        
        # Store for later use by other methods
        self._unified_allocation = unified_alloc
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
        # For hierarchical codes, builders emit their own coords (they know
        # about outer ancilla blocks).  Coords are emitted after builder
        # creation below.  For flat codes, use standard allocation coords.
        if not self._is_hierarchical:
            self._emit_qubit_coords(circuit, unified_alloc)
        
        # =====================================================================
        # Phase 1.5: Create builders BEFORE reset to know which qubits need H
        # =====================================================================
        # We need builders to know which data qubits belong to which block
        builders = self._create_builders(alloc, ctx, unified_alloc)
        self._builders = builders

        # Build rich QEC metadata from builders + allocation + gadget
        try:
            from qectostim.experiments.hardware_simulation.core.pipeline import QECMetadata
            self._qec_metadata = QECMetadata.from_gadget_experiment(
                codes=self.codes,
                builders=builders,
                allocation=unified_alloc,
                gadget=self.gadget,
                rounds_before=self.num_rounds_before,
                rounds_after=self.num_rounds_after,
            )
        except Exception:
            self._qec_metadata = None

        # Hierarchical builders emit their own qubit coords
        if self._is_hierarchical:
            for b in builders:
                b.emit_qubit_coords(circuit)
        
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
        
        # Save pre-gadget measurement indices for crossing detectors.
        # For hierarchical builders the inner entries are *compensated* with
        # outer-ancilla preparation measurements so that crossing detectors
        # across the gadget boundary are deterministic (see
        # _compensate_hierarchical_pre_meas for the full rationale).
        # This MUST be done AFTER pre-gadget rounds and BEFORE gadget
        # execution (which clears the builder's outer-prep tracking).
        pre_gadget_record = MeasurementRecord()
        pre_gadget_meas = {}
        for builder in builders:
            last_meas = builder.get_last_measurement_indices()
            # Hierarchical: wrap inner meas with outer-ancilla compensation
            last_meas = self._compensate_hierarchical_pre_meas(
                builder, last_meas
            )
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
        
        # Execute all gadget phases using the orchestrator.
        #
        # Pass pre_gadget_meas for ALL code types (flat and hierarchical).
        # The orchestrator:
        # 1. Updates pre_gadget_meas for freshly-prepared blocks after
        #    Phase 1 inter-phase EC (e.g. ancilla in teleportation).
        # 2. Emits crossing detectors during inter-phase rounds after
        #    the GATE phase, BEFORE any block is destroyed.
        #
        # For hierarchical codes, crossing detectors use compensated
        # pre-gadget measurements and raw post-gate measurements.  This
        # works because after clear_history, the first post-gate round
        # has no prior outer operations to cross — matching the same
        # mechanism that makes TransversalCNOT crossing detectors
        # deterministic on hierarchical codes.
        orchestrator_pre_meas = pre_gadget_meas
        phase_result = execute_gadget_phases(
            gadget=self.gadget,
            circuit=circuit,
            unified_alloc=unified_alloc,
            ctx=ctx,
            builders=builders,
            alloc_dict=alloc,
            use_hybrid_decoding=use_hybrid,
            pre_gadget_meas=orchestrator_pre_meas,
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
        #
        # For hierarchical codes, crossing detectors now cover ALL
        # stabilizers (inner + outer) via get_last_measurement_indices().
        # The standard emit_scheduled_rounds path handles everything.
        if phase_result.crossing_handled:
            post_gadget_pre_meas = None
        else:
            post_gadget_pre_meas = pre_gadget_meas
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

        # Save manually emitted OBSERVABLE_INCLUDE as fallback before stripping.
        # Some gadgets (e.g. KnillEC) produce correct observables via Heisenberg
        # frame derivation, but discover_observables() cannot rediscover them for
        # complex multi-block teleportation protocols.
        saved_obs_instructions = [
            inst for inst in circuit.flattened()
            if inst.name == "OBSERVABLE_INCLUDE"
        ]

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
        elif saved_obs_instructions:
            # Fallback: re-emit the manually derived observable.
            # The manual observable was produced by the Heisenberg frame
            # derivation pipeline and is correct; auto-discovery just
            # could not rediscover it for this circuit topology.
            for inst in saved_obs_instructions:
                bare.append(inst)

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
            d_inner=self.d_inner,
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
