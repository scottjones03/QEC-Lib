# src/qectostim/experiments/hardware_simulation/trapped_ion/experiments.py
"""
Trapped ion specific experiments.

Provides experiment classes for trapped ion hardware simulation,
integrating QEC codes with hardware-aware compilation and noise.
"""
from __future__ import annotations

import logging

from typing import (
    Dict,
    List,
    Optional,
    Any,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import stim
import numpy as np

from qectostim.experiments.hardware_simulation.trapped_ion.simulator import (
    TrappedIonSimulator,
)
from qectostim.experiments.hardware_simulation.base import HardwareSimulationResult
from qectostim.codes.abstract_code import Code
from qectostim.codes.abstract_css import CSSCode
from qectostim.noise.models import NoiseModel

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
        TrappedIonArchitecture,
    )
    from qectostim.experiments.hardware_simulation.trapped_ion.compiler import (
        TrappedIonCompiler,
    )
    from qectostim.experiments.hardware_simulation.trapped_ion.noise import (
        TrappedIonNoiseModel,
    )
    from qectostim.gadgets.base import Gadget


_logger = logging.getLogger(__name__)


class TrappedIonExperiment(TrappedIonSimulator):
    """Experiment for trapped ion hardware.
 User entry point
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │  TrappedIonExperiment(code, gadget, arch, hardware_noise)                   │
 │     .simulate(num_shots)                                                    │
 └───────────────────────────┬──────────────────────────────────────────────────┘
                             │
                ┌────────────▼──────────────┐
                │  build_ideal_circuit()    │  gadget=None → CSSMemoryExperiment
                │  → stim.Circuit (ideal)   │  gadget≠None → FaultTolerantGadgetExperiment
                │                           │  ← already works via experiments/memory.py
                └────────────┬──────────────┘
                             │
                ┌────────────▼──────────────┐
                │  compile(circuit)         │  HardwareCompiler.compile()
                │  ┌──────────────────────┐ │
                │  │ decompose_to_native()│─┼─► NativeCircuit  (stim ops → MS+R{X,Y,Z})
                │  ├──────────────────────┤ │
                │  │ map_qubits()         │─┼─► MappedCircuit  (logical → physical ion IDs)
                │  ├──────────────────────┤ │
                │  │ route()              │─┼─► RoutedCircuit  (ion shuttling / SAT WISE)
                │  ├──────────────────────┤ │
                │  │ schedule()           │─┼─► ScheduledCircuit (parallel batches+timing)
                │  └──────────────────────┘ │
                │  → CompiledCircuit        │  (.mapping, .scheduled, .metrics)
                └────────────┬──────────────┘
                             │
                ┌────────────▼──────────────┐
                │  ExecutionPlanner         │  Extracts idle intervals, gate swaps,
                │  .create_plan()           │  per-ion dephasing from schedule
                │  → ExecutionPlan          │  (operations, idle_intervals, gate_swaps,
                │                           │   total_duration, num_operations)
                └────────────┬──────────────┘
                             │
                ┌────────────▼──────────────┐
                │  apply_with_plan()        │  Walks ORIGINAL stim instructions, injects:
                │  TrappedIonNoiseModel     │    1. Z_ERROR     (idle dephasing before gate)
                │                           │    2. DEPOLARIZE2  (gate swap transport error)
                │                           │    3. Original instruction
                │                           │    4. DEPOLARIZE1/2, X_ERROR (gate infidelity)
                │  → stim.Circuit (noisy)   │  ← Preserves DETECTOR/OBSERVABLE/REPEAT
                └────────────┬──────────────┘
                             │
                ┌────────────▼──────────────┐
                │  Decode & sample          │  compile_detector_sampler()
                │  decoders/ (PyMatching,   │  → sample(num_shots)
                │   Fusion, UF, BP+OSD …)   │  → decode → logical error counts
                │  → HardwareSimResult      │  (logical_error_rate, physical_errors,
                │                           │   compilation_metrics, simulation_metrics)
                └───────────────────────────┘
    
    Simulates a QEC experiment (repeated stabilizer measurements)
    on trapped ion hardware with realistic noise modeling.
    
    This experiment:
    1. Initializes logical qubit in a chosen state
    2. Performs `rounds` of stabilizer measurements
    3. Measures data qubits to extract syndrome and logical outcome
    4. Applies hardware-aware noise (idle dephasing, gate errors, transport)
    
    Parameters
    ----------
    code : Code
        The quantum error correction code.
    architecture : TrappedIonArchitecture
        Hardware architecture (QCCD, linear chain, etc.).
    gadget : Optional[Gadget]
        Logical gate gadget.  If None (default), runs a memory experiment
        (repeated stabilizer measurements).  If provided, runs a
        fault-tolerant gadget experiment via FaultTolerantGadgetExperiment.
    rounds : int
        Number of stabilizer measurement rounds (memory), or rounds
        before/after the gadget (FT gadget experiment).
    compiler : Optional[TrappedIonCompiler]
        Compiler for hardware. Uses LinearChainCompiler if None.
    hardware_noise : Optional[TrappedIonNoiseModel]
        Trapped ion noise model.
    noise_model : Optional[NoiseModel]
        Additional circuit-level noise.
    logical_qubit : int
        Which logical qubit to use (for multi-qubit codes).
    initial_state : str
        Initial logical state: "0", "1", "+", "-".
    metadata : Optional[Dict[str, Any]]
        Additional experiment metadata.
        
    Examples
    --------
    Memory experiment (gadget=None):
    
    >>> from qectostim.codes.surface import RotatedSurfaceCode
    >>> from qectostim.experiments.hardware_simulation.trapped_ion import (
    ...     TrappedIonExperiment,
    ...     TrappedIonNoiseModel,
    ...     LinearChainArchitecture,
    ... )
    >>>
    >>> code = RotatedSurfaceCode(3)
    >>> arch = LinearChainArchitecture(num_ions=17)
    >>> noise = TrappedIonNoiseModel()
    >>>
    >>> exp = TrappedIonExperiment(
    ...     code=code,
    ...     architecture=arch,
    ...     gadget=None,
    ...     rounds=3,
    ...     hardware_noise=noise,
    ... )
    >>> result = exp.simulate(num_shots=10000)
    >>> print(f"Logical error rate: {result.logical_error_rate:.2e}")
    
    FT gadget experiment:
    
    >>> from qectostim.gadgets import TransversalCNOT
    >>> gadget = TransversalCNOT()
    >>> exp = TrappedIonExperiment(
    ...     code=code,
    ...     architecture=arch,
    ...     gadget=gadget,
    ...     rounds=3,
    ...     hardware_noise=noise,
    ... )
    """
    
    def __init__(
        self,
        code: Code,
        architecture: "TrappedIonArchitecture",
        gadget: Optional["Gadget"] = None,
        rounds: int = 1,
        compiler: Optional["TrappedIonCompiler"] = None,
        hardware_noise: Optional["TrappedIonNoiseModel"] = None,
        noise_model: Optional[NoiseModel] = None,
        logical_qubit: int = 0,
        initial_state: str = "0",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            code=code,
            architecture=architecture,
            compiler=compiler,
            hardware_noise=hardware_noise,
            noise_model=noise_model,
            metadata=metadata,
        )
        self.gadget = gadget
        self.rounds = rounds
        self.logical_qubit = logical_qubit
        self.initial_state = initial_state
        self.operation = "gadget" if gadget is not None else "memory"
        
        # Cache for ideal circuit
        self._ideal_circuit: Optional[stim.Circuit] = None
        # Rich QEC metadata captured from internal experiment builder
        self._qec_metadata: Optional[Any] = None
    
    def build_ideal_circuit(self) -> stim.Circuit:
        """Build the ideal (noise-free) circuit for this experiment.
        
        Dispatches based on whether a gadget is provided:
        
        - ``gadget=None``:  Memory experiment — repeated stabilizer
          measurement rounds.  Uses the code's built-in circuit
          generation, then ``CSSMemoryExperiment`` as a fallback.
        - ``gadget≠None``:  Fault-tolerant gadget experiment — delegates
          to ``FaultTolerantGadgetExperiment`` which builds the full
          Memory → Gadget → Memory → Measure pipeline.
        
        Returns
        -------
        stim.Circuit
            Ideal Stim circuit (no noise applied).
        """
        if self._ideal_circuit is not None:
            return self._ideal_circuit
        
        if self.gadget is not None:
            self._ideal_circuit = self._build_gadget_circuit()
        else:
            self._ideal_circuit = self._build_memory_circuit()
        
        return self._ideal_circuit
    
    @property
    def qec_metadata(self) -> Optional[Any]:
        """Rich QEC metadata for the hardware compiler.

        Available after ``build_ideal_circuit()`` has been called.
        Returns a :class:`QECMetadata` instance populated from the
        builder and code, or ``None`` if the ideal circuit was built
        via a path that does not produce metadata (e.g. ``code.to_stim()``).
        """
        return self._qec_metadata

    @property
    def qubit_roles(self) -> Dict[int, str]:
        """Map of qubit index → role ('D', 'X', 'Z', 'P') from QEC metadata."""
        meta = self._qec_metadata
        if meta is None:
            return {}
        return dict(meta.qubit_roles)

    @property
    def data_qubit_indices(self) -> frozenset:
        """Indices of data qubits from QEC metadata."""
        meta = self._qec_metadata
        if meta is None:
            return frozenset()
        return meta.data_qubit_indices

    @property
    def ancilla_qubit_indices(self) -> frozenset:
        """Indices of ancilla qubits from QEC metadata."""
        meta = self._qec_metadata
        if meta is None:
            return frozenset()
        return meta.ancilla_qubit_indices

    @property
    def preparation_qubit_indices(self) -> frozenset:
        """Indices of preparation qubits from QEC metadata."""
        meta = self._qec_metadata
        if meta is None:
            return frozenset()
        return meta.preparation_qubit_indices

    # ------------------------------------------------------------------
    # Memory experiment path (gadget=None)
    # ------------------------------------------------------------------
    
    def _build_memory_circuit(self) -> stim.Circuit:
        """Build a memory experiment circuit (repeated stabiliser rounds).
        
        Tries, in order:
        1. ``code.to_stim(rounds=…)``  — Stim's built-in generators
           (works for RotatedSurfaceCode, SurfaceCode, etc.)
        2. ``code.memory_experiment(rounds=…)``  — custom builder
        3. ``CSSMemoryExperiment``  — generic CSS fallback
        4. ``StabilizerMemoryExperiment``  — generic stabiliser fallback
        
        Returns
        -------
        stim.Circuit
            Ideal memory experiment circuit.
        
        Raises
        ------
        NotImplementedError
            If none of the strategies produce a valid circuit.
        """
        # Strategy 1: Code's own to_stim() (e.g. RotatedSurfaceCode wraps
        # stim.Circuit.generated("surface_code:rotated_memory_z", …)).
        if hasattr(self.code, 'to_stim'):
            try:
                circuit = self.code.to_stim(
                    rounds=self.rounds,
                    after_clifford_depolarization=0.0,
                    before_measure_flip_probability=0.0,
                )
                if circuit is not None and len(circuit) > 0:
                    return circuit
            except (TypeError, ValueError):
                pass
        
        # Strategy 2: Code's memory_experiment() builder.
        if hasattr(self.code, 'memory_experiment'):
            try:
                circuit = self.code.memory_experiment(rounds=self.rounds)
                if circuit is not None and len(circuit) > 0:
                    return circuit
            except (TypeError, ValueError, AttributeError):
                pass
        
        # Strategy 3: Generic CSS memory circuit.
        if isinstance(self.code, CSSCode):
            return self._build_css_memory_circuit()
        
        # Strategy 4: Generic stabiliser memory circuit (non-CSS codes).
        return self._build_stabilizer_memory_circuit()
    
    def _build_css_memory_circuit(self) -> stim.Circuit:
        """Build a memory circuit for CSS codes via CSSMemoryExperiment."""
        from qectostim.experiments.memory import CSSMemoryExperiment
        
        # R8 fix: initial_state is a str, compare correctly
        basis = self._initial_state_to_basis()
        
        css_exp = CSSMemoryExperiment(
            code=self.code,
            noise_model=None,  # hardware noise applied later
            rounds=self.rounds,
            basis=basis,
        )
        circuit = css_exp.to_stim()
        # Capture QECMetadata before the local experiment is discarded
        try:
            self._qec_metadata = css_exp.qec_metadata
        except (RuntimeError, AttributeError):
            pass
        return circuit
    
    def _build_stabilizer_memory_circuit(self) -> stim.Circuit:
        """Build a memory circuit for general stabiliser codes.
        
        Falls back to StabilizerMemoryExperiment for codes that are
        not CSS (e.g. non-CSS stabiliser codes).
        
        Raises
        ------
        NotImplementedError
            If no suitable experiment builder is available.
        """
        try:
            from qectostim.experiments.memory import StabilizerMemoryExperiment
            
            basis = self._initial_state_to_basis()
            stab_exp = StabilizerMemoryExperiment(
                code=self.code,
                noise_model=None,
                rounds=self.rounds,
                basis=basis,
            )
            return stab_exp.to_stim()
        except (ImportError, TypeError, ValueError, AttributeError) as exc:
            raise NotImplementedError(
                f"Cannot build ideal circuit for {self.code.__class__.__name__}. "
                f"Code must implement to_stim(), memory_experiment(), or be a "
                f"CSSCode / StabilizerCode subclass. (inner error: {exc})"
            ) from exc
    
    # ------------------------------------------------------------------
    # FT gadget experiment path (gadget≠None)
    # ------------------------------------------------------------------
    
    def _build_gadget_circuit(self) -> stim.Circuit:
        """Build a fault-tolerant gadget experiment circuit.
        
        Delegates to FaultTolerantGadgetExperiment which implements the
        full Memory → Gadget → Memory → Measure TQEC pipeline.
        
        Returns
        -------
        stim.Circuit
            Ideal FT gadget experiment circuit.
        """
        from qectostim.experiments.ft_gadget_experiment import (
            FaultTolerantGadgetExperiment,
        )
        
        ft_exp = FaultTolerantGadgetExperiment(
            codes=[self.code],
            gadget=self.gadget,
            noise_model=None,  # hardware noise applied later
            num_rounds_before=self.rounds,
            num_rounds_after=self.rounds,
            metadata=self.metadata,
        )
        circuit = ft_exp.to_stim()
        # Capture QECMetadata before the local experiment is discarded
        try:
            self._qec_metadata = ft_exp.qec_metadata
        except (RuntimeError, AttributeError):
            pass
        return circuit
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    
    def _initial_state_to_basis(self) -> str:
        """Map initial_state string to measurement basis.
        
        Returns
        -------
        str
            "Z" or "X".
        """
        # "0" and "1" are Z-basis eigenstates
        # "+" and "-" are X-basis eigenstates
        if self.initial_state in ("0", "1", "Z", "z"):
            return "Z"
        elif self.initial_state in ("+", "-", "X", "x"):
            return "X"
        else:
            # Default to Z basis for unrecognised values
            return "Z"
    
    def _get_execution_plan(
        self,
        ideal: stim.Circuit,
    ) -> "ExecutionPlan":
        """Build an ExecutionPlan, using compile() if a compiler is set.
        
        Tries these strategies in order:
        1. Full compile → ``TrappedIonExecutionPlanner.plan_execution``
           (iterates the *Stim* circuit so instruction indices align with
           the noise model's walk; uses compiled timing/fidelity metadata)
        2. ``create_simple_execution_plan`` (heuristic fallback)
        
        Note: ``build_execution_plan_from_compiled`` is NOT used here
        because its instruction indices correspond to the *native*
        scheduled-operation space (e.g. 155 decomposed gates) rather
        than the Stim circuit's instruction space (e.g. 45 logical
        gates).  The noise model walks the Stim circuit directly, so
        the plan's indices MUST match.
        
        Caches the compiled result in ``self._compiled`` so that
        subsequent calls to ``to_stim()`` or ``compile()`` do not
        trigger a redundant (expensive) routing pass.
        
        Parameters
        ----------
        ideal : stim.Circuit
            The ideal (noise-free) circuit.
            
        Returns
        -------
        ExecutionPlan
        """
        from qectostim.experiments.hardware_simulation.trapped_ion.execution import (
            TrappedIonExecutionPlanner,
            create_simple_execution_plan,
        )
        
        if self.compiler is not None:
            try:
                # Reuse cached compilation when available
                compiled = getattr(self, '_compiled', None)
                if compiled is None:
                    compiled = self.compiler.compile(ideal)
                    self._compiled = compiled
                
                calibration = getattr(self.hardware_noise, 'calibration', None)
                planner = TrappedIonExecutionPlanner(
                    compiler=self.compiler,
                    calibration=calibration,
                )
                return planner.plan_execution(ideal, compiled=compiled)
            except Exception:
                # Fallback if compilation fails (e.g. unsupported gates)
                pass
        
        # Fallback: estimate timing without compilation
        return create_simple_execution_plan(ideal)
    
    def to_stim(self) -> stim.Circuit:
        """Generate Stim circuit with hardware noise.
        
        Pipeline:
        1. Build ideal circuit (DETECTOR/OBSERVABLE annotations intact)
        2. Compile to get execution metadata (timing, routing)
        3. Build ExecutionPlan from compilation
        4. apply_with_plan() injects noise into the ORIGINAL circuit
        
        Uses cached compilation from ``_get_execution_plan()`` when
        available to avoid redundant routing passes.
        
        Returns
        -------
        stim.Circuit
            Circuit with hardware noise applied.
        """
        # Get ideal circuit
        ideal = self.build_ideal_circuit()
        
        # Apply hardware noise using execution plan
        if self.hardware_noise is not None:
            # Reuse cached execution plan if available
            plan = getattr(self, '_cached_plan', None)
            if plan is None:
                plan = self._get_execution_plan(ideal)
            
            # Apply timing-aware noise
            if hasattr(self.hardware_noise, 'apply_with_plan'):
                circuit = self.hardware_noise.apply_with_plan(ideal, plan)
            else:
                circuit = self.hardware_noise.apply(ideal)
        else:
            circuit = ideal
        
        # NOTE: noise_model is NOT applied here — the base Experiment class
        # applies it in _run_correction_path() / _run_detection_path() after
        # calling to_stim().  Applying it here would cause double application.
        
        return circuit
    
    def simulate(
        self,
        num_shots: int = 10000,
        decoder_name: Optional[str] = None,
        gate_improvements: Optional[List[float]] = None,
        verbose: bool = True,
    ) -> HardwareSimulationResult:
        """Run hardware simulation with decoding.

        Parameters
        ----------
        num_shots : int
            Number of shots to simulate.
        decoder_name : Optional[str]
            Decoder to use.  If ``None``, auto-selects.
        gate_improvements : Optional[List[float]]
            List of gate-improvement (error-scaling) factors to sweep.
            Each value divides all gate error probabilities before
            simulation, so ``2.0`` means "gates are 2× better than
            calibrated".  When provided, the results dict contains
            ``LogicalErrorRates``, ``PhysicalXErrorRates``, and
            ``PhysicalZErrorRates`` lists (one entry per factor).
            When ``None``, a single simulation at the noise model's
            current ``error_scaling`` is run.
        verbose : bool
            If True, print progress messages during simulation.

        Returns
        -------
        HardwareSimulationResult
            Simulation results with logical error rate and optional
            sweep data in ``simulation_metrics``.
        """
        import time as _time

        def _log(msg: str) -> None:
            if verbose:
                _logger.info(msg)

        _t0 = _time.monotonic()

        # ------------------------------------------------------------------
        # Compile once — the compilation is noise-independent
        # ------------------------------------------------------------------
        # Invalidate any stale compilation cache so this simulate()
        # produces a fresh compilation for the current experiment config.
        self._compiled = None
        self._cached_plan = None

        _log("  [1/4] Building ideal circuit …")
        ideal = self.build_ideal_circuit()
        _log(f"         → {ideal.num_qubits}Q, {len(ideal)} instructions")

        plan = None
        compiled = None
        if self.hardware_noise is not None:
            _log("  [2/4] Building execution plan …")
            try:
                plan = self._get_execution_plan(ideal)
                # Cache the plan so to_stim() in run_decode() reuses it
                self._cached_plan = plan
                _log(f"         → {len(plan.operations)} operations, "
                     f"{plan.total_duration:.0f} µs")
            except Exception as exc:
                _log(f"         ⚠ skipped ({type(exc).__name__})")

        # Reuse cached compilation from _get_execution_plan() above
        compiled = getattr(self, '_compiled', None)
        if compiled is None:
            _log("  [3/4] Compiling (decompose → map → route → schedule) …")
            try:
                compiled = self.compile()
                _log(f"         → compiled OK")
            except Exception as exc:
                _log(f"         ⚠ compile failed ({type(exc).__name__}: {exc})")
        else:
            _log("  [3/4] Using cached compilation (from execution plan) … OK")

        # ------------------------------------------------------------------
        # Helper: run one (to_stim → run_decode) pass at a given scaling
        # ------------------------------------------------------------------
        original_scaling = (
            self.hardware_noise.error_scaling
            if self.hardware_noise is not None
            else 1.0
        )

        def _run_at_scaling(scaling: float) -> Dict[str, Any]:
            """Run to_stim + run_decode with a specific error_scaling."""
            if self.hardware_noise is not None:
                self.hardware_noise.error_scaling = scaling
            try:
                return self.run_decode(shots=num_shots, decoder_name=decoder_name)
            finally:
                # Always restore original scaling
                if self.hardware_noise is not None:
                    self.hardware_noise.error_scaling = original_scaling

        # ------------------------------------------------------------------
        # Sweep or single-point
        # ------------------------------------------------------------------
        _log(f"  [4/4] Decoding ({num_shots} shots) …")
        if gate_improvements is not None and len(gate_improvements) > 0:
            logical_error_rates: List[float] = []
            physical_x_error_rates: List[float] = []
            physical_z_error_rates: List[float] = []

            for i, gi in enumerate(gate_improvements, 1):
                _log(f"         scaling {i}/{len(gate_improvements)} "
                     f"(improvement={gi}) …")
                result = _run_at_scaling(gi)
                ler = result.get('logical_error_rate', 0.0) or 0.0
                logical_error_rates.append(float(ler))

                # Compute analytical physical X/Z error rates from the
                # execution plan, matching the old code's per-instruction
                # accumulation logic.
                px, pz = self._compute_physical_xz_error_rates(plan, gi)
                physical_x_error_rates.append(px)
                physical_z_error_rates.append(pz)

            # Use the first (baseline) result for the top-level fields
            primary_ler = logical_error_rates[0]
            primary_num_errors = int(primary_ler * num_shots)
        else:
            # Single point — use current noise model scaling
            decode_result = _run_at_scaling(original_scaling)
            primary_ler = (
                decode_result.get('logical_error_rate')
                or decode_result.get('error_rate', 0.0)
            )
            logical_errors_raw = decode_result.get('logical_errors')
            if logical_errors_raw is not None:
                primary_num_errors = int(np.sum(logical_errors_raw))
            else:
                primary_num_errors = decode_result.get('num_errors', 0)

            logical_error_rates = None
            physical_x_error_rates = None
            physical_z_error_rates = None

        # ------------------------------------------------------------------
        # Gather compilation / scheduling metrics
        # ------------------------------------------------------------------
        compilation_metrics: Dict[str, Any] = {
            "rounds": self.rounds,
            "num_qubits": self.code.n,
        }
        if compiled is not None:
            cm = compiled.compute_metrics()
            compilation_metrics.update(cm)

        simulation_metrics: Dict[str, Any] = {
            "total_duration_us": plan.total_duration if plan else 0.0,
            "num_operations": len(plan.operations) if plan else 0,
        }

        # WISE-style result keys (mirroring old pipeline output)
        if compiled is not None:
            cm = compiled.compute_metrics()
            simulation_metrics["ElapsedTime"] = cm.get(
                "duration_us", plan.total_duration if plan else 0.0) + cm.get(
                "reconfiguration_time_us", 0.0)
            simulation_metrics["Operations"] = cm.get("total_operations", 0)
            simulation_metrics["MeanConcurrency"] = cm.get("parallelism", 0.0)
            simulation_metrics["QubitOperations"] = cm.get("two_qubit_ops", 0)
            simulation_metrics["ReconfigurationTime"] = cm.get(
                "reconfiguration_time_us", 0.0)

        if logical_error_rates is not None:
            simulation_metrics["LogicalErrorRates"] = logical_error_rates
            simulation_metrics["PhysicalXErrorRates"] = physical_x_error_rates
            simulation_metrics["PhysicalZErrorRates"] = physical_z_error_rates
            simulation_metrics["GateImprovements"] = list(gate_improvements)

        decoder_used_key = decoder_name or "auto"

        _elapsed = _time.monotonic() - _t0
        _log(f"  ✅ Done in {_elapsed:.2f}s — LER={float(primary_ler):.4e}")

        return HardwareSimulationResult(
            logical_error_rate=float(primary_ler),
            num_shots=num_shots,
            num_errors=primary_num_errors,
            compilation_metrics=compilation_metrics,
            simulation_metrics=simulation_metrics,
            decoder_used=decoder_used_key,
        )
    
    def __repr__(self) -> str:
        return (
            f"TrappedIonExperiment("
            f"code={self.code.__class__.__name__}, "
            f"rounds={self.rounds}, "
            f"architecture={self.architecture.name!r})"
        )

    # ------------------------------------------------------------------
    # Analytical physical X / Z error rates
    # ------------------------------------------------------------------

    def _compute_physical_xz_error_rates(
        self,
        plan: Optional[Any],
        error_scaling: float,
    ) -> Tuple[float, float]:
        """Compute analytical physical X/Z error rates from execution plan.

        Mirrors the old ``qccd_circuit.simulate()`` accumulation:

        * **Dephasing** (idle Z_ERROR) → Z only
        * **Gate swap** (DEPOLARIZE2) → X/2 + Z/2
        * **Measurement / Reset** (X_ERROR) → X only
        * **1Q / 2Q gate infidelity** (DEPOLARIZE) → X/2 + Z/2

        The result is the **mean per-instruction** error rate, averaged
        over instructions that had any X (or Z) noise, exactly as the
        old code's ``meanPhysicalXError / numXGates``.

        Parameters
        ----------
        plan : Optional[ExecutionPlan]
            Execution plan with timing metadata.
        error_scaling : float
            Error-scaling (gate-improvement) factor.

        Returns
        -------
        Tuple[float, float]
            ``(mean_physical_x_error, mean_physical_z_error)``
        """
        if plan is None or not plan.operations:
            return (0.0, 0.0)

        total_x = 0.0
        total_z = 0.0
        num_x_gates = 0
        num_z_gates = 0

        for timing in plan.operations:
            instr_x = 0.0
            instr_z = 0.0
            gate = timing.gate_name.upper()

            # (A) Idle dephasing before this instruction → Z only
            for idle in plan.get_idle_before(timing.instruction_index):
                dur = idle.duration * getattr(timing, 'num_native_ops', 1)
                for ch in self.hardware_noise.idle_noise(idle.qubit, dur):
                    if ch.probability > 0:
                        instr_z += ch.probability

            # (B) Gate swap noise → X/2 + Z/2
            for swap_info in plan.get_swaps_for_instruction(
                timing.instruction_index
            ):
                for ch in self.hardware_noise.swap_noise(swap_info):
                    if ch.probability > 0:
                        instr_x += ch.probability / 2
                        instr_z += ch.probability / 2

            # (C) Gate infidelity
            infidelity = (1.0 - timing.fidelity) / error_scaling
            infidelity = min(infidelity, 0.5)

            if infidelity > 0:
                is_meas_reset = gate in (
                    "M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ",
                    "R", "RESET",
                )
                if is_meas_reset:
                    # Measurement / reset → X only
                    instr_x += infidelity
                else:
                    # 1Q / 2Q gate → X/2 + Z/2
                    instr_x += infidelity / 2
                    instr_z += infidelity / 2

            if instr_x > 0:
                total_x += instr_x
                num_x_gates += 1
            if instr_z > 0:
                total_z += instr_z
                num_z_gates += 1

        mean_x = total_x / num_x_gates if num_x_gates > 0 else 0.0
        mean_z = total_z / num_z_gates if num_z_gates > 0 else 0.0
        return (mean_x, mean_z)


class TrappedIonGadgetExperiment(TrappedIonSimulator):
    """Fault-tolerant gadget experiment for trapped ions.
    
    NOT YET IMPLEMENTED - placeholder for future gadget experiments
    (transversal gates, lattice surgery, etc.)
    """
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "TrappedIonGadgetExperiment not yet implemented. "
            "Use TrappedIonExperiment for memory experiments."
        )
    
    def build_ideal_circuit(self) -> stim.Circuit:
        raise NotImplementedError()
