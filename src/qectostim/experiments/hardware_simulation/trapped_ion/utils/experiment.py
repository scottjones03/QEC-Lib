"""
Trapped-ion hardware experiment.

Implements :class:`HardwareSimulator` for QCCD trapped-ion hardware,
tying together architecture, compiler, noise injection, and decoding.

The simulation pipeline follows the ``HardwareSimulator`` contract:

1. ``build_ideal_circuit()`` — CSSMemoryExperiment or
   FaultTolerantGadgetExperiment depending on whether a gadget is
   supplied.
2. ``compile()`` — ``TrappedIonCompiler`` pipeline.
3. ``apply_hardware_noise(circuit)`` — builds an ``ExecutionPlan``
   then delegates to ``TrappedIonNoiseModel.apply_with_plan()``.
4. ``simulate()`` — sample + decode → ``HardwareSimulationResult``.

Also contains module-level electrode/DAC helper constants and the
multiprocessing worker functions ``process_circuit`` and
``process_circuit_wise_arch``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Dict,
    Any,
)

import numpy as np
from qectostim.decoders.base import Decoder
import stim

from qectostim.experiments.hardware_simulation.base import (
    HardwareSimulator,
    HardwareSimulationResult,
)
from qectostim.experiments.memory import CSSMemoryExperiment

from .qccd_nodes import Ion, QCCDWiseArch
from .qccd_operations import Operation
from .qccd_operations_on_qubits import QubitOperation
from .qccd_arch import QCCDArch
from .trapped_ion_compiler import TrappedIonCompiler
from .execution_planner import ExecutionPlanner, TrappedIonExecutionPlanner
from .noise import TrappedIonNoiseModel
from .physics import CalibrationConstants, DEFAULT_CALIBRATION

from ..compiler.qccd_parallelisation import paralleliseOperationsWithBarriers
from ..compiler.qccd_ion_routing import ionRouting
from ..compiler.qccd_WISE_ion_route import ionRoutingWISEArch
from ..compiler.routing_config import WISERoutingConfig
from qectostim.experiments.ft_gadget_experiment import (
    FaultTolerantGadgetExperiment,
)

logger = logging.getLogger(__name__)

class TrappedIonExperiment(HardwareSimulator):
    """QEC experiment on trapped-ion QCCD hardware.

    Experiment for trapped ion hardware.
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
                │  compile(circuit)         │  TrappedIonCompiler.compile()
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
    

    Simulates a QEC experiment (repeated stabiliser measurements)
    with realistic noise modelling including idle dephasing, gate
    errors, and transport noise.

    The simulation pipeline follows :class:`HardwareSimulator`:

    1. ``build_ideal_circuit()`` → stim circuit via
       :class:`CSSMemoryExperiment` (gadget=None) or
       :class:`FaultTolerantGadgetExperiment` (gadget≠None).
    2. ``compile()`` → :class:`TrappedIonCompiler` pipeline.
    3. ``apply_hardware_noise(circuit)`` → builds
       :class:`ExecutionPlan`, delegates to
       :meth:`TrappedIonNoiseModel.apply_with_plan`.
    4. ``simulate()`` → sample + decode →
       :class:`HardwareSimulationResult`.

    Parameters
    ----------
    code : Code
        Quantum error correction code.
    architecture : QCCDArch, optional
        QCCD architecture (or subclass).  For gadget experiments on
        WISE grids this can be omitted — the correct grid dimensions
        are computed automatically from the gadget metadata.
    compiler : TrappedIonCompiler, optional
        Compiler instance.  Created automatically when ``None``.
    hardware_noise : TrappedIonNoiseModel, optional
        Hardware noise model.  A default :class:`TrappedIonNoiseModel`
        is created if ``None``.
    rounds : int
        Number of stabiliser measurement rounds.
    basis : str
        Measurement basis (``"z"`` or ``"x"``).
    gadget : optional
        If supplied, ``build_ideal_circuit()`` generates a circuit
        via :class:`FaultTolerantGadgetExperiment`.
    error_scaling : float
        Scale factor for all error probabilities.
    trap_capacity : int
        Ions per trap (``k`` for :class:`QCCDWiseArch`).
        Only used when *architecture* is ``None``.
    add_spectators : bool
        Whether to add spectator ions.  Only used when
        *architecture* is ``None``.
    compact_clustering : bool
        Compact clustering flag.  Only used when
        *architecture* is ``None``.
    routing_config : WISERoutingConfig, optional
        Routing configuration forwarded to the compiler.  Only used
        when *compiler* is ``None``.
    """

    def __init__(
        self,
        code,
        architecture: QCCDArch = None,
        compiler: TrappedIonCompiler = None,
        hardware_noise: TrappedIonNoiseModel = None,
        rounds: int = 1,
        basis: str = "z",
        gadget=None,
        error_scaling: float = 1.0,
        # WISE auto-config (used when architecture is None)
        trap_capacity: int = 2,
        add_spectators: bool = True,
        compact_clustering: bool = True,
        routing_config: Optional["WISERoutingConfig"] = None,
        **kwargs,
    ):
        if hardware_noise is None:
            hardware_noise = TrappedIonNoiseModel()

        super().__init__(
            code=code,
            architecture=architecture,
            compiler=compiler,
            hardware_noise=hardware_noise,
            **kwargs,
        )
        self.rounds = rounds
        self.basis = basis
        self.gadget = gadget
        self.error_scaling = error_scaling

        # WISE auto-config params
        self._trap_capacity = trap_capacity
        self._add_spectators = add_spectators
        self._compact_clustering = compact_clustering
        self._routing_config = routing_config

        # Gadget metadata — populated by build_ideal_circuit() when
        # self.gadget is set, or can be injected externally.
        self._qec_metadata = None
        self._qubit_allocation = None
        self._ft_experiment = None

        # Cached state from the noise model walk
        self._last_mean_phys_x: float = 0.0
        self._last_mean_phys_z: float = 0.0


    def build_ideal_circuit(self) -> stim.Circuit:
        """Build the ideal stim circuit.

        If ``self.gadget`` is set, generates a fault-tolerant gadget
        experiment circuit; otherwise a standard CSS memory circuit.

        When a gadget is used, ``self._qec_metadata`` and
        ``self._qubit_allocation`` are populated automatically so that
        the downstream compile pipeline can dispatch to phase-aware
        gadget routing.
        """
        if self.gadget is not None:
            ft = FaultTolerantGadgetExperiment(
                codes=[self.code],
                gadget=self.gadget,
                noise_model=None,
                num_rounds_before=self.rounds,
                num_rounds_after=self.rounds,
            )
            circ = ft.to_stim()
            # Save metadata for the compile pipeline
            self._qec_metadata = ft.qec_metadata
            self._qubit_allocation = ft._unified_allocation
            self._ft_experiment = ft
            return circ

        mem = CSSMemoryExperiment(
            code=self.code,
            rounds=self.rounds,
            noise_model=None,
            basis=self.basis,
        )
        return mem.to_stim()

    # ------------------------------------------------------------------
    # Auto-configuration helpers
    # ------------------------------------------------------------------

    def _ensure_architecture(self) -> None:
        """Lazily build WISE architecture + compiler from gadget metadata.

        Called automatically by :meth:`compile` when ``architecture``
        was not provided at construction time.  Requires
        ``_qec_metadata`` and ``_qubit_allocation`` to have been
        populated (which :meth:`build_ideal_circuit` does).
        """
        if self.architecture is not None:
            # Already set — check if WISEArchitecture needs resolution
            from .architectures import WISEArchitecture as _WISEArch
            if (
                isinstance(self.architecture, _WISEArch)
                and not self.architecture.wise_config.is_resolved
                and self._qec_metadata is not None
            ):
                self._resolve_wise_grid()
            return

        if self._qec_metadata is None or self._qubit_allocation is None:
            raise RuntimeError(
                "Cannot auto-build architecture: gadget metadata not yet "
                "available.  Call build_ideal_circuit() first or pass an "
                "explicit architecture."
            )

        self._resolve_wise_grid()

    def _resolve_wise_grid(self) -> None:
        """Compute grid dims from gadget metadata and (re-)build arch + compiler."""
        from .gadget_routing import compute_gadget_grid_size
        from .architectures import WISEArchitecture as _WISEArch

        k = self._trap_capacity
        m, n = compute_gadget_grid_size(
            self._qec_metadata, self._qubit_allocation, k,
        )
        logger.info("Auto-computed WISE grid: m=%d, n=%d, k=%d", m, n, k)

        wise_cfg = QCCDWiseArch(m=m, n=n, k=k)
        self.architecture = _WISEArch(
            wise_config=wise_cfg,
            add_spectators=self._add_spectators,
            compact_clustering=self._compact_clustering,
        )

        # (Re-)create compiler for the new architecture
        self._compiler = TrappedIonCompiler(
            self.architecture,
            is_wise=True,
            wise_config=wise_cfg,
        )
        if self._routing_config is not None:
            self._compiler.routing_kwargs = dict(
                routing_config=self._routing_config,
            )

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self, circuit=None):
        """Compile a circuit to hardware.

        Overrides the base :meth:`HardwareSimulator.compile` to
        forward gadget metadata (``qec_metadata``, ``gadget``,
        ``qubit_allocation``) through the compilation pipeline so that
        :meth:`TrappedIonCompiler.map_qubits` and
        :meth:`TrappedIonCompiler.route` can dispatch to per-block
        topology building and phase-aware gadget routing automatically.

        When no *architecture* was provided at construction time and
        a *gadget* is set, the correct WISE grid dimensions are
        computed automatically from the gadget metadata before
        compiling.

        Parameters
        ----------
        circuit : stim.Circuit, optional
            Circuit to compile.  If ``None``, calls
            ``build_ideal_circuit()``.

        Returns
        -------
        CompiledCircuit
            Fully compiled circuit.
        """
        if circuit is None:
            circuit = self.build_ideal_circuit()

        # Ensure architecture is resolved (may auto-build for gadgets)
        self._ensure_architecture()

        # Apply pre-compile hook
        circuit = self.pre_compile(circuit)

        # Build extra metadata dict for gadget experiments
        _qec_meta = self._qec_metadata
        extra_meta = None
        if _qec_meta is not None:
            extra_meta = {}
            if self.gadget is not None:
                extra_meta["gadget"] = self.gadget
            if self._qubit_allocation is not None:
                extra_meta["qubit_allocation"] = self._qubit_allocation

        self._compiled = self.compiler.compile(
            circuit,
            qec_metadata=_qec_meta,
            extra_native_metadata=extra_meta,
        )

        # Apply post-compile hook
        self._compiled = self.post_compile(self._compiled)

        return self._compiled

    def _create_default_compiler(self) -> TrappedIonCompiler:
        """Create the default compiler for the current architecture.

        Auto-detects WISE mode when architecture is a
        :class:`WISEArchitecture`.
        """
        from .architectures import WISEArchitecture as _WISEArch

        if isinstance(self.architecture, _WISEArch):
            compiler = TrappedIonCompiler(
                self.architecture,
                is_wise=True,
                wise_config=self.architecture.wise_config,
            )
            if self._routing_config is not None:
                compiler.routing_kwargs = dict(
                    routing_config=self._routing_config,
                )
            return compiler
        return TrappedIonCompiler(self.architecture)

    def apply_hardware_noise(self) -> stim.Circuit:
        """Apply QCCD noise via :class:`TrappedIonNoiseModel`.

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
        Builds an :class:`ExecutionPlan` from the compiled operations
        and delegates to ``self.hardware_noise.apply_with_plan()``.

        If no compiled operations are available (e.g. the experiment
        was constructed for a standalone ``simulate_inline`` call)
        falls back to the calibration-based ``apply()`` path.
        """
        if self._compiled is None:
            raise RuntimeError("No compiled circuit available for noise application")

        planner = TrappedIonExecutionPlanner()
        plan = planner.create_plan(self._compiled)
        ideal = self._compiled.original_circuit
        noisy_circuit = self.hardware_noise.apply_with_plan(ideal, plan)
        return noisy_circuit

    def simulate(
        self,
        decoder: Decoder,
        num_shots: int = 100_000,
    ) -> HardwareSimulationResult:
        """Run inline noise simulation.

        Returns
        -------
        HardwareSimulationResult
            Simulation results.
        """
        self.compile()  # build_ideal_circuit + auto-arch + compile
        noisy_circuit = self.apply_hardware_noise()

        # Sample
        sampler = noisy_circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            num_shots, separate_observables=True
        )

        # Decode
        predictions = decoder.decode_batch(detection_events)
        num_errors = int(np.sum(np.any(predictions != observable_flips, axis=1)))

        return HardwareSimulationResult(
            logical_error_rate=num_errors / num_shots,
            num_shots=num_shots,
            num_errors=num_errors,
            compilation_metrics=self._compiled.compute_metrics() if hasattr(self._compiled, 'compute_metrics') else {},
            simulation_metrics={},
            decoder_used=decoder.__class__.__name__,
        )


# =========================================================================
# Constants for electrode / DAC calculations
# =========================================================================

NDE_LZ = 10
NDE_JZ = 20
NSE_Z = 10


# =========================================================================
# Multiprocessing worker functions
# =========================================================================


def process_circuit(
    distance: int,
    capacity: int,
    gate_improvements: Sequence[float],
    num_shots: int,
) -> Dict[str, Any]:
    """Multiprocessing worker: augmented-grid experiment.

    Creates a surface-code circuit at the given distance, builds an
    augmented-grid architecture, routes, simulates for each gate
    improvement factor, and returns a results dict.
    """
    from multiprocessing import get_logger as _get_logger
    from .architectures import AugmentedGridArchitecture

    logger = _get_logger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        os.path.join(
            tempfile.gettempdir(), f"process_log_{os.getpid()}.txt"
        )
    )
    formatter = logging.Formatter(
        "%(processName)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(
        f"Starting circuit generation for distance {distance}"
        f" and capacity {capacity}"
    )

    ideal = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=1,
        distance=distance,
    )
    nqubitsNeeded = 2 * distance**2 - 1
    nrowsNeeded = int(np.sqrt(nqubitsNeeded)) + 2

    logger.info(
        f"Processing circuit with {nqubitsNeeded} qubits"
        f" and {nrowsNeeded} rows"
    )

    arch = AugmentedGridArchitecture(
        trap_capacity=capacity,
        rows=nrowsNeeded,
        cols=nrowsNeeded,
    )
    compiler = TrappedIonCompiler(arch, is_wise=False)
    native = compiler.decompose_to_native(ideal)

    arch.build_topology(
        compiler.measurement_ions,
        compiler.data_ions,
        compiler.ion_mapping,
    )
    arch.refreshGraph()

    instructions = native.operations
    allOps, barriers = ionRouting(arch, instructions, capacity)
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)

    results: Dict[str, Any] = {
        "ElapsedTime": {},
        "Operations": {},
        "MeanConcurrency": {},
        "QubitOperations": {},
        "LogicalErrorRates": {},
        "PhysicalZErrorRates": {},
        "PhysicalXErrorRates": {},
        "Electrodes": {},
        "DACs": {},
    }

    label = "Forwarding"
    logger.info(
        f"Processing operations using {label} for distance {distance}"
        f" and capacity {capacity}"
    )

    # Build a circuitString function for simulate_inline
    def _circuit_string_fn(include_annotation=False):
        instructions_raw = (
            ideal.flattened()
            .decomposed()
            .without_noise()
            .__str__()
            .splitlines()
        )
        newInstructions: List[str] = []
        toMoves: List = []
        for instr in instructions_raw:
            qubits = instr.rsplit(" ")[1:]
            if instr.startswith("DETECTOR") or instr.startswith("TICK") or instr.startswith("OBSERVABLE"):
                if include_annotation:
                    newInstructions.append(instr)
                continue
            elif instr[0] in ("R", "H", "M"):
                for qubit in qubits:
                    newInstructions.append(f"{instr[0]} {qubit}")
            elif any(instr.startswith(s) for s in stim.gate_data("cnot").aliases):
                for j in range(int(len(qubits) / 2)):
                    newInstructions.append(f"CNOT {qubits[2*j]} {qubits[2*j+1]}")
                toMoves.append([])
                newInstructions.append("BARRIER")
            elif any(instr.startswith(s) for s in stim.gate_data("cz").aliases):
                for j in range(int(len(qubits) / 2)):
                    newInstructions.append(f"CZ {qubits[2*j]} {qubits[2*j+1]}")
                toMoves.append([])
                newInstructions.append("BARRIER")
            else:
                newInstructions.append(instr)
        return newInstructions, toMoves

    experiment = TrappedIonExperiment.__new__(TrappedIonExperiment)

    logicalErrors = []
    physicalZErrors = []
    physicalXErrors = []
    for gate_improvement in gate_improvements:
        logicalError, physicalXError, physicalZError = (
            experiment.simulate_inline(
                allOps,
                compiler.ion_mapping,
                _circuit_string_fn,
                num_shots=num_shots,
                error_scaling=gate_improvement,
                isWISEArch=False,
            )
        )
        logicalErrors.append(logicalError)
        physicalZErrors.append(physicalZError)
        physicalXErrors.append(physicalXError)

    logger.info(
        f"Simulated {label} method with gate improvements"
        f" for distance {distance}, capacity {capacity}"
    )

    for op in parallelOpsMap.values():
        op.calculateOperationTime()
        op.calculateFidelity()

    arch.resetArrangement()
    arch.refreshGraph()

    results["Capacity"] = capacity
    results["Distance"] = distance
    results["ElapsedTime"][label] = max(parallelOpsMap.keys())
    results["Operations"][label] = len(allOps)
    results["MeanConcurrency"][label] = np.mean(
        [len(op.operations) for op in parallelOpsMap.values()]
    )
    results["QubitOperations"][label] = len(instructions)
    results["LogicalErrorRates"][label] = logicalErrors
    results["PhysicalZErrorRates"][label] = physicalZErrors
    results["PhysicalXErrorRates"][label] = physicalXErrors

    from .qccd_nodes import Trap, Junction

    trapSet = set()
    junctionSet = set()
    for op in allOps:
        for c in op.involvedComponents:
            if isinstance(c, Trap):
                trapSet.add(c)
            elif isinstance(c, Junction):
                junctionSet.add(c)

    Njz = len(junctionSet)
    Nlz = len(trapSet) * capacity

    Nde = NDE_LZ * Nlz + NDE_JZ * Njz
    Nse = NSE_Z * (Njz + Nlz)

    Num_electrodes = Nde + Nse
    Num_DACs = Num_electrodes
    results["DACs"][label] = Num_DACs
    results["Electrodes"][label] = Num_electrodes

    logger.info(f"{distance} {capacity} {label} = {results}")
    logger.info(
        f"Finished processing for distance {distance}"
        f" and capacity {capacity}"
    )
    return results


def process_circuit_wise_arch(
    distance: int,
    capacity: int,
    gate_improvements: Sequence[float],
    num_shots: int,
) -> Dict[str, Any]:
    """Multiprocessing worker: WISE architecture experiment."""
    from multiprocessing import get_logger as _get_logger
    from .architectures import WISEArchitecture

    logger = _get_logger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("process_log_wise.txt")
    formatter = logging.Formatter(
        "%(processName)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(
        f"Starting circuit generation for distance {distance}"
        f" and capacity {capacity}"
    )

    ideal = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=1,
        distance=distance,
    )
    nqubitsNeeded = 2 * distance**2 - 1
    nrowsNeeded = int(np.sqrt(nqubitsNeeded)) + 2

    logger.info(
        f"Processing circuit with {nqubitsNeeded} qubits"
        f" and {nrowsNeeded} rows"
    )

    wiseArch = QCCDWiseArch(
        m=int(np.sqrt(capacity * nqubitsNeeded / 2)) + 1,
        n=int(np.sqrt(2 * nqubitsNeeded / capacity)) + 1,
        k=capacity,
    )
    arch = WISEArchitecture(wise_config=wiseArch)
    compiler = TrappedIonCompiler(
        arch, is_wise=True, wise_config=wiseArch
    )
    native = compiler.decompose_to_native(ideal)

    arch.build_topology(
        compiler.measurement_ions,
        compiler.data_ions,
        compiler.ion_mapping,
    )
    arch.refreshGraph()

    instructions = native.operations
    allOps, barriers = ionRoutingWISEArch(
        arch, wiseArch, instructions
    )
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)

    results: Dict[str, Any] = {
        "ElapsedTime": {},
        "Operations": {},
        "MeanConcurrency": {},
        "QubitOperations": {},
        "LogicalErrorRates": {},
        "PhysicalZErrorRates": {},
        "PhysicalXErrorRates": {},
        "Electrodes": {},
        "DACs": {},
    }

    label = "Forwarding"
    logger.info(
        f"Processing operations using {label} for distance {distance}"
        f" and capacity {capacity}"
    )

    # Build circuit string function for simulate_inline
    def _circuit_string_fn(include_annotation=False):
        instructions_raw = (
            ideal.flattened()
            .decomposed()
            .without_noise()
            .__str__()
            .splitlines()
        )
        newInstructions: List[str] = []
        toMoves: List = []
        for instr in instructions_raw:
            qubits = instr.rsplit(" ")[1:]
            if instr.startswith("DETECTOR") or instr.startswith("TICK") or instr.startswith("OBSERVABLE"):
                if include_annotation:
                    newInstructions.append(instr)
                continue
            elif instr[0] in ("R", "H", "M"):
                for qubit in qubits:
                    newInstructions.append(f"{instr[0]} {qubit}")
            elif any(instr.startswith(s) for s in stim.gate_data("cnot").aliases):
                for j in range(int(len(qubits) / 2)):
                    newInstructions.append(f"CNOT {qubits[2*j]} {qubits[2*j+1]}")
                toMoves.append([])
                newInstructions.append("BARRIER")
            elif any(instr.startswith(s) for s in stim.gate_data("cz").aliases):
                for j in range(int(len(qubits) / 2)):
                    newInstructions.append(f"CZ {qubits[2*j]} {qubits[2*j+1]}")
                toMoves.append([])
                newInstructions.append("BARRIER")
            else:
                newInstructions.append(instr)
        return newInstructions, toMoves

    experiment = TrappedIonExperiment.__new__(TrappedIonExperiment)

    logicalErrors = []
    physicalZErrors = []
    physicalXErrors = []
    for gate_improvement in gate_improvements:
        logicalError, physicalXError, physicalZError = (
            experiment.simulate_inline(
                allOps,
                compiler.ion_mapping,
                _circuit_string_fn,
                num_shots=num_shots,
                error_scaling=gate_improvement,
                isWISEArch=True,
            )
        )
        logicalErrors.append(logicalError)
        physicalZErrors.append(physicalZError)
        physicalXErrors.append(physicalXError)

    logger.info(
        f"Simulated {label} method with gate improvements"
        f" for distance {distance}, capacity {capacity}"
    )

    for op in parallelOpsMap.values():
        op.calculateOperationTime()
        op.calculateFidelity()

    arch.resetArrangement()
    arch.refreshGraph()

    results["Capacity"] = capacity
    results["Distance"] = distance
    results["ElapsedTime"][label] = max(parallelOpsMap.keys())
    results["Operations"][label] = len(allOps)
    results["MeanConcurrency"][label] = np.mean(
        [len(op.operations) for op in parallelOpsMap.values()]
    )
    results["QubitOperations"][label] = len(instructions)
    results["LogicalErrorRates"][label] = logicalErrors
    results["PhysicalZErrorRates"][label] = physicalZErrors
    results["PhysicalXErrorRates"][label] = physicalXErrors

    Njz = np.ceil(nqubitsNeeded / capacity)
    Nlz = nqubitsNeeded - Njz

    Nde = NDE_LZ * Nlz + NDE_JZ * Njz
    Nse = NSE_Z * (Njz + Nlz)

    Num_electrodes = int(Nde + Nse)
    Num_DACs = int(min(100, Nde) + np.ceil(Nse / 100))
    results["DACs"][label] = Num_DACs
    results["Electrodes"][label] = Num_electrodes

    logger.info(f"{distance} {capacity} {label} = {results}")
    logger.info(
        f"Finished processing for distance {distance}"
        f" and capacity {capacity}"
    )
    return results
