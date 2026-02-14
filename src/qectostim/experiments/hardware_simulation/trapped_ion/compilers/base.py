# src/qectostim/experiments/hardware_simulation/trapped_ion/compilers/base.py
"""
Base trapped ion compiler with common gate decompositions.

Provides:
- DecomposedGate: A gate in a decomposition sequence
- TrappedIonCompiler: Abstract base for all trapped ion compilers

Architecture hierarchy:
- TrappedIonCompiler (ABC): Technology-specific base (MS gates, ion physics)
  - WISECompiler: WISE grid architecture with SAT-based routing
  - QCCDCompiler: General QCCD with split/merge/shuttle routing
"""
from __future__ import annotations

import logging
import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    TYPE_CHECKING,
)

import stim

from qectostim.experiments.hardware_simulation.core.compiler import (
    HardwareCompiler,
    CompilationPass,
    CompilationResult,
    CompilationStage,
    DecompositionPass,
    MappingPass,
    RoutingPass,
    SchedulingPass,
)
from qectostim.experiments.hardware_simulation.core.pipeline import (
    NativeCircuit,
    MappedCircuit,
    RoutedCircuit,
    ScheduledCircuit,
    ScheduledOperation,
    CircuitLayer,
    QubitMapping,
)
from qectostim.experiments.hardware_simulation.core.gates import (
    GateSpec,
    GateType,
)
from qectostim.experiments.hardware_simulation.core.operations import (
    PhysicalOperation,
    GateOperation,
    MeasurementOperation,
    ResetOperation,
    OperationBatch,
    OperationType,
)

from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
    DEFAULT_CALIBRATION as _CAL,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
        TrappedIonArchitecture,
        WISEArchitecture,
    )
    from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
        WISERoutingConfig,
    )


_logger = logging.getLogger(__name__)


# =============================================================================
# Gate Decomposition Data
# =============================================================================

# Standard angles
PI = math.pi
PI_2 = math.pi / 2
PI_4 = math.pi / 4


@dataclass
class DecomposedGate:
    """A gate in a decomposition sequence.
    
    Attributes
    ----------
    name : str
        Gate name (MS, RX, RY, RZ).
    qubits : Tuple[int, ...]
        Target qubits.
    params : Dict[str, float]
        Gate parameters (e.g., angle).
    """
    name: str
    qubits: Tuple[int, ...]
    params: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Abstract Base: TrappedIonCompiler
# =============================================================================

class TrappedIonCompiler(HardwareCompiler):
    """Abstract base compiler for trapped ion architectures.
    
    This class provides trapped-ion-specific functionality common to
    ALL trapped ion architectures:
    - MS gate decomposition recipes
    - Ion physics constraints
    - Native gate set definition
    
    Subclasses implement architecture-specific compilation:
    - WISECompiler: WISE grid with SAT-based optimal routing
    - LinearChainCompiler: Single chain, all-to-all connectivity
    - QCCDCompiler: General QCCD with T-junctions and shuttling
    
    This class is ABSTRACT - instantiate a concrete subclass.
    """
    
    # Native gate set for trapped ions
    NATIVE_GATES = ["MS", "RX", "RY", "RZ"]
    
    def __init__(
        self,
        architecture: "TrappedIonArchitecture",
        optimization_level: int = 1,
        use_global_rotations: bool = False,
    ):
        """Initialize the trapped ion compiler.
        
        Parameters
        ----------
        architecture : TrappedIonArchitecture
            Target trapped ion architecture.
        optimization_level : int
            0 = no optimization, 1 = basic, 2 = aggressive.
        use_global_rotations : bool
            If True, use global rotations where possible (all ions rotate).
        """
        super().__init__(architecture, optimization_level)
        self.use_global_rotations = use_global_rotations
        self._decomposition_cache: Dict[str, List[DecomposedGate]] = {}
    
    # =========================================================================
    # Common trapped-ion gate decompositions
    # =========================================================================
    
    def _decompose_cnot(self, control: int, target: int) -> List[DecomposedGate]:
        """Decompose CNOT to MS + single-qubit rotations.
        
        Uses the 1-MS decomposition from PRA 99, 022330, Fig 4:
            RY(ctl)  RX(ctl)  RX(tgt)  MS(ctl,tgt)  RY(ctl)
        """
        return [
            DecomposedGate("RY", (control,), {"angle": PI_2}),
            DecomposedGate("RX", (control,), {"angle": PI_2}),
            DecomposedGate("RX", (target,), {"angle": PI_2}),
            DecomposedGate("MS", (control, target), {"angle": PI_4}),
            DecomposedGate("RY", (control,), {"angle": PI_2}),
        ]
    
    def _decompose_h(self, qubit: int) -> List[DecomposedGate]:
        """Decompose Hadamard to rotations.

        Uses H = Ry(π/2) · Rx(π/2) from p.80 of the Figgatt thesis
        (https://iontrap.umd.edu/wp-content/uploads/2013/10/FiggattThesis.pdf).
        This matches the old pipeline's ``_parseCircuitString`` decomposition.
        """
        return [
            DecomposedGate("RY", (qubit,), {"angle": PI_2}),
            DecomposedGate("RX", (qubit,), {"angle": PI_2}),
        ]
    
    def _decompose_s(self, qubit: int) -> List[DecomposedGate]:
        """Decompose S gate. S = Rz(pi/2)"""
        return [DecomposedGate("RZ", (qubit,), {"angle": PI_2})]
    
    def _decompose_s_dag(self, qubit: int) -> List[DecomposedGate]:
        """Decompose S dagger gate. S_dag = Rz(-pi/2)"""
        return [DecomposedGate("RZ", (qubit,), {"angle": -PI_2})]
    
    def _decompose_t(self, qubit: int) -> List[DecomposedGate]:
        """Decompose T gate. T = Rz(pi/4)"""
        return [DecomposedGate("RZ", (qubit,), {"angle": PI_4})]
    
    def _decompose_t_dag(self, qubit: int) -> List[DecomposedGate]:
        """Decompose T dagger gate. T_dag = Rz(-pi/4)"""
        return [DecomposedGate("RZ", (qubit,), {"angle": -PI_4})]
    
    def _decompose_x(self, qubit: int) -> List[DecomposedGate]:
        """Decompose X gate. X = Rx(pi)"""
        return [DecomposedGate("RX", (qubit,), {"angle": PI})]
    
    def _decompose_y(self, qubit: int) -> List[DecomposedGate]:
        """Decompose Y gate. Y = Ry(pi)"""
        return [DecomposedGate("RY", (qubit,), {"angle": PI})]
    
    def _decompose_z(self, qubit: int) -> List[DecomposedGate]:
        """Decompose Z gate. Z = Rz(pi)"""
        return [DecomposedGate("RZ", (qubit,), {"angle": PI})]
    
    def _decompose_cz(self, control: int, target: int) -> List[DecomposedGate]:
        """Decompose CZ to 1 MS gate.
        
        Derived via IH·CNOT·IH cancellation (PRA 99, 022330):
            RY(ctl) RX(ctl) RY(tgt) RX(tgt) MS(ctl,tgt) RY(ctl) RY(tgt) RX(tgt)
        """
        return [
            DecomposedGate("RY", (control,), {"angle": PI_2}),
            DecomposedGate("RX", (control,), {"angle": PI_2}),
            DecomposedGate("RY", (target,), {"angle": PI_2}),
            DecomposedGate("RX", (target,), {"angle": PI_2}),
            DecomposedGate("MS", (control, target), {"angle": PI_4}),
            DecomposedGate("RY", (control,), {"angle": PI_2}),
            DecomposedGate("RY", (target,), {"angle": PI_2}),
            DecomposedGate("RX", (target,), {"angle": PI_2}),
        ]
    
    def _decompose_swap(self, q1: int, q2: int) -> List[DecomposedGate]:
        """Decompose SWAP to 3 CNOTs. SWAP = CNOT_12 . CNOT_21 . CNOT_12"""
        ops = []
        ops.extend(self._decompose_cnot(q1, q2))
        ops.extend(self._decompose_cnot(q2, q1))
        ops.extend(self._decompose_cnot(q1, q2))
        return ops
    
    def _decompose_iswap(self, q1: int, q2: int) -> List[DecomposedGate]:
        """Decompose iSWAP. iSWAP can be done with 2 MS gates."""
        return [
            DecomposedGate("MS", (q1, q2), {"angle": PI_4}),
            DecomposedGate("MS", (q1, q2), {"angle": PI_4}),
        ]
    
    # Gate names that are 2-qubit (applied pair-wise on targets)
    _TWO_Q_GATE_NAMES = frozenset(
        ("CX", "CNOT", "ZCX", "CZ", "ZCZ", "SWAP", "ISWAP")
    )

    def decompose_stim_gate(
        self,
        gate_name: str,
        qubits: Tuple[int, ...],
    ) -> List[DecomposedGate]:
        """Decompose a Stim gate to native trapped ion gates.

        Multi-target 2Q instructions (e.g. ``CX 0 1 2 3 4 5``)
        are split into consecutive qubit pairs and each pair is
        decomposed independently.
        """
        # Check cache
        cache_key = f"{gate_name}_{qubits}"
        if cache_key in self._decomposition_cache:
            return self._decomposition_cache[cache_key]

        # --- Two-qubit gates (handle multi-target pairs) ---------------
        if gate_name in self._TWO_Q_GATE_NAMES and len(qubits) >= 2:
            result: List[DecomposedGate] = []
            for _pi in range(0, len(qubits) - 1, 2):
                q0, q1 = qubits[_pi], qubits[_pi + 1]
                if gate_name in ("CX", "CNOT", "ZCX"):
                    result.extend(self._decompose_cnot(q0, q1))
                elif gate_name in ("CZ", "ZCZ"):
                    result.extend(self._decompose_cz(q0, q1))
                elif gate_name == "SWAP":
                    result.extend(self._decompose_swap(q0, q1))
                elif gate_name == "ISWAP":
                    result.extend(self._decompose_iswap(q0, q1))
        # Single-qubit gates — stim may target multiple qubits in one
        # instruction (e.g. ``H 2 12``), so loop over all targets.
        elif gate_name == "H":
            result = []
            for q in qubits:
                result.extend(self._decompose_h(q))
        elif gate_name == "X":
            result = []
            for q in qubits:
                result.extend(self._decompose_x(q))
        elif gate_name == "Y":
            result = []
            for q in qubits:
                result.extend(self._decompose_y(q))
        elif gate_name == "Z":
            result = []
            for q in qubits:
                result.extend(self._decompose_z(q))
        elif gate_name == "S":
            result = []
            for q in qubits:
                result.extend(self._decompose_s(q))
        elif gate_name == "S_DAG":
            result = []
            for q in qubits:
                result.extend(self._decompose_s_dag(q))
        elif gate_name == "T":
            result = []
            for q in qubits:
                result.extend(self._decompose_t(q))
        elif gate_name == "T_DAG":
            result = []
            for q in qubits:
                result.extend(self._decompose_t_dag(q))
        elif gate_name == "SQRT_X":
            result = [DecomposedGate("RX", (q,), {"angle": PI_2}) for q in qubits]
        elif gate_name == "SQRT_X_DAG":
            result = [DecomposedGate("RX", (q,), {"angle": -PI_2}) for q in qubits]
        elif gate_name == "SQRT_Y":
            result = [DecomposedGate("RY", (q,), {"angle": PI_2}) for q in qubits]
        elif gate_name == "SQRT_Y_DAG":
            result = [DecomposedGate("RY", (q,), {"angle": -PI_2}) for q in qubits]
        # Identity / reset / measure
        elif gate_name == "I":
            result = []
        elif gate_name in ("R", "RZ"):
            # Stim R / RZ = reset in Z basis (prepare |0⟩)
            result = [DecomposedGate("R", (q,), {}) for q in qubits]
        elif gate_name == "RX":
            # Stim RX = reset in X basis (prepare |+⟩)
            # Decompose: R (reset |0⟩) then H (|0⟩→|+⟩)
            result = []
            for q in qubits:
                result.append(DecomposedGate("R", (q,), {}))
                result.extend(self._decompose_h(q))
        elif gate_name == "RY":
            # Stim RY = reset in Y basis (prepare |+i⟩)
            # Decompose: R (reset |0⟩) then S then H
            result = []
            for q in qubits:
                result.append(DecomposedGate("R", (q,), {}))
                result.extend(self._decompose_s(q))
                result.extend(self._decompose_h(q))
        elif gate_name == "MR":
            # MR = measure + reset in Z basis; split into all M's then all R's
            # to match old pipeline (stim .decomposed() splits MR → M + R)
            result = []
            for q in qubits:
                result.append(DecomposedGate("M", (q,), {}))
            for q in qubits:
                result.append(DecomposedGate("R", (q,), {}))
        elif gate_name == "MRX":
            # Stim MRX = measure in X basis then reset in X basis
            # Decompose: MX (native X-measurement) then R then H (reset to |+⟩)
            result = []
            for q in qubits:
                result.append(DecomposedGate("MX", (q,), {}))
                result.append(DecomposedGate("R", (q,), {}))
                result.extend(self._decompose_h(q))
        elif gate_name == "MRY":
            # Stim MRY = measure in Y basis then reset in Y basis
            result = []
            for q in qubits:
                result.append(DecomposedGate("MY", (q,), {}))
                result.append(DecomposedGate("R", (q,), {}))
                result.extend(self._decompose_s(q))
                result.extend(self._decompose_h(q))
        elif gate_name in ("M", "MX", "MY", "MZ"):
            # Split multi-qubit measurement into per-qubit ops
            result = [DecomposedGate(gate_name, (q,), {}) for q in qubits]
        else:
            raise ValueError(f"Unknown gate for decomposition: {gate_name}")
        
        self._decomposition_cache[cache_key] = result
        return result
    
    def get_native_gate_set(self) -> List[str]:
        """Return the native gate set for trapped ions."""
        return self.NATIVE_GATES.copy()
    
    # =========================================================================
    # Helpers: convert native ops to PhysicalOperation objects
    # =========================================================================

    # Default gate timing (μs) and fidelities — derived from CalibrationConstants
    _GATE_TIMING = {
        "MS": _CAL.ms_gate_time * 1e6,
        "RX": _CAL.single_qubit_gate_time * 1e6,
        "RY": _CAL.single_qubit_gate_time * 1e6,
        "RZ": 0.1,  # virtual-Z, ~free
        "M": _CAL.measurement_time * 1e6,
        "MX": _CAL.measurement_time * 1e6,
        "MY": _CAL.measurement_time * 1e6,
        "MZ": _CAL.measurement_time * 1e6,
        "MR": (_CAL.measurement_time + _CAL.reset_time) * 1e6,
        "R": _CAL.reset_time * 1e6,
    }
    _GATE_FIDELITY_1Q = _CAL.gate_fidelities()["RX"]
    _GATE_FIDELITY_2Q = _CAL.gate_fidelities()["MS"]
    _MEAS_FIDELITY = 1.0 - _CAL.measurement_infidelity
    _RESET_FIDELITY = 1.0 - _CAL.reset_infidelity

    def _native_ops_to_physical(
        self,
        native_ops: List[DecomposedGate],
        mapping: QubitMapping,
    ) -> List[PhysicalOperation]:
        """Convert a list of DecomposedGate objects to PhysicalOperation objects.

        Maps logical qubit indices through the qubit mapping to produce
        physical qubit indices.

        Parameters
        ----------
        native_ops : List[DecomposedGate]
            Native operations produced by decompose_to_native().
        mapping : QubitMapping
            Logical-to-physical qubit mapping.

        Returns
        -------
        List[PhysicalOperation]
            Ready-to-schedule physical operations.
        """
        physical_ops: List[PhysicalOperation] = []

        for dg in native_ops:
            # Map logical qubits to physical
            phys_qubits = tuple(
                mapping.get_physical(q) if mapping.get_physical(q) is not None else q
                for q in dg.qubits
            )
            gate_name = dg.name
            duration = self._GATE_TIMING.get(
                gate_name, _CAL.single_qubit_gate_time * 1e6
            )

            if gate_name in ("M", "MX", "MY", "MZ", "MR"):
                basis = {"M": "Z", "MX": "X", "MY": "Y", "MZ": "Z", "MR": "Z"}.get(gate_name, "Z")
                op = MeasurementOperation(
                    qubit=phys_qubits[0],
                    basis=basis,
                    duration=duration,
                    readout_fidelity=self._MEAS_FIDELITY,
                )
            elif gate_name == "R":
                op = ResetOperation(
                    qubit=phys_qubits[0],
                    duration=duration,
                    reset_fidelity=self._RESET_FIDELITY,
                )
            else:
                # Gate operation — use GateSpec wrapper
                num_q = len(phys_qubits)
                gate_type = GateType.TWO_QUBIT if num_q == 2 else GateType.SINGLE_QUBIT
                spec = GateSpec(
                    name=gate_name,
                    gate_type=gate_type,
                    num_qubits=num_q,
                    is_native=True,
                )
                fidelity = self._GATE_FIDELITY_2Q if num_q >= 2 else self._GATE_FIDELITY_1Q
                op = GateOperation(
                    gate=spec,
                    qubits=phys_qubits,
                    duration=duration,
                    base_fidelity=fidelity,
                )
            physical_ops.append(op)

        return physical_ops

    # =========================================================================
    # Helpers: interleave 1Q / measurement / reset ops with routed 2Q ops
    # =========================================================================

    @staticmethod
    def _build_native_segments(
        circuit: MappedCircuit,
    ) -> List[Dict[str, Any]]:
        """Group native ops into segments around 2Q gates.

        Walks the native circuit operations ordered by their index and
        groups them by the originating stim instruction (using
        ``stim_instruction_map``).  Each stim instruction becomes one
        segment:

        * **"2q"** – contains a 2-qubit gate.  ``pre_ops`` are the 1Q
          native ops that precede the 2Q gate within the same
          instruction (pre-rotations), and ``post_ops`` follow it
          (post-rotations).
        * **"standalone"** – contains only 1Q / measurement / reset ops
          (e.g. a standalone ``H`` or ``M`` stim instruction).

        Returns
        -------
        list[dict]
            Ordered segments.  Each dict has keys:
            ``seg_type`` ("2q" | "standalone"),
            ``gate_id`` (int | None – 0-based index among 2Q gates),
            ``pre_ops`` (list[DecomposedGate]),
            ``post_ops`` (list[DecomposedGate]).
        """
        native_ops = circuit.native_circuit.operations
        stim_map = circuit.native_circuit.stim_instruction_map

        # Sort stim instructions by earliest native op index
        ordered_stim = sorted(stim_map.items(), key=lambda kv: min(kv[1]))

        segments: List[Dict[str, Any]] = []
        gate_id = 0  # sequential counter for 2Q gates

        # Track which native ops are covered by stim_instruction_map.
        # Ops not covered (if any) will be treated as standalone.
        covered: set = set()
        for _, indices in stim_map.items():
            covered.update(indices)

        # Handle any native ops NOT in stim_instruction_map (edge case).
        uncovered = [
            i for i in range(len(native_ops)) if i not in covered
        ]
        if uncovered:
            # Group consecutive uncovered ops into a standalone segment
            uc_ops = [native_ops[i] for i in uncovered]
            # We'll insert these at the beginning; adjust if needed.
            segments.append({
                "seg_type": "standalone",
                "gate_id": None,
                "pre_ops": uc_ops,
                "post_ops": [],
            })

        for _stim_idx, native_indices in ordered_stim:
            sorted_indices = sorted(native_indices)
            ops = [native_ops[i] for i in sorted_indices]

            # Find the 2Q op in this group (if any)
            two_q_pos = None
            for k, op in enumerate(ops):
                if len(op.qubits) == 2:
                    two_q_pos = k
                    break

            if two_q_pos is not None:
                segments.append({
                    "seg_type": "2q",
                    "gate_id": gate_id,
                    "pre_ops": ops[:two_q_pos],
                    "post_ops": ops[two_q_pos + 1:],
                })
                gate_id += 1
            else:
                segments.append({
                    "seg_type": "standalone",
                    "gate_id": None,
                    "pre_ops": ops,
                    "post_ops": [],
                })

        return segments

    def _interleave_non_2q_ops(
        self,
        routed_ops: List[PhysicalOperation],
        circuit: MappedCircuit,
    ) -> List[PhysicalOperation]:
        """Insert 1Q / measurement / reset ops at correct positions.

        The routing passes only produce 2Q (MS) gate operations and
        transport operations.  This method inserts the remaining native
        operations at their correct positions relative to the MS gates,
        preserving the decomposition ordering from the original circuit.

        The algorithm mirrors the old code's drain-loop pattern:
        pre-rotations belonging to a given MS gate are emitted
        immediately before it, post-rotations immediately after, and
        standalone ops (H, M, R that are their own stim instruction)
        are placed at their correct native-circuit position.

        Parameters
        ----------
        routed_ops : list[PhysicalOperation]
            Transport + 2Q gate operations produced by the routing pass.
        circuit : MappedCircuit
            The mapped circuit with the native circuit and qubit mapping.

        Returns
        -------
        list[PhysicalOperation]
            Fully interleaved operation list.
        """
        segments = self._build_native_segments(circuit)
        mapping = circuit.mapping

        # --- Assign standalone segments to the 2Q gate they precede ---
        # gate_leading[gid] = list of standalone DecomposedGate ops that
        # appear between the previous 2Q gate and this one in native order.
        gate_leading: Dict[int, List[DecomposedGate]] = {}
        trailing: List[DecomposedGate] = []

        current_standalone: List[DecomposedGate] = []
        for seg in segments:
            if seg["seg_type"] == "standalone":
                current_standalone.extend(seg["pre_ops"])
            else:
                # 2Q segment — attach accumulated standalone ops
                gate_leading[seg["gate_id"]] = current_standalone
                current_standalone = []
        trailing = current_standalone  # standalone ops after last 2Q gate

        # --- Build map: gate_id -> (pre_ops, post_ops) ---
        gate_pre_post: Dict[int, Tuple[List, List]] = {}
        for seg in segments:
            if seg["seg_type"] == "2q":
                gate_pre_post[seg["gate_id"]] = (
                    seg["pre_ops"],
                    seg["post_ops"],
                )

        # --- Walk routed_ops and insert 1Q ops around MS gates ---
        result: List[PhysicalOperation] = []
        ms_counter = 0  # matches kth MS gate to kth native 2Q op

        for op in routed_ops:
            is_2q_gate = (
                isinstance(op, GateOperation)
                and op.operation_type == OperationType.GATE_2Q
            )
            if is_2q_gate:
                gid = ms_counter
                ms_counter += 1

                # Leading standalone ops for this gate
                if gid in gate_leading:
                    result.extend(
                        self._native_ops_to_physical(
                            gate_leading[gid], mapping
                        )
                    )

                # Pre-rotations
                pre, post = gate_pre_post.get(gid, ([], []))
                result.extend(
                    self._native_ops_to_physical(pre, mapping)
                )

                # The MS gate itself
                result.append(op)

                # Post-rotations
                result.extend(
                    self._native_ops_to_physical(post, mapping)
                )
            else:
                # Transport / sentinel ops pass through unchanged
                result.append(op)

        # Trailing standalone ops (measurements, etc. after last gate)
        result.extend(
            self._native_ops_to_physical(trailing, mapping)
        )

        return result

    @staticmethod
    def _batches_to_scheduled(
        batches: List[OperationBatch],
    ) -> Tuple[List[ScheduledOperation], List[CircuitLayer], float]:
        """Convert OperationBatch list to ScheduledOperation + CircuitLayer lists.

        Returns
        -------
        scheduled_ops : List[ScheduledOperation]
        layers : List[CircuitLayer]
        total_duration : float
        """
        scheduled_ops: List[ScheduledOperation] = []
        layers: List[CircuitLayer] = []
        current_time = 0.0

        for batch_idx, batch in enumerate(batches):
            layer = CircuitLayer(
                operations=list(batch.operations),
                start_time=current_time,
            )
            layers.append(layer)

            for op in batch.operations:
                sop = ScheduledOperation(
                    operation=op,
                    start_time=current_time,
                    end_time=current_time + op.duration,
                    parallel_group=batch_idx,
                )
                scheduled_ops.append(sop)

            current_time += layer.duration

        return scheduled_ops, layers, current_time

    # =========================================================================
    # Abstract methods - MUST be implemented by architecture-specific compilers
    # =========================================================================
    
    @abstractmethod
    def _setup_passes(self) -> None:
        """Set up architecture-specific compilation passes."""
        ...
    
    @abstractmethod
    def decompose_to_native(self, circuit: stim.Circuit) -> NativeCircuit:
        """Decompose input circuit to native MS + rotation gates."""
        ...
    
    @abstractmethod
    def map_qubits(self, circuit: NativeCircuit) -> MappedCircuit:
        """Map logical qubits to physical ions."""
        ...
    
    @abstractmethod  
    def route(self, circuit: MappedCircuit) -> RoutedCircuit:
        """Route ions to satisfy gate locality constraints."""
        ...
    
    @abstractmethod
    def schedule(self, circuit: RoutedCircuit) -> ScheduledCircuit:
        """Schedule operations with timing and parallelization."""
        ...
