# src/qectostim/experiments/hardware_simulation/tests/test_route_schedule.py
"""
Tests for the route() and schedule() pipeline stages.

Validates that:
1. LinearChainCompiler / QCCDCompiler route() produces PhysicalOperation objects
2. WISERoutingPass.__init__ accepts architecture kwarg
3. _convert_to_physical_ops returns TransportOperation (not dicts)
4. schedule() populates scheduled_ops, layers, AND batches
5. The full compile() pipeline produces a valid CompiledCircuit
6. Results match old-code noise patterns (gate fidelities, ordering)
"""
import math
import pytest
import stim

from qectostim.experiments.hardware_simulation.core.operations import (
    GateOperation,
    MeasurementOperation,
    ResetOperation,
    TransportOperation,
    PhysicalOperation,
    OperationBatch,
    OperationType,
    GreedyBatchScheduler,
)
from qectostim.experiments.hardware_simulation.core.pipeline import (
    NativeCircuit,
    MappedCircuit,
    RoutedCircuit,
    ScheduledCircuit,
    CompiledCircuit,
    CircuitLayer,
    ScheduledOperation,
    QubitMapping,
)
from qectostim.experiments.hardware_simulation.core.gates import (
    GateSpec,
    GateType,
)
from qectostim.experiments.hardware_simulation.trapped_ion.compiler import (
    TrappedIonCompiler,
    LinearChainCompiler,
    QCCDCompiler,
    DecomposedGate,
)


# =============================================================================
# Fixtures
# =============================================================================

def _make_linear_chain_arch():
    """Create a minimal LinearChainArchitecture for testing."""
    from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
        LinearChainArchitecture,
    )
    return LinearChainArchitecture(num_ions=9)


def _make_qccd_arch():
    """Create a minimal QCCDArchitecture for testing."""
    from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
        QCCDArchitecture,
    )
    return QCCDArchitecture(rows=2, cols=2, ions_per_trap=4)


def _simple_stim_circuit() -> stim.Circuit:
    """Build a tiny stim circuit with H, CNOT, M.

    Produces 2 qubits with:
      H 0
      CNOT 0 1
      M 0 1
    """
    c = stim.Circuit()
    c.append("H", [0])
    c.append("TICK")
    c.append("CX", [0, 1])
    c.append("TICK")
    c.append("M", [0, 1])
    return c


def _surface_code_d3_circuit() -> stim.Circuit:
    """Build a minimal distance-3 surface code memory circuit.

    Uses stim's built-in generator for a realistic circuit.
    """
    return stim.Circuit.generated(
        "repetition_code:memory",
        rounds=2,
        distance=3,
        after_clifford_depolarization=0,
        after_reset_flip_probability=0,
        before_measure_flip_probability=0,
        before_round_data_depolarization=0,
    )


# =============================================================================
# Test: LinearChainCompiler route()
# =============================================================================

class TestLinearChainRoute:
    """Test that LinearChainCompiler.route() produces PhysicalOperation objects."""

    def test_route_returns_physical_ops(self):
        """route() must produce a list of PhysicalOperation, not dicts or empty."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)

        assert isinstance(routed, RoutedCircuit)
        assert len(routed.operations) > 0, "route() should produce operations"
        for op in routed.operations:
            assert isinstance(op, PhysicalOperation), (
                f"Expected PhysicalOperation, got {type(op).__name__}: {op}"
            )

    def test_route_has_correct_op_types(self):
        """route() produces GateOperation, MeasurementOperation, etc."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)

        op_types = {type(op).__name__ for op in routed.operations}
        assert "GateOperation" in op_types, "Missing GateOperation"
        assert "MeasurementOperation" in op_types, "Missing MeasurementOperation"

    def test_route_has_2q_gates(self):
        """CNOT decomposes to 1 MS (2Q) gate → route should have GATE_2Q ops."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)

        gate_2q = [
            op for op in routed.operations
            if isinstance(op, GateOperation) and op.operation_type == OperationType.GATE_2Q
        ]
        assert len(gate_2q) >= 1, f"Expected ≥1 MS gates from CNOT, got {len(gate_2q)}"

    def test_route_zero_routing_overhead(self):
        """Linear chain has all-to-all: routing_overhead should be 0."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)

        assert routed.routing_overhead == 0

    def test_route_preserves_mapping(self):
        """final_mapping should be the same as input mapping (no routing)."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)

        assert routed.final_mapping is not None
        assert routed.final_mapping.logical_to_physical == mapped.mapping.logical_to_physical


# =============================================================================
# Test: QCCDCompiler route()
# =============================================================================

class TestQCCDRoute:
    """Test that QCCDCompiler.route() produces PhysicalOperation objects."""

    def test_route_returns_physical_ops(self):
        """route() must produce a list of PhysicalOperation, not dicts or empty."""
        arch = _make_qccd_arch()
        compiler = QCCDCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)

        assert isinstance(routed, RoutedCircuit)
        assert len(routed.operations) > 0
        for op in routed.operations:
            assert isinstance(op, PhysicalOperation), (
                f"Expected PhysicalOperation, got {type(op).__name__}: {op}"
            )

    def test_route_has_gate_and_measurement_ops(self):
        """route() should contain both gate and measurement operations."""
        arch = _make_qccd_arch()
        compiler = QCCDCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)

        has_gate = any(isinstance(op, GateOperation) for op in routed.operations)
        has_meas = any(isinstance(op, MeasurementOperation) for op in routed.operations)
        assert has_gate, "Missing GateOperation in routed circuit"
        assert has_meas, "Missing MeasurementOperation in routed circuit"


# =============================================================================
# Test: WISERoutingPass.__init__ architecture kwarg
# =============================================================================

class TestWISERoutingPass:
    """Test WISERoutingPass accepts architecture at construction time."""

    def test_init_accepts_architecture(self):
        """WISERoutingPass(architecture=...) should not raise."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WISERoutingPass,
            WISERoutingConfig,
        )
        arch = _make_linear_chain_arch()
        rp = WISERoutingPass(config=WISERoutingConfig(), architecture=arch)
        assert rp.architecture is arch

    def test_init_without_architecture(self):
        """WISERoutingPass() should default architecture to None."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WISERoutingPass,
        )
        rp = WISERoutingPass()
        assert rp.architecture is None

    def test_route_raises_without_architecture(self):
        """route() with no architecture anywhere should raise ValueError."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WISERoutingPass,
        )
        rp = WISERoutingPass()
        mapping = QubitMapping()
        mapping.assign(0, 0)
        native = NativeCircuit(operations=[], num_qubits=1)
        mapped = MappedCircuit(native_circuit=native, mapping=mapping)

        with pytest.raises(ValueError, match="requires an architecture"):
            rp.route(mapped)


# =============================================================================
# Test: _convert_to_physical_ops produces TransportOperation
# =============================================================================

class TestConvertToPhysicalOps:
    """Test that _convert_to_physical_ops returns TransportOperation, not dicts."""

    def test_dict_becomes_transport(self):
        """A routing dict with type='H_SWAP' should become TransportOperation."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WISERoutingPass,
        )
        rp = WISERoutingPass()
        routing_dicts = [
            {"type": "H_SWAP", "qubits": (0,), "source": "z0", "target": "z1", "distance": 2.0},
            {"type": "V_SWAP", "qubits": (1,), "source": "z1", "target": "z2", "distance": 1.0},
        ]
        result = rp._convert_to_physical_ops(routing_dicts, None)

        assert len(result) == 2
        for op in result:
            assert isinstance(op, TransportOperation), (
                f"Expected TransportOperation, got {type(op).__name__}"
            )

    def test_transport_op_has_correct_duration(self):
        """Transport duration should be distance * 10 μs."""
        from qectostim.experiments.hardware_simulation.trapped_ion.routing import (
            WISERoutingPass,
        )
        rp = WISERoutingPass()
        ops = rp._convert_to_physical_ops(
            [{"type": "transport", "qubits": (3,), "distance": 5.0}],
            None,
        )
        assert abs(ops[0].duration - 50.0) < 1e-9


# =============================================================================
# Test: schedule() populates scheduled_ops, layers, AND batches
# =============================================================================

class TestSchedulePopulation:
    """Test that schedule() fills all three fields of ScheduledCircuit."""

    def test_linear_chain_schedule_all_fields(self):
        """LinearChain schedule() should populate layers, scheduled_ops, batches."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)
        scheduled = compiler.schedule(routed)

        assert isinstance(scheduled, ScheduledCircuit)
        assert len(scheduled.batches) > 0, "batches is empty"
        assert len(scheduled.scheduled_ops) > 0, "scheduled_ops is empty"
        assert len(scheduled.layers) > 0, "layers is empty"
        assert scheduled.total_duration > 0, "total_duration should be positive"

    def test_qccd_schedule_all_fields(self):
        """QCCD schedule() should populate layers, scheduled_ops, batches."""
        arch = _make_qccd_arch()
        compiler = QCCDCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)
        scheduled = compiler.schedule(routed)

        assert len(scheduled.batches) > 0
        assert len(scheduled.scheduled_ops) > 0
        assert len(scheduled.layers) > 0

    def test_scheduled_ops_are_scheduled_operation(self):
        """Each entry in scheduled_ops should be a ScheduledOperation."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)
        scheduled = compiler.schedule(routed)

        for sop in scheduled.scheduled_ops:
            assert isinstance(sop, ScheduledOperation), (
                f"Expected ScheduledOperation, got {type(sop).__name__}"
            )
            assert isinstance(sop.operation, PhysicalOperation)
            assert sop.end_time >= sop.start_time

    def test_layers_are_circuit_layer(self):
        """Each entry in layers should be a CircuitLayer with physical ops."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)
        scheduled = compiler.schedule(routed)

        for layer in scheduled.layers:
            assert isinstance(layer, CircuitLayer)
            assert len(layer.operations) > 0
            for op in layer.operations:
                assert isinstance(op, PhysicalOperation)

    def test_timing_monotonically_increases(self):
        """Layer start times should increase monotonically."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)
        scheduled = compiler.schedule(routed)

        for i in range(1, len(scheduled.layers)):
            assert scheduled.layers[i].start_time >= scheduled.layers[i - 1].start_time


# =============================================================================
# Test: Full compile() pipeline
# =============================================================================

class TestFullCompilePipeline:
    """Test the full compile() pipeline end-to-end."""

    def test_linear_chain_compile_produces_compiled_circuit(self):
        """compile() should produce a CompiledCircuit with original_circuit set."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        circuit = _simple_stim_circuit()

        compiled = compiler.compile(circuit)

        assert isinstance(compiled, CompiledCircuit)
        assert compiled.original_circuit is not None
        assert compiled.original_circuit is circuit
        assert compiled.scheduled is not None
        assert len(compiled.scheduled.layers) > 0
        assert len(compiled.scheduled.scheduled_ops) > 0

    def test_qccd_compile_produces_compiled_circuit(self):
        """compile() should produce a CompiledCircuit for QCCD."""
        arch = _make_qccd_arch()
        compiler = QCCDCompiler(arch)
        circuit = _simple_stim_circuit()

        compiled = compiler.compile(circuit)

        assert isinstance(compiled, CompiledCircuit)
        assert compiled.original_circuit is circuit
        assert len(compiled.scheduled.layers) > 0

    def test_compile_with_surface_code(self):
        """compile() should handle a realistic surface-code circuit."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        circuit = _surface_code_d3_circuit()

        compiled = compiler.compile(circuit)

        assert isinstance(compiled, CompiledCircuit)
        assert compiled.original_circuit is circuit
        assert len(compiled.scheduled.scheduled_ops) > 0
        # Check that duration is reasonable
        assert compiled.total_duration > 0


# =============================================================================
# Test: _native_ops_to_physical helper
# =============================================================================

class TestNativeOpsToPhysical:
    """Test the _native_ops_to_physical conversion helper."""

    def test_gate_decomposed_to_gate_op(self):
        """DecomposedGate('RX', (0,)) → GateOperation with GATE_1Q."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        mapping = QubitMapping()
        mapping.assign(0, 0)

        ops = compiler._native_ops_to_physical(
            [DecomposedGate("RX", (0,), {"angle": math.pi / 2})],
            mapping,
        )
        assert len(ops) == 1
        assert isinstance(ops[0], GateOperation)
        assert ops[0].operation_type == OperationType.GATE_1Q
        assert ops[0].qubits == (0,)

    def test_ms_becomes_gate_2q(self):
        """DecomposedGate('MS', (0, 1)) → GateOperation with GATE_2Q."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        mapping = QubitMapping()
        mapping.assign(0, 0)
        mapping.assign(1, 1)

        ops = compiler._native_ops_to_physical(
            [DecomposedGate("MS", (0, 1), {"angle": math.pi / 4})],
            mapping,
        )
        assert len(ops) == 1
        assert isinstance(ops[0], GateOperation)
        assert ops[0].operation_type == OperationType.GATE_2Q
        assert ops[0].qubits == (0, 1)

    def test_measurement_becomes_measurement_op(self):
        """DecomposedGate('M', (2,)) → MeasurementOperation."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        mapping = QubitMapping()
        mapping.assign(2, 5)

        ops = compiler._native_ops_to_physical(
            [DecomposedGate("M", (2,), {})],
            mapping,
        )
        assert len(ops) == 1
        assert isinstance(ops[0], MeasurementOperation)
        assert ops[0].qubits == (5,)  # mapped through qubit mapping

    def test_reset_becomes_reset_op(self):
        """DecomposedGate('R', (0,)) → ResetOperation."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        mapping = QubitMapping()
        mapping.assign(0, 0)

        ops = compiler._native_ops_to_physical(
            [DecomposedGate("R", (0,), {})],
            mapping,
        )
        assert len(ops) == 1
        assert isinstance(ops[0], ResetOperation)

    def test_fidelities_match_defaults(self):
        """Check fidelity defaults match old-code expectations."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        mapping = QubitMapping()
        mapping.assign(0, 0)
        mapping.assign(1, 1)

        ops = compiler._native_ops_to_physical(
            [
                DecomposedGate("RX", (0,), {}),
                DecomposedGate("MS", (0, 1), {}),
                DecomposedGate("M", (0,), {}),
            ],
            mapping,
        )
        # 1Q gate fidelity
        assert ops[0].fidelity() == 0.9999
        # 2Q gate fidelity
        assert ops[1].fidelity() == 0.99
        # Measurement fidelity
        assert ops[2].fidelity() == 0.99


# =============================================================================
# Test: _batches_to_scheduled helper
# =============================================================================

class TestBatchesToScheduled:
    """Test the _batches_to_scheduled conversion helper."""

    def test_empty_batches(self):
        """Empty batch list → empty results."""
        sops, layers, dur = TrappedIonCompiler._batches_to_scheduled([])
        assert sops == []
        assert layers == []
        assert dur == 0.0

    def test_single_batch(self):
        """Single batch with two ops → one layer, two scheduled_ops."""
        spec = GateSpec("RX", GateType.SINGLE_QUBIT, 1, is_native=True)
        op0 = GateOperation(spec, (0,), duration=10.0)
        op1 = GateOperation(spec, (1,), duration=10.0)
        batch = OperationBatch(operations=[op0, op1])

        sops, layers, dur = TrappedIonCompiler._batches_to_scheduled([batch])
        assert len(layers) == 1
        assert len(sops) == 2
        assert dur == 10.0  # parallel: max duration
        assert sops[0].start_time == 0.0
        assert sops[0].end_time == 10.0

    def test_two_batches_cumulative_timing(self):
        """Two batches → timing accumulates."""
        spec1q = GateSpec("RX", GateType.SINGLE_QUBIT, 1, is_native=True)
        spec2q = GateSpec("MS", GateType.TWO_QUBIT, 2, is_native=True)
        op0 = GateOperation(spec1q, (0,), duration=10.0)
        op1 = GateOperation(spec2q, (0, 1), duration=100.0)
        b0 = OperationBatch(operations=[op0])
        b1 = OperationBatch(operations=[op1])

        sops, layers, dur = TrappedIonCompiler._batches_to_scheduled([b0, b1])
        assert len(layers) == 2
        assert layers[0].start_time == 0.0
        assert layers[1].start_time == 10.0
        assert dur == 110.0
        # Second op should start at 10.0
        assert sops[1].start_time == 10.0
        assert sops[1].end_time == 110.0


# =============================================================================
# Test: GreedyBatchScheduler with PhysicalOperation
# =============================================================================

class TestGreedyBatchScheduler:
    """Verify GreedyBatchScheduler works with real PhysicalOperation objects."""

    def test_schedule_non_conflicting_ops(self):
        """Ops on different qubits should be batched together."""
        spec = GateSpec("RX", GateType.SINGLE_QUBIT, 1, is_native=True)
        ops = [
            GateOperation(spec, (0,), duration=10.0),
            GateOperation(spec, (1,), duration=10.0),
            GateOperation(spec, (2,), duration=10.0),
        ]
        scheduler = GreedyBatchScheduler()
        batches = scheduler.schedule(ops)
        # All 3 can run in parallel → 1 batch
        assert len(batches) == 1
        assert len(batches[0].operations) == 3

    def test_schedule_conflicting_ops(self):
        """Ops on same qubit must be in separate batches."""
        spec = GateSpec("RX", GateType.SINGLE_QUBIT, 1, is_native=True)
        ops = [
            GateOperation(spec, (0,), duration=10.0),
            GateOperation(spec, (0,), duration=10.0),
        ]
        scheduler = GreedyBatchScheduler()
        batches = scheduler.schedule(ops)
        assert len(batches) == 2

    def test_schedule_empty(self):
        """Empty ops list → empty batches."""
        scheduler = GreedyBatchScheduler()
        batches = scheduler.schedule([])
        assert batches == []

    def test_schedule_mixed_ops(self):
        """Gate + measurement on different qubits in one batch."""
        spec = GateSpec("RX", GateType.SINGLE_QUBIT, 1, is_native=True)
        gate = GateOperation(spec, (0,), duration=10.0)
        meas = MeasurementOperation(qubit=1, duration=1.0)

        scheduler = GreedyBatchScheduler()
        batches = scheduler.schedule([gate, meas])
        # Different qubits → one batch
        assert len(batches) == 1


# =============================================================================
# Test: Gate count and operation count consistency
# =============================================================================

class TestOperationCounts:
    """Verify that operation counts are consistent through the pipeline."""

    def test_native_op_count_matches_routed(self):
        """Number of native ops should equal routed physical ops (no routing for linear)."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)

        assert len(routed.operations) == len(native.operations), (
            f"Native ops: {len(native.operations)}, Routed ops: {len(routed.operations)}"
        )

    def test_scheduled_ops_match_routed(self):
        """Total scheduled ops should match routed ops."""
        arch = _make_linear_chain_arch()
        compiler = LinearChainCompiler(arch)
        circuit = _simple_stim_circuit()

        native = compiler.decompose_to_native(circuit)
        mapped = compiler.map_qubits(native)
        routed = compiler.route(mapped)
        scheduled = compiler.schedule(routed)

        total_in_batches = sum(len(b.operations) for b in scheduled.batches)
        assert total_in_batches == len(routed.operations)
        assert len(scheduled.scheduled_ops) == len(routed.operations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
