"""Structured compilation metrics for the old/ module.

Captures gate counts, routing overhead, reconfig timing, etc.
from the ``ionRoutingWISEArch`` / ``ionRouting`` output in a
single dataclass instead of ad-hoc dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class CompilationMetrics:
    """Structured metrics from a trapped-ion compilation pass.

    Attributes
    ----------
    native_gate_count : int
        Total number of native gates after decomposition.
    ms_gate_count : int
        Number of Molmer-Sorensen (2-qubit) gates.
    single_qubit_gate_count : int
        Number of single-qubit rotation gates.
    measurement_count : int
        Number of measurement operations.
    reset_count : int
        Number of reset operations.
    transport_operations : int
        Total transport operations (split + merge + move + junction).
    split_count : int
        Number of split operations.
    merge_count : int
        Number of merge operations.
    move_count : int
        Number of shuttle/move operations.
    junction_crossing_count : int
        Number of junction crossing operations.
    reconfigurations : int
        Number of global reconfiguration steps (WISE).
    gate_swap_count : int
        Number of gate-swap operations.
    total_time_us : float
        Estimated total execution time in microseconds.
    circuit_depth : int
        Number of parallel batches (time steps).
    num_qubits : int
        Number of logical qubits.
    num_ions : int
        Total number of ions (qubits + cooling + placeholders).
    routing_overhead : float
        Ratio of transport ops to gate ops (lower = better).
    """

    # Gate counts
    native_gate_count: int = 0
    ms_gate_count: int = 0
    single_qubit_gate_count: int = 0
    measurement_count: int = 0
    reset_count: int = 0

    # Transport counts
    transport_operations: int = 0
    split_count: int = 0
    merge_count: int = 0
    move_count: int = 0
    junction_crossing_count: int = 0
    reconfigurations: int = 0
    gate_swap_count: int = 0

    # Timing & structure
    total_time_us: float = 0.0
    circuit_depth: int = 0
    num_qubits: int = 0
    num_ions: int = 0

    @property
    def routing_overhead(self) -> float:
        """Ratio of transport operations to gate operations."""
        gate_ops = self.ms_gate_count + self.single_qubit_gate_count
        if gate_ops == 0:
            return float("inf") if self.transport_operations > 0 else 0.0
        return self.transport_operations / gate_ops

    @property
    def total_gate_count(self) -> int:
        """Total gate operations (MS + 1Q + measurements + resets)."""
        return (
            self.ms_gate_count
            + self.single_qubit_gate_count
            + self.measurement_count
            + self.reset_count
        )

    def summary(self) -> str:
        """Human-readable summary of compilation metrics."""
        lines = [
            "=" * 50,
            "Compilation Metrics",
            "=" * 50,
            f"  Qubits: {self.num_qubits}  |  Ions: {self.num_ions}",
            f"  Circuit depth: {self.circuit_depth}",
            f"  Total time: {self.total_time_us:.1f} us",
            "",
            "  Gate counts:",
            f"    MS (2Q):      {self.ms_gate_count}",
            f"    1Q rotations: {self.single_qubit_gate_count}",
            f"    Measurements: {self.measurement_count}",
            f"    Resets:       {self.reset_count}",
            f"    Gate swaps:   {self.gate_swap_count}",
            f"    Total native: {self.native_gate_count}",
            "",
            "  Transport:",
            f"    Splits:       {self.split_count}",
            f"    Merges:       {self.merge_count}",
            f"    Moves:        {self.move_count}",
            f"    Junctions:    {self.junction_crossing_count}",
            f"    Reconfigs:    {self.reconfigurations}",
            f"    Total:        {self.transport_operations}",
            "",
            f"  Routing overhead: {self.routing_overhead:.2f}x",
            "=" * 50,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "native_gate_count": self.native_gate_count,
            "ms_gate_count": self.ms_gate_count,
            "single_qubit_gate_count": self.single_qubit_gate_count,
            "measurement_count": self.measurement_count,
            "reset_count": self.reset_count,
            "transport_operations": self.transport_operations,
            "split_count": self.split_count,
            "merge_count": self.merge_count,
            "move_count": self.move_count,
            "junction_crossing_count": self.junction_crossing_count,
            "reconfigurations": self.reconfigurations,
            "gate_swap_count": self.gate_swap_count,
            "total_time_us": self.total_time_us,
            "circuit_depth": self.circuit_depth,
            "num_qubits": self.num_qubits,
            "num_ions": self.num_ions,
            "routing_overhead": self.routing_overhead,
        }


def extract_metrics_from_operations(
    operations: Sequence[Any],
    *,
    parallel_batches: Optional[Sequence[Any]] = None,
    num_qubits: int = 0,
    num_ions: int = 0,
) -> CompilationMetrics:
    """Extract CompilationMetrics from a sequence of old/ operations.

    Parameters
    ----------
    operations : sequence
        Flat list of operations from ``ionRouting`` / ``ionRoutingWISEArch``.
    parallel_batches : sequence, optional
        If provided, the parallelised batches from
        ``paralleliseOperationsWithBarriers``. Used for circuit depth.
    num_qubits : int
        Number of logical qubits.
    num_ions : int
        Total number of ions.

    Returns
    -------
    CompilationMetrics
    """
    from .physics import DEFAULT_CALIBRATION

    metrics = CompilationMetrics(num_qubits=num_qubits, num_ions=num_ions)
    cal = DEFAULT_CALIBRATION

    for op in operations:
        type_name = type(op).__name__

        if type_name in ("TwoQubitMSGate", "MSGate"):
            metrics.ms_gate_count += 1
        elif type_name == "GateSwap":
            metrics.gate_swap_count += 1
        elif type_name in ("Measurement", "MeasurementGate"):
            metrics.measurement_count += 1
        elif type_name in ("QubitReset",):
            metrics.reset_count += 1
        elif type_name in (
            "SingleQubitGate", "XRotation", "YRotation", "ZRotation",
        ):
            metrics.single_qubit_gate_count += 1
        elif type_name == "Split":
            metrics.split_count += 1
        elif type_name == "Merge":
            metrics.merge_count += 1
        elif type_name in ("Move", "Move_"):
            metrics.move_count += 1
        elif type_name == "JunctionCrossing":
            metrics.junction_crossing_count += 1
        elif type_name in ("GlobalReconfigurations",):
            metrics.reconfigurations += 1
        elif type_name == "ParallelOperation":
            # Recurse into sub-operations
            sub_ops = getattr(op, "_operations", getattr(op, "operations", []))
            sub_metrics = extract_metrics_from_operations(sub_ops)
            metrics.ms_gate_count += sub_metrics.ms_gate_count
            metrics.single_qubit_gate_count += sub_metrics.single_qubit_gate_count
            metrics.measurement_count += sub_metrics.measurement_count
            metrics.reset_count += sub_metrics.reset_count
            metrics.gate_swap_count += sub_metrics.gate_swap_count
            metrics.split_count += sub_metrics.split_count
            metrics.merge_count += sub_metrics.merge_count
            metrics.move_count += sub_metrics.move_count
            metrics.junction_crossing_count += sub_metrics.junction_crossing_count
            metrics.reconfigurations += sub_metrics.reconfigurations

    # Derived totals
    metrics.transport_operations = (
        metrics.split_count
        + metrics.merge_count
        + metrics.move_count
        + metrics.junction_crossing_count
    )
    metrics.native_gate_count = (
        metrics.ms_gate_count
        + metrics.single_qubit_gate_count
        + metrics.measurement_count
        + metrics.reset_count
        + metrics.gate_swap_count
    )

    # Estimate total time from calibration constants
    metrics.total_time_us = (
        metrics.ms_gate_count * cal.ms_gate_time * 1e6
        + metrics.single_qubit_gate_count * cal.single_qubit_gate_time * 1e6
        + metrics.measurement_count * cal.measurement_time * 1e6
        + metrics.reset_count * cal.reset_time * 1e6
        + metrics.gate_swap_count * 3 * cal.ms_gate_time * 1e6
        + metrics.split_count * cal.split_time * 1e6
        + metrics.merge_count * cal.merge_time * 1e6
        + metrics.move_count * cal.shuttle_time * 1e6
        + metrics.junction_crossing_count * cal.junction_time * 1e6
    )

    # Circuit depth from parallel batches
    if parallel_batches is not None:
        metrics.circuit_depth = len(parallel_batches)

    return metrics
