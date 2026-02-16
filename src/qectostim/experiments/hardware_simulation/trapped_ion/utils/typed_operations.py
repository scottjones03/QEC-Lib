"""
Type-safe dataclass descriptors for old/ operation types.

Provides frozen dataclass `OperationDescriptor` for serialization,
logging and debugging of operations without holding mutable references
to the live Operation objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .qccd_operations import Operation


@dataclass(frozen=True)
class OperationDescriptor:
    """Frozen snapshot of an Operation for serialization/logging.

    Attributes
    ----------
    kind : str
        Operation type name (e.g. "Split", "Merge", "Move", "MS").
    time_us : float
        Operation time in microseconds.
    fidelity : float
        Gate fidelity.
    dephasing_fidelity : float
        Dephasing fidelity.
    involved_ions : Tuple[int, ...]
        Indices of involved ions.
    involved_components : Tuple[int, ...]
        Indices of involved components (traps, junctions).
    label : str
        Human-readable label.
    metadata : Dict[str, Any]
        Additional metadata.
    """

    kind: str
    time_us: float = 0.0
    fidelity: float = 1.0
    dephasing_fidelity: float = 1.0
    involved_ions: Tuple[int, ...] = ()
    involved_components: Tuple[int, ...] = ()
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.kind}({self.label}) "
            f"t={self.time_us:.1f}μs "
            f"F={self.fidelity:.6f}"
        )


def describe(op: "Operation") -> OperationDescriptor:
    """Create an OperationDescriptor from a live Operation.

    Parameters
    ----------
    op : Operation
        A live (mutable) operation from the old/ pipeline.

    Returns
    -------
    OperationDescriptor
        Frozen snapshot.
    """
    kind = type(op).__name__

    # Safely extract timing
    try:
        op.calculateOperationTime()
        time_us = op.operationTime()
    except Exception:
        time_us = 0.0

    # Safely extract fidelity
    try:
        op.calculateFidelity()
        fidelity = op.fidelity()
    except Exception:
        fidelity = 1.0

    # Safely extract dephasing fidelity
    try:
        op.calculateDephasingFidelity()
        dephasing = op.dephasingFidelity()
    except Exception:
        dephasing = 1.0

    # Extract involved ion indices
    try:
        ions = tuple(ion.idx for ion in op.involvedIonsForLabel())
    except Exception:
        ions = ()

    # Extract involved component indices
    try:
        components = tuple(c.idx for c in op.involvedComponents())
    except Exception:
        components = ()

    # Extract label
    try:
        label = op.label()
    except Exception:
        label = kind

    return OperationDescriptor(
        kind=kind,
        time_us=time_us,
        fidelity=fidelity,
        dephasing_fidelity=dephasing,
        involved_ions=ions,
        involved_components=components,
        label=label,
    )


def describe_schedule(
    operations: Sequence["Operation"],
) -> List[OperationDescriptor]:
    """Describe a sequence of operations.

    Parameters
    ----------
    operations : Sequence[Operation]
        Operations from the old/ pipeline.

    Returns
    -------
    List[OperationDescriptor]
        List of frozen descriptors.
    """
    return [describe(op) for op in operations]


def schedule_summary(
    operations: Sequence["Operation"],
) -> Dict[str, Any]:
    """Summary statistics for a schedule.

    Returns dict with total_time, total_fidelity, op_counts, etc.
    """
    descriptors = describe_schedule(operations)

    total_time = sum(d.time_us for d in descriptors)
    total_fidelity = 1.0
    for d in descriptors:
        total_fidelity *= d.fidelity

    op_counts: Dict[str, int] = {}
    for d in descriptors:
        op_counts[d.kind] = op_counts.get(d.kind, 0) + 1

    return {
        "total_time_us": total_time,
        "total_fidelity": total_fidelity,
        "num_operations": len(descriptors),
        "op_counts": op_counts,
    }
