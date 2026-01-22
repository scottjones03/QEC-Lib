# src/qectostim/utils/__init__.py
"""
Utility modules for QECToStim.

Provides shared functionality used across the codebase:
- scheduling_core: Shared scheduling algorithms (graph coloring, geometric scheduling)
- hierarchical_mapper: Qubit and measurement mappers for multi-level concatenation
"""

from qectostim.utils.scheduling_core import (
    CodeMetadataCache,
    graph_coloring_cnots,
    schedule_stabilizer_cnots,
    compute_circuit_depth,
)

from qectostim.utils.hierarchical_mapper import (
    HierarchicalQubitMapper,
    HierarchicalMeasurementMapper,
    MeasurementRecord,
    MeasurementType,
    create_mappers,
)

__all__ = [
    # Scheduling
    "CodeMetadataCache",
    "graph_coloring_cnots",
    "schedule_stabilizer_cnots",
    "compute_circuit_depth",
    # Hierarchical mappers
    "HierarchicalQubitMapper",
    "HierarchicalMeasurementMapper",
    "MeasurementRecord",
    "MeasurementType",
    "create_mappers",
]
