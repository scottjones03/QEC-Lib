# src/qectostim/utils/__init__.py
"""
Utility modules for QECToStim.

Provides shared functionality used across the codebase:
- scheduling_core: Shared scheduling algorithms (graph coloring, geometric scheduling)
"""

from qectostim.utils.scheduling_core import (
    CodeMetadataCache,
    graph_coloring_cnots,
    schedule_stabilizer_cnots,
    compute_circuit_depth,
)

__all__ = [
    "CodeMetadataCache",
    "graph_coloring_cnots",
    "schedule_stabilizer_cnots",
    "compute_circuit_depth",
]
