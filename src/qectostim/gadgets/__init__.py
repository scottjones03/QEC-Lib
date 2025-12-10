"""
Fault-tolerant logical gate gadgets for QEC codes.

This module provides infrastructure for constructing fault-tolerant
logical operations on quantum error correcting codes, including:

- Transversal gates (single and multi-block)
- Teleportation-based Clifford gates
- CSS surgery CNOT via lattice surgery

Key Components
--------------
Coordinates (coordinates.py)
    N-dimensional coordinate utilities for 2D-4D codes.
    
Layout (layout.py)
    Multi-block layout management with bridge ancillas.
    
Scheduling (scheduling.py)
    Parallel gate scheduling with geometric and graph coloring strategies.
"""

# Coordinate utilities
from .coordinates import (
    CoordND,
    get_code_dimension,
    get_bounding_box,
    compute_bridge_position,
    translate_coords,
    normalize_coord,
    emit_qubit_coords_nd,
    emit_detector_nd,
)

# Layout management
from .layout import (
    BlockInfo,
    BridgeAncilla,
    QubitIndexMap,
    GadgetLayout,
)

# Scheduling
from .scheduling import (
    GateType,
    ScheduledGate,
    CircuitLayer,
    GadgetScheduler,
    merge_schedules,
)

__all__ = [
    # Coordinates
    "CoordND",
    "get_code_dimension",
    "get_bounding_box",
    "compute_bridge_position",
    "translate_coords",
    "normalize_coord",
    "emit_qubit_coords_nd",
    "emit_detector_nd",
    # Layout
    "BlockInfo",
    "BridgeAncilla",
    "QubitIndexMap",
    "GadgetLayout",
    # Scheduling
    "GateType",
    "ScheduledGate",
    "CircuitLayer",
    "GadgetScheduler",
    "merge_schedules",
]
