# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/__init__.py
"""
Routing sub-package for trapped-ion QCCD architectures.

Contains the SAT-based WISE routing, greedy junction routing,
patch tiling, qubit-to-ion mapping, and scheduling/parallelisation.

Config search has moved to ``WISECompiler.search_configs()``.

Types (Ion, QCCDNode, Trap, etc.) live in architecture.py.
Routing functions accept QCCDArchitecture / WISEArchitecture directly —
no adapter layer is needed.

Primary entry points
--------------------
ionRoutingWISEArch     WISE SAT-based routing (qccd_WISE_ion_route.py)
ionRouting             Greedy junction routing (greedy_routing.py)
regularPartition       BSP + Hungarian placement (qccd_qubits_to_ions.py)
paralleliseOperations  DAG-based scheduling (qccd_parallelisation.py)
"""

# ── Configuration ────────────────────────────────────────────
from qectostim.experiments.hardware_simulation.trapped_ion.routing.config import (
    WISERoutingConfig,
    RoutingProgress,
    ProgressCallback,
    make_tqdm_progress_callback,
)

# ── WISE SAT routing ────────────────────────────────────────
from qectostim.experiments.hardware_simulation.trapped_ion.routing.qccd_WISE_ion_route import (
    ionRoutingWISEArch,
)

# ── Greedy junction routing (augmented grid) ─────────────────
from qectostim.experiments.hardware_simulation.trapped_ion.routing.greedy_routing import (
    ionRouting,
)

# ── Qubit-to-ion mapping ────────────────────────────────────
from qectostim.experiments.hardware_simulation.trapped_ion.routing.qccd_qubits_to_ions import (
    regularPartition,
    arrangeClusters,
    hillClimbOnArrangeClusters,
)

# ── Scheduling / parallelisation ─────────────────────────────
from qectostim.experiments.hardware_simulation.trapped_ion.routing.qccd_parallelisation import (
    paralleliseOperations,
    paralleliseOperationsSimple,
    paralleliseOperationsWithBarriers,
    calculateDephasingFromIdling,
    calculateDephasingFidelity,
)

# ── Utilities (kept for backward compat of external callers) ──
def old_ops_to_transport_list(old_ops, barriers):
    """Convert old operation list + barriers into a transport list.

    Kept as a thin inline helper for backward compatibility.
    """
    metadata = {
        "num_operations": len(old_ops),
        "num_barriers": len(barriers),
        "routing_source": "old_pipeline",
    }
    return list(old_ops), list(barriers), metadata

# ── Operation types (from focused modules) ───────────────────
from qectostim.experiments.hardware_simulation.trapped_ion.routing.reconfiguration import (
    ReconfigurationPlanner,
    GlobalReconfigurations,  # Alias for backward compat
)
from qectostim.experiments.hardware_simulation.trapped_ion.routing.sat_solver import (
    NoFeasibleLayoutError,
)
from qectostim.experiments.hardware_simulation.trapped_ion.operations import (
    ParallelOperation,
)

__all__ = [
    # Config
    "WISERoutingConfig",
    "RoutingProgress",
    "ProgressCallback",
    "make_tqdm_progress_callback",
    # WISE routing
    "ionRoutingWISEArch",
    # Greedy routing
    "ionRouting",
    # Qubit mapping
    "regularPartition",
    "arrangeClusters",
    "hillClimbOnArrangeClusters",
    # Scheduling
    "paralleliseOperations",
    "paralleliseOperationsSimple",
    "paralleliseOperationsWithBarriers",
    "calculateDephasingFromIdling",
    "calculateDephasingFidelity",
    # Utilities
    "old_ops_to_transport_list",
    # Operation types
    "NoFeasibleLayoutError",
    "ReconfigurationPlanner",
    "GlobalReconfigurations",  # Alias
    "ParallelOperation",
]
