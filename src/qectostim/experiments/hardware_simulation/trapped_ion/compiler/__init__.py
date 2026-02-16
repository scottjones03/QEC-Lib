"""
Compiler subpackage for trapped-ion QCCD simulation.

Provides quantum-to-ion mapping, scheduling, routing, and reconfiguration
algorithms for trapped-ion architectures (Augmented Grid, WISE).
"""

# Routing configuration (no circular deps)
from .routing_config import (
    WISERoutingConfig,
    WISESolverParams,
    RoutingProgress,
    make_tqdm_progress_callback,
    make_simple_progress_callback,
    make_logging_progress_callback,
)

# Process management utilities (no circular deps)
from .process_supervisor import (
    ProcessSupervisor,
    safe_spawn_context,
    get_safe_worker_count,
)

__all__ = [
    # Routing modules (import on demand to avoid circular imports)
    "qccd_ion_routing",
    "qccd_WISE_ion_route",
    "qccd_SAT_WISE_odd_even_sorter",
    "qccd_qubits_to_ions",
    "qccd_parallelisation",
    # Configuration
    "WISERoutingConfig",
    "WISESolverParams",
    "RoutingProgress",
    "make_tqdm_progress_callback",
    "make_simple_progress_callback",
    "make_logging_progress_callback",
    # Process management
    "ProcessSupervisor",
    "safe_spawn_context",
    "get_safe_worker_count",
]

# For convenience, expose the main routing functions via attributes
# but don't import them at module level to avoid circular imports
def __getattr__(name):
    """Lazy import of routing functions to avoid circular imports."""
    if name == "ionRouting":
        from .qccd_ion_routing import ionRouting
        return ionRouting
    elif name == "ionRoutingWISEArch":
        from .qccd_WISE_ion_route import ionRoutingWISEArch
        return ionRoutingWISEArch
    elif name == "optimal_QMR_for_WISE":
        from .qccd_SAT_WISE_odd_even_sorter import optimal_QMR_for_WISE
        return optimal_QMR_for_WISE
    elif name == "WiseSATSolver":
        from .qccd_SAT_WISE_odd_even_sorter import WiseSATSolver
        return WiseSATSolver
    elif name == "WiseSATError":
        from .qccd_SAT_WISE_odd_even_sorter import WiseSATError
        return WiseSATError
    elif name == "SATTimeoutError":
        from .qccd_SAT_WISE_odd_even_sorter import SATTimeoutError
        return SATTimeoutError
    elif name == "BTConflictError":
        from .qccd_SAT_WISE_odd_even_sorter import BTConflictError
        return BTConflictError
    elif name == "CapacityExceededError":
        from .qccd_SAT_WISE_odd_even_sorter import CapacityExceededError
        return CapacityExceededError
    elif name == "NoFeasibleLayoutError":
        from .qccd_SAT_WISE_odd_even_sorter import NoFeasibleLayoutError
        return NoFeasibleLayoutError
    elif name == "regularPartition":
        from .qccd_qubits_to_ions import regularPartition
        return regularPartition
    elif name == "arrangeClusters":
        from .qccd_qubits_to_ions import arrangeClusters
        return arrangeClusters
    elif name == "hillClimbOnArrangeClusters":
        from .qccd_qubits_to_ions import hillClimbOnArrangeClusters
        return hillClimbOnArrangeClusters
    elif name == "paralleliseOperations":
        from .qccd_parallelisation import paralleliseOperations
        return paralleliseOperations
    elif name == "paralleliseOperationsSimple":
        from .qccd_parallelisation import paralleliseOperationsSimple
        return paralleliseOperationsSimple
    elif name == "paralleliseOperationsWithBarriers":
        from .qccd_parallelisation import paralleliseOperationsWithBarriers
        return paralleliseOperationsWithBarriers
    elif name == "calculateDephasingFromIdling":
        from .qccd_parallelisation import calculateDephasingFromIdling
        return calculateDephasingFromIdling
    elif name == "calculateDephasingFidelity":
        from .qccd_parallelisation import calculateDephasingFidelity
        return calculateDephasingFidelity
    elif name == "happensBeforeForOperations":
        from .qccd_parallelisation import happensBeforeForOperations
        return happensBeforeForOperations
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

