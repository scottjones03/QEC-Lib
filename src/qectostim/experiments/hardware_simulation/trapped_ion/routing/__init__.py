# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/__init__.py
"""
WISE SAT-based routing for trapped ion QCCD architectures.

This sub-package provides routing algorithms for QCCD architectures:

Modules
-------
config            – Constants, logger, WISERoutingConfig
data_structures   – GridLayout, RoutingPass, GatePairRequirement
solvers           – Process-isolated SAT/MaxSAT solvers
layout_utils      – Layout helpers, sanity checks, heuristic fallbacks
sat_encoder       – WISESATContext, WISESATEncoder (SAT encoding)
routers           – WiseSatRouter, WisePatchRouter, GreedyIonRouter
compilation       – WISERoutingPass (pipeline integration)
cost_model        – WISECostModel (cost estimation)
orchestrator      – WISERoutingOrchestrator, WISERoutingBlock (caching)
greedy            – Junction-based greedy routing (non-SAT)
wise_sat          – Re-export stub for backward compatibility
"""

# -- Configuration & constants --
from qectostim.experiments.hardware_simulation.trapped_ion.routing.config import (
    WISERoutingConfig,
    WISE_LOGGER_NAME,
    wise_logger,
    _PYSAT_AVAILABLE,
    ROW_SWAP_HEATING,
    COL_SWAP_HEATING,
    ROW_SWAP_TIME_US,
    COL_SWAP_TIME_US,
    INITIAL_SPLIT_TIME_US,
    SPLIT_HEATING,
    H_PASS_TIME_US,
    V_PASS_TIME_US,
)

# -- Data structures --
from qectostim.experiments.hardware_simulation.trapped_ion.routing.data_structures import (
    GridLayout,
    RoutingPass,
    RoutingSchedule,
    GatePairRequirement,
    compute_target_positions,
)

# -- Solvers --
from qectostim.experiments.hardware_simulation.trapped_ion.routing.solvers import (
    run_sat_with_timeout,
    run_rc2_with_timeout,
)

# -- Layout utilities --
from qectostim.experiments.hardware_simulation.trapped_ion.routing.layout_utils import (
    NoFeasibleLayoutError,
    compute_patch_gating_capacity,
    compute_cross_boundary_prefs,
    compute_target_layout_from_pairs,
    pre_sat_sanity_checks,
    heuristic_phaseB_greedy_layout,
    heuristic_odd_even_reconfig,
    # Backward-compatible aliases
    _compute_patch_gating_capacity,
    _compute_cross_boundary_prefs,
    _pre_sat_sanity_checks,
)

# -- SAT encoder --
from qectostim.experiments.hardware_simulation.trapped_ion.routing.sat_encoder import (
    WISESATContext,
    WISESATEncoder,
)

# -- Routers --
from qectostim.experiments.hardware_simulation.trapped_ion.routing.routers import (
    WiseSatRouter,
    WisePatchRouter,
    GreedyIonRouter,
)

# -- Compilation pass --
from qectostim.experiments.hardware_simulation.trapped_ion.routing.compilation import (
    WISERoutingPass,
)

# -- Cost model --
from qectostim.experiments.hardware_simulation.trapped_ion.routing.cost_model import (
    WISECostModel,
)

# -- Orchestrator --
from qectostim.experiments.hardware_simulation.trapped_ion.routing.orchestrator import (
    WISERoutingBlock,
    WISERoutingOrchestrator,
)

# -- Greedy junction-based router (non-SAT) --
from qectostim.experiments.hardware_simulation.trapped_ion.routing.greedy import (
    RoutingBarrier,
    GateRequest,
    QCCDRoutingResult,
    route_ions_junction,
)

__all__ = [
    # Configuration
    "WISERoutingConfig",
    "WISESATContext",
    "WISE_LOGGER_NAME",
    "wise_logger",
    "_PYSAT_AVAILABLE",
    # Data structures
    "GridLayout",
    "RoutingPass",
    "RoutingSchedule",
    "GatePairRequirement",
    "compute_target_positions",
    # Solvers
    "run_sat_with_timeout",
    "run_rc2_with_timeout",
    # Layout utilities
    "NoFeasibleLayoutError",
    "compute_patch_gating_capacity",
    "compute_cross_boundary_prefs",
    "compute_target_layout_from_pairs",
    "pre_sat_sanity_checks",
    "heuristic_phaseB_greedy_layout",
    "heuristic_odd_even_reconfig",
    "_compute_patch_gating_capacity",
    "_compute_cross_boundary_prefs",
    "_pre_sat_sanity_checks",
    # SAT encoder
    "WISESATEncoder",
    # Routers
    "WiseSatRouter",
    "WisePatchRouter",
    "GreedyIonRouter",
    # Compilation pass
    "WISERoutingPass",
    # Cost model
    "WISECostModel",
    # Orchestrator
    "WISERoutingOrchestrator",
    # Constants
    "ROW_SWAP_HEATING",
    "COL_SWAP_HEATING",
    "ROW_SWAP_TIME_US",
    "COL_SWAP_TIME_US",
    "INITIAL_SPLIT_TIME_US",
    "SPLIT_HEATING",
    "H_PASS_TIME_US",
    "V_PASS_TIME_US",
    # Greedy routing
    "RoutingBarrier",
    "GateRequest",
    "QCCDRoutingResult",
    "route_ions_junction",
]
