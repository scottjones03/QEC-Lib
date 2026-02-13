# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/wise_sat.py
"""
WISE SAT-based routing for trapped ion QCCD architectures.

This module re-exports components that have been split into separate modules:
- sat_encoder.py: WISESATContext, WISESATEncoder
- routers.py: WiseSatRouter, WisePatchRouter, GreedyIonRouter
- compilation.py: WISERoutingPass
- cost_model.py: WISECostModel
- orchestrator.py: WISERoutingBlock, WISERoutingOrchestrator

For new code, prefer importing directly from the specific modules.
"""

from __future__ import annotations

# =============================================================================
# Re-exports from sat_encoder.py
# =============================================================================
from qectostim.experiments.hardware_simulation.trapped_ion.routing.sat_encoder import (
    WISESATContext,
    WISESATEncoder,
)

# =============================================================================
# Re-exports from routers.py
# =============================================================================
from qectostim.experiments.hardware_simulation.trapped_ion.routing.routers import (
    WiseSatRouter,
    WisePatchRouter,
    GreedyIonRouter,
)

# =============================================================================
# Re-exports from compilation.py
# =============================================================================
from qectostim.experiments.hardware_simulation.trapped_ion.routing.compilation import (
    WISERoutingPass,
)

# =============================================================================
# Re-exports from cost_model.py
# =============================================================================
from qectostim.experiments.hardware_simulation.trapped_ion.routing.cost_model import (
    WISECostModel,
)

# =============================================================================
# Re-exports from orchestrator.py
# =============================================================================
from qectostim.experiments.hardware_simulation.trapped_ion.routing.orchestrator import (
    WISERoutingBlock,
    WISERoutingOrchestrator,
)

# =============================================================================
# Re-exports from config.py
# =============================================================================
from qectostim.experiments.hardware_simulation.trapped_ion.routing.config import (
    WISE_LOGGER_NAME,
    wise_logger,
    WISERoutingConfig,
    ROW_SWAP_HEATING,
    COL_SWAP_HEATING,
    ROW_SWAP_TIME_US,
    COL_SWAP_TIME_US,
    INITIAL_SPLIT_TIME_US,
    SPLIT_HEATING,
    H_PASS_TIME_US,
    V_PASS_TIME_US,
    _PYSAT_AVAILABLE,
)

# =============================================================================
# Re-exports from data_structures.py
# =============================================================================
from qectostim.experiments.hardware_simulation.trapped_ion.routing.data_structures import (
    GridLayout,
    RoutingPass,
    RoutingSchedule,
    GatePairRequirement,
    compute_target_positions,
)

# =============================================================================
# Re-exports from layout_utils.py
# =============================================================================
from qectostim.experiments.hardware_simulation.trapped_ion.routing.layout_utils import (
    NoFeasibleLayoutError,
    compute_patch_gating_capacity as _compute_patch_gating_capacity,
    compute_cross_boundary_prefs as _compute_cross_boundary_prefs,
    compute_target_layout_from_pairs as _compute_target_layout_from_pairs,
    pre_sat_sanity_checks as _pre_sat_sanity_checks,
    heuristic_phaseB_greedy_layout as _heuristic_phaseB_greedy_layout,
    heuristic_odd_even_reconfig as _heuristic_odd_even_reconfig,
    # Schedule merging helpers
    pass_has_swaps as _pass_has_swaps,
    infer_pass_parity as _infer_pass_parity,
    split_patch_round_into_HV as _split_patch_round_into_HV,
    merge_phase_passes as _merge_phase_passes,
    merge_patch_round_schedules as _merge_patch_round_schedules,
    merge_patch_schedules as _merge_patch_schedules,
)

# =============================================================================
# Re-exports from solvers.py
# =============================================================================
from qectostim.experiments.hardware_simulation.trapped_ion.routing.solvers import (
    run_sat_with_timeout as _run_sat_with_timeout,
    run_rc2_with_timeout as _run_rc2_with_timeout,
    _run_solver_in_subprocess,
    _sat_subprocess_worker,
    _rc2_subprocess_worker,
    # Parallel config search
    wise_safe_sat_pool_workers as _wise_safe_sat_pool_workers,
    enumerate_pmax_configs as _enumerate_pmax_configs,
    sat_config_worker as _sat_config_worker,
    parallel_config_search as _parallel_config_search,
)

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # SAT Encoder
    "WISESATEncoder",
    "WISESATContext",
    # Routers
    "WiseSatRouter",
    "WisePatchRouter",
    "GreedyIonRouter",
    # Compilation Pass
    "WISERoutingPass",
    # Cost Model
    "WISECostModel",
    # Configuration
    "WISERoutingConfig",
    # Data structures
    "GridLayout",
    "RoutingPass",
    "RoutingSchedule",
    "GatePairRequirement",
    # Utilities
    "compute_target_positions",
    # Orchestrator
    "WISERoutingOrchestrator",
    "WISERoutingBlock",
    # Helper functions
    "_compute_patch_gating_capacity",
    "_compute_cross_boundary_prefs",
    "_compute_target_layout_from_pairs",
    "_pre_sat_sanity_checks",
    # Schedule merging
    "_pass_has_swaps",
    "_infer_pass_parity",
    "_merge_patch_schedules",
    # Parallel config search
    "_parallel_config_search",
    "_enumerate_pmax_configs",
    # Exceptions
    "NoFeasibleLayoutError",
    # Logger
    "WISE_LOGGER_NAME",
    "wise_logger",
    # pysat availability
    "_PYSAT_AVAILABLE",
]
