# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/config.py
"""
WISE routing configuration, constants, and pysat availability.

This module provides:
- Logger configuration for WISE routing
- Transport timing/heating constants
- pysat availability check
- WISERoutingConfig dataclass
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from qectostim.experiments.hardware_simulation.core.sat_interface import (
    SATRoutingConfig,
)

# =============================================================================
# Logger Configuration
# =============================================================================

WISE_LOGGER_NAME = "wise.qccd.routing"
wise_logger = logging.getLogger(WISE_LOGGER_NAME)
if not wise_logger.handlers:
    wise_logger.addHandler(logging.NullHandler())
wise_logger.propagate = False


# =============================================================================
# Reconfiguration heating constants
# =============================================================================

# Import canonical transport constants from the transport module.
from qectostim.experiments.hardware_simulation.trapped_ion.transport import (
    ROW_SWAP_HEATING as _RSH,
    COL_SWAP_HEATING as _CSH,
    ROW_SWAP_TIME_S as _RST,
    COL_SWAP_TIME_S as _CST,
    Split as _Split,
    JunctionCrossing as _JC,
    Move as _Move,
)

# Motional quanta deposited per ion per swap.
ROW_SWAP_HEATING: float = _RSH
COL_SWAP_HEATING: float = _CSH
ROW_SWAP_TIME_US: float = _RST * 1e6
COL_SWAP_TIME_US: float = _CST * 1e6

# Per-pass reconfiguration timing
INITIAL_SPLIT_TIME_US: float = _Split.SPLITTING_TIME * 1e6
SPLIT_HEATING: float = _Split.HEATING_RATE * _Split.SPLITTING_TIME
H_PASS_TIME_US: float = ROW_SWAP_TIME_US
V_PASS_TIME_US: float = (
    2.0 * _JC.CROSSING_TIME + (4.0 * _JC.CROSSING_TIME + _Move.MOVING_TIME) * 2
) * 1e6


# =============================================================================
# pysat Availability Check
# =============================================================================

_PYSAT_AVAILABLE = False
try:
    from pysat.formula import IDPool, CNF, WCNF
    from pysat.card import CardEnc, EncType
    from pysat.solvers import Minisat22, Solver
    _PYSAT_AVAILABLE = True
except ImportError:
    IDPool = None  # type: ignore[misc,assignment]
    CNF = None  # type: ignore[misc,assignment]
    WCNF = None  # type: ignore[misc,assignment]
    CardEnc = None  # type: ignore[misc,assignment]
    EncType = None  # type: ignore[misc,assignment]
    Minisat22 = None  # type: ignore[misc,assignment]
    Solver = None  # type: ignore[misc,assignment]


# =============================================================================
# WISERoutingConfig
# =============================================================================

@dataclass
class WISERoutingConfig(SATRoutingConfig):
    """Configuration for WISE SAT-based routing.
    
    Extends the base SATRoutingConfig with WISE-specific parameters
    for trapped-ion QCCD architectures.
    """
    # WISE-specific parameters
    patch_enabled: bool = False
    patch_height: int = 4
    patch_width: int = 4
    # --- Boundary / BT soft weights ---
    # Old ground truth defaults: wB_col=1, wB_row=1 in _optimal_QMR_for_WISE.
    # bt_soft_weight is computed dynamically when bt_soft=True:
    #   bt_soft_weight = max(100, max(wB_col, wB_row, 1) * 10)
    # Setting these to 1 matches the old default.
    bt_soft_weight: int = 0  # 0 = auto-compute from boundary weights when needed
    boundary_soft_weight_row: int = 1
    boundary_soft_weight_col: int = 1
    subgridsize: Tuple[int, int, int] = (6, 4, 1)
    base_pmax_in: Optional[int] = None
    lookahead_rounds: int = 2
    max_cycles: int = 10
    boundary_capacity_factor: float = 1.0
    bcf_steps: int = 6
    bcf_min: float = 0.0
    barrier_threshold: float = float('inf')
    go_back_threshold: float = 0.0

    # Grid defaults (used when architecture is not available)
    default_rows: int = 3
    default_cols: int = 6
    min_passes: int = 1
    block_capacity: int = 2

    # --- Timeout defaults (match old ground truth: 4800s each) ---
    # The base SATRoutingConfig.timeout_seconds controls SAT timeout.
    # We add a separate rc2_timeout for MaxSAT (old code had separate
    # max_sat_time / max_rc2_time parameters).
    rc2_timeout_seconds: float = 4800.0

    # Parallel SAT config search (faithful port of old ProcessPoolExecutor)
    parallel_sat_search: bool = False
    sat_workers: int = 0  # 0 = auto-detect from CPU count

    # Patch routing thresholds
    patch_size: int = 4
    patch_threshold: int = 64  # n_rows * n_cols below which we skip patching

    # Debug / diagnostic (alias for ``debug_mode``)
    debug: bool = False
    
    @classmethod
    def from_env(cls) -> "WISERoutingConfig":
        """Create config from environment variables.
        
        Environment variable mapping (faithful port of old code):
        - WISE_SAT_TIMEOUT → timeout_seconds (default 4800.0)
        - WISE_MAX_SAT_TIME → timeout_seconds (can only REDUCE, never increase)
        - WISE_MAX_RC2_TIME → rc2_timeout_seconds (can only REDUCE, never increase)
        - WISE_BOUNDARY_WEIGHT_ROW → boundary_soft_weight_row (default 1)
        - WISE_BOUNDARY_WEIGHT_COL → boundary_soft_weight_col (default 1)
        """
        def _apply_time_env(default_value: float, env_var: str) -> float:
            """Old code's _apply_time_env: env can only REDUCE, never increase."""
            env_val = os.environ.get(env_var)
            if not env_val:
                return default_value
            try:
                parsed = float(env_val)
            except ValueError:
                return default_value
            if parsed <= 0:
                return default_value
            return min(default_value, parsed)

        subgridsize_str = os.environ.get("WISE_SUBGRIDSIZE", "6,4,1")
        try:
            parts = [int(x.strip()) for x in subgridsize_str.split(",")]
            subgridsize = (parts[0], parts[1], parts[2]) if len(parts) >= 3 else (6, 4, 1)
        except (ValueError, IndexError):
            subgridsize = (6, 4, 1)
        
        base_pmax_str = os.environ.get("WISE_BASE_PMAX", "")
        base_pmax_in = int(base_pmax_str) if base_pmax_str.isdigit() else None

        # Timeout defaults: 4800s like old code
        base_sat_timeout = float(os.environ.get("WISE_SAT_TIMEOUT", "4800"))
        base_rc2_timeout = float(os.environ.get("WISE_RC2_TIMEOUT", "4800"))
        # Apply the old _apply_time_env semantics: env can only REDUCE
        sat_timeout = _apply_time_env(base_sat_timeout, "WISE_MAX_SAT_TIME")
        rc2_timeout = _apply_time_env(base_rc2_timeout, "WISE_MAX_RC2_TIME")
        
        return cls(
            timeout_seconds=sat_timeout,
            rc2_timeout_seconds=rc2_timeout,
            max_passes=int(os.environ.get("WISE_MAX_PASSES", "10")),
            use_maxsat=os.environ.get("WISE_USE_MAXSAT", "0") != "0",
            patch_enabled=os.environ.get("WISE_PATCH_ENABLED", "0") != "0",
            patch_height=int(os.environ.get("WISE_PATCH_HEIGHT", "4")),
            patch_width=int(os.environ.get("WISE_PATCH_WIDTH", "4")),
            debug_mode=os.environ.get("WISE_DEBUG", "0") != "0",
            debug=os.environ.get("WISE_DEBUG", "0") != "0",
            num_workers=int(os.environ.get("WISE_SAT_WORKERS", "1")),
            bt_soft_weight=int(os.environ.get("WISE_BT_SOFT_WEIGHT", "0")),
            boundary_soft_weight_row=int(os.environ.get("WISE_BOUNDARY_WEIGHT_ROW", "1")),
            boundary_soft_weight_col=int(os.environ.get("WISE_BOUNDARY_WEIGHT_COL", "1")),
            subgridsize=subgridsize,
            base_pmax_in=base_pmax_in,
            lookahead_rounds=int(os.environ.get("WISE_LOOKAHEAD", "2")),
            max_cycles=int(os.environ.get("WISE_MAX_CYCLES", "10")),
            bcf_steps=int(os.environ.get("WISE_BCF_STEPS", "6")),
            bcf_min=float(os.environ.get("WISE_BCF_MIN", "0.0")),
            min_passes=int(os.environ.get("WISE_MIN_PASSES", "1")),
            block_capacity=int(os.environ.get("WISE_BLOCK_CAPACITY", "2")),
            default_rows=int(os.environ.get("WISE_DEFAULT_ROWS", "3")),
            default_cols=int(os.environ.get("WISE_DEFAULT_COLS", "6")),
            patch_size=int(os.environ.get("WISE_PATCH_SIZE", "4")),
            patch_threshold=int(os.environ.get("WISE_PATCH_THRESHOLD", "64")),
            parallel_sat_search=os.environ.get("WISE_PARALLEL_SAT", "0") != "0",
            sat_workers=int(os.environ.get("WISE_SAT_WORKERS", "0")),
        )


__all__ = [
    "WISE_LOGGER_NAME",
    "wise_logger",
    "ROW_SWAP_HEATING",
    "COL_SWAP_HEATING",
    "ROW_SWAP_TIME_US",
    "COL_SWAP_TIME_US",
    "INITIAL_SPLIT_TIME_US",
    "SPLIT_HEATING",
    "H_PASS_TIME_US",
    "V_PASS_TIME_US",
    "_PYSAT_AVAILABLE",
    "WISERoutingConfig",
]
