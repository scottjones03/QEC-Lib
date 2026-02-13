# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/routers.py
"""
WISE SAT-based routers for trapped ion architectures.

This module provides the router classes that implement the Router interface
using SAT-based solving:
- WiseSatRouter: Core SAT-based router for WISE grids
- WisePatchRouter: Patch-based decomposition for large grids
- GreedyIonRouter: Fast heuristic fallback for small instances
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np

# Soft import of tqdm for progress bars (falls back to no-op)
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    class _tqdm:  # type: ignore[no-redef]
        """Minimal no-op tqdm stand-in."""
        def __init__(self, iterable=None, *a, **kw):
            self._it = iterable
        def __iter__(self):
            return iter(self._it) if self._it is not None else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            pass
        def set_postfix_str(self, s, refresh=True):
            pass
        def set_description(self, desc, refresh=True):
            pass
        def close(self):
            pass

from qectostim.experiments.hardware_simulation.core.compiler import (
    Router,
    RoutingResult,
    RoutingStrategy,
)
from qectostim.experiments.hardware_simulation.core.architecture import (
    LayoutTracker,
)
from qectostim.experiments.hardware_simulation.core.pipeline import QubitMapping
from qectostim.experiments.hardware_simulation.core.sat_interface import (
    GridSATEncoder,
    SATSolution,
    PlacementRequirement,
    ConstraintType,
    SATRoutingConfig,
    PatchRoutingConfig,
    GridLayout as CoreGridLayout,
    SortingPass as CoreSortingPass,
    RoutingSchedule as CoreRoutingSchedule,
    InteractionRequirement as CoreInteractionRequirement,
)

# Import from sibling modules
from qectostim.experiments.hardware_simulation.trapped_ion.routing.sat_encoder import (
    WISESATContext,
    WISESATEncoder,
)
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
from qectostim.experiments.hardware_simulation.trapped_ion.transport import (
    Move as _Move,
    JunctionCrossing as _JC,
)
from qectostim.experiments.hardware_simulation.trapped_ion.routing.data_structures import (
    GridLayout,
    RoutingPass,
    RoutingSchedule,
    GatePairRequirement,
    compute_target_positions,
)
from qectostim.experiments.hardware_simulation.trapped_ion.routing.layout_utils import (
    NoFeasibleLayoutError,
    compute_patch_gating_capacity as _compute_patch_gating_capacity,
    compute_cross_boundary_prefs as _compute_cross_boundary_prefs,
    compute_target_layout_from_pairs as _compute_target_layout_from_pairs,
    pre_sat_sanity_checks as _pre_sat_sanity_checks,
    heuristic_phaseB_greedy_layout as _heuristic_phaseB_greedy_layout,
    heuristic_odd_even_reconfig as _heuristic_odd_even_reconfig,
)
from qectostim.experiments.hardware_simulation.trapped_ion.routing.solvers import (
    run_sat_with_timeout as _run_sat_with_timeout,
    run_rc2_with_timeout as _run_rc2_with_timeout,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.architecture import (
        HardwareArchitecture,
    )
    from ..architecture import Ion, QCCDNode as QCCDComponent, ManipulationTrap as Trap
    from ..operations import QCCDOperationBase

# pysat imports (conditional)
if _PYSAT_AVAILABLE:
    from pysat.formula import IDPool, CNF, WCNF
    from pysat.card import CardEnc, EncType
    from pysat.solvers import Minisat22, Solver
else:
    IDPool = None  # type: ignore[misc,assignment]
    CNF = None  # type: ignore[misc,assignment]
    WCNF = None  # type: ignore[misc,assignment]
    CardEnc = None  # type: ignore[misc,assignment]
    EncType = None  # type: ignore[misc,assignment]
    Minisat22 = None  # type: ignore[misc,assignment]
    Solver = None  # type: ignore[misc,assignment]


# =============================================================================
# Module-level parallel worker (must be picklable for ProcessPoolExecutor)
# =============================================================================

def _parallel_config_worker(
    cfg: Tuple[int, float],
    *,
    context_data: Dict[str, Any],
    config_dict: Dict[str, Any],
    optimize_round_start: int,
    use_soft_bt: bool,
    per_call_timeout: float,
    stop_event: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Worker function for parallel SAT config search.

    Runs in a subprocess. Performs binary search over ΣP for one
    (P_max, bcf) configuration. Faithful port of ``_wise_sat_config_worker``.
    """
    P_max, bcf = cfg

    # Reconstruct context
    ctx = WISESATContext(
        initial_layout=np.array(context_data["initial_layout"], dtype=int),
        target_positions=context_data["target_positions"],
        gate_pairs=context_data["gate_pairs"],
        full_gate_pairs=context_data["full_gate_pairs"],
        ions=context_data["ions"],
        n_rows=context_data["n_rows"],
        n_cols=context_data["n_cols"],
        num_rounds=context_data["num_rounds"],
        block_cells=context_data["block_cells"],
        block_fully_inside=context_data["block_fully_inside"],
        block_widths=context_data["block_widths"],
        num_blocks=context_data["num_blocks"],
        grid_origin=context_data.get("grid_origin", (0, 0)),
        cross_boundary_prefs=context_data.get("cross_boundary_prefs"),
        boundary_adjacent=context_data.get("boundary_adjacent"),
        ignore_initial_reconfig=context_data.get("ignore_initial_reconfig", False),
    )

    # Reconstruct config
    from qectostim.experiments.hardware_simulation.trapped_ion.routing.config import (
        WISERoutingConfig,
    )
    config = WISERoutingConfig(
        timeout_seconds=config_dict.get("timeout_seconds", 4800.0),
        boundary_capacity_factor=bcf,
        bt_soft_weight=config_dict.get("bt_soft_weight", 0),
        boundary_soft_weight_row=config_dict.get("boundary_soft_weight_row", 0),
        boundary_soft_weight_col=config_dict.get("boundary_soft_weight_col", 0),
        block_capacity=config_dict.get("block_capacity", 2),
        debug=config_dict.get("debug", False),
    )

    R = ctx.num_rounds
    rounds_under_sum = max(1, R - optimize_round_start)
    max_bound_B = rounds_under_sum * P_max
    low = 1
    high = max_bound_B
    best: Optional[Dict[str, Any]] = None

    while low <= high:
        if stop_event is not None and stop_event.is_set():
            break

        mid = (low + high) // 2
        try:
            enc = WISESATEncoder(
                rows=ctx.n_rows, cols=ctx.n_cols,
                config=config, use_maxsat=use_soft_bt,
            )
            enc.initialize(ctx, P_max, sum_bound_B=mid)
            enc.add_all_constraints()
            sol = enc.solve(timeout=per_call_timeout, in_process=True)

            if sol.satisfiable:
                model = getattr(enc, '_last_model', None)
                if model is not None:
                    schedule = enc.decode_schedule(model)
                    # Extract per-round usage
                    model_set = {lit for lit in model if lit > 0}
                    usage: List[int] = []
                    for r in range(R):
                        P_bound = (
                            enc._pass_bounds[r]
                            if r < len(enc._pass_bounds) else P_max
                        )
                        active = 0
                        for p in range(P_bound):
                            u_var = enc.var_u(r, p)
                            if u_var <= enc.vpool.top and u_var in model_set:
                                active += 1
                        usage.append(active)
                    sum_usage = sum(usage[optimize_round_start:])

                    best = {
                        "P_max": P_max,
                        "bcf": bcf,
                        "sum_bound_B": mid,
                        "schedule": schedule,
                        "per_round_usage": usage,
                        "sum_usage": sum_usage,
                        "sat": True,
                        "status": "ok",
                    }
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                low = mid + 1
        except Exception as e:
            wise_logger.debug(
                "Parallel worker P_max=%d bcf=%.2f ΣP=%d error: %s",
                P_max, bcf, mid, e,
            )
            low = mid + 1

    return best


# =============================================================================
# WISE SAT Router
# =============================================================================

class WiseSatRouter(Router):
    """SAT-based optimal routing for WISE grid architectures.
    
    Uses SAT/MaxSAT solving to find optimal ion permutations that:
    1. Minimize total routing passes
    2. Bring gate pairs to adjacent positions
    3. Respect capacity constraints
    
    The solver encodes:
    - Ion positions as Boolean variables
    - Odd-even transposition sort structure (H-V phases)
    - Gate pair adjacency requirements
    - Optional soft constraints for optimization
    
    Example
    -------
    >>> router = WiseSatRouter(config=WISERoutingConfig(timeout_seconds=30))
    >>> result = router.route_batch(pairs, mapping, architecture)
    >>> if result.success:
    ...     for op in result.operations:
    ...         print(op)
    
    Notes
    -----
    This is a refactored port of the SAT solver from old/utils/qccd_operations.py.
    The core SAT encoding is in _build_sat_formula().
    
    See Also
    --------
    WisePatchRouter : For large grids, uses patch decomposition.
    """
    
    def __init__(
        self,
        config: Optional[WISERoutingConfig] = None,
        name: str = "wise_sat_router",
    ):
        super().__init__(RoutingStrategy.GLOBAL_OPTIMIZATION, name)
        self.config = config or WISERoutingConfig()
        self._sat_available = self._check_sat_available()
    
    def _check_sat_available(self) -> bool:
        """Check if pysat is available."""
        try:
            from pysat.formula import CNF
            from pysat.solvers import Minisat22
            return True
        except ImportError:
            wise_logger.warning(
                "pysat not available. WiseSatRouter will use fallback heuristics."
            )
            return False
    
    def route_gate(
        self,
        gate_qubits: Tuple[int, ...],
        current_mapping: QubitMapping,
        architecture: "HardwareArchitecture",
    ) -> RoutingResult:
        """Route a single gate (delegates to route_batch)."""
        if len(gate_qubits) < 2:
            return RoutingResult(
                success=True,
                operations=[],
                cost=0.0,
                metrics={"note": "single-qubit gate, no routing needed"},
            )
        
        q1, q2 = gate_qubits[:2]
        pairs = [(q1, q2)]
        
        return self.route_batch(
            physical_pairs=pairs,
            current_mapping=current_mapping,
            architecture=architecture,
        )
    
    def route_batch(
        self,
        physical_pairs: List[Tuple[int, int]],
        current_mapping: QubitMapping,
        architecture: Optional["HardwareArchitecture"] = None,
        layout_tracker: Optional[LayoutTracker] = None,
        initial_layout: Optional[GridLayout] = None,
        *,
        lookahead_pairs: Optional[List[List[Tuple[int, int]]]] = None,
        bt_positions: Optional[List[Dict[int, Tuple[int, int]]]] = None,
        full_gate_pairs: Optional[List[List[Tuple[int, int]]]] = None,
        col_offset: int = 0,
        grid_origin: Tuple[int, int] = (0, 0),
        cross_boundary_prefs: Optional[List[Dict[int, Set[str]]]] = None,
        boundary_adjacent: Optional[Dict[str, bool]] = None,
        ignore_initial_reconfig: bool = False,
    ) -> RoutingResult:
        """Route a batch of gate pairs using SAT solving.
        
        Parameters
        ----------
        physical_pairs : List[Tuple[int, int]]
            List of ion pairs that need to interact.
        current_mapping : QubitMapping
            Current logical-to-physical qubit mapping.
        architecture : Optional[HardwareArchitecture]
            Target architecture (provides grid dimensions).
        layout_tracker : Optional[LayoutTracker]
            Tracker for current ion positions.
        initial_layout : Optional[GridLayout]
            Explicit initial layout (overrides layout_tracker).
        lookahead_pairs : Optional[List[List[Tuple[int, int]]]]
            Future gate batches for multi-round lookahead routing.
        bt_positions : Optional[List[Dict[int, Tuple[int, int]]]]
            Block-target positions per round (from prior routing).
            Empty for fresh solves.
        full_gate_pairs : Optional[List[List[Tuple[int, int]]]]
            Full gate pairs including all rounds (for outer pair
            awareness in constraint generation).
        col_offset : int
            Column offset for patch sub-grid alignment.
        grid_origin : Tuple[int, int]
            (row_offset, col_offset) for coordinate translation.
        cross_boundary_prefs : Optional
            Cross-boundary directional preferences per round.
        boundary_adjacent : Optional
            Boundary adjacency flags for patch routing.
        ignore_initial_reconfig : bool
            Whether the first round starts from an arbitrary layout.
            
        Returns
        -------
        RoutingResult
            Result containing operations and metrics.
        """
        if not physical_pairs:
            return RoutingResult(
                success=True, operations=[], cost=0.0,
                metrics={"note": "empty batch"},
            )
        
        # Determine grid dimensions
        n_rows, n_cols = self._get_grid_dims(architecture, initial_layout)
        if n_rows <= 0 or n_cols <= 0:
            return RoutingResult(
                success=False,
                metrics={"error": "Cannot determine grid dimensions"},
            )
        
        # Stash architecture for capacity detection in _sat_route
        self._current_architecture = architecture
        
        # Get initial layout
        if initial_layout is not None:
            layout = initial_layout.grid.copy()
        elif layout_tracker is not None:
            layout = layout_tracker.get_layout_grid(n_rows, n_cols)
        else:
            # Default layout
            layout = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
        
        # If SAT not available, use heuristic
        if not self._sat_available:
            return self._heuristic_route(
                layout, physical_pairs, n_rows, n_cols
            )
        
        # Build and solve SAT problem
        return self._sat_route(
            layout, physical_pairs, n_rows, n_cols,
            bt_positions=bt_positions,
            lookahead_pairs=lookahead_pairs,
            full_gate_pairs=full_gate_pairs,
            col_offset=col_offset,
            grid_origin=grid_origin,
            cross_boundary_prefs=cross_boundary_prefs,
            boundary_adjacent=boundary_adjacent,
            ignore_initial_reconfig=ignore_initial_reconfig,
        )
    
    def _get_grid_dims(
        self,
        architecture: Optional["HardwareArchitecture"],
        initial_layout: Optional[GridLayout],
    ) -> Tuple[int, int]:
        """Extract grid dimensions from architecture or layout."""
        if initial_layout is not None:
            return initial_layout.grid.shape
        
        if architecture is not None:
            # Try to get from architecture attributes
            if hasattr(architecture, 'n_rows') and hasattr(architecture, 'n_cols'):
                return architecture.n_rows, architecture.n_cols
            if hasattr(architecture, 'grid_shape'):
                return architecture.grid_shape
        
        # Fallback to config
        return self.config.default_rows, self.config.default_cols
    
    def _sat_route(
        self,
        layout: np.ndarray,
        pairs: List[Tuple[int, int]],
        n_rows: int,
        n_cols: int,
        *,
        bt_positions: Optional[List[Dict[int, Tuple[int, int]]]] = None,
        lookahead_pairs: Optional[List[List[Tuple[int, int]]]] = None,
        full_gate_pairs: Optional[List[List[Tuple[int, int]]]] = None,
        col_offset: int = 0,
        grid_origin: Tuple[int, int] = (0, 0),
        cross_boundary_prefs: Optional[List[Dict[int, Set[str]]]] = None,
        boundary_adjacent: Optional[Dict[str, bool]] = None,
        ignore_initial_reconfig: bool = False,
    ) -> RoutingResult:
        """Route using SAT solver — faithful port of ``_optimal_QMR_for_WISE``.

        Key differences from the old buggy implementation:
        * BT (block-target) positions default to **empty** for fresh solves
          (the old code only populates them from prior multi-round solutions).
        * Blocks are capacity-wide column bands spanning all rows
          (via ``_compute_blocks``).
        * Multiple ``(P_max, boundary_capacity_factor)`` configurations are
          explored.  Each worker performs **binary search over ΣP** to find
          the tightest satisfiable schedule.
        * The best result across all configs is chosen.
        * Fast-paths skip SAT entirely for co-located or single-pair batches.
        """
        # ---- GAP 1 FIX: default boundary_adjacent to all-False ----
        # OLD _optimal_QMR_for_WISE defaults to all-False when called standalone.
        # The encoder defaults to all-True which is only correct for patch sub-calls
        # where patch_and_route explicitly computes boundary_adjacent per-patch.
        if boundary_adjacent is None:
            boundary_adjacent = {
                "top": False, "bottom": False,
                "left": False, "right": False,
            }

        # ---- Fast path: empty pairs ----
        if not pairs:
            return RoutingResult(
                success=True, operations=[], cost=0.0,
                metrics={"note": "empty pairs", "_final_layout": layout},
            )

        # ---- Fast path: check co-location ----
        # If ALL pairs are already in the same row AND block, no routing needed
        capacity = self.config.block_capacity if hasattr(self.config, 'block_capacity') and self.config.block_capacity else 2
        if hasattr(self, '_current_architecture') and self._current_architecture is not None:
            arch = self._current_architecture
            if hasattr(arch, 'ions_per_segment'):
                capacity = arch.ions_per_segment
            elif hasattr(arch, 'capacity'):
                capacity = arch.capacity

        ion_pos: Dict[int, Tuple[int, int]] = {}
        for r in range(n_rows):
            for c in range(n_cols):
                ion_pos[int(layout[r, c])] = (r, c)

        all_colocated = True
        for q1, q2 in pairs:
            pos1 = ion_pos.get(q1)
            pos2 = ion_pos.get(q2)
            if pos1 is None or pos2 is None:
                all_colocated = False
                break
            if pos1[0] != pos2[0]:  # different row
                all_colocated = False
                break
            if pos1[1] // capacity != pos2[1] // capacity:  # different block
                all_colocated = False
                break

        if all_colocated:
            wise_logger.debug(
                "All %d pairs already co-located, skipping SAT", len(pairs),
            )
            return RoutingResult(
                success=True, operations=[], cost=0.0,
                metrics={
                    "passes": 0,
                    "time_us": 0.0,
                    "sum_passes": 0,
                    "note": "co-located",
                    "_final_layout": layout,
                },
            )

        # ---- Build multi-round pairs list (current + lookahead) ----
        P_arr: List[List[Tuple[int, int]]] = [pairs]
        if lookahead_pairs:
            P_arr.extend(lookahead_pairs)
        R = len(P_arr)

        if full_gate_pairs is None:
            full_gate_pairs = list(P_arr)

        # ---- BT: default to empty for fresh solves ----
        if bt_positions is None or len(bt_positions) == 0:
            BT: List[Dict[int, Tuple[int, int]]] = [{} for _ in range(R)]
        else:
            BT = list(bt_positions)
            while len(BT) < R:
                BT.append({})

        # ---- Ions ----
        ions = sorted(set(int(x) for x in layout.flatten()))

        # ---- BT: verify pin integrity (faithful port of old pre-check) ----
        BT = self._verify_bt_pin_integrity(BT, ions)

        # ---- Block structure (capacity-wide column bands) ----
        block_cells, block_widths, block_inside = self._compute_blocks(
            n_rows, n_cols, capacity, col_offset=col_offset,
        )

        # ---- Build context ----
        ctx = WISESATContext(
            initial_layout=layout,
            target_positions=BT,
            gate_pairs=P_arr,
            full_gate_pairs=full_gate_pairs,
            ions=ions,
            n_rows=n_rows,
            n_cols=n_cols,
            num_rounds=R,
            block_cells=block_cells,
            block_fully_inside=block_inside,
            block_widths=block_widths,
            num_blocks=len(block_cells),
            debug_diag=self.config.debug,
            grid_origin=grid_origin,
            cross_boundary_prefs=cross_boundary_prefs,
            boundary_adjacent=boundary_adjacent,
            ignore_initial_reconfig=ignore_initial_reconfig,
        )

        # ---- Configuration search grid ----
        # Faithful port of _enumerate_pmax_configs from qccd_operations.py.
        # The old code uses:
        #   step = max(int(floor((limit - base) / 4)), 1)
        #   capacity_steps = 6, capacity_min = 0.0
        # and a per-worker SAT timeout + global budget.
        timeout = self.config.timeout_seconds
        optimize_round_start = 1 if (ignore_initial_reconfig and R > 0) else 0

        base_pmax = max(self.config.base_pmax_in or R, 1)
        limit_pmax = base_pmax + n_rows + n_cols

        bcf_steps = max(self.config.bcf_steps, 1)
        bcf_min = self.config.bcf_min
        if bcf_steps <= 1:
            bcf_factors = [1.0]
        else:
            bcf_factors = [
                max(bcf_min, 1.0 - i * (1.0 - bcf_min) / (bcf_steps - 1))
                for i in range(bcf_steps)
            ]

        # Build P_max list — use floor(range/4) step like old code for finer
        # granularity.  The old code starts at base_pmax, not 2.
        effective_base = max(base_pmax, 1)
        step = max(int(np.floor((limit_pmax - effective_base) / 4)), 1)
        pmax_values: List[int] = list(range(effective_base, limit_pmax + 1, step))
        # Always include limit_pmax if not already present
        if pmax_values and pmax_values[-1] != limit_pmax:
            pmax_values.append(limit_pmax)
        if not pmax_values:
            pmax_values = [effective_base]

        # Per-call timeout: old code used full max_sat_time (4800s) per
        # worker since they ran in parallel. For sequential execution we
        # allocate a proportional share of the global budget, but give each
        # config enough time to actually complete (at least 3s).
        n_configs_est = max(len(pmax_values) * len(bcf_factors), 1)
        per_call_timeout = max(min(timeout / max(n_configs_est, 1), timeout), 3.0)

        wise_logger.debug(
            "SAT: P_max∈%s, bcf∈%s, per_call=%.1fs",
            pmax_values, [f"{f:.2f}" for f in bcf_factors], per_call_timeout,
        )

        # Global wall-clock budget for this entire _sat_route call.
        _global_budget_s = max(timeout, 10.0)
        _t_start = time.time()

        bt_has_content = any(bt for bt in BT)
        # Old code: bt_soft_enabled = bt_soft and BT has content
        # bt_soft_weight_value = max(100, max(wB_col, wB_row, 1) * 10)
        # In the new architecture, bt_soft_weight=0 means "auto-compute from
        # boundary weights" (matching the old dynamic computation).
        effective_bt_soft_weight = self.config.bt_soft_weight
        if effective_bt_soft_weight == 0 and bt_has_content:
            base_pref_weight = max(
                self.config.boundary_soft_weight_col,
                self.config.boundary_soft_weight_row,
                1,
            )
            effective_bt_soft_weight = max(100, base_pref_weight * 10)
        use_soft_bt = (
            bt_has_content
            and effective_bt_soft_weight > 0
        )
        # Temporarily set the effective weight on config so the encoder sees it
        if use_soft_bt:
            self.config.bt_soft_weight = effective_bt_soft_weight

        # Build flat config list: (P_max, bcf) pairs
        config_list: List[Tuple[int, float]] = [
            (pm, bcf) for pm in pmax_values for bcf in bcf_factors
        ]

        # ---- Parallel path (faithful port of old ProcessPoolExecutor) ----
        if self.config.parallel_sat_search and len(config_list) > 1:
            best_result = self._parallel_sat_search(
                config_list=config_list,
                ctx=ctx,
                layout=layout,
                ions=ions,
                n_rows=n_rows,
                n_cols=n_cols,
                R=R,
                BT=BT,
                optimize_round_start=optimize_round_start,
                use_soft_bt=use_soft_bt,
                per_call_timeout=per_call_timeout,
                global_budget_s=_global_budget_s,
                ignore_initial_reconfig=ignore_initial_reconfig,
            )
        else:
            # ---- Sequential path (default) ----
            best_result = self._sequential_sat_search(
                config_list=config_list,
                ctx=ctx,
                n_rows=n_rows,
                n_cols=n_cols,
                R=R,
                optimize_round_start=optimize_round_start,
                use_soft_bt=use_soft_bt,
                per_call_timeout=per_call_timeout,
                global_budget_s=_global_budget_s,
                step=step,
                ignore_initial_reconfig=ignore_initial_reconfig,
            )

        # ---- Decode best result ----
        if best_result is not None:
            schedule = best_result["schedule"]

            # Apply grid_origin offset to schedule coordinates.
            # Faithful port of old code's coordinate rewriting: swap
            # positions are emitted in patch-local coords by the SAT
            # encoder and must be shifted to global coords when the
            # patch originates at a non-zero (row, col).
            if grid_origin != (0, 0):
                row_off, col_off = grid_origin
                for round_passes in schedule:
                    for pass_info in round_passes:
                        if isinstance(pass_info, dict):
                            if "h_swaps" in pass_info:
                                pass_info["h_swaps"] = [
                                    (r + row_off, c + col_off)
                                    for r, c in pass_info["h_swaps"]
                                ]
                            if "v_swaps" in pass_info:
                                pass_info["v_swaps"] = [
                                    (r + row_off, c + col_off)
                                    for r, c in pass_info["v_swaps"]
                                ]

            P_max = best_result["P_max"]
            enc = best_result["encoder"]
            model = best_result["model"]

            # Reconstruct per-round final layouts from model
            final_layouts = self._decode_layouts_from_model(
                model, enc, layout, ions, n_rows, n_cols, R,
                ignore_initial_reconfig,
            )

            # Convert schedule to flat operations list
            operations = self._schedule_rounds_to_operations(schedule)

            # Final layout after all rounds
            final_layout = final_layouts[-1] if final_layouts else layout

            wise_logger.info(
                "SAT solved: P_max=%d, bcf=%.2f, ΣP=%d",
                P_max, best_result["bcf"], best_result["sum_usage"],
            )

            return RoutingResult(
                success=True,
                operations=operations,
                cost=P_max * (H_PASS_TIME_US + V_PASS_TIME_US),
                metrics={
                    "passes": P_max,
                    "time_us": P_max * (H_PASS_TIME_US + V_PASS_TIME_US),
                    "sum_passes": best_result["sum_usage"],
                    "bcf": best_result["bcf"],
                    "per_round_usage": best_result["per_round_usage"],
                    "_final_layouts": final_layouts,
                    "_final_layout": final_layout,
                },
                final_mapping=None,
            )

        # Fall back to heuristic
        wise_logger.warning("SAT solve failed for all configs, using heuristic fallback")
        return self._heuristic_route(layout, pairs, n_rows, n_cols)

    # ------------------------------------------------------------------
    # Sequential SAT config search (binary search over ΣP per config)
    # ------------------------------------------------------------------

    def _sequential_sat_search(
        self,
        config_list: List[Tuple[int, float]],
        ctx: WISESATContext,
        n_rows: int,
        n_cols: int,
        R: int,
        optimize_round_start: int,
        use_soft_bt: bool,
        per_call_timeout: float,
        global_budget_s: float,
        step: int,
        ignore_initial_reconfig: bool,
    ) -> Optional[Dict[str, Any]]:
        """Sequentially search (P_max, bcf) configs with binary search over ΣP.

        Faithful port of ``_wise_sat_config_worker`` binary-search logic.
        """
        _t_start = time.time()
        best_result: Optional[Dict[str, Any]] = None
        best_score: Optional[Tuple] = None
        _found_first = False

        for P_max, bcf in config_list:
            # Check global budget
            elapsed = time.time() - _t_start
            if elapsed >= global_budget_s:
                wise_logger.debug(
                    "SAT global budget exhausted (%.1fs); returning best-so-far",
                    elapsed,
                )
                break

            # Once we have a solution, skip P_max values that can't improve
            if best_result is not None and P_max > best_result["P_max"] + step:
                continue

            # Once we have a solution, we can use a somewhat shorter timeout
            # since we're just trying to improve. But don't cut too aggressively
            # (old code used full timeout for all workers).
            call_timeout = (
                per_call_timeout if not _found_first
                else max(per_call_timeout * 0.5, 5.0)
            )

            rounds_under_sum = max(1, R - optimize_round_start)
            max_bound_B = rounds_under_sum * P_max
            low = 1
            high = max_bound_B
            cfg_best: Optional[Dict[str, Any]] = None

            while low <= high:
                if (time.time() - _t_start) >= global_budget_s:
                    break

                mid = (low + high) // 2
                try:
                    enc = WISESATEncoder(
                        rows=n_rows,
                        cols=n_cols,
                        config=self.config,
                        use_maxsat=use_soft_bt,
                    )
                    old_bcf = self.config.boundary_capacity_factor
                    self.config.boundary_capacity_factor = bcf
                    try:
                        enc.initialize(ctx, P_max, sum_bound_B=mid)
                        enc.add_all_constraints()
                        sol = enc.solve(
                            timeout=call_timeout,
                            in_process=True,
                        )
                    finally:
                        self.config.boundary_capacity_factor = old_bcf

                    if sol.satisfiable:
                        model = getattr(enc, '_last_model', None)
                        if model is not None:
                            schedule = enc.decode_schedule(model)
                            per_round_usage = self._extract_round_pass_usage(
                                model, enc, R, P_max, ignore_initial_reconfig,
                                n_rows, n_cols, optimize_round_start,
                            )
                            sum_usage = sum(
                                per_round_usage[optimize_round_start:]
                            )
                            cfg_best = {
                                "P_max": P_max,
                                "bcf": bcf,
                                "sum_bound_B": mid,
                                "schedule": schedule,
                                "model": model,
                                "per_round_usage": per_round_usage,
                                "sum_usage": sum_usage,
                                "encoder": enc,
                            }
                            high = mid - 1
                        else:
                            low = mid + 1
                    else:
                        low = mid + 1
                except NoFeasibleLayoutError:
                    low = mid + 1
                except Exception as e:
                    wise_logger.debug(
                        "SAT config P_max=%d bcf=%.2f ΣP=%d error: %s",
                        P_max, bcf, mid, e,
                    )
                    low = mid + 1

            if cfg_best is not None:
                _found_first = True
                score = (
                    -cfg_best["bcf"],
                    cfg_best["sum_usage"],
                    cfg_best["P_max"],
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_result = cfg_best
                    wise_logger.debug(
                        "SAT: new best P_max=%d bcf=%.2f ΣP=%d usage=%d",
                        cfg_best["P_max"], cfg_best["bcf"],
                        cfg_best["sum_bound_B"], cfg_best["sum_usage"],
                    )

        return best_result

    # ------------------------------------------------------------------
    # Parallel SAT config search (port of old ProcessPoolExecutor path)
    # ------------------------------------------------------------------

    def _parallel_sat_search(
        self,
        config_list: List[Tuple[int, float]],
        ctx: WISESATContext,
        layout: np.ndarray,
        ions: List[int],
        n_rows: int,
        n_cols: int,
        R: int,
        BT: List[Dict[int, Tuple[int, int]]],
        optimize_round_start: int,
        use_soft_bt: bool,
        per_call_timeout: float,
        global_budget_s: float,
        ignore_initial_reconfig: bool,
    ) -> Optional[Dict[str, Any]]:
        """Run parallel SAT config search across (P_max, bcf) pairs.

        Faithful port of old ``ProcessPoolExecutor`` + ``_wise_sat_config_worker``.
        Each worker gets a (P_max, bcf) pair and binary-searches over ΣP.
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from .solvers import wise_safe_sat_pool_workers

        max_workers = wise_safe_sat_pool_workers(len(config_list))
        if self.config.sat_workers > 0:
            max_workers = min(self.config.sat_workers, len(config_list))

        wise_logger.info(
            "Launching parallel SAT pool: %d configs × %d workers",
            len(config_list), max_workers,
        )

        # Serialisable context data for workers
        context_data = {
            "initial_layout": ctx.initial_layout.tolist(),
            "target_positions": [
                {int(k): v for k, v in d.items()} for d in ctx.target_positions
            ],
            "gate_pairs": ctx.gate_pairs,
            "full_gate_pairs": ctx.full_gate_pairs,
            "ions": ctx.ions,
            "n_rows": ctx.n_rows,
            "n_cols": ctx.n_cols,
            "num_rounds": ctx.num_rounds,
            "block_cells": ctx.block_cells,
            "block_fully_inside": ctx.block_fully_inside,
            "block_widths": ctx.block_widths,
            "num_blocks": ctx.num_blocks,
            "grid_origin": ctx.grid_origin,
            "cross_boundary_prefs": ctx.cross_boundary_prefs,
            "boundary_adjacent": ctx.boundary_adjacent,
            "ignore_initial_reconfig": ctx.ignore_initial_reconfig,
        }

        # Manager for cross-process stop_event
        try:
            manager = mp.Manager()
            stop_event = manager.Event()
        except Exception:
            stop_event = None

        try:
            pool_ctx = mp.get_context("fork")
        except ValueError:
            pool_ctx = mp.get_context()

        _t_start = time.time()
        best_result: Optional[Dict[str, Any]] = None
        best_score: Optional[Tuple] = None

        try:
            executor = ProcessPoolExecutor(
                max_workers=max_workers, mp_context=pool_ctx
            )
            futures = {}

            for cfg in config_list:
                fut = executor.submit(
                    _parallel_config_worker,
                    cfg,
                    context_data=context_data,
                    config_dict=self._config_to_dict(),
                    optimize_round_start=optimize_round_start,
                    use_soft_bt=use_soft_bt,
                    per_call_timeout=per_call_timeout,
                    stop_event=stop_event,
                )
                futures[fut] = cfg

            pending = set(futures.keys())
            while pending:
                if (time.time() - _t_start) >= global_budget_s:
                    wise_logger.info(
                        "Parallel SAT budget exhausted; collecting results"
                    )
                    if stop_event is not None:
                        stop_event.set()
                    for fut in pending:
                        fut.cancel()
                    break

                import concurrent.futures as cf
                done, pending = cf.wait(
                    pending, timeout=0.5,
                    return_when=cf.FIRST_COMPLETED,
                )
                for fut in done:
                    try:
                        res = fut.result()
                    except Exception as e:
                        wise_logger.warning("Parallel SAT worker error: %s", e)
                        continue
                    if res is None or not res.get("sat"):
                        continue
                    score = (
                        -res.get("bcf", 1.0),
                        res.get("sum_usage", 999999),
                        res.get("P_max", 999999),
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_result = res
                        wise_logger.info(
                            "Parallel SAT: new best P_max=%d bcf=%.2f ΣP=%d",
                            res["P_max"], res["bcf"], res.get("sum_usage", -1),
                        )

        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        # The parallel worker can't return the encoder object (unpicklable).
        # Re-solve the best config once locally to get model + encoder for
        # layout decoding.
        if best_result is not None and "encoder" not in best_result:
            P_max = best_result["P_max"]
            bcf = best_result["bcf"]
            # GAP 3 FIX: use *relaxed* sum_bound (rounds_under_sum * P_max)
            # for the rebuild, not the tight best-found bound.  The tight
            # bound was found inside the worker, but the main-process
            # rebuild must allow the full range so the encoder can succeed.
            rounds_under_sum_rebuild = max(1, R - optimize_round_start)
            sum_bound = rounds_under_sum_rebuild * P_max
            enc = WISESATEncoder(
                rows=n_rows, cols=n_cols,
                config=self.config, use_maxsat=use_soft_bt,
            )
            old_bcf = self.config.boundary_capacity_factor
            self.config.boundary_capacity_factor = bcf
            try:
                enc.initialize(ctx, P_max, sum_bound_B=sum_bound)
                enc.add_all_constraints()
                sol = enc.solve(timeout=per_call_timeout * 2, in_process=True)
            finally:
                self.config.boundary_capacity_factor = old_bcf

            if sol.satisfiable:
                model = getattr(enc, '_last_model', None)
                if model is not None:
                    best_result["encoder"] = enc
                    best_result["model"] = model
                    best_result["schedule"] = enc.decode_schedule(model)

        return best_result

    def _config_to_dict(self) -> Dict[str, Any]:
        """Serialise config for subprocess workers."""
        return {
            "timeout_seconds": self.config.timeout_seconds,
            "boundary_capacity_factor": self.config.boundary_capacity_factor,
            "bt_soft_weight": self.config.bt_soft_weight,
            "boundary_soft_weight_row": self.config.boundary_soft_weight_row,
            "boundary_soft_weight_col": self.config.boundary_soft_weight_col,
            "block_capacity": self.config.block_capacity,
            "debug": self.config.debug,
        }

    def _extract_round_pass_usage(
        self,
        model: List[int],
        encoder: WISESATEncoder,
        R: int,
        P_max: int,
        ignore_initial_reconfig: bool,
        n_rows: int,
        n_cols: int,
        optimize_round_start: int,
    ) -> List[int]:
        """Extract per-round active pass count from a SAT model."""
        model_set = {lit for lit in model if lit > 0}
        usage: List[int] = []
        pass_bounds = encoder._pass_bounds

        for r in range(R):
            P_bound = pass_bounds[r] if r < len(pass_bounds) else P_max
            active = 0
            for p in range(P_bound):
                u_var = encoder.var_u(r, p)
                if u_var <= encoder.vpool.top and u_var in model_set:
                    active += 1
            usage.append(active)

        return usage

    def _decode_layouts_from_model(
        self,
        model: List[int],
        encoder: WISESATEncoder,
        initial_layout: np.ndarray,
        ions: List[int],
        n_rows: int,
        n_cols: int,
        R: int,
        ignore_initial_reconfig: bool,
    ) -> List[np.ndarray]:
        """Decode per-round final layouts from a SAT model."""
        model_set = {lit for lit in model if lit > 0}
        pass_bounds = encoder._pass_bounds
        layouts: List[np.ndarray] = []

        for r in range(R):
            P_final = pass_bounds[r] if r < len(pass_bounds) else 1
            nxt = np.empty_like(initial_layout)
            for d in range(n_rows):
                for c in range(n_cols):
                    found = None
                    for ion in ions:
                        v = encoder.var_a(r, P_final, d, c, ion)
                        if v <= encoder.vpool.top and v in model_set:
                            found = ion
                            break
                    if found is None:
                        # Fallback: keep initial layout value
                        found = int(initial_layout[d, c])
                    nxt[d, c] = found
            layouts.append(nxt)

        return layouts

    def _schedule_rounds_to_operations(
        self,
        schedule: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Convert multi-round schedule to a flat operations list."""
        operations: List[Dict[str, Any]] = []

        for round_passes in schedule:
            for pass_info in round_passes:
                if not isinstance(pass_info, dict):
                    continue

                h_swaps = pass_info.get("h_swaps", [])
                for row, col in h_swaps:
                    operations.append({
                        "type": "H_SWAP",
                        "row": row,
                        "col": col,
                    })

                v_swaps = pass_info.get("v_swaps", [])
                for row, col in v_swaps:
                    operations.append({
                        "type": "V_SWAP",
                        "row": row,
                        "col": col,
                    })

        return operations
    
    def _compute_blocks(
        self,
        n_rows: int,
        n_cols: int,
        capacity: int,
        col_offset: int = 0,
    ) -> Tuple[List[List[Tuple[int, int]]], List[int], List[bool]]:
        """Compute gating block structure.

        Blocks are **capacity-wide column bands** that span **all rows**,
        matching the ground-truth logic in ``_optimal_QMR_for_WISE``.  When
        the grid is a sub-patch starting at *col_offset* inside a larger
        global grid the block boundaries are aligned to global multiples of
        *capacity* so that adjacent patches share consistent block edges.

        Returns
        -------
        block_cells : List[List[Tuple[int, int]]]
            Cells in each block (tuples of ``(row, local_col)``).
        block_widths : List[int]
            Width (number of columns) of each block.
        block_inside : List[bool]
            Whether each block is fully inside the (sub-)grid.
        """
        first_block_idx = col_offset // capacity
        last_block_idx = (col_offset + n_cols - 1) // capacity
        num_blocks = last_block_idx - first_block_idx + 1

        global_patch_start = col_offset
        global_patch_end = col_offset + n_cols

        block_cells: List[List[Tuple[int, int]]] = []
        block_widths: List[int] = []
        block_inside: List[bool] = []

        for b_local in range(num_blocks):
            b_global = first_block_idx + b_local
            global_start = b_global * capacity
            global_end = (b_global + 1) * capacity
            local_start = max(0, global_start - col_offset)
            local_end = min(n_cols, global_end - col_offset)

            cells: List[Tuple[int, int]] = []
            for d in range(n_rows):
                for j_local in range(local_start, local_end):
                    cells.append((d, j_local))

            block_cells.append(cells)
            block_widths.append(local_end - local_start)
            block_inside.append(
                (global_start >= global_patch_start)
                and (global_end <= global_patch_end)
            )

        return block_cells, block_widths, block_inside

    @staticmethod
    def _verify_bt_pin_integrity(
        BT: List[Dict[int, Tuple[int, int]]],
        ions: List[int],
    ) -> List[Dict[int, Tuple[int, int]]]:
        """Verify BT pin integrity: no two ions pinned to same cell per round.

        Faithfully ported from the old code's pre-solve sanity check in
        ``_optimal_QMR_for_WISE``.  Detects and removes conflicting pins,
        keeping the first ion encountered for each cell.

        Parameters
        ----------
        BT : list of dicts mapping ion → (row, col)
        ions : list of all ions in the grid

        Returns
        -------
        Cleaned BT list with conflicts removed.
        """
        ions_set = set(ions)
        cleaned: List[Dict[int, Tuple[int, int]]] = []

        for r, bt in enumerate(BT):
            seen: Dict[Tuple[int, int], int] = {}
            clean_bt: Dict[int, Tuple[int, int]] = {}

            for ion, (d, c) in bt.items():
                if ion not in ions_set:
                    continue
                key = (d, c)
                if key in seen:
                    wise_logger.warning(
                        "BT[%d] pins ions %d and %d to same cell "
                        "(d=%d, c=%d); dropping ion %d",
                        r, seen[key], ion, d, c, ion,
                    )
                    continue
                seen[key] = ion
                clean_bt[ion] = (d, c)

            cleaned.append(clean_bt)

        return cleaned

    def _heuristic_route(
        self,
        layout: np.ndarray,
        pairs: List[Tuple[int, int]],
        n_rows: int,
        n_cols: int,
    ) -> RoutingResult:
        """Route using greedy heuristic (fallback).

        Computes a target layout from the gate pairs, then delegates to
        ``heuristic_odd_even_reconfig`` which performs iterative odd-even
        transposition sort.
        """
        try:
            target = _compute_target_layout_from_pairs(layout, pairs, capacity)
            capacity = 2
            if hasattr(self, '_current_architecture') and self._current_architecture is not None:
                capacity = getattr(self._current_architecture, 'ions_per_segment', 2)
            schedule, _, time_s = _heuristic_odd_even_reconfig(
                layout, target, k=capacity,
            )
            operations = self._schedule_rounds_to_operations([schedule])

            return RoutingResult(
                success=True,
                operations=operations,
                cost=time_s * 1e6,
                metrics={
                    "method": "heuristic",
                    "time_us": time_s * 1e6,
                },
            )
        except Exception as e:
            wise_logger.warning(f"Heuristic routing failed: {e}")
            return RoutingResult(
                success=False,
                metrics={"error": str(e)},
            )
    
    def _schedule_to_operations(
        self,
        schedule: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert schedule to operation dictionaries."""
        operations: List[Dict[str, Any]] = []
        
        for pass_info in schedule:
            if not isinstance(pass_info, dict):
                continue
            
            # H-phase swaps
            h_swaps = pass_info.get("h_swaps", [])
            for row, col in h_swaps:
                operations.append({
                    "type": "H_SWAP",
                    "row": row,
                    "col": col,
                })
            
            # V-phase swaps
            v_swaps = pass_info.get("v_swaps", [])
            for row, col in v_swaps:
                operations.append({
                    "type": "V_SWAP",
                    "row": row,
                    "col": col,
                })
        
        return operations
    
    def _decode_schedule_from_model(
        self,
        model: List[int],
        encoder: WISESATEncoder,
        n_rows: int,
        n_cols: int,
        num_passes: int,
    ) -> List[Dict[str, Any]]:
        """Decode SAT model to routing schedule."""
        model_set = frozenset(model)
        def lit_true(v: int) -> bool:
            return v in model_set
        
        schedule: List[Dict[str, Any]] = []
        
        for p in range(1, num_passes + 1):
            pass_info: Dict[str, Any] = {
                "pass": p,
                "h_swaps": [],
                "v_swaps": [],
            }
            
            # Decode H-phase swaps (even columns on even rows, odd on odd)
            for r in range(n_rows):
                for c in range(n_cols - 1):
                    parity = c % 2
                    if r % 2 == parity:
                        v = encoder.var_s_h(0, p, r, c)
                        if v <= encoder.vpool.top and lit_true(v):
                            pass_info["h_swaps"].append((r, c))
            
            # Decode V-phase swaps
            for c in range(n_cols):
                for r in range(n_rows - 1):
                    parity = r % 2
                    if c % 2 == parity:
                        v = encoder.var_s_v(0, p, r, c)
                        if v <= encoder.vpool.top and lit_true(v):
                            pass_info["v_swaps"].append((r, c))
            
            schedule.append(pass_info)
        
        return schedule

    def _compute_heuristic_schedule(
        self,
        layout: np.ndarray,
        pairs: List[Tuple[int, int]],
        n_rows: int,
        n_cols: int,
    ) -> List[Dict[str, Any]]:
        """Compute routing schedule using odd-even heuristic.
        
        Uses iterative odd-even transposition sort phases until
        all pairs are adjacent.
        """
        current = layout.copy()
        operations: List[Dict[str, Any]] = []
        
        # Ion position lookup
        def get_pos(ion: int) -> Tuple[int, int]:
            pos = np.where(current == ion)
            if len(pos[0]) == 0:
                return (-1, -1)
            return int(pos[0][0]), int(pos[1][0])
        
        # Check if pairs are satisfied
        def pairs_satisfied() -> bool:
            for a, b in pairs:
                ra, ca = get_pos(a)
                rb, cb = get_pos(b)
                if ra < 0 or rb < 0:
                    return False
                # Adjacent = same row, adjacent columns
                if ra == rb and abs(ca - cb) == 1:
                    continue
                return False
            return True
        
        max_iterations = self.config.max_passes * 2
        iteration = 0
        
        while not pairs_satisfied() and iteration < max_iterations:
            iteration += 1
            
            # H-phase: sort within rows
            h_swaps: List[Tuple[int, int]] = []
            for r in range(n_rows):
                for c in range(0, n_cols - 1, 2):
                    if iteration % 2 == 0:
                        c = c + 1 if c + 1 < n_cols - 1 else c
                    # Swap if needed to bring pairs together
                    if c + 1 < n_cols:
                        ion_left = current[r, c]
                        ion_right = current[r, c + 1]
                        
                        # Check if swap helps any pair
                        for a, b in pairs:
                            if ion_left == a and any(
                                current[r, c + 2:] == b if c + 2 < n_cols else []
                            ):
                                h_swaps.append((r, c))
                                current[r, c], current[r, c + 1] = (
                                    current[r, c + 1], current[r, c]
                                )
                                break
            
            if h_swaps:
                operations.append({
                    "type": "H_PHASE",
                    "swaps": h_swaps,
                })
            
            # V-phase: sort within columns (bring ions to correct rows)
            v_swaps: List[Tuple[int, int]] = []
            for c in range(n_cols):
                for r in range(0, n_rows - 1, 2):
                    if iteration % 2 == 0:
                        r = r + 1 if r + 1 < n_rows - 1 else r
                    if r + 1 < n_rows:
                        ion_top = current[r, c]
                        ion_bottom = current[r + 1, c]
                        
                        # Check if swap helps
                        for a, b in pairs:
                            if ion_top == a or ion_top == b:
                                pa, pb = get_pos(a), get_pos(b)
                                if pa[0] != pb[0]:  # Not same row yet
                                    v_swaps.append((r, c))
                                    current[r, c], current[r + 1, c] = (
                                        current[r + 1, c], current[r, c]
                                    )
                                    break
            
            if v_swaps:
                operations.append({
                    "type": "V_PHASE",
                    "swaps": v_swaps,
                })
        
        # Convert to schedule format
        schedule: List[Dict[str, Any]] = []
        for op in operations:
            if op["type"] == "H_PHASE":
                for row, col in op["swaps"]:
                    # Group by parity
                    parity = col % 2
                    # Find or create pass entry
                    matching = [
                        s for s in schedule 
                        if s.get("h_parity") == parity and "h_swaps" in s
                    ]
                    if matching:
                        matching[-1]["h_swaps"].append((row, col))
                    else:
                        schedule.append({
                            "h_parity": parity,
                            "h_swaps": [(row, col)],
                            "v_swaps": [],
                        })
            elif op["type"] == "V_PHASE":
                for row, col in op["swaps"]:
                    parity = row % 2
                    matching = [
                        s for s in schedule
                        if s.get("v_parity") == parity and "v_swaps" in s
                    ]
                    if matching:
                        matching[-1]["v_swaps"].append((row, col))
                    else:
                        if schedule:
                            schedule[-1]["v_swaps"].append((row, col))
                            schedule[-1]["v_parity"] = parity
                        else:
                            schedule.append({
                                "h_swaps": [],
                                "v_parity": parity,
                                "v_swaps": [(row, col)],
                            })
        
        return schedule

    def _emit_operations_from_schedule(
        self,
        schedule: List[Dict[str, Any]],
        n_rows: int,
        n_cols: int,
    ) -> List[Dict[str, Any]]:
        """Convert a routing schedule to operation dictionaries.
        
        Groups operations by H-V phase for parallel execution.
        """
        operations: List[Dict[str, Any]] = []
        
        for pass_info in schedule:
            # Emit H-phase swaps (grouped by parity for parallel execution)
            h_swaps = pass_info.get("h_swaps", [])
            if h_swaps:
                # Group by parity
                even_group: List[Tuple[int, int]] = []
                odd_group: List[Tuple[int, int]] = []
                
                for row, col in h_swaps:
                    if (row + col) % 2 == 0:
                        even_group.append((row, col))
                    else:
                        odd_group.append((row, col))
                
                for parity_group in [even_group, odd_group]:
                    if not parity_group:
                        continue
                    for r, c in parity_group:
                        operations.append({
                            "type": "H_SWAP",
                            "row": r,
                            "col": c,
                        })
                    operations.append({"type": "PASS_BOUNDARY"})
            
            # Emit V-phase swaps (grouped similarly)
            v_swaps = pass_info.get("v_swaps", [])
            if v_swaps:
                even_group = []
                odd_group = []
                
                for row, col in v_swaps:
                    if (row + col) % 2 == 0:
                        even_group.append((row, col))
                    else:
                        odd_group.append((row, col))
                
                for parity_group in [even_group, odd_group]:
                    if not parity_group:
                        continue
                    for r, c in parity_group:
                        operations.append({
                            "type": "V_SWAP",
                            "row": r,
                            "col": c,
                        })
                    operations.append({"type": "PASS_BOUNDARY"})

        # Remove trailing boundary (not needed)
        if operations and operations[-1].get("type") == "PASS_BOUNDARY":
            operations.pop()
        
        return operations


# =============================================================================
# Patch Router (for large grids)
# =============================================================================

class WisePatchRouter(WiseSatRouter):
    """Patch-based WISE routing for large grids.
    
    Decomposes the grid into overlapping patches and solves each
    patch independently, then merges the solutions. This scales
    better than full SAT for large grids.
    
    Algorithm:
    1. Divide grid into patches (with overlap for boundary handling)
    2. Assign gate pairs to patches based on qubit locations
    3. For each patch:
       a. Extract sub-layout and sub-pairs
       b. Solve using WiseSatRouter
       c. Collect routing operations
    4. Merge solutions and handle boundary interactions
    5. Iteratively refine until all pairs satisfied
    
    Uses checkerboard decomposition to avoid boundary conflicts:
    - Phase 1: Route all "white" patches (no shared boundaries)
    - Phase 2: Route all "black" patches
    
    Attributes
    ----------
    overlap : int
        Number of cells to overlap between adjacent patches.
    max_iterations : int
        Maximum number of refinement iterations.
    """
    
    def __init__(
        self,
        config: Optional[WISERoutingConfig] = None,
        name: str = "wise_patch_router",
        overlap: int = 1,
        max_iterations: int = 5,
    ):
        super().__init__(config, name)
        if self.config:
            self.config.patch_enabled = True
        self.overlap = overlap
        self.max_iterations = max_iterations
    
    def route_batch(
        self,
        physical_pairs: List[Tuple[int, int]],
        current_mapping: QubitMapping,
        architecture: Optional["HardwareArchitecture"] = None,
        layout_tracker: Optional[LayoutTracker] = None,
        initial_layout: Optional[GridLayout] = None,
        **kwargs,
    ) -> RoutingResult:
        """Route using patch decomposition."""
        if not physical_pairs:
            return RoutingResult(
                success=True, operations=[], cost=0.0,
                metrics={"note": "empty batch"},
            )
        
        n_rows, n_cols = self._get_grid_dims(architecture, initial_layout)
        
        # For small grids, delegate to parent
        if n_rows * n_cols <= self.config.patch_threshold:
            return super().route_batch(
                physical_pairs, current_mapping, architecture,
                layout_tracker, initial_layout,
                **kwargs,
            )
        
        # Get initial layout
        if initial_layout is not None:
            layout = initial_layout.grid.copy()
        elif layout_tracker is not None:
            layout = layout_tracker.get_layout_grid(n_rows, n_cols)
        else:
            layout = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
        
        # Decompose into patches
        patches = self._compute_patches(n_rows, n_cols)
        
        # Assign pairs to patches
        pair_assignments = self._assign_pairs_to_patches(
            layout, physical_pairs, patches
        )
        
        # Route each patch
        all_operations: List[Dict[str, Any]] = []
        total_cost = 0.0
        unsatisfied_pairs = set(range(len(physical_pairs)))
        
        for iteration in range(self.max_iterations):
            if not unsatisfied_pairs:
                break
            
            wise_logger.debug(
                f"Patch iteration {iteration}, {len(unsatisfied_pairs)} pairs remaining"
            )
            
            # Checkerboard phases
            for phase in [0, 1]:
                for patch_idx, patch in enumerate(patches):
                    if (patch["row"] + patch["col"]) % 2 != phase:
                        continue
                    
                    # Get pairs for this patch
                    patch_pairs = [
                        physical_pairs[i]
                        for i in pair_assignments.get(patch_idx, [])
                        if i in unsatisfied_pairs
                    ]
                    
                    if not patch_pairs:
                        continue
                    
                    # Extract sub-layout
                    sub_layout = self._extract_patch_layout(layout, patch)
                    
                    # Route the patch
                    result = self._route_patch(
                        sub_layout, patch_pairs, patch
                    )
                    
                    if result.success:
                        # Apply operations to main layout
                        ops = self._translate_patch_ops(
                            result.operations, patch
                        )
                        all_operations.extend(ops)
                        total_cost += result.cost
                        
                        # Update layout
                        self._apply_ops_to_layout(layout, ops)
                        
                        # Check which pairs are now satisfied
                        for i in list(unsatisfied_pairs):
                            a, b = physical_pairs[i]
                            if self._check_pair_satisfied(layout, a, b):
                                unsatisfied_pairs.discard(i)
        
        success = len(unsatisfied_pairs) == 0
        
        return RoutingResult(
            success=success,
            operations=all_operations,
            cost=total_cost,
            metrics={
                "iterations": iteration + 1,
                "unsatisfied": len(unsatisfied_pairs),
                "patches": len(patches),
            },
        )
    
    def _compute_patches(
        self,
        n_rows: int,
        n_cols: int,
    ) -> List[Dict[str, Any]]:
        """Compute patch boundaries for the grid."""
        patch_size = self.config.patch_size or 4
        patches: List[Dict[str, Any]] = []
        
        for r_start in range(0, n_rows, patch_size - self.overlap):
            for c_start in range(0, n_cols, patch_size - self.overlap):
                r_end = min(r_start + patch_size, n_rows)
                c_end = min(c_start + patch_size, n_cols)
                
                patches.append({
                    "row": r_start // (patch_size - self.overlap),
                    "col": c_start // (patch_size - self.overlap),
                    "r_start": r_start,
                    "r_end": r_end,
                    "c_start": c_start,
                    "c_end": c_end,
                })
        
        return patches
    
    def _assign_pairs_to_patches(
        self,
        layout: np.ndarray,
        pairs: List[Tuple[int, int]],
        patches: List[Dict[str, Any]],
    ) -> Dict[int, List[int]]:
        """Assign gate pairs to patches based on qubit locations."""
        # Build ion position lookup
        ion_pos: Dict[int, Tuple[int, int]] = {}
        for r in range(layout.shape[0]):
            for c in range(layout.shape[1]):
                ion_pos[int(layout[r, c])] = (r, c)
        
        assignments: Dict[int, List[int]] = defaultdict(list)
        
        for pair_idx, (a, b) in enumerate(pairs):
            pa = ion_pos.get(a, (-1, -1))
            pb = ion_pos.get(b, (-1, -1))
            
            if pa[0] < 0 or pb[0] < 0:
                continue
            
            # Find the patch that contains both ions, or the best match
            best_patch = None
            best_score = -1
            
            for patch_idx, patch in enumerate(patches):
                in_a = (patch["r_start"] <= pa[0] < patch["r_end"] and
                        patch["c_start"] <= pa[1] < patch["c_end"])
                in_b = (patch["r_start"] <= pb[0] < patch["r_end"] and
                        patch["c_start"] <= pb[1] < patch["c_end"])
                
                score = int(in_a) + int(in_b)
                if score > best_score:
                    best_score = score
                    best_patch = patch_idx
            
            if best_patch is not None:
                assignments[best_patch].append(pair_idx)
        
        return assignments
    
    def _extract_patch_layout(
        self,
        layout: np.ndarray,
        patch: Dict[str, Any],
    ) -> np.ndarray:
        """Extract the sub-layout for a patch."""
        return layout[
            patch["r_start"]:patch["r_end"],
            patch["c_start"]:patch["c_end"]
        ].copy()
    
    def _route_patch(
        self,
        sub_layout: np.ndarray,
        pairs: List[Tuple[int, int]],
        patch: Dict[str, Any],
    ) -> RoutingResult:
        """Route a single patch using the parent SAT router."""
        n_rows, n_cols = sub_layout.shape
        
        # Create grid layout
        grid_layout = GridLayout(
            grid=sub_layout,
            ion_positions={
                int(sub_layout[r, c]): (r, c)
                for r in range(n_rows)
                for c in range(n_cols)
            },
        )
        
        # Use parent's SAT routing
        return super().route_batch(
            physical_pairs=pairs,
            current_mapping=QubitMapping(),
            architecture=None,
            initial_layout=grid_layout,
        )
    
    def _translate_patch_ops(
        self,
        operations: List[Dict[str, Any]],
        patch: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Translate patch-local operations to global coordinates."""
        translated: List[Dict[str, Any]] = []
        
        for op in operations:
            new_op = op.copy()
            if "row" in new_op:
                new_op["row"] = op["row"] + patch["r_start"]
            if "col" in new_op:
                new_op["col"] = op["col"] + patch["c_start"]
            translated.append(new_op)
        
        return translated
    
    def _apply_ops_to_layout(
        self,
        layout: np.ndarray,
        operations: List[Dict[str, Any]],
    ) -> None:
        """Apply operations to update the layout in-place."""
        for op in operations:
            op_type = op.get("type", "")
            row = op.get("row", 0)
            col = op.get("col", 0)
            
            if op_type == "H_SWAP" and col + 1 < layout.shape[1]:
                layout[row, col], layout[row, col + 1] = (
                    layout[row, col + 1], layout[row, col]
                )
            elif op_type == "V_SWAP" and row + 1 < layout.shape[0]:
                layout[row, col], layout[row + 1, col] = (
                    layout[row + 1, col], layout[row, col]
                )
    
    def _check_pair_satisfied(
        self,
        layout: np.ndarray,
        ion_a: int,
        ion_b: int,
    ) -> bool:
        """Check if two ions are adjacent (can interact)."""
        pos_a = np.where(layout == ion_a)
        pos_b = np.where(layout == ion_b)
        
        if len(pos_a[0]) == 0 or len(pos_b[0]) == 0:
            return False
        
        ra, ca = int(pos_a[0][0]), int(pos_a[1][0])
        rb, cb = int(pos_b[0][0]), int(pos_b[1][0])
        
        # Same row, adjacent columns
        return ra == rb and abs(ca - cb) == 1
    
    def _check_pairs_satisfied(
        self,
        layout: np.ndarray,
        pairs: List[Tuple[int, int]],
        capacity: int,
    ) -> bool:
        """Check if all pairs can interact."""
        for a, b in pairs:
            if not self._check_pair_satisfied(layout, a, b):
                return False
        return True

    def _generate_tilings(
        self,
        patch_h: int,
        patch_w: int,
        n_rows: int,
        n_cols: int,
        capacity: int = 2,
    ) -> List[Tuple[int, int]]:
        """Generate tiling offsets for checkerboard routing.
        
        Returns list of (row_offset, col_offset) tuples representing
        different tiling phases. Matches old code's tiling strategy:
        - (0, 0): Base tiling
        - (half_h, 0): Vertical offset
        - (0, half_w): Horizontal offset (if k-compatible)
        
        Parameters
        ----------
        patch_h : int
            Patch height.
        patch_w : int
            Patch width.
        n_rows : int
            Grid rows.
        n_cols : int
            Grid columns.
        capacity : int
            Block width (*k*) for k-compatibility checking.
        
        Returns
        -------
        List[Tuple[int, int]]
            List of tiling offsets.
        """
        tilings: List[Tuple[int, int]] = [(0, 0)]
        
        # Vertical offset (half patch height)
        half_h = patch_h // 2
        if half_h > 0 and half_h < n_rows:
            tilings.append((half_h, 0))
        
        # Horizontal offset (half patch width) if even AND k-compatible
        # (Fix S: port old code's _k_compatible check).
        if patch_w % 2 == 0:
            half_w = patch_w // 2
            if (
                half_w > 0
                and half_w < n_cols
                and self._k_compatible(half_w, capacity)
            ):
                tilings.append((0, half_w))
        
        return tilings
    
    @staticmethod
    def _k_compatible(width: int, k: int) -> bool:
        """Check if a width is k-compatible for WISE block alignment.
        
        Ported from old code's _k_compatible helper.
        
        Parameters
        ----------
        width : int
            The width to check.
        k : int
            Block width for alignment.
            
        Returns
        -------
        bool
            True if width is k-compatible.
        """
        if width <= 0 or width == 1:
            return False
        if width == k:
            return True
        if width > k:
            return (width % k) == 0
        # width < k
        return (k % width) == 0
    
    def _generate_non_overlapping_patches(
        self,
        n_rows: int,
        n_cols: int,
        patch_h: int,
        patch_w: int,
        offset_r: int = 0,
        offset_c: int = 0,
    ) -> List[Tuple[int, int, int, int]]:
        """Generate non-overlapping patch regions (matching old code).
        
        Tiles the grid starting at ``(offset_r, offset_c)`` with patches
        of size ``patch_h × patch_w``, clipped to grid bounds.
        
        Parameters
        ----------
        n_rows, n_cols : int
            Grid dimensions.
        patch_h, patch_w : int
            Patch dimensions.
        offset_r, offset_c : int
            Starting offset for the tiling.
        
        Returns
        -------
        List[Tuple[int, int, int, int]]
            ``(r0, c0, r1, c1)`` bounding boxes.
        """
        if patch_h <= 0 or patch_w <= 0:
            return [(0, 0, n_rows, n_cols)]

        regions: List[Tuple[int, int, int, int]] = []
        start_row = min(max(offset_r, 0), n_rows - 1) if n_rows > 0 else 0
        start_col = min(max(offset_c, 0), n_cols - 1) if n_cols > 0 else 0

        row = start_row
        while row < n_rows:
            row_end = min(row + patch_h, n_rows)
            col = start_col
            while col < n_cols:
                col_end = min(col + patch_w, n_cols)
                regions.append((row, col, row_end, col_end))
                col = col_end  # Move to next patch (non-overlapping)
            row = row_end  # Move to next row of patches

        return regions


# =============================================================================
# Greedy Ion Router
# =============================================================================

class GreedyIonRouter(Router):
    """Fast greedy routing for small instances.
    
    Uses simple heuristics instead of SAT solving for cases where
    optimal routing is not critical.
    """
    
    def __init__(self, name: str = "greedy_ion_router"):
        super().__init__(RoutingStrategy.GREEDY, name)
    
    def route_gate(
        self,
        gate_qubits: Tuple[int, ...],
        current_mapping: QubitMapping,
        architecture: "HardwareArchitecture",
    ) -> RoutingResult:
        """Route greedily for a single gate."""
        if len(gate_qubits) != 2:
            return RoutingResult(
                success=False,
                metrics={"error": "Only two-qubit gates supported"},
            )
        
        q1, q2 = gate_qubits
        p1 = current_mapping.get_physical(q1)
        p2 = current_mapping.get_physical(q2)
        
        if p1 is None or p2 is None:
            return RoutingResult(success=False)
        
        # Simple: move p2 toward p1
        # This is a placeholder - real implementation would use architecture graph
        return RoutingResult(
            success=True,
            operations=[],  # No-op for now
            cost=0.0,
        )


__all__ = [
    "WiseSatRouter",
    "WisePatchRouter",
    "GreedyIonRouter",
]
