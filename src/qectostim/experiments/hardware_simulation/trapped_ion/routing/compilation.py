# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/compilation.py
"""
WISE Routing Pass for compilation pipeline integration.

This module provides the WISERoutingPass class that integrates
SAT-based routing into the hardware simulation compilation pipeline.
"""

from __future__ import annotations

import time
from collections import defaultdict
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

from qectostim.experiments.hardware_simulation.core.pipeline import QubitMapping

from qectostim.experiments.hardware_simulation.trapped_ion.routing.config import (
    WISE_LOGGER_NAME,
    wise_logger,
    WISERoutingConfig,
    ROW_SWAP_HEATING,
    COL_SWAP_HEATING,
    INITIAL_SPLIT_TIME_US,
    SPLIT_HEATING,
    H_PASS_TIME_US,
    V_PASS_TIME_US,
)
from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
    DEFAULT_CALIBRATION as _CAL,
)
from qectostim.experiments.hardware_simulation.trapped_ion.routing.routers import (
    WiseSatRouter,
    WisePatchRouter,
)
from qectostim.experiments.hardware_simulation.trapped_ion.routing.orchestrator import (
    WISERoutingOrchestrator,
)

if TYPE_CHECKING:
    from qectostim.experiments.hardware_simulation.core.architecture import (
        HardwareArchitecture,
    )
    from qectostim.experiments.hardware_simulation.core.pipeline import (
        MappedCircuit,
        RoutedCircuit,
    )
    from qectostim.experiments.hardware_simulation.core.compiler import RoutingResult


class WISERoutingPass:
    """Compilation pass that routes using WiseSatRouter.
    
    Integrates the WISE SAT-based router into the compilation pipeline.
    
    This pass:
    1. Analyzes the MappedCircuit to identify gate pairs per round
    2. Groups gates into batches that can be routed together
    3. Uses WiseSatRouter to compute optimal ion permutations
    4. Generates transport operations for each routing schedule
    5. Outputs a RoutedCircuit with all operations
    
    Example
    -------
    >>> config = WISERoutingConfig(timeout_seconds=30, max_passes=6)
    >>> routing_pass = WISERoutingPass(config)
    >>> routed = routing_pass.route(mapped_circuit, architecture)
    
    See Also
    --------
    WiseSatRouter : Underlying SAT-based router.
    """
    
    def __init__(
        self,
        config: Optional[WISERoutingConfig] = None,
        use_patch_routing: bool = False,
        architecture: Optional["HardwareArchitecture"] = None,
        lookahead: int = 0,
    ):
        """Initialize the routing pass.
        
        Parameters
        ----------
        config : Optional[WISERoutingConfig]
            Configuration for the SAT solver.
        use_patch_routing : bool
            If True, use patch-based routing for large grids.
        architecture : Optional[HardwareArchitecture]
            Default architecture to use if not passed to route().
        lookahead : int
            Number of future gate batches to include in each routing
            window.  When > 0 the router solves for the current batch
            plus the next ``lookahead`` batches simultaneously, then
            chains the output layout to the next window.
        """
        self.config = config or WISERoutingConfig()
        self.use_patch_routing = use_patch_routing or self.config.patch_enabled
        self.architecture = architecture
        self.lookahead = lookahead
        
        # Create the appropriate router
        self._sat_router = WiseSatRouter(config=self.config)
        if self.use_patch_routing:
            self.router = WisePatchRouter(config=self.config)
            self._orchestrator = WISERoutingOrchestrator(
                router=self._sat_router,
                config=self.config,
                lookahead=max(lookahead, self.config.lookahead_rounds),
            )
        else:
            self.router = self._sat_router
            self._orchestrator = WISERoutingOrchestrator(
                router=self._sat_router,
                config=self.config,
                lookahead=max(lookahead, self.config.lookahead_rounds),
            )
        
        # Store 1Q ops for interleaving
        self._one_qubit_ops_per_batch: List[List[Tuple[int, str]]] = []
    
    def route(
        self,
        mapped_circuit: "MappedCircuit",
        architecture: Optional["HardwareArchitecture"] = None,
    ) -> "RoutedCircuit":
        """Route a mapped circuit.
        
        Parameters
        ----------
        mapped_circuit : MappedCircuit
            Circuit with logical-to-physical mapping.
        architecture : Optional[HardwareArchitecture]
            Target WISE grid architecture.  Falls back to the
            architecture passed at construction time.
            
        Returns
        -------
        RoutedCircuit
            Circuit with routing operations inserted.
        """
        from qectostim.experiments.hardware_simulation.core.pipeline import (
            MappedCircuit,
            RoutedCircuit,
        )
        
        architecture = architecture or self.architecture
        if architecture is None:
            raise ValueError(
                "WISERoutingPass.route() requires an architecture. "
                "Pass it to __init__ or route()."
            )
        
        # Extract gate pairs from circuit
        gate_batches = self._extract_gate_batches(mapped_circuit)
        
        if not gate_batches:
            # No two-qubit gates, return empty routed circuit
            return RoutedCircuit(
                operations=[],
                final_mapping=mapped_circuit.mapping.copy(),
                routing_overhead=0,
                mapped_circuit=mapped_circuit,
            )

        # ---- Orchestrator path: patch decomposition + BT propagation ----
        if self._orchestrator is not None:
            return self._route_with_orchestrator(
                mapped_circuit, architecture, gate_batches,
            )
        
        # ---- Legacy inline loop (fallback) ----
        # Route each batch and collect operations
        all_operations: List[Any] = []
        current_mapping = mapped_circuit.mapping.copy()
        total_routing_ops = 0
        total_reconfig_time = 0.0

        # Per-ion heating accumulator
        motional_quanta: Dict[int, float] = {}
        for q in range(architecture.num_qubits):
            motional_quanta[q] = 0.0

        motional_quanta_per_batch: List[Dict[int, float]] = []

        # Mode snapshot accumulator
        from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
            ModeSnapshot,
        )
        mode_snapshots_per_batch: List[Dict[int, Any]] = []

        gate_batch_map: Dict[Tuple[int, int], int] = {}

        # Lookahead configuration
        lookahead = self.lookahead
        if lookahead == 0 and self.config and self.config.lookahead_rounds > 0:
            lookahead = self.config.lookahead_rounds

        # BT cache for future-round target layouts
        _bt_cache: List[Dict[int, Tuple[int, int]]] = []

        # Block routing cache
        _BLOCK_LEN = 4
        _block_cache: Dict[Tuple, List["RoutingResult"]] = {}
        _current_block_key: Optional[Tuple] = None
        _current_cache_for_block: Optional[List["RoutingResult"]] = None
        _blk_idx = 0
        _blk_end = 0
        _recheck_cache = True

        # Wall-clock deadline
        _per_batch_timeout = (
            self.config.timeout_seconds if self.config else 4800.0
        )
        _wall_limit = max(
            120.0,
            _per_batch_timeout * len(gate_batches) * 2,
        )
        _wall_start = time.monotonic()
        
        batch_idx = 0
        _pbar = _tqdm(
            total=len(gate_batches),
            desc="WISE routing",
            unit="batch",
            leave=True,
        )
        while batch_idx < len(gate_batches):
            # Check wall-clock deadline
            if time.monotonic() - _wall_start > _wall_limit:
                wise_logger.error(
                    "WISE routing wall-clock timeout (%.0f s) after %d/%d batches",
                    _wall_limit, batch_idx, len(gate_batches),
                )
                break
            
            _pbar.set_postfix_str(
                f"pairs={len(gate_batches[batch_idx])}, "
                f"ops={total_routing_ops}"
            )

            gate_pairs = gate_batches[batch_idx]

            # Fast-path: skip SAT for empty 2Q batches (only 1Q ops)
            if not gate_pairs:
                # No 2Q gates in this batch — nothing to route
                motional_quanta_per_batch.append(dict(motional_quanta))
                from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
                    collect_mode_snapshots as _collect_snaps_empty,
                )
                mode_snapshots_per_batch.append(
                    _collect_snaps_empty(architecture, motional_quanta)
                )
                batch_idx += 1
                _pbar.update(1)
                continue

            # Lookahead aggregation
            window_end = min(batch_idx + 1 + lookahead, len(gate_batches))
            aggregated_pairs: List[List[Tuple[int, int]]] = []
            for i in range(batch_idx, window_end):
                aggregated_pairs.append(gate_batches[i])
            
            gate_pairs = gate_batches[batch_idx]
            all_window_pairs = [p for batch in aggregated_pairs for p in batch]
            
            wise_logger.debug(
                f"Routing batch {batch_idx} with {len(gate_pairs)} pairs "
                f"(lookahead window: {len(aggregated_pairs)} batches)"
            )
            
            lookahead_batch_pairs = aggregated_pairs[1:] if len(aggregated_pairs) > 1 else None

            # Block cache lookup
            if _recheck_cache or _blk_idx == 0:
                _recheck_cache = False
                _blk_end = min(batch_idx + _BLOCK_LEN, len(gate_batches))
                key_rounds = []
                for bi in range(batch_idx, _blk_end):
                    key_rounds.append(
                        tuple(sorted(gate_batches[bi]))
                    )
                new_block_key = tuple(key_rounds)
                new_cache = _block_cache.get(new_block_key)
                if new_cache is not None and len(new_cache) > 0:
                    _current_block_key = new_block_key
                    _current_cache_for_block = new_cache
                    _blk_idx = 0
                    wise_logger.debug(
                        "Block cache HIT at batch %d", batch_idx
                    )
                elif _current_cache_for_block is None:
                    _current_block_key = new_block_key
                    _blk_idx = 0
                else:
                    _current_block_key = new_block_key

            # Use cached result if available
            if (
                _current_cache_for_block is not None
                and _blk_idx < len(_current_cache_for_block)
            ):
                if _blk_idx == 0:
                    result = self.router.route_batch(
                        gate_pairs,
                        current_mapping,
                        architecture,
                        lookahead_pairs=lookahead_batch_pairs,
                        bt_positions=_bt_cache if _bt_cache else None,
                    )
                else:
                    result = _current_cache_for_block[_blk_idx]
            else:
                result = self.router.route_batch(
                    gate_pairs,
                    current_mapping,
                    architecture,
                    lookahead_pairs=lookahead_batch_pairs,
                    bt_positions=_bt_cache if _bt_cache else None,
                )

                # Store in block cache
                if _current_cache_for_block is None and _blk_idx == 0:
                    _block_cache.setdefault(_current_block_key, [])
                if _current_block_key in _block_cache:
                    if len(_block_cache[_current_block_key]) == _blk_idx:
                        _block_cache[_current_block_key].append(result)
            
            if not result.success:
                wise_logger.warning(
                    f"Routing failed for batch {batch_idx}: {result.metrics}"
                )
                batch_idx += 1
                _pbar.update(1)
                continue
            
            # Accumulate heating
            if result.operations:
                row_swaps = sum(
                    1 for op in result.operations
                    if isinstance(op, dict) and op.get("type") == "H_SWAP"
                )
                col_swaps = sum(
                    1 for op in result.operations
                    if isinstance(op, dict) and op.get("type") == "V_SWAP"
                )
            else:
                total_swaps = result.metrics.get("total_swaps", 0) if result.metrics else 0
                row_swaps = total_swaps // 2
                col_swaps = total_swaps - row_swaps

            if row_swaps + col_swaps > 0:
                h_passes = (result.metrics or {}).get("h_passes", None)
                v_passes = (result.metrics or {}).get("v_passes", None)
                if h_passes is None or v_passes is None:
                    h_passes = 1 if row_swaps > 0 else 0
                    v_passes = 1 if col_swaps > 0 else 0
                reconfig_time = (
                    INITIAL_SPLIT_TIME_US
                    + h_passes * H_PASS_TIME_US
                    + v_passes * V_PASS_TIME_US
                )
                total_reconfig_time += reconfig_time

                # Per-ion heating
                for ion_idx in motional_quanta:
                    motional_quanta[ion_idx] += SPLIT_HEATING

            # Snapshot motional quanta
            motional_quanta_per_batch.append(dict(motional_quanta))

            # Mode snapshot
            from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
                collect_mode_snapshots as _collect_snaps,
            )
            batch_mode_snapshot = _collect_snaps(architecture, motional_quanta)
            mode_snapshots_per_batch.append(batch_mode_snapshot)

            # Add routing operations
            if result.operations:
                if all_operations:
                    from qectostim.experiments.hardware_simulation.core.operations import (
                        TransportOperation as _TO,
                    )
                    all_operations.append(_TO(
                        qubit=-1,
                        source_zone="__PASS_BOUNDARY__",
                        target_zone="__PASS_BOUNDARY__",
                        duration=0.0,
                    ))
                all_operations.extend(
                    self._convert_to_physical_ops(
                        result.operations, architecture, current_mapping
                    )
                )
                total_routing_ops += len(result.operations)
            
            # Extract BTs from routing result for future rounds
            _final_layouts = (result.metrics or {}).get("_final_layouts", [])
            _future_bts = (result.metrics or {}).get("_future_bt_positions", [])
            if _final_layouts:
                # Build BT from the final layout: pin every ion to its position
                # so future rounds start from the known state.
                _bt_from_layout: List[Dict[int, Tuple[int, int]]] = []
                for lay in _final_layouts:
                    bt_round: Dict[int, Tuple[int, int]] = {}
                    lay_arr = np.asarray(lay, dtype=int)
                    for dr in range(lay_arr.shape[0]):
                        for dc in range(lay_arr.shape[1]):
                            bt_round[int(lay_arr[dr, dc])] = (dr, dc)
                    _bt_from_layout.append(bt_round)
                _bt_cache = _bt_from_layout
            elif _future_bts:
                _bt_cache = _future_bts
            else:
                _bt_cache = []

            # Update mapping from the final layout produced by routing
            _final_layout = (result.metrics or {}).get("_final_layout", None)
            if result.final_mapping:
                current_mapping = result.final_mapping
            elif _final_layout is not None:
                # Rebuild mapping from the decoded final layout grid
                _fl = np.asarray(_final_layout, dtype=int)
                _n_k = getattr(architecture, "ions_per_segment", 1)
                _n_m = getattr(architecture, "col_groups", 1)
                _n_tc = _n_m * _n_k
                for _lq, _pq in current_mapping.logical_to_physical.items():
                    pos = np.where(_fl == _pq)
                    if len(pos[0]) > 0:
                        _r, _c = int(pos[0][0]), int(pos[1][0])
                        current_mapping.zone_assignments[_pq] = f"trap_{_r}_{_c}"

            # Emit gates for co-located pairs
            from qectostim.experiments.hardware_simulation.core.operations import (
                GateOperation,
            )
            from qectostim.experiments.hardware_simulation.core.gates import (
                GateSpec,
                GateType,
            )

            ms_spec = GateSpec(
                name="MS",
                gate_type=GateType.TWO_QUBIT,
                num_qubits=2,
                is_native=True,
            )

            _post_map = current_mapping
            _n_k = getattr(architecture, "ions_per_segment", 1)
            _n_m = getattr(architecture, "col_groups", 1)
            _n_tc = _n_m * _n_k
            _post_pos: Dict[int, Tuple[int, int]] = {}

            # Prefer positions from the decoded final layout (most accurate)
            if _final_layout is not None:
                _fl_arr = np.asarray(_final_layout, dtype=int)
                for _lq, _pq in _post_map.logical_to_physical.items():
                    _where = np.where(_fl_arr == _pq)
                    if len(_where[0]) > 0:
                        _post_pos[_pq] = (int(_where[0][0]), int(_where[1][0]))

            # Fall back to zone assignments for anything not resolved
            for _lq, _pq in _post_map.logical_to_physical.items():
                if _pq in _post_pos:
                    continue
                _zn = _post_map.zone_assignments.get(_pq)
                if _zn:
                    try:
                        _pts = _zn.split("_")
                        _pr, _pc = int(_pts[-2]), int(_pts[-1])
                        _post_pos[_pq] = (_pr, _pc)
                    except (ValueError, IndexError):
                        pass
                if _pq not in _post_pos:
                    _post_pos[_pq] = (_pq // _n_tc if _n_tc else 0,
                                      _pq % _n_tc if _n_tc else _pq)

            for logical_q1, logical_q2 in gate_pairs:
                p1 = _post_map.get_physical(logical_q1)
                p2 = _post_map.get_physical(logical_q2)
                pos1 = _post_pos.get(p1) if p1 is not None else None
                pos2 = _post_pos.get(p2) if p2 is not None else None
                if pos1 is not None and pos2 is not None:
                    same_row = (pos1[0] == pos2[0])
                    same_block = (pos1[1] // _n_k == pos2[1] // _n_k)
                    if not (same_row and same_block):
                        wise_logger.debug(
                            "Gate pair (%d,%d) NOT co-located after routing",
                            logical_q1, logical_q2,
                        )
                        continue

                gate_op = GateOperation(
                    gate=ms_spec,
                    qubits=(p1, p2),
                    duration=_CAL.ms_gate_time * 1e6,
                    base_fidelity=_CAL.gate_fidelities().get("MS", 0.99),
                    metadata={
                        "batch": batch_idx,
                        "logical_qubits": (logical_q1, logical_q2),
                    },
                )
                all_operations.append(gate_op)
                gate_batch_map[(logical_q1, logical_q2)] = batch_idx
            
            batch_idx += 1
            _blk_idx += 1
            _recheck_cache = (
                _current_cache_for_block is not None
                and _blk_idx >= len(_current_cache_for_block)
            )
            if _blk_idx >= _BLOCK_LEN or batch_idx >= _blk_end:
                _blk_idx = 0
                _blk_end = 0
                _current_cache_for_block = None
                _recheck_cache = True
            _pbar.update(1)
        
        _pbar.close()

        routing_metadata: Dict[str, Any] = {
            "motional_quanta": motional_quanta,
            "motional_quanta_per_batch": motional_quanta_per_batch,
            "mode_snapshots_per_batch": mode_snapshots_per_batch,
            "gate_batch_map": gate_batch_map,
            "num_batches": len(gate_batches),
            "reconfiguration_time_us": total_reconfig_time,
            "total_routing_swaps": total_routing_ops,
        }

        return RoutedCircuit(
            operations=all_operations,
            final_mapping=current_mapping,
            routing_overhead=total_routing_ops,
            mapped_circuit=mapped_circuit,
            metadata=routing_metadata,
        )

    # ------------------------------------------------------------------
    # Orchestrator-based routing
    # ------------------------------------------------------------------

    def _route_with_orchestrator(
        self,
        mapped_circuit: "MappedCircuit",
        architecture: "HardwareArchitecture",
        gate_batches: List[List[Tuple[int, int]]],
    ) -> "RoutedCircuit":
        """Route all batches through the WISERoutingOrchestrator.

        The orchestrator handles patch decomposition, BT propagation,
        block caching, adaptive growth, and cross-boundary preferences.
        This method converts the orchestrator output (reconfigs) into
        a ``RoutedCircuit`` with physical operations.
        """
        from qectostim.experiments.hardware_simulation.core.pipeline import (
            RoutedCircuit,
        )

        # --- Build initial grid layout ---
        n_rows = getattr(architecture, "rows", 1)
        _k = getattr(architecture, "ions_per_segment", 1)
        _m = getattr(architecture, "col_groups", 1)
        n_cols = _m * _k
        capacity = _k

        layout = np.arange(n_rows * n_cols, dtype=int).reshape(n_rows, n_cols)
        current_mapping = mapped_circuit.mapping.copy()

        # Place ions according to mapping's zone assignments
        placed: set = set()
        for _log, _phys in current_mapping.logical_to_physical.items():
            zone = current_mapping.zone_assignments.get(_phys)
            if zone:
                try:
                    parts = zone.split("_")
                    zr, zc = int(parts[-2]), int(parts[-1])
                    if 0 <= zr < n_rows and 0 <= zc < n_cols:
                        layout[zr, zc] = _phys
                        placed.add((zr, zc))
                except (ValueError, IndexError):
                    pass

        # --- Call orchestrator ---
        all_reconfigs, total_time = self._orchestrator.route_all_rounds(
            initial_layout=layout,
            parallel_pairs=gate_batches,
            n_rows=n_rows,
            n_cols=n_cols,
            capacity=capacity,
            ignore_initial_reconfig=True,
        )

        # --- Convert reconfigs to physical operations ---
        all_operations: List[Any] = []
        total_routing_ops = 0

        # Per-ion heating accumulator
        motional_quanta: Dict[int, float] = {}
        for q in range(architecture.num_qubits):
            motional_quanta[q] = 0.0
        motional_quanta_per_batch: List[Dict[int, float]] = []

        from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
            collect_mode_snapshots as _collect_snaps,
        )
        mode_snapshots_per_batch: List[Dict[int, Any]] = []
        gate_batch_map: Dict[Tuple[int, int], int] = {}
        total_reconfig_time = total_time

        from qectostim.experiments.hardware_simulation.core.operations import (
            GateOperation,
            TransportOperation as _TO,
        )
        from qectostim.experiments.hardware_simulation.core.gates import (
            GateSpec,
            GateType,
        )
        ms_spec = GateSpec(
            name="MS",
            gate_type=GateType.TWO_QUBIT,
            num_qubits=2,
            is_native=True,
        )

        for reconfig_idx, (layout_after, schedule, solved_pairs) in enumerate(all_reconfigs):
            # Convert schedule ops to physical transport
            if schedule:
                if all_operations:
                    all_operations.append(_TO(
                        qubit=-1,
                        source_zone="__PASS_BOUNDARY__",
                        target_zone="__PASS_BOUNDARY__",
                        duration=0.0,
                    ))
                phys_ops = self._convert_to_physical_ops(
                    schedule, architecture, current_mapping,
                )
                all_operations.extend(phys_ops)
                total_routing_ops += len(schedule)

                # GAP 8 FIX: use run_reconfig_from_schedule for proper
                # per-ion heating (port of OLD _runOddEvenReconfig).
                from qectostim.experiments.hardware_simulation.trapped_ion.routing.layout_utils import (
                    run_reconfig_from_schedule as _run_reconfig,
                )
                try:
                    _prev_layout = layout.copy()
                    _reconfig_heating, _reconfig_time_s = _run_reconfig(
                        old_layout=_prev_layout,
                        new_layout=np.asarray(layout_after, dtype=int),
                        sat_schedule=schedule if schedule else None,
                        k=capacity,
                    )
                    # Accumulate per-ion heating
                    for ion_idx, delta_h in _reconfig_heating.items():
                        motional_quanta[ion_idx] = motional_quanta.get(ion_idx, 0.0) + delta_h
                    total_reconfig_time += _reconfig_time_s * 1e6  # convert s→µs
                except Exception as _reconfig_err:
                    # Fallback: basic estimate if schedule replay fails
                    _logger.debug(
                        "run_reconfig_from_schedule failed: %s; using basic estimate",
                        _reconfig_err,
                    )
                    row_swaps = sum(
                        1 for op in schedule
                        if isinstance(op, dict) and op.get("type") == "H_SWAP"
                    )
                    col_swaps = sum(
                        1 for op in schedule
                        if isinstance(op, dict) and op.get("type") == "V_SWAP"
                    )
                    if row_swaps + col_swaps > 0:
                        h_passes = 1 if row_swaps > 0 else 0
                        v_passes = 1 if col_swaps > 0 else 0
                        total_reconfig_time += (
                            INITIAL_SPLIT_TIME_US
                            + h_passes * H_PASS_TIME_US
                            + v_passes * V_PASS_TIME_US
                        )
                        for ion_idx in motional_quanta:
                            motional_quanta[ion_idx] += SPLIT_HEATING

            motional_quanta_per_batch.append(dict(motional_quanta))
            mode_snapshots_per_batch.append(
                _collect_snaps(architecture, motional_quanta),
            )

            # Emit MS gate ops for solved pairs
            _fl = np.asarray(layout_after, dtype=int)
            _post_pos: Dict[int, Tuple[int, int]] = {}
            for _lq, _pq in current_mapping.logical_to_physical.items():
                _where = np.where(_fl == _pq)
                if len(_where[0]) > 0:
                    _post_pos[_pq] = (int(_where[0][0]), int(_where[1][0]))

            for logical_q1, logical_q2 in solved_pairs:
                p1 = current_mapping.get_physical(logical_q1)
                p2 = current_mapping.get_physical(logical_q2)
                if p1 is not None and p2 is not None:
                    gate_op = GateOperation(
                        gate=ms_spec,
                        qubits=(p1, p2),
                        duration=_CAL.ms_gate_time * 1e6,
                        base_fidelity=_CAL.gate_fidelities().get("MS", 0.99),
                        metadata={
                            "batch": reconfig_idx,
                            "logical_qubits": (logical_q1, logical_q2),
                        },
                    )
                    all_operations.append(gate_op)
                    gate_batch_map[(logical_q1, logical_q2)] = reconfig_idx

            # Update mapping from final layout
            for _lq, _pq in current_mapping.logical_to_physical.items():
                pos = _post_pos.get(_pq)
                if pos is not None:
                    current_mapping.zone_assignments[_pq] = f"trap_{pos[0]}_{pos[1]}"

            # Track current layout for next iteration's heating calc
            layout = np.asarray(layout_after, dtype=int)

        routing_metadata: Dict[str, Any] = {
            "motional_quanta": motional_quanta,
            "motional_quanta_per_batch": motional_quanta_per_batch,
            "mode_snapshots_per_batch": mode_snapshots_per_batch,
            "gate_batch_map": gate_batch_map,
            "num_batches": len(gate_batches),
            "reconfiguration_time_us": total_reconfig_time,
            "total_routing_swaps": total_routing_ops,
        }

        return RoutedCircuit(
            operations=all_operations,
            final_mapping=current_mapping,
            routing_overhead=total_routing_ops,
            mapped_circuit=mapped_circuit,
            metadata=routing_metadata,
        )
    
    def _extract_gate_batches(
        self,
        mapped_circuit: "MappedCircuit",
    ) -> List[List[Tuple[int, int]]]:
        """Extract two-qubit gate pairs grouped into batches."""
        batches: List[List[Tuple[int, int]]] = []
        self._one_qubit_ops_per_batch = []

        remaining_ops = list(mapped_circuit.native_circuit.operations)
        while remaining_ops:
            # Phase A: collect 1Q ops
            one_q_ops: List[Tuple[int, str]] = []
            used_1q: Set[int] = set()
            leftover: List = []
            for op in remaining_ops:
                if len(op.qubits) == 1:
                    q = op.qubits[0]
                    if q not in used_1q:
                        one_q_ops.append((q, getattr(op, "name", "R")))
                        used_1q.add(q)
                    else:
                        leftover.append(op)
                else:
                    leftover.append(op)
            remaining_ops = leftover

            # Phase B: collect 2Q ops
            current_batch: List[Tuple[int, int]] = []
            used_2q: Set[int] = set()
            leftover2: List = []
            for op in remaining_ops:
                if len(op.qubits) == 2:
                    q1, q2 = op.qubits
                    if q1 not in used_2q and q2 not in used_2q:
                        current_batch.append((q1, q2))
                        used_2q.add(q1)
                        used_2q.add(q2)
                    else:
                        leftover2.append(op)
                else:
                    leftover2.append(op)
            remaining_ops = leftover2

            if current_batch:
                batches.append(current_batch)
                self._one_qubit_ops_per_batch.append(one_q_ops)
            elif one_q_ops:
                batches.append([])
                self._one_qubit_ops_per_batch.append(one_q_ops)
            else:
                break

        return batches
    
    def _convert_to_physical_ops(
        self,
        routing_ops: List[Any],
        architecture: "HardwareArchitecture",
        mapping: Optional["QubitMapping"] = None,
    ) -> List[Any]:
        """Convert routing schedule operations to physical operations."""
        from qectostim.experiments.hardware_simulation.core.operations import (
            TransportOperation,
        )

        _rows = getattr(architecture, "rows", 1)
        _k = getattr(architecture, "ions_per_segment", 1)
        _m = getattr(architecture, "col_groups", 1)
        _total_cols = _m * _k

        _grid = np.arange(_rows * _total_cols, dtype=int).reshape(
            _rows, _total_cols
        )
        if mapping is not None:
            placed: set = set()
            _n_cols = _total_cols
            for _log, _phys in mapping.logical_to_physical.items():
                zone = mapping.zone_assignments.get(_phys)
                if zone:
                    try:
                        parts = zone.split("_")
                        zr, zc = int(parts[-2]), int(parts[-1])
                        if 0 <= zr < _rows and 0 <= zc < _n_cols:
                            _grid[zr, zc] = _phys
                            placed.add((zr, zc))
                    except (ValueError, IndexError):
                        pass
            if len(placed) < len(mapping.logical_to_physical):
                for _log, _phys in mapping.logical_to_physical.items():
                    pr = _phys // _n_cols if _n_cols > 0 else 0
                    pc = _phys % _n_cols if _n_cols > 0 else _phys
                    if (0 <= pr < _rows and 0 <= pc < _n_cols
                            and (pr, pc) not in placed):
                        _grid[pr, pc] = _phys
                        placed.add((pr, pc))

        physical_ops: List[Any] = []

        for op in routing_ops:
            if isinstance(op, dict):
                op_type = op.get("type", "")

                if op_type == "PASS_BOUNDARY":
                    physical_ops.append(TransportOperation(
                        qubit=-1,
                        source_zone="__PASS_BOUNDARY__",
                        target_zone="__PASS_BOUNDARY__",
                        duration=0.0,
                    ))
                    continue

                if op_type in ("H_SWAP", "V_SWAP", "transport", "swap", "ROUTING"):
                    r = op.get("row", 0)
                    c = op.get("col", 0)
                    _has_legacy_keys = "source" in op or "target" in op

                    if op_type == "H_SWAP" and not _has_legacy_keys:
                        c2 = c + 1
                        if c2 >= _grid.shape[1]:
                            continue
                        q_a = int(_grid[r, c])
                        q_b = int(_grid[r, c2])
                        src_a = f"trap_{r}_{c}"
                        tgt_a = f"trap_{r}_{c2}"
                        src_b = f"trap_{r}_{c2}"
                        tgt_b = f"trap_{r}_{c}"
                        _grid[r, c], _grid[r, c2] = _grid[r, c2], _grid[r, c]
                    elif op_type == "V_SWAP" and not _has_legacy_keys:
                        r2 = r + 1
                        if r2 >= _grid.shape[0]:
                            continue
                        q_a = int(_grid[r, c])
                        q_b = int(_grid[r2, c])
                        src_a = f"trap_{r}_{c}"
                        tgt_a = f"trap_{r2}_{c}"
                        src_b = f"trap_{r2}_{c}"
                        tgt_b = f"trap_{r}_{c}"
                        _grid[r, c], _grid[r2, c] = _grid[r2, c], _grid[r, c]
                    else:
                        qubits = op.get("qubits", ())
                        qubit = qubits[0] if qubits else int(_grid[r, c])
                        src_zone = op.get("source", f"trap_{r}_{c}")
                        tgt_zone = op.get("target", f"trap_{r}_{c}")
                        physical_ops.append(TransportOperation(
                            qubit=qubit,
                            source_zone=str(src_zone),
                            target_zone=str(tgt_zone),
                            duration=op.get("distance", 1.0) * _CAL.shuttle_time * 1e6,
                        ))
                        continue

                    _swap_meta = {
                        "swap_type": op_type,
                        "swap_row": r,
                        "swap_col": c,
                    }
                    physical_ops.append(TransportOperation(
                        qubit=q_a,
                        source_zone=src_a,
                        target_zone=tgt_a,
                        duration=_CAL.shuttle_time * 1e6,
                        metadata=_swap_meta,
                    ))
                    physical_ops.append(TransportOperation(
                        qubit=q_b,
                        source_zone=src_b,
                        target_zone=tgt_b,
                        duration=_CAL.shuttle_time * 1e6,
                        metadata=_swap_meta,
                    ))
            else:
                physical_ops.append(op)

        return physical_ops


__all__ = [
    "WISERoutingPass",
]
